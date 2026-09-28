//! # Standard trait implementations
//!
//! Herein are the hand-written implementations of [`Debug`], [`Clone`],
//! [`PartialEq`], [`Eq`], [`Hash`], and [`Drop`] for [`Expression`] and
//! [`DiceExpression`], the two types through which every recursive field of the
//! AST passes. Derived implementations would recurse once per level of nesting,
//! overflowing the stack on deep input; these use explicit stacks instead, so
//! their stack depth is constant. The other AST types derive their
//! implementations, which recurse into these after a single level.
//!
//! [`Debug`], [`PartialEq`], and [`Hash`] delegate to the shared [traversal].
//! [`Clone`] builds its copy bottom-up with an explicit stack. [`Drop`]
//! detaches children onto an explicit stack before they can be dropped
//! recursively, and contains this crate's only `unsafe` code; see
//! [`promote_leaf`].

#![deny(clippy::undocumented_unsafe_blocks)]

use std::{
	fmt::{self, Debug, Formatter},
	hash::{Hash, Hasher},
	mem::{self, ManuallyDrop},
	ptr
};

use super::{
	Add, ArithmeticExpression, Binding, Constant, CustomDice, DiceExpression,
	Div, DropHighest, DropLowest, Exp, Expression, Group, Mod, Mul, Neg, Range,
	StandardDice, Sub, Variable,
	traversal::{self, Node}
};
use crate::span::SourceSpan;

////////////////////////////////////////////////////////////////////////////////
//                      Debug, PartialEq, Eq, and Hash.                       //
////////////////////////////////////////////////////////////////////////////////

impl Debug for Expression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		traversal::debug(Node::Expression(self), f)
	}
}

impl Debug for DiceExpression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		traversal::debug(Node::Dice(self), f)
	}
}

impl PartialEq for Expression<'_>
{
	fn eq(&self, other: &Self) -> bool
	{
		traversal::eq(Node::Expression(self), Node::Expression(other))
	}
}

impl PartialEq for DiceExpression<'_>
{
	fn eq(&self, other: &Self) -> bool
	{
		traversal::eq(Node::Dice(self), Node::Dice(other))
	}
}

impl Eq for Expression<'_> {}

impl Eq for DiceExpression<'_> {}

impl Hash for Expression<'_>
{
	fn hash<H: Hasher>(&self, state: &mut H)
	{
		traversal::hash(Node::Expression(self), state)
	}
}

impl Hash for DiceExpression<'_>
{
	fn hash<H: Hasher>(&self, state: &mut H)
	{
		traversal::hash(Node::Dice(self), state)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Clone.                                   //
////////////////////////////////////////////////////////////////////////////////

impl Clone for Expression<'_>
{
	fn clone(&self) -> Self
	{
		match self
		{
			// Leaves are common, and need no stacks.
			Expression::Constant(constant) => Expression::Constant(*constant),
			Expression::Variable(variable) =>
			{
				Expression::Variable(variable.clone())
			},
			_ =>
			{
				let mut cloner = Cloner::new(Task::Expression(self), false);
				cloner.run();
				cloner.pop_expression()
			}
		}
	}
}

impl Clone for DiceExpression<'_>
{
	fn clone(&self) -> Self
	{
		let mut cloner = Cloner::new(Task::Dice(self), false);
		cloner.run();
		cloner.pop_dice()
	}
}

/// Copy an expression with every span reset to the default, as
/// [`untethered`](crate::span::Spanned::untethered) requires, without
/// recursion.
///
/// # Parameters
/// - `expression`: The expression.
///
/// # Returns
/// The untethered copy.
pub(super) fn untether<'src>(expression: &Expression<'src>)
-> Expression<'src>
{
	let mut cloner = Cloner::new(Task::Expression(expression), true);
	cloner.run();
	cloner.pop_expression()
}

/// Copy a dice expression with every span reset to the default, as
/// [`untethered`](crate::span::Spanned::untethered) requires, without
/// recursion.
///
/// # Parameters
/// - `dice`: The dice expression.
///
/// # Returns
/// The untethered copy.
pub(super) fn untether_dice<'src>(
	dice: &DiceExpression<'src>
) -> DiceExpression<'src>
{
	let mut cloner = Cloner::new(Task::Dice(dice), true);
	cloner.run();
	cloner.pop_dice()
}

/// A unit of work for a [`Cloner`].
enum Task<'a, 'src>
{
	/// Clone an expression: schedule its build, and the cloning of its children
	/// before that.
	Expression(&'a Expression<'src>),

	/// Clone a dice expression: schedule its build, and the cloning of its
	/// children before that.
	Dice(&'a DiceExpression<'src>),

	/// Build the copy of an expression from the copies of its children, atop
	/// the value stacks.
	BuildExpression(&'a Expression<'src>),

	/// Build the copy of a dice expression from the copies of its children,
	/// atop the value stacks.
	BuildDice(&'a DiceExpression<'src>)
}

/// A post-order cloner for ASTs. Each node's children are cloned before the
/// node itself, leaving their copies atop the value stacks, in order, whence
/// the node's build pops them. The cloner either copies spans faithfully, for
/// [`Clone`], or resets them, for
/// [`untethered`](crate::span::Spanned::untethered).
struct Cloner<'a, 'src>
{
	/// Whether to reset every span to the default, rather than copy it.
	untether: bool,

	/// The pending tasks, the next of which is on top.
	tasks: Vec<Task<'a, 'src>>,

	/// The copies of expressions that await their parents.
	expressions: Vec<Expression<'src>>,

	/// The copies of dice expressions that await their parents.
	dice: Vec<DiceExpression<'src>>
}

impl<'a, 'src> Cloner<'a, 'src>
{
	/// Construct a cloner.
	///
	/// # Parameters
	/// - `root`: The task that clones the root.
	/// - `untether`: Whether to reset every span to the default, rather than
	///   copy it.
	///
	/// # Returns
	/// The cloner.
	fn new(root: Task<'a, 'src>, untether: bool) -> Self
	{
		Cloner {
			untether,
			tasks: vec![root],
			expressions: Vec::new(),
			dice: Vec::new()
		}
	}

	/// Answer the span of a copy.
	///
	/// # Parameters
	/// - `span`: The span of the original.
	///
	/// # Returns
	/// The span of the copy: the default if untethering, else `span`.
	fn span(&self, span: SourceSpan) -> SourceSpan
	{
		if self.untether
		{
			SourceSpan::default()
		}
		else
		{
			span
		}
	}

	/// Run every task. Afterward, the copy of the root is the only value on
	/// the value stacks.
	fn run(&mut self)
	{
		while let Some(task) = self.tasks.pop()
		{
			match task
			{
				Task::Expression(expression) =>
				{
					self.visit_expression(expression)
				},
				Task::Dice(dice) => self.visit_dice(dice),
				Task::BuildExpression(expression) =>
				{
					let copy = self.build_expression(expression);
					self.expressions.push(copy);
				},
				Task::BuildDice(dice) =>
				{
					let copy = self.build_dice(dice);
					self.dice.push(copy);
				}
			}
		}
	}

	/// Schedule the cloning of an expression. Leaves are cloned at once.
	///
	/// # Parameters
	/// - `expression`: The expression.
	fn visit_expression(&mut self, expression: &'a Expression<'src>)
	{
		self.tasks.push(Task::BuildExpression(expression));
		// Schedule the children in reverse, so that they are cloned in order.
		match expression
		{
			Expression::Constant(constant) =>
			{
				self.tasks.pop();
				self.expressions.push(Expression::Constant(Constant {
					span: self.span(constant.span),
					..*constant
				}));
			},
			Expression::Variable(variable) =>
			{
				self.tasks.pop();
				self.expressions.push(Expression::Variable(Variable {
					name: variable.name.clone(),
					span: self.span(variable.span)
				}));
			},
			Expression::Group(group) =>
			{
				self.tasks.push(Task::Expression(&group.expression))
			},
			Expression::Binding(binding) =>
			{
				self.tasks.push(Task::Expression(&binding.expression))
			},
			Expression::Range(range) =>
			{
				self.tasks.push(Task::Expression(&range.end));
				self.tasks.push(Task::Expression(&range.start));
			},
			Expression::Dice(dice) => self.tasks.push(Task::Dice(dice)),
			Expression::Arithmetic(arithmetic) => match arithmetic
			{
				ArithmeticExpression::Add(Add { left, right, .. })
				| ArithmeticExpression::Sub(Sub { left, right, .. })
				| ArithmeticExpression::Mul(Mul { left, right, .. })
				| ArithmeticExpression::Div(Div { left, right, .. })
				| ArithmeticExpression::Mod(Mod { left, right, .. })
				| ArithmeticExpression::Exp(Exp { left, right, .. }) =>
				{
					self.tasks.push(Task::Expression(right));
					self.tasks.push(Task::Expression(left));
				},
				ArithmeticExpression::Neg(neg) =>
				{
					self.tasks.push(Task::Expression(&neg.operand))
				},
			}
		}
	}

	/// Schedule the cloning of a dice expression.
	///
	/// # Parameters
	/// - `dice`: The dice expression.
	fn visit_dice(&mut self, dice: &'a DiceExpression<'src>)
	{
		self.tasks.push(Task::BuildDice(dice));
		// Schedule the children in reverse, so that they are cloned in order.
		match dice
		{
			DiceExpression::Standard(standard) =>
			{
				self.tasks.push(Task::Expression(&standard.faces));
				self.tasks.push(Task::Expression(&standard.count));
			},
			DiceExpression::Custom(custom) =>
			{
				self.tasks.push(Task::Expression(&custom.count))
			},
			DiceExpression::DropLowest(DropLowest { dice, drop, .. })
			| DiceExpression::DropHighest(DropHighest { dice, drop, .. }) =>
			{
				if let Some(drop) = drop
				{
					self.tasks.push(Task::Expression(drop));
				}
				self.tasks.push(Task::Dice(dice));
			}
		}
	}

	/// Build the copy of an expression from the copies of its children.
	///
	/// # Parameters
	/// - `expression`: The original expression.
	///
	/// # Returns
	/// The copy.
	fn build_expression(
		&mut self,
		expression: &'a Expression<'src>
	) -> Expression<'src>
	{
		match expression
		{
			Expression::Constant(_) | Expression::Variable(_) =>
			{
				unreachable!("leaves are cloned without a build")
			},
			Expression::Group(group) => Expression::Group(Group {
				expression: Box::new(self.pop_expression()),
				span: self.span(group.span)
			}),
			Expression::Binding(binding) => Expression::Binding(Binding {
				name: binding.name.clone(),
				name_span: self.span(binding.name_span),
				expression: Box::new(self.pop_expression()),
				span: self.span(binding.span)
			}),
			Expression::Range(range) =>
			{
				let (start, end) = self.pop_expression_pair();
				Expression::Range(Range {
					start,
					end,
					span: self.span(range.span)
				})
			},
			Expression::Dice(_) => Expression::Dice(self.pop_dice()),
			Expression::Arithmetic(arithmetic) =>
			{
				Expression::Arithmetic(self.build_arithmetic(arithmetic))
			},
		}
	}

	/// Build the copy of an arithmetic expression from the copies of its
	/// children.
	///
	/// # Parameters
	/// - `arithmetic`: The original arithmetic expression.
	///
	/// # Returns
	/// The copy.
	fn build_arithmetic(
		&mut self,
		arithmetic: &'a ArithmeticExpression<'src>
	) -> ArithmeticExpression<'src>
	{
		if let ArithmeticExpression::Neg(neg) = arithmetic
		{
			return ArithmeticExpression::Neg(Neg {
				operand: Box::new(self.pop_expression()),
				span: self.span(neg.span)
			})
		}
		let (left, right) = self.pop_expression_pair();
		match arithmetic
		{
			ArithmeticExpression::Add(add) => ArithmeticExpression::Add(Add {
				left,
				right,
				span: self.span(add.span)
			}),
			ArithmeticExpression::Sub(sub) => ArithmeticExpression::Sub(Sub {
				left,
				right,
				span: self.span(sub.span)
			}),
			ArithmeticExpression::Mul(mul) => ArithmeticExpression::Mul(Mul {
				left,
				right,
				span: self.span(mul.span)
			}),
			ArithmeticExpression::Div(div) => ArithmeticExpression::Div(Div {
				left,
				right,
				span: self.span(div.span)
			}),
			ArithmeticExpression::Mod(r#mod) =>
			{
				ArithmeticExpression::Mod(Mod {
					left,
					right,
					span: self.span(r#mod.span)
				})
			},
			ArithmeticExpression::Exp(exp) => ArithmeticExpression::Exp(Exp {
				left,
				right,
				span: self.span(exp.span)
			}),
			ArithmeticExpression::Neg(_) => unreachable!()
		}
	}

	/// Build the copy of a dice expression from the copies of its children.
	///
	/// # Parameters
	/// - `dice`: The original dice expression.
	///
	/// # Returns
	/// The copy.
	fn build_dice(
		&mut self,
		dice: &'a DiceExpression<'src>
	) -> DiceExpression<'src>
	{
		match dice
		{
			DiceExpression::Standard(standard) =>
			{
				let (count, faces) = self.pop_expression_pair();
				DiceExpression::Standard(StandardDice {
					count,
					faces,
					span: self.span(standard.span)
				})
			},
			DiceExpression::Custom(custom) =>
			{
				DiceExpression::Custom(CustomDice {
					count: Box::new(self.pop_expression()),
					faces: custom.faces.clone(),
					span: self.span(custom.span)
				})
			},
			DiceExpression::DropLowest(clause) =>
			{
				let drop = clause
					.drop
					.as_ref()
					.map(|_| Box::new(self.pop_expression()));
				DiceExpression::DropLowest(DropLowest {
					dice: Box::new(self.pop_dice()),
					drop,
					span: self.span(clause.span)
				})
			},
			DiceExpression::DropHighest(clause) =>
			{
				let drop = clause
					.drop
					.as_ref()
					.map(|_| Box::new(self.pop_expression()));
				DiceExpression::DropHighest(DropHighest {
					dice: Box::new(self.pop_dice()),
					drop,
					span: self.span(clause.span)
				})
			}
		}
	}

	/// Pop the copy of an expression.
	///
	/// # Returns
	/// The copy.
	///
	/// # Panics
	/// If there is no copy, which would be a bug in the cloner.
	fn pop_expression(&mut self) -> Expression<'src>
	{
		self.expressions
			.pop()
			.expect("the child must be cloned before its parent")
	}

	/// Pop the copies of two sibling expressions.
	///
	/// # Returns
	/// The copies, in their original order.
	///
	/// # Panics
	/// If there are fewer than two copies, which would be a bug in the cloner.
	fn pop_expression_pair(
		&mut self
	) -> (Box<Expression<'src>>, Box<Expression<'src>>)
	{
		let second = self.pop_expression();
		let first = self.pop_expression();
		(Box::new(first), Box::new(second))
	}

	/// Pop the copy of a dice expression.
	///
	/// # Returns
	/// The copy.
	///
	/// # Panics
	/// If there is no copy, which would be a bug in the cloner.
	fn pop_dice(&mut self) -> DiceExpression<'src>
	{
		self.dice
			.pop()
			.expect("the child must be cloned before its parent")
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Drop.                                    //
////////////////////////////////////////////////////////////////////////////////

impl Drop for Expression<'_>
{
	fn drop(&mut self)
	{
		let mut orphans = Vec::new();
		detach_children(self, &mut orphans);
		drop_orphans(orphans);
	}
}

impl Drop for DiceExpression<'_>
{
	fn drop(&mut self)
	{
		let mut orphans = Vec::new();
		detach_dice_children(self, &mut orphans);
		drop_orphans(orphans);
	}
}

/// The expression that takes the place of a detached child. It has no
/// children, so dropping it does nothing.
const PLACEHOLDER: Expression<'static> = Expression::Constant(Constant {
	value: 0,
	span: SourceSpan::SYNTHETIC
});

/// Drop detached expressions, detaching their own children first. Each
/// expression is therefore childless by the time that it drops, so its own
/// [`Drop`] finds nothing to do, and nothing recurses.
///
/// # Parameters
/// - `orphans`: The detached expressions.
fn drop_orphans(mut orphans: Vec<Expression<'_>>)
{
	while let Some(mut orphan) = orphans.pop()
	{
		detach_children(&mut orphan, &mut orphans);
	}
}

/// Detach an expression from its parent, unless it is a leaf, replacing it
/// with the [placeholder](PLACEHOLDER).
///
/// # Parameters
/// - `slot`: The expression.
/// - `orphans`: The detached expressions, to which to add this one.
fn detach<'src>(
	slot: &mut Expression<'src>,
	orphans: &mut Vec<Expression<'src>>
)
{
	if !matches!(slot, Expression::Constant(_) | Expression::Variable(_))
	{
		orphans.push(mem::replace(slot, PLACEHOLDER));
	}
}

/// Detach the children of an expression, leaving it shallow: dropping it no
/// longer drops any expression with children.
///
/// # Parameters
/// - `expression`: The expression.
/// - `orphans`: The detached expressions, to which to add its children.
fn detach_children<'src>(
	expression: &mut Expression<'src>,
	orphans: &mut Vec<Expression<'src>>
)
{
	match expression
	{
		Expression::Constant(_) | Expression::Variable(_) =>
		{},
		Expression::Group(group) => detach(&mut group.expression, orphans),
		Expression::Binding(binding) =>
		{
			detach(&mut binding.expression, orphans)
		},
		Expression::Range(range) =>
		{
			detach(&mut range.start, orphans);
			detach(&mut range.end, orphans);
		},
		Expression::Dice(dice) => detach_dice_children(dice, orphans),
		Expression::Arithmetic(arithmetic) => match arithmetic
		{
			ArithmeticExpression::Add(Add { left, right, .. })
			| ArithmeticExpression::Sub(Sub { left, right, .. })
			| ArithmeticExpression::Mul(Mul { left, right, .. })
			| ArithmeticExpression::Div(Div { left, right, .. })
			| ArithmeticExpression::Mod(Mod { left, right, .. })
			| ArithmeticExpression::Exp(Exp { left, right, .. }) =>
			{
				detach(left, orphans);
				detach(right, orphans);
			},
			ArithmeticExpression::Neg(neg) => detach(&mut neg.operand, orphans)
		}
	}
}

/// Detach the children of a dice expression, leaving it shallow: dropping it
/// no longer drops any expression with children, or any stack of drop clauses.
///
/// # Parameters
/// - `dice`: The dice expression.
/// - `orphans`: The detached expressions, to which to add its children.
fn detach_dice_children<'src>(
	dice: &mut DiceExpression<'src>,
	orphans: &mut Vec<Expression<'src>>
)
{
	match dice
	{
		DiceExpression::Standard(standard) =>
		{
			detach(&mut standard.count, orphans);
			detach(&mut standard.faces, orphans);
		},
		DiceExpression::Custom(custom) => detach(&mut custom.count, orphans),
		DiceExpression::DropLowest(DropLowest { dice, drop, .. })
		| DiceExpression::DropHighest(DropHighest { dice, drop, .. }) =>
		{
			if let Some(drop) = drop.take()
			{
				orphans.push(*drop);
			}
			match **dice
			{
				// The dice beneath are a leaf, so detaching their children
				// recurses no further.
				DiceExpression::Standard(_) | DiceExpression::Custom(_) =>
				{
					detach_dice_children(dice, orphans)
				},
				DiceExpression::DropLowest(_)
				| DiceExpression::DropHighest(_) => promote_leaf(dice, orphans)
			}
		}
	}
}

/// Replace a stack of drop clauses with the leaf dice expression at its bottom,
/// dropping the drop clauses between without recursion and without allocation.
/// The drop expressions of the clauses, and the children of the leaf, join the
/// orphans.
///
/// ```mermaid
/// flowchart LR
///     subgraph Before
///         direction TB
///         P1["parent"] --> C1["clause"] --> C2["clause"] --> L1["leaf"]
///     end
///     subgraph After
///         direction TB
///         P2["parent"] --> L2["leaf"]
///     end
///     Before --> After
/// ```
///
/// # Parameters
/// - `slot`: The top of the stack of drop clauses, which the leaf replaces.
/// - `orphans`: The detached expressions, to which to add the drop expressions
///   and the children of the leaf.
///
/// # Safety
/// Safe code cannot take a child out of a drop clause without supplying a
/// replacement, which costs an allocation, since every [`DiceExpression`]
/// owns a [`Box`]; nor destructure one, since [`DiceExpression`] implements
/// [`Drop`]. So this function works in two phases:
///
/// 1. The safe phase walks down the stack, detaching every drop expression and
///    the children of the leaf, so that nothing left in the stack has a
///    [`Drop`] that does any work.
/// 2. The unsafe phase moves the top clause out of `slot` with [`ptr::read`],
///    leaving `slot` logically uninitialized. It then dismantles each clause in
///    turn: moving the clause out of its [`Box`] frees the box, and reading the
///    payload out of a [`ManuallyDrop`] transfers ownership of the fields
///    without running the clause's [`Drop`]. Finally, it moves the leaf into
///    `slot` with [`ptr::write`].
///
/// Every value is thus owned exactly once throughout, and the only window in
/// which `slot` is uninitialized contains nothing that can unwind: moves,
/// matches, deallocations, and drops of `None`. So no panic can expose the
/// uninitialized `slot` to a second drop. The test suite checks this code
/// under Miri; see `tests::ast`.
#[cfg_attr(doc, aquamarine::aquamarine)]
fn promote_leaf<'src>(
	slot: &mut DiceExpression<'src>,
	orphans: &mut Vec<Expression<'src>>
)
{
	// Phase 1: Detach every drop expression, and the children of the leaf.
	let mut cursor = &mut *slot;
	loop
	{
		match cursor
		{
			DiceExpression::DropLowest(DropLowest { dice, drop, .. })
			| DiceExpression::DropHighest(DropHighest { dice, drop, .. }) =>
			{
				if let Some(drop) = drop.take()
				{
					orphans.push(*drop);
				}
				cursor = &mut **dice;
			},
			leaf
			@ (DiceExpression::Standard(_) | DiceExpression::Custom(_)) =>
			{
				detach_dice_children(leaf, orphans);
				break
			}
		}
	}
	// Phase 2: Replace the stack with its leaf. Nothing here may unwind.
	// SAFETY: `slot` is valid for reads. Ownership of its value moves to
	// `clause`, and `slot` is not used again until the `ptr::write` below
	// reinitializes it, so the value is never dropped twice.
	let mut clause = unsafe { ptr::read(slot) };
	let leaf = loop
	{
		let undropped = ManuallyDrop::new(clause);
		let dice = match &*undropped
		{
			DiceExpression::DropLowest(payload) =>
			{
				// SAFETY: `undropped` is never dropped, so the payload has
				// exactly one owner after the read. Phase 1 emptied `drop`,
				// so discarding it does nothing.
				let DropLowest {
					dice,
					drop: _,
					span: _
				} = unsafe { ptr::read(payload) };
				dice
			},
			DiceExpression::DropHighest(payload) =>
			{
				// SAFETY: As above.
				let DropHighest {
					dice,
					drop: _,
					span: _
				} = unsafe { ptr::read(payload) };
				dice
			},
			DiceExpression::Standard(_) | DiceExpression::Custom(_) =>
			{
				break ManuallyDrop::into_inner(undropped)
			},
		};
		// Move the next clause out of its box, which frees the box.
		clause = *dice;
	};
	// SAFETY: `slot` is valid for writes, and its previous value moved out
	// above, so overwriting it leaks nothing.
	unsafe { ptr::write(slot, leaf) };
}
