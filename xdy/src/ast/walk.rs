//! # Iterative walks
//!
//! Herein is the engine that drives [`ASTVisitor`] walks. A [`Walk`] is a
//! stream of [events](Event) that [enter](Event::Enter) and
//! [leave](Event::Leave) every node of an AST, left to right, in source order.
//! It uses an explicit stack in place of recursion, so its stack depth is
//! constant however deep the tree. [`fold`] drives a visitor from the stream,
//! keeping the outputs of finished children on a second explicit stack until
//! their parent is visited. Analyses that need no outputs from children, such
//! as gathering names, loop over a [`Walk`] directly.

use super::{
	ASTVisitor, Add, ArithmeticExpression, DiceExpression, Div, Exp,
	Expression, Mod, Mul, Predicate, Sub
};

////////////////////////////////////////////////////////////////////////////////
//                                   Walks.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A node that a [`Walk`] enters and leaves.
///
/// Every node of an AST, other than the root, fills a slot of type
/// [`Expression`] or [`DiceExpression`], and appears here as the corresponding
/// variant. An [`ArithmeticExpression`] never fills a slot of its own, so it
/// appears only as the root of a walk.
#[derive(Copy, Clone)]
pub(crate) enum Node<'a, 'src>
{
	/// An arbitrary expression.
	Expression(&'a Expression<'src>),

	/// A dice expression.
	Dice(&'a DiceExpression<'src>),

	/// An arithmetic expression.
	Arithmetic(&'a ArithmeticExpression<'src>)
}

/// An event in a [`Walk`].
#[derive(Copy, Clone)]
pub(crate) enum Event<'a, 'src>
{
	/// Enter a node, before any of its children.
	Enter(Node<'a, 'src>),

	/// Leave a node, after all of its children.
	Leave(Node<'a, 'src>)
}

/// The stream of [events](Event) for a walk of an AST. The stack holds the
/// pending events of every open node, so its length grows with the depth of
/// the tree, but the machine stack does not.
pub(crate) struct Walk<'a, 'src>
{
	/// The pending events, the next of which is on top. A pending
	/// [`Enter`](Event::Enter) also stands for the events of the node's
	/// descendants, which the walk schedules when it emits the
	/// [`Enter`](Event::Enter).
	stack: Vec<Event<'a, 'src>>
}

impl<'a, 'src> Walk<'a, 'src>
{
	/// Begin a walk.
	///
	/// # Parameters
	/// - `root`: The node at which to start.
	///
	/// # Returns
	/// The stream of events.
	pub(crate) fn new(root: Node<'a, 'src>) -> Self
	{
		Walk {
			stack: vec![Event::Enter(root)]
		}
	}

	/// Schedule the entry of an expression.
	///
	/// # Parameters
	/// - `expression`: The expression.
	fn expression(&mut self, expression: &'a Expression<'src>)
	{
		self.stack.push(Event::Enter(Node::Expression(expression)));
	}

	/// Schedule the entry of the children of a node, in reverse, so that the
	/// walk enters them in order.
	///
	/// # Parameters
	/// - `node`: The node.
	fn schedule_children(&mut self, node: Node<'a, 'src>)
	{
		match node
		{
			Node::Expression(expression) => match expression
			{
				Expression::Group(group) => self.expression(&group.expression),
				Expression::Constant(_) | Expression::Variable(_) =>
				{},
				Expression::Binding(binding) =>
				{
					self.expression(&binding.expression)
				},
				Expression::Range(range) =>
				{
					self.expression(&range.end);
					self.expression(&range.start);
				},
				Expression::Extremum(extremum) =>
				{
					for argument in extremum.arguments.iter().rev()
					{
						self.expression(argument);
					}
				},
				Expression::Dice(dice) => self.schedule_dice_children(dice),
				Expression::Arithmetic(arithmetic) =>
				{
					self.schedule_arithmetic_children(arithmetic)
				},
			},
			Node::Dice(dice) => self.schedule_dice_children(dice),
			Node::Arithmetic(arithmetic) =>
			{
				self.schedule_arithmetic_children(arithmetic)
			},
		}
	}

	/// Schedule the entry of the children of a dice expression, in reverse.
	///
	/// # Parameters
	/// - `dice`: The dice expression.
	fn schedule_dice_children(&mut self, dice: &'a DiceExpression<'src>)
	{
		let (dice, count) = match dice
		{
			DiceExpression::Standard(standard) =>
			{
				self.expression(&standard.faces);
				self.expression(&standard.count);
				return
			},
			DiceExpression::Custom(custom) =>
			{
				self.expression(&custom.count);
				return
			},
			DiceExpression::DropLowest(clause) => (&clause.dice, &clause.drop),
			DiceExpression::DropHighest(clause) => (&clause.dice, &clause.drop),
			DiceExpression::KeepLowest(clause) => (&clause.dice, &clause.keep),
			DiceExpression::KeepHighest(clause) => (&clause.dice, &clause.keep),
			DiceExpression::Reroll(clause) =>
			{
				if let Some(count) =
					clause.limit.as_ref().and_then(|limit| limit.operand())
				{
					self.expression(count);
				}
				// A predicate has at most two operands.
				let mut operands = clause.predicate.operands();
				let (first, second) = (operands.next(), operands.next());
				for operand in [second, first].into_iter().flatten()
				{
					self.expression(operand);
				}
				self.stack.push(Event::Enter(Node::Dice(&clause.dice)));
				return
			}
		};
		if let Some(count) = count
		{
			self.expression(count);
		}
		self.stack.push(Event::Enter(Node::Dice(dice)));
	}

	/// Schedule the entry of the children of an arithmetic expression, in
	/// reverse.
	///
	/// # Parameters
	/// - `arithmetic`: The arithmetic expression.
	fn schedule_arithmetic_children(
		&mut self,
		arithmetic: &'a ArithmeticExpression<'src>
	)
	{
		match arithmetic
		{
			ArithmeticExpression::Add(Add { left, right, .. })
			| ArithmeticExpression::Sub(Sub { left, right, .. })
			| ArithmeticExpression::Mul(Mul { left, right, .. })
			| ArithmeticExpression::Div(Div { left, right, .. })
			| ArithmeticExpression::Mod(Mod { left, right, .. })
			| ArithmeticExpression::Exp(Exp { left, right, .. }) =>
			{
				self.expression(right);
				self.expression(left);
			},
			ArithmeticExpression::Neg(neg) => self.expression(&neg.operand)
		}
	}
}

impl<'a, 'src> Iterator for Walk<'a, 'src>
{
	type Item = Event<'a, 'src>;

	fn next(&mut self) -> Option<Self::Item>
	{
		let event = self.stack.pop()?;
		if let Event::Enter(node) = event
		{
			self.stack.push(Event::Leave(node));
			self.schedule_children(node);
		}
		Some(event)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Folding.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Walk an AST with an [`ASTVisitor`], as the [trait](ASTVisitor) describes.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrowed AST.
/// - `'src`: The lifetime of the source text within the AST.
/// - `V`: The type of the visitor.
///
/// # Parameters
/// - `root`: The node at which to start.
/// - `visitor`: The visitor.
///
/// # Returns
/// The output of the root.
///
/// # Errors
/// Propagates the first error returned by the visitor.
pub(super) fn fold<'a, 'src, V: ASTVisitor<'a, 'src>>(
	root: Node<'a, 'src>,
	visitor: &mut V
) -> Result<V::Output, V::Error>
{
	// The outputs of the finished children of every open node, in order.
	let mut outputs = Vec::new();
	for event in Walk::new(root)
	{
		match event
		{
			Event::Enter(node) => enter(node, visitor)?,
			Event::Leave(node) =>
			{
				let output = leave(node, visitor, &mut outputs)?;
				outputs.push(output);
			}
		}
	}
	let output = pop(&mut outputs);
	debug_assert!(outputs.is_empty(), "a walk must consume every output");
	Ok(output)
}

/// Call the `enter_*` hook of the node inside a [`Node`].
///
/// # Parameters
/// - `node`: The node.
/// - `visitor`: The visitor.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn enter<'a, 'src, V: ASTVisitor<'a, 'src>>(
	node: Node<'a, 'src>,
	visitor: &mut V
) -> Result<(), V::Error>
{
	match node
	{
		Node::Expression(expression) => match expression
		{
			Expression::Group(group) => visitor.enter_group(group),
			Expression::Constant(constant) => visitor.enter_constant(constant),
			Expression::Variable(variable) => visitor.enter_variable(variable),
			Expression::Binding(binding) => visitor.enter_binding(binding),
			Expression::Range(range) => visitor.enter_range(range),
			Expression::Extremum(extremum) => visitor.enter_extremum(extremum),
			Expression::Dice(dice) => enter_dice(dice, visitor),
			Expression::Arithmetic(arithmetic) =>
			{
				enter_arithmetic(arithmetic, visitor)
			},
		},
		Node::Dice(dice) => enter_dice(dice, visitor),
		Node::Arithmetic(arithmetic) => enter_arithmetic(arithmetic, visitor)
	}
}

/// Call the `enter_*` hook of the node inside a [`DiceExpression`].
///
/// # Parameters
/// - `dice`: The dice expression.
/// - `visitor`: The visitor.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn enter_dice<'a, 'src, V: ASTVisitor<'a, 'src>>(
	dice: &'a DiceExpression<'src>,
	visitor: &mut V
) -> Result<(), V::Error>
{
	match dice
	{
		DiceExpression::Standard(standard) =>
		{
			visitor.enter_standard_dice(standard)
		},
		DiceExpression::Custom(custom) => visitor.enter_custom_dice(custom),
		DiceExpression::DropLowest(clause) => visitor.enter_drop_lowest(clause),
		DiceExpression::DropHighest(clause) =>
		{
			visitor.enter_drop_highest(clause)
		},
		DiceExpression::KeepLowest(clause) => visitor.enter_keep_lowest(clause),
		DiceExpression::KeepHighest(clause) =>
		{
			visitor.enter_keep_highest(clause)
		},
		DiceExpression::Reroll(clause) => visitor.enter_reroll(clause)
	}
}

/// Call the `enter_*` hook of the node inside an [`ArithmeticExpression`].
///
/// # Parameters
/// - `arithmetic`: The arithmetic expression.
/// - `visitor`: The visitor.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn enter_arithmetic<'a, 'src, V: ASTVisitor<'a, 'src>>(
	arithmetic: &'a ArithmeticExpression<'src>,
	visitor: &mut V
) -> Result<(), V::Error>
{
	match arithmetic
	{
		ArithmeticExpression::Add(add) => visitor.enter_add(add),
		ArithmeticExpression::Sub(sub) => visitor.enter_sub(sub),
		ArithmeticExpression::Mul(mul) => visitor.enter_mul(mul),
		ArithmeticExpression::Div(div) => visitor.enter_div(div),
		ArithmeticExpression::Mod(modulo) => visitor.enter_mod(modulo),
		ArithmeticExpression::Exp(exp) => visitor.enter_exp(exp),
		ArithmeticExpression::Neg(neg) => visitor.enter_neg(neg)
	}
}

/// Call the `visit_*` method of the node inside a [`Node`], with the outputs
/// of its children, and then, if the node is an [`Expression`],
/// [`visit_expression`](ASTVisitor::visit_expression).
///
/// # Parameters
/// - `node`: The node.
/// - `visitor`: The visitor.
/// - `outputs`: The outputs of finished children, the node's own atop.
///
/// # Returns
/// The output of the node.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn leave<'a, 'src, V: ASTVisitor<'a, 'src>>(
	node: Node<'a, 'src>,
	visitor: &mut V,
	outputs: &mut Vec<V::Output>
) -> Result<V::Output, V::Error>
{
	match node
	{
		Node::Expression(expression) =>
		{
			let output = match expression
			{
				Expression::Group(group) =>
				{
					let inner = pop(outputs);
					visitor.visit_group(group, inner)
				},
				Expression::Constant(constant) =>
				{
					visitor.visit_constant(constant)
				},
				Expression::Variable(variable) =>
				{
					visitor.visit_variable(variable)
				},
				Expression::Binding(binding) =>
				{
					let inner = pop(outputs);
					visitor.visit_binding(binding, inner)
				},
				Expression::Range(range) =>
				{
					let (start, end) = pop_pair(outputs);
					visitor.visit_range(range, start, end)
				},
				Expression::Extremum(extremum) =>
				{
					let arguments = pop_many(outputs, extremum.arguments.len());
					visitor.visit_extremum(extremum, arguments)
				},
				Expression::Dice(dice) => leave_dice(dice, visitor, outputs),
				Expression::Arithmetic(arithmetic) =>
				{
					leave_arithmetic(arithmetic, visitor, outputs)
				},
			}?;
			visitor.visit_expression(expression, output)
		},
		Node::Dice(dice) => leave_dice(dice, visitor, outputs),
		Node::Arithmetic(arithmetic) =>
		{
			leave_arithmetic(arithmetic, visitor, outputs)
		},
	}
}

/// Call the `visit_*` method of the node inside a [`DiceExpression`], with the
/// outputs of its children.
///
/// # Parameters
/// - `dice`: The dice expression.
/// - `visitor`: The visitor.
/// - `outputs`: The outputs of finished children, the node's own atop.
///
/// # Returns
/// The output of the node.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn leave_dice<'a, 'src, V: ASTVisitor<'a, 'src>>(
	dice: &'a DiceExpression<'src>,
	visitor: &mut V,
	outputs: &mut Vec<V::Output>
) -> Result<V::Output, V::Error>
{
	match dice
	{
		DiceExpression::Standard(standard) =>
		{
			let (count, faces) = pop_pair(outputs);
			visitor.visit_standard_dice(standard, count, faces)
		},
		DiceExpression::Custom(custom) =>
		{
			let count = pop(outputs);
			visitor.visit_custom_dice(custom, count)
		},
		DiceExpression::DropLowest(clause) =>
		{
			let drop = clause.drop.as_ref().map(|_| pop(outputs));
			let dice = pop(outputs);
			visitor.visit_drop_lowest(clause, dice, drop)
		},
		DiceExpression::DropHighest(clause) =>
		{
			let drop = clause.drop.as_ref().map(|_| pop(outputs));
			let dice = pop(outputs);
			visitor.visit_drop_highest(clause, dice, drop)
		},
		DiceExpression::KeepLowest(clause) =>
		{
			let keep = clause.keep.as_ref().map(|_| pop(outputs));
			let dice = pop(outputs);
			visitor.visit_keep_lowest(clause, dice, keep)
		},
		DiceExpression::KeepHighest(clause) =>
		{
			let keep = clause.keep.as_ref().map(|_| pop(outputs));
			let dice = pop(outputs);
			visitor.visit_keep_highest(clause, dice, keep)
		},
		DiceExpression::Reroll(clause) =>
		{
			let limit = clause
				.limit
				.as_ref()
				.map(|limit| limit.map(|_| pop(outputs)));
			let predicate = pop_predicate(&clause.predicate, outputs);
			let dice = pop(outputs);
			visitor.visit_reroll(clause, dice, predicate, limit)
		}
	}
}

/// Call the `visit_*` method of the node inside an [`ArithmeticExpression`],
/// with the outputs of its children.
///
/// # Parameters
/// - `arithmetic`: The arithmetic expression.
/// - `visitor`: The visitor.
/// - `outputs`: The outputs of finished children, the node's own atop.
///
/// # Returns
/// The output of the node.
///
/// # Errors
/// Propagates any error returned by the visitor.
fn leave_arithmetic<'a, 'src, V: ASTVisitor<'a, 'src>>(
	arithmetic: &'a ArithmeticExpression<'src>,
	visitor: &mut V,
	outputs: &mut Vec<V::Output>
) -> Result<V::Output, V::Error>
{
	if let ArithmeticExpression::Neg(neg) = arithmetic
	{
		let operand = pop(outputs);
		return visitor.visit_neg(neg, operand)
	}
	let (left, right) = pop_pair(outputs);
	match arithmetic
	{
		ArithmeticExpression::Add(add) => visitor.visit_add(add, left, right),
		ArithmeticExpression::Sub(sub) => visitor.visit_sub(sub, left, right),
		ArithmeticExpression::Mul(mul) => visitor.visit_mul(mul, left, right),
		ArithmeticExpression::Div(div) => visitor.visit_div(div, left, right),
		ArithmeticExpression::Mod(modulo) =>
		{
			visitor.visit_mod(modulo, left, right)
		},
		ArithmeticExpression::Exp(exp) => visitor.visit_exp(exp, left, right),
		ArithmeticExpression::Neg(_) => unreachable!()
	}
}

/// Pop the output of a finished child.
///
/// # Parameters
/// - `outputs`: The outputs of finished children.
///
/// # Returns
/// The output atop the stack.
///
/// # Panics
/// If the stack is empty, which would mean that the walk left a node without
/// finishing all of its children.
fn pop<T>(outputs: &mut Vec<T>) -> T
{
	outputs
		.pop()
		.expect("a child's output must precede its parent's visit")
}

/// Pop the outputs of the specified number of finished children.
///
/// # Parameters
/// - `outputs`: The outputs of finished children.
/// - `count`: The number of outputs to pop.
///
/// # Returns
/// The outputs, in order.
///
/// # Panics
/// If the stack holds fewer than `count` outputs; see [`pop`].
fn pop_many<T>(outputs: &mut Vec<T>, count: usize) -> Vec<T>
{
	let first = outputs
		.len()
		.checked_sub(count)
		.expect("a child's output must precede its parent's visit");
	outputs.split_off(first)
}

/// Pop the outputs of the operands of a predicate.
///
/// # Parameters
/// - `predicate`: The predicate.
/// - `outputs`: The outputs of finished children, those of the operands atop,
///   in order.
///
/// # Returns
/// A predicate of the same form, holding the outputs of its operands.
///
/// # Panics
/// If the stack holds fewer outputs than the predicate has operands; see
/// [`pop`].
fn pop_predicate<T, U>(
	predicate: &Predicate<T>,
	outputs: &mut Vec<U>
) -> Predicate<U>
{
	let count = predicate.operands().count();
	let first = outputs
		.len()
		.checked_sub(count)
		.expect("a child's output must precede its parent's visit");
	let mut operands = outputs.drain(first..);
	predicate.map(|_| {
		operands
			.next()
			.expect("a predicate has as many outputs as operands")
	})
}

/// Pop the outputs of two finished children.
///
/// # Parameters
/// - `outputs`: The outputs of finished children.
///
/// # Returns
/// The outputs of the first and second children, in order.
///
/// # Panics
/// If the stack holds fewer than two outputs; see [`pop`].
fn pop_pair<T>(outputs: &mut Vec<T>) -> (T, T)
{
	let second = pop(outputs);
	let first = pop(outputs);
	(first, second)
}
