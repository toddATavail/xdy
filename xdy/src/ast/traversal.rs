//! # Explicit-stack traversal
//!
//! Herein is the traversal that underlies the hand-written [`Debug`],
//! [`PartialEq`], and [`Hash`] implementations of [`Expression`] and
//! [`DiceExpression`]. It expands a node into a stream of [events](Event) that
//! mirrors the shape of the derived [`Debug`] format — structs, tuples, field
//! names, and leaves — using an explicit stack in place of recursion, so its
//! stack depth is constant however deep the tree.
//!
//! The stream serves all three traits at once:
//!
//! - [`debug`] renders the stream in the derived format, both compact (`{:?}`)
//!   and pretty (`{:#?}`).
//! - [`eq`] compares two streams in lockstep. The stream encodes the tree
//!   faithfully — every variant, field, and leaf — so two trees are equal if
//!   and only if their streams are.
//! - [`hash`] hashes the stream, which is therefore consistent with [`eq`].
//!
//! The recursive fields of every AST type are either [`Expression`]s or
//! [`DiceExpression`]s, so these two types are the only [nodes](Node); the
//! struct and enum types between them are expanded inline. The other AST types
//! derive their traits, which bottom out here after a single level.

use std::{
	fmt::{self, Debug, Formatter, Write},
	hash::{Hash, Hasher}
};

use super::{
	Add, ArithmeticExpression, Constant, DiceExpression, Div, Exp, Expression,
	Mod, Mul, Sub, Variable
};
use crate::span::SourceSpan;

////////////////////////////////////////////////////////////////////////////////
//                                  Events.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A node from which a traversal starts, and at which it expands the tree.
#[derive(Copy, Clone)]
pub(super) enum Node<'a, 'src>
{
	/// An arbitrary expression.
	Expression(&'a Expression<'src>),

	/// A dice expression.
	Dice(&'a DiceExpression<'src>)
}

/// An event in the traversal of an AST. The events mirror the calls that the
/// derived [`Debug`] implementations make on a [`Formatter`].
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(super) enum Event<'a, 'src>
{
	/// Open a struct, e.g., `Group { … }`. Its fields follow, each introduced
	/// by a [`Field`](Self::Field), and a [`Close`](Self::Close) ends it.
	Struct(&'static str),

	/// Open a tuple, e.g., `Group(…)` or `Some(…)`. Its fields follow, and a
	/// [`Close`](Self::Close) ends it.
	Tuple(&'static str),

	/// Name the next field of the innermost struct.
	Field(&'static str),

	/// A leaf, i.e., a value that is not an [`Expression`] or a
	/// [`DiceExpression`].
	Leaf(Leaf<'a, 'src>),

	/// A unit value, e.g., `None`.
	Unit(&'static str),

	/// Close the innermost struct or tuple.
	Close
}

/// A leaf of an AST.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(super) enum Leaf<'a, 'src>
{
	/// A source span.
	Span(SourceSpan),

	/// The name of a binding.
	Name(&'a str),

	/// A constant.
	Constant(Constant),

	/// A variable reference.
	Variable(&'a Variable<'src>),

	/// The faces of a custom die.
	Faces(&'a [i32])
}

impl Leaf<'_, '_>
{
	/// Answer the leaf as a [`Debug`] trait object.
	///
	/// # Returns
	/// The leaf's value.
	fn as_debug(&self) -> &dyn Debug
	{
		match self
		{
			Leaf::Span(span) => span,
			Leaf::Name(name) => name,
			Leaf::Constant(constant) => constant,
			Leaf::Variable(variable) => variable,
			Leaf::Faces(faces) => faces
		}
	}
}

/// An entry on the stack of [`Events`]: either an event ready to emit, or a
/// node awaiting expansion.
#[derive(Copy, Clone)]
enum Item<'a, 'src>
{
	/// An event ready to emit.
	Event(Event<'a, 'src>),

	/// A node awaiting expansion.
	Node(Node<'a, 'src>)
}

/// The stream of [events](Event) for a pre-order traversal of an AST. The stack
/// holds the pending work of every open node, so its length grows with the
/// depth of the tree, but the machine stack does not.
pub(super) struct Events<'a, 'src>
{
	/// The pending items, the next of which is on top.
	stack: Vec<Item<'a, 'src>>
}

impl<'a, 'src> Events<'a, 'src>
{
	/// Begin a traversal.
	///
	/// # Parameters
	/// - `root`: The node at which to start.
	///
	/// # Returns
	/// The stream of events.
	pub(super) fn new(root: Node<'a, 'src>) -> Self
	{
		Events {
			stack: vec![Item::Node(root)]
		}
	}

	/// Schedule the specified items, the first of which will be next.
	///
	/// # Parameters
	/// - `items`: The items, in order of emission.
	fn schedule(&mut self, items: &[Item<'a, 'src>])
	{
		self.stack.extend(items.iter().rev());
	}

	/// Expand a node into its events, scheduling its children for expansion
	/// in turn.
	///
	/// # Parameters
	/// - `node`: The node.
	fn expand(&mut self, node: Node<'a, 'src>)
	{
		match node
		{
			Node::Expression(expression) => self.expand_expression(expression),
			Node::Dice(dice) => self.expand_dice(dice)
		}
	}

	/// Expand an [`Expression`].
	///
	/// # Parameters
	/// - `expression`: The expression.
	fn expand_expression(&mut self, expression: &'a Expression<'src>)
	{
		match expression
		{
			Expression::Group(group) => self.schedule(&[
				tuple("Group"),
				structure("Group"),
				field("expression"),
				expression_node(&group.expression),
				field("span"),
				span(group.span),
				close(),
				close()
			]),
			Expression::Constant(constant) => self.schedule(&[
				tuple("Constant"),
				leaf(Leaf::Constant(*constant)),
				close()
			]),
			Expression::Variable(variable) => self.schedule(&[
				tuple("Variable"),
				leaf(Leaf::Variable(variable)),
				close()
			]),
			Expression::Binding(binding) => self.schedule(&[
				tuple("Binding"),
				structure("Binding"),
				field("name"),
				leaf(Leaf::Name(&binding.name)),
				field("name_span"),
				span(binding.name_span),
				field("expression"),
				expression_node(&binding.expression),
				field("span"),
				span(binding.span),
				close(),
				close()
			]),
			Expression::Range(range) => self.schedule(&[
				tuple("Range"),
				structure("Range"),
				field("start"),
				expression_node(&range.start),
				field("end"),
				expression_node(&range.end),
				field("span"),
				span(range.span),
				close(),
				close()
			]),
			Expression::Dice(dice) =>
			{
				self.schedule(&[tuple("Dice"), dice_node(dice), close()])
			},
			Expression::Arithmetic(arithmetic) =>
			{
				self.expand_arithmetic(arithmetic)
			},
		}
	}

	/// Expand an [`Expression::Arithmetic`].
	///
	/// # Parameters
	/// - `arithmetic`: The arithmetic expression.
	fn expand_arithmetic(&mut self, arithmetic: &'a ArithmeticExpression<'src>)
	{
		match arithmetic
		{
			ArithmeticExpression::Add(Add { left, right, span })
			| ArithmeticExpression::Sub(Sub { left, right, span })
			| ArithmeticExpression::Mul(Mul { left, right, span })
			| ArithmeticExpression::Div(Div { left, right, span })
			| ArithmeticExpression::Mod(Mod { left, right, span })
			| ArithmeticExpression::Exp(Exp { left, right, span }) =>
			{
				let name = match arithmetic
				{
					ArithmeticExpression::Add(_) => "Add",
					ArithmeticExpression::Sub(_) => "Sub",
					ArithmeticExpression::Mul(_) => "Mul",
					ArithmeticExpression::Div(_) => "Div",
					ArithmeticExpression::Mod(_) => "Mod",
					ArithmeticExpression::Exp(_) => "Exp",
					ArithmeticExpression::Neg(_) => unreachable!()
				};
				self.schedule(&[
					tuple("Arithmetic"),
					tuple(name),
					structure(name),
					field("left"),
					expression_node(left),
					field("right"),
					expression_node(right),
					field("span"),
					self::span(*span),
					close(),
					close(),
					close()
				])
			},
			ArithmeticExpression::Neg(neg) => self.schedule(&[
				tuple("Arithmetic"),
				tuple("Neg"),
				structure("Neg"),
				field("operand"),
				expression_node(&neg.operand),
				field("span"),
				span(neg.span),
				close(),
				close(),
				close()
			])
		}
	}

	/// Expand a [`DiceExpression`].
	///
	/// # Parameters
	/// - `dice`: The dice expression.
	fn expand_dice(&mut self, dice: &'a DiceExpression<'src>)
	{
		let (variant, name, dice, drop, span) = match dice
		{
			DiceExpression::Standard(standard) =>
			{
				return self.schedule(&[
					tuple("Standard"),
					structure("StandardDice"),
					field("count"),
					expression_node(&standard.count),
					field("faces"),
					expression_node(&standard.faces),
					field("span"),
					span(standard.span),
					close(),
					close()
				])
			},
			DiceExpression::Custom(custom) =>
			{
				return self.schedule(&[
					tuple("Custom"),
					structure("CustomDice"),
					field("count"),
					expression_node(&custom.count),
					field("faces"),
					leaf(Leaf::Faces(&custom.faces)),
					field("span"),
					span(custom.span),
					close(),
					close()
				])
			},
			DiceExpression::DropLowest(clause) => (
				"DropLowest",
				"DropLowest",
				&clause.dice,
				&clause.drop,
				clause.span
			),
			DiceExpression::DropHighest(clause) => (
				"DropHighest",
				"DropHighest",
				&clause.dice,
				&clause.drop,
				clause.span
			)
		};
		// Schedule the parts in reverse, since each goes atop its successor.
		self.schedule(&[field("span"), self::span(span), close(), close()]);
		match drop
		{
			None => self.schedule(&[Item::Event(Event::Unit("None"))]),
			Some(drop) =>
			{
				self.schedule(&[tuple("Some"), expression_node(drop), close()])
			},
		}
		self.schedule(&[
			tuple(variant),
			structure(name),
			field("dice"),
			dice_node(dice),
			field("drop")
		]);
	}
}

impl<'a, 'src> Iterator for Events<'a, 'src>
{
	type Item = Event<'a, 'src>;

	fn next(&mut self) -> Option<Self::Item>
	{
		loop
		{
			match self.stack.pop()?
			{
				Item::Event(event) => return Some(event),
				Item::Node(node) => self.expand(node)
			}
		}
	}
}

/// Answer an item that opens a struct.
///
/// # Parameters
/// - `name`: The name of the struct.
///
/// # Returns
/// The item.
fn structure<'a, 'src>(name: &'static str) -> Item<'a, 'src>
{
	Item::Event(Event::Struct(name))
}

/// Answer an item that opens a tuple.
///
/// # Parameters
/// - `name`: The name of the tuple.
///
/// # Returns
/// The item.
fn tuple<'a, 'src>(name: &'static str) -> Item<'a, 'src>
{
	Item::Event(Event::Tuple(name))
}

/// Answer an item that names a field.
///
/// # Parameters
/// - `name`: The name of the field.
///
/// # Returns
/// The item.
fn field<'a, 'src>(name: &'static str) -> Item<'a, 'src>
{
	Item::Event(Event::Field(name))
}

/// Answer an item that closes a struct or tuple.
///
/// # Returns
/// The item.
fn close<'a, 'src>() -> Item<'a, 'src> { Item::Event(Event::Close) }

/// Answer an item for a leaf.
///
/// # Parameters
/// - `leaf`: The leaf.
///
/// # Returns
/// The item.
fn leaf<'a, 'src>(leaf: Leaf<'a, 'src>) -> Item<'a, 'src>
{
	Item::Event(Event::Leaf(leaf))
}

/// Answer an item for a source span.
///
/// # Parameters
/// - `span`: The span.
///
/// # Returns
/// The item.
fn span<'a, 'src>(span: SourceSpan) -> Item<'a, 'src> { leaf(Leaf::Span(span)) }

/// Answer an item for an expression awaiting expansion.
///
/// # Parameters
/// - `expression`: The expression.
///
/// # Returns
/// The item.
fn expression_node<'a, 'src>(expression: &'a Expression<'src>)
-> Item<'a, 'src>
{
	Item::Node(Node::Expression(expression))
}

/// Answer an item for a dice expression awaiting expansion.
///
/// # Parameters
/// - `dice`: The dice expression.
///
/// # Returns
/// The item.
fn dice_node<'a, 'src>(dice: &'a DiceExpression<'src>) -> Item<'a, 'src>
{
	Item::Node(Node::Dice(dice))
}

////////////////////////////////////////////////////////////////////////////////
//                              Debug rendering.                              //
////////////////////////////////////////////////////////////////////////////////

/// Render an AST in the derived [`Debug`] format, compact or pretty according
/// to the formatter's [alternate](Formatter::alternate) flag.
///
/// # Parameters
/// - `root`: The root of the AST.
/// - `f`: The formatter.
///
/// # Returns
/// The result of writing to the formatter.
///
/// # Errors
/// Propagates any error from the formatter.
///
/// # Notes
/// In compact mode, leaves are rendered directly with `f`, so every formatter
/// flag reaches them, as in the derived format. In pretty mode, leaves are
/// rendered with `{:#?}` through an [`Indenter`], so other flags (e.g., `x` in
/// `{:#x?}`) do not reach them. The derived format keeps them, by nesting a
/// formatter per level; stable Rust offers no way to do so iteratively.
pub(super) fn debug(root: Node<'_, '_>, f: &mut Formatter<'_>) -> fmt::Result
{
	let pretty = f.alternate();
	let mut open = Vec::<Container>::new();
	for event in Events::new(root)
	{
		match event
		{
			Event::Struct(name) | Event::Tuple(name) =>
			{
				begin_value(f, &mut open, pretty)?;
				f.write_str(name)?;
				open.push(Container {
					is_struct: matches!(event, Event::Struct(_)),
					has_fields: false
				});
			},
			Event::Unit(name) =>
			{
				begin_value(f, &mut open, pretty)?;
				f.write_str(name)?;
				end_value(f, &open, pretty)?;
			},
			Event::Leaf(leaf) =>
			{
				begin_value(f, &mut open, pretty)?;
				if pretty
				{
					let mut indenter = Indenter {
						f,
						levels: open.len(),
						on_newline: false
					};
					write!(indenter, "{:#?}", leaf.as_debug())?;
				}
				else
				{
					leaf.as_debug().fmt(f)?;
				}
				end_value(f, &open, pretty)?;
			},
			Event::Field(name) =>
			{
				let levels = open.len();
				let container = open
					.last_mut()
					.expect("a field must belong to an open struct");
				if pretty
				{
					if !container.has_fields
					{
						f.write_str(" {\n")?;
					}
					indent(f, levels)?;
				}
				else
				{
					f.write_str(
						if container.has_fields { ", " } else { " { " }
					)?;
				}
				container.has_fields = true;
				f.write_str(name)?;
				f.write_str(": ")?;
			},
			Event::Close =>
			{
				let container = open
					.pop()
					.expect("a close must match an open struct or tuple");
				if container.has_fields
				{
					if pretty
					{
						indent(f, open.len())?;
					}
					f.write_str(match (container.is_struct, pretty)
					{
						(true, true) => "}",
						(true, false) => " }",
						(false, _) => ")"
					})?;
				}
				end_value(f, &open, pretty)?;
			}
		}
	}
	Ok(())
}

/// A struct or tuple that [`debug`] has opened but not yet closed.
struct Container
{
	/// Whether the container is a struct, rather than a tuple.
	is_struct: bool,

	/// Whether the container has rendered any fields yet.
	has_fields: bool
}

/// Begin rendering a value. A value inside a tuple is a new field of that
/// tuple, so it needs a separator; a value inside a struct follows the
/// separator that its [`Field`](Event::Field) already rendered.
///
/// # Parameters
/// - `f`: The formatter.
/// - `open`: The open containers.
/// - `pretty`: Whether to render in pretty mode.
///
/// # Returns
/// The result of writing to the formatter.
///
/// # Errors
/// Propagates any error from the formatter.
fn begin_value(
	f: &mut Formatter<'_>,
	open: &mut [Container],
	pretty: bool
) -> fmt::Result
{
	let levels = open.len();
	match open.last_mut()
	{
		Some(container) if !container.is_struct =>
		{
			if pretty
			{
				if !container.has_fields
				{
					f.write_str("(\n")?;
				}
				indent(f, levels)?;
			}
			else
			{
				f.write_str(if container.has_fields { ", " } else { "(" })?;
			}
			container.has_fields = true;
			Ok(())
		},
		_ => Ok(())
	}
}

/// Finish rendering a value. In pretty mode, a value inside a container ends
/// its line with a comma.
///
/// # Parameters
/// - `f`: The formatter.
/// - `open`: The open containers.
/// - `pretty`: Whether to render in pretty mode.
///
/// # Returns
/// The result of writing to the formatter.
///
/// # Errors
/// Propagates any error from the formatter.
fn end_value(
	f: &mut Formatter<'_>,
	open: &[Container],
	pretty: bool
) -> fmt::Result
{
	if pretty && !open.is_empty()
	{
		f.write_str(",\n")?;
	}
	Ok(())
}

/// Write the indentation for the specified number of levels, four spaces per
/// level.
///
/// # Parameters
/// - `f`: The destination.
/// - `levels`: The number of levels.
///
/// # Returns
/// The result of writing to the destination.
///
/// # Errors
/// Propagates any error from the destination.
fn indent(f: &mut impl Write, levels: usize) -> fmt::Result
{
	/// A run of 64 spaces, written in chunks to keep deep indentation cheap.
	const SPACES: &str =
		"                                                                ";
	let mut remaining = 4 * levels;
	while remaining > 0
	{
		let chunk = remaining.min(SPACES.len());
		f.write_str(&SPACES[..chunk])?;
		remaining -= chunk;
	}
	Ok(())
}

/// A writer that indents every line but the first, as the
/// [`Formatter`]'s own pad adapter does for each level of the derived pretty
/// [`Debug`] format.
struct Indenter<'f, 'b>
{
	/// The underlying formatter.
	f: &'f mut Formatter<'b>,

	/// The number of levels by which to indent.
	levels: usize,

	/// Whether the last character written was a newline.
	on_newline: bool
}

impl Write for Indenter<'_, '_>
{
	fn write_str(&mut self, s: &str) -> fmt::Result
	{
		for line in s.split_inclusive('\n')
		{
			if self.on_newline
			{
				indent(self.f, self.levels)?;
			}
			self.on_newline = line.ends_with('\n');
			self.f.write_str(line)?;
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Equality and hashing.                            //
////////////////////////////////////////////////////////////////////////////////

/// Compare two ASTs for structural equality, including spans.
///
/// # Parameters
/// - `left`: The root of the first AST.
/// - `right`: The root of the second AST.
///
/// # Returns
/// `true` if the ASTs are equal, `false` otherwise.
pub(super) fn eq(left: Node<'_, '_>, right: Node<'_, '_>) -> bool
{
	Events::new(left).eq(Events::new(right))
}

/// Hash an AST, consistently with [`eq`].
///
/// # Parameters
/// - `root`: The root of the AST.
/// - `state`: The hasher.
pub(super) fn hash<H: Hasher>(root: Node<'_, '_>, state: &mut H)
{
	Events::new(root).for_each(|event| event.hash(state));
}
