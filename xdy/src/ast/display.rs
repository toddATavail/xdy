//! # Iterative rendering
//!
//! Herein is the engine behind the [`Display`](std::fmt::Display)
//! implementations of every AST type. A node renders as a sequence of
//! [pieces](Piece) — literal text, names, integers, and child nodes — that its
//! [`Render`] implementation schedules on an explicit stack. The engine pops
//! the pieces in order, writing each literal at once and expanding each child
//! in place, so its stack depth is constant however deep the tree.
//!
//! Each type's [`Render`] implementation is the only statement of its textual
//! format; its [`Display`](std::fmt::Display) implementation just calls
//! [`render`]. An enum schedules the pieces of the struct that it wraps
//! directly, rather than through that struct's [`Display`](std::fmt::Display),
//! so nothing recurses.
//!
//! Children render without the formatter's flags, since the recursive
//! implementations that these replaced wrote every child through a fresh
//! `write!(f, "{}", child)`.

use std::fmt::{self, Formatter};

use super::{
	Add, ArithmeticExpression, Binding, Constant, CustomDice, DiceExpression,
	Div, DropHighest, DropLowest, Exp, Expression, Function, Group, Mod, Mul,
	Neg, Parameter, Range, StandardDice, Sub, Variable
};

////////////////////////////////////////////////////////////////////////////////
//                                  Pieces.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A piece of the rendering of an AST. The lifetime `'a` bounds both the borrow
/// of the tree and its source text, which outlives it.
#[derive(Copy, Clone)]
pub(super) enum Piece<'a>
{
	/// Literal text.
	Text(&'static str),

	/// A name borrowed from the tree.
	Name(&'a str),

	/// An integer.
	Integer(i32),

	/// The faces of a custom die, separated by commas.
	Faces(&'a [i32]),

	/// The formal parameters of a function, separated by commas.
	Parameters(&'a [Parameter<'a>]),

	/// An expression, to expand in place.
	Expression(&'a Expression<'a>),

	/// A dice expression, to expand in place.
	Dice(&'a DiceExpression<'a>)
}

/// An AST type that can [render] itself as a sequence of [pieces](Piece).
pub(super) trait Render
{
	/// Schedule the pieces of the receiver's rendering.
	///
	/// # Parameters
	/// - `stack`: The pending pieces, the next of which is on top. The
	///   receiver's pieces go atop it, so that the first of them is next.
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>);
}

/// Schedule the specified pieces, the first of which will be next.
///
/// # Parameters
/// - `stack`: The pending pieces, the next of which is on top.
/// - `pieces`: The pieces, in order of rendering.
fn schedule<'a>(stack: &mut Vec<Piece<'a>>, pieces: &[Piece<'a>])
{
	stack.extend(pieces.iter().rev());
}

////////////////////////////////////////////////////////////////////////////////
//                                 Rendering.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Render an AST node without recursion.
///
/// # Parameters
/// - `root`: The node.
/// - `f`: The formatter.
///
/// # Returns
/// The result of writing to the formatter.
///
/// # Errors
/// Propagates any error from the formatter.
pub(super) fn render(
	root: &(impl Render + ?Sized),
	f: &mut Formatter<'_>
) -> fmt::Result
{
	let mut stack = Vec::new();
	root.schedule(&mut stack);
	while let Some(piece) = stack.pop()
	{
		match piece
		{
			Piece::Text(text) | Piece::Name(text) => f.write_str(text)?,
			Piece::Integer(value) => write!(f, "{}", value)?,
			Piece::Faces(faces) =>
			{
				write_separated(f, faces, |f, face| write!(f, "{}", face))?
			},
			Piece::Parameters(parameters) =>
			{
				write_separated(f, parameters, |f, parameter| {
					write!(f, "{{{}}}", parameter.name)
				})?
			},
			Piece::Expression(expression) => expression.schedule(&mut stack),
			Piece::Dice(dice) => dice.schedule(&mut stack)
		}
	}
	Ok(())
}

/// Write a list of items, separated by commas.
///
/// # Parameters
/// - `f`: The formatter.
/// - `items`: The items.
/// - `write_item`: Writes a single item.
///
/// # Returns
/// The result of writing to the formatter.
///
/// # Errors
/// Propagates any error from the formatter.
fn write_separated<T>(
	f: &mut Formatter<'_>,
	items: &[T],
	mut write_item: impl FnMut(&mut Formatter<'_>, &T) -> fmt::Result
) -> fmt::Result
{
	for (i, item) in items.iter().enumerate()
	{
		if i > 0
		{
			f.write_str(", ")?;
		}
		write_item(f, item)?;
	}
	Ok(())
}

////////////////////////////////////////////////////////////////////////////////
//                                Scheduling.                                 //
////////////////////////////////////////////////////////////////////////////////

impl Render for Function<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		match &self.parameters
		{
			Some(parameters) => schedule(
				stack,
				&[
					Piece::Parameters(parameters),
					Piece::Text(": "),
					Piece::Expression(&self.body)
				]
			),
			None => schedule(stack, &[Piece::Expression(&self.body)])
		}
	}
}

impl Render for Parameter<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[Piece::Text("{"), Piece::Name(&self.name), Piece::Text("}")]
		);
	}
}

impl Render for Group<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[
				Piece::Text("("),
				Piece::Expression(&self.expression),
				Piece::Text(")")
			]
		);
	}
}

impl Render for Constant
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		stack.push(Piece::Integer(self.value));
	}
}

impl Render for Variable<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[Piece::Text("{"), Piece::Name(&self.name), Piece::Text("}")]
		);
	}
}

impl Render for Binding<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[
				Piece::Text("{"),
				Piece::Name(&self.name),
				Piece::Text("}@("),
				Piece::Expression(&self.expression),
				Piece::Text(")")
			]
		);
	}
}

impl Render for Range<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[
				Piece::Text("["),
				Piece::Expression(&self.start),
				Piece::Text(":"),
				Piece::Expression(&self.end),
				Piece::Text("]")
			]
		);
	}
}

impl Render for Expression<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		match self
		{
			Expression::Group(group) => group.schedule(stack),
			Expression::Constant(constant) => constant.schedule(stack),
			Expression::Variable(variable) => variable.schedule(stack),
			Expression::Binding(binding) => binding.schedule(stack),
			Expression::Range(range) => range.schedule(stack),
			Expression::Dice(dice) => dice.schedule(stack),
			Expression::Arithmetic(arithmetic) => arithmetic.schedule(stack)
		}
	}
}

impl Render for StandardDice<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[
				Piece::Expression(&self.count),
				Piece::Text("D"),
				Piece::Expression(&self.faces)
			]
		);
	}
}

impl Render for CustomDice<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(
			stack,
			&[
				Piece::Expression(&self.count),
				Piece::Text("D["),
				Piece::Faces(&self.faces),
				Piece::Text("]")
			]
		);
	}
}

impl Render for DropLowest<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_drop(stack, &self.dice, " drop lowest", self.drop.as_deref());
	}
}

impl Render for DropHighest<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_drop(stack, &self.dice, " drop highest", self.drop.as_deref());
	}
}

/// Schedule the pieces of a drop clause.
///
/// # Parameters
/// - `stack`: The pending pieces, the next of which is on top.
/// - `dice`: The dice expression from which to drop.
/// - `keywords`: The keywords of the clause, with a leading space.
/// - `drop`: The number of dice to drop, if given.
fn schedule_drop<'a>(
	stack: &mut Vec<Piece<'a>>,
	dice: &'a DiceExpression<'a>,
	keywords: &'static str,
	drop: Option<&'a Expression<'a>>
)
{
	if let Some(drop) = drop
	{
		schedule(stack, &[Piece::Text(" "), Piece::Expression(drop)]);
	}
	schedule(stack, &[Piece::Dice(dice), Piece::Text(keywords)]);
}

impl Render for DiceExpression<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		match self
		{
			DiceExpression::Standard(dice) => dice.schedule(stack),
			DiceExpression::Custom(dice) => dice.schedule(stack),
			DiceExpression::DropLowest(clause) => clause.schedule(stack),
			DiceExpression::DropHighest(clause) => clause.schedule(stack)
		}
	}
}

impl Render for Add<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " + ", &self.right);
	}
}

impl Render for Sub<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " - ", &self.right);
	}
}

impl Render for Mul<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " * ", &self.right);
	}
}

impl Render for Div<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " / ", &self.right);
	}
}

impl Render for Mod<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " % ", &self.right);
	}
}

impl Render for Exp<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule_binary(stack, &self.left, " ^ ", &self.right);
	}
}

/// Schedule the pieces of a binary operation.
///
/// # Parameters
/// - `stack`: The pending pieces, the next of which is on top.
/// - `left`: The left operand.
/// - `operator`: The operator, surrounded by spaces.
/// - `right`: The right operand.
fn schedule_binary<'a>(
	stack: &mut Vec<Piece<'a>>,
	left: &'a Expression<'a>,
	operator: &'static str,
	right: &'a Expression<'a>
)
{
	schedule(
		stack,
		&[
			Piece::Expression(left),
			Piece::Text(operator),
			Piece::Expression(right)
		]
	);
}

impl Render for Neg<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		schedule(stack, &[Piece::Text("-"), Piece::Expression(&self.operand)]);
	}
}

impl Render for ArithmeticExpression<'_>
{
	fn schedule<'a>(&'a self, stack: &mut Vec<Piece<'a>>)
	{
		match self
		{
			ArithmeticExpression::Add(add) => add.schedule(stack),
			ArithmeticExpression::Sub(sub) => sub.schedule(stack),
			ArithmeticExpression::Mul(mul) => mul.schedule(stack),
			ArithmeticExpression::Div(div) => div.schedule(stack),
			ArithmeticExpression::Mod(r#mod) => r#mod.schedule(stack),
			ArithmeticExpression::Exp(exp) => exp.schedule(stack),
			ArithmeticExpression::Neg(neg) => neg.schedule(stack)
		}
	}
}
