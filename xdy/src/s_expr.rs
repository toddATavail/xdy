//! # S-expressions
//!
//! S-expressions, à la Lisp, are a classic way of representing data structures
//! in a minimal, human-readable format. Whereas S-expressions form the primary
//! mechanism for representing both data and code in Lisp, `xDy` uses them only
//! for testing and debugging.

use std::{
	fmt::{self, Write},
	ops::Deref
};

use crate::ast::{
	Add, ArithmeticExpression, Constant, CustomDice, DiceExpression, Div,
	DropHighest, DropLowest, Exp, Expression, Function, Group, Mod, Mul, Neg,
	Range, StandardDice, Sub, Variable
};

////////////////////////////////////////////////////////////////////////////////
//                           S-expression support.                            //
////////////////////////////////////////////////////////////////////////////////

/// Provides the capability to format the receiver as an S-expression.
pub trait SExpressible
{
	/// Write an S-expression representation of the receiver to the given write
	/// stream.
	///
	/// # Parameters
	///
	/// * `f`: The write stream.
	/// * `remaining_space`: The remaining space available in the current line.
	/// * `options`: The formatting options.
	///
	/// # Returns
	///
	/// `true` if formatting split across multiple lines, `false` otherwise.
	///
	/// # Errors
	///
	/// If the formatting fails for any reason.
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result;

	/// Size an S-expression representation of the receiver. Sizing is always
	/// computed minimally, as if no indentation were applied. This is used
	/// principally for pretty-printing.
	///
	/// # Parameters
	///
	/// * `options`: The formatting options.
	///
	/// # Returns
	///
	/// The size of the S-expression representation of the receiver.
	fn size_s_expr(&self) -> usize;

	/// Produce an S-expression representation of the receiver.
	///
	/// # Parameters
	///
	/// * `options`: The formatting options.
	///
	/// # Returns
	///
	/// The S-expression representation of the receiver.
	fn to_s_expr(&self, options: SExpressibleOptions) -> String
	{
		// The `unwrap` is safe here because writing to a string buffer is
		// infallible.
		let mut buffer = String::new();
		self.write_s_expr(&mut buffer, options.soft_limit, options)
			.unwrap();
		buffer
	}
}

/// Options for customizing S-expression formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SExpressibleOptions
{
	/// The indentation level, measured in tabs.
	pub indent: usize,

	/// The number of spaces per tab.
	pub tab_width: usize,

	/// The soft limit of the line length.
	pub soft_limit: usize
}

impl SExpressibleOptions
{
	/// Create a new set of options.
	///
	/// # Parameters
	///
	/// * `indent`: The indentation level, measured in tabs.
	/// * `tab_width`: The number of spaces per tab.
	/// * `soft_limit`: The soft limit of the line length.
	///
	/// # Returns
	///
	/// A new set of options.
	#[inline]
	pub fn new(indent: usize, tab_width: usize, soft_limit: usize) -> Self
	{
		Self {
			indent,
			tab_width,
			soft_limit
		}
	}

	/// Construct a new set of options with increased indentation.
	///
	/// # Returns
	///
	/// The new set of options.
	#[inline]
	pub fn increase_indent(&self) -> Self
	{
		Self {
			indent: self.indent + 1,
			tab_width: self.tab_width,
			soft_limit: self.soft_limit
		}
	}

	/// Compute the available space for a fresh line.
	///
	/// # Returns
	///
	/// The available space for a fresh line.
	pub fn available_space(&self) -> usize
	{
		self.soft_limit - self.indent * self.tab_width
	}
}

impl Default for SExpressibleOptions
{
	fn default() -> Self
	{
		Self {
			indent: 0,
			tab_width: 4,
			soft_limit: 80
		}
	}
}

impl<T> SExpressible for &T
where
	T: SExpressible
{
	#[inline]
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		(*self).write_s_expr(f, remaining_space, options)
	}

	#[inline]
	fn size_s_expr(&self) -> usize { (*self).size_s_expr() }
}

impl SExpressible for Option<Vec<&str>>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		match self
		{
			Some(vec) => (&vec[..]).write_s_expr(f, remaining_space, options),
			None => write!(f, "[]")
		}
	}

	fn size_s_expr(&self) -> usize
	{
		match self
		{
			Some(vec) => (&vec[..]).size_s_expr(),
			None => 2
		}
	}
}

impl SExpressible for &[&str]
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		if self.is_empty()
		{
			// The vector is empty, so always write an empty list directly to
			// the target writer, ignoring any other formatting considerations.
			write!(f, "[]")
		}
		else if remaining_space >= self.size_s_expr()
		{
			// Write the vector on a single line.
			write!(f, "[")?;
			for (i, item) in self.iter().enumerate()
			{
				if i > 0
				{
					write!(f, " ")?;
				}
				write!(f, "{}", item)?;
			}
			write!(f, "]")
		}
		else
		{
			// Split the vector across multiple lines.
			write!(f, "[")?;
			for item in self.iter()
			{
				// Note that we don't care how long the item is, because we
				// can't split it across multiple lines anyway.
				write!(f, "\n{}{}", "\t".repeat(options.indent + 1), item)?;
			}
			write!(f, "\n{}]", "\t".repeat(options.indent))
		}
	}

	fn size_s_expr(&self) -> usize
	{
		// The delimiters and interposed spaces consume one character each. Note
		// that there are one fewer interposed spaces than items, so the
		// constant term is 1 after accounting for brackets (not 2).
		self.iter().copied().map(str::len).sum::<usize>() + self.len() + 1
	}
}

impl SExpressible for &[i32]
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		if self.is_empty()
		{
			// The vector is empty, so always write an empty list directly to
			// the target writer, ignoring any other formatting considerations.
			write!(f, "[]")
		}
		else if remaining_space >= self.size_s_expr()
		{
			// Write the vector on a single line.
			write!(f, "[")?;
			for (i, item) in self.iter().enumerate()
			{
				if i > 0
				{
					write!(f, " ")?;
				}
				write!(f, "{}", item)?;
			}
			write!(f, "]")
		}
		else
		{
			// Split the vector across multiple lines.
			write!(f, "[")?;
			for item in self.iter()
			{
				// Note that we don't care how long the item's print
				// representation is, because we can't split it across
				// multiple lines anyway.
				write!(f, "\n{}{}", "\t".repeat(options.indent + 1), item)?;
			}
			write!(f, "\n{}]", "\t".repeat(options.indent))
		}
	}

	fn size_s_expr(&self) -> usize
	{
		// The delimiters and interposed spaces consume one character each. Note
		// that there are one fewer interposed spaces than items, so the
		// constant term is 1 after accounting for brackets (not 2).
		self.iter()
			.map(|&x| {
				// Count the number of characters required to represent the
				// value, taking care to account correctly for negative numbers.
				(x.abs() as f64).log10().floor() as usize
					+ if x < 0 { 1 } else { 0 }
					+ 1
			})
			.sum::<usize>()
			+ self.len()
			+ 1
	}
}

impl<A, B> SExpressible for (A, B)
where
	A: AsRef<str>,
	B: SExpressible
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		let keyword = self.0.as_ref();
		write!(f, "({}", keyword)?;
		let sub = &self.1;
		// We need enough space for ourselves plus the trailing parentheses of
		// enclosing expressions, of which there are as many as the indentation
		// level.
		let space_needed = self.size_s_expr() + options.indent;
		let remaining_space = if remaining_space >= space_needed
		{
			write!(f, " ")?;
			// We know that the remainder fits, so answer an effectively
			// infinite value.
			usize::MAX
		}
		else
		{
			let options = options.increase_indent();
			write!(f, "\n{}", "\t".repeat(options.indent))?;
			options.available_space()
		};
		sub.write_s_expr(f, remaining_space, options)?;
		write!(f, ")")
	}

	fn size_s_expr(&self) -> usize
	{
		// Account for two parentheses, the keyword, one interposing space, and
		// the size of the subexpression.
		3 + self.0.as_ref().len() + self.1.size_s_expr()
	}
}

impl<A, B, C> SExpressible for (A, B, C)
where
	A: AsRef<str>,
	B: SExpressible,
	C: SExpressible
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		let keyword = self.0.as_ref();
		write!(f, "({}", keyword)?;
		let sub1 = &self.1;
		let sub2 = &self.2;
		// We need enough space for ourselves plus the trailing parentheses of
		// enclosing expressions, of which there are as many as the indentation
		// level.
		let space_needed = self.size_s_expr() + options.indent;
		if remaining_space >= space_needed
		{
			write!(f, " ")?;
			// We know that the whole expression fits, so use an effectively
			// infinite value for the remaining space of the subexpressions.
			// This avoids recursing into the subexpressions again.
			sub1.write_s_expr(f, usize::MAX, options)?;
			write!(f, " ")?;
			sub2.write_s_expr(f, usize::MAX, options)?;
		}
		else
		{
			let options = options.increase_indent();
			write!(f, "\n{}", "\t".repeat(options.indent))?;
			sub1.write_s_expr(f, options.available_space(), options)?;
			write!(f, "\n{}", "\t".repeat(options.indent))?;
			sub2.write_s_expr(f, options.available_space(), options)?;
		};
		write!(f, ")")
	}

	fn size_s_expr(&self) -> usize
	{
		// Account for two parentheses, the keyword, two spaces, and the size of
		// the subexpressions.
		5 + self.0.as_ref().len() + self.1.size_s_expr() + self.2.size_s_expr()
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        Abstract syntax tree (AST).                         //
////////////////////////////////////////////////////////////////////////////////

impl SExpressible for Function<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		let keyword = "function";
		write!(f, "({}", keyword)?;
		// The remaining space discounts the open parenthesis and the keyword.
		let remaining_space = remaining_space - 9;
		// We want to write the parameters on the same line as the keyword if
		// at all possible.
		let params = &self.parameters;
		let space_needed = params.size_s_expr();
		let remaining_space = if remaining_space >= space_needed
		{
			write!(f, " ")?;
			params.write_s_expr(f, remaining_space, options)?;
			remaining_space - space_needed
		}
		else
		{
			let options = options.increase_indent();
			write!(f, "\n{}", "\t".repeat(options.indent))?;
			params.write_s_expr(f, options.available_space(), options)?;
			options.available_space()
		};
		// Write out the body.
		let body = &self.body;
		let space_needed = body.size_s_expr();
		if remaining_space >= space_needed
		{
			write!(f, " ")?;
			body.write_s_expr(f, remaining_space, options)?;
		}
		else
		{
			let options = options.increase_indent();
			write!(f, "\n{}", "\t".repeat(options.indent))?;
			body.write_s_expr(f, options.available_space(), options)?;
		};
		write!(f, ")")
	}

	fn size_s_expr(&self) -> usize
	{
		// Account for two parentheses, the keyword, two spaces, and the size of
		// the subexpressions.
		12 + self.parameters.size_s_expr() + self.body.size_s_expr()
	}
}

impl SExpressible for Group<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Groups are transparent in the S-expression representation, so we can
		// simply forward the call to the subexpression.
		self.expression.write_s_expr(f, remaining_space, options)
	}

	fn size_s_expr(&self) -> usize { self.expression.size_s_expr() }
}

impl SExpressible for Constant
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		_remaining_space: usize,
		_options: SExpressibleOptions
	) -> fmt::Result
	{
		// Constants cannot be split, so we just write the constant value.
		write!(f, "{}", self.0)
	}

	fn size_s_expr(&self) -> usize
	{
		// Count the number of characters required to represent the value,
		// taking care to account correctly for negative numbers.
		(self.0.abs() as f64).log10().floor() as usize
			+ if self.0 < 0 { 1 } else { 0 }
			+ 1
	}
}

impl SExpressible for Variable<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		_remaining_space: usize,
		_options: SExpressibleOptions
	) -> fmt::Result
	{
		// Variables cannot be split, so we just write the variable.
		write!(f, "{}", self.0)
	}

	fn size_s_expr(&self) -> usize { self.0.len() }
}

impl SExpressible for Range<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("range", self.start.deref(), self.end.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("range", self.start.deref(), self.end.deref()).size_s_expr()
	}
}

impl SExpressible for Expression<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Expressions are transparent in the S-expression representation, so we
		// simply forward the call to the appropriate variant.
		match self
		{
			Expression::Group(group) =>
			{
				group.write_s_expr(f, remaining_space, options)
			},
			Expression::Constant(constant) =>
			{
				constant.write_s_expr(f, remaining_space, options)
			},
			Expression::Variable(variable) =>
			{
				variable.write_s_expr(f, remaining_space, options)
			},
			Expression::Range(range) =>
			{
				range.write_s_expr(f, remaining_space, options)
			},
			Expression::Dice(dice) =>
			{
				dice.write_s_expr(f, remaining_space, options)
			},
			Expression::Arithmetic(arithmetic) =>
			{
				arithmetic.write_s_expr(f, remaining_space, options)
			},
		}
	}

	fn size_s_expr(&self) -> usize
	{
		match self
		{
			Expression::Group(group) => group.size_s_expr(),
			Expression::Constant(constant) => constant.size_s_expr(),
			Expression::Variable(variable) => variable.size_s_expr(),
			Expression::Range(range) => range.size_s_expr(),
			Expression::Dice(dice) => dice.size_s_expr(),
			Expression::Arithmetic(arithmetic) => arithmetic.size_s_expr()
		}
	}
}

impl SExpressible for StandardDice<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_spaces: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("standard-dice", self.count.deref(), self.faces.deref()).write_s_expr(
			f,
			remaining_spaces,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("standard-dice", self.count.deref(), self.faces.deref()).size_s_expr()
	}
}

impl SExpressible for CustomDice<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_spaces: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("custom-dice", self.count.deref(), self.faces.deref()).write_s_expr(
			f,
			remaining_spaces,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("custom-dice", self.count.deref(), self.faces.deref()).size_s_expr()
	}
}

impl SExpressible for DropLowest<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_spaces: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		(
			"drop-lowest",
			self.dice.deref(),
			self.drop.as_ref().map_or_else(
				|| &Expression::Constant(Constant(1)),
				|drop| drop.deref()
			)
		)
			.write_s_expr(f, remaining_spaces, options)
	}

	fn size_s_expr(&self) -> usize
	{
		(
			"drop-lowest",
			self.dice.deref(),
			self.drop.as_ref().map_or_else(
				|| &Expression::Constant(Constant(1)),
				|drop| drop.deref()
			)
		)
			.size_s_expr()
	}
}

impl SExpressible for DropHighest<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_spaces: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		(
			"drop-highest",
			self.dice.deref(),
			self.drop.as_ref().map_or_else(
				|| &Expression::Constant(Constant(1)),
				|drop| drop.deref()
			)
		)
			.write_s_expr(f, remaining_spaces, options)
	}

	fn size_s_expr(&self) -> usize
	{
		(
			"drop-highest",
			self.dice.deref(),
			self.drop.as_ref().map_or_else(
				|| &Expression::Constant(Constant(1)),
				|drop| drop.deref()
			)
		)
			.size_s_expr()
	}
}

impl SExpressible for DiceExpression<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Dice expressions are transparent in the S-expression representation,
		// so we can simply forward the call to the underlying variant.
		match self
		{
			DiceExpression::Standard(dice) =>
			{
				dice.write_s_expr(f, remaining_space, options)
			},
			DiceExpression::Custom(dice) =>
			{
				dice.write_s_expr(f, remaining_space, options)
			},
			DiceExpression::DropLowest(drop) =>
			{
				drop.write_s_expr(f, remaining_space, options)
			},
			DiceExpression::DropHighest(drop) =>
			{
				drop.write_s_expr(f, remaining_space, options)
			},
		}
	}

	fn size_s_expr(&self) -> usize
	{
		match self
		{
			DiceExpression::Standard(dice) => dice.size_s_expr(),
			DiceExpression::Custom(dice) => dice.size_s_expr(),
			DiceExpression::DropLowest(drop) => drop.size_s_expr(),
			DiceExpression::DropHighest(drop) => drop.size_s_expr()
		}
	}
}

impl SExpressible for Add<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("add", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("add", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Sub<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("sub", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("sub", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Mul<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("mul", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("mul", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Div<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("div", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("div", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Mod<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("mod", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("mod", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Exp<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("exp", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self) -> usize
	{
		("exp", self.left.deref(), self.right.deref()).size_s_expr()
	}
}

impl SExpressible for Neg<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		("neg", self.operand.deref()).write_s_expr(f, remaining_space, options)
	}

	fn size_s_expr(&self) -> usize
	{
		("neg", self.operand.deref()).size_s_expr()
	}
}

impl SExpressible for ArithmeticExpression<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Arithmetic expressions are transparent in the S-expression
		// representation, so we can simply forward the call to the underlying
		// variant.
		match self
		{
			ArithmeticExpression::Add(add) =>
			{
				add.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Sub(sub) =>
			{
				sub.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Mul(mul) =>
			{
				mul.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Div(div) =>
			{
				div.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Mod(r#mod) =>
			{
				r#mod.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Exp(exp) =>
			{
				exp.write_s_expr(f, remaining_space, options)
			},
			ArithmeticExpression::Neg(neg) =>
			{
				neg.write_s_expr(f, remaining_space, options)
			},
		}
	}

	fn size_s_expr(&self) -> usize
	{
		// Arithmetic expressions are transparent in the S-expression
		// representation, so we can simply forward the call to the underlying
		// variant.
		match self
		{
			ArithmeticExpression::Add(add) => add.size_s_expr(),
			ArithmeticExpression::Sub(sub) => sub.size_s_expr(),
			ArithmeticExpression::Mul(mul) => mul.size_s_expr(),
			ArithmeticExpression::Div(div) => div.size_s_expr(),
			ArithmeticExpression::Mod(r#mod) => r#mod.size_s_expr(),
			ArithmeticExpression::Exp(exp) => exp.size_s_expr(),
			ArithmeticExpression::Neg(neg) => neg.size_s_expr()
		}
	}
}
