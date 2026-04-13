//! # S-expressions
//!
//! S-expressions, à la Lisp, are a classic way of representing data structures
//! in a minimal, human-readable format. Whereas S-expressions form the primary
//! mechanism for representing both data and code in Lisp, `xDy` uses them only
//! for testing and debugging.

use std::{
	error::Error,
	fmt::{self, Display, Formatter, Write},
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
				write_ident(f, item)?;
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
				write!(f, "\n{}", "\t".repeat(options.indent + 1))?;
				write_ident(f, item)?;
			}
			write!(f, "\n{}]", "\t".repeat(options.indent))
		}
	}

	fn size_s_expr(&self) -> usize
	{
		// The delimiters and interposed spaces consume one character each. Note
		// that there are one fewer interposed spaces than items, so the
		// constant term is 1 after accounting for brackets (not 2).
		self.iter().copied().map(ident_size).sum::<usize>() + self.len() + 1
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
//                         Identifier quoting support.                        //
////////////////////////////////////////////////////////////////////////////////

/// Answer whether the given identifier requires quoting in S-expression
/// format. An identifier needs quoting if it contains whitespace or
/// delimiter characters that would be ambiguous during parsing.
fn needs_quoting(ident: &str) -> bool
{
	ident.contains(|c: char| {
		c.is_whitespace() || matches!(c, '(' | ')' | '[' | ']' | '"')
	})
}

/// Write an identifier, quoting it with double quotes if necessary.
fn write_ident(f: &mut dyn Write, ident: &str) -> fmt::Result
{
	if needs_quoting(ident)
	{
		write!(f, "\"{}\"", ident)
	}
	else
	{
		write!(f, "{}", ident)
	}
}

/// Answer the S-expression size of an identifier, accounting for quotes if
/// necessary.
fn ident_size(ident: &str) -> usize
{
	if needs_quoting(ident)
	{
		ident.len() + 2
	}
	else
	{
		ident.len()
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
		// The exact character count of the decimal representation. Using
		// `ilog10` on the absolute value would panic on `i32::MIN`, so we
		// format directly.
		if self.0 == 0
		{
			1
		}
		else
		{
			format!("{}", self.0).len()
		}
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
		// Variables cannot be split, so we just write the variable, quoting
		// it if it contains whitespace or delimiter characters.
		write_ident(f, self.0)
	}

	fn size_s_expr(&self) -> usize { ident_size(self.0) }
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

////////////////////////////////////////////////////////////////////////////////
//                            S-expression reader.                            //
////////////////////////////////////////////////////////////////////////////////

/// An error encountered while reading an S-expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SExprError
{
	/// A description of the error.
	pub message: String,

	/// The byte offset in the input where the error was detected.
	pub offset: usize
}

impl Display for SExprError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(
			f,
			"S-expression error at byte {}: {}",
			self.offset, self.message
		)
	}
}

impl Error for SExprError {}

/// A cursor into an S-expression string being parsed.
struct SExprReader<'src>
{
	/// The full input string.
	input: &'src str,

	/// The current byte offset.
	pos: usize
}

impl<'src> SExprReader<'src>
{
	/// Create a new reader at the beginning of the given input.
	fn new(input: &'src str) -> Self { Self { input, pos: 0 } }

	/// Answer the remaining unread input.
	fn remaining(&self) -> &'src str { &self.input[self.pos..] }

	/// Skip whitespace characters.
	fn skip_whitespace(&mut self)
	{
		while self.pos < self.input.len()
			&& self.input.as_bytes()[self.pos].is_ascii_whitespace()
		{
			self.pos += 1;
		}
	}

	/// Consume and return the next character, or `None` if at end of input.
	fn next_char(&mut self) -> Option<char>
	{
		let c = self.remaining().chars().next()?;
		self.pos += c.len_utf8();
		Some(c)
	}

	/// Peek at the next character without consuming it.
	fn peek(&self) -> Option<char> { self.remaining().chars().next() }

	/// Expect and consume a specific character, or return an error.
	fn expect_char(&mut self, expected: char) -> Result<(), SExprError>
	{
		self.skip_whitespace();
		match self.next_char()
		{
			Some(c) if c == expected => Ok(()),
			Some(c) => Err(SExprError {
				message: format!("expected '{}', found '{}'", expected, c),
				offset: self.pos - c.len_utf8()
			}),
			None => Err(SExprError {
				message: format!("expected '{}', found end of input", expected),
				offset: self.pos
			})
		}
	}

	/// Read a word: a contiguous sequence of non-whitespace, non-delimiter
	/// characters. Delimiters are `(`, `)`, `[`, `]`.
	fn read_word(&mut self) -> Result<&'src str, SExprError>
	{
		self.skip_whitespace();
		let start = self.pos;
		while self.pos < self.input.len()
		{
			let c = self.input.as_bytes()[self.pos];
			if c.is_ascii_whitespace()
				|| c == b'(' || c == b')'
				|| c == b'[' || c == b']'
			{
				break;
			}
			self.pos += 1;
		}
		if self.pos == start
		{
			Err(SExprError {
				message: "expected a word".to_string(),
				offset: self.pos
			})
		}
		else
		{
			Ok(&self.input[start..self.pos])
		}
	}

	/// Read an identifier: either a bare word or a double-quoted string.
	/// Quoted strings may contain whitespace and delimiter characters.
	fn read_ident(&mut self) -> Result<&'src str, SExprError>
	{
		self.skip_whitespace();
		if self.peek() == Some('"')
		{
			self.next_char(); // consume opening quote
			let start = self.pos;
			while self.pos < self.input.len()
				&& self.input.as_bytes()[self.pos] != b'"'
			{
				self.pos += 1;
			}
			if self.pos >= self.input.len()
			{
				return Err(SExprError {
					message: "unterminated quoted identifier".to_string(),
					offset: start - 1
				});
			}
			let ident = &self.input[start..self.pos];
			self.pos += 1; // consume closing quote
			Ok(ident)
		}
		else
		{
			self.read_word()
		}
	}

	/// Read an integer literal.
	fn read_integer(&mut self) -> Result<i32, SExprError>
	{
		let word = self.read_word()?;
		word.parse::<i32>().map_err(|e| SExprError {
			message: format!("invalid integer '{}': {}", word, e),
			offset: self.pos - word.len()
		})
	}

	/// Read a parameter list: `[` ident* `]`.
	fn read_params(&mut self) -> Result<Option<Vec<&'src str>>, SExprError>
	{
		self.skip_whitespace();
		self.expect_char('[')?;
		let mut params = Vec::new();
		loop
		{
			self.skip_whitespace();
			if self.peek() == Some(']')
			{
				self.next_char();
				break;
			}
			params.push(self.read_ident()?);
		}
		if params.is_empty()
		{
			Ok(None)
		}
		else
		{
			Ok(Some(params))
		}
	}

	/// Read a custom faces list: `[` integer+ `]`.
	fn read_faces(&mut self) -> Result<Vec<i32>, SExprError>
	{
		self.skip_whitespace();
		self.expect_char('[')?;
		let mut faces = Vec::new();
		loop
		{
			self.skip_whitespace();
			if self.peek() == Some(']')
			{
				self.next_char();
				break;
			}
			faces.push(self.read_integer()?);
		}
		if faces.is_empty()
		{
			Err(SExprError {
				message: "custom dice faces list must not be empty".to_string(),
				offset: self.pos
			})
		}
		else
		{
			Ok(faces)
		}
	}

	/// Read an expression.
	fn read_expr(&mut self) -> Result<Expression<'src>, SExprError>
	{
		self.skip_whitespace();
		match self.peek()
		{
			Some('(') =>
			{
				self.next_char();
				self.skip_whitespace();
				let keyword = self.read_word()?;
				let expr = match keyword
				{
					"add" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Add(Add {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"sub" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"mul" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"div" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Div(Div {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"mod" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"exp" => self.read_binary(|l, r| {
						Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
							left: Box::new(l),
							right: Box::new(r)
						}))
					})?,
					"neg" =>
					{
						let operand = self.read_expr()?;
						Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
							operand: Box::new(operand)
						}))
					},
					"standard-dice" =>
					{
						let count = self.read_expr()?;
						let faces = self.read_expr()?;
						Expression::Dice(DiceExpression::Standard(
							StandardDice {
								count: Box::new(count),
								faces: Box::new(faces)
							}
						))
					},
					"custom-dice" =>
					{
						let count = self.read_expr()?;
						let faces = self.read_faces()?;
						Expression::Dice(DiceExpression::Custom(CustomDice {
							count: Box::new(count),
							faces
						}))
					},
					"drop-lowest" =>
					{
						let dice = self.read_dice_expr()?;
						let drop = self.read_expr()?;
						Expression::Dice(DiceExpression::DropLowest(
							DropLowest {
								dice: Box::new(dice),
								drop: Some(Box::new(drop))
							}
						))
					},
					"drop-highest" =>
					{
						let dice = self.read_dice_expr()?;
						let drop = self.read_expr()?;
						Expression::Dice(DiceExpression::DropHighest(
							DropHighest {
								dice: Box::new(dice),
								drop: Some(Box::new(drop))
							}
						))
					},
					"range" =>
					{
						let start = self.read_expr()?;
						let end = self.read_expr()?;
						Expression::Range(Range {
							start: Box::new(start),
							end: Box::new(end)
						})
					},
					"function" =>
					{
						return Err(SExprError {
							message: "unexpected 'function' inside expression"
								.to_string(),
							offset: self.pos - keyword.len()
						});
					},
					_ =>
					{
						return Err(SExprError {
							message: format!("unknown keyword '{}'", keyword),
							offset: self.pos - keyword.len()
						});
					}
				};
				self.expect_char(')')?;
				Ok(expr)
			},
			Some(c) if c == '-' || c.is_ascii_digit() =>
			{
				let value = self.read_integer()?;
				Ok(Expression::Constant(Constant(value)))
			},
			Some(_) =>
			{
				let name = self.read_ident()?;
				Ok(Expression::Variable(Variable(name)))
			},
			None => Err(SExprError {
				message: "expected expression, found end of input".to_string(),
				offset: self.pos
			})
		}
	}

	/// Read a dice expression (the first argument to `drop-lowest`/
	/// `drop-highest`). This expects an S-expression that evaluates to a
	/// [`DiceExpression`], not a general [`Expression`].
	fn read_dice_expr(&mut self) -> Result<DiceExpression<'src>, SExprError>
	{
		let expr = self.read_expr()?;
		match expr
		{
			Expression::Dice(d) => Ok(d),
			_ => Err(SExprError {
				message: "expected dice expression as first argument to \
					drop-lowest/drop-highest"
					.to_string(),
				offset: self.pos
			})
		}
	}

	/// Read two subexpressions and combine them with the given constructor.
	fn read_binary(
		&mut self,
		f: impl FnOnce(Expression<'src>, Expression<'src>) -> Expression<'src>
	) -> Result<Expression<'src>, SExprError>
	{
		let left = self.read_expr()?;
		let right = self.read_expr()?;
		Ok(f(left, right))
	}

	/// Read a complete function: `(function params body)`.
	fn read_function(&mut self) -> Result<Function<'src>, SExprError>
	{
		self.skip_whitespace();
		self.expect_char('(')?;
		self.skip_whitespace();
		let keyword = self.read_word()?;
		if keyword != "function"
		{
			return Err(SExprError {
				message: format!("expected 'function', found '{}'", keyword),
				offset: self.pos - keyword.len()
			});
		}
		let parameters = self.read_params()?;
		let body = self.read_expr()?;
		self.expect_char(')')?;
		self.skip_whitespace();
		if self.pos != self.input.len()
		{
			return Err(SExprError {
				message: format!("trailing input: '{}'", self.remaining()),
				offset: self.pos
			});
		}
		Ok(Function { parameters, body })
	}
}

/// Parse an S-expression string into a [`Function`].
///
/// The S-expression format mirrors the output of
/// [`SExpressible::to_s_expr`]: `(function [params...] body)`. Constants
/// are bare integers, variables are bare identifiers, groups are
/// transparent (not represented), and compound expressions use keywords
/// like `add`, `neg`, `standard-dice`, etc.
///
/// # Parameters
/// - `input`: The S-expression string to parse.
///
/// # Returns
/// The parsed function.
///
/// # Errors
/// * [`SExprError`] if the input is not a valid S-expression.
///
/// # Examples
/// ```
/// use xdy::s_expr::read_s_expr;
///
/// let f = read_s_expr("(function [] 42)").unwrap();
/// assert_eq!(f.parameters, None);
/// ```
pub fn read_s_expr(input: &str) -> Result<Function<'_>, SExprError>
{
	SExprReader::new(input).read_function()
}
