//! # Parser
//!
//! Herein are the parser combinators for the `xDy` language. [`parse`] is the
//! main entry point, which discards leading whitespace before parsing a
//! [function](Function). The recognized grammar is as follows, in Extended
//! Backus-Naur Form (EBNF), where non-terminals are in lowercase and terminals
//! are in uppercase:
//!
//! ```text
//! function ::= [parameters] expression
//! parameters ::= parameter {',' parameter} ':'
//! parameter ::= IDENTIFIER
//! expression ::= add_sub
//! add_sub ::= mul_div_mod {('+' | '-') mul_div_mod}
//! mul_div_mod ::= exponent {('*' | '×' | '/' | '÷' | '%') exponent}
//! exponent ::= unary {'^' unary}
//! unary ::= ('-' unary | primary)
//! primary ::= range | dice group | CONSTANT | variable
//! group ::= '(' expression ')'
//! variable ::= '{' IDENTIFIER '}'
//! range ::= '[' expression ':' expression ']'
//! dice ::= (standard_dice | custom_dice) (drop_lowest | drop_highest)*
//! standard_dice ::= dice_count D_OPERATOR standard_faces
//! custom_dice ::= dice_count D_OPERATOR custom_faces
//! dice_count ::= CONSTANT | variable | group
//! standard_faces ::= CONSTANT | variable | group
//! custom_faces ::= '[' CONSTANT {',' CONSTANT} ']'
//! drop_lowest ::= 'drop' 'lowest' [drop_expression]
//! drop_highest ::= 'drop' 'highest' [drop_expression]
//! drop_expression ::= CONSTANT | variable | group
//! CONSTANT ::= ['-'] {DIGIT}
//! D_OPERATOR ::= 'd' | 'D'
//! IDENTIFIER ::= {ALPHA1 | '_'} {ALPHANUMBERIC1 | '_' | '-'}
//! DIGIT ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
//! ALPHA1 ::= ? any alphabetic character ?
//! ALPHANUMBERIC1 ::= ? any alphanumeric character ?
//! WHITESPACE ::= ? any whitespace, permitted between tokens ?
//! ```
//!
//! Parsing is based on the [`nom`] library, which provides a combinator-based
//! approach to parsing. With the exception of [`parse`], all functions expect
//! the input to be free of leading whitespace.

use std::{
	fmt::{self, Display, Formatter, Write},
	ops::Deref
};

use nom::{
	IResult, Parser,
	branch::alt,
	bytes::complete::{tag, take_while_m_n, take_while1},
	character::complete::{char, digit1, multispace0, one_of},
	combinator::{map, opt, recognize},
	multi::{fold_many0, many0, separated_list1},
	sequence::{delimited, pair, preceded, separated_pair, terminated}
};
use nom_locate::LocatedSpan;

////////////////////////////////////////////////////////////////////////////////
//                             Debugging support.                             //
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

/// A function definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function<'a>
{
	/// The formal parameters of the function, if any.
	pub parameters: Option<Vec<&'a str>>,

	/// The body of the function.
	pub body: Expression<'a>
}

impl Display for Function<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		if let Some(ref parameters) = self.parameters
		{
			for (i, param) in parameters.iter().enumerate()
			{
				if i > 0
				{
					write!(f, ", ")?;
				}
				write!(f, "{}", param)?;
			}
			write!(f, ": ")?;
		}
		write!(f, "{}", self.body)
	}
}

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

/// A parenthesized expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group<'a>
{
	/// The expression inside the parentheses.
	pub expression: Box<Expression<'a>>
}

impl Display for Group<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "({})", self.expression)
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

/// A constant value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Constant(pub i32);

impl Display for Constant
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}", self.0)
	}
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

/// A variable reference.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Variable<'a>(pub &'a str);

impl Display for Variable<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{{{}}}", self.0)
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

/// A range expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Range<'a>
{
	/// The start of the range.
	pub start: Box<Expression<'a>>,

	/// The end of the range.
	pub end: Box<Expression<'a>>
}

impl Display for Range<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "[{}:{}]", self.start, self.end)
	}
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

/// An arbitrary expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression<'a>
{
	/// A parenthesized expression.
	Group(Group<'a>),

	/// A constant value.
	Constant(Constant),

	/// A variable reference.
	Variable(Variable<'a>),

	/// A range expression.
	Range(Range<'a>),

	/// A dice expression.
	Dice(DiceExpression<'a>),

	/// An arithmetic expression.
	Arithmetic(ArithmeticExpression<'a>)
}

impl Display for Expression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			Expression::Group(group) => write!(f, "{}", group),
			Expression::Constant(constant) => write!(f, "{}", constant),
			Expression::Variable(variable) => write!(f, "{}", variable),
			Expression::Range(range) => write!(f, "{}", range),
			Expression::Dice(dice) => write!(f, "{}", dice),
			Expression::Arithmetic(arithmetic) => write!(f, "{}", arithmetic)
		}
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

/// A standard dice expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandardDice<'a>
{
	/// The number of dice to roll.
	pub count: Box<Expression<'a>>,

	/// The number of faces on each die, starting at 1.
	pub faces: Box<Expression<'a>>
}

impl Display for StandardDice<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}D{}", self.count, self.faces)
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

/// A custom dice expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomDice<'a>
{
	/// The number of dice to roll.
	pub count: Box<Expression<'a>>,

	/// The faces themselves.
	pub faces: Vec<i32>
}

impl Display for CustomDice<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}D[", self.count)?;
		for (i, face) in self.faces.iter().enumerate()
		{
			if i > 0
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", face)?;
		}
		write!(f, "]")
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

/// A drop-lowest expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropLowest<'a>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'a>>,

	/// The number of dice to drop. Defaults to 1.
	pub drop: Option<Box<Expression<'a>>>
}

impl Display for DropLowest<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} drop lowest", self.dice)?;
		if let Some(ref drop) = self.drop
		{
			write!(f, " {}", drop)?;
		}
		Ok(())
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

/// A drop-highest expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropHighest<'a>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'a>>,

	/// The number of dice to drop. Defaults to 1.
	pub drop: Option<Box<Expression<'a>>>
}

impl Display for DropHighest<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} drop highest", self.dice)?;
		if let Some(ref drop) = self.drop
		{
			write!(f, " {}", drop)?;
		}
		Ok(())
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

/// A dice expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiceExpression<'a>
{
	/// A standard dice expression.
	Standard(StandardDice<'a>),

	/// A custom dice expression.
	Custom(CustomDice<'a>),

	/// A drop-lowest expression.
	DropLowest(DropLowest<'a>),

	/// A drop-highest expression.
	DropHighest(DropHighest<'a>)
}

impl Display for DiceExpression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			DiceExpression::Standard(dice) => write!(f, "{}", dice),
			DiceExpression::Custom(dice) => write!(f, "{}", dice),
			DiceExpression::DropLowest(drop) => write!(f, "{}", drop),
			DiceExpression::DropHighest(drop) => write!(f, "{}", drop)
		}
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

/// An addition expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add<'a>
{
	/// The augend.
	pub left: Box<Expression<'a>>,

	/// The addend.
	pub right: Box<Expression<'a>>
}

impl Display for Add<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} + {}", self.left, self.right)
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

/// A subtraction expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sub<'a>
{
	/// The minuend.
	pub left: Box<Expression<'a>>,

	/// The subtrahend.
	pub right: Box<Expression<'a>>
}

impl Display for Sub<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} - {}", self.left, self.right)
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

/// A multiplication expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul<'a>
{
	/// The multiplicand.
	pub left: Box<Expression<'a>>,

	/// The multiplier.
	pub right: Box<Expression<'a>>
}

impl Display for Mul<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} * {}", self.left, self.right)
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

/// A division expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Div<'a>
{
	/// The dividend.
	pub left: Box<Expression<'a>>,

	/// The divisor.
	pub right: Box<Expression<'a>>
}

impl Display for Div<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} / {}", self.left, self.right)
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

/// A modulo expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mod<'a>
{
	/// The dividend.
	pub left: Box<Expression<'a>>,

	/// The divisor.
	pub right: Box<Expression<'a>>
}

impl Display for Mod<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} % {}", self.left, self.right)
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

/// An exponentiation expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exp<'a>
{
	/// The base.
	pub left: Box<Expression<'a>>,

	/// The exponent.
	pub right: Box<Expression<'a>>
}

impl Display for Exp<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} ^ {}", self.left, self.right)
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

/// A negation expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neg<'a>
{
	/// The operand.
	pub operand: Box<Expression<'a>>
}

impl Display for Neg<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "-{}", self.operand)
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

/// An arithmetic expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticExpression<'a>
{
	/// An addition expression.
	Add(Add<'a>),

	/// A subtraction expression.
	Sub(Sub<'a>),

	/// A multiplication expression.
	Mul(Mul<'a>),

	/// A division expression.
	Div(Div<'a>),

	/// A modulo expression.
	Mod(Mod<'a>),

	/// An exponentiation expression.
	Exp(Exp<'a>),

	/// A negation expression.
	Neg(Neg<'a>)
}

impl Display for ArithmeticExpression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			ArithmeticExpression::Add(add) => write!(f, "{}", add),
			ArithmeticExpression::Sub(sub) => write!(f, "{}", sub),
			ArithmeticExpression::Mul(mul) => write!(f, "{}", mul),
			ArithmeticExpression::Div(div) => write!(f, "{}", div),
			ArithmeticExpression::Mod(r#mod) => write!(f, "{}", r#mod),
			ArithmeticExpression::Exp(exp) => write!(f, "{}", exp),
			ArithmeticExpression::Neg(neg) => write!(f, "{}", neg)
		}
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
//                                  Parsers.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text.
pub type Span<'a> = LocatedSpan<&'a str>;

/// Parse a function definition, discarding leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed function definition.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parse(input: &str) -> IResult<Span, Function>
{
	preceded(multispace0, function).parse_complete(Span::new(input))
}

/// Parse a function definition, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed function definition.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn function(input: Span) -> IResult<Span, Function>
{
	let (input, parameters) = opt(parameters).parse_complete(input)?;
	let (input, body) =
		preceded(multispace0, expression).parse_complete(input)?;
	let parameters = parameters.map(|params| {
		params.into_iter().map(|param| *param.fragment()).collect()
	});
	Ok((input, Function { parameters, body }))
}

/// Parse a list of formal parameters, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed list of formal parameters.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parameters(input: Span) -> IResult<Span, Vec<Span>>
{
	terminated(
		separated_list1(
			preceded(multispace0, char(',')),
			preceded(multispace0, parameter)
		),
		preceded(multispace0, char(':'))
	)
	.parse_complete(input)
}

/// Parse a formal parameter, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed formal parameter.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parameter(input: Span) -> IResult<Span, Span> { identifier(input) }

/// Parse an expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn expression(input: Span) -> IResult<Span, Expression> { add_sub(input) }

/// Parse an addition or subtraction expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn add_sub(input: Span) -> IResult<Span, Expression>
{
	let (input, initial) = mul_div_mod(input)?;
	let (input, remainder) = many0(pair(
		alt((
			preceded(multispace0, char('+')),
			preceded(multispace0, char('-'))
		)),
		preceded(multispace0, mul_div_mod)
	))
	.parse_complete(input)?;

	Ok((
		input,
		remainder
			.into_iter()
			.fold(initial, |acc, (op, expr)| match op
			{
				'+' => Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(acc),
					right: Box::new(expr)
				})),
				'-' => Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
					left: Box::new(acc),
					right: Box::new(expr)
				})),
				_ => unreachable!()
			})
	))
}

/// Parse a multiplication, division, or modulo expression, without leading
/// whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn mul_div_mod(input: Span) -> IResult<Span, Expression>
{
	let (input, initial) = exponent(input)?;
	let (input, remainder) = many0(pair(
		alt((
			preceded(multispace0, one_of("*×")),
			preceded(multispace0, one_of("/÷")),
			preceded(multispace0, char('%'))
		)),
		preceded(multispace0, exponent)
	))
	.parse_complete(input)?;

	Ok((
		input,
		remainder
			.into_iter()
			.fold(initial, |acc, (op, expr)| match op
			{
				'*' | '×' =>
				{
					Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
						left: Box::new(acc),
						right: Box::new(expr)
					}))
				},
				'/' | '÷' =>
				{
					Expression::Arithmetic(ArithmeticExpression::Div(Div {
						left: Box::new(acc),
						right: Box::new(expr)
					}))
				},
				'%' => Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
					left: Box::new(acc),
					right: Box::new(expr)
				})),
				_ => unreachable!()
			})
	))
}

/// Parse an exponentiation expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn exponent(input: Span) -> IResult<Span, Expression>
{
	let (input, initial) = unary(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, char('^')),
		preceded(multispace0, unary)
	))
	.parse_complete(input)?;
	let exponents =
		remainder.into_iter().rev().reduce(|(_, acc), (_, expr)| {
			(
				'\0',
				Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
					left: Box::new(expr),
					right: Box::new(acc)
				}))
			)
		});
	match exponents
	{
		Some((_, expr)) => Ok((
			input,
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(initial),
				right: Box::new(expr)
			}))
		)),
		None => Ok((input, initial))
	}
}

/// Parse a unary expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn unary(input: Span) -> IResult<Span, Expression>
{
	alt((
		map(preceded(char('-'), preceded(multispace0, unary)), |expr| {
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(expr)
			}))
		}),
		preceded(multispace0, primary)
	))
	.parse_complete(input)
}

/// Parse a primary expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn primary(input: Span) -> IResult<Span, Expression>
{
	alt((
		map(range, Expression::Range),
		map(dice, Expression::Dice),
		map(group, Expression::Group),
		map(variable, Expression::Variable),
		map(constant, Expression::Constant)
	))
	.parse_complete(input)
}

/// Parse a group expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed group expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn group(input: Span) -> IResult<Span, Group>
{
	delimited(
		char('('),
		map(preceded(multispace0, expression), Box::new),
		preceded(multispace0, char(')'))
	)
	.parse_complete(input)
	.map(|(input, expression)| (input, Group { expression }))
}

/// Parse a variable reference, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed variable reference.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn variable(input: Span) -> IResult<Span, Variable>
{
	delimited(
		char('{'),
		preceded(multispace0, identifier),
		preceded(multispace0, char('}'))
	)
	.parse_complete(input)
	.map(|(input, span)| (input, Variable(span.fragment())))
}

/// Parse a range expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed range expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn range(input: Span) -> IResult<Span, Range>
{
	delimited(
		char('['),
		separated_pair(
			preceded(multispace0, expression),
			preceded(multispace0, char(':')),
			preceded(multispace0, expression)
		),
		preceded(multispace0, char(']'))
	)
	.parse_complete(input)
	.map(|(input, (start, end))| {
		(
			input,
			Range {
				start: Box::new(start),
				end: Box::new(end)
			}
		)
	})
}

/// Parse a dice expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed dice expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn dice(input: Span) -> IResult<Span, DiceExpression>
{
	let (input, initial_dice) = alt((
		map(standard_dice, DiceExpression::Standard),
		map(custom_dice, DiceExpression::Custom)
	))
	.parse_complete(input)?;

	// Transient types for differentiating between drop-lowest and drop-highest
	// drop expressions.
	enum DropType<'a>
	{
		Lowest(Option<Box<Expression<'a>>>),
		Highest(Option<Box<Expression<'a>>>)
	}

	let (input, final_dice) = fold_many0(
		alt((
			map(drop_lowest, DropType::Lowest),
			map(drop_highest, DropType::Highest)
		)),
		|| initial_dice.clone(),
		|acc, t| match t
		{
			DropType::Lowest(drop) => DiceExpression::DropLowest(DropLowest {
				dice: Box::new(acc),
				drop
			}),
			DropType::Highest(drop) =>
			{
				DiceExpression::DropHighest(DropHighest {
					dice: Box::new(acc),
					drop
				})
			},
		}
	)
	.parse_complete(input)?;

	Ok((input, final_dice))
}

/// Parse a standard dice expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed standard dice expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn standard_dice(input: Span) -> IResult<Span, StandardDice>
{
	separated_pair(
		dice_count,
		preceded(multispace0, d_operator),
		preceded(multispace0, standard_faces)
	)
	.parse_complete(input)
	.map(|(input, (count, faces))| {
		(
			input,
			StandardDice {
				count: Box::new(count),
				faces: Box::new(faces)
			}
		)
	})
}

/// Parse a custom dice expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed custom dice expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn custom_dice(input: Span) -> IResult<Span, CustomDice>
{
	separated_pair(
		dice_count,
		preceded(multispace0, d_operator),
		preceded(multispace0, custom_faces)
	)
	.parse_complete(input)
	.map(|(input, (count, faces))| {
		(
			input,
			CustomDice {
				count: Box::new(count),
				faces
			}
		)
	})
}

/// Parse a dice count expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed dice count expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn dice_count(input: Span) -> IResult<Span, Expression>
{
	alt((
		map(constant, Expression::Constant),
		map(variable, Expression::Variable),
		map(group, Expression::Group)
	))
	.parse_complete(input)
}

/// Parse standard faces, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed standard faces.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn standard_faces(input: Span) -> IResult<Span, Expression>
{
	alt((
		map(constant, Expression::Constant),
		map(variable, Expression::Variable),
		map(group, Expression::Group)
	))
	.parse_complete(input)
}

/// Parse custom faces, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed custom faces.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn custom_faces(input: Span) -> IResult<Span, Vec<i32>>
{
	delimited(
		char('['),
		separated_list1(
			preceded(multispace0, char(',')),
			preceded(multispace0, map(constant, |c| c.0))
		),
		preceded(multispace0, char(']'))
	)
	.parse_complete(input)
}

/// Parse a drop-lowest expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed drop-lowest expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn drop_lowest(input: Span) -> IResult<Span, Option<Box<Expression<'_>>>>
{
	let (input, (_, _, drop)) = (
		preceded(multispace0, tag("drop")),
		preceded(multispace0, tag("lowest")),
		opt(preceded(multispace0, drop_expression))
	)
		.parse_complete(input)?;
	Ok((input, drop.map(Box::new)))
}

/// Parse a drop-highest expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed drop-highest expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn drop_highest(input: Span) -> IResult<Span, Option<Box<Expression<'_>>>>
{
	let (input, (_, _, drop)) = (
		preceded(multispace0, tag("drop")),
		preceded(multispace0, tag("highest")),
		opt(preceded(multispace0, drop_expression))
	)
		.parse_complete(input)?;
	Ok((input, drop.map(Box::new)))
}

/// Parse a drop expression, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed drop expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn drop_expression(input: Span) -> IResult<Span, Expression>
{
	alt((
		map(constant, Expression::Constant),
		map(variable, Expression::Variable),
		map(group, Expression::Group)
	))
	.parse_complete(input)
}

/// Parse a constant value, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed constant value.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn constant(input: Span) -> IResult<Span, Constant>
{
	let (input, constant) =
		recognize(pair(opt(char('-')), digit1)).parse_complete(input)?;
	let constant = constant.fragment().parse().map_err(|_| {
		nom::Err::Error(nom::error::Error::new(
			input,
			nom::error::ErrorKind::Digit
		))
	})?;
	Ok((input, Constant(constant)))
}

/// Parse a `d` or `D` operator, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed operator.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn d_operator(input: Span) -> IResult<Span, char> { one_of("dD")(input) }

/// Parse an identifier, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed identifier.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn identifier(input: Span) -> IResult<Span, Span>
{
	recognize(pair(
		alt((alpha, tag("_"))),
		many0(alt((alphanumeric1, tag("_"), tag("-"))))
	))
	.parse_complete(input)
}

////////////////////////////////////////////////////////////////////////////////
//                              Utility parsers.                              //
////////////////////////////////////////////////////////////////////////////////

/// Parse one or more alphabetic characters, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed alphabetic characters.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn alpha(input: Span) -> IResult<Span, Span>
{
	take_while_m_n(1, 1, |c: char| c.is_alphabetic())(input)
}

/// Parse one or more alphanumeric characters, without leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed alphanumeric characters.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn alphanumeric1(input: Span) -> IResult<Span, Span>
{
	take_while1(|c: char| c.is_alphanumeric())(input)
}
