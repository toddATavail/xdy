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

use crate::{
	ast::{
		Add, ArithmeticExpression, Constant, CustomDice, DiceExpression, Div,
		DropHighest, DropLowest, Exp, Expression, Function, Group, Mod, Mul,
		Neg, Parameter, Range, StandardDice, Sub, Variable
	},
	span::{SourceSpan, Spanned}
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
	/// - `f`: The write stream.
	/// - `remaining_space`: The remaining space available in the current line.
	/// - `options`: The formatting options.
	///
	/// # Returns
	/// `true` if formatting split across multiple lines, `false` otherwise.
	///
	/// # Errors
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
	/// - `options`: The formatting options. Sizing must agree with
	///   [`write_s_expr`](Self::write_s_expr) under the same options, so the
	///   receiver may consult, for example,
	///   [`with_spans`](SExpressibleOptions::with_spans) to decide whether to
	///   include the width of a `^[start end]` span-metadata prefix.
	///
	/// # Returns
	/// The size of the S-expression representation of the receiver.
	fn size_s_expr(&self, options: SExpressibleOptions) -> usize;

	/// Produce an S-expression representation of the receiver.
	///
	/// # Parameters
	/// - `options`: The formatting options.
	///
	/// # Returns
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

////////////////////////////////////////////////////////////////////////////////
//                         Span-metadata formatting.                          //
////////////////////////////////////////////////////////////////////////////////

/// Write a `^[start end]` span-metadata prefix (with trailing space) when the
/// options request span emission and the span is not
/// [synthetic](SourceSpan::SYNTHETIC). Otherwise write nothing.
///
/// # Parameters
/// - `f`: The write stream.
/// - `span`: The source span of the form about to be written.
/// - `options`: The formatting options.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// If the formatting fails for any reason.
fn write_span_prefix(
	f: &mut dyn Write,
	span: SourceSpan,
	options: SExpressibleOptions
) -> fmt::Result
{
	if options.with_spans && span != SourceSpan::SYNTHETIC
	{
		write!(f, "^[{} {}] ", span.start, span.end)
	}
	else
	{
		Ok(())
	}
}

/// Compute the width in characters of the span-metadata prefix that
/// [`write_span_prefix`] would emit for the given `span` and `options`, or
/// zero if no prefix would be emitted. The trailing separator space is
/// included.
///
/// # Parameters
/// - `span`: The source span of the form about to be written.
/// - `options`: The formatting options.
///
/// # Returns
/// The width of the prefix in characters, or zero when no prefix is emitted.
fn span_prefix_size(span: SourceSpan, options: SExpressibleOptions) -> usize
{
	if options.with_spans && span != SourceSpan::SYNTHETIC
	{
		// "^[S E] " = 1 (^) + 1 ([) + digits(start) + 1 (space) + digits(end)
		// + 1 (]) + 1 (trailing space) = 5 + digits(start) + digits(end).
		5 + decimal_digits(span.start) + decimal_digits(span.end)
	}
	else
	{
		0
	}
}

/// Count the decimal digits of a non-negative integer.
///
/// # Parameters
/// - `n`: The non-negative integer.
///
/// # Returns
/// The number of decimal digits required to represent `n`. Zero takes one
/// digit.
fn decimal_digits(n: usize) -> usize
{
	if n == 0 { 1 } else { n.ilog10() as usize + 1 }
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
	pub soft_limit: usize,

	/// Whether to emit `^[start end]` span-metadata prefixes before each form
	/// that carries a non-[synthetic](SourceSpan::SYNTHETIC) span. When
	/// `false` (the default), spans are suppressed unconditionally, producing
	/// clean output suitable for hand-written comparisons. When `true`, the
	/// writer emits enough metadata for [`read_s_expr`] to reconstruct the
	/// span-annotated AST losslessly. Spans equal to [`SourceSpan::SYNTHETIC`]
	/// are still suppressed to avoid vacuous `^[0 0]` ceremony on untethered
	/// or hand-built nodes.
	pub with_spans: bool,

	/// Whether to render [`Group`] nodes opaquely as `(group <expr>)` rather
	/// than transparently forwarding to their inner expression. The default is
	/// `false`, matching the traditional transparent rendering used by
	/// hand-authored s-expressions in the test corpus. Setting this to `true`
	/// (alongside [`with_spans`](Self::with_spans)) enables lossless
	/// round-trip through the S-expression format, because [`Group`] is
	/// load-bearing for the AST's [`Display`] fidelity.
	pub with_groups: bool
}

impl SExpressibleOptions
{
	/// Create a new set of options. [`with_spans`](Self::with_spans) and
	/// [`with_groups`](Self::with_groups) default to `false`; enable them via
	/// the corresponding builder methods.
	///
	/// # Parameters
	/// - `indent`: The indentation level, measured in tabs.
	/// - `tab_width`: The number of spaces per tab.
	/// - `soft_limit`: The soft limit of the line length.
	///
	/// # Returns
	/// A new set of options.
	#[inline]
	pub fn new(indent: usize, tab_width: usize, soft_limit: usize) -> Self
	{
		Self {
			indent,
			tab_width,
			soft_limit,
			with_spans: false,
			with_groups: false
		}
	}

	/// Construct a new set of options with increased indentation.
	///
	/// # Returns
	/// The new set of options.
	#[inline]
	pub fn increase_indent(&self) -> Self
	{
		Self {
			indent: self.indent + 1,
			tab_width: self.tab_width,
			soft_limit: self.soft_limit,
			with_spans: self.with_spans,
			with_groups: self.with_groups
		}
	}

	/// Answer a copy of these options with [`with_spans`](Self::with_spans) set
	/// to the given value.
	///
	/// # Parameters
	/// - `value`: Whether to emit span-metadata prefixes.
	///
	/// # Returns
	/// The updated options.
	#[inline]
	pub fn with_spans(mut self, value: bool) -> Self
	{
		self.with_spans = value;
		self
	}

	/// Answer a copy of these options with
	/// [`with_groups`](Self::with_groups) set to the given value.
	///
	/// # Parameters
	/// - `value`: Whether to render [`Group`] nodes opaquely.
	///
	/// # Returns
	/// The updated options.
	#[inline]
	pub fn with_groups(mut self, value: bool) -> Self
	{
		self.with_groups = value;
		self
	}

	/// Compute the available space for a fresh line.
	///
	/// # Returns
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
			soft_limit: 80,
			with_spans: false,
			with_groups: false
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
	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		(*self).size_s_expr(options)
	}
}

impl SExpressible for Option<Vec<Parameter<'_>>>
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		match self
		{
			Some(vec) => (&vec[..]).size_s_expr(options),
			None => 2
		}
	}
}

impl SExpressible for &[Parameter<'_>]
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
		else if remaining_space >= self.size_s_expr(options)
		{
			// Write the vector on a single line.
			write!(f, "[")?;
			for (i, item) in self.iter().enumerate()
			{
				if i > 0
				{
					write!(f, " ")?;
				}
				write_span_prefix(f, item.span, options)?;
				write_ident(f, item.name)?;
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
				write_span_prefix(f, item.span, options)?;
				write_ident(f, item.name)?;
			}
			write!(f, "\n{}]", "\t".repeat(options.indent))
		}
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		// The delimiters and interposed spaces consume one character each. Note
		// that there are one fewer interposed spaces than items, so the
		// constant term is 1 after accounting for brackets (not 2).
		self.iter()
			.map(|p| span_prefix_size(p.span, options) + ident_size(p.name))
			.sum::<usize>()
			+ self.len()
			+ 1
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
		else if remaining_space >= self.size_s_expr(options)
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

	fn size_s_expr(&self, _options: SExpressibleOptions) -> usize
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
		let space_needed = self.size_s_expr(options) + options.indent;
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		// Account for two parentheses, the keyword, one interposing space, and
		// the size of the subexpression.
		3 + self.0.as_ref().len() + self.1.size_s_expr(options)
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
		let space_needed = self.size_s_expr(options) + options.indent;
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		// Account for two parentheses, the keyword, two spaces, and the size of
		// the subexpressions.
		5 + self.0.as_ref().len()
			+ self.1.size_s_expr(options)
			+ self.2.size_s_expr(options)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        Identifier quoting support.                         //
////////////////////////////////////////////////////////////////////////////////

/// Answer whether the given identifier requires quoting in S-expression
/// format. An identifier needs quoting if it contains whitespace or
/// delimiter characters that would be ambiguous during parsing.
///
/// # Parameters
/// - `ident`: The identifier to inspect.
///
/// # Returns
/// `true` if `ident` must be surrounded by double quotes when written.
fn needs_quoting(ident: &str) -> bool
{
	ident.contains(|c: char| {
		c.is_whitespace() || matches!(c, '(' | ')' | '[' | ']' | '"')
	})
}

/// Write an identifier, quoting it with double quotes if necessary.
///
/// # Parameters
/// - `f`: The write stream.
/// - `ident`: The identifier to emit.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// If the formatting fails for any reason.
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
///
/// # Parameters
/// - `ident`: The identifier to size.
///
/// # Returns
/// The number of characters that [`write_ident`] would emit for `ident`.
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		let keyword = "function";
		write!(f, "({}", keyword)?;
		// The remaining space discounts the open parenthesis and the keyword.
		let remaining_space = remaining_space - 9;
		// We want to write the parameters on the same line as the keyword if
		// at all possible.
		let params = &self.parameters;
		let space_needed = params.size_s_expr(options);
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
		let space_needed = body.size_s_expr(options);
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		// Account for two parentheses, the keyword, two spaces, and the size of
		// the subexpressions.
		span_prefix_size(self.span, options)
			+ 12 + self.parameters.size_s_expr(options)
			+ self.body.size_s_expr(options)
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
		if options.with_groups
		{
			// Opaque rendering: `(group <expr>)` so the group's span and
			// structural identity survive the round-trip.
			write_span_prefix(f, self.span, options)?;
			let remaining_space = remaining_space
				.saturating_sub(span_prefix_size(self.span, options));
			("group", self.expression.deref()).write_s_expr(
				f,
				remaining_space,
				options
			)
		}
		else
		{
			// Transparent rendering: forward directly to the subexpression.
			self.expression.write_s_expr(f, remaining_space, options)
		}
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		if options.with_groups
		{
			span_prefix_size(self.span, options)
				+ ("group", self.expression.deref()).size_s_expr(options)
		}
		else
		{
			self.expression.size_s_expr(options)
		}
	}
}

impl SExpressible for Constant
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		_remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Constants cannot be split, so we just write the prefix (if any) and
		// the constant value.
		write_span_prefix(f, self.span, options)?;
		write!(f, "{}", self.value)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		// The exact character count of the decimal representation. Using
		// `ilog10` on the absolute value would panic on `i32::MIN`, so we
		// format directly.
		let body = if self.value == 0
		{
			1
		}
		else
		{
			format!("{}", self.value).len()
		};
		span_prefix_size(self.span, options) + body
	}
}

impl SExpressible for Variable<'_>
{
	fn write_s_expr(
		&self,
		f: &mut dyn Write,
		_remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		// Variables cannot be split, so we just write the prefix (if any) and
		// the variable, quoting it if it contains whitespace or delimiter
		// characters.
		write_span_prefix(f, self.span, options)?;
		write_ident(f, self.name)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options) + ident_size(self.name)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("range", self.start.deref(), self.end.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("range", self.start.deref(), self.end.deref())
				.size_s_expr(options)
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
		// simply forward the call to the appropriate variant. Each variant
		// emits its own span prefix (if any).
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		match self
		{
			Expression::Group(group) => group.size_s_expr(options),
			Expression::Constant(constant) => constant.size_s_expr(options),
			Expression::Variable(variable) => variable.size_s_expr(options),
			Expression::Range(range) => range.size_s_expr(options),
			Expression::Dice(dice) => dice.size_s_expr(options),
			Expression::Arithmetic(arithmetic) =>
			{
				arithmetic.size_s_expr(options)
			},
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
		write_span_prefix(f, self.span, options)?;
		let remaining_spaces = remaining_spaces
			.saturating_sub(span_prefix_size(self.span, options));
		("standard-dice", self.count.deref(), self.faces.deref()).write_s_expr(
			f,
			remaining_spaces,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("standard-dice", self.count.deref(), self.faces.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_spaces = remaining_spaces
			.saturating_sub(span_prefix_size(self.span, options));
		("custom-dice", self.count.deref(), self.faces.deref()).write_s_expr(
			f,
			remaining_spaces,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("custom-dice", self.count.deref(), self.faces.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_spaces = remaining_spaces
			.saturating_sub(span_prefix_size(self.span, options));
		match self.drop.as_ref()
		{
			Some(drop) => ("drop-lowest", self.dice.deref(), drop.deref())
				.write_s_expr(f, remaining_spaces, options),
			None => ("drop-lowest", self.dice.deref()).write_s_expr(
				f,
				remaining_spaces,
				options
			)
		}
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ match self.drop.as_ref()
			{
				Some(drop) => ("drop-lowest", self.dice.deref(), drop.deref())
					.size_s_expr(options),
				None => ("drop-lowest", self.dice.deref()).size_s_expr(options)
			}
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
		write_span_prefix(f, self.span, options)?;
		let remaining_spaces = remaining_spaces
			.saturating_sub(span_prefix_size(self.span, options));
		match self.drop.as_ref()
		{
			Some(drop) => ("drop-highest", self.dice.deref(), drop.deref())
				.write_s_expr(f, remaining_spaces, options),
			None => ("drop-highest", self.dice.deref()).write_s_expr(
				f,
				remaining_spaces,
				options
			)
		}
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ match self.drop.as_ref()
			{
				Some(drop) => ("drop-highest", self.dice.deref(), drop.deref())
					.size_s_expr(options),
				None => ("drop-highest", self.dice.deref()).size_s_expr(options)
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
		// so we can simply forward the call to the underlying variant. Each
		// variant emits its own span prefix (if any).
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		match self
		{
			DiceExpression::Standard(dice) => dice.size_s_expr(options),
			DiceExpression::Custom(dice) => dice.size_s_expr(options),
			DiceExpression::DropLowest(drop) => drop.size_s_expr(options),
			DiceExpression::DropHighest(drop) => drop.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("add", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("add", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("sub", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("sub", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("mul", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("mul", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("div", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("div", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("mod", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("mod", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("exp", self.left.deref(), self.right.deref()).write_s_expr(
			f,
			remaining_space,
			options
		)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("exp", self.left.deref(), self.right.deref())
				.size_s_expr(options)
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
		write_span_prefix(f, self.span, options)?;
		let remaining_space = remaining_space
			.saturating_sub(span_prefix_size(self.span, options));
		("neg", self.operand.deref()).write_s_expr(f, remaining_space, options)
	}

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		span_prefix_size(self.span, options)
			+ ("neg", self.operand.deref()).size_s_expr(options)
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
		// variant. Each variant emits its own span prefix (if any).
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

	fn size_s_expr(&self, options: SExpressibleOptions) -> usize
	{
		match self
		{
			ArithmeticExpression::Add(add) => add.size_s_expr(options),
			ArithmeticExpression::Sub(sub) => sub.size_s_expr(options),
			ArithmeticExpression::Mul(mul) => mul.size_s_expr(options),
			ArithmeticExpression::Div(div) => div.size_s_expr(options),
			ArithmeticExpression::Mod(r#mod) => r#mod.size_s_expr(options),
			ArithmeticExpression::Exp(exp) => exp.size_s_expr(options),
			ArithmeticExpression::Neg(neg) => neg.size_s_expr(options)
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
	/// Create a new reader positioned at the beginning of the given input.
	///
	/// # Parameters
	/// - `input`: The complete S-expression string to parse.
	///
	/// # Returns
	/// A fresh reader ready to consume `input` from byte offset zero.
	fn new(input: &'src str) -> Self { Self { input, pos: 0 } }

	/// Answer the remaining unread input from the current position onward.
	///
	/// # Returns
	/// A slice of the original input covering bytes from the current
	/// position to end of input.
	fn remaining(&self) -> &'src str { &self.input[self.pos..] }

	/// Advance past any ASCII whitespace characters at the current position.
	fn skip_whitespace(&mut self)
	{
		while self.pos < self.input.len()
			&& self.input.as_bytes()[self.pos].is_ascii_whitespace()
		{
			self.pos += 1;
		}
	}

	/// Consume and return the next character.
	///
	/// # Returns
	/// `Some(char)` with the consumed character, advancing the cursor past
	/// its UTF-8 encoding. `None` if the cursor is at end of input.
	fn next_char(&mut self) -> Option<char>
	{
		let c = self.remaining().chars().next()?;
		self.pos += c.len_utf8();
		Some(c)
	}

	/// Peek at the next character without consuming it.
	///
	/// # Returns
	/// `Some(char)` with the next character, or `None` at end of input.
	fn peek(&self) -> Option<char> { self.remaining().chars().next() }

	/// Skip leading whitespace and consume a specific expected character.
	///
	/// # Parameters
	/// - `expected`: The character that must appear next (after whitespace).
	///
	/// # Returns
	/// `Ok(())` after consuming the expected character.
	///
	/// # Errors
	/// - [`SExprError`] if the next character does not match `expected`, or if
	///   end of input is reached first.
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
	///
	/// # Returns
	/// A slice of the original input covering the word. Leading whitespace
	/// is skipped first.
	///
	/// # Errors
	/// - [`SExprError`] if no word characters are available at the current
	///   position (e.g., immediately followed by a delimiter or end of input).
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
	///
	/// # Returns
	/// A slice of the original input covering the identifier, with the
	/// surrounding quotes stripped when present.
	///
	/// # Errors
	/// - [`SExprError`] if a quoted identifier is unterminated or if the bare
	///   word reader fails.
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

	/// Read a signed integer literal.
	///
	/// # Returns
	/// The parsed [`i32`] value.
	///
	/// # Errors
	/// - [`SExprError`] if no word is available or the word does not parse as
	///   an [`i32`] (including overflow).
	fn read_integer(&mut self) -> Result<i32, SExprError>
	{
		let word = self.read_word()?;
		word.parse::<i32>().map_err(|e| SExprError {
			message: format!("invalid integer '{}': {}", word, e),
			offset: self.pos - word.len()
		})
	}

	/// Read a non-negative integer byte offset, as appears inside a
	/// `^[start end]` span-metadata prefix.
	///
	/// # Returns
	/// The parsed byte offset.
	///
	/// # Errors
	/// - [`SExprError`] if no word is available or the word does not parse as a
	///   non-negative integer.
	fn read_usize(&mut self) -> Result<usize, SExprError>
	{
		let word = self.read_word()?;
		word.parse::<usize>().map_err(|e| SExprError {
			message: format!("invalid byte offset '{}': {}", word, e),
			offset: self.pos - word.len()
		})
	}

	/// Read an optional `^[start end]` span-metadata prefix. The reader
	/// consumes whitespace only *after* the `]` terminator, so the caller
	/// can sit right against the following form (which itself may be
	/// preceded by whitespace via its own leading `skip_whitespace`).
	///
	/// Only the bracket-vector shape is recognized — an unrecognized metadata
	/// shape (e.g., `^{...}`, `^symbol`) is an explicit error, not silently
	/// skipped.
	///
	/// # Returns
	/// The parsed [`SourceSpan`], or [`SourceSpan::default`] (synthetic) when
	/// no prefix is present at the current position.
	///
	/// # Errors
	/// - [`SExprError`] if `^` is followed by anything other than `[`, if the
	///   integers inside the brackets are malformed, if `]` is missing, or if
	///   the resulting span has `start > end`.
	fn read_span_prefix(&mut self) -> Result<SourceSpan, SExprError>
	{
		self.skip_whitespace();
		if self.peek() != Some('^')
		{
			return Ok(SourceSpan::default());
		}
		let caret_offset = self.pos;
		self.next_char(); // consume `^`
		// `^` is a committed signal — no whitespace between `^` and `[`.
		match self.peek()
		{
			Some('[') =>
			{
				self.next_char();
			},
			Some(c) =>
			{
				return Err(SExprError {
					message: format!(
						"expected '[' after '^' (only '^[start end]' span \
					 metadata is supported), found '{}'",
						c
					),
					offset: self.pos
				})
			},
			None =>
			{
				return Err(SExprError {
					message: "expected '[' after '^', found end of input"
						.to_string(),
					offset: self.pos
				})
			},
		}
		let start = self.read_usize()?;
		let end = self.read_usize()?;
		self.skip_whitespace();
		self.expect_char(']')?;
		if start > end
		{
			return Err(SExprError {
				message: format!("span start {} exceeds end {}", start, end),
				offset: caret_offset
			});
		}
		Ok(SourceSpan { start, end })
	}

	/// Validate that `child` is contained within `parent`.
	///
	/// # Parameters
	/// - `parent`: The enclosing form's source span.
	/// - `child`: The child form's source span.
	/// - `offset`: The byte offset in the s-expression input to attribute any
	///   error to (typically the position at which the child began reading).
	///
	/// # Returns
	/// `Ok(())` if either operand is [synthetic](SourceSpan::SYNTHETIC), or if
	/// `parent.start <= child.start && child.end <= parent.end`.
	///
	/// # Errors
	/// - [`SExprError`] if both operands are non-synthetic and `child` escapes
	///   `parent` on either boundary.
	fn validate_containment(
		&self,
		parent: SourceSpan,
		child: SourceSpan,
		offset: usize
	) -> Result<(), SExprError>
	{
		if parent == SourceSpan::SYNTHETIC || child == SourceSpan::SYNTHETIC
		{
			return Ok(());
		}
		if parent.start <= child.start && child.end <= parent.end
		{
			Ok(())
		}
		else
		{
			Err(SExprError {
				message: format!(
					"child span {}..{} escapes parent span {}..{}",
					child.start, child.end, parent.start, parent.end
				),
				offset
			})
		}
	}

	/// Validate that `next` appears strictly after `prev` in source order.
	///
	/// # Parameters
	/// - `prev`: The previous sibling's source span.
	/// - `next`: The current sibling's source span.
	/// - `offset`: The byte offset in the s-expression input to attribute any
	///   error to (typically the position at which `next` began reading).
	///
	/// # Returns
	/// `Ok(())` if either operand is [synthetic](SourceSpan::SYNTHETIC), or
	/// if `prev.end <= next.start`.
	///
	/// # Errors
	/// - [`SExprError`] if both operands are non-synthetic and `next` overlaps
	///   or precedes `prev`.
	fn validate_sibling_order(
		&self,
		prev: SourceSpan,
		next: SourceSpan,
		offset: usize
	) -> Result<(), SExprError>
	{
		if prev == SourceSpan::SYNTHETIC || next == SourceSpan::SYNTHETIC
		{
			return Ok(());
		}
		if prev.end <= next.start
		{
			Ok(())
		}
		else
		{
			Err(SExprError {
				message: format!(
					"sibling span {}..{} overlaps or precedes prior sibling \
					 span {}..{}",
					next.start, next.end, prev.start, prev.end
				),
				offset
			})
		}
	}

	/// Read a subexpression and validate its span against the enclosing
	/// `parent` and the `prev_sibling` (if any).
	///
	/// # Parameters
	/// - `parent`: The enclosing compound form's span, for containment
	///   validation.
	/// - `prev_sibling`: The span of the immediately preceding sibling, for
	///   sibling-order validation. Pass [`SourceSpan::default`] when no prior
	///   sibling exists; the check is a no-op in that case.
	///
	/// # Returns
	/// The parsed subexpression, whose span has been validated against
	/// `parent` and `prev_sibling`.
	///
	/// # Errors
	/// - [`SExprError`] if the subexpression fails to parse, or if its span
	///   violates the containment or sibling-ordering constraints.
	fn read_child(
		&mut self,
		parent: SourceSpan,
		prev_sibling: SourceSpan
	) -> Result<Expression<'src>, SExprError>
	{
		self.skip_whitespace();
		let offset = self.pos;
		let expr = self.read_expr()?;
		let child = expr.span();
		self.validate_containment(parent, child, offset)?;
		self.validate_sibling_order(prev_sibling, child, offset)?;
		Ok(expr)
	}

	/// Read a parameter list: `[` (span-prefix? ident)* `]`. Parameters may
	/// each be preceded by a `^[start end]` span-metadata prefix.
	///
	/// # Parameters
	/// - `parent`: The enclosing function's span, for containment validation.
	///
	/// # Returns
	/// A pair of the parameter list (if any) and the span of the last
	/// parameter (or [`SourceSpan::default`] if no parameters were read).
	/// The latter is threaded into the body's sibling-ordering check.
	///
	/// # Errors
	/// - [`SExprError`] if the opening `[`, any identifier, or the closing `]`
	///   is malformed, or if any parameter's span violates containment or
	///   sibling-ordering constraints.
	fn read_params(
		&mut self,
		parent: SourceSpan
	) -> Result<(Option<Vec<Parameter<'src>>>, SourceSpan), SExprError>
	{
		self.skip_whitespace();
		self.expect_char('[')?;
		let mut params = Vec::new();
		let mut prev_sibling = SourceSpan::default();
		loop
		{
			self.skip_whitespace();
			if self.peek() == Some(']')
			{
				self.next_char();
				break;
			}
			let offset = self.pos;
			let span = self.read_span_prefix()?;
			let name = self.read_ident()?;
			self.validate_containment(parent, span, offset)?;
			self.validate_sibling_order(prev_sibling, span, offset)?;
			prev_sibling = span;
			params.push(Parameter { name, span });
		}
		let result = if params.is_empty()
		{
			None
		}
		else
		{
			Some(params)
		};
		Ok((result, prev_sibling))
	}

	/// Read a custom faces list: `[` integer+ `]`. Faces lists must contain
	/// at least one face.
	///
	/// # Returns
	/// The parsed list of face values.
	///
	/// # Errors
	/// - [`SExprError`] if the opening `[`, the closing `]`, or any integer is
	///   malformed, or if the list contains no faces.
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

	/// Read an expression, optionally preceded by a `^[start end]` span
	/// prefix. The prefix, when present, populates the resulting node's
	/// span; when absent, the span defaults to [`SourceSpan::default`].
	///
	/// # Returns
	/// The parsed expression.
	///
	/// # Errors
	/// - [`SExprError`] if the expression is malformed, if any child's span
	///   violates containment or sibling-ordering constraints, or if an
	///   unexpected keyword or end of input is encountered.
	fn read_expr(&mut self) -> Result<Expression<'src>, SExprError>
	{
		let span = self.read_span_prefix()?;
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
					"add" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Add(Add {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"sub" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"mul" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"div" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Div(Div {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"mod" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"exp" => self.read_binary(span, |l, r| {
						Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
							left: Box::new(l),
							right: Box::new(r),
							span
						}))
					})?,
					"neg" =>
					{
						let operand =
							self.read_child(span, SourceSpan::default())?;
						Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
							operand: Box::new(operand),
							span
						}))
					},
					"group" =>
					{
						let inner =
							self.read_child(span, SourceSpan::default())?;
						Expression::Group(Group {
							expression: Box::new(inner),
							span
						})
					},
					"standard-dice" =>
					{
						let count =
							self.read_child(span, SourceSpan::default())?;
						let faces = self.read_child(span, count.span())?;
						Expression::Dice(DiceExpression::Standard(
							StandardDice {
								count: Box::new(count),
								faces: Box::new(faces),
								span
							}
						))
					},
					"custom-dice" =>
					{
						let count =
							self.read_child(span, SourceSpan::default())?;
						let faces = self.read_faces()?;
						Expression::Dice(DiceExpression::Custom(CustomDice {
							count: Box::new(count),
							faces,
							span
						}))
					},
					"drop-lowest" =>
					{
						let dice =
							self.read_dice_child(span, SourceSpan::default())?;
						let drop =
							self.read_optional_drop(span, dice.span())?;
						Expression::Dice(DiceExpression::DropLowest(
							DropLowest {
								dice: Box::new(dice),
								drop,
								span
							}
						))
					},
					"drop-highest" =>
					{
						let dice =
							self.read_dice_child(span, SourceSpan::default())?;
						let drop =
							self.read_optional_drop(span, dice.span())?;
						Expression::Dice(DiceExpression::DropHighest(
							DropHighest {
								dice: Box::new(dice),
								drop,
								span
							}
						))
					},
					"range" =>
					{
						let start =
							self.read_child(span, SourceSpan::default())?;
						let end = self.read_child(span, start.span())?;
						Expression::Range(Range {
							start: Box::new(start),
							end: Box::new(end),
							span
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
				Ok(Expression::Constant(Constant { value, span }))
			},
			Some(_) =>
			{
				let name = self.read_ident()?;
				Ok(Expression::Variable(Variable { name, span }))
			},
			None => Err(SExprError {
				message: "expected expression, found end of input".to_string(),
				offset: self.pos
			})
		}
	}

	/// Read the optional drop-amount argument of a `drop-lowest` or
	/// `drop-highest` form. The drop amount is present iff another
	/// subexpression follows the dice expression before the closing
	/// parenthesis of the form. Its absence represents the
	/// parser-originated `drop: None` case (implicit single-die drop), so
	/// the format faithfully distinguishes `(drop-lowest d)` from
	/// `(drop-lowest d 1)`.
	///
	/// # Parameters
	/// - `parent`: The enclosing drop-lowest/drop-highest form's span.
	/// - `prev_sibling`: The span of the dice expression that preceded this
	///   position.
	///
	/// # Returns
	/// `Some(expr)` if a drop-amount subexpression is present, or `None` if
	/// the form closes immediately after its dice expression.
	///
	/// # Errors
	/// - [`SExprError`] if a subexpression is present but malformed, or if its
	///   span violates containment or sibling-ordering constraints.
	fn read_optional_drop(
		&mut self,
		parent: SourceSpan,
		prev_sibling: SourceSpan
	) -> Result<Option<Box<Expression<'src>>>, SExprError>
	{
		self.skip_whitespace();
		if self.peek() == Some(')')
		{
			return Ok(None);
		}
		let drop = self.read_child(parent, prev_sibling)?;
		Ok(Some(Box::new(drop)))
	}

	/// Read a dice expression (the first argument to `drop-lowest`/
	/// `drop-highest`) as a child of the enclosing compound form, validating
	/// containment and sibling order. Expects the child to parse to a
	/// [`DiceExpression`], not a general [`Expression`].
	///
	/// # Parameters
	/// - `parent`: The enclosing drop-lowest/drop-highest form's span.
	/// - `prev_sibling`: The span of the immediately preceding sibling, or
	///   [`SourceSpan::default`] when no prior sibling exists.
	///
	/// # Returns
	/// The parsed dice expression.
	///
	/// # Errors
	/// - [`SExprError`] if the subexpression fails to parse, if its span
	///   violates containment or sibling-ordering constraints, or if it does
	///   not parse to a [`DiceExpression`].
	fn read_dice_child(
		&mut self,
		parent: SourceSpan,
		prev_sibling: SourceSpan
	) -> Result<DiceExpression<'src>, SExprError>
	{
		let offset = self.pos;
		let expr = self.read_child(parent, prev_sibling)?;
		match expr
		{
			Expression::Dice(d) => Ok(d),
			_ => Err(SExprError {
				message: "expected dice expression as first argument to \
					drop-lowest/drop-highest"
					.to_string(),
				offset
			})
		}
	}

	/// Read two subexpressions — enclosed by `parent` — and combine them with
	/// the given constructor. Each child is validated for containment in
	/// `parent` and for sibling ordering.
	///
	/// # Parameters
	/// - `parent`: The enclosing binary form's span.
	/// - `f`: The constructor that combines the two parsed subexpressions into
	///   a single [`Expression`].
	///
	/// # Returns
	/// The expression produced by `f`, given the two parsed and validated
	/// subexpressions.
	///
	/// # Errors
	/// - [`SExprError`] if either subexpression fails to parse, or if either
	///   span violates containment or sibling-ordering constraints.
	fn read_binary(
		&mut self,
		parent: SourceSpan,
		f: impl FnOnce(Expression<'src>, Expression<'src>) -> Expression<'src>
	) -> Result<Expression<'src>, SExprError>
	{
		let left = self.read_child(parent, SourceSpan::default())?;
		let right = self.read_child(parent, left.span())?;
		Ok(f(left, right))
	}

	/// Read a complete function: optionally preceded by a `^[start end]`
	/// span prefix, then `(function params body)`. Consumes trailing
	/// whitespace after the closing parenthesis and errors if any further
	/// input remains.
	///
	/// # Returns
	/// The parsed [`Function`], whose span is taken from the leading prefix
	/// (or [`SourceSpan::default`] if absent).
	///
	/// # Errors
	/// - [`SExprError`] if the input is not a well-formed `(function params
	///   body)` s-expression, if any span validation fails, or if trailing
	///   input remains after the closing parenthesis.
	fn read_function(&mut self) -> Result<Function<'src>, SExprError>
	{
		let span = self.read_span_prefix()?;
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
		let (parameters, last_param_span) = self.read_params(span)?;
		let body = self.read_child(span, last_param_span)?;
		self.expect_char(')')?;
		self.skip_whitespace();
		if self.pos != self.input.len()
		{
			return Err(SExprError {
				message: format!("trailing input: '{}'", self.remaining()),
				offset: self.pos
			});
		}
		Ok(Function {
			parameters,
			body,
			span
		})
	}
}

/// Parse an S-expression string into a [`Function`].
///
/// The S-expression format mirrors the output of [`SExpressible::to_s_expr`]:
/// `(function [params...] body)`. Constants are bare
/// integers, variables are bare identifiers, and compound expressions use
/// keywords like `add`, `neg`, `standard-dice`, etc.
///
/// Parenthesized [`Group`] nodes may be either transparent (the subexpression
/// appears directly in the s-expression, matching the default writer output) or
/// opaque as `(group <expr>)` (the form emitted by the writer when
/// [`SExpressibleOptions::with_groups`] is enabled). The reader accepts both
/// forms.
///
/// Any form may be preceded by an optional `^[start end]` span-metadata prefix
/// specifying byte offsets within the originating source. When present, the
/// prefix populates the resulting AST node's [`SourceSpan`]; when absent, the
/// span defaults to [`SourceSpan::SYNTHETIC`]. Parameters inside the parameter
/// list `[...]` may each carry their own prefix. Only the bracket-vector shape
/// is recognized — unrecognized metadata shapes (e.g., `^{...}` or `^symbol`)
/// are an explicit error.
///
/// The reader validates span consistency: every non-synthetic span must be
/// strictly contained within its enclosing parent's non-synthetic span, and
/// siblings must appear in source order without overlap.
///
/// # Parameters
/// - `input`: The S-expression string to parse.
///
/// # Returns
/// The parsed function.
///
/// # Errors
/// - [`SExprError`] if the input is not a valid S-expression, if a span is
///   malformed (`start > end`), if a child's span escapes its parent, or if
///   siblings overlap or appear out of source order.
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
