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

use nom::{
	IResult, Input,
	bytes::complete::take_while,
	error::{ErrorKind, ParseError as NomParseError}
};
use nom_locate::LocatedSpan;

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

	/// Compute the available space for a fresh line. When the indentation
	/// consumes more than the [`soft_limit`](Self::soft_limit) allows, zero is
	/// returned — the soft limit is advisory rather than strictly enforced, and
	/// deep nesting beneath a tight limit is tolerated by emitting slightly
	/// wider output rather than panicking on underflow.
	///
	/// # Returns
	/// The available space for a fresh line, saturating at zero.
	pub fn available_space(&self) -> usize
	{
		self.soft_limit.saturating_sub(self.indent * self.tab_width)
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
		if self.is_empty()
		{
			// The empty vector renders as `[]`, mirroring the writer's
			// fast-path branch.
			return 2
		}
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
		if self.is_empty()
		{
			// The empty vector renders as `[]`, mirroring the writer's
			// fast-path branch.
			return 2
		}
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
		// Account for two parentheses, the keyword, two interposing spaces,
		// and the size of the subexpressions.
		4 + self.0.as_ref().len()
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
		// Saturate at zero to tolerate ambitious soft limits without panicking
		// — the remaining space drives a fits-or-wraps heuristic, not a hard
		// budget.
		let remaining_space = remaining_space.saturating_sub(9);
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

/// Parse an S-expression string into a [`Function`].
///
/// The S-expression format mirrors the output of [`SExpressible::to_s_expr`]:
/// `(function [params...] body)`. Constants are bare integers, variables are
/// bare identifiers, and compound expressions use keywords like `add`, `neg`,
/// `standard-dice`, etc.
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
///   siblings overlap or appear out of source order. Each variant pinpoints a
///   specific failure mode and carries an [`SExprLocation`].
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
	let span = Span::new(input);
	let (rest, function) = read_function(span).map_err(|e| match e
	{
		nom::Err::Error(e) | nom::Err::Failure(e) => e,
		nom::Err::Incomplete(_) => unreachable!()
	})?;
	if !rest.fragment().is_empty()
	{
		return Err(SExprError::TrailingInput {
			text: rest.fragment().to_string(),
			location: SExprLocation::of(rest)
		})
	}
	Ok(function)
}

////////////////////////////////////////////////////////////////////////////////
//                               Reader errors.                               //
////////////////////////////////////////////////////////////////////////////////

/// A source-position locator carried by every [`SExprError`] variant. Byte
/// offset plus 1-based line and column, the latter measured in `char`s —
/// matching the conventions of `nom_locate`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SExprLocation
{
	/// The byte offset within the input.
	pub offset: usize,

	/// The 1-based line number.
	pub line: u32,

	/// The 1-based column, measured in `char`s.
	pub column: usize
}

impl SExprLocation
{
	/// Capture the position at the given span's start.
	///
	/// # Parameters
	/// - `span`: The span whose start to capture.
	///
	/// # Returns
	/// The captured location.
	fn of(span: Span<'_>) -> Self
	{
		Self {
			offset: span.location_offset(),
			line: span.location_line(),
			column: span.get_utf8_column()
		}
	}
}

/// An error encountered while reading an S-expression. Each variant identifies
/// a specific failure mode — syntactic or semantic — so callers can
/// mechanically distinguish cases without parsing human-readable messages.
/// Every variant carries an [`SExprLocation`] locating the offending token
/// within the input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SExprError
{
	/// Generic syntax failure reported by a `nom` primitive that was not
	/// intercepted by a typed branch. In practice every decision point in the
	/// reader emits a typed variant, so this variant is defensive and should
	/// not normally appear. The `description` field carries a short hint
	/// derived from the [`ErrorKind`] that produced it.
	Syntax
	{
		/// A short hint describing the syntactic failure.
		description: String,

		/// The source position at which the parse was rejected.
		location: SExprLocation
	},

	/// The reader expected a specific character but found a different one, or
	/// end of input.
	ExpectedChar
	{
		/// The character the reader expected.
		expected: char,

		/// The character the reader actually observed, or `None` at end of
		/// input.
		found: Option<char>,

		/// The source position of the offending character (or of end-of-input
		/// when `found` is `None`).
		location: SExprLocation
	},

	/// The reader expected a bare word but the cursor sat on a delimiter or at
	/// end of input.
	ExpectedWord
	{
		/// The source position at which the word was expected.
		location: SExprLocation
	},

	/// A double-quoted identifier was opened but never terminated.
	UnterminatedQuotedIdent
	{
		/// The source position of the opening quote.
		location: SExprLocation
	},

	/// A word in integer position failed to parse as an [`i32`].
	InvalidInteger
	{
		/// The offending word, as written.
		text: String,

		/// The underlying parse error message.
		reason: String,

		/// The source position of the offending word.
		location: SExprLocation
	},

	/// A word inside a `^[start end]` span prefix failed to parse as a
	/// non-negative [`usize`].
	InvalidByteOffset
	{
		/// The offending word, as written.
		text: String,

		/// The underlying parse error message.
		reason: String,

		/// The source position of the offending word.
		location: SExprLocation
	},

	/// A `^` metadata sigil was followed by a character other than `[` (or by
	/// end of input). Only the bracket-vector shape is recognized —
	/// unrecognized shapes such as `^{...}`, `^symbol`, or `^ [...]` (with
	/// intervening whitespace) are rejected rather than silently skipped.
	UnrecognizedSpanShape
	{
		/// The character the reader actually observed, or `None` at end of
		/// input.
		found: Option<char>,

		/// The source position of the offending character (or of end-of-input
		/// when `found` is `None`).
		location: SExprLocation
	},

	/// A `^[start end]` span prefix with `start > end` — the range is inverted.
	InvertedSpan
	{
		/// The span's start offset.
		start: usize,

		/// The span's end offset.
		end: usize,

		/// The source position of the `^` that introduced the prefix.
		location: SExprLocation
	},

	/// A child node's [`SourceSpan`] escaped the containing parent's span.
	ChildSpanEscapesParent
	{
		/// The child's span.
		child: SourceSpan,

		/// The enclosing parent's span.
		parent: SourceSpan,

		/// The source position at which the child was read.
		location: SExprLocation
	},

	/// Two sibling nodes appear out of source order, or their
	/// [spans](SourceSpan) overlap.
	SiblingSpanOutOfOrder
	{
		/// The preceding sibling's span.
		prev: SourceSpan,

		/// The current sibling's span.
		next: SourceSpan,

		/// The source position at which the current sibling was read.
		location: SExprLocation
	},

	/// A `(custom-dice N [])` form carried an empty faces list. Source-level
	/// grammar forbids faceless custom dice and the reader preserves that
	/// contract.
	FacelessCustomDice
	{
		/// The source position immediately after the closing `]`.
		location: SExprLocation
	},

	/// The first argument to `(drop-lowest …)` or `(drop-highest …)` was not a
	/// dice expression.
	ExpectedDiceExpression
	{
		/// The source position at which the non-dice form was read.
		location: SExprLocation
	},

	/// The top-level form opened with a keyword other than `function`.
	ExpectedTopLevelFunction
	{
		/// The keyword that appeared instead.
		found: String,

		/// The source position of the offending keyword.
		location: SExprLocation
	},

	/// A `function` keyword appeared in nested expression position rather than
	/// at the top level.
	NestedFunctionKeyword
	{
		/// The source position of the offending keyword.
		location: SExprLocation
	},

	/// A compound form opened with an unrecognized keyword.
	UnknownKeyword
	{
		/// The offending keyword.
		keyword: String,

		/// The source position of the offending keyword.
		location: SExprLocation
	},

	/// An expression was expected but the input ended.
	ExpectedExpression
	{
		/// The source position of end-of-input.
		location: SExprLocation
	},

	/// Trailing input remained after the closing `)` of the top-level form.
	TrailingInput
	{
		/// The trailing text.
		text: String,

		/// The source position of the trailing text.
		location: SExprLocation
	}
}

impl SExprError
{
	/// Answer the source position at which the error was detected.
	///
	/// # Returns
	/// The offending token's [`SExprLocation`].
	pub fn location(&self) -> SExprLocation
	{
		match self
		{
			Self::Syntax { location, .. }
			| Self::ExpectedChar { location, .. }
			| Self::ExpectedWord { location }
			| Self::UnterminatedQuotedIdent { location }
			| Self::InvalidInteger { location, .. }
			| Self::InvalidByteOffset { location, .. }
			| Self::UnrecognizedSpanShape { location, .. }
			| Self::InvertedSpan { location, .. }
			| Self::ChildSpanEscapesParent { location, .. }
			| Self::SiblingSpanOutOfOrder { location, .. }
			| Self::FacelessCustomDice { location }
			| Self::ExpectedDiceExpression { location }
			| Self::ExpectedTopLevelFunction { location, .. }
			| Self::NestedFunctionKeyword { location }
			| Self::UnknownKeyword { location, .. }
			| Self::ExpectedExpression { location }
			| Self::TrailingInput { location, .. } => *location
		}
	}
}

impl Display for SExprError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		let loc = self.location();
		write!(
			f,
			"S-expression error at line {}, column {} (byte {}): ",
			loc.line, loc.column, loc.offset
		)?;
		match self
		{
			Self::Syntax { description, .. } =>
			{
				write!(f, "syntax error ({})", description)
			},
			Self::ExpectedChar {
				expected,
				found: None,
				..
			} => write!(f, "expected '{}', found end of input", expected),
			Self::ExpectedChar {
				expected,
				found: Some(c),
				..
			} => write!(f, "expected '{}', found '{}'", expected, c),
			Self::ExpectedWord { .. } => write!(f, "expected a word"),
			Self::UnterminatedQuotedIdent { .. } =>
			{
				write!(f, "unterminated quoted identifier")
			},
			Self::InvalidInteger { text, reason, .. } =>
			{
				write!(f, "invalid integer '{}': {}", text, reason)
			},
			Self::InvalidByteOffset { text, reason, .. } =>
			{
				write!(f, "invalid byte offset '{}': {}", text, reason)
			},
			Self::UnrecognizedSpanShape { found: None, .. } =>
			{
				write!(f, "expected '[' after '^', found end of input")
			},
			Self::UnrecognizedSpanShape { found: Some(c), .. } => write!(
				f,
				"expected '[' after '^' (only '^[start end]' span metadata \
				 is supported), found '{}'",
				c
			),
			Self::InvertedSpan { start, end, .. } =>
			{
				write!(f, "span start {} exceeds end {}", start, end)
			},
			Self::ChildSpanEscapesParent { child, parent, .. } => write!(
				f,
				"child span {}..{} escapes parent span {}..{}",
				child.start, child.end, parent.start, parent.end
			),
			Self::SiblingSpanOutOfOrder { prev, next, .. } => write!(
				f,
				"sibling span {}..{} overlaps or precedes prior sibling span \
				 {}..{}",
				next.start, next.end, prev.start, prev.end
			),
			Self::FacelessCustomDice { .. } =>
			{
				write!(f, "custom dice faces list must not be empty")
			},
			Self::ExpectedDiceExpression { .. } => write!(
				f,
				"expected dice expression as first argument to \
				 drop-lowest/drop-highest"
			),
			Self::ExpectedTopLevelFunction { found, .. } =>
			{
				write!(f, "expected 'function', found '{}'", found)
			},
			Self::NestedFunctionKeyword { .. } =>
			{
				write!(f, "unexpected 'function' inside expression")
			},
			Self::UnknownKeyword { keyword, .. } =>
			{
				write!(f, "unknown keyword '{}'", keyword)
			},
			Self::ExpectedExpression { .. } =>
			{
				write!(f, "expected expression, found end of input")
			},
			Self::TrailingInput { text, .. } =>
			{
				write!(f, "trailing input: '{}'", text)
			}
		}
	}
}

impl Error for SExprError {}

impl<'src> NomParseError<Span<'src>> for SExprError
{
	fn from_error_kind(input: Span<'src>, kind: ErrorKind) -> Self
	{
		Self::Syntax {
			description: format!("nom::{:?}", kind),
			location: SExprLocation::of(input)
		}
	}

	fn append(_input: Span<'src>, _kind: ErrorKind, other: Self) -> Self
	{
		other
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Parser combinators.                             //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text, as threaded through the `nom` combinators.
type Span<'src> = LocatedSpan<&'src str>;

/// The result type of every combinator in this module.
type SExprResult<'src, T> = IResult<Span<'src>, T, SExprError>;

/// Advance past any leading ASCII whitespace. Always succeeds; the returned
/// span points at the first non-whitespace character (or at end of input).
///
/// # Parameters
/// - `input`: The input text to advance over.
///
/// # Returns
/// The remaining input after skipping whitespace.
///
/// # Errors
/// Never fails.
fn skip_ws(input: Span<'_>) -> SExprResult<'_, ()>
{
	let (input, _) = take_while(|c: char| c.is_ascii_whitespace())(input)?;
	Ok((input, ()))
}

/// Skip leading whitespace and consume the given expected character. Emits a
/// typed [`SExprError::ExpectedChar`] on mismatch — distinguishing "found a
/// different character" from "found end of input" so the caller can report
/// cleanly.
///
/// # Parameters
/// - `expected`: The character that must appear next (after whitespace).
///
/// # Returns
/// A combinator that matches `expected`.
///
/// # Errors
/// - [`SExprError::ExpectedChar`] if the next character does not match
///   `expected`, or if end of input is reached first.
fn expect_char<'src>(
	expected: char
) -> impl FnMut(Span<'src>) -> SExprResult<'src, ()>
{
	move |input| {
		let (input, _) = skip_ws(input)?;
		match input.fragment().chars().next()
		{
			Some(c) if c == expected =>
			{
				let len = c.len_utf8();
				Ok((input.take_from(len), ()))
			},
			Some(c) => Err(nom::Err::Failure(SExprError::ExpectedChar {
				expected,
				found: Some(c),
				location: SExprLocation::of(input)
			})),
			None => Err(nom::Err::Failure(SExprError::ExpectedChar {
				expected,
				found: None,
				location: SExprLocation::of(input)
			}))
		}
	}
}

/// Skip leading whitespace and consume a bare word: a contiguous run of
/// non-whitespace, non-delimiter characters. Delimiters are `(`, `)`, `[`, and
/// `]`. Emits a typed [`SExprError::ExpectedWord`] when the cursor sits on a
/// delimiter or at end of input.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// The span covering the word, from which callers may extract either the
/// fragment or the [`SExprLocation`] at which it began.
///
/// # Errors
/// - [`SExprError::ExpectedWord`] if no word characters are available at the
///   current position.
fn read_word(input: Span<'_>) -> SExprResult<'_, Span<'_>>
{
	let (input, _) = skip_ws(input)?;
	let mark = input;
	let (rest, word) = take_while(|c: char| {
		!c.is_ascii_whitespace() && !matches!(c, '(' | ')' | '[' | ']')
	})(input)?;
	if word.fragment().is_empty()
	{
		Err(nom::Err::Failure(SExprError::ExpectedWord {
			location: SExprLocation::of(mark)
		}))
	}
	else
	{
		Ok((rest, word))
	}
}

/// Skip leading whitespace and consume an identifier: either a bare word or a
/// double-quoted string. Quoted strings may contain whitespace and delimiter
/// characters.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and a slice of the original input covering the
/// identifier, with surrounding quotes stripped when present.
///
/// # Errors
/// - [`SExprError::UnterminatedQuotedIdent`] if a quoted identifier is opened
///   but never closed.
/// - [`SExprError::ExpectedWord`] if a bare word reader fails at the current
///   position.
fn read_ident(input: Span<'_>) -> SExprResult<'_, &str>
{
	let (input, _) = skip_ws(input)?;
	match input.fragment().chars().next()
	{
		Some('"') =>
		{
			let mark = input;
			let after_quote = input.take_from(1);
			match after_quote.fragment().find('"')
			{
				Some(pos) =>
				{
					let ident = after_quote.take(pos);
					let rest = after_quote.take_from(pos + 1);
					Ok((rest, *ident.fragment()))
				},
				None => Err(nom::Err::Failure(
					SExprError::UnterminatedQuotedIdent {
						location: SExprLocation::of(mark)
					}
				))
			}
		},
		_ =>
		{
			let (rest, word) = read_word(input)?;
			Ok((rest, *word.fragment()))
		}
	}
}

/// Skip leading whitespace and consume a signed integer literal.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed [`i32`] value.
///
/// # Errors
/// - [`SExprError::ExpectedWord`] if no word is available.
/// - [`SExprError::InvalidInteger`] if the word does not parse as an `i32`
///   (including overflow/underflow).
fn read_integer(input: Span<'_>) -> SExprResult<'_, i32>
{
	let (rest, word) = read_word(input)?;
	match word.fragment().parse::<i32>()
	{
		Ok(value) => Ok((rest, value)),
		Err(e) => Err(nom::Err::Failure(SExprError::InvalidInteger {
			text: word.fragment().to_string(),
			reason: e.to_string(),
			location: SExprLocation::of(word)
		}))
	}
}

/// Skip leading whitespace and consume a non-negative integer byte offset, as
/// appears inside a `^[start end]` span-metadata prefix.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed byte offset.
///
/// # Errors
/// - [`SExprError::ExpectedWord`] if no word is available.
/// - [`SExprError::InvalidByteOffset`] if the word does not parse as a
///   non-negative integer.
fn read_usize(input: Span<'_>) -> SExprResult<'_, usize>
{
	let (rest, word) = read_word(input)?;
	match word.fragment().parse::<usize>()
	{
		Ok(value) => Ok((rest, value)),
		Err(e) => Err(nom::Err::Failure(SExprError::InvalidByteOffset {
			text: word.fragment().to_string(),
			reason: e.to_string(),
			location: SExprLocation::of(word)
		}))
	}
}

/// Skip leading whitespace and consume an optional `^[start end]`
/// span-metadata prefix. Only the bracket-vector shape is recognized — an
/// unrecognized shape (e.g., `^{...}`, `^symbol`) is an explicit error, not
/// silently skipped.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed [`SourceSpan`], or
/// [`SourceSpan::default`] (synthetic) when no prefix is present at the current
/// position.
///
/// # Errors
/// - [`SExprError::UnrecognizedSpanShape`] if `^` is followed by anything other
///   than `[` (or end of input).
/// - [`SExprError::InvalidByteOffset`] if either integer inside the brackets is
///   malformed.
/// - [`SExprError::ExpectedChar`] if the closing `]` is missing.
/// - [`SExprError::InvertedSpan`] if the resulting span has `start > end`.
fn read_span_prefix(input: Span<'_>) -> SExprResult<'_, SourceSpan>
{
	let (input, _) = skip_ws(input)?;
	match input.fragment().chars().next()
	{
		Some('^') =>
		{
			let caret_mark = input;
			let after_caret = input.take_from(1);
			match after_caret.fragment().chars().next()
			{
				Some('[') =>
				{
					let after_bracket = after_caret.take_from(1);
					let (input, start) = read_usize(after_bracket)?;
					let (input, end) = read_usize(input)?;
					let (input, _) = expect_char(']')(input)?;
					if start > end
					{
						Err(nom::Err::Failure(SExprError::InvertedSpan {
							start,
							end,
							location: SExprLocation::of(caret_mark)
						}))
					}
					else
					{
						Ok((input, SourceSpan { start, end }))
					}
				},
				Some(c) =>
				{
					Err(nom::Err::Failure(SExprError::UnrecognizedSpanShape {
						found: Some(c),
						location: SExprLocation::of(after_caret)
					}))
				},
				None =>
				{
					Err(nom::Err::Failure(SExprError::UnrecognizedSpanShape {
						found: None,
						location: SExprLocation::of(after_caret)
					}))
				},
			}
		},
		_ => Ok((input, SourceSpan::default()))
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Validation helpers.                             //
////////////////////////////////////////////////////////////////////////////////

/// Validate that `child` is contained within `parent`, emitting a typed
/// [`SExprError::ChildSpanEscapesParent`] on violation.
///
/// # Parameters
/// - `parent`: The enclosing form's source span.
/// - `child`: The child form's source span.
/// - `at`: The input position to attribute any error to (typically the position
///   at which the child began reading).
///
/// # Returns
/// `Ok(())` if either operand is [synthetic](SourceSpan::SYNTHETIC), or if
/// `parent.start <= child.start && child.end <= parent.end`.
///
/// # Errors
/// - [`SExprError::ChildSpanEscapesParent`] if both operands are non-synthetic
///   and `child` escapes `parent` on either boundary.
fn validate_containment(
	parent: SourceSpan,
	child: SourceSpan,
	at: Span<'_>
) -> Result<(), nom::Err<SExprError>>
{
	if parent == SourceSpan::SYNTHETIC || child == SourceSpan::SYNTHETIC
	{
		return Ok(())
	}
	if parent.start <= child.start && child.end <= parent.end
	{
		Ok(())
	}
	else
	{
		Err(nom::Err::Failure(SExprError::ChildSpanEscapesParent {
			child,
			parent,
			location: SExprLocation::of(at)
		}))
	}
}

/// Validate that `next` appears strictly after `prev` in source order, emitting
/// a typed [`SExprError::SiblingSpanOutOfOrder`] on violation.
///
/// # Parameters
/// - `prev`: The previous sibling's source span.
/// - `next`: The current sibling's source span.
/// - `at`: The input position to attribute any error to (typically the position
///   at which `next` began reading).
///
/// # Returns
/// `Ok(())` if either operand is [synthetic](SourceSpan::SYNTHETIC), or if
/// `prev.end <= next.start`.
///
/// # Errors
/// - [`SExprError::SiblingSpanOutOfOrder`] if both operands are non-synthetic
///   and `next` overlaps or precedes `prev`.
fn validate_sibling_order(
	prev: SourceSpan,
	next: SourceSpan,
	at: Span<'_>
) -> Result<(), nom::Err<SExprError>>
{
	if prev == SourceSpan::SYNTHETIC || next == SourceSpan::SYNTHETIC
	{
		return Ok(())
	}
	if prev.end <= next.start
	{
		Ok(())
	}
	else
	{
		Err(nom::Err::Failure(SExprError::SiblingSpanOutOfOrder {
			prev,
			next,
			location: SExprLocation::of(at)
		}))
	}
}

////////////////////////////////////////////////////////////////////////////////
//                         Grammar-level combinators.                         //
////////////////////////////////////////////////////////////////////////////////

/// Read the complete top-level form: an optional `^[start end]` span prefix
/// followed by `(function params body)`. Trailing whitespace after the closing
/// parenthesis is consumed; the caller is responsible for verifying that no
/// input remains.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed [`Function`].
///
/// # Errors
/// - [`SExprError`] if the input is not a well-formed `(function params body)`
///   s-expression or if any span validation fails.
fn read_function(input: Span<'_>) -> SExprResult<'_, Function<'_>>
{
	let (input, span) = read_span_prefix(input)?;
	let (input, _) = expect_char('(')(input)?;
	let (input, _) = skip_ws(input)?;
	let kw_mark = input;
	let (input, kw) = read_word(input)?;
	let keyword = *kw.fragment();
	if keyword != "function"
	{
		return Err(nom::Err::Failure(SExprError::ExpectedTopLevelFunction {
			found: keyword.to_string(),
			location: SExprLocation::of(kw_mark)
		}))
	}
	let (input, (parameters, last_param_span)) = read_params(input, span)?;
	let (input, body) = read_child(input, span, last_param_span)?;
	let (input, _) = expect_char(')')(input)?;
	let (input, _) = skip_ws(input)?;
	Ok((
		input,
		Function {
			parameters,
			body,
			span
		}
	))
}

/// Read a parameter list: `[` (span-prefix? ident)* `]`. Each parameter may be
/// preceded by a `^[start end]` span-metadata prefix.
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `parent`: The enclosing function's span, for containment validation.
///
/// # Returns
/// A triple of the remaining input, the parameter list (or `None` if empty),
/// and the span of the last parameter (or [`SourceSpan::default`] if no
/// parameters were read). The latter is threaded into the body's
/// sibling-ordering check.
///
/// # Errors
/// - [`SExprError`] if the opening `[`, any identifier, or the closing `]` is
///   malformed, or if any parameter's span violates containment or
///   sibling-ordering constraints.
fn read_params<'src>(
	input: Span<'src>,
	parent: SourceSpan
) -> SExprResult<'src, (Option<Vec<Parameter<'src>>>, SourceSpan)>
{
	let (input, _) = expect_char('[')(input)?;
	let mut params: Vec<Parameter<'src>> = Vec::new();
	let mut prev_sibling = SourceSpan::default();
	let mut cur = input;
	loop
	{
		let (next, _) = skip_ws(cur)?;
		match next.fragment().chars().next()
		{
			Some(']') =>
			{
				cur = next.take_from(1);
				break
			},
			_ =>
			{
				let mark = next;
				let (next, span) = read_span_prefix(next)?;
				let (next, name) = read_ident(next)?;
				validate_containment(parent, span, mark)?;
				validate_sibling_order(prev_sibling, span, mark)?;
				prev_sibling = span;
				params.push(Parameter { name, span });
				cur = next;
			}
		}
	}
	let result = if params.is_empty()
	{
		None
	}
	else
	{
		Some(params)
	};
	Ok((cur, (result, prev_sibling)))
}

/// Read a custom faces list: `[` integer+ `]`. Faces lists must contain at
/// least one face.
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed list of face values.
///
/// # Errors
/// - [`SExprError`] if the opening `[`, the closing `]`, or any integer is
///   malformed.
/// - [`SExprError::FacelessCustomDice`] if the list contains no faces.
fn read_faces(input: Span<'_>) -> SExprResult<'_, Vec<i32>>
{
	let (input, _) = expect_char('[')(input)?;
	let mut faces = Vec::new();
	let mut cur = input;
	loop
	{
		let (next, _) = skip_ws(cur)?;
		match next.fragment().chars().next()
		{
			Some(']') =>
			{
				cur = next.take_from(1);
				break
			},
			_ =>
			{
				let (next, value) = read_integer(next)?;
				faces.push(value);
				cur = next;
			}
		}
	}
	if faces.is_empty()
	{
		Err(nom::Err::Failure(SExprError::FacelessCustomDice {
			location: SExprLocation::of(cur)
		}))
	}
	else
	{
		Ok((cur, faces))
	}
}

/// Read a subexpression and validate its span against the enclosing `parent`
/// and the `prev_sibling` (if any).
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `parent`: The enclosing compound form's span, for containment validation.
/// - `prev_sibling`: The span of the immediately preceding sibling, for
///   sibling-order validation. Pass [`SourceSpan::default`] when no prior
///   sibling exists; the check is a no-op in that case.
///
/// # Returns
/// A pair of the remaining input and the parsed subexpression, whose span has
/// been validated against `parent` and `prev_sibling`.
///
/// # Errors
/// - [`SExprError`] if the subexpression fails to parse, or if its span
///   violates the containment or sibling-ordering constraints.
fn read_child<'src>(
	input: Span<'src>,
	parent: SourceSpan,
	prev_sibling: SourceSpan
) -> SExprResult<'src, Expression<'src>>
{
	let (input, _) = skip_ws(input)?;
	let mark = input;
	let (input, expr) = read_expr(input)?;
	let child = expr.span();
	validate_containment(parent, child, mark)?;
	validate_sibling_order(prev_sibling, child, mark)?;
	Ok((input, expr))
}

/// Read the dice-expression first argument of a `drop-lowest`/`drop-highest`
/// form. Expects the child to parse to a [`DiceExpression`], not a general
/// [`Expression`].
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `parent`: The enclosing drop-lowest/drop-highest form's span.
/// - `prev_sibling`: The span of the immediately preceding sibling, or
///   [`SourceSpan::default`] when no prior sibling exists.
///
/// # Returns
/// A pair of the remaining input and the parsed dice expression.
///
/// # Errors
/// - [`SExprError`] if the subexpression fails to parse, if its span violates
///   containment or sibling-ordering constraints, or if it does not parse to a
///   [`DiceExpression`].
fn read_dice_child<'src>(
	input: Span<'src>,
	parent: SourceSpan,
	prev_sibling: SourceSpan
) -> SExprResult<'src, DiceExpression<'src>>
{
	let mark = input;
	let (input, expr) = read_child(input, parent, prev_sibling)?;
	match expr
	{
		Expression::Dice(d) => Ok((input, d)),
		_ => Err(nom::Err::Failure(SExprError::ExpectedDiceExpression {
			location: SExprLocation::of(mark)
		}))
	}
}

/// Read the optional drop-amount argument of a `drop-lowest` or
/// `drop-highest` form. The drop amount is present iff another subexpression
/// follows the dice expression before the closing parenthesis of the form.
/// Its absence represents the parser-originated `drop: None` case (implicit
/// single-die drop), so the format faithfully distinguishes `(drop-lowest d)`
/// from `(drop-lowest d 1)`.
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `parent`: The enclosing drop-lowest/drop-highest form's span.
/// - `prev_sibling`: The span of the dice expression that preceded this
///   position.
///
/// # Returns
/// A pair of the remaining input and `Some(expr)` if a drop-amount
/// subexpression is present, or `None` if the form closes immediately after
/// its dice expression.
///
/// # Errors
/// - [`SExprError`] if a subexpression is present but malformed, or if its span
///   violates containment or sibling-ordering constraints.
fn read_optional_drop<'src>(
	input: Span<'src>,
	parent: SourceSpan,
	prev_sibling: SourceSpan
) -> SExprResult<'src, Option<Box<Expression<'src>>>>
{
	let (probe, _) = skip_ws(input)?;
	if probe.fragment().starts_with(')')
	{
		return Ok((input, None))
	}
	let (input, drop) = read_child(input, parent, prev_sibling)?;
	Ok((input, Some(Box::new(drop))))
}

/// Read two subexpressions — enclosed by `parent` — and combine them with the
/// given constructor. Each child is validated for containment in `parent` and
/// for sibling ordering.
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `parent`: The enclosing binary form's span.
/// - `f`: The constructor that combines the two parsed subexpressions into a
///   single [`Expression`].
///
/// # Returns
/// A pair of the remaining input and the expression produced by `f`.
///
/// # Errors
/// - [`SExprError`] if either subexpression fails to parse, or if either span
///   violates containment or sibling-ordering constraints.
fn read_binary<'src, F>(
	input: Span<'src>,
	parent: SourceSpan,
	f: F
) -> SExprResult<'src, Expression<'src>>
where
	F: FnOnce(Expression<'src>, Expression<'src>) -> Expression<'src>
{
	let (input, left) = read_child(input, parent, SourceSpan::default())?;
	let left_span = left.span();
	let (input, right) = read_child(input, parent, left_span)?;
	Ok((input, f(left, right)))
}

/// Read an expression, optionally preceded by a `^[start end]` span prefix. The
/// prefix, when present, populates the resulting node's span; when absent, the
/// span defaults to [`SourceSpan::default`].
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed expression.
///
/// # Errors
/// - [`SExprError`] if the expression is malformed, if any child's span
///   violates containment or sibling-ordering constraints, or if an unexpected
///   keyword or end of input is encountered.
fn read_expr(input: Span<'_>) -> SExprResult<'_, Expression<'_>>
{
	let (input, span) = read_span_prefix(input)?;
	let (input, _) = skip_ws(input)?;
	match input.fragment().chars().next()
	{
		Some('(') =>
		{
			let input = input.take_from(1);
			let (input, _) = skip_ws(input)?;
			let kw_mark = input;
			let (input, kw) = read_word(input)?;
			let keyword = *kw.fragment();
			let (input, expr) = match keyword
			{
				"add" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Add(Add {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"sub" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"mul" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"div" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Div(Div {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"mod" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"exp" => read_binary(input, span, |l, r| {
					Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
						left: Box::new(l),
						right: Box::new(r),
						span
					}))
				}),
				"neg" =>
				{
					let (input, operand) =
						read_child(input, span, SourceSpan::default())?;
					Ok((
						input,
						Expression::Arithmetic(ArithmeticExpression::Neg(
							Neg {
								operand: Box::new(operand),
								span
							}
						))
					))
				},
				"group" =>
				{
					let (input, inner) =
						read_child(input, span, SourceSpan::default())?;
					Ok((
						input,
						Expression::Group(Group {
							expression: Box::new(inner),
							span
						})
					))
				},
				"standard-dice" =>
				{
					let (input, count) =
						read_child(input, span, SourceSpan::default())?;
					let count_span = count.span();
					let (input, faces) = read_child(input, span, count_span)?;
					Ok((
						input,
						Expression::Dice(DiceExpression::Standard(
							StandardDice {
								count: Box::new(count),
								faces: Box::new(faces),
								span
							}
						))
					))
				},
				"custom-dice" =>
				{
					let (input, count) =
						read_child(input, span, SourceSpan::default())?;
					let (input, faces) = read_faces(input)?;
					Ok((
						input,
						Expression::Dice(DiceExpression::Custom(CustomDice {
							count: Box::new(count),
							faces,
							span
						}))
					))
				},
				"drop-lowest" =>
				{
					let (input, dice) =
						read_dice_child(input, span, SourceSpan::default())?;
					let dice_span = dice.span();
					let (input, drop) =
						read_optional_drop(input, span, dice_span)?;
					Ok((
						input,
						Expression::Dice(DiceExpression::DropLowest(
							DropLowest {
								dice: Box::new(dice),
								drop,
								span
							}
						))
					))
				},
				"drop-highest" =>
				{
					let (input, dice) =
						read_dice_child(input, span, SourceSpan::default())?;
					let dice_span = dice.span();
					let (input, drop) =
						read_optional_drop(input, span, dice_span)?;
					Ok((
						input,
						Expression::Dice(DiceExpression::DropHighest(
							DropHighest {
								dice: Box::new(dice),
								drop,
								span
							}
						))
					))
				},
				"range" =>
				{
					let (input, start) =
						read_child(input, span, SourceSpan::default())?;
					let start_span = start.span();
					let (input, end) = read_child(input, span, start_span)?;
					Ok((
						input,
						Expression::Range(Range {
							start: Box::new(start),
							end: Box::new(end),
							span
						})
					))
				},
				"function" =>
				{
					Err(nom::Err::Failure(SExprError::NestedFunctionKeyword {
						location: SExprLocation::of(kw_mark)
					}))
				},
				_ => Err(nom::Err::Failure(SExprError::UnknownKeyword {
					keyword: keyword.to_string(),
					location: SExprLocation::of(kw_mark)
				}))
			}?;
			let (input, _) = expect_char(')')(input)?;
			Ok((input, expr))
		},
		Some(c) if c == '-' || c.is_ascii_digit() =>
		{
			let (input, value) = read_integer(input)?;
			Ok((input, Expression::Constant(Constant { value, span })))
		},
		Some(_) =>
		{
			let (input, name) = read_ident(input)?;
			Ok((input, Expression::Variable(Variable { name, span })))
		},
		None => Err(nom::Err::Failure(SExprError::ExpectedExpression {
			location: SExprLocation::of(input)
		}))
	}
}
