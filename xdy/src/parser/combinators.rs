//! # Parser combinators
//!
//! The `xDy` parser comprises a large set of combinators that correspond to the
//! production rules of the language, such that each combinator recognizes a
//! specific production rule. Direct usage of the combinators is not necessary
//! for most uses of `xDy`. Even if you need direct access to the abstract
//! syntax tree (AST), use [parse](crate::Parser::parse) instead. For typical
//! usage of `xDy`, try [compile](crate::compile) or
//! [evaluate](crate::evaluate). Nonetheless, the combinators are public and may
//! be used directly for advanced use cases.
//!
//! The combinators for the recursive productions of the grammar do not recurse.
//! Each delegates to a shared engine that parses its production with an
//! explicit stack, so any of them can parse arbitrarily deep input. They
//! remain ordinary `nom` parsers, and compose with other `nom` combinators as
//! before.
//!
//! All combinators expect the input to be free of leading whitespace.

use std::{borrow::Cow, num::IntErrorKind};

use nom::{
	IResult, Input, Parser,
	branch::alt,
	bytes::complete::{take_while, take_while_m_n, take_while1},
	character::complete::{anychar, char, digit1, multispace0, one_of},
	combinator::{cut, eof, fail, map, opt, recognize},
	error::{ErrorKind, ParseError as NomParseError, context},
	multi::{separated_list0, separated_list1},
	sequence::{pair, preceded, terminated}
};
use nom_locate::LocatedSpan;

use crate::{
	ast::{
		Binding, Constant, CustomDice, DiceExpression, Expression, Function,
		Group, Parameter, Range, StandardDice, Variable
	},
	parser::NomErrorKind,
	span::SourceSpan
};

use super::{
	CLOSING_BRACE_CONTEXT, CLOSING_BRACKET_CONTEXT, CONSTANT_CONTEXT,
	IDENTIFIER_CONTEXT, NEXT_PARAMETER_CONTEXT, PARAMETER_CONTEXT, ParseError,
	engine::{Goal, run}
};

////////////////////////////////////////////////////////////////////////////////
//                                Combinators.                                //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
pub type Span<'src> = LocatedSpan<&'src str>;

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
pub fn function(input: Span) -> IResult<Span, Function, ParseError>
{
	run(Goal::Function, input)
		.map(|(rest, value)| (rest, value.into_function()))
}

/// Parse a list of formal parameters, without leading whitespace. Each
/// [parameter] is a braced name, and a `:` ends the list. A lone braced name
/// without a `:` after it begins the body instead, as a variable or a binding,
/// so the combinator consumes nothing and answers no parameters.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed list of formal parameters.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parameters(
	input: Span<'_>
) -> IResult<Span<'_>, Option<Vec<Parameter<'_>>>, ParseError<'_>>
{
	let start = input;
	let (input, parameters) = separated_list0(
		preceded(multispace0, char(',')),
		preceded(multispace0, parameter)
	)
	.parse_complete(input)?;
	// When we discover a trailing comma, we want to set the expectation that
	// another parameter should follow (but doesn't).
	let (input, _) = match terminated(
		preceded(multispace0, char(',')),
		context(PARAMETER_CONTEXT, fail::<_, (), ParseError>())
	)
	.parse_complete(input)
	{
		Ok((input, _)) => (input, ()),
		Err(err) => match err
		{
			nom::Err::Error(e) | nom::Err::Failure(e) => {
				if matches!(e.errors[0].1, NomErrorKind::Nom(ErrorKind::Fail))
				{
					Err(nom::Err::Failure(e))
				}
				else
				{
					Ok((input, ()))
				}
			}?,
			nom::Err::Incomplete(_) => unreachable!()
		}
	};
	// A lone braced name without a `:` after it is no parameter at all, but a
	// variable or binding at the start of the body, which a `,` or `:` can
	// never follow.
	let colon: IResult<Span, char, ParseError> =
		preceded(multispace0, char(':')).parse_complete(input);
	match parameters.len()
	{
		0 => Ok((input, None)),
		1 if colon.is_err() => Ok((start, None)),
		_ =>
		{
			let (input, _) = preceded(
				multispace0,
				context(NEXT_PARAMETER_CONTEXT, char(':'))
			)
			.parse_complete(input)?;
			Ok((
				input,
				Some(
					parameters
						.iter()
						.map(|p| Parameter {
							name: canonical_name(p.fragment()),
							span: SourceSpan {
								start: p.location_offset(),
								end: p.location_offset() + p.fragment().len()
							}
						})
						.collect()
				)
			))
		}
	}
}

/// Parse a formal parameter, without leading whitespace. A parameter is a
/// [braced name](braced_name), as a [variable] is, but it fails recoverably
/// wherever the braced name fails, since the body, which may begin with a
/// variable, follows wherever the parameters do not.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The name of the parsed formal parameter, without its braces.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parameter(input: Span) -> IResult<Span, Span, ParseError>
{
	braced_name(input).map_err(|e| match e
	{
		nom::Err::Failure(e) => nom::Err::Error(e),
		e => e
	})
}

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
pub fn expression(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::AddSub, input)
		.map(|(rest, value)| (rest, value.into_expression()))
}

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
pub fn add_sub(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::AddSub, input)
		.map(|(rest, value)| (rest, value.into_expression()))
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
pub fn mul_div_mod(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::MulDivMod, input)
		.map(|(rest, value)| (rest, value.into_expression()))
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
pub fn unary(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Unary, input).map(|(rest, value)| (rest, value.into_expression()))
}

/// Parse a negated constant, producing `Constant(-N)` directly rather than
/// `Neg(Constant(N))`. This ensures that negative constants always appear
/// as a single [`Constant`] node in the AST, regardless of magnitude —
/// including `i32::MIN`, which cannot be represented as the negation of a
/// positive `i32`.
///
/// When the digits are followed by a dice operator (`d`/`D`) or the
/// exponentiation operator (`^`), this combinator bails so that `unary`'s
/// general negation path handles the expression correctly:
/// - `-3D6` means `-(3D6)`, not `(-3)D6`
/// - `-2^3` means `-(2^3)`, not `(-2)^3`
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed constant expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input does not match a negated constant.
pub(super) fn negative_constant(
	input: Span
) -> IResult<Span, Expression, ParseError>
{
	let start = input.location_offset();
	let (after_sign, _) = char('-')(input)?;
	let (after_ws, _) = multispace0(after_sign)?;
	let (remaining, digits) = digit1(after_ws)?;
	// Bail if followed by a dice operator or exponentiation operator, so
	// that the general negation path handles these expressions:
	// - `-3D6` → `Neg(Dice(…))`
	// - `-2^3` → `Neg(Exp(…))`
	let peeked = remaining.fragment().trim_start_matches(is_token_space);
	if peeked.starts_with('d')
		|| peeked.starts_with('D')
		|| peeked.starts_with('^')
	{
		return Err(nom::Err::Error(NomParseError::from_error_kind(
			input,
			ErrorKind::Digit
		)));
	}
	let end = remaining.location_offset();
	// Parse the complete signed constant, including the leading `-`. The
	// `recognize` above only yields `-<digits>`, so the sole parse failure
	// is `NegOverflow`; no guard is needed to disambiguate error kinds and
	// saturating at `i32::MIN` captures the intent directly.
	let text = format!("-{}", digits.fragment());
	let value = text.parse::<i32>().unwrap_or(i32::MIN);
	Ok((
		remaining,
		Expression::Constant(Constant {
			value,
			span: SourceSpan { start, end }
		})
	))
}

/// Parse an exponentiation expression, without leading whitespace.
/// Exponentiation binds tighter than unary negation, so `-a^2` is `-(a^2)`, not
/// `(-a)^2`.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn exponent(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Exponent, input)
		.map(|(rest, value)| (rest, value.into_expression()))
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
pub fn primary(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Primary, input)
		.map(|(rest, value)| (rest, value.into_expression()))
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
pub fn group(input: Span) -> IResult<Span, Group, ParseError>
{
	run(Goal::Group, input).map(|(rest, value)| (rest, value.into_group()))
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
pub fn variable(input: Span) -> IResult<Span, Variable, ParseError>
{
	let start = input.location_offset();
	let (input, name) = braced_name(input)?;
	let end = input.location_offset();
	Ok((
		input,
		Variable {
			name: canonical_name(name.fragment()),
			span: SourceSpan { start, end }
		}
	))
}

/// Parse a name delimited by braces, without leading whitespace: `{`, then
/// `cut(preceded(name_space0, context(IDENTIFIER_CONTEXT, identifier)))`, then
/// `cut(preceded(name_space0, context(CLOSING_BRACE_CONTEXT, char('}'))))`.
/// Every name in the language is braced, whether it names a [formal
/// parameter](parameter), a [variable], or a [local binding](binding), so no
/// name can run into the text around it.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The name, without its braces, exactly as written: [`canonical_name`]
/// canonicalizes it.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed, unrecoverably after
///   the `{`.
pub fn braced_name(input: Span) -> IResult<Span, Span, ParseError>
{
	let (input, _) = char('{').parse_complete(input)?;
	let (input, name) = cut(preceded(
		name_space0,
		context(IDENTIFIER_CONTEXT, identifier)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		name_space0,
		context(CLOSING_BRACE_CONTEXT, char('}'))
	))
	.parse_complete(input)?;
	Ok((input, name))
}

/// Answer whether `c` is whitespace that may separate tokens, exactly as
/// [`multispace0`], which the grammar uses between tokens, admits: U+0020
/// SPACE, the tab, the line feed, or the carriage return. Other Unicode
/// whitespace, e.g., U+00A0 NO-BREAK SPACE, may occur only inside the braces
/// of a [braced name](braced_name), where [`name_space0`] and [identifier]
/// admit it.
///
/// Whatever scans the source between tokens, as the
/// [diagnostics](crate::diagnostics) do, must skip whitespace by this
/// predicate, lest it step over a character on which the parser stops.
///
/// # Parameters
/// - `c`: The character to classify.
///
/// # Returns
/// `true` if `c` may separate tokens; `false` otherwise.
pub(crate) fn is_token_space(c: char) -> bool
{
	matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// Parse zero or more whitespace characters (per [`char::is_whitespace`])
/// inside the braces of a [braced name](braced_name), around its [identifier].
/// Unlike [`multispace0`], which the rest of the grammar uses, this admits
/// every Unicode whitespace character, just as an identifier does.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed whitespace.
///
/// # Errors
/// Never.
pub(crate) fn name_space0(input: Span) -> IResult<Span, Span, ParseError>
{
	take_while(char::is_whitespace).parse_complete(input)
}

/// Parse a local binding, without leading whitespace. The syntax is
/// `{name}@(expr)`, where `{name}` is a [braced name](braced_name) and `expr`
/// is any [expression]. The bound name is written just as a [variable
/// reference](variable) to it is, but followed by `@`.
///
/// The combinator commits (via [`cut`]) once it has consumed the `@` operator,
/// so a braced name followed by anything other than `@` backtracks cleanly and
/// lets the enclosing [`alt`] try another alternative. Within the engine, the
/// alternatives that read a variable read a binding too, by looking for the
/// `@` after the braced name, so that they never read the name twice.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed local binding.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn binding(input: Span) -> IResult<Span, Binding, ParseError>
{
	run(Goal::Binding, input).map(|(rest, value)| (rest, value.into_binding()))
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
pub fn range(input: Span) -> IResult<Span, Range, ParseError>
{
	run(Goal::Range, input).map(|(rest, value)| (rest, value.into_range()))
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
pub fn dice(input: Span) -> IResult<Span, DiceExpression, ParseError>
{
	run(Goal::Dice, input).map(|(rest, value)| (rest, value.into_dice()))
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
pub fn standard_dice(input: Span) -> IResult<Span, StandardDice, ParseError>
{
	run(Goal::StandardDice, input)
		.map(|(rest, value)| (rest, value.into_standard_dice()))
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
pub fn custom_dice(input: Span) -> IResult<Span, CustomDice, ParseError>
{
	run(Goal::CustomDice, input)
		.map(|(rest, value)| (rest, value.into_custom_dice()))
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
pub fn dice_count(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Atom, input).map(|(rest, value)| (rest, value.into_expression()))
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
pub fn standard_faces(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Atom, input).map(|(rest, value)| (rest, value.into_expression()))
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
pub fn custom_faces(input: Span) -> IResult<Span, Vec<i32>, ParseError>
{
	let (input, _) = char('[').parse_complete(input)?;
	let (input, faces) = cut(separated_list1(
		preceded(multispace0, char(',')),
		preceded(
			multispace0,
			context(CONSTANT_CONTEXT, map(constant, |c| c.value))
		)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_BRACKET_CONTEXT, char(']'))
	))
	.parse_complete(input)?;
	Ok((input, faces))
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
pub fn drop_lowest(
	input: Span<'_>
) -> IResult<Span<'_>, Option<Box<Expression<'_>>>, ParseError<'_>>
{
	run(Goal::DropLowest, input).map(|(rest, value)| (rest, value.into_drop()))
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
pub fn drop_highest(
	input: Span<'_>
) -> IResult<Span<'_>, Option<Box<Expression<'_>>>, ParseError<'_>>
{
	run(Goal::DropHighest, input).map(|(rest, value)| (rest, value.into_drop()))
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
pub fn drop_expression(input: Span) -> IResult<Span, Expression, ParseError>
{
	run(Goal::Atom, input).map(|(rest, value)| (rest, value.into_expression()))
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
pub fn constant(input: Span) -> IResult<Span, Constant, ParseError>
{
	let (input, recognized) =
		recognize(pair(opt(char('-')), digit1)).parse_complete(input)?;
	// `recognize(pair(opt(char('-')), digit1))` only produces strings of
	// optional `-` followed by digits, so parsing as `i32` can only fail
	// with `PosOverflow` or `NegOverflow`. The first arm handles positive
	// overflow explicitly; any remaining `Err` must therefore be
	// negative overflow and saturates at `i32::MIN`.
	let value = match recognized.fragment().parse::<i32>()
	{
		Ok(value) => value,
		Err(e) if e.kind() == &IntErrorKind::PosOverflow => i32::MAX,
		Err(_) => i32::MIN
	};
	let start = recognized.location_offset();
	let end = start + recognized.fragment().len();
	Ok((
		input,
		Constant {
			value,
			span: SourceSpan { start, end }
		}
	))
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
pub fn d_operator(input: Span) -> IResult<Span, char, ParseError>
{
	one_of("dD")(input)
}

/// Answer whether `c` may occur in an [identifier].
///
/// An identifier admits every code point except the braces that delimit it,
/// the control characters (per [`char::is_control`]) other than whitespace,
/// the bidirectional formatting controls (U+061C, U+200E, U+200F,
/// U+202A–U+202E, and U+2066–U+2069), and the other invisible characters that
/// could make two distinct names look alike (U+00AD, U+115F, U+1160, U+180E,
/// U+200B, U+2060–U+2064, U+3164, U+FEFF, and U+FFA0). Every whitespace
/// character (per [`char::is_whitespace`]) is admitted, including the tab, the
/// line feed, and U+00A0 NO-BREAK SPACE, so that a long name may be broken
/// over lines; but [`canonical_name`] collapses every run of whitespace to a
/// single U+0020 SPACE, so a name is never distinguished by its whitespace.
/// Every other character of an identifier is visible, or else joins visible
/// characters, as U+200C ZERO WIDTH NON-JOINER and U+200D ZERO WIDTH JOINER
/// do.
///
/// This predicate is the single source of truth for the identifier character
/// set, which [`identifier`], [`is_canonical_name`], the
/// [S-expression](crate::s_expr) reader, and the assembler share.
///
/// # Parameters
/// - `c`: The character to classify.
///
/// # Returns
/// `true` if `c` may occur in an identifier; `false` otherwise.
pub fn is_identifier_char(c: char) -> bool
{
	c.is_whitespace()
		|| !(c.is_control()
			|| matches!(
				c,
				'{' | '}'
				// Bidirectional formatting controls.
				| '\u{061C}'
				| '\u{200E}'
				| '\u{200F}'
				| '\u{202A}'..='\u{202E}'
				| '\u{2066}'..='\u{2069}'
				// Other invisible characters.
				| '\u{00AD}'
				| '\u{115F}'
				| '\u{1160}'
				| '\u{180E}'
				| '\u{200B}'
				| '\u{2060}'..='\u{2064}'
				| '\u{3164}'
				| '\u{FEFF}'
				| '\u{FFA0}'
			))
}

/// Answer whether `name` is a canonical [identifier], as [`canonical_name`]
/// produces it: a nonempty run of [identifier characters](is_identifier_char)
/// whose only whitespace is single U+0020 SPACE characters between the words,
/// so that it neither begins nor ends with one. Wherever a name is written
/// without the source language's leniency about whitespace, as in the
/// [S-expression](crate::s_expr) format and the assembler, it must be
/// canonical.
///
/// # Parameters
/// - `name`: The candidate name.
///
/// # Returns
/// `true` if `name` is a canonical identifier; `false` otherwise.
pub fn is_canonical_name(name: &str) -> bool
{
	name.split(' ').all(|word| {
		!word.is_empty()
			&& word
				.chars()
				.all(|c| !c.is_whitespace() && is_identifier_char(c))
	})
}

/// Canonicalize the [identifier] `name`, exactly as written, to the name that
/// it denotes: trim it at both ends, and collapse every run of whitespace
/// (per [`char::is_whitespace`]) within it to a single U+0020 SPACE. So
/// `a b`, `a  b`, and `a`, a line feed, and `   b` all denote `a b`.
///
/// # Parameters
/// - `name`: The identifier, exactly as written.
///
/// # Returns
/// The canonical name, which [`is_canonical_name`] accepts if `name` is an
/// identifier. It borrows `name` if `name` is already canonical, as it usually
/// is, and owns a fresh string otherwise.
pub fn canonical_name(name: &str) -> Cow<'_, str>
{
	if name
		.split(' ')
		.all(|word| !word.is_empty() && !word.contains(char::is_whitespace))
	{
		return Cow::Borrowed(name);
	}
	let mut canonical = String::with_capacity(name.len());
	for word in name.split_whitespace()
	{
		if !canonical.is_empty()
		{
			canonical.push(' ');
		}
		canonical.push_str(word);
	}
	Cow::Owned(canonical)
}

/// Parse an identifier, without leading whitespace. An identifier is a run of
/// [identifier characters](is_identifier_char), which may include whitespace,
/// but neither begins nor ends with any: the trailing whitespace remains in
/// the input. Every identifier in the language is [braced](braced_name), so an
/// identifier ends at its closing brace, e.g., `{an external variable}` or
/// `{weapon: 2/3}`. The identifier is exactly as written, so it may differ
/// from the name that it denotes, which [`canonical_name`] produces.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed identifier.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn identifier(input: Span) -> IResult<Span, Span, ParseError>
{
	let (_, run) = recognize(pair(
		take_while_m_n(1, 1, |c: char| {
			!c.is_whitespace() && is_identifier_char(c)
		}),
		take_while(is_identifier_char)
	))
	.parse_complete(input)?;
	let len = run.fragment().trim_end().len();
	Ok(input.take_split(len))
}

/// Answer whether `c` may begin a [bare word](bare_word).
///
/// The permitted starting characters are any Unicode alphabetic code point
/// (per [`char::is_alphabetic`]) plus the sigils `_`, `$`, `#`, and `'`.
///
/// # Parameters
/// - `c`: The character to classify.
///
/// # Returns
/// `true` if `c` may start a bare word; `false` otherwise.
pub(crate) fn is_bare_word_start(c: char) -> bool
{
	c.is_alphabetic() || matches!(c, '_' | '$' | '#' | '\'')
}

/// Answer whether `c` may continue a [bare word](bare_word) after its first
/// character.
///
/// Every [start character](is_bare_word_start) is also a continuation
/// character. The continuation set additionally admits any Unicode numeric
/// code point (per [`char::is_numeric`]), the connectors `-` and `.`, the
/// selector characters `|`, `?`, `!`, and `~`, and inline whitespace (any
/// [`char::is_whitespace`] code point other than `\n` or `\r`).
///
/// # Parameters
/// - `c`: The character to classify.
///
/// # Returns
/// `true` if `c` may continue a bare word; `false` otherwise.
pub(crate) fn is_bare_word_continue(c: char) -> bool
{
	is_bare_word_start(c)
		|| c.is_numeric()
		|| matches!(c, '-' | '.' | '|' | '?' | '!' | '~')
		|| (c.is_whitespace() && !matches!(c, '\n' | '\r'))
}

/// Parse a bare word, without leading whitespace: text that reads as a name,
/// but lacks the braces that every [identifier] requires. The grammar has no
/// bare words; error reporting and the [diagnostics](crate::diagnostics) read
/// them, to show the offending token and to guess at a name that its author
/// forgot to brace, e.g., `an external variable` in `an external variable +
/// 1`. Unlike an identifier, a bare word must stop at the operators and
/// delimiters around it, so it admits only the narrower character set of
/// [`is_bare_word_start`] and [`is_bare_word_continue`]. Any trailing
/// [whitespace between tokens](is_token_space) remains in the input, but other
/// trailing whitespace, e.g., U+00A0 NO-BREAK SPACE, which may not separate
/// tokens, belongs to the bare word, so that a fix that braces the bare word
/// braces it too.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed bare word.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub(crate) fn bare_word(input: Span) -> IResult<Span, Span, ParseError>
{
	let (_, run) = recognize(pair(
		take_while_m_n(1, 1, is_bare_word_start),
		take_while(is_bare_word_continue)
	))
	.parse_complete(input)?;
	let len = run.fragment().trim_end_matches(is_token_space).len();
	Ok(input.take_split(len))
}

/// The characters before which the name of an unclosed variable may
/// [break](unclosed_name_ends): the operators, the closing delimiters other
/// than `}`, and the separators of formal parameters and bindings.
const NAME_BREAKS: &[char] = &[
	'+', '-', '*', '×', '/', '÷', '%', '^', ')', ']', ',', ':', '@'
];

/// Answer where the name of a variable whose `}` is missing may break. Every
/// [identifier] character but a brace may occur in a name, so the name of an
/// unclosed variable reads everything up to the next brace or the end of the
/// input, e.g., `x + 2) * 3` in `({x + 2) * 3`. More likely, its author forgot
/// the `}` just before an operator or a delimiter that follows the name. Most
/// likely, the name breaks before the first [break character](NAME_BREAKS)
/// that whitespace precedes, e.g., `{x} + 2` for `{x + 2`, and
/// `{hit-points} + 2` for `{hit-points + 2`. Less likely, it breaks before an
/// earlier break character, after the start of the name, that none precedes,
/// e.g., `{hit}-points + 2`, or `{x}@(3)` for `{x@(3)`. The answer is lexical,
/// and takes time linear in the length of the name.
///
/// # Parameters
/// - `name`: The name as written, from its first character, which is not
///   whitespace, to where the `}` was expected.
///
/// # Returns
/// The byte offsets in `name` of the ends of the broken name, the more likely
/// first, and either or both `None`. An end excludes the [whitespace between
/// tokens](is_token_space) before the break, but not other whitespace, e.g.,
/// U+00A0 NO-BREAK SPACE, which may not separate tokens, and so stays inside
/// the braces.
///
/// # Examples
/// `x + 2) * 3` breaks at 1 only, `hit-points + 2` at 10, then 3, `x+2` at 1
/// only, and `x y` not at all.
pub(crate) fn unclosed_name_ends(name: &str) -> [Option<usize>; 2]
{
	let mut unspaced = None;
	let mut previous = None;
	for (i, c) in name.char_indices()
	{
		if i > 0 && NAME_BREAKS.contains(&c)
		{
			if previous.is_some_and(char::is_whitespace)
			{
				return [
					Some(name[..i].trim_end_matches(is_token_space).len()),
					unspaced
				];
			}
			unspaced.get_or_insert(i);
		}
		previous = Some(c);
	}
	[unspaced, None]
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
pub fn alpha(input: Span) -> IResult<Span, Span, ParseError>
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
pub fn alphanumeric1(input: Span) -> IResult<Span, Span, ParseError>
{
	take_while1(|c: char| c.is_alphanumeric())(input)
}

/// Parse an arbitary token. This is used only for error reporting.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed token, or `None` if the input is empty.
pub fn token(input: Span) -> Option<Span>
{
	match alt((eof, recognize(constant), bare_word, recognize(anychar)))
		.parse_complete(input)
		.unwrap()
		.1
	{
		span if span.fragment().is_empty() => None,
		span => Some(span)
	}
}
