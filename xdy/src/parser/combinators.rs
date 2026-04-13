//! # Parser combinators
//!
//! The `xDy` parser comprises a large set of combinators that correspond to the
//! production rules of the language, such that each combinator recognizes a
//! specific production rule. Direct usage of the combinators is not necessary
//! for most uses of `xDy`. Even if you need direct access to the abstract
//! syntax tree (AST), use [parse](crate::parser::parse) instead. For typical
//! usage of `xDy`, try [compile](crate::compile) or
//! [evaluate](crate::evaluate). Nonetheless, the combinators are public and may
//! be used directly for advanced use cases.
//!
//! All combinators expect the input to be free of leading whitespace.

use std::num::IntErrorKind;

use nom::{
	IResult, Parser,
	branch::alt,
	bytes::complete::{tag, take_while_m_n, take_while1},
	character::complete::{anychar, char, digit1, multispace0, one_of},
	combinator::{cut, eof, fail, map, opt, recognize},
	error::{
		ErrorKind, FromExternalError, ParseError as NomParseError, context
	},
	multi::{fold_many0, many0, separated_list0, separated_list1},
	sequence::{pair, preceded, separated_pair, terminated}
};
use nom_locate::LocatedSpan;

use crate::{
	ast::{
		Add, ArithmeticExpression, Constant, CustomDice, DiceExpression, Div,
		DropHighest, DropLowest, Exp, Expression, Function, Group, Mod, Mul,
		Neg, Range, StandardDice, Sub, Variable
	},
	parser::NomErrorKind
};

use super::{
	CLOSING_BRACE_CONTEXT, CLOSING_BRACKET_CONTEXT, CLOSING_PAREN_CONTEXT,
	CONSTANT_CONTEXT, CUSTOM_FACES_CONTEXT, DICE_CONTEXT, DICE_COUNT_CONTEXT,
	DROP_DIRECTION_CONTEXT, DROP_EXPRESSION_CONTEXT, EXPRESSION_CONTEXT,
	FUNCTION_BODY_CONTEXT, GROUP_CONTEXT, IDENTIFIER_CONTEXT,
	NEXT_PARAMETER_CONTEXT, PARAMETER_CONTEXT, ParseError, RANGE_CONTEXT,
	RANGE_END_CONTEXT, RANGE_START_CONTEXT, RIGHT_OPERAND_CONTEXT,
	STANDARD_FACES_CONTEXT, VARIABLE_CONTEXT
};

////////////////////////////////////////////////////////////////////////////////
//                                Combinators.                                //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text.
///
/// # Lifetimes
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
	let (input, parameters) = parameters.parse_complete(input)?;
	let (input, body) =
		preceded(multispace0, context(FUNCTION_BODY_CONTEXT, expression))
			.parse_complete(input)?;
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
pub fn parameters(
	input: Span<'_>
) -> IResult<Span<'_>, Option<Vec<&str>>, ParseError<'_>>
{
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
	match parameters.is_empty()
	{
		true => Ok((input, None)),
		false =>
		{
			let (input, _) = preceded(
				multispace0,
				context(NEXT_PARAMETER_CONTEXT, char(':'))
			)
			.parse_complete(input)?;
			Ok((
				input,
				Some(parameters.iter().map(|p| *p.fragment()).collect())
			))
		}
	}
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
pub fn parameter(input: Span) -> IResult<Span, Span, ParseError>
{
	identifier(input)
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
	add_sub(input)
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
	let (input, initial) = mul_div_mod.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, one_of("+-")),
		cut(preceded(
			multispace0,
			context(RIGHT_OPERAND_CONTEXT, mul_div_mod)
		))
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
pub fn mul_div_mod(input: Span) -> IResult<Span, Expression, ParseError>
{
	let (input, initial) = unary.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, one_of("*×/÷%")),
		cut(preceded(multispace0, context(RIGHT_OPERAND_CONTEXT, unary)))
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
pub fn unary(input: Span) -> IResult<Span, Expression, ParseError>
{
	alt((
		negative_overflowed_constant,
		map(preceded(char('-'), preceded(multispace0, unary)), |expr| {
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(expr)
			}))
		}),
		preceded(multispace0, exponent)
	))
	.parse_complete(input)
}

/// Parse a negated constant whose absolute value overflows positive `i32`. This
/// handles the `i32::MIN` boundary correctly: the `constant` combinator clamps
/// positive overflow to [`i32::MAX`], which loses the ability to represent
/// [`i32::MIN`] via [`i32::saturating_neg`]. By parsing the sign and digits as
/// a unit, the combined value is parsed as a negative `i32` directly.
///
/// This combinator only activates for values that overflow positive `i32`,
/// so it does not change the AST shape for ordinary negative constants like
/// `-5` (which remain as `Neg(Constant(5))`).
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed constant expression.
///
/// # Errors
/// * [`Err`](nom::Err) if the input does not match a negated overflow constant.
fn negative_overflowed_constant(
	input: Span
) -> IResult<Span, Expression, ParseError>
{
	let (after_sign, _) = char('-')(input)?;
	let (after_ws, _) = multispace0(after_sign)?;
	let (remaining, digits) = digit1(after_ws)?;
	// Bail if followed by a dice operator, so that expressions like
	// `-2147483648D6` are parsed as `Neg(Dice(…))` rather than
	// `Constant(…)` with unparsed `D6`.
	if remaining.fragment().starts_with('d')
		|| remaining.fragment().starts_with('D')
	{
		return Err(nom::Err::Error(NomParseError::from_error_kind(
			input,
			ErrorKind::Digit
		)));
	}
	// Only activate for values that overflow positive i32; smaller values
	// should fall through to the general `Neg(Constant(…))` path.
	match digits.fragment().parse::<i32>()
	{
		Ok(_) => Err(nom::Err::Error(NomParseError::from_error_kind(
			input,
			ErrorKind::Digit
		))),
		Err(e) if e.kind() == &IntErrorKind::PosOverflow =>
		{
			let text = format!("-{}", digits.fragment());
			let value = match text.parse::<i32>()
			{
				Ok(v) => v,
				Err(e) if e.kind() == &IntErrorKind::NegOverflow => i32::MIN,
				Err(_) => unreachable!()
			};
			Ok((remaining, Expression::Constant(Constant(value))))
		},
		Err(_) => Err(nom::Err::Error(NomParseError::from_error_kind(
			input,
			ErrorKind::Digit
		)))
	}
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
	let (input, initial) = primary.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, char('^')),
		cut(preceded(multispace0, context(RIGHT_OPERAND_CONTEXT, unary)))
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
	alt((
		context(RANGE_CONTEXT, map(range, Expression::Range)),
		context(DICE_CONTEXT, map(dice, Expression::Dice)),
		context(GROUP_CONTEXT, map(group, Expression::Group)),
		context(VARIABLE_CONTEXT, map(variable, Expression::Variable)),
		context(CONSTANT_CONTEXT, map(constant, Expression::Constant))
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
pub fn group(input: Span) -> IResult<Span, Group, ParseError>
{
	let (input, _) = char('(').parse_complete(input)?;
	let (input, expression) = cut(preceded(
		multispace0,
		context(EXPRESSION_CONTEXT, expression)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_PAREN_CONTEXT, char(')'))
	))
	.parse_complete(input)?;
	Ok((
		input,
		Group {
			expression: Box::new(expression)
		}
	))
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
	let (input, _) = char('{').parse_complete(input)?;
	let (input, span) = cut(preceded(
		multispace0,
		context(IDENTIFIER_CONTEXT, identifier)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_BRACE_CONTEXT, char('}'))
	))
	.parse_complete(input)?;
	Ok((input, Variable(span.fragment())))
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
	let (input, _) = char('[').parse_complete(input)?;
	let (input, start) = cut(preceded(
		multispace0,
		context(RANGE_START_CONTEXT, expression)
	))
	.parse_complete(input)?;
	let (input, _) =
		cut(preceded(multispace0, char(':'))).parse_complete(input)?;
	let (input, end) = cut(preceded(
		multispace0,
		context(RANGE_END_CONTEXT, expression)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_BRACKET_CONTEXT, char(']'))
	))
	.parse_complete(input)?;
	Ok((
		input,
		Range {
			start: Box::new(start),
			end: Box::new(end)
		}
	))
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
	// Parse the common prefix shared by standard and custom dice.
	let (input, count) =
		context(DICE_COUNT_CONTEXT, dice_count).parse_complete(input)?;
	let (input, _) = preceded(multispace0, d_operator).parse_complete(input)?;
	// After the `d`/`D` operator, we're committed to a dice expression.
	let (input, initial_dice) = cut(preceded(
		multispace0,
		alt((
			context(
				STANDARD_FACES_CONTEXT,
				map(standard_faces, {
					let count = count.clone();
					move |faces| {
						DiceExpression::Standard(StandardDice {
							count: Box::new(count.clone()),
							faces: Box::new(faces)
						})
					}
				})
			),
			context(
				CUSTOM_FACES_CONTEXT,
				map(custom_faces, move |faces| {
					DiceExpression::Custom(CustomDice {
						count: Box::new(count.clone()),
						faces
					})
				})
			)
		))
	))
	.parse_complete(input)?;

	let (input, final_dice) = fold_many0(
		drop_clause,
		|| initial_dice.clone(),
		|acc, (direction, drop)| match direction
		{
			DropDirection::Lowest => DiceExpression::DropLowest(DropLowest {
				dice: Box::new(acc),
				drop
			}),
			DropDirection::Highest => DiceExpression::DropHighest(DropHighest {
				dice: Box::new(acc),
				drop
			})
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
pub fn standard_dice(input: Span) -> IResult<Span, StandardDice, ParseError>
{
	separated_pair(
		context(DICE_COUNT_CONTEXT, dice_count),
		preceded(multispace0, d_operator),
		preceded(multispace0, context(STANDARD_FACES_CONTEXT, standard_faces))
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
pub fn custom_dice(input: Span) -> IResult<Span, CustomDice, ParseError>
{
	separated_pair(
		context(DICE_COUNT_CONTEXT, dice_count),
		preceded(multispace0, d_operator),
		preceded(multispace0, context(CUSTOM_FACES_CONTEXT, custom_faces))
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
pub fn dice_count(input: Span) -> IResult<Span, Expression, ParseError>
{
	alt((
		context(CONSTANT_CONTEXT, map(constant, Expression::Constant)),
		context(VARIABLE_CONTEXT, map(variable, Expression::Variable)),
		context(GROUP_CONTEXT, map(group, Expression::Group))
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
pub fn standard_faces(input: Span) -> IResult<Span, Expression, ParseError>
{
	alt((
		context(CONSTANT_CONTEXT, map(constant, Expression::Constant)),
		context(VARIABLE_CONTEXT, map(variable, Expression::Variable)),
		context(GROUP_CONTEXT, map(group, Expression::Group))
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
pub fn custom_faces(input: Span) -> IResult<Span, Vec<i32>, ParseError>
{
	let (input, _) = char('[').parse_complete(input)?;
	let (input, faces) = cut(separated_list1(
		preceded(multispace0, char(',')),
		preceded(
			multispace0,
			context(CONSTANT_CONTEXT, map(constant, |c| c.0))
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

/// The direction of a drop clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DropDirection
{
	Lowest,
	Highest
}

/// Parse a drop clause (`drop lowest [N]` or `drop highest [N]`), without
/// leading whitespace.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed drop direction and optional drop count.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
fn drop_clause(
	input: Span<'_>
) -> IResult<
	Span<'_>,
	(DropDirection, Option<Box<Expression<'_>>>),
	ParseError<'_>
>
{
	let (input, _) =
		preceded(multispace0, tag("drop")).parse_complete(input)?;
	let (input, direction) = cut(preceded(
		multispace0,
		context(
			DROP_DIRECTION_CONTEXT,
			alt((
				map(tag("lowest"), |_| DropDirection::Lowest),
				map(tag("highest"), |_| DropDirection::Highest)
			))
		)
	))
	.parse_complete(input)?;
	let (input, drop) = opt(preceded(
		multispace0,
		context(DROP_EXPRESSION_CONTEXT, drop_expression)
	))
	.parse_complete(input)?;
	Ok((input, (direction, drop.map(Box::new))))
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
	let (input, (_, _, drop)) = (
		preceded(multispace0, tag("drop")),
		preceded(multispace0, tag("lowest")),
		opt(preceded(
			multispace0,
			context(DROP_EXPRESSION_CONTEXT, drop_expression)
		))
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
pub fn drop_highest(
	input: Span<'_>
) -> IResult<Span<'_>, Option<Box<Expression<'_>>>, ParseError<'_>>
{
	let (input, (_, _, drop)) = (
		preceded(multispace0, tag("drop")),
		preceded(multispace0, tag("highest")),
		opt(preceded(
			multispace0,
			context(DROP_EXPRESSION_CONTEXT, drop_expression)
		))
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
pub fn drop_expression(input: Span) -> IResult<Span, Expression, ParseError>
{
	alt((
		context(CONSTANT_CONTEXT, map(constant, Expression::Constant)),
		context(VARIABLE_CONTEXT, map(variable, Expression::Variable)),
		context(GROUP_CONTEXT, map(group, Expression::Group))
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
pub fn constant(input: Span) -> IResult<Span, Constant, ParseError>
{
	let (input, constant) =
		recognize(pair(opt(char('-')), digit1)).parse_complete(input)?;
	let constant = match constant.fragment().parse::<i32>()
	{
		Ok(value) => value,
		Err(e) if e.kind() == &IntErrorKind::PosOverflow => i32::MAX,
		Err(e) if e.kind() == &IntErrorKind::NegOverflow => i32::MIN,
		Err(e) =>
		{
			return Err(nom::Err::Failure(ParseError::from_external_error(
				input,
				ErrorKind::Digit,
				e
			)))
		},
	};
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
pub fn d_operator(input: Span) -> IResult<Span, char, ParseError>
{
	one_of("dD")(input)
}

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
pub fn identifier(input: Span) -> IResult<Span, Span, ParseError>
{
	// Identifiers may contain inline whitespace (e.g., "an external
	// variable"), but must not include trailing whitespace. We greedily
	// match the full identifier including any inline whitespace, then
	// trim trailing whitespace by rewinding the input span.
	let (remaining, span) = recognize(pair(
		alt((alpha, tag("_"))),
		many0(alt((
			alphanumeric1,
			tag("_"),
			tag("-"),
			tag("."),
			recognize(take_while_m_n(1, 1, |c: char| {
				c.is_whitespace() && !matches!(c, '\n' | '\r')
			}))
		)))
	))
	.parse_complete(input)?;
	let trimmed = span.fragment().trim_end();
	if trimmed.len() < span.fragment().len()
	{
		// Rewind the input to just after the trimmed identifier, giving back
		// the trailing whitespace to the remaining input.
		let trail = span.fragment().len() - trimmed.len();
		let new_remaining = unsafe {
			Span::new_from_raw_offset(
				remaining.location_offset() - trail,
				remaining.location_line(),
				&input.fragment()[trimmed.len()..],
				()
			)
		};
		let trimmed_span = unsafe {
			Span::new_from_raw_offset(
				span.location_offset(),
				span.location_line(),
				trimmed,
				()
			)
		};
		Ok((new_remaining, trimmed_span))
	}
	else
	{
		Ok((remaining, span))
	}
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
	match alt((eof, recognize(constant), identifier, recognize(anychar)))
		.parse_complete(input)
		.unwrap()
		.1
	{
		span if span.fragment().is_empty() => None,
		span => Some(span)
	}
}
