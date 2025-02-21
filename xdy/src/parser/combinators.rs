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

use nom::{
	IResult, Parser,
	branch::alt,
	bytes::complete::{tag, take_while_m_n, take_while1},
	character::complete::{anychar, char, digit1, multispace0, one_of},
	combinator::{eof, fail, map, opt, recognize},
	error::{ErrorKind, FromExternalError, context},
	multi::{fold_many0, many0, separated_list0, separated_list1},
	sequence::{delimited, pair, preceded, separated_pair, terminated}
};
use nom_locate::LocatedSpan;

use crate::{
	ast::{
		Add, ArithmeticExpression, Constant, CustomDice, DiceExpression, Div,
		DropHighest, DropLowest, Exp, Expression, Function, Group, Mod, Mul,
		Neg, Range, StandardDice, Sub, Variable
	},
	parser::{CUSTOM_DICE_CONTEXT, NomErrorKind, STANDARD_DICE_CONTEXT}
};

use super::{
	CONSTANT_CONTEXT, CUSTOM_FACES_CONTEXT, DICE_CONTEXT, DICE_COUNT_CONTEXT,
	DROP_EXPRESSION_CONTEXT, EXPRESSION_CONTEXT, FUNCTION_BODY_CONTEXT,
	GROUP_CONTEXT, IDENTIFIER_CONTEXT, NEXT_PARAMETER_CONTEXT, NomError,
	PARAMETER_CONTEXT, RANGE_CONTEXT, RANGE_END_CONTEXT, RANGE_START_CONTEXT,
	STANDARD_FACES_CONTEXT, VARIABLE_CONTEXT
};

////////////////////////////////////////////////////////////////////////////////
//                                Combinators.                                //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text.
pub type Span<'a> = LocatedSpan<&'a str>;

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
pub fn function(input: Span) -> IResult<Span, Function, NomError>
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
pub fn parameters(input: Span) -> IResult<Span, Option<Vec<&str>>, NomError>
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
		context(PARAMETER_CONTEXT, fail::<_, (), NomError>())
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
pub fn parameter(input: Span) -> IResult<Span, Span, NomError>
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
pub fn expression(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn add_sub(input: Span) -> IResult<Span, Expression, NomError>
{
	let (input, initial) = mul_div_mod.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, one_of("+-")),
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
pub fn mul_div_mod(input: Span) -> IResult<Span, Expression, NomError>
{
	let (input, initial) = exponent.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, one_of("*×/÷%")),
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
pub fn exponent(input: Span) -> IResult<Span, Expression, NomError>
{
	let (input, initial) = unary.parse_complete(input)?;
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
pub fn unary(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn primary(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn group(input: Span) -> IResult<Span, Group, NomError>
{
	delimited(
		char('('),
		map(
			preceded(multispace0, context(EXPRESSION_CONTEXT, expression)),
			Box::new
		),
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
pub fn variable(input: Span) -> IResult<Span, Variable, NomError>
{
	delimited(
		char('{'),
		preceded(multispace0, context(IDENTIFIER_CONTEXT, identifier)),
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
pub fn range(input: Span) -> IResult<Span, Range, NomError>
{
	delimited(
		char('['),
		separated_pair(
			preceded(multispace0, context(RANGE_START_CONTEXT, expression)),
			preceded(multispace0, char(':')),
			preceded(multispace0, context(RANGE_END_CONTEXT, expression))
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
pub fn dice(input: Span) -> IResult<Span, DiceExpression, NomError>
{
	let (input, initial_dice) = alt((
		context(
			STANDARD_DICE_CONTEXT,
			map(standard_dice, DiceExpression::Standard)
		),
		context(
			CUSTOM_DICE_CONTEXT,
			map(custom_dice, DiceExpression::Custom)
		)
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
pub fn standard_dice(input: Span) -> IResult<Span, StandardDice, NomError>
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
pub fn custom_dice(input: Span) -> IResult<Span, CustomDice, NomError>
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
pub fn dice_count(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn standard_faces(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn custom_faces(input: Span) -> IResult<Span, Vec<i32>, NomError>
{
	delimited(
		char('['),
		separated_list1(
			preceded(multispace0, char(',')),
			preceded(
				multispace0,
				context(CONSTANT_CONTEXT, map(constant, |c| c.0))
			)
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
pub fn drop_lowest(
	input: Span
) -> IResult<Span, Option<Box<Expression<'_>>>, NomError>
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
	input: Span
) -> IResult<Span, Option<Box<Expression<'_>>>, NomError>
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
pub fn drop_expression(input: Span) -> IResult<Span, Expression, NomError>
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
pub fn constant(input: Span) -> IResult<Span, Constant, NomError>
{
	let (input, constant) =
		recognize(pair(opt(char('-')), digit1)).parse_complete(input)?;
	let constant = constant.fragment().parse().map_err(|e| {
		nom::Err::Failure(NomError::from_external_error(
			input,
			ErrorKind::Digit,
			e
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
pub fn d_operator(input: Span) -> IResult<Span, char, NomError>
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
pub fn identifier(input: Span) -> IResult<Span, Span, NomError>
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
pub fn alpha(input: Span) -> IResult<Span, Span, NomError>
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
pub fn alphanumeric1(input: Span) -> IResult<Span, Span, NomError>
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
