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
//! All combinators expect the input to be free of leading whitespace.

use std::num::IntErrorKind;

use nom::{
	IResult, Parser,
	branch::alt,
	bytes::complete::{tag, take_while_m_n, take_while1},
	character::complete::{anychar, char, digit1, multispace0, one_of},
	combinator::{cut, eof, fail, map, opt, recognize},
	error::{ErrorKind, ParseError as NomParseError, context},
	multi::{fold_many0, many0, separated_list0, separated_list1},
	sequence::{pair, preceded, separated_pair, terminated}
};
use nom_locate::LocatedSpan;

use crate::{
	ast::{
		Add, ArithmeticExpression, Binding, Constant, CustomDice,
		DiceExpression, Div, DropHighest, DropLowest, Exp, Expression,
		Function, Group, Mod, Mul, Neg, Parameter, Range, StandardDice, Sub,
		Variable
	},
	parser::NomErrorKind,
	span::{SourceSpan, Spanned}
};

use super::{
	BINDING_CONTEXT, BINDING_EXPRESSION_CONTEXT, CLOSING_BRACE_CONTEXT,
	CLOSING_BRACKET_CONTEXT, CLOSING_PAREN_CONTEXT, CONSTANT_CONTEXT,
	CUSTOM_FACES_CONTEXT, DICE_CONTEXT, DICE_COUNT_CONTEXT,
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
	let start = input.location_offset();
	let (input, parameters) = parameters.parse_complete(input)?;
	let (input, body) =
		preceded(multispace0, context(FUNCTION_BODY_CONTEXT, expression))
			.parse_complete(input)?;
	let end = input.location_offset();
	Ok((
		input,
		Function {
			parameters,
			body,
			span: SourceSpan { start, end }
		}
	))
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
) -> IResult<Span<'_>, Option<Vec<Parameter<'_>>>, ParseError<'_>>
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
				Some(
					parameters
						.iter()
						.map(|p| Parameter {
							name: p.fragment(),
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

/// Parse a formal parameter, without leading whitespace.
///
/// A parameter is an [identifier](identifier), with one caveat: if the
/// identifier is immediately followed by an `@` (after optional inline
/// whitespace), it is the head of a [local binding](binding) rather than a
/// parameter declaration, so the combinator fails without consuming input. The
/// rewind is necessary because `x@(expr)` at the start of a function body would
/// otherwise be misinterpreted as a parameter `x` missing its `:` separator,
/// and the enclosing [`parameters`] combinator cannot backtrack after
/// `separated_list0` has committed to the identifier.
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
	let original = input;
	let (after_name, name) = identifier(input)?;
	let peek: IResult<Span, char, ParseError> =
		preceded(multispace0, char('@')).parse_complete(after_name);
	if peek.is_ok()
	{
		return Err(nom::Err::Error(
			<ParseError as NomParseError<Span>>::from_error_kind(
				original,
				ErrorKind::Tag
			)
		));
	}
	Ok((after_name, name))
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
		remainder.into_iter().fold(initial, |acc, (op, expr)| {
			let span = SourceSpan {
				start: acc.span().start,
				end: expr.span().end
			};
			match op
			{
				'+' => Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(acc),
					right: Box::new(expr),
					span
				})),
				'-' => Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
					left: Box::new(acc),
					right: Box::new(expr),
					span
				})),
				_ => unreachable!()
			}
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
		remainder.into_iter().fold(initial, |acc, (op, expr)| {
			let span = SourceSpan {
				start: acc.span().start,
				end: expr.span().end
			};
			match op
			{
				'*' | '×' =>
				{
					Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
						left: Box::new(acc),
						right: Box::new(expr),
						span
					}))
				},
				'/' | '÷' =>
				{
					Expression::Arithmetic(ArithmeticExpression::Div(Div {
						left: Box::new(acc),
						right: Box::new(expr),
						span
					}))
				},
				'%' => Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
					left: Box::new(acc),
					right: Box::new(expr),
					span
				})),
				_ => unreachable!()
			}
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
	alt((negative_constant, negation, preceded(multispace0, exponent)))
		.parse_complete(input)
}

/// Parse the general negation path of a [unary](unary) expression: a leading
/// `-`, optional whitespace, and a [unary](unary) operand.
///
/// Factored out of [`unary`] so that the span start offset can be captured
/// before the leading `-` is consumed without running afoul of the lifetime
/// elision rules that govern inline closures inside [`alt`](nom::branch::alt).
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed [`Neg`] expression, with its span covering the `-` through the
/// end of the operand.
///
/// # Errors
/// * [`Err`](nom::Err) if the input does not begin with `-` followed by a
///   parseable unary expression.
fn negation(
	input: Span<'_>
) -> IResult<Span<'_>, Expression<'_>, ParseError<'_>>
{
	let start = input.location_offset();
	let (input, expr) = preceded(char('-'), preceded(multispace0, unary))
		.parse_complete(input)?;
	let end = expr.span().end;
	Ok((
		input,
		Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
			operand: Box::new(expr),
			span: SourceSpan { start, end }
		}))
	))
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
fn negative_constant(input: Span) -> IResult<Span, Expression, ParseError>
{
	let start = input.location_offset();
	let (after_sign, _) = char('-')(input)?;
	let (after_ws, _) = multispace0(after_sign)?;
	let (remaining, digits) = digit1(after_ws)?;
	// Bail if followed by a dice operator or exponentiation operator, so
	// that the general negation path handles these expressions:
	// - `-3D6` → `Neg(Dice(…))`
	// - `-2^3` → `Neg(Exp(…))`
	let peeked = remaining.fragment().trim_start();
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
	let (input, initial) = primary.parse_complete(input)?;
	let (input, remainder) = many0(pair(
		preceded(multispace0, char('^')),
		cut(preceded(multispace0, context(RIGHT_OPERAND_CONTEXT, unary)))
	))
	.parse_complete(input)?;
	// Because the RHS of `^` is parsed by `unary`, which recurses into
	// `exponent`, `many0` at any single level only ever collects at most one
	// pair — the recursive call consumes all subsequent `^` operators.
	match remainder.into_iter().next()
	{
		Some((_, expr)) =>
		{
			let span = SourceSpan {
				start: initial.span().start,
				end: expr.span().end
			};
			Ok((
				input,
				Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
					left: Box::new(initial),
					right: Box::new(expr),
					span
				}))
			))
		},
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
		context(BINDING_CONTEXT, map(binding, Expression::Binding)),
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
	let start = input.location_offset();
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
	let end = input.location_offset();
	Ok((
		input,
		Group {
			expression: Box::new(expression),
			span: SourceSpan { start, end }
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
	let start = input.location_offset();
	let (input, _) = char('{').parse_complete(input)?;
	let (input, name) = cut(preceded(
		multispace0,
		context(IDENTIFIER_CONTEXT, identifier)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_BRACE_CONTEXT, char('}'))
	))
	.parse_complete(input)?;
	let end = input.location_offset();
	Ok((
		input,
		Variable {
			name: name.fragment(),
			span: SourceSpan { start, end }
		}
	))
}

/// Parse a local binding, without leading whitespace. The syntax is
/// `name@(expr)`, where `name` is an [identifier](identifier) and `expr` is any
/// [expression](expression). The bound name must appear to the left of the `@`
/// without delimiters — note that this is distinct from a [variable
/// reference](variable), which uses `{name}`.
///
/// The combinator commits (via [`cut`]) once it has consumed the `@` operator,
/// so a bare identifier followed by anything other than `@` backtracks cleanly
/// and lets the enclosing [`alt`] try another alternative.
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
	// Commit to the binding form only after seeing `identifier @`, and on
	// failure return an error at the original input position rather than at the
	// post-identifier position. This preserves the diagnostic shape of
	// bare-identifier-in-expression errors: without the rewind, an `alt`
	// combinator would see a rightmost error whose span lies past the
	// identifier, masking the true "bare identifier is not legal here"
	// position.
	let original_input = input;
	let start = input.location_offset();
	let (after_name, name) = identifier(input)?;
	let name_span = SourceSpan {
		start: name.location_offset(),
		end: name.location_offset() + name.fragment().len()
	};
	let check: IResult<Span, char, ParseError> =
		preceded(multispace0, char('@')).parse_complete(after_name);
	let input = match check
	{
		Ok((rest, _)) => rest,
		Err(_) =>
		{
			return Err(nom::Err::Error(
				<ParseError as NomParseError<Span>>::from_error_kind(
					original_input,
					ErrorKind::Tag
				)
			));
		}
	};
	let (input, _) =
		cut(preceded(multispace0, char('('))).parse_complete(input)?;
	let (input, expression) = cut(preceded(
		multispace0,
		context(BINDING_EXPRESSION_CONTEXT, expression)
	))
	.parse_complete(input)?;
	let (input, _) = cut(preceded(
		multispace0,
		context(CLOSING_PAREN_CONTEXT, char(')'))
	))
	.parse_complete(input)?;
	let end = input.location_offset();
	Ok((
		input,
		Binding {
			name: name.fragment(),
			name_span,
			expression: Box::new(expression),
			span: SourceSpan { start, end }
		}
	))
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
	let span_start = input.location_offset();
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
	let span_end = input.location_offset();
	Ok((
		input,
		Range {
			start: Box::new(start),
			end: Box::new(end),
			span: SourceSpan {
				start: span_start,
				end: span_end
			}
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
	let dice_start = input.location_offset();
	let (input, count) =
		context(DICE_COUNT_CONTEXT, dice_count).parse_complete(input)?;
	let (input, _) = preceded(multispace0, d_operator).parse_complete(input)?;
	// After the `d`/`D` operator, we're committed to a dice expression. For
	// standard dice, the face expression carries its own span (so the wrapper's
	// span falls out of its children); for custom dice the face list is a bare
	// `Vec<i32>`, so we patch the wrapper's span from the outer offsets after
	// parsing completes.
	let (input, initial_dice) = cut(preceded(
		multispace0,
		alt((
			context(
				STANDARD_FACES_CONTEXT,
				map(standard_faces, {
					let count = count.clone();
					move |faces| {
						let span = SourceSpan {
							start: count.span().start,
							end: faces.span().end
						};
						DiceExpression::Standard(StandardDice {
							count: Box::new(count.clone()),
							faces: Box::new(faces),
							span
						})
					}
				})
			),
			context(
				CUSTOM_FACES_CONTEXT,
				map(custom_faces, move |faces| {
					DiceExpression::Custom(CustomDice {
						count: Box::new(count.clone()),
						faces,
						span: SourceSpan::default()
					})
				})
			)
		))
	))
	.parse_complete(input)?;
	// Patch the CustomDice span now that we have the post-parse offset.
	let initial_dice = match initial_dice
	{
		DiceExpression::Custom(dice) => DiceExpression::Custom(CustomDice {
			span: SourceSpan {
				start: dice_start,
				end: input.location_offset()
			},
			..dice
		}),
		other => other
	};

	let (input, final_dice) = fold_many0(
		drop_clause,
		|| initial_dice.clone(),
		|acc, (direction, drop, clause_span)| {
			let span = SourceSpan {
				start: acc.span().start,
				end: clause_span.end
			};
			match direction
			{
				DropDirection::Lowest =>
				{
					DiceExpression::DropLowest(DropLowest {
						dice: Box::new(acc),
						drop,
						span
					})
				},
				DropDirection::Highest =>
				{
					DiceExpression::DropHighest(DropHighest {
						dice: Box::new(acc),
						drop,
						span
					})
				},
			}
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
		let span = SourceSpan {
			start: count.span().start,
			end: faces.span().end
		};
		(
			input,
			StandardDice {
				count: Box::new(count),
				faces: Box::new(faces),
				span
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
	let start = input.location_offset();
	let (input, (count, faces)) = separated_pair(
		context(DICE_COUNT_CONTEXT, dice_count),
		preceded(multispace0, d_operator),
		preceded(multispace0, context(CUSTOM_FACES_CONTEXT, custom_faces))
	)
	.parse_complete(input)?;
	let end = input.location_offset();
	Ok((
		input,
		CustomDice {
			count: Box::new(count),
			faces,
			span: SourceSpan { start, end }
		}
	))
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
		context(BINDING_CONTEXT, map(binding, Expression::Binding)),
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
		context(BINDING_CONTEXT, map(binding, Expression::Binding)),
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
/// A triple `(direction, drop, span)`:
/// - `direction`: [`DropDirection::Lowest`] or [`DropDirection::Highest`].
/// - `drop`: the optional drop-count expression.
/// - `span`: the [source span](SourceSpan) of the clause itself, starting at
///   `drop` and extending through the drop count (or the direction keyword if
///   no count is given). Does not include any leading whitespace preceding
///   `drop`.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
fn drop_clause(
	input: Span<'_>
) -> IResult<
	Span<'_>,
	(DropDirection, Option<Box<Expression<'_>>>, SourceSpan),
	ParseError<'_>
>
{
	let (input, _) = multispace0.parse_complete(input)?;
	let start = input.location_offset();
	let (input, _) = tag("drop").parse_complete(input)?;
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
	let end = input.location_offset();
	Ok((
		input,
		(direction, drop.map(Box::new), SourceSpan { start, end })
	))
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
		context(BINDING_CONTEXT, map(binding, Expression::Binding)),
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
	// Rewind the input to just after the trimmed identifier, giving back
	// any trailing whitespace to the remaining input. When no trailing
	// whitespace is present the `trail` computes to zero and the rewind
	// is a no-op, so the branchless construction preserves the original
	// spans exactly; the unconditional form also precludes a spurious
	// equivalent mutant on the boundary comparison.
	let trimmed = span.fragment().trim_end();
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
