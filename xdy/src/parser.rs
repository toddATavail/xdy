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
//! primary ::= group | CONSTANT | variable | range | dice
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

use std::fmt::{self, Display, Formatter};

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
pub fn parse(input: Span) -> IResult<Span, Function>
{
	preceded(multispace0, function).parse_complete(input)
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
		map(group, Expression::Group),
		map(variable, Expression::Variable),
		map(range, Expression::Range),
		map(dice, Expression::Dice),
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
