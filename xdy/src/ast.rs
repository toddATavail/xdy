//! # Abstract syntax tree (AST)
//!
//! The abstract syntax tree (AST) represents the structure of a semantically
//! correct `xDy` program. The [parser](crate::parser::parse) generates the AST
//! from the source code, and due to the simple rules of the dice language, the
//! AST is guaranteed to be semantically correct. The [compiler](crate::compile)
//! walks the AST to generate `xDy`'s intermediate representation (IR), which
//! may then be [optimized](crate::Optimizer::optimize) and
//! [evaluated](crate::evaluate).
//!
//! The root of the AST is a [`Function`].

use std::fmt::{self, Display, Formatter};

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
