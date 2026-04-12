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

impl<'a> Expression<'a>
{
	/// Dispatch this expression to the appropriate method on the given
	/// [`ASTVisitor`]. Enum variants that are themselves enums
	/// ([`Dice`](Self::Dice), [`Arithmetic`](Self::Arithmetic)) delegate to
	/// their own [`accept()`](DiceExpression::accept) methods.
	///
	/// # Parameters
	/// - `visitor`: The visitor to dispatch to.
	///
	/// # Returns
	/// The value produced by the visitor.
	///
	/// # Errors
	/// Propagates any error returned by the visitor.
	pub fn accept<V: ASTVisitor<'a>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		match self
		{
			Expression::Group(g) => visitor.visit_group(g),
			Expression::Constant(c) => visitor.visit_constant(c),
			Expression::Variable(v) => visitor.visit_variable(v),
			Expression::Range(r) => visitor.visit_range(r),
			Expression::Dice(d) => d.accept(visitor),
			Expression::Arithmetic(a) => a.accept(visitor)
		}
	}
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

impl<'a> DiceExpression<'a>
{
	/// Dispatch this dice expression to the appropriate method on the given
	/// [`ASTVisitor`].
	///
	/// # Parameters
	/// - `visitor`: The visitor to dispatch to.
	///
	/// # Returns
	/// The value produced by the visitor.
	///
	/// # Errors
	/// Propagates any error returned by the visitor.
	pub fn accept<V: ASTVisitor<'a>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		match self
		{
			DiceExpression::Standard(d) => visitor.visit_standard_dice(d),
			DiceExpression::Custom(d) => visitor.visit_custom_dice(d),
			DiceExpression::DropLowest(d) => visitor.visit_drop_lowest(d),
			DiceExpression::DropHighest(d) => visitor.visit_drop_highest(d)
		}
	}
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

////////////////////////////////////////////////////////////////////////////////
//                                AST visitor.                                //
////////////////////////////////////////////////////////////////////////////////

/// A visitor for walking the abstract syntax tree (AST).
///
/// Each method visits a single AST node type and returns a value of the
/// associated [`Output`](Self::Output) type. The enum types ([`Expression`],
/// [`DiceExpression`], [`ArithmeticExpression`]) are not visited directly —
/// they provide [`accept()`](Expression::accept) methods that dispatch to the
/// appropriate visitor method.
///
/// The [`CodeGenerator`](crate::codegen::CodeGenerator) is the reference
/// implementation of this trait.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrowed source text within the AST.
///
/// # Associated types
/// - `Output`: The value produced by visiting a node.
/// - `Error`: The error type returned on failure.
pub trait ASTVisitor<'a>
{
	/// The value produced by visiting a node.
	type Output;

	/// The error type returned on failure.
	type Error;

	/// Visit a [function](Function) definition.
	fn visit_function(
		&mut self,
		node: &'a Function<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [group](Group) (parenthesized expression).
	fn visit_group(
		&mut self,
		node: &'a Group<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [constant](Constant) value.
	fn visit_constant(
		&mut self,
		node: &Constant
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [variable](Variable) reference.
	fn visit_variable(
		&mut self,
		node: &'a Variable<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [range](Range) expression.
	fn visit_range(
		&mut self,
		node: &'a Range<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [standard dice](StandardDice) expression.
	fn visit_standard_dice(
		&mut self,
		node: &'a StandardDice<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [custom dice](CustomDice) expression.
	fn visit_custom_dice(
		&mut self,
		node: &'a CustomDice<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-lowest](DropLowest) expression.
	fn visit_drop_lowest(
		&mut self,
		node: &'a DropLowest<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-highest](DropHighest) expression.
	fn visit_drop_highest(
		&mut self,
		node: &'a DropHighest<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [addition](Add) expression.
	fn visit_add(
		&mut self,
		node: &'a Add<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [subtraction](Sub) expression.
	fn visit_sub(
		&mut self,
		node: &'a Sub<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [multiplication](Mul) expression.
	fn visit_mul(
		&mut self,
		node: &'a Mul<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [division](Div) expression.
	fn visit_div(
		&mut self,
		node: &'a Div<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [modulo](Mod) expression.
	fn visit_mod(
		&mut self,
		node: &'a Mod<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [exponentiation](Exp) expression.
	fn visit_exp(
		&mut self,
		node: &'a Exp<'a>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [negation](Neg) expression.
	fn visit_neg(
		&mut self,
		node: &'a Neg<'a>
	) -> Result<Self::Output, Self::Error>;
}

////////////////////////////////////////////////////////////////////////////////
//                          Arithmetic expressions.                           //
////////////////////////////////////////////////////////////////////////////////

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

impl<'a> ArithmeticExpression<'a>
{
	/// Dispatch this arithmetic expression to the appropriate method on the
	/// given [`ASTVisitor`].
	///
	/// # Parameters
	/// - `visitor`: The visitor to dispatch to.
	///
	/// # Returns
	/// The value produced by the visitor.
	///
	/// # Errors
	/// Propagates any error returned by the visitor.
	pub fn accept<V: ASTVisitor<'a>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		match self
		{
			ArithmeticExpression::Add(a) => visitor.visit_add(a),
			ArithmeticExpression::Sub(s) => visitor.visit_sub(s),
			ArithmeticExpression::Mul(m) => visitor.visit_mul(m),
			ArithmeticExpression::Div(d) => visitor.visit_div(d),
			ArithmeticExpression::Mod(m) => visitor.visit_mod(m),
			ArithmeticExpression::Exp(e) => visitor.visit_exp(e),
			ArithmeticExpression::Neg(n) => visitor.visit_neg(n)
		}
	}
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
