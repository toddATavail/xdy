//! # Abstract syntax tree (AST)
//!
//! The abstract syntax tree (AST) represents the structure of a semantically
//! correct `xDy` program. The [parser](crate::Parser::parse) generates the AST
//! from the source code, and due to the simple rules of the dice language, the
//! AST is guaranteed to be semantically correct. The [compiler](crate::compile)
//! walks the AST to generate `xDy`'s intermediate representation (IR), which
//! may then be [optimized](crate::Optimizer::optimize) and
//! [evaluated](crate::evaluate).
//!
//! Every AST node carries a [source span](SourceSpan) referencing the byte
//! range of the original input from which it was parsed. The [`Spanned`] trait
//! provides uniform access to this metadata and offers an
//! [`untethered`](Spanned::untethered) operation for position-independent
//! structural comparison.
//!
//! The root of the AST is a [`Function`].

use std::fmt::{self, Display, Formatter};

use crate::span::{SourceSpan, Spanned};

////////////////////////////////////////////////////////////////////////////////
//                        Abstract syntax tree (AST).                         //
////////////////////////////////////////////////////////////////////////////////

/// A function definition.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which this AST was parsed.
///   Parameter names and variable identifiers are borrowed directly from the
///   source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function<'src>
{
	/// The formal parameters of the function, if any.
	pub parameters: Option<Vec<Parameter<'src>>>,

	/// The body of the function.
	pub body: Expression<'src>,

	/// The span of the entire function definition in the original source.
	pub span: SourceSpan
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

/// A formal parameter of a [function](Function).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. The parameter name is borrowed
///   directly from the source.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<'src>
{
	/// The name of the parameter.
	pub name: &'src str,

	/// The span of this parameter in the original source.
	pub span: SourceSpan
}

impl Display for Parameter<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}", self.name)
	}
}

/// A parenthesized expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Group<'src>
{
	/// The expression inside the parentheses.
	pub expression: Box<Expression<'src>>,

	/// The span of the group, including both parentheses.
	pub span: SourceSpan
}

impl Display for Group<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "({})", self.expression)
	}
}

/// A constant value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constant
{
	/// The integer value of the constant.
	pub value: i32,

	/// The span of the constant in the original source.
	pub span: SourceSpan
}

impl Display for Constant
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}", self.value)
	}
}

/// A variable reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable<'src>
{
	/// The name of the variable, without the surrounding braces.
	pub name: &'src str,

	/// The span of the variable reference, including the surrounding braces.
	pub span: SourceSpan
}

impl Display for Variable<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{{{}}}", self.name)
	}
}

/// A local binding that names a subexpression so its integer result can be
/// referred to by [variable reference](Variable) later in the same enclosing
/// function body. The syntax is `name@(expr)`: the bound name appears to the
/// left of the `@` operator without delimiters (unlike a
/// [variable reference](Variable), which uses `{name}`), followed by the bound
/// expression enclosed in parentheses.
///
/// # Semantics
/// - The binding introduces `name` into a single flat namespace shared by
///   formal parameters, environment variables, and all other local bindings in
///   the same function body — no shadowing or nested scopes.
/// - References to `name` are forward-only: a binding must lexically precede
///   every reference to it, and must not appear inside its own bound expression
///   (i.e., self-reference is a compile error).
/// - The bound value is the integer main effect of `expr` — the same `i32` that
///   the expression would contribute to its parent. Rolling records produced by
///   dice inside `expr` still flow anonymously into `Evaluation.records` in
///   lexical order.
/// - A binding expression evaluates to the bound value, so it can appear
///   anywhere a [variable reference](Variable) is legal — as a primary, a dice
///   count, standard faces, or a drop count.
///
/// Collision, rebinding, and use-before-bind errors are reported by the
/// [`Validator`](crate::Validator).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. The bound name is borrowed
///   directly from the source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Binding<'src>
{
	/// The bound name.
	pub name: &'src str,

	/// The span of the bound name alone in the original source, excluding the
	/// `@` operator and the parenthesized bound expression.
	pub name_span: SourceSpan,

	/// The bound expression.
	pub expression: Box<Expression<'src>>,

	/// The span of the entire binding, from the first character of `name`
	/// through the closing `)` of the bound expression.
	pub span: SourceSpan
}

impl Display for Binding<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}@({})", self.name, self.expression)
	}
}

/// A range expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Range<'src>
{
	/// The start of the range.
	pub start: Box<Expression<'src>>,

	/// The end of the range.
	pub end: Box<Expression<'src>>,

	/// The span of the range expression, including the surrounding brackets.
	pub span: SourceSpan
}

impl Display for Range<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "[{}:{}]", self.start, self.end)
	}
}

/// An arbitrary expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. Inherited from the enclosing
///   [`Function`]; individual expression nodes borrow variable names from the
///   source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expression<'src>
{
	/// A parenthesized expression.
	Group(Group<'src>),

	/// A constant value.
	Constant(Constant),

	/// A variable reference.
	Variable(Variable<'src>),

	/// A local binding.
	Binding(Binding<'src>),

	/// A range expression.
	Range(Range<'src>),

	/// A dice expression.
	Dice(DiceExpression<'src>),

	/// An arithmetic expression.
	Arithmetic(ArithmeticExpression<'src>)
}

impl<'src> Expression<'src>
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
	pub fn accept<V: ASTVisitor<'src>>(
		&'src self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		match self
		{
			Expression::Group(g) => visitor.visit_group(g),
			Expression::Constant(c) => visitor.visit_constant(c),
			Expression::Variable(v) => visitor.visit_variable(v),
			Expression::Binding(b) => visitor.visit_binding(b),
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
			Expression::Binding(binding) => write!(f, "{}", binding),
			Expression::Range(range) => write!(f, "{}", range),
			Expression::Dice(dice) => write!(f, "{}", dice),
			Expression::Arithmetic(arithmetic) => write!(f, "{}", arithmetic)
		}
	}
}

/// A standard dice expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StandardDice<'src>
{
	/// The number of dice to roll.
	pub count: Box<Expression<'src>>,

	/// The number of faces on each die, starting at 1.
	pub faces: Box<Expression<'src>>,

	/// The span of the dice expression, from `count` through `faces`.
	pub span: SourceSpan
}

impl Display for StandardDice<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}D{}", self.count, self.faces)
	}
}

/// A custom dice expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomDice<'src>
{
	/// The number of dice to roll.
	pub count: Box<Expression<'src>>,

	/// The faces themselves.
	pub faces: Vec<i32>,

	/// The span of the dice expression, from `count` through the closing
	/// bracket of the face list.
	pub span: SourceSpan
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropLowest<'src>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'src>>,

	/// The number of dice to drop. Defaults to 1.
	pub drop: Option<Box<Expression<'src>>>,

	/// The span of the drop-lowest expression, from the dice expression
	/// through the drop count (or the `lowest` keyword if no count is given).
	pub span: SourceSpan
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropHighest<'src>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'src>>,

	/// The number of dice to drop. Defaults to 1.
	pub drop: Option<Box<Expression<'src>>>,

	/// The span of the drop-highest expression, from the dice expression
	/// through the drop count (or the `highest` keyword if no count is given).
	pub span: SourceSpan
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiceExpression<'src>
{
	/// A standard dice expression.
	Standard(StandardDice<'src>),

	/// A custom dice expression.
	Custom(CustomDice<'src>),

	/// A drop-lowest expression.
	DropLowest(DropLowest<'src>),

	/// A drop-highest expression.
	DropHighest(DropHighest<'src>)
}

impl<'src> DiceExpression<'src>
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
	pub fn accept<V: ASTVisitor<'src>>(
		&'src self,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Add<'src>
{
	/// The augend.
	pub left: Box<Expression<'src>>,

	/// The addend.
	pub right: Box<Expression<'src>>,

	/// The span of the addition expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Add<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} + {}", self.left, self.right)
	}
}

/// A subtraction expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sub<'src>
{
	/// The minuend.
	pub left: Box<Expression<'src>>,

	/// The subtrahend.
	pub right: Box<Expression<'src>>,

	/// The span of the subtraction expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Sub<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} - {}", self.left, self.right)
	}
}

/// A multiplication expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Mul<'src>
{
	/// The multiplicand.
	pub left: Box<Expression<'src>>,

	/// The multiplier.
	pub right: Box<Expression<'src>>,

	/// The span of the multiplication expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Mul<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} * {}", self.left, self.right)
	}
}

/// A division expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Div<'src>
{
	/// The dividend.
	pub left: Box<Expression<'src>>,

	/// The divisor.
	pub right: Box<Expression<'src>>,

	/// The span of the division expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Div<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} / {}", self.left, self.right)
	}
}

/// A modulo expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Mod<'src>
{
	/// The dividend.
	pub left: Box<Expression<'src>>,

	/// The divisor.
	pub right: Box<Expression<'src>>,

	/// The span of the modulo expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Mod<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} % {}", self.left, self.right)
	}
}

/// An exponentiation expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Exp<'src>
{
	/// The base.
	pub left: Box<Expression<'src>>,

	/// The exponent.
	pub right: Box<Expression<'src>>,

	/// The span of the exponentiation expression, from `left` through `right`.
	pub span: SourceSpan
}

impl Display for Exp<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} ^ {}", self.left, self.right)
	}
}

/// A negation expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Neg<'src>
{
	/// The operand.
	pub operand: Box<Expression<'src>>,

	/// The span of the negation expression, from the leading `-` through
	/// `operand`.
	pub span: SourceSpan
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
/// The [`Compiler`](crate::Compiler) is the reference implementation of this
/// trait.
///
/// # Type parameters
/// - `'src`: The lifetime of the borrowed source text within the AST.
///
/// # Associated types
/// - `Output`: The value produced by visiting a node.
/// - `Error`: The error type returned on failure.
pub trait ASTVisitor<'src>
{
	/// The value produced by visiting a node.
	type Output;

	/// The error type returned on failure.
	type Error;

	/// Visit a [function](Function) definition.
	fn visit_function(
		&mut self,
		node: &'src Function<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [group](Group) (parenthesized expression).
	fn visit_group(
		&mut self,
		node: &'src Group<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [constant](Constant) value.
	fn visit_constant(
		&mut self,
		node: &Constant
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [variable](Variable) reference.
	fn visit_variable(
		&mut self,
		node: &'src Variable<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a local [binding](Binding).
	fn visit_binding(
		&mut self,
		node: &'src Binding<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [range](Range) expression.
	fn visit_range(
		&mut self,
		node: &'src Range<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [standard dice](StandardDice) expression.
	fn visit_standard_dice(
		&mut self,
		node: &'src StandardDice<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [custom dice](CustomDice) expression.
	fn visit_custom_dice(
		&mut self,
		node: &'src CustomDice<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-lowest](DropLowest) expression.
	fn visit_drop_lowest(
		&mut self,
		node: &'src DropLowest<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-highest](DropHighest) expression.
	fn visit_drop_highest(
		&mut self,
		node: &'src DropHighest<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [addition](Add) expression.
	fn visit_add(
		&mut self,
		node: &'src Add<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [subtraction](Sub) expression.
	fn visit_sub(
		&mut self,
		node: &'src Sub<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [multiplication](Mul) expression.
	fn visit_mul(
		&mut self,
		node: &'src Mul<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [division](Div) expression.
	fn visit_div(
		&mut self,
		node: &'src Div<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [modulo](Mod) expression.
	fn visit_mod(
		&mut self,
		node: &'src Mod<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [exponentiation](Exp) expression.
	fn visit_exp(
		&mut self,
		node: &'src Exp<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [negation](Neg) expression.
	fn visit_neg(
		&mut self,
		node: &'src Neg<'src>
	) -> Result<Self::Output, Self::Error>;
}

////////////////////////////////////////////////////////////////////////////////
//                          Arithmetic expressions.                           //
////////////////////////////////////////////////////////////////////////////////

/// An arithmetic expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithmeticExpression<'src>
{
	/// An addition expression.
	Add(Add<'src>),

	/// A subtraction expression.
	Sub(Sub<'src>),

	/// A multiplication expression.
	Mul(Mul<'src>),

	/// A division expression.
	Div(Div<'src>),

	/// A modulo expression.
	Mod(Mod<'src>),

	/// An exponentiation expression.
	Exp(Exp<'src>),

	/// A negation expression.
	Neg(Neg<'src>)
}

impl<'src> ArithmeticExpression<'src>
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
	pub fn accept<V: ASTVisitor<'src>>(
		&'src self,
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

////////////////////////////////////////////////////////////////////////////////
//                          Spanned implementations.                          //
////////////////////////////////////////////////////////////////////////////////

impl Spanned for Function<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Function {
			parameters: self
				.parameters
				.as_ref()
				.map(|ps| ps.iter().map(Spanned::untethered).collect()),
			body: self.body.untethered(),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Parameter<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Parameter {
			name: self.name,
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Group<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Group {
			expression: Box::new(self.expression.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Constant
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Constant {
			value: self.value,
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Variable<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Variable {
			name: self.name,
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Binding<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Binding {
			name: self.name,
			name_span: SourceSpan::default(),
			expression: Box::new(self.expression.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Range<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Range {
			start: Box::new(self.start.untethered()),
			end: Box::new(self.end.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Expression<'_>
{
	fn span(&self) -> SourceSpan
	{
		match self
		{
			Expression::Group(g) => g.span(),
			Expression::Constant(c) => c.span(),
			Expression::Variable(v) => v.span(),
			Expression::Binding(b) => b.span(),
			Expression::Range(r) => r.span(),
			Expression::Dice(d) => d.span(),
			Expression::Arithmetic(a) => a.span()
		}
	}

	fn untethered(&self) -> Self
	{
		match self
		{
			Expression::Group(g) => Expression::Group(g.untethered()),
			Expression::Constant(c) => Expression::Constant(c.untethered()),
			Expression::Variable(v) => Expression::Variable(v.untethered()),
			Expression::Binding(b) => Expression::Binding(b.untethered()),
			Expression::Range(r) => Expression::Range(r.untethered()),
			Expression::Dice(d) => Expression::Dice(d.untethered()),
			Expression::Arithmetic(a) => Expression::Arithmetic(a.untethered())
		}
	}
}

impl Spanned for StandardDice<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		StandardDice {
			count: Box::new(self.count.untethered()),
			faces: Box::new(self.faces.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for CustomDice<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		CustomDice {
			count: Box::new(self.count.untethered()),
			faces: self.faces.clone(),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for DropLowest<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		DropLowest {
			dice: Box::new(self.dice.untethered()),
			drop: self.drop.as_ref().map(|d| Box::new(d.untethered())),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for DropHighest<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		DropHighest {
			dice: Box::new(self.dice.untethered()),
			drop: self.drop.as_ref().map(|d| Box::new(d.untethered())),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for DiceExpression<'_>
{
	fn span(&self) -> SourceSpan
	{
		match self
		{
			DiceExpression::Standard(d) => d.span(),
			DiceExpression::Custom(d) => d.span(),
			DiceExpression::DropLowest(d) => d.span(),
			DiceExpression::DropHighest(d) => d.span()
		}
	}

	fn untethered(&self) -> Self
	{
		match self
		{
			DiceExpression::Standard(d) =>
			{
				DiceExpression::Standard(d.untethered())
			},
			DiceExpression::Custom(d) => DiceExpression::Custom(d.untethered()),
			DiceExpression::DropLowest(d) =>
			{
				DiceExpression::DropLowest(d.untethered())
			},
			DiceExpression::DropHighest(d) =>
			{
				DiceExpression::DropHighest(d.untethered())
			},
		}
	}
}

impl Spanned for Add<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Add {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Sub<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Sub {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Mul<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Mul {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Div<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Div {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Mod<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Mod {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Exp<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Exp {
			left: Box::new(self.left.untethered()),
			right: Box::new(self.right.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Neg<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Neg {
			operand: Box::new(self.operand.untethered()),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for ArithmeticExpression<'_>
{
	fn span(&self) -> SourceSpan
	{
		match self
		{
			ArithmeticExpression::Add(a) => a.span(),
			ArithmeticExpression::Sub(s) => s.span(),
			ArithmeticExpression::Mul(m) => m.span(),
			ArithmeticExpression::Div(d) => d.span(),
			ArithmeticExpression::Mod(m) => m.span(),
			ArithmeticExpression::Exp(e) => e.span(),
			ArithmeticExpression::Neg(n) => n.span()
		}
	}

	fn untethered(&self) -> Self
	{
		match self
		{
			ArithmeticExpression::Add(a) =>
			{
				ArithmeticExpression::Add(a.untethered())
			},
			ArithmeticExpression::Sub(s) =>
			{
				ArithmeticExpression::Sub(s.untethered())
			},
			ArithmeticExpression::Mul(m) =>
			{
				ArithmeticExpression::Mul(m.untethered())
			},
			ArithmeticExpression::Div(d) =>
			{
				ArithmeticExpression::Div(d.untethered())
			},
			ArithmeticExpression::Mod(m) =>
			{
				ArithmeticExpression::Mod(m.untethered())
			},
			ArithmeticExpression::Exp(e) =>
			{
				ArithmeticExpression::Exp(e.untethered())
			},
			ArithmeticExpression::Neg(n) =>
			{
				ArithmeticExpression::Neg(n.untethered())
			},
		}
	}
}
