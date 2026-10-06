//! # Abstract syntax tree (AST)
//!
//! The abstract syntax tree (AST) represents the structure of a syntactically
//! well-formed `xDy` program. The [parser](crate::Parser::parse) generates the
//! AST from the source code, but a well-formed AST is not necessarily
//! semantically valid, e.g., `{x}, {x}: {x} + 1` parses but declares the same
//! parameter twice. The [validator](crate::Validator::validate) performs the
//! semantic checks. The [compiler](crate::compile) then walks the validated AST
//! to generate `xDy`'s intermediate representation (IR), which may then be
//! [optimized](crate::Optimizer::optimize) and [evaluated](crate::evaluate).
//!
//! Every AST node carries a [source span](SourceSpan) referencing the byte
//! range of the original input from which it was parsed. The [`Spanned`] trait
//! provides uniform access to this metadata and offers an
//! [`untethered`](Spanned::untethered) operation for position-independent
//! structural comparison.
//!
//! The root of the AST is a [`Function`].
//!
//! # Deep nesting
//!
//! Every recursive field of the AST is an [`Expression`] or a
//! [`DiceExpression`], whose implementations of [`Debug`](std::fmt::Debug),
//! [`Clone`], [`PartialEq`], [`Eq`], [`Hash`](std::hash::Hash), and [`Drop`]
//! use explicit stacks rather than recursion. An AST of any depth can therefore
//! be formatted, cloned, compared, hashed, and dropped without exhausting the
//! stack. The other AST types derive these traits, and reach the iterative
//! implementations after a single level. Because [`Expression`] and
//! [`DiceExpression`] implement [`Drop`], their variants cannot be moved out by
//! pattern; match on a reference instead.
//!
//! Every AST type's [`Display`] implementation, and the
//! [`untethered`](Spanned::untethered) operation of [`Expression`] and
//! [`DiceExpression`], likewise use explicit stacks, as do the
//! [S-expression](crate::s_expr) writer and sizer, and the driver that walks
//! an AST with an [`ASTVisitor`]. The crate's other analyses of an AST, in the
//! compiler, the validator, and the diagnostics, loop over the same iterative
//! walk.

mod display;
mod impls;
mod traversal;
mod walk;

use std::{
	borrow::Cow,
	fmt::{self, Display, Formatter}
};

pub(crate) use walk::{Event, Node, Walk};

use crate::span::{SourceSpan, Spanned};

////////////////////////////////////////////////////////////////////////////////
//                        Abstract syntax tree (AST).                         //
////////////////////////////////////////////////////////////////////////////////

/// A function definition.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which this AST was parsed.
///   Parameter, variable, and binding names are borrowed directly from the
///   source whenever the source spells them
///   [canonically](crate::parser::canonical_name).
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

impl<'src> Function<'src>
{
	/// Walk this function with the given [`ASTVisitor`], iteratively, as the
	/// [trait](ASTVisitor) describes: enter the function, walk its body, and
	/// then visit the function with the output of its body.
	///
	/// # Type parameters
	/// - `'a`: The lifetime of the borrowed function.
	/// - `V`: The type of the visitor.
	///
	/// # Parameters
	/// - `visitor`: The visitor.
	///
	/// # Returns
	/// The output of this function.
	///
	/// # Errors
	/// Propagates the first error returned by the visitor.
	pub fn accept<'a, V: ASTVisitor<'a, 'src>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		visitor.enter_function(self)?;
		let body = self.body.accept(visitor)?;
		visitor.visit_function(self, body)
	}
}

impl Display for Function<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
	}
}

/// A formal parameter of a [function](Function).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. The parameter name is borrowed
///   directly from the source whenever the source spells it
///   [canonically](crate::parser::canonical_name).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<'src>
{
	/// The [canonical](crate::parser::canonical_name) name of the parameter.
	pub name: Cow<'src, str>,

	/// The span of this parameter in the original source. The source text
	/// that it covers is the name exactly as written, which may differ from
	/// the canonical [`name`](Self::name) in its whitespace.
	pub span: SourceSpan
}

impl Display for Parameter<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
	}
}

/// A variable reference.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. The variable name is borrowed
///   directly from the source whenever the source spells it
///   [canonically](crate::parser::canonical_name).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable<'src>
{
	/// The [canonical](crate::parser::canonical_name) name of the variable,
	/// without the surrounding braces.
	pub name: Cow<'src, str>,

	/// The span of the variable reference, including the surrounding braces.
	pub span: SourceSpan
}

impl Display for Variable<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
	}
}

/// A local binding that names a subexpression so its integer result can be
/// referred to by [variable reference](Variable) later in the same enclosing
/// function body. The syntax is `{name}@(expr)`: the bound name appears to the
/// left of the `@` operator, braced just as in a [variable reference](Variable)
/// to it, followed by the bound expression enclosed in parentheses.
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
///   count, standard faces, the count of a drop or keep clause, or the
///   threshold, an endpoint, or the count of a reroll clause.
///
/// Collision, rebinding, and use-before-bind errors are reported by the
/// [`Validator`](crate::Validator).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. The bound name is borrowed
///   directly from the source whenever the source spells it
///   [canonically](crate::parser::canonical_name).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Binding<'src>
{
	/// The [canonical](crate::parser::canonical_name) bound name.
	pub name: Cow<'src, str>,

	/// The span of the bound identifier alone in the original source,
	/// excluding its surrounding braces, any whitespace just inside them,
	/// the `@` operator, and the parenthesized bound expression. The
	/// source text that it covers is the name exactly as written, which
	/// may differ from the canonical [`name`](Self::name) in its
	/// whitespace.
	pub name_span: SourceSpan,

	/// The bound expression.
	pub expression: Box<Expression<'src>>,

	/// The span of the entire binding, from the opening `{` of the bound name
	/// through the closing `)` of the bound expression.
	pub span: SourceSpan
}

impl Display for Binding<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
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
		display::render(self, f)
	}
}

/// An arbitrary expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. Inherited from the enclosing
///   [`Function`]; individual expression nodes borrow variable and binding
///   names from the source whenever the source spells them
///   [canonically](crate::parser::canonical_name).
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
	/// Walk this expression with the given [`ASTVisitor`], iteratively, as
	/// the [trait](ASTVisitor) describes. The walk ends with a call to
	/// [`visit_expression`](ASTVisitor::visit_expression) for this
	/// expression.
	///
	/// # Type parameters
	/// - `'a`: The lifetime of the borrowed expression.
	/// - `V`: The type of the visitor.
	///
	/// # Parameters
	/// - `visitor`: The visitor.
	///
	/// # Returns
	/// The output of this expression.
	///
	/// # Errors
	/// Propagates the first error returned by the visitor.
	pub fn accept<'a, V: ASTVisitor<'a, 'src>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		walk::fold(walk::Node::Expression(self), visitor)
	}
}

impl Display for Expression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
	}
}

/// A keep-lowest expression, which keeps the lowest dice of those that the
/// earlier clauses of its dice expression left kept, and drops the rest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeepLowest<'src>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'src>>,

	/// The number of dice to keep. Defaults to 1. A count of zero or less
	/// keeps none.
	pub keep: Option<Box<Expression<'src>>>,

	/// The span of the keep-lowest expression, from the dice expression
	/// through the keep count (or the `lowest` keyword if no count is given).
	pub span: SourceSpan
}

impl Display for KeepLowest<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
	}
}

/// A keep-highest expression, which keeps the highest dice of those that the
/// earlier clauses of its dice expression left kept, and drops the rest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeepHighest<'src>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'src>>,

	/// The number of dice to keep. Defaults to 1. A count of zero or less
	/// keeps none.
	pub keep: Option<Box<Expression<'src>>>,

	/// The span of the keep-highest expression, from the dice expression
	/// through the keep count (or the `highest` keyword if no count is given).
	pub span: SourceSpan
}

impl Display for KeepHighest<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
	}
}

/// A reroll expression, which rerolls each of the dice that the earlier clauses
/// of its dice expression left kept, and that satisfy its predicate, up to its
/// limit of times. A reroll crosses out the die that it rerolls, and appends
/// the replacement to the rolling record.
///
/// The operands of the predicate and the limit are each evaluated once, in the
/// order written, after the dice and before any reroll.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reroll<'src>
{
	/// The dice expression.
	pub dice: Box<DiceExpression<'src>>,

	/// The predicate that a die must satisfy to be rerolled.
	pub predicate: Predicate<Box<Expression<'src>>>,

	/// The number of times that each die may be rerolled. Defaults to once.
	pub limit: Option<Limit<Box<Expression<'src>>>>,

	/// The span of the reroll expression, from the dice expression through the
	/// limit (or the predicate if no limit is given).
	pub span: SourceSpan
}

impl Display for Reroll<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
	}
}

/// The predicate of a [reroll](Reroll): the test that a die must pass to be
/// rerolled. It mirrors the [predicate](crate::Predicate) of the reroll
/// instruction, but holds operands of any type, so that the same shape serves
/// both the AST, whose operands are [expressions](Expression), and the
/// [visitor](ASTVisitor), whose operands are the outputs of those expressions.
///
/// # Type parameters
/// - `T`: The type of the operands: the thresholds and the endpoints of the
///   ranges.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Predicate<T>
{
	/// Matches a result equal to the threshold, written `= t`.
	Eq(T),

	/// Matches a result not equal to the threshold, written `≠ t` or `!= t`.
	Ne(T),

	/// Matches a result less than the threshold, written `< t`.
	Lt(T),

	/// Matches a result less than or equal to the threshold, written `≤ t` or
	/// `<= t`.
	Le(T),

	/// Matches a result greater than the threshold, written `> t`.
	Gt(T),

	/// Matches a result greater than or equal to the threshold, written `≥ t`
	/// or `>= t`.
	Ge(T),

	/// Matches a result among the listed values, written `in [a, b, …]`. The
	/// values are literals, so that their members are always known. An empty
	/// list matches nothing.
	In(Vec<i32>),

	/// Matches a result within the half-open range `start..end`, written
	/// `in start..end`, which is empty if `end` does not exceed `start`.
	InRange
	{
		/// The least result matched.
		start: T,

		/// The least result beyond the range.
		end: T
	},

	/// Matches a result within the inclusive range `start..=end`, written
	/// `in start..=end`, which is empty if `end` is less than `start`.
	InRangeInclusive
	{
		/// The least result matched.
		start: T,

		/// The greatest result matched.
		end: T
	}
}

impl<T> Predicate<T>
{
	/// Answer the operands of the predicate, in the order written.
	///
	/// # Returns
	/// The operands, of which there are none for a list, two for a range, and
	/// one otherwise.
	pub fn operands(&self) -> impl Iterator<Item = &T>
	{
		match self
		{
			Self::Eq(threshold)
			| Self::Ne(threshold)
			| Self::Lt(threshold)
			| Self::Le(threshold)
			| Self::Gt(threshold)
			| Self::Ge(threshold) => [Some(threshold), None],
			Self::In(_) => [None, None],
			Self::InRange { start, end }
			| Self::InRangeInclusive { start, end } => [Some(start), Some(end)]
		}
		.into_iter()
		.flatten()
	}

	/// Take the operands of the predicate, in the order written.
	///
	/// # Returns
	/// The operands, of which there are none for a list, two for a range, and
	/// one otherwise.
	pub fn into_operands(self) -> impl Iterator<Item = T>
	{
		match self
		{
			Self::Eq(threshold)
			| Self::Ne(threshold)
			| Self::Lt(threshold)
			| Self::Le(threshold)
			| Self::Gt(threshold)
			| Self::Ge(threshold) => [Some(threshold), None],
			Self::In(_) => [None, None],
			Self::InRange { start, end }
			| Self::InRangeInclusive { start, end } => [Some(start), Some(end)]
		}
		.into_iter()
		.flatten()
	}

	/// Answer a predicate of the same form, whose operands are replaced by the
	/// specified function. A list keeps a copy of its values.
	///
	/// # Type parameters
	/// - `U`: The type of the replacement operands.
	///
	/// # Parameters
	/// - `replace`: The function that replaces each operand. It is called once
	///   per operand, in the order written.
	///
	/// # Returns
	/// The new predicate.
	pub fn map<'s, U>(
		&'s self,
		mut replace: impl FnMut(&'s T) -> U
	) -> Predicate<U>
	{
		match self
		{
			Self::Eq(threshold) => Predicate::Eq(replace(threshold)),
			Self::Ne(threshold) => Predicate::Ne(replace(threshold)),
			Self::Lt(threshold) => Predicate::Lt(replace(threshold)),
			Self::Le(threshold) => Predicate::Le(replace(threshold)),
			Self::Gt(threshold) => Predicate::Gt(replace(threshold)),
			Self::Ge(threshold) => Predicate::Ge(replace(threshold)),
			Self::In(values) => Predicate::In(values.clone()),
			Self::InRange { start, end } =>
			{
				let start = replace(start);
				Predicate::InRange {
					start,
					end: replace(end)
				}
			},
			Self::InRangeInclusive { start, end } =>
			{
				let start = replace(start);
				Predicate::InRangeInclusive {
					start,
					end: replace(end)
				}
			}
		}
	}
}

/// The limit of a [reroll](Reroll): the number of times that each die may be
/// rerolled. It mirrors the [limit](crate::Limit) of the reroll instruction,
/// but holds an operand of any type, as [`Predicate`] does.
///
/// # Type parameters
/// - `T`: The type of the count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Limit<T>
{
	/// Reroll each die at most this many times, written `n time` or
	/// `n times`, or not at all if the count is zero or less.
	Times(T),

	/// Reroll each die until it no longer matches, written `∞ times` or
	/// `unlimited times`, or not at all if every face of the die matches.
	Unlimited
}

impl<T> Limit<T>
{
	/// Answer the operand of the limit, if any.
	///
	/// # Returns
	/// The count of a [finite](Self::Times) limit, or `None` if the limit is
	/// [unlimited](Self::Unlimited).
	pub fn operand(&self) -> Option<&T>
	{
		match self
		{
			Self::Times(count) => Some(count),
			Self::Unlimited => None
		}
	}

	/// Answer a limit of the same form, whose operand, if any, is replaced by
	/// the specified function.
	///
	/// # Type parameters
	/// - `U`: The type of the replacement operand.
	///
	/// # Parameters
	/// - `replace`: The function that replaces the operand.
	///
	/// # Returns
	/// The new limit.
	pub fn map<'s, U>(&'s self, replace: impl FnOnce(&'s T) -> U) -> Limit<U>
	{
		match self
		{
			Self::Times(count) => Limit::Times(replace(count)),
			Self::Unlimited => Limit::Unlimited
		}
	}
}

/// A dice expression.
pub enum DiceExpression<'src>
{
	/// A standard dice expression.
	Standard(StandardDice<'src>),

	/// A custom dice expression.
	Custom(CustomDice<'src>),

	/// A drop-lowest expression.
	DropLowest(DropLowest<'src>),

	/// A drop-highest expression.
	DropHighest(DropHighest<'src>),

	/// A keep-lowest expression.
	KeepLowest(KeepLowest<'src>),

	/// A keep-highest expression.
	KeepHighest(KeepHighest<'src>),

	/// A reroll expression.
	Reroll(Reroll<'src>)
}

impl<'src> DiceExpression<'src>
{
	/// Walk this dice expression with the given [`ASTVisitor`], iteratively,
	/// as the [trait](ASTVisitor) describes. A dice expression does not fill
	/// a slot of type [`Expression`], so the walk does not call
	/// [`visit_expression`](ASTVisitor::visit_expression) for it.
	///
	/// # Type parameters
	/// - `'a`: The lifetime of the borrowed dice expression.
	/// - `V`: The type of the visitor.
	///
	/// # Parameters
	/// - `visitor`: The visitor.
	///
	/// # Returns
	/// The output of this dice expression.
	///
	/// # Errors
	/// Propagates the first error returned by the visitor.
	pub fn accept<'a, V: ASTVisitor<'a, 'src>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		walk::fold(walk::Node::Dice(self), visitor)
	}
}

impl Display for DiceExpression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
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
		display::render(self, f)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                AST visitor.                                //
////////////////////////////////////////////////////////////////////////////////

/// A visitor for walking the abstract syntax tree (AST), folding it bottom-up
/// into a single value.
///
/// A driver walks the tree iteratively, with an explicit stack, so a visitor
/// can walk an AST of any depth without exhausting the machine stack. Start a
/// walk by calling `accept` on a [`Function`], an [`Expression`], a
/// [`DiceExpression`], or an [`ArithmeticExpression`].
///
/// The walk visits children left to right, in source order. For each node, the
/// driver calls:
///
/// 1. the node's `enter_*` hook, before any of the node's children. The hooks
///    do nothing by default.
/// 2. the node's `visit_*` method, after all of the node's children. The method
///    receives the node and the [outputs](Self::Output) of its children, in
///    source order, and produces the node's own output.
/// 3. [`visit_expression`](Self::visit_expression), if the node fills a slot of
///    type [`Expression`], i.e., if it is not the [`DiceExpression`] beneath a
///    [drop-lowest](DropLowest), [drop-highest](DropHighest),
///    [keep-lowest](KeepLowest), [keep-highest](KeepHighest), or
///    [reroll](Reroll) expression. The hook receives the node's output and may
///    replace it. By default, it returns the output unchanged.
///
/// The enum types ([`Expression`], [`DiceExpression`],
/// [`ArithmeticExpression`]) are not visited directly; the driver visits the
/// node inside each variant. If any method answers an error, the walk stops at
/// once and `accept` answers that error.
///
/// The [`Compiler`](crate::Compiler) is the reference implementation of this
/// trait.
///
/// # Migrating from 0.12
///
/// In 0.12, each `visit_*` method received only its node, and walked the node's
/// children itself by calling `accept` on them, so every visitor recursed once
/// per level of nesting and overflowed the stack on deep trees. Since 0.13, the
/// driver walks the tree, and a visitor only answers for each node:
///
/// - The trait takes two lifetimes, `ASTVisitor<'a, 'src>`, where 0.12 took
///   one, `ASTVisitor<'src>`: `'a` borrows the tree and `'src` the source text,
///   so a visitor's outputs and errors may borrow the source for longer than
///   the walk.
/// - A `visit_*` method receives the outputs of the node's children as
///   arguments, in source order, and must not call `accept` on them. The drop
///   count of a drop clause, which may be absent, arrives as an [`Option`].
/// - Work that must happen before the children, e.g., opening a scope, moves to
///   the node's `enter_*` hook.
/// - Work that applies to every expression once it is complete, e.g., reducing
///   a rolling record to its sum, moves to
///   [`visit_expression`](Self::visit_expression).
///
/// # Type parameters
/// - `'a`: The lifetime of the borrowed AST.
/// - `'src`: The lifetime of the source text within the AST.
///
/// # Associated types
/// - `Output`: The value produced by visiting a node.
/// - `Error`: The error type returned on failure.
///
/// # Examples
/// The driver calls the visitor in this order to walk the expression `1 + 2`,
/// an [`Add`] of two [`Constant`]s.
///
/// ```mermaid
/// sequenceDiagram
///     participant D as Driver
///     participant V as Visitor
///     D->>V: enter_add(1 + 2)
///     D->>V: enter_constant(1)
///     D->>V: visit_constant(1)
///     V-->>D: a
///     D->>V: visit_expression(1, a)
///     V-->>D: a′
///     D->>V: enter_constant(2)
///     D->>V: visit_constant(2)
///     V-->>D: b
///     D->>V: visit_expression(2, b)
///     V-->>D: b′
///     D->>V: visit_add(1 + 2, a′, b′)
///     V-->>D: c
///     D->>V: visit_expression(1 + 2, c)
///     V-->>D: c′
/// ```
///
/// A visitor that counts the dice in an expression:
///
/// ```rust
/// use std::convert::Infallible;
///
/// use xdy::{
///     Parser,
///     ast::{
///         ASTVisitor, Add, Binding, Constant, CustomDice, Div, DropHighest,
///         DropLowest, Exp, Function, Group, KeepHighest, KeepLowest, Limit,
///         Mod, Mul, Neg, Predicate, Range, Reroll, StandardDice, Sub,
///         Variable
///     }
/// };
///
/// struct DiceCounter;
///
/// impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for DiceCounter
/// {
///     type Output = usize;
///     type Error = Infallible;
///
///     fn visit_function(&mut self, _: &'a Function<'src>, body: usize)
///         -> Result<usize, Infallible> { Ok(body) }
///     fn visit_group(&mut self, _: &'a Group<'src>, expression: usize)
///         -> Result<usize, Infallible> { Ok(expression) }
///     fn visit_constant(&mut self, _: &'a Constant)
///         -> Result<usize, Infallible> { Ok(0) }
///     fn visit_variable(&mut self, _: &'a Variable<'src>)
///         -> Result<usize, Infallible> { Ok(0) }
///     fn visit_binding(&mut self, _: &'a Binding<'src>, expression: usize)
///         -> Result<usize, Infallible> { Ok(expression) }
///     fn visit_range(&mut self, _: &'a Range<'src>, start: usize, end: usize)
///         -> Result<usize, Infallible> { Ok(start + end) }
///     fn visit_standard_dice(
///         &mut self, _: &'a StandardDice<'src>, count: usize, faces: usize
///     ) -> Result<usize, Infallible> { Ok(1 + count + faces) }
///     fn visit_custom_dice(&mut self, _: &'a CustomDice<'src>, count: usize)
///         -> Result<usize, Infallible> { Ok(1 + count) }
///     fn visit_drop_lowest(
///         &mut self, _: &'a DropLowest<'src>, dice: usize, drop: Option<usize>
///     ) -> Result<usize, Infallible> { Ok(dice + drop.unwrap_or(0)) }
///     fn visit_drop_highest(
///         &mut self, _: &'a DropHighest<'src>, dice: usize, drop: Option<usize>
///     ) -> Result<usize, Infallible> { Ok(dice + drop.unwrap_or(0)) }
///     fn visit_keep_lowest(
///         &mut self, _: &'a KeepLowest<'src>, dice: usize, keep: Option<usize>
///     ) -> Result<usize, Infallible> { Ok(dice + keep.unwrap_or(0)) }
///     fn visit_keep_highest(
///         &mut self, _: &'a KeepHighest<'src>, dice: usize, keep: Option<usize>
///     ) -> Result<usize, Infallible> { Ok(dice + keep.unwrap_or(0)) }
///     fn visit_reroll(
///         &mut self,
///         _: &'a Reroll<'src>,
///         dice: usize,
///         predicate: Predicate<usize>,
///         limit: Option<Limit<usize>>
///     ) -> Result<usize, Infallible>
///     {
///         Ok(dice
///             + predicate.operands().sum::<usize>()
///             + limit.as_ref().and_then(Limit::operand).unwrap_or(&0))
///     }
///     fn visit_add(&mut self, _: &'a Add<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_sub(&mut self, _: &'a Sub<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_mul(&mut self, _: &'a Mul<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_div(&mut self, _: &'a Div<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_mod(&mut self, _: &'a Mod<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_exp(&mut self, _: &'a Exp<'src>, left: usize, right: usize)
///         -> Result<usize, Infallible> { Ok(left + right) }
///     fn visit_neg(&mut self, _: &'a Neg<'src>, operand: usize)
///         -> Result<usize, Infallible> { Ok(operand) }
/// }
///
/// let ast =
///     Parser::parse("(1D6)D[1, 2] + 3D8 drop lowest + 4D6 keep highest 3")
///         .unwrap();
/// assert_eq!(ast.accept(&mut DiceCounter), Ok(4));
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait ASTVisitor<'a, 'src: 'a>
{
	/// The value produced by visiting a node.
	type Output;

	/// The error type returned on failure.
	type Error;

	/// Enter a [function](Function) definition, before its body.
	fn enter_function(
		&mut self,
		_node: &'a Function<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [group](Group) (parenthesized expression), before its
	/// expression.
	fn enter_group(&mut self, _node: &'a Group<'src>)
	-> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [constant](Constant) value.
	fn enter_constant(&mut self, _node: &'a Constant)
	-> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [variable](Variable) reference.
	fn enter_variable(
		&mut self,
		_node: &'a Variable<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a local [binding](Binding), before its bound expression.
	fn enter_binding(
		&mut self,
		_node: &'a Binding<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [range](Range) expression, before its bounds.
	fn enter_range(&mut self, _node: &'a Range<'src>)
	-> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [standard dice](StandardDice) expression, before its count and
	/// faces.
	fn enter_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [custom dice](CustomDice) expression, before its count.
	fn enter_custom_dice(
		&mut self,
		_node: &'a CustomDice<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [drop-lowest](DropLowest) expression, before its dice and drop
	/// count.
	fn enter_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [drop-highest](DropHighest) expression, before its dice and
	/// drop count.
	fn enter_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [keep-lowest](KeepLowest) expression, before its dice and keep
	/// count.
	fn enter_keep_lowest(
		&mut self,
		_node: &'a KeepLowest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [keep-highest](KeepHighest) expression, before its dice and
	/// keep count.
	fn enter_keep_highest(
		&mut self,
		_node: &'a KeepHighest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [reroll](Reroll) expression, before its dice, the operands of
	/// its predicate, and its limit.
	fn enter_reroll(
		&mut self,
		_node: &'a Reroll<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter an [addition](Add) expression, before its operands.
	fn enter_add(&mut self, _node: &'a Add<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [subtraction](Sub) expression, before its operands.
	fn enter_sub(&mut self, _node: &'a Sub<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [multiplication](Mul) expression, before its operands.
	fn enter_mul(&mut self, _node: &'a Mul<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [division](Div) expression, before its operands.
	fn enter_div(&mut self, _node: &'a Div<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [modulo](Mod) expression, before its operands.
	fn enter_mod(&mut self, _node: &'a Mod<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter an [exponentiation](Exp) expression, before its operands.
	fn enter_exp(&mut self, _node: &'a Exp<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Enter a [negation](Neg) expression, before its operand.
	fn enter_neg(&mut self, _node: &'a Neg<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	/// Visit a [function](Function) definition, after its body.
	///
	/// # Parameters
	/// - `node`: The function.
	/// - `body`: The output of the body.
	fn visit_function(
		&mut self,
		node: &'a Function<'src>,
		body: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [group](Group) (parenthesized expression), after its
	/// expression.
	///
	/// # Parameters
	/// - `node`: The group.
	/// - `expression`: The output of the expression inside the parentheses.
	fn visit_group(
		&mut self,
		node: &'a Group<'src>,
		expression: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [constant](Constant) value.
	///
	/// # Parameters
	/// - `node`: The constant.
	fn visit_constant(
		&mut self,
		node: &'a Constant
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [variable](Variable) reference.
	///
	/// # Parameters
	/// - `node`: The variable reference.
	fn visit_variable(
		&mut self,
		node: &'a Variable<'src>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a local [binding](Binding), after its bound expression.
	///
	/// # Parameters
	/// - `node`: The binding.
	/// - `expression`: The output of the bound expression.
	fn visit_binding(
		&mut self,
		node: &'a Binding<'src>,
		expression: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [range](Range) expression, after its bounds.
	///
	/// # Parameters
	/// - `node`: The range.
	/// - `start`: The output of the start of the range.
	/// - `end`: The output of the end of the range.
	fn visit_range(
		&mut self,
		node: &'a Range<'src>,
		start: Self::Output,
		end: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [standard dice](StandardDice) expression, after its count and
	/// faces.
	///
	/// # Parameters
	/// - `node`: The dice expression.
	/// - `count`: The output of the count.
	/// - `faces`: The output of the faces.
	fn visit_standard_dice(
		&mut self,
		node: &'a StandardDice<'src>,
		count: Self::Output,
		faces: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [custom dice](CustomDice) expression, after its count.
	///
	/// # Parameters
	/// - `node`: The dice expression.
	/// - `count`: The output of the count.
	fn visit_custom_dice(
		&mut self,
		node: &'a CustomDice<'src>,
		count: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-lowest](DropLowest) expression, after its dice and drop
	/// count.
	///
	/// # Parameters
	/// - `node`: The drop-lowest expression.
	/// - `dice`: The output of the dice expression.
	/// - `drop`: The output of the drop count, if the expression has one.
	fn visit_drop_lowest(
		&mut self,
		node: &'a DropLowest<'src>,
		dice: Self::Output,
		drop: Option<Self::Output>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [drop-highest](DropHighest) expression, after its dice and drop
	/// count.
	///
	/// # Parameters
	/// - `node`: The drop-highest expression.
	/// - `dice`: The output of the dice expression.
	/// - `drop`: The output of the drop count, if the expression has one.
	fn visit_drop_highest(
		&mut self,
		node: &'a DropHighest<'src>,
		dice: Self::Output,
		drop: Option<Self::Output>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [keep-lowest](KeepLowest) expression, after its dice and keep
	/// count.
	///
	/// # Parameters
	/// - `node`: The keep-lowest expression.
	/// - `dice`: The output of the dice expression.
	/// - `keep`: The output of the keep count, if the expression has one.
	fn visit_keep_lowest(
		&mut self,
		node: &'a KeepLowest<'src>,
		dice: Self::Output,
		keep: Option<Self::Output>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [keep-highest](KeepHighest) expression, after its dice and keep
	/// count.
	///
	/// # Parameters
	/// - `node`: The keep-highest expression.
	/// - `dice`: The output of the dice expression.
	/// - `keep`: The output of the keep count, if the expression has one.
	fn visit_keep_highest(
		&mut self,
		node: &'a KeepHighest<'src>,
		dice: Self::Output,
		keep: Option<Self::Output>
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [reroll](Reroll) expression, after its dice, the operands of its
	/// predicate, and its limit.
	///
	/// # Parameters
	/// - `node`: The reroll expression.
	/// - `dice`: The output of the dice expression.
	/// - `predicate`: The predicate, holding the outputs of its operands.
	/// - `limit`: The limit, holding the output of its count, if the expression
	///   has one.
	fn visit_reroll(
		&mut self,
		node: &'a Reroll<'src>,
		dice: Self::Output,
		predicate: Predicate<Self::Output>,
		limit: Option<Limit<Self::Output>>
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [addition](Add) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The addition.
	/// - `left`: The output of the augend.
	/// - `right`: The output of the addend.
	fn visit_add(
		&mut self,
		node: &'a Add<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [subtraction](Sub) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The subtraction.
	/// - `left`: The output of the minuend.
	/// - `right`: The output of the subtrahend.
	fn visit_sub(
		&mut self,
		node: &'a Sub<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [multiplication](Mul) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The multiplication.
	/// - `left`: The output of the multiplicand.
	/// - `right`: The output of the multiplier.
	fn visit_mul(
		&mut self,
		node: &'a Mul<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [division](Div) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The division.
	/// - `left`: The output of the dividend.
	/// - `right`: The output of the divisor.
	fn visit_div(
		&mut self,
		node: &'a Div<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [modulo](Mod) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The modulo expression.
	/// - `left`: The output of the dividend.
	/// - `right`: The output of the divisor.
	fn visit_mod(
		&mut self,
		node: &'a Mod<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit an [exponentiation](Exp) expression, after its operands.
	///
	/// # Parameters
	/// - `node`: The exponentiation.
	/// - `left`: The output of the base.
	/// - `right`: The output of the exponent.
	fn visit_exp(
		&mut self,
		node: &'a Exp<'src>,
		left: Self::Output,
		right: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Visit a [negation](Neg) expression, after its operand.
	///
	/// # Parameters
	/// - `node`: The negation.
	/// - `operand`: The output of the operand.
	fn visit_neg(
		&mut self,
		node: &'a Neg<'src>,
		operand: Self::Output
	) -> Result<Self::Output, Self::Error>;

	/// Finish an [expression](Expression) that fills a slot of type
	/// [`Expression`], after the `visit_*` method of the node inside it.
	/// Every node but the [`DiceExpression`] beneath a
	/// [drop-lowest](DropLowest), [drop-highest](DropHighest),
	/// [keep-lowest](KeepLowest), [keep-highest](KeepHighest), or
	/// [reroll](Reroll) expression fills such a slot, as does the root of a
	/// walk that starts at an [`Expression`] or a [`Function`].
	///
	/// # Parameters
	/// - `node`: The expression.
	/// - `output`: The output of the node inside the expression.
	///
	/// # Returns
	/// The output of the expression. By default, `output` unchanged.
	fn visit_expression(
		&mut self,
		_node: &'a Expression<'src>,
		output: Self::Output
	) -> Result<Self::Output, Self::Error>
	{
		Ok(output)
	}
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
	/// Walk this arithmetic expression with the given [`ASTVisitor`],
	/// iteratively, as the [trait](ASTVisitor) describes. The walk does not
	/// call [`visit_expression`](ASTVisitor::visit_expression) for this
	/// arithmetic expression itself, since it is not an [`Expression`].
	///
	/// # Type parameters
	/// - `'a`: The lifetime of the borrowed arithmetic expression.
	/// - `V`: The type of the visitor.
	///
	/// # Parameters
	/// - `visitor`: The visitor.
	///
	/// # Returns
	/// The output of this arithmetic expression.
	///
	/// # Errors
	/// Propagates the first error returned by the visitor.
	pub fn accept<'a, V: ASTVisitor<'a, 'src>>(
		&'a self,
		visitor: &mut V
	) -> Result<V::Output, V::Error>
	{
		walk::fold(walk::Node::Arithmetic(self), visitor)
	}
}

impl Display for ArithmeticExpression<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		display::render(self, f)
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
			name: self.name.clone(),
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
			name: self.name.clone(),
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
			name: self.name.clone(),
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

	fn untethered(&self) -> Self { impls::untether(self) }
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

impl Spanned for KeepLowest<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		KeepLowest {
			dice: Box::new(self.dice.untethered()),
			keep: self.keep.as_ref().map(|k| Box::new(k.untethered())),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for KeepHighest<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		KeepHighest {
			dice: Box::new(self.dice.untethered()),
			keep: self.keep.as_ref().map(|k| Box::new(k.untethered())),
			span: SourceSpan::default()
		}
	}
}

impl Spanned for Reroll<'_>
{
	fn span(&self) -> SourceSpan { self.span }

	fn untethered(&self) -> Self
	{
		Reroll {
			dice: Box::new(self.dice.untethered()),
			predicate: self.predicate.map(|t| Box::new(t.untethered())),
			limit: self
				.limit
				.as_ref()
				.map(|limit| limit.map(|n| Box::new(n.untethered()))),
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
			DiceExpression::DropHighest(d) => d.span(),
			DiceExpression::KeepLowest(d) => d.span(),
			DiceExpression::KeepHighest(d) => d.span(),
			DiceExpression::Reroll(d) => d.span()
		}
	}

	fn untethered(&self) -> Self { impls::untether_dice(self) }
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
