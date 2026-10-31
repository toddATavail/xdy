//! # Parsing engine
//!
//! Herein is the engine that drives the recursive productions of the grammar,
//! from [`function`](super::function) down to [`group`](super::group) and
//! [`binding`](super::binding). Every public combinator on the recursive cycle
//! is a thin wrapper that [runs](run) the engine at its own [goal](Goal).
//!
//! The engine performs recursive descent with an explicit stack in place of
//! the machine stack. Wherever a production needs a subproduction, it pushes a
//! [frame](Frame) that records what it will do with the outcome, and _calls_
//! the subproduction. When the subproduction finishes, its outcome _returns_ to
//! the frame on top of the stack. So the stack of frames grows with the nesting
//! of the input, but the machine stack does not. Each frame keeps only what is
//! small, and the few that must keep something bulky keep it apart, as a
//! [payload](Payload), so that a deep nesting takes as little memory as it
//! can. The flat parts of the grammar, such as [constants](super::constant),
//! [variables](super::variable), punctuation, and whitespace, are ordinary
//! [`nom`] combinators, which the engine calls directly.
//!
//! The engine reproduces the behavior of the recursive combinators that it
//! replaced, exactly: every value, remaining input, and error, though the
//! [list](ParseError::errors) of a [`ParseError`] now keeps at most one entry
//! beyond those at its rightmost position. To that end, each production applies
//! the same [contexts](nom::error::context), [cuts](nom::combinator::cut), and
//! [alternations](nom::branch::alt) as the original, in the same order, as
//! documented alongside each frame. The one deliberate difference is in
//! [`primary`](super::primary), which the originals parsed in time exponential
//! in the nesting of its dice counts; see [`resume_primary_range`].
//!
//! The engine also has a _recovery mode_, [`run_recovering`], for the
//! [doctor](crate::diagnostics). Wherever the parse cannot continue, it
//! consults a [policy](Recovery), which may [repair](Repair) the failure: the
//! engine then continues as though the source had been edited to supply the
//! missing token, without parsing the source again. A failure is detected
//! where it becomes certain, while the stack still holds every production in
//! progress, so a repair never parses again what the productions in progress
//! have already parsed, and the whole run takes time linear in the length of
//! the input, however many repairs it makes. The
//! ordinary parse uses the [`Strict`] policy, for which the recovery hooks
//! compile away; see [`Recoverer`].

use std::borrow::Cow;

use nom::{
	IResult, Input as _, Parser as _,
	branch::alt,
	bytes::complete::tag,
	character::complete::{char, multispace0, one_of},
	combinator::{cut, fail as fail_parser, map},
	error::{ContextError, ErrorKind, ParseError as _, context},
	sequence::{preceded, terminated}
};

use crate::{
	ast::{
		Add, ArithmeticExpression, Binding, Constant, CustomDice,
		DiceExpression, Direction, Div, DropHighest, DropLowest, Exp, Explode,
		Expression, Extremum, Function, Group, KeepHighest, KeepLowest, Limit,
		Mod, Mul, Neg, Parameter, Predicate, Range, Reroll, StandardDice, Sub,
		Variable
	},
	classify::{Role, Token},
	span::{SourceSpan, Spanned}
};

use super::{
	ARGUMENT_CONTEXT, BINDING_CONTEXT, BINDING_EXPRESSION_CONTEXT,
	CLOSING_BRACE_CONTEXT, CLOSING_BRACKET_CONTEXT, CLOSING_PAREN_CONTEXT,
	CONSTANT_CONTEXT, CUSTOM_FACES_CONTEXT, DICE_CONTEXT, DICE_COUNT_CONTEXT,
	DROP_DIRECTION_CONTEXT, DROP_EXPRESSION_CONTEXT, EXPLODE_CONTEXTS,
	EXPRESSION_CONTEXT, FUNCTION_BODY_CONTEXT, FUNCTION_CONTEXT, GROUP_CONTEXT,
	IDENTIFIER_CONTEXT, KEEP_DIRECTION_CONTEXT, KEEP_EXPRESSION_CONTEXT,
	NEXT_PARAMETER_CONTEXT, NomErrorKind, PARAMETER_CONTEXT, ParseError,
	PredicatedContexts, RANGE_CONTEXT, RANGE_END_CONTEXT, RANGE_START_CONTEXT,
	REROLL_CONTEXTS, RIGHT_OPERAND_CONTEXT, RerollParts,
	STANDARD_FACES_CONTEXT, Span, VARIABLE_CONTEXT, begins_call,
	begins_misplaced_clause, begins_operand, begins_verb, braced_name,
	canonical_name, constant, custom_faces, d_operator, identifier, integer,
	is_token_space, keyword, name_space0, negative_constant, parameter,
	parameters
};

////////////////////////////////////////////////////////////////////////////////
//                                   Goals.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A production that the engine can parse, whether as the goal of a
/// [run] or as a subproduction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Goal
{
	/// [`function`](super::function), producing [`Value::Function`].
	Function,

	/// [`expression`](super::expression) or [`add_sub`](super::add_sub),
	/// producing [`Value::Expression`].
	AddSub,

	/// [`mul_div_mod`](super::mul_div_mod), producing [`Value::Expression`].
	MulDivMod,

	/// [`unary`](super::unary), producing [`Value::Expression`].
	Unary,

	/// [`exponent`](super::exponent), producing [`Value::Expression`].
	Exponent,

	/// [`primary`](super::primary), producing [`Value::Expression`].
	Primary,

	/// [`group`](super::group), producing [`Value::Group`].
	Group,

	/// [`binding`](super::binding), producing [`Value::Binding`].
	Binding,

	/// [`range`](super::range), producing [`Value::Range`].
	Range,

	/// [`dice`](super::dice), producing [`Value::Dice`].
	Dice,

	/// [`dice`](super::dice) as the second alternative of
	/// [`primary`](super::primary). If the dice count is not followed by a
	/// dice operator, this produces the dice count itself, as a
	/// [`Value::Expression`], rather than failing; see
	/// [`resume_primary_range`].
	PrimaryDice,

	/// [`standard_dice`](super::standard_dice), producing
	/// [`Value::StandardDice`].
	StandardDice,

	/// [`custom_dice`](super::custom_dice), producing [`Value::CustomDice`].
	CustomDice,

	/// [`dice_count`](super::dice_count) or
	/// [`drop_expression`](super::drop_expression), which are identical, and
	/// whose constant is unsigned, producing [`Value::Expression`]. The counts
	/// of keep clauses and of the limits of reroll and explode clauses are
	/// drop expressions too.
	Atom,

	/// [`standard_faces`](super::standard_faces), which is [`Goal::Atom`] but
	/// with a signed [integer] in place of the constant, producing
	/// [`Value::Expression`]. The thresholds and endpoints of the predicates
	/// of reroll and explode clauses are standard faces too.
	Faces,

	/// [`drop_lowest`](super::drop_lowest), producing [`Value::Drop`].
	DropLowest,

	/// [`drop_highest`](super::drop_highest), producing [`Value::Drop`].
	DropHighest,

	/// [`keep_lowest`](super::keep_lowest), producing [`Value::Drop`].
	KeepLowest,

	/// [`keep_highest`](super::keep_highest), producing [`Value::Drop`].
	KeepHighest,

	/// [`reroll`](super::reroll), producing [`Value::Predicated`].
	Reroll,

	/// [`explode`](super::explode), producing [`Value::Predicated`].
	Explode,

	/// [`extremum`](super::extremum), producing [`Value::Expression`]: the
	/// call, unless a [repair](Repair::Unwrap) replaced a call of one argument
	/// with that argument.
	Extremum
}

/// The value produced by a [goal](Goal).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
#[derive(Debug)]
pub(super) enum Value<'src>
{
	/// A function.
	Function(Function<'src>),

	/// An expression. Where a repair made a call the [dice
	/// count](Site::UngroupedCall) of a dice expression in place of a primary
	/// expression, the [dice expression](Value::Dice) stands for the
	/// expression instead; see [`into_expression`](Value::into_expression).
	Expression(Expression<'src>),

	/// A group.
	Group(Group<'src>),

	/// A binding.
	Binding(Binding<'src>),

	/// A range.
	Range(Range<'src>),

	/// A dice expression.
	Dice(DiceExpression<'src>),

	/// A standard dice expression.
	StandardDice(StandardDice<'src>),

	/// A custom dice expression.
	CustomDice(CustomDice<'src>),

	/// The optional drop expression of a drop or keep clause, i.e., its count.
	Drop(Option<Box<Expression<'src>>>),

	/// The predicate and the optional limit of a reroll or explode clause.
	Predicated
	{
		/// The predicate.
		predicate: Predicate<Box<Expression<'src>>>,

		/// The limit, if any.
		limit: Option<Limit<Box<Expression<'src>>>>
	}
}

impl<'src> Value<'src>
{
	/// Answer the function.
	///
	/// # Returns
	/// The function.
	///
	/// # Panics
	/// If the receiver is not a [function](Value::Function).
	pub(super) fn into_function(self) -> Function<'src>
	{
		match self
		{
			Value::Function(function) => function,
			value => unreachable!("expected a function: {:?}", value)
		}
	}

	/// Answer the expression. A [dice expression](Value::Dice) is an
	/// expression too, which a primary expression produces where a repair made
	/// a call the [dice count](Site::UngroupedCall) of a dice expression.
	///
	/// # Returns
	/// The expression.
	///
	/// # Panics
	/// If the receiver is neither an [expression](Value::Expression) nor a
	/// [dice expression](Value::Dice).
	pub(super) fn into_expression(self) -> Expression<'src>
	{
		match self
		{
			Value::Expression(expression) => expression,
			Value::Dice(dice) => Expression::Dice(dice),
			value => unreachable!("expected an expression: {:?}", value)
		}
	}

	/// Answer the call of `highest` or `lowest`.
	///
	/// # Returns
	/// The call.
	///
	/// # Panics
	/// If the receiver is not an [expression](Value::Expression) that is a
	/// call.
	pub(super) fn into_extremum(self) -> Extremum<'src>
	{
		match self
		{
			// `Expression` implements `Drop`, so the call cannot move out of
			// it; take its arguments instead.
			Value::Expression(mut expression) => match &mut expression
			{
				Expression::Extremum(call) => Extremum {
					direction: call.direction,
					arguments: std::mem::take(&mut call.arguments),
					span: call.span
				},
				_ => unreachable!("expected a call: {:?}", expression)
			},
			value => unreachable!("expected a call: {:?}", value)
		}
	}

	/// Answer the group.
	///
	/// # Returns
	/// The group.
	///
	/// # Panics
	/// If the receiver is not a [group](Value::Group).
	pub(super) fn into_group(self) -> Group<'src>
	{
		match self
		{
			Value::Group(group) => group,
			value => unreachable!("expected a group: {:?}", value)
		}
	}

	/// Answer the binding.
	///
	/// # Returns
	/// The binding.
	///
	/// # Panics
	/// If the receiver is not a [binding](Value::Binding).
	pub(super) fn into_binding(self) -> Binding<'src>
	{
		match self
		{
			Value::Binding(binding) => binding,
			value => unreachable!("expected a binding: {:?}", value)
		}
	}

	/// Answer the range.
	///
	/// # Returns
	/// The range.
	///
	/// # Panics
	/// If the receiver is not a [range](Value::Range).
	pub(super) fn into_range(self) -> Range<'src>
	{
		match self
		{
			Value::Range(range) => range,
			value => unreachable!("expected a range: {:?}", value)
		}
	}

	/// Answer the dice expression.
	///
	/// # Returns
	/// The dice expression.
	///
	/// # Panics
	/// If the receiver is not a [dice expression](Value::Dice).
	pub(super) fn into_dice(self) -> DiceExpression<'src>
	{
		match self
		{
			Value::Dice(dice) => dice,
			value => unreachable!("expected a dice expression: {:?}", value)
		}
	}

	/// Answer the standard dice expression.
	///
	/// # Returns
	/// The standard dice expression.
	///
	/// # Panics
	/// If the receiver is not a [standard dice
	/// expression](Value::StandardDice).
	pub(super) fn into_standard_dice(self) -> StandardDice<'src>
	{
		match self
		{
			Value::StandardDice(dice) => dice,
			value => unreachable!("expected standard dice: {:?}", value)
		}
	}

	/// Answer the custom dice expression.
	///
	/// # Returns
	/// The custom dice expression.
	///
	/// # Panics
	/// If the receiver is not a [custom dice expression](Value::CustomDice).
	pub(super) fn into_custom_dice(self) -> CustomDice<'src>
	{
		match self
		{
			Value::CustomDice(dice) => dice,
			value => unreachable!("expected custom dice: {:?}", value)
		}
	}

	/// Answer the optional drop expression.
	///
	/// # Returns
	/// The optional drop expression.
	///
	/// # Panics
	/// If the receiver is not a [drop expression](Value::Drop).
	pub(super) fn into_drop(self) -> Option<Box<Expression<'src>>>
	{
		match self
		{
			Value::Drop(drop) => drop,
			value => unreachable!("expected a drop expression: {:?}", value)
		}
	}

	/// Answer the predicate and the optional limit of a reroll or explode
	/// clause.
	///
	/// # Returns
	/// The predicate and the limit.
	///
	/// # Panics
	/// If the receiver is not a [reroll or explode clause](Value::Predicated).
	pub(super) fn into_predicated(self) -> RerollParts<'src>
	{
		match self
		{
			Value::Predicated { predicate, limit } => (predicate, limit),
			value =>
			{
				unreachable!("expected a reroll or explode clause: {:?}", value)
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Frames.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The outcome of a [goal](Goal): the remaining input and the
/// [value](Value), or an error.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
pub(super) type Outcome<'src> =
	IResult<Span<'src>, Value<'src>, ParseError<'src>>;

/// A step of the engine.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
enum Step<'src>
{
	/// Parse a goal at some input, and return its outcome to the frame on top
	/// of the stack.
	Call(Goal, Span<'src>),

	/// Return an outcome to the frame on top of the stack, or finish the run if
	/// the stack is empty.
	Return(Outcome<'src>)
}

/// A binary operator level, which parses a sequence of operands separated by
/// operators of the same precedence, and folds them to the left.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Level
{
	/// [`add_sub`](super::add_sub): `+` and `-` over
	/// [`mul_div_mod`](super::mul_div_mod).
	Additive,

	/// [`mul_div_mod`](super::mul_div_mod): `*`, `×`, `/`, `÷`, and `%` over
	/// [`unary`](super::unary).
	Multiplicative
}

impl Level
{
	/// Answer the operators of the level.
	///
	/// # Returns
	/// The operators, as expected by [`one_of`].
	fn operators(self) -> &'static str
	{
		match self
		{
			Level::Additive => "+-",
			Level::Multiplicative => "*×/÷%"
		}
	}

	/// Answer the goal that parses an operand of the level.
	///
	/// # Returns
	/// The goal.
	fn operand(self) -> Goal
	{
		match self
		{
			Level::Additive => Goal::MulDivMod,
			Level::Multiplicative => Goal::Unary
		}
	}

	/// Answer the role of the operators of the level.
	///
	/// # Returns
	/// The role.
	fn role(self) -> Role
	{
		match self
		{
			Level::Additive => Role::Additive,
			Level::Multiplicative => Role::Multiplicative
		}
	}
}

/// The kind of a drop or keep clause.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ClauseKind
{
	/// `drop lowest`.
	DropLowest,

	/// `drop highest`.
	DropHighest,

	/// `keep lowest`.
	KeepLowest,

	/// `keep highest`.
	KeepHighest
}

impl ClauseKind
{
	/// Answer the kind of a clause from its verb and its direction.
	///
	/// # Parameters
	/// - `keep`: Whether the verb is `keep` rather than `drop`.
	/// - `lowest`: Whether the direction is `lowest` rather than `highest`.
	///
	/// # Returns
	/// The kind.
	const fn of(keep: bool, lowest: bool) -> Self
	{
		match (keep, lowest)
		{
			(false, true) => ClauseKind::DropLowest,
			(false, false) => ClauseKind::DropHighest,
			(true, true) => ClauseKind::KeepLowest,
			(true, false) => ClauseKind::KeepHighest
		}
	}

	/// Answer whether a clause of this kind keeps rather than drops.
	///
	/// # Returns
	/// `true` for a keep, `false` for a drop.
	const fn keeps(self) -> bool
	{
		matches!(self, ClauseKind::KeepLowest | ClauseKind::KeepHighest)
	}

	/// Answer the keywords of a clause of this kind.
	///
	/// # Returns
	/// The verb, `drop` or `keep`, and the direction, `lowest` or `highest`.
	const fn keywords(self) -> (&'static str, &'static str)
	{
		match self
		{
			ClauseKind::DropLowest => ("drop", "lowest"),
			ClauseKind::DropHighest => ("drop", "highest"),
			ClauseKind::KeepLowest => ("keep", "lowest"),
			ClauseKind::KeepHighest => ("keep", "highest")
		}
	}

	/// Answer the context classifier of the count of a clause of this kind.
	///
	/// # Returns
	/// The classifier.
	const fn count_context(self) -> &'static str
	{
		if self.keeps()
		{
			KEEP_EXPRESSION_CONTEXT
		}
		else
		{
			DROP_EXPRESSION_CONTEXT
		}
	}

	/// Build a clause of this kind.
	///
	/// # Parameters
	/// - `dice`: The dice expression that the clause modifies.
	/// - `count`: The count of the clause, if any.
	/// - `span`: The span of the clause.
	///
	/// # Returns
	/// The clause.
	fn build<'src>(
		self,
		dice: DiceExpression<'src>,
		count: Option<Box<Expression<'src>>>,
		span: SourceSpan
	) -> DiceExpression<'src>
	{
		let dice = Box::new(dice);
		match self
		{
			ClauseKind::DropLowest => DiceExpression::DropLowest(DropLowest {
				dice,
				drop: count,
				span
			}),
			ClauseKind::DropHighest =>
			{
				DiceExpression::DropHighest(DropHighest {
					dice,
					drop: count,
					span
				})
			},
			ClauseKind::KeepLowest => DiceExpression::KeepLowest(KeepLowest {
				dice,
				keep: count,
				span
			}),
			ClauseKind::KeepHighest =>
			{
				DiceExpression::KeepHighest(KeepHighest {
					dice,
					keep: count,
					span
				})
			},
		}
	}
}

/// The kind of a reroll or explode clause, which share their predicates and
/// their limits, and differ only in their verbs and what they build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PredicatedKind
{
	/// `reroll`.
	Reroll,

	/// `explode`.
	Explode
}

impl PredicatedKind
{
	/// Answer the verb of a clause of this kind.
	///
	/// # Returns
	/// `reroll` or `explode`.
	pub(crate) const fn keyword(self) -> &'static str
	{
		match self
		{
			PredicatedKind::Reroll => "reroll",
			PredicatedKind::Explode => "explode"
		}
	}

	/// Answer the role of the verb of a clause of this kind.
	///
	/// # Returns
	/// [`Role::Reroll`] or [`Role::Explode`].
	const fn role(self) -> Role
	{
		match self
		{
			PredicatedKind::Reroll => Role::Reroll,
			PredicatedKind::Explode => Role::Explode
		}
	}

	/// Answer the context classifiers of the parts of a clause of this kind.
	///
	/// # Returns
	/// The classifiers.
	pub(crate) const fn contexts(self) -> &'static PredicatedContexts
	{
		match self
		{
			PredicatedKind::Reroll => &REROLL_CONTEXTS,
			PredicatedKind::Explode => &EXPLODE_CONTEXTS
		}
	}

	/// Answer the threshold of the predicate, `=`, that a [repair](Repair::Fix)
	/// supplies wherever a clause of this kind lacks its predicate: for a
	/// reroll, [`PLACEHOLDER_THRESHOLD`], since a reroll most often rerolls
	/// ones; for an explode, the highest face of its dice, since an explode
	/// most often explodes the highest face, or [`PLACEHOLDER_FACES`], the
	/// faces of the commonest dice, if the faces are not constant.
	///
	/// # Parameters
	/// - `highest`: The [highest face](highest_face) of the dice, if known.
	///
	/// # Returns
	/// The threshold.
	pub(crate) const fn placeholder_threshold(self, highest: Option<i32>)
	-> i32
	{
		match (self, highest)
		{
			(PredicatedKind::Reroll, _) => PLACEHOLDER_THRESHOLD,
			(PredicatedKind::Explode, Some(highest)) => highest,
			(PredicatedKind::Explode, None) => PLACEHOLDER_FACES
		}
	}

	/// Build a clause of this kind.
	///
	/// # Parameters
	/// - `dice`: The dice expression that the clause modifies.
	/// - `predicate`: The predicate.
	/// - `limit`: The limit, if any.
	/// - `span`: The span of the clause.
	///
	/// # Returns
	/// The clause.
	fn build<'src>(
		self,
		dice: DiceExpression<'src>,
		predicate: Predicate<Box<Expression<'src>>>,
		limit: Option<Limit<Box<Expression<'src>>>>,
		span: SourceSpan
	) -> DiceExpression<'src>
	{
		let dice = Box::new(dice);
		match self
		{
			PredicatedKind::Reroll => DiceExpression::Reroll(Reroll {
				dice,
				predicate,
				limit,
				span
			}),
			PredicatedKind::Explode => DiceExpression::Explode(Explode {
				dice,
				predicate,
				limit,
				span
			})
		}
	}
}

/// The comparator of the predicate of a reroll or explode clause.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Comparison
{
	/// `=`.
	Eq,

	/// `!=` or `≠`.
	Ne,

	/// `<`.
	Lt,

	/// `<=` or `≤`.
	Le,

	/// `>`.
	Gt,

	/// `>=` or `≥`.
	Ge
}

impl Comparison
{
	/// Build a predicate that compares a result with a threshold by this
	/// comparator.
	///
	/// # Parameters
	/// - `threshold`: The threshold.
	///
	/// # Returns
	/// The predicate.
	fn predicate<'src>(
		self,
		threshold: Expression<'src>
	) -> Predicate<Box<Expression<'src>>>
	{
		let threshold = Box::new(threshold);
		match self
		{
			Comparison::Eq => Predicate::Eq(threshold),
			Comparison::Ne => Predicate::Ne(threshold),
			Comparison::Lt => Predicate::Lt(threshold),
			Comparison::Le => Predicate::Le(threshold),
			Comparison::Gt => Predicate::Gt(threshold),
			Comparison::Ge => Predicate::Ge(threshold)
		}
	}
}

/// A suspended production, awaiting the outcome of a subproduction. Each
/// variant is named for the production and the subproduction that it awaits.
/// Fields named `input` hold the input at which the awaited subproduction
/// began, which is the position of any context that the production attaches
/// to its errors.
///
/// The stack holds a frame for every production in progress, which is several
/// for every level of the nesting of the input, so a frame keeps only what is
/// small. Whatever is bulky, such as an operand already parsed, a frame that
/// [carries](Frame::carries) it keeps in a [payload](Payload) on the
/// [stack](Stack) instead, as documented alongside each variant.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
enum Frame<'src>
{
	/// [`Goal::Function`], awaiting the body. It carries the formal
	/// [parameters](Payload::Parameters).
	FunctionBody
	{
		/// The start of the function.
		start: usize,

		/// Whether the function has formal parameters.
		formal: bool,

		/// The input of the body.
		input: Span<'src>
	},

	/// [`Goal::AddSub`] or [`Goal::MulDivMod`], awaiting the first operand.
	BinaryFirst(Level),

	/// [`Goal::AddSub`] or [`Goal::MulDivMod`], awaiting a right operand. It
	/// carries the fold of the operands so far, as an
	/// [expression](Payload::Expression).
	BinaryRight
	{
		/// The level.
		level: Level,

		/// The operator before the awaited operand.
		operator: char,

		/// The input of the awaited operand.
		input: Span<'src>
	},

	/// [`Goal::Unary`], awaiting the operand of a negation, its second
	/// alternative. It holds the input of the unary expression. The error of
	/// the first alternative is [recomputed](negation_error) if the second
	/// fails.
	UnaryNegation(Span<'src>),

	/// [`Goal::Unary`], awaiting an exponentiation, its third alternative. If
	/// the negation failed after its `-`, it carries the merged
	/// [errors](Payload::Error) of the first two alternatives. Otherwise, both
	/// failed at the input, and their errors are [recomputed](unary_error) if
	/// the third fails.
	UnaryExponent
	{
		/// The input of the unary expression.
		input: Span<'src>,

		/// Whether the frame carries the merged errors.
		merged: bool
	},

	/// [`Goal::Exponent`], awaiting the base.
	ExponentBase,

	/// [`Goal::Exponent`], awaiting the power. It holds the input of the power,
	/// and carries the base, as an [expression](Payload::Expression).
	ExponentPower(Span<'src>),

	/// [`Goal::Primary`], awaiting a range, its first alternative.
	PrimaryRange(Span<'src>),

	/// [`Goal::Primary`], awaiting a call of `highest` or `lowest`, which it
	/// tries after a range, wherever the input [begins a call](begins_call).
	/// It holds the input of the primary expression.
	PrimaryCall(Span<'src>),

	/// [`Goal::Primary`], awaiting dice, its second alternative. It holds the
	/// input of the primary expression. The error of the first alternative is
	/// [recomputed](range_error) if the second fails.
	PrimaryDice(Span<'src>),

	/// [`Goal::Primary`], awaiting a group, its third alternative. It holds the
	/// input of the primary expression, and carries the merged
	/// [errors](Payload::Error) of the first two alternatives.
	PrimaryGroup(Span<'src>),

	/// [`Goal::Atom`] or [`Goal::Faces`], awaiting a group, its third
	/// alternative. The merged errors of the first two alternatives are
	/// [recomputed](atom_error) if the third fails.
	AtomGroup
	{
		/// The input of the atom.
		input: Span<'src>,

		/// Whether the goal is [`Goal::Faces`], whose literal is signed.
		signed: bool
	},

	/// [`Goal::Atom`] or [`Goal::Faces`], awaiting a call of `highest` or
	/// `lowest` that stands ungrouped where only an atom may, which a repair at
	/// [`Site::UngroupedCall`] reads as though grouped. It holds the input of
	/// the atom. Only a recovering parse pushes it.
	AtomCall(Span<'src>),

	/// [`Goal::Extremum`], awaiting an argument. It carries the
	/// [call](Payload::Call) so far.
	CallArgument
	{
		/// The byte offset of the `(` of the call.
		opener: usize,

		/// The input of the argument.
		input: Span<'src>
	},

	/// [`Goal::Group`], awaiting the grouped expression.
	GroupExpression
	{
		/// The start of the group.
		start: usize,

		/// The input of the grouped expression.
		input: Span<'src>
	},

	/// [`Goal::Binding`], or the variable alternative of [`Goal::Atom`] or
	/// [`Goal::Primary`] continued as a binding, awaiting the bound expression.
	/// It carries the [head](Payload::Head) of the binding.
	BindingExpression
	{
		/// The byte offset of the `(` of the binding, or of where a repair
		/// supplied it.
		paren: usize,

		/// The input of the bound expression.
		input: Span<'src>
	},

	/// [`Goal::Range`], awaiting the start of the range.
	RangeStart
	{
		/// The start of the range expression.
		start: usize,

		/// The input of the start of the range.
		input: Span<'src>
	},

	/// [`Goal::Range`], awaiting the end of the range. It carries the start of
	/// the range, as an [expression](Payload::Expression).
	RangeEnd
	{
		/// The start of the range expression.
		start: usize,

		/// The input of the end of the range.
		input: Span<'src>
	},

	/// [`Goal::Dice`] or [`Goal::PrimaryDice`], awaiting the dice count.
	DiceCount
	{
		/// Whether the goal is [`Goal::PrimaryDice`].
		primary: bool,

		/// The input of the dice expression, and of the dice count.
		input: Span<'src>
	},

	/// [`Goal::Dice`] or [`Goal::PrimaryDice`], awaiting standard faces. It
	/// carries the dice count, as an [expression](Payload::Expression).
	DiceFaces
	{
		/// The start of the dice expression.
		start: usize,

		/// The input of the faces.
		input: Span<'src>
	},

	/// [`Goal::Dice`] or [`Goal::PrimaryDice`], awaiting the drop expression
	/// of a drop or keep clause. It carries the dice expression that the
	/// clause modifies, and the input after the direction, as a
	/// [drop](Payload::Drop).
	DiceDrop
	{
		/// The kind of the clause.
		kind: ClauseKind,

		/// The input of the drop expression.
		input: Span<'src>
	},

	/// [`Goal::StandardDice`], awaiting the dice count.
	StandardCount(Span<'src>),

	/// [`Goal::StandardDice`], awaiting the faces. It holds the input of the
	/// faces, and carries the dice count, as an
	/// [expression](Payload::Expression).
	StandardFaces(Span<'src>),

	/// [`Goal::CustomDice`], awaiting the dice count.
	CustomCount(Span<'src>),

	/// [`Goal::DropLowest`], [`Goal::DropHighest`], [`Goal::KeepLowest`], or
	/// [`Goal::KeepHighest`], awaiting the drop expression. It carries the
	/// input after the direction, where the clause ends if it has no drop
	/// expression, as a [rest](Payload::Rest).
	DropExpression
	{
		/// The kind of the clause.
		kind: ClauseKind,

		/// The input of the drop expression.
		input: Span<'src>
	},

	/// A reroll or explode clause of [`Goal::Dice`], [`Goal::PrimaryDice`],
	/// [`Goal::Reroll`], or [`Goal::Explode`], awaiting the threshold of a
	/// comparison, or the start of a range. It carries the
	/// [clause](Payload::Predicated).
	PredicateFirst
	{
		/// The kind of the clause.
		kind: PredicatedKind,

		/// The comparator, or `None` if the predicate is a range.
		comparison: Option<Comparison>,

		/// The input of the threshold or the start.
		input: Span<'src>
	},

	/// A reroll or explode clause, awaiting the end of a range. It carries the
	/// [clause](Payload::Predicated), which holds the start of the range.
	PredicateEnd
	{
		/// The kind of the clause.
		kind: PredicatedKind,

		/// Whether the range is inclusive, `..=`, rather than half-open, `..`.
		inclusive: bool,

		/// The input of the end.
		input: Span<'src>
	},

	/// A reroll or explode clause, awaiting the optional count of its limit.
	/// It carries the [clause](Payload::Predicated), which holds the predicate
	/// and the input after it, where the clause ends if it has no limit.
	LimitCount
	{
		/// The kind of the clause.
		kind: PredicatedKind,

		/// The input of the count.
		input: Span<'src>
	}
}

// A frame is at most as large as a span and an offset, and a tag.
const _: () = assert!(size_of::<Frame<'static>>() <= 48);

impl Frame<'_>
{
	/// Answer whether the frame carries a [payload](Payload).
	///
	/// # Returns
	/// `true` if the frame carries a payload, `false` otherwise.
	fn carries(&self) -> bool
	{
		match self
		{
			Frame::FunctionBody { .. }
			| Frame::BinaryRight { .. }
			| Frame::ExponentPower(_)
			| Frame::PrimaryGroup(_)
			| Frame::BindingExpression { .. }
			| Frame::RangeEnd { .. }
			| Frame::DiceFaces { .. }
			| Frame::DiceDrop { .. }
			| Frame::StandardFaces(_)
			| Frame::DropExpression { .. }
			| Frame::PredicateFirst { .. }
			| Frame::PredicateEnd { .. }
			| Frame::LimitCount { .. }
			| Frame::CallArgument { .. } => true,
			Frame::UnaryExponent { merged, .. } => *merged,
			Frame::BinaryFirst(_)
			| Frame::UnaryNegation(_)
			| Frame::ExponentBase
			| Frame::PrimaryRange(_)
			| Frame::PrimaryCall(_)
			| Frame::PrimaryDice(_)
			| Frame::AtomGroup { .. }
			| Frame::AtomCall(_)
			| Frame::GroupExpression { .. }
			| Frame::RangeStart { .. }
			| Frame::DiceCount { .. }
			| Frame::StandardCount(_)
			| Frame::CustomCount(_) => false
		}
	}

	/// Answer where a frame that [catches](Disposition::Catch) a recoverable
	/// error began the alternative that it abandons, from which the engine
	/// [retracts](TokenSink::retract) the tokens that the alternative read.
	///
	/// # Returns
	/// The byte offset of the input of the frame, if the frame catches, or
	/// `None` otherwise.
	fn catches_from(&self) -> Option<usize>
	{
		let from = match self
		{
			Frame::UnaryNegation(input)
			| Frame::PrimaryRange(input)
			| Frame::PrimaryDice(input)
			| Frame::PrimaryGroup(input)
			| Frame::DiceFaces { input, .. }
			| Frame::DiceDrop { input, .. }
			| Frame::DropExpression { input, .. }
			| Frame::LimitCount { input, .. } => Some(input.location_offset()),
			_ => None
		};
		debug_assert_eq!(
			from.is_some(),
			Disposition::of(self) == Disposition::Catch,
			"every frame that catches, and only those, retracts"
		);
		from
	}
}

/// The head of a [binding](Binding), everything before its bound expression,
/// which [`Frame::BindingExpression`] carries until the bound expression is
/// done.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
#[derive(Clone)]
struct BindingHead<'src>
{
	/// The start of the binding.
	start: usize,

	/// The [canonical](canonical_name) bound name.
	name: Cow<'src, str>,

	/// The span of the bound name.
	name_span: SourceSpan,

	/// The input of the atom or primary expression, if the binding is the
	/// variable alternative of one, continued; see [`variable_or_binding`].
	atom: Option<Span<'src>>
}

/// The bulky part of a suspended production, which the [frame](Frame) that
/// [carries](Frame::carries) it keeps on the [stack](Stack) apart from itself.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
enum Payload<'src>
{
	/// The formal parameters of [`Frame::FunctionBody`].
	Parameters(Option<Vec<Parameter<'src>>>),

	/// An expression already parsed: the fold of the operands of
	/// [`Frame::BinaryRight`], the base of [`Frame::ExponentPower`], the start
	/// of [`Frame::RangeEnd`], or the dice count of [`Frame::DiceFaces`] or
	/// [`Frame::StandardFaces`].
	Expression(Expression<'src>),

	/// The merged errors of the alternatives before [`Frame::UnaryExponent`]
	/// or [`Frame::PrimaryGroup`], which failed beyond their input, and so
	/// cannot be recomputed from it.
	Error(ParseError<'src>),

	/// The head of the binding of [`Frame::BindingExpression`].
	Head(BindingHead<'src>),

	/// The dice expression that the clause of [`Frame::DiceDrop`] modifies,
	/// the highest face of its dice, and the input after the direction, where
	/// the clause ends if it has no drop expression.
	Drop
	{
		/// The dice expression.
		dice: DiceExpression<'src>,

		/// The [highest face](highest_face) of the dice, if known.
		highest: Option<i32>,

		/// The input after the direction.
		rest: Span<'src>
	},

	/// The input after the direction of [`Frame::DropExpression`], where the
	/// clause ends if it has no drop expression.
	Rest(Span<'src>),

	/// The parts of the reroll or explode clause of [`Frame::PredicateFirst`],
	/// [`Frame::PredicateEnd`], or [`Frame::LimitCount`] read so far. They are
	/// boxed, lest they enlarge every payload.
	Predicated(Box<PredicatedClause<'src>>),

	/// The call of [`Frame::CallArgument`] so far. It is boxed, lest it
	/// enlarge every payload.
	Call(Box<Call<'src>>)
}

/// A call of `highest` or `lowest` so far, which [`Frame::CallArgument`]
/// [carries](Payload::Call).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
struct Call<'src>
{
	/// The start of the call, at its keyword.
	start: usize,

	/// The byte offset of the `(` of the call.
	opener: usize,

	/// The direction of the call.
	direction: Direction,

	/// The arguments read so far.
	arguments: Vec<Expression<'src>>,

	/// The span of the first argument, as written, which differs from the
	/// span of the argument where a repair replaced a call there with its own
	/// argument.
	first: SourceSpan
}

/// The parts of a reroll or explode clause read so far, which the frames of
/// the clause [carry](Payload::Predicated) in turn.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
struct PredicatedClause<'src>
{
	/// The dice expression that the clause modifies, and the [highest
	/// face](highest_face) of its dice, if known; or `None` if the clause is
	/// parsed alone, as [`Goal::Reroll`] or [`Goal::Explode`].
	dice: Option<(DiceExpression<'src>, Option<i32>)>,

	/// The start of a range, once read, which [`Frame::PredicateEnd`] holds.
	start: Option<Expression<'src>>,

	/// The predicate, and the input after it, where the clause ends if it has
	/// no limit, once read, which [`Frame::LimitCount`] holds.
	predicate: Option<(Predicate<Box<Expression<'src>>>, Span<'src>)>
}

impl<'src> Payload<'src>
{
	/// Answer the formal parameters.
	///
	/// # Returns
	/// The formal parameters.
	///
	/// # Panics
	/// If the payload is not [`Payload::Parameters`].
	fn into_parameters(self) -> Option<Vec<Parameter<'src>>>
	{
		match self
		{
			Payload::Parameters(parameters) => parameters,
			_ => unreachable!("the payload is not the formal parameters")
		}
	}

	/// Answer the expression.
	///
	/// # Returns
	/// The expression.
	///
	/// # Panics
	/// If the payload is not [`Payload::Expression`].
	fn into_expression(self) -> Expression<'src>
	{
		match self
		{
			Payload::Expression(expression) => expression,
			_ => unreachable!("the payload is not an expression")
		}
	}

	/// Answer the merged errors.
	///
	/// # Returns
	/// The merged errors.
	///
	/// # Panics
	/// If the payload is not [`Payload::Error`].
	fn into_error(self) -> ParseError<'src>
	{
		match self
		{
			Payload::Error(error) => error,
			_ => unreachable!("the payload is not an error")
		}
	}

	/// Answer the head of the binding.
	///
	/// # Returns
	/// The head.
	///
	/// # Panics
	/// If the payload is not [`Payload::Head`].
	fn into_head(self) -> BindingHead<'src>
	{
		match self
		{
			Payload::Head(head) => head,
			_ => unreachable!("the payload is not the head of a binding")
		}
	}

	/// Answer the dice expression, the highest face of its dice, and the input
	/// after the direction of a drop clause.
	///
	/// # Returns
	/// The dice expression, the highest face, and the input.
	///
	/// # Panics
	/// If the payload is not [`Payload::Drop`].
	fn into_drop(self) -> (DiceExpression<'src>, Option<i32>, Span<'src>)
	{
		match self
		{
			Payload::Drop {
				dice,
				highest,
				rest
			} => (dice, highest, rest),
			_ => unreachable!("the payload is not a drop clause")
		}
	}

	/// Answer the input after the direction of a drop expression.
	///
	/// # Returns
	/// The input.
	///
	/// # Panics
	/// If the payload is not [`Payload::Rest`].
	fn into_rest(self) -> Span<'src>
	{
		match self
		{
			Payload::Rest(rest) => rest,
			_ => unreachable!("the payload is not the rest of a drop")
		}
	}

	/// Answer the parts of a reroll or explode clause read so far.
	///
	/// # Returns
	/// The parts.
	///
	/// # Panics
	/// If the payload is not [`Payload::Predicated`].
	fn into_predicated(self) -> Box<PredicatedClause<'src>>
	{
		match self
		{
			Payload::Predicated(clause) => clause,
			_ => unreachable!("the payload is not a reroll or explode clause")
		}
	}

	/// Answer the call so far.
	///
	/// # Returns
	/// The call.
	///
	/// # Panics
	/// If the payload is not [`Payload::Call`].
	fn into_call(self) -> Box<Call<'src>>
	{
		match self
		{
			Payload::Call(call) => call,
			_ => unreachable!("the payload is not a call")
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Stack.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The stack of suspended productions: their [frames](Frame), and, apart from
/// them, the [payloads](Payload) of the frames that [carry](Frame::carries)
/// them, in the same order. The payloads are much larger than the frames, but
/// few frames carry one: in a deep nesting of groups, none does. Neither stack
/// allocates but to grow.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
struct Stack<'src>
{
	/// The frames, from the bottom of the stack to the top.
	frames: Vec<Frame<'src>>,

	/// The payloads of the frames that carry them, in the same order.
	payloads: Vec<Payload<'src>>
}

impl<'src> Stack<'src>
{
	/// Construct an empty stack.
	///
	/// # Returns
	/// The stack.
	fn new() -> Self
	{
		Self {
			frames: Vec::new(),
			payloads: Vec::new()
		}
	}

	/// Answer the number of frames on the stack.
	///
	/// # Returns
	/// The number of frames.
	fn len(&self) -> usize { self.frames.len() }

	/// Answer whether the stack is empty.
	///
	/// # Returns
	/// `true` if the stack holds no frames, `false` otherwise.
	fn is_empty(&self) -> bool { self.frames.is_empty() }

	/// Answer the frame at the bottom of the stack.
	///
	/// # Returns
	/// The frame, or `None` if the stack is empty.
	fn first(&self) -> Option<&Frame<'src>> { self.frames.first() }

	/// Answer the frame on top of the stack.
	///
	/// # Returns
	/// The frame, or `None` if the stack is empty.
	fn last(&self) -> Option<&Frame<'src>> { self.frames.last() }

	/// Push a frame that carries no payload.
	///
	/// # Parameters
	/// - `frame`: The frame.
	fn push(&mut self, frame: Frame<'src>)
	{
		debug_assert!(!frame.carries(), "the frame carries a payload");
		self.frames.push(frame);
	}

	/// Push a frame that carries a payload, and its payload.
	///
	/// # Parameters
	/// - `frame`: The frame.
	/// - `payload`: The payload.
	fn push_with(&mut self, frame: Frame<'src>, payload: Payload<'src>)
	{
		debug_assert!(frame.carries(), "the frame carries no payload");
		self.frames.push(frame);
		self.payloads.push(payload);
	}

	/// Pop the frame on top of the stack. If it [carries](Frame::carries) a
	/// payload, then the caller must [pop](Self::pop_payload) that next,
	/// before it uses the stack otherwise. Most frames carry none, so the
	/// payload does not travel with the frame.
	///
	/// # Returns
	/// The frame, or `None` if the stack is empty.
	fn pop(&mut self) -> Option<Frame<'src>> { self.frames.pop() }

	/// Pop the payload of the frame just [popped](Self::pop).
	///
	/// # Returns
	/// The payload.
	///
	/// # Panics
	/// If no frame on the stack carries a payload.
	fn pop_payload(&mut self) -> Payload<'src>
	{
		self.payloads.pop().expect("the frame carries a payload")
	}

	/// Discard every frame above the specified number of them, and their
	/// payloads.
	///
	/// # Parameters
	/// - `len`: The number of frames to keep. If the stack holds no more than
	///   this, then nothing happens.
	fn truncate(&mut self, len: usize)
	{
		let Some(discarded) = self.frames.get(len..)
		else
		{
			return;
		};
		let carried = discarded.iter().filter(|frame| frame.carries()).count();
		self.frames.truncate(len);
		self.payloads.truncate(self.payloads.len() - carried);
	}

	/// Discard every frame, and every payload.
	fn clear(&mut self)
	{
		self.frames.clear();
		self.payloads.clear();
	}

	/// Answer the frames, from the top of the stack to the bottom, each with
	/// its payload, if it carries one.
	///
	/// # Returns
	/// The frames and their payloads.
	fn iter_rev(
		&self
	) -> impl Iterator<Item = (&Frame<'src>, Option<&Payload<'src>>)>
	{
		let mut end = self.payloads.len();
		self.frames.iter().rev().map(move |frame| {
			let payload = frame.carries().then(|| {
				end -= 1;
				&self.payloads[end]
			});
			(frame, payload)
		})
	}
}

impl<'src> std::ops::Index<usize> for Stack<'src>
{
	type Output = Frame<'src>;

	fn index(&self, index: usize) -> &Self::Output { &self.frames[index] }
}

////////////////////////////////////////////////////////////////////////////////
//                                  Driver.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Parse the specified goal at the specified input, without leading whitespace.
///
/// The engine alternates between two kinds of [step](Step). A call starts a
/// production, which either finishes at once or pushes a [frame](Frame) and
/// calls a subproduction. A return pops the frame on top of the stack and
/// resumes its production with the outcome, which again either finishes or
/// calls another subproduction. The run ends when an outcome returns to an
/// empty stack.
///
/// ```mermaid
/// flowchart TD
///     S(["run(goal, input)"]) --> C["Call(goal, input)"]
///     C -->|"start the production"| P{"needs a subproduction?"}
///     P -->|"yes: push a frame"| C
///     P -->|"no"| R["Return(outcome)"]
///     R --> E{"stack empty?"}
///     E -->|"yes"| D(["outcome of the run"])
///     E -->|"no: pop a frame"| F{"resume the production"}
///     F -->|"needs another subproduction:<br/>push a frame"| C
///     F -->|"finished"| R
/// ```
///
/// The subproductions of each production are as follows. The edges that close
/// cycles are the ones that would recurse without the engine.
///
/// ```mermaid
/// flowchart LR
///     Function --> AddSub
///     AddSub --> MulDivMod --> Unary
///     Unary -->|"negation"| Unary
///     Unary --> Exponent
///     Exponent -->|"base"| Primary
///     Exponent -->|"power"| Unary
///     Primary --> Range & Extremum & PrimaryDice & Group & Binding
///     Extremum --> AddSub
///     Range --> AddSub
///     Group --> AddSub
///     Binding --> AddSub
///     PrimaryDice & Dice & StandardDice & CustomDice --> Atom
///     DropLowest & DropHighest & KeepLowest & KeepHighest --> Atom
///     Dice & PrimaryDice & Reroll --> Faces & Atom
///     Faces & Atom --> Binding & Group
///     Faces & Atom -.->|"recovering"| Extremum
/// ```
///
/// # Parameters
/// - `goal`: The goal.
/// - `input`: The input text to parse.
///
/// # Returns
/// The remaining input and the [value](Value) of the goal.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub(super) fn run(goal: Goal, input: Span<'_>) -> Outcome<'_>
{
	drive(
		goal,
		input,
		&mut Recoverer::new(&mut Strict, &mut Silent, input)
	)
}

/// Parse a [function](Goal::Function) at the specified input, without leading
/// whitespace, in _recovery mode_: wherever the parse cannot continue, consult
/// a [policy](Recovery), which may [repair](Repair) the failure so that the
/// parse continues as though the source had been edited. The engine parses the
/// original source only once, whatever the number of repairs.
///
/// Until the first failure that the policy declines to repair, the run is
/// identical to [`run`]; after it, the run continues exactly as [`run`] would
/// from that point, without consulting the policy again.
///
/// A failure is either an unrecoverable error, created at one of the
/// [sites](Site) that the engine guards, or a recoverable error that no frame
/// can catch, which fails the [goal](Site::Goal) in progress. The policy
/// answers a [repair](Repair) of each. At a guarded site, the parse continues
/// from the failure, as though the source had been edited. At a goal, the
/// engine [overlays](Overlay) the token that the repair supplies at the
/// failure, or discards the source that it [skips](Repair::Skip), and parses
/// the goal of the nearest cutting frame again. Either way, a repair that
/// makes a name the first formal parameter, or a skip at the start of the body
/// of a function without formal parameters, parses the whole function again.
/// The policy declines by answering [`Stop`](Repair::Stop), or a repair that
/// the site does not admit; the engine also stops recovering if the parse
/// fails again before it reads the token of the last repair.
///
/// ```mermaid
/// flowchart TD
///     S(["run_recovering(input, policy)"]) --> P["Parse as run would"]
///     P -->|"the function parses"| D(["the function, with the<br/>token of every repair"])
///     P -->|"a recoverable error that<br/>no frame can catch"| G{"policy.repair(Site::Goal)"}
///     P -->|"an unrecoverable error<br/>at a guarded site"| U{"policy.repair(site)"}
///     G -->|"Fix, Variable, or Skip"| A["Parse the goal of the<br/>nearest cutting frame again"]
///     G -->|"a name becomes the first formal<br/>parameter, or a skip starts the<br/>body of a function without them"| F["Parse the function again"]
///     U -->|"a name becomes the<br/>first formal parameter"| F
///     U -->|"repaired"| C["Continue from the failure"]
///     G -->|"declined"| X["Consult the policy no more"]
///     U -->|"declined"| X
///     A & F & C --> P
///     X --> E(["the error of the failure"])
/// ```
///
/// The engine reports every token of the source that it reads to a
/// [sink](TokenSink), but none that a repair supplies.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
/// - `R`: The type of the policy.
/// - `S`: The type of the sink.
///
/// # Parameters
/// - `input`: The input text to parse.
/// - `policy`: The recovery policy.
/// - `sink`: The sink of the tokens that the engine reads.
///
/// # Returns
/// The remaining input and the [function](Value::Function), in which every
/// repair appears as the token that it supplied.
///
/// # Errors
/// * [`Err`](nom::Err) if the policy declined to repair a failure.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub(super) fn run_recovering<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	policy: &mut R,
	sink: &mut S
) -> Outcome<'src>
{
	drive(
		Goal::Function,
		input,
		&mut Recoverer::new(policy, sink, input)
	)
}

/// Drive the engine, as described by [`run`] and, when recovering, by
/// [`run_recovering`].
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
/// - `R`: The type of the recovery policy. For [`Strict`], every recovery hook
///   compiles away, so the run is exactly the ordinary one.
/// - `S`: The type of the sink of tokens. For [`Silent`], every report compiles
///   away.
///
/// # Parameters
/// - `goal`: The goal.
/// - `input`: The input text to parse.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input and the [value](Value) of the goal.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
fn drive<'src, R: Recovery<'src>, S: TokenSink>(
	goal: Goal,
	input: Span<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Outcome<'src>
{
	let mut stack = Stack::new();
	let mut step = Step::Call(goal, input);
	loop
	{
		#[cfg(test)]
		STEPS.with(|steps| steps.set(steps.get() + 1));
		if rec.recovering()
			&& let Some(input) = rec.restart.take()
		{
			step = rec.restart_function(input, &mut stack);
		}
		step = match step
		{
			Step::Call(goal, input) =>
			{
				rec.called(goal, input, &stack);
				start(goal, input, &mut stack, rec)
			},
			Step::Return(outcome) if rec.intercepts(&outcome, &stack) =>
			{
				rec.recover_goal(outcome, &mut stack)
			},
			Step::Return(outcome) => match stack.pop()
			{
				Some(frame) =>
				{
					rec.popped(stack.len());
					resume(frame, outcome, &mut stack, rec)
				},
				None => return outcome
			}
		}
	}
}

#[cfg(test)]
thread_local! {
	/// The number of steps that the engine has taken on this thread.
	static STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Answer the number of steps that the engine has taken on the current thread,
/// so that tests can bound the work of a run.
///
/// # Returns
/// The number of steps.
#[cfg(test)]
pub(crate) fn steps() -> usize { STEPS.with(std::cell::Cell::get) }

/// Start a production.
///
/// # Parameters
/// - `goal`: The production.
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start<'src, R: Recovery<'src>, S: TokenSink>(
	goal: Goal,
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match goal
	{
		Goal::Function => start_function(input, stack, rec),
		Goal::AddSub => call(
			stack,
			Frame::BinaryFirst(Level::Additive),
			Goal::MulDivMod,
			input
		),
		Goal::MulDivMod => call(
			stack,
			Frame::BinaryFirst(Level::Multiplicative),
			Goal::Unary,
			input
		),
		Goal::Unary => start_unary(input, stack, rec),
		Goal::Exponent =>
		{
			call(stack, Frame::ExponentBase, Goal::Primary, input)
		},
		Goal::Primary =>
		{
			call(stack, Frame::PrimaryRange(input), Goal::Range, input)
		},
		Goal::Group => start_group(input, stack, rec),
		Goal::Binding => start_binding(input, stack, rec),
		Goal::Range => start_range(input, stack, rec),
		Goal::Dice => call(
			stack,
			Frame::DiceCount {
				primary: false,
				input
			},
			Goal::Atom,
			input
		),
		Goal::PrimaryDice => call(
			stack,
			Frame::DiceCount {
				primary: true,
				input
			},
			Goal::Atom,
			input
		),
		Goal::StandardDice =>
		{
			call(stack, Frame::StandardCount(input), Goal::Atom, input)
		},
		Goal::CustomDice =>
		{
			call(stack, Frame::CustomCount(input), Goal::Atom, input)
		},
		Goal::Atom => start_atom(input, false, stack, rec),
		Goal::Faces => start_atom(input, true, stack, rec),
		Goal::DropLowest =>
		{
			start_drop(input, ClauseKind::DropLowest, stack, rec)
		},
		Goal::DropHighest =>
		{
			start_drop(input, ClauseKind::DropHighest, stack, rec)
		},
		Goal::KeepLowest =>
		{
			start_drop(input, ClauseKind::KeepLowest, stack, rec)
		},
		Goal::KeepHighest =>
		{
			start_drop(input, ClauseKind::KeepHighest, stack, rec)
		},
		Goal::Reroll =>
		{
			start_predicated(PredicatedKind::Reroll, input, stack, rec)
		},
		Goal::Explode =>
		{
			start_predicated(PredicatedKind::Explode, input, stack, rec)
		},
		Goal::Extremum => start_extremum(input, stack, rec)
	}
}

/// Resume a suspended production with the outcome of its subproduction.
///
/// # Parameters
/// - `frame`: The suspended production, just [popped](Stack::pop).
/// - `outcome`: The outcome of the subproduction.
/// - `stack`: The stack of suspended productions, whose top payload is that of
///   `frame`, if it [carries](Frame::carries) one.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume<'src, R: Recovery<'src>, S: TokenSink>(
	frame: Frame<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	// A catching frame abandons the alternative that failed recoverably, so
	// retract every token that the alternative read.
	if S::ENABLED
		&& matches!(outcome, Err(nom::Err::Error(_)))
		&& let Some(from) = frame.catches_from()
	{
		rec.retract(from);
	}
	match frame
	{
		Frame::FunctionBody { start, input, .. } =>
		{
			let parameters = stack.pop_payload().into_parameters();
			resume_function_body(start, parameters, input, outcome)
		},
		Frame::BinaryFirst(level) => match outcome
		{
			Ok((rest, left)) =>
			{
				binary_continue(level, left.into_expression(), rest, stack, rec)
			},
			Err(e) => Step::Return(Err(e))
		},
		Frame::BinaryRight {
			level,
			operator,
			input
		} =>
		{
			let left = stack.pop_payload().into_expression();
			resume_binary_right(
				level, left, operator, input, outcome, stack, rec
			)
		},
		Frame::UnaryNegation(input) =>
		{
			resume_unary_negation(input, outcome, stack)
		},
		Frame::UnaryExponent { input, merged } =>
		{
			let error = merged.then(|| stack.pop_payload().into_error());
			resume_unary_exponent(input, error, outcome)
		},
		Frame::ExponentBase => resume_exponent_base(outcome, stack, rec),
		Frame::ExponentPower(input) =>
		{
			let base = stack.pop_payload().into_expression();
			resume_exponent_power(base, input, outcome)
		},
		Frame::PrimaryRange(input) =>
		{
			resume_primary_range(input, outcome, stack, rec)
		},
		Frame::PrimaryCall(input) =>
		{
			resume_primary_call(input, outcome, stack, rec)
		},
		Frame::PrimaryDice(input) => resume_primary_dice(input, outcome, stack),
		Frame::PrimaryGroup(input) =>
		{
			let error = stack.pop_payload().into_error();
			resume_primary_group(input, error, outcome, stack, rec)
		},
		Frame::AtomGroup { input, signed } =>
		{
			resume_atom_group(input, signed, outcome)
		},
		Frame::AtomCall(input) => resume_atom_call(input, outcome, stack, rec),
		Frame::CallArgument { input, .. } =>
		{
			let call = stack.pop_payload().into_call();
			resume_call_argument(call, input, outcome, stack, rec)
		},
		Frame::GroupExpression { start, input } =>
		{
			resume_group_expression(start, input, outcome, stack, rec)
		},
		Frame::BindingExpression { paren, input } =>
		{
			let head = stack.pop_payload().into_head();
			resume_binding_expression(head, paren, input, outcome, stack, rec)
		},
		Frame::RangeStart { start, input } =>
		{
			resume_range_start(start, input, outcome, stack, rec)
		},
		Frame::RangeEnd { start, input } =>
		{
			let first = stack.pop_payload().into_expression();
			resume_range_end(start, first, input, outcome, stack, rec)
		},
		Frame::DiceCount { primary, input } =>
		{
			resume_dice_count(primary, input, outcome, stack, rec)
		},
		Frame::DiceFaces { start, input } =>
		{
			let count = stack.pop_payload().into_expression();
			resume_dice_faces(start, count, input, outcome, stack, rec)
		},
		Frame::DiceDrop { kind, input } =>
		{
			let (dice, highest, rest) = stack.pop_payload().into_drop();
			resume_dice_drop(
				(dice, highest),
				kind,
				rest,
				input,
				outcome,
				stack,
				rec
			)
		},
		Frame::StandardCount(input) =>
		{
			resume_standard_count(input, outcome, stack, rec)
		},
		Frame::StandardFaces(input) =>
		{
			let count = stack.pop_payload().into_expression();
			resume_standard_faces(count, input, outcome)
		},
		Frame::CustomCount(input) => resume_custom_count(input, outcome, rec),
		Frame::DropExpression { kind, input } =>
		{
			let rest = stack.pop_payload().into_rest();
			resume_drop_expression(kind, rest, input, outcome)
		},
		Frame::PredicateFirst {
			kind,
			comparison,
			input
		} =>
		{
			let clause = stack.pop_payload().into_predicated();
			resume_predicate_first(
				kind, clause, comparison, input, outcome, stack, rec
			)
		},
		Frame::PredicateEnd {
			kind,
			inclusive,
			input
		} =>
		{
			let clause = stack.pop_payload().into_predicated();
			resume_predicate_end(
				kind, clause, inclusive, input, outcome, stack, rec
			)
		},
		Frame::LimitCount { kind, input } =>
		{
			let clause = stack.pop_payload().into_predicated();
			resume_limit_count(kind, *clause, input, outcome, stack, rec)
		}
	}
}

/// Suspend a production that carries no [payload](Payload), and call a
/// subproduction.
///
/// # Parameters
/// - `stack`: The stack of suspended productions.
/// - `frame`: The suspended production.
/// - `goal`: The subproduction.
/// - `input`: The input of the subproduction.
///
/// # Returns
/// The next step.
fn call<'src>(
	stack: &mut Stack<'src>,
	frame: Frame<'src>,
	goal: Goal,
	input: Span<'src>
) -> Step<'src>
{
	stack.push(frame);
	Step::Call(goal, input)
}

/// Suspend a production that [carries](Frame::carries) a payload, and call a
/// subproduction.
///
/// # Parameters
/// - `stack`: The stack of suspended productions.
/// - `frame`: The suspended production.
/// - `payload`: The payload of the frame.
/// - `goal`: The subproduction.
/// - `input`: The input of the subproduction.
///
/// # Returns
/// The next step.
fn call_with<'src>(
	stack: &mut Stack<'src>,
	frame: Frame<'src>,
	payload: Payload<'src>,
	goal: Goal,
	input: Span<'src>
) -> Step<'src>
{
	stack.push_with(frame, payload);
	Step::Call(goal, input)
}

/// Finish a production successfully.
///
/// # Parameters
/// - `rest`: The remaining input.
/// - `value`: The value of the production.
///
/// # Returns
/// The next step.
fn succeed<'src>(rest: Span<'src>, value: Value<'src>) -> Step<'src>
{
	Step::Return(Ok((rest, value)))
}

/// Finish a production with an error.
///
/// # Parameters
/// - `e`: The error.
///
/// # Returns
/// The next step.
fn fail(e: nom::Err<ParseError<'_>>) -> Step<'_> { Step::Return(Err(e)) }

////////////////////////////////////////////////////////////////////////////////
//                              Error algebra.                                //
////////////////////////////////////////////////////////////////////////////////

/// Attach a context to an error, as [`context`] does.
///
/// # Parameters
/// - `input`: The input at which the context applies.
/// - `label`: The context.
/// - `e`: The error.
///
/// # Returns
/// The error, of the same severity, with the context attached.
fn with_context<'src>(
	input: Span<'src>,
	label: &'static str,
	e: nom::Err<ParseError<'src>>
) -> nom::Err<ParseError<'src>>
{
	e.map(|e| ParseError::add_context(input, label, e))
}

/// Make an error unrecoverable, as [`cut`] does.
///
/// # Parameters
/// - `e`: The error.
///
/// # Returns
/// The error, as a [`Failure`](nom::Err::Failure).
fn cut_error(e: nom::Err<ParseError<'_>>) -> nom::Err<ParseError<'_>>
{
	match e
	{
		nom::Err::Error(e) => nom::Err::Failure(e),
		e => e
	}
}

/// Merge the error of an alternative into the errors of the alternatives
/// before it, as [`alt`] does.
///
/// # Parameters
/// - `accumulated`: The merged errors of the earlier alternatives, if any.
/// - `e`: The error of the alternative.
///
/// # Returns
/// The merged errors.
fn merge<'src>(
	accumulated: ParseError<'src>,
	e: ParseError<'src>
) -> ParseError<'src>
{
	accumulated.or(e)
}

/// Finish an alternation whose every alternative failed recoverably, as
/// [`alt`] does.
///
/// # Parameters
/// - `input`: The input of the alternation.
/// - `error`: The merged errors of the alternatives.
///
/// # Returns
/// The next step.
fn exhaust<'src>(input: Span<'src>, error: ParseError<'src>) -> Step<'src>
{
	fail(nom::Err::Error(ParseError::append(
		input,
		ErrorKind::Alt,
		error
	)))
}

/// Answer the error of an alternative that the engine has tried again, to
/// recompute the error with which it first failed recoverably.
///
/// Nearly every alternative that fails is followed by one that succeeds, so the
/// engine does not keep the errors of alternatives that fail at the input of
/// their production, but recomputes them only when it must merge them, from
/// the input that the frame keeps anyway: see [`negation_error`],
/// [`unary_error`], [`range_error`], and [`atom_error`]. A frame that kept them
/// would allocate them at every level of a deep nesting. The recomputed error
/// is the original: each is that of a plain combinator, a function of the
/// input alone, since a repair's [overlay](Overlay) only ever lets such an
/// alternative succeed.
///
/// # Type parameters
/// - `T`: The type of the value of the alternative.
///
/// # Parameters
/// - `result`: The result of the alternative.
///
/// # Returns
/// The error.
///
/// # Panics
/// If the alternative did not fail recoverably, as it did before.
fn recoverable<'src, T>(
	result: IResult<Span<'src>, T, ParseError<'src>>
) -> ParseError<'src>
{
	match result
	{
		Err(nom::Err::Error(e)) => e,
		_ => unreachable!("the alternative failed recoverably before")
	}
}

/// Recompute the error of the first alternative of [`Goal::Unary`], a negative
/// constant, which failed at the specified input.
///
/// # Parameters
/// - `input`: The input of the unary expression.
///
/// # Returns
/// The error.
///
/// # Panics
/// If the negative constant did not fail recoverably.
fn negation_error(input: Span<'_>) -> ParseError<'_>
{
	recoverable(negative_constant(input))
}

/// Recompute the merged errors of the first two alternatives of
/// [`Goal::Unary`], a negative constant and a negation, which both failed at
/// the specified input, the negation at its `-`.
///
/// # Parameters
/// - `input`: The input of the unary expression.
///
/// # Returns
/// The merged errors.
///
/// # Panics
/// If either alternative did not fail recoverably.
fn unary_error(input: Span<'_>) -> ParseError<'_>
{
	let sign: IResult<Span, char, ParseError> = char('-')(input);
	merge(negation_error(input), recoverable(sign))
}

/// Recompute the error of the first alternative of [`Goal::Primary`], a range,
/// which failed at its `[`, the only place where it fails recoverably.
///
/// # Parameters
/// - `input`: The input of the primary expression.
///
/// # Returns
/// The error.
///
/// # Panics
/// If the `[` did not fail recoverably.
fn range_error(input: Span<'_>) -> ParseError<'_>
{
	let bracket: IResult<Span, char, ParseError> = char('[')(input);
	ParseError::add_context(input, RANGE_CONTEXT, recoverable(bracket))
}

/// Recompute the merged errors of the first two alternatives of
/// [`Goal::Atom`] or [`Goal::Faces`], a literal and a variable, which both
/// failed at the specified input.
///
/// # Parameters
/// - `input`: The input of the atom.
/// - `signed`: Whether the goal is [`Goal::Faces`], whose literal is signed.
///
/// # Returns
/// The merged errors.
///
/// # Panics
/// If either alternative did not fail recoverably.
fn atom_error(input: Span<'_>, signed: bool) -> ParseError<'_>
{
	merge(
		recoverable(
			context(CONSTANT_CONTEXT, literal(signed)).parse_complete(input)
		),
		recoverable(
			context(VARIABLE_CONTEXT, braced_name).parse_complete(input)
		)
	)
}

////////////////////////////////////////////////////////////////////////////////
//                                Productions.                                //
////////////////////////////////////////////////////////////////////////////////

/// Start [`Goal::Function`]: `parameters`, then
/// `preceded(multispace0, context(FUNCTION_BODY_CONTEXT, expression))`. If a
/// repair [overlaid](Overlay) the first formal parameter, as when the engine
/// parses the function again, then the parameters are
/// [recovered](recover_parameters) from the start.
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_function<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let start = input.location_offset();
	let overlaid = rec.recovering() && rec.overlays_parameter(input);
	let (rest, parameters) = match parameters.parse_complete(input)
	{
		Ok((rest, parameters)) if !overlaid =>
		{
			if let Some(parameters) = &parameters
			{
				report_parameters(input, parameters, rec);
			}
			(rest, parameters)
		},
		Err(e) if !rec.recovering() => return fail(e),
		_ => match recover_parameters(input, stack, rec)
		{
			Ok(parsed) => parsed,
			Err(e) => return fail(e)
		}
	};
	let input = skip_whitespace(rest);
	call_with(
		stack,
		Frame::FunctionBody {
			start,
			formal: parameters.is_some(),
			input
		},
		Payload::Parameters(parameters),
		Goal::AddSub,
		input
	)
}

/// Report the formal parameters of a function, which [`parameters`] has just
/// read, by scanning them again, so that no path of the parse changes: the
/// braces and identifier of each, the `,` after each but the last, and the `:`
/// after the last, all [separators](Role::Separator).
///
/// # Parameters
/// - `input`: The input of the function.
/// - `parameters`: The formal parameters.
/// - `rec`: The recovery state.
///
/// # Panics
/// If a parameter lacks a brace, which [`parameters`] never answers.
fn report_parameters<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	parameters: &[Parameter<'src>],
	rec: &mut Recoverer<'src, '_, R, S>
)
{
	if !S::ENABLED
	{
		return;
	}
	let base = input.location_offset();
	let source = *input.fragment();
	let mut from = 0;
	for parameter in parameters
	{
		let name = parameter.span;
		// Only whitespace separates the identifier from its braces, and only
		// token space separates the `}` from the `,` or `:`.
		let opener = source[from..name.start - base]
			.rfind('{')
			.expect("a parameter has a `{`")
			+ from;
		let closer = source[name.end - base..]
			.find('}')
			.expect("a parameter has a `}`")
			+ name.end
			- base;
		let after = &source[closer + 1..];
		let separator = closer + 1 + after.len()
			- after.trim_start_matches(is_token_space).len();
		let braced = SourceSpan {
			start: base + opener,
			end: base + closer + 1
		};
		rec.report_name(braced, name, Role::ParameterDeclaration);
		rec.report(base + separator, base + separator + 1, Role::Separator);
		from = separator + 1;
	}
}

/// Resume [`Goal::Function`] with the outcome of the body.
///
/// # Parameters
/// - `start`: The start of the function.
/// - `parameters`: The formal parameters.
/// - `input`: The input of the body.
/// - `outcome`: The outcome of the body.
///
/// # Returns
/// The next step.
fn resume_function_body<'src>(
	start: usize,
	parameters: Option<Vec<Parameter<'src>>>,
	input: Span<'src>,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, body)) =>
		{
			let end = rest.location_offset();
			succeed(
				rest,
				Value::Function(Function {
					parameters,
					body: body.into_expression(),
					span: SourceSpan { start, end }
				})
			)
		},
		Err(e) => fail(with_context(input, FUNCTION_BODY_CONTEXT, e))
	}
}

/// Continue [`Goal::AddSub`] or [`Goal::MulDivMod`] after an operand, as
/// `many0(pair(preceded(multispace0, one_of(OPERATORS)), cut(preceded(
/// multispace0, context(RIGHT_OPERAND_CONTEXT, OPERAND)))))`. If an operator
/// follows, call the next operand; otherwise, finish with the fold of the
/// operands so far.
///
/// # Parameters
/// - `level`: The level.
/// - `left`: The fold of the operands so far.
/// - `rest`: The input after the last operand.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn binary_continue<'src, R: Recovery<'src>, S: TokenSink>(
	level: Level,
	left: Expression<'src>,
	rest: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let operator: IResult<Span, char, ParseError> =
		preceded(multispace0, one_of(level.operators())).parse_complete(rest);
	match operator
	{
		Ok((after, operator)) =>
		{
			rec.report_read(rest, after, level.role());
			let input = skip_whitespace(after);
			call_with(
				stack,
				Frame::BinaryRight {
					level,
					operator,
					input
				},
				Payload::Expression(left),
				level.operand(),
				input
			)
		},
		// `many0` stops at the first recoverable error, and discards it. The
		// operator is a single character, so it cannot fail unrecoverably.
		Err(_) => succeed(rest, Value::Expression(left))
	}
}

/// Resume [`Goal::AddSub`] or [`Goal::MulDivMod`] with the outcome of a right
/// operand.
///
/// # Parameters
/// - `level`: The level.
/// - `left`: The fold of the operands before the operator.
/// - `operator`: The operator.
/// - `input`: The input of the right operand.
/// - `outcome`: The outcome of the right operand.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_binary_right<'src, R: Recovery<'src>, S: TokenSink>(
	level: Level,
	left: Expression<'src>,
	operator: char,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, right)) =>
		{
			let right = right.into_expression();
			let span = SourceSpan {
				start: left.span().start,
				end: right.span().end
			};
			let left = Box::new(left);
			let right = Box::new(right);
			let arithmetic = match operator
			{
				'+' => ArithmeticExpression::Add(Add { left, right, span }),
				'-' => ArithmeticExpression::Sub(Sub { left, right, span }),
				'*' | '×' =>
				{
					ArithmeticExpression::Mul(Mul { left, right, span })
				},
				'/' | '÷' =>
				{
					ArithmeticExpression::Div(Div { left, right, span })
				},
				'%' => ArithmeticExpression::Mod(Mod { left, right, span }),
				_ => unreachable!("unexpected operator: {}", operator)
			};
			binary_continue(
				level,
				Expression::Arithmetic(arithmetic),
				rest,
				stack,
				rec
			)
		},
		Err(e) => fail(cut_error(with_context(input, RIGHT_OPERAND_CONTEXT, e)))
	}
}

/// Start [`Goal::Unary`]: `alt((negative_constant, negation,
/// preceded(multispace0, exponent)))`, where `negation` is `preceded(
/// char('-'), preceded(multispace0, unary))`. The errors of the alternatives
/// that fail here are discarded, and [recomputed](unary_error) only if the
/// last alternative fails too.
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_unary<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match parse_negative_constant(input, rec)
	{
		Ok((rest, constant)) =>
		{
			return succeed(rest, Value::Expression(constant))
		},
		Err(nom::Err::Error(_)) =>
		{},
		Err(e) => return fail(e)
	}
	let sign: IResult<Span, char, ParseError> = char('-')(input);
	match sign
	{
		Ok((after, _)) =>
		{
			// A catching frame retracts the sign if the negation fails.
			rec.report_read(input, after, Role::Negation);
			let operand = skip_whitespace(after);
			call(stack, Frame::UnaryNegation(input), Goal::Unary, operand)
		},
		Err(nom::Err::Error(_)) => unary_exponent(input, None, stack),
		Err(e) => fail(e)
	}
}

/// Try the third alternative of [`Goal::Unary`], an exponentiation.
///
/// # Parameters
/// - `input`: The input of the unary expression.
/// - `error`: The merged errors of the first two alternatives, if the negation
///   failed after its `-`, or `None` if both failed at the input.
/// - `stack`: The stack of suspended productions.
///
/// # Returns
/// The next step.
fn unary_exponent<'src>(
	input: Span<'src>,
	error: Option<ParseError<'src>>,
	stack: &mut Stack<'src>
) -> Step<'src>
{
	let exponent = skip_whitespace(input);
	match error
	{
		Some(error) => call_with(
			stack,
			Frame::UnaryExponent {
				input,
				merged: true
			},
			Payload::Error(error),
			Goal::Exponent,
			exponent
		),
		None => call(
			stack,
			Frame::UnaryExponent {
				input,
				merged: false
			},
			Goal::Exponent,
			exponent
		)
	}
}

/// Resume [`Goal::Unary`] with the outcome of the operand of a negation.
///
/// # Parameters
/// - `input`: The input of the unary expression, which begins with the `-`.
/// - `outcome`: The outcome of the operand.
/// - `stack`: The stack of suspended productions.
///
/// # Returns
/// The next step.
///
/// # Notes
/// A negation of the constant `0` is the constant `0`, spanning the whole
/// negation, since there is no negative zero among integers. The operand of
/// `--0` is `-0`, which [`negative_constant`] reads as `0`, so without this
/// fold, `--0` would render as `-0` and then parse as `0`. A group is left
/// alone, so `-(0)` is still a negation.
fn resume_unary_negation<'src>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, operand)) =>
		{
			let operand = operand.into_expression();
			let span = SourceSpan {
				start: input.location_offset(),
				end: operand.span().end
			};
			let negation = match operand
			{
				Expression::Constant(Constant { value: 0, .. }) =>
				{
					Expression::Constant(Constant { value: 0, span })
				},
				operand =>
				{
					Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
						operand: Box::new(operand),
						span
					}))
				},
			};
			succeed(rest, Value::Expression(negation))
		},
		Err(nom::Err::Error(e)) =>
		{
			let error = merge(negation_error(input), e);
			unary_exponent(input, Some(error), stack)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Unary`] with the outcome of an exponentiation, its last
/// alternative.
///
/// # Parameters
/// - `input`: The input of the unary expression.
/// - `error`: The merged errors of the first two alternatives, if the negation
///   failed after its `-`, or `None` if both failed at the input.
/// - `outcome`: The outcome of the exponentiation.
///
/// # Returns
/// The next step.
fn resume_unary_exponent<'src>(
	input: Span<'src>,
	error: Option<ParseError<'src>>,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, value)) => succeed(rest, value),
		Err(nom::Err::Error(e)) =>
		{
			let error = error.unwrap_or_else(|| unary_error(input));
			exhaust(input, merge(error, e))
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Exponent`] with the outcome of the base, and continue as
/// `many0(pair(preceded(multispace0, char('^')), cut(preceded(multispace0,
/// context(RIGHT_OPERAND_CONTEXT, unary)))))`.
///
/// The power is a [`Goal::Unary`], which parses every `^` that follows it
/// before it returns, so the original `many0` only ever collected one pair: the
/// second iteration always failed recoverably, which `many0` discards. So the
/// engine does not look for a second `^` after the power.
///
/// # Parameters
/// - `outcome`: The outcome of the base.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_exponent_base<'src, R: Recovery<'src>, S: TokenSink>(
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, base) = match outcome
	{
		Ok((rest, base)) => (rest, base.into_expression()),
		Err(e) => return fail(e)
	};
	let caret: IResult<Span, char, ParseError> =
		preceded(multispace0, char('^')).parse_complete(rest);
	match caret
	{
		Ok((after, _)) =>
		{
			rec.report_read(rest, after, Role::Exponent);
			let input = skip_whitespace(after);
			call_with(
				stack,
				Frame::ExponentPower(input),
				Payload::Expression(base),
				Goal::Unary,
				input
			)
		},
		Err(_) => succeed(rest, Value::Expression(base))
	}
}

/// Resume [`Goal::Exponent`] with the outcome of the power.
///
/// # Parameters
/// - `base`: The base.
/// - `input`: The input of the power.
/// - `outcome`: The outcome of the power.
///
/// # Returns
/// The next step.
fn resume_exponent_power<'src>(
	base: Expression<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, power)) =>
		{
			let power = power.into_expression();
			let span = SourceSpan {
				start: base.span().start,
				end: power.span().end
			};
			succeed(
				rest,
				Value::Expression(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(base),
						right: Box::new(power),
						span
					})
				))
			)
		},
		Err(e) => fail(cut_error(with_context(input, RIGHT_OPERAND_CONTEXT, e)))
	}
}

/// Resume [`Goal::Primary`] with the outcome of a range, its first alternative.
///
/// [`Goal::Primary`] is `alt((context(RANGE_CONTEXT, range),
/// context(DICE_CONTEXT, dice), context(GROUP_CONTEXT, group),
/// context(VARIABLE_CONTEXT, variable), context(BINDING_CONTEXT, binding),
/// context(CONSTANT_CONTEXT, constant)))`. As the original did, the engine
/// tries dice before a group, a variable, a binding, or a constant, since each
/// of those can begin a dice expression as its dice count. But whenever the
/// dice count parses and no dice operator follows, the original went on to
/// parse the dice count all over again, as the alternative of the same kind;
/// the dice count of the dice count did the same; and so on, so that the time
/// was exponential in the nesting of groups and bindings. The engine instead
/// takes the dice count that it already has as the value of the primary
/// expression, via [`Goal::PrimaryDice`]. This is always what the original
/// produced, since the four kinds begin with different characters, so exactly
/// one of the later alternatives could succeed, and it would parse the same
/// text in the same way. It also discarded the errors of the failed
/// alternatives, so no error differs either.
///
/// If the range fails, its error is discarded, and [recomputed](range_error)
/// only if the dice fail too.
///
/// Between the range and the dice, the engine tries a call of `highest` or
/// `lowest`, but only where the input [begins a call](begins_call), or a repair
/// [overlaid](Overlay::Call) the keyword of one, since a call commits once it
/// reads its `(`. Otherwise, nothing else begins with the keyword, so the call
/// would fail recoverably at the input, and its error would add nothing to the
/// errors of the other alternatives.
///
/// # Parameters
/// - `input`: The input of the primary expression.
/// - `outcome`: The outcome of the range.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_primary_range<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match with_context_on_error(input, RANGE_CONTEXT, outcome)
	{
		Ok((rest, range)) => succeed(
			rest,
			Value::Expression(Expression::Range(range.into_range()))
		),
		Err(nom::Err::Error(_))
			if begins_call(input.fragment()).is_some()
				|| rec.recovering() && rec.overlays_call(input) =>
		{
			call(stack, Frame::PrimaryCall(input), Goal::Extremum, input)
		},
		Err(nom::Err::Error(_)) =>
		{
			call(stack, Frame::PrimaryDice(input), Goal::PrimaryDice, input)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Primary`] with the outcome of a call of `highest` or
/// `lowest`.
///
/// A call is no dice count, so the ordinary parse ends the primary expression
/// at the call, even if a dice operator follows it, e.g., `highest(1, 2)D6`,
/// and fails later, wherever the dice operator stands. While recovering, a
/// dice operator after the call, other than the `d` of `drop`, offers a repair
/// at [`Site::UngroupedCall`], whose fix groups the call, and makes it the dice
/// count of a dice expression, e.g., `(highest(1, 2))D6`. The fix supplies its
/// `(` before the call, and its `)` after the call, at a [`Site::Closer`].
///
/// # Parameters
/// - `input`: The input of the primary expression.
/// - `outcome`: The outcome of the call.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state, which remembers the dice operator for a
///   [reread](Repair::Reread).
///
/// # Returns
/// The next step. Where the repair made the call a dice count, the dice
/// expression, as a [`Value::Dice`], stands for the primary expression.
fn resume_primary_call<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, call) = match outcome
	{
		Ok((rest, Value::Expression(call)))
			if rec.recovering() && matches!(call, Expression::Extremum(_)) =>
		{
			(rest, call)
		},
		outcome => return Step::Return(outcome)
	};
	let operator: IResult<Span, char, ParseError> =
		preceded(multispace0, d_operator).parse_complete(rest);
	let after = match operator
	{
		Ok((after, _))
			if !begins_verb(skip_whitespace(rest).fragment(), "drop")
				&& rec.offer(
					Site::UngroupedCall {
						operand: Operand::DiceCount
					},
					offered(input),
					stack
				) =>
		{
			after
		},
		_ => return succeed(rest, Value::Expression(call))
	};
	let count = match group_call(input, call, rest, stack, rec)
	{
		Ok(count) => count,
		Err(e) => return fail(e)
	};
	rec.report_before(after, Role::DiceOperator);
	rec.operator = Some(skip_whitespace(rest));
	let faces = skip_whitespace(after);
	call_with(
		stack,
		Frame::DiceFaces {
			start: input.location_offset(),
			input: faces
		},
		Payload::Expression(count),
		Goal::Faces,
		faces
	)
}

/// Resume [`Goal::Primary`] with the outcome of dice, its second alternative.
///
/// # Parameters
/// - `input`: The input of the primary expression.
/// - `outcome`: The outcome of the dice.
/// - `stack`: The stack of suspended productions.
///
/// # Returns
/// The next step.
fn resume_primary_dice<'src>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>
) -> Step<'src>
{
	match with_context_on_error(input, DICE_CONTEXT, outcome)
	{
		Ok((rest, Value::Dice(dice))) =>
		{
			succeed(rest, Value::Expression(Expression::Dice(dice)))
		},
		// The dice count, without a dice operator.
		Ok((rest, count)) => succeed(rest, count),
		Err(nom::Err::Error(e)) => call_with(
			stack,
			Frame::PrimaryGroup(input),
			Payload::Error(merge(range_error(input), e)),
			Goal::Group,
			input
		),
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Primary`] with the outcome of a group, its third
/// alternative, and try a variable, its fourth, which [continues as a
/// binding](variable_or_binding) if an `@` follows it, and a constant, its
/// last. The original tried a binding as the fifth alternative, but a binding
/// begins as a variable does, so it fails wherever the variable does.
///
/// # Parameters
/// - `input`: The input of the primary expression.
/// - `error`: The merged errors of the first two alternatives.
/// - `outcome`: The outcome of the group.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_primary_group<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	error: ParseError<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let error = match with_context_on_error(input, GROUP_CONTEXT, outcome)
	{
		Ok((rest, group)) =>
		{
			return succeed(
				rest,
				Value::Expression(Expression::Group(group.into_group()))
			);
		},
		Err(nom::Err::Error(e)) => merge(error, e),
		Err(e) => return fail(e)
	};
	let error = match parse_variable(input, stack, rec)
	{
		Ok((rest, (variable, name_span))) =>
		{
			return variable_or_binding(
				input, rest, variable, name_span, stack, rec
			);
		},
		Err(nom::Err::Error(e)) => merge(error, e),
		Err(e) => return fail(e)
	};
	match parse_constant(input, false, rec)
	{
		Ok((rest, constant)) =>
		{
			succeed(rest, Value::Expression(Expression::Constant(constant)))
		},
		Err(nom::Err::Error(e)) => exhaust(input, merge(error, e)),
		Err(e) => fail(e)
	}
}

/// Finish a variable that an alternative of [`Goal::Atom`] or [`Goal::Primary`]
/// has read, or, if an `@` follows it, continue it as a binding,
/// `{name}@(expr)`, as [`Goal::Binding`] would. A binding begins just as a
/// variable does, so the alternative reads the braced name only once, rather
/// than trying a binding, then a variable, which would read the name again
/// wherever no `@` follows it, as it almost never does.
///
/// # Parameters
/// - `input`: The input of the atom or primary expression.
/// - `rest`: The input after the variable.
/// - `variable`: The variable.
/// - `name_span`: The span of the name of the variable.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn variable_or_binding<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	rest: Span<'src>,
	variable: Variable<'src>,
	name_span: SourceSpan,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let at: IResult<Span, char, ParseError> =
		preceded(multispace0, char('@')).parse_complete(rest);
	match at
	{
		Ok((after_at, _)) =>
		{
			rec.report_name(variable.span, name_span, Role::BindingDeclaration);
			rec.report_before(after_at, Role::Bind);
			let head = BindingHead {
				start: variable.span.start,
				name: variable.name,
				name_span,
				atom: Some(input)
			};
			bind(head, after_at, stack, rec)
		},
		Err(_) =>
		{
			// Name resolution will tell references apart.
			rec.report_name(variable.span, name_span, Role::Unresolved);
			succeed(rest, Value::Expression(Expression::Variable(variable)))
		}
	}
}

/// Start [`Goal::Atom`]: `alt((context(CONSTANT_CONTEXT, constant),
/// context(VARIABLE_CONTEXT, variable), context(GROUP_CONTEXT, group)))`, where
/// the variable [continues as a binding](variable_or_binding) if an `@` follows
/// it. The original tried a binding as the third alternative, but a binding
/// begins as a variable does, so it fails wherever the variable does. The
/// errors of the alternatives that fail here are discarded, and
/// [recomputed](atom_error) only if the group fails too. [`Goal::Faces`] is the
/// same, but with a signed [`integer`] in place of the [`constant`].
///
/// # Parameters
/// - `input`: The input of the production.
/// - `signed`: Whether the goal is [`Goal::Faces`], whose literal is signed.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_atom<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	signed: bool,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	// A call that stands ungrouped where only an atom may fails here, as no
	// alternative begins with its keyword. While recovering, offer to read it
	// as though grouped.
	if rec.recovering()
		&& begins_call(input.fragment()).is_some()
		&& rec.offer(
			Site::UngroupedCall {
				operand: Operand::of(stack.last())
			},
			offered(input),
			stack
		)
	{
		return call(stack, Frame::AtomCall(input), Goal::Extremum, input);
	}
	match parse_constant(input, signed, rec)
	{
		Ok((rest, constant)) =>
		{
			return succeed(
				rest,
				Value::Expression(Expression::Constant(constant))
			);
		},
		Err(nom::Err::Error(_)) =>
		{},
		Err(e) => return fail(e)
	}
	match parse_variable(input, stack, rec)
	{
		Ok((rest, (variable, name_span))) =>
		{
			variable_or_binding(input, rest, variable, name_span, stack, rec)
		},
		Err(nom::Err::Error(_)) => call(
			stack,
			Frame::AtomGroup { input, signed },
			Goal::Group,
			input
		),
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Atom`] or [`Goal::Faces`] with the outcome of a group, its
/// last alternative.
///
/// # Parameters
/// - `input`: The input of the atom.
/// - `signed`: Whether the goal is [`Goal::Faces`], whose literal is signed.
/// - `outcome`: The outcome of the group.
///
/// # Returns
/// The next step.
fn resume_atom_group<'src>(
	input: Span<'src>,
	signed: bool,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match with_context_on_error(input, GROUP_CONTEXT, outcome)
	{
		Ok((rest, group)) => succeed(
			rest,
			Value::Expression(Expression::Group(group.into_group()))
		),
		Err(nom::Err::Error(e)) =>
		{
			exhaust(input, merge(atom_error(input, signed), e))
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Atom`] or [`Goal::Faces`] with the outcome of a call that
/// stands ungrouped where only an atom may, which a repair at
/// [`Site::UngroupedCall`] reads as though grouped. The repair supplied the
/// `(` before the call, and its `)` is supplied after the call, at a
/// [`Site::Closer`].
///
/// # Parameters
/// - `input`: The input of the atom, at the start of the call.
/// - `outcome`: The outcome of the call.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_atom_call<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, call) = match outcome
	{
		Ok((rest, call)) => (rest, call.into_expression()),
		Err(e) => return fail(e)
	};
	match group_call(input, call, rest, stack, rec)
	{
		Ok(group) => succeed(rest, Value::Expression(group)),
		Err(e) => fail(e)
	}
}

/// Group a call that a repair at [`Site::UngroupedCall`] reads as though
/// grouped, by supplying the `)` after it at a [`Site::Closer`], whose `(` is
/// the start of the call, where the repair supplied it.
///
/// # Parameters
/// - `input`: The input at the start of the call.
/// - `call`: The call, or what a repair made of it.
/// - `rest`: The input after the call.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The group.
///
/// # Errors
/// * [`Failure`](nom::Err::Failure) if the policy declines to supply the `)`.
fn group_call<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	call: Expression<'src>,
	rest: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Result<Expression<'src>, nom::Err<ParseError<'src>>>
{
	let start = input.location_offset();
	let site = Site::Closer {
		opener: start,
		closer: ')'
	};
	// The source has no `)` for the supplied `(`, so the `)` is missing after
	// the call, whatever follows it.
	let e = nom::Err::Failure(ParseError::add_context(
		rest,
		CLOSING_PAREN_CONTEXT,
		ParseError::from_char(rest, ')')
	));
	let at = rec.repair(site, e, stack)?;
	Ok(Expression::Group(Group {
		expression: Box::new(call),
		span: SourceSpan {
			start,
			end: at.location_offset()
		}
	}))
}

/// Answer the error that an offer of a repair at a call presents: one at the
/// start of the call. Since an offer is no failure of the ordinary parse, only
/// its position matters.
///
/// # Parameters
/// - `input`: The input at the start of the call.
///
/// # Returns
/// The error.
fn offered(input: Span<'_>) -> nom::Err<ParseError<'_>>
{
	nom::Err::Failure(ParseError::from_error_kind(input, ErrorKind::Tag))
}

/// Start [`Goal::Group`]: `char('(')`, then `cut(preceded(multispace0,
/// context(EXPRESSION_CONTEXT, expression)))`, then [a closing
/// parenthesis](close).
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_group<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let start = input.location_offset();
	let paren: IResult<Span, char, ParseError> = char('(')(input);
	match paren
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::Parenthesis);
			let input = skip_whitespace(after);
			call(
				stack,
				Frame::GroupExpression { start, input },
				Goal::AddSub,
				input
			)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Group`] with the outcome of the grouped expression.
///
/// # Parameters
/// - `start`: The start of the group.
/// - `input`: The input of the grouped expression.
/// - `outcome`: The outcome of the grouped expression.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_group_expression<'src, R: Recovery<'src>, S: TokenSink>(
	start: usize,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, expression) = match outcome
	{
		Ok((rest, expression)) => (rest, expression.into_expression()),
		Err(e) =>
		{
			return fail(cut_error(with_context(input, EXPRESSION_CONTEXT, e)));
		}
	};
	let site = Site::Closer {
		opener: start,
		closer: ')'
	};
	let rest = match close(')', CLOSING_PAREN_CONTEXT, rest)
	{
		Ok(rest) =>
		{
			rec.report_before(rest, Role::Parenthesis);
			rest
		},
		Err(e) => match rec.repair(site, e, stack)
		{
			Ok(at) => at,
			Err(e) => return fail(e)
		}
	};
	let end = rest.location_offset();
	succeed(
		rest,
		Value::Group(Group {
			expression: Box::new(expression),
			span: SourceSpan { start, end }
		})
	)
}

/// Start [`Goal::Binding`]: a [braced name](braced_name), then a check for `@`
/// that fails recoverably at the start of the binding if the `@` is missing,
/// then the rest of the binding; see [`bind`].
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_binding<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (after_name, name) = match braced_name(input)
	{
		Ok(parsed) => parsed,
		Err(e) => return fail(e)
	};
	let at: IResult<Span, char, ParseError> =
		preceded(multispace0, char('@')).parse_complete(after_name);
	match at
	{
		Ok((after_at, _)) =>
		{
			let braced = SourceSpan {
				start: input.location_offset(),
				end: after_name.location_offset()
			};
			rec.report_name(braced, span_of(name), Role::BindingDeclaration);
			rec.report_before(after_at, Role::Bind);
			let head = BindingHead {
				start: input.location_offset(),
				name: canonical_name(name.fragment()),
				name_span: span_of(name),
				atom: None
			};
			bind(head, after_at, stack, rec)
		},
		Err(_) => fail(nom::Err::Error(ParseError::from_error_kind(
			input,
			ErrorKind::Tag
		)))
	}
}

/// Continue a binding after its `@`: `cut(preceded(multispace0, char('(')))`,
/// then `cut(preceded(multispace0, context(BINDING_EXPRESSION_CONTEXT,
/// expression)))`, then [a closing parenthesis](close).
///
/// # Parameters
/// - `head`: The head of the binding.
/// - `after_at`: The input after the `@`.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn bind<'src, R: Recovery<'src>, S: TokenSink>(
	head: BindingHead<'src>,
	after_at: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let paren: IResult<Span, char, ParseError> =
		cut(preceded(multispace0, char('('))).parse_complete(after_at);
	let (paren, after) = match paren
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::Parenthesis);
			(after.location_offset() - 1, after)
		},
		Err(e) =>
		{
			let e = within_binding(head.atom, e);
			match rec.repair(Site::BindingParen, e, stack)
			{
				Ok(at) => (at.location_offset(), at),
				Err(e) => return fail(e)
			}
		}
	};
	let input = skip_whitespace(after);
	call_with(
		stack,
		Frame::BindingExpression { paren, input },
		Payload::Head(head),
		Goal::AddSub,
		input
	)
}

/// Resume [`Goal::Binding`] with the outcome of the bound expression.
///
/// # Parameters
/// - `head`: The head of the binding.
/// - `paren`: The byte offset of the `(` of the binding, or of where a repair
///   supplied it.
/// - `input`: The input of the bound expression.
/// - `outcome`: The outcome of the bound expression.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_binding_expression<'src, R: Recovery<'src>, S: TokenSink>(
	head: BindingHead<'src>,
	paren: usize,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, expression) = match outcome
	{
		Ok((rest, expression)) => (rest, expression.into_expression()),
		Err(e) =>
		{
			return fail(within_binding(
				head.atom,
				cut_error(with_context(input, BINDING_EXPRESSION_CONTEXT, e))
			));
		}
	};
	let site = Site::Closer {
		opener: paren,
		closer: ')'
	};
	let rest = match close(')', CLOSING_PAREN_CONTEXT, rest)
	{
		Ok(rest) =>
		{
			rec.report_before(rest, Role::Parenthesis);
			rest
		},
		Err(e) => match rec.repair(site, within_binding(head.atom, e), stack)
		{
			Ok(at) => at,
			Err(e) => return fail(e)
		}
	};
	let end = rest.location_offset();
	let binding = Binding {
		name: head.name,
		name_span: head.name_span,
		expression: Box::new(expression),
		span: SourceSpan {
			start: head.start,
			end
		}
	};
	match head.atom
	{
		Some(_) =>
		{
			succeed(rest, Value::Expression(Expression::Binding(binding)))
		},
		None => succeed(rest, Value::Binding(binding))
	}
}

/// Start [`Goal::Range`]: `char('[')`, then `cut(preceded(multispace0,
/// context(RANGE_START_CONTEXT, expression)))`, then `cut(preceded(
/// multispace0, char(':')))`, then `cut(preceded(multispace0,
/// context(RANGE_END_CONTEXT, expression)))`, then [a closing
/// bracket](close).
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_range<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let start = input.location_offset();
	let bracket: IResult<Span, char, ParseError> = char('[')(input);
	match bracket
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::RangeBracket);
			let input = skip_whitespace(after);
			call(
				stack,
				Frame::RangeStart { start, input },
				Goal::AddSub,
				input
			)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Range`] with the outcome of the start of the range.
///
/// # Parameters
/// - `start`: The start of the range expression.
/// - `input`: The input of the start of the range.
/// - `outcome`: The outcome of the start of the range.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_range_start<'src, R: Recovery<'src>, S: TokenSink>(
	start: usize,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, first) = match outcome
	{
		Ok((rest, first)) => (rest, first.into_expression()),
		Err(e) =>
		{
			return fail(cut_error(with_context(input, RANGE_START_CONTEXT, e)));
		}
	};
	let colon: IResult<Span, char, ParseError> =
		cut(preceded(multispace0, char(':'))).parse_complete(rest);
	let after = match colon
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::RangeOperator);
			after
		},
		Err(e) => match rec.repair(Site::Colon { opener: start }, e, stack)
		{
			Ok(at) => at,
			Err(e) => return fail(e)
		}
	};
	let input = skip_whitespace(after);
	call_with(
		stack,
		Frame::RangeEnd { start, input },
		Payload::Expression(first),
		Goal::AddSub,
		input
	)
}

/// Resume [`Goal::Range`] with the outcome of the end of the range.
///
/// # Parameters
/// - `start`: The start of the range expression.
/// - `first`: The start of the range.
/// - `input`: The input of the end of the range.
/// - `outcome`: The outcome of the end of the range.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_range_end<'src, R: Recovery<'src>, S: TokenSink>(
	start: usize,
	first: Expression<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, last) = match outcome
	{
		Ok((rest, last)) => (rest, last.into_expression()),
		Err(e) =>
		{
			return fail(cut_error(with_context(input, RANGE_END_CONTEXT, e)));
		}
	};
	let site = Site::Closer {
		opener: start,
		closer: ']'
	};
	let rest = match close(']', CLOSING_BRACKET_CONTEXT, rest)
	{
		Ok(rest) =>
		{
			rec.report_before(rest, Role::RangeBracket);
			rest
		},
		Err(e) => match rec.repair(site, e, stack)
		{
			Ok(at) => at,
			Err(e) => return fail(e)
		}
	};
	let end = rest.location_offset();
	succeed(
		rest,
		Value::Range(Range {
			start: Box::new(first),
			end: Box::new(last),
			span: SourceSpan { start, end }
		})
	)
}

/// Resume [`Goal::Dice`] or [`Goal::PrimaryDice`] with the outcome of the dice
/// count.
///
/// [`Goal::Dice`] is `context(DICE_COUNT_CONTEXT, dice_count)`, then
/// `preceded(multispace0, d_operator)`, then `cut(preceded(multispace0,
/// alt((context(STANDARD_FACES_CONTEXT, standard_faces),
/// context(CUSTOM_FACES_CONTEXT, custom_faces)))))`, then
/// `fold_many0(alt((drop_clause, keep_clause)))`. [`Goal::PrimaryDice`] is the
/// same, except that it produces the dice count if no dice operator follows it.
///
/// # Parameters
/// - `primary`: Whether the goal is [`Goal::PrimaryDice`].
/// - `input`: The input of the dice expression.
/// - `outcome`: The outcome of the dice count.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state, which remembers the dice operator for a
///   [reread](Repair::Reread).
///
/// # Returns
/// The next step.
fn resume_dice_count<'src, R: Recovery<'src>, S: TokenSink>(
	primary: bool,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, count) =
		match with_context_on_error(input, DICE_COUNT_CONTEXT, outcome)
		{
			Ok((rest, count)) => (rest, count.into_expression()),
			Err(e) => return fail(e)
		};
	let operator: IResult<Span, char, ParseError> =
		preceded(multispace0, d_operator).parse_complete(rest);
	match operator
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::DiceOperator);
			if rec.recovering()
			{
				rec.operator = Some(skip_whitespace(rest));
			}
			let faces = skip_whitespace(after);
			call_with(
				stack,
				Frame::DiceFaces {
					start: input.location_offset(),
					input: faces
				},
				Payload::Expression(count),
				Goal::Faces,
				faces
			)
		},
		Err(e) if primary =>
		{
			// A keep, reroll, or explode clause after the dice count follows
			// no dice expression. A fix supplies the dice operator and faces
			// just after the count, and the clause then modifies the dice
			// expression. If the policy declines, the expression ends at the
			// dice count, as in the ordinary parse.
			if rec.recovering()
				&& begins_misplaced_clause(skip_whitespace(rest).fragment())
					.is_some()
				&& rec.offer(Site::DiceOperator, cut_error(e), stack)
			{
				let at = rest.location_offset();
				let faces = Expression::Constant(Constant {
					value: PLACEHOLDER_FACES,
					span: SourceSpan { start: at, end: at }
				});
				return dice_continue(
					standard_dice(count, faces),
					Some(PLACEHOLDER_FACES),
					rest,
					stack,
					rec
				);
			}
			succeed(rest, Value::Expression(count))
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::Dice`] or [`Goal::PrimaryDice`] with the outcome of standard
/// faces, the first alternative for the faces. If they failed recoverably, try
/// custom faces, the second.
///
/// # Parameters
/// - `start`: The start of the dice expression.
/// - `count`: The dice count.
/// - `input`: The input of the faces.
/// - `outcome`: The outcome of the standard faces.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_dice_faces<'src, R: Recovery<'src>, S: TokenSink>(
	start: usize,
	count: Expression<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let error =
		match with_context_on_error(input, STANDARD_FACES_CONTEXT, outcome)
		{
			Ok((rest, faces)) =>
			{
				let dice = standard_dice(count, faces.into_expression());
				let highest = highest_face(&dice);
				return dice_continue(dice, highest, rest, stack, rec);
			},
			Err(nom::Err::Error(e)) => e,
			Err(e) => return fail(e)
		};
	let custom = match context(CUSTOM_FACES_CONTEXT, custom_faces)
		.parse_complete(input)
	{
		Err(nom::Err::Failure(_)) if rec.recovering() =>
		{
			recover_custom_faces(input, stack, rec)
		},
		Ok((rest, faces)) =>
		{
			rec.report_literals(input, rest, Role::FacesBracket);
			Ok((rest, faces))
		},
		custom => custom
	};
	match custom
	{
		Ok((rest, faces)) =>
		{
			let end = rest.location_offset();
			let dice = DiceExpression::Custom(CustomDice {
				count: Box::new(count),
				faces,
				span: SourceSpan { start, end }
			});
			let highest = highest_face(&dice);
			dice_continue(dice, highest, rest, stack, rec)
		},
		Err(nom::Err::Error(e)) =>
		{
			let e = nom::Err::Failure(ParseError::append(
				input,
				ErrorKind::Alt,
				merge(error, e)
			));
			match rec.repair_faces(e, input, stack)
			{
				// The faces may have failed after a `-`, so parse them again,
				// with the repair's token overlaid.
				Ok(overlay) =>
				{
					rec.overlay = Some(overlay);
					call_with(
						stack,
						Frame::DiceFaces { start, input },
						Payload::Expression(count),
						Goal::Faces,
						input
					)
				},
				Err(e) => fail(e)
			}
		},
		Err(e) => fail(e)
	}
}

/// Answer standard dice with the specified dice count and faces.
///
/// # Parameters
/// - `count`: The dice count.
/// - `faces`: The faces.
///
/// # Returns
/// The dice expression.
#[inline]
fn standard_dice<'src>(
	count: Expression<'src>,
	faces: Expression<'src>
) -> DiceExpression<'src>
{
	let span = SourceSpan {
		start: count.span().start,
		end: faces.span().end
	};
	DiceExpression::Standard(StandardDice {
		count: Box::new(count),
		faces: Box::new(faces),
		span
	})
}

/// Answer the highest face of standard or custom dice, if their faces are
/// constant: the faces of standard dice, which show `0` if they have none, or
/// the greatest of custom faces. The clauses of a dice expression leave the
/// faces of its dice alone, so the parser answers this once, as it reads the
/// dice, rather than seeking the dice beneath the clauses.
///
/// # Parameters
/// - `dice`: The dice expression.
///
/// # Returns
/// The highest face, or `None` if the faces are not constant, or `dice` is
/// not standard or custom dice.
fn highest_face(dice: &DiceExpression<'_>) -> Option<i32>
{
	match dice
	{
		DiceExpression::Standard(StandardDice { faces, .. }) => match **faces
		{
			Expression::Constant(Constant { value, .. }) => Some(value.max(0)),
			_ => None
		},
		DiceExpression::Custom(CustomDice { faces, .. }) =>
		{
			faces.iter().copied().max()
		},
		_ => None
	}
}

/// Continue [`Goal::Dice`] or [`Goal::PrimaryDice`] after the dice expression
/// so far, as `fold_many0(alt((reroll_clause, explode_clause, drop_clause,
/// keep_clause)))`. If a reroll or explode clause follows, parse its
/// predicate; see [`read_predicate`]. If a drop or keep clause follows, parse
/// its direction and call its drop expression. Otherwise, finish with the dice
/// expression so far.
///
/// A drop clause is `multispace0`, then `tag("drop")`, then
/// `cut(preceded(multispace0, context(DROP_DIRECTION_CONTEXT,
/// alt((tag("lowest"), tag("highest"))))))`, then `opt(preceded(multispace0,
/// context(DROP_EXPRESSION_CONTEXT, drop_expression)))`. A keep clause is the
/// same, but with `tag("keep")`, [`KEEP_DIRECTION_CONTEXT`], and
/// [`KEEP_EXPRESSION_CONTEXT`]. A reroll clause is `multispace0`, then the
/// [keyword] `reroll`, and then its predicate and its limit. An explode clause
/// is the same, but with the [keyword] `explode`.
///
/// # Parameters
/// - `dice`: The dice expression so far.
/// - `highest`: The [highest face](highest_face) of its dice, if known, which
///   an explode clause that lacks its predicate supplies as its threshold.
/// - `rest`: The input after the dice expression so far.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn dice_continue<'src, R: Recovery<'src>, S: TokenSink>(
	dice: DiceExpression<'src>,
	highest: Option<i32>,
	rest: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let verb: IResult<Span, PredicatedKind, ParseError> = preceded(
		multispace0,
		alt((
			map(keyword("reroll"), |_| PredicatedKind::Reroll),
			map(keyword("explode"), |_| PredicatedKind::Explode)
		))
	)
	.parse_complete(rest);
	if let Ok((after_keyword, kind)) = verb
	{
		rec.report_read(rest, after_keyword, kind.role());
		let clause = PredicatedClause {
			dice: Some((dice, highest)),
			start: None,
			predicate: None
		};
		return read_predicate(
			kind,
			Box::new(clause),
			after_keyword,
			stack,
			rec
		);
	}
	let keyword: IResult<Span, bool, ParseError> = preceded(
		multispace0,
		alt((map(tag("drop"), |_| false), map(tag("keep"), |_| true)))
	)
	.parse_complete(rest);
	let (after_keyword, keep) = match keyword
	{
		Ok((after_keyword, keep)) =>
		{
			rec.report_read(rest, after_keyword, Role::Selection);
			(after_keyword, keep)
		},
		// `fold_many0` stops at the first recoverable error, and discards it.
		// The keyword cannot fail unrecoverably.
		Err(_) => return succeed(rest, Value::Dice(dice))
	};
	let label = if keep
	{
		KEEP_DIRECTION_CONTEXT
	}
	else
	{
		DROP_DIRECTION_CONTEXT
	};
	let direction: IResult<Span, bool, ParseError> = cut(preceded(
		multispace0,
		context(
			label,
			alt((map(tag("lowest"), |_| true), map(tag("highest"), |_| false)))
		)
	))
	.parse_complete(after_keyword);
	let (after_direction, kind) = match direction
	{
		Ok((after_direction, lowest)) =>
		{
			rec.report_read(after_keyword, after_direction, Role::Direction);
			(after_direction, ClauseKind::of(keep, lowest))
		},
		// The repair supplies the commonest direction of the verb: `drop
		// lowest`, or `keep highest`.
		Err(e) => match rec.repair(Site::Direction { keep }, e, stack)
		{
			Ok(at) => (at, ClauseKind::of(keep, !keep)),
			Err(e) => return fail(e)
		}
	};
	let input = skip_whitespace(after_direction);
	if begins_with_minus(input)
	{
		return finish_drop_clause(
			dice,
			highest,
			kind,
			after_direction,
			None,
			stack,
			rec
		);
	}
	call_with(
		stack,
		Frame::DiceDrop { kind, input },
		Payload::Drop {
			dice,
			highest,
			rest: after_direction
		},
		Goal::Atom,
		input
	)
}

/// Resume [`Goal::Dice`] or [`Goal::PrimaryDice`] with the outcome of the
/// optional drop expression of a drop or keep clause.
///
/// # Parameters
/// - `dice`: The dice expression that the clause modifies, and the [highest
///   face](highest_face) of its dice, if known.
/// - `kind`: The kind of the clause.
/// - `rest`: The input after the direction.
/// - `input`: The input of the drop expression.
/// - `outcome`: The outcome of the drop expression.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_dice_drop<'src, R: Recovery<'src>, S: TokenSink>(
	(dice, highest): (DiceExpression<'src>, Option<i32>),
	kind: ClauseKind,
	rest: Span<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match optional_drop(kind, rest, input, outcome)
	{
		Ok((rest, drop)) =>
		{
			finish_drop_clause(dice, highest, kind, rest, drop, stack, rec)
		},
		Err(e) => fail(e)
	}
}

/// Finish a drop or keep clause of [`Goal::Dice`] or [`Goal::PrimaryDice`],
/// and look for another.
///
/// # Parameters
/// - `dice`: The dice expression that the clause modifies.
/// - `highest`: The [highest face](highest_face) of the dice, if known.
/// - `kind`: The kind of the clause.
/// - `rest`: The input after the clause.
/// - `drop`: The drop expression of the clause, if any.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn finish_drop_clause<'src, R: Recovery<'src>, S: TokenSink>(
	dice: DiceExpression<'src>,
	highest: Option<i32>,
	kind: ClauseKind,
	rest: Span<'src>,
	drop: Option<Box<Expression<'src>>>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let span = SourceSpan {
		start: dice.span().start,
		end: rest.location_offset()
	};
	dice_continue(kind.build(dice, drop, span), highest, rest, stack, rec)
}

/// Continue a reroll or explode clause after its verb, `reroll` or `explode`,
/// by parsing its predicate.
///
/// The predicate is `preceded(multispace0, alt((comparison, list, range)))`:
///
/// - A comparison is a comparator, `=`, `!=` or `≠`, `<`, `<=` or `≤`, `>`, or
///   `>=` or `≥`, then `cut(preceded(multispace0, context(threshold,
///   standard_faces)))`.
/// - A list is the [keyword] `in`, then `[`, then `separated_list0(comma,
///   integer)`, then [a closing bracket](close). Its values are literals, and
///   it may be empty.
/// - A range is the [keyword] `in`, then `cut(preceded(multispace0,
///   context(range_start, standard_faces)))`, then `cut(preceded(multispace0,
///   context(range_operator, alt((tag("..="), tag(".."))))))`, then
///   `cut(preceded(multispace0, context(range_end, standard_faces)))`.
///
/// The predicate itself is cut, under the `predicate` context, since nothing
/// else may follow the verb. The limit follows; see [`read_limit`]. The
/// [contexts](PredicatedKind::contexts) are those of the kind of the clause.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `clause`: The parts of the clause read so far.
/// - `after_keyword`: The input after the verb.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn read_predicate<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	clause: Box<PredicatedClause<'src>>,
	after_keyword: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let comparator: IResult<Span, Comparison, ParseError> = preceded(
		multispace0,
		alt((
			map(alt((tag("<="), tag("≤"))), |_| Comparison::Le),
			map(tag("<"), |_| Comparison::Lt),
			map(alt((tag(">="), tag("≥"))), |_| Comparison::Ge),
			map(tag(">"), |_| Comparison::Gt),
			map(alt((tag("!="), tag("≠"))), |_| Comparison::Ne),
			map(tag("="), |_| Comparison::Eq)
		))
	)
	.parse_complete(after_keyword);
	if let Ok((after, comparison)) = comparator
	{
		rec.report_read(after_keyword, after, Role::Predicate);
		let input = skip_whitespace(after);
		return call_with(
			stack,
			Frame::PredicateFirst {
				kind,
				comparison: Some(comparison),
				input
			},
			Payload::Predicated(clause),
			Goal::Faces,
			input
		);
	}
	let within: IResult<Span, Span, ParseError> =
		preceded(multispace0, keyword("in")).parse_complete(after_keyword);
	if let Ok((after, _)) = within
	{
		rec.report_read(after_keyword, after, Role::Predicate);
		let input = skip_whitespace(after);
		if input.fragment().starts_with('[')
		{
			return match read_values(input, stack, rec)
			{
				Ok((rest, values)) => read_limit(
					kind,
					clause,
					Predicate::In(values),
					rest,
					stack,
					rec
				),
				Err(e) => fail(e)
			};
		}
		return call_with(
			stack,
			Frame::PredicateFirst {
				kind,
				comparison: None,
				input
			},
			Payload::Predicated(clause),
			Goal::Faces,
			input
		);
	}
	// Nothing else may follow the verb. The repair supplies the commonest
	// predicate of the verb: `= 1` for a reroll, and the highest face for an
	// explode.
	let at = skip_whitespace(after_keyword);
	let e = nom::Err::Failure(ParseError::add_context(
		at,
		kind.contexts().predicate,
		ParseError::from_error_kind(at, ErrorKind::Tag)
	));
	let highest = clause.dice.as_ref().and_then(|(_, highest)| *highest);
	let threshold = kind.placeholder_threshold(highest);
	match rec.repair(Site::Predicate { kind, threshold }, e, stack)
	{
		Ok(at) =>
		{
			let position = at.location_offset();
			let threshold = Expression::Constant(Constant {
				value: threshold,
				span: SourceSpan {
					start: position,
					end: position
				}
			});
			let predicate = Comparison::Eq.predicate(threshold);
			read_limit(kind, clause, predicate, at, stack, rec)
		},
		Err(e) => fail(e)
	}
}

/// Parse the list of a reroll or explode clause, `[`, then, unless `]`
/// follows, `separated_list1(preceded(multispace0, char(',')),
/// cut(preceded(multispace0, context(CONSTANT_CONTEXT, integer))))`, then [a
/// closing bracket](close), whose absence a repair may supply. A value must
/// follow a `,`, so a malformed value fails where it is, rather than ending
/// the list before the `,`; see [`read_more_values`].
///
/// # Parameters
/// - `input`: The input of the list, which begins with `[`.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The input after the list, and its values.
///
/// # Errors
/// * [`Err`](nom::Err) if the list could not be parsed, or repaired.
fn read_values<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, Vec<i32>, ParseError<'src>>
{
	let opener = input.location_offset();
	let (after, _) = char('[').parse_complete(input)?;
	let mut values = Vec::new();
	let (after, retracted) =
		if skip_whitespace(after).fragment().starts_with(']')
		{
			(after, None)
		}
		else
		{
			let after = match read_value(opener, after, &mut values, stack, rec)
			{
				Ok(after) => after,
				Err(e) =>
				{
					// The `[` was read, even though the parse fails.
					rec.report_literals(input, after, Role::ListBracket);
					return Err(e);
				}
			};
			match read_more_values(opener, after, &mut values, stack, rec)
			{
				Ok(more) => more,
				Err((at, e)) =>
				{
					// The values were read, even though the parse fails.
					rec.report_literals(input, at, Role::ListBracket);
					return Err(e);
				}
			}
		};
	let rest = match close(']', CLOSING_BRACKET_CONTEXT, after)
	{
		Ok(rest) => rest,
		Err(e) =>
		{
			let site = Site::Closer {
				opener,
				closer: ']'
			};
			// The supplied `]` is not in the source, so is not reported, but
			// the values were read, even if the parse fails.
			match rec.repair(site, e, stack)
			{
				Ok(rest) => rest,
				Err(e) =>
				{
					rec.report_literals(input, after, Role::ListBracket);
					return Err(e);
				}
			}
		}
	};
	report_list(input, retracted, rest, Role::ListBracket, rec);
	Ok((rest, values))
}

/// Answer the parser of a value of a list, a signed literal after any
/// whitespace, as `preceded(multispace0, context(CONSTANT_CONTEXT,
/// map(integer, |c| c.value)))`.
///
/// # Returns
/// The parser of the value.
#[inline(always)]
fn value<'src>()
-> impl nom::Parser<Span<'src>, Output = i32, Error = ParseError<'src>>
{
	preceded(
		multispace0,
		context(CONSTANT_CONTEXT, map(integer, |c| c.value))
	)
}

/// The input before and after a comma that a [repair](Site::TrailingComma)
/// retracted from a list of values, as [`read_more_values`] answers it.
type RetractedComma<'src> = (Span<'src>, Span<'src>);

/// Parse the rest of a list of values after its first, the custom faces of
/// dice or the list of a reroll or explode clause, as
/// `many0(preceded(preceded(multispace0, char(',')), cut(value)))`: once a
/// `,` is read, a value must follow it, so that a malformed value fails where
/// it is, rather than ending the list before the `,`.
///
/// While recovering, a `,` that a `]` follows offers a repair at
/// [`Site::TrailingComma`], whose fix retracts the comma, which then ends the
/// list. Otherwise, the ordinary parse fails at the `]`, where it expects a
/// value, and a value that fails may be [repaired](read_value).
///
/// # Parameters
/// - `opener`: The byte offset of the `[` of the list.
/// - `after`: The input after the first value.
/// - `values`: The values, to which the rest are appended.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The input after the last value, or after the retracted comma, and the
/// input before and after the retracted comma, if any.
///
/// # Errors
/// The input after the last value that was read, and the error of the value
/// that failed, if no repair supplied it.
fn read_more_values<'src, R: Recovery<'src>, S: TokenSink>(
	opener: usize,
	mut after: Span<'src>,
	values: &mut Vec<i32>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Result<
	(Span<'src>, Option<RetractedComma<'src>>),
	(Span<'src>, nom::Err<ParseError<'src>>)
>
{
	loop
	{
		let comma: IResult<Span, char, ParseError> =
			preceded(multispace0, char(',')).parse_complete(after);
		let Ok((next, _)) = comma
		else
		{
			return Ok((after, None));
		};
		if rec.recovering()
			&& skip_whitespace(next).fragment().starts_with(']')
			&& rec.offer(
				Site::TrailingComma,
				nom::Err::Failure(ParseError::from_char(
					skip_whitespace(after),
					']'
				)),
				stack
			)
		{
			// Continue as though the source lacked the comma.
			return Ok((next, Some((after, next))));
		}
		after = read_value(opener, next, values, stack, rec)
			.map_err(|e| (after, e))?;
	}
}

/// Parse a value of a list, as `cut(value)`, and append it to the values.
/// While recovering, a value that fails offers a repair at
/// [`Site::FaceValue`], whose fix supplies [`PLACEHOLDER_FACE`].
///
/// # Parameters
/// - `opener`: The byte offset of the `[` of the list.
/// - `input`: The input of the value.
/// - `values`: The values, to which the value is appended.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The input after the value.
///
/// # Errors
/// * [`Err`](nom::Err) if the value could not be parsed, or repaired.
fn read_value<'src, R: Recovery<'src>, S: TokenSink>(
	opener: usize,
	input: Span<'src>,
	values: &mut Vec<i32>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
{
	match cut(value()).parse_complete(input)
	{
		Ok((rest, value)) =>
		{
			values.push(value);
			Ok(rest)
		},
		Err(e) =>
		{
			let at = rec.repair(Site::FaceValue { opener }, e, stack)?;
			values.push(PLACEHOLDER_FACE);
			Ok(at)
		}
	}
}

/// Report a list of values that the engine has read, but not a comma that a
/// repair [retracted](read_more_values), since the source as repaired lacks
/// it.
///
/// # Parameters
/// - `input`: The input of the list, which begins with `[`.
/// - `retracted`: The input before and after the retracted comma, if any.
/// - `rest`: The input after the list.
/// - `bracket`: The role of the brackets.
/// - `rec`: The recovery state.
fn report_list<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	retracted: Option<(Span<'src>, Span<'src>)>,
	rest: Span<'src>,
	bracket: Role,
	rec: &mut Recoverer<'src, '_, R, S>
)
{
	match retracted
	{
		Some((before, after)) =>
		{
			rec.report_literals(input, before, bracket);
			rec.report_literals(after, rest, bracket);
		},
		None => rec.report_literals(input, rest, bracket)
	}
}

/// Resume a reroll or explode clause with the outcome of the threshold of a
/// comparison, or of the start of a range, after which the range continues
/// with its operator and its end.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `clause`: The parts of the clause read so far.
/// - `comparison`: The comparator, or `None` if the predicate is a range.
/// - `input`: The input of the threshold or the start.
/// - `outcome`: The outcome of the threshold or the start.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_predicate_first<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	mut clause: Box<PredicatedClause<'src>>,
	comparison: Option<Comparison>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let label = first_context(kind, comparison);
	let (rest, first) = match outcome
	{
		Ok((rest, first)) => (rest, first.into_expression()),
		Err(e) => return fail(cut_error(with_context(input, label, e)))
	};
	if let Some(comparison) = comparison
	{
		let predicate = comparison.predicate(first);
		return read_limit(kind, clause, predicate, rest, stack, rec);
	}
	let operator: IResult<Span, bool, ParseError> = cut(preceded(
		multispace0,
		context(
			kind.contexts().range_operator,
			alt((map(tag("..="), |_| true), map(tag(".."), |_| false)))
		)
	))
	.parse_complete(rest);
	// The repair supplies the inclusive operator, `..=`, since the author of
	// two endpoints most likely means to include both.
	let (after, inclusive) = match operator
	{
		Ok((after, inclusive)) =>
		{
			rec.report_read(rest, after, Role::RangeOperator);
			(after, inclusive)
		},
		Err(e) => match rec.repair(Site::RangeOperator { kind }, e, stack)
		{
			Ok(at) => (at, true),
			Err(e) => return fail(e)
		}
	};
	clause.start = Some(first);
	let input = skip_whitespace(after);
	call_with(
		stack,
		Frame::PredicateEnd {
			kind,
			inclusive,
			input
		},
		Payload::Predicated(clause),
		Goal::Faces,
		input
	)
}

/// Answer the context classifier of the first operand of a predicate.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `comparison`: The comparator, or `None` if the predicate is a range.
///
/// # Returns
/// The [classifier](PredicatedContexts::threshold) of the threshold of a
/// comparison, or that of the [start](PredicatedContexts::range_start) of a
/// range, for clauses of the kind.
const fn first_context(
	kind: PredicatedKind,
	comparison: Option<Comparison>
) -> &'static str
{
	let contexts = kind.contexts();
	match comparison
	{
		Some(_) => contexts.threshold,
		None => contexts.range_start
	}
}

/// Resume a reroll or explode clause with the outcome of the end of a range.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `clause`: The parts of the clause read so far, among them the start of the
///   range.
/// - `inclusive`: Whether the range is inclusive.
/// - `input`: The input of the end.
/// - `outcome`: The outcome of the end.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_predicate_end<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	mut clause: Box<PredicatedClause<'src>>,
	inclusive: bool,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, end) = match outcome
	{
		Ok((rest, end)) => (rest, Box::new(end.into_expression())),
		Err(e) =>
		{
			return fail(cut_error(with_context(
				input,
				kind.contexts().range_end,
				e
			)));
		}
	};
	let start = Box::new(clause.start.take().expect("a range has a start"));
	let predicate = if inclusive
	{
		Predicate::InRangeInclusive { start, end }
	}
	else
	{
		Predicate::InRange { start, end }
	};
	read_limit(kind, clause, predicate, rest, stack, rec)
}

/// Continue a reroll or explode clause after its predicate, by parsing its
/// optional limit: `opt(alt((unlimited, count)))`, where `unlimited` is
/// `preceded(multispace0, alt((keyword("unlimited"), tag("∞"))))`, then
/// [times], and `count` is `preceded(multispace0, context(count,
/// drop_expression))`, then [times], under the
/// [context](PredicatedContexts::count) of the kind of the clause. A count
/// never begins with `-`, so after a predicate, a `-` is subtraction, as after
/// a drop clause; see [`begins_with_minus`].
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `clause`: The parts of the clause read so far.
/// - `predicate`: The predicate.
/// - `rest`: The input after the predicate, where the clause ends if it has no
///   limit.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn read_limit<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	mut clause: Box<PredicatedClause<'src>>,
	predicate: Predicate<Box<Expression<'src>>>,
	rest: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let unlimited: IResult<Span, Span, ParseError> =
		preceded(multispace0, alt((keyword("unlimited"), tag("∞"))))
			.parse_complete(rest);
	if let Ok((after, _)) = unlimited
	{
		rec.report_read(rest, after, Role::Limit);
		return match times(kind, after, false, stack, rec)
		{
			Ok(after) => finish_predicated(
				kind,
				clause.dice,
				predicate,
				Some(Limit::Unlimited),
				after,
				stack,
				rec
			),
			Err(e) => fail(e)
		};
	}
	let input = skip_whitespace(rest);
	if begins_with_minus(input)
	{
		return finish_predicated(
			kind,
			clause.dice,
			predicate,
			None,
			rest,
			stack,
			rec
		);
	}
	clause.predicate = Some((predicate, rest));
	call_with(
		stack,
		Frame::LimitCount { kind, input },
		Payload::Predicated(clause),
		Goal::Atom,
		input
	)
}

/// Resume a reroll or explode clause with the outcome of the optional count of
/// its limit.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `clause`: The parts of the clause read so far, among them the predicate.
/// - `input`: The input of the count.
/// - `outcome`: The outcome of the count.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_limit_count<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	clause: PredicatedClause<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let PredicatedClause {
		dice, predicate, ..
	} = clause;
	let (predicate, rest) = predicate.expect("a limit follows a predicate");
	match with_context_on_error(input, kind.contexts().count, outcome)
	{
		Ok((after, count)) =>
		{
			let count = count.into_expression();
			let singular = matches!(
				count,
				Expression::Constant(Constant { value: 1, .. })
			);
			match times(kind, after, singular, stack, rec)
			{
				Ok(after) => finish_predicated(
					kind,
					dice,
					predicate,
					Some(Limit::Times(Box::new(count))),
					after,
					stack,
					rec
				),
				Err(e) => fail(e)
			}
		},
		// `opt` rewinds after a recoverable error, and discards it.
		Err(nom::Err::Error(_)) =>
		{
			finish_predicated(kind, dice, predicate, None, rest, stack, rec)
		},
		Err(e) => fail(e)
	}
}

/// Parse the `time` or `times` that follows the count of a limit, or `∞` or
/// `unlimited`, as `cut(preceded(multispace0, context(times,
/// alt((keyword("times"), keyword("time"))))))`, under the
/// [context](PredicatedContexts::times) of the kind of the clause, whose
/// absence a repair may supply.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `input`: The input after the count, `∞`, or `unlimited`.
/// - `singular`: Whether the count is the constant `1`, after which a repair
///   supplies `time` rather than `times`.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The input after the `time` or `times`.
///
/// # Errors
/// * [`Failure`](nom::Err::Failure) if it is missing, and not repaired.
fn times<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	input: Span<'src>,
	singular: bool,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
{
	let parsed: IResult<Span, Span, ParseError> = cut(preceded(
		multispace0,
		context(
			kind.contexts().times,
			alt((keyword("times"), keyword("time")))
		)
	))
	.parse_complete(input);
	match parsed
	{
		Ok((after, _)) =>
		{
			rec.report_read(input, after, Role::Limit);
			Ok(after)
		},
		Err(e) => rec.repair(Site::Times { kind, singular }, e, stack)
	}
}

/// Finish a reroll or explode clause: as a clause of [`Goal::Dice`] or
/// [`Goal::PrimaryDice`], whose dice expression it modifies, after which look
/// for another clause; or as [`Goal::Reroll`] or [`Goal::Explode`].
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `dice`: The dice expression that the clause modifies, and the [highest
///   face](highest_face) of its dice, if known; or `None` if the clause is
///   parsed alone, as [`Goal::Reroll`] or [`Goal::Explode`].
/// - `predicate`: The predicate.
/// - `limit`: The limit, if any.
/// - `rest`: The input after the clause.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn finish_predicated<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	dice: Option<(DiceExpression<'src>, Option<i32>)>,
	predicate: Predicate<Box<Expression<'src>>>,
	limit: Option<Limit<Box<Expression<'src>>>>,
	rest: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let Some((dice, highest)) = dice
	else
	{
		return succeed(rest, Value::Predicated { predicate, limit });
	};
	let span = SourceSpan {
		start: dice.span().start,
		end: rest.location_offset()
	};
	let dice = kind.build(dice, predicate, limit, span);
	dice_continue(dice, highest, rest, stack, rec)
}

/// Start [`Goal::Reroll`] or [`Goal::Explode`]: the [keyword] of the verb,
/// `reroll` or `explode`, then the predicate and the limit of the clause, as
/// [`read_predicate`] describes.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_predicated<'src, R: Recovery<'src>, S: TokenSink>(
	kind: PredicatedKind,
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let verb: IResult<Span, Span, ParseError> =
		keyword(kind.keyword()).parse_complete(input);
	match verb
	{
		Ok((after_keyword, _)) =>
		{
			rec.report_read(input, after_keyword, kind.role());
			let clause = PredicatedClause {
				dice: None,
				start: None,
				predicate: None
			};
			read_predicate(kind, Box::new(clause), after_keyword, stack, rec)
		},
		Err(e) => fail(e)
	}
}

/// Start [`Goal::Extremum`]: the keyword, `highest` or `lowest`, then
/// `preceded(multispace0, char('('))`, then `cut(preceded(multispace0,
/// context(ARGUMENT_CONTEXT, expression)))`, then the rest of the arguments;
/// see [`call_continue`]. The keyword must [begin a call](begins_call), so the
/// call fails recoverably at its input unless its `(` follows, and commits once
/// it does. A repair may instead [overlay](Overlay::Call) the keyword, as it
/// does for `max` in `max(1, 2)`.
///
/// # Parameters
/// - `input`: The input of the production.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_extremum<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let overlaid = if rec.recovering()
	{
		rec.take_call(input)
	}
	else
	{
		None
	};
	let keyword = overlaid.or_else(|| {
		begins_call(input.fragment()).map(|direction| {
			(input.take_from(direction.keyword().len()), direction)
		})
	});
	let paren = keyword.and_then(|(after, direction)| {
		let paren: IResult<Span, char, ParseError> =
			preceded(multispace0, char('(')).parse_complete(after);
		paren.ok().map(|(after, _)| (after, direction))
	});
	let Some((after, direction)) = paren
	else
	{
		return fail(nom::Err::Error(ParseError::from_error_kind(
			input,
			ErrorKind::Tag
		)));
	};
	if overlaid.is_none()
	{
		let start = input.location_offset();
		rec.report(start, start + direction.keyword().len(), Role::Call);
	}
	rec.report_before(after, Role::Parenthesis);
	let opener = after.location_offset() - 1;
	let call = Call {
		start: input.location_offset(),
		opener,
		direction,
		arguments: Vec::new(),
		first: SourceSpan::default()
	};
	argument(Box::new(call), skip_whitespace(after), stack)
}

/// Call the next argument of a call.
///
/// # Parameters
/// - `call`: The call so far.
/// - `input`: The input of the argument.
/// - `stack`: The stack of suspended productions.
///
/// # Returns
/// The next step.
fn argument<'src>(
	call: Box<Call<'src>>,
	input: Span<'src>,
	stack: &mut Stack<'src>
) -> Step<'src>
{
	call_with(
		stack,
		Frame::CallArgument {
			opener: call.opener,
			input
		},
		Payload::Call(call),
		Goal::AddSub,
		input
	)
}

/// Resume [`Goal::Extremum`] with the outcome of an argument.
///
/// # Parameters
/// - `call`: The call so far.
/// - `input`: The input of the argument.
/// - `outcome`: The outcome of the argument.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_call_argument<'src, R: Recovery<'src>, S: TokenSink>(
	mut call: Box<Call<'src>>,
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	match outcome
	{
		Ok((rest, argument)) =>
		{
			if call.arguments.is_empty()
			{
				call.first = SourceSpan {
					start: input.location_offset(),
					end: rest.location_offset()
				};
			}
			call.arguments.push(argument.into_expression());
			call_continue(call, rest, stack, rec)
		},
		Err(e) => fail(cut_error(with_context(input, ARGUMENT_CONTEXT, e)))
	}
}

/// Continue [`Goal::Extremum`] after an argument. After the first, only a `,`
/// may follow, as `cut(preceded(multispace0, char(',')))`, since a call has
/// two or more arguments; after a later one, a `,` or the closing `)` may, as
/// `cut(preceded(multispace0, alt((char(','), char(')')))))`. After a `,`,
/// call the next argument; after the `)`, finish the call.
///
/// While recovering, a `,` that a `)` follows offers a repair at
/// [`Site::TrailingComma`], whose fix retracts the comma. If neither a `,` nor
/// a `)` follows where one must, the failure is at [`Site::Comma`] after the
/// first argument, or wherever the next token [begins an
/// operand](begins_operand), as another argument would, and at a
/// [`Site::Closer`] otherwise.
///
/// # Parameters
/// - `call`: The call so far.
/// - `rest`: The input after the last argument.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn call_continue<'src, R: Recovery<'src>, S: TokenSink>(
	call: Box<Call<'src>>,
	rest: Span<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let separator: IResult<Span, char, ParseError> = if call.arguments.len() < 2
	{
		cut(preceded(multispace0, char(','))).parse_complete(rest)
	}
	else
	{
		cut(preceded(multispace0, alt((char(','), char(')')))))
			.parse_complete(rest)
	};
	match separator
	{
		Ok((after, ',')) =>
		{
			let next = skip_whitespace(after);
			if rec.recovering()
				&& next.fragment().starts_with(')')
				&& rec.offer(
					Site::TrailingComma,
					nom::Err::Failure(ParseError::from_char(
						skip_whitespace(rest),
						')'
					)),
					stack
				)
			{
				// Continue as though the source lacked the comma.
				return call_continue(call, next, stack, rec);
			}
			rec.report_before(after, Role::Separator);
			argument(call, next, stack)
		},
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::Parenthesis);
			finish_call(*call, after)
		},
		Err(e) => recover_separator(call, rest, e, stack, rec)
	}
}

/// Recover from a missing `,` or `)` after an argument of a call, unless the
/// engine is not recovering.
///
/// # Parameters
/// - `call`: The call so far.
/// - `rest`: The input after the last argument.
/// - `e`: The error of the missing separator.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn recover_separator<'src, R: Recovery<'src>, S: TokenSink>(
	call: Box<Call<'src>>,
	rest: Span<'src>,
	e: nom::Err<ParseError<'src>>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	if !rec.recovering()
	{
		return fail(e);
	}
	let next = skip_whitespace(rest);
	let site = match &call.arguments[..]
	{
		[only] =>
		{
			let lone = next.fragment().starts_with(')').then(|| LoneArgument {
				call: call.start,
				argument: call.first,
				direction: call.direction,
				dice: matches!(only, Expression::Dice(_)),
				bare: stands_bare(only)
			});
			Site::Comma {
				opener: call.opener,
				lone
			}
		},
		_ if begins_operand(next.fragment()) => Site::Comma {
			opener: call.opener,
			lone: None
		},
		_ => Site::Closer {
			opener: call.opener,
			closer: ')'
		}
	};
	// A `,` is missing again where a repair supplied one, so the argument
	// after it read nothing, and another would read nothing too.
	if matches!(site, Site::Comma { .. })
		&& rec.comma == Some((call.opener, next.location_offset()))
	{
		rec.active = false;
		return fail(e);
	}
	let repaired =
		rec.consult_with(site, e, stack, |repair, at| match (repair, site)
		{
			(Repair::Fix, _) => Some((repair, at)),
			(Repair::Unwrap, Site::Comma { lone: Some(_), .. })
			| (
				Repair::Keep,
				Site::Comma {
					lone: Some(LoneArgument { dice: true, .. }),
					..
				}
			) => Some((repair, at)),
			_ => None
		});
	match (repaired, site)
	{
		(Err(e), _) => fail(e),
		(Ok((Repair::Fix, at)), Site::Comma { .. }) =>
		{
			rec.comma = Some((call.opener, at.location_offset()));
			argument(call, at, stack)
		},
		// The supplied `)` ends the call at the failure.
		(Ok((Repair::Fix, at)), _) => finish_call(*call, at),
		(Ok((repair, at)), _) =>
		{
			// The `)` of the call is in the source, and still closes its `(`.
			rec.report_before(at.take_from(1), Role::Parenthesis);
			unwrap_call(*call, repair == Repair::Keep, at)
		}
	}
}

/// Finish a call of `highest` or `lowest`.
///
/// # Parameters
/// - `call`: The call.
/// - `rest`: The input after the call.
///
/// # Returns
/// The next step.
fn finish_call<'src>(call: Call<'src>, rest: Span<'src>) -> Step<'src>
{
	succeed(
		rest,
		Value::Expression(Expression::Extremum(Extremum {
			direction: call.direction,
			arguments: call.arguments,
			span: SourceSpan {
				start: call.start,
				end: rest.location_offset()
			}
		}))
	)
}

/// Replace a call of one argument with that argument, as a
/// [repair](Repair::Unwrap) at [`Site::Comma`] does: grouped, unless it
/// [stands bare](stands_bare), or, for [`Repair::Keep`], with a keep clause of
/// the direction of the call.
///
/// # Parameters
/// - `call`: The call, whose only argument is a dice expression if `keep`.
/// - `keep`: Whether to keep the highest or lowest die of the argument.
/// - `at`: The input at the `)` of the call.
///
/// # Returns
/// The next step.
fn unwrap_call<'src>(
	mut call: Call<'src>,
	keep: bool,
	at: Span<'src>
) -> Step<'src>
{
	let rest = at.take_from(1);
	let span = SourceSpan {
		start: call.start,
		end: rest.location_offset()
	};
	let argument = call.arguments.pop().expect("a call of one argument");
	let value = match into_dice(argument)
	{
		Ok(dice) if keep =>
		{
			let kind =
				ClauseKind::of(true, call.direction == Direction::Lowest);
			Expression::Dice(kind.build(dice, None, span))
		},
		Ok(dice) => Expression::Dice(dice),
		Err(argument) if stands_bare(&argument) => argument,
		Err(argument) => Expression::Group(Group {
			expression: Box::new(argument),
			span
		})
	};
	succeed(rest, Value::Expression(value))
}

/// Answer the dice expression that an expression is, if it is one.
///
/// # Parameters
/// - `expression`: The expression.
///
/// # Returns
/// The dice expression.
///
/// # Errors
/// * The expression itself, if it is not a dice expression.
fn into_dice(
	mut expression: Expression<'_>
) -> Result<DiceExpression<'_>, Expression<'_>>
{
	match &mut expression
	{
		// `Expression` implements `Drop`, so the dice expression cannot move
		// out of it; swap in a childless placeholder instead.
		Expression::Dice(dice) =>
		{
			let placeholder = DiceExpression::Custom(CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 0,
					span: SourceSpan::SYNTHETIC
				})),
				faces: Vec::new(),
				span: SourceSpan::SYNTHETIC
			});
			Ok(std::mem::replace(dice, placeholder))
		},
		_ => Err(expression)
	}
}

/// Answer whether an expression may stand bare in place of a call, as a
/// primary expression whose meaning no neighbor can change: anything but an
/// arithmetic expression, or a negative constant, which a `^` after it would
/// bind more tightly than its sign, e.g., `-1 ^ 2`.
///
/// # Parameters
/// - `expression`: The expression.
///
/// # Returns
/// `true` if the expression may stand bare, `false` if it must be grouped.
fn stands_bare(expression: &Expression<'_>) -> bool
{
	match expression
	{
		Expression::Constant(constant) => constant.value >= 0,
		Expression::Arithmetic(_) => false,
		_ => true
	}
}

/// Resume [`Goal::StandardDice`] with the outcome of the dice count.
///
/// [`Goal::StandardDice`] is `separated_pair(context(DICE_COUNT_CONTEXT,
/// dice_count), preceded(multispace0, d_operator), preceded(multispace0,
/// context(STANDARD_FACES_CONTEXT, standard_faces)))`.
///
/// # Parameters
/// - `input`: The input of the dice expression.
/// - `outcome`: The outcome of the dice count.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_standard_count<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, count) =
		match with_context_on_error(input, DICE_COUNT_CONTEXT, outcome)
		{
			Ok((rest, count)) => (rest, count.into_expression()),
			Err(e) => return fail(e)
		};
	let operator: IResult<Span, char, ParseError> =
		preceded(multispace0, d_operator).parse_complete(rest);
	match operator
	{
		Ok((after, _)) =>
		{
			rec.report_before(after, Role::DiceOperator);
			let input = skip_whitespace(after);
			call_with(
				stack,
				Frame::StandardFaces(input),
				Payload::Expression(count),
				Goal::Faces,
				input
			)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::StandardDice`] with the outcome of the faces.
///
/// # Parameters
/// - `count`: The dice count.
/// - `input`: The input of the faces.
/// - `outcome`: The outcome of the faces.
///
/// # Returns
/// The next step.
fn resume_standard_faces<'src>(
	count: Expression<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match with_context_on_error(input, STANDARD_FACES_CONTEXT, outcome)
	{
		Ok((rest, faces)) =>
		{
			let faces = faces.into_expression();
			let span = SourceSpan {
				start: count.span().start,
				end: faces.span().end
			};
			succeed(
				rest,
				Value::StandardDice(StandardDice {
					count: Box::new(count),
					faces: Box::new(faces),
					span
				})
			)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::CustomDice`] with the outcome of the dice count, and finish
/// it.
///
/// [`Goal::CustomDice`] is `separated_pair(context(DICE_COUNT_CONTEXT,
/// dice_count), preceded(multispace0, d_operator), preceded(multispace0,
/// context(CUSTOM_FACES_CONTEXT, custom_faces)))`.
///
/// # Parameters
/// - `input`: The input of the dice expression.
/// - `outcome`: The outcome of the dice count.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn resume_custom_count<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	outcome: Outcome<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (rest, count) =
		match with_context_on_error(input, DICE_COUNT_CONTEXT, outcome)
		{
			Ok((rest, count)) => (rest, count.into_expression()),
			Err(e) => return fail(e)
		};
	let faces = preceded(multispace0, d_operator)
		.and(preceded(
			multispace0,
			context(CUSTOM_FACES_CONTEXT, custom_faces)
		))
		.parse_complete(rest);
	match faces
	{
		Ok((after, (_, faces))) =>
		{
			let faces_input = skip_whitespace(rest).take_from(1);
			rec.report_before(faces_input, Role::DiceOperator);
			rec.report_literals(faces_input, after, Role::FacesBracket);
			let rest = after;
			let start = input.location_offset();
			let end = rest.location_offset();
			succeed(
				rest,
				Value::CustomDice(CustomDice {
					count: Box::new(count),
					faces,
					span: SourceSpan { start, end }
				})
			)
		},
		Err(e) => fail(e)
	}
}

/// Start [`Goal::DropLowest`], [`Goal::DropHighest`], [`Goal::KeepLowest`],
/// or [`Goal::KeepHighest`]: `preceded(multispace0, tag(VERB))`, then
/// `preceded(multispace0, tag(DIRECTION))`, then `opt(preceded(multispace0,
/// context(COUNT_CONTEXT, drop_expression)))`, where `COUNT_CONTEXT` is
/// [`DROP_EXPRESSION_CONTEXT`] or [`KEEP_EXPRESSION_CONTEXT`].
///
/// # Parameters
/// - `input`: The input of the production.
/// - `kind`: The kind of the clause.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The next step.
fn start_drop<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	kind: ClauseKind,
	stack: &mut Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Step<'src>
{
	let (verb, direction) = kind.keywords();
	let keywords: IResult<Span, (Span, Span), ParseError> =
		preceded(multispace0, tag(verb))
			.and(preceded(multispace0, tag(direction)))
			.parse_complete(input);
	match keywords
	{
		Ok((rest, (verb, direction))) =>
		{
			let (verb, direction) = (span_of(verb), span_of(direction));
			rec.report(verb.start, verb.end, Role::Selection);
			rec.report(direction.start, direction.end, Role::Direction);
			let input = skip_whitespace(rest);
			if begins_with_minus(input)
			{
				return succeed(rest, Value::Drop(None));
			}
			call_with(
				stack,
				Frame::DropExpression { kind, input },
				Payload::Rest(rest),
				Goal::Atom,
				input
			)
		},
		Err(e) => fail(e)
	}
}

/// Resume [`Goal::DropLowest`], [`Goal::DropHighest`], [`Goal::KeepLowest`],
/// or [`Goal::KeepHighest`] with the outcome of the optional drop expression,
/// and finish it.
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `rest`: The input after the direction.
/// - `input`: The input of the drop expression.
/// - `outcome`: The outcome of the drop expression.
///
/// # Returns
/// The next step.
fn resume_drop_expression<'src>(
	kind: ClauseKind,
	rest: Span<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>
) -> Step<'src>
{
	match optional_drop(kind, rest, input, outcome)
	{
		Ok((rest, drop)) => succeed(rest, Value::Drop(drop)),
		Err(e) => fail(e)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Recovery.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The value of the operand that a [repair](Repair::Fix) supplies wherever an
/// expression is missing.
pub(crate) const PLACEHOLDER_OPERAND: i32 = 0;

/// The value of the standard faces that a [repair](Repair::Fix) supplies
/// wherever the faces of a dice expression are missing.
pub(crate) const PLACEHOLDER_FACES: i32 = 6;

/// The value of the face that a [repair](Repair::Fix) supplies wherever custom
/// faces lack their first face.
pub(crate) const PLACEHOLDER_FACE: i32 = 0;

/// The threshold of the predicate, `=`, that a [repair](Repair::Fix) supplies
/// wherever a reroll clause lacks its predicate, since a reroll most often
/// rerolls ones. An explode clause instead gets the highest face of its dice;
/// see [`PredicatedKind::placeholder_threshold`].
pub(crate) const PLACEHOLDER_THRESHOLD: i32 = 1;

/// The name that a [repair](Repair::Fix) supplies wherever a variable or a
/// formal parameter lacks its name.
pub(crate) const PLACEHOLDER_NAME: &str = "x";

/// A policy for recovering from syntax errors, which [`run_recovering`]
/// consults wherever the parse cannot continue.
///
/// The engine never edits the source. A [repair](Repair) instead makes the
/// engine continue as though the source had been edited, e.g., as though a
/// missing closing parenthesis were present. Each repair corresponds to an
/// edit of the source, which the policy is responsible for recording: an
/// insertion of the supplied token at the [position](FailureSite::position) of
/// the failure, unless the [site](Site) says otherwise.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
pub(crate) trait Recovery<'src>
{
	/// Whether the policy takes part in recovery at all. Only [`Strict`], the
	/// policy of the ordinary parse, opts out, which removes every recovery
	/// hook from the engine at compile time.
	const ENABLED: bool = true;

	/// Decide how to repair a failure.
	///
	/// # Parameters
	/// - `site`: The failure.
	///
	/// # Returns
	/// The repair.
	fn repair(&mut self, site: &FailureSite<'src>) -> Repair;
}

/// The policy of the ordinary parse, which repairs nothing.
pub(crate) struct Strict;

impl<'src> Recovery<'src> for Strict
{
	const ENABLED: bool = false;

	fn repair(&mut self, _site: &FailureSite<'src>) -> Repair { Repair::Stop }
}

/// A receiver of the [tokens](Token) that the engine reads, each tagged with
/// its [role](Role), for [source classification](crate::classify).
///
/// The engine reports each token where a leaf of the grammar reads it, once
/// the alternative that reads it has committed, so the tokens arrive in source
/// order. A token read by an alternative that a [catching](Disposition::Catch)
/// frame then abandons is [retracted](Self::retract), so that the tokens of a
/// clean parse are exactly those of the source as parsed.
///
/// While recovering, the engine reports only the tokens of the source: never
/// one that a [repair](Repair) supplied, nor source that a repair discarded,
/// or read as something else, such as a name without its braces. Where a
/// repair parses a goal or the function again, the engine retracts the tokens
/// of the attempt that failed, and reports them again as it reads them again.
/// Where the policy declines a repair, the tokens read before the failure
/// remain.
pub(crate) trait TokenSink
{
	/// Whether the sink receives tokens at all. Only [`Silent`], the sink of
	/// the ordinary parse and of the doctor, opts out, which removes every
	/// report from the engine at compile time.
	const ENABLED: bool = true;

	/// Receive a token.
	///
	/// # Parameters
	/// - `span`: The span of the token.
	/// - `role`: The role of the token.
	fn token(&mut self, span: SourceSpan, role: Role);

	/// Discard every token that begins at or after a byte offset, i.e., every
	/// token reported since the engine read past the offset.
	///
	/// # Parameters
	/// - `from`: The byte offset.
	fn retract(&mut self, from: usize);
}

/// The sink of the ordinary parse, which receives nothing.
pub(crate) struct Silent;

impl TokenSink for Silent
{
	const ENABLED: bool = false;

	fn token(&mut self, _span: SourceSpan, _role: Role) {}

	fn retract(&mut self, _from: usize) {}
}

impl TokenSink for Vec<Token>
{
	fn token(&mut self, span: SourceSpan, role: Role)
	{
		self.push(Token {
			span,
			role,
			depth: None
		});
	}

	fn retract(&mut self, from: usize)
	{
		while self.last().is_some_and(|token| token.span.start >= from)
		{
			self.pop();
		}
	}
}

/// A repair of a [failure](FailureSite), as decided by a [policy](Recovery).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Repair
{
	/// Decline to repair the failure. The run then ends as the ordinary parse
	/// would from that point, with the error of the failure, and consults the
	/// policy no more. At [`Site::DiceOperator`], which is not a failure of the
	/// ordinary parse, the run instead continues as the ordinary parse does,
	/// and still consults the policy.
	Stop,

	/// Repair the failure in the canonical way for its [site](Site), by
	/// supplying the missing token at the [position](FailureSite::position) of
	/// the failure, as follows:
	///
	/// | Site                  | Supplied                                  |
	/// |-----------------------|-------------------------------------------|
	/// | [`Site::Goal`]        | an operand, [`PLACEHOLDER_OPERAND`]       |
	/// | [`Site::ParameterName`] | a parameter, `{`[`PLACEHOLDER_NAME`]`}` |
	/// | [`Site::ParameterColon`] | `:`                                    |
	/// | [`Site::Closer`]      | the closing delimiter; see [`Break`](Repair::Break) |
	/// | [`Site::Colon`]       | `:`                                       |
	/// | [`Site::BindingParen`] | `(`                                      |
	/// | [`Site::VariableName`] | a name, [`PLACEHOLDER_NAME`]             |
	/// | [`Site::Faces`]       | standard faces, [`PLACEHOLDER_FACES`]; see [`Reread`](Repair::Reread) |
	/// | [`Site::FaceValue`]   | a face, [`PLACEHOLDER_FACE`]              |
	/// | [`Site::Direction`]   | `lowest` after `drop`, `highest` after `keep` |
	/// | [`Site::DiceOperator`] | a dice operator and standard faces, [`PLACEHOLDER_FACES`] |
	/// | [`Site::Predicate`]   | `=` and its threshold                     |
	/// | [`Site::RangeOperator`] | `..=`                                   |
	/// | [`Site::Times`]       | `time` after the count `1`, and `times` otherwise |
	/// | [`Site::Comma`]       | `,`, before another argument              |
	/// | [`Site::TrailingComma`] | nothing: the comma is retracted, as though the source lacked it |
	/// | [`Site::UngroupedCall`] | `(` before the call, and then `)` after it, at a [`Site::Closer`] |
	/// | [`Site::TrailingInput`] | nothing: the trailing input is discarded |
	///
	/// At [`Site::Goal`], the edit must separate the operand from a `-` just
	/// before it, e.g., by a space: in the source, `-0` is a constant, which a
	/// production that the parse has already finished might have read. At
	/// [`Site::Faces`], or at a [`Site::Goal`] of an operand of a reroll, it
	/// must not, since the faces or the operand are the constant that begins
	/// at such a `-`. At [`Site::LeadingComma`], which has no fix, this
	/// is the same as [`Stop`](Repair::Stop).
	Fix,

	/// Only at [`Site::Goal`], [`Site::Faces`], and [`Site::ParameterName`]:
	/// read the identifier that ends at `end` as a name, as though the source
	/// enclosed it in braces. At [`Site::Goal`] and [`Site::Faces`], the
	/// identifier begins at the [position](FailureSite::position) of the
	/// failure, and the engine parses the goal or the faces again, reading
	/// the identifier as the name of a [variable](Variable); at the start of
	/// the body of a function without formal parameters, if a `,` or `:`
	/// follows the identifier, the engine parses the function again instead,
	/// reading it as the first formal parameter, e.g., `x` in `x, y: 1`. At
	/// [`Site::Faces`], or at a [`Site::Goal`] of an operand of a reroll, the
	/// identifier must begin the faces or the operand, e.g., not follow a
	/// `-`. At [`Site::ParameterName`], the identifier begins after any
	/// [whitespace between tokens](crate::parser::is_token_space) after the
	/// comma, and is the name of the next formal parameter, e.g., `y` in `{x},
	/// y: 1`. Elsewhere, or otherwise, this is the same as
	/// [`Stop`](Repair::Stop).
	///
	/// # Panics
	/// If `end` is not a character boundary of the source.
	Variable
	{
		/// The byte offset of the end of the name, which must exceed the
		/// position of the failure.
		end: usize
	},

	/// Only at [`Site::Goal`]: discard the source from the
	/// [position](FailureSite::position) of the failure to `end`, e.g., a
	/// closing delimiter that nothing opened, and parse the goal again after
	/// it. The goal that the engine parses again is that of the nearest
	/// [cutting](Disposition::Cut) frame, as for [`Fix`](Repair::Fix), so the
	/// failure must lie at the start of its input, past any whitespace;
	/// otherwise, and elsewhere, this is the same as [`Stop`](Repair::Stop).
	/// If the goal is the body of a function without formal parameters, the
	/// engine parses the whole function again, since formal parameters may
	/// follow the discarded source.
	///
	/// # Panics
	/// If `end` is not a character boundary of the source.
	Skip
	{
		/// The byte offset of the end of the discarded source, which must
		/// exceed the position of the failure.
		end: usize
	},

	/// Only at [`Site::Faces`], where the faces fail at their start: supply a
	/// dice operator and standard faces, [`PLACEHOLDER_FACES`], just before
	/// the dice operator that the source supplied, and read the source again
	/// from that operator, as what follows the dice expression. The source's
	/// operator most likely begins a drop clause that follows no dice
	/// expression, e.g., `3D6 drop lowest` for `3 drop lowest`, whose `d` of
	/// `drop` the engine read as the dice operator. Elsewhere, or otherwise,
	/// this is the same as [`Stop`](Repair::Stop).
	Reread,

	/// Only at [`Site::ParameterName`], where a `,` or `:` follows the comma:
	/// retract the comma, continuing the formal parameters from the
	/// [position](FailureSite::position) of the failure as though the source
	/// lacked it, e.g., `{x}: 1` for `{x},: 1`, and `{x}, {y}: 1` for `{x}, ,
	/// {y}: 1`. Elsewhere, or otherwise, this is the same as
	/// [`Stop`](Repair::Stop).
	Retract,

	/// Only at a [`Site::Closer`] for the `}` of a variable whose name the
	/// source supplied: end the name at `end`, before the
	/// [position](FailureSite::position) of the failure, supply the `}` there,
	/// and continue the parse from there, as though the source had a `}` at
	/// `end`, e.g., `{x} + 2` for `{x + 2`. At the start of the body of a
	/// function without formal parameters, if a `,` or `:` follows `end`, the
	/// engine parses the function again instead, reading the name as the first
	/// formal parameter, e.g., `{x}, {y}: {x}` for `{x, {y}: {x}`. Elsewhere,
	/// or otherwise, this is the same as [`Stop`](Repair::Stop).
	///
	/// # Panics
	/// If `end` is not a character boundary of the source.
	Break
	{
		/// The byte offset of the end of the name, which must exceed the start
		/// of the name, and not exceed its end.
		end: usize
	},

	/// Only at [`Site::Goal`], where an expression may begin: read the word
	/// that begins at the [position](FailureSite::position) of the failure and
	/// ends at `end` as the keyword of a call of the specified direction, as
	/// though the source had the keyword there, e.g., `highest(1, 2)` for
	/// `max(1, 2)`, and parse the goal again. A `(` must follow the word, after
	/// any [whitespace between tokens](crate::parser::is_token_space).
	/// Elsewhere, or otherwise, this is the same as [`Stop`](Repair::Stop).
	///
	/// # Panics
	/// If `end` is not a character boundary of the source.
	Call
	{
		/// The byte offset of the end of the word, which must exceed the
		/// position of the failure.
		end: usize,

		/// The direction of the call.
		direction: Direction
	},

	/// Only at a [`Site::Comma`] of a [lone argument](LoneArgument): replace
	/// the call with its only argument, grouped unless it may [stand
	/// bare](LoneArgument::bare), and continue after the `)` of the call,
	/// e.g., `(1 + 2) * 3` for `highest(1 + 2) * 3`. Elsewhere, this is the
	/// same as [`Stop`](Repair::Stop).
	Unwrap,

	/// Only at a [`Site::Comma`] of a [lone argument](LoneArgument) that is a
	/// dice expression: replace the call with its only argument and a keep
	/// clause of the direction of the call, without a count, and continue
	/// after the `)` of the call, e.g., `4D6 keep highest` for `highest(4D6)`.
	/// Elsewhere, this is the same as [`Stop`](Repair::Stop).
	Keep
}

/// The kind of a [failure](FailureSite): where in the grammar the parse could
/// not continue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Site
{
	/// The goal failed recoverably, and no production can recover from the
	/// failure by trying an alternative, so the parse cannot continue. The goal
	/// is that of the innermost production still in progress, e.g., the
	/// [primary expression](Goal::Primary) that a missing operand should have
	/// begun.
	Goal
	{
		/// The goal.
		goal: Goal,

		/// The byte offset of the `[` of the range, if the failure lies in
		/// the end of a range.
		range: Option<usize>,

		/// The kind of the clause, if the failure lies in an operand of the
		/// predicate of a reroll or explode clause, a threshold or an
		/// endpoint, whose literal is signed, as that of [standard
		/// faces](Goal::Faces) is, so that a `-` just before the failure is
		/// its sign.
		operand: Option<PredicatedKind>,

		/// The byte offset of the `(` of the call, if the failure lies in an
		/// argument of a call of `highest` or `lowest`, as the innermost
		/// production that cuts.
		call: Option<usize>
	},

	/// A comma after a formal parameter is not followed by another.
	ParameterName,

	/// The formal parameters are not followed by a `:`.
	ParameterColon,

	/// The input begins with a comma, as though after formal parameters, but
	/// there are none. It has no [fix](Repair::Fix).
	LeadingComma,

	/// A closing delimiter is missing.
	Closer
	{
		/// The byte offset of the opening delimiter that the delimiter would
		/// close, or of where a repair supplied it.
		opener: usize,

		/// The missing delimiter.
		closer: char
	},

	/// The `:` that separates the start of a range from its end is missing.
	Colon
	{
		/// The byte offset of the `[` of the range.
		opener: usize
	},

	/// The `(` after the `@` of a binding is missing.
	BindingParen,

	/// The name of a variable is missing after its `{`.
	VariableName
	{
		/// The byte offset of the `{`.
		opener: usize
	},

	/// The faces of a dice expression are missing after its dice operator.
	Faces,

	/// A value of a list is missing or malformed, a face of custom faces or a
	/// value of the list of a reroll or explode clause: the first, or one
	/// after a `,`.
	FaceValue
	{
		/// The byte offset of the `[` of the list.
		opener: usize
	},

	/// The direction of a drop or keep clause is missing after its verb.
	Direction
	{
		/// Whether the verb is `keep` rather than `drop`.
		keep: bool
	},

	/// A keep, reroll, or explode clause follows a dice count without a dice
	/// operator and faces between them, e.g., `3 keep lowest` or `3 reroll =
	/// 1`, so the
	/// clause follows no dice expression. Only a
	/// [recovering](Recovery::ENABLED) parse looks for this failure, since the
	/// ordinary parse ends the expression at the dice count, and fails later,
	/// wherever the clause stands, e.g., as trailing input, or before a missing
	/// `)`. A misplaced drop clause needs no such site, since the `d` of `drop`
	/// reads as a dice operator, whose faces then fail at [`Site::Faces`]. The
	/// [position](FailureSite::position) of the failure is that of the verb,
	/// `keep`, `reroll`, or `explode`. Since the ordinary parse does not fail
	/// here,
	/// [declining](Repair::Stop) the repair does not end recovery.
	DiceOperator,

	/// The predicate of a reroll or explode clause is missing after its verb:
	/// neither a comparator nor `in` follows it.
	Predicate
	{
		/// The kind of the clause.
		kind: PredicatedKind,

		/// The threshold of the predicate, `=`, that a [fix](Repair::Fix)
		/// supplies, as [`placeholder_threshold`](
		/// PredicatedKind::placeholder_threshold) answers it.
		threshold: i32
	},

	/// The range operator, `..` or `..=`, is missing after the start of the
	/// range of a reroll or explode clause.
	RangeOperator
	{
		/// The kind of the clause.
		kind: PredicatedKind
	},

	/// The `time` or `times` is missing after the count of the limit of a
	/// reroll or explode clause, or after its `∞` or `unlimited`.
	Times
	{
		/// The kind of the clause.
		kind: PredicatedKind,

		/// Whether the count is the constant `1`, after which a repair
		/// supplies `time` rather than `times`.
		singular: bool
	},

	/// Neither a `,` nor the closing `)` follows an argument of a call of
	/// `highest` or `lowest`, where one must: a `,` after its first argument,
	/// since a call has two or more, and either after a later one. A `,` is
	/// missing after the first argument, or wherever the next token [begins an
	/// operand](crate::parser::begins_operand), as another argument would;
	/// otherwise, the failure is at a [`Site::Closer`] for the `)`.
	Comma
	{
		/// The byte offset of the `(` of the call.
		opener: usize,

		/// The call, if the `)` follows its only argument.
		lone: Option<LoneArgument>
	},

	/// A `,` after an argument of a call is followed by the `)` of the call,
	/// rather than by another argument, or a `,` after a value of a list is
	/// followed by the `]` of the list, rather than by another value. Only a
	/// [recovering](Recovery::ENABLED) parse looks for this failure, since the
	/// ordinary parse fails at the `)`, where it expects an argument, at a
	/// [goal](Site::Goal), or at the `]`, where it expects a value, at a
	/// [`Site::FaceValue`]. The
	/// [position](FailureSite::position) of the failure is that of the comma.
	/// Since the ordinary parse does not fail here,
	/// [declining](Repair::Stop) the repair does not end recovery.
	TrailingComma,

	/// A call of `highest` or `lowest` stands ungrouped where only a constant,
	/// a variable, a binding, or a group may, e.g., `3Dhighest(1, 2)` or `4D6
	/// drop lowest lowest({x}, 2)`. Only a [recovering](Recovery::ENABLED)
	/// parse looks for this failure, since the ordinary parse fails where the
	/// call stands, or, where the operand is optional, after the clause, e.g.,
	/// as trailing input. The [position](FailureSite::position) of the failure
	/// is that of the keyword of the call. Since the ordinary parse does not
	/// fail here, [declining](Repair::Stop) the repair does not end recovery.
	UngroupedCall
	{
		/// The operand that the call stands as.
		operand: Operand
	},

	/// A complete function is followed by input that is not whitespace.
	TrailingInput
}

/// The operand that a [call](Site::UngroupedCall) stands as, ungrouped, where
/// only a constant, a variable, a binding, or a group may.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Operand
{
	/// The dice count of a dice expression.
	DiceCount,

	/// The standard faces of a dice expression.
	Faces,

	/// The count of a drop clause.
	DropCount,

	/// The count of a keep clause.
	KeepCount,

	/// The threshold of the comparison of a reroll or explode clause, of the
	/// kind.
	Threshold(PredicatedKind),

	/// The start of the range of a reroll or explode clause, of the kind.
	RangeStart(PredicatedKind),

	/// The end of the range of a reroll or explode clause, of the kind.
	RangeEnd(PredicatedKind),

	/// The count of the limit of a reroll or explode clause, of the kind.
	Limit(PredicatedKind)
}

impl Operand
{
	/// Answer the operand that an atom stands as, by the frame that awaits it.
	///
	/// # Parameters
	/// - `frame`: The frame on top of the stack, which awaits the atom.
	///
	/// # Returns
	/// The operand.
	fn of(frame: Option<&Frame<'_>>) -> Self
	{
		match frame
		{
			Some(Frame::DiceFaces { .. } | Frame::StandardFaces(_)) =>
			{
				Operand::Faces
			},
			Some(
				Frame::DiceDrop { kind, .. }
				| Frame::DropExpression { kind, .. }
			) if kind.keeps() => Operand::KeepCount,
			Some(Frame::DiceDrop { .. } | Frame::DropExpression { .. }) =>
			{
				Operand::DropCount
			},
			Some(Frame::PredicateFirst {
				kind,
				comparison: Some(_),
				..
			}) => Operand::Threshold(*kind),
			Some(Frame::PredicateFirst { kind, .. }) =>
			{
				Operand::RangeStart(*kind)
			},
			Some(Frame::PredicateEnd { kind, .. }) => Operand::RangeEnd(*kind),
			Some(Frame::LimitCount { kind, .. }) => Operand::Limit(*kind),
			_ => Operand::DiceCount
		}
	}
}

/// A call of `highest` or `lowest` whose `)` follows its only argument, which
/// a [`Site::Comma`] presents.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoneArgument
{
	/// The byte offset of the start of the call, at its keyword.
	pub(crate) call: usize,

	/// The span of the argument, as written, from its first token to its
	/// last.
	pub(crate) argument: SourceSpan,

	/// The direction of the call.
	pub(crate) direction: Direction,

	/// Whether the argument is a dice expression, which a
	/// [keep](Repair::Keep) may replace the call with.
	pub(crate) dice: bool,

	/// Whether the argument may stand bare in place of the call, as a primary
	/// expression whose meaning no neighbor can change, rather than grouped.
	pub(crate) bare: bool
}

impl Site
{
	/// Answer whether the site has a [fix](Repair::Fix).
	///
	/// # Returns
	/// `true` if the site has a fix, `false` otherwise.
	fn is_fixable(self) -> bool { !matches!(self, Site::LeadingComma) }
}

/// A failure of the parse, as presented to a [policy](Recovery).
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FailureSite<'src>
{
	/// The kind of failure.
	pub(crate) site: Site,

	/// The leading entries of the error that the parse reports if the policy
	/// declines to repair the failure: those at the position of the failure,
	/// from which [expectations](NomErrorKind::expectations) are drawn. It is
	/// never empty.
	pub(crate) error: ParseError<'src>
}

impl<'src> FailureSite<'src>
{
	/// Answer the position of the failure.
	///
	/// # Returns
	/// The byte offset of the failure.
	pub(crate) fn position(&self) -> usize
	{
		self.error.errors[0].0.location_offset()
	}

	/// Answer the input at the position of the failure.
	///
	/// # Returns
	/// The remaining input.
	pub(crate) fn input(&self) -> Span<'src> { self.error.errors[0].0 }
}

/// How a [frame](Frame) treats a recoverable error of the subproduction that it
/// awaits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Disposition
{
	/// It recovers by trying another alternative, or by treating the
	/// subproduction as optional.
	Catch,

	/// It makes the error unrecoverable, as [`cut`] does.
	Cut,

	/// It passes the error on, perhaps with a context attached, or merged with
	/// the errors of earlier alternatives when the subproduction is the last
	/// alternative.
	Pass
}

impl Disposition
{
	/// Answer the disposition of the specified frame.
	///
	/// # Parameters
	/// - `frame`: The frame.
	///
	/// # Returns
	/// The disposition.
	fn of(frame: &Frame<'_>) -> Self
	{
		match frame
		{
			Frame::UnaryNegation(_)
			| Frame::PrimaryRange(_)
			| Frame::PrimaryDice(_)
			| Frame::PrimaryGroup(_)
			| Frame::DiceFaces { .. }
			| Frame::DiceDrop { .. }
			| Frame::DropExpression { .. }
			| Frame::LimitCount { .. } => Disposition::Catch,
			Frame::BinaryRight { .. }
			| Frame::ExponentPower(_)
			| Frame::PredicateFirst { .. }
			| Frame::PredicateEnd { .. }
			| Frame::CallArgument { .. }
			| Frame::GroupExpression { .. }
			| Frame::BindingExpression { .. }
			| Frame::RangeStart { .. }
			| Frame::RangeEnd { .. } => Disposition::Cut,
			Frame::FunctionBody { .. }
			| Frame::BinaryFirst(_)
			| Frame::UnaryExponent { .. }
			| Frame::ExponentBase
			| Frame::PrimaryCall(_)
			| Frame::AtomGroup { .. }
			| Frame::AtomCall(_)
			| Frame::DiceCount { .. }
			| Frame::StandardCount(_)
			| Frame::StandardFaces(_)
			| Frame::CustomCount(_) => Disposition::Pass
		}
	}
}

/// A token that a [repair](Repair) supplies at a position of the source, which
/// the engine reads in place of the source text there. Only the leaves that
/// would read the token, were it in the source, look for it. In the source,
/// `-0` is a [negated constant](negative_constant), or faces that are an
/// [`integer`], so either also begins at a `-` just before the token.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Overlay<'src>
{
	/// A constant, inserted at a position.
	Constant
	{
		/// The byte offset of the position.
		at: usize,

		/// The value of the constant.
		value: i32
	},

	/// Standard faces, [`PLACEHOLDER_FACES`], inserted at a position, after
	/// which the engine [reads the source again](Repair::Reread) from the
	/// dice operator before the faces.
	Reread
	{
		/// The byte offset of the position.
		at: usize,

		/// The input at the dice operator.
		operator: Span<'src>
	},

	/// A variable, or the first formal parameter of a function that the
	/// engine parses again, whose name is the identifier that spans a range of
	/// the source, as though the source enclosed it in braces.
	Variable
	{
		/// The byte offset of the start of the name.
		at: usize,

		/// The byte offset of the end of the name.
		end: usize
	},

	/// The keyword of a call, `highest` or `lowest`, in place of the word that
	/// spans a range of the source, as a [call](Repair::Call) supplies it.
	Call
	{
		/// The byte offset of the start of the word.
		at: usize,

		/// The byte offset of the end of the word.
		end: usize,

		/// The direction of the call.
		direction: Direction
	},

	/// A braced name whose `}` a [break](Repair::Break) supplied, as the first
	/// formal parameter of a function that the engine parses again. The policy
	/// was already consulted about the `}`, so the engine reads the `}` where
	/// the break supplied it, rather than consult the policy again.
	Brace
	{
		/// The byte offset of the `{`.
		opener: usize,

		/// The byte offset of the end of the name, where the `}` is supplied.
		end: usize
	}
}

/// The state of recovery during a run of the engine.
///
/// A failure is detected where it becomes certain, while the stack still holds
/// every production in progress. An unrecoverable error is certain where it is
/// created, at one of the sites that [`repair`](Recoverer::repair) guards. A
/// recoverable error is certain once no frame between the top of the stack and
/// the nearest frame that [cuts](Disposition::Cut) can
/// [catch](Disposition::Catch) it. The recoverer keeps a stack of the frames
/// that catch or cut, alongside the stack of frames, so that it can tell in
/// constant time.
///
/// The recoverer also holds the [sink](TokenSink) of the run, to which the
/// leaves of the grammar report the tokens that they read. For [`Strict`] and
/// [`Silent`], the policy and the sink of the ordinary parse, every hook of
/// recovery and every report compiles away.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being parsed.
/// - `'p`: The lifetime of the policy and of the sink.
/// - `R`: The type of the policy.
/// - `S`: The type of the sink.
struct Recoverer<'src, 'p, R, S>
{
	/// The policy.
	policy: &'p mut R,

	/// The sink of the tokens that the engine reads.
	sink: &'p mut S,

	/// Whether the policy is still consulted: not after it declines a repair.
	active: bool,

	/// The input of the run, where the parser attaches the outermost context.
	root: Span<'src>,

	/// The token that a repair supplies, until the engine reads it.
	overlay: Option<Overlay<'src>>,

	/// The input at the last dice operator that the engine read while
	/// recovering, which a [reread](Repair::Reread) reads again.
	operator: Option<Span<'src>>,

	/// For each frame on the stack, the goal that it awaits and the input of
	/// the goal.
	calls: Vec<(Goal, Span<'src>)>,

	/// The indices, in the stack, of the frames that catch or cut, from the
	/// bottom up.
	barriers: Vec<usize>,

	/// The input from which to parse the function again, if a repair of the
	/// name at the start of its body made the name its first formal parameter.
	/// The engine restarts before its next step.
	restart: Option<Span<'src>>,

	/// The byte offsets of the `(` of the call and of the last `,` that a
	/// repair supplied between the arguments of a call. If the argument after
	/// it fails without reading anything, e.g., at a misplaced clause that a
	/// repair read past without consuming, a `,` is missing there again, and
	/// supplying another would never end, so the engine stops recovering
	/// instead.
	comma: Option<(usize, usize)>
}

impl<'src, 'p, R: Recovery<'src>, S: TokenSink> Recoverer<'src, 'p, R, S>
{
	/// Create the state of recovery for a run.
	///
	/// # Parameters
	/// - `policy`: The policy.
	/// - `sink`: The sink of the tokens that the engine reads.
	/// - `root`: The input of the run.
	///
	/// # Returns
	/// The state.
	fn new(policy: &'p mut R, sink: &'p mut S, root: Span<'src>) -> Self
	{
		Self {
			policy,
			sink,
			active: R::ENABLED,
			root,
			overlay: None,
			operator: None,
			calls: Vec::new(),
			barriers: Vec::new(),
			restart: None,
			comma: None
		}
	}

	/// Answer whether the engine is recovering, i.e., still consulting the
	/// policy.
	///
	/// # Returns
	/// `true` if the engine is recovering, `false` otherwise.
	#[inline(always)]
	fn recovering(&self) -> bool { R::ENABLED && self.active }

	/// Report a token to the sink, unless the sink is [`Silent`].
	///
	/// # Parameters
	/// - `start`: The byte offset of the start of the token.
	/// - `end`: The byte offset of the end of the token.
	/// - `role`: The role of the token.
	#[inline(always)]
	fn report(&mut self, start: usize, end: usize, role: Role)
	{
		if S::ENABLED
		{
			self.sink.token(SourceSpan { start, end }, role);
		}
	}

	/// Report the token that a leaf just read to the sink, unless the sink is
	/// [`Silent`]: the source from the end of any whitespace at one input to
	/// another, as a leaf preceded by `multispace0` reads it.
	///
	/// # Parameters
	/// - `before`: The input of the leaf, perhaps at whitespace.
	/// - `after`: The input after the token.
	/// - `role`: The role of the token.
	#[inline(always)]
	fn report_read(&mut self, before: Span<'src>, after: Span<'src>, role: Role)
	{
		if S::ENABLED
		{
			let start = skip_whitespace(before).location_offset();
			self.report(start, after.location_offset(), role);
		}
	}

	/// Report the delimiter that ends just before an input to the sink, unless
	/// the sink is [`Silent`].
	///
	/// # Parameters
	/// - `after`: The input after the delimiter, a single byte.
	/// - `role`: The role of the delimiter.
	#[inline(always)]
	fn report_before(&mut self, after: Span<'src>, role: Role)
	{
		if S::ENABLED
		{
			let end = after.location_offset();
			self.report(end - 1, end, role);
		}
	}

	/// Report a braced name to the sink, unless the sink is [`Silent`]: its
	/// braces, and its identifier, exactly the span of the name, so that any
	/// whitespace that pads the identifier within the braces belongs to no
	/// token. A repair may have supplied a part of the name, which is not in
	/// the source, and so is not reported: the identifier, whose span is then
	/// empty, or the `}`. A name that a repair read without its braces, as
	/// though the source enclosed it in them, is not reported at all, since
	/// the source has no braced name there.
	///
	/// # Parameters
	/// - `braced`: The span of the braced name, from its `{` through its `}`,
	///   or through the end of the identifier if the `}` was supplied; or just
	///   the identifier, if a repair read it without its braces.
	/// - `name`: The span of the identifier.
	/// - `role`: The role of the identifier.
	#[inline(always)]
	fn report_name(&mut self, braced: SourceSpan, name: SourceSpan, role: Role)
	{
		if !S::ENABLED || self.byte_at(braced.start) != Some(b'{')
		{
			return;
		}
		self.report(braced.start, braced.start + 1, Role::NameBrace);
		if name.start < name.end
		{
			self.report(name.start, name.end, role);
		}
		let closer = braced.end - 1;
		if closer > braced.start
			&& closer >= name.end
			&& self.byte_at(closer) == Some(b'}')
		{
			self.report(closer, braced.end, Role::NameBrace);
		}
	}

	/// Answer the byte of the source at an offset.
	///
	/// # Parameters
	/// - `offset`: The byte offset.
	///
	/// # Returns
	/// The byte, or `None` if the offset lies outside of the input of the run.
	fn byte_at(&self, offset: usize) -> Option<u8>
	{
		let index = offset.checked_sub(self.root.location_offset())?;
		self.root.fragment().as_bytes().get(index).copied()
	}

	/// Report a bracketed list of literals to the sink, unless the sink is
	/// [`Silent`]: the custom faces of dice, or the list of a predicate, which
	/// a flat combinator has just read, and which the engine scans again only
	/// to report its tokens, so that no path of the parse changes. The brackets
	/// take the specified role, each `,` is a [separator](Role::Separator), and
	/// each literal, with its sign, a [face](Role::Face).
	///
	/// # Parameters
	/// - `input`: The input of the list, perhaps at whitespace before its `[`.
	/// - `rest`: The input after the list.
	/// - `bracket`: The role of the brackets.
	fn report_literals(
		&mut self,
		input: Span<'src>,
		rest: Span<'src>,
		bracket: Role
	)
	{
		if !S::ENABLED
		{
			return;
		}
		let base = input.location_offset();
		// The list is ASCII, so every byte is a character.
		let list =
			&input.fragment().as_bytes()[..rest.location_offset() - base];
		let mut at = 0;
		while at < list.len()
		{
			let start = at;
			at += 1;
			let role = match list[start]
			{
				b'[' | b']' => bracket,
				b',' => Role::Separator,
				b'-' | b'0'..=b'9' =>
				{
					while list.get(at).is_some_and(u8::is_ascii_digit)
					{
						at += 1;
					}
					Role::Face
				},
				_ => continue
			};
			self.report(base + start, base + at, role);
		}
	}

	/// Retract every token that begins at or after a byte offset from the
	/// sink, unless the sink is [`Silent`].
	///
	/// # Parameters
	/// - `from`: The byte offset.
	#[inline(always)]
	fn retract(&mut self, from: usize)
	{
		if S::ENABLED
		{
			self.sink.retract(from);
		}
	}

	/// Record that a goal is called. Unless the stack is empty, its outcome
	/// returns to the frame on top of the stack, which is either new or, if the
	/// engine is parsing a goal again, the frame that awaited it before.
	///
	/// # Parameters
	/// - `goal`: The goal.
	/// - `input`: The input of the goal.
	/// - `stack`: The stack of suspended productions.
	#[inline(always)]
	fn called(&mut self, goal: Goal, input: Span<'src>, stack: &Stack<'src>)
	{
		if !self.recovering() || stack.is_empty()
		{
			return;
		}
		let index = stack.len() - 1;
		if self.calls.len() > index
		{
			self.calls.truncate(index);
		}
		else if Disposition::of(&stack[index]) != Disposition::Pass
		{
			self.barriers.push(index);
		}
		self.calls.push((goal, input));
	}

	/// Record that the frame at the specified index of the stack was popped.
	///
	/// # Parameters
	/// - `index`: The index of the popped frame.
	#[inline(always)]
	fn popped(&mut self, index: usize)
	{
		if !self.recovering()
		{
			return;
		}
		self.calls.truncate(index);
		if self.barriers.last() == Some(&index)
		{
			self.barriers.pop();
		}
	}

	/// Answer whether an outcome is a certain failure, i.e., a recoverable
	/// error that no frame can catch.
	///
	/// # Parameters
	/// - `outcome`: The outcome, about to return to the top of the stack.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// `true` if the outcome is a certain failure, `false` otherwise.
	#[inline(always)]
	fn intercepts(&self, outcome: &Outcome<'src>, stack: &Stack<'src>) -> bool
	{
		self.recovering()
			&& matches!(outcome, Err(nom::Err::Error(_)))
			&& self.barriers.last().is_none_or(|&index| {
				Disposition::of(&stack[index]) == Disposition::Cut
			})
	}

	/// Consult the policy about a [goal](Site::Goal) that failed with
	/// certainty, as detected by [`intercepts`](Self::intercepts).
	///
	/// The failure need not lie within the goal on top of the stack: an
	/// alternative that reached further may already have failed, and its
	/// frames been popped, e.g., the negation in `- x`, whose operand fails at
	/// `x`, after which the unary expression tries an exponentiation at `-`. So
	/// to repair the failure, the engine parses again from the goal that the
	/// nearest [cutting](Disposition::Cut) frame awaits, or the bottom frame if
	/// none cuts, with the repair's token [overlaid](Overlay) at the position
	/// of the failure. Every frame above that one awaits the start of its own
	/// goal, so this parses again only the input of the failed attempt. A
	/// [skip](Repair::Skip) instead parses the goal again after the discarded
	/// source, which is possible only if the failure lies at the start of the
	/// goal. A [variable](Repair::Variable) at the start of the body of a
	/// function without formal parameters, followed by a `,` or `:`, is the
	/// first formal parameter of the edited source, so the engine instead
	/// parses the function again, from the variable. Formal parameters may
	/// likewise follow source skipped at the start of the body of a function
	/// without them, so the engine parses that function again too, after the
	/// discarded source.
	///
	/// # Parameters
	/// - `outcome`: The error of the goal.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// The next step: a call to parse the goal again, or, if the policy
	/// declines, the return of the error.
	fn recover_goal(
		&mut self,
		outcome: Outcome<'src>,
		stack: &mut Stack<'src>
	) -> Step<'src>
	{
		let Err(e) = &outcome
		else
		{
			unreachable!("only errors are intercepted")
		};
		let Some(&(goal, _)) = self.calls.last()
		else
		{
			return self.stop(outcome);
		};
		if self.overlay.is_some()
		{
			return self.stop(outcome);
		}
		// Every frame above the nearest barrier passes errors on, so the
		// nearest barrier is the frame that cuts, if any.
		let anchor = self.barriers.last().copied().unwrap_or(0);
		let range = match &stack[anchor]
		{
			Frame::RangeEnd { start, .. } => Some(*start),
			_ => None
		};
		let operand = match &stack[anchor]
		{
			Frame::PredicateFirst { kind, .. }
			| Frame::PredicateEnd { kind, .. } => Some(*kind),
			_ => None
		};
		let call = match &stack[anchor]
		{
			Frame::CallArgument { opener, .. } => Some(*opener),
			_ => None
		};
		let site = FailureSite {
			site: Site::Goal {
				goal,
				range,
				operand,
				call
			},
			error: settle(e.clone(), stack, self.root)
		};
		let at = site.position();
		let (goal, input) = self.calls[anchor];
		match self.policy.repair(&site)
		{
			Repair::Fix =>
			{
				self.overlay = Some(Overlay::Constant {
					at,
					value: PLACEHOLDER_OPERAND
				});
			},
			Repair::Call { end, direction } if end > at =>
			{
				self.overlay = Some(Overlay::Call { at, end, direction });
			},
			Repair::Variable { end } if end > at =>
			{
				self.overlay = Some(Overlay::Variable { at, end });
				if starts_body(&stack[anchor], at)
					&& precedes_parameters(input.take_from(end - at))
				{
					return self.restart_function(input, stack);
				}
			},
			Repair::Skip { end }
				if end > at
					&& skip_whitespace(input).location_offset() == at =>
			{
				// The discarded source begins the goal, so it is the goal's
				// failed attempt that is retracted.
				self.retract(at);
				let input =
					skip_whitespace(skip_whitespace(input).take_from(end - at));
				// Formal parameters may follow the discarded source at the
				// start of a function without them, so parse the function
				// again.
				if let Frame::FunctionBody { formal: false, .. } = stack[anchor]
				{
					return self.restart_function(input, stack);
				}
				stack.truncate(anchor + 1);
				self.calls.truncate(anchor + 1);
				return Step::Call(goal, input);
			},
			_ => return self.stop(outcome)
		}
		// The engine parses the goal again, so the tokens of the failed
		// attempt are read again.
		self.retract(input.location_offset());
		stack.truncate(anchor + 1);
		self.calls.truncate(anchor + 1);
		Step::Call(goal, input)
	}

	/// Parse the function again from the specified input, discarding every
	/// production in progress.
	///
	/// # Parameters
	/// - `input`: The input of the function.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// The next step: a call to parse the function.
	fn restart_function(
		&mut self,
		input: Span<'src>,
		stack: &mut Stack<'src>
	) -> Step<'src>
	{
		self.retract(input.location_offset());
		stack.clear();
		self.calls.clear();
		self.barriers.clear();
		Step::Call(Goal::Function, input)
	}

	/// Consult the policy about an unrecoverable error, just created at the
	/// specified site, unless the engine is not recovering.
	///
	/// # Parameters
	/// - `site`: The site.
	/// - `e`: The error.
	/// - `stack`: The stack of suspended productions, without the frame of the
	///   production that created the error.
	///
	/// # Returns
	/// The input at the position of the failure, where the parse continues as
	/// though the token that the site lacks were present.
	///
	/// # Errors
	/// * `e`, if the engine is not recovering or the policy declines to repair
	///   the failure.
	#[inline(always)]
	fn repair(
		&mut self,
		site: Site,
		e: nom::Err<ParseError<'src>>,
		stack: &Stack<'src>
	) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
	{
		if !self.recovering()
		{
			return Err(e);
		}
		self.consult(site, e, stack)
	}

	/// Offer the policy a repair at a site that is not a failure of the
	/// ordinary parse, [`Site::DiceOperator`], unless the engine is not
	/// recovering, or a repair's token is still to be read. Unlike at a
	/// failure, declining the repair does not stop recovery, since the parse
	/// then continues as the ordinary parse does, and fails, if at all, later.
	///
	/// # Parameters
	/// - `site`: The site.
	/// - `e`: The error that the site presents.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// `true` if the policy [fixes](Repair::Fix) the site, `false` otherwise.
	fn offer(
		&mut self,
		site: Site,
		e: nom::Err<ParseError<'src>>,
		stack: &Stack<'src>
	) -> bool
	{
		if !self.recovering() || self.overlay.is_some()
		{
			return false;
		}
		let site = FailureSite {
			site,
			error: settle(e, stack, self.root)
		};
		self.policy.repair(&site) == Repair::Fix
	}

	/// Consult the policy on behalf of [`repair`](Self::repair).
	///
	/// # Parameters
	/// - `site`: The site.
	/// - `e`: The error.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// The input at the position of the failure.
	///
	/// # Errors
	/// * `e`, if the policy declines to repair the failure.
	fn consult(
		&mut self,
		site: Site,
		e: nom::Err<ParseError<'src>>,
		stack: &Stack<'src>
	) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
	{
		self.consult_with(site, e, stack, |repair, input| {
			(repair == Repair::Fix && site.is_fixable()).then_some(input)
		})
	}

	/// Consult the policy about an unrecoverable error, just created at the
	/// specified site, and accept the repairs that the site admits.
	///
	/// # Type parameters
	/// - `T`: The type of an accepted repair.
	///
	/// # Parameters
	/// - `site`: The site.
	/// - `e`: The error.
	/// - `stack`: The stack of suspended productions.
	/// - `accept`: Given the policy's repair and the input at the position of
	///   the failure, answer the accepted repair, or `None` to stop recovering.
	///
	/// # Returns
	/// The accepted repair.
	///
	/// # Errors
	/// * `e`, if the policy declines to repair the failure, or `accept` rejects
	///   the repair.
	fn consult_with<T>(
		&mut self,
		site: Site,
		e: nom::Err<ParseError<'src>>,
		stack: &Stack<'src>,
		accept: impl FnOnce(Repair, Span<'src>) -> Option<T>
	) -> Result<T, nom::Err<ParseError<'src>>>
	{
		if self.overlay.is_some()
		{
			self.active = false;
			return Err(e);
		}
		let site = FailureSite {
			site,
			error: settle(e.clone(), stack, self.root)
		};
		match accept(self.policy.repair(&site), site.input())
		{
			Some(accepted) => Ok(accepted),
			None =>
			{
				self.active = false;
				Err(e)
			}
		}
	}

	/// Consult the policy about missing faces, unless the engine is not
	/// recovering.
	///
	/// # Parameters
	/// - `e`: The error.
	/// - `input`: The input of the faces.
	/// - `stack`: The stack of suspended productions.
	///
	/// # Returns
	/// The token to overlay at the position of the failure: the placeholder
	/// faces, a [variable](Repair::Variable), which must begin the faces, or
	/// the placeholder faces of a [reread](Repair::Reread), which must also
	/// begin them.
	///
	/// # Errors
	/// * `e`, if the engine is not recovering or the policy declines to repair
	///   the failure.
	fn repair_faces(
		&mut self,
		e: nom::Err<ParseError<'src>>,
		input: Span<'src>,
		stack: &Stack<'src>
	) -> Result<Overlay<'src>, nom::Err<ParseError<'src>>>
	{
		if !self.recovering()
		{
			return Err(e);
		}
		let faces = skip_whitespace(input).location_offset();
		// The faces fail at their start, so the last dice operator that the
		// engine read should be the one before them. Check, rather than trust,
		// that it is.
		let operator = self.operator.filter(|operator| {
			skip_whitespace(operator.take_from(1)).location_offset() == faces
		});
		self.consult_with(Site::Faces, e, stack, |repair, input| {
			let at = input.location_offset();
			match repair
			{
				Repair::Fix => Some(Overlay::Constant {
					at,
					value: PLACEHOLDER_FACES
				}),
				Repair::Variable { end } if at == faces && end > at =>
				{
					Some(Overlay::Variable { at, end })
				},
				Repair::Reread if at == faces =>
				{
					operator.map(|operator| Overlay::Reread { at, operator })
				},
				_ => None
			}
		})
	}

	/// Stop recovering, and return an error.
	///
	/// # Parameters
	/// - `outcome`: The error.
	///
	/// # Returns
	/// The next step.
	fn stop(&mut self, outcome: Outcome<'src>) -> Step<'src>
	{
		self.active = false;
		Step::Return(outcome)
	}

	/// Answer whether the [overlaid](Overlay) token is a formal parameter that
	/// begins at the specified input: a [variable](Overlay::Variable) or a
	/// [braced name](Overlay::Brace).
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// `true` if the overlay begins a formal parameter at `input`, `false`
	/// otherwise.
	fn overlays_parameter(&self, input: Span<'src>) -> bool
	{
		let start = input.location_offset();
		match self.overlay
		{
			Some(Overlay::Variable { at, .. }) => at == start,
			Some(Overlay::Brace { opener, .. }) => opener == start,
			_ => false
		}
	}

	/// Read the [overlaid](Overlay) formal parameter, if it begins at the
	/// specified input: a [variable](Overlay::Variable), as though the source
	/// enclosed it in braces, or a [braced name](Overlay::Brace), as though the
	/// source closed it.
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// The input after the parameter, and the parameter, if the overlay begins
	/// a formal parameter at `input`.
	fn take_parameter(
		&mut self,
		input: Span<'src>
	) -> Option<(Span<'src>, Parameter<'src>)>
	{
		let start = input.location_offset();
		// The name ends before any whitespace before the break, but the input
		// after it begins at the break.
		let (name, name_end, end) = match self.overlay?
		{
			Overlay::Variable { at, end } if at == start => (start, end, end),
			Overlay::Brace { opener, end } if opener == start =>
			{
				// Skip the `{`, and the whitespace around the name.
				let name = input.fragment()[1..end - start].trim_start();
				let name_start = end - name.len();
				(name_start, name_start + name.trim_end().len(), end)
			},
			_ => return None
		};
		self.overlay = None;
		Some((
			input.take_from(end - start),
			Parameter {
				name: canonical_name(
					&input.fragment()[name - start..name_end - start]
				),
				span: SourceSpan {
					start: name,
					end: name_end
				}
			}
		))
	}

	/// Answer whether the [overlaid](Overlay::Call) keyword of a call begins at
	/// the specified input.
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// `true` if the overlay is the keyword of a call that begins at `input`,
	/// `false` otherwise.
	fn overlays_call(&self, input: Span<'src>) -> bool
	{
		matches!(
			self.overlay,
			Some(Overlay::Call { at, .. }) if at == input.location_offset()
		)
	}

	/// Read the [overlaid](Overlay::Call) keyword of a call, if it begins at
	/// the specified input, and a `(` follows the word that it replaces, after
	/// any whitespace. Otherwise, the overlay remains unread, so that the
	/// failure of the goal stops recovery.
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// The input after the word that the keyword replaces, and the direction
	/// of the call, if the overlay begins at `input`, and a `(` follows it.
	fn take_call(
		&mut self,
		input: Span<'src>
	) -> Option<(Span<'src>, Direction)>
	{
		let Some(Overlay::Call { at, end, direction }) = self.overlay
		else
		{
			return None;
		};
		let after = input.take_from(end - at);
		if at != input.location_offset()
			|| !skip_whitespace(after).fragment().starts_with('(')
		{
			return None;
		}
		self.overlay = None;
		Some((after, direction))
	}

	/// Read the [overlaid](Overlay::Variable) variable, if it begins at the
	/// specified input.
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// The input after the name, and the variable, if the overlay begins at
	/// `input`.
	fn take_variable(
		&mut self,
		input: Span<'src>
	) -> Option<(Span<'src>, Variable<'src>)>
	{
		let start = input.location_offset();
		let Some(Overlay::Variable { at, end }) = self.overlay
		else
		{
			return None;
		};
		if at != start
		{
			return None;
		}
		self.overlay = None;
		let length = end - start;
		Some((
			input.take_from(length),
			Variable {
				name: canonical_name(&input.fragment()[..length]),
				span: SourceSpan { start, end }
			}
		))
	}

	/// Read the [overlaid](Overlay::Constant) constant as [`constant`] would,
	/// if it begins at the specified input, or as [`integer`] would, if it
	/// begins there or just after a `-` there; or read the placeholder faces of
	/// a [reread](Overlay::Reread), if they begin at the specified input.
	///
	/// # Parameters
	/// - `input`: The input.
	/// - `signed`: Whether to read the constant as [`integer`] would.
	///
	/// # Returns
	/// The input after the constant, and the constant, if the overlay begins
	/// at `input`, or just after a `-` there if `signed`. After the faces of a
	/// reread, the input is that at its dice operator.
	fn take_constant(
		&mut self,
		input: Span<'src>,
		signed: bool
	) -> Option<(Span<'src>, Constant)>
	{
		let start = input.location_offset();
		let (rest, at, value) = match self.overlay?
		{
			Overlay::Constant { at, value } if at == start =>
			{
				(input, at, value)
			},
			Overlay::Constant { at, value }
				if signed
					&& at == start + 1
					&& input.fragment().starts_with('-') =>
			{
				// The sign is in the source, and begins the literal.
				self.report(start, at, Role::Face);
				(input.take_from(1), at, -value)
			},
			Overlay::Reread { at, operator } if at == start =>
			{
				// The engine reads the source again from the dice operator,
				// which it already reported as such.
				self.retract(operator.location_offset());
				(operator, at, PLACEHOLDER_FACES)
			},
			_ => return None
		};
		self.overlay = None;
		Some((
			rest,
			Constant {
				value,
				span: SourceSpan { start, end: at }
			}
		))
	}

	/// Read the [overlaid](Overlay::Constant) constant as [`negative_constant`]
	/// would, if it follows a `-` and any whitespace at the specified input.
	///
	/// # Parameters
	/// - `input`: The input.
	///
	/// # Returns
	/// The input after the constant, and the constant, if the overlay follows
	/// a `-` at `input`, and no dice operator or `^` follows the overlay.
	fn take_negative_constant(
		&mut self,
		input: Span<'src>
	) -> Option<(Span<'src>, Expression<'src>)>
	{
		let Some(Overlay::Constant { at, value }) = self.overlay
		else
		{
			return None;
		};
		let sign: IResult<Span, char, ParseError> = char('-')(input);
		let rest = skip_whitespace(sign.ok()?.0);
		let after = rest.fragment().trim_start_matches(is_token_space);
		if rest.location_offset() != at
			|| after.starts_with(['d', 'D', '^'])
			|| begins_misplaced_clause(after).is_some()
		{
			return None;
		}
		self.overlay = None;
		// The sign is in the source, and is negation.
		let start = input.location_offset();
		self.report(start, start + 1, Role::Negation);
		Some((
			rest,
			Expression::Constant(Constant {
				value: -value,
				span: SourceSpan {
					start: input.location_offset(),
					end: at
				}
			})
		))
	}
}

/// Compute the leading entries of the error that the parse reports for a
/// failure, by unwinding the error through the frames on the stack, as the
/// ordinary parse would, but only as far as they can attach entries at the
/// position of the failure. Every cycle of the grammar consumes a token, so
/// only a few frames can await input at the same position, and the unwinding
/// takes constant time, whatever the depth of the stack.
///
/// # Parameters
/// - `e`: The error.
/// - `stack`: The stack of suspended productions.
/// - `root`: The input of the run, where the parser attaches the outermost
///   context.
///
/// # Returns
/// The leading entries of the error, i.e., those at its position.
fn settle<'src>(
	mut e: nom::Err<ParseError<'src>>,
	stack: &Stack<'src>,
	root: Span<'src>
) -> ParseError<'src>
{
	for (frame, payload) in stack.iter_rev()
	{
		// A frame can only attach entries to an unrecoverable error, so once an
		// entry is not at the position, no later entry is leading.
		if let nom::Err::Failure(error) = &e
			&& leading(error) < error.errors.len()
		{
			break;
		}
		e = unwind(frame, payload, e);
	}
	let mut error = match with_context(root, FUNCTION_CONTEXT, e)
	{
		nom::Err::Error(e) | nom::Err::Failure(e) => e,
		nom::Err::Incomplete(_) => unreachable!("parsing is complete")
	};
	error.errors.truncate(leading(&error));
	error
}

/// Answer the number of leading entries of an error, i.e., those at its
/// position.
///
/// # Parameters
/// - `error`: The error.
///
/// # Returns
/// The number of leading entries.
fn leading(error: &ParseError<'_>) -> usize
{
	let position = error.errors[0].0.location_offset();
	error
		.errors
		.iter()
		.take_while(|(span, _)| span.location_offset() == position)
		.count()
}

/// Unwind an error through a frame, as the frame's production would when it
/// resumed with the error, but without trying another alternative.
///
/// # Parameters
/// - `frame`: The frame. If the error is recoverable, it must not
///   [catch](Disposition::Catch) it.
/// - `payload`: The payload of the frame, if it [carries](Frame::carries) one.
/// - `e`: The error.
///
/// # Returns
/// The error, as the frame's production would finish with it.
///
/// # Panics
/// If the frame carries a payload of the wrong kind.
fn unwind<'src>(
	frame: &Frame<'src>,
	payload: Option<&Payload<'src>>,
	e: nom::Err<ParseError<'src>>
) -> nom::Err<ParseError<'src>>
{
	debug_assert!(
		matches!(e, nom::Err::Failure(_))
			|| Disposition::of(frame) != Disposition::Catch,
		"a frame that catches the error cannot unwind it"
	);
	match frame
	{
		Frame::FunctionBody { input, .. } =>
		{
			with_context(*input, FUNCTION_BODY_CONTEXT, e)
		},
		Frame::BinaryFirst(_)
		| Frame::UnaryNegation(_)
		| Frame::ExponentBase
		| Frame::PrimaryCall(_)
		| Frame::AtomCall(_) => e,
		Frame::BinaryRight { input, .. } | Frame::ExponentPower(input) =>
		{
			cut_error(with_context(*input, RIGHT_OPERAND_CONTEXT, e))
		},
		Frame::UnaryExponent { input, .. } => match e
		{
			nom::Err::Error(e) =>
			{
				let error = match payload
				{
					Some(Payload::Error(error)) => error.clone(),
					None => unary_error(*input),
					Some(_) => unreachable!("the payload is not an error")
				};
				exhausted(*input, merge(error, e))
			},
			e => e
		},
		Frame::PrimaryRange(input) => with_context(*input, RANGE_CONTEXT, e),
		Frame::PrimaryDice(input) => with_context(*input, DICE_CONTEXT, e),
		Frame::PrimaryGroup(input) => with_context(*input, GROUP_CONTEXT, e),
		Frame::AtomGroup { input, signed } =>
		{
			match with_context(*input, GROUP_CONTEXT, e)
			{
				nom::Err::Error(e) =>
				{
					exhausted(*input, merge(atom_error(*input, *signed), e))
				},
				e => e
			}
		},
		Frame::GroupExpression { input, .. } =>
		{
			cut_error(with_context(*input, EXPRESSION_CONTEXT, e))
		},
		Frame::BindingExpression { input, .. } =>
		{
			let Some(Payload::Head(head)) = payload
			else
			{
				unreachable!("the payload is not the head of a binding")
			};
			within_binding(
				head.atom,
				cut_error(with_context(*input, BINDING_EXPRESSION_CONTEXT, e))
			)
		},
		Frame::RangeStart { input, .. } =>
		{
			cut_error(with_context(*input, RANGE_START_CONTEXT, e))
		},
		Frame::RangeEnd { input, .. } =>
		{
			cut_error(with_context(*input, RANGE_END_CONTEXT, e))
		},
		Frame::DiceCount { input, .. }
		| Frame::StandardCount(input)
		| Frame::CustomCount(input) => with_context(*input, DICE_COUNT_CONTEXT, e),
		Frame::DiceFaces { input, .. } | Frame::StandardFaces(input) =>
		{
			with_context(*input, STANDARD_FACES_CONTEXT, e)
		},
		Frame::DiceDrop { kind, input }
		| Frame::DropExpression { kind, input } =>
		{
			with_context(*input, kind.count_context(), e)
		},
		Frame::PredicateFirst {
			kind,
			comparison,
			input
		} => cut_error(with_context(
			*input,
			first_context(*kind, *comparison),
			e
		)),
		Frame::PredicateEnd { kind, input, .. } =>
		{
			cut_error(with_context(*input, kind.contexts().range_end, e))
		},
		Frame::LimitCount { kind, input } =>
		{
			with_context(*input, kind.contexts().count, e)
		},
		Frame::CallArgument { input, .. } =>
		{
			cut_error(with_context(*input, ARGUMENT_CONTEXT, e))
		},
	}
}

/// Answer the error of an alternation whose every alternative failed
/// recoverably, as [`exhaust`] finishes with.
///
/// # Parameters
/// - `input`: The input of the alternation.
/// - `error`: The merged errors of the alternatives.
///
/// # Returns
/// The error.
fn exhausted<'src>(
	input: Span<'src>,
	error: ParseError<'src>
) -> nom::Err<ParseError<'src>>
{
	nom::Err::Error(ParseError::append(input, ErrorKind::Alt, error))
}

/// Parse a literal, as `context(CONSTANT_CONTEXT, constant)`, or as
/// `context(CONSTANT_CONTEXT, integer)` if `signed`, but read an
/// [overlaid](Overlay::Constant) constant.
///
/// # Parameters
/// - `input`: The input.
/// - `signed`: Whether the literal is a signed [`integer`], rather than an
///   unsigned [`constant`].
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input and the constant.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
#[inline(always)]
fn parse_constant<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	signed: bool,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, Constant, ParseError<'src>>
{
	if rec.recovering()
		&& let Some(parsed) = rec.take_constant(input, signed)
	{
		return Ok(parsed);
	}
	let parsed =
		context(CONSTANT_CONTEXT, literal(signed)).parse_complete(input);
	if let Ok((_, constant)) = &parsed
	{
		let role = if signed { Role::Face } else { Role::Constant };
		rec.report(constant.span.start, constant.span.end, role);
	}
	parsed
}

/// Answer the parser of a literal: a signed [`integer`], as the faces of dice
/// take, or an unsigned [`constant`], as everything else does.
///
/// # Parameters
/// - `signed`: Whether the literal is signed.
///
/// # Returns
/// The parser of the literal.
#[inline(always)]
fn literal<'src>(
	signed: bool
) -> fn(Span<'src>) -> IResult<Span<'src>, Constant, ParseError<'src>>
{
	if signed { integer } else { constant }
}

/// Parse a negative constant, as [`negative_constant`], but read an
/// [overlaid](Overlay::Constant) constant after the `-`. While recovering,
/// decline a negative constant that a keep, reroll, or explode clause follows,
/// as [`negative_constant`] declines one that a dice operator follows, so that
/// the constant is read as a dice count, and the clause offers a repair at
/// [`Site::DiceOperator`], e.g., `-3D6 keep lowest` for `-3 keep lowest`. The
/// input fails in the ordinary parse anyway, so only the error differs.
///
/// # Parameters
/// - `input`: The input.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input and the constant.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
#[inline(always)]
fn parse_negative_constant<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, Expression<'src>, ParseError<'src>>
{
	if rec.recovering()
		&& let Some(parsed) = rec.take_negative_constant(input)
	{
		return Ok(parsed);
	}
	match negative_constant(input)
	{
		Ok((rest, _))
			if rec.recovering()
				&& begins_misplaced_clause(
					skip_whitespace(rest).fragment()
				)
				.is_some() =>
		{
			Err(nom::Err::Error(ParseError::from_error_kind(
				input,
				ErrorKind::Digit
			)))
		},
		parsed =>
		{
			if let Ok((rest, _)) = &parsed
			{
				// The sign is negation, and the digits an unsigned constant.
				let start = input.location_offset();
				let end = rest.location_offset();
				let digits = input.fragment()[1..end - start]
					.trim_start_matches(is_token_space);
				rec.report(start, start + 1, Role::Negation);
				rec.report(end - digits.len(), end, Role::Constant);
			}
			parsed
		}
	}
}

/// Parse a variable, as `context(VARIABLE_CONTEXT, variable)`, but read an
/// [overlaid](Repair::Variable) identifier as a variable, and repair the
/// variable if it fails unrecoverably.
///
/// # Parameters
/// - `input`: The input.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input, and the variable and the span of its name.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed, or repaired.
#[inline(always)]
fn parse_variable<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, (Variable<'src>, SourceSpan), ParseError<'src>>
{
	if rec.recovering()
		&& let Some((rest, variable)) = rec.take_variable(input)
	{
		let name_span = variable.span;
		return Ok((rest, (variable, name_span)));
	}
	match context(VARIABLE_CONTEXT, braced_name).parse_complete(input)
	{
		Ok((rest, name)) =>
		{
			let variable = Variable {
				name: canonical_name(name.fragment()),
				span: SourceSpan {
					start: input.location_offset(),
					end: rest.location_offset()
				}
			};
			Ok((rest, (variable, span_of(name))))
		},
		Err(nom::Err::Failure(_)) if rec.recovering() =>
		{
			recover_variable(input, stack, rec)
		},
		Err(e) => Err(e)
	}
}

/// Parse a variable that failed unrecoverably, as `context(VARIABLE_CONTEXT,
/// variable)`, but repair its missing name or closing brace. The repair of the
/// closing brace may [break](Repair::Break) the name early, since the name
/// reads everything up to the next brace or the end of the input.
///
/// # Parameters
/// - `input`: The input of the variable, which begins with `{`.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input, and the variable and the span of its name. A supplied
/// name has an empty span, at the position where the repair supplies it. A
/// broken name ends where the repair supplies the closing brace.
///
/// # Errors
/// * [`Err`](nom::Err) if the policy declined a repair.
fn recover_variable<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, (Variable<'src>, SourceSpan), ParseError<'src>>
{
	let start = input.location_offset();
	let within = |e| with_context(input, VARIABLE_CONTEXT, e);
	let (after, _) = char('{').parse_complete(input)?;
	let (after, name, name_span) = match cut(preceded(
		name_space0,
		context(IDENTIFIER_CONTEXT, identifier)
	))
	.parse_complete(after)
	{
		Ok((after, name)) =>
		{
			(after, canonical_name(name.fragment()), span_of(name))
		},
		Err(e) =>
		{
			let site = Site::VariableName { opener: start };
			let at = rec.repair(site, within(e), stack)?;
			let position = at.location_offset();
			let name_span = SourceSpan {
				start: position,
				end: position
			};
			(at, Cow::Borrowed(PLACEHOLDER_NAME), name_span)
		}
	};
	let brace: IResult<Span, char, ParseError> = cut(preceded(
		name_space0,
		context(CLOSING_BRACE_CONTEXT, char('}'))
	))
	.parse_complete(after);
	let (rest, name, name_span) = match brace
	{
		Ok((rest, _)) => (rest, name, name_span),
		Err(e) =>
		{
			let site = Site::Closer {
				opener: start,
				closer: '}'
			};
			// The name may break before the failure, but only if the source
			// supplied it; a supplied name has an empty span.
			let broken =
				rec.consult_with(site, within(e), stack, |repair, at| {
					match repair
					{
						Repair::Fix => Some((at, None)),
						Repair::Break { end }
							if name_span.start < end
								&& end <= name_span.end =>
						{
							Some((input.take_from(end - start), Some(end)))
						},
						_ => None
					}
				})?;
			match broken
			{
				(rest, None) => (rest, name, name_span),
				(rest, Some(end)) =>
				{
					// A broken name at the start of the body of a function
					// without formal parameters, followed by a `,` or `:`, is
					// the first formal parameter of the edited source, so
					// parse the function again, reading the `}` where the
					// break supplied it.
					if stack
						.first()
						.is_some_and(|frame| starts_body(frame, start))
						&& precedes_parameters(rest)
					{
						rec.overlay =
							Some(Overlay::Brace { opener: start, end });
						rec.restart = Some(input);
					}
					// The name ends before any whitespace before the break.
					let written = input.fragment()
						[name_span.start - start..end - start]
						.trim_end();
					let name_span = SourceSpan {
						start: name_span.start,
						end: name_span.start + written.len()
					};
					(rest, canonical_name(written), name_span)
				}
			}
		}
	};
	let variable = Variable {
		name,
		span: SourceSpan {
			start,
			end: rest.location_offset()
		}
	};
	Ok((rest, (variable, name_span)))
}

/// Parse custom faces that failed unrecoverably, as
/// `context(CUSTOM_FACES_CONTEXT, custom_faces)`, but repair their missing
/// first face or closing bracket.
///
/// # Parameters
/// - `input`: The input of the custom faces, which begins with `[`.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input and the faces.
///
/// # Errors
/// * [`Err`](nom::Err) if the policy declined a repair.
fn recover_custom_faces<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, Vec<i32>, ParseError<'src>>
{
	let opener = input.location_offset();
	let within = |e| with_context(input, CUSTOM_FACES_CONTEXT, e);
	let (after, _) = char('[').parse_complete(input)?;
	// The faces are `cut(separated_list1(comma, cut(face)))`: a missing first
	// face is unrecoverable, as is a comma without a face after it.
	let (after, first) = match value().parse_complete(after)
	{
		Ok(parsed) => parsed,
		Err(e) =>
		{
			let site = Site::FaceValue { opener };
			match rec.repair(site, within(cut_error(e)), stack)
			{
				Ok(at) => (at, PLACEHOLDER_FACE),
				Err(e) =>
				{
					// The `[` was read, even though the parse fails.
					rec.report_literals(input, after, Role::FacesBracket);
					return Err(e);
				}
			}
		}
	};
	let mut faces = vec![first];
	let (after, retracted) =
		match read_more_values(opener, after, &mut faces, stack, rec)
		{
			Ok(more) => more,
			Err((at, e)) =>
			{
				// The faces were read, even though the parse fails.
				rec.report_literals(input, at, Role::FacesBracket);
				return Err(within(e));
			}
		};
	let bracket: IResult<Span, char, ParseError> = cut(preceded(
		multispace0,
		context(CLOSING_BRACKET_CONTEXT, char(']'))
	))
	.parse_complete(after);
	let rest = match bracket
	{
		Ok((rest, _)) => rest,
		Err(e) =>
		{
			let site = Site::Closer {
				opener,
				closer: ']'
			};
			match rec.repair(site, within(e), stack)
			{
				Ok(rest) => rest,
				Err(e) =>
				{
					// The faces were read, even though the parse fails.
					rec.report_literals(input, after, Role::FacesBracket);
					return Err(e);
				}
			}
		}
	};
	// The supplied first face or `]` is not in the source, so is not
	// reported.
	report_list(input, retracted, rest, Role::FacesBracket, rec);
	Ok((rest, faces))
}

/// Parse formal parameters that failed, or whose first a repair
/// [overlaid](Recoverer::take_parameter), as [`parameters`], but read the
/// overlaid parameter, and repair a trailing comma, a bare word after a comma,
/// or a missing `:`.
///
/// # Parameters
/// - `input`: The input of the parameters.
/// - `stack`: The stack of suspended productions.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input and the formal parameters, if any.
///
/// # Errors
/// * [`Err`](nom::Err) if the policy declined a repair.
fn recover_parameters<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	stack: &Stack<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> IResult<Span<'src>, Option<Vec<Parameter<'src>>>, ParseError<'src>>
{
	let comma = || preceded(multispace0, char(','));
	let (mut rest, mut parameters) = match rec.take_parameter(input)
	{
		Some((rest, first)) =>
		{
			// The `{` of an overlaid braced name is in the source, but its
			// `}` is not; an overlaid variable has no braces at all.
			let braced = SourceSpan {
				start: input.location_offset(),
				end: first.span.end
			};
			rec.report_name(braced, first.span, Role::ParameterDeclaration);
			// Continue the list after the overlaid parameter, as
			// `separated_list0` would.
			let mut parameters = vec![first];
			let rest = read_parameters(rest, &mut parameters, rec)?;
			(rest, parameters)
		},
		None =>
		{
			// As `separated_list0`.
			let mut parameters = Vec::new();
			let rest = match read_parameter(input)?
			{
				Some((rest, first)) =>
				{
					report_parameter(input, rest, &first, rec);
					parameters.push(first);
					read_parameters(rest, &mut parameters, rec)?
				},
				None => input
			};
			(rest, parameters)
		}
	};
	loop
	{
		// A comma after the parameters is unrecoverable, as in `parameters`.
		let trailing: IResult<Span, char, ParseError> = terminated(
			comma(),
			context(PARAMETER_CONTEXT, fail_parser::<_, (), ParseError>())
		)
		.parse_complete(rest);
		let e = match trailing
		{
			Err(nom::Err::Error(e))
				if matches!(
					e.errors[0].1,
					NomErrorKind::Nom(ErrorKind::Fail)
				) =>
			{
				nom::Err::Failure(e)
			},
			_ => break
		};
		let site = if parameters.is_empty()
		{
			Site::LeadingComma
		}
		else
		{
			Site::ParameterName
		};
		// Supply a parameter, read a bare word after the comma as one, as
		// though the source enclosed it in braces, or retract the comma.
		let (at, supplied) =
			rec.consult_with(site, e, stack, |repair, at| match repair
			{
				Repair::Fix if site.is_fixable() =>
				{
					let position = at.location_offset();
					let supplied = Parameter {
						name: Cow::Borrowed(PLACEHOLDER_NAME),
						span: SourceSpan {
							start: position,
							end: position
						}
					};
					Some((at, Some(supplied)))
				},
				Repair::Retract
					if site == Site::ParameterName
						&& precedes_parameters(at) =>
				{
					Some((at, None))
				},
				Repair::Variable { end } if site == Site::ParameterName =>
				{
					let name = skip_whitespace(at);
					let start = name.location_offset();
					(end > start).then(|| {
						let length = end - start;
						let read = Parameter {
							name: canonical_name(&name.fragment()[..length]),
							span: SourceSpan { start, end }
						};
						(name.take_from(length), Some(read))
					})
				},
				_ => None
			})?;
		// The comma is in the source, unless the repair retracted it. A
		// parameter that the repair supplied, or read from a bare word, is
		// not reported, since the source has no braced name there.
		if supplied.is_some()
		{
			let (after, _) = comma().parse_complete(rest)?;
			rec.report_before(after, Role::Separator);
		}
		parameters.extend(supplied);
		// Continue the list after the supplied parameter, or the retracted
		// comma, as `separated_list0` would.
		rest = read_parameters(at, &mut parameters, rec)?;
	}
	if parameters.is_empty()
	{
		return Ok((rest, None));
	}
	let colon: IResult<Span, char, ParseError> =
		preceded(multispace0, context(NEXT_PARAMETER_CONTEXT, char(':')))
			.parse_complete(rest);
	let rest = match colon
	{
		Ok((rest, _)) =>
		{
			rec.report_before(rest, Role::Separator);
			rest
		},
		// A lone braced name without a `:` after it begins the body, as in
		// `parameters`. An overlaid parameter is never lone, since the engine
		// overlays one only where a `,` or `:` follows it.
		Err(_) if parameters.len() == 1 =>
		{
			rec.retract(input.location_offset());
			return Ok((input, None));
		},
		Err(e) => rec.repair(Site::ParameterColon, e, stack)?
	};
	Ok((rest, Some(parameters)))
}

/// Read a formal parameter, as `preceded(multispace0, parameter)`.
///
/// # Parameters
/// - `input`: The input.
///
/// # Returns
/// The remaining input and the parameter, which [`report_parameter`] reports,
/// or `None` if no parameter begins at the input, after any whitespace.
///
/// # Errors
/// * [`Err`](nom::Err) if the parameter fails unrecoverably.
fn read_parameter<'src>(
	input: Span<'src>
) -> Result<Option<(Span<'src>, Parameter<'src>)>, nom::Err<ParseError<'src>>>
{
	match preceded(multispace0, parameter).parse_complete(input)
	{
		Ok((rest, name)) => Ok(Some((
			rest,
			Parameter {
				name: canonical_name(name.fragment()),
				span: span_of(name)
			}
		))),
		Err(nom::Err::Error(_)) => Ok(None),
		Err(e) => Err(e)
	}
}

/// Report a formal parameter that [`read_parameter`] read: its braces and its
/// identifier.
///
/// # Parameters
/// - `input`: The input of the parameter, perhaps at whitespace.
/// - `rest`: The input after the parameter.
/// - `parameter`: The parameter.
/// - `rec`: The recovery state.
fn report_parameter<'src, R: Recovery<'src>, S: TokenSink>(
	input: Span<'src>,
	rest: Span<'src>,
	parameter: &Parameter<'src>,
	rec: &mut Recoverer<'src, '_, R, S>
)
{
	let braced = SourceSpan {
		start: skip_whitespace(input).location_offset(),
		end: rest.location_offset()
	};
	rec.report_name(braced, parameter.span, Role::ParameterDeclaration);
}

/// Read the formal parameters that follow another, each after a `,`, as
/// `many0(preceded(comma, preceded(multispace0, parameter)))`, and report each
/// `,` and parameter.
///
/// # Parameters
/// - `input`: The input after the parameter that they follow.
/// - `parameters`: The parameters, to which those read are appended.
/// - `rec`: The recovery state.
///
/// # Returns
/// The remaining input.
///
/// # Errors
/// * [`Err`](nom::Err) if a parameter fails unrecoverably.
fn read_parameters<'src, R: Recovery<'src>, S: TokenSink>(
	mut input: Span<'src>,
	parameters: &mut Vec<Parameter<'src>>,
	rec: &mut Recoverer<'src, '_, R, S>
) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
{
	loop
	{
		let comma: IResult<Span, char, ParseError> =
			preceded(multispace0, char(',')).parse_complete(input);
		let Ok((after, _)) = comma
		else
		{
			return Ok(input);
		};
		let Some((rest, read)) = read_parameter(after)?
		else
		{
			return Ok(input);
		};
		rec.report_before(after, Role::Separator);
		report_parameter(after, rest, &read, rec);
		parameters.push(read);
		input = rest;
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Utilities.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Skip any leading whitespace, as [`multispace0`] does.
///
/// # Parameters
/// - `input`: The input.
///
/// # Returns
/// The input after the whitespace.
fn skip_whitespace(input: Span<'_>) -> Span<'_>
{
	let skipped: IResult<Span, Span, ParseError> =
		multispace0.parse_complete(input);
	// `multispace0` accepts the empty string, so it cannot fail.
	skipped.map_or(input, |(rest, _)| rest)
}

/// Answer whether the input begins with `-`. After the direction of a drop or
/// keep clause, or the predicate of a reroll or explode clause, this precludes
/// a count, which never begins with `-`, so after a clause, a `-` is always
/// subtraction: `4D6 drop lowest -1` is `(4D6 drop lowest) - 1`. A count of
/// zero or less drops, keeps, rerolls, or explodes nothing, so reading
/// `-1` as the count would quietly discard the clause, or every die, which is
/// never what the author meant. A negative count must be grouped, as in `4D6
/// drop lowest (-1)`, or bound to a variable. The [constant] of a count is
/// unsigned anyway, but the count is optional, so the clause ends before the
/// `-`, rather than the engine trying, and perhaps [repairing](Repair), a count
/// there.
///
/// # Parameters
/// - `input`: The input, without leading whitespace.
///
/// # Returns
/// `true` if the input begins with `-`, `false` otherwise.
pub(super) fn begins_with_minus(input: Span<'_>) -> bool
{
	input.fragment().starts_with('-')
}

/// Answer whether a position is the start of the body of a function without
/// formal parameters.
///
/// # Parameters
/// - `frame`: The frame at the bottom of the stack, or of the nearest barrier.
/// - `at`: The byte offset of the position.
///
/// # Returns
/// `true` if `frame` awaits the body of a function without formal parameters,
/// and the body begins at `at`, `false` otherwise.
fn starts_body(frame: &Frame<'_>, at: usize) -> bool
{
	matches!(
		frame,
		Frame::FunctionBody {
			formal: false,
			input,
			..
		} if input.location_offset() == at
	)
}

/// Answer whether a `,` or `:` follows any whitespace at the start of the
/// input, as it follows a formal parameter.
///
/// # Parameters
/// - `input`: The input.
///
/// # Returns
/// `true` if a `,` or `:` begins the input after its whitespace, `false`
/// otherwise.
fn precedes_parameters(input: Span<'_>) -> bool
{
	skip_whitespace(input).fragment().starts_with([',', ':'])
}

/// Answer the span of a name.
///
/// # Parameters
/// - `name`: The name.
///
/// # Returns
/// The span of the name.
fn span_of(name: Span<'_>) -> SourceSpan
{
	let start = name.location_offset();
	SourceSpan {
		start,
		end: start + name.fragment().len()
	}
}

/// Attach [`BINDING_CONTEXT`] to an error of a binding, if the binding is the
/// variable alternative of an atom or a primary expression, continued, as the
/// original attached it to the error of the binding alternative.
///
/// # Parameters
/// - `atom`: The input of the atom or primary expression, if the binding is the
///   variable alternative of one, continued; see [`variable_or_binding`].
/// - `e`: The error.
///
/// # Returns
/// The error, with the context attached if appropriate.
fn within_binding<'src>(
	atom: Option<Span<'src>>,
	e: nom::Err<ParseError<'src>>
) -> nom::Err<ParseError<'src>>
{
	match atom
	{
		Some(input) => with_context(input, BINDING_CONTEXT, e),
		None => e
	}
}

/// Attach a context to the error of an outcome, if any, as [`context`] does.
///
/// # Parameters
/// - `input`: The input at which the context applies.
/// - `label`: The context.
/// - `outcome`: The outcome.
///
/// # Returns
/// The outcome, with the context attached to its error.
fn with_context_on_error<'src>(
	input: Span<'src>,
	label: &'static str,
	outcome: Outcome<'src>
) -> Outcome<'src>
{
	outcome.map_err(|e| with_context(input, label, e))
}

/// Answer a closing delimiter, as `cut(preceded(multispace0, context(LABEL,
/// char(CLOSER))))`.
///
/// # Parameters
/// - `closer`: The closing delimiter.
/// - `label`: The context of a missing delimiter.
/// - `input`: The input.
///
/// # Returns
/// The input after the delimiter.
///
/// # Errors
/// * [`Failure`](nom::Err::Failure) if the delimiter is missing.
fn close<'src>(
	closer: char,
	label: &'static str,
	input: Span<'src>
) -> Result<Span<'src>, nom::Err<ParseError<'src>>>
{
	cut(preceded(multispace0, context(label, char(closer))))
		.parse_complete(input)
		.map(|(rest, _)| rest)
}

/// Settle the optional drop expression of a drop or keep clause, as
/// `opt(preceded(multispace0, context(COUNT_CONTEXT, drop_expression)))`,
/// where `COUNT_CONTEXT` is [`DROP_EXPRESSION_CONTEXT`] or
/// [`KEEP_EXPRESSION_CONTEXT`].
///
/// # Parameters
/// - `kind`: The kind of the clause.
/// - `rest`: The input after the direction, where the clause ends if it has no
///   drop expression.
/// - `input`: The input of the drop expression.
/// - `outcome`: The outcome of the drop expression.
///
/// # Returns
/// The input after the clause, and the drop expression, if any.
///
/// # Errors
/// * [`Failure`](nom::Err::Failure) if the drop expression failed
///   unrecoverably.
fn optional_drop<'src>(
	kind: ClauseKind,
	rest: Span<'src>,
	input: Span<'src>,
	outcome: Outcome<'src>
) -> IResult<Span<'src>, Option<Box<Expression<'src>>>, ParseError<'src>>
{
	match with_context_on_error(input, kind.count_context(), outcome)
	{
		Ok((rest, drop)) => Ok((rest, Some(Box::new(drop.into_expression())))),
		// `opt` rewinds after a recoverable error, and discards it.
		Err(nom::Err::Error(_)) => Ok((rest, None)),
		Err(e) => Err(e)
	}
}
