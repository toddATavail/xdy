//! # Iterative S-expression reader
//!
//! Herein is the engine behind [`read_s_expr`](super::read_s_expr). It reads
//! the compound forms of the S-expression grammar with an explicit stack in
//! place of the machine stack, so its stack depth is constant however deep the
//! input.
//!
//! Each compound form that the reader has opened but not yet closed is a
//! [frame](Frame) on the stack, which records the kind and span of the form,
//! the parts of it read so far, and what it needs to validate its next child.
//! The engine reads one subexpression at a time on behalf of the frame on top
//! of the stack. A constant or variable is complete at once, but a compound
//! form pushes a frame of its own. A complete subexpression is _delivered_ to
//! the frame on top of the stack, which validates it and then either awaits
//! another child or closes, delivering its own form to the frame below. The
//! [function](Function) is the bottom frame, so reading is complete when it
//! closes. The flat parts of the grammar — span prefixes, words, identifiers,
//! integers, and the parameter and faces lists — are the leaf readers of the
//! parent module, which the engine calls directly.
//!
//! The engine preserves the behavior of the recursive reader that it replaced
//! — its values, remaining input, and errors, down to their locations — except
//! where 0.13 added checks: every name must be braced and
//! [canonical](crate::parser::is_canonical_name), and the name span of a
//! binding must lie within the binding's span. In particular, a child is
//! validated for containment within its parent, then for order among its
//! siblings, and only then for the type that its position demands; and an
//! [`ExpectedDiceExpression`] is located before the whitespace that precedes
//! the offending child, not at the child itself.
//!
//! [`ExpectedDiceExpression`]: SExprError::ExpectedDiceExpression

use std::{borrow::Cow, mem};

use nom::Input;

use super::{
	SExprError, SExprLocation, SExprResult, Span, expect_char, read_faces,
	read_ident, read_integer, read_params, read_span_prefix, read_word,
	skip_ws, validate_containment, validate_sibling_order
};
use crate::{
	ast::{
		Add, ArithmeticExpression, Binding, Constant, CustomDice,
		DiceExpression, Div, DropHighest, DropLowest, Exp, Expression,
		Function, Group, Mod, Mul, Neg, Parameter, Range, StandardDice, Sub,
		Variable
	},
	span::{SourceSpan, Spanned}
};

////////////////////////////////////////////////////////////////////////////////
//                                  Frames.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A compound form that the reader has opened but not yet closed.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being read.
struct Frame<'src>
{
	/// The kind of the form, together with the parts of it read so far.
	form: Form<'src>,

	/// The span of the form, which must contain the span of each child.
	span: SourceSpan,

	/// The span of the previous sibling of the awaited child, which must
	/// precede the child's span, or [`SourceSpan::default`] if the child has
	/// no previous sibling.
	prev: SourceSpan,

	/// The input at which the awaited child began, after whitespace, to which
	/// containment and sibling-order errors are attributed. The engine sets it
	/// whenever it begins to read a child.
	mark: Span<'src>
}

/// The kind of a [frame](Frame)'s form, together with the parts of it read so
/// far. Each variant awaits the first of its missing parts.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being read.
enum Form<'src>
{
	/// `(function [parameters] body)`, awaiting the body.
	Function
	{
		/// The formal parameters.
		parameters: Option<Vec<Parameter<'src>>>
	},

	/// `(operator left right)` for a binary arithmetic operator.
	Binary
	{
		/// The operator.
		operator: Operator,

		/// The left operand, once read.
		left: Option<Expression<'src>>
	},

	/// `(neg operand)`.
	Neg,

	/// `(group expression)`.
	Group,

	/// `(standard-dice count faces)`.
	StandardDice
	{
		/// The count, once read.
		count: Option<Expression<'src>>
	},

	/// `(custom-dice count [faces])`, awaiting the count, after which it reads
	/// the faces directly.
	CustomDice,

	/// `(drop-lowest dice drop?)` or `(drop-highest dice drop?)`.
	Drop
	{
		/// The direction of the drop.
		direction: Direction,

		/// The input immediately after the keyword, before any whitespace, to
		/// which an [`ExpectedDiceExpression`] is attributed.
		///
		/// [`ExpectedDiceExpression`]: SExprError::ExpectedDiceExpression
		after_keyword: Span<'src>,

		/// The dice expression, once read.
		dice: Option<DiceExpression<'src>>
	},

	/// `(range start end)`.
	Range
	{
		/// The start, once read.
		start: Option<Expression<'src>>
	},

	/// `(binding name expression)`, whose name is read with its keyword.
	Binding
	{
		/// The bound name.
		name: &'src str,

		/// The span of the bound name.
		name_span: SourceSpan
	}
}

/// A binary arithmetic operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Operator
{
	/// `add`.
	Add,

	/// `sub`.
	Sub,

	/// `mul`.
	Mul,

	/// `div`.
	Div,

	/// `mod`.
	Mod,

	/// `exp`.
	Exp
}

impl Operator
{
	/// Combine two operands with the receiver.
	///
	/// # Parameters
	/// - `left`: The left operand.
	/// - `right`: The right operand.
	/// - `span`: The span of the form.
	///
	/// # Returns
	/// The arithmetic expression.
	fn apply<'src>(
		self,
		left: Expression<'src>,
		right: Expression<'src>,
		span: SourceSpan
	) -> Expression<'src>
	{
		let left = Box::new(left);
		let right = Box::new(right);
		Expression::Arithmetic(match self
		{
			Operator::Add =>
			{
				ArithmeticExpression::Add(Add { left, right, span })
			},
			Operator::Sub =>
			{
				ArithmeticExpression::Sub(Sub { left, right, span })
			},
			Operator::Mul =>
			{
				ArithmeticExpression::Mul(Mul { left, right, span })
			},
			Operator::Div =>
			{
				ArithmeticExpression::Div(Div { left, right, span })
			},
			Operator::Mod =>
			{
				ArithmeticExpression::Mod(Mod { left, right, span })
			},
			Operator::Exp =>
			{
				ArithmeticExpression::Exp(Exp { left, right, span })
			},
		})
	}
}

/// The direction of a drop form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction
{
	/// `drop-lowest`.
	Lowest,

	/// `drop-highest`.
	Highest
}

impl Direction
{
	/// Build a drop form in the receiver's direction.
	///
	/// # Parameters
	/// - `dice`: The dice expression.
	/// - `drop`: The drop amount, if any.
	/// - `span`: The span of the form.
	///
	/// # Returns
	/// The drop expression.
	fn apply<'src>(
		self,
		dice: DiceExpression<'src>,
		drop: Option<Expression<'src>>,
		span: SourceSpan
	) -> Expression<'src>
	{
		let dice = Box::new(dice);
		let drop = drop.map(Box::new);
		Expression::Dice(match self
		{
			Direction::Lowest =>
			{
				DiceExpression::DropLowest(DropLowest { dice, drop, span })
			},
			Direction::Highest =>
			{
				DiceExpression::DropHighest(DropHighest { dice, drop, span })
			},
		})
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Steps.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A step of the engine.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being read.
enum Step<'src>
{
	/// Read a subexpression at the input, on behalf of the frame on top of the
	/// stack.
	Read(Span<'src>),

	/// Deliver a complete subexpression to the frame on top of the stack. The
	/// input follows the subexpression.
	Deliver(Span<'src>, Expression<'src>),

	/// Finish reading, with the remaining input and the function.
	Finish(Span<'src>, Function<'src>)
}

/// The outcome of a [step](Step): the next step, or an error.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text being read.
type Outcome<'src> = Result<Step<'src>, nom::Err<SExprError>>;

////////////////////////////////////////////////////////////////////////////////
//                                  Driver.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Read the complete top-level form: an optional `^[start end]` span prefix
/// followed by `(function params body)`. Trailing whitespace after the closing
/// parenthesis is consumed; the caller is responsible for verifying that no
/// input remains.
///
/// After the parameters, the engine alternates between reading and
/// delivering. Reading either produces a complete leaf, or opens a compound
/// form by pushing a [frame](Frame) and reads again on its behalf. Delivering
/// pops the frame on top of the stack and validates the subexpression against
/// it, and then the frame either awaits another child, so it is pushed back
/// and reading resumes, or closes, so its own form is delivered to the frame
/// below. Reading finishes when the function frame closes.
///
/// ```mermaid
/// flowchart TD
///     S(["read_function(input)"]) --> P["read prefix, keyword, and<br/>parameters; push the function frame"]
///     P --> R["Read(input)"]
///     R -->|"constant or variable"| D["Deliver(input, child)"]
///     R -->|"compound form: push a frame"| R
///     D -->|"pop a frame; validate the child"| F{"frame complete?"}
///     F -->|"no: push the frame back"| R
///     F -->|"yes, and it is the function"| E(["Finish(input, function)"])
///     F -->|"yes: read ')'"| D
/// ```
///
/// # Parameters
/// - `input`: The input text to read from.
///
/// # Returns
/// A pair of the remaining input and the parsed [`Function`].
///
/// # Errors
/// - [`SExprError`] if the input is not a well-formed `(function params body)`
///   s-expression or if any span validation fails.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub(super) fn read_function(input: Span<'_>) -> SExprResult<'_, Function<'_>>
{
	let (input, span) = read_span_prefix(input)?;
	let (input, _) = expect_char('(')(input)?;
	let (input, _) = skip_ws(input)?;
	let kw_mark = input;
	let (input, kw) = read_word(input)?;
	let keyword = *kw.fragment();
	if keyword != "function"
	{
		return Err(nom::Err::Failure(SExprError::ExpectedTopLevelFunction {
			found: keyword.to_string(),
			location: SExprLocation::of(kw_mark)
		}))
	}
	let (input, (parameters, last_param_span)) = read_params(input, span)?;
	let mut stack = Vec::new();
	let mut step = await_child(
		&mut stack,
		Form::Function { parameters },
		span,
		last_param_span,
		input
	);
	loop
	{
		step = match step
		{
			Step::Read(input) => read(input, &mut stack)?,
			Step::Deliver(input, child) =>
			{
				// The function frame is the last to be popped, and popping it
				// finishes the run, so the stack cannot be empty here.
				let frame = stack.pop().expect("stack must not be empty");
				deliver(frame, child, input, &mut stack)?
			},
			Step::Finish(input, function) => return Ok((input, function))
		}
	}
}

/// Read a subexpression, optionally preceded by a `^[start end]` span prefix,
/// on behalf of the frame on top of the stack. A constant or variable is
/// complete at once, but a compound form is opened by [`open`].
///
/// # Parameters
/// - `input`: The input text to read from.
/// - `stack`: The stack of open forms, which must not be empty.
///
/// # Returns
/// The next step.
///
/// # Errors
/// - [`SExprError`] if the subexpression is malformed, or if the input ends.
fn read<'src>(input: Span<'src>, stack: &mut Vec<Frame<'src>>)
-> Outcome<'src>
{
	let (input, _) = skip_ws(input)?;
	if let Some(parent) = stack.last_mut()
	{
		parent.mark = input;
	}
	let (input, span) = read_span_prefix(input)?;
	let (input, _) = skip_ws(input)?;
	match input.fragment().chars().next()
	{
		Some('(') => open(input.take_from(1), span, stack),
		Some(c) if c == '-' || c.is_ascii_digit() =>
		{
			let (input, value) = read_integer(input)?;
			Ok(Step::Deliver(
				input,
				Expression::Constant(Constant { value, span })
			))
		},
		Some(_) =>
		{
			let (input, name) = read_ident(input)?;
			Ok(Step::Deliver(
				input,
				Expression::Variable(Variable {
					name: Cow::Borrowed(name),
					span
				})
			))
		},
		None => Err(nom::Err::Failure(SExprError::ExpectedExpression {
			location: SExprLocation::of(input)
		}))
	}
}

/// Open a compound form, whose `(` has been read, by reading its keyword and
/// pushing a frame for it. A binding's name is read here as well.
///
/// # Parameters
/// - `input`: The input text after the `(`.
/// - `span`: The span of the form.
/// - `stack`: The stack of open forms.
///
/// # Returns
/// The next step, which reads the first child of the form.
///
/// # Errors
/// - [`SExprError::NestedFunctionKeyword`] if the keyword is `function`.
/// - [`SExprError::UnknownKeyword`] if the keyword is unrecognized.
/// - [`SExprError::ChildSpanEscapesParent`] if a binding's name escapes the
///   binding's span.
/// - [`SExprError`] if the keyword or a binding's name is malformed.
fn open<'src>(
	input: Span<'src>,
	span: SourceSpan,
	stack: &mut Vec<Frame<'src>>
) -> Outcome<'src>
{
	let (input, _) = skip_ws(input)?;
	let kw_mark = input;
	let (mut input, kw) = read_word(input)?;
	let mut prev = SourceSpan::default();
	let binary = |operator| Form::Binary {
		operator,
		left: None
	};
	let form = match *kw.fragment()
	{
		"add" => binary(Operator::Add),
		"sub" => binary(Operator::Sub),
		"mul" => binary(Operator::Mul),
		"div" => binary(Operator::Div),
		"mod" => binary(Operator::Mod),
		"exp" => binary(Operator::Exp),
		"neg" => Form::Neg,
		"group" => Form::Group,
		"standard-dice" => Form::StandardDice { count: None },
		"custom-dice" => Form::CustomDice,
		"drop-lowest" => Form::Drop {
			direction: Direction::Lowest,
			after_keyword: input,
			dice: None
		},
		"drop-highest" => Form::Drop {
			direction: Direction::Highest,
			after_keyword: input,
			dice: None
		},
		"range" => Form::Range { start: None },
		"binding" =>
		{
			// The bound name is emitted as an ident with an optional span
			// prefix (only when spans are serialized). Read the prefix
			// explicitly so the [`Binding::name_span`] survives the lossless
			// round-trip; fall back to the default span when no prefix is
			// present. The name's span lies within the binding's, and
			// precedes the expression's.
			let (rest, _) = skip_ws(input)?;
			let name_mark = rest;
			let (rest, name_span) = read_span_prefix(rest)?;
			validate_containment(span, name_span, name_mark)?;
			let (rest, name) = read_ident(rest)?;
			input = rest;
			prev = name_span;
			Form::Binding { name, name_span }
		},
		"function" =>
		{
			return Err(nom::Err::Failure(SExprError::NestedFunctionKeyword {
				location: SExprLocation::of(kw_mark)
			}))
		},
		keyword =>
		{
			return Err(nom::Err::Failure(SExprError::UnknownKeyword {
				keyword: keyword.to_string(),
				location: SExprLocation::of(kw_mark)
			}))
		},
	};
	Ok(await_child(stack, form, span, prev, input))
}

/// Deliver a complete subexpression to the frame that awaited it. The child
/// is validated for containment within the frame's form, then for order after
/// its previous sibling, and then the frame either awaits its next child or
/// closes.
///
/// # Parameters
/// - `frame`: The frame, popped from the stack.
/// - `child`: The subexpression.
/// - `input`: The input text after the subexpression.
/// - `stack`: The stack of open forms, without the frame.
///
/// # Returns
/// The next step.
///
/// # Errors
/// - [`SExprError::ChildSpanEscapesParent`] if the child's span escapes the
///   form's span.
/// - [`SExprError::SiblingSpanOutOfOrder`] if the child's span overlaps or
///   precedes its previous sibling's span.
/// - [`SExprError::ExpectedDiceExpression`] if the first child of a drop form
///   is not a dice expression.
/// - [`SExprError`] if a faces list or the closing `)` is malformed.
fn deliver<'src>(
	frame: Frame<'src>,
	child: Expression<'src>,
	input: Span<'src>,
	stack: &mut Vec<Frame<'src>>
) -> Outcome<'src>
{
	let Frame {
		form,
		span,
		prev,
		mark
	} = frame;
	let child_span = child.span();
	validate_containment(span, child_span, mark)?;
	validate_sibling_order(prev, child_span, mark)?;
	match form
	{
		Form::Function { parameters } =>
		{
			let (input, _) = expect_char(')')(input)?;
			let (input, _) = skip_ws(input)?;
			Ok(Step::Finish(
				input,
				Function {
					parameters,
					body: child,
					span
				}
			))
		},
		Form::Binary {
			operator,
			left: None
		} => Ok(await_child(
			stack,
			Form::Binary {
				operator,
				left: Some(child)
			},
			span,
			child_span,
			input
		)),
		Form::Binary {
			operator,
			left: Some(left)
		} => close(input, operator.apply(left, child, span)),
		Form::Neg => close(
			input,
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(child),
				span
			}))
		),
		Form::Group => close(
			input,
			Expression::Group(Group {
				expression: Box::new(child),
				span
			})
		),
		Form::StandardDice { count: None } => Ok(await_child(
			stack,
			Form::StandardDice { count: Some(child) },
			span,
			child_span,
			input
		)),
		Form::StandardDice { count: Some(count) } => close(
			input,
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: Box::new(count),
				faces: Box::new(child),
				span
			}))
		),
		Form::CustomDice =>
		{
			let (input, faces) = read_faces(input)?;
			close(
				input,
				Expression::Dice(DiceExpression::Custom(CustomDice {
					count: Box::new(child),
					faces,
					span
				}))
			)
		},
		Form::Drop {
			direction,
			after_keyword,
			dice: None
		} =>
		{
			let Some(dice) = into_dice(child)
			else
			{
				return Err(nom::Err::Failure(
					SExprError::ExpectedDiceExpression {
						location: SExprLocation::of(after_keyword)
					}
				))
			};
			// The drop amount is present iff another subexpression follows
			// the dice expression before the closing parenthesis. Its
			// absence represents the parser-originated `drop: None` case
			// (implicit single-die drop), so the format faithfully
			// distinguishes `(drop-lowest d)` from `(drop-lowest d 1)`.
			let (probe, _) = skip_ws(input)?;
			if probe.fragment().starts_with(')')
			{
				close(input, direction.apply(dice, None, span))
			}
			else
			{
				Ok(await_child(
					stack,
					Form::Drop {
						direction,
						after_keyword,
						dice: Some(dice)
					},
					span,
					child_span,
					input
				))
			}
		},
		Form::Drop {
			direction,
			dice: Some(dice),
			..
		} => close(input, direction.apply(dice, Some(child), span)),
		Form::Range { start: None } => Ok(await_child(
			stack,
			Form::Range { start: Some(child) },
			span,
			child_span,
			input
		)),
		Form::Range { start: Some(start) } => close(
			input,
			Expression::Range(Range {
				start: Box::new(start),
				end: Box::new(child),
				span
			})
		),
		Form::Binding { name, name_span } => close(
			input,
			Expression::Binding(Binding {
				name: Cow::Borrowed(name),
				name_span,
				expression: Box::new(child),
				span
			})
		)
	}
}

/// Push a frame that awaits a child, and read the child.
///
/// # Parameters
/// - `stack`: The stack of open forms.
/// - `form`: The form, with the parts of it read so far.
/// - `span`: The span of the form.
/// - `prev`: The span of the previous sibling of the child, or
///   [`SourceSpan::default`] if the child has no previous sibling.
/// - `input`: The input text at which to read the child.
///
/// # Returns
/// The next step, which reads the child.
fn await_child<'src>(
	stack: &mut Vec<Frame<'src>>,
	form: Form<'src>,
	span: SourceSpan,
	prev: SourceSpan,
	input: Span<'src>
) -> Step<'src>
{
	stack.push(Frame {
		form,
		span,
		prev,
		mark: input
	});
	Step::Read(input)
}

/// Close a compound form by reading its `)`, and deliver it.
///
/// # Parameters
/// - `input`: The input text before the `)`.
/// - `expression`: The form.
///
/// # Returns
/// The next step, which delivers the form.
///
/// # Errors
/// - [`SExprError::ExpectedChar`] if the `)` is missing.
fn close<'src>(input: Span<'src>, expression: Expression<'src>)
-> Outcome<'src>
{
	let (input, _) = expect_char(')')(input)?;
	Ok(Step::Deliver(input, expression))
}

/// Extract the dice expression from an expression, if it is one.
///
/// # Parameters
/// - `expression`: The expression.
///
/// # Returns
/// The dice expression, or `None` if the expression is not a dice
/// expression.
fn into_dice(mut expression: Expression<'_>) -> Option<DiceExpression<'_>>
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
			Some(mem::replace(dice, placeholder))
		},
		_ => None
	}
}
