//! # AST trait tests
//!
//! Herein are tests for the standard trait implementations of the recursive
//! abstract syntax tree (AST) types — [`Debug`], [`Display`](fmt::Display),
//! [`Clone`], [`PartialEq`], [`Hash`], and [`Drop`] — and for
//! [`untethered`](Spanned::untethered) and [`SExpressible`], all of which must
//! traverse with explicit stacks rather than recursion. The golden files
//! `tests/test_ast_debug.txt` and `tests/test_ast_debug_pretty.txt` were
//! captured from the derived implementations that these replaced, so they pin
//! the [`Debug`] output to the derived format.

use std::{
	collections::hash_map::DefaultHasher,
	fmt::{self, Write},
	hash::{Hash, Hasher}
};

use pretty_assertions::assert_eq;

use crate::{
	Parser, SourceSpan, Spanned,
	ast::*,
	s_expr::{SExpressible, SExpressibleOptions},
	support::{on_small_stack, read_compilation_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                               Debug parity.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that `{:?}` matches the derived format for every case in
/// `test_ast_debug.txt`.
#[test]
fn test_debug_matches_derived()
{
	let test_cases = read_compilation_test_cases(include_str!(
		"../../tests/test_ast_debug.txt"
	));
	assert!(!test_cases.is_empty());
	for (source, expected) in test_cases
	{
		let function = Parser::parse(source).unwrap();
		assert_eq!(format!("{:?}", function), expected, "{}", source);
	}
}

/// Ensure that `{:#?}` matches the derived format for every case in
/// `test_ast_debug_pretty.txt`.
#[test]
fn test_debug_pretty_matches_derived()
{
	let test_cases = read_compilation_test_cases(include_str!(
		"../../tests/test_ast_debug_pretty.txt"
	));
	assert!(!test_cases.is_empty());
	for (source, expected) in test_cases
	{
		let function = Parser::parse(source).unwrap();
		assert_eq!(format!("{:#?}", function), expected, "{}", source);
	}
}

/// Ensure that `{:?}` passes the formatter's flags through to the leaves, as
/// the derived format does. The expectation was captured from the derived
/// implementation.
#[test]
fn test_debug_passes_flags_to_leaves()
{
	let function = Parser::parse("{x}: {x}D[-1, 10] drop highest 12").unwrap();
	assert_eq!(
		format!("{:x?}", function.body),
		"Dice(DropHighest(DropHighest { dice: Custom(CustomDice { count: \
		 Variable(Variable { name: \"x\", span: SourceSpan { start: 5, end: \
		 8 } }), faces: [ffffffff, a], span: SourceSpan { start: 5, end: 11 \
		 } }), drop: Some(Constant(Constant { value: c, span: SourceSpan { \
		 start: 1f, end: 21 } })), span: SourceSpan { start: 5, end: 21 } }))"
	);
}

/// Ensure that `{:#?}` renders a deep negation chain exactly as the derived
/// format would, as reconstructed independently by [`pretty_negations`].
#[test]
fn test_debug_pretty_deep()
{
	on_small_stack(|| {
		let depth = 500;
		let expression = nest(Nesting::Negation, depth, 1);
		let actual = format!("{:#?}", expression);
		assert!(actual == pretty_negations(depth, 1), "mismatch");
	});
}

////////////////////////////////////////////////////////////////////////////////
//                                 Semantics.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a clone equals its original, hashes identically, and renders
/// identically.
#[test]
fn test_clone_equals_original()
{
	let test_cases = read_compilation_test_cases(include_str!(
		"../../tests/test_ast_debug.txt"
	));
	for (source, _) in test_cases
	{
		let function = Parser::parse(source).unwrap();
		let body = function.body.clone();
		assert!(body == function.body, "{}", source);
		assert_eq!(hash_of(&body), hash_of(&function.body), "{}", source);
		assert_eq!(
			format!("{:#?}", body),
			format!("{:#?}", function.body),
			"{}",
			source
		);
	}
}

/// Ensure that equality notices every kind of difference: variants, leaf
/// values, spans, names, faces, and the presence of a drop expression.
#[test]
fn test_equality_distinguishes()
{
	let pairs = [
		("1 + 2", "1 + 3"),
		("1 + 2", "1 - 2"),
		("1 + 2", "1  + 2"),
		("(1)", "{a}@(1)"),
		("{a}@(1)", "{b}@(1)"),
		("[1:2]", "[1:3]"),
		("3D[1, 2]", "3D[1, 3]"),
		("4D6 drop lowest", "4D6 drop lowest 1"),
		("4D6 drop lowest", "4D6 drop highest"),
		(
			"4D6 drop lowest drop lowest",
			"4D6 drop lowest drop highest"
		),
		("{x}: {x}", "{y}: {y}"),
		("-(1)", "-(2)")
	];
	for (left, right) in pairs
	{
		let left_body = Parser::parse(left).unwrap().body;
		let right_body = Parser::parse(right).unwrap().body;
		assert!(left_body != right_body, "{} == {}", left, right);
		assert!(left_body == left_body.clone(), "{}", left);
	}
}

/// Ensure that separately parsed but identical sources hash identically.
#[test]
fn test_hash_agrees_with_equality()
{
	let source =
		"{x}: 8D8 drop lowest (2) drop highest {a}@({x}) + [1:6] ^ -(3)";
	let first = Parser::parse(source).unwrap();
	let second = Parser::parse(source).unwrap();
	assert!(first == second);
	assert_eq!(hash_of(&first), hash_of(&second));
	assert_eq!(hash_of(&first.body), hash_of(&second.body));
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// The depth of the deep-nesting tests. Some of them need gigabytes of memory
/// at this depth, so every test that builds a source this deep is ignored by
/// default; `just stress` runs them.
pub(super) const DEPTH: usize = 1_000_000;

/// The value of the innermost constant of a deep source. Its rendering occurs
/// nowhere else in the source, so that a test can find it and replace it.
pub(super) const LEAF: i32 = 7_777_777;

/// Ensure that a deep chain of groups survives every trait on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_group() { exercise_on_small_stack(Nesting::Group) }

/// Ensure that a deep chain of bindings survives every trait on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_binding() { exercise_on_small_stack(Nesting::Binding) }

/// Ensure that a deep chain of range starts survives every trait on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_range_start() { exercise_on_small_stack(Nesting::RangeStart) }

/// Ensure that a deep chain of range ends survives every trait on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_range_end() { exercise_on_small_stack(Nesting::RangeEnd) }

/// Ensure that a deep chain of negations survives every trait on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_negation() { exercise_on_small_stack(Nesting::Negation) }

/// Ensure that a deep right-nested chain of exponents survives every trait on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_exponent() { exercise_on_small_stack(Nesting::Exponent) }

/// Ensure that a deep left-nested chain of additions survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_addition() { exercise_on_small_stack(Nesting::Addition) }

/// Ensure that a deep left-nested chain of subtractions survives every trait
/// on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_subtraction() { exercise_on_small_stack(Nesting::Subtraction) }

/// Ensure that a deep left-nested chain of multiplications survives every
/// trait on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_multiplication()
{
	exercise_on_small_stack(Nesting::Multiplication)
}

/// Ensure that a deep left-nested chain of divisions survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_division() { exercise_on_small_stack(Nesting::Division) }

/// Ensure that a deep left-nested chain of modulos survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_modulo() { exercise_on_small_stack(Nesting::Modulo) }

/// Ensure that a deep chain of dice counts survives every trait on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_dice_count() { exercise_on_small_stack(Nesting::DiceCount) }

/// Ensure that a deep chain of standard dice faces survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_dice_faces() { exercise_on_small_stack(Nesting::DiceFaces) }

/// Ensure that a deep chain of custom dice counts survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_custom_count() { exercise_on_small_stack(Nesting::CustomCount) }

/// Ensure that a deep chain of drop expressions survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_drop_expression()
{
	exercise_on_small_stack(Nesting::DropExpression)
}

/// Ensure that a deep chain of stacked drop clauses survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_drop_clauses() { exercise_on_small_stack(Nesting::DropClauses) }

/// Ensure that a deep interleaving of every nesting construct survives every
/// trait on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_mixed() { exercise_on_small_stack(Nesting::Mixed) }

////////////////////////////////////////////////////////////////////////////////
//                                   Miri.                                    //
////////////////////////////////////////////////////////////////////////////////

// These tests are small enough to run under Miri, which checks the `unsafe`
// code in `Drop` for undefined behavior and leaks:
//
// ```sh
// cargo +nightly miri test -p xdy --lib tests::ast::test_miri
// MIRIFLAGS=-Zmiri-tree-borrows \
//     cargo +nightly miri test -p xdy --lib tests::ast::test_miri
// ```

/// Ensure that every trait works on a shallow interleaving of every nesting
/// construct.
#[test]
fn test_miri_mixed() { exercise(|leaf| nest(Nesting::Mixed, 60, leaf)) }

/// Ensure that every trait works on a chain of stacked drop clauses.
#[test]
fn test_miri_drop_clauses()
{
	exercise(|leaf| nest(Nesting::DropClauses, 40, leaf))
}

/// Ensure that a standalone chain of stacked drop clauses drops, both as a
/// [`DiceExpression`] and as a bare [`DropLowest`].
#[test]
fn test_miri_standalone_drop_clauses()
{
	let chain = drop_clauses(standard_dice(constant(1)), 40);
	drop(chain.clone());
	match &chain
	{
		DiceExpression::DropLowest(clause) => drop(clause.clone()),
		DiceExpression::DropHighest(clause) => drop(clause.clone()),
		_ => unreachable!()
	}
	drop(chain);
}

/// Ensure that a drop clause directly atop a leaf dice expression drops.
#[test]
fn test_miri_single_drop_clause()
{
	let expression = Parser::parse("{x}: {x}D[1, 2] drop lowest (2D6)")
		.unwrap()
		.body;
	drop(expression.clone());
	drop(expression);
}

////////////////////////////////////////////////////////////////////////////////
//                                  Helpers.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A way to nest one expression inside another.
#[derive(Copy, Clone, Debug)]
pub(super) enum Nesting
{
	/// `(e)`.
	Group,

	/// `{a}@(e)`.
	Binding,

	/// `[e:1]`.
	RangeStart,

	/// `[1:e]`.
	RangeEnd,

	/// `-e`.
	Negation,

	/// `2 ^ e`.
	Exponent,

	/// `e + 1`.
	Addition,

	/// `e - 1`.
	Subtraction,

	/// `e * 1`.
	Multiplication,

	/// `e / 1`.
	Division,

	/// `e % 1`.
	Modulo,

	/// `(e)D6`.
	DiceCount,

	/// `1D(e)`.
	DiceFaces,

	/// `(e)D[1, 2]`.
	CustomCount,

	/// `1D6 drop lowest (e)`.
	DropExpression,

	/// `1D6 drop lowest (e) drop highest`: a stack of drop clauses. Nested
	/// alone, this produces a single chain of drop clauses whose leaf count is
	/// `e`.
	DropClauses,

	/// Every other construct, in rotation.
	Mixed
}

impl Nesting
{
	/// The constructs through which [`Mixed`](Self::Mixed) rotates.
	pub(super) const ROTATION: [Nesting; 16] = [
		Nesting::Group,
		Nesting::Binding,
		Nesting::RangeStart,
		Nesting::RangeEnd,
		Nesting::Negation,
		Nesting::Exponent,
		Nesting::Addition,
		Nesting::Subtraction,
		Nesting::Multiplication,
		Nesting::Division,
		Nesting::Modulo,
		Nesting::DiceCount,
		Nesting::DiceFaces,
		Nesting::CustomCount,
		Nesting::DropExpression,
		Nesting::DropClauses
	];

	/// Answer whether nesting this way introduces [bindings](Binding). Every
	/// such binding is named `a`, so a deep nesting binds `a` inside itself.
	///
	/// # Returns
	/// `true` for [`Binding`](Self::Binding) and [`Mixed`](Self::Mixed).
	pub(super) const fn binds(self) -> bool
	{
		matches!(self, Nesting::Binding | Nesting::Mixed)
	}
}

/// Build an expression by nesting a constant `depth` times.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `depth`: The number of levels of nesting.
/// - `leaf`: The value of the innermost constant.
///
/// # Returns
/// The expression.
pub(super) fn nest(
	nesting: Nesting,
	depth: usize,
	leaf: i32
) -> Expression<'static>
{
	if let Nesting::DropClauses = nesting
	{
		return Expression::Dice(drop_clauses(
			standard_dice(constant(leaf)),
			depth
		))
	}
	let mut expression = constant(leaf);
	for level in 0..depth
	{
		let nesting = match nesting
		{
			Nesting::Mixed =>
			{
				Nesting::ROTATION[level % Nesting::ROTATION.len()]
			},
			nesting => nesting
		};
		expression = wrap(nesting, expression);
	}
	expression
}

/// Build a function without parameters whose body adds a reference to the
/// external variable `x` to a constant nested `depth` times. A left-to-right
/// walk must therefore traverse the whole nesting before it reaches the
/// reference.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `depth`: The number of levels of nesting.
///
/// # Returns
/// The function.
pub(super) fn nest_function(nesting: Nesting, depth: usize)
-> Function<'static>
{
	let span = SourceSpan::SYNTHETIC;
	Function {
		parameters: None,
		body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
			left: Box::new(nest(nesting, depth, 1)),
			right: Box::new(Expression::Variable(Variable {
				name: "x".into(),
				span
			})),
			span
		})),
		span
	}
}

/// Build an expression by nesting a constant `depth` times, as [`nest`] does,
/// but in a form that the parser can produce, so that its
/// [rendering](fmt::Display) parses back to the same expression. [`nest`]
/// places dice and arithmetic directly where the grammar admits only a
/// constant, a variable, a binding, or a group, e.g., as a dice count, and
/// negates a positive constant, which renders as a negative constant. This
/// function wraps the inner expression in a [`Group`] wherever the grammar
/// requires one.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `depth`: The number of levels of nesting.
/// - `leaf`: The value of the innermost constant.
///
/// # Returns
/// The expression.
pub(super) fn nest_parsable(
	nesting: Nesting,
	depth: usize,
	leaf: i32
) -> Expression<'static>
{
	if let Nesting::DropClauses = nesting
	{
		return nest(nesting, depth, leaf)
	}
	let mut expression = constant(leaf);
	for level in 0..depth
	{
		let nesting = match nesting
		{
			Nesting::Mixed =>
			{
				Nesting::ROTATION[level % Nesting::ROTATION.len()]
			},
			nesting => nesting
		};
		let atomic = matches!(
			expression,
			Expression::Constant(_)
				| Expression::Variable(_)
				| Expression::Binding(_)
				| Expression::Group(_)
		);
		let needs_group = match nesting
		{
			Nesting::DiceCount
			| Nesting::DiceFaces
			| Nesting::CustomCount
			| Nesting::DropExpression
			| Nesting::DropClauses => !atomic,
			Nesting::Negation => matches!(expression, Expression::Constant(_)),
			Nesting::Multiplication | Nesting::Division | Nesting::Modulo =>
			{
				matches!(
					expression,
					Expression::Arithmetic(
						ArithmeticExpression::Add(_)
							| ArithmeticExpression::Sub(_)
					)
				)
			},
			Nesting::Exponent => matches!(
				expression,
				Expression::Arithmetic(
					ArithmeticExpression::Add(_)
						| ArithmeticExpression::Sub(_)
						| ArithmeticExpression::Mul(_)
						| ArithmeticExpression::Div(_)
						| ArithmeticExpression::Mod(_)
				)
			),
			_ => false
		};
		if needs_group
		{
			expression = Expression::Group(Group {
				expression: Box::new(expression),
				span: SourceSpan::SYNTHETIC
			});
		}
		expression = wrap(nesting, expression);
	}
	expression
}

/// Wrap an expression in one level of nesting.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `inner`: The expression to wrap.
///
/// # Returns
/// The wrapped expression.
fn wrap(nesting: Nesting, inner: Expression<'static>) -> Expression<'static>
{
	let span = SourceSpan::SYNTHETIC;
	let inner = Box::new(inner);
	match nesting
	{
		Nesting::Group => Expression::Group(Group {
			expression: inner,
			span
		}),
		Nesting::Binding => Expression::Binding(Binding {
			name: "a".into(),
			name_span: span,
			expression: inner,
			span
		}),
		Nesting::RangeStart => Expression::Range(Range {
			start: inner,
			end: Box::new(constant(1)),
			span
		}),
		Nesting::RangeEnd => Expression::Range(Range {
			start: Box::new(constant(1)),
			end: inner,
			span
		}),
		Nesting::Negation =>
		{
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: inner,
				span
			}))
		},
		Nesting::Exponent =>
		{
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(constant(2)),
				right: inner,
				span
			}))
		},
		Nesting::Addition =>
		{
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: inner,
				right: Box::new(constant(1)),
				span
			}))
		},
		Nesting::Subtraction =>
		{
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: inner,
				right: Box::new(constant(1)),
				span
			}))
		},
		Nesting::Multiplication =>
		{
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: inner,
				right: Box::new(constant(1)),
				span
			}))
		},
		Nesting::Division =>
		{
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: inner,
				right: Box::new(constant(1)),
				span
			}))
		},
		Nesting::Modulo =>
		{
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: inner,
				right: Box::new(constant(1)),
				span
			}))
		},
		Nesting::DiceCount =>
		{
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: inner,
				faces: Box::new(constant(6)),
				span
			}))
		},
		Nesting::DiceFaces =>
		{
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: Box::new(constant(1)),
				faces: inner,
				span
			}))
		},
		Nesting::CustomCount =>
		{
			Expression::Dice(DiceExpression::Custom(CustomDice {
				count: inner,
				faces: vec![1, 2],
				span
			}))
		},
		Nesting::DropExpression =>
		{
			Expression::Dice(DiceExpression::DropLowest(DropLowest {
				dice: Box::new(standard_dice(constant(1))),
				drop: Some(inner),
				span
			}))
		},
		Nesting::DropClauses =>
		{
			Expression::Dice(DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::DropLowest(DropLowest {
					dice: Box::new(standard_dice(constant(1))),
					drop: Some(inner),
					span
				})),
				drop: None,
				span
			}))
		},
		Nesting::Mixed => unreachable!()
	}
}

/// Stack `depth` drop clauses atop a dice expression, alternating between
/// drop-lowest and drop-highest, with a drop expression on every third clause.
///
/// # Parameters
/// - `dice`: The dice expression at the bottom of the stack.
/// - `depth`: The number of drop clauses.
///
/// # Returns
/// The stack of drop clauses.
fn drop_clauses(
	dice: DiceExpression<'static>,
	depth: usize
) -> DiceExpression<'static>
{
	let span = SourceSpan::SYNTHETIC;
	let mut dice = dice;
	for level in 0..depth
	{
		let drop = (level % 3 == 0).then(|| Box::new(constant(1)));
		let inner = Box::new(dice);
		dice = if level % 2 == 0
		{
			DiceExpression::DropLowest(DropLowest {
				dice: inner,
				drop,
				span
			})
		}
		else
		{
			DiceExpression::DropHighest(DropHighest {
				dice: inner,
				drop,
				span
			})
		};
	}
	dice
}

/// Answer a constant with a synthetic span.
///
/// # Parameters
/// - `value`: The value of the constant.
///
/// # Returns
/// The constant.
fn constant(value: i32) -> Expression<'static>
{
	Expression::Constant(Constant {
		value,
		span: SourceSpan::SYNTHETIC
	})
}

/// Answer a standard dice expression with six faces.
///
/// # Parameters
/// - `count`: The number of dice.
///
/// # Returns
/// The dice expression.
fn standard_dice(count: Expression<'static>) -> DiceExpression<'static>
{
	DiceExpression::Standard(StandardDice {
		count: Box::new(count),
		faces: Box::new(constant(6)),
		span: SourceSpan::SYNTHETIC
	})
}

/// Build a pair of `DEPTH`-deep expressions on a small stack, and
/// [exercise](exercise) every trait on them.
///
/// # Parameters
/// - `nesting`: The way to nest.
fn exercise_on_small_stack(nesting: Nesting)
{
	on_small_stack(|| exercise(|leaf| nest(nesting, DEPTH, leaf)));
}

/// Exercise every trait on an expression: equality against a copy that
/// differs only in its innermost leaf, [`Clone`], equality and hashing against
/// the clone, compact [`Debug`] and [`Display`](fmt::Display) rendering,
/// [`untethered`](Spanned::untethered), S-expression sizing and writing, and
/// [`Drop`]. Uses `assert!` rather than `assert_eq!` on the expressions
/// themselves, since a failure would otherwise try to render them in full.
///
/// The S-expressions are written with an unbounded soft limit, so that they
/// fit on one line, and their length must then equal their size. Wrapped
/// output would be quadratic in the depth, since each level indents by a tab,
/// so [`test_read_deep_wrapped`](super::parser::s_expr_deep) checks wrapping at
/// a more modest depth instead.
///
/// # Parameters
/// - `build`: Builds the expression, given the value of its innermost leaf.
fn exercise(build: impl Fn(i32) -> Expression<'static>)
{
	let original = build(1);
	let different = build(2);
	assert!(original != different, "differing leaves compared equal");
	drop(different);
	let copy = original.clone();
	assert!(original == copy, "clone compared unequal");
	assert_eq!(hash_of(&original), hash_of(&copy));
	let length = debug_length(&original);
	assert!(length > 0);
	assert_eq!(length, debug_length(&copy));
	let length = display_length(&original);
	assert!(length > 0);
	assert_eq!(length, display_length(&copy));
	// Every span of the tree is synthetic, i.e., the default, so untethering
	// must reproduce it exactly.
	assert!(
		original.untethered() == original,
		"untethering changed spans"
	);
	for with_groups in [false, true]
	{
		let options =
			SExpressibleOptions::new(0, 4, usize::MAX).with_groups(with_groups);
		let size = original.size_s_expr(options);
		assert!(size > 0);
		assert_eq!(s_expr_length(&original, options), size);
	}
}

/// Hash a value with the default hasher.
///
/// # Parameters
/// - `value`: The value to hash.
///
/// # Returns
/// The hash.
fn hash_of(value: &impl Hash) -> u64
{
	let mut hasher = DefaultHasher::new();
	value.hash(&mut hasher);
	hasher.finish()
}

/// A sink that counts the bytes written to it.
struct Counter(usize);

impl Write for Counter
{
	fn write_str(&mut self, s: &str) -> fmt::Result
	{
		self.0 += s.len();
		Ok(())
	}
}

/// Measure the compact [`Debug`] rendering of a value without storing it.
///
/// # Parameters
/// - `value`: The value to render.
///
/// # Returns
/// The length of the rendering, in bytes.
fn debug_length(value: &impl fmt::Debug) -> usize
{
	let mut counter = Counter(0);
	write!(counter, "{:?}", value).unwrap();
	counter.0
}

/// Measure the [`Display`](fmt::Display) rendering of a value without storing
/// it.
///
/// # Parameters
/// - `value`: The value to render.
///
/// # Returns
/// The length of the rendering, in bytes.
fn display_length(value: &impl fmt::Display) -> usize
{
	let mut counter = Counter(0);
	write!(counter, "{}", value).unwrap();
	counter.0
}

/// Measure the S-expression representation of a value without storing it.
///
/// # Parameters
/// - `value`: The value to write.
/// - `options`: The formatting options.
///
/// # Returns
/// The length of the representation, in bytes.
fn s_expr_length(
	value: &impl SExpressible,
	options: SExpressibleOptions
) -> usize
{
	let mut counter = Counter(0);
	value
		.write_s_expr(&mut counter, options.soft_limit, options)
		.unwrap();
	counter.0
}

/// Reconstruct the derived `{:#?}` rendering of `depth` negations of a
/// constant, all with synthetic spans, without recursion.
///
/// # Parameters
/// - `depth`: The number of negations.
/// - `leaf`: The value of the constant.
///
/// # Returns
/// The rendering.
fn pretty_negations(depth: usize, leaf: i32) -> String
{
	let indent = |level: usize| "    ".repeat(level);
	let span = |level: usize| {
		format!(
			"SourceSpan {{\n{}start: 0,\n{}end: 0,\n{}}}",
			indent(level + 1),
			indent(level + 1),
			indent(level)
		)
	};
	let mut out = String::new();
	for level in 0..depth
	{
		let base = 3 * level;
		out.push_str("Arithmetic(\n");
		out.push_str(&format!("{}Neg(\n", indent(base + 1)));
		out.push_str(&format!("{}Neg {{\n", indent(base + 2)));
		out.push_str(&format!("{}operand: ", indent(base + 3)));
	}
	let base = 3 * depth;
	out.push_str(&format!(
		"Constant(\n{}Constant {{\n{}value: {},\n{}span: {},\n{}}},\n{})",
		indent(base + 1),
		indent(base + 2),
		leaf,
		indent(base + 2),
		span(base + 2),
		indent(base + 1),
		indent(base)
	));
	for level in (0..depth).rev()
	{
		let base = 3 * level;
		out.push_str(&format!(
			",\n{}span: {},\n{}}},\n{}),\n{})",
			indent(base + 3),
			span(base + 3),
			indent(base + 2),
			indent(base + 1),
			indent(base)
		));
	}
	out
}
