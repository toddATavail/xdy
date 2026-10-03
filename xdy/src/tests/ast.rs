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

/// Ensure that `{:?}` and `{:#?}` render each kind of drop and keep clause
/// exactly as the derived format would: the variant wraps the clause, whose
/// derived [`Debug`] names its own type and its count field, `drop` or `keep`.
#[test]
fn test_debug_clauses_match_derived()
{
	for s_expr in [
		"(function [] (drop-lowest (standard-dice 4 6)))",
		"(function [] (drop-highest (standard-dice 4 6) 2))",
		"(function [] (keep-lowest (standard-dice 4 6)))",
		"(function [] (keep-highest (drop-lowest (standard-dice 4 6)) 3))"
	]
	{
		let body = crate::s_expr::read_s_expr(s_expr).unwrap().body;
		let Expression::Dice(dice) = &body
		else
		{
			unreachable!()
		};
		let (compact, pretty) = match dice
		{
			DiceExpression::DropLowest(clause) => (
				format!("Dice(DropLowest({:?}))", clause),
				format!("Dice(\n    DropLowest(\n{}", indent(clause))
			),
			DiceExpression::DropHighest(clause) => (
				format!("Dice(DropHighest({:?}))", clause),
				format!("Dice(\n    DropHighest(\n{}", indent(clause))
			),
			DiceExpression::KeepLowest(clause) => (
				format!("Dice(KeepLowest({:?}))", clause),
				format!("Dice(\n    KeepLowest(\n{}", indent(clause))
			),
			DiceExpression::KeepHighest(clause) => (
				format!("Dice(KeepHighest({:?}))", clause),
				format!("Dice(\n    KeepHighest(\n{}", indent(clause))
			),
			_ => unreachable!()
		};
		assert_eq!(format!("{:?}", body), compact, "{s_expr}");
		assert_eq!(format!("{:#?}", body), pretty, "{s_expr}");
	}
}

/// Answer the derived pretty [`Debug`] rendering of a clause, indented two
/// levels, and followed by the closing of the two levels that enclose it, as
/// it appears within the pretty rendering of its [`Expression`].
///
/// # Parameters
/// - `clause`: The clause.
///
/// # Returns
/// The rendering.
fn indent(clause: &impl fmt::Debug) -> String
{
	let mut rendering = String::new();
	for line in format!("{:#?}", clause).lines()
	{
		writeln!(rendering, "        {line}").unwrap();
	}
	// The derived format ends each field, including the last, with a comma.
	rendering.pop();
	rendering.push_str(",\n    ),\n)");
	rendering
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
		("4D6 keep lowest", "4D6 drop lowest"),
		("4D6 keep highest", "4D6 keep lowest"),
		("4D6 keep highest", "4D6 keep highest 1"),
		(
			"4D6 drop lowest keep highest 2",
			"4D6 keep highest 2 drop lowest"
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

/// Ensure that a keep clause is distinct from the drop clause of the same
/// shape, and from the keep clause of the other direction, in equality and in
/// hashing, since both compare and hash by the names of their variants.
#[test]
fn test_keeps_are_distinct_from_drops()
{
	let forms = [
		"(drop-lowest (standard-dice 4 6))",
		"(drop-highest (standard-dice 4 6))",
		"(keep-lowest (standard-dice 4 6))",
		"(keep-highest (standard-dice 4 6))",
		"(drop-lowest (standard-dice 4 6) 2)",
		"(drop-highest (standard-dice 4 6) 2)",
		"(keep-lowest (standard-dice 4 6) 2)",
		"(keep-highest (standard-dice 4 6) 2)"
	];
	let sources = forms
		.iter()
		.map(|form| format!("(function [] {form})"))
		.collect::<Vec<_>>();
	let bodies = sources
		.iter()
		.map(|source| crate::s_expr::read_s_expr(source).unwrap().body)
		.collect::<Vec<_>>();
	for (i, left) in bodies.iter().enumerate()
	{
		for (j, right) in bodies.iter().enumerate()
		{
			assert_eq!(left == right, i == j, "{} vs {}", forms[i], forms[j]);
			if i != j
			{
				assert!(
					hash_of(left) != hash_of(right),
					"{} and {} hash alike",
					forms[i],
					forms[j]
				);
			}
		}
	}
}

/// Ensure that keep clauses render as their source, chained with drops in
/// either order, a negative constant count grouped as for drops, that the
/// renderings parse back to the same trees, and that the S-expressions of the
/// trees read back to the same trees.
#[test]
fn test_display_keep_clauses()
{
	for (s_expr, expected) in [
		(
			"(function [] (keep-lowest (standard-dice 4 6)))",
			"4D6 keep lowest"
		),
		(
			"(function [] (keep-highest (custom-dice 3 [-1 0 1]) 2))",
			"3D[-1, 0, 1] keep highest 2"
		),
		(
			"(function [] (keep-highest (drop-lowest (standard-dice 4 6) 1) 2))",
			"4D6 drop lowest 1 keep highest 2"
		),
		(
			"(function [] (drop-highest (keep-lowest (standard-dice 4 6) 3)))",
			"4D6 keep lowest 3 drop highest"
		),
		(
			"(function [] (keep-lowest (keep-highest (standard-dice 5 10) 3) \
			 -1))",
			"5D10 keep highest 3 keep lowest (-1)"
		),
		(
			"(function [{n}] (keep-highest (standard-dice {n} 6) (group (sub \
			 {n} 1))))",
			"{n}: {n}D6 keep highest ({n} - 1)"
		)
	]
	{
		let function = crate::s_expr::read_s_expr(s_expr).unwrap();
		let options = SExpressibleOptions::default().with_groups(true);
		assert_eq!(function.to_string(), expected);
		assert_eq!(function.to_s_expr(options), s_expr);
		// A reparsed negative count is grouped, so compare without groups.
		let reparsed = Parser::parse(expected).unwrap();
		assert_eq!(
			reparsed.to_s_expr(SExpressibleOptions::default()),
			function.to_s_expr(SExpressibleOptions::default())
		);
	}
}

/// Ensure that a negative constant drop count, which only a tree that was not
/// parsed can hold, renders grouped, since a drop expression never begins with
/// `-`, so that the rendering reads back as the same drop.
#[test]
fn test_display_groups_negative_drop_count()
{
	for (s_expr, expected) in [
		(
			"(function [] (drop-lowest (standard-dice 4 6) -1))",
			"4D6 drop lowest (-1)"
		),
		(
			"(function [] (drop-highest (drop-lowest (standard-dice 4 6) 1) -2))",
			"4D6 drop lowest 1 drop highest (-2)"
		)
	]
	{
		let function = crate::s_expr::read_s_expr(s_expr).unwrap();
		let source = function.to_string();
		assert_eq!(source, expected);
		let reparsed = Parser::parse(&source).unwrap();
		assert_eq!(reparsed.to_s_expr(SExpressibleOptions::default()), s_expr);
	}
}

/// Ensure that separately parsed but identical sources hash identically.
#[test]
fn test_hash_agrees_with_equality()
{
	let source = "{x}: 8D8 drop lowest (2) keep highest 5 drop highest \
	              {a}@({x}) keep lowest + [1:6] ^ -(3)";
	let first = Parser::parse(source).unwrap();
	let second = Parser::parse(source).unwrap();
	assert!(first == second);
	assert_eq!(hash_of(&first), hash_of(&second));
	assert_eq!(hash_of(&first.body), hash_of(&second.body));
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// The greatest depth of the deep-nesting tests. Each [nesting](Nesting) nests
/// to its own [depth](Nesting::depth), at most this deep. Some of the tests
/// need gigabytes of memory at this depth, so every test that builds a source
/// this deep is ignored by default; `just stress` runs them.
const DEPTH: usize = 1_000_000;

/// The size, in bytes, of the [rendered](fmt::Display) source of a deep
/// nesting, to which each [nesting](Nesting) scales its
/// [depth](Nesting::depth): that of [`DEPTH`] additions, each of which renders
/// as ` + 1`. The work of a deep test, whether it parses, diagnoses, reads, or
/// walks, grows with the size of the source rather than with its depth alone,
/// so a nesting whose levels render larger, such as `1D6 drop lowest (e)`,
/// would otherwise do several times the work of the others.
const SOURCE_BUDGET: usize = 4 * DEPTH;

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

/// Ensure that a deep chain of keep expressions survives every trait on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_keep_expression()
{
	exercise_on_small_stack(Nesting::KeepExpression)
}

/// Ensure that a deep stack of drop and keep clauses, of every kind, survives
/// every trait on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_deep_mixed_clauses() { exercise_on_small_stack(Nesting::MixedClauses) }

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

/// Ensure that every trait works on a chain of drop and keep clauses of every
/// kind.
#[test]
fn test_miri_mixed_clauses()
{
	exercise(|leaf| nest(Nesting::MixedClauses, 40, leaf))
}

/// Ensure that a standalone chain of drop and keep clauses drops, both as a
/// [`DiceExpression`] and as a bare clause, whichever kind of clause tops it.
#[test]
fn test_miri_standalone_mixed_clauses()
{
	for depth in 37..=40
	{
		let chain = mixed_clauses(standard_dice(constant(1)), depth);
		drop(chain.clone());
		match &chain
		{
			DiceExpression::DropLowest(clause) => drop(clause.clone()),
			DiceExpression::DropHighest(clause) => drop(clause.clone()),
			DiceExpression::KeepLowest(clause) => drop(clause.clone()),
			DiceExpression::KeepHighest(clause) => drop(clause.clone()),
			_ => unreachable!()
		}
		drop(chain);
	}
}

/// Ensure that a keep clause directly atop a leaf dice expression drops.
#[test]
fn test_miri_single_keep_clause()
{
	for s_expr in [
		"(function [{x}] (keep-lowest (custom-dice {x} [1 2]) (standard-dice \
		 2 6)))",
		"(function [{x}] (keep-highest (standard-dice {x} 6)))"
	]
	{
		let expression = crate::s_expr::read_s_expr(s_expr).unwrap().body;
		drop(expression.clone());
		drop(expression);
	}
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

	/// `1D6 keep highest (e)`.
	KeepExpression,

	/// `1D6 drop lowest (e) keep highest`: a stack of drop and keep clauses.
	/// Nested alone, this produces a single chain of [clauses of every
	/// kind](mixed_clauses) whose leaf count is `e`.
	MixedClauses,

	/// Every other construct, in rotation.
	Mixed
}

impl Nesting
{
	/// The constructs through which [`Mixed`](Self::Mixed) rotates.
	pub(super) const ROTATION: [Nesting; 18] = [
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
		Nesting::DropClauses,
		Nesting::KeepExpression,
		Nesting::MixedClauses
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

	/// Answer the depth to which the deep-nesting tests nest this way: as many
	/// levels as fit in [`SOURCE_BUDGET`], at most [`DEPTH`]. The size of a
	/// level is measured from the [rendering](fmt::Display) of
	/// [`nest_parsable`], over one period of [`ROTATION`](Self::ROTATION),
	/// beyond the first, so that it averages the constructs of
	/// [`Mixed`](Self::Mixed) and excludes the grouping of the innermost
	/// levels.
	///
	/// # Returns
	/// The depth.
	pub(super) fn depth(self) -> usize
	{
		let period = Self::ROTATION.len();
		let size = |depth| nest_parsable(self, depth, 1).to_string().len();
		let bytes = size(2 * period) - size(period);
		(SOURCE_BUDGET * period / bytes).min(DEPTH)
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
	match nesting
	{
		Nesting::DropClauses =>
		{
			return Expression::Dice(drop_clauses(
				standard_dice(constant(leaf)),
				depth
			))
		},
		Nesting::MixedClauses =>
		{
			return Expression::Dice(mixed_clauses(
				standard_dice(constant(leaf)),
				depth
			))
		},
		_ =>
		{}
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
	if let Nesting::DropClauses | Nesting::MixedClauses = nesting
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
			| Nesting::DropClauses
			| Nesting::KeepExpression
			| Nesting::MixedClauses => !atomic,
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
		Nesting::KeepExpression =>
		{
			Expression::Dice(DiceExpression::KeepHighest(KeepHighest {
				dice: Box::new(standard_dice(constant(1))),
				keep: Some(inner),
				span
			}))
		},
		Nesting::MixedClauses =>
		{
			Expression::Dice(DiceExpression::KeepHighest(KeepHighest {
				dice: Box::new(DiceExpression::DropLowest(DropLowest {
					dice: Box::new(standard_dice(constant(1))),
					drop: Some(inner),
					span
				})),
				keep: None,
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

/// Stack `depth` drop and keep clauses atop a dice expression, rotating
/// through drop-lowest, keep-highest, drop-highest, and keep-lowest, with a
/// count on every third clause.
///
/// # Parameters
/// - `dice`: The dice expression at the bottom of the stack.
/// - `depth`: The number of clauses.
///
/// # Returns
/// The stack of clauses.
fn mixed_clauses(
	dice: DiceExpression<'static>,
	depth: usize
) -> DiceExpression<'static>
{
	let span = SourceSpan::SYNTHETIC;
	let mut dice = dice;
	for level in 0..depth
	{
		let count = (level % 3 == 0).then(|| Box::new(constant(1)));
		let inner = Box::new(dice);
		dice = match level % 4
		{
			0 => DiceExpression::DropLowest(DropLowest {
				dice: inner,
				drop: count,
				span
			}),
			1 => DiceExpression::KeepHighest(KeepHighest {
				dice: inner,
				keep: count,
				span
			}),
			2 => DiceExpression::DropHighest(DropHighest {
				dice: inner,
				drop: count,
				span
			}),
			_ => DiceExpression::KeepLowest(KeepLowest {
				dice: inner,
				keep: count,
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

/// Build a pair of [deep](Nesting::depth) expressions on a small stack, and
/// [exercise](exercise) every trait on them.
///
/// # Parameters
/// - `nesting`: The way to nest.
fn exercise_on_small_stack(nesting: Nesting)
{
	let depth = nesting.depth();
	on_small_stack(|| exercise(|leaf| nest(nesting, depth, leaf)));
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
