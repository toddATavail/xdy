//! # Compiler tests
//!
//! Herein are tests for the compiler, covering vanilla code generation only.
//! The actual test cases are stored in
//! `../../tests/test_compile_unoptimized.txt`, which comprises a series of
//! test cases, each of which consists of a source dice expression and an
//! expected print rendition. The optimizing pipeline, [`compile`], is checked
//! against the fully-loaded optimizer's test cases in
//! `../../tests/test_full_optimization.txt`.

use std::{
	collections::HashSet,
	time::{Duration, Instant}
};

use pretty_assertions::{assert_eq, assert_ne};
use rand::{SeedableRng, rngs::StdRng};

use super::{
	ast::{Nesting, nest_function},
	recovery::SCALE
};
use crate::{
	Compiler, Evaluator, Passes, SourceSpan, Validator,
	ast::{self, Constant, Direction, Expression, Extremum},
	compile, compile_unoptimized,
	support::{
		compile_valid, on_small_stack, on_small_stack_within, optimize,
		read_compilation_test_cases
	}
};

////////////////////////////////////////////////////////////////////////////////
//                           Code generation tests.                           //
////////////////////////////////////////////////////////////////////////////////

/// Test that the compiler generates the expected output for the test cases.
#[test]
fn test_compile_unoptimized()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_compile_unoptimized.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let actual = format!("{}", compile_valid(source));
		assert_eq!(actual.trim(), *expected, "case {}: {}", index + 1, source);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Optimizing pipeline.                             //
////////////////////////////////////////////////////////////////////////////////

/// Test that [`compile`] applies every optimization pass, so that it agrees
/// with the fully-loaded optimizer on every one of its test cases.
#[test]
fn test_compile_optimizes_fully()
{
	for (index, (source, _)) in read_compilation_test_cases(include_str!(
		"../../tests/test_full_optimization.txt"
	))
	.iter()
	.enumerate()
	{
		let expected = optimize(compile_valid(source), Passes::all());
		let actual = compile(source).unwrap();
		assert_eq!(actual, expected, "case {}: {}", index + 1, source);
	}
}

/// Test that [`compile`] folds constant arithmetic that
/// [`compile_unoptimized`] leaves alone.
#[test]
fn test_compile_folds_constants()
{
	let source = "(2 + 3) * 1D6 + 4 * 2";
	let optimized = compile(source).unwrap();
	let unoptimized = compile_unoptimized(source).unwrap();
	assert_ne!(optimized, unoptimized);
	assert_eq!(optimized.instructions.len(), 5, "{}", optimized);
	let rendition = optimized.to_string();
	assert!(!rendition.contains("2 + 3"), "{}", rendition);
	assert!(!rendition.contains("4 * 2"), "{}", rendition);
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the compiler emits a keep instruction for each keep clause, in
/// order with the drops, with a count of 1 when the clause gives none.
#[test]
fn test_compile_keep_clauses()
{
	for (s_expr, expected) in [
		(
			"(function [] (keep-lowest (standard-dice 3 6)))",
			"\
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅0 <- keep lowest 1 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [] (keep-highest (drop-lowest (standard-dice 4 6)) 2))",
			"\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- keep highest 2 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [{n}] (drop-highest (keep-lowest (custom-dice 4 [1 3 5]) \
			 {n})))",
			"\
\t\t⚅0 <- roll custom dice 4D[1, 3, 5]
\t\t⚅0 <- keep lowest @0 from ⚅0
\t\t⚅0 <- drop highest 1 from ⚅0
\t\t@1 <- sum rolling record ⚅0
\t\treturn @1
"
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		let function = Compiler::compile(&ast).to_string();
		assert_eq!(function.split_once("body:\n").unwrap().1, expected);
	}
}

/// Ensure that the compiler emits a reroll instruction for each reroll clause,
/// in order with the drops and keeps, with a limit of 1 when the clause gives
/// none; that it computes the operands of the predicate and the limit after
/// the dice that they reroll, in the order written, and before the reroll, each
/// once; and that it discovers the external variables among them.
#[test]
fn test_compile_reroll_clauses()
{
	for (s_expr, expected) in [
		(
			"(function [] (reroll (standard-dice 3 6) (eq 1)))",
			"\
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅0 <- reroll = 1 up to 1 times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [] (reroll (standard-dice 3 6) (le (standard-dice 1 4)) \
			 (unlimited)))",
			"\
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅1 <- roll standard dice 1D4
\t\t@0 <- sum rolling record ⚅1
\t\t⚅0 <- reroll <= @0 unlimited times from ⚅0
\t\t@1 <- sum rolling record ⚅0
\t\treturn @1
"
		),
		(
			"(function [{n}] (reroll (custom-dice 4 [-1 0 1]) (in-range \
			 (neg 1) (group (add {n} 1))) (times {n})))",
			"\
\t\t⚅0 <- roll custom dice 4D[-1, 0, 1]
\t\t@1 <- @0 + 1
\t\t⚅0 <- reroll in -1..@1 up to @0 times from ⚅0
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
"
		),
		(
			"(function [] (reroll (standard-dice 2 6) (in-range-inclusive {x} \
			 {y}) (times 2)))",
			"\
\t\t⚅0 <- roll standard dice 2D6
\t\t⚅0 <- reroll in @0..=@1 up to 2 times from ⚅0
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
"
		),
		(
			"(function [] (keep-highest (reroll (drop-lowest (standard-dice 4 6)) \
			 (in [1 2])) 3))",
			"\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- reroll in [1, 2] up to 1 times from ⚅0
\t\t⚅0 <- keep highest 3 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [] (reroll (reroll (standard-dice 4 6) (ne 6)) (in []) \
			 (unlimited)))",
			"\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- reroll != 6 up to 1 times from ⚅0
\t\t⚅0 <- reroll in [] unlimited times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		let function = Compiler::compile(&ast).to_string();
		assert_eq!(function.split_once("body:\n").unwrap().1, expected);
	}
	let ast = crate::s_expr::read_s_expr(
		"(function [] (reroll (standard-dice 2 6) (in-range-inclusive {x} {y}) \
		 (times 2)))"
	)
	.unwrap();
	let function = Compiler::compile(&ast);
	assert_eq!(
		function.to_string().lines().nth(1).map(str::trim),
		Some("extern[{x}@0, {y}@1]")
	);
}

/// Ensure that the compiler emits an explode instruction for each explode
/// clause, in order with the drops, keeps, and rerolls, with a limit of 1 when
/// the clause gives none; that it computes the operands of the predicate and
/// the limit after the dice that they explode, in the order written, and before
/// the explode, each once; and that it discovers the external variables among
/// them.
#[test]
fn test_compile_explode_clauses()
{
	for (s_expr, expected) in [
		(
			"(function [] (explode (standard-dice 3 6) (eq 6)))",
			"\
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅0 <- explode = 6 up to 1 times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [] (explode (standard-dice 3 6) (ge (standard-dice 1 4)) \
			 (unlimited)))",
			"\
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅1 <- roll standard dice 1D4
\t\t@0 <- sum rolling record ⚅1
\t\t⚅0 <- explode >= @0 unlimited times from ⚅0
\t\t@1 <- sum rolling record ⚅0
\t\treturn @1
"
		),
		(
			"(function [{n}] (explode (custom-dice 4 [-1 0 1]) (in-range \
			 (neg 1) (group (add {n} 1))) (times {n})))",
			"\
\t\t⚅0 <- roll custom dice 4D[-1, 0, 1]
\t\t@1 <- @0 + 1
\t\t⚅0 <- explode in -1..@1 up to @0 times from ⚅0
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
"
		),
		(
			"(function [] (explode (standard-dice 2 6) (in-range-inclusive {x} \
			 {y}) (times 2)))",
			"\
\t\t⚅0 <- roll standard dice 2D6
\t\t⚅0 <- explode in @0..=@1 up to 2 times from ⚅0
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
"
		),
		(
			"(function [] (keep-highest (explode (drop-lowest (standard-dice 5 \
			 10)) (in [9 10])) 3))",
			"\
\t\t⚅0 <- roll standard dice 5D10
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- explode in [9, 10] up to 1 times from ⚅0
\t\t⚅0 <- keep highest 3 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		),
		(
			"(function [] (explode (reroll (standard-dice 4 6) (eq 1)) (in []) \
			 (unlimited)))",
			"\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- reroll = 1 up to 1 times from ⚅0
\t\t⚅0 <- explode in [] unlimited times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		let function = Compiler::compile(&ast).to_string();
		assert_eq!(function.split_once("body:\n").unwrap().1, expected);
	}
	let ast = crate::s_expr::read_s_expr(
		"(function [] (explode (standard-dice 2 6) (in-range-inclusive {x} {y}) \
		 (times 2)))"
	)
	.unwrap();
	let function = Compiler::compile(&ast);
	assert_eq!(
		function.to_string().lines().nth(1).map(str::trim),
		Some("extern[{x}@0, {y}@1]")
	);
}

/// Ensure that the functions compiled from explode clauses evaluate as their
/// trees describe, and to the same results after optimization, which may
/// rewrite the records of dice of one face: an explode extends each
/// matching die once unless it says otherwise, and every extension may itself
/// explode, up to the limit; later drops and keeps count each chain as one die,
/// worth its total; a later reroll acts on each die of a chain; an operand
/// rolled from dice is rolled once, after the dice that explode; and an
/// unlimited explode that matches every face saturates each chain toward the
/// sign of its mean face value, or to `0` if that mean is `0`.
#[test]
fn test_evaluate_explode_clauses()
{
	for (s_expr, result, records) in [
		(
			"(function [] (explode (custom-dice 2 [3]) (eq 3)))",
			12,
			&["2d[3]: [3, 3, 3, 3]"][..]
		),
		(
			"(function [] (explode (custom-dice 2 [3]) (eq 3) (times 2)))",
			18,
			&["2d[3]: [3, 3, 3, 3, 3, 3]"][..]
		),
		(
			"(function [] (explode (custom-dice 2 [3]) (ne 3) (unlimited)))",
			6,
			&["2d[3]: [3, 3]"][..]
		),
		(
			"(function [] (keep-lowest (explode (custom-dice 3 [2]) (eq 2) \
			 (times 1)) 2))",
			8,
			&["3d[2]: [~2, 2, 2, ~2, 2, 2]"][..]
		),
		(
			"(function [] (reroll (explode (custom-dice 1 [6]) (eq 6)) (eq 6)))",
			12,
			&["1d[6]: [~6, ~6, 6, 6]"][..]
		),
		(
			"(function [] (explode (custom-dice 1 [5]) (in-range-inclusive \
			 (custom-dice 1 [4]) 6) (times 3)))",
			20,
			&["1d[5]: [5, 5, 5, 5]", "1d[4]: [4]"][..]
		),
		(
			"(function [] (explode (custom-dice 2 [3]) (ge 1) (unlimited)))",
			i32::MAX,
			&["2d[3]: [3, 3, ∞, ∞]"][..]
		),
		(
			"(function [] (explode (custom-dice 2 [0]) (eq 0) (unlimited)))",
			0,
			&["2d[0]: [0, 0, 0, 0]"][..]
		),
		(
			"(function [] (explode (custom-dice 2 [-2]) (eq -2) (unlimited)))",
			i32::MIN,
			&["2d[-2]: [-2, -2, -∞, -∞]"][..]
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		Validator::validate(&ast).unwrap();
		let function = Compiler::compile(&ast);
		let evaluate = |function| {
			Evaluator::new(function)
				.evaluate([], &mut StdRng::seed_from_u64(0))
				.unwrap()
		};
		let evaluation = evaluate(function.clone());
		assert_eq!(evaluation.result, result, "{}", s_expr);
		assert_eq!(
			evaluation
				.records
				.iter()
				.map(ToString::to_string)
				.collect::<Vec<_>>(),
			records,
			"{}",
			s_expr
		);
		let optimized = evaluate(optimize(function, Passes::all()));
		assert_eq!(optimized.result, result, "{} optimized", s_expr);
	}
}

/// Ensure that the compiler computes the arguments of an extremum in the order
/// written, each once, and then chains `max` or `min` over their values from
/// left to right; that it discovers the external variables among them in the
/// order written; and that the functions that it compiles evaluate to the
/// greatest or the least of the arguments, with every roll in the records, in
/// the order written.
#[test]
fn test_compile_extrema()
{
	for (s_expr, expected) in [
		(
			"(function [] (highest (standard-dice 1 20) (standard-dice 1 20) \
			 (standard-dice 1 20)))",
			"\
\t\t⚅0 <- roll standard dice 1D20
\t\t@0 <- sum rolling record ⚅0
\t\t⚅1 <- roll standard dice 1D20
\t\t@1 <- sum rolling record ⚅1
\t\t⚅2 <- roll standard dice 1D20
\t\t@2 <- sum rolling record ⚅2
\t\t@3 <- @0 max @1
\t\t@4 <- @3 max @2
\t\treturn @4
"
		),
		(
			"(function [{POW}] (lowest (mul {POW} 5) 99))",
			"\
\t\t@1 <- @0 * 5
\t\t@2 <- @1 min 99
\t\treturn @2
"
		),
		(
			"(function [] (highest 3 5))",
			"\
\t\t@0 <- 3 max 5
\t\treturn @0
"
		),
		(
			"(function [] (lowest {y} (highest {x} {y}) {z}))",
			"\
\t\t@3 <- @1 max @0
\t\t@4 <- @0 min @3
\t\t@5 <- @4 min @2
\t\treturn @5
"
		),
		(
			"(function [] (drop-lowest (standard-dice 4 6) (group (lowest {x} \
			 2))))",
			"\
\t\t⚅0 <- roll standard dice 4D6
\t\t@1 <- @0 min 2
\t\t⚅0 <- drop lowest @1 from ⚅0
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
"
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		let function = Compiler::compile(&ast).to_string();
		assert_eq!(function.split_once("body:\n").unwrap().1, expected);
	}
	let ast = crate::s_expr::read_s_expr(
		"(function [] (lowest {y} (highest {x} {y}) {z}))"
	)
	.unwrap();
	assert_eq!(
		Compiler::compile(&ast)
			.to_string()
			.lines()
			.nth(1)
			.map(str::trim),
		Some("extern[{y}@0, {x}@1, {z}@2]")
	);
	for (s_expr, result, records) in [
		("(function [] (highest 3 5 4))", 5, &[][..]),
		("(function [] (lowest 3 5 4))", 3, &[][..]),
		(
			"(function [] (highest (custom-dice 1 [4]) (custom-dice 2 [-1]) \
			 (custom-dice 1 [9])))",
			9,
			&["1d[4]: [4]", "2d[-1]: [-1, -1]", "1d[9]: [9]"][..]
		),
		(
			"(function [] (lowest (custom-dice 1 [4]) (custom-dice 2 [-1]) \
			 (custom-dice 1 [9])))",
			-2,
			&["1d[4]: [4]", "2d[-1]: [-1, -1]", "1d[9]: [9]"][..]
		)
	]
	{
		let ast = crate::s_expr::read_s_expr(s_expr).unwrap();
		let mut evaluator = Evaluator::new(Compiler::compile(&ast));
		let evaluation = evaluator
			.evaluate([], &mut StdRng::seed_from_u64(0))
			.unwrap();
		assert_eq!(evaluation.result, result, "{}", s_expr);
		assert_eq!(
			evaluation
				.records
				.iter()
				.map(ToString::to_string)
				.collect::<Vec<_>>(),
			records,
			"{}",
			s_expr
		);
	}
}

/// Ensure that the compiler accepts an extremum built with fewer than two
/// arguments, which neither the parser nor the S-expression reader admits: an
/// extremum of one argument has the value of that argument, and an extremum of
/// none has the identity of its direction.
#[test]
fn test_compile_extrema_of_fewer_than_two()
{
	let span = SourceSpan::SYNTHETIC;
	for (direction, arguments, expected) in [
		(Direction::Highest, vec![7], 7),
		(Direction::Lowest, vec![7], 7),
		(Direction::Highest, vec![], i32::MIN),
		(Direction::Lowest, vec![], i32::MAX)
	]
	{
		let ast = ast::Function {
			parameters: None,
			body: Expression::Extremum(Extremum {
				direction,
				arguments: arguments
					.into_iter()
					.map(|value| Expression::Constant(Constant { value, span }))
					.collect(),
				span
			}),
			span
		};
		Validator::validate(&ast).unwrap();
		let function = Compiler::compile(&ast);
		assert_eq!(
			function.to_string().split_once("body:\n").unwrap().1,
			format!("\t\treturn {}\n", expected)
		);
		let mut evaluator = Evaluator::new(function);
		let evaluation = evaluator
			.evaluate([], &mut StdRng::seed_from_u64(0))
			.unwrap();
		assert_eq!(evaluation.result, expected);
	}
}

/// Ensure that the compiler survives a deep chain of every nesting construct
/// on a small stack, and still discovers the external variable that follows
/// the chain.
#[test]
#[ignore = "stress: run with just stress"]
fn test_compile_deep()
{
	on_small_stack(|| {
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			let function =
				Compiler::compile(&nest_function(nesting, nesting.depth()));
			assert_eq!(function.externals, ["x"], "{:?}", nesting);
		}
	});
}

/// The factor by which [`test_optimize_deep`] shortens the chain of each
/// nesting from its [depth](Nesting::depth). Any recursion of the optimizer
/// that grew with the depth would overflow the small stack long before the
/// shortened depth, yet the chains stay short enough to optimize promptly in a
/// debug build; [`test_compile_is_linear`] bounds the time at depth.
const OPTIMIZATION_DIVISOR: usize = 10;

/// The time budget of [`test_optimize_deep`], which optimizes a long function
/// for every nesting construct in a debug build.
const OPTIMIZATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Ensure that the optimizer survives the function compiled from a deep chain
/// of every nesting construct on a small stack, and still retains the external
/// variable that follows the chain.
#[test]
#[ignore = "stress: run with just stress"]
fn test_optimize_deep()
{
	on_small_stack_within(OPTIMIZATION_TIMEOUT, || {
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			let depth = nesting.depth() / OPTIMIZATION_DIVISOR;
			let function = Compiler::compile(&nest_function(nesting, depth));
			let function = optimize(function, Passes::all());
			assert_eq!(function.externals, ["x"], "{:?}", nesting);
		}
	});
}

////////////////////////////////////////////////////////////////////////////////
//                                 Linearity.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The depth of the shallower member of each family in the linearity test.
const WIDTH: usize = 2_000;

/// The number of times that the linearity test compiles each member, keeping
/// the fastest time, so that a stall of the machine does not fail the test.
const RUNS: usize = 3;

/// The greatest factor by which the time of a compilation may grow when the
/// depth grows by [`SCALE`]. It leaves room for the noise of the machine and
/// the cost of touching more memory, but quadratic time would grow by
/// [`SCALE`] times more.
const TIME_SCALE: u32 = 3 * SCALE as u32;

/// The least time against which the linearity test measures the growth of the
/// time of a compilation.
const TIME_FLOOR: Duration = Duration::from_micros(100);

/// A builder of the member of a family of inputs with the given depth.
type Member = fn(usize) -> String;

/// The families of the linearity test: their name, and the builder of their
/// member of the given depth. Some fold to a constant, and some cannot fold,
/// since they reach a variable or a roll, so that the optimizer keeps every
/// instruction.
const FAMILIES: &[(&str, Member)] = &[
	("negations of a constant", |n| format!("{}1", "-".repeat(n))),
	("negations of a variable", |n| {
		format!("{}{{x}}", "-".repeat(n))
	}),
	("nested ranges", |n| {
		format!("{}1{}", "[1:".repeat(n), "]".repeat(n))
	}),
	("nested ranges of a variable", |n| {
		format!("{}{{x}}{}", "[1:".repeat(n), "]".repeat(n))
	}),
	("nested powers", |n| format!("{}2", "2^".repeat(n))),
	("nested groups", |n| {
		format!("{}1{}", "(".repeat(n), ")".repeat(n))
	}),
	("nested dice counts", |n| {
		format!("{}1{}", "(".repeat(n), ")D6".repeat(n))
	}),
	("sums of constants", |n| format!("1{}", " + 1".repeat(n))),
	("sums of a variable", |n| {
		format!("{{x}}{}", " + {x} - 1".repeat(n))
	}),
	("products of rolls", |n| {
		format!("1D6{}", " * 1D6".repeat(n))
	}),
	("differences of divisions", |n| {
		format!("{{x}}{}", " / 2 - {x} % 3".repeat(n))
	}),
	("stacked drop clauses", |n| {
		format!("{}D6{}", n + 1, " drop lowest 1".repeat(n))
	})
];

/// Ensure that [`compile`], which optimizes fully, takes time linear in the
/// depth of its input, for chains that fold away and chains that cannot. The
/// optimizer's passes each once took time quadratic in the length of such a
/// chain. Wall-clock time is noisy under a busy machine, so the test is
/// ignored by default; `just stress` runs it alone, after the other stress
/// tests, whose memory traffic would slow its larger inputs more than its
/// smaller ones.
#[test]
#[ignore = "stress: run with just stress"]
fn test_compile_is_linear()
{
	for (name, source) in FAMILIES
	{
		let sources = [WIDTH, SCALE * WIDTH].map(source);
		let mut times = [Duration::MAX; 2];
		for _ in 0..RUNS
		{
			for (i, source) in sources.iter().enumerate()
			{
				let start = Instant::now();
				let result = compile(source);
				times[i] = times[i].min(start.elapsed());
				assert!(result.is_ok(), "{}: {:?}", name, result.err());
			}
		}
		assert!(
			times[1] <= TIME_SCALE * times[0].max(TIME_FLOOR),
			"{} took {:?}, then {:?} for {} times the depth",
			name,
			times[0],
			times[1],
			SCALE
		);
	}
}
