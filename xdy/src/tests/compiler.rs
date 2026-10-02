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

use super::{
	ast::{Nesting, nest_function},
	recovery::SCALE
};
use crate::{
	Compiler, Passes, compile, compile_unoptimized,
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
/// chain. Wall-clock time is noisy under a busy machine, so the
/// test is ignored by default; `just stress` runs it.
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
