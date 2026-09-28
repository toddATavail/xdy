//! # Optimizer tests
//!
//! Herein are tests for the optimizer. The actual test cases are stored in
//! `../../tests`, variously among:
//!
//! * `test_common_subexpression_elimination.txt`
//! * `test_constant_commuting.txt`
//! * `test_constant_folding.txt`
//! * `test_full_optimization.txt`
//! * `test_register_coalescence.txt`
//! * `test_strength_reduction.txt`
//!
//! Each file comprises a series of test cases, each of which consists of a
//! source dice expression and an expected print rendition.

use std::collections::HashSet;

use pretty_assertions::assert_eq;

use rand::{SeedableRng, rngs::StdRng};

use crate::{
	Evaluator, Pass, Passes,
	support::{compile_valid, optimize, read_compilation_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                             Other test cases.                              //
////////////////////////////////////////////////////////////////////////////////

/// Test that the common subexpression eliminator generates the expected output
/// for the test cases.
#[test]
fn test_common_subexpression_elimination()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_common_subexpression_elimination.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(
			function.clone(),
			Pass::CommonSubexpressionElimination.into()
		);
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant commuter generates the expected output for the test
/// cases.
#[test]
fn test_constant_commuting()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_constant_commuting.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantCommuting.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant folder generates the expected output for the test
/// cases.
#[test]
fn test_constant_folding()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_constant_folding.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantFolding.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the strength reducer generates the expected output for the test
/// cases.
#[test]
fn test_strength_reduction()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_strength_reduction.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::StrengthReduction.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the register coalescer generates the expected output for the test
/// cases.
#[test]
fn test_register_coalescence()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_register_coalescence.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::RegisterCoalescing.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the fully-loaded optimizer produces the expected output for the
/// test cases.
#[test]
fn test_full_optimization()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_full_optimization.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(function.clone(), Passes::all());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the optimizer folds a constant [maximum](crate::Max), which only
/// the optimizer emits, applies its identities, and gathers the constants of a
/// chain of maxima so that they fold.
#[test]
fn test_optimize_max()
{
	for (body, expected) in [
		// Constant folding.
		("@1 <- 3 max 5\n\t\treturn @1", "return 5"),
		// The least value is the identity.
		("@1 <- @0 max -2147483648\n\t\treturn @1", "return @0"),
		// The greatest value absorbs.
		(
			"@1 <- 2147483647 max @0\n\t\treturn @1",
			"return 2147483647"
		),
		// The maximum is idempotent.
		("@1 <- @0 max @0\n\t\treturn @1", "return @0"),
		// Commutation gathers the constants of a chain, which then fold.
		(
			"@1 <- @0 max 3\n\t\t@2 <- 5 max @1\n\t\treturn @2",
			"@0 <- 5 max @0\n\t\treturn @0"
		)
	]
	{
		let registers = body.matches("<-").count() + 1;
		let text = format!(
			"Function({{x}}@0) r#{} ⚅#0\n\textern[]\n\tbody:\n\t\t{}\n",
			registers, body
		);
		let function = crate::Assembler::assemble(&text).unwrap();
		let optimized = optimize(function, Passes::all()).to_string();
		let actual = optimized.trim().split_once("body:\n\t\t").unwrap().1;
		assert_eq!(actual, expected, "{}", body);
	}
}

/// Test that register coalescing never lets a write that nothing reads clobber
/// a register that is live across it (xdy-i0q.30).
#[test]
fn test_coalescing_keeps_dead_writes_apart()
{
	let function = crate::Assembler::assemble(
		"\
Function({x}@0) r#4 ⚅#0
\textern[]
\tbody:
\t\t@1 <- @0 + 1
\t\t@2 <- @0 + 2
\t\t@3 <- @1 + @0
\t\treturn @3
"
	)
	.unwrap();
	let coalesced = optimize(function.clone(), Pass::RegisterCoalescing.into());
	let evaluate = |function| {
		Evaluator::new(function)
			.evaluate([10], &mut StdRng::seed_from_u64(0))
			.unwrap()
			.result
	};
	assert_eq!(evaluate(coalesced.clone()), 21, "{}", coalesced);
	assert_eq!(evaluate(function), 21);
}

////////////////////////////////////////////////////////////////////////////////
//                                 Exactness.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Values of arguments that exercise the edges of the arithmetic: zero, the
/// units, and the extremes of [`i32`], where arithmetic saturates.
const EDGES: [i32; 9] = [
	i32::MIN,
	i32::MIN + 1,
	-65536,
	-1,
	0,
	1,
	65536,
	i32::MAX - 1,
	i32::MAX
];

/// The greatest number of combinations of [edge](EDGES) values over which
/// [`assert_exact`] evaluates a function. Functions with many variables take
/// their values from a prefix of the edges, so that the combinations stay
/// within it.
const COMBINATIONS: usize = 10_000;

/// Assert that optimizing the specified dice-free function with the specified
/// passes never changes its result, for every combination of [edge](EDGES)
/// values of its formal parameters and external variables.
///
/// # Parameters
/// - `source`: The source of the function, which rolls no dice.
/// - `passes`: The passes to apply.
fn assert_exact(source: &str, passes: Passes)
{
	let unoptimized = compile_valid(source);
	let optimized = optimize(unoptimized.clone(), passes);
	let arity = unoptimized.arity();
	let variables = arity + unoptimized.externals.len();
	// Take as many edges as keep the combinations within bounds.
	let mut edges = EDGES.len();
	while edges > 2 && edges.pow(variables as u32) > COMBINATIONS
	{
		edges -= 1;
	}
	let edges = &EDGES[EDGES.len() - edges..];
	for mut combination in 0..edges.len().pow(variables as u32)
	{
		let values = (0..variables)
			.map(|_| {
				let value = edges[combination % edges.len()];
				combination /= edges.len();
				value
			})
			.collect::<Vec<_>>();
		let evaluate = |function| {
			let mut evaluator = Evaluator::new(function);
			for (name, value) in
				unoptimized.externals.iter().zip(&values[arity..])
			{
				evaluator.bind(name, *value).unwrap();
			}
			evaluator
				.evaluate(
					values[..arity].iter().copied(),
					&mut StdRng::seed_from_u64(0)
				)
				.unwrap()
				.result
		};
		assert_eq!(
			evaluate(optimized.clone()),
			evaluate(unoptimized.clone()),
			"{} at {:?}\noptimized:\n{}",
			source,
			values,
			optimized
		);
	}
}

/// Test that no pass changes the result of any dice-free function of the
/// optimizer's test cases, at the edges of the arithmetic, alone or together
/// with the others.
#[test]
fn test_test_cases_are_exact()
{
	let files = [
		(
			include_str!(
				"../../tests/test_common_subexpression_elimination.txt"
			),
			Passes::from(Pass::CommonSubexpressionElimination)
		),
		(
			include_str!("../../tests/test_constant_commuting.txt"),
			Pass::ConstantCommuting.into()
		),
		(
			include_str!("../../tests/test_constant_folding.txt"),
			Pass::ConstantFolding.into()
		),
		(
			include_str!("../../tests/test_strength_reduction.txt"),
			Pass::StrengthReduction.into()
		),
		(
			include_str!("../../tests/test_register_coalescence.txt"),
			Pass::RegisterCoalescing.into()
		),
		(
			include_str!("../../tests/test_full_optimization.txt"),
			Passes::all()
		)
	];
	for (file, passes) in files
	{
		for (source, _) in read_compilation_test_cases(file)
		{
			if compile_valid(source).rolling_record_count == 0
			{
				assert_exact(source, passes);
				assert_exact(source, Passes::all());
			}
		}
	}
}

/// Test that strength reduction never changes a result: a value divided by
/// itself is zero when the value is zero, and a double negation of
/// [`i32::MIN`] is `-i32::MAX`, so neither reduces (xdy-i0q.26). The last
/// source nests a double negation around another instruction, which once
/// tripped an assertion.
#[test]
fn test_strength_reduction_is_exact()
{
	for source in [
		"{x}: {x} / {x}",
		"{x}: --{x}",
		"{x}: ---{x}",
		"{x}: --{x} + 2",
		"{x}, {y}: {a}@(-{x}) + {y} * 2 + -{a}",
		"{x}: {x} * -1",
		"{x}: {x} / -1",
		"{x}: 0 - {x}",
		"{x}: {x} * 2",
		"{x}: {x} ^ 2",
		"{x}: {x} % 1",
		"{x}: {x} ^ 0",
		"{x}: 1 ^ {x}"
	]
	{
		assert_exact(source, Pass::StrengthReduction.into());
		assert_exact(source, Passes::all());
	}
}
