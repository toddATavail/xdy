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

use crate::{
	Pass, Passes,
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
