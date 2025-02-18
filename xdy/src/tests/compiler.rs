//! # Compiler tests
//!
//! Herein are tests for the compiler, covering vanilla code generation only.
//! The actual test cases are stored in
//! `../../tests/test_compile_unoptimized.txt`, which comprises a series of
//! test cases, each of which consists of a source dice expression and an
//! expected print rendition.

use std::collections::HashSet;

use pretty_assertions::assert_eq;

use crate::support::{compile_valid, read_compilation_test_cases};

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
