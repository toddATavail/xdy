//! # Data-driven parser test cases
//!
//! Herein are the data-driven parser tests, which exercise the parser against
//! the curated corpus in `tests/test_parse.txt` — verifying both the parser's
//! AST output and the round-trip behaviour of the S-expression reader and
//! writer.

use crate::parser::*;
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                         Data-driven parser tests.                          //
////////////////////////////////////////////////////////////////////////////////

/// Test that the parser produces the expected AST for every test case in
/// `test_parse.txt`. Each case is parsed, the resulting AST is serialized to an
/// S-expression, and compared against the expected S-expression.
#[test]
fn test_parse_to_s_expr()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions};
	use crate::support::read_compilation_test_cases;
	use std::collections::HashSet;

	let test_cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	assert!(
		!test_cases.is_empty(),
		"no test cases found in test_parse.txt"
	);

	let opts = SExpressibleOptions::default();
	let mut seen = HashSet::new();
	for (index, (source, expected)) in test_cases.iter().enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let ast = match Parser::parse(source)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {}: parse failed for {:?}: {}",
				index + 1,
				source,
				e
			)
		};
		let actual = ast.to_s_expr(opts);
		assert_eq!(actual.trim(), *expected, "case {}: {}", index + 1, source);
	}
}

/// Test that the S-expression reader and writer roundtrip perfectly: for every
/// expected S-expression in `test_parse.txt`, reading it back and writing it
/// again produces the identical string.
#[test]
fn test_s_expr_roundtrip()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions, read_s_expr};
	use crate::support::read_compilation_test_cases;

	let test_cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	let opts = SExpressibleOptions::default();
	for (index, (source, expected)) in test_cases.iter().enumerate()
	{
		let ast = match read_s_expr(expected)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {}: S-expr read failed for {:?} (source {:?}): {}",
				index + 1,
				expected,
				source,
				e
			)
		};
		let roundtripped = ast.to_s_expr(opts);
		assert_eq!(
			roundtripped.trim(),
			*expected,
			"case {}: roundtrip mismatch for {:?}",
			index + 1,
			source
		);
	}
}
