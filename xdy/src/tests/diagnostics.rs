//! # Diagnostic test cases
//!
//! Herein are the data-driven test cases for the diagnostics module. The actual
//! test cases are stored in `../../tests/test_errors.txt`, which comprises a
//! series of broken source expressions and their expected diagnostics.

use std::collections::HashSet;

use pretty_assertions::assert_eq;

use crate::{
	diagnostics::{self, DiagnosticKind},
	parser,
	support::read_error_test_cases
};

////////////////////////////////////////////////////////////////////////////////
//                           Diagnostic test cases.                           //
////////////////////////////////////////////////////////////////////////////////

/// Map a [`DiagnosticKind`] to its test-file name.
fn kind_name(kind: &DiagnosticKind) -> &'static str
{
	match kind
	{
		DiagnosticKind::UnclosedDelimiter { .. } => "UnclosedDelimiter",
		DiagnosticKind::UnopenedDelimiter { .. } => "UnopenedDelimiter",
		DiagnosticKind::MissingRightOperand { .. } => "MissingRightOperand",
		DiagnosticKind::MissingLeftOperand { .. } => "MissingLeftOperand",
		DiagnosticKind::BareIdentifier => "BareIdentifier",
		DiagnosticKind::MissingDiceFaces => "MissingDiceFaces",
		DiagnosticKind::IncompleteDropClause => "IncompleteDropClause",
		DiagnosticKind::IncompleteParameterDefinition =>
		{
			"IncompleteParameterDefinition"
		},
		DiagnosticKind::TrailingInput => "TrailingInput",
		DiagnosticKind::EmptyExpression => "EmptyExpression",
		DiagnosticKind::UnexpectedToken => "UnexpectedToken",
		DiagnosticKind::UnexpectedEof => "UnexpectedEof"
	}
}

/// Test that the diagnostics module produces the expected diagnostics for each
/// test case in `test_errors.txt`.
#[test]
fn test_error_diagnostics()
{
	let test_cases =
		read_error_test_cases(include_str!("../../tests/test_errors.txt"));
	assert!(
		!test_cases.is_empty(),
		"no test cases found in test_errors.txt"
	);

	let mut seen = HashSet::new();
	for (index, case) in test_cases.iter().enumerate()
	{
		assert!(
			seen.insert(case.source),
			"duplicate test case: {:?}",
			case.source
		);

		let result = diagnostics::diagnose(case.source);

		// Check diagnostic count.
		assert_eq!(
			result.diagnostics.len(),
			case.expected_diagnostics.len(),
			"case {}: {:?} — expected {} diagnostics, got {}: {:?}",
			index + 1,
			case.source,
			case.expected_diagnostics.len(),
			result.diagnostics.len(),
			result
				.diagnostics
				.iter()
				.map(|d| format!("{}", d))
				.collect::<Vec<_>>()
		);

		// Check each diagnostic.
		for (di, (actual, expected)) in result
			.diagnostics
			.iter()
			.zip(case.expected_diagnostics.iter())
			.enumerate()
		{
			// Check kind.
			assert_eq!(
				kind_name(&actual.kind),
				expected.kind,
				"case {}: {:?} — diagnostic {} kind mismatch",
				index + 1,
				case.source,
				di + 1
			);

			// Check span.
			assert_eq!(
				(actual.span.start, actual.span.end),
				expected.span,
				"case {}: {:?} — diagnostic {} span mismatch",
				index + 1,
				case.source,
				di + 1
			);

			// Check message.
			assert_eq!(
				actual.message,
				expected.message,
				"case {}: {:?} — diagnostic {} message mismatch",
				index + 1,
				case.source,
				di + 1
			);

			// Check suggestions.
			assert_eq!(
				actual.suggestions.len(),
				expected.suggestions.len(),
				"case {}: {:?} — diagnostic {} suggestion count \
				 mismatch: expected {}, got {}",
				index + 1,
				case.source,
				di + 1,
				expected.suggestions.len(),
				actual.suggestions.len()
			);

			for (si, expected_suggestion) in
				expected.suggestions.iter().enumerate()
			{
				let suggestion = &actual.suggestions[si];

				// Check corrected source.
				assert_eq!(
					suggestion.corrected_source,
					expected_suggestion.corrected_source,
					"case {}: {:?} — diagnostic {} suggestion {} \
					 corrected_source mismatch",
					index + 1,
					case.source,
					di + 1,
					si + 1
				);

				// Check that the corrected source parses cleanly
				// if it's a single-diagnostic case.
				if case.expected_diagnostics.len() == 1
				{
					assert!(
						parser::parse(&suggestion.corrected_source).is_ok(),
						"case {}: {:?} — suggestion {} {:?} does \
						 not parse cleanly",
						index + 1,
						case.source,
						si + 1,
						suggestion.corrected_source
					);
				}

				// Check placeholders.
				assert_eq!(
					suggestion.placeholders.len(),
					expected_suggestion.placeholders.len(),
					"case {}: {:?} — diagnostic {} suggestion {} \
					 placeholder count mismatch",
					index + 1,
					case.source,
					di + 1,
					si + 1
				);

				for (pi, (actual_ph, expected_ph)) in suggestion
					.placeholders
					.iter()
					.zip(expected_suggestion.placeholders.iter())
					.enumerate()
				{
					assert_eq!(
						(actual_ph.span.start, actual_ph.span.end),
						expected_ph.span,
						"case {}: {:?} — diagnostic {} suggestion \
						 {} placeholder {} span mismatch",
						index + 1,
						case.source,
						di + 1,
						si + 1,
						pi + 1
					);
					assert_eq!(
						actual_ph.description,
						expected_ph.description,
						"case {}: {:?} — diagnostic {} suggestion \
						 {} placeholder {} description mismatch",
						index + 1,
						case.source,
						di + 1,
						si + 1,
						pi + 1
					);
					assert_eq!(
						actual_ph.valid_kinds.to_vec(),
						expected_ph.valid_kinds,
						"case {}: {:?} — diagnostic {} suggestion \
						 {} placeholder {} valid_kinds mismatch",
						index + 1,
						case.source,
						di + 1,
						si + 1,
						pi + 1
					);
				}
			}
		}
	}
}

/// Test that all corrected sources produced by the doctor parse cleanly.
#[test]
fn test_corrected_sources_parse()
{
	let test_cases =
		read_error_test_cases(include_str!("../../tests/test_errors.txt"));
	for case in &test_cases
	{
		let result = diagnostics::diagnose(case.source);
		if let Some(corrected) = &result.corrected_source
		{
			assert!(
				parser::parse(corrected).is_ok(),
				"corrected source for {:?} does not parse: {:?}",
				case.source,
				corrected
			);
		}
	}
}

/// Test that valid programs produce zero diagnostics and an unchanged
/// corrected source.
#[test]
fn test_valid_programs_produce_no_diagnostics()
{
	let valid = vec![
		"0",
		"42",
		"-1",
		"3D6",
		"3d6",
		"1D20 + 5",
		"2D8 - 1D4",
		"3 * 4 + 2",
		"2 ^ 10",
		"10 % 3",
		"(1D6 + 2) * 3",
		"((1 + 2))",
		"{x}",
		"{x}D6",
		"{x}D{y}",
		"x: {x}D6",
		"x, y: {x} + {y}",
		"a, b, c: ({a} + {b}) * {c}",
		"[1:20]",
		"[1:6]",
		"2D[1,2,3]",
		"4D6 drop lowest",
		"4D6 drop highest",
		"4D6 drop lowest 1",
		"8D6 drop lowest 3 drop highest 1",
		"1D6 + 1D8 + 1D10",
	];
	for source in &valid
	{
		let result = diagnostics::diagnose(source);
		assert!(
			result.diagnostics.is_empty(),
			"valid program {:?} produced {} diagnostics: {:?}",
			source,
			result.diagnostics.len(),
			result
				.diagnostics
				.iter()
				.map(|d| format!("{}", d))
				.collect::<Vec<_>>()
		);
		assert_eq!(
			result.corrected_source.as_deref(),
			Some(*source),
			"valid program {:?} corrected_source mismatch",
			source
		);
	}
}

/// Test that the diagnostics pipeline is fast enough for keystroke-speed
/// invocation.
#[test]
fn test_diagnostics_performance()
{
	let expressions = vec![
		"3D6 + 2",
		"1D20",
		"4D6 drop lowest",
		"2D8 + 1D4 - 3",
		"{x}D{y}",
		"x: {x}D6 + {x}",
		"(1D6 + 2) * 3",
		"[1:20]",
		"1D6 + 1D8 + 1D10",
		"a, b: {a}D{b} + 5",
		// Invalid expressions.
		"3D6 +",
		"(3D6",
		"xD6",
		"3D",
		"4D6 drop",
		"+ 1D6",
		"",
		"3D6)",
		"1 + * 2",
		"3D6 + + 1D3 -",
	];
	let start = std::time::Instant::now();
	for _ in 0..100
	{
		for expr in &expressions
		{
			let _ = diagnostics::diagnose(expr);
		}
	}
	let elapsed = start.elapsed();
	// 2000 diagnose() calls should complete well within 1 second.
	assert!(
		elapsed.as_millis() < 1000,
		"diagnostics too slow: {}ms for 2000 calls",
		elapsed.as_millis()
	);
}
