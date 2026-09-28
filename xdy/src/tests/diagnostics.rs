//! # Diagnostic test cases
//!
//! Herein are the data-driven test cases for the diagnostics module. The actual
//! test cases are stored in `../../tests/test_parser_errors.txt`, which
//! comprises a series of broken source expressions and their expected
//! diagnostics.

use std::{
	collections::{BTreeMap, HashSet},
	time::{Duration, Instant}
};

use pretty_assertions::assert_eq;

use super::{
	ast::{DEPTH, LEAF, Nesting, nest_function, nest_parsable},
	recovery::{FAMILIES, SCALE}
};
use crate::{
	Parser, SourceSpan,
	diagnostics::{self, DiagnoseResult, DiagnosticKind, Edit},
	support::{on_small_stack, read_error_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                           Diagnostic test cases.                           //
////////////////////////////////////////////////////////////////////////////////

/// Map a [`DiagnosticKind`] to its test-file name.
pub(super) fn kind_name(kind: &DiagnosticKind) -> &'static str
{
	match kind
	{
		DiagnosticKind::UnclosedDelimiter { .. } => "UnclosedDelimiter",
		DiagnosticKind::MissingExpression { .. } => "MissingExpression",
		DiagnosticKind::UnopenedDelimiter { .. } => "UnopenedDelimiter",
		DiagnosticKind::MissingRightOperand { .. } => "MissingRightOperand",
		DiagnosticKind::MissingLeftOperand { .. } => "MissingLeftOperand",
		DiagnosticKind::BareIdentifier => "BareIdentifier",
		DiagnosticKind::MissingDiceFaces => "MissingDiceFaces",
		DiagnosticKind::IncompleteDropClause => "IncompleteDropClause",
		DiagnosticKind::MisplacedDropClause => "MisplacedDropClause",
		DiagnosticKind::IncompleteParameterDefinition =>
		{
			"IncompleteParameterDefinition"
		},
		DiagnosticKind::MissingParameter => "MissingParameter",
		DiagnosticKind::TrailingInput => "TrailingInput",
		DiagnosticKind::EmptyExpression => "EmptyExpression",
		DiagnosticKind::UnexpectedToken => "UnexpectedToken",
		DiagnosticKind::UnexpectedEof => "UnexpectedEof",
		DiagnosticKind::DuplicateParameter { .. } => "DuplicateParameter",
		DiagnosticKind::BindingCollidesWithParameter { .. } =>
		{
			"BindingCollidesWithParameter"
		},
		DiagnosticKind::DuplicateBinding { .. } => "DuplicateBinding",
		DiagnosticKind::UseBeforeBind { .. } => "UseBeforeBind"
	}
}

/// A [suggestion](diagnostics::Suggestion), replayed as the complete source
/// that results from applying it together with the first suggestions of all
/// earlier diagnostics. This is how `test_parser_errors.txt` writes
/// suggestions, as the doctor did before `xdy-i0q.14`.
#[derive(Debug)]
pub(super) struct Replayed
{
	/// The corrected source.
	pub corrected_source: String,

	/// The placeholders of the suggestion's own edits, as spans of the
	/// corrected source, with their descriptions and valid kinds.
	pub placeholders: Vec<(SourceSpan, &'static str, &'static [&'static str])>
}

/// Replay every suggestion of a [diagnosis](DiagnoseResult).
///
/// # Parameters
/// - `source`: The diagnosed source.
/// - `result`: The diagnosis.
///
/// # Returns
/// The replayed suggestions of each diagnostic, in order.
///
/// # Panics
/// If a suggestion's edits overlap those of an earlier diagnostic's first
/// suggestion.
pub(super) fn replay(
	source: &str,
	result: &DiagnoseResult
) -> Vec<Vec<Replayed>>
{
	let mut fixes: Vec<&Edit> = Vec::new();
	result
		.diagnostics
		.iter()
		.map(|diagnostic| {
			let replayed = diagnostic
				.suggestions
				.iter()
				.map(|suggestion| compose(source, &fixes, &suggestion.edits))
				.collect();
			if let Some(first) = diagnostic.suggestions.first()
			{
				fixes.extend(&first.edits);
			}
			replayed
		})
		.collect()
}

/// Apply the edits of a suggestion, together with the edits of earlier
/// fixes, to a source. Insertions at the same position apply in the order
/// given, the earlier fixes first, and before a replacement there.
///
/// The doctor's fix-and-retry loop, before `xdy-i0q.14`, computed each fix
/// against the source as corrected by the earlier fixes, so a fix may rewrite
/// a region that an earlier fix edited, e.g., to remove trailing input that
/// begins with a space that an earlier fix inserted. Mapped back to the
/// original source, the later edit's span covers the earlier edit's. So an edit
/// is superseded by any later edit whose span covers it: a replacement whose
/// span lies within the later span, or an insertion strictly inside it.
///
/// # Parameters
/// - `source`: The source.
/// - `fixes`: The edits of the earlier fixes, whose placeholders are ignored.
/// - `own`: The edits of the suggestion.
///
/// # Returns
/// The replayed suggestion.
///
/// # Panics
/// If any edits overlap.
fn compose(source: &str, fixes: &[&Edit], own: &[Edit]) -> Replayed
{
	let all = fixes
		.iter()
		.map(|&edit| (edit, false))
		.chain(own.iter().map(|edit| (edit, true)))
		.collect::<Vec<_>>();
	let covers = |later: &Edit, earlier: &Edit| {
		if earlier.span.start == earlier.span.end
		{
			later.span.start < earlier.span.start
				&& earlier.span.start < later.span.end
		}
		else
		{
			later.span.start <= earlier.span.start
				&& earlier.span.end <= later.span.end
		}
	};
	let mut edits = all
		.iter()
		.enumerate()
		.filter(|&(i, (edit, _))| {
			!all[i + 1..].iter().any(|(later, _)| covers(later, edit))
		})
		.map(|(_, &pair)| pair)
		.collect::<Vec<_>>();
	edits.sort_by_key(|(edit, _)| (edit.span.start, edit.span.end));
	let mut corrected_source = String::new();
	let mut placeholders = Vec::new();
	let mut copied = 0;
	for (edit, is_own) in edits.iter().copied()
	{
		assert!(
			edit.span.start >= copied,
			"overlapping edits in {:?}: {:?}",
			source,
			edits
		);
		corrected_source.push_str(&source[copied..edit.span.start]);
		let base = corrected_source.len();
		corrected_source.push_str(&edit.replacement);
		if is_own
		{
			placeholders.extend(edit.placeholders.iter().map(|p| {
				(
					SourceSpan {
						start: base + p.span.start,
						end: base + p.span.end
					},
					p.description,
					p.valid_kinds
				)
			}));
		}
		copied = edit.span.end;
	}
	corrected_source.push_str(&source[copied..]);
	Replayed {
		corrected_source,
		placeholders
	}
}

/// Test that the diagnostics module produces the expected diagnostics for each
/// test case in `test_parser_errors.txt`.
#[test]
fn test_error_diagnostics()
{
	let test_cases = read_error_test_cases(include_str!(
		"../../tests/test_parser_errors.txt"
	));
	assert!(
		!test_cases.is_empty(),
		"no test cases found in test_parser_errors.txt"
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
		let replayed = replay(case.source, &result);

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

			// Check the full Display rendering. This is a redundant but
			// load-bearing assertion: any divergence between the component-wise
			// fields above and the `Display` impl — for any of
			// `DiagnosticKind`, `SourceSpan`, `Diagnostic` — surfaces here with
			// a readable diff.
			assert_eq!(
				format!("{}", actual),
				expected.rendered,
				"case {}: {:?} — diagnostic {} rendered output mismatch",
				index + 1,
				case.source,
				di + 1
			);

			// Check related labels.
			assert_eq!(
				actual.related.len(),
				expected.related.len(),
				"case {}: {:?} — diagnostic {} related count mismatch: \
				 expected {}, got {}",
				index + 1,
				case.source,
				di + 1,
				expected.related.len(),
				actual.related.len()
			);
			for (ri, (actual_rel, expected_rel)) in actual
				.related
				.iter()
				.zip(expected.related.iter())
				.enumerate()
			{
				assert_eq!(
					(actual_rel.span.start, actual_rel.span.end),
					expected_rel.span,
					"case {}: {:?} — diagnostic {} related {} span mismatch",
					index + 1,
					case.source,
					di + 1,
					ri + 1
				);
				assert_eq!(
					actual_rel.message,
					expected_rel.message,
					"case {}: {:?} — diagnostic {} related {} message mismatch",
					index + 1,
					case.source,
					di + 1,
					ri + 1
				);
			}

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
				let suggestion = &replayed[di][si];

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
						Parser::parse(&suggestion.corrected_source).is_ok(),
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
						(actual_ph.0.start, actual_ph.0.end),
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
						actual_ph.1,
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
						actual_ph.2.to_vec(),
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
	let test_cases = read_error_test_cases(include_str!(
		"../../tests/test_parser_errors.txt"
	));
	for case in &test_cases
	{
		let result = diagnostics::diagnose(case.source);
		if let Some(corrected) = &result.corrected_source
		{
			assert!(
				Parser::parse(corrected).is_ok(),
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
		"{x}: {x}D6",
		"{x}, {y}: {x} + {y}",
		"{a}, {b}, {c}: ({a} + {b}) * {c}",
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

/// A duplicate-parameter program produces exactly one semantic diagnostic,
/// whose [`Display`](std::fmt::Display) rendering surfaces the caret-level
/// position of the duplicate and names the offending identifier.
#[test]
fn test_diagnose_duplicate_parameter_rendered_output()
{
	let result = diagnostics::diagnose("{x}, {x}: {x}");
	assert_eq!(result.diagnostics.len(), 1);
	let diag = &result.diagnostics[0];
	assert!(matches!(
		diag.kind,
		DiagnosticKind::DuplicateParameter { ref name } if name == "x"
	));
	assert_eq!(
		format!("{}", diag),
		"duplicate parameter `x` (6..7): parameter `x` is declared more than \
		 once; review references to `x` in the body — one may have meant a \
		 different parameter or an external variable"
	);
	// Related label points at the first occurrence.
	assert_eq!(diag.related.len(), 1);
	assert_eq!(diag.related[0].span.start, 1);
	assert_eq!(diag.related[0].span.end, 2);
	assert_eq!(diag.related[0].message, "first declared here");
}

/// Test that the doctor reports a duplicate parameter by its canonical name,
/// even when the source spells the two occurrences differently, and that its
/// rename suggestion replaces the duplicate exactly as written, across lines.
#[test]
fn test_diagnose_duplicate_canonical_parameter()
{
	let source = "{a b}, {a\n   b}: 1";
	let result = diagnostics::diagnose(source);
	assert_eq!(result.diagnostics.len(), 1);
	let diag = &result.diagnostics[0];
	assert!(matches!(
		diag.kind,
		DiagnosticKind::DuplicateParameter { ref name } if name == "a b"
	));
	assert_eq!(diag.span, SourceSpan { start: 8, end: 14 });
	let corrected = diag.suggestions[0].apply(source);
	assert_eq!(corrected, "{a b}, {new0}: 1");
	assert!(Parser::parse(&corrected).is_ok());
}

/// Test that the semantic validator is intentionally bypassed when the
/// recovering parse had to repair the source to obtain a clean parse.
/// Attaching semantic diagnostics (`DuplicateParameter`, etc.) to spans in a
/// fix-synthesized source would point at characters the user never typed, so
/// `diagnose()` runs the validator only when the original parses cleanly.
#[test]
fn test_validator_skipped_after_parse_fix()
{
	// Source has a parser error (`{x` is missing `}`) and a would-be semantic
	// error (`{x}, {x}` duplicates a parameter). The recovering parse repairs
	// it to `{x}, {x}: {x}`; we must not then run the validator on that
	// synthesized source and report a duplicate parameter, because the second
	// `x` the user typed is a statement the validator has not been asked to
	// corroborate yet.
	let result = diagnostics::diagnose("{x}, {x}: {x");
	assert_eq!(
		result.diagnostics.len(),
		1,
		"expected only the parser diagnostic, got {:?}",
		result
			.diagnostics
			.iter()
			.map(|d| format!("{}", d))
			.collect::<Vec<_>>()
	);
	assert!(
		!matches!(
			result.diagnostics[0].kind,
			DiagnosticKind::DuplicateParameter { .. }
		),
		"validator should not run after the recovering parse; got {:?}",
		result.diagnostics[0].kind
	);
}

////////////////////////////////////////////////////////////////////////////////
//                       DiagnosticKind Display tests.                        //
////////////////////////////////////////////////////////////////////////////////

/// The [`DiagnosticKind::UnopenedDelimiter`] variant's
/// [`Display`](std::fmt::Display) rendering shows the unmatched closer in
/// backticks. This variant is part of the public API but is not currently
/// emitted by the analyzer (closing brackets without openers surface as
/// [`UnexpectedToken`](DiagnosticKind::UnexpectedToken) today); the direct test
/// guards the `Display` impl regardless.
#[test]
fn test_diagnostic_kind_unopened_delimiter_display()
{
	let kind = DiagnosticKind::UnopenedDelimiter { closer: ')' };
	assert_eq!(format!("{}", kind), "unexpected `)`");
	let kind = DiagnosticKind::UnopenedDelimiter { closer: ']' };
	assert_eq!(format!("{}", kind), "unexpected `]`");
	let kind = DiagnosticKind::UnopenedDelimiter { closer: '}' };
	assert_eq!(format!("{}", kind), "unexpected `}`");
}

/// The [`DiagnosticKind::UnexpectedEof`] variant's
/// [`Display`](std::fmt::Display) renders a fixed human-readable phrase. This
/// variant is reachable only from the catch-all of the doctor's analysis, a
/// branch the current pattern set almost never routes to; the direct test
/// guards the `Display` impl regardless.
#[test]
fn test_diagnostic_kind_unexpected_eof_display()
{
	let kind = DiagnosticKind::UnexpectedEof;
	assert_eq!(format!("{}", kind), "unexpected end of input");
}

/// When the catch-all emits an
/// [`UnexpectedToken`](DiagnosticKind::UnexpectedToken) diagnostic, the token
/// span ends at the first whitespace byte after the error position — exercising
/// the success path of `unexpected_token`'s search for whitespace, which other
/// test cases (whose trailing text contains no interior whitespace) don't
/// reach.
#[test]
fn test_diagnose_unexpected_token_stops_at_whitespace()
{
	let result = diagnostics::diagnose("@\t");
	assert_eq!(result.diagnostics.len(), 1);
	let diag = &result.diagnostics[0];
	assert!(matches!(diag.kind, DiagnosticKind::UnexpectedToken));
	// Span ends at byte offset 1, where the tab begins — not at source.len().
	assert_eq!((diag.span.start, diag.span.end), (0, 1));
	assert_eq!(diag.message, "unexpected `@`");
	assert!(diag.suggestions.is_empty());
}

////////////////////////////////////////////////////////////////////////////////
//                    Whitespace that separates no tokens.                    //
////////////////////////////////////////////////////////////////////////////////

/// Test that the doctor replaces each character of stray whitespace, i.e.,
/// whitespace between tokens that may not separate them, with a space, wherever
/// it lies, and shows it by its code point (`xdy-zl4.10`).
#[test]
fn test_diagnose_stray_whitespace()
{
	for (source, expected, spans) in [
		("\u{a0}1", " 1", vec![(0, 2)]),
		("1\u{a0}", "1 ", vec![(1, 3)]),
		("1 +\u{a0}2", "1 + 2", vec![(3, 5)]),
		("(1\u{a0})", "(1 )", vec![(2, 4)]),
		(
			"3d\u{a0}6 drop\u{2003}lowest",
			"3d 6 drop lowest",
			vec![(2, 4), (10, 13)]
		),
		("{x},\u{b}{y}: {x}", "{x}, {y}: {x}", vec![(4, 5)]),
		("{x}@\u{a0}(1)", "{x}@ (1)", vec![(4, 6)])
	]
	{
		let result = diagnostics::diagnose(source);
		assert_eq!(result.corrected_source.as_deref(), Some(expected));
		assert!(Parser::parse(expected).is_ok());
		assert_eq!(
			result
				.diagnostics
				.iter()
				.map(|diag| (diag.span.start, diag.span.end))
				.collect::<Vec<_>>(),
			spans,
			"{:?}",
			source
		);
		for diag in &result.diagnostics
		{
			assert!(matches!(diag.kind, DiagnosticKind::UnexpectedToken));
			assert_eq!(diag.suggestions.len(), 1);
		}
	}
	let result = diagnostics::diagnose("\u{a0}1");
	let diag = &result.diagnostics[0];
	assert_eq!(
		diag.message,
		"unexpected `U+00A0`; only spaces, tabs, and line breaks may separate \
		 tokens"
	);
	assert_eq!(
		diag.suggestions[0].description,
		"replace `U+00A0` with a space"
	);
}

/// Test that whitespace inside braces is not stray, since it belongs to the
/// name, but whitespace after the break of an unclosed name is
/// (`xdy-zl4.10`).
#[test]
fn test_diagnose_stray_whitespace_outside_braces()
{
	let result = diagnostics::diagnose("{hit\u{a0}points} +\u{a0}1");
	assert_eq!(
		result.corrected_source.as_deref(),
		Some("{hit\u{a0}points} + 1")
	);
	assert_eq!(result.diagnostics.len(), 1);
	assert_eq!(
		result.diagnostics[0].span,
		SourceSpan { start: 15, end: 17 }
	);
	let result = diagnostics::diagnose("{x + 2\u{a0}* 3");
	assert_eq!(result.corrected_source.as_deref(), Some("{x} + 2 * 3"));
	assert_eq!(result.diagnostics.len(), 2);
	assert!(matches!(
		result.diagnostics[0].kind,
		DiagnosticKind::UnclosedDelimiter { opener: '{', .. }
	));
	assert_eq!(result.diagnostics[1].span, SourceSpan { start: 6, end: 8 });
}

/// Test that the doctor fixes stray whitespace before a bare parameter, and
/// then the parameter, in source order (`xdy-zl4.10`).
#[test]
fn test_diagnose_stray_whitespace_before_bare_parameter()
{
	let result = diagnostics::diagnose("{x}, \u{a0}hit points: 0");
	assert_eq!(
		result.corrected_source.as_deref(),
		Some("{x},  {hit points}: 0")
	);
	assert_eq!(result.diagnostics.len(), 2);
	assert!(matches!(
		result.diagnostics[0].kind,
		DiagnosticKind::UnexpectedToken
	));
	assert_eq!(result.diagnostics[0].span, SourceSpan { start: 5, end: 7 });
	assert!(matches!(
		result.diagnostics[1].kind,
		DiagnosticKind::BareIdentifier
	));
	assert_eq!(result.diagnostics[1].span, SourceSpan { start: 7, end: 17 });
}

/// Test that a fix that removes trailing input subsumes the stray whitespace
/// within it (`xdy-zl4.10`).
#[test]
fn test_diagnose_stray_whitespace_in_trailing_input()
{
	let result = diagnostics::diagnose("1 2\u{a0}3");
	assert_eq!(result.corrected_source.as_deref(), Some("1"));
	assert_eq!(result.diagnostics.len(), 1);
	assert!(matches!(
		result.diagnostics[0].kind,
		DiagnosticKind::TrailingInput
	));
}

/// Test that a fix that braces a bare name leaves the stray whitespace after
/// it to a fix of its own, and that the diagnostic shows the name without it
/// (`xdy-zl4.9`, `xdy-zl4.10`).
#[test]
fn test_diagnose_bare_name_before_stray_whitespace()
{
	for (source, expected, what) in [
		("hit points\u{a0}: 0", "{hit points} : 0", "parameter"),
		(
			"{x}, hit points\u{a0}: 0",
			"{x}, {hit points} : 0",
			"parameter"
		),
		("1 + hit points\u{a0}", "1 + {hit points} ", "variable")
	]
	{
		let result = diagnostics::diagnose(source);
		assert_eq!(result.corrected_source.as_deref(), Some(expected));
		assert!(Parser::parse(expected).is_ok());
		assert_eq!(result.diagnostics.len(), 2);
		let diag = &result.diagnostics[0];
		assert!(matches!(diag.kind, DiagnosticKind::BareIdentifier));
		assert_eq!(
			diag.message,
			format!(
				"bare identifier `hit points` is not valid here; {}s must be \
				 wrapped in `{{}}`",
				what
			)
		);
		assert_eq!(
			diag.suggestions[0].description,
			format!("use `hit points` as a {} name", what)
		);
		assert!(matches!(
			result.diagnostics[1].kind,
			DiagnosticKind::UnexpectedToken
		));
	}
}

/// Test that a fix that breaks the name of an unclosed variable before an
/// operator keeps the whitespace before the operator that may not separate
/// tokens inside the braces (`xdy-zl4.9`).
#[test]
fn test_diagnose_unclosed_name_braces_foreign_whitespace()
{
	let result = diagnostics::diagnose("{x\u{a0}+ 2");
	assert_eq!(result.corrected_source.as_deref(), Some("{x\u{a0}}+ 2"));
	assert!(Parser::parse("{x\u{a0}}+ 2").is_ok());
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
		"{x}: {x}D6 + {x}",
		"(1D6 + 2) * 3",
		"[1:20]",
		"1D6 + 1D8 + 1D10",
		"{a}, {b}: {a}D{b} + 5",
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

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the names gathered for rename suggestions survive a deep chain
/// of every nesting construct on a small stack. The `test_diagnose_deep_*`
/// tests exercise the suggestions themselves through
/// [`diagnose`](diagnostics::diagnose).
#[test]
#[ignore = "stress: run with just stress"]
fn test_collect_in_use_names_deep()
{
	on_small_stack(|| {
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			let function = nest_function(nesting, DEPTH);
			let names = diagnostics::collect_in_use_names(&function);
			let expected = if nesting.binds()
			{
				HashSet::from(["a", "x"])
			}
			else
			{
				HashSet::from(["x"])
			};
			assert_eq!(names, expected, "{:?}", nesting);
		}
	});
}

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of groups on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_group() { diagnose_deep(Nesting::Group) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of range starts on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_range_start() { diagnose_deep(Nesting::RangeStart) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of range ends on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_range_end() { diagnose_deep(Nesting::RangeEnd) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of negations on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_negation() { diagnose_deep(Nesting::Negation) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of exponents on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_exponent() { diagnose_deep(Nesting::Exponent) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of additions on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_addition() { diagnose_deep(Nesting::Addition) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of subtractions on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_subtraction() { diagnose_deep(Nesting::Subtraction) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of multiplications on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_multiplication()
{
	diagnose_deep(Nesting::Multiplication)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of divisions on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_division() { diagnose_deep(Nesting::Division) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of modulos on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_modulo() { diagnose_deep(Nesting::Modulo) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of dice counts on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_dice_count() { diagnose_deep(Nesting::DiceCount) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of dice faces on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_dice_faces() { diagnose_deep(Nesting::DiceFaces) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of custom dice counts on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_custom_count() { diagnose_deep(Nesting::CustomCount) }

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep chain of drop expressions on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_drop_expression()
{
	diagnose_deep(Nesting::DropExpression)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) suggests renames past a
/// deep stack of drop clauses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_drop_clauses() { diagnose_deep(Nesting::DropClauses) }

/// On a small stack, [diagnose](diagnostics::diagnose) a binding that collides
/// with a parameter, a duplicate binding, and a use before binding, each
/// followed by a [`DEPTH`]-deep nesting and then a reference to the external
/// variable `x`. Ensure that each yields exactly its diagnostic, and that the
/// diagnostic suggests renaming the binding `b` to `y`. The rename skips past
/// the highest single-letter name in use, so it is `y` only if the walk that
/// gathers the names in use crossed the whole nesting to find `x`; otherwise,
/// it would be `c`.
///
/// # Parameters
/// - `nesting`: The way to nest. It must not introduce bindings, since every
///   such binding is named `a`, so that the nesting itself binds a name twice,
///   and the validator reports that duplicate binding before the error that the
///   case expects.
fn diagnose_deep(nesting: Nesting)
{
	assert!(!nesting.binds(), "{:?} introduces bindings", nesting);
	on_small_stack(|| {
		let nest = nest_parsable(nesting, DEPTH, 1).to_string();
		let cases = [
			(
				format!("{{b}}: {{b}}@(1) + {} + {{x}}", nest),
				format!("{{b}}: {{y}}@(1) + {} + {{x}}", nest),
				"BindingCollidesWithParameter"
			),
			(
				format!("{{b}}@(1) + {{b}}@(2) + {} + {{x}}", nest),
				format!("{{b}}@(1) + {{y}}@(2) + {} + {{x}}", nest),
				"DuplicateBinding"
			),
			(
				format!("{{b}} + {{b}}@(1) + {} + {{x}}", nest),
				format!("{{b}} + {{y}}@(1) + {} + {{x}}", nest),
				"UseBeforeBind"
			)
		];
		for (source, corrected, kind) in cases
		{
			let result = diagnostics::diagnose(&source);
			assert_eq!(result.diagnostics.len(), 1, "{:?}: {}", nesting, kind);
			let diagnostic = &result.diagnostics[0];
			assert_eq!(kind_name(&diagnostic.kind), kind, "{:?}", nesting);
			assert!(
				diagnostic.suggestions[0].apply(&source) == corrected,
				"{:?}: {} did not suggest renaming `b` to `y`",
				nesting,
				kind
			);
		}
	});
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep failures.                               //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// groups with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_group()
{
	diagnose_deep_failing(Nesting::Group, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// bindings with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_binding()
{
	diagnose_deep_failing(Nesting::Binding, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range starts with a [hole](Breakage::Hole) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_range_start()
{
	diagnose_deep_failing(Nesting::RangeStart, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range ends with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_range_end()
{
	diagnose_deep_failing(Nesting::RangeEnd, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// negations with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_negation()
{
	diagnose_deep_failing(Nesting::Negation, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// exponents with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_exponent()
{
	diagnose_deep_failing(Nesting::Exponent, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// additions with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_addition()
{
	diagnose_deep_failing(Nesting::Addition, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// subtractions with a [hole](Breakage::Hole) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_subtraction()
{
	diagnose_deep_failing(Nesting::Subtraction, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// multiplications with a [hole](Breakage::Hole) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_multiplication()
{
	diagnose_deep_failing(Nesting::Multiplication, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// divisions with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_division()
{
	diagnose_deep_failing(Nesting::Division, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// modulos with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_modulo()
{
	diagnose_deep_failing(Nesting::Modulo, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice counts with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_dice_count()
{
	diagnose_deep_failing(Nesting::DiceCount, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice faces with a [hole](Breakage::Hole) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_dice_faces()
{
	diagnose_deep_failing(Nesting::DiceFaces, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// custom dice counts with a [hole](Breakage::Hole) at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_custom_count()
{
	diagnose_deep_failing(Nesting::CustomCount, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// drop expressions with a [hole](Breakage::Hole) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_drop_expression()
{
	diagnose_deep_failing(Nesting::DropExpression, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep stack of
/// drop clauses with a [hole](Breakage::Hole) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_drop_clauses()
{
	diagnose_deep_failing(Nesting::DropClauses, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep mixture of
/// every nesting construct with a [hole](Breakage::Hole) at its innermost
/// level, on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_hole_mixed()
{
	diagnose_deep_failing(Nesting::Mixed, Breakage::Hole)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// groups [truncated](Breakage::Truncated) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_group()
{
	diagnose_deep_failing(Nesting::Group, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// bindings [truncated](Breakage::Truncated) at its innermost level, on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_binding()
{
	diagnose_deep_failing(Nesting::Binding, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range starts [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_range_start()
{
	diagnose_deep_failing(Nesting::RangeStart, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range ends [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_range_end()
{
	diagnose_deep_failing(Nesting::RangeEnd, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// negations [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_negation()
{
	diagnose_deep_failing(Nesting::Negation, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// exponents [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_exponent()
{
	diagnose_deep_failing(Nesting::Exponent, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// additions [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_addition()
{
	diagnose_deep_failing(Nesting::Addition, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// subtractions [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_subtraction()
{
	diagnose_deep_failing(Nesting::Subtraction, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// multiplications [truncated](Breakage::Truncated) at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_multiplication()
{
	diagnose_deep_failing(Nesting::Multiplication, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// divisions [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_division()
{
	diagnose_deep_failing(Nesting::Division, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// modulos [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_modulo()
{
	diagnose_deep_failing(Nesting::Modulo, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice counts [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_dice_count()
{
	diagnose_deep_failing(Nesting::DiceCount, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice faces [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_dice_faces()
{
	diagnose_deep_failing(Nesting::DiceFaces, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// custom dice counts [truncated](Breakage::Truncated) at its innermost level,
/// on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_custom_count()
{
	diagnose_deep_failing(Nesting::CustomCount, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// drop expressions [truncated](Breakage::Truncated) at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_drop_expression()
{
	diagnose_deep_failing(Nesting::DropExpression, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep stack of
/// drop clauses [truncated](Breakage::Truncated) at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_drop_clauses()
{
	diagnose_deep_failing(Nesting::DropClauses, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep mixture of
/// every nesting construct [truncated](Breakage::Truncated) at its innermost
/// level, on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_truncated_mixed()
{
	diagnose_deep_failing(Nesting::Mixed, Breakage::Truncated)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// groups with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_group()
{
	diagnose_deep_failing(Nesting::Group, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// bindings with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_binding()
{
	diagnose_deep_failing(Nesting::Binding, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range starts with a [stray](Breakage::Stray) `@` at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_range_start()
{
	diagnose_deep_failing(Nesting::RangeStart, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// range ends with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_range_end()
{
	diagnose_deep_failing(Nesting::RangeEnd, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// negations with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_negation()
{
	diagnose_deep_failing(Nesting::Negation, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// exponents with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_exponent()
{
	diagnose_deep_failing(Nesting::Exponent, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// additions with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_addition()
{
	diagnose_deep_failing(Nesting::Addition, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// subtractions with a [stray](Breakage::Stray) `@` at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_subtraction()
{
	diagnose_deep_failing(Nesting::Subtraction, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// multiplications with a [stray](Breakage::Stray) `@` at its innermost level,
/// on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_multiplication()
{
	diagnose_deep_failing(Nesting::Multiplication, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// divisions with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_division()
{
	diagnose_deep_failing(Nesting::Division, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// modulos with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_modulo()
{
	diagnose_deep_failing(Nesting::Modulo, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice counts with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_dice_count()
{
	diagnose_deep_failing(Nesting::DiceCount, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// dice faces with a [stray](Breakage::Stray) `@` at its innermost level, on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_dice_faces()
{
	diagnose_deep_failing(Nesting::DiceFaces, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// custom dice counts with a [stray](Breakage::Stray) `@` at its innermost
/// level, on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_custom_count()
{
	diagnose_deep_failing(Nesting::CustomCount, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep chain of
/// drop expressions with a [stray](Breakage::Stray) `@` at its innermost level,
/// on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_drop_expression()
{
	diagnose_deep_failing(Nesting::DropExpression, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep stack of
/// drop clauses with a [stray](Breakage::Stray) `@` at its innermost level, on
/// a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_drop_clauses()
{
	diagnose_deep_failing(Nesting::DropClauses, Breakage::Stray)
}

/// Ensure that [`diagnose`](diagnostics::diagnose) diagnoses a deep mixture of
/// every nesting construct with a [stray](Breakage::Stray) `@` at its innermost
/// level, on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_deep_stray_mixed()
{
	diagnose_deep_failing(Nesting::Mixed, Breakage::Stray)
}

/// A way to break a nested source at its innermost constant, the
/// [`LEAF`].
#[derive(Clone, Copy, Debug)]
enum Breakage
{
	/// Delete the leaf, so that the source lacks an operand at its innermost
	/// level, but closes every level.
	Hole,

	/// Delete the leaf and everything after it, so that the source closes no
	/// level.
	Truncated,

	/// Replace the leaf with a stray `@`, which begins no expression.
	Stray
}

impl Breakage
{
	/// Break a nested source.
	///
	/// # Parameters
	/// - `nesting`: The way to nest.
	/// - `depth`: The number of levels of nesting.
	///
	/// # Returns
	/// The broken source.
	fn source(self, nesting: Nesting, depth: usize) -> String
	{
		let source = nest_parsable(nesting, depth, LEAF).to_string();
		let leaf = LEAF.to_string();
		assert_eq!(source.matches(&leaf).count(), 1, "{:?}", nesting);
		let start = source.find(&leaf).unwrap();
		let end = start + leaf.len();
		match self
		{
			Breakage::Hole => format!("{}{}", &source[..start], &source[end..]),
			Breakage::Truncated => source[..start].to_string(),
			Breakage::Stray =>
			{
				format!("{}@{}", &source[..start], &source[end..])
			}
		}
	}
}

/// The number of diagnostics of each [kind](kind_name) in a diagnosis.
type Census = BTreeMap<&'static str, usize>;

/// Count the diagnostics of each [kind](kind_name) in a diagnosis.
///
/// # Parameters
/// - `result`: The diagnosis.
///
/// # Returns
/// The census.
fn census(result: &DiagnoseResult) -> Census
{
	let mut census = Census::new();
	for diagnostic in &result.diagnostics
	{
		*census.entry(kind_name(&diagnostic.kind)).or_default() += 1;
	}
	census
}

/// Predict the [census](Census) of the diagnosis of a [`DEPTH`]-deep source,
/// broken as specified, from the diagnoses of shallow ones. Every nesting
/// repeats with a period of [`Nesting::ROTATION`]'s length, so the number of
/// diagnostics of each kind must be an affine function of the number of
/// periods; the prediction extrapolates it from a base depth congruent to
/// [`DEPTH`] and the depth one period deeper, and checks that two more periods
/// continue the line.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `breakage`: The way to break the source.
///
/// # Returns
/// The predicted census, and whether the shallow sources were fixed.
fn predict_census(nesting: Nesting, breakage: Breakage) -> (Census, bool)
{
	let period = Nesting::ROTATION.len();
	let base = period + DEPTH % period;
	let results = [base, base + period, base + 2 * period, base + 3 * period]
		.map(|depth| diagnostics::diagnose(&breakage.source(nesting, depth)));
	let fixed = results[0].corrected_source.is_some();
	assert!(
		results
			.iter()
			.all(|r| r.corrected_source.is_some() == fixed),
		"{:?}, {:?}: fixability varies with depth",
		nesting,
		breakage
	);
	let censuses = results.each_ref().map(census);
	let kinds = censuses
		.iter()
		.flat_map(|census| census.keys().copied())
		.collect::<HashSet<_>>();
	let periods = (DEPTH - base) / period;
	let mut predicted = Census::new();
	for kind in kinds
	{
		let counts = censuses
			.each_ref()
			.map(|census| census.get(kind).copied().unwrap_or(0) as isize);
		let step = counts[1] - counts[0];
		assert!(
			counts[2] - counts[1] == step && counts[3] - counts[2] == step,
			"{:?}, {:?}: {} diagnostics do not grow linearly: {:?}",
			nesting,
			breakage,
			kind,
			counts
		);
		let count = counts[0] + periods as isize * step;
		if count > 0
		{
			predicted.insert(kind, count as usize);
		}
	}
	(predicted, fixed)
}

/// On a small stack, [diagnose](diagnostics::diagnose) a [`DEPTH`]-deep
/// source, broken at its innermost constant. Ensure that the diagnosis has the
/// [census](Census) that [`predict_census`] extrapolates from shallow sources,
/// so that no diagnostic is lost or duplicated at depth, and that the doctor
/// fixes the source just when it fixes the shallow ones, producing a source
/// that parses. The default time budget of the small stack also bounds the time
/// of the diagnosis: the fix-and-retry loop that preceded the recovering parse
/// took time quadratic in the depth, so it would not finish.
///
/// # Parameters
/// - `nesting`: The way to nest.
/// - `breakage`: The way to break the source.
fn diagnose_deep_failing(nesting: Nesting, breakage: Breakage)
{
	on_small_stack(|| {
		let (predicted, fixed) = predict_census(nesting, breakage);
		let source = breakage.source(nesting, DEPTH);
		let result = diagnostics::diagnose(&source);
		assert_eq!(census(&result), predicted, "{:?}, {:?}", nesting, breakage);
		assert_eq!(
			result.corrected_source.is_some(),
			fixed,
			"{:?}, {:?}",
			nesting,
			breakage
		);
		// Use `assert!` rather than `expect`, since a failure would otherwise
		// render the whole error.
		if let Some(corrected) = &result.corrected_source
		{
			assert!(
				Parser::parse(corrected).is_ok(),
				"{:?}, {:?}: corrected source fails to parse",
				nesting,
				breakage
			);
		}
	});
}

////////////////////////////////////////////////////////////////////////////////
//                                 Linearity.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The number of repetitions in the smaller inputs of the linearity test. It is
/// smaller than that of the [engine's](super::recovery) linearity test, since
/// each run of the test takes several diagnoses of each input.
const WIDTH: usize = 2_000;

/// The number of times that the linearity test diagnoses each input, keeping
/// the fastest time, so that a stall of the machine does not fail the test.
const RUNS: usize = 3;

/// The greatest factor by which the time of a diagnosis may grow when the input
/// grows by [`SCALE`]. It leaves room for the noise of the machine and the
/// cost of touching more memory, but quadratic time would grow by [`SCALE`]
/// times more.
const TIME_SCALE: u32 = 3 * SCALE as u32;

/// The least time against which the linearity test measures the growth of the
/// time of a diagnosis. Some families end the diagnosis at their first
/// failure, whatever the input, in a few microseconds, which a stall of the
/// machine could multiply by more than [`TIME_SCALE`].
const TIME_FLOOR: Duration = Duration::from_micros(100);

/// A builder of the member of a family of inputs with the given number of
/// repetitions.
type Member = fn(usize) -> String;

/// The families of inputs for the linearity test that only the doctor meets,
/// beyond the [engine's](FAMILIES): their name, and the [builder](Member) of
/// their members. Some make a single repair that consumes every repetition.
const DIAGNOSE_FAMILIES: &[(&str, Member)] = &[
	("leading closers", |n| format!("{}1", ")".repeat(n))),
	("bare dice count", |n| format!("{}D6", "x".repeat(n))),
	("parameter list", |n| vec!["{x}"; n].join(", ")),
	("bare identifier sum", |n| vec!["x"; n].join(" + ")),
	("long identifier", |n| "xd".repeat(n)),
	("trailing input", |n| format!("1{}", " 2".repeat(n))),
	("stray whitespace", |n| vec!["1"; n].join(" +\u{a0}")),
	("strays after unclosed names", |n| "{x + 2\u{a0}".repeat(n)),
	("misplaced drops", |n| vec!["3 drop lowest"; n].join(" + ")),
	("operandless drops", |n| {
		format!("1{}", " + drop lowest".repeat(n))
	})
];

/// Ensure that [`diagnose`](diagnostics::diagnose) takes time and produces
/// output linear in the length of the input, for each of the
/// [engine's families](FAMILIES) of inputs and [its own](DIAGNOSE_FAMILIES):
/// that [`SCALE`] times as many repetitions produce at most one more than
/// [`SCALE`] times as much [text](text_size), and take at most
/// [`TIME_SCALE`] times as long, or as [`TIME_FLOOR`] if that is longer, where
/// quadratic growth would multiply either by [`SCALE`] again. The engine's own
/// linearity test counts its steps, which do not include the doctor's
/// inspection of the source, so this test measures time instead. Wall-clock
/// time is noisy under a busy machine, so the test is ignored by default;
/// `just stress` runs it.
#[test]
#[ignore = "stress: run with just stress"]
fn test_diagnose_is_linear()
{
	let families = FAMILIES
		.iter()
		.map(|family| (family.name, family.source))
		.chain(DIAGNOSE_FAMILIES.iter().copied());
	for (name, source) in families
	{
		let sources = [WIDTH, SCALE * WIDTH].map(source);
		let mut times = [Duration::MAX; 2];
		let mut sizes = [0; 2];
		for _ in 0..RUNS
		{
			for (i, source) in sources.iter().enumerate()
			{
				let start = Instant::now();
				let result = diagnostics::diagnose(source);
				times[i] = times[i].min(start.elapsed());
				assert!(!result.diagnostics.is_empty(), "{}", name);
				sizes[i] = text_size(&result);
			}
		}
		assert!(
			sizes[1] <= (SCALE + 1) * sizes[0],
			"{} produced {} bytes, then {} for {} times the input",
			name,
			sizes[0],
			sizes[1],
			SCALE
		);
		assert!(
			times[1] <= TIME_SCALE * times[0].max(TIME_FLOOR),
			"{} took {:?}, then {:?} for {} times the input",
			name,
			times[0],
			times[1],
			SCALE
		);
	}
}

/// Answer the size of the text of a diagnosis: the messages, the descriptions
/// of the suggestions, and the replacements of their edits.
///
/// # Parameters
/// - `result`: The diagnosis.
///
/// # Returns
/// The size, in bytes.
fn text_size(result: &DiagnoseResult) -> usize
{
	result
		.diagnostics
		.iter()
		.map(|diagnostic| {
			diagnostic.message.len()
				+ diagnostic
					.related
					.iter()
					.map(|label| label.message.len())
					.sum::<usize>()
				+ diagnostic
					.suggestions
					.iter()
					.map(|suggestion| {
						suggestion.description.len()
							+ suggestion
								.edits
								.iter()
								.map(|edit| edit.replacement.len())
								.sum::<usize>()
					})
					.sum::<usize>()
		})
		.sum()
}
