//! # S-expression reader error tests
//!
//! Herein are the tests that exercise every error branch of
//! [`read_s_expr`](crate::s_expr::read_s_expr) and its supporting [`nom`]-based
//! combinators. Each negative case pins down a distinct [`SExprError`] variant
//! — the rendered-message substring and the byte offset — so that a mutation to
//! the reader's match arms, guard expressions, or span arithmetic surfaces as a
//! diff.
//!
//! [`SExprError`]: crate::s_expr::SExprError

use crate::s_expr::*;

////////////////////////////////////////////////////////////////////////////////
//                      S-expression reader error tests.                      //
////////////////////////////////////////////////////////////////////////////////

/// Assert that reading `input` produces an error whose rendered Display
/// contains the given substring and whose location offset matches the expected
/// byte position.
fn assert_reader_error(input: &str, message_contains: &str, offset: usize)
{
	let err = read_s_expr(input).expect_err(&format!(
		"expected reader error for {:?} (matching {:?}), got success",
		input, message_contains
	));
	let rendered = err.to_string();
	assert!(
		rendered.contains(message_contains),
		"input {:?}: expected message to contain {:?}, got {:?}",
		input,
		message_contains,
		rendered
	);
	assert_eq!(
		err.location().offset,
		offset,
		"input {:?}: expected offset {} but got {} (rendered was {:?})",
		input,
		offset,
		err.location().offset,
		rendered
	);
}

/// An empty input cannot produce a function: the reader reports an
/// unexpected end of input where it would otherwise consume `(`.
#[test]
fn test_reader_rejects_empty_input()
{
	assert_reader_error("", "expected '('", 0);
}

/// The top-level form must begin with `(function …)`. Any other keyword
/// produces a dedicated [`SExprError::ExpectedTopLevelFunction`] whose offset
/// points at the start of the rogue keyword — pinning down the keyword-mark
/// capture in `read_function`. Leading whitespace pushes the cursor well past
/// byte 0, so a mutation that collapses the mark to some constant origin would
/// land a distinct offset.
#[test]
fn test_reader_rejects_non_function_top_level_keyword()
{
	assert_reader_error(
		"     (body 42)",
		"expected 'function', found 'body'",
		6
	);
}

/// A `function` keyword that appears inside an expression (rather than at the
/// top) produces a dedicated error. This pins down the explicit match arm in
/// `read_expr` that rejects nested `function` forms — a mutation that deletes
/// the arm would allow `function` to fall through to the generic "unknown
/// keyword" handler with a different message.
#[test]
fn test_reader_rejects_nested_function_keyword()
{
	assert_reader_error(
		"(function [] (function [] 42))",
		"unexpected 'function' inside expression",
		14
	);
}

/// An unknown keyword in operator position produces
/// [`SExprError::UnknownKeyword`] whose offset points at the keyword's start.
/// This pins down the keyword-mark capture in `read_expr` alongside the
/// default match arm.
#[test]
fn test_reader_rejects_unknown_keyword()
{
	assert_reader_error(
		"(function [] (bogus 1 2))",
		"unknown keyword 'bogus'",
		14
	);
}

/// A top-level input that doesn't open with `(` errors with the expected
/// byte-offset pointing at the offending character. This pins down the
/// `expect_char` error path — the mutant that replaces the guard with `true`
/// would silently accept the offending character.
#[test]
fn test_reader_rejects_missing_opening_paren()
{
	assert_reader_error("42", "expected '('", 0);
}

/// A top-level input missing its closing `)` after the body errors with
/// "expected ')', found end of input". The offset reports the position at
/// end-of-input.
#[test]
fn test_reader_rejects_missing_top_level_close_paren()
{
	assert_reader_error(
		"(function [] 42",
		"expected ')', found end of input",
		15
	);
}

/// A `)` replaced by any other delimiter character errors with the exact
/// "expected ')', found ']'" phrasing and the offset points at the `]`. This
/// anchors the pre-consume span capture in `expect_char`. A non-delimiter
/// character would be folded into the body's identifier or integer, so a
/// delimiter is required to surface the wrong-character error.
#[test]
fn test_reader_rejects_wrong_char_where_close_paren_expected()
{
	assert_reader_error("(function [] 42]", "expected ')', found ']'", 15);
}

/// The reader tolerates no input after the closing `)` of the top-level
/// function; any residue is reported as trailing input.
#[test]
fn test_reader_rejects_trailing_input()
{
	assert_reader_error(
		"(function [] 42) extra",
		"trailing input: 'extra'",
		17
	);
}

/// A bare word at the start of an expression yields "expected a word" only when
/// the bare region is empty. When the reader's cursor sits on a delimiter where
/// a word is required — for instance, after the keyword in an empty compound
/// form — the failure is reported at that position.
#[test]
fn test_reader_rejects_empty_compound_form()
{
	// `( )` — no keyword follows the open paren; the word reader fails.
	assert_reader_error("(function [] ( ))", "expected a word", 15);
}

/// A parameter-list that is missing its opening `[` reports the substitute
/// character precisely. Anchors the `expect_char('[')` path in `read_params`.
#[test]
fn test_reader_rejects_missing_params_open_bracket()
{
	assert_reader_error("(function x 42)", "expected '[', found 'x'", 10);
}

/// A parameter-list missing its closing `]` (reaching end of input while
/// scanning identifiers) fails in `read_ident` with the bare-word reader
/// failing at EOF.
#[test]
fn test_reader_rejects_unterminated_parameter_list()
{
	// End-of-input occurs inside `read_ident`'s call to `read_word` after
	// the comma-scanning loop.
	assert_reader_error("(function [x", "expected a word", 12);
}

/// A quoted identifier missing its closing `"` errors with
/// "unterminated quoted identifier" and the offset points at the opening
/// quote. Anchors the pre-advance span capture in `read_ident`.
#[test]
fn test_reader_rejects_unterminated_quoted_identifier()
{
	// The unterminated string straddles the end of input.
	assert_reader_error(
		r#"(function ["unterminated"#,
		"unterminated quoted identifier",
		11
	);
}

/// A non-integer where an integer constant is expected fails with
/// "invalid integer" and the offset points at the start of the offending
/// word. Anchors the word-span capture in `read_integer`.
#[test]
fn test_reader_rejects_invalid_integer_literal()
{
	// The body position accepts an expression; a bare `-xyz` (with a
	// leading minus) takes the signed-integer branch, which then fails to
	// parse the remainder as `i32`.
	assert_reader_error("(function [] -xyz)", "invalid integer '-xyz'", 13);
}

/// An integer that exceeds `i32::MAX` in the positive direction errors as an
/// invalid integer with overflow detail.
#[test]
fn test_reader_rejects_integer_overflow()
{
	assert_reader_error(
		"(function [] 99999999999999)",
		"invalid integer '99999999999999'",
		13
	);
}

/// An integer that exceeds `i32::MIN` in the negative direction errors as an
/// invalid integer with overflow detail.
#[test]
fn test_reader_rejects_integer_underflow()
{
	assert_reader_error(
		"(function [] -99999999999999)",
		"invalid integer '-99999999999999'",
		13
	);
}

/// A `custom-dice` with an empty `[]` faces list errors explicitly. The reader
/// disallows zero-face custom dice to match the parser's grammar.
#[test]
fn test_reader_rejects_empty_custom_dice_faces_list()
{
	assert_reader_error(
		"(function [] (custom-dice 1 []))",
		"custom dice faces list must not be empty",
		30
	);
}

/// A malformed face value inside a `custom-dice` list propagates the
/// integer-parsing error — this pins down `read_integer`'s use inside
/// `read_faces`, with the offset pointing at the bad word.
#[test]
fn test_reader_rejects_invalid_face_value()
{
	assert_reader_error(
		"(function [] (custom-dice 1 [1 xyz 3]))",
		"invalid integer 'xyz'",
		31
	);
}

/// Passing a non-dice expression (a bare integer) as the first argument to
/// `drop-lowest` fails type-checking inside `read_dice_child`. The offset is
/// captured before whitespace is consumed, so it points at the space
/// immediately after the keyword.
#[test]
fn test_reader_rejects_non_dice_child_of_drop_lowest()
{
	assert_reader_error(
		"(function [] (drop-lowest 5))",
		"expected dice expression",
		25
	);
}

/// The same type-check fires for `drop-highest`, with the offset at the
/// pre-whitespace cursor position.
#[test]
fn test_reader_rejects_non_dice_child_of_drop_highest()
{
	assert_reader_error(
		"(function [] (drop-highest 5))",
		"expected dice expression",
		26
	);
}

/// A `^[start end]` span prefix with `start > end` is rejected. The error
/// offset points at the `^` that introduced the bad span.
#[test]
fn test_reader_rejects_inverted_span_prefix()
{
	assert_reader_error(
		"^[5 2] (function [] 0)",
		"span start 5 exceeds end 2",
		0
	);
}

/// A `^` followed by a character other than `[` is an explicit error (catches
/// `^{…}`, `^symbol`, `^ […]` with an intervening space).
#[test]
fn test_reader_rejects_non_bracket_metadata_shape()
{
	assert_reader_error("^{0 5} (function [] 0)", "expected '[' after '^'", 1);
}

/// A `^` at end of input errors with "found end of input".
#[test]
fn test_reader_rejects_dangling_caret()
{
	assert_reader_error("^", "expected '[' after '^', found end of input", 1);
}

/// A non-numeric start-of-span errors via `read_usize`.
#[test]
fn test_reader_rejects_non_numeric_span_start()
{
	assert_reader_error("^[x 5] (function [] 0)", "invalid byte offset 'x'", 2);
}

/// A span with a missing closing `]` errors via `expect_char(']')` — the
/// offset of the trailing garbage is reported.
#[test]
fn test_reader_rejects_missing_span_close_bracket()
{
	assert_reader_error("^[0 5 (function [] 0)", "expected ']', found '('", 6);
}

/// A zero-length span (start == end) is legal — this anchors the
/// `start > end` guard in `read_span_prefix`. A mutant that loosened the
/// check to `>=` would error here.
#[test]
fn test_reader_accepts_zero_length_span()
{
	let ast = read_s_expr("^[3 3] (function [] 0)").unwrap();
	assert_eq!(ast.span, crate::span::SourceSpan { start: 3, end: 3 });
}

/// A synthetic-parent + non-synthetic-child pairing must still succeed: both
/// the containment and sibling-order checks bail out when either operand is
/// synthetic. This anchors the `||` short-circuit in `validate_containment` and
/// `validate_sibling_order` — mutants that swap `||` for `&&` would require
/// *both* operands to be synthetic, causing the positional check to fire (and
/// fail) here.
#[test]
fn test_reader_allows_synthetic_parent_with_annotated_child()
{
	// Synthetic parent (no span prefix on `function`) with non-synthetic
	// add body. No error should result.
	let ast = read_s_expr("(function [] ^[0 5] (add 1 2))").unwrap();
	assert_eq!(ast.span, crate::span::SourceSpan::SYNTHETIC);
	assert_eq!(
		crate::span::Spanned::span(&ast.body),
		crate::span::SourceSpan { start: 0, end: 5 }
	);
}

/// Siblings mixing annotated and synthetic spans must not trigger the
/// positional sibling-order check. A parameter with a real span followed by one
/// without exercises the `||` short-circuit in [`validate_sibling_order`]: the
/// original code bails out whenever either operand is synthetic, but a mutant
/// that swaps `||` for `&&` would only bail when *both* are synthetic, causing
/// the positional check (`prev.end <= next.start`) to fire with
/// `10 <= 0 = false` and report a spurious overlap.
///
/// [`validate_sibling_order`]: crate::s_expr
#[test]
fn test_reader_allows_mixed_sibling_annotation_order()
{
	let ast = read_s_expr("(function [^[5 10] alpha beta] 42)").unwrap();
	let params = ast.parameters.as_ref().expect("expected parameters");
	assert_eq!(params.len(), 2);
	assert_eq!(params[0].name, "alpha");
	assert_eq!(
		params[0].span,
		crate::span::SourceSpan { start: 5, end: 10 }
	);
	assert_eq!(params[1].name, "beta");
	assert_eq!(params[1].span, crate::span::SourceSpan::SYNTHETIC);
}

/// A non-synthetic parent (e.g., `function` with an outer `^[0 9]`) must accept
/// a synthetic child (e.g., an `add` with no span prefix) — the early-out
/// short-circuit is otherwise load-bearing.
#[test]
fn test_reader_allows_annotated_parent_with_synthetic_child()
{
	let ast = read_s_expr("^[0 9] (function [] (add 1 2))").unwrap();
	assert_eq!(ast.span, crate::span::SourceSpan { start: 0, end: 9 });
	assert_eq!(
		crate::span::Spanned::span(&ast.body),
		crate::span::SourceSpan::SYNTHETIC
	);
}

/// A non-synthetic parameter whose span falls outside the function's declared
/// range is rejected by the containment check.
#[test]
fn test_reader_rejects_parameter_span_escaping_parent()
{
	let err = read_s_expr("^[0 5] (function [^[10 11] x] 0)")
		.expect_err("expected containment failure");
	let rendered = err.to_string();
	assert!(
		rendered.contains("escapes parent span"),
		"message was {:?}",
		rendered
	);
	assert!(
		matches!(err, SExprError::ChildSpanEscapesParent { .. }),
		"expected ChildSpanEscapesParent, got {:?}",
		err
	);
}

/// Out-of-order parameters (second declared before first in source position)
/// are rejected by the sibling-order check.
#[test]
fn test_reader_rejects_out_of_order_parameters()
{
	let err = read_s_expr("^[0 9] (function [^[5 6] a ^[2 3] b] 0)")
		.expect_err("expected sibling-order failure");
	let rendered = err.to_string();
	assert!(
		rendered.contains("overlaps or precedes prior sibling"),
		"message was {:?}",
		rendered
	);
	assert!(
		matches!(err, SExprError::SiblingSpanOutOfOrder { .. }),
		"expected SiblingSpanOutOfOrder, got {:?}",
		err
	);
}

/// The reader rejects a compound form that closes before its keyword supplies
/// the required operand count. `(neg)` lacks its single operand; the reader's
/// attempt to read the subexpression falls through to `read_ident`'s bare-word
/// reader, which reports "expected a word" at the `)` position that blocked
/// word-scanning.
#[test]
fn test_reader_rejects_neg_missing_operand()
{
	assert_reader_error("(function [] (neg))", "expected a word", 17);
}

/// Similar arity violation for `add`: missing right-hand operand.
#[test]
fn test_reader_rejects_add_missing_right_operand()
{
	assert_reader_error("(function [] (add 1))", "expected a word", 19);
}

////////////////////////////////////////////////////////////////////////////////
//                     SExprError::Display pinning test.                      //
////////////////////////////////////////////////////////////////////////////////

/// [`SExprError`] renders as `S-expression error at line L, column C (byte N):
/// <message>`. A mutation that drops the Display implementation entirely would
/// replace the output with an empty default; this test anchors the exact
/// format.
#[test]
fn test_sexpr_error_display_format()
{
	let err = read_s_expr("(function [] )").unwrap_err();
	let rendered = err.to_string();
	let location = err.location();
	assert!(
		rendered.starts_with("S-expression error at line "),
		"unexpected Display prefix: {:?}",
		rendered
	);
	assert!(
		rendered.contains(&format!(
			"line {}, column {} (byte {})",
			location.line, location.column, location.offset
		)),
		"Display output must include location suffix: {:?}",
		rendered
	);
	assert!(
		rendered.contains(&location.offset.to_string()),
		"Display output must contain the offset: {:?}",
		rendered
	);
}

////////////////////////////////////////////////////////////////////////////////
//                    SExprLocation line/column accuracy.                     //
////////////////////////////////////////////////////////////////////////////////

/// Single-line inputs report line 1 and a column equal to `offset + 1`. A
/// mutation that swapped `location_line` for a constant or dropped the line
/// increment entirely would fail this pin.
#[test]
fn test_location_single_line_byte_and_column_agree()
{
	let err = read_s_expr("     (body 42)").expect_err("expected error");
	let loc = err.location();
	assert_eq!(loc.line, 1, "expected line 1 on single-line input");
	assert_eq!(loc.offset, 6, "expected byte offset 6");
	assert_eq!(loc.column, 7, "expected 1-based column 7 (= offset + 1)");
}

/// A multi-line input should report the 1-based line number of the offending
/// token plus a column measured from the start of its line, *not* from the
/// start of input. The input straddles two newlines — the `(` opens on line
/// 1, the keyword `body` sits on line 3 — which pins the `\n` accounting in
/// `nom_locate`.
#[test]
fn test_location_multiline_reports_line_and_column()
{
	let input = "\n\n     (body 42)";
	let err = read_s_expr(input).expect_err("expected error");
	let loc = err.location();
	assert_eq!(loc.line, 3, "expected line 3");
	assert_eq!(loc.column, 7, "expected column 7 (after 6 spaces)");
	assert_eq!(
		loc.offset, 8,
		"byte offset counts the newlines plus the leading spaces"
	);
}

/// CRLF line endings must be counted as line terminators. The `\r\n` sequence
/// bumps the line number once, not twice — a regression that treated `\r` as
/// its own line would report line 3 here.
#[test]
fn test_location_crlf_line_endings()
{
	let input = "\r\n     (body 42)";
	let err = read_s_expr(input).expect_err("expected error");
	let loc = err.location();
	assert_eq!(loc.line, 2, "CRLF counts as a single line terminator");
	assert_eq!(loc.column, 7);
}

/// Containment-violation errors surface a typed
/// [`SExprError::ChildSpanEscapesParent`] carrying both the child and parent
/// spans for mechanical inspection. This pins the variant shape alongside the
/// substring-match coverage above.
#[test]
fn test_child_span_escapes_parent_variant_shape()
{
	let err = read_s_expr("^[0 5] (function [^[10 11] x] 0)")
		.expect_err("expected containment failure");
	let SExprError::ChildSpanEscapesParent { child, parent, .. } = err
	else
	{
		panic!("expected ChildSpanEscapesParent")
	};
	assert_eq!(child, crate::span::SourceSpan { start: 10, end: 11 });
	assert_eq!(parent, crate::span::SourceSpan { start: 0, end: 5 });
}

/// Inverted-span errors surface a typed [`SExprError::InvertedSpan`] carrying
/// the start and end as written, for mechanical inspection.
#[test]
fn test_inverted_span_variant_shape()
{
	let err = read_s_expr("^[5 2] (function [] 0)")
		.expect_err("expected inverted-span error");
	let SExprError::InvertedSpan { start, end, .. } = err
	else
	{
		panic!("expected InvertedSpan")
	};
	assert_eq!(start, 5);
	assert_eq!(end, 2);
}

/// Unknown-keyword errors carry the rejected keyword for mechanical inspection.
#[test]
fn test_unknown_keyword_variant_shape()
{
	let err = read_s_expr("(function [] (bogus 1 2))")
		.expect_err("expected unknown-keyword error");
	let SExprError::UnknownKeyword { keyword, .. } = err
	else
	{
		panic!("expected UnknownKeyword")
	};
	assert_eq!(keyword, "bogus");
}

/// `ExpectedChar` distinguishes "found a different character" from "found end
/// of input" by setting `found` to `None` for the latter.
#[test]
fn test_expected_char_distinguishes_eoi_from_mismatch()
{
	let eoi = read_s_expr("(function [] 42")
		.expect_err("expected missing-close-paren error");
	let SExprError::ExpectedChar {
		expected, found, ..
	} = eoi
	else
	{
		panic!("expected ExpectedChar")
	};
	assert_eq!(expected, ')');
	assert_eq!(found, None);

	let mismatch = read_s_expr("(function [] 42]")
		.expect_err("expected wrong-close-paren error");
	let SExprError::ExpectedChar {
		expected, found, ..
	} = mismatch
	else
	{
		panic!("expected ExpectedChar")
	};
	assert_eq!(expected, ')');
	assert_eq!(found, Some(']'));
}
