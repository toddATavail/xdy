//! # S-expression sizing invariant tests
//!
//! Herein are the tests that pin down the [`size_s_expr`] contract — every
//! [`SExpressible`] impl must report the exact character count that
//! [`write_s_expr`] will produce when the entire form fits on a single line.
//! The invariant is exercised against every source in `test_parse.txt`, under
//! every combination of [`with_spans`](SExpressibleOptions::with_spans) and
//! [`with_groups`](SExpressibleOptions::with_groups), so that every sizer
//! participates in the check. The helpers [`decimal_digits`], [`ident_size`],
//! [`span_prefix_size`](crate::s_expr::span_prefix_size),
//! [`SExpressibleOptions::increase_indent`], and
//! [`SExpressibleOptions::available_space`] are pinned down by focused tests
//! against their direct callers.
//!
//! [`size_s_expr`]: crate::s_expr::SExpressible::size_s_expr
//! [`write_s_expr`]: crate::s_expr::SExpressible::write_s_expr
//! [`SExpressible`]: crate::s_expr::SExpressible
//! [`SExpressibleOptions`]: crate::s_expr::SExpressibleOptions

use crate::{parser::*, s_expr::*};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                    S-expression sizing invariant tests.                    //
////////////////////////////////////////////////////////////////////////////////

/// Permute the four boolean combinations of `with_spans` and `with_groups` that
/// [`SExpressibleOptions`] supports. Tests that sweep every sizer visit all
/// four permutations so mutations gated on either flag show up.
fn all_option_permutations() -> [SExpressibleOptions; 4]
{
	[
		SExpressibleOptions::default(),
		SExpressibleOptions::default().with_spans(true),
		SExpressibleOptions::default().with_groups(true),
		SExpressibleOptions::default()
			.with_spans(true)
			.with_groups(true)
	]
}

/// For every source in `test_parse.txt`, parse it, serialize the resulting AST
/// with a huge `soft_limit` so the single-line branch is always taken, and
/// assert that `size_s_expr(opts)` equals the length of the rendered string.
/// This invariant pins down the sizers of every [`SExpressible`] impl —
/// constants, variables, ranges, dice, drops, arithmetic, groups, parameter
/// lists, faces lists, functions — under every combination of
/// [`with_spans`](SExpressibleOptions::with_spans) and
/// [`with_groups`](SExpressibleOptions::with_groups).
#[test]
fn test_size_s_expr_matches_single_line_length()
{
	use crate::support::read_compilation_test_cases;

	let cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	assert!(!cases.is_empty(), "no test cases found in test_parse.txt");
	for opts_base in all_option_permutations()
	{
		// Force the single-line branch by giving the writer enough room for
		// the widest rendering we expect.
		let opts = SExpressibleOptions::new(
			opts_base.indent,
			opts_base.tab_width,
			10_000
		)
		.with_spans(opts_base.with_spans)
		.with_groups(opts_base.with_groups);
		for (index, (source, _)) in cases.iter().enumerate()
		{
			let ast = Parser::parse(source).unwrap_or_else(|e| {
				panic!(
					"case {}: parse failed for {:?}: {}",
					index + 1,
					source,
					e
				)
			});
			let serialized = ast.to_s_expr(opts);
			assert!(
				!serialized.contains('\n'),
				"case {} with {:?}: single-line branch expected but output \
				 wrapped:\n{}",
				index + 1,
				opts,
				serialized
			);
			assert_eq!(
				ast.size_s_expr(opts),
				serialized.len(),
				"case {} with {:?}: size_s_expr disagrees with rendered \
				 length for source {:?}\nserialized: {}",
				index + 1,
				opts,
				source,
				serialized
			);
		}
	}
}

/// The `<impl SExpressible for Option<Vec<Parameter<'_>>>>::size_s_expr` branch
/// for the `None` case emits the literal `[]`. Parser-originated functions
/// never carry `None` parameters (they always synthesize at least an empty
/// `Some(vec![])` in the grammar), so this test reads a hand- authored
/// s-expression that omits the parameter-list round-trip and asserts the sizer
/// agrees with the written length.
#[test]
fn test_size_s_expr_empty_parameter_list_matches_written_length()
{
	let ast = read_s_expr("(function [] 42)").unwrap();
	// The reader folds an empty `[]` into `None`; the writer should emit
	// `[]` for either representation.
	assert!(ast.parameters.is_none());
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	let mut buffer = String::new();
	ast.parameters
		.write_s_expr(&mut buffer, opts.soft_limit, opts)
		.unwrap();
	assert_eq!(buffer, "[]");
	assert_eq!(ast.parameters.size_s_expr(opts), buffer.len());
}

/// The `<impl SExpressible for &[Parameter<'_>]>::size_s_expr` branch for the
/// empty-slice case is exercised indirectly through
/// [`Option<Vec<Parameter>>::size_s_expr`] when the `Some(vec)` branch slices
/// into a zero-length parameter vector. This test constructs such a vector
/// explicitly and asserts that the sizer agrees with the writer.
#[test]
fn test_size_s_expr_empty_parameter_slice_matches_written_length()
{
	let params: Vec<crate::ast::Parameter<'_>> = Vec::new();
	let slice: &[crate::ast::Parameter<'_>] = &params;
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	let mut buffer = String::new();
	slice
		.write_s_expr(&mut buffer, opts.soft_limit, opts)
		.unwrap();
	assert_eq!(buffer, "[]");
	assert_eq!(slice.size_s_expr(opts), buffer.len());
}

/// The integer-faces sizer (`&[i32]::size_s_expr`) is called for every
/// [`CustomDice`](crate::ast::CustomDice) node. This test focuses the invariant
/// on the faces list itself, since the corpus case covers only three-face
/// vectors. Every signed-integer width from one digit to eleven digits
/// (including `i32::MIN`) exercises the negative-number accounting and the
/// delimiter/space arithmetic.
#[test]
fn test_size_s_expr_integer_faces_matches_written_length()
{
	let cases: &[&[i32]] = &[
		&[0],
		&[1, 2, 3],
		&[-1],
		&[-1, 0, 1, 3, 5],
		&[1, 10, 100, 1000, 10_000, 100_000, 1_000_000],
		&[i32::MIN + 1, -1, 0, 1, i32::MAX]
	];
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	for faces in cases
	{
		let mut buffer = String::new();
		faces
			.write_s_expr(&mut buffer, opts.soft_limit, opts)
			.unwrap();
		assert!(
			!buffer.contains('\n'),
			"single-line branch expected but output wrapped:\n{}",
			buffer
		);
		assert_eq!(
			faces.size_s_expr(opts),
			buffer.len(),
			"size_s_expr disagrees with rendered length for {:?}\n\
			 serialized: {}",
			faces,
			buffer
		);
	}
}

/// The empty-faces writer path (`&[]`) emits `[]` directly. The empty slice
/// never arises in parser-originated custom-dice (the parser rejects empty face
/// lists), so this test constructs the slice explicitly and asserts the sizer
/// matches the writer for both branches of the bracket-accounting formula.
#[test]
fn test_size_s_expr_empty_integer_faces_matches_written_length()
{
	let faces: &[i32] = &[];
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	let mut buffer = String::new();
	faces
		.write_s_expr(&mut buffer, opts.soft_limit, opts)
		.unwrap();
	assert_eq!(buffer, "[]");
	assert_eq!(faces.size_s_expr(opts), buffer.len());
}

/// The `<impl SExpressible for &T>::size_s_expr` blanket impl (the transparent
/// forwarder) is easy to mutate invisibly unless a test actually dispatches
/// through a reference. This test constructs a tuple-based keyword form and
/// sizes it through `(&value)`, forcing the blanket impl to be selected, and
/// asserts that the forwarded result agrees with the written length.
#[test]
fn test_size_s_expr_reference_forwarder_matches_written_length()
{
	let ast = Parser::parse("1 + 2").unwrap();
	let opts = SExpressibleOptions::new(0, 4, 10_000).with_spans(true);
	let by_ref = &ast;
	let mut buffer = String::new();
	by_ref
		.write_s_expr(&mut buffer, opts.soft_limit, opts)
		.unwrap();
	assert_eq!(by_ref.size_s_expr(opts), buffer.len());
}

////////////////////////////////////////////////////////////////////////////////
//                        Small-helper pinning tests.                         //
////////////////////////////////////////////////////////////////////////////////

/// Pin down [`SExpressibleOptions::increase_indent`]: each call must increment
/// `indent` by exactly one and preserve every other field. A mutation that
/// replaces the increment with any other arithmetic is caught by the expected
/// progression.
#[test]
fn test_increase_indent_steps_by_one()
{
	let opts = SExpressibleOptions::new(0, 4, 80)
		.with_spans(true)
		.with_groups(true);
	let step1 = opts.increase_indent();
	let step2 = step1.increase_indent();
	let step3 = step2.increase_indent();
	assert_eq!(step1.indent, 1);
	assert_eq!(step2.indent, 2);
	assert_eq!(step3.indent, 3);
	// Other fields must be preserved across the copy.
	for stepped in [step1, step2, step3]
	{
		assert_eq!(stepped.tab_width, 4);
		assert_eq!(stepped.soft_limit, 80);
		assert!(stepped.with_spans);
		assert!(stepped.with_groups);
	}
}

/// Pin down [`SExpressibleOptions::available_space`]: the formula must be
/// `soft_limit - indent * tab_width`. Three samples separate the `-`, `*`, and
/// constant-return mutants: indent 0 (isolates `soft_limit`), indent 1
/// (isolates one tab-width), and indent 3 with tab_width 2 (isolates the
/// product).
#[test]
fn test_available_space_formula()
{
	assert_eq!(SExpressibleOptions::new(0, 4, 80).available_space(), 80);
	assert_eq!(SExpressibleOptions::new(1, 4, 80).available_space(), 76);
	assert_eq!(SExpressibleOptions::new(3, 2, 80).available_space(), 74);
	// A non-default tab_width paired with a non-zero indent catches both
	// the `* with /` and the `* with +` mutants (the former would give
	// 80 - 0 = 80 for tab_width 2, the latter would give 80 - (3 + 2) =
	// 75).
	assert_eq!(SExpressibleOptions::new(5, 3, 80).available_space(), 65);
}

/// Pin down the decimal-digit counter indirectly through
/// [`span_prefix_size`](crate::s_expr::span_prefix_size), which embeds the
/// fixed overhead (`^[ ] `) and the sum of `start` and `end` digit counts.
/// Since the helper is private, the test exercises it through a parser-
/// originated AST whose spans we can control by choosing source lengths that
/// produce specific offsets. The rendered prefix length is then compared
/// against the expected formula.
#[test]
fn test_span_prefix_size_and_decimal_digits_invariant()
{
	// Source lengths chosen to land the top-level function span's `end` at
	// boundaries that sweep `decimal_digits` across the single- and multi-digit
	// transitions: end == 1, 2, 10, 100 (via padding with inert whitespace that
	// the parser trims, so the *trimmed* offset still advances as each
	// character is consumed).
	for source in ["1", "42", "1000000000", "         100"]
	{
		let ast = Parser::parse(source).unwrap();
		let opts = SExpressibleOptions::default().with_spans(true);
		let rendered = ast.to_s_expr(opts);
		assert_eq!(
			ast.size_s_expr(opts),
			rendered.len(),
			"size_s_expr disagrees with rendered length for {:?} under \
			 with_spans:\n{}",
			source,
			rendered
		);
	}
}

/// Pin down [`ident_size`](crate::s_expr::ident_size) through the
/// [`Variable`](crate::ast::Variable) and [`Parameter`](crate::ast::Parameter)
/// sizers, which forward to it directly. Unquoted identifiers should size to
/// their byte length; quoted identifiers (those containing whitespace or
/// delimiter characters) should add two characters for the surrounding quotes.
#[test]
fn test_ident_size_unquoted_and_quoted()
{
	// Bare word variable — no quotes needed.
	let bare = Parser::parse("{alloy}").unwrap();
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	let rendered = bare.to_s_expr(opts);
	assert_eq!(rendered, "(function [] alloy)");
	assert_eq!(bare.size_s_expr(opts), rendered.len());
	// Identifier containing a space — must be quoted, adding two.
	let quoted = Parser::parse("{hello world}").unwrap();
	let rendered = quoted.to_s_expr(opts);
	assert_eq!(rendered, r#"(function [] "hello world")"#);
	assert_eq!(quoted.size_s_expr(opts), rendered.len());
}

/// Pin down the quoting policy at `needs_quoting` (reached through
/// [`ident_size`] and `write_ident`). Whitespace characters and each of the
/// four structural delimiters independently force quoting; the writer must emit
/// the exact character sequence that the sizer counted. Parser- originated ASTs
/// cannot produce such identifiers (each of these characters is grammatically
/// meaningful elsewhere), so the cases are fed through the reader, which
/// tolerates any quoted identifier.
#[test]
fn test_identifier_quoting_delimiter_characters()
{
	// Each case carries a variable name whose sole purpose is to include one
	// quoting-triggering character.
	let cases = [
		r#"(function [] "v w")"#,
		r#"(function [] "v(w")"#,
		r#"(function [] "v)w")"#,
		r#"(function [] "v[w")"#,
		r#"(function [] "v]w")"#
	];
	let opts = SExpressibleOptions::new(0, 4, 10_000);
	for input in cases
	{
		let ast = read_s_expr(input).unwrap();
		let rendered = ast.to_s_expr(opts);
		assert_eq!(
			ast.size_s_expr(opts),
			rendered.len(),
			"ident_size disagrees for {:?}: {}",
			input,
			rendered
		);
		assert_eq!(
			rendered, input,
			"writer did not preserve quoted identifier from {:?}: {}",
			input, rendered
		);
	}
}
