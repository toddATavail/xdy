//! # Span-metadata S-expression test cases
//!
//! Herein are the test cases covering the span-metadata extensions to the
//! S-expression format: lossless round-tripping, span-prefix emission and
//! parsing, synthetic-span suppression, opaque-group handling, and reader
//! tolerance and rejection behaviour.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                     Span-metadata S-expression tests.                      //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that every source in `test_parse.txt` round-trips losslessly
/// through the S-expression format when span metadata and opaque groups
/// are both enabled. The parser-originated AST is serialized with
/// `with_spans = true` and `with_groups = true`, then parsed back; the
/// resulting AST must compare equal to the original *without* the
/// position-normalising [`Spanned::untethered`] pass.
#[test]
fn test_s_expr_lossless_roundtrip()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions, read_s_expr};
	use crate::support::read_compilation_test_cases;

	let test_cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	let opts = SExpressibleOptions::default()
		.with_spans(true)
		.with_groups(true);
	for (index, (source, _expected)) in test_cases.iter().enumerate()
	{
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
		let serialized = ast.to_s_expr(opts);
		let reparsed = match read_s_expr(&serialized)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {}: lossless s-expr read failed for {:?}\n\
				 serialized: {}\n\
				 error: {}",
				index + 1,
				source,
				serialized,
				e
			)
		};
		assert_eq!(
			reparsed,
			ast,
			"case {}: lossless round-trip mismatch for {:?}\n\
			 serialized: {}",
			index + 1,
			source,
			serialized
		);
	}
}

/// Ensure that the default [`SExpressibleOptions`] omit both the span prefix
/// and the opaque group form, preserving the hand-written format used
/// throughout the test corpus.
#[test]
fn test_s_expr_default_options_omit_span_metadata_and_groups()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions};

	let ast = Parser::parse("(2 + 3) * 4").unwrap();
	let output = ast.to_s_expr(SExpressibleOptions::default());
	assert!(
		!output.contains('^'),
		"default output must not include span prefixes: {}",
		output
	);
	assert!(
		!output.contains("(group"),
		"default output must not include opaque groups: {}",
		output
	);
}

/// Ensure that [synthetic](SourceSpan::SYNTHETIC) spans are suppressed from the
/// writer output even when `with_spans = true`. Hand-built ASTs and the output
/// of [`Spanned::untethered`] should render without vacuous `^[0 0]` ceremony.
#[test]
fn test_s_expr_synthetic_spans_are_suppressed()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions};

	let ast = Parser::parse("{x} + 1").unwrap().untethered();
	let opts = SExpressibleOptions::default()
		.with_spans(true)
		.with_groups(true);
	let output = ast.to_s_expr(opts);
	assert!(
		!output.contains('^'),
		"untethered AST must not emit span prefixes: {}",
		output
	);
}

/// Ensure that the writer emits `^[start end]` prefixes at the expected
/// positions for a parser-originated AST when `with_spans = true`.
#[test]
fn test_s_expr_writer_emits_span_prefixes()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions};

	let ast = Parser::parse("42").unwrap();
	let opts = SExpressibleOptions::default().with_spans(true);
	let output = ast.to_s_expr(opts);
	assert_eq!(
		output.trim(),
		"^[0 2] (function [] ^[0 2] 42)",
		"writer output mismatch: {}",
		output
	);
}

/// Ensure that the reader accepts a span-annotated form and populates the
/// resulting node's [`SourceSpan`] accordingly.
#[test]
fn test_s_expr_reader_accepts_span_prefix()
{
	use crate::s_expr::read_s_expr;

	let ast = read_s_expr("^[0 2] (function [] ^[0 2] 42)").unwrap();
	assert_eq!(ast.span, SourceSpan { start: 0, end: 2 });
	match &ast.body
	{
		Expression::Constant(c) =>
		{
			assert_eq!(c.value, 42);
			assert_eq!(c.span, SourceSpan { start: 0, end: 2 });
		},
		other => panic!("unexpected body: {:?}", other)
	}
}

/// Ensure that the reader tolerates arbitrary whitespace between the `]`
/// terminator of a span prefix and the form it annotates.
#[test]
fn test_s_expr_reader_tolerates_whitespace_after_span_prefix()
{
	use crate::s_expr::read_s_expr;

	let ast = read_s_expr("^[0 2]    (function [] ^[0 2]\n\t42)").unwrap();
	assert_eq!(ast.span, SourceSpan { start: 0, end: 2 });
}

/// Ensure that the reader accepts per-parameter span prefixes in the parameter
/// list.
#[test]
fn test_s_expr_reader_accepts_parameter_span_prefixes()
{
	use crate::s_expr::read_s_expr;

	let ast =
		read_s_expr("^[0 7] (function [^[0 1] a ^[3 4] b] ^[6 7] 0)").unwrap();
	let params = ast.parameters.as_ref().expect("expected parameters");
	assert_eq!(params.len(), 2);
	assert_eq!(params[0].name, "a");
	assert_eq!(params[0].span, SourceSpan { start: 0, end: 1 });
	assert_eq!(params[1].name, "b");
	assert_eq!(params[1].span, SourceSpan { start: 3, end: 4 });
}

/// Ensure that the reader accepts the opaque `(group <expr>)` form and returns
/// a structurally equivalent [`Group`] node.
#[test]
fn test_s_expr_reader_accepts_opaque_group()
{
	use crate::s_expr::read_s_expr;

	let ast = read_s_expr("(function [] (group 42))").unwrap();
	match &ast.body
	{
		Expression::Group(g) => match g.expression.as_ref()
		{
			Expression::Constant(c) => assert_eq!(c.value, 42),
			other => panic!("expected Constant, got {:?}", other)
		},
		other => panic!("expected Group, got {:?}", other)
	}
}

/// Ensure that the reader still accepts span-less s-expressions, giving every
/// node the default (synthetic) span.
#[test]
fn test_s_expr_reader_accepts_span_less_input()
{
	use crate::s_expr::read_s_expr;

	let ast = read_s_expr("(function [x] (add {x} 1))").unwrap();
	assert_eq!(ast.span, SourceSpan::SYNTHETIC);
	let params = ast.parameters.as_ref().unwrap();
	assert_eq!(params[0].span, SourceSpan::SYNTHETIC);
	assert_eq!(ast.body.span(), SourceSpan::SYNTHETIC);
}

/// Ensure that the reader rejects `^` followed by anything other than `[`,
/// rather than silently skipping unrecognized metadata shapes.
#[test]
fn test_s_expr_reader_rejects_unrecognized_metadata()
{
	use crate::s_expr::read_s_expr;

	for bad in [
		"^{start 0 end 5} 42",
		"^symbol 42",
		"^\"quoted\" 42",
		"^ [0 5] 42"
	]
	{
		let input = format!("(function [] {})", bad);
		let err = read_s_expr(&input).unwrap_err();
		assert!(
			err.to_string().contains("expected '[' after '^'"),
			"input {:?} — unexpected error: {}",
			bad,
			err
		);
	}
}

/// Ensure that the reader rejects a span whose start exceeds its end.
#[test]
fn test_s_expr_reader_rejects_inverted_span()
{
	use crate::s_expr::read_s_expr;

	let err = read_s_expr("^[5 2] (function [] 0)").unwrap_err();
	assert!(
		err.to_string().contains("span start 5 exceeds end 2"),
		"unexpected error: {}",
		err
	);
}

/// Ensure that the reader rejects a child span that escapes its enclosing
/// parent's span.
#[test]
fn test_s_expr_reader_rejects_containment_violation()
{
	use crate::s_expr::read_s_expr;

	// The outer add spans 0..5 but its right child claims to span 0..9,
	// which escapes the parent on the right.
	let input = "(function [] ^[0 5] (add ^[0 1] 1 ^[4 9] 2))";
	let err = read_s_expr(input).unwrap_err();
	assert!(
		err.to_string().contains("escapes parent span"),
		"unexpected error: {}",
		err
	);
}

/// Ensure that the reader rejects siblings that overlap or appear out of source
/// order.
#[test]
fn test_s_expr_reader_rejects_sibling_overlap()
{
	use crate::s_expr::read_s_expr;

	let input = "(function [] ^[0 5] (add ^[0 3] 1 ^[2 5] 2))";
	let err = read_s_expr(input).unwrap_err();
	assert!(
		err.to_string()
			.contains("overlaps or precedes prior sibling"),
		"unexpected error: {}",
		err
	);
}

/// Ensure that the reader accepts a synthetic child span within a non-synthetic
/// parent — containment checks must not fire when either operand is
/// [synthetic](SourceSpan::SYNTHETIC), allowing hand-written mixes of annotated
/// and unannotated forms.
#[test]
fn test_s_expr_reader_allows_mixed_synthetic_and_annotated()
{
	use crate::s_expr::read_s_expr;

	// The outer form is annotated, but the inner constant is not.
	let ast = read_s_expr("^[0 5] (function [] ^[0 5] (add 1 2))").unwrap();
	assert_eq!(ast.span, SourceSpan { start: 0, end: 5 });
}
