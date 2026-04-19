//! # S-expression multi-line formatting tests
//!
//! Herein are the tests that exercise the wrap-on-overflow branches of
//! [`SExpressible::write_s_expr`](crate::s_expr::SExpressible::write_s_expr).
//! The [`soft_limit`](crate::s_expr::SExpressibleOptions::soft_limit) is
//! deliberately shrunk to force the writer to break content across multiple
//! lines, exposing the arithmetic that predicts whether each form will fit.
//! Every wrap branch — function parameters and body, parameter lists,
//! integer-face lists, binary and ternary keyword forms, and unary forms — has
//! at least one dedicated case.

use crate::{parser::*, s_expr::*};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                 S-expression multi-line formatting tests.                  //
////////////////////////////////////////////////////////////////////////////////

/// Render `source` with the given `soft_limit` and assert the resulting
/// multi-line layout exactly matches `expected`. Panics if the writer produces
/// a single line (i.e., the soft-limit boundary was miscalibrated and no wrap
/// occurred).
fn assert_multiline(source: &str, soft_limit: usize, expected: &str)
{
	let ast = Parser::parse(source)
		.unwrap_or_else(|e| panic!("parse failed for {:?}: {}", source, e));
	let opts = SExpressibleOptions::new(0, 4, soft_limit);
	let actual = ast.to_s_expr(opts);
	assert!(
		actual.contains('\n'),
		"multi-line branch expected for {:?} at soft_limit {} but got \
		 single line:\n{}",
		source,
		soft_limit,
		actual
	);
	assert_eq!(
		actual, expected,
		"multi-line layout mismatch for {:?} at soft_limit {}",
		source, soft_limit
	);
}

/// At `soft_limit` 19, `(function [] (add 1 2))` does not fit on one line. The
/// parameter list still fits on the `(function` line, so only the body wraps to
/// its own line at indent 1. This pins down the fits/wraps boundary for
/// [`Function::write_s_expr`](crate::s_expr::SExpressible) — mutants that
/// perturb the subtraction arithmetic around `- 9` or the comparison against
/// `body.size_s_expr` flip this boundary.
#[test]
fn test_function_body_wraps_under_tight_soft_limit()
{
	assert_multiline("1 + 2", 19, "(function []\n\t(add 1 2))");
}

/// At `soft_limit` 15, the parameter list `[alpha beta gamma]` does not fit on
/// the `(function ` line, so the writer wraps the parameters onto their own
/// line at indent 1 and splits each parameter onto its own line at indent 2.
/// The body `42` still fits on the closing-bracket line. This pins down the
/// wrap branch of
/// [`<&[Parameter<'_>]>::write_s_expr`](crate::s_expr::SExpressible).
#[test]
fn test_function_parameters_wrap_each_on_own_line()
{
	assert_multiline(
		"alpha, beta, gamma: 42",
		15,
		"(function\n\t[\n\t\talpha\n\t\tbeta\n\t\tgamma\n\t] 42)"
	);
}

/// At `soft_limit` 13, the body `(add 1 2)` wraps to indent 1 and — since the
/// tab-widened budget at indent 1 is exactly one character short of fitting the
/// ternary form plus its trailing paren — its two subexpressions wrap again to
/// indent 2. This pins down the wrap branch of
/// [`<(A, B, C)>::write_s_expr`](crate::s_expr::SExpressible) at the boundary
/// where `size + indent` straddles the remaining space: a mutant that replaces
/// `+` with `*` would compute `size * 1 = size`, flipping the fit/wrap decision
/// from wrap to fit.
#[test]
fn test_ternary_form_wraps_each_operand()
{
	assert_multiline("1 + 2", 13, "(function []\n\t(add\n\t\t1\n\t\t2))");
}

/// At `soft_limit` 14, the body `(neg abcd)` wraps to indent 1 and its single
/// operand wraps again to indent 2. This pins down the wrap branch of
/// [`<(A, B)>::write_s_expr`](crate::s_expr::SExpressible) at the boundary
/// where `size + indent` straddles the remaining space — a mutant that replaces
/// `+` with `*` would compute `size * 1 = size`, flipping the fit/wrap
/// decision. [`Neg`](crate::ast::Neg) of an identifier exercises the unary
/// tuple impl (the lexer folds `-<integer>` into a signed [`Constant`] at parse
/// time, so a literal like `-3` would never reach this path).
#[test]
fn test_unary_form_wraps_operand()
{
	assert_multiline("-{abcd}", 14, "(function []\n\t(neg\n\t\tabcd))");
}

/// Recursive wrapping must push the indentation one tab deeper at each level. A
/// left-associated `1 + 2 + 3` forces three nested wraps at `soft_limit` 12:
/// the outer function body, the outer `add`, and the inner `add`. The expected
/// tab depths (1, 2, 3 for the innermost constants) verify
/// [`SExpressibleOptions::increase_indent`](crate::s_expr::SExpressibleOptions::increase_indent)
/// applied iteratively.
#[test]
fn test_recursive_wrap_increments_indent_per_level()
{
	assert_multiline(
		"1 + 2 + 3",
		12,
		"(function []\n\t(add\n\t\t(add\n\t\t\t1\n\t\t\t2)\n\t\t3))"
	);
}

/// An integer-face list wraps to one face per line when the enclosing
/// `custom-dice` form does not fit. At `soft_limit` 15, the body, the
/// `custom-dice`, and the faces vector all wrap, producing a cascade that pins
/// down the wrap branch of
/// [`<&[i32]>::write_s_expr`](crate::s_expr::SExpressible).
#[test]
fn test_custom_dice_faces_wrap_each_on_own_line()
{
	assert_multiline(
		"1d[1, 2, 3, 4, 5, 6, 7, 8]",
		15,
		"(function []\n\t(custom-dice\n\t\t1\n\t\t\
		 [\n\t\t\t1\n\t\t\t2\n\t\t\t3\n\t\t\t4\n\t\t\t5\n\t\t\t6\n\t\t\t7\n\t\t\
		 \t8\n\t\t]))"
	);
}

/// The writer emits `^[start end]` prefixes at every level when
/// [`with_spans`](crate::s_expr::SExpressibleOptions::with_spans) is enabled —
/// at the outer function, at the body, and at each constant. Under a tight
/// [`soft_limit`], wrapping interleaves with the prefixes without corrupting
/// their placement (each prefix sits outside its form's opening paren, with a
/// trailing space).
///
/// [`soft_limit`]: crate::s_expr::SExpressibleOptions::soft_limit
#[test]
fn test_multiline_with_spans_emits_prefixes_at_every_level()
{
	let ast = Parser::parse("1 + 2").unwrap();
	let opts = SExpressibleOptions::new(0, 4, 30).with_spans(true);
	let rendered = ast.to_s_expr(opts);
	assert_eq!(
		rendered,
		"^[0 5] (function []\n\t^[0 5] \
		 (add\n\t\t^[0 1] 1\n\t\t^[4 5] 2))"
	);
}

/// Lossless round-trip invariant under tight soft-limits: serializing every
/// parser-originated AST from `test_parse.txt` and reading it back must produce
/// an equivalent function even when the writer wraps across multiple lines.
/// This extends the default-limit coverage in
/// [`test_s_expr_lossless_roundtrip`] by forcing every wrap branch.
///
/// [`test_s_expr_lossless_roundtrip`]: crate::tests::parser::s_expr_format
#[test]
fn test_multiline_lossless_roundtrip_under_tight_limits()
{
	use crate::support::read_compilation_test_cases;

	let cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	// Representative limits: 10 is tight enough to wrap most compound forms; 30
	// wraps larger cases; 80 is the default.
	for soft_limit in [10usize, 30, 80]
	{
		let opts = SExpressibleOptions::new(0, 4, soft_limit)
			.with_spans(true)
			.with_groups(true);
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
			let reparsed = read_s_expr(&serialized).unwrap_or_else(|e| {
				panic!(
					"case {} at soft_limit {}: re-read failed for {:?}\n\
					 serialized: {}\nerror: {}",
					index + 1,
					soft_limit,
					source,
					serialized,
					e
				)
			});
			assert_eq!(
				reparsed,
				ast,
				"case {} at soft_limit {}: roundtrip mismatch for {:?}\n\
				 serialized: {}",
				index + 1,
				soft_limit,
				source,
				serialized
			);
		}
	}
}

/// When the function has no parameters (parser-originated functions with no
/// formal parameters produce `None`), the writer renders `[]` on the same line
/// as the keyword — the empty-list sizer returns 2 and the fits-check compares
/// against that. If the sizer were mutated to return 0 or 1, the body would
/// still fit on the line (bringing the overall length in under soft_limit by
/// the mutation's offset), but the space accounting would be one or two
/// characters short for the trailing close-paren. This test checks a
/// tight-budget case where the size prediction is load-bearing.
#[test]
fn test_function_with_no_parameters_reserves_two_for_empty_brackets()
{
	// At soft_limit 12 with an empty-params function, the body `(range 0 5)`
	// must wrap because 12 - 9 - 2 = 1 < 11. A mutation that drops the
	// empty-params size to 0 or 1 would change the remaining-space bookkeeping
	// and flip the wrap decision in observable ways; this test anchors the
	// expected layout.
	assert_multiline("[0:5]", 12, "(function []\n\t(range\n\t\t0\n\t\t5))");
}
