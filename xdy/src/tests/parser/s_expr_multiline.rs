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
		"{alpha}, {beta}, {gamma}: 42",
		15,
		"(function\n\t[\n\t\t{alpha}\n\t\t{beta}\n\t\t{gamma}\n\t] 42)"
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

/// At `soft_limit` 14, the body `(neg {ab})` wraps to indent 1 and its single
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
	assert_multiline("-{ab}", 14, "(function []\n\t(neg\n\t\t{ab}))");
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
	// must wrap because 12 - 9 - 1 - 2 = 0 < 1 + 11 + 1. A mutation that drops
	// the empty-params size to 0 or 1 would change the remaining-space
	// bookkeeping and flip the wrap decision in observable ways; this test
	// anchors the expected layout.
	assert_multiline("[0:5]", 12, "(function []\n\t(range\n\t\t0\n\t\t5))");
}

////////////////////////////////////////////////////////////////////////////////
//                           Soft-limit boundaries.                           //
////////////////////////////////////////////////////////////////////////////////

/// Render `source` with the given `soft_limit` and assert that the result is
/// exactly `expected`, and that no line of it exceeds the soft limit. Tabs are
/// four columns wide.
fn assert_within_limit(source: &str, soft_limit: usize, expected: &str)
{
	let ast = Parser::parse(source)
		.unwrap_or_else(|e| panic!("parse failed for {:?}: {}", source, e));
	let opts = SExpressibleOptions::new(0, 4, soft_limit);
	let actual = ast.to_s_expr(opts);
	assert_eq!(
		actual, expected,
		"layout mismatch for {:?} at soft_limit {}",
		source, soft_limit
	);
	for line in actual.lines()
	{
		assert!(
			line_width(line, 4) <= soft_limit,
			"line exceeds soft_limit {} for {:?}: {:?}",
			soft_limit,
			source,
			line
		);
	}
}

/// Measure a line of S-expression output, counting each leading tab as
/// `tab_width` columns and every other character as one.
///
/// # Parameters
/// - `line`: The line, without its line terminator.
/// - `tab_width`: The number of columns per tab.
///
/// # Returns
/// The width of the line, in columns.
fn line_width(line: &str, tab_width: usize) -> usize
{
	let text = line.trim_start_matches('\t');
	tab_width * (line.len() - text.len()) + text.chars().count()
}

/// A function fits on one line when its whole width does: here, exactly 23
/// columns.
#[test]
fn test_function_fits_exactly_at_soft_limit()
{
	assert_within_limit("1 + 2", 23, "(function [] (add 1 2))");
}

/// A function that is one column too wide wraps its body. The function must
/// count the spaces before its parameters and its body, and its closing
/// parenthesis, which once let it overrun the soft limit by three columns.
#[test]
fn test_function_wraps_one_column_past_soft_limit()
{
	assert_within_limit("1 + 2", 22, "(function []\n\t(add 1 2))");
}

/// When the parameters wrap onto their own line, the body may follow them there
/// only if it fits in what remains of that line: here, exactly 24 columns.
#[test]
fn test_body_fits_exactly_after_wrapped_parameters()
{
	assert_within_limit(
		"{alpha}, {beta}: 42",
		24,
		"(function\n\t[{alpha} {beta}] 42)"
	);
}

/// When the parameters wrap onto their own line and the body does not fit after
/// them, the body wraps too. The function once budgeted the body a whole fresh
/// line, ignoring the parameters already on it.
#[test]
fn test_body_wraps_after_wrapped_parameters()
{
	assert_within_limit(
		"{alpha}, {beta}: 420",
		24,
		"(function\n\t[{alpha} {beta}]\n\t420)"
	);
}

/// When the parameters split across lines, the body may follow the closing
/// bracket only if it fits in what remains of that line: here, exactly 17
/// columns.
#[test]
fn test_body_fits_exactly_after_split_parameters()
{
	assert_within_limit(
		"{alpha}, {beta}, {gamma}: -{ab}",
		17,
		"(function\n\t[\n\t\t{alpha}\n\t\t{beta}\n\t\t{gamma}\n\t] (neg {ab}))"
	);
}

/// When the parameters split across lines and the body does not fit after the
/// closing bracket, the body wraps.
#[test]
fn test_body_wraps_after_split_parameters()
{
	assert_within_limit(
		"{alpha}, {beta}, {gamma}: -{abc}",
		17,
		"(function\n\t[\n\t\t{alpha}\n\t\t{beta}\n\t\t{gamma}\n\t]\n\t(neg {abc}))"
	);
}

/// A parameter list reserves room for the closing parentheses of enclosing
/// forms, as the keyword forms do, so a list that exactly fills its line
/// splits.
#[test]
fn test_parameter_list_reserves_enclosing_parentheses()
{
	assert_within_limit(
		"{alpha}, {beta}, {gam}: 1",
		26,
		"(function\n\t[\n\t\t{alpha}\n\t\t{beta}\n\t\t{gam}\n\t] 1)"
	);
}

/// A binding fits on one line when its whole width does, including its closing
/// parenthesis and those of enclosing forms: here, exactly 20 columns.
#[test]
fn test_binding_fits_exactly_at_soft_limit()
{
	assert_within_limit("{x}@(1)", 20, "(function []\n\t(binding {x} 1))");
}

/// A binding that is one column too wide wraps its name and expression. The
/// binding once omitted its closing parenthesis from its budget.
#[test]
fn test_binding_wraps_one_column_past_soft_limit()
{
	assert_within_limit(
		"{x}@(1)",
		19,
		"(function []\n\t(binding\n\t\t{x}\n\t\t1))"
	);
}

/// A face list fits on one line when its whole width does, including the
/// closing parentheses of enclosing forms: here, exactly 17 columns.
#[test]
fn test_face_list_fits_exactly_at_soft_limit()
{
	assert_within_limit(
		"1d[1, 2, 3]",
		17,
		"(function []\n\t(custom-dice\n\t\t1\n\t\t[1 2 3]))"
	);
}

/// A face list that is one column too wide splits. The list once ignored the
/// closing parentheses of enclosing forms.
#[test]
fn test_face_list_wraps_one_column_past_soft_limit()
{
	assert_within_limit(
		"1d[1, 2, 3]",
		16,
		"(function []\n\t(custom-dice\n\t\t1\n\t\t[\n\t\t\t1\n\t\t\t2\n\t\t\t3\n\t\t]))"
	);
}

/// When a unary form wraps its operand onto a new line, the operand lays out
/// its own subexpressions from that line's indentation. Here the `add` fits
/// exactly at indent 2, with the closing parentheses of `neg` and `function`.
#[test]
fn test_wrapped_unary_operand_fits_exactly_at_soft_limit()
{
	assert_within_limit(
		"-(1 + 2)",
		19,
		"(function []\n\t(neg\n\t\t(add 1 2)))"
	);
}

/// When a unary form wraps its operand onto a new line and the operand is one
/// column too wide, the operand wraps its own subexpressions one level deeper
/// still. The unary form once passed its own indentation to the operand, so
/// the operand's subexpressions landed at the operand's level, and its budget
/// omitted the unary form's closing parenthesis.
#[test]
fn test_wrapped_unary_operand_wraps_one_level_deeper()
{
	assert_within_limit(
		"-(1 + 2)",
		18,
		"(function []\n\t(neg\n\t\t(add\n\t\t\t1\n\t\t\t2)))"
	);
}

/// No line of output exceeds the soft limit unless it is a single token that
/// cannot be split: an atom, a keyword, or a bracket, perhaps with closing
/// parentheses. Checked for every case in `test_parse.txt`, at every soft limit
/// from 0 to 100, with groups both transparent and opaque. Spans are off, since
/// a span prefix and its atom cannot be split either, but contain a space.
#[test]
fn test_lines_exceed_soft_limit_only_when_unsplittable()
{
	use crate::support::read_compilation_test_cases;

	let cases = read_compilation_test_cases(include_str!(
		"../../../tests/test_parse.txt"
	));
	for with_groups in [false, true]
	{
		for soft_limit in 0..=100
		{
			let opts = SExpressibleOptions::new(0, 4, soft_limit)
				.with_groups(with_groups);
			for (source, _) in &cases
			{
				let ast = Parser::parse(source).unwrap();
				let rendered = ast.to_s_expr(opts);
				for line in rendered.lines()
				{
					assert!(
						line_width(line, 4) <= soft_limit
							|| is_single_token(line),
						"line exceeds soft_limit {} (with_groups {}) for \
						 {:?}: {:?}\nrendered:\n{}",
						soft_limit,
						with_groups,
						source,
						line,
						rendered
					);
				}
			}
		}
	}
}

/// Answer whether a line of S-expression output holds a single token, i.e.,
/// has no space outside of an identifier once its indentation is removed.
///
/// # Parameters
/// - `line`: The line, without its line terminator.
///
/// # Returns
/// `true` if the line holds a single token, `false` otherwise.
fn is_single_token(line: &str) -> bool
{
	let mut braced = false;
	for c in line.trim_start_matches('\t').chars()
	{
		match c
		{
			'{' => braced = true,
			'}' => braced = false,
			' ' if !braced => return false,
			_ =>
			{}
		}
	}
	true
}
