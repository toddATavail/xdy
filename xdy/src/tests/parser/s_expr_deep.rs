//! # Deep and wide S-expression reader test cases
//!
//! Herein are the tests that hold [`read_s_expr`] to its promise of constant
//! stack depth and linear time. Every [nesting construct](Nesting) is parsed
//! at its [depth](Nesting::depth), written as an S-expression with spans and
//! groups on one line, and read back on a [small stack](on_small_stack), both
//! as a valid S-expression and as one that fails at its innermost leaf.
//! Functions with [very many](WIDTH) parameters and dice with very many faces
//! are read back the same way. The default time budget of the small stack also
//! bounds the read time, so a reader that took time quadratic in the depth or
//! the width would not finish. The deep reads are ignored by default; `just
//! stress` runs them.

use crate::{
	Parser,
	ast::{DiceExpression, Expression, Function},
	s_expr::{
		SExprError, SExprLocation, SExpressible, SExpressibleOptions,
		read_s_expr
	},
	support::on_small_stack,
	tests::ast::{Nesting, nest_parsable}
};

////////////////////////////////////////////////////////////////////////////////
//                                Deep reads.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a deep chain of groups reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_group() { read_deep(Nesting::Group) }

/// Ensure that a deep chain of bindings reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_binding() { read_deep(Nesting::Binding) }

/// Ensure that a deep chain of range starts reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_range_start() { read_deep(Nesting::RangeStart) }

/// Ensure that a deep chain of range ends reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_range_end() { read_deep(Nesting::RangeEnd) }

/// Ensure that a deep chain of negations reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_negation() { read_deep(Nesting::Negation) }

/// Ensure that a deep chain of exponents reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_exponent() { read_deep(Nesting::Exponent) }

/// Ensure that a deep chain of additions reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_addition() { read_deep(Nesting::Addition) }

/// Ensure that a deep chain of subtractions reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_subtraction() { read_deep(Nesting::Subtraction) }

/// Ensure that a deep chain of multiplications reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_multiplication() { read_deep(Nesting::Multiplication) }

/// Ensure that a deep chain of divisions reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_division() { read_deep(Nesting::Division) }

/// Ensure that a deep chain of modulos reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_modulo() { read_deep(Nesting::Modulo) }

/// Ensure that a deep chain of dice counts reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_dice_count() { read_deep(Nesting::DiceCount) }

/// Ensure that a deep chain of dice faces reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_dice_faces() { read_deep(Nesting::DiceFaces) }

/// Ensure that a deep chain of custom dice counts reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_custom_count() { read_deep(Nesting::CustomCount) }

/// Ensure that a deep chain of drop expressions reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_drop_expression() { read_deep(Nesting::DropExpression) }

/// Ensure that a deep stack of drop clauses reads on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_drop_clauses() { read_deep(Nesting::DropClauses) }

/// Ensure that a deep mixture of every nesting construct reads on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_read_deep_mixed() { read_deep(Nesting::Mixed) }

/// The value of the innermost constant of a deep source. Its rendering occurs
/// nowhere else in the S-expression, so that [`read_deep`] can find it and
/// replace it. It has ten digits, more than any byte offset in a span prefix
/// can have, so no prefix can contain it.
const LEAF: i32 = 1_234_567_890;

/// An integer that overflows [`i32`], and so cannot be read as a constant.
const OVERFLOW: &str = "99999999999";

/// Parse a [deep](Nesting::depth) source, and ensure that its S-expression
/// reads back to it on a small stack. Then replace its innermost constant with
/// an [integer that overflows](OVERFLOW), and ensure that reading fails there
/// with [`InvalidInteger`](SExprError::InvalidInteger).
///
/// # Parameters
/// - `nesting`: The way to nest.
fn read_deep(nesting: Nesting)
{
	on_small_stack(|| {
		let source = nest_parsable(nesting, nesting.depth(), LEAF).to_string();
		let function = Parser::parse(&source).expect("deep source failed");
		let s_expr = round_trip(&function);
		drop(function);

		let leaf = LEAF.to_string();
		assert_eq!(s_expr.matches(&leaf).count(), 1, "{:?}", nesting);
		let offset = s_expr.find(&leaf).unwrap();
		let failing = s_expr.replacen(&leaf, OVERFLOW, 1);
		let error = read_s_expr(&failing).expect_err("failing input read");
		assert_eq!(
			error,
			SExprError::InvalidInteger {
				text: OVERFLOW.to_string(),
				reason: OVERFLOW.parse::<i32>().unwrap_err().to_string(),
				location: SExprLocation {
					offset,
					line: 1,
					column: offset + 1
				}
			},
			"{:?}",
			nesting
		);
	});
}

////////////////////////////////////////////////////////////////////////////////
//                               Wrapped reads.                               //
////////////////////////////////////////////////////////////////////////////////

/// The depth of the wrapped reads. Each level of a wrapped S-expression
/// indents by a tab, so its length is quadratic in the depth.
const WRAPPED_DEPTH: usize = 2_000;

/// Ensure that every nesting construct, [`WRAPPED_DEPTH`] levels deep, writes
/// as a wrapped S-expression that reads back to it. At this depth, nearly every
/// form wraps under the default soft limit, and the indentation of the
/// innermost forms far exceeds it.
#[test]
fn test_read_deep_wrapped()
{
	for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
	{
		let source = nest_parsable(nesting, WRAPPED_DEPTH, 1).to_string();
		let function = Parser::parse(&source).expect("deep source failed");
		let options = SExpressibleOptions::default()
			.with_spans(true)
			.with_groups(true);
		let s_expr = function.to_s_expr(options);
		assert!(s_expr.lines().count() > WRAPPED_DEPTH, "{:?}", nesting);
		let read = read_s_expr(&s_expr).expect("S-expression failed to read");
		assert!(read == function, "{:?}: read back differently", nesting);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Wide reads.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The number of parameters or faces in a wide source.
const WIDTH: usize = 1_000_000;

/// Ensure that a function with very many parameters reads on a small stack.
#[test]
fn test_read_wide_parameters()
{
	on_small_stack(|| {
		let parameters = (0..WIDTH)
			.map(|i| format!("{{p{}}}", i))
			.collect::<Vec<_>>()
			.join(", ");
		let source = format!("{}: 1", parameters);
		let function = Parser::parse(&source).expect("wide source failed");
		assert_eq!(function.parameters.as_ref().map_or(0, Vec::len), WIDTH);
		round_trip(&function);
	});
}

/// Ensure that custom dice with very many faces read on a small stack.
#[test]
fn test_read_wide_faces()
{
	on_small_stack(|| {
		let faces = (1..=WIDTH)
			.map(|i| i.to_string())
			.collect::<Vec<_>>()
			.join(", ");
		let source = format!("1D[{}]", faces);
		let function = Parser::parse(&source).expect("wide source failed");
		match &function.body
		{
			Expression::Dice(DiceExpression::Custom(dice)) =>
			{
				assert_eq!(dice.faces.len(), WIDTH)
			},
			_ => panic!("wide source did not parse as custom dice")
		}
		round_trip(&function);
	});
}

////////////////////////////////////////////////////////////////////////////////
//                                  Helpers.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Write the S-expression of a parsed function with spans and groups on one
/// line, and ensure that it reads back to the function, spans and all. Uses
/// `assert!` rather than `assert_eq!` on functions, since a failure would
/// otherwise try to render them in full.
///
/// # Parameters
/// - `function`: The parsed function.
///
/// # Returns
/// The S-expression.
///
/// # Panics
/// If the S-expression does not read back to the function.
fn round_trip(function: &Function<'_>) -> String
{
	let options = SExpressibleOptions::new(0, 4, usize::MAX)
		.with_spans(true)
		.with_groups(true);
	let s_expr = function.to_s_expr(options);
	let read = read_s_expr(&s_expr).expect("S-expression failed to read");
	assert!(read == *function, "S-expression read back differently");
	drop(read);
	s_expr
}
