//! # Validator tests
//!
//! Herein are the tests for the [validator](crate::validator) pass, which
//! performs semantic analysis on a parsed [AST](crate::ast) before code
//! generation. The current concrete check is **duplicate parameter names**.

use pretty_assertions::assert_eq;

use crate::{
	CompilationError, EvaluationError, Parser, SourceSpan, Validator, compile,
	compile_unoptimized, evaluate
};

////////////////////////////////////////////////////////////////////////////////
//                         Acceptance — well-formed.                          //
////////////////////////////////////////////////////////////////////////////////

/// Validator accepts a function with no parameters.
#[test]
fn validate_no_parameters()
{
	let ast = Parser::parse("1D6 + 2").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a function with a single parameter.
#[test]
fn validate_single_parameter()
{
	let ast = Parser::parse("x: {x} + 1").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a function with multiple distinct parameters.
#[test]
fn validate_multiple_distinct_parameters()
{
	let ast = Parser::parse("x, y, z: {x} + {y} + {z}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a parameter name with a repeated character, distinguishing
/// character-level repetition from whole-name repetition.
#[test]
fn validate_repeated_character_not_duplicate()
{
	let ast = Parser::parse("xx: {xx} + 1").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

////////////////////////////////////////////////////////////////////////////////
//                          Rejection — duplicates.                           //
////////////////////////////////////////////////////////////////////////////////

/// Validator rejects two identical single-character parameter names.
#[test]
fn validate_duplicate_single_char_parameter()
{
	let ast = Parser::parse("x, x: {x} + 1").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// Validator rejects duplicate multi-character identifiers and reports the full
/// span of each occurrence.
#[test]
fn validate_duplicate_multichar_parameter()
{
	let ast = Parser::parse("abc, abc: 1").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "abc",
			first: SourceSpan { start: 0, end: 3 },
			duplicate: SourceSpan { start: 5, end: 8 }
		})
	);
}

/// Validator surfaces the first duplicate encountered when a later parameter
/// collides with an earlier one, ignoring intervening distinct names.
#[test]
fn validate_duplicate_after_distinct_parameter()
{
	let ast = Parser::parse("a, b, a: {a} + {b}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "a",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// Validator reports the first pair of duplicates it encounters and does not
/// continue scanning after the first violation.
#[test]
fn validate_duplicate_reports_first_only()
{
	let ast = Parser::parse("x, x, x: {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// Validator remains correct when parameters are separated by extra
/// whitespace, because spans are computed from byte offsets in the original
/// source.
#[test]
fn validate_duplicate_with_whitespace()
{
	let ast = Parser::parse("x,   x: {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 5, end: 6 }
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                           Pipeline propagation.                            //
////////////////////////////////////////////////////////////////////////////////

/// `compile_unoptimized()` runs the validator between parsing and code
/// generation and surfaces duplicate-parameter errors to the caller.
#[test]
fn compile_unoptimized_propagates_duplicate_parameter()
{
	let result = compile_unoptimized("x, x: 1");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// `compile()` — the optimizing pipeline — also runs the validator and surfaces
/// duplicate-parameter errors without reaching the optimizer.
#[test]
fn compile_propagates_duplicate_parameter()
{
	let result = compile("x, x: 1");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// `evaluate()` converts a [`CompilationError::DuplicateParameter`] into the
/// mirror variant on [`EvaluationError`] via the existing `From` impl.
#[test]
fn evaluate_propagates_duplicate_parameter()
{
	let mut rng = rand::rng();
	let result = evaluate("x, x: 1", vec![0, 0], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::DuplicateParameter {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 3, end: 4 }
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                             Error formatting.                              //
////////////////////////////////////////////////////////////////////////////////

/// The [`Display`](std::fmt::Display) rendering of
/// [`DuplicateParameter`](CompilationError::DuplicateParameter) names the
/// offending identifier and cites both occurrences by byte range.
#[test]
fn duplicate_parameter_display()
{
	let err = CompilationError::DuplicateParameter {
		name: "x",
		first: SourceSpan { start: 0, end: 1 },
		duplicate: SourceSpan { start: 3, end: 4 }
	};
	assert_eq!(
		format!("{}", err),
		"duplicate parameter 'x' at 3..4 (first declared at 0..1)"
	);
}

/// The [`EvaluationError`] mirror variant renders identically to the
/// [`CompilationError`] variant.
#[test]
fn duplicate_parameter_display_on_evaluation_error()
{
	let err = EvaluationError::DuplicateParameter {
		name: "x",
		first: SourceSpan { start: 0, end: 1 },
		duplicate: SourceSpan { start: 3, end: 4 }
	};
	assert_eq!(
		format!("{}", err),
		"duplicate parameter 'x' at 3..4 (first declared at 0..1)"
	);
}

////////////////////////////////////////////////////////////////////////////////
//                        ASTVisitor trait-driven use.                        //
////////////////////////////////////////////////////////////////////////////////

/// Callers may drive the [`Validator`] directly through the
/// [`ASTVisitor`](crate::ast::ASTVisitor) trait.
#[test]
fn validator_can_be_driven_through_the_trait()
{
	use crate::ast::ASTVisitor;
	let ast = Parser::parse("x, x: 1").unwrap();
	let mut validator = Validator::new();
	let result = validator.visit_function(&ast);
	assert!(matches!(
		result,
		Err(CompilationError::DuplicateParameter { name: "x", .. })
	));
}
