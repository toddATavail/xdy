//! # Validator tests
//!
//! Herein are the tests for the [validator](crate::validator) pass, which
//! performs semantic analysis on a parsed [AST](crate::ast) before code
//! generation: duplicate parameters, bindings that collide with parameters or
//! with each other, and uses before bind.

use pretty_assertions::assert_eq;

use super::ast::{Nesting, nest_function};
use crate::{
	CompilationError, EvaluationError, Parser, SourceSpan, Validator, compile,
	compile_unoptimized, evaluate, support::on_small_stack
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
	let ast = Parser::parse("{x}: {x} + 1").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a function with multiple distinct parameters.
#[test]
fn validate_multiple_distinct_parameters()
{
	let ast = Parser::parse("{x}, {y}, {z}: {x} + {y} + {z}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a parameter name with a repeated character, distinguishing
/// character-level repetition from whole-name repetition.
#[test]
fn validate_repeated_character_not_duplicate()
{
	let ast = Parser::parse("{xx}: {xx} + 1").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

////////////////////////////////////////////////////////////////////////////////
//                          Rejection — duplicates.                           //
////////////////////////////////////////////////////////////////////////////////

/// Validator rejects two identical single-character parameter names.
#[test]
fn validate_duplicate_single_char_parameter()
{
	let ast = Parser::parse("{x}, {x}: {x} + 1").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// Validator rejects duplicate multi-character identifiers and reports the full
/// span of each occurrence.
#[test]
fn validate_duplicate_multichar_parameter()
{
	let ast = Parser::parse("{abc}, {abc}: 1").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "abc".into(),
			first: SourceSpan { start: 1, end: 4 },
			duplicate: SourceSpan { start: 8, end: 11 }
		})
	);
}

/// Validator surfaces the first duplicate encountered when a later parameter
/// collides with an earlier one, ignoring intervening distinct names.
#[test]
fn validate_duplicate_after_distinct_parameter()
{
	let ast = Parser::parse("{a}, {b}, {a}: {a} + {b}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "a".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 11, end: 12 }
		})
	);
}

/// Validator reports the first pair of duplicates it encounters and does not
/// continue scanning after the first violation.
#[test]
fn validate_duplicate_reports_first_only()
{
	let ast = Parser::parse("{x}, {x}, {x}: {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// Validator remains correct when parameters are separated by extra
/// whitespace, because spans are computed from byte offsets in the original
/// source.
#[test]
fn validate_duplicate_with_whitespace()
{
	let ast = Parser::parse("{x},   {x}: {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 8, end: 9 }
		})
	);
}

/// Validator compares parameter names canonically: whitespace within a name
/// collapses to a single space, so `{a b}` and `{a\tb}` declare the same
/// parameter, and the error reports the canonical name, with the spans of the
/// names as written.
#[test]
fn validate_duplicate_canonical_parameter()
{
	let ast = Parser::parse("{a b}, {a\tb}: 1").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateParameter {
			name: "a b".into(),
			first: SourceSpan { start: 1, end: 4 },
			duplicate: SourceSpan { start: 8, end: 11 }
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                        Acceptance — local bindings.                        //
////////////////////////////////////////////////////////////////////////////////

/// Validator accepts a single binding used once after its declaration.
#[test]
fn validate_binding_single_use()
{
	let ast = Parser::parse("{x}@(3D6) + {x}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a binding used multiple times, all after its declaration.
#[test]
fn validate_binding_multiple_uses()
{
	let ast = Parser::parse("{x}@(3D6) + {x} + {x}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts two distinct bindings interleaved in arithmetic.
#[test]
fn validate_two_distinct_bindings()
{
	let ast = Parser::parse("{a}@(3D6) + {b}@(2D4) + {a} + {b}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a binding whose bound expression references an earlier
/// binding — nested bindings are legal because the namespace is flat.
#[test]
fn validate_binding_references_earlier_binding_in_rhs()
{
	let ast = Parser::parse("{a}@({b}@(3D6) + {b}) + {a}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator resolves a reference to a binding canonically, so a name broken
/// over lines refers to the binding of the same name written on one line, and
/// the compiler finds no external variable.
#[test]
fn validate_binding_referenced_canonically()
{
	let ast = Parser::parse("{a b}@(3D6) + {a\n   b} + {  a  b  }").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
	let function = compile_unoptimized("{a b}@(3D6) + {a\n   b}").unwrap();
	assert!(function.externals.is_empty());
}

/// Validator accepts a binding in a dice-count, faces, or drop-count position.
#[test]
fn validate_binding_in_restricted_positions()
{
	let ast_count = Parser::parse("{n}@(2+3)D6 + {n}").unwrap();
	assert_eq!(Validator::validate(&ast_count), Ok(()));
	let ast_faces = Parser::parse("4D{f}@(6) + {f}").unwrap();
	assert_eq!(Validator::validate(&ast_faces), Ok(()));
	let ast_drop = Parser::parse("4D6 drop lowest {k}@(2) + {k}").unwrap();
	assert_eq!(Validator::validate(&ast_drop), Ok(()));
}

////////////////////////////////////////////////////////////////////////////////
//                      Rejection — binding collisions.                       //
////////////////////////////////////////////////////////////////////////////////

/// A binding that reuses a formal-parameter name is rejected with spans
/// pointing at both the parameter declaration and the binding site.
#[test]
fn validate_binding_collides_with_parameter()
{
	let ast = Parser::parse("{x}: {x}@(3D6) + {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::BindingCollidesWithParameter {
			name: "x".into(),
			parameter: SourceSpan { start: 1, end: 2 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// Binding the same name twice within the same function body is rejected. The
/// span reported as `duplicate` is the second binding site.
#[test]
fn validate_duplicate_binding()
{
	let ast = Parser::parse("{x}@(3D6) + {x}@(1D4)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 13, end: 14 }
		})
	);
}

/// Binding a name within the bound expression of a binding of the same name is
/// rejected, with the outer binding as `first` and the inner one as
/// `duplicate`, whether or not the name is referenced afterward.
#[test]
fn validate_nested_duplicate_binding()
{
	for source in ["{x}@({x}@(1))", "{x}@({x}@(1)) + {x}"]
	{
		let ast = Parser::parse(source).unwrap();
		assert_eq!(
			Validator::validate(&ast),
			Err(CompilationError::DuplicateBinding {
				name: "x".into(),
				first: SourceSpan { start: 1, end: 2 },
				duplicate: SourceSpan { start: 6, end: 7 }
			}),
			"{}",
			source
		);
	}
}

/// A binding nested within a binding of a different name is accepted, and so
/// are references to either name after both bindings.
#[test]
fn validate_nested_distinct_bindings()
{
	let ast = Parser::parse("{x}@({y}@(1) + 1) + {x} + {y}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Referring to a binding before its lexical declaration is rejected, with the
/// primary span on the offending reference and the related span on the binding
/// site that appears later.
#[test]
fn validate_use_before_bind_simple()
{
	let ast = Parser::parse("{x} + {x}@(3D6)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 7, end: 8 }
		})
	);
}

/// A reference to a bound name inside that same binding's bound expression is
/// rejected as use-before-bind, because the binding is not in scope until after
/// its RHS has been evaluated. This is the sole mechanism by which
/// self-reference is rejected.
#[test]
fn validate_self_reference_inside_binding()
{
	let ast = Parser::parse("{x}@(1 + {x})").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 9, end: 12 },
			binding: SourceSpan { start: 1, end: 2 }
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
	let result = compile_unoptimized("{x}, {x}: 1");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `compile()` — the optimizing pipeline — also runs the validator and surfaces
/// duplicate-parameter errors without reaching the optimizer.
#[test]
fn compile_propagates_duplicate_parameter()
{
	let result = compile("{x}, {x}: 1");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `evaluate()` converts a [`CompilationError::DuplicateParameter`] into the
/// mirror variant on [`EvaluationError`] via the existing `From` impl.
#[test]
fn evaluate_propagates_duplicate_parameter()
{
	let mut rng = rand::rng();
	let result = evaluate("{x}, {x}: 1", vec![0, 0], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::DuplicateParameter {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 6, end: 7 }
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
		name: "x".into(),
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
		name: "x".into(),
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
	let ast = Parser::parse("{x}, {x}: 1").unwrap();
	let mut validator = Validator::new();
	let result = ast.accept(&mut validator);
	assert!(matches!(
		result,
		Err(CompilationError::DuplicateParameter { name, .. }) if name == "x"
	));
}

/// The [`Validator`]'s [`ASTVisitor`](crate::ast::ASTVisitor) implementation
/// accepts a semantically clean function that touches every node type the
/// visitor can encounter, driving each method at least once, so that any
/// future rewrite that rejects a well-formed node is caught.
#[test]
fn validator_visitor_accepts_every_node_type()
{
	// Build an AST that exercises every node type the visitor can encounter:
	// Group, Constant, Variable, Binding, Range, StandardDice, CustomDice,
	// DropLowest, DropHighest, Add, Sub, Mul, Div, Mod, Exp, and Neg. The
	// enclosing function is semantically clean, so the walk returns `Ok(())`.
	let ast = Parser::parse(
		"{x}: [1:{x}] + (1D6 - 1D[1,2,3]) * 2D6 drop lowest / \
		 3D8 drop highest + -1 ^ 2 % {y}@(1)"
	)
	.unwrap();
	let mut validator = Validator::new();
	assert_eq!(ast.accept(&mut validator), Ok(()));
}

/// Driving the [`Validator`] through the [`ASTVisitor`](crate::ast::ASTVisitor)
/// trait performs every check, not just the check of the parameters.
#[test]
fn validator_driven_through_the_trait_checks_bindings()
{
	let ast = Parser::parse("{x} + {x}@(3D6)").unwrap();
	let mut validator = Validator::new();
	assert_eq!(
		ast.accept(&mut validator),
		Err(CompilationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 7, end: 8 }
		})
	);
}

/// A [`Validator`] resets itself upon entering a function, so the bindings and
/// references of one function do not leak into the validation of the next.
#[test]
fn validator_can_be_reused()
{
	let first = Parser::parse("{x}@(1) + {y}").unwrap();
	let second = Parser::parse("{y}@(2) + {x}@(3) + {x}").unwrap();
	let mut validator = Validator::new();
	assert_eq!(first.accept(&mut validator), Ok(()));
	assert_eq!(second.accept(&mut validator), Ok(()));
}

////////////////////////////////////////////////////////////////////////////////
//                Additional local-binding validator coverage.                //
////////////////////////////////////////////////////////////////////////////////

/// Validator rejects a multi-character binding name duplicated in the same
/// function body, reporting spans covering the full identifier at each site.
#[test]
fn validate_duplicate_binding_multichar()
{
	let ast = Parser::parse("{abc}@(3D6) + {abc}@(1D4)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "abc".into(),
			first: SourceSpan { start: 1, end: 4 },
			duplicate: SourceSpan { start: 15, end: 18 }
		})
	);
}

/// When the same name is bound more than twice, the validator surfaces the
/// first pair and does not continue scanning after the first violation.
#[test]
fn validate_duplicate_binding_reports_first_pair_only()
{
	let ast = Parser::parse("{x}@(1) + {x}@(2) + {x}@(3)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 11, end: 12 }
		})
	);
}

/// A binding whose name collides with a later parameter in the parameter list
/// is rejected even when the colliding parameter appears after other distinct
/// parameters.
#[test]
fn validate_binding_collides_with_later_parameter()
{
	let ast = Parser::parse("{a}, {b}, {x}: {x}@(3D6) + {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::BindingCollidesWithParameter {
			name: "x".into(),
			parameter: SourceSpan { start: 11, end: 12 },
			binding: SourceSpan { start: 16, end: 17 }
		})
	);
}

/// Use-before-bind fires for a reference buried inside a restricted grammar
/// position (dice count), proving that the lexical-order walk descends into
/// every subexpression.
#[test]
fn validate_use_before_bind_in_dice_count()
{
	let ast = Parser::parse("{n}D6 + {n}@(3)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "n".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 9, end: 10 }
		})
	);
}

/// A binding whose RHS references a name that is itself a *later* binding in
/// the same function body is rejected as use-before-bind, proving that the
/// full-AST pre-pass over binding names successfully identifies the later
/// binding before the lexical walk reaches it.
#[test]
fn validate_use_before_bind_across_sibling_bindings()
{
	let ast = Parser::parse("{x}@({y}) + {y}@(1)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "y".into(),
			reference: SourceSpan { start: 5, end: 8 },
			binding: SourceSpan { start: 13, end: 14 }
		})
	);
}

/// A collision outranks a use before bind, even one that precedes it: the
/// self-reference within the first binding is reported only if the body binds
/// no name twice.
#[test]
fn validate_collision_outranks_earlier_use_before_bind()
{
	let ast = Parser::parse("{x}@({x}) + {x}@(1)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 13, end: 14 }
		})
	);
	let ast = Parser::parse("{p}: {x} + {p}@(1) + {x}@(2)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::BindingCollidesWithParameter {
			name: "p".into(),
			parameter: SourceSpan { start: 1, end: 2 },
			binding: SourceSpan { start: 12, end: 13 }
		})
	);
}

/// When several references precede their bindings, the first reference in
/// source order is reported, even if its binding comes last.
#[test]
fn validate_use_before_bind_reports_first_reference()
{
	let ast = Parser::parse("{y} + {x} + {x}@(1) + {y}@(2)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "y".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 23, end: 24 }
		})
	);
}

/// Names that are neither parameters nor bindings flow through validation
/// untouched — they become external variables when the compiler runs. A body
/// that mixes bindings, parameters, and externals validates cleanly.
#[test]
fn validate_mixed_parameter_binding_external()
{
	let ast = Parser::parse("{p}: {x}@(1D6) + {p} + {x} + {env}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

////////////////////////////////////////////////////////////////////////////////
//                    Pipeline propagation — local bindings.                  //
////////////////////////////////////////////////////////////////////////////////

/// `compile_unoptimized()` surfaces
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// errors from the validator without reaching the compiler.
#[test]
fn compile_unoptimized_propagates_binding_collides_with_parameter()
{
	let result = compile_unoptimized("{x}: {x}@(3D6) + {x}");
	assert_eq!(
		result.err(),
		Some(CompilationError::BindingCollidesWithParameter {
			name: "x".into(),
			parameter: SourceSpan { start: 1, end: 2 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `compile()` — the optimizing pipeline — also surfaces
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// without reaching the optimizer.
#[test]
fn compile_propagates_binding_collides_with_parameter()
{
	let result = compile("{x}: {x}@(3D6) + {x}");
	assert_eq!(
		result.err(),
		Some(CompilationError::BindingCollidesWithParameter {
			name: "x".into(),
			parameter: SourceSpan { start: 1, end: 2 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `evaluate()` converts a
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// to the mirror variant on [`EvaluationError`].
#[test]
fn evaluate_propagates_binding_collides_with_parameter()
{
	let mut rng = rand::rng();
	let result = evaluate("{x}: {x}@(3D6) + {x}", vec![0], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::BindingCollidesWithParameter {
			name: "x".into(),
			parameter: SourceSpan { start: 1, end: 2 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `compile_unoptimized()` surfaces
/// [`DuplicateBinding`](CompilationError::DuplicateBinding).
#[test]
fn compile_unoptimized_propagates_duplicate_binding()
{
	let result = compile_unoptimized("{x}@(3D6) + {x}@(1D4)");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 13, end: 14 }
		})
	);
}

/// `compile()` surfaces
/// [`DuplicateBinding`](CompilationError::DuplicateBinding).
#[test]
fn compile_propagates_duplicate_binding()
{
	let result = compile("{x}@(3D6) + {x}@(1D4)");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 13, end: 14 }
		})
	);
}

/// `evaluate()` converts
/// [`DuplicateBinding`](CompilationError::DuplicateBinding) to its mirror
/// variant.
#[test]
fn evaluate_propagates_duplicate_binding()
{
	let mut rng = rand::rng();
	let result = evaluate("{x}@(3D6) + {x}@(1D4)", vec![], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::DuplicateBinding {
			name: "x".into(),
			first: SourceSpan { start: 1, end: 2 },
			duplicate: SourceSpan { start: 13, end: 14 }
		})
	);
}

/// `compile_unoptimized()` surfaces
/// [`UseBeforeBind`](CompilationError::UseBeforeBind).
#[test]
fn compile_unoptimized_propagates_use_before_bind()
{
	let result = compile_unoptimized("{x} + {x}@(3D6)");
	assert_eq!(
		result.err(),
		Some(CompilationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 7, end: 8 }
		})
	);
}

/// `compile()` surfaces
/// [`UseBeforeBind`](CompilationError::UseBeforeBind).
#[test]
fn compile_propagates_use_before_bind()
{
	let result = compile("{x} + {x}@(3D6)");
	assert_eq!(
		result.err(),
		Some(CompilationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 7, end: 8 }
		})
	);
}

/// `evaluate()` converts [`UseBeforeBind`](CompilationError::UseBeforeBind)
/// to its mirror variant.
#[test]
fn evaluate_propagates_use_before_bind()
{
	let mut rng = rand::rng();
	let result = evaluate("{x} + {x}@(3D6)", vec![], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::UseBeforeBind {
			name: "x".into(),
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 7, end: 8 }
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//               Error formatting — local bindings.                           //
////////////////////////////////////////////////////////////////////////////////

/// The [`Display`](std::fmt::Display) rendering of
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// names the offending identifier and cites both the binding and parameter
/// spans.
#[test]
fn binding_collides_with_parameter_display()
{
	let err = CompilationError::BindingCollidesWithParameter {
		name: "x".into(),
		parameter: SourceSpan { start: 0, end: 1 },
		binding: SourceSpan { start: 3, end: 4 }
	};
	assert_eq!(
		format!("{}", err),
		"local binding 'x' at 3..4 collides with formal parameter declared at 0..1"
	);
}

/// The [`EvaluationError`] mirror variant of
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// renders identically.
#[test]
fn binding_collides_with_parameter_display_on_evaluation_error()
{
	let err = EvaluationError::BindingCollidesWithParameter {
		name: "x".into(),
		parameter: SourceSpan { start: 0, end: 1 },
		binding: SourceSpan { start: 3, end: 4 }
	};
	assert_eq!(
		format!("{}", err),
		"local binding 'x' at 3..4 collides with formal parameter declared at 0..1"
	);
}

/// The [`Display`](std::fmt::Display) rendering of
/// [`DuplicateBinding`](CompilationError::DuplicateBinding) names the rebound
/// identifier and cites both binding-site spans.
#[test]
fn duplicate_binding_display()
{
	let err = CompilationError::DuplicateBinding {
		name: "x".into(),
		first: SourceSpan { start: 0, end: 1 },
		duplicate: SourceSpan { start: 10, end: 11 }
	};
	assert_eq!(
		format!("{}", err),
		"duplicate local binding 'x' at 10..11 (first bound at 0..1)"
	);
}

/// The [`EvaluationError`] mirror variant of
/// [`DuplicateBinding`](CompilationError::DuplicateBinding) renders
/// identically.
#[test]
fn duplicate_binding_display_on_evaluation_error()
{
	let err = EvaluationError::DuplicateBinding {
		name: "x".into(),
		first: SourceSpan { start: 0, end: 1 },
		duplicate: SourceSpan { start: 10, end: 11 }
	};
	assert_eq!(
		format!("{}", err),
		"duplicate local binding 'x' at 10..11 (first bound at 0..1)"
	);
}

/// The [`Display`](std::fmt::Display) rendering of
/// [`UseBeforeBind`](CompilationError::UseBeforeBind) names the offending
/// identifier and cites both the reference and binding spans.
#[test]
fn use_before_bind_display()
{
	let err = CompilationError::UseBeforeBind {
		name: "x".into(),
		reference: SourceSpan { start: 0, end: 3 },
		binding: SourceSpan { start: 6, end: 7 }
	};
	assert_eq!(
		format!("{}", err),
		"reference to 'x' at 0..3 precedes its binding at 6..7"
	);
}

/// The [`EvaluationError`] mirror variant of
/// [`UseBeforeBind`](CompilationError::UseBeforeBind) renders identically.
#[test]
fn use_before_bind_display_on_evaluation_error()
{
	let err = EvaluationError::UseBeforeBind {
		name: "x".into(),
		reference: SourceSpan { start: 0, end: 3 },
		binding: SourceSpan { start: 6, end: 7 }
	};
	assert_eq!(
		format!("{}", err),
		"reference to 'x' at 0..3 precedes its binding at 6..7"
	);
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the validator survives a deep chain of every nesting construct
/// on a small stack. The chains that nest bindings named `a` inside one another
/// are rejected as duplicate bindings.
#[test]
#[ignore = "stress: run with just stress"]
fn test_validate_deep()
{
	on_small_stack(|| {
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			let result =
				Validator::validate(&nest_function(nesting, nesting.depth()));
			if nesting.binds()
			{
				assert!(
					matches!(
						&result,
						Err(CompilationError::DuplicateBinding { name, .. })
							if name == "a"
					),
					"{:?}: {:?}",
					nesting,
					result
				);
			}
			else
			{
				assert_eq!(result, Ok(()), "{:?}", nesting);
			}
		}
	});
}
