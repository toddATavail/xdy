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
//                        Acceptance — local bindings.                        //
////////////////////////////////////////////////////////////////////////////////

/// Validator accepts a single binding used once after its declaration.
#[test]
fn validate_binding_single_use()
{
	let ast = Parser::parse("x@(3D6) + {x}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a binding used multiple times, all after its declaration.
#[test]
fn validate_binding_multiple_uses()
{
	let ast = Parser::parse("x@(3D6) + {x} + {x}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts two distinct bindings interleaved in arithmetic.
#[test]
fn validate_two_distinct_bindings()
{
	let ast = Parser::parse("a@(3D6) + b@(2D4) + {a} + {b}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a binding whose bound expression references an earlier
/// binding — nested bindings are legal because the namespace is flat.
#[test]
fn validate_binding_references_earlier_binding_in_rhs()
{
	let ast = Parser::parse("a@(b@(3D6) + {b}) + {a}").unwrap();
	assert_eq!(Validator::validate(&ast), Ok(()));
}

/// Validator accepts a binding in a dice-count, faces, or drop-count position.
#[test]
fn validate_binding_in_restricted_positions()
{
	let ast_count = Parser::parse("n@(2+3)D6 + {n}").unwrap();
	assert_eq!(Validator::validate(&ast_count), Ok(()));
	let ast_faces = Parser::parse("4Df@(6) + {f}").unwrap();
	assert_eq!(Validator::validate(&ast_faces), Ok(()));
	let ast_drop = Parser::parse("4D6 drop lowest k@(2) + {k}").unwrap();
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
	let ast = Parser::parse("x: x@(3D6) + {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::BindingCollidesWithParameter {
			name: "x",
			parameter: SourceSpan { start: 0, end: 1 },
			binding: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// Binding the same name twice within the same function body is rejected. The
/// span reported as `duplicate` is the second binding site.
#[test]
fn validate_duplicate_binding()
{
	let ast = Parser::parse("x@(3D6) + x@(1D4)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 10, end: 11 }
		})
	);
}

/// Referring to a binding before its lexical declaration is rejected, with the
/// primary span on the offending reference and the related span on the binding
/// site that appears later.
#[test]
fn validate_use_before_bind_simple()
{
	let ast = Parser::parse("{x} + x@(3D6)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "x",
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 6, end: 7 }
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
	let ast = Parser::parse("x@(1 + {x})").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "x",
			reference: SourceSpan { start: 7, end: 10 },
			binding: SourceSpan { start: 0, end: 1 }
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

/// Every non-[`visit_function`](crate::ast::ASTVisitor::visit_function) arm of
/// the [`Validator`]'s [`ASTVisitor`] implementation is a no-op that returns
/// `Ok(())`. This test walks an AST that touches every node type the visitor
/// can encounter, driving each arm at least once, so that any future rewrite
/// that silently turns a `Ok(())` arm into something else is caught.
#[test]
fn validator_visitor_arms_are_no_ops()
{
	use crate::ast::{
		ASTVisitor, ArithmeticExpression, DiceExpression, Expression
	};

	// Build an AST that exercises every node type the visitor can encounter:
	// Group, Constant, Variable, Range, StandardDice, CustomDice, DropLowest,
	// DropHighest, Add, Sub, Mul, Div, Mod, Exp, and Neg. The enclosing
	// function is semantically clean, so `visit_function` returns `Ok(())`.
	let ast = Parser::parse(
		"x: [1:{x}] + (1D6 - 1D[1,2,3]) * 2D6 drop lowest / \
		 3D8 drop highest + -1 ^ 2 % 1"
	)
	.unwrap();
	let mut validator = Validator::new();

	assert_eq!(validator.visit_function(&ast), Ok(()));

	// Additionally drive every non-function arm directly by walking the AST.
	// The walkers recurse into every child so that every reachable node is
	// handed to its corresponding visitor method at least once.
	fn drive_expression<'src>(
		validator: &mut Validator,
		expr: &'src Expression<'src>
	)
	{
		match expr
		{
			Expression::Group(g) =>
			{
				assert_eq!(validator.visit_group(g), Ok(()));
				drive_expression(validator, &g.expression);
			},
			Expression::Constant(c) =>
			{
				assert_eq!(validator.visit_constant(c), Ok(()));
			},
			Expression::Variable(v) =>
			{
				assert_eq!(validator.visit_variable(v), Ok(()));
			},
			Expression::Binding(b) =>
			{
				assert_eq!(validator.visit_binding(b), Ok(()));
				drive_expression(validator, &b.expression);
			},
			Expression::Range(r) =>
			{
				assert_eq!(validator.visit_range(r), Ok(()));
				drive_expression(validator, &r.start);
				drive_expression(validator, &r.end);
			},
			Expression::Dice(d) => drive_dice(validator, d),
			Expression::Arithmetic(a) => drive_arithmetic(validator, a)
		}
	}

	fn drive_dice<'src>(
		validator: &mut Validator,
		dice: &'src DiceExpression<'src>
	)
	{
		match dice
		{
			DiceExpression::Standard(d) =>
			{
				assert_eq!(validator.visit_standard_dice(d), Ok(()));
				drive_expression(validator, &d.count);
				drive_expression(validator, &d.faces);
			},
			DiceExpression::Custom(d) =>
			{
				assert_eq!(validator.visit_custom_dice(d), Ok(()));
				drive_expression(validator, &d.count);
			},
			DiceExpression::DropLowest(d) =>
			{
				assert_eq!(validator.visit_drop_lowest(d), Ok(()));
				drive_dice(validator, &d.dice);
				if let Some(drop) = &d.drop
				{
					drive_expression(validator, drop);
				}
			},
			DiceExpression::DropHighest(d) =>
			{
				assert_eq!(validator.visit_drop_highest(d), Ok(()));
				drive_dice(validator, &d.dice);
				if let Some(drop) = &d.drop
				{
					drive_expression(validator, drop);
				}
			}
		}
	}

	fn drive_arithmetic<'src>(
		validator: &mut Validator,
		arith: &'src ArithmeticExpression<'src>
	)
	{
		match arith
		{
			ArithmeticExpression::Add(a) =>
			{
				assert_eq!(validator.visit_add(a), Ok(()));
				drive_expression(validator, &a.left);
				drive_expression(validator, &a.right);
			},
			ArithmeticExpression::Sub(s) =>
			{
				assert_eq!(validator.visit_sub(s), Ok(()));
				drive_expression(validator, &s.left);
				drive_expression(validator, &s.right);
			},
			ArithmeticExpression::Mul(m) =>
			{
				assert_eq!(validator.visit_mul(m), Ok(()));
				drive_expression(validator, &m.left);
				drive_expression(validator, &m.right);
			},
			ArithmeticExpression::Div(d) =>
			{
				assert_eq!(validator.visit_div(d), Ok(()));
				drive_expression(validator, &d.left);
				drive_expression(validator, &d.right);
			},
			ArithmeticExpression::Mod(m) =>
			{
				assert_eq!(validator.visit_mod(m), Ok(()));
				drive_expression(validator, &m.left);
				drive_expression(validator, &m.right);
			},
			ArithmeticExpression::Exp(e) =>
			{
				assert_eq!(validator.visit_exp(e), Ok(()));
				drive_expression(validator, &e.left);
				drive_expression(validator, &e.right);
			},
			ArithmeticExpression::Neg(n) =>
			{
				assert_eq!(validator.visit_neg(n), Ok(()));
				drive_expression(validator, &n.operand);
			}
		}
	}

	drive_expression(&mut validator, &ast.body);
}

////////////////////////////////////////////////////////////////////////////////
//                Additional local-binding validator coverage.                //
////////////////////////////////////////////////////////////////////////////////

/// Validator rejects a multi-character binding name duplicated in the same
/// function body, reporting spans covering the full identifier at each site.
#[test]
fn validate_duplicate_binding_multichar()
{
	let ast = Parser::parse("abc@(3D6) + abc@(1D4)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "abc",
			first: SourceSpan { start: 0, end: 3 },
			duplicate: SourceSpan { start: 12, end: 15 }
		})
	);
}

/// When the same name is bound more than twice, the validator surfaces the
/// first pair and does not continue scanning after the first violation.
#[test]
fn validate_duplicate_binding_reports_first_pair_only()
{
	let ast = Parser::parse("x@(1) + x@(2) + x@(3)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::DuplicateBinding {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 8, end: 9 }
		})
	);
}

/// A binding whose name collides with a later parameter in the parameter list
/// is rejected even when the colliding parameter appears after other distinct
/// parameters.
#[test]
fn validate_binding_collides_with_later_parameter()
{
	let ast = Parser::parse("a, b, x: x@(3D6) + {x}").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::BindingCollidesWithParameter {
			name: "x",
			parameter: SourceSpan { start: 6, end: 7 },
			binding: SourceSpan { start: 9, end: 10 }
		})
	);
}

/// Use-before-bind fires for a reference buried inside a restricted grammar
/// position (dice count), proving that the lexical-order walk descends into
/// every subexpression.
#[test]
fn validate_use_before_bind_in_dice_count()
{
	let ast = Parser::parse("{n}D6 + n@(3)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "n",
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 8, end: 9 }
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
	let ast = Parser::parse("x@({y}) + y@(1)").unwrap();
	assert_eq!(
		Validator::validate(&ast),
		Err(CompilationError::UseBeforeBind {
			name: "y",
			reference: SourceSpan { start: 3, end: 6 },
			binding: SourceSpan { start: 10, end: 11 }
		})
	);
}

/// Names that are neither parameters nor bindings flow through validation
/// untouched — they become external variables when the compiler runs. A body
/// that mixes bindings, parameters, and externals validates cleanly.
#[test]
fn validate_mixed_parameter_binding_external()
{
	let ast = Parser::parse("p: x@(1D6) + {p} + {x} + {env}").unwrap();
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
	let result = compile_unoptimized("x: x@(3D6) + {x}");
	assert_eq!(
		result.err(),
		Some(CompilationError::BindingCollidesWithParameter {
			name: "x",
			parameter: SourceSpan { start: 0, end: 1 },
			binding: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// `compile()` — the optimizing pipeline — also surfaces
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// without reaching the optimizer.
#[test]
fn compile_propagates_binding_collides_with_parameter()
{
	let result = compile("x: x@(3D6) + {x}");
	assert_eq!(
		result.err(),
		Some(CompilationError::BindingCollidesWithParameter {
			name: "x",
			parameter: SourceSpan { start: 0, end: 1 },
			binding: SourceSpan { start: 3, end: 4 }
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
	let result = evaluate("x: x@(3D6) + {x}", vec![0], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::BindingCollidesWithParameter {
			name: "x",
			parameter: SourceSpan { start: 0, end: 1 },
			binding: SourceSpan { start: 3, end: 4 }
		})
	);
}

/// `compile_unoptimized()` surfaces
/// [`DuplicateBinding`](CompilationError::DuplicateBinding).
#[test]
fn compile_unoptimized_propagates_duplicate_binding()
{
	let result = compile_unoptimized("x@(3D6) + x@(1D4)");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateBinding {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 10, end: 11 }
		})
	);
}

/// `compile()` surfaces
/// [`DuplicateBinding`](CompilationError::DuplicateBinding).
#[test]
fn compile_propagates_duplicate_binding()
{
	let result = compile("x@(3D6) + x@(1D4)");
	assert_eq!(
		result.err(),
		Some(CompilationError::DuplicateBinding {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 10, end: 11 }
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
	let result = evaluate("x@(3D6) + x@(1D4)", vec![], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::DuplicateBinding {
			name: "x",
			first: SourceSpan { start: 0, end: 1 },
			duplicate: SourceSpan { start: 10, end: 11 }
		})
	);
}

/// `compile_unoptimized()` surfaces
/// [`UseBeforeBind`](CompilationError::UseBeforeBind).
#[test]
fn compile_unoptimized_propagates_use_before_bind()
{
	let result = compile_unoptimized("{x} + x@(3D6)");
	assert_eq!(
		result.err(),
		Some(CompilationError::UseBeforeBind {
			name: "x",
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `compile()` surfaces
/// [`UseBeforeBind`](CompilationError::UseBeforeBind).
#[test]
fn compile_propagates_use_before_bind()
{
	let result = compile("{x} + x@(3D6)");
	assert_eq!(
		result.err(),
		Some(CompilationError::UseBeforeBind {
			name: "x",
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 6, end: 7 }
		})
	);
}

/// `evaluate()` converts [`UseBeforeBind`](CompilationError::UseBeforeBind)
/// to its mirror variant.
#[test]
fn evaluate_propagates_use_before_bind()
{
	let mut rng = rand::rng();
	let result = evaluate("{x} + x@(3D6)", vec![], vec![], &mut rng);
	assert_eq!(
		result.err(),
		Some(EvaluationError::UseBeforeBind {
			name: "x",
			reference: SourceSpan { start: 0, end: 3 },
			binding: SourceSpan { start: 6, end: 7 }
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
		name: "x",
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
		name: "x",
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
		name: "x",
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
		name: "x",
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
		name: "x",
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
		name: "x",
		reference: SourceSpan { start: 0, end: 3 },
		binding: SourceSpan { start: 6, end: 7 }
	};
	assert_eq!(
		format!("{}", err),
		"reference to 'x' at 0..3 precedes its binding at 6..7"
	);
}
