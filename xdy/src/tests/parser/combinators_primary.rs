//! # Primary-expression combinator test cases
//!
//! Herein are the test cases for the [`primary`], [`group`], [`variable`],
//! and [`range`] combinators.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                    Primary-expression combinator tests.                    //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`primary`] behaves as expected.
#[test]
fn test_primary()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"(1 + 2)",
			"(1 + 2)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			})
		),
		(
			"{x}",
			"{x}",
			Expression::Variable(Variable {
				name: "x",
				span: SourceSpan::default()
			})
		),
		(
			"[1:10]",
			"[1:10]",
			Expression::Range(Range {
				start: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				end: Box::new(Expression::Constant(Constant {
					value: 10,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			})
		),
		(
			"3d6",
			"3D6",
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"42",
			"42",
			Expression::Constant(Constant {
				value: 42,
				span: SourceSpan::default()
			})
		),
		(
			"-42",
			"-42",
			Expression::Constant(Constant {
				value: -42,
				span: SourceSpan::default()
			})
		)
	]
	{
		let span = Span::new(input);
		match primary(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected_str,
					"Failed for input: {}",
					input
				);
				assert_eq!(
					result.untethered(),
					expected_ast.untethered(),
					"AST mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", "+", "*", "/", "^", "3+4"]
	{
		let span = Span::new(input);
		let result = primary(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`group`] behaves as expected.
#[test]
fn test_group()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"(1)",
			"(1)",
			Group {
				expression: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"(1 + 2)",
			"(1 + 2)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		),
		(
			"((1 + 2) * 3)",
			"((1 + 2) * 3)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Group(Group {
							expression: Box::new(Expression::Arithmetic(
								ArithmeticExpression::Add(Add {
									left: Box::new(Expression::Constant(
										Constant {
											value: 1,
											span: SourceSpan::default()
										}
									)),
									right: Box::new(Expression::Constant(
										Constant {
											value: 2,
											span: SourceSpan::default()
										}
									)),
									span: SourceSpan::default()
								})
							)),
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		),
		(
			"({x})",
			"({x})",
			Group {
				expression: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"(3d6)",
			"(3D6)",
			Group {
				expression: Box::new(Expression::Dice(
					DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		),
		(
			"(1d20 + 5)",
			"(1D20 + 5)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Dice(
							DiceExpression::Standard(StandardDice {
								count: Box::new(Expression::Constant(
									Constant {
										value: 1,
										span: SourceSpan::default()
									}
								)),
								faces: Box::new(Expression::Constant(
									Constant {
										value: 20,
										span: SourceSpan::default()
									}
								)),
								span: SourceSpan::default()
							})
						)),
						right: Box::new(Expression::Constant(Constant {
							value: 5,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		),
		(
			"( 1 )",
			"(1)",
			Group {
				expression: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"(  1  +  2  )",
			"(1 + 2)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		)
	]
	{
		let span = Span::new(input);
		match group(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected_str,
					"Failed for input: {}",
					input
				);
				assert_eq!(
					result.untethered(),
					expected_ast.untethered(),
					"AST mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", "(", ")", "1", "(1", "1)", "(1)2", "((1)"]
	{
		let span = Span::new(input);
		let result = group(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`variable`] behaves as expected.
#[test]
fn test_variable()
{
	// Happy paths.
	for (input, expected) in [
		("{x}", "{x}"),
		("{variable}", "{variable}"),
		("{long_variable_name}", "{long_variable_name}"),
		("{hello-world}", "{hello-world}"),
		("{こんにちは}", "{こんにちは}"),
		("{Здравствуй_мир}", "{Здравствуй_мир}"),
		("{Γειά-σου-κόσμε}", "{Γειά-σου-κόσμε}"),
		("{x1}", "{x1}"),
		("{x_1}", "{x_1}"),
		("{x-1}", "{x-1}"),
		("{hello world}", "{hello world}"),
		("{hello world }", "{hello world}"),
		("{an external variable}", "{an external variable}"),
		("{foo.bar}", "{foo.bar}")
	]
	{
		let span = Span::new(input);
		match variable(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected,
					"Failed for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in [
		"",
		" ",
		"x",
		"{}",
		"{ }",
		"{1}",
		"{-x}",
		"x{}",
		"{x}y",
		"{नमस्ते दुनिया}" // This contains nonspacing marks.
	]
	{
		let span = Span::new(input);
		let result = variable(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`range`] behaves as expected.
#[test]
fn test_range()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"[1:10]",
			"[1:10]",
			Range {
				start: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				end: Box::new(Expression::Constant(Constant {
					value: 10,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"[-10:10]",
			"[-10:10]",
			Range {
				start: Box::new(Expression::Constant(Constant {
					value: -10,
					span: SourceSpan::default()
				})),
				end: Box::new(Expression::Constant(Constant {
					value: 10,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"[{min}:{max}]",
			"[{min}:{max}]",
			Range {
				start: Box::new(Expression::Variable(Variable {
					name: "min",
					span: SourceSpan::default()
				})),
				end: Box::new(Expression::Variable(Variable {
					name: "max",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"[1+2:3*4]",
			"[1 + 2:3 * 4]",
			Range {
				start: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				end: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 4,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}
		),
		(
			"[(1+2):(3*4)]",
			"[(1 + 2):(3 * 4)]",
			Range {
				start: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant {
								value: 1,
								span: SourceSpan::default()
							})),
							right: Box::new(Expression::Constant(Constant {
								value: 2,
								span: SourceSpan::default()
							})),
							span: SourceSpan::default()
						})
					)),
					span: SourceSpan::default()
				})),
				end: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Mul(Mul {
							left: Box::new(Expression::Constant(Constant {
								value: 3,
								span: SourceSpan::default()
							})),
							right: Box::new(Expression::Constant(Constant {
								value: 4,
								span: SourceSpan::default()
							})),
							span: SourceSpan::default()
						})
					)),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"[1d6:2d8]",
			"[1D6:2D8]",
			Range {
				start: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				end: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 8,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}
		)
	]
	{
		let span = Span::new(input);
		match range(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected_str,
					"Failed for input: {}",
					input
				);
				assert_eq!(
					result.untethered(),
					expected_ast.untethered(),
					"AST mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", "1:10", "[1]", "[1:]", "[:10]", "[1,10]", "1..10"]
	{
		let span = Span::new(input);
		let result = range(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`binding`] accepts every well-formed binding form and rejects
/// shapes that are not bindings.
#[test]
fn test_binding()
{
	// Happy paths — single-character, multi-char, dotted, multi-word, with
	// surrounding whitespace, with a rich bound expression, and nested.
	for (input, expected) in [
		("x@(3D6)", "x@(3D6)"),
		("long_name@(1D20)", "long_name@(1D20)"),
		("x-y@(1D4)", "x-y@(1D4)"),
		("x.y.z@(4D8)", "x.y.z@(4D8)"),
		("a new id@(2D4)", "a new id@(2D4)"),
		("x @ (3D6)", "x@(3D6)"),
		("x@ ( 3D6 )", "x@(3D6)"),
		("x@(1 + 2)", "x@(1 + 2)"),
		("x@(3D6 + 2)", "x@(3D6 + 2)"),
		("x@(1D6 drop lowest)", "x@(1D6 drop lowest)"),
		("a@(b@(1D4) + {b})", "a@(b@(1D4) + {b})"),
		("x@([1:6])", "x@([1:6])"),
		("x@(-3)", "x@(-3)")
	]
	{
		let span = Span::new(input);
		match binding(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected,
					"Rendering mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs — missing `@`, missing parentheses, empty bound
	// expression, leading digit in name, or a bare identifier that isn't
	// followed by `@`. A plain identifier must rewind so the enclosing
	// grammar can try other alternatives.
	for input in [
		"x", "xyz", "3@(1)", "x(1D6)", "x@1D6", "x@()", "@x(1D6)", ""
	]
	{
		let span = Span::new(input);
		let result = binding(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {:?}",
			input
		);
	}
}

/// The `binding` combinator must appear transparently in every grammar
/// position where a variable reference is legal — `primary`, `dice_count`,
/// `standard_faces`, and `drop_expression`. This test drives each combinator
/// with a binding form and asserts successful recognition.
#[test]
fn test_binding_in_restricted_positions()
{
	// dice_count: `x@(2+3)D6` — the dice_count is the full binding.
	let (residue, expr) = dice_count(Span::new("x@(2+3)")).unwrap();
	assert!(residue.fragment().is_empty());
	assert!(matches!(expr, Expression::Binding(_)));
	// standard_faces: `3Df@(6)` — after `3D`, standard_faces sees `f@(6)`.
	let (residue, expr) = standard_faces(Span::new("f@(6)")).unwrap();
	assert!(residue.fragment().is_empty());
	assert!(matches!(expr, Expression::Binding(_)));
	// drop_expression: `4D6 drop lowest k@(2)` — the drop_expression sees
	// `k@(2)`.
	let (residue, expr) = drop_expression(Span::new("k@(2)")).unwrap();
	assert!(residue.fragment().is_empty());
	assert!(matches!(expr, Expression::Binding(_)));
	// primary: bindings are legal primaries, so a binding after a binary
	// operator flows through the expression pipeline cleanly.
	let (residue, expr) = expression(Span::new("1 + x@(3D6)")).unwrap();
	assert!(residue.fragment().is_empty());
	let rendered = expr.to_string();
	assert_eq!(rendered, "1 + x@(3D6)");
}

/// A binding that fails on its `@` peek must not consume input — otherwise an
/// [`alt`](nom::branch::alt) combinator would see an error span past the
/// identifier and mask the real parse failure. This is the mechanism that
/// preserves [`BareIdentifier`](crate::diagnostics::DiagnosticKind::BareIdentifier)
/// detection for `4 + hello`.
#[test]
fn test_binding_rewinds_on_no_at_sign()
{
	let input = "hello";
	let span = Span::new(input);
	let result = binding(span);
	assert!(result.is_err(), "binding should reject `{}`", input);
	if let Err(nom::Err::Error(e)) = result
	{
		// The recorded error position must be at offset 0 (the start of
		// `hello`), not after the identifier — that's how the enclosing
		// `alt` learns to keep its rightmost-error pointer undisturbed.
		assert_eq!(
			e.errors[0].0.location_offset(),
			0,
			"binding error should rewind to the start of the input, not \
			 point past the consumed identifier"
		);
	}
	else
	{
		panic!("expected nom::Err::Error, got: {:?}", result);
	}
}
