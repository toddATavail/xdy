//! # Parser test cases
//!
//! Herein are the test cases for the various parser functions.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                             Parser test cases.                             //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`function`] behaves as expected.
#[test]
fn test_function()
{
	// Happy paths.
	for (input, expected_string, expected_ast) in [
		(
			"x: 42",
			"x: 42",
			Function {
				parameters: Some(vec![Parameter {
					name: "x",
					span: SourceSpan::default()
				}]),
				body: Expression::Constant(Constant {
					value: 42,
					span: SourceSpan::default()
				}),
				span: SourceSpan::default()
			}
		),
		(
			"3d6",
			"3D6",
			Function {
				parameters: None,
				body: Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				)),
				span: SourceSpan::default()
			}
		),
		(
			"x, y: {x} + {y}",
			"x, y: {x} + {y}",
			Function {
				parameters: Some(vec![
					Parameter {
						name: "x",
						span: SourceSpan::default()
					},
					Parameter {
						name: "y",
						span: SourceSpan::default()
					},
				]),
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Variable(Variable {
						name: "x",
						span: SourceSpan::default()
					})),
					right: Box::new(Expression::Variable(Variable {
						name: "y",
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"2d20 + 5",
			"2D20 + 5",
			Function {
				parameters: None,
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Dice(DiceExpression::Standard(
						StandardDice {
							count: Box::new(Expression::Constant(Constant {
								value: 2,
								span: SourceSpan::default()
							})),
							faces: Box::new(Expression::Constant(Constant {
								value: 20,
								span: SourceSpan::default()
							})),
							span: SourceSpan::default()
						}
					))),
					right: Box::new(Expression::Constant(Constant {
						value: 5,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"a, b, c: ({a} + {b}) * {c}",
			"a, b, c: ({a} + {b}) * {c}",
			Function {
				parameters: Some(vec![
					Parameter {
						name: "a",
						span: SourceSpan::default()
					},
					Parameter {
						name: "b",
						span: SourceSpan::default()
					},
					Parameter {
						name: "c",
						span: SourceSpan::default()
					},
				]),
				body: Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
					left: Box::new(Expression::Group(Group {
						expression: Box::new(Expression::Arithmetic(
							ArithmeticExpression::Add(Add {
								left: Box::new(Expression::Variable(
									Variable {
										name: "a",
										span: SourceSpan::default()
									}
								)),
								right: Box::new(Expression::Variable(
									Variable {
										name: "b",
										span: SourceSpan::default()
									}
								)),
								span: SourceSpan::default()
							})
						)),
						span: SourceSpan::default()
					})),
					right: Box::new(Expression::Variable(Variable {
						name: "c",
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"4d6 drop lowest",
			"4D6 drop lowest",
			Function {
				parameters: None,
				body: Expression::Dice(DiceExpression::DropLowest(
					DropLowest {
						dice: Box::new(DiceExpression::Standard(
							StandardDice {
								count: Box::new(Expression::Constant(
									Constant {
										value: 4,
										span: SourceSpan::default()
									}
								)),
								faces: Box::new(Expression::Constant(
									Constant {
										value: 6,
										span: SourceSpan::default()
									}
								)),
								span: SourceSpan::default()
							}
						)),
						drop: None,
						span: SourceSpan::default()
					}
				)),
				span: SourceSpan::default()
			}
		),
		(
			"4d6 drop highest {z}",
			"4D6 drop highest {z}",
			Function {
				parameters: None,
				body: Expression::Dice(DiceExpression::DropHighest(
					DropHighest {
						dice: Box::new(DiceExpression::Standard(
							StandardDice {
								count: Box::new(Expression::Constant(
									Constant {
										value: 4,
										span: SourceSpan::default()
									}
								)),
								faces: Box::new(Expression::Constant(
									Constant {
										value: 6,
										span: SourceSpan::default()
									}
								)),
								span: SourceSpan::default()
							}
						)),
						drop: Some(Box::new(Expression::Variable(Variable {
							name: "z",
							span: SourceSpan::default()
						}))),
						span: SourceSpan::default()
					}
				)),
				span: SourceSpan::default()
			}
		),
		(
			"x: {x} * 2 + 1",
			"x: {x} * 2 + 1",
			Function {
				parameters: Some(vec![Parameter {
					name: "x",
					span: SourceSpan::default()
				}]),
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Mul(Mul {
							left: Box::new(Expression::Variable(Variable {
								name: "x",
								span: SourceSpan::default()
							})),
							right: Box::new(Expression::Constant(Constant {
								value: 2,
								span: SourceSpan::default()
							})),
							span: SourceSpan::default()
						})
					)),
					right: Box::new(Expression::Constant(Constant {
						value: 1,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"a, b: [{a}:{b}]",
			"a, b: [{a}:{b}]",
			Function {
				parameters: Some(vec![
					Parameter {
						name: "a",
						span: SourceSpan::default()
					},
					Parameter {
						name: "b",
						span: SourceSpan::default()
					},
				]),
				body: Expression::Range(Range {
					start: Box::new(Expression::Variable(Variable {
						name: "a",
						span: SourceSpan::default()
					})),
					end: Box::new(Expression::Variable(Variable {
						name: "b",
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				}),
				span: SourceSpan::default()
			}
		),
		(
			"3D[1,2,3] + 5",
			"3D[1, 2, 3] + 5",
			Function {
				parameters: None,
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Dice(DiceExpression::Custom(
						CustomDice {
							count: Box::new(Expression::Constant(Constant {
								value: 3,
								span: SourceSpan::default()
							})),
							faces: vec![1, 2, 3],
							span: SourceSpan::default()
						}
					))),
					right: Box::new(Expression::Constant(Constant {
						value: 5,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		)
	]
	{
		let span = Span::new(input);
		match function(span)
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
					expected_string,
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
	for input in ["", " ", ":", "x:", "x: ", "x, y", "x, y:", "1 + "]
	{
		let span = Span::new(input);
		let result = function(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`parameters`] behaves as expected.
#[test]
fn test_parameters()
{
	// Happy paths.
	fn param(name: &str) -> Parameter<'_>
	{
		Parameter {
			name,
			span: SourceSpan::default()
		}
	}
	for (input, expected_names, expected_ast) in [
		("x:", vec!["x"], vec![param("x")]),
		("x, y:", vec!["x", "y"], vec![param("x"), param("y")]),
		(
			"x, y, z:",
			vec!["x", "y", "z"],
			vec![param("x"), param("y"), param("z")]
		),
		(
			"long_variable_name:",
			vec!["long_variable_name"],
			vec![param("long_variable_name")]
		),
		(
			"x1, x2, x3:",
			vec!["x1", "x2", "x3"],
			vec![param("x1"), param("x2"), param("x3")]
		),
		(
			"hello-world:",
			vec!["hello-world"],
			vec![param("hello-world")]
		),
		(
			"x, y, z:",
			vec!["x", "y", "z"],
			vec![param("x"), param("y"), param("z")]
		),
		(
			" x , y , z :",
			vec!["x", "y", "z"],
			vec![param("x"), param("y"), param("z")]
		)
	]
	{
		let span = Span::new(input);
		match parameters(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				let result = result.unwrap();
				let names: Vec<&str> = result.iter().map(|p| p.name).collect();
				assert_eq!(
					names, expected_names,
					"Failed for input: {}",
					input
				);
				let result_untethered: Vec<Parameter<'_>> =
					result.iter().map(Spanned::untethered).collect();
				let expected_untethered: Vec<Parameter<'_>> =
					expected_ast.iter().map(Spanned::untethered).collect();
				assert_eq!(
					result_untethered, expected_untethered,
					"AST mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in [
		"x", "x,", "x, ", "x, y", "x, y,", ":x", "x:y", "1, 2, 3:", "x, 1, y:"
	]
	{
		let span = Span::new(input);
		let result = parameters(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`parameter`] behaves as expected.
#[test]
fn test_parameter()
{
	// Happy paths.
	for (input, expected) in [
		("x", "x"),
		("variable", "variable"),
		("long_variable_name", "long_variable_name"),
		("hello-world", "hello-world"),
		("こんにちは", "こんにちは"),
		("Здравствуй-мир", "Здравствуй-мир"),
		("Γειά-σου-κόσμε", "Γειά-σου-κόσμε"),
		("x1", "x1"),
		("x_1", "x_1"),
		("x-1", "x-1")
	]
	{
		let span = Span::new(input);
		match parameter(span)
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
		"", " ", "1x", "x:", "x,", "x{", "x}", "x(", "x)", "x[", "x]"
	]
	{
		let span = Span::new(input);
		let result = parameter(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`expression`] behaves as expected.
#[test]
fn test_expression()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
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
			"2 + 3",
			"2 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"4 - 2",
			"4 - 2",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"3 * 5",
			"3 * 5",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"10 / 2",
			"10 / 2",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant {
					value: 10,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"7 % 3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant {
					value: 7,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2 ^ 3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2 + 3 * 4",
			"2 + 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Arithmetic(
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
			}))
		),
		(
			"(2 + 3) * 4",
			"(2 + 3) * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant {
								value: 2,
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
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"1d20 + 5",
			"1D20 + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 1,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 20,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				right: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"3d6 drop lowest",
			"3D6 drop lowest",
			Expression::Dice(DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant {
						value: 3,
						span: SourceSpan::default()
					})),
					faces: Box::new(Expression::Constant(Constant {
						value: 6,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				drop: None,
				span: SourceSpan::default()
			}))
		),
		(
			"4d6 drop highest 1",
			"4D6 drop highest 1",
			Expression::Dice(DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant {
						value: 4,
						span: SourceSpan::default()
					})),
					faces: Box::new(Expression::Constant(Constant {
						value: 6,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				drop: Some(Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				}))),
				span: SourceSpan::default()
			}))
		)
	]
	{
		let span = Span::new(input);
		match expression(span)
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
	for input in ["", " ", "+", "*", "/", "^", "d6", "drop lowest"]
	{
		let span = Span::new(input);
		let result = expression(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`add_sub`] behaves as expected.
#[test]
fn test_add_sub()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"-5",
			"-5",
			Expression::Constant(Constant {
				value: -5,
				span: SourceSpan::default()
			})
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant {
					value: -5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
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
				span: SourceSpan::default()
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		(
			"2",
			"2",
			Expression::Constant(Constant {
				value: 2,
				span: SourceSpan::default()
			})
		),
		(
			"-2",
			"-2",
			Expression::Constant(Constant {
				value: -2,
				span: SourceSpan::default()
			})
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant {
							value: 3,
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
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant {
								value: 2,
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
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6/3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"7%3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant {
					value: 7,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2×3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6÷3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3*4",
			"2 * 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"12/4/3",
			"12 / 4 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant {
							value: 12,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 4,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"10%3%2",
			"10 % 3 % 2",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant {
							value: 10,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3/4",
			"2 * 3 / 4",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6/2*3",
			"6 / 2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"8%3*2",
			"8 % 3 * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant {
							value: 8,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"{x}*2",
			"{x} * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*(3+4)",
			"2 * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"(1+2)*(3+4)",
			"(1 + 2) * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"5 + 3",
			"5 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"5 - 3",
			"5 - 3",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"5 + 3 - 2",
			"5 + 3 - 2",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 5,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"5 - 3 + 2",
			"5 - 3 + 2",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Sub(Sub {
						left: Box::new(Expression::Constant(Constant {
							value: 5,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"1 + 2 + 3",
			"1 + 2 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Arithmetic(
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
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"1 - 2 - 3",
			"1 - 2 - 3",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Sub(Sub {
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
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"{x} + 5",
			"{x} + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"5 + {x}",
			"5 + {x}",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"(1 + 2) + 3",
			"(1 + 2) + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"1 + (2 + 3)",
			"1 + (2 + 3)",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant {
								value: 2,
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
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"3d6 + 5",
			"3D6 + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				right: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"5 + 3d6",
			"5 + 3D6",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		(
			"1d20 + 5 - 2",
			"1D20 + 5 - 2",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Arithmetic(
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
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		)
	]
	{
		let span = Span::new(input);
		match add_sub(span)
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
	for input in ["", " ", "+", "-", "5 +", "+ 5", "5 - ", "5 + +"]
	{
		let span = Span::new(input);
		let result = add_sub(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`mul_div_mod`] behaves as expected.
#[test]
fn test_mul_div_mod()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"-5",
			"-5",
			Expression::Constant(Constant {
				value: -5,
				span: SourceSpan::default()
			})
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant {
					value: -5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
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
				span: SourceSpan::default()
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		(
			"2",
			"2",
			Expression::Constant(Constant {
				value: 2,
				span: SourceSpan::default()
			})
		),
		(
			"-2",
			"-2",
			Expression::Constant(Constant {
				value: -2,
				span: SourceSpan::default()
			})
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant {
							value: 3,
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
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant {
								value: 2,
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
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6/3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"7%3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant {
					value: 7,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2×3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6÷3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3*4",
			"2 * 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"12/4/3",
			"12 / 4 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant {
							value: 12,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 4,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"10%3%2",
			"10 % 3 % 2",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant {
							value: 10,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*3/4",
			"2 * 3 / 4",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"6/2*3",
			"6 / 2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 2,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"8%3*2",
			"8 % 3 * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant {
							value: 8,
							span: SourceSpan::default()
						})),
						right: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})
				)),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"{x}*2",
			"{x} * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2*(3+4)",
			"2 * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"(1+2)*(3+4)",
			"(1 + 2) * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		)
	]
	{
		let span = Span::new(input);
		match mul_div_mod(span)
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
	for input in ["", " ", "*", "/", "%", "2*", "*2"]
	{
		let span = Span::new(input);
		let result = mul_div_mod(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`exponent`] behaves as expected. Note that `exponent` calls
/// `primary` directly; negation is handled by `unary`, which sits above
/// `exponent` in the precedence tower.
#[test]
fn test_exponent()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"2",
			"2",
			Expression::Constant(Constant {
				value: 2,
				span: SourceSpan::default()
			})
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant {
							value: 3,
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
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
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
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
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
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant {
								value: 2,
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
				})),
				right: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		)
	]
	{
		let span = Span::new(input);
		match exponent(span)
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
	for input in ["", " ", "^", "2^", "^2", "2^^3", "2^3^"]
	{
		let span = Span::new(input);
		let result = exponent(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`unary`] behaves as expected.
#[test]
fn test_unary()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"-5",
			"-5",
			Expression::Constant(Constant {
				value: -5,
				span: SourceSpan::default()
			})
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant {
					value: -5,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable {
					name: "x",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
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
				span: SourceSpan::default()
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		// i32::MIN boundary via negative_overflowed_constant.
		(
			"-2147483648",
			"-2147483648",
			Expression::Constant(Constant {
				value: -2147483648,
				span: SourceSpan::default()
			})
		),
		// Massive negative overflow saturates to i32::MIN.
		(
			"-9999999999999",
			"-2147483648",
			Expression::Constant(Constant {
				value: -2147483648,
				span: SourceSpan::default()
			})
		),
		// Overflow negative constant followed by dice operator: the count
		// saturates to i32::MAX because negative_constant bails on `d`.
		(
			"-2147483648d6",
			"-2147483647D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 2147483647,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		// Massive overflow followed by dice operator: saturates to i32::MAX.
		(
			"-9999999999999d6",
			"-2147483647D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 2147483647,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		// Overflow negative constant with custom faces.
		(
			"-2147483648D[1,2,3]",
			"-2147483647D[1, 2, 3]",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Custom(
					CustomDice {
						count: Box::new(Expression::Constant(Constant {
							value: 2147483647,
							span: SourceSpan::default()
						})),
						faces: vec![1, 2, 3],
						span: SourceSpan::default()
					}
				))),
				span: SourceSpan::default()
			}))
		),
		// Overflow negative constant with drop clause.
		(
			"-2147483648d6 drop lowest",
			"-2147483647D6 drop lowest",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(
					DiceExpression::DropLowest(DropLowest {
						dice: Box::new(DiceExpression::Standard(
							StandardDice {
								count: Box::new(Expression::Constant(
									Constant {
										value: 2147483647,
										span: SourceSpan::default()
									}
								)),
								faces: Box::new(Expression::Constant(
									Constant {
										value: 6,
										span: SourceSpan::default()
									}
								)),
								span: SourceSpan::default()
							}
						)),
						drop: None,
						span: SourceSpan::default()
					})
				)),
				span: SourceSpan::default()
			}))
		)
	]
	{
		let span = Span::new(input);
		match unary(span)
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
	for input in ["", " ", "-", "+5", "*5", "/5"]
	{
		let span = Span::new(input);
		let result = unary(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

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

/// Ensure that [`dice`] behaves as expected.
#[test]
fn test_dice()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"3d6",
			"3D6",
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
		),
		(
			"2D20",
			"2D20",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Constant(Constant {
					value: 20,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			})
		),
		(
			"3D[1,2,3]",
			"3D[1, 2, 3]",
			DiceExpression::Custom(CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				faces: vec![1, 2, 3],
				span: SourceSpan::default()
			})
		),
		(
			"4d6 drop lowest",
			"4D6 drop lowest",
			DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant {
						value: 4,
						span: SourceSpan::default()
					})),
					faces: Box::new(Expression::Constant(Constant {
						value: 6,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				drop: None,
				span: SourceSpan::default()
			})
		),
		(
			"5D20 drop highest 2",
			"5D20 drop highest 2",
			DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant {
						value: 5,
						span: SourceSpan::default()
					})),
					faces: Box::new(Expression::Constant(Constant {
						value: 20,
						span: SourceSpan::default()
					})),
					span: SourceSpan::default()
				})),
				drop: Some(Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				}))),
				span: SourceSpan::default()
			})
		),
		(
			"3d6 drop lowest 1 drop highest 1",
			"3D6 drop lowest 1 drop highest 1",
			DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::DropLowest(DropLowest {
					dice: Box::new(DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})),
					drop: Some(Box::new(Expression::Constant(Constant {
						value: 1,
						span: SourceSpan::default()
					}))),
					span: SourceSpan::default()
				})),
				drop: Some(Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				}))),
				span: SourceSpan::default()
			})
		),
		(
			"3d6 drop highest drop lowest",
			"3D6 drop highest drop lowest",
			DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::DropHighest(DropHighest {
					dice: Box::new(DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant {
							value: 3,
							span: SourceSpan::default()
						})),
						faces: Box::new(Expression::Constant(Constant {
							value: 6,
							span: SourceSpan::default()
						})),
						span: SourceSpan::default()
					})),
					drop: None,
					span: SourceSpan::default()
				})),
				drop: None,
				span: SourceSpan::default()
			})
		),
		(
			"{count}d{faces}",
			"{count}D{faces}",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Variable(Variable {
					name: "count",
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Variable(Variable {
					name: "faces",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			})
		),
		(
			"(1+2)D(3+3)",
			"(1 + 2)D(3 + 3)",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Group(Group {
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
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant {
								value: 3,
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
				})),
				span: SourceSpan::default()
			})
		),
		(
			"(3d6)D(1d4)",
			"(3D6)D(1D4)",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Group(Group {
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
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant {
								value: 1,
								span: SourceSpan::default()
							})),
							faces: Box::new(Expression::Constant(Constant {
								value: 4,
								span: SourceSpan::default()
							})),
							span: SourceSpan::default()
						})
					)),
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			})
		)
	]
	{
		let span = Span::new(input);
		match dice(span)
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
	for input in [
		"",
		" ",
		"d6",
		"3",
		"3d",
		"d",
		"3+4d6",
		"(3d6)",
		"drop lowest",
		"3d6 drop",
		"3d6 drop middle"
	]
	{
		let span = Span::new(input);
		let result = dice(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`standard_dice`] behaves as expected.
#[test]
fn test_standard_dice()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"3d6",
			"3D6",
			StandardDice {
				count: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Constant(Constant {
					value: 6,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"2D20",
			"2D20",
			StandardDice {
				count: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Constant(Constant {
					value: 20,
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"{count}d{faces}",
			"{count}D{faces}",
			StandardDice {
				count: Box::new(Expression::Variable(Variable {
					name: "count",
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Variable(Variable {
					name: "faces",
					span: SourceSpan::default()
				})),
				span: SourceSpan::default()
			}
		),
		(
			"(1+2)D(3+3)",
			"(1 + 2)D(3 + 3)",
			StandardDice {
				count: Box::new(Expression::Group(Group {
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
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant {
								value: 3,
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
				})),
				span: SourceSpan::default()
			}
		),
		(
			"(3d6)D(1d4)",
			"(3D6)D(1D4)",
			StandardDice {
				count: Box::new(Expression::Group(Group {
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
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant {
								value: 1,
								span: SourceSpan::default()
							})),
							faces: Box::new(Expression::Constant(Constant {
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
		)
	]
	{
		let span = Span::new(input);
		match standard_dice(span)
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
	for input in ["", " ", "d6", "3", "3d", "d", "3d[1,2,3]", "3+4d6", "(3d6)"]
	{
		let span = Span::new(input);
		let result = standard_dice(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`custom_dice`] behaves as expected.
#[test]
fn test_custom_dice()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"3D[1,2,3]",
			"3D[1, 2, 3]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				faces: vec![1, 2, 3],
				span: SourceSpan::default()
			}
		),
		(
			"{count}D[1, 2, 3]",
			"{count}D[1, 2, 3]",
			CustomDice {
				count: Box::new(Expression::Variable(Variable {
					name: "count",
					span: SourceSpan::default()
				})),
				faces: vec![1, 2, 3],
				span: SourceSpan::default()
			}
		),
		(
			"(1+2)D[ 1 , 2 , 3 ]",
			"(1 + 2)D[1, 2, 3]",
			CustomDice {
				count: Box::new(Expression::Group(Group {
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
				faces: vec![1, 2, 3],
				span: SourceSpan::default()
			}
		),
		(
			"2D[10]",
			"2D[10]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 2,
					span: SourceSpan::default()
				})),
				faces: vec![10],
				span: SourceSpan::default()
			}
		),
		(
			"1D[-1,0,1]",
			"1D[-1, 0, 1]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 1,
					span: SourceSpan::default()
				})),
				faces: vec![-1, 0, 1],
				span: SourceSpan::default()
			}
		),
		(
			"4D[-3, -2, -1]",
			"4D[-3, -2, -1]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 4,
					span: SourceSpan::default()
				})),
				faces: vec![-3, -2, -1],
				span: SourceSpan::default()
			}
		),
		(
			"5D[100, 200, 300]",
			"5D[100, 200, 300]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant {
					value: 5,
					span: SourceSpan::default()
				})),
				faces: vec![100, 200, 300],
				span: SourceSpan::default()
			}
		),
		(
			"(3d6)d[-1,0,1]",
			"(3D6)D[-1, 0, 1]",
			CustomDice {
				count: Box::new(Expression::Group(Group {
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
				})),
				faces: vec![-1, 0, 1],
				span: SourceSpan::default()
			}
		)
	]
	{
		let span = Span::new(input);
		match custom_dice(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.is_empty(),
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
	for input in [
		"",
		" ",
		"3",
		"D[1,2,3]",
		"[]",
		"3D[]",
		"3D[,]",
		"3D[1,]",
		"3D[,1]",
		"3D[1,2,]",
		"3D[a,b,c]",
		"3D1,2,3",
		"3D(1,2,3)"
	]
	{
		let span = Span::new(input);
		let result = custom_dice(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`dice_count`] behaves as expected.
#[test]
fn test_dice_count()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"3",
			"3",
			Expression::Constant(Constant {
				value: 3,
				span: SourceSpan::default()
			})
		),
		(
			"{count}",
			"{count}",
			Expression::Variable(Variable {
				name: "count",
				span: SourceSpan::default()
			})
		),
		(
			"(1+2)",
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
		)
	]
	{
		let span = Span::new(input);
		match dice_count(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.is_empty(),
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
	for input in ["", " ", " 3", "d6", "2d6", "[1,2,3]", "2+3", "a"]
	{
		let span = Span::new(input);
		let result = dice_count(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`standard_faces`] behaves as expected.
#[test]
fn test_standard_faces()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"6",
			"6",
			Expression::Constant(Constant {
				value: 6,
				span: SourceSpan::default()
			})
		),
		(
			"20",
			"20",
			Expression::Constant(Constant {
				value: 20,
				span: SourceSpan::default()
			})
		),
		(
			"{faces}",
			"{faces}",
			Expression::Variable(Variable {
				name: "faces",
				span: SourceSpan::default()
			})
		),
		(
			"(2+3)",
			"(2 + 3)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
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
			})
		)
	]
	{
		let span = Span::new(input);
		match standard_faces(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.is_empty(),
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
	for input in ["", " ", " 3", "d6", "2d6", "[1,2,3]", "2+3", "a"]
	{
		let span = Span::new(input);
		let result = standard_faces(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`custom_faces`] behaves as expected.
#[test]
fn test_custom_faces()
{
	// Happy paths.
	for (input, expected) in [
		("[1,2,3]", vec![1, 2, 3]),
		("[1, 2, 3]", vec![1, 2, 3]),
		("[ 1 , 2 , 3 ]", vec![1, 2, 3]),
		("[10]", vec![10]),
		("[-1,0,1]", vec![-1, 0, 1]),
		("[-3, -2, -1]", vec![-3, -2, -1]),
		("[100, 200, 300]", vec![100, 200, 300])
	]
	{
		let span = Span::new(input);
		match custom_faces(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(result, expected, "Failed for input: {}", input);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in [
		"", " ", "[]", "[,]", "[1,]", "[,1]", "[1,2,]", "[a,b,c]", "1,2,3",
		"(1,2,3)"
	]
	{
		let span = Span::new(input);
		let result = custom_faces(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`drop_lowest`] behaves as expected.
#[test]
fn test_drop_lowest()
{
	// Happy paths.
	for (input, expected) in [
		("drop lowest", ""),
		("drop lowest 2", "2"),
		("drop lowest {num}", "{num}")
	]
	{
		let span = Span::new(input);
		match drop_lowest(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				match result
				{
					Some(drop) => assert_eq!(
						drop.to_string(),
						expected,
						"Failed for input: {}",
						input
					),
					None =>
					{
						assert_eq!(expected, "", "Failed for input: {}", input)
					}
				}
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Edge cases and invalid inputs.
	for input in ["", " ", "2d6 drop", "drop highest"]
	{
		let span = Span::new(input);
		let result = drop_lowest(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`drop_highest`] behaves as expected.
#[test]
fn test_drop_highest()
{
	// Happy paths.
	for (input, expected) in [
		("drop highest", ""),
		("drop highest 2", "2"),
		("drop highest {num}", "{num}")
	]
	{
		let span = Span::new(input);
		match drop_highest(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.fragment().is_empty(),
					"Residue not empty for input: {}",
					input
				);
				match result
				{
					Some(drop) => assert_eq!(
						drop.to_string(),
						expected,
						"Failed for input: {}",
						input
					),
					None =>
					{
						assert_eq!(expected, "", "Failed for input: {}", input)
					}
				}
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Edge cases and invalid inputs.
	for input in ["", " ", "2d6 drop", "drop lowest"]
	{
		let span = Span::new(input);
		let result = drop_highest(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`drop_expression`] behaves as expected.
#[test]
fn test_drop_expression()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
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
		),
		(
			"{var}",
			"{var}",
			Expression::Variable(Variable {
				name: "var",
				span: SourceSpan::default()
			})
		),
		(
			"(2+3)",
			"(2 + 3)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant {
							value: 2,
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
			})
		)
	]
	{
		let span = Span::new(input);
		match drop_expression(span)
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
	for input in ["", " ", "d6", "2d6", "[1:10]", "2+3", "{var}+2"]
	{
		let span = Span::new(input);
		let result = drop_expression(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`constant`] behaves as expected.
#[test]
fn test_constant()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"0",
			"0",
			Constant {
				value: 0,
				span: SourceSpan::default()
			}
		),
		(
			"42",
			"42",
			Constant {
				value: 42,
				span: SourceSpan::default()
			}
		),
		(
			"-42",
			"-42",
			Constant {
				value: -42,
				span: SourceSpan::default()
			}
		),
		(
			"9999",
			"9999",
			Constant {
				value: 9999,
				span: SourceSpan::default()
			}
		),
		(
			"-9999",
			"-9999",
			Constant {
				value: -9999,
				span: SourceSpan::default()
			}
		),
		(
			"2147483647",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		), // i32::MAX
		(
			"-2147483648",
			"-2147483648",
			Constant {
				value: -2147483648,
				span: SourceSpan::default()
			}
		), // i32::MIN
		(
			"2147483648",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		), // Saturates
		(
			"-2147483649",
			"-2147483648",
			Constant {
				value: -2147483648,
				span: SourceSpan::default()
			}
		), // Saturates
		(
			"123456789123456789123456789123456789",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		) // massive overflow saturates
	]
	{
		let span = Span::new(input);
		match constant(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.is_empty(),
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
	for input in ["", " ", " 42", "a", "+42", "--42", "a42", "42a"]
	{
		let span = Span::new(input);
		let result = constant(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`d_operator`] behaves as expected.
#[test]
fn test_d_operator()
{
	// Happy paths.
	for input in ["d", "D"]
	{
		let span = Span::new(input);
		match d_operator(span)
		{
			Ok(result) => assert_eq!(
				result.1,
				input.chars().next().unwrap(),
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", "a", "3", " d", " D"]
	{
		let span = Span::new(input);
		let result = d_operator(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`identifier`] behaves as expected.
#[test]
fn test_identifier()
{
	for (input, expected) in [
		("hello", "hello"),
		("hello123", "hello123"),
		("hello_world", "hello_world"),
		("hello-world", "hello-world"),
		("_hello", "_hello"),
		("_", "_"),
		("h3llo_w0rld-123_test", "h3llo_w0rld-123_test"),
		("αβγ", "αβγ"),
		("_123", "_123"),
		("a", "a"),
		("hello world", "hello world"),
		("hello.world", "hello.world"),
		("an external variable", "an external variable"),
		("a.b.c", "a.b.c")
	]
	{
		let span = Span::new(input);
		match identifier(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Test cases that should fail
	for input in [
		"123hello",
		"-hello",
		" hello",
		"",
		"!hello world",
		"hello!world",
		"hello@world",
		"x "
	]
	{
		let span = Span::new(input);
		let result = identifier(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`alpha`] behaves as expected.
#[test]
fn test_alpha()
{
	let span = Span::new("h");
	assert_eq!(alpha(span).unwrap().1.fragment(), &"h");

	let span = Span::new("1");
	assert!(alpha(span).is_err());

	let span = Span::new("");
	assert!(alpha(span).is_err());

	let span = Span::new("_");
	assert!(alpha(span).is_err());

	for (input, expected) in [
		("H", "H"),
		("a", "a"),
		("é", "é"),
		("こ", "こ"),
		("П", "П"),
		("Γ", "Γ")
	]
	{
		let span = Span::new(input);
		match alpha(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}
}

/// Ensure that [`alphanumeric1`] behaves as expected.
#[test]
fn test_alphanumeric1()
{
	let span = Span::new("hello123");
	assert_eq!(alphanumeric1(span).unwrap().1.fragment(), &"hello123");

	let span = Span::new("123hello");
	assert_eq!(alphanumeric1(span).unwrap().1.fragment(), &"123hello");

	let span = Span::new("");
	assert!(alphanumeric1(span).is_err());

	let span = Span::new("_hello");
	assert!(alphanumeric1(span).is_err());

	for (input, expected) in [
		("Hello", "Hello"),
		("hElLo", "hElLo"),
		("héllo", "héllo"),
		("こんにちは", "こんにちは"),
		("Привет", "Привет"),
		("Γειά σου", "Γειά"),
		("hello world", "hello"),
		("1234567890", "1234567890"),
		("1234567890hello", "1234567890hello"),
		("hello1234567890", "hello1234567890"),
		("hello1234567890world", "hello1234567890world")
	]
	{
		let span = Span::new(input);
		match alphanumeric1(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                         Data-driven parser tests.                          //
////////////////////////////////////////////////////////////////////////////////

/// Test that the parser produces the expected AST for every test case in
/// `test_parse.txt`. Each case is parsed, the resulting AST is serialized to an
/// S-expression, and compared against the expected S-expression.
#[test]
fn test_parse_to_s_expr()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions};
	use crate::support::read_compilation_test_cases;
	use std::collections::HashSet;

	let test_cases =
		read_compilation_test_cases(include_str!("../../tests/test_parse.txt"));
	assert!(
		!test_cases.is_empty(),
		"no test cases found in test_parse.txt"
	);

	let opts = SExpressibleOptions::default();
	let mut seen = HashSet::new();
	for (index, (source, expected)) in test_cases.iter().enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
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
		let actual = ast.to_s_expr(opts);
		assert_eq!(actual.trim(), *expected, "case {}: {}", index + 1, source);
	}
}

/// Test that the S-expression reader and writer roundtrip perfectly: for every
/// expected S-expression in `test_parse.txt`, reading it back and writing it
/// again produces the identical string.
#[test]
fn test_s_expr_roundtrip()
{
	use crate::s_expr::{SExpressible as _, SExpressibleOptions, read_s_expr};
	use crate::support::read_compilation_test_cases;

	let test_cases =
		read_compilation_test_cases(include_str!("../../tests/test_parse.txt"));
	let opts = SExpressibleOptions::default();
	for (index, (source, expected)) in test_cases.iter().enumerate()
	{
		let ast = match read_s_expr(expected)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {}: S-expr read failed for {:?} (source {:?}): {}",
				index + 1,
				expected,
				source,
				e
			)
		};
		let roundtripped = ast.to_s_expr(opts);
		assert_eq!(
			roundtripped.trim(),
			*expected,
			"case {}: roundtrip mismatch for {:?}",
			index + 1,
			source
		);
	}
}

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

	let test_cases =
		read_compilation_test_cases(include_str!("../../tests/test_parse.txt"));
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
			err.message.contains("expected '[' after '^'"),
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
		err.message.contains("span start 5 exceeds end 2"),
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
		err.message.contains("escapes parent span"),
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
		err.message.contains("overlaps or precedes prior sibling"),
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

////////////////////////////////////////////////////////////////////////////////
//                         Internal mechanism tests.                          //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`token`] correctly identifies the next token for error
/// reporting.
#[test]
fn test_token()
{
	// Empty input — no token (EOF).
	assert_eq!(token(Span::new("")).map(|s| *s.fragment()), None);

	// Integer constant.
	assert_eq!(token(Span::new("42")).map(|s| *s.fragment()), Some("42"));

	// Negative constant.
	assert_eq!(token(Span::new("-5")).map(|s| *s.fragment()), Some("-5"));

	// Identifier.
	assert_eq!(
		token(Span::new("hello")).map(|s| *s.fragment()),
		Some("hello")
	);

	// Single punctuation character.
	assert_eq!(token(Span::new("+")).map(|s| *s.fragment()), Some("+"));
	assert_eq!(token(Span::new("*")).map(|s| *s.fragment()), Some("*"));
	assert_eq!(token(Span::new("(")).map(|s| *s.fragment()), Some("("));
	assert_eq!(token(Span::new(")")).map(|s| *s.fragment()), Some(")"));
	assert_eq!(token(Span::new("{")).map(|s| *s.fragment()), Some("{"));
	assert_eq!(token(Span::new("}")).map(|s| *s.fragment()), Some("}"));
	assert_eq!(token(Span::new("[")).map(|s| *s.fragment()), Some("["));
	assert_eq!(token(Span::new("]")).map(|s| *s.fragment()), Some("]"));
	assert_eq!(token(Span::new("@")).map(|s| *s.fragment()), Some("@"));
}

/// Ensure that [`identifier`] correctly handles trailing whitespace, returning
/// the trimmed identifier and giving back the whitespace to the remaining
/// input.
#[test]
fn test_identifier_trailing_whitespace()
{
	// Trailing whitespace should be trimmed from the identifier and returned as
	// part of the remaining input.
	let span = Span::new("hello )");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "hello");
	assert_eq!(*residue.fragment(), " )");

	// Multiple trailing spaces — the exact number must be preserved.
	let span = Span::new("foo   +");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "foo");
	assert_eq!(*residue.fragment(), "   +");
	assert_eq!(residue.location_offset(), 3, "Residue offset after 'foo'");

	// Two trailing spaces — distinguishes subtraction from division in the
	// trail calculation.
	let span = Span::new("ab  )");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "ab");
	assert_eq!(*residue.fragment(), "  )");
	assert_eq!(residue.location_offset(), 2, "Residue offset after 'ab'");

	// Tab as trailing whitespace.
	let span = Span::new("bar\t}");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "bar");
	assert_eq!(*residue.fragment(), "\t}");

	// No trailing whitespace — residue should be what follows immediately.
	let span = Span::new("baz)");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "baz");
	assert_eq!(*residue.fragment(), ")");
}

////////////////////////////////////////////////////////////////////////////////
//                        Source span position tests.                         //
////////////////////////////////////////////////////////////////////////////////

/// Shorthand for building a [`SourceSpan`] from a pair of offsets.
fn span(start: usize, end: usize) -> SourceSpan { SourceSpan { start, end } }

////////////////////////////////////////////////////////////////////////////////
//                         Primary-expression spans.                          //
////////////////////////////////////////////////////////////////////////////////

/// A [`Constant`]'s span covers exactly its numeric literal.
#[test]
fn test_constant_span()
{
	let (_, result) = constant(Span::new("42")).unwrap();
	assert_eq!(result.span, span(0, 2));
}

/// A negative numeric literal parses directly to a [`Constant`] with a negative
/// value (rather than a [`Neg`] wrapping a positive [`Constant`]), so that
/// `i32::MIN` is representable. The span covers the leading `-` as well as the
/// digits.
#[test]
fn test_negative_constant_span()
{
	let (_, result) = expression(Span::new("-123")).unwrap();
	assert_eq!(result.span(), span(0, 4));
	match result
	{
		Expression::Constant(c) =>
		{
			assert_eq!(c.value, -123);
			assert_eq!(c.span, span(0, 4));
		},
		other => panic!("expected Constant, got {:?}", other)
	}
}

/// A [`Variable`]'s span includes the surrounding braces.
#[test]
fn test_variable_span_includes_braces()
{
	let (_, result) = variable(Span::new("{abc}")).unwrap();
	assert_eq!(result.name, "abc");
	assert_eq!(result.span, span(0, 5));
}

/// A [`Variable`] embedded in a larger expression preserves its brace-inclusive
/// span, and the enclosing binary op takes its span from its children.
#[test]
fn test_variable_span_inside_expression()
{
	let (_, result) = expression(Span::new("{x} + 1")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.span, span(0, 7));
			match add.left.as_ref()
			{
				Expression::Variable(v) => assert_eq!(v.span, span(0, 3)),
				other => panic!("expected Variable, got {:?}", other)
			}
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// A [`Range`]'s span includes both `[` and `]`; the endpoint expressions'
/// spans do not.
#[test]
fn test_range_span_includes_brackets()
{
	let (_, result) = range(Span::new("[1:20]")).unwrap();
	assert_eq!(result.span, span(0, 6));
	assert_eq!(result.start.span(), span(1, 2));
	assert_eq!(result.end.span(), span(3, 5));
}

/// A [`Group`]'s span includes its delimiters; the inner expression's span does
/// not.
#[test]
fn test_group_span_includes_delimiters()
{
	let (_, result) = expression(Span::new("(1 + 2)")).unwrap();
	match result
	{
		Expression::Group(group) =>
		{
			assert_eq!(group.span, span(0, 7));
			assert_eq!(group.expression.span(), span(1, 6));
		},
		other => panic!("expected Group, got {:?}", other)
	}
}

/// Nested [`Group`]s each carry their own brace-inclusive span.
#[test]
fn test_nested_group_spans()
{
	let (_, result) = expression(Span::new("((1+2))")).unwrap();
	match result
	{
		Expression::Group(outer) =>
		{
			assert_eq!(outer.span, span(0, 7));
			match outer.expression.as_ref()
			{
				Expression::Group(inner) =>
				{
					assert_eq!(inner.span, span(1, 6));
					assert_eq!(inner.expression.span(), span(2, 5));
				},
				other => panic!("expected inner Group, got {:?}", other)
			}
		},
		other => panic!("expected outer Group, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Dice-expression spans.                           //
////////////////////////////////////////////////////////////////////////////////

/// A bare [`StandardDice`]'s span covers the count, the `d`/`D` operator, and
/// the face count — exactly the source text of the dice expression.
#[test]
fn test_standard_dice_span()
{
	let (_, result) = dice(Span::new("3D6")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 3));
			assert_eq!(sd.count.span(), span(0, 1));
			assert_eq!(sd.faces.span(), span(2, 3));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`StandardDice`] with variable count and variable faces takes its span
/// from the start of the count through the closing brace of the faces.
#[test]
fn test_standard_dice_with_variable_operands_span()
{
	let (_, result) = dice(Span::new("{x}D{y}")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 7));
			assert_eq!(sd.count.span(), span(0, 3));
			assert_eq!(sd.faces.span(), span(4, 7));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`StandardDice`] with a parenthesized count takes its span from the
/// opening `(` of the group through the end of the faces.
#[test]
fn test_standard_dice_with_group_count_span()
{
	let (_, result) = dice(Span::new("(1+2)D6")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 7));
			assert_eq!(sd.count.span(), span(0, 5));
			assert_eq!(sd.faces.span(), span(6, 7));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`CustomDice`]'s span starts at the count and extends through the closing
/// `]` of the face list.
#[test]
fn test_custom_dice_span()
{
	let (_, result) = dice(Span::new("2D[1,2,3]")).unwrap();
	match result
	{
		DiceExpression::Custom(cd) =>
		{
			assert_eq!(cd.span, span(0, 9));
			assert_eq!(cd.count.span(), span(0, 1));
			assert_eq!(cd.faces, vec![1, 2, 3]);
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// A [`CustomDice`] with negative face values still covers through the closing
/// `]`; face values do not have their own spans but are recorded verbatim.
#[test]
fn test_custom_dice_with_negative_faces_span()
{
	let (_, result) = dice(Span::new("3D[-1,0,1]")).unwrap();
	match result
	{
		DiceExpression::Custom(cd) =>
		{
			assert_eq!(cd.span, span(0, 10));
			assert_eq!(cd.faces, vec![-1, 0, 1]);
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// A [`DropLowest`]'s span extends from the start of the inner dice to the end
/// of the drop clause. The inner [`StandardDice`]'s own span covers only the
/// dice, not the drop clause.
#[test]
fn test_drop_lowest_no_count_span()
{
	let (_, result) = dice(Span::new("3d6 drop lowest")).unwrap();
	match result
	{
		DiceExpression::DropLowest(drop) =>
		{
			assert_eq!(drop.span, span(0, 15));
			match drop.dice.as_ref()
			{
				DiceExpression::Standard(std_dice) =>
				{
					assert_eq!(std_dice.span, span(0, 3));
				},
				other => panic!("expected StandardDice, got {:?}", other)
			}
			assert!(drop.drop.is_none());
		},
		other => panic!("expected DropLowest, got {:?}", other)
	}
}

/// A [`DropLowest`] with an explicit drop count extends its span through that
/// drop count, while the inner dice retains its pre-clause span.
#[test]
fn test_drop_lowest_with_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop lowest 2")).unwrap();
	match result
	{
		DiceExpression::DropLowest(drop) =>
		{
			assert_eq!(drop.span, span(0, 17));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert_eq!(
				drop.drop.as_ref().expect("drop count").span(),
				span(16, 17)
			);
		},
		other => panic!("expected DropLowest, got {:?}", other)
	}
}

/// A [`DropHighest`] without an explicit drop count carries a span extending
/// through the `highest` keyword.
#[test]
fn test_drop_highest_no_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop highest")).unwrap();
	match result
	{
		DiceExpression::DropHighest(drop) =>
		{
			assert_eq!(drop.span, span(0, 16));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert!(drop.drop.is_none());
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// A [`DropHighest`] with an explicit drop count extends its span through that
/// drop count.
#[test]
fn test_drop_highest_with_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop highest 1")).unwrap();
	match result
	{
		DiceExpression::DropHighest(drop) =>
		{
			assert_eq!(drop.span, span(0, 18));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert_eq!(
				drop.drop.as_ref().expect("drop count").span(),
				span(17, 18)
			);
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// Stacked drop clauses produce nested wrappers, each with a span extending
/// through its own clause; inner wrappers keep the pre-outer-clause span.
#[test]
fn test_stacked_drop_clauses_span()
{
	let (_, result) =
		dice(Span::new("8D6 drop lowest 3 drop highest 1")).unwrap();
	match result
	{
		DiceExpression::DropHighest(outer) =>
		{
			assert_eq!(outer.span, span(0, 32));
			match outer.dice.as_ref()
			{
				DiceExpression::DropLowest(inner) =>
				{
					assert_eq!(inner.span, span(0, 17));
					assert_eq!(inner.dice.span(), span(0, 3));
				},
				other => panic!("expected inner DropLowest, got {:?}", other)
			}
		},
		other => panic!("expected outer DropHighest, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        Arithmetic-expression spans.                        //
////////////////////////////////////////////////////////////////////////////////

/// A binary [`Add`] derives its span from its children: start of `left` through
/// end of `right`. Whitespace inside the expression is *included* (because
/// `right.span().end` is past the last operand character), but whitespace
/// *outside* the binary op is not.
#[test]
fn test_add_span_from_children()
{
	let (_, result) = expression(Span::new("1 + 22")).unwrap();
	assert_eq!(result.span(), span(0, 6));
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.left.span(), span(0, 1));
			assert_eq!(add.right.span(), span(4, 6));
			assert_eq!(add.span, span(0, 6));
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// A binary [`Sub`] takes its span from its operands.
#[test]
fn test_sub_span_from_children()
{
	let (_, result) = expression(Span::new("10 - 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Sub(sub)) =>
		{
			assert_eq!(sub.span, span(0, 6));
			assert_eq!(sub.left.span(), span(0, 2));
			assert_eq!(sub.right.span(), span(5, 6));
		},
		other => panic!("expected Sub, got {:?}", other)
	}
}

/// A binary [`Mul`] takes its span from its operands.
#[test]
fn test_mul_span_from_children()
{
	let (_, result) = expression(Span::new("2 * 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Mul(mul)) =>
		{
			assert_eq!(mul.span, span(0, 5));
			assert_eq!(mul.left.span(), span(0, 1));
			assert_eq!(mul.right.span(), span(4, 5));
		},
		other => panic!("expected Mul, got {:?}", other)
	}
}

/// A binary [`Div`] takes its span from its operands.
#[test]
fn test_div_span_from_children()
{
	let (_, result) = expression(Span::new("10 / 2")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Div(div)) =>
		{
			assert_eq!(div.span, span(0, 6));
			assert_eq!(div.left.span(), span(0, 2));
			assert_eq!(div.right.span(), span(5, 6));
		},
		other => panic!("expected Div, got {:?}", other)
	}
}

/// A binary [`Mod`] takes its span from its operands.
#[test]
fn test_mod_span_from_children()
{
	let (_, result) = expression(Span::new("10 % 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Mod(r#mod)) =>
		{
			assert_eq!(r#mod.span, span(0, 6));
			assert_eq!(r#mod.left.span(), span(0, 2));
			assert_eq!(r#mod.right.span(), span(5, 6));
		},
		other => panic!("expected Mod, got {:?}", other)
	}
}

/// A binary [`Exp`] takes its span from its operands.
#[test]
fn test_exp_span_from_children()
{
	let (_, result) = expression(Span::new("2 ^ 10")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Exp(exp)) =>
		{
			assert_eq!(exp.span, span(0, 6));
			assert_eq!(exp.left.span(), span(0, 1));
			assert_eq!(exp.right.span(), span(4, 6));
		},
		other => panic!("expected Exp, got {:?}", other)
	}
}

/// Exponentiation is right-associative: `2^3^2` parses as `2 ^ (3 ^ 2)`. The
/// outer [`Exp`] covers the whole expression; the inner [`Exp`] is the right
/// child and carries the trailing sub-span.
#[test]
fn test_exp_right_associative_span()
{
	let (_, result) = expression(Span::new("2^3^2")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Exp(outer)) =>
		{
			assert_eq!(outer.span, span(0, 5));
			assert_eq!(outer.left.span(), span(0, 1));
			match outer.right.as_ref()
			{
				Expression::Arithmetic(ArithmeticExpression::Exp(inner)) =>
				{
					assert_eq!(inner.span, span(2, 5));
					assert_eq!(inner.left.span(), span(2, 3));
					assert_eq!(inner.right.span(), span(4, 5));
				},
				other => panic!("expected inner Exp, got {:?}", other)
			}
		},
		other => panic!("expected outer Exp, got {:?}", other)
	}
}

/// A [`Neg`] of a variable takes its span from the leading `-` through the
/// closing brace of the variable.
#[test]
fn test_neg_of_variable_span()
{
	let (_, result) = expression(Span::new("-{x}")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			assert_eq!(neg.operand.span(), span(1, 4));
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

/// A [`Neg`] of a parenthesized expression takes its span from the leading `-`
/// through the closing `)` of the group.
#[test]
fn test_neg_of_group_span()
{
	let (_, result) = expression(Span::new("-(1+2)")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 6));
			assert_eq!(neg.operand.span(), span(1, 6));
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

/// Double-negation (`-` applied to a negative constant) produces a [`Neg`]
/// wrapping a [`Constant`]. The outer [`Neg`] begins at the leading `-` and
/// extends through the constant.
#[test]
fn test_neg_of_negative_constant_span()
{
	let (_, result) = expression(Span::new("- -1")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			match neg.operand.as_ref()
			{
				Expression::Constant(c) =>
				{
					assert_eq!(c.value, -1);
					assert_eq!(c.span, span(2, 4));
				},
				other => panic!("expected Constant, got {:?}", other)
			}
		},
		other => panic!("expected outer Neg, got {:?}", other)
	}
}

/// A [`Neg`] of a dice expression (`-3D6`) takes its span from the leading `-`
/// through the end of the dice, using the general negation path because
/// [`negative_constant`] bails when followed by `d`/`D`.
#[test]
fn test_neg_of_dice_span()
{
	let (_, result) = expression(Span::new("-3D6")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			match neg.operand.as_ref()
			{
				Expression::Dice(DiceExpression::Standard(sd)) =>
				{
					assert_eq!(sd.span, span(1, 4));
				},
				other => panic!("expected StandardDice, got {:?}", other)
			}
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                       Function and parameter spans.                        //
////////////////////////////////////////////////////////////////////////////////

/// Each [`Parameter`] carries the span of its identifier, and the enclosing
/// [`Function`] covers the whole source.
#[test]
fn test_parameter_and_function_spans()
{
	let source = "x, y: 1";
	let (_, result) = function(Span::new(source)).unwrap();
	assert_eq!(result.span, span(0, 7));
	let parameters = result.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].span, span(0, 1));
	assert_eq!(parameters[1].span, span(3, 4));
	assert_eq!(result.body.span(), span(6, 7));
}

/// A [`Function`] with no parameters has its span anchored at the start of the
/// body.
#[test]
fn test_function_no_parameters_span()
{
	let (_, result) = function(Span::new("42")).unwrap();
	assert!(result.parameters.is_none());
	assert_eq!(result.span, span(0, 2));
	assert_eq!(result.body.span(), span(0, 2));
}

/// [`Parameter`] spans cover only the identifier, not surrounding whitespace or
/// separators, regardless of how much whitespace the source contains.
#[test]
fn test_parameter_span_excludes_whitespace()
{
	let source = "  abc  ,  def  : 1";
	let (_, result) = function(Span::new(source)).unwrap();
	let parameters = result.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].name, "abc");
	assert_eq!(parameters[0].span, span(2, 5));
	assert_eq!(parameters[1].name, "def");
	assert_eq!(parameters[1].span, span(10, 13));
}

/// [`Parser::parse`] strips leading whitespace before invoking [`function`], so
/// the resulting [`Function`] span begins at the first non-whitespace byte.
/// Trailing whitespace is stripped outside the body, so the span end matches
/// the body end.
#[test]
fn test_parser_parse_strips_outer_whitespace_from_function_span()
{
	let ast = Parser::parse("  1D6   ").unwrap();
	assert_eq!(ast.span, span(2, 5));
	assert_eq!(ast.body.span(), span(2, 5));
}

////////////////////////////////////////////////////////////////////////////////
//                          Spanned::span dispatch.                           //
////////////////////////////////////////////////////////////////////////////////

/// [`Expression::span`] dispatches to the active variant, returning the same
/// span as the contained node would.
#[test]
fn test_expression_span_dispatches_to_variant()
{
	// Group.
	let group = expression(Span::new("(1)")).unwrap().1;
	assert_eq!(group.span(), span(0, 3));
	assert!(matches!(group, Expression::Group(_)));

	// Constant.
	let constant_expr = expression(Span::new("7")).unwrap().1;
	assert_eq!(constant_expr.span(), span(0, 1));
	assert!(matches!(constant_expr, Expression::Constant(_)));

	// Variable.
	let variable_expr = expression(Span::new("{x}")).unwrap().1;
	assert_eq!(variable_expr.span(), span(0, 3));
	assert!(matches!(variable_expr, Expression::Variable(_)));

	// Range.
	let range_expr = expression(Span::new("[1:6]")).unwrap().1;
	assert_eq!(range_expr.span(), span(0, 5));
	assert!(matches!(range_expr, Expression::Range(_)));

	// Dice.
	let dice_expr = expression(Span::new("2D8")).unwrap().1;
	assert_eq!(dice_expr.span(), span(0, 3));
	assert!(matches!(dice_expr, Expression::Dice(_)));

	// Arithmetic.
	let arith_expr = expression(Span::new("1+2")).unwrap().1;
	assert_eq!(arith_expr.span(), span(0, 3));
	assert!(matches!(arith_expr, Expression::Arithmetic(_)));
}

/// [`DiceExpression::span`] dispatches to the active variant.
#[test]
fn test_dice_expression_span_dispatches_to_variant()
{
	let standard = dice(Span::new("2D8")).unwrap().1;
	assert_eq!(standard.span(), span(0, 3));
	assert!(matches!(standard, DiceExpression::Standard(_)));

	let custom = dice(Span::new("2D[1,2]")).unwrap().1;
	assert_eq!(custom.span(), span(0, 7));
	assert!(matches!(custom, DiceExpression::Custom(_)));

	let drop_lowest = dice(Span::new("4D6 drop lowest")).unwrap().1;
	assert_eq!(drop_lowest.span(), span(0, 15));
	assert!(matches!(drop_lowest, DiceExpression::DropLowest(_)));

	let drop_highest = dice(Span::new("4D6 drop highest")).unwrap().1;
	assert_eq!(drop_highest.span(), span(0, 16));
	assert!(matches!(drop_highest, DiceExpression::DropHighest(_)));
}

/// [`ArithmeticExpression::span`] dispatches to the active variant. The parser
/// produces these only wrapped inside [`Expression::Arithmetic`], so we unwrap
/// once to exercise the inner dispatch.
#[test]
fn test_arithmetic_expression_span_dispatches_to_variant()
{
	fn unwrap_arith(expr: Expression<'_>) -> ArithmeticExpression<'_>
	{
		match expr
		{
			Expression::Arithmetic(a) => a,
			other => panic!("expected Arithmetic, got {:?}", other)
		}
	}

	let add = unwrap_arith(expression(Span::new("1+2")).unwrap().1);
	assert_eq!(add.span(), span(0, 3));
	assert!(matches!(add, ArithmeticExpression::Add(_)));

	let sub = unwrap_arith(expression(Span::new("3-1")).unwrap().1);
	assert_eq!(sub.span(), span(0, 3));
	assert!(matches!(sub, ArithmeticExpression::Sub(_)));

	let mul = unwrap_arith(expression(Span::new("2*3")).unwrap().1);
	assert_eq!(mul.span(), span(0, 3));
	assert!(matches!(mul, ArithmeticExpression::Mul(_)));

	let div = unwrap_arith(expression(Span::new("6/2")).unwrap().1);
	assert_eq!(div.span(), span(0, 3));
	assert!(matches!(div, ArithmeticExpression::Div(_)));

	let r#mod = unwrap_arith(expression(Span::new("7%3")).unwrap().1);
	assert_eq!(r#mod.span(), span(0, 3));
	assert!(matches!(r#mod, ArithmeticExpression::Mod(_)));

	let exp = unwrap_arith(expression(Span::new("2^3")).unwrap().1);
	assert_eq!(exp.span(), span(0, 3));
	assert!(matches!(exp, ArithmeticExpression::Exp(_)));

	let neg = unwrap_arith(expression(Span::new("-{x}")).unwrap().1);
	assert_eq!(neg.span(), span(0, 4));
	assert!(matches!(neg, ArithmeticExpression::Neg(_)));
}

/// [`Function::span`] and [`Parameter::span`] expose the struct field through
/// the [`Spanned`] trait. Other tests access the field directly; this one
/// exercises the trait dispatch specifically.
#[test]
fn test_function_and_parameter_spanned_trait_dispatch()
{
	let ast = Parser::parse("a, b: 1").unwrap();
	// Exercise the trait method, not the struct field.
	assert_eq!(Spanned::span(&ast), span(0, 7));
	let params = ast.parameters.as_ref().unwrap();
	assert_eq!(Spanned::span(&params[0]), span(0, 1));
	assert_eq!(Spanned::span(&params[1]), span(3, 4));
}

////////////////////////////////////////////////////////////////////////////////
//                       Spanned::untethered coverage.                        //
////////////////////////////////////////////////////////////////////////////////

/// [`Spanned::untethered`] zeroes every span in a complex AST — including
/// parameter spans, arithmetic operand spans, dice wrapper spans, and nested
/// variable spans — without altering the structure or identifiers.
#[test]
fn test_untethered_zeroes_all_spans()
{
	let ast = Parser::parse("a, b: {a}D6 drop lowest 1 + {b}").unwrap();
	assert_ne!(ast.span, SourceSpan::default());
	let untethered = ast.untethered();

	assert_eq!(untethered.span, SourceSpan::default());
	let parameters = untethered.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].name, "a");
	assert_eq!(parameters[0].span, SourceSpan::default());
	assert_eq!(parameters[1].name, "b");
	assert_eq!(parameters[1].span, SourceSpan::default());

	// Walk the body: Add(DropLowest(Standard(Variable, Constant), Some(1)),
	// Variable).
	match untethered.body
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.span, SourceSpan::default());
			match *add.left
			{
				Expression::Dice(DiceExpression::DropLowest(drop)) =>
				{
					assert_eq!(drop.span, SourceSpan::default());
					let drop_count =
						drop.drop.as_ref().expect("explicit drop count");
					assert_eq!(drop_count.span(), SourceSpan::default());
					match *drop.dice
					{
						DiceExpression::Standard(sd) =>
						{
							assert_eq!(sd.span, SourceSpan::default());
							assert_eq!(sd.count.span(), SourceSpan::default());
							assert_eq!(sd.faces.span(), SourceSpan::default());
						},
						other =>
						{
							panic!("expected Standard, got {:?}", other)
						}
					}
				},
				other => panic!("expected DropLowest, got {:?}", other)
			}
			match *add.right
			{
				Expression::Variable(v) =>
				{
					assert_eq!(v.name, "b");
					assert_eq!(v.span, SourceSpan::default());
				},
				other => panic!("expected Variable, got {:?}", other)
			}
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// [`Spanned::untethered`] on a parameter-less function leaves `parameters` as
/// `None` rather than materializing an empty vector.
#[test]
fn test_untethered_preserves_none_parameters()
{
	let ast = Parser::parse("1D6").unwrap();
	let untethered = ast.untethered();
	assert!(untethered.parameters.is_none());
	assert_eq!(untethered.span, SourceSpan::default());
}

/// [`Spanned::untethered`] on a [`CustomDice`] preserves the face vector
/// exactly (faces are raw `i32` values with no span of their own).
#[test]
fn test_untethered_custom_dice_preserves_faces()
{
	let ast = Parser::parse("2D[-1,0,1]").unwrap();
	let untethered = ast.untethered();
	match untethered.body
	{
		Expression::Dice(DiceExpression::Custom(cd)) =>
		{
			assert_eq!(cd.faces, vec![-1, 0, 1]);
			assert_eq!(cd.span, SourceSpan::default());
			assert_eq!(cd.count.span(), SourceSpan::default());
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// [`Spanned::untethered`] on a [`DropHighest`] without an explicit count keeps
/// the `drop` field as `None` (the optional payload is preserved verbatim).
#[test]
fn test_untethered_drop_highest_no_count()
{
	let ast = Parser::parse("4D6 drop highest").unwrap();
	let untethered = ast.untethered();
	match untethered.body
	{
		Expression::Dice(DiceExpression::DropHighest(drop)) =>
		{
			assert!(drop.drop.is_none());
			assert_eq!(drop.span, SourceSpan::default());
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// [`Spanned::untethered`] applied twice is idempotent: the second application
/// is a no-op on already-zeroed spans.
#[test]
fn test_untethered_is_idempotent()
{
	let ast = Parser::parse("1 + {x}").unwrap();
	let once = ast.untethered();
	let twice = once.untethered();
	assert_eq!(once, twice);
}

/// A structurally identical AST built by hand with default spans compares equal
/// to the [`Spanned::untethered`] form of the parser-originated AST.
#[test]
fn test_untethered_matches_hand_built_equivalent()
{
	let untethered = Parser::parse("{x} + 1").unwrap().untethered();
	let handbuilt = Function {
		parameters: None,
		body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
			left: Box::new(Expression::Variable(Variable {
				name: "x",
				span: SourceSpan::default()
			})),
			right: Box::new(Expression::Constant(Constant {
				value: 1,
				span: SourceSpan::default()
			})),
			span: SourceSpan::default()
		})),
		span: SourceSpan::default()
	};
	assert_eq!(untethered, handbuilt);
}
