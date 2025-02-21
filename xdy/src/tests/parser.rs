//! # Parser test cases
//!
//! Herein are the test cases for the various parser functions.

use crate::{ast::*, parser::*};
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
				parameters: Some(vec!["x"]),
				body: Expression::Constant(Constant(42))
			}
		),
		(
			"3d6",
			"3D6",
			Function {
				parameters: None,
				body: Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				))
			}
		),
		(
			"x, y: {x} + {y}",
			"x, y: {x} + {y}",
			Function {
				parameters: Some(vec!["x", "y"]),
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Variable(Variable("x"))),
					right: Box::new(Expression::Variable(Variable("y")))
				}))
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
							count: Box::new(Expression::Constant(Constant(2))),
							faces: Box::new(Expression::Constant(Constant(20)))
						}
					))),
					right: Box::new(Expression::Constant(Constant(5)))
				}))
			}
		),
		(
			"a, b, c: ({a} + {b}) * {c}",
			"a, b, c: ({a} + {b}) * {c}",
			Function {
				parameters: Some(vec!["a", "b", "c"]),
				body: Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
					left: Box::new(Expression::Group(Group {
						expression: Box::new(Expression::Arithmetic(
							ArithmeticExpression::Add(Add {
								left: Box::new(Expression::Variable(Variable(
									"a"
								))),
								right: Box::new(Expression::Variable(
									Variable("b")
								))
							})
						))
					})),
					right: Box::new(Expression::Variable(Variable("c")))
				}))
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
									Constant(4)
								)),
								faces: Box::new(Expression::Constant(
									Constant(6)
								))
							}
						)),
						drop: None
					}
				))
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
									Constant(4)
								)),
								faces: Box::new(Expression::Constant(
									Constant(6)
								))
							}
						)),
						drop: Some(Box::new(Expression::Variable(Variable(
							"z"
						))))
					}
				))
			}
		),
		(
			"x: {x} * 2 + 1",
			"x: {x} * 2 + 1",
			Function {
				parameters: Some(vec!["x"]),
				body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
					left: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Mul(Mul {
							left: Box::new(Expression::Variable(Variable("x"))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					)),
					right: Box::new(Expression::Constant(Constant(1)))
				}))
			}
		),
		(
			"a, b: [{a}:{b}]",
			"a, b: [{a}:{b}]",
			Function {
				parameters: Some(vec!["a", "b"]),
				body: Expression::Range(Range {
					start: Box::new(Expression::Variable(Variable("a"))),
					end: Box::new(Expression::Variable(Variable("b")))
				})
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
							count: Box::new(Expression::Constant(Constant(3))),
							faces: vec![1, 2, 3]
						}
					))),
					right: Box::new(Expression::Constant(Constant(5)))
				}))
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
					result, expected_ast,
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
	for (input, expected_str, expected_ast) in [
		("x:", vec!["x"], vec!["x"]),
		("x, y:", vec!["x", "y"], vec!["x", "y"]),
		("x, y, z:", vec!["x", "y", "z"], vec!["x", "y", "z"]),
		(
			"long_variable_name:",
			vec!["long_variable_name"],
			vec!["long_variable_name"]
		),
		(
			"x1, x2, x3:",
			vec!["x1", "x2", "x3"],
			vec!["x1", "x2", "x3"]
		),
		("hello-world:", vec!["hello-world"], vec!["hello-world"]),
		("x, y, z:", vec!["x", "y", "z"], vec!["x", "y", "z"]),
		(" x , y , z :", vec!["x", "y", "z"], vec!["x", "y", "z"])
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
				assert_eq!(result, expected_str, "Failed for input: {}", input);
				assert_eq!(
					result, expected_ast,
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
		("42", "42", Expression::Constant(Constant(42))),
		(
			"-42",
			"-42",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(42)))
			}))
		),
		("{x}", "{x}", Expression::Variable(Variable("x"))),
		(
			"(1 + 2)",
			"(1 + 2)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
			})
		),
		(
			"[1:10]",
			"[1:10]",
			Expression::Range(Range {
				start: Box::new(Expression::Constant(Constant(1))),
				end: Box::new(Expression::Constant(Constant(10)))
			})
		),
		(
			"3d6",
			"3D6",
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Constant(Constant(3))),
				faces: Box::new(Expression::Constant(Constant(6)))
			}))
		),
		(
			"2 + 3",
			"2 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"4 - 2",
			"4 - 2",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Constant(Constant(4))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"3 * 5",
			"3 * 5",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(3))),
				right: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"10 / 2",
			"10 / 2",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant(10))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"7 % 3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant(7))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2 ^ 3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"-2 ^ 3",
			"-2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2 + 3 * 4",
			"2 + 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(3))),
						right: Box::new(Expression::Constant(Constant(4)))
					})
				))
			}))
		),
		(
			"(2 + 3) * 4",
			"(2 + 3) * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(2))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				})),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"1d20 + 5",
			"1D20 + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(1))),
						faces: Box::new(Expression::Constant(Constant(20)))
					}
				))),
				right: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"3d6 drop lowest",
			"3D6 drop lowest",
			Expression::Dice(DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant(3))),
					faces: Box::new(Expression::Constant(Constant(6)))
				})),
				drop: None
			}))
		),
		(
			"4d6 drop highest 1",
			"4D6 drop highest 1",
			Expression::Dice(DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant(4))),
					faces: Box::new(Expression::Constant(Constant(6)))
				})),
				drop: Some(Box::new(Expression::Constant(Constant(1))))
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
					result, expected_ast,
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
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(5)))
					})
				))
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable("x")))
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				}))
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				)))
			}))
		),
		("2", "2", Expression::Constant(Constant(2))),
		(
			"-2",
			"-2",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"-2^-3",
			"-2 ^ -3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(3)))
					})
				))
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant(3))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant(2))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				})),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"2*3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"6/3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant(6))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"7%3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant(7))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2×3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"6÷3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant(6))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2*3*4",
			"2 * 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"12/4/3",
			"12 / 4 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant(12))),
						right: Box::new(Expression::Constant(Constant(4)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"10%3%2",
			"10 % 3 % 2",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant(10))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2*3/4",
			"2 * 3 / 4",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"6/2*3",
			"6 / 2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant(6))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"8%3*2",
			"8 % 3 * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant(8))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"{x}*2",
			"{x} * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2*(3+4)",
			"2 * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"(1+2)*(3+4)",
			"(1 + 2) * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"5 + 3",
			"5 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(5))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"5 - 3",
			"5 - 3",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Constant(Constant(5))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"5 + 3 - 2",
			"5 + 3 - 2",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(5))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"5 - 3 + 2",
			"5 - 3 + 2",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Sub(Sub {
						left: Box::new(Expression::Constant(Constant(5))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"1 + 2 + 3",
			"1 + 2 + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"1 - 2 - 3",
			"1 - 2 - 3",
			Expression::Arithmetic(ArithmeticExpression::Sub(Sub {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Sub(Sub {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"{x} + 5",
			"{x} + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"5 + {x}",
			"5 + {x}",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(5))),
				right: Box::new(Expression::Variable(Variable("x")))
			}))
		),
		(
			"(1 + 2) + 3",
			"(1 + 2) + 3",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"1 + (2 + 3)",
			"1 + (2 + 3)",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(1))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(2))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				}))
			}))
		),
		(
			"3d6 + 5",
			"3D6 + 5",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				))),
				right: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"5 + 3d6",
			"5 + 3D6",
			Expression::Arithmetic(ArithmeticExpression::Add(Add {
				left: Box::new(Expression::Constant(Constant(5))),
				right: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				)))
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
									Constant(1)
								)),
								faces: Box::new(Expression::Constant(
									Constant(20)
								))
							})
						)),
						right: Box::new(Expression::Constant(Constant(5)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
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
					result, expected_ast,
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
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(5)))
					})
				))
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable("x")))
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				}))
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				)))
			}))
		),
		("2", "2", Expression::Constant(Constant(2))),
		(
			"-2",
			"-2",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"-2^-3",
			"-2 ^ -3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(3)))
					})
				))
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant(3))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant(2))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				})),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"2*3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"6/3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant(6))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"7%3",
			"7 % 3",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Constant(Constant(7))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2×3",
			"2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"6÷3",
			"6 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Constant(Constant(6))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"2*3*4",
			"2 * 3 * 4",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"12/4/3",
			"12 / 4 / 3",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant(12))),
						right: Box::new(Expression::Constant(Constant(4)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"10%3%2",
			"10 % 3 % 2",
			Expression::Arithmetic(ArithmeticExpression::Mod(Mod {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant(10))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2*3/4",
			"2 * 3 / 4",
			Expression::Arithmetic(ArithmeticExpression::Div(Div {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(4)))
			}))
		),
		(
			"6/2*3",
			"6 / 2 * 3",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Div(Div {
						left: Box::new(Expression::Constant(Constant(6))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"8%3*2",
			"8 % 3 * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mod(Mod {
						left: Box::new(Expression::Constant(Constant(8))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				)),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"{x}*2",
			"{x} * 2",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2*(3+4)",
			"2 * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"(1+2)*(3+4)",
			"(1 + 2) * (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Mul(Mul {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
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
					result, expected_ast,
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

/// Ensure that [`exponent`] behaves as expected.
#[test]
fn test_exponent()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"-5",
			"-5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(5)))
					})
				))
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable("x")))
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				}))
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				)))
			}))
		),
		("2", "2", Expression::Constant(Constant(2))),
		(
			"-2",
			"-2",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"2^3",
			"2 ^ 3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Constant(Constant(3)))
			}))
		),
		(
			"-2^-3",
			"-2 ^ -3",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(3)))
					})
				))
			}))
		),
		(
			"2^3^2",
			"2 ^ 3 ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Exp(Exp {
						left: Box::new(Expression::Constant(Constant(3))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
			}))
		),
		(
			"{x}^2",
			"{x} ^ 2",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Variable(Variable("x"))),
				right: Box::new(Expression::Constant(Constant(2)))
			}))
		),
		(
			"(1+2)^(3+4)",
			"(1 + 2) ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"2^(3+4)",
			"2 ^ (3 + 4)",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Constant(Constant(2))),
				right: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}))
		),
		(
			"(2^3)^4",
			"(2 ^ 3) ^ 4",
			Expression::Arithmetic(ArithmeticExpression::Exp(Exp {
				left: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Exp(Exp {
							left: Box::new(Expression::Constant(Constant(2))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				})),
				right: Box::new(Expression::Constant(Constant(4)))
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
					result, expected_ast,
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
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Constant(Constant(5)))
			}))
		),
		(
			"- - 5",
			"--5",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(5)))
					})
				))
			}))
		),
		(
			"-{x}",
			"-{x}",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Variable(Variable("x")))
			}))
		),
		(
			"-(1 + 2)",
			"-(1 + 2)",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				}))
			}))
		),
		(
			"-3d6",
			"-3D6",
			Expression::Arithmetic(ArithmeticExpression::Neg(Neg {
				operand: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				)))
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
					result, expected_ast,
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
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
			})
		),
		("{x}", "{x}", Expression::Variable(Variable("x"))),
		(
			"[1:10]",
			"[1:10]",
			Expression::Range(Range {
				start: Box::new(Expression::Constant(Constant(1))),
				end: Box::new(Expression::Constant(Constant(10)))
			})
		),
		(
			"3d6",
			"3D6",
			Expression::Dice(DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Constant(Constant(3))),
				faces: Box::new(Expression::Constant(Constant(6)))
			}))
		),
		("42", "42", Expression::Constant(Constant(42))),
		("-42", "-42", Expression::Constant(Constant(-42)))
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
					result, expected_ast,
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
				expression: Box::new(Expression::Constant(Constant(1)))
			}
		),
		(
			"(1 + 2)",
			"(1 + 2)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
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
										Constant(1)
									)),
									right: Box::new(Expression::Constant(
										Constant(2)
									))
								})
							))
						})),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				))
			}
		),
		(
			"({x})",
			"({x})",
			Group {
				expression: Box::new(Expression::Variable(Variable("x")))
			}
		),
		(
			"(3d6)",
			"(3D6)",
			Group {
				expression: Box::new(Expression::Dice(
					DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					})
				))
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
									Constant(1)
								)),
								faces: Box::new(Expression::Constant(
									Constant(20)
								))
							})
						)),
						right: Box::new(Expression::Constant(Constant(5)))
					})
				))
			}
		),
		(
			"( 1 )",
			"(1)",
			Group {
				expression: Box::new(Expression::Constant(Constant(1)))
			}
		),
		(
			"(  1  +  2  )",
			"(1 + 2)",
			Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
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
					result, expected_ast,
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
		("{x-1}", "{x-1}")
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
				start: Box::new(Expression::Constant(Constant(1))),
				end: Box::new(Expression::Constant(Constant(10)))
			}
		),
		(
			"[-10:10]",
			"[-10:10]",
			Range {
				start: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Neg(Neg {
						operand: Box::new(Expression::Constant(Constant(10)))
					})
				)),
				end: Box::new(Expression::Constant(Constant(10)))
			}
		),
		(
			"[{min}:{max}]",
			"[{min}:{max}]",
			Range {
				start: Box::new(Expression::Variable(Variable("min"))),
				end: Box::new(Expression::Variable(Variable("max")))
			}
		),
		(
			"[1+2:3*4]",
			"[1 + 2:3 * 4]",
			Range {
				start: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				)),
				end: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Mul(Mul {
						left: Box::new(Expression::Constant(Constant(3))),
						right: Box::new(Expression::Constant(Constant(4)))
					})
				))
			}
		),
		(
			"[(1+2):(3*4)]",
			"[(1 + 2):(3 * 4)]",
			Range {
				start: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				end: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Mul(Mul {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
			}
		),
		(
			"[1d6:2d8]",
			"[1D6:2D8]",
			Range {
				start: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(1))),
						faces: Box::new(Expression::Constant(Constant(6)))
					}
				))),
				end: Box::new(Expression::Dice(DiceExpression::Standard(
					StandardDice {
						count: Box::new(Expression::Constant(Constant(2))),
						faces: Box::new(Expression::Constant(Constant(8)))
					}
				)))
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
					result, expected_ast,
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
				count: Box::new(Expression::Constant(Constant(3))),
				faces: Box::new(Expression::Constant(Constant(6)))
			})
		),
		(
			"2D20",
			"2D20",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Constant(Constant(2))),
				faces: Box::new(Expression::Constant(Constant(20)))
			})
		),
		(
			"3D[1,2,3]",
			"3D[1, 2, 3]",
			DiceExpression::Custom(CustomDice {
				count: Box::new(Expression::Constant(Constant(3))),
				faces: vec![1, 2, 3]
			})
		),
		(
			"4d6 drop lowest",
			"4D6 drop lowest",
			DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant(4))),
					faces: Box::new(Expression::Constant(Constant(6)))
				})),
				drop: None
			})
		),
		(
			"5D20 drop highest 2",
			"5D20 drop highest 2",
			DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::Standard(StandardDice {
					count: Box::new(Expression::Constant(Constant(5))),
					faces: Box::new(Expression::Constant(Constant(20)))
				})),
				drop: Some(Box::new(Expression::Constant(Constant(2))))
			})
		),
		(
			"3d6 drop lowest 1 drop highest 1",
			"3D6 drop lowest 1 drop highest 1",
			DiceExpression::DropHighest(DropHighest {
				dice: Box::new(DiceExpression::DropLowest(DropLowest {
					dice: Box::new(DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					})),
					drop: Some(Box::new(Expression::Constant(Constant(1))))
				})),
				drop: Some(Box::new(Expression::Constant(Constant(1))))
			})
		),
		(
			"3d6 drop highest drop lowest",
			"3D6 drop highest drop lowest",
			DiceExpression::DropLowest(DropLowest {
				dice: Box::new(DiceExpression::DropHighest(DropHighest {
					dice: Box::new(DiceExpression::Standard(StandardDice {
						count: Box::new(Expression::Constant(Constant(3))),
						faces: Box::new(Expression::Constant(Constant(6)))
					})),
					drop: None
				})),
				drop: None
			})
		),
		(
			"{count}d{faces}",
			"{count}D{faces}",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Variable(Variable("count"))),
				faces: Box::new(Expression::Variable(Variable("faces")))
			})
		),
		(
			"(1+2)D(3+3)",
			"(1 + 2)D(3 + 3)",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				}))
			})
		),
		(
			"(3d6)D(1d4)",
			"(3D6)D(1D4)",
			DiceExpression::Standard(StandardDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant(3))),
							faces: Box::new(Expression::Constant(Constant(6)))
						})
					))
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant(1))),
							faces: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
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
					result, expected_ast,
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
				count: Box::new(Expression::Constant(Constant(3))),
				faces: Box::new(Expression::Constant(Constant(6)))
			}
		),
		(
			"2D20",
			"2D20",
			StandardDice {
				count: Box::new(Expression::Constant(Constant(2))),
				faces: Box::new(Expression::Constant(Constant(20)))
			}
		),
		(
			"{count}d{faces}",
			"{count}D{faces}",
			StandardDice {
				count: Box::new(Expression::Variable(Variable("count"))),
				faces: Box::new(Expression::Variable(Variable("faces")))
			}
		),
		(
			"(1+2)D(3+3)",
			"(1 + 2)D(3 + 3)",
			StandardDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(3))),
							right: Box::new(Expression::Constant(Constant(3)))
						})
					))
				}))
			}
		),
		(
			"(3d6)D(1d4)",
			"(3D6)D(1D4)",
			StandardDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant(3))),
							faces: Box::new(Expression::Constant(Constant(6)))
						})
					))
				})),
				faces: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant(1))),
							faces: Box::new(Expression::Constant(Constant(4)))
						})
					))
				}))
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
					result, expected_ast,
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
				count: Box::new(Expression::Constant(Constant(3))),
				faces: vec![1, 2, 3]
			}
		),
		(
			"{count}D[1, 2, 3]",
			"{count}D[1, 2, 3]",
			CustomDice {
				count: Box::new(Expression::Variable(Variable("count"))),
				faces: vec![1, 2, 3]
			}
		),
		(
			"(1+2)D[ 1 , 2 , 3 ]",
			"(1 + 2)D[1, 2, 3]",
			CustomDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Arithmetic(
						ArithmeticExpression::Add(Add {
							left: Box::new(Expression::Constant(Constant(1))),
							right: Box::new(Expression::Constant(Constant(2)))
						})
					))
				})),
				faces: vec![1, 2, 3]
			}
		),
		(
			"2D[10]",
			"2D[10]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant(2))),
				faces: vec![10]
			}
		),
		(
			"1D[-1,0,1]",
			"1D[-1, 0, 1]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant(1))),
				faces: vec![-1, 0, 1]
			}
		),
		(
			"4D[-3, -2, -1]",
			"4D[-3, -2, -1]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant(4))),
				faces: vec![-3, -2, -1]
			}
		),
		(
			"5D[100, 200, 300]",
			"5D[100, 200, 300]",
			CustomDice {
				count: Box::new(Expression::Constant(Constant(5))),
				faces: vec![100, 200, 300]
			}
		),
		(
			"(3d6)d[-1,0,1]",
			"(3D6)D[-1, 0, 1]",
			CustomDice {
				count: Box::new(Expression::Group(Group {
					expression: Box::new(Expression::Dice(
						DiceExpression::Standard(StandardDice {
							count: Box::new(Expression::Constant(Constant(3))),
							faces: Box::new(Expression::Constant(Constant(6)))
						})
					))
				})),
				faces: vec![-1, 0, 1]
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
					result, expected_ast,
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
		("3", "3", Expression::Constant(Constant(3))),
		(
			"{count}",
			"{count}",
			Expression::Variable(Variable("count"))
		),
		(
			"(1+2)",
			"(1 + 2)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(1))),
						right: Box::new(Expression::Constant(Constant(2)))
					})
				))
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
					result, expected_ast,
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
		("6", "6", Expression::Constant(Constant(6))),
		("20", "20", Expression::Constant(Constant(20))),
		(
			"{faces}",
			"{faces}",
			Expression::Variable(Variable("faces"))
		),
		(
			"(2+3)",
			"(2 + 3)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				))
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
					result, expected_ast,
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
		("42", "42", Expression::Constant(Constant(42))),
		("-42", "-42", Expression::Constant(Constant(-42))),
		("{var}", "{var}", Expression::Variable(Variable("var"))),
		(
			"(2+3)",
			"(2 + 3)",
			Expression::Group(Group {
				expression: Box::new(Expression::Arithmetic(
					ArithmeticExpression::Add(Add {
						left: Box::new(Expression::Constant(Constant(2))),
						right: Box::new(Expression::Constant(Constant(3)))
					})
				))
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
					result, expected_ast,
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
		("0", "0", Constant(0)),
		("42", "42", Constant(42)),
		("-42", "-42", Constant(-42)),
		("9999", "9999", Constant(9999)),
		("-9999", "-9999", Constant(-9999)),
		("2147483647", "2147483647", Constant(2147483647)), // i32::MAX
		("-2147483648", "-2147483648", Constant(-2147483648))  // i32::MIN
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
					result, expected_ast,
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
		" 42",
		"a",
		"+42",
		"2147483648",  // i32::MAX + 1
		"-2147483649", // i32::MIN - 1
		"--42",
		"a42",
		"42a"
	]
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
		("a", "a")
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
