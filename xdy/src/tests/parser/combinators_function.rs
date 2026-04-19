//! # Function-level combinator test cases
//!
//! Herein are the test cases for the [`function`], [`parameters`], and
//! [`parameter`] combinators.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                      Function-level combinator tests.                      //
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
