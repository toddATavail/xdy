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
