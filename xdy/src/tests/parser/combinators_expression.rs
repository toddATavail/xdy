//! # Top-level expression combinator test cases
//!
//! Herein are the test cases for the [`expression`], [`exponent`], and
//! [`unary`] combinators.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                   Top-level expression combinator tests.                   //
////////////////////////////////////////////////////////////////////////////////

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
