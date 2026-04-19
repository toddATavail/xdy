//! # Addition and subtraction combinator test cases
//!
//! Herein are the test cases for the [`add_sub`] combinator.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                 Addition and subtraction combinator tests.                 //
////////////////////////////////////////////////////////////////////////////////

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
