//! # Dice combinator test cases
//!
//! Herein are the test cases for the [`dice`] combinator and its
//! sub-combinators: [`standard_dice`], [`custom_dice`], [`dice_count`],
//! [`standard_faces`], [`custom_faces`], [`drop_lowest`], [`drop_highest`], and
//! [`drop_expression`].

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                           Dice combinator tests.                           //
////////////////////////////////////////////////////////////////////////////////

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
