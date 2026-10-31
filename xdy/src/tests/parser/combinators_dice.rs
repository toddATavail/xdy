//! # Dice combinator test cases
//!
//! Herein are the test cases for the [`dice`] combinator and its
//! sub-combinators: [`standard_dice`], [`custom_dice`], [`dice_count`],
//! [`standard_faces`], [`custom_faces`], [`drop_lowest`], [`drop_highest`],
//! [`keep_lowest`], [`keep_highest`], [`reroll`], [`explode`], and
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
					name: "count".into(),
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Variable(Variable {
					name: "faces".into(),
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
		"3d6 drop middle",
		// A dice count never begins with `-`.
		"-3d6",
		"- 3d6",
		"-3d[1,2]",
		"-3d6 drop lowest"
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
			"3d-6",
			"3D-6",
			StandardDice {
				count: Box::new(Expression::Constant(Constant {
					value: 3,
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Constant(Constant {
					value: -6,
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
					name: "count".into(),
					span: SourceSpan::default()
				})),
				faces: Box::new(Expression::Variable(Variable {
					name: "faces".into(),
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
	// A dice count never begins with `-`.
	for input in [
		"",
		" ",
		"d6",
		"3",
		"3d",
		"d",
		"3d[1,2,3]",
		"3+4d6",
		"(3d6)",
		"-3d6",
		"- 3d6"
	]
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
					name: "count".into(),
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
		"3D(1,2,3)",
		// A dice count never begins with `-`.
		"-3D[1,2,3]",
		"- 3D[1,2,3]"
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
				name: "count".into(),
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
	// A dice count never begins with `-`, since its constant is unsigned.
	for input in [
		"", " ", " 3", "d6", "2d6", "[1,2,3]", "2+3", "a", "-3", "-0", "- 3",
		"-{count}"
	]
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
		// Unlike a dice count, standard faces may be a signed integer.
		(
			"-6",
			"-6",
			Expression::Constant(Constant {
				value: -6,
				span: SourceSpan::default()
			})
		),
		(
			"{faces}",
			"{faces}",
			Expression::Variable(Variable {
				name: "faces".into(),
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
	for input in [
		"", " ", " 3", "d6", "2d6", "[1,2,3]", "2+3", "a", "- 6", "--6"
	]
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

/// Ensure that [`dice`] reads keep clauses, alone and chained with drop
/// clauses in either order, each modifying the clauses before it, and that it
/// rejects a keep clause without a direction, or with a count that begins with
/// `-`, which ends the dice expression before the `-`.
#[test]
fn test_dice_keep_clauses()
{
	for (input, rendering, s_expr) in [
		(
			"4d6 keep highest",
			"4D6 keep highest",
			"(keep-highest (standard-dice 4 6))"
		),
		(
			"4d6 keep lowest 3",
			"4D6 keep lowest 3",
			"(keep-lowest (standard-dice 4 6) 3)"
		),
		(
			"2D20 keep highest {n}",
			"2D20 keep highest {n}",
			"(keep-highest (standard-dice 2 20) {n})"
		),
		(
			"{x}d[1,2,3] keep lowest (1+1)",
			"{x}D[1, 2, 3] keep lowest (1 + 1)",
			"(keep-lowest (custom-dice {x} [1 2 3]) (group (add 1 1)))"
		),
		(
			"5d10 drop lowest keep highest 3",
			"5D10 drop lowest keep highest 3",
			"(keep-highest (drop-lowest (standard-dice 5 10)) 3)"
		),
		(
			"5d10 keep highest 3 drop lowest",
			"5D10 keep highest 3 drop lowest",
			"(drop-lowest (keep-highest (standard-dice 5 10) 3))"
		),
		(
			"5d10 keep highest 4 keep lowest 2 drop highest 1",
			"5D10 keep highest 4 keep lowest 2 drop highest 1",
			"(drop-highest (keep-lowest (keep-highest (standard-dice 5 10) 4) \
			 2) 1)"
		),
		(
			"4d6\nkeep\thighest\n3",
			"4D6 keep highest 3",
			"(keep-highest (standard-dice 4 6) 3)"
		),
		(
			"4d6 keeplowest",
			"4D6 keep lowest",
			"(keep-lowest (standard-dice 4 6))"
		)
	]
	{
		let (residue, result) = dice(Span::new(input))
			.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
		assert!(residue.fragment().is_empty(), "residue for {input:?}");
		assert_eq!(result.to_string(), rendering, "{input:?}");
		let source = format!("(function [] {s_expr})");
		let expected = crate::s_expr::read_s_expr(&source).unwrap();
		let Expression::Dice(expected) = &expected.body
		else
		{
			unreachable!()
		};
		assert!(
			result.untethered() == expected.untethered(),
			"AST mismatch for {input:?}"
		);
	}
	for input in [
		"keep lowest",
		"3d6 keep",
		"3d6 keep middle",
		"3 keep lowest"
	]
	{
		let result = dice(Span::new(input));
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
	let (residue, result) = dice(Span::new("4d6 keep lowest -1")).unwrap();
	assert_eq!(result.to_string(), "4D6 keep lowest");
	assert_eq!(*residue.fragment(), " -1");
}

/// Ensure that [`keep_lowest`] and [`keep_highest`] behave as expected.
#[test]
fn test_keep_lowest_and_highest()
{
	type Combinator = fn(
		Span<'_>
	) -> nom::IResult<
		Span<'_>,
		Option<Box<Expression<'_>>>,
		ParseError<'_>
	>;
	for (combinator, direction, other) in [
		(keep_lowest as Combinator, "lowest", "highest"),
		(keep_highest as Combinator, "highest", "lowest")
	]
	{
		for (input, expected) in [
			(format!("keep {direction}"), None),
			(format!("keep {direction} 2"), Some("2")),
			(format!("keep {direction} {{num}}"), Some("{num}")),
			(format!("keep  {direction}  (3)"), Some("(3)"))
		]
		{
			let (residue, result) = combinator(Span::new(&input))
				.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
			assert!(residue.fragment().is_empty(), "residue for {input:?}");
			assert_eq!(
				result.map(|count| count.to_string()).as_deref(),
				expected,
				"{input:?}"
			);
		}
		for input in [
			String::new(),
			" ".into(),
			"2d6 keep".into(),
			format!("keep {other}"),
			format!("drop {direction}")
		]
		{
			let result = combinator(Span::new(&input));
			assert!(
				result.is_err() || !result.unwrap().0.fragment().is_empty(),
				"Failed to reject invalid input: {}",
				input
			);
		}
	}
}

/// Ensure that [`dice`] reads reroll clauses: every comparator, in ASCII and
/// in Unicode, with literal, negative, named, bound, and computed thresholds;
/// lists, empty or not, of negative literals too; half-open and inclusive
/// ranges, with endpoints of every kind; limits of `time` or `times`, and of
/// `∞` or `unlimited`; with and without whitespace between their tokens; and
/// chained with drops, keeps, and other rerolls in any order.
#[test]
fn test_dice_reroll_clauses()
{
	for (input, rendering, s_expr) in [
		(
			"3d6 reroll = 1",
			"3D6 reroll = 1",
			"(reroll (standard-dice 3 6) (eq 1))"
		),
		(
			"3d6 reroll=1",
			"3D6 reroll = 1",
			"(reroll (standard-dice 3 6) (eq 1))"
		),
		(
			"3d6 reroll != 1 2 times",
			"3D6 reroll ≠ 1 2 times",
			"(reroll (standard-dice 3 6) (ne 1) (times 2))"
		),
		(
			"3d6 reroll ≠ -1 1 time",
			"3D6 reroll ≠ -1 1 time",
			"(reroll (standard-dice 3 6) (ne -1) (times 1))"
		),
		(
			"3d6 reroll < 3 1 times",
			"3D6 reroll < 3 1 time",
			"(reroll (standard-dice 3 6) (lt 3) (times 1))"
		),
		(
			"3d6 reroll <= 2 ∞ times",
			"3D6 reroll ≤ 2 ∞ times",
			"(reroll (standard-dice 3 6) (le 2) (unlimited))"
		),
		(
			"3d6 reroll ≤ {x} unlimited time",
			"3D6 reroll ≤ {x} ∞ times",
			"(reroll (standard-dice 3 6) (le {x}) (unlimited))"
		),
		(
			"3d6 reroll > 5 {n} times",
			"3D6 reroll > 5 {n} times",
			"(reroll (standard-dice 3 6) (gt 5) (times {n}))"
		),
		(
			"3d6 reroll >= (1+1) 2 time",
			"3D6 reroll ≥ (1 + 1) 2 times",
			"(reroll (standard-dice 3 6) (ge (group (add 1 1))) (times 2))"
		),
		(
			"3d6 reroll ≥ {a}@(6)",
			"3D6 reroll ≥ {a}@(6)",
			"(reroll (standard-dice 3 6) (ge (binding {a} 6)))"
		),
		(
			"3d[-1,0,1] reroll in [-1, 1]",
			"3D[-1, 0, 1] reroll in [-1, 1]",
			"(reroll (custom-dice 3 [-1 0 1]) (in [-1 1]))"
		),
		(
			"3d6 reroll in []",
			"3D6 reroll in []",
			"(reroll (standard-dice 3 6) (in []))"
		),
		(
			"3d6 reroll in[1,2] unlimited times",
			"3D6 reroll in [1, 2] ∞ times",
			"(reroll (standard-dice 3 6) (in [1 2]) (unlimited))"
		),
		(
			"3d[-3,-2,-1] reroll in -3..-1",
			"3D[-3, -2, -1] reroll in -3..-1",
			"(reroll (custom-dice 3 [-3 -2 -1]) (in-range -3 -1))"
		),
		(
			"3d6 reroll in 1..=2 2 times",
			"3D6 reroll in 1..=2 2 times",
			"(reroll (standard-dice 3 6) (in-range-inclusive 1 2) (times 2))"
		),
		(
			"3d6 reroll in1..={x}",
			"3D6 reroll in 1..={x}",
			"(reroll (standard-dice 3 6) (in-range-inclusive 1 {x}))"
		),
		(
			"3d6 reroll in ({x}-1)..=({x}+1) ({x}) times",
			"3D6 reroll in ({x} - 1)..=({x} + 1) ({x}) times",
			"(reroll (standard-dice 3 6) (in-range-inclusive (group (sub {x} \
			 1)) (group (add {x} 1))) (times (group {x})))"
		),
		(
			"4d6 drop lowest reroll = 1 keep highest 2",
			"4D6 drop lowest reroll = 1 keep highest 2",
			"(keep-highest (reroll (drop-lowest (standard-dice 4 6)) (eq 1)) 2)"
		),
		(
			"4d6 reroll = 1 reroll = 2 1 time drop highest",
			"4D6 reroll = 1 reroll = 2 1 time drop highest",
			"(drop-highest (reroll (reroll (standard-dice 4 6) (eq 1)) (eq 2) \
			 (times 1)))"
		),
		(
			"3d6\nreroll\t=\n1\n2\ntimes",
			"3D6 reroll = 1 2 times",
			"(reroll (standard-dice 3 6) (eq 1) (times 2))"
		)
	]
	{
		let (residue, result) = dice(Span::new(input))
			.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
		assert!(residue.fragment().is_empty(), "residue for {input:?}");
		assert_eq!(result.to_string(), rendering, "{input:?}");
		let source = format!("(function [{{a}}] {s_expr})");
		let expected = crate::s_expr::read_s_expr(&source).unwrap();
		let Expression::Dice(expected) = &expected.body
		else
		{
			unreachable!()
		};
		assert!(
			result.untethered() == expected.untethered(),
			"AST mismatch for {input:?}"
		);
	}
	for input in [
		"reroll = 1",
		"3d6 reroll",
		"3d6 reroll 1",
		"3d6 reroll == 1",
		"3d6 reroll = -{x}",
		"3d6 reroll = 1 2",
		"3d6 reroll = 1 2 timesx",
		"3d6 reroll = 1 ∞",
		"3d6 reroll in",
		"3d6 reroll in 1 3",
		"3d6 reroll in 1..",
		"3d6 reroll in [{x}]",
		"3d6 reroll in [1, 2",
		"3 reroll = 1"
	]
	{
		let result = dice(Span::new(input));
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
	// A count never begins with `-`, so a `-` after a predicate is
	// subtraction, and a word that begins with `reroll` is no reroll.
	for (input, rendering, residue) in [
		("3d6 reroll = 1 -1", "3D6 reroll = 1", " -1"),
		("3d6 reroll = 1 + 1", "3D6 reroll = 1", " + 1"),
		("3d6 rerolled", "3D6", " rerolled")
	]
	{
		let (rest, result) = dice(Span::new(input)).unwrap();
		assert_eq!(result.to_string(), rendering, "{input:?}");
		assert_eq!(*rest.fragment(), residue, "{input:?}");
	}
}

/// Ensure that [`reroll`] reads a reroll clause alone, and rejects anything
/// else.
#[test]
fn test_reroll()
{
	for (input, predicate, limit) in [
		("reroll = 1", "(eq 1)", ""),
		("reroll≤-2 3 times", "(le -2)", " (times 3)"),
		("reroll in [] ∞ time", "(in [])", " (unlimited)"),
		("reroll in {x}..2", "(in-range {x} 2)", ""),
		(
			"reroll  in  1 ..= 2  {n}  time",
			"(in-range-inclusive 1 2)",
			" (times {n})"
		)
	]
	{
		let (residue, (actual_predicate, actual_limit)) =
			reroll(Span::new(input))
				.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
		assert!(residue.fragment().is_empty(), "residue for {input:?}");
		let source = format!(
			"(function [] (reroll (standard-dice 1 6) {predicate}{limit}))"
		);
		let expected = crate::s_expr::read_s_expr(&source).unwrap();
		let Expression::Dice(DiceExpression::Reroll(expected)) = &expected.body
		else
		{
			unreachable!()
		};
		assert_eq!(
			actual_predicate.map(|operand| operand.untethered()),
			expected.predicate.map(|operand| operand.untethered()),
			"{input:?}"
		);
		assert_eq!(
			actual_limit.map(|limit| limit.map(|count| count.untethered())),
			expected
				.limit
				.as_ref()
				.map(|limit| limit.map(|count| count.untethered())),
			"{input:?}"
		);
	}
	for input in [
		"",
		" reroll = 1",
		"rerolled = 1",
		"reroll",
		"reroll in",
		"drop lowest",
		"3d6 reroll = 1"
	]
	{
		let result = reroll(Span::new(input));
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`dice`] reads explode clauses: every comparator, in ASCII and
/// in Unicode, with literal, negative, named, bound, and computed thresholds;
/// lists, empty or not, of negative literals too; half-open and inclusive
/// ranges, with endpoints of every kind; limits of `time` or `times`, and of
/// `∞` or `unlimited`; with and without whitespace between their tokens; and
/// chained with drops, keeps, rerolls, and other explodes in any order.
#[test]
fn test_dice_explode_clauses()
{
	for (input, rendering, s_expr) in [
		(
			"3d6 explode = 6",
			"3D6 explode = 6",
			"(explode (standard-dice 3 6) (eq 6))"
		),
		(
			"3d6 explode=6",
			"3D6 explode = 6",
			"(explode (standard-dice 3 6) (eq 6))"
		),
		(
			"3d6 explode != 1 2 times",
			"3D6 explode ≠ 1 2 times",
			"(explode (standard-dice 3 6) (ne 1) (times 2))"
		),
		(
			"3d6 explode ≠ -1 1 time",
			"3D6 explode ≠ -1 1 time",
			"(explode (standard-dice 3 6) (ne -1) (times 1))"
		),
		(
			"3d6 explode < 2 1 times",
			"3D6 explode < 2 1 time",
			"(explode (standard-dice 3 6) (lt 2) (times 1))"
		),
		(
			"3d6 explode <= 1 ∞ times",
			"3D6 explode ≤ 1 ∞ times",
			"(explode (standard-dice 3 6) (le 1) (unlimited))"
		),
		(
			"3d6 explode ≤ {x} unlimited time",
			"3D6 explode ≤ {x} ∞ times",
			"(explode (standard-dice 3 6) (le {x}) (unlimited))"
		),
		(
			"3d6 explode > 5 {n} times",
			"3D6 explode > 5 {n} times",
			"(explode (standard-dice 3 6) (gt 5) (times {n}))"
		),
		(
			"3d6 explode >= (3+2) 2 time",
			"3D6 explode ≥ (3 + 2) 2 times",
			"(explode (standard-dice 3 6) (ge (group (add 3 2))) (times 2))"
		),
		(
			"3d6 explode ≥ {a}@(6)",
			"3D6 explode ≥ {a}@(6)",
			"(explode (standard-dice 3 6) (ge (binding {a} 6)))"
		),
		(
			"3d[-1,0,1] explode in [-1, 1]",
			"3D[-1, 0, 1] explode in [-1, 1]",
			"(explode (custom-dice 3 [-1 0 1]) (in [-1 1]))"
		),
		(
			"3d6 explode in []",
			"3D6 explode in []",
			"(explode (standard-dice 3 6) (in []))"
		),
		(
			"3d6 explode in[5,6] unlimited times",
			"3D6 explode in [5, 6] ∞ times",
			"(explode (standard-dice 3 6) (in [5 6]) (unlimited))"
		),
		(
			"3d[-3,-2,-1] explode in -3..-1",
			"3D[-3, -2, -1] explode in -3..-1",
			"(explode (custom-dice 3 [-3 -2 -1]) (in-range -3 -1))"
		),
		(
			"3d6 explode in 5..=6 2 times",
			"3D6 explode in 5..=6 2 times",
			"(explode (standard-dice 3 6) (in-range-inclusive 5 6) (times 2))"
		),
		(
			"3d6 explode in5..={x}",
			"3D6 explode in 5..={x}",
			"(explode (standard-dice 3 6) (in-range-inclusive 5 {x}))"
		),
		(
			"3d6 explode in ({x}-1)..=({x}+1) ({x}) times",
			"3D6 explode in ({x} - 1)..=({x} + 1) ({x}) times",
			"(explode (standard-dice 3 6) (in-range-inclusive (group (sub {x} \
			 1)) (group (add {x} 1))) (times (group {x})))"
		),
		(
			"4d6 drop lowest explode = 6 keep highest 2",
			"4D6 drop lowest explode = 6 keep highest 2",
			"(keep-highest (explode (drop-lowest (standard-dice 4 6)) (eq 6)) \
			 2)"
		),
		(
			"4d6 explode = 6 reroll = 1 explode = 5 1 time drop highest",
			"4D6 explode = 6 reroll = 1 explode = 5 1 time drop highest",
			"(drop-highest (explode (reroll (explode (standard-dice 4 6) (eq \
			 6)) (eq 1)) (eq 5) (times 1)))"
		),
		(
			"4d6 reroll = 1 explode = 6 ∞ times",
			"4D6 reroll = 1 explode = 6 ∞ times",
			"(explode (reroll (standard-dice 4 6) (eq 1)) (eq 6) (unlimited))"
		),
		(
			"3d6\nexplode\t=\n6\n2\ntimes",
			"3D6 explode = 6 2 times",
			"(explode (standard-dice 3 6) (eq 6) (times 2))"
		)
	]
	{
		let (residue, result) = dice(Span::new(input))
			.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
		assert!(residue.fragment().is_empty(), "residue for {input:?}");
		assert_eq!(result.to_string(), rendering, "{input:?}");
		let source = format!("(function [{{a}}] {s_expr})");
		let expected = crate::s_expr::read_s_expr(&source).unwrap();
		let Expression::Dice(expected) = &expected.body
		else
		{
			unreachable!()
		};
		assert!(
			result.untethered() == expected.untethered(),
			"AST mismatch for {input:?}"
		);
	}
	for input in [
		"explode = 6",
		"3d6 explode",
		"3d6 explode 6",
		"3d6 explode == 6",
		"3d6 explode = -{x}",
		"3d6 explode = 6 2",
		"3d6 explode = 6 2 timesx",
		"3d6 explode = 6 ∞",
		"3d6 explode in",
		"3d6 explode in 5 6",
		"3d6 explode in 5..",
		"3d6 explode in [{x}]",
		"3d6 explode in [5, 6",
		"3 explode = 6"
	]
	{
		let result = dice(Span::new(input));
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
	// A count never begins with `-`, so a `-` after a predicate is
	// subtraction, and a word that begins with `explode` is no explode.
	for (input, rendering, residue) in [
		("3d6 explode = 6 -1", "3D6 explode = 6", " -1"),
		("3d6 explode = 6 + 1", "3D6 explode = 6", " + 1"),
		("3d6 exploded", "3D6", " exploded"),
		("3d6 explodehighest", "3D6", " explodehighest")
	]
	{
		let (rest, result) = dice(Span::new(input)).unwrap();
		assert_eq!(result.to_string(), rendering, "{input:?}");
		assert_eq!(*rest.fragment(), residue, "{input:?}");
	}
}

/// Ensure that [`explode`] reads an explode clause alone, and rejects anything
/// else.
#[test]
fn test_explode()
{
	for (input, predicate, limit) in [
		("explode = 6", "(eq 6)", ""),
		("explode≥-2 3 times", "(ge -2)", " (times 3)"),
		("explode in [] ∞ time", "(in [])", " (unlimited)"),
		("explode in {x}..7", "(in-range {x} 7)", ""),
		(
			"explode  in  5 ..= 6  {n}  time",
			"(in-range-inclusive 5 6)",
			" (times {n})"
		)
	]
	{
		let (residue, (actual_predicate, actual_limit)) =
			explode(Span::new(input))
				.unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
		assert!(residue.fragment().is_empty(), "residue for {input:?}");
		let source = format!(
			"(function [] (explode (standard-dice 1 6) {predicate}{limit}))"
		);
		let expected = crate::s_expr::read_s_expr(&source).unwrap();
		let Expression::Dice(DiceExpression::Explode(expected)) =
			&expected.body
		else
		{
			unreachable!()
		};
		assert_eq!(
			actual_predicate.map(|operand| operand.untethered()),
			expected.predicate.map(|operand| operand.untethered()),
			"{input:?}"
		);
		assert_eq!(
			actual_limit.map(|limit| limit.map(|count| count.untethered())),
			expected
				.limit
				.as_ref()
				.map(|limit| limit.map(|count| count.untethered())),
			"{input:?}"
		);
	}
	for input in [
		"",
		" explode = 6",
		"exploded = 6",
		"explode",
		"explode in",
		"reroll = 1",
		"drop lowest",
		"3d6 explode = 6"
	]
	{
		let result = explode(Span::new(input));
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
	// Neither reads the other's clause.
	assert!(reroll(Span::new("explode = 6")).is_err());
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
			"{var}",
			"{var}",
			Expression::Variable(Variable {
				name: "var".into(),
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
	// A drop expression never begins with `-`, so a negative count must be
	// grouped.
	for input in [
		"", " ", "d6", "2d6", "[1:10]", "2+3", "{var}+2", "-42", "- 42"
	]
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

/// Ensure that a parse error renders, rather than panics, when a `-` after a
/// die's `d` begins no face count, and that it expects a digit where the
/// digits should begin.
#[test]
fn test_dangling_sign_messages()
{
	for (input, expected) in [
		(
			"4d-",
			"Parse error @ 1:4 (byte 3): expected digit, but found end of input"
		),
		(
			"1d-∞",
			"Parse error @ 1:4 (byte 3): expected digit, but found `∞`"
		),
		(
			"4d-x",
			"Parse error @ 1:4 (byte 3): expected digit, but found `x`"
		),
		(
			"4d-é",
			"Parse error @ 1:4 (byte 3): expected digit, but found `é`"
		),
		(
			"4d-?",
			"Parse error @ 1:4 (byte 3): expected digit, but found `?`"
		),
		(
			"4d--",
			"Parse error @ 1:4 (byte 3): expected digit, but found `-`"
		),
		(
			"4 d -∞",
			"Parse error @ 1:6 (byte 5): expected digit, but found `∞`"
		),
		(
			"-4 d -∞",
			"Parse error @ 1:7 (byte 6): expected digit, but found `∞`"
		),
		(
			"3/-4 d -∞",
			"Parse error @ 1:9 (byte 8): expected digit, but found `∞`"
		),
		(
			"3/-4d-∞",
			"Parse error @ 1:7 (byte 6): expected digit, but found `∞`"
		),
		// Without a sign, the digits expect nothing of their own.
		(
			"4d∞",
			"Parse error @ 1:3 (byte 2): expected `(`, `[`, `{`, or integer, but \
			 found `∞`"
		),
		(
			"d -∞",
			"Parse error @ 1:1 (byte 0): expected `(`, `-`, `[`, `highest(`, \
			 `lowest(`, `{`, dice expression, integer, or parameter definition, \
			 but found `d -`"
		)
	]
	{
		let error = Parser::parse(input).expect_err(input);
		let rendered = error.to_string();
		assert_eq!(
			rendered.lines().next(),
			Some(expected),
			"Failed for input: {}",
			input
		);
	}
}
