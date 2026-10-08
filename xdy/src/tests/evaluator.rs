//! # Evaluator tests
//!
//! Herein are the tests for the evaluator. The actual test cases are stored in
//! `../../tests/test_evaluation.txt`, which comprises a series of test cases,
//! each of which consists of a source dice expression and an expected print
//! rendition.

use std::{
	collections::{HashMap, HashSet},
	ops::RangeInclusive
};

use pretty_assertions::assert_eq;
use rand::{Rng as _, RngExt as _, SeedableRng, rngs::StdRng};

use crate::{
	AddressingMode, Assembler, Budget, Criterion, Die, EvaluationBounds,
	EvaluationError, Evaluator, Fate, Immediate, Origin, Passes, Predicate,
	RegisterIndex, RollingRecord, RollingRecordKind, Unobserved,
	support::{
		compile_valid, on_small_stack, optimize, read_evaluation_test_cases
	}
};

////////////////////////////////////////////////////////////////////////////////
//                             Evaluation tests.                              //
////////////////////////////////////////////////////////////////////////////////

/// Test that the evaluator produces the expected output for the test cases.
#[test]
fn test_evaluation()
{
	let mut seen = HashSet::new();
	for (index, (source, args, externs, expected)) in
		read_evaluation_test_cases(include_str!(
			"../../tests/test_evaluation.txt"
		))
		.iter()
		.enumerate()
	{
		let key = (source, args.clone(), externs.clone());
		let key = format!("{:?}", key);
		assert!(seen.insert(key.clone()), "duplicate test case: {}", key);
		let function = compile_valid(source);
		let function = optimize(function, Passes::all());
		let mut evaluator = Evaluator::new(function);
		for (name, value) in externs.iter()
		{
			evaluator.bind(name, *value).unwrap();
		}
		// Ensure that the evaluator produces the expected bounds. Every binding
		// is supplied, and supplied exactly, so the bounds are as tight as the
		// interval arithmetic can make them.
		let bounds = evaluator
			.bounds_over(
				args.iter().map(|arg| Some((*arg).into())),
				externs.iter().map(|(name, value)| (*name, (*value).into()))
			)
			.unwrap();
		assert_eq!(
			bounds.to_string(),
			*expected,
			"case {}: {}",
			index + 1,
			key
		);
		// The seed is arbitrary, chosen by smashing the keyboard. This is to
		// ensure that the test cases are deterministic.
		let mut rng = StdRng::seed_from_u64(24987829587102357);
		// Ensure that the evaluator only produces results within the expected
		// bounds.
		for _ in 0..1000
		{
			let result =
				evaluator.evaluate(args.iter().copied(), &mut rng).unwrap();
			// Ensure that the evaluator rolls no more dice than the worst case,
			// and reports exactly the dice that it rolled.
			assert!(
				result.dice <= bounds.dice,
				"case {}: {}: too many dice: {} > {}",
				index + 1,
				key,
				result.dice,
				bounds.dice
			);
			assert_eq!(
				result.dice,
				result
					.records
					.iter()
					.map(|record| record.dice.len() as u64)
					.sum::<u64>(),
				"case {}: {}: misreported dice",
				index + 1,
				key
			);
			let bounds: RangeInclusive<i32> = bounds.value.into();
			assert!(
				bounds.contains(&result.result),
				"case {}: {}: result out of bounds: {} ∉ {}..={}: rolls: {}",
				index + 1,
				key,
				result.result,
				bounds.start(),
				bounds.end(),
				result
					.records
					.iter()
					.map(|record| record.dice[0].value.to_string())
					.collect::<Vec<_>>()
					.join(", ")
			);
			// Ensure that none of the die rolls are outside the expected
			// bounds. A reroll appends each replacement to the dice that the
			// roll rolled, so count only the latter.
			for record in result.records
			{
				match record.kind
				{
					RollingRecordKind::Uninitialized => unreachable!(),
					RollingRecordKind::Range { start, end } =>
					{
						assert_eq!(
							record
								.dice
								.iter()
								.filter(|die| die.origin == Origin::Rolled)
								.count(),
							1,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						if end < start
						{
							assert_eq!(
								record.dice[0].value,
								0,
								"case {}: {}: roll out of bounds: {} ∉ {}..={}",
								index + 1,
								key,
								record.dice[0].value,
								start,
								end
							);
						}
						else
						{
							assert!(
								(start..=end).contains(&record.dice[0].value),
								"case {}: {}: roll out of bounds: {} ∉ {}..={}",
								index + 1,
								key,
								record.dice[0].value,
								start,
								end
							);
						}
					},
					RollingRecordKind::Standard { count, faces } =>
					{
						assert_eq!(
							record
								.dice
								.iter()
								.filter(|die| die.origin == Origin::Rolled)
								.count(),
							count.max(0) as usize,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						for result in record.dice.iter().map(|die| die.value)
						{
							match faces <= 0
							{
								false => assert!(
									(1..=faces).contains(&result),
									"case {}: {}: roll out of bounds: {} ∉ 1..={}",
									index + 1,
									key,
									result,
									faces
								),
								true => assert_eq!(
									result,
									0,
									"case {}: {}: roll out of bounds: {} ≠ 0",
									index + 1,
									key,
									result
								)
							}
						}
					},
					RollingRecordKind::Custom { count, faces } =>
					{
						assert_eq!(
							record
								.dice
								.iter()
								.filter(|die| die.origin == Origin::Rolled)
								.count(),
							count as usize,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						for result in record.dice.iter().map(|die| die.value)
						{
							assert!(
								faces.contains(&result),
								"case {}: {}: roll out of bounds: {} ∉ {:?}",
								index + 1,
								key,
								result,
								faces
							);
						}
					}
				}
			}
		}
	}
}

/// Test that the evaluator produces the expected error when a function is
/// applied to the wrong number of arguments.
#[test]
fn test_bad_arity()
{
	let function = compile_valid("{x}: {x}");
	let mut evaluator = Evaluator::new(function);
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(409568093489576902);
	assert_eq!(
		evaluator.evaluate([], &mut rng),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
	assert_eq!(
		evaluator.evaluate([1, 2].iter().copied(), &mut rng),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 2
		})
	);
	assert_eq!(
		evaluator.bounds_over([], []),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
	assert_eq!(
		evaluator.bounds_over([Some(1.into()), Some(2.into())], []),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 2
		})
	);
}

/// Test that the evaluator produces the expected error when an unrecognized
/// external variable is bound to a function.
#[test]
fn test_unrecognized_external()
{
	let function = compile_valid("{x}: {x}");
	let mut evaluator = Evaluator::new(function);
	assert_eq!(
		evaluator.bind("y", 1),
		Err(EvaluationError::UnrecognizedExternal("y"))
	);
}

/// Test that the evaluator binds an external variable by its canonical name,
/// whose whitespace collapses to single spaces, however the source spells it.
#[test]
fn test_bind_canonical_external()
{
	let function = compile_valid("{a\n   b} + 1");
	assert_eq!(function.externals, vec!["a b".to_string()]);
	let mut evaluator = Evaluator::new(function);
	assert_eq!(evaluator.bind("a b", 1), Ok(()));
	assert_eq!(
		evaluator.bind("a\n   b", 1),
		Err(EvaluationError::UnrecognizedExternal("a\n   b"))
	);
}

/// Test that rolling records answer the correct counts.
#[test]
fn test_rolling_record_count()
{
	assert_eq!(RollingRecordKind::<i32>::Uninitialized.count(), None);
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(69873748728957892);
	for (src, expected) in [("[3:8]", 1), ("3D6", 3), ("8D[-1, -1, -2, 5]", 8)]
	{
		let function = compile_valid(src);
		let mut evaluator = Evaluator::new(function);
		let result = evaluator.evaluate([], &mut rng).unwrap();
		assert_eq!(result.records.len(), 1);
		assert_eq!(result.records[0].kind.count(), Some(expected));
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Dice metering.                               //
////////////////////////////////////////////////////////////////////////////////

/// Test that a roll exceeding the dice budget is refused before it rolls
/// anything, so that even the greatest count is refused promptly, without
/// drawing from the pRNG or allocating the dice.
#[test]
fn test_dice_budget_refuses_before_rolling()
{
	on_small_stack(|| {
		for (source, arg) in [
			("{x}: {x}D6", i32::MAX),
			("{x}: {x}D[1, 2, 3]", i32::MAX),
			// The count is computed, and saturates to the greatest count.
			("{x}: ({x} * {x})D6", 46_341)
		]
		{
			let mut evaluator = Evaluator::new(compile_valid(source));
			// The seed is arbitrary, chosen by smashing the keyboard. This is
			// to ensure that the test cases are deterministic.
			let seed = 5829175027591875;
			let mut rng = StdRng::seed_from_u64(seed);
			let mut untouched = StdRng::seed_from_u64(seed);
			assert_eq!(
				evaluator.evaluate_metered([arg], &mut rng, 100),
				Err(EvaluationError::DiceBudgetExhausted {
					requested: i32::MAX as u64,
					remaining: 100,
					consumed: 0
				}),
				"{}",
				source
			);
			assert_eq!(rng.next_u64(), untouched.next_u64(), "{}", source);
		}
	});
}

/// Test that a roll exactly at the dice budget succeeds and reports exactly
/// that consumption, and that one die more is refused.
#[test]
fn test_dice_budget_exact()
{
	// The outer roll depends on the inner one, so the order of the rolls, and
	// therefore of the charges, is fixed.
	let mut evaluator = Evaluator::new(compile_valid("{x}: ({x}D1)D6"));
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(2098357109857129);
	let evaluation = evaluator.evaluate_metered([7], &mut rng, 14).unwrap();
	assert_eq!(evaluation.dice, 14);
	assert_eq!(
		evaluator.evaluate_metered([7], &mut rng, 13),
		Err(EvaluationError::DiceBudgetExhausted {
			requested: 7,
			remaining: 6,
			consumed: 7
		})
	);
	assert_eq!(
		evaluator.evaluate_metered([8], &mut rng, 14),
		Err(EvaluationError::DiceBudgetExhausted {
			requested: 8,
			remaining: 6,
			consumed: 8
		})
	);
}

/// Test that a nonpositive count of dice costs nothing, and that a range costs
/// one die, even when it is empty.
#[test]
fn test_dice_budget_costs()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(7120985710298375);
	let mut evaluator = Evaluator::new(compile_valid("{x}: {x}D6"));
	for count in [0, -1, i32::MIN]
	{
		let evaluation =
			evaluator.evaluate_metered([count], &mut rng, 0).unwrap();
		assert_eq!(evaluation.dice, 0, "{}", count);
	}
	for source in ["{x}: [1:{x}]", "{x}: [{x}:1]"]
	{
		let mut evaluator = Evaluator::new(compile_valid(source));
		assert_eq!(
			evaluator.evaluate_metered([6], &mut rng, 0),
			Err(EvaluationError::DiceBudgetExhausted {
				requested: 1,
				remaining: 0,
				consumed: 0
			}),
			"{}",
			source
		);
		let evaluation = evaluator.evaluate_metered([6], &mut rng, 1).unwrap();
		assert_eq!(evaluation.dice, 1, "{}", source);
	}
}

/// Test that metering within a sufficient budget does not disturb the draws
/// from the pRNG, so that metered and unmetered evaluation agree.
#[test]
fn test_dice_budget_agrees_with_unmetered()
{
	let mut evaluator = Evaluator::new(compile_valid(
		"4D6 drop lowest + [1:20] + 3D[-1, 0, 1] + (1D4)D8"
	));
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let seed = 9812750918273509;
	let bounds = evaluator.bounds_over([], []).unwrap();
	assert_eq!(bounds.dice, 13);
	for i in 0..100
	{
		let unmetered = evaluator
			.evaluate([], &mut StdRng::seed_from_u64(seed + i))
			.unwrap();
		let metered = evaluator
			.evaluate_metered([], &mut StdRng::seed_from_u64(seed + i), 13)
			.unwrap();
		assert_eq!(metered, unmetered);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Maximum.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that a [maximum](crate::Max), which only the optimizer emits, evaluates
/// to the greater of its operands, and that its bounds are the greater of the
/// operands' bounds.
#[test]
fn test_max()
{
	let mut evaluator = Evaluator::new(
		crate::Assembler::assemble(
			"\
Function({x}@0) r#2 ⚅#0
\textern[]
\tbody:
\t\t@1 <- @0 max 0
\t\treturn @1
"
		)
		.unwrap()
	);
	for (x, expected) in
		[(i32::MIN, 0), (-3, 0), (0, 0), (5, 5), (i32::MAX, i32::MAX)]
	{
		// The seed is arbitrary, since the function rolls nothing.
		let evaluation = evaluator
			.evaluate([x], &mut StdRng::seed_from_u64(0))
			.unwrap();
		assert_eq!(evaluation.result, expected, "{}", x);
		assert_eq!(evaluation.dice, 0, "{}", x);
	}
	for ((min, max), expected) in
		[((-3, 5), (0, 5)), ((-7, -2), (0, 0)), ((2, 9), (2, 9))]
	{
		let bounds = evaluator
			.bounds_over([Some((min, max).into())], [])
			.unwrap();
		assert_eq!(
			(bounds.value.min, bounds.value.max),
			expected,
			"[{}, {}]",
			min,
			max
		);
		assert_eq!(bounds.dice, 0);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Minimum.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that a [minimum](crate::Min) evaluates to the lesser of its operands,
/// and that its bounds are the lesser of the operands' bounds.
#[test]
fn test_min()
{
	let mut evaluator = Evaluator::new(
		crate::Assembler::assemble(
			"\
Function({x}@0) r#2 ⚅#0
\textern[]
\tbody:
\t\t@1 <- @0 min 0
\t\treturn @1
"
		)
		.unwrap()
	);
	for (x, expected) in [
		(i32::MIN, i32::MIN),
		(-3, -3),
		(0, 0),
		(5, 0),
		(i32::MAX, 0)
	]
	{
		// The seed is arbitrary, since the function rolls nothing.
		let evaluation = evaluator
			.evaluate([x], &mut StdRng::seed_from_u64(0))
			.unwrap();
		assert_eq!(evaluation.result, expected, "{}", x);
		assert_eq!(evaluation.dice, 0, "{}", x);
	}
	for ((min, max), expected) in
		[((-3, 5), (-3, 0)), ((-7, -2), (-7, -2)), ((2, 9), (0, 0))]
	{
		let bounds = evaluator
			.bounds_over([Some((min, max).into())], [])
			.unwrap();
		assert_eq!(
			(bounds.value.min, bounds.value.max),
			expected,
			"[{}, {}]",
			min,
			max
		);
		assert_eq!(bounds.dice, 0);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Unsummed rolls.                               //
////////////////////////////////////////////////////////////////////////////////

/// Test that a rolling record that is never summed, which neither the compiler
/// nor the optimizer emits, still counts in the bounds: each of its
/// outcomes is a distinct path, so it multiplies the count of outcomes, and its
/// dice count toward the worst case, although it cannot affect the value. The
/// distribution agrees, with a total of the count. Before, the optimizer left
/// such a record behind wherever a drop kept its roll alive, and the evaluation
/// corpus covered it thereby.
#[test]
fn test_unsummed_roll_counts()
{
	for (text, value, count, dice) in [
		(
			"\
Function({x}@0) r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll standard dice 1D@0
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\treturn 0
",
			(0, 0),
			2,
			1
		),
		(
			"\
Function({x}@0) r#2 ⚅#2
\textern[]
\tbody:
\t\t⚅0 <- roll standard dice 1D6
\t\t@1 <- sum rolling record ⚅0
\t\t⚅1 <- roll standard dice 2D@0
\t\t⚅1 <- drop highest 1 from ⚅1
\t\treturn @1
",
			(1, 6),
			24,
			3
		)
	]
	{
		let evaluator =
			Evaluator::new(crate::Assembler::assemble(text).unwrap());
		let bounds = evaluator.bounds_over([Some(2.into())], []).unwrap();
		assert_eq!((bounds.value.min, bounds.value.max), value, "{}", text);
		assert_eq!(bounds.count, Some(count), "{}", text);
		assert_eq!(bounds.dice, dice, "{}", text);
		let distribution = evaluator
			.plan_distribution([2])
			.unwrap()
			.build(Budget::UNLIMITED, &Unobserved)
			.unwrap();
		assert_eq!(distribution.total(), &count.into(), "{}", text);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Drop counts.                                //
////////////////////////////////////////////////////////////////////////////////

/// Test that a negative drop count drops nothing, and so restores nothing that
/// an earlier drop clause dropped: the evaluator, the worst case of the dice,
/// and the meter agree. Before 0.13.0, the evaluator clamped the drop
/// count after every clause, so a negative count restored a die that the
/// bounds took to stay dropped, and the outer roll rolled a die more than the
/// worst case.
#[test]
fn test_negative_drop_count_drops_nothing()
{
	for source in [
		"(1D1 drop lowest 2 drop lowest (-1))D6",
		"(1D1 drop highest 2 drop highest (-1))D6",
		"(1D1 drop lowest (-1) drop lowest 2)D6"
	]
	{
		let mut evaluator = Evaluator::new(compile_valid(source));
		let bounds = evaluator.bounds_over([], []).unwrap();
		assert_eq!(bounds.dice, 1, "{}", source);
		// The seed is arbitrary, chosen by smashing the keyboard. This is to
		// ensure that the test cases are deterministic.
		let mut rng = StdRng::seed_from_u64(6120957120985710);
		let evaluation = evaluator.evaluate_metered([], &mut rng, 1).unwrap();
		assert_eq!(evaluation.dice, 1, "{}", source);
		assert_eq!(evaluation.result, 0, "{}", source);
	}
}

/// Test that the order of drop clauses does not matter, even when a count is
/// negative, so that the optimizer, which reorders the drop counts of a
/// record, agrees with the unoptimized function, and the clauses agree in
/// either order.
#[test]
fn test_drop_order_is_irrelevant()
{
	for (source, reversed) in [
		(
			"{x}: 3D6 drop lowest {x} drop lowest (-1)",
			"{x}: 3D6 drop lowest (-1) drop lowest {x}"
		),
		(
			"{x}: 3D6 drop highest {x} drop highest (-2)",
			"{x}: 3D6 drop highest (-2) drop highest {x}"
		)
	]
	{
		let functions = [source, reversed].map(|source| {
			let function = crate::compile_unoptimized(source).unwrap();
			[function.clone(), optimize(function, Passes::all())]
		});
		for (i, x) in [-1, 0, 1, 2, 3, 4].into_iter().enumerate()
		{
			// The seed is arbitrary, chosen by smashing the keyboard. This is
			// to ensure that the test cases are deterministic.
			let seed = 1098275019827350 + i as u64;
			let results = functions
				.iter()
				.flatten()
				.map(|function| {
					Evaluator::new(function.clone())
						.evaluate([x], &mut StdRng::seed_from_u64(seed))
						.unwrap()
						.result
				})
				.collect::<Vec<_>>();
			assert!(
				results.iter().all(|&result| result == results[0]),
				"{} with {}: {:?}",
				source,
				x,
				results
			);
		}
	}
}

/// Test that the optimizer respects that a negative count of dice rolls
/// nothing and a drop count of zero or less drops nothing, so that optimized
/// and unoptimized functions have the same distribution. The
/// test compares exact distributions rather than evaluations from the same
/// seed, since the optimizer rolls no dice of one face, which an unoptimized
/// function rolls, drawing from the pRNG. Before, strength reduction
/// replaced `{x}D1` with `{x}` and a single-face custom roll with a product,
/// even for a negative count; rewrote a drop from such a value as a
/// subtraction, which went negative or subtracted the drop count rather than
/// the dropped faces; and merged stacked drops into one drop of their sum.
/// Constant folding summed negative drop counts too, and folded a
/// single-valued range that strength reduction had given drops.
#[test]
fn test_optimizer_respects_clamping()
{
	for source in [
		"{x}: {x}D1",
		"{x}: {x}D[5]",
		"{x}: {x}D1 drop lowest 1",
		"{x}: {x}D1 drop lowest 5",
		"{x}: {x}D[5] drop highest 1",
		"{x}: 3D1 drop lowest {x}",
		"{x}: 3D[5] drop lowest {x}",
		"{x}: 1D1 drop lowest {x}",
		"{x}: 1D6 drop highest {x}",
		"{x}: 1D[2, 3, 4] drop lowest {x}",
		"{x}: 3D1 drop lowest (-1) + {x}",
		"{x}: 3D6 drop lowest 2 drop lowest (-1) + {x}",
		"{x}: 3D6 drop lowest {x} drop lowest (-1)",
		"{x}: 3D6 drop lowest {x} drop lowest 1 drop lowest 1",
		"{x}: 3D6 drop highest (-2) drop highest {x} drop lowest 1",
		"{x}: ({x}D1 drop lowest 2 drop lowest (-1))D6"
	]
	{
		let function = crate::compile_unoptimized(source).unwrap();
		let optimized = optimize(function.clone(), Passes::all());
		for x in [i32::MIN, -3, -1, 0, 1, 2, 3, 5]
		{
			let [unoptimized, optimized] =
				[&function, &optimized].map(|function| {
					Evaluator::new(function.clone())
						.plan_distribution([x])
						.unwrap()
						.build(Budget::UNLIMITED, &Unobserved)
						.unwrap()
				});
			assert_eq!(optimized, unoptimized, "{} with {}", source, x);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Crossed-out records.                            //
////////////////////////////////////////////////////////////////////////////////

/// Answer a rolling record of standard dice with the specified results, all of
/// them kept.
///
/// # Parameters
/// - `faces`: The number of faces on each die.
/// - `results`: The results, in the order rolled.
///
/// # Returns
/// The rolling record.
fn record(faces: i32, results: &[i32]) -> RollingRecord
{
	RollingRecord {
		kind: RollingRecordKind::Standard {
			count: results.len() as i32,
			faces
		},
		dice: results.iter().copied().map(Die::kept).collect()
	}
}

/// Test that drops and keeps cross out the lowest or highest dice still kept,
/// and of equal dice, the first rolled, and that a record prints its dice in
/// the order rolled, each crossed-out die prefixed with `~`.
#[test]
fn test_cross_out_first_rolled_first()
{
	type Clause = fn(&mut RollingRecord, i32);
	let drop_lowest: Clause = RollingRecord::drop_lowest;
	let drop_highest: Clause = RollingRecord::drop_highest;
	let keep_lowest: Clause = RollingRecord::keep_lowest;
	let keep_highest: Clause = RollingRecord::keep_highest;
	for (results, clauses, expected) in [
		(
			&[3, 1, 4, 1][..],
			&[(drop_lowest, 1)][..],
			"4D6: [3, ~1, 4, 1]"
		),
		(&[3, 1, 4, 1], &[(drop_lowest, 2)], "4D6: [3, ~1, 4, ~1]"),
		(&[2, 6, 1, 6], &[(drop_highest, 1)], "4D6: [2, ~6, 1, 6]"),
		(&[3, 3], &[(keep_highest, 1)], "2D6: [~3, 3]"),
		(&[3, 3], &[(keep_lowest, 1)], "2D6: [~3, 3]"),
		(
			&[5, 5, 5],
			&[(drop_lowest, 1), (drop_highest, 1)],
			"3D6: [~5, ~5, 5]"
		),
		(
			&[5, 2, 6, 3],
			&[(drop_lowest, 1), (keep_lowest, 2)],
			"4D6: [5, ~2, ~6, 3]"
		),
		(
			&[5, 2, 6, 3],
			&[(keep_lowest, 2), (drop_lowest, 1)],
			"4D6: [~5, ~2, ~6, 3]"
		),
		(
			&[4, 1, 6],
			&[(drop_lowest, -1), (drop_highest, 0)],
			"3D6: [4, 1, 6]"
		),
		(&[4, 1, 6], &[(drop_lowest, 7)], "3D6: [~4, ~1, ~6]"),
		(&[4, 1, 6], &[(keep_highest, 0)], "3D6: [~4, ~1, ~6]"),
		(&[4, 1, 6], &[(keep_highest, 7)], "3D6: [4, 1, 6]"),
		(&[], &[(drop_lowest, 1), (keep_highest, 1)], "0D6: []")
	]
	{
		let mut record = record(6, results);
		for (clause, count) in clauses
		{
			clause(&mut record, *count);
		}
		assert_eq!(record.to_string(), expected, "{:?}", results);
		assert_eq!(record.dice.len(), results.len());
		assert_eq!(
			record.dice.iter().map(|die| die.value).collect::<Vec<_>>(),
			results,
			"the dice moved"
		);
	}
}

/// Test that crossing out dice keeps the same results as counting the drops
/// from the ends of the sorted results did through 0.14, for random results
/// with many ties and random clauses with every kind of count, so that every
/// sum is unchanged. The forward pass and the Lean specification still count
/// drops, so they rely on this agreement.
#[test]
fn test_cross_out_agrees_with_drop_counts()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(5091283750192837);
	for _ in 0..10_000
	{
		let len = rng.random_range(0..=6);
		let results = (0..len)
			.map(|_| rng.random_range(-2..=3))
			.collect::<Vec<_>>();
		let mut record = record(6, &results);
		// The model of 0.14: sorted results and the counts of the lowest and
		// highest results dropped, each clamped to the number of results.
		let mut sorted = results.clone();
		sorted.sort();
		let (mut lowest, mut highest) = (0i32, 0i32);
		let drop = |total: i32, count: i32| {
			total.saturating_add(count.max(0)).clamp(0, len)
		};
		for _ in 0..rng.random_range(0..=4)
		{
			let count = rng.random_range(-2..=len + 2);
			let kept = (len - lowest - highest).max(0);
			let keep = kept - count.clamp(0, kept);
			match rng.random_range(0..4)
			{
				0 =>
				{
					record.drop_lowest(count);
					lowest = drop(lowest, count);
				},
				1 =>
				{
					record.drop_highest(count);
					highest = drop(highest, count);
				},
				2 =>
				{
					record.keep_lowest(count);
					highest = drop(highest, keep);
				},
				_ =>
				{
					record.keep_highest(count);
					lowest = drop(lowest, keep);
				}
			}
		}
		let kept = (len - lowest - highest).max(0) as usize;
		let expected = sorted
			.iter()
			.copied()
			.skip(lowest as usize)
			.take(kept)
			.collect::<Vec<_>>();
		let mut actual = record.kept();
		actual.sort();
		assert_eq!(actual, expected, "{}", record);
		let mut dropped = record.dropped();
		dropped.extend(record.kept());
		dropped.sort();
		assert_eq!(dropped, sorted, "{}", record);
		assert_eq!(
			record.sum(),
			expected.iter().fold(0i32, |a, b| a.saturating_add(*b)),
			"{}",
			record
		);
	}
}

/// Test that a record sums its kept results in ascending order, not in the
/// order rolled, since saturating addition is not associative.
#[test]
fn test_sum_folds_in_ascending_order()
{
	let mut record = RollingRecord {
		kind: RollingRecordKind::Custom {
			count: 3,
			faces: vec![i32::MAX, 1, -1]
		},
		dice: [i32::MAX, 1, -1].into_iter().map(Die::kept).collect()
	};
	assert_eq!(record.sum(), i32::MAX);
	record.drop_highest(1);
	assert_eq!(record.sum(), 0);
	assert_eq!(
		record.to_string(),
		"3d[2147483647, 1, -1]: [~2147483647, 1, -1]"
	);
}

/// Test that evaluation leaves the dice of each record in the order rolled,
/// crossing out the dice that its clauses drop, so that the record shows the
/// whole roll: the dice that a drop crosses out are the lowest, and the first
/// rolled of any equal to them, and the kept dice sum to the result.
#[test]
fn test_evaluation_crosses_out_dice()
{
	let mut evaluator = Evaluator::new(compile_valid("8D3 drop lowest 3"));
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(7120598172305981);
	for _ in 0..100
	{
		let evaluation = evaluator.evaluate([], &mut rng).unwrap();
		let [record] = &evaluation.records[..]
		else
		{
			panic!("expected one record: {:?}", evaluation.records)
		};
		assert_eq!(record.dice.len(), 8);
		assert_eq!(record.dropped().len(), 3);
		assert_eq!(record.kept().iter().sum::<i32>(), evaluation.result);
		// Crossing out the lowest dice, first rolled first, crosses out exactly
		// the first 3 dice in order of value and then of position.
		let mut order = (0..8).collect::<Vec<_>>();
		order.sort_by_key(|&i| (record.dice[i].value, i));
		for (rank, &i) in order.iter().enumerate()
		{
			assert_eq!(
				record.dice[i].is_kept(),
				rank >= 3,
				"die {} of {}",
				i,
				record
			);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Chains.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Answer a kept die with the specified origin.
///
/// # Parameters
/// - `value`: The result of the die.
/// - `origin`: The origin of the die.
///
/// # Returns
/// The die.
fn die(value: i32, origin: Origin) -> Die
{
	Die {
		value,
		fate: Fate::Kept,
		origin
	}
}

/// Test that every die joins the chain of the die from which it descends,
/// whatever links it, and whatever its fate, that the chains follow the order
/// in which their first dice were rolled, and that a link that names no
/// earlier die begins a chain of its own.
#[test]
fn test_chains_follow_links()
{
	let mut record = record(6, &[]);
	record.dice = vec![
		die(6, Origin::Rolled),
		die(3, Origin::Rolled),
		die(6, Origin::Extends(0)),
		die(1, Origin::Replaces(2)),
		die(4, Origin::Extends(2)),
		die(i32::MAX, Origin::Saturates(1)),
		die(2, Origin::Replaces(6)),
		die(5, Origin::Extends(7)),
		die(5, Origin::Rolled),
	];
	record.dice[2].fate = Fate::Rerolled;
	assert_eq!(
		record.chains(),
		vec![vec![0, 2, 3, 4], vec![1, 5], vec![6], vec![7], vec![8]]
	);
	assert_eq!(record.to_string(), "0D6: [6, 3, ~6, 1, 4, ∞, 2, 5, 5]");
	assert_eq!(
		record_of(RollingRecordKind::Uninitialized, &[]).chains(),
		Vec::<Vec<usize>>::new()
	);
}

/// Test that the sum and every drop and keep count each chain still kept as
/// one die, worth the sum of its kept dice, or the value of its saturated die;
/// that crossing out a chain crosses out every kept die of it, and leaves the
/// dice that a reroll crossed out as they were; and that of equal chains, the
/// one whose earliest kept die was rolled first is crossed out first.
#[test]
fn test_chains_count_as_one_die()
{
	// [6, 2] → 8, [3] → 3, [5, ~1, 3] → 8, [4] → 4.
	let chains = || {
		let mut record = record(6, &[]);
		record.dice = vec![
			die(6, Origin::Rolled),
			die(3, Origin::Rolled),
			die(5, Origin::Rolled),
			die(4, Origin::Rolled),
			die(2, Origin::Extends(0)),
			die(1, Origin::Extends(2)),
			die(3, Origin::Replaces(5)),
		];
		record.dice[5].fate = Fate::Rerolled;
		record
	};
	let mut whole = chains();
	assert_eq!(whole.sum(), 23);
	assert_eq!(whole.kept(), vec![6, 3, 5, 4, 2, 3]);
	type Clause = fn(&mut RollingRecord, i32);
	let drop_lowest: Clause = RollingRecord::drop_lowest;
	let drop_highest: Clause = RollingRecord::drop_highest;
	let keep_lowest: Clause = RollingRecord::keep_lowest;
	let keep_highest: Clause = RollingRecord::keep_highest;
	for (clause, count, expected, sum) in [
		(drop_lowest, 1, "0D6: [6, ~3, 5, 4, 2, ~1, 3]", 20),
		(drop_lowest, 2, "0D6: [6, ~3, 5, ~4, 2, ~1, 3]", 16),
		// The chains worth 8 tie, and the first, whose earliest kept die is
		// the `6`, goes first.
		(drop_highest, 1, "0D6: [~6, 3, 5, 4, ~2, ~1, 3]", 15),
		(drop_highest, 2, "0D6: [~6, 3, ~5, 4, ~2, ~1, ~3]", 7),
		// Of the chains worth 8, keeping the highest crosses out the first.
		(keep_highest, 1, "0D6: [~6, ~3, 5, ~4, ~2, ~1, 3]", 8),
		(keep_lowest, 3, "0D6: [~6, 3, 5, 4, ~2, ~1, 3]", 15),
		(keep_lowest, 4, "0D6: [6, 3, 5, 4, 2, ~1, 3]", 23),
		(drop_lowest, 9, "0D6: [~6, ~3, ~5, ~4, ~2, ~1, ~3]", 0),
		(keep_highest, 0, "0D6: [~6, ~3, ~5, ~4, ~2, ~1, ~3]", 0)
	]
	{
		let mut record = chains();
		clause(&mut record, count);
		assert_eq!(record.to_string(), expected);
		assert_eq!(record.sum(), sum, "{}", expected);
		assert_eq!(record.rerolled(), vec![1], "{}", expected);
	}
	// Of equal chains, the earliest kept die decides, not the first die: the
	// replacement of the `1` that begins the first chain was rolled after the
	// `3`, so the `3` is crossed out first.
	let mut record = record(6, &[]);
	record.dice = vec![
		die(1, Origin::Rolled),
		die(3, Origin::Rolled),
		die(3, Origin::Replaces(0)),
	];
	record.dice[0].fate = Fate::Rerolled;
	record.drop_lowest(1);
	assert_eq!(record.to_string(), "0D6: [~1, ~3, 3]");
}

/// Test that a saturated die decides the total of its chain, whatever the
/// other kept dice of the chain show, that drops and keeps count a saturated
/// chain as one die worth that total, and that a saturated die prints as `∞`
/// or `-∞`.
#[test]
fn test_saturated_chains()
{
	let saturated = |value| {
		let mut record = record_of(
			RollingRecordKind::Custom {
				count: 3,
				faces: vec![-10, 5]
			},
			&[]
		);
		record.dice = vec![
			die(5, Origin::Rolled),
			die(-10, Origin::Rolled),
			die(5, Origin::Rolled),
			die(value, Origin::Saturates(0)),
			die(value, Origin::Saturates(1)),
			die(value, Origin::Saturates(2)),
		];
		record
	};
	let mut record = saturated(i32::MIN);
	assert_eq!(record.sum(), i32::MIN);
	assert_eq!(record.to_string(), "3d[-10, 5]: [5, -10, 5, -∞, -∞, -∞]");
	record.keep_highest(1);
	assert_eq!(record.sum(), i32::MIN);
	assert_eq!(
		record.to_string(),
		"3d[-10, 5]: [~5, ~-10, 5, ~-∞, ~-∞, -∞]"
	);
	let mut record = saturated(i32::MAX);
	assert_eq!(record.sum(), i32::MAX);
	assert_eq!(record.to_string(), "3d[-10, 5]: [5, -10, 5, ∞, ∞, ∞]");
	record.drop_lowest(2);
	assert_eq!(record.sum(), i32::MAX);
	record.drop_lowest(1);
	assert_eq!(record.sum(), 0);
	let mut record = saturated(0);
	assert_eq!(record.sum(), 0);
	assert_eq!(record.to_string(), "3d[-10, 5]: [5, -10, 5, 0, 0, 0]");
	// Only a saturated die prints as infinite.
	let mut record = record_of(
		RollingRecordKind::Custom {
			count: 1,
			faces: vec![i32::MAX]
		},
		&[i32::MAX]
	);
	assert_eq!(record.to_string(), "1d[2147483647]: [2147483647]");
	assert_eq!(record.sum(), i32::MAX);
	// The sum folds the totals of the chains in ascending order, and each
	// chain folds its kept dice in ascending order.
	let mut record = record_of(
		RollingRecordKind::Custom {
			count: 2,
			faces: vec![i32::MAX, 1, -1]
		},
		&[]
	);
	record.dice = vec![
		die(1, Origin::Rolled),
		die(-1, Origin::Rolled),
		die(i32::MAX, Origin::Extends(0)),
		die(-1, Origin::Extends(2)),
	];
	// The first chain folds `-1`, `1`, and `i32::MAX`, to `i32::MAX`, and then
	// the sum folds `-1` and `i32::MAX`, to `i32::MAX - 1`.
	assert_eq!(record.sum(), i32::MAX - 1);
}

/// Test that, without dice that extend others, drops and keeps cross out
/// exactly the dice that crossing out dice one by one would, as through the
/// first release of crossed-out records, for random records with rerolls,
/// drops and keeps in any order, so that every record and sum is unchanged.
#[test]
fn test_chains_without_extensions_are_dice()
{
	/// Cross out dice one by one, as records did before chains.
	fn cross_out(record: &mut RollingRecord, count: i32, highest: bool)
	{
		let mut kept = record
			.dice
			.iter()
			.enumerate()
			.filter(|(_, die)| die.is_kept())
			.map(|(index, die)| {
				(
					if highest
					{
						-(die.value as i64)
					}
					else
					{
						die.value as i64
					},
					index
				)
			})
			.collect::<Vec<_>>();
		kept.sort_unstable();
		for &(_, index) in kept.iter().take(count.max(0) as usize)
		{
			record.dice[index].fate = Fate::Dropped;
		}
	}
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(6618230984571203);
	for _ in 0..10_000
	{
		let len = rng.random_range(0..=6);
		let results = (0..len)
			.map(|_| rng.random_range(1..=4))
			.collect::<Vec<_>>();
		let mut record = record(4, &results);
		let mut model = record.clone();
		for _ in 0..rng.random_range(0..=5)
		{
			let count = rng.random_range(-1..=len + 1);
			let kept = model.kept().len() as i32;
			let keep = kept - count.clamp(0, kept);
			match rng.random_range(0..5)
			{
				0 =>
				{
					record.drop_lowest(count);
					cross_out(&mut model, count, false);
				},
				1 =>
				{
					record.drop_highest(count);
					cross_out(&mut model, count, true);
				},
				2 =>
				{
					record.keep_lowest(count);
					cross_out(&mut model, keep, true);
				},
				3 =>
				{
					record.keep_highest(count);
					cross_out(&mut model, keep, false);
				},
				_ =>
				{
					// Reroll both records alike.
					let seed = rng.random::<u64>();
					let threshold = rng.random_range(1..=4);
					let criterion = Criterion::within(1, threshold);
					let limit = Some(rng.random_range(0..=2));
					record.reroll(
						&mut StdRng::seed_from_u64(seed),
						&criterion,
						limit
					);
					model.reroll(
						&mut StdRng::seed_from_u64(seed),
						&criterion,
						limit
					);
				}
			}
			assert_eq!(record, model, "{} vs {}", record, model);
		}
		let mut expected = model.kept();
		expected.sort();
		assert_eq!(
			record.sum(),
			expected.iter().fold(0i32, |a, b| a.saturating_add(*b)),
			"{}",
			record
		);
		assert!(
			record.chains().iter().all(|chain| chain
				.iter()
				.filter(|&&i| record.dice[i].is_kept())
				.count()
				<= 1),
			"{}",
			record
		);
	}
}

/// Test that drops and keeps over random records with chains keep the same
/// totals as dropping from the ends of the sorted totals of the chains, which
/// a reference computes by following each die's links to the die that begins
/// its chain, and that crossing out a chain leaves none of its dice kept.
#[test]
fn test_chains_agree_with_sorted_totals()
{
	/// Answer the index of the first die of the chain of the specified die.
	fn root(record: &RollingRecord, index: usize) -> usize
	{
		match record.dice[index].origin.link()
		{
			Some(link) if link < index => root(record, link),
			_ => index
		}
	}
	/// Answer the totals of the chains still kept, by first die.
	fn totals(record: &RollingRecord) -> Vec<(usize, i32)>
	{
		let mut chains = std::collections::BTreeMap::<usize, Vec<Die>>::new();
		for (index, die) in record.dice.iter().enumerate()
		{
			if die.is_kept()
			{
				chains.entry(root(record, index)).or_default().push(*die);
			}
		}
		chains
			.into_iter()
			.map(|(root, dice)| {
				// The earliest saturated die decides, should several share a
				// chain.
				let saturated = dice
					.iter()
					.find(|die| matches!(die.origin, Origin::Saturates(_)));
				let total = match saturated
				{
					Some(die) => die.value,
					None =>
					{
						let mut values = dice
							.iter()
							.map(|die| die.value)
							.collect::<Vec<_>>();
						values.sort();
						values.iter().fold(0i32, |a, b| a.saturating_add(*b))
					}
				};
				(root, total)
			})
			.collect()
	}
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(2093857120938471);
	for _ in 0..10_000
	{
		let len = rng.random_range(0..=8);
		let mut record = record(6, &[]);
		for index in 0..len
		{
			let value = rng.random_range(-2..=6);
			let origin = match (index, rng.random_range(0..6))
			{
				(0, _) | (_, 0..=2) => Origin::Rolled,
				(_, 3) => Origin::Extends(rng.random_range(0..index)),
				(_, 4) =>
				{
					let link = rng.random_range(0..index);
					record.dice[link].fate = Fate::Rerolled;
					Origin::Replaces(link)
				},
				_ =>
				{
					let value = [i32::MIN, 0, i32::MAX][rng.random_range(0..3)];
					record.dice.push(die(
						value,
						Origin::Saturates(rng.random_range(0..index))
					));
					continue
				}
			};
			record.dice.push(die(value, origin));
		}
		let mut sorted = totals(&record)
			.into_iter()
			.map(|(_, total)| total)
			.collect::<Vec<_>>();
		sorted.sort();
		let before = record.clone();
		let kept = sorted.len() as i32;
		let count = rng.random_range(-1..=kept + 1);
		let keep = (kept - count.clamp(0, kept)) as usize;
		let lowest = rng.random_bool(0.5);
		let expected: Vec<i32> = match (rng.random_bool(0.5), lowest)
		{
			(true, true) =>
			{
				record.drop_lowest(count);
				sorted[(count.clamp(0, kept) as usize)..].to_vec()
			},
			(true, false) =>
			{
				record.drop_highest(count);
				sorted[..keep].to_vec()
			},
			(false, true) =>
			{
				record.keep_lowest(count);
				sorted[..(count.clamp(0, kept) as usize)].to_vec()
			},
			(false, false) =>
			{
				record.keep_highest(count);
				sorted[keep..].to_vec()
			}
		};
		let mut actual = totals(&record)
			.into_iter()
			.map(|(_, total)| total)
			.collect::<Vec<_>>();
		actual.sort();
		assert_eq!(actual, expected, "{} -> {}", before, record);
		assert_eq!(
			record.sum(),
			expected.iter().fold(0i32, |a, b| a.saturating_add(*b)),
			"{}",
			record
		);
		// A chain is kept whole or crossed out whole, and the dice that a
		// reroll crossed out stay as they were.
		for (index, (after, die)) in
			record.dice.iter().zip(&before.dice).enumerate()
		{
			assert_eq!(after.value, die.value);
			assert_eq!(after.origin, die.origin);
			match die.fate
			{
				Fate::Kept => assert!(
					after.is_kept()
						|| (after.fate == Fate::Dropped
							&& record.dice.iter().enumerate().all(
								|(other, d)| {
									root(&record, other) != root(&record, index)
										|| !d.is_kept()
								}
							)),
					"{} -> {}",
					before,
					record
				),
				fate => assert_eq!(after.fate, fate)
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Rerolls.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that every form of predicate resolves to the set of results that it
/// names, including at the extremes of `i32`, where a strict comparison or a
/// half-open range may name nothing, and that empty and inverted ranges match
/// nothing.
#[test]
fn test_predicates_resolve()
{
	let imm = |value| AddressingMode::Immediate(Immediate(value));
	let value = |op: AddressingMode| match op
	{
		AddressingMode::Immediate(Immediate(value)) => value,
		_ => unreachable!()
	};
	let thresholds = [
		i32::MIN,
		i32::MIN + 1,
		-3,
		-1,
		0,
		1,
		3,
		i32::MAX - 1,
		i32::MAX
	];
	let mut probes = thresholds.to_vec();
	probes.extend(-5..=5);
	type Comparison = fn(AddressingMode) -> Predicate;
	type Expected = fn(i32, i32) -> bool;
	let comparisons: [(Comparison, Expected); 6] = [
		(Predicate::Eq, |v, t| v == t),
		(Predicate::Ne, |v, t| v != t),
		(Predicate::Lt, |v, t| v < t),
		(Predicate::Le, |v, t| v <= t),
		(Predicate::Gt, |v, t| v > t),
		(Predicate::Ge, |v, t| v >= t)
	];
	for (predicate, expected) in comparisons
	{
		for t in thresholds
		{
			let criterion = predicate(imm(t)).resolve(value);
			for &v in &probes
			{
				assert_eq!(
					criterion.matches(v),
					expected(v, t),
					"{} with {}: {:?}",
					predicate(imm(t)),
					v,
					criterion
				);
			}
		}
	}
	for start in thresholds
	{
		for end in thresholds
		{
			let half_open = Predicate::InRange {
				start: imm(start),
				end: imm(end)
			}
			.resolve(value);
			let inclusive = Predicate::InRangeInclusive {
				start: imm(start),
				end: imm(end)
			}
			.resolve(value);
			for &v in &probes
			{
				assert_eq!(
					half_open.matches(v),
					start <= v && v < end,
					"{}..{} with {}",
					start,
					end,
					v
				);
				assert_eq!(
					inclusive.matches(v),
					start <= v && v <= end,
					"{}..={} with {}",
					start,
					end,
					v
				);
			}
		}
	}
	let list = Predicate::In(vec![3, -1, 3, i32::MIN]).resolve(value);
	assert_eq!(list, Criterion::Among(vec![i32::MIN, -1, 3]));
	for &v in &probes
	{
		assert_eq!(list.matches(v), [3, -1, i32::MIN].contains(&v), "{}", v);
	}
	assert!(!Predicate::In(Vec::new()).resolve(value).matches(0));
}

/// Answer a record of the specified kind holding the specified results, in
/// the order rolled, none of them crossed out.
///
/// # Parameters
/// - `kind`: The kind of the record.
/// - `results`: The results, in the order rolled.
///
/// # Returns
/// The rolling record.
fn record_of(kind: RollingRecordKind<i32>, results: &[i32]) -> RollingRecord
{
	RollingRecord {
		kind,
		dice: results.iter().copied().map(Die::kept).collect()
	}
}

/// Test that a reroll crosses out each matching die, appends its replacement,
/// linked to it, after the dice already rolled, and finishes each die before
/// beginning the next, and that a record prints its replacements in the order
/// rolled.
#[test]
fn test_reroll_appends_linked_replacements()
{
	// A die whose only face is `3` always rolls a `3`, so the replacements are
	// known, even though the record holds other results.
	let kind = || RollingRecordKind::Custom {
		count: 3,
		faces: vec![3]
	};
	let mut rng = StdRng::seed_from_u64(0);
	let mut record = record_of(kind(), &[2, 1, 4]);
	assert_eq!(
		record.reroll(&mut rng, &Criterion::Within(1..=1), Some(1)),
		1
	);
	assert_eq!(record.to_string(), "3d[3]: [2, ~1, 4, 3]");
	assert_eq!(record.dice[1].fate, Fate::Rerolled);
	assert_eq!(record.dice[3].origin, Origin::Replaces(1));
	assert_eq!(record.rerolled(), vec![1]);
	assert_eq!(record.dropped(), Vec::<i32>::new());
	assert_eq!(record.sum(), 9);
	// Each die finishes its chain before the next begins: the `1` and its
	// replacements precede the `4` and its replacements.
	let mut record = record_of(kind(), &[1, 2, 4]);
	assert_eq!(
		record.reroll(&mut rng, &Criterion::Outside(2..=2), Some(3)),
		6
	);
	assert_eq!(
		record.to_string(),
		"3d[3]: [~1, 2, ~4, ~3, ~3, 3, ~3, ~3, 3]"
	);
	let links = record.dice.iter().map(|die| die.origin).collect::<Vec<_>>();
	assert_eq!(
		links,
		[
			Origin::Rolled,
			Origin::Rolled,
			Origin::Rolled,
			Origin::Replaces(0),
			Origin::Replaces(3),
			Origin::Replaces(4),
			Origin::Replaces(2),
			Origin::Replaces(6),
			Origin::Replaces(7)
		]
	);
	// Only the dice still kept are rerolled, and their replacements are kept,
	// whatever the drops before them.
	let mut record = record_of(kind(), &[1, 1, 5]);
	record.drop_lowest(1);
	assert_eq!(record.reroll(&mut rng, &Criterion::Within(1..=1), None), 1);
	assert_eq!(record.to_string(), "3d[3]: [~1, ~1, 5, 3]");
	assert_eq!(record.dropped(), vec![1]);
	assert_eq!(record.rerolled(), vec![1]);
	assert_eq!(record.kept(), vec![5, 3]);
}

/// Test the degenerate rerolls: a limit of zero or less rerolls nothing, an
/// unlimited reroll that matches every face rolls nothing but crosses out
/// every matching die, a die without faces and an empty range show only the
/// face `0`, and a record without dice is unchanged.
#[test]
fn test_reroll_degeneracies()
{
	let mut rng = StdRng::seed_from_u64(0);
	let standard = |faces| RollingRecordKind::Standard { count: 3, faces };
	let everything = Criterion::Outside(RangeInclusive::new(1, 0));
	for limit in [0, -1, i32::MIN]
	{
		let mut record = record_of(standard(6), &[1, 2, 3]);
		assert_eq!(record.reroll(&mut rng, &everything, Some(limit)), 0);
		assert_eq!(record.to_string(), "3D6: [1, 2, 3]");
	}
	for (kind, results, criterion, expected) in [
		(
			standard(1),
			&[1, 1, 1][..],
			Criterion::Within(1..=1),
			"3D1: [~1, ~1, ~1]"
		),
		(
			standard(6),
			&[1, 6, 3],
			Criterion::Within(-5..=6),
			"3D6: [~1, ~6, ~3]"
		),
		(
			standard(6),
			&[1, 6],
			Criterion::Outside(7..=9),
			"3D6: [~1, ~6]"
		),
		(
			standard(6),
			&[2, 5],
			Criterion::Among(vec![0, 1, 2, 3, 4, 5, 6, 7]),
			"3D6: [~2, ~5]"
		),
		(
			standard(0),
			&[0, 0],
			Criterion::Within(0..=0),
			"3D0: [~0, ~0]"
		),
		(standard(-1), &[0], Criterion::Among(vec![0]), "3D-1: [~0]"),
		(
			RollingRecordKind::Range { start: 4, end: 2 },
			&[0],
			Criterion::Within(0..=0),
			"[4:2]: [~0]"
		),
		(
			RollingRecordKind::Range { start: -2, end: 2 },
			&[-1],
			Criterion::Within(-2..=2),
			"[-2:2]: [~-1]"
		),
		(
			RollingRecordKind::Custom {
				count: 2,
				faces: vec![2, 4, 2]
			},
			&[4, 2],
			Criterion::Among(vec![2, 4]),
			"2d[2, 4, 2]: [~4, ~2]"
		),
		(
			RollingRecordKind::Range {
				start: i32::MIN,
				end: i32::MAX
			},
			&[7],
			Criterion::Within(i32::MIN..=i32::MAX),
			"[-2147483648:2147483647]: [~7]"
		)
	]
	{
		let mut record = record_of(kind, results);
		assert_eq!(
			record.reroll(&mut rng, &criterion, None),
			0,
			"{}",
			expected
		);
		assert_eq!(record.to_string(), expected);
		assert_eq!(record.sum(), 0, "{}", expected);
	}
	// A die without faces rolls only zeros, so a limited reroll of its zeros
	// rolls zeros.
	let mut record = record_of(standard(0), &[0, 0]);
	assert_eq!(
		record.reroll(&mut rng, &Criterion::Within(0..=0), Some(2)),
		4
	);
	assert_eq!(record.kept(), vec![0, 0]);
	assert_eq!(record.rerolled(), vec![0, 0, 0, 0]);
	// No dice, and an uninitialized record, have nothing to reroll.
	for kind in [standard(6), RollingRecordKind::Uninitialized]
	{
		let mut record = record_of(kind, &[]);
		assert_eq!(record.reroll(&mut rng, &everything, None), 0);
		assert_eq!(record.reroll(&mut rng, &everything, Some(3)), 0);
		assert!(record.dice.is_empty());
	}
}

/// Test, over random records, criteria, and limits, that a reroll keeps its
/// invariants: it crosses out exactly the dice that it rerolls, only after
/// they match; every replacement follows the dice already rolled, links to the
/// die that it replaces, and shows a face of the die; each chain begins at a
/// die kept and matching when the reroll began, holds at most as many rerolls
/// as the limit, and ends at a kept die that no longer matches or that
/// exhausted the limit; the chains follow one another in the order of the dice
/// that begin them; and the reroll rolls one die per reroll.
#[test]
fn test_reroll_invariants()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(1098257109387512);
	for _ in 0..10_000
	{
		let (kind, faces): (_, Vec<i32>) = match rng.random_range(0..4)
		{
			0 =>
			{
				let faces = rng.random_range(-1..=6);
				(
					RollingRecordKind::Standard { count: 0, faces },
					match faces <= 0
					{
						true => vec![0],
						false => (1..=faces).collect()
					}
				)
			},
			1 =>
			{
				let faces = (0..rng.random_range(1..=4))
					.map(|_| rng.random_range(-3..=3))
					.collect::<Vec<_>>();
				(
					RollingRecordKind::Custom {
						count: 0,
						faces: faces.clone()
					},
					faces
				)
			},
			_ =>
			{
				let start = rng.random_range(-3..=3);
				let end = rng.random_range(-3..=3);
				(
					RollingRecordKind::Range { start, end },
					match end < start
					{
						true => vec![0],
						false => (start..=end).collect()
					}
				)
			}
		};
		let len = rng.random_range(0..=5);
		let results = (0..len)
			.map(|_| faces[rng.random_range(0..faces.len())])
			.collect::<Vec<_>>();
		let mut record = record_of(kind, &results);
		if rng.random_bool(0.5)
		{
			record.drop_lowest(rng.random_range(0..=2));
		}
		let criterion = match rng.random_range(0..3)
		{
			0 =>
			{
				let start = rng.random_range(-4..=4);
				Criterion::within(start, rng.random_range(-4..=4))
			},
			1 =>
			{
				let start = rng.random_range(-4..=4);
				Criterion::Outside(start..=rng.random_range(-4..=4))
			},
			_ =>
			{
				let mut values = (0..rng.random_range(0..=4))
					.map(|_| rng.random_range(-4..=4))
					.collect::<Vec<_>>();
				values.sort();
				values.dedup();
				Criterion::Among(values)
			}
		};
		let limit = match rng.random_bool(0.3)
		{
			true => None,
			false => Some(rng.random_range(-1..=3))
		};
		let before = record.clone();
		let rolled = record.reroll(&mut rng, &criterion, limit);
		let context = format!(
			"{} reroll {:?} {:?} -> {}",
			before, criterion, limit, record
		);
		let len = before.dice.len();
		assert_eq!(record.dice.len() - len, rolled as usize, "{}", context);
		let everything =
			limit.is_none() && faces.iter().all(|f| criterion.matches(*f));
		let mut chains = Vec::new();
		for (index, die) in before.dice.iter().enumerate()
		{
			let after = record.dice[index];
			assert_eq!(after.value, die.value, "{}", context);
			let starts = die.is_kept() && criterion.matches(die.value);
			match starts && limit.is_none_or(|limit| limit > 0)
			{
				true => assert_eq!(after.fate, Fate::Rerolled, "{}", context),
				false => assert_eq!(after.fate, die.fate, "{}", context)
			}
			if starts && !everything
			{
				chains.push(index);
			}
		}
		// Follow each chain from the die that begins it.
		let mut next = len;
		for start in chains
		{
			let mut current = start;
			let mut times = 0;
			while next < record.dice.len()
				&& record.dice[next].origin == Origin::Replaces(current)
			{
				assert_eq!(
					record.dice[current].fate,
					Fate::Rerolled,
					"{}",
					context
				);
				assert!(
					criterion.matches(record.dice[current].value),
					"{}",
					context
				);
				assert!(
					faces.contains(&record.dice[next].value),
					"{}",
					context
				);
				current = next;
				next += 1;
				times += 1;
			}
			let end = record.dice[current];
			match limit
			{
				Some(limit) =>
				{
					assert!(times <= limit.max(0) as usize, "{}", context);
					if times == 0
					{
						assert!(limit <= 0, "{}", context);
					}
					else
					{
						assert!(end.is_kept(), "{}", context);
						assert!(
							!criterion.matches(end.value)
								|| times == limit as usize,
							"{}",
							context
						);
					}
				},
				None =>
				{
					assert!(end.is_kept(), "{}", context);
					assert!(!criterion.matches(end.value), "{}", context);
				}
			}
		}
		assert_eq!(next, record.dice.len(), "{}", context);
	}
}

/// Answer an evaluator for a function that rolls `4D6`, rerolls by the
/// specified clause, and sums the record.
///
/// # Parameters
/// - `clause`: The predicate and limit of the reroll, as the assembler reads
///   them.
///
/// # Returns
/// The evaluator.
fn reroll_evaluator(clause: &str) -> Evaluator
{
	Evaluator::new(
		Assembler::assemble(&format!(
			"Function({{t}}@0) r#2 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll \
			 standard dice 4D6\n\t\t⚅0 <- reroll {} from ⚅0\n\t\t@1 <- sum \
			 rolling record ⚅0\n\t\treturn @1\n",
			clause
		))
		.unwrap()
	)
}

/// Test that evaluation rerolls as the instruction directs: it reads the
/// threshold and limit from their registers, sums the kept dice, counts each
/// reroll as a die, and stays within the bounds, whose count of outcomes
/// counts every reroll that each die might need.
#[test]
fn test_evaluation_rerolls()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(9810257109825710);
	for (clause, t, count) in [
		("<= @0 up to 2 times", 2, 6u128.pow(4 + 2 * 4)),
		("< @0 unlimited times", 4, 6u128.pow(4) * 3u128.pow(4)),
		("in 2..=@0 up to @0 times", 3, 6u128.pow(4 + 3 * 4)),
		("in [1, 6] unlimited times", 0, 6u128.pow(4) * 4u128.pow(4)),
		("!= @0 up to 1 times", 6, 6u128.pow(4 + 4)),
		(">= @0 unlimited times", 1, 6u128.pow(4))
	]
	{
		let mut evaluator = reroll_evaluator(clause);
		let bounds = evaluator.bounds_over([Some(t.into())], []).unwrap();
		assert_eq!(bounds.count, Some(count), "{}", clause);
		for _ in 0..200
		{
			let evaluation = evaluator.evaluate([t], &mut rng).unwrap();
			let [record] = &evaluation.records[..]
			else
			{
				panic!("expected one record: {:?}", evaluation.records)
			};
			assert_eq!(
				record.kept().len(),
				match clause
				{
					">= @0 unlimited times" => 0,
					_ => 4
				},
				"{}: {}",
				clause,
				record
			);
			assert_eq!(
				record.kept().iter().sum::<i32>(),
				evaluation.result,
				"{}: {}",
				clause,
				record
			);
			assert_eq!(evaluation.dice, record.dice.len() as u64, "{}", record);
			assert!(bounds.value.contains(evaluation.result), "{}", record);
			assert!(evaluation.dice <= bounds.dice, "{}", record);
			match clause
			{
				"< @0 unlimited times" =>
				{
					assert!(record.kept().iter().all(|&v| v >= 4), "{}", record)
				},
				"in [1, 6] unlimited times" =>
				{
					assert!(
						record.kept().iter().all(|&v| v != 1 && v != 6),
						"{}",
						record
					)
				},
				_ =>
				{}
			}
		}
	}
}

/// Test that a metered evaluation charges each reroll as a die, so that a
/// budget bounds even an unlimited reroll, and refuses the first reroll that
/// would exceed it, and that metering within a sufficient budget does not
/// disturb the draws.
#[test]
fn test_dice_budget_meters_rerolls()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let seed = 1209857102398571;
	let mut evaluator = reroll_evaluator("<= @0 unlimited times");
	for i in 0..100
	{
		let unmetered = evaluator
			.evaluate([5], &mut StdRng::seed_from_u64(seed + i))
			.unwrap();
		let dice = unmetered.dice;
		assert!(dice >= 4);
		let metered = evaluator
			.evaluate_metered([5], &mut StdRng::seed_from_u64(seed + i), dice)
			.unwrap();
		assert_eq!(metered, unmetered);
		let mut rng = StdRng::seed_from_u64(seed + i);
		match dice
		{
			// No die rerolled, so one die fewer refuses the roll itself.
			4 => assert_eq!(
				evaluator.evaluate_metered([5], &mut rng, dice - 1),
				Err(EvaluationError::DiceBudgetExhausted {
					requested: 4,
					remaining: 3,
					consumed: 0
				})
			),
			_ => assert_eq!(
				evaluator.evaluate_metered([5], &mut rng, dice - 1),
				Err(EvaluationError::DiceBudgetExhausted {
					requested: 1,
					remaining: 0,
					consumed: dice - 1
				})
			)
		}
	}
	// A reroll that matches every face rolls nothing, so it costs nothing.
	let mut evaluator = reroll_evaluator("<= @0 unlimited times");
	let evaluation = evaluator
		.evaluate_metered([6], &mut StdRng::seed_from_u64(seed), 4)
		.unwrap();
	assert_eq!(evaluation.result, 0);
	assert_eq!(evaluation.dice, 4);
}

/// Answer every nonempty interval whose endpoints lie within the specified
/// range.
///
/// # Parameters
/// - `range`: The range of endpoints.
///
/// # Returns
/// The intervals.
fn intervals(range: RangeInclusive<i32>) -> Vec<EvaluationBounds>
{
	range
		.clone()
		.flat_map(|min| (min..=*range.end()).map(move |max| (min, max).into()))
		.collect()
}

/// Answer the faces of a die of the exhaustive reroll bounds grid and the
/// number of dice that it rolls, given the values of its parameters, as the
/// evaluator rolls them. A die without faces, or an empty range, shows only the
/// face `0`.
///
/// # Parameters
/// - `die`: The index of the die among those of the grid: `2D@2`, `@2:@3`, or
///   `2D[-1, 1, 1, 3]`.
/// - `c`: The value of `@2`.
/// - `d`: The value of `@3`.
///
/// # Returns
/// The faces and the number of dice.
fn grid_faces(die: usize, c: i32, d: i32) -> (Vec<i32>, i32)
{
	match die
	{
		0 if c <= 0 => (vec![0], 2),
		0 => ((1..=c).collect(), 2),
		1 if d < c => (vec![0], 1),
		1 => ((c..=d).collect(), 1),
		_ => (vec![-1, 1, 1, 3], 2)
	}
}

/// Test that the bounds of the sum of a rerolled record are exact over an
/// exhaustive grid of dice, predicates, limits, and intervals of their
/// operands, against the brute-force extrema of every assignment of their
/// operands. A limited reroll leaves its dice showing any face of the die, and
/// an unlimited one any face that the predicate does not match, or crosses out
/// every die if it matches every face.
#[test]
fn test_reroll_bounds_exhaustive()
{
	let dice = [
		("roll standard dice 2D@2", 1),
		("roll range @2:@3", 2),
		("roll custom dice 2D[-1, 1, 1, 3]", 0)
	];
	let (a, b) = (
		AddressingMode::from(RegisterIndex(0)),
		AddressingMode::from(RegisterIndex(1))
	);
	let predicates = [
		Predicate::Eq(a),
		Predicate::Ne(a),
		Predicate::Lt(a),
		Predicate::Le(a),
		Predicate::Gt(a),
		Predicate::Ge(a),
		Predicate::In(vec![]),
		Predicate::In(vec![3, -1, 1]),
		Predicate::In(vec![0, 2, 3]),
		Predicate::InRange { start: a, end: b },
		Predicate::InRangeInclusive { start: a, end: b }
	];
	for (die, (roll, die_params)) in dice.iter().enumerate()
	{
		for predicate in &predicates
		{
			for limit in ["up to 1 times", "unlimited times"]
			{
				let function = Assembler::assemble(&format!(
					"Function({{a}}@0, {{b}}@1, {{c}}@2, {{d}}@3) r#5 ⚅#1\n\
					 \textern[]\n\tbody:\n\t\t⚅0 <- {}\n\t\t⚅0 <- reroll {} {} \
					 from ⚅0\n\t\t@4 <- sum rolling record ⚅0\n\t\treturn @4\n",
					roll, predicate, limit
				))
				.unwrap();
				let evaluator = Evaluator::new(function);
				let fixed = vec![EvaluationBounds::from(0)];
				let pred_intervals = match predicate.operands().len()
				{
					0 => vec![fixed.clone(), fixed.clone()],
					1 => vec![intervals(-1..=4), fixed.clone()],
					_ => vec![intervals(0..=3), intervals(0..=3)]
				};
				let die_intervals = match die_params
				{
					0 => vec![fixed.clone(), fixed.clone()],
					1 => vec![intervals(-1..=3), fixed.clone()],
					_ => vec![intervals(-1..=2), intervals(-1..=2)]
				};
				for i0 in &pred_intervals[0]
				{
					for i1 in &pred_intervals[1]
					{
						for i2 in &die_intervals[0]
						{
							for i3 in &die_intervals[1]
							{
								let bounds = evaluator
									.bounds_over(
										[*i0, *i1, *i2, *i3].map(Some),
										[]
									)
									.unwrap();
								let mut truth: Option<(i32, i32)> = None;
								for v0 in i0.min..=i0.max
								{
									for v1 in i1.min..=i1.max
									{
										let criterion =
											predicate.resolve(|op| match op
											{
												op if op == a => v0,
												_ => v1
											});
										for v2 in i2.min..=i2.max
										{
											for v3 in i3.min..=i3.max
											{
												let (faces, count) =
													grid_faces(die, v2, v3);
												let rest = faces
													.iter()
													.copied()
													.filter(|face| {
														limit
															!= "unlimited times"
															|| !criterion
																.matches(*face)
													})
													.collect::<Vec<_>>();
												let sum = match (
													rest.iter().min(),
													rest.iter().max()
												)
												{
													(Some(min), Some(max)) => (
														count * min,
														count * max
													),
													_ => (0, 0)
												};
												truth = Some(match truth
												{
													Some((min, max)) => (
														min.min(sum.0),
														max.max(sum.1)
													),
													None => sum
												});
											}
										}
									}
								}
								assert_eq!(
									bounds.value,
									truth.unwrap().into(),
									"{} reroll {} {} over {:?}",
									roll,
									predicate,
									limit,
									[i0, i1, i2, i3]
								);
							}
						}
					}
				}
			}
		}
	}
}

/// Answer the bounds of the specified assembled body, which takes no
/// parameters and returns `@0`.
///
/// # Parameters
/// - `body`: The instructions, one per line.
///
/// # Returns
/// The bounds.
fn body_bounds(body: &str) -> crate::Bounds
{
	// Count the registers and records up to the greatest index of each.
	let count = |sigil: char| {
		body.split(sigil)
			.skip(1)
			.filter_map(|rest| {
				let digits = rest
					.chars()
					.take_while(char::is_ascii_digit)
					.collect::<String>();
				digits.parse::<usize>().ok()
			})
			.max()
			.map_or(0, |index| index + 1)
	};
	let (registers, records) = (count('@'), count('⚅'));
	let text = format!(
		"Function() r#{} ⚅#{}\n\textern[]\n\tbody:\n{}",
		registers,
		records,
		body.lines()
			.map(|line| format!("\t\t{}\n", line.trim()))
			.collect::<String>()
	);
	Evaluator::new(Assembler::assemble(&text).unwrap())
		.bounds_over([], [])
		.unwrap()
}

/// Test that the bounds of a reroll chained with drops, keeps, and other
/// rerolls are exact: an unlimited reroll narrows the results of the dice
/// still kept, a limited reroll that may match restores the faces of the die,
/// and a reroll that cannot match changes nothing.
#[test]
fn test_reroll_bounds_chains()
{
	for (body, value) in [
		// A later limited reroll may roll a 1 again.
		(
			"⚅0 <- roll standard dice 1D6
			⚅0 <- reroll = 1 unlimited times from ⚅0
			⚅0 <- reroll = 6 up to 1 times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(1, 6)
		),
		// No die still kept shows a 1, so the second reroll cannot happen.
		(
			"⚅0 <- roll standard dice 1D6
			⚅0 <- reroll = 1 unlimited times from ⚅0
			⚅0 <- reroll = 1 up to 1 times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(2, 6)
		),
		(
			"⚅0 <- roll standard dice 1D6
			⚅0 <- reroll <= 2 unlimited times from ⚅0
			⚅0 <- reroll = 6 unlimited times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(1, 5)
		),
		(
			"⚅0 <- roll standard dice 4D6
			⚅0 <- drop lowest 1 from ⚅0
			⚅0 <- reroll <= 2 unlimited times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(9, 18)
		),
		(
			"⚅0 <- roll standard dice 4D6
			⚅0 <- keep highest 2 from ⚅0
			⚅0 <- reroll >= 1 unlimited times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(0, 0)
		),
		(
			"⚅0 <- roll custom dice 2D[1, 9]
			⚅0 <- reroll = 1 unlimited times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0",
			(18, 18)
		),
		// The range may be empty, showing 0, which the reroll matches.
		(
			"⚅0 <- roll standard dice 1D3
			@0 <- sum rolling record ⚅0
			⚅1 <- roll range 2:@0
			⚅1 <- reroll in 0..2 unlimited times from ⚅1
			@1 <- sum rolling record ⚅1
			return @1",
			(0, 3)
		)
	]
	{
		assert_eq!(body_bounds(body).value, value.into(), "{}", body);
	}
}

/// Test that the bounds count every reroll that each die still kept might
/// need, as though it were rolled in advance: a limited reroll multiplies the
/// outcomes by the faces of the die, raised to the limit, for every die still
/// kept, and an unlimited one by the faces that it does not match; that a
/// reroll that cannot happen counts nothing; and that a reroll by a random
/// operand, or of a random number of dice, counts nothing at all.
#[test]
fn test_reroll_bounds_count()
{
	let reroll = |roll: &str, clause: &str| {
		body_bounds(&format!(
			"⚅0 <- roll {}\n⚅0 <- reroll {} from ⚅0\n@0 <- sum rolling record \
			 ⚅0\nreturn @0",
			roll, clause
		))
		.count
	};
	for (roll, clause, count) in [
		(
			"standard dice 2D6",
			"= 1 up to 2 times",
			Some(6u128.pow(2 + 2 * 2))
		),
		("standard dice 2D6", "= 1 unlimited times", Some(36 * 25)),
		(
			"standard dice 2D6",
			"in [1, 2, 7] unlimited times",
			Some(36 * 16)
		),
		// Every face matches, so nothing is rolled.
		("standard dice 2D6", ">= 1 unlimited times", Some(36)),
		// Every face matches, but the limit rolls each die again regardless.
		("standard dice 2D6", ">= 1 up to 1 times", Some(36 * 36)),
		// No face matches.
		("standard dice 2D6", "= 7 up to 3 times", Some(36)),
		("standard dice 2D6", "in 4..4 unlimited times", Some(36)),
		("standard dice 2D6", "in 5..=2 up to 1 times", Some(36)),
		// The limit is not positive.
		("standard dice 2D6", "= 1 up to 0 times", Some(36)),
		("standard dice 2D6", "= 1 up to -1 times", Some(36)),
		// A face of a custom die counts once for every time that it is listed.
		("custom dice 2D[1, 1, 2]", "= 1 unlimited times", Some(9)),
		(
			"custom dice 2D[1, 1, 2]",
			"= 2 unlimited times",
			Some(9 * 4)
		),
		("custom dice 2D[1, 1, 2]", "= 1 up to 1 times", Some(9 * 9)),
		// A die without faces, or an empty range, shows only `0`.
		("standard dice 2D0", "= 0 up to 1 times", Some(1)),
		("standard dice 2D0", "= 0 unlimited times", Some(1)),
		("range 3:1", "= 0 up to 2 times", Some(1)),
		("range 1:3", "= 2 unlimited times", Some(6)),
		("range 1:3", "!= 2 up to 1 times", Some(9))
	]
	{
		assert_eq!(reroll(roll, clause), count, "{} reroll {}", roll, clause);
	}
	// A reroll counts the dice still kept.
	assert_eq!(
		body_bounds(
			"⚅0 <- roll standard dice 3D6
			⚅0 <- keep highest 1 from ⚅0
			⚅0 <- reroll = 1 up to 1 times from ⚅0
			@0 <- sum rolling record ⚅0
			return @0"
		)
		.count,
		Some(6u128.pow(3 + 1))
	);
	// A random threshold, endpoint, or limit counts nothing.
	for clause in [
		"< @0 up to 1 times",
		"in 1..@0 unlimited times",
		"= 1 up to @0 times"
	]
	{
		assert_eq!(
			body_bounds(&format!(
				"⚅0 <- roll standard dice 1D3
				@0 <- sum rolling record ⚅0
				⚅1 <- roll standard dice 2D6
				⚅1 <- reroll {} from ⚅1
				@1 <- sum rolling record ⚅1
				return @1",
				clause
			))
			.count,
			None,
			"{}",
			clause
		);
	}
	// A random number of dice still kept counts nothing, unless the reroll
	// cannot happen, whence it counts the same whatever the number.
	for (clause, count) in [
		("= 1 up to 1 times", None),
		("= 7 up to 1 times", Some(3 * 6u128.pow(4)))
	]
	{
		assert_eq!(
			body_bounds(&format!(
				"⚅0 <- roll standard dice 1D3
				@0 <- sum rolling record ⚅0
				⚅1 <- roll standard dice 4D6
				⚅1 <- drop lowest @0 from ⚅1
				⚅1 <- reroll {} from ⚅1
				@1 <- sum rolling record ⚅1
				return @1",
				clause
			))
			.count,
			count,
			"{}",
			clause
		);
	}
}

/// Test that the bounds of the dice rolled count every reroll that each die
/// still kept may need: none for a reroll that cannot match, as many as the
/// limit for each die for a limited reroll, and without bound for an unlimited
/// reroll that may roll, but none for one that matches every face.
#[test]
fn test_reroll_bounds_dice()
{
	for (clause, dice) in [
		("= 1 up to 3 times", 4 + 4 * 3),
		("= 1 up to 0 times", 4),
		("= 7 up to 3 times", 4),
		("= 7 unlimited times", 4),
		("= 1 unlimited times", u64::MAX),
		(">= 1 unlimited times", 4)
	]
	{
		assert_eq!(
			body_bounds(&format!(
				"⚅0 <- roll standard dice 4D6\n⚅0 <- reroll {} from ⚅0\n@0 <- \
				 sum rolling record ⚅0\nreturn @0",
				clause
			))
			.dice,
			dice,
			"{}",
			clause
		);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Explodes.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that an explode keeps each matching die, appends its extensions,
/// linked to it, after the dice already rolled, and finishes each die before
/// beginning the next; that only the dice still kept explode; and that the sum
/// and every later keep count each chain as one die, worth its total.
#[test]
fn test_explode_appends_linked_extensions()
{
	// A die whose only face is `2` always rolls a `2`, and so does a die whose
	// only face is `6` always roll a `6`, so the extensions are known, even
	// though the record holds other results.
	let kind = |faces| RollingRecordKind::Custom {
		count: 3,
		faces: vec![faces]
	};
	let mut rng = StdRng::seed_from_u64(0);
	let mut record = record_of(kind(2), &[6, 3, 6]);
	assert_eq!(
		record.explode(&mut rng, &Criterion::Within(6..=6), Some(1)),
		2
	);
	assert_eq!(record.to_string(), "3d[2]: [6, 3, 6, 2, 2]");
	let links = record.dice.iter().map(|die| die.origin).collect::<Vec<_>>();
	assert_eq!(
		links,
		[
			Origin::Rolled,
			Origin::Rolled,
			Origin::Rolled,
			Origin::Extends(0),
			Origin::Extends(2)
		]
	);
	assert!(record.dice.iter().all(Die::is_kept));
	assert_eq!(record.chains(), vec![vec![0, 3], vec![1], vec![2, 4]]);
	assert_eq!(record.sum(), 19);
	// Of the chains worth `8`, `3`, and `8`, keeping the highest crosses out
	// the `3` and then the `8` whose earliest die was rolled first.
	record.keep_highest(1);
	assert_eq!(record.to_string(), "3d[2]: [~6, ~3, 6, ~2, 2]");
	assert_eq!(record.dropped(), vec![6, 3, 2]);
	assert_eq!(record.sum(), 8);
	// Each die finishes its chain before the next begins: the first `6` and
	// its extensions precede the extensions of the second, and each chain
	// stops at the limit, though its last die still matches.
	let mut record = record_of(kind(6), &[6, 1, 6]);
	assert_eq!(
		record.explode(&mut rng, &Criterion::Within(6..=6), Some(2)),
		4
	);
	assert_eq!(record.to_string(), "3d[6]: [6, 1, 6, 6, 6, 6, 6]");
	let links = record.dice.iter().map(|die| die.origin).collect::<Vec<_>>();
	assert_eq!(
		links,
		[
			Origin::Rolled,
			Origin::Rolled,
			Origin::Rolled,
			Origin::Extends(0),
			Origin::Extends(3),
			Origin::Extends(2),
			Origin::Extends(5)
		]
	);
	assert_eq!(record.sum(), 37);
	// Only the dice still kept explode.
	let mut record = record_of(kind(2), &[6, 6]);
	record.drop_lowest(1);
	assert_eq!(
		record.explode(&mut rng, &Criterion::Within(6..=6), Some(1)),
		1
	);
	assert_eq!(record.to_string(), "3d[2]: [~6, 6, 2]");
	assert_eq!(record.dice[2].origin, Origin::Extends(1));
	assert_eq!(record.sum(), 8);
}

/// Test that a chain grows only from its end: a later explode extends only
/// the die that nothing extends yet, and a replacement takes the place in its
/// chain of the die that it replaces, so that rerolling a die of a chain that
/// some die extends leaves its replacement extended, and rerolling the end of
/// a chain leaves its replacement free to grow. A later reroll acts on each
/// die of a chain alone.
#[test]
fn test_explode_grows_chains_from_their_ends()
{
	let mut rng = StdRng::seed_from_u64(0);
	let only = |face| RollingRecordKind::Custom {
		count: 1,
		faces: vec![face]
	};
	let six = Criterion::Within(6..=6);
	let one = Criterion::Within(1..=1);
	let mut record = record_of(only(6), &[6]);
	assert_eq!(record.explode(&mut rng, &six, Some(1)), 1);
	assert_eq!(record.explode(&mut rng, &six, Some(1)), 1);
	let links = record.dice.iter().map(|die| die.origin).collect::<Vec<_>>();
	assert_eq!(
		links,
		[Origin::Rolled, Origin::Extends(0), Origin::Extends(1)]
	);
	assert_eq!(record.chains(), vec![vec![0, 1, 2]]);
	assert_eq!(record.sum(), 18);
	// Reroll both dice of the chain `[6, 6]`, and then explode: the
	// replacement of the first takes its place, which the second extends, so
	// only the replacement of the second grows the chain.
	let mut record = record_of(only(1), &[]);
	record.dice = vec![die(6, Origin::Rolled), die(6, Origin::Extends(0))];
	assert_eq!(record.reroll(&mut rng, &six, Some(1)), 2);
	assert_eq!(record.to_string(), "1d[1]: [~6, ~6, 1, 1]");
	assert_eq!(record.explode(&mut rng, &one, Some(1)), 1);
	assert_eq!(record.dice[4].origin, Origin::Extends(3));
	assert_eq!(record.to_string(), "1d[1]: [~6, ~6, 1, 1, 1]");
	assert_eq!(record.chains(), vec![vec![0, 1, 2, 3, 4]]);
	assert_eq!(record.sum(), 3);
	// Reroll the end of the chain `[1, 6]`: its replacement ends the chain.
	let mut record = record_of(only(1), &[]);
	record.dice = vec![die(1, Origin::Rolled), die(6, Origin::Extends(0))];
	assert_eq!(record.reroll(&mut rng, &six, Some(1)), 1);
	assert_eq!(record.explode(&mut rng, &one, Some(1)), 1);
	assert_eq!(record.dice[3].origin, Origin::Extends(2));
	assert_eq!(record.to_string(), "1d[1]: [1, ~6, 1, 1]");
	assert_eq!(record.sum(), 3);
	// A later reroll rerolls the `1` of the chain `[6, 1]` alone.
	let mut record = record_of(only(5), &[]);
	record.dice = vec![die(6, Origin::Rolled), die(1, Origin::Extends(0))];
	assert_eq!(record.reroll(&mut rng, &one, Some(1)), 1);
	assert_eq!(record.to_string(), "1d[5]: [6, ~1, 5]");
	assert_eq!(record.dice[2].origin, Origin::Replaces(1));
	assert_eq!(record.chains(), vec![vec![0, 1, 2]]);
	assert_eq!(record.sum(), 11);
}

/// Test the degenerate explodes: a limit of zero or less explodes nothing; an
/// unlimited explode that matches every face rolls nothing, but appends to
/// every matching die a saturated die, worth `i32::MAX` or `i32::MIN` by the
/// sign of the mean face value, or `0` if that mean is `0`; a die without faces
/// and an empty range show only the face `0`; later rerolls and explodes leave
/// saturated chains alone, but act on the other chains; and a record without
/// dice is unchanged.
#[test]
fn test_explode_saturates()
{
	let mut rng = StdRng::seed_from_u64(0);
	let standard = |faces| RollingRecordKind::Standard { count: 3, faces };
	let custom = |faces: &[i32]| RollingRecordKind::Custom {
		count: 2,
		faces: faces.to_vec()
	};
	let everything = Criterion::Outside(RangeInclusive::new(1, 0));
	for limit in [0, -1, i32::MIN]
	{
		let mut record = record_of(standard(6), &[1, 2, 3]);
		assert_eq!(record.explode(&mut rng, &everything, Some(limit)), 0);
		assert_eq!(record.to_string(), "3D6: [1, 2, 3]");
	}
	for (kind, results, criterion, expected, sum) in [
		(
			standard(6),
			&[1, 6, 3][..],
			Criterion::Within(1..=6),
			"3D6: [1, 6, 3, ∞, ∞, ∞]",
			i32::MAX
		),
		(
			standard(1),
			&[1],
			Criterion::Within(1..=1),
			"3D1: [1, ∞]",
			i32::MAX
		),
		(
			custom(&[-3, 1]),
			&[1, -3],
			Criterion::Within(-3..=1),
			"2d[-3, 1]: [1, -3, -∞, -∞]",
			i32::MIN
		),
		(
			custom(&[-1, 1]),
			&[1, -1],
			Criterion::Outside(0..=0),
			"2d[-1, 1]: [1, -1, 0, 0]",
			0
		),
		(
			standard(0),
			&[0, 0],
			Criterion::Within(0..=0),
			"3D0: [0, 0, 0, 0]",
			0
		),
		(
			standard(-1),
			&[0],
			Criterion::Among(vec![0]),
			"3D-1: [0, 0]",
			0
		),
		(
			RollingRecordKind::Range { start: 4, end: 2 },
			&[0],
			Criterion::Within(0..=0),
			"[4:2]: [0, 0]",
			0
		),
		(
			RollingRecordKind::Range { start: -5, end: 2 },
			&[-1],
			Criterion::Within(-5..=2),
			"[-5:2]: [-1, -∞]",
			i32::MIN
		),
		(
			RollingRecordKind::Range { start: -2, end: 2 },
			&[2],
			Criterion::Within(-2..=2),
			"[-2:2]: [2, 0]",
			0
		),
		(
			RollingRecordKind::Range {
				start: i32::MIN,
				end: i32::MAX
			},
			&[7],
			Criterion::Within(i32::MIN..=i32::MAX),
			"[-2147483648:2147483647]: [7, -∞]",
			i32::MIN
		),
		// A die that shows no face of its die does not match, so only the
		// `6` saturates.
		(
			custom(&[6]),
			&[6, 3],
			Criterion::Within(6..=6),
			"2d[6]: [6, 3, ∞]",
			i32::MAX
		)
	]
	{
		let mut record = record_of(kind, results);
		assert_eq!(
			record.explode(&mut rng, &criterion, None),
			0,
			"{}",
			expected
		);
		assert_eq!(record.to_string(), expected);
		assert_eq!(record.sum(), sum, "{}", expected);
		let saturated = record
			.dice
			.iter()
			.filter(|die| matches!(die.origin, Origin::Saturates(_)))
			.count();
		assert_eq!(
			saturated,
			results.iter().filter(|&&r| criterion.matches(r)).count(),
			"{}",
			expected
		);
		// Later rerolls and explodes leave the saturated chains alone.
		if saturated == results.len()
		{
			let before = record.clone();
			assert_eq!(record.explode(&mut rng, &everything, Some(2)), 0);
			assert_eq!(record.explode(&mut rng, &everything, None), 0);
			assert_eq!(record.reroll(&mut rng, &everything, Some(2)), 0);
			assert_eq!(record.reroll(&mut rng, &everything, None), 0);
			assert_eq!(record, before, "{}", expected);
		}
	}
	// Later rerolls and explodes act on the chains that do not saturate, die by
	// die, and leave the saturated chains alone.
	let mut record = record_of(custom(&[6]), &[6, 3]);
	assert_eq!(record.explode(&mut rng, &Criterion::Within(6..=6), None), 0);
	assert_eq!(record.to_string(), "2d[6]: [6, 3, ∞]");
	assert_eq!(
		record.reroll(&mut rng, &Criterion::Within(3..=3), Some(1)),
		1
	);
	assert_eq!(record.to_string(), "2d[6]: [6, ~3, ∞, 6]");
	assert_eq!(
		record.explode(&mut rng, &Criterion::Within(6..=6), Some(1)),
		1
	);
	assert_eq!(record.to_string(), "2d[6]: [6, ~3, ∞, 6, 6]");
	assert_eq!(record.dice[4].origin, Origin::Extends(3));
	assert_eq!(record.reroll(&mut rng, &everything, None), 0);
	assert_eq!(record.to_string(), "2d[6]: [6, ~3, ∞, ~6, ~6]");
	assert_eq!(record.chains(), vec![vec![0, 2], vec![1, 3, 4]]);
	assert_eq!(record.sum(), i32::MAX);
	// No dice, and an uninitialized record, have nothing to explode.
	for kind in [standard(6), RollingRecordKind::Uninitialized]
	{
		let mut record = record_of(kind, &[]);
		assert_eq!(record.explode(&mut rng, &everything, None), 0);
		assert_eq!(record.explode(&mut rng, &everything, Some(3)), 0);
		assert!(record.dice.is_empty());
	}
}

/// Test, over random records, criteria, and limits, that an explode keeps its
/// invariants: it changes no die already rolled; every extension follows the
/// dice already rolled, links to the die that it extends, and shows a face of
/// the die; each chain begins at a die kept and matching when the explode
/// began, extends only dice that match, holds at most as many extensions as
/// the limit, and ends at a die that no longer matches or that exhausted the
/// limit; the chains follow one another in the order of the dice that begin
/// them; every die stays kept; the explode rolls one die per extension; and
/// an unlimited explode that matches every face rolls nothing, but saturates
/// every matching die, by the sign of the mean face value.
#[test]
fn test_explode_invariants()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(5019827350198273);
	for _ in 0..10_000
	{
		let (kind, faces): (_, Vec<i32>) = match rng.random_range(0..4)
		{
			0 =>
			{
				let faces = rng.random_range(-1..=6);
				(
					RollingRecordKind::Standard { count: 0, faces },
					match faces <= 0
					{
						true => vec![0],
						false => (1..=faces).collect()
					}
				)
			},
			1 =>
			{
				let faces = (0..rng.random_range(1..=4))
					.map(|_| rng.random_range(-3..=3))
					.collect::<Vec<_>>();
				(
					RollingRecordKind::Custom {
						count: 0,
						faces: faces.clone()
					},
					faces
				)
			},
			_ =>
			{
				let start = rng.random_range(-3..=3);
				let end = rng.random_range(-3..=3);
				(
					RollingRecordKind::Range { start, end },
					match end < start
					{
						true => vec![0],
						false => (start..=end).collect()
					}
				)
			}
		};
		let len = rng.random_range(0..=5);
		let results = (0..len)
			.map(|_| faces[rng.random_range(0..faces.len())])
			.collect::<Vec<_>>();
		let mut record = record_of(kind, &results);
		if rng.random_bool(0.5)
		{
			record.drop_lowest(rng.random_range(0..=2));
		}
		let criterion = match rng.random_range(0..3)
		{
			0 =>
			{
				let start = rng.random_range(-4..=4);
				Criterion::within(start, rng.random_range(-4..=4))
			},
			1 =>
			{
				let start = rng.random_range(-4..=4);
				Criterion::Outside(start..=rng.random_range(-4..=4))
			},
			_ =>
			{
				let mut values = (0..rng.random_range(0..=4))
					.map(|_| rng.random_range(-4..=4))
					.collect::<Vec<_>>();
				values.sort();
				values.dedup();
				Criterion::Among(values)
			}
		};
		let limit = match rng.random_bool(0.3)
		{
			true => None,
			false => Some(rng.random_range(-1..=3))
		};
		let before = record.clone();
		let rolled = record.explode(&mut rng, &criterion, limit);
		let context = format!(
			"{} explode {:?} {:?} -> {}",
			before, criterion, limit, record
		);
		let len = before.dice.len();
		assert_eq!(&record.dice[..len], &before.dice[..], "{}", context);
		let starts = before
			.dice
			.iter()
			.enumerate()
			.filter(|(_, die)| die.is_kept() && criterion.matches(die.value))
			.map(|(index, _)| index)
			.collect::<Vec<_>>();
		if limit.is_none() && faces.iter().all(|f| criterion.matches(*f))
		{
			// Every face matches, so every matching die saturates.
			let saturated = match faces.iter().sum::<i32>().signum()
			{
				1 => i32::MAX,
				-1 => i32::MIN,
				_ => 0
			};
			let expected = starts
				.iter()
				.map(|&index| Die::saturating(saturated, index))
				.collect::<Vec<_>>();
			assert_eq!(rolled, 0, "{}", context);
			assert_eq!(&record.dice[len..], &expected[..], "{}", context);
			continue
		}
		assert_eq!(record.dice.len() - len, rolled as usize, "{}", context);
		assert!(record.dice[len..].iter().all(Die::is_kept), "{}", context);
		// Follow each chain from the die that begins it.
		let mut next = len;
		for start in starts
		{
			let mut current = start;
			let mut times = 0;
			while next < record.dice.len()
				&& record.dice[next].origin == Origin::Extends(current)
			{
				assert!(
					criterion.matches(record.dice[current].value),
					"{}",
					context
				);
				assert!(
					faces.contains(&record.dice[next].value),
					"{}",
					context
				);
				current = next;
				next += 1;
				times += 1;
			}
			let end = record.dice[current];
			match limit
			{
				Some(limit) =>
				{
					assert!(times <= limit.max(0) as usize, "{}", context);
					assert!(
						!criterion.matches(end.value)
							|| times == limit.max(0) as usize,
						"{}",
						context
					);
				},
				None =>
				{
					assert!(!criterion.matches(end.value), "{}", context);
				}
			}
		}
		assert_eq!(next, record.dice.len(), "{}", context);
		// Without saturation, the sum of the chains is the sum of the kept
		// dice, which is far too small to overflow.
		assert_eq!(
			record.sum(),
			record.kept().iter().sum::<i32>(),
			"{}",
			context
		);
	}
}

/// Answer an evaluator for a function that rolls the specified dice, explodes
/// them by the specified clause, applies the specified further clauses, and
/// sums the record.
///
/// # Parameters
/// - `roll`: The roll, as the assembler reads it after `roll`.
/// - `clause`: The predicate and limit of the explode, as the assembler reads
///   them.
/// - `clauses`: The further clauses, each as the assembler reads it before
///   `from`.
///
/// # Returns
/// The evaluator.
fn explode_evaluator(roll: &str, clause: &str, clauses: &[&str]) -> Evaluator
{
	let clauses = clauses
		.iter()
		.map(|clause| format!("\t\t⚅0 <- {} from ⚅0\n", clause))
		.collect::<String>();
	Evaluator::new(
		Assembler::assemble(&format!(
			"Function({{t}}@0) r#2 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll \
			 {}\n\t\t⚅0 <- explode {} from ⚅0\n{}\t\t@1 <- sum rolling \
			 record ⚅0\n\t\treturn @1\n",
			roll, clause, clauses
		))
		.unwrap()
	)
}

/// Test that evaluation explodes as the instruction directs: it reads the
/// threshold and limit from their registers, sums the chains, counts each
/// extension as a die, and stays within the bounds, which count the outcomes
/// of a limited explode as though every die rolled its extensions in advance;
/// that an unlimited explode saturates, whether its predicate matches
/// every face statically or only for the threshold given at runtime, toward
/// the sign of the mean face value; and that drops and keeps after an explode
/// cross out whole chains, and an explode after a keep explodes only the dice
/// still kept.
#[test]
fn test_evaluation_explodes()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(7120398571203985);
	for (clause, t, saturates, count) in [
		("= @0 up to 2 times", 6, false, Some(6u128.pow(4 + 4 * 2))),
		("> @0 unlimited times", 4, false, None),
		("in [5, 6] up to @0 times", 1, false, Some(6u128.pow(4 + 4))),
		("!= @0 up to 1 times", 3, false, Some(6u128.pow(4 + 4))),
		("in @0..=6 unlimited times", 1, true, Some(6u128.pow(4))),
		("<= @0 unlimited times", 6, true, Some(6u128.pow(4))),
		(">= 1 unlimited times", 0, true, Some(6u128.pow(4)))
	]
	{
		let mut evaluator = explode_evaluator("standard dice 4D6", clause, &[]);
		let bounds = evaluator.bounds_over([Some(t.into())], []).unwrap();
		assert_eq!(bounds.count, count, "{}", clause);
		for _ in 0..200
		{
			let evaluation = evaluator.evaluate([t], &mut rng).unwrap();
			let [record] = &evaluation.records[..]
			else
			{
				panic!("expected one record: {:?}", evaluation.records)
			};
			assert_eq!(record.chains().len(), 4, "{}: {}", clause, record);
			assert!(record.dice.iter().all(Die::is_kept), "{}", record);
			assert_eq!(evaluation.result, record.clone().sum(), "{}", record);
			assert!(bounds.value.contains(evaluation.result), "{}", record);
			assert!(evaluation.dice <= bounds.dice, "{}", record);
			match saturates
			{
				true =>
				{
					assert_eq!(evaluation.result, i32::MAX, "{}", record);
					assert_eq!(evaluation.dice, 4, "{}", record);
					assert!(record.to_string().ends_with("∞, ∞, ∞, ∞]"));
				},
				false =>
				{
					assert_eq!(
						evaluation.dice,
						record.dice.len() as u64,
						"{}",
						record
					);
					assert_eq!(
						evaluation.result,
						record.kept().iter().sum::<i32>(),
						"{}",
						record
					);
				}
			}
			if clause == "> @0 unlimited times"
			{
				// Every chain ends at a die that shows at most the threshold.
				for chain in record.chains()
				{
					let end = record.dice[*chain.last().unwrap()];
					assert!(end.value <= t, "{}", record);
				}
			}
		}
	}
	// An explode that cannot match leaves the bounds as they were.
	let evaluator =
		explode_evaluator("standard dice 4D6", "= 7 unlimited times", &[]);
	let bounds = evaluator.bounds_over([Some(0.into())], []).unwrap();
	assert_eq!(bounds.value, (4, 24).into());
	assert_eq!(bounds.count, Some(6u128.pow(4)));
	assert_eq!(bounds.dice, 4);
	// An unlimited explode of a die whose faces have a negative mean
	// saturates to `i32::MIN`, and one whose faces have a zero mean to `0`.
	for (roll, result, rendered) in [
		("custom dice 2D[-3, 1]", i32::MIN, "-∞"),
		("custom dice 2D[-1, 1]", 0, "0")
	]
	{
		let mut evaluator =
			explode_evaluator(roll, "<= @0 unlimited times", &[]);
		let evaluation = evaluator.evaluate([1], &mut rng).unwrap();
		assert_eq!(evaluation.result, result, "{}", roll);
		assert_eq!(evaluation.dice, 2, "{}", roll);
		assert!(
			evaluation.records[0]
				.to_string()
				.ends_with(&format!(", {rendered}, {rendered}]")),
			"{}",
			evaluation.records[0]
		);
	}
	// Drops and keeps after an explode cross out whole chains: of three
	// chains of three `6`s, dropping the lowest crosses out the first, and
	// keeping the highest keeps the last; whereas an explode after a keep
	// explodes only the dice still kept.
	for (clauses, result, dice, rendered) in [
		(
			&["drop lowest 1"][..],
			36,
			9,
			"3d[6]: [~6, 6, 6, ~6, ~6, 6, 6, 6, 6]"
		),
		(
			&["keep highest 1"],
			18,
			9,
			"3d[6]: [~6, ~6, 6, ~6, ~6, ~6, ~6, 6, 6]"
		),
		(
			&["keep highest 2", "explode = 6 up to 1 times"],
			48,
			11,
			"3d[6]: [~6, 6, 6, ~6, ~6, 6, 6, 6, 6, 6, 6]"
		)
	]
	{
		let function = explode_evaluator(
			"custom dice 3D[6]",
			"= @0 up to 2 times",
			clauses
		)
		.function;
		for passes in [Passes::none(), Passes::all()]
		{
			let mut evaluator =
				Evaluator::new(optimize(function.clone(), passes));
			let evaluation = evaluator.evaluate([6], &mut rng).unwrap();
			assert_eq!(evaluation.result, result, "{:?}", clauses);
			assert_eq!(evaluation.dice, dice, "{:?}", clauses);
			assert_eq!(
				evaluation.records[0].to_string(),
				rendered,
				"{:?}",
				clauses
			);
		}
	}
}

/// Test that a metered evaluation charges each extension as a die, so that a
/// budget bounds even an unlimited explode, and refuses the first extension
/// that would exceed it; that metering within a sufficient budget does not
/// disturb the draws; and that a saturated chain rolls nothing, so it costs
/// nothing.
#[test]
fn test_dice_budget_meters_explodes()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let seed = 9182375019283750;
	let mut evaluator =
		explode_evaluator("standard dice 4D6", ">= @0 unlimited times", &[]);
	let mut extended = 0;
	for i in 0..100
	{
		let unmetered = evaluator
			.evaluate([4], &mut StdRng::seed_from_u64(seed + i))
			.unwrap();
		let dice = unmetered.dice;
		assert!(dice >= 4);
		let metered = evaluator
			.evaluate_metered([4], &mut StdRng::seed_from_u64(seed + i), dice)
			.unwrap();
		assert_eq!(metered, unmetered);
		let mut rng = StdRng::seed_from_u64(seed + i);
		match dice
		{
			// No die exploded, so one die fewer refuses the roll itself.
			4 => assert_eq!(
				evaluator.evaluate_metered([4], &mut rng, dice - 1),
				Err(EvaluationError::DiceBudgetExhausted {
					requested: 4,
					remaining: 3,
					consumed: 0
				})
			),
			_ =>
			{
				extended += 1;
				assert_eq!(
					evaluator.evaluate_metered([4], &mut rng, dice - 1),
					Err(EvaluationError::DiceBudgetExhausted {
						requested: 1,
						remaining: 0,
						consumed: dice - 1
					})
				)
			}
		}
	}
	assert!(extended > 0);
	// An explode that matches every face saturates, rolling nothing.
	let evaluation = evaluator
		.evaluate_metered([1], &mut StdRng::seed_from_u64(seed), 4)
		.unwrap();
	assert_eq!(evaluation.result, i32::MAX);
	assert_eq!(evaluation.dice, 4);
}

/// Test that the bounds of an explode are exact: a limited explode bounds each
/// chain by its least and greatest totals, and counts the outcomes as though
/// every chain still kept rolled its extensions in advance; an unlimited one
/// that may grow a chain without bound bounds its total only by `i32`, toward
/// the sign of the faces that may extend it, and counts nothing, and one that
/// matches every face saturates every chain, rolling nothing; an explode that
/// cannot extend a die changes nothing; drops and keeps after an explode count
/// whole chains; and a later reroll or explode of an exploded record that may
/// match is sound but loose, though a later limited explode still rolls at
/// most its limit for each chain still kept.
#[test]
fn test_explode_bounds()
{
	let unconstrained = EvaluationBounds::unconstrained();
	for (clauses, value, count, dice) in [
		// The chains total 1 to 5, 7 to 11, or 13 to 18.
		(
			&["explode = 6 up to 2 times"][..],
			(4, 72).into(),
			Some(6u128.pow(4 + 4 * 2)),
			4 + 4 * 2
		),
		(
			&["explode = 6 up to 0 times"],
			(4, 24).into(),
			Some(6u128.pow(4)),
			4
		),
		(
			&["explode = 7 up to 2 times"],
			(4, 24).into(),
			Some(6u128.pow(4)),
			4
		),
		(
			&["explode = 6 unlimited times"],
			(4, i32::MAX).into(),
			None,
			u64::MAX
		),
		(
			&["explode >= 1 unlimited times"],
			i32::MAX.into(),
			Some(6u128.pow(4)),
			4
		),
		(
			&["keep highest 2", "explode = 6 up to 3 times"],
			(2, 48).into(),
			Some(6u128.pow(4 + 2 * 3)),
			4 + 2 * 3
		),
		(
			&["explode = 6 up to 1 times", "keep highest 0"],
			0.into(),
			Some(6u128.pow(4 + 4)),
			4 + 4
		),
		(
			&["explode = 6 up to 1 times", "drop lowest 3"],
			(1, 12).into(),
			Some(6u128.pow(4 + 4)),
			4 + 4
		),
		// A reroll that cannot match leaves the chains as they were.
		(
			&["explode = 6 up to 1 times", "reroll = 7 up to 1 times"],
			(4, 48).into(),
			Some(6u128.pow(4 + 4)),
			4 + 4
		),
		(
			&["explode = 6 up to 1 times", "explode = 6 up to 1 times"],
			unconstrained,
			None,
			4 + 4 + 4
		),
		(
			&["explode = 6 up to 1 times", "reroll = 1 up to 1 times"],
			unconstrained,
			None,
			u64::MAX
		)
	]
	{
		let clauses = clauses
			.iter()
			.map(|clause| format!("⚅0 <- {} from ⚅0\n", clause))
			.collect::<String>();
		let bounds = body_bounds(&format!(
			"⚅0 <- roll standard dice 4D6\n{}@0 <- sum rolling record \
			 ⚅0\nreturn @0",
			clauses
		));
		assert_eq!(bounds.value, value, "{}", clauses);
		assert_eq!(bounds.count, count, "{}", clauses);
		assert_eq!(bounds.dice, dice, "{}", clauses);
	}
}

/// The greatest number of extensions to which [`chain_hull`] grows a chain
/// without limit.
const CHAIN_DEPTH: i32 = 4;

/// Answer the hull of the totals of every chain that an explode may grow from
/// a die still kept, by enumerating the chains. Without limit, enumerate only
/// the chains that end within `depth` extensions.
///
/// # Parameters
/// - `starts`: The results that a die still kept may show.
/// - `faces`: The faces of the die, as it rolls them.
/// - `criterion`: The criterion of the explode.
/// - `limit`: The limit, or `None` for no limit.
/// - `depth`: The greatest number of extensions to enumerate without limit.
///
/// # Returns
/// The least and greatest totals.
fn chain_hull(
	starts: &[i32],
	faces: &[i32],
	criterion: &Criterion,
	limit: Option<i32>,
	depth: i32
) -> (i32, i32)
{
	/// Grow the chain by every face, until it ends, adding the totals of the
	/// chains that end.
	fn grow(
		chain: &mut Vec<i32>,
		faces: &[i32],
		criterion: &Criterion,
		remaining: i32,
		limited: bool,
		hull: &mut (i32, i32)
	)
	{
		let matches = criterion.matches(*chain.last().unwrap());
		if !matches || remaining == 0
		{
			// A chain without limit that still matches has not ended.
			if limited || !matches
			{
				let mut dice = chain.clone();
				dice.sort_unstable();
				let total = dice.into_iter().fold(0, i32::saturating_add);
				*hull = (hull.0.min(total), hull.1.max(total));
			}
			return
		}
		for face in faces
		{
			chain.push(*face);
			grow(chain, faces, criterion, remaining - 1, limited, hull);
			chain.pop();
		}
	}
	let mut hull = (i32::MAX, i32::MIN);
	if limit.is_none() && faces.iter().all(|face| criterion.matches(*face))
	{
		// No chain ends, so every chain saturates by the sign of the mean.
		let saturated = match faces.iter().map(|f| *f as i64).sum::<i64>()
		{
			sum if sum > 0 => i32::MAX,
			sum if sum < 0 => i32::MIN,
			_ => 0
		};
		for _ in starts
		{
			hull = (saturated, saturated);
		}
		return hull
	}
	let (remaining, limited) = match limit
	{
		Some(limit) => (limit.max(0), true),
		None => (depth, false)
	};
	for start in starts
	{
		grow(
			&mut vec![*start],
			faces,
			criterion,
			remaining,
			limited,
			&mut hull
		);
	}
	hull
}

/// Answer the faces of a die of the exhaustive explode bounds grid and the
/// number of dice that it rolls, given the values of its parameters, as the
/// evaluator rolls them: those of the [reroll bounds grid](grid_faces),
/// followed by `1D[1, 4]` and `1D[-4, -1]`.
///
/// # Parameters
/// - `die`: The index of the die among those of the grid.
/// - `c`: The value of `@2`.
/// - `d`: The value of `@3`.
///
/// # Returns
/// The faces and the number of dice.
fn explode_grid_faces(die: usize, c: i32, d: i32) -> (Vec<i32>, i32)
{
	match die
	{
		3 => (vec![1, 4], 1),
		4 => (vec![-4, -1], 1),
		die => grid_faces(die, c, d)
	}
}

/// The memo of [`chain_hull`], by its arguments.
type ChainMemo =
	HashMap<(Vec<i32>, Vec<i32>, Criterion, Option<i32>, i32), (i32, i32)>;

/// Answer the brute-force extrema of the sum of a record of the exhaustive
/// explode bounds grid, over every assignment of the operands, by enumerating
/// the chains of each.
///
/// # Parameters
/// - `die`: The index of the die among those of the grid.
/// - `prelude`: The criterion of an unlimited reroll before the explode, if
///   any.
/// - `predicate`: The predicate of the explode, whose operands are `@0` and
///   `@1`.
/// - `limited`: Whether the limit of the explode is `@4`, rather than none.
/// - `operands`: The intervals of `@0` through `@4`.
/// - `depth`: The greatest number of extensions to enumerate without limit.
/// - `memo`: The memo of [`chain_hull`].
///
/// # Returns
/// The least and greatest sums.
fn explode_grid_truth(
	die: usize,
	prelude: Option<&Criterion>,
	predicate: &Predicate,
	limited: bool,
	operands: [EvaluationBounds; 5],
	depth: i32,
	memo: &mut ChainMemo
) -> (i32, i32)
{
	let a = AddressingMode::from(RegisterIndex(0));
	let mut truth = (i32::MAX, i32::MIN);
	let [i0, i1, i2, i3, i4] = operands;
	for v0 in i0.min..=i0.max
	{
		for v1 in i1.min..=i1.max
		{
			let criterion = predicate.resolve(|op| match op == a
			{
				true => v0,
				false => v1
			});
			for v2 in i2.min..=i2.max
			{
				for v3 in i3.min..=i3.max
				{
					let (faces, count) = explode_grid_faces(die, v2, v3);
					// An unlimited reroll leaves each die showing a face that
					// it does not match, or crosses out every die.
					let starts = faces
						.iter()
						.copied()
						.filter(|face| {
							prelude
								.is_none_or(|prelude| !prelude.matches(*face))
						})
						.collect::<Vec<_>>();
					for v4 in i4.min..=i4.max
					{
						let limit = limited.then_some(v4);
						let key = (
							starts.clone(),
							faces.clone(),
							criterion.clone(),
							limit,
							depth
						);
						let (min, max) =
							*memo.entry(key).or_insert_with(|| {
								chain_hull(
									&starts, &faces, &criterion, limit, depth
								)
							});
						let sum = match starts.is_empty()
						{
							true => (0, 0),
							false => (
								min.saturating_mul(count),
								max.saturating_mul(count)
							)
						};
						truth = (truth.0.min(sum.0), truth.1.max(sum.1));
					}
				}
			}
		}
	}
	truth
}

/// Test that the bounds of the sum of an exploded record are exact over an
/// exhaustive grid of dice, unlimited rerolls before the explode, predicates,
/// limits, and intervals of their operands, against the brute-force extrema of
/// every assignment of their operands, found by enumerating the chains of
/// each. Without limit, the enumeration ends at a fixed depth, so the bounds
/// must agree with it where its extrema have settled, and saturate where they
/// still grow.
#[test]
fn test_explode_bounds_exhaustive()
{
	let dice = [
		("roll standard dice 2D@2", 1),
		("roll range @2:@3", 2),
		("roll custom dice 2D[-1, 1, 1, 3]", 0),
		("roll custom dice 1D[1, 4]", 0),
		("roll custom dice 1D[-4, -1]", 0)
	];
	let preludes = [
		None,
		Some((">= 2", Criterion::Within(2..=i32::MAX))),
		Some(("<= 1", Criterion::Within(i32::MIN..=1)))
	];
	let (a, b) = (
		AddressingMode::from(RegisterIndex(0)),
		AddressingMode::from(RegisterIndex(1))
	);
	let predicates = [
		Predicate::Eq(a),
		Predicate::Ne(a),
		Predicate::Lt(a),
		Predicate::Le(a),
		Predicate::Gt(a),
		Predicate::Ge(a),
		Predicate::In(vec![]),
		Predicate::In(vec![3, -1, 1]),
		Predicate::In(vec![0, 2, 3]),
		Predicate::InRange { start: a, end: b },
		Predicate::InRangeInclusive { start: a, end: b }
	];
	// The limits, as the explode reads them: the limit in `@4`, over each of
	// these intervals, or no limit.
	let singletons = (0..=3).map(EvaluationBounds::from).collect::<Vec<_>>();
	let limits = [Some(singletons), Some(intervals(-1..=2)), None];
	let fixed = vec![EvaluationBounds::from(0)];
	let mut memo = ChainMemo::new();
	for (die, (roll, die_params)) in dice.iter().enumerate()
	{
		for prelude in &preludes
		{
			// Only fixed dice follow a reroll, since the bounds of a reroll
			// unite the results of every die.
			if prelude.is_some() && *die_params > 0
			{
				continue
			}
			for predicate in &predicates
			{
				for limits in &limits
				{
					// Only fixed dice, and predicates of at most one operand,
					// range over the intervals of the limit, to keep the grid
					// small.
					if matches!(limits, Some(limits) if limits.len() > 4)
						&& (*die_params > 0 || predicate.operands().len() > 1)
					{
						continue
					}
					let limited = limits.is_some();
					let clause = format!(
						"explode {} {}",
						predicate,
						match limited
						{
							true => "up to @4 times",
							false => "unlimited times"
						}
					);
					let reroll = match prelude
					{
						Some((prelude, _)) => format!(
							"\t\t⚅0 <- reroll {} unlimited times from ⚅0\n",
							prelude
						),
						None => String::new()
					};
					let function = Assembler::assemble(&format!(
						"Function({{a}}@0, {{b}}@1, {{c}}@2, {{d}}@3, {{e}}@4) \
						 r#6 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- {}\n{}\t\t\
						 ⚅0 <- {} from ⚅0\n\t\t@5 <- sum rolling record \
						 ⚅0\n\t\treturn @5\n",
						roll, reroll, clause
					))
					.unwrap();
					let evaluator = Evaluator::new(function);
					let pred_intervals = match predicate.operands().len()
					{
						0 => [fixed.clone(), fixed.clone()],
						1 => [intervals(-1..=3), fixed.clone()],
						_ => [intervals(0..=2), intervals(0..=2)]
					};
					let die_intervals = match die_params
					{
						0 => [fixed.clone(), fixed.clone()],
						1 => [intervals(-1..=3), fixed.clone()],
						_ => [intervals(-1..=1), intervals(-1..=1)]
					};
					let limit_intervals =
						limits.clone().unwrap_or(fixed.clone());
					for i0 in &pred_intervals[0]
					{
						for i1 in &pred_intervals[1]
						{
							for i2 in &die_intervals[0]
							{
								for i3 in &die_intervals[1]
								{
									for i4 in &limit_intervals
									{
										let operands =
											[*i0, *i1, *i2, *i3, *i4];
										let bounds = evaluator
											.bounds_over(operands.map(Some), [])
											.unwrap();
										let [deep, shallow] =
											[CHAIN_DEPTH, CHAIN_DEPTH - 1].map(
												|depth| {
													explode_grid_truth(
														die,
														prelude
															.as_ref()
															.map(|(_, c)| c),
														predicate,
														limited,
														operands,
														depth,
														&mut memo
													)
												}
											);
										let context = format!(
											"{} {}{} over {:?}",
											roll, reroll, clause, operands
										);
										// A limited explode, or one whose
										// extrema have settled, is exact; one
										// whose extrema still grow saturates.
										match deep.0 == shallow.0
										{
											true => assert_eq!(
												bounds.value.min, deep.0,
												"{}",
												context
											),
											false => assert!(
												bounds.value.min
													<= i32::MIN + 4,
												"{}: {}",
												context,
												bounds.value
											)
										}
										match deep.1 == shallow.1
										{
											true => assert_eq!(
												bounds.value.max, deep.1,
												"{}",
												context
											),
											false => assert_eq!(
												bounds.value.max,
												i32::MAX,
												"{}",
												context
											)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/// Test that the bounds of an unlimited explode are exact where a chain may
/// grow without bound: a chain that may extend by a positive face may total
/// `i32::MAX`, and one that may extend by a negative face saturates at
/// `i32::MIN`, before the folding of its positive dice lifts it again; that a
/// chain whose every face matches saturates by the sign of the mean face
/// value; that a die that may or may not saturate bounds both ways; and that
/// too many criteria to enumerate leave the bounds sound but loose.
#[test]
fn test_explode_bounds_unlimited()
{
	for (body, value) in [
		// A chain of `-1`s ends at a `1` or a `3`, which lifts the saturated
		// total.
		(
			"⚅0 <- roll custom dice 1D[-1, 1, 1, 3]
			⚅0 <- explode = -1 unlimited times from ⚅0",
			(i32::MIN + 1, 3)
		),
		(
			"⚅0 <- roll custom dice 1D[-2000000000, 5]
			⚅0 <- explode = -2000000000 unlimited times from ⚅0",
			(i32::MIN + 5, 5)
		),
		(
			"⚅0 <- roll custom dice 1D[-1, 3]
			⚅0 <- explode = 3 unlimited times from ⚅0",
			(-1, i32::MAX)
		),
		// Two chains that total nearly `i32::MIN` sum to `i32::MIN`.
		(
			"⚅0 <- roll custom dice 2D[-1, 1, 1, 3]
			⚅0 <- explode = -1 unlimited times from ⚅0",
			(i32::MIN, 6)
		),
		(
			"⚅0 <- roll custom dice 2D[-1, 1]
			⚅0 <- explode >= -1 unlimited times from ⚅0",
			(0, 0)
		),
		(
			"⚅0 <- roll custom dice 2D[-3, 1]
			⚅0 <- explode <= 1 unlimited times from ⚅0",
			(i32::MIN, i32::MIN)
		),
		// A die of one face saturates, but one of two or three ends at a `2`
		// or a `3`.
		(
			"⚅1 <- roll standard dice 1D3
			@1 <- sum rolling record ⚅1
			⚅0 <- roll standard dice 1D@1
			⚅0 <- explode <= 1 unlimited times from ⚅0",
			(2, i32::MAX)
		),
		// An explode after an unlimited reroll extends a `1`, which no face
		// matches, by any face.
		(
			"⚅0 <- roll standard dice 1D6
			⚅0 <- reroll >= 2 unlimited times from ⚅0
			⚅0 <- explode = 1 up to 3 times from ⚅0",
			(3, 9)
		)
	]
	{
		let bounds = body_bounds(&format!(
			"{}\n@0 <- sum rolling record ⚅0\nreturn @0",
			body
		));
		assert_eq!(bounds.value, value.into(), "{}", body);
	}
	// Too many thresholds to enumerate unite the faces that every threshold
	// matches, which forgets that every middle die of a chain shows the same
	// threshold, so the least total is `i32::MIN` rather than `i32::MIN + 1`.
	let evaluator = Evaluator::new(
		Assembler::assemble(
			"Function({t}@0) r#2 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll \
			 custom dice 1D[-1, 1, 1, 3]\n\t\t⚅0 <- explode = @0 unlimited \
			 times from ⚅0\n\t\t@1 <- sum rolling record ⚅0\n\t\treturn @1\n"
		)
		.unwrap()
	);
	for (t, value) in [
		((-1, 1000), (i32::MIN + 1, i32::MAX)),
		((-1, 5000), (i32::MIN, i32::MAX))
	]
	{
		assert_eq!(
			evaluator.bounds_over([Some(t.into())], []).unwrap().value,
			value.into(),
			"{:?}",
			t
		);
	}
}

/// Test that the bounds count every extension that each chain still kept might
/// need, as though it were rolled in advance, as for a limited reroll: a
/// limited explode multiplies the outcomes by the faces of the die, raised to
/// the limit, for every chain still kept, even if no die still kept may match;
/// that an explode that cannot happen, or that saturates, counts nothing; and
/// that an unlimited explode that may roll, an explode by a random operand or
/// of a random number of chains, or a later explode or reroll of an exploded
/// record, counts nothing at all.
#[test]
fn test_explode_bounds_count()
{
	let count = |body: &str| {
		body_bounds(&format!(
			"{}\n@0 <- sum rolling record ⚅0\nreturn @0",
			body
		))
		.count
	};
	for (roll, clause, expected) in [
		(
			"standard dice 2D6",
			"= 6 up to 2 times",
			Some(6u128.pow(2 + 2 * 2))
		),
		("standard dice 2D6", "= 6 unlimited times", None),
		// Every face matches, so every chain saturates, rolling nothing.
		("standard dice 2D6", ">= 1 unlimited times", Some(36)),
		// Every face matches, but the limit rolls each die again regardless.
		("standard dice 2D6", ">= 1 up to 1 times", Some(36 * 36)),
		// No face matches.
		("standard dice 2D6", "= 7 up to 3 times", Some(36)),
		("standard dice 2D6", "in 4..4 unlimited times", Some(36)),
		// The limit is not positive.
		("standard dice 2D6", "= 6 up to 0 times", Some(36)),
		("standard dice 2D6", "= 6 up to -1 times", Some(36)),
		// A face of a custom die counts once for every time that it is listed.
		("custom dice 2D[1, 1, 2]", "= 1 up to 1 times", Some(9 * 9)),
		("custom dice 2D[1, 1, 2]", "= 1 unlimited times", None),
		// A die without faces, or an empty range, shows only `0`.
		("standard dice 2D0", "= 0 up to 1 times", Some(1)),
		("standard dice 2D0", "= 0 unlimited times", Some(1)),
		("range 3:1", "= 0 up to 2 times", Some(1)),
		("range 1:3", "!= 2 up to 1 times", Some(9))
	]
	{
		assert_eq!(
			count(&format!(
				"⚅0 <- roll {}\n⚅0 <- explode {} from ⚅0",
				roll, clause
			)),
			expected,
			"{} explode {}",
			roll,
			clause
		);
	}
	for (body, expected) in [
		// An explode counts the chains still kept.
		(
			"⚅0 <- roll standard dice 3D6
			⚅0 <- keep highest 1 from ⚅0
			⚅0 <- explode = 6 up to 1 times from ⚅0",
			Some(6u128.pow(3 + 1))
		),
		// No die still kept shows a `6`, but a `6` still counts.
		(
			"⚅0 <- roll standard dice 2D6
			⚅0 <- reroll = 6 unlimited times from ⚅0
			⚅0 <- explode = 6 up to 1 times from ⚅0",
			Some(36 * 25 * 36)
		),
		// A drop or keep after an explode counts nothing more.
		(
			"⚅0 <- roll standard dice 2D6
			⚅0 <- explode = 6 up to 1 times from ⚅0
			⚅0 <- drop lowest 1 from ⚅0",
			Some(36 * 36)
		),
		(
			"⚅0 <- roll standard dice 2D6
			⚅0 <- explode = 6 up to 1 times from ⚅0
			⚅0 <- explode = 6 up to 1 times from ⚅0",
			None
		),
		(
			"⚅0 <- roll standard dice 2D6
			⚅0 <- explode = 6 up to 1 times from ⚅0
			⚅0 <- reroll = 1 up to 1 times from ⚅0",
			None
		),
		// A later explode or reroll that cannot match counts as it would
		// before.
		(
			"⚅0 <- roll standard dice 2D6
			⚅0 <- explode = 6 up to 1 times from ⚅0
			⚅0 <- explode = 7 up to 1 times from ⚅0",
			Some(36 * 36)
		)
	]
	{
		assert_eq!(count(body), expected, "{}", body);
	}
	// A random threshold, endpoint, or limit counts nothing.
	for clause in [
		"< @0 up to 1 times",
		"in 1..@0 up to 1 times",
		"= 6 up to @0 times"
	]
	{
		assert_eq!(
			count(&format!(
				"⚅1 <- roll standard dice 1D3
				@0 <- sum rolling record ⚅1
				⚅0 <- roll standard dice 2D6
				⚅0 <- explode {} from ⚅0",
				clause
			)),
			None,
			"{}",
			clause
		);
	}
	// A random number of chains still kept counts nothing, unless the explode
	// cannot happen, whence it counts the same whatever the number.
	for (clause, expected) in [
		("= 6 up to 1 times", None),
		("= 7 up to 1 times", Some(3 * 6u128.pow(4)))
	]
	{
		assert_eq!(
			count(&format!(
				"⚅1 <- roll standard dice 1D3
				@0 <- sum rolling record ⚅1
				⚅0 <- roll standard dice 4D6
				⚅0 <- drop lowest @0 from ⚅0
				⚅0 <- explode {} from ⚅0",
				clause
			)),
			expected,
			"{}",
			clause
		);
	}
}

/// Test that the bounds of random functions that explode dice contain every
/// sampled evaluation: each rolls standard dice, custom dice, or a range, and
/// then drops, keeps, rerolls, and explodes them in any order, by predicates
/// whose operands are constants or an argument, whose bounds contain the
/// argument; and that no evaluation rolls more dice than the bounds allow.
#[test]
fn test_explode_bounds_sampled()
{
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(8102938471029384);
	let t = AddressingMode::from(RegisterIndex(0));
	for _ in 0..2_000
	{
		let roll = match rng.random_range(0..3)
		{
			0 => format!(
				"standard dice {}D{}",
				rng.random_range(0..=3),
				rng.random_range(-1..=6)
			),
			1 => format!(
				"custom dice {}D[{}]",
				rng.random_range(0..=3),
				(0..rng.random_range(1..=4))
					.map(|_| rng.random_range(-3..=3).to_string())
					.collect::<Vec<_>>()
					.join(", ")
			),
			_ => format!(
				"range {}:{}",
				rng.random_range(-3..=3),
				rng.random_range(-3..=4)
			)
		};
		let operand = |rng: &mut StdRng| match rng.random_bool(0.3)
		{
			true => t,
			false => AddressingMode::from(Immediate(rng.random_range(-3..=7)))
		};
		let clauses = (0..rng.random_range(1..=3))
			.map(|_| {
				let predicate = match rng.random_range(0..9)
				{
					0 => Predicate::Eq(operand(&mut rng)),
					1 => Predicate::Ne(operand(&mut rng)),
					2 => Predicate::Lt(operand(&mut rng)),
					3 => Predicate::Le(operand(&mut rng)),
					4 => Predicate::Gt(operand(&mut rng)),
					5 => Predicate::Ge(operand(&mut rng)),
					6 => Predicate::In(
						(0..rng.random_range(0..=3))
							.map(|_| rng.random_range(-3..=6))
							.collect()
					),
					7 => Predicate::InRange {
						start: operand(&mut rng),
						end: operand(&mut rng)
					},
					_ => Predicate::InRangeInclusive {
						start: operand(&mut rng),
						end: operand(&mut rng)
					}
				};
				let limit = match rng.random_bool(0.3)
				{
					true => "unlimited times".to_string(),
					false => format!("up to {} times", rng.random_range(-1..=3))
				};
				match rng.random_range(0..6)
				{
					0 => format!("drop lowest {}", rng.random_range(-1..=2)),
					1 => format!("keep highest {}", rng.random_range(-1..=3)),
					2 => format!("reroll {} {}", predicate, limit),
					_ => format!("explode {} {}", predicate, limit)
				}
			})
			.map(|clause| format!("\t\t⚅0 <- {} from ⚅0\n", clause))
			.collect::<String>();
		let function = Assembler::assemble(&format!(
			"Function({{t}}@0) r#2 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll \
			 {}\n{}\t\t@1 <- sum rolling record ⚅0\n\t\treturn @1\n",
			roll, clauses
		))
		.unwrap();
		let mut evaluator = Evaluator::new(function);
		let arg = rng.random_range(-3..=7);
		let bounds = evaluator
			.bounds_over(
				[Some((arg - rng.random_range(0..=2), arg).into())],
				[]
			)
			.unwrap();
		for _ in 0..50
		{
			let evaluation = evaluator.evaluate([arg], &mut rng).unwrap();
			assert!(
				bounds.value.contains(evaluation.result),
				"{} out of bounds {}: {}\n{}",
				evaluation.result,
				bounds.value,
				evaluation.records[0],
				evaluator.function
			);
			assert!(
				evaluation.dice <= bounds.dice,
				"{} dice over {}: {}\n{}",
				evaluation.dice,
				bounds.dice,
				evaluation.records[0],
				evaluator.function
			);
		}
	}
}
