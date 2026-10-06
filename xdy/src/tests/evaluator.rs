//! # Evaluator tests
//!
//! Herein are the tests for the evaluator. The actual test cases are stored in
//! `../../tests/test_evaluation.txt`, which comprises a series of test cases,
//! each of which consists of a source dice expression and an expected print
//! rendition.

use std::{collections::HashSet, ops::RangeInclusive};

use pretty_assertions::assert_eq;
use rand::{Rng as _, RngExt as _, SeedableRng, rngs::StdRng};

use crate::{
	AddressingMode, Assembler, Budget, Criterion, Die, EvaluationBounds,
	EvaluationError, Evaluator, Fate, Immediate, Passes, Predicate,
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
								.filter(|die| die.replaces.is_none())
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
								.filter(|die| die.replaces.is_none())
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
								.filter(|die| die.replaces.is_none())
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
	assert_eq!(record.dice[3].replaces, Some(1));
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
	let links = record
		.dice
		.iter()
		.map(|die| die.replaces)
		.collect::<Vec<_>>();
	assert_eq!(
		links,
		[
			None,
			None,
			None,
			Some(0),
			Some(3),
			Some(4),
			Some(2),
			Some(6),
			Some(7)
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
				&& record.dice[next].replaces == Some(current)
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
