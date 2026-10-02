//! # Evaluator tests
//!
//! Herein are the tests for the evaluator. The actual test cases are stored in
//! `../../tests/test_evaluation.txt`, which comprises a series of test cases,
//! each of which consists of a source dice expression and an expected print
//! rendition.

use std::{collections::HashSet, ops::RangeInclusive};

use pretty_assertions::assert_eq;
use rand::{Rng as _, SeedableRng, rngs::StdRng};

use crate::{
	Budget, EvaluationError, Evaluator, Passes, RollingRecordKind, Unobserved,
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
					.map(|record| record.results.len() as u64)
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
					.map(|record| record.results[0].to_string())
					.collect::<Vec<_>>()
					.join(", ")
			);
			// Ensure that none of the die rolls are outside the expected
			// bounds.
			for record in result.records
			{
				match record.kind
				{
					RollingRecordKind::Uninitialized => unreachable!(),
					RollingRecordKind::Range { start, end } =>
					{
						assert_eq!(
							record.results.len(),
							1,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						if end < start
						{
							assert_eq!(
								record.results[0],
								0,
								"case {}: {}: roll out of bounds: {} ∉ {}..={}",
								index + 1,
								key,
								record.results[0],
								start,
								end
							);
						}
						else
						{
							assert!(
								(start..=end).contains(&record.results[0]),
								"case {}: {}: roll out of bounds: {} ∉ {}..={}",
								index + 1,
								key,
								record.results[0],
								start,
								end
							);
						}
					},
					RollingRecordKind::Standard { count, faces } =>
					{
						assert_eq!(
							record.results.len(),
							count.max(0) as usize,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						for result in record.results
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
							record.results.len(),
							count as usize,
							"case {}: {}: wrong number of results",
							index + 1,
							key
						);
						for result in record.results
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
