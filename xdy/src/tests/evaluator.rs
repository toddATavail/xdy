//! # Evaluator tests
//!
//! Herein are the tests for the evaluator. The actual test cases are stored in
//! `../../tests/test_evaluation.txt`, which comprises a series of test cases,
//! each of which consists of a source dice expression and an expected print
//! rendition.

use std::{collections::HashSet, ops::RangeInclusive};

use pretty_assertions::assert_eq;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	EvaluationError, Evaluator, Passes, RollingRecordKind,
	support::{compile_valid, optimize, read_evaluation_test_cases}
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
		assert!(seen.insert(key.clone()), "duplicate test case: {}", &key);
		let function = compile_valid(source);
		let function = optimize(function, Passes::all());
		let mut evaluator = Evaluator::new(function);
		for (name, value) in externs.iter()
		{
			evaluator.bind(name, *value).unwrap();
		}
		// Ensure that the evaluator produces the expected bounds.
		let bounds = evaluator.bounds(args.iter().copied()).unwrap();
		assert_eq!(
			bounds.to_string(),
			*expected,
			"case {}: {}",
			index + 1,
			&key
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
			let bounds: RangeInclusive<i32> = bounds.value.into();
			assert!(
				bounds.contains(&result.result),
				"case {}: {}: result out of bounds: {} ∉ {}..={}: rolls: {}",
				index + 1,
				&key,
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
							&key
						);
						if end < start
						{
							assert_eq!(
								record.results[0],
								0,
								"case {}: {}: roll out of bounds: {} ∉ {}..={}",
								index + 1,
								&key,
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
								&key,
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
							&key
						);
						for result in record.results
						{
							match faces <= 0
							{
								false => assert!(
									(1..=faces).contains(&result),
									"case {}: {}: roll out of bounds: {} ∉ 1..={}",
									index + 1,
									&key,
									result,
									faces
								),
								true => assert_eq!(
									result,
									0,
									"case {}: {}: roll out of bounds: {} ≠ 0",
									index + 1,
									&key,
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
							&key
						);
						for result in record.results
						{
							assert!(
								faces.contains(&result),
								"case {}: {}: roll out of bounds: {} ∉ {:?}",
								index + 1,
								&key,
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
	let function = compile_valid("x: {x}");
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
		evaluator.bounds([].iter().copied()),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
	assert_eq!(
		evaluator.bounds([1, 2].iter().copied()),
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
	let function = compile_valid("x: {x}");
	let mut evaluator = Evaluator::new(function);
	assert_eq!(
		evaluator.bind("y", 1),
		Err(EvaluationError::UnrecognizedExternal("y"))
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
