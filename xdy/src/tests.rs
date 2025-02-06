//! # Tests
//!
//! Herein are test cases for the compiler. The test cases are stored in
//! files in the `../tests` directory. Each file contains a series of test
//! cases, each of which consists of a source dice expression and an expected
//! print rendition.

use std::{
	collections::{BTreeMap, HashSet},
	ops::RangeInclusive
};

use pretty_assertions::assert_eq;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	EvaluationError, Evaluator, HistogramBuilder, Pass, Passes,
	RollingRecordKind, serial,
	support::{
		compile_valid, optimize, read_compilation_test_cases,
		read_evaluation_test_cases, read_histogram_test_cases
	}
};

////////////////////////////////////////////////////////////////////////////////
//                             Histogram support.                             //
////////////////////////////////////////////////////////////////////////////////

/// Run a set of histogram test cases.
///
/// # Parameters
/// - `source`: The contents of a test case file.
/// - `builder`: A function that constructs a histogram builder from an
///   evaluator.
pub fn histogram_test<I, T, B>(source: &'static str, builder: B)
where
	I: 'static,
	T: HistogramBuilder<'static, I>,
	B: Fn(Evaluator) -> T
{
	let mut seen = HashSet::new();
	for (index, (source, args, externs, expected)) in
		read_histogram_test_cases(source).iter().enumerate()
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
		let bounds = evaluator.bounds(args.iter().copied()).unwrap();
		// Ensure that the number of outcomes is reasonable for testing.
		assert!(
			bounds.count.map(|c| c <= 50000).unwrap_or(true),
			"case {}: {}: too many outcomes: {} > 50000",
			index + 1,
			&key,
			bounds.count.unwrap()
		);
		// Set up the map of expected results.
		let expected = expected
			.iter()
			.map(|(key, value)| (*key, *value as u64))
			.collect::<BTreeMap<_, _>>();
		// Build the histogram.
		let builder = builder(evaluator);
		let histogram = builder.build(args.iter().copied()).unwrap();
		{
			let histogram_map = histogram
				.iter()
				.map(|(outcome, count)| (*outcome, *count))
				.collect::<BTreeMap<_, _>>();
			assert_eq!(histogram_map, expected, "case {}: {}", index + 1, &key);
		}
		// Ensure that the histogram bounds are correct.
		let value_bounds = bounds.value;
		assert_eq!(
			*histogram.keys().min().unwrap(),
			value_bounds.min,
			"case {}: {}: min value bound mismatch",
			index + 1,
			&key
		);
		assert_eq!(
			*histogram.keys().max().unwrap(),
			value_bounds.max,
			"case {}: {}: max value bound mismatch",
			index + 1,
			&key
		);
		// Ensure that the outcome counts are correct.
		if let Some(expected_outcomes) = bounds.count
		{
			let actual_outcomes: u128 =
				histogram.values().map(|count| *count as u128).sum();
			assert_eq!(
				actual_outcomes,
				expected_outcomes,
				"case {}: {}: outcome count mismatch",
				index + 1,
				&key
			);
		}
		// Ensure that the odds are correct.
		let total = histogram.total();
		for (outcome, count) in histogram.iter()
		{
			let odds = histogram.odds(*outcome);
			let expected_odds = (*count, total - *count);
			assert_eq!(
				odds,
				expected_odds,
				"case {}: {}: odds mismatch for outcome {}",
				index + 1,
				&key,
				outcome
			);
			let percent = histogram.percent_chance(*outcome);
			let expected_percent = (*count as f64 / total as f64) * 100.0;
			assert_eq!(
				percent,
				expected_percent,
				"case {}: {}: percent mismatch for outcome {}",
				index + 1,
				&key,
				outcome
			);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Test cases.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Test that the compiler generates the expected output for the test cases.
#[test]
fn test_compile_unoptimized()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_compile_unoptimized.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let actual = format!("{}", compile_valid(source));
		assert_eq!(actual.trim(), *expected, "case {}: {}", index + 1, source);
	}
}

/// Test that the common subexpression eliminator generates the expected output
/// for the test cases.
#[test]
fn test_common_subexpression_elimination()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_common_subexpression_elimination.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(
			function.clone(),
			Pass::CommonSubexpressionElimination.into()
		);
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant commuter generates the expected output for the test
/// cases.
#[test]
fn test_constant_commuting()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_constant_commuting.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantCommuting.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant folder generates the expected output for the test
/// cases.
#[test]
fn test_constant_folding()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_constant_folding.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantFolding.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the strength reducer generates the expected output for the test
/// cases.
#[test]
fn test_strength_reduction()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_strength_reduction.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::StrengthReduction.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the register coalescer generates the expected output for the test
/// cases.
#[test]
fn test_register_coalescence()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_register_coalescence.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::RegisterCoalescing.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the fully-loaded optimizer produces the expected output for the
/// test cases.
#[test]
fn test_full_optimization()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../tests/test_full_optimization.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(function.clone(), Passes::all());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the evaluator produces the expected output for the test cases.
#[test]
fn test_evaluation()
{
	let mut seen = HashSet::new();
	for (index, (source, args, externs, expected)) in
		read_evaluation_test_cases(include_str!("../tests/test_evaluation.txt"))
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

/// The content of the test file for histograms.
const HISTOGRAM_TEST_SOURCE: &str =
	include_str!("../tests/test_histograms.txt");

/// Test that histograms are built correctly by the serial builder.
#[test]
fn test_serial_histogram_building()
{
	histogram_test(HISTOGRAM_TEST_SOURCE, serial::HistogramBuilder::new)
}

/// Test that histograms are built correctly by the parallel builder.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_building()
{
	histogram_test(
		HISTOGRAM_TEST_SOURCE,
		crate::parallel::HistogramBuilder::new
	)
}
