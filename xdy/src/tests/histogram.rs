//! # Histogram test cases
//!
//! Herein are tests for the histogram builder. Actual test cases are stored in
//! in `../../tests/test_histograms.txt`, which comprises a series of test
//! cases, each of which consists of a source dice expression and an expected
//! print rendition.

use std::collections::{BTreeMap, HashSet};

use pretty_assertions::assert_eq;

use crate::{
	Evaluator, HistogramBuilder, Passes, serial,
	support::{compile_valid, optimize, read_histogram_test_cases}
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
//                              Histogram tests.                              //
////////////////////////////////////////////////////////////////////////////////

/// The content of the test file for histograms.
const HISTOGRAM_TEST_SOURCE: &str =
	include_str!("../../tests/test_histograms.txt");

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
