//! # Histogram test cases
//!
//! Herein are tests for the histogram builder. Actual test cases are stored in
//! in `../../tests/test_histograms.txt`, which comprises a series of test
//! cases, each of which consists of a source dice expression and an expected
//! print rendition.

use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "parallel-histogram")]
use std::{
	panic::{self, AssertUnwindSafe},
	sync::atomic::{AtomicBool, Ordering}
};

use pretty_assertions::assert_eq;
#[cfg(feature = "parallel-histogram")]
use rayon::iter::ParallelIterator;

#[cfg(feature = "parallel-histogram")]
use crate::support::on_small_stack;
use crate::{
	EvaluationError, Evaluator, HistogramBuilder, Passes, serial,
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
		assert!(seen.insert(key.clone()), "duplicate test case: {}", key);
		let function = compile_valid(source);
		let function = optimize(function, Passes::all());
		let mut evaluator = Evaluator::new(function);
		for (name, value) in externs.iter()
		{
			evaluator.bind(name, *value).unwrap();
		}
		let bounds = evaluator
			.bounds_over(
				args.iter().map(|arg| Some((*arg).into())),
				externs.iter().map(|(name, value)| (*name, (*value).into()))
			)
			.unwrap();
		// Ensure that the number of outcomes is reasonable for testing.
		assert!(
			bounds.count.map(|c| c <= 50000).unwrap_or(true),
			"case {}: {}: too many outcomes: {} > 50000",
			index + 1,
			key,
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
			assert_eq!(histogram_map, expected, "case {}: {}", index + 1, key);
		}
		// Ensure that the histogram bounds are correct.
		let value_bounds = bounds.value;
		assert_eq!(
			*histogram.keys().min().unwrap(),
			value_bounds.min,
			"case {}: {}: min value bound mismatch",
			index + 1,
			key
		);
		assert_eq!(
			*histogram.keys().max().unwrap(),
			value_bounds.max,
			"case {}: {}: max value bound mismatch",
			index + 1,
			key
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
				key
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
				key,
				outcome
			);
			let percent = histogram.percent_chance(*outcome);
			let expected_percent = (*count as f64 / total as f64) * 100.0;
			assert_eq!(
				percent,
				expected_percent,
				"case {}: {}: percent mismatch for outcome {}",
				index + 1,
				key,
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

////////////////////////////////////////////////////////////////////////////////
//                           Deep histogram tests.                            //
////////////////////////////////////////////////////////////////////////////////

/// The stack size of the worker threads in the deep histogram tests: the
/// default stack size of a `rayon` worker.
#[cfg(feature = "parallel-histogram")]
const WORKER_STACK_SIZE: usize = 2 * 1024 * 1024;

/// Ensure that the parallel builder explores the deep state space of many dice
/// on a single worker with a small stack.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_many_dice_one_thread()
{
	on_small_stack(|| build_many_dice(1))
}

/// Ensure that the parallel builder explores the deep state space of many dice
/// on several workers with small stacks.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_many_dice_four_threads()
{
	on_small_stack(|| build_many_dice(4))
}

/// Ensure that the parallel builder explores the deep state space of dice
/// with many faces on a single worker with a small stack, and that it agrees
/// with the serial builder.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_many_faces_one_thread()
{
	on_small_stack(|| build_many_faces(1))
}

/// Ensure that the parallel builder explores the deep state space of dice
/// with many faces on several workers with small stacks, and that it agrees
/// with the serial builder.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_many_faces_four_threads()
{
	on_small_stack(|| build_many_faces(4))
}

/// Ensure that a panic in the consumer of the parallel builder's iterator
/// propagates to the caller, rather than stranding the other workers.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_histogram_consumer_panic()
{
	on_small_stack(|| {
		// Panic only once, so that the other workers survive and must finish
		// the exploration without the panicking worker.
		let builder = parallel_builder("6D6");
		let panicked = AtomicBool::new(false);
		let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
			in_pool(4, || {
				builder.iter([]).unwrap().for_each(|state| {
					if state.result == Some(21)
						&& !panicked.swap(true, Ordering::Relaxed)
					{
						panic!("deliberate panic in the consumer")
					}
				})
			})
		}));
		assert!(outcome.is_err());
	})
}

/// Build a partial histogram of `800D6`, whose state space is 800 dice deep,
/// with the parallel builder, and check that it contains exactly the requested
/// number of plausible outcomes.
///
/// # Parameters
/// - `threads`: The number of worker threads.
#[cfg(feature = "parallel-histogram")]
fn build_many_dice(threads: usize)
{
	const LIMIT: u64 = 1000;
	let builder = parallel_builder("800D6");
	let histogram =
		in_pool(threads, || builder.build_with_limit([], LIMIT).unwrap());
	assert_eq!(histogram.total(), LIMIT);
	assert!(
		histogram
			.keys()
			.all(|outcome| (800..=4800).contains(outcome)),
		"implausible outcome: {:?}",
		histogram.iter().collect::<BTreeMap<_, _>>()
	);
}

/// Build the complete histogram of `2D1000`, which drove the recursive
/// parallel builder about a thousand levels deep, with the parallel builder,
/// and check that it agrees with the serial builder.
///
/// # Parameters
/// - `threads`: The number of worker threads.
#[cfg(feature = "parallel-histogram")]
fn build_many_faces(threads: usize)
{
	const SOURCE: &str = "2D1000";
	let builder = parallel_builder(SOURCE);
	let parallel = in_pool(threads, || builder.build([]).unwrap());
	let serial =
		serial::HistogramBuilder::new(Evaluator::new(compile_valid(SOURCE)))
			.build([])
			.unwrap();
	assert_eq!(parallel.total(), 1_000_000);
	assert_eq!(
		parallel.iter().collect::<BTreeMap<_, _>>(),
		serial.iter().collect::<BTreeMap<_, _>>()
	);
}

/// Construct a parallel histogram builder for the specified source.
///
/// # Parameters
/// - `source`: The source of the dice expression, which must be closed.
///
/// # Returns
/// The builder.
#[cfg(feature = "parallel-histogram")]
fn parallel_builder(source: &str) -> crate::parallel::HistogramBuilder
{
	crate::parallel::HistogramBuilder::new(Evaluator::new(compile_valid(
		source
	)))
}

/// Run the specified closure in a dedicated `rayon` thread pool whose workers
/// have [small stacks](WORKER_STACK_SIZE).
///
/// # Parameters
/// - `threads`: The number of worker threads.
/// - `f`: The closure to run.
///
/// # Returns
/// The result of the closure.
#[cfg(feature = "parallel-histogram")]
fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R
{
	rayon::ThreadPoolBuilder::new()
		.num_threads(threads)
		.stack_size(WORKER_STACK_SIZE)
		.build()
		.unwrap()
		.install(f)
}

/// Test that the histogram builder takes the [maximum](crate::Max), which only
/// the optimizer emits, of every outcome.
#[test]
fn test_histogram_max()
{
	let function = crate::Assembler::assemble(
		"\
Function() r#2 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll range 1:6
\t\t@0 <- sum rolling record ⚅0
\t\t@1 <- @0 max 3
\t\treturn @1
"
	)
	.unwrap();
	let histogram = serial::HistogramBuilder::new(Evaluator::new(function))
		.build([])
		.unwrap();
	assert_eq!(
		histogram.iter().collect::<BTreeMap<_, _>>(),
		BTreeMap::from([(&3, &3), (&4, &1), (&5, &1), (&6, &1)])
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                 Metering.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Construct a serial histogram builder for the specified source.
///
/// # Parameters
/// - `source`: The source of the dice expression.
///
/// # Returns
/// The builder.
fn serial_builder(source: &str) -> serial::HistogramBuilder
{
	serial::HistogramBuilder::new(Evaluator::new(compile_valid(source)))
}

/// Answer the least budget within which the serial builder builds the
/// histogram of the specified source, i.e., the number of branches that it
/// enumerates, by bisection.
///
/// # Parameters
/// - `source`: The source of the dice expression.
/// - `args`: The arguments.
///
/// # Returns
/// The least sufficient budget.
fn least_budget(source: &str, args: &[i32]) -> u64
{
	let builder = serial_builder(source);
	let (mut low, mut high) = (0, 1u64 << 20);
	assert!(builder.build_metered(args.iter().copied(), high).is_ok());
	while low < high
	{
		let middle = low + (high - low) / 2;
		match builder.build_metered(args.iter().copied(), middle)
		{
			Ok(_) => high = middle,
			Err(EvaluationError::HistogramBudgetExhausted { .. }) =>
			{
				low = middle + 1
			},
			Err(e) => panic!("{}: {}", source, e)
		}
	}
	low
}

/// Test that a metered build charges each range its width and each die its
/// number of faces, once for each path that reaches it, and that a build
/// within budget is exactly the unmetered build.
#[test]
fn test_metered_charges()
{
	for (source, args, branches) in [
		("[1:6]", &[][..], 6),
		("[3:1]", &[], 0),
		("0D6", &[], 0),
		("1D0", &[], 1),
		("1D6", &[], 6),
		("3D6", &[], 6 + 36 + 216),
		("2D[1,2,3]", &[], 3 + 9),
		("{n}: {n}D6", &[2], 42),
		("[1:2] + [1:3]", &[], 2 + 2 * 3),
		("(1D2)D2", &[], 2 + 2 + (2 + 2 * 2))
	]
	{
		assert_eq!(least_budget(source, args), branches, "{}", source);
		let builder = serial_builder(source);
		assert_eq!(
			builder
				.build_metered(args.iter().copied(), branches)
				.unwrap(),
			builder.build(args.iter().copied()).unwrap(),
			"{}",
			source
		);
	}
}

/// Test that a metered build refuses a range or die that exceeds what remains
/// of the budget before enumerating any of it, and reports the refusal
/// exactly.
#[test]
fn test_metered_refusal()
{
	let builder = serial_builder("3D6");
	assert_eq!(
		builder.build_metered([], 257),
		Err(EvaluationError::HistogramBudgetExhausted {
			requested: 6,
			remaining: 5,
			consumed: 252
		})
	);
	assert_eq!(
		builder.build_metered([], 0),
		Err(EvaluationError::HistogramBudgetExhausted {
			requested: 6,
			remaining: 0,
			consumed: 0
		})
	);
}

/// Test that a metered build of a vast state space refuses promptly: `{n}D6`
/// with `n` at [`i32::MAX`], which yields no outcome before its last die, and
/// the widest range, of 2³² values, which is refused at once.
#[test]
fn test_metered_vast_state_spaces()
{
	let builder = serial_builder("{n}: {n}D6");
	assert!(matches!(
		builder.build_metered([i32::MAX], 10_000),
		Err(EvaluationError::HistogramBudgetExhausted { .. })
	));
	let builder = serial_builder("{a}, {b}: [{a}:{b}]");
	assert_eq!(
		builder.build_metered([i32::MIN, i32::MAX], 10_000),
		Err(EvaluationError::HistogramBudgetExhausted {
			requested: 1 << 32,
			remaining: 10_000,
			consumed: 0
		})
	);
}

/// Test that the builder enumerates the widest range, whose width does not fit
/// in an [`i32`], from its start, without overflowing.
#[test]
fn test_widest_range()
{
	let histogram = serial_builder("{a}, {b}: [{a}:{b}]")
		.build_with_limit([i32::MIN, i32::MAX], 3)
		.unwrap();
	assert_eq!(
		histogram.iter().collect::<BTreeMap<_, _>>(),
		BTreeMap::from([
			(&i32::MIN, &1),
			(&(i32::MIN + 1), &1),
			(&(i32::MIN + 2), &1)
		])
	);
}

/// Test that the parallel builder agrees with the serial builder on exactly
/// which budgets suffice, whatever the number of threads and however often it
/// runs, and that its builds within budget are exactly the serial builds.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_metered_agrees()
{
	for (source, args) in [
		("3D6", &[][..]),
		("(1D3)D3", &[]),
		("{n}: {n}D4 drop lowest 1", &[5]),
		("[1:6] * 2D[1,3,5] + 1D10", &[])
	]
	{
		let least = least_budget(source, args);
		let serial =
			serial_builder(source).build(args.iter().copied()).unwrap();
		let builder = parallel_builder(source);
		for threads in [1, 4]
		{
			for _ in 0..8
			{
				let (within, beyond) = in_pool(threads, || {
					(
						builder.build_metered(args.iter().copied(), least),
						builder.build_metered(args.iter().copied(), least - 1)
					)
				});
				assert_eq!(within.as_ref(), Ok(&serial), "{}", source);
				assert!(
					matches!(
						beyond,
						Err(EvaluationError::HistogramBudgetExhausted { .. })
					),
					"{}: {:?}",
					source,
					beyond
				);
			}
		}
	}
}

/// Test that the parallel builder refuses the vast state spaces promptly, and
/// leaves no partial histogram behind to spoil a later build.
#[cfg(feature = "parallel-histogram")]
#[test]
fn test_parallel_metered_vast_state_spaces()
{
	let builder = parallel_builder("{n}: {n}D6");
	for threads in [1, 4]
	{
		let refused =
			in_pool(threads, || builder.build_metered([i32::MAX], 10_000));
		assert!(
			matches!(
				refused,
				Err(EvaluationError::HistogramBudgetExhausted { .. })
			),
			"{:?}",
			refused
		);
		let histogram = in_pool(threads, || builder.build([2]).unwrap());
		assert_eq!(histogram.total(), 36);
	}
	let builder = parallel_builder("{a}, {b}: [{a}:{b}]");
	assert_eq!(
		in_pool(4, || builder.build_metered([i32::MIN, i32::MAX], 10_000)),
		Err(EvaluationError::HistogramBudgetExhausted {
			requested: 1 << 32,
			remaining: 10_000,
			consumed: 0
		})
	);
}
