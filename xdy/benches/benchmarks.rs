//! # Benchmarks
//!
//! Herein are benchmarks for the dice expression compiler.

#[cfg(feature = "bench")]
use std::time::Duration;

#[cfg(feature = "bench")]
use criterion::{BenchmarkGroup, Criterion, measurement::Measurement};

#[cfg(feature = "bench")]
use rand::{SeedableRng, rngs::StdRng};

#[cfg(feature = "bench")]
use xdy::{
	Evaluator, HistogramBuilder, Passes, serial,
	support::{
		compile_valid, optimize, read_compilation_test_cases,
		read_evaluation_test_cases, read_histogram_test_cases
	}
};

#[cfg(all(feature = "bench", feature = "parallel-histogram"))]
use xdy::parallel;

////////////////////////////////////////////////////////////////////////////////
//                                Benchmarks.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Benchmark the full optimization of each of the test cases.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
#[cfg(feature = "bench")]
fn bench_optimize<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (index, (source, _)) in read_compilation_test_cases(include_str!(
		"../tests/test_full_optimization.txt"
	))
	.iter()
	.enumerate()
	{
		// Criterion does not correctly escape an octothorpe, so we dropped it
		// from the case numbers everywhere. I prefer the octothorpe, but it's
		// better to have correct names than not.
		let label = format!("case {}: {}", index, source);
		g.bench_function(label, |b| {
			b.iter(|| optimize(compile_valid(source), Passes::all()));
		});
	}
}

/// Benchmark the evaluation test cases.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
#[cfg(feature = "bench")]
fn bench_evaluate<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (index, (source, args, externs, _)) in
		read_evaluation_test_cases(include_str!("../tests/test_evaluation.txt"))
			.iter()
			.enumerate()
	{
		let key = (source, args.clone(), externs.clone());
		let key = format!("{:?}", key);
		let label = format!("case {}: {}", index, key);
		let function = compile_valid(source);
		let function = optimize(function, Passes::all());
		let mut evaluator = Evaluator::new(function);
		for (name, value) in externs.iter()
		{
			evaluator.bind(name, *value).unwrap();
		}
		// The seed is arbitrary, chosen by smashing the keyboard. This is to
		// ensure that the test cases are deterministic.
		let mut rng = StdRng::seed_from_u64(12094850988972349040);
		g.bench_function(&label, |b| {
			b.iter(|| {
				evaluator.evaluate(args.iter().copied(), &mut rng).unwrap()
			});
		});
		let label = format!("{}: bounds", label);
		g.bench_function(label, |b| {
			b.iter(|| evaluator.bounds(args.iter().copied()).unwrap());
		});
	}
}

/// Benchmark the test cases for serial histogram building.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
/// - `source`: The source code of the test cases.
#[cfg(feature = "bench")]
fn bench_histogram<'inst, I, T, B, M: Measurement>(
	g: &mut BenchmarkGroup<M>,
	source: &'static str,
	builder: B
) where
	I: 'inst,
	T: HistogramBuilder<'inst, I>,
	B: Fn(Evaluator) -> T
{
	for (index, (source, args, externs, _)) in
		read_histogram_test_cases(source).iter().enumerate()
	{
		let key = (source, args.clone(), externs.clone());
		let key = format!("{:?}", key);
		let label = format!("case {}: {}", index, key);
		let function = compile_valid(source);
		let function = optimize(function, Passes::all());
		let mut evaluator = Evaluator::new(function);
		for (name, value) in externs.iter()
		{
			evaluator.bind(name, *value).unwrap();
		}
		let builder = builder(evaluator);
		g.bench_function(&label, |b| {
			b.iter(|| builder.build(args.iter().copied()).unwrap());
		});
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Harness.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Run all benchmarks.
#[cfg(feature = "bench")]
fn main()
{
	// Run the benchmarks.
	let mut criterion = Criterion::default().configure_from_args();
	let mut group = criterion.benchmark_group("optimizations");
	group.measurement_time(Duration::from_secs(1));
	bench_optimize(&mut group);
	group.finish();
	let mut group = criterion.benchmark_group("evaluations");
	group.measurement_time(Duration::from_secs(1));
	bench_evaluate(&mut group);
	group.finish();
	let histogram_test_cases = include_str!("../tests/test_histograms.txt");
	let mut group = criterion.benchmark_group("serial histograms");
	bench_histogram(&mut group, histogram_test_cases, |evaluator| {
		serial::HistogramBuilder::new(evaluator)
	});
	group.finish();
	#[cfg(feature = "parallel-histogram")]
	{
		let mut group = criterion.benchmark_group("parallel histograms");
		bench_histogram(&mut group, histogram_test_cases, |evaluator| {
			parallel::HistogramBuilder::new(evaluator)
		});
		group.finish();
	}

	// Generate the final summary.
	criterion.final_summary();
}

/// Dummy harness to prevent benchmarking without the `bench` feature.
#[cfg(not(feature = "bench"))]
fn main()
{
	panic!(
		"benchmarks are disabled; enable the 'bench' feature to run benchmarks"
	);
}
