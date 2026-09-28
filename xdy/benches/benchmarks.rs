//! # Benchmarks
//!
//! Herein are benchmarks for the dice expression compiler.

#[cfg(feature = "bench")]
use std::time::Duration;

#[cfg(feature = "bench")]
use criterion::{
	BenchmarkGroup, BenchmarkId, Criterion, SamplingMode,
	measurement::Measurement
};

#[cfg(feature = "bench")]
use rand::{SeedableRng, rngs::StdRng};

#[cfg(feature = "bench")]
use xdy::{
	Evaluator, HistogramBuilder, Parser, Passes, compile,
	diagnostics::diagnose,
	serial,
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

/// Benchmark parsing alone for each of the full optimization test cases. The
/// cases and labels match [`bench_optimize`], so the parser's share of the
/// full pipeline can be read off case by case.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
#[cfg(feature = "bench")]
fn bench_parse<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (index, (source, _)) in read_compilation_test_cases(include_str!(
		"../tests/test_full_optimization.txt"
	))
	.iter()
	.enumerate()
	{
		let label = format!("case {}: {}", index, source);
		g.bench_function(label, |b| {
			b.iter(|| Parser::parse(source).is_ok());
		});
	}
}

/// A family of nested dice expressions, parameterized by nesting depth.
///
/// # Notes
/// The families reproduce the complaints that motivated the iterative parser:
/// in `xDy` 0.12.0, some families took time exponential in depth, and all of
/// them recursed once per level, so a deep enough member overflowed the stack.
#[cfg(feature = "bench")]
struct Nesting
{
	/// The name of the family, used as the benchmark function name.
	name: &'static str,

	/// The depths at which to benchmark the family.
	depths: &'static [usize],

	/// Build the source text of the family member at the given depth.
	source: fn(usize) -> String
}

/// The depths for the families whose parse time is exponential in depth in
/// `xDy` 0.12.0. Beyond `16`, a single parse takes too long to benchmark.
#[cfg(feature = "bench")]
const EXPONENTIAL_DEPTHS: &[usize] = &[1, 2, 4, 8, 12, 16];

/// The depths for the families whose parse time is linear in depth in `xDy`
/// 0.12.0, but whose recursion eventually overflows the stack. The deepest
/// stays well below the overflow depth of an optimized build on the main
/// thread.
#[cfg(feature = "bench")]
const RECURSIVE_DEPTHS: &[usize] = &[1, 4, 16, 64, 256];

/// The nested families to benchmark.
#[cfg(feature = "bench")]
const NESTINGS: &[Nesting] = &[
	Nesting {
		name: "group",
		depths: EXPONENTIAL_DEPTHS,
		source: |n| format!("{}1{}", "(".repeat(n), ")".repeat(n))
	},
	Nesting {
		name: "group then dice",
		depths: EXPONENTIAL_DEPTHS,
		source: |n| format!("{}1{}D6", "(".repeat(n), ")".repeat(n))
	},
	Nesting {
		name: "binding",
		depths: EXPONENTIAL_DEPTHS,
		// Each binding binds its own name, since a binding may not rebind a
		// name that an enclosing binding binds.
		source: |n| {
			let bindings =
				(0..n).map(|i| format!("{{x{}}}@(", i)).collect::<String>();
			format!("{}1{}", bindings, ")".repeat(n))
		}
	},
	Nesting {
		name: "dice count",
		depths: RECURSIVE_DEPTHS,
		source: |n| format!("{}1{}", "(".repeat(n), ")D6".repeat(n))
	},
	Nesting {
		name: "negation",
		depths: RECURSIVE_DEPTHS,
		source: |n| format!("{}1", "-".repeat(n))
	},
	Nesting {
		name: "exponent",
		depths: RECURSIVE_DEPTHS,
		source: |n| format!("{}2", "2^".repeat(n))
	},
	Nesting {
		name: "range",
		depths: RECURSIVE_DEPTHS,
		source: |n| format!("{}1{}", "[1:".repeat(n), "]".repeat(n))
	}
];

/// Benchmark the [nested families](NESTINGS) at each of their depths.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
/// - `run`: The operation to benchmark, which answers whether it succeeded.
///
/// # Panics
/// If `run` fails for any family member, since a benchmark of the error path
/// would misrepresent the cost of the success path.
#[cfg(feature = "bench")]
fn bench_nesting<M: Measurement>(
	g: &mut BenchmarkGroup<M>,
	run: fn(&str) -> bool
)
{
	for nesting in NESTINGS
	{
		for &depth in nesting.depths
		{
			let source = (nesting.source)(depth);
			assert!(run(&source), "{} at depth {}", nesting.name, depth);
			g.bench_with_input(
				BenchmarkId::new(nesting.name, depth),
				&source,
				|b, source| b.iter(|| run(source))
			);
		}
	}
}

/// A family of failing dice expressions, parameterized by size, for
/// benchmarking [`diagnose`].
///
/// # Notes
/// The families include the complaints that motivated the recovering parse:
/// before it, [`diagnose`] parsed the source again after each fix, so it took
/// time quadratic in the number of errors, whether they nested or not.
#[cfg(feature = "bench")]
struct Failure
{
	/// The name of the family, used as the benchmark function name.
	name: &'static str,

	/// Build the source text of the family member of the given size.
	source: fn(usize) -> String
}

/// The sizes at which to benchmark the [failing families](FAILURES).
#[cfg(feature = "bench")]
const FAILURE_SIZES: &[usize] = &[10, 100, 1_000, 10_000, 100_000];

/// The failing families to benchmark.
#[cfg(feature = "bench")]
const FAILURES: &[Failure] = &[
	Failure {
		name: "unclosed groups",
		source: |n| format!("{}1", "(".repeat(n))
	},
	Failure {
		name: "bare identifiers",
		source: |n| vec!["x"; n].join(" + ")
	},
	Failure {
		name: "leading closers",
		source: |n| format!("{}1", ")".repeat(n))
	},
	Failure {
		name: "missing operands",
		source: |n| format!("1{}", " +".repeat(n))
	}
];

/// Benchmark [`diagnose`] on the [failing families](FAILURES) at each of the
/// [sizes](FAILURE_SIZES).
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
///
/// # Panics
/// If [`diagnose`] finds no error in any family member.
#[cfg(feature = "bench")]
fn bench_failing_diagnose<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for failure in FAILURES
	{
		for &size in FAILURE_SIZES
		{
			let source = (failure.source)(size);
			assert!(
				!diagnose(&source).diagnostics.is_empty(),
				"{} at size {}",
				failure.name,
				size
			);
			g.bench_with_input(
				BenchmarkId::new(failure.name, size),
				&source,
				|b, source| b.iter(|| diagnose(source))
			);
		}
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
			b.iter(|| {
				evaluator
					.bounds_over(
						args.iter().map(|arg| Some((*arg).into())),
						externs
							.iter()
							.map(|(name, value)| (*name, (*value).into()))
					)
					.unwrap()
			});
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
	let mut group = criterion.benchmark_group("parse");
	group.measurement_time(Duration::from_secs(1));
	bench_parse(&mut group);
	group.finish();
	// The deepest family members of the exponential families take tens of
	// milliseconds per iteration, so use flat sampling, which Criterion
	// recommends for long-running benchmarks.
	let mut group = criterion.benchmark_group("nesting parse");
	group.measurement_time(Duration::from_secs(1));
	group.sampling_mode(SamplingMode::Flat);
	bench_nesting(&mut group, |source| Parser::parse(source).is_ok());
	group.finish();
	let mut group = criterion.benchmark_group("nesting compile");
	group.measurement_time(Duration::from_secs(1));
	group.sampling_mode(SamplingMode::Flat);
	bench_nesting(&mut group, |source| compile(source).is_ok());
	group.finish();
	// The largest family members take hundreds of milliseconds per iteration.
	let mut group = criterion.benchmark_group("failing diagnose");
	group.measurement_time(Duration::from_secs(1));
	group.sampling_mode(SamplingMode::Flat);
	bench_failing_diagnose(&mut group);
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
