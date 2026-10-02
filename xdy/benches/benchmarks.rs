//! # Benchmarks
//!
//! Herein are benchmarks for the dice expression pipeline: parsing,
//! compilation and optimization, [`diagnose`], evaluation, bounds, and the
//! estimation and building of distributions, including some whose weights
//! need arbitrary precision.

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
	Budget, Evaluator, Parser, Passes, Unobserved, compile,
	diagnostics::diagnose,
	support::{
		compile_valid, optimize, read_compilation_test_cases,
		read_distribution_test_cases, read_evaluation_test_cases
	}
};

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

/// The source code of the test cases for distributions.
#[cfg(feature = "bench")]
const DISTRIBUTION_TEST_CASES: &str =
	include_str!("../tests/test_distributions.txt");

/// Prepare the [test cases for distributions](DISTRIBUTION_TEST_CASES) for
/// benchmarking.
///
/// # Returns
/// The label, the fully optimized evaluator, bound to the external variables,
/// and the arguments of each test case.
#[cfg(feature = "bench")]
fn distribution_test_cases() -> Vec<(String, Evaluator, Vec<i32>)>
{
	read_distribution_test_cases(DISTRIBUTION_TEST_CASES)
		.into_iter()
		.enumerate()
		.map(|(index, (source, args, externs, _))| {
			let key = (source, args.clone(), externs.clone());
			let label = format!("case {}: {:?}", index, key);
			let function = compile_valid(source);
			let function = optimize(function, Passes::all());
			let mut evaluator = Evaluator::new(function);
			for (name, value) in externs.iter()
			{
				evaluator.bind(name, *value).unwrap();
			}
			(label, evaluator, args)
		})
		.collect()
}

/// Benchmark [planning](Evaluator::plan_distribution) the distribution of each
/// of the [test cases](DISTRIBUTION_TEST_CASES), which estimates its cost.
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
#[cfg(feature = "bench")]
fn bench_estimate<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (label, evaluator, args) in distribution_test_cases()
	{
		g.bench_function(&label, |b| {
			b.iter(|| {
				evaluator.plan_distribution(args.iter().copied()).unwrap()
			});
		});
	}
}

/// Benchmark [building](xdy::DistributionPlan::build) the distribution of each
/// of the [test cases](DISTRIBUTION_TEST_CASES) from its plan, within the
/// [unlimited budget](Budget::UNLIMITED).
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
#[cfg(feature = "bench")]
fn bench_distribution<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (label, evaluator, args) in distribution_test_cases()
	{
		let plan = evaluator.plan_distribution(args.iter().copied()).unwrap();
		g.bench_function(&label, |b| {
			b.iter(|| plan.build(Budget::UNLIMITED, &Unobserved).unwrap());
		});
	}
}

/// Expressions whose distributions have weights beyond [`u128::MAX`], so that
/// building them exercises the arbitrary-precision arithmetic of
/// [`Weight`](xdy::Weight): large pools of dice, drops over them, random
/// counts, whose mixtures scale their setups to large common denominators,
/// and splits. The width of each total, in bits, is noted beside it.
#[cfg(feature = "bench")]
const BIG_WEIGHT_CASES: &[&str] = &[
	// Convolution powers: 259, 517, 260, and 333 bits.
	"100D6",
	"200D6",
	"60D20",
	"50D100",
	// Order statistics: 259 bits.
	"100D6 drop lowest 10",
	// Random counts, with and without drops: 266, 959, 440, and 565 bits.
	"(1D100)D6",
	"(2D20)D(3D6)",
	"(3D6)D(3D6) drop lowest 3",
	"(1D20)D(1D20) drop lowest 1",
	// Sums of mixtures with different denominators: 818 bits.
	"(3D6)D(1D20) + (2D10)D(1D12)",
	// Splits: 281 and 174 bits.
	"{x}@(5D20) + {x}D6",
	"{x}@(1D6)D(1D20) + {x} * 2"
];

/// Benchmark [building](xdy::DistributionPlan::build) the distribution of each
/// of the [big-weight cases](BIG_WEIGHT_CASES), within the
/// [unlimited budget](Budget::UNLIMITED).
///
/// # Parameters
/// - `g`: The benchmark group to which the benchmarks will be added.
///
/// # Panics
/// If the total of any case fits in a [`u128`], since it would then measure
/// only machine arithmetic.
#[cfg(feature = "bench")]
fn bench_big_weights<M: Measurement>(g: &mut BenchmarkGroup<M>)
{
	for (index, source) in BIG_WEIGHT_CASES.iter().enumerate()
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = Evaluator::new(function);
		let plan = evaluator.plan_distribution([]).unwrap();
		let distribution = plan.build(Budget::UNLIMITED, &Unobserved).unwrap();
		assert!(
			distribution.total().bits() > u128::BITS as u64,
			"the total of {source} fits in a u128"
		);
		let label = format!("case {}: {}", index, source);
		g.bench_function(&label, |b| {
			b.iter(|| plan.build(Budget::UNLIMITED, &Unobserved).unwrap());
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
	// The deepest members of the exponential families took tens of
	// milliseconds per iteration in xDy 0.12.0, so use flat sampling, which
	// Criterion recommends for long-running benchmarks.
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
	let mut group = criterion.benchmark_group("estimates");
	bench_estimate(&mut group);
	group.finish();
	let mut group = criterion.benchmark_group("distributions");
	bench_distribution(&mut group);
	group.finish();
	// The largest cases take tens of milliseconds per iteration.
	let mut group = criterion.benchmark_group("big weights");
	group.sampling_mode(SamplingMode::Flat);
	bench_big_weights(&mut group);
	group.finish();

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
