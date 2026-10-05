//! # Propagation tests
//!
//! Herein are the tests of the [forward pass](propagate) that computes exact
//! distributions by propagating them through a function's instructions. The
//! expected distributions are those that the enumerating builders, which the
//! pass replaced, computed by walking every path through each function, and
//! with which the pass agreed before they retired: the [distribution
//! corpus](DISTRIBUTION_TEST_SOURCE) holds those of sources, which the Lean
//! oracle checks, and the tests hold those of functions assembled by hand. The
//! pass must answer each optimized source weight for weight, and each function
//! weight for weight unless the setups of some roll are chosen along paths of
//! unequal probability, when it must agree in probability. It must answer
//! every valid function, conditioning on every random value read more than
//! once, even a rolling record summed more than once, which the compiler never
//! emits; random programs hold it to its bounds, to the evaluator, and to
//! itself, optimized and not.

use std::{
	cell::RefCell,
	collections::{BTreeMap, HashSet, btree_map::Entry},
	mem,
	ops::ControlFlow,
	sync::OnceLock
};

use pretty_assertions::assert_eq;
use proptest::{
	option, prelude::*, sample::select, test_runner::TestCaseError
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	Add, AddressingMode, Assembler, BuildError, Distribution, EvaluationError,
	Evaluator, Function, Immediate, Instruction, Limit, Pass, Passes,
	Predicate, ProgramCounter, Rational, RegisterIndex, Return,
	RollingRecordIndex, SumRollingRecord, Weight,
	compiler::{compile, compile_unoptimized},
	distribution::propagation::{
		PropagationError,
		cost::{Class, Cost, estimate},
		meter::{Budget, Dimension, Exhausted, Unobserved, Usage},
		plan::{Location, Plan, Step},
		propagate, propagate_worlds,
		record::{convolve_dense, convolve_packed}
	},
	support::{
		DistributionTestCase, compile_valid, optimize,
		read_distribution_test_cases
	},
	tests::property::{
		NAMES, OPERATORS, agree, binding, check_within, variable,
		with_parameters
	}
};

////////////////////////////////////////////////////////////////////////////////
//                                  Support.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Answer an evaluator of the specified function, with the specified
/// externals bound.
///
/// # Parameters
/// - `function`: The function.
/// - `externals`: The names and values of the externals.
///
/// # Returns
/// The evaluator.
fn evaluator(function: Function, externals: &[(&str, i32)]) -> Evaluator
{
	let mut evaluator = Evaluator::new(function);
	for (name, value) in externals
	{
		evaluator.bind(name, *value).unwrap();
	}
	evaluator
}

/// The answer of the pass: the distribution and the usage, or the refusal.
type Answer = Result<(Distribution, Usage), PropagationError>;

/// Answer the distribution that the pass computes for the specified evaluator
/// and arguments within the specified budget, and its usage, reporting its
/// progress to no one.
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `budget`: The budget.
///
/// # Returns
/// The distribution and the usage, or the refusal.
fn propagate_within(
	evaluator: &Evaluator,
	args: &[i32],
	budget: Budget
) -> Answer
{
	let estimate = estimate(evaluator, args.iter().copied())?;
	propagate_worlds(
		evaluator,
		args.iter().copied(),
		budget,
		&estimate,
		&Unobserved
	)
}

/// Assert that the pass answers the specified function, which takes no
/// arguments, exactly as expected, weight for weight.
///
/// # Parameters
/// - `function`: The function.
/// - `expected`: The expected distribution.
///
/// # Panics
/// If the pass refuses the function, or answers other than expected.
fn assert_agrees(function: Function, expected: &Distribution)
{
	let evaluator = Evaluator::new(function);
	let propagated = propagate(&evaluator, []).unwrap_or_else(|e| {
		panic!("refused: {e:?}\nfunction:\n{}", evaluator.function)
	});
	assert_eq!(&propagated, expected, "function:\n{}", evaluator.function);
}

/// Assert that the pass answers the specified function, which takes no
/// arguments, as expected, in probability, and answer its distribution.
///
/// # Parameters
/// - `function`: The function.
/// - `expected`: The expected distribution.
///
/// # Returns
/// The distribution that the pass answers.
///
/// # Panics
/// If the pass refuses the function, or answers other than expected in
/// probability.
fn assert_agrees_in_probability(
	function: Function,
	expected: &Distribution
) -> Distribution
{
	let evaluator = Evaluator::new(function);
	let propagated = propagate(&evaluator, []).unwrap_or_else(|e| {
		panic!("refused: {e:?}\nfunction:\n{}", evaluator.function)
	});
	assert!(
		agree(&propagated, expected),
		"function:\n{}\npropagated:\n{}\nexpected:\n{}",
		evaluator.function,
		propagated,
		expected
	);
	propagated
}

/// Assert that the pass answers the specified source as the [distribution
/// corpus](DISTRIBUTION_TEST_SOURCE) expects: optimized, weight for weight, and
/// unoptimized, in probability, since optimization may change the number of
/// branches that reach each outcome, e.g., by eliminating dice that are always
/// dropped.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments.
/// - `externals`: The names and values of the externals.
///
/// # Panics
/// If the corpus has no case for the source, arguments, and externals, or the
/// pass refuses either function, or answers other than expected.
fn assert_source_agrees(source: &str, args: &[i32], externals: &[(&str, i32)])
{
	let expected = corpus_case(source, args, externals);
	let unoptimized = compile_valid(source);
	let optimized = optimize(unoptimized.clone(), Passes::all());
	for (function, exact) in [(unoptimized, false), (optimized, true)]
	{
		let evaluator = evaluator(function, externals);
		let propagated = propagate(&evaluator, args.iter().copied())
			.unwrap_or_else(|e| {
				panic!("refused: {e:?}\nfunction:\n{}", evaluator.function)
			});
		if exact
		{
			assert_eq!(
				propagated, expected,
				"function:\n{}",
				evaluator.function
			);
		}
		else
		{
			assert!(
				agree(&propagated, &expected),
				"function:\n{}\npropagated:\n{}\nexpected:\n{}",
				evaluator.function,
				propagated,
				expected
			);
		}
	}
}

/// Answer the distribution that the [distribution
/// corpus](DISTRIBUTION_TEST_SOURCE) expects of the specified source,
/// arguments, and externals.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments.
/// - `externals`: The names and values of the externals.
///
/// # Returns
/// The expected distribution.
///
/// # Panics
/// If the corpus has no such case.
fn corpus_case(
	source: &str,
	args: &[i32],
	externals: &[(&str, i32)]
) -> Distribution
{
	static CORPUS: OnceLock<Vec<DistributionTestCase>> = OnceLock::new();
	let (.., expected) = CORPUS
		.get_or_init(|| read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE))
		.iter()
		.find(|(s, a, e, _)| *s == source && a == args && e == externals)
		.unwrap_or_else(|| {
			panic!("no case in the corpus: {source} {args:?} {externals:?}")
		});
	Distribution::from_weights(
		expected
			.iter()
			.map(|&(outcome, weight)| (outcome, Weight::from(weight)))
	)
	.unwrap()
}

/// Answer the distribution that the pass computes for the specified source,
/// optimized.
///
/// # Parameters
/// - `source`: The source, which takes no arguments.
///
/// # Returns
/// The distribution, or the refusal.
fn propagate_source(source: &str) -> Result<Distribution, PropagationError>
{
	propagate(&Evaluator::new(compile(source).unwrap()), [])
}

/// Answer the distribution that the pass computes for the specified assembly.
///
/// # Parameters
/// - `assembly`: The text form of a function that takes no arguments.
///
/// # Returns
/// The distribution.
///
/// # Panics
/// If the assembly is malformed, or the pass refuses it.
fn propagate_assembly(assembly: &str) -> Distribution
{
	let function = Assembler::assemble(assembly).unwrap();
	propagate(&Evaluator::new(function), []).unwrap()
}

/// Construct the distribution that weighs each of the specified outcomes as
/// specified.
///
/// # Parameters
/// - `weights`: The outcomes and their weights.
///
/// # Returns
/// The distribution.
fn weights(weights: &[(i32, u64)]) -> Distribution
{
	Distribution::from_weights(
		weights
			.iter()
			.map(|&(outcome, weight)| (outcome, Weight::from(weight)))
	)
	.unwrap()
}

////////////////////////////////////////////////////////////////////////////////
//                                  Corpus.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The content of the test file for distributions.
const DISTRIBUTION_TEST_SOURCE: &str =
	include_str!("../../tests/test_distributions.txt");

/// Ensure that the pass answers every case of the distribution corpus exactly
/// as the corpus expects, weight for weight, even where rolls have random
/// operands, and where random values are read more than once; that the
/// [bounds](Evaluator::bounds_over) of every case contain its outcomes, tightly
/// unless some random value is read more than once, and count them, if they
/// count them, of which there are few enough for testing; and that no case
/// appears twice.
#[test]
fn test_propagation_distribution_corpus()
{
	let mut seen = HashSet::new();
	for (index, (source, args, externs, expected)) in
		read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE)
			.iter()
			.enumerate()
	{
		assert!(
			seen.insert((source, args, externs)),
			"case {}: duplicate: {} {:?} {:?}",
			index + 1,
			source,
			args,
			externs
		);
		let case = format!("case {}: {source} {args:?} {externs:?}", index + 1);
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = evaluator(function, externs);
		let distribution = propagate(&evaluator, args.iter().copied())
			.unwrap_or_else(|e| panic!("{case}: {e:?}"));
		let expected = Distribution::from_weights(
			expected
				.iter()
				.map(|&(outcome, weight)| (outcome, Weight::from(weight)))
		)
		.unwrap();
		assert_eq!(distribution, expected, "{case}");
		let bounds = evaluator
			.bounds_over(
				args.iter().map(|arg| Some((*arg).into())),
				externs.iter().map(|(name, value)| (*name, (*value).into()))
			)
			.unwrap();
		// Interval arithmetic cannot see that a value read more than once takes
		// the same value at each read, so it bounds such a function soundly,
		// but not tightly.
		if estimate(&evaluator, args.iter().copied())
			.unwrap()
			.class
			.correlated
		{
			assert!(
				bounds.value.min <= distribution.min()
					&& distribution.max() <= bounds.value.max,
				"{case}: out of bounds {}",
				bounds.value
			);
		}
		else
		{
			assert_eq!(
				(distribution.min(), distribution.max()),
				(bounds.value.min, bounds.value.max),
				"{case}: loose bounds"
			);
		}
		if let Some(count) = bounds.count
		{
			assert!(count <= 50_000, "{case}: too many outcomes: {count}");
			assert_eq!(distribution.total(), &Weight::from(count), "{case}");
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Rolls.                                      //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that degenerate rolls answer as expected: no dice, dice without
/// faces, and empty ranges each sum to zero.
#[test]
fn test_degenerate_rolls()
{
	for (source, args) in [
		("0D6", &[][..]),
		("{n}: {n}D6", &[-1]),
		("{n}: {n}D6", &[i32::MIN]),
		("{f}: 3D{f}", &[0]),
		("{f}: 3D{f}", &[-1]),
		("{f}: 3D{f}", &[i32::MIN]),
		("{n}: {n}D[1, 2]", &[0]),
		("{a}, {b}: [{a}:{b}]", &[6, 1]),
		("{a}, {b}: [{a}:{b}]", &[i32::MAX, i32::MIN])
	]
	{
		assert_source_agrees(source, args, &[]);
	}
}

/// Ensure that dice without drops, whose sums are convolution powers, answer
/// as expected, including repeated and mixed-sign faces.
#[test]
fn test_convolution_powers()
{
	for source in [
		"1D6",
		"3D6",
		"7D2",
		"5D3",
		"2D10 + 1D4",
		"3D[1, 1, 2]",
		"4D[-1, 0, 1]",
		"3D[-2, 5, 5, 11]",
		"3D-6",
		"[1:6]",
		"[-3:3] + [0:2]"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that dice with drops, whose sums are order statistics, answer as
/// expected, including drops that exceed the dice.
#[test]
fn test_order_statistics()
{
	for source in [
		"4D6 drop lowest",
		"4D6 drop highest",
		"4D6 drop lowest drop highest",
		"5D4 drop lowest 2 drop highest 2",
		"3D6 drop lowest 2 drop highest 2",
		"3D6 drop lowest 5",
		"3D6 drop highest 3",
		"4D[1, 1, 2, 3] drop lowest",
		"4D[-3, -1, 0, 2, 7] drop highest 2",
		"6D6 drop lowest 2",
		"3D6 drop lowest 0",
		"2D6 drop lowest (-1)"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that sums that saturate answer as expected, folding the kept dice in
/// ascending order, so that a fold that saturates at [`i32::MIN`] climbs back
/// from there.
#[test]
fn test_saturating_sums()
{
	for source in [
		"3D[2147483647, 1]",
		"3D[-2147483648, -1]",
		"2D[-2147483648, 2147483647]",
		"3D[-2147483648, 0, 2147483647]",
		"3D[-2147483648, 1, 2147483647] drop lowest",
		"4D[-2147483648, -1073741824, 1, 2147483647] drop highest",
		"2D[-1073741825, 1073741824, 2147483647]"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that the pass answers rolls far beyond the reach of enumerating their
/// paths.
#[test]
fn test_large_rolls()
{
	let d6 = Weight::from(6u8);
	// 100D6 is a convolution power.
	let distribution = propagate_source("100D6").unwrap();
	assert_eq!(distribution.total(), &d6.pow(100));
	assert_eq!((distribution.min(), distribution.max()), (100, 600));
	assert_eq!(distribution.get(100), &Weight::ONE);
	assert_eq!(distribution.get(101), &Weight::from(100u8));
	let mean = Rational::new(false, Weight::from(350u16), Weight::ONE).unwrap();
	assert_eq!(distribution.mean(), mean);
	for outcome in 100..=600
	{
		assert_eq!(distribution.get(outcome), distribution.get(700 - outcome));
	}
	// 30D6 drop lowest 3 is an order statistic.
	let distribution = propagate_source("30D6 drop lowest 3").unwrap();
	assert_eq!(distribution.total(), &d6.pow(30));
	assert_eq!((distribution.min(), distribution.max()), (27, 162));
	// Only all ones reach the least sum, but the greatest needs only 27 sixes,
	// since the 3 dropped dice may show anything: the sum over `k ≥ 27` of
	// `C(30, k) · 5³⁰⁻ᵏ`.
	assert_eq!(distribution.get(27), &Weight::ONE);
	assert_eq!(distribution.get(162), &Weight::from(518_526u32));
	// Reflecting every face `v` to `7 - v` exchanges the lowest dice for the
	// highest, so dropping the highest 3 mirrors dropping the lowest 3 about
	// the midpoint of the kept sums.
	let mirror = propagate_source("30D6 drop highest 3").unwrap();
	for outcome in 27..=162
	{
		assert_eq!(distribution.get(outcome), mirror.get(189 - outcome));
	}
}

/// Answer a strategy that generates the outcomes and nonzero weights of a
/// distribution over wide outcomes, in ascending order of outcome: a few
/// outcomes near some origin, perhaps with gaps, weighed from every region that
/// [weights](crate::tests::weight::value) span.
///
/// # Returns
/// The strategy.
fn wide() -> impl Strategy<Value = Vec<(i64, Weight)>>
{
	(
		-50i64..=50,
		proptest::collection::btree_map(
			0i64..=12,
			crate::tests::weight::value()
				.prop_filter("nonzero", |v| *v != num_bigint::BigUint::ZERO),
			1..=8
		)
	)
		.prop_map(|(origin, weights)| {
			weights
				.into_iter()
				.map(|(x, w)| (origin + x, Weight::from_biguint(w)))
				.collect()
		})
}

proptest! {
	/// Kronecker substitution answers the convolution that accumulating every
	/// pair of products answers, whenever the products may exceed `u128`, and
	/// declines otherwise.
	#[test]
	fn test_packed_convolution_agrees(xs in wide(), ys in wide())
	{
		let least = xs[0].0 + ys[0].0;
		let span = (xs[xs.len() - 1].0 + ys[ys.len() - 1].0 - least + 1) as u128;
		let dense = convolve_dense(&xs, &ys, least, span);
		let bits = |ws: &[(i64, Weight)]| {
			ws.iter().map(|(_, w)| w.bits()).max().unwrap()
		};
		match convolve_packed(&xs, &ys, least, span)
		{
			Some(packed) => prop_assert_eq!(packed, dense),
			None => prop_assert!(bits(&xs) + bits(&ys) <= 128)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Arithmetic.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that arithmetic on independent operands answers as expected,
/// including every edge case of the primitives.
#[test]
fn test_arithmetic()
{
	for source in [
		"1D6 + 1D6",
		"2D6 - 1D4",
		"1D6 * 1D6",
		"1D10 / 1D4",
		"[-6:6] / [-2:2]",
		"[-6:6] % [-2:2]",
		"[-2:3] ^ [-2:3]",
		"-(3D6)",
		"-(1D6) * 3",
		"(1D6 + 2) * 3 - 1D4 / 2"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
	for args in [[i32::MIN, -1], [i32::MAX, 1], [i32::MIN, i32::MAX]]
	{
		for op in OPERATORS
		{
			let source =
				format!("{{a}}, {{b}}: ({{a}} + [0:1]) {op} ({{b}} + [-1:0])");
			assert_source_agrees(&source, &args, &[]);
		}
		assert_source_agrees("{a}, {b}: -({a} + [-1:1])", &args, &[]);
	}
}

/// Ensure that an instruction whose operands are the same register, as the
/// strength reducer rewrites `x * 2` and `x ^ 2`, reads it once, as a
/// function of one value.
#[test]
fn test_same_register_operands()
{
	for source in ["3D6 * 2", "2 * 3D6", "(1D6 - 3) ^ 2", "[-3:3] ^ 2"]
	{
		let optimized = compile(source).unwrap();
		assert!(
			optimized.instructions.iter().any(|inst| {
				matches!(
					inst.sources()[..],
					[op1 @ AddressingMode::Register(_), op2] if op1 == op2
				)
			}),
			"{source}: no instruction reads one register twice\n{optimized}"
		);
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that the pass takes the [maximum](crate::Max), which only the
/// optimizer emits, of every outcome.
#[test]
fn test_max()
{
	let distribution = propagate_assembly(
		"Function() r#2 ⚅#1
	extern[]
	body:
		⚅0 <- roll range 1:6
		@0 <- sum rolling record ⚅0
		@1 <- @0 max 3
		return @1"
	);
	assert_eq!(distribution, weights(&[(3, 3), (4, 1), (5, 1), (6, 1)]));
}

/// Ensure that arguments and externals enter as fixed values, which may be
/// read any number of times.
#[test]
fn test_fixed_values()
{
	assert_source_agrees("{n}, {f}: {n}D{f} + {n} * {f}", &[3, 4], &[]);
	assert_source_agrees(
		"{x}D6 drop lowest {y} + {x}",
		&[],
		&[("x", 4), ("y", 1)]
	);
	// An unbound external is zero, though it is bounded by every `i32`, so the
	// case stays out of the corpus, whose every case is bounded tightly.
	for function in [
		compile_valid("{x}D6 + {y}"),
		compile("{x}D6 + {y}").unwrap()
	]
	{
		let evaluator = evaluator(function, &[("y", 2)]);
		assert_eq!(propagate(&evaluator, []).unwrap(), weights(&[(2, 1)]));
	}
}

/// Ensure that a random value that collapses to a point mass may be read
/// freely, as an operand of a roll or more than once, and that its branches
/// still weigh the answer.
#[test]
fn test_point_masses()
{
	// Unoptimized, the count of `(1D6 * 0 + 2)D4` rolls one die of six faces.
	let function = compile_valid("(1D6 * 0 + 2)D4");
	assert_agrees(
		function,
		&weights(&[
			(2, 6),
			(3, 12),
			(4, 18),
			(5, 24),
			(6, 18),
			(7, 12),
			(8, 6)
		])
	);
	let function = compile_valid("{x}@(1D1 * 3) + {x}D6");
	assert_agrees(
		function,
		&weights(&[
			(6, 1),
			(7, 3),
			(8, 6),
			(9, 10),
			(10, 15),
			(11, 21),
			(12, 25),
			(13, 27),
			(14, 27),
			(15, 25),
			(16, 21),
			(17, 15),
			(18, 10),
			(19, 6),
			(20, 3),
			(21, 1)
		])
	);
}

////////////////////////////////////////////////////////////////////////////////
//                           Unused and reused rolls.                         //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a roll whose value is never read still weighs the answer by
/// its branches.
#[test]
fn test_unused_rolls()
{
	// A record rolled but never summed.
	let function = Assembler::assemble(
		"Function() r#1 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 2D3
		⚅1 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅1
		return @0"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9)])
	);
	// A register summed but never read.
	let function = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll range 1:4
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D6
		@1 <- sum rolling record ⚅1
		return @1"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)])
	);
	// A random register overwritten before it is read.
	let function = Assembler::assemble(
		"Function() r#1 ⚅#2
	extern[]
	body:
		⚅0 <- roll range 1:4
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅1
		return @0"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)])
	);
}

/// Ensure that a record rolled again is replaced, as the
/// [evaluator](Evaluator) replaces it, and that the earlier roll, never
/// summed, still weighs the answer.
#[test]
fn test_rerolled_record()
{
	let function = Assembler::assemble(
		"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 2D3
		⚅0 <- drop lowest 1 from ⚅0
		⚅0 <- roll standard dice 1D2
		@0 <- sum rolling record ⚅0
		return @0"
	)
	.unwrap();
	assert_agrees(function, &weights(&[(1, 9), (2, 9)]));
}

/// Ensure that a record that is summed without being rolled sums to zero, as
/// the [evaluator](Evaluator)'s empty record does.
#[test]
fn test_unrolled_record()
{
	let distribution = propagate_assembly(
		"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- drop lowest 1 from ⚅0
		@0 <- sum rolling record ⚅0
		return @0"
	);
	assert_eq!(distribution, weights(&[(0, 1)]));
}

/// Ensure that the estimate bounds the pass over a record that is summed
/// without being rolled, whatever is dropped from it, and that the pass
/// reports its progress over it faithfully.
///
/// The bounds evaluator, on which the estimate rests, once panicked on such a
/// record.
#[test]
fn test_unrolled_record_estimate()
{
	for assembly in [
		"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- drop lowest 1 from ⚅0
		@0 <- sum rolling record ⚅0
		return @0",
		"Function() r#1 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- drop lowest @0 from ⚅1
		@0 <- sum rolling record ⚅1
		return @0"
	]
	{
		let evaluator = Evaluator::new(Assembler::assemble(assembly).unwrap());
		let (distribution, usage) =
			propagate_within(&evaluator, &[], Budget::UNLIMITED).unwrap();
		if let Err(e) = check_estimate(&evaluator, &[], &distribution, usage)
			.map(|_| ())
			.and_then(|()| {
				check_exact_budget(&evaluator, &[], &distribution, usage)
			})
			.and_then(|()| {
				check_progress(
					&evaluator,
					&[],
					&distribution,
					usage,
					usize::MAX
				)
			})
		{
			panic!("{assembly}: {e}");
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Mixtures.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that dice of random counts answer as expected, weight for weight,
/// including counts that are not positive, custom faces, and drops.
#[test]
fn test_dynamic_counts()
{
	// In units of 1/81: one die weighs 9 per outcome, two dice 3 per roll, and
	// three dice 1 per roll.
	assert_eq!(
		propagate_source("(1D3)D3").unwrap(),
		weights(&[
			(1, 9),
			(2, 9 + 3),
			(3, 9 + 6 + 1),
			(4, 9 + 3),
			(5, 6 + 6),
			(6, 3 + 7),
			(7, 6),
			(8, 3),
			(9, 1)
		])
	);
	for source in [
		"(1D3)D3",
		"(2D3)D4",
		"(1D4 - 2)D6",
		"(1D3)D[1, 1, 2]",
		"(1D3)D3 drop lowest",
		"(1D4)D4 drop highest 2",
		"(1D3)D[-2147483648, 2147483647]"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that dice of random faces answer as expected, weight for weight,
/// including dice without faces.
#[test]
fn test_dynamic_faces()
{
	for source in ["1D(1D3)", "3D(1D2)", "2D(1D4 - 2)", "(1D2)D(1D3)"]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that random drop counts answer as expected, weight for weight,
/// including counts that are negative or exceed the dice, and records that are
/// dropped but never rolled.
#[test]
fn test_dynamic_drops()
{
	for source in [
		"3D6 drop lowest (1D2)",
		"(1D3)D4 drop lowest (1D2)",
		"4D4 drop highest (1D3 - 2)",
		"3D6 drop lowest (1D2) drop highest (1D2)",
		"2D4 drop lowest (1D4)"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
	let function = Assembler::assemble(
		"Function() r#1 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- drop lowest @0 from ⚅1
		@0 <- sum rolling record ⚅1
		return @0"
	)
	.unwrap();
	assert_agrees(function, &weights(&[(0, 3)]));
}

/// Ensure that keeps answer as expected, weight for weight: the highest of
/// `2D6`, whose outcome `k` arises `2k - 1` ways; a keep that counts from the
/// dice that a drop left, rather than from all of them; keeps that keep more
/// dice than remain, or none; and a keep whose count is random.
#[test]
fn test_keeps()
{
	let program = |clauses: &str| {
		format!(
			"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 4D4
{clauses}		@0 <- sum rolling record ⚅0
		return @0"
		)
	};
	assert_eq!(
		propagate_assembly(
			"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 2D6
		⚅0 <- keep highest 1 from ⚅0
		@0 <- sum rolling record ⚅0
		return @0"
		),
		weights(&[(1, 1), (2, 3), (3, 5), (4, 7), (5, 9), (6, 11)])
	);
	// Keeping the lowest 2 of the 3 dice that remain after the lowest is
	// dropped keeps the second and third lowest, unlike dropping 2 more of
	// the highest, which keeps only the second lowest.
	for (clauses, reference) in [
		(
			"		⚅0 <- drop lowest 1 from ⚅0
		⚅0 <- keep lowest 2 from ⚅0
",
			"		⚅0 <- drop lowest 1 from ⚅0
		⚅0 <- drop highest 1 from ⚅0
"
		),
		(
			"		⚅0 <- keep highest 3 from ⚅0
		⚅0 <- keep highest 2 from ⚅0
",
			"		⚅0 <- drop lowest 2 from ⚅0
"
		),
		(
			"		⚅0 <- keep lowest 7 from ⚅0
",
			""
		),
		(
			"		⚅0 <- keep highest 0 from ⚅0
",
			"		⚅0 <- drop lowest 4 from ⚅0
"
		),
		(
			"		⚅0 <- keep highest -2 from ⚅0
",
			"		⚅0 <- drop highest 4 from ⚅0
"
		)
	]
	{
		assert_eq!(
			propagate_assembly(&program(clauses)),
			propagate_assembly(&program(reference)),
			"{clauses}"
		);
	}
	// Keeping 1D2 of the highest is dropping the lowest 3 or 2, equally.
	let random = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll range 1:2
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 4D4
		⚅1 <- keep highest @0 from ⚅1
		@1 <- sum rolling record ⚅1
		return @1"
	)
	.unwrap();
	let one = propagate_assembly(&program(
		"		⚅0 <- keep highest 1 from ⚅0
"
	));
	let two = propagate_assembly(&program(
		"		⚅0 <- keep highest 2 from ⚅0
"
	));
	let expected = Distribution::from_weights(
		(1..=8).map(|outcome| (outcome, one.get(outcome) + two.get(outcome)))
	)
	.unwrap();
	assert_agrees(random, &expected);
}

/// Ensure that ranges of random endpoints answer as expected, weight for
/// weight, including empty ranges.
#[test]
fn test_dynamic_ranges()
{
	for source in [
		"[1:(1D6)]",
		"[(1D3):(1D3 + 3)]",
		"[[1:3]:[3:6]]",
		"[(1D4):2]",
		"[(1D3 - 2):(1D3 * 2)] + 1D4"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that a roll whose operands are one random register reads it once,
/// as a function of one value.
#[test]
fn test_same_register_mixtures()
{
	// The optimizer reduces `[x:x]` to `x`, but not before optimization.
	for source in ["({x}@(1D3))D{x}", "[{x}@(1D4 - 2):{x}]"]
	{
		let unoptimized = compile_valid(source);
		assert!(
			unoptimized.instructions.iter().any(|inst| {
				matches!(
					inst.sources()[..],
					[op1 @ AddressingMode::Register(_), op2] if op1 == op2
				)
			}),
			"{source}: no instruction reads one register twice\n{unoptimized}"
		);
		assert_source_agrees(source, &[], &[]);
	}
}

/// Ensure that a mixture whose record is never summed still weighs the answer
/// by the branches of its roll and of the operands that chose it.
#[test]
fn test_unused_mixtures()
{
	let function = Assembler::assemble(
		"Function() r#1 ⚅#3
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice @0D2
		⚅2 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅2
		return @0"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(1, 24), (2, 24), (3, 24), (4, 24), (5, 24), (6, 24)])
	);
}

/// Ensure that the pass answers random counts far beyond the reach of
/// enumerating their paths, stepping up from one convolution power to the next.
#[test]
fn test_large_mixtures()
{
	let distribution = propagate_source("(10D10)D10").unwrap();
	let d10 = Weight::from(10u8);
	// The common denominator of 10ᵏ dice over `10 ≤ k ≤ 100` is 10¹⁰⁰.
	assert_eq!(distribution.total(), &(d10.pow(10) * d10.pow(100)));
	assert_eq!((distribution.min(), distribution.max()), (10, 1000));
	// Only ten ones roll ten dice of ten ones.
	assert_eq!(distribution.get(10), &d10.pow(90));
	// By Wald's identity, the mean is 55 · 5.5.
	let mean = Rational::new(false, Weight::from(605u16), Weight::from(2u8));
	assert_eq!(distribution.mean(), mean.unwrap());
}

/// Ensure that a mixture whose setups are chosen along paths of unequal
/// probability answers as expected in probability, though its common
/// denominator is a multiple of the total of the paths.
#[test]
fn test_unequal_paths()
{
	// Every path weighs 1/4 or 1/8, so the paths total 8; but
	// the range weighs its outcomes over 8, and the outer die has 2 faces for
	// some of them, so the mixture's total is 16.
	let propagated = assert_agrees_in_probability(
		compile("1D(2 - [(1D2):(1D2)])").unwrap(),
		&weights(&[(0, 3), (1, 4), (2, 1)])
	);
	assert_eq!(propagated.total(), &Weight::from(16u8));
}

////////////////////////////////////////////////////////////////////////////////
//                                   Plans.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Answer the steps that the plan of the specified assembly schedules after
/// each instruction that has any.
///
/// # Parameters
/// - `assembly`: The text form of a function.
///
/// # Returns
/// The program counters of the instructions, each with its steps.
///
/// # Panics
/// If the assembly is malformed.
fn schedule(assembly: &str) -> Vec<(usize, Vec<Step>)>
{
	let function = Assembler::assemble(assembly).unwrap();
	let plan = Plan::new(&function.instructions);
	(0..function.instructions.len())
		.map(|pc| (pc, plan.steps(ProgramCounter(pc)).to_vec()))
		.filter(|(_, steps)| !steps.is_empty())
		.collect()
}

/// Answer the location of the specified register.
///
/// # Parameters
/// - `index`: The index of the register.
///
/// # Returns
/// The location.
fn register(index: usize) -> Location
{
	Location::Register(RegisterIndex(index))
}

/// Answer the location of the specified rolling record.
///
/// # Parameters
/// - `index`: The index of the rolling record.
///
/// # Returns
/// The location.
fn record(index: usize) -> Location
{
	Location::RollingRecord(RollingRecordIndex(index))
}

/// Ensure that a plan splits on a random value that two instructions read,
/// right after it is written, and merges it once one value carries its
/// influence; and that it plans nothing for a value that one instruction
/// reads twice, or for a fixed value.
#[test]
fn test_plan_fan_out()
{
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 * 3
		@2 <- @0 + @1
		return @2"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(
				3,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(2)),
					dead: vec![register(0), register(1)]
				}]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#2 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + @0
		return @1"
		),
		vec![]
	);
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#0
	extern[]
	body:
		@0 <- 1 + 2
		@1 <- @0 * 3
		@2 <- @0 + @1
		return @2"
		),
		vec![]
	);
}

/// Ensure that a plan follows values rather than registers, as register
/// coalescence reuses them.
#[test]
fn test_plan_coalesced_registers()
{
	assert_eq!(
		schedule(
			"Function() r#2 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 * 3
		@0 <- @0 + @1
		return @0"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(
				3,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(0)),
					dead: vec![register(1)]
				}]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@0 <- @0 + @0
		return @0"
		),
		vec![]
	);
}

/// Ensure that a plan splits on a rolling record that two sums read, even
/// across a drop, and never merges into a rolling record, whose influence is
/// not yet one value.
#[test]
fn test_plan_rolling_records()
{
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 + @1
		return @2"
		),
		vec![
			(0, vec![Step::Split(record(0))]),
			(
				3,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(2)),
					dead: vec![record(0), register(0), register(1)]
				}]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		⚅0 <- drop lowest 1 from ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 - @1
		return @2"
		),
		vec![
			(0, vec![Step::Split(record(0))]),
			(
				4,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(2)),
					dead: vec![register(0), record(0), register(1)]
				}]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D4
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice @0D6
		⚅1 <- drop lowest @0 from ⚅1
		@1 <- sum rolling record ⚅1
		return @1"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(
				4,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(1)),
					dead: vec![register(0), record(1)]
				}]
			),
		]
	);
}

/// Ensure that a split merges beneath a later split that is independent of
/// it, but waits for a later split that depends on it to merge first.
#[test]
fn test_plan_nested_splits()
{
	assert_eq!(
		schedule(
			"Function() r#5 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D4
		@1 <- sum rolling record ⚅1
		@2 <- @0 * @1
		@3 <- @0 + @2
		@4 <- @1 + @3
		return @4"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(3, vec![Step::Split(register(1))]),
			(
				5,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(3)),
					dead: vec![register(0), register(2)]
				}]
			),
			(
				6,
				vec![Step::Merge {
					split: 0,
					survivor: Some(register(4)),
					dead: vec![register(1), register(2), register(3)]
				}]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#5 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * 2
		@3 <- @1 + 1
		@4 <- @1 * @3
		return @4"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(2, vec![Step::Split(register(1))]),
			(
				5,
				vec![
					Step::Merge {
						split: 1,
						survivor: Some(register(4)),
						dead: vec![register(1), register(3)]
					},
					Step::Merge {
						split: 0,
						survivor: Some(register(4)),
						dead: vec![
							register(0),
							register(1),
							register(2),
							register(3)
						]
					},
				]
			),
		]
	);
}

/// Ensure that a plan merges every split by the last instruction, into the
/// answer if the answer is the only value that depends on it, or into nothing
/// if no live value does.
#[test]
fn test_plan_final_merges()
{
	// The split on @1 may not merge into itself, and the split on @0 must wait
	// for it, until the answer carries both.
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * @1
		return @1"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(2, vec![Step::Split(register(1))]),
			(
				4,
				vec![
					Step::Merge {
						split: 1,
						survivor: Some(Location::Answer),
						dead: vec![register(1), register(2)]
					},
					Step::Merge {
						split: 0,
						survivor: Some(Location::Answer),
						dead: vec![register(0), register(1), register(2)]
					},
				]
			),
		]
	);
	assert_eq!(
		schedule(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * 2
		return 0"
		),
		vec![
			(1, vec![Step::Split(register(0))]),
			(
				3,
				vec![Step::Merge {
					split: 0,
					survivor: None,
					dead: vec![register(0), register(1), register(2)]
				}]
			),
		]
	);
}

/// Ensure that the plans of compiled programs split on exactly the random
/// values that more than one instruction reads.
#[test]
fn test_plan_compiled_programs()
{
	let splits = |source| {
		let function = compile(source).unwrap();
		let plan = Plan::new(&function.instructions);
		(0..function.instructions.len())
			.flat_map(|pc| plan.steps(ProgramCounter(pc)))
			.filter(|step| matches!(step, Step::Split(_)))
			.count()
	};
	assert_eq!(splits("3D6"), 0);
	assert_eq!(splits("{x}@(3D6) + {x}"), 0);
	assert_eq!(splits("{x}@(3D6) + {x} * 3"), 1);
	assert_eq!(splits("{x}@(1D6) * ({x} + 1)"), 1);
	assert_eq!(splits("{x}@(1D6) + {y}@(1D4) + {x} * {y} + {x} - {y}"), 2);
}

////////////////////////////////////////////////////////////////////////////////
//                               Conditioning.                                //
////////////////////////////////////////////////////////////////////////////////

/// Answer the most worlds that the pass keeps at once for the specified
/// source, optimized.
///
/// # Parameters
/// - `source`: The source, which takes no arguments.
///
/// # Returns
/// The most worlds alive at once.
///
/// # Panics
/// If the source does not compile, or the pass refuses it.
fn worlds(source: &str) -> u64
{
	propagate_within(
		&Evaluator::new(compile(source).unwrap()),
		&[],
		Budget::UNLIMITED
	)
	.unwrap()
	.1
	.worlds
}

/// Ensure that the pass conditions on random values that more than one
/// instruction reads, answering them weight for weight, optimized and not.
#[test]
fn test_fan_out()
{
	for source in [
		"{x}@(3D6) + {x} * 3",
		"{x}@(1D6) * ({x} + 1)",
		"{x}@(2D6) + {x} + {x}",
		"{x}@(1D6) + {y}@(1D4) + {x} * {y} + {x} - {y}",
		"{x}@(1D6) + {y}@({x} + 1D4) * {y} - {x}",
		"{x}@(1D1) + {x} * 2",
		"{x}@(3D6 drop lowest) - {x} / 2 + 1D4"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
	assert_source_agrees("{a}: {x}@({a}D6) + {x} * {a}", &[2], &[]);
	assert_source_agrees("{x}@(1D6 + {e}) * {x}", &[], &[("e", 3)]);
}

/// Ensure that the pass conditions on random values that choose the operands
/// of rolls, answering as expected.
#[test]
fn test_dynamic_fan_out()
{
	for source in [
		"({x}@(1D3))D6 + {x}",
		"({x}@(1D3))D6 drop lowest {x}",
		"[{x}@(1D4):({x} + 1D2)] * {x}"
	]
	{
		assert_source_agrees(source, &[], &[]);
	}
	let function = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D4
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice @0D6
		⚅1 <- drop lowest @0 from ⚅1
		@1 <- sum rolling record ⚅1
		return @1"
	)
	.unwrap();
	assert_agrees_in_probability(function, &weights(&[(0, 5184)]));
}

/// Ensure that the pass conditions on values rather than registers, as
/// register coalescence reuses them, and merges splits in the order that the
/// plan schedules: beneath a later, independent split, after a later,
/// dependent one, and into the answer or into nothing.
#[test]
fn test_merges()
{
	for (assembly, expected) in [
		// Coalesced registers.
		(
			"Function() r#2 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 * 3
		@0 <- @0 + @1
		return @0",
			weights(&[
				(12, 1),
				(16, 3),
				(20, 6),
				(24, 10),
				(28, 15),
				(32, 21),
				(36, 25),
				(40, 27),
				(44, 27),
				(48, 25),
				(52, 21),
				(56, 15),
				(60, 10),
				(64, 6),
				(68, 3),
				(72, 1)
			])
		),
		// A split merges beneath a later, independent one.
		(
			"Function() r#5 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D4
		@1 <- sum rolling record ⚅1
		@2 <- @0 * @1
		@3 <- @0 + @2
		@4 <- @1 + @3
		return @4",
			weights(&[
				(3, 1),
				(5, 2),
				(7, 2),
				(8, 1),
				(9, 2),
				(11, 3),
				(13, 1),
				(14, 2),
				(15, 1),
				(17, 1),
				(19, 2),
				(20, 1),
				(23, 1),
				(24, 1),
				(27, 1),
				(29, 1),
				(34, 1)
			])
		),
		// A split waits for a later, dependent one.
		(
			"Function() r#5 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * 2
		@3 <- @1 + 1
		@4 <- @1 * @3
		return @4",
			weights(&[(6, 1), (12, 1), (20, 1), (30, 1), (42, 1), (56, 1)])
		),
		// Both splits merge into the answer.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * @1
		return @1",
			weights(&[(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])
		),
		// The split merges into nothing.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- @0 + 1
		@2 <- @0 * 2
		return 0",
			weights(&[(0, 6)])
		)
	]
	{
		assert_agrees(Assembler::assemble(assembly).unwrap(), &expected);
	}
}

/// Ensure that a merge folds into each world the weight of the dead values
/// that depend on the split but were never read: a register, and a record
/// whose roll the split chose.
#[test]
fn test_dead_dependents()
{
	let function = Assembler::assemble(
		"Function() r#5 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D4
		@1 <- sum rolling record ⚅1
		@2 <- @0 + @1
		@3 <- @0 * 3
		@4 <- @0 + @3
		return @4"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(4, 4), (8, 4), (12, 4), (16, 4), (20, 4), (24, 4)])
	);
	let function = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice @0D2
		@1 <- @0 * 3
		@0 <- @0 + @1
		return @0"
	)
	.unwrap();
	assert_agrees_in_probability(
		function,
		&weights(&[(4, 8), (8, 8), (12, 8)])
	);
}

/// Ensure that the pass keeps one world for each outcome of each open split,
/// and no more.
#[test]
fn test_worlds()
{
	assert_eq!(worlds("3D6"), 1);
	assert_eq!(worlds("{x}@(3D6) + {x}"), 1);
	assert_eq!(worlds("{x}@(3D6) + {x} * 3"), 16);
	assert_eq!(worlds("{x}@(1D6) + {y}@(1D4) + {x} * {y} + {x} - {y}"), 24);
}

/// Ensure that the pass conditions on the results of a rolling record read more
/// than once, across drops and before and after them, over every kind of roll,
/// answering weight for weight where no roll or drop has a random operand, and
/// keeping one world for each multiset of results.
#[test]
fn test_record_fan_out()
{
	for (assembly, expected) in [
		// Two sums, one of them dead.
		(
			"Function() r#2 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D6
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		return @1",
			weights(&[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)])
		),
		// Two sums that cancel.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 - @1
		return @2",
			weights(&[(0, 216)])
		),
		// A sum before a drop and a sum after it.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		⚅0 <- drop lowest 1 from ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 - @1
		return @2",
			weights(&[(1, 91), (2, 61), (3, 37), (4, 19), (5, 7), (6, 1)])
		),
		// The value that a drop writes fans out.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 4D6
		⚅0 <- drop lowest 1 from ⚅0
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 * @1
		return @2",
			weights(&[
				(9, 1),
				(16, 4),
				(25, 10),
				(36, 21),
				(49, 38),
				(64, 62),
				(81, 91),
				(100, 122),
				(121, 148),
				(144, 167),
				(169, 172),
				(196, 160),
				(225, 131),
				(256, 94),
				(289, 54),
				(324, 21)
			])
		),
		// A drop after the last sum, which no sum reads.
		(
			"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 2D6
		@0 <- sum rolling record ⚅0
		⚅0 <- drop highest 1 from ⚅0
		return @0",
			weights(&[
				(2, 1),
				(3, 2),
				(4, 3),
				(5, 4),
				(6, 5),
				(7, 6),
				(8, 5),
				(9, 4),
				(10, 3),
				(11, 2),
				(12, 1)
			])
		),
		// Custom dice with repeated faces, which saturate.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll custom dice 3D[1, 1, -2, 2147483647]
		@0 <- sum rolling record ⚅0
		⚅0 <- drop highest 1 from ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 + @1
		return @2",
			weights(&[
				(-10, 1),
				(-7, 6),
				(-1, 12),
				(5, 8),
				(2147483639, 3),
				(2147483645, 12),
				(2147483647, 22)
			])
		),
		// A range, and an empty range.
		(
			"Function() r#5 ⚅#2
	extern[]
	body:
		⚅0 <- roll range 1:4
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		⚅1 <- roll range 4:1
		@2 <- sum rolling record ⚅1
		@3 <- sum rolling record ⚅1
		@4 <- @0 * @1
		@4 <- @4 + @2
		@4 <- @4 + @3
		return @4",
			weights(&[(1, 1), (4, 1), (9, 1), (16, 1)])
		),
		// No dice, and dice without faces.
		(
			"Function() r#5 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 0D6
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 3D0
		⚅1 <- drop lowest 1 from ⚅1
		@2 <- sum rolling record ⚅1
		@3 <- sum rolling record ⚅1
		@4 <- @0 + @1
		@4 <- @4 + @2
		@4 <- @4 + @3
		return @4",
			weights(&[(0, 1)])
		),
		// A record never rolled.
		(
			"Function() r#3 ⚅#1
	extern[]
	body:
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 + @1
		return @2",
			weights(&[(0, 1)])
		),
		// A split record beneath a split register, and beneath another record.
		(
			"Function() r#6 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 2D4
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 2D3
		@1 <- sum rolling record ⚅1
		@2 <- sum rolling record ⚅1
		@3 <- @1 * @2
		@4 <- sum rolling record ⚅0
		@5 <- @0 * @4
		@5 <- @5 + @3
		@5 <- @5 + @1
		return @5",
			weights(&[
				(10, 1),
				(15, 2),
				(16, 2),
				(21, 4),
				(22, 3),
				(24, 3),
				(28, 6),
				(29, 6),
				(31, 4),
				(34, 2),
				(36, 9),
				(37, 8),
				(39, 4),
				(42, 3),
				(45, 12),
				(46, 7),
				(48, 6),
				(51, 2),
				(55, 10),
				(56, 9),
				(58, 3),
				(61, 4),
				(66, 6),
				(67, 4),
				(69, 6),
				(70, 1),
				(76, 2),
				(78, 3),
				(79, 4),
				(84, 3),
				(91, 2),
				(94, 2),
				(106, 1)
			])
		)
	]
	{
		assert_agrees(Assembler::assemble(assembly).unwrap(), &expected);
	}
	for (assembly, expected) in [
		// A random count and a random drop before the split.
		(
			"Function() r#4 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice @0D4
		⚅1 <- drop highest @0 from ⚅1
		@1 <- sum rolling record ⚅1
		@2 <- sum rolling record ⚅1
		@3 <- @1 * @2
		return @3",
			weights(&[(0, 192)])
		),
		// Random drops after the split, the last of which no sum reads.
		(
			"Function() r#3 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D3
		@1 <- sum rolling record ⚅1
		⚅0 <- drop lowest @1 from ⚅0
		@2 <- sum rolling record ⚅0
		@2 <- @0 - @2
		⚅0 <- drop highest @1 from ⚅0
		return @2",
			weights(&[
				(1, 91),
				(2, 77),
				(3, 65),
				(4, 56),
				(5, 49),
				(6, 45),
				(7, 42),
				(8, 40),
				(9, 37),
				(10, 34),
				(11, 30),
				(12, 26),
				(13, 21),
				(14, 15),
				(15, 10),
				(16, 6),
				(17, 3),
				(18, 1)
			])
		),
		// Setups that reach the same results, no dice and fewer than none,
		// above a split that merges beneath them.
		(
			"Function() r#7 ⚅#3
	extern[]
	body:
		⚅0 <- roll standard dice 1D2
		@0 <- sum rolling record ⚅0
		⚅1 <- roll custom dice 1D[0, -1, -1]
		@1 <- sum rolling record ⚅1
		⚅2 <- roll standard dice @1D0
		@2 <- sum rolling record ⚅2
		@3 <- sum rolling record ⚅2
		@4 <- @0 + 1
		@5 <- @0 * @4
		@6 <- @2 + @3
		@6 <- @6 + @5
		return @6",
			weights(&[(2, 3), (6, 3)])
		)
	]
	{
		assert_agrees_in_probability(
			Assembler::assemble(assembly).unwrap(),
			&expected
		);
	}
	// Each multiset of `3D6` is a world.
	let function = Assembler::assemble(
		"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 * @1
		return @2"
	)
	.unwrap();
	assert_eq!(
		propagate_within(&Evaluator::new(function), &[], Budget::UNLIMITED)
			.unwrap()
			.1
			.worlds,
		56
	);
	// Many dice with one face, or none, and no dice with many faces, have one
	// outcome, which takes little space.
	assert_eq!(
		propagate_assembly(
			"Function() r#7 ⚅#3
	extern[]
	body:
		⚅0 <- roll standard dice 2147483647D1
		@0 <- sum rolling record ⚅0
		⚅0 <- drop lowest 1 from ⚅0
		@1 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 2147483647D0
		@2 <- sum rolling record ⚅1
		@3 <- sum rolling record ⚅1
		⚅2 <- roll standard dice -1D2147483647
		@5 <- sum rolling record ⚅2
		@6 <- sum rolling record ⚅2
		@4 <- @0 - @1
		@4 <- @4 + @2
		@4 <- @4 + @3
		@4 <- @4 + @5
		@4 <- @4 + @6
		return @4"
		),
		weights(&[(1, 1)])
	);
}

/// Ensure that the pass conditions weight for weight on a random drop that no
/// sum reads, folding in the weight of its count: a drop writes a new value,
/// even after a sum has read the record.
#[test]
fn test_unread_drop_after_sum()
{
	let function = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 3D6
		@0 <- sum rolling record ⚅0
		⚅1 <- roll standard dice 1D3
		@1 <- sum rolling record ⚅1
		⚅0 <- drop lowest @1 from ⚅0
		return @0"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[
			(3, 3),
			(4, 9),
			(5, 18),
			(6, 30),
			(7, 45),
			(8, 63),
			(9, 75),
			(10, 81),
			(11, 81),
			(12, 75),
			(13, 63),
			(14, 45),
			(15, 30),
			(16, 18),
			(17, 9),
			(18, 3)
		])
	);
}

/// Ensure that the pass refuses arguments that disagree with the parameters.
#[test]
fn test_refusals()
{
	assert_eq!(
		propagate(&Evaluator::new(compile("{a}: {a}D6").unwrap()), []),
		Err(PropagationError::BadArity {
			expected: 1,
			given: 0
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                  Budgets.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A budget ample for any small expression, and small enough that no
/// operation that it admits takes noticeable time or memory.
const SMALL: Budget = Budget {
	steps: 10_000_000,
	cells: 1_000_000
};

/// Answer how the pass refuses the specified source, with the specified
/// arguments, within the specified budget.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments.
/// - `budget`: The budget.
///
/// # Returns
/// The refusal.
///
/// # Panics
/// If the source does not compile, or the pass does not exhaust the budget.
fn exhaust(source: &str, args: &[i32], budget: Budget) -> Exhausted
{
	let evaluator = Evaluator::new(compile(source).unwrap());
	match propagate_within(&evaluator, args, budget)
	{
		Err(PropagationError::Exhausted(exhausted)) => exhausted,
		other => panic!("{source} {args:?}: expected a refusal: {other:?}")
	}
}

/// Check that the pass succeeds within a budget exactly when the budget
/// covers its usage: within the usage itself, it answers the same
/// distribution with the same usage, and within one step or one cell less, it
/// refuses, naming the dimension that it lacks.
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `expected`: The distribution that the pass answers within the unlimited
///   budget.
/// - `usage`: The usage of the pass within the unlimited budget.
///
/// # Errors
/// A description of the first failure.
fn check_exact_budget(
	evaluator: &Evaluator,
	args: &[i32],
	expected: &Distribution,
	usage: Usage
) -> Result<(), String>
{
	let exact = Budget {
		steps: usage.steps,
		cells: usage.cells
	};
	let run = |budget| propagate_within(evaluator, args, budget);
	match run(exact)
	{
		Ok((distribution, again))
			if distribution == *expected && again == usage =>
		{},
		other =>
		{
			return Err(format!(
				"within its usage {usage:?}: {other:?}\nfunction:\n{}",
				evaluator.function
			))
		},
	}
	let short = [
		(
			Dimension::Steps,
			Budget {
				steps: exact.steps.wrapping_sub(1),
				..exact
			}
		),
		(
			Dimension::Cells,
			Budget {
				cells: exact.cells - 1,
				..exact
			}
		)
	];
	for (dimension, budget) in short
	{
		if budget.steps == u64::MAX
		{
			// The pass took no steps, so none can be withheld.
			continue
		}
		match run(budget)
		{
			Err(PropagationError::Exhausted(exhausted))
				if exhausted.dimension == dimension =>
			{},
			other =>
			{
				return Err(format!(
					"within {budget:?}, short of {usage:?}: {other:?}\n\
					function:\n{}",
					evaluator.function
				))
			},
		}
	}
	Ok(())
}

/// Check that the estimate of the pass bounds what the pass consumes: its
/// steps, its peak cells, and its most worlds bound those of its usage, and
/// its bits bound those of the total weight of its answer.
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `distribution`: The distribution that the pass answers.
/// - `usage`: The usage of the pass within the unlimited budget.
///
/// # Returns
/// The estimate.
///
/// # Errors
/// A description of the first bound that the pass exceeds.
fn check_estimate(
	evaluator: &Evaluator,
	args: &[i32],
	distribution: &Distribution,
	usage: Usage
) -> Result<Cost, String>
{
	let cost = estimate(evaluator, args.iter().copied())
		.map_err(|e| format!("no estimate: {e:?}"))?;
	let bits = distribution.total().bits();
	if cost.steps < usage.steps
		|| cost.cells < usage.cells
		|| cost.worlds < usage.worlds
		|| cost.bits < bits
	{
		return Err(format!(
			"{cost:?} does not bound {usage:?} and {bits} bits\nfunction:\n{}",
			evaluator.function
		))
	}
	Ok(cost)
}

/// Ensure that a roll of many dice is refused before it expands anything,
/// however it is summed, dropped, or left unread, since it charges its dice
/// when it fills its record.
#[test]
fn test_many_dice()
{
	for source in [
		"{n}: {n}D6",
		"{n}: {n}D6 drop lowest {n}",
		"{n}: {n}D[1, 5, 9] drop highest 2"
	]
	{
		let exhausted = exhaust(source, &[i32::MAX], SMALL);
		assert_eq!(exhausted.dimension, Dimension::Steps, "{source}");
		assert!(exhausted.requested >= i32::MAX as u64, "{source}");
		assert!(exhausted.consumed < 100, "{source}: {exhausted:?}");
	}
}

/// Ensure that rolls whose counts or faces are computed are refused before
/// they expand anything, even though bounds analysis cannot count their
/// outcomes.
#[test]
fn test_computed_counts()
{
	for (source, args) in [
		("([1:2147483647])D6", &[][..]),
		("{n}: ({n}D6)D6", &[1_000_000]),
		("{n}: 1D({n}D6)", &[1_000_000]),
		("(100D100)D100", &[])
	]
	{
		let exhausted = exhaust(source, args, SMALL);
		assert!(
			exhausted.requested > exhausted.remaining,
			"{source}: {exhausted:?}"
		);
	}
}

/// Ensure that wide arithmetic is refused before it pairs its operands.
#[test]
fn test_wide_arithmetic()
{
	let exhausted = exhaust(
		"[1:1000000] * [1:1000000]",
		&[],
		Budget {
			steps: 100_000_000,
			cells: 10_000_000
		}
	);
	assert_eq!(exhausted.dimension, Dimension::Steps);
	assert_eq!(exhausted.requested, 1_000_000_000_000);
}

/// Ensure that a budget of cells bounds the memory of a pass that takes few
/// steps: the outcomes of a sum, and the worlds of nested splits.
#[test]
fn test_cells()
{
	let budget = Budget {
		steps: u64::MAX - 1,
		cells: 100
	};
	for source in ["100D6", "{a}@(1D100) * {a} + 2 * {a}"]
	{
		let exhausted = exhaust(source, &[], budget);
		assert_eq!(exhausted.dimension, Dimension::Cells, "{source}");
		assert!(exhausted.consumed <= budget.cells, "{source}");
	}
}

/// Ensure that the pass succeeds within a budget exactly when the budget
/// covers its usage, on every case of the distribution corpus.
#[test]
fn test_exact_budgets()
{
	for (index, (source, args, externs, _)) in
		read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE)
			.iter()
			.enumerate()
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = evaluator(function, externs);
		let (distribution, usage) =
			propagate_within(&evaluator, args, Budget::UNLIMITED).unwrap();
		if let Err(e) =
			check_exact_budget(&evaluator, args, &distribution, usage).and_then(
				|()| check_estimate(&evaluator, args, &distribution, usage)
			)
		{
			panic!("case {}: {}: {}", index + 1, source, e);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Progress.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The most reports of one pass of the corpus that [`check_progress`]
/// cancels at.
const CANCELLATIONS: usize = 32;

/// The most reports of one pass of the property tests that [`check_progress`]
/// cancels at: the first, the last, and one between, since the property tests
/// run many passes.
const PROPERTY_CANCELLATIONS: usize = 3;

/// Run the pass over the specified evaluator and arguments within the
/// unlimited budget, recording every report, and cancelling the pass at the
/// report of the specified index, if any.
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `cancel`: The index of the report whose recipient cancels the pass, if
///   any.
///
/// # Returns
/// The answer of the pass, and its reports, in order: the usage so far and
/// the estimate of each.
///
/// # Panics
/// If the pass cannot be estimated.
fn report(
	evaluator: &Evaluator,
	args: &[i32],
	cancel: Option<usize>
) -> (Answer, Vec<(Usage, Cost)>)
{
	let cost = estimate(evaluator, args.iter().copied()).unwrap();
	let reports = RefCell::new(Vec::new());
	let progress = |consumed: Usage, estimate: &Cost| {
		let mut reports = reports.borrow_mut();
		reports.push((consumed, *estimate));
		if Some(reports.len() - 1) == cancel
		{
			ControlFlow::Break(())
		}
		else
		{
			ControlFlow::Continue(())
		}
	};
	let answer = propagate_worlds(
		evaluator,
		args.iter().copied(),
		Budget::UNLIMITED,
		&cost,
		&progress
	);
	(answer, reports.into_inner())
}

/// Check that the pass reports its progress faithfully, and cancels cleanly:
/// it reports its own estimate every time, and a usage that never decreases,
/// in any dimension, and whose last is the usage that it answers; and a
/// recipient that cancels the pass at any report makes it answer
/// [`Cancelled`](PropagationError::Cancelled), after reporting exactly what
/// it reported up to that report when not cancelled, and nothing more.
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `expected`: The distribution that the pass answers within the unlimited
///   budget.
/// - `usage`: The usage of the pass within the unlimited budget.
/// - `most`: The most reports to cancel at, each in a pass of its own; if the
///   pass reports more, it is cancelled at reports spread evenly from its first
///   to its last, which is at least two.
///
/// # Errors
/// A description of the first report or cancellation that is amiss.
fn check_progress(
	evaluator: &Evaluator,
	args: &[i32],
	expected: &Distribution,
	usage: Usage,
	most: usize
) -> Result<(), String>
{
	let function = &evaluator.function;
	let cost = estimate(evaluator, args.iter().copied())
		.map_err(|e| format!("no estimate: {e:?}"))?;
	let (answer, reports) = report(evaluator, args, None);
	match answer
	{
		Ok((distribution, again))
			if distribution == *expected && again == usage =>
		{},
		other =>
		{
			return Err(format!("observed: {other:?}\nfunction:\n{function}"))
		},
	}
	if let Some((_, estimate)) =
		reports.iter().find(|(_, estimate)| *estimate != cost)
	{
		return Err(format!(
			"reported {estimate:?}, not {cost:?}\nfunction:\n{function}"
		))
	}
	if let Some(pair) = reports.windows(2).find(|pair| {
		let ((before, _), (after, _)) = (pair[0], pair[1]);
		after.steps < before.steps
			|| after.cells < before.cells
			|| after.worlds < before.worlds
	})
	{
		return Err(format!(
			"the usage decreased: {:?} then {:?}\nfunction:\n{function}",
			pair[0].0, pair[1].0
		))
	}
	match reports.last()
	{
		Some((last, _)) if *last == usage =>
		{},
		last =>
		{
			return Err(format!(
				"the last report {last:?} is not the usage {usage:?}\n\
				function:\n{function}"
			))
		},
	}
	let count = reports.len();
	let cancellations = if count <= most
	{
		(0..count).collect::<Vec<_>>()
	}
	else
	{
		(0..most).map(|i| i * (count - 1) / (most - 1)).collect()
	};
	for cancel in cancellations
	{
		let (answer, prefix) = report(evaluator, args, Some(cancel));
		if !matches!(answer, Err(PropagationError::Cancelled))
		{
			return Err(format!(
				"cancelled at report {cancel} of {count}: {answer:?}\n\
				function:\n{function}"
			))
		}
		if prefix != reports[..=cancel]
		{
			return Err(format!(
				"cancelled at report {cancel} of {count}, reported {:?}, not \
				{:?}\nfunction:\n{function}",
				prefix,
				&reports[..=cancel]
			))
		}
	}
	Ok(())
}

/// Ensure that the pass reports its progress faithfully, and cancels cleanly
/// at every report, on functions that roll, drop, mix setups, and split and
/// merge worlds.
#[test]
fn test_progress()
{
	for source in [
		"3",
		"3D6",
		"4D6 drop lowest 1",
		"(1D6)D6 drop lowest (1D3)",
		"[(1D10):(1D10 + 10)]",
		"1D6 * 1D8",
		"{x}@(3D6) + {x} * 3",
		"{x}@(2D6) * {x} + {y}@(2D6) * {y} + {x} * {y}"
	]
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = Evaluator::new(function);
		let (distribution, usage) =
			propagate_within(&evaluator, &[], Budget::UNLIMITED).unwrap();
		if let Err(e) =
			check_progress(&evaluator, &[], &distribution, usage, usize::MAX)
		{
			panic!("{source}: {e}");
		}
	}
}

/// Ensure that the pass reports its progress faithfully, and cancels cleanly,
/// on every case of the distribution corpus.
#[test]
fn test_progress_corpus()
{
	for (index, (source, args, externs, _)) in
		read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE)
			.iter()
			.enumerate()
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = evaluator(function, externs);
		let (distribution, usage) =
			propagate_within(&evaluator, args, Budget::UNLIMITED).unwrap();
		if let Err(e) = check_progress(
			&evaluator,
			args,
			&distribution,
			usage,
			CANCELLATIONS
		)
		{
			panic!("case {}: {}: {}", index + 1, source, e);
		}
	}
}

/// Ensure that a recipient may cancel the pass once it has taken more than
/// half of its estimated steps, as the example of
/// [`Progress`](crate::distribution::propagation::meter::Progress) does, and
/// that the pass then stops before it takes many more.
#[test]
fn test_cancel_halfway()
{
	let evaluator = Evaluator::new(compile("100D6").unwrap());
	let estimate = estimate(&evaluator, []).unwrap();
	let last = RefCell::new(Usage::default());
	let halfway = |consumed: Usage, estimate: &Cost| {
		*last.borrow_mut() = consumed;
		if consumed.steps > estimate.steps / 2
		{
			ControlFlow::Break(())
		}
		else
		{
			ControlFlow::Continue(())
		}
	};
	assert_eq!(
		propagate_worlds(
			&evaluator,
			[],
			Budget::UNLIMITED,
			&estimate,
			&halfway
		),
		Err(PropagationError::Cancelled)
	);
	let last = last.into_inner();
	assert!(estimate.steps / 2 < last.steps && last.steps < estimate.steps);
}

////////////////////////////////////////////////////////////////////////////////
//                                 Estimates.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Answer the estimate of the specified source, optimized, and the usage of
/// the pass within the unlimited budget, having checked that the estimate
/// bounds the usage.
///
/// # Parameters
/// - `source`: The source, which takes no arguments.
///
/// # Returns
/// The estimate and the usage.
///
/// # Panics
/// If the source does not compile, the pass refuses it, or the estimate does
/// not bound the usage.
fn estimated(source: &str) -> (Cost, Usage)
{
	let function = optimize(compile_valid(source), Passes::all());
	let evaluator = Evaluator::new(function);
	let (distribution, usage) =
		propagate_within(&evaluator, &[], Budget::UNLIMITED).unwrap();
	let cost = check_estimate(&evaluator, &[], &distribution, usage)
		.unwrap_or_else(|e| panic!("{source}: {e}"));
	(cost, usage)
}

/// Ensure that the estimate of a pool of dice with fixed operands, with or
/// without drops, or of arithmetic over such pools, is exact.
#[test]
fn test_exact_estimates()
{
	for source in [
		"100D6",
		"100D6 + 50D8",
		"15D6",
		"4D6 drop lowest 1",
		"10D10 drop highest 2",
		"30D6 drop lowest 5 drop highest 5",
		"10D[1, 5, 9] drop highest 2",
		"20D[-5, 0, 5]",
		"5D[1, 1000000]",
		"3D6 * 3D6",
		"[1:100] * [1:100]"
	]
	{
		let (cost, usage) = estimated(source);
		assert_eq!(
			(cost.steps, cost.cells, cost.worlds),
			(usage.steps, usage.cells, usage.worlds),
			"{source}"
		);
	}
}

/// Ensure that the estimate bounds a mixture of dice whose count takes fewer
/// values than its support bounds, leaving a gap, as `3D[1, 2, 4]` never
/// sums to `11`, so that the powers step up by more than one die at once.
#[test]
fn test_gapped_counts()
{
	for source in [
		"(3D[1, 2, 4])D[-1, 2]",
		"(3D[1, 2, 4])D[-2, 1]",
		"(4D[1, 2, 4])D[-1, 2]",
		"(4D[1, 2, 4])D[-2, 1]",
		"(3D[1, 2, 4])D[0, 1, 10]",
		"(2D[1, 2, 7])D[-1, 2]",
		"(1D2 * 9 + 1)D[0, 1, 20]",
		"(3D[1, 2, 4])D6"
	]
	{
		estimated(source);
	}
}

/// Ensure that the estimate stays tight where it is not exact: within the
/// specified ratio of the usage, in steps and in cells, on mixtures, splits,
/// and every case of the distribution corpus. The budget decides which
/// functions are reasonable, so a loose estimate would refuse reasonable
/// functions.
#[test]
fn test_tight_estimates()
{
	let within = |source: &str, cost: Cost, usage: Usage, ratio: f64| {
		for (estimate, used) in
			[(cost.steps, usage.steps), (cost.cells, usage.cells)]
		{
			assert!(
				estimate as f64 <= ratio * used.max(1) as f64,
				"{source}: {cost:?} is looser than {ratio} × {usage:?}"
			);
		}
	};
	for (source, ratio) in [
		("(20D6)D6", 1.1),
		("(2D6)D10", 1.1),
		("(1D20)D(1D20)", 1.1),
		("3D(1D100)", 1.1),
		("(3D6)D6 drop lowest 1", 1.1),
		("(1D6)D6 drop lowest (1D3)", 1.3),
		// Random drops are priced by the pairs of drops that keep a die, but
		// every setup may be a point mass, though drops beyond the dice merge.
		("3D6 drop lowest (1D2) drop highest (1D2)", 1.2),
		("2D4 drop lowest (1D4)", 1.3),
		("4D4 drop highest (1D3 - 2)", 1.2),
		("1D(1D100)", 1.2),
		("[(1D100):(1D100 + 100)]", 1.1),
		("{x}@(3D6) + {x} * 3", 1.1),
		("{a}@(1D100) * {a} + 2 * {a}", 1.1),
		("{x}@(2D6) * {x} + {y}@(2D6) * {y} + {x} * {y}", 1.1),
		// A split fixes the operands of later rolls, so each world is priced
		// by its own lane.
		("{x}@(10D6) + {x}D6", 1.1),
		("{x}@(1D20) + {x}D6", 1.1),
		("{x}@(1D20) + {x}D6 drop lowest 1", 1.1),
		("{x}@(3D6) + {x}D{x}", 1.1),
		("{x}@(1D10) + {x}D[1, 3, 7]", 1.1),
		("{x}@(1D10) + [{x}:{x} + 10]", 1.1),
		("{x}@(1D6) + {y}@(1D6) + ({x} + {y})D6", 1.1),
		// The survivor of the merge holds only the outcomes of its worlds.
		("({x}@(1D20) * 1000 + {x}D2) * 1D100", 1.1),
		// Nested or wide splits share the lanes, in buckets of several values.
		("{x}@(1D30) + {y}@(1D30) + [{x}:{x} + {y}]", 1.1),
		("{x}@(1D1000) + [1:{x}]", 1.1),
		// A bucket of counts is priced by the dearest of their powers.
		("{x}@(1D300) + {x}D2", 1.1),
		("{x}@(1D300) + {x}D4", 1.1),
		("{x}@(1D600) + {x}D2", 1.1),
		("{x}@(1D300) + {x}D[1, 3, 7]", 1.1)
	]
	{
		let (cost, usage) = estimated(source);
		within(source, cost, usage, ratio);
	}
	for (source, args, externs, _) in
		read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE)
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = evaluator(function, &externs);
		let (_, usage) =
			propagate_within(&evaluator, &args, Budget::UNLIMITED).unwrap();
		let cost = estimate(&evaluator, args.iter().copied()).unwrap();
		within(source, cost, usage, 2.0);
	}
}

/// Ensure that the estimate of an enormous function saturates, promptly,
/// rather than overflowing, and exceeds any small budget, however the
/// function grows: by many dice, many faces, wide arithmetic, many worlds,
/// or the multisets of a record read twice.
#[test]
fn test_saturated_estimates()
{
	for (source, args) in [
		("{n}: {n}D6", &[i32::MAX][..]),
		("{n}: {n}D6 drop lowest 3", &[i32::MAX]),
		("{n}: {n}D6 drop lowest {n}", &[i32::MAX]),
		("{n}: {n}D[-5, 0, 5] drop highest 2", &[i32::MAX]),
		("{n}: 3D{n}", &[i32::MAX]),
		("{n}: ({n}D{n})D({n}D{n})", &[i32::MAX]),
		("[1:2147483647] * [1:2147483647]", &[]),
		("[-2147483648:2147483647] * [-2147483648:2147483647]", &[]),
		(
			"{x}@([1:2147483647]) * {x} + {y}@([1:2147483647]) * {y} + {x} * {y}",
			&[]
		)
	]
	{
		let evaluator = Evaluator::new(compile(source).unwrap());
		let cost = estimate(&evaluator, args.iter().copied()).unwrap();
		assert!(
			cost.steps > SMALL.steps || cost.cells > SMALL.cells,
			"{source}: {cost:?}"
		);
	}
	// A record read twice splits on the multisets of its results.
	for (source, args) in
		[("{n}: {n}D6", &[i32::MAX][..]), ("{n}: 3D{n}", &[i32::MAX])]
	{
		let function = resum(compile(source).unwrap(), &[true]);
		let evaluator = Evaluator::new(function);
		let cost = estimate(&evaluator, args.iter().copied()).unwrap();
		assert_eq!(cost.worlds, u64::MAX, "{source}: {cost:?}");
	}
	// The saturated estimate of a record of all but one die dropped reports
	// the width of its total.
	let evaluator =
		Evaluator::new(compile("{n}: {n}D6 drop lowest 3").unwrap());
	let cost = estimate(&evaluator, [i32::MAX]).unwrap();
	assert!(
		cost.bits >= (i32::MAX as f64 * 6f64.log2()) as u64,
		"{cost:?}"
	);
}

/// Ensure that the estimate classifies the examples of the cost model by
/// their structural traits.
#[test]
fn test_classes()
{
	let class = |source: &str, args: &[i32]| {
		let evaluator = Evaluator::new(compile(source).unwrap());
		estimate(&evaluator, args.iter().copied()).unwrap().class
	};
	let none = Class::default();
	for (source, args, expected) in [
		("100D6 + 5", &[][..], none),
		("0", &[], none),
		(
			"3D6 * 3D6 + 2",
			&[],
			Class {
				pairwise: true,
				..none
			}
		),
		("3D6 * 2 + 3D6", &[], none),
		(
			"100D6 drop lowest 3",
			&[],
			Class {
				drops: true,
				..none
			}
		),
		("4D6 drop lowest 0", &[], none),
		(
			"{x}, {y}, {z}: ({x}D{y})D{z}",
			&[3, 6, 6],
			Class {
				dynamic: true,
				..none
			}
		),
		(
			"(1D6)D6 drop lowest (1D3)",
			&[],
			Class {
				dynamic: true,
				drops: true,
				..none
			}
		),
		(
			"{x}@(3D6) + {x} * 3",
			&[],
			Class {
				correlated: true,
				..none
			}
		),
		(
			"[1:1000000] * [1:1000000]",
			&[],
			Class {
				pairwise: true,
				..none
			}
		),
		(
			"{x}@(1D6) + {x}D6 drop highest 1",
			&[],
			Class {
				drops: true,
				dynamic: true,
				correlated: true,
				..none
			}
		)
	]
	{
		assert_eq!(class(source, args), expected, "{source}");
	}
}

/// Ensure that the estimate refuses arguments that disagree with the
/// parameters.
#[test]
fn test_estimate_refusals()
{
	assert_eq!(
		estimate(&Evaluator::new(compile("{a}: {a}D6").unwrap()), []),
		Err(PropagationError::BadArity {
			expected: 1,
			given: 0
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                   Plans.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a plan estimates as the cost model does, and builds, within a
/// budget that covers its estimate, what the pass answers.
#[test]
fn test_plan_distribution()
{
	for (source, args) in [
		("3D6", &[][..]),
		("{a}: {a}D6 drop lowest", &[4][..]),
		("{x}@(1D6) + {y}@(1D4) + {x} * {y} + {x} - {y}", &[][..]),
		("(1D3)D3", &[][..])
	]
	{
		let evaluator = Evaluator::new(compile(source).unwrap());
		let plan = evaluator.plan_distribution(args.iter().copied()).unwrap();
		let cost = estimate(&evaluator, args.iter().copied()).unwrap();
		assert_eq!(plan.estimate(), cost, "{source}");
		let budget = Budget {
			steps: cost.steps,
			cells: cost.cells
		};
		assert_eq!(
			plan.build(budget, &Unobserved),
			Ok(propagate(&evaluator, args.iter().copied()).unwrap()),
			"{source}"
		);
	}
}

/// Ensure that a plan refuses arguments that disagree with the parameters.
#[test]
fn test_plan_distribution_refusals()
{
	assert_eq!(
		Evaluator::new(compile("{a}: {a}D6").unwrap()).plan_distribution([]),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
}

/// Ensure that a build refuses exactly where the pass does, carrying the
/// estimate, and that its progress cancels it.
#[test]
fn test_build_refusals()
{
	let evaluator = Evaluator::new(compile("{a}: {a}D6").unwrap());
	for (args, budget) in [
		(
			[i32::MAX],
			Budget {
				steps: 1_000_000,
				cells: 1_000_000
			}
		),
		(
			[100],
			Budget {
				steps: u64::MAX,
				cells: 10
			}
		)
	]
	{
		let plan = evaluator.plan_distribution(args).unwrap();
		let Err(PropagationError::Exhausted(Exhausted {
			dimension,
			requested,
			remaining,
			consumed
		})) = propagate_within(&evaluator, &args, budget)
		else
		{
			panic!("{args:?} fits {budget:?}")
		};
		assert_eq!(
			plan.build(budget, &Unobserved),
			Err(BuildError::BudgetExhausted {
				estimate: plan.estimate(),
				dimension,
				requested,
				remaining,
				consumed
			})
		);
	}
	let plan = evaluator.plan_distribution([3]).unwrap();
	let cancel = |_: Usage, _: &Cost| ControlFlow::Break(());
	assert_eq!(
		plan.build(Budget::UNLIMITED, &cancel),
		Err(BuildError::Cancelled)
	);
}

/// Ensure that budgets, usages, and estimates serialize as their fields, and
/// round trip.
#[cfg(feature = "serde")]
#[test]
fn test_serde()
{
	let budget = Budget { steps: 1, cells: 2 };
	let json = serde_json::to_string(&budget).unwrap();
	assert_eq!(json, r#"{"steps":1,"cells":2}"#);
	assert_eq!(serde_json::from_str::<Budget>(&json).unwrap(), budget);
	let usage = Usage {
		steps: 1,
		cells: 2,
		worlds: 3
	};
	let json = serde_json::to_string(&usage).unwrap();
	assert_eq!(json, r#"{"steps":1,"cells":2,"worlds":3}"#);
	assert_eq!(serde_json::from_str::<Usage>(&json).unwrap(), usage);
	let evaluator =
		Evaluator::new(compile("{x}@(1D6 drop lowest) + {x}").unwrap());
	let cost = evaluator.plan_distribution([]).unwrap().estimate();
	let json = serde_json::to_string(&cost).unwrap();
	assert_eq!(serde_json::from_str::<Cost>(&json).unwrap(), cost);
	for dimension in [Dimension::Steps, Dimension::Cells]
	{
		let json = serde_json::to_string(&dimension).unwrap();
		assert_eq!(json, format!("\"{dimension:?}\""));
		assert_eq!(
			serde_json::from_str::<Dimension>(&json).unwrap(),
			dimension
		);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Properties.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The number of cases of the property.
const CASES: u32 = 1_000;

/// The budget of each pass of the property, beyond which it abstains.
const BUDGET: Budget = Budget {
	steps: 200_000,
	cells: 100_000
};

/// The seeds of the evaluations whose results each answer of the property
/// must hold.
const SEEDS: [u64; 4] = [0, 1, 2, 3];

/// The dice budget of each evaluation of the property, beyond which it
/// abstains.
const DICE: u64 = 10_000;

/// Ensure that the pass answers every random [static program](static_program)
/// that compiles, optimized and not, over random bindings, soundly and alike,
/// whenever it fits [`BUDGET`].
#[test]
fn test_propagated_programs()
{
	check_within(
		CASES,
		file!(),
		|| {
			(
				static_program(),
				[binding(), binding()],
				[binding(), binding(), binding(), binding(), binding()]
			)
		},
		|(source, args, externals)| {
			check_propagation(&source, &args, &externals, true, |f| f)
		}
	);
}

/// Ensure that the pass answers every random [dynamic
/// program](dynamic_program) that compiles, optimized and not, over random
/// bindings, soundly and alike, whenever it fits [`BUDGET`].
#[test]
fn test_propagated_dynamic_programs()
{
	check_within(
		CASES,
		file!(),
		|| {
			(
				dynamic_program(),
				[binding(), binding()],
				[binding(), binding(), binding(), binding(), binding()]
			)
		},
		|(source, args, externals)| {
			check_propagation(&source, &args, &externals, true, |f| f)
		}
	);
}

/// Ensure that the pass answers every random [static program](static_program)
/// that compiles, optimized and not, [resummed](resum), over random bindings,
/// soundly, whenever it fits [`BUDGET`]. Resumming the two functions alike
/// need not keep them equivalent, so their answers need not agree.
#[test]
fn test_propagated_resummed_programs()
{
	check_within(
		CASES,
		file!(),
		|| {
			(
				static_program(),
				[binding(), binding()],
				[binding(), binding(), binding(), binding(), binding()],
				prop::collection::vec(any::<bool>(), 1..8)
			)
		},
		|(source, args, externals, chosen)| {
			check_propagation(&source, &args, &externals, false, |f| {
				resum(f, &chosen)
			})
		}
	);
}

/// Ensure that the pass answers every random [dynamic
/// program](dynamic_program) that compiles, optimized and not,
/// [resummed](resum), over random bindings, soundly, whenever it fits
/// [`BUDGET`]. Resumming the two functions alike need not keep them
/// equivalent, so their answers need not agree.
#[test]
fn test_propagated_resummed_dynamic_programs()
{
	check_within(
		CASES,
		file!(),
		|| {
			(
				dynamic_program(),
				[binding(), binding()],
				[binding(), binding(), binding(), binding(), binding()],
				prop::collection::vec(any::<bool>(), 1..8)
			)
		},
		|(source, args, externals, chosen)| {
			check_propagation(&source, &args, &externals, false, |f| {
				resum(f, &chosen)
			})
		}
	);
}

/// Ensure that the pass answers every random [clause program](ClauseProgram),
/// a roll followed by drops and keeps in any order, unoptimized, optimized by
/// constant folding or strength reduction alone, and optimized by every pass,
/// exactly as [enumerating](ClauseProgram::enumerate) its outcomes through a
/// model that slices the sorted results, clause by clause, does; that it
/// answers them soundly, within its estimate; that the bounds are exact when
/// every operand is fixed; and that the evaluator answers each evaluation as
/// the model does on the dice that it records.
#[test]
fn test_propagated_clauses()
{
	check_within(CASES, file!(), clause_program, |program| {
		let function = Assembler::assemble(&program.assembly()).unwrap();
		let expected = program.enumerate();
		let mut evaluator = Evaluator::new(function.clone());
		for seed in SEEDS
		{
			let evaluation = evaluator
				.evaluate(Vec::<i32>::new(), &mut StdRng::seed_from_u64(seed))
				.unwrap();
			prop_assert_eq!(
				evaluation.result,
				program.model(&evaluation.records),
				"evaluated otherwise than the model\nfunction:\n{}\n\
					records: {:?}",
				function,
				evaluation.records
			);
		}
		for function in [
			function.clone(),
			optimize(function.clone(), Pass::ConstantFolding.into()),
			optimize(function.clone(), Pass::StrengthReduction.into()),
			optimize(function, Passes::all())
		]
		{
			let evaluator = Evaluator::new(function);
			let (propagated, usage) =
				match propagate_within(&evaluator, &[], BUDGET)
				{
					Ok(answer) => answer,
					Err(PropagationError::Exhausted(_)) => continue,
					Err(e) =>
					{
						return Err(TestCaseError::fail(format!(
							"refused: {e:?}\nfunction:\n{}",
							evaluator.function
						)))
					},
				};
			prop_assert!(
				agree(&propagated, &expected),
				"propagated otherwise than enumerated\nfunction:\n{}\n\
					propagated:\n{}\nexpected:\n{}",
				evaluator.function,
				propagated,
				expected
			);
			check_sound(&evaluator, &[], &[], &propagated)?;
			if program.fixed()
			{
				let bounds = evaluator
					.bounds_over([], std::iter::empty::<(&str, _)>())
					.unwrap();
				let outcomes = propagated.iter().map(|(outcome, _)| outcome);
				prop_assert_eq!(
					(bounds.value.min, bounds.value.max),
					(outcomes.clone().min().unwrap(), outcomes.max().unwrap()),
					"inexact bounds\nfunction:\n{}",
					evaluator.function
				);
			}
			check_exact_budget(&evaluator, &[], &propagated, usage)
				.map_err(TestCaseError::fail)?;
			check_estimate(&evaluator, &[], &propagated, usage)
				.map_err(TestCaseError::fail)?;
			check_progress(
				&evaluator,
				&[],
				&propagated,
				usage,
				PROPERTY_CANCELLATIONS
			)
			.map_err(TestCaseError::fail)?;
		}
		Ok(())
	});
}

/// A random program that rolls dice and applies drops and keeps to them, in
/// order, whose counts are fixed or rolled from ranges, for
/// [`test_propagated_clauses`].
#[derive(Debug, Clone)]
struct ClauseProgram
{
	/// The number of dice.
	count: ClauseCount,

	/// The die.
	die: ClauseDie,

	/// The clauses, in order.
	clauses: Vec<Clause>
}

/// The die of a [`ClauseProgram`].
#[derive(Debug, Clone)]
enum ClauseDie
{
	/// A standard die of the specified number of faces, which may be none.
	Standard(i32),

	/// A custom die of the specified faces.
	Custom(Vec<i32>)
}

/// A drop or keep of a [`ClauseProgram`].
#[derive(Debug, Clone, Copy)]
struct Clause
{
	/// Whether the clause keeps rather than drops.
	keep: bool,

	/// Whether the clause drops or keeps the lowest dice rather than the
	/// highest.
	lowest: bool,

	/// The count of the clause.
	count: ClauseCount
}

/// A count of a [`ClauseProgram`]: fixed, or rolled from an inclusive range,
/// which may be empty, and then answers zero.
#[derive(Debug, Clone, Copy)]
enum ClauseCount
{
	/// A fixed count.
	Fixed(i32),

	/// A count rolled from the range of the specified endpoints.
	Rolled(i32, i32)
}

impl ClauseCount
{
	/// Answer the values of the count, one for each branch of its roll.
	///
	/// # Returns
	/// The values.
	fn values(self) -> Vec<i32>
	{
		match self
		{
			ClauseCount::Fixed(count) => vec![count],
			ClauseCount::Rolled(start, end) if end < start => vec![0],
			ClauseCount::Rolled(start, end) => (start..=end).collect()
		}
	}
}

impl ClauseProgram
{
	/// Answer the counts of the program that it rolls, in the order that it
	/// rolls them: the number of dice first, then the counts of the clauses.
	///
	/// # Returns
	/// The rolled counts.
	fn counts(&self) -> Vec<ClauseCount>
	{
		std::iter::once(self.count)
			.chain(self.clauses.iter().map(|clause| clause.count))
			.collect()
	}

	/// Answer whether every count of the program is fixed.
	///
	/// # Returns
	/// `true` if every count is fixed.
	fn fixed(&self) -> bool
	{
		self.counts()
			.iter()
			.all(|count| matches!(count, ClauseCount::Fixed(_)))
	}

	/// Answer the text of the program: each rolled count rolls its range into
	/// a record of its own, which it sums into a register of its own, in
	/// order, and then the dice roll into the last record, and the clauses
	/// apply to it, in order.
	///
	/// # Returns
	/// The text, which assembles.
	fn assembly(&self) -> String
	{
		let mut body = String::new();
		let mut next = 0;
		let mut operand = |count: ClauseCount, body: &mut String| match count
		{
			ClauseCount::Fixed(count) => count.to_string(),
			ClauseCount::Rolled(start, end) =>
			{
				body.push_str(&format!(
					"\t\t⚅{next} <- roll range {start}:{end}\n\t\t@{next} <- \
					 sum rolling record ⚅{next}\n"
				));
				next += 1;
				format!("@{}", next - 1)
			}
		};
		let count = operand(self.count, &mut body);
		let operands = self
			.clauses
			.iter()
			.map(|clause| operand(clause.count, &mut body))
			.collect::<Vec<_>>();
		let dice = next;
		match &self.die
		{
			ClauseDie::Standard(faces) => body.push_str(&format!(
				"\t\t⚅{dice} <- roll standard dice {count}D{faces}\n"
			)),
			ClauseDie::Custom(faces) => body.push_str(&format!(
				"\t\t⚅{dice} <- roll custom dice {count}D[{}]\n",
				faces
					.iter()
					.map(i32::to_string)
					.collect::<Vec<_>>()
					.join(", ")
			))
		}
		for (clause, operand) in self.clauses.iter().zip(operands)
		{
			body.push_str(&format!(
				"\t\t⚅{dice} <- {} {} {operand} from ⚅{dice}\n",
				if clause.keep { "keep" } else { "drop" },
				if clause.lowest { "lowest" } else { "highest" }
			));
		}
		body.push_str(&format!(
			"\t\t@{dice} <- sum rolling record ⚅{dice}\n\t\treturn @{dice}"
		));
		format!(
			"Function() r#{} ⚅#{}\n\textern[]\n\tbody:\n{body}",
			dice + 1,
			dice + 1
		)
	}

	/// Answer the result of the program, as a model that slices the sorted
	/// results, clause by clause, computes it: a drop removes dice from one
	/// end of those still kept, and a keep removes all but some of them.
	///
	/// # Parameters
	/// - `counts`: The values of the counts, in the order of
	///   [`counts`](Self::counts).
	/// - `results`: The results of the dice.
	///
	/// # Returns
	/// The result.
	fn slice(&self, counts: &[i32], results: &[i32]) -> i32
	{
		let mut kept = results.to_vec();
		kept.sort();
		let mut kept = &kept[..];
		for (clause, &count) in self.clauses.iter().zip(&counts[1..])
		{
			let count = (count.max(0) as usize).min(kept.len());
			let remove = match clause.keep
			{
				true => kept.len() - count,
				false => count
			};
			kept = match clause.lowest != clause.keep
			{
				true => &kept[remove..],
				false => &kept[..kept.len() - remove]
			};
		}
		kept.iter()
			.fold(0, |sum, result| sum.saturating_add(*result))
	}

	/// Answer the result of an evaluation of the program, as the
	/// [model](Self::slice) computes it from the records of the evaluation.
	///
	/// # Parameters
	/// - `records`: The rolling records of the evaluation.
	///
	/// # Returns
	/// The result.
	fn model(&self, records: &[crate::RollingRecord]) -> i32
	{
		let (dice, ranges) = records.split_last().unwrap();
		let mut ranges = ranges.iter().map(|record| record.dice[0].value);
		let counts = self
			.counts()
			.into_iter()
			.map(|count| match count
			{
				ClauseCount::Fixed(count) => count,
				ClauseCount::Rolled(..) => ranges.next().unwrap()
			})
			.collect::<Vec<_>>();
		let results = dice.dice.iter().map(|die| die.value).collect::<Vec<_>>();
		self.slice(&counts, &results)
	}

	/// Answer the distribution of the program, by enumerating every branch of
	/// every count and every die, and computing each result with the
	/// [model](Self::slice). Every branch of the counts is equally likely, so
	/// each result weighs the branches of the dice that the most dice have
	/// for each branch of the dice that its own have.
	///
	/// # Returns
	/// The distribution.
	fn enumerate(&self) -> Distribution
	{
		let mut weights = std::collections::BTreeMap::<i32, u64>::new();
		let mut counts = vec![Vec::new()];
		for count in self.counts()
		{
			counts = counts
				.into_iter()
				.flat_map(|prefix: Vec<i32>| {
					count.values().into_iter().map(move |value| {
						let mut counts = prefix.clone();
						counts.push(value);
						counts
					})
				})
				.collect();
		}
		let faces = match &self.die
		{
			ClauseDie::Standard(faces) if *faces <= 0 => vec![0],
			ClauseDie::Standard(faces) => (1..=*faces).collect(),
			ClauseDie::Custom(faces) => faces.clone()
		};
		let rolls = counts
			.into_iter()
			.map(|counts| {
				let mut results = vec![Vec::new()];
				for _ in 0..counts[0].max(0)
				{
					results = results
						.into_iter()
						.flat_map(|prefix: Vec<i32>| {
							faces.iter().map(move |face| {
								let mut results = prefix.clone();
								results.push(*face);
								results
							})
						})
						.collect();
				}
				(counts, results)
			})
			.collect::<Vec<_>>();
		let most = rolls.iter().map(|(_, results)| results.len()).max();
		for (counts, results) in &rolls
		{
			let weight = (most.unwrap() / results.len()) as u64;
			for results in results
			{
				*weights.entry(self.slice(counts, results)).or_default() +=
					weight;
			}
		}
		Distribution::from_weights(
			weights
				.into_iter()
				.map(|(outcome, weight)| (outcome, Weight::from(weight)))
		)
		.unwrap()
	}
}

/// Answer a strategy that generates [clause programs](ClauseProgram), small
/// enough to [enumerate](ClauseProgram::enumerate).
///
/// # Returns
/// The strategy.
fn clause_program() -> impl Strategy<Value = ClauseProgram>
{
	let count = |fixed: std::ops::RangeInclusive<i32>| {
		prop_oneof![
			3 => fixed.prop_map(ClauseCount::Fixed),
			1 => (-1..=2, 0..=4)
				.prop_map(|(start, end)| ClauseCount::Rolled(start, end))
		]
	};
	let die = prop_oneof![
		(-1..=4).prop_map(ClauseDie::Standard),
		prop::collection::vec(-3..=3, 1..=3).prop_map(ClauseDie::Custom)
	];
	let clause = (any::<bool>(), any::<bool>(), count(-1..=5)).prop_map(
		|(keep, lowest, count)| Clause {
			keep,
			lowest,
			count
		}
	);
	(count(-1..=4), die, prop::collection::vec(clause, 1..=3)).prop_map(
		|(count, die, clauses)| ClauseProgram {
			count,
			die,
			clauses
		}
	)
}

/// Rewrite a function so that it reads its rolling records more than once, as
/// the compiler never does: right after each roll or drop that `chosen`
/// selects, sum the record into a fresh register, and add every such register
/// to the answer.
///
/// # Parameters
/// - `function`: The function.
/// - `chosen`: Whether to sum after each roll or drop, in order, repeated as
///   needed, which is not empty.
///
/// # Returns
/// The rewritten function, which is valid if the function is.
pub(super) fn resum(function: Function, chosen: &[bool]) -> Function
{
	let mut next = function.register_count;
	let mut fresh = Vec::new();
	let mut choices = chosen.iter().cycle();
	let mut instructions = Vec::with_capacity(function.instructions.len());
	for inst in function.instructions
	{
		let record = match &inst
		{
			Instruction::RollRange(inst) => Some(inst.dest),
			Instruction::RollStandardDice(inst) => Some(inst.dest),
			Instruction::RollCustomDice(inst) => Some(inst.dest),
			Instruction::DropLowest(inst) => Some(inst.dest),
			Instruction::DropHighest(inst) => Some(inst.dest),
			Instruction::Return(ret) =>
			{
				let mut answer = ret.src;
				for reg in mem::take(&mut fresh)
				{
					let dest = RegisterIndex(next);
					next += 1;
					instructions.push(
						Add {
							dest,
							op1: answer,
							op2: AddressingMode::Register(reg)
						}
						.into()
					);
					answer = AddressingMode::Register(dest);
				}
				instructions.push(Return { src: answer }.into());
				continue
			},
			_ => None
		};
		instructions.push(inst);
		if let Some(src) = record
			&& *choices.next().expect("choices are not empty")
		{
			let dest = RegisterIndex(next);
			next += 1;
			instructions.push(SumRollingRecord { dest, src }.into());
			fresh.push(dest);
		}
	}
	let function = Function {
		register_count: next,
		instructions,
		..function
	};
	debug_assert_eq!(function.validate(), Ok(()), "{function}");
	function
}

/// Check that the pass answers a source, if it compiles, optimized and not,
/// within [`BUDGET`], soundly and faithfully: its answer lies within the
/// [bounds](Evaluator::bounds_over) of the function, totals the branches that
/// the bounds count, if they count them, and holds the result of every
/// [evaluation](Evaluator::evaluate_metered) of [`SEEDS`] within [`DICE`]; it
/// succeeds within a budget
/// exactly when the budget covers its usage, and passes [`check_estimate`] and
/// [`check_progress`]; and, if the rewrite preserves the equivalence of the
/// functions, its answers to both agree in probability. The distribution corpus
/// and the Lean oracle check the answers themselves.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments, of which the function takes as many as its arity.
/// - `externals`: The values of the externals, by index into [`NAMES`].
/// - `equivalent`: Whether the rewritten functions, optimized and not, are
///   equivalent, so that the answers to both must agree in probability.
/// - `rewrite`: The rewrite of each function, optimized and not, before the
///   pass sees it.
///
/// # Errors
/// [`TestCaseError`] if the pass refuses the source other than for
/// [`BUDGET`], or answers unsoundly, or its answers disagree, or it fails
/// [`check_exact_budget`], [`check_estimate`], or [`check_progress`].
fn check_propagation(
	source: &str,
	args: &[i32],
	externals: &[i32],
	equivalent: bool,
	rewrite: impl Fn(Function) -> Function
) -> Result<(), TestCaseError>
{
	let Ok(unoptimized) = compile_unoptimized(source)
	else
	{
		return Ok(())
	};
	let optimized = optimize(unoptimized.clone(), Passes::all());
	let args = &args[..unoptimized.arity()];
	let mut answers = Vec::new();
	for function in [unoptimized, optimized]
	{
		let function = rewrite(function);
		// Every plan merges every split by the last instruction.
		Plan::new(&function.instructions);
		let externals = NAMES
			.iter()
			.zip(externals)
			.filter(|(name, _)| function.externals.contains(&name.to_string()))
			.map(|(name, value)| (*name, *value))
			.collect::<Vec<_>>();
		let evaluator = evaluator(function, &externals);
		let (propagated, usage) =
			match propagate_within(&evaluator, args, BUDGET)
			{
				Ok(answer) => answer,
				Err(PropagationError::Exhausted(_)) => continue,
				Err(e) =>
				{
					return Err(TestCaseError::fail(format!(
						"refused: {e:?}\nfunction:\n{}",
						evaluator.function
					)))
				},
			};
		check_sound(&evaluator, args, &externals, &propagated)?;
		check_exact_budget(&evaluator, args, &propagated, usage)
			.map_err(TestCaseError::fail)?;
		check_estimate(&evaluator, args, &propagated, usage)
			.map_err(TestCaseError::fail)?;
		check_progress(
			&evaluator,
			args,
			&propagated,
			usage,
			PROPERTY_CANCELLATIONS
		)
		.map_err(TestCaseError::fail)?;
		answers.push((evaluator.function, propagated));
	}
	if let [(unoptimized, expected), (optimized, propagated)] = &answers[..]
		&& equivalent
	{
		prop_assert!(
			agree(propagated, expected),
			"optimization changed the answer\nunoptimized:\n{}\n\
				optimized:\n{}\npropagated:\n{}\nexpected:\n{}",
			unoptimized,
			optimized,
			propagated,
			expected
		);
	}
	Ok(())
}

/// Check that an answer of the pass is sound: that it lies within the
/// [bounds](Evaluator::bounds_over) of the function, totals the branches that
/// the bounds count, if they count them, and holds the result of every
/// [evaluation](Evaluator::evaluate_metered) of [`SEEDS`] within [`DICE`].
///
/// # Parameters
/// - `evaluator`: The evaluator.
/// - `args`: The arguments.
/// - `externals`: The names and values of the externals.
/// - `propagated`: The answer of the pass.
///
/// # Errors
/// [`TestCaseError`] if the answer is unsound.
fn check_sound(
	evaluator: &Evaluator,
	args: &[i32],
	externals: &[(&str, i32)],
	propagated: &Distribution
) -> Result<(), TestCaseError>
{
	let bounds = evaluator
		.bounds_over(
			args.iter().map(|arg| Some((*arg).into())),
			externals
				.iter()
				.map(|(name, value)| (*name, (*value).into()))
		)
		.unwrap();
	for (outcome, _) in propagated
	{
		prop_assert!(
			bounds.value.contains(outcome),
			"outcome {} out of bounds {}\nfunction:\n{}",
			outcome,
			bounds.value,
			evaluator.function
		);
	}
	// The count saturates, and so counts nothing once it reaches its limit.
	if let Some(count) = bounds.count
		&& count < u128::MAX
	{
		prop_assert_eq!(
			propagated.total(),
			&Weight::from(count),
			"total is not the count\nfunction:\n{}",
			evaluator.function
		);
	}
	let mut evaluator = evaluator.clone();
	for seed in SEEDS
	{
		// The pass may answer many dice of one face far faster than the
		// evaluator can roll them.
		let result = match evaluator.evaluate_metered(
			args.iter().copied(),
			&mut StdRng::seed_from_u64(seed),
			DICE
		)
		{
			Ok(evaluation) => evaluation.result,
			Err(EvaluationError::DiceBudgetExhausted { .. }) => break,
			Err(e) => return Err(TestCaseError::fail(format!("{e}")))
		};
		prop_assert!(
			!propagated.get(result).is_zero(),
			"lacks the evaluated result {}\nfunction:\n{}",
			result,
			evaluator.function
		);
	}
	Ok(())
}

/// Answer a strategy that generates static programs: functions, with or
/// without [parameters](with_parameters), whose bodies are [static
/// expressions](static_expression).
///
/// # Returns
/// The strategy.
fn static_program() -> impl Strategy<Value = String>
{
	with_parameters(static_expression())
}

/// Answer a strategy that generates static expressions: arithmetic over
/// constants, variables, and rolls whose operands are fixed.
///
/// # Returns
/// The strategy.
fn static_expression() -> impl Strategy<Value = String>
{
	let leaf = prop_oneof![
		2 => static_operand(),
		3 => static_dice(),
		1 => (static_operand(), static_operand())
			.prop_map(|(start, end)| format!("[{}:{}]", start, end))
	];
	leaf.prop_recursive(3, 12, 2, |inner| {
		prop_oneof![
			3 => (inner.clone(), select(OPERATORS), inner.clone()).prop_map(
				|(left, op, right)| format!("{} {} {}", left, op, right)
			),
			1 => inner.clone().prop_map(|operand| format!("-({})", operand)),
			1 => (variable(), inner)
				.prop_map(|(name, e)| format!("{}@({})", name, e))
		]
	})
}

/// Answer a strategy that generates dice whose counts, faces, drops, and keeps
/// are fixed, with custom faces that often reach the extremes of [`i32`], so
/// that their sums saturate.
///
/// # Returns
/// The strategy.
fn static_dice() -> impl Strategy<Value = String>
{
	let face = prop_oneof![
		3 => -3i32..=6,
		1 => select(
			&[
				i32::MIN,
				i32::MIN + 1,
				-1_073_741_825,
				1_073_741_824,
				i32::MAX - 1,
				i32::MAX
			][..]
		)
	];
	let faces = prop_oneof![
		2 => static_operand(),
		1 => prop::collection::vec(face, 1 .. 5).prop_map(|faces| {
			let faces = faces.iter().map(i32::to_string).collect::<Vec<_>>();
			format!("[{}]", faces.join(", "))
		})
	];
	// A drop expression never begins with `-`, so a negative constant must be
	// grouped.
	let drop = static_operand().prop_map(|drop| {
		if drop.starts_with('-')
		{
			format!("({})", drop)
		}
		else
		{
			drop
		}
	});
	let clause = (
		select(&["drop", "keep"][..]),
		select(&["lowest", "highest"][..]),
		option::of(drop)
	)
		.prop_map(|(verb, direction, drop)| match drop
		{
			Some(drop) => format!(" {} {} {}", verb, direction, drop),
			None => format!(" {} {}", verb, direction)
		});
	(static_operand(), faces, prop::collection::vec(clause, 0..3)).prop_map(
		|(count, faces, clauses)| {
			format!("{}D{}{}", count, faces, clauses.concat())
		}
	)
}

/// Answer a strategy that generates fixed operands: small constants and
/// variables.
///
/// # Returns
/// The strategy.
fn static_operand() -> impl Strategy<Value = String>
{
	prop_oneof![
		3 => (-2i32 ..= 5).prop_map(|n| n.to_string()),
		1 => variable()
	]
}

/// Answer a strategy that generates dynamic programs: functions, with or
/// without [parameters](with_parameters), whose bodies are [dynamic
/// expressions](dynamic_expression).
///
/// # Returns
/// The strategy.
fn dynamic_program() -> impl Strategy<Value = String>
{
	with_parameters(dynamic_expression())
}

/// Answer a strategy that generates dynamic expressions: [static
/// expressions](static_expression), within dice whose counts, faces, and drop
/// and keep counts, and ranges whose endpoints, may be random.
///
/// # Returns
/// The strategy.
fn dynamic_expression() -> impl Strategy<Value = String>
{
	static_expression().prop_recursive(2, 8, 2, |inner| {
		let operand = prop_oneof![
			1 => static_operand(),
			2 => inner.prop_map(|operand| format!("({})", operand))
		];
		let faces = prop_oneof![
			3 => operand.clone(),
			1 => prop::collection::vec(-2i32..=4, 1..4).prop_map(|faces| {
				let faces =
					faces.iter().map(i32::to_string).collect::<Vec<_>>();
				format!("[{}]", faces.join(", "))
			})
		];
		// A drop expression never begins with `-`, so every count of a drop
		// or keep is grouped.
		let clause = (
			select(&["drop", "keep"][..]),
			select(&["lowest", "highest"][..]),
			operand.clone()
		)
			.prop_map(|(verb, direction, drop)| {
				format!(" {} {} ({})", verb, direction, drop)
			});
		prop_oneof![
			3 => (operand.clone(), faces, prop::collection::vec(clause, 0..3))
				.prop_map(|(count, faces, clauses)| {
					format!("{}D{}{}", count, faces, clauses.concat())
				}),
			1 => (operand.clone(), operand)
				.prop_map(|(start, end)| format!("[{}:{}]", start, end))
		]
	})
}

////////////////////////////////////////////////////////////////////////////////
//                                  Rerolls.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that, until the cost model prices rerolls, the estimate prices a
/// function that rerolls beyond every budget but the unlimited one, though the
/// pass, which charges its work as it goes, answers within the unlimited one.
#[test]
fn test_reroll_estimates_exceed_every_budget()
{
	let function = Assembler::assemble(
		"\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll standard dice 1D6
\t\t⚅0 <- reroll = 1 up to 1 times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
	)
	.unwrap();
	let evaluator = Evaluator::new(function);
	let plan = evaluator.plan_distribution([]).unwrap();
	assert_eq!(plan.estimate().steps, u64::MAX);
	assert_eq!(
		plan.build(Budget::UNLIMITED, &Unobserved),
		Ok(weights(&[(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]))
	);
}

/// Answer the assembly of a function that rolls `roll` into `⚅0`, applies the
/// specified clauses to it, sums it, and returns the sum.
///
/// # Parameters
/// - `roll`: The roll, as in `roll standard dice 3D6`.
/// - `clauses`: The clauses, each as in `reroll = 1 up to 1 times`.
///
/// # Returns
/// The assembly.
fn rerolled(roll: &str, clauses: &[&str]) -> String
{
	let mut assembly =
		format!("Function() r#1 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- {roll}\n");
	for clause in clauses
	{
		assembly.push_str(&format!("\t\t⚅0 <- {clause} from ⚅0\n"));
	}
	assembly.push_str("\t\t@0 <- sum rolling record ⚅0\n\t\treturn @0\n");
	assembly
}

/// Test that rerolls answer as expected, weight for weight, counting their
/// branches as [`Bounds::count`](crate::Bounds::count) does: each die still
/// kept grows by the branches of its die raised to the limit, or, without
/// limit, by the branches that the predicate does not match.
#[test]
fn test_rerolls()
{
	for (roll, clauses, expected) in [
		// A `1` rerolled once shows `1` again along one path of 36, and any
		// other face along 6 + 1.
		(
			"roll standard dice 1D6",
			&["reroll = 1 up to 1 times"][..],
			&[(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)][..]
		),
		// Without limit, each face but `1` is equally likely, over 6 · 5.
		(
			"roll standard dice 1D6",
			&["reroll = 1 unlimited times"],
			&[(2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
		),
		// Twice: `1` along 1, any other face along 36 + 6 + 1, over 6³.
		(
			"roll standard dice 1D6",
			&["reroll = 1 up to 2 times"],
			&[(1, 1), (2, 43), (3, 43), (4, 43), (5, 43), (6, 43)]
		),
		// A reroll that cannot happen changes nothing.
		(
			"roll standard dice 1D4",
			&["reroll = 5 up to 3 times"],
			&[(1, 1), (2, 1), (3, 1), (4, 1)]
		),
		(
			"roll standard dice 1D4",
			&["reroll = 1 up to 0 times"],
			&[(1, 1), (2, 1), (3, 1), (4, 1)]
		),
		(
			"roll standard dice 1D4",
			&["reroll in 3..3 unlimited times"],
			&[(1, 1), (2, 1), (3, 1), (4, 1)]
		),
		(
			"roll standard dice 1D4",
			&["reroll in 3..=2 up to 1 times"],
			&[(1, 1), (2, 1), (3, 1), (4, 1)]
		),
		// An unlimited reroll that matches every face crosses out every die.
		(
			"roll standard dice 3D1",
			&["reroll = 1 unlimited times"],
			&[(0, 1)]
		),
		(
			"roll standard dice 2D3",
			&["reroll < 4 unlimited times"],
			&[(0, 9)]
		),
		(
			"roll standard dice 2D0",
			&["reroll = 0 unlimited times"],
			&[(0, 1)]
		),
		("roll range 3:2", &["reroll = 0 unlimited times"], &[(0, 1)]),
		// A die without faces rerolls to the face `0` again.
		(
			"roll standard dice 2D0",
			&["reroll = 0 up to 2 times"],
			&[(0, 1)]
		),
		// A reroll after a drop conditions on the dice still kept: of 2D2,
		// the highest is `1` only for `[1, 1]`, and then rerolled.
		(
			"roll standard dice 2D2",
			&["drop lowest 1", "reroll = 1 up to 1 times"],
			&[(1, 1), (2, 7)]
		),
		// A range is one die.
		(
			"roll range 1:3",
			&["reroll >= 2 unlimited times"],
			&[(1, 3)]
		),
		(
			"roll range -1:1",
			&["reroll != 0 up to 1 times"],
			&[(-1, 2), (0, 5), (1, 2)]
		),
		// A custom die rerolls from its faces, each as often as it is listed.
		(
			"roll custom dice 1D[1, 1, 2]",
			&["reroll = 1 unlimited times"],
			&[(2, 3)]
		),
		(
			"roll custom dice 1D[1, 1, 2]",
			&["reroll in [2] up to 1 times"],
			&[(1, 8), (2, 1)]
		)
	]
	{
		let assembly = rerolled(roll, clauses);
		assert_eq!(
			propagate_assembly(&assembly),
			weights(expected),
			"{assembly}"
		);
	}
}

/// Test that a reroll that follows a split on its record, which fixes the
/// record's results, draws from the die of the roll that the split fixed: a
/// record summed both before and after a reroll, with and without a drop
/// between, and a record whose sum before the reroll is never read.
#[test]
fn test_rerolls_after_splits()
{
	// `1D4` shows `x`; `x ≤ 2` is rerolled to `y`, else `y = x`; the answer is
	// `x + y`, over 4 · 4.
	let function = Assembler::assemble(
		"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 1D4
		@0 <- sum rolling record ⚅0
		⚅0 <- reroll <= 2 up to 1 times from ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 + @1
		return @2"
	)
	.unwrap();
	assert_agrees(
		function,
		&weights(&[(2, 1), (3, 2), (4, 2), (5, 2), (6, 5), (8, 4)])
	);
	// `2D3` shows `[a, b]`; the higher is rerolled once if it is `1`, which
	// it is only for `[1, 1]`, and the answer is the sum of both rolls.
	let function = Assembler::assemble(
		"Function() r#3 ⚅#1
	extern[]
	body:
		⚅0 <- roll standard dice 2D3
		@0 <- sum rolling record ⚅0
		⚅0 <- drop lowest 1 from ⚅0
		⚅0 <- reroll = 1 up to 1 times from ⚅0
		@1 <- sum rolling record ⚅0
		@2 <- @0 + @1
		return @2"
	)
	.unwrap();
	let mut expected = BTreeMap::<i32, u64>::new();
	for a in 1..=3
	{
		for b in 1..=3
		{
			let high = a.max(b);
			for c in 1..=3
			{
				let kept = match high
				{
					1 => c,
					_ => high
				};
				*expected.entry(a + b + kept).or_default() += 1;
			}
		}
	}
	assert_agrees(
		function,
		&weights(&expected.into_iter().collect::<Vec<_>>())
	);
}

/// A chance, as a fraction that the reference reduces as it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Chance
{
	/// The numerator.
	numerator: Weight,

	/// The denominator, which is positive.
	denominator: Weight
}

impl Chance
{
	/// Answer the chance `numerator / denominator`, reduced.
	///
	/// # Parameters
	/// - `numerator`: The numerator.
	/// - `denominator`: The denominator, which is positive.
	///
	/// # Returns
	/// The chance.
	fn new(numerator: Weight, denominator: Weight) -> Self
	{
		let gcd = numerator.gcd(&denominator);
		Self {
			numerator: numerator.exact_div(&gcd),
			denominator: denominator.exact_div(&gcd)
		}
	}

	/// Answer the chance one in `n`.
	///
	/// # Parameters
	/// - `n`: The number of equally likely possibilities, which is positive.
	///
	/// # Returns
	/// The chance.
	fn one_in(n: usize) -> Self { Self::new(Weight::ONE, Weight::from(n)) }

	/// Answer the product of two chances.
	///
	/// # Parameters
	/// - `other`: The other chance.
	///
	/// # Returns
	/// The product.
	fn times(&self, other: &Self) -> Self
	{
		Self::new(
			&self.numerator * &other.numerator,
			&self.denominator * &other.denominator
		)
	}

	/// Add another chance to this one.
	///
	/// # Parameters
	/// - `other`: The other chance.
	fn add(&mut self, other: &Self)
	{
		*self = Self::new(
			&self.numerator * &other.denominator
				+ &other.numerator * &self.denominator,
			&self.denominator * &other.denominator
		);
	}
}

/// A law over values, as the reference computes it.
type Law<T> = BTreeMap<T, Chance>;

/// Add a chance of the specified value to a law.
///
/// # Parameters
/// - `law`: The law.
/// - `value`: The value.
/// - `chance`: The chance.
fn weigh<T: Ord>(law: &mut Law<T>, value: T, chance: Chance)
{
	match law.entry(value)
	{
		Entry::Vacant(entry) =>
		{
			entry.insert(chance);
		},
		Entry::Occupied(mut entry) => entry.get_mut().add(&chance)
	}
}

/// The roll of a [reroll program](RerollProgram).
#[derive(Debug, Clone)]
enum RerollRoll
{
	/// Standard dice, of a count and faces.
	Standard(AddressingMode, AddressingMode),

	/// Custom dice, of a count and the faces listed.
	Custom(AddressingMode, Vec<i32>),

	/// A range, of a start and end.
	Range(AddressingMode, AddressingMode)
}

/// A clause of a [reroll program](RerollProgram).
#[derive(Debug, Clone)]
enum RerollClause
{
	/// Drop the lowest results, by the count.
	DropLowest(AddressingMode),

	/// Drop the highest results, by the count.
	DropHighest(AddressingMode),

	/// Keep the lowest results, by the count.
	KeepLowest(AddressingMode),

	/// Keep the highest results, by the count.
	KeepHighest(AddressingMode),

	/// Reroll the results that satisfy the predicate, up to the limit.
	Reroll(Predicate, Limit)
}

/// A program that rolls a rolling record, applies clauses to it, and sums it,
/// whose operands may be immediates or registers, each register the sum of
/// one roll and an offset. The [reference](RerollProgram::reference) answers
/// its law by enumerating every roll and reroll that it might make.
#[derive(Debug, Clone)]
struct RerollProgram
{
	/// The registers, each the sum of `count` dice of `faces` faces, plus an
	/// offset.
	registers: Vec<(i32, i32, i32)>,

	/// The roll.
	roll: RerollRoll,

	/// The clauses, in order.
	clauses: Vec<RerollClause>
}

impl RerollProgram
{
	/// Answer the assembly of the program, in static single assignment form,
	/// as the optimizer expects: the sum of the roll of register `i` is
	/// `@2i`, and the register itself, the sum and its offset, is `@2i+1`.
	///
	/// # Returns
	/// The assembly.
	fn assembly(&self) -> String
	{
		let n = self.registers.len();
		let mut assembly = format!(
			"Function() r#{} ⚅#{}\n\textern[]\n\tbody:\n",
			2 * n + 1,
			n + 1
		);
		for (i, (count, faces, offset)) in self.registers.iter().enumerate()
		{
			assembly.push_str(&format!(
				"\t\t⚅{} <- roll standard dice {count}D{faces}\n\
				 \t\t@{} <- sum rolling record ⚅{}\n\
				 \t\t@{} <- @{} + {offset}\n",
				i + 1,
				2 * i,
				i + 1,
				2 * i + 1,
				2 * i
			));
		}
		let op = |op: AddressingMode| match op
		{
			AddressingMode::Register(reg) =>
			{
				RegisterIndex(2 * reg.0 + 1).into()
			},
			op => op
		};
		let roll = match &self.roll
		{
			RerollRoll::Standard(count, faces) =>
			{
				format!("roll standard dice {}D{}", op(*count), op(*faces))
			},
			RerollRoll::Custom(count, faces) =>
			{
				let faces = faces
					.iter()
					.map(ToString::to_string)
					.collect::<Vec<_>>()
					.join(", ");
				format!("roll custom dice {}D[{faces}]", op(*count))
			},
			RerollRoll::Range(start, end) =>
			{
				format!("roll range {}:{}", op(*start), op(*end))
			}
		};
		assembly.push_str(&format!("\t\t⚅0 <- {roll}\n"));
		for clause in &self.clauses
		{
			let clause = match clause
			{
				RerollClause::DropLowest(count) =>
				{
					format!("drop lowest {}", op(*count))
				},
				RerollClause::DropHighest(count) =>
				{
					format!("drop highest {}", op(*count))
				},
				RerollClause::KeepLowest(count) =>
				{
					format!("keep lowest {}", op(*count))
				},
				RerollClause::KeepHighest(count) =>
				{
					format!("keep highest {}", op(*count))
				},
				RerollClause::Reroll(predicate, limit) =>
				{
					format!("reroll {} {}", predicate.map(op), limit.map(op))
				}
			};
			assembly.push_str(&format!("\t\t⚅0 <- {clause} from ⚅0\n"));
		}
		assembly.push_str(&format!(
			"\t\t@{} <- sum rolling record ⚅0\n\t\treturn @{}\n",
			2 * n,
			2 * n
		));
		assembly
	}

	/// Answer the law of the program's answer, by enumerating every value of
	/// its registers, every roll of its dice, and every reroll that each die
	/// still kept may make, independently of the forward pass.
	///
	/// # Returns
	/// The law, as a distribution, which agrees with the program's in
	/// probability.
	fn reference(&self) -> Distribution
	{
		let mut assignments = vec![(Vec::<i32>::new(), Chance::one_in(1))];
		for &(count, faces, offset) in &self.registers
		{
			let mut sums = Law::<i32>::from([(offset, Chance::one_in(1))]);
			for _ in 0..count
			{
				let mut next = Law::new();
				for (sum, chance) in &sums
				{
					for face in 1..=faces
					{
						weigh(
							&mut next,
							sum + face,
							chance.times(&Chance::one_in(faces as usize))
						);
					}
				}
				sums = next;
			}
			assignments = assignments
				.iter()
				.flat_map(|(values, chance)| {
					sums.iter().map(move |(sum, sum_chance)| {
						let mut values = values.clone();
						values.push(*sum);
						(values, chance.times(sum_chance))
					})
				})
				.collect();
		}
		let mut answers = Law::<i32>::new();
		for (values, chance) in assignments
		{
			for (sum, sum_chance) in self.sums(&values)
			{
				weigh(&mut answers, sum, chance.times(&sum_chance));
			}
		}
		let denominator = answers
			.values()
			.fold(Weight::ONE, |lcm, chance| lcm.lcm(&chance.denominator));
		Distribution::from_weights(answers.into_iter().map(|(sum, chance)| {
			(
				sum,
				chance.numerator * denominator.exact_div(&chance.denominator)
			)
		}))
		.unwrap()
	}

	/// Answer the law of the sum of the record, given the values of the
	/// registers.
	///
	/// # Parameters
	/// - `values`: The values of the registers.
	///
	/// # Returns
	/// The law of the sum.
	fn sums(&self, values: &[i32]) -> Law<i32>
	{
		let value = |op: AddressingMode| match op
		{
			AddressingMode::Immediate(immediate) => immediate.0,
			AddressingMode::Register(reg) => values[reg.0],
			AddressingMode::RollingRecord(_) => unreachable!()
		};
		// A die without faces, or an empty range, shows `0`.
		let (count, faces) = match &self.roll
		{
			RerollRoll::Standard(count, faces) => match value(*faces)
			{
				faces if faces <= 0 => (value(*count), vec![0]),
				faces => (value(*count), (1..=faces).collect())
			},
			RerollRoll::Custom(count, faces) => (value(*count), faces.clone()),
			RerollRoll::Range(start, end) =>
			{
				let (start, end) = (value(*start), value(*end));
				match start <= end
				{
					true => (1, (start..=end).collect()),
					false => (1, vec![0])
				}
			}
		};
		// Each state is the sorted multiset of the dice still kept.
		let mut states = Law::<Vec<i32>>::from([(vec![], Chance::one_in(1))]);
		for _ in 0..count.max(0)
		{
			let mut next = Law::new();
			for (kept, chance) in &states
			{
				for face in &faces
				{
					let mut kept = kept.clone();
					kept.push(*face);
					kept.sort();
					weigh(
						&mut next,
						kept,
						chance.times(&Chance::one_in(faces.len()))
					);
				}
			}
			states = next;
		}
		for clause in &self.clauses
		{
			let mut next = Law::new();
			for (kept, chance) in states
			{
				let len = kept.len();
				let clamp = |count: AddressingMode| {
					value(count).clamp(0, len as i32) as usize
				};
				let outcomes = match clause
				{
					RerollClause::DropLowest(count) =>
					{
						vec![(
							kept[clamp(*count)..].to_vec(),
							Chance::one_in(1)
						)]
					},
					RerollClause::DropHighest(count) => vec![(
						kept[..len - clamp(*count)].to_vec(),
						Chance::one_in(1)
					)],
					RerollClause::KeepLowest(count) =>
					{
						vec![(
							kept[..clamp(*count)].to_vec(),
							Chance::one_in(1)
						)]
					},
					RerollClause::KeepHighest(count) => vec![(
						kept[len - clamp(*count)..].to_vec(),
						Chance::one_in(1)
					)],
					RerollClause::Reroll(predicate, limit) => reroll_reference(
						&kept,
						&faces,
						|result| satisfies(predicate, result, value),
						match limit
						{
							Limit::Times(count) => Some(value(*count)),
							Limit::Unlimited => None
						}
					)
					.into_iter()
					.collect()
				};
				for (kept, outcome_chance) in outcomes
				{
					weigh(&mut next, kept, chance.times(&outcome_chance));
				}
			}
			states = next;
		}
		let mut sums = Law::new();
		for (kept, chance) in states
		{
			let sum = kept
				.iter()
				.fold(0i32, |sum, result| sum.saturating_add(*result));
			weigh(&mut sums, sum, chance);
		}
		sums
	}
}

/// Answer whether a result satisfies a predicate, given the values of its
/// operands, by the predicate's definition.
///
/// # Parameters
/// - `predicate`: The predicate.
/// - `result`: The result.
/// - `value`: The function that answers the value of an operand.
///
/// # Returns
/// `true` if the result satisfies the predicate, `false` otherwise.
fn satisfies(
	predicate: &Predicate,
	result: i32,
	value: impl Fn(AddressingMode) -> i32
) -> bool
{
	match predicate
	{
		Predicate::Eq(op) => result == value(*op),
		Predicate::Ne(op) => result != value(*op),
		Predicate::Lt(op) => result < value(*op),
		Predicate::Le(op) => result <= value(*op),
		Predicate::Gt(op) => result > value(*op),
		Predicate::Ge(op) => result >= value(*op),
		Predicate::In(values) => values.contains(&result),
		Predicate::InRange { start, end } =>
		{
			value(*start) <= result && result < value(*end)
		},
		Predicate::InRangeInclusive { start, end } =>
		{
			value(*start) <= result && result <= value(*end)
		},
	}
}

/// Answer the law of the dice still kept after a reroll, by its definition:
/// each die that matches rolls a replacement from the die, which rolls again
/// while it matches and the limit allows, or, without limit, finally shows
/// each face that does not match as likely as any other, unless every face
/// matches, when every die is crossed out.
///
/// # Parameters
/// - `kept`: The dice still kept, sorted.
/// - `faces`: The faces of the die, each as often as it is listed.
/// - `matches`: Whether the reroll matches a result.
/// - `limit`: The number of times that each die may be rerolled, or `None` for
///   no limit.
///
/// # Returns
/// The law of the dice still kept, sorted.
fn reroll_reference(
	kept: &[i32],
	faces: &[i32],
	matches: impl Fn(i32) -> bool,
	limit: Option<i32>
) -> Law<Vec<i32>>
{
	let one = Chance::one_in(1);
	if limit.is_some_and(|limit| limit <= 0)
	{
		return Law::from([(kept.to_vec(), one)])
	}
	if limit.is_none() && faces.iter().all(|face| matches(*face))
	{
		let kept = kept.iter().copied().filter(|result| !matches(*result));
		return Law::from([(kept.collect(), one)])
	}
	// The law of a replacement that may be rerolled `n` more times.
	fn replacement(
		faces: &[i32],
		matches: &dyn Fn(i32) -> bool,
		n: Option<i32>
	) -> Law<i32>
	{
		let mut law = Law::new();
		match n
		{
			None =>
			{
				let rest = faces
					.iter()
					.filter(|face| !matches(**face))
					.collect::<Vec<_>>();
				for face in &rest
				{
					weigh(&mut law, **face, Chance::one_in(rest.len()));
				}
			},
			Some(n) =>
			{
				for face in faces
				{
					let chance = Chance::one_in(faces.len());
					match matches(*face) && n > 0
					{
						true =>
						{
							for (result, again) in
								replacement(faces, matches, Some(n - 1))
							{
								weigh(&mut law, result, chance.times(&again));
							}
						},
						false => weigh(&mut law, *face, chance)
					}
				}
			}
		}
		law
	}
	let mut states = Law::from([(Vec::<i32>::new(), one)]);
	for result in kept
	{
		let die = match matches(*result)
		{
			true => replacement(faces, &matches, limit.map(|limit| limit - 1)),
			false => Law::from([(*result, Chance::one_in(1))])
		};
		let mut next = Law::new();
		for (state, chance) in &states
		{
			for (result, result_chance) in &die
			{
				let mut state = state.clone();
				state.push(*result);
				state.sort();
				weigh(&mut next, state, chance.times(result_chance));
			}
		}
		states = next;
	}
	states
}

/// Answer a strategy that generates an operand: an immediate in the specified
/// range, or one of the specified number of registers.
///
/// # Parameters
/// - `registers`: The number of registers.
/// - `least`: The least immediate.
/// - `greatest`: The greatest immediate.
///
/// # Returns
/// The strategy.
fn reroll_operand(
	registers: usize,
	least: i32,
	greatest: i32
) -> BoxedStrategy<AddressingMode>
{
	let immediate =
		(least..=greatest).prop_map(|value| Immediate(value).into());
	match registers
	{
		0 => immediate.boxed(),
		registers => prop_oneof![
			immediate,
			(0..registers).prop_map(|reg| RegisterIndex(reg).into())
		]
		.boxed()
	}
}

/// Answer a strategy that generates a reroll clause, of every predicate form,
/// limited or not.
///
/// # Parameters
/// - `registers`: The number of registers.
///
/// # Returns
/// The strategy.
fn reroll_clause(registers: usize) -> BoxedStrategy<RerollClause>
{
	let op = move || reroll_operand(registers, -1, 5);
	let predicate = prop_oneof![
		op().prop_map(Predicate::Eq),
		op().prop_map(Predicate::Ne),
		op().prop_map(Predicate::Lt),
		op().prop_map(Predicate::Le),
		op().prop_map(Predicate::Gt),
		op().prop_map(Predicate::Ge),
		prop::collection::vec(-2..=5, 0..=3).prop_map(Predicate::In),
		(op(), op()).prop_map(|(start, end)| Predicate::InRange { start, end }),
		(op(), op()).prop_map(|(start, end)| {
			Predicate::InRangeInclusive { start, end }
		})
	];
	let limit = prop_oneof![
		reroll_operand(registers, -1, 3).prop_map(Limit::Times),
		Just(Limit::Unlimited)
	];
	(predicate, limit)
		.prop_map(|(predicate, limit)| RerollClause::Reroll(predicate, limit))
		.boxed()
}

/// Answer a strategy that generates a [reroll program](RerollProgram): up to
/// two registers, standard or custom dice or a range, and a reroll among up
/// to two other clauses, drops, keeps, or rerolls.
///
/// # Returns
/// The strategy.
fn reroll_program() -> impl Strategy<Value = RerollProgram>
{
	prop::collection::vec((Just(1), 2..=3, -1..=1), 0..=2).prop_flat_map(
		|registers| {
			let n = registers.len();
			let roll = prop_oneof![
				(reroll_operand(n, 0, 3), reroll_operand(n, -1, 4)).prop_map(
					|(count, faces)| RerollRoll::Standard(count, faces)
				),
				(
					reroll_operand(n, 0, 3),
					prop::collection::vec(-2..=3, 1..=3)
				)
					.prop_map(|(count, faces)| {
						RerollRoll::Custom(count, faces)
					}),
				(reroll_operand(n, -1, 3), reroll_operand(n, -1, 3))
					.prop_map(|(start, end)| RerollRoll::Range(start, end))
			];
			let count = move || reroll_operand(n, -1, 3);
			let clause = prop_oneof![
				count().prop_map(RerollClause::DropLowest),
				count().prop_map(RerollClause::DropHighest),
				count().prop_map(RerollClause::KeepLowest),
				count().prop_map(RerollClause::KeepHighest),
				reroll_clause(n)
			];
			(
				Just(registers),
				roll,
				prop::collection::vec(clause, 0..=2),
				reroll_clause(n),
				any::<prop::sample::Index>()
			)
				.prop_map(
					|(registers, roll, mut clauses, reroll, at)| {
						clauses.insert(at.index(clauses.len() + 1), reroll);
						RerollProgram {
							registers,
							roll,
							clauses
						}
					}
				)
		}
	)
}

proptest! {
	#![proptest_config(ProptestConfig::with_cases(512))]

	/// Rerolls answer the law of their definition, optimized or not, whether
	/// before or after drops and keeps, of every predicate form, limited or
	/// not, with fixed or random operands, some of them shared, and their
	/// totals are the counts of outcomes that their bounds report.
	#[test]
	fn test_rerolls_agree_with_reference(program in reroll_program())
	{
		let assembly = program.assembly();
		let function = Assembler::assemble(&assembly).unwrap();
		let expected = program.reference();
		for function in [function.clone(), optimize(function, Passes::all())]
		{
			let evaluator = Evaluator::new(function);
			let propagated = propagate(&evaluator, []).unwrap();
			prop_assert!(
				agree(&propagated, &expected),
				"{assembly}\npropagated:\n{propagated}\nexpected:\n{expected}"
			);
			if let Some(count) = evaluator.bounds_over([], []).unwrap().count
				&& count < u128::MAX
			{
				prop_assert_eq!(
					propagated.total(),
					&Weight::from(count),
					"{}",
					assembly
				);
			}
		}
	}
}
