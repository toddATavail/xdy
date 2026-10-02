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
	cell::RefCell, collections::HashSet, mem, ops::ControlFlow, sync::OnceLock
};

use pretty_assertions::assert_eq;
use proptest::{
	option, prelude::*, sample::select, test_runner::TestCaseError
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	Add, AddressingMode, Assembler, BuildError, Distribution, EvaluationError,
	Evaluator, Function, Instruction, Passes, ProgramCounter, Rational,
	RegisterIndex, Return, RollingRecordIndex, SumRollingRecord, Weight,
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

/// Answer a strategy that generates dice whose counts, faces, and drops are
/// fixed, with custom faces that often reach the extremes of [`i32`], so that
/// their sums saturate.
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
	let clause = (select(&["lowest", "highest"][..]), option::of(drop))
		.prop_map(|(direction, drop)| match drop
		{
			Some(drop) => format!(" drop {} {}", direction, drop),
			None => format!(" drop {}", direction)
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
/// counts, and ranges whose endpoints, may be random.
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
		// A drop expression never begins with `-`, so every drop is grouped.
		let clause = (select(&["lowest", "highest"][..]), operand.clone())
			.prop_map(|(direction, drop)| {
				format!(" drop {} ({})", direction, drop)
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
