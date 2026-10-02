//! # Sampling tests
//!
//! Herein are the tests of [Monte Carlo sampling](Evaluator::sample). The
//! exact distributions of the [forward pass](crate::DistributionPlan::build)
//! are the reference: at any confidence, the cumulative distribution function
//! of a sample must stay within its [error bound](Sampled::error_bound) of the
//! exact one, at every outcome at once, at least as often as the confidence
//! promises. Every seed is fixed, so that the tests are deterministic.

use std::num::NonZeroU64;

use pretty_assertions::assert_eq;
use rand::{Rng as _, SeedableRng, rngs::StdRng};

use crate::{
	Budget, Distribution, EvaluationError, Evaluator, Probability, Sampled,
	Unobserved, Weight, support::compile_valid
};

////////////////////////////////////////////////////////////////////////////////
//                                  Support.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Answer the number of samples as a [`NonZeroU64`].
///
/// # Parameters
/// - `n`: The number of samples, which must not be zero.
///
/// # Returns
/// The number of samples.
///
/// # Panics
/// If `n` is zero.
fn samples(n: u64) -> NonZeroU64 { NonZeroU64::new(n).unwrap() }

/// Answer the confidence `numerator/denominator`.
///
/// # Parameters
/// - `numerator`: The numerator.
/// - `denominator`: The denominator.
///
/// # Returns
/// The confidence.
fn confidence(numerator: u8, denominator: u8) -> Probability
{
	Probability::new(Weight::from(numerator), Weight::from(denominator))
		.unwrap()
}

/// Answer the exact distribution of the outcomes of the specified evaluator's
/// function, for no arguments.
///
/// # Parameters
/// - `evaluator`: The evaluator.
///
/// # Returns
/// The distribution.
fn exact(evaluator: &Evaluator) -> Distribution
{
	evaluator
		.plan_distribution([])
		.unwrap()
		.build(Budget::UNLIMITED, &Unobserved)
		.unwrap()
}

/// Answer the greatest difference between the cumulative distribution
/// functions of the counts of the specified sample and the specified exact
/// distribution, which is the statistic that the error bound bounds.
///
/// # Parameters
/// - `sampled`: The sample.
/// - `exact`: The exact distribution, whose support includes every outcome of
///   the sample.
///
/// # Returns
/// The greatest difference, over every outcome.
fn deviation(sampled: &Sampled, exact: &Distribution) -> f64
{
	// Both functions step only at outcomes of the exact distribution, so the
	// greatest difference is at one of them.
	exact
		.iter()
		.map(|(outcome, _)| {
			(sampled.counts().cdf(outcome).to_f64()
				- exact.cdf(outcome).to_f64())
			.abs()
		})
		.fold(0.0, f64::max)
}

////////////////////////////////////////////////////////////////////////////////
//                               Error bounds.                                //
////////////////////////////////////////////////////////////////////////////////

/// Test that the error bound is `√(ln(2/δ) / 2n)`: 0.0061 at 50,000 samples
/// and 95% confidence, halved by four times the samples, and infinite at
/// certainty.
#[test]
fn test_error_bound()
{
	// A constant rolls no dice, so it samples within an empty budget.
	let mut evaluator = Evaluator::new(compile_valid("1"));
	let mut rng = StdRng::seed_from_u64(0);
	let sampled = evaluator.sample([], &mut rng, samples(50_000), 0).unwrap();
	let epsilon = sampled.error_bound(&confidence(19, 20));
	assert!((epsilon - (40f64.ln() / 100_000.0).sqrt()).abs() < 1e-15);
	assert!((epsilon - 0.006_073_6).abs() < 1e-7, "{epsilon}");
	assert_eq!(format!("{epsilon:.4}"), "0.0061");
	let quadrupled =
		evaluator.sample([], &mut rng, samples(200_000), 0).unwrap();
	let halved = quadrupled.error_bound(&confidence(19, 20));
	assert!((halved - epsilon / 2.0).abs() < 1e-15);
	assert_eq!(
		sampled.error_bound(&Probability::ZERO),
		(2f64.ln() / 100_000.0).sqrt()
	);
	assert_eq!(sampled.error_bound(&Probability::ONE), f64::INFINITY);
	// An unreduced confidence bounds as its reduction does.
	assert_eq!(
		sampled.error_bound(&confidence(38, 40)),
		sampled.error_bound(&confidence(19, 20))
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                 Accuracy.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that 50,000 samples of each of several known distributions stay
/// within the error bound at 95% confidence of the exact distribution, and
/// that their counts total the samples: pools with and without drops, a
/// computed count, and a correlated value.
#[test]
fn test_sample_within_error_bound()
{
	for source in [
		"3D6",
		"4D6 drop lowest 1",
		"(1D4)D6",
		"[1:6] + 1D20 drop highest",
		"{x}@(1D6) + {x} * 2"
	]
	{
		let mut evaluator = Evaluator::new(compile_valid(source));
		let exact = exact(&evaluator);
		// The seed is arbitrary, chosen by smashing the keyboard. This is to
		// ensure that the test cases are deterministic.
		let mut rng = StdRng::seed_from_u64(7093814572093487);
		let n = samples(50_000);
		let sampled = evaluator.sample([], &mut rng, n, u64::MAX).unwrap();
		assert_eq!(sampled.samples(), n, "{source}");
		assert_eq!(
			sampled.counts().total(),
			&Weight::from(n.get()),
			"{source}"
		);
		assert!(
			sampled.counts().min() >= exact.min()
				&& sampled.counts().max() <= exact.max(),
			"{source}"
		);
		let epsilon = sampled.error_bound(&confidence(19, 20));
		let deviation = deviation(&sampled, &exact);
		assert!(deviation <= epsilon, "{source}: {deviation} > {epsilon}");
	}
}

/// Test that samples stray beyond the error bound no more often than the
/// confidence allows: of 200 samples of 500 rolls of `2D6`, each from its own
/// seed, no more than 10% stray beyond the bound at 90% confidence.
#[test]
fn test_sample_confidence()
{
	let mut evaluator = Evaluator::new(compile_valid("2D6"));
	let exact = exact(&evaluator);
	let confidence = confidence(9, 10);
	let strays = (0..200u64)
		.filter(|&seed| {
			let mut rng = StdRng::seed_from_u64(seed);
			let sampled = evaluator
				.sample([], &mut rng, samples(500), u64::MAX)
				.unwrap();
			deviation(&sampled, &exact) > sampled.error_bound(&confidence)
		})
		.count();
	assert!(strays <= 20, "{strays} of 200 samples strayed");
}

////////////////////////////////////////////////////////////////////////////////
//                                  Budgets.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Test that a dice budget that the first roll would exceed is refused before
/// it rolls anything, as is a bad arity, without drawing from the pRNG.
#[test]
fn test_sample_refuses_before_rolling()
{
	let mut evaluator = Evaluator::new(compile_valid("{x}: {x}D6"));
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let seed = 3409857120983475;
	let mut rng = StdRng::seed_from_u64(seed);
	let mut untouched = StdRng::seed_from_u64(seed);
	assert_eq!(
		evaluator.sample([i32::MAX], &mut rng, samples(50_000), 1_000_000),
		Err(EvaluationError::DiceBudgetExhausted {
			requested: i32::MAX as u64,
			remaining: 1_000_000,
			consumed: 0
		})
	);
	assert_eq!(
		evaluator.sample([], &mut rng, samples(50_000), 1_000_000),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
	assert_eq!(rng.next_u64(), untouched.next_u64());
}

/// Test that one dice budget covers every evaluation together: a budget of
/// exactly the dice of every sample suffices, and one die less is refused
/// partway through an evaluation, reporting what remains of the whole budget
/// and the dice that every evaluation consumed.
#[test]
fn test_sample_dice_budget_spans_samples()
{
	// Each evaluation rolls one die for its count, and then that one die.
	let mut evaluator = Evaluator::new(compile_valid("{x}: ({x}D1)D6"));
	// The seed is arbitrary, chosen by smashing the keyboard. This is to
	// ensure that the test cases are deterministic.
	let mut rng = StdRng::seed_from_u64(8471029384710293);
	let sampled = evaluator.sample([1], &mut rng, samples(3), 6).unwrap();
	assert_eq!(sampled.counts().total(), &Weight::from(3u8));
	assert_eq!(
		evaluator.sample([1], &mut rng, samples(3), 5),
		Err(EvaluationError::DiceBudgetExhausted {
			requested: 1,
			remaining: 0,
			consumed: 5
		})
	);
}

////////////////////////////////////////////////////////////////////////////////
//                       Formatting and serialization.                        //
////////////////////////////////////////////////////////////////////////////////

/// Test that a sample formats with its label, and then its counts.
#[test]
fn test_sample_display()
{
	let mut evaluator = Evaluator::new(compile_valid("1D1 + 2"));
	let mut rng = StdRng::seed_from_u64(0);
	let sampled = evaluator.sample([], &mut rng, samples(3), 3).unwrap();
	assert_eq!(sampled.to_string(), "sampled (n = 3):\n3: 3\n");
}

/// Test that a sample serializes as its number of samples and its counts,
/// and deserializes only when its counts total its samples.
#[cfg(feature = "serde")]
#[test]
fn test_sample_serde()
{
	let mut evaluator = Evaluator::new(compile_valid("1D1 + 2"));
	let mut rng = StdRng::seed_from_u64(0);
	let sampled = evaluator.sample([], &mut rng, samples(3), 3).unwrap();
	let json = serde_json::to_string(&sampled).unwrap();
	assert_eq!(
		json,
		r#"{"samples":3,"counts":{"weights":{"3":"3"},"total":"3"}}"#
	);
	assert_eq!(serde_json::from_str::<Sampled>(&json).unwrap(), sampled);
	for (invalid, reason) in [
		(
			r#"{"samples":4,"counts":{"weights":{"3":"3"},"total":"3"}}"#,
			"total 3, not its 4 samples"
		),
		(
			r#"{"samples":0,"counts":{"weights":{"3":"3"},"total":"3"}}"#,
			"nonzero"
		),
		(
			r#"{"samples":3,"counts":{"weights":{"3":"2"},"total":"3"}}"#,
			"not the sum"
		),
		(
			r#"{"samples":3,"counts":{"weights":{"3":"3"},"total":"3"},"x":1}"#,
			"unknown field"
		)
	]
	{
		let error = serde_json::from_str::<Sampled>(invalid).unwrap_err();
		assert!(error.to_string().contains(reason), "{invalid}: {error}");
	}
}
