//! # Distribution tests
//!
//! Herein are the tests for [`Distribution`], [`Probability`], and
//! [`Rational`], the exact distribution of a dice expression and the exact
//! probabilities and other rational numbers that it answers. A map from
//! outcomes to plain [`BigUint`] weights is the reference throughout: every
//! query must agree with it exactly.
//!
//! Every conversion to [`f64`] must be correctly rounded, ties to even. Rather
//! than trust a second implementation of the rounding, the tests check each
//! answer against the definition: the exact quotient must lie between the
//! midpoints that separate the answer from its neighbors.

use std::{cmp::Ordering, collections::BTreeMap};

use num_bigint::{BigInt, BigUint, Sign};
use proptest::{collection::vec, prelude::*};

use super::weight::value;
use crate::{
	Distribution, EmptyDistributionError, Probability, ProbabilityError,
	Rational, Weight, ZeroDenominatorError
};

////////////////////////////////////////////////////////////////////////////////
//                                  Support.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Answer the weight with the given value.
///
/// # Parameters
/// - `n`: The value.
///
/// # Returns
/// The weight.
fn w(n: u128) -> Weight { Weight::from(n) }

/// Answer the probability with the given numerator and denominator.
///
/// # Parameters
/// - `numerator`: The numerator.
/// - `denominator`: The denominator.
///
/// # Returns
/// The probability.
///
/// # Panics
/// If the numerator and denominator do not make a probability.
fn p(numerator: u128, denominator: u128) -> Probability
{
	Probability::new(w(numerator), w(denominator)).unwrap()
}

/// Answer the rational number with the given sign, numerator, and
/// denominator.
///
/// # Parameters
/// - `negative`: Whether the number is negative.
/// - `numerator`: The magnitude of the numerator.
/// - `denominator`: The denominator.
///
/// # Returns
/// The rational number.
///
/// # Panics
/// If the denominator is zero.
fn r(negative: bool, numerator: u128, denominator: u128) -> Rational
{
	Rational::new(negative, w(numerator), w(denominator)).unwrap()
}

/// Answer the distribution with the given weights.
///
/// # Parameters
/// - `weights`: The outcomes and their weights.
///
/// # Returns
/// The distribution.
///
/// # Panics
/// If no weight is nonzero.
fn distribution(weights: &[(i32, u128)]) -> Distribution
{
	Distribution::from_weights(weights.iter().map(|&(v, n)| (v, w(n)))).unwrap()
}

/// Answer the distribution of `(1D3)D3`.
///
/// # Returns
/// The distribution, over a total of 81.
fn one_d3_d3() -> Distribution
{
	let weights = [9, 12, 16, 12, 12, 10, 6, 3, 1];
	distribution(&(1..).zip(weights).collect::<Vec<_>>())
}

/// Answer 2 raised to the given power, as a big integer.
///
/// # Parameters
/// - `exponent`: The exponent.
///
/// # Returns
/// The power.
fn pow2(exponent: u64) -> BigUint { BigUint::from(1u8) << exponent }

/// An exact rational number, as a signed numerator over a positive
/// denominator.
type Ratio = (BigInt, BigUint);

/// Answer the exact value of a finite [`f64`].
///
/// # Parameters
/// - `x`: The float, which must be finite.
///
/// # Returns
/// The value of `x`.
fn exact(x: f64) -> Ratio
{
	assert!(x.is_finite());
	let bits = x.to_bits();
	let biased = ((bits >> 52) & 0x7ff) as i64;
	let fraction = bits & ((1 << 52) - 1);
	let (significand, exponent) = match biased
	{
		0 => (fraction, -1074),
		_ => (fraction | (1 << 52), biased - 1075)
	};
	let sign = if x.is_sign_negative()
	{
		Sign::Minus
	}
	else
	{
		Sign::Plus
	};
	let significand = BigInt::from_biguint(sign, BigUint::from(significand));
	match exponent
	{
		e if e >= 0 => (significand << e as u64, BigUint::from(1u8)),
		e => (significand, pow2(-e as u64))
	}
}

/// Compare two exact rational numbers.
///
/// # Parameters
/// - `a`: The first number.
/// - `b`: The second number.
///
/// # Returns
/// The ordering of `a` and `b`.
fn compare(a: &Ratio, b: &Ratio) -> Ordering
{
	(&a.0 * BigInt::from(b.1.clone())).cmp(&(&b.0 * BigInt::from(a.1.clone())))
}

/// Answer the midpoint of two exact rational numbers.
///
/// # Parameters
/// - `a`: The first number.
/// - `b`: The second number.
///
/// # Returns
/// The number halfway between `a` and `b`.
fn midpoint(a: &Ratio, b: &Ratio) -> Ratio
{
	let numerator =
		&a.0 * BigInt::from(b.1.clone()) + &b.0 * BigInt::from(a.1.clone());
	(numerator, &a.1 * &b.1 * 2u8)
}

/// Check that a float is the correctly rounded value of an exact rational
/// number, rounding ties to even.
///
/// # Parameters
/// - `answer`: The float.
/// - `q`: The exact number.
///
/// # Errors
/// If `answer` is not the float nearest to `q`, or is the farther of two
/// equally near, or is NaN.
fn check_rounding(answer: f64, q: &Ratio) -> Result<(), TestCaseError>
{
	prop_assert!(!answer.is_nan(), "{:?} rounded to NaN", q);
	prop_assert!(
		answer == 0.0
			|| answer.is_sign_negative() == (q.0.sign() == Sign::Minus),
		"{} has the wrong sign for {:?}",
		answer,
		q
	);
	// Round the magnitude, since rounding to nearest is symmetric.
	let q = (BigInt::from(q.0.magnitude().clone()), q.1.clone());
	let answer = answer.abs();
	let even = answer.to_bits() & 1 == 0;
	// A bound is admissible if q lies strictly within it, or on it when the
	// answer is even, which wins ties.
	let within = |bound: &Ratio, side: Ordering| match compare(&q, bound)
	{
		Ordering::Equal => even,
		ordering => ordering == side
	};
	// The least number that rounds to infinity: halfway from f64::MAX to
	// 2¹⁰²⁴, where the next float would be if the exponent went further.
	let overflow = (BigInt::from(pow2(1024) - pow2(970)), BigUint::from(1u8));
	if answer.is_infinite()
	{
		prop_assert!(
			compare(&q, &overflow) != Ordering::Less,
			"{:?} rounded to infinity",
			q
		);
		return Ok(())
	}
	let value = exact(answer);
	if answer > 0.0
	{
		let below = midpoint(&exact(answer.next_down()), &value);
		prop_assert!(
			within(&below, Ordering::Greater),
			"{:?} rounded up to {}",
			q,
			answer
		);
	}
	let above = match answer == f64::MAX
	{
		true => overflow,
		false => midpoint(&value, &exact(answer.next_up()))
	};
	prop_assert!(
		within(&above, Ordering::Less),
		"{:?} rounded down to {}",
		q,
		answer
	);
	Ok(())
}

/// Answer the exact value of a quotient of big integers.
///
/// # Parameters
/// - `numerator`: The numerator.
/// - `denominator`: The denominator.
///
/// # Returns
/// The quotient.
fn ratio(numerator: &BigUint, denominator: &BigUint) -> Ratio
{
	(BigInt::from(numerator.clone()), denominator.clone())
}

/// Answer the exact value of a signed quotient of big integers.
///
/// # Parameters
/// - `negative`: Whether to negate the quotient.
/// - `numerator`: The magnitude of the numerator.
/// - `denominator`: The denominator.
///
/// # Returns
/// The quotient, negated if `negative`.
fn signed(negative: bool, numerator: &BigUint, denominator: &BigUint) -> Ratio
{
	let sign = if negative { Sign::Minus } else { Sign::Plus };
	(
		BigInt::from_biguint(sign, numerator.clone()),
		denominator.clone()
	)
}

/// Answer the reference for the weights of a distribution: the sums of the
/// weights of each outcome, omitting outcomes whose sum is zero.
///
/// # Parameters
/// - `weights`: The outcomes and their weights.
///
/// # Returns
/// The reference.
fn reference(weights: &[(i32, BigUint)]) -> BTreeMap<i32, BigUint>
{
	let mut merged = BTreeMap::<i32, BigUint>::new();
	for (outcome, weight) in weights
	{
		*merged.entry(*outcome).or_default() += weight;
	}
	merged.retain(|_, weight| *weight != BigUint::ZERO);
	merged
}

/// Answer the distribution with the given weights.
///
/// # Parameters
/// - `weights`: The outcomes and their weights.
///
/// # Returns
/// The distribution, or [`EmptyDistributionError`] if no weight is nonzero.
fn from_big(
	weights: &[(i32, BigUint)]
) -> Result<Distribution, EmptyDistributionError>
{
	Distribution::from_weights(
		weights
			.iter()
			.map(|(v, n)| (*v, Weight::from_biguint(n.clone())))
	)
}

/// Answer a strategy that generates outcomes: small ones, which collide
/// often, the extremes of [`i32`], and anything between.
///
/// # Returns
/// The strategy.
fn outcome() -> impl Strategy<Value = i32>
{
	prop_oneof![-8i32..=8, Just(i32::MIN), Just(i32::MAX), any::<i32>()]
}

/// Answer a strategy that generates the outcomes and weights of a
/// distribution, possibly with repeated outcomes and zero weights, and
/// possibly with no nonzero weight at all.
///
/// # Returns
/// The strategy.
fn weights() -> impl Strategy<Value = Vec<(i32, BigUint)>>
{
	vec((outcome(), value()), 0..=12)
}

/// Answer a strategy that generates the weights of distributions with at
/// least one nonzero weight.
///
/// # Returns
/// The strategy.
fn nonempty_weights() -> impl Strategy<Value = Vec<(i32, BigUint)>>
{
	weights().prop_filter("some weight is nonzero", |weights| {
		!reference(weights).is_empty()
	})
}

/// Answer a strategy that generates big integers across the whole range of
/// [`f64`] exponents and beyond, so that their quotients cover overflow,
/// subnormals, and underflow.
///
/// # Returns
/// The strategy.
fn wide() -> impl Strategy<Value = BigUint>
{
	prop_oneof![
		value(),
		(value(), 0u64..=1200).prop_map(|(v, shift)| v << shift),
		(1u64..=8, 1070u64..=1080).prop_map(|(v, shift)| pow2(shift) + v),
		(1u64..=8, 1070u64..=1080).prop_map(|(v, shift)| pow2(shift) - v)
	]
}

/// Answer a strategy that generates probabilities, as the numerators and
/// denominators of their reference values, across the whole range of `f64`
/// exponents.
///
/// # Returns
/// The strategy.
fn probability() -> impl Strategy<Value = (BigUint, BigUint)>
{
	(wide(), wide()).prop_map(|(a, b)| match a.cmp(&b)
	{
		_ if b == BigUint::ZERO && a == BigUint::ZERO =>
		{
			(BigUint::ZERO, BigUint::from(1u8))
		},
		Ordering::Greater => (b, a),
		_ => (a, b)
	})
}

/// Answer a strategy that generates rational numbers, as the signs,
/// numerators, and denominators of their reference values, across the whole
/// range of `f64` exponents.
///
/// # Returns
/// The strategy.
fn rational() -> impl Strategy<Value = (bool, BigUint, BigUint)>
{
	(
		any::<bool>(),
		wide(),
		wide().prop_filter("nonzero", |d| *d != BigUint::ZERO)
	)
}

////////////////////////////////////////////////////////////////////////////////
//                               Probabilities.                               //
////////////////////////////////////////////////////////////////////////////////

/// A probability needs a nonzero denominator that is not less than its
/// numerator.
#[test]
fn test_probability_construction()
{
	assert_eq!(
		Probability::new(w(0), w(0)),
		Err(ProbabilityError::ZeroDenominator)
	);
	assert_eq!(
		Probability::new(w(3), w(2)),
		Err(ProbabilityError::ExceedsOne)
	);
	let half = p(2, 4);
	assert_eq!(half.numerator(), &w(2));
	assert_eq!(half.denominator(), &w(4));
	assert_eq!(Probability::ZERO.to_string(), "0/1");
	assert_eq!(Probability::ONE.to_string(), "1/1");
}

/// Probabilities compare as rational numbers, unreduced.
#[test]
fn test_probability_comparison()
{
	assert_eq!(p(2, 4), p(1, 2));
	assert_eq!(p(2, 4).to_string(), "2/4");
	assert_eq!(p(0, 7), Probability::ZERO);
	assert_eq!(p(7, 7), Probability::ONE);
	assert!(p(1, 3) < p(1, 2));
	assert!(p(2, 3) > p(3, 5));
	assert_eq!(p(1, 3).cmp(&p(2, 6)), Ordering::Equal);
}

////////////////////////////////////////////////////////////////////////////////
//                                 Rationals.                                 //
////////////////////////////////////////////////////////////////////////////////

/// A rational number needs a nonzero denominator, and is never a negative
/// zero.
#[test]
fn test_rational_construction()
{
	assert_eq!(Rational::new(false, w(1), w(0)), Err(ZeroDenominatorError));
	assert_eq!(Rational::new(true, w(0), w(0)), Err(ZeroDenominatorError));
	let loss = r(true, 21, 6);
	assert!(loss.is_negative());
	assert_eq!(loss.numerator(), &w(21));
	assert_eq!(loss.denominator(), &w(6));
	assert_eq!(loss.to_string(), "-21/6");
	assert_eq!(r(false, 21, 6).to_string(), "21/6");
	let zero = r(true, 0, 5);
	assert!(!zero.is_negative());
	assert_eq!(zero.to_string(), "0/5");
	assert_eq!(Rational::ZERO.to_string(), "0/1");
	assert_eq!(Rational::ONE.to_string(), "1/1");
}

/// Rational numbers compare as rational numbers, unreduced, by sign and then
/// by magnitude.
#[test]
fn test_rational_comparison()
{
	assert_eq!(r(true, 21, 6), r(true, 7, 2));
	assert_ne!(r(true, 7, 2), r(false, 7, 2));
	assert_eq!(r(true, 0, 5), Rational::ZERO);
	assert_eq!(r(false, 4, 4), Rational::ONE);
	assert!(r(true, 1, 2) < Rational::ZERO);
	assert!(Rational::ZERO < r(false, 1, 1000));
	assert!(r(true, 9, 1) < r(false, 1, 9));
	assert!(r(false, 3, 7) < r(false, 4, 7));
	assert!(r(true, 3, 7) > r(true, 4, 7));
	assert!(r(true, 1, 2) < r(true, 1, 3));
	assert_eq!(r(true, 1, 3).cmp(&r(true, 2, 6)), Ordering::Equal);
}

/// Rational numbers convert to floats of their sign, even when they round to
/// zero or to infinity, but zero converts to positive zero.
#[test]
fn test_rational_to_f64()
{
	assert_eq!(r(true, 21, 6).to_f64(), -3.5);
	assert_eq!(r(false, 21, 6).to_f64(), 3.5);
	assert_eq!(r(true, 0, 6).to_f64().to_bits(), 0f64.to_bits());
	let big = Weight::from_biguint;
	let tiny = Rational::new(true, w(1), big(pow2(5000))).unwrap();
	assert_eq!(tiny.to_f64().to_bits(), (-0f64).to_bits());
	let vast = Rational::new(true, big(pow2(5000)), w(1)).unwrap();
	assert_eq!(vast.to_f64(), f64::NEG_INFINITY);
}

////////////////////////////////////////////////////////////////////////////////
//                        Floating-point conversion.                          //
////////////////////////////////////////////////////////////////////////////////

/// Quotients round correctly at the edges of the subnormals, at the edge of
/// overflow, and when both operands are beyond [`f64::MAX`].
#[test]
fn test_ratio_to_f64_edges()
{
	let big = Weight::from_biguint;
	let ratio = |n: BigUint, d: BigUint| big(n).ratio_to_f64(&big(d));
	let one = || BigUint::from(1u8);
	let min_subnormal = f64::from_bits(1);
	assert_eq!(ratio(one(), pow2(1074)), min_subnormal);
	// Exactly half the least subnormal ties to even, which is zero.
	assert_eq!(ratio(one(), pow2(1075)), 0.0);
	assert_eq!(ratio(one() + 0u8, pow2(1075) - 1u8), min_subnormal);
	assert_eq!(ratio(BigUint::from(3u8), pow2(1076)), min_subnormal);
	assert_eq!(ratio(one(), pow2(5000)), 0.0);
	// The least normal f64.
	assert_eq!(ratio(one(), pow2(1022)), f64::MIN_POSITIVE);
	assert_eq!(ratio(pow2(1023), one()), 2f64.powi(1023));
	// Exactly halfway from f64::MAX to 2¹⁰²⁴ ties to even, which is infinity.
	assert_eq!(ratio(pow2(1024) - pow2(970), one()), f64::INFINITY);
	assert_eq!(ratio(pow2(1024) - pow2(970) - 1u8, one()), f64::MAX);
	assert_eq!(ratio(pow2(3000), pow2(1976)), f64::INFINITY);
	// Both operands beyond f64::MAX.
	let total = BigUint::from(6u8).pow(1000);
	assert_eq!(ratio(&total - 1u8, total.clone()), 1.0);
	assert_eq!(ratio(total.clone(), &total * 2u8), 0.5);
	assert_eq!(ratio(total.clone() * 3u8, total), 3.0);
	assert_eq!(w(0).ratio_to_f64(&w(5)), 0.0);
}

/// Division by a zero weight panics.
#[test]
#[should_panic(expected = "division by zero weight")]
fn test_ratio_to_f64_by_zero() { w(1).ratio_to_f64(&w(0)); }

////////////////////////////////////////////////////////////////////////////////
//                               Distributions.                               //
////////////////////////////////////////////////////////////////////////////////

/// A point mass weighs its one outcome as one, over one.
#[test]
fn test_point()
{
	let point = Distribution::point(-4);
	assert_eq!(point.len(), 1);
	assert_eq!(point.get(-4), &w(1));
	assert_eq!(point.get(4), &w(0));
	assert_eq!(point.total(), &w(1));
	assert_eq!((point.min(), point.max()), (-4, -4));
	assert_eq!(point.probability(-4), Probability::ONE);
	assert_eq!(point.cdf(-5), Probability::ZERO);
	assert_eq!(point.cdf(-4), Probability::ONE);
	assert_eq!(point.quantile(&Probability::ZERO), -4);
	assert_eq!(point.quantile(&Probability::ONE), -4);
	assert_eq!(point.mean().to_string(), "-4/1");
	assert_eq!(point.to_f64(), BTreeMap::from([(-4, 1.0)]));
	let least = Distribution::point(i32::MIN).mean();
	assert_eq!(least.to_string(), "-2147483648/1");
	assert_eq!(least.to_f64(), i32::MIN as f64);
	let greatest = Distribution::point(i32::MAX).mean();
	assert_eq!(greatest.to_string(), "2147483647/1");
	assert_eq!(greatest.to_f64(), i32::MAX as f64);
}

/// Construction sums the weights of repeated outcomes, omits zero weights,
/// and refuses weights that are all zero.
#[test]
fn test_from_weights()
{
	let merged = distribution(&[(3, 2), (1, 0), (3, 5), (2, 1), (2, 0)]);
	assert_eq!(merged.len(), 2);
	assert_eq!(merged.iter().collect::<Vec<_>>(), [(2, &w(1)), (3, &w(7))]);
	assert_eq!(merged.total(), &w(8));
	assert_eq!(merged.get(1), &w(0));
	assert_eq!(Distribution::from_weights([]), Err(EmptyDistributionError));
	assert_eq!(
		Distribution::from_weights([(1, w(0)), (2, w(0))]),
		Err(EmptyDistributionError)
	);
}

/// Equality compares weights, not merely probabilities, since weights are
/// never reduced.
#[test]
fn test_unreduced_equality()
{
	let coin = distribution(&[(0, 1), (1, 1)]);
	let doubled = distribution(&[(0, 2), (1, 2)]);
	assert_ne!(coin, doubled);
	assert_eq!(coin.probability(1), doubled.probability(1));
	assert_eq!(doubled.probability(1).to_string(), "2/4");
	assert_eq!(coin, distribution(&[(1, 1), (0, 1)]));
}

/// The distribution of `(1D3)D3` answers its exact probabilities, cumulative
/// probabilities, quantiles, and mean.
#[test]
fn test_one_d3_d3()
{
	let d = one_d3_d3();
	assert_eq!(d.total(), &w(81));
	assert_eq!((d.min(), d.max(), d.len()), (1, 9, 9));
	let cumulative = [0, 9, 21, 37, 49, 61, 71, 77, 80, 81, 81];
	for (outcome, expected) in (0..).zip(cumulative)
	{
		assert_eq!(d.cdf(outcome).to_string(), format!("{expected}/81"));
	}
	assert_eq!(d.probability(3).to_string(), "16/81");
	assert_eq!(d.probability(10).to_string(), "0/81");
	assert_eq!(d.quantile(&Probability::ZERO), 1);
	assert_eq!(d.quantile(&p(9, 81)), 1);
	assert_eq!(d.quantile(&p(10, 81)), 2);
	assert_eq!(d.quantile(&p(1, 2)), 4);
	assert_eq!(d.quantile(&p(80, 81)), 8);
	assert_eq!(d.quantile(&Probability::ONE), 9);
	assert_eq!(d.mean().to_string(), "324/81");
	assert_eq!(d.mean(), r(false, 4, 1));
	assert_eq!(d.mean().to_f64(), 4.0);
	assert_eq!(d.to_f64()[&3], 16.0 / 81.0);
}

/// The mean weighs negative outcomes correctly, and a mean of zero is not
/// negative.
#[test]
fn test_mean_with_negative_outcomes()
{
	let cases = [
		(distribution(&[(-3, 1), (1, 1)]), "-2/2", -1.0f64),
		(distribution(&[(-3, 1), (3, 1)]), "0/2", 0.0),
		(distribution(&[(-1, 1), (0, 2)]), "-1/3", -1.0 / 3.0),
		(distribution(&[(i32::MIN, 1), (i32::MAX, 1)]), "-1/2", -0.5)
	];
	for (d, exact, approximate) in cases
	{
		assert_eq!(d.mean().to_string(), exact);
		assert_eq!(d.mean().to_f64().to_bits(), approximate.to_bits());
	}
	assert!(!distribution(&[(-3, 1), (3, 1)]).mean().is_negative());
}

/// A distribution whose total is beyond [`f64::MAX`] converts its
/// probabilities without NaN.
#[test]
fn test_to_f64_with_huge_total()
{
	let total = BigUint::from(6u8).pow(1000);
	let huge = from_big(&[(0, &total - 1u8), (1, BigUint::from(1u8))]).unwrap();
	assert_eq!(huge.total().to_f64(), f64::INFINITY);
	assert_eq!(huge.to_f64(), BTreeMap::from([(0, 1.0), (1, 0.0)]));
	assert_eq!(huge.probability(0).to_f64(), 1.0);
	assert_eq!(huge.mean().numerator(), &w(1));
	assert_eq!(huge.mean().to_f64(), 0.0);
	let even = from_big(&[(2, total.clone()), (4, total)]).unwrap();
	assert_eq!(even.to_f64(), BTreeMap::from([(2, 0.5), (4, 0.5)]));
	assert_eq!(even.mean(), r(false, 3, 1));
	assert_eq!(even.mean().to_f64(), 3.0);
}

/// Iteration visits the outcomes in ascending order, from either end, and
/// knows its length.
#[test]
fn test_iteration()
{
	let d = distribution(&[(5, 1), (-2, 3), (0, 2)]);
	let forward: Vec<_> = d.iter().collect();
	assert_eq!(forward, [(-2, &w(3)), (0, &w(2)), (5, &w(1))]);
	let backward: Vec<_> = d.iter().rev().collect();
	assert_eq!(backward, [(5, &w(1)), (0, &w(2)), (-2, &w(3))]);
	assert_eq!(d.iter().len(), 3);
	let borrowed: Vec<_> = (&d).into_iter().map(|(v, _)| v).collect();
	assert_eq!(borrowed, [-2, 0, 5]);
	let owned: Vec<_> = d.into_iter().collect();
	assert_eq!(owned, [(-2, w(3)), (0, w(2)), (5, w(1))]);
}

/// A distribution formats one line per outcome, in ascending order.
#[test]
fn test_display()
{
	let d = distribution(&[(2, 1), (-1, 3)]);
	assert_eq!(d.to_string(), "-1: 3\n2: 1\n");
}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// A distribution serializes as its weights and total, each weight as a
/// decimal string, and deserializes only when its invariants hold.
#[cfg(feature = "serde")]
#[test]
fn test_serde()
{
	let d = distribution(&[(2, 1), (-1, 3)]);
	let json = serde_json::to_string(&d).unwrap();
	assert_eq!(json, r#"{"weights":{"-1":"3","2":"1"},"total":"4"}"#);
	assert_eq!(serde_json::from_str::<Distribution>(&json).unwrap(), d);
	for (invalid, reason) in [
		(r#"{"weights":{},"total":"0"}"#, "no outcomes"),
		(
			r#"{"weights":{"1":"0","2":"1"},"total":"1"}"#,
			"outcome 1 as zero"
		),
		(r#"{"weights":{"1":"2"},"total":"3"}"#, "not the sum"),
		(r#"{"weights":{"1":"1"}}"#, "missing field"),
		(
			r#"{"weights":{"1":"1"},"total":"1","x":1}"#,
			"unknown field"
		),
		(r#"{"weights":{"1":1},"total":"1"}"#, "decimal string")
	]
	{
		let error = serde_json::from_str::<Distribution>(invalid).unwrap_err();
		assert!(error.to_string().contains(reason), "{invalid}: {error}");
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Property tests.                               //
////////////////////////////////////////////////////////////////////////////////

proptest! {
	/// Construction agrees with the reference: it sums repeated outcomes,
	/// omits zero weights, totals the rest, and refuses exactly when nothing
	/// remains.
	#[test]
	fn test_from_weights_agrees(weights in weights(), probe in outcome())
	{
		let expected = reference(&weights);
		let d = match from_big(&weights)
		{
			Ok(d) => d,
			Err(EmptyDistributionError) =>
			{
				prop_assert!(expected.is_empty());
				return Ok(())
			}
		};
		let actual: BTreeMap<_, _> =
			d.iter().map(|(v, n)| (v, n.to_biguint())).collect();
		prop_assert_eq!(&actual, &expected);
		prop_assert!(d.iter().all(|(_, n)| !n.is_zero()));
		prop_assert_eq!(d.total().to_biguint(), expected.values().sum::<BigUint>());
		prop_assert_eq!(d.len(), expected.len());
		prop_assert_eq!(d.iter().len(), expected.len());
		prop_assert_eq!(d.min(), *expected.keys().next().unwrap());
		prop_assert_eq!(d.max(), *expected.keys().next_back().unwrap());
		prop_assert_eq!(
			d.get(probe).to_biguint(),
			expected.get(&probe).cloned().unwrap_or_default()
		);
	}

	/// Probabilities and cumulative probabilities agree with the reference,
	/// over the unreduced total.
	#[test]
	fn test_probability_and_cdf_agree(
		weights in nonempty_weights(),
		probe in outcome()
	)
	{
		let expected = reference(&weights);
		let total: BigUint = expected.values().sum();
		let d = from_big(&weights).unwrap();
		let outcomes = expected.keys().copied().chain([probe]);
		for outcome in outcomes
		{
			let probability = d.probability(outcome);
			prop_assert_eq!(probability.denominator().to_biguint(), total.clone());
			prop_assert_eq!(
				probability.numerator().to_biguint(),
				expected.get(&outcome).cloned().unwrap_or_default()
			);
			let cdf = d.cdf(outcome);
			prop_assert_eq!(cdf.denominator().to_biguint(), total.clone());
			prop_assert_eq!(
				cdf.numerator().to_biguint(),
				expected.range(..=outcome).map(|(_, n)| n).sum::<BigUint>()
			);
		}
		prop_assert_eq!(d.cdf(d.max()), Probability::ONE);
		prop_assert_eq!(d.cdf(i32::MAX), Probability::ONE);
		if d.min() > i32::MIN
		{
			prop_assert_eq!(d.cdf(d.min() - 1), Probability::ZERO);
		}
	}

	/// The quantile of a probability is the least outcome whose cumulative
	/// probability reaches it, by the reference.
	#[test]
	fn test_quantile_agrees(
		weights in nonempty_weights(),
		(numerator, denominator) in probability()
	)
	{
		let expected = reference(&weights);
		let total: BigUint = expected.values().sum();
		let d = from_big(&weights).unwrap();
		let p = Probability::new(
			Weight::from_biguint(numerator.clone()),
			Weight::from_biguint(denominator.clone())
		)
		.unwrap();
		// cumulative / total ≥ numerator / denominator.
		let reaches =
			|cumulative: &BigUint| cumulative * &denominator >= &numerator * &total;
		let mut cumulative = BigUint::ZERO;
		let least = expected
			.iter()
			.find(|(_, n)| {
				cumulative += *n;
				reaches(&cumulative)
			})
			.map(|(v, _)| *v);
		prop_assert_eq!(Some(d.quantile(&p)), least);
	}

	/// Probabilities compare as the rational numbers that they denote.
	#[test]
	fn test_probability_ordering_agrees(
		(a, b) in probability(),
		(c, e) in probability()
	)
	{
		let x = Probability::new(
			Weight::from_biguint(a.clone()),
			Weight::from_biguint(b.clone())
		)
		.unwrap();
		let y = Probability::new(
			Weight::from_biguint(c.clone()),
			Weight::from_biguint(e.clone())
		)
		.unwrap();
		let expected = compare(&ratio(&a, &b), &ratio(&c, &e));
		prop_assert_eq!(x.cmp(&y), expected);
		prop_assert_eq!(x == y, expected == Ordering::Equal);
		prop_assert_eq!(x.partial_cmp(&y), Some(expected));
	}

	/// The quotient of any two weights, the second nonzero, is correctly
	/// rounded, whether it overflows, underflows, or is subnormal.
	#[test]
	fn test_ratio_to_f64_rounds_correctly(
		numerator in wide(),
		denominator in wide().prop_filter("nonzero", |d| *d != BigUint::ZERO)
	)
	{
		let answer = Weight::from_biguint(numerator.clone())
			.ratio_to_f64(&Weight::from_biguint(denominator.clone()));
		check_rounding(answer, &ratio(&numerator, &denominator))?;
	}

	/// Probabilities convert to correctly rounded floats.
	#[test]
	fn test_probability_to_f64_rounds_correctly(
		(numerator, denominator) in probability()
	)
	{
		let p = Probability::new(
			Weight::from_biguint(numerator.clone()),
			Weight::from_biguint(denominator.clone())
		)
		.unwrap();
		let answer = p.to_f64();
		prop_assert!((0.0..=1.0).contains(&answer));
		check_rounding(answer, &ratio(&numerator, &denominator))?;
	}

	/// Rational numbers compare as the rational numbers that they denote.
	#[test]
	fn test_rational_ordering_agrees(
		(a, b, c) in rational(),
		(d, e, f) in rational()
	)
	{
		let x = Rational::new(
			a,
			Weight::from_biguint(b.clone()),
			Weight::from_biguint(c.clone())
		)
		.unwrap();
		let y = Rational::new(
			d,
			Weight::from_biguint(e.clone()),
			Weight::from_biguint(f.clone())
		)
		.unwrap();
		let expected = compare(&signed(a, &b, &c), &signed(d, &e, &f));
		prop_assert_eq!(x.cmp(&y), expected);
		prop_assert_eq!(x == y, expected == Ordering::Equal);
		prop_assert_eq!(x.partial_cmp(&y), Some(expected));
		prop_assert_eq!(x.is_negative(), a && b != BigUint::ZERO);
	}

	/// Rational numbers convert to correctly rounded floats.
	#[test]
	fn test_rational_to_f64_rounds_correctly(
		(negative, numerator, denominator) in rational()
	)
	{
		let x = Rational::new(
			negative,
			Weight::from_biguint(numerator.clone()),
			Weight::from_biguint(denominator.clone())
		)
		.unwrap();
		check_rounding(x.to_f64(), &signed(negative, &numerator, &denominator))?;
	}

	/// The mean is the exact mean, over the unreduced total, and converts to
	/// the correctly rounded mean, and the floating-point view holds the
	/// correctly rounded probability of every outcome.
	#[test]
	fn test_mean_and_to_f64_round_correctly(weights in nonempty_weights())
	{
		let expected = reference(&weights);
		let total: BigUint = expected.values().sum();
		let d = from_big(&weights).unwrap();
		let sum: BigInt = expected
			.iter()
			.map(|(v, n)| BigInt::from(*v) * BigInt::from(n.clone()))
			.sum();
		let mean = d.mean();
		prop_assert_eq!(mean.is_negative(), sum.sign() == Sign::Minus);
		prop_assert_eq!(&mean.numerator().to_biguint(), sum.magnitude());
		prop_assert_eq!(mean.denominator().to_biguint(), total.clone());
		check_rounding(mean.to_f64(), &(sum, total.clone()))?;
		let view = d.to_f64();
		prop_assert!(view.keys().eq(expected.keys()));
		for (outcome, n) in &expected
		{
			prop_assert_eq!(view[outcome], d.probability(*outcome).to_f64());
			check_rounding(view[outcome], &ratio(n, &total))?;
		}
	}

	/// Formatting lists the reference weights, one line per outcome.
	#[test]
	fn test_display_agrees(weights in nonempty_weights())
	{
		let expected: String = reference(&weights)
			.iter()
			.map(|(v, n)| format!("{v}: {n}\n"))
			.collect();
		prop_assert_eq!(from_big(&weights).unwrap().to_string(), expected);
	}

	/// Serialization round trips.
	#[cfg(feature = "serde")]
	#[test]
	fn test_serde_round_trips(weights in nonempty_weights())
	{
		let d = from_big(&weights).unwrap();
		let json = serde_json::to_string(&d).unwrap();
		prop_assert_eq!(serde_json::from_str::<Distribution>(&json).unwrap(), d);
	}
}
