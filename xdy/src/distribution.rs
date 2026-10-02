//! # Exact distributions
//!
//! Functionality for describing the outcomes of dice expressions exactly. A
//! distribution weighs each possible outcome by an unbounded, nonnegative
//! integer [weight](Weight), so that the probability of an outcome is the
//! quotient of its weight and the total weight, with no rounding anywhere.
//!
//! The module also [plans](DistributionPlan) the computation of a distribution,
//! [estimating](Cost) its cost before building it within a [budget](Budget),
//! and [samples](Sampled) the outcomes of a function when its exact
//! distribution would cost too much.

mod probability;
pub(crate) mod propagation;
mod rational;
mod sampling;
mod weight;

use std::{
	collections::{BTreeMap, btree_map},
	error::Error,
	fmt::{self, Display, Formatter},
	iter::FusedIterator
};

pub use probability::*;
pub use propagation::{
	BuildError, DistributionPlan,
	cost::{Class, Cost},
	meter::{Budget, Dimension, Progress, Unobserved, Usage}
};
pub use rational::*;
pub use sampling::Sampled;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, de};
pub use weight::*;

////////////////////////////////////////////////////////////////////////////////
//                               Distributions.                               //
////////////////////////////////////////////////////////////////////////////////

/// An exact probability distribution over the outcomes of a dice expression, as
/// a map from outcomes to [weights](Weight), over their total.
///
/// Every distribution has at least one outcome, and every outcome has a nonzero
/// weight, so the outcomes are exactly the support of the distribution. The
/// probability of an outcome is its weight over the total of all weights.
///
/// The weights are never reduced by their greatest common divisor, so they keep
/// whatever meaning their producer gave them, like the number of paths through
/// an expression that reach each outcome. Equality therefore compares the
/// weights, not merely the probabilities: a coin weighed as `1` head to `1`
/// tail is not equal to one weighed as `2` heads to `2` tails, even though
/// every [probability](Probability) that they answer is equal.
///
/// # Examples
/// `(1D3)D3` rolls a three-sided die to decide how many three-sided dice to
/// roll. Each count has probability 1/3, and each roll of its dice has
/// probability 1/3, 1/9, or 1/27, so its distribution weighs each outcome over
/// a total of 81:
///
/// ```rust
/// use xdy::{Distribution, Probability, Weight};
///
/// let weights = [9u8, 12, 16, 12, 12, 10, 6, 3, 1];
/// let distribution = Distribution::from_weights(
///     (1..).zip(weights.map(Weight::from))
/// )?;
/// assert_eq!(distribution.total(), &Weight::from(81u8));
/// assert_eq!((distribution.min(), distribution.max()), (1, 9));
/// assert_eq!(distribution.probability(3).to_string(), "16/81");
/// assert_eq!(distribution.cdf(3).to_string(), "37/81");
/// assert_eq!(distribution.probability(0), Probability::ZERO);
/// assert_eq!(distribution.mean().to_string(), "324/81");
/// assert_eq!(distribution.mean().to_f64(), 4.0);
///
/// let half = Probability::new(Weight::ONE, Weight::from(2u8))?;
/// assert_eq!(distribution.quantile(&half), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Distribution
{
	/// The weight of each outcome, none of them zero.
	weights: BTreeMap<i32, Weight>,

	/// The sum of the weights, which is never zero.
	total: Weight
}

/// The weight of every outcome absent from a distribution, which
/// [`Distribution::get`] borrows.
static ZERO: Weight = Weight::ZERO;

////////////////////////////////////////////////////////////////////////////////
//                               Construction.                                //
////////////////////////////////////////////////////////////////////////////////

impl Distribution
{
	/// Construct the distribution that always answers the given outcome.
	///
	/// # Parameters
	/// - `outcome`: The outcome.
	///
	/// # Returns
	/// The distribution, which weighs `outcome` as one, over a total of one.
	pub fn point(outcome: i32) -> Self
	{
		Self {
			weights: BTreeMap::from([(outcome, Weight::ONE)]),
			total: Weight::ONE
		}
	}

	/// Construct a distribution from the weights of its outcomes. The weights
	/// of an outcome that occurs more than once are summed, and outcomes whose
	/// weights are zero are omitted.
	///
	/// # Parameters
	/// - `weights`: The outcomes and their weights, in any order.
	///
	/// # Returns
	/// The distribution.
	///
	/// # Errors
	/// [`EmptyDistributionError`] if no outcome has a nonzero weight.
	pub fn from_weights(
		weights: impl IntoIterator<Item = (i32, Weight)>
	) -> Result<Self, EmptyDistributionError>
	{
		let mut merged = BTreeMap::<i32, Weight>::new();
		for (outcome, weight) in weights
		{
			if !weight.is_zero()
			{
				*merged.entry(outcome).or_default() += weight;
			}
		}
		if merged.is_empty()
		{
			return Err(EmptyDistributionError)
		}
		let total = merged.values().sum();
		Ok(Self {
			weights: merged,
			total
		})
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Exact queries.                               //
////////////////////////////////////////////////////////////////////////////////

impl Distribution
{
	/// Answer the weight of the given outcome.
	///
	/// # Parameters
	/// - `outcome`: The outcome.
	///
	/// # Returns
	/// The weight of `outcome`, which is zero if the outcome is impossible.
	pub fn get(&self, outcome: i32) -> &Weight
	{
		self.weights.get(&outcome).unwrap_or(&ZERO)
	}

	/// Answer the total weight of the distribution.
	///
	/// # Returns
	/// The sum of the weights of the outcomes, which is never zero.
	#[inline]
	pub fn total(&self) -> &Weight { &self.total }

	/// Answer the number of possible outcomes.
	///
	/// # Returns
	/// The number of outcomes with nonzero weight, which is never zero.
	// A distribution is never empty, so `is_empty` would always answer
	// `false`.
	#[allow(clippy::len_without_is_empty)]
	#[inline]
	pub fn len(&self) -> usize { self.weights.len() }

	/// Answer the least possible outcome.
	///
	/// # Returns
	/// The least outcome with nonzero weight.
	pub fn min(&self) -> i32
	{
		*self
			.weights
			.keys()
			.next()
			.expect("distribution is never empty")
	}

	/// Answer the greatest possible outcome.
	///
	/// # Returns
	/// The greatest outcome with nonzero weight.
	pub fn max(&self) -> i32
	{
		*self
			.weights
			.keys()
			.next_back()
			.expect("distribution is never empty")
	}

	/// Answer the probability of the given outcome.
	///
	/// # Parameters
	/// - `outcome`: The outcome.
	///
	/// # Returns
	/// The weight of `outcome` over the [total](Self::total).
	pub fn probability(&self, outcome: i32) -> Probability
	{
		Probability::new_unchecked(
			self.get(outcome).clone(),
			self.total.clone()
		)
	}

	/// Answer the probability that an outcome does not exceed the given one,
	/// which is the value of the cumulative distribution function there.
	///
	/// # Parameters
	/// - `outcome`: The outcome.
	///
	/// # Returns
	/// The sum of the weights of the outcomes up to and including `outcome`,
	/// over the [total](Self::total).
	pub fn cdf(&self, outcome: i32) -> Probability
	{
		let cumulative = self.weights.range(..=outcome).map(|(_, w)| w).sum();
		Probability::new_unchecked(cumulative, self.total.clone())
	}

	/// Answer the least possible outcome at which the
	/// [cumulative distribution function](Self::cdf) reaches the given
	/// probability.
	///
	/// # Parameters
	/// - `p`: The probability.
	///
	/// # Returns
	/// The least outcome `v` with nonzero weight for which `cdf(v)` is at
	/// least `p`. This is the [least outcome](Self::min) when `p` is zero,
	/// and the [greatest outcome](Self::max) when `p` is one.
	///
	/// # Examples
	/// ```rust
	/// use xdy::{Distribution, Probability, Weight};
	///
	/// let d6 = Distribution::from_weights((1..=6).map(|v| (v, Weight::ONE)))?;
	/// let third = Probability::new(Weight::ONE, Weight::from(3u8))?;
	/// assert_eq!(d6.quantile(&third), 2);
	/// assert_eq!(d6.quantile(&Probability::ZERO), 1);
	/// assert_eq!(d6.quantile(&Probability::ONE), 6);
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn quantile(&self, p: &Probability) -> i32
	{
		// cumulative / total ≥ numerator / denominator, by
		// cross-multiplication.
		let target = p.numerator() * &self.total;
		let mut cumulative = Weight::ZERO;
		for (&outcome, weight) in &self.weights
		{
			cumulative += weight;
			if &cumulative * p.denominator() >= target
			{
				return outcome
			}
		}
		unreachable!("the total weight reaches every probability")
	}

	/// Answer the mean outcome.
	///
	/// # Returns
	/// The sum of the outcomes, each multiplied by its weight, over the
	/// [total](Self::total), unreduced. [`Rational::to_f64`] rounds it once to
	/// the nearest [`f64`].
	///
	/// # Examples
	/// ```rust
	/// use xdy::{Distribution, Weight};
	///
	/// let d4 = Distribution::from_weights((-2..=1).map(|v| (v, Weight::ONE)))?;
	/// assert_eq!(d4.mean().to_string(), "-2/4");
	/// assert_eq!(d4.mean().to_f64(), -0.5);
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn mean(&self) -> Rational
	{
		// Sum the positive and negative contributions separately, since
		// weights are never negative.
		let mut positive = Weight::ZERO;
		let mut negative = Weight::ZERO;
		for (&outcome, weight) in &self.weights
		{
			let magnitude = Weight::from(outcome.unsigned_abs());
			if outcome > 0
			{
				positive += magnitude * weight;
			}
			else
			{
				negative += magnitude * weight;
			}
		}
		match positive.checked_sub(&negative)
		{
			Some(difference) =>
			{
				Rational::new_unchecked(false, difference, self.total.clone())
			},
			None =>
			{
				let difference = negative
					.checked_sub(&positive)
					.expect("negative exceeds positive");
				Rational::new_unchecked(true, difference, self.total.clone())
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Floating-point views.                            //
////////////////////////////////////////////////////////////////////////////////

impl Distribution
{
	/// Answer the probability of every possible outcome, as the nearest
	/// [`f64`], for charts and other consumers that do not need exactness.
	///
	/// # Returns
	/// The correctly rounded probability of each outcome with nonzero weight.
	/// An outcome whose probability is smaller than half the least subnormal
	/// `f64` has probability `0`, and the probabilities need not sum to
	/// exactly `1`.
	pub fn to_f64(&self) -> BTreeMap<i32, f64>
	{
		self.weights
			.iter()
			.map(|(&outcome, weight)| {
				(outcome, weight.ratio_to_f64(&self.total))
			})
			.collect()
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Iteration.                                 //
////////////////////////////////////////////////////////////////////////////////

impl Distribution
{
	/// Answer an iterator over the possible outcomes and their weights.
	///
	/// # Returns
	/// An iterator over the outcomes with nonzero weight, in ascending order,
	/// and their weights.
	pub fn iter(&self) -> DistributionIter<'_>
	{
		DistributionIter(self.weights.iter())
	}
}

/// An iterator over the possible outcomes of a [`Distribution`] and their
/// weights, in ascending order of outcome.
#[derive(Debug, Clone)]
pub struct DistributionIter<'a>(btree_map::Iter<'a, i32, Weight>);

impl<'a> Iterator for DistributionIter<'a>
{
	type Item = (i32, &'a Weight);

	fn next(&mut self) -> Option<Self::Item>
	{
		self.0.next().map(|(&outcome, weight)| (outcome, weight))
	}

	fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl DoubleEndedIterator for DistributionIter<'_>
{
	fn next_back(&mut self) -> Option<Self::Item>
	{
		self.0
			.next_back()
			.map(|(&outcome, weight)| (outcome, weight))
	}
}

impl ExactSizeIterator for DistributionIter<'_> {}

impl FusedIterator for DistributionIter<'_> {}

impl<'a> IntoIterator for &'a Distribution
{
	type Item = (i32, &'a Weight);
	type IntoIter = DistributionIter<'a>;

	fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl IntoIterator for Distribution
{
	type Item = (i32, Weight);
	type IntoIter = btree_map::IntoIter<i32, Weight>;

	fn into_iter(self) -> Self::IntoIter { self.weights.into_iter() }
}

////////////////////////////////////////////////////////////////////////////////
//                                Formatting.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Formats one line per possible outcome, `outcome: weight`, in ascending order
/// of outcome.
impl Display for Distribution
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		for (outcome, weight) in self
		{
			writeln!(f, "{}: {}", outcome, weight)?;
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// A [`Distribution`] as it deserializes, before its invariants are checked.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedDistribution
{
	/// The weight of each outcome.
	weights: BTreeMap<i32, Weight>,

	/// The purported sum of the weights.
	total: Weight
}

/// Deserializes a distribution from its weights and total, refusing one that
/// is empty, weighs an outcome as zero, or whose total is not the sum of its
/// weights.
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Distribution
{
	fn deserialize<D: Deserializer<'de>>(
		deserializer: D
	) -> Result<Self, D::Error>
	{
		use de::Error as _;
		let UncheckedDistribution { weights, total } =
			UncheckedDistribution::deserialize(deserializer)?;
		if weights.is_empty()
		{
			return Err(D::Error::custom("distribution has no outcomes"))
		}
		if let Some((outcome, _)) = weights.iter().find(|(_, w)| w.is_zero())
		{
			return Err(D::Error::custom(format_args!(
				"distribution weighs outcome {outcome} as zero"
			)))
		}
		let sum: Weight = weights.values().sum();
		if sum != total
		{
			return Err(D::Error::custom(format_args!(
				"distribution total {total} is not the sum of its weights, \
				 {sum}"
			)))
		}
		Ok(Self { weights, total })
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The reason that weights do not make a [`Distribution`]: no outcome has a
/// nonzero weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyDistributionError;

impl Display for EmptyDistributionError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "distribution has no outcome with nonzero weight")
	}
}

impl Error for EmptyDistributionError {}
