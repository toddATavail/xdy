//! # Probabilities
//!
//! The exact probabilities that an exact [distribution](crate::Distribution)
//! answers, each the quotient of two [weights](Weight).

use std::{
	cmp::Ordering,
	error::Error,
	fmt::{self, Display, Formatter}
};

use crate::Weight;

////////////////////////////////////////////////////////////////////////////////
//                               Probabilities.                               //
////////////////////////////////////////////////////////////////////////////////

/// An exact probability, as the quotient of a numerator and a nonzero
/// denominator that is not less than it.
///
/// A distribution answers its probabilities over its total weight, which is
/// never reduced, so the numerator and denominator are whatever the
/// distribution holds: the chance of a head on a coin weighed as `2` heads to
/// `2` tails is `2/4`, not `1/2`. Equality and ordering are nevertheless those
/// of the rational numbers, so `2/4` equals `1/2`.
///
/// # Notes
/// A probability does not implement [`Hash`], since equal probabilities may
/// have different numerators and denominators, and reducing them to hash
/// them would cost a greatest common divisor.
///
/// # Examples
/// ```rust
/// use xdy::{Probability, Weight};
///
/// let half = Probability::new(Weight::from(2u8), Weight::from(4u8))?;
/// assert_eq!(half.to_string(), "2/4");
/// assert_eq!(half, Probability::new(Weight::ONE, Weight::from(2u8))?);
/// assert!(Probability::ZERO < half && half < Probability::ONE);
/// assert_eq!(half.to_f64(), 0.5);
/// # Ok::<(), xdy::ProbabilityError>(())
/// ```
///
/// Converting to an [`f64`] rounds only once, so it is safe even when the
/// numerator and denominator are both beyond [`f64::MAX`]:
///
/// ```rust
/// use xdy::{Probability, Weight};
///
/// let total = Weight::from(6u8).pow(1000);
/// let mostly = total.checked_sub(&Weight::ONE).unwrap();
/// assert_eq!(total.to_f64(), f64::INFINITY);
/// assert_eq!(Probability::new(mostly, total.clone())?.to_f64(), 1.0);
/// assert_eq!(Probability::new(Weight::ONE, total)?.to_f64(), 0.0);
/// # Ok::<(), xdy::ProbabilityError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Probability
{
	/// The numerator, which does not exceed the denominator.
	numerator: Weight,

	/// The denominator, which is not zero.
	denominator: Weight
}

impl Probability
{
	/// The probability zero, as `0/1`.
	pub const ZERO: Self = Self {
		numerator: Weight::ZERO,
		denominator: Weight::ONE
	};

	/// The probability one, as `1/1`.
	pub const ONE: Self = Self {
		numerator: Weight::ONE,
		denominator: Weight::ONE
	};

	/// Construct a probability from its numerator and denominator.
	///
	/// # Parameters
	/// - `numerator`: The numerator.
	/// - `denominator`: The denominator.
	///
	/// # Returns
	/// The probability `numerator/denominator`, unreduced.
	///
	/// # Errors
	/// - [`ZeroDenominator`](ProbabilityError::ZeroDenominator) if
	///   `denominator` is zero.
	/// - [`ExceedsOne`](ProbabilityError::ExceedsOne) if `numerator` exceeds
	///   `denominator`.
	pub fn new(
		numerator: Weight,
		denominator: Weight
	) -> Result<Self, ProbabilityError>
	{
		if denominator.is_zero()
		{
			return Err(ProbabilityError::ZeroDenominator)
		}
		if numerator > denominator
		{
			return Err(ProbabilityError::ExceedsOne)
		}
		Ok(Self::new_unchecked(numerator, denominator))
	}

	/// Construct a probability from a numerator and denominator already known
	/// to be valid.
	///
	/// # Parameters
	/// - `numerator`: The numerator, which must not exceed `denominator`.
	/// - `denominator`: The denominator, which must not be zero.
	///
	/// # Returns
	/// The probability `numerator/denominator`, unreduced.
	pub(crate) fn new_unchecked(numerator: Weight, denominator: Weight)
	-> Self
	{
		debug_assert!(!denominator.is_zero() && numerator <= denominator);
		Self {
			numerator,
			denominator
		}
	}

	/// Answer the numerator of the probability, as constructed.
	///
	/// # Returns
	/// The numerator.
	#[inline]
	pub fn numerator(&self) -> &Weight { &self.numerator }

	/// Answer the denominator of the probability, as constructed.
	///
	/// # Returns
	/// The denominator, which is never zero.
	#[inline]
	pub fn denominator(&self) -> &Weight { &self.denominator }

	/// Answer the probability as the nearest [`f64`].
	///
	/// # Returns
	/// The correctly rounded probability, between `0` and `1` inclusive. A
	/// probability smaller than half the least subnormal `f64` answers `0`.
	pub fn to_f64(&self) -> f64
	{
		self.numerator.ratio_to_f64(&self.denominator)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Comparison.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Compares the rational values, by cross-multiplication.
impl PartialEq for Probability
{
	fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal }
}

impl Eq for Probability {}

/// Compares the rational values, by cross-multiplication.
impl PartialOrd for Probability
{
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>
	{
		Some(self.cmp(other))
	}
}

/// Compares the rational values, by cross-multiplication.
impl Ord for Probability
{
	fn cmp(&self, other: &Self) -> Ordering
	{
		if self.denominator == other.denominator
		{
			return self.numerator.cmp(&other.numerator)
		}
		(&self.numerator * &other.denominator)
			.cmp(&(&other.numerator * &self.denominator))
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Formatting.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Formats the probability as `numerator/denominator`, unreduced.
impl Display for Probability
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}/{}", self.numerator, self.denominator)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The reason that a numerator and denominator do not make a
/// [`Probability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityError
{
	/// The denominator is zero.
	ZeroDenominator,

	/// The numerator exceeds the denominator, so the quotient exceeds one.
	ExceedsOne
}

impl Display for ProbabilityError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			Self::ZeroDenominator =>
			{
				write!(f, "probability has a zero denominator")
			},
			Self::ExceedsOne => write!(f, "probability exceeds one")
		}
	}
}

impl Error for ProbabilityError {}
