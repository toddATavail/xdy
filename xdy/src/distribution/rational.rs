//! # Rational numbers
//!
//! The exact signed rational numbers that an exact
//! [distribution](crate::Distribution) answers when its answer may be negative
//! or exceed one, like its [mean](crate::Distribution::mean), each a sign and
//! the quotient of two [weights](Weight).

use std::{
	cmp::Ordering,
	error::Error,
	fmt::{self, Display, Formatter}
};

use crate::Weight;

////////////////////////////////////////////////////////////////////////////////
//                                 Rationals.                                 //
////////////////////////////////////////////////////////////////////////////////

/// An exact rational number, as a sign and the quotient of a numerator and a
/// nonzero denominator.
///
/// Like a [`Probability`](crate::Probability), a rational number is never
/// reduced, so the numerator and denominator are whatever its producer gave
/// it: the mean of a die weighed as one of each face from `1` to `6` is `21/6`,
/// not `7/2`. Equality and ordering are nevertheless those of the rational
/// numbers, so `21/6` equals `7/2`. Zero is never negative, so there is
/// exactly one sign for every value.
///
/// # Notes
/// A rational number does not implement [`Hash`], since equal rational numbers
/// may have different numerators and denominators, and reducing them to hash
/// them would cost a greatest common divisor.
///
/// # Examples
/// ```rust
/// use xdy::{Rational, Weight};
///
/// let loss = Rational::new(true, Weight::from(21u8), Weight::from(6u8))?;
/// assert!(loss.is_negative());
/// assert_eq!(loss.to_string(), "-21/6");
/// assert_eq!(loss, Rational::new(true, Weight::from(7u8), Weight::from(2u8))?);
/// assert!(loss < Rational::ZERO && Rational::ZERO < Rational::ONE);
/// assert_eq!(loss.to_f64(), -3.5);
/// # Ok::<(), xdy::ZeroDenominatorError>(())
/// ```
///
/// A negative zero is zero:
///
/// ```rust
/// use xdy::{Rational, Weight};
///
/// let zero = Rational::new(true, Weight::ZERO, Weight::from(5u8))?;
/// assert!(!zero.is_negative());
/// assert_eq!(zero.to_string(), "0/5");
/// assert_eq!(zero, Rational::ZERO);
/// # Ok::<(), xdy::ZeroDenominatorError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Rational
{
	/// Whether the number is negative, which it never is when the numerator
	/// is zero.
	negative: bool,

	/// The numerator, which is the magnitude of the number's numerator.
	numerator: Weight,

	/// The denominator, which is not zero.
	denominator: Weight
}

impl Rational
{
	/// The rational number zero, as `0/1`.
	pub const ZERO: Self = Self {
		negative: false,
		numerator: Weight::ZERO,
		denominator: Weight::ONE
	};

	/// The rational number one, as `1/1`.
	pub const ONE: Self = Self {
		negative: false,
		numerator: Weight::ONE,
		denominator: Weight::ONE
	};

	/// Construct a rational number from its sign, numerator, and denominator.
	///
	/// # Parameters
	/// - `negative`: Whether the number is negative. This is ignored when
	///   `numerator` is zero, since zero is never negative.
	/// - `numerator`: The magnitude of the numerator.
	/// - `denominator`: The denominator.
	///
	/// # Returns
	/// The rational number `numerator/denominator`, negated if `negative`,
	/// unreduced.
	///
	/// # Errors
	/// [`ZeroDenominatorError`] if `denominator` is zero.
	pub fn new(
		negative: bool,
		numerator: Weight,
		denominator: Weight
	) -> Result<Self, ZeroDenominatorError>
	{
		if denominator.is_zero()
		{
			return Err(ZeroDenominatorError)
		}
		let negative = negative && !numerator.is_zero();
		Ok(Self::new_unchecked(negative, numerator, denominator))
	}

	/// Construct a rational number from a sign, numerator, and denominator
	/// already known to be valid.
	///
	/// # Parameters
	/// - `negative`: Whether the number is negative, which must be `false` when
	///   `numerator` is zero.
	/// - `numerator`: The magnitude of the numerator.
	/// - `denominator`: The denominator, which must not be zero.
	///
	/// # Returns
	/// The rational number `numerator/denominator`, negated if `negative`,
	/// unreduced.
	pub(crate) fn new_unchecked(
		negative: bool,
		numerator: Weight,
		denominator: Weight
	) -> Self
	{
		debug_assert!(!denominator.is_zero());
		debug_assert!(!(negative && numerator.is_zero()));
		Self {
			negative,
			numerator,
			denominator
		}
	}

	/// Answer whether the rational number is negative.
	///
	/// # Returns
	/// `true` if the number is less than zero, `false` otherwise.
	#[inline]
	pub fn is_negative(&self) -> bool { self.negative }

	/// Answer the magnitude of the numerator of the rational number, as
	/// constructed.
	///
	/// # Returns
	/// The numerator, without its sign.
	#[inline]
	pub fn numerator(&self) -> &Weight { &self.numerator }

	/// Answer the denominator of the rational number, as constructed.
	///
	/// # Returns
	/// The denominator, which is never zero.
	#[inline]
	pub fn denominator(&self) -> &Weight { &self.denominator }

	/// Answer the rational number as the nearest [`f64`].
	///
	/// # Returns
	/// The correctly rounded number. A number smaller in magnitude than half
	/// the least subnormal `f64` answers `0`, or `-0` if it is negative, and a
	/// number too large in magnitude for an `f64` answers an infinity of its
	/// sign.
	pub fn to_f64(&self) -> f64
	{
		let magnitude = self.numerator.ratio_to_f64(&self.denominator);
		match self.negative
		{
			true => -magnitude,
			false => magnitude
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Comparison.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Compares the rational values, by sign and then cross-multiplication.
impl PartialEq for Rational
{
	fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal }
}

impl Eq for Rational {}

/// Compares the rational values, by sign and then cross-multiplication.
impl PartialOrd for Rational
{
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>
	{
		Some(self.cmp(other))
	}
}

/// Compares the rational values, by sign and then cross-multiplication.
impl Ord for Rational
{
	fn cmp(&self, other: &Self) -> Ordering
	{
		// Zero is never negative, so every negative number is less than
		// every number that is not.
		match (self.negative, other.negative)
		{
			(false, true) => Ordering::Greater,
			(true, false) => Ordering::Less,
			(false, false) => self.cmp_magnitude(other),
			(true, true) => other.cmp_magnitude(self)
		}
	}
}

impl Rational
{
	/// Compare the magnitudes of two rational numbers.
	///
	/// # Parameters
	/// - `other`: The other number.
	///
	/// # Returns
	/// The ordering of the magnitude of `self` and that of `other`.
	fn cmp_magnitude(&self, other: &Self) -> Ordering
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

/// Formats the rational number as `numerator/denominator`, unreduced, preceded
/// by `-` if it is negative.
impl Display for Rational
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		let sign = if self.negative { "-" } else { "" };
		write!(f, "{sign}{}/{}", self.numerator, self.denominator)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The reason that a numerator and denominator do not make a [`Rational`]:
/// the denominator is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroDenominatorError;

impl Display for ZeroDenominatorError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "rational has a zero denominator")
	}
}

impl Error for ZeroDenominatorError {}
