//! # Weights
//!
//! The unbounded nonnegative integers that weigh the outcomes of an exact
//! [distribution](crate::distribution). Most weights fit comfortably in a
//! [`u128`], but some do not: the total of `100D6` is 6¹⁰⁰, about 259 bits. A
//! weight therefore uses machine arithmetic while it can, and promotes to an
//! arbitrary-precision integer only when that arithmetic would overflow.

use std::{
	error::Error,
	fmt::{self, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign},
	str::FromStr
};

use num_bigint::BigUint;
use num_traits::{CheckedSub, ToPrimitive};
#[cfg(feature = "serde")]
use serde::{
	Deserialize, Deserializer, Serialize, Serializer,
	de::{self, Visitor}
};

////////////////////////////////////////////////////////////////////////////////
//                                  Weights.                                  //
////////////////////////////////////////////////////////////////////////////////

/// An unbounded nonnegative integer, which weighs an outcome of an exact
/// distribution.
///
/// A weight is held as a [`u128`] until arithmetic would overflow it, and only
/// then promotes to an arbitrary-precision integer, so the weights of
/// dice-sized expressions never allocate. Every weight has exactly one
/// representation: a weight that fits in a `u128` is always held as one, so
/// arithmetic that brings a large weight back within range, like subtraction
/// or multiplication by zero, demotes it again. Equality, ordering, and hashing
/// are therefore those of the integers themselves.
///
/// # Notes
/// The arbitrary-precision representation is private, so that the choice of
/// big-integer crate stays an implementation detail. With the `serde` feature,
/// a weight serializes as a decimal string, since many consumers of JSON read
/// every number as an [`f64`], which is exact only up to 2⁵³.
///
/// # Examples
/// Arithmetic promotes past [`u128::MAX`], and demotes back within it:
///
/// ```rust
/// use xdy::Weight;
///
/// let max = Weight::from(u128::MAX);
/// let promoted = &max + &Weight::ONE;
/// assert_eq!(
///     promoted.to_string(),
///     "340282366920938463463374607431768211456"
/// );
/// assert_eq!(promoted.bits(), 129);
/// assert_eq!(promoted.checked_sub(&Weight::ONE), Some(max));
/// ```
///
/// The total of `100D6` is 6¹⁰⁰, which converts to the nearest [`f64`]:
///
/// ```rust
/// use xdy::Weight;
///
/// let total = Weight::from(6u8).pow(100);
/// assert_eq!(total.bits(), 259);
/// assert_eq!(total.to_f64(), 6.533186235000709e77);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Weight(Repr);

/// The representation of a [`Weight`].
///
/// # Notes
/// The derived orderings compare variants before values, and
/// [`Small`](Repr::Small) precedes [`Big`](Repr::Big). That is correct only
/// because every weight has exactly one representation, so every `Big`
/// exceeds every `Small`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Repr
{
	/// A weight that fits in a [`u128`].
	Small(u128),

	/// A weight that exceeds [`u128::MAX`].
	Big(BigUint)
}

impl Default for Repr
{
	fn default() -> Self { Repr::Small(0) }
}

////////////////////////////////////////////////////////////////////////////////
//                        Construction and conversion.                        //
////////////////////////////////////////////////////////////////////////////////

impl Weight
{
	/// The weight zero.
	pub const ZERO: Self = Self(Repr::Small(0));

	/// The weight one.
	pub const ONE: Self = Self(Repr::Small(1));

	/// Answer the weight with the value of the given big integer, demoting it
	/// if it fits in a [`u128`].
	///
	/// # Parameters
	/// - `big`: The value.
	///
	/// # Returns
	/// The weight.
	fn from_big(big: BigUint) -> Self
	{
		match u128::try_from(&big)
		{
			Ok(small) => Self(Repr::Small(small)),
			Err(_) => Self(Repr::Big(big))
		}
	}

	/// Answer the value of the weight as a big integer.
	///
	/// # Returns
	/// The value.
	pub(crate) fn to_biguint(&self) -> BigUint
	{
		match &self.0
		{
			Repr::Small(n) => BigUint::from(*n),
			Repr::Big(n) => n.clone()
		}
	}

	/// Write the binary digits of the weight, 32 at a time, least significant
	/// first, into the given digits, which must hold them all, and leave the
	/// rest alone.
	///
	/// # Parameters
	/// - `digits`: The digits, which must number at least the weight's
	///   [bits](Self::bits) over 32.
	///
	/// # Notes
	/// `convolve_packed`, in `propagation/record.rs`, packs whole
	/// distributions into one integer this way.
	pub(crate) fn write_u32_digits(&self, digits: &mut [u32])
	{
		debug_assert!(
			digits.len() as u64 * 32 >= self.bits(),
			"{} bits exceed {} digits",
			self.bits(),
			digits.len()
		);
		match &self.0
		{
			Repr::Small(n) =>
			{
				for (i, digit) in digits.iter_mut().take(4).enumerate()
				{
					*digit = (n >> (32 * i)) as u32;
				}
			},
			Repr::Big(n) =>
			{
				for (digit, d) in digits.iter_mut().zip(n.iter_u32_digits())
				{
					*digit = d;
				}
			}
		}
	}

	/// Answer the binary digits of the weight, 32 at a time, least significant
	/// first, without leading zeros.
	///
	/// # Returns
	/// The digits, which are empty just when the weight is zero.
	pub(crate) fn to_u32_digits(&self) -> Vec<u32>
	{
		match &self.0
		{
			Repr::Small(n) =>
			{
				let len = (self.bits() as usize).div_ceil(32);
				(0..len).map(|i| (n >> (32 * i)) as u32).collect()
			},
			Repr::Big(n) => n.to_u32_digits()
		}
	}

	/// Answer the weight whose binary digits, 32 at a time, least significant
	/// first, are given.
	///
	/// # Parameters
	/// - `digits`: The digits, which may have leading zeros.
	///
	/// # Returns
	/// The weight, held as a [`u128`] if it fits in one.
	pub(crate) fn from_u32_digits(digits: &[u32]) -> Self
	{
		let len = digits.iter().rposition(|d| *d != 0).map_or(0, |i| i + 1);
		if len <= 4
		{
			let n = digits[..len]
				.iter()
				.rev()
				.fold(0u128, |n, d| (n << 32) | *d as u128);
			Self(Repr::Small(n))
		}
		else
		{
			Self(Repr::Big(BigUint::from_slice(&digits[..len])))
		}
	}

	/// Answer whether the weight is zero.
	///
	/// # Returns
	/// `true` if the weight is zero, `false` otherwise.
	#[inline]
	pub fn is_zero(&self) -> bool { matches!(self.0, Repr::Small(0)) }

	/// Answer the number of significant bits in the weight, which is the
	/// ceiling of its binary logarithm, less one for exact powers of two.
	///
	/// # Returns
	/// The number of bits needed to represent the weight, or zero if the
	/// weight is zero.
	pub fn bits(&self) -> u64
	{
		match &self.0
		{
			Repr::Small(n) => (u128::BITS - n.leading_zeros()) as u64,
			Repr::Big(n) => n.bits()
		}
	}

	/// Answer the weight as the nearest [`f64`]. This is lossy for weights
	/// above 2⁵³, and saturates to infinity for weights beyond [`f64::MAX`].
	///
	/// # Returns
	/// The nearest `f64`, or [`f64::INFINITY`] if the weight is too large.
	pub fn to_f64(&self) -> f64
	{
		match &self.0
		{
			Repr::Small(n) => *n as f64,
			Repr::Big(n) => n.to_f64().unwrap_or(f64::INFINITY)
		}
	}

	/// Answer the quotient of the weight and the given nonzero weight, as the
	/// nearest [`f64`], rounding ties to even.
	///
	/// # Parameters
	/// - `denominator`: The divisor.
	///
	/// # Returns
	/// The correctly rounded quotient. It is `0` if the quotient is smaller
	/// than half the least subnormal `f64`, and [`f64::INFINITY`] if it is too
	/// large for an `f64`, but never NaN, however large the operands: dividing
	/// their separate conversions would answer NaN for any two weights beyond
	/// [`f64::MAX`].
	///
	/// # Panics
	/// If `denominator` is zero.
	///
	/// # Notes
	/// Operands that are exact as `f64`s divide in floating point, which is
	/// correctly rounded by itself. Otherwise the quotient is taken as an
	/// integer, scaled so that it has at least two bits beyond the precision
	/// of its result, with a sticky bit for the remainder; rounding that
	/// integer once to the precision of the result, which is fewer than 53
	/// bits for a subnormal result, is then exact.
	pub(crate) fn ratio_to_f64(&self, denominator: &Weight) -> f64
	{
		assert!(!denominator.is_zero(), "division by zero weight");
		if self.is_zero()
		{
			return 0.0
		}
		// Integers up to 2⁵³ are exact as f64s, and IEEE 754 division of
		// exact operands is correctly rounded.
		if self.bits() <= f64::MANTISSA_DIGITS as u64
			&& denominator.bits() <= f64::MANTISSA_DIGITS as u64
		{
			return self.to_f64() / denominator.to_f64()
		}
		// The quotient q lies strictly between 2^(e - 1) and 2^(e + 1).
		let e = self.bits() as i64 - denominator.bits() as i64;
		if e >= 1025
		{
			// q exceeds 2¹⁰²⁴, beyond every finite f64.
			return f64::INFINITY
		}
		if e <= -1076
		{
			// q is less than 2⁻¹⁰⁷⁵, half the least subnormal f64.
			return 0.0
		}
		// Scale q by 2^s, so that the integer part has at least 56 bits, but
		// has no unit finer than 2⁻¹⁰⁷⁶, which is two bits below the least
		// subnormal f64. Either way, it is less than 2⁵⁷.
		let s = (56 - e).min(1076);
		let numerator = self.to_biguint();
		let denominator = denominator.to_biguint();
		let (numerator, denominator) = if s >= 0
		{
			(numerator << s as u64, denominator)
		}
		else
		{
			(numerator, denominator << -s as u64)
		};
		let quotient = &numerator / &denominator;
		let remainder = numerator % denominator;
		let scaled = quotient.to_u64().expect("scaled quotient fits in u64")
			| u64::from(remainder != BigUint::ZERO);
		// The exponent of the leading bit of q, and the exponent of the unit
		// in the last place of the result: 52 places below the leading bit, but
		// no finer than the least subnormal f64.
		let leading = (u64::BITS - scaled.leading_zeros()) as i64 - 1 - s;
		let unit = (leading - 52).max(-1074);
		// Round to the unit, ties to even. The unit is at least two places
		// above the sticky bit, so the half bit is exact and the sticky bit
		// breaks every false tie.
		let drop = (unit + s) as u32;
		let half = 1u64 << (drop - 1);
		let mut rounded = scaled >> drop;
		let rest = scaled & ((1u64 << drop) - 1);
		if rest > half || (rest == half && rounded & 1 == 1)
		{
			rounded += 1;
		}
		if unit + (u64::BITS - rounded.leading_zeros()) as i64 > 1024
		{
			// The result rounds past f64::MAX.
			return f64::INFINITY
		}
		// The rounded quotient has at most 53 bits, so the conversion is exact,
		// and so is the scaling, since the product is representable.
		rounded as f64 * pow2(unit)
	}
}

/// Answer 2 raised to the given power, which must be the exponent of a finite
/// [`f64`], normal or subnormal.
///
/// # Parameters
/// - `exponent`: The exponent, between -1074 and 1023.
///
/// # Returns
/// The power, exactly.
fn pow2(exponent: i64) -> f64
{
	debug_assert!((-1074..=1023).contains(&exponent));
	if exponent >= -1022
	{
		// A normal f64: its biased exponent, and no fraction.
		f64::from_bits(((exponent + 1023) as u64) << 52)
	}
	else
	{
		// A subnormal f64: a single bit of fraction.
		f64::from_bits(1 << (exponent + 1074))
	}
}

/// Implement [`From`] for [`Weight`] over the given unsigned integer types.
macro_rules! weight_from_unsigned {
	($($t:ty),*) => {
		$(
			impl From<$t> for Weight
			{
				fn from(n: $t) -> Self { Self(Repr::Small(n as u128)) }
			}
		)*
	};
}

weight_from_unsigned!(u8, u16, u32, u64, u128, usize);

#[cfg(test)]
impl Weight
{
	/// Answer the weight with the value of the given big integer, as the
	/// tests construct it.
	///
	/// # Parameters
	/// - `big`: The value.
	///
	/// # Returns
	/// The weight.
	pub(crate) fn from_biguint(big: BigUint) -> Self { Self::from_big(big) }

	/// Answer whether the weight is held as a [`u128`], which the tests
	/// check against its value to hold every weight to its one
	/// representation.
	///
	/// # Returns
	/// `true` if the weight is held as a `u128`, `false` otherwise.
	pub(crate) fn is_small(&self) -> bool { matches!(self.0, Repr::Small(_)) }
}

////////////////////////////////////////////////////////////////////////////////
//                                Arithmetic.                                 //
////////////////////////////////////////////////////////////////////////////////

impl Weight
{
	/// Raise the weight to the given power.
	///
	/// # Parameters
	/// - `exponent`: The exponent.
	///
	/// # Returns
	/// The weight raised to `exponent`. Any weight raised to zero is one.
	pub fn pow(&self, exponent: u32) -> Self
	{
		match &self.0
		{
			Repr::Small(base) => match base.checked_pow(exponent)
			{
				Some(power) => Self(Repr::Small(power)),
				// The power exceeds u128::MAX, so it is big.
				None => Self(Repr::Big(BigUint::from(*base).pow(exponent)))
			},
			// Only the zeroth power of a big weight is small.
			Repr::Big(_) if exponent == 0 => Self::ONE,
			Repr::Big(base) => Self(Repr::Big(base.pow(exponent)))
		}
	}

	/// Subtract the given weight from this one, unless it is larger.
	///
	/// # Parameters
	/// - `rhs`: The subtrahend.
	///
	/// # Returns
	/// The difference, or [`None`] if `rhs` exceeds the weight.
	pub fn checked_sub(&self, rhs: &Self) -> Option<Self>
	{
		match (&self.0, &rhs.0)
		{
			(Repr::Small(a), Repr::Small(b)) =>
			{
				u128::checked_sub(*a, *b).map(|d| Self(Repr::Small(d)))
			},
			// Every big weight exceeds every small weight.
			(Repr::Small(_), Repr::Big(_)) => None,
			(Repr::Big(a), Repr::Small(b)) => Some(Self::from_big(a - *b)),
			(Repr::Big(a), Repr::Big(b)) => a.checked_sub(b).map(Self::from_big)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Divisibility.                                //
////////////////////////////////////////////////////////////////////////////////

impl Weight
{
	/// Answer the greatest common divisor of the weight and the given one.
	///
	/// # Parameters
	/// - `other`: The other weight.
	///
	/// # Returns
	/// The greatest weight that divides both, which is zero just when both are
	/// zero.
	pub(crate) fn gcd(&self, other: &Self) -> Self
	{
		match (&self.0, &other.0)
		{
			(Repr::Small(a), Repr::Small(b)) =>
			{
				let (mut a, mut b) = (*a, *b);
				while b != 0
				{
					(a, b) = (b, a % b);
				}
				Self(Repr::Small(a))
			},
			_ =>
			{
				let (mut a, mut b) = (self.to_biguint(), other.to_biguint());
				while b != BigUint::ZERO
				{
					let r = &a % &b;
					(a, b) = (b, r);
				}
				Self::from_big(a)
			}
		}
	}

	/// Answer the least common multiple of the weight and the given one.
	///
	/// # Parameters
	/// - `other`: The other weight.
	///
	/// # Returns
	/// The least weight that both divide, which is zero just when either is
	/// zero.
	pub(crate) fn lcm(&self, other: &Self) -> Self
	{
		if self.is_zero() || other.is_zero()
		{
			return Self::ZERO
		}
		self.exact_div(&self.gcd(other)) * other
	}

	/// Divide the weight by the given one, which divides it exactly.
	///
	/// # Parameters
	/// - `divisor`: The divisor, which divides the weight.
	///
	/// # Returns
	/// The quotient.
	///
	/// # Panics
	/// If `divisor` is zero. In debug builds, also if `divisor` does not divide
	/// the weight.
	pub(crate) fn exact_div(&self, divisor: &Self) -> Self
	{
		match (&self.0, &divisor.0)
		{
			(Repr::Small(a), Repr::Small(b)) =>
			{
				debug_assert!(a % b == 0, "{b} does not divide {a}");
				Self(Repr::Small(a / b))
			},
			// Every big weight exceeds every small weight, so the quotient is
			// zero, and only zero is divisible by a big weight.
			(Repr::Small(a), Repr::Big(b)) =>
			{
				debug_assert!(*a == 0, "{b} does not divide {a}");
				Self::ZERO
			},
			_ =>
			{
				let (a, b) = (self.to_biguint(), divisor.to_biguint());
				debug_assert!(
					&a % &b == BigUint::ZERO,
					"{b} does not divide {a}"
				);
				Self::from_big(a / b)
			}
		}
	}
}

impl AddAssign<&Weight> for Weight
{
	fn add_assign(&mut self, rhs: &Weight)
	{
		match (&mut self.0, &rhs.0)
		{
			(Repr::Small(a), Repr::Small(b)) => match a.checked_add(*b)
			{
				Some(sum) => *a = sum,
				// The sum exceeds u128::MAX, so it is big.
				None => self.0 = Repr::Big(BigUint::from(*a) + *b)
			},
			(Repr::Small(a), Repr::Big(b)) => self.0 = Repr::Big(b + *a),
			(Repr::Big(a), Repr::Small(b)) => *a += *b,
			(Repr::Big(a), Repr::Big(b)) => *a += b
		}
	}
}

impl AddAssign for Weight
{
	fn add_assign(&mut self, rhs: Weight)
	{
		match (&self.0, rhs.0)
		{
			// Reuse the storage of the big addend.
			(Repr::Small(a), Repr::Big(mut b)) =>
			{
				b += *a;
				self.0 = Repr::Big(b);
			},
			(_, rhs) => *self += &Weight(rhs)
		}
	}
}

impl MulAssign<&Weight> for Weight
{
	fn mul_assign(&mut self, rhs: &Weight)
	{
		match (&mut self.0, &rhs.0)
		{
			(Repr::Small(a), Repr::Small(b)) => match a.checked_mul(*b)
			{
				Some(product) => *a = product,
				// The product exceeds u128::MAX, so it is big.
				None => self.0 = Repr::Big(BigUint::from(*a) * *b)
			},
			// Multiplication by zero is the only way to shrink a big weight.
			(Repr::Small(0), Repr::Big(_)) =>
			{},
			(Repr::Big(_), Repr::Small(0)) => self.0 = Repr::Small(0),
			(Repr::Small(a), Repr::Big(b)) => self.0 = Repr::Big(b * *a),
			(Repr::Big(a), Repr::Small(b)) => *a *= *b,
			(Repr::Big(a), Repr::Big(b)) => *a *= b
		}
	}
}

impl MulAssign for Weight
{
	fn mul_assign(&mut self, rhs: Weight)
	{
		match (&self.0, rhs.0)
		{
			// Reuse the storage of the big multiplier.
			(Repr::Small(a), Repr::Big(mut b)) if *a != 0 =>
			{
				b *= *a;
				self.0 = Repr::Big(b);
			},
			(_, rhs) => *self *= &Weight(rhs)
		}
	}
}

/// Implement a commutative binary operator over [`Weight`]s, for every
/// combination of operands of which at least one is owned, in terms of its
/// compound assignment operator, so that the result reuses the owned
/// operand's storage.
macro_rules! weight_binop {
	($op:ident, $method:ident, $assign:ident) => {
		impl $op for Weight
		{
			type Output = Weight;

			fn $method(mut self, rhs: Weight) -> Weight
			{
				self.$assign(rhs);
				self
			}
		}

		impl $op<&Weight> for Weight
		{
			type Output = Weight;

			fn $method(mut self, rhs: &Weight) -> Weight
			{
				self.$assign(rhs);
				self
			}
		}

		impl $op<Weight> for &Weight
		{
			type Output = Weight;

			fn $method(self, mut rhs: Weight) -> Weight
			{
				rhs.$assign(self);
				rhs
			}
		}
	};
}

weight_binop!(Add, add, add_assign);
weight_binop!(Mul, mul, mul_assign);

impl Add<&Weight> for &Weight
{
	type Output = Weight;

	fn add(self, rhs: &Weight) -> Weight
	{
		let mut sum = self.clone();
		sum += rhs;
		sum
	}
}

/// Multiplies borrowed weights directly, so that a big product allocates only
/// its own storage, rather than a copy of an operand that it then replaces.
/// Accumulating products, as `sum += &a * &b`, which convolutions do for every
/// pair of outcomes, then adds each in place and frees it.
impl Mul<&Weight> for &Weight
{
	type Output = Weight;

	fn mul(self, rhs: &Weight) -> Weight
	{
		match (&self.0, &rhs.0)
		{
			(Repr::Small(a), Repr::Small(b)) => match a.checked_mul(*b)
			{
				Some(product) => Weight(Repr::Small(product)),
				// The product exceeds u128::MAX, so it is big.
				None => Weight(Repr::Big(BigUint::from(*a) * *b))
			},
			// Multiplication by zero is the only way to shrink a big weight.
			(Repr::Small(0), Repr::Big(_)) | (Repr::Big(_), Repr::Small(0)) =>
			{
				Weight::ZERO
			},
			(Repr::Small(a), Repr::Big(b)) => Weight(Repr::Big(b * *a)),
			(Repr::Big(a), Repr::Small(b)) => Weight(Repr::Big(a * *b)),
			(Repr::Big(a), Repr::Big(b)) => Weight(Repr::Big(a * b))
		}
	}
}

impl Sum for Weight
{
	fn sum<I: Iterator<Item = Weight>>(iter: I) -> Self
	{
		iter.fold(Weight::ZERO, |sum, weight| sum + weight)
	}
}

impl<'a> Sum<&'a Weight> for Weight
{
	fn sum<I: Iterator<Item = &'a Weight>>(iter: I) -> Self
	{
		iter.fold(Weight::ZERO, |sum, weight| sum + weight)
	}
}

impl Product for Weight
{
	fn product<I: Iterator<Item = Weight>>(iter: I) -> Self
	{
		iter.fold(Weight::ONE, |product, weight| product * weight)
	}
}

impl<'a> Product<&'a Weight> for Weight
{
	fn product<I: Iterator<Item = &'a Weight>>(iter: I) -> Self
	{
		iter.fold(Weight::ONE, |product, weight| product * weight)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                          Formatting and parsing.                           //
////////////////////////////////////////////////////////////////////////////////

impl Display for Weight
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match &self.0
		{
			Repr::Small(n) => Display::fmt(n, f),
			Repr::Big(n) => Display::fmt(n, f)
		}
	}
}

/// Parses a weight from its decimal digits, optionally preceded by `+`, as
/// [`u128`] does, but without an upper bound.
impl FromStr for Weight
{
	type Err = ParseWeightError;

	fn from_str(s: &str) -> Result<Self, Self::Err>
	{
		let digits = s.strip_prefix('+').unwrap_or(s);
		if digits.is_empty()
		{
			return Err(ParseWeightError::Empty)
		}
		if !digits.bytes().all(|b| b.is_ascii_digit())
		{
			return Err(ParseWeightError::InvalidDigit)
		}
		match digits.parse::<u128>()
		{
			Ok(n) => Ok(Self(Repr::Small(n))),
			// Decimal digits fail to parse as a u128 only by overflowing it,
			// and always parse as a big integer.
			Err(_) => Ok(Self(Repr::Big(
				digits.parse().expect("decimal digits parse as big")
			)))
		}
	}
}

/// The reason that a string does not [parse](Weight::from_str) as a
/// [`Weight`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseWeightError
{
	/// The string has no digits.
	Empty,

	/// The string has a character other than a decimal digit, besides an
	/// optional leading `+`.
	InvalidDigit
}

impl Display for ParseWeightError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			Self::Empty => write!(f, "cannot parse weight from empty string"),
			Self::InvalidDigit => write!(f, "invalid digit found in string")
		}
	}
}

impl Error for ParseWeightError {}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// Serializes a weight as a decimal string.
#[cfg(feature = "serde")]
impl Serialize for Weight
{
	fn serialize<S: Serializer>(&self, serializer: S)
	-> Result<S::Ok, S::Error>
	{
		serializer.collect_str(self)
	}
}

/// Deserializes a weight from a decimal string.
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Weight
{
	fn deserialize<D: Deserializer<'de>>(
		deserializer: D
	) -> Result<Self, D::Error>
	{
		deserializer.deserialize_str(WeightVisitor)
	}
}

/// The [`Visitor`] that deserializes a [`Weight`] from a decimal string.
#[cfg(feature = "serde")]
struct WeightVisitor;

#[cfg(feature = "serde")]
impl Visitor<'_> for WeightVisitor
{
	type Value = Weight;

	fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		f.write_str("a nonnegative integer as a decimal string")
	}

	fn visit_str<E: de::Error>(self, v: &str) -> Result<Weight, E>
	{
		v.parse()
			.map_err(|e| E::custom(format_args!("invalid weight {v:?}: {e}")))
	}
}
