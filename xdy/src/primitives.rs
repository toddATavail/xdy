//! # Primitives
//!
//! Herein are the primitive actions that underlie the execution of
//! instructions and the evaluation of expressions. Codification here
//! allows clients to reuse the low-level primitives for their own purposes,
//! and ensures compatibility with the [evaluator](Evaluator).

use std::ops::RangeInclusive;

use rand::{Rng, RngExt as _};

#[cfg(doc)]
use crate::Evaluator;
use crate::{RollingRecord, RollingRecordKind};

////////////////////////////////////////////////////////////////////////////////
//                                Primitives.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Generate a value within the specified range, using the same algorithm as
/// the [evaluator](Evaluator). If the range is empty (i.e., `end<start`), then
/// the result is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `range`: The range of values to generate.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a value within the range `1..=6`:
///
/// ```rust
/// # use core::ops::RangeInclusive;
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_range};
/// let rolling_record = roll_range(&mut rng(), 1..=6);
/// assert_eq!(rolling_record.results.len(), 1);
/// assert!(rolling_record.results[0] >= 1 && rolling_record.results[0] <= 6);
/// ```
///
/// Use an empty range:
/// ```rust
/// # use core::ops::RangeInclusive;
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_range};
/// let rolling_record = roll_range(&mut rng(), 6..=1);
/// assert_eq!(rolling_record.results.len(), 1);
/// assert_eq!(rolling_record.results[0], 0);
/// ```
pub fn roll_range<R>(rng: &mut R, range: RangeInclusive<i32>) -> RollingRecord
where
	R: Rng + ?Sized
{
	let (start, end) = range.into_inner();
	let result = match end < start
	{
		false => rng.random_range(start..=end),
		true => 0
	};
	RollingRecord {
		kind: RollingRecordKind::Range { start, end },
		results: vec![result],
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

/// Generate a standard dice roll, using the same algorithm as the
/// [evaluator](Evaluator). If `count≤0`, then no dice are rolled. If `faces≤0`,
/// then the result of each die is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `count`: The number of dice to roll.
/// - `faces`: The number of faces on each die, numbered from 1 to `faces`.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a roll of 3 six-sided dice (i.e., `3D6`):
///
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut rng(), 3, 6);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///     assert!(result >= 1 && result <= 6);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut rng(), 0, 6);
/// assert_eq!(rolling_record.results.len(), 0);
///
/// let rolling_record = roll_standard_dice(&mut rng(), -1, 6);
/// assert_eq!(rolling_record.results.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut rng(), 3, 0);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///    assert_eq!(result, 0);
/// });
///
/// let rolling_record = roll_standard_dice(&mut rng(), 3, -1);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///   assert_eq!(result, 0);
/// });
/// ```
pub fn roll_standard_dice<R>(
	rng: &mut R,
	count: i32,
	faces: i32
) -> RollingRecord
where
	R: Rng + ?Sized
{
	let results = (0..count)
		.map(|_| match faces <= 0
		{
			false => rng.random_range(1..=faces),
			true => 0
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Standard { count, faces },
		results,
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

/// Generate a custom dice roll, using the same algorithm as the
/// [evaluator](Evaluator). If `count≤0`, then no dice are rolled. If `faces`
/// is empty, then the result of each die is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `count`: The number of dice to roll.
/// - `faces`: The faces on each die.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a roll of 3 dice with faces `-1`, `0`, and `1` (i.e.,
/// `3D[-1,0,1]`):
///
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut rng(), 3, vec![-1, 0, 1]);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///    assert!(result >= -1 && result <= 1);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut rng(), 0, vec![1, 2, 3]);
/// assert_eq!(rolling_record.results.len(), 0);
///
/// let rolling_record = roll_custom_dice(&mut rng(), -1, vec![1, 2, 3]);
/// assert_eq!(rolling_record.results.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut rng(), 3, vec![]);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///   assert_eq!(result, 0);
/// });
/// ```
pub fn roll_custom_dice<R>(
	rng: &mut R,
	count: i32,
	faces: Vec<i32>
) -> RollingRecord
where
	R: Rng + ?Sized
{
	let face_count = faces.len();
	let results = (0..count)
		.map(|_| match faces.is_empty()
		{
			false => faces[rng.random_range(0..face_count)],
			true => 0
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Custom { count, faces },
		results,
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

impl RollingRecord
{
	/// Mark the lowest `count` results as dropped. A `count` of zero or less
	/// drops nothing. Drops accumulate across calls, and the total is clamped
	/// to the number of results, so the order of the calls does not matter.
	///
	/// # Parameters
	/// - `count`: The number of results to drop.
	///
	/// # Examples
	/// Roll `3D12` and drop the lowest result:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 12);
	/// rolling_record.drop_lowest(1);
	/// assert_eq!(rolling_record.results.len(), 3);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// ```
	///
	/// Roll `3D12`, but try to drop `4` results:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 12);
	/// rolling_record.drop_lowest(4);
	/// assert_eq!(rolling_record.results.len(), 3);
	/// assert_eq!(rolling_record.lowest_dropped, 3);
	/// ```
	///
	/// Roll `3D12` and drop the lowest 3 results, then try to drop `-1`
	/// results, which restores none of them:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 12);
	/// rolling_record.drop_lowest(3);
	/// rolling_record.drop_lowest(-1);
	/// assert_eq!(rolling_record.lowest_dropped, 3);
	/// ```
	pub fn drop_lowest(&mut self, count: i32)
	{
		self.lowest_dropped = self
			.lowest_dropped
			.saturating_add(count.max(0))
			.clamp(0, self.results.len() as i32);
	}

	/// Answer the results that have been dropped because they were the lowest.
	///
	/// # Returns
	/// The lowest dropped results, in ascending order.
	pub fn lowest_dropped(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		results
			.iter()
			.take(self.lowest_dropped as usize)
			.copied()
			.collect::<Vec<_>>()
	}

	/// Mark the highest `count` results as dropped. A `count` of zero or less
	/// drops nothing. Drops accumulate across calls, and the total is clamped
	/// to the number of results, so the order of the calls does not matter.
	///
	/// # Parameters
	/// - `count`: The number of results to drop.
	///
	/// # Examples
	/// Roll `5D8` and drop the highest result:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 8);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.results.len(), 5);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// ```
	///
	/// Roll `5D8`, but try to drop `7` results:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 8);
	/// rolling_record.drop_highest(7);
	/// assert_eq!(rolling_record.results.len(), 5);
	/// assert_eq!(rolling_record.highest_dropped, 5);
	/// ```
	///
	/// Roll `5D8` and drop the highest 5 results, then try to drop `-1`
	/// results, which restores none of them:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 8);
	/// rolling_record.drop_highest(5);
	/// rolling_record.drop_highest(-1);
	/// assert_eq!(rolling_record.highest_dropped, 5);
	/// ```
	pub fn drop_highest(&mut self, count: i32)
	{
		self.highest_dropped = self
			.highest_dropped
			.saturating_add(count.max(0))
			.clamp(0, self.results.len() as i32);
	}

	/// Answer the results that have been dropped because they were the highest.
	///
	/// # Returns
	/// The highest dropped results, in ascending order.
	pub fn highest_dropped(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		let lowest_dropped = self.lowest_dropped as usize;
		let highest_dropped = self.highest_dropped as usize;
		let count = results
			.len()
			.saturating_sub(lowest_dropped)
			.saturating_sub(highest_dropped);
		results
			.iter()
			.skip(lowest_dropped)
			.skip(count)
			.copied()
			.collect::<Vec<_>>()
	}

	/// Keep the lowest `count` of the results that are still kept, marking the
	/// rest of them as dropped from the high end. A `count` of zero or less
	/// keeps nothing. Unlike a drop, a keep counts from the results that are
	/// still kept, so it does not commute with drops or other keeps.
	///
	/// # Parameters
	/// - `count`: The number of results to keep.
	///
	/// # Examples
	/// Roll `4D6` and keep the lowest 3 results:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.keep_lowest(3);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// assert_eq!(rolling_record.kept().len(), 3);
	/// ```
	///
	/// Roll `4D6`, drop the lowest result, and keep the lowest 2 of the other
	/// 3, which drops 1 more:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.drop_lowest(1);
	/// rolling_record.keep_lowest(2);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// ```
	///
	/// Roll `4D6`, but try to keep `7` results, which keeps all `4`, or `0`
	/// results, which keeps none:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.keep_lowest(7);
	/// assert_eq!(rolling_record.kept().len(), 4);
	/// rolling_record.keep_lowest(0);
	/// assert_eq!(rolling_record.kept().len(), 0);
	/// ```
	pub fn keep_lowest(&mut self, count: i32)
	{
		self.drop_highest(keep_drops(
			self.results.len() as i32,
			self.lowest_dropped,
			self.highest_dropped,
			count
		));
	}

	/// Keep the highest `count` of the results that are still kept, marking
	/// the rest of them as dropped from the low end. A `count` of zero or less
	/// keeps nothing. Unlike a drop, a keep counts from the results that are
	/// still kept, so it does not commute with drops or other keeps.
	///
	/// # Parameters
	/// - `count`: The number of results to keep.
	///
	/// # Examples
	/// Roll `2D20` and keep the highest result:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 2, 20);
	/// rolling_record.keep_highest(1);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// assert_eq!(rolling_record.kept().len(), 1);
	/// ```
	///
	/// Roll `5D10`, keep the highest 3 results, and then keep the highest 2 of
	/// those, which drops 1 more:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 10);
	/// rolling_record.keep_highest(3);
	/// rolling_record.keep_highest(2);
	/// assert_eq!(rolling_record.lowest_dropped, 3);
	/// assert_eq!(rolling_record.kept().len(), 2);
	/// ```
	pub fn keep_highest(&mut self, count: i32)
	{
		self.drop_lowest(keep_drops(
			self.results.len() as i32,
			self.lowest_dropped,
			self.highest_dropped,
			count
		));
	}

	/// Answer the kept results, dropping the lowest and highest results as
	/// necessary.
	///
	/// # Returns
	/// The kept results, in ascending order.
	///
	/// # Examples
	/// Roll `4D6`, drop the lowest and highest die, leaving the middle two:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.drop_lowest(1);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// let kept = rolling_record.kept();
	/// assert_eq!(kept.len(), 2);
	/// for result in rolling_record.lowest_dropped() {
	///     assert!(result <= kept[0]);
	/// }
	/// for result in rolling_record.highest_dropped() {
	///    assert!(result >= kept[kept.len() - 1]);
	/// }
	/// ```
	pub fn kept(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		let lowest_dropped = self.lowest_dropped as usize;
		let highest_dropped = self.highest_dropped as usize;
		let count = results.len();
		let count = count
			.saturating_sub(lowest_dropped)
			.saturating_sub(highest_dropped);
		results
			.iter()
			.skip(lowest_dropped)
			.take(count)
			.copied()
			.collect()
	}

	/// Compute the sum of the [kept](Self::kept) results, i.e., excluding the
	/// dropped lowest and highest results. If the lowest and highest drops
	/// together cover every result, then nothing is kept and the sum is `0`.
	/// Saturate the sum on overflow. Consecutive calls are idempotent.
	///
	/// # Returns
	/// The sum of the results.
	///
	/// # Examples
	/// Roll `5D12` and sum the results:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 12);
	/// let sum = rolling_record.sum();
	/// assert!(sum >= 5 && sum <= 60);
	/// ```
	///
	/// Roll `4D6`, drop the lowest and highest die, and sum the middle die:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.drop_lowest(1);
	/// rolling_record.drop_highest(1);
	/// let sum = rolling_record.sum();
	/// assert!(sum >= 2 && sum <= 12);
	/// ```
	pub fn sum(&mut self) -> i32
	{
		self.kept().iter().fold(0i32, |a, b| a.saturating_add(*b))
	}
}

/// Answer how many of the kept results of a rolling record a keep drops: all
/// of them but `count`, or none if no more than `count` are kept. This serves
/// as the definition of [`RollingRecord::keep_lowest`] and
/// [`RollingRecord::keep_highest`], which drop this many results from the
/// opposite end, and of the keeps of the forward pass, which share it.
///
/// # Parameters
/// - `results`: The number of results in the record.
/// - `lowest`: The number of lowest results dropped, clamped to `results`.
/// - `highest`: The number of highest results dropped, clamped to `results`.
/// - `count`: The number of results to keep. A count of zero or less keeps
///   nothing.
///
/// # Returns
/// The number of results to drop, between `0` and the number kept.
#[inline]
pub(crate) fn keep_drops(
	results: i32,
	lowest: i32,
	highest: i32,
	count: i32
) -> i32
{
	// Each drop is clamped to the results, so the difference cannot overflow.
	let kept = (results - lowest - highest).max(0);
	kept - count.clamp(0, kept)
}

/// Computes the sum of the two operands, saturating on overflow. This serves as
/// the definition of the addition operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The augend.
/// - `op2`: The addend.
///
/// # Returns
/// The sum of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::add;
/// assert_eq!(add(1, 2), 3);
/// assert_eq!(add(i32::MAX, 1), i32::MAX);
/// assert_eq!(add(i32::MIN, -1), i32::MIN);
/// ```
#[inline]
pub fn add(op1: i32, op2: i32) -> i32 { op1.saturating_add(op2) }

/// Computes the difference of the two operands, saturating on overflow. This
/// serves as the definition of the subtraction operation in the dice expression
/// language.
///
/// # Parameters
/// - `op1`: The minuend.
/// - `op2`: The subtrahend.
///
/// # Returns
/// The difference of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::sub;
/// assert_eq!(sub(1, 2), -1);
/// assert_eq!(sub(i32::MAX, -1), i32::MAX);
/// assert_eq!(sub(i32::MIN, 1), i32::MIN);
/// ```
#[inline]
pub fn sub(op1: i32, op2: i32) -> i32 { op1.saturating_sub(op2) }

/// Computes the product of the two operands, saturating on overflow. This
/// serves as the definition of the multiplication operation in the dice
/// expression language.
///
/// # Parameters
/// - `op1`: The multiplicand.
/// - `op2`: The multiplier.
///
/// # Returns
/// The product of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::mul;
/// assert_eq!(mul(2, 3), 6);
/// assert_eq!(mul(i32::MAX, 2), i32::MAX);
/// assert_eq!(mul(i32::MIN, 2), i32::MIN);
/// assert_eq!(mul(i32::MAX, -1), i32::MIN + 1);
/// ```
#[inline]
pub fn mul(op1: i32, op2: i32) -> i32 { op1.saturating_mul(op2) }

/// Computes the quotient of the first operand divided by the second operand,
/// saturating on overflow. Division by zero is treated as zero. This serves as
/// the definition of the division operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The dividend.
/// - `op2`: The divisor.
///
/// # Returns
/// The quotient of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::div;
/// assert_eq!(div(6, 2), 3);
/// assert_eq!(div(6, 0), 0);
/// assert_eq!(div(i32::MIN, -1), i32::MAX);
/// ```
#[inline]
pub fn div(op1: i32, op2: i32) -> i32
{
	match op2
	{
		0 => 0,
		_ => op1.saturating_div(op2)
	}
}

/// Computes the remainder of the first operand divided by the second operand,
/// saturating on overflow. This serves as the definition of the remainder
/// operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The dividend.
/// - `op2`: The divisor.
///
/// # Returns
/// The remainder of `op1` divided by `op2`.
///
/// # Examples
/// ```rust
/// # use xdy::r#mod;
/// assert_eq!(r#mod(6, 4), 2);
/// assert_eq!(r#mod(6, 0), 0);
/// assert_eq!(r#mod(i32::MIN, -1), 0);
/// ```
pub fn r#mod(op1: i32, op2: i32) -> i32
{
	match (op1, op2)
	{
		(i32::MIN, -1) => 0,
		(_, 0) => 0,
		(_, _) => op1 % op2
	}
}

/// Raises the first operand to the power of the second operand, saturating on
/// overflow. Produces correct answers for base 0 or 1 even for negative
/// exponents, otherwise treats `x^y` when `y<0` as `0`. This serves as the
/// definition of exponentiation in the dice expression language.
///
/// # Parameters
/// - `op1`: The base.
/// - `op2`: The exponent.
///
/// # Returns
/// The result of `op1` raised to the power of `op2`.
///
/// # Examples
/// ```rust
/// # use xdy::exp;
/// assert_eq!(exp(2, 3), 8);
/// assert_eq!(exp(2, 0), 1);
/// assert_eq!(exp(0, 0), 1);
/// assert_eq!(exp(0, 1), 0);
/// assert_eq!(exp(0, -1), 0);
/// assert_eq!(exp(1, 0), 1);
/// assert_eq!(exp(1, 1), 1);
/// assert_eq!(exp(1, -1), 1);
/// assert_eq!(exp(-1, 0), 1);
/// assert_eq!(exp(-1, 1), -1);
/// assert_eq!(exp(-1, 2), 1);
/// assert_eq!(exp(-1, 3), -1);
/// assert_eq!(exp(-1, -1), -1);
/// assert_eq!(exp(-1, -2), 1);
/// assert_eq!(exp(-1, -3), -1);
/// assert_eq!(exp(2, -1), 0);
/// assert_eq!(exp(10, 20), i32::MAX);
/// assert_eq!(exp(-10, 20), i32::MAX);
/// assert_eq!(exp(-10, 21), i32::MIN);
/// ```
pub fn exp(op1: i32, op2: i32) -> i32
{
	match (op1, op2)
	{
		(0, 0) => 1,
		(0, _) => 0,
		(1, _) => 1,
		(-1, op2) => match op2 % 2 == 0
		{
			true => 1,
			false => -1
		},
		(_, ..0) => 0,
		(op1, op2) => op1.saturating_pow(op2.clamp(0, i32::MAX) as u32)
	}
}

/// Computes the negation of the operand, saturating on overflow. This serves as
/// the definition of the negation operation throughout the library.
///
/// # Parameters
/// - `op`: The operand.
///
/// # Returns
/// The negation of the operand.
///
/// # Examples
/// ```rust
/// # use xdy::neg;
/// assert_eq!(neg(1), -1);
/// assert_eq!(neg(i32::MAX), i32::MIN + 1);
/// assert_eq!(neg(i32::MIN), i32::MAX);
/// ```
#[inline]
pub fn neg(op: i32) -> i32 { op.saturating_neg() }

/// Computes the greater of the two operands. The dice expression language has
/// no maximum operation, but the optimizer uses this one to clamp values that
/// the language clamps implicitly, e.g., a negative count of dice, which rolls
/// nothing, without rolling any dice.
///
/// # Parameters
/// - `op1`: The first operand.
/// - `op2`: The second operand.
///
/// # Returns
/// The greater of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::max;
/// assert_eq!(max(1, 2), 2);
/// assert_eq!(max(0, -3), 0);
/// assert_eq!(max(i32::MIN, i32::MAX), i32::MAX);
/// ```
#[inline]
pub fn max(op1: i32, op2: i32) -> i32 { op1.max(op2) }
