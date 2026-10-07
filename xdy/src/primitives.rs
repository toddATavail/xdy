//! # Primitives
//!
//! Herein are the primitive actions that underlie the execution of
//! instructions and the evaluation of expressions. Codification here
//! allows clients to reuse the low-level primitives for their own purposes,
//! and ensures compatibility with the [evaluator](Evaluator).

use std::{cmp::Reverse, ops::RangeInclusive};

use rand::{Rng, RngExt as _};

#[cfg(doc)]
use crate::Evaluator;
use crate::{Die, Fate, RollingRecord, RollingRecordKind};

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
/// assert_eq!(rolling_record.dice.len(), 1);
/// assert!(rolling_record.dice[0].value >= 1 && rolling_record.dice[0].value <= 6);
/// ```
///
/// Use an empty range:
/// ```rust
/// # use core::ops::RangeInclusive;
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_range};
/// let rolling_record = roll_range(&mut rng(), 6..=1);
/// assert_eq!(rolling_record.dice.len(), 1);
/// assert_eq!(rolling_record.dice[0].value, 0);
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
		dice: vec![Die::kept(result)]
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
/// assert_eq!(rolling_record.dice.len(), 3);
/// rolling_record.kept().into_iter().for_each(|result| {
///     assert!(result >= 1 && result <= 6);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut rng(), 0, 6);
/// assert_eq!(rolling_record.dice.len(), 0);
///
/// let rolling_record = roll_standard_dice(&mut rng(), -1, 6);
/// assert_eq!(rolling_record.dice.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut rng(), 3, 0);
/// assert_eq!(rolling_record.dice.len(), 3);
/// rolling_record.kept().into_iter().for_each(|result| {
///    assert_eq!(result, 0);
/// });
///
/// let rolling_record = roll_standard_dice(&mut rng(), 3, -1);
/// assert_eq!(rolling_record.dice.len(), 3);
/// rolling_record.kept().into_iter().for_each(|result| {
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
	let dice = (0..count)
		.map(|_| match faces <= 0
		{
			false => Die::kept(rng.random_range(1..=faces)),
			true => Die::kept(0)
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Standard { count, faces },
		dice
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
/// assert_eq!(rolling_record.dice.len(), 3);
/// rolling_record.kept().into_iter().for_each(|result| {
///    assert!(result >= -1 && result <= 1);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut rng(), 0, vec![1, 2, 3]);
/// assert_eq!(rolling_record.dice.len(), 0);
///
/// let rolling_record = roll_custom_dice(&mut rng(), -1, vec![1, 2, 3]);
/// assert_eq!(rolling_record.dice.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut rng(), 3, vec![]);
/// assert_eq!(rolling_record.dice.len(), 3);
/// rolling_record.kept().into_iter().for_each(|result| {
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
	let dice = (0..count)
		.map(|_| match faces.is_empty()
		{
			false => Die::kept(faces[rng.random_range(0..face_count)]),
			true => Die::kept(0)
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Custom { count, faces },
		dice
	}
}

impl RollingRecord
{
	/// Cross out the lowest `count` of the dice that are still kept. Of equal
	/// dice, cross out the first rolled first. A `count` of zero or less
	/// crosses out nothing, and a `count` of at least the dice still kept
	/// crosses out all of them. Drops accumulate across calls, and their order
	/// does not change which results are kept.
	///
	/// # Parameters
	/// - `count`: The number of dice to drop.
	///
	/// # Examples
	/// Roll `3D12` and drop the lowest die:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 12);
	/// rolling_record.drop_lowest(1);
	/// assert_eq!(rolling_record.dice.len(), 3);
	/// assert_eq!(rolling_record.dropped().len(), 1);
	/// ```
	///
	/// Drop the lowest of `[3, 1, 4, 1]`, which crosses out the first `1`:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 4, faces: 4 },
	///     dice: [3, 1, 4, 1].into_iter().map(Die::kept).collect()
	/// };
	/// rolling_record.drop_lowest(1);
	/// assert_eq!(rolling_record.to_string(), "4D4: [3, ~1, 4, 1]");
	/// ```
	///
	/// Roll `3D12`, but try to drop `4` dice, and then `-1` dice, which
	/// restores none of them:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 12);
	/// rolling_record.drop_lowest(4);
	/// assert_eq!(rolling_record.dropped().len(), 3);
	/// rolling_record.drop_lowest(-1);
	/// assert_eq!(rolling_record.dropped().len(), 3);
	/// ```
	pub fn drop_lowest(&mut self, count: i32) { self.cross_out(count, |v| v); }

	/// Cross out the highest `count` of the dice that are still kept. Of equal
	/// dice, cross out the first rolled first. A `count` of zero or less
	/// crosses out nothing, and a `count` of at least the dice still kept
	/// crosses out all of them. Drops accumulate across calls, and their order
	/// does not change which results are kept.
	///
	/// # Parameters
	/// - `count`: The number of dice to drop.
	///
	/// # Examples
	/// Roll `5D8` and drop the highest die:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 8);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.dice.len(), 5);
	/// assert_eq!(rolling_record.dropped().len(), 1);
	/// ```
	///
	/// Drop the highest of `[2, 7, 1, 7]`, which crosses out the first `7`:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 4, faces: 8 },
	///     dice: [2, 7, 1, 7].into_iter().map(Die::kept).collect()
	/// };
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.to_string(), "4D8: [2, ~7, 1, 7]");
	/// ```
	///
	/// Roll `5D8`, but try to drop `7` dice, and then `-1` dice, which
	/// restores none of them:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 8);
	/// rolling_record.drop_highest(7);
	/// assert_eq!(rolling_record.dropped().len(), 5);
	/// rolling_record.drop_highest(-1);
	/// assert_eq!(rolling_record.dropped().len(), 5);
	/// ```
	pub fn drop_highest(&mut self, count: i32)
	{
		self.cross_out(count, Reverse);
	}

	/// Keep the lowest `count` of the dice that are still kept, crossing out
	/// the rest of them, highest first. A `count` of zero or less keeps
	/// nothing. Unlike a drop, a keep counts from the dice that are still
	/// kept, so it does not commute with drops or other keeps.
	///
	/// # Parameters
	/// - `count`: The number of dice to keep.
	///
	/// # Examples
	/// Roll `4D6` and keep the lowest 3 dice:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 4, 6);
	/// rolling_record.keep_lowest(3);
	/// assert_eq!(rolling_record.dropped().len(), 1);
	/// assert_eq!(rolling_record.kept().len(), 3);
	/// ```
	///
	/// Drop the lowest of `[5, 2, 6, 3]`, and keep the lowest 2 of the other
	/// 3, which crosses out the `6`:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 4, faces: 6 },
	///     dice: [5, 2, 6, 3].into_iter().map(Die::kept).collect()
	/// };
	/// rolling_record.drop_lowest(1);
	/// rolling_record.keep_lowest(2);
	/// assert_eq!(rolling_record.to_string(), "4D6: [5, ~2, ~6, 3]");
	/// assert_eq!(rolling_record.kept(), vec![5, 3]);
	/// ```
	///
	/// Roll `4D6`, but try to keep `7` dice, which keeps all `4`, or `0`
	/// dice, which keeps none:
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
		self.drop_highest(keep_drops(self.kept_count(), 0, 0, count));
	}

	/// Keep the highest `count` of the dice that are still kept, crossing out
	/// the rest of them, lowest first. A `count` of zero or less keeps
	/// nothing. Unlike a drop, a keep counts from the dice that are still
	/// kept, so it does not commute with drops or other keeps.
	///
	/// # Parameters
	/// - `count`: The number of dice to keep.
	///
	/// # Examples
	/// Roll `2D20` and keep the highest die:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 2, 20);
	/// rolling_record.keep_highest(1);
	/// assert_eq!(rolling_record.dropped().len(), 1);
	/// assert_eq!(rolling_record.kept().len(), 1);
	/// ```
	///
	/// Keep the highest of `[3, 3]`, which crosses out the first `3`:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 2, faces: 20 },
	///     dice: [3, 3].into_iter().map(Die::kept).collect()
	/// };
	/// rolling_record.keep_highest(1);
	/// assert_eq!(rolling_record.to_string(), "2D20: [~3, 3]");
	/// ```
	///
	/// Roll `5D10`, keep the highest 3 dice, and then keep the highest 2 of
	/// those, which crosses out 1 more:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 5, 10);
	/// rolling_record.keep_highest(3);
	/// rolling_record.keep_highest(2);
	/// assert_eq!(rolling_record.dropped().len(), 3);
	/// assert_eq!(rolling_record.kept().len(), 2);
	/// ```
	pub fn keep_highest(&mut self, count: i32)
	{
		self.drop_lowest(keep_drops(self.kept_count(), 0, 0, count));
	}

	/// Reroll the dice that are still kept and that the criterion matches, each
	/// up to `limit` times, or until it no longer matches if `limit` is `None`.
	/// Consider the dice in the order rolled, and finish each before beginning
	/// the next: cross out a matching die, roll its replacement from the same
	/// die, append the replacement, [linked](Die::replaces) to the die that it
	/// replaces, and repeat with the replacement while it matches and the limit
	/// allows. A `limit` of zero or less rerolls nothing. If the limit is
	/// `None` and the criterion matches every face of the die, then roll
	/// nothing, but cross out every matching die, so that each contributes
	/// zero. A die without faces, or an empty range, shows only the face `0`.
	///
	/// # Type parameters
	/// - `R`: The type of the pseudo-random number generator.
	///
	/// # Parameters
	/// - `rng`: The pseudo-random number generator.
	/// - `criterion`: The set of results to reroll.
	/// - `limit`: The number of times that each die may be rerolled, or `None`
	///   for no limit.
	///
	/// # Returns
	/// The number of dice rolled, one for each reroll.
	///
	/// # Notes
	/// An unlimited reroll terminates with probability one, but the number of
	/// dice that it rolls is unbounded: if the criterion matches all but one of
	/// `n` equally likely faces, each die takes `n` rolls on average. The
	/// [evaluator](Evaluator) [meters](Evaluator::evaluate_metered) every
	/// reroll as a die, so that a budget bounds them.
	///
	/// `lean/spec/XdySpec/Unlimited.lean` justifies the specification's law of
	/// an unlimited reroll against this loop: after `n` rolls, a die rolls
	/// again only with the `n`th power of the probability that it matches
	/// (`matched_limitedReplacement`), and as `n` grows, the laws of rerolls
	/// limited to `n` times converge to the conditional law of the die, given
	/// that it does not match (`tendsto_limitedReplacement`, `tendsto_reroll`).
	/// If every face matches, then no limited reroll changes the law of the die
	/// (`limitedReplacement_of_matches`), so the loop has no limit; crossing
	/// out is a convention, under which each matching die contributes zero
	/// (`sum_reroll_crossOut`).
	///
	/// # Examples
	/// Reroll the `1` of `[2, 1, 4]` once, which appends its replacement:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{Criterion, Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 3, faces: 4 },
	///     dice: [2, 1, 4].into_iter().map(Die::kept).collect()
	/// };
	/// let rolled = rolling_record.reroll(&mut rng(), &Criterion::Within(1..=1), Some(1));
	/// assert_eq!(rolled, 1);
	/// assert_eq!(rolling_record.dice.len(), 4);
	/// assert_eq!(rolling_record.dice[3].replaces, Some(1));
	/// assert!(rolling_record.to_string().starts_with("3D4: [2, ~1, 4, "));
	/// ```
	///
	/// Reroll every die of `3D1` that shows `1`, without limit, which rolls
	/// nothing, since every face matches:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{Criterion, RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 3, 1);
	/// let rolled = rolling_record.reroll(&mut rng(), &Criterion::Within(1..=1), None);
	/// assert_eq!(rolled, 0);
	/// assert_eq!(rolling_record.to_string(), "3D1: [~1, ~1, ~1]");
	/// assert_eq!(rolling_record.sum(), 0);
	/// ```
	pub fn reroll<R>(
		&mut self,
		rng: &mut R,
		criterion: &Criterion,
		limit: Option<i32>
	) -> u64
	where
		R: Rng + ?Sized
	{
		self.reroll_within(rng, criterion, limit, u64::MAX)
			.expect("no reroll rolls `u64::MAX` dice")
	}

	/// Reroll the dice that are still kept and that the criterion matches, as
	/// [`reroll`](Self::reroll) does, but roll no more than `budget` dice.
	///
	/// # Type parameters
	/// - `R`: The type of the pseudo-random number generator.
	///
	/// # Parameters
	/// - `rng`: The pseudo-random number generator.
	/// - `criterion`: The set of results to reroll.
	/// - `limit`: The number of times that each die may be rerolled, or `None`
	///   for no limit.
	/// - `budget`: The greatest number of dice that the reroll may roll.
	///
	/// # Returns
	/// The number of dice rolled, or `None` if the reroll needed to roll more
	/// than `budget` dice, in which case it rolled exactly `budget` dice and
	/// then stopped, drawing nothing more from the pRNG.
	pub(crate) fn reroll_within<R>(
		&mut self,
		rng: &mut R,
		criterion: &Criterion,
		limit: Option<i32>,
		budget: u64
	) -> Option<u64>
	where
		R: Rng + ?Sized
	{
		let matching = self
			.dice
			.iter()
			.enumerate()
			.filter(|(_, die)| die.is_kept() && criterion.matches(die.value))
			.map(|(index, _)| index)
			.collect::<Vec<_>>();
		if limit.is_none() && self.kind.faces_within(criterion)
		{
			// Every face matches, so no reroll could ever stop. Roll nothing,
			// but cross out every matching die.
			for index in matching
			{
				self.dice[index].fate = Fate::Rerolled;
			}
			return Some(0)
		}
		let limit = limit.map(|limit| u64::try_from(limit).unwrap_or(0));
		let mut rolled = 0u64;
		for index in matching
		{
			let mut current = index;
			let mut times = 0u64;
			while criterion.matches(self.dice[current].value)
				&& limit.is_none_or(|limit| times < limit)
			{
				if rolled == budget
				{
					return None
				}
				rolled += 1;
				times += 1;
				self.dice[current].fate = Fate::Rerolled;
				let value = self.kind.roll_die(rng);
				self.dice.push(Die::replacing(value, current));
				current = self.dice.len() - 1;
			}
		}
		Some(rolled)
	}

	/// Answer the results of the dice that are still kept.
	///
	/// # Returns
	/// The kept results, in the order rolled.
	///
	/// # Examples
	/// Drop the lowest and highest of `[4, 1, 6, 3]`, leaving the middle two:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 4, faces: 6 },
	///     dice: [4, 1, 6, 3].into_iter().map(Die::kept).collect()
	/// };
	/// rolling_record.drop_lowest(1);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.kept(), vec![4, 3]);
	/// assert_eq!(rolling_record.dropped(), vec![1, 6]);
	/// ```
	pub fn kept(&self) -> Vec<i32>
	{
		self.dice
			.iter()
			.filter(|die| die.is_kept())
			.map(|die| die.value)
			.collect()
	}

	/// Answer the results of the dice that a drop or keep crossed out.
	///
	/// # Returns
	/// The dropped results, in the order rolled.
	pub fn dropped(&self) -> Vec<i32>
	{
		self.dice
			.iter()
			.filter(|die| die.fate == Fate::Dropped)
			.map(|die| die.value)
			.collect()
	}

	/// Answer the results of the dice that a reroll crossed out.
	///
	/// # Returns
	/// The rerolled results, in the order rolled.
	pub fn rerolled(&self) -> Vec<i32>
	{
		self.dice
			.iter()
			.filter(|die| die.fate == Fate::Rerolled)
			.map(|die| die.value)
			.collect()
	}

	/// Compute the sum of the [kept](Self::kept) results, i.e., excluding the
	/// dice crossed out. If every die is crossed out, then nothing is kept and
	/// the sum is `0`. Saturate the sum on overflow. Consecutive calls are
	/// idempotent.
	///
	/// # Returns
	/// The sum of the results.
	///
	/// # Notes
	/// The sum folds the kept results in ascending order, not in the order
	/// rolled. Saturating addition is not associative, so the order matters
	/// when the results have mixed signs and the sum saturates: `i32::MAX`,
	/// `1`, and `-1` sum to `i32::MAX` in ascending order, but to
	/// `i32::MAX - 1` in the order given. Folding in ascending order makes the
	/// sum depend only on the multiset of kept results, as the forward pass
	/// and the Lean specification require.
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
	/// Roll `4D6`, drop the lowest and highest die, and sum the middle dice:
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
		let mut kept = self.kept();
		kept.sort();
		kept.iter().fold(0i32, |a, b| a.saturating_add(*b))
	}

	/// Answer the number of dice that are still kept.
	///
	/// # Returns
	/// The number of kept dice, saturated to `i32::MAX`.
	fn kept_count(&self) -> i32
	{
		let kept = self.dice.iter().filter(|die| die.is_kept()).count();
		i32::try_from(kept).unwrap_or(i32::MAX)
	}

	/// Cross out the first `count` of the dice that are still kept, ordered by
	/// the specified key and then by the order rolled, so that of dice with
	/// equal keys the first rolled is crossed out first. A `count` of zero or
	/// less crosses out nothing.
	///
	/// # Type parameters
	/// - `K`: The type of the key.
	///
	/// # Parameters
	/// - `count`: The number of dice to cross out.
	/// - `key`: The key of a result: the result itself to cross out the lowest
	///   dice, or its [reverse](Reverse) to cross out the highest.
	fn cross_out<K>(&mut self, count: i32, key: impl Fn(i32) -> K)
	where
		K: Ord
	{
		let count = usize::try_from(count).unwrap_or(0);
		if count == 0
		{
			return
		}
		let mut kept = self
			.dice
			.iter()
			.enumerate()
			.filter(|(_, die)| die.is_kept())
			.map(|(index, die)| (key(die.value), index))
			.collect::<Vec<_>>();
		kept.sort_unstable();
		for &(_, index) in kept.iter().take(count)
		{
			self.dice[index].fate = Fate::Dropped;
		}
	}
}

impl RollingRecordKind<i32>
{
	/// Roll one die of the kind, as the roll that filled the record rolled
	/// each of its dice.
	///
	/// # Type parameters
	/// - `R`: The type of the pseudo-random number generator.
	///
	/// # Parameters
	/// - `rng`: The pseudo-random number generator.
	///
	/// # Returns
	/// The result of the die.
	///
	/// # Panics
	/// If the record is uninitialized, since it has no die.
	fn roll_die<R>(&self, rng: &mut R) -> i32
	where
		R: Rng + ?Sized
	{
		match self
		{
			RollingRecordKind::Range { start, end } if end < start => 0,
			RollingRecordKind::Range { start, end } =>
			{
				rng.random_range(*start..=*end)
			},
			RollingRecordKind::Standard { faces, .. } if *faces <= 0 => 0,
			RollingRecordKind::Standard { faces, .. } =>
			{
				rng.random_range(1..=*faces)
			},
			RollingRecordKind::Custom { faces, .. } if faces.is_empty() => 0,
			RollingRecordKind::Custom { faces, .. } =>
			{
				faces[rng.random_range(0..faces.len())]
			},
			RollingRecordKind::Uninitialized =>
			{
				unreachable!("an uninitialized record has no die")
			}
		}
	}

	/// Answer whether the criterion matches every face of the die of the
	/// kind. A die without faces, or an empty range, shows only the face `0`.
	///
	/// # Parameters
	/// - `criterion`: The criterion.
	///
	/// # Returns
	/// `true` if every face matches, `false` otherwise. An uninitialized
	/// record has no die, so every face of it matches vacuously.
	fn faces_within(&self, criterion: &Criterion) -> bool
	{
		match self
		{
			RollingRecordKind::Range { start, end } if end < start =>
			{
				criterion.matches(0)
			},
			RollingRecordKind::Range { start, end } =>
			{
				criterion.covers(*start..=*end)
			},
			RollingRecordKind::Standard { faces, .. } if *faces <= 0 =>
			{
				criterion.matches(0)
			},
			RollingRecordKind::Standard { faces, .. } =>
			{
				criterion.covers(1..=*faces)
			},
			RollingRecordKind::Custom { faces, .. } if faces.is_empty() =>
			{
				criterion.matches(0)
			},
			RollingRecordKind::Custom { faces, .. } =>
			{
				faces.iter().all(|face| criterion.matches(*face))
			},
			RollingRecordKind::Uninitialized => true
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Criteria.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The set of results that a [reroll](RollingRecord::reroll) rerolls: a
/// [predicate](crate::Predicate) whose operands are
/// [resolved](crate::Predicate::resolve) to values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Criterion
{
	/// Matches every result within the inclusive range, or none if the range
	/// is empty.
	Within(RangeInclusive<i32>),

	/// Matches every result outside the inclusive range, or every result if
	/// the range is empty.
	Outside(RangeInclusive<i32>),

	/// Matches every listed result, which are in ascending order, without
	/// repetition.
	Among(Vec<i32>)
}

impl Criterion
{
	/// Answer the criterion that matches every result from `start` to `end`,
	/// inclusive, of those that fit in an `i32`, or none if there are none.
	///
	/// # Parameters
	/// - `start`: The least result matched.
	/// - `end`: The greatest result matched.
	///
	/// # Returns
	/// The criterion.
	pub(crate) fn within(start: i64, end: i64) -> Self
	{
		let start = start.max(i32::MIN as i64);
		let end = end.min(i32::MAX as i64);
		match start <= end
		{
			true => Self::Within(start as i32..=end as i32),
			// Any empty range will do.
			false => Self::Within(RangeInclusive::new(1, 0))
		}
	}

	/// Answer whether the criterion matches the specified result.
	///
	/// # Parameters
	/// - `value`: The result.
	///
	/// # Returns
	/// `true` if the criterion matches the result, `false` otherwise.
	///
	/// # Examples
	/// ```rust
	/// use xdy::Criterion;
	///
	/// assert!(Criterion::Within(1..=2).matches(2));
	/// assert!(!Criterion::Outside(1..=2).matches(2));
	/// assert!(Criterion::Among(vec![1, 3]).matches(3));
	/// assert!(!Criterion::Within(2..=1).matches(2));
	/// ```
	pub fn matches(&self, value: i32) -> bool
	{
		match self
		{
			Self::Within(range) => range.contains(&value),
			Self::Outside(range) => !range.contains(&value),
			Self::Among(values) => values.binary_search(&value).is_ok()
		}
	}

	/// Answer whether the criterion matches every result of the specified
	/// range, in time independent of its width.
	///
	/// # Parameters
	/// - `range`: The range, which is not empty.
	///
	/// # Returns
	/// `true` if the criterion matches every result of the range, `false`
	/// otherwise.
	fn covers(&self, range: RangeInclusive<i32>) -> bool
	{
		let (start, end) = range.into_inner();
		match self
		{
			Self::Within(within) =>
			{
				within.start() <= &start && &end <= within.end()
			},
			Self::Outside(outside) =>
			{
				outside.is_empty()
					|| end < *outside.start()
					|| *outside.end() < start
			},
			Self::Among(values) =>
			{
				// The values are distinct, so they cover the range exactly if
				// as many of them lie within it as it has results.
				let first = values.partition_point(|value| *value < start);
				let last = values.partition_point(|value| *value <= end);
				(last - first) as i64 == end as i64 - start as i64 + 1
			}
		}
	}
}

/// Answer how many of the kept results of a rolling record a keep drops: all
/// of them but `count`, or none if no more than `count` are kept. This serves
/// as the definition of [`RollingRecord::keep_lowest`] and
/// [`RollingRecord::keep_highest`], which drop this many results from the
/// opposite end, and of the keeps of the forward pass, which share it. A
/// rolling record passes the number of dice that it still keeps, and no drops,
/// since it has already crossed out the dice that it dropped.
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

/// Computes the lesser of the two operands.
///
/// # Parameters
/// - `op1`: The first operand.
/// - `op2`: The second operand.
///
/// # Returns
/// The lesser of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::min;
/// assert_eq!(min(1, 2), 1);
/// assert_eq!(min(0, -3), -3);
/// assert_eq!(min(i32::MIN, i32::MAX), i32::MIN);
/// ```
#[inline]
pub fn min(op1: i32, op2: i32) -> i32 { op1.min(op2) }
