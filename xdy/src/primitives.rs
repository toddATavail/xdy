//! # Primitives
//!
//! Herein are the primitive actions that underlie the execution of
//! instructions and the evaluation of expressions. Codification here
//! allows clients to reuse the low-level primitives for their own purposes,
//! and ensures compatibility with the [evaluator](Evaluator).

use std::{cmp::Reverse, mem, ops::RangeInclusive};

use rand::{Rng, RngExt as _};

#[cfg(doc)]
use crate::Evaluator;
use crate::{Die, Fate, Origin, RollingRecord, RollingRecordKind};

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
	/// Cross out the lowest `count` of the dice that are still kept, counting
	/// each [chain](Self::chains) as one die, worth its total, and crossing out
	/// every kept die of each chain that it crosses out. Of chains of equal
	/// totals, cross out first the least by its [contents](Self::chains), and
	/// of identical chains, the one whose earliest kept die was rolled first. A
	/// `count` of zero or less crosses out nothing, and a `count` of at least
	/// the chains still kept crosses out all of them. Drops accumulate across
	/// calls, and their order does not change which results are kept.
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

	/// Cross out the highest `count` of the dice that are still kept, counting
	/// each [chain](Self::chains) as one die, worth its total, and crossing out
	/// every kept die of each chain that it crosses out. Of chains of equal
	/// totals, cross out first the greatest by its [contents](Self::chains),
	/// and of identical chains, the one whose earliest kept die was rolled
	/// first. A `count` of zero or less crosses out nothing, and a `count` of
	/// at least the chains still kept crosses out all of them. Drops accumulate
	/// across calls, and their order does not change which results are kept.
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
	/// the rest of them, highest first, and counting each
	/// [chain](Self::chains) as one die, worth its total, as
	/// [`drop_highest`](Self::drop_highest) does. A `count` of zero or less
	/// keeps nothing. Unlike a drop, a keep counts from the chains that are
	/// still kept, so it does not commute with drops or other keeps.
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
	/// the rest of them, lowest first, and counting each [chain](Self::chains)
	/// as one die, worth its total, as [`drop_lowest`](Self::drop_lowest)
	/// does. A `count` of zero or less keeps nothing. Unlike a drop, a keep
	/// counts from the chains that are still kept, so it does not commute with
	/// drops or other keeps.
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
	/// die, append the replacement, [linked](Origin::Replaces) to the die that
	/// it replaces, and repeat with the replacement while it matches and the
	/// limit allows. Every die of a [chain](Self::chains) is rerolled alone,
	/// except that the dice of a chain that a [saturated](Origin::Saturates)
	/// die decides are left alone, since the endless chain for which the
	/// saturated die stands has no finite die to reroll. A `limit` of zero or
	/// less rerolls nothing. If the
	/// limit is `None` and the criterion matches every face of the die,
	/// then roll nothing, but cross out every matching die, so that each
	/// contributes zero. A die without faces, or an empty range, shows only
	/// the face `0`.
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
	/// # use xdy::{Criterion, Die, Origin, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 3, faces: 4 },
	///     dice: [2, 1, 4].into_iter().map(Die::kept).collect()
	/// };
	/// let rolled = rolling_record.reroll(&mut rng(), &Criterion::Within(1..=1), Some(1));
	/// assert_eq!(rolled, 1);
	/// assert_eq!(rolling_record.dice.len(), 4);
	/// assert_eq!(rolling_record.dice[3].origin, Origin::Replaces(1));
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
		let saturated = self.saturated();
		let matching = self
			.dice
			.iter()
			.enumerate()
			.filter(|&(index, die)| {
				die.is_kept()
					&& !saturated[index]
					&& criterion.matches(die.value)
			})
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

	/// Explode the dice that are still kept and that the criterion matches,
	/// each up to `limit` times, or until it no longer matches if `limit` is
	/// `None`. Consider the dice in the order rolled, and finish each before
	/// beginning the next: roll an extension of a matching die from the same
	/// die, append the extension, [linked](Origin::Extends) to the die that it
	/// extends, which stays kept, and repeat with the extension while it
	/// matches and the limit allows. Every die stays kept, so the sum and every
	/// later drop and keep count each [chain](Self::chains) as one die, worth
	/// its total.
	///
	/// A chain grows only from its end, so skip every die that another already
	/// extends; a replacement takes the place of the die that it replaces in
	/// its chain, so a die counts as extended if some die extends it or any die
	/// that it replaces. Skip, too, every die of a chain that a
	/// [saturated](Origin::Saturates) die decides, whose endless chain has no
	/// end from which to grow. A `limit` of zero or less explodes nothing.
	///
	/// If the limit is `None` and the criterion matches every face of the die,
	/// then no chain could ever end, so roll nothing, but append to every
	/// matching die one [saturated](Die::saturating) die, which stands for its
	/// endless chain, and which is worth `i32::MAX` or `i32::MIN` by the sign
	/// of the mean face value, or `0` if that mean is `0`. A die without faces,
	/// or an empty range, shows only the face `0`.
	///
	/// # Type parameters
	/// - `R`: The type of the pseudo-random number generator.
	///
	/// # Parameters
	/// - `rng`: The pseudo-random number generator.
	/// - `criterion`: The set of results to explode.
	/// - `limit`: The number of times that each die may be exploded, or `None`
	///   for no limit.
	///
	/// # Returns
	/// The number of dice rolled, one for each extension.
	///
	/// # Notes
	/// An unlimited explode that does not saturate terminates with probability
	/// one, but the number of dice that it rolls is unbounded, as for an
	/// unlimited [reroll](Self::reroll). The [evaluator](Evaluator)
	/// [meters](Evaluator::evaluate_metered) every extension as a die, so that
	/// a budget bounds them. A saturated die costs nothing, since nothing
	/// rolls it.
	///
	/// # Examples
	/// Explode the `6` of `[6, 3]` once, which appends its extension, so that
	/// the chain of the `6` is worth the `6` and its extension together:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{Criterion, Die, Origin, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 2, faces: 6 },
	///     dice: [6, 3].into_iter().map(Die::kept).collect()
	/// };
	/// let rolled = rolling_record.explode(&mut rng(), &Criterion::Within(6..=6), Some(1));
	/// assert_eq!(rolled, 1);
	/// assert_eq!(rolling_record.dice.len(), 3);
	/// assert_eq!(rolling_record.dice[2].origin, Origin::Extends(0));
	/// assert_eq!(rolling_record.chains(), vec![vec![0, 2], vec![1]]);
	/// let extension = rolling_record.dice[2].value;
	/// assert_eq!(rolling_record.sum(), 6 + extension + 3);
	/// ```
	///
	/// Explode every die of `2D6` without limit when it shows at least `1`,
	/// which every face does, so each chain saturates:
	///
	/// ```rust
	/// # use rand::{Rng, rng};
	/// # use xdy::{Criterion, RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut rng(), 2, 6);
	/// let rolled = rolling_record.explode(&mut rng(), &Criterion::Within(1..=6), None);
	/// assert_eq!(rolled, 0);
	/// assert!(rolling_record.to_string().ends_with(", ∞, ∞]"));
	/// assert_eq!(rolling_record.sum(), i32::MAX);
	/// ```
	pub fn explode<R>(
		&mut self,
		rng: &mut R,
		criterion: &Criterion,
		limit: Option<i32>
	) -> u64
	where
		R: Rng + ?Sized
	{
		self.explode_within(rng, criterion, limit, u64::MAX)
			.expect("no explode rolls `u64::MAX` dice")
	}

	/// Explode the dice that are still kept and that the criterion matches, as
	/// [`explode`](Self::explode) does, but roll no more than `budget` dice.
	///
	/// # Type parameters
	/// - `R`: The type of the pseudo-random number generator.
	///
	/// # Parameters
	/// - `rng`: The pseudo-random number generator.
	/// - `criterion`: The set of results to explode.
	/// - `limit`: The number of times that each die may be exploded, or `None`
	///   for no limit.
	/// - `budget`: The greatest number of dice that the explode may roll.
	///
	/// # Returns
	/// The number of dice rolled, or `None` if the explode needed to roll more
	/// than `budget` dice, in which case it rolled exactly `budget` dice and
	/// then stopped, drawing nothing more from the pRNG.
	pub(crate) fn explode_within<R>(
		&mut self,
		rng: &mut R,
		criterion: &Criterion,
		limit: Option<i32>,
		budget: u64
	) -> Option<u64>
	where
		R: Rng + ?Sized
	{
		let saturated = self.saturated();
		let extended = self.extended();
		let matching = self
			.dice
			.iter()
			.enumerate()
			.filter(|&(index, die)| {
				die.is_kept()
					&& !saturated[index]
					&& !extended[index]
					&& criterion.matches(die.value)
			})
			.map(|(index, _)| index)
			.collect::<Vec<_>>();
		if limit.is_none() && self.kind.faces_within(criterion)
		{
			// Every face matches, so no chain could ever end. Roll nothing,
			// but saturate every matching chain.
			let value = self.kind.saturated_value();
			for index in matching
			{
				self.dice.push(Die::saturating(value, index));
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
				let value = self.kind.roll_die(rng);
				self.dice.push(Die::extending(value, current));
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

	/// Answer the chains of the record. Every die descends, by its
	/// [origin](crate::Die::origin), from a die that the roll that filled the
	/// record rolled, and the dice that descend from the same one form a chain.
	/// A replacement joins the chain of the die that it replaces, and a die
	/// that extends another joins the chain of the die that it extends, so a
	/// chain holds at most one kept die unless some die extends another. A link
	/// that names no earlier die begins a chain of its own.
	///
	/// The sum and every drop and keep count a chain still kept as one die,
	/// worth its total: the value of its earliest kept
	/// [saturated](Origin::Saturates) die if it has one, or otherwise the sum
	/// of its kept dice, folded in ascending order.
	///
	/// Drops and keeps order the chains still kept by their totals, and chains
	/// of equal totals by their _contents_: their kept dice in ascending order,
	/// compared lexicographically, and then the value of the die at their end,
	/// the last kept die that no die extends. Only identical chains are then
	/// ordered by the order in which their earliest kept dice were rolled.
	/// Without explosions, every chain still kept holds one kept die, so chains
	/// of equal totals are identical.
	///
	/// # Returns
	/// The indices of the dice of each chain, whatever their fates, in the
	/// order rolled, and the chains in the order in which their first dice
	/// were rolled.
	///
	/// # Examples
	/// A `6` extended by a `2`, beside a `4`, forms two chains, worth `8` and
	/// `4`, so dropping the lowest die crosses out the `4`, and dropping the
	/// highest crosses out the `6` and the `2`:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 2, faces: 6 },
	///     dice: vec![Die::kept(6), Die::kept(4), Die::extending(2, 0)]
	/// };
	/// assert_eq!(rolling_record.chains(), vec![vec![0, 2], vec![1]]);
	/// assert_eq!(rolling_record.sum(), 12);
	///
	/// let mut lowest = rolling_record.clone();
	/// lowest.drop_lowest(1);
	/// assert_eq!(lowest.to_string(), "2D6: [6, ~4, 2]");
	/// assert_eq!(lowest.sum(), 8);
	///
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.to_string(), "2D6: [~6, 4, ~2]");
	/// assert_eq!(rolling_record.sum(), 4);
	/// ```
	///
	/// A `5` extended by a `2`, beside a `6` extended by a `1`, forms two
	/// chains, both worth `7`, whose kept dice in ascending order are `[2, 5]`
	/// and `[1, 6]`, so the chain of the `6` is the lesser, though the `5` was
	/// rolled first:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Standard { count: 2, faces: 6 },
	///     dice: vec![
	///         Die::kept(5),
	///         Die::kept(6),
	///         Die::extending(2, 0),
	///         Die::extending(1, 1),
	///     ]
	/// };
	///
	/// let mut lowest = rolling_record.clone();
	/// lowest.drop_lowest(1);
	/// assert_eq!(lowest.to_string(), "2D6: [5, ~6, 2, ~1]");
	///
	/// let mut highest = rolling_record;
	/// highest.drop_highest(1);
	/// assert_eq!(highest.to_string(), "2D6: [~5, 6, ~2, 1]");
	/// ```
	pub fn chains(&self) -> Vec<Vec<usize>>
	{
		let roots = self.roots();
		// The chain of each die that begins one, by the index of the die.
		let mut slots = vec![0usize; self.dice.len()];
		let mut chains = Vec::<Vec<usize>>::new();
		for (index, root) in roots.into_iter().enumerate()
		{
			match root == index
			{
				true =>
				{
					slots[index] = chains.len();
					chains.push(vec![index]);
				},
				false => chains[slots[root]].push(index)
			}
		}
		chains
	}

	/// Compute the sum of the record: the sum of the totals of the
	/// [chains](Self::chains) still kept, which, unless some die extends
	/// another, are the [kept](Self::kept) results. If every die is crossed
	/// out, then nothing is kept and the sum is `0`. Saturate the sum on
	/// overflow. Consecutive calls are idempotent.
	///
	/// # Returns
	/// The sum of the results.
	///
	/// # Notes
	/// The sum folds the totals in ascending order, not in the order rolled,
	/// and so does the total of each chain fold its kept dice. Saturating
	/// addition is not associative, so the order matters when the results have
	/// mixed signs and the sum saturates: `i32::MAX`, `1`, and `-1` sum to
	/// `i32::MAX` in ascending order, but to `i32::MAX - 1` in the order given.
	/// Folding in ascending order makes the sum depend only on the multiset of
	/// totals, as the forward pass and the Lean specification require.
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
	///
	/// A saturated die decides the total of its chain, whatever the other dice
	/// of the chain show:
	///
	/// ```rust
	/// # use xdy::{Die, RollingRecord, RollingRecordKind};
	/// let mut rolling_record = RollingRecord {
	///     kind: RollingRecordKind::Custom { count: 1, faces: vec![-1, 1] },
	///     dice: vec![Die::kept(1), Die::saturating(0, 0)]
	/// };
	/// assert_eq!(rolling_record.sum(), 0);
	/// ```
	pub fn sum(&mut self) -> i32
	{
		let mut totals = match self.is_unlinked()
		{
			// Every die begins a chain of its own, so the totals are the kept
			// results.
			true => self.kept(),
			false => self
				.kept_chains()
				.1
				.into_iter()
				.map(|chain| chain.total)
				.collect::<Vec<_>>()
		};
		totals.sort_unstable();
		totals.into_iter().fold(0i32, |a, b| a.saturating_add(b))
	}

	/// Answer the number of [chains](Self::chains) that are still kept.
	///
	/// # Returns
	/// The number of kept chains, saturated to `i32::MAX`.
	fn kept_count(&self) -> i32
	{
		i32::try_from(self.kept_chains().1.len()).unwrap_or(i32::MAX)
	}

	/// Answer whether the roll that filled the record rolled every die of it,
	/// so that every die begins a [chain](Self::chains) of its own, as it does
	/// unless a reroll or explosion has added dice.
	///
	/// # Returns
	/// `true` if no die links to another, `false` otherwise.
	fn is_unlinked(&self) -> bool
	{
		self.dice.iter().all(|die| die.origin == Origin::Rolled)
	}

	/// Answer, for each die, whether a kept [saturated](Origin::Saturates) die
	/// decides its [chain](Self::chains).
	///
	/// # Returns
	/// Whether each die belongs to a saturated chain, in the order rolled.
	fn saturated(&self) -> Vec<bool>
	{
		if self.is_unlinked()
		{
			// Every die begins a chain of its own, and none saturates.
			return vec![false; self.dice.len()]
		}
		let roots = self.roots();
		let mut saturated = vec![false; self.dice.len()];
		for (die, &root) in self.dice.iter().zip(&roots)
		{
			if die.is_kept() && matches!(die.origin, Origin::Saturates(_))
			{
				saturated[root] = true;
			}
		}
		roots.into_iter().map(|root| saturated[root]).collect()
	}

	/// Answer, for each die, whether some die extends it or any die that it
	/// replaces, so that an explosion may not extend it. A replacement takes
	/// the place of the die that it replaces in its [chain](Self::chains), so
	/// only the die at the end of the chain may grow it.
	///
	/// # Returns
	/// Whether each die is extended, in the order rolled.
	fn extended(&self) -> Vec<bool>
	{
		// The index of the die whose place in its chain each die takes: its
		// own, unless it replaces an earlier die.
		let mut places = Vec::with_capacity(self.dice.len());
		for (index, die) in self.dice.iter().enumerate()
		{
			let place = match die.origin
			{
				Origin::Replaces(link) if link < index => places[link],
				_ => index
			};
			places.push(place);
		}
		let mut extended = vec![false; self.dice.len()];
		for (index, die) in self.dice.iter().enumerate()
		{
			match die.origin
			{
				Origin::Extends(link) | Origin::Saturates(link)
					if link < index =>
				{
					extended[places[link]] = true;
				},
				_ =>
				{}
			}
		}
		places.into_iter().map(|place| extended[place]).collect()
	}

	/// Answer, for each die, the index of the die that begins its
	/// [chain](Self::chains). A link that names no earlier die begins a chain
	/// of its own.
	///
	/// # Returns
	/// The index of the first die of the chain of each die, in the order
	/// rolled.
	fn roots(&self) -> Vec<usize>
	{
		let mut roots = Vec::with_capacity(self.dice.len());
		for (index, die) in self.dice.iter().enumerate()
		{
			let root = match die.origin.link()
			{
				Some(link) if link < index => roots[link],
				_ => index
			};
			roots.push(root);
		}
		roots
	}

	/// Answer the [chains](Self::chains) that are still kept, i.e., that hold
	/// a kept die.
	///
	/// # Returns
	/// The index of the first die of the chain of each die, in the order
	/// rolled, and the chains still kept, in the order in which their first
	/// dice were rolled.
	fn kept_chains(&self) -> (Vec<usize>, Vec<KeptChain>)
	{
		let roots = self.roots();
		if self.is_unlinked()
		{
			// Every die begins a chain of its own, so skip the grouping.
			let chains = self
				.dice
				.iter()
				.enumerate()
				.filter(|(_, die)| die.is_kept())
				.map(|(index, die)| KeptChain {
					root: index,
					earliest: index,
					total: die.value
				})
				.collect();
			return (roots, chains)
		}
		// Order the kept dice by chain, and then by value, so that each chain
		// folds its dice in ascending order.
		let mut kept = self
			.dice
			.iter()
			.enumerate()
			.filter(|(_, die)| die.is_kept())
			.map(|(index, die)| (roots[index], die.value, index, die.origin))
			.collect::<Vec<_>>();
		kept.sort_unstable_by_key(|&(root, value, index, _)| {
			(root, value, index)
		});
		let chains = kept
			.chunk_by(|a, b| a.0 == b.0)
			.map(|dice| {
				let saturated = dice
					.iter()
					.filter(|(.., origin)| {
						matches!(origin, Origin::Saturates(_))
					})
					.min_by_key(|&&(_, _, index, _)| index);
				KeptChain {
					root: dice[0].0,
					earliest: dice
						.iter()
						.map(|&(_, _, index, _)| index)
						.min()
						.expect("a chunk is never empty"),
					total: match saturated
					{
						Some(&(_, value, ..)) => value,
						None => dice
							.iter()
							.fold(0i32, |a, &(_, b, ..)| a.saturating_add(b))
					}
				}
			})
			.collect();
		(roots, chains)
	}

	/// Answer the [contents](Self::chains) of every chain, by which drops and
	/// keeps order chains of equal totals.
	///
	/// # Parameters
	/// - `roots`: The index of the first die of the chain of each die, in the
	///   order rolled.
	///
	/// # Returns
	/// The contents of the chain that each die begins, by the index of the
	/// die, which are empty for a die that begins no chain, or a chain that is
	/// no longer kept.
	fn contents(&self, roots: &[usize]) -> Vec<Contents>
	{
		let extended = self.extended();
		let mut contents = vec![Contents::default(); self.dice.len()];
		for (index, die) in self.dice.iter().enumerate()
		{
			if die.is_kept()
			{
				let chain = &mut contents[roots[index]];
				chain.dice.push(die.value);
				if !extended[index]
				{
					chain.end = Some(die.value);
				}
			}
		}
		for chain in &mut contents
		{
			chain.dice.sort_unstable();
		}
		contents
	}

	/// Cross out the first `count` of the [chains](Self::chains) that are
	/// still kept, ordered by the specified key of their totals and contents,
	/// and then by the order in which their earliest kept dice were rolled, so
	/// that of chains with equal keys the one whose earliest kept die was
	/// rolled first is crossed out first. Crossing out a chain crosses out
	/// every kept die of it. A `count` of zero or less crosses out nothing.
	///
	/// # Type parameters
	/// - `K`: The type of the key.
	///
	/// # Parameters
	/// - `count`: The number of chains to cross out.
	/// - `key`: The key of a total and contents: the pair itself to cross out
	///   the lowest chains, or its [reverse](Reverse) to cross out the highest.
	fn cross_out<K>(&mut self, count: i32, key: impl Fn((i32, Contents)) -> K)
	where
		K: Ord
	{
		let count = usize::try_from(count).unwrap_or(0);
		if count == 0
		{
			return
		}
		let (roots, chains) = self.kept_chains();
		let mut chains = match self.is_unlinked()
		{
			// Every chain holds one kept die, so chains of equal totals are
			// identical, and their contents need not be compared.
			true => chains
				.into_iter()
				.map(|chain| (Contents::default(), chain))
				.collect::<Vec<_>>(),
			false =>
			{
				let mut contents = self.contents(&roots);
				chains
					.into_iter()
					.map(|chain| (mem::take(&mut contents[chain.root]), chain))
					.collect()
			}
		};
		chains.sort_by_cached_key(|(contents, chain)| {
			(key((chain.total, contents.clone())), chain.earliest)
		});
		let mut crossed = vec![false; self.dice.len()];
		for (_, chain) in chains.iter().take(count)
		{
			crossed[chain.root] = true;
		}
		for (die, root) in self.dice.iter_mut().zip(roots)
		{
			if die.is_kept() && crossed[root]
			{
				die.fate = Fate::Dropped;
			}
		}
	}
}

/// A [chain](RollingRecord::chains) of a [rolling record](RollingRecord) that
/// is still kept.
struct KeptChain
{
	/// The index of the first die of the chain.
	root: usize,

	/// The index of the earliest kept die of the chain.
	earliest: usize,

	/// The total of the chain: the value of its earliest kept saturated die if
	/// it has one, or otherwise the sum of its kept dice, folded in ascending
	/// order.
	total: i32
}

/// The [contents](RollingRecord::chains) of a chain still kept, by which
/// drops and keeps order chains of equal totals: its kept dice in ascending
/// order, compared lexicographically, and then the value of the die at its
/// end, the last kept die that no die extends.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Contents
{
	/// The values of the kept dice of the chain, in ascending order.
	dice: Vec<i32>,

	/// The value of the last kept die of the chain that no die extends, if
	/// any.
	end: Option<i32>
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

	/// Answer the value of a [saturated](Origin::Saturates) die of the kind,
	/// which stands for an endless chain of its dice: `i32::MAX` or `i32::MIN`
	/// by the sign of the mean face value, or `0` if that mean is `0`. A die
	/// without faces, or an empty range, shows only the face `0`.
	///
	/// # Returns
	/// The saturated value. An uninitialized record has no die, so its value
	/// is `0`.
	pub(crate) fn saturated_value(&self) -> i32
	{
		// The sign of the sum of the faces is the sign of their mean.
		let sum = match self
		{
			RollingRecordKind::Range { start, end } if end < start => 0,
			RollingRecordKind::Range { start, end } =>
			{
				*start as i128 + *end as i128
			},
			RollingRecordKind::Standard { faces, .. } =>
			{
				(*faces).max(0) as i128
			},
			RollingRecordKind::Custom { faces, .. } =>
			{
				faces.iter().map(|&face| face as i128).sum()
			},
			RollingRecordKind::Uninitialized => 0
		};
		match sum.signum()
		{
			1 => i32::MAX,
			-1 => i32::MIN,
			_ => 0
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

/// The set of results that a [reroll](RollingRecord::reroll) rerolls, or that
/// an [explode](RollingRecord::explode) explodes: a
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
/// rolling record passes the number of [chains](RollingRecord::chains) that it
/// still keeps, each of which counts as one result, and no drops, since it has
/// already crossed out the dice that it dropped.
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
