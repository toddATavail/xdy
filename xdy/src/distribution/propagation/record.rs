//! # Fused rolling records
//!
//! A rolling record is written by exactly one roll, trimmed by any number of
//! drops, and read by exactly one sum, so the roll, its drops, and its sum fuse
//! into a single operator: given the roll and the final drop counts, answer
//! the distribution of the sum of the kept results. The operator never
//! enumerates the rolls themselves, only the sums that they reach.
//!
//! Assembled instructions may read a record more than once, as when two sums
//! read it, so that both depend on the same dice. The forward pass then
//! conditions on the record's [outcomes](Roll::outcomes), the multisets of its
//! results, each of which it fixes in its own world as a roll of [fixed
//! results](Roll::Results), which it drops and sums there.
//!
//! A reroll changes what the record holds, but never what the operator needs.
//! Before any drop, its dice are independent and alike, so a reroll changes
//! only the law of each die, and the roll becomes one of [custom
//! dice](Roll::Custom) whose faces weigh that law. After a drop, or on fixed
//! results, the dice still kept are not alike, so the reroll enumerates their
//! multisets, and the roll becomes a [pool](Roll::Pool) of the multisets that
//! the reroll leaves, each with its weight. Either way, the record keeps the
//! [die](Die) from which later rerolls draw.

use std::{
	collections::{BTreeMap, HashMap, hash_map::Entry},
	iter
};

use super::meter::{Halt, Meter, charge_of};
use crate::{Criterion, Distribution, Weight};

////////////////////////////////////////////////////////////////////////////////
//                                   Rolls.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A roll that fills a rolling record, with its operands resolved to values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Roll
{
	/// A roll of the range `[start:end]`.
	Range
	{
		/// The start of the range.
		start: i32,

		/// The end of the range.
		end: i32
	},

	/// A roll of `count` standard dice, whose faces are numbered from `1` to
	/// `faces`.
	Standard
	{
		/// The number of dice.
		count: i32,

		/// The number of faces on each die.
		faces: i32
	},

	/// A roll of `count` custom dice.
	Custom
	{
		/// The number of dice.
		count: i32,

		/// The distinct faces of each die, in ascending order, each weighed by
		/// the number of times that it appears on the die.
		faces: Vec<(i32, Weight)>
	},

	/// The results of a roll, fixed by conditioning on them: each distinct
	/// result, in ascending order, with its positive number of copies, so that
	/// many dice with few faces take little space.
	Results(Vec<(i32, i32)>),

	/// The results that a [reroll](Self::pool) leaves after a drop, or on
	/// fixed results: each multiset of them, as [fixed results](Self::Results)
	/// hold one, with the positive number of branches that reach it, in
	/// ascending order of multiset. Every multiset holds the same number of
	/// results, which `Roll.Even`, in `lean/spec/XdySpec/Mixture.lean`, names,
	/// and on which the laws of drops and keeps there rest.
	Pool(Vec<(Vec<(i32, i32)>, Weight)>)
}

/// The die from which a reroll draws the replacements of the dice of a roll,
/// with its operands resolved to values. A die without faces, or an empty
/// range, shows only the face `0`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Die
{
	/// A die whose faces are the range `[start:end]`.
	Range
	{
		/// The least face.
		start: i32,

		/// The greatest face.
		end: i32
	},

	/// A standard die of the specified number of faces, numbered from `1`.
	Standard(i32),

	/// A custom die, whose distinct faces are in ascending order, each weighed
	/// by the number of times that it appears on the die.
	Custom(Vec<(i32, Weight)>)
}

impl Die
{
	/// Answer the least and greatest faces of a die whose faces leave no gap,
	/// if it has any.
	///
	/// # Returns
	/// The least and greatest faces, or `None` for a custom die, or a die
	/// without faces.
	fn interval(&self) -> Option<(i32, i32)>
	{
		match *self
		{
			Die::Range { start, end } if start <= end => Some((start, end)),
			Die::Standard(faces) if faces >= 1 => Some((1, faces)),
			_ => None
		}
	}

	/// Answer the number of distinct faces of the die, which is its length
	/// once [listed](Self::faces).
	///
	/// # Returns
	/// The number of distinct faces, which is one for a die without faces.
	pub(super) fn distinct(&self) -> u64
	{
		match (self.interval(), self)
		{
			(Some((least, greatest)), _) =>
			{
				(greatest as i64 - least as i64 + 1) as u64
			},
			(None, Die::Custom(faces)) if !faces.is_empty() =>
			{
				faces.len() as u64
			},
			_ => 1
		}
	}

	/// Answer the number of cells that the die occupies beyond the setup that
	/// holds it: one for each distinct face of a custom die.
	///
	/// # Returns
	/// The number of cells.
	pub(super) fn cells(&self) -> u64
	{
		match self
		{
			Die::Custom(faces) => faces.len() as u64,
			Die::Range { .. } | Die::Standard(_) => 0
		}
	}

	/// Answer the number of branches of the die, as a roll counts them: one
	/// for each face, or for each time that a face appears on a custom die.
	///
	/// # Returns
	/// The number of branches, which is one for a die without faces.
	pub(super) fn branches(&self) -> Weight
	{
		match (self.interval(), self)
		{
			(Some(_), _) => Weight::from(self.distinct()),
			(None, Die::Custom(faces)) if !faces.is_empty() =>
			{
				faces.iter().map(|(_, weight)| weight).sum()
			},
			_ => Weight::ONE
		}
	}

	/// Answer the number of branches of the die that show a result that the
	/// criterion matches, without listing the faces of a range or of a
	/// standard die.
	///
	/// # Parameters
	/// - `criterion`: The criterion.
	///
	/// # Returns
	/// The number of matching branches.
	pub(super) fn matching(&self, criterion: &Criterion) -> Weight
	{
		match (self.interval(), self)
		{
			(Some((least, greatest)), _) =>
			{
				Weight::from(matches_between(criterion, least, greatest))
			},
			(None, Die::Custom(faces)) if !faces.is_empty() => faces
				.iter()
				.filter(|(face, _)| criterion.matches(*face))
				.map(|(_, weight)| weight)
				.sum(),
			_ => Weight::from(criterion.matches(0) as u8)
		}
	}

	/// Answer the distinct faces of the die, in ascending order, each weighed
	/// by the number of its branches.
	///
	/// # Returns
	/// The faces, which are the face `0` alone for a die without faces.
	pub(super) fn faces(&self) -> Vec<(i32, Weight)>
	{
		match (self.interval(), self)
		{
			(Some((least, greatest)), _) =>
			{
				(least..=greatest).map(|face| (face, Weight::ONE)).collect()
			},
			(None, Die::Custom(faces)) if !faces.is_empty() => faces.clone(),
			_ => vec![(0, Weight::ONE)]
		}
	}
}

/// Answer how many of the results from `least` to `greatest`, inclusive, the
/// criterion matches, in time independent of their number.
///
/// # Parameters
/// - `criterion`: The criterion.
/// - `least`: The least result.
/// - `greatest`: The greatest result, not less than `least`.
///
/// # Returns
/// The number of matching results.
fn matches_between(criterion: &Criterion, least: i32, greatest: i32) -> u64
{
	let width = (greatest as i64 - least as i64 + 1) as u64;
	let overlap = |range: &std::ops::RangeInclusive<i32>| {
		let start = (*range.start()).max(least) as i64;
		let end = (*range.end()).min(greatest) as i64;
		(end - start + 1).max(0) as u64
	};
	match criterion
	{
		Criterion::Within(range) => overlap(range),
		Criterion::Outside(range) => width - overlap(range),
		Criterion::Among(values) =>
		{
			let first = values.partition_point(|value| *value < least);
			let last = values.partition_point(|value| *value <= greatest);
			(last - first) as u64
		}
	}
}

/// Answer the distinct faces of a custom die, in ascending order, each weighed
/// by the number of times that it appears on the die, as a [custom
/// roll](Roll::Custom) holds them.
///
/// # Parameters
/// - `faces`: The faces of the die, in any order, possibly repeated.
///
/// # Returns
/// The distinct faces and their weights.
///
/// # Notes
/// `weightedDie_eq_customDie`, in `lean/spec/XdySpec/Mixture.lean`, proves
/// that the distinct faces, weighed so, have the law of the die.
pub(super) fn distinct_faces(faces: &[i32]) -> Vec<(i32, Weight)>
{
	let mut distinct = BTreeMap::<i32, Weight>::new();
	for &face in faces
	{
		*distinct.entry(face).or_default() += Weight::ONE;
	}
	distinct.into_iter().collect()
}

impl Roll
{
	/// Answer the number of results in the record that the roll fills, as
	/// the [evaluator](crate::Evaluator) fills it, which bounds the drops.
	///
	/// # Returns
	/// The number of results.
	///
	/// # Notes
	/// `Roll.mem_support_law`, in `lean/spec/XdySpec/Mixture.lean`, proves that
	/// every record that the roll fills holds this many results, if every
	/// multiset of a pool holds as many as the first.
	pub(super) fn results(&self) -> i32
	{
		match self
		{
			// Even an empty range yields one result, zero.
			Roll::Range { .. } => 1,
			// No dice yields no results, and dice without faces yield zeros.
			Roll::Standard { count, .. } | Roll::Custom { count, .. } =>
			{
				(*count).max(0)
			},
			Roll::Results(results) =>
			{
				results.iter().map(|(_, copies)| copies).sum()
			},
			Roll::Pool(pool) => pool[0].0.iter().map(|(_, copies)| copies).sum()
		}
	}

	/// Answer the number of dice of the roll, which it charges as steps when
	/// it fills its record, since its [total](Self::total) is the number of
	/// faces raised to it. A range, fixed results, and a pool roll none.
	///
	/// # Returns
	/// The number of dice.
	pub(super) fn dice(&self) -> u64
	{
		match self
		{
			Roll::Standard { count, .. } | Roll::Custom { count, .. } =>
			{
				(*count).max(0) as u64
			},
			Roll::Range { .. } | Roll::Results(_) | Roll::Pool(_) => 0
		}
	}

	/// Answer the number of cells that the roll occupies: one, and one more
	/// for each distinct face of a custom die, or for each distinct fixed
	/// result, or for each multiset of a pool and each of its distinct
	/// results.
	///
	/// # Returns
	/// The number of cells.
	pub(super) fn cells(&self) -> u64
	{
		1 + match self
		{
			Roll::Custom { faces, .. } => faces.len() as u64,
			Roll::Results(results) => results.len() as u64,
			Roll::Pool(pool) => pool
				.iter()
				.map(|(results, _)| 1 + results.len() as u64)
				.sum(),
			Roll::Range { .. } | Roll::Standard { .. } => 0
		}
	}

	/// Answer the number of distinct faces of one die of the roll, which is
	/// the length of its [die](Self::die).
	///
	/// # Returns
	/// The number of distinct faces, which is zero for a range, fixed
	/// results, or a pool.
	fn faces(&self) -> u64
	{
		match self
		{
			Roll::Standard { faces, .. } => (*faces).max(0) as u64,
			Roll::Custom { faces, .. } => faces.len() as u64,
			Roll::Range { .. } | Roll::Results(_) | Roll::Pool(_) => 0
		}
	}

	/// Answer the total weight of the roll, which is the number of paths
	/// through it: the width of a range, or the number of faces of a die raised
	/// to the number of dice, or the branches of the multisets of a pool. A
	/// roll without branches, like an empty range, no dice, dice without
	/// faces, or fixed results, weighs one.
	///
	/// # Returns
	/// The total weight.
	pub(super) fn total(&self) -> Weight
	{
		match self
		{
			Roll::Range { start, end } if end < start => Weight::ONE,
			Roll::Range { start, end } =>
			{
				// The widest range has more than `u32::MAX` values.
				Weight::from((*end as i64 - *start as i64 + 1) as u64)
			},
			Roll::Standard { count, faces } if *count > 0 && *faces > 0 =>
			{
				Weight::from(*faces as u32).pow(*count as u32)
			},
			Roll::Custom { count, faces }
				if *count > 0 && !faces.is_empty() =>
			{
				faces
					.iter()
					.map(|(_, weight)| weight)
					.sum::<Weight>()
					.pow(*count as u32)
			},
			Roll::Pool(pool) => pool.iter().map(|(_, weight)| weight).sum(),
			_ => Weight::ONE
		}
	}

	/// Answer the distinct faces of one die of the roll, in ascending order,
	/// each weighed by the number of times that it appears on the die.
	///
	/// # Returns
	/// The faces, which are empty for a die without faces.
	///
	/// # Panics
	/// If the roll is a range.
	fn die(&self) -> Vec<(i32, Weight)>
	{
		match self
		{
			Roll::Standard { faces, .. } =>
			{
				(1..=*faces).map(|face| (face, Weight::ONE)).collect()
			},
			Roll::Custom { faces, .. } => faces.clone(),
			Roll::Range { .. } => unreachable!("a range has no die"),
			Roll::Results(_) | Roll::Pool(_) =>
			{
				unreachable!("fixed results have no die")
			}
		}
	}

	/// Answer the die from which a reroll draws the replacements of the dice
	/// of the roll, if the roll shows it.
	///
	/// # Returns
	/// The die, or `None` for fixed results or a pool, which no longer show
	/// the die that rolled them.
	pub(super) fn replacement(&self) -> Option<Die>
	{
		match self
		{
			&Roll::Range { start, end } => Some(Die::Range { start, end }),
			&Roll::Standard { faces, .. } => Some(Die::Standard(faces)),
			Roll::Custom { faces, .. } => Some(Die::Custom(faces.clone())),
			Roll::Results(_) | Roll::Pool(_) => None
		}
	}

	/// Answer every outcome of the roll: each multiset of its results, in
	/// ascending order, weighed by the number of branches of the roll that
	/// reach it, so that the weights sum to [`total`](Self::total).
	///
	/// The multisets grow one distinct face at a time, in ascending order, as
	/// the [order statistics](order_statistics) do: placing `m` copies of the
	/// next face after `j` results weighs `C(j + m, m) · wᵐ`, where `w` is the
	/// number of times that the face appears on the die, and over all faces
	/// the binomials telescope to the multinomial coefficient.
	///
	/// # Parameters
	/// - `meter`: The meter, which the enumeration charges for each face before
	///   it grows the multisets by that face, and which the outcomes stay
	///   charged to.
	///
	/// # Returns
	/// The outcomes, each a roll of [fixed results](Roll::Results), with their
	/// weights. There are at most as many as the roll has branches.
	///
	/// # Errors
	/// [`Halt`] if the enumeration would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `Roll.tsum_law_sorted`, in `lean/spec/XdySpec/Outcomes.lean`, proves
	/// that the outcomes, over the total, are the law of the record that the
	/// roll fills, sorted, and `Roll.wellFormed_of_mem_outcomes` that each is
	/// fixed results in ascending order.
	pub(super) fn outcomes(
		&self,
		meter: &mut Meter
	) -> Result<Vec<(Roll, Weight)>, Halt>
	{
		let die = match self
		{
			// An empty range yields one result, zero.
			Roll::Range { start, end } if end < start =>
			{
				meter.charge(1, 2)?;
				return Ok(vec![(Roll::Results(vec![(0, 1)]), Weight::ONE)])
			},
			&Roll::Range { start, end } =>
			{
				let width = (end as i64 - start as i64 + 1) as u64;
				meter.charge(width, 2 * width)?;
				return Ok((start..=end)
					.map(|result| {
						(Roll::Results(vec![(result, 1)]), Weight::ONE)
					})
					.collect())
			},
			Roll::Results(_) =>
			{
				meter.charge(1, self.cells())?;
				return Ok(vec![(self.clone(), Weight::ONE)])
			},
			Roll::Pool(pool) =>
			{
				// Each multiset becomes fixed results, of as many cells.
				meter.charge(pool.len() as u64, self.cells() - 1)?;
				return Ok(pool
					.iter()
					.map(|(results, weight)| {
						(Roll::Results(results.clone()), weight.clone())
					})
					.collect())
			},
			// No dice yield no results, however many faces they have.
			_ if self.results() == 0 =>
			{
				meter.charge(1, 1)?;
				return Ok(vec![(Roll::Results(Vec::new()), Weight::ONE)])
			},
			Roll::Standard { .. } | Roll::Custom { .. } =>
			{
				let faces = self.faces();
				meter.charge(faces, faces)?;
				self.die()
			}
		};
		let n = self.results() as usize;
		if die.is_empty()
		{
			// Dice without faces yield zeros.
			meter.charge(1, 2)?;
			return Ok(vec![(Roll::Results(vec![(0, n as i32)]), Weight::ONE)])
		}
		// Each partial outcome holds its results, the number of them, and its
		// weight, and occupies one cell, and one more for each of its distinct
		// results.
		meter.charge(1, 1)?;
		let mut held = 1;
		let mut partial = vec![(Vec::<(i32, i32)>::new(), 0usize, Weight::ONE)];
		for (index, (face, weight)) in die.iter().enumerate()
		{
			let greatest = index == die.len() - 1;
			// A partial outcome of `j` results grows into `n - j + 1` others,
			// or, by the greatest face, into one, whose binomial takes `j`
			// steps. Each holds at most one more distinct result.
			let (steps, grown) = partial.iter().fold(
				(0u128, 0u128),
				|(steps, grown), (_, j, _)| match greatest
				{
					true => (steps + *j as u128 + 1, grown + 1),
					false =>
					{
						let m = (n - j + 1) as u128;
						(steps + m, grown + m)
					}
				}
			);
			let cells = charge_of(grown * (index as u128 + 2));
			meter.charge(charge_of(steps), cells)?;
			let mut next = Vec::new();
			for (results, j, w) in partial
			{
				if greatest
				{
					// The greatest face fills every position that remains, so
					// `m = n - j`, and `C(n, m) = C(n, j)`.
					let m = n - j;
					let binomial = (1..=j).fold(Weight::ONE, |binomial, i| {
						(binomial * Weight::from(n - j + i))
							.exact_div(&Weight::from(i))
					});
					let power = weight.pow(m as u32);
					let mut results = results;
					if m > 0
					{
						results.push((*face, m as i32));
					}
					next.push((results, n, w * binomial * power));
					continue
				}
				// `C(j + m, m)` and `weightᵐ`, maintained incrementally.
				let mut binomial = Weight::ONE;
				let mut power = Weight::ONE;
				for m in 0..=n - j
				{
					let mut results = results.clone();
					if m > 0
					{
						binomial = (binomial * Weight::from(j + m))
							.exact_div(&Weight::from(m));
						power *= weight;
						results.push((*face, m as i32));
					}
					next.push((results, j + m, &w * &binomial * &power));
				}
			}
			let occupied = next
				.iter()
				.map(|(results, _, _)| results.len() as u64 + 1)
				.sum::<u64>();
			meter.free(cells - occupied + held);
			held = occupied;
			partial = next;
		}
		// The outcomes stay charged, and occupy the cells of their partial
		// outcomes; the die does not.
		meter.free(die.len() as u64);
		Ok(partial
			.into_iter()
			.map(|(results, _, weight)| (Roll::Results(results), weight))
			.collect())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Sums.                                    //
////////////////////////////////////////////////////////////////////////////////

impl Roll
{
	/// Answer the distribution of the sum of the kept results of the roll,
	/// after the lowest `lowest` and the highest `highest` results are dropped.
	/// The sum is the one that
	/// [`RollingRecord::sum`](crate::RollingRecord::sum) computes, a
	/// saturating fold over the kept results in ascending order,
	/// and each outcome weighs the number of branches of the roll that reach
	/// it, so the total is always [`total`](Self::total).
	///
	/// The operator chooses its method by the shape of the roll:
	///
	/// ```mermaid
	/// flowchart TD
	///     F{"Fixed results?"}
	///     F -->|yes| FS["Point mass at<br/>the sum of the kept results"]
	///     F -->|no| A
	///     A{"Degenerate?<br/>empty range, no dice,<br/>or dice without faces"}
	///     A -->|yes| P["Point mass at 0"]
	///     A -->|no| R{"Range?"}
	///     R -->|yes| RD{"Dropped?"}
	///     RD -->|yes| RZ["Point mass at 0,<br/>weighing the width"]
	///     RD -->|no| U["Uniform over the range"]
	///     R -->|no| K{"Any dice kept?"}
	///     K -->|no| Z["Point mass at 0,<br/>weighing every roll"]
	///     K -->|yes| D{"No drops, and the fold<br/>equals one clamp?"}
	///     D -->|yes| C["Convolution power, stepped up<br/>from the greatest known power<br/>of the same die,<br/>then one clamp"]
	///     D -->|no| O["Order statistics:<br/>dynamic program over the faces"]
	/// ```
	///
	/// # Parameters
	/// - `lowest`: The number of lowest results dropped, already clamped to
	///   [`results`](Self::results), as the drops clamp it.
	/// - `highest`: The number of highest results dropped, already clamped to
	///   [`results`](Self::results), as the drops clamp it.
	/// - `powers`: The convolution powers of the dice summed so far, which this
	///   sum consults and extends.
	/// - `meter`: The meter, which the sum charges before each stage of its
	///   work, and which its distribution stays charged to.
	///
	/// # Returns
	/// The distribution of the sum.
	///
	/// # Errors
	/// [`Halt`] if the sum would exceed the budget or is cancelled.
	///
	/// # Notes
	/// The sum charges no steps for the [total](Self::total) of a roll whose
	/// dice it drops, since the roll charged its dice when it filled its
	/// record.
	///
	/// `lean/spec/XdySpec/Mixture.lean` proves the law of each branch:
	/// `Setup.map_sum_law_fixed` for fixed results,
	/// `Setup.map_sum_law_range_empty`, `Setup.map_sum_law_range_dropped` and
	/// `Setup.map_sum_law_range` for ranges, and
	/// `Setup.map_sum_law_standard_eq_zero` and
	/// `Setup.map_sum_law_custom_eq_zero` for degenerate dice and dice all
	/// dropped. `rollDice_saturating_sum`, in
	/// `lean/spec/XdySpec/Convolution.lean`, proves the convolution power, and
	/// `Setup.map_sum_law_standard_order` and
	/// `Setup.map_sum_law_custom_order`, in
	/// `lean/spec/XdySpec/OrderStatistics.lean`, the order statistics.
	#[cfg_attr(doc, aquamarine::aquamarine)]
	pub(super) fn sum(
		&self,
		lowest: i32,
		highest: i32,
		powers: &mut Powers,
		meter: &mut Meter
	) -> Result<Distribution, Halt>
	{
		match self
		{
			Roll::Results(results) =>
			{
				meter.charge(results.len() as u64, 1)?;
				Ok(Distribution::point(kept_sum(results, lowest, highest)))
			},
			Roll::Pool(pool) =>
			{
				// Each multiset contributes one sum, a step for each of its
				// distinct results.
				let steps = pool
					.iter()
					.map(|(results, _)| results.len() as u64)
					.sum::<u64>();
				let len = pool.len() as u64;
				meter.charge(steps, len)?;
				let sum = Distribution::from_weights(pool.iter().map(
					|(results, weight)| {
						(kept_sum(results, lowest, highest), weight.clone())
					}
				))
				.expect("a pool has multisets");
				meter.free(len - sum.len() as u64);
				Ok(sum)
			},
			Roll::Range { start, end } if end < start =>
			{
				meter.charge(1, 1)?;
				Ok(Distribution::point(0))
			},
			Roll::Range { .. } if lowest > 0 || highest > 0 =>
			{
				meter.charge(1, 1)?;
				Ok(Distribution::from_weights([(0, self.total())])
					.expect("a roll has a nonzero total"))
			},
			&Roll::Range { start, end } =>
			{
				let width = (end as i64 - start as i64 + 1) as u64;
				meter.charge(width, width)?;
				Ok(Distribution::from_weights(
					(start..=end).map(|outcome| (outcome, Weight::ONE))
				)
				.expect("a range that is not empty has outcomes"))
			},
			Roll::Standard { count, .. } | Roll::Custom { count, .. } =>
			{
				let count = *count;
				let faces = self.faces();
				if count <= 0 || faces == 0
				{
					// No dice, or dice without faces, which roll only zeros.
					meter.charge(1, 1)?;
					return Ok(Distribution::point(0))
				}
				if lowest as i64 + highest as i64 >= count as i64
				{
					// Every die is dropped, so every roll sums to zero.
					meter.charge(1, 1)?;
					return Ok(Distribution::from_weights([(0, self.total())])
						.expect("a roll has a nonzero total"))
				}
				meter.charge(faces, faces)?;
				let die = self.die();
				if lowest == 0 && highest == 0 && folds_to_clamp(&die, count)
				{
					// The wide die replaces the die.
					let die = die
						.into_iter()
						.map(|(face, weight)| (face as i64, weight))
						.collect::<Vec<_>>();
					clamp(powers.power(die, count as u32, meter)?, meter)
				}
				else
				{
					let sum =
						order_statistics(&die, count, lowest, highest, meter)?;
					meter.free(faces);
					Ok(sum)
				}
			}
		}
	}
}

/// Answer the saturating sum of the kept results of a roll of fixed results,
/// in ascending order, as [`RollingRecord::sum`](crate::RollingRecord::sum)
/// folds them. Adding copies of one result saturates at most once, so the
/// adds of each distinct result are one clamp, as in the [order
/// statistics](order_statistics).
///
/// # Parameters
/// - `results`: The distinct results, in ascending order, each with its
///   positive number of copies.
/// - `lowest`: The number of lowest results dropped, already clamped.
/// - `highest`: The number of highest results dropped, already clamped.
///
/// # Returns
/// The sum.
///
/// # Notes
/// `Record.sum_fixed`, in `lean/spec/XdySpec/Mixture.lean`, proves that this
/// is the saturating fold of the kept results, one at a time.
fn kept_sum(results: &[(i32, i32)], lowest: i32, highest: i32) -> i32
{
	let count = results
		.iter()
		.map(|(_, copies)| *copies as i64)
		.sum::<i64>();
	let kept = lowest as i64..count - highest as i64;
	let mut j = 0i64;
	let mut sum = 0i32;
	for &(result, copies) in results
	{
		let placed = kept.start.max(j)..kept.end.min(j + copies as i64);
		let added = (placed.end - placed.start).max(0) * result as i64;
		sum =
			(sum as i64 + added).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
		j += copies as i64;
	}
	sum
}

/// Answer whether the saturating fold of every roll of the given dice, in
/// ascending order, equals the true sum of the roll clamped once to `i32`.
///
/// The fold adds the negative faces first, then the positive faces. While it
/// adds faces of one sign, its partial sums move in one direction, so it
/// saturates at most once, and saturating then equals clamping at the end. It
/// can only differ from one clamp of the true sum if the negative faces
/// saturate at [`i32::MIN`] and the positive faces then climb back from
/// there. That cannot happen if the faces all share a sign, or if even `count`
/// copies of the least face stay within `i32`.
///
/// # Parameters
/// - `die`: The distinct faces of one die, in ascending order, which are not
///   empty.
/// - `count`: The number of dice, which is positive.
///
/// # Returns
/// `true` if the fold equals one clamp of the true sum, `false` otherwise.
///
/// # Notes
/// `Record.sum_eq_clamp`, in `lean/Xdy/Saturation.lean`, proves this test
/// sound, and `rollDice_saturating_sum`, in
/// `lean/spec/XdySpec/Convolution.lean`, that a roll without drops that passes
/// it has the law of the die's convolution power, clamped once.
fn folds_to_clamp(die: &[(i32, Weight)], count: i32) -> bool
{
	let least = die[0].0 as i64;
	let greatest = die[die.len() - 1].0 as i64;
	least >= 0 || greatest <= 0 || count as i64 * least >= i32::MIN as i64
}

/// Clamp every outcome of the given wide distribution to `i32`, merging the
/// weights of outcomes that clamp to the same value.
///
/// # Parameters
/// - `wide`: The outcomes and their weights, as [`convolve`] answers them,
///   charged to the meter.
/// - `meter`: The meter, to which the clamped distribution stays charged in
///   place of the wide one.
///
/// # Returns
/// The clamped distribution.
///
/// # Errors
/// [`Halt`] if clamping would exceed the budget or is cancelled.
pub(super) fn clamp(
	wide: Vec<(i64, Weight)>,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	let len = wide.len() as u64;
	meter.charge(len, len)?;
	let clamped = Distribution::from_weights(wide.into_iter().map(
		|(outcome, weight)| {
			(
				outcome.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
				weight
			)
		}
	))
	.expect("a convolution of distributions has outcomes");
	meter.free(2 * len - clamped.len() as u64);
	Ok(clamped)
}

////////////////////////////////////////////////////////////////////////////////
//                                  Rerolls.                                  //
////////////////////////////////////////////////////////////////////////////////

/// What a reroll does to the dice still kept of a roll, given the die from
/// which it draws their replacements.
#[derive(Debug)]
pub(super) enum Effect<'c>
{
	/// Nothing, since no face matches, the limit is not positive, or the die
	/// has one face and the limit is finite, so that every die shows what it
	/// showed.
	Nothing,

	/// The reroll crosses out every die still kept, rolling nothing, since it
	/// has no limit and matches every face.
	CrossOut,

	/// The reroll may change the dice still kept.
	Redraw(Redraw<'c>)
}

/// A reroll that may change the dice still kept of a roll, which counts its
/// branches as though each die still kept rolled in advance every reroll that
/// it might need, so that every path is equally likely, as
/// [`Bounds::count`](crate::Bounds::count) counts them.
///
/// Let the die have `F` branches, of which the criterion matches `m`, and let
/// `w(v)` be the branches that show `v`. A die that the criterion matched
/// finally shows `v` along `g(v)` branches, of `G` in all:
///
/// ```text
/// limited to n times: g(v) = w(v) · ([v ∉ R] · Σ_{k<n−1} mᵏ · Fⁿ⁻¹⁻ᵏ + mⁿ⁻¹),
///                     G = Fⁿ
/// unlimited:          g(v) = w(v) · [v ∉ R],
///                     G = F − m
/// ```
///
/// A die limited to `n` rerolls that the criterion matched rolls a
/// replacement, and then behaves as a die limited to `n − 1`: its replacement
/// first shows a face that does not match after `k` more rerolls, along
/// `mᵏ · w(v) · Fⁿ⁻¹⁻ᵏ` branches, padded by the rerolls that it never needed,
/// or shows any face after the last, along `mⁿ⁻¹ · w(v)`. A die without limit
/// finally shows a face that does not match, each as likely as its branches.
/// A die that the criterion did not match keeps its result, and every one of
/// the `G` branches of its rerolls reaches it.
#[derive(Debug)]
pub(super) struct Redraw<'c>
{
	/// The results that the reroll rerolls.
	criterion: &'c Criterion,

	/// The branches that the reroll adds to each die still kept, `G`.
	padding: Weight,

	/// The final results of a die that the criterion matched, each weighed by
	/// `g`, in ascending order, without zero weights.
	law: Vec<(i32, Weight)>
}

/// Resolve a reroll against the die from which it draws: decide whether it
/// may change the dice still kept, and if so, how a die that it matches
/// finally falls.
///
/// # Parameters
/// - `die`: The die.
/// - `criterion`: The results that the reroll rerolls.
/// - `limit`: The number of times that each die may be rerolled, or `None` for
///   no limit.
/// - `meter`: The meter, which the resolution charges for counting the branches
///   of the die, and for listing its faces and weighing each, and a step for
///   each reroll that a die may need, since the padding is a power of the die's
///   branches.
///
/// # Returns
/// The effect of the reroll.
///
/// # Errors
/// [`Halt`] if the resolution would exceed the budget or is cancelled.
///
/// # Notes
/// `Effect.resolve`, in `lean/spec/XdySpec/Reroll.lean`, mirrors the
/// resolution. `weightedDie_limitedLaw` proves that the limited law weighs
/// each final result as the specification's `limitedReplacement` draws it, and
/// `weightedDie_unlimitedLaw` that the unlimited law is the die's conditional
/// law, given that it does not match; `length_expand_limitedLaw` and
/// `length_expand_unlimitedLaw` prove that each weighs the padding in all, and
/// `unmatchedPaths_mul` and `unmatchedPaths_self` that `unmatched` is the sum
/// that it closes.
pub(super) fn redraw<'c>(
	die: &Die,
	criterion: &'c Criterion,
	limit: Option<i32>,
	meter: &mut Meter
) -> Result<Effect<'c>, Halt>
{
	// Counting the branches takes a step, and one for each face of a custom
	// die.
	meter.charge(1 + die.cells(), 0)?;
	let branches = die.branches();
	let matching = die.matching(criterion);
	match limit
	{
		_ if matching.is_zero() => return Ok(Effect::Nothing),
		Some(limit) if limit <= 0 || branches == Weight::ONE =>
		{
			return Ok(Effect::Nothing)
		},
		None if matching == branches => return Ok(Effect::CrossOut),
		_ =>
		{}
	}
	// The faces are listed, and then weighed.
	let distinct = die.distinct();
	let powers = limit.map_or(1, |limit| limit as u128);
	meter.charge(
		charge_of(2 * distinct as u128 + powers),
		charge_of(2 * distinct as u128)
	)?;
	let faces = die.faces();
	let (padding, law) = match limit
	{
		Some(limit) =>
		{
			let j = limit as u32 - 1;
			let (fj, mj) = (branches.pow(j), matching.pow(j));
			// `Σ_{k<j} mᵏ · Fʲ⁻ᵏ = F · (Fʲ − mʲ) / (F − m)`, or `j · Fʲ` if
			// every face matches.
			let unmatched = match branches.checked_sub(&matching)
			{
				Some(rest) if !rest.is_zero() =>
				{
					(fj.checked_sub(&mj).expect("m ≤ F") * &branches)
						.exact_div(&rest)
				},
				_ => Weight::from(j) * &fj
			};
			let law = faces
				.into_iter()
				.map(|(face, weight)| match criterion.matches(face)
				{
					true => (face, weight * &mj),
					false => (face, weight * (&unmatched + &mj))
				})
				.collect::<Vec<_>>();
			(fj * &branches, law)
		},
		None =>
		{
			let law = faces
				.into_iter()
				.filter(|(face, _)| !criterion.matches(*face))
				.collect::<Vec<_>>();
			let padding = branches
				.checked_sub(&matching)
				.expect("the criterion matches no more than every branch");
			(padding, law)
		}
	};
	meter.free(distinct);
	Ok(Effect::Redraw(Redraw {
		criterion,
		padding,
		law
	}))
}

impl Roll
{
	/// Reroll the dice of the roll, none of which has been dropped, as the
	/// redraw directs. The dice are independent and alike, so each reroll
	/// changes only the law of each die: a die that shows `v` along `f(v)`
	/// branches afterward shows it along
	///
	/// ```text
	/// f′(v) = f(v) · [v ∉ R] · G + f(R) · g(v)
	/// ```
	///
	/// branches, where `f(R)` is the branches that show a result that the
	/// criterion matches, so that the total of each die grows by the padding
	/// `G`. The die before the reroll may differ from the die from which it
	/// draws, if an earlier reroll changed its law.
	///
	/// # Parameters
	/// - `redraw`: The redraw.
	/// - `meter`: The meter, which the rewrite charges for listing the faces of
	///   the dice and for weighing the new ones, and to which the new faces
	///   stay charged.
	///
	/// # Returns
	/// The rerolled dice, as custom dice of the same count, or one for a
	/// range.
	///
	/// # Errors
	/// [`Halt`] if the rewrite would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the roll is fixed results or a pool.
	///
	/// # Notes
	/// `Roll.rewrite`, in `lean/spec/XdySpec/Reroll.lean`, mirrors the rewrite,
	/// and `weightedDie_rewriteFaces` proves that each rewritten die falls as
	/// the die, each face that the criterion matches redrawn. The specification
	/// rerolls the kept results in ascending order rather than each die in
	/// roll order, which `bind_rollDice_bagDraws` proves leaves the same
	/// multisets of results.
	pub(super) fn rewrite(
		&self,
		redraw: &Redraw,
		meter: &mut Meter
	) -> Result<Roll, Halt>
	{
		let (count, distinct) = match self
		{
			Roll::Range { .. } => (1, self.replacement().unwrap().distinct()),
			Roll::Standard { count, faces } => (*count, (*faces).max(1) as u64),
			Roll::Custom { count, faces } => (*count, faces.len() as u64),
			Roll::Results(_) | Roll::Pool(_) =>
			{
				unreachable!("fixed results are not alike")
			}
		};
		// The faces are listed, and each, with each final result of a matched
		// die, weighed anew.
		let weighed = distinct as u128 + redraw.law.len() as u128;
		let charged = charge_of(distinct as u128 + weighed);
		meter.charge(charged, charged)?;
		let faces = match self
		{
			Roll::Custom { faces, .. } => faces.clone(),
			_ => self.replacement().expect("dice show their die").faces()
		};
		let matched = faces
			.iter()
			.filter(|(face, _)| redraw.criterion.matches(*face))
			.map(|(_, weight)| weight)
			.sum::<Weight>();
		let mut rewritten = BTreeMap::<i32, Weight>::new();
		for (face, weight) in faces
		{
			if !redraw.criterion.matches(face)
			{
				*rewritten.entry(face).or_default() += weight * &redraw.padding;
			}
		}
		if !matched.is_zero()
		{
			for (face, weight) in &redraw.law
			{
				*rewritten.entry(*face).or_default() += weight * &matched;
			}
		}
		meter.free(charged - rewritten.len() as u64);
		Ok(Roll::Custom {
			count,
			faces: rewritten.into_iter().collect()
		})
	}

	/// Reroll the dice still kept of the roll, after its lowest `lowest` and
	/// highest `highest` results are dropped, as the redraw directs, and
	/// answer the pool of the multisets of the dice that remain kept. The dice
	/// still kept are not alike, so the reroll enumerates the multisets of
	/// the roll's results, and then, for each, the multisets of the
	/// replacements of the `k` dice still kept that the criterion matches,
	/// each of which finally falls as the redraw's law directs. A multiset
	/// reached along `b` branches, whose replacements are reached along `v`,
	/// weighs `b · v · G^(K − k)` in the pool, where `K` is the number of dice
	/// still kept, so that the total of the roll grows by `Gᴷ`.
	///
	/// # Parameters
	/// - `lowest`: The number of lowest results dropped, already clamped.
	/// - `highest`: The number of highest results dropped, already clamped,
	///   such that at least one result is still kept.
	/// - `redraw`: The redraw.
	/// - `meter`: The meter, which the reroll charges for the outcomes of the
	///   roll and of the replacements, and for combining them, and to which the
	///   pool stays charged.
	///
	/// # Returns
	/// The pool, whose results are not dropped.
	///
	/// # Errors
	/// [`Halt`] if the reroll would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `Roll.rerollPool`, in `lean/spec/XdySpec/Reroll.lean`, mirrors the pool,
	/// with `survivors` and `join`, though it does not merge the multisets that
	/// coincide, and `Setup.rerolls_of_pools` proves that it has the law of the
	/// multiset that the specification's reroll leaves of the dice still kept:
	/// the survivors are the kept results that the criterion does not match
	/// (`survivors_spec`), the replacements of the `k` that it does are
	/// independent draws of the redraw's law (`bagDraws_kept`), and since every
	/// outcome weighs `Gᴷ` in all, the pool is the outcomes followed by their
	/// replacements (`weighted_flatMap`). `Roll.wellFormed_rerollPool` proves
	/// that every multiset holds the dice still kept.
	pub(super) fn pool(
		&self,
		lowest: i32,
		highest: i32,
		redraw: &Redraw,
		meter: &mut Meter
	) -> Result<Roll, Halt>
	{
		let kept = self.results() - lowest - highest;
		debug_assert!(kept > 0, "a die is still kept");
		let outcomes = self.outcomes(meter)?;
		// The paddings of the dice that the criterion did not match, by their
		// number.
		meter.charge(kept as u64 + 1, kept as u64 + 1)?;
		let paddings = iter::successors(Some(Weight::ONE), |padding| {
			Some(padding * &redraw.padding)
		})
		.take(kept as usize + 1)
		.collect::<Vec<_>>();
		let mut replacements = HashMap::<i32, Vec<(Roll, Weight)>>::new();
		let mut pool = BTreeMap::<Vec<(i32, i32)>, Weight>::new();
		for (outcome, branches) in outcomes
		{
			let Roll::Results(results) = outcome
			else
			{
				unreachable!("an outcome is fixed results")
			};
			let len = results.len() as u64;
			meter.charge(len, len)?;
			let (survivors, matched) =
				survivors(&results, lowest, highest, redraw.criterion);
			let replaced = match replacements.entry(matched)
			{
				Entry::Occupied(entry) => entry.into_mut(),
				Entry::Vacant(entry) =>
				{
					// The replacements are dice of the redraw's law.
					let faces = redraw.law.len() as u64;
					meter.charge(faces, faces + 1)?;
					let dice = Roll::Custom {
						count: matched,
						faces: redraw.law.clone()
					};
					entry.insert(dice.outcomes(meter)?)
				}
			};
			// Each multiset of the replacements joins the survivors, in a
			// multiset of the pool.
			let joined = replaced.iter().fold(0u128, |cells, (replaced, _)| {
				cells + 1 + survivors.len() as u128 + replaced.cells() as u128
			});
			let joined = charge_of(joined);
			meter.charge(joined, joined)?;
			let factor = branches * &paddings[(kept - matched) as usize];
			for (replaced, weight) in replaced.iter()
			{
				let Roll::Results(replaced) = replaced
				else
				{
					unreachable!("an outcome is fixed results")
				};
				*pool.entry(join(&survivors, replaced)).or_default() +=
					&factor * weight;
			}
		}
		Ok(Roll::Pool(pool.into_iter().collect()))
	}
}

/// Answer the results still kept of fixed results, after the lowest `lowest`
/// and the highest `highest` are dropped, that the criterion does not match,
/// and the number that it does.
///
/// # Parameters
/// - `results`: The distinct results, in ascending order, each with its
///   positive number of copies.
/// - `lowest`: The number of lowest results dropped, already clamped.
/// - `highest`: The number of highest results dropped, already clamped.
/// - `criterion`: The criterion.
///
/// # Returns
/// The distinct results still kept that the criterion does not match, in
/// ascending order, each with its positive number of copies, and the number
/// of results still kept that it matches.
fn survivors(
	results: &[(i32, i32)],
	lowest: i32,
	highest: i32,
	criterion: &Criterion
) -> (Vec<(i32, i32)>, i32)
{
	let count = results.iter().map(|(_, copies)| *copies).sum::<i32>();
	let kept = lowest..count - highest;
	let mut survivors = Vec::new();
	let mut matched = 0;
	let mut j = 0;
	for &(result, copies) in results
	{
		let placed = (kept.end.min(j + copies) - kept.start.max(j)).max(0);
		j += copies;
		match (placed, criterion.matches(result))
		{
			(0, _) =>
			{},
			(placed, true) => matched += placed,
			(placed, false) => survivors.push((result, placed))
		}
	}
	(survivors, matched)
}

/// Answer the union of two multisets of fixed results.
///
/// # Parameters
/// - `xs`: The distinct results of one, in ascending order, each with its
///   positive number of copies.
/// - `ys`: The distinct results of the other, likewise.
///
/// # Returns
/// The distinct results of the union, in ascending order, each with its
/// positive number of copies.
fn join(xs: &[(i32, i32)], ys: &[(i32, i32)]) -> Vec<(i32, i32)>
{
	let mut joined = BTreeMap::<i32, i32>::new();
	for &(result, copies) in xs.iter().chain(ys)
	{
		*joined.entry(result).or_default() += copies;
	}
	joined.into_iter().collect()
}

////////////////////////////////////////////////////////////////////////////////
//                                Convolution.                                //
////////////////////////////////////////////////////////////////////////////////

/// Convolve two distributions over wide outcomes: weigh every sum of an outcome
/// of each by the product of their weights. The sums are exact, never
/// saturated.
///
/// The sums accumulate in a dense array when its length does not exceed the
/// number of pairs, so that the array costs no more than the work, and in a
/// sorted map otherwise, as for dice whose few faces lie far apart. A dense
/// convolution whose products may exceed [`u128`] instead multiplies two
/// integers into which its operands are packed ([`convolve_packed`]).
///
/// # Parameters
/// - `xs`: The outcomes and nonzero weights of one distribution, in ascending
///   order of outcome, which are not empty.
/// - `ys`: The outcomes and nonzero weights of the other, likewise.
/// - `meter`: The meter, which the convolution charges a step for each pair and
///   a cell for each sum, twice, for its accumulator and its answer, and to
///   which the answer stays charged.
///
/// # Returns
/// The outcomes and nonzero weights of the convolution, in ascending order of
/// outcome.
///
/// # Errors
/// [`Halt`] if the convolution would exceed the budget or is cancelled.
///
/// # Panics
/// If a sum overflows `i64`, which no sum of fewer than `2³²` values of `i32`
/// does.
pub(super) fn convolve(
	xs: &[(i64, Weight)],
	ys: &[(i64, Weight)],
	meter: &mut Meter
) -> Result<Vec<(i64, Weight)>, Halt>
{
	let least = xs[0].0 + ys[0].0;
	let greatest = xs[xs.len() - 1].0 + ys[ys.len() - 1].0;
	let span = (greatest as i128 - least as i128 + 1) as u128;
	let pairs = xs.len() as u128 * ys.len() as u128;
	let sums = charge_of(span.min(pairs));
	meter.charge(charge_of(pairs), sums.saturating_mul(2))?;
	let convolution = convolve_charged(xs, ys, least, span, pairs);
	meter.free(sums.saturating_mul(2) - convolution.len() as u64);
	Ok(convolution)
}

/// Convolve two distributions over wide outcomes, as [`convolve`] does, once
/// it has charged the work.
///
/// # Parameters
/// - `xs`: The outcomes and nonzero weights of one distribution, in ascending
///   order of outcome, which are not empty.
/// - `ys`: The outcomes and nonzero weights of the other, likewise.
/// - `least`: The least sum.
/// - `span`: The number of sums from the least to the greatest.
/// - `pairs`: The number of pairs of outcomes.
///
/// # Returns
/// The outcomes and nonzero weights of the convolution, in ascending order of
/// outcome.
fn convolve_charged(
	xs: &[(i64, Weight)],
	ys: &[(i64, Weight)],
	least: i64,
	span: u128,
	pairs: u128
) -> Vec<(i64, Weight)>
{
	if span <= pairs
	{
		convolve_packed(xs, ys, least, span)
			.unwrap_or_else(|| convolve_dense(xs, ys, least, span))
	}
	else
	{
		let mut sums = BTreeMap::<i64, Weight>::new();
		for (x, wx) in xs
		{
			for (y, wy) in ys
			{
				*sums.entry(x + y).or_default() += wx * wy;
			}
		}
		sums.into_iter().collect()
	}
}

/// Convolve two distributions over wide outcomes, as [`convolve`] does, by
/// accumulating the product of every pair of weights in a dense array.
///
/// # Parameters
/// - `xs`: The outcomes and nonzero weights of one distribution, in ascending
///   order of outcome, which are not empty.
/// - `ys`: The outcomes and nonzero weights of the other, likewise.
/// - `least`: The least sum.
/// - `span`: The number of sums from the least to the greatest.
///
/// # Returns
/// The outcomes and nonzero weights of the convolution, in ascending order of
/// outcome.
pub(crate) fn convolve_dense(
	xs: &[(i64, Weight)],
	ys: &[(i64, Weight)],
	least: i64,
	span: u128
) -> Vec<(i64, Weight)>
{
	let mut sums = vec![Weight::ZERO; span as usize];
	for (x, wx) in xs
	{
		for (y, wy) in ys
		{
			sums[(x + y - least) as usize] += wx * wy;
		}
	}
	sums.into_iter()
		.zip(least..)
		.filter(|(weight, _)| !weight.is_zero())
		.map(|(weight, outcome)| (outcome, weight))
		.collect()
}

/// Convolve two distributions over wide outcomes, as [`convolve`] does, by
/// Kronecker substitution, if their products may exceed [`u128`]: read each
/// as a polynomial in `2ᴮ`, whose coefficient at the power of each outcome,
/// less the least, is its weight, so that their product is one integer, whose
/// coefficients are the weights of the sums. One multiplication of big
/// integers replaces one for every pair of outcomes, and allocates once.
///
/// ```mermaid
/// flowchart LR
///     Xs["xs"] -->|"pack, B bits<br/>per outcome"| X["X"]
///     Ys["ys"] -->|"pack"| Y["Y"]
///     X --> P["X · Y"]
///     Y --> P
///     P -->|"unpack, B bits<br/>per sum"| Sums["sums"]
/// ```
///
/// # Parameters
/// - `xs`: The outcomes and nonzero weights of one distribution, in ascending
///   order of outcome, which are not empty.
/// - `ys`: The outcomes and nonzero weights of the other, likewise.
/// - `least`: The least sum.
/// - `span`: The number of sums from the least to the greatest.
///
/// # Returns
/// The outcomes and nonzero weights of the convolution, in ascending order of
/// outcome, or [`None`] if every product of weights fits in a `u128`, whose
/// pairs multiply without allocating.
///
/// # Notes
/// Each sum's weight is the sum of at most `m = min(|xs|, |ys|)` products,
/// each less than `2^(bx + by)`, where `bx` and `by` are the most bits of a
/// weight of each, so it is less than `2ᴮ` for `B = bx + by + bits(m)`, and no
/// coefficient carries into the next. `B` is rounded up to whole digits of 32
/// bits, so that packing and unpacking copy digits.
pub(crate) fn convolve_packed(
	xs: &[(i64, Weight)],
	ys: &[(i64, Weight)],
	least: i64,
	span: u128
) -> Option<Vec<(i64, Weight)>>
{
	let bits = |ws: &[(i64, Weight)]| {
		ws.iter().map(|(_, w)| w.bits()).max().unwrap_or(0)
	};
	let (bx, by) = (bits(xs), bits(ys));
	if bx + by <= u128::BITS as u64
	{
		return None
	}
	let terms = xs.len().min(ys.len()) as u64;
	let slot = (bx + by + (u64::BITS - terms.leading_zeros()) as u64)
		.div_ceil(u32::BITS as u64) as usize;
	let pack = |ws: &[(i64, Weight)]| {
		let origin = ws[0].0;
		let mut digits =
			vec![0u32; (ws[ws.len() - 1].0 - origin + 1) as usize * slot];
		for (outcome, weight) in ws
		{
			let at = (outcome - origin) as usize * slot;
			weight.write_u32_digits(&mut digits[at..at + slot]);
		}
		Weight::from_u32_digits(&digits)
	};
	let digits = (&pack(xs) * &pack(ys)).to_u32_digits();
	Some(
		(0..span as usize)
			.zip(least..)
			.filter_map(|(k, outcome)| {
				let start = (k * slot).min(digits.len());
				let end = ((k + 1) * slot).min(digits.len());
				let weight = Weight::from_u32_digits(&digits[start..end]);
				(!weight.is_zero()).then_some((outcome, weight))
			})
			.collect()
	)
}

/// The greatest convolution power of each die computed so far, from which a
/// greater power of the same die steps up, so that the sums of `k₁ < k₂ < …`
/// dice take one convolution each when the counts are consecutive, as they
/// are for a [mixture](super::mixture) over a dynamic count, rather than
/// `O(log kᵢ)` apiece.
#[derive(Debug, Default)]
pub(super) struct Powers(HashMap<Wide, (u32, Wide)>);

/// The outcomes and nonzero weights of a distribution over wide outcomes, in
/// ascending order of outcome, as [`convolve`] answers them.
type Wide = Vec<(i64, Weight)>;

impl Powers
{
	/// Answer the sum of the given number of the given dice: from the greatest
	/// power of the die computed so far, if it does not exceed `n`, convolved
	/// with the power of the difference, or else by squaring. The answer
	/// becomes the greatest power of the die, unless a greater one is known.
	///
	/// # Parameters
	/// - `die`: The outcomes and nonzero weights of one die, in ascending order
	///   of outcome, which are not empty.
	/// - `n`: The number of dice, which is positive.
	/// - `meter`: The meter, to which the die is charged, which the powers
	///   charge for their work and for what they keep, and to which the sum
	///   stays charged in place of the die.
	///
	/// # Returns
	/// The outcomes and nonzero weights of the sum of `n` dice, in ascending
	/// order of outcome.
	///
	/// # Errors
	/// [`Halt`] if the powers would exceed the budget or are cancelled.
	///
	/// # Notes
	/// `convPow_add`, in `lean/spec/XdySpec/Convolution.lean`, proves that
	/// stepping up is sound.
	fn power(
		&mut self,
		die: Vec<(i64, Weight)>,
		n: u32,
		meter: &mut Meter
	) -> Result<Vec<(i64, Weight)>, Halt>
	{
		let faces = die.len() as u64;
		let power = match self.0.get(&die)
		{
			Some((k, power)) if *k == n =>
			{
				let len = power.len() as u64;
				meter.charge(len, len)?;
				meter.free(faces);
				return Ok(power.clone())
			},
			Some((k, power)) if *k < n =>
			{
				meter.charge(faces, faces)?;
				let step = convolution_power(die.clone(), n - k, meter)?;
				let power = convolve(power, &step, meter)?;
				meter.free(step.len() as u64);
				power
			},
			// A lesser power does not replace a greater one.
			Some(_) => return convolution_power(die, n, meter),
			None =>
			{
				meter.charge(faces, faces)?;
				convolution_power(die.clone(), n, meter)?
			}
		};
		let len = power.len() as u64;
		meter.charge(len, len)?;
		if let Some((_, old)) = self.0.insert(die, (n, power.clone()))
		{
			// The map keeps its own copy of the die, and drops this one.
			meter.free(old.len() as u64 + faces);
		}
		Ok(power)
	}
}

/// Convolve a distribution with itself the given number of times, by squaring,
/// so that `n` dice take `O(log n)` convolutions.
///
/// # Parameters
/// - `die`: The outcomes and nonzero weights of one die, in ascending order of
///   outcome, which are not empty.
/// - `n`: The number of dice, which is positive.
/// - `meter`: The meter, to which the die is charged, which each convolution
///   charges, and to which the sum stays charged in place of the die.
///
/// # Returns
/// The outcomes and nonzero weights of the sum of `n` dice, in ascending
/// order of outcome.
///
/// # Errors
/// [`Halt`] if a convolution would exceed the budget or is cancelled.
///
/// # Notes
/// `rollDice_sum`, in `lean/spec/XdySpec/Convolution.lean`, proves that the
/// convolution power is the law of the sum of `n` dice, and `convPow_add`
/// that squaring is sound.
fn convolution_power(
	die: Vec<(i64, Weight)>,
	mut n: u32,
	meter: &mut Meter
) -> Result<Vec<(i64, Weight)>, Halt>
{
	debug_assert!(n > 0, "at least one die");
	let mut power = None::<Vec<(i64, Weight)>>;
	let mut square = die;
	loop
	{
		if n & 1 == 1
		{
			power = Some(match power
			{
				None =>
				{
					let len = square.len() as u64;
					meter.charge(len, len)?;
					square.clone()
				},
				Some(power) =>
				{
					let next = convolve(&power, &square, meter)?;
					meter.free(power.len() as u64);
					next
				}
			});
		}
		n >>= 1;
		if n == 0
		{
			meter.free(square.len() as u64);
			return Ok(power.expect("at least one die"))
		}
		let next = convolve(&square, &square, meter)?;
		meter.free(square.len() as u64);
		square = next;
	}
}

////////////////////////////////////////////////////////////////////////////////
//                             Order statistics.                              //
////////////////////////////////////////////////////////////////////////////////

/// Answer the distribution of the saturating sum of the kept dice of a roll,
/// by a dynamic program over the distinct faces in ascending order.
///
/// Sorting a roll puts every copy of the least face first, then every copy of
/// the next, and so on. The state `(j, s)` says that the first `j` sorted
/// positions are filled, and that the saturating fold of the kept dice among
/// them is `s`. Placing `m` copies of the next face `v` at positions `j` to
/// `j + m - 1` keeps those that lie within `[lowest, count - highest)`, and
/// adds each to `s`; adding copies of one face saturates at most once, so the
/// adds are one clamp. The rolls that sort to the same multiset of faces are
/// its permutations, so the placement weighs `C(j + m, m) · wᵐ`, where `w` is
/// the number of times that `v` appears on the die; over all faces, the
/// binomials telescope to the multinomial coefficient.
///
/// The cost is `O(k · n² · |S|)` for `k` distinct faces, `n` dice, and `|S|`
/// distinct sums.
///
/// # Parameters
/// - `die`: The distinct faces of one die, in ascending order, each weighed by
///   the number of times that it appears on the die, which are not empty.
/// - `count`: The number of dice, which is positive.
/// - `lowest`: The number of lowest dice dropped.
/// - `highest`: The number of highest dice dropped, such that at least one die
///   is kept.
/// - `meter`: The meter, which the program charges for each face before it
///   places that face, and to which the distribution stays charged.
///
/// # Returns
/// The distribution of the sum, whose total is the number of faces raised to
/// the number of dice.
///
/// # Errors
/// [`Halt`] if the program would exceed the budget or is cancelled.
///
/// # Notes
/// `map_sum_rollDice_order`, in `lean/spec/XdySpec/OrderStatistics.lean`,
/// proves that the weights of this program, over the number of faces raised to
/// the number of dice, are the law of the sum of the kept dice.
fn order_statistics(
	die: &[(i32, Weight)],
	count: i32,
	lowest: i32,
	highest: i32,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	let n = count as usize;
	let kept = lowest as usize..n - highest as usize;
	// `states[j]` weighs each saturating sum of the kept dice among the first
	// `j` sorted positions. The states occupy a cell for each position, and
	// one for each sum.
	let slots = n as u64 + 1;
	meter.charge(slots, slots + 1)?;
	let mut held = slots + 1;
	let mut states = vec![BTreeMap::<i32, Weight>::new(); n + 1];
	states[0].insert(0, Weight::ONE);
	for (index, (face, weight)) in die.iter().enumerate()
	{
		// The greatest face fills every position that remains.
		let greatest = index == die.len() - 1;
		// Each sum after `j` positions moves to `n - j + 1` others, or, by the
		// greatest face, to one, and each placement updates a binomial.
		let (steps, moved) = states.iter().enumerate().fold(
			(slots as u128, 0u128),
			|(steps, moved), (j, sums)| {
				let placements = (n - j + 1) as u128;
				let sums = sums.len() as u128;
				let moves = match greatest
				{
					true => sums,
					false => sums * placements
				};
				(steps + placements + moves, moved + moves)
			}
		);
		let cells = charge_of(moved).saturating_add(slots);
		meter.charge(charge_of(steps), cells)?;
		let mut next = vec![BTreeMap::<i32, Weight>::new(); n + 1];
		for (j, sums) in states.iter().enumerate()
		{
			if sums.is_empty()
			{
				continue
			}
			// `C(j + m, m)` and `weightᵐ`, maintained incrementally.
			let mut binomial = Weight::ONE;
			let mut power = Weight::ONE;
			for m in 0..=n - j
			{
				if m > 0
				{
					binomial = (binomial * Weight::from(j + m))
						.exact_div(&Weight::from(m));
					power *= weight;
				}
				if greatest && m < n - j
				{
					continue
				}
				let placed = kept.start.max(j)..kept.end.min(j + m);
				let added = placed.len() as i64 * *face as i64;
				let factor = &binomial * &power;
				let target = &mut next[j + m];
				for (&sum, w) in sums
				{
					let sum = (sum as i64 + added)
						.clamp(i32::MIN as i64, i32::MAX as i64)
						as i32;
					*target.entry(sum).or_default() += w * &factor;
				}
			}
		}
		let occupied =
			slots + next.iter().map(|sums| sums.len() as u64).sum::<u64>();
		meter.free(cells - occupied + held);
		held = occupied;
		states = next;
	}
	let sums = states.pop().expect("n + 1 states");
	let len = sums.len() as u64;
	meter.charge(len, len)?;
	let distribution =
		Distribution::from_weights(sums).expect("a roll has outcomes");
	meter.free(held + len - distribution.len() as u64);
	Ok(distribution)
}
