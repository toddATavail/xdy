//! # Mixtures
//!
//! When the count, faces, or range endpoints of a roll, the counts of its
//! drops, or the operands of its rerolls or explodes come from random
//! registers, the rolling record holds not one [setup](Setup) but a weighted
//! mixture of them, each a roll with fixed operands and fixed drop counts. By
//! the law of total probability, the sum of the record is the mixture of the
//! sums of its setups.
//!
//! A setup of weight `w` answers sums of total `T`, and the setups' totals
//! differ, as the totals `zᵏ` of `k` dice of `z` faces do, so the mixture
//! scales each setup's sums to the common denominator `L`, the least common
//! multiple of the totals, before it adds them:
//!
//! ```text
//! weight(v) = Σ w · (L / T) · weight_T(v),    total = W · L
//! ```
//!
//! where `W` is the total weight of the setups, the product of the totals of
//! the random operands that chose them.
//!
//! `lean/spec/XdySpec/Mixture.lean` gives each setup the law of the record
//! that it fills, and a mixture the mixture of its setups' laws, and proves
//! that rolling, dropping, keeping and summing a mixture act on that law as the
//! instructions do on the record; `lean/spec/XdySpec/Reroll.lean` proves that
//! rerolling does, as every later instruction views the record. The
//! specification does not yet model explosion, so nothing proves that
//! exploding does, nor that rerolling [chains](super::chain) does. When those
//! operands are rolls with fixed operands, every path through them is equally
//! likely, and the mixture weighs each outcome by the number of paths that
//! reach it. When some path to a setup is likelier than another, the least
//! common multiple of the paths' denominators may be a proper divisor of
//! `W · L`, though each outcome still weighs its probability over `W · L`.

use std::{
	collections::{BTreeMap, BTreeSet},
	mem
};

use super::{
	chain::{Burst, burst},
	meter::{Halt, Meter, charge_of},
	record::{Die, Effect, Powers, Roll, redraw}
};
use crate::{Criterion, Distribution, Weight, primitives::keep_drops};

////////////////////////////////////////////////////////////////////////////////
//                                  Setups.                                   //
////////////////////////////////////////////////////////////////////////////////

/// One setup of a rolling record: its roll, if any, as its rerolls left it,
/// and its drops, with every operand resolved to a value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Setup
{
	/// The roll that filled the record, if any.
	roll: Option<Roll>,

	/// The die from which rerolls draw replacements, if the roll no longer
	/// shows it, as when a reroll changed it, or a split fixed its results
	/// before a reroll. Fixed results that no reroll reads hold none.
	die: Option<Die>,

	/// The number of lowest results dropped, clamped to the number of results.
	lowest: i32,

	/// The number of highest results dropped, clamped to the number of
	/// results.
	highest: i32
}

impl Setup
{
	/// Answer the number of results in the record, which bounds the drops.
	///
	/// # Returns
	/// The number of results.
	fn results(&self) -> i32 { self.roll.as_ref().map_or(0, Roll::results) }

	/// Answer the total weight of the sums of the setup: that of its roll, or
	/// one if it has none.
	///
	/// # Returns
	/// The total weight.
	fn total(&self) -> Weight
	{
		self.roll.as_ref().map_or(Weight::ONE, Roll::total)
	}

	/// Answer the number of dice of the setup's roll, which bounds the width of
	/// its [total](Self::total).
	///
	/// # Returns
	/// The number of dice.
	fn dice(&self) -> u64 { self.roll.as_ref().map_or(0, Roll::dice) }

	/// Answer the number of cells that the setup occupies: those of its roll,
	/// or one if it has none, and those of its die.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u64
	{
		self.roll.as_ref().map_or(1, Roll::cells)
			+ self.die.as_ref().map_or(0, Die::cells)
	}

	/// Answer the die from which rerolls draw replacements.
	///
	/// # Returns
	/// The die, or `None` if the setup has no roll, or fixed results that no
	/// reroll reads.
	fn die(&self) -> Option<Die>
	{
		self.die
			.clone()
			.or_else(|| self.roll.as_ref().and_then(Roll::replacement))
	}

	/// Reroll the dice still kept of the setup.
	///
	/// A reroll that does nothing leaves the setup as it was, and one that
	/// crosses out every die still kept drops them all, unless a saturated die
	/// decides every chain still kept, which it leaves alone. Otherwise, if no
	/// result has been dropped and the roll is a range or dice, then its dice
	/// are alike, and the reroll [rewrites](Roll::rewrite) the law of each;
	/// if not, it [pools](Roll::pool) the multisets of the dice still kept. A
	/// roll of [chains](super::chain) [rerolls](Roll::reroll_chains) the kept
	/// dice of each. Either way, the setup keeps its die, and the total of its
	/// roll grows by the [padding](super::record::Redraw) of each die still
	/// kept.
	///
	/// # Parameters
	/// - `criterion`: The results that the reroll rerolls.
	/// - `limit`: The number of times that each die may be rerolled, or `None`
	///   for no limit.
	/// - `meter`: The meter, to which the rerolled setup stays charged beyond
	///   the cells of the setup.
	///
	/// # Returns
	/// The rerolled setup.
	///
	/// # Errors
	/// [`Halt`] if the reroll would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the setup holds fixed results without their die, which the
	/// [plan](super::plan::rerolled) prevents.
	///
	/// # Notes
	/// `Setup.reroll`, in `lean/spec/XdySpec/Reroll.lean`, mirrors the reroll,
	/// and `Setup.law_reroll` proves that, whether it does nothing, crosses
	/// out, rewrites, or pools, the rerolled setup describes the
	/// specification's reroll of the setup's records, as every later
	/// instruction views them: their kept results and their die, which is all
	/// that any reads. Only the view agrees, since a setup that the reroll
	/// cannot change keeps its drops, and one that it crosses out drops every
	/// die still kept, where the specification's record keeps only its kept
	/// results. The proof asks that every result be a face of the die, as
	/// `Setup.faithful_roll` and `Setup.reroll_invariants` prove of every roll
	/// and every reroll. The specification does not yet model explosion, so
	/// the proof does not cover chains.
	fn reroll(
		&self,
		criterion: &Criterion,
		limit: Option<i32>,
		meter: &mut Meter
	) -> Result<Setup, Halt>
	{
		let results = self.results();
		let kept = results - self.lowest - self.highest;
		let Some(roll) = self.roll.as_ref().filter(|_| kept > 0)
		else
		{
			// Without dice still kept, there is nothing to reroll.
			return Ok(self.clone())
		};
		let die = self.die().expect("a reroll reads fixed results with a die");
		match redraw(&die, criterion, limit, meter)?
		{
			Effect::Nothing => Ok(self.clone()),
			// The reroll leaves alone every chain that a saturated die decides.
			Effect::CrossOut if roll.is_saturated() => Ok(self.clone()),
			Effect::CrossOut => Ok(Setup {
				lowest: results,
				..self.clone()
			}),
			Effect::Redraw(redraw) =>
			{
				meter.charge(0, die.cells())?;
				let roll = match (roll, self.lowest, self.highest)
				{
					(
						Roll::Range { .. }
						| Roll::Standard { .. }
						| Roll::Custom { .. },
						0,
						0
					) => roll.rewrite(&redraw, meter)?,
					(Roll::Chains { .. } | Roll::Stacks(_), ..) => roll
						.reroll_chains(
							self.lowest,
							self.highest,
							&redraw,
							meter
						)?,
					_ => roll.pool(self.lowest, self.highest, &redraw, meter)?
				};
				Ok(Setup {
					roll: Some(roll),
					die: Some(die),
					lowest: 0,
					highest: 0
				})
			}
		}
	}

	/// Explode the chains still kept of the setup, counting each die still
	/// kept as a chain of one die if no explode has extended it yet.
	///
	/// An explode that does nothing, or that has no limit but cannot match
	/// the end of any chain of the roll, leaves the setup as it was, as does
	/// any explode of chains that saturated dice decide, which have no end
	/// from which to grow, so that it counts no branches of them, as
	/// [`Bounds::count`](crate::Bounds::count) does. One that saturates every
	/// chain still kept answers one multiset of saturated chains, weighing the
	/// total of the roll. Otherwise, the roll
	/// [explodes](Roll::explode) the chains still kept, transforming the law of
	/// each if no chain has been dropped and the chains are alike, and their
	/// multisets if not. Either way, the setup keeps its die, and the total of
	/// its roll grows by the padding of each chain still kept.
	///
	/// # Parameters
	/// - `criterion`: The results that the explode explodes.
	/// - `limit`: The number of times that each chain may be exploded, or
	///   `None` for no limit.
	/// - `whole`: Whether to keep the chains whole, since a later reroll or
	///   explode reads them, or reduce them to their totals.
	/// - `meter`: The meter, to which the exploded setup stays charged beyond
	///   the cells of the setup.
	///
	/// # Returns
	/// The exploded setup.
	///
	/// # Errors
	/// [`Halt`] if the explode would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the setup holds fixed results without their die, which the
	/// [plan](super::plan::rerolled) prevents; or if the explode has no limit,
	/// does not match every face, and matches the end of a chain still kept,
	/// which the pass refuses before it begins.
	///
	/// # Notes
	/// The Lean specification does not yet model explosion, so this has no
	/// proof.
	fn explode(
		&self,
		criterion: &Criterion,
		limit: Option<i32>,
		whole: bool,
		meter: &mut Meter
	) -> Result<Setup, Halt>
	{
		let results = self.results();
		let kept = results - self.lowest - self.highest;
		let Some(roll) = self
			.roll
			.as_ref()
			.filter(|roll| kept > 0 && !roll.is_saturated())
		else
		{
			// Without chains still kept, or with only chains that saturated
			// dice decide, which have no end from which to grow, there is
			// nothing to explode, and no branch to count.
			return Ok(self.clone())
		};
		let die = self
			.die()
			.expect("an explode reads fixed results with a die");
		let roll = match burst(&die, criterion, limit, whole, meter)?
		{
			Burst::Nothing => return Ok(self.clone()),
			Burst::Unbounded(criterion) if !roll.shows(criterion) =>
			{
				return Ok(self.clone())
			},
			Burst::Saturate(value) =>
			{
				roll.saturate(value, kept, whole, meter)?
			},
			burst =>
			{
				roll.explode(self.lowest, self.highest, &burst, whole, meter)?
			},
		};
		meter.charge(0, die.cells())?;
		Ok(Setup {
			roll: Some(roll),
			die: Some(die),
			lowest: 0,
			highest: 0
		})
	}

	/// Reduce the [chains](super::chain) of the setup's roll to their totals,
	/// if it has any, as when no later reroll or explode reads them.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the reduced setup stays charged beyond
	///   the cells of the setup.
	///
	/// # Returns
	/// The reduced setup.
	///
	/// # Errors
	/// [`Halt`] if reducing would exceed the budget or is cancelled.
	fn collapse(self, meter: &mut Meter) -> Result<Setup, Halt>
	{
		match &self.roll
		{
			Some(roll @ (Roll::Chains { .. } | Roll::Stacks(_))) =>
			{
				let roll = roll.collapse(meter)?;
				Ok(Setup {
					roll: Some(roll),
					..self
				})
			},
			_ => Ok(self)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Mixtures.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A weighted mixture of the [setups](Setup) of a rolling record. Each setup
/// weighs the number of branches of the random operands that choose it, and
/// no weight is zero. A record whose operands are all fixed holds one setup,
/// of weight one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Mixture(BTreeMap<Setup, Weight>);

impl Default for Mixture
{
	/// Answer the mixture of a record that has not been rolled, which sums to
	/// zero. `Setup.law_empty`, in `lean/spec/XdySpec/Mixture.lean`, proves
	/// that it describes the empty record.
	fn default() -> Self
	{
		Self(BTreeMap::from([(Setup::default(), Weight::ONE)]))
	}
}

impl Mixture
{
	/// Construct the mixture of a record freshly filled by one of the given
	/// rolls, without drops.
	///
	/// # Parameters
	/// - `rolls`: The rolls, each with the number of branches of the random
	///   operands that choose it, in any order, possibly repeated.
	///
	/// # Returns
	/// The mixture, which merges the weights of repeated rolls.
	///
	/// # Notes
	/// `mixtureLaw_rolls`, in `lean/spec/XdySpec/Mixture.lean`, proves that
	/// the mixture describes the rolls' laws mixed over their weights.
	pub(super) fn rolls(rolls: impl IntoIterator<Item = (Roll, Weight)>)
	-> Self
	{
		let mut setups = BTreeMap::<Setup, Weight>::new();
		for (roll, weight) in rolls
		{
			let setup = Setup {
				roll: Some(roll),
				..Setup::default()
			};
			*setups.entry(setup).or_default() += weight;
		}
		Self(setups)
	}

	/// Answer the number of cells that the mixture occupies: those of its
	/// setups.
	///
	/// # Returns
	/// The number of cells.
	pub(super) fn cells(&self) -> u64 { self.0.keys().map(Setup::cells).sum() }

	/// Drop the lowest results of every setup, by each of the given counts.
	///
	/// # Parameters
	/// - `counts`: The distribution of the number of results to drop.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if the drop would exceed the budget or is cancelled.
	pub(super) fn drop_lowest(
		&mut self,
		counts: &Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.drop(counts, |setup| &mut setup.lowest, meter)
	}

	/// Keep the lowest results of every setup, by each of the given counts,
	/// dropping all of the results still kept but that many from the high end.
	///
	/// # Parameters
	/// - `counts`: The distribution of the number of results to keep.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if the keep would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `mixtureLaw_keepLowest`, in `lean/spec/XdySpec/Mixture.lean`, proves
	/// that the mixture then describes lemma 1's lift of the keep over the
	/// record and the counts, and `Setup.law_keepLowest` that a setup may count
	/// the results that its keep drops itself, from its own results and drops.
	pub(super) fn keep_lowest(
		&mut self,
		counts: &Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.clause(
			counts,
			|setup, count| {
				let results = setup.results();
				let dropped =
					keep_drops(results, setup.lowest, setup.highest, count);
				drop_into(&mut setup.highest, dropped, results);
			},
			meter
		)
	}

	/// Keep the highest results of every setup, by each of the given counts,
	/// dropping all of the results still kept but that many from the low end.
	///
	/// # Parameters
	/// - `counts`: The distribution of the number of results to keep.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if the keep would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `mixtureLaw_keepHighest`, in `lean/spec/XdySpec/Mixture.lean`, proves
	/// that the mixture then describes lemma 1's lift of the keep over the
	/// record and the counts, and `Setup.law_keepHighest` that a setup may
	/// count the results that its keep drops itself, from its own results and
	/// drops.
	pub(super) fn keep_highest(
		&mut self,
		counts: &Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.clause(
			counts,
			|setup, count| {
				let results = setup.results();
				let dropped =
					keep_drops(results, setup.lowest, setup.highest, count);
				drop_into(&mut setup.lowest, dropped, results);
			},
			meter
		)
	}

	/// Drop the highest results of every setup, by each of the given counts.
	///
	/// # Parameters
	/// - `counts`: The distribution of the number of results to drop.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if the drop would exceed the budget or is cancelled.
	pub(super) fn drop_highest(
		&mut self,
		counts: &Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.drop(counts, |setup| &mut setup.highest, meter)
	}

	/// Drop results of every setup, by each of the given counts, splitting
	/// each setup by each count and merging the setups that coincide, as when
	/// every count exceeds the results.
	///
	/// # Parameters
	/// - `counts`: The distribution of the number of results to drop.
	/// - `dropped`: The accessor of the count of results that a setup drops at
	///   the end in question.
	/// - `meter`: The meter, which the drop charges for each new setup, and for
	///   its dice, since the new setup's total is computed anew.
	///
	/// # Errors
	/// [`Halt`] if the drop would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `mixtureLaw_dropLowest` and `mixtureLaw_dropHighest`, in
	/// `lean/spec/XdySpec/Mixture.lean`, prove that the mixture then describes
	/// lemma 1's lift of the drop over the record and the counts, and
	/// `Setup.law_dropLowest` and `Setup.law_dropHighest` that a setup may
	/// clamp its drops itself.
	fn drop(
		&mut self,
		counts: &Distribution,
		dropped: impl Fn(&mut Setup) -> &mut i32,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.clause(
			counts,
			|setup, count| {
				let results = setup.results();
				drop_into(dropped(setup), count, results);
			},
			meter
		)
	}

	/// Apply a drop or keep to every setup, by each of the given counts,
	/// splitting each setup by each count and merging the setups that coincide.
	///
	/// # Parameters
	/// - `counts`: The distribution of the count of the drop or keep.
	/// - `apply`: The drop or keep, which updates a setup by one count.
	/// - `meter`: The meter, which the clause charges for each new setup, and
	///   for its dice, since the new setup's total is computed anew.
	///
	/// # Errors
	/// [`Halt`] if the clause would exceed the budget or is cancelled.
	fn clause(
		&mut self,
		counts: &Distribution,
		apply: impl Fn(&mut Setup, i32),
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		let outcomes = counts.len() as u128;
		let (steps, cells) =
			self.0.keys().fold((0u128, 0u128), |(steps, cells), setup| {
				(
					steps + outcomes * (1 + setup.dice() as u128),
					cells + outcomes * setup.cells() as u128
				)
			});
		meter.charge(charge_of(steps), charge_of(cells))?;
		let mut setups = BTreeMap::<Setup, Weight>::new();
		for (setup, weight) in mem::take(&mut self.0)
		{
			for (count, count_weight) in counts
			{
				let mut setup = setup.clone();
				apply(&mut setup, count);
				*setups.entry(setup).or_default() += &weight * count_weight;
			}
		}
		self.0 = setups;
		Ok(())
	}

	/// Reroll the dice still kept of every setup, by each of the given
	/// resolutions of the reroll's operands, splitting each setup by each
	/// resolution and merging the setups that coincide, as when the reroll
	/// does nothing.
	///
	/// # Parameters
	/// - `rerolls`: The resolutions of the reroll: each set of results that it
	///   rerolls, with the number of times that each die may be rerolled, or
	///   `None` for no limit, and the weight of the outcome of the operands
	///   that chose them.
	/// - `whole`: Whether a later reroll or explode reads the record before a
	///   roll refills it, so that its [chains](super::chain) stay whole, rather
	///   than reduce to their totals.
	/// - `meter`: The meter, which the reroll charges a step and the cells of a
	///   setup for each new setup, before each [setup's reroll](Setup::reroll)
	///   charges its own work.
	///
	/// # Errors
	/// [`Halt`] if the reroll would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `mixtureLaw_reroll`, in `lean/spec/XdySpec/Reroll.lean`, proves that the
	/// mixture then describes, as every later instruction views it, the
	/// specification's reroll of the record's law, independent of the
	/// resolutions' law, as `law_reroll` and `law_reroll_resolved` give it.
	pub(super) fn reroll(
		&mut self,
		rerolls: &[(Criterion, Option<i32>, Weight)],
		whole: bool,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.resolve(
			rerolls,
			|setup, criterion, limit, meter| {
				let setup = setup.reroll(criterion, limit, meter)?;
				match whole
				{
					true => Ok(setup),
					false => setup.collapse(meter)
				}
			},
			meter
		)
	}

	/// Explode the chains still kept of every setup, by each of the given
	/// resolutions of the explode's operands, splitting each setup by each
	/// resolution and merging the setups that coincide, as when the explode
	/// does nothing.
	///
	/// # Parameters
	/// - `explodes`: The resolutions of the explode: each set of results that
	///   it explodes, with the number of times that each chain may be exploded,
	///   or `None` for no limit, and the weight of the outcome of the operands
	///   that chose them.
	/// - `whole`: Whether a later reroll or explode reads the record before a
	///   roll refills it, so that its [chains](super::chain) stay whole, rather
	///   than reduce to their totals.
	/// - `meter`: The meter, which the explode charges a step and the cells of
	///   a setup for each new setup, before each [setup's
	///   explode](Setup::explode) charges its own work.
	///
	/// # Errors
	/// [`Halt`] if the explode would exceed the budget or is cancelled.
	///
	/// # Notes
	/// The Lean specification does not yet model explosion, so this has no
	/// proof.
	pub(super) fn explode(
		&mut self,
		explodes: &[(Criterion, Option<i32>, Weight)],
		whole: bool,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		self.resolve(
			explodes,
			|setup, criterion, limit, meter| {
				let setup = setup.explode(criterion, limit, whole, meter)?;
				match whole
				{
					true => Ok(setup),
					false => setup.collapse(meter)
				}
			},
			meter
		)
	}

	/// Apply a reroll or an explode to every setup, by each of the given
	/// resolutions of its operands, splitting each setup by each resolution
	/// and merging the setups that coincide.
	///
	/// # Parameters
	/// - `resolutions`: The resolutions: each criterion, with its limit, or
	///   `None` for no limit, and the weight of the outcome of the operands
	///   that chose them.
	/// - `apply`: The reroll or explode, which acts on a setup by one
	///   resolution, and charges its own work.
	/// - `meter`: The meter, which the clause charges a step and the cells of a
	///   setup for each new setup.
	///
	/// # Errors
	/// [`Halt`] if the clause would exceed the budget or is cancelled.
	fn resolve(
		&mut self,
		resolutions: &[(Criterion, Option<i32>, Weight)],
		apply: impl Fn(
			&Setup,
			&Criterion,
			Option<i32>,
			&mut Meter
		) -> Result<Setup, Halt>,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		let outcomes = resolutions.len() as u128;
		let (steps, cells) =
			self.0.keys().fold((0u128, 0u128), |(steps, cells), setup| {
				(steps + outcomes, cells + outcomes * setup.cells() as u128)
			});
		meter.charge(charge_of(steps), charge_of(cells))?;
		let mut setups = BTreeMap::<Setup, Weight>::new();
		for (setup, weight) in mem::take(&mut self.0)
		{
			for (criterion, limit, resolution_weight) in resolutions
			{
				let setup = apply(&setup, criterion, *limit, meter)?;
				*setups.entry(setup).or_default() +=
					&weight * resolution_weight;
			}
		}
		self.0 = setups;
		Ok(())
	}

	/// Answer the least common multiple of the totals of the setups, the
	/// common denominator to which the mixture scales their sums.
	///
	/// # Parameters
	/// - `meter`: The meter, which the denominator charges a step for each
	///   setup. Each setup charged its dice, which bound the width of its
	///   total, when it was made.
	///
	/// # Returns
	/// The common denominator.
	///
	/// # Errors
	/// [`Halt`] if the denominator would exceed the budget or is cancelled.
	fn denominator(&self, meter: &mut Meter) -> Result<Weight, Halt>
	{
		meter.charge(self.0.len() as u64, 0)?;
		Ok(self
			.0
			.keys()
			.fold(Weight::ONE, |lcm, setup| lcm.lcm(&setup.total())))
	}

	/// Answer the total weight of the mixture, the total weight of its setups
	/// times their [common denominator](Self::denominator), which is the total
	/// of its [sum](Self::sum).
	///
	/// # Parameters
	/// - `meter`: The meter.
	///
	/// # Returns
	/// The total weight.
	///
	/// # Errors
	/// [`Halt`] if the total would exceed the budget or is cancelled.
	pub(super) fn total(&self, meter: &mut Meter) -> Result<Weight, Halt>
	{
		meter.charge(self.0.len() as u64, 0)?;
		Ok(self.0.values().sum::<Weight>() * self.denominator(meter)?)
	}

	/// Answer the distribution of the sum of the record: the sums of the
	/// setups, each scaled to the [common denominator](Self::denominator) and
	/// weighed by its setup.
	///
	/// A setup's sum comes from [`Roll::sum`], except that the uniform sums of
	/// ranges without drops accumulate in a difference array, so that ranges
	/// of random endpoints cost `O(p log p + w)` for `p` pairs of endpoints
	/// over a span of width `w`, rather than `O(p · w)`. The sums of dice
	/// without drops share their [convolution powers](Powers), so that a
	/// random count steps up from one power to the next.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the sum stays charged.
	///
	/// # Returns
	/// The distribution of the sum, whose total is [`total`](Self::total).
	///
	/// # Errors
	/// [`Halt`] if the sum would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `map_sum_mixtureLaw`, in `lean/spec/XdySpec/Mixture.lean`, proves that
	/// the sum of the record is the mixture of its setups' sums, and
	/// `Dist.hasLaw_mixture`, in `lean/spec/XdySpec/Law.lean`, that scaling
	/// them to their common denominator mixes their laws.
	pub(super) fn sum(&self, meter: &mut Meter) -> Result<Distribution, Halt>
	{
		let mut powers = Powers::default();
		if self.0.len() == 1
			&& let Some((setup, weight)) = self.0.first_key_value()
			&& *weight == Weight::ONE
		{
			// A record whose operands are all fixed needs no scaling.
			return setup_sum(setup, &mut powers, meter)
		}
		let denominator = self.denominator(meter)?;
		let mut sums = BTreeMap::<i32, Weight>::new();
		let mut ranges = Ranges::default();
		for (setup, weight) in &self.0
		{
			meter.charge(1, 0)?;
			let scale = weight * denominator.exact_div(&setup.total());
			match setup
			{
				&Setup {
					roll: Some(Roll::Range { start, end }),
					lowest: 0,
					highest: 0,
					..
				} if start <= end => ranges.add(start, end, scale, meter)?,
				_ =>
				{
					let sum = setup_sum(setup, &mut powers, meter)?;
					// The sums grow by at most as many outcomes as the setup
					// has, and the setup's sum is then dropped.
					let len = sum.len() as u64;
					let before = sums.len() as u64;
					meter.charge(len, len)?;
					for (outcome, weight) in sum
					{
						*sums.entry(outcome).or_default() += weight * &scale;
					}
					let grown = sums.len() as u64 - before;
					meter.free(2 * len - grown);
				}
			}
		}
		ranges.sweep(&mut sums, meter)?;
		let len = sums.len() as u64;
		meter.charge(len, len)?;
		let distribution =
			Distribution::from_weights(sums).expect("a mixture has outcomes");
		meter.free(2 * len - distribution.len() as u64);
		Ok(distribution)
	}

	/// Answer every outcome of the record: each setup with each multiset of
	/// the [results](Roll::outcomes) of its roll, fixed, keeping its drops,
	/// and its die, if a reroll will draw from it.
	/// An outcome of a setup of weight `w` and total `T` whose roll reaches it
	/// along `b` branches weighs `w · (L / T) · b`, so that the outcomes of
	/// each setup share the [common denominator](Self::denominator) `L`, as its
	/// sums do, and the weights sum to the [total](Self::total). Setups that
	/// reach the same outcome, as no dice and fewer than none do, merge their
	/// weights, so that the outcomes are distinct.
	///
	/// # Parameters
	/// - `rerolled`: Whether a reroll reads the record before a roll refills
	///   it, so that each outcome must keep the die from which it draws.
	/// - `meter`: The meter, to which the outcomes stay charged.
	///
	/// # Returns
	/// The distinct outcomes, each a mixture of one setup of weight one, whose
	/// sum is a point mass, with their weights.
	///
	/// # Errors
	/// [`Halt`] if the outcomes would exceed the budget or are cancelled.
	///
	/// # Notes
	/// `Setup.tsum_law_toOracle`, in `lean/spec/XdySpec/Outcomes.lean`, proves
	/// that each setup's outcomes, over its total, are the law of its record
	/// read sorted, with its drops, and `map_toOracle_mixtureLaw` that the
	/// record's law, read sorted, mixes them; `Dist.hasLaw_mixture`, in
	/// `lean/spec/XdySpec/Law.lean`, proves that scaling them to their common
	/// denominator mixes their laws.
	pub(super) fn outcomes(
		&self,
		rerolled: bool,
		meter: &mut Meter
	) -> Result<Vec<(Mixture, Weight)>, Halt>
	{
		let denominator = self.denominator(meter)?;
		let mut outcomes = BTreeMap::<Setup, Weight>::new();
		for (setup, weight) in &self.0
		{
			meter.charge(1, 0)?;
			let scale = weight * denominator.exact_div(&setup.total());
			let rolls = match &setup.roll
			{
				Some(roll) => roll
					.outcomes(meter)?
					.into_iter()
					.map(|(roll, branches)| (Some(roll), branches))
					.collect(),
				// A record never rolled has no results.
				None =>
				{
					meter.charge(1, 1)?;
					vec![(None, Weight::ONE)]
				}
			};
			// Each outcome moves its cells into the setup that holds it, with
			// a copy of the die, if it keeps it.
			let die = setup.die().filter(|_| rerolled);
			let len = rolls.len() as u64;
			meter.charge(len, len * die.as_ref().map_or(0, Die::cells))?;
			for (roll, branches) in rolls
			{
				let outcome = Setup {
					roll,
					die: die.clone(),
					..setup.clone()
				};
				*outcomes.entry(outcome).or_default() += branches * &scale;
			}
		}
		Ok(outcomes
			.into_iter()
			.map(|(setup, weight)| {
				(Self(BTreeMap::from([(setup, Weight::ONE)])), weight)
			})
			.collect())
	}
}

/// Add a count of results dropped to a setup's running total at one end,
/// exactly as [`RollingRecord::drop_lowest`](crate::RollingRecord::drop_lowest)
/// and [`RollingRecord::drop_highest`](crate::RollingRecord::drop_highest) do:
/// a count of zero or less drops nothing, and the total is clamped to the
/// results.
///
/// # Parameters
/// - `dropped`: The running total of results dropped at one end.
/// - `count`: The number of results to drop.
/// - `results`: The number of results of the setup.
fn drop_into(dropped: &mut i32, count: i32, results: i32)
{
	*dropped = dropped.saturating_add(count.max(0)).clamp(0, results);
}

/// Answer the distribution of the sum of one setup.
///
/// # Parameters
/// - `setup`: The setup.
/// - `powers`: The convolution powers of the dice summed so far.
/// - `meter`: The meter, to which the sum stays charged.
///
/// # Returns
/// The distribution of the sum, whose total is that of the setup.
///
/// # Errors
/// [`Halt`] if the sum would exceed the budget or is cancelled.
fn setup_sum(
	setup: &Setup,
	powers: &mut Powers,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	match &setup.roll
	{
		Some(roll) => roll.sum(setup.lowest, setup.highest, powers, meter),
		// An empty record sums to zero, as `Setup.map_sum_law_of_none`, in
		// `lean/spec/XdySpec/Mixture.lean`, proves.
		None =>
		{
			meter.charge(1, 1)?;
			Ok(Distribution::point(0))
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Ranges.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A difference array of uniform weights over ranges, which accumulates each
/// range in `O(log p)` and answers all of them in one sweep.
#[derive(Debug, Default)]
struct Ranges
{
	/// The weight that each outcome adds to the level of every outcome from it
	/// onward: the sum of the weights of the ranges that start there.
	rises: BTreeMap<i64, Weight>,

	/// The weight that each outcome removes from the level of every outcome
	/// from it onward: the sum of the weights of the ranges that end just
	/// before it.
	falls: BTreeMap<i64, Weight>
}

impl Ranges
{
	/// Weigh every outcome of the given range, which is not empty, by the
	/// given weight.
	///
	/// # Parameters
	/// - `start`: The least outcome of the range.
	/// - `end`: The greatest outcome of the range, not less than `start`.
	/// - `weight`: The weight of each outcome.
	/// - `meter`: The meter, which the range charges for its rise and its fall.
	///
	/// # Errors
	/// [`Halt`] if the range would exceed the budget or is cancelled.
	fn add(
		&mut self,
		start: i32,
		end: i32,
		weight: Weight,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		meter.charge(2, 2)?;
		*self.falls.entry(end as i64 + 1).or_default() += &weight;
		*self.rises.entry(start as i64).or_default() += weight;
		Ok(())
	}

	/// Add the weight of every outcome of the ranges to the given sums, by
	/// sweeping the outcomes in ascending order and adjusting the level at
	/// each start or end of a range.
	///
	/// # Parameters
	/// - `sums`: The weights of the outcomes.
	/// - `meter`: The meter, which the sweep charges for its breaks and for
	///   every outcome between the first and the last, and which the sums stay
	///   charged to.
	///
	/// # Errors
	/// [`Halt`] if the sweep would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `Span.sweep_eq_cover`, in `lean/spec/XdySpec/Mixture.lean`, proves that
	/// the sweep gives every outcome the weight of the ranges that cover it,
	/// as adding each range's outcomes one by one would, and so that the level
	/// never falls below zero.
	fn sweep(
		self,
		sums: &mut BTreeMap<i32, Weight>,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		let (Some(first), Some(last)) =
			(self.rises.first_key_value(), self.falls.last_key_value())
		else
		{
			// No ranges.
			return Ok(())
		};
		let span = (last.0 - first.0) as u64;
		let edges = (self.rises.len() + self.falls.len()) as u64;
		meter.charge(span + edges, span + 2 * edges)?;
		let before = sums.len() as u64;
		let breaks = self
			.rises
			.keys()
			.chain(self.falls.keys())
			.copied()
			.collect::<BTreeSet<_>>()
			.into_iter()
			.collect::<Vec<_>>();
		let mut level = Weight::ZERO;
		for pair in breaks.windows(2)
		{
			let (here, next) = (pair[0], pair[1]);
			if let Some(rise) = self.rises.get(&here)
			{
				level += rise;
			}
			if let Some(fall) = self.falls.get(&here)
			{
				level = level
					.checked_sub(fall)
					.expect("a range ends only after it starts");
			}
			if !level.is_zero()
			{
				for outcome in here..next
				{
					*sums.entry(outcome as i32).or_default() += &level;
				}
			}
		}
		// The breaks are dropped, and the sums keep only what they grew by.
		let grown = sums.len() as u64 - before;
		meter.free(span + 2 * edges - grown);
		Ok(())
	}
}
