//! # Chains
//!
//! An [explode](crate::Explode) extends a die still kept with more dice, rolled
//! from the same die, which form a _chain_ with it, and every later drop, keep,
//! and sum counts each chain still kept as one die, worth its total, as
//! [`RollingRecord::chains`](crate::RollingRecord::chains) describes. The
//! forward pass tracks a [chain](Chain) by all that any later instruction can
//! tell of it: the multiset of its kept dice, and the die at its end, from
//! which a later explode grows it; or, for a chain that a saturated die
//! decides, only its value, since no later reroll or explode touches it.
//!
//! A chain's total behaves exactly as the result of a die does: drops and
//! keeps order chains by their totals first, and the sum adds them, so where
//! no later reroll or explode reads a record before a roll refills it, as is
//! usual, the explode reduces each chain to its total at once, and fills the
//! record with [custom dice](Roll::Custom), or a [pool](Roll::Pool), of
//! totals, which the rest of the pass treats as any other. Only where a later
//! reroll or explode reads the record does the explode keep whole chains: as
//! [i.i.d. chains](Roll::Chains) before any drop, or as [stacks](Roll::Stacks)
//! after one. Stacks hold the multisets of the chains still kept as of the
//! drop, or the split that fixed them, and the [law](Laws) of what each
//! distinct chain of them has become since. Rerolls and explodes act on each
//! chain alone, so a later one rewrites only those laws, and the multisets of
//! the chains themselves are joined only when a drop or a split needs them
//! together, which would otherwise multiply the work of every later reroll
//! or explode by the multisets of the earlier ones. After the last such
//! reader, the chains reduce to their totals, joining the laws of their
//! totals rather than of the chains.
//!
//! An explode limited to `n` times counts its branches as though each chain
//! still kept rolled in advance every extension that it might need, as
//! [`Bounds::count`](crate::Bounds::count) does, so that every path is equally
//! likely: each chain weighs `Fⁿ` in all, for a die of `F` branches. A chain
//! whose end the explode does not match keeps its dice along all `Fⁿ`. One
//! whose end matches rolls an extension, which shows `x` along `w(x)` branches;
//! if `x` matches, and fewer than `n` extensions have been rolled, the chain
//! extends again from `x`, and otherwise it ends at `x`, padded by `Fⁿ⁻ʲ` for
//! the `n − j` extensions after the `j`th that it never needed. An unlimited
//! explode whose predicate matches every face saturates every chain still kept,
//! whose total becomes the saturated value of its die, so the record keeps its
//! total weight; no later reroll or explode touches a saturated chain, nor
//! counts a branch of it. Any
//! other unlimited explode may roll without bound, and has no exact
//! distribution of finite weight, so the pass refuses every function that has
//! one that may extend a chain before it begins.
//!
//! A reroll of chains rerolls every kept die of every chain that no saturated
//! die decides, alone, as [`Redraw`] directs, and the replacement of a chain's
//! end becomes its new end. A chain of `d` kept dice then weighs `Gᵈ` in all,
//! where `G` is the padding of the reroll, so chains of fewer dice are padded
//! to the most that any holds, so that every path through the record weighs
//! the same.
//!
//! `lean/spec/XdySpec/Explode.lean` proves the laws of [`burst`], whose tails,
//! over their padding, are the specification's extension of a chain whose end
//! an explode matches (`Burst.weighted_chains`), and whose folds total as the
//! grown chains do (`Burst.weighted_totals`); of [`Roll::explode`] on a range,
//! dice, or i.i.d. chains none of which has been dropped; of
//! [`Roll::saturate`]; and of [`Roll::collapse`] of i.i.d. chains
//! (`Setup.law_burst_alike`). `lean/spec/XdySpec/Stacks.lean` proves that the
//! multisets of the chains still kept after a drop, which [`kept_pool`] keeps,
//! are the law of those chains (`Setup.map_keptChains_law`); that
//! [`transform_pool`] draws what each chain still kept becomes independently
//! (`weighted_transformPool`); and so the laws of [`Roll::explode`] on the
//! chains of a roll after a drop, or of fixed results or a pool, into stacks
//! or a pool of totals (`Setup.law_burst`). It proves too that what
//! [`rerolled`] makes of a chain, over its padding, is the specification's
//! reroll of each of its dice (`weighted_rerolled`), and so the laws of
//! [`Roll::reroll_chains`] on i.i.d. chains, dropped or not
//! (`Setup.law_reroll`), whose stacks weigh each multiset by the branches of
//! its deferred laws, as [`joined`] does (`stackWeight`); that [`joined`],
//! over its weights, draws what each chain of stacks has become, so that the
//! joins that stacks defer are the eager ones (`weighted_joined`); and that
//! [`Roll::collapse`] of stacks, by [`joined_totals`], leaves the results of
//! the records that they fill (`Roll.map_results_collapse`). The rerolls and
//! explodes of stacks none of whose chains has been dropped, which compose the
//! deferred laws of stacks, have no proof yet; the tests check every law
//! against an independent enumeration.

use std::{
	cmp::Ordering,
	collections::{BTreeMap, HashMap, hash_map::Entry},
	iter
};

use super::{
	meter::{Halt, Meter, charge_of},
	record::{
		Die, Element, Multisets, Redraw, Roll, join, kept_sum, multiset_cells,
		multisets
	}
};
use crate::{Criterion, Weight};

////////////////////////////////////////////////////////////////////////////////
//                                   Folds.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The saturating fold of some values in ascending order, by its parts: the
/// sum of the negative values, which the fold adds first, and so saturates at
/// most once, at `i32::MIN`, and the sum of the positive values, which it adds
/// next, from there, and so saturates at most once, at `i32::MAX`. The parts
/// of a union of values are the sums of their parts, clamped, so folds form a
/// monoid, and the total of the union never needs the values themselves.
///
/// # Notes
/// `Fold.add_ofList`, in `lean/spec/XdySpec/Explode.lean`, proves that the
/// fold of a union is the union of the folds, and `Fold.total_ofList` that the
/// total of a fold is the saturating sum of its values in ascending order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct Fold
{
	/// The sum of the negative values, clamped below at `i32::MIN`.
	negative: i64,

	/// The sum of the positive values, clamped above at [`Self::CEILING`].
	positive: i64
}

impl Fold
{
	/// The greatest sum of positive values that a fold keeps, which exceeds
	/// the distance from `i32::MIN` to `i32::MAX`, so that clamping there never
	/// changes a total.
	const CEILING: i64 = 1 << 33;

	/// Answer the fold of copies of one value.
	///
	/// # Parameters
	/// - `value`: The value.
	/// - `copies`: The number of copies, which is positive.
	///
	/// # Returns
	/// The fold.
	fn of(value: i32, copies: i32) -> Self
	{
		let sum = value as i64 * copies as i64;
		match value < 0
		{
			true => Self {
				negative: sum.max(i32::MIN as i64),
				positive: 0
			},
			false => Self {
				negative: 0,
				positive: sum.min(Self::CEILING)
			}
		}
	}

	/// Answer the fold of the union of the values of two folds.
	///
	/// # Parameters
	/// - `other`: The other fold.
	///
	/// # Returns
	/// The fold of the union.
	fn add(self, other: Self) -> Self
	{
		Self {
			negative: (self.negative + other.negative).max(i32::MIN as i64),
			positive: (self.positive + other.positive).min(Self::CEILING)
		}
	}

	/// Answer the total of the fold: the saturating sum of its values in
	/// ascending order, as [`kept_sum`] computes it.
	///
	/// # Returns
	/// The total.
	fn total(self) -> i32
	{
		(self.negative + self.positive).min(i32::MAX as i64) as i32
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Chains.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A chain still kept of a rolling record, as every later instruction views
/// it: the multiset of its kept dice and the die at its end, or, if a
/// saturated die decides it, only its value.
///
/// Chains are ordered as drops and keeps order them: by their totals, then by
/// their kept dice in ascending order, compared lexicographically, and then by
/// the dice at their ends, as
/// [`RollingRecord::chains`](crate::RollingRecord::chains) describes. Every
/// chain still kept of a record is saturated or none is, so the order of a
/// saturated chain and one that is not never matters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Chain
{
	/// The total of the chain: the saturating sum of its kept dice in
	/// ascending order, or the value of its saturated die.
	total: i32,

	/// The distinct values of the kept dice of the chain, in ascending order,
	/// each with its positive number of copies, including the die at its end,
	/// or none if a saturated die decides it.
	dice: Vec<(i32, i32)>,

	/// The value of the die at the end of the chain, from which an explode
	/// grows it, or `None` if a saturated die decides it.
	end: Option<i32>
}

impl Chain
{
	/// Answer the chain of one die, which no die extends.
	///
	/// # Parameters
	/// - `value`: The result of the die.
	///
	/// # Returns
	/// The chain.
	pub(super) fn die(value: i32) -> Self
	{
		Self {
			total: value,
			dice: vec![(value, 1)],
			end: Some(value)
		}
	}

	/// Answer a chain that a saturated die decides.
	///
	/// # Parameters
	/// - `value`: The value of the saturated die.
	///
	/// # Returns
	/// The chain.
	pub(super) fn saturated(value: i32) -> Self
	{
		Self {
			total: value,
			dice: Vec::new(),
			end: None
		}
	}

	/// Answer the chain of the specified kept dice, which ends at the
	/// specified die.
	///
	/// # Parameters
	/// - `dice`: The distinct values of the kept dice, in ascending order, each
	///   with its positive number of copies, including the end.
	/// - `end`: The value of the die at the end.
	///
	/// # Returns
	/// The chain.
	fn of(dice: Vec<(i32, i32)>, end: i32) -> Self
	{
		Self {
			total: kept_sum(&dice, 0, 0),
			dice,
			end: Some(end)
		}
	}

	/// Answer the total of the chain, which a drop, a keep, or a sum reads.
	///
	/// # Returns
	/// The total.
	#[inline]
	pub(super) fn total(&self) -> i32 { self.total }

	/// Answer whether a saturated die decides the chain.
	///
	/// # Returns
	/// `true` if a saturated die decides the chain, `false` otherwise.
	#[inline]
	pub(super) fn is_saturated(&self) -> bool { self.end.is_none() }

	/// Answer the number of kept dice of the chain that a reroll may reroll.
	///
	/// # Returns
	/// The number of kept dice, which is zero if a saturated die decides the
	/// chain.
	fn length(&self) -> u64
	{
		self.dice.iter().map(|(_, copies)| *copies as u64).sum()
	}

	/// Answer the fold of the kept dice of the chain, which does not decide
	/// it.
	///
	/// # Returns
	/// The fold.
	fn fold(&self) -> Fold
	{
		self.dice
			.iter()
			.fold(Fold::default(), |fold, &(value, copies)| {
				fold.add(Fold::of(value, copies))
			})
	}

	/// Answer the values of the kept dice of the chain, in ascending order,
	/// each as often as it is kept.
	///
	/// # Returns
	/// The values.
	fn values(&self) -> impl Iterator<Item = i32> + '_
	{
		self.dice
			.iter()
			.flat_map(|&(value, copies)| iter::repeat_n(value, copies as usize))
	}
}

impl Ord for Chain
{
	fn cmp(&self, other: &Self) -> Ordering
	{
		self.total
			.cmp(&other.total)
			.then_with(|| self.values().cmp(other.values()))
			.then_with(|| self.end.cmp(&other.end))
	}
}

impl PartialOrd for Chain
{
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>
	{
		Some(self.cmp(other))
	}
}

impl Element for Chain
{
	/// A chain occupies one cell, and one more for each distinct value of its
	/// kept dice.
	fn cells(&self) -> u64 { 1 + self.dice.len() as u64 }
}

/// The dice that an explode adds to a chain whose end it matches: the
/// multiset of its extensions, and the last of them, which becomes the end of
/// the chain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Tail
{
	/// The distinct values of the extensions, in ascending order, each with
	/// its positive number of copies.
	dice: Vec<(i32, i32)>,

	/// The value of the last extension.
	end: i32
}

impl Tail
{
	/// Answer the number of cells that the tail occupies, with its weight.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u64 { 2 + self.dice.len() as u64 }
}

////////////////////////////////////////////////////////////////////////////////
//                                 Explodes.                                  //
////////////////////////////////////////////////////////////////////////////////

/// What an explode does to the chains still kept of a roll, given the die from
/// which it draws their extensions.
#[derive(Debug)]
pub(super) enum Burst<'c>
{
	/// Nothing, since no face matches, or the limit is not positive.
	Nothing,

	/// The explode saturates every chain still kept at the specified value,
	/// rolling nothing, since it has no limit and matches every face.
	Saturate(i32),

	/// The explode has no limit, and does not match every face, so it may
	/// extend a chain without bound. The pass refuses every function in which
	/// such an explode may extend a chain, so it extends none.
	Unbounded(&'c Criterion),

	/// The explode may extend the chains still kept, each up to its limit.
	Extend(Extend<'c>)
}

/// A limited explode that may extend the chains still kept of a roll, which
/// counts the branches of each chain as though it rolled in advance every
/// extension that it might need.
#[derive(Debug)]
pub(super) struct Extend<'c>
{
	/// The results that the explode explodes.
	criterion: &'c Criterion,

	/// The branches of each chain, `Fⁿ`, for a die of `F` branches and a limit
	/// of `n`.
	padding: Weight,

	/// The tails that the explode may add to a chain whose end it matches,
	/// each weighed by its branches, which sum to the padding.
	tails: Tails
}

/// The tails that an explode may add to a chain whose end it matches.
#[derive(Debug)]
enum Tails
{
	/// Their folds alone, when the chains reduce to their totals at once.
	Folds(Vec<(Fold, Weight)>),

	/// The tails themselves, when a later reroll or explode reads the chains.
	Tails(Vec<(Tail, Weight)>)
}

/// Resolve an explode against the die from which it draws: decide whether it
/// may extend the chains still kept, and if so, how a chain whose end it
/// matches may grow.
///
/// # Parameters
/// - `die`: The die.
/// - `criterion`: The results that the explode explodes.
/// - `limit`: The number of times that each chain may be exploded, or `None`
///   for no limit.
/// - `whole`: Whether the chains stay whole, for a later reroll or explode, or
///   reduce to their totals at once.
/// - `meter`: The meter, which the resolution charges for counting the branches
///   of the die, for listing its faces, a step for each extension that a chain
///   may need, since the padding is a power of the die's branches, and for
///   every state of the extensions, and to which the tails stay charged.
///
/// # Returns
/// The effect of the explode.
///
/// # Errors
/// [`Halt`] if the resolution would exceed the budget or is cancelled.
pub(super) fn burst<'c>(
	die: &Die,
	criterion: &'c Criterion,
	limit: Option<i32>,
	whole: bool,
	meter: &mut Meter
) -> Result<Burst<'c>, Halt>
{
	// Counting the branches takes a step, and one for each face of a custom
	// die.
	meter.charge(1 + die.cells(), 0)?;
	let branches = die.branches();
	let matching = die.matching(criterion);
	let limit = match limit
	{
		_ if matching.is_zero() => return Ok(Burst::Nothing),
		Some(limit) if limit <= 0 => return Ok(Burst::Nothing),
		None if matching == branches =>
		{
			return Ok(Burst::Saturate(die.saturated_value()))
		},
		None => return Ok(Burst::Unbounded(criterion)),
		Some(limit) => limit as u32
	};
	let distinct = die.distinct();
	meter.charge(
		charge_of(distinct as u128 + limit as u128),
		charge_of(2 * distinct as u128)
	)?;
	let faces = die.faces();
	// The tails stay charged, and occupy the cells of the ways in which a
	// chain grows; the states that may still grow, and the faces, do not.
	let tails = match whole
	{
		true =>
		{
			let (tails, charged) = extensions(
				&faces,
				criterion,
				&branches,
				limit,
				Vec::<(i32, i32)>::new(),
				|dice, face| join(dice, &[(face, 1)]),
				|dice, end| Tail { dice, end },
				|j| j as u64 + 2,
				meter
			)?;
			let occupied =
				tails.iter().map(|(tail, _)| tail.cells()).sum::<u64>();
			meter.free(charged - occupied);
			Tails::Tails(tails)
		},
		false =>
		{
			let (folds, charged) = extensions(
				&faces,
				criterion,
				&branches,
				limit,
				Fold::default(),
				|fold, face| fold.add(Fold::of(face, 1)),
				|fold, _| fold,
				|_| 1,
				meter
			)?;
			meter.free(charged - folds.len() as u64);
			Tails::Folds(folds)
		}
	};
	meter.free(2 * distinct);
	Ok(Burst::Extend(Extend {
		criterion,
		padding: branches.pow(limit),
		tails
	}))
}

/// Answer every way in which a chain whose end an explode matches may grow, as
/// [`burst`] describes: a dynamic program over the extensions, whose states
/// are the extensions so far of the chains that may still grow.
///
/// # Type parameters
/// - `S`: The type of a state.
/// - `K`: The type of a way in which a chain may grow.
///
/// # Parameters
/// - `faces`: The distinct faces of the die, in ascending order, each with its
///   branches.
/// - `criterion`: The results that the explode explodes.
/// - `branches`: The branches of the die, `F`.
/// - `limit`: The limit, `n`, which is positive.
/// - `start`: The state before any extension.
/// - `extend`: The state after another extension.
/// - `close`: The way in which a chain grows, given its last state and its last
///   extension.
/// - `cells`: The most cells that a state or way occupies after the specified
///   number of extensions.
/// - `meter`: The meter, which the program charges for each extension before it
///   rolls it.
///
/// # Returns
/// The ways, in ascending order, each with its branches, padded by those of
/// the extensions that it never needed, so that they sum to `Fⁿ`, and the
/// cells that the program charged.
///
/// # Errors
/// [`Halt`] if the program would exceed the budget or is cancelled.
///
/// # Notes
/// `growths`, in `lean/spec/XdySpec/Explode.lean`, mirrors the program,
/// depth first rather than stage by stage; `sum_growths` proves that the ways
/// weigh `Fⁿ` in all, and `weighted_growths` and `extension_eq_growthLaw`
/// that, over their weights, they are the specification's extension of a
/// chain. `growthLaw_folds` proves that the folds are those of the tails.
#[allow(clippy::too_many_arguments)]
fn extensions<S, K>(
	faces: &[(i32, Weight)],
	criterion: &Criterion,
	branches: &Weight,
	limit: u32,
	start: S,
	extend: impl Fn(&S, i32) -> S,
	close: impl Fn(S, i32) -> K,
	cells: impl Fn(u32) -> u64,
	meter: &mut Meter
) -> Result<(Vec<(K, Weight)>, u64), Halt>
where
	S: Ord,
	K: Ord
{
	let mut open = BTreeMap::from([(start, Weight::ONE)]);
	let mut closed = BTreeMap::<K, Weight>::new();
	let mut charged = 0u64;
	// The branches of the extensions after the current one, `Fⁿ⁻ʲ`.
	let mut padding = branches.pow(limit - 1);
	for j in 1..=limit
	{
		// Each state rolls each face, and the padding falls by a factor of
		// `F`.
		let moves = open.len() as u128 * faces.len() as u128;
		let stage = charge_of(moves * cells(j) as u128);
		meter.charge(charge_of(moves + 1), stage)?;
		charged = charged.saturating_add(stage);
		let mut next = BTreeMap::<S, Weight>::new();
		for (state, weight) in &open
		{
			for &(face, ref branches) in faces
			{
				let grown = extend(state, face);
				let weight = weight * branches;
				match j < limit && criterion.matches(face)
				{
					true => *next.entry(grown).or_default() += weight,
					false =>
					{
						*closed.entry(close(grown, face)).or_default() +=
							weight * &padding;
					}
				}
			}
		}
		open = next;
		if open.is_empty()
		{
			break
		}
		padding = padding.exact_div(branches);
	}
	Ok((closed.into_iter().collect(), charged))
}

impl Burst<'_>
{
	/// Answer every chain that the specified chain still kept may become after
	/// the explode, each with its branches, which sum to the padding.
	///
	/// # Parameters
	/// - `chain`: The chain.
	/// - `meter`: The meter, to which the chains stay charged.
	///
	/// # Returns
	/// The chains, in ascending order, and the number of kept dice by which a
	/// reroll pads them, which is zero.
	///
	/// # Errors
	/// [`Halt`] if the chains would exceed the budget or are cancelled.
	///
	/// # Panics
	/// If the explode does nothing or saturates, which the
	/// [setup](super::mixture) handles whole; if the explode has no limit, and
	/// matches the end of the chain, which the pass refuses; or if the explode
	/// reduces chains to their totals.
	fn chains(
		&self,
		chain: &Chain,
		meter: &mut Meter
	) -> Result<(Vec<(Chain, Weight)>, u64), Halt>
	{
		let chains = match self.grows(chain)
		{
			None =>
			{
				meter.charge(1, chain.cells())?;
				vec![(chain.clone(), self.padding())]
			},
			Some(Tails::Tails(tails)) =>
			{
				let cells = tails.iter().fold(0u128, |cells, (tail, _)| {
					cells + (chain.cells() + tail.cells()) as u128
				});
				meter.charge(tails.len() as u64, charge_of(cells))?;
				let mut chains = tails
					.iter()
					.map(|(tail, weight)| {
						let dice = join(&chain.dice, &tail.dice);
						(Chain::of(dice, tail.end), weight.clone())
					})
					.collect::<Vec<_>>();
				chains.sort();
				let occupied =
					chains.iter().map(|(chain, _)| chain.cells()).sum::<u64>();
				meter.free(charge_of(cells) - occupied);
				chains
			},
			Some(Tails::Folds(_)) =>
			{
				unreachable!("the chains reduce to their totals")
			}
		};
		Ok((chains, 0))
	}

	/// Answer every total that the specified chain still kept may reach after
	/// the explode, each with its branches, which sum to the padding.
	///
	/// # Parameters
	/// - `chain`: The chain.
	/// - `meter`: The meter, to which the totals stay charged.
	///
	/// # Returns
	/// The totals, in ascending order, and the number of kept dice by which a
	/// reroll pads them, which is zero.
	///
	/// # Errors
	/// [`Halt`] if the totals would exceed the budget or are cancelled.
	///
	/// # Panics
	/// As [`chains`](Self::chains) does, but if the explode keeps the chains
	/// whole.
	fn totals(
		&self,
		chain: &Chain,
		meter: &mut Meter
	) -> Result<(Vec<(i32, Weight)>, u64), Halt>
	{
		let totals = match self.grows(chain)
		{
			None =>
			{
				meter.charge(1, 1)?;
				vec![(chain.total(), self.padding())]
			},
			Some(Tails::Folds(folds)) =>
			{
				let len = folds.len() as u64;
				meter.charge(len + chain.dice.len() as u64, len)?;
				let fold = chain.fold();
				let mut totals = BTreeMap::<i32, Weight>::new();
				for (tail, weight) in folds
				{
					*totals.entry(fold.add(*tail).total()).or_default() +=
						weight;
				}
				meter.free(len - totals.len() as u64);
				totals.into_iter().collect()
			},
			Some(Tails::Tails(_)) => unreachable!("the chains stay whole")
		};
		Ok((totals, 0))
	}

	/// Answer how the explode may grow the specified chain still kept.
	///
	/// # Parameters
	/// - `chain`: The chain.
	///
	/// # Returns
	/// The tails that the chain may grow, or `None` if it keeps its dice.
	///
	/// # Panics
	/// As [`chains`](Self::chains) does.
	fn grows(&self, chain: &Chain) -> Option<&Tails>
	{
		let matches = |criterion: &Criterion| {
			chain.end.is_some_and(|end| criterion.matches(end))
		};
		match self
		{
			Burst::Extend(extend) if matches(extend.criterion) =>
			{
				Some(&extend.tails)
			},
			Burst::Extend(_) => None,
			Burst::Unbounded(criterion) if matches(criterion) =>
			{
				unreachable!(
					"the pass refuses every unlimited explode that may extend a \
					 chain"
				)
			},
			Burst::Unbounded(_) => None,
			Burst::Nothing | Burst::Saturate(_) =>
			{
				unreachable!(
					"the setup handles explodes that act on none alone"
				)
			}
		}
	}

	/// Answer the branches of each chain still kept: `Fⁿ` for a limited
	/// explode, and one for an unlimited one.
	///
	/// # Returns
	/// The branches.
	fn padding(&self) -> Weight
	{
		match self
		{
			Burst::Extend(extend) => extend.padding.clone(),
			_ => Weight::ONE
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Rerolls.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The multisets of the replacements of the specified number of dice that a
/// reroll matched, by that number.
type Draws = HashMap<i32, Multisets<i32>>;

/// Answer every chain that the specified chain still kept may become after a
/// reroll, which rerolls each of its kept dice that it matches alone, so that
/// the replacement of its end becomes its new end. Each die that the reroll
/// does not match keeps its value along all `G` branches of its rerolls, so
/// the chains weigh `Gᵈ` in all, for a chain of `d` kept dice, or one for a
/// chain that a saturated die decides, which the reroll leaves alone.
///
/// # Parameters
/// - `chain`: The chain.
/// - `redraw`: The redraw.
/// - `draws`: The multisets of the replacements drawn so far, which this
///   consults and extends.
/// - `meter`: The meter, to which the chains, and any new replacements, stay
///   charged.
///
/// # Returns
/// The chains, in ascending order, and the number of kept dice of the chain,
/// `d`.
///
/// # Errors
/// [`Halt`] if the chains would exceed the budget or are cancelled.
///
/// # Notes
/// `rerolled`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `weighted_rerolled` and `sum_rerolled`, in `lean/spec/XdySpec/Stacks.lean`,
/// prove that the chains, over their weights, are the specification's reroll
/// of the chain (`rerollChain`), and weigh `Gᵈ` in all.
fn rerolled(
	chain: &Chain,
	redraw: &Redraw,
	draws: &mut Draws,
	meter: &mut Meter
) -> Result<(Vec<(Chain, Weight)>, u64), Halt>
{
	let Some(end) = chain.end
	else
	{
		meter.charge(1, chain.cells())?;
		return Ok((vec![(chain.clone(), Weight::ONE)], 0))
	};
	let distinct = chain.dice.len() as u64;
	meter.charge(distinct, distinct + 1)?;
	// The kept dice other than the end that the reroll does not match, and the
	// number that it does.
	let mut survivors = Vec::new();
	let mut matched = 0;
	for &(value, copies) in &chain.dice
	{
		let copies = copies - (value == end) as i32;
		match redraw.criterion.matches(value)
		{
			_ if copies == 0 =>
			{},
			true => matched += copies,
			false => survivors.push((value, copies))
		}
	}
	let replaced = redraw.criterion.matches(end);
	if !replaced
	{
		survivors = join(&survivors, &[(end, 1)]);
	}
	let length = chain.length();
	let unmatched = length - matched as u64 - replaced as u64;
	meter.charge(unmatched, 0)?;
	let factor = redraw.padding.pow(unmatched as u32);
	let drawn = match draws.entry(matched)
	{
		Entry::Occupied(entry) => entry.into_mut(),
		Entry::Vacant(entry) =>
		{
			entry.insert(multisets(&redraw.law, matched as usize, meter)?)
		},
	};
	let ends = match replaced
	{
		true => redraw.law.clone(),
		false => vec![(end, Weight::ONE)]
	};
	// Each multiset of replacements joins the survivors, and then the end.
	let most = drawn
		.iter()
		.map(|(replacements, _)| replacements.len() as u128)
		.max()
		.unwrap_or(0);
	let len = drawn.len() as u128 * ends.len() as u128;
	let cells = charge_of(len * (3 + survivors.len() as u128 + most));
	meter.charge(charge_of(len), cells)?;
	let mut chains = BTreeMap::<Chain, Weight>::new();
	for (replacements, weight) in drawn.iter()
	{
		let dice = join(&survivors, replacements);
		let weight = weight * &factor;
		for (end, end_weight) in &ends
		{
			let dice = match replaced
			{
				true => join(&dice, &[(*end, 1)]),
				false => dice.clone()
			};
			*chains.entry(Chain::of(dice, *end)).or_default() +=
				&weight * end_weight;
		}
	}
	let chains = chains.into_iter().collect::<Vec<_>>();
	let occupied = chains.iter().map(|(chain, _)| chain.cells()).sum::<u64>();
	meter.free(cells - occupied + distinct + 1);
	Ok((chains, length))
}

////////////////////////////////////////////////////////////////////////////////
//                                Transforms.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Answer the number of cells that a law over values occupies: those of each
/// value, with its weight.
///
/// # Type parameters
/// - `T`: The type of the values.
///
/// # Parameters
/// - `law`: The law.
///
/// # Returns
/// The number of cells.
fn law_cells<T: Element>(law: &[(T, Weight)]) -> u64
{
	law.iter().map(|(value, _)| value.cells()).sum()
}

/// The law of what a chain becomes after a reroll or an explode, and the
/// number of kept dice of the chain by which a reroll pads it.
type Become<T> = (Vec<(T, Weight)>, u64);

/// Transform the law of independent and alike chains, none of which has been
/// dropped, by the specified law of what each becomes, padding each to the
/// most kept dice of any, so that every chain weighs the same in all.
///
/// # Type parameters
/// - `T`: The type of what a chain becomes: a chain, or a total.
///
/// # Parameters
/// - `law`: The distinct chains that each may be, in ascending order, each with
///   its branches.
/// - `padding`: The branches that a reroll adds to each kept die, or one for an
///   explode.
/// - `becomes`: The law of what a chain becomes, which charges it to the meter.
/// - `meter`: The meter, to which the new law stays charged.
///
/// # Returns
/// The new law, in ascending order.
///
/// # Errors
/// [`Halt`] if the transform would exceed the budget or is cancelled.
///
/// # Notes
/// `transformLaw`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `weighted_transformLaw_rerolled`, in `lean/spec/XdySpec/Stacks.lean`,
/// proves that the law of each chain, rerolled and padded, is the reroll of
/// the law of each; `Setup.law_burst_alike`, in
/// `lean/spec/XdySpec/Explode.lean`, proves the same of an explode.
fn transform_law<T: Element>(
	law: &[(Chain, Weight)],
	padding: &Weight,
	mut becomes: impl FnMut(&Chain, &mut Meter) -> Result<Become<T>, Halt>,
	meter: &mut Meter
) -> Result<Vec<(T, Weight)>, Halt>
{
	let mut laws = Vec::with_capacity(law.len());
	let mut scratch = 0u64;
	for (chain, _) in law
	{
		let (each, dice) = becomes(chain, meter)?;
		scratch = scratch.saturating_add(law_cells(&each));
		laws.push((each, dice));
	}
	let most = laws.iter().map(|(_, dice)| *dice).max().unwrap_or(0);
	// Each chain's law is padded and weighed, and its entries merged.
	let entries = laws
		.iter()
		.map(|(each, _)| each.len() as u128)
		.sum::<u128>();
	let steps = entries + law.len() as u128 * (1 + most as u128);
	meter.charge(charge_of(steps), scratch)?;
	let mut transformed = BTreeMap::<T, Weight>::new();
	for ((_, weight), (each, dice)) in law.iter().zip(laws)
	{
		let factor = weight * padding.pow((most - dice) as u32);
		for (value, branches) in each
		{
			*transformed.entry(value).or_default() += &factor * branches;
		}
	}
	let transformed = transformed.into_iter().collect::<Vec<_>>();
	meter.free(2 * scratch - law_cells(&transformed));
	Ok(transformed)
}

/// Transform each multiset of the chains of a roll, after its lowest `lowest`
/// and its highest `highest` chains are dropped, by the specified law of what
/// each chain still kept becomes, and answer the multisets of what the chains
/// still kept become. Each group of `c` copies of a chain becomes each multiset
/// of `c` independent draws of its law, and the multisets of the groups join.
/// A multiset whose chains hold `d` kept dice in all is padded to the most of
/// any, `D`, by `G^(D − d)`, so that every multiset weighs the same in all.
///
/// # Type parameters
/// - `T`: The type of what a chain becomes: a chain, or a total.
///
/// # Parameters
/// - `outcomes`: The multisets of the chains of the roll, each in ascending
///   order, with its branches.
/// - `lowest`: The number of lowest chains dropped, already clamped.
/// - `highest`: The number of highest chains dropped, already clamped, such
///   that at least one chain is still kept.
/// - `padding`: The branches that a reroll adds to each kept die, `G`, or one
///   for an explode.
/// - `becomes`: The law of what a chain becomes, which charges it to the meter.
/// - `meter`: The meter, to which the new multisets stay charged.
///
/// # Returns
/// The new multisets, in ascending order, each with its branches.
///
/// # Errors
/// [`Halt`] if the transform would exceed the budget or is cancelled.
///
/// # Notes
/// `transformPool`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `weighted_transformPool`, in `lean/spec/XdySpec/Stacks.lean`, proves that
/// the new multisets, over their weights, are each multiset of the chains
/// sliced, each copy of each chain still kept drawn independently from the law
/// of what it becomes, if every multiset weighs alike after its padding.
fn transform_pool<T: Element>(
	outcomes: &Multisets<Chain>,
	lowest: i32,
	highest: i32,
	padding: &Weight,
	mut becomes: impl FnMut(&Chain, &mut Meter) -> Result<Become<T>, Halt>,
	meter: &mut Meter
) -> Result<Multisets<T>, Halt>
{
	// The cells charged here that the new multisets do not occupy.
	let mut scratch = 0u64;
	let mut laws = BTreeMap::<Chain, Become<T>>::new();
	let mut kept = Vec::with_capacity(outcomes.len());
	for (chains, branches) in outcomes
	{
		let cells = multiset_cells(chains);
		meter.charge(chains.len() as u64, cells)?;
		scratch = scratch.saturating_add(cells);
		let groups = slice(chains, lowest, highest);
		let mut dice = 0u64;
		for (chain, copies) in &groups
		{
			if !laws.contains_key(chain)
			{
				let law = becomes(chain, meter)?;
				scratch = scratch.saturating_add(law_cells(&law.0));
				laws.insert(chain.clone(), law);
			}
			dice = dice.saturating_add(laws[chain].1 * *copies as u64);
		}
		kept.push((groups, dice, branches));
	}
	let most = kept.iter().map(|(_, dice, _)| *dice).max().unwrap_or(0);
	let mut draws = BTreeMap::<(Chain, i32), Multisets<T>>::new();
	let mut joined = 0u64;
	let mut pool = BTreeMap::<Vec<(T, i32)>, Weight>::new();
	for (groups, dice, branches) in kept
	{
		meter.charge(1 + most - dice, 1)?;
		joined += 1;
		let mut multisets_so_far = vec![(
			Vec::<(T, i32)>::new(),
			branches * padding.pow((most - dice) as u32)
		)];
		for (chain, copies) in groups
		{
			let key = (chain, copies);
			if !draws.contains_key(&key)
			{
				let law = &laws[&key.0].0;
				let drawn = multisets(law, copies as usize, meter)?;
				scratch = scratch.saturating_add(
					drawn
						.iter()
						.map(|(multiset, _)| multiset_cells(multiset))
						.sum()
				);
				draws.insert(key.clone(), drawn);
			}
			let drawn = &draws[&key];
			let most_so_far = multisets_so_far
				.iter()
				.map(|(multiset, _)| multiset_cells(multiset))
				.max()
				.unwrap_or(1);
			let most_drawn = drawn
				.iter()
				.map(|(multiset, _)| multiset_cells(multiset))
				.max()
				.unwrap_or(1);
			let len = multisets_so_far.len() as u128 * drawn.len() as u128;
			let cells = charge_of(len * (most_so_far + most_drawn) as u128);
			meter.charge(charge_of(len), cells)?;
			joined = joined.saturating_add(cells);
			multisets_so_far = multisets_so_far
				.iter()
				.flat_map(|(multiset, weight)| {
					drawn.iter().map(move |(more, branches)| {
						(join(multiset, more), weight * branches)
					})
				})
				.collect();
		}
		for (multiset, weight) in multisets_so_far
		{
			*pool.entry(multiset).or_default() += weight;
		}
	}
	let pool = pool.into_iter().collect::<Vec<_>>();
	let occupied = pool
		.iter()
		.map(|(multiset, _)| multiset_cells(multiset))
		.sum::<u64>();
	meter.free(scratch.saturating_add(joined) - occupied);
	Ok(pool)
}

/// Answer the chains still kept of a multiset of chains, after its lowest
/// `lowest` and its highest `highest` are dropped.
///
/// # Parameters
/// - `chains`: The distinct chains, in ascending order, each with its positive
///   number of copies.
/// - `lowest`: The number of lowest chains dropped, already clamped.
/// - `highest`: The number of highest chains dropped, already clamped.
///
/// # Returns
/// The distinct chains still kept, in ascending order, each with its positive
/// number of copies.
///
/// # Notes
/// `slice`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `coe_window_slice`, in `lean/spec/XdySpec/Stacks.lean`, proves that it
/// keeps the window of the chains sorted by their totals and then their dice,
/// as every drop and keep orders them, which depends only on their multiset.
fn slice(
	chains: &[(Chain, i32)],
	lowest: i32,
	highest: i32
) -> Vec<(Chain, i32)>
{
	let count = chains.iter().map(|(_, copies)| *copies).sum::<i32>();
	let kept = lowest..count - highest;
	let mut slice = Vec::new();
	let mut j = 0;
	for (chain, copies) in chains
	{
		let placed = (kept.end.min(j + copies) - kept.start.max(j)).max(0);
		j += copies;
		if placed > 0
		{
			slice.push((chain.clone(), placed));
		}
	}
	slice
}

////////////////////////////////////////////////////////////////////////////////
//                                  Stacks.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The law of what each distinct chain of [stacks](Roll::Stacks) has become
/// since its multisets were last joined: the distinct chains that each copy
/// of it may now be, independently of every other, in ascending order, each
/// with its branches. A chain that is still itself has none.
pub(super) type Laws = BTreeMap<Chain, Vec<(Chain, Weight)>>;

/// Answer the number of cells that the laws of stacks occupy: those of each
/// chain, and of its law.
///
/// # Parameters
/// - `laws`: The laws.
///
/// # Returns
/// The number of cells.
pub(super) fn laws_cells(laws: &Laws) -> u64
{
	laws.iter()
		.map(|(chain, law)| chain.cells() + law_cells(law))
		.sum()
}

/// Answer the chains that the chains of stacks may now be, kept or dropped:
/// those of the law of each distinct chain of the pool, or the chain itself
/// if it is still itself.
///
/// # Parameters
/// - `pool`: The multisets of the stacks.
/// - `laws`: The laws of the stacks.
///
/// # Returns
/// The chains, perhaps with repeats.
fn current<'a>(
	pool: &'a Multisets<Chain>,
	laws: &'a Laws
) -> impl Iterator<Item = &'a Chain>
{
	pool.iter()
		.flat_map(|(chains, _)| chains)
		.flat_map(|(chain, _)| match laws.get(chain)
		{
			Some(law) => law.iter().map(|(chain, _)| chain).collect::<Vec<_>>(),
			None => vec![chain]
		})
}

/// Compose the law of what a chain of stacks has become with the specified law
/// of what each chain becomes after a reroll or an explode: each chain that it
/// may have become becomes the law that `becomes` answers, the first time
/// that it arises, padded to the most kept dice of any, so that the composed
/// law weighs the same whichever it had become.
///
/// # Type parameters
/// - `T`: The type of what a chain becomes: a chain, or a total.
///
/// # Parameters
/// - `chain`: The chain of the pool of the stacks.
/// - `laws`: The laws of the stacks.
/// - `padding`: The branches that a reroll adds to each kept die, `G`, or one
///   for an explode.
/// - `became`: The law of what each chain becomes, by chain, which this
///   consults and extends, and whose laws stay charged to the meter.
/// - `becomes`: The law of what a chain becomes, which charges it to the meter.
/// - `meter`: The meter, to which the composed law stays charged.
///
/// # Returns
/// The composed law, in ascending order, and the most kept dice of any chain
/// that the chain may have become, by which a reroll pads the multisets that
/// hold it.
///
/// # Errors
/// [`Halt`] if the composition would exceed the budget or is cancelled.
///
/// # Notes
/// `compose`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, answering what
/// `becomes` answers of a chain that is still itself; the composition of the
/// laws of stacks has no proof yet.
fn compose<T: Element>(
	chain: &Chain,
	laws: &Laws,
	padding: &Weight,
	became: &mut BTreeMap<Chain, Become<T>>,
	becomes: &mut impl FnMut(&Chain, &mut Meter) -> Result<Become<T>, Halt>,
	meter: &mut Meter
) -> Result<Become<T>, Halt>
{
	let itself;
	let law = match laws.get(chain)
	{
		Some(law) => law.as_slice(),
		None =>
		{
			meter.charge(1, chain.cells())?;
			itself = [(chain.clone(), Weight::ONE)];
			&itself[..]
		}
	};
	for (each, _) in law
	{
		if !became.contains_key(each)
		{
			let law = becomes(each, meter)?;
			became.insert(each.clone(), law);
		}
	}
	let most = law
		.iter()
		.map(|(each, _)| became[each].1)
		.max()
		.unwrap_or(0);
	// Each law is padded and weighed, and its entries merged.
	let entries = law
		.iter()
		.map(|(each, _)| became[each].0.len() as u128)
		.sum::<u128>();
	let steps = entries + law.len() as u128 * (1 + most as u128);
	let cells = law
		.iter()
		.map(|(each, _)| law_cells(&became[each].0))
		.sum::<u64>();
	meter.charge(charge_of(steps), cells)?;
	let mut composed = BTreeMap::<T, Weight>::new();
	for (each, weight) in law
	{
		let (law, dice) = &became[each];
		let factor = weight * padding.pow((most - dice) as u32);
		for (value, branches) in law
		{
			*composed.entry(value.clone()).or_default() += &factor * branches;
		}
	}
	let composed = composed.into_iter().collect::<Vec<_>>();
	let identity = match laws.contains_key(chain)
	{
		true => 0,
		false => chain.cells()
	};
	meter.free(cells - law_cells(&composed) + identity);
	Ok((composed, most))
}

/// Transform stacks by the specified law of what each chain becomes after a
/// reroll or an explode, chain by chain, without joining them: compose the
/// law of what each distinct chain of the pool has become with it, and pad
/// each multiset by `G^(D − d)`, where `d` is the sum, over its copies of its
/// chains, of the most kept dice that each may have become, and `D` the most
/// of any multiset. Every chain that a copy may have become is padded to the
/// most of its law, so that every multiset weighs the same in all, as the
/// multisets that [`transform_pool`] would join from the joined stacks do, and
/// with the same weights, since the copies are independent.
///
/// # Parameters
/// - `pool`: The multisets of the stacks, each in ascending order, with its
///   branches.
/// - `laws`: The laws of the stacks.
/// - `padding`: The branches that a reroll adds to each kept die, `G`, or one
///   for an explode.
/// - `becomes`: The law of what a chain becomes, which charges it to the meter.
/// - `meter`: The meter, to which the new stacks stay charged.
///
/// # Returns
/// The new stacks.
///
/// # Errors
/// [`Halt`] if the transform would exceed the budget or is cancelled.
///
/// # Notes
/// `transformStacks`, in `lean/spec/XdySpec/Chains.lean`, mirrors this.
/// `Setup.law_burst` and `Setup.law_reroll`, in
/// `lean/spec/XdySpec/Stacks.lean`, prove the explodes and rerolls of stacks
/// without laws, such as [`kept_pool`] leaves; the composition of the laws of
/// stacks has no proof yet.
fn transform_stacks(
	pool: &Multisets<Chain>,
	laws: &Laws,
	padding: &Weight,
	mut becomes: impl FnMut(&Chain, &mut Meter) -> Result<Become<Chain>, Halt>,
	meter: &mut Meter
) -> Result<Roll, Halt>
{
	let mut became = BTreeMap::<Chain, Become<Chain>>::new();
	let mut composed = BTreeMap::<Chain, Become<Chain>>::new();
	let mut dice = Vec::with_capacity(pool.len());
	for (chains, _) in pool
	{
		meter.charge(chains.len() as u64, 0)?;
		let mut sum = 0u64;
		for (chain, copies) in chains
		{
			if !composed.contains_key(chain)
			{
				meter.charge(0, chain.cells())?;
				let law = compose(
					chain,
					laws,
					padding,
					&mut became,
					&mut becomes,
					meter
				)?;
				composed.insert(chain.clone(), law);
			}
			sum = sum.saturating_add(composed[chain].1 * *copies as u64);
		}
		dice.push(sum);
	}
	// Each multiset is copied and padded.
	let most = dice.iter().copied().max().unwrap_or(0);
	let steps = dice
		.iter()
		.map(|dice| 1 + (most - dice) as u128)
		.sum::<u128>();
	meter.charge(charge_of(steps), stacks_cells(pool))?;
	let pool = pool
		.iter()
		.zip(dice)
		.map(|((chains, branches), dice)| {
			(chains.clone(), branches * padding.pow((most - dice) as u32))
		})
		.collect();
	meter.free(became.values().map(|(law, _)| law_cells(law)).sum());
	Ok(Roll::Stacks {
		pool,
		laws: composed
			.into_iter()
			.map(|(chain, (law, _))| (chain, law))
			.collect()
	})
}

/// Answer the multisets of the chains still kept of the specified multisets of
/// chains, after their lowest `lowest` and their highest `highest` are
/// dropped, merging those that coincide, as the pool of stacks whose chains
/// are still themselves.
///
/// # Parameters
/// - `outcomes`: The multisets of the chains, each in ascending order, with its
///   branches.
/// - `lowest`: The number of lowest chains dropped, already clamped.
/// - `highest`: The number of highest chains dropped, already clamped, such
///   that at least one chain is still kept.
/// - `meter`: The meter, to which the multisets stay charged.
///
/// # Returns
/// The multisets of the chains still kept, in ascending order, each with its
/// branches.
///
/// # Errors
/// [`Halt`] if the multisets would exceed the budget or are cancelled.
///
/// # Notes
/// `keptPool`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `Setup.map_keptChains_law`, in `lean/spec/XdySpec/Stacks.lean`, proves that
/// the kept pool of a roll's [stacks](Roll::stacks), after a setup's drops and
/// keeps, is the law of the chains still kept of the records that the setup
/// describes.
fn kept_pool(
	outcomes: &Multisets<Chain>,
	lowest: i32,
	highest: i32,
	meter: &mut Meter
) -> Result<Multisets<Chain>, Halt>
{
	let mut charged = 0u64;
	let mut pool = BTreeMap::<Vec<(Chain, i32)>, Weight>::new();
	for (chains, branches) in outcomes
	{
		let cells = multiset_cells(chains);
		meter.charge(chains.len() as u64, cells)?;
		charged = charged.saturating_add(cells);
		*pool.entry(slice(chains, lowest, highest)).or_default() += branches;
	}
	let pool = pool.into_iter().collect::<Vec<_>>();
	meter.free(charged - stacks_cells(&pool));
	Ok(pool)
}

/// Join stacks: answer each multiset of the chains that they hold, with its
/// branches, by joining independent draws of the law of each chain of each
/// multiset of the pool, one for each copy, as [`transform_pool`] does.
///
/// # Parameters
/// - `pool`: The multisets of the stacks, each in ascending order, with its
///   branches.
/// - `laws`: The laws of the stacks.
/// - `meter`: The meter, to which the multisets stay charged.
///
/// # Returns
/// The multisets, in ascending order, each with its branches.
///
/// # Errors
/// [`Halt`] if the multisets would exceed the budget or are cancelled.
///
/// # Notes
/// `joined`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and `Roll.law`,
/// in `lean/spec/XdySpec/Mixture.lean`, weighs each multiset of stacks by its
/// branches times those of the draws that this joins (`stackWeight`).
/// `weighted_joined`, in `lean/spec/XdySpec/Stacks.lean`, proves that the
/// multisets, over their weights, are that law, so that the joins that stacks
/// defer until a drop or a split needs them are the eager ones.
fn joined(
	pool: &Multisets<Chain>,
	laws: &Laws,
	meter: &mut Meter
) -> Result<Multisets<Chain>, Halt>
{
	if laws.is_empty()
	{
		// Every chain is still itself.
		meter.charge(pool.len() as u64, stacks_cells(pool))?;
		return Ok(pool.clone())
	}
	transform_pool(
		pool,
		0,
		0,
		&Weight::ONE,
		|chain, meter| match laws.get(chain)
		{
			Some(law) =>
			{
				meter.charge(law.len() as u64, law_cells(law))?;
				Ok((law.clone(), 0))
			},
			None =>
			{
				meter.charge(1, chain.cells())?;
				Ok((vec![(chain.clone(), Weight::ONE)], 0))
			}
		},
		meter
	)
}

/// Answer the multisets of the totals of the chains that stacks hold, with
/// their branches, by joining independent draws of the law of the total of
/// each chain of each multiset of the pool, one for each copy, as
/// [`transform_pool`] does, and so without joining the chains themselves.
///
/// # Parameters
/// - `pool`: The multisets of the stacks, each in ascending order, with its
///   branches.
/// - `laws`: The laws of the stacks.
/// - `meter`: The meter, to which the multisets stay charged.
///
/// # Returns
/// The multisets of the totals, in ascending order, each with its branches.
///
/// # Errors
/// [`Halt`] if the multisets would exceed the budget or are cancelled.
///
/// # Notes
/// `joinedTotals`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
/// `Roll.map_results_collapse`, in `lean/spec/XdySpec/Stacks.lean`, proves
/// that the pool of these multisets, which [`Roll::collapse`] leaves, fills
/// records with the results of those that the stacks fill, as a multiset.
fn joined_totals(
	pool: &Multisets<Chain>,
	laws: &Laws,
	meter: &mut Meter
) -> Result<Multisets<i32>, Halt>
{
	transform_pool(
		pool,
		0,
		0,
		&Weight::ONE,
		|chain, meter| match laws.get(chain)
		{
			Some(law) =>
			{
				let len = law.len() as u64;
				meter.charge(len, len)?;
				let mut totals = BTreeMap::<i32, Weight>::new();
				for (chain, weight) in law
				{
					*totals.entry(chain.total()).or_default() += weight;
				}
				meter.free(len - totals.len() as u64);
				Ok((totals.into_iter().collect(), 0))
			},
			None =>
			{
				meter.charge(1, 1)?;
				Ok((vec![(chain.total(), Weight::ONE)], 0))
			}
		},
		meter
	)
}

////////////////////////////////////////////////////////////////////////////////
//                                   Rolls.                                   //
////////////////////////////////////////////////////////////////////////////////

impl Roll
{
	/// Explode the chains still kept of the roll, after its lowest `lowest` and
	/// its highest `highest` chains are dropped, as the burst directs, which
	/// neither does nothing nor saturates. If none has been dropped and the
	/// roll is a range, dice, or i.i.d. chains, then its chains are
	/// independent and alike, and the explode transforms the law of each. If
	/// none has been dropped and the roll is stacks, then the explode
	/// transforms the law of what each chain of their pool has become,
	/// without joining them, into stacks if `whole`, and otherwise joins the
	/// totals of those laws. Otherwise, it transforms each multiset of the
	/// chains still kept, which conditions exactly on them: into the pool of
	/// stacks whose laws it transforms if `whole`, and into a pool of totals
	/// otherwise. Either way, the chains that the explode does not extend keep
	/// their dice, and the total of the roll grows by the padding of each
	/// chain still kept.
	///
	/// # Parameters
	/// - `lowest`: The number of lowest chains dropped, already clamped.
	/// - `highest`: The number of highest chains dropped, already clamped, such
	///   that at least one chain is still kept.
	/// - `burst`: The burst.
	/// - `whole`: Whether to keep the chains whole, for a later reroll or
	///   explode, or reduce them to their totals.
	/// - `meter`: The meter, to which the exploded roll stays charged.
	///
	/// # Returns
	/// The exploded roll, whose chains are not dropped: i.i.d. chains or
	/// stacks if `whole`, and custom dice or a pool of totals otherwise.
	///
	/// # Errors
	/// [`Halt`] if the explode would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `Roll.explode`, in `lean/spec/XdySpec/Explode.lean`, mirrors this,
	/// `Roll.explodeStacks` its second arm, and `Roll.explodeMultisets` its
	/// last. `Setup.law_burst_alike` proves the law of the first arm, and
	/// `Setup.law_burst`, in `lean/spec/XdySpec/Stacks.lean`, that of the last
	/// on every roll but stacks; the explodes of stacks have no proof yet.
	pub(super) fn explode(
		&self,
		lowest: i32,
		highest: i32,
		burst: &Burst,
		whole: bool,
		meter: &mut Meter
	) -> Result<Roll, Halt>
	{
		meter.charge(1, 1)?;
		match (self, lowest, highest)
		{
			(
				Roll::Range { .. }
				| Roll::Standard { .. }
				| Roll::Custom { .. }
				| Roll::Chains { .. },
				0,
				0
			) =>
			{
				let (count, law) = self.chain_law(meter)?;
				let roll = match whole
				{
					true => Roll::Chains {
						count,
						law: transform_law(
							&law,
							&Weight::ONE,
							|chain, meter| burst.chains(chain, meter),
							meter
						)?
					},
					false => Roll::Custom {
						count,
						faces: transform_law(
							&law,
							&Weight::ONE,
							|chain, meter| burst.totals(chain, meter),
							meter
						)?
					}
				};
				meter.free(law_cells(&law));
				Ok(roll)
			},
			(Roll::Stacks { pool, laws }, 0, 0) => match whole
			{
				true => transform_stacks(
					pool,
					laws,
					&Weight::ONE,
					|chain, meter| burst.chains(chain, meter),
					meter
				),
				false =>
				{
					// Each chain of the pool reduces to the totals of what it
					// has become, exploded, without joining the chains.
					let mut became = BTreeMap::new();
					let pool = transform_pool(
						pool,
						0,
						0,
						&Weight::ONE,
						|chain, meter| {
							compose(
								chain,
								laws,
								&Weight::ONE,
								&mut became,
								&mut |chain, meter| burst.totals(chain, meter),
								meter
							)
						},
						meter
					)?;
					meter.free(
						became.values().map(|(law, _)| law_cells(law)).sum()
					);
					Ok(Roll::Pool(pool))
				}
			},
			_ =>
			{
				let outcomes = self.stacks(meter)?;
				let roll = match whole
				{
					true =>
					{
						let pool =
							kept_pool(&outcomes, lowest, highest, meter)?;
						let roll = transform_stacks(
							&pool,
							&Laws::new(),
							&Weight::ONE,
							|chain, meter| burst.chains(chain, meter),
							meter
						)?;
						meter.free(stacks_cells(&pool));
						roll
					},
					false => Roll::Pool(transform_pool(
						&outcomes,
						lowest,
						highest,
						&Weight::ONE,
						|chain, meter| burst.totals(chain, meter),
						meter
					)?)
				};
				meter.free(stacks_cells(&outcomes));
				Ok(roll)
			}
		}
	}

	/// Reroll the chains still kept of the roll, i.i.d. chains or stacks,
	/// after its lowest `lowest` and its highest `highest` chains are dropped,
	/// as the redraw directs. If none has been dropped and the roll is i.i.d.
	/// chains, then the reroll transforms the law of each, and if it is
	/// stacks, the law of what each chain of their pool has become, without
	/// joining them; otherwise, it transforms the law of each chain still kept
	/// of each multiset of the chains still kept, into stacks. Either way, the
	/// total of the roll grows by the padding of the most kept dice of any
	/// chain, or multiset, as [`rerolled`] and [`transform_stacks`] pad them.
	///
	/// # Parameters
	/// - `lowest`: The number of lowest chains dropped, already clamped.
	/// - `highest`: The number of highest chains dropped, already clamped, such
	///   that at least one chain is still kept.
	/// - `redraw`: The redraw.
	/// - `meter`: The meter, to which the rerolled roll stays charged.
	///
	/// # Returns
	/// The rerolled roll, whose chains are not dropped: i.i.d. chains or
	/// stacks.
	///
	/// # Errors
	/// [`Halt`] if the reroll would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the roll is neither i.i.d. chains nor stacks.
	///
	/// # Notes
	/// `Roll.rerollChains`, in `lean/spec/XdySpec/Chains.lean`, mirrors this,
	/// and `Setup.law_reroll`, in `lean/spec/XdySpec/Stacks.lean`, proves the
	/// laws of the first and last arms on i.i.d. chains; the composition of the
	/// laws of stacks has no proof yet.
	pub(super) fn reroll_chains(
		&self,
		lowest: i32,
		highest: i32,
		redraw: &Redraw,
		meter: &mut Meter
	) -> Result<Roll, Halt>
	{
		meter.charge(1, 1)?;
		let mut draws = Draws::new();
		match (self, lowest, highest)
		{
			(Roll::Chains { count, law }, 0, 0) => Ok(Roll::Chains {
				count: *count,
				law: transform_law(
					law,
					&redraw.padding,
					|chain, meter| rerolled(chain, redraw, &mut draws, meter),
					meter
				)?
			}),
			(Roll::Stacks { pool, laws }, 0, 0) => transform_stacks(
				pool,
				laws,
				&redraw.padding,
				|chain, meter| rerolled(chain, redraw, &mut draws, meter),
				meter
			),
			(Roll::Chains { .. } | Roll::Stacks { .. }, ..) =>
			{
				let outcomes = self.stacks(meter)?;
				let pool = kept_pool(&outcomes, lowest, highest, meter)?;
				meter.free(stacks_cells(&outcomes));
				let stacks = transform_stacks(
					&pool,
					&Laws::new(),
					&redraw.padding,
					|chain, meter| rerolled(chain, redraw, &mut draws, meter),
					meter
				)?;
				meter.free(stacks_cells(&pool));
				Ok(stacks)
			},
			_ => unreachable!("only chains are rerolled as chains")
		}
	}

	/// Saturate every chain still kept of the roll, which keeps its total
	/// weight, since the explode that saturates them rolls nothing.
	///
	/// # Parameters
	/// - `value`: The saturated value of the die.
	/// - `kept`: The number of chains still kept, which is positive.
	/// - `whole`: Whether to keep the chains whole, for a later reroll or
	///   explode, or reduce them to their totals.
	/// - `meter`: The meter, to which the saturated roll stays charged.
	///
	/// # Returns
	/// Stacks if `whole`, and a pool otherwise, of one multiset of `kept`
	/// copies of the saturated value, weighing the total of the roll.
	///
	/// # Errors
	/// [`Halt`] if saturating would exceed the budget or is cancelled.
	pub(super) fn saturate(
		&self,
		value: i32,
		kept: i32,
		whole: bool,
		meter: &mut Meter
	) -> Result<Roll, Halt>
	{
		// The total takes a step for each die.
		meter.charge(1 + self.dice(), 3)?;
		let total = self.total();
		Ok(match whole
		{
			true => Roll::Stacks {
				pool: vec![(vec![(Chain::saturated(value), kept)], total)],
				laws: Laws::new()
			},
			false => Roll::Pool(vec![(vec![(value, kept)], total)])
		})
	}

	/// Answer whether a saturated die decides the chains of the roll, which
	/// are then left alone by every reroll.
	///
	/// # Returns
	/// `true` if the roll is stacks of saturated chains, `false` otherwise.
	///
	/// # Notes
	/// `Roll.isSaturated`, in `lean/spec/XdySpec/Chains.lean`, mirrors this.
	pub(super) fn is_saturated(&self) -> bool
	{
		match self
		{
			Roll::Stacks { pool, laws } =>
			{
				current(pool, laws).any(|chain| chain.is_saturated())
			},
			_ => false
		}
	}

	/// Answer whether the end of some chain of the roll, kept or dropped, may
	/// show a result that the criterion matches.
	///
	/// # Parameters
	/// - `criterion`: The criterion.
	///
	/// # Returns
	/// `true` if some end may match, `false` otherwise.
	pub(super) fn shows(&self, criterion: &Criterion) -> bool
	{
		match self
		{
			Roll::Range { .. } | Roll::Standard { .. } => !self
				.replacement()
				.expect("dice show their die")
				.matching(criterion)
				.is_zero(),
			Roll::Custom { faces, .. } if faces.is_empty() =>
			{
				criterion.matches(0)
			},
			Roll::Custom { faces, .. } =>
			{
				faces.iter().any(|(face, _)| criterion.matches(*face))
			},
			Roll::Results(results) =>
			{
				results.iter().any(|(result, _)| criterion.matches(*result))
			},
			Roll::Pool(pool) => pool.iter().any(|(results, _)| {
				results.iter().any(|(result, _)| criterion.matches(*result))
			}),
			Roll::Chains { law, .. } => law.iter().any(|(chain, _)| {
				chain.end.is_some_and(|e| criterion.matches(e))
			}),
			Roll::Stacks { pool, laws } => current(pool, laws)
				.any(|chain| chain.end.is_some_and(|e| criterion.matches(e)))
		}
	}

	/// Reduce the chains of the roll to their totals, which behave as the
	/// results of dice do: i.i.d. chains to custom dice whose faces are their
	/// totals, and stacks to a pool of the multisets of their totals, which
	/// [joins](joined_totals) the laws of their totals rather than their
	/// chains. Any other roll stays as it is.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the reduced roll stays charged.
	///
	/// # Returns
	/// The reduced roll.
	///
	/// # Errors
	/// [`Halt`] if reducing would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `Roll.collapse`, in `lean/spec/XdySpec/Explode.lean`, mirrors this.
	/// `Setup.law_collapse` there proves that reducing i.i.d. chains leaves
	/// the records of a setup as every later drop, keep, and sum views them,
	/// and `Roll.map_results_collapse`, in `lean/spec/XdySpec/Stacks.lean`,
	/// that the records that reduced stacks fill hold the results of those
	/// that the stacks fill.
	pub(super) fn collapse(&self, meter: &mut Meter) -> Result<Roll, Halt>
	{
		match self
		{
			Roll::Chains { count, law } =>
			{
				let len = law.len() as u64;
				meter.charge(len, 1 + len)?;
				let mut faces = BTreeMap::<i32, Weight>::new();
				for (chain, weight) in law
				{
					*faces.entry(chain.total()).or_default() += weight;
				}
				meter.free(len - faces.len() as u64);
				Ok(Roll::Custom {
					count: *count,
					faces: faces.into_iter().collect()
				})
			},
			Roll::Stacks { pool, laws } =>
			{
				meter.charge(1, 1)?;
				Ok(Roll::Pool(joined_totals(pool, laws, meter)?))
			},
			_ =>
			{
				meter.charge(1, self.cells())?;
				Ok(self.clone())
			}
		}
	}

	/// Answer the law of one chain of a roll of independent and alike chains:
	/// a range, dice, or i.i.d. chains, and their number.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the law stays charged.
	///
	/// # Returns
	/// The number of chains, which is one for a range, and the distinct chains
	/// that each may be, in ascending order, each with its branches.
	///
	/// # Errors
	/// [`Halt`] if the law would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the roll is fixed results, a pool, or stacks.
	fn chain_law(
		&self,
		meter: &mut Meter
	) -> Result<(i32, Vec<(Chain, Weight)>), Halt>
	{
		let count = match self
		{
			Roll::Range { .. } => 1,
			Roll::Standard { count, .. } | Roll::Custom { count, .. } => *count,
			Roll::Chains { count, law } =>
			{
				let cells = law_cells(law);
				meter.charge(law.len() as u64, cells)?;
				return Ok((*count, law.clone()))
			},
			Roll::Results(_) | Roll::Pool(_) | Roll::Stacks { .. } =>
			{
				unreachable!("fixed results are not alike")
			}
		};
		let faces = match self
		{
			Roll::Custom { faces, .. } if !faces.is_empty() => faces.clone(),
			_ => self.replacement().expect("dice show their die").faces()
		};
		let len = faces.len() as u64;
		meter.charge(len, 2 * len)?;
		Ok((
			count,
			faces
				.into_iter()
				.map(|(face, weight)| (Chain::die(face), weight))
				.collect()
		))
	}

	/// Answer every outcome of the roll as multisets of chains: each multiset
	/// of the chains of its records, in ascending order, weighed by the number
	/// of branches of the roll that reach it, as [`outcomes`](Self::outcomes)
	/// answers them, where a result is the chain of one die.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the multisets stay charged.
	///
	/// # Returns
	/// The multisets, with their weights.
	///
	/// # Errors
	/// [`Halt`] if the multisets would exceed the budget or are cancelled.
	///
	/// # Notes
	/// `Roll.stacks`, in `lean/spec/XdySpec/Chains.lean`, mirrors this, and
	/// `Roll.map_keptChains_law`, in `lean/spec/XdySpec/Stacks.lean`, proves
	/// that, over their weights, the multisets are the law of the chains of
	/// the records that the roll fills.
	fn stacks(&self, meter: &mut Meter) -> Result<Multisets<Chain>, Halt>
	{
		match self
		{
			Roll::Chains { count, law } =>
			{
				multisets(law, (*count).max(0) as usize, meter)
			},
			Roll::Stacks { pool, laws } => joined(pool, laws, meter),
			_ =>
			{
				let outcomes = self.outcomes(meter)?;
				// Each result becomes the chain of one die, of one more cell.
				let results = outcomes
					.iter()
					.map(|(outcome, _)| outcome.cells() - 1)
					.sum::<u64>();
				meter.charge(results, results)?;
				Ok(outcomes
					.into_iter()
					.map(|(outcome, weight)| {
						let Roll::Results(results) = outcome
						else
						{
							unreachable!("an outcome is fixed results")
						};
						let chains = results
							.into_iter()
							.map(|(result, copies)| {
								(Chain::die(result), copies)
							})
							.collect();
						(chains, weight)
					})
					.collect())
			}
		}
	}

	/// Answer every outcome of a roll of i.i.d. chains or stacks: each
	/// multiset of its chains, as [stacks](Roll::Stacks) of one multiset of
	/// weight one, with its branches.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the outcomes stay charged.
	///
	/// # Returns
	/// The outcomes, with their weights.
	///
	/// # Errors
	/// [`Halt`] if the outcomes would exceed the budget or are cancelled.
	pub(super) fn chain_outcomes(
		&self,
		meter: &mut Meter
	) -> Result<Vec<(Roll, Weight)>, Halt>
	{
		let outcomes = self.stacks(meter)?;
		// Each multiset becomes stacks, of one more cell.
		let len = outcomes.len() as u64;
		meter.charge(len, len)?;
		Ok(outcomes
			.into_iter()
			.map(|(chains, weight)| {
				let stacks = Roll::Stacks {
					pool: vec![(chains, Weight::ONE)],
					laws: Laws::new()
				};
				(stacks, weight)
			})
			.collect())
	}
}

/// Answer the number of cells that multisets of chains occupy: those of each.
///
/// # Parameters
/// - `stacks`: The multisets, each with its weight.
///
/// # Returns
/// The number of cells.
fn stacks_cells(stacks: &Multisets<Chain>) -> u64
{
	stacks
		.iter()
		.map(|(chains, _)| multiset_cells(chains))
		.sum()
}

#[cfg(test)]
mod tests
{
	use std::collections::BTreeMap;

	use proptest::prelude::*;

	use super::{Burst, Draws, Laws, burst, joined, rerolled, transform_pool};
	use crate::{
		Criterion, Weight,
		distribution::propagation::{
			cost::{Class, Cost},
			meter::{Budget, Meter, Unobserved},
			record::{Die, Effect, Multisets, Roll, distinct_faces, redraw}
		}
	};

	/// A step of [`test_deferred_stacks_join_as_eager`]: whether it explodes
	/// or rerolls, the results that it explodes or rerolls, its limit, and the
	/// numbers of lowest and highest chains dropped before it.
	type Step = (bool, Criterion, i32, i32, i32);

	/// Answer a strategy that generates a criterion of every form over the
	/// faces of small dice.
	///
	/// # Returns
	/// The strategy.
	fn criterion() -> impl Strategy<Value = Criterion>
	{
		prop_oneof![
			(-2..=4, -2..=4).prop_map(|(a, b): (i32, i32)| {
				Criterion::within(a.min(b) as i64, a.max(b) as i64)
			}),
			(-2..=4).prop_map(|v| Criterion::Outside(v..=v)),
			prop::collection::vec(-2..=4, 0..=3).prop_map(|mut values| {
				values.sort_unstable();
				values.dedup();
				Criterion::Among(values)
			})
		]
	}

	/// Answer the multisets of the totals of the specified multisets of
	/// chains, with their branches.
	///
	/// # Parameters
	/// - `stacks`: The multisets of chains, with their branches.
	///
	/// # Returns
	/// The multisets of their totals, in ascending order, with their branches.
	fn totals(stacks: &Multisets<super::Chain>) -> Multisets<i32>
	{
		let mut pool = BTreeMap::<Vec<(i32, i32)>, Weight>::new();
		for (chains, weight) in stacks
		{
			let mut totals = BTreeMap::<i32, i32>::new();
			for (chain, copies) in chains
			{
				*totals.entry(chain.total()).or_default() += copies;
			}
			*pool.entry(totals.into_iter().collect()).or_default() += weight;
		}
		pool.into_iter().collect()
	}

	proptest! {
		/// Stacks that defer their joins, through any sequence of explodes and
		/// rerolls, with drops between them, hold exactly the multisets of
		/// chains, with exactly the branches, that joining them after every
		/// step does, as the pass once did; and their totals join alike.
		#[test]
		fn test_deferred_stacks_join_as_eager(
			faces in prop_oneof![
				(2..=4).prop_map(Die::Standard),
				prop::collection::vec(-2..=3, 1..=3)
					.prop_map(|faces| Die::Custom(distinct_faces(&faces)))
			],
			count in 1..=3,
			steps in prop::collection::vec(
				(any::<bool>(), criterion(), 1..=2, 0..=1, 0..=1),
				1..=3
			)
		)
		{
			let estimate = Cost {
				steps: 0,
				cells: 0,
				worlds: 0,
				bits: 0,
				class: Class::default()
			};
			let mut meter =
				Meter::new(Budget::UNLIMITED, 0, &estimate, &Unobserved)
					.unwrap();
			let roll = match &faces
			{
				&Die::Standard(faces) => Roll::Standard { count, faces },
				Die::Custom(faces) => Roll::Custom {
					count,
					faces: faces.clone()
				},
				Die::Range { .. } => unreachable!("no ranges are generated")
			};
			let mut eager = roll.stacks(&mut meter).unwrap();
			let mut deferred = Roll::Stacks {
				pool: eager.clone(),
				laws: Laws::new()
			};
			let mut count = count;
			for step in steps
			{
				let (explodes, criterion, limit, lowest, highest): Step = step;
				if lowest + highest >= count
				{
					continue
				}
				if explodes
				{
					let burst =
						burst(&faces, &criterion, Some(limit), true, &mut meter)
							.unwrap();
					if !matches!(burst, Burst::Extend(_))
					{
						continue
					}
					eager = transform_pool(
						&eager,
						lowest,
						highest,
						&Weight::ONE,
						|chain, meter| burst.chains(chain, meter),
						&mut meter
					)
					.unwrap();
					deferred = deferred
						.explode(lowest, highest, &burst, true, &mut meter)
						.unwrap();
				}
				else
				{
					let Effect::Redraw(redraw) =
						redraw(&faces, &criterion, Some(limit), &mut meter)
							.unwrap()
					else
					{
						continue
					};
					let mut draws = Draws::new();
					eager = transform_pool(
						&eager,
						lowest,
						highest,
						&redraw.padding,
						|chain, meter| {
							rerolled(chain, &redraw, &mut draws, meter)
						},
						&mut meter
					)
					.unwrap();
					deferred = deferred
						.reroll_chains(lowest, highest, &redraw, &mut meter)
						.unwrap();
				}
				count -= lowest + highest;
				let Roll::Stacks { pool, laws } = &deferred
				else
				{
					unreachable!("whole chains stay stacks")
				};
				prop_assert_eq!(&joined(pool, laws, &mut meter).unwrap(), &eager);
				let Roll::Pool(collapsed) = deferred.collapse(&mut meter).unwrap()
				else
				{
					unreachable!("stacks collapse to a pool")
				};
				prop_assert_eq!(collapsed, totals(&eager));
			}
		}
	}
}
