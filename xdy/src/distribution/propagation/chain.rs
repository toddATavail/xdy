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
//! [i.i.d. chains](Roll::Chains) before any drop, or as [stacks](Roll::Stacks),
//! the multisets of the chains still kept, after one. After the last such
//! reader, the chains reduce to their totals.
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
//! The Lean specification does not yet model explosion, so nothing here has a
//! proof; the tests check every law against an independent enumeration.

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
//                                   Rolls.                                   //
////////////////////////////////////////////////////////////////////////////////

impl Roll
{
	/// Explode the chains still kept of the roll, after its lowest `lowest` and
	/// its highest `highest` chains are dropped, as the burst directs, which
	/// neither does nothing nor saturates. If none has been dropped and the
	/// roll is a range, dice, or i.i.d. chains, then its chains are
	/// independent and alike, and the explode transforms the law of each;
	/// otherwise, it transforms each multiset of the chains still kept, which
	/// conditions exactly on them. Either way, the chains that the explode
	/// does not extend keep their dice, and the total of the roll grows by the
	/// padding of each chain still kept.
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
			_ =>
			{
				let outcomes = self.stacks(meter)?;
				let roll = match whole
				{
					true => Roll::Stacks(transform_pool(
						&outcomes,
						lowest,
						highest,
						&Weight::ONE,
						|chain, meter| burst.chains(chain, meter),
						meter
					)?),
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
	/// chains, then the reroll transforms the law of each; otherwise, it
	/// transforms each multiset of the chains still kept. Either way, the total
	/// of the roll grows by the padding of the most kept dice of any chain, or
	/// multiset, as [`rerolled`] and [`transform_pool`] pad them.
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
			(Roll::Chains { .. } | Roll::Stacks(_), ..) =>
			{
				let outcomes = self.stacks(meter)?;
				let stacks = transform_pool(
					&outcomes,
					lowest,
					highest,
					&redraw.padding,
					|chain, meter| rerolled(chain, redraw, &mut draws, meter),
					meter
				)?;
				meter.free(stacks_cells(&outcomes));
				Ok(Roll::Stacks(stacks))
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
			true => Roll::Stacks(vec![(
				vec![(Chain::saturated(value), kept)],
				total
			)]),
			false => Roll::Pool(vec![(vec![(value, kept)], total)])
		})
	}

	/// Answer whether a saturated die decides the chains of the roll, which
	/// are then left alone by every reroll.
	///
	/// # Returns
	/// `true` if the roll is stacks of saturated chains, `false` otherwise.
	pub(super) fn is_saturated(&self) -> bool
	{
		match self
		{
			Roll::Stacks(pool) => pool.iter().any(|(chains, _)| {
				chains.iter().any(|(c, _)| c.is_saturated())
			}),
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
			Roll::Stacks(pool) => pool.iter().any(|(chains, _)| {
				chains.iter().any(|(chain, _)| {
					chain.end.is_some_and(|e| criterion.matches(e))
				})
			})
		}
	}

	/// Reduce the chains of the roll to their totals, which behave as the
	/// results of dice do: i.i.d. chains to custom dice whose faces are their
	/// totals, and stacks to a pool of the multisets of their totals. Any other
	/// roll stays as it is.
	///
	/// # Parameters
	/// - `meter`: The meter, to which the reduced roll stays charged.
	///
	/// # Returns
	/// The reduced roll.
	///
	/// # Errors
	/// [`Halt`] if reducing would exceed the budget or is cancelled.
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
			Roll::Stacks(stacks) =>
			{
				let entries = stacks
					.iter()
					.map(|(chains, _)| 1 + chains.len() as u64)
					.sum::<u64>();
				meter.charge(entries, 1 + entries)?;
				let mut pool = BTreeMap::<Vec<(i32, i32)>, Weight>::new();
				for (chains, weight) in stacks
				{
					let mut totals = BTreeMap::<i32, i32>::new();
					for (chain, copies) in chains
					{
						*totals.entry(chain.total()).or_default() += copies;
					}
					*pool.entry(totals.into_iter().collect()).or_default() +=
						weight;
				}
				let pool = Roll::Pool(pool.into_iter().collect());
				meter.free(1 + entries - pool.cells());
				Ok(pool)
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
			Roll::Results(_) | Roll::Pool(_) | Roll::Stacks(_) =>
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
	fn stacks(&self, meter: &mut Meter) -> Result<Multisets<Chain>, Halt>
	{
		match self
		{
			Roll::Chains { count, law } =>
			{
				multisets(law, (*count).max(0) as usize, meter)
			},
			Roll::Stacks(stacks) =>
			{
				meter.charge(stacks.len() as u64, stacks_cells(stacks))?;
				Ok(stacks.clone())
			},
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
				(Roll::Stacks(vec![(chains, Weight::ONE)]), weight)
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
