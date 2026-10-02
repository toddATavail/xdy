//! # Meters
//!
//! The forward pass does work in proportion to the supports of the
//! distributions that it combines, and keeps memory in proportion to the
//! entries of the distributions that it holds, across all of its worlds. Both
//! can grow without bound: `{n}D6` convolves `n` dice, `[1:1000000] *
//! [1:1000000]` weighs 10¹² pairs, and nested splits multiply the worlds. A
//! [meter](Meter) bounds both against a [budget](Budget), in two dimensions:
//!
//! - _Steps_, the elementary operations on weights, like the product of the
//!   weights of one pair of outcomes, or one transition of a dynamic program,
//!   whatever the widths of the weights. A roll also charges one step for each
//!   of its dice, since the total weight of `n` dice of `F` faces is `Fⁿ`,
//!   whose width grows with `n`.
//! - _Cells_, the entries alive at once: each outcome of a distribution, each
//!   setup of a rolling record, each slot of a register bank, in every world,
//!   and the working memory of the operation under way. The meter tracks the
//!   cells alive now, and the budget bounds their peak.
//!
//! Every operation charges the most steps and cells that it may take _before_
//! it takes any, so that an operation that would exceed the budget is refused
//! before it expands anything. The charges depend only on the function and its
//! bindings, so the same budget always succeeds or always refuses, and a
//! budget succeeds exactly when it covers the [usage](Usage) of an unlimited
//! build.
//!
//! The meter [reports](Progress::report) the usage so far at every charge,
//! together with the [estimate](Cost) of the whole pass, so that a caller may
//! show the progress of the pass, and cancel it. A cancelled pass stops before
//! the charged operation expands anything, and answers no distribution.

use std::{
	error::Error,
	fmt::{self, Debug, Display, Formatter},
	ops::ControlFlow
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::cost::Cost;

////////////////////////////////////////////////////////////////////////////////
//                                  Budgets.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The budget of a forward pass: the most steps that it may take, and the
/// most cells that it may keep alive at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Budget
{
	/// The most steps that the pass may take.
	pub steps: u64,

	/// The most cells that the pass may keep alive at once.
	pub cells: u64
}

impl Budget
{
	/// The budget that never refuses, since no pass can take [`u64::MAX`] steps
	/// or keep that many cells in any reasonable time or memory.
	pub const UNLIMITED: Self = Self {
		steps: u64::MAX,
		cells: u64::MAX
	};
}

/// The resources that a forward pass consumed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Usage
{
	/// The steps taken.
	pub steps: u64,

	/// The most cells alive at once.
	pub cells: u64,

	/// The most worlds alive at once, which is one if the pass never splits.
	pub worlds: u64
}

////////////////////////////////////////////////////////////////////////////////
//                                  Meters.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The meter of a forward pass, which charges its steps and cells against its
/// [budget](Budget).
///
/// The cells alive are of two kinds. The _held_ cells are those of the
/// worlds, which the pass [settles](Self::settle) after each instruction in
/// each world, and after each step of its plan. The _working_ cells are those
/// that the operation under way has charged, for its result and its scratch
/// space; an operation may [free](Self::free) scratch space that it no longer
/// needs, and settling forgets the rest, since the held cells then count
/// whatever the operation left behind.
///
/// The meter [reports](Progress::report) its usage at every charge that the
/// budget admits, and a report that breaks cancels the pass.
pub(crate) struct Meter<'a>
{
	/// The budget.
	budget: Budget,

	/// The steps taken.
	steps: u64,

	/// The cells held by the worlds, as last settled.
	held: u64,

	/// The cells charged by the operation under way.
	working: u64,

	/// The most cells alive at once.
	peak: u64,

	/// The most worlds alive at once.
	worlds: u64,

	/// The estimate of the whole pass, which the meter reports.
	estimate: &'a Cost,

	/// The recipient of the reports.
	progress: &'a dyn Progress
}

impl<'a> Meter<'a>
{
	/// Construct a meter with the specified budget, and with the specified
	/// cells already held by the one initial world. Report the initial usage.
	///
	/// # Parameters
	/// - `budget`: The budget.
	/// - `held`: The cells of the initial world.
	/// - `estimate`: The estimate of the whole pass.
	/// - `progress`: The recipient of the reports.
	///
	/// # Returns
	/// The meter.
	///
	/// # Errors
	/// - [`Exhausted`](Halt::Exhausted) if the initial world alone exceeds the
	///   budget.
	/// - [`Cancelled`](Halt::Cancelled) if the progress cancels the pass.
	pub(crate) fn new(
		budget: Budget,
		held: u64,
		estimate: &'a Cost,
		progress: &'a dyn Progress
	) -> Result<Self, Halt>
	{
		let mut meter = Self {
			budget,
			steps: 0,
			held: 0,
			working: 0,
			peak: 0,
			worlds: 1,
			estimate,
			progress
		};
		meter.charge(0, held)?;
		meter.settle(0, held);
		Ok(meter)
	}

	/// Charge the specified steps and cells, unless they would exceed the
	/// budget, in which case charge nothing. Report the usage after the
	/// charge, before the operation expands anything.
	///
	/// # Parameters
	/// - `steps`: The steps that the operation may take.
	/// - `cells`: The cells that the operation may create.
	///
	/// # Errors
	/// - [`Exhausted`](Halt::Exhausted) if the steps or the cells would exceed
	///   the budget, naming the steps first if both would.
	/// - [`Cancelled`](Halt::Cancelled) if the progress cancels the pass.
	pub(crate) fn charge(&mut self, steps: u64, cells: u64)
	-> Result<(), Halt>
	{
		let remaining = self.budget.steps - self.steps;
		if steps > remaining
		{
			return Err(Halt::Exhausted(Exhausted {
				dimension: Dimension::Steps,
				requested: steps,
				remaining,
				consumed: self.steps
			}))
		}
		let alive = self.held.saturating_add(self.working);
		let remaining = self.budget.cells.saturating_sub(alive);
		if cells > remaining
		{
			return Err(Halt::Exhausted(Exhausted {
				dimension: Dimension::Cells,
				requested: cells,
				remaining,
				consumed: alive
			}))
		}
		self.steps += steps;
		self.working = self.working.saturating_add(cells);
		self.peak = self.peak.max(alive.saturating_add(cells));
		match self.progress.report(self.usage(), self.estimate)
		{
			ControlFlow::Continue(()) => Ok(()),
			ControlFlow::Break(()) => Err(Halt::Cancelled)
		}
	}

	/// Free working cells that the operation under way charged and no longer
	/// needs.
	///
	/// # Parameters
	/// - `cells`: The cells, no more than the operation charged.
	pub(crate) fn free(&mut self, cells: u64)
	{
		debug_assert!(cells <= self.working, "free only what was charged");
		self.working -= cells;
	}

	/// Settle the operations since the last settlement: forget their working
	/// cells, and replace the cells that the locations that they touched held
	/// before with those that they hold now.
	///
	/// # Parameters
	/// - `before`: The cells that the touched locations held before the
	///   operations, which the meter holds.
	/// - `after`: The cells that the touched locations hold now, which never
	///   exceed those that they held before and those that the operations
	///   charged.
	pub(crate) fn settle(&mut self, before: u64, after: u64)
	{
		debug_assert!(before <= self.held, "the locations were held");
		debug_assert!(
			after <= before + self.working,
			"the locations keep no more than was charged: {after} > {before} + {}",
			self.working
		);
		self.held = self.held - before + after;
		self.working = 0;
	}

	/// Answer the cells that the worlds hold, as last settled.
	///
	/// # Returns
	/// The number of cells.
	#[inline]
	pub(crate) fn held(&self) -> u64 { self.held }

	/// Record the specified number of worlds alive now, after a split or a
	/// merge, which the next report includes if it is the most yet.
	///
	/// # Parameters
	/// - `worlds`: The number of worlds.
	#[inline]
	pub(crate) fn count_worlds(&mut self, worlds: usize)
	{
		self.worlds = self.worlds.max(worlds as u64);
	}

	/// Answer the resources consumed so far.
	///
	/// # Returns
	/// The usage.
	pub(crate) fn usage(&self) -> Usage
	{
		Usage {
			steps: self.steps,
			cells: self.peak,
			worlds: self.worlds
		}
	}
}

impl Debug for Meter<'_>
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		f.debug_struct("Meter")
			.field("budget", &self.budget)
			.field("steps", &self.steps)
			.field("held", &self.held)
			.field("working", &self.working)
			.field("peak", &self.peak)
			.field("worlds", &self.worlds)
			.field("estimate", &self.estimate)
			.finish_non_exhaustive()
	}
}

/// Convert a count to a charge, saturating at [`u64::MAX`], which no budget
/// but the [unlimited](Budget::UNLIMITED) one admits.
///
/// # Parameters
/// - `count`: The count.
///
/// # Returns
/// The charge.
#[inline]
pub(crate) fn charge_of(count: u128) -> u64
{
	u64::try_from(count).unwrap_or(u64::MAX)
}

////////////////////////////////////////////////////////////////////////////////
//                                 Progress.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The recipient of the progress of a forward pass, which may cancel it.
///
/// The pass [reports](Self::report) whenever it charges an operation against
/// its [budget](Budget), which precedes every operation that expands anything,
/// so a report never waits long unless one operation is enormous; a budget
/// bounds those. The usage in the reports never decreases, dimension by
/// dimension, and the last report of a pass that succeeds is exactly the usage
/// of the whole pass. Since the [estimate](Cost) bounds that usage, and bounds
/// it tightly, the ratio of the steps consumed to the steps estimated never
/// exceeds one, and is a fair fraction of the pass completed.
///
/// A closure implements this trait, and [`Unobserved`] ignores every report.
///
/// # Examples
/// Cancel a pass once it has taken more than half of its estimated steps:
///
/// ```rust
/// use std::ops::ControlFlow;
///
/// use xdy::{BuildError, Budget, Cost, Evaluator, Usage, compile};
///
/// let evaluator = Evaluator::new(compile("100D6")?);
/// let plan = evaluator.plan_distribution([])?;
/// let halfway = |consumed: Usage, estimate: &Cost| {
///     if consumed.steps > estimate.steps / 2
///     {
///         ControlFlow::Break(())
///     }
///     else
///     {
///         ControlFlow::Continue(())
///     }
/// };
/// assert_eq!(
///     plan.build(Budget::UNLIMITED, &halfway),
///     Err(BuildError::Cancelled)
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Progress
{
	/// Report the progress of the pass, before the operation that it charged
	/// last expands anything.
	///
	/// # Parameters
	/// - `consumed`: The resources consumed so far, including the charge.
	/// - `estimate`: The estimate of the whole pass.
	///
	/// # Returns
	/// [`Continue`](ControlFlow::Continue) to continue the pass, or
	/// [`Break`](ControlFlow::Break) to cancel it, whereupon it answers
	/// [`Cancelled`](crate::BuildError::Cancelled) and reports nothing more.
	fn report(&self, consumed: Usage, estimate: &Cost) -> ControlFlow<()>;
}

impl<F> Progress for F
where
	F: Fn(Usage, &Cost) -> ControlFlow<()>
{
	#[inline]
	fn report(&self, consumed: Usage, estimate: &Cost) -> ControlFlow<()>
	{
		self(consumed, estimate)
	}
}

/// The recipient that ignores every report, and never cancels the pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Unobserved;

impl Progress for Unobserved
{
	#[inline]
	fn report(&self, _consumed: Usage, _estimate: &Cost) -> ControlFlow<()>
	{
		ControlFlow::Continue(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A dimension of a [budget](Budget).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Dimension
{
	/// The steps taken.
	Steps,

	/// The cells alive at once.
	Cells
}

impl Display for Dimension
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		match self
		{
			Dimension::Steps => write!(f, "steps"),
			Dimension::Cells => write!(f, "cells")
		}
	}
}

/// The refusal of an operation whose charge would exceed the
/// [budget](Budget), before it expanded anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Exhausted
{
	/// The dimension of the budget that the charge would exceed.
	pub(crate) dimension: Dimension,

	/// The steps or cells that the operation requested.
	pub(crate) requested: u64,

	/// The steps or cells that remained of the budget, fewer than requested.
	pub(crate) remaining: u64,

	/// The steps already taken, or the cells alive, when the operation was
	/// refused.
	pub(crate) consumed: u64
}

impl Display for Exhausted
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		write!(
			f,
			"budget of {} exhausted: {} requested, {} remaining, {} consumed",
			self.dimension, self.requested, self.remaining, self.consumed
		)
	}
}

impl Error for Exhausted {}

/// The halt of a forward pass before it answers: its budget would be exceeded,
/// or its [progress](Progress) cancelled it. Either way, the pass halts before
/// the operation that it charged last expands anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Halt
{
	/// An operation would have exceeded the budget.
	Exhausted(Exhausted),

	/// The progress cancelled the pass.
	Cancelled
}

impl Display for Halt
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		match self
		{
			Halt::Exhausted(exhausted) => Display::fmt(exhausted, f),
			Halt::Cancelled => write!(f, "cancelled")
		}
	}
}

impl Error for Halt {}
