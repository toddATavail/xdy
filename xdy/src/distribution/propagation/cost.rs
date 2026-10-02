//! # Cost estimates
//!
//! Before the forward pass computes any weight, a static pass over the
//! function and its [plan](Plan) predicts what the pass will charge to its
//! [meter](super::meter::Meter): an upper bound on the steps that it takes, on
//! the cells that it keeps alive at once, and on the worlds that it keeps at
//! once, together with an upper bound on the width of the total weight of its
//! answer, and the [class](Class) of the function. A
//! [budget](super::meter::Budget) that covers the estimate never refuses the
//! pass, so a caller may refuse a function, or raise its budget, before any
//! work begins.
//!
//! The estimate shadows the pass instruction by instruction, holding bounds
//! where the pass holds distributions. Each register holds its interval, from
//! the [bounds evaluator](BoundsEvaluator), and a bound on its _support_, the
//! number of its outcomes in any one world: the width of its interval, or the
//! product of the supports of the operands that made it, whichever is less.
//! Each rolling record holds bounds on its setups and their cells, and each
//! open split the number of its outcomes, so that the worlds number at most the
//! product of the outcomes of the open splits.
//!
//! The budget should decide which functions are reasonable, so the estimate
//! must be tight as well as sound. Where the pass runs an algorithm, the
//! estimate follows its shape over these bounds, rather than pricing it by one
//! formula: convolution powers square and multiply as the pass does, the order
//! statistics sum their placements face by face in closed form, the multisets
//! of a record count as binomials, and a mixture prices its setups by the
//! greatest counts and faces that they may have, and by the pairs of drop
//! counts, clamped to its dice, that keep a die. It prices the convolution
//! powers of a random count as consecutive only if the count certainly takes
//! every value of its interval, and otherwise for the costliest counts that
//! its support allows, since a count that skips values steps up its powers by
//! more than one die at once. On a pool of dice with fixed operands, with or
//! without drops, the estimate is exact.
//!
//! One shadow bounds every world by the costliest, which is loose when a
//! split fixes the operands of later rolls, since each world rolls with its
//! own. So the shadow runs in [lanes](Lane): at the split of a register, a
//! lane forks into children, each of which bounds the worlds whose value lies
//! in one bucket of the register's interval, and the estimate charges the
//! costliest outcomes that the split may have, bucket by bucket. The lanes
//! number at most [`LANES`], shared fairly among the splits, so that a split
//! of few values is priced value by value, and a split of many by buckets of
//! several values, which loosen gracefully.
//!
//! Every quantity saturates rather than overflows, so an estimate that exceeds
//! [`u64::MAX`] reports [`u64::MAX`], which no budget but the
//! [unlimited](super::meter::Budget::UNLIMITED) one admits.

use std::{cmp::Reverse, collections::BTreeSet, f64::consts::LN_2};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
	PropagationError,
	plan::{Location, Plan, Step}
};
use crate::{
	Add, AddressingMode, CanVisitInstructions as _, Div, DropHighest,
	DropLowest, EvaluationBounds, Evaluator, Exp, Function, Instruction,
	InstructionVisitor, Max, Mod, Mul, Neg, ProgramCounter, RegisterIndex,
	Return, RollCustomDice, RollRange, RollStandardDice, Sub, SumRollingRecord,
	evaluator::BoundsEvaluator
};

////////////////////////////////////////////////////////////////////////////////
//                                 Estimates.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The predicted cost of the forward pass over a function, before any weight
/// is computed. A [budget](super::meter::Budget) of at least these steps and
/// cells never refuses the pass, and every [usage](super::meter::Usage) that
/// the pass reports is within them. Every quantity saturates at [`u64::MAX`]
/// rather than overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Cost
{
	/// An upper bound on the steps that the pass takes.
	pub steps: u64,

	/// An upper bound on the cells that the pass keeps alive at once.
	pub cells: u64,

	/// An upper bound on the worlds that the pass keeps alive at once.
	pub worlds: u64,

	/// An upper bound on the number of bits of the total weight of the answer,
	/// which bounds every weight of the answer.
	pub bits: u64,

	/// The class of the function.
	pub class: Class
}

/// The class of a function: the structural traits that drive the cost of its
/// forward pass, per the design of the cost model. A function with none of
/// them is a _static pool_: rolls with fixed operands, summed and combined by
/// arithmetic, whose cost is that of convolution. The traits say nothing of
/// magnitude; how wide a function is, the [estimate](Cost) says.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Class
{
	/// Whether some drop may drop a result, so that some record may be summed
	/// by order statistics rather than by convolution.
	pub drops: bool,

	/// Whether the count, faces, or range endpoints of some roll, or the
	/// count of some drop, depend on a roll, so that some record holds a
	/// mixture of setups.
	pub dynamic: bool,

	/// Whether some random value fans out, so that the pass splits its worlds
	/// on it.
	pub correlated: bool,

	/// Whether some multiplication, division, remainder, exponentiation, or
	/// maximum combines two distinct random values, whose pairs the pass
	/// enumerates.
	pub pairwise: bool
}

/// Estimate the cost of the forward pass over the evaluator's function, for
/// the given arguments and the evaluator's external variables, as
/// [`propagate_worlds`](super::propagate_worlds) would run it.
///
/// # Parameters
/// - `evaluator`: The evaluator, suitably [bound](Evaluator::bind).
/// - `args`: The arguments to the function.
///
/// # Returns
/// The estimate, whose steps, cells, and worlds are never less than those
/// that the pass reports in its [usage](super::meter::Usage), and whose bits
/// are never less than those of the total weight of its answer.
///
/// # Errors
/// [`BadArity`](PropagationError::BadArity) if the number of arguments
/// disagrees with the number of formal parameters of the function.
pub(crate) fn estimate(
	evaluator: &Evaluator,
	args: impl IntoIterator<Item = i32>
) -> Result<Cost, PropagationError>
{
	observe(evaluator, args, |_| {})
}

/// Estimate the cost of the forward pass, as [`estimate`] does, showing the
/// lane to the specified observer after every step of the plan.
///
/// # Parameters
/// - `evaluator`: The evaluator, suitably [bound](Evaluator::bind).
/// - `args`: The arguments to the function.
/// - `observer`: The observer.
///
/// # Returns
/// The estimate.
///
/// # Errors
/// [`BadArity`](PropagationError::BadArity) if the number of arguments
/// disagrees with the number of formal parameters of the function.
fn observe(
	evaluator: &Evaluator,
	args: impl IntoIterator<Item = i32>,
	mut observer: impl FnMut(&Lane)
) -> Result<Cost, PropagationError>
{
	let function = &evaluator.function;
	let arity = function.arity();
	let args = args.into_iter().collect::<Vec<_>>();
	if args.len() != arity
	{
		return Err(PropagationError::BadArity {
			expected: arity,
			given: args.len()
		})
	}
	// The pass fixes every binding, and an unbound external variable holds
	// zero, so every binding is a degenerate interval.
	let mut bounds = BoundsEvaluator::new(function);
	bounds
		.seed(
			args.into_iter().map(|arg| Some(arg.into())),
			evaluator
				.environment
				.iter()
				.map(|(index, value)| (*index, (*value).into())),
			EvaluationBounds::default()
		)
		.expect("the arity agrees");
	let mut lane = Lane {
		bounds,
		shadow: Shadow::new(function),
		fork: None
	};
	let mut consumed = Consumed::new(lane.shadow.cells());
	let plan = Plan::new(&function.instructions);
	let mut depths = depths(&plan, function.instructions.len()).into_iter();
	for (pc, inst) in function.instructions.iter().enumerate()
	{
		consumed.charge(lane.instruction(inst));
		for step in plan.steps(ProgramCounter(pc))
		{
			consumed.charge(lane.step(step));
			if let Step::Split(location) = step
			{
				let depth = depths.next().expect("every split has a depth");
				if let Location::Register(reg) = location
				{
					lane.fork(*reg, depth);
				}
			}
			observer(&lane);
		}
	}
	debug_assert!(lane.fork.is_none(), "every split merges by the end");
	consumed.charge(lane.shadow.finish());
	Ok(lane.shadow.cost(&consumed))
}

////////////////////////////////////////////////////////////////////////////////
//                                  Events.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The cost of one event of the pass, in bounds, across every world: an
/// instruction, which the pass runs in each world in turn, a split, a merge,
/// or the finish.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Event
{
	/// The steps taken.
	steps: u128,

	/// The cells that the worlds hold when no work is under way, before the
	/// event and after it, whichever is more.
	settled: u128,

	/// The most cells alive at once during the event.
	peak: u128,

	/// The worlds alive after the event.
	worlds: u128
}

impl Event
{
	/// Answer the event that bounds each quantity by the lesser of this
	/// event's and another's, both of which bound it.
	///
	/// # Parameters
	/// - `other`: The other event.
	///
	/// # Returns
	/// The lesser event.
	fn min(self, other: Event) -> Event
	{
		Event {
			steps: self.steps.min(other.steps),
			settled: self.settled.min(other.settled),
			peak: self.peak.min(other.peak),
			worlds: self.worlds.min(other.worlds)
		}
	}
}

/// The resources that the pass has consumed so far, in bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Consumed
{
	/// The steps taken.
	steps: u128,

	/// The most cells alive at once.
	peak: u128,

	/// The most worlds alive at once.
	most: u128
}

impl Consumed
{
	/// Construct the resources consumed by the initial world, before any
	/// event.
	///
	/// # Parameters
	/// - `cells`: The cells of the initial world.
	///
	/// # Returns
	/// The resources.
	fn new(cells: u128) -> Self
	{
		Self {
			steps: 0,
			peak: cells,
			most: 1
		}
	}

	/// Charge the specified event.
	///
	/// # Parameters
	/// - `event`: The event.
	fn charge(&mut self, event: Event)
	{
		self.steps = self.steps.saturating_add(event.steps);
		self.peak = self.peak.max(event.peak);
		self.most = self.most.max(event.worlds);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Lanes.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The most lanes that an estimate keeps at once, counting every lane that
/// has forked and every child of each.
const LANES: usize = 256;

/// A shadow of the pass, together with the intervals of the registers from
/// which it prices each instruction, and perhaps a fork.
///
/// A lane prices every world as its costliest, which is loose when the value of
/// a split fixes the operands of later instructions, as in
/// `{x}@(10D6) + {x}D6`, whose worlds roll from 10 to 60 dice. So a lane
/// _forks_ at a split of a register: it divides the interval of the register
/// into buckets, and gains a child for each, a copy of the lane whose interval
/// of the register is narrowed to the bucket, and whose split has one outcome,
/// so that the child bounds the cost of each outcome in its bucket. The
/// children price every event in lockstep with the lane, and their events
/// combine by _allotment_: the outcomes of the split, at most `k` in any world,
/// fall at most one to each value of a bucket, so they cost at most the
/// costliest buckets, each taken as often as it has values, until `k` are
/// taken. The lane keeps pricing its own events too, and answers the lesser
/// bound of each quantity, so a fork never loosens the estimate. When the split
/// merges, the children go, and the lane continues alone.
///
/// ```mermaid
/// flowchart TD
///     L["Lane<br/>x ∈ [10, 60]"] -->|"split on x"| F{"Lanes to spare?"}
///     F -->|no| A["Lane prices alone"]
///     F -->|yes| C1["Child<br/>x = 10"]
///     F -->|yes| C2["Child<br/>x = 11"]
///     F -->|yes| CN["… Child<br/>x = 60"]
///     C1 & C2 & CN --> E["Allot each event's<br/>costliest outcomes"]
///     L --> O["Lane's own event"]
///     E & O --> M["Lesser of each quantity"]
///     M -->|"merge of x"| R["Lane continues alone"]
/// ```
///
/// Children may fork in turn, at later splits, but the lanes number at most
/// [`LANES`], which they share fairly: at each split, every lane without
/// children takes the same number of buckets, or fewer if its interval is
/// narrower, reserving enough for the splits that open while this one is open
/// to fork as widely. Beyond the limit, a bucket holds several values, and
/// costs as its costliest, so that the estimate loosens gracefully.
#[cfg_attr(doc, aquamarine::aquamarine)]
#[derive(Debug, Clone)]
struct Lane<'f>
{
	/// The bounds evaluator, which holds the interval of each register.
	bounds: BoundsEvaluator<'f>,

	/// The shadow.
	shadow: Shadow,

	/// The fork of the lane, if the lane has forked at an open split.
	fork: Option<Fork<'f>>
}

/// The children of a lane that has forked at the split of a register.
#[derive(Debug, Clone)]
struct Fork<'f>
{
	/// The position of the split in the stack, from the bottom.
	split: usize,

	/// A bound on the outcomes of the split in any one world.
	outcomes: u128,

	/// The children, each with the number of values in its bucket.
	children: Vec<(u128, Lane<'f>)>
}

/// How the pass settles its worlds during an event, which decides how many of
/// them may hold working cells at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settling
{
	/// The pass settles each world before it works on the next, as it does for
	/// an instruction, so only one world holds working cells at once.
	Each,

	/// The pass settles every world at once, after working on all of them, as
	/// it does for a split or a merge, so every world's working cells may be
	/// alive at once.
	Once
}

impl<'f> Lane<'f>
{
	/// Price the specified instruction, which the pass runs in each world in
	/// turn.
	///
	/// # Parameters
	/// - `inst`: The instruction.
	///
	/// # Returns
	/// The event.
	fn instruction(&mut self, inst: &Instruction) -> Event
	{
		let before = self.bounds.registers().to_vec();
		self.bounds.step(inst);
		let held = self.shadow.cells();
		let mut pricing = Pricing {
			shadow: &mut self.shadow,
			before: &before,
			after: self.bounds.registers(),
			tally: Tally::default()
		};
		inst.visit(&mut pricing).unwrap();
		let tally = pricing.tally;
		// The pass runs the instruction in each world in turn, and settles
		// each before the next, so the worlds before the current one hold
		// their cells after it, and the current one and those after it their
		// cells before it.
		let worlds = self.shadow.worlds();
		let most = held.max(self.shadow.cells());
		let own = Event {
			steps: tally.times(worlds).steps,
			settled: worlds.saturating_mul(most),
			peak: (worlds - 1)
				.saturating_mul(most)
				.saturating_add(held)
				.saturating_add(tally.peak),
			worlds
		};
		match &mut self.fork
		{
			None => own,
			Some(fork) =>
			{
				let events = fork
					.children
					.iter_mut()
					.map(|(values, child)| (*values, child.instruction(inst)))
					.collect::<Vec<_>>();
				own.min(fork.combine(&events, Settling::Each))
			}
		}
	}

	/// Price the specified step of the plan, which the pass takes across
	/// every world at once.
	///
	/// # Parameters
	/// - `step`: The step.
	///
	/// # Returns
	/// The event.
	fn step(&mut self, step: &Step) -> Event
	{
		let own = match step
		{
			Step::Split(location) => self.shadow.split(*location),
			Step::Merge {
				split,
				survivor,
				dead
			} => self.shadow.merge(*split, *survivor, dead)
		};
		let Some(mut fork) = self.fork.take()
		else
		{
			return own
		};
		let events = fork
			.children
			.iter_mut()
			.map(|(values, child)| (*values, child.step(step)))
			.collect::<Vec<_>>();
		let mut combined = fork.combine(&events, Settling::Once);
		match *step
		{
			Step::Merge {
				split, survivor, ..
			} if split == fork.split =>
			{
				// The groups of the merge mix the worlds of every child, so the
				// worlds that remain are the lane's own. Each group mixes the
				// survivors of its outcomes, so its support is at most the sum
				// of theirs.
				combined.worlds = own.worlds;
				if let Some(survivor) = survivor
				{
					let support = fork.allot(fork.children.iter().map(
						|(values, child)| {
							(*values, child.shadow.support(survivor))
						}
					));
					self.shadow.narrow(survivor, support);
				}
				return own.min(combined)
			},
			Step::Merge { split, .. } if split < fork.split => fork.split -= 1,
			_ =>
			{}
		}
		self.fork = Some(fork);
		own.min(combined)
	}

	/// Fork every lane without children at the split of the specified register,
	/// which the last step opened, sharing the lanes to spare fairly among
	/// them.
	///
	/// Each takes the same number of buckets, `L`, or as many as its interval
	/// has values, if fewer, and forks if that is at least two. A bucket may
	/// fork again at each of the `d - 1` splits of a register that open while
	/// this one is open, at most, so the lanes that descend from `b` buckets
	/// number about `bᵈ`, and `L` is the greatest level whose descendants
	/// number no more than the lanes to spare.
	///
	/// # Parameters
	/// - `reg`: The index of the register.
	/// - `depth`: The most splits of registers open at once while this split is
	///   open, counting it, `d`.
	fn fork(&mut self, reg: RegisterIndex, depth: u32)
	{
		let spare = LANES.saturating_sub(self.lanes()) as u128;
		let mut leaves = Vec::new();
		self.leaves(&mut leaves);
		let leaves = leaves
			.into_iter()
			.filter_map(|leaf| {
				let values = width(leaf.bounds.registers()[reg.0]);
				let outcomes =
					leaf.shadow.splits.last().map_or(1, |s| s.outcomes);
				(values >= 2 && outcomes >= 2).then_some((values, leaf))
			})
			.collect::<Vec<_>>();
		let descendants = |level: u128| {
			leaves.iter().fold(0u128, |sum, (values, _)| {
				sum.saturating_add((*values).min(level).saturating_pow(depth))
			})
		};
		// Find the greatest level whose descendants fit, by bisection.
		let (mut fits, mut exceeds) = (0u128, spare + 1);
		while exceeds - fits > 1
		{
			let level = fits + (exceeds - fits) / 2;
			match descendants(level) <= spare
			{
				true => fits = level,
				false => exceeds = level
			}
		}
		for (values, leaf) in leaves
		{
			let buckets = values.min(fits);
			if buckets >= 2
			{
				leaf.divide(reg, buckets);
			}
		}
	}

	/// Divide the interval of the specified register, whose split the last step
	/// opened, into the specified number of buckets of nearly equal width, and
	/// fork the lane into a child for each.
	///
	/// # Parameters
	/// - `reg`: The index of the register.
	/// - `buckets`: The number of buckets, which is at least two and at most
	///   the width of the interval.
	fn divide(&mut self, reg: RegisterIndex, buckets: u128)
	{
		let interval = self.bounds.registers()[reg.0];
		let least = interval.min as i128;
		let values = width(interval) as i128;
		let split = self.shadow.splits.len() - 1;
		let outcomes = self.shadow.splits[split].outcomes;
		let buckets = buckets as i128;
		let children = (0..buckets)
			.map(|i| {
				let start = least + i * values / buckets;
				let end = least + (i + 1) * values / buckets - 1;
				let mut child = self.clone();
				child.bounds.narrow(reg, (start as i32, end as i32).into());
				// The child bounds each outcome of its bucket.
				child.shadow.splits[split].outcomes = 1;
				((end - start + 1) as u128, child)
			})
			.collect();
		self.fork = Some(Fork {
			split,
			outcomes,
			children
		});
	}

	/// Answer the number of lanes of this lane, counting itself and every lane
	/// that descends from it.
	///
	/// # Returns
	/// The number of lanes.
	fn lanes(&self) -> usize
	{
		1 + self.fork.as_ref().map_or(0, |fork| {
			fork.children.iter().map(|(_, child)| child.lanes()).sum()
		})
	}

	/// Collect the lanes without children that descend from this lane, or this
	/// lane, if it has none.
	///
	/// # Parameters
	/// - `leaves`: The lanes collected so far.
	fn leaves<'a>(&'a mut self, leaves: &mut Vec<&'a mut Lane<'f>>)
	{
		if self.fork.is_none()
		{
			leaves.push(self);
			return
		}
		for (_, child) in
			self.fork.iter_mut().flat_map(|fork| &mut fork.children)
		{
			child.leaves(leaves);
		}
	}
}

impl Fork<'_>
{
	/// Answer a bound on the sum of a quantity over the outcomes of the split
	/// in any one world, given a bound on the quantity for each outcome of each
	/// bucket: the sum over the costliest buckets, each taken as often as it
	/// has values, until the outcomes are taken.
	///
	/// # Parameters
	/// - `buckets`: The number of values of each bucket, and its bound.
	///
	/// # Returns
	/// The bound.
	fn allot(&self, buckets: impl Iterator<Item = (u128, u128)>) -> u128
	{
		let mut buckets = buckets.collect::<Vec<_>>();
		buckets.sort_unstable_by_key(|&(_, bound)| Reverse(bound));
		let mut left = self.outcomes;
		let mut sum = 0u128;
		for (values, bound) in buckets
		{
			let taken = values.min(left);
			sum = sum.saturating_add(taken.saturating_mul(bound));
			left -= taken;
			if left == 0
			{
				break
			}
		}
		sum
	}

	/// Combine the events of the children into a bound on the event of the
	/// lane.
	///
	/// The children's worlds settle as the lane's do. If the pass settles each
	/// world in turn, then one world works at once while the others hold their
	/// settled cells, so the cells alive are at most the settled cells of every
	/// outcome, and the most that one outcome's worlds exceed their settled
	/// cells. If it settles every world at once, then every outcome may reach
	/// its peak at once.
	///
	/// # Parameters
	/// - `events`: The number of values of each child's bucket, and the event
	///   of each outcome of it.
	/// - `settling`: How the pass settles the worlds during the event.
	///
	/// # Returns
	/// The combined event.
	fn combine(&self, events: &[(u128, Event)], settling: Settling) -> Event
	{
		let allot = |quantity: fn(&Event) -> u128| {
			self.allot(
				events
					.iter()
					.map(|(values, event)| (*values, quantity(event)))
			)
		};
		let settled = allot(|event| event.settled);
		let peak = match settling
		{
			Settling::Each => settled.saturating_add(
				events
					.iter()
					.map(|(_, event)| event.peak.saturating_sub(event.settled))
					.max()
					.unwrap_or(0)
			),
			Settling::Once => allot(|event| event.peak.max(event.settled))
		};
		Event {
			steps: allot(|event| event.steps),
			settled,
			peak,
			worlds: allot(|event| event.worlds)
		}
	}
}

/// Answer, for each split of the plan, in order, the most splits of registers
/// open at once while it is open, counting it if it splits a register.
///
/// # Parameters
/// - `plan`: The plan.
/// - `len`: The number of instructions.
///
/// # Returns
/// The depth of each split.
fn depths(plan: &Plan, len: usize) -> Vec<u32>
{
	let mut depths = Vec::new();
	// The open splits, from the bottom of the stack: the position of each in
	// the depths, and whether it splits a register.
	let mut open = Vec::<(usize, bool)>::new();
	for pc in 0..len
	{
		for step in plan.steps(ProgramCounter(pc))
		{
			match step
			{
				Step::Split(location) =>
				{
					open.push((
						depths.len(),
						matches!(location, Location::Register(_))
					));
					depths.push(0);
					// Each open split counts the splits of registers at or
					// above it.
					let mut above = 0;
					for &(split, register) in open.iter().rev()
					{
						above += register as u32;
						depths[split] = depths[split].max(above);
					}
				},
				Step::Merge { split, .. } =>
				{
					open.remove(*split);
				}
			}
		}
	}
	depths
}

////////////////////////////////////////////////////////////////////////////////
//                                  Tallies.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A tally of the steps and cells that some work charges to the meter, as the
/// meter counts them, in bounds: the steps taken, the working cells left
/// charged, and the most working cells charged at once. Each bound is at
/// least what the meter counts, so long as every cell that the tally frees
/// was charged as the same bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Tally
{
	/// The steps taken.
	steps: u128,

	/// The working cells left charged.
	working: u128,

	/// The most working cells charged at once.
	peak: u128
}

impl Tally
{
	/// Charge the specified steps and cells.
	///
	/// # Parameters
	/// - `steps`: The steps.
	/// - `cells`: The cells.
	fn charge(&mut self, steps: u128, cells: u128)
	{
		self.steps = self.steps.saturating_add(steps);
		self.working = self.working.saturating_add(cells);
		self.peak = self.peak.max(self.working);
	}

	/// Free working cells charged earlier, as the same bound.
	///
	/// # Parameters
	/// - `cells`: The cells.
	fn free(&mut self, cells: u128)
	{
		self.working = self.working.saturating_sub(cells);
	}

	/// Replace working cells charged earlier with the fewer cells that remain
	/// of them, as when an operation frees all but its answer.
	///
	/// # Parameters
	/// - `charged`: The cells charged earlier, as the same bound.
	/// - `kept`: A bound on the cells that remain.
	fn keep(&mut self, charged: u128, kept: u128)
	{
		self.working =
			self.working.saturating_sub(charged).saturating_add(kept);
	}

	/// Follow this tally with another, whose working cells add to those that
	/// this one leaves charged.
	///
	/// # Parameters
	/// - `next`: The tally of the work that follows.
	fn then(&mut self, next: Tally)
	{
		self.steps = self.steps.saturating_add(next.steps);
		self.peak = self.peak.max(self.working.saturating_add(next.peak));
		self.working = self.working.saturating_add(next.working);
	}

	/// Interleave this tally with another, in any order, so that either may
	/// reach its peak while the other has left all that it leaves charged.
	///
	/// # Parameters
	/// - `other`: The tally of the other work.
	///
	/// # Returns
	/// The tally of both.
	fn beside(self, other: Tally) -> Tally
	{
		Tally {
			steps: self.steps.saturating_add(other.steps),
			working: self.working.saturating_add(other.working),
			peak: self
				.peak
				.saturating_add(other.working)
				.max(other.peak.saturating_add(self.working))
		}
	}

	/// Answer the tally of the specified number of repetitions of this work,
	/// none of which frees what another leaves charged.
	///
	/// # Parameters
	/// - `n`: The number of repetitions.
	///
	/// # Returns
	/// The tally of the repetitions.
	fn times(self, n: u128) -> Tally
	{
		match n
		{
			0 => Tally::default(),
			n => Tally {
				steps: self.steps.saturating_mul(n),
				working: self.working.saturating_mul(n),
				peak: self
					.working
					.saturating_mul(n - 1)
					.saturating_add(self.peak)
			}
		}
	}

	/// Answer a tally that bounds both this one and another, as when the work
	/// takes one of two paths.
	///
	/// # Parameters
	/// - `other`: The other tally.
	///
	/// # Returns
	/// The bounding tally.
	fn max(self, other: Tally) -> Tally
	{
		Tally {
			steps: self.steps.max(other.steps),
			working: self.working.max(other.working),
			peak: self.peak.max(other.peak)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Shadows.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The shadow of the forward pass: bounds on the state of every world, and on
/// the resources that the pass has consumed so far.
#[derive(Debug, Clone)]
struct Shadow
{
	/// The bounds of the registers.
	registers: Vec<Slot>,

	/// The bounds of the rolling records.
	records: Vec<Record>,

	/// The bounds of the answer, once a return has set it.
	answer: Option<Slot>,

	/// The open splits, from the bottom of the stack.
	splits: Vec<Split>,

	/// Whether the [common factor](super::World) of some world may differ
	/// from one, so that finishing must scale the answer.
	scaled: bool,

	/// A bound on the base-two logarithm of the total weight of the answer.
	log: f64,

	/// The class of the function.
	class: Class
}

/// The bounds of a value in every world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot
{
	/// A bound on the support of the value in any one world, which is one if
	/// the value is fixed in every world.
	support: u128,

	/// Whether the value may still be random and unread in some world, and
	/// so occupy cells beyond the register's slot.
	unread: bool,

	/// Whether the value certainly has at least two outcomes in every world,
	/// so that it is random, never fixed.
	varied: bool,

	/// Whether the value depends on a roll.
	random: bool,

	/// Whether the value certainly takes every value of its interval in every
	/// world, so that its support is the interval's width.
	dense: bool,

	/// The width of the interval of the value, which bounds its support.
	width: u128
}

impl Slot
{
	/// The slot of a fixed value: a binding, or a register not yet written.
	const FIXED: Self = Self {
		support: 1,
		unread: false,
		varied: false,
		random: false,
		dense: false,
		width: 1
	};

	/// Answer a bound on the cells that the register's value occupies beyond
	/// its slot.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u128
	{
		match self.unread && self.support > 1
		{
			true => self.support,
			false => 0
		}
	}
}

/// An open split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Split
{
	/// A bound on the outcomes of the split in any one world.
	outcomes: u128,

	/// A bound on the cells of one outcome.
	cells: u128
}

impl Shadow
{
	/// Construct the shadow of the initial world of the specified function.
	///
	/// # Parameters
	/// - `function`: The function.
	///
	/// # Returns
	/// The shadow.
	fn new(function: &Function) -> Self
	{
		Self {
			registers: vec![Slot::FIXED; function.register_count],
			records: vec![Record::default(); function.rolling_record_count],
			answer: None,
			splits: Vec::new(),
			scaled: false,
			log: 0.0,
			class: Class::default()
		}
	}

	/// Answer a bound on the worlds alive now: the product of the outcomes of
	/// the open splits.
	///
	/// # Returns
	/// The number of worlds.
	fn worlds(&self) -> u128
	{
		self.splits
			.iter()
			.fold(1u128, |worlds, split| worlds.saturating_mul(split.outcomes))
	}

	/// Answer a bound on the cells of any one world, as
	/// [`World::cells`](super::World) counts them.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u128
	{
		let registers = self
			.registers
			.iter()
			.fold(self.registers.len() as u128, |cells, slot| {
				cells.saturating_add(slot.cells())
			});
		let records = self
			.records
			.iter()
			.fold(0u128, |cells, record| cells.saturating_add(record.cells));
		registers
			.saturating_add(records)
			.saturating_add(self.answer.map_or(0, |answer| answer.support))
			.saturating_add(self.fixed())
	}

	/// Answer a bound on the cells of the outcomes of the open splits, which
	/// each world holds atop its stack.
	///
	/// # Returns
	/// The number of cells.
	fn fixed(&self) -> u128
	{
		self.splits
			.iter()
			.fold(0u128, |cells, split| cells.saturating_add(split.cells))
	}

	/// Answer the event of the specified work across every world, which changed
	/// the worlds from those of the specified number and cells to those of the
	/// shadow now.
	///
	/// # Parameters
	/// - `steps`: The steps of the work, across every world.
	/// - `worlds`: The worlds before the work.
	/// - `cells`: A bound on the cells of each world before the work.
	/// - `peak`: A bound on the most working cells that the work charges at
	///   once.
	///
	/// # Returns
	/// The event.
	fn event(&self, steps: u128, worlds: u128, cells: u128, peak: u128)
	-> Event
	{
		let held = worlds.saturating_mul(cells);
		let after = self.worlds();
		Event {
			steps,
			settled: held.max(after.saturating_mul(self.cells())),
			peak: held.saturating_add(peak),
			worlds: after
		}
	}

	/// Answer the estimate.
	///
	/// # Parameters
	/// - `consumed`: The resources that the pass consumes.
	///
	/// # Returns
	/// The estimate.
	fn cost(&self, consumed: &Consumed) -> Cost
	{
		// The total weight `T` of the answer satisfies `log₂ T ≤ self.log`, so
		// it has at most `⌊self.log⌋ + 1` bits. The margin absorbs the rounding
		// of the sum.
		let log = self.log * (1.0 + 1e-12) + 1e-9;
		let bits = match log.is_finite() && log < u64::MAX as f64
		{
			true => log.floor() as u64 + 1,
			false => u64::MAX
		};
		Cost {
			steps: saturate(consumed.steps),
			cells: saturate(consumed.peak),
			worlds: saturate(consumed.most),
			bits,
			class: self.class
		}
	}
}

/// Convert a bound to a `u64`, saturating at [`u64::MAX`].
///
/// # Parameters
/// - `bound`: The bound.
///
/// # Returns
/// The saturated bound.
#[inline]
fn saturate(bound: u128) -> u64 { u64::try_from(bound).unwrap_or(u64::MAX) }

////////////////////////////////////////////////////////////////////////////////
//                                 Intervals.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Answer the number of values in the specified interval.
///
/// # Parameters
/// - `bounds`: The interval.
///
/// # Returns
/// The width, which is at least one.
fn width(bounds: EvaluationBounds) -> u128
{
	(bounds.max as i64 - bounds.min as i64 + 1).max(1) as u128
}

/// Answer whether no value of the specified interval saturates: whether it
/// lies strictly within `i32`, so that the exact results that it bounds were
/// never clamped.
///
/// # Parameters
/// - `bounds`: The interval.
///
/// # Returns
/// `true` if no value saturates.
fn unsaturated(bounds: EvaluationBounds) -> bool
{
	bounds.min > i32::MIN && bounds.max < i32::MAX
}

/// Answer the sum of the `k` greatest nonnegative values in the specified
/// interval, which bounds the sum of any `k` distinct values in it, less those
/// that are negative.
///
/// # Parameters
/// - `bounds`: The interval.
/// - `k`: The number of values.
///
/// # Returns
/// The sum.
fn top_sum(bounds: EvaluationBounds, k: u128) -> u128
{
	let greatest = bounds.max.max(0) as u128;
	let least = bounds.min.max(0) as u128;
	let k = k.min(greatest - least + 1);
	// The values run from `greatest - k + 1` to `greatest`.
	k.saturating_mul(2 * greatest + 1 - k) / 2
}

/// The most values that [`tops`] lists one by one.
const LISTED: u128 = 256;

/// Answer values that bound any `k` distinct values of the specified interval
/// that are at least `floor`, each with its multiplicity: the `k` greatest,
/// each once, or, if there are more than [`LISTED`], the greatest, `k` times.
///
/// # Parameters
/// - `bounds`: The interval.
/// - `floor`: The least value of interest, which is positive.
/// - `k`: The number of values.
///
/// # Returns
/// The values and their multiplicities, in descending order of value, which
/// are empty if no value of the interval is at least `floor`.
fn tops(bounds: EvaluationBounds, floor: i32, k: u128) -> Vec<(u128, u128)>
{
	if bounds.max < floor
	{
		return Vec::new()
	}
	let greatest = bounds.max as u128;
	let k = k.min(greatest - bounds.min.max(floor) as u128 + 1);
	match k <= LISTED
	{
		true => (0..k).map(|i| (greatest - i, 1)).collect(),
		false => vec![(greatest, k)]
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Registers.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The bounds of an operand of an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Operand
{
	/// The interval of the operand's value.
	bounds: EvaluationBounds,

	/// A bound on the support of the operand's value in any one world.
	support: u128,

	/// Whether the value certainly has at least two outcomes in every world.
	varied: bool,

	/// Whether the value depends on a roll.
	random: bool,

	/// Whether the value certainly takes every value of its interval in every
	/// world, so that its support is the interval's width.
	dense: bool
}

impl Operand
{
	/// Answer the number of values in the operand's interval.
	///
	/// # Returns
	/// The width.
	fn width(&self) -> u128 { width(self.bounds) }

	/// Answer whether the operand is never zero.
	///
	/// # Returns
	/// `true` if the operand's interval excludes zero.
	fn nonzero(&self) -> bool { !self.bounds.contains(0) }
}

/// The pricing of one instruction in one world, which updates the shadow and
/// tallies the work.
struct Pricing<'a>
{
	/// The shadow.
	shadow: &'a mut Shadow,

	/// The intervals of the registers before the instruction.
	before: &'a [EvaluationBounds],

	/// The intervals of the registers after the instruction.
	after: &'a [EvaluationBounds],

	/// The tally of the instruction's work in one world.
	tally: Tally
}

impl Pricing<'_>
{
	/// Answer the bounds of the specified operand.
	///
	/// # Parameters
	/// - `op`: The operand, an immediate or a register.
	///
	/// # Returns
	/// The bounds.
	fn operand(&self, op: AddressingMode) -> Operand
	{
		match op
		{
			AddressingMode::Immediate(value) => Operand {
				bounds: value.0.into(),
				support: 1,
				varied: false,
				random: false,
				dense: true
			},
			AddressingMode::Register(reg) =>
			{
				let slot = &self.shadow.registers[reg.0];
				let width = width(self.before[reg.0]);
				Operand {
					bounds: self.before[reg.0],
					support: slot.support.min(width),
					varied: slot.varied,
					random: slot.random,
					// A value of one possibility takes it.
					dense: width == 1 || (slot.dense && slot.width == width)
				}
			},
			AddressingMode::RollingRecord(_) => unreachable!()
		}
	}

	/// Read the specified operand, as [`World::read`](super::World) does,
	/// which charges a step and a cell for a fixed value.
	///
	/// # Parameters
	/// - `op`: The operand, an immediate or a register.
	///
	/// # Returns
	/// The bounds of the operand.
	fn read(&mut self, op: AddressingMode) -> Operand
	{
		let operand = self.operand(op);
		if !operand.varied
		{
			self.tally.charge(1, 1);
		}
		if let AddressingMode::Register(reg) = op
		{
			self.shadow.registers[reg.0].unread = false;
		}
		operand
	}

	/// Read the pairs of values of the specified operands, as
	/// [`World::read_pair`](super::World) does.
	///
	/// # Parameters
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The bounds of the operands, and a bound on the number of pairs.
	fn read_pair(
		&mut self,
		op1: AddressingMode,
		op2: AddressingMode
	) -> (Operand, Operand, u128)
	{
		match op1
		{
			AddressingMode::Register(_) if op1 == op2 =>
			{
				let x = self.read(op1);
				self.tally.charge(x.support, x.support);
				(x, x, x.support)
			},
			_ =>
			{
				let x = self.read(op1);
				let y = self.read(op2);
				let pairs = x.support.saturating_mul(y.support);
				self.tally.charge(pairs, pairs);
				(x, y, pairs)
			}
		}
	}

	/// Write a value to the specified register, as
	/// [`World::write`](super::World) does, which charges a step if the value
	/// is fixed, folding its weight into the common factor, and a step to
	/// discard the old value if it is random and unread, folding its weight
	/// likewise.
	///
	/// # Parameters
	/// - `dest`: The index of the register.
	/// - `support`: A bound on the support of the value.
	/// - `varied`: Whether the value certainly has at least two outcomes in
	///   every world.
	/// - `random`: Whether the value depends on a roll.
	/// - `dense`: Whether the value certainly takes every value of the
	///   register's interval after the instruction in every world.
	fn write(
		&mut self,
		dest: usize,
		support: u128,
		varied: bool,
		random: bool,
		dense: bool
	)
	{
		let old = self.shadow.registers[dest];
		let discard = old.cells() > 0;
		self.tally.charge(!varied as u128 + discard as u128, 0);
		self.shadow.scaled |= discard || (random && !varied);
		let width = width(self.after[dest]);
		self.shadow.registers[dest] = Slot {
			support: support.min(width).max(1),
			unread: true,
			varied,
			random,
			// A value of one possibility takes it.
			dense: dense || width == 1,
			width
		};
	}

	/// Note that the value just written to the specified register certainly
	/// takes every value of its interval in every world, if its operands do,
	/// and if the interval is exactly the one that they reach, so that no
	/// value saturates.
	///
	/// # Parameters
	/// - `dest`: The index of the register.
	/// - `dense`: Whether the operands certainly take every value of their
	///   intervals in every world.
	/// - `least`: The least value that the operands reach.
	/// - `greatest`: The greatest value that the operands reach.
	fn densify(&mut self, dest: usize, dense: bool, least: i64, greatest: i64)
	{
		let bounds = self.after[dest];
		if dense && bounds.min as i64 == least && bounds.max as i64 == greatest
		{
			self.shadow.registers[dest].dense = true;
		}
	}

	/// Apply a binary operation, as
	/// [`Execution::binary`](super::Execution) does, and write its result.
	///
	/// # Parameters
	/// - `dest`: The index of the destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	/// - `op`: The operation.
	///
	/// # Returns
	/// The bounds of the operands, if they are distinct.
	fn binary(
		&mut self,
		dest: usize,
		op1: AddressingMode,
		op2: AddressingMode,
		op: Op
	) -> Option<(Operand, Operand)>
	{
		// An exact result that never saturates is injective in each operand
		// for addition, subtraction, and multiplication by a value never zero,
		// and a product of two operands of two outcomes apiece has two.
		let exact = unsaturated(self.after[dest]);
		let (result, varied, random, operands) = match op1
		{
			AddressingMode::Register(_) if op1 == op2 =>
			{
				let x = self.read(op1);
				self.tally.charge(x.support, x.support);
				let varied = exact && x.varied && op == Op::Add;
				(x.support, varied, x.random, None)
			},
			_ =>
			{
				let x = self.read(op1);
				let y = self.read(op2);
				let (result, varied) = match op
				{
					Op::Add => (
						self.saturating_sum(x, y, dest),
						exact && (x.varied || y.varied)
					),
					Op::Mul | Op::Other =>
					{
						let pairs = x.support.saturating_mul(y.support);
						self.tally.charge(pairs, pairs);
						let varied = op == Op::Mul
							&& exact
							&& ((x.varied && y.nonzero())
								|| (y.varied && x.nonzero())
								|| (x.varied && y.varied));
						(pairs, varied)
					}
				};
				(result, varied, x.random || y.random, Some((x, y)))
			}
		};
		self.write(dest, result, varied, random, false);
		operands
	}

	/// Add or subtract independent operands, as
	/// [`saturating_sum`](super::saturating_sum) does: widen them, convolve
	/// them, and clamp the convolution.
	///
	/// # Parameters
	/// - `x`: The first operand.
	/// - `y`: The second operand.
	/// - `dest`: The index of the destination register.
	///
	/// # Returns
	/// A bound on the support of the result.
	fn saturating_sum(&mut self, x: Operand, y: Operand, dest: usize) -> u128
	{
		let wide = x.support.saturating_add(y.support);
		self.tally.charge(wide, wide);
		// Negating the second operand preserves its width, so the sums or
		// differences span the same number of values.
		let span = x.width() + y.width() - 1;
		let pairs = x.support.saturating_mul(y.support);
		let sums = span.min(pairs);
		self.tally.charge(pairs, sums.saturating_mul(2));
		self.tally.keep(sums.saturating_mul(2), sums);
		self.tally.free(wide);
		clamp(&mut self.tally, sums, width(self.after[dest]))
	}
}

/// The kind of a binary operation, as far as its cost and outcomes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op
{
	/// Addition or subtraction, which convolves its operands.
	Add,

	/// Multiplication, which pairs its operands.
	Mul,

	/// Division, remainder, exponentiation, or maximum, which pair their
	/// operands.
	Other
}

/// Clamp a wide distribution to `i32`, as [`clamp`](super::record::clamp)
/// does.
///
/// # Parameters
/// - `tally`: The tally, to which the wide distribution is charged.
/// - `len`: A bound on the length of the wide distribution.
/// - `limit`: A bound on the length of the clamped distribution.
///
/// # Returns
/// A bound on the length of the clamped distribution.
fn clamp(tally: &mut Tally, len: u128, limit: u128) -> u128
{
	tally.charge(len, len);
	let clamped = len.min(limit);
	tally.keep(len.saturating_mul(2), clamped);
	clamped
}

impl InstructionVisitor<()> for Pricing<'_>
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		let (start, end, pairs) = self.read_pair(inst.start, inst.end);
		self.tally.charge(pairs, pairs);
		self.shadow.class.dynamic |= start.random || end.random;
		// The totals of the worlds' setups divide the least common multiple of
		// every width that the range may have in any world.
		let widest = width((start.bounds.min, end.bounds.max).into());
		let narrowest = width((start.bounds.max, end.bounds.min).into());
		self.shadow.log += lcm_log(widest, widest - narrowest + 1);
		let record = Record {
			kind: Kind::Range { start, end },
			setups: pairs,
			cells: pairs,
			dice: 0,
			lowest: 0.into(),
			highest: 0.into(),
			summed: false,
			random: true
		};
		self.fill(inst.dest.0, record);
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		let (count, faces, pairs) = self.read_pair(inst.count, inst.faces);
		self.shadow.class.dynamic |= count.random || faces.random;
		// Each pair charges a step, and one for each of its dice. When the
		// count and the faces are one register, each number of faces has one
		// count.
		let (counts, dies, dice) = match inst.count == inst.faces
		{
			true => (1, pairs, top_sum(count.bounds, pairs)),
			false => (
				count.support,
				faces.support,
				faces
					.support
					.saturating_mul(top_sum(count.bounds, count.support))
			)
		};
		self.tally.charge(pairs.saturating_add(dice), pairs);
		// The totals of the worlds' setups divide the least common multiple of
		// every number of faces that the dice may have in any world, raised to
		// the greatest count.
		let greatest = count.bounds.max.max(0) as f64;
		let faces_in_any =
			width((faces.bounds.min.max(1), faces.bounds.max).into());
		self.shadow.log +=
			greatest * lcm_log(faces.bounds.max.max(0) as u128, faces_in_any);
		let record = Record {
			kind: Kind::Standard {
				count,
				faces,
				counts,
				dies
			},
			setups: pairs,
			cells: pairs,
			dice,
			lowest: 0.into(),
			highest: 0.into(),
			summed: false,
			random: true
		};
		self.fill(inst.dest.0, record);
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		let count = self.read(inst.count);
		self.shadow.class.dynamic |= count.random;
		let raw = inst.faces.len() as u128;
		let distinct = inst.faces.iter().copied().collect::<BTreeSet<_>>();
		let die = Die::custom(distinct.into_iter().collect());
		let dice = top_sum(count.bounds, count.support);
		// The distinct faces take a step apiece, and each count a step, one for
		// each of its dice, and a setup that copies the faces.
		self.tally.charge(
			raw.saturating_add(count.support).saturating_add(dice),
			raw.saturating_add(count.support.saturating_mul(1 + raw))
		);
		self.shadow.log +=
			count.bounds.max.max(0) as f64 * (raw as f64).log2().max(0.0);
		let cells = count.support.saturating_mul(1 + die.faces);
		let record = Record {
			kind: Kind::Custom { count, die },
			setups: count.support,
			cells,
			dice,
			lowest: 0.into(),
			highest: 0.into(),
			summed: false,
			random: true
		};
		self.fill(inst.dest.0, record);
		Ok(())
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		self.drop(inst.dest.0, inst.count, true);
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		self.drop(inst.dest.0, inst.count, false);
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		let limit = width(self.after[inst.dest.0]);
		let record = &mut self.shadow.records[inst.src.0];
		record.summed = true;
		let (tally, support) = record.sum(limit);
		let (varied, random) = (record.varied(), record.random);
		let dense = record.dense(self.after[inst.dest.0]);
		self.tally.then(tally);
		self.write(inst.dest.0, support, varied, random, dense);
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		if let Some((x, y)) =
			self.binary(inst.dest.0, inst.op1, inst.op2, Op::Add)
		{
			// The sums of two runs of values are a run.
			self.densify(
				inst.dest.0,
				x.dense && y.dense,
				x.bounds.min as i64 + y.bounds.min as i64,
				x.bounds.max as i64 + y.bounds.max as i64
			);
		}
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		// A register less itself is zero, which is never varied.
		if let AddressingMode::Register(_) = inst.op1
			&& inst.op1 == inst.op2
		{
			let x = self.read(inst.op1);
			self.tally.charge(x.support, x.support);
			self.write(inst.dest.0, x.support, false, x.random, false);
			return Ok(())
		}
		if let Some((x, y)) =
			self.binary(inst.dest.0, inst.op1, inst.op2, Op::Add)
		{
			// The differences of two runs of values are a run.
			self.densify(
				inst.dest.0,
				x.dense && y.dense,
				x.bounds.min as i64 - y.bounds.max as i64,
				x.bounds.max as i64 - y.bounds.min as i64
			);
		}
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.pairwise(inst.dest.0, inst.op1, inst.op2, Op::Mul);
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		self.pairwise(inst.dest.0, inst.op1, inst.op2, Op::Other);
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		self.pairwise(inst.dest.0, inst.op1, inst.op2, Op::Other);
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		self.pairwise(inst.dest.0, inst.op1, inst.op2, Op::Other);
		Ok(())
	}

	fn visit_max(&mut self, inst: &Max) -> Result<(), ()>
	{
		self.pairwise(inst.dest.0, inst.op1, inst.op2, Op::Other);
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		let x = self.read(inst.op);
		self.tally.charge(x.support, x.support);
		// Negation is injective, but for the least value, which saturates.
		let varied = x.varied && x.bounds.min > i32::MIN;
		self.write(inst.dest.0, x.support, varied, x.random, false);
		self.densify(
			inst.dest.0,
			x.dense,
			-(x.bounds.max as i64),
			-(x.bounds.min as i64)
		);
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		let x = self.read(inst.src);
		self.shadow.answer = Some(Slot {
			support: x.support,
			unread: true,
			varied: x.varied,
			random: x.random,
			dense: x.dense,
			width: x.width()
		});
		Ok(())
	}
}

impl Pricing<'_>
{
	/// Apply an operation that pairs its operands, and note whether it pairs
	/// two distinct random values.
	///
	/// # Parameters
	/// - `dest`: The index of the destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	/// - `op`: The operation.
	fn pairwise(
		&mut self,
		dest: usize,
		op1: AddressingMode,
		op2: AddressingMode,
		op: Op
	)
	{
		if let Some((x, y)) = self.binary(dest, op1, op2, op)
		{
			self.shadow.class.pairwise |= x.random && y.random;
		}
	}

	/// Fill the specified rolling record with a new roll, as
	/// [`World::roll`](super::World) does, retiring the old one.
	///
	/// # Parameters
	/// - `dest`: The index of the record.
	/// - `record`: The bounds of the new record.
	fn fill(&mut self, dest: usize, record: Record)
	{
		let old = std::mem::replace(&mut self.shadow.records[dest], record);
		self.tally.charge(old.retirement(), 0);
		self.shadow.scaled |= old.scales();
	}

	/// Drop results of the specified rolling record, as
	/// [`Mixture::drop`](super::mixture::Mixture) does.
	///
	/// # Parameters
	/// - `dest`: The index of the record.
	/// - `op`: The count of results to drop.
	/// - `lowest`: Whether to drop the lowest results rather than the highest.
	fn drop(&mut self, dest: usize, op: AddressingMode, lowest: bool)
	{
		let count = self.read(op);
		self.shadow.class.dynamic |= count.random;
		self.shadow.class.drops |= count.bounds.max > 0;
		let record = &mut self.shadow.records[dest];
		let outcomes = count.support;
		self.tally.charge(
			outcomes.saturating_mul(record.setups.saturating_add(record.dice)),
			outcomes.saturating_mul(record.cells)
		);
		record.setups = record.setups.saturating_mul(outcomes);
		record.cells = record.cells.saturating_mul(outcomes);
		record.dice = record.dice.saturating_mul(outcomes);
		record.random |= count.random;
		record.summed = false;
		// A negative count drops nothing, and the drops accumulate as the
		// bounds evaluator accumulates them.
		let count = EvaluationBounds::from((
			count.bounds.min.max(0),
			count.bounds.max.max(0)
		));
		match lowest
		{
			true => record.lowest += count,
			false => record.highest += count
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Splits and merges.                            //
////////////////////////////////////////////////////////////////////////////////

impl Shadow
{
	/// Split every world on the value at the specified location, as
	/// [`split`](super::split) does.
	///
	/// # Parameters
	/// - `location`: The location, which is not the answer.
	///
	/// # Returns
	/// The event.
	fn split(&mut self, location: Location) -> Event
	{
		self.class.correlated = true;
		let worlds = self.worlds();
		let cells = self.cells();
		let mut tally = Tally::default();
		let (outcomes, outcome_cells) = match location
		{
			Location::Register(reg) =>
			{
				let slot = &mut self.registers[reg.0];
				let outcomes = slot.support;
				tally.charge(outcomes, outcomes);
				// Each world fixes the register, at a value that may differ
				// from one world to the next.
				slot.support = 1;
				slot.unread = false;
				slot.varied = false;
				slot.dense = slot.width == 1;
				(outcomes, 1)
			},
			Location::RollingRecord(rec) =>
			{
				let record = &mut self.records[rec.0];
				let (outcomes, outcome_cells) = record.outcomes(&mut tally);
				record.fix(outcome_cells);
				(outcomes, outcome_cells)
			},
			Location::Answer => unreachable!("no instruction reads the answer")
		};
		// Each world of the split copies its world, and holds its outcome
		// twice.
		let copies = outcomes.saturating_mul(
			cells.saturating_add(outcome_cells.saturating_mul(2))
		);
		tally.charge(copies, copies);
		let tally = tally.times(worlds);
		self.splits.push(Split {
			outcomes,
			cells: outcome_cells
		});
		self.event(tally.steps, worlds, cells, tally.peak)
	}

	/// Merge the worlds of the specified split, as [`merge`](super::merge)
	/// does.
	///
	/// # Parameters
	/// - `split`: The position of the split in the stack, from the bottom.
	/// - `survivor`: The location of the survivor, if any.
	/// - `dead`: The locations of the dead values that depend on the split.
	///
	/// # Returns
	/// The event.
	fn merge(
		&mut self,
		split: usize,
		survivor: Option<Location>,
		dead: &[Location]
	) -> Event
	{
		let worlds = self.worlds();
		let cells = self.cells();
		let mut tally = Tally::default();
		for &location in dead
		{
			match location
			{
				Location::Register(reg) =>
				{
					let slot = &mut self.registers[reg.0];
					tally.charge((slot.cells() > 0) as u128, 0);
					slot.unread = false;
				},
				Location::RollingRecord(rec) =>
				{
					let record = &mut self.records[rec.0];
					tally.charge(record.retirement(), 0);
					*record = Record {
						summed: true,
						..Record::default()
					};
				},
				Location::Answer => unreachable!("the answer lives to the end")
			}
		}
		// The key copies the outcomes of the open splits.
		tally.charge(self.splits.len() as u128, self.fixed());
		let mut tally = tally.times(worlds);
		let merged = self.splits.remove(split);
		let groups = self.worlds();
		// Each group mixes the survivors of its worlds, one for each outcome of
		// the split, and puts the mixture in the survivor's place. The merge
		// folds the common factor of every world into its survivor, so only a
		// survivor that may be fixed, or none, leaves a factor in the merged
		// world.
		let slot = match survivor
		{
			Some(Location::Register(reg)) => self.registers[reg.0],
			Some(Location::Answer) => self.answer.unwrap_or(Slot::FIXED),
			Some(Location::RollingRecord(_)) =>
			{
				unreachable!("a survivor is never a rolling record")
			},
			None => Slot::FIXED
		};
		let outcomes = merged.outcomes.saturating_mul(slot.support);
		let mut mix = Tally::default();
		mix.charge(merged.outcomes, merged.outcomes);
		mix.charge(outcomes, outcomes);
		mix.charge(!slot.varied as u128, 0);
		tally.then(mix.times(groups));
		// A survivor that takes every value of its interval in every world
		// still does once mixed.
		let merged = Slot {
			support: outcomes.min(slot.width).max(1),
			unread: true,
			..slot
		};
		match survivor
		{
			Some(Location::Register(reg)) =>
			{
				self.registers[reg.0] = merged;
				self.scaled = !merged.varied;
			},
			Some(Location::Answer) =>
			{
				self.answer = Some(merged);
				self.scaled = false;
			},
			_ => self.scaled = true
		}
		self.event(tally.steps, worlds, cells, tally.peak)
	}

	/// Answer a bound on the support of the survivor of a merge, in any one
	/// world.
	///
	/// # Parameters
	/// - `survivor`: The location of the survivor, which is not a rolling
	///   record.
	///
	/// # Returns
	/// The bound.
	fn support(&self, survivor: Location) -> u128
	{
		match survivor
		{
			Location::Register(reg) => self.registers[reg.0].support,
			Location::Answer => self.answer.map_or(1, |answer| answer.support),
			Location::RollingRecord(_) =>
			{
				unreachable!("a survivor is never a rolling record")
			}
		}
	}

	/// Narrow the bound on the support of the survivor of a merge to the
	/// specified bound, if it is less.
	///
	/// # Parameters
	/// - `survivor`: The location of the survivor, which is not a rolling
	///   record.
	/// - `support`: Another bound on its support.
	fn narrow(&mut self, survivor: Location, support: u128)
	{
		let slot = match survivor
		{
			Location::Register(reg) => &mut self.registers[reg.0],
			Location::Answer => match &mut self.answer
			{
				Some(answer) => answer,
				None => return
			},
			Location::RollingRecord(_) =>
			{
				unreachable!("a survivor is never a rolling record")
			}
		};
		slot.support = slot.support.min(support).max(1);
	}

	/// Finish the pass, as [`World::finish`](super::World) does: discard every
	/// random value never read, retire every record never summed, and scale
	/// the answer if the common factor may differ from one.
	///
	/// # Returns
	/// The event.
	fn finish(&mut self) -> Event
	{
		let cells = self.cells();
		let mut tally = Tally::default();
		for slot in &self.registers
		{
			let discard = slot.cells() > 0;
			tally.charge(discard as u128, 0);
			self.scaled |= discard;
		}
		for record in &self.records
		{
			tally.charge(record.retirement(), 0);
			self.scaled |= record.scales();
		}
		if self.scaled
		{
			let answer = self.answer.map_or(1, |answer| answer.support);
			tally.charge(answer, answer);
		}
		// One world remains.
		self.event(tally.steps, 1, cells, tally.peak)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Rolling records.                              //
////////////////////////////////////////////////////////////////////////////////

/// The bounds of a rolling record in every world.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Record
{
	/// The kind of the roll that filled the record.
	kind: Kind,

	/// A bound on the number of setups of the record's mixture.
	setups: u128,

	/// A bound on the cells of the record's mixture.
	cells: u128,

	/// A bound on the dice of all the setups of the record's mixture.
	dice: u128,

	/// The interval of the number of lowest results dropped.
	lowest: EvaluationBounds,

	/// The interval of the number of highest results dropped.
	highest: EvaluationBounds,

	/// Whether the record's current value has been summed.
	summed: bool,

	/// Whether the record's sum depends on a roll.
	random: bool
}

impl Default for Record
{
	/// Answer the bounds of a record never rolled, whose mixture holds one
	/// setup without a roll.
	fn default() -> Self
	{
		Self {
			kind: Kind::Empty,
			setups: 1,
			cells: 1,
			dice: 0,
			lowest: 0.into(),
			highest: 0.into(),
			summed: false,
			random: false
		}
	}
}

/// The kind of the roll that filled a rolling record, with the bounds of its
/// operands.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind
{
	/// No roll.
	Empty,

	/// A range.
	Range
	{
		/// The start of the range.
		start: Operand,

		/// The end of the range.
		end: Operand
	},

	/// Standard dice.
	Standard
	{
		/// The number of dice.
		count: Operand,

		/// The number of faces of each die.
		faces: Operand,

		/// A bound on the distinct counts of the setups of one number of
		/// faces.
		counts: u128,

		/// A bound on the distinct numbers of faces of the setups.
		dies: u128
	},

	/// Custom dice.
	Custom
	{
		/// The number of dice.
		count: Operand,

		/// The die.
		die: Die
	},

	/// Results fixed by a split, of at most the specified number of distinct
	/// results.
	Results(u128)
}

impl Record
{
	/// Answer the steps that retiring the record takes, as
	/// [`World::retire`](super::World) does: the [total](
	/// super::mixture::Mixture::total) of a record never summed.
	///
	/// # Returns
	/// The steps.
	fn retirement(&self) -> u128
	{
		match self.summed
		{
			true => 0,
			false => self.setups.saturating_mul(2).saturating_add(1)
		}
	}

	/// Answer whether retiring the record may fold a total other than one
	/// into the common factor.
	///
	/// # Returns
	/// `true` if it may.
	fn scales(&self) -> bool
	{
		!self.summed && (self.kind != Kind::Empty || self.setups > 1)
	}

	/// Answer the least numbers of lowest and highest results dropped, each
	/// clamped to be nonnegative.
	///
	/// # Returns
	/// The least numbers of lowest and highest results dropped.
	fn least_drops(&self) -> (u128, u128)
	{
		(
			self.lowest.min.max(0) as u128,
			self.highest.min.max(0) as u128
		)
	}

	/// Answer whether the sum of the record certainly has at least two
	/// outcomes in every world: whether every setup keeps a die of at least
	/// two faces, or a range of at least two values, without saturating, so
	/// that rolling every die at its least face and every die at its greatest
	/// reach different sums.
	///
	/// # Returns
	/// `true` if the sum is varied.
	fn varied(&self) -> bool
	{
		let dropped = self.lowest.max as i64 + self.highest.max as i64;
		let kept = |count: &Operand| count.bounds.min as i64 - dropped >= 1;
		match &self.kind
		{
			Kind::Range { start, end } =>
			{
				end.bounds.min as i64 > start.bounds.max as i64 && dropped == 0
			},
			Kind::Standard { count, faces, .. } =>
			{
				kept(count)
					&& faces.bounds.min >= 2
					&& count.bounds.max as i64 * faces.bounds.max as i64
						<= i32::MAX as i64
			},
			Kind::Custom { count, die } =>
			{
				let largest =
					die.least.unsigned_abs().max(die.greatest.unsigned_abs());
				kept(count)
					&& die.faces >= 2
					&& count.bounds.max as i64 * largest as i64
						<= i32::MAX as i64
			},
			Kind::Empty | Kind::Results(_) => false
		}
	}

	/// Answer whether the sum of the record certainly takes every value of the
	/// specified interval in every world: whether its operands and drops are
	/// fixed, and it keeps a range, or dice whose faces leave no gap, or
	/// whether it rolls standard dice of fixed faces, without drops, and a
	/// count that takes every value of its interval, so that its kept results
	/// reach every sum from their least to their greatest, and those are the
	/// ends of the interval, so that no sum saturates.
	///
	/// # Parameters
	/// - `bounds`: The interval of the sum.
	///
	/// # Returns
	/// `true` if the sum takes every value of the interval.
	fn dense(&self, bounds: EvaluationBounds) -> bool
	{
		let fixed = |op: &Operand| op.bounds.min == op.bounds.max;
		if self.lowest.min != self.lowest.max
			|| self.highest.min != self.highest.max
		{
			return false
		}
		let dropped = self.lowest.max as i64 + self.highest.max as i64;
		// The dropped dice may take the extreme faces, so the kept dice reach
		// every sum of as many dice.
		let reached = |count: &Operand, least: i32, greatest: i32| {
			let kept = count.bounds.min as i64 - dropped;
			(fixed(count) && kept >= 1)
				.then(|| (kept * least as i64, kept * greatest as i64))
		};
		let reached = match &self.kind
		{
			Kind::Range { start, end }
				if fixed(start)
					&& fixed(end)
					&& dropped == 0
					&& start.bounds.min <= end.bounds.min =>
			{
				Some((start.bounds.min as i64, end.bounds.min as i64))
			},
			Kind::Standard { count, faces, .. }
				if fixed(faces) && faces.bounds.min >= 1 =>
			{
				match fixed(count)
				{
					true => reached(count, 1, faces.bounds.min),
					// The sums of `n` dice and of `n + 1` meet, so a count that
					// takes every value of its interval, without drops, reaches
					// every sum from the least count, or zero for none, to the
					// greatest times the faces.
					false =>
					{
						(count.dense && dropped == 0 && count.bounds.max >= 1)
							.then(|| {
								(
									count.bounds.min.max(0) as i64,
									count.bounds.max as i64
										* faces.bounds.min as i64
								)
							})
					},
				}
			},
			Kind::Custom { count, die } if die.contiguous() =>
			{
				reached(count, die.least, die.greatest)
			},
			_ => None
		};
		reached.is_some_and(|(least, greatest)| {
			least == bounds.min as i64 && greatest == bounds.max as i64
		})
	}

	/// Fix the record to one of its outcomes, as a split does.
	///
	/// # Parameters
	/// - `cells`: A bound on the cells of an outcome.
	fn fix(&mut self, cells: u128)
	{
		*self = Record {
			kind: Kind::Results(cells - 1),
			setups: 1,
			cells,
			dice: 0,
			lowest: self.lowest,
			highest: self.highest,
			summed: false,
			random: self.random
		};
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Dice.                                    //
////////////////////////////////////////////////////////////////////////////////

/// The shape of one die, which bounds the sums of many of them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Die
{
	/// The number of distinct faces, which is positive.
	faces: u128,

	/// The least face.
	least: i32,

	/// The greatest face.
	greatest: i32,

	/// The greatest common divisor of the differences of the faces from the
	/// least, or one if there is only one face. Every sum of `k` dice is `k`
	/// times the least face plus a multiple of it.
	step: u128,

	/// The distinct faces, in ascending order, or `None` for a standard die,
	/// whose faces run from one to [`faces`](Self::faces).
	values: Option<Vec<i32>>
}

impl Die
{
	/// Answer the shape of a standard die of the specified number of faces.
	///
	/// # Parameters
	/// - `faces`: The number of faces, which is positive.
	///
	/// # Returns
	/// The shape.
	fn standard(faces: u128) -> Self
	{
		Self {
			faces,
			least: 1,
			greatest: faces as i32,
			step: 1,
			values: None
		}
	}

	/// Answer the shape of a custom die of the specified distinct faces.
	///
	/// # Parameters
	/// - `values`: The distinct faces, in ascending order, which may be empty.
	///
	/// # Returns
	/// The shape, which has one face, zero, if there are none.
	fn custom(values: Vec<i32>) -> Self
	{
		match (values.first(), values.last())
		{
			(Some(&least), Some(&greatest)) =>
			{
				let step = values
					.iter()
					.map(|&value| (value as i64 - least as i64) as u128)
					.fold(0, gcd)
					.max(1);
				Self {
					faces: values.len() as u128,
					least,
					greatest,
					step,
					values: Some(values)
				}
			},
			_ => Self::custom(vec![0])
		}
	}

	/// Answer the greatest face less the least.
	///
	/// # Returns
	/// The spread.
	fn spread(&self) -> u128
	{
		(self.greatest as i64 - self.least as i64) as u128
	}

	/// Answer whether the die's faces are every value from its least to its
	/// greatest, so that its sums leave no gap.
	///
	/// # Returns
	/// `true` if the faces leave no gap.
	fn contiguous(&self) -> bool { self.faces == self.spread() + 1 }

	/// Answer the number of multiples of the [step](Self::step) that the
	/// spread holds, so that `k` dice reach at most `k` times as many sums
	/// above their least, plus one.
	///
	/// # Returns
	/// The number of steps.
	fn alpha(&self) -> u128 { self.spread() / self.step }

	/// Answer the number of values between the least and the greatest sum of
	/// the specified number of dice, inclusive, which a dense array of their
	/// sums spans.
	///
	/// # Parameters
	/// - `k`: The number of dice.
	///
	/// # Returns
	/// The span.
	fn span(&self, k: u128) -> u128
	{
		self.spread().saturating_mul(k).saturating_add(1)
	}

	/// Answer the number of multiples of the step between the least and the
	/// greatest sum of the specified number of dice, inclusive.
	///
	/// # Parameters
	/// - `k`: The number of dice.
	///
	/// # Returns
	/// The number of sums.
	fn linear(&self, k: u128) -> u128
	{
		self.alpha().saturating_mul(k).saturating_add(1)
	}

	/// Answer the sum of [`linear`](Self::linear) over the dice from `a` to
	/// `b`, inclusive.
	///
	/// # Parameters
	/// - `a`: The fewest dice.
	/// - `b`: The most dice.
	///
	/// # Returns
	/// The sum, which is zero if `a > b`.
	fn linear_sum(&self, a: u128, b: u128) -> u128
	{
		match a > b
		{
			true => 0,
			false =>
			{
				let n = b - a + 1;
				let dice = (a + b).saturating_mul(n) / 2;
				self.alpha().saturating_mul(dice).saturating_add(n)
			}
		}
	}

	/// Answer a bound on the number of distinct sums of the specified number
	/// of dice, `σ(k)`: the smaller of their [linear](Self::linear) bound and
	/// the number of multisets of their faces.
	///
	/// # Parameters
	/// - `k`: The number of dice.
	///
	/// # Returns
	/// The bound.
	fn support(&self, k: u128) -> u128
	{
		self.linear(k)
			.min(binomial(k + self.faces - 1, self.faces - 1))
	}

	/// Answer whether the saturating fold of any roll of at most the specified
	/// number of dice equals one clamp of its true sum, as
	/// [`folds_to_clamp`](super::record) decides.
	///
	/// # Parameters
	/// - `count`: The greatest number of dice.
	///
	/// # Returns
	/// `true` if every such roll folds to one clamp.
	fn folds(&self, count: u128) -> bool
	{
		self.least >= 0
			|| self.greatest <= 0
			|| count as i128 * self.least as i128 >= i32::MIN as i128
	}
}

/// Answer the greatest common divisor of two numbers.
///
/// # Parameters
/// - `a`: The first number.
/// - `b`: The second number.
///
/// # Returns
/// The greatest common divisor, which is zero if both are.
fn gcd(a: u128, b: u128) -> u128
{
	match b
	{
		0 => a,
		b => gcd(b, a % b)
	}
}

/// Answer the binomial coefficient `C(n, k)`, saturating at [`u128::MAX`].
///
/// # Parameters
/// - `n`: The number of things.
/// - `k`: The number chosen.
///
/// # Returns
/// The coefficient, which is zero if `k > n`.
fn binomial(n: u128, k: u128) -> u128
{
	if k > n
	{
		return 0
	}
	let k = k.min(n - k);
	let mut c = 1u128;
	for i in 0..k
	{
		// `c · (n - i)` is divisible by `i + 1`, since `c = C(n, i)`.
		match c.checked_mul(n - i)
		{
			Some(product) => c = product / (i + 1),
			None => return u128::MAX
		}
	}
	c
}

/// Answer a bound on the base-two logarithm of the least common multiple of
/// at most `count` distinct positive integers no greater than `greatest`.
///
/// # Parameters
/// - `greatest`: The greatest integer.
/// - `count`: The number of integers.
///
/// # Returns
/// The bound, which is zero if `greatest` is at most one.
///
/// # Notes
/// The least common multiple of `1, …, x` is `e^ψ(x)`, and `ψ(x) < 1.03883 x`
/// for every positive `x` (Rosser and Schoenfeld, 1962, Theorem 12).
fn lcm_log(greatest: u128, count: u128) -> f64
{
	if greatest <= 1
	{
		return 0.0
	}
	let greatest = greatest as f64;
	(count as f64 * greatest.log2()).min(1.03883 * greatest / LN_2)
}

////////////////////////////////////////////////////////////////////////////////
//                                   Sums.                                    //
////////////////////////////////////////////////////////////////////////////////

impl Record
{
	/// Sum the record, as [`Mixture::sum`](super::mixture::Mixture::sum)
	/// does.
	///
	/// # Parameters
	/// - `limit`: The width of the interval of the sum.
	///
	/// # Returns
	/// The tally of the sum, and a bound on its support.
	fn sum(&self, limit: u128) -> (Tally, u128)
	{
		if self.setups == 1
		{
			// A record of one setup of weight one needs no scaling.
			return self.setup_sum(limit)
		}
		let (setups, outcomes) = match &self.kind
		{
			Kind::Empty | Kind::Results(_) =>
			{
				let (mut each, support) = self.setup_sum(limit);
				// The setup's sum joins the sums.
				each.charge(support, support);
				each.keep(support.saturating_mul(2), 0);
				(each.times(self.setups), support.saturating_mul(self.setups))
			},
			Kind::Range { start, end } => (self.range_sums(start, end), limit),
			Kind::Standard { .. } | Kind::Custom { .. } => self.dice_sums(limit)
		};
		// The sums hold the outcomes of the setups' sums, and no more than the
		// interval holds.
		let outcomes = outcomes.min(limit);
		let mut tally = Tally::default();
		// The denominator, and a step for each setup. The sums grow as the
		// setups join them, to at most the outcomes of the sum.
		tally.charge(self.setups.saturating_mul(2), outcomes);
		tally.then(setups);
		// The sums become the distribution.
		tally.charge(outcomes, outcomes);
		tally.keep(outcomes.saturating_mul(2), outcomes);
		(tally, outcomes)
	}

	/// Sum a record of one setup, as [`setup_sum`](super::mixture) does,
	/// whatever the values of its operands in each world.
	///
	/// # Parameters
	/// - `limit`: The width of the interval of the sum.
	///
	/// # Returns
	/// The tally of the sum, and a bound on its support.
	fn setup_sum(&self, limit: u128) -> (Tally, u128)
	{
		let mut tally = Tally::default();
		match &self.kind
		{
			Kind::Empty =>
			{
				tally.charge(1, 1);
				(tally, 1)
			},
			Kind::Results(distinct) =>
			{
				tally.charge(*distinct, 1);
				(tally, 1)
			},
			Kind::Range { start, end } =>
			{
				let widest = width((start.bounds.min, end.bounds.max).into());
				tally.charge(widest, widest);
				(tally, widest.min(limit))
			},
			Kind::Standard { count, faces, .. } => match faces.bounds.max
			{
				..=0 =>
				{
					tally.charge(1, 1);
					(tally, 1)
				},
				greatest => self.dice_sum(
					count,
					&Die::standard(greatest as u128),
					limit
				)
			},
			Kind::Custom { count, die } => self.dice_sum(count, die, limit)
		}
	}

	/// Sum a record of one setup of dice, as [`Roll::sum`](super::record::Roll)
	/// does, whatever its count and drops in each world.
	///
	/// # Parameters
	/// - `count`: The number of dice.
	/// - `die`: The die, or the greatest die of standard dice.
	/// - `limit`: The width of the interval of the sum.
	///
	/// # Returns
	/// The tally of the sum, and a bound on its support.
	fn dice_sum(&self, count: &Operand, die: &Die, limit: u128)
	-> (Tally, u128)
	{
		let mut tally = Tally::default();
		tally.charge(1, 1);
		let (lowest, highest) = self.least_drops();
		let greatest = count.bounds.max.max(0) as u128;
		if greatest == 0 || lowest + highest >= greatest
		{
			// No dice, or every die is dropped.
			return (tally, 1)
		}
		let least = count.bounds.min.max(1) as u128;
		let mut support = 1;
		if lowest == 0 && highest == 0 && die.folds(least)
		{
			let (power, len) = power_sum(die, least, greatest, limit);
			tally = tally.max(power);
			support = support.max(len);
		}
		if self.lowest.max > 0 || self.highest.max > 0 || !die.folds(greatest)
		{
			let (order, len) = order_sum(die, greatest, lowest, highest, limit);
			tally = tally.max(order);
			support = support.max(len);
		}
		(tally, support)
	}

	/// Sum a mixture of setups of ranges, as
	/// [`Mixture::sum`](super::mixture::Mixture::sum) does: a range without
	/// drops joins a difference array, which one sweep adds to the sums, and
	/// any other contributes a point mass.
	///
	/// # Parameters
	/// - `start`: The start of the ranges.
	/// - `end`: The end of the ranges.
	///
	/// # Returns
	/// The tally of the setups, whose sums the sums keep.
	fn range_sums(&self, start: &Operand, end: &Operand) -> Tally
	{
		let widest = width((start.bounds.min, end.bounds.max).into());
		// A range joins the array at a rise and a fall, which stay charged; a
		// point mass is summed, and joins the sums.
		let mut each = Tally::default();
		each.charge(2, 2);
		let mut point = Tally::default();
		point.charge(2, 2);
		point.keep(2, 0);
		let mut tally = each.max(point).times(self.setups);
		// The array rises at each distinct start, and falls after each
		// distinct end.
		let edges = self
			.setups
			.saturating_mul(2)
			.min(start.support.saturating_add(end.support));
		let sweep = widest.saturating_add(edges.saturating_mul(2));
		let mut swept = Tally::default();
		swept.charge(widest.saturating_add(edges), sweep);
		swept.keep(sweep, 0);
		tally.then(swept);
		tally
	}

	/// Sum a mixture of setups of dice, as
	/// [`Mixture::sum`](super::mixture::Mixture::sum) does. The setups without
	/// drops of each die share its convolution powers, stepping up from one
	/// count to the next, those with drops take the order statistics, and
	/// those without dice or without kept dice are point masses. Each setup is
	/// of one of these kinds, so the estimate adds the three, pricing the
	/// setups by the greatest faces and counts that they may have, and bounds
	/// the outcomes of their sums by the supports of the three. The setups of
	/// each die and count that take the order statistics differ in their
	/// drops, clamped to the dice, so they number at most the [pairs of
	/// drops](kept_pairs) that keep a die, less the pair of none when the power
	/// takes it; and since the order statistics cost no more for more drops,
	/// each is priced at the fewest drops of those pairs.
	///
	/// # Parameters
	/// - `limit`: The width of the interval of the sum.
	///
	/// # Returns
	/// The tally of the setups, whose sums the sums keep, and a bound on the
	/// outcomes of their sums together.
	fn dice_sums(&self, limit: u128) -> (Tally, u128)
	{
		let (count, dies, counts) = match &self.kind
		{
			Kind::Standard {
				count,
				faces,
				counts,
				dies
			} =>
			{
				let dies = tops(faces.bounds, 1, (*dies).min(self.setups))
					.into_iter()
					.map(|(faces, times)| (Die::standard(faces), times))
					.collect::<Vec<_>>();
				(count, dies, *counts)
			},
			Kind::Custom { count, die } =>
			{
				(count, vec![(die.clone(), 1)], count.support)
			},
			_ => unreachable!("the record holds dice")
		};
		let (lowest, highest) = self.least_drops();
		let (most_lowest, most_highest) = (
			self.lowest.max.max(0) as u128,
			self.highest.max.max(0) as u128
		);
		let counts = counts.min(self.setups);
		// The setups of each die and count differ in their drops.
		let variants = self.setups.div_ceil(
			counts
				.saturating_mul(dies.iter().map(|(_, times)| times).sum())
				.max(1)
		);
		// Every setup may be a point mass, whose sum joins the sums.
		let mut point = Tally::default();
		point.charge(2, 2);
		point.keep(2, 0);
		let mut tally = point.times(self.setups);
		let mut outcomes = self.setups;
		let least = count.bounds.min.max(1) as u128;
		let greatest = count.bounds.max.max(0) as u128;
		if greatest == 0
		{
			return (tally, outcomes)
		}
		let listed = tops(count.bounds, 1, counts);
		for (die, times) in &dies
		{
			if lowest == 0 && highest == 0 && die.folds(least)
			{
				let family = power_family(
					die,
					least,
					greatest,
					counts,
					count.dense,
					limit
				);
				tally = tally.beside(family.times(*times));
				// The powers of at most `m` distinct counts, whose supports are
				// at most those of the `m` greatest.
				let m = counts.clamp(1, greatest - least + 1);
				let supports = die
					.linear_sum(greatest - m + 1, greatest)
					.min(m.saturating_mul(die.support(greatest)));
				outcomes =
					outcomes.saturating_add(supports.saturating_mul(*times));
			}
			for &(n, per) in &listed
			{
				// The setup without drops takes the power if the die folds.
				let power = lowest == 0 && highest == 0 && die.folds(n);
				let pairs = kept_pairs(
					n,
					(lowest, most_lowest),
					(highest, most_highest)
				) - power as u128;
				if pairs == 0
				{
					continue
				}
				// The order sums are dearest at the fewest drops.
				let dearest = match power
				{
					true => vec![(1, 0), (0, 1)],
					false => vec![(lowest, highest)]
				};
				let (mut order, len) = dearest
					.into_iter()
					.filter(|&(l, h)| {
						l <= most_lowest && h <= most_highest && l + h < n
					})
					.map(|(l, h)| order_sum(die, n, l, h, limit))
					.reduce(|(a, la), (b, lb)| (a.max(b), la.max(lb)))
					.expect("a pair keeps a die");
				// The setup's sum joins the sums.
				order.charge(len, len);
				order.keep(len.saturating_mul(2), 0);
				let setups = per
					.saturating_mul(variants.min(pairs))
					.saturating_mul(*times);
				tally = tally.beside(order.times(setups));
				outcomes = outcomes.saturating_add(len.saturating_mul(setups));
			}
		}
		(tally, outcomes)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Convolution powers.                             //
////////////////////////////////////////////////////////////////////////////////

/// Convolve the sums of `a` dice with those of `b` dice, as
/// [`convolve`](super::record::convolve) does.
///
/// # Parameters
/// - `tally`: The tally.
/// - `die`: The die.
/// - `a`: The dice of the first operand, and a bound on its length.
/// - `b`: The dice of the second operand, and a bound on its length.
/// - `cap`: The most dice that any sum holds.
///
/// # Returns
/// A bound on the length of the convolution.
fn convolve(
	tally: &mut Tally,
	die: &Die,
	(a, la): (u128, u128),
	(b, lb): (u128, u128),
	cap: u128
) -> u128
{
	let dice = (a + b).min(cap);
	let pairs = la.saturating_mul(lb);
	let sums = die.span(dice).min(pairs);
	tally.charge(pairs, sums.saturating_mul(2));
	let len = sums.min(die.support(dice));
	tally.keep(sums.saturating_mul(2), len);
	len
}

/// Convolve a die with itself the specified number of times, by squaring, as
/// [`convolution_power`](super::record) does.
///
/// # Parameters
/// - `tally`: The tally, to which the die is charged.
/// - `die`: The die.
/// - `n`: The number of dice, which is positive.
/// - `cap`: The most dice that any sum holds, which bounds the sizes of the
///   squares and partial powers when `n` stands for any count up to `cap`.
///
/// # Returns
/// A bound on the length of the power.
fn convolution_power(
	tally: &mut Tally,
	die: &Die,
	mut n: u128,
	cap: u128
) -> u128
{
	let mut power = None::<(u128, u128)>;
	let mut square = (1u128, die.faces);
	loop
	{
		if n & 1 == 1
		{
			power = Some(match power
			{
				None =>
				{
					tally.charge(square.1, square.1);
					square
				},
				Some(power) =>
				{
					let len = convolve(tally, die, power, square, cap);
					tally.free(power.1);
					((power.0 + square.0).min(cap), len)
				}
			});
		}
		n >>= 1;
		if n == 0
		{
			tally.free(square.1);
			return power.expect("at least one die").1
		}
		let next = convolve(tally, die, square, square, cap);
		tally.free(square.1);
		square = ((square.0 * 2).min(cap), next);
	}
}

/// Tally the convolution power of any number of dice from `least` to
/// `greatest`, as the dearest of the exact powers of a few counts among them,
/// which bound the powers of every count.
///
/// Every count `n` from `least` to `greatest` is either `greatest`, or agrees
/// with `greatest` above some bit `j` that `greatest` sets and `n` clears. Then
/// the one bits of `n` are among those of `cⱼ`, which is `greatest` with bit
/// `j` cleared and every lower bit set, and `n ≤ cⱼ < greatest`, so `cⱼ` is
/// itself a count. The power of `cⱼ` takes every square that that of `n`
/// takes, and more if `j` is at least the width of `n`, and convolves its
/// partial powers at every bit that `n` does, and others, and its partial
/// powers hold at least as many dice, so it charges at least as many steps and
/// cells. So the dearest of the powers of `greatest`
/// and of each such `cⱼ` is the dearest of the powers of every count, and at
/// most `b + 1` powers price them, where `b` is the width of `greatest`.
///
/// # Parameters
/// - `tally`: The tally, to which the die is charged.
/// - `die`: The die.
/// - `least`: The least number of dice, which is positive.
/// - `greatest`: The greatest number of dice.
///
/// # Returns
/// A bound on the length of the power.
fn any_power(tally: &mut Tally, die: &Die, least: u128, greatest: u128)
-> u128
{
	let exact = |n: u128| {
		let mut power = Tally::default();
		let len = convolution_power(&mut power, die, n, n);
		(power, len)
	};
	let (mut dearest, mut len) = exact(greatest);
	for j in 0..u128::BITS - greatest.leading_zeros()
	{
		let bit = 1u128 << j;
		// Setting the lower bits also sets those that `greatest` sets.
		let count = (greatest ^ bit) | (bit - 1);
		if greatest & bit != 0 && count >= least
		{
			let (power, power_len) = exact(count);
			dearest = dearest.max(power);
			len = len.max(power_len);
		}
	}
	tally.then(dearest);
	len
}

/// Sum one setup of dice without drops, as
/// [`Roll::sum`](super::record::Roll::sum) does by way of
/// [`Powers::power`](super::record::Powers) with no power yet computed, and
/// then clamp the power.
///
/// # Parameters
/// - `die`: The die.
/// - `least`: The least number of dice, which is positive.
/// - `greatest`: The greatest number of dice.
/// - `limit`: The width of the interval of the sum.
///
/// # Returns
/// The tally, which leaves the die and a copy of the power in the powers, and
/// a bound on the support of the sum.
fn power_sum(
	die: &Die,
	least: u128,
	greatest: u128,
	limit: u128
) -> (Tally, u128)
{
	let mut tally = Tally::default();
	// The die, then its copy, which the powers consume.
	tally.charge(die.faces, die.faces);
	tally.charge(die.faces, die.faces);
	let len = any_power(&mut tally, die, least, greatest);
	// The powers keep a copy of the power.
	tally.charge(len, len);
	let support = clamp(&mut tally, len, limit);
	(tally, support)
}

/// Sum the setups without drops of one die, of at most `m` distinct counts
/// from `least` to `greatest`, as a mixture does, stepping each count's power
/// up from the last.
///
/// Let `σ(k)` bound the sums of `k` dice, and let there be `m'` counts, of
/// which `k = m' - 1` step up. The first count `c₁` is at most
/// `greatest - k`, and takes its power from scratch, which [`any_power`]
/// bounds. Each later count `cᵢ` takes the power of `dᵢ = cᵢ - cᵢ₋₁` dice, and
/// convolves it with the last, pairing `σ(cᵢ₋₁) · σ(dᵢ)` sums. The excesses
/// `dᵢ - 1` sum to at most `E = greatest - least - k`, which is zero when the
/// counts are consecutive, and each is at most `E`. The power of `d` dice
/// pairs at most `R(d - 1)` sums, where `R(e) = σ(e + 1)² - σ(1)²` for a die
/// of at least two faces, or `2e` for a die of one, and `R` is superadditive,
/// so the powers pair at most `R(E)`. And `σ(d) - σ(1)` is at most
/// `α · d + 1 - σ(1)`, where `α` is the number of steps in the spread of the
/// die, so the convolutions pair at most
/// `σ(1) · Σᵢ σ(cᵢ₋₁) + σ(greatest) · Δ`, where
/// `Δ = α · (greatest - least) - k · (σ(1) - 1)`.
///
/// If the counts are consecutive, then `m' = m = greatest - least + 1`. If not,
/// then `m'` may be any number up to `m`: a count may take fewer values than
/// its support bounds, as `3D[1, 2, 4]` never sums to `11`, and fewer counts
/// leave greater gaps. The first power then takes at most `greatest` dice, and
/// every bound that grows with `k` takes `k = m - 1`, and every bound that
/// shrinks with it, the excesses, `R(E)`, `Δ`, and the greatest difference,
/// takes `k = 1`, which bounds the family of every `m'`.
///
/// # Parameters
/// - `die`: The die.
/// - `least`: The least number of dice, which is positive.
/// - `greatest`: The greatest number of dice.
/// - `m`: A bound on the number of distinct counts.
/// - `consecutive`: Whether the counts certainly take every value from `least`
///   to `greatest`.
/// - `limit`: The width of the interval of the sum.
///
/// # Returns
/// The tally, whose sums the sums keep.
fn power_family(
	die: &Die,
	least: u128,
	greatest: u128,
	m: u128,
	consecutive: bool,
	limit: u128
) -> Tally
{
	let m = m.clamp(1, greatest - least + 1);
	let later = m - 1;
	let consecutive = consecutive && m == greatest - least + 1;
	let first = match consecutive
	{
		true => greatest - later,
		false => greatest
	};
	let (mut family, support) = power_sum(die, least, first, limit);
	// The setup's sum joins the sums.
	family.charge(support, support);
	family.keep(support.saturating_mul(2), 0);
	if later == 0
	{
		return family
	}
	let top = die.support(greatest);
	let one = die.support(1);
	// The fewest counts that step up, at which the excesses are greatest.
	let fewest = match consecutive
	{
		true => later,
		false => 1
	};
	let excess = greatest - least - fewest;
	let delta = die
		.alpha()
		.saturating_mul(greatest - least)
		.saturating_sub(fewest.saturating_mul(one - 1));
	let previous = die
		.linear_sum(greatest - later, greatest - 1)
		.min(later.saturating_mul(die.support(greatest - 1)));
	let current = die
		.linear_sum(greatest - later + 1, greatest)
		.min(later.saturating_mul(top));
	let powers = match die.faces
	{
		1 => excess.saturating_mul(2),
		_ =>
		{
			let widest = die.support(excess + 1);
			widest.saturating_mul(widest) - one * one
		}
	};
	let convolutions = one
		.saturating_mul(previous)
		.saturating_add(top.saturating_mul(delta));
	// Each power charges its first square.
	let squares = later.saturating_mul(one).saturating_add(delta);
	// Each later setup charges its die, the copy of the die for the powers,
	// and the copy of its power for the powers, its clamp, and its sums.
	let each = later.saturating_mul(die.faces.saturating_mul(2));
	family.charge(
		each.saturating_add(current.saturating_mul(3))
			.saturating_add(squares)
			.saturating_add(powers)
			.saturating_add(convolutions),
		0
	);
	// Stepping up follows the powers, from the die and the last power that
	// they keep: the setup's die, its copy, which becomes the power of the
	// difference, their convolution, which becomes the new power, its copy,
	// which replaces the last, its clamp, and the setup's sum, which joins the
	// sums.
	let kept = die.faces.saturating_add(top);
	let mut step = Tally {
		working: kept,
		peak: kept,
		..Tally::default()
	};
	step.charge(0, die.faces.saturating_mul(2));
	let mut difference = Tally::default();
	let len = any_power(&mut difference, die, 1, excess + 1);
	step.then(difference);
	let sums = die.span(greatest).min(top.saturating_mul(len));
	step.charge(0, sums.saturating_mul(2));
	step.keep(sums.saturating_mul(2), top);
	step.free(len);
	step.charge(0, top);
	step.free(top.saturating_add(die.faces));
	step.charge(0, top);
	step.keep(top.saturating_mul(2), top);
	step.charge(0, top);
	step.keep(top.saturating_mul(2), 0);
	family.working = family.working.max(kept);
	family.peak = family.peak.max(step.peak);
	family
}

////////////////////////////////////////////////////////////////////////////////
//                             Order statistics.                              //
////////////////////////////////////////////////////////////////////////////////

/// Sum one setup of dice by the order statistics, as
/// [`order_statistics`](super::record) does, together with the die that
/// [`Roll::sum`](super::record::Roll::sum) charges for it.
///
/// For `n` dice, of which the lowest `l` and highest `h` are dropped, the
/// states after `j` positions hold the sums of the `kⱼ` kept dice among them,
/// where `kⱼ` rises from zero after `l` positions to `K = n - l - h` after
/// `l + K`. Before the face of index `i` places, they hold at most
/// `kⱼ · (vᵢ₋₁ - v₀) / g + 1` sums, where `g` is the step of the die, and at
/// most `C(kⱼ + i - 1, i - 1)`, the multisets of the first `i` faces, and each
/// moves to `n - j + 1` states, or to one by the greatest face.
///
/// # Parameters
/// - `die`: The die.
/// - `n`: The number of dice, which is positive.
/// - `l`: The number of lowest dice dropped.
/// - `h`: The number of highest dice dropped, such that at least one die is
///   kept.
/// - `limit`: The width of the interval of the sum.
///
/// # Returns
/// The tally, and a bound on the support of the sum.
fn order_sum(die: &Die, n: u128, l: u128, h: u128, limit: u128)
-> (Tally, u128)
{
	let shape = Shape::new(n, l, h);
	let mut tally = Tally::default();
	tally.charge(die.faces, die.faces);
	let base = tally.working;
	let slots = n + 1;
	tally.charge(slots, slots + 1);
	let placements = shape.placements();
	let d = die.faces;
	// The steps in the spread of the faces placed before the face of index
	// `i`.
	let spread = |i: u128| -> u128 {
		match &die.values
		{
			None => i - 1,
			Some(values) =>
			{
				(values[i as usize - 1] as i64 - values[0] as i64) as u128
					/ die.step
			},
		}
	};
	// The sums of the states before the face of index `i`, weighed by their
	// moves, and unweighed.
	let sums = |i: u128| -> (u128, u128) {
		match i
		{
			0 => (n + 1, 1),
			i =>
			{
				let linear = shape.linear(spread(i));
				match &die.values
				{
					None => linear,
					Some(_) =>
					{
						let multisets = shape.multisets(i);
						(linear.0.min(multisets.0), linear.1.min(multisets.1))
					}
				}
			}
		}
	};
	// The states hold the slots, and their sums.
	let held = |sums: u128| slots.saturating_add(sums);
	// The steps, and the most cells alive at once.
	let mut placed = (0u128, 0u128);
	let place = |placed: &mut (u128, u128), moves: u128, held: u128| {
		placed.0 = placed
			.0
			.saturating_add(slots)
			.saturating_add(placements)
			.saturating_add(moves);
		let cells = moves.saturating_add(slots);
		placed.1 = placed.1.max(held.saturating_add(cells));
	};
	match (&die.values, d)
	{
		(_, 1) => place(&mut placed, 1, slots + 1),
		(None, d) =>
		{
			// The first face, the faces between, and the greatest face, in
			// closed form.
			place(&mut placed, n + 1, slots + 1);
			let (a, a1) = (shape.weighed(), shape.unweighed());
			let b = placements;
			if d > 2
			{
				// `Σᵢ ((i - 1) · A + B)` over `i = 1, …, d - 2`, and the most
				// cells, before the last face between.
				let between = d - 2;
				let triangle = between.saturating_mul(between - 1) / 2;
				let moves = a
					.saturating_mul(triangle)
					.saturating_add(b.saturating_mul(between));
				placed.0 = placed
					.0
					.saturating_add(
						slots.saturating_add(b).saturating_mul(between)
					)
					.saturating_add(moves);
				let last = a.saturating_mul(d - 3).saturating_add(b);
				let before = held(sums(d - 2).1);
				placed.1 = placed
					.1
					.max(before.saturating_add(last).saturating_add(slots));
			}
			let before = held(a1.saturating_mul(d - 2).saturating_add(n + 1));
			place(&mut placed, sums(d - 1).1, before);
		},
		(Some(_), d) =>
		{
			let mut before = slots + 1;
			for i in 0..d
			{
				let (weighed, unweighed) = sums(i);
				let moves = match i == d - 1
				{
					true => unweighed,
					false => weighed
				};
				place(&mut placed, moves, before);
				before = held(sums(i + 1).1);
			}
		}
	}
	tally.charge(placed.0, 0);
	tally.peak = tally.peak.max(base.saturating_add(placed.1));
	// The final states hold the sums of every kept die.
	let len = die.support(shape.kept).min(1u128 << 32);
	let states = held(sums(d).1);
	tally.keep(slots + 1, states);
	tally.charge(len, len);
	let support = len.min(limit);
	tally.keep(states.saturating_add(len), support);
	// The die is freed.
	tally.free(d);
	(tally, support)
}

/// Count the pairs of drops that keep a die of `n`, as
/// [`Mixture::drop`](super::mixture::Mixture) clamps them: the pairs `(l, h)`
/// of lowest and highest dice dropped, each clamped to `[0, n]`, such that
/// `l + h < n`.
///
/// For each `l`, `h` runs from `c` to the lesser of `d` and `n - 1 - l`, so
/// the pairs number `d - c + 1` for each `l < n - d`, and `n - l - c` for each
/// greater `l` up to `n - 1 - c`, an arithmetic series.
///
/// # Parameters
/// - `n`: The number of dice, which is positive.
/// - `(a, b)`: The least and greatest numbers of lowest dice dropped, which are
///   nonnegative and ordered.
/// - `(c, d)`: The least and greatest numbers of highest dice dropped, which
///   are nonnegative and ordered.
///
/// # Returns
/// The number of pairs.
fn kept_pairs(n: u128, (a, b): (u128, u128), (c, d): (u128, u128)) -> u128
{
	let (b, d) = (b.min(n), d.min(n));
	if a + c >= n
	{
		return 0
	}
	// The greatest `l` that keeps a die, and the least whose `h` stops short
	// of `d`.
	let top = b.min(n - 1 - c);
	let split = n - d;
	let full = (top + 1).min(split).saturating_sub(a) * (d - c + 1);
	let start = a.max(split);
	let partial = match start <= top
	{
		true => ((n - start - c) + (n - top - c)) * (top - start + 1) / 2,
		false => 0
	};
	full + partial
}

/// The shape of the kept positions of a roll: `kⱼ`, the kept dice among the
/// first `j` sorted positions, for `j = 0, …, n`.
#[derive(Debug, Clone, Copy)]
struct Shape
{
	/// The number of dice.
	n: u128,

	/// The number of lowest dice dropped.
	l: u128,

	/// The number of highest dice dropped.
	h: u128,

	/// The number of dice kept, `K = n - l - h`, which is positive.
	kept: u128
}

impl Shape
{
	/// Construct the shape.
	///
	/// # Parameters
	/// - `n`: The number of dice.
	/// - `l`: The number of lowest dice dropped.
	/// - `h`: The number of highest dice dropped.
	///
	/// # Returns
	/// The shape.
	fn new(n: u128, l: u128, h: u128) -> Self
	{
		Self {
			n,
			l,
			h,
			kept: n - l - h
		}
	}

	/// Answer `B = Σⱼ (n - j + 1)`, the placements of one face.
	///
	/// # Returns
	/// The sum.
	fn placements(&self) -> u128 { (self.n + 1) * (self.n + 2) / 2 }

	/// Answer `A = Σⱼ (n - j + 1) · kⱼ`.
	///
	/// # Returns
	/// The sum.
	fn weighed(&self) -> u128
	{
		let k = self.kept;
		// `j = l + t` for `t = 1, …, K` weighs `(N - t) · t`, where
		// `N = n - l + 1`, and `j = l + K + u` for `u = 1, …, h` weighs
		// `(h - u + 1) · K`.
		let big_n = self.n - self.l + 1;
		let rising = big_n * (k * (k + 1) / 2) - k * (k + 1) * (2 * k + 1) / 6;
		rising + k * (self.h * (self.h + 1) / 2)
	}

	/// Answer `A' = Σⱼ kⱼ`.
	///
	/// # Returns
	/// The sum.
	fn unweighed(&self) -> u128
	{
		let k = self.kept;
		k * (k + 1) / 2 + k * self.h
	}

	/// Answer the sums `Σⱼ (n - j + 1) · (kⱼ · w + 1)` and
	/// `Σⱼ (kⱼ · w + 1)`, which bound the sums of the states, weighed by their
	/// moves and not, when the faces placed so far spread over `w` steps.
	///
	/// # Parameters
	/// - `w`: The steps in the spread of the faces placed so far.
	///
	/// # Returns
	/// The weighed and unweighed sums.
	fn linear(&self, w: u128) -> (u128, u128)
	{
		(
			self.weighed()
				.saturating_mul(w)
				.saturating_add(self.placements()),
			self.unweighed()
				.saturating_mul(w)
				.saturating_add(self.n + 1)
		)
	}

	/// Answer the sums `Σⱼ (n - j + 1) · C(kⱼ + i - 1, i - 1)` and
	/// `Σⱼ C(kⱼ + i - 1, i - 1)`, which bound the sums of the states, weighed
	/// by their moves and not, after `i` distinct faces.
	///
	/// # Parameters
	/// - `i`: The number of faces placed, which is positive.
	///
	/// # Returns
	/// The weighed and unweighed sums.
	fn multisets(&self, i: u128) -> (u128, u128)
	{
		let (n, l, h, k) = (self.n, self.l, self.h, self.kept);
		let checked = || -> Option<(u128, u128)> {
			// `j ≤ l`: `kⱼ = 0`, whose one multiset is empty.
			let low = (l + 1) * (n + 1) - l * (l + 1) / 2;
			// `j = l + t`: `Σₜ (N - t) · C(t + i - 1, i - 1)` over `t = 1, …,
			// K`.
			let all = binomial(k + i, i);
			let big_n = n - l + 1;
			let rising = big_n
				.checked_mul(all.checked_sub(1)?)?
				.checked_sub(i.checked_mul(binomial(k + i, i + 1))?)?;
			// `j = l + K + u`: `C(K + i - 1, i - 1)` apiece.
			let top = binomial(k + i - 1, i - 1);
			let high = top.checked_mul(h * (h + 1) / 2)?;
			let weighed = low.checked_add(rising)?.checked_add(high)?;
			let unweighed = (l + 1)
				.checked_add(all - 1)?
				.checked_add(top.checked_mul(h)?)?;
			(all != u128::MAX && top != u128::MAX)
				.then_some((weighed, unweighed))
		};
		checked().unwrap_or((u128::MAX, u128::MAX))
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Outcomes.                                  //
////////////////////////////////////////////////////////////////////////////////

impl Record
{
	/// Enumerate the outcomes of the record, as
	/// [`Mixture::outcomes`](super::mixture::Mixture::outcomes) does.
	///
	/// # Parameters
	/// - `tally`: The tally, to which the outcomes stay charged.
	///
	/// # Returns
	/// A bound on the number of outcomes, and on the cells of each.
	fn outcomes(&self, tally: &mut Tally) -> (u128, u128)
	{
		// The denominator, and a step for each setup.
		tally.charge(self.setups.saturating_mul(2), 0);
		let mut each = Tally::default();
		let (outcomes, cells) = match &self.kind
		{
			Kind::Empty =>
			{
				each.charge(1, 1);
				(1, 1)
			},
			Kind::Results(distinct) =>
			{
				each.charge(1, distinct + 1);
				(1, distinct + 1)
			},
			Kind::Range { start, end } =>
			{
				let widest = width((start.bounds.min, end.bounds.max).into());
				each.charge(widest, widest.saturating_mul(2));
				(widest, 2)
			},
			Kind::Standard { count, faces, .. } => match faces.bounds.max
			{
				..=0 =>
				{
					each.charge(1, 2);
					(1, 2)
				},
				greatest => multisets(
					&mut each,
					count,
					&Die::standard(greatest as u128)
				)
			},
			Kind::Custom { count, die } => multisets(&mut each, count, die)
		};
		// Each outcome moves its cells into the setup that holds it.
		each.charge(outcomes, 0);
		tally.then(each.times(self.setups));
		(outcomes.saturating_mul(self.setups), cells)
	}
}

/// Enumerate the multisets of the results of dice, as
/// [`Roll::outcomes`](super::record::Roll::outcomes) does, whatever their
/// count in each world.
///
/// After the faces of index below `i`, the partial outcomes of `j` results
/// number `C(j + i - 1, i - 1)`, and all of them `C(n + i, i)`. The face of
/// index `i` grows each into `n - j + 1` others, which sum to
/// `C(n + i + 1, i + 1)`, or, if it is the greatest, completes each, with a
/// binomial of `j` steps, which sum to `i · C(n + i, i + 1) + C(n + i, i)`.
///
/// # Parameters
/// - `tally`: The tally, to which the outcomes stay charged.
/// - `count`: The number of dice.
/// - `die`: The die.
///
/// # Returns
/// A bound on the number of outcomes, and on the cells of each.
fn multisets(tally: &mut Tally, count: &Operand, die: &Die) -> (u128, u128)
{
	let n = count.bounds.max.max(0) as u128;
	if n == 0
	{
		tally.charge(1, 1);
		return (1, 1)
	}
	let d = die.faces;
	tally.charge(d, d);
	let base = tally.working;
	tally.charge(1, 1);
	// Each partial outcome after `i` faces holds at most `min(i, n)` distinct
	// results.
	let cells = |i: u128| i.min(n) + 1;
	let partials = |i: u128| binomial(n + i, i);
	// The faces before the greatest, then the greatest.
	let between = binomial(n + d, d - 1).saturating_sub(1);
	let complete = (d - 1)
		.saturating_mul(binomial(n + d - 1, d))
		.saturating_add(partials(d - 1));
	tally.charge(between.saturating_add(complete), 0);
	// The cells that each face grows increase with the face, so the most
	// alive at once are those of the first face, the last face between, or the
	// greatest face.
	let mut peak = 0u128;
	for i in [0, d.saturating_sub(2), d - 1]
	{
		let held = match i
		{
			0 => 1,
			i => partials(i).saturating_mul(cells(i))
		};
		let grown = match i == d - 1
		{
			true => partials(d - 1),
			false => binomial(n + i + 1, i + 1)
		};
		peak = peak.max(held.saturating_add(grown.saturating_mul(i + 2)));
	}
	tally.peak = tally.peak.max(base.saturating_add(peak));
	let outcomes = partials(d - 1);
	let outcome_cells = cells(d);
	tally.keep(1, outcomes.saturating_mul(outcome_cells));
	// The die is freed.
	tally.free(d);
	(outcomes, outcome_cells)
}

////////////////////////////////////////////////////////////////////////////////
//                                   Tests.                                   //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests
{
	use pretty_assertions::assert_eq;

	use super::{
		Die, LANES, Lane, Tally, any_power, convolution_power, kept_pairs,
		observe, order_sum
	};
	use crate::{
		Evaluator, Passes,
		support::{compile_valid, optimize}
	};

	/// The shape of a fork, after some step of the plan: the number of buckets
	/// of the forking lane, and, for each child, the number of its buckets and
	/// of the values of its interval, or `None` if it has not forked.
	type Shape = (usize, Vec<Option<(usize, u128)>>);

	/// Estimate the specified function, and answer the most lanes alive at
	/// once, and the shape of the top lane's fork after each step that has one.
	///
	/// # Parameters
	/// - `source`: The source of the function, which has no parameters.
	///
	/// # Returns
	/// The most lanes, and the shapes.
	fn lanes(source: &str) -> (usize, Vec<Shape>)
	{
		let function = optimize(compile_valid(source), Passes::all());
		let evaluator = Evaluator::new(function);
		let mut most = 0;
		let mut shapes = Vec::new();
		observe(&evaluator, [], |lane: &Lane| {
			most = most.max(lane.lanes());
			if let Some(fork) = &lane.fork
			{
				let children = fork
					.children
					.iter()
					.map(|(_, child)| {
						child.fork.as_ref().map(|fork| {
							let values = fork
								.children
								.iter()
								.map(|(values, _)| values)
								.sum();
							(fork.children.len(), values)
						})
					})
					.collect();
				shapes.push((fork.children.len(), children));
			}
		})
		.unwrap();
		(most, shapes)
	}

	/// Ensure that a split of more values than there are lanes takes every lane
	/// to spare, in buckets that cover its interval, and that a split of fewer
	/// takes one lane for each value.
	#[test]
	fn test_wide_and_narrow_forks()
	{
		let (most, shapes) = lanes("{x}@(1D1000) + [1:{x}]");
		assert_eq!(most, LANES);
		assert_eq!(shapes.first().map(|shape| shape.0), Some(LANES - 1));
		let (most, shapes) = lanes("{x}@(1D20) + {x}D6");
		assert_eq!(most, 21);
		assert_eq!(shapes.first().map(|shape| shape.0), Some(20));
	}

	/// Ensure that a split reserves lanes for a split nested within it, so that
	/// each forks as widely as the other, and that siblings share the lanes to
	/// spare evenly.
	#[test]
	fn test_fair_forks()
	{
		// Each of 15 buckets of `x` forks into 16 of `y`: 15² ≤ 255 < 16²,
		// and 15 · 16 = 240 lanes remain.
		let (most, shapes) =
			lanes("{x}@(1D300) + {y}@(1D300) + [{x}:{x} + {y}]");
		assert_eq!(most, LANES);
		let nested = shapes
			.iter()
			.find(|(_, children)| children.iter().all(Option::is_some))
			.expect("the children fork");
		assert_eq!(nested.0, 15);
		assert!(
			nested.1.iter().all(|child| child.unwrap().0 == 16),
			"{nested:?}"
		);
		// Where the children's intervals differ, each takes the same number of
		// buckets, or as many as its interval has values, if fewer.
		let (most, shapes) =
			lanes("{x}@(1D40) + {y}@(1D{x}) + [{x}:{x} + {y}]");
		assert!(most <= LANES, "{most}");
		let (_, children) = shapes
			.iter()
			.find(|(_, children)| children.iter().any(Option::is_some))
			.expect("the children fork");
		let level = children
			.iter()
			.flatten()
			.map(|(buckets, _)| *buckets)
			.max()
			.unwrap();
		for (buckets, values) in children.iter().flatten()
		{
			assert!(
				*buckets == level || *buckets as u128 == *values,
				"{children:?}"
			);
		}
		assert!(
			children
				.iter()
				.flatten()
				.any(|(buckets, _)| *buckets < level),
			"{children:?}"
		);
		// Three splits nest: `x` forks into 6, since 6³ ≤ 255 < 7³, each child
		// forks `y` into 6, since 6 · 6² ≤ 249 < 6 · 7², and the 36
		// grandchildren share the 213 lanes that remain, 5 apiece.
		let (most, _) = lanes(
			"{x}@(1D30) + {y}@(1D30) + {z}@(1D30) + [{x}:{y}] * [{y}:{z}] * \
			 [{z}:{x}]"
		);
		assert_eq!(most, 1 + 6 + 36 + 36 * 5);
	}

	/// Ensure that the lanes never exceed the limit, however many splits of
	/// however many values nest.
	#[test]
	fn test_lanes_within_limit()
	{
		for source in [
			"{x}@(1D300) + {y}@(1D300) + [{x}:{x} + {y}]",
			"{x}@(1D300) + {y}@(1D2) + {z}@(1D300) + [{x}:{x} + {y} + {z}]",
			"{x}@(1D30) + {y}@(1D30) + {z}@(1D30) + {x}D{y} + {y}D{z} + {z}D{x}",
			"{a}@(1D300) + {b}@(1D300) + {c}@(1D300) + {d}@(1D300) + [{a}:{b}] \
			 * [{b}:{c}] * [{c}:{d}] * [{d}:{a}]"
		]
		{
			let (most, _) = lanes(source);
			assert!(most <= LANES, "{source}: {most}");
		}
	}

	/// Ensure that the power of any number of dice from `least` to `greatest`
	/// is the dearest of the exact powers of every such count, in steps, in
	/// cells, and in length, and no dearer than the power of `2ᵇ - 1` dice,
	/// capped at `greatest`, that priced it before, where `b` is the width of
	/// `greatest`.
	#[test]
	fn test_any_power()
	{
		for die in [
			Die::standard(1),
			Die::standard(2),
			Die::standard(6),
			Die::custom(vec![1, 3, 7]),
			Die::custom(vec![-5, 0, 5])
		]
		{
			let power = |n: u128, cap: u128| {
				let mut tally = Tally::default();
				let len = convolution_power(&mut tally, &die, n, cap);
				(tally, len)
			};
			for greatest in 1..=64u128
			{
				let all = u128::MAX >> greatest.leading_zeros();
				let (old, old_len) = power(all, greatest);
				for least in 1..=greatest
				{
					let (dearest, len) = (least..=greatest)
						.map(|n| power(n, n))
						.reduce(|(a, la), (b, lb)| (a.max(b), la.max(lb)))
						.unwrap();
					let mut tally = Tally::default();
					let any = any_power(&mut tally, &die, least, greatest);
					assert_eq!(
						(tally, any),
						(dearest, len),
						"{die:?}: {least} to {greatest}"
					);
					assert!(
						tally.max(old) == old && any <= old_len,
						"{die:?}: {least} to {greatest}: {tally:?} > {old:?}"
					);
				}
			}
		}
	}

	/// Ensure that the pairs of drops that keep a die are counted exactly,
	/// whether the drops reach the dice or not.
	#[test]
	fn test_kept_pairs()
	{
		for n in 1..=8u128
		{
			for a in 0..=10
			{
				for b in a..=10
				{
					for c in 0..=10
					{
						for d in c..=10
						{
							let mut pairs = (a..=b)
								.flat_map(|l| (c..=d).map(move |h| (l, h)))
								.map(|(l, h)| (l.min(n), h.min(n)))
								.filter(|&(l, h)| l + h < n)
								.collect::<Vec<_>>();
							pairs.sort();
							pairs.dedup();
							assert_eq!(
								kept_pairs(n, (a, b), (c, d)),
								pairs.len() as u128,
								"{n}: [{a}, {b}] × [{c}, {d}]"
							);
						}
					}
				}
			}
		}
	}

	/// Ensure that the order statistics cost no more, in steps, in cells, or in
	/// length, for more drops at either end, so that the fewest drops price
	/// every pair of drops.
	#[test]
	fn test_order_sum_by_drops()
	{
		for die in [
			Die::standard(1),
			Die::standard(2),
			Die::standard(6),
			Die::custom(vec![1, 3, 7]),
			Die::custom(vec![-5, 0, 5])
		]
		{
			for n in 1..=16u128
			{
				for l in 0..n
				{
					for h in 0..n - l
					{
						let (fewer, fewer_len) =
							order_sum(&die, n, l, h, u128::MAX);
						for (l2, h2) in [(l + 1, h), (l, h + 1)]
						{
							if l2 + h2 >= n
							{
								continue
							}
							let (more, len) =
								order_sum(&die, n, l2, h2, u128::MAX);
							assert!(
								fewer.max(more) == fewer && len <= fewer_len,
								"{die:?}: {n} dice, ({l}, {h}) to ({l2}, \
								 {h2}): {more:?} > {fewer:?}"
							);
						}
					}
				}
			}
		}
	}
}
