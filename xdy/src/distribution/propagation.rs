//! # Distribution propagation
//!
//! Functionality for computing the exact [distribution](Distribution) of the
//! outcomes of a dice expression in one forward pass over its instructions,
//! in which every register holds the distribution of its value rather than one
//! value. Arguments and external variables are fixed, so they enter as point
//! masses. Each rolling record fuses its roll, drops, keeps, and sum into [one
//! operator](Roll::sum), and each arithmetic instruction combines the
//! distributions of its operands pair by pair, through the same
//! [primitives](crate::primitives) that the [evaluator](Evaluator) uses, so
//! that every edge case is inherited rather than rederived.
//!
//! A roll whose count, faces, or range endpoints are random, or whose drop or
//! keep counts are, fills its record with a weighted [mixture](Mixture) of
//! rolls with fixed operands, one for each outcome of its operands, whose sum
//! is the mixture of their sums.
//!
//! A reroll counts its branches as though each die still kept rolled in
//! advance every reroll that it might need, as
//! [`Bounds::count`](crate::Bounds::count) does, so that every path is equally
//! likely. Before any drop, the dice of a roll are independent and alike, so a
//! reroll changes only the law of each die; after a drop, it conditions
//! exactly on the multisets of the dice still kept. A reroll whose thresholds,
//! endpoints, or limit are random splits each setup of its record by their
//! joint outcomes, as a drop does by its count.
//!
//! An explode extends each die still kept into a [chain], which every
//! later drop, keep, and sum counts as one die, worth its total. A chain's
//! total behaves as a die's result does, so an explode that no later reroll or
//! explode reads reduces its chains to their totals at once, and the record
//! holds dice, or a pool, of totals; otherwise, the record holds the chains
//! whole, and a later reroll rerolls each of their kept dice alone. Rerolls
//! and explodes act on each chain alone, so after a drop the record defers
//! joining the chains still kept until a drop, a split, or a sum needs them
//! together, and only rewrites the law of what each has become. An
//! explode limited to `n` times counts the branches of each chain still kept
//! as though it rolled every extension that it might need in advance, as a
//! limited reroll does, and one without limit that matches every face
//! saturates every chain, rolling nothing. Any other explode without limit may
//! roll without bound, so the pass refuses every function in which one may
//! extend a chain, before it begins. The Lean specification proves the laws of
//! explodes that do nothing, saturate every chain, or extend the chains of any
//! roll, dropped or not, and of the rerolls of chains, and of the explodes and
//! rerolls that leave stacks of saturated chains alone, as cited below.
//!
//! Combining operands pair by pair is exact only if the operands are
//! independent. Every roll is independent of every other, so the operands of
//! an instruction are independent unless some random value reaches both, which
//! requires a random value to be read more than once. The pass conditions on
//! such a value, as its [plan](Plan) directs: it splits each of its _worlds_,
//! each a complete state of the pass, into one for each outcome of the value,
//! in which the value is fixed, and later merges them, once one value carries
//! the whole influence of the conditioned one, by mixing that value's
//! distributions over the worlds. The worlds of open splits form a stack,
//! merged last in, first out, so that the worlds merged at once differ only in
//! the split that merges. The pass conditions on a rolling record read more
//! than once, as when assembled instructions sum it twice, by its results: each
//! world fixes one multiset of them, and drops and sums it there. Values that
//! are fixed may be read freely, and one instruction may read one random value
//! as both of its operands, since its result then depends on that value alone.
//! See [`propagate_worlds`] for a worked example, with a diagram of its worlds
//! splitting and merging.
//!
//! Conditioning multiplies the worlds by the outcomes of each open split, so
//! the pass costs the most where splits nest deeply. It merges each split as
//! soon as the plan allows, rather than at the end. Its worlds never outnumber
//! the paths through the function, each of which takes one branch of every
//! range and of every die that it rolls: `worlds_le_paths_of_step`, in
//! `lean/spec/XdySpec/Cost.lean`, proves it of every function that reaches no
//! explode that may grow a chain without bound, as every function that the
//! pass builds does.
//!
//! The pass runs within a [budget](Budget) of steps and cells, which each of
//! its operations charges to a [meter](Meter) before it expands anything, so
//! that it refuses work that would exceed the budget rather than start it. A
//! static [estimate](cost::estimate) predicts those charges, and the worlds,
//! before the pass begins.
//!
//! Where no roll has random operands, every path through the function is
//! equally likely, and every outcome weighs exactly the number of paths that
//! reach it. A roll that reaches the answer contributes its branches through
//! the operands that it feeds; a roll that collapses to a point mass, or whose
//! value is never read, contributes them as a common factor. Where rolls have
//! random operands, paths may be unequally likely, as in `(1D3)D3`, and every
//! outcome weighs its probability over a total that the least common multiple
//! of the paths' denominators divides, and often equals; see [`Mixture`] and
//! [`merge`].

mod chain;
pub(crate) mod cost;
pub(crate) mod meter;
mod mixture;
pub(crate) mod plan;
pub(crate) mod record;

use std::{
	collections::HashMap,
	error::Error,
	fmt::{self, Display, Formatter},
	mem
};

use cost::Cost;
use meter::{
	Budget, Dimension, Exhausted, Halt, Meter, Progress, Usage, charge_of
};
use mixture::Mixture;
use plan::{Location, Plan, Step, rerolled};
use record::{Roll, clamp, convolve, distinct_faces};

use crate::{
	Add, AddressingMode, CanAllocate as _, CanVisitInstructions as _,
	Criterion, Distribution, Div, DropHighest, DropLowest, EvaluationBounds,
	EvaluationError, Evaluator, Exp, Explode, Instruction, InstructionVisitor,
	KeepHighest, KeepLowest, Limit, Max, Min, Mod, Mul, Neg, Predicate,
	ProgramCounter, RegisterIndex, Reroll, Return, RollCustomDice, RollRange,
	RollStandardDice, RollingRecordIndex, Sub, SumRollingRecord, Weight, add,
	div, evaluator::BoundsEvaluator, exp, max, min, r#mod, mul, neg, sub
};

////////////////////////////////////////////////////////////////////////////////
//                                   Plans.                                   //
////////////////////////////////////////////////////////////////////////////////

impl Evaluator
{
	/// Plan the computation of the exact [distribution](Distribution) of the
	/// outcomes of the function, for the given arguments and the evaluator's
	/// external variables, and estimate its cost, without computing any
	/// weight.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	///
	/// # Returns
	/// The plan, whose [estimate](DistributionPlan::estimate) a caller may
	/// weigh before it [builds](DistributionPlan::build) anything.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// disagrees with the number of formal parameters of the function.
	///
	/// # Examples
	/// Refuse a function whose estimate exceeds the budget, before any work
	/// begins:
	///
	/// ```rust
	/// use xdy::{Budget, Evaluator, compile};
	///
	/// let evaluator = Evaluator::new(compile("{n}: {n}D6")?);
	/// let budget = Budget {
	///     steps: 1_000_000,
	///     cells: 1_000_000
	/// };
	///
	/// let plan = evaluator.plan_distribution([10])?;
	/// assert!(plan.estimate().steps <= budget.steps);
	/// assert!(plan.estimate().cells <= budget.cells);
	///
	/// let plan = evaluator.plan_distribution([i32::MAX])?;
	/// assert!(plan.estimate().steps > budget.steps);
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn plan_distribution(
		&self,
		args: impl IntoIterator<Item = i32>
	) -> Result<DistributionPlan<'_>, EvaluationError<'static>>
	{
		let args = args.into_iter().collect::<Vec<_>>();
		let cost =
			cost::estimate(self, args.iter().copied()).map_err(|e| match e
			{
				PropagationError::BadArity { expected, given } =>
				{
					EvaluationError::BadArity { expected, given }
				},
				PropagationError::Exhausted(_)
				| PropagationError::Cancelled
				| PropagationError::Unbounded =>
				{
					unreachable!(
						"estimation neither meters nor reports, and prices \
						 every explode"
					)
				}
			})?;
		Ok(DistributionPlan {
			evaluator: self,
			args,
			cost
		})
	}
}

/// The plan of the computation of the exact [distribution](Distribution) of
/// the outcomes of a function, for fixed arguments and external variables,
/// which [`Evaluator::plan_distribution`] makes.
///
/// The plan [estimates](Self::estimate) the cost of the computation before it
/// computes any weight, so that a caller may refuse the function, or choose a
/// [budget](Budget), in advance. It then [builds](Self::build) the distribution
/// in one forward pass over the instructions of the function, within the
/// budget, [reporting](Progress::report) its progress to a recipient that may
/// cancel it:
///
/// ```mermaid
/// sequenceDiagram
///     participant C as Caller
///     participant E as Evaluator
///     participant P as DistributionPlan
///     participant R as Progress
///     C->>E: plan_distribution(args)
///     E-->>C: plan
///     C->>P: estimate()
///     P-->>C: cost
///     C->>P: build(budget, progress)
///     loop before each operation expands anything
///         alt the operation would exceed the budget
///             P-->>C: Err(BudgetExhausted { estimate, … })
///         else within the budget
///             P->>R: report(consumed, estimate)
///             alt Continue
///                 R-->>P: Continue
///             else Break
///                 R-->>P: Break
///                 P-->>C: Err(Cancelled)
///             end
///         end
///     end
///     P-->>C: Ok(distribution)
/// ```
///
/// # Type parameters
/// - `'eval`: The lifetime of the evaluator.
#[cfg_attr(doc, aquamarine::aquamarine)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionPlan<'eval>
{
	/// The evaluator, whose function and external variables the plan fixes.
	evaluator: &'eval Evaluator,

	/// The arguments to the function.
	args: Vec<i32>,

	/// The estimate of the cost of the forward pass.
	cost: Cost
}

impl DistributionPlan<'_>
{
	/// Answer the estimated cost of [building](Self::build) the distribution.
	///
	/// # Returns
	/// The estimate, whose steps and cells bound those that the build
	/// consumes, and whose [class](Cost::class) describes the function.
	#[inline]
	pub fn estimate(&self) -> Cost { self.cost }

	/// Build the exact distribution of the outcomes of the function, within
	/// the specified budget, reporting the progress of the build as it goes.
	///
	/// Every operation of the build charges the most steps and cells that it
	/// may take against the budget before it expands anything, so an operation
	/// that would exceed the budget is refused at once, and the refusal ends
	/// the build. The charges depend only on the function and its bindings, so
	/// a budget that suffices once always suffices, and one that covers the
	/// [estimate](Self::estimate) never refuses. The build
	/// [reports](Progress::report) its usage so far at every charge, and stops
	/// there if the report cancels it.
	///
	/// # Parameters
	/// - `budget`: The budget.
	/// - `progress`: The recipient of the reports, which may cancel the build;
	///   [`Unobserved`](crate::Unobserved) ignores them.
	///
	/// # Returns
	/// The distribution of the outcomes.
	///
	/// # Errors
	/// - [`BudgetExhausted`](BuildError::BudgetExhausted) if an operation would
	///   exceed the budget.
	/// - [`Cancelled`](BuildError::Cancelled) if the progress cancels the
	///   build.
	/// - [`Unbounded`](BuildError::Unbounded) if an explode without limit,
	///   whose predicate does not match every face, may extend a chain, before
	///   the build charges anything.
	///
	/// In every case, there is no partial distribution.
	///
	/// # Panics
	/// In debug builds, if the function is not
	/// [valid](crate::Function::validate), which every function that the
	/// compiler, the assembler, or deserialization produces is.
	///
	/// # Examples
	/// Build within a budget that covers the estimate:
	///
	/// ```rust
	/// use xdy::{Budget, Evaluator, Unobserved, Weight, compile};
	///
	/// let evaluator = Evaluator::new(compile("3D6")?);
	/// let plan = evaluator.plan_distribution([])?;
	/// let estimate = plan.estimate();
	/// let budget = Budget {
	///     steps: estimate.steps,
	///     cells: estimate.cells
	/// };
	/// let distribution = plan.build(budget, &Unobserved)?;
	/// assert_eq!(distribution.total(), &Weight::from(216u8));
	/// assert_eq!(distribution.get(10), &Weight::from(27u8));
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	///
	/// A build that would exceed its budget is refused, and the refusal
	/// carries the estimate, so that the caller may raise the budget, or give
	/// up:
	///
	/// ```rust
	/// use xdy::{BuildError, Budget, Evaluator, Unobserved, compile};
	///
	/// let evaluator = Evaluator::new(compile("{n}: {n}D6")?);
	/// let plan = evaluator.plan_distribution([i32::MAX])?;
	/// let budget = Budget {
	///     steps: 1_000_000,
	///     cells: 1_000_000
	/// };
	/// let Err(BuildError::BudgetExhausted { estimate, .. }) =
	///     plan.build(budget, &Unobserved)
	/// else
	/// {
	///     panic!("the build fits its budget");
	/// };
	/// assert_eq!(estimate, plan.estimate());
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	///
	/// A build that its progress cancels answers no distribution, as when a
	/// user presses Cancel:
	///
	/// ```rust
	/// use std::{
	///     ops::ControlFlow,
	///     sync::atomic::{AtomicBool, Ordering}
	/// };
	///
	/// use xdy::{BuildError, Budget, Cost, Evaluator, Usage, compile};
	///
	/// let cancel = AtomicBool::new(false);
	/// let progress = |consumed: Usage, estimate: &Cost| {
	///     // A progress bar would show `consumed.steps` of `estimate.steps`.
	///     if cancel.load(Ordering::Relaxed)
	///     {
	///         ControlFlow::Break(())
	///     }
	///     else
	///     {
	///         ControlFlow::Continue(())
	///     }
	/// };
	///
	/// let evaluator = Evaluator::new(compile("100D6")?);
	/// let plan = evaluator.plan_distribution([])?;
	/// assert!(plan.build(Budget::UNLIMITED, &progress).is_ok());
	///
	/// cancel.store(true, Ordering::Relaxed);
	/// assert_eq!(
	///     plan.build(Budget::UNLIMITED, &progress),
	///     Err(BuildError::Cancelled)
	/// );
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn build(
		&self,
		budget: Budget,
		progress: &dyn Progress
	) -> Result<Distribution, BuildError>
	{
		propagate_worlds(
			self.evaluator,
			self.args.iter().copied(),
			budget,
			&self.cost,
			progress
		)
		.map(|(distribution, _)| distribution)
		.map_err(|e| match e
		{
			PropagationError::BadArity { .. } =>
			{
				unreachable!("the plan checked the arity")
			},
			PropagationError::Exhausted(Exhausted {
				dimension,
				requested,
				remaining,
				consumed
			}) => BuildError::BudgetExhausted {
				estimate: self.cost,
				dimension,
				requested,
				remaining,
				consumed
			},
			PropagationError::Cancelled => BuildError::Cancelled,
			PropagationError::Unbounded => BuildError::Unbounded
		})
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Propagation.                                //
////////////////////////////////////////////////////////////////////////////////

/// Compute the exact distribution of the outcomes of the evaluator's function,
/// for the given arguments and the evaluator's external variables, by
/// propagating distributions forward through its instructions.
///
/// # Parameters
/// - `evaluator`: The evaluator, suitably [bound](Evaluator::bind).
/// - `args`: The arguments to the function.
///
/// # Returns
/// The distribution of the outcomes, which weighs each outcome by the number
/// of paths through the function that reach it if no roll or drop has a random
/// operand, and in proportion to its probability otherwise.
///
/// # Errors
/// [`BadArity`](PropagationError::BadArity) if the number of arguments
/// disagrees with the number of formal parameters of the function.
///
/// # Notes
/// The pass runs within the [unlimited budget](Budget::UNLIMITED), so its
/// work is bounded only by the function and its bindings, and reports its
/// progress to no one. When either is untrusted, use [`propagate_worlds`] with
/// a budget instead.
#[cfg(test)]
pub(crate) fn propagate(
	evaluator: &Evaluator,
	args: impl IntoIterator<Item = i32>
) -> Result<Distribution, PropagationError>
{
	let args = args.into_iter().collect::<Vec<_>>();
	let estimate = cost::estimate(evaluator, args.iter().copied())?;
	propagate_worlds(
		evaluator,
		args,
		Budget::UNLIMITED,
		&estimate,
		&meter::Unobserved
	)
	.map(|(distribution, _)| distribution)
}

/// Compute the exact distribution of the outcomes of the evaluator's function,
/// as [`propagate`] does, but within the specified budget, reporting its
/// progress as it goes, and answer the resources that the pass consumed,
/// including the most worlds that it kept at once.
///
/// The pass runs each instruction in every world, and then performs the steps
/// that its [plan](Plan) schedules after that instruction. Consider this
/// function, numbered by instruction:
///
/// ```text
/// 0  ⚅0 <- roll standard dice 1D2
/// 1  @0 <- sum rolling record ⚅0
/// 2  ⚅1 <- roll standard dice 1D2
/// 3  @1 <- sum rolling record ⚅1
/// 4  @2 <- @0 * @1
/// 5  @3 <- @0 + @2
/// 6  @4 <- @1 + @3
/// 7  return @4
/// ```
///
/// Two instructions read each of `@0` and `@1`, so the plan splits on each
/// right after its sum writes it. After instruction 5, `@3` is the only live
/// value that depends on `@0`, and the split on `@1` does not depend on `@0`,
/// so the split on `@0` merges beneath it, into `@3`. After instruction 6,
/// `@4` is the only live value that depends on `@1`, so its split merges into
/// `@4`. In the diagram, each box is a world just before the step on the edges
/// that leave it. A value that the world fixes is shown with `=`, and a random
/// value with its outcomes and their weights in braces:
///
/// ```mermaid
/// flowchart LR
///     W["@0: {1: 1, 2: 1}"]
///     W -->|"after 1: split @0"| A1["@0 = 1<br/>@1: {1: 1, 2: 1}"]
///     W -->|"after 1: split @0"| A2["@0 = 2<br/>@1: {1: 1, 2: 1}"]
///     A1 -->|"after 3: split @1"| B11["@0 = 1, @1 = 1<br/>@3 = 2"]
///     A1 -->|"after 3: split @1"| B12["@0 = 1, @1 = 2<br/>@3 = 3"]
///     A2 -->|"after 3: split @1"| B21["@0 = 2, @1 = 1<br/>@3 = 4"]
///     A2 -->|"after 3: split @1"| B22["@0 = 2, @1 = 2<br/>@3 = 6"]
///     B11 -->|"after 5: merge @0 into @3"| C1["@1 = 1<br/>@3: {2: 1, 4: 1}, so<br/>@4: {3: 1, 5: 1}"]
///     B21 -->|"after 5: merge @0 into @3"| C1
///     B12 -->|"after 5: merge @0 into @3"| C2["@1 = 2<br/>@3: {3: 1, 6: 1}, so<br/>@4: {5: 1, 8: 1}"]
///     B22 -->|"after 5: merge @0 into @3"| C2
///     C1 -->|"after 6: merge @1 into @4"| R["@4: {3: 1, 5: 2, 8: 1}"]
///     C2 -->|"after 6: merge @1 into @4"| R
/// ```
///
/// The merge on `@0` joins only the worlds that agree on `@1`, and it mixes
/// their values of `@3`. The merge on `@1` then mixes the values of `@4` from
/// the two remaining worlds. The answer weighs each outcome by the number of
/// paths that reach it. At most four worlds are alive at once, as many as
/// there are paths.
///
/// Every operation of the pass charges its steps and cells to a
/// [meter](Meter) before it expands anything, and the pass settles the cells
/// of each world after each instruction there, and of all worlds after each
/// split or merge. The charges depend only on the function and its bindings,
/// so the pass succeeds within a budget exactly when the budget covers the
/// [usage](Usage) of the pass within the unlimited one. The meter
/// [reports](Progress::report) the usage so far at every charge, before the
/// operation expands anything, and the pass stops there if the report
/// cancels it.
///
/// # Parameters
/// - `evaluator`: The evaluator, suitably [bound](Evaluator::bind).
/// - `args`: The arguments to the function.
/// - `budget`: The budget.
/// - `estimate`: The [estimate](cost::estimate) of the pass, for the same
///   evaluator and arguments, which the pass reports to the progress.
/// - `progress`: The recipient of the reports, which may cancel the pass.
///
/// # Returns
/// The distribution of the outcomes, and the usage of the pass, whose worlds
/// are one if the pass never splits.
///
/// # Errors
/// - [`BadArity`](PropagationError::BadArity) if the number of arguments
///   disagrees with the number of formal parameters of the function.
/// - [`Exhausted`](PropagationError::Exhausted) if an operation would exceed
///   the budget, which the pass refuses before the operation expands anything.
/// - [`Cancelled`](PropagationError::Cancelled) if the progress cancels the
///   pass.
/// - [`Unbounded`](PropagationError::Unbounded) if an explode without limit,
///   whose predicate does not match every face, may extend a chain, before the
///   pass charges anything.
///
/// # Panics
/// In debug builds, if the function is not [valid](crate::Function::validate),
/// which every function that the compiler, the assembler, or deserialization
/// produces is.
///
/// # Notes
/// The count of worlds bounds the space that the pass needs. It never exceeds
/// the number of paths through the function: `worlds_le_paths_of_step`, in
/// `lean/spec/XdySpec/Cost.lean`, proves it after every step of the plan, of
/// every function that reaches no explode that may grow a chain without
/// bound, as every function that the pass builds does.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub(crate) fn propagate_worlds(
	evaluator: &Evaluator,
	args: impl IntoIterator<Item = i32>,
	budget: Budget,
	estimate: &Cost,
	progress: &dyn Progress
) -> Result<(Distribution, Usage), PropagationError>
{
	let function = &evaluator.function;
	debug_assert_eq!(function.validate(), Ok(()), "{function}");
	let arity = function.arity();
	let args = args.into_iter().collect::<Vec<_>>();
	if args.len() != arity
	{
		return Err(PropagationError::BadArity {
			expected: arity,
			given: args.len()
		})
	}
	if unbounded(evaluator, &args)
	{
		return Err(PropagationError::Unbounded)
	}
	let mut world = World {
		registers: vec![Register::Point(0); function.register_count],
		records: vec![Record::default(); function.rolling_record_count],
		scale: Weight::ONE,
		result: None,
		fixed: Vec::new()
	};
	for (i, arg) in args.into_iter().enumerate()
	{
		world.registers[i] = Register::Point(arg);
	}
	for (index, value) in &evaluator.environment
	{
		world.registers[arity + *index] = Register::Point(*value);
	}
	let mut meter = Meter::new(budget, world.cells(), estimate, progress)?;
	let plan = Plan::new(&function.instructions);
	let mut worlds = vec![world];
	let mut pc = ProgramCounter::default();
	for inst in &function.instructions
	{
		let touched = touched(inst);
		// A reroll or explode keeps the chains of its record whole only if a
		// later one reads them.
		let whole = match inst
		{
			Instruction::Reroll(Reroll { dest, .. })
			| Instruction::Explode(Explode { dest, .. }) => rerolled(
				&function.instructions,
				pc,
				Location::RollingRecord(*dest)
			),
			_ => false
		};
		for world in &mut worlds
		{
			let before = world.cells_at(&touched);
			inst.visit(&mut Execution {
				world,
				meter: &mut meter,
				whole
			})?;
			meter.settle(before, world.cells_at(&touched));
		}
		for step in plan.steps(pc)
		{
			let before = cells(&worlds);
			worlds = match step
			{
				Step::Split(location) =>
				{
					let rerolled =
						rerolled(&function.instructions, pc, *location);
					split(worlds, *location, rerolled, &mut meter)?
				},
				Step::Merge {
					split,
					survivor,
					dead
				} => merge(worlds, *split, *survivor, dead, &mut meter)?
			};
			meter.settle(before, cells(&worlds));
			meter.count_worlds(worlds.len());
		}
		debug_assert_eq!(
			meter.held(),
			cells(&worlds),
			"the meter holds the cells of the worlds"
		);
		pc.allocate();
	}
	let world = worlds.pop().expect("one world remains");
	debug_assert!(worlds.is_empty(), "every split merges by the end");
	let distribution = world.finish(&mut meter)?;
	Ok((distribution, meter.usage()))
}

/// Answer whether the evaluator's function, for the given arguments and the
/// evaluator's external variables, has an explode without limit, whose
/// predicate does not match every face of its die, that may extend a chain,
/// which may then grow without bound, so that the function has no exact
/// distribution of finite weight. The [bounds](BoundsEvaluator) decide it,
/// conservatively: an explode whose random threshold may match every face,
/// or may match none, is unbounded too.
///
/// # Parameters
/// - `evaluator`: The evaluator, suitably [bound](Evaluator::bind).
/// - `args`: The arguments to the function, as many as its formal parameters.
///
/// # Returns
/// `true` if some such explode may extend a chain, `false` otherwise.
fn unbounded(evaluator: &Evaluator, args: &[i32]) -> bool
{
	let function = &evaluator.function;
	if !function
		.instructions
		.iter()
		.any(|inst| matches!(inst, Instruction::Explode(_)))
	{
		return false
	}
	// The pass fixes every binding, and an unbound external variable holds
	// zero, so every binding is a degenerate interval.
	let mut bounds = BoundsEvaluator::new(function);
	bounds
		.seed(
			args.iter().map(|arg| Some((*arg).into())),
			evaluator
				.environment
				.iter()
				.map(|(index, value)| (*index, (*value).into())),
			EvaluationBounds::default()
		)
		.expect("the arity agrees");
	for inst in &function.instructions
	{
		bounds.step(inst);
	}
	bounds.unbounded()
}

/// Answer the locations that the specified instruction reads or writes, each
/// once, whose cells the pass settles after the instruction.
///
/// # Parameters
/// - `inst`: The instruction.
///
/// # Returns
/// The locations.
fn touched(inst: &Instruction) -> Vec<Location>
{
	let mut touched = inst
		.sources()
		.into_iter()
		.filter_map(Location::read_by)
		.collect::<Vec<_>>();
	touched.push(Location::written_by(inst));
	touched.sort_by_key(|location| match location
	{
		Location::Register(reg) => (0, reg.0),
		Location::RollingRecord(rec) => (1, rec.0),
		Location::Answer => (2, 0)
	});
	touched.dedup();
	touched
}

/// Answer the cells that the specified worlds occupy.
///
/// # Parameters
/// - `worlds`: The worlds.
///
/// # Returns
/// The number of cells.
fn cells(worlds: &[World]) -> u64 { worlds.iter().map(World::cells).sum() }

////////////////////////////////////////////////////////////////////////////////
//                                  Worlds.                                   //
////////////////////////////////////////////////////////////////////////////////

/// One world of a forward pass over the instructions of a function: its whole
/// state, given the outcomes of the values of the open splits.
#[derive(Debug, Clone, PartialEq, Eq)]
struct World
{
	/// The register bank, containing the distribution of each register.
	registers: Vec<Register>,

	/// The rolling records, containing the mixture of rolls and drops of each
	/// record.
	records: Vec<Record>,

	/// The product of the total weights of every roll whose branches reach the
	/// answer other than through its operands: the rolls whose values collapse
	/// to point masses, and the rolls whose values are never read, together
	/// with the random operands that chose them. It changes no probability:
	/// `Dist.hasLaw_scale`, in `lean/spec/XdySpec/Law.lean`, proves that
	/// scaling the answer by it keeps the answer's law, and
	/// `Dist.hasLaw_mixture` that a merge, which folds it into the common
	/// denominator, mixes the law of the survivors whatever it is.
	scale: Weight,

	/// The distribution of the answer, once a [`Return`] has set it.
	result: Option<Distribution>,

	/// The outcome of the value of each open split, fixed in this world, with
	/// its weight, from the bottom of the stack.
	fixed: Vec<(Outcome, Weight)>
}

/// An outcome of the value of a split, which a world fixes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Outcome
{
	/// A value of a register.
	Value(i32),

	/// The results of a rolling record, with its drops, as a mixture of one
	/// setup of weight one.
	Record(Mixture)
}

impl Outcome
{
	/// Answer the number of cells that the outcome occupies.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u64
	{
		match self
		{
			Outcome::Value(_) => 1,
			Outcome::Record(mixture) => mixture.cells()
		}
	}
}

impl World
{
	/// Answer the number of cells that the world occupies: one for each
	/// register, and those of each random register, each rolling record, the
	/// answer, and the outcome of each open split.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u64
	{
		self.registers.len() as u64
			+ self.registers.iter().map(Register::cells).sum::<u64>()
			+ self
				.records
				.iter()
				.map(|record| record.mixture.cells())
				.sum::<u64>()
			+ self.result.as_ref().map_or(0, |d| d.len() as u64)
			+ self
				.fixed
				.iter()
				.map(|(outcome, _)| outcome.cells())
				.sum::<u64>()
	}

	/// Answer the number of cells that the values at the specified locations
	/// occupy, beyond the slot of each register, which [`cells`](Self::cells)
	/// counts once and for all.
	///
	/// # Parameters
	/// - `locations`: The locations, each once.
	///
	/// # Returns
	/// The number of cells.
	fn cells_at(&self, locations: &[Location]) -> u64
	{
		locations
			.iter()
			.map(|location| match location
			{
				Location::Register(reg) => self.registers[reg.0].cells(),
				Location::RollingRecord(rec) =>
				{
					self.records[rec.0].mixture.cells()
				},
				Location::Answer =>
				{
					self.result.as_ref().map_or(0, |d| d.len() as u64)
				},
			})
			.sum()
	}

	/// Finish the pass: account for every random value that was never read,
	/// then weigh the answer by the [common factor](Self::scale).
	///
	/// # Parameters
	/// - `meter`: The meter.
	///
	/// # Returns
	/// The distribution of the answer.
	///
	/// # Errors
	/// [`Halt`] if finishing would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If no instruction returned an answer, which every function does.
	///
	/// # Notes
	/// `Dist.hasLaw_scale`, in `lean/spec/XdySpec/Law.lean`, proves that
	/// scaling every weight of the answer by the common factor keeps its law.
	fn finish(mut self, meter: &mut Meter) -> Result<Distribution, Halt>
	{
		for register in mem::take(&mut self.registers)
		{
			self.discard(register, meter)?;
		}
		for record in mem::take(&mut self.records)
		{
			self.retire(record, meter)?;
		}
		let result = self.result.expect("every function returns an answer");
		if self.scale == Weight::ONE
		{
			return Ok(result)
		}
		let len = result.len() as u64;
		meter.charge(len, len)?;
		Ok(Distribution::from_weights(
			result
				.into_iter()
				.map(|(outcome, weight)| (outcome, weight * &self.scale))
		)
		.expect("scaling preserves outcomes"))
	}
}

/// Split every world by the outcomes of the value at the specified location,
/// which an instruction just wrote: one world for each outcome, in which the
/// location holds it fixed, and which records it and its weight atop its
/// stack. The outcomes of a rolling record are the multisets of its results.
///
/// # Parameters
/// - `worlds`: The worlds.
/// - `location`: The location, which is not the answer.
/// - `rerolled`: Whether a reroll reads the location, a rolling record, before
///   a roll refills it, so that its outcomes keep the die from which the reroll
///   draws.
/// - `meter`: The meter, which the split charges for the outcomes of each
///   world, and for the cells of each world that it makes, before it makes
///   them.
///
/// # Returns
/// The worlds of the split.
///
/// # Errors
/// [`Halt`] if the split would exceed the budget or is cancelled.
///
/// # Notes
/// `exec_split`, in `lean/spec/XdySpec/Conditioning.lean`, proves that
/// splitting never changes the law of the program, and `exec_split_sorted`,
/// in `lean/spec/XdySpec/Outcomes.lean`, that fixing a rolling record at the
/// multiset of its results, rather than at a record in the order rolled,
/// changes nothing that any later instruction can tell. `Worlds.split`, in
/// `lean/spec/XdySpec/Invariant.lean`, proves that this split keeps the
/// plan's invariant, read sorted, with worlds that are the multisets.
fn split(
	worlds: Vec<World>,
	location: Location,
	rerolled: bool,
	meter: &mut Meter
) -> Result<Vec<World>, Halt>
{
	let mut split = Vec::new();
	for world in worlds
	{
		let outcomes = world.outcomes(location, rerolled, meter)?;
		// Each world of the split copies this one, and holds its outcome twice:
		// at the location, and atop the stack.
		let cells = world.cells() as u128;
		let cells = charge_of(outcomes.iter().fold(0, |sum, (outcome, _)| {
			sum + cells + 2 * outcome.cells() as u128
		}));
		meter.charge(cells, cells)?;
		for (outcome, weight) in outcomes
		{
			let mut world = world.clone();
			world.fix(location, &outcome);
			world.fixed.push((outcome, weight));
			split.push(world);
		}
	}
	Ok(split)
}

impl World
{
	/// Answer the outcomes of the value at the specified location, each with
	/// its weight.
	///
	/// # Parameters
	/// - `location`: The location, which is not the answer, and whose value no
	///   instruction has read yet.
	/// - `rerolled`: Whether a reroll reads the location, a rolling record,
	///   before a roll refills it.
	/// - `meter`: The meter, to which the outcomes stay charged.
	///
	/// # Returns
	/// The outcomes, whose weights sum to the total weight of the value.
	///
	/// # Errors
	/// [`Halt`] if the outcomes would exceed the budget or are cancelled.
	fn outcomes(
		&self,
		location: Location,
		rerolled: bool,
		meter: &mut Meter
	) -> Result<Vec<(Outcome, Weight)>, Halt>
	{
		match location
		{
			Location::Register(reg) => match &self.registers[reg.0]
			{
				Register::Point(value) =>
				{
					meter.charge(1, 1)?;
					Ok(vec![(Outcome::Value(*value), Weight::ONE)])
				},
				Register::Random(distribution) =>
				{
					let len = distribution.len() as u64;
					meter.charge(len, len)?;
					Ok(distribution
						.iter()
						.map(|(value, weight)| {
							(Outcome::Value(value), weight.clone())
						})
						.collect())
				},
				Register::Consumed =>
				{
					unreachable!("a split follows the write of its value")
				}
			},
			Location::RollingRecord(rec) => Ok(self.records[rec.0]
				.mixture
				.outcomes(rerolled, meter)?
				.into_iter()
				.map(|(mixture, weight)| (Outcome::Record(mixture), weight))
				.collect()),
			Location::Answer => unreachable!("no instruction reads the answer")
		}
	}

	/// Fix the value at the specified location to the specified outcome.
	///
	/// # Parameters
	/// - `location`: The location, which is not the answer.
	/// - `outcome`: The outcome, one of the [outcomes](Self::outcomes) of the
	///   value at the location.
	fn fix(&mut self, location: Location, outcome: &Outcome)
	{
		match (location, outcome)
		{
			(Location::Register(reg), Outcome::Value(value)) =>
			{
				self.registers[reg.0] = Register::Point(*value);
			},
			(Location::RollingRecord(rec), Outcome::Record(mixture)) =>
			{
				self.records[rec.0].mixture = mixture.clone();
			},
			_ => unreachable!("an outcome fits the location of its value")
		}
	}
}

/// Merge the worlds of the specified split. The worlds that agree on the
/// outcomes of every other split merge into one world, in which the survivor
/// holds the mixture of its distributions. Every split above the merging one is
/// independent of it, so each such group has one world for each outcome of the
/// merging split, and its worlds differ only in the survivor and in the dead
/// values that depend on the split.
///
/// First, each world folds the weight that its dead values still hold into its
/// scale, and then its scale into its survivor. A world of weight `wᵢ`, of
/// scale `sᵢ`, whose survivor has distribution `dᵢ` of total `tᵢ`, has total
/// `Tᵢ = sᵢ · tᵢ`, and the worlds' totals differ, as the totals of rolls of
/// different counts do, so the merge scales each world's survivor to the
/// common denominator `L`, the least common multiple of their totals, before
/// it adds them, as [`Mixture`] does:
///
/// ```text
/// weight(v) = Σ wᵢ · (L / Tᵢ) · sᵢ · dᵢ(v) = Σ wᵢ · (L / tᵢ) · dᵢ(v),
/// total = W · L
/// ```
///
/// where `W` is the total weight of the worlds. Where no roll has random
/// operands, every world has the same total, so `L` is that total, and the
/// merge weighs each outcome by the number of paths that reach it. The merged
/// world's scale is one, since the scale of every world is folded into its
/// survivor. If no value survives, then the survivor of each world is the point
/// mass at zero, and the merged world's scale is `W · L`.
///
/// # Parameters
/// - `worlds`: The worlds.
/// - `split`: The position of the split in the stack, counting from the bottom.
/// - `survivor`: The location of the survivor, if any.
/// - `dead`: The locations of the dead values that depend on the split.
/// - `meter`: The meter, which the merge charges for grouping the worlds and
///   for mixing their survivors.
///
/// # Returns
/// The merged worlds, in the order of their first worlds.
///
/// # Errors
/// [`Halt`] if the merge would exceed the budget or is cancelled.
///
/// # Notes
/// `exec_merge`, in `lean/spec/XdySpec/Conditioning.lean`, proves that merging
/// preserves the law of the program if, within each group, each world combines
/// its survivor independently with one law of everything else, the same in
/// every world of the group, and that a split may merge beneath a later one
/// that does not depend on it. The plan's merges need no such hypotheses:
/// `Worlds.merge`, in `lean/spec/XdySpec/Invariant.lean`, proves that each
/// keeps the plan's invariant, so that the worlds hold the law of the program
/// before every instruction (`invariant_run`).
fn merge(
	worlds: Vec<World>,
	split: usize,
	survivor: Option<Location>,
	dead: &[Location],
	meter: &mut Meter
) -> Result<Vec<World>, Halt>
{
	let mut groups = Vec::<Vec<(Weight, World)>>::new();
	let mut positions = HashMap::<Vec<Outcome>, usize>::new();
	for mut world in worlds
	{
		for &location in dead
		{
			world.bury(location, meter)?;
		}
		// The key copies the outcomes of the other open splits.
		let key = world
			.fixed
			.iter()
			.map(|(outcome, _)| outcome.cells())
			.sum::<u64>();
		meter.charge(world.fixed.len() as u64, key)?;
		let (_, weight) = world.fixed.remove(split);
		let key = world
			.fixed
			.iter()
			.map(|(outcome, _)| outcome.clone())
			.collect();
		let position = *positions.entry(key).or_insert_with(|| {
			groups.push(Vec::new());
			groups.len() - 1
		});
		groups[position].push((weight, world));
	}
	groups
		.into_iter()
		.map(|group| mix(group, survivor, meter))
		.collect()
}

/// Mix the survivors of a group of worlds that differ only in the outcome of
/// the merging split, and answer the merged world. See [`merge`].
///
/// # Parameters
/// - `group`: The worlds, each with its weight in the merging split, whose dead
///   values that depend on the split are already buried.
/// - `survivor`: The location of the survivor, if any.
/// - `meter`: The meter, which the mixture charges a step for each outcome of
///   each survivor, and a cell for each outcome of the mixture.
///
/// # Returns
/// The merged world.
///
/// # Errors
/// [`Halt`] if the mixture would exceed the budget or is cancelled.
///
/// # Notes
/// `Dist.hasLaw_mixture`, in `lean/spec/XdySpec/Law.lean`, proves that the
/// mixture has the law of the survivors mixed over the split's outcome, over
/// any common multiple of the survivors' totals, so that each world's common
/// factor, which the denominator includes, cancels.
fn mix(
	group: Vec<(Weight, World)>,
	survivor: Option<Location>,
	meter: &mut Meter
) -> Result<World, Halt>
{
	// A survivor that a world fixes becomes a point mass, of one cell.
	let survivors = group.len() as u64;
	meter.charge(survivors, survivors)?;
	let mut parts = Vec::with_capacity(group.len());
	let mut worlds = Vec::with_capacity(group.len());
	for (weight, mut world) in group
	{
		let distribution = world.take(survivor);
		let scale = mem::replace(&mut world.scale, Weight::ONE);
		parts.push((weight, distribution, scale));
		worlds.push(world);
	}
	let outcomes = parts
		.iter()
		.map(|(_, distribution, _)| distribution.len() as u64)
		.sum::<u64>();
	meter.charge(outcomes, outcomes)?;
	let denominator = parts.iter().fold(Weight::ONE, |lcm, (_, d, scale)| {
		lcm.lcm(&(d.total() * scale))
	});
	let mixture = Distribution::from_weights(parts.iter().flat_map(
		|(weight, distribution, _)| {
			let factor = weight * denominator.exact_div(distribution.total());
			distribution
				.iter()
				.map(move |(value, w)| (value, w * &factor))
		}
	))
	.expect("a mixture has outcomes");
	let mut world = worlds.swap_remove(0);
	debug_assert!(
		worlds.iter().all(|other| *other == world),
		"the worlds of a merge differ only in the survivor and the dead"
	);
	world.put(survivor, mixture, meter)?;
	Ok(world)
}

impl World
{
	/// Bury the dead value at the specified location, which no instruction
	/// reads again: fold any weight that it still holds into the [common
	/// factor](Self::scale), and leave the location as though read, so that
	/// every world of a merge holds the same there.
	///
	/// # Parameters
	/// - `location`: The location, which is not the answer.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if burying would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `exec_bury`, in `lean/spec/XdySpec/Conditioning.lean`, proves that
	/// burying a location that no instruction reads again leaves the law of
	/// the answer unchanged.
	fn bury(
		&mut self,
		location: Location,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		match location
		{
			Location::Register(reg) =>
			{
				let old = mem::replace(
					&mut self.registers[reg.0],
					Register::Consumed
				);
				self.discard(old, meter)
			},
			Location::RollingRecord(rec) =>
			{
				let old = mem::replace(
					&mut self.records[rec.0],
					Record {
						mixture: Mixture::default(),
						summed: true
					}
				);
				self.retire(old, meter)
			},
			Location::Answer => unreachable!("the answer lives to the end")
		}
	}

	/// Take the distribution of the survivor of a merge, leaving its location
	/// as though read.
	///
	/// # Parameters
	/// - `survivor`: The location of the survivor, if any, which is not a
	///   rolling record.
	///
	/// # Returns
	/// The distribution of the survivor, or the point mass at zero if there is
	/// none.
	fn take(&mut self, survivor: Option<Location>) -> Distribution
	{
		match survivor
		{
			Some(Location::Register(reg)) =>
			{
				match mem::replace(
					&mut self.registers[reg.0],
					Register::Consumed
				)
				{
					Register::Point(value) => Distribution::point(value),
					Register::Random(distribution) => distribution,
					Register::Consumed =>
					{
						unreachable!("no instruction has read a survivor")
					}
				}
			},
			Some(Location::Answer) =>
			{
				self.result.take().expect("a Return wrote the survivor")
			},
			Some(Location::RollingRecord(_)) =>
			{
				unreachable!("a survivor is never a rolling record")
			},
			None => Distribution::point(0)
		}
	}

	/// Put the mixture of the survivors of a merge in the survivor's location,
	/// or, if there is no survivor, fold its total weight into the [common
	/// factor](Self::scale).
	///
	/// # Parameters
	/// - `survivor`: The location of the survivor, if any, which is not a
	///   rolling record.
	/// - `mixture`: The mixture.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if putting the mixture would exceed the budget or is cancelled.
	fn put(
		&mut self,
		survivor: Option<Location>,
		mixture: Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		match survivor
		{
			Some(Location::Register(reg)) => self.write(reg, mixture, meter)?,
			Some(Location::Answer) => self.result = Some(mixture),
			Some(Location::RollingRecord(_)) =>
			{
				unreachable!("a survivor is never a rolling record")
			},
			None =>
			{
				meter.charge(1, 0)?;
				self.scale *= mixture.total();
			}
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Registers.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The distribution of the value of a register.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Register
{
	/// A fixed value, which may be read any number of times.
	Point(i32),

	/// A random value, which has not yet been read. It has at least two
	/// outcomes.
	Random(Distribution),

	/// A random value, which has been read once, and is never read again.
	Consumed
}

impl Register
{
	/// Answer the number of cells that the register's value occupies beyond its
	/// slot: the outcomes of a random value.
	///
	/// # Returns
	/// The number of cells.
	fn cells(&self) -> u64
	{
		match self
		{
			Register::Random(distribution) => distribution.len() as u64,
			Register::Point(_) | Register::Consumed => 0
		}
	}
}

impl World
{
	/// Read the distribution of the specified operand, consuming it if it is
	/// random.
	///
	/// # Parameters
	/// - `op`: The operand, an immediate or a register.
	/// - `meter`: The meter, which a fixed value charges for the point mass
	///   that it becomes.
	///
	/// # Returns
	/// The distribution of the operand.
	///
	/// # Errors
	/// [`Halt`] if reading would exceed the budget or is cancelled.
	///
	/// # Panics
	/// If the operand is a random value that has already been read, which the
	/// [plan](Plan) prevents by splitting on every random value that more than
	/// one instruction reads.
	///
	/// # Notes
	/// `map_value_product_operand`, in `lean/spec/XdySpec/World.lean`, proves
	/// that this is the law of the operand in the world.
	fn read(
		&mut self,
		op: AddressingMode,
		meter: &mut Meter
	) -> Result<Distribution, Halt>
	{
		match op
		{
			AddressingMode::Immediate(value) =>
			{
				meter.charge(1, 1)?;
				Ok(Distribution::point(value.0))
			},
			AddressingMode::Register(reg) => match &self.registers[reg.0]
			{
				&Register::Point(value) =>
				{
					meter.charge(1, 1)?;
					Ok(Distribution::point(value))
				},
				Register::Random(_) => match mem::replace(
					&mut self.registers[reg.0],
					Register::Consumed
				)
				{
					Register::Random(distribution) => Ok(distribution),
					_ => unreachable!("the register is random")
				},
				Register::Consumed =>
				{
					unreachable!("the plan splits every value read twice")
				}
			},
			AddressingMode::RollingRecord(_) => unreachable!()
		}
	}

	/// Read the joint distribution of the specified operands, which are
	/// independent unless they are the same register, whose value they then
	/// share.
	///
	/// # Parameters
	/// - `op1`: The first operand, an immediate or a register.
	/// - `op2`: The second operand, an immediate or a register.
	/// - `meter`: The meter, which the pairs charge a step and a cell apiece.
	///
	/// # Returns
	/// The pairs of values of the operands, each with its weight.
	///
	/// # Errors
	/// [`Halt`] if reading would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `map_value_product_pair`, in `lean/spec/XdySpec/World.lean`, proves that
	/// this is the joint law of the operands in the world.
	fn read_pair(
		&mut self,
		op1: AddressingMode,
		op2: AddressingMode,
		meter: &mut Meter
	) -> Result<Pairs, Halt>
	{
		Ok(self
			.read_joint(&[op1, op2], meter)?
			.into_iter()
			.map(|(values, w)| ((values[0], values[1]), w))
			.collect())
	}

	/// Read the joint distribution of the specified operands, which are
	/// independent unless some are the same register, whose value they then
	/// share. Each distinct register is read once, in the order of its first
	/// operand.
	///
	/// # Parameters
	/// - `ops`: The operands, each an immediate or a register.
	/// - `meter`: The meter, which the tuples charge a step and a cell apiece.
	///
	/// # Returns
	/// The tuples of values of the operands, in the order of the operands,
	/// each with its weight. Without operands, there is one empty tuple, of
	/// weight one.
	///
	/// # Errors
	/// [`Halt`] if reading would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `bind_value_product_joint`, in `lean/spec/XdySpec/World.lean`, proves
	/// that the values of the operands in a world have `jointLaw`, which reads
	/// each distinct register once.
	fn read_joint(
		&mut self,
		ops: &[AddressingMode],
		meter: &mut Meter
	) -> Result<Vec<(Vec<i32>, Weight)>, Halt>
	{
		// The position of each operand's value among the values read.
		let mut read = Vec::<AddressingMode>::new();
		let mut positions = Vec::with_capacity(ops.len());
		let mut values = Vec::<Distribution>::new();
		for &op in ops
		{
			match read.iter().position(|&other| {
				matches!(op, AddressingMode::Register(_)) && other == op
			})
			{
				Some(position) => positions.push(position),
				None =>
				{
					positions.push(read.len());
					read.push(op);
					values.push(self.read(op, meter)?);
				}
			}
		}
		let tuples = charge_of(
			values
				.iter()
				.fold(1u128, |tuples, xs| tuples * xs.len() as u128)
		);
		meter.charge(tuples, tuples)?;
		let mut joint = vec![(Vec::<i32>::new(), Weight::ONE)];
		for xs in &values
		{
			joint = joint
				.iter()
				.flat_map(|(tuple, w)| {
					xs.iter().map(move |(x, wx)| {
						let mut tuple = tuple.clone();
						tuple.push(x);
						(tuple, w * wx)
					})
				})
				.collect();
		}
		Ok(joint
			.into_iter()
			.map(|(tuple, w)| {
				(positions.iter().map(|&p| tuple[p]).collect(), w)
			})
			.collect())
	}

	/// Set the distribution of the specified register. A distribution with
	/// one outcome becomes a fixed value, and its total weight joins the
	/// [common factor](Self::scale), so that it may be read freely.
	///
	/// # Parameters
	/// - `reg`: The register.
	/// - `distribution`: The distribution of its new value.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if writing would exceed the budget or is cancelled.
	fn write(
		&mut self,
		reg: RegisterIndex,
		distribution: Distribution,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		let register = match distribution.len()
		{
			1 =>
			{
				meter.charge(1, 0)?;
				self.scale *= distribution.total();
				Register::Point(distribution.min())
			},
			_ => Register::Random(distribution)
		};
		let old = mem::replace(&mut self.registers[reg.0], register);
		self.discard(old, meter)
	}

	/// Discard the value of a register, which is being overwritten or
	/// abandoned. If it is random and was never read, then its total weight
	/// joins the [common factor](Self::scale).
	///
	/// # Parameters
	/// - `register`: The register's value.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if discarding would exceed the budget or is cancelled.
	fn discard(
		&mut self,
		register: Register,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		if let Register::Random(distribution) = register
		{
			meter.charge(1, 0)?;
			self.scale *= distribution.total();
		}
		Ok(())
	}
}

/// The pairs of values of two operands, each with its weight.
type Pairs = Vec<((i32, i32), Weight)>;

////////////////////////////////////////////////////////////////////////////////
//                              Rolling records.                              //
////////////////////////////////////////////////////////////////////////////////

/// The state of a rolling record: the mixture of its rolls and drops, and
/// whether its current value, which its last roll or drop wrote, has been
/// summed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Record
{
	/// The mixture of the rolls and drops of the record.
	mixture: Mixture,

	/// Whether the current value of the record has been summed.
	summed: bool
}

impl World
{
	/// Fill the specified rolling record with a new roll, replacing any
	/// earlier one, as the [evaluator](Evaluator) does.
	///
	/// # Parameters
	/// - `dest`: The record.
	/// - `rolls`: The rolls, one for each outcome of the random operands, each
	///   with the number of their branches that choose it, charged to the
	///   meter.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if rolling would exceed the budget or is cancelled.
	fn roll(
		&mut self,
		dest: RollingRecordIndex,
		rolls: impl IntoIterator<Item = (Roll, Weight)>,
		meter: &mut Meter
	) -> Result<(), Halt>
	{
		let old = mem::replace(
			&mut self.records[dest.0],
			Record {
				mixture: Mixture::rolls(rolls),
				summed: false
			}
		);
		self.retire(old, meter)
	}

	/// Retire a rolling record, which is being refilled or abandoned. If it was
	/// never summed, then its total weight joins the [common
	/// factor](Self::scale); that of a record never rolled is one, unless its
	/// drops read random values.
	///
	/// # Parameters
	/// - `record`: The record.
	/// - `meter`: The meter.
	///
	/// # Errors
	/// [`Halt`] if retiring would exceed the budget or is cancelled.
	fn retire(&mut self, record: Record, meter: &mut Meter)
	-> Result<(), Halt>
	{
		if !record.summed
		{
			let total = record.mixture.total(meter)?;
			meter.charge(1, 0)?;
			self.scale *= total;
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Arithmetic.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Apply a unary operation to an operand, merging the weights of outcomes that
/// map to the same result.
///
/// # Parameters
/// - `xs`: The distribution of the operand.
/// - `op`: The operation.
/// - `meter`: The meter, which the mapping charges a step and a cell for each
///   outcome.
///
/// # Returns
/// The distribution of the result.
///
/// # Errors
/// [`Halt`] if the mapping would exceed the budget or is cancelled.
///
/// # Notes
/// `law_neg` and `law_binary_self`, in `lean/spec/XdySpec/World.lean`, prove
/// that this is the law of a negation, and of an arithmetic instruction on the
/// same register twice; `law_binary_immediate_left` and
/// `law_binary_immediate_right`, of one on a register and an immediate.
fn map(
	xs: Distribution,
	op: impl Fn(i32) -> i32,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	let len = xs.len() as u64;
	meter.charge(len, len)?;
	Ok(
		Distribution::from_weights(xs.into_iter().map(|(x, w)| (op(x), w)))
			.expect("a mapping of a distribution has outcomes")
	)
}

/// Apply a binary operation to independent operands: weigh the result of each
/// pair of outcomes by the product of their weights.
///
/// # Parameters
/// - `xs`: The distribution of the first operand.
/// - `ys`: The distribution of the second operand.
/// - `op`: The operation.
/// - `meter`: The meter, which the operation charges a step and a cell for each
///   pair.
///
/// # Returns
/// The distribution of the result.
///
/// # Errors
/// [`Halt`] if the operation would exceed the budget or is cancelled.
///
/// # Notes
/// `lift_apply`, in `lean/spec/XdySpec/Convolution.lean`, proves that this is
/// the law of the operation over independent operands.
fn pairwise(
	xs: &Distribution,
	ys: &Distribution,
	op: fn(i32, i32) -> i32,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	let pairs = charge_of(xs.len() as u128 * ys.len() as u128);
	meter.charge(pairs, pairs)?;
	Ok(Distribution::from_weights(
		xs.iter().flat_map(|(x, wx)| {
			ys.iter().map(move |(y, wy)| (op(x, y), wx * wy))
		})
	)
	.expect("a product of distributions has outcomes"))
}

/// Add or subtract independent operands, saturating, by convolving them exactly
/// and clamping once, since a saturating sum or difference of two values is
/// their true sum or difference, clamped.
///
/// # Parameters
/// - `xs`: The distribution of the first operand.
/// - `ys`: The distribution of the second operand.
/// - `negate`: Whether to subtract the second operand rather than add it.
/// - `meter`: The meter, which the operation charges for widening the operands
///   and convolving them.
///
/// # Returns
/// The distribution of the sum or difference.
///
/// # Errors
/// [`Halt`] if the operation would exceed the budget or is cancelled.
///
/// # Notes
/// `lift_add` and `lift_sub`, in `lean/spec/XdySpec/Convolution.lean`, prove
/// this sound.
fn saturating_sum(
	xs: Distribution,
	ys: Distribution,
	negate: bool,
	meter: &mut Meter
) -> Result<Distribution, Halt>
{
	let wide = (xs.len() + ys.len()) as u64;
	meter.charge(wide, wide)?;
	let xs = xs
		.into_iter()
		.map(|(x, w)| (x as i64, w))
		.collect::<Vec<_>>();
	let ys = match negate
	{
		false => ys
			.into_iter()
			.map(|(y, w)| (y as i64, w))
			.collect::<Vec<_>>(),
		true => ys
			.into_iter()
			.rev()
			.map(|(y, w)| (-(y as i64), w))
			.collect()
	};
	let sum = convolve(&xs, &ys, meter)?;
	meter.free(wide);
	clamp(sum, meter)
}

////////////////////////////////////////////////////////////////////////////////
//                               Instructions.                                //
////////////////////////////////////////////////////////////////////////////////

/// The execution of an instruction in one world, which charges its work to
/// the meter of the pass.
struct Execution<'a, 'b>
{
	/// The world.
	world: &'a mut World,

	/// The meter.
	meter: &'a mut Meter<'b>,

	/// Whether a later reroll or explode reads the rolling record that a
	/// reroll or explode writes, before a roll refills it, so that the
	/// [chains](chain) of the record stay whole, rather than reduce to their
	/// totals.
	whole: bool
}

impl Execution<'_, '_>
{
	/// Apply a binary operation to the specified operands, and write the
	/// result to the specified register.
	///
	/// When both operands are the same register, as the strength reducer makes
	/// them when it rewrites `x * 2` as `x + x`, or `x ^ 2` as `x * x`, the
	/// result depends on one value alone, so the operation applies to each of
	/// its outcomes paired with itself, and the register is read once.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	/// - `op`: The operation, over values.
	/// - `combine`: The operation, over the distributions of independent
	///   operands, which charges its work to the meter.
	///
	/// # Errors
	/// [`Halt`] if the operation would exceed the budget or is cancelled.
	///
	/// # Notes
	/// `law_binary_operands`, in `lean/spec/XdySpec/World.lean`, proves that
	/// the result has the law of the operation over the operands' joint law:
	/// lemma 1's lift for distinct registers (`law_binary`), and the operand's
	/// law, mapped, for the same register twice (`law_binary_self`).
	fn binary(
		&mut self,
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode,
		op: fn(i32, i32) -> i32,
		combine: impl FnOnce(
			Distribution,
			Distribution,
			&mut Meter
		) -> Result<Distribution, Halt>
	) -> Result<(), Halt>
	{
		let Self { world, meter, .. } = self;
		let result = match op1
		{
			AddressingMode::Register(_) if op1 == op2 =>
			{
				map(world.read(op1, meter)?, |x| op(x, x), meter)?
			},
			_ =>
			{
				let xs = world.read(op1, meter)?;
				let ys = world.read(op2, meter)?;
				combine(xs, ys, meter)?
			}
		};
		world.write(dest, result, meter)
	}

	/// Resolve the predicate and limit of a reroll or an explode by each joint
	/// outcome of their operands.
	///
	/// # Parameters
	/// - `operands`: The operands of the predicate and the limit.
	/// - `predicate`: The predicate.
	/// - `limit`: The limit.
	///
	/// # Returns
	/// The resolutions: each set of results that the predicate matches, with
	/// the limit, or `None` for no limit, and the weight of the joint outcome
	/// of the operands that chose them.
	///
	/// # Errors
	/// [`Halt`] if resolving would exceed the budget or is cancelled.
	fn resolve(
		&mut self,
		operands: &[AddressingMode],
		predicate: &Predicate,
		limit: Limit
	) -> Result<Vec<(Criterion, Option<i32>, Weight)>, Halt>
	{
		let Self { world, meter, .. } = self;
		let joint = world.read_joint(operands, meter)?;
		// Each joint outcome of the operands resolves the predicate, which
		// holds a copy of a list of literals.
		let listed = match predicate
		{
			Predicate::In(values) => values.len() as u128,
			_ => 0
		};
		let resolved = charge_of(joint.len() as u128 * (1 + listed));
		meter.charge(resolved, resolved)?;
		Ok(joint
			.into_iter()
			.map(|(values, w)| {
				let value = |op: AddressingMode| {
					values[operands.iter().position(|&o| o == op).unwrap()]
				};
				let criterion = predicate.resolve(value);
				let limit = match limit
				{
					Limit::Times(count) => Some(value(count)),
					Limit::Unlimited => None
				};
				(criterion, limit, w)
			})
			.collect())
	}
}

impl InstructionVisitor<Halt> for Execution<'_, '_>
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), Halt>
	{
		// `law_rollRange_operands`, in `lean/spec/XdySpec/World.lean`, proves
		// that the record has the law of the range at each pair of endpoints,
		// mixed over their joint law.
		let Self { world, meter, .. } = self;
		let ends = world.read_pair(inst.start, inst.end, meter)?;
		// Each pair of endpoints makes one setup, of one cell.
		let setups = ends.len() as u64;
		meter.charge(setups, setups)?;
		world.roll(
			inst.dest,
			ends.into_iter()
				.map(|((start, end), w)| (Roll::Range { start, end }, w)),
			meter
		)
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), Halt>
	{
		// `law_rollStandardDice`, in `lean/spec/XdySpec/World.lean`, proves
		// that the record has the law of the dice at each pair of count and
		// faces, mixed over their joint law.
		let Self { world, meter, .. } = self;
		let dice = world.read_pair(inst.count, inst.faces, meter)?;
		// Each pair of count and faces makes one setup, of one cell, and
		// charges its dice.
		let steps = dice.iter().fold(0u128, |steps, ((count, _), _)| {
			steps + 1 + (*count).max(0) as u128
		});
		meter.charge(charge_of(steps), dice.len() as u64)?;
		world.roll(
			inst.dest,
			dice.into_iter().map(|((count, faces), w)| {
				(Roll::Standard { count, faces }, w)
			}),
			meter
		)
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), Halt>
	{
		// `law_rollCustomDice`, in `lean/spec/XdySpec/World.lean`, proves that
		// the record has the law of the dice at each count, mixed over its law.
		let Self { world, meter, .. } = self;
		let counts = world.read(inst.count, meter)?;
		let faces = inst.faces.len() as u128;
		// The distinct faces take a step for each face. Each count makes one
		// setup, which holds a copy of them, and charges its dice.
		let (steps, cells) =
			counts
				.iter()
				.fold((faces, faces), |(steps, cells), (count, _)| {
					(steps + 1 + count.max(0) as u128, cells + 1 + faces)
				});
		meter.charge(charge_of(steps), charge_of(cells))?;
		let faces = distinct_faces(&inst.faces);
		world.roll(
			inst.dest,
			counts.into_iter().map(|(count, w)| {
				let faces = faces.clone();
				(Roll::Custom { count, faces }, w)
			}),
			meter
		)
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), Halt>
	{
		// `law_dropLowest`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the drop over independent records and counts.
		let Self { world, meter, .. } = self;
		let counts = world.read(inst.count, meter)?;
		let record = &mut world.records[inst.dest.0];
		record.mixture.drop_lowest(&counts, meter)?;
		// The drop writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), Halt>
	{
		// `law_dropHighest`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the drop over independent records and counts.
		let Self { world, meter, .. } = self;
		let counts = world.read(inst.count, meter)?;
		let record = &mut world.records[inst.dest.0];
		record.mixture.drop_highest(&counts, meter)?;
		// The drop writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_keep_lowest(&mut self, inst: &KeepLowest) -> Result<(), Halt>
	{
		// `law_keepLowest`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the keep over independent records and counts.
		let Self { world, meter, .. } = self;
		let counts = world.read(inst.count, meter)?;
		let record = &mut world.records[inst.dest.0];
		record.mixture.keep_lowest(&counts, meter)?;
		// The keep writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_keep_highest(&mut self, inst: &KeepHighest) -> Result<(), Halt>
	{
		// `law_keepHighest`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the keep over independent records and counts.
		let Self { world, meter, .. } = self;
		let counts = world.read(inst.count, meter)?;
		let record = &mut world.records[inst.dest.0];
		record.mixture.keep_highest(&counts, meter)?;
		// The keep writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_reroll(&mut self, inst: &Reroll) -> Result<(), Halt>
	{
		// `law_reroll`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the reroll over independent records and joint
		// outcomes of the operands, and `law_reroll_resolved`, in
		// `lean/spec/XdySpec/Reroll.lean`, and `mixtureLaw_reroll_whole` and
		// `mixtureLaw_reroll_totals`, in `lean/spec/XdySpec/Stacks.lean`, that
		// rerolling every setup by each resolution of the operands describes
		// it, as every later instruction reads the record, for the rerolls
		// that `Setup.law_reroll` covers.
		let rerolls =
			self.resolve(&inst.operands(), &inst.predicate, inst.limit)?;
		let Self {
			world,
			meter,
			whole
		} = self;
		let record = &mut world.records[inst.dest.0];
		record.mixture.reroll(&rerolls, *whole, meter)?;
		// The reroll writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_explode(&mut self, inst: &Explode) -> Result<(), Halt>
	{
		// `law_explode`, in `lean/spec/XdySpec/World.lean`, proves that the
		// record has the law of the explode over independent records and joint
		// outcomes of the operands, and `law_explode_resolved`,
		// `mixtureLaw_burst_whole`, and `mixtureLaw_burst_totals`, in
		// `lean/spec/XdySpec/Stacks.lean`, that exploding every setup by each
		// resolution of the operands describes it, as every later instruction
		// reads the record, for the explodes that `Setup.law_burst` covers.
		let explodes =
			self.resolve(&inst.operands(), &inst.predicate, inst.limit)?;
		let Self {
			world,
			meter,
			whole
		} = self;
		let record = &mut world.records[inst.dest.0];
		record.mixture.explode(&explodes, *whole, meter)?;
		// The explode writes a new value, which no sum has read yet.
		record.summed = false;
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), Halt>
	{
		// `law_sumRollingRecord`, in `lean/spec/XdySpec/World.lean`, proves
		// that the register has the law of the record's sum.
		let Self { world, meter, .. } = self;
		// A record summed before is fixed, since the plan splits on every
		// random record read twice, so summing it again adds no weight.
		let record = &mut world.records[inst.src.0];
		record.summed = true;
		let sum = record.mixture.sum(meter)?;
		world.write(inst.dest, sum, meter)
	}

	// Each arithmetic instruction computes its law with `binary`, which
	// `law_binary_operands`, in `lean/spec/XdySpec/World.lean`, proves sound.
	fn visit_add(&mut self, inst: &Add) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, add, |xs, ys, meter| {
			saturating_sum(xs, ys, false, meter)
		})
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, sub, |xs, ys, meter| {
			saturating_sum(xs, ys, true, meter)
		})
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, mul, |xs, ys, meter| {
			pairwise(&xs, &ys, mul, meter)
		})
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, div, |xs, ys, meter| {
			pairwise(&xs, &ys, div, meter)
		})
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, r#mod, |xs, ys, meter| {
			pairwise(&xs, &ys, r#mod, meter)
		})
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, exp, |xs, ys, meter| {
			pairwise(&xs, &ys, exp, meter)
		})
	}

	fn visit_max(&mut self, inst: &Max) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, max, |xs, ys, meter| {
			pairwise(&xs, &ys, max, meter)
		})
	}

	fn visit_min(&mut self, inst: &Min) -> Result<(), Halt>
	{
		self.binary(inst.dest, inst.op1, inst.op2, min, |xs, ys, meter| {
			pairwise(&xs, &ys, min, meter)
		})
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), Halt>
	{
		// `law_neg`, in `lean/spec/XdySpec/World.lean`, proves that the
		// register has the operand's law, negated.
		let Self { world, meter, .. } = self;
		let xs = world.read(inst.op, meter)?;
		let result = map(xs, neg, meter)?;
		world.write(inst.dest, result, meter)
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), Halt>
	{
		// `run_eq_forward`, in `lean/spec/XdySpec/Invariant.lean`, proves that
		// the function answers the law of the returned register, mixed over
		// the worlds.
		let Self { world, meter, .. } = self;
		world.result = Some(world.read(inst.src, meter)?);
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The reason that [`propagate_worlds`] answers no distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PropagationError
{
	/// The number of arguments disagrees with the number of formal parameters.
	BadArity
	{
		/// The number of formal parameters.
		expected: usize,

		/// The number of arguments given.
		given: usize
	},

	/// An operation would have exceeded the budget, and was refused before
	/// it expanded anything.
	Exhausted(Exhausted),

	/// The [progress](Progress) cancelled the pass, before the operation that
	/// it charged last expanded anything.
	Cancelled,

	/// An explode without limit, whose predicate does not match every face,
	/// may extend a chain, which may then grow without bound, so the pass
	/// refused the function before it charged anything.
	Unbounded
}

/// The reason that [`DistributionPlan::build`] answers no distribution.
/// Whatever the reason, the build stopped before the operation that it charged
/// last, if any, expanded anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildError
{
	/// An operation would have exceeded the budget, and was refused before it
	/// expanded anything. This reports exhaustion, not a defect: the function
	/// asked for more work than the build allows, and is not thereby wrong.
	BudgetExhausted
	{
		/// The [estimate](DistributionPlan::estimate) of the whole build.
		estimate: Cost,

		/// The dimension of the budget that the operation would have
		/// exceeded.
		dimension: Dimension,

		/// The steps or cells that the operation requested.
		requested: u64,

		/// The steps or cells that remained of the budget, fewer than
		/// `requested`.
		remaining: u64,

		/// The steps already taken, or the cells alive, when the operation
		/// was refused.
		consumed: u64
	},

	/// The [progress](Progress) cancelled the build.
	Cancelled,

	/// An [explode](crate::Explode) without limit, whose predicate does not
	/// match every face of its die, may extend a chain, which may then hold
	/// any number of dice, so that the function has no exact distribution of
	/// finite weight, and the build refused it before it charged anything.
	/// This reports no defect, and no greater budget would help;
	/// [sample](crate::Evaluator::sample) the function instead. The build
	/// decides it by the [bounds](crate::Evaluator::bounds), conservatively,
	/// so it also refuses an explode whose random threshold may match every
	/// face or none. Limited explodes, and unlimited ones that match every
	/// face, and so saturate, are built. The
	/// [estimate](DistributionPlan::estimate) of a function that the build
	/// refuses so exceeds every budget but the [unlimited](Budget::UNLIMITED)
	/// one.
	Unbounded
}

impl Display for BuildError
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		match self
		{
			BuildError::BudgetExhausted {
				estimate,
				dimension,
				requested,
				remaining,
				consumed
			} => write!(
				f,
				"distribution budget of {dimension} exhausted: {requested} \
				 requested, {remaining} remaining, {consumed} consumed, of an \
				 estimated {} steps and {} cells",
				estimate.steps, estimate.cells
			),
			BuildError::Cancelled => write!(f, "distribution build cancelled"),
			BuildError::Unbounded => write!(
				f,
				"no exact distribution: an explode without limit may grow a \
				 chain without bound, so sample the function instead"
			)
		}
	}
}

impl Error for BuildError {}

impl From<Halt> for PropagationError
{
	fn from(halt: Halt) -> Self
	{
		match halt
		{
			Halt::Exhausted(exhausted) => Self::Exhausted(exhausted),
			Halt::Cancelled => Self::Cancelled
		}
	}
}
