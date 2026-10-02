//! # Fan-out plans
//!
//! Combining the operands of an instruction pair by pair is exact only if the
//! operands are independent, and they are unless some random value reaches
//! both. A random value that two or more instructions read _fans out_, and its
//! influence may reconverge downstream, as in `x@(3D6) + {x} * 3`, whose
//! operands both depend on `x`. The forward pass handles such a value by
//! _conditioning_ on it: it splits its state into one world for each outcome
//! of the value, in which the value is fixed, so that the values downstream of
//! it are again independent, and later merges those worlds into one, at the
//! first instruction after which at most one live value depends on it. That
//! value then carries the whole influence of the conditioned one, and the
//! merge mixes its distributions over the worlds. This is variable elimination
//! from Bayesian-network inference, specialized to straight-line code.
//!
//! A [plan](Plan) schedules the splits and merges before the pass runs. It
//! reasons about _values_ rather than registers, since register coalescence
//! reuses registers, so that the instructions are not in static single
//! assignment form: every write to a register or a rolling record, including
//! every drop, which rewrites its record, makes a new value, and every read
//! reads the value most recently written there. The instructions never branch
//! or loop, so this numbering is exact.
//!
//! The pass keeps its worlds as a stack of splits, which merge last split,
//! first merged, so that the worlds merged at once always share every other
//! split. A split may merge beneath a later one only if the later one does not
//! depend on it, since its worlds are then identical whatever the earlier
//! split fixes; otherwise the merge waits for the later split to merge first.

use std::collections::{BTreeSet, HashMap};

use crate::{
	AddressingMode, Instruction, ProgramCounter, RegisterIndex,
	RollingRecordIndex
};

////////////////////////////////////////////////////////////////////////////////
//                                   Plans.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The splits and merges that the forward pass performs after each
/// instruction, so that every random value that fans out is conditioned on
/// until its influence reconverges.
///
/// # Examples
/// The function of the [worked example](super::propagate_worlds) has this
/// plan, by instruction, omitting the instructions without steps:
///
/// ```text
/// 1  Split(@0)
/// 3  Split(@1)
/// 5  Merge { split: 0, survivor: @3, dead: [@0, @2] }
/// 6  Merge { split: 0, survivor: @4, dead: [@1, @2, @3] }
/// ```
///
/// The first merge rotates the split on `@0`, at the bottom of the stack,
/// beneath the split on `@1`. The second merge then finds the split on `@1` at
/// the bottom. The dead values of each merge are the ones that depend on its
/// split and still occupy their registers. Here each is fixed or already read
/// in every world, so burying them folds no weight into the worlds' scales.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan(Vec<Vec<Step>>);

/// What the forward pass does after an instruction, beyond executing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Step
{
	/// Condition on the value that the instruction just wrote to the location:
	/// split every world into one for each of its outcomes, in which it is
	/// fixed, and push the split atop the stack.
	Split(Location),

	/// Merge the worlds of a split, and remove it from the stack. Every split
	/// above it is independent of it.
	Merge
	{
		/// The position of the split in the stack, counting from the bottom.
		split: usize,

		/// The location of the only live value that depends on the split, if
		/// any, which is never a rolling record.
		survivor: Option<Location>,

		/// The locations of the other values that depend on the split and
		/// still occupy their locations, in the order written,
		/// including the split value itself unless a later write
		/// displaced it. Each is dead, but may yet hold weight that no
		/// reader took, which each world folds into its scale before
		/// the merge. Every other location holds the same
		/// value in every world that the merge joins.
		dead: Vec<Location>
	}
}

/// A location that holds a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Location
{
	/// A register.
	Register(RegisterIndex),

	/// A rolling record.
	RollingRecord(RollingRecordIndex),

	/// The answer, which [`Return`](crate::Return) writes.
	Answer
}

impl Location
{
	/// Answer the location that the specified operand reads, if any.
	///
	/// # Parameters
	/// - `op`: The operand.
	///
	/// # Returns
	/// The location, or `None` if the operand is an immediate.
	pub(crate) fn read_by(op: AddressingMode) -> Option<Self>
	{
		match op
		{
			AddressingMode::Immediate(_) => None,
			AddressingMode::Register(reg) => Some(Location::Register(reg)),
			AddressingMode::RollingRecord(rec) =>
			{
				Some(Location::RollingRecord(rec))
			},
		}
	}

	/// Answer the location that the specified instruction writes.
	///
	/// # Parameters
	/// - `inst`: The instruction.
	///
	/// # Returns
	/// The location, which is the answer for a [`Return`](crate::Return).
	pub(crate) fn written_by(inst: &Instruction) -> Self
	{
		match inst.destination()
		{
			Some(AddressingMode::Register(reg)) => Location::Register(reg),
			Some(AddressingMode::RollingRecord(rec)) =>
			{
				Location::RollingRecord(rec)
			},
			Some(AddressingMode::Immediate(_)) => unreachable!(),
			None => Location::Answer
		}
	}
}

impl Plan
{
	/// Plan the splits and merges of the specified instructions.
	///
	/// # Parameters
	/// - `instructions`: The instructions of a function.
	///
	/// # Returns
	/// The plan, which merges every split by the last instruction.
	pub(crate) fn new(instructions: &[Instruction]) -> Self
	{
		let values = Values::number(instructions);
		Self(values.schedule(instructions.len()))
	}

	/// Answer the steps that follow the specified instruction.
	///
	/// # Parameters
	/// - `pc`: The program counter of the instruction.
	///
	/// # Returns
	/// The steps, in the order that the pass performs them.
	pub(crate) fn steps(&self, pc: ProgramCounter) -> &[Step] { &self.0[pc.0] }
}

////////////////////////////////////////////////////////////////////////////////
//                                  Values.                                   //
////////////////////////////////////////////////////////////////////////////////

/// One write to a location.
#[derive(Debug)]
struct Value
{
	/// The location that holds the value.
	location: Location,

	/// Whether the value may be random: whether some roll reaches it. Values
	/// that no roll reaches are fixed by the arguments and externals.
	random: bool,

	/// The instructions that read the value, in ascending order, each once,
	/// even if it reads the value as more than one operand.
	readers: Vec<ProgramCounter>,

	/// Whether the value lives until the end of the function, as the last
	/// answer does.
	lasting: bool,

	/// The values that [fan out](Value::fans_out) on which the value depends,
	/// by index, including itself if it fans out.
	depends_on: BTreeSet<usize>
}

impl Value
{
	/// Answer whether the value fans out: whether it may be random, and more
	/// than one instruction reads it.
	///
	/// # Returns
	/// `true` if the value fans out, `false` otherwise.
	fn fans_out(&self) -> bool { self.random && self.readers.len() > 1 }

	/// Answer whether the value is live after the specified instruction:
	/// whether a later instruction reads it, or it lasts until the end.
	///
	/// # Parameters
	/// - `pc`: The program counter of the instruction.
	///
	/// # Returns
	/// `true` if the value is live, `false` otherwise.
	fn is_live_after(&self, pc: ProgramCounter) -> bool
	{
		self.lasting || self.readers.last().is_some_and(|last| *last > pc)
	}
}

/// The values of a function, numbered in the order that its instructions write
/// them, with the values that each instruction reads and writes.
#[derive(Debug)]
struct Values
{
	/// The values, in the order written.
	values: Vec<Value>,

	/// The values that each instruction reads, by index, each once. Locations
	/// never written hold fixed values, which are omitted.
	reads: Vec<Vec<usize>>,

	/// The value that each instruction writes, by index.
	writes: Vec<usize>
}

impl Values
{
	/// Number the values of the specified instructions.
	///
	/// # Parameters
	/// - `instructions`: The instructions of a function.
	///
	/// # Returns
	/// The values.
	fn number(instructions: &[Instruction]) -> Self
	{
		let mut values = Vec::<Value>::new();
		let mut reads = Vec::with_capacity(instructions.len());
		let mut writes = Vec::with_capacity(instructions.len());
		let mut current = HashMap::<Location, usize>::new();
		for (pc, inst) in instructions.iter().enumerate()
		{
			let pc = ProgramCounter(pc);
			let mut read = Vec::<usize>::new();
			for location in
				inst.sources().into_iter().filter_map(Location::read_by)
			{
				if let Some(&value) = current.get(&location)
					&& !read.contains(&value)
				{
					read.push(value);
					values[value].readers.push(pc);
				}
			}
			let rolls = matches!(
				inst,
				Instruction::RollRange(_)
					| Instruction::RollStandardDice(_)
					| Instruction::RollCustomDice(_)
			);
			let random = rolls || read.iter().any(|&v| values[v].random);
			let location = Location::written_by(inst);
			let value = values.len();
			values.push(Value {
				location,
				random,
				readers: Vec::new(),
				lasting: false,
				depends_on: BTreeSet::new()
			});
			current.insert(location, value);
			reads.push(read);
			writes.push(value);
		}
		if let Some(&answer) = current.get(&Location::Answer)
		{
			values[answer].lasting = true;
		}
		// Whether a value fans out is known only once all of its readers are,
		// so its dependencies follow in a second sweep, in the order written.
		for (pc, &value) in writes.iter().enumerate()
		{
			let mut depends_on = reads[pc]
				.iter()
				.flat_map(|&v| values[v].depends_on.iter().copied())
				.collect::<BTreeSet<_>>();
			if values[value].fans_out()
			{
				depends_on.insert(value);
			}
			values[value].depends_on = depends_on;
		}
		Self {
			values,
			reads,
			writes
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Scheduling.                                 //
////////////////////////////////////////////////////////////////////////////////

impl Values
{
	/// Schedule the splits and merges of the values: split each value that
	/// fans out right after its instruction writes it, and merge each split as
	/// soon as it [may merge](Self::survivor), topmost first.
	///
	/// # Parameters
	/// - `len`: The number of instructions.
	///
	/// # Returns
	/// The steps that follow each instruction.
	fn schedule(&self, len: usize) -> Vec<Vec<Step>>
	{
		let mut plan = Vec::with_capacity(len);
		let mut live = BTreeSet::<usize>::new();
		let mut current = BTreeSet::<usize>::new();
		let mut splits = Vec::<usize>::new();
		for pc in 0..len
		{
			let pc = ProgramCounter(pc);
			let mut steps = Vec::new();
			for &value in &self.reads[pc.0]
			{
				if !self.values[value].is_live_after(pc)
				{
					live.remove(&value);
				}
			}
			let value = self.writes[pc.0];
			let location = self.values[value].location;
			current.retain(|&v| self.values[v].location != location);
			current.insert(value);
			if self.values[value].is_live_after(pc)
			{
				live.insert(value);
			}
			if self.values[value].fans_out()
			{
				splits.push(value);
				steps.push(Step::Split(self.values[value].location));
			}
			'merge: loop
			{
				for split in (0..splits.len()).rev()
				{
					if let Some(survivor) = self.survivor(&splits, split, &live)
					{
						let dead = self.dead(splits[split], survivor, &current);
						splits.remove(split);
						steps.push(Step::Merge {
							split,
							survivor,
							dead
						});
						continue 'merge
					}
				}
				break
			}
			plan.push(steps);
		}
		debug_assert!(splits.is_empty(), "every split merges by the end");
		plan
	}

	/// Decide whether the specified split may merge, and if so, answer the
	/// location of the value that survives it. A split may merge once at most
	/// one live value depends on it, other than itself, and that value is not
	/// a rolling record, whose influence is not yet one value; and once every
	/// split above it is independent of it.
	///
	/// # Parameters
	/// - `splits`: The stack of splits, by value, from the bottom.
	/// - `split`: The position of the split in the stack.
	/// - `live`: The live values.
	///
	/// # Returns
	/// `None` if the split may not merge yet, `Some(None)` if it may merge and
	/// no live value depends on it, or `Some(Some(location))` if it may merge
	/// into the value at the location.
	fn survivor(
		&self,
		splits: &[usize],
		split: usize,
		live: &BTreeSet<usize>
	) -> Option<Option<Location>>
	{
		let value = splits[split];
		if splits[split + 1..]
			.iter()
			.any(|&above| self.values[above].depends_on.contains(&value))
		{
			return None
		}
		let mut dependents = live
			.iter()
			.filter(|&&v| self.values[v].depends_on.contains(&value));
		match (dependents.next(), dependents.next())
		{
			(None, _) => Some(None),
			(Some(&survivor), None) if survivor != value =>
			{
				match self.values[survivor].location
				{
					Location::RollingRecord(_) => None,
					location => Some(Some(location))
				}
			},
			_ => None
		}
	}

	/// Answer the locations of the values, other than the survivor, that
	/// depend on the specified split and still occupy their locations.
	///
	/// # Parameters
	/// - `split`: The split, by value.
	/// - `survivor`: The location of the survivor, if any.
	/// - `current`: The values that occupy their locations, by index.
	///
	/// # Returns
	/// The locations, in the order that their values were written.
	fn dead(
		&self,
		split: usize,
		survivor: Option<Location>,
		current: &BTreeSet<usize>
	) -> Vec<Location>
	{
		current
			.iter()
			.map(|&v| &self.values[v])
			.filter(|v| {
				v.depends_on.contains(&split) && Some(v.location) != survivor
			})
			.map(|v| v.location)
			.collect()
	}
}
