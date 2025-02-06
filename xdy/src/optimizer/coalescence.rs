//! # Register coalescence
//!
//! The final pass in the optimizer is register coalescence. This pass breaks
//! static single assignment (SSA) form by reassigning dead registers to reduce
//! the number of registers used. The register coalescer uses a graph coloring
//! algorithm to determine which registers can be merged.

use std::{
	collections::{BTreeMap, HashSet},
	ops::RangeInclusive
};

use crate::{
	Add, AddressingMode, CanVisitInstructions as _, DependencyAnalyzer, Div,
	DropHighest, DropLowest, Exp, Function, Instruction, InstructionVisitor,
	Mod, Mul, Neg, Optimizer, ProgramCounter, RegisterIndex, Return,
	RollCustomDice, RollRange, RollStandardDice, Sub, SumRollingRecord
};

////////////////////////////////////////////////////////////////////////////////
//                           Register coalescence.                            //
////////////////////////////////////////////////////////////////////////////////

/// A liveness map, as a map from registers to the range of program counters
/// wherein they are live.
type LivenessMap = BTreeMap<RegisterIndex, RangeInclusive<ProgramCounter>>;

/// An interference graph, as a map from registers to the set of registers that
/// are contemporaneous with them. These registers interfere with each other and
/// thus cannot be coalesced.
type InterferenceGraph = BTreeMap<RegisterIndex, HashSet<RegisterIndex>>;

/// A coloring, as a map from addressing modes to register indices. The map
/// only contains entries for ordinary registers, not immediates or rolling
/// records.
type Coloring = BTreeMap<RegisterIndex, RegisterIndex>;

/// A register coalescer merges registers to reduce the number of registers used
/// in a function.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterCoalescer
{
	/// The original function.
	function: Option<Function>,

	/// The mapping from registers to their colors.
	coloring: Coloring,

	/// The replacement instructions.
	instructions: Vec<Instruction>
}

impl Optimizer<()> for RegisterCoalescer
{
	fn optimize(mut self, mut function: Function) -> Result<Function, ()>
	{
		self.function = Some(function.clone());
		let analyzer =
			DependencyAnalyzer::analyze(&self.function().instructions);
		let liveness = self.compute_liveness(analyzer);
		let interference = self.compute_interference(liveness);
		self.coloring = self.colorize(interference);
		// Visit each instruction to apply the coloring.
		for instruction in &function.instructions
		{
			instruction.visit(&mut self).unwrap();
		}
		// Update the function to reflect the new instructions.
		function.register_count = self
			.coloring
			.values()
			.max()
			.map(|count| count.0 + 1)
			.unwrap_or(0);
		function.instructions = self.instructions;
		Ok(function)
	}
}

impl RegisterCoalescer
{
	/// Answer the function being optimized.
	///
	/// # Returns
	/// The requested function.
	fn function(&self) -> &Function { self.function.as_ref().unwrap() }

	/// Compute the liveness of each register in the function.
	///
	/// # Parameters
	/// - `analyzer`: A dependency analyzer.
	///
	/// # Returns
	/// The liveness map.
	fn compute_liveness(&self, analyzer: DependencyAnalyzer<'_>)
	-> LivenessMap
	{
		let mut liveness = LivenessMap::new();
		for r in 0..self.function().register_count
		{
			// Registers without writers represent parameters or external
			// variables and should be considered live starting at the first
			// instruction. If a register is unread, then it must be a parameter
			// or external variable; in this case, the liveness range is empty.
			let r: RegisterIndex = r.into();
			let start = match analyzer.writers().get(&r.into())
			{
				Some(writers) =>
				{
					// Exclude the first writer, as the register is live only
					// after the writer completes.
					(writers.first().unwrap().0 + 1).into()
				},
				None => 0.into()
			};
			let end = match analyzer.readers().get(&r.into())
			{
				Some(readers) => *readers.last().unwrap(),
				None => 0.into()
			};
			let range = start..=end;
			liveness.insert(r, range);
		}
		liveness
	}

	/// Compute the interference graph for the function. This graph represents
	/// the registers whose lifetimes overlap and thus cannot be coalesced.
	///
	/// # Parameters
	/// - `liveness`: The liveness map.
	///
	/// # Returns
	/// The interference graph.
	fn compute_interference(&self, liveness: LivenessMap) -> InterferenceGraph
	{
		let mut graph = InterferenceGraph::new();
		for r0 in 0..self.function().register_count
		{
			let r0 = r0.into();
			let interference = graph.entry(r0).or_default();
			for r1 in 0..self.function().register_count
			{
				let r1 = r1.into();
				if r0 != r1 && liveness[&r0].overlaps(&liveness[&r1])
				{
					interference.insert(r1);
				}
			}
		}
		graph
	}

	/// Color the function using the interference graph. The algorithm assigns a
	/// color to each register such that no two interfering registers share the
	/// same color.
	///
	/// # Parameters
	/// - `graph`: The interference graph.
	///
	/// # Returns
	/// The coloring map, which maps each register to its replacement in a
	/// potentially smaller palette.
	fn colorize(&self, graph: InterferenceGraph) -> Coloring
	{
		let mut colors = Coloring::new();
		// Bound the register palette to the original register set size.
		let available_colors = (0..self.function().register_count)
			.map(RegisterIndex::from)
			.collect::<HashSet<_>>();
		for (r, interference) in graph
		{
			// Remove the colors of interfering registers from the available
			// color choices.
			let mut unused_colors = available_colors.clone();
			for i in interference
			{
				if let Some(color) = colors.get(&i)
				{
					unused_colors.remove(color);
				}
			}
			// Assign the first available color to the register.
			colors.insert(r, *unused_colors.iter().min().unwrap());
		}
		colors
	}

	/// Use the coloring to answer the appropriate replacement for the
	/// specified addressing mode.
	///
	/// # Parameters
	/// - `op`: The addressing mode to color.
	///
	/// # Returns
	/// The colored addressing mode.
	#[inline]
	fn color(&self, op: impl Into<AddressingMode>) -> AddressingMode
	{
		match op.into()
		{
			AddressingMode::Register(r) => self.coloring[&r].into(),
			op => op
		}
	}

	/// Emit the specified instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to emit.
	#[inline]
	fn emit(&mut self, inst: impl Into<Instruction>) -> Result<(), ()>
	{
		self.instructions.push(inst.into());
		Ok(())
	}
}

impl InstructionVisitor<()> for RegisterCoalescer
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		self.emit(RollRange {
			dest: inst.dest,
			start: self.color(inst.start),
			end: self.color(inst.end)
		})
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		self.emit(RollStandardDice {
			dest: inst.dest,
			count: self.color(inst.count),
			faces: self.color(inst.faces)
		})
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		self.emit(RollCustomDice {
			dest: inst.dest,
			count: self.color(inst.count),
			faces: inst.faces.clone()
		})
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		self.emit(DropLowest {
			dest: inst.dest,
			count: self.color(inst.count)
		})
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		self.emit(DropHighest {
			dest: inst.dest,
			count: self.color(inst.count)
		})
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		self.emit(SumRollingRecord {
			dest: self.color(inst.dest).try_into().unwrap(),
			src: inst.src
		})
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		self.emit(Add {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		self.emit(Sub {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.emit(Mul {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		self.emit(Div {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		self.emit(Mod {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		self.emit(Exp {
			dest: self.color(inst.dest).try_into().unwrap(),
			op1: self.color(inst.op1),
			op2: self.color(inst.op2)
		})
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		self.emit(Neg {
			dest: self.color(inst.dest).try_into().unwrap(),
			op: self.color(inst.op)
		})
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		self.emit(Return {
			src: self.color(inst.src)
		})
	}
}

/// Capability to detect overlap between two instances.
pub trait CanOverlap
{
	/// Determine if the receiver overlaps with the argument.
	///
	/// # Parameters
	/// * `other` - The other instance to compare against.
	///
	/// # Returns
	fn overlaps(&self, other: &Self) -> bool;
}

impl<T> CanOverlap for RangeInclusive<T>
where
	T: PartialOrd
{
	#[inline]
	fn overlaps(&self, other: &Self) -> bool
	{
		self.start() <= other.end() && other.start() <= self.end()
	}
}
