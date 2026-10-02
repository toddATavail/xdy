//! # Register coalescence
//!
//! The final pass in the optimizer is register coalescence. This pass breaks
//! static single assignment (SSA) form by reassigning dead registers to reduce
//! the number of registers used. The register coalescer colors the interval
//! graph of the registers' live ranges by a linear scan: it visits the ranges
//! in order of their starts, and gives each register the least color that no
//! live range still holds, so that registers whose ranges do not overlap may
//! share a color. The instructions never branch or loop, so each live range is
//! a single interval, and the scan is exact and optimal, and takes time
//! `O(n log n)` in the number of registers.

use std::{
	cmp::Reverse,
	collections::{BTreeMap, BTreeSet, BinaryHeap},
	ops::RangeInclusive
};

use crate::{
	Add, AddressingMode, CanVisitInstructions as _, DependencyAnalyzer, Div,
	DropHighest, DropLowest, Exp, Function, Instruction, InstructionVisitor,
	Max, Mod, Mul, Neg, Optimizer, ProgramCounter, RegisterIndex, Return,
	RollCustomDice, RollRange, RollStandardDice, Sub, SumRollingRecord
};

////////////////////////////////////////////////////////////////////////////////
//                           Register coalescence.                            //
////////////////////////////////////////////////////////////////////////////////

/// A liveness map, as a map from registers to the range of program counters
/// wherein they are live.
type LivenessMap = BTreeMap<RegisterIndex, RangeInclusive<ProgramCounter>>;

/// A coloring, as a map from original registers to their colors, i.e., the
/// registers that replace them. The map only contains entries for ordinary
/// registers, not immediates or rolling records.
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
		self.coloring = Self::colorize(&liveness);
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
			// instruction.
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
			// A register that nothing reads is live only as its writer
			// completes, so that its writer clobbers no register that is live
			// then.
			let end = match analyzer.readers().get(&r.into())
			{
				Some(readers) => *readers.last().unwrap(),
				None => start
			};
			let range = start..=end;
			liveness.insert(r, range);
		}
		liveness
	}

	/// Color the registers of the function by a linear scan of their live
	/// ranges, so that no two registers whose ranges overlap share a color.
	/// The scan visits the ranges in order of their starts, and gives each
	/// register the least color that no range still live holds. The formal
	/// parameters and external variables, which are live from the first
	/// instruction and come first, keep their own registers.
	///
	/// # Parameters
	/// - `liveness`: The liveness map.
	///
	/// # Returns
	/// The coloring map, which maps each register to its replacement in a
	/// potentially smaller palette.
	fn colorize(liveness: &LivenessMap) -> Coloring
	{
		let mut order = liveness
			.iter()
			.map(|(r, range)| (*range.start(), *r))
			.collect::<Vec<_>>();
		order.sort();
		let mut colors = Coloring::new();
		// The colors that no live range holds, and the ranges still live, by
		// their ends, with their colors.
		let mut free = BTreeSet::<RegisterIndex>::new();
		let mut live =
			BinaryHeap::<Reverse<(ProgramCounter, RegisterIndex)>>::new();
		let mut palette = 0;
		for (start, r) in order
		{
			// Free the colors of the ranges that end before this one starts.
			while let Some(&Reverse((end, color))) = live.peek()
				&& end < start
			{
				live.pop();
				free.insert(color);
			}
			let color = free.pop_first().unwrap_or_else(|| {
				palette += 1;
				RegisterIndex(palette - 1)
			});
			colors.insert(r, color);
			live.push(Reverse((*liveness[&r].end(), color)));
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

	fn visit_max(&mut self, inst: &Max) -> Result<(), ()>
	{
		self.emit(Max {
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
