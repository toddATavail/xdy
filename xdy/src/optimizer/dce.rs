//! # Dead code elimination
//!
//! Dead code elimination is the process of removing code that does not
//! contribute to the final output of a program. Dead code can only the result
//! of previous optimizations. The dead code eliminator requires the function to
//! be in static single assignment (SSA) form.

use std::collections::HashMap;

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _,
	DependencyAnalyzer, Div, DropHighest, DropLowest, Exp, Function,
	Instruction, InstructionVisitor, Mod, Mul, Neg, Optimizer, RegisterIndex,
	Return, RollCustomDice, RollRange, RollStandardDice, RollingRecordIndex,
	Sub, SumRollingRecord
};

////////////////////////////////////////////////////////////////////////////////
//                           Dead code elimination.                           //
////////////////////////////////////////////////////////////////////////////////

/// A dead code eliminator removes instructions that do not contribute to the
/// final output of a function.
#[derive(Debug, Clone, Default)]
pub struct DeadCodeEliminator
{
	/// Replacements for registers, due to renumbering, as a map from originals
	/// to replacements. The keys are relative to the original function, and
	/// the values are relative to the optimized function.
	replacements: HashMap<AddressingMode, AddressingMode>,

	/// The next register index to allocate.
	next_register: RegisterIndex,

	/// The next rolling record index to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The replacement instructions.
	instructions: Vec<Instruction>
}

impl Optimizer<()> for DeadCodeEliminator
{
	fn optimize(mut self, mut function: Function) -> Result<Function, ()>
	{
		// Skip over the parameters and externals when initializing the register
		// allocator. Don't bother to install them in the replacement map, as
		// they cannot be altered.
		let start_register =
			function.parameters.len() + function.externals.len();
		loop
		{
			// Reset the optimizer state.
			self.replacements.clear();
			self.next_register = RegisterIndex(start_register);
			self.next_rolling_record = RollingRecordIndex::default();
			self.instructions.clear();
			// Visit each instruction in the function body.
			let analyzer = DependencyAnalyzer::analyze(&function.instructions);
			for instruction in &function.instructions
			{
				instruction
					.visit(&mut AnalyticalVisitor(&analyzer, &mut self))
					.unwrap();
			}
			// Compare the old and new instructions. If they are the same, we
			// have reached a fixed point and can return the optimized function.
			if function.instructions == self.instructions
			{
				return Ok(function)
			}
			// Update the function to reflect the new instructions.
			function.register_count = self.next_register.0;
			function.instructions = self.instructions.clone();
			// Reset the optimizer state.
			self.replacements.clear();
			self.next_register = RegisterIndex(start_register);
			self.next_rolling_record = RollingRecordIndex::default();
			self.instructions.clear();
		}
	}
}

/// A visitor that performs the analysis necessary for dead code elimination.
struct AnalyticalVisitor<'inst>(
	&'inst DependencyAnalyzer<'inst>,
	&'inst mut DeadCodeEliminator
);

impl InstructionVisitor<()> for AnalyticalVisitor<'_>
{
	fn visit_roll_range(&mut self, range: &RollRange) -> Result<(), ()>
	{
		if self.has_readers(*range)
		{
			let dest = self.next_rolling_record();
			self.replace(range.dest, dest);
			self.emit(RollRange {
				dest,
				start: self.replacement(range.start),
				end: self.replacement(range.end)
			});
		}
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		roll: &RollStandardDice
	) -> Result<(), ()>
	{
		if self.has_readers(*roll)
		{
			let dest = self.next_rolling_record();
			self.replace(roll.dest, dest);
			self.emit(RollStandardDice {
				dest,
				count: self.replacement(roll.count),
				faces: self.replacement(roll.faces)
			});
		}
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		roll: &RollCustomDice
	) -> Result<(), ()>
	{
		if self.has_readers(roll.clone())
		{
			let dest = self.next_rolling_record();
			self.replace(roll.dest, dest);
			self.emit(RollCustomDice {
				dest,
				count: self.replacement(roll.count),
				faces: roll.faces.clone()
			});
		}
		Ok(())
	}

	fn visit_drop_lowest(&mut self, drop: &DropLowest) -> Result<(), ()>
	{
		if self.has_readers(*drop)
		{
			self.emit(DropLowest {
				dest: self.replacement(drop.dest).try_into().unwrap(),
				count: self.replacement(drop.count)
			});
		}
		Ok(())
	}

	fn visit_drop_highest(&mut self, drop: &DropHighest) -> Result<(), ()>
	{
		if self.has_readers(*drop)
		{
			self.emit(DropHighest {
				dest: self.replacement(drop.dest).try_into().unwrap(),
				count: self.replacement(drop.count)
			});
		}
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		sum: &SumRollingRecord
	) -> Result<(), ()>
	{
		if self.has_readers(*sum)
		{
			let dest = self.next_register();
			self.replace(sum.dest, dest);
			self.emit(SumRollingRecord {
				dest,
				src: self.replacement(sum.src).try_into().unwrap()
			});
		}
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Add {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Sub {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Mul {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Div {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Mod {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Exp {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op2)
			});
		}
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		if self.has_readers(*inst)
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg {
				dest,
				op: self.replacement(inst.op)
			});
		}
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		self.emit(Return {
			src: self.replacement(inst.src)
		});
		Ok(())
	}
}

impl AnalyticalVisitor<'_>
{
	/// Answer the next available register index.
	///
	/// # Returns
	/// The next available register index.
	fn next_register(&mut self) -> RegisterIndex
	{
		self.1.next_register.allocate()
	}

	/// Answer the next available rolling record index.
	///
	/// # Returns
	/// The next available rolling record index.
	fn next_rolling_record(&mut self) -> RollingRecordIndex
	{
		self.1.next_rolling_record.allocate()
	}

	/// Set the replacement for the specified operand.
	///
	/// # Parameters
	/// - `old`: The operand to replace.
	/// - `new`: The replacement operand.
	fn replace(
		&mut self,
		old: impl Into<AddressingMode>,
		new: impl Into<AddressingMode>
	)
	{
		self.1.replacements.insert(old.into(), new.into());
	}

	/// Determine whether the specified instruction has downstream readers. If
	/// it does not, then the instruction is dead. Should only be used with
	/// instructions that have a destination.
	///
	/// # Parameters
	/// - `inst`: The instruction to check.
	///
	/// # Returns
	/// `true` if the instruction has downstream readers, `false` otherwise.
	fn has_readers(&self, inst: impl Into<Instruction>) -> bool
	{
		self.0
			.readers()
			.get(&inst.into().destination().unwrap())
			.map(|readers| !readers.is_empty())
			.unwrap_or(false)
	}

	/// Look up the replacement of the specified operand, falling back to the
	/// operand itself if no replacement is found.
	///
	/// # Parameters
	/// - `op`: The operand to look up.
	///
	/// # Returns
	/// The replacement for the operand.
	#[inline]
	fn replacement(&self, op: impl Into<AddressingMode>) -> AddressingMode
	{
		let op: AddressingMode = op.into();
		*self.1.replacements.get(&op).unwrap_or(&op)
	}

	/// Emit the specified instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to emit.
	#[inline]
	fn emit(&mut self, inst: impl Into<Instruction>)
	{
		self.1.instructions.push(inst.into());
	}
}
