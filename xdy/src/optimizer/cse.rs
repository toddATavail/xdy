//! # Common subexpression elimination
//!
//! Common subexpression elimination (CSE) is an optimization that eliminates
//! redundant computations, based on the observation that an expression that
//! has been previously computed previously can simply be reused. CSE requires
//! the function to be in static single assignment (SSA) form.

use std::collections::{hash_map::Entry, HashMap};

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, Exp, Function, Instruction, InstructionVisitor,
	Mod, Mul, Neg, RegisterIndex, Return, RollCustomDice, RollRange,
	RollStandardDice, Sub, SumRollingRecord
};

use super::Optimizer;

////////////////////////////////////////////////////////////////////////////////
//                     Common subexpression elimination.                      //
////////////////////////////////////////////////////////////////////////////////

/// A common subexpression eliminator. The standard optimizer applies this
/// pass to a [function](Function) to eliminate common subexpressions. The
/// eliminator works by replacing eligible duplicate instructions with the
/// destination register of the original instruction. Note that dice
/// instructions cannot be eliminated, as they are not pure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommonSubexpressionEliminator
{
	/// The scrubbed original instructions, as a map from scrubbed instructions
	/// to their destination registers. Instruction scrubbing involves
	/// replacing its destination register with a sentinel value, so that the
	/// instruction can be compared for equality with another scrubbed
	/// instruction. When a common subexpression is found, the redundant
	/// instruction is replaced with the destination register of the original
	/// instruction.
	originals: HashMap<Instruction, RegisterIndex>,

	/// Replacements for registers, due to renumbering or instruction
	/// elimination, as a map from originals to replacements. The keys are
	/// relative to the original function, and the values are relative to the
	/// optimized function.
	replacements: HashMap<AddressingMode, AddressingMode>,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The replacement instructions.
	instructions: Vec<Instruction>
}

impl Optimizer<()> for CommonSubexpressionEliminator
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
			self.originals.clear();
			self.replacements.clear();
			self.next_register = RegisterIndex(start_register);
			self.instructions.clear();
			// Visit each instruction in the function body.
			for instruction in &function.instructions
			{
				instruction.visit(&mut self).unwrap();
			}
			// Compare the old and new instructions. If they are the same, we
			// have reached a fixed point and can return the optimized function.
			if function.instructions == self.instructions
			{
				return Ok(function);
			}
			// Update the function to reflect the new instructions.
			function.register_count = self.next_register.0;
			function.instructions = self.instructions.clone();
		}
	}
}

impl InstructionVisitor<()> for CommonSubexpressionEliminator
{
	fn visit_roll_range(&mut self, range: &crate::RollRange) -> Result<(), ()>
	{
		// Range instructions are not pure, so they cannot be eliminated. The
		// rolling records are not affected by this optimization, so we don't
		// need to allocate replacements for them.
		self.emit(RollRange {
			dest: range.dest,
			start: self.replacement(range.start),
			end: self.replacement(range.end)
		});
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		roll: &RollStandardDice
	) -> Result<(), ()>
	{
		// Dice instructions are not pure, so they cannot be eliminated. The
		// rolling records are not affected by this optimization, so we don't
		// need to allocate replacements for them.
		self.emit(RollStandardDice {
			dest: roll.dest,
			count: self.replacement(roll.count),
			faces: self.replacement(roll.faces)
		});
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		roll: &RollCustomDice
	) -> Result<(), ()>
	{
		// Dice instructions are not pure, so they cannot be eliminated. The
		// rolling records are not affected by this optimization, so we don't
		// need to allocate replacements for them.
		self.emit(RollCustomDice {
			dest: roll.dest,
			count: self.replacement(roll.count),
			faces: roll.faces.clone()
		});
		Ok(())
	}

	fn visit_drop_lowest(&mut self, drop: &DropLowest) -> Result<(), ()>
	{
		// Dice instructions are not pure, so they cannot be eliminated. The
		// rolling records are not affected by this optimization, so we don't
		// need to allocate replacements for them.
		self.emit(DropLowest {
			dest: drop.dest,
			count: self.replacement(drop.count)
		});
		Ok(())
	}

	fn visit_drop_highest(&mut self, drop: &DropHighest) -> Result<(), ()>
	{
		// Dice instructions are not pure, so they cannot be eliminated. The
		// rolling records are not affected by this optimization, so we don't
		// need to allocate replacements for them.
		self.emit(DropHighest {
			dest: drop.dest,
			count: self.replacement(drop.count)
		});
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		sum: &SumRollingRecord
	) -> Result<(), ()>
	{
		// The compiler currently cannot produce duplicate SumRollingRecord
		// instructions, so this is future-proofing only.
		if let (dest, true) = self.canonicalize(*sum)
		{
			// This is the original instruction, so we need to emit it after
			// replacing the destination register.
			self.emit(SumRollingRecord { dest, src: sum.src });
		}
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Add { dest, op1, op2 }
		)
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Sub { dest, op1, op2 }
		)
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Mul { dest, op1, op2 }
		)
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Div { dest, op1, op2 }
		)
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Mod { dest, op1, op2 }
		)
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		self.canonicalize_binary_op(
			*inst,
			|| (inst.op1, inst.op2),
			|dest, op1, op2| Exp { dest, op1, op2 }
		)
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		if let (dest, true) = self.canonicalize(*inst)
		{
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

impl CommonSubexpressionEliminator
{
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
		*self.replacements.get(&op).unwrap_or(&op)
	}

	/// Canonicalize the specified instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to canonicalize.
	///
	/// # Returns
	/// A 2-tuple comprising (1) the canonical destination register for the
	/// instruction and (2) whether the instruction holds the first occurrence
	/// of the canonical destination register.
	fn canonicalize(
		&mut self,
		inst: impl Into<Instruction>
	) -> (RegisterIndex, bool)
	{
		let inst = inst.into();
		match self.originals.entry(inst.scrub())
		{
			Entry::Occupied(entry) =>
			{
				let dest = *entry.get();
				self.replacements
					.insert(inst.destination().unwrap(), dest.into());
				(dest, false)
			},
			Entry::Vacant(entry) =>
			{
				let dest = self.next_register.allocate();
				self.replacements
					.insert(inst.destination().unwrap(), dest.into());
				(*entry.insert(dest), true)
			}
		}
	}

	/// Canonicalize the specified binary operation, emitting a new
	/// instruction if this is the first occurrence of the canonical
	/// destination register.
	///
	/// # Type Parameters
	/// - `I`: The type of the new instruction to emit.
	///
	/// # Parameters
	/// - `inst`: The instruction to canonicalize.
	/// - `extractor`: A closure that extracts the two source operands from the
	///   instructrion.
	/// - `constructor`: A closure that constructs a new instruction with the
	///   given canonical destination register and source operands.
	fn canonicalize_binary_op<I>(
		&mut self,
		inst: impl Into<Instruction>,
		extractor: impl Fn() -> (AddressingMode, AddressingMode),
		constructor: impl Fn(RegisterIndex, AddressingMode, AddressingMode) -> I
	) -> Result<(), ()>
	where
		I: Into<Instruction>
	{
		if let (dest, true) = self.canonicalize(inst)
		{
			// This is the original instruction, so we need to emit it after
			// replacing the destination register.
			let (op1, op2) = extractor();
			self.instructions.push(
				constructor(
					dest,
					self.replacements.get(&op1).copied().unwrap_or(op1),
					self.replacements.get(&op2).copied().unwrap_or(op2)
				)
				.into()
			);
		}
		Ok(())
	}

	/// Emit the specified instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to emit.
	#[inline]
	fn emit(&mut self, inst: impl Into<Instruction>)
	{
		self.instructions.push(inst.into());
	}
}

/// Provides the capability to scrub the destination register of an
/// [instruction](Instruction).
trait CanScrub
{
	/// Scrub the destination register of the instruction, so that it can be
	/// compared for equality with another instruction for the purpose of
	/// identifying common subexpressions.
	///
	/// # Returns
	/// A scrubbed variant of the receiver.
	fn scrub(&self) -> Self;
}

impl CanScrub for RollRange
{
	fn scrub(&self) -> Self
	{
		// The destination is a rolling record, so it unaffected.
		RollRange {
			dest: self.dest,
			start: self.start,
			end: self.end
		}
	}
}

impl CanScrub for RollStandardDice
{
	fn scrub(&self) -> Self
	{
		// The destination is a rolling record, so it unaffected.
		RollStandardDice {
			dest: self.dest,
			count: self.count,
			faces: self.faces
		}
	}
}

impl CanScrub for RollCustomDice
{
	fn scrub(&self) -> Self
	{
		// The destination is a rolling record, so it unaffected.
		RollCustomDice {
			dest: self.dest,
			count: self.count,
			faces: self.faces.clone()
		}
	}
}

impl CanScrub for DropLowest
{
	fn scrub(&self) -> Self
	{
		// The destination is a rolling record, so it unaffected.
		DropLowest {
			dest: self.dest,
			count: self.count
		}
	}
}

impl CanScrub for DropHighest
{
	fn scrub(&self) -> Self
	{
		// The destination is a rolling record, so it unaffected.
		DropHighest {
			dest: self.dest,
			count: self.count
		}
	}
}

impl CanScrub for SumRollingRecord
{
	fn scrub(&self) -> Self
	{
		SumRollingRecord {
			dest: SCRUBBED_DEST,
			src: self.src
		}
	}
}

impl CanScrub for Add
{
	fn scrub(&self) -> Self
	{
		Add {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Sub
{
	fn scrub(&self) -> Self
	{
		Sub {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Mul
{
	fn scrub(&self) -> Self
	{
		Mul {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Div
{
	fn scrub(&self) -> Self
	{
		Div {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Mod
{
	fn scrub(&self) -> Self
	{
		Mod {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Exp
{
	fn scrub(&self) -> Self
	{
		Exp {
			dest: SCRUBBED_DEST,
			op1: self.op1,
			op2: self.op2
		}
	}
}

impl CanScrub for Neg
{
	fn scrub(&self) -> Self
	{
		Neg {
			dest: SCRUBBED_DEST,
			op: self.op
		}
	}
}

impl CanScrub for Return
{
	fn scrub(&self) -> Self { Return { src: self.src } }
}

impl CanScrub for Instruction
{
	fn scrub(&self) -> Self
	{
		match self
		{
			Instruction::RollRange(inst) => inst.scrub().into(),
			Instruction::RollStandardDice(inst) => inst.scrub().into(),
			Instruction::RollCustomDice(inst) => inst.scrub().into(),
			Instruction::DropLowest(inst) => inst.scrub().into(),
			Instruction::DropHighest(inst) => inst.scrub().into(),
			Instruction::SumRollingRecord(inst) => inst.scrub().into(),
			Instruction::Add(inst) => inst.scrub().into(),
			Instruction::Sub(inst) => inst.scrub().into(),
			Instruction::Mul(inst) => inst.scrub().into(),
			Instruction::Div(inst) => inst.scrub().into(),
			Instruction::Mod(inst) => inst.scrub().into(),
			Instruction::Exp(inst) => inst.scrub().into(),
			Instruction::Neg(inst) => inst.scrub().into(),
			Instruction::Return(inst) => inst.scrub().into()
		}
	}
}

/// The sentinel value used to indicate that a register index has been scrubbed.
/// The value is in-band, and so very technically a valid register index, but
/// reality dictates that no real function will have over two billion registers.
const SCRUBBED_DEST: RegisterIndex = RegisterIndex(usize::MAX);
