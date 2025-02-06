//! # Constant folding
//!
//! The constant folder is a simple optimizer that folds expressions with only
//! constant operands into immediates. The constant folder requires the function
//! to be in static single assignment (SSA) form.

use std::collections::HashMap;

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, Exp, Function, Immediate, InstructionVisitor, Mod,
	Mul, Neg, ProgramCounter, RegisterIndex, Return, RollCustomDice, RollRange,
	RollStandardDice, RollingRecordIndex, Sub, SumRollingRecord,
	ir::Instruction
};

use crate::{Optimizer, add, div, exp, r#mod, mul, neg, sub};

////////////////////////////////////////////////////////////////////////////////
//                             Constant folding.                              //
////////////////////////////////////////////////////////////////////////////////

/// A constant folder. The standard optimizer uses a constant folder to fold
/// expressions with only constant operands into [immediates]Immediate).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstantFolder
{
	/// The previous version of the function body.
	previous: Option<Vec<Instruction>>,

	/// Replacements for registers, due to renumbering or folding, as a map
	/// from originals to replacements. The keys are relative to the original
	/// function, and the values are relative to the optimized function.
	replacements: HashMap<AddressingMode, AddressingMode>,

	/// The program counter. Always set to the correct value before visiting an
	/// instruction.
	pc: ProgramCounter,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The next rolling record to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The replacement instructions.
	instructions: Vec<Instruction>
}

impl Optimizer<()> for ConstantFolder
{
	fn optimize(mut self, mut function: Function) -> Result<Function, ()>
	{
		// Skip over the parameters and externals when initializing the register
		// allocator. Don't bother to install them in the replacement map, as
		// they cannot be altered.
		let start_register =
			function.parameters.len() + function.externals.len();
		self.next_register = RegisterIndex(start_register);
		loop
		{
			// Preserve the previous version of the function body, to support
			// forward scanning of instructions.
			self.previous = Some(function.instructions.clone());
			// Visit each instruction in the function body.
			for instruction in &function.instructions
			{
				instruction.visit(&mut self).unwrap();
				self.pc.allocate();
			}
			// Compare the old and new instructions. If they are the same, we
			// have reached a fixed point and can return the optimized function.
			if function.instructions == self.instructions
			{
				return Ok(function);
			}
			// Update the function to reflect the new instructions.
			function.register_count = self.next_register.0;
			function.rolling_record_count = self.next_rolling_record.0;
			function.instructions = self.instructions.clone();
			// Reset the optimizer state.
			self.replacements.clear();
			self.pc = ProgramCounter::default();
			self.next_register = RegisterIndex(start_register);
			self.next_rolling_record = RollingRecordIndex::default();
			self.instructions.clear();
		}
	}
}

impl InstructionVisitor<()> for ConstantFolder
{
	fn visit_roll_range(&mut self, range: &RollRange) -> Result<(), ()>
	{
		if let Some((start, end)) = range.const_ops()
		{
			// Ranges can't have drop expressions, so we can fold the range if
			// it contains one or fewer possible outcomes. Folding involves
			// eliminating the instruction completely and replacing its rolling
			// record with the new constant value. When we subsequently
			// encounter the corresponding SumRollingRecord instruction, we
			// will eliminate it also and replace its destination register with
			// the new constant value.
			match end.saturating_sub(start).saturating_add(1)
			{
				..=0 =>
				{
					// The range is empty, so the result is zero.
					self.replace(range.dest, Immediate(0));
					return Ok(())
				},
				1 =>
				{
					// The range contains a single value, which must be the
					// result.
					self.replace(range.dest, Immediate(start));
					return Ok(())
				},
				_ =>
				{}
			}
		}
		// The range can't be folded, but we may still need to allocate a new
		// rolling record due to renumbering. The operands are guaranteed to be
		// in the map already unless they are constants, as they must have been
		// the destinations of previous instructions.
		let dest = self.next_rolling_record();
		self.replace(range.dest, dest);
		self.emit(RollRange {
			dest,
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
		if let AddressingMode::Immediate(Immediate(..=0)) = roll.count
		{
			// There are no dice, so it doesn't matter what the faces are.
			self.replace(roll.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(..=0)) = roll.faces
		{
			// There are no faces, so it doesn't matter what the count is.
			self.replace(roll.dest, Immediate(0));
			return Ok(())
		}
		if let Some((count, faces)) = roll.const_ops()
		{
			// Scan forward for any drop expressions that target the same
			// rolling record. If we are guaranteed to drop all the dice,
			// then it doesn't matter how many faces the dice have.
			let drops = self.find_drop_instructions(roll.dest);
			let can_fold =
				drops.map(|drop| drop.const_ops()).collect::<Vec<_>>();
			if can_fold.iter().all(Option::is_some)
			{
				// Identify the number of dice to drop, then reduce the
				// count accordingly.
				let dropped = can_fold
					.iter()
					.map(|can_fold| can_fold.unwrap().0)
					.sum::<i32>();
				let count = (count.saturating_sub(dropped)).clamp(0, i32::MAX);
				if count <= 0
				{
					// All dice are dropped, so the contribution is zero.
					self.replace(roll.dest, Immediate(0));
					return Ok(())
				}
				else if faces == 1
				{
					// The dice have only one face, so the result is the
					// count itself.
					self.replace(roll.dest, Immediate(count));
					return Ok(())
				}
			}
		}
		// The roll can't be folded, but we may still need to allocate a new
		// rolling record due to renumbering. The operands are guaranteed to be
		// in the map already unless they are constants, as they must have been
		// the destinations of previous instructions.
		let dest = self.next_rolling_record();
		self.replace(roll.dest, dest);
		self.emit(RollStandardDice {
			dest,
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
		if let AddressingMode::Immediate(Immediate(..=0)) = roll.count
		{
			// There are no dice, so it doesn't matter what the faces are.
			self.replace(roll.dest, Immediate(0));
			return Ok(())
		}
		if roll.faces.is_empty()
		{
			// If there are no faces, then the result has to be zero. We can
			// safely ignore any subsequent drop instructions because they will
			// have no effect. Note that this case is forbidden by the grammar,
			// so this case is future-proofing only.
			self.replace(roll.dest, Immediate(0));
			return Ok(())
		}
		if let Some(count) = roll.const_ops()
		{
			// Scan forward for any drop expressions that target the same
			// rolling record. If we are guaranteed to drop all the dice,
			// then it doesn't matter how many faces the dice have.
			let drops = self.find_drop_instructions(roll.dest);
			let can_fold =
				drops.map(|drop| drop.const_ops()).collect::<Vec<_>>();
			if can_fold.iter().all(Option::is_some)
			{
				// Identify the number of dice to drop, then reduce the
				// count accordingly.
				let dropped = can_fold
					.iter()
					.map(|can_fold| can_fold.unwrap().0)
					.sum::<i32>();
				let count = (count.saturating_sub(dropped)).clamp(0, i32::MAX);
				if count <= 0
				{
					// All dice are dropped, so the contribution is zero.
					self.replace(roll.dest, Immediate(0));
					return Ok(())
				}
				if roll.distinct_faces() == 1
				{
					// There's only one distinct face, so multiply the
					// count by the face value to get the result.
					let face = roll.faces[0];
					self.replace(
						roll.dest,
						Immediate(count.saturating_mul(face))
					);
					return Ok(())
				}
			}
		}
		// The roll can't be folded, but we may still need to allocate a new
		// rolling record due to renumbering. The operands are guaranteed to be
		// in the map already unless they are constants, as they must have been
		// the destinations of previous instructions.
		let dest = self.next_rolling_record();
		self.replace(roll.dest, dest);
		self.emit(RollCustomDice {
			dest,
			count: self.replacement(roll.count),
			faces: roll.faces.clone()
		});
		Ok(())
	}

	fn visit_drop_lowest(&mut self, drop: &DropLowest) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the drop instruction in terms of the
		// new rolling record or we eliminate it entirely.
		match self.replacement(drop.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away. Eliminate the drop instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// drop instruction in terms of the new rolling record.
				self.emit(DropLowest {
					dest,
					count: self.replacement(drop.count)
				});
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_drop_highest(&mut self, drop: &DropHighest) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the drop instruction in terms of the
		// new rolling record or we eliminate it entirely.
		match self.replacement(drop.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away. Eliminate the drop instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// drop instruction in terms of the new rolling record.
				self.emit(DropHighest {
					dest,
					count: self.replacement(drop.count)
				});
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_sum_rolling_record(
		&mut self,
		sum: &SumRollingRecord
	) -> Result<(), ()>
	{
		match self.replacement(sum.src)
		{
			src @ AddressingMode::Immediate(_) =>
			{
				// The roll was folded away, so we need to propagate the
				// constant to any subsequent instructions that reference the
				// sum.
				self.replace(sum.dest, src);
				Ok(())
			},
			AddressingMode::RollingRecord(src) =>
			{
				// The roll was not folded away, so we need to rewrite the sum
				// instruction in terms of the potentially new rolling record.
				let dest = self.next_register();
				self.replace(sum.dest, dest);
				self.emit(SumRollingRecord { dest, src });
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			add,
			|dest, op1, op2| Add { dest, op1, op2 }
		)
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			sub,
			|dest, op1, op2| Sub { dest, op1, op2 }
		)
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			mul,
			|dest, op1, op2| Mul { dest, op1, op2 }
		)
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			div,
			|dest, op1, op2| Div { dest, op1, op2 }
		)
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			r#mod,
			|dest, op1, op2| Mod { dest, op1, op2 }
		)
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			exp,
			|dest, op1, op2| Exp { dest, op1, op2 }
		)
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		if let Some(op1) = inst.const_ops()
		{
			self.replace(inst.dest, Immediate(neg(op1)));
		}
		else
		{
			let new = self.next_register();
			self.replace(inst.dest, new);
			self.emit(Neg {
				dest: new,
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

impl ConstantFolder
{
	/// Answer the next available register index.
	///
	/// # Returns
	/// The next available register index.
	#[inline]
	fn next_register(&mut self) -> RegisterIndex
	{
		self.next_register.allocate()
	}

	/// Answer the next available rolling record index.
	///
	/// # Returns
	/// The next available rolling record index.
	#[inline]
	fn next_rolling_record(&mut self) -> RollingRecordIndex
	{
		self.next_rolling_record.allocate()
	}

	/// Set the replacement for the specified operand.
	///
	/// # Parameters
	/// - `old`: The operand to replace.
	/// - `new`: The replacement operand.
	#[inline]
	fn replace(
		&mut self,
		old: impl Into<AddressingMode>,
		new: impl Into<AddressingMode>
	)
	{
		self.replacements.insert(old.into(), new.into());
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
		*self.replacements.get(&op).unwrap_or(&op)
	}

	/// Find every drop instruction subsequent to the current program counter
	/// that targets the given rolling record.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	///
	/// # Returns
	/// The relevant drop instructions.
	fn find_drop_instructions(
		&self,
		dest: RollingRecordIndex
	) -> impl Iterator<Item = &Instruction>
	{
		self.previous.as_ref().unwrap()[self.pc.0 + 1..]
			.iter()
			.filter(move |inst| match inst
			{
				Instruction::DropLowest(inst) => inst.dest == dest,
				Instruction::DropHighest(inst) => inst.dest == dest,
				_ => false
			})
	}

	/// Fold a qualifying binary operation, or emit a new instruction if folding
	/// is not possible.
	///
	/// # Parameters
	/// - `inst`: The instruction to consider.
	/// - `extractor`: A closure that extracts the two source operands from the
	///   instruction.
	/// - `folder`: A closure that performs the binary operation.
	/// - `constructor`: A closure that constructs a new instruction with the
	///   given destination register and source operands.
	fn maybe_fold_binary_instruction<I>(
		&mut self,
		inst: impl Into<Instruction>,
		extractor: impl FnOnce() -> (AddressingMode, AddressingMode),
		folder: impl FnOnce(i32, i32) -> i32,
		constructor: impl FnOnce(RegisterIndex, AddressingMode, AddressingMode) -> I
	) -> Result<(), ()>
	where
		I: Into<Instruction>
	{
		let inst = inst.into();
		let old = inst.destination().unwrap();
		if let Some((op1, op2)) = inst.const_ops()
		{
			self.replace(old, Immediate(folder(op1, op2.unwrap())));
		}
		else
		{
			let new = self.next_register();
			self.replacements.insert(old, new.into());
			let (op1, op2) = extractor();
			self.emit(constructor(
				new,
				self.replacement(op1),
				self.replacement(op2)
			));
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

/// Extract a complete set of [immediate](Immediate) operands from an
/// instruction.
///
/// # Type Parameters
/// - `T`: The tuple type of a successful return.
trait GetConstantOperands<T>
{
	/// Extract the operands from an instruction, but only if they are all
	/// [immediate](Immediate) operands.
	///
	/// # Returns
	/// The tuple of constant operands, or `None` if some operands are not
	/// immediates.
	fn const_ops(&self) -> Option<T>;
}

impl GetConstantOperands<(i32, i32)> for RollRange
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.start, self.end)
		{
			(
				AddressingMode::Immediate(Immediate(start)),
				AddressingMode::Immediate(Immediate(end))
			) => Some((start, end)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for RollStandardDice
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.count, self.faces)
		{
			(
				AddressingMode::Immediate(Immediate(count)),
				AddressingMode::Immediate(Immediate(faces))
			) => Some((count, faces)),
			_ => None
		}
	}
}

impl GetConstantOperands<i32> for RollCustomDice
{
	fn const_ops(&self) -> Option<i32>
	{
		match self.count
		{
			AddressingMode::Immediate(Immediate(count)) => Some(count),
			_ => None
		}
	}
}

impl GetConstantOperands<i32> for DropLowest
{
	fn const_ops(&self) -> Option<i32>
	{
		match self.count
		{
			AddressingMode::Immediate(Immediate(count)) => Some(count),
			_ => None
		}
	}
}

impl GetConstantOperands<i32> for DropHighest
{
	fn const_ops(&self) -> Option<i32>
	{
		match self.count
		{
			AddressingMode::Immediate(Immediate(count)) => Some(count),
			_ => None
		}
	}
}

impl GetConstantOperands<()> for SumRollingRecord
{
	fn const_ops(&self) -> Option<()> { None }
}

impl GetConstantOperands<(i32, i32)> for Add
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for Sub
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for Mul
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for Div
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for Mod
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<(i32, i32)> for Exp
{
	fn const_ops(&self) -> Option<(i32, i32)>
	{
		match (self.op1, self.op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) => Some((op1, op2)),
			_ => None
		}
	}
}

impl GetConstantOperands<i32> for Neg
{
	fn const_ops(&self) -> Option<i32>
	{
		match self.op
		{
			AddressingMode::Immediate(Immediate(op)) => Some(op),
			_ => None
		}
	}
}

impl GetConstantOperands<()> for Return
{
	fn const_ops(&self) -> Option<()> { None }
}

impl GetConstantOperands<(i32, Option<i32>)> for Instruction
{
	fn const_ops(&self) -> Option<(i32, Option<i32>)>
	{
		match self
		{
			Instruction::RollRange(roll) =>
			{
				roll.const_ops().map(|(start, end)| (start, Some(end)))
			},
			Instruction::RollStandardDice(roll) =>
			{
				roll.const_ops().map(|(count, faces)| (count, Some(faces)))
			},
			Instruction::RollCustomDice(roll) =>
			{
				roll.const_ops().map(|count| (count, None))
			},
			Instruction::DropLowest(drop) =>
			{
				drop.const_ops().map(|count| (count, None))
			},
			Instruction::DropHighest(drop) =>
			{
				drop.const_ops().map(|count| (count, None))
			},
			Instruction::SumRollingRecord(_) => None,
			Instruction::Add(add) =>
			{
				add.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Sub(sub) =>
			{
				sub.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Mul(mul) =>
			{
				mul.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Div(div) =>
			{
				div.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Mod(mod_) =>
			{
				mod_.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Exp(exp) =>
			{
				exp.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Neg(neg) =>
			{
				neg.const_ops().map(|result| (result, None))
			},
			Instruction::Return(_) => None
		}
	}
}
