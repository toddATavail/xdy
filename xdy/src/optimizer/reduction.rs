//! # Strength reduction
//!
//! Strength reduction is the process of replacing expensive operations with
//! cheaper ones. The strength reducer heavily utilizes mathematical identities
//! to simplify expressions. For example, the expression `x * 0` can be replaced
//! with `0`, and the expression `x * 1` can be replaced with `x`. It also
//! implements special rules for reducing the strength of dice rolls, such as
//! transforming `1D6` into `[1:6]` instead (as range expressions are cheaper
//! because they don't have to loop). The strength reducer requires the function
//! to be in static single assignment (SSA) form.

use std::collections::HashMap;

use crate::{
	Add, AddressingMode, CanAllocate as _, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, Exp, Function, Immediate, Instruction,
	InstructionVisitor, Mod, Mul, Neg, ProgramCounter, RegisterIndex, Return,
	RollCustomDice, RollRange, RollStandardDice, RollingRecordIndex, Sub,
	SumRollingRecord
};

use crate::Optimizer;

////////////////////////////////////////////////////////////////////////////////
//                            Strength reduction.                             //
////////////////////////////////////////////////////////////////////////////////

/// A strength reducer. The standard optimizer uses a strength reducer to
/// reduce the strength of expressions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StrengthReducer
{
	/// Replacements for registers, due to renumbering or strength reduction,
	/// as a map from originals to replacements. The keys are relative to the
	/// original function, and the values are relative to the optimized
	/// function.
	replacements: HashMap<AddressingMode, AddressingMode>,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The next rolling record to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The replacement instructions.
	instructions: Vec<Instruction>
}

impl Optimizer<()> for StrengthReducer
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
			// Visit each instruction in the function body.
			for instruction in &function.instructions
			{
				instruction.visit(&mut self).unwrap();
			}
			// Compare the old and new instructions. If they are the same, we
			// have reached a fixed point and can return the optimized function.
			if function.instructions == self.instructions
			{
				return Ok(function)
			}
			// Update the function to reflect the new instructions.
			function.register_count = self.next_register.0;
			function.rolling_record_count = self.next_rolling_record.0;
			function.instructions = self.instructions.clone();
			// Reset the optimizer state.
			self.replacements.clear();
			self.next_register = RegisterIndex(start_register);
			self.next_rolling_record = RollingRecordIndex::default();
			self.instructions.clear();
		}
	}
}

impl InstructionVisitor<()> for StrengthReducer
{
	fn visit_roll_range(&mut self, range: &RollRange) -> Result<(), ()>
	{
		// If the bounds are equal, then we can replace the roll with the sole
		// value in the range.
		if range.start == range.end
		{
			self.replace(range.dest, range.start);
			return Ok(());
		}
		// Otherwise, just copy over the instruction.
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
		// If the die has only one face, then we can replace the roll with just
		// the count.
		if let AddressingMode::Immediate(Immediate(1)) = roll.faces
		{
			self.replace(roll.dest, roll.count);
			return Ok(());
		}
		// If the count is exactly one, then we can replace the roll with a
		// range instruction. Range instruction are more efficient than dice
		// rolls, because they don't have to loop.
		if let AddressingMode::Immediate(Immediate(1)) = roll.count
		{
			let dest = self.next_rolling_record();
			self.replace(roll.dest, dest);
			self.emit(RollRange {
				dest,
				start: Immediate(1).into(),
				end: self.replacement(roll.faces)
			});
			return Ok(());
		}
		// We couldn't reduce the strength of the roll, so we just copy it over.
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
		// If the die has only one distinct face, then we can replace the roll
		// with a multiplication.
		if roll.distinct_faces() == 1
		{
			let dest = self.next_register.allocate();
			self.replace(roll.dest, dest);
			self.emit(Mul {
				dest,
				op1: self.replacement(roll.count),
				op2: Immediate(roll.faces[0]).into()
			});
			return Ok(());
		}
		// Organize the faces to determine whether they form a contiguous
		// sequence.
		let mut faces = roll.faces.to_vec();
		faces.sort();
		let mut counter = faces[0];
		let mut contiguous = true;
		for face in &faces[1..]
		{
			if *face == counter + 1
			{
				counter = *face;
			}
			else
			{
				contiguous = false;
				break;
			}
		}
		if let AddressingMode::Immediate(Immediate(1)) = roll.count
		{
			if contiguous
			{
				// There's exactly one face and the allegedly custom faces are
				// actually a contiguous range, so we can perform the
				// replacement.
				let dest = self.next_rolling_record();
				self.replace(roll.dest, dest);
				self.emit(RollRange {
					dest,
					start: Immediate(faces[0]).into(),
					end: Immediate(faces[faces.len() - 1]).into()
				});
				return Ok(());
			}
		}
		if contiguous && faces[0] == 1
		{
			// The faces are contiguous and the smallest face is one, so we can
			// replace this instruction with a standard dice roll.
			let dest = self.next_rolling_record();
			self.replace(roll.dest, dest);
			self.emit(RollStandardDice {
				dest,
				count: self.replacement(roll.count),
				faces: Immediate(faces[faces.len() - 1]).into()
			});
			return Ok(())
		}
		// We couldn't reduce the strength of the roll, so we just copy it over.
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
		match self.replacement(drop.dest)
		{
			AddressingMode::Register(_) | AddressingMode::Immediate(_) =>
			{
				// If the rolling record has been replaced with an ordinary
				// register or immediate, then we can replace the drop lowest
				// instruction with a subtraction.
				let dest = self.next_register();
				self.emit(Sub {
					dest,
					op1: self.replacement(drop.dest),
					op2: self.replacement(drop.count)
				});
				self.replace(drop.dest, dest);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// If the count is zero, then we can eliminate the instruction.
				let count = self.replacement(drop.count);
				if let AddressingMode::Immediate(Immediate(0)) = count
				{
					self.replace(drop.dest, dest);
					return Ok(());
				}
				// If there are previous drop lowest instructions, then we can
				// combine them by dropping the previous instruction and adding
				// its count to our count.
				if let Some(pc) = self.find_drop_instruction(|inst| {
					match DropLowest::try_from(inst.clone()).ok()
					{
						Some(drop) => drop.dest == dest,
						_ => false
					}
				})
				{
					let previous = self.instructions.remove(pc.0);
					let previous_count = *previous.sources().last().unwrap();
					let sum = self.next_register();
					self.emit(Add {
						dest: sum,
						op1: previous_count,
						op2: count
					});
					self.emit(DropLowest {
						dest,
						count: sum.into()
					});
					return Ok(())
				}
				// Otherwise, just copy the drop lowest instruction over.
				self.emit(DropLowest {
					dest,
					count: self.replacement(drop.count)
				});
				Ok(())
			}
		}
	}

	fn visit_drop_highest(&mut self, drop: &DropHighest) -> Result<(), ()>
	{
		match self.replacement(drop.dest)
		{
			AddressingMode::Register(_) | AddressingMode::Immediate(_) =>
			{
				// If the rolling record has been replaced with an ordinary
				// register or immediate, then we can replace the drop lowest
				// instruction with a subtraction.
				let dest = self.next_register();
				self.emit(Sub {
					dest,
					op1: self.replacement(drop.dest),
					op2: self.replacement(drop.count)
				});
				self.replace(drop.dest, dest);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// If the count is zero, then we can eliminate the instruction.
				let count = self.replacement(drop.count);
				if let AddressingMode::Immediate(Immediate(0)) = count
				{
					self.replace(drop.dest, dest);
					return Ok(());
				}
				// If there are previous drop highest instructions, then we can
				// combine them by dropping the previous instruction and adding
				// its count to our count.
				if let Some(pc) = self.find_drop_instruction(|inst| {
					match DropHighest::try_from(inst.clone()).ok()
					{
						Some(drop) => drop.dest == dest,
						_ => false
					}
				})
				{
					let previous = self.instructions.remove(pc.0);
					let previous_count = *previous.sources().last().unwrap();
					let sum = self.next_register();
					self.emit(Add {
						dest: sum,
						op1: previous_count,
						op2: count
					});
					self.emit(DropHighest {
						dest,
						count: sum.into()
					});
					return Ok(())
				}
				// Otherwise, just copy the drop lowest instruction over.
				self.emit(DropHighest {
					dest,
					count: self.replacement(drop.count)
				});
				Ok(())
			}
		}
	}

	fn visit_sum_rolling_record(
		&mut self,
		sum: &SumRollingRecord
	) -> Result<(), ()>
	{
		match self.replacement(sum.src)
		{
			AddressingMode::Immediate(src) =>
			{
				// The source has been replaced with an immediate, so we can
				// eliminate the instruction altogether.
				self.replace(sum.dest, src);
				Ok(())
			},
			AddressingMode::Register(src) =>
			{
				// The rolling record has been replaced with an ordinary
				// register, so we can eliminate the instruction altogether.
				self.replace(sum.dest, src);
				Ok(())
			},
			AddressingMode::RollingRecord(_) =>
			{
				// Copy the instruction over.
				let dest = self.next_register();
				self.replace(sum.dest, dest);
				self.emit(SumRollingRecord {
					dest,
					src: self.replacement(sum.src).try_into().unwrap()
				});
				Ok(())
			}
		}
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		// If either of the operands are zero, we can eliminate the addition
		// entirely.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op1
		{
			self.replace(inst.dest, self.replacement(inst.op2));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, self.replacement(inst.op1));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Add {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		// If the first operand is zero, then we can convert the instruction to
		// a negation.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg {
				dest,
				op: self.replacement(inst.op2)
			});
			return Ok(())
		}
		// If the second operand is zero, then we can eliminate the subtraction
		// completely.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, self.replacement(inst.op1));
			return Ok(())
		}
		// If the operands are equal, then we can eliminate the subtraction
		// completely.
		if inst.op1 == inst.op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Sub {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		// If either of the operands are zero, we can fold the multiplication.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If either of the operands are one, we can eliminate the
		// multiplication.
		if let AddressingMode::Immediate(Immediate(1)) = inst.op1
		{
			self.replace(inst.dest, self.replacement(inst.op2));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(1)) = inst.op2
		{
			self.replace(inst.dest, self.replacement(inst.op1));
			return Ok(())
		}
		// If either of the operands are negative one, we can convert the
		// multiplication to a negation.
		if let AddressingMode::Immediate(Immediate(-1)) = inst.op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg {
				dest,
				op: self.replacement(inst.op2)
			});
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(-1)) = inst.op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg {
				dest,
				op: self.replacement(inst.op1)
			});
			return Ok(())
		}
		// If other operand is two, we can convert the multiplication to an
		// addition.
		if let AddressingMode::Immediate(Immediate(2)) = inst.op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Add {
				dest,
				op1: self.replacement(inst.op2),
				op2: self.replacement(inst.op2)
			});
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(2)) = inst.op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Add {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op1)
			});
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Mul {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		// If either of the operands are zero, we can fold the division.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If the divisor is one, we can eliminate the division.
		if let AddressingMode::Immediate(Immediate(1)) = inst.op2
		{
			self.replace(inst.dest, self.replacement(inst.op1));
			return Ok(())
		}
		// If the divisor is negative one, we can convert the division to a
		// negation.
		if let AddressingMode::Immediate(Immediate(-1)) = inst.op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg {
				dest,
				op: self.replacement(inst.op1)
			});
			return Ok(())
		}
		// If both operands are equal, we can fold the division.
		if inst.op1 == inst.op2
		{
			self.replace(inst.dest, Immediate(1));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Div {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		// If either of the operands are zero, we can fold the modulus.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If the divisor is one, we can eliminate the modulus.
		if let AddressingMode::Immediate(Immediate(1)) = inst.op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Mod {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		// If the exponent is zero, we can fold the exponentiation. This rule
		// also causes 0^0 to be folded to 1.
		if let AddressingMode::Immediate(Immediate(0)) = inst.op2
		{
			self.replace(inst.dest, Immediate(1));
			return Ok(())
		}
		// If the exponent is one, we can eliminate the exponentiation.
		if let AddressingMode::Immediate(Immediate(1)) = inst.op2
		{
			self.replace(inst.dest, self.replacement(inst.op1));
			return Ok(())
		}
		// If the base is one, we can fold the exponentiation. This also works
		// for negative exponents.
		if let AddressingMode::Immediate(Immediate(1)) = inst.op1
		{
			self.replace(inst.dest, Immediate(1));
			return Ok(())
		}
		// If the exponent is two, then we can convert the exponentiation to a
		// multiplication.
		if let AddressingMode::Immediate(Immediate(2)) = inst.op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Mul {
				dest,
				op1: self.replacement(inst.op1),
				op2: self.replacement(inst.op1)
			});
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Exp {
			dest,
			op1: self.replacement(inst.op1),
			op2: self.replacement(inst.op2)
		});
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		// If the operand is also a negation, then we can eliminate both
		// negations.
		if let Some((pc, src)) = self.find_neg_instruction(inst.op)
		{
			// The negation has to be the previous instruction, so we can put
			// back the register that it allocated.
			assert_eq!(pc.0, self.instructions.len() - 1);
			self.next_register.0 -= 1;
			self.instructions.remove(pc.0);
			self.replace(inst.dest, src);
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Neg {
			dest,
			op: self.replacement(inst.op)
		});
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		// Just copy the instruction over.
		self.emit(Return {
			src: self.replacement(inst.src)
		});
		Ok(())
	}
}

impl StrengthReducer
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

	/// Find the negation instruction that writes to the specified register.
	///
	/// # Parameters
	/// - `op`: The target register.
	///
	/// # Returns
	/// The program counter and the source operand of the negation instruction,
	/// if found.
	fn find_neg_instruction(
		&self,
		op: AddressingMode
	) -> Option<(ProgramCounter, AddressingMode)>
	{
		match op
		{
			AddressingMode::Register(reg) =>
			{
				self.instructions.iter().enumerate().rev().find_map(
					|(pc, inst)| match Neg::try_from(inst.clone()).ok()
					{
						Some(neg) if neg.dest == reg =>
						{
							Some((pc.into(), neg.op))
						},
						_ => None
					}
				)
			},
			_ => None
		}
	}

	/// Find the program counter of the last drop instruction that satisfies the
	/// specified filter.
	///
	/// # Parameters
	/// - `dest`: The target register.
	///
	/// # Returns
	/// The program counter of the drop instruction, if found.
	fn find_drop_instruction(
		&self,
		filter: impl Fn(&Instruction) -> bool
	) -> Option<ProgramCounter>
	{
		self.instructions.iter().enumerate().rev().find_map(
			move |(pc, inst)| match filter(inst)
			{
				true => Some(pc.into()),
				false => None
			}
		)
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
