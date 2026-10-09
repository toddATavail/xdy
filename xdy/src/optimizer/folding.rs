//! # Constant folding
//!
//! The constant folder is a simple optimizer that folds expressions with only
//! constant operands into immediates. It also folds a roll whose result its
//! clauses leave known, applying them in order: a roll whose clauses keep no
//! die is zero, e.g., `3D6 reroll ≤ 6 ∞ times`, since an unlimited reroll that
//! matches every face crosses out every die; and a roll of dice with one face
//! is that face times the dice kept, since no reroll can change it, e.g.,
//! `3D1 reroll = 1` is `3`. It follows the chains of a roll through its
//! explodes, too: a roll whose chains still kept an unlimited explode that
//! matches every face saturated is their saturated value, e.g., `3D6 explode ≥
//! 1 ∞ times` is `i32::MAX`, since later rerolls and explodes leave saturated
//! chains alone; and a roll of dice with one face whose explodes match it, or
//! miss it, a known number of times is that face times the dice of its chains,
//! e.g., `3D1 explode = 1 2 times` is `9`. The constant folder requires the
//! function to be in static single assignment (SSA) form.

use std::collections::HashMap;

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, Exp, Explode, Function, Immediate,
	InstructionVisitor, KeepHighest, KeepLowest, Limit, Max, Min, Mod, Mul,
	Neg, ProgramCounter, RegisterIndex, Reroll, Return, RollCustomDice,
	RollRange, RollStandardDice, RollingRecordIndex, RollingRecordKind, Sub,
	SumRollingRecord, evaluator::ResultSet, ir::Instruction
};

use super::{criteria, faces};
use crate::{
	Optimizer, add, div, exp, max, min, r#mod, mul, neg,
	primitives::keep_drops, sub
};

////////////////////////////////////////////////////////////////////////////////
//                             Constant folding.                              //
////////////////////////////////////////////////////////////////////////////////

/// A constant folder. The standard optimizer uses a constant folder to fold
/// expressions with only constant operands into [immediates](Immediate).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstantFolder
{
	/// The previous version of the function body.
	previous: Option<Vec<Instruction>>,

	/// The program counters of the drop, keep, reroll, and explode
	/// instructions of the previous version of the function body, by the
	/// rolling record that they
	/// target, so that finding the clauses of a roll takes time proportional
	/// to their number.
	drops: HashMap<RollingRecordIndex, Vec<usize>>,

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
			self.drops.clear();
			for (pc, instruction) in function.instructions.iter().enumerate()
			{
				let target = match instruction
				{
					Instruction::DropLowest(drop) => drop.dest,
					Instruction::DropHighest(drop) => drop.dest,
					Instruction::KeepLowest(keep) => keep.dest,
					Instruction::KeepHighest(keep) => keep.dest,
					Instruction::Reroll(reroll) => reroll.dest,
					Instruction::Explode(explode) => explode.dest,
					_ => continue
				};
				self.drops.entry(target).or_default().push(pc);
			}
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
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass folded is constant here too, and the whole
		// of a chain folds in a single pass.
		let range = &RollRange {
			dest: range.dest,
			start: self.replacement(range.start),
			end: self.replacement(range.end)
		};
		if let Some((start, end)) = range.const_ops()
		{
			// We can fold the range if it contains one or fewer possible
			// outcomes, or if its clauses cross out its die, or saturate its
			// chain. An empty range is zero whatever its clauses do, but a
			// range with a single value is that value times the dice of its
			// chain only if its clauses keep its die: the grammar admits no
			// clauses on ranges, but strength reduction turns 1Dn into [1:n],
			// clauses and all. Folding involves eliminating the
			// instruction completely and replacing its rolling record with the
			// new constant value. When we subsequently encounter the
			// corresponding SumRollingRecord instruction, we will eliminate it
			// also and replace its destination register with the new constant
			// value.
			if end < start
			{
				// The range is empty, so the result is zero.
				self.replace(range.dest, Immediate(0));
				return Ok(())
			}
			// A range with a single value is a die with one face, which no
			// reroll can change, so its chain holds only that value.
			let faces = faces(&(*range).into());
			let saturated =
				RollingRecordKind::Range { start, end }.saturated_value();
			if let Some(sum) = self.known_sum(
				range.dest,
				Some(1),
				&faces,
				(start == end).then_some(start),
				saturated
			)
			{
				self.replace(range.dest, Immediate(sum));
				return Ok(())
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
			start: range.start,
			end: range.end
		});
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		roll: &RollStandardDice
	) -> Result<(), ()>
	{
		// Resolve the operands once, as for ranges.
		let roll = &RollStandardDice {
			dest: roll.dest,
			count: self.replacement(roll.count),
			faces: self.replacement(roll.faces)
		};
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
		if let AddressingMode::Immediate(Immediate(faces)) = roll.faces
		{
			// Scan forward for any clauses that target the same rolling
			// record, and apply them, in order, to the chains of the dice,
			// whose count need not be constant if the clauses cross out every
			// die. Dice with only one face show `1`, which no reroll can
			// change.
			let count = roll.const_ops().map(|(count, _)| count);
			let die = super::faces(&(*roll).into());
			// The count of dice does not decide the saturated value.
			let saturated = RollingRecordKind::Standard { count: 0, faces }
				.saturated_value();
			if let Some(sum) = self.known_sum(
				roll.dest,
				count,
				&die,
				(faces == 1).then_some(1),
				saturated
			)
			{
				self.replace(roll.dest, Immediate(sum));
				return Ok(())
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
			count: roll.count,
			faces: roll.faces
		});
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		roll: &RollCustomDice
	) -> Result<(), ()>
	{
		// Resolve the count once, as for ranges.
		let roll = &RollCustomDice {
			dest: roll.dest,
			count: self.replacement(roll.count),
			faces: roll.faces.clone()
		};
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
		// Scan forward for any clauses that target the same rolling record, and
		// apply them, in order, to the chains of the dice, whose count need not
		// be constant if the clauses cross out every die. A die with only one
		// distinct face shows it, which no reroll can change.
		let faces = faces(&roll.clone().into());
		// The count of dice does not decide the saturated value, but every
		// face does, however often it repeats.
		let saturated = RollingRecordKind::Custom {
			count: 0,
			faces: roll.faces.clone()
		}
		.saturated_value();
		if let Some(sum) = self.known_sum(
			roll.dest,
			roll.const_ops(),
			&faces,
			(roll.distinct_faces() == 1).then_some(roll.faces[0]),
			saturated
		)
		{
			self.replace(roll.dest, Immediate(sum));
			return Ok(())
		}
		// The roll can't be folded, but we may still need to allocate a new
		// rolling record due to renumbering. The operands are guaranteed to be
		// in the map already unless they are constants, as they must have been
		// the destinations of previous instructions.
		let dest = self.next_rolling_record();
		self.replace(roll.dest, dest);
		self.emit(RollCustomDice {
			dest,
			count: roll.count,
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

	fn visit_keep_lowest(&mut self, keep: &KeepLowest) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the keep instruction in terms of the
		// new rolling record or we eliminate it entirely.
		match self.replacement(keep.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away. Eliminate the keep instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// keep instruction in terms of the new rolling record.
				self.emit(KeepLowest {
					dest,
					count: self.replacement(keep.count)
				});
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_keep_highest(&mut self, keep: &KeepHighest) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the keep instruction in terms of the
		// new rolling record or we eliminate it entirely.
		match self.replacement(keep.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away. Eliminate the keep instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// keep instruction in terms of the new rolling record.
				self.emit(KeepHighest {
					dest,
					count: self.replacement(keep.count)
				});
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_reroll(&mut self, reroll: &Reroll) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the reroll instruction in terms of
		// the new rolling record or we eliminate it entirely.
		match self.replacement(reroll.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away, which it is only if its result is
				// known despite its rerolls. Eliminate the reroll instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// reroll instruction in terms of the new rolling record.
				self.emit(reroll.map(dest, |op| self.replacement(op)));
				Ok(())
			},
			_ => unreachable!()
		}
	}

	fn visit_explode(&mut self, explode: &Explode) -> Result<(), ()>
	{
		// The heavy lifting was already done when the roll instruction was
		// encountered. We either rewrite the explode instruction in terms of
		// the new rolling record or we eliminate it entirely.
		match self.replacement(explode.dest)
		{
			AddressingMode::Immediate(_) =>
			{
				// The rolling record is an immediate value, so the roll must
				// have been folded away, which it is only if its result is
				// known despite its explodes. Eliminate the explode
				// instruction.
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// The rolling record is a register, so we need to rewrite the
				// explode instruction in terms of the new rolling record.
				self.emit(explode.map(dest, |op| self.replacement(op)));
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

	fn visit_max(&mut self, inst: &Max) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			max,
			|dest, op1, op2| Max { dest, op1, op2 }
		)
	}

	fn visit_min(&mut self, inst: &Min) -> Result<(), ()>
	{
		self.maybe_fold_binary_instruction(
			*inst,
			|| (inst.op1, inst.op2),
			min,
			|dest, op1, op2| Min { dest, op1, op2 }
		)
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		// Resolve the operand once, as for binary instructions.
		match self.replacement(inst.op)
		{
			AddressingMode::Immediate(Immediate(op)) =>
			{
				self.replace(inst.dest, Immediate(neg(op)));
			},
			op =>
			{
				let new = self.next_register();
				self.replace(inst.dest, new);
				self.emit(Neg { dest: new, op });
			}
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

	/// Find every drop, keep, reroll, and explode instruction subsequent to the
	/// current program counter that targets the given rolling record.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	///
	/// # Returns
	/// The relevant drop, keep, reroll, and explode instructions.
	fn find_drop_instructions(
		&self,
		dest: RollingRecordIndex
	) -> impl Iterator<Item = &Instruction>
	{
		let previous = self.previous.as_ref().unwrap();
		let pc = self.pc.0;
		self.drops
			.get(&dest)
			.into_iter()
			.flatten()
			.filter(move |&&drop| drop > pc)
			.map(move |&drop| &previous[drop])
	}

	/// Answer the sum of a roll, if every subsequent clause that targets its
	/// rolling record leaves it known: `0` if no [chain](Chains) is still
	/// kept, however many dice there are and whatever their faces; the
	/// saturated value if every chain still kept is saturated, since the sum
	/// folds the totals of the chains, which are all that value, and
	/// saturated chains stay so; and for a die with one face, that face times
	/// the dice of every chain still kept, if their number is known.
	///
	/// # Parameters
	/// - `dest`: The rolling record of the roll.
	/// - `count`: The number of dice, which is positive, or `None` if it is not
	///   constant.
	/// - `faces`: The faces of the die, exactly.
	/// - `face`: The sole face of the die, or `None` if it has several.
	/// - `saturated`: The value of a saturated die of the roll.
	///
	/// # Returns
	/// The sum, or `None` if it is not known.
	fn known_sum(
		&self,
		dest: RollingRecordIndex,
		count: Option<i32>,
		faces: &ResultSet,
		face: Option<i32>,
		saturated: i32
	) -> Option<i32>
	{
		match self.chains(dest, count, faces)
		{
			Chains { count: Some(0), .. } => Some(0),
			Chains {
				count: Some(_),
				saturated: Some(true),
				..
			} => Some(saturated),
			Chains {
				count: Some(count),
				saturated: Some(false),
				length: Some(length)
			} =>
			{
				// The dice of a chain share their sign, so the ascending
				// saturating folds of the dice of a chain, and of the totals
				// of the chains, saturate only where their products do.
				face.map(|face| {
					count.saturating_mul(length.saturating_mul(face))
				})
			},
			_ => None
		}
	}

	/// Answer what is known of the [chains](Chains) of the specified number of
	/// dice of a roll after every subsequent clause that targets its rolling
	/// record. The clauses apply in order, since a keep counts from the chains
	/// still kept, and so does not commute with drops or other keeps; see
	/// [`dropped_chains`](Self::dropped_chains),
	/// [`rerolled_chains`](Self::rerolled_chains), and
	/// [`exploded_chains`](Self::exploded_chains). Once no chain is kept, none
	/// is kept after any clause, so neither the count of dice nor the operands
	/// of the later clauses need be known.
	///
	/// # Parameters
	/// - `dest`: The rolling record of the roll.
	/// - `count`: The number of dice, which is positive, or `None` if it is not
	///   constant.
	/// - `faces`: The faces of the die, exactly.
	///
	/// # Returns
	/// The chains still kept.
	fn chains(
		&self,
		dest: RollingRecordIndex,
		count: Option<i32>,
		faces: &ResultSet
	) -> Chains
	{
		// Each die begins a chain of its own, which no explode has saturated.
		let chains = Chains {
			count,
			saturated: Some(false),
			length: Some(1)
		};
		self.find_drop_instructions(dest)
			.fold(chains, |chains, clause| match clause
			{
				_ if chains.count == Some(0) => chains,
				Instruction::Reroll(reroll) =>
				{
					self.rerolled_chains(chains, reroll, faces)
				},
				Instruction::Explode(explode) =>
				{
					self.exploded_chains(chains, explode, faces)
				},
				clause => self.dropped_chains(chains, clause)
			})
	}

	/// Answer what is known of the [chains](Chains) still kept after the
	/// specified drop or keep, which crosses out whole chains: a drop of `c`
	/// drops `c` of the chains still kept, or none if `c` is zero or less, and
	/// a keep of `c` drops [all but `c`](keep_drops) of them, or all of them if
	/// `c` is zero or less. Every chain still kept is saturated, or none is,
	/// and every one that is not holds the same number of kept dice, so the
	/// chains left are like those crossed out.
	///
	/// # Parameters
	/// - `chains`: The chains still kept before the drop or keep.
	/// - `drop`: The drop or keep, relative to the original function.
	///
	/// # Returns
	/// The chains still kept after the drop or keep.
	fn dropped_chains(&self, chains: Chains, drop: &Instruction) -> Chains
	{
		let count =
			chains
				.count
				.zip(self.resolved_count(drop))
				.map(|(kept, count)| {
					let dropped = match drop
					{
						Instruction::KeepLowest(_)
						| Instruction::KeepHighest(_) => keep_drops(kept, 0, 0, count),
						_ => count.clamp(0, kept)
					};
					kept - dropped
				});
		Chains { count, ..chains }
	}

	/// Answer what is known of the [chains](Chains) still kept after the
	/// specified reroll, which leaves saturated chains alone, and acts on
	/// each kept die of every other: it crosses out every such die if it is
	/// unlimited and its predicate matches every face of the die, since no
	/// reroll could ever stop, and so crosses out every chain that is not
	/// saturated; and any other reroll replaces each die that it crosses out,
	/// in its place in its chain, and so keeps as many chains as it found,
	/// each with as many kept dice.
	///
	/// # Parameters
	/// - `chains`: The chains still kept before the reroll.
	/// - `reroll`: The reroll, relative to the original function.
	/// - `faces`: The faces of the die, exactly.
	///
	/// # Returns
	/// The chains still kept after the reroll.
	fn rerolled_chains(
		&self,
		chains: Chains,
		reroll: &Reroll,
		faces: &ResultSet
	) -> Chains
	{
		let count = match chains.saturated
		{
			// Rerolls leave saturated chains alone.
			Some(true) => chains.count,
			Some(false) => self.rerolled_dice(chains.count, reroll, faces),
			// The reroll crosses out the chains only if they are not
			// saturated, so the count is known only if the reroll keeps them.
			None => self
				.rerolled_dice(chains.count, reroll, faces)
				.filter(|&count| Some(count) == chains.count)
		};
		Chains { count, ..chains }
	}

	/// Answer what is known of the [chains](Chains) still kept after the
	/// specified explode, which leaves saturated chains alone, and extends the
	/// end of every other that it matches. An explode limited to zero times or
	/// fewer, or whose predicate matches no face, extends nothing. An
	/// unlimited explode whose predicate matches every face saturates every
	/// chain still kept, since every end matches. A limited explode whose
	/// predicate matches every face extends every chain by exactly its limit,
	/// since every extension matches too. Any other explode extends chains by
	/// numbers of dice that are not known, and an unlimited one whose
	/// predicate may match every face may saturate them. No explode crosses
	/// out a chain.
	///
	/// # Parameters
	/// - `chains`: The chains still kept before the explode.
	/// - `explode`: The explode, relative to the original function.
	/// - `faces`: The faces of the die, exactly.
	///
	/// # Returns
	/// The chains still kept after the explode.
	fn exploded_chains(
		&self,
		chains: Chains,
		explode: &Explode,
		faces: &ResultSet
	) -> Chains
	{
		let limit = match explode.limit
		{
			Limit::Times(limit) => match self.replacement(limit)
			{
				AddressingMode::Immediate(Immediate(..=0)) => return chains,
				AddressingMode::Immediate(Immediate(limit)) =>
				{
					Some(Some(limit))
				},
				_ => Some(None)
			},
			Limit::Unlimited => None
		};
		let criteria =
			criteria(&explode.predicate.map(|op| self.replacement(op)));
		if chains.saturated == Some(true) || !faces.meets(&criteria.possible)
		{
			// Explodes leave saturated chains alone, and the predicate matches
			// no face.
			return chains
		}
		let certain = faces.is_subset(&criteria.certain);
		match limit
		{
			None if certain => Chains {
				saturated: Some(true),
				..chains
			},
			None if faces.is_subset(&criteria.possible) => Chains {
				saturated: None,
				length: None,
				..chains
			},
			Some(Some(limit)) if certain => Chains {
				length: chains
					.length
					.map(|length| length.saturating_add(limit)),
				..chains
			},
			_ => Chains {
				length: None,
				..chains
			}
		}
	}

	/// Answer how many of the specified number of dice of a roll remain kept
	/// after the specified reroll, which crosses out every die still kept if
	/// it is unlimited and its predicate matches every face of the die, and
	/// otherwise keeps as many dice as it found.
	///
	/// # Parameters
	/// - `kept`: The number of dice kept before the reroll, or `None` if it is
	///   not known.
	/// - `reroll`: The reroll, relative to the original function.
	/// - `faces`: The faces of the die, exactly.
	///
	/// # Returns
	/// The number of dice kept after the reroll, or `None` if it is not known.
	fn rerolled_dice(
		&self,
		kept: Option<i32>,
		reroll: &Reroll,
		faces: &ResultSet
	) -> Option<i32>
	{
		match reroll.limit
		{
			Limit::Times(_) => kept,
			Limit::Unlimited =>
			{
				let criteria =
					criteria(&reroll.predicate.map(|op| self.replacement(op)));
				if faces.is_subset(&criteria.certain)
				{
					// The predicate surely matches every face.
					Some(0)
				}
				else if !faces.is_subset(&criteria.possible)
				{
					// The predicate surely misses some face.
					kept
				}
				else
				{
					None
				}
			}
		}
	}

	/// Answer the count of the specified drop or keep instruction, if it is
	/// constant, whether in the instruction or by an earlier fold of this
	/// pass.
	///
	/// # Parameters
	/// - `drop`: A drop or keep instruction.
	///
	/// # Returns
	/// The constant count, or `None` if the count is not constant, or if the
	/// instruction is not a drop or keep.
	fn resolved_count(&self, drop: &Instruction) -> Option<i32>
	{
		let count = match drop
		{
			Instruction::DropLowest(drop) => drop.count,
			Instruction::DropHighest(drop) => drop.count,
			Instruction::KeepLowest(keep) => keep.count,
			Instruction::KeepHighest(keep) => keep.count,
			_ => return None
		};
		match self.replacement(count)
		{
			AddressingMode::Immediate(Immediate(count)) => Some(count),
			_ => None
		}
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
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass folded is constant here too, and the whole
		// of a chain folds in a single pass.
		let (op1, op2) = extractor();
		let (op1, op2) = (self.replacement(op1), self.replacement(op2));
		match (op1, op2)
		{
			(
				AddressingMode::Immediate(Immediate(op1)),
				AddressingMode::Immediate(Immediate(op2))
			) =>
			{
				self.replace(old, Immediate(folder(op1, op2)));
			},
			_ =>
			{
				let new = self.next_register();
				self.replacements.insert(old, new.into());
				self.emit(constructor(new, op1, op2));
			}
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

/// What a [constant folder](ConstantFolder) knows of the chains still kept of
/// a roll, as its clauses leave them. Each die of the roll begins a chain, and
/// drops and keeps cross out whole chains, so the chains still kept are alike:
/// an unlimited explode that matches every face saturates all of them, and
/// any other explode or reroll acts on each alike, so either all of them are
/// saturated or none is, and those that are not hold equally many kept dice
/// whenever their number is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chains
{
	/// The number of chains still kept, which is never negative, or `None` if
	/// it depends on a count of dice, a drop, or a keep that is not constant,
	/// or on whether a predicate that is not constant matches every face.
	count: Option<i32>,

	/// Whether every chain still kept is saturated, `Some(true)`, or none is,
	/// `Some(false)`, or `None` if it depends on whether a predicate that is
	/// not constant matches every face.
	saturated: Option<bool>,

	/// The number of kept dice of each chain still kept that is not
	/// saturated, or `None` if it depends on the faces that the dice show, or
	/// on an operand that is not constant.
	length: Option<i32>
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

impl GetConstantOperands<i32> for KeepLowest
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

impl GetConstantOperands<i32> for KeepHighest
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

impl GetConstantOperands<(i32, i32)> for Max
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

impl GetConstantOperands<(i32, i32)> for Min
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
			Instruction::KeepLowest(drop) =>
			{
				drop.const_ops().map(|count| (count, None))
			},
			Instruction::KeepHighest(drop) =>
			{
				drop.const_ops().map(|count| (count, None))
			},
			Instruction::Reroll(_) => None,
			Instruction::Explode(_) => None,
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
			Instruction::Max(max) =>
			{
				max.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Min(min) =>
			{
				min.const_ops().map(|(op1, op2)| (op1, Some(op2)))
			},
			Instruction::Neg(neg) =>
			{
				neg.const_ops().map(|result| (result, None))
			},
			Instruction::Return(_) => None
		}
	}
}
