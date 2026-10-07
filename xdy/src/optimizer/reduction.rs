//! # Strength reduction
//!
//! Strength reduction is the process of replacing expensive operations with
//! cheaper ones. The strength reducer heavily utilizes mathematical identities
//! to simplify expressions. For example, the expression `x * 0` can be replaced
//! with `0`, and the expression `x * 1` can be replaced with `x`. It also
//! implements special rules for reducing the strength of dice rolls, such as
//! transforming `1D6` into `[1:6]` instead (as range expressions are cheaper
//! because they don't have to loop), and a roll of dice that all show one face,
//! e.g., `{x}D1 keep highest 2`, with its count of dice, which its drops reduce
//! and its keeps cap with a [minimum](Min). It eliminates rerolls that cannot
//! change any die: those limited to zero times or fewer, limited rerolls of
//! dice with one face, and rerolls whose predicates match none of the results
//! that the dice still kept may show, which it tracks through the clauses of
//! each roll, e.g., the second reroll of `3D6 reroll = 1 ∞ times reroll = 1`.
//!
//! The strength reducer knows the bounds of every register, as the [bounds
//! evaluator](crate::evaluator::BoundsEvaluator) computes them over every
//! binding, and eliminates a [maximum](Max) or [minimum](Min) whose operands'
//! bounds prove that one of them always wins, e.g., `max(x, 0)` where `x` is
//! the sum of `1D6`, or `min(max(x, 5), 3)`, which is always `3`. The identity
//! and absorbing elements of each, the extremes of `i32`, are cases of this
//! rule.
//!
//! The strength reducer requires the function to be in static single
//! assignment (SSA) form.

use std::collections::{HashMap, HashSet};

use crate::{
	Add, AddressingMode, CanAllocate as _, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, EvaluationBounds, Exp, Function, Immediate,
	Instruction, InstructionVisitor, KeepHighest, KeepLowest, Limit, Max, Min,
	Mod, Mul, Neg, ProgramCounter, RegisterIndex, Reroll, Return,
	RollCustomDice, RollRange, RollStandardDice, RollingRecordIndex, Sub,
	SumRollingRecord, evaluator::ResultSet
};

use super::{criteria, faces, register_bounds};
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

	/// The rolling records that drop, keep, or reroll instructions target,
	/// relative to the original function. A range into one of them must
	/// remain a range, even if it has only one value, since a drop, keep, or
	/// reroll can remove that value.
	dropped: HashSet<RollingRecordIndex>,

	/// The rolling records that reroll instructions target, relative to the
	/// original function. A roll into one of them is never [reduced to its
	/// count of dice](Self::reduce_to_count), since the reduction accounts for
	/// drops and keeps but not for rerolls.
	rerolled: HashSet<RollingRecordIndex>,

	/// The bounds of each register of the original function, over every
	/// binding, by index. A replacement holds the same value as its original,
	/// so these bound the replacements too.
	original_bounds: Vec<EvaluationBounds>,

	/// The bounds of the registers of the optimized function, as far as they
	/// are known, which bound the [greatest](Self::greater) and
	/// [least](Self::lesser) of operands. A register absent from the map is
	/// unconstrained.
	bounds: HashMap<RegisterIndex, EvaluationBounds>,

	/// The sole face of each rolling record, relative to the original
	/// function, that was reduced to its count of dice, which is never
	/// negative. Every die shows the same face, so a drop subtracts from the
	/// count and a keep caps it, whichever dice they cross out, and the sum is
	/// the count times the face.
	faces: HashMap<RollingRecordIndex, i32>,

	/// The results that the dice of each rolling record may show, relative to
	/// the original function, for each roll that remains a roll.
	results: HashMap<RollingRecordIndex, Results>,

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
			self.original_bounds = register_bounds(&function);
			self.dropped = function
				.instructions
				.iter()
				.filter_map(|inst| match inst
				{
					Instruction::DropLowest(drop) => Some(drop.dest),
					Instruction::DropHighest(drop) => Some(drop.dest),
					Instruction::KeepLowest(keep) => Some(keep.dest),
					Instruction::KeepHighest(keep) => Some(keep.dest),
					Instruction::Reroll(reroll) => Some(reroll.dest),
					_ => None
				})
				.collect();
			self.rerolled = function
				.instructions
				.iter()
				.filter_map(|inst| match inst
				{
					Instruction::Reroll(reroll) => Some(reroll.dest),
					_ => None
				})
				.collect();
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
			self.bounds.clear();
			self.faces.clear();
			self.results.clear();
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
		// Resolve the bounds once, as for arithmetic instructions.
		let (start, end) =
			(self.replacement(range.start), self.replacement(range.end));
		self.track_roll(RollRange {
			dest: range.dest,
			start,
			end
		});
		// If the bounds are equal, then we can replace the roll with the sole
		// value in the range, unless its result may be dropped.
		if start == end && !self.dropped.contains(&range.dest)
		{
			self.replace(range.dest, start);
			return Ok(());
		}
		// Otherwise, just copy over the instruction.
		let dest = self.next_rolling_record();
		self.replace(range.dest, dest);
		self.emit(RollRange { dest, start, end });
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		roll: &RollStandardDice
	) -> Result<(), ()>
	{
		// If the die has only one face, then every die shows 1, so we can
		// replace the roll with its count of dice, or zero if the count is
		// negative.
		// Resolve the operands once, as for arithmetic instructions. The
		// helpers take the operands of the original function, and resolve them
		// themselves.
		let (count, faces) =
			(self.replacement(roll.count), self.replacement(roll.faces));
		self.track_roll(RollStandardDice {
			dest: roll.dest,
			count,
			faces
		});
		if let AddressingMode::Immediate(Immediate(1)) = faces
			&& !self.rerolled.contains(&roll.dest)
		{
			self.reduce_to_count(roll.dest, roll.count, 1);
			return Ok(());
		}
		// If the count is exactly one, then we can replace the roll with a
		// range instruction. Range instruction are more efficient than dice
		// rolls, because they don't have to loop.
		if let AddressingMode::Immediate(Immediate(1)) = count
		{
			let dest = self.next_rolling_record();
			self.replace(roll.dest, dest);
			self.emit(RollRange {
				dest,
				start: Immediate(1).into(),
				end: faces
			});
			return Ok(());
		}
		// We couldn't reduce the strength of the roll, so we just copy it over.
		let dest = self.next_rolling_record();
		self.replace(roll.dest, dest);
		self.emit(RollStandardDice { dest, count, faces });
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		roll: &RollCustomDice
	) -> Result<(), ()>
	{
		self.track_roll(roll.clone());
		// If the die has only one distinct face, then every die shows it, so
		// we can replace the roll with its count of dice, or zero if the count
		// is negative. The sum multiplies the count by the face.
		if roll.distinct_faces() == 1 && !self.rerolled.contains(&roll.dest)
		{
			self.reduce_to_count(roll.dest, roll.count, roll.faces[0]);
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
			if counter.checked_add(1) == Some(*face)
			{
				counter = *face;
			}
			else
			{
				contiguous = false;
				break;
			}
		}
		if let AddressingMode::Immediate(Immediate(1)) =
			self.replacement(roll.count)
			&& contiguous
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
				// The roll was reduced to its count of dice, which all show the
				// same face, so dropping dice subtracts from the count.
				self.drop_from_count(drop.dest, drop.count);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// If the count is zero or less, then the instruction drops
				// nothing, so we can eliminate it.
				let count = self.replacement(drop.count);
				if let AddressingMode::Immediate(Immediate(..=0)) = count
				{
					self.replace(drop.dest, dest);
					return Ok(());
				}
				// If there is a previous drop lowest instruction with a
				// constant count, and our count is also constant, then we can
				// combine them by dropping the previous instruction and adding
				// its count to our count, unless a keep intervenes, since a
				// keep counts from the dice that the drops before it left. Both
				// counts are positive, since a count of zero or less was
				// eliminated. A count that is not constant may be negative,
				// which drops nothing, so it cannot be added.
				if let AddressingMode::Immediate(Immediate(count)) = count
					&& let Some(pc) = self.find_drop_instruction(dest, |inst| {
						match DropLowest::try_from(inst.clone()).ok()
						{
							Some(drop) =>
							{
								drop.dest == dest
									&& matches!(
										drop.count,
										AddressingMode::Immediate(_)
									)
							},
							_ => false
						}
					})
				{
					let previous = self.instructions.remove(pc.0);
					let AddressingMode::Immediate(Immediate(previous_count)) =
						*previous.sources().last().unwrap()
					else
					{
						unreachable!()
					};
					self.emit(DropLowest {
						dest,
						count: Immediate(previous_count.saturating_add(count))
							.into()
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
				// The roll was reduced to its count of dice, which all show the
				// same face, so dropping dice subtracts from the count.
				self.drop_from_count(drop.dest, drop.count);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// If the count is zero or less, then the instruction drops
				// nothing, so we can eliminate it.
				let count = self.replacement(drop.count);
				if let AddressingMode::Immediate(Immediate(..=0)) = count
				{
					self.replace(drop.dest, dest);
					return Ok(());
				}
				// If there is a previous drop highest instruction with a
				// constant count, and our count is also constant, then we can
				// combine them by dropping the previous instruction and adding
				// its count to our count, unless a keep intervenes, since a
				// keep counts from the dice that the drops before it left. Both
				// counts are positive, since a count of zero or less was
				// eliminated. A count that is not constant may be negative,
				// which drops nothing, so it cannot be added.
				if let AddressingMode::Immediate(Immediate(count)) = count
					&& let Some(pc) = self.find_drop_instruction(dest, |inst| {
						match DropHighest::try_from(inst.clone()).ok()
						{
							Some(drop) =>
							{
								drop.dest == dest
									&& matches!(
										drop.count,
										AddressingMode::Immediate(_)
									)
							},
							_ => false
						}
					})
				{
					let previous = self.instructions.remove(pc.0);
					let AddressingMode::Immediate(Immediate(previous_count)) =
						*previous.sources().last().unwrap()
					else
					{
						unreachable!()
					};
					self.emit(DropHighest {
						dest,
						count: Immediate(previous_count.saturating_add(count))
							.into()
					});
					return Ok(())
				}
				// Otherwise, just copy the drop highest instruction over.
				self.emit(DropHighest {
					dest,
					count: self.replacement(drop.count)
				});
				Ok(())
			}
		}
	}

	fn visit_keep_lowest(&mut self, keep: &KeepLowest) -> Result<(), ()>
	{
		match self.replacement(keep.dest)
		{
			AddressingMode::Register(_) | AddressingMode::Immediate(_) =>
			{
				// The roll was reduced to its count of dice, which all show the
				// same face, so keeping dice caps the count.
				self.keep_from_count(keep.dest, keep.count);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// A keep keeps nothing for a count of zero or less, so, unlike
				// a drop, it is never eliminated.
				self.emit(KeepLowest {
					dest,
					count: self.replacement(keep.count)
				});
				Ok(())
			}
		}
	}

	fn visit_keep_highest(&mut self, keep: &KeepHighest) -> Result<(), ()>
	{
		match self.replacement(keep.dest)
		{
			AddressingMode::Register(_) | AddressingMode::Immediate(_) =>
			{
				// The roll was reduced to its count of dice, which all show the
				// same face, so keeping dice caps the count.
				self.keep_from_count(keep.dest, keep.count);
				Ok(())
			},
			AddressingMode::RollingRecord(dest) =>
			{
				// A keep keeps nothing for a count of zero or less, so, unlike
				// a drop, it is never eliminated.
				self.emit(KeepHighest {
					dest,
					count: self.replacement(keep.count)
				});
				Ok(())
			}
		}
	}

	fn visit_reroll(&mut self, reroll: &Reroll) -> Result<(), ()>
	{
		// A roll that rerolls is never reduced to a value, so its record
		// remains a record.
		let AddressingMode::RollingRecord(dest) = self.replacement(reroll.dest)
		else
		{
			unreachable!("a reroll of a roll reduced to a value")
		};
		let new = reroll.map(dest, |op| self.replacement(op));
		let criteria = criteria(&new.predicate);
		let unlimited = match new.limit
		{
			// A limit of zero or less rerolls nothing, so we can eliminate the
			// reroll.
			Limit::Times(AddressingMode::Immediate(Immediate(..=0))) =>
			{
				return Ok(())
			},
			Limit::Times(_) => false,
			Limit::Unlimited => true
		};
		let results = self.results.get_mut(&reroll.dest).unwrap();
		// If the predicate cannot match any result that the dice still kept may
		// show, then the reroll crosses out nothing, so we can eliminate it. If
		// the die has only one face, then a limited reroll replaces each die
		// that it crosses out with that same face, so we can eliminate it too.
		if !results.live.meets(&criteria.possible)
			|| (!unlimited && results.faces.len() == 1)
		{
			return Ok(())
		}
		// Update the results that the dice still kept may show. An unlimited
		// reroll that surely matches every face crosses out every die still
		// kept; any other unlimited reroll leaves only faces that the
		// predicate misses, though which faces these are may not be known; and
		// a limited reroll may leave any face at all, since the limit may stop
		// it on a face that the predicate matches.
		results.live = match unlimited
		{
			true if results.faces.is_subset(&criteria.certain) =>
			{
				ResultSet::default()
			},
			true => results.faces.difference(&criteria.certain),
			false => results.faces.clone()
		};
		self.emit(new);
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		sum: &SumRollingRecord
	) -> Result<(), ()>
	{
		match self.replacement(sum.src)
		{
			src @ (AddressingMode::Immediate(_)
			| AddressingMode::Register(_)) =>
			{
				// The rolling record has been replaced with a value, so we can
				// eliminate the instruction altogether. If the value is a count
				// of dice that all show some other face than 1, then the sum
				// is the count times the face.
				let sum_value = match self.faces.get(&sum.src)
				{
					Some(&face) if face != 1 => match src
					{
						AddressingMode::Immediate(Immediate(count)) =>
						{
							Immediate(count.saturating_mul(face)).into()
						},
						count =>
						{
							let dest = self.next_register();
							self.bound(
								dest,
								self.operand_bounds(count) * face.into()
							);
							self.emit(Mul {
								dest,
								op1: count,
								op2: Immediate(face).into()
							});
							dest.into()
						}
					},
					_ => src
				};
				self.replace(sum.dest, sum_value);
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
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If either of the operands are zero, we can eliminate the addition
		// entirely.
		if let AddressingMode::Immediate(Immediate(0)) = op1
		{
			self.replace(inst.dest, op2);
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, op1);
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Add { dest, op1, op2 });
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If the first operand is zero, then we can convert the instruction to
		// a negation.
		if let AddressingMode::Immediate(Immediate(0)) = op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg { dest, op: op2 });
			return Ok(())
		}
		// If the second operand is zero, then we can eliminate the subtraction
		// completely.
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, op1);
			return Ok(())
		}
		// If the operands are equal, then we can eliminate the subtraction
		// completely.
		if op1 == op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Sub { dest, op1, op2 });
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If either of the operands are zero, we can fold the multiplication.
		if let AddressingMode::Immediate(Immediate(0)) = op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If either of the operands are one, we can eliminate the
		// multiplication.
		if let AddressingMode::Immediate(Immediate(1)) = op1
		{
			self.replace(inst.dest, op2);
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(1)) = op2
		{
			self.replace(inst.dest, op1);
			return Ok(())
		}
		// If either of the operands are negative one, we can convert the
		// multiplication to a negation.
		if let AddressingMode::Immediate(Immediate(-1)) = op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg { dest, op: op2 });
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(-1)) = op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg { dest, op: op1 });
			return Ok(())
		}
		// If other operand is two, we can convert the multiplication to an
		// addition.
		if let AddressingMode::Immediate(Immediate(2)) = op1
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Add {
				dest,
				op1: op2,
				op2
			});
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(2)) = op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Add {
				dest,
				op1,
				op2: op1
			});
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Mul { dest, op1, op2 });
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If either of the operands are zero, we can fold the division.
		if let AddressingMode::Immediate(Immediate(0)) = op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If the divisor is one, we can eliminate the division.
		if let AddressingMode::Immediate(Immediate(1)) = op2
		{
			self.replace(inst.dest, op1);
			return Ok(())
		}
		// If the divisor is negative one, we can convert the division to a
		// negation.
		if let AddressingMode::Immediate(Immediate(-1)) = op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Neg { dest, op: op1 });
			return Ok(())
		}
		// A value divided by itself is not always one, since division by zero
		// is zero, so equal operands do not fold.
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Div { dest, op1, op2 });
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If either of the operands are zero, we can fold the modulus.
		if let AddressingMode::Immediate(Immediate(0)) = op1
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// If the divisor is one, we can eliminate the modulus.
		if let AddressingMode::Immediate(Immediate(1)) = op2
		{
			self.replace(inst.dest, Immediate(0));
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Mod { dest, op1, op2 });
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		// Resolve the operands once, so that an operand that an earlier
		// instruction of this pass reduced is seen as reduced, and a whole
		// chain reduces in a single pass.
		let (op1, op2) =
			(self.replacement(inst.op1), self.replacement(inst.op2));
		// If the exponent is zero, we can fold the exponentiation. This rule
		// also causes 0^0 to be folded to 1.
		if let AddressingMode::Immediate(Immediate(0)) = op2
		{
			self.replace(inst.dest, Immediate(1));
			return Ok(())
		}
		// If the exponent is one, we can eliminate the exponentiation.
		if let AddressingMode::Immediate(Immediate(1)) = op2
		{
			self.replace(inst.dest, op1);
			return Ok(())
		}
		// If the base is one, we can fold the exponentiation. This also works
		// for negative exponents.
		if let AddressingMode::Immediate(Immediate(1)) = op1
		{
			self.replace(inst.dest, Immediate(1));
			return Ok(())
		}
		// If the exponent is two, then we can convert the exponentiation to a
		// multiplication.
		if let AddressingMode::Immediate(Immediate(2)) = op2
		{
			let dest = self.next_register();
			self.replace(inst.dest, dest);
			self.emit(Mul {
				dest,
				op1,
				op2: op1
			});
			return Ok(())
		}
		// Otherwise, just copy the instruction over.
		let dest = self.next_register();
		self.replace(inst.dest, dest);
		self.emit(Exp { dest, op1, op2 });
		Ok(())
	}

	fn visit_max(&mut self, inst: &Max) -> Result<(), ()>
	{
		let max = self
			.greater(self.replacement(inst.op1), self.replacement(inst.op2));
		self.replace(inst.dest, max);
		Ok(())
	}

	fn visit_min(&mut self, inst: &Min) -> Result<(), ()>
	{
		let min =
			self.lesser(self.replacement(inst.op1), self.replacement(inst.op2));
		self.replace(inst.dest, min);
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		// A double negation is not always the identity, since negation
		// saturates: the negation of the negation of `i32::MIN` is
		// `-i32::MAX`. So a negation never reduces.
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
		let (old, new) = (old.into(), new.into());
		// The replacement holds the same value as the original, so it shares
		// the original's bounds.
		if let (AddressingMode::Register(old), AddressingMode::Register(new)) =
			(old, new)
		{
			self.bound(new, self.original_bounds[old.0]);
		}
		self.replacements.insert(old, new);
	}

	/// Narrow the bounds of the specified register of the optimized function
	/// to their intersection with the specified bounds.
	///
	/// # Parameters
	/// - `reg`: The register, relative to the optimized function.
	/// - `bounds`: Bounds that contain every value that the register may hold.
	fn bound(&mut self, reg: RegisterIndex, bounds: EvaluationBounds)
	{
		let current = self
			.bounds
			.entry(reg)
			.or_insert(EvaluationBounds::unconstrained());
		*current =
			(current.min.max(bounds.min), current.max.min(bounds.max)).into();
	}

	/// Answer the bounds of the specified operand of the optimized function.
	///
	/// # Parameters
	/// - `op`: The operand, an immediate or a register, relative to the
	///   optimized function.
	///
	/// # Returns
	/// The bounds of the operand: a single value for an immediate, and the
	/// known bounds of a register, or the whole of `i32` if none are known.
	fn operand_bounds(&self, op: AddressingMode) -> EvaluationBounds
	{
		match op
		{
			AddressingMode::Immediate(Immediate(value)) => value.into(),
			AddressingMode::Register(reg) => self
				.bounds
				.get(&reg)
				.copied()
				.unwrap_or(EvaluationBounds::unconstrained()),
			AddressingMode::RollingRecord(_) =>
			{
				unreachable!("a rolling record has no value")
			}
		}
	}

	/// Answer an operand whose value is the greater of the specified
	/// operands: either operand, if their bounds prove that it is never less
	/// than the other, or else a new register that a [maximum](Max) computes.
	/// The least value of `i32` is the identity of the maximum and the
	/// greatest absorbs any other, so either is eliminated thus, as is the
	/// maximum of two constants.
	///
	/// # Parameters
	/// - `op1`: The first operand, relative to the optimized function.
	/// - `op2`: The second operand, relative to the optimized function.
	///
	/// # Returns
	/// The greater operand, relative to the optimized function.
	fn greater(
		&mut self,
		op1: AddressingMode,
		op2: AddressingMode
	) -> AddressingMode
	{
		let (bounds1, bounds2) =
			(self.operand_bounds(op1), self.operand_bounds(op2));
		// The maximum of a value and itself is the value.
		if op1 == op2 || bounds1.min >= bounds2.max
		{
			return op1
		}
		if bounds2.min >= bounds1.max
		{
			return op2
		}
		let dest = self.next_register();
		self.bound(dest, bounds1.max(bounds2));
		self.emit(Max { dest, op1, op2 });
		dest.into()
	}

	/// Answer an operand whose value is the lesser of the specified operands:
	/// either operand, if their bounds prove that it is never greater than the
	/// other, or else a new register that a [minimum](Min) computes. The
	/// greatest value of `i32` is the identity of the minimum and the least
	/// absorbs any other, so either is eliminated thus, as is the minimum of
	/// two constants.
	///
	/// # Parameters
	/// - `op1`: The first operand, relative to the optimized function.
	/// - `op2`: The second operand, relative to the optimized function.
	///
	/// # Returns
	/// The lesser operand, relative to the optimized function.
	fn lesser(
		&mut self,
		op1: AddressingMode,
		op2: AddressingMode
	) -> AddressingMode
	{
		let (bounds1, bounds2) =
			(self.operand_bounds(op1), self.operand_bounds(op2));
		// The minimum of a value and itself is the value.
		if op1 == op2 || bounds1.max <= bounds2.min
		{
			return op1
		}
		if bounds2.max <= bounds1.min
		{
			return op2
		}
		let dest = self.next_register();
		self.bound(dest, bounds1.min(bounds2));
		self.emit(Min { dest, op1, op2 });
		dest.into()
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

	/// Find the program counter of the last drop instruction that satisfies the
	/// specified filter, provided that no keep or reroll of the specified
	/// rolling record follows it. Drops of a record commute with one another,
	/// but not with its keeps or rerolls, so a drop may combine with a later
	/// one only if no keep or reroll separates them.
	///
	/// # Parameters
	/// - `dest`: The rolling record, relative to the optimized function.
	/// - `filter`: The predicate that the drop instruction must satisfy.
	///
	/// # Returns
	/// The program counter of the drop instruction, if found.
	fn find_drop_instruction(
		&self,
		dest: RollingRecordIndex,
		filter: impl Fn(&Instruction) -> bool
	) -> Option<ProgramCounter>
	{
		for (pc, inst) in self.instructions.iter().enumerate().rev()
		{
			match inst
			{
				Instruction::KeepLowest(KeepLowest { dest: kept, .. })
				| Instruction::KeepHighest(KeepHighest {
					dest: kept, ..
				})
				| Instruction::Reroll(Reroll { dest: kept, .. })
					if *kept == dest =>
				{
					return None
				},
				inst if filter(inst) => return Some(pc.into()),
				_ =>
				{}
			}
		}
		None
	}

	/// Answer an operand whose value is that of the specified operand, or zero
	/// if that is negative: the operand itself, if it is never negative, a
	/// constant, clamped at compile time, or else a new register that a
	/// [maximum](Max) computes.
	///
	/// # Parameters
	/// - `op`: The operand, relative to the optimized function.
	///
	/// # Returns
	/// The clamped operand, relative to the optimized function.
	#[inline]
	fn at_least_zero(&mut self, op: AddressingMode) -> AddressingMode
	{
		self.greater(op, Immediate(0).into())
	}

	/// Begin to track the results that the dice of a roll may show, which are
	/// all of the faces of its die.
	///
	/// # Parameters
	/// - `roll`: The roll, whose rolling record is relative to the original
	///   function, and whose operands are relative to the optimized function.
	fn track_roll(&mut self, roll: impl Into<Instruction>)
	{
		let roll = roll.into();
		let faces = faces(&roll);
		let Some(AddressingMode::RollingRecord(dest)) = roll.destination()
		else
		{
			unreachable!("a roll without a rolling record")
		};
		self.results.insert(
			dest,
			Results {
				live: faces.clone(),
				faces
			}
		);
	}

	/// Replace a roll of dice that all show the same face with its count of
	/// dice, or zero if the count is negative, which rolls nothing. The drops
	/// from the rolling record then subtract from the count, its keeps cap the
	/// count, and its sum multiplies the count by the face.
	///
	/// # Parameters
	/// - `dest`: The rolling record of the roll, relative to the original
	///   function.
	/// - `count`: The count of dice, relative to the original function.
	/// - `face`: The face that every die shows.
	fn reduce_to_count(
		&mut self,
		dest: RollingRecordIndex,
		count: AddressingMode,
		face: i32
	)
	{
		let count = self.at_least_zero(self.replacement(count));
		self.replace(dest, count);
		self.faces.insert(dest, face);
	}

	/// Drop dice from a rolling record that was [reduced to its count of
	/// dice](Self::reduce_to_count). Every die shows the same face, so
	/// whichever dice the drop drops, it subtracts its count, or nothing if
	/// that is negative, from the count of dice, which stays at least zero.
	///
	/// # Parameters
	/// - `dest`: The rolling record, relative to the original function.
	/// - `dropped`: The count of dice to drop, relative to the original
	///   function.
	fn drop_from_count(
		&mut self,
		dest: RollingRecordIndex,
		dropped: AddressingMode
	)
	{
		debug_assert!(
			self.faces.contains_key(&dest),
			"a drop from a value that is not a count of dice"
		);
		let count = self.replacement(dest);
		let dropped = self.at_least_zero(self.replacement(dropped));
		let remaining = match (count, dropped)
		{
			(_, AddressingMode::Immediate(Immediate(0))) => count,
			(
				AddressingMode::Immediate(Immediate(count)),
				AddressingMode::Immediate(Immediate(dropped))
			) => Immediate(count.saturating_sub(dropped).max(0)).into(),
			(count, dropped) =>
			{
				let difference = self.next_register();
				self.bound(
					difference,
					self.operand_bounds(count) - self.operand_bounds(dropped)
				);
				self.emit(Sub {
					dest: difference,
					op1: count,
					op2: dropped
				});
				self.at_least_zero(difference.into())
			}
		};
		self.replace(dest, remaining);
	}

	/// Keep dice of a rolling record that was [reduced to its count of
	/// dice](Self::reduce_to_count). Every die shows the same face, so
	/// whichever dice the keep keeps, it caps the count of dice at its count,
	/// or at zero if that is negative.
	///
	/// # Parameters
	/// - `dest`: The rolling record, relative to the original function.
	/// - `kept`: The count of dice to keep, relative to the original function.
	fn keep_from_count(
		&mut self,
		dest: RollingRecordIndex,
		kept: AddressingMode
	)
	{
		debug_assert!(
			self.faces.contains_key(&dest),
			"a keep from a value that is not a count of dice"
		);
		let count = self.replacement(dest);
		let kept = self.at_least_zero(self.replacement(kept));
		let remaining = self.lesser(count, kept);
		self.replace(dest, remaining);
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

/// The results that the dice of a rolling record may show, as a [strength
/// reducer](StrengthReducer) tracks them, to eliminate the rerolls that cannot
/// match any of them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Results
{
	/// The faces that the die of the record may show, which include every face
	/// that it may have, if its faces are not constant.
	faces: ResultSet,

	/// The results that the dice still kept may show, which are among the
	/// [faces](Self::faces): all of them at first, and fewer after an
	/// unlimited reroll, which leaves only the faces that its predicate
	/// misses. Drops and keeps leave them as they were, since they leave
	/// some of the same dice.
	live: ResultSet
}
