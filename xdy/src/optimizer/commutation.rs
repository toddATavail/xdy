//! # Commutation
//!
//! Commutative operations can reorder their operands without changing the
//! meaning or result. Both addition and multiplication are commutative, so
//! reordering the operands can create additional opportunities for folding
//! constants and reducing strength. This pass reorders the operands of
//! commutative operations so that immediates appear before registers. The
//! constant commuter requires the function to be in static single assignment
//! (SSA) form.

use std::{
	cmp::{max, min},
	collections::{BTreeSet, HashMap}
};

use crate::{
	Add, AddressingMode, CanVisitInstructions, DependencyAnalyzer, Div,
	DropHighest, DropLowest, Exp, Instruction, InstructionVisitor, Mod, Mul,
	Neg, Return, RollCustomDice, RollRange, RollStandardDice, Sub,
	SumRollingRecord
};

////////////////////////////////////////////////////////////////////////////////
//                                Commutation.                                //
////////////////////////////////////////////////////////////////////////////////

/// A commuter that shuffles the immediate operands of commutative instructions
/// to the front in order to simplify constant folding and strength reduction.
/// The standard optimizer applies this pass to a [function](Function) before
/// folding constants or reducing strength.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantCommuter<'inst>
{
	/// The instructions to organize.
	instructions: &'inst [Instruction],

	/// The dependency analyzer.
	analyzer: DependencyAnalyzer<'inst>,

	/// An arrangement of the instructions of a function body into commutative
	/// groups. Each instruction is branded with a group number, and all
	/// instructions sharing the same group number are commutative with respect
	/// to each other.
	groups: HashMap<Instruction, usize>,

	/// The next group number to assign.
	next_group: usize,

	/// Replacement instructions, indexed by original instructions.
	replacements: HashMap<Instruction, Instruction>
}

impl<'inst> ConstantCommuter<'inst>
{
	/// Construct a new instruction organizer.
	///
	/// # Parameters
	/// - `instructions`: The instructions to organize.
	///
	/// # Returns
	/// The new instruction organizer.
	pub fn commute(instructions: &'inst [Instruction]) -> Vec<Instruction>
	{
		let mut commuter = Self {
			instructions,
			analyzer: DependencyAnalyzer::analyze(instructions),
			groups: HashMap::new(),
			next_group: 0,
			replacements: HashMap::new()
		};
		// Visit each instruction in the function body to organize them into
		// commutative groups.
		for instruction in instructions
		{
			instruction.visit(&mut commuter).unwrap();
		}
		// For each commutative group, rewrite the instructions to shuffle their
		// immediate operands to the front of the group. Use the first
		// instruction of a group to classify the type of the group.
		let groups = commuter.groups.values().copied().collect::<BTreeSet<_>>();
		for group in groups
		{
			let representative = commuter
				.instructions
				.iter()
				.find(|inst| commuter.groups.get(inst) == Some(&group))
				.unwrap();
			match representative
			{
				Instruction::DropLowest(_) => commuter
					.rewrite_commutative_group(group, |dest, srcs| {
						DropLowest {
							dest: dest.try_into().unwrap(),
							count: srcs[0]
						}
					}),
				Instruction::DropHighest(_) => commuter
					.rewrite_commutative_group(group, |dest, srcs| {
						DropHighest {
							dest: dest.try_into().unwrap(),
							count: srcs[0]
						}
					}),
				Instruction::Add(_) =>
				{
					commuter.rewrite_commutative_group(group, |dest, srcs| {
						Add {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						}
					})
				},
				Instruction::Sub(_) => commuter
					.rewrite_nearly_commutative_group(
						group,
						|dest, srcs| Add {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						},
						|dest, srcs| Sub {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						}
					),
				Instruction::Mul(_) =>
				{
					commuter.rewrite_commutative_group(group, |dest, srcs| {
						Mul {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						}
					})
				},
				Instruction::Div(_) => commuter
					.rewrite_nearly_commutative_group(
						group,
						|dest, srcs| Mul {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						},
						|dest, srcs| Div {
							dest: dest.try_into().unwrap(),
							op1: srcs[0],
							op2: srcs[1]
						}
					),
				_ =>
				{}
			}
		}
		// Answer the replacement function body.
		instructions
			.iter()
			.map(|inst| commuter.replacements.get(inst).unwrap_or(inst))
			.cloned()
			.collect()
	}
}

impl InstructionVisitor<()> for ConstantCommuter<'_>
{
	fn visit_roll_range(&mut self, _inst: &RollRange) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		_inst: &RollStandardDice
	) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		_inst: &RollCustomDice
	) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_drop_lowest(&mut self, _inst: &DropLowest) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_drop_highest(&mut self, _inst: &DropHighest) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		_inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		self.organize(*inst, |inst| matches!(inst, Instruction::Add(_)))
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		// Subtraction is not commutative, but a chain of subtractions can be
		// rewritten as a single subtraction with the leading operand and the
		// sum of the subtrahends.
		self.organize(*inst, |inst| matches!(inst, Instruction::Sub(_)))
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.organize(*inst, |inst| matches!(inst, Instruction::Mul(_)))
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		// Division is not commutative, but a chain of divisions can be
		// rewritten as a single division with the leading operand and the
		// product of the divisors.
		self.organize(*inst, |inst| matches!(inst, Instruction::Div(_)))
	}

	fn visit_mod(&mut self, _inst: &Mod) -> Result<(), ()> { Ok(()) }

	fn visit_exp(&mut self, _inst: &Exp) -> Result<(), ()> { Ok(()) }

	fn visit_neg(&mut self, _inst: &Neg) -> Result<(), ()> { Ok(()) }

	fn visit_return(&mut self, _inst: &Return) -> Result<(), ()> { Ok(()) }
}

impl ConstantCommuter<'_>
{
	/// Place the specified commutative instruction into a group, using the
	/// supplied filter to recognize other instructions of the same type.
	///
	/// # Parameters
	/// - `inst`: The instruction to organize.
	/// - `filter`: A function that answers `true` if the instruction is of the
	///   same type as the one being organized.
	fn organize(
		&mut self,
		inst: impl Into<Instruction>,
		filter: impl Fn(&Instruction) -> bool
	) -> Result<(), ()>
	{
		let inst = inst.into();
		// Obtain our group, creating it if necessary.
		let group = self.group_id(&inst);
		// Merge each reader's group with ours if the reader is the same type of
		// instruction as us. This reduces the total number of expensive merges.
		for reader in self
			.analyzer
			.readers()
			.get(&inst.destination().unwrap())
			.unwrap()
			.clone()
		{
			let reader = &self.instructions[reader.0];
			if filter(reader)
			{
				let reader_group = self.group_id(&reader.clone());
				self.merge_group_ids(group, reader_group);
			}
		}
		Ok(())
	}

	/// Answer the identifier of the group to which the specified instruction
	/// belongs, creating a new group and branding the instruction if
	/// necessary.
	///
	/// # Parameters
	/// - `inst`: The instruction.
	///
	/// # Returns
	/// The requested group identifier.
	fn group_id(&mut self, inst: &Instruction) -> usize
	{
		match self.groups.get(inst)
		{
			Some(index) => *index,
			None =>
			{
				let index = self.next_group;
				self.groups.insert(inst.clone(), index);
				self.next_group += 1;
				index
			}
		}
	}

	/// Merge two groups of commutative instructions.
	///
	/// # Parameters
	/// - `first`: The first group identifier.
	/// - `second`: The second group identifier.
	fn merge_group_ids(&mut self, first: usize, second: usize)
	{
		if first != second
		{
			// We prefer to keep the lower group number.
			let min = min(first, second);
			let max = max(first, second);
			// For each instruction in the second group, move it to the first
			// group.
			self.groups.iter_mut().for_each(|(_, group)| {
				if *group == max
				{
					*group = min;
				}
			});
			assert!(self.groups.values().all(|&group| group != max));
		}
	}

	/// Answer the instructions in the specified group, sorted by destination
	/// register.
	///
	/// # Parameters
	/// - `group`: The group identifier.
	///
	/// # Returns
	/// The instruction group.
	fn group(&self, group: usize) -> Vec<Instruction>
	{
		let mut instructions = self
			.groups
			.iter()
			.filter_map(|(inst, g)| if *g == group { Some(inst) } else { None })
			.map(|inst| self.replacements.get(inst).unwrap_or(inst).clone())
			.collect::<Vec<_>>();
		instructions.sort_by_key(Instruction::destination);
		instructions
	}

	/// Rewrite the instructions in a commutative group such that the immediate
	/// operands are read first and all registers are read in ascending order.
	/// Hoisting the immediate operands to the front of a chain of commutative
	/// instructions makes them more amenable to other optimizations. Do not
	/// emit the instructions, just populate the replacement map.
	///
	/// # Type Parameters
	/// - `I`: The type of instruction to rewrite.
	///
	/// # Parameters
	/// - `group`: The group of instructions to rewrite.
	/// - `constructor`: A function that constructs a new instruction of the
	///   appropriate type from the destination and source operands of the
	///   original instruction.
	fn rewrite_commutative_group<I>(
		&mut self,
		group: usize,
		constructor: impl Fn(AddressingMode, &[AddressingMode]) -> I
	) where
		I: Into<Instruction>
	{
		// Collect all instructions within the specified commutative group,
		// using any replacements that have already been established. Extract
		// their operands and sort them in descending order, such that
		// immediates are at the front and registers are at the back. Ensure
		// that registers are written before they are read by preserving
		// ascending order. Vectors can only efficiently pop from the back, so
		// we reverse the order of the operands before iterating over them.
		let instructions = self.group(group);
		let mut ops = instructions
			.iter()
			.flat_map(Instruction::sources)
			.collect::<Vec<_>>();
		ops.sort();
		ops.reverse();
		let arity = ops.len() / instructions.len();
		instructions.iter().for_each(|inst| {
			let ops =
				(0..arity).map(|_| ops.pop().unwrap()).collect::<Vec<_>>();
			let new_inst =
				constructor(inst.destination().unwrap(), &ops).into();
			self.replacements.insert((*inst).clone(), new_inst);
		});
	}

	/// Rewrite the instructions in a nearly commutative group such that the
	/// commutative immediate operands are read first and all commutative
	/// registers are read in ascending order. Hoisting the immediate operands
	/// to the front of a chain of nearly commutative instructions makes them
	/// more amenable to other optimizations. Do not emit the instructions, just
	/// populate the replacement map. The first instruction in the group does
	/// not commute with the rest, so we handle it specially and emit it last.
	///
	/// # Type Parameters
	/// - `C`: The type of commutative instruction to rewrite.
	/// - `F`: The type of final instruction to rewrite.
	///
	/// # Parameters
	/// - `group`: The group of instructions to rewrite.
	/// - `commutative_constructor`: A function that constructs a new
	///   commutative instruction of the appropriate type from the destination
	///   and source operands of the original instruction.
	/// - `final_constructor`: A function that constructs a new terminal
	///   instruction of the appropriate type from the destination and source
	///   operands of the original instruction.
	fn rewrite_nearly_commutative_group<C, F>(
		&mut self,
		group: usize,
		commutative_constructor: impl Fn(AddressingMode, &[AddressingMode]) -> C,
		final_constructor: impl Fn(AddressingMode, &[AddressingMode]) -> F
	) where
		C: Into<Instruction>,
		F: Into<Instruction>
	{
		// Collect all instructions within the nearly commutative group. If
		// there's only a single element in the group, then there's nothing to
		// do.
		let mut instructions = self.group(group);
		if instructions.len() > 1
		{
			// Extract all of the destination registers from the complete chain,
			// including the terminal instruction. They are already sorted.
			let mut targets = instructions
				.iter()
				.flat_map(Instruction::destination)
				.collect::<Vec<_>>();
			// Extract all of the operands from the complete chain.
			let mut ops = instructions
				.iter()
				.flat_map(Instruction::sources)
				.collect::<Vec<_>>();
			// The very first operand is not commutative with the rest, so we
			// handle it specially. We'll pop it off the front of the list and
			// re-inject it when we emit the final non-commutative instruction.
			let first_op = ops.remove(0);
			// The next to last operand is the destination of the last
			// commutative instruction. We swap it with the last operand so that
			// we can pop it off the back of the list. We sort the remaining
			// operands, which are all part of the commutative chain.
			let len = ops.len();
			ops.swap(len - 2, len - 1);
			let commutative_result = ops.pop().unwrap();
			ops.sort();
			// The first instruction is non-commutative, and needs to be emitted
			// last. We remove it from the list of instructions, rewrite it with
			// the correct operands, and save it to emit at the end.
			let first_inst = instructions.remove(0);
			let terminal = final_constructor(
				targets.pop().unwrap(),
				&[first_op, commutative_result]
			)
			.into();
			// Reverse the remaining operands and destinations prior to
			// traversing the commutative instructions.
			ops.reverse();
			targets.reverse();
			// Rewrite the commutative instructions in the chain, now that
			// everything is in the correct order for further optimization.
			let arity = ops.len() / instructions.len();
			let mut previous = first_inst.clone();
			instructions.iter().for_each(|inst| {
				let dest = targets.pop().unwrap();
				let ops =
					(0..arity).map(|_| ops.pop().unwrap()).collect::<Vec<_>>();
				let new_inst = commutative_constructor(dest, &ops).into();
				self.replacements.insert(previous.clone(), new_inst);
				previous = inst.clone();
			});
			self.replacements.insert(previous, terminal);
		}
	}
}
