//! # Dice expression histograms
//!
//! Functionality for producing histograms of dice expression outcomes.
//! A histogram is a mapping from the possible outcomes of a dice expression to
//! the number of times that outcome was rolled. Computing a histogram is
//! useful for understanding the distribution of outcomes of a dice expression,
//! but it can be computationally expensive for dice expressions with many
//! ranges, many dice, and/or many-sided dice. When the `rayon` feature is
//! enabled, we use parallelism to speed up histogram computation. Mechanisms
//! are provided to bound the total number of outcomes considered and to
//! control the number of threads used.

#[cfg(feature = "parallel-histogram")]
pub mod parallel;
pub mod serial;

use std::{
	collections::{HashMap, hash_map::IntoIter},
	fmt::{Display, Formatter},
	ops::{Deref, DerefMut, Index},
	sync::atomic::{AtomicU64, Ordering}
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions, Div, DropHighest,
	DropLowest, EvaluationError, Evaluator, Exp, Function, Instruction,
	InstructionVisitor, Mod, Mul, Neg, ProgramCounter, RegisterIndex, Return,
	RollCustomDice, RollRange, RollStandardDice, RollingRecord,
	RollingRecordIndex, RollingRecordKind, Sub, SumRollingRecord, add, div,
	exp, r#mod, mul, neg, sub
};

////////////////////////////////////////////////////////////////////////////////
//                                Histograms.                                 //
////////////////////////////////////////////////////////////////////////////////

/// A histogram of the outcomes of a dice expression, as a map from outcomes to
/// counts. Counts are `u64` to allow for large numbers of outcomes, and
/// protected from overflow only by their enormous size.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Histogram(pub HashMap<i32, u64>);

impl Histogram
{
	/// Answer the number of times the given outcome was rolled.
	///
	/// # Returns
	/// The number of times the given outcome was rolled. If the outcome was
	/// not rolled, this method returns zero.
	pub fn get(&self, outcome: i32) -> u64
	{
		self.0.get(&outcome).copied().unwrap_or(0)
	}

	/// Answer the total number of outcomes in the histogram.
	///
	/// # Returns
	/// The total number of outcomes in the histogram.
	#[inline]
	pub fn total(&self) -> u64 { self.0.values().sum() }

	/// Answer the number of unique outcomes in the histogram.
	///
	/// # Returns
	/// The number of unique outcomes in the histogram.
	#[inline]
	pub fn unique(&self) -> u64 { self.0.len() as u64 }

	/// Answer the outcomes in the histogram, sorted in ascending order.
	///
	/// # Returns
	/// The outcomes in the histogram, sorted in ascending order.
	pub fn outcomes(&self) -> Vec<i32>
	{
		let mut outcomes: Vec<_> = self.0.keys().copied().collect();
		outcomes.sort();
		outcomes
	}

	/// Answers the odds of the given outcome as _`x` to `y` in favor_.
	///
	/// # Parameters
	/// - `outcome`: The outcome whose odds are desired.
	///
	/// # Returns
	/// A tuple containing the number of times the given outcome was rolled and
	/// the number of times it was not rolled.
	pub fn odds(&self, outcome: i32) -> (u64, u64)
	{
		let count = self.get(outcome);
		let total = self.total();
		(count, total - count)
	}

	/// Answers the odds of the given outcome as a percentage.
	///
	/// # Parameters
	/// - `outcome`: The outcome whose odds are desired.
	///
	/// # Returns
	/// The percentage chance of the given outcome.
	pub fn percent_chance(&self, outcome: i32) -> f64
	{
		(self.get(outcome) as f64 / self.total() as f64) * 100.0
	}
}

impl Deref for Histogram
{
	type Target = HashMap<i32, u64>;

	fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for Histogram
{
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl IntoIterator for Histogram
{
	type Item = (i32, u64);
	type IntoIter = IntoIter<i32, u64>;

	fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl Index<i32> for Histogram
{
	type Output = u64;

	fn index(&self, outcome: i32) -> &Self::Output
	{
		self.0.get(&outcome).unwrap_or(&0)
	}
}

impl Display for Histogram
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		let mut outcomes: Vec<_> = self.0.iter().collect();
		outcomes.sort_by_key(|(outcome, _)| *outcome);
		for (outcome, count) in outcomes
		{
			writeln!(f, "{}: {}", outcome, count)?;
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Histogram building.                             //
////////////////////////////////////////////////////////////////////////////////

/// A builder for histograms of the outcomes of dice expressions.
///
/// # Type Parameters
/// - `T`: The type of iterator to use when building the histogram. Deliberately
///   not bounded by `Iterator` to allow for parallel iterators.
pub trait HistogramBuilder<'inst, T>
where
	T: 'inst
{
	/// Constructs a new histogram builder from the given
	/// [evaluator](Evaluator).
	///
	/// # Parameters
	/// - `evaluator`: The evaluator, suitably [configured](Evaluator::bind) to
	///   evaluate the function.
	///
	/// # Returns
	/// A new histogram builder.
	fn new(evaluator: Evaluator) -> Self;

	/// Build the histogram of the outcomes of the dice expression, irrespective
	/// of the number of outcomes. The histogram will contain all possible
	/// outcomes rolled by the dice expression. Highly convenient, but not
	/// recommended for dice expressions with many ranges, many dice, and/or
	/// many-sided dice.
	///
	/// # Parameters
	/// - `args`: The arguments to the dice expression.
	///
	/// # Returns
	/// The histogram of the outcomes of the dice expression.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	fn build(
		&self,
		args: impl IntoIterator<Item = i32> + Send
	) -> Result<Histogram, EvaluationError<'_>>
	{
		self.build_while(args, |_| true)
	}

	/// Build the histogram of the outcomes of the dice expression, stopping
	/// short after `limit` outcomes have been rolled. The histogram will
	/// contain at most `limit` outcomes rolled by the dice expression.
	///
	/// # Parameters
	/// - `args`: The arguments to the dice expression.
	/// - `limit`: The maximum number of outcomes to roll.
	///
	/// # Returns
	/// The histogram of the outcomes of the dice expression.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	fn build_with_limit(
		&self,
		args: impl IntoIterator<Item = i32> + Send,
		limit: u64
	) -> Result<Histogram, EvaluationError<'_>>
	{
		let count = AtomicU64::new(0);
		self.build_while(args, |_| {
			count.fetch_add(1, Ordering::Relaxed) < limit
		})
	}

	/// Build the histogram of the outcomes of the dice expression for as long
	/// as the supplied condition evaluates to `true`. The histogram will
	/// contain possible outcomes of the dice expression, but may be incomplete.
	///
	/// # Parameters
	/// - `args`: The arguments to the dice expression.
	/// - `condition`: The condition that must be satisfied for histogram
	///   construction to continue. Applied to an outcome of the dice
	///   expression.
	///
	/// # Returns
	/// The histogram of the outcomes of the dice expression.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	fn build_while(
		&self,
		args: impl IntoIterator<Item = i32> + Send,
		condition: impl Fn(&i32) -> bool + Send + Sync
	) -> Result<Histogram, EvaluationError<'_>>;

	/// Build an iterator that can be consumed to compute the histogram of the
	/// outcomes of the dice expression. The iterator will compute the histogram
	/// lazily, and may be terminated early for any reason, such as exceeding
	/// the desired maximum number of outcomes or running out of time.
	/// Afterward, the (potentially incomplete) histogram may be retrieved from
	/// the builder.
	///
	/// # Returns
	/// An iterator of outcomes.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	fn iter(
		&'inst self,
		args: impl IntoIterator<Item = i32>
	) -> Result<T, EvaluationError<'inst>>;
}

/// The complete machine state of a histogram builder during execution. Each
/// state serves as a continuation of the machine, and may be used to resume
/// execution. The machine is suspended after every range or die roll, which
/// only occur **inside** [RollRange], [RollStandardDice], and [RollCustomDice]
/// instructions.
#[derive(Debug, Clone)]
pub struct EvaluationState<'inst>
{
	/// The function body.
	instructions: &'inst [Instruction],

	/// The program counter, indicating the current instruction.
	pc: ProgramCounter,

	/// The register bank, containing the current values of each register.
	registers: Vec<i32>,

	/// The rolling records for the dice subexpressions.
	records: Vec<RollingRecord>,

	/// The internal state of a range or roll instruction, expressed as the
	/// current iteration and the next index within the iteration,
	/// respectively.
	substate: (i32, i32),

	/// Whether the evaluation has run out of gas. Stored as separate state to
	/// allow early exits from range and roll expressions.
	out_of_gas: bool,

	/// Successor states to evaluate, sometime after the current state is
	/// evaluated. Only `Some` for states that have resumed. Drained by the
	/// [iterator](EvaluationStateIterator).
	successors: Option<Vec<EvaluationState<'inst>>>,

	/// The final result of the dice expression, if it has been computed.
	pub result: Option<i32>
}

/// Denotes a type that can build a histogram.
///
/// # Type Parameters
/// - `T`: The type of iterator to use when building the histogram. Deliberately
///   not bounded by `Iterator` to allow for parallel iterators.
trait CanBuildHistogram<'inst, T>
where
	T: 'inst
{
	/// Answer the function being evaluated.
	///
	/// # Returns
	/// The function.
	fn function(&self) -> &Function;

	/// Answer the environment in which to evaluate the function, as a map from
	/// external variable indices to values. Missing bindings default to zero.
	///
	/// # Returns
	/// The environment.
	fn environment(&self) -> &HashMap<usize, i32>;

	/// Construct the iterator itself.
	///
	/// # Parameters
	/// - `initial_state`: The initial state of the evaluation machine.
	///
	/// # Returns
	/// An iterator of outcomes.
	fn create_iterator(&'inst self, initial_state: EvaluationState<'inst>)
	-> T;

	/// Answer an iterator that can be consumed to compute the histogram of the
	/// outcomes of the dice expression. The iterator will compute the histogram
	/// lazily, and may be terminated early for any reason, such as exceeding
	/// the desired maximum number of outcomes or running out of time.
	///
	/// # Parameters
	/// - `initial_state`: The initial state of the evaluation machine.
	///
	/// # Returns
	/// An iterator of outcomes.
	fn iter(
		&'inst self,
		args: impl IntoIterator<Item = i32>
	) -> Result<T, EvaluationError<'inst>>
	{
		// Check the argument count.
		let function = self.function();
		let arity = function.arity();
		let args = args.into_iter().collect::<Vec<_>>();
		if args.len() != arity
		{
			return Err(EvaluationError::BadArity {
				expected: arity,
				given: args.len()
			});
		}
		// Initialize the state, which will cause evaluation to begin with the
		// first instruction.
		let mut state = EvaluationState::initial(function);
		// Bind the arguments to their registers.
		for (i, arg) in args.into_iter().enumerate()
		{
			state.registers[i] = arg;
		}
		// Bind the external variables to their registers.
		for (index, value) in self.environment()
		{
			state.registers[arity + *index] = *value;
		}
		// Everything is ready, so answer the iterator. The caller consumes the
		// iterator to build as much of the histogram as desired.
		Ok(self.create_iterator(state))
	}
}

/// Denotes a type that can iterate over the states of a histogram builder.
trait CanIterate<'inst>: Sized
{
	/// Answer the next state to evaluate.
	///
	/// # Returns
	/// The next state to evaluate, or `None` if there are no more states to
	/// evaluate.
	fn next_state(&mut self) -> Option<EvaluationState<'inst>>;

	/// Inject the successors of the given state back into the iterator. Vacates
	/// the state's successor list.
	///
	/// # Parameters
	/// - `state`: The state whose successors should be injected.
	fn inject_successors(&mut self, state: &mut EvaluationState<'inst>);
}

impl<'inst> EvaluationState<'inst>
{
	/// Construct an initial evaluation state with the given number of
	/// registers.
	///
	/// # Parameters
	/// - `registers`: The number of registers to allocate.
	///
	/// # Returns
	/// An initial evaluation state.
	fn initial(function: &'inst Function) -> Self
	{
		EvaluationState {
			instructions: &function.instructions,
			pc: ProgramCounter::default(),
			registers: vec![0; function.register_count],
			records: vec![
				RollingRecord::default();
				function.rolling_record_count
			],
			substate: (0, 0),
			out_of_gas: false,
			successors: None,
			result: None
		}
	}

	/// The maximum number of successor states to generate during evaluation.
	pub(crate) const MAX_SUCCESSORS: usize = 30;

	/// Resume the function. Run the function until it completes or until
	/// [enough](Self::MAX_SUCCESSORS) successor states have been generated.
	///
	/// # Returns
	/// The result of the function, if it has completed. Whether the function
	/// completed or not, it may have generated additional successor states.
	fn evaluate(&mut self) -> Option<i32>
	{
		for inst in &self.instructions[self.pc.0..]
		{
			inst.visit(self).unwrap();
			self.pc.allocate();
			self.substate = (0, 0);
			if self.out_of_gas
			{
				// We ran out of gas, so we must discontinue evaluation. All
				// applicable successor states should have been generated
				// already, so we can simply return.
				return None
			}
		}
		self.result
	}

	/// Obtain the value associated with the specified operand. The operand may
	/// be an immediate or a register, but must not be a rolling record.
	///
	/// # Parameters
	/// - `op`: The operand to evaluate.
	///
	/// # Returns
	/// The value associated with the operand.
	fn value(&self, op: AddressingMode) -> i32
	{
		match op
		{
			AddressingMode::Immediate(value) => value.0,
			AddressingMode::Register(reg) => self.registers[reg.0],
			AddressingMode::RollingRecord(_) => unreachable!()
		}
	}

	/// Set the value of the specified register.
	///
	/// # Parameters
	/// - `reg`: The register to set.
	/// - `value`: The new value of the register.
	#[inline]
	fn set_register(&mut self, reg: RegisterIndex, value: i32)
	{
		self.registers[reg.0] = value;
	}

	/// Obtain the specified rolling record.
	///
	/// # Parameters
	/// - `op`: The target rolling record.
	///
	/// # Returns
	/// The requested rolling record.
	#[inline]
	fn record_mut(&mut self, op: RollingRecordIndex) -> &mut RollingRecord
	{
		&mut self.records[op.0]
	}

	/// Add a successor state to the current state.
	///
	/// # Parameters
	/// - `successor`: The successor state.
	///
	/// # Returns
	/// `true` if there is room for more successors, `false` otherwise. The
	/// accounting always leaves headroom for one more successor, to simplify
	/// the logic of the range and roll instructions.
	fn add_successor(&mut self, successor: EvaluationState<'inst>) -> bool
	{
		match &mut self.successors
		{
			Some(successors) =>
			{
				successors.push(successor);
			},
			_ =>
			{
				self.successors = Some(vec![successor]);
			}
		}
		self.out_of_gas =
			self.successors.as_ref().unwrap().len() >= Self::MAX_SUCCESSORS - 1;
		!self.out_of_gas
	}

	/// Create a new successor state that is a copy of the current state, but
	/// with its own successor states cleared and its gas indicator green lit.
	/// The receiver's result must not be set, and is not cleared.
	///
	/// # Returns
	/// A new successor state.
	fn new_successor(&self) -> Self
	{
		let mut state = self.clone();
		state.successors = None;
		state.out_of_gas = false;
		state
	}

	/// Create a new successor state that is a copy of the current state, but
	/// positioned at the beginning of the specified iteration of the current
	/// range or roll instruction. The successor state has no successor states
	/// of its own, and its gas indicator is green lit. The receiver's result
	/// must not be set, and is not cleared.
	///
	/// # Parameters
	/// - `record`: The rolling record to update.
	/// - `iteration`: The ordinal of the next iteration.
	/// - `max_iteration`: The maximum number of iterations.
	/// - `result`: The result of the current iteration.
	///
	/// # Returns
	/// A new successor state.
	fn new_iteration_successor(
		&self,
		record: RollingRecordIndex,
		iteration: i32,
		max_iterations: i32,
		result: i32
	) -> Self
	{
		let mut successor = self.new_successor();
		if iteration == max_iterations
		{
			// This is the final iteration, so advance to the next
			// instruction and reset the substate.
			successor.pc.allocate();
			successor.substate = (0, 0);
		}
		else
		{
			// Reset the index in order to traverse all values in the next
			// iteration.
			successor.substate = (iteration, 0);
		}
		let record = successor.record_mut(record);
		record.results.push(result);
		successor
	}

	/// Create a new successor state that is a copy of the current state, but
	/// positioned at the beginning of the specified index within the specified
	/// iteration of the current range or roll instruction. The successor state
	/// has no successor states of its own, and its gas indicator is green lit.
	/// The receiver's result must not be set, and is not cleared.
	///
	/// # Parameters
	/// - `iteration`: The ordinal of the current iteration.
	/// - `index`: The current index.
	///
	/// # Returns
	/// A new successor state.
	fn new_index_successor(&self, iteration: i32, index: i32) -> Self
	{
		let mut successor = self.new_successor();
		successor.substate = (iteration, index);
		successor
	}
}

impl Display for EvaluationState<'_>
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		write!(f, "{}{:?}", self.pc, self.substate)?;
		if !self.records.is_empty()
		{
			write!(f, " {{ ")?;
			for (i, record) in self.records.iter().enumerate()
			{
				if i > 0
				{
					write!(f, "; ")?;
				}
				write!(f, "{}", record)?;
			}
			write!(f, " }}")?;
		}
		if let Some(result) = self.result
		{
			write!(f, " -> {}", result)?;
		}
		Ok(())
	}
}

impl InstructionVisitor<()> for EvaluationState<'_>
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		// We continue using the value indicated by the substate, generating a
		// collection of successor states, each of which represents a possible
		// outcome of the range.
		let start = self.value(inst.start);
		let end = self.value(inst.end);
		{
			let record = self.record_mut(inst.dest);
			record.kind = RollingRecordKind::Range { start, end };
		}
		if end < start
		{
			// The range is empty, so the result is zero.
			let record = self.record_mut(inst.dest);
			record.results.push(0);
			return Ok(())
		}
		let (iteration, index) = self.substate;
		if iteration == 0
		{
			for index in index..=end - start
			{
				if !self.add_successor(self.new_iteration_successor(
					inst.dest,
					iteration + 1,
					1,
					start + index
				))
				{
					// We ran out of gas, so add a successor that will resume
					// evaluation from here.
					self.add_successor(
						self.new_index_successor(iteration, index + 1)
					);
					return Ok(())
				}
			}
			// We've generated all the successors. To reduce complexity, we
			// deliberately empty the gas tank here.
			self.out_of_gas = true;
			return Ok(())
		}
		// It is impossible to reach this point, as result-bearing continuations
		// must resume at the next instruction.
		unreachable!()
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		// We continue using the value indicated by the substate, generating a
		// collection of successor states, each of which represents a possible
		// outcome of the range.
		let count = self.value(inst.count);
		let faces = self.value(inst.faces);
		{
			let record = self.record_mut(inst.dest);
			record.kind = RollingRecordKind::Standard { count, faces };
		}
		if count <= 0
		{
			// There are no dice to roll, so the result is a single zero.
			let record = self.record_mut(inst.dest);
			record.results.push(0);
			return Ok(())
		}
		let (iteration, index) = self.substate;
		if iteration < count
		{
			if faces <= 0
			{
				// There are no faces on the die, so the result is a single
				// zero.
				self.add_successor(self.new_iteration_successor(
					inst.dest,
					iteration + 1,
					count,
					0
				));
			}
			// Generate a successor for each possible face of the die.
			for index in index..faces
			{
				if !self.add_successor(self.new_iteration_successor(
					inst.dest,
					iteration + 1,
					count,
					index + 1
				))
				{
					// We ran out of gas, so add a successor that will resume
					// evaluation from here.
					self.add_successor(
						self.new_index_successor(iteration, index + 1)
					);
					return Ok(())
				}
			}
			// We've generated all the successors for this iteration. To reduce
			// complexity, we deliberately empty the gas tank here.
			self.out_of_gas = true;
			return Ok(())
		}
		// It is impossible to reach this point, as result-bearing continuations
		// must resume at the next instruction.
		unreachable!()
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		// We continue using the value indicated by the substate, generating a
		// collection of successor states, each of which represents a possible
		// outcome of the range.
		let count = self.value(inst.count);
		let faces = inst.faces.len();
		{
			let record = self.record_mut(inst.dest);
			record.kind = RollingRecordKind::Custom {
				count,
				faces: inst.faces.clone()
			};
		}
		if count <= 0
		{
			// There are no dice to roll, so the result is a single zero.
			let record = self.record_mut(inst.dest);
			record.results.push(0);
			return Ok(())
		}
		// Note that the value is the index of the face, not the face itself.
		let (iteration, index) = self.substate;
		if iteration < count
		{
			if faces == 0
			{
				// There are no faces on the die, so the result is a single
				// zero.
				self.add_successor(self.new_iteration_successor(
					inst.dest,
					iteration + 1,
					count,
					0
				));
			}
			// This isn't the final iteration, so all we want to do is generate
			// a successor for each possible face value.
			for index in index..faces as i32
			{
				if !self.add_successor(self.new_iteration_successor(
					inst.dest,
					iteration + 1,
					count,
					inst.faces[index as usize]
				))
				{
					// We ran out of gas, so add a successor that will resume
					// evaluation from here.
					self.add_successor(
						self.new_index_successor(iteration, index + 1)
					);
					return Ok(())
				}
			}
			// We've generated all the successors for this iteration. To reduce
			// complexity, we deliberately empty the gas tank here.
			self.out_of_gas = true;
			return Ok(())
		}
		// It is impossible to reach this point, as result-bearing continuations
		// must resume at the next instruction.
		unreachable!()
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		// Don't clamp the count here; we let SumRollingRecord do that so that
		// we don't lose any precision until the last possible moment.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.drop_lowest(count);
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		// Don't clamp the count here; we let SumRollingRecord do that so that
		// we don't lose any precision until the last possible moment.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.drop_highest(count);
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		let sum = self.record_mut(inst.src).sum();
		self.set_register(inst.dest, sum);
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, add(op1, op2));
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, sub(op1, op2));
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, mul(op1, op2));
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, div(op1, op2));
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, r#mod(op1, op2));
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, exp(op1, op2));
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		let op = self.value(inst.op);
		self.set_register(inst.dest, neg(op));
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		self.result = Some(self.value(inst.src));
		Ok(())
	}
}
