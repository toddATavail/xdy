//! # Evaluation
//!
//! Evaluation is the process of determining the value of a dice expression. It
//! requires a compiled [function](Function) and a pseudo-random number
//! generator. Evaluation produces the final result of an expression and the
//! individual outcomes of each dice subexpression.

use std::{
	collections::{HashMap, HashSet},
	error::Error,
	fmt::{Display, Formatter},
	ops::RangeInclusive
};

use rand::Rng;

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _, Div,
	DropHighest, DropLowest, Exp, Function, InstructionVisitor, Mod, Mul, Neg,
	ProgramCounter, RegisterIndex, Return, RollCustomDice, RollRange,
	RollStandardDice, RollingRecordIndex, Sub, SumRollingRecord
};

////////////////////////////////////////////////////////////////////////////////
//                                Evaluation.                                 //
////////////////////////////////////////////////////////////////////////////////

/// A record of a roll of dice. The record includes the results of each die in
/// a set of dice. [`RollStandardDice`] and [`RollCustomDice`] both populate a
/// [`RollingRecord`] in the frame's rolling record set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct RollingRecord
{
	/// The kind of rolling record.
	pub kind: RollingRecordKind<i32>,

	/// The results of each die in the roll.
	pub results: Vec<i32>,

	/// The number of lowest dice dropped, clamped to the size of the result
	/// vector.
	pub lowest_dropped: i32,

	/// The number of highest dice dropped, clamped to the size of the result
	/// vector.
	pub highest_dropped: i32
}

impl Display for RollingRecord
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		// Sort the results, then print them in order. Dropped dice are
		// separated by semicolons.
		let sorted = {
			let mut results = self.results.to_vec();
			results.sort_unstable();
			results
		};
		write!(f, "{}: ", self.kind)?;
		write!(f, "[")?;
		let start = 0;
		let end = self.lowest_dropped as usize;
		for i in start..end
		{
			if i != start
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", sorted[i])?;
			if i + 1 == self.lowest_dropped as usize && end != sorted.len()
			{
				write!(f, "; ")?;
			}
		}
		let start = self.lowest_dropped as usize;
		let end = sorted.len() - self.highest_dropped as usize;
		for i in start..end
		{
			if i != start
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", sorted[i])?;
			if i + 1 == end && end != sorted.len()
			{
				write!(f, "; ")?;
			}
		}
		let start = sorted.len() - self.highest_dropped as usize;
		let end = sorted.len();
		#[allow(clippy::needless_range_loop)]
		for i in start..end
		{
			if i != start
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", sorted[i])?;
		}
		write!(f, "]")
	}
}

/// The kind of rolling record.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum RollingRecordKind<T>
{
	/// An uninitialized rolling record.
	#[default]
	Uninitialized,

	/// A record of a range roll.
	Range
	{
		/// The minimum value of the range.
		start: T,

		/// The maximum value of the range.
		end: T
	},

	/// A record of a standard dice roll.
	Standard
	{
		/// The number of dice rolled.
		count: T,

		/// The number of faces on each die.
		faces: T
	},

	/// A record of a custom dice roll.
	Custom
	{
		/// The number of dice rolled.
		count: T,

		/// The faces of each die.
		faces: Vec<i32>
	}
}

impl<T> Display for RollingRecordKind<T>
where
	T: Display
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			RollingRecordKind::Uninitialized =>
			{
				write!(f, "uninitialized")
			},
			RollingRecordKind::Range { start, end } =>
			{
				write!(f, "[{}:{}]", start, end)
			},
			RollingRecordKind::Standard { count, faces } =>
			{
				write!(f, "{}D{}", count, faces)
			},
			RollingRecordKind::Custom { count, faces } =>
			{
				write!(f, "{}d[", count)?;
				for (i, face) in faces.iter().enumerate()
				{
					if i != 0
					{
						write!(f, ", ")?;
					}
					write!(f, "{}", face)?;
				}
				write!(f, "]")
			}
		}
	}
}

/// A dice expression evaluator. The evaluator uses a client-supplied
/// pseudo-random number to generate the results of dice rolls.
///
/// Underflows and overflows saturate to the minimum and maximum values of the
/// `i32` type, and special arithmetic rules for division, remainder, and
/// exponentation ensure that no undefined behavior occurs. In particular:
///
/// * Division by zero is treated as zero,
/// * Zero to the power of zero is treated as one,
/// * Bases raised to negative exponents are treated as zero.
#[derive(Debug, Clone)]
pub struct Evaluator
{
	/// The function to evaluate.
	pub function: Function,

	/// The environment in which to evaluate the function, as a map from
	/// external variable indices to values. Missing bindings default to zero.
	pub environment: HashMap<usize, i32>
}

/// The complete machine state of an evaluator during evaluation.
#[derive(Debug)]
struct EvaluatorState<'r, R>
where
	R: Rng + ?Sized
{
	/// The pseudo-random number generator used to generate dice rolls.
	rng: &'r mut R,

	/// The program counter, indicating the current instruction.
	pc: ProgramCounter,

	/// The register bank, containing the current values of each register.
	registers: Vec<i32>,

	/// The rolling records for the dice subexpressions.
	records: Vec<RollingRecord>,

	/// The result register, written by a [Return] instruction.
	result: i32
}

/// The result of a dice expression evaluation. The result includes the final
/// value of the expression and the rolling records of the dice subexpressions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Evaluation
{
	/// The result of the entire dice expression.
	pub result: i32,

	/// The rolling records for the dice subexpressions.
	pub records: Vec<RollingRecord>
}

impl<R> Display for EvaluatorState<'_, R>
where
	R: Rng + ?Sized
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		writeln!(f, "evaluator:\n\tpc: {}", self.pc)?;
		if !self.registers.is_empty()
		{
			writeln!(f, "\tregisters:")?;
			for (i, register) in self.registers.iter().enumerate()
			{
				writeln!(f, "\t\t@{} = {}", i, register)?;
			}
		}
		if !self.records.is_empty()
		{
			writeln!(f, "\trecords:")?;
			for (i, record) in self.records.iter().enumerate()
			{
				writeln!(f, "\t\t⚅{} = {}", i, record)?;
			}
		}
		Ok(())
	}
}

impl Display for Evaluation
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{}", self.result)?;
		if !self.records.is_empty()
		{
			write!(f, ": (")?;
			for (i, record) in self.records.iter().enumerate()
			{
				if i != 0
				{
					write!(f, ", ")?;
				}
				write!(f, "{}", record)?;
			}
			write!(f, ")")?;
		}
		Ok(())
	}
}

impl<'r, R> From<EvaluatorState<'r, R>> for Evaluation
where
	R: Rng + ?Sized
{
	fn from(value: EvaluatorState<'r, R>) -> Self
	{
		Self {
			result: value.result,
			records: value.records
		}
	}
}

impl Evaluator
{
	/// Construct an evaluator for the given function.
	///
	/// # Parameters
	/// - `function`: The function to evaluate.
	///
	/// # Returns
	/// The constructed evaluator.
	pub fn new(function: Function) -> Self
	{
		Self {
			function,
			environment: HashMap::new()
		}
	}

	/// Bind an external variable to a value. The external variable must be
	/// declared in the function's signature.
	///
	/// # Parameters
	/// - `name`: The name of the target external variable.
	/// - `value`: The value to bind to the external variable.
	///
	/// # Errors
	/// [`UnrecognizedExternal`](EvaluationError::UnrecognizedExternal) if the
	/// alleged external variable is unrecognized.
	pub fn bind<'s>(
		&mut self,
		name: &'s str,
		value: i32
	) -> Result<(), EvaluationError<'s>>
	{
		let index = self
			.function
			.externals
			.iter()
			.enumerate()
			.find_map(
				|(index, variable)| {
					if variable == name { Some(index) } else { None }
				}
			)
			.ok_or(EvaluationError::UnrecognizedExternal(name))?;
		self.environment.insert(index, value);
		Ok(())
	}

	/// Evaluate the function using the given arguments and pseudo-random number
	/// generator (pRNG). Be sure to bind all external variables, and seed the
	/// pRNG if desired, before calling this method. Missing external variables
	/// default to zero during evaluation, but all parameters myst be bound.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	/// - `rng`: The pseudo-random number generator to use for range and dice
	///   rolls.
	///
	/// # Returns
	/// The result of the evaluation.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	pub fn evaluate<'s, 'r, R>(
		&'s mut self,
		args: impl IntoIterator<Item = i32>,
		rng: &'r mut R
	) -> Result<Evaluation, EvaluationError<'s>>
	where
		R: Rng + ?Sized,
		'r: 's
	{
		// Check the argument count.
		let arity = self.function.arity();
		let args = args.into_iter().collect::<Vec<_>>();
		if args.len() != arity
		{
			return Err(EvaluationError::BadArity {
				expected: arity,
				given: args.len()
			});
		}
		// Create the initial machine state for the evaluator, reserving enough
		// registers for the arguments, external variables, and locals, and
		// reserving enough rolling records for the range and dice instructions.
		let mut state = EvaluatorState::new(
			rng,
			self.function.register_count,
			self.function.rolling_record_count
		);
		// Bind the arguments to their registers.
		for (i, arg) in args.into_iter().enumerate()
		{
			state.registers[i] = arg;
		}
		// Bind the external variables to their registers.
		for (index, value) in &self.environment
		{
			state.registers[arity + *index] = *value;
		}
		for instruction in &self.function.instructions
		{
			instruction.visit(&mut state).unwrap();
			state.pc.allocate();
		}
		Ok(state.into())
	}
}

impl<R> InstructionVisitor<()> for EvaluatorState<'_, R>
where
	R: Rng + ?Sized
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		let start = self.value(inst.start);
		let end = self.value(inst.end);
		let record = roll_range(self.rng, start..=end);
		*self.record_mut(inst.dest) = record;
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let faces = self.value(inst.faces);
		let record = roll_standard_dice(self.rng, count, faces);
		*self.record_mut(inst.dest) = record;
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let record = roll_custom_dice(self.rng, count, inst.faces.clone());
		*self.record_mut(inst.dest) = record;
		Ok(())
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
		self.result = self.value(inst.src);
		Ok(())
	}
}

impl<'r, R> EvaluatorState<'r, R>
where
	R: Rng + ?Sized
{
	/// Construct a new evaluator state.
	///
	/// # Parameters
	/// - `rng`: The random number generator to use.
	/// - `registers`: The number of registers to allocate.
	/// - `records`: The number of rolling records to allocate.
	///
	/// # Returns
	/// A fresh evaluator state.
	fn new(rng: &'r mut R, registers: usize, records: usize) -> Self
	{
		Self {
			rng,
			pc: ProgramCounter::default(),
			registers: vec![0; registers],
			records: vec![RollingRecord::default(); records],
			result: 0
		}
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
}

impl RollingRecordKind<i32>
{
	/// Obtain the number of dice in the rolling record. Applicable for all
	/// initialized rolling records.
	///
	/// # Returns
	/// The count of dice in the rolling record, or `None` if the rolling record
	/// hasn't been initialized yet.
	pub fn count(&self) -> Option<i32>
	{
		match self
		{
			RollingRecordKind::Range { .. } => Some(1),
			RollingRecordKind::Standard { count, .. } => Some(*count),
			RollingRecordKind::Custom { count, .. } => Some(*count),
			_ => None
		}
	}
}

/// An error that may occur during the evaluation of a dice expression. Note
/// that evaluation itself never causes an error, but setup may fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationError<'error>
{
	/// The number of arguments provided to a function does not match the number
	/// of parameters expected.
	BadArity
	{
		/// The number of parameters expected by the function.
		expected: usize,

		/// The number of arguments provided to the function.
		given: usize
	},

	/// An external variable was not recognized.
	UnrecognizedExternal(&'error str)
}

impl Display for EvaluationError<'_>
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		match self
		{
			EvaluationError::BadArity { expected, given } =>
			{
				write!(
					f,
					"expected {} arguments but received {}",
					expected, given
				)
			},
			EvaluationError::UnrecognizedExternal(name) =>
			{
				write!(f, "unrecognized external variable: {}", name)
			}
		}
	}
}

impl Error for EvaluationError<'_> {}

////////////////////////////////////////////////////////////////////////////////
//                            Bounds calculation.                             //
////////////////////////////////////////////////////////////////////////////////

impl Evaluator
{
	/// Compute the bounds of the function for the specified arguments and
	/// external variables.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	///
	/// # Returns
	/// The bounds of the function, covering both the value and the number of
	/// possible outcomes. The possible outcome count is `None` if the function
	/// contains dynamic range or roll expressions, i.e., range or roll
	/// expressions whose count or faces are themselves determined by range or
	/// roll expressions. In such cases, the outcome count cannot be determined
	/// without generating a complete histogram of the function, which is too
	/// expensive to do without a specific request.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	#[inline]
	pub fn bounds(
		&self,
		args: impl IntoIterator<Item = i32>
	) -> Result<Bounds, EvaluationError<'_>>
	{
		BoundsEvaluator::new(&self.function, &self.environment).evaluate(args)
	}
}

/// A bounds evaluator determines the minimum and maximum values of a dice
/// expression, as well as the count of total outcomes _except_ when there are
/// dynamic range or roll expressions. Both are static analyses that do not
/// require any random number generation.
#[derive(Debug, Clone)]
struct BoundsEvaluator<'eval>
{
	/// The function for which to calculate bounds.
	function: &'eval Function,

	/// The environment in which to evaluate the bounds of the function, as a
	/// map from external variable indices to values. Missing bindings default
	/// to zero.
	environment: &'eval HashMap<usize, i32>,

	/// The current program counter.
	pc: ProgramCounter,

	/// The bounds of the registers.
	registers: Vec<EvaluationBounds>,

	/// Flag registers that track whether the corresponding register was
	/// produced by a [SumRollingRecord] operation. Range and roll instructions
	/// read these registers to determine whether to disable outcome counting.
	sums: Vec<bool>,

	/// The bounds of the rolling records.
	records: Vec<RollingRecordBounds>,

	/// The special register holding the final bounds of the evaluation.
	result: EvaluationBounds,

	/// The special register holding the number of outcomes. Holds `None` if
	/// the number of outcomes could not be determined because of dynamic range
	/// or dice expressions.
	count: Option<u128>
}

/// The bounds of a function evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvaluationBounds
{
	/// The minimum value of the function.
	pub min: i32,

	/// The maximum value of the function.
	pub max: i32
}

impl EvaluationBounds
{
	/// Determine whether the specified value is contained within the bounds.
	///
	/// # Parameters
	/// - `x`: The value to check.
	///
	/// # Returns
	/// `true` if the value is contained within the bounds, and `false`
	/// otherwise.
	#[inline]
	pub fn contains(self, x: i32) -> bool { self.min <= x && x <= self.max }
}

impl Display for EvaluationBounds
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		write!(f, "{}, {}", self.min, self.max)
	}
}

impl From<i32> for EvaluationBounds
{
	fn from(value: i32) -> Self
	{
		Self {
			min: value,
			max: value
		}
	}
}

impl From<(i32, i32)> for EvaluationBounds
{
	fn from((min, max): (i32, i32)) -> Self { Self { min, max } }
}

impl<'eval> From<BoundsEvaluator<'eval>> for EvaluationBounds
{
	fn from(eval: BoundsEvaluator<'eval>) -> Self { eval.result }
}

impl From<EvaluationBounds> for (i32, i32)
{
	fn from(bounds: EvaluationBounds) -> Self { (bounds.min, bounds.max) }
}

impl From<EvaluationBounds> for RangeInclusive<i32>
{
	fn from(bounds: EvaluationBounds) -> Self { bounds.min..=bounds.max }
}

/// The bounds of a rolling record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RollingRecordBounds
{
	/// The kind of rolling record.
	kind: RollingRecordKind<EvaluationBounds>,

	/// The bounds on the number of lowest dice dropped.
	lowest_dropped: EvaluationBounds,

	/// The bounds on the number of highest dice dropped.
	highest_dropped: EvaluationBounds
}

impl RollingRecordKind<EvaluationBounds>
{
	/// Obtain the number of dice in the rolling record. Applicable for all
	/// initialized rolling records.
	///
	/// # Returns
	/// The count of dice in the rolling record, or `None` if the rolling record
	/// hasn't been initialized yet.
	pub fn count(&self) -> Option<EvaluationBounds>
	{
		match self
		{
			RollingRecordKind::Range { .. } => Some(1.into()),
			RollingRecordKind::Standard { count, .. } => Some(*count),
			RollingRecordKind::Custom { count, .. } => Some(*count),
			_ => None
		}
	}
}

/// The bounds of a function, as computed by a
/// [bounds evaluator](BoundsEvaluator).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bounds
{
	/// The minimum and maximum values of the function.
	pub value: EvaluationBounds,

	/// The number of outcomes, or `None` if the count could not be computed
	/// efficiently because of dynamic range or dice expressions, i.e., when
	/// the count or faces are themselves determined by a range or dice
	/// expression.
	pub count: Option<u128>
}

impl Display for Bounds
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		write!(
			f,
			"value ∈ [{}], count = {}",
			self.value,
			match self.count
			{
				Some(count) => count.to_string(),
				None => "None".to_string()
			}
		)
	}
}

impl<'eval> From<BoundsEvaluator<'eval>> for Bounds
{
	fn from(eval: BoundsEvaluator<'eval>) -> Self
	{
		Self {
			value: eval.result,
			count: eval.count
		}
	}
}

impl<'eval> BoundsEvaluator<'eval>
{
	/// Construct a new bounds evaluator for the specified function.
	///
	/// # Parameters
	/// - `function`: The function for which to calculate bounds.
	///
	/// # Returns
	/// The requested bounds evaluator.
	fn new(
		function: &'eval Function,
		environment: &'eval HashMap<usize, i32>
	) -> Self
	{
		Self {
			function,
			environment,
			pc: ProgramCounter::default(),
			registers: vec![
				EvaluationBounds::default();
				function.register_count
			],
			sums: vec![false; function.register_count],
			records: vec![
				RollingRecordBounds::default();
				function.rolling_record_count
			],
			result: EvaluationBounds::default(),
			count: Some(1)
		}
	}

	/// Evaluate the bounds of the function using the given arguments. Missing
	/// external variables default to zero during evaluation, but all parameters
	/// myst be bound.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	///
	/// # Returns
	/// The bounds of the function, covering both the value and the number of
	/// possible outcomes.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	pub fn evaluate(
		mut self,
		args: impl IntoIterator<Item = i32>
	) -> Result<Bounds, EvaluationError<'eval>>
	{
		// Check the argument count.
		let arity = self.function.arity();
		let args = args.into_iter().collect::<Vec<_>>();
		if args.len() != arity
		{
			return Err(EvaluationError::BadArity {
				expected: arity,
				given: args.len()
			});
		}
		// Bind the arguments to their registers.
		for (i, arg) in args.into_iter().enumerate()
		{
			self.registers[i] = (arg, arg).into();
		}
		// Bind the external variables to their registers.
		for (index, value) in self.environment
		{
			self.registers[arity + *index] = (*value, *value).into();
		}
		for instruction in &self.function.instructions
		{
			instruction.visit(&mut self).unwrap();
			self.pc.allocate();
		}
		Ok(self.into())
	}

	/// Obtain the value associated with the specified operand. The operand may
	/// be an immediate or a register, but must not be a rolling record.
	///
	/// # Parameters
	/// - `op`: The operand to evaluate.
	///
	/// # Returns
	/// The value associated with the operand.
	fn value(&self, op: AddressingMode) -> EvaluationBounds
	{
		match op
		{
			AddressingMode::Immediate(value) => value.0.into(),
			AddressingMode::Register(reg) => self.registers[reg.0],
			AddressingMode::RollingRecord(_) => unreachable!()
		}
	}

	/// Obtain the summation flag associated with the specified operand. The
	/// operand may be an immediate or a register, but must not be a rolling
	/// record.
	///
	/// # Parameters
	/// - `op`: The operand whose count is desired.
	///
	/// # Returns
	/// The summation flag.
	fn sum(&self, op: AddressingMode) -> bool
	{
		match op
		{
			AddressingMode::Immediate(_) => false,
			AddressingMode::Register(reg) => self.sums[reg.0],
			AddressingMode::RollingRecord(_) => unreachable!()
		}
	}

	/// Set the value of the specified register.
	///
	/// # Parameters
	/// - `index`: The index of the register to set.
	/// - `value`: The new value of the register.
	#[inline]
	fn set_register(
		&mut self,
		index: impl Into<usize>,
		value: impl Into<EvaluationBounds>
	)
	{
		self.registers[index.into()] = value.into();
	}

	/// Set the summation flag of the specified register.
	///
	/// # Parameters
	/// - `index`: The index of the counting register to set.
	/// - `flag`: The new value of the register.
	#[inline]
	fn set_sum(&mut self, index: impl Into<usize>, flag: bool)
	{
		self.sums[index.into()] = flag;
	}

	/// Obtain the specified rolling record bounds.
	///
	/// # Parameters
	/// - `op`: The target rolling record bounds.
	///
	/// # Returns
	/// The requested rolling record bounds.
	#[inline]
	fn record(&self, op: RollingRecordIndex) -> &RollingRecordBounds
	{
		&self.records[op.0]
	}

	/// Obtain the specified rolling record bounds.
	///
	/// # Parameters
	/// - `op`: The target rolling record bounds.
	///
	/// # Returns
	/// The requested rolling record bounds.
	#[inline]
	fn record_mut(&mut self, op: RollingRecordIndex)
	-> &mut RollingRecordBounds
	{
		&mut self.records[op.0]
	}
}

impl InstructionVisitor<()> for BoundsEvaluator<'_>
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		let start = self.value(inst.start);
		let end = self.value(inst.end);
		let record = self.record_mut(inst.dest);
		record.kind = RollingRecordKind::Range {
			start: (start.min, start.max).into(),
			end: (end.min, end.max).into()
		};
		if self.sum(inst.start) || self.sum(inst.end)
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let faces = self.value(inst.faces);
		let record = self.record_mut(inst.dest);
		record.kind = RollingRecordKind::Standard {
			count: (count.min, count.max).into(),
			faces: (faces.min, faces.max).into()
		};
		if self.sum(inst.count) || self.sum(inst.faces)
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let mut faces = inst.faces.clone();
		faces.sort();
		let record = self.record_mut(inst.dest);
		record.kind = RollingRecordKind::Custom {
			count: (count.min, count.max).into(),
			faces
		};
		if self.sum(inst.count)
		{
			// The count depends on a previous range or roll, so disable
			// counting for the remainder of the function.
			self.count = None;
		}
		Ok(())
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.lowest_dropped += count;
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.highest_dropped += count;
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		let record = self.record(inst.src);
		let count = record.kind.count().unwrap();
		let count: EvaluationBounds =
			(count.min.max(0), count.max.max(0)).into();
		let lowest_dropped = record.lowest_dropped;
		let highest_dropped = record.highest_dropped;
		let kept =
			(count - lowest_dropped - highest_dropped).clamp(0.into(), count);
		let (sum, outcomes) = match record.kind
		{
			RollingRecordKind::Uninitialized => unreachable!(),
			RollingRecordKind::Range { start, end } =>
			{
				match end.max < start.min
				{
					true =>
					{
						// Take care to normalize an empty range to 0.
						(0.into(), Some(1))
					},
					false => (
						match start.max > end.min
						{
							false => kept * (start.min, end.max).into(),
							true =>
							{
								// The upper bound of the start is less than
								// the lower bound of the end, so there's an
								// overlap. Overlap can lead to degenerate
								// ranges, so make sure to include 0 in the
								// bounds.
								let range: EvaluationBounds =
									(start.min, end.max).into();
								kept * range.union(Some(0.into())).unwrap()
							}
						},
						self.count.map(|c| {
							c.saturating_mul(
								(end.max as i128)
									.saturating_sub(start.min as i128)
									.saturating_add(1)
									.max(1) as u128
							)
							.max(1)
						})
					)
				}
			},
			RollingRecordKind::Standard {
				count: dice_count,
				faces
			} =>
			{
				// Faces might be negative. Each standard die represents a range
				// of faces in `[1, faces]`. Given both facts, clamp the minimum
				// to `[0, 1]`.
				(
					kept * (faces.min.clamp(0, 1), faces.max.max(0)).into(),
					self.count.map(|c| {
						c.saturating_mul(
							(faces.max.max(0) as u128)
								.saturating_pow(dice_count.max.max(0) as u32)
								.max(1)
						)
						.max(1)
					})
				)
			},
			RollingRecordKind::Custom {
				count: dice_count,
				ref faces
			} =>
			{
				// Oddly, this is simplest, since we can just grab the edges of
				// the sorted faces.
				(
					kept * (faces[0], faces[faces.len() - 1]).into(),
					self.count.map(|c| {
						c.saturating_mul(
							(faces.len() as u128)
								.saturating_pow(dice_count.max.max(0) as u32)
								.max(1)
						)
						.max(1)
					})
				)
			}
		};
		self.set_register(inst.dest, sum);
		self.count = outcomes;
		// Mark the register as having been produced by a sum operation. We have
		// to give up counting if the operands of range or roll expressions are
		// dynamic.
		self.set_sum(inst.dest, true);
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1 + op2);
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1 - op2);
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1 * op2);
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1 / op2);
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1 % op2);
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1.exp(op2));
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		let op = self.value(inst.op);
		self.set_register(inst.dest, -op);
		self.set_sum(inst.dest, self.sum(inst.op));
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		self.result = self.value(inst.src);
		Ok(())
	}
}

impl std::ops::Add for EvaluationBounds
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output
	{
		Self {
			min: add(self.min, rhs.min),
			max: add(self.max, rhs.max)
		}
	}
}

impl std::ops::AddAssign for EvaluationBounds
{
	fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl std::ops::Sub for EvaluationBounds
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output
	{
		Self {
			min: sub(self.min, rhs.max),
			max: sub(self.max, rhs.min)
		}
	}
}

impl std::ops::SubAssign for EvaluationBounds
{
	fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl std::ops::Mul for EvaluationBounds
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output
	{
		// Compute the four possible products, based on the signs of the
		// operands.
		let min_min = mul(self.min, rhs.min);
		let min_max = mul(self.min, rhs.max);
		let max_min = mul(self.max, rhs.min);
		let max_max = mul(self.max, rhs.max);
		// Now find the minimum and maximum among them.
		let min = min_min.min(min_max).min(max_min).min(max_max);
		let max = min_min.max(min_max).max(max_min).max(max_max);
		Self { min, max }
	}
}

impl std::ops::MulAssign for EvaluationBounds
{
	fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl std::ops::Div for EvaluationBounds
{
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output
	{
		// Division is complex because it contains a singularity at zero.
		match rhs
		{
			EvaluationBounds { min: 0, max: 0 } => Self { min: 0, max: 0 },
			rhs if rhs.contains(0) =>
			{
				// The divisor contains zero, so we must split the divisor into
				// positive and negative parts, compute the bounds for each
				// part, and throw zero back into the mix at the end.
				let mut ops = Vec::new();
				if rhs.min < 0
				{
					ops.push(self / (rhs.min, -1).into());
				}
				if rhs.max > 0
				{
					ops.push(self / (1, rhs.max).into());
				}
				ops.push(self / 0.into());
				let min = ops.iter().map(|b| b.min).min().unwrap();
				let max = ops.iter().map(|b| b.max).max().unwrap();
				Self { min, max }
			},
			rhs =>
			{
				let min_min = div(self.min, rhs.min);
				let min_max = div(self.min, rhs.max);
				let max_min = div(self.max, rhs.min);
				let max_max = div(self.max, rhs.max);
				let min = min_min.min(min_max).min(max_min).min(max_max);
				let max = min_min.max(min_max).max(max_min).max(max_max);
				Self { min, max }
			}
		}
	}
}

impl std::ops::DivAssign for EvaluationBounds
{
	fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl std::ops::Rem for EvaluationBounds
{
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output
	{
		// Remainder is quite complex, especially because of the singularity at
		// zero and the saturation semantics.
		match (self.into(), rhs.into())
		{
			((_, _), (0, 0)) => 0.into(),
			((op1_min, op1_max), (op2_min, op2_max))
				if op2_min == op2_max
					&& op1_min.saturating_div(op2_min)
						== op1_max.saturating_div(op2_min) =>
			{
				// The divisor is known exactly, and the quotients are the same,
				// so we can tighten the bounds by performing the remainder
				// operation and sorting the results.
				let rem1 = r#mod(op1_min, op2_min);
				let rem2 = r#mod(op1_max, op2_min);
				(rem1.min(rem2), rem1.max(rem2)).into()
			},
			((_, ..0), (op2_min @ ..0, _)) =>
			{
				// The dividend is negative, so the remainder is negative. The
				// divisor is negative, so we just bump the minimum up by one.
				(op2_min + 1, 0).into()
			},
			((_, ..0), (op2_min, _)) =>
			{
				// The dividend is negative, so the remainder is negative.
				(-op2_min + 1, 0).into()
			},
			((0.., _), (_, op2_max @ 0..)) =>
			{
				// The dividend is positive, so the remainder is positive.
				(0, (op2_max - 1).max(0)).into()
			},
			((0.., _), (_, op2_max)) =>
			{
				// The dividend is positive, so the remainder is positive.
				(0, (op2_max.saturating_neg() - 1).max(0)).into()
			},
			((_, _), (op2_min, op2_max)) if op2_min == op2_max =>
			{
				// The dividend could be negative, zero, or positive, so we can
				// only bound the result by the divisor.
				let abs = op2_min.saturating_abs();
				(-abs + 1, abs - 1).into()
			},
			((_, _), (op2_min, op2_max)) =>
			{
				// The dividend could be negative, zero, or positive, so we can
				// only bound the result by the divisor.
				let op2_min = op2_min.saturating_abs();
				let op2_max = op2_max.saturating_abs();
				let abs_min = op2_min.min(op2_max);
				let abs_max = op2_min.max(op2_max);
				(-abs_min, abs_max - 1).into()
			}
		}
	}
}

impl std::ops::RemAssign for EvaluationBounds
{
	fn rem_assign(&mut self, rhs: Self) { *self = *self % rhs; }
}

impl std::ops::Neg for EvaluationBounds
{
	type Output = Self;

	fn neg(self) -> Self::Output
	{
		Self {
			min: neg(self.max),
			max: neg(self.min)
		}
	}
}

impl EvaluationBounds
{
	/// Clamps the bounds to the given range.
	///
	/// # Parameters
	/// - `min`: The minimum bound.
	/// - `max`: The maximum bound.
	///
	/// # Returns
	/// The clamped bounds.
	fn clamp(self, min: EvaluationBounds, max: EvaluationBounds) -> Self
	{
		Self {
			min: self.min.clamp(min.min, max.min),
			max: self.max.clamp(min.max, max.max)
		}
	}

	/// Compute the bounds of exponentiation for two bounds.
	///
	/// # Parameters
	/// - `rhs`: The bounds of the exponent.
	///
	/// # Returns
	/// The bounds of the result.
	fn exp(self, rhs: EvaluationBounds) -> Self
	{
		let interesting_bases = HashSet::from([self.min, -1, 0, 1, self.max]);
		let interesting_powers = HashSet::from([
			rhs.min,
			rhs.min.saturating_add(1),
			(rhs.max.saturating_sub(1)).max(0),
			rhs.max
		]);
		let mut result = None;
		for base in interesting_bases
		{
			if self.contains(base)
			{
				for power in interesting_powers.clone()
				{
					if rhs.contains(power)
					{
						let value: EvaluationBounds = exp(base, power).into();
						result = value.union(result);
					}
				}
			}
		}
		result.unwrap()
	}

	/// Compute the union of the receiver and the specified bounds.
	///
	/// # Parameters
	/// - `rhs`: Another set of bounds, where `None` represents the bottom type
	///   (i.e., an empty set of bounds).
	///
	/// # Returns
	/// The union of the two sets of bounds. Always returns `Some`, as bottom
	/// type is an identity element for the union operation.
	#[inline]
	fn union(self, rhs: Option<Self>) -> Option<Self>
	{
		match rhs
		{
			Some(rhs) => Some(Self {
				min: self.min.min(rhs.min),
				max: self.max.max(rhs.max)
			}),
			None => Some(self)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Primitives.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Generate a value within the specified range, using the same algorithm as
/// the [evaluator](Evaluator). If the range is empty (i.e., `end<start`), then
/// the result is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `range`: The range of values to generate.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a value within the range `1..=6`:
///
/// ```rust
/// # use core::ops::RangeInclusive;
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_range};
/// let rolling_record = roll_range(&mut thread_rng(), 1..=6);
/// assert_eq!(rolling_record.results.len(), 1);
/// assert!(rolling_record.results[0] >= 1 && rolling_record.results[0] <= 6);
/// ```
///
/// Use an empty range:
/// ```rust
/// # use core::ops::RangeInclusive;
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_range};
/// let rolling_record = roll_range(&mut thread_rng(), 6..=1);
/// assert_eq!(rolling_record.results.len(), 1);
/// assert_eq!(rolling_record.results[0], 0);
/// ```
pub fn roll_range<R>(rng: &mut R, range: RangeInclusive<i32>) -> RollingRecord
where
	R: Rng + ?Sized
{
	let (start, end) = range.into_inner();
	let result = match end < start
	{
		false => rng.gen_range(start..=end),
		true => 0
	};
	RollingRecord {
		kind: RollingRecordKind::Range { start, end },
		results: vec![result],
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

/// Generate a standard dice roll, using the same algorithm as the
/// [evaluator](Evaluator). If `count≤0`, then no dice are rolled. If `faces≤0`,
/// then the result of each die is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `count`: The number of dice to roll.
/// - `faces`: The number of faces on each die, numbered from 1 to `faces`.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a roll of 3 six-sided dice (i.e., `3D6`):
///
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut thread_rng(), 3, 6);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///     assert!(result >= 1 && result <= 6);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut thread_rng(), 0, 6);
/// assert_eq!(rolling_record.results.len(), 0);
///
/// let rolling_record = roll_standard_dice(&mut thread_rng(), -1, 6);
/// assert_eq!(rolling_record.results.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_standard_dice};
/// let rolling_record = roll_standard_dice(&mut thread_rng(), 3, 0);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///    assert_eq!(result, 0);
/// });
///
/// let rolling_record = roll_standard_dice(&mut thread_rng(), 3, -1);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///   assert_eq!(result, 0);
/// });
/// ```
pub fn roll_standard_dice<R>(
	rng: &mut R,
	count: i32,
	faces: i32
) -> RollingRecord
where
	R: Rng + ?Sized
{
	let results = (0..count)
		.map(|_| match faces <= 0
		{
			false => rng.gen_range(1..=faces),
			true => 0
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Standard { count, faces },
		results,
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

/// Generate a custom dice roll, using the same algorithm as the
/// [evaluator](Evaluator). If `count≤0`, then no dice are rolled. If `faces`
/// is empty, then the result of each die is `0`.
///
/// # Parameters
/// - `rng`: The pseudo-random number generator.
/// - `count`: The number of dice to roll.
/// - `faces`: The faces on each die.
///
/// # Returns
/// The resultant [rolling record](RollingRecord).
///
/// # Examples
/// Generate a roll of 3 dice with faces `-1`, `0`, and `1` (i.e.,
/// `3D[-1,0,1]`):
///
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut thread_rng(), 3, vec![-1, 0, 1]);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///    assert!(result >= -1 && result <= 1);
/// });
/// ```
///
/// Roll no dice:
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut thread_rng(), 0, vec![1, 2, 3]);
/// assert_eq!(rolling_record.results.len(), 0);
///
/// let rolling_record = roll_custom_dice(&mut thread_rng(), -1, vec![1, 2, 3]);
/// assert_eq!(rolling_record.results.len(), 0);
/// ```
///
/// Use dice with no faces:
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use xdy::{RollingRecord, roll_custom_dice};
/// let rolling_record = roll_custom_dice(&mut thread_rng(), 3, vec![]);
/// assert_eq!(rolling_record.results.len(), 3);
/// rolling_record.results.iter().for_each(|&result| {
///   assert_eq!(result, 0);
/// });
/// ```
pub fn roll_custom_dice<R>(
	rng: &mut R,
	count: i32,
	faces: Vec<i32>
) -> RollingRecord
where
	R: Rng + ?Sized
{
	let face_count = faces.len();
	let results = (0..count)
		.map(|_| match faces.is_empty()
		{
			false => faces[rng.gen_range(0..face_count)],
			true => 0
		})
		.collect::<Vec<_>>();
	RollingRecord {
		kind: RollingRecordKind::Custom { count, faces },
		results,
		lowest_dropped: 0,
		highest_dropped: 0
	}
}

impl RollingRecord
{
	/// Mark the lowest `count` results as dropped, clamping the drop count to
	/// the number of results.
	///
	/// # Parameters
	/// - `count`: The number of results to drop.
	///
	/// # Panics
	///
	/// Panics if `count` the receiver is
	/// [uninitialized](RollingRecordKind::Uninitialized).
	///
	/// # Examples
	/// Roll `3D12` and drop the lowest result:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 3, 12);
	/// rolling_record.drop_lowest(1);
	/// assert_eq!(rolling_record.results.len(), 3);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// ```
	///
	/// Roll `3D12`, but try to drop `4` results:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 3, 12);
	/// rolling_record.drop_lowest(4);
	/// assert_eq!(rolling_record.results.len(), 3);
	/// assert_eq!(rolling_record.lowest_dropped, 3);
	/// ```
	pub fn drop_lowest(&mut self, count: i32)
	{
		self.lowest_dropped = self
			.lowest_dropped
			.saturating_add(count)
			.clamp(0, self.results.len() as i32);
	}

	/// Answer the results that have been dropped because they were the lowest.
	///
	/// # Returns
	/// The lowest dropped results, in ascending order.
	pub fn lowest_dropped(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		results
			.iter()
			.take(self.lowest_dropped as usize)
			.copied()
			.collect::<Vec<_>>()
	}

	/// Mark the highest `count` results as dropped, clamping the drop count to
	/// the number of results.
	///
	/// # Parameters
	/// - `count`: The number of results to drop.
	///
	/// # Panics
	///
	/// Panics if `count` the receiver is
	/// [uninitialized](RollingRecordKind::Uninitialized).
	///
	/// # Examples
	/// Roll `5D8` and drop the highest result:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 5, 8);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.results.len(), 5);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// ```
	///
	/// Roll `5D8`, but try to drop `7` results:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 5, 8);
	/// rolling_record.drop_highest(7);
	/// assert_eq!(rolling_record.results.len(), 5);
	/// assert_eq!(rolling_record.highest_dropped, 5);
	/// ```
	pub fn drop_highest(&mut self, count: i32)
	{
		self.highest_dropped = self
			.highest_dropped
			.saturating_add(count)
			.clamp(0, self.results.len() as i32);
	}

	/// Answer the results that have been dropped because they were the highest.
	///
	/// # Returns
	/// The highest dropped results, in ascending order.
	pub fn highest_dropped(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		let lowest_dropped = self.lowest_dropped as usize;
		let highest_dropped = self.highest_dropped as usize;
		let count = results
			.len()
			.saturating_sub(lowest_dropped)
			.saturating_sub(highest_dropped);
		results
			.iter()
			.skip(lowest_dropped)
			.skip(count)
			.copied()
			.collect::<Vec<_>>()
	}

	/// Answer the kept results, dropping the lowest and highest results as
	/// necessary.
	///
	/// # Returns
	/// The kept results, in ascending order.
	///
	/// # Examples
	/// Roll `4D6`, drop the lowest and highest die, leaving the middle two:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 4, 6);
	/// rolling_record.drop_lowest(1);
	/// rolling_record.drop_highest(1);
	/// assert_eq!(rolling_record.lowest_dropped, 1);
	/// assert_eq!(rolling_record.highest_dropped, 1);
	/// let kept = rolling_record.kept();
	/// assert_eq!(kept.len(), 2);
	/// for result in rolling_record.lowest_dropped() {
	///     assert!(result <= kept[0]);
	/// }
	/// for result in rolling_record.highest_dropped() {
	///    assert!(result >= kept[kept.len() - 1]);
	/// }
	/// ```
	pub fn kept(&self) -> Vec<i32>
	{
		let mut results = self.results.to_vec();
		results.sort();
		let lowest_dropped = self.lowest_dropped as usize;
		let highest_dropped = self.highest_dropped as usize;
		let count = results.len();
		let count = count
			.saturating_sub(lowest_dropped)
			.saturating_sub(highest_dropped);
		results
			.iter()
			.skip(lowest_dropped)
			.take(count)
			.copied()
			.collect()
	}

	/// Compute the sum of the results, dropping the lowest and highest results
	/// as necessary. Clamp the drop counts to the range
	/// `[0, self.results.len()]`. Saturate the sum on overflow. Consecutive
	/// calls are idempotent.
	///
	/// # Returns
	/// The sum of the results.
	///
	/// # Examples
	/// Roll `5D12` and sum the results:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 5, 12);
	/// let sum = rolling_record.sum();
	/// assert!(sum >= 5 && sum <= 60);
	/// ```
	///
	/// Roll `4D6`, drop the lowest and highest die, and sum the middle die:
	///
	/// ```rust
	/// # use rand::{Rng, thread_rng};
	/// # use xdy::{RollingRecord, roll_standard_dice};
	/// let mut rolling_record = roll_standard_dice(&mut thread_rng(), 4, 6);
	/// rolling_record.drop_lowest(1);
	/// rolling_record.drop_highest(1);
	/// let sum = rolling_record.sum();
	/// assert!(sum >= 2 && sum <= 12);
	/// ```
	pub fn sum(&mut self) -> i32
	{
		self.kept().iter().fold(0i32, |a, b| a.saturating_add(*b))
	}
}

/// Computes the sum of the two operands, saturating on overflow. This serves as
/// the definition of the addition operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The augend.
/// - `op2`: The addend.
///
/// # Returns
/// The sum of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::add;
/// assert_eq!(add(1, 2), 3);
/// assert_eq!(add(i32::MAX, 1), i32::MAX);
/// assert_eq!(add(i32::MIN, -1), i32::MIN);
/// ```
#[inline]
pub fn add(op1: i32, op2: i32) -> i32 { op1.saturating_add(op2) }

/// Computes the difference of the two operands, saturating on overflow. This
/// serves as the definition of the subtraction operation in the dice expression
/// language.
///
/// # Parameters
/// - `op1`: The minuend.
/// - `op2`: The subtrahend.
///
/// # Returns
/// The difference of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::sub;
/// assert_eq!(sub(1, 2), -1);
/// assert_eq!(sub(i32::MAX, -1), i32::MAX);
/// assert_eq!(sub(i32::MIN, 1), i32::MIN);
/// ```
#[inline]
pub fn sub(op1: i32, op2: i32) -> i32 { op1.saturating_sub(op2) }

/// Computes the product of the two operands, saturating on overflow. This
/// serves as the definition of the multiplication operation in the dice
/// expression language.
///
/// # Parameters
/// - `op1`: The multiplicand.
/// - `op2`: The multiplier.
///
/// # Returns
/// The product of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::mul;
/// assert_eq!(mul(2, 3), 6);
/// assert_eq!(mul(i32::MAX, 2), i32::MAX);
/// assert_eq!(mul(i32::MIN, 2), i32::MIN);
/// assert_eq!(mul(i32::MAX, -1), i32::MIN + 1);
/// ```
#[inline]
pub fn mul(op1: i32, op2: i32) -> i32 { op1.saturating_mul(op2) }

/// Computes the quotient of the first operand divided by the second operand,
/// saturating on overflow. Division by zero is treated as zero. This serves as
/// the definition of the division operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The dividend.
/// - `op2`: The divisor.
///
/// # Returns
/// The quotient of the operands.
///
/// # Examples
/// ```rust
/// # use xdy::div;
/// assert_eq!(div(6, 2), 3);
/// assert_eq!(div(6, 0), 0);
/// assert_eq!(div(i32::MIN, -1), i32::MAX);
/// ```
#[inline]
pub fn div(op1: i32, op2: i32) -> i32
{
	match op2
	{
		0 => 0,
		_ => op1.saturating_div(op2)
	}
}

/// Computes the remainder of the first operand divided by the second operand,
/// saturating on overflow. This serves as the definition of the remainder
/// operation in the dice expression language.
///
/// # Parameters
/// - `op1`: The dividend.
/// - `op2`: The divisor.
///
/// # Returns
/// The remainder of `op1` divided by `op2`.
///
/// # Examples
/// ```rust
/// # use xdy::r#mod;
/// assert_eq!(r#mod(6, 4), 2);
/// assert_eq!(r#mod(6, 0), 0);
/// assert_eq!(r#mod(i32::MIN, -1), 0);
/// ```
pub fn r#mod(op1: i32, op2: i32) -> i32
{
	match (op1, op2)
	{
		(i32::MIN, -1) => 0,
		(_, 0) => 0,
		(_, _) => op1 % op2
	}
}

/// Raises the first operand to the power of the second operand, saturating on
/// overflow. Produces correct answers for base 0 or 1 even for negative
/// expornents, otherwise treats `x^y` when `y<0` as `0`. This serves as the
/// definition of exponentation in the dice expression language.
///
/// # Parameters
/// - `op1`: The base.
/// - `op2`: The exponent.
///
/// # Returns
/// The result of `op1` raised to the power of `op2`.
///
/// # Examples
/// ```rust
/// # use xdy::exp;
/// assert_eq!(exp(2, 3), 8);
/// assert_eq!(exp(2, 0), 1);
/// assert_eq!(exp(0, 0), 1);
/// assert_eq!(exp(0, 1), 0);
/// assert_eq!(exp(0, -1), 0);
/// assert_eq!(exp(1, 0), 1);
/// assert_eq!(exp(1, 1), 1);
/// assert_eq!(exp(1, -1), 1);
/// assert_eq!(exp(-1, 0), 1);
/// assert_eq!(exp(-1, 1), -1);
/// assert_eq!(exp(-1, 2), 1);
/// assert_eq!(exp(-1, 3), -1);
/// assert_eq!(exp(-1, -1), -1);
/// assert_eq!(exp(-1, -2), 1);
/// assert_eq!(exp(-1, -3), -1);
/// assert_eq!(exp(2, -1), 0);
/// assert_eq!(exp(10, 20), i32::MAX);
/// assert_eq!(exp(-10, 20), i32::MAX);
/// assert_eq!(exp(-10, 21), i32::MIN);
/// ```
pub fn exp(op1: i32, op2: i32) -> i32
{
	match (op1, op2)
	{
		(0, 0) => 1,
		(0, _) => 0,
		(1, _) => 1,
		(-1, op2) => match op2 % 2 == 0
		{
			true => 1,
			false => -1
		},
		(_, ..0) => 0,
		(op1, op2) => op1.saturating_pow(op2.clamp(0, i32::MAX) as u32)
	}
}

/// Computes the negation of the operand, saturating on overflow. This serves as
/// the definition of the negation operation throughout the library.
///
/// # Parameters
/// - `op`: The operand.
///
/// # Returns
/// The negation of the operand.
///
/// # Examples
/// ```rust
/// # use xdy::neg;
/// assert_eq!(neg(1), -1);
/// assert_eq!(neg(i32::MAX), i32::MIN + 1);
/// assert_eq!(neg(i32::MIN), i32::MAX);
/// ```
#[inline]
pub fn neg(op: i32) -> i32 { op.saturating_neg() }
