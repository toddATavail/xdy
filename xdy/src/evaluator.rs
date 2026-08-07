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
	hash::{Hash, Hasher},
	ops::RangeInclusive
};

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
	Add, AddressingMode, CanAllocate, CanVisitInstructions as _,
	CompilationError, Div, DropHighest, DropLowest, Exp, Function,
	InstructionVisitor, Mod, Mul, Neg, ProgramCounter, RegisterIndex, Return,
	RollCustomDice, RollRange, RollStandardDice, RollingRecordIndex,
	SourceSpan, Sub, SumRollingRecord, add, div, exp, r#mod, mul, neg,
	parser::ParseError, roll_custom_dice, roll_range, roll_standard_dice, sub
};

////////////////////////////////////////////////////////////////////////////////
//                                Evaluation.                                 //
////////////////////////////////////////////////////////////////////////////////

/// A record of a roll of dice. The record includes the results of each die in
/// a set of dice. [`RollStandardDice`] and [`RollCustomDice`] both populate a
/// [`RollingRecord`] in the frame's rolling record set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
/// exponentiation ensure that no undefined behavior occurs. In particular:
///
/// * Division by zero is treated as zero,
/// * Zero to the power of zero is treated as one,
/// * Bases raised to negative exponents are treated as zero.
///
/// # Virtual machine model
///
/// The evaluator is a simple register machine with two register files:
///
/// ```mermaid
/// graph TD
///     subgraph VM["Evaluator VM"]
///         direction TB
///         PC["Program Counter"]
///         subgraph RF["Register Bank (i32)"]
///             R0["@0: param/extern"]
///             R1["@1: param/extern"]
///             RN["@N: computed"]
///         end
///         subgraph RR["Rolling Record Bank"]
///             RR0["⚅0: dice/range results"]
///             RR1["⚅1: dice/range results"]
///         end
///         RES["Result Register"]
///     end
///     F["Function (IR)"] --> PC
///     RNG["pRNG"] --> RR
///     ARGS["Arguments"] --> RF
///     ENV["Environment"] --> RF
///     VM --> OUT["Evaluation"]
///     style VM fill:#e8f4fd,stroke:#333,color:#000
///     style RF fill:#d4edda,stroke:#333,color:#000
///     style RR fill:#fff3cd,stroke:#333,color:#000
///     style OUT fill:#9f9,stroke:#333,color:#000
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
	/// default to zero during evaluation, but all parameters must be bound.
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
	pub fn evaluate<R>(
		&mut self,
		args: impl IntoIterator<Item = i32>,
		rng: &mut R
	) -> Result<Evaluation, EvaluationError<'static>>
	where
		R: Rng + ?Sized
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
		// Execute the instructions sequentially. The last instruction must be a
		// return instruction.
		for instruction in &self.function.instructions
		{
			instruction.visit(&mut state).unwrap();
			state.pc.allocate();
		}
		Ok(state.into())
	}
}

impl Hash for Evaluator
{
	fn hash<H: Hasher>(&self, state: &mut H)
	{
		self.function.hash(state);
		self.environment.iter().for_each(|entry| entry.hash(state));
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
///
/// # Type parameters
/// - `'error`: The lifetime of the source text or external variable name that
///   caused the error. For [`ParseError`](Self::ParseError), this is the source
///   code; for [`UnrecognizedExternal`](Self::UnrecognizedExternal), it is the
///   variable name passed to [`Evaluator::bind()`](crate::Evaluator::bind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError<'error>
{
	/// The source code could not be parsed. Produced only by the simple
	/// [one-shot evaluator](crate::evaluate).
	ParseError(ParseError<'error>),

	/// A formal parameter name was declared more than once in the function
	/// signature. Produced only by the simple
	/// [one-shot evaluator](crate::evaluate).
	DuplicateParameter
	{
		/// The duplicated parameter name, borrowed from the source text.
		name: &'error str,

		/// The span of the first occurrence of the name in the parameter list.
		first: SourceSpan,

		/// The span of the duplicate occurrence that triggered the error.
		duplicate: SourceSpan
	},

	/// A [local binding](crate::ast::Binding) uses a name that is already
	/// declared as a formal parameter. Produced only by the simple
	/// [one-shot evaluator](crate::evaluate).
	BindingCollidesWithParameter
	{
		/// The colliding name, borrowed from the source text.
		name: &'error str,

		/// The span of the parameter declaration in the function signature.
		parameter: SourceSpan,

		/// The span of the binding-site name that triggered the error.
		binding: SourceSpan
	},

	/// The same name is bound more than once by
	/// [local bindings](crate::ast::Binding) within a single function body.
	/// Produced only by the simple [one-shot evaluator](crate::evaluate).
	DuplicateBinding
	{
		/// The rebound name, borrowed from the source text.
		name: &'error str,

		/// The span of the first binding-site name.
		first: SourceSpan,

		/// The span of the duplicate binding-site name that triggered the
		/// error.
		duplicate: SourceSpan
	},

	/// A [variable reference](crate::ast::Variable) appears lexically before
	/// the [binding](crate::ast::Binding) that introduces its name. Produced
	/// only by the simple [one-shot evaluator](crate::evaluate).
	UseBeforeBind
	{
		/// The name that was referenced before being bound, borrowed from the
		/// source text.
		name: &'error str,

		/// The span of the offending reference.
		reference: SourceSpan,

		/// The span of the binding-site name that appears later in the source.
		binding: SourceSpan
	},

	/// The function could not be optimized. Produced only by the simple
	/// [one-shot evaluator](crate::evaluate).
	OptimizationFailed,

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
			EvaluationError::ParseError(e) =>
			{
				write!(f, "{}", e)
			},
			EvaluationError::DuplicateParameter {
				name,
				first,
				duplicate
			} =>
			{
				write!(
					f,
					"duplicate parameter '{}' at {} (first declared at {})",
					name, duplicate, first
				)
			},
			EvaluationError::OptimizationFailed =>
			{
				write!(f, "optimization failed")
			},
			EvaluationError::BadArity { expected, given } =>
			{
				write!(
					f,
					"expected {} arguments but received {}",
					expected, given
				)
			},
			EvaluationError::BindingCollidesWithParameter {
				name,
				parameter,
				binding
			} =>
			{
				write!(
					f,
					"local binding '{}' at {} collides with formal \
					 parameter declared at {}",
					name, binding, parameter
				)
			},
			EvaluationError::DuplicateBinding {
				name,
				first,
				duplicate
			} =>
			{
				write!(
					f,
					"duplicate local binding '{}' at {} (first bound at {})",
					name, duplicate, first
				)
			},
			EvaluationError::UseBeforeBind {
				name,
				reference,
				binding
			} =>
			{
				write!(
					f,
					"reference to '{}' at {} precedes its binding at {}",
					name, reference, binding
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

impl<'src> From<CompilationError<'src>> for EvaluationError<'src>
{
	fn from(e: CompilationError<'src>) -> Self
	{
		match e
		{
			CompilationError::ParseError(e) => Self::ParseError(e),
			CompilationError::DuplicateParameter {
				name,
				first,
				duplicate
			} => Self::DuplicateParameter {
				name,
				first,
				duplicate
			},
			CompilationError::BindingCollidesWithParameter {
				name,
				parameter,
				binding
			} => Self::BindingCollidesWithParameter {
				name,
				parameter,
				binding
			},
			CompilationError::DuplicateBinding {
				name,
				first,
				duplicate
			} => Self::DuplicateBinding {
				name,
				first,
				duplicate
			},
			CompilationError::UseBeforeBind {
				name,
				reference,
				binding
			} => Self::UseBeforeBind {
				name,
				reference,
				binding
			},
			CompilationError::OptimizationFailed => Self::OptimizationFailed
		}
	}
}

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
	///
	/// # Notes
	/// Unsupplied external variables default to zero, which is a roll-time
	/// convention that does not survive contact with a static analysis: on
	/// `1D6 + {x}` with nothing bound, this answers `[1, 6]`, which is a
	/// confidently wrong bound rather than a diagnosable one. Use
	/// [`bounds_over`](Self::bounds_over) instead, which treats an unsupplied
	/// binding as the whole of `i32`.
	#[inline]
	#[deprecated(
		since = "0.12.0",
		note = "unsupplied externals silently default to zero, which \
			under-approximates the bounds; use `bounds_over`, which treats an \
			unsupplied binding as the whole of `i32`. Removed in 1.0.0."
	)]
	pub fn bounds(
		&self,
		args: impl IntoIterator<Item = i32>
	) -> Result<Bounds, EvaluationError<'_>>
	{
		BoundsEvaluator::new(&self.function).evaluate(
			args.into_iter().map(|arg| Some(arg.into())),
			self.environment
				.iter()
				.map(|(index, value)| (*index, (*value).into())),
			EvaluationBounds::default()
		)
	}

	/// Compute the bounds of the function over interval-valued bindings, where
	/// each binding may be supplied or left unconstrained. An unsupplied
	/// binding is bounded by the whole of `i32`, so the answer is sound no
	/// matter what the caller knows.
	///
	/// # Parameters
	/// - `args`: The intervals of the arguments, one per formal parameter, in
	///   declaration order. `None` constrains nothing, and is an explicit
	///   admission of ignorance rather than an omission; the list is therefore
	///   always exactly as long as the arity.
	/// - `externals`: The intervals of the external variables, by name. Names
	///   may be supplied in any order, and any subset may be supplied; an
	///   unmentioned external constrains nothing. A name supplied more than
	///   once takes its last interval.
	///
	/// # Returns
	/// The bounds of the function, covering both the value and the number of
	/// possible outcomes. The possible outcome count is `None` if the function
	/// contains dynamic range or roll expressions, i.e., range or roll
	/// expressions whose count or faces are themselves determined by range or
	/// roll expressions, and also whenever any binding is non-degenerate — see
	/// the notes below.
	///
	/// # Errors
	/// - [`BadArity`](EvaluationError::BadArity) if the number of argument
	///   intervals disagrees with the number of formal parameters in the
	///   function signature.
	/// - [`UnrecognizedExternal`](EvaluationError::UnrecognizedExternal) if a
	///   name in `externals` is not declared by the function.
	///
	/// # Notes
	/// This is a static query, and it deliberately **ignores**
	/// [`environment`](Self::environment). A binding established by
	/// [`bind`](Self::bind) for the benefit of [`evaluate`](Self::evaluate) is
	/// a roll-time convention, and silently inheriting it would make the same
	/// call answer differently depending on which [`bind`](Self::bind) calls
	/// happened to precede it. To constrain an external here, pass it in
	/// `externals`.
	///
	/// The outcome count is exact only when every binding is degenerate, i.e.,
	/// a single value. As soon as one is an interval, the count would be an
	/// exact-looking number that is merely one of the counts the function might
	/// have, so it is reported as `None` instead.
	///
	/// The analysis is interval arithmetic, which is subject to the dependency
	/// problem: two occurrences of the same binding are not recognized as one
	/// value. So `x: {x} - {x}` over `x ∈ [-a, a]` answers `[-2a, 2a]` rather
	/// than `[0, 0]`. Over-approximation is the safe direction — the bounds
	/// always contain every reachable value — but they are not always tight.
	///
	/// # Examples
	/// ```rust
	/// use xdy::{compile, EvaluationBounds, Evaluator};
	///
	/// let function = compile("x: {x}D6")?;
	/// let evaluator = Evaluator::new(function);
	/// let bounds = evaluator.bounds_over([Some((1, 20).into())], [])?;
	///
	/// assert_eq!(bounds.value, (1, 120).into());
	/// assert_eq!(bounds.count, None);
	/// # Ok::<(), xdy::EvaluationError>(())
	/// ```
	pub fn bounds_over<'s>(
		&self,
		args: impl IntoIterator<Item = Option<EvaluationBounds>>,
		externals: impl IntoIterator<Item = (&'s str, EvaluationBounds)>
	) -> Result<Bounds, EvaluationError<'s>>
	{
		// Resolve the external names before evaluating anything, so that an
		// unrecognized name is reported rather than silently ignored.
		let externals = externals
			.into_iter()
			.map(|(name, bounds)| {
				let index = self
					.function
					.externals
					.iter()
					.position(|external| external == name)
					.ok_or(EvaluationError::UnrecognizedExternal(name))?;
				Ok((index, bounds))
			})
			.collect::<Result<Vec<_>, EvaluationError>>()?;
		BoundsEvaluator::new(&self.function).evaluate(
			args,
			externals,
			EvaluationBounds::unconstrained()
		)
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
	/// Answer the bounds that constrain nothing, i.e., the whole of `i32`.
	///
	/// # Returns
	/// The unconstrained bounds.
	///
	/// # Notes
	/// [`Default`] answers `[0, 0]`, which constrains a value to exactly zero,
	/// so it must not be pressed into service here.
	///
	/// # Examples
	/// ```rust
	/// use xdy::EvaluationBounds;
	///
	/// let bounds = EvaluationBounds::unconstrained();
	///
	/// assert!(bounds.contains(i32::MIN));
	/// assert!(bounds.contains(0));
	/// assert!(bounds.contains(i32::MAX));
	/// ```
	#[inline]
	pub const fn unconstrained() -> Self
	{
		Self {
			min: i32::MIN,
			max: i32::MAX
		}
	}

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
	fn from(evaluator: BoundsEvaluator<'eval>) -> Self { evaluator.result }
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

/// The bounds of a function, as computed by a bounds evaluator.
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
	fn from(evaluator: BoundsEvaluator<'eval>) -> Self
	{
		Self {
			value: evaluator.result,
			count: evaluator.count
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
	fn new(function: &'eval Function) -> Self
	{
		Self {
			function,
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

	/// Evaluate the bounds of the function over the given bindings. Both
	/// channels admit a partial supply; whatever is not supplied is bounded by
	/// `unsupplied`, which is the sole point of difference between the two
	/// public entry points.
	///
	/// # Parameters
	/// - `args`: The intervals of the arguments, one per formal parameter, in
	///   declaration order, where `None` denotes an unsupplied argument.
	/// - `externals`: The intervals of the external variables, by index into
	///   [`externals`](Function::externals). An index absent from the iterable
	///   is unsupplied.
	/// - `unsupplied`: The interval that bounds an unsupplied binding in either
	///   channel.
	///
	/// # Returns
	/// The bounds of the function, covering both the value and the number of
	/// possible outcomes. The outcome count is `None` whenever any binding is
	/// non-degenerate, as it would otherwise be an exact-looking number that
	/// holds for only one member of the interval.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	pub fn evaluate<'error>(
		mut self,
		args: impl IntoIterator<Item = Option<EvaluationBounds>>,
		externals: impl IntoIterator<Item = (usize, EvaluationBounds)>,
		unsupplied: EvaluationBounds
	) -> Result<Bounds, EvaluationError<'error>>
	{
		// Check the argument count. `None` is an explicit admission of
		// ignorance rather than an omission, so the list is as long as the
		// arity no matter how little the caller knows.
		let arity = self.function.arity();
		let args = args.into_iter().collect::<Vec<_>>();
		if args.len() != arity
		{
			return Err(EvaluationError::BadArity {
				expected: arity,
				given: args.len()
			});
		}
		// Assemble the intervals of every binding, arguments first and then
		// external variables, taking the unsupplied interval wherever the
		// caller supplied nothing.
		let mut bindings = args
			.into_iter()
			.map(|arg| arg.unwrap_or(unsupplied))
			.collect::<Vec<_>>();
		bindings.resize(arity + self.function.externals.len(), unsupplied);
		for (index, bounds) in externals
		{
			if let Some(binding) = bindings.get_mut(arity + index)
			{
				*binding = bounds;
			}
		}
		let degenerate = bindings.iter().all(|bounds| bounds.min == bounds.max);
		// Seed the binding registers. The optimizer recomputes the register
		// count from the surviving instructions, so a binding that no
		// instruction reads may have no register at all; such a binding cannot
		// influence the result, so skip it rather than indexing past the
		// register file.
		for (index, bounds) in bindings.into_iter().enumerate()
		{
			if let Some(register) = self.registers.get_mut(index)
			{
				*register = bounds;
			}
		}
		for instruction in &self.function.instructions
		{
			instruction.visit(&mut self).unwrap();
			self.pc.allocate();
		}
		if !degenerate
		{
			// The outcome count is an exact count of the outcomes of one
			// binding of the function. That is a meaningful answer only when
			// the bindings pick out exactly one such function.
			self.count = None;
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

/// The greatest number of divisor magnitudes that [`Rem`](std::ops::Rem) will
/// enumerate in pursuit of an exact answer. Beyond this it approximates. The
/// budget is generous enough to cover any single die — percentile dice span a
/// hundred magnitudes — while bounding the work at a fixed, trivial cost paid
/// once per remainder instruction during bounds analysis.
const REM_ENUMERATION_BUDGET: i64 = 128;

/// Compute the interval of magnitudes assumed by a divisor interval, i.e.,
/// `{|y| : y ∈ divisor}`.
///
/// `x % y == x % -y`, so only the magnitude of a divisor affects the remainder.
/// Reducing the divisor to its magnitudes up front is what keeps
/// [`Rem`](std::ops::Rem) from having to case on signs, which is where the
/// original implementation went wrong: it bounded the remainder by the divisor
/// endpoint *nearest* zero rather than the one *farthest* from it.
///
/// # Parameters
/// - `divisor`: The divisor interval.
///
/// # Returns
/// The least and greatest magnitudes, as `i64`. The widening is load bearing:
/// `|i32::MIN|` does not fit in an `i32`, and clamping it to [`i32::MAX`] would
/// discard exactly the magnitude that makes `i32::MAX % i32::MIN == i32::MAX`
/// escape the bound.
///
/// # Notes
/// The magnitudes of a contiguous interval are themselves contiguous: an
/// interval straddling zero contributes every magnitude from zero up to that of
/// its farthest endpoint, and an interval on one side of zero contributes the
/// magnitudes of its endpoints and everything between.
fn magnitudes(divisor: EvaluationBounds) -> (i64, i64)
{
	let (min, max) = (divisor.min as i64, divisor.max as i64);
	match (min, max)
	{
		(min, max) if min <= 0 && max >= 0 => (0, (-min).max(max)),
		(min, max) if min > 0 => (min, max),
		(min, max) => (-max, -min)
	}
}

/// Compute the exact hull of `{p % magnitude : p ∈ [low, high]}`, for
/// nonnegative dividends and a single divisor magnitude.
///
/// # Parameters
/// - `low`: The least dividend. Must be nonnegative and no greater than `high`.
/// - `high`: The greatest dividend.
/// - `magnitude`: The magnitude of the divisor. Zero denotes division by zero,
///   which the expression language defines to answer zero.
///
/// # Returns
/// The least and greatest remainders.
///
/// # Notes
/// `p % magnitude` ascends by one with `p` until it wraps to zero at each
/// multiple of `magnitude`. Three cases follow. If the dividends span at least
/// a full period, every residue occurs. Otherwise the dividends span part of a
/// period, and either lie within a single period — whence the endpoints give
/// the extrema — or straddle exactly one wrap, whence the hull is again the
/// whole period, since the wrap contributes zero and its immediate predecessor
/// contributes `magnitude - 1`.
fn remainders_of(low: i64, high: i64, magnitude: i64) -> (i64, i64)
{
	debug_assert!(0 <= low && low <= high);
	if magnitude == 0
	{
		return (0, 0)
	}
	if high - low >= magnitude - 1
	{
		return (0, magnitude - 1)
	}
	match (low % magnitude, high % magnitude)
	{
		(low, high) if low <= high => (low, high),
		_ => (0, magnitude - 1)
	}
}

/// Compute a hull of `{p % m : p ∈ [low, high], m ∈ [least, greatest]}`, for
/// nonnegative dividends and an interval of divisor magnitudes.
///
/// # Parameters
/// - `low`: The least dividend. Must be nonnegative and no greater than `high`.
/// - `high`: The greatest dividend.
/// - `least`: The least divisor magnitude.
/// - `greatest`: The greatest divisor magnitude. Must be positive and no less
///   than `least`.
///
/// # Returns
/// The least and greatest remainders. Exact when the magnitudes fall within
/// [`REM_ENUMERATION_BUDGET`], and otherwise a sound over-approximation.
///
/// # Notes
/// Hulling distributes over a union, so hulling the union of the per-magnitude
/// hulls answers the exact hull of the whole. That is affordable only for
/// narrow magnitude intervals, so two approximations cover the rest. A divisor
/// magnitude exceeding every dividend leaves the dividend unchanged, so the
/// dividends are their own remainders. Failing that, a remainder is smaller
/// than the magnitude that produced it and no larger than the dividend that
/// produced it, and zero is assumed to be attainable.
fn remainders(low: i64, high: i64, least: i64, greatest: i64) -> (i64, i64)
{
	debug_assert!(0 <= low && low <= high);
	debug_assert!(0 <= least && least <= greatest && greatest > 0);
	if greatest - least < REM_ENUMERATION_BUDGET
	{
		return (least..=greatest).fold((i64::MAX, i64::MIN), |hull, m| {
			let (min, max) = remainders_of(low, high, m);
			(hull.0.min(min), hull.1.max(max))
		})
	}
	if least > high
	{
		return (low, high)
	}
	(0, high.min(greatest - 1))
}

impl std::ops::Rem for EvaluationBounds
{
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output
	{
		// Remainder is quite complex, especially because of the singularity at
		// zero and the saturation semantics. Reduce the divisor to the
		// magnitudes it assumes, since the sign of a divisor cannot affect a
		// remainder.
		let (least, greatest) = magnitudes(rhs);
		if greatest == 0
		{
			// The divisor is known to be zero, which the expression language
			// defines to answer zero.
			return 0.into()
		}
		// A remainder takes the sign of its dividend, and negating a dividend
		// negates its remainder, so the negative dividends are a mirror image
		// of the nonnegative ones. Treat each side as a nonnegative problem,
		// reflecting the negative side through zero on the way in and on the
		// way out. Note that the sides overlap only at zero, and that at least
		// one side is inhabited, since the interval is never inverted.
		let (low, high) = ((self.min as i64).max(0), self.max as i64);
		let (mirror_low, mirror_high) =
			((-(self.max as i64)).max(0), -(self.min as i64));
		let min = match self.min <= 0
		{
			true => -remainders(mirror_low, mirror_high, least, greatest).1,
			false => remainders(low, high, least, greatest).0
		};
		let max = match self.max >= 0
		{
			true => remainders(low, high, least, greatest).1,
			false => -remainders(mirror_low, mirror_high, least, greatest).0
		};
		// Both are back within `i32`: a remainder is smaller in magnitude than
		// its divisor, and no divisor magnitude exceeds `|i32::MIN|`, so no
		// remainder can reach `|i32::MIN|` itself.
		Self {
			min: min as i32,
			max: max as i32
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
	/// Unlike its siblings, this is not a structural interval operation. It
	/// evaluates [`exp`] at a small set of interesting bases and exponents and
	/// unions the results, relying on the fact that the extrema of integer
	/// exponentiation over an interval always occur at one of them. The
	/// sampling is exact over every case reachable with small operands; see
	/// `tests::bounds` for the exhaustive grid that establishes this.
	///
	/// # Parameters
	/// - `rhs`: The bounds of the exponent.
	///
	/// # Returns
	/// The bounds of the result.
	pub(crate) fn exp(self, rhs: EvaluationBounds) -> Self
	{
		let interesting_bases = HashSet::from([self.min, -1, 0, 1, self.max]);
		// Zero must be sampled explicitly. It is the only exponent for which
		// `exp(0, power)` is one rather than zero, so omitting it
		// under-approximates whenever the base is exactly zero and the exponent
		// interval straddles zero without either neighbor of an endpoint
		// landing on it.
		let interesting_powers = HashSet::from([
			rhs.min,
			rhs.min.saturating_add(1),
			0,
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
