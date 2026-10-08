//! # Evaluation
//!
//! Evaluation is the process of determining the value of a dice expression. It
//! requires a compiled [function](Function) and a pseudo-random number
//! generator. Evaluation produces the final result of an expression and the
//! individual outcomes of each dice subexpression.

use std::{
	borrow::Cow,
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
	CompilationError, Criterion, Div, DropHighest, DropLowest, Exp, Explode,
	Function, Instruction, InstructionVisitor, KeepHighest, KeepLowest, Limit,
	Max, Min, Mod, Mul, Neg, Predicate, ProgramCounter, RegisterIndex, Reroll,
	Return, RollCustomDice, RollRange, RollStandardDice, RollingRecordIndex,
	SourceSpan, Sub, SumRollingRecord, add, div, exp, max, min, r#mod, mul,
	neg, parser::ParseError, roll_custom_dice, roll_range, roll_standard_dice,
	sub
};

////////////////////////////////////////////////////////////////////////////////
//                                Evaluation.                                 //
////////////////////////////////////////////////////////////////////////////////

/// A record of a roll of dice. The record includes every die of a set of dice,
/// in the order rolled, each marked with its [fate](Fate). [`RollRange`],
/// [`RollStandardDice`], and [`RollCustomDice`] each populate a
/// [`RollingRecord`] in the frame's rolling record set.
///
/// A die that a drop or keep dropped stays in the record, crossed out, so the
/// record shows everything that happened to its dice. So does a die that a
/// [reroll](crate::Reroll) rerolled, whose replacement follows the dice
/// already rolled, [linked](Origin::Replaces) to it, so that each die's chain
/// of rerolls can be traced. Only the [kept](Fate::Kept) dice reach the sum or
/// any later drop, keep, reroll, or [explode](crate::Explode).
///
/// Every die descends, by its [origin](Die::origin), from a die that the roll
/// that filled the record rolled, and every die that descends from the same
/// one belongs to the same [chain](RollingRecord::chains). A die that extends
/// another, as an explosion rolls it, joins the chain of the die that it
/// extends, and a replacement joins the chain of the die that it replaces, so
/// a chain holds at most one kept die unless some die extends another. The sum
/// and every drop and keep count a chain as one die, worth the sum of its kept
/// dice, or worth the value of its [saturated](Origin::Saturates) die if it
/// has one, so the multiset of the totals of the chains still kept decides
/// everything that the record computes.
///
/// A record prints its kind and then its dice, in the order rolled, each
/// crossed-out die prefixed with `~`, whatever crossed it out, as in
/// `3D6: [5, ~1, 3]`. A saturated die prints as `∞` or `-∞`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingRecord
{
	/// The kind of rolling record.
	pub kind: RollingRecordKind<i32>,

	/// Every die in the roll, in the order rolled.
	pub dice: Vec<Die>
}

impl Display for RollingRecord
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{}: [", self.kind)?;
		for (i, die) in self.dice.iter().enumerate()
		{
			if i != 0
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", die)?;
		}
		write!(f, "]")
	}
}

/// A die of a [rolling record](RollingRecord): its result, its fate, and its
/// origin. A die prints as its result, prefixed with `~` if it is crossed out;
/// a [saturated](Origin::Saturates) die whose result is `i32::MAX` or
/// `i32::MIN` prints as `∞` or `-∞`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Die
{
	/// The result of the die.
	pub value: i32,

	/// Whether the die still counts.
	pub fate: Fate,

	/// What rolled the die: the roll that filled the record, or a reroll or
	/// explosion of an earlier die of the record.
	pub origin: Origin
}

impl Die
{
	/// Answer a freshly rolled die, which is [kept](Fate::Kept).
	///
	/// # Parameters
	/// - `value`: The result of the die.
	///
	/// # Returns
	/// The kept die.
	#[inline]
	pub const fn kept(value: i32) -> Self
	{
		Self {
			value,
			fate: Fate::Kept,
			origin: Origin::Rolled
		}
	}

	/// Answer a freshly rolled die, which is [kept](Fate::Kept), that replaces
	/// a die that a reroll crossed out.
	///
	/// # Parameters
	/// - `value`: The result of the die.
	/// - `replaces`: The index, among the dice of the record, of the die that
	///   it replaces.
	///
	/// # Returns
	/// The kept die.
	#[inline]
	pub const fn replacing(value: i32, replaces: usize) -> Self
	{
		Self {
			value,
			fate: Fate::Kept,
			origin: Origin::Replaces(replaces)
		}
	}

	/// Answer a freshly rolled die, which is [kept](Fate::Kept), that extends
	/// a die that an explosion matched.
	///
	/// # Parameters
	/// - `value`: The result of the die.
	/// - `extends`: The index, among the dice of the record, of the die that it
	///   extends.
	///
	/// # Returns
	/// The kept die.
	#[inline]
	pub const fn extending(value: i32, extends: usize) -> Self
	{
		Self {
			value,
			fate: Fate::Kept,
			origin: Origin::Extends(extends)
		}
	}

	/// Answer a [saturated](Origin::Saturates) die, which is
	/// [kept](Fate::Kept), that stands for the endless chain of dice that an
	/// explosion matching every face of its die would roll.
	///
	/// # Parameters
	/// - `value`: The total of the chain: `i32::MAX` or `i32::MIN` by the sign
	///   of the mean face value, or `0` if that mean is `0`.
	/// - `saturates`: The index, among the dice of the record, of the die after
	///   which the endless chain would begin.
	///
	/// # Returns
	/// The kept die.
	#[inline]
	pub const fn saturating(value: i32, saturates: usize) -> Self
	{
		Self {
			value,
			fate: Fate::Kept,
			origin: Origin::Saturates(saturates)
		}
	}

	/// Answer whether the die is [kept](Fate::Kept), i.e., not crossed out.
	///
	/// # Returns
	/// `true` if the die still counts, `false` otherwise.
	#[inline]
	pub const fn is_kept(&self) -> bool { matches!(self.fate, Fate::Kept) }
}

impl Display for Die
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		if !self.is_kept()
		{
			write!(f, "~")?;
		}
		match (self.origin, self.value)
		{
			(Origin::Saturates(_), i32::MAX) => write!(f, "∞"),
			(Origin::Saturates(_), i32::MIN) => write!(f, "-∞"),
			(_, value) => write!(f, "{}", value)
		}
	}
}

/// The origin of a [die](Die) of a [rolling record](RollingRecord): the roll
/// that filled the record, or the earlier die of the record from which the die
/// descends. Following the links from any die leads to the die that begins its
/// [chain](RollingRecord::chains). A link that names no earlier die of the
/// record begins a chain of its own, as [`Rolled`](Self::Rolled) does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Origin
{
	/// The roll that filled the record rolled the die, which begins a chain.
	#[default]
	Rolled,

	/// The die replaces the die at this index, which a
	/// [reroll](Fate::Rerolled) crossed out, and joins its chain.
	Replaces(usize),

	/// The die extends the die at this index, which an explosion matched and
	/// which stays [kept](Fate::Kept), and joins its chain.
	Extends(usize),

	/// The die stands for the endless chain of dice that an explosion matching
	/// every face of its die would roll after the die at this index, and joins
	/// its chain. Its value is the total of the chain, whatever its other dice
	/// show: `i32::MAX` or `i32::MIN` by the sign of the mean face value,
	/// or `0` if that mean is `0`.
	Saturates(usize)
}

impl Origin
{
	/// Answer the index of the die from which the die descends, if any.
	///
	/// # Returns
	/// The index that the origin names, or `None` if the roll that filled the
	/// record rolled the die.
	#[inline]
	pub const fn link(&self) -> Option<usize>
	{
		match self
		{
			Self::Rolled => None,
			Self::Replaces(index)
			| Self::Extends(index)
			| Self::Saturates(index) => Some(*index)
		}
	}
}

/// The fate of a [die](Die) of a [rolling record](RollingRecord): whether it
/// still counts, and if not, what crossed it out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Fate
{
	/// The die still counts toward the sum of its record, and any later drop,
	/// keep, or reroll may cross it out.
	#[default]
	Kept,

	/// A drop or keep crossed out the die, so it no longer counts.
	Dropped,

	/// A reroll crossed out the die, so it no longer counts. The reroll
	/// appended the die's replacement to the record, linked to it, unless
	/// every face of the die matched an unlimited reroll, which rolls nothing.
	Rerolled
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
/// pseudo-random number generator to generate the results of dice rolls.
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
///         DM["Dice Meter<br/>(remaining, consumed)"]
///     end
///     F["Function (IR)"] --> PC
///     RNG["pRNG"] --> RR
///     ARGS["Arguments"] --> RF
///     ENV["Environment"] --> RF
///     BUD["Dice Budget"] --> DM
///     DM -->|"charges each roll<br/>before it rolls"| RR
///     VM --> OUT["Evaluation<br/>(result, records, dice)"]
///     VM -.->|"a roll would<br/>exceed the budget"| ERR["DiceBudgetExhausted"]
///     style VM fill:#e8f4fd,stroke:#333,color:#000
///     style RF fill:#d4edda,stroke:#333,color:#000
///     style RR fill:#fff3cd,stroke:#333,color:#000
///     style OUT fill:#9f9,stroke:#333,color:#000
///     style ERR fill:#f99,stroke:#333,color:#000
/// ```
///
/// [`evaluate`](Self::evaluate) is [metered](Self::evaluate_metered) with an
/// unlimited budget.
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
	result: i32,

	/// The number of dice that the evaluation may still roll.
	remaining: u64,

	/// The number of dice that the evaluation has rolled so far.
	consumed: u64
}

/// The result of a dice expression evaluation. The result includes the final
/// value of the expression, the rolling records of the dice subexpressions,
/// and the number of dice rolled.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluation
{
	/// The result of the entire dice expression.
	pub result: i32,

	/// The rolling records for the dice subexpressions.
	pub records: Vec<RollingRecord>,

	/// The number of dice rolled, where a range counts as one die, and each
	/// reroll or extension as one more. This is exactly the consumption
	/// charged against the budget of a
	/// [metered](Evaluator::evaluate_metered) evaluation, so a caller that
	/// carries one budget across several evaluations can deduct it.
	#[cfg_attr(feature = "serde", serde(default))]
	pub dice: u64
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
			records: value.records,
			dice: value.consumed
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
	/// - `name`: The [canonical](crate::parser::canonical_name) name of the
	///   target external variable, whose whitespace is single spaces between
	///   words, so `{a  b}` is bound as `"a b"`.
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
	///
	/// # Notes
	/// Evaluation is unmetered, so the number of dice rolled is bounded only by
	/// the expression and its bindings: `{n}D6` with `n` bound to [`i32::MAX`]
	/// rolls over two billion dice. When the expression or its bindings are
	/// untrusted, use [`evaluate_metered`](Self::evaluate_metered) instead.
	pub fn evaluate<R>(
		&mut self,
		args: impl IntoIterator<Item = i32>,
		rng: &mut R
	) -> Result<Evaluation, EvaluationError<'static>>
	where
		R: Rng + ?Sized
	{
		// No evaluation can roll `u64::MAX` dice, since their results would
		// not fit in memory, so this budget never refuses.
		self.evaluate_metered(args, rng, u64::MAX)
	}

	/// Evaluate the function, as [`evaluate`](Self::evaluate) does, but roll no
	/// more than `budget` dice in total, where a range counts as one die. A
	/// roll that would exceed what remains of the budget is refused before it
	/// draws anything from the pRNG or allocates anything, and the refusal ends
	/// the evaluation; there is no partial result.
	///
	/// The instructions of a compiled function never branch, and loop only to
	/// roll dice, as a reroll or explode does, which charges each die that it
	/// rolls, so
	/// every cost of evaluation beyond the size of the function is
	/// proportional to the dice rolled. Metering the dice therefore meters the
	/// whole evaluation.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	/// - `rng`: The pseudo-random number generator to use for range and dice
	///   rolls.
	/// - `budget`: The greatest number of dice that the evaluation may roll.
	///
	/// # Returns
	/// The result of the evaluation, whose [`dice`](Evaluation::dice) reports
	/// the dice actually rolled. Within the budget, metering never disturbs the
	/// draws from the pRNG, so the result is exactly that of
	/// [`evaluate`](Self::evaluate) for the same arguments and pRNG state.
	///
	/// # Errors
	/// - [`BadArity`](EvaluationError::BadArity) if the number of arguments
	///   provided disagrees with the number of formal parameters in the
	///   function signature.
	/// - [`DiceBudgetExhausted`](EvaluationError::DiceBudgetExhausted) if the
	///   evaluation asks for more dice than the budget allows.
	///
	/// # Examples
	/// ```rust
	/// use rand::rng;
	/// use xdy::{compile, EvaluationError, Evaluator};
	///
	/// let mut evaluator = Evaluator::new(compile("{x}: {x}D6")?);
	///
	/// let evaluation = evaluator.evaluate_metered([3], &mut rng(), 10)?;
	/// assert_eq!(evaluation.dice, 3);
	///
	/// let error = evaluator.evaluate_metered([i32::MAX], &mut rng(), 10);
	/// assert_eq!(
	///     error,
	///     Err(EvaluationError::DiceBudgetExhausted {
	///         requested: i32::MAX as u64,
	///         remaining: 10,
	///         consumed: 0
	///     })
	/// );
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn evaluate_metered<R>(
		&mut self,
		args: impl IntoIterator<Item = i32>,
		rng: &mut R,
		budget: u64
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
			self.function.rolling_record_count,
			budget
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
		// return instruction. Only a roll that exceeds the budget can fail.
		for instruction in &self.function.instructions
		{
			instruction.visit(&mut state)?;
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

impl<R> InstructionVisitor<EvaluationError<'static>> for EvaluatorState<'_, R>
where
	R: Rng + ?Sized
{
	fn visit_roll_range(
		&mut self,
		inst: &RollRange
	) -> Result<(), EvaluationError<'static>>
	{
		let start = self.value(inst.start);
		let end = self.value(inst.end);
		// Even an empty range records a result, so every range costs one die.
		self.meter(1)?;
		let record = roll_range(self.rng, start..=end);
		*self.record_mut(inst.dest) = record;
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), EvaluationError<'static>>
	{
		let count = self.value(inst.count);
		let faces = self.value(inst.faces);
		self.meter(dice(count))?;
		let record = roll_standard_dice(self.rng, count, faces);
		*self.record_mut(inst.dest) = record;
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), EvaluationError<'static>>
	{
		let count = self.value(inst.count);
		self.meter(dice(count))?;
		let record = roll_custom_dice(self.rng, count, inst.faces.clone());
		*self.record_mut(inst.dest) = record;
		Ok(())
	}

	fn visit_drop_lowest(
		&mut self,
		inst: &DropLowest
	) -> Result<(), EvaluationError<'static>>
	{
		// RollingRecord treats a negative count as zero, and crosses out at
		// most the dice still kept.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.drop_lowest(count);
		Ok(())
	}

	fn visit_drop_highest(
		&mut self,
		inst: &DropHighest
	) -> Result<(), EvaluationError<'static>>
	{
		// RollingRecord treats a negative count as zero, and crosses out at
		// most the dice still kept.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.drop_highest(count);
		Ok(())
	}

	fn visit_keep_lowest(
		&mut self,
		inst: &KeepLowest
	) -> Result<(), EvaluationError<'static>>
	{
		// RollingRecord keeps nothing for a count of zero or less, and keeps
		// every die still kept for a count that exceeds them.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.keep_lowest(count);
		Ok(())
	}

	fn visit_keep_highest(
		&mut self,
		inst: &KeepHighest
	) -> Result<(), EvaluationError<'static>>
	{
		// RollingRecord keeps nothing for a count of zero or less, and keeps
		// every die still kept for a count that exceeds them.
		let count = self.value(inst.count);
		let record = self.record_mut(inst.dest);
		record.keep_highest(count);
		Ok(())
	}

	fn visit_reroll(
		&mut self,
		inst: &Reroll
	) -> Result<(), EvaluationError<'static>>
	{
		// Read every operand once, before the first reroll.
		let criterion = inst.predicate.resolve(|op| self.value(op));
		let limit = match inst.limit
		{
			Limit::Times(count) => Some(self.value(count)),
			Limit::Unlimited => None
		};
		// Each reroll rolls one die, which the budget meters before it draws
		// anything from the pRNG.
		let record = &mut self.records[inst.dest.0];
		match record.reroll_within(self.rng, &criterion, limit, self.remaining)
		{
			Some(rolled) =>
			{
				self.remaining -= rolled;
				self.consumed = self.consumed.saturating_add(rolled);
				Ok(())
			},
			None => Err(EvaluationError::DiceBudgetExhausted {
				requested: 1,
				remaining: 0,
				consumed: self.consumed.saturating_add(self.remaining)
			})
		}
	}

	fn visit_explode(
		&mut self,
		inst: &Explode
	) -> Result<(), EvaluationError<'static>>
	{
		// Read every operand once, before the first explosion.
		let criterion = inst.predicate.resolve(|op| self.value(op));
		let limit = match inst.limit
		{
			Limit::Times(count) => Some(self.value(count)),
			Limit::Unlimited => None
		};
		// Each extension rolls one die, which the budget meters before it
		// draws anything from the pRNG. A saturated die rolls nothing.
		let record = &mut self.records[inst.dest.0];
		match record.explode_within(self.rng, &criterion, limit, self.remaining)
		{
			Some(rolled) =>
			{
				self.remaining -= rolled;
				self.consumed = self.consumed.saturating_add(rolled);
				Ok(())
			},
			None => Err(EvaluationError::DiceBudgetExhausted {
				requested: 1,
				remaining: 0,
				consumed: self.consumed.saturating_add(self.remaining)
			})
		}
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), EvaluationError<'static>>
	{
		let sum = self.record_mut(inst.src).sum();
		self.set_register(inst.dest, sum);
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, add(op1, op2));
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, sub(op1, op2));
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, mul(op1, op2));
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, div(op1, op2));
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, r#mod(op1, op2));
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, exp(op1, op2));
		Ok(())
	}

	fn visit_max(&mut self, inst: &Max)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, max(op1, op2));
		Ok(())
	}

	fn visit_min(&mut self, inst: &Min)
	-> Result<(), EvaluationError<'static>>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, min(op1, op2));
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg)
	-> Result<(), EvaluationError<'static>>
	{
		let op = self.value(inst.op);
		self.set_register(inst.dest, neg(op));
		Ok(())
	}

	fn visit_return(
		&mut self,
		inst: &Return
	) -> Result<(), EvaluationError<'static>>
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
	/// - `budget`: The greatest number of dice that the evaluation may roll.
	///
	/// # Returns
	/// A fresh evaluator state.
	fn new(
		rng: &'r mut R,
		registers: usize,
		records: usize,
		budget: u64
	) -> Self
	{
		Self {
			rng,
			pc: ProgramCounter::default(),
			registers: vec![0; registers],
			records: vec![RollingRecord::default(); records],
			result: 0,
			remaining: budget,
			consumed: 0
		}
	}

	/// Charge the specified number of dice against the budget, before rolling
	/// any of them.
	///
	/// # Parameters
	/// - `dice`: The number of dice that the roll requests.
	///
	/// # Errors
	/// [`DiceBudgetExhausted`](EvaluationError::DiceBudgetExhausted) if the
	/// dice exceed what remains of the budget, in which case nothing is
	/// charged.
	fn meter(&mut self, dice: u64) -> Result<(), EvaluationError<'static>>
	{
		if dice > self.remaining
		{
			return Err(EvaluationError::DiceBudgetExhausted {
				requested: dice,
				remaining: self.remaining,
				consumed: self.consumed
			})
		}
		self.remaining -= dice;
		self.consumed = self.consumed.saturating_add(dice);
		Ok(())
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

/// Compute the number of dice that a roll of `count` dice actually rolls. A
/// nonpositive count rolls none.
///
/// # Parameters
/// - `count`: The count of dice requested by a roll instruction.
///
/// # Returns
/// The number of dice rolled.
#[inline]
fn dice(count: i32) -> u64 { u64::try_from(count).unwrap_or(0) }

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

/// An error that may occur during the evaluation of a dice expression. Most
/// variants report a failure of setup: of compilation, binding, or supplying
/// arguments. Evaluation itself fails only when a
/// [metered](Evaluator::evaluate_metered) evaluation exhausts its
/// [dice budget](Self::DiceBudgetExhausted). A
/// [distribution build](crate::DistributionPlan::build) reports its failures
/// as a [`BuildError`](crate::BuildError) instead.
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
		/// The [canonical](crate::parser::canonical_name) duplicated parameter
		/// name, borrowed from the source text wherever it is spelled
		/// canonically there.
		name: Cow<'error, str>,

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
		/// The [canonical](crate::parser::canonical_name) colliding name,
		/// borrowed from the source text wherever it is spelled canonically
		/// there.
		name: Cow<'error, str>,

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
		/// The [canonical](crate::parser::canonical_name) rebound name,
		/// borrowed from the source text wherever it is spelled canonically
		/// there.
		name: Cow<'error, str>,

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
		/// The [canonical](crate::parser::canonical_name) name that was
		/// referenced before being bound, borrowed from the source text
		/// wherever it is spelled canonically there.
		name: Cow<'error, str>,

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
	UnrecognizedExternal(&'error str),

	/// A [metered](Evaluator::evaluate_metered) evaluation asked for more dice
	/// than its budget allows, and was stopped before rolling them. This
	/// reports exhaustion, not a defect: the expression asked for more work
	/// than this evaluation allows, and is not thereby wrong. The refusing
	/// roll drew nothing from the pRNG.
	DiceBudgetExhausted
	{
		/// The number of dice requested by the refusing roll.
		requested: u64,

		/// The budget remaining when the roll was refused, which is less than
		/// `requested`.
		remaining: u64,

		/// The number of dice already rolled when the roll was refused.
		consumed: u64
	}
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
			},
			EvaluationError::DiceBudgetExhausted {
				requested,
				remaining,
				consumed
			} =>
			{
				write!(
					f,
					"dice budget exhausted: a roll of {} dice exceeds the {} \
					 remaining after {} rolled",
					requested, remaining, consumed
				)
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
	/// The bounds of the function, covering the value, the number of possible
	/// outcomes, and the number of dice rolled. The possible outcome count is
	/// `None` if the function contains dynamic range or roll expressions,
	/// i.e., range or roll expressions whose count or faces are themselves
	/// determined by range or roll expressions. In such cases, the outcome
	/// count cannot be determined without generating the complete distribution
	/// of the function, which is too expensive to do without a specific
	/// request.
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
	/// - `externals`: The intervals of the external variables, by
	///   [canonical](crate::parser::canonical_name) name. Names may be supplied
	///   in any order, and any subset may be supplied; an unmentioned external
	///   constrains nothing. A name supplied more than once takes its last
	///   interval.
	///
	/// # Returns
	/// The bounds of the function, covering the value, the number of possible
	/// outcomes, and the number of dice rolled. The possible outcome count is
	/// `None` if the function contains dynamic range or roll expressions,
	/// i.e., range or roll expressions whose count or faces are themselves
	/// determined by range or roll expressions, and also whenever any binding
	/// is non-degenerate — see the notes below.
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
	/// value. So `{x}: {x} + -{x}` over `x ∈ [-a, a]` answers `[-2a, 2a]`
	/// rather than `[0, 0]`. (The optimizer rewrites `{x} - {x}` to `0`, so
	/// that spelling answers `[0, 0]`.) Over-approximation is the safe
	/// direction — the bounds
	/// always contain every reachable value — but they are not always tight.
	///
	/// # Examples
	/// ```rust
	/// use xdy::{compile, EvaluationBounds, Evaluator};
	///
	/// let function = compile("{x}: {x}D6")?;
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
pub(crate) struct BoundsEvaluator<'eval>
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
	count: Option<u128>,

	/// The special register holding the worst-case number of dice rolled.
	dice: u64
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

	/// The bounds on the number of dice still kept, which every roll sets
	/// and every drop and keep narrows, clause by clause. Keeping the count
	/// itself, rather than the drops at either end, keeps the bounds of a keep
	/// exact: a keep counts from the dice still kept, so `{x}D6 keep highest
	/// 2` keeps at most 2 dice, however many it rolls.
	kept: EvaluationBounds,

	/// The results that a die still kept may show, which every roll sets to
	/// the [faces](RollingRecordKind::faces) of its die, and which an
	/// unlimited reroll narrows to those that its predicate cannot match, and
	/// a limited reroll restores, since a replacement may show any face.
	results: ResultSet,

	/// The totals that a [chain](crate::RollingRecord::chains) still kept may
	/// show, whose hull bounds the sum of the record, with `kept`. Until an
	/// [explode](crate::Explode) extends a die, each chain still kept holds
	/// one kept die, so these are the `results`; an explode sets them to
	/// cover the totals of the chains that it may grow, which may have
	/// gaps that these fill, but whose hull is exact.
	totals: ResultSet,

	/// Whether an [explode](crate::Explode) may have extended some die of the
	/// record since its roll, which every roll clears. A later reroll or
	/// explode acts on each die of a chain, whose number and order the bounds
	/// do not yet track, so the bounds of such an instruction are sound but
	/// loose: `results` covers every die of a chain, and the totals may be
	/// any value at all.
	exploded: bool
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

	/// Answer the faces that a die of the record may show, over the bounds of
	/// its faces or endpoints. A die without faces, or an empty range, shows
	/// only the face `0`.
	///
	/// # Returns
	/// The faces, which are empty if the record is uninitialized, since it has
	/// no die.
	pub(crate) fn faces(&self) -> ResultSet
	{
		match self
		{
			RollingRecordKind::Uninitialized => ResultSet::default(),
			RollingRecordKind::Range { start, end } =>
			{
				let faces =
					ResultSet::interval(start.min as i64, end.max as i64);
				// The range is empty if its start may exceed its end.
				match start.max > end.min
				{
					true => faces.union(&ResultSet::of([0])),
					false => faces
				}
			},
			RollingRecordKind::Standard { faces: count, .. } =>
			{
				let faces = ResultSet::interval(1, count.max as i64);
				// The die has no faces if its count of faces may be
				// nonpositive.
				match count.min <= 0
				{
					true => faces.union(&ResultSet::of([0])),
					false => faces
				}
			},
			RollingRecordKind::Custom { faces, .. } if faces.is_empty() =>
			{
				ResultSet::of([0])
			},
			RollingRecordKind::Custom { faces, .. } =>
			{
				ResultSet::of(faces.iter().copied())
			},
		}
	}

	/// Answer the least dice of the record: the dice, among those that the
	/// bounds of its faces or endpoints admit, whose faces include no other
	/// such die's, so that every die of the record shows every face of one of
	/// them. A predicate matches every face of some die of the record exactly
	/// when it matches every face of one of these.
	///
	/// # Returns
	/// The least dice, which are none if the record is uninitialized.
	///
	/// # Notes
	/// The dice of a standard roll grow with their faces, so only the fewest
	/// faces and a die without faces are least. A range may hold any one of
	/// many single values, none of which includes another, so these are least
	/// together.
	pub(crate) fn least_dice(&self) -> Vec<LeastDice>
	{
		match self
		{
			RollingRecordKind::Uninitialized => Vec::new(),
			RollingRecordKind::Range { start, end } =>
			{
				let mut dice = Vec::new();
				if start.max > end.min
				{
					// The range may be empty.
					dice.push(LeastDice::Fixed(ResultSet::of([0])));
				}
				if start.min <= end.max
				{
					dice.push(match start.max >= end.min
					{
						// Every value that may both start and end the range is
						// a range of its own, and every range includes one.
						true => LeastDice::Singletons(ResultSet::interval(
							start.min.max(end.min) as i64,
							start.max.min(end.max) as i64
						)),
						// Every range includes the one from the greatest start
						// to the least end.
						false => LeastDice::Fixed(ResultSet::interval(
							start.max as i64,
							end.min as i64
						))
					});
				}
				dice
			},
			RollingRecordKind::Standard { faces, .. } =>
			{
				let mut dice = Vec::new();
				if faces.min <= 0
				{
					dice.push(LeastDice::Fixed(ResultSet::of([0])));
				}
				if faces.max >= 1
				{
					dice.push(LeastDice::Fixed(ResultSet::interval(
						1,
						faces.min.max(1) as i64
					)));
				}
				dice
			},
			RollingRecordKind::Custom { .. } =>
			{
				vec![LeastDice::Fixed(self.faces())]
			}
		}
	}

	/// Answer the branches of one die of the record, as its roll counts them,
	/// and how many of them show a result that the criterion matches. A die
	/// without faces, or an empty range, has one branch, which shows `0`.
	///
	/// # Parameters
	/// - `criterion`: The results matched.
	///
	/// # Returns
	/// The number of branches and the number of matching branches. A face of
	/// a custom die counts once for every time that it is listed.
	pub(crate) fn branches(&self, criterion: &ResultSet) -> (u128, u128)
	{
		let empty = (1, criterion.contains(0) as u128);
		match self
		{
			RollingRecordKind::Uninitialized => (1, 0),
			RollingRecordKind::Range { start, end } if end.max < start.min =>
			{
				empty
			},
			RollingRecordKind::Range { start, end } =>
			{
				let faces =
					ResultSet::interval(start.min as i64, end.max as i64);
				(faces.len(), faces.intersection(criterion).len())
			},
			RollingRecordKind::Standard { faces, .. } if faces.max <= 0 =>
			{
				empty
			},
			RollingRecordKind::Standard { faces, .. } =>
			{
				let faces = ResultSet::interval(1, faces.max as i64);
				(faces.len(), faces.intersection(criterion).len())
			},
			RollingRecordKind::Custom { faces, .. } if faces.is_empty() =>
			{
				empty
			},
			RollingRecordKind::Custom { faces, .. } =>
			{
				let matching = faces
					.iter()
					.filter(|face| criterion.contains(**face))
					.count();
				(faces.len() as u128, matching as u128)
			}
		}
	}
}

/// A set of results, as the disjoint, ascending intervals that cover it, no
/// two of them adjacent, so that equal sets have equal intervals. The bounds of
/// a reroll describe with these the faces of a die and the results that a
/// predicate matches, either of which may have gaps, as `[1, 9]` or `in [2,
/// 4]` do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ResultSet(Vec<EvaluationBounds>);

impl ResultSet
{
	/// Answer the set of every result from `min` to `max`, inclusive, of
	/// those that fit in an `i32`, or the empty set if there are none.
	///
	/// # Parameters
	/// - `min`: The least result.
	/// - `max`: The greatest result.
	///
	/// # Returns
	/// The requested set.
	pub(crate) fn interval(min: i64, max: i64) -> Self
	{
		let min = min.max(i32::MIN as i64);
		let max = max.min(i32::MAX as i64);
		match min <= max
		{
			true => Self(vec![(min as i32, max as i32).into()]),
			false => Self::default()
		}
	}

	/// Answer the set of every result.
	///
	/// # Returns
	/// The requested set.
	fn all() -> Self { Self(vec![EvaluationBounds::unconstrained()]) }

	/// Answer the set of the specified results.
	///
	/// # Parameters
	/// - `results`: The results, in any order, with or without repetition.
	///
	/// # Returns
	/// The requested set.
	pub(crate) fn of(results: impl IntoIterator<Item = i32>) -> Self
	{
		Self::covering(results.into_iter().map(EvaluationBounds::from))
	}

	/// Answer the set of the results within any of the specified intervals.
	///
	/// # Parameters
	/// - `intervals`: The intervals, in any order, which may overlap.
	///
	/// # Returns
	/// The requested set.
	fn covering(intervals: impl IntoIterator<Item = EvaluationBounds>) -> Self
	{
		let mut intervals = intervals.into_iter().collect::<Vec<_>>();
		intervals.sort_unstable_by_key(|interval| interval.min);
		let mut set = Vec::<EvaluationBounds>::with_capacity(intervals.len());
		for interval in intervals
		{
			match set.last_mut()
			{
				// Merge an interval that overlaps or adjoins the last.
				Some(last) if interval.min as i64 <= last.max as i64 + 1 =>
				{
					last.max = last.max.max(interval.max);
				},
				_ => set.push(interval)
			}
		}
		Self(set)
	}

	/// Answer the union of the receiver and the specified set.
	///
	/// # Parameters
	/// - `other`: The other set.
	///
	/// # Returns
	/// The union.
	pub(crate) fn union(&self, other: &Self) -> Self
	{
		Self::covering(self.0.iter().chain(&other.0).copied())
	}

	/// Answer the intersection of the receiver and the specified set.
	///
	/// # Parameters
	/// - `other`: The other set.
	///
	/// # Returns
	/// The intersection.
	pub(crate) fn intersection(&self, other: &Self) -> Self
	{
		// Each interval of the intersection lies within one interval of each
		// set, and no two such intervals are adjacent, so neither are these.
		let mut set = Vec::new();
		let (mut i, mut j) = (0, 0);
		while let (Some(a), Some(b)) = (self.0.get(i), other.0.get(j))
		{
			let (min, max) = (a.min.max(b.min), a.max.min(b.max));
			if min <= max
			{
				set.push((min, max).into());
			}
			match a.max < b.max
			{
				true => i += 1,
				false => j += 1
			}
		}
		Self(set)
	}

	/// Answer the complement of the receiver, the results that it lacks.
	///
	/// # Returns
	/// The complement.
	fn complement(&self) -> Self
	{
		let mut set = Vec::new();
		let mut next = i32::MIN as i64;
		for interval in &self.0
		{
			if next < interval.min as i64
			{
				set.push((next as i32, interval.min - 1).into());
			}
			next = interval.max as i64 + 1;
		}
		if next <= i32::MAX as i64
		{
			set.push((next as i32, i32::MAX).into());
		}
		Self(set)
	}

	/// Answer the results of the receiver that the specified set lacks.
	///
	/// # Parameters
	/// - `other`: The other set.
	///
	/// # Returns
	/// The difference.
	pub(crate) fn difference(&self, other: &Self) -> Self
	{
		self.intersection(&other.complement())
	}

	/// Answer whether the receiver is empty.
	///
	/// # Returns
	/// `true` if the receiver has no results, `false` otherwise.
	#[inline]
	pub(crate) fn is_empty(&self) -> bool { self.0.is_empty() }

	/// Answer the disjoint, ascending intervals that cover the set, no two of
	/// them adjacent.
	///
	/// # Returns
	/// The intervals.
	#[inline]
	pub(crate) fn intervals(&self) -> &[EvaluationBounds] { &self.0 }

	/// Answer whether the receiver holds the specified result.
	///
	/// # Parameters
	/// - `result`: The result.
	///
	/// # Returns
	/// `true` if the receiver holds the result, `false` otherwise.
	fn contains(&self, result: i32) -> bool
	{
		self.0.iter().any(|interval| interval.contains(result))
	}

	/// Answer whether every result of the receiver belongs to the specified
	/// set.
	///
	/// # Parameters
	/// - `other`: The other set.
	///
	/// # Returns
	/// `true` if the receiver is a subset of `other`, `false` otherwise.
	pub(crate) fn is_subset(&self, other: &Self) -> bool
	{
		self.difference(other).is_empty()
	}

	/// Answer whether the receiver and the specified set share a result.
	///
	/// # Parameters
	/// - `other`: The other set.
	///
	/// # Returns
	/// `true` if the sets meet, `false` otherwise.
	pub(crate) fn meets(&self, other: &Self) -> bool
	{
		!self.intersection(other).is_empty()
	}

	/// Answer the number of results of the receiver.
	///
	/// # Returns
	/// The number of results.
	pub(crate) fn len(&self) -> u128
	{
		self.0
			.iter()
			.map(|interval| {
				(interval.max as i64 - interval.min as i64 + 1) as u128
			})
			.sum()
	}

	/// Answer the least interval that covers the receiver.
	///
	/// # Returns
	/// The hull, or `None` if the receiver is empty.
	pub(crate) fn hull(&self) -> Option<EvaluationBounds>
	{
		match (self.0.first(), self.0.last())
		{
			(Some(first), Some(last)) => Some((first.min, last.max).into()),
			_ => None
		}
	}
}

/// The [least dice](RollingRecordKind::least_dice) of a rolling record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LeastDice
{
	/// A die with these faces.
	Fixed(ResultSet),

	/// Every die with a single face among these.
	Singletons(ResultSet)
}

/// The criteria to which the [predicate](Predicate) of a reroll or explode may
/// resolve, over the bounds of its operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Criteria
{
	/// The results that every criterion matches.
	pub(crate) certain: ResultSet,

	/// The results that some criterion matches.
	pub(crate) possible: ResultSet,

	/// The shape of the criteria, which decides which dice some criterion
	/// matches entirely.
	shape: Shape
}

/// The shape of the [criteria](Criteria) of a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape
{
	/// The criteria include the one that matches every possible result, so that
	/// some criterion matches a set of results exactly when that one does.
	/// Comparisons by order, lists, and ranges are so shaped, since their
	/// criteria grow with their operands in one direction.
	Greatest,

	/// The criteria match the single results within these bounds, one each.
	Equal(EvaluationBounds),

	/// The criteria match every result but one within these bounds, one each.
	NotEqual(EvaluationBounds)
}

impl Criteria
{
	/// Answer the criteria to which the specified predicate may resolve.
	///
	/// # Parameters
	/// - `predicate`: The predicate.
	/// - `value`: The function that answers the bounds of an operand.
	///
	/// # Returns
	/// The requested criteria.
	pub(crate) fn new(
		predicate: &Predicate,
		value: impl Fn(AddressingMode) -> EvaluationBounds
	) -> Self
	{
		let (least, greatest) = (i32::MIN as i64, i32::MAX as i64);
		let nested = |certain, possible| Self {
			certain,
			possible,
			shape: Shape::Greatest
		};
		match predicate
		{
			Predicate::Eq(op) =>
			{
				let t = value(*op);
				Self {
					certain: match t.min == t.max
					{
						true => t.into(),
						false => ResultSet::default()
					},
					possible: t.into(),
					shape: Shape::Equal(t)
				}
			},
			Predicate::Ne(op) =>
			{
				let t = value(*op);
				Self {
					certain: ResultSet::from(t).complement(),
					possible: match t.min == t.max
					{
						true => ResultSet::from(t).complement(),
						false => ResultSet::all()
					},
					shape: Shape::NotEqual(t)
				}
			},
			Predicate::Lt(op) =>
			{
				let t = value(*op);
				nested(
					ResultSet::interval(least, t.min as i64 - 1),
					ResultSet::interval(least, t.max as i64 - 1)
				)
			},
			Predicate::Le(op) =>
			{
				let t = value(*op);
				nested(
					ResultSet::interval(least, t.min as i64),
					ResultSet::interval(least, t.max as i64)
				)
			},
			Predicate::Gt(op) =>
			{
				let t = value(*op);
				nested(
					ResultSet::interval(t.max as i64 + 1, greatest),
					ResultSet::interval(t.min as i64 + 1, greatest)
				)
			},
			Predicate::Ge(op) =>
			{
				let t = value(*op);
				nested(
					ResultSet::interval(t.max as i64, greatest),
					ResultSet::interval(t.min as i64, greatest)
				)
			},
			Predicate::In(values) =>
			{
				let values = ResultSet::of(values.iter().copied());
				nested(values.clone(), values)
			},
			Predicate::InRange { start, end } =>
			{
				let (start, end) = (value(*start), value(*end));
				nested(
					ResultSet::interval(start.max as i64, end.min as i64 - 1),
					ResultSet::interval(start.min as i64, end.max as i64 - 1)
				)
			},
			Predicate::InRangeInclusive { start, end } =>
			{
				let (start, end) = (value(*start), value(*end));
				nested(
					ResultSet::interval(start.max as i64, end.min as i64),
					ResultSet::interval(start.min as i64, end.max as i64)
				)
			}
		}
	}

	/// Answer whether some criterion matches every face of some die among the
	/// specified least dice.
	///
	/// # Parameters
	/// - `dice`: The least dice.
	///
	/// # Returns
	/// `true` if some criterion matches every face of such a die, `false`
	/// otherwise.
	pub(crate) fn may_cover(&self, dice: &LeastDice) -> bool
	{
		match (self.shape, dice)
		{
			(Shape::Greatest, LeastDice::Fixed(faces)) =>
			{
				faces.is_subset(&self.possible)
			},
			// A criterion that matches a single face matches its die entirely,
			// so some criterion does if some face is possibly matched.
			(
				Shape::Greatest | Shape::Equal(_),
				LeastDice::Singletons(faces)
			) => faces.meets(&self.possible),
			// A criterion matches a single result, so it matches a die entirely
			// only if the die has a single face.
			(Shape::Equal(_), LeastDice::Fixed(faces)) =>
			{
				faces.len() == 1 && faces.is_subset(&self.possible)
			},
			// A criterion matches every result but its threshold, so it matches
			// a die entirely if its threshold is not a face of the die.
			(Shape::NotEqual(t), LeastDice::Fixed(faces)) =>
			{
				!ResultSet::from(t).is_subset(faces)
			},
			(Shape::NotEqual(t), LeastDice::Singletons(faces)) =>
			{
				let t = ResultSet::from(t);
				!(t.len() == 1 && *faces == t)
			}
		}
	}
}

impl From<EvaluationBounds> for ResultSet
{
	fn from(bounds: EvaluationBounds) -> Self
	{
		Self::interval(bounds.min as i64, bounds.max as i64)
	}
}

impl From<&Criterion> for ResultSet
{
	fn from(criterion: &Criterion) -> Self
	{
		let within = |range: &RangeInclusive<i32>| {
			Self::interval(*range.start() as i64, *range.end() as i64)
		};
		match criterion
		{
			Criterion::Within(range) => within(range),
			Criterion::Outside(range) => within(range).complement(),
			Criterion::Among(values) => Self::of(values.iter().copied())
		}
	}
}

/// The greatest number of pairs of a die and a criterion that the bounds of an
/// [explode](crate::Explode) will enumerate in pursuit of an exact answer.
/// Beyond this they approximate. Every die of a chain is the same die, and
/// every one is matched by the same criterion, which the union of the faces of
/// every die, and of the results that every criterion matches, forgets; so the
/// bounds explode each pair alone, and unite the answers. The budget is
/// generous enough to cover two operands that range over a few dozen values
/// each, while bounding the work at a trivial cost paid once per explode
/// during bounds analysis.
const EXPLODE_ENUMERATION_BUDGET: u128 = 1024;

/// The number of middle dice of a [chain](crate::RollingRecord::chains) that
/// stands for a chain without bound: many enough that the total saturates,
/// unless every middle die shows `0`, since a nonzero die moves the total by at
/// least `1`, and an `i32` spans fewer than `2^33` values.
const UNBOUNDED_DICE: i128 = 1 << 33;

/// A die that a rolling record may roll, as the bounds of an
/// [explode](crate::Explode) enumerate them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DieFaces
{
	/// The faces of the die. A die without faces, or an empty range, shows
	/// only the face `0`.
	faces: ResultSet,

	/// The value of a [saturated](Origin::Saturates) die of the die.
	saturated: i32
}

impl RollingRecordKind<EvaluationBounds>
{
	/// Answer every die that the bounds of the faces or endpoints of the record
	/// admit, without repetition, if there are no more than `budget` of them.
	///
	/// # Parameters
	/// - `budget`: The greatest number of dice to enumerate.
	///
	/// # Returns
	/// The dice, which are none if the record is uninitialized, or `None` if
	/// the bounds admit more than `budget` of them.
	fn dice_within(&self, budget: u128) -> Option<Vec<DieFaces>>
	{
		let faceless = DieFaces {
			faces: ResultSet::of([0]),
			saturated: 0
		};
		let width = |bounds: &EvaluationBounds| {
			(bounds.max as i64 - bounds.min as i64 + 1) as u128
		};
		match self
		{
			RollingRecordKind::Uninitialized => Some(Vec::new()),
			RollingRecordKind::Standard { faces, .. } =>
			{
				// Every count of faces that is not positive answers the same
				// die, which has no faces.
				let least = faces.min.max(1);
				let count = (faces.max as i64 - least as i64 + 1).max(0)
					as u128
					+ (faces.min <= 0) as u128;
				if count > budget
				{
					return None
				}
				let mut dice = Vec::with_capacity(count as usize);
				if faces.min <= 0
				{
					dice.push(faceless);
				}
				dice.extend((least..=faces.max).map(|faces| {
					DieFaces {
						faces: ResultSet::interval(1, faces as i64),
						saturated: RollingRecordKind::Standard {
							count: 0,
							faces
						}
						.saturated_value()
					}
				}));
				Some(dice)
			},
			RollingRecordKind::Range { start, end } =>
			{
				if width(start).saturating_mul(width(end)) > budget
				{
					return None
				}
				let mut dice = Vec::new();
				for start in start.min..=start.max
				{
					for end in end.min..=end.max
					{
						let die = match end < start
						{
							true => faceless.clone(),
							false => DieFaces {
								faces: ResultSet::interval(
									start as i64,
									end as i64
								),
								saturated: RollingRecordKind::Range {
									start,
									end
								}
								.saturated_value()
							}
						};
						if !dice.contains(&die)
						{
							dice.push(die);
						}
					}
				}
				Some(dice)
			},
			RollingRecordKind::Custom { faces, .. } => Some(vec![DieFaces {
				faces: self.faces(),
				saturated: RollingRecordKind::Custom {
					count: 0,
					faces: faces.clone()
				}
				.saturated_value()
			}])
		}
	}

	/// Answer the values that a [saturated](Origin::Saturates) die of the
	/// record may take, over the bounds of its faces or endpoints: `i32::MAX`
	/// or `i32::MIN` by the sign of the mean face value of some die of the
	/// record, or `0` if that mean is `0`.
	///
	/// # Returns
	/// The values, which cover every value that a saturated die may take. They
	/// are none if the record is uninitialized.
	fn saturated_values(&self) -> ResultSet
	{
		// Answer the values by the signs that a mean may take.
		let signs = |least: i64, greatest: i64| {
			[
				(least < 0, i32::MIN),
				(least <= 0 && 0 <= greatest, 0),
				(greatest > 0, i32::MAX)
			]
			.into_iter()
			.filter(|(possible, _)| *possible)
			.map(|(_, value)| value)
			.collect::<Vec<_>>()
		};
		match self
		{
			RollingRecordKind::Uninitialized => ResultSet::default(),
			// A die without faces has mean `0`, and every other die a positive
			// mean.
			RollingRecordKind::Standard { faces, .. } =>
			{
				ResultSet::of(signs(faces.min.min(1) as i64, faces.max as i64))
			},
			RollingRecordKind::Range { start, end } =>
			{
				// The sign of the mean of a range is the sign of the sum of its
				// endpoints, and an empty range shows only `0`.
				let mut values = Vec::new();
				if start.max > end.min
				{
					values.push(0);
				}
				if start.min <= end.max
				{
					values.extend(signs(
						start.min as i64 + end.min as i64,
						start.max as i64 + end.max as i64
					));
				}
				ResultSet::of(values)
			},
			RollingRecordKind::Custom { faces, .. } =>
			{
				ResultSet::of([RollingRecordKind::Custom {
					count: 0,
					faces: faces.clone()
				}
				.saturated_value()])
			},
		}
	}
}

/// Answer every criterion to which the specified predicate may resolve, over
/// the bounds of its operands, without repetition, as the sets of results that
/// they match, if their operands take no more than `budget` assignments.
///
/// # Parameters
/// - `predicate`: The predicate.
/// - `value`: The function that answers the bounds of an operand.
/// - `budget`: The greatest number of assignments of the operands to enumerate.
///
/// # Returns
/// The criteria, or `None` if the operands take more than `budget`
/// assignments.
fn criteria_within(
	predicate: &Predicate,
	value: impl Fn(AddressingMode) -> EvaluationBounds,
	budget: u128
) -> Option<Vec<ResultSet>>
{
	// An operand that occurs twice takes the same value at each occurrence.
	let mut operands = Vec::<AddressingMode>::new();
	for op in predicate.operands()
	{
		if !operands.contains(&op)
		{
			operands.push(op);
		}
	}
	let bounds = operands.iter().map(|op| value(*op)).collect::<Vec<_>>();
	let count = bounds.iter().try_fold(1u128, |count, bounds| {
		count.checked_mul((bounds.max as i64 - bounds.min as i64 + 1) as u128)
	})?;
	if count > budget
	{
		return None
	}
	// Count through the assignments, as an odometer does.
	let mut assignment =
		bounds.iter().map(|bounds| bounds.min).collect::<Vec<_>>();
	let mut criteria = Vec::new();
	loop
	{
		let criterion = predicate.resolve(|op| {
			assignment[operands.iter().position(|o| *o == op).unwrap()]
		});
		let criterion = ResultSet::from(&criterion);
		if !criteria.contains(&criterion)
		{
			criteria.push(criterion);
		}
		let mut digit = 0;
		loop
		{
			if digit == assignment.len()
			{
				return Some(criteria)
			}
			if assignment[digit] < bounds[digit].max
			{
				assignment[digit] += 1;
				break
			}
			assignment[digit] = bounds[digit].min;
			digit += 1;
		}
	}
}

/// An [explode](crate::Explode) of the chains still kept by a rolling record
/// that no explode has extended yet, each of which holds one kept die, over
/// the dice and criteria that it may resolve to, for the purpose of bounding
/// the totals of the chains that it grows.
#[derive(Debug)]
struct Explosion<'a>
{
	/// The results that a die still kept may show.
	results: &'a ResultSet,

	/// The faces that an extension may show.
	faces: &'a ResultSet,

	/// The results that some criterion matches.
	possible: &'a ResultSet,

	/// The results that every criterion matches.
	certain: &'a ResultSet,

	/// The values that a saturated chain may take, which are none if no chain
	/// may saturate.
	saturated: &'a ResultSet,

	/// The bounds of the limit, or `None` for no limit.
	limit: Option<EvaluationBounds>
}

impl Explosion<'_>
{
	/// Answer the totals that a chain still kept may show after the explode.
	///
	/// # Returns
	/// The totals, which cover every total that a chain may show, and whose
	/// hull is exact if the die and criterion are each one, but which may fill
	/// the gaps between them.
	///
	/// # Notes
	/// A chain that grows from a matching die `r` holds `r`, then `m` middle
	/// dice that match, and then a last die: one that does not match, ending
	/// the chain, or, if the limit is spent, any face. A chain whose die cannot
	/// match keeps its die alone. The total of a chain folds its dice in
	/// ascending order, saturating, which never decreases as any die grows,
	/// so the least total of a chain of such a shape shows the least result
	/// at each die, and the greatest, the greatest. Nor does it ever move
	/// against the sign of a middle die as `m` grows, so the extreme totals
	/// over a range of `m` lie at its ends.
	///
	/// Under a limit `L`, the last die ends the chain after `1` to `L - 1`
	/// extensions, so `m` ranges from `0` to `L - 2`; or the limit is spent
	/// after `L` extensions, so `m` is `L - 1`, and the last die shows any
	/// face. A chain that ends after `L - 1` extensions is never more extreme
	/// than both the chain that ends after one and the chain whose limit is
	/// spent, which holds one middle die more and a last die that may show any
	/// face: if the middle dice move the total toward an extreme, the spent
	/// chain is at least as extreme, and otherwise the short one. A limit of
	/// zero or less extends nothing, leaving `r` alone. The extreme totals over
	/// a range of limits lie at its ends, too. Without limit, the last die ends
	/// the chain after any number of extensions, so `m` grows without bound,
	/// unless every face matches, whence the chain saturates instead.
	fn totals(&self) -> ResultSet
	{
		// A die that the criterion does not match stays alone.
		let mut totals = self.results.difference(self.certain);
		let Some(starts) = self.results.intersection(self.possible).hull()
		else
		{
			return totals
		};
		totals = totals.union(self.saturated);
		// Every result lies among the faces, so some face matches whenever
		// some result does.
		let middles = self
			.faces
			.intersection(self.possible)
			.hull()
			.unwrap_or(starts);
		let faces = self.faces.hull().unwrap_or(starts);
		let ends = self.faces.difference(self.certain).hull();
		// Each shape of chain: the number of middle dice, and the results of
		// the last die, if the chain has one.
		let mut shapes = Vec::<(i128, Option<EvaluationBounds>)>::new();
		match self.limit
		{
			None =>
			{
				if let Some(ends) = ends
				{
					shapes.push((0, Some(ends)));
					shapes.push((UNBOUNDED_DICE, Some(ends)));
				}
			},
			Some(limit) =>
			{
				let (least, greatest) =
					(limit.min.max(0) as i128, limit.max.max(0) as i128);
				if least == 0
				{
					shapes.push((0, None));
				}
				if let Some(ends) = ends
					&& greatest >= 2
				{
					shapes.push((0, Some(ends)));
				}
				if greatest >= 1
				{
					for limit in [least.max(1), greatest]
					{
						shapes.push((limit - 1, Some(faces)));
					}
				}
			}
		}
		for (count, last) in shapes
		{
			let least = fold(
				[(starts.min, 1), (middles.min, count)]
					.into_iter()
					.chain(last.map(|last| (last.min, 1)))
			);
			let greatest = fold(
				[(starts.max, 1), (middles.max, count)]
					.into_iter()
					.chain(last.map(|last| (last.max, 1)))
			);
			totals = totals
				.union(&ResultSet::interval(least as i64, greatest as i64));
		}
		totals
	}
}

/// Fold the specified dice in ascending order, saturating, as the total of a
/// [chain](crate::RollingRecord::chains) does, in time independent of their
/// number.
///
/// # Parameters
/// - `dice`: The results of the dice, each with the number of dice that show
///   it, in any order.
///
/// # Returns
/// The total.
///
/// # Notes
/// The negative results come first, and only fall, so the fold saturates at
/// `i32::MIN` only if they sum below it, and then stays there until the rest
/// arrive; the rest only rise, so the fold saturates at `i32::MAX` only if
/// they lift it above.
fn fold(dice: impl IntoIterator<Item = (i32, i128)>) -> i32
{
	let (mut negative, mut positive) = (0i128, 0i128);
	for (result, count) in dice
	{
		match result < 0
		{
			true => negative += result as i128 * count,
			false => positive += result as i128 * count
		}
	}
	(negative.max(i32::MIN as i128) + positive).min(i32::MAX as i128) as i32
}

impl RollingRecordBounds
{
	/// Answer the totals that a chain still kept may show after the specified
	/// [explode](crate::Explode) of the record, which no explode has extended
	/// yet.
	///
	/// # Parameters
	/// - `predicate`: The predicate of the explode.
	/// - `value`: The function that answers the bounds of an operand.
	/// - `limit`: The bounds of the limit, or `None` for no limit.
	///
	/// # Returns
	/// The totals, which cover every total that a chain may show, and whose
	/// hull is exact if the dice of the record and the criteria of the
	/// predicate are few enough to enumerate, and the results that a die
	/// still kept may show are all that the bounds admit of each die.
	///
	/// # Notes
	/// If the dice and criteria are few enough, explode each pair alone, and
	/// unite the answers. Otherwise, explode every die and criterion at once,
	/// which is sound, since each set is a superset of each of its members,
	/// but may be loose, since it mixes the faces of one die, or the results
	/// that one criterion matches, with those of another.
	fn explode(
		&self,
		predicate: &Predicate,
		value: impl Fn(AddressingMode) -> EvaluationBounds,
		limit: Option<EvaluationBounds>
	) -> ResultSet
	{
		let pairs = self.kind.dice_within(EXPLODE_ENUMERATION_BUDGET).and_then(
			|dice| {
				let budget =
					EXPLODE_ENUMERATION_BUDGET / dice.len().max(1) as u128;
				Some((dice, criteria_within(predicate, &value, budget)?))
			}
		);
		match pairs
		{
			Some((dice, criteria)) =>
			{
				let mut totals = ResultSet::default();
				for die in &dice
				{
					let results = self.results.intersection(&die.faces);
					for criterion in &criteria
					{
						// Without limit, a die whose faces all match saturates.
						let saturated = match limit.is_none()
							&& die.faces.is_subset(criterion)
						{
							true => ResultSet::of([die.saturated]),
							false => ResultSet::default()
						};
						let explosion = Explosion {
							results: &results,
							faces: &die.faces,
							possible: criterion,
							certain: criterion,
							saturated: &saturated,
							limit
						};
						totals = totals.union(&explosion.totals());
					}
				}
				totals
			},
			None =>
			{
				let criteria = Criteria::new(predicate, &value);
				// Without limit, a die may saturate if some criterion matches
				// every face of it.
				let saturated = match limit.is_none()
					&& self
						.kind
						.least_dice()
						.iter()
						.any(|die| criteria.may_cover(die))
				{
					true => self.kind.saturated_values(),
					false => ResultSet::default()
				};
				Explosion {
					results: &self.results,
					faces: &self.kind.faces(),
					possible: &criteria.possible,
					certain: &criteria.certain,
					saturated: &saturated,
					limit
				}
				.totals()
			}
		}
	}
}

/// The bounds of a function, as computed by a bounds evaluator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bounds
{
	/// The minimum and maximum values of the function.
	pub value: EvaluationBounds,

	/// The number of equally likely outcomes, counting every branch of every
	/// range and roll, which is the [total](crate::Distribution::total) weight
	/// of the exact distribution, not the number of [distinct
	/// values](crate::Distribution::len); or `None` if the count could not be
	/// computed efficiently because of dynamic range or dice expressions,
	/// i.e., when the count or faces are themselves determined by a range
	/// or dice expression, or if any binding supplied to
	/// [`bounds_over`](Evaluator::bounds_over) is not a single value.
	///
	/// A [reroll](crate::Reroll) counts its branches as though each die still
	/// kept rolled in advance every reroll that it might need, which makes
	/// every path equally likely. A reroll limited to `n` times multiplies the
	/// count by the branches of its die, raised to `n`, for every die still
	/// kept, and an unlimited reroll by the branches of its die that its
	/// predicate does not match, for every die still kept, since the die
	/// finally shows one of them whether or not it first matched. A reroll
	/// that cannot happen, because no face matches, the limit is not
	/// positive, or, without limit, every face matches, counts nothing.
	/// The count is `None` if the predicate or limit of a reroll, or the
	/// number of dice still kept that a reroll that can happen rerolls, is
	/// determined by a range or dice expression.
	///
	/// An [explode](crate::Explode) counts its branches likewise, as though
	/// each chain still kept rolled in advance every extension that it might
	/// need. An explode limited to `n` times multiplies the count by the
	/// branches of its die, raised to `n`, for every chain still kept, even if
	/// no die still kept may match. An explode that cannot happen, because no
	/// face matches or the limit is not positive, or that saturates, because,
	/// without limit, every face matches, counts nothing. The count is `None`
	/// if the predicate or limit of an explode that can happen, or the number
	/// of chains still kept that it extends, is determined by a range or dice
	/// expression; if an unlimited explode may roll, since its paths are
	/// endless; and once a reroll or explode may act on a record that an
	/// explode may have extended, since the number of dice of a chain varies.
	pub count: Option<u128>,

	/// An upper bound on the number of dice that the function may roll over
	/// the bindings, where a range counts as one die, saturating at
	/// [`u64::MAX`]. No evaluation over the bindings rolls more, so no
	/// [metered](Evaluator::evaluate_metered) evaluation with at least this
	/// budget can exhaust it, and a caller can compare it against a budget
	/// without evaluating anything. Unlike the outcome count, it is available
	/// even for dynamic range and roll expressions, whose counts are bounded
	/// like any other value. The bound is computed by interval arithmetic, so,
	/// like [`value`](Self::value), it may over-approximate, e.g.,
	/// `({a}@(1D6) + -{a})D6` always rolls exactly one die, but its bound is
	/// `6`. A limited [explode](crate::Explode) may roll its limit for every
	/// chain still kept. An unlimited explode that may roll a die, unless it
	/// saturates, or a reroll that may act on a record that an explode may
	/// have extended, saturates the bound.
	pub dice: u64
}

impl Display for Bounds
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		write!(
			f,
			"value ∈ [{}], count = {}, dice = {}",
			self.value,
			match self.count
			{
				Some(count) => count.to_string(),
				None => "None".to_string()
			},
			self.dice
		)
	}
}

impl<'eval> From<BoundsEvaluator<'eval>> for Bounds
{
	fn from(evaluator: BoundsEvaluator<'eval>) -> Self
	{
		Self {
			value: evaluator.result,
			count: evaluator.count,
			dice: evaluator.dice
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
	pub(crate) fn new(function: &'eval Function) -> Self
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
			count: Some(1),
			dice: 0
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
	/// The bounds of the function, covering the value, the number of possible
	/// outcomes, and the number of dice rolled. The outcome count is `None`
	/// whenever any binding is non-degenerate, as it would otherwise be an
	/// exact-looking number that holds for only one member of the interval.
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
		let degenerate = self.seed(args, externals, unsupplied)?;
		for instruction in &self.function.instructions
		{
			self.step(instruction);
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

	/// Seed the registers of the bindings with their intervals, so that the
	/// evaluator may [step](Self::step) through the instructions.
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
	/// `true` if every binding is a single value, `false` otherwise.
	///
	/// # Errors
	/// [`BadArity`](EvaluationError::BadArity) if the number of arguments
	/// provided disagrees with the number of formal parameters in the function
	/// signature.
	pub(crate) fn seed<'error>(
		&mut self,
		args: impl IntoIterator<Item = Option<EvaluationBounds>>,
		externals: impl IntoIterator<Item = (usize, EvaluationBounds)>,
		unsupplied: EvaluationBounds
	) -> Result<bool, EvaluationError<'error>>
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
		Ok(degenerate)
	}

	/// Bound the effect of the specified instruction, the next one of the
	/// function, on the registers and rolling records.
	///
	/// # Parameters
	/// - `instruction`: The instruction.
	pub(crate) fn step(&mut self, instruction: &Instruction)
	{
		instruction.visit(self).unwrap();
		self.pc.allocate();
	}

	/// Narrow the interval of the specified register to its intersection with
	/// the specified interval, as when the caller knows that the register's
	/// value lies within it, so that the steps that follow bound only such
	/// values.
	///
	/// # Parameters
	/// - `reg`: The index of the register.
	/// - `bounds`: The interval, which meets the register's.
	pub(crate) fn narrow(
		&mut self,
		reg: RegisterIndex,
		bounds: EvaluationBounds
	)
	{
		let current = &mut self.registers[reg.0];
		debug_assert!(
			bounds.min <= current.max && current.min <= bounds.max,
			"the intervals meet"
		);
		*current =
			(current.min.max(bounds.min), current.max.min(bounds.max)).into();
	}

	/// Answer the bounds of the registers, as of the last
	/// [step](Self::step).
	///
	/// # Returns
	/// The bounds of each register, by index.
	#[inline]
	pub(crate) fn registers(&self) -> &[EvaluationBounds] { &self.registers }

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

	/// Answer the kind of the roll that last filled the specified rolling
	/// record, with the bounds of its operands, which describes the die from
	/// which every reroll of the record draws.
	///
	/// # Parameters
	/// - `op`: The rolling record.
	///
	/// # Returns
	/// The kind.
	#[inline]
	pub(crate) fn roll(
		&self,
		op: RollingRecordIndex
	) -> &RollingRecordKind<EvaluationBounds>
	{
		&self.record(op).kind
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

	/// Narrow the number of dice still kept by a rolling record by a drop. Only
	/// the number of dice dropped matters to the bounds, not the end from
	/// which they are dropped.
	///
	/// # Parameters
	/// - `dest`: The rolling record.
	/// - `count`: The number of dice to drop. A count of zero or less drops
	///   nothing.
	fn drop(&mut self, dest: RollingRecordIndex, count: AddressingMode)
	{
		let count = self.value(count).clamp(0.into(), i32::MAX.into());
		let record = self.record_mut(dest);
		record.kept = (record.kept - count).max(0.into());
	}

	/// Narrow the number of dice still kept by a rolling record by a keep,
	/// which keeps no more than `count` of them. Only the number of dice kept
	/// matters to the bounds, not the end from which they are kept.
	///
	/// # Parameters
	/// - `dest`: The rolling record.
	/// - `count`: The number of dice to keep. A count of zero or less keeps
	///   nothing.
	fn keep(&mut self, dest: RollingRecordIndex, count: AddressingMode)
	{
		let count = self.value(count);
		let record = self.record_mut(dest);
		record.kept = record.kept.min(count).max(0.into());
	}

	/// Multiply the number of outcomes by the number of branches of a range or
	/// roll, whether or not its rolling record is ever summed, since the total
	/// weight of the exact [distribution](crate::Distribution) counts every
	/// branch of every range and roll. Leaves an unknown number of outcomes
	/// unknown.
	///
	/// # Parameters
	/// - `branches`: The number of branches, at least `1`.
	#[inline]
	fn count_branches(&mut self, branches: u128)
	{
		self.count = self.count.map(|count| count.saturating_mul(branches));
	}

	/// Multiply the number of outcomes by the number of branches of an
	/// [explode](crate::Explode) of a record that no explode has extended yet,
	/// counting the branches of each chain still kept as though every extension
	/// that it might need were rolled in advance, as for a limited
	/// [reroll](crate::Reroll), which makes every path equally likely: a
	/// limited explode rolls each die up to the limit more times, whatever it
	/// shows. An unlimited explode that may roll has no such bound, so it gives
	/// up counting, but one whose predicate matches every face saturates,
	/// rolling nothing. An explode that cannot happen, because no face matches
	/// or the limit is not positive, adds no branches.
	///
	/// # Parameters
	/// - `kept`: The bounds of the number of chains still kept.
	/// - `limit`: The bounds of the limit, or `None` for no limit.
	/// - `branches`: The number of branches of one die of the record.
	/// - `matching`: The number of those branches that the predicate matches.
	fn count_explode(
		&mut self,
		kept: EvaluationBounds,
		limit: Option<EvaluationBounds>,
		branches: u128,
		matching: u128
	)
	{
		match limit
		{
			Some(limit) if matching > 0 =>
			{
				let branches = branches.saturating_pow(limit.max.max(0) as u32);
				if branches > 1
				{
					match kept.min == kept.max
					{
						true => self.count_branches(
							branches.saturating_pow(kept.max as u32)
						),
						// The number of chains extended depends on a previous
						// range or roll, so the paths through them are
						// unequally likely.
						false => self.count = None
					}
				}
			},
			None if 0 < matching && matching < branches => self.count = None,
			_ =>
			{}
		}
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
		record.kept = 1.into();
		record.results = record.kind.faces();
		record.totals = record.results.clone();
		record.exploded = false;
		self.dice = self.dice.saturating_add(1);
		if self.sum(inst.start) || self.sum(inst.end)
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		// A range has one branch per integer within it, or one branch, which
		// answers 0, if it is empty.
		let branches = (end.max as i128)
			.saturating_sub(start.min as i128)
			.saturating_add(1)
			.max(1) as u128;
		self.count_branches(branches);
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
		record.kept = count.max(0.into());
		record.results = record.kind.faces();
		record.totals = record.results.clone();
		record.exploded = false;
		self.dice = self.dice.saturating_add(dice(count.max));
		if self.sum(inst.count) || self.sum(inst.faces)
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		// Each die has one branch per face, or one branch, which answers 0, if
		// it has no faces.
		let branches = (faces.max.max(0) as u128)
			.saturating_pow(count.max.max(0) as u32)
			.max(1);
		self.count_branches(branches);
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
		record.kept = count.max(0.into());
		record.results = record.kind.faces();
		record.totals = record.results.clone();
		record.exploded = false;
		self.dice = self.dice.saturating_add(dice(count.max));
		if self.sum(inst.count)
		{
			// The count depends on a previous range or roll, so disable
			// counting for the remainder of the function.
			self.count = None;
		}
		let branches = (inst.faces.len() as u128)
			.saturating_pow(count.max.max(0) as u32)
			.max(1);
		self.count_branches(branches);
		Ok(())
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		self.drop(inst.dest, inst.count);
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		self.drop(inst.dest, inst.count);
		Ok(())
	}

	fn visit_keep_lowest(&mut self, inst: &KeepLowest) -> Result<(), ()>
	{
		self.keep(inst.dest, inst.count);
		Ok(())
	}

	fn visit_keep_highest(&mut self, inst: &KeepHighest) -> Result<(), ()>
	{
		self.keep(inst.dest, inst.count);
		Ok(())
	}

	fn visit_reroll(&mut self, inst: &Reroll) -> Result<(), ()>
	{
		let criteria = Criteria::new(&inst.predicate, |op| self.value(op));
		let limit = inst.limit.operand().map(|count| self.value(count));
		if inst.operands().into_iter().any(|op| self.sum(op))
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		let record = self.record(inst.dest);
		let kept = record.kept;
		let (branches, matching) = record.kind.branches(&criteria.certain);
		let faces = record.kind.faces();
		let least_dice = record.kind.least_dice();
		// A die still kept may be rerolled only if some criterion matches some
		// result that it may show.
		let may_match =
			kept.max > 0 && record.results.meets(&criteria.possible);
		if record.exploded
		{
			// A chain may hold any number of dice, each of which the reroll
			// rerolls alone, so give up counting and bound the dice loosely.
			// A replacement may show any face, and the reroll may cross out
			// every die of a chain, or none, so a total may be any value.
			if may_match
			{
				self.count = None;
				self.dice = u64::MAX;
				let record = self.record_mut(inst.dest);
				record.kept = (0, kept.max).into();
				record.results = faces.union(&record.results);
				record.totals = ResultSet::all();
			}
			return Ok(())
		}
		// Count the branches of each die still kept as though every reroll that
		// it might need were rolled in advance, which makes every path equally
		// likely: a limited reroll rolls each die up to the limit more times,
		// whatever it shows, and an unlimited one rolls each die once more,
		// from the faces that do not match, since the die finally shows one of
		// these whether or not it first matched. A reroll that cannot happen,
		// because no face matches, the limit is not positive, or, without
		// limit, every face matches, adds no branches.
		let branches = match limit
		{
			Some(limit) if matching > 0 =>
			{
				branches.saturating_pow(limit.max.max(0) as u32)
			},
			None if 0 < matching && matching < branches => branches - matching,
			_ => 1
		};
		if branches > 1
		{
			match kept.min == kept.max
			{
				true => self
					.count_branches(branches.saturating_pow(kept.max as u32)),
				// The number of dice rerolled depends on a previous range or
				// roll, so the paths through them are unequally likely.
				false => self.count = None
			}
		}
		// Bound the dice still kept and their results.
		let record = self.record_mut(inst.dest);
		let rerolls = match limit
		{
			Some(limit) => match may_match && limit.max > 0
			{
				// A replacement may show any face of the die, and may match
				// again once the limit is spent. Each die still kept may be
				// rerolled up to the limit.
				true =>
				{
					record.results = faces;
					dice(kept.max).saturating_mul(dice(limit.max))
				},
				false => 0
			},
			None =>
			{
				// Without limit, a die finally shows a face that no criterion
				// matches, if there is one; if every face matches, the reroll
				// crosses out every die still kept instead, rolling nothing.
				let rest = faces.difference(&criteria.certain);
				if rest.is_empty()
				{
					record.kept = 0.into();
				}
				else if least_dice.iter().any(|die| criteria.may_cover(die))
				{
					record.kept = (0, kept.max).into();
				}
				match may_match
				{
					true =>
					{
						// A die may be rerolled any number of times, unless
						// every face matches.
						let rerolls = match rest.is_empty()
						{
							true => 0,
							false => u64::MAX
						};
						record.results = rest;
						rerolls
					},
					false => 0
				}
			}
		};
		// Each chain still kept holds its one kept die.
		record.totals = record.results.clone();
		self.dice = self.dice.saturating_add(rerolls);
		Ok(())
	}

	fn visit_explode(&mut self, inst: &Explode) -> Result<(), ()>
	{
		let criteria = Criteria::new(&inst.predicate, |op| self.value(op));
		let limit = inst.limit.operand().map(|count| self.value(count));
		if inst.operands().into_iter().any(|op| self.sum(op))
		{
			// At least one of the operands depends on a previous range or
			// roll, so disable counting for the remainder of the function.
			self.count = None;
		}
		let record = self.record(inst.dest);
		let kept = record.kept;
		let exploded = record.exploded;
		let (branches, matching) = record.kind.branches(&criteria.certain);
		let faces = record.kind.faces();
		// A die still kept may be exploded only if some criterion matches some
		// result that it may show, and the limit may be positive.
		let may_match =
			kept.max > 0 && record.results.meets(&criteria.possible);
		let may_explode = may_match && limit.is_none_or(|limit| limit.max > 0);
		let totals = match (exploded, may_explode)
		{
			// A chain may hold any number of dice, which the bounds do not
			// track, so give up counting and bound the totals loosely.
			(true, true) =>
			{
				self.count = None;
				ResultSet::all()
			},
			(true, false) => return Ok(()),
			// Count the branches that the explode would roll, even if no die
			// still kept may match.
			(false, false) =>
			{
				self.count_explode(kept, limit, branches, matching);
				return Ok(())
			},
			(false, true) =>
			{
				let totals =
					record.explode(&inst.predicate, |op| self.value(op), limit);
				self.count_explode(kept, limit, branches, matching);
				totals
			}
		};
		// Each chain still kept grows only from its end, so a limited explode
		// extends each up to the limit. Without limit, a chain saturates,
		// rolling nothing, if every face matches; otherwise it may grow
		// without bound.
		let extensions = match limit
		{
			Some(limit) => dice(kept.max).saturating_mul(dice(limit.max)),
			None => match faces.difference(&criteria.certain).is_empty()
			{
				true => 0,
				false => u64::MAX
			}
		};
		self.dice = self.dice.saturating_add(extensions);
		// An extension may show any face.
		let record = self.record_mut(inst.dest);
		record.results = faces.union(&record.results);
		record.totals = totals;
		record.exploded = true;
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		let record = self.record(inst.src);
		// A record that was never rolled is empty, so it sums to zero, as the
		// evaluator's does. Zero is fixed, not the result of a roll.
		if matches!(record.kind, RollingRecordKind::Uninitialized)
		{
			self.set_register(inst.dest, 0);
			self.set_sum(inst.dest, false);
			return Ok(())
		}
		// Each chain still kept shows one of the totals that the record admits,
		// among them `0` for a die without faces or an empty range. No total
		// is admitted only if a reroll crossed out every die. The sum folds
		// the totals in ascending order, saturating, which never decreases
		// as any total grows, so it is least when every chain shows the least
		// total, and greatest when every chain shows the greatest; and the
		// sum of `n` equal totals is their saturated product by `n`.
		let sum = match record.totals.hull()
		{
			Some(totals) => record.kept * totals,
			None => 0.into()
		};
		self.set_register(inst.dest, sum);
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
		let product = match inst.op1
		{
			// When both operands are the same register, as the strength reducer
			// makes them when it rewrites `x ^ 2` as `x * x`, the product is a
			// square, which interval multiplication would widen across zero.
			// Bound it as the square that it is.
			AddressingMode::Register(_) if inst.op1 == inst.op2 =>
			{
				op1.exp(2.into())
			},
			_ => op1 * op2
		};
		self.set_register(inst.dest, product);
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

	fn visit_max(&mut self, inst: &Max) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1.max(op2));
		self.set_sum(inst.dest, self.sum(inst.op1) || self.sum(inst.op2));
		Ok(())
	}

	fn visit_min(&mut self, inst: &Min) -> Result<(), ()>
	{
		let op1 = self.value(inst.op1);
		let op2 = self.value(inst.op2);
		self.set_register(inst.dest, op1.min(op2));
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

	/// Compute the bounds of the maximum of two bounds. The greater of two
	/// values is monotonic in each, so its bounds are the greater of the lower
	/// bounds and the greater of the upper bounds.
	///
	/// # Parameters
	/// - `rhs`: The other bounds.
	///
	/// # Returns
	/// The bounds of the result.
	pub(crate) fn max(self, rhs: EvaluationBounds) -> Self
	{
		Self {
			min: self.min.max(rhs.min),
			max: self.max.max(rhs.max)
		}
	}

	/// Compute the bounds of the minimum of two bounds. The lesser of two
	/// values is monotonic in each, so its bounds are the lesser of the lower
	/// bounds and the lesser of the upper bounds.
	///
	/// # Parameters
	/// - `rhs`: The other bounds.
	///
	/// # Returns
	/// The bounds of the result.
	pub(crate) fn min(self, rhs: EvaluationBounds) -> Self
	{
		Self {
			min: self.min.min(rhs.min),
			max: self.max.min(rhs.max)
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
