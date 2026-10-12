//! # Evaluation
//!
//! Evaluation is the process of determining the value of a dice expression. It
//! requires a compiled [function](Function) and a pseudo-random number
//! generator. Evaluation produces the final result of an expression and the
//! individual outcomes of each dice subexpression.

use std::{
	borrow::Cow,
	cmp::Ordering,
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
	dice: u64,

	/// Whether some explode without limit, whose predicate does not match
	/// every face, may extend a chain, which may then grow without bound.
	unbounded: bool
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

	/// The chains still kept, by the die that rolled them: one entry for each
	/// die that the bounds of the faces or endpoints of the record admit, once
	/// a reroll or explode [splits](Self::split) them, or one for any die
	/// before then, or if the dice are too many to enumerate. A record never
	/// rolled has none.
	dice: Vec<DieChains>,

	/// Whether the chains have [split](Self::split) by die since the roll, or
	/// tried to.
	split: bool
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

/// Sets order by their intervals, lexicographically, so that the bounds of a
/// [shape](ChainShape) of chain may sort its runs by their results.
impl Ord for ResultSet
{
	fn cmp(&self, other: &Self) -> Ordering
	{
		let key = |set: &Self| {
			set.0
				.iter()
				.map(|interval| (interval.min, interval.max))
				.collect::<Vec<_>>()
		};
		key(self).cmp(&key(other))
	}
}

impl PartialOrd for ResultSet
{
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>
	{
		Some(self.cmp(other))
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

	/// Answer the criteria that hold only the specified criterion.
	///
	/// # Parameters
	/// - `criterion`: The results that the criterion matches.
	///
	/// # Returns
	/// The requested criteria.
	fn exact(criterion: ResultSet) -> Self
	{
		Self {
			certain: criterion.clone(),
			possible: criterion,
			shape: Shape::Greatest
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

/// The greatest number of pairs of a die and a criterion, each counted once for
/// every [shape](ChainShape) of chain that its die may leave, that the bounds
/// of a [reroll](crate::Reroll) or an [explode](crate::Explode) will enumerate
/// in pursuit of an exact answer. Beyond this they approximate. Every die of a
/// chain is the same die, and every one is matched by the same criterion,
/// which the union of the faces of every die, and of the results that every
/// criterion matches, forgets; so the bounds transform the chains of each pair
/// alone, and unite the answers. The budget is generous enough to cover two
/// operands that range over a few dozen values each, while bounding the work
/// at a trivial cost paid once per clause during bounds analysis.
const ENUMERATION_BUDGET: u128 = 1024;

/// The number of middle dice of a [chain](crate::RollingRecord::chains) that
/// stands for a chain without bound: many enough that the total saturates,
/// unless every middle die shows `0`, since a nonzero die moves the total by at
/// least `1`, and an `i32` spans fewer than `2^33` values.
const UNBOUNDED_DICE: i128 = 1 << 33;

/// The greatest number of [shapes](ChainShape) of chain that the bounds of a
/// rolling record keep for one die. Beyond this, they
/// [coarsen](ChainShape::coarsen) the shapes of the die into one, which covers
/// every chain of each, but whose totals may be loose. Each explode may
/// multiply the shapes of a die by a few, and each criterion to which its
/// predicate may resolve may add its own, so without a budget the shapes of a
/// run of explodes over wide operands might grow exponentially. The budget is
/// generous enough for several explodes and rerolls, each of a few criteria.
const SHAPE_BUDGET: usize = 64;

/// A die that a rolling record may roll, as the bounds of a
/// [reroll](crate::Reroll) or an [explode](crate::Explode) enumerate them.
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

/// A run of the middle dice of a [chain](crate::RollingRecord::chains), those
/// that some die extends, each of which may show any of the same results,
/// independently of the others.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Segment
{
	/// The results that each die of the run may show, which are never empty.
	results: ResultSet,

	/// The least number of dice of the run.
	least: u64,

	/// The greatest number of dice of the run, saturating at `u64::MAX`, which
	/// stands for any number.
	greatest: u64
}

/// A shape of a [chain](crate::RollingRecord::chains) still kept that no
/// [saturated](Origin::Saturates) die decides: its middle dice, those that
/// some die extends, in runs, and the results that its end, the kept die that
/// no die extends, may show. Each die of the chain may show any result of its
/// run, or of the end, independently of the others, and each run may hold any
/// number of dice within its bounds, so the shape describes every chain that
/// these choices make.
///
/// A chain still kept that no saturated die decides always has an end, since
/// only an unlimited reroll that matches every face crosses out a die without
/// replacing it, and that crosses out every die of the chain. A later reroll
/// acts on each die of a chain alike, so it maps the results of every run, and
/// of the end; a later explode extends only the end, so it appends runs.
///
/// `ChainShape`, in `lean/spec/XdySpec/Bounds.lean`, mirrors this type, and
/// `ChainShape.Has` there describes the chains that a shape describes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChainShape
{
	/// The runs of middle dice, ascending by their results, no two of which
	/// show the same results, and none of which is empty.
	middles: Vec<Segment>,

	/// The results that the end may show, which are never empty.
	end: ResultSet
}

impl ChainShape
{
	/// Answer the shape of the specified runs of middle dice and end, sorting
	/// the runs by their results, merging the runs of the same results, whose
	/// numbers of dice add, and discarding the runs that hold no dice.
	///
	/// # Parameters
	/// - `middles`: The runs of middle dice, in any order.
	/// - `end`: The results that the end may show.
	///
	/// # Returns
	/// The requested shape.
	fn new(middles: impl IntoIterator<Item = Segment>, end: ResultSet) -> Self
	{
		let mut runs = middles
			.into_iter()
			.filter(|run| run.greatest > 0)
			.collect::<Vec<_>>();
		runs.sort_by(|a, b| a.results.cmp(&b.results));
		let mut middles = Vec::<Segment>::with_capacity(runs.len());
		for run in runs
		{
			match middles.last_mut()
			{
				// The dice of both runs may show the same results, so they form
				// one run.
				Some(last) if last.results == run.results =>
				{
					last.least = last.least.saturating_add(run.least);
					last.greatest = last.greatest.saturating_add(run.greatest);
				},
				_ => middles.push(run)
			}
		}
		Self { middles, end }
	}

	/// Answer the least and greatest totals of the chains of the shape.
	///
	/// # Returns
	/// The totals, which are exact.
	///
	/// # Notes
	/// The total of a chain folds its dice in ascending order, saturating,
	/// which never decreases as any die grows, so the least total shows the
	/// least result of its run at every die, and the greatest, the greatest.
	/// Nor does it ever move against the sign of a die as their number grows,
	/// so the least total holds the most dice of a run whose least result is
	/// negative, and the fewest of any other, and the greatest total the most
	/// dice of a run whose greatest result is positive, and the fewest of any
	/// other. Each of these chains has the shape. A run of any number of dice
	/// holds [`UNBOUNDED_DICE`] of them, which saturate the total unless they
	/// show `0`.
	///
	/// `fold_mono_results`, `fold_mono_copies`, and `fold_unbounded`, in
	/// `lean/spec/XdySpec/Bounds.lean`, prove the properties of the fold on
	/// which these extremes rest, and `fold_eq_of_cap` that capping the dice
	/// of a run at [`UNBOUNDED_DICE`] changes no fold. `ChainShape.totals_le`
	/// there proves that the totals bound every chain of the shape, and
	/// `ChainShape.totals_realized` that each is the total of a corner of the
	/// shape, which, unless the shapes were coarsened, is a chain that the
	/// rerolls and explodes may make.
	fn totals(&self) -> EvaluationBounds
	{
		let copies = |count: u64| (count as i128).min(UNBOUNDED_DICE);
		let hull = |results: &ResultSet| {
			results
				.hull()
				.expect("every die of a shape shows some result")
		};
		let end = hull(&self.end);
		let least = fold([(end.min, 1)].into_iter().chain(
			self.middles.iter().map(|run| {
				let least = hull(&run.results).min;
				let count = match least < 0
				{
					true => run.greatest,
					false => run.least
				};
				(least, copies(count))
			})
		));
		let greatest = fold([(end.max, 1)].into_iter().chain(
			self.middles.iter().map(|run| {
				let greatest = hull(&run.results).max;
				let count = match greatest > 0
				{
					true => run.greatest,
					false => run.least
				};
				(greatest, copies(count))
			})
		));
		(least, greatest).into()
	}

	/// Answer the greatest number of kept dice of a chain of the shape.
	///
	/// # Returns
	/// The number of dice, saturating at `u64::MAX`.
	fn dice(&self) -> u64
	{
		self.middles
			.iter()
			.fold(1u64, |dice, run| dice.saturating_add(run.greatest))
	}

	/// Answer the greatest number of kept dice of a chain of the shape that may
	/// show a result of the specified set.
	///
	/// # Parameters
	/// - `results`: The results.
	///
	/// # Returns
	/// The number of dice, saturating at `u64::MAX`.
	fn matching(&self, results: &ResultSet) -> u64
	{
		self.middles
			.iter()
			.filter(|run| run.results.meets(results))
			.fold(self.end.meets(results) as u64, |dice, run| {
				dice.saturating_add(run.greatest)
			})
	}

	/// Answer the shape whose dice show the results to which the specified
	/// function maps the results of the receiver's, run by run, as a reroll
	/// replaces each die alike.
	///
	/// # Parameters
	/// - `map`: The function from the results that a die may show to those that
	///   it may show afterward.
	///
	/// # Returns
	/// The requested shape.
	fn map(&self, map: impl Fn(&ResultSet) -> ResultSet) -> Self
	{
		Self::new(
			self.middles.iter().map(|run| Segment {
				results: map(&run.results),
				..run.clone()
			}),
			map(&self.end)
		)
	}

	/// Answer whether every chain of the receiver has the specified shape.
	///
	/// # Parameters
	/// - `other`: The other shape.
	///
	/// # Returns
	/// `true` if the receiver's chains are among the other's, `false` if they
	/// may not be.
	///
	/// # Notes
	/// The answer compares the runs in order, so it may miss some shapes that
	/// another covers, but never claims one that it does not.
	fn within(&self, other: &Self) -> bool
	{
		self.middles.len() == other.middles.len()
			&& self.end.is_subset(&other.end)
			&& self.middles.iter().zip(&other.middles).all(|(run, other)| {
				run.results.is_subset(&other.results)
					&& other.least <= run.least
					&& run.greatest <= other.greatest
			})
	}

	/// Answer one shape that stands for both the receiver and the specified
	/// shape, through every later reroll and explode, if there is one that
	/// the bounds find.
	///
	/// # Parameters
	/// - `other`: The other shape.
	///
	/// # Returns
	/// The united shape, or `None` if the shapes differ too much.
	///
	/// # Notes
	/// If one shape covers the other, then it stands for both. Otherwise, two
	/// shapes that differ only in their ends, or only in the results or only
	/// in the numbers of dice of one run, unite into the shape that unites
	/// those. Uniting the ends is exact, since a chain has one end. Uniting
	/// the results of a run of several dice admits chains that mix them, and
	/// uniting the numbers of dice of a run may admit numbers between them,
	/// but neither moves the totals of the united shape beyond those of
	/// either, nor will any later reroll or explode: a reroll maps each result
	/// of a run alike, and an explode leaves the runs alone, so each run of
	/// the united shape always shows the results of both, and its extreme
	/// totals show the extreme result at every die of a run, as a chain of
	/// one of the shapes does, and the extreme number of dice of each run,
	/// as a chain of one of them does. Nor does uniting them change the most
	/// dice of a chain that may match a criterion.
	///
	/// `ChainShape.unite_has`, in `lean/spec/XdySpec/Bounds.lean`, proves that
	/// the united shape describes every chain of either, and
	/// `ChainShape.unite_realized` that each of its corners, which its extreme
	/// totals show, is a corner of one of them.
	fn unite(&self, other: &Self) -> Option<Self>
	{
		if self.within(other)
		{
			return Some(other.clone())
		}
		if other.within(self)
		{
			return Some(self.clone())
		}
		if self.middles.len() != other.middles.len()
		{
			return None
		}
		let differences = self
			.middles
			.iter()
			.zip(&other.middles)
			.enumerate()
			.filter(|(_, (run, other))| run != other)
			.map(|(index, _)| index)
			.collect::<Vec<_>>();
		match (differences.as_slice(), self.end == other.end)
		{
			([], _) => Some(Self {
				middles: self.middles.clone(),
				end: self.end.union(&other.end)
			}),
			([index], true) =>
			{
				let (run, theirs) =
					(&self.middles[*index], &other.middles[*index]);
				let united = match (
					run.results == theirs.results,
					(run.least, run.greatest)
						== (theirs.least, theirs.greatest)
				)
				{
					(true, _) => Segment {
						results: run.results.clone(),
						least: run.least.min(theirs.least),
						greatest: run.greatest.max(theirs.greatest)
					},
					(false, true) => Segment {
						results: run.results.union(&theirs.results),
						..run.clone()
					},
					(false, false) => return None
				};
				let mut middles = self.middles.clone();
				middles[*index] = united;
				Some(Self::new(middles, self.end.clone()))
			},
			_ => None
		}
	}

	/// Answer one shape that covers every chain of each of the specified
	/// shapes: one run, which may show any result of any of their runs, and
	/// holds as few or as many dice as the fewest or most of theirs, and an end
	/// that may show any result of any of their ends.
	///
	/// # Parameters
	/// - `shapes`: The shapes, which are not empty.
	///
	/// # Returns
	/// The coarsened shape, which may admit totals that none of the shapes
	/// admits.
	///
	/// # Notes
	/// `ChainShape.coarsen_has`, in `lean/spec/XdySpec/Bounds.lean`, proves
	/// that the coarsened shape describes every chain of each.
	fn coarsen(shapes: &[Self]) -> Self
	{
		let mut results = ResultSet::default();
		let mut end = ResultSet::default();
		let (mut least, mut greatest) = (u64::MAX, 0u64);
		for shape in shapes
		{
			for run in &shape.middles
			{
				results = results.union(&run.results);
			}
			least = least.min(
				shape
					.middles
					.iter()
					.fold(0u64, |dice, run| dice.saturating_add(run.least))
			);
			greatest = greatest.max(
				shape
					.middles
					.iter()
					.fold(0u64, |dice, run| dice.saturating_add(run.greatest))
			);
			end = end.union(&shape.end);
		}
		let run = Segment {
			results,
			least,
			greatest
		};
		match run.results.is_empty()
		{
			true => Self::new([], end),
			false => Self::new([run], end)
		}
	}
}

/// Unite the specified shapes into as few as the bounds find that stand for
/// all of them, through every later reroll and explode, as
/// [`ChainShape::unite`] does for two. If more than [`SHAPE_BUDGET`] remain,
/// [coarsen](ChainShape::coarsen) them into one.
///
/// # Parameters
/// - `shapes`: The shapes.
///
/// # Returns
/// The united shapes.
///
/// # Notes
/// `uniteShapes`, in `lean/spec/XdySpec/Bounds.lean`, mirrors this function,
/// and also reports whether it coarsened the shapes. `uniteShapes_has` there
/// proves that some united shape describes every chain of each shape, and
/// `uniteShapes_preserves` that, unless it coarsened them, the united shapes
/// keep every property of the shapes that uniting two keeps, such as that
/// every corner is a chain that the clauses may make.
fn unite_shapes(shapes: Vec<ChainShape>) -> Vec<ChainShape>
{
	let mut united = Vec::<ChainShape>::new();
	for mut shape in shapes
	{
		// A shape that unites with another may then unite with a third, so
		// begin again after each.
		let mut index = 0;
		while index < united.len()
		{
			match united[index].unite(&shape)
			{
				Some(both) =>
				{
					shape = both;
					united.swap_remove(index);
					index = 0;
				},
				None => index += 1
			}
		}
		united.push(shape);
		if united.len() > SHAPE_BUDGET
		{
			united = vec![ChainShape::coarsen(&united)];
		}
	}
	united
}

/// The chains still kept by a rolling record whose dice are all the same die,
/// or, if the bounds cannot tell its dice apart, any die that the record may
/// roll.
///
/// `DieChains`, in `lean/spec/XdySpec/Bounds.lean`, mirrors this type, with a
/// function that stands for [`least`](Self::least) and the shapes of the
/// criteria.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DieChains
{
	/// The faces of the die, any of which an extension or a replacement may
	/// show.
	faces: ResultSet,

	/// The least dice of the die, which decide whether some criterion may
	/// match every face: the die itself, or, for any die of the record, the
	/// [least dice](RollingRecordKind::least_dice) of the record.
	least: Vec<LeastDice>,

	/// The values that a [saturated](Origin::Saturates) die of the die may
	/// take.
	saturating: ResultSet,

	/// The shapes of the chains still kept that no saturated die decides.
	live: Vec<ChainShape>,

	/// The values that a chain still kept that a saturated die decides may
	/// take, which are none until an unlimited explode may saturate a chain.
	saturated: ResultSet
}

impl DieChains
{
	/// Answer the chains of a die just rolled, each of which holds one die,
	/// which may show any face.
	///
	/// # Parameters
	/// - `faces`: The faces of the die.
	/// - `least`: The least dice of the die.
	/// - `saturating`: The values that a saturated die of the die may take.
	///
	/// # Returns
	/// The requested chains, of which there are none if the die has no faces,
	/// as an uninitialized record does not.
	fn rolled(
		faces: ResultSet,
		least: Vec<LeastDice>,
		saturating: ResultSet
	) -> Self
	{
		let live = match faces.is_empty()
		{
			true => Vec::new(),
			false => vec![ChainShape::new([], faces.clone())]
		};
		Self {
			faces,
			least,
			saturating,
			live,
			saturated: ResultSet::default()
		}
	}

	/// Answer the totals that a chain still kept may show.
	///
	/// # Returns
	/// The totals, whose hull is exact, unless the shapes were coarsened. They
	/// may leave gaps that some chain fills, so only their hull is meaningful.
	fn totals(&self) -> ResultSet
	{
		self.live
			.iter()
			.map(|shape| ResultSet::from(shape.totals()))
			.fold(self.saturated.clone(), |totals, shape| totals.union(&shape))
	}

	/// Answer whether every chain of the die may be crossed out, i.e., if
	/// none is still kept, whatever the drops and keeps.
	///
	/// # Returns
	/// `true` if no chain is still kept, `false` otherwise.
	fn is_empty(&self) -> bool
	{
		self.live.is_empty() && self.saturated.is_empty()
	}

	/// Answer whether the specified criteria certainly match every face of the
	/// die.
	///
	/// # Parameters
	/// - `criteria`: The criteria.
	///
	/// # Returns
	/// `true` if every criterion matches every face, `false` otherwise.
	fn covered(&self, criteria: &Criteria) -> bool
	{
		self.faces.is_subset(&criteria.certain)
	}

	/// Answer whether some criterion may match every face of the die.
	///
	/// # Parameters
	/// - `criteria`: The criteria.
	///
	/// # Returns
	/// `true` if some criterion may match every face, `false` otherwise.
	fn may_be_covered(&self, criteria: &Criteria) -> bool
	{
		self.least.iter().any(|die| criteria.may_cover(die))
	}

	/// Answer whether some criterion may match some kept die of a chain still
	/// kept that no saturated die decides.
	///
	/// # Parameters
	/// - `criteria`: The criteria.
	///
	/// # Returns
	/// `true` if some die may match, `false` otherwise.
	fn may_match(&self, criteria: &Criteria) -> bool
	{
		self.live
			.iter()
			.any(|shape| shape.matching(&criteria.possible) > 0)
	}

	/// Answer whether some criterion may match the end of a chain still kept
	/// that no saturated die decides.
	///
	/// # Parameters
	/// - `criteria`: The criteria.
	///
	/// # Returns
	/// `true` if some end may match, `false` otherwise.
	fn may_extend(&self, criteria: &Criteria) -> bool
	{
		self.live
			.iter()
			.any(|shape| shape.end.meets(&criteria.possible))
	}

	/// Answer whether an [explode](crate::Explode) may have extended or
	/// saturated some chain of the die.
	///
	/// # Returns
	/// `true` if some chain may hold a middle die or may saturate, `false`
	/// otherwise.
	fn exploded(&self) -> bool
	{
		!self.saturated.is_empty()
			|| self.live.iter().any(|shape| !shape.middles.is_empty())
	}

	/// Answer the chains after a [reroll](crate::Reroll) by any of the
	/// specified criteria, uniting the chains of each.
	///
	/// # Parameters
	/// - `criteria`: The criteria, each of which the predicate may resolve to,
	///   or one that stands for every one.
	/// - `limit`: The bounds of the limit, or `None` for no limit.
	///
	/// # Returns
	/// The chains; whether the reroll may cross out every chain that no
	/// saturated die decides, of which there are some; and the greatest number
	/// of dice that it may roll for one chain still kept, saturating at
	/// `u64::MAX`.
	///
	/// # Notes
	/// The reroll leaves alone every chain that a saturated die decides, and
	/// rerolls every other die that a criterion matches: up to the limit, so
	/// that it may show any face, or without limit, until it shows a face that
	/// the criterion does not match. Without limit, a criterion that matches
	/// every face crosses out every die instead, rolling nothing, and so every
	/// chain that no saturated die decides. A limit that may be zero or less
	/// may leave the chains as they were.
	///
	/// `DieChains.reroll_covers`, in `lean/spec/XdySpec/Bounds.lean`, proves
	/// that the chains answered cover every chain that the reroll may make of
	/// a chain that the receiver covered.
	fn reroll(
		&self,
		criteria: &[Criteria],
		limit: Option<EvaluationBounds>
	) -> (Self, bool, u64)
	{
		let mut live = Vec::new();
		let mut crosses = false;
		let mut rolls = 0u64;
		let (stays, rerolls) = match limit
		{
			Some(limit) => (limit.min <= 0, limit.max > 0),
			None => (false, true)
		};
		for criteria in criteria
		{
			if limit.is_none() && self.may_be_covered(criteria)
			{
				crosses |= !self.live.is_empty();
			}
			if limit.is_none() && self.covered(criteria)
			{
				continue
			}
			let replacements = match limit
			{
				Some(_) => self.faces.clone(),
				None => self.faces.difference(&criteria.certain)
			};
			let reroll = |results: &ResultSet| {
				let kept = results.difference(&criteria.certain);
				match results.meets(&criteria.possible)
				{
					true => kept.union(&replacements),
					false => kept
				}
			};
			for shape in &self.live
			{
				if stays
				{
					live.push(shape.clone());
				}
				if rerolls
				{
					live.push(shape.map(reroll));
					// Each matching die may roll up to the limit more times,
					// or without limit, any number.
					let times = limit.map_or(u64::MAX, |limit| dice(limit.max));
					rolls = rolls.max(
						shape
							.matching(&criteria.possible)
							.saturating_mul(times)
					);
				}
			}
		}
		let chains = Self {
			faces: self.faces.clone(),
			least: self.least.clone(),
			saturating: self.saturating.clone(),
			live: unite_shapes(live),
			saturated: self.saturated.clone()
		};
		(chains, crosses, rolls)
	}

	/// Answer the chains after an [explode](crate::Explode) by any of the
	/// specified criteria, uniting the chains of each.
	///
	/// # Parameters
	/// - `criteria`: The criteria, each of which the predicate may resolve to,
	///   or one that stands for every one.
	/// - `limit`: The bounds of the limit, or `None` for no limit.
	///
	/// # Returns
	/// The chains; and the greatest number of dice that the explode may roll
	/// for one chain still kept, saturating at `u64::MAX`, which stands for
	/// any number, as an unlimited explode that does not saturate may roll.
	///
	/// # Notes
	/// The explode leaves alone every chain that a saturated die decides, and
	/// extends only the end of every other, so a chain whose end the criterion
	/// does not match stays as it was. A chain whose end `e` it matches holds
	/// `e` among its middle dice, then `m` more middle dice that it matches,
	/// and then a new end: one that it does not match, ending the chain, or,
	/// if the limit is spent, any face. Under a limit `L`, the chain ends after
	/// `1` to `L - 1` extensions, so `m` ranges from `0` to `L - 2`, or the
	/// limit is spent after `L` extensions, so `m` is `L - 1`; over a range of
	/// limits, these unite. A limit that may be zero or less may leave the
	/// chains as they were. Without limit, `m` is any number, unless the
	/// criterion matches every face, whence every chain saturates instead,
	/// rolling nothing.
	///
	/// `DieChains.explode_covers`, in `lean/spec/XdySpec/Bounds.lean`, proves
	/// that the chains answered cover every chain that the explode may make of
	/// a chain that the receiver covered.
	fn explode(
		&self,
		criteria: &[Criteria],
		limit: Option<EvaluationBounds>
	) -> (Self, u64)
	{
		let mut live = Vec::new();
		let mut saturated = self.saturated.clone();
		let mut extensions = 0u64;
		let (stays, extends) = match limit
		{
			Some(limit) => (limit.min <= 0, limit.max > 0),
			None => (false, true)
		};
		for criteria in criteria
		{
			let may_extend = self.may_extend(criteria);
			if limit.is_none() && may_extend && self.may_be_covered(criteria)
			{
				saturated = saturated.union(&self.saturating);
			}
			if limit.is_none() && self.covered(criteria)
			{
				continue
			}
			if extends && may_extend
			{
				extensions = extensions
					.max(limit.map_or(u64::MAX, |limit| dice(limit.max)));
			}
			let middles = self.faces.intersection(&criteria.possible);
			let ends = self.faces.difference(&criteria.certain);
			for shape in &self.live
			{
				if stays
				{
					live.push(shape.clone());
				}
				if !extends
				{
					continue
				}
				let rest = shape.end.difference(&criteria.certain);
				if !rest.is_empty()
				{
					live.push(ChainShape::new(shape.middles.clone(), rest));
				}
				let start = shape.end.intersection(&criteria.possible);
				if start.is_empty()
				{
					continue
				}
				let grow = |least: u64, greatest: u64, end: ResultSet| {
					ChainShape::new(
						shape.middles.iter().cloned().chain([
							Segment {
								results: start.clone(),
								least: 1,
								greatest: 1
							},
							Segment {
								results: middles.clone(),
								least,
								greatest
							}
						]),
						end
					)
				};
				match limit
				{
					// Some face does not match, or the criteria would cover the
					// die.
					None => live.push(grow(0, u64::MAX, ends.clone())),
					Some(limit) =>
					{
						let (least, greatest) =
							(dice(limit.min).max(1), dice(limit.max));
						if greatest >= 2 && !ends.is_empty()
						{
							live.push(grow(0, greatest - 2, ends.clone()));
						}
						live.push(grow(
							least - 1,
							greatest - 1,
							self.faces.clone()
						));
					}
				}
			}
		}
		let chains = Self {
			faces: self.faces.clone(),
			least: self.least.clone(),
			saturating: self.saturating.clone(),
			live: unite_shapes(live),
			saturated
		};
		(chains, extensions)
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
///
/// `fold`, in `lean/spec/XdySpec/Bounds.lean`, mirrors this function:
/// `fold_eq_total` proves that it is the total of every chain of the dice,
/// `fold_mono_results` and `fold_mono_copies` that it never decreases as any
/// die grows nor moves against the sign of a die as its copies grow, and
/// `fold_unbounded` that `UNBOUNDED_DICE` copies of a nonzero die saturate
/// it.
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
	/// Reset the chains of the record after a roll fills it: each still kept
	/// holds one die, which may show any face of any die that the record may
	/// roll.
	fn rolled(&mut self)
	{
		self.dice = vec![DieChains::rolled(
			self.kind.faces(),
			self.kind.least_dice(),
			self.kind.saturated_values()
		)];
		self.split = false;
	}

	/// Split the chains of the record by the die that it rolls, if the bounds
	/// of its faces or endpoints admit few enough dice, and it has not yet
	/// split. Every die of a chain is the same die, so a reroll or explode
	/// treats the chains of each die alone. Until the first reroll or explode,
	/// each chain holds one die, which may show any face, so the split loses
	/// nothing.
	fn split(&mut self)
	{
		if self.split
		{
			return
		}
		self.split = true;
		if let Some(dice) = self.kind.dice_within(ENUMERATION_BUDGET)
		{
			self.dice = dice
				.into_iter()
				.map(|die| {
					DieChains::rolled(
						die.faces.clone(),
						vec![LeastDice::Fixed(die.faces)],
						ResultSet::of([die.saturated])
					)
				})
				.collect();
		}
	}

	/// Answer the criteria to which the specified predicate may resolve, over
	/// the bounds of its operands, for a reroll or explode of the record.
	///
	/// # Parameters
	/// - `predicate`: The predicate.
	/// - `value`: The function that answers the bounds of an operand.
	///
	/// # Returns
	/// Each criterion alone, if they are few enough to transform every shape
	/// of every die by each within [`ENUMERATION_BUDGET`], or otherwise the
	/// criteria together, which stand for every one.
	fn criteria(
		&self,
		predicate: &Predicate,
		value: impl Fn(AddressingMode) -> EvaluationBounds
	) -> Vec<Criteria>
	{
		let shapes = self
			.dice
			.iter()
			.map(|die| die.live.len().max(1) as u128)
			.sum::<u128>()
			.max(1);
		match criteria_within(predicate, &value, ENUMERATION_BUDGET / shapes)
		{
			Some(criteria) =>
			{
				criteria.into_iter().map(Criteria::exact).collect()
			},
			None => vec![Criteria::new(predicate, &value)]
		}
	}

	/// Answer whether an [explode](crate::Explode) may have extended or
	/// saturated some chain of the record since its roll. If not, each chain
	/// still kept holds one kept die, which ends it.
	///
	/// # Returns
	/// `true` if some chain may hold more than one kept die or may saturate,
	/// `false` otherwise.
	fn exploded(&self) -> bool { self.dice.iter().any(DieChains::exploded) }

	/// Answer the totals that a chain still kept may show.
	///
	/// # Returns
	/// The totals, whose hull is exact, unless the operands of the record and
	/// of its rerolls and explodes are too many to enumerate, or the shapes of
	/// its chains were coarsened. They may leave gaps that some chain fills,
	/// so only their hull is meaningful.
	///
	/// # Notes
	/// `DieChains.run_sound`, in `lean/spec/XdySpec/Bounds.lean`, proves that
	/// the totals of the chains of a die hold the total of every chain still
	/// kept after any sequence of rerolls, explodes, drops, and keeps, and
	/// `DieChains.run_exact` that, for exact criteria and shapes never
	/// coarsened, their hull is exactly that of the totals of the chains that
	/// the clauses may make of one face.
	fn totals(&self) -> ResultSet
	{
		self.dice.iter().fold(ResultSet::default(), |totals, die| {
			totals.union(&die.totals())
		})
	}

	/// Answer the greatest number of kept dice of a chain still kept that no
	/// saturated die decides.
	///
	/// # Returns
	/// The number of kept dice, saturating at `u64::MAX`, which stands for any
	/// number. A roll sets it to `1`, and only an explode raises it. It is at
	/// least `1` for a record rolled, even once no such chain remains, and `0`
	/// for a record never rolled, which has no dice.
	fn length(&self) -> u64
	{
		match self.kind
		{
			RollingRecordKind::Uninitialized => 0,
			_ => self
				.dice
				.iter()
				.flat_map(|die| &die.live)
				.map(ChainShape::dice)
				.fold(1, u64::max)
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
	/// chain still kept, if the end of some chain may match, and a limited
	/// [reroll](crate::Reroll), for each kept die of a chain that may match,
	/// for every chain still kept; the bounds track the shapes of the chains,
	/// so the bound of `4D6 explode = 6 1 times reroll = 1` is `12`, since the
	/// `6` from which a chain grows never shows `1`. An unlimited explode that
	/// may roll a die, unless it saturates, saturates the bound.
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
			dice: 0,
			unbounded: false
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

	/// Answer whether some explode stepped so far has no limit, does not
	/// certainly match every face, and may extend a chain, which may then grow
	/// without bound, so that the function has no exact distribution of finite
	/// weight.
	///
	/// # Returns
	/// `true` if some such explode may extend a chain, `false` otherwise.
	#[inline]
	pub(crate) fn unbounded(&self) -> bool { self.unbounded }

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

	/// Answer the least interval that holds every total that a
	/// [chain](crate::RollingRecord::chains) still kept by the specified
	/// rolling record may show, which is exact.
	///
	/// # Parameters
	/// - `op`: The rolling record.
	///
	/// # Returns
	/// The interval, or `None` if no chain may be kept.
	#[inline]
	pub(crate) fn totals(
		&self,
		op: RollingRecordIndex
	) -> Option<EvaluationBounds>
	{
		self.record(op).totals().hull()
	}

	/// Answer the greatest number of kept dice of a
	/// [chain](crate::RollingRecord::chains) still kept by the specified
	/// rolling record that no saturated die decides.
	///
	/// # Parameters
	/// - `op`: The rolling record.
	///
	/// # Returns
	/// The number of kept dice, which saturates at `u64::MAX` if an explode
	/// without limit may grow a chain without bound, and is `0` for a record
	/// never rolled.
	#[inline]
	pub(crate) fn length(&self, op: RollingRecordIndex) -> u64
	{
		self.record(op).length()
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
		record.rolled();
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
		record.rolled();
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
		record.rolled();
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
		self.record_mut(inst.dest).split();
		let record = self.record(inst.dest);
		let kept = record.kept;
		let (branches, matching) = record.kind.branches(&criteria.certain);
		let resolutions = record.criteria(&inst.predicate, |op| self.value(op));
		// A die still kept may be rerolled only if some criterion matches some
		// result that it may show.
		let may_match = kept.max > 0
			&& record.dice.iter().any(|die| {
				resolutions.iter().any(|criteria| die.may_match(criteria))
			});
		let exploded = record.exploded();
		if exploded && !may_match
		{
			// A reroll that cannot match leaves the chains as they were.
			return Ok(())
		}
		// Reroll the chains of each die by each criterion. The reroll leaves
		// alone every chain that a saturated die decides.
		let mut crosses = false;
		let mut rolls = 0u64;
		let chains = record
			.dice
			.iter()
			.map(|die| {
				let (chains, crossed, rolled) = die.reroll(&resolutions, limit);
				crosses |= crossed;
				rolls = rolls.max(rolled);
				chains
			})
			.collect::<Vec<_>>();
		match exploded
		{
			// A chain may hold several dice, each of which the reroll rerolls
			// alone, so the number of dice rerolled varies; give up counting.
			true => self.count = None,
			false =>
			{
				// Count the branches of each die still kept as though every
				// reroll that it might need were rolled in advance, which makes
				// every path equally likely: a limited reroll rolls each die up
				// to the limit more times, whatever it shows, and an unlimited
				// one rolls each die once more, from the faces that do not
				// match, since the die finally shows one of these whether or
				// not it first matched. A reroll that cannot happen, because no
				// face matches, the limit is not positive, or, without limit,
				// every face matches, adds no branches.
				let branches = match limit
				{
					Some(limit) if matching > 0 =>
					{
						branches.saturating_pow(limit.max.max(0) as u32)
					},
					None if 0 < matching && matching < branches =>
					{
						branches - matching
					},
					_ => 1
				};
				if branches > 1
				{
					match kept.min == kept.max
					{
						true => self.count_branches(
							branches.saturating_pow(kept.max as u32)
						),
						// The number of dice rerolled depends on a previous
						// range or roll, so the paths through them are
						// unequally likely.
						false => self.count = None
					}
				}
			}
		}
		let record = self.record_mut(inst.dest);
		record.dice = chains;
		if crosses
		{
			// Without limit, a reroll that matches every face crosses out every
			// chain that no saturated die decides, so only the chains that
			// saturated dice decide may stay kept.
			record.kept = match record.dice.iter().all(DieChains::is_empty)
			{
				true => 0.into(),
				false => (0, kept.max).into()
			};
		}
		self.dice = self
			.dice
			.saturating_add(dice(kept.max).saturating_mul(rolls));
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
		self.record_mut(inst.dest).split();
		let record = self.record(inst.dest);
		let kept = record.kept;
		let (branches, matching) = record.kind.branches(&criteria.certain);
		let resolutions = record.criteria(&inst.predicate, |op| self.value(op));
		// A chain still kept may be extended only if some criterion matches
		// some result that its end may show, and the limit may be positive.
		let may_explode = kept.max > 0
			&& limit.is_none_or(|limit| limit.max > 0)
			&& record.dice.iter().any(|die| {
				resolutions.iter().any(|criteria| die.may_extend(criteria))
			});
		let exploded = record.exploded();
		if !may_explode
		{
			// Count the branches that the explode would roll, even if no die
			// still kept may match, unless a chain may hold several dice.
			if !exploded
			{
				self.count_explode(kept, limit, branches, matching);
			}
			return Ok(())
		}
		// Explode the chains of each die by each criterion. Each chain still
		// kept grows only from its end, so a limited explode extends each up
		// to the limit, and a saturating one rolls nothing.
		let mut extensions = 0u64;
		let chains = record
			.dice
			.iter()
			.map(|die| {
				let (chains, extended) = die.explode(&resolutions, limit);
				extensions = extensions.max(extended);
				chains
			})
			.collect::<Vec<_>>();
		match exploded
		{
			// A chain may hold several dice, of which the explode extends only
			// the last, so the number of dice rolled varies; give up counting.
			true => self.count = None,
			false => self.count_explode(kept, limit, branches, matching)
		}
		if extensions == u64::MAX
		{
			// Some chain may grow without bound.
			self.unbounded = true;
		}
		self.dice = self
			.dice
			.saturating_add(dice(kept.max).saturating_mul(extensions));
		self.record_mut(inst.dest).dice = chains;
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
		let sum = match record.totals().hull()
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
