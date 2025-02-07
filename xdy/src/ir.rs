//! # Intermediate representation
//!
//! The intermediate representation (IR) of the dice language. The instruction
//! set specifies a simple register transfer language (RTL) for the dice
//! language. There is no control flow, so the language is purely functional.
//! No control flow graph (CFG) is necessary, as all instructions reside within
//! a single basic block.

use std::{
	cell::{Ref, RefCell},
	collections::{BTreeSet, HashMap, HashSet},
	fmt::{Display, Formatter}
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::RollingRecord;

////////////////////////////////////////////////////////////////////////////////
//                              Instruction set.                              //
////////////////////////////////////////////////////////////////////////////////

/// Instruction: Roll an inclusive range.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollRange
{
	/// The destination rolling record for the results.
	pub dest: RollingRecordIndex,

	/// The start of the range.
	pub start: AddressingMode,

	/// The end of the range.
	pub end: AddressingMode
}

impl Display for RollRange
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- roll range {}:{}", self.dest, self.start, self.end)
	}
}

/// Instruction: Roll a set of standard dice.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollStandardDice
{
	/// The destination rolling record for the results.
	pub dest: RollingRecordIndex,

	/// The number of dice to roll.
	pub count: AddressingMode,

	/// The number of faces on each die.
	pub faces: AddressingMode
}

impl Display for RollStandardDice
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(
			f,
			"{} <- roll standard dice {}D{}",
			self.dest, self.count, self.faces
		)
	}
}

/// Instruction: Roll a set of custom dice.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollCustomDice
{
	/// The destination rolling record for the results.
	pub dest: RollingRecordIndex,

	/// The number of dice to roll.
	pub count: AddressingMode,

	/// The faces of each die in the homogeneous set.
	pub faces: Vec<i32>
}

impl RollCustomDice
{
	/// Answer the number of distinct faces of the custom die.
	///
	/// # Returns
	/// The number of distinct faces.
	pub fn distinct_faces(&self) -> usize
	{
		self.faces.iter().collect::<HashSet<_>>().len()
	}
}

impl Display for RollCustomDice
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- roll custom dice {}D[", self.dest, self.count)?;
		let mut first = true;
		for face in &self.faces
		{
			if !first
			{
				write!(f, ", ")?;
			}
			write!(f, "{}", face)?;
			first = false;
		}
		write!(f, "]")
	}
}

/// Instruction: Drop the lowest dice from a rolling record. This simply
/// involves marking the lowest dice as dropped, so that they are not included
/// in the sum computed by [SumRollingRecord]. A negative count will put back
/// lowest dice that were previously dropped. The number of dice to drop is
/// clamped to the number of dice in the record.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DropLowest
{
	/// The [rolling record](RollingRecord) from which to drop dice.
	pub dest: RollingRecordIndex,

	/// The number of dice to drop.
	pub count: AddressingMode
}

impl Display for DropLowest
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(
			f,
			"{} <- drop lowest {} from {}",
			self.dest, self.count, self.dest
		)
	}
}

/// Instruction: Drop the highest dice from a rolling record. This simply
/// involves marking the highest dice as dropped, so that they are not included
/// in the sum computed by [SumRollingRecord]. A negative count will put back
/// highest dice that were previously dropped. The number of dice to drop is
/// clamped to the number of dice in the record.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DropHighest
{
	/// The [rolling record](RollingRecord) from which to drop dice.
	pub dest: RollingRecordIndex,

	/// The number of dice to drop.
	pub count: AddressingMode
}

impl Display for DropHighest
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(
			f,
			"{} <- drop highest {} from {}",
			self.dest, self.count, self.dest
		)
	}
}

/// Instruction: Compute the result of a [rolling record](RollingRecord),
/// ignoring any dice marked as dropped by [DropLowest] or [DropHighest].
/// The summation is clamped to the range of an `i32`.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SumRollingRecord
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The source rolling record to compute.
	pub src: RollingRecordIndex
}

impl Display for SumRollingRecord
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- sum rolling record {}", self.dest, self.src)
	}
}

/// Instruction: Add two values together, saturating on overflow.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Add
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Add
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} + {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Subtract one value from another, saturating on overflow.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sub
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Sub
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} - {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Multiply two values together, saturating on overflow.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mul
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Mul
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} * {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Divide one value by another, saturating on overflow. Treat
/// division by zero as zero.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Div
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Div
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} / {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Compute the remainder of dividing one value by another,
/// saturating on overflow. Treat division by zero as zero.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mod
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Mod
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} % {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Compute the exponentiation of one value by another,
/// saturating on overflow.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Exp
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The first operand.
	pub op1: AddressingMode,

	/// The second operand.
	pub op2: AddressingMode
}

impl Display for Exp
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- {} ^ {}", self.dest, self.op1, self.op2)
	}
}

/// Instruction: Compute the negation of a value, saturating on overflow.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neg
{
	/// The destination register for the result.
	pub dest: RegisterIndex,

	/// The operand.
	pub op: AddressingMode
}

impl Display for Neg
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{} <- -{}", self.dest, self.op)
	}
}

/// Instruction: Return a value from the function.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Return
{
	/// The return value.
	pub src: AddressingMode
}

impl Display for Return
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "return {}", self.src)
	}
}

/// An instruction in the intermediate representation of the dice language.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Instruction
{
	RollRange(RollRange),
	RollStandardDice(RollStandardDice),
	RollCustomDice(RollCustomDice),
	DropLowest(DropLowest),
	DropHighest(DropHighest),
	SumRollingRecord(SumRollingRecord),
	Add(Add),
	Sub(Sub),
	Mul(Mul),
	Div(Div),
	Mod(Mod),
	Exp(Exp),
	Neg(Neg),
	Return(Return)
}

impl Instruction
{
	/// Get the destination for the instruction, if any.
	///
	/// # Returns
	/// The destination, or `None` if the instruction has no destination.
	pub fn destination(&self) -> Option<AddressingMode>
	{
		match self
		{
			Self::RollRange(inst) => Some(inst.dest.into()),
			Self::RollStandardDice(inst) => Some(inst.dest.into()),
			Self::RollCustomDice(inst) => Some(inst.dest.into()),
			Self::DropLowest(inst) => Some(inst.dest.into()),
			Self::DropHighest(inst) => Some(inst.dest.into()),
			Self::SumRollingRecord(inst) => Some(inst.dest.into()),
			Self::Add(inst) => Some(inst.dest.into()),
			Self::Sub(inst) => Some(inst.dest.into()),
			Self::Mul(inst) => Some(inst.dest.into()),
			Self::Div(inst) => Some(inst.dest.into()),
			Self::Mod(inst) => Some(inst.dest.into()),
			Self::Exp(inst) => Some(inst.dest.into()),
			Self::Neg(inst) => Some(inst.dest.into()),
			Self::Return(_) => None
		}
	}

	/// Get the sources for the instruction, if any.
	///
	/// # Returns
	/// The sources, or an empty vector if the instruction has no sources.
	pub fn sources(&self) -> Vec<AddressingMode>
	{
		match self
		{
			Self::RollRange(inst) => vec![inst.start, inst.end],
			Self::RollStandardDice(inst) => vec![inst.count, inst.faces],
			Self::RollCustomDice(inst) => vec![inst.count],
			Self::DropLowest(inst) => vec![inst.dest.into(), inst.count],
			Self::DropHighest(inst) => vec![inst.dest.into(), inst.count],
			Self::SumRollingRecord(inst) => vec![inst.src.into()],
			Self::Add(inst) => vec![inst.op1, inst.op2],
			Self::Sub(inst) => vec![inst.op1, inst.op2],
			Self::Mul(inst) => vec![inst.op1, inst.op2],
			Self::Div(inst) => vec![inst.op1, inst.op2],
			Self::Mod(inst) => vec![inst.op1, inst.op2],
			Self::Exp(inst) => vec![inst.op1, inst.op2],
			Self::Neg(inst) => vec![inst.op],
			Self::Return(inst) => vec![inst.src]
		}
	}

	/// Create an instruction to roll a range of dice.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	/// - `start`: The start of the range.
	/// - `end`: The end of the range.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn roll_range(
		dest: RollingRecordIndex,
		start: AddressingMode,
		end: AddressingMode
	) -> Self
	{
		Self::RollRange(RollRange { dest, start, end })
	}

	/// Create an instruction to roll standard dice.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	/// - `count`: The number of dice to roll.
	/// - `faces`: The number of faces on each die.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn roll_standard_dice(
		dest: RollingRecordIndex,
		count: AddressingMode,
		faces: AddressingMode
	) -> Self
	{
		Self::RollStandardDice(RollStandardDice { dest, count, faces })
	}

	/// Create an instruction to roll custom dice.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	/// - `count`: The number of dice to roll.
	/// - `faces`: The number of faces on each die.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn roll_custom_dice(
		dest: RollingRecordIndex,
		count: AddressingMode,
		faces: Vec<i32>
	) -> Self
	{
		Self::RollCustomDice(RollCustomDice { dest, count, faces })
	}

	/// Create an instruction to drop the lowest value from a rolling record.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	/// - `count`: The number of dice to drop.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn drop_lowest(
		dest: RollingRecordIndex,
		count: AddressingMode
	) -> Self
	{
		Self::DropLowest(DropLowest { dest, count })
	}

	/// Create an instruction to drop the highest value from a rolling record.
	///
	/// # Parameters
	/// - `dest`: The destination rolling record.
	/// - `count`: The number of dice to drop.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn drop_highest(
		dest: RollingRecordIndex,
		count: AddressingMode
	) -> Self
	{
		Self::DropHighest(DropHighest { dest, count })
	}

	/// Create an instruction to sum the values in a rolling record.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `src`: The rolling record to sum.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn sum_rolling_record(
		dest: RegisterIndex,
		src: RollingRecordIndex
	) -> Self
	{
		Self::SumRollingRecord(SumRollingRecord { dest, src })
	}

	/// Create an instruction to add two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn add(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Add(Add { dest, op1, op2 })
	}

	/// Create an instruction to subtract two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn sub(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Sub(Sub { dest, op1, op2 })
	}

	/// Create an instruction to multiply two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn mul(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Mul(Mul { dest, op1, op2 })
	}

	/// Create an instruction to divide two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn div(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Div(Div { dest, op1, op2 })
	}

	/// Create an instruction to compute the remainder of two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn r#mod(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Mod(Mod { dest, op1, op2 })
	}

	/// Create an instruction to compute the exponentiation of two values.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op1`: The first operand.
	/// - `op2`: The second operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn exp(
		dest: RegisterIndex,
		op1: AddressingMode,
		op2: AddressingMode
	) -> Self
	{
		Self::Exp(Exp { dest, op1, op2 })
	}

	/// Create an instruction to negate a value.
	///
	/// # Parameters
	/// - `dest`: The destination register.
	/// - `op`: The source operand.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn neg(dest: RegisterIndex, op: AddressingMode) -> Self
	{
		Self::Neg(Neg { dest, op })
	}

	/// Create an instruction to return a value from the function.
	///
	/// # Parameters
	/// - `src`: The return value.
	///
	/// # Returns
	/// The instruction.
	#[inline]
	pub const fn r#return(src: AddressingMode) -> Self
	{
		Self::Return(Return { src })
	}
}

impl Display for Instruction
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			Self::RollRange(inst) => write!(f, "{}", inst),
			Self::RollStandardDice(inst) => write!(f, "{}", inst),
			Self::RollCustomDice(inst) => write!(f, "{}", inst),
			Self::DropLowest(inst) => write!(f, "{}", inst),
			Self::DropHighest(inst) => write!(f, "{}", inst),
			Self::SumRollingRecord(inst) => write!(f, "{}", inst),
			Self::Add(inst) => write!(f, "{}", inst),
			Self::Sub(inst) => write!(f, "{}", inst),
			Self::Mul(inst) => write!(f, "{}", inst),
			Self::Div(inst) => write!(f, "{}", inst),
			Self::Mod(inst) => write!(f, "{}", inst),
			Self::Exp(inst) => write!(f, "{}", inst),
			Self::Neg(inst) => write!(f, "{}", inst),
			Self::Return(inst) => write!(f, "{}", inst)
		}
	}
}

impl From<RollRange> for Instruction
{
	fn from(inst: RollRange) -> Self { Self::RollRange(inst) }
}

impl From<RollStandardDice> for Instruction
{
	fn from(inst: RollStandardDice) -> Self { Self::RollStandardDice(inst) }
}

impl From<RollCustomDice> for Instruction
{
	fn from(inst: RollCustomDice) -> Self { Self::RollCustomDice(inst) }
}

impl From<DropLowest> for Instruction
{
	fn from(inst: DropLowest) -> Self { Self::DropLowest(inst) }
}

impl From<DropHighest> for Instruction
{
	fn from(inst: DropHighest) -> Self { Self::DropHighest(inst) }
}

impl From<SumRollingRecord> for Instruction
{
	fn from(inst: SumRollingRecord) -> Self { Self::SumRollingRecord(inst) }
}

impl From<Add> for Instruction
{
	fn from(inst: Add) -> Self { Self::Add(inst) }
}

impl From<Sub> for Instruction
{
	fn from(inst: Sub) -> Self { Self::Sub(inst) }
}

impl From<Mul> for Instruction
{
	fn from(inst: Mul) -> Self { Self::Mul(inst) }
}

impl From<Div> for Instruction
{
	fn from(inst: Div) -> Self { Self::Div(inst) }
}

impl From<Mod> for Instruction
{
	fn from(inst: Mod) -> Self { Self::Mod(inst) }
}

impl From<Exp> for Instruction
{
	fn from(inst: Exp) -> Self { Self::Exp(inst) }
}

impl From<Neg> for Instruction
{
	fn from(inst: Neg) -> Self { Self::Neg(inst) }
}

impl From<Return> for Instruction
{
	fn from(inst: Return) -> Self { Self::Return(inst) }
}

impl TryFrom<Instruction> for RollRange
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::RollRange(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for RollStandardDice
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::RollStandardDice(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for RollCustomDice
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::RollCustomDice(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for DropLowest
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::DropLowest(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for DropHighest
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::DropHighest(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for SumRollingRecord
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::SumRollingRecord(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Add
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Add(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Sub
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Sub(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Mul
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Mul(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Div
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Div(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Mod
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Mod(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Exp
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Exp(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Neg
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Neg(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

impl TryFrom<Instruction> for Return
{
	type Error = ();

	fn try_from(inst: Instruction) -> Result<Self, Self::Error>
	{
		match inst
		{
			Instruction::Return(inst) => Ok(inst),
			_ => Err(())
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                             Addressing modes.                              //
////////////////////////////////////////////////////////////////////////////////

/// The addressing mode for instructions. An addressing mode specifies how to
/// access the operand of an instruction. The operand can be an immediate
/// constant or a register. Registers are identified by their index into the
/// frame's register set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AddressingMode
{
	/// The addressing mode is an immediate constant.
	Immediate(Immediate),

	/// The addressing mode is a register.
	Register(RegisterIndex),

	/// The addressing mode is a rolling record.
	RollingRecord(RollingRecordIndex)
}

impl Display for AddressingMode
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			Self::Immediate(imm) => write!(f, "{}", imm),
			Self::Register(reg) => write!(f, "{}", reg),
			Self::RollingRecord(rec) => write!(f, "{}", rec)
		}
	}
}

/// An immediate constant value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Immediate(pub i32);

impl Display for Immediate
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{}", self.0)
	}
}

impl std::ops::Add for Immediate
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output { Self(self.0 + rhs.0) }
}

impl std::ops::Sub for Immediate
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output { Self(self.0 - rhs.0) }
}

impl std::ops::Mul for Immediate
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output { Self(self.0 * rhs.0) }
}

impl std::ops::Div for Immediate
{
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output { Self(self.0 / rhs.0) }
}

impl std::ops::Rem for Immediate
{
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output { Self(self.0 % rhs.0) }
}

impl std::ops::Neg for Immediate
{
	type Output = Self;

	fn neg(self) -> Self::Output { Self(-self.0) }
}

impl From<usize> for Immediate
{
	fn from(immediate: usize) -> Self { Self(immediate as i32) }
}

impl From<Immediate> for i32
{
	fn from(immediate: Immediate) -> Self { immediate.0 }
}

impl From<Immediate> for AddressingMode
{
	fn from(immediate: Immediate) -> Self { Self::Immediate(immediate) }
}

/// The index of a register in the frame's register set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegisterIndex(pub usize);

impl Display for RegisterIndex
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "@{}", self.0)
	}
}

impl From<usize> for RegisterIndex
{
	fn from(reg: usize) -> Self { Self(reg) }
}

impl From<RegisterIndex> for usize
{
	fn from(reg: RegisterIndex) -> Self { reg.0 }
}

impl From<RegisterIndex> for AddressingMode
{
	fn from(reg: RegisterIndex) -> Self { Self::Register(reg) }
}

/// The index of a rolling record in the frame's rolling record set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingRecordIndex(pub usize);

impl Display for RollingRecordIndex
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "⚅{}", self.0)
	}
}

impl From<usize> for RollingRecordIndex
{
	fn from(rec: usize) -> Self { Self(rec) }
}

impl From<RollingRecordIndex> for usize
{
	fn from(rec: RollingRecordIndex) -> Self { rec.0 }
}

impl From<RollingRecordIndex> for AddressingMode
{
	fn from(rec: RollingRecordIndex) -> Self { Self::RollingRecord(rec) }
}

impl TryFrom<AddressingMode> for Immediate
{
	type Error = ();

	fn try_from(value: AddressingMode) -> Result<Self, Self::Error>
	{
		match value
		{
			AddressingMode::Immediate(imm) => Ok(imm),
			_ => Err(())
		}
	}
}

impl TryFrom<AddressingMode> for RegisterIndex
{
	type Error = ();

	fn try_from(value: AddressingMode) -> Result<Self, Self::Error>
	{
		match value
		{
			AddressingMode::Register(reg) => Ok(reg),
			_ => Err(())
		}
	}
}

impl TryFrom<AddressingMode> for RollingRecordIndex
{
	type Error = ();

	fn try_from(value: AddressingMode) -> Result<Self, Self::Error>
	{
		match value
		{
			AddressingMode::RollingRecord(rec) => Ok(rec),
			_ => Err(())
		}
	}
}

/// A program counter, which references an instruction in a compiled function.
/// As there are no control flow instructions in the dice language, the program
/// counter is not really an addressing mode, per se, but is still useful for
/// various tools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramCounter(pub usize);

impl Display for ProgramCounter
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "*{}", self.0)
	}
}

impl From<usize> for ProgramCounter
{
	fn from(pc: usize) -> Self { Self(pc) }
}

impl From<ProgramCounter> for usize
{
	fn from(pc: ProgramCounter) -> Self { pc.0 }
}

/// Provides the ability to allocate a new instance of the receiver. This is
/// useful for generating new register and rolling record indices.
pub trait CanAllocate
{
	/// Allocate a new instance of the receiver.
	///
	/// # Returns
	/// The newly allocated instance.
	fn allocate(&mut self) -> Self;
}

impl CanAllocate for RegisterIndex
{
	fn allocate(&mut self) -> Self
	{
		let register = *self;
		self.0 += 1;
		register
	}
}

impl CanAllocate for RollingRecordIndex
{
	fn allocate(&mut self) -> Self
	{
		let record = *self;
		self.0 += 1;
		record
	}
}

impl CanAllocate for ProgramCounter
{
	fn allocate(&mut self) -> Self
	{
		let pc = *self;
		self.0 += 1;
		pc
	}
}

////////////////////////////////////////////////////////////////////////////////
//                          Instruction visitation.                           //
////////////////////////////////////////////////////////////////////////////////

/// A visitor for instructions in the intermediate representation of the dice
/// language.
///
/// # Type Parameters
/// - `E`: The type of error that the visitor may return.
pub trait InstructionVisitor<E>
{
	/// Visit a range roll instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), E>;

	/// Visit a standard dice roll instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), E>;

	/// Visit a custom dice roll instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), E>;

	/// Visit a drop lowest instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), E>;

	/// Visit a drop highest instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), E>;

	/// Visit a sum rolling record instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), E>;

	/// Visit an addition instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_add(&mut self, inst: &Add) -> Result<(), E>;

	/// Visit a subtraction instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_sub(&mut self, inst: &Sub) -> Result<(), E>;

	/// Visit a multiplication instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_mul(&mut self, inst: &Mul) -> Result<(), E>;

	/// Visit a division instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_div(&mut self, inst: &Div) -> Result<(), E>;

	/// Visit a modulo instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_mod(&mut self, inst: &Mod) -> Result<(), E>;

	/// Visit an exponentiation instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_exp(&mut self, inst: &Exp) -> Result<(), E>;

	/// Visit a negation instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_neg(&mut self, inst: &Neg) -> Result<(), E>;

	/// Visit a return instruction.
	///
	/// # Parameters
	/// - `inst`: The instruction to visit.
	///
	/// # Errors
	/// An error if the visitor fails to visit the instruction.
	fn visit_return(&mut self, inst: &Return) -> Result<(), E>;
}

/// Provides the ability to apply a [visitor](InstructionVisitor).
///
/// # Type Parameters
/// - `V`: The type of visitor to apply.
/// - `E`: The type of error that can occur while visiting.
pub trait CanVisitInstructions<V, E>
where
	V: InstructionVisitor<E>
{
	/// Apply the specified [visitor](InstructionVisitor) to the receiver.
	///
	/// # Parameters
	/// - `visitor`: The visitor to apply.
	///
	/// # Errors
	/// An error if one occurs while visiting the receiver.
	fn visit(&self, visitor: &mut V) -> Result<(), E>;
}

impl<V, E> CanVisitInstructions<V, E> for Instruction
where
	V: InstructionVisitor<E>
{
	fn visit(&self, visitor: &mut V) -> Result<(), E>
	{
		match self
		{
			Instruction::RollRange(inst) => visitor.visit_roll_range(inst),
			Instruction::RollStandardDice(inst) =>
			{
				visitor.visit_roll_standard_dice(inst)
			},
			Instruction::RollCustomDice(inst) =>
			{
				visitor.visit_roll_custom_dice(inst)
			},
			Instruction::DropLowest(inst) => visitor.visit_drop_lowest(inst),
			Instruction::DropHighest(inst) => visitor.visit_drop_highest(inst),
			Instruction::SumRollingRecord(inst) =>
			{
				visitor.visit_sum_rolling_record(inst)
			},
			Instruction::Add(inst) => visitor.visit_add(inst),
			Instruction::Sub(inst) => visitor.visit_sub(inst),
			Instruction::Mul(inst) => visitor.visit_mul(inst),
			Instruction::Div(inst) => visitor.visit_div(inst),
			Instruction::Mod(inst) => visitor.visit_mod(inst),
			Instruction::Exp(inst) => visitor.visit_exp(inst),
			Instruction::Neg(inst) => visitor.visit_neg(inst),
			Instruction::Return(inst) => visitor.visit_return(inst)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Register dependencies.                           //
////////////////////////////////////////////////////////////////////////////////

/// Analyzes the dependencies between instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyAnalyzer<'inst>
{
	// The function being analyzed.
	instructions: &'inst [Instruction],

	/// The program counter. Always set to the correct value before visiting an
	/// instruction.
	pc: ProgramCounter,

	/// The register and rolling record writers, indexed by target. When the
	/// function is in static single assignment (SSA) form, then each register
	/// and rolling record writer will only have one writer.
	writers: HashMap<AddressingMode, BTreeSet<ProgramCounter>>,

	/// The register and rolling record readers, indexed by source.
	readers: HashMap<AddressingMode, BTreeSet<ProgramCounter>>,

	/// The transitive register and rolling record dependencies, as a map from
	/// registers to all instructions that contribute to them.
	transitive_dependencies:
		RefCell<Option<HashMap<AddressingMode, BTreeSet<ProgramCounter>>>>,

	/// The transitive register and rolling record dependents, as a map from
	/// registers to all instructions that consume from them.
	transitive_dependents:
		RefCell<Option<HashMap<AddressingMode, BTreeSet<ProgramCounter>>>>
}

impl<'inst> DependencyAnalyzer<'inst>
{
	/// Analyze the dependencies between instructions in the specified function
	/// body.
	///
	/// # Parameters
	/// - `instructions`: The function body to analyze.
	///
	/// # Returns
	/// The populated dependency analyzer.
	pub fn analyze(instructions: &'inst [Instruction]) -> Self
	{
		let mut analyzer = Self {
			instructions,
			pc: ProgramCounter::default(),
			writers: HashMap::new(),
			readers: HashMap::new(),
			transitive_dependencies: RefCell::default(),
			transitive_dependents: RefCell::default()
		};
		// Visit each instruction in the function body.
		for instruction in analyzer.instructions
		{
			instruction.visit(&mut analyzer).unwrap();
			analyzer.pc.allocate();
		}
		analyzer
	}

	/// Answer the instructions being analyzed.
	///
	/// # Returns
	/// The requested instructions.
	pub fn instructions(&self) -> &'inst [Instruction] { self.instructions }

	/// Answer the direct writers of registers and rolling records, indexed by
	/// target.
	///
	/// # Returns
	/// The requested map.
	pub fn writers(&self)
	-> &HashMap<AddressingMode, BTreeSet<ProgramCounter>>
	{
		&self.writers
	}

	/// Answer the direct readers of registers and rolling records, indexed by
	/// source.
	///
	/// # Returns
	/// The requested map.
	pub fn readers(&self)
	-> &HashMap<AddressingMode, BTreeSet<ProgramCounter>>
	{
		&self.readers
	}

	/// Compute the transitive dependencies of registers and rolling records, as
	/// a map from registers to all instructions that contribute to them.
	///
	/// # Returns
	/// The requested map.
	pub fn transitive_dependencies(
		&self
	) -> Ref<'_, HashMap<AddressingMode, BTreeSet<ProgramCounter>>>
	{
		// If a cached value is available, then return it.
		if self.transitive_dependencies.borrow().is_some()
		{
			return Ref::map(self.transitive_dependencies.borrow(), |x| {
				x.as_ref().unwrap()
			})
		}
		let mut transitive_dependencies =
			HashMap::<AddressingMode, BTreeSet<ProgramCounter>>::new();
		// This is a worklist algorithm. Start with the direct writers of each
		// register and rolling record, and then add all of the writers of each
		// source register, and so on, until no new dependencies are found.
		for (dest, writers) in &self.writers
		{
			let mut all_dependencies = BTreeSet::new();
			let mut to_visit = writers.clone();
			while let Some(pc) = to_visit.pop_first()
			{
				if all_dependencies.insert(pc)
				{
					// This instruction contributes to the register or rolling
					// record and it hasn't been encountered yet. Enqueue all of
					// the writers of its sources to the worklist.
					let dependency = &self.instructions[pc.0];
					dependency.sources().iter().for_each(|src| {
						if let Some(writers) = self.writers.get(src)
						{
							// Only registers and rolling records that do not
							// represent parameters or external variables
							// have writers.
							to_visit.extend(writers.iter().clone());
						}
					});
				}
			}
			transitive_dependencies.insert(*dest, all_dependencies);
		}
		*self.transitive_dependencies.borrow_mut() =
			Some(transitive_dependencies);
		Ref::map(self.transitive_dependencies.borrow(), |x| {
			x.as_ref().unwrap()
		})
	}

	/// Compute the transitive dependents of registers and rolling records, as
	/// a map from registers to all instructions that consume from them.
	///
	/// # Returns
	/// The requested map.
	pub fn transitive_dependents(
		&self
	) -> Ref<'_, HashMap<AddressingMode, BTreeSet<ProgramCounter>>>
	{
		// If a cached value is available, then return it.
		if self.transitive_dependents.borrow().is_some()
		{
			return Ref::map(self.transitive_dependents.borrow(), |x| {
				x.as_ref().unwrap()
			})
		}
		let mut transitive_dependents =
			HashMap::<AddressingMode, BTreeSet<ProgramCounter>>::new();
		for (dest, readers) in &self.readers
		{
			let mut all_dependents = BTreeSet::new();
			let mut to_visit = readers.clone();
			while let Some(pc) = to_visit.pop_first()
			{
				if all_dependents.insert(pc)
				{
					// This instruction consumes from the register or rolling
					// record and it hasn't been encountered yet. Enqueue all of
					// the readers of its destination to the worklist.
					let dependency = &self.instructions[pc.0];
					if let Some(dest) = dependency.destination()
					{
						if let Some(readers) = self.readers.get(&dest)
						{
							// Only registers and rolling records that do not
							// represent parameters or external variables have
							// readers.
							to_visit.extend(readers.iter().clone());
						}
					}
				}
			}
			transitive_dependents.insert(*dest, all_dependents);
		}
		*self.transitive_dependents.borrow_mut() = Some(transitive_dependents);
		Ref::map(self.transitive_dependents.borrow(), |x| x.as_ref().unwrap())
	}
}

impl InstructionVisitor<()> for DependencyAnalyzer<'_>
{
	fn visit_roll_range(&mut self, inst: &RollRange) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.start).or_default().insert(self.pc);
		self.readers.entry(inst.end).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_roll_standard_dice(
		&mut self,
		inst: &RollStandardDice
	) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.count).or_default().insert(self.pc);
		self.readers.entry(inst.faces).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_roll_custom_dice(
		&mut self,
		inst: &RollCustomDice
	) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.count).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_drop_lowest(&mut self, inst: &DropLowest) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.count).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_drop_highest(&mut self, inst: &DropHighest) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.count).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_sum_rolling_record(
		&mut self,
		inst: &SumRollingRecord
	) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers
			.entry(inst.src.into())
			.or_default()
			.insert(self.pc);
		Ok(())
	}

	fn visit_add(&mut self, inst: &Add) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_sub(&mut self, inst: &Sub) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_mul(&mut self, inst: &Mul) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_div(&mut self, inst: &Div) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_mod(&mut self, inst: &Mod) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_exp(&mut self, inst: &Exp) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op1).or_default().insert(self.pc);
		self.readers.entry(inst.op2).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_neg(&mut self, inst: &Neg) -> Result<(), ()>
	{
		self.writers
			.entry(inst.dest.into())
			.or_default()
			.insert(self.pc);
		self.readers.entry(inst.op).or_default().insert(self.pc);
		Ok(())
	}

	fn visit_return(&mut self, inst: &Return) -> Result<(), ()>
	{
		self.readers.entry(inst.src).or_default().insert(self.pc);
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                   Tests.                                   //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test
{
	use std::collections::BTreeSet;

	use pretty_assertions::assert_eq;

	use crate::{
		AddressingMode, DependencyAnalyzer, Function, Immediate, Instruction,
		ProgramCounter, RegisterIndex
	};

	/// Answer the function `x: 1 + {x} + 1 + {x} + 1 + {x}`, which is used by
	/// the tests in this module.
	///
	/// # Returns
	/// The function described above.
	fn function() -> Function
	{
		// x: 1 + {x} + 1 + {x} + 1 + {x}
		// =
		// Function(x@0) r#6 ⚅#0   (pc)
		//   @1 <- 1 + @0         #0
		//   @2 <- @1 + 1         #1
		//   @3 <- @2 + @0        #2
		//   @4 <- @3 + 1         #3
		//   @5 <- @4 + @0        #4
		//   return @5            #5
		Function {
			parameters: vec!["x".to_string()],
			externals: Vec::new(),
			register_count: 6,
			rolling_record_count: 0,
			instructions: vec![
				Instruction::add(
					1.into(),
					Immediate(1).into(),
					RegisterIndex(0).into()
				),
				Instruction::add(
					2.into(),
					RegisterIndex(1).into(),
					Immediate(1).into()
				),
				Instruction::add(
					3.into(),
					RegisterIndex(2).into(),
					RegisterIndex(0).into()
				),
				Instruction::add(
					4.into(),
					RegisterIndex(3).into(),
					Immediate(1).into()
				),
				Instruction::add(
					5.into(),
					RegisterIndex(4).into(),
					RegisterIndex(0).into()
				),
				Instruction::r#return(RegisterIndex(5).into()),
			]
		}
	}

	/// Ensure that direct dependency analysis produces the correct writer map.
	#[test]
	fn test_analyze_writers()
	{
		let function = function();
		let analyzer = DependencyAnalyzer::analyze(&function.instructions);
		assert_eq!(
			analyzer.writers(),
			&[
				(
					AddressingMode::Register(RegisterIndex(1)),
					BTreeSet::from([ProgramCounter(0)])
				),
				(
					AddressingMode::Register(RegisterIndex(2)),
					BTreeSet::from([ProgramCounter(1)])
				),
				(
					AddressingMode::Register(RegisterIndex(3)),
					BTreeSet::from([ProgramCounter(2)])
				),
				(
					AddressingMode::Register(RegisterIndex(4)),
					BTreeSet::from([ProgramCounter(3)])
				),
				(
					AddressingMode::Register(RegisterIndex(5)),
					BTreeSet::from([ProgramCounter(4)])
				)
			]
			.into()
		);
	}

	/// Ensure that direct dependency analysis produces the correct reader map.
	#[test]
	fn test_analyze_readers()
	{
		let function = function();
		let analyzer = DependencyAnalyzer::analyze(&function.instructions);
		assert_eq!(
			analyzer.readers(),
			&[
				(
					AddressingMode::Immediate(Immediate(1)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(3)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(0)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(2),
						ProgramCounter(4)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(1)),
					BTreeSet::from([ProgramCounter(1)])
				),
				(
					AddressingMode::Register(RegisterIndex(2)),
					BTreeSet::from([ProgramCounter(2)])
				),
				(
					AddressingMode::Register(RegisterIndex(3)),
					BTreeSet::from([ProgramCounter(3)])
				),
				(
					AddressingMode::Register(RegisterIndex(4)),
					BTreeSet::from([ProgramCounter(4)])
				),
				(
					AddressingMode::Register(RegisterIndex(5)),
					BTreeSet::from([ProgramCounter(5)])
				),
			]
			.into()
		);
	}

	/// Ensure that transitive dependency analysis produces the correct
	/// dependencies map.
	#[test]
	fn test_dependencies()
	{
		let function = function();
		let analyzer = DependencyAnalyzer::analyze(&function.instructions);
		let dependencies = analyzer.transitive_dependencies();
		assert_eq!(
			*dependencies,
			[
				(
					AddressingMode::Register(RegisterIndex(1)),
					BTreeSet::from([ProgramCounter(0)])
				),
				(
					AddressingMode::Register(RegisterIndex(2)),
					BTreeSet::from([ProgramCounter(0), ProgramCounter(1)])
				),
				(
					AddressingMode::Register(RegisterIndex(3)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(2)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(4)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(2),
						ProgramCounter(3)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(5)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(2),
						ProgramCounter(3),
						ProgramCounter(4)
					])
				)
			]
			.into()
		);
	}

	/// Ensure that transitive dependency analysis produces the correct
	/// dependents map.
	#[test]
	fn test_dependents()
	{
		let function = function();
		let analyzer = DependencyAnalyzer::analyze(&function.instructions);
		let dependents = analyzer.transitive_dependents();
		assert_eq!(
			*dependents,
			[
				(
					AddressingMode::Immediate(Immediate(1)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(2),
						ProgramCounter(3),
						ProgramCounter(4),
						ProgramCounter(5)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(0)),
					BTreeSet::from([
						ProgramCounter(0),
						ProgramCounter(1),
						ProgramCounter(2),
						ProgramCounter(3),
						ProgramCounter(4),
						ProgramCounter(5)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(1)),
					BTreeSet::from([
						ProgramCounter(1),
						ProgramCounter(2),
						ProgramCounter(3),
						ProgramCounter(4),
						ProgramCounter(5)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(2)),
					BTreeSet::from([
						ProgramCounter(2),
						ProgramCounter(3),
						ProgramCounter(4),
						ProgramCounter(5)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(3)),
					BTreeSet::from([
						ProgramCounter(3),
						ProgramCounter(4),
						ProgramCounter(5)
					])
				),
				(
					AddressingMode::Register(RegisterIndex(4)),
					BTreeSet::from([ProgramCounter(4), ProgramCounter(5)])
				),
				(
					AddressingMode::Register(RegisterIndex(5)),
					BTreeSet::from([ProgramCounter(5)])
				)
			]
			.into()
		);
	}
}
