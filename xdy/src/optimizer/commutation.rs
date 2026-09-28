//! # Commutation
//!
//! Commutative operations can reorder their operands without changing the
//! meaning or result. Addition, multiplication, and maximum are commutative,
//! so this pass puts the operands of each such instruction in a canonical
//! order, immediates before registers, which lets common subexpression
//! elimination recognize `{x} + 1` and `1 + {x}` as the same. It also merges an
//! instruction that applies a constant to the result of another that applies a
//! constant, e.g., `(x + 1) + 2`, into one that applies a single constant,
//! e.g., `x + 3`, creating new opportunities for constant folding and strength
//! reduction.
//!
//! The arithmetic of the dice language saturates at the bounds of `i32`, so
//! regrouping is not exact in general: `(x + 1) + y` and `(x + y) + 1` differ
//! when `x` is `i32::MAX` and `y` is `-1`. The pass therefore merges only
//! where the merged instruction answers exactly what the pair answered, for
//! every value of `x`:
//!
//! | Pair                       | Merged          | When                          |
//! |----------------------------|-----------------|-------------------------------|
//! | `(x ± a) ± b`              | `x + (±a ± b)`  | the addends share a sign and their sum fits |
//! | `(x * a) * b`              | `x * (a * b)`   | `a` and `b` are positive and their product fits |
//! | `(x / a) / b`              | `x / (a * b)`   | `a` and `b` are positive and their product fits |
//! | `max(max(x, a), b)`        | `max(x, max(a, b))` | always                    |
//!
//! Addends that share a sign push a saturating sum toward the same bound, so
//! saturating once or twice agrees; a positive multiplier preserves the sign
//! of a saturated product; and truncating division by positive divisors
//! composes. The merged instruction replaces the outer one in place, and reads
//! only what the inner one read, which precedes both, so every register is
//! still written before it is read. The inner instruction is merged only if
//! nothing else reads its result, which [dead code
//! elimination](crate::Pass::DeadCodeElimination) then removes. The constant
//! commuter requires the function to be in static single assignment (SSA)
//! form.

use std::collections::HashMap;

use crate::{
	Add, AddressingMode, Div, Immediate, Instruction, Max, Mul, RegisterIndex,
	Sub
};

////////////////////////////////////////////////////////////////////////////////
//                                Commutation.                                //
////////////////////////////////////////////////////////////////////////////////

/// A commuter that puts the operands of commutative instructions in canonical
/// order and merges exactly composable pairs of instructions that apply
/// constants. The standard optimizer applies this pass to a
/// [function](crate::Function) before folding constants or reducing strength.
pub struct ConstantCommuter;

/// An instruction that applies a constant to a register, in the form that the
/// [commuter](ConstantCommuter) merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Application
{
	/// `x + delta`, from an addition, or from a subtraction of `-delta`.
	Offset(i32),

	/// `x * factor`.
	Scale(i32),

	/// `x / divisor`.
	Quotient(i32),

	/// `max(x, floor)`.
	Floor(i32)
}

impl ConstantCommuter
{
	/// Commute the operands of the commutative instructions of a function
	/// body into canonical order, and merge its exactly composable pairs of
	/// instructions that apply constants.
	///
	/// # Parameters
	/// - `instructions`: The function body, in SSA form.
	///
	/// # Returns
	/// The replacement function body, with the same number of instructions.
	pub fn commute(instructions: &[Instruction]) -> Vec<Instruction>
	{
		// Count the reads of each register, so that an inner instruction is
		// merged only if its sole reader is the outer one.
		let mut reads = HashMap::<RegisterIndex, usize>::new();
		for instruction in instructions
		{
			for source in instruction.sources()
			{
				if let AddressingMode::Register(register) = source
				{
					*reads.entry(register).or_default() += 1;
				}
			}
		}
		// The applications written by instructions already rewritten, by
		// destination, so that a chain merges from the inside out in a single
		// pass.
		let mut applications =
			HashMap::<RegisterIndex, (RegisterIndex, Application)>::new();
		let mut commuted = Vec::with_capacity(instructions.len());
		for instruction in instructions
		{
			let mut instruction = canonical(instruction);
			if let Some((dest, (operand, outer))) =
				Self::application(&instruction)
			{
				let merged = applications
					.get(&operand)
					.filter(|_| reads.get(&operand) == Some(&1))
					.and_then(|&(inner_operand, inner)| {
						merge(inner, outer)
							.map(|merged| (inner_operand, merged))
					});
				let (operand, application) = match merged
				{
					Some((inner_operand, merged)) =>
					{
						// The outer instruction now reads the inner one's
						// operand instead of its result.
						*reads.get_mut(&operand).unwrap() -= 1;
						*reads.entry(inner_operand).or_default() += 1;
						instruction = emit(dest, inner_operand, merged);
						(inner_operand, merged)
					},
					None => (operand, outer)
				};
				applications.insert(dest, (operand, application));
			}
			commuted.push(instruction);
		}
		commuted
	}

	/// Recognize an instruction that applies a constant to a register.
	///
	/// # Parameters
	/// - `instruction`: The instruction, with its operands in canonical order.
	///
	/// # Returns
	/// The destination, the register, and the application, if the instruction
	/// applies a constant to a register.
	fn application(
		instruction: &Instruction
	) -> Option<(RegisterIndex, (RegisterIndex, Application))>
	{
		use AddressingMode::{Immediate as Imm, Register as Reg};
		let (dest, operand, application) = match *instruction
		{
			Instruction::Add(Add {
				dest,
				op1: Imm(Immediate(c)),
				op2: Reg(x)
			}) => (dest, x, Application::Offset(c)),
			Instruction::Sub(Sub {
				dest,
				op1: Reg(x),
				op2: Imm(Immediate(c))
			}) => (dest, x, Application::Offset(c.checked_neg()?)),
			Instruction::Mul(Mul {
				dest,
				op1: Imm(Immediate(c)),
				op2: Reg(x)
			}) => (dest, x, Application::Scale(c)),
			Instruction::Div(Div {
				dest,
				op1: Reg(x),
				op2: Imm(Immediate(c))
			}) => (dest, x, Application::Quotient(c)),
			Instruction::Max(Max {
				dest,
				op1: Imm(Immediate(c)),
				op2: Reg(x)
			}) => (dest, x, Application::Floor(c)),
			_ => return None
		};
		Some((dest, (operand, application)))
	}
}

/// Answer the specified instruction with its operands in canonical order, if
/// it is commutative: immediates before registers, and registers in ascending
/// order.
///
/// # Parameters
/// - `instruction`: The instruction.
///
/// # Returns
/// The canonical instruction.
fn canonical(instruction: &Instruction) -> Instruction
{
	let order = |op1: AddressingMode, op2: AddressingMode| {
		if op2 < op1 { (op2, op1) } else { (op1, op2) }
	};
	match *instruction
	{
		Instruction::Add(Add { dest, op1, op2 }) =>
		{
			let (op1, op2) = order(op1, op2);
			Add { dest, op1, op2 }.into()
		},
		Instruction::Mul(Mul { dest, op1, op2 }) =>
		{
			let (op1, op2) = order(op1, op2);
			Mul { dest, op1, op2 }.into()
		},
		Instruction::Max(Max { dest, op1, op2 }) =>
		{
			let (op1, op2) = order(op1, op2);
			Max { dest, op1, op2 }.into()
		},
		ref instruction => instruction.clone()
	}
}

/// Merge two applications, the inner applied first, into one that answers
/// exactly what the pair answers, for every operand, if there is such an
/// application of the same kind.
///
/// # Parameters
/// - `inner`: The application applied first.
/// - `outer`: The application applied to the result of `inner`.
///
/// # Returns
/// The merged application, if the pair merges exactly.
fn merge(inner: Application, outer: Application) -> Option<Application>
{
	match (inner, outer)
	{
		// Addends of the same sign push toward the same bound, so saturating
		// once or twice agrees, provided that their sum does not itself
		// saturate.
		(Application::Offset(a), Application::Offset(b))
			if (a >= 0) == (b >= 0) || a == 0 || b == 0 =>
		{
			a.checked_add(b).map(Application::Offset)
		},
		// A positive factor keeps the sign of a saturated product, so the
		// product saturates at the same bound either way.
		(Application::Scale(a), Application::Scale(b)) if a > 0 && b > 0 =>
		{
			a.checked_mul(b).map(Application::Scale)
		},
		// Truncating division by positive divisors composes.
		(Application::Quotient(a), Application::Quotient(b))
			if a > 0 && b > 0 =>
		{
			a.checked_mul(b).map(Application::Quotient)
		},
		// The maximum is associative and commutative.
		(Application::Floor(a), Application::Floor(b)) =>
		{
			Some(Application::Floor(a.max(b)))
		},
		_ => None
	}
}

/// Construct the instruction that applies the specified application, in
/// canonical order.
///
/// # Parameters
/// - `dest`: The destination register.
/// - `operand`: The register to which the constant applies.
/// - `application`: The application.
///
/// # Returns
/// The instruction.
fn emit(
	dest: RegisterIndex,
	operand: RegisterIndex,
	application: Application
) -> Instruction
{
	let x = AddressingMode::Register(operand);
	let c = |c| AddressingMode::Immediate(Immediate(c));
	match application
	{
		// Subtract a negative offset, for legibility, unless its magnitude
		// does not fit.
		Application::Offset(delta) if delta < 0 && delta != i32::MIN => Sub {
			dest,
			op1: x,
			op2: c(-delta)
		}
		.into(),
		Application::Offset(delta) => Add {
			dest,
			op1: c(delta),
			op2: x
		}
		.into(),
		Application::Scale(factor) => Mul {
			dest,
			op1: c(factor),
			op2: x
		}
		.into(),
		Application::Quotient(divisor) => Div {
			dest,
			op1: x,
			op2: c(divisor)
		}
		.into(),
		Application::Floor(floor) => Max {
			dest,
			op1: c(floor),
			op2: x
		}
		.into()
	}
}
