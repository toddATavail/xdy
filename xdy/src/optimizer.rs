//! # Optimization
//!
//! The optimizer is capable of reducing the size of the generated
//! intermediate representation (IR) and improving its execution speed. The
//! optimizer operates on a [function](Function), which comprises a single basic
//! block of instructions in static single assignment (SSA) form. The optimizer
//! applies a series of transformations to the function, which may include:
//!
//! - Common subexpression elimination
//! - Constant commuting
//! - Constant folding
//! - Strength reduction
//! - Dead code elimination
//! - Register coalescing
//!
//! The optimizer is implemented as a series of passes, each of which applies a
//! single transformation to the function. The passes are applied in sequence,
//! and the optimizer continues to apply passes until no further changes are
//! made to the function. Register coalescing is the final pass, as it breaks
//! SSA form.

mod coalescence;
mod commutation;
mod cse;
mod dce;
mod folding;
mod reduction;

use std::{collections::HashSet, ops::BitOr};

use coalescence::RegisterCoalescer;
use commutation::ConstantCommuter;
use cse::CommonSubexpressionEliminator;
use dce::DeadCodeEliminator;
use folding::ConstantFolder;
use reduction::StrengthReducer;

use crate::{
	AddressingMode, EvaluationBounds, Function, Immediate, Instruction,
	Predicate, RollingRecordKind,
	evaluator::{BoundsEvaluator, Criteria, ResultSet}
};

////////////////////////////////////////////////////////////////////////////////
//                                Optimizers.                                 //
////////////////////////////////////////////////////////////////////////////////

/// An optimizer performs a specific optimization on a function. It might be
/// one of the standard optimization passes or some custom client-provided
/// optimization.
pub trait Optimizer<E>
{
	/// Optimize a function.
	///
	/// # Parameters
	/// - `function`: The function to optimize. The standard optimizer and its
	///   passes require it to be in [static single assignment
	///   form](StandardOptimizer#static-single-assignment-form).
	///
	/// # Returns
	/// The optimized function.
	///
	/// # Errors
	/// An error is returned if the function cannot be optimized.
	fn optimize(self, function: Function) -> Result<Function, E>;
}

/// The standard optimizer applies the specified standard optimization passes to
/// a [function](Function). The passes are applied in sequence, and the
/// optimizer continues to apply passes until no further changes are made to the
/// function.
///
/// # Static single assignment form
///
/// Every pass but register coalescence requires the function to be in static
/// single assignment (SSA) form, as the [compiler](crate::Compiler) emits it:
///
/// - No instruction writes a register that holds a parameter or an external
///   variable, and at most one instruction writes each other register.
/// - At most one roll writes each rolling record, before any drop, keep,
///   reroll, or explode of it. Drops, keeps, rerolls, and explodes rewrite the
///   record in place, and are the only exception to single assignment.
///
/// Register coalescence breaks SSA form, so it runs last, and a function that
/// it has coalesced, or a function assembled by hand that reuses a register,
/// is not fit to optimize again. The passes do not check their input, so
/// such a function may be miscompiled, or may cause a panic; debug builds
/// assert the form before the first pass.
///
/// # Fixed-point optimization loop
///
/// ```mermaid
/// graph TD
///     A["Input Function (SSA)"] --> B["CSE"]
///     B --> C["Constant Commuting"]
///     C --> D["Constant Folding"]
///     D --> E["Strength Reduction"]
///     E --> F["Dead Code Elimination"]
///     F --> G{"Changed?"}
///     G -- Yes --> B
///     G -- No --> H["Register Coalescing"]
///     H --> I["Optimized Function"]
///     style A fill:#f9f,stroke:#333,color:#000
///     style I fill:#9f9,stroke:#333,color:#000
///     style G fill:#ff9,stroke:#333,color:#000
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct StandardOptimizer(Passes);

impl StandardOptimizer
{
	/// Create a new standard optimizer.
	///
	/// # Parameters
	/// - `passes`: The passes to apply.
	///
	/// # Returns
	/// The requested optimizer.
	#[inline]
	pub fn new(passes: Passes) -> Self { Self(passes) }
}

impl Optimizer<()> for StandardOptimizer
{
	fn optimize(self, function: Function) -> Result<Function, ()>
	{
		if self.0.is_empty()
		{
			// No passes to apply; return the function as-is.
			return Ok(function)
		}
		debug_assert!(
			ssa_violation(&function).is_none(),
			"the optimizer requires static single assignment form, which \
			 instruction {} violates:\n{function}",
			ssa_violation(&function).unwrap_or_default()
		);
		let mut previous = function;
		loop
		{
			let mut optimized = previous.clone();
			// The internal unwraps are safe because none of the standard passes
			// ever actually fail. We prefer to panic if something goes wrong,
			// as it indicates a legitimate bug in the optimizer.
			if self.0.contains(Pass::CommonSubexpressionElimination)
			{
				optimized = CommonSubexpressionEliminator::default()
					.optimize(optimized)
					.unwrap();
			}
			if self.0.contains(Pass::ConstantCommuting)
			{
				optimized.instructions =
					ConstantCommuter::commute(&optimized.instructions);
			}
			if self.0.contains(Pass::ConstantFolding)
			{
				optimized =
					ConstantFolder::default().optimize(optimized).unwrap();
			}
			if self.0.contains(Pass::StrengthReduction)
			{
				optimized =
					StrengthReducer::default().optimize(optimized).unwrap();
			}
			if self.0.contains(Pass::DeadCodeElimination)
			{
				optimized =
					DeadCodeEliminator::default().optimize(optimized).unwrap();
			}
			// All passes have been applied for this cycle. If no changes were
			// made, the function is fully optimized and ready for potential
			// register coalescing.
			if optimized == previous
			{
				if self.0.contains(Pass::RegisterCoalescing)
				{
					optimized = RegisterCoalescer::default()
						.optimize(optimized)
						.unwrap();
				}
				return Ok(optimized)
			}
			previous = optimized;
		}
	}
}

/// Answer the first instruction of the function that violates [static single
/// assignment form](StandardOptimizer#static-single-assignment-form), if any.
///
/// # Parameters
/// - `function`: The function.
///
/// # Returns
/// The index of the first instruction that writes a register that holds a
/// binding or that an earlier instruction wrote, or that rolls into a rolling
/// record that an earlier instruction wrote, or `None` if there is none.
fn ssa_violation(function: &Function) -> Option<usize>
{
	let bindings = function.arity() + function.externals.len();
	let mut registers = HashSet::new();
	let mut records = HashSet::new();
	function.instructions.iter().position(|inst| {
		match inst.destination()
		{
			Some(AddressingMode::Register(reg)) =>
			{
				reg.0 < bindings || !registers.insert(reg)
			},
			Some(AddressingMode::RollingRecord(rec)) =>
			{
				let rolls = matches!(
					inst,
					Instruction::RollRange(_)
						| Instruction::RollStandardDice(_)
						| Instruction::RollCustomDice(_)
				);
				// A roll must be the first write of its record.
				!records.insert(rec) && rolls
			},
			Some(AddressingMode::Immediate(_)) | None => false
		}
	})
}

/// Answer the bounds of an operand as the passes know them: a single value
/// for an [immediate](Immediate), and the whole of `i32` otherwise.
///
/// # Parameters
/// - `op`: The operand.
///
/// # Returns
/// The bounds of the operand.
fn operand_bounds(op: AddressingMode) -> EvaluationBounds
{
	match op
	{
		AddressingMode::Immediate(Immediate(value)) => value.into(),
		_ => EvaluationBounds::unconstrained()
	}
}

/// Answer the bounds of every register of a function, over every binding of
/// its parameters and external variables, as the [bounds
/// evaluator](BoundsEvaluator) computes them.
///
/// # Parameters
/// - `function`: The function, in [static single assignment
///   form](StandardOptimizer#static-single-assignment-form), so that each
///   register holds the value of the sole instruction that writes it.
///
/// # Returns
/// The bounds of each register, by index, which contain every value that it
/// may hold over any binding.
fn register_bounds(function: &Function) -> Vec<EvaluationBounds>
{
	let mut bounds = BoundsEvaluator::new(function);
	// Every binding is unconstrained, so the arity always agrees.
	bounds
		.seed(
			vec![None; function.arity()],
			[],
			EvaluationBounds::unconstrained()
		)
		.unwrap();
	for instruction in &function.instructions
	{
		bounds.step(instruction);
	}
	bounds.registers().to_vec()
}

/// Answer the faces that a die of the specified roll may show, given the
/// operands of the roll as they stand. A die without faces, or an empty
/// range, shows only the face `0`.
///
/// # Parameters
/// - `roll`: The roll, a [range](crate::RollRange), [standard
///   dice](crate::RollStandardDice), or [custom dice](crate::RollCustomDice).
///
/// # Returns
/// The faces, which are exact if every operand of the roll that decides its
/// faces is an [immediate](Immediate), and otherwise include every face that
/// the die may have.
///
/// # Panics
/// If the instruction is not a roll.
fn faces(roll: &Instruction) -> ResultSet
{
	let kind = match roll
	{
		Instruction::RollRange(range) => RollingRecordKind::Range {
			start: operand_bounds(range.start),
			end: operand_bounds(range.end)
		},
		Instruction::RollStandardDice(roll) => RollingRecordKind::Standard {
			count: operand_bounds(roll.count),
			faces: operand_bounds(roll.faces)
		},
		Instruction::RollCustomDice(roll) => RollingRecordKind::Custom {
			count: operand_bounds(roll.count),
			faces: roll.faces.clone()
		},
		_ => unreachable!("not a roll: {}", roll)
	};
	kind.faces()
}

/// Answer the [criteria](Criteria) to which the specified predicate of a
/// [reroll](crate::Reroll) or [explode](crate::Explode) may resolve, given its
/// operands as they stand. If every operand is an [immediate](Immediate), then
/// the predicate resolves to a single criterion, which the criteria's
/// `certain` and `possible` results both match exactly.
///
/// # Parameters
/// - `predicate`: The predicate.
///
/// # Returns
/// The criteria: `certain` holds the results that the predicate surely
/// matches, and `possible` the results that it may match.
fn criteria(predicate: &Predicate) -> Criteria
{
	Criteria::new(predicate, operand_bounds)
}

////////////////////////////////////////////////////////////////////////////////
//                            Optimization passes.                            //
////////////////////////////////////////////////////////////////////////////////

/// A standard optimization. Each pass applies a single optimization to a
/// function. Each pass is represented as a single bit in a bitfield of
/// [passes](Passes). The passes are performed by the standard optimizer in
/// definition order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pass
{
	/// Eliminate common subexpressions by identifying and replacing identical
	/// expressions. Iterate until no further changes are made.
	CommonSubexpressionElimination = 0x01,

	/// Put the operands of commutative instructions in canonical order,
	/// immediates first, and merge an instruction that applies a constant to
	/// the result of another that applies one, e.g., `(x + 1) + 2` into
	/// `x + 3`, wherever the merged instruction is exact despite saturation,
	/// thereby creating new opportunities for constant folding and strength
	/// reduction. Only a single iteration is performed, but it merges whole
	/// chains of such instructions, and the standard optimizer reapplies it
	/// until its fixed point is reached.
	ConstantCommuting = 0x02,

	/// Fold expressions with constant operands into
	/// [immediates](crate::ir::Immediate). Iterate until no further changes are
	/// made.
	ConstantFolding = 0x04,

	/// Reduce the strength of expressions by replacing expensive operations
	/// with cheaper ones, and eliminate a maximum or minimum whose operands'
	/// bounds prove that one always wins. Iterate until no further changes are
	/// made.
	StrengthReduction = 0x08,

	/// Eliminate dead code by removing instructions whose results are never
	/// used. Iterate until no further changes are made.
	DeadCodeElimination = 0x10,

	/// Coalesce registers by reusing registers whose live ranges do not
	/// overlap. This pass breaks SSA form, so it is the final pass.
	RegisterCoalescing = 0x80
}

impl BitOr<Pass> for Pass
{
	type Output = Passes;

	fn bitor(self, rhs: Pass) -> Self::Output { Passes(self as u8 | rhs as u8) }
}

impl BitOr<Passes> for Pass
{
	type Output = Passes;

	fn bitor(self, rhs: Passes) -> Self::Output { Passes(self as u8 | rhs.0) }
}

/// A set of optimization passes. An [optimizer](Optimizer) applies a
/// [set of passes](Passes) to a function until a fixed point is reached. The
/// [default](Passes::default) is [every pass](Passes::all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Passes(u8);

impl Passes
{
	/// Create a complete set of passes.
	///
	/// # Returns
	/// A set of all optimization passes.
	#[inline]
	pub fn all() -> Self { Self(0xFF) }

	/// Create an empty set of passes. An [optimizer](StandardOptimizer) with no
	/// passes answers its function unchanged.
	///
	/// # Returns
	/// A set of no optimization passes.
	#[inline]
	pub fn none() -> Self { Self(0) }

	/// Check if the receiver is empty.
	///
	/// # Returns
	/// `true` if the receiver is empty, otherwise `false`.
	#[inline]
	pub fn is_empty(&self) -> bool { self.0 == 0 }

	/// Check if the receiver contains a specific pass.
	///
	/// # Parameters
	/// - `pass`: The pass to check.
	///
	/// # Returns
	/// `true` if the pass is present in the receiver, otherwise `false`.
	#[inline]
	pub fn contains(&self, pass: Pass) -> bool { self.0 & (pass as u8) != 0 }
}

impl Default for Passes
{
	/// Answer the default set of passes, which is [every pass](Passes::all),
	/// so that a default optimizer optimizes fully.
	///
	/// # Returns
	/// A set of all optimization passes.
	#[inline]
	fn default() -> Self { Self::all() }
}

impl std::ops::BitOr<Pass> for Passes
{
	type Output = Self;

	fn bitor(self, rhs: Pass) -> Self::Output { Self(self.0 | rhs as u8) }
}

impl std::ops::BitOr<Passes> for Passes
{
	type Output = Self;

	fn bitor(self, rhs: Passes) -> Self::Output { Self(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign<Pass> for Passes
{
	fn bitor_assign(&mut self, rhs: Pass) { self.0 |= rhs as u8; }
}

impl From<Pass> for Passes
{
	fn from(pass: Pass) -> Self { Self(pass as u8) }
}

////////////////////////////////////////////////////////////////////////////////
//                                   Tests.                                   //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests
{
	use pretty_assertions::assert_eq;

	use crate::{
		Pass, Passes,
		support::{compile_valid, optimize}
	};

	// Test that the passes are correctly defined.
	#[test]
	fn test_passes()
	{
		let passes = Passes::all();
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(passes.contains(Pass::ConstantCommuting));
		assert!(passes.contains(Pass::ConstantFolding));
		assert!(passes.contains(Pass::StrengthReduction));
		assert!(passes.contains(Pass::DeadCodeElimination));
		assert!(passes.contains(Pass::RegisterCoalescing));

		let passes = Passes::from(Pass::CommonSubexpressionElimination);
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(!passes.contains(Pass::ConstantCommuting));
		assert!(!passes.contains(Pass::ConstantFolding));
		assert!(!passes.contains(Pass::StrengthReduction));
		assert!(!passes.contains(Pass::DeadCodeElimination));
		assert!(!passes.contains(Pass::RegisterCoalescing));

		let passes =
			Pass::CommonSubexpressionElimination | Pass::ConstantCommuting;
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(passes.contains(Pass::ConstantCommuting));
		assert!(!passes.contains(Pass::ConstantFolding));
		assert!(!passes.contains(Pass::StrengthReduction));
		assert!(!passes.contains(Pass::DeadCodeElimination));
		assert!(!passes.contains(Pass::RegisterCoalescing));

		let passes = Passes::from(Pass::ConstantCommuting)
			| Pass::CommonSubexpressionElimination;
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(passes.contains(Pass::ConstantCommuting));
		assert!(!passes.contains(Pass::ConstantFolding));
		assert!(!passes.contains(Pass::StrengthReduction));
		assert!(!passes.contains(Pass::DeadCodeElimination));
		assert!(!passes.contains(Pass::RegisterCoalescing));

		let passes = Pass::CommonSubexpressionElimination
			| Passes::from(Pass::ConstantCommuting);
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(passes.contains(Pass::ConstantCommuting));
		assert!(!passes.contains(Pass::ConstantFolding));
		assert!(!passes.contains(Pass::StrengthReduction));
		assert!(!passes.contains(Pass::DeadCodeElimination));
		assert!(!passes.contains(Pass::RegisterCoalescing));

		let passes = Passes::none()
			| Pass::CommonSubexpressionElimination
			| Pass::ConstantCommuting
			| Pass::ConstantFolding
			| Pass::StrengthReduction
			| Pass::DeadCodeElimination
			| Pass::RegisterCoalescing;
		assert!(passes.contains(Pass::CommonSubexpressionElimination));
		assert!(passes.contains(Pass::ConstantCommuting));
		assert!(passes.contains(Pass::ConstantFolding));
		assert!(passes.contains(Pass::StrengthReduction));
		assert!(passes.contains(Pass::DeadCodeElimination));
		assert!(passes.contains(Pass::RegisterCoalescing));
	}

	/// Test that running the optimizer without any passes does not change the
	/// input function.
	#[test]
	fn test_no_passes()
	{
		let passes = Passes::none();
		assert!(passes.is_empty());
		let function = compile_valid("{x}: 3D6 + {x}");
		assert_eq!(function, optimize(function.clone(), Passes::none()));
	}

	/// Test that the default set of passes is every pass.
	#[test]
	fn test_default_passes()
	{
		assert_eq!(Passes::default(), Passes::all());
	}
}
