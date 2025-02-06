//! # Optimization
//!
//! The optimizer is capable of reducing the size of the generated
//! intermediate representation (IR) and improving its execution speed. The
//! optimizer operates on a [function](Function), which comprises a single basic
//! block of instructions in static single assignment (SSA) form. The optimizer
//! applies a series of transformations to the function, which may include:
//!
//! - Constant folding
//! - Common subexpression elimination
//! - Strength reduction
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

use std::ops::BitOr;

use coalescence::RegisterCoalescer;
use commutation::ConstantCommuter;
use cse::CommonSubexpressionEliminator;
use dce::DeadCodeEliminator;
use folding::ConstantFolder;
use reduction::StrengthReducer;

use crate::Function;

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
	/// - `function`: The function to optimize.
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

	/// Commute immediate operands to the left side of instructions, thereby
	/// creating new opportunities for constant folding and strength reduction.
	/// Iterate until no further changes are made.
	ConstantCommuting = 0x02,

	/// Fold expressions with constant operands into
	/// [immediates](crate::ir::Immediate). Iterate until no further changes are
	/// made.
	ConstantFolding = 0x04,

	/// Reduce the strength of expressions by replacing expensive operations
	/// with cheaper ones. Only a single iteration is performed.
	StrengthReduction = 0x08,

	/// Eliminate dead code by removing instructions whose results are never
	/// used. Iterate until no further changes are made.
	DeadCodeElimination = 0x10,

	/// Coalesce registers by merging equivalent registers. This pass breaks SSA
	/// form, so it is the final pass.
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
/// [set of passes](Passes) to a function until a fixed point is reached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Passes(u8);

impl Passes
{
	/// Create a complete set of passes.
	///
	/// # Returns
	/// A set of all optimization passes.
	#[inline]
	pub fn all() -> Self { Self(0xFF) }

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

		let passes = Passes::default()
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
		let passes = Passes::default();
		assert!(passes.is_empty());
		let function = compile_valid("x: 3D6 + {x}");
		assert_eq!(function, optimize(function.clone(), Passes::default()));
	}
}
