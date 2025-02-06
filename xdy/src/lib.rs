//! # Dice
//!
//! Dice is a simple expression language that allows you to roll
//! complex dice expressions. It supports parameters and external
//! variables, range and dice expressions (xDy), discarding of
//! lowest and highest dice, and basic arithmetic operations. Rolls
//! are recorded, so clients may display them alongside the result.
//! An optimizer is provided to reduce code size and improve execution
//! speed.

mod compiler;
mod evaluator;
mod histogram;
mod ir;
mod optimizer;
#[cfg(any(test, feature = "bench"))]
pub mod support;
#[cfg(test)]
mod tests;

pub use compiler::*;
pub use evaluator::*;
pub use histogram::*;
pub use ir::*;
pub use optimizer::*;
