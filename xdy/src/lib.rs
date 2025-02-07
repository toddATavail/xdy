//! # Dice
//!
//! Dice is a simple expression language that allows you to roll
//! complex dice expressions. It supports parameters and external
//! variables, range and dice expressions (`xDy`), discarding of
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

////////////////////////////////////////////////////////////////////////////////
//                           Convenient evaluation.                           //
////////////////////////////////////////////////////////////////////////////////

/// Evaluate the dice expression using the given arguments, environment, and
/// pseudo-random number generator (`pRNG`). Missing external variables default
/// to zero during evaluation, but all parameters must be bound. Do not optimize
/// the function before evaluation.
///
/// This is a one-shot convenience function that compiles and evaluates the
/// dice expression in a single step. If you need to evaluate the same dice
/// expression multiple times, consider [compiling](compile) the function once
/// and reusing it with the [`Evaluator`] type.
///
/// # Parameters
/// - `source`: The source code to compile and evaluate.
/// - `args`: The arguments to the function.
/// - `environment`: The environment in which to evaluate the function, as a
///   vector of external variable bindings. The bindings are pairs of variable
///   names and values. Missing bindings default to zero.
/// - `rng`: The pseudo-random number generator to use for range and dice rolls.
///
/// # Returns
/// The result of the evaluation.
///
/// # Errors
/// * [`CompilationFailed`](EvaluationError::CompilationFailed) if the function
///   could not be compiled.
/// * [`BadArity`](EvaluationError::BadArity) if the number of arguments
///   provided disagrees with the number of formal parameters in the function
///   signature.
/// * [`UnrecognizedExternal`](EvaluationError::UnrecognizedExternal) if an
///   external variable is not recognized.
pub fn evaluate_unoptimized<'s, 'e, 'r, R>(
	source: &'s str,
	args: impl IntoIterator<Item = i32>,
	environment: impl IntoIterator<Item = (&'e str, i32)>,
	rng: &'r mut R
) -> Result<Evaluation, EvaluationError<'s>>
where
	R: rand::Rng + ?Sized,
	'e: 's
{
	let function = compile_unoptimized(source)?;
	let mut evaluator = Evaluator::new(function);
	for (name, value) in environment
	{
		evaluator.bind(name, value)?;
	}
	evaluator.evaluate(args, rng)
}

/// Evaluate the dice expression using the given arguments, environment, and
/// pseudo-random number generator (`pRNG`). Missing external variables default
/// to zero during evaluation, but all parameters must be bound. Optimize the
/// function using the [standard optimizer](StandardOptimizer).
///
/// This is a one-shot convenience function that compiles and evaluates the
/// dice expression in a single step. If you need to evaluate the same dice
/// expression multiple times, consider [compiling](compile) the function once
/// and reusing it with the [`Evaluator`] type.
///
/// # Parameters
/// - `source`: The source code to compile and evaluate.
/// - `args`: The arguments to the function.
/// - `environment`: The environment in which to evaluate the function, as a
///   vector of external variable bindings. The bindings are pairs of variable
///   names and values. Missing bindings default to zero.
/// - `rng`: The pseudo-random number generator to use for range and dice rolls.
///
/// # Returns
/// The result of the evaluation.
///
/// # Errors
/// * [`CompilationFailed`](EvaluationError::CompilationFailed) if the function
///   could not be compiled.
/// * [`BadArity`](EvaluationError::BadArity) if the number of arguments
///   provided disagrees with the number of formal parameters in the function
///   signature.
/// * [`UnrecognizedExternal`](EvaluationError::UnrecognizedExternal) if an
///   external variable is not recognized.
///
/// # Examples
/// Roll one standard six-sided die:
///
/// ```rust
/// use xdy::evaluate;
/// use rand::thread_rng;
///
/// let result = evaluate("1D6", vec![], vec![], &mut thread_rng()).unwrap();
/// assert!(1 <= result.result && result.result <= 6);
/// assert!(result.records.len() == 1);
/// assert!(result.records[0].results.len() == 1);
/// assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
/// ```
///
/// Roll a variable number of dice, using an argument named `x` to supply the
/// number of dice to roll:
///
/// ```rust
/// use xdy::evaluate;
/// use rand::thread_rng;
///
/// let result =
///     evaluate("x: {x}D6", vec![3], vec![], &mut thread_rng()).unwrap();
/// assert!(3 <= result.result && result.result <= 18);
/// assert!(result.records.len() == 1);
/// assert!(result.records[0].results.len() == 3);
/// assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
/// assert!(1 <= result.records[0].results[1] && result.records[0].results[1] <= 6);
/// assert!(1 <= result.records[0].results[2] && result.records[0].results[2] <= 6);
/// ```
///
/// Roll a variable number of dice, using an external variable named `x` to
/// supply the number of dice to roll:
///
/// ```rust
/// use xdy::evaluate;
/// use rand::thread_rng;
///
/// let result =
///     evaluate("{x}D6", vec![], vec![("x", 3)], &mut thread_rng()).unwrap();
/// assert!(3 <= result.result && result.result <= 18);
/// assert!(result.records.len() == 1);
/// assert!(result.records[0].results.len() == 3);
/// assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
/// assert!(1 <= result.records[0].results[1] && result.records[0].results[1] <= 6);
/// assert!(1 <= result.records[0].results[2] && result.records[0].results[2] <= 6);
/// ```
///
/// Evaluate an arithmetic expression involving different types of dice:
///
/// ```rust
/// use xdy::evaluate;
/// use rand::thread_rng;
///
/// let result = evaluate("1D6 + 2D8 - 1D10", vec![], vec![], &mut thread_rng()).unwrap();
/// assert!(-7 <= result.result && result.result <= 21);
/// assert!(result.records.len() == 3);
/// assert!(result.records[0].results.len() == 1);
/// assert!(result.records[1].results.len() == 2);
/// assert!(result.records[2].results.len() == 1);
/// assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
/// assert!(1 <= result.records[1].results[0] && result.records[1].results[0] <= 8);
/// assert!(1 <= result.records[1].results[1] && result.records[1].results[1] <= 8);
/// assert!(1 <= result.records[2].results[0] && result.records[2].results[0] <= 10);
/// ```
pub fn evaluate<'s, 'e, 'r, R>(
	source: &'s str,
	args: impl IntoIterator<Item = i32>,
	environment: impl IntoIterator<Item = (&'e str, i32)>,
	rng: &'r mut R
) -> Result<Evaluation, EvaluationError<'s>>
where
	R: rand::Rng + ?Sized,
	'e: 's
{
	let function = compile(source)?;
	let mut evaluator = Evaluator::new(function);
	for (name, value) in environment
	{
		evaluator.bind(name, value)?;
	}
	evaluator.evaluate(args, rng)
}
