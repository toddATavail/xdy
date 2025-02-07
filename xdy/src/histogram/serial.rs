//! # Serial implementation of the histogram builder
//!
//! This module contains the serial implementation of the histogram builder,
//! which is always available.

use std::{
	cell::RefCell,
	collections::{HashMap, VecDeque}
};

use crate::{EvaluationError, Evaluator, Function};

use super::{CanBuildHistogram, CanIterate, EvaluationState, Histogram};

////////////////////////////////////////////////////////////////////////////////
//                            Histogram building.                             //
////////////////////////////////////////////////////////////////////////////////

/// Builds a histogram of the outcomes of a dice expression.
#[derive(Debug, Clone)]
pub struct HistogramBuilder
{
	/// The function to evaluate.
	function: Function,

	/// The environment in which to evaluate the function, as a map from
	/// external variable indices to values. Missing bindings default to zero.
	environment: HashMap<usize, i32>
}

/// Increment the histogram for the given outcome.
///
/// # Parameters
/// - `outcome`: The outcome to increment.
fn increment(histogram: &mut Histogram, outcome: i32)
{
	histogram
		.entry(outcome)
		.and_modify(|count| *count += 1)
		.or_insert(1);
}

impl<'inst> super::HistogramBuilder<'inst, EvaluationStateIterator<'inst>>
	for HistogramBuilder
{
	#[inline]
	fn new(evaluator: Evaluator) -> Self
	{
		HistogramBuilder {
			function: evaluator.function,
			environment: evaluator.environment
		}
	}

	fn build_while(
		&self,
		args: impl IntoIterator<Item = i32>,
		condition: impl Fn(&i32) -> bool
	) -> Result<Histogram, EvaluationError<'_>>
	{
		let mut histogram = Histogram::default();

		super::HistogramBuilder::iter(self, args)?
			.flat_map(|state| state.result)
			.take_while(condition)
			.for_each(|outcome| increment(&mut histogram, outcome));
		Ok(histogram)
	}

	#[inline]
	fn iter(
		&'inst self,
		args: impl IntoIterator<Item = i32>
	) -> Result<EvaluationStateIterator<'inst>, EvaluationError<'inst>>
	{
		CanBuildHistogram::iter(self, args)
	}
}

impl<'inst> CanBuildHistogram<'inst, EvaluationStateIterator<'inst>>
	for HistogramBuilder
{
	#[inline]
	fn function(&self) -> &Function { &self.function }

	#[inline]
	fn environment(&self) -> &HashMap<usize, i32> { &self.environment }

	fn create_iterator(
		&'inst self,
		initial_state: EvaluationState<'inst>
	) -> EvaluationStateIterator<'inst>
	{
		EvaluationStateIterator {
			states: RefCell::new([initial_state].into())
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Iteration.                                 //
////////////////////////////////////////////////////////////////////////////////

/// An open-ended iterator of [states](EvaluationState), representing the
/// continuations of a histogram builder's execution. Each state represents a
/// point in the builder's execution where it may be suspended and later
/// resumed. Each state corresponds to the internal state of a range or roll
/// instruction, as these are the only instructions that may be suspended. Each
/// state represents having rolled a particular sequence of ranges or dice, and
/// the full iterator represents the entire space of possible outcomes of a
/// dice expression. The consumer may choose to terminate the iteration early,
/// in which case the builder will contain the partial histogram computed up to
/// the point of termination.
#[derive(Debug)]
pub struct EvaluationStateIterator<'inst>
{
	/// The remaining states to evaluate.
	states: RefCell<VecDeque<EvaluationState<'inst>>>
}

impl<'inst> CanIterate<'inst> for EvaluationStateIterator<'inst>
{
	fn next_state(&mut self) -> Option<EvaluationState<'inst>>
	{
		self.states.borrow_mut().pop_front()
	}

	fn inject_successors(&mut self, state: &mut EvaluationState<'inst>)
	{
		if let Some(ref mut successors) = state.successors
		{
			self.states.borrow_mut().extend(successors.drain(..))
		}
	}
}

impl<'inst> Iterator for EvaluationStateIterator<'inst>
{
	type Item = EvaluationState<'inst>;

	fn next(&mut self) -> Option<Self::Item>
	{
		match self.next_state()
		{
			Some(mut state) =>
			{
				state.evaluate();
				self.inject_successors(&mut state);
				Some(state)
			},
			None => None
		}
	}
}
