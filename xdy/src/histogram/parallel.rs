//! # Parallel implementation of the histogram builder
//!
//! This module contains the parallel implementation of the histogram builder,
//! which is always available. The parallel implementation is only available if
//! the `parallel-histogram` feature is enabled.

use std::{
	cell::RefCell,
	collections::{HashMap, VecDeque},
	num::NonZero,
	sync::{
		Once,
		atomic::{AtomicU64, Ordering}
	},
	thread::available_parallelism
};

use rayon::{
	ThreadPoolBuilder, broadcast,
	iter::{
		ParallelIterator,
		plumbing::{Folder, Reducer, UnindexedConsumer, UnindexedProducer}
	},
	join, scope
};

use super::{CanBuildHistogram, CanIterate, EvaluationState, Histogram};
use crate::{EvaluationError, Evaluator, Function};

////////////////////////////////////////////////////////////////////////////////
//                                Histograms.                                 //
////////////////////////////////////////////////////////////////////////////////

impl Histogram
{
	/// Merges another histogram into this one.
	///
	/// # Parameters
	/// - `other`: The other histogram to merge into this one.
	fn merge(&mut self, other: Self)
	{
		for (key, value) in other.into_iter()
		{
			*self.entry(key).or_insert(0) += value;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Histogram building.                             //
////////////////////////////////////////////////////////////////////////////////

/// Builds a histogram of the outcomes of a dice expression, using parallel
/// computation. Each thread involved in the computation will build a partial
/// histogram, using thread local storage to eliminate contention. When the
/// parallel computation ends, the partial histograms are merged into a final
/// histogram.
#[derive(Debug, Clone)]
pub struct HistogramBuilder
{
	/// The function to evaluate.
	function: Function,

	/// The environment in which to evaluate the function, as a map from
	/// external variable indices to values. Missing bindings default to zero.
	environment: HashMap<usize, i32>,

	/// The generation number of the histogram, which is used to identify the
	/// histogram being built by a particular iterator.
	generation: u64
}

/// The thread pool initializer for the histogram builder.
static THREAD_POOL_INITIALIZER: Once = Once::new();

impl HistogramBuilder
{
	/// Tell the global thread pool to use all available parallelism. If the
	/// global thread pool has already been initialized, this function does
	/// nothing. Calling this function is optional. Using [HistogramBuilder] to
	/// create an iterator will automatically call this function.
	pub fn use_available_parallelism()
	{
		// We don't care whether the initialization succeeds or fails, but we
		// want to ensure that we default to using all available parallelism if
		// the user hasn't already initialized the `rayon` global thread pool.
		THREAD_POOL_INITIALIZER.call_once(|| {
			let _ = ThreadPoolBuilder::new()
				.num_threads(available_parallelism().map_or(1, NonZero::get))
				.build_global();
		});
	}

	/// Tell the global thread pool to use the specified number of threads. If
	/// the global thread pool has already been initialized, this function does
	/// nothing. Calling this function is optional.
	///
	/// # Parameters
	/// - `num_threads`: The number of threads to use in the global thread pool.
	pub fn set_parallelism(&self, num_threads: usize)
	{
		// We don't care whether the initialization succeeds or fails, but we
		// want to ensure that we default to using all available parallelism if
		// the user hasn't already initialized the `rayon` global thread pool.
		THREAD_POOL_INITIALIZER.call_once(|| {
			let _ = ThreadPoolBuilder::new()
				.num_threads(num_threads)
				.build_global();
		});
	}
}

/// Increment the histogram for the given outcome.
///
/// # Parameters
/// - `generation`: The generation number of the histogram.
/// - `outcome`: The outcome to increment.
fn increment(generation: u64, outcome: i32)
{
	HISTOGRAM.with_borrow_mut(|histogram| {
		*histogram
			.entry(generation)
			.or_default()
			.entry(outcome)
			.or_insert(0) += 1;
	});
}

/// Build the final histogram by merging the partial histograms from all
/// threads involved in the computation.
///
/// # Parameters
/// - `generation`: The generation number of the histogram.
///
/// # Returns
/// The final histogram.
fn finish(generation: u64) -> Histogram
{
	// Merge the partial histogram into the global histogram. We need to
	// broadcast a merge request to all threads within the current
	// `rayon` scope to ensure that the global histogram is updated
	// with the partial histograms from all threads.
	let partials = broadcast(|_| {
		HISTOGRAM
			.with_borrow_mut(|histogram| histogram.remove(&generation))
			.unwrap_or_default()
	});
	partials
		.into_iter()
		.reduce(|mut a, b| {
			a.merge(b);
			a
		})
		.unwrap()
}

impl<'inst> super::HistogramBuilder<'inst, EvaluationStateIterator<'inst>>
	for HistogramBuilder
{
	fn new(evaluator: Evaluator) -> Self
	{
		HistogramBuilder {
			function: evaluator.function,
			environment: evaluator.environment,
			generation: GENERATION.fetch_add(1, Ordering::Relaxed)
		}
	}

	fn build_while(
		&self,
		args: impl IntoIterator<Item = i32> + Send,
		condition: impl Fn(&i32) -> bool + Send + Sync
	) -> Result<Histogram, EvaluationError<'_>>
	{
		// Establish a scope for the parallel computation, to ensure that the
		// thread local storage is available to all threads.
		let generation = self.generation;
		scope(|_| {
			CanBuildHistogram::iter(self, args)?
				.flat_map(|state| state.result)
				.take_any_while(condition)
				.for_each(|outcome| increment(generation, outcome));
			Ok(finish(generation))
		})
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
		HistogramBuilder::use_available_parallelism();
		EvaluationStateIterator {
			generation: self.generation,
			states: [initial_state].into(),
			completed: VecDeque::new()
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Iteration.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The generation number of the histogram. The generation number is incremented
/// each time a new histogram is started, and is used to identify the histogram
/// being built by a particular iterator.
static GENERATION: AtomicU64 = AtomicU64::new(0);

thread_local! {
	/// The histograms being built by this thread, keyed by generation number.
	/// Each generation number corresponds to a different histogram being built
	/// concurrently. Partial histograms are stored in a thread-local variable
	/// to avoid the need for synchronization, then merged into the global
	/// histogram when the relevant iterator is dropped.
	pub static HISTOGRAM: RefCell<HashMap<u64, Histogram>> =
		RefCell::new(Default::default());
}

/// An open-ended iterator of [states](HistogramBuilderState), representing the
/// continuations of a histogram builder's execution. Each state represents a
/// point in the builder's execution where it may be suspended and later
/// resumed. Each state corresponds to the internal state of a range or roll
/// instruction, as these are the only instructions that may be suspended. Each
/// state represents having rolled a particular sequence of ranges or dice, and
/// the full iterator represents the entire space of possible outcomes of a
/// dice expression. The consumer may choose to terminate the iteration early,
/// in which case the builder will contain the partial histogram computed up to
/// the point of termination.
#[derive(Debug, Clone)]
pub struct EvaluationStateIterator<'inst>
{
	/// The generation number of the histogram, which uniquely identifies the
	/// histogram being built by this iterator.
	generation: u64,

	/// The remaining states to evaluate.
	states: VecDeque<EvaluationState<'inst>>,

	/// The completed states that have been evaluated and are ready for
	/// consumption.
	completed: VecDeque<EvaluationState<'inst>>
}

impl<'inst> CanIterate<'inst> for EvaluationStateIterator<'inst>
{
	/// Answer the next state to evaluate.
	///
	/// # Returns
	/// The next state to evaluate, or `None` if there are no more states to
	/// evaluate.
	fn next_state(&mut self) -> Option<EvaluationState<'inst>>
	{
		self.states.pop_front()
	}

	/// Inject the successors of the given state back into the iterator.
	///
	/// # Parameters
	/// - `state`: The state whose successors should be injected.
	fn inject_successors(&mut self, state: &mut EvaluationState<'inst>)
	{
		if let Some(ref mut successors) = state.successors
		{
			self.states.extend(successors.drain(..));
		}
	}
}

impl<'inst> ParallelIterator for EvaluationStateIterator<'inst>
{
	type Item = EvaluationState<'inst>;

	fn drive_unindexed<C>(mut self, consumer: C) -> C::Result
	where
		C: UnindexedConsumer<Self::Item>
	{
		// We absolutely cannot use any of `rayon`'s bridging methods here, as
		// they are incapable of dealing with dynamic workloads. Because we are
		// exhaustively exploring a state space, we cannot reliably predict the
		// number of states that will be generated in the general case (where
		// some range and roll instructions are dynamically controlled by other
		// range and roll instructions). So we have to oscillate between
		// evaluating states to generate successors and outcomes, splitting
		// successors into new iterators and waiting to reduce their results,
		// and consuming leaf states with outcomes until the consumer is
		// satisfied or the state space is exhausted.
		loop
		{
			if consumer.full()
			{
				// The consumer has found whatever it was looking for, and is no
				// longer interested in consuming more states. We can stop here.
				return consumer.into_folder().complete()
			}
			else
			{
				// Evaluate states until enough successors are available to make
				// splitting worthwhile.
				while self.states.len() <= EvaluationState::MAX_SUCCESSORS
				{
					match self.next_state()
					{
						Some(mut state) =>
						{
							// Evaluate the state, injecting its successors back
							// into the iterator.
							let outcome = state.evaluate();
							self.inject_successors(&mut state);
							if outcome.is_some()
							{
								// The state has an outcome, and therefore it
								// does not have any successors, so
								// place it into the completed queue for
								// the consumer to consume.
								self.completed.push_back(state);
							}
						},
						None =>
						{
							// There are no more states to evaluate, so we can
							// consume the completed states and return control
							// to the caller.
							return UnindexedProducer::fold_with(
								self,
								consumer.into_folder()
							)
							.complete()
						}
					}
				}
				// Split the iterator.
				match self.split()
				{
					(left_producer, Some(right_producer)) =>
					{
						// Split the consumer. Fork jobs for the left and right
						// producer-consumer pairs, then reduce the results.
						let (reducer, left_consumer, right_consumer) = (
							consumer.to_reducer(),
							consumer.split_off_left(),
							consumer
						);
						let (left_result, right_result) = join(
							|| left_producer.drive_unindexed(left_consumer),
							|| right_producer.drive_unindexed(right_consumer)
						);
						return reducer.reduce(left_result, right_result)
					},
					(producer, None) =>
					{
						// There weren't enough states to be worth splitting, so
						// loop back around and continue evaluating states.
						self = producer;
					}
				}
			}
		}
	}
}

impl<'inst> UnindexedProducer for EvaluationStateIterator<'inst>
{
	type Item = EvaluationState<'inst>;

	fn split(mut self) -> (Self, Option<Self>)
	{
		match self.states.len()
		{
			0..=EvaluationState::MAX_SUCCESSORS =>
			{
				// There are not enough states to be worth splitting.
				(self, None)
			},
			n =>
			{
				// Split the states into two halves.
				let right = self.states.split_off(n / 2);
				let right = EvaluationStateIterator {
					generation: self.generation,
					states: right,
					completed: VecDeque::new()
				};
				(self, Some(right))
			}
		}
	}

	fn fold_with<F>(mut self, mut folder: F) -> F
	where
		F: Folder<Self::Item>
	{
		// Only completed states are consumed by the folder, so the folder can
		// never generate new states.
		let completed = self.completed.drain(..).collect::<Vec<_>>();
		for state in completed
		{
			folder = folder.consume(state);
			if folder.full()
			{
				return folder
			}
		}
		folder
	}
}
