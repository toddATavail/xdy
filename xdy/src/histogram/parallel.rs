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
		Condvar, Mutex, MutexGuard, Once, PoisonError,
		atomic::{AtomicBool, AtomicU64, Ordering}
	},
	thread::available_parallelism
};

use rayon::{
	Scope, ThreadPoolBuilder, broadcast, current_num_threads,
	iter::{
		ParallelIterator,
		plumbing::{Folder, Reducer, UnindexedConsumer}
	},
	scope
};

use super::{CanBuildHistogram, EvaluationState, Histogram, Meter};
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

	fn build_metered(
		&self,
		args: impl IntoIterator<Item = i32> + Send,
		budget: u64
	) -> Result<Histogram, EvaluationError<'static>>
	{
		// Establish a scope for the parallel computation, to ensure that the
		// thread local storage is available to all threads. Always finish the
		// histogram, even if the meter refuses a charge, so that no partial
		// histogram lingers in thread local storage.
		let generation = self.generation;
		let meter = Meter::new(budget);
		let histogram = scope(|_| {
			CanBuildHistogram::iter(self, args, meter.limited())?
				.flat_map(|state| state.result)
				.for_each(|outcome| increment(generation, outcome));
			Ok::<_, EvaluationError<'static>>(finish(generation))
		})?;
		meter.verdict()?;
		Ok(histogram)
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
			CanBuildHistogram::iter(self, args, None)?
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
		CanBuildHistogram::iter(self, args, None)
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
		initial_state: EvaluationState<'inst>,
		meter: Option<&'inst Meter>
	) -> EvaluationStateIterator<'inst>
	{
		HistogramBuilder::use_available_parallelism();
		EvaluationStateIterator {
			states: [initial_state].into(),
			meter
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
#[derive(Debug, Clone)]
pub struct EvaluationStateIterator<'inst>
{
	/// The states from which to begin the exploration.
	states: VecDeque<EvaluationState<'inst>>,

	/// The meter of the evaluation, if it is metered. Once the meter refuses a
	/// charge, the exploration ends.
	meter: Option<&'inst Meter>
}

impl<'inst> ParallelIterator for EvaluationStateIterator<'inst>
{
	type Item = EvaluationState<'inst>;

	fn drive_unindexed<C>(self, consumer: C) -> C::Result
	where
		C: UnindexedConsumer<Self::Item>
	{
		// We cannot use any of `rayon`'s bridging methods here, as they are
		// incapable of dealing with dynamic workloads: we are exhaustively
		// exploring a state space whose size we cannot predict in the general
		// case (where some range and roll instructions are dynamically
		// controlled by other range and roll instructions). Nor can we split
		// the work recursively with `join`, as the depth of the recursion would
		// grow with the depth of the state space. Instead, a crew of workers
		// explores the state space, each with its own consumer, sharing work
		// through a common queue. The crew starts with one worker, and recruits
		// helpers only once the state space proves large enough to share.
		//
		// Neither reducers nor folders are `Send`, so we split the consumer and
		// reduce the results here, on the calling thread, and only send the
		// consumers to the workers. Each split peels a consumer off the left,
		// so the reducers nest to the right.
		let workers = current_num_threads().max(1);
		let mut consumers = Vec::with_capacity(workers);
		let mut reducers = Vec::with_capacity(workers - 1);
		for _ in 1..workers
		{
			reducers.push(consumer.to_reducer());
			consumers.push(consumer.split_off_left());
		}
		consumers.push(consumer);
		let mut consumers = consumers.into_iter().enumerate();
		let (_, first) = consumers.next().unwrap();
		let crew = Crew::new(self.states, consumers.collect(), self.meter);
		scope(|scope| crew.work(scope, 0, first));
		// Every worker has finished, so every recruit's result is present. Any
		// consumers still in reserve never worked, so finish them here.
		let mut results = crew
			.results
			.into_iter()
			.map(|result| result.into_inner().unwrap())
			.collect::<Vec<_>>();
		for (index, consumer) in crew.reserve.into_inner().unwrap()
		{
			results[index] = Some(consumer.into_folder().complete());
		}
		// Reduce the results from the right, mirroring the order of the
		// splits.
		let mut results = results.into_iter().map(Option::unwrap);
		let mut result = results.next_back().unwrap();
		for (reducer, left) in reducers.into_iter().rev().zip(results.rev())
		{
			result = reducer.reduce(left, result);
		}
		result
	}
}

/// The crew of workers that explores a state space on behalf of an
/// [`EvaluationStateIterator`]. Each worker explores its own states depth
/// first, on a stack that lives on the heap, so neither the worker's call
/// stack nor its memory grows with the breadth of the state space. The crew
/// begins with a single worker, which keeps the exploration of a small state
/// space on a single thread. Whenever a worker has evaluated
/// [enough](Self::RECRUITMENT_INTERVAL) states since it last recruited, it
/// recruits a helper with a consumer held in reserve, and gives the helper the
/// shallowest half of its stack, whose states root the largest subspaces. Once
/// the reserve is spent, a busy worker shares its stack whenever some worker is
/// idle. The exploration ends when no worker is busy and the queue is empty,
/// when every working consumer is full, or when the meter of a metered
/// exploration refuses a charge.
///
/// ```mermaid
/// sequenceDiagram
///     participant W as Worker
///     participant Q as Work queue
///     participant H as Helper
///     W->>W: evaluate the top state, push its successors
///     W->>Q: the interval has elapsed and a consumer is in reserve, so share
///     W->>H: spawn with the reserved consumer
///     H->>Q: acquire the front half of the queue
///     H->>H: evaluate the top state, push its successors
///     W->>Q: end the shift with an empty stack, notify
///     W->>Q: acquire: nothing queued, so wait
///     H->>Q: somebody waits, so share the bottom half of the stack, notify
///     Q-->>W: wake
///     W->>Q: acquire the front half of the queue
///     H->>Q: end the shift with an empty stack, notify
///     W->>Q: end the shift with an empty stack, notify
///     Q-->>H: nothing queued and nobody busy, so finish
///     Q-->>W: nothing queued and nobody busy, so finish
/// ```
///
/// # Type parameters
/// - `C`: The type of consumer.
#[cfg_attr(doc, aquamarine::aquamarine)]
struct Crew<'inst, C>
where
	C: UnindexedConsumer<EvaluationState<'inst>>
{
	/// The work queue.
	queue: WorkQueue<'inst>,

	/// The consumers held in reserve for helpers not yet recruited, each with
	/// the index of its result.
	reserve: Mutex<Vec<(usize, C)>>,

	/// A hint that the [reserve](Self::reserve) is not yet spent. Read without
	/// the lock, so it may be stale, which costs at most an unnecessary lock.
	recruiting: AtomicBool,

	/// The results of the workers, indexed by the order of their consumers.
	results: Vec<Mutex<Option<C::Result>>>,

	/// The meter of the exploration, if it is metered. Once the meter refuses
	/// a charge, every worker stops.
	meter: Option<&'inst Meter>
}

/// The work shared by a [`Crew`] of workers: the states that no worker has
/// claimed, and the means to wait for them.
struct WorkQueue<'inst>
{
	/// The shared part of the work, guarded by a lock.
	shared: Mutex<SharedWork<'inst>>,

	/// Signaled whenever states are queued or a worker's shift ends, either
	/// of which may let a waiting worker proceed.
	changed: Condvar,

	/// A hint that some worker is waiting for states. Read without the lock,
	/// so it may be stale, which costs at most an unnecessary or a delayed
	/// share.
	hungry: AtomicBool
}

/// The shared part of the work of a [`WorkQueue`].
struct SharedWork<'inst>
{
	/// The states that no worker has claimed.
	states: VecDeque<EvaluationState<'inst>>,

	/// The number of workers that are busy exploring states that they have
	/// claimed. Such a worker may yet share new states.
	busy: usize
}

/// A worker's shift on a [`WorkQueue`]: a claim on some states, which the
/// worker explores depth first. When the shift ends, even by unwinding, any
/// unexplored states go back to the queue, and the worker is no longer busy.
struct Shift<'queue, 'inst>
{
	/// The work queue.
	queue: &'queue WorkQueue<'inst>,

	/// The stack of states to explore. The shallowest states are at the
	/// bottom.
	stack: Vec<EvaluationState<'inst>>
}

impl<'inst, C> Crew<'inst, C>
where
	C: UnindexedConsumer<EvaluationState<'inst>>
{
	/// The number of states that a worker evaluates between recruitments. A
	/// state space too small to reach it stays on a single thread, and a larger
	/// one recruits helpers exponentially fast, as each helper also recruits.
	/// Recruiting by stack depth instead would starve the crew, because depth
	/// first exploration keeps the stack shallow.
	const RECRUITMENT_INTERVAL: usize = 64;

	/// Construct a crew that begins with the specified states.
	///
	/// # Parameters
	/// - `states`: The states from which to begin the exploration.
	/// - `reserve`: The consumers to hold in reserve for helpers, each with the
	///   index of its result. Index `0` belongs to the first worker.
	/// - `meter`: The meter of the exploration, if it is metered.
	///
	/// # Returns
	/// The crew.
	fn new(
		states: VecDeque<EvaluationState<'inst>>,
		reserve: Vec<(usize, C)>,
		meter: Option<&'inst Meter>
	) -> Self
	{
		let workers = reserve.len() + 1;
		Crew {
			queue: WorkQueue::new(states),
			recruiting: AtomicBool::new(!reserve.is_empty()),
			reserve: Mutex::new(reserve),
			results: (0..workers).map(|_| Mutex::new(None)).collect(),
			meter
		}
	}

	/// Answer whether the crew must stop working: whether the specified folder
	/// is full, or the meter has refused a charge.
	///
	/// # Parameters
	/// - `folder`: The folder of a worker.
	///
	/// # Returns
	/// `true` if the worker must stop, `false` otherwise.
	#[inline]
	fn must_stop(&self, folder: &C::Folder) -> bool
	{
		folder.full() || self.meter.is_some_and(Meter::refused)
	}

	/// Work as a member of the crew, feeding completed states to the specified
	/// consumer until the state space is exhausted or the consumer is full,
	/// then record the consumer's result.
	///
	/// # Parameters
	/// - `scope`: The scope in which to spawn helpers.
	/// - `index`: The index of the consumer's result.
	/// - `consumer`: The consumer.
	fn work<'scope>(
		&'scope self,
		scope: &Scope<'scope>,
		index: usize,
		consumer: C
	) where
		'inst: 'scope,
		C: 'scope
	{
		let mut folder = consumer.into_folder();
		// The number of states evaluated since this worker last recruited a
		// helper.
		let mut evaluated = 0;
		while !self.must_stop(&folder)
		{
			let Some(mut shift) = self.queue.acquire()
			else
			{
				// The state space is exhausted.
				break
			};
			while !self.must_stop(&folder)
			{
				let Some(mut state) = shift.stack.pop()
				else
				{
					break
				};
				let outcome = state.evaluate();
				if let Some(successors) = state.successors.take()
				{
					// Push the successors in reverse, so that the first is
					// explored first and the continuation of the range or roll,
					// which is always last, is explored last. This bounds the
					// stack by the depth of the state space.
					shift.stack.extend(successors.into_iter().rev());
				}
				if outcome.is_some()
				{
					// The state has an outcome, and therefore it does not have
					// any successors, so feed it to the consumer.
					folder = folder.consume(state);
				}
				evaluated += 1;
				if evaluated >= Self::RECRUITMENT_INTERVAL
					&& shift.stack.len() > 1
					&& self.recruiting.load(Ordering::Relaxed)
				{
					evaluated = 0;
					self.recruit(scope, &mut shift.stack);
				}
				else if shift.stack.len() > 1
					&& self.queue.hungry.load(Ordering::Relaxed)
				{
					self.queue.share(&mut shift.stack);
				}
			}
			// Ending the shift returns any unexplored states to the queue, for
			// the benefit of any worker whose consumer is not yet full.
		}
		*self.results[index].lock().unwrap() = Some(folder.complete());
	}

	/// Recruit a helper with a consumer from the reserve, if any remains, and
	/// share the bottom half of the specified stack with it.
	///
	/// # Parameters
	/// - `scope`: The scope in which to spawn the helper.
	/// - `stack`: The stack of the recruiting worker.
	fn recruit<'scope>(
		&'scope self,
		scope: &Scope<'scope>,
		stack: &mut Vec<EvaluationState<'inst>>
	) where
		'inst: 'scope,
		C: 'scope
	{
		let mut reserve = self.reserve.lock().unwrap();
		let recruit = reserve.pop();
		if reserve.is_empty()
		{
			self.recruiting.store(false, Ordering::Relaxed);
		}
		drop(reserve);
		if let Some((index, consumer)) = recruit
		{
			// Share before spawning, so that the helper finds work at once.
			self.queue.share(stack);
			scope.spawn(move |scope| self.work(scope, index, consumer));
		}
	}
}

impl<'inst> WorkQueue<'inst>
{
	/// Construct a work queue that begins with the specified states.
	///
	/// # Parameters
	/// - `states`: The states from which to begin the exploration.
	///
	/// # Returns
	/// The work queue.
	fn new(states: VecDeque<EvaluationState<'inst>>) -> Self
	{
		WorkQueue {
			shared: Mutex::new(SharedWork { states, busy: 0 }),
			changed: Condvar::new(),
			hungry: AtomicBool::new(false)
		}
	}

	/// Lock the shared part of the work. A worker that panics never holds the
	/// lock, so the lock is never poisoned in practice, but a poisoned lock
	/// is harmless anyway: the shared work is consistent between operations.
	///
	/// # Returns
	/// The guard of the shared part of the work.
	fn lock(&self) -> MutexGuard<'_, SharedWork<'inst>>
	{
		self.shared.lock().unwrap_or_else(PoisonError::into_inner)
	}

	/// Claim the front half of the queued states, waiting for some to be queued
	/// if necessary.
	///
	/// # Returns
	/// A new shift over the claimed states, or `None` if the state space is
	/// exhausted, i.e., no states are queued and no worker is busy.
	fn acquire(&self) -> Option<Shift<'_, 'inst>>
	{
		let mut shared = self.lock();
		loop
		{
			let queued = shared.states.len();
			if queued > 0
			{
				// The queue's front holds the shallowest states, which belong
				// at the bottom of the stack.
				let stack = shared
					.states
					.drain(..queued.div_ceil(2))
					.collect::<Vec<_>>();
				shared.busy += 1;
				return Some(Shift { queue: self, stack })
			}
			if shared.busy == 0
			{
				return None
			}
			self.hungry.store(true, Ordering::Relaxed);
			shared = self
				.changed
				.wait(shared)
				.unwrap_or_else(PoisonError::into_inner);
		}
	}

	/// Share the bottom half of the specified stack with the other workers.
	///
	/// # Parameters
	/// - `stack`: The stack of a busy worker.
	fn share(&self, stack: &mut Vec<EvaluationState<'inst>>)
	{
		let mut shared = self.lock();
		shared.states.extend(stack.drain(..stack.len() / 2));
		// The waiting workers are about to be fed. Any that remain hungry will
		// say so again before they wait.
		self.hungry.store(false, Ordering::Relaxed);
		drop(shared);
		self.changed.notify_all();
	}
}

impl Drop for Shift<'_, '_>
{
	fn drop(&mut self)
	{
		let mut shared = self.queue.lock();
		shared.states.extend(self.stack.drain(..));
		shared.busy -= 1;
		drop(shared);
		self.queue.changed.notify_all();
	}
}
