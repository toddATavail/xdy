//! # Small-stack harness tests
//!
//! Herein are tests for the small-stack harness,
//! [`try_on_small_stack`](crate::support::try_on_small_stack), which the
//! deep-nesting tests rely upon to report a stack overflow or a hang as an
//! ordinary test failure. Each test checks that the harness classifies one
//! fate of the closure correctly.

use std::{hint::black_box, thread, time::Duration};

use crate::support::{
	SMALL_STACK_TIMEOUT, SmallStackFailure, on_small_stack, try_on_small_stack
};

////////////////////////////////////////////////////////////////////////////////
//                              Classification.                               //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a closure that returns normally passes.
#[test]
fn test_small_stack_passes()
{
	on_small_stack(|| assert_eq!(black_box(2) + 2, 4));
}

/// Ensure that a closure that panics is reported as
/// [`Panicked`](SmallStackFailure::Panicked), with the panic message in the
/// transcript.
#[test]
fn test_small_stack_reports_panic()
{
	let outcome = try_on_small_stack(SMALL_STACK_TIMEOUT, || {
		panic!("deliberate panic in the small-stack child")
	});
	match outcome
	{
		Err(SmallStackFailure::Panicked { transcript }) => assert!(
			transcript.contains("deliberate panic in the small-stack child"),
			"{}",
			transcript
		),
		other => panic!("expected Panicked, got {:?}", other)
	}
}

/// Ensure that a closure that recurses without bound is reported as
/// [`Overflowed`](SmallStackFailure::Overflowed).
#[test]
fn test_small_stack_reports_overflow()
{
	let outcome =
		try_on_small_stack(SMALL_STACK_TIMEOUT, || _ = recurse(black_box(0)));
	assert!(
		matches!(outcome, Err(SmallStackFailure::Overflowed { .. })),
		"expected Overflowed, got {:?}",
		outcome
	);
}

/// Ensure that a closure that outlives its time budget is reported as
/// [`TimedOut`](SmallStackFailure::TimedOut).
#[test]
fn test_small_stack_reports_timeout()
{
	let outcome = try_on_small_stack(Duration::from_secs(1), || {
		thread::sleep(Duration::from_secs(60))
	});
	assert!(
		matches!(outcome, Err(SmallStackFailure::TimedOut { .. })),
		"expected TimedOut, got {:?}",
		outcome
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                Stack size.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the small stack holds a 1 MiB frame, i.e., that it is not
/// smaller than intended.
#[test]
fn test_small_stack_holds_one_mebibyte()
{
	on_small_stack(|| _ = big_frame::<{ 1024 * 1024 }>());
}

/// Ensure that the small stack does not hold a 3 MiB frame, i.e., that it is
/// not larger than intended.
#[test]
fn test_small_stack_rejects_three_mebibytes()
{
	let outcome = try_on_small_stack(SMALL_STACK_TIMEOUT, || {
		_ = big_frame::<{ 3 * 1024 * 1024 }>()
	});
	assert!(
		matches!(outcome, Err(SmallStackFailure::Overflowed { .. })),
		"expected Overflowed, got {:?}",
		outcome
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                  Helpers.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Recurse until the stack overflows. The frame buffer and the arithmetic on
/// the result defeat tail-call elimination and inlining, even in an optimized
/// build.
///
/// # Parameters
/// - `depth`: The current depth.
///
/// # Returns
/// Never, in practice; the bound on `depth` exists only to placate the
/// `unconditional_recursion` lint.
#[inline(never)]
fn recurse(depth: u64) -> u64
{
	let frame = black_box([depth; 16]);
	if depth == u64::MAX
	{
		return frame[0]
	}
	frame[1].wrapping_add(recurse(depth + 1))
}

/// Occupy a frame of `N` bytes, once. The frame is never passed by value,
/// which a debug build may copy, so that the stack holds a single copy on
/// every platform: on Windows, a 1 MiB array passed through
/// [`black_box`] by value overflowed a 2 MiB stack.
///
/// # Type parameters
/// - `N`: The size of the frame, in bytes.
///
/// # Returns
/// The first byte of the frame, to keep the frame alive.
#[inline(never)]
fn big_frame<const N: usize>() -> u8
{
	let mut frame = [0u8; N];
	black_box(&mut frame);
	frame[0]
}
