//! # Bounds tests
//!
//! Herein are the tests for the interval arithmetic that underlies the bounds
//! evaluator. Unlike the evaluator tests, these do not read a corpus file;
//! they brute-force the truth directly, by enumerating every member of each
//! operand interval and comparing the resulting extrema against the interval
//! operation under test.
//!
//! Two complementary strategies appear throughout:
//!
//! * An **exhaustive grid** over a small range of endpoints, which establishes
//!   a property over the whole small-case space rather than sampling it. Where
//!   an operation is exact, the grid asserts equality; asserting mere
//!   containment would pass for a trivially widened implementation and would
//!   not test the operation at all.
//! * A **boundary set** of `i32` extrema enumerated across all four interval
//!   endpoints, which reaches the saturating arithmetic that the grid cannot.
//!   Truth cannot be brute-forced over such intervals, so these assert
//!   soundness against sampled members, plus the structural invariant that no
//!   operation may answer an inverted interval.

use crate::{EvaluationBounds, exp};

////////////////////////////////////////////////////////////////////////////////
//                                  Support.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The `i32` values at and adjacent to the boundaries of the type, together
/// with the small magnitudes that the arithmetic primitives special-case.
/// Enumerated across all four interval endpoints, these reach the saturating
/// paths that an exhaustive grid over a small range cannot.
const BOUNDARIES: [i32; 9] = [
	i32::MIN,
	i32::MIN + 1,
	-2,
	-1,
	0,
	1,
	2,
	i32::MAX - 1,
	i32::MAX
];

/// Brute-force the true extrema of a binary operation over the Cartesian
/// product of two intervals.
///
/// # Parameters
/// - `lhs`: The left operand interval.
/// - `rhs`: The right operand interval.
/// - `op`: The scalar operation to apply to each pair of members.
///
/// # Returns
/// The tightest interval containing every result.
fn brute_force(
	lhs: EvaluationBounds,
	rhs: EvaluationBounds,
	op: impl Fn(i32, i32) -> i32
) -> EvaluationBounds
{
	let mut min = i32::MAX;
	let mut max = i32::MIN;
	for x in lhs.min..=lhs.max
	{
		for y in rhs.min..=rhs.max
		{
			let value = op(x, y);
			min = min.min(value);
			max = max.max(value);
		}
	}
	(min, max).into()
}

/// Answer the members of [`BOUNDARIES`] that lie within the specified interval,
/// together with its own endpoints. Used to sample intervals too wide to
/// enumerate.
///
/// # Parameters
/// - `bounds`: The interval to sample.
///
/// # Returns
/// Representative members of the interval.
fn samples(bounds: EvaluationBounds) -> Vec<i32>
{
	let mut values = vec![bounds.min, bounds.max];
	values.extend(BOUNDARIES.iter().copied().filter(|&x| bounds.contains(x)));
	values
}

/// Enumerate every interval whose endpoints are drawn from [`BOUNDARIES`].
///
/// # Returns
/// The intervals, in ascending order of minimum then maximum.
fn boundary_intervals() -> Vec<EvaluationBounds>
{
	let mut intervals = Vec::new();
	for (i, &min) in BOUNDARIES.iter().enumerate()
	{
		for &max in BOUNDARIES.iter().skip(i)
		{
			intervals.push((min, max).into());
		}
	}
	intervals
}

////////////////////////////////////////////////////////////////////////////////
//                           Exponentiation bounds.                           //
////////////////////////////////////////////////////////////////////////////////

/// The inclusive range of base endpoints covered by the exhaustive
/// exponentiation grid.
const EXP_BASES: std::ops::RangeInclusive<i32> = -20..=20;

/// The inclusive range of exponent endpoints covered by the exhaustive
/// exponentiation grid. Negative exponents are legal in the expression
/// language — [`exp`] answers zero for them, except for bases `0` and `±1` —
/// so the range must straddle zero to exercise that arm.
const EXP_POWERS: std::ops::RangeInclusive<i32> = -10..=10;

/// Test that [`EvaluationBounds::exp`] is *exact* over an exhaustive grid of
/// small base and exponent intervals.
///
/// [`EvaluationBounds::exp`] is not a structural interval operation like its
/// siblings: it samples a handful of interesting bases and exponents and unions
/// the results. Its correctness therefore does not follow from the same
/// argument as [`Mul`](std::ops::Mul) or [`Div`](std::ops::Div), and has to be
/// established directly.
///
/// Exactness is asserted rather than containment. Containment would pass for an
/// implementation that widened every answer to the whole of `i32` and so would
/// not test the sampling heuristic at all.
#[test]
fn test_exp_bounds_exhaustive()
{
	let mut checked = 0usize;
	for base_min in EXP_BASES
	{
		for base_max in base_min..=*EXP_BASES.end()
		{
			let base: EvaluationBounds = (base_min, base_max).into();
			for power_min in EXP_POWERS
			{
				for power_max in power_min..=*EXP_POWERS.end()
				{
					let power: EvaluationBounds = (power_min, power_max).into();
					let expected = brute_force(base, power, exp);
					let actual = base.exp(power);
					assert_eq!(
						actual, expected,
						"[{}] ^ [{}]: expected [{}], got [{}]",
						base, power, expected, actual
					);
					checked += 1;
				}
			}
		}
	}
	// Guard against the loop bounds silently collapsing.
	assert_eq!(checked, 198891);
}

/// Test that [`EvaluationBounds::exp`] is total and sound at the `i32`
/// boundaries.
///
/// The exhaustive grid cannot reach the saturating arithmetic, and a
/// debug-mode overflow panic reachable through a public entry point would be a
/// defect. Truth cannot be brute-forced over intervals this wide, so soundness
/// is asserted against sampled members instead.
#[test]
fn test_exp_bounds_at_boundaries()
{
	for base in boundary_intervals()
	{
		for power in boundary_intervals()
		{
			let actual = base.exp(power);
			assert!(
				actual.min <= actual.max,
				"[{}] ^ [{}]: inverted interval [{}]",
				base,
				power,
				actual
			);
			for x in samples(base)
			{
				for y in samples(power)
				{
					let value = exp(x, y);
					assert!(
						actual.contains(value),
						"[{}] ^ [{}]: {} ^ {} = {} ∉ [{}]",
						base,
						power,
						x,
						y,
						value,
						actual
					);
				}
			}
		}
	}
}
