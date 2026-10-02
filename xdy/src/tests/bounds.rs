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

use crate::{
	Assembler, EvaluationBounds, EvaluationError, Evaluator, Passes, exp,
	r#mod,
	support::{compile_valid, optimize}
};

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

////////////////////////////////////////////////////////////////////////////////
//                             Remainder bounds.                              //
////////////////////////////////////////////////////////////////////////////////

/// The inclusive range of endpoints covered by the exhaustive remainder grid.
/// Every divisor interval drawn from this range spans at most fifteen
/// magnitudes, and so lies within the enumeration budget of
/// [`Rem`](std::ops::Rem); the grid therefore exercises only the exact path.
/// [`test_rem_bounds_wide_divisors`] covers the approximating path.
const REM_ENDPOINTS: std::ops::RangeInclusive<i32> = -14..=14;

/// Divisor intervals spanning more magnitudes than [`Rem`](std::ops::Rem) will
/// enumerate, which therefore reach the approximating path that
/// [`test_rem_bounds_exhaustive`] cannot. Each is narrow enough to brute-force
/// against a dividend drawn from [`REM_ENDPOINTS`], and together they cover
/// every shape that path distinguishes: divisors straddling zero, wholly
/// positive, and wholly negative; divisors that dominate every dividend and
/// divisors that do not; and both `i32` extremes.
const WIDE_DIVISORS: [(i32, i32); 12] = [
	(-129, 129),
	(-200, 60),
	(-60, 200),
	(0, 140),
	(-140, 0),
	(1, 140),
	(-140, -1),
	(14, 153),
	(-153, -14),
	(200, 340),
	(i32::MIN, i32::MIN + 140),
	(i32::MAX - 140, i32::MAX)
];

/// Test that [`Rem`](std::ops::Rem) is *exact* over an exhaustive grid of small
/// dividend and divisor intervals.
///
/// Exactness is asserted rather than containment, on the same reasoning as
/// [`test_exp_bounds_exhaustive`]: containment would pass for an implementation
/// that widened every answer to the whole of `i32`. Every divisor interval here
/// falls within the enumeration budget, where the operation is exact by
/// construction — it hulls a union of per-magnitude hulls, each of which is
/// itself exact.
#[test]
fn test_rem_bounds_exhaustive()
{
	let mut checked = 0usize;
	for min in REM_ENDPOINTS
	{
		for max in min..=*REM_ENDPOINTS.end()
		{
			let dividend: EvaluationBounds = (min, max).into();
			for divisor_min in REM_ENDPOINTS
			{
				for divisor_max in divisor_min..=*REM_ENDPOINTS.end()
				{
					let divisor: EvaluationBounds =
						(divisor_min, divisor_max).into();
					let expected = brute_force(dividend, divisor, r#mod);
					let actual = dividend % divisor;
					assert_eq!(
						actual, expected,
						"[{}] % [{}]: expected [{}], got [{}]",
						dividend, divisor, expected, actual
					);
					checked += 1;
				}
			}
		}
	}
	// Guard against the loop bounds silently collapsing.
	assert_eq!(checked, 189225);
}

/// Test that [`Rem`](std::ops::Rem) is sound for divisor intervals too wide to
/// enumerate.
///
/// Beyond the enumeration budget the operation approximates, so soundness is
/// asserted rather than exactness. This is the only test that reaches that
/// path: the exhaustive grid stays within the budget, and the boundary set is
/// too wide to brute-force.
#[test]
fn test_rem_bounds_wide_divisors()
{
	for (divisor_min, divisor_max) in WIDE_DIVISORS
	{
		// Guard against an entry drifting back within the enumeration budget,
		// which would silently stop testing the approximating path.
		let low = match (divisor_min, divisor_max)
		{
			(min, max) if min <= 0 && max >= 0 => 0,
			(min, max) => (min as i64).abs().min((max as i64).abs())
		};
		let high = (divisor_min as i64).abs().max((divisor_max as i64).abs());
		assert!(
			high - low >= 128,
			"[{}, {}] spans only {} magnitudes",
			divisor_min,
			divisor_max,
			high - low + 1
		);
		let divisor: EvaluationBounds = (divisor_min, divisor_max).into();
		for min in REM_ENDPOINTS
		{
			for max in min..=*REM_ENDPOINTS.end()
			{
				let dividend: EvaluationBounds = (min, max).into();
				let expected = brute_force(dividend, divisor, r#mod);
				let actual = dividend % divisor;
				assert!(
					actual.min <= actual.max,
					"[{}] % [{}]: inverted interval [{}]",
					dividend,
					divisor,
					actual
				);
				assert!(
					actual.min <= expected.min && actual.max >= expected.max,
					"[{}] % [{}]: [{}] ⊉ [{}]",
					dividend,
					divisor,
					actual,
					expected
				);
			}
		}
	}
}

/// Test that [`Rem`](std::ops::Rem) is total and sound at the `i32` boundaries.
///
/// As with [`test_exp_bounds_at_boundaries`], the exhaustive grid cannot reach
/// the widened arithmetic, truth cannot be brute-forced over intervals this
/// wide, and a debug-mode overflow panic reachable through a public entry point
/// would be a defect.
#[test]
fn test_rem_bounds_at_boundaries()
{
	for dividend in boundary_intervals()
	{
		for divisor in boundary_intervals()
		{
			let actual = dividend % divisor;
			assert!(
				actual.min <= actual.max,
				"[{}] % [{}]: inverted interval [{}]",
				dividend,
				divisor,
				actual
			);
			for x in samples(dividend)
			{
				for y in samples(divisor)
				{
					let value = r#mod(x, y);
					assert!(
						actual.contains(value),
						"[{}] % [{}]: {} % {} = {} ∉ [{}]",
						dividend,
						divisor,
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

/// Test the two defect shapes that motivated the repair of
/// [`Rem`](std::ops::Rem), pinned as named cases so that a regression reports
/// the original symptom rather than an anonymous grid coordinate.
///
/// Both arose from bounding the remainder by the divisor endpoint *nearest*
/// zero, when `|x % y|` is bounded by the magnitude of the endpoint *farthest*
/// from zero.
#[test]
fn test_rem_bounds_regressions()
{
	// Under-approximation: the divisor endpoint nearest zero is -1, whose
	// remainders are all zero, but -4 admits remainders as large as 3.
	let dividend: EvaluationBounds = (1, 6).into();
	let divisor: EvaluationBounds = (-4, -1).into();
	assert_eq!(dividend % divisor, (0, 3).into());
	// Inversion: the answer was [1, 0], which contains nothing at all.
	let dividend: EvaluationBounds = i32::MIN.into();
	let divisor: EvaluationBounds = (0, i32::MAX).into();
	let actual = dividend % divisor;
	assert!(actual.min <= actual.max, "inverted interval [{}]", actual);
	assert!(actual.contains(r#mod(i32::MIN, i32::MAX)));
}

/// Test that the under-approximation is unreachable through the public bounds
/// evaluator, which is where it would actually harm a caller.
///
/// `1D6 % (1D4 - 5)` has a divisor of `[-4, -1]`, and reported `[0, 0]` while
/// the expression can plainly produce 3, as `5 % -4`.
#[test]
fn test_rem_bounds_end_to_end()
{
	let function = compile_valid("1D6 % (1D4 - 5)");
	let evaluator = Evaluator::new(function);
	let bounds = evaluator.bounds_over([], []).unwrap().value;
	assert!(bounds.contains(3), "1D6 % (1D4 - 5): 3 ∉ [{}]", bounds);
}

/// Test that the strength reducer's rewrite of `x ^ 2` as `x * x` keeps the
/// bounds of the square.
///
/// `[-3:3] ^ 2` squares a range that straddles zero, so its minimum is 0.
/// Multiplying the intervals of the two operands as though they were
/// independent answered a minimum of -9 after optimization.
#[test]
fn test_square_bounds_survive_optimization()
{
	let source = "[-3:3] ^ 2";
	let unoptimized = compile_valid(source);
	let optimized = optimize(unoptimized.clone(), Passes::all());
	for function in [unoptimized, optimized]
	{
		let evaluator = Evaluator::new(function.clone());
		let bounds = evaluator.bounds_over([], []).unwrap().value;
		assert_eq!(
			bounds,
			(0, 9).into(),
			"{}: expected [0, 9], got [{}]\n{}",
			source,
			bounds,
			function
		);
	}
}

/// Test that a rolling record summed without being rolled, which the compiler
/// never emits but the assembler admits, is bounded as the evaluator sums it:
/// empty, so zero, whatever is dropped from it.
///
/// The bounds evaluator once panicked on such a record.
#[test]
fn test_unrolled_record_bounds()
{
	let function = Assembler::assemble(
		"Function() r#2 ⚅#2
	extern[]
	body:
		⚅0 <- roll standard dice 1D3
		@0 <- sum rolling record ⚅0
		⚅1 <- drop lowest @0 from ⚅1
		@1 <- sum rolling record ⚅1
		@0 <- @0 + @1
		return @0"
	)
	.unwrap();
	assert_eq!(function.validate(), Ok(()));
	let bounds = Evaluator::new(function).bounds_over([], []).unwrap();
	assert_eq!(bounds.value, (1, 3).into());
}

////////////////////////////////////////////////////////////////////////////////
//                         Interval-valued bindings.                          //
////////////////////////////////////////////////////////////////////////////////

/// Brute-force the truth of a single-parameter function over an interval-valued
/// argument, by unioning the bounds answered at every member of the interval.
///
/// Each member is itself a bounds query, so this is not ground truth about the
/// dice — the roll bounds are still computed rather than rolled — but it is
/// ground truth about the *interval* binding, which is what
/// [`Evaluator::bounds_over`] adds. An interval binding that fails to contain
/// this union has lost a value that the same analysis finds at a fixed binding.
///
/// # Parameters
/// - `source`: The source of a function of exactly one formal parameter.
/// - `binding`: The interval to enumerate.
///
/// # Returns
/// The union of the bounds over every member of the interval.
fn union_over_members(
	source: &str,
	binding: EvaluationBounds
) -> EvaluationBounds
{
	let evaluator = Evaluator::new(compile_valid(source));
	let mut min = i32::MAX;
	let mut max = i32::MIN;
	for x in binding.min..=binding.max
	{
		let bounds = evaluator.bounds_over([Some(x.into())], []).unwrap().value;
		min = min.min(bounds.min);
		max = max.max(bounds.max);
	}
	(min, max).into()
}

/// Assert that a single-parameter function answers the expected bounds over an
/// interval-valued argument, and that those bounds are sound with respect to
/// [`union_over_members`].
///
/// # Parameters
/// - `source`: The source of a function of exactly one formal parameter.
/// - `binding`: The interval to bind to the parameter.
/// - `expected`: The expected value bounds.
fn assert_interval_binding(
	source: &str,
	binding: EvaluationBounds,
	expected: EvaluationBounds
)
{
	let evaluator = Evaluator::new(compile_valid(source));
	let actual = evaluator.bounds_over([Some(binding)], []).unwrap();
	assert_eq!(
		actual.value, expected,
		"{} over [{}]: expected [{}], got [{}]",
		source, binding, expected, actual.value
	);
	assert_eq!(
		actual.count, None,
		"{} over [{}]: outcome count survived a non-degenerate binding",
		source, binding
	);
	let truth = union_over_members(source, binding);
	assert!(
		actual.value.min <= truth.min && truth.max <= actual.value.max,
		"{} over [{}]: [{}] ⊉ [{}]",
		source,
		binding,
		actual.value,
		truth
	);
}

/// Test that an interval-valued argument bounds a dynamic die count, which is
/// the motivating case for [`Evaluator::bounds_over`].
#[test]
fn test_bounds_over_interval_argument()
{
	assert_interval_binding("{x}: {x}D6", (1, 20).into(), (1, 120).into());
}

/// Test that an unsupplied binding is bounded by the whole of `i32` rather than
/// by zero.
///
/// This is the defect that motivated deprecating
/// [`bounds`](Evaluator::bounds), which answers `[1, 6]` here — a confidently
/// wrong bound, since `{x}` is unknown and the expression can produce very
/// nearly any `i32` at all.
///
/// The minimum is `i32::MIN + 1` rather than `i32::MIN`, because the die
/// contributes at least one and the addition saturates rather than clamps. The
/// bound is therefore not merely wide but correct at the edge.
#[test]
fn test_bounds_over_unsupplied_external()
{
	let evaluator = Evaluator::new(compile_valid("1D6 + {x}"));
	let bounds = evaluator.bounds_over([], []).unwrap();
	assert_eq!(bounds.value, (i32::MIN + 1, i32::MAX).into());
	assert_eq!(bounds.count, None);
	// The bound that the zero-default convention would have answered.
	assert!(bounds.value.contains(1) && bounds.value.contains(6));
	// And the values that convention would have excluded.
	assert!(bounds.value.contains(i32::MIN + 1));
	assert!(bounds.value.contains(i32::MAX));
}

/// Test that an unsupplied formal parameter is likewise bounded by the whole of
/// `i32`, and that supplying it in the same position constrains it again.
#[test]
fn test_bounds_over_unsupplied_argument()
{
	let evaluator = Evaluator::new(compile_valid("{x}: 1D6 + {x}"));
	let bounds = evaluator.bounds_over([None], []).unwrap();
	assert_eq!(bounds.value, (i32::MIN + 1, i32::MAX).into());
	let bounds = evaluator.bounds_over([Some(10.into())], []).unwrap();
	assert_eq!(bounds.value, (11, 16).into());
}

/// Test that [`Evaluator::bounds_over`] ignores the environment.
///
/// A binding established for the benefit of [`Evaluator::evaluate`] is a
/// roll-time convention. Inheriting it here would make the same static query
/// answer differently depending on which [`Evaluator::bind`] calls happened to
/// precede it.
#[test]
fn test_bounds_over_ignores_environment()
{
	let mut evaluator = Evaluator::new(compile_valid("1D6 + {x}"));
	evaluator.bind("x", 3).unwrap();
	let bounds = evaluator.bounds_over([], []).unwrap();
	assert_eq!(bounds.value, (i32::MIN + 1, i32::MAX).into());
	// The remedy: supply the external explicitly.
	let bounds = evaluator.bounds_over([], [("x", 3.into())]).unwrap();
	assert_eq!(bounds.value, (4, 9).into());
	assert_eq!(bounds.count, Some(6));
}

/// Test that the outcome count survives exactly the degenerate bindings.
///
/// The count is an exact count of the outcomes of one binding of the function.
/// A non-degenerate binding picks out many such functions, so an exact-looking
/// number would be wrong for all but one of them.
#[test]
fn test_bounds_over_count_requires_degenerate_bindings()
{
	let evaluator = Evaluator::new(compile_valid("{x}: 1D6 + {x}"));
	assert_eq!(
		evaluator.bounds_over([Some(2.into())], []).unwrap().count,
		Some(6)
	);
	assert_eq!(
		evaluator
			.bounds_over([Some((2, 3).into())], [])
			.unwrap()
			.count,
		None
	);
	assert_eq!(evaluator.bounds_over([None], []).unwrap().count, None);
	// An unsupplied external is non-degenerate too, even though nothing was
	// passed in the argument channel.
	let evaluator = Evaluator::new(compile_valid("1D6 + {x}"));
	assert_eq!(evaluator.bounds_over([], []).unwrap().count, None);
	assert_eq!(
		evaluator.bounds_over([], [("x", 3.into())]).unwrap().count,
		Some(6)
	);
}

/// Test that the argument list is still checked against the arity, and that an
/// undeclared external is still rejected.
#[test]
fn test_bounds_over_rejects_bad_bindings()
{
	let evaluator = Evaluator::new(compile_valid("{x}: {x}D6"));
	assert_eq!(
		evaluator.bounds_over([], []),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 0
		})
	);
	assert_eq!(
		evaluator.bounds_over([None, None], []),
		Err(EvaluationError::BadArity {
			expected: 1,
			given: 2
		})
	);
	assert_eq!(
		evaluator.bounds_over([None], [("y", 1.into())]),
		Err(EvaluationError::UnrecognizedExternal("y"))
	);
}

/// Test that an interval die count spanning negative values folds to zero dice
/// rather than to negative dice.
///
/// The clamp in `visit_sum_rolling_record` already handled this for dynamic
/// counts; an interval binding is simply a second way to reach it.
#[test]
fn test_bounds_over_negative_interval_count()
{
	assert_interval_binding("{x}: {x}D6", (-3, 5).into(), (0, 30).into());
	// The dynamic form of the same shape, which has no binding to vary.
	let evaluator = Evaluator::new(compile_valid("(1D[-5, -4, 0, 4, 5])D6"));
	let bounds = evaluator.bounds_over([], []).unwrap().value;
	assert_eq!(bounds, (0, 30).into());
}

/// Test that an interval face count spanning zero and negative values yields a
/// sound face bound.
///
/// Each standard die spans the faces `[1, faces]`, which is empty for a
/// non-positive face count, so the minimum folds to zero rather than going
/// negative.
#[test]
fn test_bounds_over_interval_faces()
{
	assert_interval_binding("{x}: 1D{x}", (-4, 6).into(), (0, 6).into());
}

/// Test that an interval drop count cannot keep more dice than were rolled, nor
/// fewer than none.
#[test]
fn test_bounds_over_interval_drop_count()
{
	assert_interval_binding(
		"{x}: 5D6 drop lowest {x}",
		(0, 10).into(),
		(0, 30).into()
	);
	assert_interval_binding(
		"{x}: 5D6 drop highest {x}",
		(-2, 3).into(),
		(2, 30).into()
	);
}
