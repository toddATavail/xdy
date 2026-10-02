//! # Sampling
//!
//! Functionality for estimating the distribution of the outcomes of a dice
//! expression by Monte Carlo sampling, as a fallback for a function whose
//! exact [distribution](Distribution) costs more than the caller will
//! [spend](crate::Budget). Sampling evaluates the function many times, within
//! one budget of dice, and counts its outcomes. The counts are exact, but the
//! distribution that they describe is only an estimate, so they are
//! [`Sampled`], never a bare [`Distribution`], lest a sample be mistaken for
//! an exact answer.
//!
//! The Dvoretzky–Kiefer–Wolfowitz inequality, with Massart's constant, bounds
//! the error of the whole estimate at once: after `n` samples, the cumulative
//! distribution function of the counts differs from the exact one by more than
//! `ε` at some outcome with probability at most `2e^(-2nε²)`. At confidence
//! `1 - δ`, the [error bound](Sampled::error_bound) is therefore
//! `ε = √(ln(2/δ) / 2n)`, whatever the function.

use std::{
	collections::BTreeMap,
	fmt::{self, Display, Formatter},
	num::NonZeroU64
};

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{Distribution, EvaluationError, Evaluator, Probability, Weight};

////////////////////////////////////////////////////////////////////////////////
//                                 Sampling.                                  //
////////////////////////////////////////////////////////////////////////////////

impl Evaluator
{
	/// Sample the outcomes of the function, for the given arguments and the
	/// evaluator's external variables, by [evaluating](Self::evaluate_metered)
	/// it `n` times, rolling no more than `dice_budget` dice across all of the
	/// evaluations together, where a range counts as one die.
	///
	/// Each evaluation may roll whatever the evaluations before it left of the
	/// budget, and a roll that would exceed that is refused before it draws
	/// anything from the pRNG or allocates anything. The refusal ends the
	/// sampling; there is no partial sample.
	///
	/// # Parameters
	/// - `args`: The arguments to the function.
	/// - `rng`: The pseudo-random number generator to use for range and dice
	///   rolls.
	/// - `n`: The number of samples.
	/// - `dice_budget`: The greatest number of dice that all of the evaluations
	///   together may roll.
	///
	/// # Returns
	/// The counts of the outcomes of the evaluations, which total `n`.
	///
	/// # Errors
	/// - [`BadArity`](EvaluationError::BadArity) if the number of arguments
	///   provided disagrees with the number of formal parameters in the
	///   function signature. The first evaluation refuses them before it rolls
	///   anything.
	/// - [`DiceBudgetExhausted`](EvaluationError::DiceBudgetExhausted) if the
	///   evaluations together ask for more dice than the budget allows. The
	///   refusal reports what remains of the whole budget, and the dice that
	///   every evaluation has consumed, including those that the refused one
	///   rolled before it.
	///
	/// # Examples
	/// Estimate the distribution of `3D6` from 50,000 samples, whose error is
	/// within 0.0061 at 95% confidence:
	///
	/// ```rust
	/// use std::num::NonZeroU64;
	///
	/// use rand::{SeedableRng, rngs::StdRng};
	/// use xdy::{Budget, Evaluator, Probability, Unobserved, Weight, compile};
	///
	/// let mut evaluator = Evaluator::new(compile("3D6")?);
	/// let n = NonZeroU64::new(50_000).unwrap();
	/// let mut rng = StdRng::seed_from_u64(1);
	/// let sampled = evaluator.sample([], &mut rng, n, 3 * 50_000)?;
	/// assert_eq!(sampled.counts().total(), &Weight::from(50_000u32));
	///
	/// let confidence = Probability::new(Weight::from(19u8), Weight::from(20u8))?;
	/// let epsilon = sampled.error_bound(&confidence);
	/// assert_eq!(format!("{epsilon:.4}"), "0.0061");
	///
	/// let exact = evaluator
	///     .plan_distribution([])?
	///     .build(Budget::UNLIMITED, &Unobserved)?;
	/// for outcome in 3..=18
	/// {
	///     let error = sampled.counts().cdf(outcome).to_f64()
	///         - exact.cdf(outcome).to_f64();
	///     assert!(error.abs() <= epsilon);
	/// }
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	///
	/// Sampling that would exceed its budget is refused, even before its first
	/// roll:
	///
	/// ```rust
	/// use std::num::NonZeroU64;
	///
	/// use rand::rng;
	/// use xdy::{EvaluationError, Evaluator, compile};
	///
	/// let mut evaluator = Evaluator::new(compile("{n}: {n}D6")?);
	/// let n = NonZeroU64::new(50_000).unwrap();
	/// assert_eq!(
	///     evaluator.sample([i32::MAX], &mut rng(), n, 1_000_000),
	///     Err(EvaluationError::DiceBudgetExhausted {
	///         requested: i32::MAX as u64,
	///         remaining: 1_000_000,
	///         consumed: 0
	///     })
	/// );
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn sample<R>(
		&mut self,
		args: impl IntoIterator<Item = i32>,
		rng: &mut R,
		n: NonZeroU64,
		dice_budget: u64
	) -> Result<Sampled, EvaluationError<'static>>
	where
		R: Rng + ?Sized
	{
		let args = args.into_iter().collect::<Vec<_>>();
		let mut remaining = dice_budget;
		let mut consumed = 0u64;
		let mut counts = BTreeMap::<i32, u64>::new();
		for _ in 0..n.get()
		{
			let evaluation = self
				.evaluate_metered(args.iter().copied(), rng, remaining)
				.map_err(|e| match e
				{
					// The refused evaluation reports only its own dice, and
					// what remains of the budget that it was given, which is
					// what remains of the whole budget.
					EvaluationError::DiceBudgetExhausted {
						requested,
						remaining,
						consumed: within
					} => EvaluationError::DiceBudgetExhausted {
						requested,
						remaining,
						consumed: consumed + within
					},
					e => e
				})?;
			remaining -= evaluation.dice;
			consumed += evaluation.dice;
			*counts.entry(evaluation.result).or_default() += 1;
		}
		let counts = Distribution::from_weights(
			counts
				.into_iter()
				.map(|(outcome, count)| (outcome, Weight::from(count)))
		)
		.expect("at least one sample");
		Ok(Sampled { samples: n, counts })
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Samples.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The outcomes of a function, [sampled](Evaluator::sample) by evaluating it
/// many times, which estimate its exact [distribution](Distribution).
///
/// The [counts](Self::counts) weigh each outcome by the number of samples that
/// answered it, over a total of the number of [samples](Self::samples), so
/// that every query of a distribution answers its estimate from the samples.
/// The estimate is not exact, but its error is bounded: at any
/// [confidence](Self::error_bound), its cumulative distribution function is
/// within `ε` of the exact one at every outcome. The bound holds for every
/// distribution, discrete ones included, so it holds whatever the function.
/// An outcome that no sample answered is absent from the counts, however
/// possible it is.
///
/// # Examples
/// ```rust
/// use std::num::NonZeroU64;
///
/// use rand::{SeedableRng, rngs::StdRng};
/// use xdy::{Evaluator, Probability, Weight, compile};
///
/// let mut evaluator = Evaluator::new(compile("1D2")?);
/// let n = NonZeroU64::new(100).unwrap();
/// let sampled = evaluator.sample([], &mut StdRng::seed_from_u64(7), n, 100)?;
/// assert_eq!(sampled.samples(), n);
/// assert_eq!(sampled.counts().len(), 2);
///
/// let heads = sampled.counts().get(1);
/// let tails = sampled.counts().get(2);
/// assert_eq!(
///     sampled.to_string(),
///     format!("sampled (n = 100):\n1: {heads}\n2: {tails}\n")
/// );
///
/// let confidence = Probability::new(Weight::from(9u8), Weight::from(10u8))?;
/// let half = Probability::new(Weight::ONE, Weight::from(2u8))?;
/// let error = sampled.counts().cdf(1).to_f64() - half.to_f64();
/// assert!(error.abs() <= sampled.error_bound(&confidence));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Sampled
{
	/// The number of samples.
	samples: NonZeroU64,

	/// The number of samples that answered each outcome, which total
	/// [`samples`](Self::samples).
	counts: Distribution
}

impl Sampled
{
	/// Answer the number of samples.
	///
	/// # Returns
	/// The number of samples, which is the [total](Distribution::total) of the
	/// [counts](Self::counts).
	#[inline]
	pub fn samples(&self) -> NonZeroU64 { self.samples }

	/// Answer the counts of the outcomes of the samples.
	///
	/// # Returns
	/// The distribution that weighs each outcome by the number of samples that
	/// answered it.
	#[inline]
	pub fn counts(&self) -> &Distribution { &self.counts }

	/// Answer the counts of the outcomes of the samples, consuming the sample.
	///
	/// # Returns
	/// The distribution that weighs each outcome by the number of samples that
	/// answered it.
	#[inline]
	pub fn into_counts(self) -> Distribution { self.counts }

	/// Answer the bound on the error of the estimate at the given confidence,
	/// by the Dvoretzky–Kiefer–Wolfowitz inequality: with probability at least
	/// `confidence`, the [cumulative distribution function](Distribution::cdf)
	/// of the [counts](Self::counts) differs from the exact one by at most the
	/// bound, at every outcome at once.
	///
	/// # Parameters
	/// - `confidence`: The confidence, `1 - δ`.
	///
	/// # Returns
	/// The bound `ε = √(ln(2/δ) / 2n)`, where `n` is the number of
	/// [samples](Self::samples). The bound shrinks as the square root of the
	/// samples grows, and it is [`f64::INFINITY`] when `confidence` is one,
	/// since no finite sample is certain. A bound of one or more says nothing,
	/// since no two probabilities differ by more than one.
	///
	/// # Notes
	/// `2/δ` is computed exactly and rounded once, so the bound is accurate
	/// unless `2/δ` exceeds [`f64::MAX`], when it is [`f64::INFINITY`].
	///
	/// # References
	/// - A. Dvoretzky, J. Kiefer, and J. Wolfowitz, "[Asymptotic Minimax
	///   Character of the Sample Distribution Function and of the Classical
	///   Multinomial Estimator](https://doi.org/10.1214/aoms/1177728174)", _The
	///   Annals of Mathematical Statistics_ 27(3), 642–669, 1956, which proves
	///   the inequality, with an unspecified constant.
	/// - P. Massart, "[The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz
	///   Inequality](https://doi.org/10.1214/aop/1176990746)", _The Annals of
	///   Probability_ 18(3), 1269–1283, 1990, which proves the constant `2` of
	///   the two-sided bound, and that it cannot be improved.
	/// - H. W. J. Reeve, "[A short proof of the
	///   Dvoretzky–Kiefer–Wolfowitz–Massart
	///   inequality](https://arxiv.org/abs/2403.16651)", 2024, whose one-sided
	///   bound holds for any distribution function, discrete ones included, and
	///   whose two sides together give this bound.
	///
	/// # Examples
	/// ```rust
	/// use std::num::NonZeroU64;
	///
	/// use rand::rng;
	/// use xdy::{Evaluator, Probability, Weight, compile};
	///
	/// let mut evaluator = Evaluator::new(compile("3D6")?);
	/// let n = NonZeroU64::new(50_000).unwrap();
	/// let sampled = evaluator.sample([], &mut rng(), n, 3 * 50_000)?;
	/// let confidence = Probability::new(Weight::from(19u8), Weight::from(20u8))?;
	/// assert_eq!(format!("{:.4}", sampled.error_bound(&confidence)), "0.0061");
	/// assert_eq!(sampled.error_bound(&Probability::ONE), f64::INFINITY);
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn error_bound(&self, confidence: &Probability) -> f64
	{
		// δ = 1 - numerator/denominator = (denominator - numerator) /
		// denominator.
		let denominator = confidence.denominator();
		let risk = denominator
			.checked_sub(confidence.numerator())
			.expect("a probability never exceeds one");
		if risk.is_zero()
		{
			return f64::INFINITY
		}
		let ratio = (&Weight::from(2u8) * denominator).ratio_to_f64(&risk);
		(ratio.ln() / (2.0 * self.samples.get() as f64)).sqrt()
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Formatting.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Formats a label, `sampled (n = samples):`, and then one line per outcome
/// that some sample answered, `outcome: count`, in ascending order of outcome,
/// as a [`Distribution`] formats its weights.
impl Display for Sampled
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		writeln!(f, "sampled (n = {}):", self.samples)?;
		write!(f, "{}", self.counts)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// A [`Sampled`] as it deserializes, before its invariants are checked.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedSampled
{
	/// The purported number of samples.
	samples: NonZeroU64,

	/// The counts of the outcomes.
	counts: Distribution
}

/// Deserializes a sample from its number of samples and its counts, refusing
/// one whose counts do not total its number of samples.
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Sampled
{
	fn deserialize<D: Deserializer<'de>>(
		deserializer: D
	) -> Result<Self, D::Error>
	{
		use de::Error as _;
		let UncheckedSampled { samples, counts } =
			UncheckedSampled::deserialize(deserializer)?;
		if counts.total() != &Weight::from(samples.get())
		{
			return Err(D::Error::custom(format_args!(
				"sample counts total {}, not its {samples} samples",
				counts.total()
			)))
		}
		Ok(Self { samples, counts })
	}
}
