//! # Weight tests
//!
//! Herein are the tests for [`Weight`], the unbounded integer that weighs the
//! outcomes of an exact distribution. Plain [`BigUint`] arithmetic is the
//! reference throughout: every operation must agree with it, and must answer
//! a weight in its one representation, held as a [`u128`] just when its value
//! fits in one.
//!
//! Unit tests pin the promotions and demotions at [`u128::MAX`]. Property
//! tests draw operands from every region that matters, including either side
//! of that boundary and values many limbs wide, and hold every operation to
//! the reference. A [counting allocator](CountingAllocator) proves that
//! arithmetic on weights that fit in a `u128` never allocates.

use std::{
	alloc::{GlobalAlloc, Layout, System},
	cell::Cell,
	fmt::{self, Write as _},
	hash::{DefaultHasher, Hash, Hasher},
	hint::black_box,
	num::IntErrorKind
};

use num_bigint::BigUint;
use num_traits::{CheckedSub, ToPrimitive};
use proptest::{collection::vec, prelude::*};

use crate::{ParseWeightError, Weight};

////////////////////////////////////////////////////////////////////////////////
//                                  Support.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The system allocator, counting the allocations of each thread, so that a
/// test can prove that some code does not allocate. It serves the whole test
/// binary, at the cost of one thread-local increment per allocation.
struct CountingAllocator;

thread_local! {
	/// The number of allocations made by the current thread.
	static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Count an allocation by the current thread.
fn count_allocation()
{
	// A thread that is exiting has no counter left to increment.
	let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

// SAFETY: Every method forwards to the system allocator unchanged; counting
// touches only a thread-local counter, which itself never allocates.
unsafe impl GlobalAlloc for CountingAllocator
{
	unsafe fn alloc(&self, layout: Layout) -> *mut u8
	{
		count_allocation();
		// SAFETY: The caller upholds the contract of `GlobalAlloc::alloc`.
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8
	{
		count_allocation();
		// SAFETY: The caller upholds the contract of
		// `GlobalAlloc::alloc_zeroed`.
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize
	) -> *mut u8
	{
		count_allocation();
		// SAFETY: The caller upholds the contract of `GlobalAlloc::realloc`.
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout)
	{
		// SAFETY: The caller upholds the contract of `GlobalAlloc::dealloc`.
		unsafe { System.dealloc(ptr, layout) }
	}
}

/// The allocator of the test binary.
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run a closure, and count the allocations it makes.
///
/// # Parameters
/// - `f`: The closure.
///
/// # Returns
/// The number of allocations that `f` made, including any that its result
/// holds.
fn allocations<R>(f: impl FnOnce() -> R) -> u64
{
	let before = ALLOCATIONS.with(Cell::get);
	black_box(f());
	ALLOCATIONS.with(Cell::get) - before
}

/// A [writer](fmt::Write) that discards everything, so that formatting can be
/// tested without allocating a string.
struct Discard;

impl fmt::Write for Discard
{
	fn write_str(&mut self, _: &str) -> fmt::Result { Ok(()) }
}

/// Answer the hash of a value.
///
/// # Parameters
/// - `value`: The value.
///
/// # Returns
/// The hash.
fn hash_of(value: &impl Hash) -> u64
{
	let mut hasher = DefaultHasher::new();
	value.hash(&mut hasher);
	hasher.finish()
}

/// Answer [`u128::MAX`], as a big integer.
///
/// # Returns
/// The big integer.
fn max() -> BigUint { BigUint::from(u128::MAX) }

/// Answer 2¹²⁸, the least weight that does not fit in a [`u128`].
///
/// # Returns
/// The weight.
fn least_big() -> Weight { Weight::from(u128::MAX) + Weight::ONE }

/// Check that a weight has the expected value, in its one representation.
///
/// # Parameters
/// - `weight`: The weight.
/// - `expected`: The expected value.
///
/// # Errors
/// If the weight has another value, or is held as a [`u128`] just when its
/// value does not fit in one.
fn check(weight: &Weight, expected: &BigUint) -> Result<(), TestCaseError>
{
	prop_assert_eq!(&weight.to_biguint(), expected);
	prop_assert_eq!(
		weight.is_small(),
		*expected <= max(),
		"{:?} is not in its one representation",
		weight
	);
	Ok(())
}

/// Answer a strategy that generates the values of weights, from every region
/// that matters: tiny values, including zero and one; anywhere in [`u128`];
/// either side of [`u128::MAX`]; and values many limbs wide.
///
/// # Returns
/// The strategy.
pub(super) fn value() -> impl Strategy<Value = BigUint>
{
	prop_oneof![
		(0u128..=16).prop_map(BigUint::from),
		any::<u128>().prop_map(BigUint::from),
		(0u128..=1024).prop_map(|d| BigUint::from(u128::MAX - d)),
		(1u128..=1024).prop_map(|d| max() + d),
		vec(any::<u32>(), 5..=12).prop_map(BigUint::new)
	]
}

////////////////////////////////////////////////////////////////////////////////
//                            Promotion boundary.                             //
////////////////////////////////////////////////////////////////////////////////

/// Addition promotes exactly when the sum exceeds [`u128::MAX`].
#[test]
fn test_promotion_by_addition()
{
	let at_max = Weight::from(u128::MAX - 1) + Weight::ONE;
	assert!(at_max.is_small());
	assert_eq!(at_max.to_biguint(), max());
	let past_max = least_big();
	assert!(!past_max.is_small());
	assert_eq!(past_max.to_biguint(), max() + 1u8);
}

/// Multiplication promotes exactly when the product exceeds [`u128::MAX`].
#[test]
fn test_promotion_by_multiplication()
{
	let at_max = Weight::from(u64::MAX) * Weight::from(u64::MAX as u128 + 2);
	assert!(at_max.is_small());
	assert_eq!(at_max.to_biguint(), max());
	let two_64 = Weight::from(1u128 << 64);
	let past_max = &two_64 * &two_64;
	assert!(!past_max.is_small());
	assert_eq!(past_max, least_big());
}

/// Powers promote exactly when they exceed [`u128::MAX`], and agree with big
/// integer powers beyond it.
#[test]
fn test_promotion_by_power()
{
	let two = Weight::from(2u8);
	assert!(two.pow(127).is_small());
	assert_eq!(two.pow(128), least_big());
	assert_eq!(
		Weight::from(6u8).pow(100).to_biguint(),
		BigUint::from(6u8).pow(100)
	);
}

/// Subtraction demotes a big weight whenever the difference fits in a
/// [`u128`], and refuses a subtrahend larger than the minuend.
#[test]
fn test_demotion_by_subtraction()
{
	let big = least_big();
	let demoted = big.checked_sub(&Weight::ONE).unwrap();
	assert!(demoted.is_small());
	assert_eq!(demoted, Weight::from(u128::MAX));
	let bigger = &big + &big;
	let demoted = bigger.checked_sub(&big).unwrap().checked_sub(&Weight::ONE);
	assert_eq!(demoted, Some(Weight::from(u128::MAX)));
	assert_eq!(big.checked_sub(&big), Some(Weight::ZERO));
	assert!(big.checked_sub(&big).unwrap().is_small());
	assert_eq!(Weight::from(u128::MAX).checked_sub(&big), None);
	assert_eq!(big.checked_sub(&bigger), None);
	assert_eq!(Weight::ZERO.checked_sub(&Weight::ONE), None);
}

/// Multiplication by zero demotes a big weight, in every combination of owned
/// and borrowed operands.
#[test]
fn test_demotion_by_multiplication_by_zero()
{
	let big = least_big();
	let zero = Weight::ZERO;
	for product in [
		&big * &zero,
		&zero * &big,
		big.clone() * zero.clone(),
		zero.clone() * big.clone(),
		big.clone() * &zero,
		zero.clone() * &big,
		&big * zero.clone(),
		&zero * big.clone()
	]
	{
		assert!(product.is_small());
		assert!(product.is_zero());
	}
}

/// The zeroth power of a big weight is one.
#[test]
fn test_demotion_by_zeroth_power()
{
	let one = least_big().pow(0);
	assert!(one.is_small());
	assert_eq!(one, Weight::ONE);
	assert_eq!(Weight::ZERO.pow(0), Weight::ONE);
}

/// Every big weight exceeds every small one.
#[test]
fn test_ordering_across_boundary()
{
	assert!(Weight::from(u128::MAX) < least_big());
	assert!(Weight::ZERO < least_big());
	assert!(least_big() < &least_big() + &Weight::ONE);
	assert_eq!(Weight::default(), Weight::ZERO);
}

/// The empty sum is zero, and the empty product is one.
#[test]
fn test_empty_sum_and_product()
{
	assert_eq!(std::iter::empty::<Weight>().sum::<Weight>(), Weight::ZERO);
	assert_eq!(
		std::iter::empty::<Weight>().product::<Weight>(),
		Weight::ONE
	);
}

////////////////////////////////////////////////////////////////////////////////
//                                Conversions.                                //
////////////////////////////////////////////////////////////////////////////////

/// Bits and floating-point conversion hold at their edges.
#[test]
fn test_bits_and_to_f64()
{
	assert_eq!(Weight::ZERO.bits(), 0);
	assert_eq!(Weight::ONE.bits(), 1);
	assert_eq!(Weight::from(u128::MAX).bits(), 128);
	assert_eq!(least_big().bits(), 129);
	assert_eq!(Weight::from(u128::MAX).to_f64(), 2f64.powi(128));
	assert_eq!(Weight::from(2u8).pow(1023).to_f64(), 2f64.powi(1023));
	assert_eq!(Weight::from(2u8).pow(1024).to_f64(), f64::INFINITY);
}

/// Parsing accepts exactly what [`u128`] accepts, without its upper bound,
/// and answers every weight in its one representation.
#[test]
fn test_parse()
{
	assert_eq!("0".parse(), Ok(Weight::ZERO));
	assert_eq!("+5".parse(), Ok(Weight::from(5u8)));
	let padded = format!("{}1", "0".repeat(60));
	let parsed: Weight = padded.parse().unwrap();
	assert!(parsed.is_small());
	assert_eq!(parsed, Weight::ONE);
	let big: Weight =
		"340282366920938463463374607431768211456".parse().unwrap();
	assert!(!big.is_small());
	assert_eq!(big, least_big());
	assert_eq!("".parse::<Weight>(), Err(ParseWeightError::Empty));
	assert_eq!("+".parse::<Weight>(), Err(ParseWeightError::Empty));
	let nines = "9".repeat(60);
	for invalid in [
		"++1",
		"-1",
		"1_000",
		" 1",
		"1.0",
		"0x10",
		// Invalid characters past the point where a u128 would overflow.
		&format!("{nines}x"),
		&format!("{nines}_9"),
		&format!("{nines}+9")
	]
	{
		assert_eq!(
			invalid.parse::<Weight>(),
			Err(ParseWeightError::InvalidDigit),
			"{invalid:?}"
		);
	}
}

/// Weights serialize as decimal strings, and deserialize only from them.
#[cfg(feature = "serde")]
#[test]
fn test_serde()
{
	assert_eq!(serde_json::to_string(&Weight::from(5u8)).unwrap(), "\"5\"");
	assert_eq!(
		serde_json::to_string(&least_big()).unwrap(),
		"\"340282366920938463463374607431768211456\""
	);
	assert_eq!(
		serde_json::from_str::<Weight>("\"5\"").unwrap(),
		Weight::from(5u8)
	);
	let big: Weight =
		serde_json::from_str("\"340282366920938463463374607431768211456\"")
			.unwrap();
	assert_eq!(big, least_big());
	for invalid in ["5", "\"-1\"", "\"\"", "null", "5.0"]
	{
		assert!(
			serde_json::from_str::<Weight>(invalid).is_err(),
			"{invalid} deserialized"
		);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                Allocation.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Nothing that stays within [`u128`] allocates: not arithmetic, comparison,
/// hashing, conversion, formatting, nor parsing.
#[test]
fn test_small_never_allocates()
{
	let count = allocations(|| {
		let a = Weight::from(u64::MAX);
		let b = Weight::from(12345u32);
		let mut w = &a * &b;
		w += &b;
		w += b.clone();
		w *= &b;
		w *= b.clone();
		let w = w.clone() + a.clone() + &b;
		let w = &w * b.clone();
		let w = w.checked_sub(&a).unwrap();
		let power = Weight::from(6u8).pow(49);
		let sum: Weight = [a.clone(), b.clone(), w.clone()].into_iter().sum();
		let product: Weight = [&b, &b, &b].into_iter().product();
		let ordered = a < b && b <= w && power != sum;
		let hashes = hash_of(&w) ^ hash_of(&product);
		let converted = w.to_f64() + power.bits() as f64;
		let zero = Weight::ZERO.is_zero() && !w.is_zero();
		write!(Discard, "{w} {power} {sum} {product}").unwrap();
		let parsed: Weight =
			"340282366920938463463374607431768211455".parse().unwrap();
		(
			w, power, sum, product, ordered, hashes, converted, zero, parsed
		)
	});
	assert_eq!(count, 0);
}

/// The allocation counter counts, so that [`test_small_never_allocates`]
/// proves something.
#[test]
fn test_promotion_allocates()
{
	assert!(allocations(least_big) > 0);
}

/// A product of borrowed big weights allocates only its own storage, without
/// copying either operand, and accumulating it into a big sum allocates
/// nothing more, unless the sum outgrows its storage, as convolutions
/// accumulate their products.
#[test]
fn test_borrowed_product_allocates_once()
{
	let a = Weight::from(u128::MAX).pow(2);
	let b = Weight::from(u128::MAX).pow(3);
	assert_eq!(allocations(|| &a * &b), 1);
	let mut sum = Weight::from(u128::MAX).pow(9);
	assert_eq!(allocations(|| sum += &a * &b), 1);
}

////////////////////////////////////////////////////////////////////////////////
//                               Divisibility.                                //
////////////////////////////////////////////////////////////////////////////////

/// Answer the greatest common divisor of two big integers by the binary
/// algorithm, which shares nothing with the Euclidean algorithm of
/// [`Weight::gcd`], as a reference for it.
///
/// # Parameters
/// - `a`: The first big integer.
/// - `b`: The second big integer.
///
/// # Returns
/// The greatest common divisor, which is zero just when both are zero.
fn binary_gcd(mut a: BigUint, mut b: BigUint) -> BigUint
{
	let (Some(za), Some(zb)) = (a.trailing_zeros(), b.trailing_zeros())
	else
	{
		// One of them is zero, so the other is the divisor.
		return a | b
	};
	a >>= za;
	b >>= zb;
	loop
	{
		if a > b
		{
			std::mem::swap(&mut a, &mut b);
		}
		b -= &a;
		match b.trailing_zeros()
		{
			None => return a << za.min(zb),
			Some(z) => b >>= z
		}
	}
}

/// Greatest common divisors, least common multiples, and exact quotients at
/// zero, at one, and either side of [`u128::MAX`].
#[test]
fn test_divisibility()
{
	let w = |n: u128| Weight::from(n);
	assert_eq!(w(0).gcd(&w(0)), w(0));
	assert_eq!(w(0).gcd(&w(7)), w(7));
	assert_eq!(w(12).gcd(&w(18)), w(6));
	assert_eq!(w(0).lcm(&w(7)), w(0));
	assert_eq!(w(4).lcm(&w(6)), w(12));
	assert_eq!(w(81).lcm(&w(27)), w(81));
	assert_eq!(w(12).exact_div(&w(4)), w(3));
	assert_eq!(w(0).exact_div(&least_big()), w(0));
	// Two big weights whose greatest common divisor is small, and whose
	// least common multiple is big.
	let big = least_big();
	let odd = &big + &Weight::ONE;
	assert_eq!(big.gcd(&odd), Weight::ONE);
	assert!(big.gcd(&odd).is_small());
	assert_eq!(big.lcm(&odd), &big * &odd);
	// The least common multiple of two small weights may be big.
	let lcm = w(u128::MAX).lcm(&w(2));
	assert_eq!(lcm.to_biguint(), max() * 2u8);
	assert_eq!(lcm.exact_div(&w(2)), w(u128::MAX));
	assert!(lcm.exact_div(&w(2)).is_small());
}

////////////////////////////////////////////////////////////////////////////////
//                              Property tests.                               //
////////////////////////////////////////////////////////////////////////////////

proptest! {
	/// Addition agrees with big integer addition, in every combination of
	/// owned and borrowed operands and through compound assignment.
	#[test]
	fn test_add_agrees(a in value(), b in value())
	{
		let (x, y) = (Weight::from_biguint(a.clone()), Weight::from_biguint(b.clone()));
		let expected = &a + &b;
		check(&(&x + &y), &expected)?;
		check(&(x.clone() + y.clone()), &expected)?;
		check(&(x.clone() + &y), &expected)?;
		check(&(&x + y.clone()), &expected)?;
		let mut z = x.clone();
		z += &y;
		check(&z, &expected)?;
		let mut z = x.clone();
		z += y.clone();
		check(&z, &expected)?;
	}

	/// Multiplication agrees with big integer multiplication, in every
	/// combination of owned and borrowed operands and through compound
	/// assignment.
	#[test]
	fn test_mul_agrees(a in value(), b in value())
	{
		let (x, y) = (Weight::from_biguint(a.clone()), Weight::from_biguint(b.clone()));
		let expected = &a * &b;
		check(&(&x * &y), &expected)?;
		check(&(x.clone() * y.clone()), &expected)?;
		check(&(x.clone() * &y), &expected)?;
		check(&(&x * y.clone()), &expected)?;
		let mut z = x.clone();
		z *= &y;
		check(&z, &expected)?;
		let mut z = x.clone();
		z *= y.clone();
		check(&z, &expected)?;
	}

	/// Subtraction agrees with big integer subtraction, and undoes addition
	/// down to the same representation and hash.
	#[test]
	fn test_checked_sub_agrees(a in value(), b in value())
	{
		let (x, y) = (Weight::from_biguint(a.clone()), Weight::from_biguint(b.clone()));
		match (x.checked_sub(&y), a.checked_sub(&b))
		{
			(Some(d), Some(expected)) => check(&d, &expected)?,
			(None, None) => {},
			(d, expected) => prop_assert!(false, "{:?} != {:?}", d, expected)
		}
		let undone = (&x + &y).checked_sub(&y).unwrap();
		prop_assert_eq!(&undone, &x);
		prop_assert_eq!(hash_of(&undone), hash_of(&x));
	}

	/// Powers agree with big integer powers.
	#[test]
	fn test_pow_agrees(a in value(), e in 0u32..=40)
	{
		check(&Weight::from_biguint(a.clone()).pow(e), &a.pow(e))?;
	}

	/// Sums and products agree with big integer sums and products, over owned
	/// and borrowed weights.
	#[test]
	fn test_sum_and_product_agree(values in vec(value(), 0..=8))
	{
		let weights: Vec<_> = values.iter().cloned().map(Weight::from_biguint).collect();
		let sum = values.iter().sum::<BigUint>();
		check(&weights.iter().sum(), &sum)?;
		check(&weights.iter().cloned().sum(), &sum)?;
		let product = values.iter().product::<BigUint>();
		check(&weights.iter().product(), &product)?;
		check(&weights.into_iter().product(), &product)?;
	}

	/// Greatest common divisors agree with the binary algorithm, least common
	/// multiples are the product over the greatest common divisor, and exact
	/// division undoes multiplication.
	#[test]
	fn test_divisibility_agrees(a in value(), b in value())
	{
		let (x, y) = (Weight::from_biguint(a.clone()), Weight::from_biguint(b.clone()));
		let gcd = binary_gcd(a.clone(), b.clone());
		check(&x.gcd(&y), &gcd)?;
		check(&y.gcd(&x), &gcd)?;
		let lcm = if gcd == BigUint::ZERO { BigUint::ZERO } else { &a * &b / &gcd };
		check(&x.lcm(&y), &lcm)?;
		if !y.is_zero()
		{
			check(&(&x * &y).exact_div(&y), &a)?;
		}
	}

	/// Equality and ordering agree with the big integers.
	#[test]
	fn test_ordering_agrees(a in value(), b in value())
	{
		let (x, y) = (Weight::from_biguint(a.clone()), Weight::from_biguint(b.clone()));
		prop_assert_eq!(x.cmp(&y), a.cmp(&b));
		prop_assert_eq!(x == y, a == b);
	}

	/// Digits agree with the big integers' 32-bit digits, written into wider
	/// slots and read back from them, leading zeros and all.
	#[test]
	fn test_u32_digits_agree(a in value(), pad in 0usize..=3)
	{
		let x = Weight::from_biguint(a.clone());
		prop_assert_eq!(x.to_u32_digits(), a.to_u32_digits());
		let mut digits = vec![0u32; a.to_u32_digits().len() + pad];
		x.write_u32_digits(&mut digits);
		check(&Weight::from_u32_digits(&digits), &a)?;
	}

	/// Construction, bits, zero, and floating-point conversion agree with the
	/// big integers.
	#[test]
	fn test_conversions_agree(a in value())
	{
		let x = Weight::from_biguint(a.clone());
		check(&x, &a)?;
		prop_assert_eq!(x.bits(), a.bits());
		prop_assert_eq!(x.is_zero(), a == BigUint::ZERO);
		prop_assert_eq!(x.to_f64(), a.to_f64().unwrap());
	}

	/// Formatting agrees with the big integers, and parsing inverts it.
	#[test]
	fn test_display_and_parse_agree(a in value())
	{
		let x = Weight::from_biguint(a.clone());
		let text = x.to_string();
		prop_assert_eq!(&text, &a.to_string());
		check(&text.parse().unwrap(), &a)?;
	}

	/// Parsing agrees with [`u128`] wherever `u128` does not overflow, and
	/// with the big integers on every string of decimal digits, however long.
	#[test]
	fn test_parse_agrees(
		s in prop_oneof![
			"[+]?[0-9+_x-]{0,60}",
			"[+]?[0-9]{30,60}",
			"[+]?[0-9]{30,60}[+_x-][0-9]{0,5}"
		]
	)
	{
		let parsed = s.parse::<Weight>();
		match s.parse::<u128>()
		{
			Ok(n) => check(&parsed.unwrap(), &BigUint::from(n))?,
			Err(e) if *e.kind() != IntErrorKind::PosOverflow =>
			{
				prop_assert!(parsed.is_err(), "{:?} parsed", s)
			},
			Err(_) =>
			{
				let digits = s.strip_prefix('+').unwrap_or(&s);
				if digits.bytes().all(|b| b.is_ascii_digit())
				{
					check(&parsed.unwrap(), &digits.parse().unwrap())?
				}
				else
				{
					prop_assert_eq!(parsed, Err(ParseWeightError::InvalidDigit))
				}
			}
		}
	}

	/// Serialization round trips through a decimal string.
	#[cfg(feature = "serde")]
	#[test]
	fn test_serde_round_trips(a in value())
	{
		let json = serde_json::to_string(&Weight::from_biguint(a.clone())).unwrap();
		prop_assert_eq!(&json, &format!("\"{a}\""));
		check(&serde_json::from_str(&json).unwrap(), &a)?;
	}
}
