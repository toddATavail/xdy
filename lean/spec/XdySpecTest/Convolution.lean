import XdySpec.Convolution

/-!
# Convolution tests

Lifts, convolutions and convolution powers of small laws, proved of the
definitions and the lemmas, since a `PMF` cannot be computed.
-/

namespace Xdy.Spec.Test

/-! ## Lifts -/

example : lift (· * ·) (PMF.pure 2) (PMF.pure 3) = PMF.pure 6 := by
  simp [lift, PMF.pure_map]
-- Saturating addition clamps the true sum.
example : lift add (PMF.pure i32Max) (PMF.pure 1) = PMF.pure i32Max := by
  rw [lift_add]; simp [conv, lift, PMF.pure_map, clamp, i32Min, i32Max]
-- Saturating subtraction clamps the true difference.
example : lift sub (PMF.pure i32Min) (PMF.pure 1) = PMF.pure i32Min := by
  rw [lift_sub]; simp [conv, lift, PMF.pure_map, clamp, i32Min, i32Max]

/-! ## Convolution powers -/

/-- Two faces, each with probability `1 / 2`. -/
private theorem coin (x : Int) :
    standardDie 2 x = if x = 1 ∨ x = 2 then 2⁻¹ else 0 := by
  simp only [standardDie, show ¬(2 : Int) ≤ 0 by decide, dite_false,
    PMF.ofMultiset_apply]
  split_ifs with h
  · rcases h with rfl | rfl <;>
      simp [Multiset.count_eq_one_of_mem (Finset.nodup _)]
  · simp [Multiset.count_eq_zero]; omega

/-- Two two-sided dice sum to `3` in two ways of four. -/
private theorem two_coins : convPow (standardDie 2) 2 3 = 1 / 2 := by
  rw [convPow, convPow, convPow, zero_conv, conv, lift_apply]
  simp only [coin]
  rw [tsum_eq_sum (s := {1, 2}) (by
      intro x hx
      simp at hx
      exact ENNReal.tsum_eq_zero.mpr fun y => by simp; omega),
    Finset.sum_pair (by decide),
    tsum_eq_single 2 (by intro y hy; simp; omega),
    tsum_eq_single 1 (by intro y hy; simp; omega)]
  simp
  rw [← two_mul, ← mul_assoc,
    ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_mul]
-- A certain value, summed, is a certain multiple of it.
example (n : Nat) : convPow (PMF.pure 3) n = PMF.pure (3 * (n : Int)) := by
  induction n with
  | zero => rfl
  | succ n ih => simp [convPow, ih, conv, lift, PMF.pure_map, Int.mul_add]
example (d : PMF Int) :
    convPow d 5 = conv (convPow d 2) (convPow d 3) := convPow_add d 2 3

/-! ## Sets of dice -/

example (die : PMF Int) :
    (rollDice (-1) die).map (·.results.sum) = PMF.pure 0 := by
  rw [rollDice_sum]; rfl
example :
    (rollDice 2 (standardDie 2)).map (·.results.sum) 3 = 1 / 2 := by
  rw [rollDice_sum]; exact two_coins

/-! ## Case 1: sums without drops -/

-- Faces of mixed sign pass the test when every copy of the least face stays
-- within `i32`: three dice with faces `-2` and `5`.
example : (rollDice 3 (customDie [-2, 5])).map (·.sum)
    = (convPow (customDie [-2, 5]) 3).map clamp :=
  rollDice_saturating_sum 3 _ (-2) 5 (fun x hx => by
    simp [customDie] at hx; omega)
    (.inr (.inr (by decide)))
-- Nonpositive faces pass the test however many dice there are.
example (n : Int) : (rollDice n (customDie [i32Min, -1])).map (·.sum)
    = (convPow (customDie [i32Min, -1]) n.toNat).map clamp :=
  rollDice_saturating_sum n _ i32Min (-1) (fun x hx => by
    simp [customDie] at hx
    rcases hx with rfl | rfl <;> decide)
    (.inr (.inl (by decide)))
-- A record without drops sums to one clamp of its true sum, in any order.
example : ({ results := [5, -2, 5] } : Record).sum = clamp 8 := by
  rw [Record.sum_eq_clamp _ rfl rfl (-2) 5 (by decide) (by decide)
    (.inr (.inr (by decide)))]
  rfl

/-! ## Docstring examples, verbatim -/

example : lift (· * ·) (PMF.pure 2) (PMF.pure 3) = PMF.pure 6 := by
  simp [lift, PMF.pure_map]
example : conv (PMF.pure 1) (PMF.pure 2) = PMF.pure 3 := by
  simp [conv, lift, PMF.pure_map]
example (d : PMF Int) : convPow d 0 = PMF.pure 0 := rfl
example (d : PMF Int) : convPow d 1 = d := by simp [convPow, zero_conv]
example (d : PMF Int) (n : Nat) :
    convPow d (n + n) = conv (convPow d n) (convPow d n) :=
  convPow_add d n n
example (die : PMF Int) :
    (rollDice 2 die).map (·.results.sum) = conv die die := by
  rw [rollDice_sum]; simp [convPow, zero_conv]
example : (rollDice 3 (standardDie 6)).map (·.sum)
    = (convPow (standardDie 6) 3).map clamp :=
  rollDice_saturating_sum 3 _ 1 6 (fun x hx => by
    simp [standardDie] at hx; omega)
    (.inl (by decide))

end Xdy.Spec.Test
