import XdySpec.Unlimited

/-!
# Unlimited reroll tests

The closed form of limited rerolls of a six-sided die, the probability that
they still match, their limits, die by die and record by record, and the
rerolls of dice whose every face matches, which the limited rerolls leave
alone and the unlimited one crosses out.
-/

namespace Xdy.Spec.Test.Unlimited

open Filter Topology
open scoped ENNReal

/-- The faces of a six-sided die. -/
private abbrev d6 : List Int := [1, 2, 3, 4, 5, 6]

/-- A record whose results are sorted keeps them all, if it drops none. -/
private theorem kept_sorted {xs : List Int} {d : Die}
    (h : xs.Pairwise (· ≤ ·)) :
    ({ results := xs, die := d } : Record).kept = xs := by
  have : xs.mergeSort (fun a b => decide (a ≤ b)) = xs :=
    List.mergeSort_of_pairwise (by simpa using h)
  simp [Record.kept, Record.sorted, this]

/-- The faces of a six-sided die that `= 1` does not match. -/
private theorem unmatched_d6 :
    (Die.custom d6).faces.filter (!Criterion.matches (.within 1 1) ·) =
      [2, 3, 4, 5, 6] := by
  decide

/-! ## Closed forms -/

-- `1D6 reroll = 1` matches with probability `1/6`.
example : matched (.within 1 1) (customDie d6) = 1 / 6 := by
  rw [matched_customDie _ (by decide)]
  norm_num [Criterion.matches]

-- Limited to 3 times, it shows a `1` only if every draw showed one, and a `2`
-- if any draw first showed it.
example : limitedReplacement (.within 1 1) (customDie d6) 3 1 =
    (1 / 6) ^ 2 * (1 / 6) := by
  rw [show 3 = 2 + 1 from rfl, limitedReplacement_apply,
    matched_customDie _ (by decide), customDie_apply (by decide)]
  norm_num [Criterion.matches]
example : limitedReplacement (.within 1 1) (customDie d6) 3 2 =
    (1 + 1 / 6 + (1 / 6) ^ 2) * (1 / 6) := by
  rw [show 3 = 2 + 1 from rfl, limitedReplacement_apply,
    matched_customDie _ (by decide), customDie_apply (by decide)]
  norm_num [Criterion.matches, Finset.sum_range_succ]

-- After 3 draws, the die would draw again with probability `(1/6)³`.
example : matched (.within 1 1) (limitedReplacement (.within 1 1)
    (customDie d6) 3) = (1 / 6) ^ 3 := by
  rw [matched_limitedReplacement _ _ (by decide),
    matched_customDie _ (by decide)]
  norm_num [Criterion.matches]

/-! ## Limits -/

-- `1D6 reroll = 1 ∞ times` shows a `2` with probability `1/5`, the limit of
-- the limited rerolls, and a `1` never.
example : Tendsto (fun n => limitedReplacement (.within 1 1) (customDie d6) n 2)
    atTop (𝓝 (1 / 5)) := by
  have h := tendsto_limitedReplacement_faces
    (d := .custom d6) (c := .within 1 1) (by decide) 2
  convert h using 2
  · rfl
  · rw [PMF.ofMultiset_apply]
    simp [unmatched_d6]
example : Tendsto (fun n => limitedReplacement (.within 1 1) (customDie d6) n 1)
    atTop (𝓝 0) := by
  have h := tendsto_limitedReplacement_faces
    (d := .custom d6) (c := .within 1 1) (by decide) 1
  convert h using 2
  · rfl
  · rw [PMF.ofMultiset_apply]
    simp [unmatched_d6]

-- `1D2 reroll = 1`, from a roll of `1`, shows a certain `2` in the limit.
example : Tendsto (fun n => reroll { results := [1], die := .standard 2 }
      (.within 1 1) (some n) { results := [2], die := .standard 2 })
    atTop (𝓝 1) := by
  have hw : (Die.standard 2).facesWithin (.within 1 1) = false := by decide
  have hf : (Die.standard 2).faces = [1, 2] := by decide
  have h2 : ∀ h, PMF.ofMultiset ({2} : Multiset Int) h = PMF.pure 2 :=
    fun h => PMF.ext fun x => by
      by_cases hx : x = 2 <;> simp [PMF.ofMultiset_apply, hx]
  have hnone : reroll { results := [1], die := .standard 2 } (.within 1 1)
      none = PMF.pure { results := [2], die := .standard 2 } := by
    simp only [reroll,
      kept_sorted (by decide : ([1] : List Int).Pairwise (· ≤ ·)), hw,
      Bool.false_eq_true, ↓reduceDIte, rerollKept]
    simp only [Criterion.matches, hf, List.foldl]
    simp [h2, PMF.pure_map]
    rfl
  have h := tendsto_reroll (r := { results := [1], die := .standard 2 })
    (c := .within 1 1) (by simp [hw]) { results := [2], die := .standard 2 }
  simpa [hnone] using h

/-! ## Every face matches -/

-- `1D1 reroll = 1 twice` leaves the die alone.
example (n : Nat) : limitedReplacement (.within 1 1) (customDie [1]) n =
    customDie [1] :=
  limitedReplacement_faces_of_facesWithin (d := .standard 1) (by decide)
    (by decide) n

-- `2D1 reroll = 1 ∞ times` crosses out both dice, which sum to `0`.
example : (reroll { results := [1, 1], die := .standard 1 } (.within 1 1)
    none).map Record.sum = PMF.pure 0 := by
  rw [sum_reroll_crossOut (by decide),
    kept_sorted (by decide : ([1, 1] : List Int).Pairwise (· ≤ ·))]
  rfl

/-! ## Docstring examples, verbatim -/

example : matched (.within 1 1) (PMF.pure 1) = 1 := by
  rw [matched, tsum_eq_single 1 fun x hx => by simp [hx]]
  simp [Criterion.matches]

end Xdy.Spec.Test.Unlimited
