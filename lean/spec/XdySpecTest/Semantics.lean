import XdySpec.Semantics

/-!
# Semantics tests

The laws of ranges, dice and rerolls, including their degenerate cases, proved
of the definitions, since a `PMF` cannot be computed.
-/

namespace Xdy.Spec.Test

/-! ## Ranges -/

example : rollRange 3 1 = PMF.pure { results := [0], die := .range 3 1 } := by
  simp [rollRange]
example : rollRange 1 3 { results := [2], die := .range 1 3 } = 1 / 3 := by
  simp [rollRange, PMF.map_apply, PMF.ofMultiset_apply,
    Multiset.count_eq_one_of_mem (Finset.nodup _)]

/-! ## Dice -/

example : standardDie 0 = PMF.pure 0 := by simp [standardDie]
example : standardDie (-3) = PMF.pure 0 := by simp [standardDie]
example : standardDie 6 3 = 1 / 6 := by
  simp [standardDie, PMF.ofMultiset_apply,
    Multiset.count_eq_one_of_mem (Finset.nodup _)]
example : customDie [] = PMF.pure 0 := by simp [customDie]
example : customDie [1, 1, 2] 1 = 2 / 3 := by
  simp [customDie, PMF.ofMultiset_apply]
example : customDie [1, 1, 2] 3 = 0 := by
  simp [customDie, PMF.ofMultiset_apply]

/-! ## Sets of dice -/

example (die : PMF Int) : rollDice 0 die = PMF.pure {} := rfl
example (die : PMF Int) : rollDice (-1) die = PMF.pure {} := rfl
-- One die records its face as the only result.
example (die : PMF Int) :
    rollDice 1 die = die.map fun x => { results := [x] } := by
  simp [rollDice, Nat.repeat, PMF.pure_bind]; rfl

/-! ## Rerolls -/

/-- A record whose results are sorted keeps them all, if it drops none. -/
private theorem kept_sorted {xs : List Int} {d : Die}
    (h : xs.Pairwise (· ≤ ·)) :
    ({ results := xs, die := d } : Record).kept = xs := by
  have : xs.mergeSort (fun a b => decide (a ≤ b)) = xs :=
    List.mergeSort_of_pairwise (by simpa using h)
  simp [Record.kept, Record.sorted, this]

-- A redraw keeps a result that does not match, and a limited reroll draws
-- once if its limit is one, or `0`, which rerolls nothing.
example (die : PMF Int) :
    redraw (.within 1 1) die (PMF.pure 2) = PMF.pure 2 := by
  simp [redraw, Criterion.matches]
example (c : Criterion) (die : PMF Int) :
    limitedReplacement c die 1 = die := rfl
example (c : Criterion) (die : PMF Int) :
    limitedReplacement c die 0 = die := rfl
-- `1D2 reroll = 1`, from a roll of `1`, draws the die again.
example : reroll { results := [1], die := .standard 2 } (.within 1 1) (some 1)
    = (customDie [1, 2]).map
      fun x => { results := [x], die := .standard 2 } := by
  have hf : (Die.standard 2).faces = [1, 2] := by decide
  simp only [reroll, rerollKept,
    kept_sorted (by decide : ([1] : List Int).Pairwise (· ≤ ·)), hf,
    List.foldl, limitedReplacement]
  simp [Criterion.matches, PMF.pure_bind]
  rfl
-- A result that does not match is kept, and the drops are forgotten.
example : reroll { results := [2], die := .standard 2 } (.within 1 1)
    (some 3) = PMF.pure { results := [2], die := .standard 2 } := by
  simp only [reroll, rerollKept,
    kept_sorted (by decide : ([2] : List Int).Pairwise (· ≤ ·))]
  simp [Criterion.matches, PMF.pure_map, Record.push]
-- `1D2 reroll = 1 ∞ times`, from a roll of `1`, answers a certain `2`, the
-- conditional law of the die.
example : reroll { results := [1], die := .standard 2 } (.within 1 1) none
    = PMF.pure { results := [2], die := .standard 2 } := by
  have hw : (Die.standard 2).facesWithin (.within 1 1) = false := by decide
  have hf : (Die.standard 2).faces = [1, 2] := by decide
  have h2 : ∀ h, PMF.ofMultiset ({2} : Multiset Int) h = PMF.pure 2 :=
    fun h => PMF.ext fun x => by
      by_cases hx : x = 2 <;> simp [PMF.ofMultiset_apply, hx]
  simp only [reroll,
    kept_sorted (by decide : ([1] : List Int).Pairwise (· ≤ ·)), hw,
    Bool.false_eq_true, ↓reduceDIte, rerollKept]
  simp only [Criterion.matches, hf, List.foldl]
  simp [h2, PMF.pure_map]
  rfl
-- `2D1 reroll = 1 ∞ times` crosses out both dice, since every face matches.
example : reroll { results := [1, 1], die := .standard 1 } (.within 1 1)
    none = PMF.pure { die := .standard 1 } := by
  have hw : (Die.standard 1).facesWithin (.within 1 1) = true := by decide
  simp only [reroll,
    kept_sorted (by decide : ([1, 1] : List Int).Pairwise (· ≤ ·)), hw,
    ↓reduceDIte]
  simp [Criterion.matches]

/-! ## Docstring examples, verbatim -/

example : rollRange 3 1 = PMF.pure { results := [0], die := .range 3 1 } := by
  simp [rollRange]
example : standardDie 0 = PMF.pure 0 := by simp [standardDie]
example : customDie [] = PMF.pure 0 := by simp [customDie]
example : customDie [1, 1, 2] 1 = 2 / 3 := by
  simp [customDie, PMF.ofMultiset_apply]
example (die : PMF Int) : rollDice (-1) die = PMF.pure {} := rfl
example (die : PMF Int) :
    redraw (.within 1 1) die (PMF.pure 2) = PMF.pure 2 := by
  simp [redraw, Criterion.matches]
example (c : Criterion) (die : PMF Int) :
    limitedReplacement c die 1 = die := rfl
example (p : PMF Int) : rerollKept .none [] (fun _ => true) p =
    PMF.pure { die := .none } := rfl
-- A record never rolled holds no dice to reroll.
example (c : Criterion) (l : Option Nat) : reroll {} c l = PMF.pure {} := by
  cases l <;> simp [reroll, rerollKept, Record.kept, Record.sorted,
    Die.facesWithin, Die.faces]

end Xdy.Spec.Test
