import XdySpec.Semantics

/-!
# Semantics tests

The laws of ranges, dice, rerolls and explodes, including their degenerate
cases, proved of the definitions, since a `PMF` cannot be computed.
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
  simp only [reroll, rerollEach, List.isEmpty_nil, ↓reduceIte, rerollKept,
    kept_sorted (by decide : ([1] : List Int).Pairwise (· ≤ ·)), hf,
    List.foldl, limitedReplacement]
  simp [Criterion.matches, PMF.pure_bind]
  rfl
-- A result that does not match is kept, and the drops are forgotten.
example : reroll { results := [2], die := .standard 2 } (.within 1 1)
    (some 3) = PMF.pure { results := [2], die := .standard 2 } := by
  simp only [reroll, rerollEach, List.isEmpty_nil, ↓reduceIte, rerollKept,
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
  simp only [reroll, rerollEach, List.isEmpty_nil, ↓reduceIte,
    kept_sorted (by decide : ([1] : List Int).Pairwise (· ≤ ·)), hw,
    Bool.false_eq_true, ↓reduceDIte, rerollKept]
  simp only [Criterion.matches, hf, List.foldl]
  simp [h2, PMF.pure_map]
  rfl
-- `2D1 reroll = 1 ∞ times` crosses out both dice, since every face matches.
example : reroll { results := [1, 1], die := .standard 1 } (.within 1 1)
    none = PMF.pure { die := .standard 1 } := by
  have hw : (Die.standard 1).facesWithin (.within 1 1) = true := by decide
  simp only [reroll, List.isEmpty_nil, ↓reduceIte,
    kept_sorted (by decide : ([1, 1] : List Int).Pairwise (· ≤ ·)), hw,
    ↓reduceDIte]
  simp [Criterion.matches]

/-! ## Explodes -/

-- An explode limited to no times grows nothing, but leaves the chains still
-- kept, as single dice.
example : explode { results := [2], die := .standard 2 } (.within 2 2)
    (some 0) = PMF.pure (Record.ofChains (.standard 2) [.single 2]) := by
  simp [explode, explodeChains, Record.keptChains,
    kept_sorted (by decide : ([2] : List Int).Pairwise (· ≤ ·)), PMF.pure_map]
-- An unlimited explode that matches every face saturates every chain, at the
-- sign of the mean face.
example : explode { results := [1, 1], die := .standard 1 } (.within 1 1) none
    = PMF.pure (Record.ofChains (.standard 1)
      [.saturated i32Max, .saturated i32Max]) := by
  have hw : (Die.standard 1).facesWithin (.within 1 1) = true := by decide
  simp only [explode, hw, ↓reduceDIte, Record.keptChains, List.isEmpty_nil,
    ↓reduceIte, kept_sorted (by decide : ([1, 1] : List Int).Pairwise (· ≤ ·))]
  rfl
-- The number of matching draws of an unlimited explode is geometric: none
-- with the probability that a draw does not match, and one more each time
-- with the probability that it does.
example : geometric 2⁻¹ (by norm_num) 0 = 2⁻¹ := by
  rw [geometric_apply, pow_zero, one_mul, ENNReal.one_sub_inv_two]
example (q : ENNReal) (hq : q < 1) (k : Nat) :
    geometric q hq (k + 1) = q * geometric q hq k := by
  simp only [geometric_apply]
  ring
-- A draw of `1D6` matches `6` with probability `1/6`.
example : matchedFaces (.standard 6) (.within 6 6) = 1 / 6 := by
  have h : (Die.standard 6).faces = [1, 2, 3, 4, 5, 6] := by decide
  simp [matchedFaces, h, Criterion.matches]
/-- A state whose rolling record `0` holds a `6` of `1D6`. -/
private def six : State :=
  { registers := #[], records := #[{ results := [6], die := .standard 6 }] }

-- Only an unlimited explode that does not saturate, but grows a chain still
-- kept, may draw without bound.
#guard unbounded six (.explode 0 (.eq (.immediate 6)) .unlimited)
#guard !unbounded six (.explode 0 (.ge (.immediate 1)) .unlimited)
#guard !unbounded six (.explode 0 (.eq (.immediate 5)) .unlimited)
#guard !unbounded six (.explode 0 (.eq (.immediate 6)) (.times (.immediate 9)))
#guard !unbounded six (.reroll 0 (.eq (.immediate 6)) .unlimited)

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
example (p : PMF Int) : rerollChain (fun _ => true) p (.saturated 0) =
    PMF.pure (.saturated 0) := rfl
example (p : PMF Int) : rerollChains .none [] (fun _ => true) p =
    PMF.pure (Record.ofChains .none []) := by
  simp [rerollChains, PMF.pure_map]
example (rerolled : Int → Bool) (p : PMF Int) :
    rerollEach {} rerolled p = PMF.pure {} := by
  simp [rerollEach, rerollKept, Record.kept, Record.sorted]
example (c : Criterion) (die : PMF Int) :
    extension c die 1 = die.map ([], ·) := rfl
example : geometric 0 (by norm_num) 0 = 1 := by simp [geometric_apply]
example (p : PMF (List Int × Int)) :
    explodeChains .none [] (fun _ => true) p =
      PMF.pure (Record.ofChains .none []) := by
  simp [explodeChains, PMF.pure_map]
-- A record never rolled holds no dice to explode.
example (c : Criterion) (l : Option Nat) :
    explode {} c l = PMF.pure (Record.ofChains .none []) := by
  cases l <;> simp [explode, explodeChains, Record.keptChains,
    Record.kept, Record.sorted, Die.facesWithin, Die.faces, PMF.pure_map]
-- A record never rolled holds no dice to reroll.
example (c : Criterion) (l : Option Nat) : reroll {} c l = PMF.pure {} := by
  cases l <;> simp [reroll, rerollEach, rerollKept, Record.kept,
    Record.sorted, Die.facesWithin, Die.faces]

end Xdy.Spec.Test
