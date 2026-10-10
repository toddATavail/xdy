import XdySpec.Explode

/-!
# Explode tests

The ways in which an explode grows a chain, with their weights and padding,
against the Rust's extensions; the folds of their totals; the burst of an
explode against a die, and what it makes of each chain; the explode of a
setup, which does nothing, saturates, or extends independent chains; and the
laws of a setup's and a mixture's explode, whole or reduced to totals, as
every later instruction reads them, with the invariants that every explode
keeps.
-/

namespace Xdy.Spec.Test.Explode

open Xdy (Chain)

/-! ## Extensions -/

-- A die of two faces, exploded on `2` at most twice: a first `1` ends the
-- chain, padded by the extension that it never needed; a `2` extends it once
-- more.
#guard growths [(1, 1), (2, 1)] (.within 2 2) tailExtend tailClose 1 [] ==
  [(([], 1), 2), (([2], 1), 1), (([2], 2), 1)]
-- The ways weigh `Fⁿ` in all, whatever the limit and the weights of the faces.
#guard (List.range 5).all fun r =>
  ((growths [(1, 2), (3, 1), (4, 1)] (.within 3 4) tailExtend tailClose r
    []).map (·.2)).sum == 4 ^ (r + 1)
-- The extensions besides the last are kept in ascending order.
#guard growths [(5, 1), (6, 1)] (.within 5 6) tailExtend tailClose 1 [] ==
  [(([5], 5), 1), (([5], 6), 1), (([6], 5), 1), (([6], 6), 1)]

/-! ## Folds -/

#guard Fold.of (-3) 2 == { negative := -6 }
#guard Fold.of 5 2 == { positive := 10 }
#guard Fold.ofList [3, -1, 2] == { negative := -1, positive := 5 }
-- A fold saturates at most once at each end, as the ascending sum does.
#guard ((Fold.of i32Max 1).add (Fold.of 1 1)).total == i32Max
#guard (((Fold.of i32Min 1).add (Fold.of (-1) 1)).add (Fold.of 5 1)).total ==
  i32Min + 5
#guard [[i32Max, i32Min, i32Max], [i32Min, -1, 3], [7, -2, 0, i32Max],
    [], [i32Max, i32Max, i32Min, i32Min]].all fun xs =>
  (Fold.ofList xs).total == (xs.mergeSort).foldl Xdy.add 0
#guard chainFold (.live [2, 2] 5) == { positive := 9 }
#guard growChain (.single 6) ([6], 2) == .live [6, 6] 2

/-! ## Bursts -/

#guard Burst.resolve (.standard 2) (.within 2 2) (some 2) true ==
  .extend 4 (.tails [(([], 1), 2), (([2], 1), 1), (([2], 2), 1)])
#guard Burst.resolve (.standard 6) (.within 1 6) none true == .saturate i32Max
#guard Burst.resolve (.standard 6) (.within 6 6) none true == .unbounded
#guard Burst.resolve (.standard 6) (.within 7 7) (some 3) true == .nothing
#guard Burst.resolve (.standard 6) (.within 6 6) (some 0) true == .nothing

-- A chain whose end matches grows by each tail; any other keeps its dice
-- along all `Fⁿ` branches.
#guard (Burst.resolve (.standard 2) (.within 2 2) (some 2) true).chains
    (.within 2 2) (.single 2) ==
  [(.live [2] 1, 2), (.live [2, 2] 1, 1), (.live [2, 2] 2, 1)]
#guard (Burst.resolve (.standard 2) (.within 2 2) (some 2) true).chains
    (.within 2 2) (.single 1) == [(.single 1, 4)]
-- The totals of the grown chains, from their folds.
#guard (Burst.resolve (.standard 2) (.within 2 2) (some 2) false).totals
    (.within 2 2) (.single 2) == [(3, 2), (5, 1), (6, 1)]

/-! ## Records of chains -/

-- Normal chains sort alike in any order.
#guard Record.ofChains .none [.live [2] 1, .live [1] 2] ==
  Record.ofChains .none [.live [1] 2, .live [2] 1]
#guard (Record.ofChains .none [.live [6] 2, .single 4]).chainView ==
  ((([.single 4, .live [6] 2] : List Chain) : Multiset Chain), .none)

/-! ## Setups -/

/-- Two two-sided dice. -/
private def twoD2 : Setup := { roll := some (.standard 2 2) }

-- Exploding them on `2` once keeps the chains whole, or reduces them to the
-- custom dice of their totals.
#guard twoD2.burst (.within 2 2) (some 1) true ==
  { roll := some (.chains 2 [(.single 1, 2), (.live [2] 1, 1),
      (.live [2] 2, 1)]),
    die := some (.standard 2) }
#guard twoD2.burst (.within 2 2) (some 1) false ==
  { roll := some (.custom 2 [(1, 2), (3, 1), (4, 1)]),
    die := some (.standard 2) }
-- An unlimited explode that matches every face saturates both chains, which
-- keep the total weight of the roll.
#guard twoD2.burst (.within 1 2) none true ==
  { roll := some (.stacked [([(.saturated i32Max, 2)], 4)] []),
    die := some (.standard 2) }
#guard twoD2.burst (.within 1 2) none false ==
  { roll := some (.pool [([(i32Max, 2)], 4)]), die := some (.standard 2) }
-- An explode that matches no face, or is limited to no times, does nothing.
#guard twoD2.burst (.within 3 3) (some 2) true == twoD2
#guard twoD2.burst (.within 2 2) (some 0) true == twoD2

/-! ## Laws -/

/-- Two two-sided dice are well formed, grounded, and have faces. -/
private theorem twoD2_invariants :
    twoD2.WellFormed ∧ twoD2.Grounded ∧ twoD2.Faced :=
  ⟨fun _ h => by cases h; trivial,
    ⟨by decide, fun ρ h => by
      cases h
      exact Setup.faithful_roll rfl⟩,
    fun _ h => by cases h; trivial⟩

/-- An explode that may not draw without bound, of a setup whose chains are
independent and alike, none dropped, is one that the laws describe. -/
private theorem explodes_of {s : Setup} {c : Criterion} {limit : Option Nat}
    (hu : Burst.resolve s.replacement c limit true ≠ .unbounded)
    (hlike : ∀ ρ ∈ s.roll, ρ.chainLike ∧ s.lowest = 0 ∧ s.highest = 0) :
    s.Explodes c limit := by
  intro ρ hρ _
  generalize Burst.resolve s.replacement c limit true = b at hu ⊢
  cases b with
  | nothing | saturate => trivial
  | unbounded => exact absurd rfl hu
  | extend => exact hlike ρ hρ

/-- Two two-sided dice are alike, and drop nothing. -/
private theorem twoD2_alike : ∀ ρ ∈ twoD2.roll,
    ρ.chainLike ∧ twoD2.lowest = 0 ∧ twoD2.highest = 0 :=
  fun _ h => by cases h; exact ⟨rfl, rfl, rfl⟩

-- Exploding two dice on `2` once, keeping the chains whole, describes the
-- specification's explode as a later reroll or explode reads it.
example : twoD2.Bursts (.within 2 2) (some 1) true :=
  let ⟨hs, hg, hf⟩ := twoD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide) twoD2_alike)

-- Reducing them to their totals describes it as a later sum reads it.
example : twoD2.Bursts (.within 2 2) (some 3) false :=
  let ⟨hs, hg, hf⟩ := twoD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide) twoD2_alike)

-- Saturating describes it too.
example : twoD2.Bursts (.within 1 2) none true :=
  let ⟨hs, hg, hf⟩ := twoD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide) twoD2_alike)

-- An explode of the chains that an explode left describes the
-- specification's, as every explode of chains that stay whole leaves them
-- grounded.
example : (twoD2.burst (.within 2 2) (some 1) true).Bursts (.within 1 1)
    (some 2) false := by
  obtain ⟨hs, hg, hf⟩ := twoD2_invariants
  obtain ⟨hs', hf', hg'⟩ := Setup.burst_invariants (.within 2 2) (some 1) true
    hs hg hf (explodes_of (by decide) twoD2_alike)
  refine Setup.law_burst _ _ _ hs' (hg' rfl fun v h => by
    simp only [Burst.resolve] at h
    split at h <;> cases h) hf' (explodes_of (by decide) fun ρ h => ?_)
  cases h
  exact ⟨rfl, rfl, rfl⟩

-- A mixture of two dice, exploded by independent criteria and limits that do
-- not draw without bound, describes the specification's explode of the dice.
example (κ : PMF (Criterion × Option Nat))
    (hκ : ∀ q ∈ κ.support,
      Burst.resolve (.standard 2) q.1 q.2 true ≠ .unbounded) :
    (mixtureLaw (lift (fun s q => s.burst q.1 q.2 true) (PMF.pure twoD2)
        κ)).map Record.chainView =
      ((lift Prod.mk ((rollDice 2 (standardDie 2)).map fun r =>
          { r with die := .standard 2 }) κ).bind fun p =>
        explode p.1 p.2.1 p.2.2).map Record.chainView := by
  rw [mixtureLaw_burst_whole (fun s hs => by
      rw [PMF.support_pure, Set.mem_singleton_iff] at hs
      subst hs
      obtain ⟨hw, hg, hf⟩ := twoD2_invariants
      exact ⟨hw, hg, hf, fun q hq => explodes_of (hκ q hq) twoD2_alike⟩),
    mixtureLaw_pure, twoD2, Setup.law_roll]
  rfl

end Xdy.Spec.Test.Explode
