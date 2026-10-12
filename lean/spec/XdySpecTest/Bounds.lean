import XdySpec.Bounds

/-!
# Bounds tests

The fold by which the bounds evaluator totals the shapes of the chains that
an explode may grow: its saturation at either end, its agreement with the
totals of chains, its monotonicity in each die and its copies, and the
saturation of `unboundedDice` copies.
-/

namespace Xdy.Spec.Test.Bounds

open Xdy (Chain)
open Xdy.Spec.Bounds

/-! ## Folds -/

#guard fold [(6, 2), (-1, 1)] == 11
#guard fold [] == 0
#guard fold [(3, 0)] == 0
-- The negative dice saturate first, and the rest climb back from `i32Min`.
#guard fold [(i32Min, 1), (-1, 1), (1, 1)] == i32Min + 1
#guard fold [(i32Max, 2), (i32Min, 1)] == i32Max - 1
#guard fold [(i32Max, 2)] == i32Max
-- The fold of a chain's dice is its total, which folds them in ascending
-- order, whatever their order here.
#guard fold [(i32Max, 2), (i32Min, 1)] ==
  (Chain.live [i32Max, i32Max] i32Min).total
#guard fold [(1, 1), (-3, 2)] == (Chain.live [-3, 1] (-3)).total
#guard fold [(-1, unboundedDice)] == i32Min
#guard fold [(1, unboundedDice), (i32Min, 1)] == i32Max

example : fold [(6, 2), (3, 1)] = (Chain.live [6, 6] 3).total :=
  fold_eq_total (by decide)

example : fold [(1, 1), (2, 3)] ≤ fold [(1, 1), (5, 3)] :=
  fold_mono_results (by simp)

example : fold [(1, 1), (-2, 5)] ≤ fold [(1, 1), (-2, 3)] :=
  (fold_mono_copies (-2) (by decide) [(1, 1)] []).2 (by decide)

example : fold [(1, 1), (2, 3)] ≤ fold [(1, 1), (2, 5)] :=
  (fold_mono_copies 2 (by decide) [(1, 1)] []).1 (by decide)

example : fold [(1, 1), (-1, 2 ^ 40)] = fold [(1, 1), (-1, unboundedDice)] :=
  fold_unbounded (by decide) (by decide) [(1, 1)] []

example : fold [(4, 1), (2, 0), (3, 1)] = fold [(4, 1), (3, 1)] :=
  fold_zero 2 [(4, 1)] [(3, 1)]

example : fold [(4, 1), (-3, 2)] = fold [(-3, 2), (4, 1)] :=
  fold_perm (.swap _ _ [])

/-! ## Tails -/

/-- The faces of a six-sided die. -/
private def d6 : List Int := [1, 2, 3, 4, 5, 6]

-- A tail limited to three times: up to two middle sixes, and a last die that
-- does not match unless the limit is spent.
example : LimitedTail d6 (.within 6 6) 3 ([6], 2) := by decide
example : LimitedTail d6 (.within 6 6) 3 ([6, 6], 6) := by decide
example : ¬LimitedTail d6 (.within 6 6) 3 ([], 6) := by decide
example : ¬LimitedTail d6 (.within 6 6) 3 ([6, 6, 6], 1) := by decide
example : ¬LimitedTail d6 (.within 6 6) 3 ([5], 1) := by decide

example : ([6, 6], 3) ∈ (extension (.within 6 6) (customDie d6) 3).support :=
  (mem_support_extension_iff (by decide) (by decide)).mpr (by decide)

example : ([6, 6], 6) ∉ (extension (.within 6 6) (customDie d6) 2).support :=
  fun h => absurd ((mem_support_extension_iff (by decide) (by decide)).mp h)
    (by decide)

-- Without limit, any number of middle sixes, and a last die that does not
-- match.
example : UnlimitedTail d6 (.within 6 6) ([6, 6, 6, 6], 1) := by decide
example : ¬UnlimitedTail d6 (.within 6 6) ([6], 6) := by decide

example : ([6, 6, 6], 2) ∈
    (unlimitedExtension (.within 6 6) (.custom d6) (by decide)).support :=
  (mem_support_unlimitedExtension_iff _).mpr (by decide)

/-! ## What an explode makes of a chain -/

/-- A six-sided die that shows `6`. -/
private def six : Record := { results := [6], die := .custom d6 }

private theorem six_faces :
    ∀ x ∈ six.die.faces, i32Min ≤ x ∧ x ≤ i32Max := by decide

private theorem six_kept :
    ∀ ch ∈ six.keptChains, NormalChain ch ∧ ∀ x ∈ ch.dice, x ∈ six.die.faces := by
  intro ch hch
  have : six.keptChains = [.single 6] := by decide +kernel
  rw [this, List.mem_singleton] at hch
  subst hch
  exact ⟨⟨by simp, by decide +kernel⟩, by decide +kernel⟩

-- An explode of a six limited to three times may leave the six grown by a
-- six and then a two, totalling `14`, which lies in none of the shapes that
-- `Explosion.totals` folds, but within their hull.
example : ∃ r' ∈ (explode six (.within 6 6) (some 3)).support,
    growChain (.single 6) ([6], 2) ∈ r'.keptChains := by
  refine (mem_explode_iff six_faces six_kept _).mpr
    ⟨.single 6, by decide +kernel, ?_⟩
  have hg : (Chain.single 6).grows (.within 6 6) = true := by decide +kernel
  simp only [Becomes, hg, Nat.ofNat_pos, and_self, ↓reduceIte]
  exact ⟨([6], 2), by decide, rfl⟩

#guard (growChain (.single 6) ([6], 2)).total == 14

-- An explode limited to no times leaves the six alone.
example : ∀ r' ∈ (explode six (.within 6 6) (some 0)).support,
    ∀ ch ∈ r'.keptChains, ch = .single 6 := by
  intro r' hr' ch hch
  obtain ⟨ch₀, hch₀, hb⟩ := (mem_explode_iff six_faces six_kept ch).mp
    ⟨r', hr', hch⟩
  have : six.keptChains = [.single 6] := by decide +kernel
  rw [this, List.mem_singleton] at hch₀
  subst hch₀
  simpa [Becomes] using hb

/-! ## Shapes -/

#guard shapeFold 6 6 2 (some 1) == 19
#guard shapeFold 3 6 0 none == 3
#guard shapeFold 6 (-1) unboundedDice (some 1) == i32Min + 7
#guard shapeFold 6 1 unboundedDice none == i32Max

example : (Chain.single 4).total = shapeFold 4 9 0 none :=
  total_single_eq 4 9

example : (growChain (.single 6) ([6], 2)).total =
    fold [(6, 1), (2, 1), (6, 1)] :=
  total_growChain_single 6 ([6], 2)

-- The chains of a six that end after one extension and that spend a limit of
-- three bracket those that end after two.
example : Min.min (shapeFold 6 6 0 (some 1)) (shapeFold 6 6 2 (some 1)) ≤
    shapeFold 6 6 1 (some 1) :=
  (shapeFold_short 6 6 (by decide) le_rfl (by decide : (1 : Int) ≤ 6)).1

example : shapeFold 6 6 1 (some 5) ≤
    Max.max (shapeFold 6 6 0 (some 5)) (shapeFold 6 6 2 (some 6)) :=
  (shapeFold_short 6 6 (by decide) (by decide : (1 : Int) ≤ 5)
    (by decide)).2

example : Min.min (shapeFold 1 (-2) 0 none) (shapeFold 1 (-2) unboundedDice none)
    ≤ shapeFold 1 (-2) (2 ^ 40) none :=
  (shapeFold_unbounded 1 (-2) _ none).1

/-! ## Hulls -/

#guard hull {3, 1, 4} == some (1, 4)
#guard hull ∅ == none

example : InHull {1, 4} 3 := ⟨1, by decide, 4, by decide, by decide, by decide⟩

/-! ## Explosions -/

/-- A six that explodes on six, at most three times. -/
private def sixes : Explosion :=
  { results := {6}, faces := {1, 2, 3, 4, 5, 6}, possible := {6},
    certain := {6}, saturated := ∅, limit := some (3, 3) }

-- The chain that ends after one extension, and the chain that spends the
-- limit; the chains that end after two extensions, `13` to `17`, lie in no
-- shape, but within the hull.
#guard sixes.shapes (1, 6) == [(0, some (1, 5)), (2, some (1, 6)), (2, some (1, 6))]
#guard sixes.totals == {7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24}
#guard 13 ∉ sixes.totals
#guard hull sixes.totals == some (7, 24)

-- A d6 of any result that explodes on six, at most three times: a result
-- that does not match stays alone.
#guard ({ sixes with results := {1, 2, 3, 4, 5, 6} } : Explosion).totals ==
  {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24}

-- A limit that may be `0` leaves a matching die alone, too, and a limit of
-- `1` spends itself after one extension, on any face.
#guard ({ sixes with limit := some (0, 1) } : Explosion).totals ==
  {6, 7, 8, 9, 10, 11, 12}

-- Without limit, the chains run from `7` up to saturation.
#guard hull ({ sixes with limit := none } : Explosion).totals ==
  some (7, i32Max)

-- Without limit, a die whose every face matches saturates.
#guard ({ sixes with faces := {6}, saturated := {i32Max}, limit := none } :
    Explosion).totals == {i32Max}

-- Nothing matches: every result stays alone.
#guard ({ sixes with results := {1, 2}, possible := ∅, certain := ∅ } :
    Explosion).totals == {1, 2}

-- The pair of a six-sided die and the criterion of sixes.
#guard (pair {6} (.custom d6) {6} (some (3, 3))).totals == sixes.totals
#guard (pair {1, 6} (.custom d6) {1, 2, 3, 4, 5, 6} none).saturated ==
  {(Die.custom d6).saturatedValue}
#guard explodePairs {6} [.custom d6, .custom [2, 4, 6]] [{6}] (some (1, 1)) ==
  {7, 8, 9, 10, 11, 12}

-- The hull of the totals of the sixes covers every chain that the explode may
-- grow from a six, and its least and greatest are some such chain's.
example : Exact sixes.totals (reach (.custom d6) (.within 6 6) {6} (some (3, 3))) := by
  have := pair_exact (R := {6}) (C := {6}) (d := .custom d6) (c := .within 6 6)
    (limit := some (3, 3)) (by decide) (by decide) (by simp)
  have h₁ : (pair {6} (.custom d6) {6} (some (3, 3))).totals = sixes.totals := by
    decide
  have h₂ : ({6} : Finset Int) ∩ (Die.custom d6).faces.toFinset = {6} := by
    decide
  rwa [h₁, h₂] at this

-- So `24`, the greatest total, is the total of some chain.
example : 24 ∈ reach (.custom d6) (.within 6 6) {6} (some (3, 3)) := by
  have := pair_exact (R := {6}) (C := {6}) (d := .custom d6) (c := .within 6 6)
    (limit := some (3, 3)) (by decide) (by decide) (by simp)
  have h₁ : hull (pair {6} (.custom d6) {6} (some (3, 3))).totals =
      some (7, 24) := by decide
  have h₂ : ({6} : Finset Int) ∩ (Die.custom d6).faces.toFinset = {6} := by
    decide
  have := (this.hull h₁).2
  rwa [h₂] at this

-- The six of a record, exploded on six at most three times, totals within the
-- hull of the sixes.
example : ∀ r' ∈ (explode six (.within 6 6) (some 3)).support,
    ∀ ch ∈ r'.keptChains, InHull sixes.totals ch.total := by
  have hk : six.keptChains = [.single 6] := by decide +kernel
  refine explode_sound (fun y hy => ⟨six_faces y hy, ?_⟩) (fun ch hch => ?_)
    (fun y _ hm => ?_) (fun y _ h => ?_) (fun h => by cases h)
    ⟨3, le_rfl, le_rfl, rfl⟩
  · have : y ∈ d6 := hy
    simp only [sixes]
    fin_cases this <;> decide
  · rw [hk, List.mem_singleton] at hch
    exact ⟨6, by decide, by decide +kernel, hch⟩
  · have : y ∈ d6 := ‹y ∈ six.die.faces›
    fin_cases this <;> simp_all [Criterion.matches, sixes]
  · simp only [sixes, Finset.mem_singleton] at h
    subst h
    decide

end Xdy.Spec.Test.Bounds
