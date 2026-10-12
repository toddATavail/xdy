import XdySpec.Bounds

/-!
# Bounds tests

The fold by which the bounds evaluator totals the shapes of chains: its
saturation at either end, its agreement with the totals of chains, its
monotonicity in each die and its copies, and the saturation of
`unboundedDice` copies. The tails and the chains that rerolls and explodes
leave; the shapes of chains, their totals, and their normalization, union,
and coarsening; the chains of a die through runs of rerolls, explodes, and
drops, mirroring examples of the Rust; and the exactness of their totals.
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
    ∀ ch ∈ six.keptChains, NormalChain ch ∧
      (ch.saturates = false → ∀ x ∈ ch.dice, x ∈ six.die.faces) := by
  intro ch hch
  have : six.keptChains = [.single 6] := by decide +kernel
  rw [this, List.mem_singleton] at hch
  subst hch
  exact ⟨⟨by simp, by decide +kernel⟩, fun _ => by decide +kernel⟩

-- An explode of a six limited to three times may leave the six grown by a
-- six and then a two, totalling `14`.
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

/-! ## What a reroll makes of a chain -/

-- A reroll of ones, once, may replace the last die of `[6, 1]` by any face,
-- and leaves the six alone.
example : RerollBecomes (.custom d6) (.within 1 1) (some 1) (.live [6] 1)
    (.live [6] 4) :=
  ⟨[6], 4, by decide, by decide, by simp⟩
example : RerollBecomes (.custom d6) (.within 1 1) (some 1) (.live [6] 1)
    (.live [6] 1) :=
  ⟨[6], 1, by decide, by decide, by simp⟩
-- A reroll limited to no times rerolls nothing.
example : ¬RerollBecomes (.custom d6) (.within 1 1) (some 0) (.live [6] 1)
    (.live [6] 4) := by
  rintro ⟨ys, y, -, hy, h⟩
  have : y = 1 := by simpa [Redraw] using hy
  simp [this] at h
-- An unlimited reroll of ones replaces each one by a face that is not one,
-- and the dice besides the last sort.
example : RerollBecomes (.custom d6) (.within 1 1) none (.live [1, 6] 6)
    (.live [5, 6] 6) := by
  have hw : (Die.custom d6).facesWithin (.within 1 1) = false := by decide
  simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
  exact ⟨[5, 6], 6, by decide, by decide, by
    rw [List.mergeSort_of_pairwise (by decide)]⟩
example : ¬RerollBecomes (.custom d6) (.within 1 1) none (.live [1] 6)
    (.live [1] 6) := by
  have hw : (Die.custom d6).facesWithin (.within 1 1) = false := by decide
  simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
  rintro ⟨ys, y, hys, -, h⟩
  obtain ⟨z, rfl⟩ : ∃ z, ys = [z] := by
    rcases hys with _ | ⟨_, hys⟩
    cases hys
    exact ⟨_, rfl⟩
  have hz : z ≠ 1 := by
    rcases hys with _ | ⟨hz, -⟩
    have hm : (Criterion.within 1 1).matches 1 = true := by decide
    simp only [Redraw, hm, ↓reduceIte] at hz
    rintro rfl
    rw [hm] at hz
    exact Bool.noConfusion hz.2
  simp only [List.mergeSort_singleton, Chain.live.injEq,
    List.cons.injEq] at h
  exact hz h.1.1.symm
-- An unlimited reroll that matches every face crosses out every chain that
-- may still grow, and leaves every saturated chain alone.
example (ch : Chain) : ¬RerollBecomes (.custom d6) (.within 1 6) none
    (.live [6] 3) ch := by
  have hw : (Die.custom d6).facesWithin (.within 1 6) = true := by decide
  simp [RerollBecomes, hw, Chain.saturates]
example : RerollBecomes (.custom d6) (.within 1 6) none (.saturated i32Max)
    (.saturated i32Max) := by
  have hw : (Die.custom d6).facesWithin (.within 1 6) = true := by decide
  simp [RerollBecomes, hw, Chain.saturates]

/-! ## Hulls -/

#guard hull {3, 1, 4} == some (1, 4)
#guard hull ∅ == none

example : InHull {1, 4} 3 := ⟨1, by decide, 4, by decide, by decide, by decide⟩

/-! ## Chain shapes -/

#guard low {3, -1, 4} == -1
#guard high {3, -1, 4} == 4
#guard low ∅ == 0 && high ∅ == 0

#guard intervalsOf [1, 2, 3, 5, 7, 8] == [(1, 3), (5, 5), (7, 8)]
#guard intervalsOf [-1, 0, 1] == [(-1, 1)]
-- Sets order by their intervals, as `ResultSet` orders them, not by their
-- results: `{1, 3}` precedes `{1, 2, 3}`, though `[1, 2, 3]` precedes
-- `[1, 3]`.
#guard decide (orderKey {1, 3} < orderKey {1, 2, 3})
#guard decide (orderKey {1, 2, 3} < orderKey {5})

/-- A run of one die, of any of some results. -/
private def one (results : Finset Int) : Segment :=
  { results, least := 1, greatest := 1 }

/-- A run of six, of one to three dice. -/
private def sixes3 : Segment := { results := {6}, least := 1, greatest := 3 }

/-- A run of minus one or minus two, of one to three dice. -/
private def negatives : Segment :=
  { results := {-2, -1}, least := 1, greatest := 3 }

/-- A run of any number of dice, each showing `-1` or `2`. -/
private def mixed : Segment := { results := {-1, 2}, least := 0, greatest := ⊤ }

/-- A run of any number of zeroes. -/
private def zeroes : Segment := { results := {0}, least := 0, greatest := ⊤ }

-- A run that may hold no die vanishes, the runs sort by their results, and
-- runs of the same results merge, their numbers of dice adding.
#guard ChainShape.new
    [one {6}, { results := {1, 2}, least := 0, greatest := 0 },
      { results := {6}, least := 0, greatest := 2 }] {1} ==
  { middles := [sixes3], last := {1} }
#guard ChainShape.new [one {5}, one {1, 2, 3}, one {1, 3}] {1} ==
  { middles := [one {1, 3}, one {1, 2, 3}, one {5}], last := {1} }
#guard ChainShape.new [one {6}, { results := {6}, least := 0, greatest := ⊤ }]
    {1} ==
  { middles := [{ results := {6}, least := 1, greatest := ⊤ }], last := {1} }

#guard copies 3 == 3
#guard copies (2 ^ 40 : ℕ) == unboundedDice
#guard copies ⊤ == unboundedDice

-- A six, then up to two more sixes, and a last die of one to five.
#guard (ChainShape.totals { middles := [sixes3], last := {1, 2, 3, 4, 5} }) ==
  (7, 23)
-- A run of negative results counts its most dice toward the least total,
-- and its fewest toward the greatest.
#guard (ChainShape.totals { middles := [negatives], last := {0} }) == (-6, -1)
-- Any number of middle dice saturates either total that they move.
#guard (ChainShape.totals { middles := [mixed], last := {1} }) ==
  (i32Min + 1, i32Max)
-- Middle dice of `0` move neither.
#guard (ChainShape.totals { middles := [zeroes], last := {1, 2} }) == (1, 2)

example : (ChainShape.totals { middles := [sixes3], last := {1, 2, 3, 4, 5} }).1
    ≤ (Chain.live [6, 6] 3).total :=
  (ChainShape.totals_le (S := { middles := [sixes3], last := {1, 2, 3, 4, 5} })
    (ch := .live [6, 6] 3) ⟨by decide, [[6, 6]], by decide, by decide⟩).1

example : (Chain.live [-2, -1, -2] 0).total ≤
    (ChainShape.totals { middles := [negatives], last := {0} }).2 :=
  (ChainShape.totals_le (S := { middles := [negatives], last := {0} })
    (ch := .live [-2, -1, -2] 0)
    ⟨by decide, [[-2, -1, -2]], by decide, by decide⟩).2

example : (ChainShape.new [one {6}, one {6}] {1, 2}).Has (.live [6, 6] 2) :=
  ChainShape.has_new ⟨by decide, [[6], [6]], by decide, by decide⟩

/-! ## Transforming and uniting shapes -/

-- A reroll maps each run and the last die alike, and the runs merge if their
-- results come to agree.
#guard (ChainShape.map { middles := [one {6}], last := {1, 6} } (· ∪ {7})) ==
  { middles := [one {6, 7}], last := {1, 6, 7} }
#guard (ChainShape.map { middles := [one {5}, one {6}], last := {1} }
    fun _ => {1, 2}) ==
  { middles := [{ results := {1, 2}, least := 2, greatest := 2 }],
    last := {1, 2} }

-- A shape within another unites into the other.
#guard (ChainShape.unite { middles := [one {6}], last := {1} }
    { middles := [sixes3], last := {1, 2} }) ==
  some { middles := [sixes3], last := {1, 2} }
-- Shapes that differ only in their last dice unite them.
#guard (ChainShape.unite { middles := [], last := {1} }
    { middles := [], last := {3} }) ==
  some { middles := [], last := {1, 3} }
-- Shapes that differ only in the numbers of dice of one run unite them.
#guard (ChainShape.unite { middles := [one {6}], last := {1} }
    { middles := [{ results := {6}, least := 3, greatest := 3 }],
      last := {1} }) ==
  some { middles := [sixes3], last := {1} }
-- Shapes that differ only in the results of one run unite them.
#guard (ChainShape.unite { middles := [one {5}], last := {1} }
    { middles := [one {6}], last := {1} }) ==
  some { middles := [one {5, 6}], last := {1} }
-- Shapes that differ in a run and their last dice, or in two runs, or in
-- their numbers of runs, do not unite.
#guard (ChainShape.unite { middles := [one {5}], last := {1} }
    { middles := [one {6}], last := {2} }) == none
#guard (ChainShape.unite { middles := [one {1}, one {5}], last := {1} }
    { middles := [one {2}, one {6}], last := {1} }) == none
#guard (ChainShape.unite { middles := [one {5}], last := {1} }
    { middles := [], last := {2} }) == none

#guard ChainShape.coarsen [{ middles := [one {6}], last := {1} },
    { middles := [{ results := {5}, least := 2, greatest := 2 }, one {6}],
      last := {2} }] ==
  { middles := [{ results := {5, 6}, least := 1, greatest := 3 }],
    last := {1, 2} }
-- Shapes without runs coarsen into one without runs.
#guard ChainShape.coarsen [{ middles := [], last := {1} },
    { middles := [], last := {4} }] == { middles := [], last := {1, 4} }

#guard swapRemove [1, 2, 3, 4] 1 == [1, 4, 3]
#guard swapRemove [1, 2, 3, 4] 3 == [1, 2, 3]
#guard swapRemove [1] 0 == []

#guard uniteShapes [{ middles := [], last := {1} },
    { middles := [], last := {3} }] ==
  ([{ middles := [], last := {1, 3} }], false)
-- A shape that unites with one may then unite with another.
#guard uniteShapes [{ middles := [one {5}], last := {1} },
    { middles := [], last := {2} }, { middles := [one {6}], last := {1} },
    { middles := [one {5, 6}], last := {3} }] ==
  ([{ middles := [], last := {2} },
    { middles := [one {5, 6}], last := {1, 3} }], false)

/-- Shapes that unite with none of the others, each of a different number of
runs. -/
private def apart (n : Nat) : List ChainShape :=
  (List.range n).map fun k =>
    { middles := List.replicate k (one {6}), last := {1} }

-- The budget holds 64 shapes, and coarsens 65 into one.
#guard (uniteShapes (apart 64)).1.length == 64
#guard !(uniteShapes (apart 64)).2
#guard uniteShapes (apart 65) ==
  ([{ middles := [{ results := {6}, least := 0, greatest := 64 }],
      last := {1} }], true)

example : (ChainShape.coarsen [{ middles := [one {6}], last := {1} },
    { middles := [], last := {2} }]).Has (.live [] 2) :=
  ChainShape.coarsen_has (S := { middles := [], last := {2} }) (by simp)
    (ch := .live [] 2) ⟨by decide, [], .nil, .nil⟩

/-! ## The chains of a die -/

/-- The chains of a die of some faces just rolled, as the bounds track them
once they split by die, under exact criteria: some criterion may match every
face exactly when one matches each. -/
private def rolledDie (F : Finset Int) (saturated : Int) : DieChains :=
  .rolled F (fun cr => decide (F ⊆ cr.possible)) {saturated}

/-- The faces of a six-sided die. -/
private def d6set : Finset Int := {1, 2, 3, 4, 5, 6}

/-- The hull of the totals of some chains, from the extremes of each shape and
the saturated values, without enumerating the totals between them, which may
span the `i32` range. -/
private def span (D : DieChains) : Option (Int × Int) :=
  hull (D.saturated ∪
    unionAll (D.live.map fun S => ({S.totals.1, S.totals.2} : Finset Int)))

-- `1D6 explode = 6 explode = 5` totals at most `17`: a six, a five, and a six.
#guard span (runBounds (rolledDie d6set i32Max)
    [.explode [.exact {6}] (some (1, 1)),
      .explode [.exact {5}] (some (1, 1))]).1 == some (1, 17)
#guard !(runBounds (rolledDie d6set i32Max)
    [.explode [.exact {6}] (some (1, 1)),
      .explode [.exact {5}] (some (1, 1))]).2
-- `1D6 explode = 6 1 times explode >= 1 ∞ times` saturates every chain.
#guard (runBounds (rolledDie d6set i32Max)
    [.explode [.exact {6}] (some (1, 1)),
      .explode [.exact (Finset.Icc 1 6)] none]).1.totals == {i32Max}
-- `D[-2, 3] explode = 3 up to 2 times reroll = -2 up to 1 times` totals from
-- `-2`: a chain that ends on `-2`, rerolled to `-2`.
#guard span (runBounds (rolledDie (Finset.Icc (-2) 3) i32Max)
    [.explode [.exact {3}] (some (2, 2)),
      .reroll [.exact {-2}] (some (1, 1))]).1 == some (-2, 9)
-- An unlimited reroll of every face crosses out every chain.
#guard (runBounds (rolledDie d6set i32Max)
    [.reroll [.exact (Finset.Icc 1 6)] none]).1.totals == ∅
-- A drop leaves the shapes alone.
#guard span (runBounds (rolledDie d6set i32Max)
    [.explode [.exact {6}] none, .drop]).1 == some (1, i32Max)
-- Over a range of limits, a limit that may be zero leaves the chains too.
#guard span (runBounds (rolledDie d6set i32Max)
    [.explode [.exact {6}] (some (0, 2))]).1 == some (1, 18)

example (r r' : Record) (h : (rolledDie d6set i32Max).Tracks (.custom d6) r)
    (hr : Reaches r [.explode (.within 6 6) (some 1)] r') :
    ∀ ch ∈ r'.keptChains, ch.total ∈ (runBounds (rolledDie d6set i32Max)
      [.explode [.exact {6}] (some (1, 1))]).1.totals :=
  DieChains.run_sound (d := .custom d6)
    (cls := [.explode (.within 6 6) (some 1)]) (by decide)
    ⟨by decide, by decide⟩
    (.cons ⟨⟨_, List.mem_singleton_self _, ⟨by decide, by decide, by decide⟩⟩,
      ⟨1, le_rfl, le_rfl, rfl⟩⟩ .nil) h hr

/-! ## Exactness of the shapes -/

/-- The bounds of `1D6 explode = 6 explode = 5`, each explode once. -/
private def sixFive : List ClauseBounds :=
  [.explode [.exact {6}] (some (1, 1)), .explode [.exact {5}] (some (1, 1))]

-- The bounds are exact for a six-sided die, and do not coarsen.
example : ∀ b ∈ sixFive, b.Exact (.custom d6) := by
  intro b hb
  simp only [sixFive, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl
  · exact ⟨by simp, fun cr hcr => ⟨.within 6 6, by
      simp only [List.mem_singleton] at hcr
      subst hcr
      exact ⟨rfl, by decide⟩⟩, by simp [Ordered]⟩
  · exact ⟨by simp, fun cr hcr => ⟨.within 5 5, by
      simp only [List.mem_singleton] at hcr
      subst hcr
      exact ⟨rfl, by decide⟩⟩, by simp [Ordered]⟩
#guard !(runBounds (.ofDie (.custom d6)) sixFive).2
#guard (DieChains.ofDie (.custom d6)).faces == d6set

-- The greatest total, `17`, is the total of a six that explodes into a five,
-- which explodes into a six.
example : ReachesChain (.custom d6) sixFive (.single 6) (.live [5, 6] 6) := by
  refine ⟨.live [6] 5, ⟨_, List.mem_singleton_self _, .within 6 6,
      ⟨rfl, by decide⟩, some 1, ⟨1, le_rfl, le_rfl, rfl⟩, ?_⟩,
    .live [5, 6] 6, ⟨_, List.mem_singleton_self _, .within 5 5,
      ⟨rfl, by decide⟩, some 1, ⟨1, le_rfl, le_rfl, rfl⟩, ?_⟩, rfl⟩
  · have hg : (Chain.single 6).grows (.within 6 6) = true := by decide
    simp only [Becomes, hg, Nat.one_pos, and_self, ↓reduceIte]
    refine ⟨([], 5), ⟨by simp, by simp, by decide, by decide, by simp⟩, ?_⟩
    simp [growChain, Chain.single, Chain.dice]
  · have hg : (Chain.live [6] 5).grows (.within 5 5) = true := by decide
    simp only [Becomes, hg, Nat.one_pos, and_self, ↓reduceIte]
    refine ⟨([], 6), ⟨by simp, by simp, by decide, by decide, by simp⟩, ?_⟩
    simp only [growChain, Chain.dice, List.append_nil]
    rw [List.mergeSort_of_pairwise (l := [5, 6]) (by decide),
      List.mergeSort_of_pairwise (l := [5, 6]) (by decide)]
#guard (Chain.live [5, 6] 6).total == 17

-- Without coarsening, the least and greatest totals are those of chains that
-- the clauses may make of a chain of one face.
example (bs : List ClauseBounds) (he : ∀ b ∈ bs, b.Exact (.custom d6))
    (hc : (runBounds (.ofDie (.custom d6)) bs).2 = false) {p : Int × Int}
    (hp : hull (runBounds (.ofDie (.custom d6)) bs).1.totals = some p) :
    ∃ x ∈ d6, ∃ ch, ReachesChain (.custom d6) bs (.single x) ch ∧
      ch.total = p.2 :=
  ((DieChains.run_exact (by decide) (by decide) he hc).hull hp).2

end Xdy.Spec.Test.Bounds
