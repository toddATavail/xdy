import XdySpec.Mixture

/-!
# Mixture tests

The number of results of each roll, the clamped drops of a setup, and the
laws of mixtures: a custom die of weighted faces, a roll of a random count of
dice, a drop of a random count, and the sum of a record never rolled.
-/

namespace Xdy.Spec.Test.Mixture

/-! ## Rolls and setups -/

#guard Roll.expand [(1, 2), (4, 1)] == [1, 1, 4]
#guard (Roll.range 3 1).results == 1
#guard (Roll.standard 3 6).results == 3
#guard (Roll.standard (-2) 6).results == 0
#guard (Roll.custom 2 []).results == 2
#guard (Roll.fixed [(1, 2), (4, 1)]).results == 3

/-- Two six-sided dice. -/
private def twoD6 : Setup := { roll := some (.standard 2 6) }

-- Drops accumulate, and clamp to the number of results.
#guard (twoD6.dropLowest 1).lowest == 1
#guard ((twoD6.dropLowest 1).dropLowest 5).lowest == 2
#guard (twoD6.dropHighest (-1)).highest == 0
-- A record never rolled has no results to drop.
#guard (({} : Setup).dropLowest 3).lowest == 0

/-! ## Laws -/

-- A die of weighted faces is the custom die of the faces repeated.
example : weightedDie [(1, 2), (2, 1)] = customDie [1, 1, 2] :=
  weightedDie_eq_customDie (by decide)

-- Weights need not follow the order of the faces.
example : weightedDie [(2, 1), (1, 2)] = customDie [1, 2, 1] :=
  weightedDie_eq_customDie (by decide)

-- Dropping the lowest of two dice describes the setup that drops it.
example : twoD6.law.map (·.dropLowest 1) =
    ({ roll := some (.standard 2 6), lowest := 1 } : Setup).law := by
  rw [Setup.law_dropLowest]
  rfl

-- A roll of a random count of dice mixes the dice of each count.
example (π : PMF Int) :
    mixtureLaw (π.map fun c => { roll := some (.standard c 6) }) =
      π.bind fun c => rollDice c (standardDie 6) :=
  mixtureLaw_rolls _ _

-- Dropping a random count of the lowest of three dice is lemma 1's lift of
-- the drop over the dice and the count.
example (κ : PMF Int) :
    mixtureLaw (lift Setup.dropLowest
      (PMF.pure { roll := some (.standard 3 6) }) κ) =
      lift Record.dropLowest (rollDice 3 (standardDie 6)) κ := by
  rw [mixtureLaw_dropLowest, mixtureLaw_pure, Setup.law_roll]
  rfl

-- So is dropping the highest.
example (κ : PMF Int) :
    mixtureLaw (lift Setup.dropHighest
      (PMF.pure { roll := some (.standard 3 6) }) κ) =
      lift Record.dropHighest (rollDice 3 (standardDie 6)) κ := by
  rw [mixtureLaw_dropHighest, mixtureLaw_pure, Setup.law_roll]
  rfl

-- A record never rolled is empty, and sums to `0`.
example : mixtureLaw (PMF.pure {}) = PMF.pure {} := by
  rw [mixtureLaw_pure, Setup.law_empty]

example : ({} : Setup).law.map Record.sum = PMF.pure 0 :=
  Setup.map_sum_law_of_none rfl

-- The sum of a mixture mixes the sums of its setups.
example (μ : PMF Setup) :
    (mixtureLaw μ).map Record.sum = μ.bind fun s => s.law.map Record.sum :=
  map_sum_mixtureLaw μ

/-! ## Sums of setups -/

/-- Fixed results across both signs, with repeats: `-3, -3, 0, 2, 2, 2`. -/
private def fixed : List (Int × Nat) := [(-3, 2), (0, 1), (2, 3)]

-- `keptSum` adds each result's kept copies with one clamp, as folding the
-- kept results one at a time does, for every pair of drops.
#guard (List.range 7).all fun lo => (List.range 7).all fun hi =>
  keptSum fixed lo hi ==
    ({ results := Roll.expand fixed, lowest := lo, highest := hi } :
      Record).sum
-- Copies that saturate do so once.
#guard keptSum [(i32Min, 2), (i32Max, 1)] 0 0 == -1
#guard ({ results := Roll.expand [(i32Min, 2), (i32Max, 1)] } : Record).sum
  == -1

example : ({ roll := some (.fixed fixed), lowest := 1, highest := 2 } :
    Setup).law.map Record.sum = PMF.pure (keptSum fixed 1 2) :=
  Setup.map_sum_law_fixed _ _ _ (by decide)

-- A range that drops its result, or is empty, sums to `0`.
example : ({ roll := some (.range 1 6), lowest := 1 } : Setup).law.map
    Record.sum = PMF.pure 0 :=
  Setup.map_sum_law_range_dropped _ _ _ _ (.inl (by decide))

example : ({ roll := some (.range 6 1) } : Setup).law.map Record.sum =
    PMF.pure 0 :=
  Setup.map_sum_law_range_empty _ _ (by decide)

-- A range sums uniformly over its values.
example : ({ roll := some (.range 1 6) } : Setup).law.map Record.sum =
    PMF.ofMultiset (Finset.Icc 1 6).val (by simp) :=
  Setup.map_sum_law_range (by decide) (by decide) (by decide)

-- No dice, dice without faces, and dice all dropped sum to `0`.
example : ({ roll := some (.standard (-1) 6) } : Setup).law.map Record.sum =
    PMF.pure 0 :=
  Setup.map_sum_law_standard_eq_zero _ _ (.inl (by decide))

example : ({ roll := some (.standard 3 0) } : Setup).law.map Record.sum =
    PMF.pure 0 :=
  Setup.map_sum_law_standard_eq_zero _ _ (.inr (.inl (by decide)))

/-- Three dice of faces `1` and `5, 5`, of which two are dropped low and one
high. -/
private def allDropped : Setup :=
  { roll := some (.custom 3 [(1, 1), (5, 2)]), lowest := 2, highest := 1 }

example : allDropped.law.map Record.sum = PMF.pure 0 :=
  Setup.map_sum_law_custom_eq_zero _ _ (.inr (.inr (by decide)))

example : ({ roll := some (.custom 3 []) } : Setup).law.map Record.sum =
    PMF.pure 0 :=
  Setup.map_sum_law_custom_eq_zero _ _ (.inr (.inl rfl))

/-! ## Ranges -/

/-- Two ranges: `[1, 3]`, each of weight `2`, and `[2, 5]`, each of weight
`1`. -/
private def spans : List Span := [⟨1, 3, 2⟩, ⟨2, 5, 1⟩]

#guard Span.rise spans 1 == 2 && Span.rise spans 2 == 1
#guard Span.fall spans 4 == 2 && Span.fall spans 6 == 1
-- The sweep over the keys gives every outcome the weight of its ranges.
#guard ([-1, 0, 1, 2, 3, 4, 5, 6] : List Int).all fun v =>
  Span.sweep spans 0 [1, 2, 4, 6] v == Span.cover spans v
-- Breaks that are no key change nothing.
#guard ([-1, 0, 1, 2, 3, 4, 5, 6] : List Int).all fun v =>
  Span.sweep spans 0 [0, 1, 2, 3, 4, 6, 9] v == Span.cover spans v
-- A range of one outcome rises and falls at consecutive breaks.
#guard Span.sweep [⟨7, 7, 3⟩] 0 [7, 8] 7 == 3

example (v : Int) : Span.sweep spans 0 [1, 2, 4, 6] v = Span.cover spans v :=
  Span.sweep_eq_cover (by decide) (by decide) (by decide) (by decide) v

/-! ## Docstring examples, verbatim -/

#guard keptSum [(1, 2), (4, 1)] 1 0 == 5
#guard keptSum [(1, 1), (i32Max, 2)] 0 0 == i32Max

example : weightedDie [] = PMF.pure 0 := by simp [weightedDie, customDie]

#guard Roll.expand [(1, 2), (4, 1)] == [1, 1, 4]

#guard (Roll.standard (-2) 6).results == 0
#guard (Roll.fixed [(1, 2), (4, 1)]).results == 3

#guard ([-1, 0, 1, 2, 3, 4, 5, 6] : List Int).map
    (Span.sweep [⟨1, 3, 2⟩, ⟨2, 5, 1⟩] 0 [1, 2, 4, 6])
  == [0, 0, 2, 3, 3, 1, 1, 0]

end Xdy.Spec.Test.Mixture
