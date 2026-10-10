import XdySpec.Stacks

/-!
# Stacks tests

The laws of the kept chains of setups after drops and keeps, as the kept pool
of their roll's multisets of chains weighs them: of dice, whose chains are of
one die, and of i.i.d. chains that an explode grew. The explodes of dice and
chains after a drop, and of fixed results and pools, which transform the
multisets of their chains still kept, and their laws, whole or reduced to
totals, and of a mixture; the explodes and rerolls of stacks none of whose
chains has been dropped, which compose the laws that they defer; and the
joins of stacks whose chains have become others, under laws that weigh
unlike, which a drop needs, and the totals to which they collapse; and the
compositions of the laws of stacks with what each chain becomes, and their
weights.
-/

namespace Xdy.Spec.Test.Stacks

open Xdy (Chain Criterion)

/-! ## Dice -/

/-- Three two-sided dice, the lowest dropped. -/
private def threeD2 : Setup := { roll := some (.standard 3 2), lowest := 1 }

/-- The results of two-sided dice lie within the `i32` range. -/
private theorem standard_normal : (Roll.standard 3 2).Normal := by
  intro r hr x hx
  have hf : x ∈ (Die.standard 2).faces := Setup.faithful_roll
    (ρ := .standard 3 2) rfl r (by rw [Setup.law_roll]; exact hr) x hx
  rw [show (Die.standard 2).faces = [1, 2] by decide] at hf
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> simp [i32Min, i32Max]

-- Dropping the lowest of three two-sided dice keeps the higher two of each
-- outcome.
#guard keptPool (Roll.standard 3 2).stacks 1 0 ==
  [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 3),
    ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]

-- Their kept chains have the law of the kept pool.
example : threeD2.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] (keptPool (Roll.standard 3 2).stacks 1 0)).map
      (fun ms => (Roll.expand ms : Multiset Chain)) :=
  Setup.map_keptChains_law rfl trivial standard_normal

/-! ## Chains -/

/-- The law of a chain of a two-sided die exploded on `2` once: a `1`, or a
`2` extended by a `2`, as the explode leaves the chains that a later reroll
reads, the extension by a `1` aside. -/
private def law : List (Chain × Nat) := [(.single 1, 1), (.live [2] 2, 1)]

/-- Two such chains, the lowest dropped. -/
private def twoChains : Setup :=
  { roll := some (.chains 2 law), die := some (.standard 2), lowest := 1 }

-- The higher chain is the chain of two dice unless both are of one die.
#guard keptPool (Roll.chains 2 law).stacks 1 0 ==
  [([(.single 1, 1)], 1), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]

/-- The chains are normal. -/
private theorem law_normal : (Roll.chains 2 law).Normal := by
  intro e he
  simp only [law, List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl <;>
    simp [NormalChain, Chain.single, i32Min, i32Max]

-- Their kept chains have the law of the kept pool.
example : twoChains.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] (keptPool (Roll.chains 2 law).stacks 1 0)).map
      (fun ms => (Roll.expand ms : Multiset Chain)) :=
  Setup.map_keptChains_law rfl ⟨by decide, by decide⟩ law_normal

/-! ## Explodes of multisets of chains -/

/-- An explode that may not draw without bound is not one that the pass
refuses. -/
private theorem explodes_of {s : Setup} {c : Criterion} {limit : Option Nat}
    (hu : Burst.resolve s.replacement c limit true ≠ .unbounded) :
    s.Explodes c limit := by
  intro ρ hρ _
  generalize Burst.resolve s.replacement c limit true = b at hu ⊢
  cases b <;> trivial

/-- Three two-sided dice, the lowest dropped, are well formed, grounded, and
have faces. -/
private theorem threeD2_invariants :
    threeD2.WellFormed ∧ threeD2.Grounded ∧ threeD2.Faced := by
  refine ⟨fun _ h => by cases h; trivial, ⟨by decide, fun ρ h => ?_⟩,
    fun _ h => by cases h; trivial⟩
  cases h
  intro r hr x hx
  simp only [Setup.law, PMF.support_map, Set.mem_image] at hr
  obtain ⟨r', hr', rfl⟩ := hr
  exact Setup.faithful_roll (ρ := .standard 3 2) rfl r'
    (by rw [Setup.law_roll]; exact hr') x hx

-- Dropping the lowest of three two-sided dice and exploding the others on `2`
-- once keeps the kept pool, as stacks whose chains have become their
-- explodes.
#guard threeD2.burst (.within 2 2) (some 1) true ==
  { roll := some (.stacked
      [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 3),
        ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]
      [(.single 1, [(.single 1, 2)]),
        (.single 2, [(.live [2] 1, 1), (.live [2] 2, 1)])]),
    die := some (.standard 2) }
-- Reduced to their totals, they leave the pool of the multisets of the totals
-- of the chains still kept, each multiset padded by the explode.
#guard threeD2.burst (.within 2 2) (some 1) false ==
  { roll := some (.pool
      [([(1, 2)], 4), ([(1, 1), (3, 1)], 6), ([(1, 1), (4, 1)], 6),
        ([(3, 2)], 3), ([(3, 1), (4, 1)], 6), ([(4, 2)], 3), ([(3, 2)], 1),
        ([(3, 1), (4, 1)], 2), ([(4, 2)], 1)]),
    die := some (.standard 2) }

-- Exploding those stacks again composes the law of what each chain has
-- become with its explode: a `1` keeps its dice along both branches, and only
-- the chain that ends in a `2` grows.
#guard (threeD2.burst (.within 2 2) (some 1) true).burst (.within 2 2)
    (some 1) true ==
  { roll := some (.stacked
      [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 3),
        ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]
      [(.single 1, [(.single 1, 4)]),
        (.single 2, [(.live [2] 1, 2), (.live [2, 2] 1, 1),
          (.live [2, 2] 2, 1)])]),
    die := some (.standard 2) }
-- Rerolling every `1` of them once pads each multiset to the most kept dice
-- that its chains may have become, four, by `2` for each that it lacks.
#guard (threeD2.burst (.within 2 2) (some 1) true).reroll (.within 1 1)
    (some 1) ==
  { roll := some (.stacked
      [([(.single 1, 2)], 4), ([(.single 1, 1), (.single 2, 1)], 6),
        ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]
      [(.single 1, [(.single 1, 2), (.single 2, 2)]),
        (.single 2, [(.live [2] 1, 2), (.live [2] 2, 2), (.live [2] 2, 4)])]),
    die := some (.standard 2) }

/-- The stacks that exploding three two-sided dice on `2` once leaves, after
the lowest is dropped, are well formed, grounded, and have faces. -/
private theorem threeD2Stacks_invariants :
    (threeD2.burst (.within 2 2) (some 1) true).WellFormed ∧
      (threeD2.burst (.within 2 2) (some 1) true).Grounded ∧
      (threeD2.burst (.within 2 2) (some 1) true).Faced := by
  obtain ⟨hs, hg, hf⟩ := threeD2_invariants
  obtain ⟨hs', hf', hg'⟩ :=
    Setup.burst_invariants (.within 2 2) (some 1) true hs hg hf
  refine ⟨hs', hg' rfl fun v h => ?_, hf'⟩
  rw [Burst.resolve_some true (by decide)] at h
  cases h

-- Exploding those stacks again, and rerolling them, whole or reduced to
-- their totals, describe the specification's explodes and rerolls.
example : (threeD2.burst (.within 2 2) (some 1) true).Bursts (.within 2 2)
    (some 1) true :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))
example : (threeD2.burst (.within 2 2) (some 1) true).Bursts (.within 2 2)
    (some 1) false :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))
example : (threeD2.burst (.within 2 2) (some 1) true).RerollsAs (.within 1 1)
    (some 1) true :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  Setup.law_reroll _ _ _ hs hg hf
example : (threeD2.burst (.within 2 2) (some 1) true).RerollsAs (.within 1 1)
    (some 1) false :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  Setup.law_reroll _ _ _ hs hg hf
-- What they leave is well formed, and grounded if the chains stay whole.
example : ((threeD2.burst (.within 2 2) (some 1) true).burst (.within 2 2)
    (some 1) false).WellFormed :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  (Setup.burst_invariants _ _ _ hs hg hf).1
example : ((threeD2.burst (.within 2 2) (some 1) true).rerollAs (.within 1 1)
    (some 1) true).Grounded :=
  let ⟨hs, hg, hf⟩ := threeD2Stacks_invariants
  (Setup.reroll_invariants _ _ _ hs hg hf).2.2 rfl

-- Both describe the specification's explode, as every later instruction
-- reads it.
example : threeD2.Bursts (.within 2 2) (some 1) true :=
  let ⟨hs, hg, hf⟩ := threeD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))
example : threeD2.Bursts (.within 2 2) (some 1) false :=
  let ⟨hs, hg, hf⟩ := threeD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))

-- The pool that they leave is well formed.
example : (threeD2.burst (.within 2 2) (some 1) false).WellFormed :=
  let ⟨hs, hg, hf⟩ := threeD2_invariants
  (Setup.burst_invariants _ _ _ hs hg hf).1

/-- Fixed results of a two-sided die. -/
private def fixedD2 : Setup :=
  { roll := some (.fixed [(1, 1), (2, 1)]), die := some (.standard 2) }

/-- Fixed results of faces of their die are well formed, grounded, and have
faces. -/
private theorem fixedD2_invariants :
    fixedD2.WellFormed ∧ fixedD2.Grounded ∧ fixedD2.Faced := by
  refine ⟨fun _ h => by cases h; exact ⟨by decide, by decide⟩,
    ⟨by decide, fun ρ h => ?_⟩, fun _ h => by cases h; trivial⟩
  cases h
  intro r hr x hx
  rw [show fixedD2.replacement.faces = [1, 2] by decide]
  simp only [fixedD2, Setup.law, Option.map_some, Option.getD_some, Roll.law,
    PMF.pure_map, PMF.mem_support_pure_iff] at hr
  subst hr
  revert x
  decide

-- The `2` grows into a `3` or a `4`; the `1` stays, along both branches.
#guard fixedD2.burst (.within 2 2) (some 1) false ==
  { roll := some (.pool [([(1, 1), (3, 1)], 2), ([(1, 1), (4, 1)], 2)]),
    die := some (.standard 2) }
example : fixedD2.Bursts (.within 2 2) (some 1) false :=
  let ⟨hs, hg, hf⟩ := fixedD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))

/-- A pool of results of a two-sided die. -/
private def poolD2 : Setup :=
  { roll := some (.pool [([(1, 2)], 1), ([(1, 1), (2, 1)], 2)]),
    die := some (.standard 2) }

/-- A pool of faces of its die is well formed, grounded, and has faces. -/
private theorem poolD2_invariants :
    poolD2.WellFormed ∧ poolD2.Grounded ∧ poolD2.Faced := by
  refine ⟨fun _ h => by
      cases h
      refine ⟨by decide, by decide, fun e he => ?_⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with rfl | rfl <;> rfl,
    ⟨by decide, fun ρ h => ?_⟩, fun _ h => by cases h; trivial⟩
  cases h
  intro r hr x hx
  rw [show poolD2.replacement.faces = [1, 2] by decide]
  simp only [poolD2, Setup.law, Option.map_some, Option.getD_some, Roll.law,
    PMF.map_comp, PMF.support_map, Set.mem_image] at hr
  obtain ⟨rs, hrs, rfl⟩ := hr
  simp only [Function.comp_apply] at hx
  rcases mem_support_weighted hrs with ⟨e, he, rfl⟩ | rfl
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at he
    rcases he with rfl | rfl <;> revert x <;> decide
  · revert x
    decide

#guard poolD2.burst (.within 2 2) (some 1) false ==
  { roll := some (.pool [([(1, 2)], 4), ([(1, 1), (3, 1)], 4),
      ([(1, 1), (4, 1)], 4)]),
    die := some (.standard 2) }
example : poolD2.Bursts (.within 2 2) (some 1) true :=
  let ⟨hs, hg, hf⟩ := poolD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))
example : poolD2.Bursts (.within 2 2) (some 1) false :=
  let ⟨hs, hg, hf⟩ := poolD2_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))

/-- The two chains, the lowest dropped, are well formed, grounded, and have
faces. -/
private theorem twoChains_invariants :
    twoChains.WellFormed ∧ twoChains.Grounded ∧ twoChains.Faced := by
  refine ⟨fun _ h => by cases h; exact ⟨by decide, by decide⟩,
    ⟨by decide, fun ρ h => ?_⟩, fun _ h => by cases h; trivial⟩
  cases h
  refine ⟨rfl, fun e he => ⟨law_normal e he, ?_⟩⟩
  rw [show twoChains.replacement.faces = [1, 2] by decide]
  -- Each chain's dice, sorted, are faces.
  have hd : ∀ (body : List Int) (last : Int), (∀ x ∈ last :: body, x ∈ [1, 2]) →
      ∀ x ∈ (Chain.live body last).dice, x ∈ [1, 2] := fun body last h x hx =>
    h x ((List.mergeSort_perm _ _).mem_iff.mp hx)
  simp only [law, List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl
  · exact ⟨rfl, hd _ _ (by decide)⟩
  · exact ⟨rfl, hd _ _ (by decide)⟩

-- Exploding the higher chain again, after the drop, extends a chain of two
-- dice into a chain of three.
#guard twoChains.burst (.within 2 2) (some 1) false ==
  { roll := some (.pool [([(1, 1)], 2), ([(5, 1)], 2), ([(6, 1)], 2),
      ([(5, 1)], 1), ([(6, 1)], 1)]),
    die := some (.standard 2) }
example : twoChains.Bursts (.within 2 2) (some 1) true :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))
example : twoChains.Bursts (.within 2 2) (some 1) false :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  Setup.law_burst _ _ _ hs hg hf (explodes_of (by decide))

/-! ## Rerolls of chains -/

-- Rerolling every `1` once leaves a chain of `2`s alone, along the two
-- branches of each of its dice.
#guard rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.live [2] 2) ==
  ([(.live [2] 2, 4)], 2)
-- The `1` of its body draws anew.
#guard rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.live [1] 2) ==
  ([(.live [1] 2, 2), (.live [2] 2, 2)], 2)

-- What a chain becomes, over its weights, is the specification's reroll of
-- the chain, and weighs the padding raised to its dice.
example : weighted (.single 0)
      (rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.live [1] 2)).1 =
    rerollChain (Criterion.within 1 1).matches (weightedDie [(1, 1), (2, 1)])
      (.live [1] 2) :=
  weighted_rerolled (by decide) (by decide) _ _
example : ((rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.live [1] 2)).1.map
      (·.2)).sum = 2 ^ (rerolled (.within 1 1) 2 [(1, 1), (2, 1)]
        (.live [1] 2)).2 :=
  sum_rerolled (by decide) _

/-- The two chains, none dropped. -/
private def twoChainsWhole : Setup := { twoChains with lowest := 0 }

/-- The two chains, none dropped, are well formed, grounded, and have
faces. -/
private theorem twoChainsWhole_invariants : twoChainsWhole.WellFormed ∧
    twoChainsWhole.Grounded ∧ twoChainsWhole.Faced :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  ⟨hs, ⟨hg.1, fun _ h => by cases h; exact hg.2 _ rfl⟩, hf⟩

-- Rerolling every `1` once, none dropped, transforms the law of each chain,
-- padding the chain of one die to the chain of two.
#guard twoChainsWhole.reroll (.within 1 1) (some 1) ==
  { roll := some (.chains 2
      [(.single 1, 2), (.single 2, 2), (.live [2] 2, 4)]),
    die := some (.standard 2) }
example : twoChainsWhole.RerollsAs (.within 1 1) (some 1) true :=
  let ⟨hs, hg, hf⟩ := twoChainsWhole_invariants
  Setup.law_reroll _ _ _ hs hg hf
example : twoChainsWhole.RerollsAs (.within 1 1) (some 1) false :=
  let ⟨hs, hg, hf⟩ := twoChainsWhole_invariants
  Setup.law_reroll _ _ _ hs hg hf

-- After the drop, the reroll leaves stacks of the kept pool, each multiset
-- padded to the most kept dice of any, and the law of what each chain has
-- become, whose weights the stacks multiply back in.
#guard twoChains.reroll (.within 1 1) (some 1) ==
  { roll := some (.stacked
      [([(.single 1, 1)], 2), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]
      [(.single 1, [(.single 1, 1), (.single 2, 1)]),
        (.live [2] 2, [(.live [2] 2, 4)])]),
    die := some (.standard 2) }
example : twoChains.RerollsAs (.within 1 1) (some 1) true :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  Setup.law_reroll _ _ _ hs hg hf
example : twoChains.RerollsAs (.within 1 1) (some 1) false :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  Setup.law_reroll _ _ _ hs hg hf
-- The stacks are well formed, and their totals too.
example : (twoChains.rerollAs (.within 1 1) (some 1) true).WellFormed :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  (Setup.reroll_invariants _ _ _ hs hg hf).1
example : (twoChains.rerollAs (.within 1 1) (some 1) false).WellFormed :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  (Setup.reroll_invariants _ _ _ hs hg hf).1

-- An unlimited reroll of every face crosses out every chain still kept.
#guard twoChains.reroll (.within 1 2) none ==
  { twoChains with lowest := 2 }
example : twoChains.RerollsAs (.within 1 2) none true :=
  let ⟨hs, hg, hf⟩ := twoChains_invariants
  Setup.law_reroll _ _ _ hs hg hf

-- Dice after a drop reroll as a pool, as before.
example : threeD2.RerollsAs (.within 1 1) (some 1) false :=
  let ⟨hs, hg, hf⟩ := threeD2_invariants
  Setup.law_reroll _ _ _ hs hg hf

/-! ## Deferred joins -/

/-- The stacks that rerolling every `1` once leaves of two chains after a
drop, the laws of whose chains weigh unlike. -/
private def rerolledStacks : Roll :=
  .stacked
    [([(.single 1, 1)], 2), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]
    [(.single 1, [(.single 1, 1), (.single 2, 1)]),
      (.live [2] 2, [(.live [2] 2, 4)])]

/-- The stacks are well formed. -/
private theorem rerolledStacks_wellFormed : rerolledStacks.WellFormed := by
  simp only [rerolledStacks, Roll.WellFormed, Roll.Even]
  decide

/-- The stacks are normal. -/
private theorem rerolledStacks_normal : rerolledStacks.Normal := by
  simp only [rerolledStacks, Roll.Normal, List.mem_cons, List.not_mem_nil,
    or_false, forall_eq_or_imp, forall_eq, Chain.single]
  simp [NormalChain, i32Min, i32Max]

-- Joining them draws what each chain has become, weighing each multiset by
-- the branches of its law.
#guard rerolledStacks.stacks ==
  [([(.single 1, 1)], 2), ([(.single 2, 1)], 2), ([(.live [2] 2, 1)], 8),
    ([(.live [2] 2, 1)], 4)]
#guard drawWeight (fun ch => (([(.single 1, [(.single 1, 1), (.single 2, 1)]),
      (Chain.live [2] 2, [(.live [2] 2, 4)])].lookup ch).getD [(ch, 1)], 0))
    [(.single 1, 2), (.live [2] 2, 1)] == 16

-- Their kept chains have the law of the joined stacks.
example : rerolledStacks.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] rerolledStacks.stacks).map
      (fun ms => (Roll.expand ms : Multiset Chain)) :=
  Roll.map_keptChains_law rerolledStacks_wellFormed rerolledStacks_normal

-- Reduced to their totals, they leave the pool of the totals of what each
-- chain has become, whose records hold the same results.
#guard rerolledStacks.collapse ==
  .pool [([(1, 1)], 2), ([(2, 1)], 2), ([(4, 1)], 8), ([(4, 1)], 4)]
example : rerolledStacks.collapse.law.map
      (fun r => (r.results : Multiset Int)) =
    rerolledStacks.law.map (fun r => (r.results : Multiset Int)) :=
  Roll.map_results_collapse rerolledStacks_wellFormed


/-- The stacks that exploding three two-sided dice on `2` once leaves after
the lowest is dropped, the lowest of whose chains is then dropped again. -/
private def explodedStacks : Setup :=
  { roll := some (.stacked
      [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 3),
        ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]
      [(.single 1, [(.single 1, 2)]),
        (.single 2, [(.live [2] 1, 1), (.live [2] 2, 1)])]),
    die := some (.standard 2), lowest := 1 }

/-- The exploded stacks are well formed and normal. -/
private theorem explodedStacks_props : ∀ ρ ∈ explodedStacks.roll,
    ρ.WellFormed ∧ ρ.Normal := by
  intro ρ hρ
  cases hρ
  refine ⟨?_, ?_⟩
  · simp only [Roll.WellFormed, Roll.Even]
    decide
  · simp only [Roll.Normal, List.mem_cons, List.not_mem_nil, or_false,
      forall_eq_or_imp, forall_eq, Chain.single]
    simp [NormalChain, i32Min, i32Max]

-- Dropping the lowest chain joins the stacks first, each chain drawn from
-- what it has become, and then keeps the higher.
#guard keptPool (explodedStacks.roll.map Roll.stacks |>.getD []) 1 0 ==
  [([(.single 1, 1)], 4), ([(.live [2] 1, 1)], 6), ([(.live [2] 2, 1)], 6),
    ([(.live [2] 1, 1)], 3), ([(.live [2] 2, 1)], 6), ([(.live [2] 2, 1)], 3),
    ([(.live [2] 1, 1)], 1), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]
example : explodedStacks.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] (keptPool (explodedStacks.roll.map Roll.stacks |>.getD [])
      1 0)).map (fun ms => (Roll.expand ms : Multiset Chain)) :=
  let ⟨hw, hn⟩ := explodedStacks_props _ rfl
  Setup.map_keptChains_law rfl hw hn

/-! ## Composed laws -/

/-- What rerolling every `1` of a chain of two-sided dice once makes of it. -/
private def rerollOnes : Chain → List (Chain × Nat) × Nat :=
  rerolled (.within 1 1) 2 [(1, 1), (2, 1)]

/-- The laws of what the chains of `rerolledStacks` have become. -/
private def rerolledLaws : List (Chain × List (Chain × Nat)) :=
  [(.single 1, [(.single 1, 1), (.single 2, 1)]),
    (.live [2] 2, [(.live [2] 2, 4)])]

/-- What rerolling every `1` once makes of a chain weighs `2` raised to its
kept dice. -/
private theorem sum_rerollOnes (ch : Chain) :
    ((rerollOnes ch).1.map (·.2)).sum = 1 * 2 ^ (rerollOnes ch).2 := by
  rw [one_mul]
  exact sum_rerolled (c := .within 1 1) (G := 2) (g := [(1, 1), (2, 1)])
    (by decide) ch

-- Rerolling them again composes the law of each chain: the chain of one die
-- that may have become a `2` weighs `2` raised to its kept dice, times the
-- `2` branches of what it had become.
#guard compose rerolledLaws 2 rerollOnes (.single 1) ==
  ([(.single 1, 1), (.single 2, 1), (.single 2, 2)], 1)
#guard compose rerolledLaws 2 rerollOnes (.live [2] 2) ==
  ([(.live [2] 2, 16)], 2)
-- A multiset of both chains gains `2` raised to their kept dice, three, from
-- the composed laws, beside the branches of the laws that they had become.
#guard stackWeight ([Chain.single 1, .live [2] 2].map fun ch =>
      (ch, (compose rerolledLaws 2 rerollOnes ch).1))
    [(.single 1, 1), (.live [2] 2, 1)] ==
  2 ^ 3 * stackWeight rerolledLaws [(.single 1, 1), (.live [2] 2, 1)]
-- Composed without padding with what weighs `2`, each copy of each chain
-- gains `2` branches.
#guard drawWeight (compose rerolledLaws 1 fun ch =>
      ([(ch.total, 1), (ch.total + 2, 1)], 0)) [(.single 1, 2)] ==
  2 ^ 2 * stackWeight rerolledLaws [(.single 1, 2)]

-- The composed law of what each chain has become is what it had become,
-- followed by its reroll.
example (ch : Chain) : weighted ch (compose rerolledLaws 2 rerollOnes ch).1 =
    (became rerolledLaws ch).bind fun x => weighted x (rerollOnes x).1 :=
  weighted_compose ch id _ 2 _ one_ne_zero two_ne_zero sum_rerollOnes
    (by decide) ch

-- Rerolling the stacks pads each multiset to make up what its composed laws
-- lack, so that the stacks weigh each multiset as the laws that they had
-- become did, and draw each chain from its composed law.
example := law_transformStacks
  [([(.single 1, 1)], 2), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]
  rerolledLaws 2 rerollOnes one_ne_zero two_ne_zero sum_rerollOnes (n := 1)
  (by decide)
example := map_coe_stackDraws_composed one_ne_zero two_ne_zero sum_rerollOnes
  (laws := rerolledLaws) (by decide) [.single 1, .live [2] 2]
  (m := [(.single 1, 1)]) (by decide)

/-! ## Mixtures -/

-- A mixture of three dice, the lowest dropped, exploded by independent
-- criteria and limits that do not draw without bound, describes the
-- specification's explode of its law.
example (κ : PMF (Criterion × Option Nat))
    (hκ : ∀ q ∈ κ.support,
      Burst.resolve (.standard 2) q.1 q.2 true ≠ .unbounded) :
    (mixtureLaw (lift (fun s q => s.burst q.1 q.2 true) (PMF.pure threeD2)
        κ)).map Record.chainView =
      ((lift Prod.mk (mixtureLaw (PMF.pure threeD2)) κ).bind fun p =>
        explode p.1 p.2.1 p.2.2).map Record.chainView :=
  mixtureLaw_burst_whole fun s hs => by
    rw [PMF.support_pure, Set.mem_singleton_iff] at hs
    subst hs
    obtain ⟨hw, hg, hf⟩ := threeD2_invariants
    exact ⟨hw, hg, hf, fun q hq => explodes_of (hκ q hq)⟩


-- A mixture of two chains, the lowest dropped, rerolled by independent
-- criteria and limits, describes the specification's reroll of its law, as
-- every later drop, keep, and sum reads it.
example (κ : PMF (Criterion × Option Nat)) :
    (mixtureLaw (lift (fun s q => s.rerollAs q.1 q.2 false)
        (PMF.pure twoChains) κ)).map Record.view =
      ((lift Prod.mk (mixtureLaw (PMF.pure twoChains)) κ).bind fun p =>
        reroll p.1 p.2.1 p.2.2).map Record.view :=
  mixtureLaw_reroll_totals fun s hs => by
    rw [PMF.support_pure, Set.mem_singleton_iff] at hs
    subst hs
    exact twoChains_invariants

end Xdy.Spec.Test.Stacks
