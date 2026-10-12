import XdySpec.Splits

/-!
# Splits tests

The outcomes of setups, which keep their drops and their die: of dice, as the
multisets of their results, and of i.i.d. chains and stacks, as the
multisets of their chains, each stacks of one multiset whose weights total
the roll's; the records that the outcomes hold; and the law that the weights
give, of dice and of chains.
-/

namespace Xdy.Spec.Test.Splits

open Xdy (Chain)

/-! ## Dice -/

-- A setup's outcomes keep its die and its drops, and a setup without a roll
-- is its own.
#guard ({ roll := some (.standard 2 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 2)]), die := some (.standard 2),
      lowest := 1 }, 1),
    ({ roll := some (.fixed [(1, 1), (2, 1)]), die := some (.standard 2),
       lowest := 1 }, 2),
    ({ roll := some (.fixed [(2, 2)]), die := some (.standard 2),
       lowest := 1 }, 1)]
#guard ({ roll := some (.fixed [(1, 1)]), die := some (.standard 2) } :
    Setup).outcomes ==
  [({ roll := some (.fixed [(1, 1)]), die := some (.standard 2) }, 1)]
#guard ({ highest := 0 } : Setup).outcomes == [({}, 1)]
#guard (({ roll := some (.standard 2 2), lowest := 1 } : Setup).outcomes.map
    (·.1.held)) ==
  [{ results := [1, 1], lowest := 1, die := .standard 2 },
    { results := [1, 2], lowest := 1, die := .standard 2 },
    { results := [2, 2], lowest := 1, die := .standard 2 }]

/-- A `1` and a `2` of two two-sided dice, the lowest dropped. -/
private def oneTwo : Xdy.Record :=
  { results := [1, 2], lowest := 1, die := .standard 2 }

-- The law of `2D2`, its lowest dropped, read sorted: `[1, 2]` along two of
-- its four rolls.
example : (({ roll := some (.standard 2 2), lowest := 1 } : Setup).law.map
    Record.toOracle) oneTwo = 2 / 4 := by
  rw [Setup.map_toOracle_law _ (fun ρ h => by cases h; trivial)
      (fun ρ h => by cases h; exact .inl trivial),
    show OrderStatistics.integral
        ({ roll := some (.standard 2 2), lowest := 1 } : Setup).outcomes
        (fun o => if o.held = oneTwo then 1 else 0) = 2 by decide,
    show ({ roll := some (.standard 2 2), lowest := 1 } : Setup).total = 4 by
      decide]
  norm_num

/-! ## Chains -/

/-- The chains of two dice that an explode of a `6` grew, as a single `1` or
a `6` and then a `2`, the lowest dropped. -/
private def twoChains : Setup :=
  { roll := some (.chains 2 [(.single 1, 1), (.live [6] 2, 1)]),
    die := some (.standard 6), lowest := 1 }

-- Its outcomes are the multisets of its chains, as stacks of one multiset,
-- with its drops and its die.
#guard twoChains.outcomes ==
  [({ roll := some (.stacked [([(.single 1, 2)], 1)] []),
      die := some (.standard 6), lowest := 1 }, 1),
    ({ roll := some (.stacked [([(.single 1, 1), (.live [6] 2, 1)], 1)] []),
       die := some (.standard 6), lowest := 1 }, 2),
    ({ roll := some (.stacked [([(.live [6] 2, 2)], 1)] []),
       die := some (.standard 6), lowest := 1 }, 1)]
-- Each holds the record of its chains, sorted, and holds no chains if every
-- chain is a single die.
#guard twoChains.outcomes.map (·.1.held) ==
  [{ results := [1, 1], lowest := 1, die := .standard 6 },
    { results := [1, 8], lowest := 1, die := .standard 6,
      chains := [.single 1, .live [6] 2] },
    { results := [8, 8], lowest := 1, die := .standard 6,
      chains := [.live [6] 2, .live [6] 2] }]
-- The weights total the roll's.
#guard (twoChains.outcomes.map (·.2)).sum == twoChains.total

-- Stacks whose chain has become another split on what it has become.
#guard (Roll.stacked [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])]).chainOutcomes ==
  [(.stacked [([(.single 2, 1)], 1)] [], 3),
    (.stacked [([(.live [6] 2, 1)], 1)] [], 3)]
#guard ((Roll.stacked [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])]).chainOutcomes.map
      (·.2)).sum ==
  (Roll.stacked [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])]).total
-- No chains split once, on no chains.
#guard (Roll.chains 0 [(.single 1, 1)]).chainOutcomes ==
  [(.stacked [([], 1)] [], 1)]

/-- The two chains are well formed. -/
private theorem twoChains_wellFormed : twoChains.WellFormed :=
  fun ρ h => by cases h; exact ⟨by decide, by decide⟩

/-- A single `1` beside a `6` and a `2`, the lowest dropped. -/
private def oneEight : Xdy.Record :=
  { results := [1, 8], lowest := 1, die := .standard 6,
    chains := [.single 1, .live [6] 2] }

-- The law of the two chains, read sorted: a single `1` beside a `6` and a
-- `2` along two of their four draws.
#guard OrderStatistics.integral twoChains.outcomes
  (fun o => if o.held = oneEight then 1 else 0) == 2
example : (twoChains.law.map Record.toOracle) oneEight =
    (OrderStatistics.integral twoChains.outcomes
      (fun o => if o.held = oneEight then 1 else 0) : ENNReal) / 4 := by
  rw [Setup.map_toOracle_law twoChains twoChains_wellFormed
      (fun ρ h => by
        cases h
        exact .inr (by
          simp [Roll.Normal, NormalChain, Chain.single, i32Min, i32Max])),
    show twoChains.total = 4 by decide]
  rfl

-- Every outcome of the two chains holds its one record, already sorted.
example : ∀ o ∈ twoChains.outcomes,
    o.1.law = PMF.pure (Record.ofOracle o.1.held) :=
  fun _ ho => (Setup.law_of_mem_outcomes twoChains_wellFormed ho).1

/-! ## Docstring examples, verbatim -/

#guard (Roll.chains 1 [(.single 1, 1), (.live [6] 2, 2)]).chainOutcomes ==
  [(.stacked [([(.single 1, 1)], 1)] [], 1),
    (.stacked [([(.live [6] 2, 1)], 1)] [], 2)]

#guard ({ roll := some (.range 1 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 1)]), die := some (.range 1 2),
      lowest := 1 }, 1),
    ({ roll := some (.fixed [(2, 1)]), die := some (.range 1 2),
       lowest := 1 }, 1)]

#guard ({ roll := some (.fixed [(1, 2), (3, 1)]), highest := 1 } : Setup).held
  == { results := [1, 1, 3], highest := 1 }
#guard ({ roll := some (.stacked [([(.live [6] 2, 1), (.single 3, 1)], 1)] []),
          lowest := 1 } : Setup).held
  == { results := [3, 8], lowest := 1, chains := [.single 3, .live [6] 2] }

end Xdy.Spec.Test.Splits
