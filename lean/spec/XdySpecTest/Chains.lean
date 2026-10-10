import XdySpec.Chains

/-!
# Chains tests

The multisets of draws of chains, as `multisets` in `propagation/record.rs`
answers them; the slices of multisets of chains, which keep a window of the
chains sorted by `Xdy.Chain.le`, their totals first, whatever the order in
which they come; the kept pools, the transforms of pools and of stacks, and
the compositions of the laws of stacks, and the joins of stacks and of their
totals; the multisets of chains of every kind of roll; and the transforms of
laws and the rerolls of chains and of stacks.
-/

namespace Xdy.Spec.Test.Chains

open Xdy (Chain Criterion)

/-! ## Multisets -/

-- Two draws of two chains weigh their multisets by the multinomial
-- coefficients times the weights, and `2²` in all.
#guard multisets [(Chain.single 1, 1), (.live [6] 2, 1)] 2 ==
  [([(.single 1, 2)], 1), ([(.single 1, 1), (.live [6] 2, 1)], 2),
    ([(.live [6] 2, 2)], 1)]
#guard (List.range 4).all fun n =>
  ((multisets [(Chain.single 1, 2), (.live [6] 2, 3)] n).map (·.2)).sum ==
    5 ^ n
-- No draws are the empty multiset, once.
#guard multisets [(Chain.single 1, 2)] 0 == [([], 1)]

/-! ## Slices -/

-- The window counts each chain as one die worth its total: a chain of two
-- dice totalling `8` outranks a chain of one die of `6`.
#guard slice [(.live [6] 2, 1), (.single 6, 1), (.single 1, 1)] 1 1 ==
  [(.single 6, 1)]
#guard slice [(.live [6] 2, 1), (.single 6, 1), (.single 1, 1)] 0 1 ==
  [(.single 1, 1), (.single 6, 1)]
-- Chains of equal totals are ordered by their dice.
#guard slice [(.live [3] 3, 1), (.live [2] 4, 1)] 0 1 == [(.live [2] 4, 1)]
-- Copies of a chain split across the window.
#guard slice [(.single 4, 3), (.single 1, 1)] 2 0 == [(.single 4, 2)]
-- Drops that cover every chain keep none.
#guard slice [(.single 4, 3)] 2 2 == []

/-! ## Pools -/

#guard keptPool [([(.single 1, 2), (.single 4, 1)], 1),
    ([(.single 4, 3)], 2)] 0 1 ==
  [([(.single 1, 2)], 1), ([(.single 4, 2)], 2)]
-- A multiset whose kept chains hold fewer kept dice is padded to the most of
-- any: here a reroll pads by `3` for each kept die short of `2`.
#guard transformPool [([(.single 1, 1)], 1), ([(.live [1] 1, 1)], 1)] 0 0 3
    (fun ch => ([(ch, 1)], ch.dice.length)) ==
  [([(.single 1, 1)], 3), ([(.live [1] 1, 1)], 1)]
-- Each copy of a group draws independently, and the groups join.
#guard transformPool [([(.single 2, 2)], 1)] 0 0 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  [([(.single 2, 2)], 1), ([(.single 2, 1), (.live [6] 2, 1)], 2),
    ([(.live [6] 2, 2)], 1)]

/-! ## Stacks -/

-- Stacks keep their pool, and hold the law of what each distinct chain of it
-- becomes, once.
#guard transformStacks [([(.single 2, 2)], 1), ([(.single 1, 1),
    (.single 2, 1)], 2)] [] 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  .stacked [([(.single 2, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 2)]
    [(.single 1, [(.single 1, 1), (.live [6] 1, 1)]),
      (.single 2, [(.single 2, 1), (.live [6] 2, 1)])]
-- A multiset whose chains hold fewer kept dice is padded to the most of any.
#guard transformStacks [([(.single 1, 1)], 1), ([(.live [1] 1, 1)], 1)] [] 3
    (fun ch => ([(ch, 1)], ch.dice.length)) ==
  .stacked [([(.single 1, 1)], 3), ([(.live [1] 1, 1)], 1)]
    [(.single 1, [(.single 1, 1)]), (.live [1] 1, [(.live [1] 1, 1)])]

/-- A reroll-like law of what a chain becomes: itself, along one branch, with
its kept dice. -/
private def itself (ch : Chain) : List (Chain × Nat) × Nat :=
  ([(ch, 1)], ch.dice.length)

-- Composing the law of what a chain has become pads each chain that it may
-- have become to the most kept dice of any, and weighs it by its weight.
#guard compose [(.single 1, [(.single 1, 2), (.live [1] 1, 1)])] 3 itself
    (.single 1) ==
  ([(.single 1, 6), (.live [1] 1, 1)], 2)
-- A chain that is still itself becomes what it becomes.
#guard compose [(.single 1, [(.live [1] 1, 1)])] 3 itself (.single 2) ==
  itself (.single 2)
-- The composed law of each chain becomes its law in the stacks, and each
-- multiset is padded by the most kept dice that its chains may have become:
-- here the chain of one die may have become a chain of two.
#guard transformStacks [([(.single 1, 1)], 1), ([(.single 2, 1)], 1)]
    [(.single 1, [(.single 1, 1), (.live [1] 1, 1)])] 3 itself ==
  .stacked [([(.single 1, 1)], 1), ([(.single 2, 1)], 3)]
    [(.single 1, [(.single 1, 3), (.live [1] 1, 1)]),
      (.single 2, [(.single 2, 1)])]
-- Explodes compose without padding: each chain that a chain may have become
-- becomes each that it explodes to, weighed by both.
#guard transformStacks [([(.single 2, 2)], 1)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 2)])] 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  .stacked [([(.single 2, 2)], 1)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1), (.live [6] 2, 2),
      (.live [6] 2, 2)])]

/-! ## Rolls -/

-- Dice become the chains of one die of their outcomes.
#guard (Roll.custom 2 [(1, 1), (2, 2)]).stacks ==
  [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 4),
    ([(.single 2, 2)], 4)]
-- A pool becomes its multisets of results, each the chain of one die.
#guard (Roll.pool [([(1, 1)], 2), ([(3, 1)], 1)]).stacks ==
  [([(.single 1, 1)], 2), ([(.single 3, 1)], 1)]
-- Stacks whose chains are still themselves are their pool.
#guard (Roll.stacked [([(.live [6] 2, 1)], 5)] []).stacks ==
  [([(.live [6] 2, 1)], 5)]
-- No i.i.d. chains are the empty multiset.
#guard (Roll.chains (-1) [(.single 1, 1)]).stacks == [([], 1)]

/-! ## Rerolls -/

/-- A law of a chain of one die and a chain of two. -/
private def law : List (Chain × Nat) := [(.single 1, 1), (.live [2] 2, 1)]

-- A transformed law pads each chain's law to the most kept dice of any.
#guard transformLaw law 2 (rerolled (.within 1 1) 2 [(1, 1), (2, 1)]) ==
  [(.single 1, 2), (.single 2, 2), (.live [2] 2, 4)]
-- Without drops, a reroll of i.i.d. chains transforms the law of each.
#guard (Roll.chains 2 law).rerollChains 0 0 (.within 1 1) 2 [(1, 1), (2, 1)] ==
  .chains 2 [(.single 1, 2), (.single 2, 2), (.live [2] 2, 4)]
-- After a drop, it defers the reroll of each chain of the kept pool.
#guard (Roll.chains 2 law).rerollChains 1 0 (.within 1 1) 2 [(1, 1), (2, 1)] ==
  .stacked
    [([(.single 1, 1)], 2), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]
    [(.single 1, [(.single 1, 1), (.single 2, 1)]),
      (.live [2] 2, [(.live [2] 2, 4)])]
-- Without drops, a reroll of stacks composes the law of what each chain of
-- the pool has become with its reroll, padding each multiset to the most kept
-- dice that its chains may have become.
#guard (Roll.stacked [([(.single 1, 1)], 1), ([(.single 2, 1)], 2)]
    [(.single 1, [(.single 1, 1), (.live [2] 2, 1)])]).rerollChains 0 0
    (.within 1 1) 2 [(1, 1), (2, 1)] ==
  .stacked [([(.single 1, 1)], 1), ([(.single 2, 1)], 4)]
    [(.single 1, [(.single 1, 2), (.single 2, 2), (.live [2] 2, 4)]),
      (.single 2, [(.single 2, 2)])]
-- Stacks whose chains are still themselves reroll each chain.
#guard (Roll.stacked [([(.single 1, 1)], 1)] []).rerollChains 0 0
    (.within 1 1) 2 [(1, 1), (2, 1)] ==
  .stacked [([(.single 1, 1)], 1)]
    [(.single 1, [(.single 1, 1), (.single 2, 1)])]
#guard (Roll.chains 2 law).chained && !(Roll.standard 2 6).chained
#guard (Roll.stacked [([(.saturated 7, 1)], 1)] []).isSaturated
#guard !(Roll.chains 2 law).isSaturated

/-! ## Docstring examples, verbatim -/

#guard multisets [((1 : Int), 1), (2, 2)] 2 ==
  [([(1, 2)], 1), ([(1, 1), (2, 1)], 4), ([(2, 2)], 4)]

#guard group [.single 2, .single 1, .single 2] ==
  [(.single 1, 1), (.single 2, 2)]

#guard slice [(.single 3, 1), (.single 1, 2), (.single 2, 1)] 1 1 ==
  [(.single 1, 1), (.single 2, 1)]

#guard keptPool [([(.single 1, 1), (.single 4, 1)], 2)] 1 0 ==
  [([(.single 4, 1)], 2)]

-- Each kept chain of one die becomes itself or a chain of two.
#guard transformPool [([(.single 1, 1), (.single 2, 1)], 1)] 1 0 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  [([(.single 2, 1)], 1), ([(.live [6] 2, 1)], 1)]

-- A chain of one die that may have become a chain of two pads to it.
#guard compose [(.single 1, [(.single 1, 1), (.live [1] 1, 1)])] 3
    (fun ch => ([(ch, 1)], ch.dice.length)) (.single 1) ==
  ([(.single 1, 3), (.live [1] 1, 1)], 2)
#guard compose [] 3 (fun ch => ([(ch, 1)], ch.dice.length)) (.single 1) ==
  ([(.single 1, 1)], 1)

#guard transformStacks [([(.single 2, 1)], 3)] [] 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  .stacked [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])]

#guard joined [([(.single 2, 1)], 3)] [] == [([(.single 2, 1)], 3)]
#guard joined [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])] ==
  [([(.single 2, 1)], 3), ([(.live [6] 2, 1)], 3)]

-- The totals of stacks draw the total of what each copy of each chain has
-- become, without joining the chains.
#guard joinedTotals [([(.single 2, 2)], 3)] [] == [([(2, 2)], 3)]
#guard joinedTotals [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])] ==
  [([(2, 1)], 3), ([(8, 1)], 3)]
#guard joinedTotals [([(.single 1, 1), (.single 2, 1)], 1)]
    [(.single 1, [(.live [1] 1, 2)])] ==
  [([(2, 1), (2, 1)], 2)]

#guard (Roll.range 1 2).stacks ==
  [([(.single 1, 1)], 1), ([(.single 2, 1)], 1)]
#guard (Roll.chains 1 [(.live [6] 2, 1)]).stacks == [([(.live [6] 2, 1)], 1)]

#guard transformLaw [(.single 1, 1), (.live [1] 2, 1)] 3
    (fun ch => ([(ch, 1)], ch.dice.length)) ==
  [(.single 1, 3), (.live [1] 2, 1)]

#guard rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.live [1, 2] 1) ==
  ([(.live [1, 2] 1, 2), (.live [1, 2] 2, 2), (.live [2, 2] 1, 2),
    (.live [2, 2] 2, 2)], 3)
#guard rerolled (.within 1 1) 2 [(1, 1), (2, 1)] (.saturated 7) ==
  ([(.saturated 7, 1)], 0)

end Xdy.Spec.Test.Chains
