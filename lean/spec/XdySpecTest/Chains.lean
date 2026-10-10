import XdySpec.Chains

/-!
# Chains tests

The multisets of draws of chains, as `multisets` in `propagation/record.rs`
answers them; the slices of multisets of chains, which keep a window of the
chains sorted by `Xdy.Chain.le`, their totals first, whatever the order in
which they come; the kept pools, the transforms of pools and the joins of
stacks; and the multisets of chains of every kind of roll.
-/

namespace Xdy.Spec.Test.Chains

open Xdy (Chain)

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

#guard joined [([(.single 2, 1)], 3)] [] == [([(.single 2, 1)], 3)]
#guard joined [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])] ==
  [([(.single 2, 1)], 3), ([(.live [6] 2, 1)], 3)]

#guard (Roll.range 1 2).stacks ==
  [([(.single 1, 1)], 1), ([(.single 2, 1)], 1)]
#guard (Roll.chains 1 [(.live [6] 2, 1)]).stacks == [([(.live [6] 2, 1)], 1)]

end Xdy.Spec.Test.Chains
