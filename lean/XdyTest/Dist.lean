import Xdy.Dist

/-!
# Finite distribution tests

Construction, merging of equal outcomes, `bind` over continuations with equal
and unequal totals, and reduction to lowest terms.
-/

namespace Xdy.Test

open Xdy

/-- The distribution of one die with faces `1` through `n`. -/
private def die (n : Nat) : Dist Nat := Dist.uniform (List.range' 1 n)

/-- The distribution of the sum of two dice with faces `1` through `n`. -/
private def twoDice (n : Nat) : Dist Nat :=
  (die n).bind fun a => (die n).map (a + ·)

/-! ## Construction -/

#guard (Dist.pure 7).toList == [(7, 1)]
#guard (Dist.pure 7).weight 8 == 0
#guard (Dist.uniform [2, 1, 2]).toList == [(1, 1), (2, 2)]
#guard (Dist.uniform ([] : List Nat)).total == 0
#guard (die 6).total == 6

/-! ## Equality is by weight, not by insertion order or probability -/

#guard Dist.uniform [1, 2, 3] == Dist.uniform [3, 1, 2]
#guard Dist.uniform [1, 2] != Dist.uniform [1, 2, 3]
#guard Dist.uniform [1, 2] != Dist.uniform [1, 1, 2, 2]

/-! ## `map` merges outcomes that become equal -/

#guard ((die 6).map (· % 3)).toList == [(0, 2), (1, 2), (2, 2)]
#guard ((die 6).map fun _ => 0) == Dist.uniform [0, 0, 0, 0, 0, 0]

/-! ## `bind` with equal totals -/

#guard (twoDice 2).toList == [(2, 1), (3, 2), (4, 1)]
#guard (twoDice 6).toList ==
  [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6),
   (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)]
-- Binding into `pure` changes nothing.
#guard (die 6).bind Dist.pure == die 6
-- Binding `pure` applies the continuation.
#guard (Dist.pure 3).bind die == die 3

/-! ## `bind` with unequal totals scales to the least common multiple -/

-- `1D(1D2)`: the continuations have totals 1 and 2, so the result is over 4.
#guard ((die 2).bind die).toList == [(1, 3), (2, 1)]
-- `1D(1D3)`: totals 1, 2 and 3, least common multiple 6, so over 18.
#guard ((die 3).bind die).toList == [(1, 11), (2, 5), (3, 2)]
#guard ((die 3).bind die).total == 18
-- The least common multiple, not the product: totals 2 and 4 scale to 4, so
-- the result is over 2 × 4 = 8 rather than 2 × 8 = 16.
#guard ((Dist.uniform [2, 4]).bind die).total == 8

/-! ## Lowest terms -/

#guard (Dist.uniform [1, 1, 2, 2]).lowestTerms == Dist.uniform [1, 2]
#guard (Dist.uniform [1, 1, 2]).lowestTerms == Dist.uniform [1, 1, 2]
-- Scaling can leave a common factor, which lowest terms removes.
#guard ((Dist.uniform [2, 2]).bind die).total == 4
#guard ((Dist.uniform [2, 2]).bind die).lowestTerms == die 2
#guard (Dist.uniform ([] : List Nat)).lowestTerms.total == 0

/-! ## Docstring examples, verbatim -/

#guard (Dist.uniform [1, 2, 3]).total == 3
#guard ((Dist.pure 7).add 7 2).weight 7 == 3
#guard ((Dist.pure 7).add 8 2).total == 3
#guard (Dist.pure 7).weight 7 == 1
#guard (Dist.pure 7).total == 1
#guard (Dist.uniform [1, 2, 2]).weight 2 == 2
#guard (Dist.uniform [1, 2, 2]).total == 3
#guard ((Dist.uniform [1, 2, 3]).map (· % 2)).weight 1 == 2
#guard ((Dist.uniform [1, 2]).bind fun n => Dist.uniform (List.range' 1 n))
  == Dist.uniform [1, 1, 1, 2]
#guard (Dist.uniform [1, 1, 2, 2]).lowestTerms == Dist.uniform [1, 2]
#guard (Dist.uniform [3, 1, 3]).toList == [(1, 1), (3, 2)]

end Xdy.Test
