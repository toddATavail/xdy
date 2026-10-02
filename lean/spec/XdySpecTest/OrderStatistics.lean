import XdySpec.OrderStatistics

/-!
# Order statistics tests

The program of `order_statistics` against every roll of a few dice, sorted
and summed as `Record.sum` sums them: each sum weighs the rolls that reach
it, faces that saturate fold once per face, and dice that are all dropped
sum to `0`. Then the law that the program's weights give.
-/

namespace Xdy.Spec.Test.OrderStatistics

open Xdy.Spec.OrderStatistics

/-- Every roll of `n` dice of the given faces, in order, with its weight. -/
private def rolls (die : List (Int × Nat)) : Nat → List (List Int × Nat)
  | 0 => [([], 1)]
  | n + 1 => (rolls die n).flatMap fun (xs, w) =>
    die.map fun (v, u) => (xs ++ [v], w * u)

/-- The weight of the rolls whose kept dice sum to `t`, by brute force. -/
private def brute (die : List (Int × Nat)) (n lowest highest : Nat) (t : Int) :
    Nat :=
  ((rolls die n).filter fun (xs, _) =>
    ({ results := xs, lowest, highest } : Record).sum == t).map (·.2) |>.sum

/-- Whether the program weighs every sum that some roll reaches as the rolls
do, and its weights total the die's raised to the number of dice. -/
private def agrees (die : List (Int × Nat)) (n lowest highest : Nat) : Bool :=
  let program := orderStatistics die n lowest highest
  (rolls die n).all (fun (xs, _) =>
    let t := ({ results := xs, lowest, highest } : Record).sum
    weight program t == brute die n lowest highest t) &&
  (program.map (·.2)).sum == total die ^ n

-- `4D6 drop lowest 1`, whose distribution is well known.
#guard (List.range 19).map (fun t => weight
    (orderStatistics (standardFaces 6) 4 1 0) (t : Int)) ==
  [0, 0, 0, 1, 4, 10, 21, 38, 62, 91, 122, 148, 167, 172, 160, 131, 94, 54,
    21]
#guard agrees (standardFaces 6) 4 1 0
#guard agrees (standardFaces 4) 5 1 2
-- Faces of both signs, weighed by their multiplicity.
#guard agrees [(-3, 1), (1, 2), (4, 1)] 3 1 1
#guard agrees [(-2, 1), (0, 1), (3, 2)] 4 0 2
-- Faces that saturate, which the fold clamps once per face.
#guard agrees [(i32Min, 1), (5, 2), (i32Max, 1)] 3 0 1
#guard agrees [(i32Min, 2), (i32Max, 1)] 4 1 0
-- Every die dropped, and more than every die.
#guard agrees [(1, 1), (2, 1)] 2 1 1
#guard agrees [(1, 1), (2, 1)] 2 3 0
-- One die, and one face.
#guard agrees [(7, 3)] 3 1 0
#guard agrees (standardFaces 6) 1 0 0

-- The law of `2D[1, 2, 2] drop lowest 1`: the higher die is `2` unless both
-- roll `1`, along one of the nine rolls.
example : ((rollDice 2 (weightedDie [(1, 1), (2, 2)])).map
    fun r => ({ r with lowest := 1, highest := 0 } : Record).sum) 1 =
      1 / 9 := by
  rw [map_sum_rollDice_order (by decide) (by decide),
    show weight (orderStatistics [(1, 1), (2, 2)] (2 : Int).toNat 1 0) 1 = 1
      by decide,
    show total [(1, 1), (2, 2)] = 3 by decide]
  norm_num

/-! ## Docstring examples, verbatim -/

#guard OrderStatistics.words [(1, 1), (2, 3)] 2 (fun ys => ys.sum.toNat) == 56

#guard OrderStatistics.window 1 2 [1, 3, 4, 6] == 7

-- `4D6 drop lowest 1` sums to `18` along 21 of its 1296 rolls.
#guard OrderStatistics.weight (OrderStatistics.orderStatistics
  (OrderStatistics.standardFaces 6) 4 1 0) 18 == 21

end Xdy.Spec.Test.OrderStatistics
