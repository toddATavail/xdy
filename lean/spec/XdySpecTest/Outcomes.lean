import XdySpec.Outcomes

/-!
# Outcomes tests

The program of `Roll::outcomes` against every roll of a few dice, sorted as
`Record.sorted` sorts them: each multiset weighs the rolls that sort to it,
the multisets are distinct, ascend strictly, and total the die's weight
raised to the number of dice. Then the other branches of `Roll.outcomes`,
the outcomes of setups, which keep their drops, and the law that the
weights give.
-/

namespace Xdy.Spec.Test.Outcomes

open Xdy.Spec.OrderStatistics (standardFaces)

/-- Every roll of `n` dice of the given faces, in order, with its weight. -/
private def rolls (die : List (Int × Nat)) : Nat → List (List Int × Nat)
  | 0 => [([], 1)]
  | n + 1 => (rolls die n).flatMap fun (xs, w) =>
    die.map fun (v, u) => (xs ++ [v], w * u)

/-- The weight of the rolls that sort to `ys`, by brute force. -/
private def brute (die : List (Int × Nat)) (n : Nat) (ys : List Int) : Nat :=
  ((rolls die n).filter fun (xs, _) =>
    ({ results := xs } : Record).sorted == ys).map (·.2) |>.sum

/-- The weight of the outcomes that hold `ys`. -/
private def weight (outcomes : List (List (Int × Nat) × Nat)) (ys : List Int) :
    Nat :=
  (outcomes.filter fun (rs, _) => Roll.expand rs == ys).map (·.2) |>.sum

/-- Whether a multiset ascends strictly and holds each result at least
once. -/
private def ascends (rs : List (Int × Nat)) : Bool :=
  (rs.zip rs.tail).all (fun (a, b) => a.1 < b.1) && rs.all (0 < ·.2)

/-- Whether the outcomes of `n` dice weigh every multiset that some roll
sorts to as the rolls do, are distinct and well formed, and total the die's
weight raised to the number of dice. -/
private def agrees (die : List (Int × Nat)) (n : Nat) : Bool :=
  let outcomes := Roll.diceOutcomes n die
  (rolls die n).all (fun (xs, _) =>
    let ys := ({ results := xs } : Record).sorted
    weight outcomes ys == brute die n ys) &&
  (outcomes.map (Roll.expand ·.1)).Nodup &&
  outcomes.all (ascends ·.1) &&
  (outcomes.map (·.2)).sum == OrderStatistics.total die ^ n

#guard agrees (standardFaces 6) 4
#guard agrees (standardFaces 3) 5
-- Faces of both signs, weighed by their multiplicity.
#guard agrees [(-3, 1), (1, 2), (4, 1)] 3
#guard agrees [(-2, 1), (0, 1), (3, 2)] 4
-- One die, one face, and no dice.
#guard agrees (standardFaces 6) 1
#guard agrees [(7, 3)] 3
#guard agrees [(1, 1), (2, 2)] 0
-- `4D6` has `C(9, 4)` multisets, which total its 1296 rolls.
#guard (Roll.standard 4 6).outcomes.length == 126
#guard ((Roll.standard 4 6).outcomes.map (·.2)).sum ==
  (Roll.standard 4 6).total

-- The other branches: a range, an empty one, no dice, dice without faces,
-- and fixed results.
#guard (Roll.range (-1) 1).outcomes ==
  [([(-1, 1)], 1), ([(0, 1)], 1), ([(1, 1)], 1)]
#guard (Roll.range (-1) 1).total == 3
#guard (Roll.standard (-2) 6).outcomes == [([], 1)]
#guard (Roll.standard 3 0).outcomes == [([(0, 3)], 1)]
#guard (Roll.standard 3 0).total == 1
#guard (Roll.fixed [(1, 2), (4, 1)]).outcomes == [([(1, 2), (4, 1)], 1)]

-- A setup's outcomes keep its drops, and a setup without a roll is its own.
#guard ({ roll := some (.standard 2 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 2)]), lowest := 1 }, 1),
    ({ roll := some (.fixed [(1, 1), (2, 1)]), lowest := 1 }, 2),
    ({ roll := some (.fixed [(2, 2)]), lowest := 1 }, 1)]
#guard ({ highest := 0 } : Setup).outcomes == [({}, 1)]
#guard (({ roll := some (.standard 2 2), lowest := 1 } : Setup).outcomes.map
    (·.1.held)) ==
  [{ results := [1, 1], lowest := 1 }, { results := [1, 2], lowest := 1 },
    { results := [2, 2], lowest := 1 }]

-- The law of `2D[1, 2, 2]`, sorted: `[1, 2]` along four of its nine rolls.
example : ((Roll.custom 2 [(1, 1), (2, 2)]).law.map Record.sorted) [1, 2] =
    4 / 9 := by
  rw [Roll.map_sorted_law (.custom 2 [(1, 1), (2, 2)])
      (show _ ∧ _ from ⟨by decide, by decide⟩),
    show OrderStatistics.integral (Roll.custom 2 [(1, 1), (2, 2)]).outcomes
      (fun rs => if Roll.expand rs = [1, 2] then 1 else 0) = 4 by decide,
    show (Roll.custom 2 [(1, 1), (2, 2)]).total = 9 by decide]
  norm_num

/-! ## Docstring examples, verbatim -/

#guard Outcomes.grow [(1, 2)] 3 0 == [(1, 2)]
#guard Outcomes.grow [(1, 2)] 3 2 == [(1, 2), (3, 2)]

-- Two dice of `[1, 2, 2]` roll `{1, 1}` once, `{1, 2}` four ways, and
-- `{2, 2}` four ways.
#guard Outcomes.outcomeStates [(1, 1), (2, 2)] 2 ==
  [([(1, 2)], 1), ([(1, 1), (2, 1)], 4), ([(2, 2)], 4)]

#guard (Roll.standard 3 6).total == 216
#guard (Roll.custom 2 [(1, 1), (2, 2)]).total == 9
#guard (Roll.range 4 3).total == 1

#guard (Roll.range 4 3).outcomes == [([(0, 1)], 1)]
#guard (Roll.range 1 2).outcomes == [([(1, 1)], 1), ([(2, 1)], 1)]
#guard (Roll.custom 2 []).outcomes == [([(0, 2)], 1)]

#guard ({ roll := some (.range 1 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 1)]), lowest := 1 }, 1),
    ({ roll := some (.fixed [(2, 1)]), lowest := 1 }, 1)]

#guard ({ roll := some (.fixed [(1, 2), (3, 1)]), highest := 1 } : Setup).held
  == { results := [1, 1, 3], highest := 1 }

end Xdy.Spec.Test.Outcomes
