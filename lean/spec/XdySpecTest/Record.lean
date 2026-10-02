import XdySpec.Record

/-!
# Rolling record tests

Roll order, drop accumulation and clamping, the kept window of the sorted
results, and the ascending saturating fold, mirroring `RollingRecord` in
`xdy/src/primitives.rs`.
-/

namespace Xdy.Spec.Test

/-- A record holding the given results, pushed in the order given. -/
private def rec (xs : List Int) : Record := xs.foldl Record.push {}

/-! ## Roll order -/

#guard (rec [3, 1, 2]).results == [3, 1, 2]
#guard (rec [3, 1, 2]).sorted == [1, 2, 3]
-- Records of the same results in different orders differ, but keep and sum
-- alike.
#guard rec [6, 1] != rec [1, 6]
#guard (rec [6, 1]).kept == (rec [1, 6]).kept

/-! ## Reading records as the oracle's -/

-- Sorting merges what roll order keeps apart.
#guard (rec [6, 1]).toOracle == (rec [1, 6]).toOracle
#guard (rec [3, 1, 2]).toOracle.results == [1, 2, 3]
#guard (rec [3, 1, 2]).toOracle == ((({} : Xdy.Record).push 3).push 1).push 2
#guard ((rec [5, 1, 3]).dropLowest 1).toOracle.sum
  == ((rec [5, 1, 3]).dropLowest 1).sum
example : (rec [3, 1]).toOracle = ((rec [3]).toOracle.push 1) :=
  Record.toOracle_push _ 1

/-! ## Drops accumulate and clamp to the number of results -/

#guard ((rec [3, 1, 2]).dropLowest 1 |>.dropLowest 1).lowest == 2
#guard ((rec [3, 1, 2]).dropLowest 2 |>.dropLowest 2).lowest == 3
#guard ((rec [3, 1, 2]).dropLowest (-5)).lowest == 0
#guard ((rec [3, 1, 2]).dropHighest i32Max).highest == 3
-- Dropping leaves the results, and their order, alone.
#guard ((rec [3, 1, 2]).dropLowest 1).results == [3, 1, 2]

/-! ## The kept window -/

#guard (rec [5, 1, 3]).kept == [1, 3, 5]
#guard ((rec [5, 1, 3]).dropLowest 1).kept == [3, 5]
#guard ((rec [5, 1, 3]).dropHighest 1).kept == [1, 3]
#guard ((rec [5, 1, 3]).dropLowest 1 |>.dropHighest 1).kept == [3]
#guard ((rec [5, 1, 3]).dropLowest 2 |>.dropHighest 2).kept == []
#guard (rec []).kept == []

/-! ## The ascending saturating fold -/

#guard (rec [1, 2, 3]).sum == 6
#guard ((rec [6, 2, 4, 1]).dropLowest 1).sum == 12
#guard (rec []).sum == 0
-- The fold sorts first, whatever the roll order, so the negatives come first.
#guard (rec [i32Max, i32Max, i32Min]).sum == i32Max - 1
#guard (rec [i32Max, i32Min, i32Max]).sum == i32Max - 1
#guard (rec [i32Max, i32Max, -1]).sum == i32Max

/-! ## Reading the oracle's records -/

#guard (Record.ofOracle (Xdy.Record.push {} 2 |>.push 1)).results == [1, 2]
#guard (Record.ofOracle ((Xdy.Record.push {} 2 |>.push 1).dropLowest 1)).kept
  == [2]

/-! ## Docstring examples, verbatim -/

#guard (({} : Record).push 3 |>.push 1 |>.push 2).results == [3, 1, 2]
#guard (({ results := [3, 1, 2] } : Record).dropLowest 2).lowest == 2
#guard (({ results := [3, 1, 2] } : Record).dropLowest 5).lowest == 3
#guard (({ results := [3, 1, 2] } : Record).dropLowest (-1)).lowest == 0
#guard (({ results := [3, 1, 2] } : Record).dropHighest 1).highest == 1
#guard ({ results := [3, 1, 2] } : Record).sorted == [1, 2, 3]
#guard ({ results := [4, 1, 3, 2], lowest := 1, highest := 1 } : Record).kept
  == [2, 3]
#guard ({ results := [2, 1], lowest := 2, highest := 2 } : Record).kept == []
#guard ({ results := [3, 1, 2], lowest := 1 } : Record).sum == 5
#guard ({ results := [i32Max, i32Min, i32Max] } : Record).sum == i32Max - 1
#guard (Record.ofOracle ((({} : Xdy.Record).push 3).push 1)).results
  == [1, 3]
#guard (Record.toOracle { results := [3, 1, 2], lowest := 1 }).results
  == [1, 2, 3]

end Xdy.Spec.Test
