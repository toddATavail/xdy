import Xdy.Record

/-!
# Rolling record tests

Sorting on arrival, drop accumulation and clamping, the kept window, and the
ascending saturating fold, mirroring `RollingRecord` in `xdy/src/primitives.rs`.
-/

namespace Xdy.Test

open Xdy

/-- A record holding the given results, pushed in the order given. -/
private def rec (xs : List Int) : Record := xs.foldl Record.push {}

/-! ## Sorting on arrival -/

#guard (rec [3, 1, 2]).results == [1, 2, 3]
#guard (rec [2, 2, 1]).results == [1, 2, 2]
#guard rec [6, 1] == rec [1, 6]

/-! ## Lawful equality -/

-- Equality is decided structurally, so it is lawful, as the laws of `Dist`
-- over records require.
example : LawfulBEq Record := inferInstance
example : LawfulHashable Record := inferInstance

/-! ## Pushing keeps a sorted permutation -/

example : (rec [3, 1]).results.Perm [1, 3] :=
  (Record.push_perm _ 1).trans (.cons 1 (Record.push_perm {} 3))
example : (rec [3, 1, 2]).results.Pairwise (· ≤ ·) :=
  Record.push_sorted _ 2 (Record.push_sorted _ 1
    (Record.push_sorted {} 3 .nil))

/-! ## Drops accumulate and clamp to the number of results -/

#guard ((rec [1, 2, 3]).dropLowest 1 |>.dropLowest 1).lowest == 2
#guard ((rec [1, 2, 3]).dropLowest 2 |>.dropLowest 2).lowest == 3
#guard ((rec [1, 2, 3]).dropLowest (-5)).lowest == 0
#guard ((rec [1, 2, 3]).dropHighest i32Max).highest == 3
-- The lowest and highest totals clamp independently of each other.
#guard ((rec [1, 2, 3]).dropLowest 3 |>.dropHighest 3).highest == 3

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
#guard ((rec [1, 2]).dropLowest 2).sum == 0
-- The negatives come first, so the positives can pull the sum back down from
-- the lower bound but it saturates at the upper one.
#guard (rec [i32Min, i32Max, i32Max]).sum == i32Max - 1
#guard (rec [i32Max, i32Max, i32Min]).sum == i32Max - 1
#guard (rec [-1, i32Max, i32Max]).sum == i32Max

/-! ## Docstring examples, verbatim -/

#guard (({} : Record).push 3 |>.push 1 |>.push 2).results == [1, 2, 3]
#guard (({ results := [1, 2, 3] } : Record).dropLowest 2).lowest == 2
#guard (({ results := [1, 2, 3] } : Record).dropLowest 5).lowest == 3
#guard (({ results := [1, 2, 3] } : Record).dropLowest (-1)).lowest == 0
#guard (({ results := [1, 2, 3] } : Record).dropHighest 1).highest == 1
#guard ({ results := [1, 2, 3, 4], lowest := 1, highest := 1 } : Record).kept
  == [2, 3]
#guard ({ results := [1, 2], lowest := 2, highest := 2 } : Record).kept == []
#guard ({ results := [1, 2, 3], lowest := 1 } : Record).sum == 5
#guard ({ results := [i32Min, i32Max, i32Max] } : Record).sum == i32Max - 1

end Xdy.Test
