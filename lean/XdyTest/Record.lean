import Xdy.Record

/-!
# Rolling record tests

Sorting on arrival, drop accumulation and clamping, keeps, the kept window,
and the ascending saturating fold, mirroring `RollingRecord` in
`xdy/src/primitives.rs`.
-/

namespace Xdy.Test

open Xdy

/-- A record holding the given results, pushed in the order given. -/
private def rec (xs : List Int) : Record := xs.foldl Record.push {}

/-! ## Dice -/

-- A standard die is the range of its faces, and a custom die sorts its faces,
-- so that dice that roll alike compare equal.
#guard Die.standard 6 == .range 1 6
#guard Die.standard 0 == .range 1 0
#guard Die.ofFaces [2, -1, 2] == Die.ofFaces [2, 2, -1]
#guard Die.ofFaces [] == .custom []
-- A record never rolled has no die.
#guard ({} : Record).die == .none
-- Records differing only in their dice differ.
#guard ({ results := [1] } : Record) != { results := [1], die := .range 1 6 }

/-! ## Faces -/

-- A die lists each face once for each position that shows it, and a die that
-- has no faces, or an empty range, shows only `0`.
#guard (Die.range (-1) 1).faces == [-1, 0, 1]
#guard (Die.range 3 3).faces == [3]
#guard (Die.range 3 2).faces == [0]
#guard (Die.standard (-2)).faces == [0]
#guard (Die.ofFaces [2, 1, 2]).faces == [1, 2, 2]
-- A record never rolled has no die, and so no faces.
#guard Die.none.faces == []
#guard Die.none.facesWithin (.among [])

/-! ## Criteria -/

#guard (Criterion.within 1 3).matches 1 && (Criterion.within 1 3).matches 3
#guard !(Criterion.within 1 3).matches 0 && !(Criterion.within 1 3).matches 4
-- An empty range matches nothing inside it, and everything outside it.
#guard !(Criterion.within 3 1).matches 2
#guard (Criterion.outside 3 1).matches 2
#guard !(Criterion.outside 2 2).matches 2 && (Criterion.outside 2 2).matches 3
#guard (Criterion.among [3, 1]).matches 1 && !(Criterion.among [3, 1]).matches 2
#guard !(Criterion.among []).matches 0
-- Every face must match, by position or not.
#guard (Die.ofFaces [1, 3, 1]).facesWithin (.among [1, 3])
#guard !(Die.ofFaces [1, 3, 1]).facesWithin (.among [1])
#guard (Die.range 1 4).facesWithin (.outside 5 9)
#guard !(Die.range 1 4).facesWithin (.outside 4 9)
#guard (Die.range 4 3).facesWithin (.within 0 0)
#guard !(Die.ofFaces []).facesWithin (.within 1 1)

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

/-! ## Keeps count from the results still kept -/

#guard Record.keepDrops 3 0 0 1 == 2
#guard Record.keepDrops 3 0 0 3 == 0
#guard Record.keepDrops 3 0 0 i32Max == 0
#guard Record.keepDrops 3 0 0 0 == 3
#guard Record.keepDrops 3 0 0 i32Min == 3
#guard Record.keepDrops 3 1 1 1 == 0
#guard Record.keepDrops 3 3 3 1 == 0
#guard ((rec [5, 1, 3]).keepLowest 2).kept == [1, 3]
#guard ((rec [5, 1, 3]).keepHighest 2).kept == [3, 5]
#guard ((rec [5, 1, 3]).dropLowest 1 |>.keepLowest 1).kept == [3]
#guard ((rec [5, 1, 3]).keepHighest 2 |>.keepLowest 1).kept == [3]
-- Keeps do not commute with drops.
#guard ((rec [5, 1, 3]).keepHighest 2 |>.dropLowest 1).kept == [5]
#guard ((rec [5, 1, 3]).dropLowest 1 |>.keepHighest 2).kept == [3, 5]
-- A keep drops from the opposite end, within the clamp of the drops.
#guard ((rec [5, 1, 3]).keepLowest 0) == (rec [5, 1, 3]).dropHighest 3
#guard ((rec [5, 1, 3]).keepHighest (-1)) == (rec [5, 1, 3]).dropLowest 3
#guard ((rec [5, 1, 3]).dropLowest 3 |>.keepHighest 0).lowest == 3
#guard ((rec []).keepLowest 1).kept == []

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

#guard Die.standard 6 == .range 1 6
#guard Die.ofFaces [3, 1, 3] == .custom [1, 3, 3]
#guard (Die.range 2 4).faces == [2, 3, 4]
#guard (Die.standard 0).faces == [0]
#guard (Die.ofFaces [3, 1, 3]).faces == [1, 3, 3]
#guard (Die.ofFaces []).faces == [0]
#guard Die.none.faces == []
#guard (Criterion.within 1 2).matches 2
#guard !(Criterion.outside 1 2).matches 2
#guard (Criterion.among [1, 3]).matches 3
#guard !(Criterion.within 2 1).matches 2
#guard (Die.standard 3).facesWithin (.within 1 3)
#guard !(Die.standard 3).facesWithin (.within 1 2)
#guard (Die.standard 0).facesWithin (.among [0])
#guard (({} : Record).push 3 |>.push 1 |>.push 2).results == [1, 2, 3]
#guard (({ results := [1, 2, 3] } : Record).dropLowest 2).lowest == 2
#guard (({ results := [1, 2, 3] } : Record).dropLowest 5).lowest == 3
#guard (({ results := [1, 2, 3] } : Record).dropLowest (-1)).lowest == 0
#guard (({ results := [1, 2, 3] } : Record).dropHighest 1).highest == 1
#guard ({ results := [1, 2, 3, 4], lowest := 1, highest := 1 } : Record).kept
  == [2, 3]
#guard ({ results := [1, 2], lowest := 2, highest := 2 } : Record).kept == []
#guard Record.keepDrops 4 0 0 3 == 1
#guard Record.keepDrops 4 1 0 3 == 0
#guard Record.keepDrops 4 1 1 (-1) == 2
#guard Record.keepDrops 2 2 2 1 == 0
#guard (({ results := [1, 2, 3, 4] } : Record).keepLowest 3).highest == 1
#guard (({ results := [1, 2, 3, 4], lowest := 1 } : Record).keepLowest 2
  |>.kept) == [2, 3]
#guard (({ results := [1, 2, 3] } : Record).keepLowest 5).highest == 0
#guard (({ results := [1, 2, 3] } : Record).keepLowest 0).kept == []
#guard (({ results := [1, 2, 3, 4] } : Record).keepHighest 3).lowest == 1
#guard (({ results := [1, 2, 3, 4], highest := 1 } : Record).keepHighest 2
  |>.kept) == [2, 3]
#guard (({ results := [1, 2, 3] } : Record).keepHighest (-2)).kept == []
#guard ({ results := [1, 2, 3], lowest := 1 } : Record).sum == 5
#guard ({ results := [i32Min, i32Max, i32Max] } : Record).sum == i32Max - 1

end Xdy.Test
