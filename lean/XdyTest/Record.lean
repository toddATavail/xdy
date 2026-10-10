import Xdy.Record

/-!
# Rolling record tests

Sorting on arrival, drop accumulation and clamping, keeps, the kept window,
the ascending saturating fold, and chains, their totals and order, mirroring
`RollingRecord` in `xdy/src/primitives.rs`.
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

/-! ## Saturated values -/

-- The sign of the mean face, or `0` if that mean is `0`, as for a die without
-- faces, an empty range, and a record never rolled.
#guard (Die.range 1 1).saturatedValue == i32Max
#guard (Die.range (-1) 1).saturatedValue == 0
#guard (Die.range (-2) 1).saturatedValue == i32Min
#guard (Die.range 3 2).saturatedValue == 0
#guard (Die.ofFaces [-5, 1, 1, 1, 1]).saturatedValue == i32Min
#guard (Die.ofFaces []).saturatedValue == 0
#guard Die.none.saturatedValue == 0
-- A range of any width answers at once.
#guard (Die.range i32Min i32Max).saturatedValue == i32Min

/-! ## Chains -/

-- A chain's total folds its dice in ascending order, saturating, unless it
-- has saturated, when it is its value.
#guard (Chain.single 4).total == 4
#guard (Chain.live [2, 3] 1).total == 6
#guard (Chain.live [i32Max] 1).total == i32Max
#guard (Chain.live [i32Min, i32Max] 1).total == 0
#guard (Chain.saturated 0).total == 0
-- Its dice are its body and its last, sorted.
#guard (Chain.live [5] 2).dice == [2, 5]
#guard (Chain.single 4).dice == [4]
-- Chains are ordered by total, then by their dice, then by their last die.
#guard (Chain.single 3).le (.single 4) && !(Chain.single 4).le (.single 3)
#guard (Chain.live [3] 1).le (.live [2] 2)
#guard !(Chain.live [2] 2).le (.live [3] 1)
#guard (Chain.live [3] 1).le (.live [1] 3)
#guard !(Chain.live [1] 3).le (.live [3] 1)
#guard (Chain.live [6] 1).le (.live [6] 1)
#guard (Chain.live [] i32Max).le (.saturated i32Max)
#guard !(Chain.saturated i32Max).le (.live [] i32Max)
-- Only a chain that may still grow grows, from its last die.
#guard (Chain.single 6).grows (.within 6 6)
#guard !(Chain.live [6] 2).grows (.within 6 6)
#guard !(Chain.saturated 6).grows (.within 6 6)

/-! ## Records of chains -/

-- A record whose chains are all single dice holds none, and equals the record
-- of their results.
#guard Record.ofChains (.standard 6) [.single 5, .single 2]
  == { results := [2, 5], die := .standard 6 }
#guard Record.ofChains .none [] == {}
-- Otherwise it holds them, sorted, and its results are their totals. Of
-- chains of equal totals, `[2, 6]` precedes `[3, 5]`.
#guard Record.ofChains .none [.live [5] 3, .live [6] 2, .single 9]
  == { results := [8, 8, 9]
       chains := [.live [6] 2, .live [5] 3, .single 9] }
#guard Record.ofChains .none [.saturated i32Max, .saturated i32Max]
  == { results := [i32Max, i32Max]
       chains := [.saturated i32Max, .saturated i32Max] }
-- Drops and keeps count chains, by their totals and then their dice, so the
-- kept window is a window of the chains.
#guard ((Record.ofChains .none [.live [6] 2, .live [5] 3, .single 9]
  ).dropLowest 1).keptChains == [.live [5] 3, .single 9]
#guard ((Record.ofChains .none [.live [6] 2, .live [5] 3, .single 9]
  ).keepLowest 1).keptChains == [.live [6] 2]
#guard ((Record.ofChains .none [.live [6] 2, .live [5] 3, .single 9]
  ).dropHighest 2).sum == 8
#guard (Record.ofChains .none [.live [6] 2, .single 9]).sum == 17
-- A record without chains is a record of single dice.
#guard ((rec [5, 1, 3]).dropHighest 1).keptChains == [.single 1, .single 3]
#guard (rec []).keptChains == []

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
#guard (Die.standard 6).saturatedValue == i32Max
#guard (Die.range (-3) 2).saturatedValue == i32Min
#guard (Die.ofFaces [-1, 1]).saturatedValue == 0
#guard (Die.standard 0).saturatedValue == 0
#guard (Chain.single 3).isSingle
#guard !(Chain.live [1] 3).isSingle
#guard !(Chain.saturated 0).isSingle
#guard (Chain.live [2, 6] 1).dice == [1, 2, 6]
#guard (Chain.saturated i32Max).dice == [i32Max]
#guard (Chain.live [6] 2).total == 8
#guard (Chain.live [i32Max, i32Max] i32Min).total == i32Max - 1
#guard (Chain.saturated i32Min).total == i32Min
#guard (Chain.live [6] 1).le (.live [5] 2)
#guard !(Chain.live [5] 2).le (.live [6] 1)
#guard (Chain.single 4).le (.live [6] 2)
#guard (Chain.live [1] 6).grows (.within 6 6)
#guard !(Chain.live [6] 1).grows (.within 6 6)
#guard !(Chain.saturated i32Max).grows (.within i32Min i32Max)
#guard (Chain.live [6] 6).saturate i32Max == .saturated i32Max
#guard (Chain.saturated 0).saturate i32Max == .saturated 0
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
#guard ({ results := [2, 7], highest := 1
          chains := [.live [] 2, .live [1] 6] } : Record).keptChains
  == [.single 2]
#guard ({ results := [1, 3], lowest := 1 } : Record).keptChains
  == [.single 3]
#guard Record.ofChains .none [.single 3, .single 1]
  == { results := [1, 3] }
#guard Record.ofChains .none [.live [6] 2, .single 4]
  == { results := [4, 8], chains := [.single 4, .live [6] 2] }

end Xdy.Test
