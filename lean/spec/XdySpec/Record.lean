import Xdy.Record

/-!
# Rolling records, in roll order

A rolling record holds the results of one range or one set of dice, together
with how many of its lowest and highest results have been dropped, and the die
from which they were drawn, the oracle's `Xdy.Die`. Models
`RollingRecord` in `xdy/src/evaluator.rs` and its methods in
`xdy/src/primitives.rs`.

The Rust record crosses out each die that a drop or keep drops, always the
lowest or highest of the dice still kept. Only the multiset of the kept dice
reaches its sum or any later drop or keep, and counting the lowest and highest
results dropped determines that multiset, so this record keeps counts instead
of crossing anything out. Like the Rust record, and unlike the oracle's
`Xdy.Record`, this one keeps its results in the order they were rolled, and
sorts them only to find the ones that its drops keep. The oracle sorts on arrival so that it can merge states
that differ only in the order of their rolls; the specification merges
nothing, so it follows the Rust as written. `toOracle` reads a record as the
oracle's by sorting it, and the lemmas beside it, that sorting commutes with
rolling, dropping, keeping and summing, are how `XdySpec/Oracle.lean` proves
that sorting early changes nothing.
-/

namespace Xdy.Spec

open Xdy (Die)

/-- The results of a range or a set of dice, in the order rolled, the drops
applied to them, and the die from which they were drawn. -/
structure Record where
  /-- The results, in the order rolled. -/
  results : List Int := []
  /-- The number of lowest results dropped, at most the number of results. -/
  lowest : Nat := 0
  /-- The number of highest results dropped, at most the number of results. -/
  highest : Nat := 0
  /-- The die from which the results were drawn. -/
  die : Die := .none
  deriving Repr, DecidableEq, Inhabited

namespace Record

/--
Record one more result, after those already rolled.

# Parameters
- `r`: The record.
- `x`: The result to add.

# Returns
The record with `x` as its last result.

# Examples
```lean
#guard (({} : Record).push 3 |>.push 1 |>.push 2).results == [3, 1, 2]
```
-/
def push (r : Record) (x : Int) : Record :=
  { r with results := r.results ++ [x] }

/--
Add a drop count to a running total, as `drop_lowest` and `drop_highest` in
`primitives.rs` do: a negative count drops nothing, and the total is clamped to
the number of results.

# Parameters
- `total`: The running total of dropped results.
- `count`: The number of results to drop.
- `len`: The number of results.

# Returns
The new running total.

# Notes
Rust adds with saturation at `i32Max` and then clamps to the number of
results, which never exceeds `i32Max`, so adding exactly and then clamping
gives the same total. The oracle's `Xdy.Record` accumulates the same way.
-/
private def accumulate (total : Nat) (count : Int) (len : Nat) : Nat :=
  min (total + count.toNat) len

/--
Drop the lowest `count` results, in addition to any already dropped.

# Parameters
- `r`: The record.
- `count`: The number of results to drop. A negative count drops nothing.

# Returns
The record with its drop count updated. The results themselves, and their
order, are unchanged.

# Examples
```lean
#guard (({ results := [3, 1, 2] } : Record).dropLowest 2).lowest == 2
#guard (({ results := [3, 1, 2] } : Record).dropLowest 5).lowest == 3
#guard (({ results := [3, 1, 2] } : Record).dropLowest (-1)).lowest == 0
```
-/
def dropLowest (r : Record) (count : Int) : Record :=
  { r with lowest := accumulate r.lowest count r.results.length }

/--
Drop the highest `count` results, in addition to any already dropped.

# Parameters
- `r`: The record.
- `count`: The number of results to drop. A negative count drops nothing.

# Returns
The record with its drop count updated. The results themselves, and their
order, are unchanged.

# Examples
```lean
#guard (({ results := [3, 1, 2] } : Record).dropHighest 1).highest == 1
```
-/
def dropHighest (r : Record) (count : Int) : Record :=
  { r with highest := accumulate r.highest count r.results.length }

/--
Keep the lowest `count` of the results still kept, dropping the rest of them
from the high end, as many as `Xdy.Record.keepDrops` counts. Mirrors
`keep_lowest` in `primitives.rs`.

# Parameters
- `r`: The record.
- `count`: The number of results to keep. A count of zero or less keeps
  nothing.

# Returns
The record with its count of highest results dropped updated. The results
themselves, and their order, are unchanged.

# Examples
```lean
#guard (({ results := [3, 1, 4, 2] } : Record).keepLowest 3).highest == 1
#guard (({ results := [3, 1, 2] } : Record).keepLowest 0).highest == 3
```
-/
def keepLowest (r : Record) (count : Int) : Record :=
  r.dropHighest (Xdy.Record.keepDrops r.results.length r.lowest r.highest count)

/--
Keep the highest `count` of the results still kept, dropping the rest of them
from the low end, as many as `Xdy.Record.keepDrops` counts. Mirrors
`keep_highest` in `primitives.rs`.

# Parameters
- `r`: The record.
- `count`: The number of results to keep. A count of zero or less keeps
  nothing.

# Returns
The record with its count of lowest results dropped updated. The results
themselves, and their order, are unchanged.

# Examples
```lean
#guard (({ results := [3, 1, 4, 2] } : Record).keepHighest 3).lowest == 1
#guard (({ results := [3, 1, 2], highest := 1 } : Record).keepHighest 1).lowest
  == 1
```
-/
def keepHighest (r : Record) (count : Int) : Record :=
  r.dropLowest (Xdy.Record.keepDrops r.results.length r.lowest r.highest count)

/--
Answer the results in ascending order, the order in which `sum` in
`primitives.rs` folds the kept results.

# Parameters
- `r`: The record.

# Returns
The results, sorted in ascending order.

# Notes
Rust's `sort` is a stable merge sort, as `List.mergeSort` is. Any sort gives
the same list of integers; the choice only mirrors the Rust.

# Examples
```lean
#guard ({ results := [3, 1, 2] } : Record).sorted == [1, 2, 3]
```
-/
def sorted (r : Record) : List Int := r.results.mergeSort (· ≤ ·)

/--
Recording a die leaves the sorted results alone.

# Parameters
- `r`: The record.
- `x`: The die.
-/
theorem sorted_die (r : Record) (x : Die) :
    ({ r with die := x } : Record).sorted = r.sorted := rfl

/--
Answer the results that survive the drops: sort the results, skip the lowest
dropped, then take what the highest dropped leave. These are the results that
`kept` in `primitives.rs` answers, sorted rather than in the order rolled.

# Parameters
- `r`: The record.

# Returns
The kept results, in ascending order. Empty if the drops cover every result.

# Examples
```lean
#guard ({ results := [4, 1, 3, 2], lowest := 1, highest := 1 } : Record).kept
  == [2, 3]
#guard ({ results := [2, 1], lowest := 2, highest := 2 } : Record).kept == []
```
-/
def kept (r : Record) : List Int :=
  (r.sorted.drop r.lowest).take (r.results.length - r.lowest - r.highest)

/--
Sum the kept results, folding in ascending order with saturating addition.
Mirrors `sum` in `primitives.rs`.

# Parameters
- `r`: The record.

# Returns
The saturating sum of the kept results, or `0` if none are kept.

# Notes
The order of the fold matters when the results have mixed signs and the sum
saturates part way, which is why it must follow the Rust exactly: it folds
the kept results, which are sorted, not the results in the order rolled.

# Examples
```lean
#guard ({ results := [3, 1, 2], lowest := 1 } : Record).sum == 5
#guard ({ results := [i32Max, i32Min, i32Max] } : Record).sum == i32Max - 1
```
-/
def sum (r : Record) : Int := r.kept.foldl add 0

/--
Read a record of the oracle as a record of the specification.

# Parameters
- `r`: The oracle's record, whose results are sorted.

# Returns
The record with the same results, in the same order, the same drops and the
same die.

# Notes
The oracle's sorted results are one order in which they could have been
rolled, so the specification's record describes the same roll.

# Examples
```lean
#guard (Record.ofOracle ((({} : Xdy.Record).push 3).push 1)).results
  == [1, 3]
```
-/
def ofOracle (r : Xdy.Record) : Record :=
  { results := r.results, lowest := r.lowest, highest := r.highest,
    die := r.die }

/--
Read a record of the specification as a record of the oracle, by sorting its
results. Inverse to `ofOracle` on sorted records.

# Parameters
- `r`: The specification's record, whose results are in roll order.

# Returns
The record with the same results, sorted in ascending order, as the oracle
keeps them, the same drops and the same die.

# Notes
The oracle's distribution of states is the specification's, with every
record read by `toOracle`: that is how the proof that the oracle meets the
specification relates them.

# Examples
```lean
#guard (Record.toOracle { results := [3, 1, 2], lowest := 1 }).results
  == [1, 2, 3]
```
-/
def toOracle (r : Record) : Xdy.Record :=
  { results := r.sorted, lowest := r.lowest, highest := r.highest,
    die := r.die }

/--
The sorted results are sorted.

# Parameters
- `r`: The record.
-/
theorem sorted_pairwise (r : Record) : r.sorted.Pairwise (· ≤ ·) := by
  have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
    (by intro a b c; simp; omega) (by intro a b; simp; omega) r.results
  unfold sorted
  simpa using this

/--
Sorting commutes with recording a result: rolling in order and then sorting
gives the oracle's record, which sorts as it rolls.

# Parameters
- `r`: The record.
- `x`: The result to add.
-/
theorem toOracle_push (r : Record) (x : Int) :
    (r.push x).toOracle = r.toOracle.push x := by
  have hres : (r.push x).sorted = (r.toOracle.push x).results := by
    refine List.Perm.eq_of_pairwise (le := (· ≤ ·))
      (fun a b _ _ h1 h2 => by omega) (sorted_pairwise _)
      (Xdy.Record.push_sorted _ x (sorted_pairwise r)) ?_
    exact ((List.mergeSort_perm _ _).trans
      (List.perm_append_singleton x r.results)).trans
      ((List.Perm.cons x (List.mergeSort_perm r.results _)).symm.trans
        (Xdy.Record.push_perm r.toOracle x).symm)
  rw [show r.toOracle.push x =
      { results := (r.toOracle.push x).results, lowest := r.lowest,
        highest := r.highest, die := r.die } from rfl, ← hres]
  rfl

/--
Sorting commutes with dropping the lowest results, which depends only on how
many there are.

# Parameters
- `r`: The record.
- `c`: The number of results to drop.
-/
theorem toOracle_dropLowest (r : Record) (c : Int) :
    (r.dropLowest c).toOracle = r.toOracle.dropLowest c := by
  simp only [toOracle, dropLowest, Xdy.Record.dropLowest, sorted,
    List.length_mergeSort]
  rfl

/--
Sorting commutes with dropping the highest results, which depends only on
how many there are.

# Parameters
- `r`: The record.
- `c`: The number of results to drop.
-/
theorem toOracle_dropHighest (r : Record) (c : Int) :
    (r.dropHighest c).toOracle = r.toOracle.dropHighest c := by
  simp only [toOracle, dropHighest, Xdy.Record.dropHighest, sorted,
    List.length_mergeSort]
  rfl

/--
Sorting commutes with keeping the lowest results, which, as the drop that it
is, depends only on how many results there are and how many are dropped.

# Parameters
- `r`: The record.
- `c`: The number of results to keep.
-/
theorem toOracle_keepLowest (r : Record) (c : Int) :
    (r.keepLowest c).toOracle = r.toOracle.keepLowest c := by
  rw [keepLowest, toOracle_dropHighest]
  simp only [Xdy.Record.keepLowest, toOracle, sorted, List.length_mergeSort]

/--
Sorting commutes with keeping the highest results, which, as the drop that it
is, depends only on how many results there are and how many are dropped.

# Parameters
- `r`: The record.
- `c`: The number of results to keep.
-/
theorem toOracle_keepHighest (r : Record) (c : Int) :
    (r.keepHighest c).toOracle = r.toOracle.keepHighest c := by
  rw [keepHighest, toOracle_dropLowest]
  simp only [Xdy.Record.keepHighest, toOracle, sorted, List.length_mergeSort]

/--
The oracle's sum of the sorted record is the specification's sum, which
sorts before it keeps and folds.

# Parameters
- `r`: The record.
-/
theorem sum_toOracle (r : Record) : r.toOracle.sum = r.sum := by
  simp only [Xdy.Record.sum, Xdy.Record.kept, sum, kept, toOracle, sorted,
    List.length_mergeSort]

/-- The empty record sorts to the empty record. -/
theorem toOracle_empty : ({} : Record).toOracle = {} := by
  simp [toOracle, sorted]

/--
Sorting the specification's reading of an oracle's record gives it back, if
its results are sorted, as the oracle keeps them.

# Parameters
- `r`: The oracle's record.

# Hypotheses
- `sorted`: The results are sorted in ascending order.
-/
theorem toOracle_ofOracle (r : Xdy.Record)
    (sorted : r.results.Pairwise (· ≤ ·)) : (ofOracle r).toOracle = r := by
  have : r.results.mergeSort (fun a b => decide (a ≤ b)) = r.results :=
    List.mergeSort_of_pairwise (by simpa using sorted)
  simp only [toOracle, ofOracle, Record.sorted, this]

/--
Sort a record's results, keeping its drops: the record of the specification
that its sorted reading describes.

# Parameters
- `r`: The record.

# Returns
The record with the same results, sorted in ascending order, the same drops
and the same die.

# Notes
The forward pass fixes a split record at its sorted reading, as `split` in
`propagation.rs` fixes it at the multiset of its results.

# Examples
```lean
#guard (Record.sort { results := [3, 1, 2], lowest := 1 }).results == [1, 2, 3]
```
-/
def sort (r : Record) : Record := ofOracle r.toOracle

/--
Sorting a record leaves its sorted reading alone.

# Parameters
- `r`: The record.
-/
theorem toOracle_sort (r : Record) : r.sort.toOracle = r.toOracle :=
  toOracle_ofOracle _ (sorted_pairwise r)

/--
Sorting a sorted record leaves it alone.

# Parameters
- `r`: The record.
-/
theorem sort_sort (r : Record) : r.sort.sort = r.sort := by
  rw [sort, toOracle_sort]
  rfl

/--
Two records sort alike exactly when their sorted readings agree.

# Parameters
- `r`, `r'`: The records.
-/
theorem sort_eq_sort_iff {r r' : Record} :
    r.sort = r'.sort ↔ r.toOracle = r'.toOracle := by
  constructor
  · intro h
    rw [← toOracle_sort r, h, toOracle_sort]
  · intro h
    rw [sort, sort, h]

end Record

end Xdy.Spec
