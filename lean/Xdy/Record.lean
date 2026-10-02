import Xdy.Primitives

/-!
# Rolling records

A rolling record holds the results of one range or one set of dice, together
with how many of its lowest and highest results have been dropped. Mirrors
`RollingRecord` in `xdy/src/evaluator.rs` and its methods in
`xdy/src/primitives.rs`.

Unlike the Rust record, this one keeps its results sorted in ascending order
as they arrive. The Rust record sorts its results only when it drops or sums
them, so nothing that reads a record can observe the order in which its
results were rolled. Sorting early means that two records holding the same
multiset of results are equal, which lets the oracle merge machine states that
differ only in the order of their rolls: `3D6` then has `56` states rather
than `216`.
-/

namespace Xdy

/-- The results of a range or a set of dice, and the drops applied to them.
Equality is decided, not merely a `BEq`, so that it is lawful, as proofs
about a `Dist` of records require. -/
structure Record where
  /-- The results, sorted in ascending order. -/
  results : List Int := []
  /-- The number of lowest results dropped, at most the number of results. -/
  lowest : Nat := 0
  /-- The number of highest results dropped, at most the number of results. -/
  highest : Nat := 0
  deriving Repr, DecidableEq, Hashable, Inhabited

namespace Record

/--
Insert a value into a list sorted in ascending order, keeping it sorted.

# Parameters
- `x`: The value to insert.
- `xs`: A list sorted in ascending order.

# Returns
The sorted list with `x` inserted.
-/
private def insertSorted (x : Int) : List Int → List Int
  | [] => [x]
  | y :: ys => if x <= y then x :: y :: ys else y :: insertSorted x ys

/--
Record one more result.

# Parameters
- `r`: The record.
- `x`: The result to add.

# Returns
The record with `x` among its results.

# Examples
```lean
#guard (({} : Record).push 3 |>.push 1 |>.push 2).results == [1, 2, 3]
```
-/
def push (r : Record) (x : Int) : Record :=
  { r with results := insertSorted x r.results }

/--
Inserting a value into a list gives a permutation of the list with the value
in front.

# Parameters
- `x`: The value to insert.
- `xs`: The list.
-/
private theorem insertSorted_perm (x : Int) (xs : List Int) :
    (insertSorted x xs).Perm (x :: xs) := by
  induction xs with
  | nil => exact .refl _
  | cons y ys ih =>
    simp only [insertSorted]
    split
    · exact .refl _
    · exact (ih.cons y).trans (.swap x y ys)

/--
Inserting a value into a list sorted in ascending order keeps it sorted.

# Parameters
- `x`: The value to insert.
- `xs`: The list.

# Hypotheses
- `sorted`: The list is sorted in ascending order.
-/
private theorem insertSorted_sorted (x : Int) (xs : List Int)
    (sorted : xs.Pairwise (· ≤ ·)) : (insertSorted x xs).Pairwise (· ≤ ·) := by
  induction xs with
  | nil => simp [insertSorted]
  | cons y ys ih =>
    rw [List.pairwise_cons] at sorted
    simp only [insertSorted]
    split
    · refine List.Pairwise.cons ?_ (List.pairwise_cons.mpr sorted)
      intro z hz
      rcases List.mem_cons.mp hz with rfl | hz
      · assumption
      · have := sorted.1 z hz; omega
    · refine List.Pairwise.cons ?_ (ih sorted.2)
      intro z hz
      rcases List.mem_cons.mp ((insertSorted_perm x ys).mem_iff.mp hz) with
        rfl | hz
      · omega
      · exact sorted.1 z hz

/--
Recording a result gives a permutation of the results with the new one in
front, so the results are those rolled, in some order.

# Parameters
- `r`: The record.
- `x`: The result to add.
-/
theorem push_perm (r : Record) (x : Int) :
    (r.push x).results.Perm (x :: r.results) :=
  insertSorted_perm x r.results

/--
Recording a result keeps sorted results sorted.

# Parameters
- `r`: The record.
- `x`: The result to add.

# Hypotheses
- `sorted`: The results are sorted in ascending order.
-/
theorem push_sorted (r : Record) (x : Int)
    (sorted : r.results.Pairwise (· ≤ ·)) :
    (r.push x).results.Pairwise (· ≤ ·) :=
  insertSorted_sorted x r.results sorted

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
gives the same total.
-/
private def accumulate (total : Nat) (count : Int) (len : Nat) : Nat :=
  min (total + count.toNat) len

/--
Drop the lowest `count` results, in addition to any already dropped.

# Parameters
- `r`: The record.
- `count`: The number of results to drop. A negative count drops nothing.

# Returns
The record with its drop count updated.

# Examples
```lean
#guard (({ results := [1, 2, 3] } : Record).dropLowest 2).lowest == 2
#guard (({ results := [1, 2, 3] } : Record).dropLowest 5).lowest == 3
#guard (({ results := [1, 2, 3] } : Record).dropLowest (-1)).lowest == 0
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
The record with its drop count updated.

# Examples
```lean
#guard (({ results := [1, 2, 3] } : Record).dropHighest 1).highest == 1
```
-/
def dropHighest (r : Record) (count : Int) : Record :=
  { r with highest := accumulate r.highest count r.results.length }

/--
Answer the results that survive the drops: skip the lowest dropped, then take
what the highest dropped leave. Mirrors `kept` in `primitives.rs`.

# Parameters
- `r`: The record.

# Returns
The kept results, in ascending order. Empty if the drops cover every result.

# Examples
```lean
#guard ({ results := [1, 2, 3, 4], lowest := 1, highest := 1 } : Record).kept
  == [2, 3]
#guard ({ results := [1, 2], lowest := 2, highest := 2 } : Record).kept == []
```
-/
def kept (r : Record) : List Int :=
  (r.results.drop r.lowest).take (r.results.length - r.lowest - r.highest)

/--
Sum the kept results, folding in ascending order with saturating addition.
Mirrors `sum` in `primitives.rs`.

# Parameters
- `r`: The record.

# Returns
The saturating sum of the kept results, or `0` if none are kept.

# Notes
The order of the fold matters when the results have mixed signs and the sum
saturates part way, which is why it must follow the Rust exactly.

# Examples
```lean
#guard ({ results := [1, 2, 3], lowest := 1 } : Record).sum == 5
#guard ({ results := [i32Min, i32Max, i32Max] } : Record).sum == i32Max - 1
```
-/
def sum (r : Record) : Int := r.kept.foldl add 0

end Record

end Xdy
