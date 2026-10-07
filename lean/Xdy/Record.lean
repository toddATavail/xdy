import Xdy.Primitives

/-!
# Rolling records

A rolling record holds the results of one range or one set of dice, together
with how many of its lowest and highest results have been dropped, and the die
from which they were drawn, whose faces a reroll draws. Models `RollingRecord`
in `xdy/src/evaluator.rs` and its methods in `xdy/src/primitives.rs`, and the
criteria by which a reroll chooses the dice to reroll, `Criterion` in
`primitives.rs`.

The Rust record keeps its dice in the order rolled and crosses out each die
that a drop or keep drops, always the lowest or highest of the dice still
kept, so that it shows everything that happened to its dice. Only the multiset
of the kept dice reaches its sum or any later drop, keep or reroll, and
counting the lowest and highest results dropped determines that multiset, so
this record keeps counts instead of crossing anything out. A reroll changes
dice in the middle of the kept window, which counts cannot express, so the
record that it leaves holds its kept results alone, without drops. The record
also keeps its results sorted in ascending order as they arrive. Sorting early
means that two records holding the same multiset of results are equal, which
lets the oracle merge machine states that differ only in the order of their
rolls: `3D6` then has `56` states rather than `216`.
-/

namespace Xdy

/-! ## Dice -/

/-- The die from which the results of a rolling record were drawn. Models
`RollingRecordKind` in `xdy/src/evaluator.rs`, less the number of dice, which
nothing reads once they are rolled. A record holds its die so that a reroll
can draw from it. The forms are few, so that records holding equal dice are
equal, which lets the oracle merge their states: a standard die is the range
of its faces, and a custom die keeps its faces sorted. -/
inductive Die where
  /-- No die, as for a record never rolled. Mirrors `Uninitialized`. -/
  | none
  /-- A die uniform over the inclusive range `[start, stop]`, or showing only
  `0` if the range is empty. -/
  | range (start stop : Int)
  /-- A die of the faces, each equally likely by position, sorted in
  ascending order, or showing only `0` if there are none. -/
  | custom (faces : List Int)
  deriving Repr, DecidableEq, Hashable, Inhabited

namespace Die

/--
The die of a standard roll: the range of its faces, which is empty if it has
none.

# Parameters
- `faces`: The number of faces.

# Returns
The die.

# Examples
```lean
#guard Die.standard 6 == .range 1 6
```
-/
def standard (faces : Int) : Die := .range 1 faces

/--
The die of a custom roll: its faces, sorted, so that dice whose faces differ
only in order are equal.

# Parameters
- `faces`: The faces, in any order, possibly repeated.

# Returns
The die.

# Examples
```lean
#guard Die.ofFaces [3, 1, 3] == .custom [1, 3, 3]
```
-/
def ofFaces (faces : List Int) : Die := .custom (faces.mergeSort (· ≤ ·))

/--
The faces of a die, each listed once for each position that shows it: the
values of a range, in ascending order, or the faces of a custom die. A die that
has no faces, or an empty range, shows only `0`, as `roll_die` in
`primitives.rs` rolls it; and no die has no faces at all, since a record never
rolled holds no dice to reroll.

# Parameters
- `d`: The die.

# Returns
The faces.

# Examples
```lean
#guard (Die.range 2 4).faces == [2, 3, 4]
#guard (Die.standard 0).faces == [0]
#guard (Die.ofFaces [3, 1, 3]).faces == [1, 3, 3]
#guard (Die.ofFaces []).faces == [0]
#guard Die.none.faces == []
```
-/
def faces : Die → List Int
  | .none => []
  | .range start stop =>
    if stop < start then [0]
    else (List.range (stop - start + 1).toNat).map fun (i : Nat) => start + i
  | .custom [] => [0]
  | .custom faces => faces

end Die

/-! ## Criteria -/

/-- The set of results that a reroll rerolls: a predicate whose operands are
resolved to values. Mirrors `Criterion` in `primitives.rs`. -/
inductive Criterion where
  /-- Matches every result from `start` to `stop`, inclusive, or none if the
  range is empty. -/
  | within (start stop : Int)
  /-- Matches every result outside the inclusive range from `start` to
  `stop`, or every result if the range is empty. -/
  | outside (start stop : Int)
  /-- Matches every listed result. -/
  | among (values : List Int)
  deriving Repr, BEq, Inhabited

namespace Criterion

/--
Whether a criterion matches a result. Mirrors `Criterion::matches`.

# Parameters
- `c`: The criterion.
- `x`: The result.

# Returns
`true` if `c` matches `x`, `false` otherwise.

# Examples
```lean
#guard (Criterion.within 1 2).matches 2
#guard !(Criterion.outside 1 2).matches 2
#guard (Criterion.among [1, 3]).matches 3
#guard !(Criterion.within 2 1).matches 2
```
-/
def «matches» : Criterion → Int → Bool
  | .within start stop, x => decide (start ≤ x ∧ x ≤ stop)
  | .outside start stop, x => !decide (start ≤ x ∧ x ≤ stop)
  | .among values, x => values.contains x

end Criterion

namespace Die

/--
Whether a criterion matches every face of a die, as an unlimited reroll asks
before it rerolls anything. Mirrors `faces_within` in `primitives.rs`.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Returns
`true` if `c` matches every face of `d`, as it does vacuously if `d` is
`none`, which has none; `false` otherwise.

# Notes
The Rust decides a range in time independent of its width; this lists its
faces, which costs no more than the oracle's roll of the die did.

# Examples
```lean
#guard (Die.standard 3).facesWithin (.within 1 3)
#guard !(Die.standard 3).facesWithin (.within 1 2)
#guard (Die.standard 0).facesWithin (.among [0])
```
-/
def facesWithin (d : Die) (c : Criterion) : Bool := d.faces.all c.matches

end Die

/-! ## Records -/

/-- The results of a range or a set of dice, the drops applied to them, and
the die from which they were drawn. Equality is decided, not merely a `BEq`,
so that it is lawful, as proofs about a `Dist` of records require. -/
structure Record where
  /-- The results, sorted in ascending order. -/
  results : List Int := []
  /-- The number of lowest results dropped, at most the number of results. -/
  lowest : Nat := 0
  /-- The number of highest results dropped, at most the number of results. -/
  highest : Nat := 0
  /-- The die from which the results were drawn. -/
  die : Die := .none
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
  Min.min (total + count.toNat) len

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
Answer how many of the kept results of a record a keep drops: all of them but
`count`, or none if no more than `count` are kept. Mirrors `keep_drops` in
`primitives.rs`, which serves as the definition of both keeps, in the
evaluator and in the forward pass alike.

# Parameters
- `results`: The number of results.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `count`: The number of results to keep. A count of zero or less keeps
  nothing.

# Returns
The number of results to drop, between `0` and the number kept.

# Notes
Rust clamps the number kept at `0`, as subtraction of naturals does here.

# Examples
```lean
#guard Record.keepDrops 4 0 0 3 == 1
#guard Record.keepDrops 4 1 0 3 == 0
#guard Record.keepDrops 4 1 1 (-1) == 2
#guard Record.keepDrops 2 2 2 1 == 0
```
-/
def keepDrops (results lowest highest : Nat) (count : Int) : Int :=
  let kept : Nat := results - lowest - highest
  kept - Min.min (max count 0) kept

/--
Keep the lowest `count` of the results still kept, dropping the rest of them
from the high end. Mirrors `keep_lowest` in `primitives.rs`.

# Parameters
- `r`: The record.
- `count`: The number of results to keep. A count of zero or less keeps
  nothing.

# Returns
The record with its count of highest results dropped updated.

# Examples
```lean
#guard (({ results := [1, 2, 3, 4] } : Record).keepLowest 3).highest == 1
#guard (({ results := [1, 2, 3, 4], lowest := 1 } : Record).keepLowest 2
  |>.kept) == [2, 3]
#guard (({ results := [1, 2, 3] } : Record).keepLowest 5).highest == 0
#guard (({ results := [1, 2, 3] } : Record).keepLowest 0).kept == []
```
-/
def keepLowest (r : Record) (count : Int) : Record :=
  r.dropHighest (keepDrops r.results.length r.lowest r.highest count)

/--
Keep the highest `count` of the results still kept, dropping the rest of them
from the low end. Mirrors `keep_highest` in `primitives.rs`.

# Parameters
- `r`: The record.
- `count`: The number of results to keep. A count of zero or less keeps
  nothing.

# Returns
The record with its count of lowest results dropped updated.

# Examples
```lean
#guard (({ results := [1, 2, 3, 4] } : Record).keepHighest 3).lowest == 1
#guard (({ results := [1, 2, 3, 4], highest := 1 } : Record).keepHighest 2
  |>.kept) == [2, 3]
#guard (({ results := [1, 2, 3] } : Record).keepHighest (-2)).kept == []
```
-/
def keepHighest (r : Record) (count : Int) : Record :=
  r.dropLowest (keepDrops r.results.length r.lowest r.highest count)

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
