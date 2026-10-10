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

An explode grows _chains_, as `chains` in `primitives.rs` describes them:
every die that descends from one rolled die, by replacement or extension. The
sum and every later drop and keep count a chain as one die, worth its total,
so a record that an explode grew holds its chains beside its results, and its
results are their totals. Drops and keeps order chains by their totals, then
by their contents, so the chains are kept sorted in that order, and the drop
counts count chains, exactly as they count results: the drops, keeps, kept
window and sum of a record need not know that it holds chains at all. Only a
reroll, which acts on each die of a chain, and an explode, which grows a chain
from its end, look at them. A record holds chains only if some chain holds
more than one die or saturated, so that a record whose explode grew nothing
equals the record without chains that holds the same results, and their
states merge.
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
Whether a criterion matches every face of a die, as an unlimited reroll or
explode asks before it rolls anything. Mirrors `faces_within` in
`primitives.rs`.

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

/--
The value of a saturated die of a die, which stands for an endless chain of
its dice: `i32Max` or `i32Min` by the sign of the mean face value, or `0` if
that mean is `0`. Mirrors `saturated_value` in `primitives.rs`.

# Parameters
- `d`: The die.

# Returns
The saturated value. A die without faces, or an empty range, shows only `0`,
and a record never rolled has no die, so the value of each is `0`.

# Notes
The sign of the mean is the sign of the sum of the faces, and the sum of a
range has the sign of the sum of its endpoints, so the value takes time
independent of the width of a range, as in the Rust.

# Examples
```lean
#guard (Die.standard 6).saturatedValue == i32Max
#guard (Die.range (-3) 2).saturatedValue == i32Min
#guard (Die.ofFaces [-1, 1]).saturatedValue == 0
#guard (Die.standard 0).saturatedValue == 0
```
-/
def saturatedValue (d : Die) : Int :=
  let sum : Int := match d with
    | .none => 0
    | .range start stop => if stop < start then 0 else start + stop
    | .custom faces => faces.sum
  if 0 < sum then i32Max else if sum < 0 then i32Min else 0

end Die

/-! ## Chains -/

/-- A chain of a record that an explode grew, still kept: every kept die that
descends from one rolled die, by replacement or extension, as `chains` in
`primitives.rs` describes it. The sum and every drop and keep count a chain as
one die, worth its total. -/
inductive Chain where
  /-- A chain that may still grow: its kept dice other than its `last`,
  sorted in ascending order, and its `last`, the die at its end, which no die
  extends, and from which alone an explode grows the chain. A chain of one die
  has no other dice. -/
  | live (body : List Int) (last : Int)
  /-- A chain that a saturated die decides, worth its `value`, which no later
  reroll or explode changes.

  The Rust chain keeps the dice that the saturated die extends, which order
  it among chains of equal totals. An explode saturates every chain still
  kept of its record at the same value, so no drop or keep can tell apart the
  chains of a record that holds one, and this chain forgets them. -/
  | saturated (value : Int)
  deriving Repr, DecidableEq, Hashable, Inhabited

namespace Chain

/--
The chain of one die, as a roll leaves every die.

# Parameters
- `x`: The result of the die.

# Returns
The chain.
-/
def single (x : Int) : Chain := .live [] x

/--
Whether a chain is the chain of one die, as a roll leaves every die.

# Parameters
- `c`: The chain.

# Returns
`true` if `c` holds one die and has not saturated, `false` otherwise.

# Examples
```lean
#guard (Chain.single 3).isSingle
#guard !(Chain.live [1] 3).isSingle
#guard !(Chain.saturated 0).isSingle
```
-/
def isSingle : Chain → Bool
  | .live [] _ => true
  | _ => false

/--
The kept dice of a chain, in ascending order: those of its body and its last.

# Parameters
- `c`: The chain.

# Returns
The dice. A saturated chain shows only its value.

# Examples
```lean
#guard (Chain.live [2, 6] 1).dice == [1, 2, 6]
#guard (Chain.saturated i32Max).dice == [i32Max]
```
-/
def dice : Chain → List Int
  | .live body last => (last :: body).mergeSort
  | .saturated value => [value]

/--
The die at the end of a chain, from which alone an explode grows it.

# Parameters
- `c`: The chain.

# Returns
The last die of a chain that may still grow, or the value of a saturated
chain, whose saturated die stands at its end.
-/
def last : Chain → Int
  | .live _ last => last
  | .saturated value => value

/--
Whether a chain has saturated.

# Parameters
- `c`: The chain.

# Returns
`true` if a saturated die decides `c`, `false` otherwise.
-/
def saturates : Chain → Bool
  | .live .. => false
  | .saturated _ => true

/--
The total of a chain, which the sum and every drop and keep count: the value
of a saturated chain, or else the sum of its dice, folded in ascending order
with saturating addition, as `kept_chains` in `primitives.rs` folds them.

# Parameters
- `c`: The chain.

# Returns
The total.

# Examples
```lean
#guard (Chain.live [6] 2).total == 8
#guard (Chain.live [i32Max, i32Max] i32Min).total == i32Max - 1
#guard (Chain.saturated i32Min).total == i32Min
```
-/
def total : Chain → Int
  | .saturated value => value
  | c => c.dice.foldl add 0

/--
Whether a chain precedes another, or equals it, in the order of drops and
keeps: by their totals, then by their contents, their dice in ascending
order, compared lexicographically, and then their last dice, as `cross_out` in
`primitives.rs` orders them. A chain that may still grow precedes a saturated
chain that agrees with it so far, though no record holds both.

# Parameters
- `a`, `b`: The chains.

# Returns
`true` if `a` precedes `b` or equals it, `false` otherwise.

# Notes
The Rust orders only identical chains by the order in which their earliest
kept dice were rolled. Chains that agree on every key here are equal, so the
order is total, and sorting a multiset of chains answers one list.

# Examples
```lean
-- `[5, 2]` and `[6, 1]` both total `7`, but `[1, 6]` precedes `[2, 5]`.
#guard (Chain.live [6] 1).le (.live [5] 2)
#guard !(Chain.live [5] 2).le (.live [6] 1)
#guard (Chain.single 4).le (.live [6] 2)
```
-/
def le (a b : Chain) : Bool :=
  decide (a.total < b.total) || a.total == b.total &&
    (decide (a.dice < b.dice) || a.dice == b.dice &&
      (decide (a.last < b.last) || a.last == b.last &&
        (!a.saturates || b.saturates)))

/--
Of any two chains, one precedes the other, or equals it, so that sorting by
`le` sorts every list of chains.

# Parameters
- `a`, `b`: The chains.
-/
theorem le_total (a b : Chain) : a.le b = true ∨ b.le a = true := by
  simp only [le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
    beq_iff_eq, Bool.not_eq_true']
  rcases Int.lt_trichotomy a.total b.total with h | h | h
  · exact .inl (.inl h)
  · rcases Std.lt_trichotomy a.dice b.dice with h' | h' | h'
    · exact .inl (.inr ⟨h, .inl h'⟩)
    · rcases Int.lt_trichotomy a.last b.last with h'' | h'' | h''
      · exact .inl (.inr ⟨h, .inr ⟨h', .inl h''⟩⟩)
      · cases ha : a.saturates <;> cases hb : b.saturates <;> simp_all
      · exact .inr (.inr ⟨h.symm, .inr ⟨h'.symm, .inl h''⟩⟩)
    · exact .inr (.inr ⟨h.symm, .inl h'⟩)
  · exact .inr (.inl h)

/--
The order of chains is transitive, so that sorting by `le` sorts every list of
chains.

# Parameters
- `a`, `b`, `c`: The chains.

# Hypotheses
- `hab`: `a` precedes `b`, or equals it.
- `hbc`: `b` precedes `c`, or equals it.
-/
theorem le_trans {a b c : Chain} (hab : a.le b = true) (hbc : b.le c = true) :
    a.le c = true := by
  simp only [le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
    beq_iff_eq, Bool.not_eq_true'] at hab hbc ⊢
  rcases hab with h1 | ⟨e1, h1 | ⟨d1, h1 | ⟨l1, s1⟩⟩⟩ <;>
  rcases hbc with h2 | ⟨e2, h2 | ⟨d2, h2 | ⟨l2, s2⟩⟩⟩
  all_goals first
    | exact .inl (by omega)
    | skip
  all_goals refine .inr ⟨by omega, ?_⟩
  · exact .inl (List.lt_trans h1 h2)
  · exact .inl (d2 ▸ h1)
  · exact .inl (d2 ▸ h1)
  · exact .inl (d1 ▸ h2)
  · exact .inr ⟨d1.trans d2, .inl (by omega)⟩
  · exact .inr ⟨d1.trans d2, .inl (by omega)⟩
  · exact .inl (d1 ▸ h2)
  · exact .inr ⟨d1.trans d2, .inl (by omega)⟩
  · refine .inr ⟨d1.trans d2, .inr ⟨by omega, ?_⟩⟩
    rcases s1 with s1 | s1
    · exact .inl s1
    · rcases s2 with s2 | s2
      · simp [s1] at s2
      · exact .inr s2

/--
A chain that precedes another totals no more than it, so that the totals of
chains sorted by `le` are sorted.

# Parameters
- `a`, `b`: The chains.

# Hypotheses
- `h`: `a` precedes `b`, or equals it.
-/
theorem total_le_of_le {a b : Chain} (h : a.le b = true) :
    a.total ≤ b.total := by
  simp only [le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
    beq_iff_eq] at h
  omega

/--
Whether an explode that matches a criterion grows a chain: whether the
chain may still grow, and the criterion matches its last die.

# Parameters
- `c`: The chain.
- `crit`: The criterion.

# Returns
`true` if an explode by `crit` extends `c`, `false` otherwise.

# Examples
```lean
#guard (Chain.live [1] 6).grows (.within 6 6)
#guard !(Chain.live [6] 1).grows (.within 6 6)
#guard !(Chain.saturated i32Max).grows (.within i32Min i32Max)
```
-/
def grows (c : Chain) (crit : Criterion) : Bool :=
  !c.saturates && crit.matches c.last

/--
Saturate a chain, as an unlimited explode whose criterion matches every face
of the die saturates every chain still kept, appending a saturated die to its
end.

# Parameters
- `value`: The saturated value of the die.
- `c`: The chain.

# Returns
The saturated chain of the value, or `c` itself if it has saturated already,
since an explode leaves a saturated chain alone.

# Examples
```lean
#guard (Chain.live [6] 6).saturate i32Max == .saturated i32Max
#guard (Chain.saturated 0).saturate i32Max == .saturated 0
```
-/
def saturate (value : Int) : Chain → Chain
  | .live .. => .saturated value
  | c => c

end Chain

/-! ## Records -/

/-- The results of a range or a set of dice, the drops applied to them, and
the die from which they were drawn, with the chains that an explode grew, if
any. Equality is decided, not merely a `BEq`, so that it is lawful, as proofs
about a `Dist` of records require. -/
structure Record where
  /-- The results, sorted in ascending order: the totals of the chains, if the
  record holds any. -/
  results : List Int := []
  /-- The number of lowest results dropped, at most the number of results. -/
  lowest : Nat := 0
  /-- The number of highest results dropped, at most the number of results. -/
  highest : Nat := 0
  /-- The die from which the results were drawn. -/
  die : Die := .none
  /-- The chains, sorted by `Chain.le`, one for each result, which is its
  total; or none, if every chain is a single die, which is its result. -/
  chains : List Chain := []
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

/--
Answer the chains that survive the drops, as `kept` answers their totals.

# Parameters
- `r`: The record.

# Returns
The kept chains, sorted by `Chain.le`: the chains in the kept window, if `r`
holds chains, or else the chain of each kept result, a single die. Empty if the
drops cover every chain.

# Examples
```lean
#guard ({ results := [2, 7], highest := 1
          chains := [.live [] 2, .live [1] 6] } : Record).keptChains
  == [.single 2]
#guard ({ results := [1, 3], lowest := 1 } : Record).keptChains
  == [.single 3]
```
-/
def keptChains (r : Record) : List Chain :=
  if r.chains.isEmpty then r.kept.map .single
  else (r.chains.drop r.lowest).take (r.results.length - r.lowest - r.highest)

/--
Build a record of chains, without drops, as a reroll or an explode leaves its
chains still kept.

# Parameters
- `die`: The die from which the dice of the chains were drawn.
- `cs`: The chains, in any order.

# Returns
The record whose results are the totals of the chains, sorted by
`Chain.le`, and which holds the chains, unless every chain is a single die.

# Examples
```lean
#guard Record.ofChains .none [.single 3, .single 1]
  == { results := [1, 3] }
#guard Record.ofChains .none [.live [6] 2, .single 4]
  == { results := [4, 8], chains := [.single 4, .live [6] 2] }
```
-/
def ofChains (die : Die) (cs : List Chain) : Record :=
  let cs := cs.mergeSort Chain.le
  { results := cs.map Chain.total, die
    chains := if cs.all Chain.isSingle then [] else cs }

end Record

end Xdy
