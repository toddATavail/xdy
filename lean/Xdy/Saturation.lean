import Xdy.Record

/-!
# Saturating folds

Lemma 3 of the design: closed forms for the saturating fold that sums a
rolling record, and the soundness of the test, `folds_to_clamp` in
`xdy/src/distribution/propagation/record.rs`, by which the Rust forward pass
replaces that fold with one clamp of the true sum.

`Record.sum` folds the kept results in ascending order with saturating
addition, so every negative result comes before every positive one. While the
fold adds values of one sign, its partial sums move in one direction, so it
saturates at most once, and saturating along the way then equals clamping at
the end. Across the change of sign, it can differ from one clamp: the
negatives may saturate at `i32Min`, and the positives then climb back from
there rather than from their true sum. So the fold of a sorted list is
`clamp (clamp S⁻ + S⁺)`, where `S⁻` is the sum of the negatives and `S⁺` the
sum of the rest: the design's `min(MAX, max(MIN, S⁻) + S⁺)`.

`folds_to_clamp` in `xdy/src/distribution/propagation/record.rs` convolves the
dice and clamps the true sum once, rather than folding, when every face shares
a sign, or when even `count` copies of the least face stay within `i32`.
`Record.sum_eq_clamp` proves that sound.

These are statements about integers alone, so they need no Mathlib, and they
live beside the oracle rather than with the specification.
-/

namespace Xdy

/-! ## Folds of one sign -/

/--
The sum of nonnegative values is nonnegative.

# Parameters
- `xs`: The values.

# Hypotheses
- `hxs`: Every value is nonnegative.
-/
private theorem sum_nonneg (xs : List Int) (hxs : ∀ x ∈ xs, 0 ≤ x) :
    0 ≤ xs.sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have := hxs x (by simp)
    have := ih fun y hy => hxs y (by simp [hy])
    simp; omega

/--
The sum of nonpositive values is nonpositive.

# Parameters
- `xs`: The values.

# Hypotheses
- `hxs`: Every value is nonpositive.
-/
private theorem sum_nonpos (xs : List Int) (hxs : ∀ x ∈ xs, x ≤ 0) :
    xs.sum ≤ 0 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have := hxs x (by simp)
    have := ih fun y hy => hxs y (by simp [hy])
    simp; omega

/--
A saturating fold of values of one sign, from a start within the `i32` range,
equals one clamp of the true sum: its partial sums move in one direction, so
it saturates at most once.

# Parameters
- `xs`: The values.
- `a`: The start of the fold.

# Hypotheses
- `hxs`: The values are either all nonnegative or all nonpositive.
- `lo`, `hi`: The start lies within the `i32` range.
-/
theorem foldl_add_from (xs : List Int) (a : Int)
    (hxs : (∀ x ∈ xs, 0 ≤ x) ∨ (∀ x ∈ xs, x ≤ 0))
    (lo : i32Min ≤ a) (hi : a ≤ i32Max) :
    xs.foldl add a = clamp (a + xs.sum) := by
  induction xs generalizing a with
  | nil => simp only [clamp, i32Min, i32Max] at *; simp; omega
  | cons x xs ih =>
    have signs : 0 ≤ x ∧ 0 ≤ xs.sum ∨ x ≤ 0 ∧ xs.sum ≤ 0 := by
      rcases hxs with h | h
      · exact .inl ⟨h x (by simp), sum_nonneg xs fun y hy => h y (by simp [hy])⟩
      · exact .inr ⟨h x (by simp), sum_nonpos xs fun y hy => h y (by simp [hy])⟩
    have tail : (∀ y ∈ xs, 0 ≤ y) ∨ (∀ y ∈ xs, y ≤ 0) := by
      rcases hxs with h | h
      · exact .inl fun y hy => h y (by simp [hy])
      · exact .inr fun y hy => h y (by simp [hy])
    rw [List.foldl_cons, List.sum_cons, ih _ tail]
      <;> simp only [add, clamp, i32Min, i32Max] at signs lo hi ⊢ <;> omega

/--
The saturating fold of values of one sign equals one clamp of their sum.

# Parameters
- `xs`: The values, in any order.

# Hypotheses
- `hxs`: The values are either all nonnegative or all nonpositive.

# Examples
```lean
example : [i32Max, 1].foldl add 0 = clamp (i32Max + 1) := by decide
```
-/
theorem foldl_add_of_same_sign (xs : List Int)
    (hxs : (∀ x ∈ xs, 0 ≤ x) ∨ (∀ x ∈ xs, x ≤ 0)) :
    xs.foldl add 0 = clamp xs.sum := by
  rw [foldl_add_from xs 0 hxs (by decide) (by decide), Int.zero_add]

/-! ## Folds of mixed signs -/

/--
A list sorted in ascending order is its negative values followed by the rest.

# Parameters
- `xs`: The values.

# Hypotheses
- `sorted`: The values are sorted in ascending order.
-/
private theorem filter_neg_append_filter_nonneg (xs : List Int)
    (sorted : xs.Pairwise (· ≤ ·)) :
    xs.filter (· < 0) ++ xs.filter (0 ≤ ·) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.pairwise_cons] at sorted
    by_cases hx : x < 0
    · simp [hx, Int.not_le.mpr hx, ih sorted.2]
    · have hge : ∀ y ∈ xs, 0 ≤ y := fun y hy => by
        have := sorted.1 y hy; omega
      have hneg : xs.filter (· < 0) = [] :=
        List.filter_eq_nil_iff.mpr fun y hy => by
          have := hge y hy; simp; omega
      have hnonneg : xs.filter (0 ≤ ·) = xs :=
        List.filter_eq_self.mpr fun y hy => by simpa using hge y hy
      simp [hx, Int.not_lt.mp hx, hneg, hnonneg]

/--
The saturating fold of values sorted in ascending order equals
`clamp (clamp S⁻ + S⁺)`, where `S⁻` is the sum of the negative values and
`S⁺` the sum of the rest: the negatives saturate at most once, at `i32Min`,
and the rest then climb from there, saturating at most once, at `i32Max`.

# Parameters
- `xs`: The values.

# Hypotheses
- `sorted`: The values are sorted in ascending order, as `Record.sum` folds
  them.

# Notes
Since `clamp S⁻ + S⁺` is never less than `i32Min`, this is the design's
`min(MAX, max(MIN, S⁻) + S⁺)`.

# Examples
```lean
-- The negatives saturate, and the positives climb back from `i32Min`.
example : [i32Min, -1, i32Max].foldl add 0 = i32Min + i32Max := by decide
```
-/
theorem foldl_add_of_sorted (xs : List Int) (sorted : xs.Pairwise (· ≤ ·)) :
    xs.foldl add 0 =
      clamp (clamp (xs.filter (· < 0)).sum + (xs.filter (0 ≤ ·)).sum) := by
  conv => lhs; rw [← filter_neg_append_filter_nonneg xs sorted]
  have hneg : ∀ x ∈ xs.filter (· < 0), x ≤ 0 := fun x hx => by
    have := (List.mem_filter.mp hx).2; simp at this; omega
  have hnonneg : ∀ x ∈ xs.filter (0 ≤ ·), 0 ≤ x := fun x hx => by
    simpa using (List.mem_filter.mp hx).2
  rw [List.foldl_append, foldl_add_from _ 0 (.inr hneg) (by decide) (by decide),
    foldl_add_from _ _ (.inl hnonneg)
      (by simp only [clamp, i32Min, i32Max]; omega)
      (by simp only [clamp, i32Min, i32Max]; omega),
    Int.zero_add]

/--
The saturating fold of values sorted in ascending order equals one clamp of
their sum if their negatives sum to no less than `i32Min`, since then the
negatives never saturate.

# Parameters
- `xs`: The values.

# Hypotheses
- `sorted`: The values are sorted in ascending order.
- `hneg`: The negative values sum to at least `i32Min`.
-/
theorem foldl_add_eq_clamp (xs : List Int) (sorted : xs.Pairwise (· ≤ ·))
    (hneg : i32Min ≤ (xs.filter (· < 0)).sum) :
    xs.foldl add 0 = clamp xs.sum := by
  have hs : (xs.filter (· < 0)).sum ≤ 0 := sum_nonpos _ fun x hx => by
    have := (List.mem_filter.mp hx).2; simp at this; omega
  have hsum : xs.sum = (xs.filter (· < 0)).sum + (xs.filter (0 ≤ ·)).sum := by
    conv => lhs; rw [← filter_neg_append_filter_nonneg xs sorted]
    exact List.sum_append
  rw [foldl_add_of_sorted xs sorted, hsum]
  simp only [clamp, i32Min, i32Max] at *; omega

/--
A list of values, each at least `least`, sums to at least its length times
`least`.

# Parameters
- `ys`: The values.
- `least`: The bound.

# Hypotheses
- `h`: Every value is at least `least`.
-/
private theorem length_mul_le_sum (ys : List Int) (least : Int)
    (h : ∀ y ∈ ys, least ≤ y) : ys.length * least ≤ ys.sum := by
  induction ys with
  | nil => simp
  | cons y ys ih =>
    have := h y (by simp)
    have := ih fun z hz => h z (by simp [hz])
    rw [List.length_cons, List.sum_cons, Int.natCast_add, Int.add_mul]
    omega

/-! ## Rolling records -/

namespace Record

/--
The kept results of a record whose results are sorted are sorted too, since
they are a contiguous run of them.

# Parameters
- `r`: The record.

# Hypotheses
- `sorted`: The results are sorted in ascending order.
-/
theorem kept_sorted (r : Record) (sorted : r.results.Pairwise (· ≤ ·)) :
    r.kept.Pairwise (· ≤ ·) :=
  sorted.sublist ((List.take_sublist _ _).trans (List.drop_sublist _ _))

/--
The sum of a record whose results are sorted is `clamp (clamp S⁻ + S⁺)`,
where `S⁻` is the sum of the negative kept results and `S⁺` the sum of the
rest. See `foldl_add_of_sorted`.

# Parameters
- `r`: The record.

# Hypotheses
- `sorted`: The results are sorted in ascending order, as `push` keeps them.
-/
theorem sum_eq_of_sorted (r : Record) (sorted : r.results.Pairwise (· ≤ ·)) :
    r.sum =
      clamp (clamp (r.kept.filter (· < 0)).sum + (r.kept.filter (0 ≤ ·)).sum) :=
  foldl_add_of_sorted r.kept (r.kept_sorted sorted)

/--
The sum of a record without drops equals one clamp of the true sum of its
results under the conditions that `folds_to_clamp` in
`xdy/src/distribution/propagation/record.rs` tests: every result is at least
`least` and at most `greatest`, and either `least` is nonnegative, `greatest`
is nonpositive, or the number of results times `least` is no less than
`i32Min`.

# Parameters
- `r`: The record.
- `least`: A lower bound on the results, such as the least face of the die.
- `greatest`: An upper bound on the results, such as the greatest face of the
  die.

# Hypotheses
- `sorted`: The results are sorted in ascending order, as `push` keeps them.
- `lowest`, `highest`: The record drops none of its results.
- `hleast`: Every result is at least `least`.
- `hgreatest`: Every result is at most `greatest`.
- `test`: The test of `folds_to_clamp` passes: `least` is nonnegative,
  `greatest` is nonpositive, or the number of results times `least` is no
  less than `i32Min`.

# Notes
The Rust tests the least and greatest faces of one die and the number of
dice, which bound every roll of those dice, so the test is sound for every
roll that the convolution power weighs.
-/
theorem sum_eq_clamp (r : Record) (sorted : r.results.Pairwise (· ≤ ·))
    (lowest : r.lowest = 0) (highest : r.highest = 0) (least greatest : Int)
    (hleast : ∀ x ∈ r.results, least ≤ x)
    (hgreatest : ∀ x ∈ r.results, x ≤ greatest)
    (test : 0 ≤ least ∨ greatest ≤ 0 ∨ i32Min ≤ r.results.length * least) :
    r.sum = clamp r.results.sum := by
  have kept : r.kept = r.results := by simp [kept, lowest, highest]
  simp only [sum, kept]
  rcases test with h | h | h
  · exact foldl_add_of_same_sign _ (.inl fun x hx => by
      have := hleast x hx; omega)
  · exact foldl_add_of_same_sign _ (.inr fun x hx => by
      have := hgreatest x hx; omega)
  · refine foldl_add_eq_clamp _ sorted ?_
    -- Each negative result is at least `least`, and there are no more of
    -- them than there are results.
    have hle := length_mul_le_sum (r.results.filter (· < 0)) least
      fun y hy => hleast y (List.mem_filter.mp hy).1
    have hlen := List.length_filter_le (· < 0) r.results
    simp only [i32Min] at h ⊢
    by_cases hl : 0 ≤ least
    · have : 0 ≤ (r.results.filter (· < 0)).length * least :=
        Int.mul_nonneg (by omega) hl
      omega
    · have : (r.results.length : Int) * least
          ≤ (r.results.filter (· < 0)).length * least :=
        Int.mul_le_mul_of_nonpos_right (by omega) (by omega)
      omega

end Record

end Xdy
