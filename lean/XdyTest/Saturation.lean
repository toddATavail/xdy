import Xdy.Saturation

/-!
# Saturating fold tests

The closed forms of lemma 3 on concrete folds, including the fold of mixed
signs that one clamp of the true sum gets wrong.
-/

namespace Xdy.Test

open Xdy

/-! ## Folds of one sign -/

#guard [i32Max, 1].foldl add 0 == clamp (i32Max + 1)
#guard [i32Min, -1].foldl add 0 == clamp (i32Min - 1)
#guard ([] : List Int).foldl add 0 == clamp 0

/-! ## Folds of mixed signs -/

-- The negatives saturate at `i32Min`, and the positives climb back from
-- there, so the fold is `-1`, while one clamp of the true sum is `-2`.
#guard [i32Min, -1, i32Max].foldl add 0 == -1
#guard clamp (i32Min - 1 + i32Max) == -2
#guard [i32Min, -1, i32Max].foldl add 0
  == clamp (clamp (i32Min - 1) + i32Max)
-- The negatives stay within range, so one clamp of the true sum is exact.
#guard [-3, 1, i32Max].foldl add 0 == clamp (-3 + 1 + i32Max)

/-! ## Rolling records -/

-- Three copies of the least face `-2` stay within range, so the sum of any
-- roll of three dice with faces `[-2, 5]` is one clamp of its true sum.
example (r : Record) (sorted : r.results.Pairwise (· ≤ ·))
    (lowest : r.lowest = 0) (highest : r.highest = 0)
    (faces : ∀ x ∈ r.results, -2 ≤ x ∧ x ≤ 5) (count : r.results.length = 3) :
    r.sum = clamp r.results.sum :=
  r.sum_eq_clamp sorted lowest highest (-2) 5
    (fun x hx => (faces x hx).1) (fun x hx => (faces x hx).2)
    (.inr (.inr (by rw [count]; decide)))

/-! ## Docstring examples, verbatim -/

example : [i32Max, 1].foldl add 0 = clamp (i32Max + 1) := by decide
example : [i32Min, -1, i32Max].foldl add 0 = i32Min + i32Max := by decide

end Xdy.Test
