import XdySpec.Mixture

/-!
# Order statistics

The sum of a roll of dice with drops, as `order_statistics` in
`xdy/src/distribution/propagation/record.rs` computes it: a dynamic program
over the distinct faces of the die, in ascending order. Its state after some
faces holds, for each number `j` of sorted positions that they fill, the
weight of each saturating fold `s` of the kept dice among those positions.
Placing `m` copies of the next face `v`, of weight `w`, after `j` positions
keeps those of positions `j` to `j + m - 1` that lie within the kept window,
adds them to `s` with one clamp, and weighs `C(j + m, m) · wᵐ`: the rolls that
sort to the same multiset of faces are its permutations, and over all faces
the binomials telescope to the multinomial coefficient. This module models
the program (`orderStates`, `orderStatistics`) and proves that its weights,
over the total weight of the die raised to the number of dice, are the law of
the sum of the kept dice (`map_sum_rollDice_order`), for custom dice
(`Setup.map_sum_law_custom_order`) and standard dice
(`Setup.map_sum_law_standard_order`) alike.

The proof never counts permutations. It weighs every roll by the product of
its faces' weights, sorted as it is rolled, die by die (`words`), and then:

```mermaid
flowchart TD
    W["words: every roll, weighed,<br/>sorted die by die"]
    P["words_peel: peel the greatest face,<br/>C(J, m) · wᵐ, by Pascal's rule"]
    I["integral_orderStates: the states are<br/>the words, through the window"]
    L["tsum_rollDice: the law of a roll, sorted,<br/>is the words over Wᴶ"]
    T["map_sum_rollDice_order: the sum's law<br/>is the weights over Wⁿ"]
    W --> P --> I --> T
    W --> L --> T
```

The Rust places only all the remaining copies of the greatest face, since the
states that it leaves short of `n` dice are never read; the model computes
them anyway, and reads the state of `n` dice, which is the same. It keeps its
weights in lists of outcomes and weights, merging nothing, where the Rust
merges equal sums; a sum's weight is the total of its entries (`weight`).
-/

open scoped ENNReal

namespace Xdy.Spec

namespace OrderStatistics

/--
Insert a value into a sorted list, keeping it sorted, as sorting a record
after it rolls one more die places the die.

# Parameters
- `x`: The value.
- `ys`: The list, in ascending order.

# Returns
The list with `x` before its first value no less than `x`.
-/
abbrev ins (x : Int) (ys : List Int) : List Int := ys.orderedInsert (· ≤ ·) x

/--
The weighed rolls of a die: the sum, over every roll of `J` dice, in order, of
the product of their faces' weights times `φ` of the roll sorted. The rolls
sort die by die, as `sorted_push` proves a record does.

# Parameters
- `die`: The faces of the die, each with its weight.
- `J`: The number of dice.
- `φ`: The function of the sorted roll.

# Returns
The weighed sum.

# Examples
```lean
#guard OrderStatistics.words [(1, 1), (2, 3)] 2 (fun ys => ys.sum.toNat) == 56
```
-/
def words (die : List (Int × Nat)) : Nat → (List Int → Nat) → Nat
  | 0, φ => φ []
  | J + 1, φ =>
    (die.map fun p => p.2 * words die J (fun ys => φ (ins p.1 ys))).sum

/--
No dice roll once, empty.

# Parameters
- `die`: The die.
- `φ`: The function of the sorted roll.
-/
theorem words_zero (die : List (Int × Nat)) (φ : List Int → Nat) :
    words die 0 φ = φ [] := rfl

/--
One more die rolls each face after the others.

# Parameters
- `die`: The die.
- `J`: The number of dice before it.
- `φ`: The function of the sorted roll.
-/
theorem words_succ (die : List (Int × Nat)) (J : Nat) (φ : List Int → Nat) :
    words die (J + 1) φ =
      (die.map fun p => p.2 * words die J (fun ys => φ (ins p.1 ys))).sum := rfl

/--
The weighed rolls read only rolls of the die's faces.

# Parameters
- `J`: The number of dice.

# Hypotheses
- `hdie`: Every face of the die satisfies `P`.
- `h`: `φ` and `ψ` agree on every list of values that satisfy `P`.
-/
theorem words_congr {die : List (Int × Nat)} {P : Int → Prop}
    (hdie : ∀ p ∈ die, P p.1) :
    ∀ (J : Nat) {φ ψ : List Int → Nat},
      (∀ ys : List Int, (∀ y ∈ ys, P y) → φ ys = ψ ys) →
      words die J φ = words die J ψ
  | 0, φ, ψ, h => h [] (by simp)
  | J + 1, φ, ψ, h => by
    simp only [words_succ]
    refine congrArg List.sum (List.map_congr_left fun p hp => ?_)
    refine congrArg (p.2 * ·) (words_congr hdie J fun ys hys => h _ ?_)
    intro y hy
    rcases List.mem_cons.mp ((List.perm_orderedInsert _ _ _).mem_iff.mp hy)
      with rfl | hy
    · exact hdie p hp
    · exact hys y hy

/-- Inserting a value no greater than the last block of copies inserts it
before them. -/
theorem ins_append_replicate {x v : Int} (h : x ≤ v) (ys : List Int)
    (m : Nat) :
    ins x (ys ++ List.replicate m v) = ins x ys ++ List.replicate m v := by
  induction ys with
  | nil =>
    cases m with
    | zero => simp [ins]
    | succ m => simp [ins, List.replicate_succ, List.orderedInsert, h]
  | cons y ys ih =>
    simp only [ins, List.cons_append, List.orderedInsert] at ih ⊢
    split_ifs <;> simp [ih]

/-- Inserting a value no less than every value of a list appends it. -/
theorem ins_greatest {v : Int} (ys : List Int) (hys : ∀ y ∈ ys, y < v)
    (m : Nat) :
    ins v (ys ++ List.replicate m v) = ys ++ List.replicate (m + 1) v := by
  induction ys with
  | nil =>
    cases m with
    | zero => simp [ins]
    | succ m => simp [ins, List.replicate_succ]
  | cons y ys ih =>
    have hy := hys y (by simp)
    simp only [ins, List.cons_append, List.orderedInsert] at ih ⊢
    rw [ih fun z hz => hys z (by simp [hz])]
    simp only [show ¬ v ≤ y by omega, ↓reduceIte]

/--
A list's sum of finite sums is the finite sum of its sums.

# Parameters
- `l`: The list.
- `s`: The finite set.
- `f`: The summand.
-/
theorem list_sum_finset_sum {β : Type} (l : List β) (s : Finset Nat)
    (f : β → Nat → Nat) :
    (l.map fun p => ∑ i ∈ s, f p i).sum =
      ∑ i ∈ s, (l.map fun p => f p i).sum := by
  induction l with
  | nil => simp
  | cons p l ih => simp [ih, Finset.sum_add_distrib]

/--
Pascal's rule, summed: placing the copies of a face among `J` dice and then
one more, or the one more first, places them among `J + 1`.

# Parameters
- `w`: The weight of the face.
- `J`: The number of dice.
- `a`: The weighed rolls of the rest, by the number of copies.
-/
theorem pascal_sum (w J : Nat) (a : Nat → Nat) :
    ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m * a m +
      ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ (m + 1) * a (m + 1) =
    ∑ m ∈ Finset.range (J + 2), (J + 1).choose m * w ^ m * a m := by
  rw [Finset.sum_range_succ' _ (J + 1)]
  simp only [Nat.choose_succ_succ', Nat.add_mul, Finset.sum_add_distrib]
  rw [Finset.sum_range_succ' (fun m => J.choose m * w ^ m * a m)]
  have hz :
      ∑ m ∈ Finset.range (J + 1), J.choose (m + 1) * w ^ (m + 1) * a (m + 1)
      = ∑ m ∈ Finset.range J, J.choose (m + 1) * w ^ (m + 1) * a (m + 1) := by
    rw [Finset.sum_range_succ, Nat.choose_succ_self, Nat.zero_mul, Nat.zero_mul,
      Nat.add_zero]
  rw [hz]
  simp only [Nat.choose_zero_right, pow_zero]
  ring

/--
Peeling the greatest face: rolling a die whose greatest face is `v`, of weight
`w`, rolls `m` copies of `v` at `C(J, m) · wᵐ` ways, and the rest of the dice
from the other faces, sorted beneath the copies. This is the step of the
program, from the faces beneath `v` to `v`.

# Parameters
- `w`: The weight of the greatest face.
- `J`: The number of dice.
- `φ`: The function of the sorted roll.

# Hypotheses
- `hv`: Every other face is less than `v`.
-/
theorem words_peel {die : List (Int × Nat)} {v : Int} (w : Nat)
    (hv : ∀ p ∈ die, p.1 < v) :
    ∀ (J : Nat) (φ : List Int → Nat),
      words (die ++ [(v, w)]) J φ =
        ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          words die (J - m) (fun ys => φ (ys ++ List.replicate m v))
  | 0, φ => by simp [words_zero]
  | J + 1, φ => by
    have hle : ∀ p ∈ die, p.1 ≤ v := fun p hp => (hv p hp).le
    -- Each lower face inserts beneath the copies of `v`, and `v` atop them.
    have hlow : ∀ p ∈ die, words (die ++ [(v, w)]) J (fun ys => φ (ins p.1 ys))
        = ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          words die (J - m)
            (fun ys => φ (ins p.1 ys ++ List.replicate m v)) := by
      intro p hp
      rw [words_peel w hv J]
      refine Finset.sum_congr rfl fun m _ => ?_
      simp only [ins_append_replicate (hle p hp)]
    have htop : words (die ++ [(v, w)]) J (fun ys => φ (ins v ys))
        = ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          words die (J - m) (fun ys => φ (ys ++ List.replicate (m + 1) v)) := by
      rw [words_peel w hv J]
      refine Finset.sum_congr rfl fun m _ => congrArg _ ?_
      exact words_congr (P := (· < v)) hv _ fun ys hys => by
        rw [ins_greatest ys hys]
    rw [words_succ, List.map_append, List.sum_append, List.map_singleton,
      List.sum_singleton, htop]
    rw [List.map_congr_left fun p hp => congrArg (p.2 * ·) (hlow p hp)]
    simp only [Finset.mul_sum]
    rw [list_sum_finset_sum]
    -- The lower faces fold back into the words of one more die.
    have hfold : ∀ i ∈ Finset.range (J + 1),
        (die.map fun p => p.2 * (J.choose i * w ^ i *
          words die (J - i) fun ys => φ (ins p.1 ys ++ List.replicate i v))).sum
        = J.choose i * w ^ i *
          words die (J + 1 - i) (fun ys => φ (ys ++ List.replicate i v)) := by
      intro i hi
      have hi : i ≤ J := Nat.lt_succ_iff.mp (Finset.mem_range.mp hi)
      rw [show J + 1 - i = (J - i) + 1 by omega, words_succ,
        ← List.sum_map_mul_left]
      refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
      ring
    rw [Finset.sum_congr rfl hfold]
    have hshift : ∀ i ∈ Finset.range (J + 1),
        w * (J.choose i * w ^ i *
          words die (J - i) fun ys => φ (ys ++ List.replicate (i + 1) v))
        = J.choose i * w ^ (i + 1) *
          words die (J + 1 - (i + 1))
            (fun ys => φ (ys ++ List.replicate (i + 1) v))
        := by
      intro i _
      rw [show J + 1 - (i + 1) = J - i by omega]
      ring
    rw [Finset.sum_congr rfl hshift]
    exact pascal_sum w J fun m =>
      words die (J + 1 - m) (fun ys => φ (ys ++ List.replicate m v))

/-! ## The program -/

/--
Integrate a function against weighted outcomes: the sum of each outcome's
weight times its value.

# Parameters
- `l`: The outcomes, each with its weight, possibly repeated.
- `φ`: The function.

# Returns
The integral.
-/
def integral {α : Type} (l : List (α × Nat)) (φ : α → Nat) : Nat :=
  (l.map fun e => e.2 * φ e.1).sum

/--
Integrating against two lists of outcomes adds.

# Parameters
- `l`, `l'`: The outcomes.
- `φ`: The function.
-/
theorem integral_append {α : Type} (l l' : List (α × Nat)) (φ : α → Nat) :
    integral (l ++ l') φ = integral l φ + integral l' φ := by
  simp [integral]

/--
Integrating against a list of lists of outcomes sums their integrals.

# Parameters
- `l`: The list.
- `f`: The outcomes of each element.
- `φ`: The function.
-/
theorem integral_flatMap {α β : Type} (l : List β) (f : β → List (α × Nat))
    (φ : α → Nat) :
    integral (l.flatMap f) φ = (l.map fun b => integral (f b) φ).sum := by
  induction l with
  | nil => rfl
  | cons b l ih => simp [List.flatMap_cons, integral_append, ih]

/--
Transforming the outcomes and scaling the weights transforms the function and
scales the integral.

# Parameters
- `l`: The outcomes.
- `f`: The transformation.
- `c`: The scale.
- `φ`: The function.
-/
theorem integral_map {α β : Type} (l : List (α × Nat)) (f : α → β) (c : Nat)
    (φ : β → Nat) :
    integral (l.map fun e => (f e.1, c * e.2)) φ =
      c * integral l (fun a => φ (f a)) := by
  simp only [integral, List.map_map, ← List.sum_map_mul_left]
  refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
  simp only [Function.comp_apply]
  ring

/--
A sum over `List.range` is a sum over `Finset.range`.

# Parameters
- `n`: The bound.
- `f`: The summand.
-/
theorem list_range_sum (n : Nat) (f : Nat → Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih,
      Finset.sum_range_succ]; simp

/--
The saturating fold of the kept positions of a sorted roll, as `Record.sum`
folds them: those from `lo` on, at most `len` of them. A roll of fewer dice
than the record holds is folded as far as it goes.

# Parameters
- `lo`: The first kept position: the number of lowest dice dropped.
- `len`: The number of kept positions.
- `xs`: The sorted roll.

# Returns
The fold.

# Examples
```lean
#guard OrderStatistics.window 1 2 [1, 3, 4, 6] == 7
```
-/
def window (lo len : Nat) (xs : List Int) : Int :=
  ((xs.drop lo).take len).foldl add 0

/--
The copies of a face kept when it fills positions `j` to `j + m - 1`, as
`order_statistics` counts them.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `j`: The positions filled before the face.
- `m`: The copies of the face.

# Returns
The number of the copies' positions within the kept window.
-/
def placed (lo len j m : Nat) : Nat :=
  Min.min (lo + len) (j + m) - Max.max lo j

/--
Place every number of copies of one face, of value `v` and weight `w`. Mirrors
the loop over the faces of `order_statistics` in `record.rs`: the state of `J`
positions gathers, for each number `m` of copies, the states of `J - m`
positions, each sum moved to the fold of the copies kept and weighed by
`C(J, m) · wᵐ`, which is the Rust's `C(j + m, m) · wᵐ` with `j = J - m`.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `states`: The states before the face: for each number of positions, the
  sums and their weights.
- `face`: The face and its weight.

# Returns
The states after the face.
-/
def orderStep (lo len : Nat) (states : Nat → List (Int × Nat))
    (face : Int × Nat) : Nat → List (Int × Nat) :=
  fun J => (List.range (J + 1)).flatMap fun m =>
    (states (J - m)).map fun e =>
      (clamp (e.1 + placed lo len (J - m) m * face.1),
        J.choose m * face.2 ^ m * e.2)

/-- The states before any face: no positions filled, summing to `0` once. -/
def initial : Nat → List (Int × Nat) :=
  fun j => if j = 0 then [(0, 1)] else []

/--
The states of the program after every face of a die, in order.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `die`: The distinct faces, in ascending order, each with its weight.

# Returns
The states.
-/
def orderStates (lo len : Nat) (die : List (Int × Nat)) :
    Nat → List (Int × Nat) :=
  die.foldl (orderStep lo len) initial

/-- A saturating fold from within the `i32` range stays within it. -/
private theorem foldl_add_bounds (xs : List Int) {a : Int} (lo : i32Min ≤ a)
    (hi : a ≤ i32Max) :
    i32Min ≤ xs.foldl add a ∧ xs.foldl add a ≤ i32Max := by
  induction xs generalizing a with
  | nil => exact ⟨lo, hi⟩
  | cons x xs ih =>
    exact ih (by simp only [add, clamp, i32Min, i32Max]; omega)
      (by simp only [add, clamp, i32Min, i32Max]; omega)

/--
Adding copies of one value after a roll adds those that are kept with one
clamp, since they saturate at most once.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `ys`: The roll.
- `m`: The copies.
- `v`: The value.
-/
theorem window_append (lo len : Nat) (ys : List Int) (m : Nat) (v : Int) :
    window lo len (ys ++ List.replicate m v) =
      clamp (window lo len ys + placed lo len ys.length m * v) := by
  unfold window
  have hb := foldl_add_bounds ((ys.drop lo).take len) (a := 0) (by decide)
    (by decide)
  rw [List.drop_append, List.drop_replicate, List.take_append,
    List.take_replicate, List.foldl_append,
    foldl_add_from _ _ (by
      by_cases h : 0 ≤ v
      · exact .inl fun x hx => (List.eq_of_mem_replicate hx) ▸ h
      · exact .inr fun x hx => (List.eq_of_mem_replicate hx) ▸ by omega)
      hb.1 hb.2,
    List.sum_replicate, nsmul_eq_mul]
  simp only [List.length_drop, placed]
  congr 3
  omega

/--
The weighed rolls of `J` dice read only rolls of `J` dice.

# Parameters
- `die`: The die.
- `J`: The number of dice.

# Hypotheses
- `h`: `φ` and `ψ` agree on every list of length `J`.
-/
theorem words_congr_length (die : List (Int × Nat)) :
    ∀ (J : Nat) {φ ψ : List Int → Nat},
      (∀ ys : List Int, ys.length = J → φ ys = ψ ys) →
      words die J φ = words die J ψ
  | 0, φ, ψ, h => h [] rfl
  | J + 1, φ, ψ, h => by
    simp only [words_succ]
    refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
    refine congrArg (p.2 * ·) (words_congr_length die J fun ys hys => h _ ?_)
    simp [ins, List.orderedInsert_length, hys]

/--
One more face takes one more step.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `die`: The faces before it.
- `face`: The face.
-/
theorem orderStates_append (lo len : Nat) (die : List (Int × Nat))
    (face : Int × Nat) :
    orderStates lo len (die ++ [face]) =
      orderStep lo len (orderStates lo len die) face := by
  simp [orderStates, List.foldl_append]

/--
The program's state of `J` positions, integrated against `ψ`, is the weighed
rolls of `J` dice, read through the window: each step peels the greatest face
of the die so far, as `words_peel` does.

# Parameters
- `lo`: The first kept position.
- `len`: The number of kept positions.
- `die`: The faces.

# Hypotheses
- The faces ascend strictly.
-/
theorem integral_orderStates (lo len : Nat) :
    ∀ (die : List (Int × Nat)), die.Pairwise (fun a b => a.1 < b.1) →
      ∀ (J : Nat) (ψ : Int → Nat),
        integral (orderStates lo len die J) ψ =
          words die J (fun ys => ψ (window lo len ys)) := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro _ J ψ
    cases J with
    | zero => simp [orderStates, initial, integral, words_zero, window]
    | succ J => simp [orderStates, initial, integral, words_succ]
  | append_singleton die face ih =>
    intro hsorted J ψ
    rw [List.pairwise_append] at hsorted
    obtain ⟨hdie, _, hlt⟩ := hsorted
    have hv : ∀ p ∈ die, p.1 < face.1 := fun p hp => hlt p hp face (by simp)
    obtain ⟨v, w⟩ := face
    rw [orderStates_append, words_peel w hv]
    simp only [orderStep, integral_flatMap, list_range_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [integral_map _ (fun a => clamp (a + placed lo len (J - m) m * v)),
      ih hdie]
    refine congrArg _ (words_congr_length die _ fun ys hys => ?_)
    rw [window_append, hys]

/-! ## The law -/

/--
A sum against a bind is a sum of sums against its continuations.

# Parameters
- `p`: The first law.
- `k`: The continuation.
- `f`: The function summed.
-/
theorem tsum_bind_mul {α β : Type} (p : PMF α) (k : α → PMF β) (f : β → ℝ≥0∞) :
    ∑' b, (p.bind k) b * f b = ∑' a, p a * ∑' b, k a b * f b := by
  simp only [PMF.bind_apply, ← ENNReal.tsum_mul_right, ← ENNReal.tsum_mul_left]
  rw [ENNReal.tsum_comm]
  exact tsum_congr fun a => tsum_congr fun b => by ring

/--
A sum against a map is a sum of the function, transformed.

# Parameters
- `p`: The law.
- `g`: The transformation.
- `f`: The function summed.
-/
theorem tsum_map_mul {α β : Type} (p : PMF α) (g : α → β) (f : β → ℝ≥0∞) :
    ∑' b, (p.map g) b * f b = ∑' a, p a * f (g a) := by
  simp only [PMF.map_apply, ← ENNReal.tsum_mul_right]
  rw [ENNReal.tsum_comm]
  refine tsum_congr fun a => ?_
  rw [tsum_eq_single (g a) fun b hb => by simp [hb]]
  simp

/--
A sum of a list's sums is the list's sum of sums.

# Parameters
- `l`: The list.
- `g`: The summand of each element.
-/
theorem tsum_list_sum {β γ : Type} (l : List γ) (g : γ → β → ℝ≥0∞) :
    ∑' b, (l.map fun c => g c b).sum = (l.map fun c => ∑' b, g c b).sum := by
  induction l with
  | nil => simp
  | cons c l ih => simp [ENNReal.tsum_add, ih]

/--
Summing over the faces repeated by their weights weighs each face.

# Parameters
- `faces`: The faces, each with its weight.
- `f`: The function summed.
-/
theorem sum_expand (faces : List (Int × Nat)) (f : Int → ℝ≥0∞) :
    ((Roll.expand faces).map f).sum =
      (faces.map fun p => (p.2 : ℝ≥0∞) * f p.1).sum := by
  induction faces with
  | nil => rfl
  | cons p faces ih =>
    simp [Roll.expand, List.flatMap_cons, List.sum_append, ih] at *

/-- Counting in a multiset does not depend on how equality is decided. -/
private theorem count_inst {s : Multiset Int} {x : Int} :
    @Multiset.count Int (fun a b => Classical.propDecidable (a = b)) x s =
      Multiset.count x s := by
  congr

/--
The total weight of a die's faces, the number of ways one die rolls.

# Parameters
- `faces`: The faces, each with its weight.

# Returns
The total.
-/
def total (faces : List (Int × Nat)) : Nat := (faces.map (·.2)).sum

/--
A sum against a die of weighted faces weighs each face by its weight over the
total.

# Parameters
- `faces`: The faces.
- `f`: The function summed.

# Hypotheses
- `hW`: The total weight is positive.
-/
theorem tsum_weightedDie (faces : List (Int × Nat)) (hW : total faces ≠ 0)
    (f : Int → ℝ≥0∞) :
    ∑' x, weightedDie faces x * f x =
      (faces.map fun p => (p.2 : ℝ≥0∞) * f p.1).sum / (total faces : ℝ≥0∞) := by
  have hlen : (Roll.expand faces).length = total faces := by
    simp [Roll.expand, List.length_flatMap, total]
  have hne : Roll.expand faces ≠ [] := fun h => hW (by rw [← hlen, h]; rfl)
  have hdie : weightedDie faces =
      PMF.ofMultiset (Roll.expand faces : Multiset Int)
        (by simpa using hne) := by
    unfold weightedDie customDie
    simp only [show List.flatMap _ faces ≠ [] from hne, ↓reduceDIte]
    rfl
  rw [hdie, ← sum_expand, ← hlen]
  rw [tsum_eq_sum (s := (Roll.expand faces : Multiset Int).toFinset)
    fun x hx => by
      simp only [PMF.ofMultiset_apply, count_inst]
      rw [Multiset.count_eq_zero.mpr (by simpa using hx)]
      simp]
  simp only [PMF.ofMultiset_apply, count_inst, ENNReal.div_eq_inv_mul]
  have : ((Roll.expand faces).map f).sum =
      ((Roll.expand faces : Multiset Int).map f).sum := by simp
  rw [this, Finset.sum_multiset_map_count, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [nsmul_eq_mul, mul_assoc, Multiset.coe_card]

/--
Sorting a record after one more die inserts the die into the record sorted.

# Parameters
- `r`: The record.
- `x`: The die.
-/
theorem sorted_push (r : Record) (x : Int) :
    (r.push x).sorted = ins x r.sorted := by
  refine List.Perm.eq_of_pairwise (le := (· ≤ ·))
    (fun a b _ _ h1 h2 => le_antisymm h1 h2) (Record.sorted_pairwise _)
    (List.Pairwise.orderedInsert x _ (Record.sorted_pairwise r)) ?_
  refine ((List.mergeSort_perm _ _).trans ?_).trans
    (List.perm_orderedInsert _ _ _).symm
  simp only [Record.push]
  exact (List.perm_append_singleton x r.results).trans
    (List.Perm.cons x (List.mergeSort_perm _ _).symm)

/--
One more die rolls after the others.

# Parameters
- `J`: The number of dice before it.
- `die`: The law of one die.
-/
theorem rollDice_succ (J : Nat) (die : PMF Int) :
    rollDice ((J + 1 : Nat) : Int) die =
      (rollDice (J : Int) die).bind fun r => die.map r.push := by
  simp [rollDice, Nat.repeat]

/--
The law of a roll of `J` dice of weighted faces, sorted, is their weighed
rolls over the total weight raised to `J`: a sum of `φ` of the sorted roll
against it is `words`, over `Wᴶ`.

# Parameters
- `faces`: The faces.
- `J`: The number of dice.
- `φ`: The function of the sorted roll.

# Hypotheses
- `hW`: The total weight is positive.
-/
theorem tsum_rollDice (faces : List (Int × Nat)) (hW : total faces ≠ 0) :
    ∀ (J : Nat) (φ : List Int → Nat),
      ∑' r, rollDice (J : Int) (weightedDie faces) r * (φ r.sorted : ℝ≥0∞) =
        (words faces J φ : ℝ≥0∞) / (total faces : ℝ≥0∞) ^ J
  | 0, φ => by
    rw [show rollDice ((0 : Nat) : Int) (weightedDie faces) = PMF.pure {}
      from rfl,
      tsum_eq_single {} fun r hr => by simp [hr]]
    simp [words_zero, Record.sorted]
  | J + 1, φ => by
    have hW' : (total faces : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW
    rw [rollDice_succ, tsum_bind_mul]
    simp only [tsum_map_mul, sorted_push, tsum_weightedDie faces hW,
      div_eq_mul_inv]
    simp only [← mul_assoc, ENNReal.tsum_mul_right, ← List.sum_map_mul_left]
    rw [tsum_list_sum]
    simp only [mul_assoc, ENNReal.tsum_mul_left, mul_comm (rollDice _ _ _)]
    have ih : ∀ c : Int × Nat, ∑' r : Record, (φ (ins c.1 r.sorted) : ℝ≥0∞) *
        rollDice (J : Int) (weightedDie faces) r =
        (words faces J (fun ys => φ (ins c.1 ys)) : ℝ≥0∞) *
          ((total faces : ℝ≥0∞) ^ J)⁻¹ := fun c => by
      rw [← div_eq_mul_inv, ← tsum_rollDice faces hW J]
      exact tsum_congr fun r => mul_comm _ _
    simp only [ih, words_succ, Nat.cast_list_sum, List.map_map,
      Function.comp_def,
      Nat.cast_mul, ← mul_assoc, ← List.sum_map_mul_right]
    refine congrArg List.sum (List.map_congr_left fun c _ => ?_)
    rw [pow_succ, ENNReal.mul_inv (a := (total faces : ℝ≥0∞) ^ J)
      (b := total faces) (.inr (ENNReal.natCast_ne_top _)) (.inr hW'),
      ← mul_assoc]

/--
The weight of one sum among the program's outcomes, as `Distribution` merges
the weights of equal outcomes.

# Parameters
- `l`: The outcomes, each with its weight.
- `t`: The sum.

# Returns
The total weight of the entries of `t`.
-/
def weight (l : List (Int × Nat)) (t : Int) : Nat :=
  integral l (fun s => if s = t then 1 else 0)

/--
The sums of the kept dice of a roll and their weights. Mirrors
`order_statistics` in `record.rs`.

# Parameters
- `die`: The distinct faces, in ascending order, each with its weight.
- `n`: The number of dice.
- `lowest`: The number of lowest dice dropped.
- `highest`: The number of highest dice dropped.

# Returns
The sums, each with its weight, whose total is the total weight of the die
raised to `n`.

# Examples
```lean
-- `4D6 drop lowest 1` sums to `18` along 21 of its 1296 rolls.
#guard OrderStatistics.weight (OrderStatistics.orderStatistics
  (OrderStatistics.standardFaces 6) 4 1 0) 18 == 21
```
-/
def orderStatistics (die : List (Int × Nat)) (n lowest highest : Nat) :
    List (Int × Nat) :=
  orderStates lowest (n - lowest - highest) die n

/--
A set of dice rolls as many dice as its count, or none if it is negative.

# Parameters
- `count`: The number of dice.
- `die`: The law of one die.
-/
theorem rollDice_toNat (count : Int) (die : PMF Int) :
    rollDice count die = rollDice ((count.toNat : Nat) : Int) die := by
  simp only [rollDice, Int.toNat_natCast]

/--
The order statistics are the law of the sum of the kept dice: the probability
that the dice, less their lowest and highest drops, sum to `t` is the
program's weight of `t` over the total weight of the die raised to the number
of dice.

# Parameters
- `count`: The number of dice.
- `lowest`: The number of lowest dice dropped.
- `highest`: The number of highest dice dropped.
- `t`: The sum.

# Hypotheses
- `sorted`: The faces ascend strictly, as `distinct_faces` gives them.
- `hW`: The total weight is positive.
-/
theorem map_sum_rollDice_order {faces : List (Int × Nat)}
    (sorted : faces.Pairwise (fun a b => a.1 < b.1)) (hW : total faces ≠ 0)
    (count : Int) (lowest highest : Nat) (t : Int) :
    ((rollDice count (weightedDie faces)).map
        fun r => ({ r with lowest, highest } : Record).sum) t =
      (weight (orderStatistics faces count.toNat lowest highest) t : ℝ≥0∞) /
        (total faces : ℝ≥0∞) ^ count.toNat := by
  rw [rollDice_toNat, PMF.map_apply]
  set n := count.toNat
  have hsum : ∀ r ∈ (rollDice (n : Int) (weightedDie faces)).support,
      ({ r with lowest, highest } : Record).sum =
        window lowest (n - lowest - highest) r.sorted := by
    intro r hr
    have hlen := (mem_support_rollDice hr).2.2.1
    simp only [Int.toNat_natCast] at hlen
    simp only [Record.sum, Record.kept, window, Record.sorted, hlen]
  have hφ : ∀ r : Record, (if t = ({ r with lowest, highest } : Record).sum
      then rollDice (n : Int) (weightedDie faces) r else 0) =
      rollDice (n : Int) (weightedDie faces) r *
        ((if window lowest (n - lowest - highest) r.sorted = t then 1 else 0 :
          Nat) : ℝ≥0∞) := by
    intro r
    by_cases hr : r ∈ (rollDice (n : Int) (weightedDie faces)).support
    · rw [hsum r hr]
      split_ifs with h1 h2 h2 <;> simp_all [eq_comm]
    · rw [PMF.mem_support_iff, not_not] at hr
      simp [hr]
  calc _ = ∑' r, rollDice (n : Int) (weightedDie faces) r *
          ((if window lowest (n - lowest - highest) r.sorted = t then 1 else 0 :
            Nat) : ℝ≥0∞) := tsum_congr fun r => by convert hφ r
    _ = _ := by
      rw [tsum_rollDice faces hW n
        (fun ys =>
          if window lowest (n - lowest - highest) ys = t then 1 else 0),
        weight, orderStatistics, integral_orderStates _ _ faces sorted]

/--
The faces of a standard die, each of weight one, as `Roll::die` lists them.

# Parameters
- `faces`: The number of faces.

# Returns
The faces from `1` to `faces`.
-/
def standardFaces (faces : Int) : List (Int × Nat) :=
  (List.range faces.toNat).map fun (i : Nat) => ((i : Int) + 1, 1)

/--
The faces of a standard die, repeated by their weights, run from `1`.

# Parameters
- `faces`: The number of faces.
-/
theorem expand_standardFaces (faces : Int) :
    Roll.expand (standardFaces faces) =
      (List.range faces.toNat).map fun (i : Nat) => (i : Int) + 1 := by
  simp only [Roll.expand, standardFaces, List.flatMap_map]
  induction (List.range faces.toNat) with
  | nil => rfl
  | cons i l ih => simp_all

/--
A standard die is the die of its faces, each of weight one.

# Hypotheses
- `h`: The die has faces.
-/
theorem standardDie_eq {faces : Int} (h : 0 < faces) :
    standardDie faces = weightedDie (standardFaces faces) := by
  have hne : Roll.expand (standardFaces faces) ≠ [] := by
    rw [expand_standardFaces]
    simp only [ne_eq, List.map_eq_nil_iff, List.range_eq_nil]
    omega
  unfold standardDie weightedDie customDie
  simp only [show ¬ faces ≤ 0 by omega, show List.flatMap _ _ ≠ [] from hne,
    ↓reduceDIte]
  congr 1
  rw [show List.flatMap _ _ = Roll.expand (standardFaces faces) from rfl,
    expand_standardFaces]
  refine (Multiset.Nodup.ext (Finset.nodup _) ?_).mpr fun x => ?_
  · exact Multiset.coe_nodup.mpr ((List.nodup_range).map fun a b hab => by
      simpa using hab)
  · simp only [Finset.mem_val, Finset.mem_Icc, Multiset.mem_coe,
      List.mem_map, List.mem_range]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨(x - 1).toNat, by omega, by omega⟩
    · rintro ⟨i, hi, rfl⟩
      omega

/--
The faces of a standard die ascend strictly.

# Parameters
- `faces`: The number of faces.
-/
theorem standardFaces_sorted (faces : Int) :
    (standardFaces faces).Pairwise (fun a b => a.1 < b.1) := by
  simp only [standardFaces, List.pairwise_map]
  exact (List.pairwise_lt_range).imp fun h => by omega

/--
A standard die rolls as many ways as it has faces.

# Parameters
- `faces`: The number of faces.
-/
theorem total_standardFaces (faces : Int) :
    total (standardFaces faces) = faces.toNat := by
  simp [total, standardFaces, Function.comp_def]

end OrderStatistics

open OrderStatistics in
/--
Custom dice with drops that the convolution power does not sum have the law of
the order statistics' weights over the total weight raised to the number of
dice, as `Roll::sum` builds it.

# Parameters
- `lowest`, `highest`: The drops.
- `t`: The sum.

# Hypotheses
- `sorted`: The faces ascend strictly.
- `hW`: The total weight is positive.
-/
theorem Setup.map_sum_law_custom_order {count : Int}
    {faces : List (Int × Nat)} (lowest highest : Nat)
    (sorted : faces.Pairwise (fun a b => a.1 < b.1)) (hW : total faces ≠ 0)
    (t : Int) :
    (({ roll := some (.custom count faces), lowest, highest } : Setup).law.map
        Record.sum) t =
      (weight (orderStatistics faces count.toNat lowest highest) t : ℝ≥0∞) /
        (total faces : ℝ≥0∞) ^ count.toNat := by
  rw [Setup.law_some, PMF.map_comp]
  exact map_sum_rollDice_order sorted hW count lowest highest t

open OrderStatistics in
/--
Standard dice with drops that the convolution power does not sum have the law
of the order statistics' weights over the faces raised to the number of dice,
as `Roll::sum` builds it.

# Parameters
- `lowest`, `highest`: The drops.
- `t`: The sum.

# Hypotheses
- `h`: The dice have faces.
-/
theorem Setup.map_sum_law_standard_order {count faces : Int}
    (lowest highest : Nat) (h : 0 < faces) (t : Int) :
    (({ roll := some (.standard count faces), lowest, highest } :
        Setup).law.map Record.sum) t =
      (weight (orderStatistics (standardFaces faces) count.toNat lowest highest)
          t : ℝ≥0∞) / (faces.toNat : ℝ≥0∞) ^ count.toNat := by
  rw [Setup.law_some, PMF.map_comp]
  simp only [Roll.law, standardDie_eq h]
  have := map_sum_rollDice_order (standardFaces_sorted faces)
    (by rw [total_standardFaces]; omega) count lowest highest t
  rw [total_standardFaces] at this
  exact this

end Xdy.Spec
