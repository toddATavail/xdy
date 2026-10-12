import XdySpec.Explode

/-!
# The bounds of explodes

The bounds evaluator in `xdy/src/evaluator.rs` bounds the totals of the chains
that an explode may grow from a record that no explode has extended yet, each
of whose chains still kept holds one kept die (`Explosion::totals`). It never
enumerates the chains, but folds a few shapes of them, each die at its least
or greatest result, and every reader of the totals reads only their hull.
This module mirrors it and proves that the hull is sound, and exact for one
die and one criterion:

- `fold` mirrors `fold`, `min(i32Max, max(i32Min, Σneg) + Σpos)` over the
  negative and other dice, each with its copies, and is the total of every
  chain of those dice, the ascending saturating sum (`fold_eq_total`). It
  never decreases as any die grows (`fold_mono_results`), nor moves against
  the sign of a die as its copies grow (`fold_mono_copies`), and
  `unboundedDice` copies of a nonzero die saturate it (`fold_unbounded`).
- The tails that the specification's explode may add to a chain whose end it
  matches are exactly the limited tails (`mem_support_extension_iff`): middle
  dice that it matches, fewer than the limit, and a last die that it does not,
  unless the limit is spent; or, without limit, the unlimited tails, of any
  number of middle dice (`mem_support_unlimitedExtension_iff`). So an explode
  may leave a chain exactly when it may make it of some chain still kept
  (`Becomes`, `mem_explode_iff`).
- The total of a chain of one die grown by a tail lies between the folds of
  its shape at the least and the greatest results (`shapeFold_le_total`). The
  extremes over a range of middle dice, or of limits, lie at its ends
  (`shapeFold_between`), or, without limit, at none and at `unboundedDice`
  (`shapeFold_unbounded`); and a chain that ends before its limit is spent is
  never more extreme than both the chain that ends after one extension and
  the chain that spends its limit (`shapeFold_short`), so `Explosion::totals`
  need not fold the chains that end after `L - 1` extensions.
- `Explosion.totals` mirrors `Explosion::totals`, over results modeled as
  finite sets. Its hull covers the total of every chain that the explode may
  make of a chain of one die, for any die, criterion, and limit that the
  explosion admits (`totals_sound`), as when `RollingRecordBounds::explode`
  explodes every die and criterion at once, and so of every chain still kept
  that the explode leaves of a record (`explode_sound`). For one die and one
  criterion (`pair`), its least and greatest are some chain's
  (`pair_exact`, `Exact.hull`), and so are those of the union of every pair
  (`explodePairs_exact`), as `RollingRecordBounds::explode` unites them when
  they are few enough. The totals themselves may leave gaps between the
  shapes that some chain fills, which no reader sees.

```mermaid
flowchart TD
    F["fold_eq_total:<br/>fold is the total of a chain"]
    M["fold_mono_results, fold_mono_copies,<br/>fold_unbounded"]
    T["mem_explode_iff:<br/>the chains an explode leaves"]
    D["shapeFold_le_total, shapeFold_between,<br/>shapeFold_short, shapeFold_unbounded"]
    S["totals_sound, explode_sound:<br/>the hull covers every total"]
    E["pair_exact, explodePairs_exact:<br/>the hull's ends are totals"]
    F --> D
    M --> D
    T --> S
    D --> S
    S --> E
    T --> E
```
-/

namespace Xdy.Spec

namespace Bounds

/-! ## Folds -/

/--
The sum of the negative dice, each times its copies.

# Parameters
- `dice`: The results of the dice, each with its copies.

# Returns
The sum, which is at most `0`.
-/
def negative (dice : List (Int × Nat)) : Int :=
  (dice.map fun p => Min.min p.1 0 * (p.2 : Int)).sum

/--
The sum of the dice that are not negative, each times its copies.

# Parameters
- `dice`: The results of the dice, each with its copies.

# Returns
The sum, which is at least `0`.
-/
def positive (dice : List (Int × Nat)) : Int :=
  (dice.map fun p => Max.max p.1 0 * (p.2 : Int)).sum

/--
Fold some dice in ascending order, saturating, as the total of a chain does,
in time independent of their number: the negative dice come first, and only
fall, so the fold saturates at `i32Min` only if they sum below it, and stays
there until the rest arrive, which only rise, so that it saturates at
`i32Max` only if they lift it above. Mirrors `fold` in `evaluator.rs`.

# Parameters
- `dice`: The results of the dice, each with its copies, in any order.

# Returns
The total.

# Examples
```lean
#guard Bounds.fold [(6, 2), (-1, 1)] == 11
#guard Bounds.fold [(i32Min, 1), (-1, 1), (1, 1)] == i32Min + 1
#guard Bounds.fold [(i32Max, 2), (i32Min, 1)] == i32Max - 1
```
-/
def fold (dice : List (Int × Nat)) : Int :=
  Min.min i32Max (Max.max i32Min (negative dice) + positive dice)

/--
The number of middle dice of a chain that stands for a chain without bound,
`2³³`: many enough that the total saturates, unless every middle die shows
`0`, since a nonzero die moves the total by at least `1`, and the `i32` range
spans fewer than `2³³` values. Mirrors `UNBOUNDED_DICE` in `evaluator.rs`.
-/
def unboundedDice : Nat := 2 ^ 33

/-- The negative dice sum to at most `0`. -/
theorem negative_nonpos (dice : List (Int × Nat)) : negative dice ≤ 0 := by
  induction dice with
  | nil => rfl
  | cons p l ih =>
    simp only [negative, List.map_cons, List.sum_cons] at ih ⊢
    have := mul_nonpos_of_nonpos_of_nonneg (min_le_right p.1 0)
      (Int.natCast_nonneg p.2)
    omega

/-- The other dice sum to at least `0`. -/
theorem positive_nonneg (dice : List (Int × Nat)) : 0 ≤ positive dice :=
  List.sum_nonneg fun x hx => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.mp hx
    exact mul_nonneg (le_max_right _ _) (Int.natCast_nonneg _)

/-- The negative dice of some dice and of others sum apart. -/
@[simp]
theorem negative_append (l l' : List (Int × Nat)) :
    negative (l ++ l') = negative l + negative l' := by
  simp [negative]

/-- The other dice of some dice and of others sum apart. -/
@[simp]
theorem positive_append (l l' : List (Int × Nat)) :
    positive (l ++ l') = positive l + positive l' := by
  simp [positive]

/-- The negative dice of one die and of others. -/
@[simp]
theorem negative_cons (p : Int × Nat) (l : List (Int × Nat)) :
    negative (p :: l) = Min.min p.1 0 * p.2 + negative l := by
  simp [negative]

/-- The other dice of one die and of others. -/
@[simp]
theorem positive_cons (p : Int × Nat) (l : List (Int × Nat)) :
    positive (p :: l) = Max.max p.1 0 * p.2 + positive l := by
  simp [positive]

/-- No dice. -/
@[simp]
theorem negative_nil : negative [] = 0 := rfl

/-- No dice. -/
@[simp]
theorem positive_nil : positive [] = 0 := rfl

/-- The negative dice of copies of some dice sum the negative values. -/
private theorem negative_eq (dice : List (Int × Nat)) :
    negative dice = ((Roll.expand dice).filter (· < 0)).sum := by
  induction dice with
  | nil => rfl
  | cons p l ih =>
    simp only [negative_cons, ih, Roll.expand, List.flatMap_cons,
      List.filter_append, List.sum_append]
    congr 1
    by_cases h : p.1 < 0
    · rw [List.filter_eq_self.mpr (fun x hx => by
        rw [List.eq_of_mem_replicate hx]; simpa using h), List.sum_replicate,
        min_eq_left h.le, nsmul_eq_mul, mul_comm]
    · rw [List.filter_eq_nil_iff.mpr (fun x hx => by
        rw [List.eq_of_mem_replicate hx]; simpa using h)]
      simp [min_eq_right (not_lt.mp h)]

/-- The other dice of copies of some dice sum the other values. -/
private theorem positive_eq (dice : List (Int × Nat)) :
    positive dice = ((Roll.expand dice).filter (0 ≤ ·)).sum := by
  induction dice with
  | nil => rfl
  | cons p l ih =>
    simp only [positive_cons, ih, Roll.expand, List.flatMap_cons,
      List.filter_append, List.sum_append]
    congr 1
    by_cases h : 0 ≤ p.1
    · rw [List.filter_eq_self.mpr (fun x hx => by
        rw [List.eq_of_mem_replicate hx]; simpa using h), List.sum_replicate,
        max_eq_left h, nsmul_eq_mul, mul_comm]
    · rw [List.filter_eq_nil_iff.mpr (fun x hx => by
        rw [List.eq_of_mem_replicate hx]; simpa using h)]
      simp [max_eq_right (not_le.mp h).le]

/--
The fold of some dice is the total of the fold of their copies, as
`propagation/chain.rs` folds them.

# Parameters
- `dice`: The results of the dice, each with its copies.
-/
theorem fold_eq_ofList (dice : List (Int × Nat)) :
    fold dice = (Fold.ofList (Roll.expand dice)).total := by
  have h₁ := negative_nonpos dice
  have h₂ := positive_nonneg dice
  simp only [fold, Fold.total, Fold.ofList, ← negative_eq, ← positive_eq,
    Fold.ceiling, i32Min, i32Max]
  omega

/--
The fold of some dice is the ascending saturating sum of their copies, as a
chain totals its dice.

# Parameters
- `dice`: The results of the dice, each with its copies, in any order.
-/
theorem fold_eq_foldl (dice : List (Int × Nat)) :
    fold dice = (Roll.expand dice).mergeSort.foldl Xdy.add 0 := by
  rw [fold_eq_ofList, Fold.total_ofList]

/--
The fold of some dice is the total of every chain of those dice that may
still grow, whatever their order, as the bounds evaluator totals the shapes
of the chains that an explode may grow.

# Parameters
- `dice`: The results of the dice, each with its copies.
- `body`, `last`: The chain.

# Hypotheses
- `h`: The copies of the dice are the dice of the chain, in some order.

# Examples
```lean
example : Bounds.fold [(6, 2), (3, 1)] = (Chain.live [6, 6] 3).total :=
  Bounds.fold_eq_total (by decide)
```
-/
theorem fold_eq_total {dice : List (Int × Nat)} {body : List Int} {last : Int}
    (h : (Roll.expand dice).Perm (last :: body)) :
    fold dice = (Chain.live body last).total := by
  rw [fold_eq_ofList, Fold.ofList_perm h, Fold.total_ofList]
  rfl

/--
The fold depends only on the dice and their copies, not their order.

# Parameters
- `l`, `l'`: The dice.

# Hypotheses
- `h`: The dice are the same, in some order.
-/
theorem fold_perm {l l' : List (Int × Nat)} (h : l.Perm l') :
    fold l = fold l' := by
  simp only [fold, negative, positive, (h.map _).sum_eq]

/--
A die of no copies folds as nothing.

# Parameters
- `x`: The die.
- `pre`, `post`: The other dice.
-/
@[simp]
theorem fold_zero (x : Int) (pre post : List (Int × Nat)) :
    fold (pre ++ (x, 0) :: post) = fold (pre ++ post) := by
  simp [fold]

/--
The fold never falls as its negative dice sum higher or its others do.

# Parameters
- `l`, `l'`: The dice.

# Hypotheses
- `hn`: The negative dice of `l` sum no higher than those of `l'`.
- `hp`: The other dice of `l` sum no higher than those of `l'`.
-/
theorem fold_mono {l l' : List (Int × Nat)} (hn : negative l ≤ negative l')
    (hp : positive l ≤ positive l') : fold l ≤ fold l' := by
  simp only [fold]
  omega

/--
The fold never decreases as any die grows: if each die shows no more than its
counterpart, with as many copies, then the fold of the dice is no more than
that of their counterparts. So the least total of a chain of a shape shows the
least result at each die, and the greatest the greatest.

# Parameters
- `l`, `l'`: The dice and their counterparts.

# Hypotheses
- `h`: Each die shows no more than its counterpart, with as many copies.

# Examples
```lean
example : Bounds.fold [(1, 1), (2, 3)] ≤ Bounds.fold [(1, 1), (5, 3)] :=
  Bounds.fold_mono_results (by simp)
```
-/
theorem fold_mono_results {l l' : List (Int × Nat)}
    (h : List.Forall₂ (fun a b => a.1 ≤ b.1 ∧ a.2 = b.2) l l') :
    fold l ≤ fold l' := by
  have : negative l ≤ negative l' ∧ positive l ≤ positive l' := by
    induction h with
    | nil => simp
    | @cons a b l l' hab _ ih =>
      simp only [negative_cons, positive_cons, hab.2]
      have hk : (0 : Int) ≤ b.2 := by omega
      exact ⟨add_le_add (mul_le_mul_of_nonneg_right (min_le_min_right _ hab.1)
          hk) ih.1,
        add_le_add (mul_le_mul_of_nonneg_right (max_le_max_right _ hab.1)
          hk) ih.2⟩
  exact fold_mono this.1 this.2

/--
The fold never moves against the sign of a die as its copies grow: more
copies of a die that is not negative never lower it, and more copies of a
negative die never raise it. So the extreme folds over a range of copies lie
at its ends.

# Parameters
- `x`: The die.
- `k`, `k'`: The numbers of copies.
- `pre`, `post`: The other dice.

# Hypotheses
- `hk`: There are no more of the first than of the second.

# Returns
The fold of the fewer copies is no more than that of the more if the die is
not negative, and no less otherwise.

# Examples
```lean
example : Bounds.fold [(1, 1), (-2, 5)] ≤ Bounds.fold [(1, 1), (-2, 3)] :=
  (Bounds.fold_mono_copies (-2) (by decide) [(1, 1)] []).2 (by decide)
```
-/
theorem fold_mono_copies (x : Int) {k k' : Nat} (hk : k ≤ k')
    (pre post : List (Int × Nat)) :
    (0 ≤ x → fold (pre ++ (x, k) :: post) ≤ fold (pre ++ (x, k') :: post)) ∧
      (x < 0 →
        fold (pre ++ (x, k') :: post) ≤ fold (pre ++ (x, k) :: post)) := by
  have hk' : (k : Int) ≤ k' := by exact_mod_cast hk
  refine ⟨fun hx => fold_mono ?_ ?_, fun hx => fold_mono ?_ ?_⟩ <;>
    simp only [negative_append, negative_cons, positive_append,
      positive_cons, add_le_add_iff_left, add_le_add_iff_right]
  · simp [min_eq_right hx]
  · exact mul_le_mul_of_nonneg_left hk' (le_max_right _ _)
  · exact mul_le_mul_of_nonpos_left hk' (by simp [min_eq_left hx.le]; omega)
  · simp [max_eq_right hx.le]

/--
`unboundedDice` copies of a nonzero die saturate the fold, so every greater
number of copies folds alike, as `UNBOUNDED_DICE` relies on.

# Parameters
- `x`: The die.
- `k`: The number of copies.
- `pre`, `post`: The other dice.

# Hypotheses
- `hx`: The die is not `0`.
- `hk`: There are at least `unboundedDice` copies.

# Examples
```lean
example : Bounds.fold [(1, 1), (-1, 2 ^ 40)] =
    Bounds.fold [(1, 1), (-1, Bounds.unboundedDice)] :=
  Bounds.fold_unbounded (by decide) (by decide) [(1, 1)] []
```
-/
theorem fold_unbounded {x : Int} {k : Nat} (hx : x ≠ 0)
    (hk : unboundedDice ≤ k) (pre post : List (Int × Nat)) :
    fold (pre ++ (x, k) :: post) =
      fold (pre ++ (x, unboundedDice) :: post) := by
  have hk' : (2 ^ 33 : Int) ≤ k := by exact_mod_cast hk
  have h₁ := negative_nonpos pre
  have h₂ := negative_nonpos post
  have h₃ := positive_nonneg pre
  have h₄ := positive_nonneg post
  simp only [fold, negative_append, negative_cons, positive_append,
    positive_cons, unboundedDice, i32Min, i32Max]
  rcases lt_or_gt_of_ne hx with hx | hx
  · simp only [min_eq_left hx.le, max_eq_right hx.le, zero_mul]
    have : x * (k : Int) ≤ -2 ^ 33 := by nlinarith
    push_cast
    omega
  · simp only [min_eq_right hx.le, max_eq_left hx.le, zero_mul]
    have : 2 ^ 33 ≤ x * (k : Int) := by nlinarith
    push_cast
    omega

/-! ## Tails -/

/-- Values sorted by `mergeSort` ascend. -/
private theorem pairwise_mergeSort (l : List Int) :
    l.mergeSort.Pairwise (· ≤ ·) := by
  have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
    (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
    (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) l
  simpa using this

/-- Values that ascend already sort to themselves. -/
private theorem mergeSort_eq_self {l : List Int} (h : l.Pairwise (· ≤ ·)) :
    l.mergeSort = l :=
  List.mergeSort_of_pairwise (by simpa using h)

/--
A tail that an explode limited to `limit` times may add to a chain whose end
it matches: the middle dice, in ascending order, each a face that the explode
matches, fewer than `limit` of them, and the last die, a face that ends the
chain, which the explode does not match unless the limit is spent.

# Parameters
- `F`: The faces of the die.
- `c`: The criterion.
- `limit`: The limit.
- `t`: The tail: the middle dice and the last.
-/
def LimitedTail (F : List Int) (c : Criterion) (limit : Nat)
    (t : List Int × Int) : Prop :=
  t.1.Pairwise (· ≤ ·) ∧ (∀ x ∈ t.1, x ∈ F ∧ c.matches x = true) ∧ t.2 ∈ F ∧
    t.1.length < limit ∧ (t.1.length + 1 < limit → c.matches t.2 = false)

/--
A tail that an unlimited explode may add to a chain whose end it matches:
the middle dice, in ascending order, each a face that the explode matches, of
any number, and the last die, a face that it does not match.

# Parameters
- `F`: The faces of the die.
- `c`: The criterion.
- `t`: The tail: the middle dice and the last.
-/
def UnlimitedTail (F : List Int) (c : Criterion) (t : List Int × Int) :
    Prop :=
  t.1.Pairwise (· ≤ ·) ∧ (∀ x ∈ t.1, x ∈ F ∧ c.matches x = true) ∧ t.2 ∈ F ∧
    c.matches t.2 = false

/-- Whether a tail is a limited tail is decided. -/
instance (F : List Int) (c : Criterion) (limit : Nat) (t : List Int × Int) :
    Decidable (LimitedTail F c limit t) := by
  unfold LimitedTail
  infer_instance

/-- Whether a tail is an unlimited tail is decided. -/
instance (F : List Int) (c : Criterion) (t : List Int × Int) :
    Decidable (UnlimitedTail F c t) := by
  unfold UnlimitedTail
  infer_instance

/-- A custom die with faces shows only its faces. -/
private theorem mem_of_customDie {F : List Int} (hF : F ≠ []) {x : Int}
    (hx : x ∈ (customDie F).support) : x ∈ F := by
  rcases mem_of_mem_support_customDie hx with h | ⟨h, _⟩
  · exact h
  · exact absurd h hF

/-- Every way in which a chain grows from sorted extensions so far draws
matching faces, at most as many as it may, and then a face that ends it,
which does not match unless no extension is left. -/
private theorem growthLaw_tail {F : List Int} (hF : F ≠ []) (c : Criterion) :
    ∀ (r : Nat) (s : List Int), s.Pairwise (· ≤ ·) →
      ∀ t ∈ (growthLaw (customDie F) c tailExtend tailClose r s).support,
        t.1.Pairwise (· ≤ ·) ∧ ∃ ds : List Int, t.1.Perm (s ++ ds) ∧
          (∀ x ∈ ds, x ∈ F ∧ c.matches x = true) ∧ t.2 ∈ F ∧
          ds.length ≤ r ∧ (ds.length < r → c.matches t.2 = false)
  | 0, s, hs, t, ht => by
    simp only [growthLaw, PMF.support_map, Set.mem_image] at ht
    obtain ⟨x, hx, rfl⟩ := ht
    exact ⟨hs, [], by simp [tailClose], by simp, mem_of_customDie hF hx,
      le_rfl, by simp⟩
  | r + 1, s, hs, t, ht => by
    simp only [growthLaw, PMF.mem_support_bind_iff] at ht
    obtain ⟨x, hx, ht⟩ := ht
    split at ht
    · rename_i hm
      obtain ⟨hp, ds, hds, hdf, hy, hl, he⟩ :=
        growthLaw_tail hF c r _ (pairwise_mergeSort _) t ht
      refine ⟨hp, x :: ds, ?_, ?_, hy, by simp; omega,
        fun h => he (by simp at h; omega)⟩
      · refine hds.trans ?_
        exact ((List.mergeSort_perm _ _).append_right _).trans
          (List.perm_middle.symm)
      · intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · exact ⟨mem_of_customDie hF hx, hm⟩
        · exact hdf z hz
    · rename_i hm
      rw [PMF.mem_support_pure_iff] at ht
      subst ht
      exact ⟨hs, [], by simp [tailClose], by simp, mem_of_customDie hF hx,
        by simp, fun _ => by simpa [tailClose] using hm⟩

/-- Drawing values one at a time into sorted extensions leaves them sorted,
with the values among them. -/
private theorem foldl_tailExtend :
    ∀ (ds s : List Int), s.Pairwise (· ≤ ·) →
      (ds.foldl tailExtend s).Pairwise (· ≤ ·) ∧
        (ds.foldl tailExtend s).Perm (s ++ ds)
  | [], s, hs => ⟨hs, by simp⟩
  | x :: ds, s, _ => by
    obtain ⟨hp, hq⟩ := foldl_tailExtend ds _ (pairwise_mergeSort (x :: s))
    refine ⟨hp, hq.trans ?_⟩
    exact ((List.mergeSort_perm _ _).append_right _).trans
      List.perm_middle.symm

/-- A chain may grow by drawing any matching faces, at most as many as it
may, and then a face that ends it, which does not match unless no extension
is left. -/
private theorem mem_growthLaw_of_tail {F : List Int} (c : Criterion) :
    ∀ (ds : List Int) (r : Nat) (s : List Int) (y : Int),
      (∀ x ∈ ds, x ∈ F ∧ c.matches x = true) → y ∈ F → ds.length ≤ r →
      (ds.length < r → c.matches y = false) →
      (ds.foldl tailExtend s, y) ∈
        (growthLaw (customDie F) c tailExtend tailClose r s).support
  | [], 0, s, y, _, hy, _, _ => by
    simp only [growthLaw, PMF.support_map, Set.mem_image]
    exact ⟨y, mem_support_customDie hy, rfl⟩
  | [], r + 1, s, y, _, hy, _, he => by
    simp only [growthLaw, PMF.mem_support_bind_iff]
    refine ⟨y, mem_support_customDie hy, ?_⟩
    simp [he (by simp), tailClose]
  | x :: ds, 0, s, y, _, _, hl, _ => by simp at hl
  | x :: ds, r + 1, s, y, hd, hy, hl, he => by
    simp only [growthLaw, PMF.mem_support_bind_iff]
    have hx := hd x List.mem_cons_self
    refine ⟨x, mem_support_customDie hx.1, ?_⟩
    simp only [hx.2, ↓reduceIte, List.foldl_cons]
    exact mem_growthLaw_of_tail c ds r _ y
      (fun z hz => hd z (List.mem_cons_of_mem _ hz)) hy
      (by simp at hl; omega) (fun h => he (by simp; omega))

/--
The tails that an explode limited to a positive number of times may add to a
chain whose end it matches are exactly the limited tails: middle dice that it
matches, and a last die that it does not, unless the limit is spent.

# Parameters
- `F`: The faces of the die.
- `c`: The criterion.
- `limit`: The limit.
- `t`: The tail.

# Hypotheses
- `hF`: The die has faces.
- `hl`: The limit is positive.
-/
theorem mem_support_extension_iff {F : List Int} {c : Criterion} {limit : Nat}
    {t : List Int × Int} (hF : F ≠ []) (hl : 0 < limit) :
    t ∈ (extension c (customDie F) limit).support ↔ LimitedTail F c limit t := by
  rw [extension_eq_growthLaw]
  constructor
  · intro ht
    obtain ⟨hp, ds, hds, hdf, hy, hlen, he⟩ :=
      growthLaw_tail hF c _ [] (by simp) t ht
    have hlen' := hds.length_eq
    simp only [List.nil_append] at hds hlen'
    exact ⟨hp, fun x hx => hdf x (hds.mem_iff.mp hx), hy, by omega,
      fun h => he (by omega)⟩
  · rintro ⟨hp, hdf, hy, hlen, he⟩
    obtain ⟨hp', hq⟩ := foldl_tailExtend t.1 [] (by simp)
    have heq : t.1.foldl tailExtend [] = t.1 :=
      List.Perm.eq_of_pairwise (le := (· ≤ ·)) (fun a b _ _ => by omega) hp'
        hp (by simpa using hq)
    have := mem_growthLaw_of_tail c t.1 (limit - 1) [] t.2 hdf hy (by omega)
      (fun h => he (by omega))
    rwa [heq] at this

/-- The program of `k` draws of matching faces, sorted, as `unlimitedExtension`
draws its middle dice. -/
private noncomputable def middles (M : List Int) (k : Nat) : PMF (List Int) :=
  k.repeat
    (fun (p : PMF (List Int)) => p.bind fun drawn =>
      (customDie M).map fun x => (x :: drawn).mergeSort)
    (PMF.pure [])

/-- The middle dice drawn from faces are exactly the sorted lists of as many
of them. -/
private theorem mem_support_middles {M : List Int} :
    ∀ (k : Nat), (0 < k → M ≠ []) → ∀ ds : List Int,
      ds ∈ (middles M k).support ↔
        ds.Pairwise (· ≤ ·) ∧ ds.length = k ∧ ∀ x ∈ ds, x ∈ M
  | 0, _, ds => by
    simp only [middles, Nat.repeat, PMF.mem_support_pure_iff]
    constructor
    · rintro rfl
      simp
    · rintro ⟨-, h, -⟩
      exact List.eq_nil_of_length_eq_zero h
  | k + 1, hM, ds => by
    have hM := hM (by omega)
    have ih := mem_support_middles k (fun _ => hM)
    show ds ∈ ((middles M k).bind fun drawn =>
      (customDie M).map fun x => (x :: drawn).mergeSort).support ↔ _
    simp only [PMF.mem_support_bind_iff, PMF.support_map, Set.mem_image]
    constructor
    · rintro ⟨drawn, hd, x, hx, rfl⟩
      obtain ⟨-, hl, hm⟩ := (ih drawn).mp hd
      refine ⟨pairwise_mergeSort _, by simp [hl], fun z hz => ?_⟩
      rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hz) with
        rfl | hz
      · exact mem_of_customDie hM hx
      · exact hm z hz
    · rintro ⟨hp, hl, hm⟩
      obtain ⟨x, rest, rfl⟩ := List.exists_cons_of_length_eq_add_one hl
      refine ⟨rest, (ih rest).mpr ⟨(List.pairwise_cons.mp hp).2,
          by simpa using hl, fun z hz => hm z (List.mem_cons_of_mem _ hz)⟩,
        x, mem_support_customDie (hm x List.mem_cons_self),
        mergeSort_eq_self hp⟩

/--
The tails that an unlimited explode that does not match every face may add to
a chain whose end it matches are exactly the unlimited tails: any number of
middle dice that it matches, and a last die that it does not.

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`.
- `t`: The tail.
-/
theorem mem_support_unlimitedExtension_iff {c : Criterion} {d : Die}
    (h : ¬d.facesWithin c = true) {t : List Int × Int} :
    t ∈ (unlimitedExtension c d h).support ↔ UnlimitedTail d.faces c t := by
  have hq : matchedFaces d c ≠ 0 ↔ d.faces.filter c.matches ≠ [] := by
    unfold matchedFaces
    constructor
    · intro hq hn
      simp [hn] at hq
    · intro hn
      refine (ENNReal.div_pos_iff.mpr ⟨?_, by simp⟩).ne'
      simpa using hn
  have hgeo : ∀ k, geometric (matchedFaces d c) (matchedFaces_lt_one h) k ≠ 0 ↔
      (0 < k → d.faces.filter c.matches ≠ []) := by
    intro k
    rw [geometric_apply, mul_ne_zero_iff, ← hq]
    have h₁ : 1 - matchedFaces d c ≠ 0 :=
      (tsub_pos_iff_lt.mpr (matchedFaces_lt_one h)).ne'
    simp only [h₁, ne_eq, not_false_eq_true, and_true, pow_eq_zero_iff',
      not_and, not_not]
    constructor
    · intro h₂ hk hq
      have := h₂ hq
      omega
    · intro h₂ hq
      by_contra hk
      exact h₂ (by omega) hq
  unfold unlimitedExtension
  simp only [PMF.mem_support_bind_iff, PMF.support_map, Set.mem_image,
    PMF.mem_support_ofMultiset_iff, Multiset.mem_toFinset, Multiset.mem_coe,
    List.mem_filter, Bool.not_eq_true']
  constructor
  · rintro ⟨k, hk, drawn, hd, y, ⟨hy, hm⟩, rfl⟩
    obtain ⟨hp, -, hdm⟩ := (mem_support_middles k ((hgeo k).mp hk) drawn).mp hd
    refine ⟨hp, fun x hx => ?_, hy, hm⟩
    have := List.mem_filter.mp (hdm x hx)
    exact ⟨this.1, this.2⟩
  · rintro ⟨hp, hdm, hy, hm⟩
    have hne : 0 < t.1.length → d.faces.filter c.matches ≠ [] := fun hl => by
      obtain ⟨x, hx⟩ := List.exists_mem_of_length_pos hl
      exact List.ne_nil_of_mem (List.mem_filter.mpr (hdm x hx))
    exact ⟨t.1.length, (hgeo _).mpr hne, t.1,
      (mem_support_middles _ hne t.1).mpr ⟨hp, rfl,
        fun x hx => List.mem_filter.mpr (hdm x hx)⟩, t.2, ⟨hy, hm⟩, rfl⟩

/-! ## What an explode makes of a chain -/

/--
Whether an explode may make one chain of another still kept: an explode
limited to a positive number of times grows a chain whose end it matches by a
limited tail, an unlimited explode that matches every face of the die
saturates every chain, and one that does not grows a chain whose end it
matches by an unlimited tail; every other chain stays as it was. Mirrors the
shapes of chain that `Explosion::totals` in `evaluator.rs` considers.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The limit, or `none` for no limit.
- `ch`: The chain still kept.
- `ch'`: What the explode makes of it.
-/
def Becomes (d : Die) (c : Criterion) : Option Nat → Chain → Chain → Prop
  | some limit, ch, ch' =>
    if 0 < limit ∧ ch.grows c = true then
      ∃ t, LimitedTail d.faces c limit t ∧ ch' = growChain ch t
    else ch' = ch
  | none, ch, ch' =>
    if d.facesWithin c = true then ch' = ch.saturate d.saturatedValue
    else if ch.grows c = true then
      ∃ t, UnlimitedTail d.faces c t ∧ ch' = growChain ch t
    else ch' = ch

/-- The draws of independent values can hold any one draw of any one of them,
since every law can draw something. -/
private theorem exists_mem_bagDraws {α β : Type} {step : α → PMF β} :
    ∀ {xs : List α} {x : α}, x ∈ xs → ∀ {y : β}, y ∈ (step x).support →
      ∃ M ∈ (bagDraws step xs).support, y ∈ M
  | [], _, hx, _, _ => absurd hx List.not_mem_nil
  | z :: xs, x, hx, y, hy => by
    simp only [bagDraws, lift, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff]
    rcases List.mem_cons.mp hx with rfl | hx
    · obtain ⟨N, hN⟩ := (bagDraws step xs).support_nonempty
      exact ⟨_, ⟨y, hy, N, hN, rfl⟩, Multiset.mem_cons_self _ _⟩
    · obtain ⟨w, hw⟩ := (step z).support_nonempty
      obtain ⟨M, hM, hyM⟩ := exists_mem_bagDraws hx hy
      exact ⟨_, ⟨w, hw, M, hM, rfl⟩, Multiset.mem_cons_of_mem hyM⟩

/-- The kept chains of a record of normal chains are those chains. -/
private theorem mem_keptChains_ofChains {die : Die} {cs : List Chain}
    (hn : ∀ x ∈ cs, NormalChain x) {ch : Chain} :
    ch ∈ (Record.ofChains die cs).keptChains ↔ ch ∈ cs := by
  rw [← Multiset.mem_coe, keptChains_ofChains die hn, Multiset.mem_coe]

/-- An explode of normal chains, each of which it grows into normal chains,
leaves each kept chain grown by a possible extension, if it grows, or else
itself, and may leave each such chain. -/
private theorem mem_explodeChains {die : Die} {kept : List Chain}
    {grows : Chain → Bool} {ext : PMF (List Int × Int)}
    (hn : ∀ ch ∈ kept, NormalChain ch)
    (hg : ∀ ch ∈ kept, grows ch → ∀ t ∈ ext.support,
      NormalChain (growChain ch t)) (ch' : Chain) :
    (∃ r' ∈ (explodeChains die kept grows ext).support,
        ch' ∈ r'.keptChains) ↔
      ∃ ch ∈ kept, if grows ch then ∃ t ∈ ext.support, ch' = growChain ch t
        else ch' = ch := by
  rw [explodeChains_eq_bag die grows ext hn hg]
  have hstep : ∀ ch ∈ kept, ∀ y ∈ (if grows ch then ext.map (growChain ch)
      else PMF.pure ch).support, NormalChain y := fun ch hch y hy => by
    split at hy
    · rename_i hgr
      obtain ⟨t, ht, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hy
      exact hg ch hch hgr t ht
    · rw [PMF.mem_support_pure_iff] at hy
      exact hy ▸ hn ch hch
  have hiff : ∀ ch, ch' ∈ (if grows ch then ext.map (growChain ch)
      else PMF.pure ch).support ↔
        if grows ch then ∃ t ∈ ext.support, ch' = growChain ch t
        else ch' = ch := fun ch => by
    split
    · simp only [PMF.mem_support_map_iff]
      exact ⟨fun ⟨t, ht, h⟩ => ⟨t, ht, h.symm⟩, fun ⟨t, ht, h⟩ =>
        ⟨t, ht, h.symm⟩⟩
    · exact PMF.mem_support_pure_iff _ _
  simp only [PMF.support_map, Set.mem_image]
  constructor
  · rintro ⟨_, ⟨M, hM, rfl⟩, hch⟩
    have hnM : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
      obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x
        (Multiset.mem_toList.mp hx)
      exact hstep ch hch x hx
    obtain ⟨ch, hch, hy⟩ := mem_support_bagDraws hM ch'
      (Multiset.mem_toList.mp ((mem_keptChains_ofChains hnM).mp hch))
    exact ⟨ch, hch, (hiff ch).mp hy⟩
  · rintro ⟨ch, hch, hy⟩
    obtain ⟨M, hM, hyM⟩ := exists_mem_bagDraws (step := fun ch =>
      if grows ch then ext.map (growChain ch) else PMF.pure ch) hch
      ((hiff ch).mpr hy)
    have hnM : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
      obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x
        (Multiset.mem_toList.mp hx)
      exact hstep ch hch x hx
    exact ⟨_, ⟨M, hM, rfl⟩, (mem_keptChains_ofChains hnM).mpr
      (Multiset.mem_toList.mpr hyM)⟩

/-- Every kept chain holds some die. -/
private theorem dice_ne_nil (ch : Chain) : ch.dice ≠ [] := by
  cases ch with
  | live body last =>
    intro h
    have := congrArg List.length h
    simp [Chain.dice, List.length_mergeSort] at this
  | saturated v => simp [Chain.dice]

/--
An explode may leave a chain among the kept chains of some record exactly
when it may make that chain of some chain still kept, for a record of normal
chains of faces of its die, which lie within the `i32` range: so the totals of
the chains that it leaves are the totals of what it may make of each.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `ch'`: A chain.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hk`: Every kept chain is normal, and holds only faces of the die.
-/
theorem mem_explode_iff {r : Record} {c : Criterion} {l : Option Nat}
    (hF : ∀ x ∈ r.die.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hk : ∀ ch ∈ r.keptChains, NormalChain ch ∧ ∀ x ∈ ch.dice, x ∈ r.die.faces)
    (ch' : Chain) :
    (∃ r' ∈ (explode r c l).support, ch' ∈ r'.keptChains) ↔
      ∃ ch ∈ r.keptChains, Becomes r.die c l ch ch' := by
  have hne : ∀ ch ∈ r.keptChains, r.die.faces ≠ [] := fun ch hch hn => by
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ (dice_ne_nil ch)
    simpa [hn] using (hk ch hch).2 x hx
  cases l with
  | some limit =>
    have hg : ∀ ch ∈ r.keptChains, (0 < limit && ch.grows c) = true →
        ∀ t ∈ (extension c (customDie r.die.faces) limit).support,
          NormalChain (growChain ch t) := fun ch hch _ t ht =>
      normalChain_growChain hF (hk ch hch).2
        (mem_support_extension (hne ch hch) ht)
    simp only [explode]
    rw [mem_explodeChains (fun ch h => (hk ch h).1) hg]
    refine exists_congr fun ch => and_congr_right fun hch => ?_
    simp only [Becomes, Bool.and_eq_true, decide_eq_true_eq]
    split
    · rename_i hgr
      refine exists_congr fun t => ?_
      rw [mem_support_extension_iff (hne ch hch) hgr.1]
    · rfl
  | none =>
    by_cases hw : r.die.facesWithin c = true
    · simp only [explode, hw, ↓reduceDIte, PMF.mem_support_pure_iff,
        exists_eq_left, Becomes, ↓reduceIte]
      have hn : ∀ x ∈ r.keptChains.map (·.saturate r.die.saturatedValue),
          NormalChain x := fun x hx => by
        obtain ⟨ch, hch, rfl⟩ := List.mem_map.mp hx
        rcases ch with ⟨body, last⟩ | v
        · exact trivial
        · exact (hk _ hch).1
      rw [mem_keptChains_ofChains hn, List.mem_map]
      exact exists_congr fun ch => and_congr_right fun _ => eq_comm
    · have hg : ∀ ch ∈ r.keptChains, ch.grows c = true →
          ∀ t ∈ (unlimitedExtension c r.die hw).support,
            NormalChain (growChain ch t) := fun ch hch _ t ht => by
        obtain ⟨-, hdm, hy, -⟩ :=
          (mem_support_unlimitedExtension_iff hw).mp ht
        exact normalChain_growChain hF (hk ch hch).2
          ⟨fun x hx => (hdm x hx).1, hy⟩
      simp only [explode, hw, Bool.false_eq_true, ↓reduceDIte]
      rw [mem_explodeChains (fun ch h => (hk ch h).1) hg]
      refine exists_congr fun ch => and_congr_right fun hch => ?_
      simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte]
      split
      · refine exists_congr fun t => ?_
        rw [mem_support_unlimitedExtension_iff hw]
      · rfl

/-! ## Shapes -/

/--
The fold of a chain of a shape: a first die, `count` middle dice that show
alike, and a last die, if the chain has one, as `Explosion::totals` in
`evaluator.rs` folds each shape at its least or greatest results.

# Parameters
- `first`: The first die.
- `middle`: Each middle die.
- `count`: The number of middle dice.
- `last`: The last die, or `none` if the chain holds only its first.

# Returns
The fold.

# Examples
```lean
#guard Bounds.shapeFold 6 6 2 (some 1) == 19
#guard Bounds.shapeFold 3 6 0 none == 3
```
-/
def shapeFold (first middle : Int) (count : Nat) (last : Option Int) : Int :=
  fold ([(first, 1), (middle, count)] ++ last.toList.map ((·, 1)))

/-- The fold depends only on the copies of the dice, in any order. -/
private theorem fold_congr {l l' : List (Int × Nat)}
    (h : (Roll.expand l).Perm (Roll.expand l')) : fold l = fold l' := by
  rw [fold_eq_ofList, fold_eq_ofList, Fold.ofList_perm h]

/-- One copy of each value expands to the values. -/
private theorem expand_singles (xs : List Int) :
    Roll.expand (xs.map ((·, 1))) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons, Roll.expand, List.flatMap_cons,
      List.replicate_one, List.singleton_append] at ih ⊢
    rw [ih]

/--
A chain of one die totals the fold of the shape of that die alone.

# Parameters
- `x`: The die.
- `middle`: Any middle die, of which the shape holds none.
-/
theorem total_single_eq (x middle : Int) :
    (Chain.single x).total = shapeFold x middle 0 none := by
  rw [shapeFold, Option.toList_none, List.map_nil, List.append_nil,
    show [(x, 1), (middle, 0)] = [(x, 1)] ++ (middle, 0) :: [] from rfl,
    fold_zero, List.append_nil]
  exact (fold_eq_total (body := []) (by simp [Roll.expand])).symm

/--
A chain of one die grown by a tail totals the fold of the die, the middle
dice, and the last.

# Parameters
- `x`: The die.
- `t`: The tail.
-/
theorem total_growChain_single (x : Int) (t : List Int × Int) :
    (growChain (.single x) t).total =
      fold ((x, 1) :: (t.2, 1) :: t.1.map ((·, 1))) := by
  rw [growChain]
  refine (fold_eq_total ?_).symm
  simp only [Roll.expand, List.flatMap_cons, List.replicate_one,
    List.singleton_append]
  rw [show List.flatMap (fun p : Int × Nat => List.replicate p.2 p.1)
      (t.1.map ((·, 1))) = t.1 from expand_singles t.1]
  refine List.Perm.trans (List.Perm.swap _ _ _) (List.Perm.cons _ ?_)
  refine List.Perm.trans ?_ (List.mergeSort_perm _ _).symm
  simp [Chain.single, Chain.dice]

/-- One copy of each of some values, each at least `lo` and at most `hi`,
sums its negative values and its others between those of as many copies of
`lo` and of `hi`. -/
private theorem singles_bounds {lo hi : Int} :
    ∀ {xs : List Int}, (∀ z ∈ xs, lo ≤ z ∧ z ≤ hi) →
      Min.min lo 0 * xs.length ≤ negative (xs.map ((·, 1))) ∧
        negative (xs.map ((·, 1))) ≤ Min.min hi 0 * xs.length ∧
        Max.max lo 0 * xs.length ≤ positive (xs.map ((·, 1))) ∧
        positive (xs.map ((·, 1))) ≤ Max.max hi 0 * xs.length
  | [], _ => by simp
  | z :: xs, h => by
    obtain ⟨a, b, c, d⟩ := singles_bounds fun w hw =>
      h w (List.mem_cons_of_mem _ hw)
    have hz := h z List.mem_cons_self
    have e₁ : Min.min lo 0 ≤ Min.min z 0 := min_le_min_right _ hz.1
    have e₂ : Min.min z 0 ≤ Min.min hi 0 := min_le_min_right _ hz.2
    have e₃ : Max.max lo 0 ≤ Max.max z 0 := max_le_max_right _ hz.1
    have e₄ : Max.max z 0 ≤ Max.max hi 0 := max_le_max_right _ hz.2
    simp only [List.map_cons, negative_cons, positive_cons, List.length_cons,
      Nat.cast_add, Nat.cast_one, mul_one]
    refine ⟨?_, ?_, ?_, ?_⟩ <;> linarith

/--
The total of a chain of one die grown by a tail lies between the folds of its
shape at the least and at the greatest results: if its first die, each middle
die, and its last lie within bounds, then so does its total, between the
folds of the shape of as many middle dice with each die at its bound. So the
least total of a chain of a shape shows the least result at each die, and
the greatest the greatest.

# Parameters
- `x`: The first die.
- `t`: The tail.
- `lo₁`, `hi₁`: The bounds of the first die.
- `lo₂`, `hi₂`: The bounds of each middle die.
- `lo₃`, `hi₃`: The bounds of the last die.

# Hypotheses
- `hx`: The first die lies within its bounds.
- `hm`: Each middle die lies within its bounds.
- `hy`: The last die lies within its bounds.
-/
theorem shapeFold_le_total {x : Int} {t : List Int × Int}
    {lo₁ hi₁ lo₂ hi₂ lo₃ hi₃ : Int} (hx : lo₁ ≤ x ∧ x ≤ hi₁)
    (hm : ∀ z ∈ t.1, lo₂ ≤ z ∧ z ≤ hi₂) (hy : lo₃ ≤ t.2 ∧ t.2 ≤ hi₃) :
    shapeFold lo₁ lo₂ t.1.length (some lo₃) ≤ (growChain (.single x) t).total ∧
      (growChain (.single x) t).total ≤
        shapeFold hi₁ hi₂ t.1.length (some hi₃) := by
  obtain ⟨a, b, c, d⟩ := singles_bounds hm
  have e₁ := min_le_min_right 0 hx.1
  have e₂ := min_le_min_right 0 hx.2
  have e₃ := max_le_max_right 0 hx.1
  have e₄ := max_le_max_right 0 hx.2
  have e₅ := min_le_min_right 0 hy.1
  have e₆ := min_le_min_right 0 hy.2
  have e₇ := max_le_max_right 0 hy.1
  have e₈ := max_le_max_right 0 hy.2
  rw [total_growChain_single]
  constructor <;> refine fold_mono ?_ ?_ <;>
    simp only [negative_cons, positive_cons, negative_append, positive_append,
      negative_nil, positive_nil, Option.toList_some, List.map_cons,
      List.map_nil, Nat.cast_one, mul_one, add_zero] <;> linarith

/--
The fold of a shape never decreases as any of its dice grows.

# Parameters
- `count`: The number of middle dice.

# Hypotheses
- `h₁`: The first die grows.
- `h₂`: Each middle die grows.
- `h₃`: The last die grows, if there is one.
-/
theorem shapeFold_mono {first first' middle middle' : Int} (count : Nat)
    {last last' : Option Int} (h₁ : first ≤ first') (h₂ : middle ≤ middle')
    (h₃ : Option.Rel (· ≤ ·) last last') :
    shapeFold first middle count last ≤ shapeFold first' middle' count last' := by
  refine fold_mono_results ?_
  refine List.Forall₂.cons ⟨h₁, rfl⟩ (List.Forall₂.cons ⟨h₂, rfl⟩ ?_)
  cases h₃ with
  | none => exact List.Forall₂.nil
  | some h => exact List.Forall₂.cons ⟨h, rfl⟩ List.Forall₂.nil

/-- The shape of a fold, with its middle dice apart. -/
private theorem shapeFold_eq (first middle : Int) (count : Nat)
    (last : Option Int) :
    shapeFold first middle count last =
      fold ([(first, 1)] ++ (middle, count) :: last.toList.map ((·, 1))) :=
  rfl

/--
The extreme folds of a shape over a range of middle dice lie at its ends,
since the fold never moves against the sign of the middle die as their number
grows: the extremes over a range of limits, whose chains spend them, lie at
its ends, too.

# Parameters
- `first`, `middle`: The first die and each middle die.
- `a`, `m`, `b`: The ends of the range, and a number of middle dice within it.
- `last`: The last die, if any.

# Hypotheses
- `ha`, `hb`: `m` lies between `a` and `b`.
-/
theorem shapeFold_between (first middle : Int) {a m b : Nat} (ha : a ≤ m)
    (hb : m ≤ b) (last : Option Int) :
    Min.min (shapeFold first middle a last) (shapeFold first middle b last) ≤
        shapeFold first middle m last ∧
      shapeFold first middle m last ≤
        Max.max (shapeFold first middle a last)
          (shapeFold first middle b last) := by
  simp only [shapeFold_eq]
  have h₁ := fold_mono_copies middle ha [(first, 1)]
    (last.toList.map ((·, 1)))
  have h₂ := fold_mono_copies middle hb [(first, 1)]
    (last.toList.map ((·, 1)))
  rcases le_or_gt 0 middle with h | h
  · have := h₁.1 h
    have := h₂.1 h
    constructor <;> omega
  · have := h₁.2 h
    have := h₂.2 h
    constructor <;> omega

/--
A chain that ends before its limit is spent is never more extreme than both
the chain that ends after one extension and the chain whose limit is spent,
which holds at least as many middle dice and a last die that may show any
face, at least as extreme as one that ends the chain: if the middle dice move
the total toward an extreme, the spent chain is at least as extreme, and
otherwise the short one. So `Explosion::totals` need not fold the chains that
end after `L - 1` extensions.

# Parameters
- `first`, `middle`: The first die and each middle die.
- `m`, `M`: The middle dice of the chain and of the spent chain.
- `e`, `lo`, `hi`: The last die of the chain, and the least and greatest of
  the spent chain.

# Hypotheses
- `hm`: The spent chain holds at least as many middle dice.
- `hlo`, `hhi`: Its last die may show `e` or beyond, either way.
-/
theorem shapeFold_short (first middle : Int) {m M : Nat} (hm : m ≤ M)
    {e lo hi : Int} (hlo : lo ≤ e) (hhi : e ≤ hi) :
    Min.min (shapeFold first middle 0 (some e))
        (shapeFold first middle M (some lo)) ≤
          shapeFold first middle m (some e) ∧
      shapeFold first middle m (some e) ≤
        Max.max (shapeFold first middle 0 (some e))
          (shapeFold first middle M (some hi)) := by
  have h := shapeFold_between first middle (Nat.zero_le m) hm (some e)
  have h₁ := shapeFold_mono M le_rfl le_rfl (Option.Rel.some hlo)
    (first := first) (middle := middle)
  have h₂ := shapeFold_mono M le_rfl le_rfl (Option.Rel.some hhi)
    (first := first) (middle := middle)
  constructor <;> omega

/--
The extreme folds of a shape of any number of middle dice lie at no middle
dice and at `unboundedDice`, which stands for a chain without bound: beyond
it, every nonzero middle die has saturated the fold, and a middle die of `0`
never moves it.

# Parameters
- `first`, `middle`: The first die and each middle die.
- `m`: The number of middle dice.
- `last`: The last die, if any.
-/
theorem shapeFold_unbounded (first middle : Int) (m : Nat)
    (last : Option Int) :
    Min.min (shapeFold first middle 0 last)
        (shapeFold first middle unboundedDice last) ≤
          shapeFold first middle m last ∧
      shapeFold first middle m last ≤
        Max.max (shapeFold first middle 0 last)
          (shapeFold first middle unboundedDice last) := by
  rcases le_or_gt m unboundedDice with hm | hm
  · exact shapeFold_between first middle (Nat.zero_le m) hm last
  · by_cases hz : middle = 0
    · subst hz
      have : shapeFold first 0 m last = shapeFold first 0 0 last := by
        simp [shapeFold, fold]
      rw [this]
      exact ⟨min_le_left _ _, le_max_left _ _⟩
    · have := fold_unbounded hz hm.le [(first, 1)] (last.toList.map ((·, 1)))
      simp only [shapeFold_eq, this]
      exact ⟨min_le_right _ _, le_max_right _ _⟩

/-! ## Hulls -/

/--
The least and greatest of a set of results, or `none` if it is empty. Mirrors
`ResultSet::hull` in `evaluator.rs`, whose sets of results this module models
as finite sets.

# Parameters
- `s`: The results.

# Returns
The least and greatest, or `none`.

# Examples
```lean
#guard Bounds.hull {3, 1, 4} == some (1, 4)
#guard Bounds.hull ∅ == none
```
-/
def hull (s : Finset Int) : Option (Int × Int) :=
  if h : s.Nonempty then some (s.min' h, s.max' h) else none

/--
The hull of a set of results holds its least and greatest, which bound every
result.

# Parameters
- `s`: The results.
- `p`: The hull.

# Hypotheses
- `h`: The hull of `s` is `p`.
-/
theorem hull_spec {s : Finset Int} {p : Int × Int} (h : hull s = some p) :
    p.1 ∈ s ∧ p.2 ∈ s ∧ ∀ x ∈ s, p.1 ≤ x ∧ x ≤ p.2 := by
  unfold hull at h
  split at h
  · rename_i hs
    cases h
    exact ⟨Finset.min'_mem _ hs, Finset.max'_mem _ hs,
      fun x hx => ⟨Finset.min'_le _ _ hx, Finset.le_max' _ _ hx⟩⟩
  · cases h

/--
A set of results that holds some result has a hull, which bounds it.

# Parameters
- `s`: The results.
- `x`: A result.

# Hypotheses
- `hx`: `x` lies in `s`.
-/
theorem exists_hull {s : Finset Int} {x : Int} (hx : x ∈ s) :
    ∃ p, hull s = some p ∧ p.1 ≤ x ∧ x ≤ p.2 := by
  refine ⟨(s.min' ⟨x, hx⟩, s.max' ⟨x, hx⟩), ?_, Finset.min'_le _ _ hx,
    Finset.le_max' _ _ hx⟩
  have hs : s.Nonempty := ⟨x, hx⟩
  simp only [hull, hs, ↓reduceDIte]

/-- An empty set of results has no hull. -/
private theorem hull_eq_none {s : Finset Int} (h : hull s = none) :
    s = ∅ := by
  unfold hull at h
  split at h
  · cases h
  · exact Finset.not_nonempty_iff_eq_empty.mp ‹_›

/-- The bounds of a hull are ordered. -/
private theorem hull_le {s : Finset Int} {p : Int × Int}
    (h : Bounds.hull s = some p) : p.1 ≤ p.2 :=
  ((hull_spec h).2.2 _ (hull_spec h).1).2

/-- The bounds of a hull, or of a default whose bounds are ordered, are
ordered. -/
private theorem getD_le {s : Finset Int} {d : Int × Int} (hd : d.1 ≤ d.2) :
    ((Bounds.hull s).getD d).1 ≤ ((Bounds.hull s).getD d).2 := by
  cases h : Bounds.hull s with
  | none => exact hd
  | some p => exact hull_le h

/--
Whether a value lies within the hull of a set of results: between some
result and some other.

# Parameters
- `s`: The results.
- `v`: The value.
-/
def InHull (s : Finset Int) (v : Int) : Prop :=
  ∃ a ∈ s, ∃ b ∈ s, a ≤ v ∧ v ≤ b

/--
A value within the hull of a set of results lies within its least and its
greatest, as every reader of the totals of an explode reads them, through
their hull.

# Parameters
- `s`: The results.
- `v`: The value.

# Hypotheses
- `h`: `v` lies within the hull of `s`.
-/
theorem InHull.hull {s : Finset Int} {v : Int} (h : InHull s v) :
    ∃ p, Bounds.hull s = some p ∧ p.1 ≤ v ∧ v ≤ p.2 := by
  obtain ⟨a, ha, b, hb, hav, hvb⟩ := h
  obtain ⟨p, hp, h₁, -⟩ := exists_hull ha
  have := (hull_spec hp).2.2 b hb
  exact ⟨p, hp, h₁.trans hav, hvb.trans this.2⟩

/-- A value within the hull of some results lies within the hull of more. -/
theorem InHull.mono {s s' : Finset Int} {v : Int} (h : InHull s v)
    (hs : s ⊆ s') : InHull s' v := by
  obtain ⟨a, ha, b, hb, h⟩ := h
  exact ⟨a, hs ha, b, hs hb, h⟩

/-- A result lies within the hull of its set. -/
theorem InHull.of_mem {s : Finset Int} {v : Int} (h : v ∈ s) : InHull s v :=
  ⟨v, h, v, h, le_rfl, le_rfl⟩

/-- A value between the lesser of two results and the greater of two lies
within the hull. -/
private theorem InHull.of_min_max {s : Finset Int} {v a b a' b' : Int}
    (ha : a ∈ s) (hb : b ∈ s) (ha' : a' ∈ s) (hb' : b' ∈ s)
    (h₁ : Min.min a b ≤ v) (h₂ : v ≤ Max.max a' b') : InHull s v := by
  refine ⟨Min.min a b, ?_, Max.max a' b', ?_, h₁, h₂⟩
  · rcases min_choice a b with h | h <;> rw [h] <;> assumption
  · rcases max_choice a' b' with h | h <;> rw [h] <;> assumption

/--
Whether a set of results bounds a set of values soundly, and exactly in its
hull: every value lies within the hull of the results, and every result lies
between two values, so that the least and greatest results are values
(`Exact.hull`).

# Parameters
- `s`: The results.
- `A`: The values.
-/
def Exact (s : Finset Int) (A : Set Int) : Prop :=
  (∀ a ∈ A, InHull s a) ∧ ∀ v ∈ s, (∃ a ∈ A, a ≤ v) ∧ ∃ b ∈ A, v ≤ b

/--
The hull of a set of results that bounds some values exactly is the least
and the greatest of the values.

# Parameters
- `s`: The results.
- `A`: The values.
- `p`: The hull.

# Hypotheses
- `h`: `s` bounds `A` exactly.
- `hp`: The hull of `s` is `p`.
-/
theorem Exact.hull {s : Finset Int} {A : Set Int} {p : Int × Int}
    (h : Exact s A) (hp : Bounds.hull s = some p) : p.1 ∈ A ∧ p.2 ∈ A := by
  obtain ⟨h₁, h₂, h₃⟩ := hull_spec hp
  obtain ⟨⟨a, ha, hal⟩, -⟩ := h.2 _ h₁
  obtain ⟨-, ⟨b, hb, hbl⟩⟩ := h.2 _ h₂
  obtain ⟨u, hu, -, -, hua, -⟩ := h.1 a ha
  obtain ⟨-, -, w, hw, -, hbw⟩ := h.1 b hb
  have := (h₃ u hu).1
  have := (h₃ w hw).2
  exact ⟨(show a = p.1 by omega) ▸ ha, (show b = p.2 by omega) ▸ hb⟩

/--
Results that bound values exactly, united, bound the values united exactly,
as `RollingRecordBounds::explode` in `evaluator.rs` unites its answers for
each pair of a die and a criterion.

# Parameters
- `s`, `s'`: The results.
- `A`, `A'`: The values.

# Hypotheses
- `h`: `s` bounds `A` exactly.
- `h'`: `s'` bounds `A'` exactly.
-/
theorem Exact.union {s s' : Finset Int} {A A' : Set Int} (h : Exact s A)
    (h' : Exact s' A') : Exact (s ∪ s') (A ∪ A') := by
  refine ⟨fun a ha => ?_, fun v hv => ?_⟩
  · rcases ha with ha | ha
    · exact (h.1 a ha).mono Finset.subset_union_left
    · exact (h'.1 a ha).mono Finset.subset_union_right
  · rcases Finset.mem_union.mp hv with hv | hv
    · obtain ⟨⟨a, ha, h₁⟩, ⟨b, hb, h₂⟩⟩ := h.2 v hv
      exact ⟨⟨a, .inl ha, h₁⟩, ⟨b, .inl hb, h₂⟩⟩
    · obtain ⟨⟨a, ha, h₁⟩, ⟨b, hb, h₂⟩⟩ := h'.2 v hv
      exact ⟨⟨a, .inr ha, h₁⟩, ⟨b, .inr hb, h₂⟩⟩

/-! ## Explosions -/

/-- An explode of the chains still kept by a rolling record that no explode
has extended yet, each of which holds one kept die, over the dice and criteria
that it may resolve to, for the purpose of bounding the totals of the chains
that it grows. Mirrors `Explosion` in `evaluator.rs`. -/
structure Explosion where
  /-- The results that a die still kept may show. -/
  results : Finset Int
  /-- The faces that an extension may show. -/
  faces : Finset Int
  /-- The results that some criterion matches. -/
  possible : Finset Int
  /-- The results that every criterion matches. -/
  certain : Finset Int
  /-- The values that a saturated chain may take, which are none if no chain
  may saturate. -/
  saturated : Finset Int
  /-- The bounds of the limit, or `none` for no limit. -/
  limit : Option (Int × Int)

namespace Explosion

/--
The shapes of the chains whose extreme totals bound those of every chain that
the explode may grow from a matching die: each the number of middle dice, and
the bounds of the last die, if the chain has one. Under a limit, the die alone
if the limit may be `0`; the die and one die that does not match, if the limit
may exceed `1`; and the chain that spends each end of the range of limits, at
least `1`, whose last die may show any face. Without limit, the die and one
die that does not match, after no middle dice or after `unboundedDice` of
them. Mirrors the shapes in `Explosion::totals` in `evaluator.rs`.

# Parameters
- `e`: The explosion.
- `faces`: The bounds of the faces.

# Returns
The shapes.
-/
def shapes (e : Explosion) (faces : Int × Int) :
    List (Nat × Option (Int × Int)) :=
  let ends := Bounds.hull (e.faces \ e.certain)
  match e.limit with
  | none =>
    match ends with
    | some ends => [(0, some ends), (unboundedDice, some ends)]
    | none => []
  | some (lo, hi) =>
    (if lo.toNat = 0 then [(0, none)] else []) ++
      (match ends with
        | some ends => if 2 ≤ hi.toNat then [(0, some ends)] else []
        | none => []) ++
      (if 1 ≤ hi.toNat then
        [(Max.max lo.toNat 1 - 1, some faces), (hi.toNat - 1, some faces)]
      else [])

/--
The fold of a shape with each die at one end of its bounds, the least or the
greatest.

# Parameters
- `sel`: The end of the bounds, `Prod.fst` or `Prod.snd`.
- `starts`: The bounds of the first die.
- `middles`: The bounds of each middle die.
- `shape`: The shape.

# Returns
The fold.
-/
def extreme (sel : Int × Int → Int) (starts middles : Int × Int)
    (shape : Nat × Option (Int × Int)) : Int :=
  shapeFold (sel starts) (sel middles) shape.1 (shape.2.map sel)

/--
The totals that a chain of a shape may show, between its extremes.

# Parameters
- `starts`: The bounds of the first die.
- `middles`: The bounds of each middle die.
- `shape`: The shape.

# Returns
The totals.
-/
def interval (starts middles : Int × Int)
    (shape : Nat × Option (Int × Int)) : Finset Int :=
  Finset.Icc (extreme Prod.fst starts middles shape)
    (extreme Prod.snd starts middles shape)

/--
The totals that a chain still kept may show after the explode: each result
that some criterion does not match, alone; the values of saturated chains;
and the totals of each shape, between its extremes. Their hull covers every
total that a chain may show (`totals_sound`), and is exact if the die and
criterion are each one (`pair_exact`), but they may leave gaps between
the shapes that some chain fills. Mirrors `Explosion::totals` in
`evaluator.rs`.

# Parameters
- `e`: The explosion.

# Returns
The totals.

# Examples
```lean
-- A six that explodes on six, at most three times: a chain that ends after
-- two extensions totals `13` to `17`, which lie in no shape, but within the
-- hull.
#guard (Bounds.Explosion.totals { results := {6}, faces := {1, 2, 3, 4, 5, 6},
    possible := {6}, certain := {6}, saturated := ∅,
    limit := some (3, 3) }) ==
  {7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24}
```
-/
def totals (e : Explosion) : Finset Int :=
  match Bounds.hull (e.results ∩ e.possible) with
  | none => e.results \ e.certain
  | some starts =>
    let middles := (Bounds.hull (e.faces ∩ e.possible)).getD starts
    let faces := (Bounds.hull e.faces).getD starts
    (e.shapes faces).foldl (fun acc shape => acc ∪ interval starts middles shape)
      (e.results \ e.certain ∪ e.saturated)

/-- The union of intervals over some shapes holds what it starts with, and
each interval. -/
private theorem mem_foldl_union {starts middles : Int × Int} :
    ∀ (shapes : List (Nat × Option (Int × Int))) (init : Finset Int) (v : Int),
      v ∈ shapes.foldl (fun acc shape => acc ∪ interval starts middles shape)
          init ↔
        v ∈ init ∨ ∃ shape ∈ shapes, v ∈ interval starts middles shape
  | [], init, v => by simp
  | shape :: shapes, init, v => by
    rw [List.foldl_cons, mem_foldl_union shapes, Finset.mem_union]
    simp only [List.mem_cons, exists_eq_or_imp]
    tauto

/-- The totals, when some result may match, by the shapes. -/
private theorem mem_totals {e : Explosion} {starts : Int × Int}
    (h : Bounds.hull (e.results ∩ e.possible) = some starts) (v : Int) :
    v ∈ e.totals ↔ v ∈ e.results \ e.certain ∨ v ∈ e.saturated ∨
      ∃ shape ∈ e.shapes ((Bounds.hull e.faces).getD starts),
        v ∈ interval starts ((Bounds.hull (e.faces ∩ e.possible)).getD starts)
          shape := by
  simp only [totals, h, mem_foldl_union, Finset.mem_union, or_assoc]

/-- Every result that every criterion may fail to match is among the
totals. -/
private theorem mem_totals_alone {e : Explosion} {x : Int}
    (hx : x ∈ e.results) (hc : x ∉ e.certain) : x ∈ e.totals := by
  have hs : x ∈ e.results \ e.certain := Finset.mem_sdiff.mpr ⟨hx, hc⟩
  unfold totals
  split
  · exact hs
  · exact (mem_foldl_union _ _ _).mpr (.inl (Finset.mem_union_left _ hs))

/-- Each extreme of each shape lies among the totals, if its bounds are
ordered. -/
private theorem extreme_mem_totals {e : Explosion} {starts : Int × Int}
    (h : Bounds.hull (e.results ∩ e.possible) = some starts)
    (hs : starts.1 ≤ starts.2)
    (hm : ((Bounds.hull (e.faces ∩ e.possible)).getD starts).1 ≤
      ((Bounds.hull (e.faces ∩ e.possible)).getD starts).2)
    {shape : Nat × Option (Int × Int)}
    (hsh : shape ∈ e.shapes ((Bounds.hull e.faces).getD starts))
    (hl : Option.Rel (· ≤ ·) (shape.2.map Prod.fst) (shape.2.map Prod.snd)) :
    ∀ sel ∈ [Prod.fst, Prod.snd], extreme sel starts
      ((Bounds.hull (e.faces ∩ e.possible)).getD starts) shape ∈ e.totals := by
  have hle := shapeFold_mono shape.1 hs hm hl
  intro sel hsel
  refine (mem_totals h _).mpr (.inr (.inr ⟨shape, hsh, ?_⟩))
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hsel
  rcases hsel with rfl | rfl <;>
    simp only [interval, Finset.mem_Icc, extreme] <;> exact ⟨by omega, by omega⟩

/-- The shapes, under a limit. -/
private theorem mem_shapes_some {e : Explosion} {lo hi : Int}
    (hl : e.limit = some (lo, hi)) (faces : Int × Int)
    (shape : Nat × Option (Int × Int)) :
    shape ∈ e.shapes faces ↔
      (lo.toNat = 0 ∧ shape = (0, none)) ∨
      (∃ ends, Bounds.hull (e.faces \ e.certain) = some ends ∧
        2 ≤ hi.toNat ∧ shape = (0, some ends)) ∨
      (1 ≤ hi.toNat ∧ (shape = (Max.max lo.toNat 1 - 1, some faces) ∨
        shape = (hi.toNat - 1, some faces))) := by
  simp only [shapes, hl]
  cases Bounds.hull (e.faces \ e.certain) <;>
    simp only [List.mem_append, List.mem_ite_nil_right, List.mem_cons,
      List.not_mem_nil, or_false, reduceCtorEq, false_and, exists_false,
      false_or, Option.some.injEq, exists_eq_left']
  all_goals tauto

/-- The shapes, without limit. -/
private theorem mem_shapes_none {e : Explosion} (hl : e.limit = none)
    (faces : Int × Int) (shape : Nat × Option (Int × Int)) :
    shape ∈ e.shapes faces ↔
      ∃ ends, Bounds.hull (e.faces \ e.certain) = some ends ∧
        (shape = (0, some ends) ∨ shape = (unboundedDice, some ends)) := by
  simp only [shapes, hl]
  cases Bounds.hull (e.faces \ e.certain) <;> simp

end Explosion

/--
The limits that the bounds of a limit admit, as the specification resolves
them: without limit, none; and otherwise each value within the bounds, or
`0` if it is negative.

# Parameters
- `limit`: The bounds of the limit, or `none` for no limit.
- `l`: A limit, or `none` for no limit.
-/
def Limits : Option (Int × Int) → Option Nat → Prop
  | none, l => l = none
  | some (lo, hi), l => ∃ n, lo ≤ n ∧ n ≤ hi ∧ l = some n.toNat

/-! ## Soundness -/

/-- A chain of one die grows exactly when the criterion matches its die. -/
private theorem grows_single (x : Int) (c : Criterion) :
    (Chain.single x).grows c = c.matches x := by
  simp [Chain.grows, Chain.single, Chain.saturates, Chain.last]

/--
A chain of one die grown by middle dice that show alike and a last die totals
the fold of its shape.

# Parameters
- `x`: The first die.
- `y`: Each middle die.
- `z`: The last die.
- `m`: The number of middle dice.
-/
theorem total_growChain_replicate (x y z : Int) (m : Nat) :
    (growChain (.single x) (List.replicate m y, z)).total =
      shapeFold x y m (some z) := by
  rw [total_growChain_single, shapeFold]
  refine fold_congr ?_
  simp only [List.map_replicate, Roll.expand, List.flatMap_cons,
    List.replicate_one, Option.toList_some,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    List.flatMap_nil, List.append_nil]
  rw [show List.flatMap (fun p : Int × Nat => List.replicate p.2 p.1)
      (List.replicate m (y, 1)) = List.replicate m y by
    simpa [Roll.expand] using expand_singles (List.replicate m y)]
  exact (List.perm_append_singleton z _).symm.cons x

/--
The hull of the totals of an explosion covers the total of every chain that
an explode may make of a chain of one die, which may show any result of the
explosion: for any die whose faces the explosion's faces include, any
criterion that matches only results that some criterion of the explosion
matches, and every result that every one does, and any limit that the
explosion's bounds admit. So it covers every one die and criterion of a
predicate, as `RollingRecordBounds::explode` in `evaluator.rs` explodes them
all at once, when they are too many to explode one pair at a time.

# Parameters
- `e`: The explosion.
- `d`: The die.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `x`: The result of the die.
- `ch`: What the explode makes of its chain.

# Hypotheses
- `hx`: The explosion admits the result.
- `hxF`: The result is a face of the die.
- `hx32`: The result lies within the `i32` range.
- `hF`: The explosion admits every face of the die.
- `hP`: Some criterion of the explosion matches every face that `c` does.
- `hC`: `c` matches every face that every criterion of the explosion does.
- `hS`: The explosion admits the saturated value of the die if the explode,
  without limit, matches every face.
- `hL`: The explosion's bounds admit the limit.
- `hb`: The explode may make `ch` of the chain of the result.
-/
theorem totals_sound {e : Explosion} {d : Die} {c : Criterion}
    {l : Option Nat} {x : Int} {ch : Chain} (hx : x ∈ e.results)
    (hxF : x ∈ d.faces) (hx32 : i32Min ≤ x ∧ x ≤ i32Max)
    (hF : ∀ y ∈ d.faces, y ∈ e.faces)
    (hP : ∀ y ∈ d.faces, c.matches y = true → y ∈ e.possible)
    (hC : ∀ y ∈ d.faces, y ∈ e.certain → c.matches y = true)
    (hS : l = none → d.facesWithin c = true →
      d.saturatedValue ∈ e.saturated)
    (hL : Limits e.limit l) (hb : Becomes d c l (.single x) ch) :
    InHull e.totals ch.total := by
  cases hm : c.matches x with
  | false =>
    -- A die that the criterion does not match stays alone.
    have hch : ch = .single x := by
      cases l with
      | some L => simpa [Becomes, grows_single, hm] using hb
      | none =>
        have hw : d.facesWithin c = false := by
          simp only [Die.facesWithin, List.all_eq_false]
          exact ⟨x, hxF, by simp [hm]⟩
        simpa [Becomes, hw, grows_single, hm] using hb
    subst hch
    rw [total_single hx32]
    exact .of_mem (Explosion.mem_totals_alone hx fun h => by
      simp [hC x hxF h] at hm)
  | true =>
    have hxs : x ∈ e.results ∩ e.possible :=
      Finset.mem_inter.mpr ⟨hx, hP x hxF hm⟩
    obtain ⟨starts, hst, hs₁, hs₂⟩ := exists_hull hxs
    have hsle : starts.1 ≤ starts.2 := by omega
    have hmle := getD_le (s := e.faces ∩ e.possible) hsle
    have hext := fun {shape} hsh hl =>
      Explosion.extreme_mem_totals hst hsle hmle (shape := shape) hsh hl
    -- Each middle die, each last die, and each last die that ends a chain,
    -- within their bounds.
    have hmb : ∀ z ∈ d.faces, c.matches z = true →
        ((hull (e.faces ∩ e.possible)).getD starts).1 ≤ z ∧
          z ≤ ((hull (e.faces ∩ e.possible)).getD starts).2 := by
      intro z hz hzm
      obtain ⟨p, hp, h₁, h₂⟩ :=
        exists_hull (Finset.mem_inter.mpr ⟨hF z hz, hP z hz hzm⟩)
      simp only [hp, Option.getD_some]
      exact ⟨h₁, h₂⟩
    have hfb : ∀ z ∈ d.faces, ((hull e.faces).getD starts).1 ≤ z ∧
        z ≤ ((hull e.faces).getD starts).2 := by
      intro z hz
      obtain ⟨p, hp, h₁, h₂⟩ := exists_hull (hF z hz)
      simp only [hp, Option.getD_some]
      exact ⟨h₁, h₂⟩
    have heb : ∀ z ∈ d.faces, c.matches z = false → ∃ ends,
        hull (e.faces \ e.certain) = some ends ∧ ends.1 ≤ z ∧ z ≤ ends.2 ∧
          ((hull e.faces).getD starts).1 ≤ ends.1 ∧
          ends.2 ≤ ((hull e.faces).getD starts).2 := by
      intro z hz hzm
      have hzc : z ∈ e.faces \ e.certain := Finset.mem_sdiff.mpr
        ⟨hF z hz, fun h => by simp [hC z hz h] at hzm⟩
      obtain ⟨ends, hends, h₁, h₂⟩ := exists_hull hzc
      obtain ⟨he₁, he₂, -⟩ := hull_spec hends
      obtain ⟨p, hp, -⟩ := exists_hull (hF z hz)
      have hp' := (hull_spec hp).2.2
      simp only [hp, Option.getD_some]
      exact ⟨ends, hends, h₁, h₂, (hp' _ (Finset.mem_sdiff.mp he₁).1).1,
        (hp' _ (Finset.mem_sdiff.mp he₂).1).2⟩
    have hfle := getD_le (s := e.faces) hsle
    cases l with
    | some L =>
      cases hlim : e.limit with
      | none => simp [Limits, hlim] at hL
      | some b =>
        obtain ⟨lo, hi⟩ := b
        obtain ⟨n, hlo, hhi, hn⟩ : ∃ n, lo ≤ n ∧ n ≤ hi ∧ some L = some n.toNat :=
          by simpa [Limits, hlim] using hL
        cases hn
        have hL₁ : lo.toNat ≤ n.toNat := Int.toNat_le_toNat hlo
        have hL₂ : n.toNat ≤ hi.toNat := Int.toNat_le_toNat hhi
        have hsh := Explosion.mem_shapes_some hlim ((hull e.faces).getD starts)
        by_cases hg : 0 < n.toNat
        · simp only [Becomes, grows_single, hm, hg, and_self, ↓reduceIte] at hb
          obtain ⟨t, ⟨-, hdm, hy, hlen, hend⟩, rfl⟩ := hb
          obtain ⟨hlow, hup⟩ := shapeFold_le_total (x := x) (t := t) ⟨hs₁, hs₂⟩
            (fun z hz => hmb z (hdm z hz).1 (hdm z hz).2) (hfb t.2 hy)
          by_cases hshort : t.1.length + 1 < n.toNat
          · obtain ⟨ends, hends, he₁, he₂, hfe₁, hfe₂⟩ :=
              heb t.2 hy (hend hshort)
            obtain ⟨hlow, hup⟩ := shapeFold_le_total (x := x) (t := t)
              ⟨hs₁, hs₂⟩ (fun z hz => hmb z (hdm z hz).1 (hdm z hz).2)
              ⟨he₁, he₂⟩
            have hA := (hsh (0, some ends)).mpr (.inr (.inl ⟨ends, hends,
              by omega, rfl⟩))
            have hB := (hsh (hi.toNat - 1, some ((hull e.faces).getD starts))).mpr
              (.inr (.inr ⟨by omega, .inr rfl⟩))
            have hAl : Option.Rel (· ≤ ·) ((0, some ends).2.map Prod.fst)
                ((0, some ends).2.map Prod.snd) := .some (hull_le hends)
            have hBl : Option.Rel (· ≤ ·)
                ((hi.toNat - 1, some ((hull e.faces).getD starts)).2.map Prod.fst)
                ((hi.toNat - 1, some ((hull e.faces).getD starts)).2.map
                  Prod.snd) := .some hfle
            have h₁ := (shapeFold_short starts.1
              ((hull (e.faces ∩ e.possible)).getD starts).1
              (m := t.1.length) (M := hi.toNat - 1) (by omega) hfe₁
              (le_refl ends.1)).1
            have h₂ := (shapeFold_short starts.2
              ((hull (e.faces ∩ e.possible)).getD starts).2
              (m := t.1.length) (M := hi.toNat - 1) (by omega) (le_refl ends.2)
              hfe₂).2
            exact InHull.of_min_max (hext hA hAl _ (by simp))
              (hext hB hBl _ (by simp)) (hext hA hAl _ (by simp))
              (hext hB hBl _ (by simp)) (h₁.trans hlow) (hup.trans h₂)
          · have hA := (hsh (Max.max lo.toNat 1 - 1,
              some ((hull e.faces).getD starts))).mpr
              (.inr (.inr ⟨by omega, .inl rfl⟩))
            have hB := (hsh (hi.toNat - 1, some ((hull e.faces).getD starts))).mpr
              (.inr (.inr ⟨by omega, .inr rfl⟩))
            have h₁ := (shapeFold_between starts.1
              ((hull (e.faces ∩ e.possible)).getD starts).1
              (a := Max.max lo.toNat 1 - 1) (m := t.1.length)
              (b := hi.toNat - 1) (by omega) (by omega)
              (some ((hull e.faces).getD starts).1)).1
            have h₂ := (shapeFold_between starts.2
              ((hull (e.faces ∩ e.possible)).getD starts).2
              (a := Max.max lo.toNat 1 - 1) (m := t.1.length)
              (b := hi.toNat - 1) (by omega) (by omega)
              (some ((hull e.faces).getD starts).2)).2
            exact InHull.of_min_max (hext hA (.some hfle) _ (by simp))
              (hext hB (.some hfle) _ (by simp)) (hext hA (.some hfle) _ (by simp))
              (hext hB (.some hfle) _ (by simp)) (h₁.trans hlow) (hup.trans h₂)
        · simp only [Becomes, grows_single, hm, hg, false_and, ↓reduceIte] at hb
          subst hb
          have hA := (hsh (0, none)).mpr (.inl ⟨by omega, rfl⟩)
          refine ⟨_, hext hA .none Prod.fst (by simp), _,
            hext hA .none Prod.snd (by simp), ?_, ?_⟩
          · rw [total_single_eq x ((hull (e.faces ∩ e.possible)).getD starts).1]
            exact shapeFold_mono 0 hs₁ le_rfl .none
          · rw [total_single_eq x ((hull (e.faces ∩ e.possible)).getD starts).2]
            exact shapeFold_mono 0 hs₂ le_rfl .none
    | none =>
      have hlim : e.limit = none := by
        cases hlim : e.limit with
        | none => rfl
        | some b => simp [Limits, hlim] at hL
      by_cases hw : d.facesWithin c = true
      · simp only [Becomes, hw, ↓reduceIte] at hb
        subst hb
        exact .of_mem ((Explosion.mem_totals hst _).mpr (.inr (.inl
          (hS rfl hw))))
      · simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte, grows_single,
          hm] at hb
        obtain ⟨t, ⟨-, hdm, hy, hend⟩, rfl⟩ := hb
        obtain ⟨ends, hends, he₁, he₂, -, -⟩ := heb t.2 hy hend
        obtain ⟨hlow, hup⟩ := shapeFold_le_total (x := x) (t := t) ⟨hs₁, hs₂⟩
          (fun z hz => hmb z (hdm z hz).1 (hdm z hz).2) ⟨he₁, he₂⟩
        have hsh := Explosion.mem_shapes_none hlim ((hull e.faces).getD starts)
        have hA := (hsh (0, some ends)).mpr ⟨ends, hends, .inl rfl⟩
        have hB := (hsh (unboundedDice, some ends)).mpr ⟨ends, hends, .inr rfl⟩
        have hl : Option.Rel (· ≤ ·) ((some ends).map Prod.fst)
            ((some ends).map Prod.snd) := .some (hull_le hends)
        have h₁ := (shapeFold_unbounded starts.1
          ((hull (e.faces ∩ e.possible)).getD starts).1 t.1.length
          (some ends.1)).1
        have h₂ := (shapeFold_unbounded starts.2
          ((hull (e.faces ∩ e.possible)).getD starts).2 t.1.length
          (some ends.2)).2
        exact InHull.of_min_max (hext hA hl _ (by simp)) (hext hB hl _ (by simp))
          (hext hA hl _ (by simp)) (hext hB hl _ (by simp)) (h₁.trans hlow)
          (hup.trans h₂)

/--
The hull of the totals of an explosion covers the total of every chain still
kept that an explode leaves of a record that no explode has extended yet,
each of whose chains still kept holds one die showing a result of the
explosion, and every reader of the totals reads only their hull: so the bounds
evaluator bounds the sums, drops, and keeps of the record soundly.

# Parameters
- `r`: The record.
- `e`: The explosion.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.

# Hypotheses
- `hF`: The faces of the record's die lie within the `i32` range, and the
  explosion admits each.
- `hk`: Each chain still kept holds one die, showing a face of the die that
  the explosion admits.
- `hP`, `hC`, `hS`, `hL`: As for `totals_sound`.
-/
theorem explode_sound {r : Record} {e : Explosion} {c : Criterion}
    {l : Option Nat} (hF : ∀ y ∈ r.die.faces,
      (i32Min ≤ y ∧ y ≤ i32Max) ∧ y ∈ e.faces)
    (hk : ∀ ch ∈ r.keptChains, ∃ x ∈ e.results, x ∈ r.die.faces ∧
      ch = .single x)
    (hP : ∀ y ∈ r.die.faces, c.matches y = true → y ∈ e.possible)
    (hC : ∀ y ∈ r.die.faces, y ∈ e.certain → c.matches y = true)
    (hS : l = none → r.die.facesWithin c = true →
      r.die.saturatedValue ∈ e.saturated)
    (hL : Limits e.limit l) :
    ∀ r' ∈ (explode r c l).support, ∀ ch ∈ r'.keptChains,
      InHull e.totals ch.total := by
  intro r' hr' ch hch
  have hk' : ∀ ch ∈ r.keptChains, NormalChain ch ∧
      ∀ x ∈ ch.dice, x ∈ r.die.faces := fun ch hch => by
    obtain ⟨x, -, hxF, rfl⟩ := hk ch hch
    refine ⟨⟨by simp, fun y hy => ?_⟩, fun y hy => ?_⟩
    · simp only [List.mem_singleton] at hy
      exact hy ▸ (hF x hxF).1
    · simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
        List.mem_singleton] at hy
      exact hy ▸ hxF
  obtain ⟨ch₀, hch₀, hb⟩ :=
    (mem_explode_iff (fun y hy => (hF y hy).1) hk' ch).mp ⟨r', hr', hch⟩
  obtain ⟨x, hx, hxF, rfl⟩ := hk ch₀ hch₀
  exact totals_sound hx hxF (hF x hxF).1 (fun y hy => (hF y hy).2) hP hC hS hL
    hb

/-! ## Exactness -/

/--
The explosion of one die and one criterion, as `RollingRecordBounds::explode`
in `evaluator.rs` explodes each pair when they are few enough: the results
that the die may show, its faces, the criterion as both the results that some
criterion matches and those that every one does, and the die's saturated
value, if the explode, without limit, matches every face and some die is
still kept.

# Parameters
- `results`: The results that a die still kept may show.
- `d`: The die.
- `criterion`: The results that the criterion matches.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The explosion.
-/
def pair (results : Finset Int) (d : Die) (criterion : Finset Int)
    (limit : Option (Int × Int)) : Explosion :=
  { results := results ∩ d.faces.toFinset
    faces := d.faces.toFinset
    possible := criterion
    certain := criterion
    saturated :=
      if limit.isNone ∧ d.faces.toFinset ⊆ criterion ∧
          (results ∩ d.faces.toFinset).Nonempty then {d.saturatedValue}
      else ∅
    limit }

/--
The totals of the chains that an explode of a die by a criterion may make of a
chain of one die, showing any of some results, under any limit that some
bounds admit.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `results`: The results.
- `limit`: The bounds of the limit, or `none` for no limit.
-/
def reach (d : Die) (c : Criterion) (results : Finset Int)
    (limit : Option (Int × Int)) : Set Int :=
  {v | ∃ x ∈ results, ∃ l, Limits limit l ∧
    ∃ ch, Becomes d c l (.single x) ch ∧ ch.total = v}

/-- Bounds of a limit that are ordered admit some limit. -/
private theorem exists_limits {limit : Option (Int × Int)}
    (hl : ∀ lo hi, limit = some (lo, hi) → lo ≤ hi) : ∃ l, Limits limit l := by
  cases limit with
  | none => exact ⟨none, rfl⟩
  | some b =>
    obtain ⟨lo, hi⟩ := b
    exact ⟨_, lo, le_rfl, hl lo hi rfl, rfl⟩

/-- A die that the criterion does not match stays alone, under any limit. -/
private theorem becomes_alone {d : Die} {c : Criterion} {x : Int}
    (hxF : x ∈ d.faces) (hm : c.matches x = false) (l : Option Nat) :
    Becomes d c l (.single x) (.single x) := by
  cases l with
  | some L => simp [Becomes, grows_single, hm]
  | none =>
    have hw : d.facesWithin c = false := by
      simp only [Die.facesWithin, List.all_eq_false]
      exact ⟨x, hxF, by simp [hm]⟩
    simp [Becomes, hw, grows_single, hm]

/-- A result alone that the criterion does not match is reached. -/
private theorem alone_reach {d : Die} {c : Criterion} {R : Finset Int}
    {limit : Option (Int × Int)} (hF : ∀ y ∈ d.faces, i32Min ≤ y ∧ y ≤ i32Max)
    (hl : ∀ lo hi, limit = some (lo, hi) → lo ≤ hi) {x : Int}
    (hx : x ∈ R ∩ d.faces.toFinset) (hm : c.matches x = false) :
    x ∈ reach d c (R ∩ d.faces.toFinset) limit := by
  have hxF := List.mem_toFinset.mp (Finset.mem_inter.mp hx).2
  obtain ⟨l, hL⟩ := exists_limits hl
  exact ⟨x, hx, l, hL, _, becomes_alone hxF hm l, total_single (hF x hxF)⟩

/-- A shape of matching dice and a last die is reached, under a limit, if the
limit is positive, admits as many middle dice, and admits a last die that
matches only if it is spent. -/
private theorem limited_reach {d : Die} {c : Criterion} {R : Finset Int}
    {lo hi n : Int} {x y z : Int} {m : Nat}
    (hx : x ∈ R ∩ d.faces.toFinset) (hxm : c.matches x = true)
    (hy : y ∈ d.faces) (hym : c.matches y = true) (hz : z ∈ d.faces)
    (hn : lo ≤ n ∧ n ≤ hi) (hm : m < n.toNat)
    (hzm : m + 1 < n.toNat → c.matches z = false) :
    shapeFold x y m (some z) ∈ reach d c (R ∩ d.faces.toFinset) (some (lo, hi)) := by
  refine ⟨x, hx, some n.toNat, ⟨n, hn.1, hn.2, rfl⟩, _, ?_,
    total_growChain_replicate x y z m⟩
  simp only [Becomes, grows_single, hxm, show 0 < n.toNat by omega, and_self,
    ↓reduceIte]
  refine ⟨_, ⟨List.pairwise_replicate.mpr (.inr le_rfl), fun w hw => ?_, hz,
    by simpa using hm, by simpa using hzm⟩, rfl⟩
  rw [List.eq_of_mem_replicate hw]
  exact ⟨hy, hym⟩

/-- A shape of matching dice and a last die that does not match is reached,
without limit. -/
private theorem unlimited_reach {d : Die} {c : Criterion} {R : Finset Int}
    {x y z : Int} (m : Nat) (hx : x ∈ R ∩ d.faces.toFinset)
    (hxm : c.matches x = true) (hy : y ∈ d.faces) (hym : c.matches y = true)
    (hz : z ∈ d.faces) (hzm : c.matches z = false) :
    shapeFold x y m (some z) ∈ reach d c (R ∩ d.faces.toFinset) none := by
  refine ⟨x, hx, none, rfl, _, ?_, total_growChain_replicate x y z m⟩
  have hw : d.facesWithin c = false := by
    simp only [Die.facesWithin, List.all_eq_false]
    exact ⟨z, hz, by simp [hzm]⟩
  simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte, grows_single, hxm]
  refine ⟨_, ⟨List.pairwise_replicate.mpr (.inr le_rfl), fun w hw => ?_, hz,
    hzm⟩, rfl⟩
  rw [List.eq_of_mem_replicate hw]
  exact ⟨hy, hym⟩

/-- Each extreme of each shape of the explosion of one die and one criterion is
reached. -/
private theorem extreme_reach {R C : Finset Int} {d : Die} {c : Criterion}
    {limit : Option (Int × Int)}
    (hC : ∀ y ∈ d.faces, y ∈ C ↔ c.matches y = true)
    (hl : ∀ lo hi, limit = some (lo, hi) → lo ≤ hi) {starts : Int × Int}
    (hst : hull ((pair R d C limit).results ∩ (pair R d C limit).possible) =
      some starts)
    {shape : Nat × Option (Int × Int)}
    (hsh : shape ∈ (pair R d C limit).shapes
      ((hull (pair R d C limit).faces).getD starts))
    (sel : Int × Int → Int) (hsel : sel = Prod.fst ∨ sel = Prod.snd) :
    Explosion.extreme sel starts
        ((hull ((pair R d C limit).faces ∩ (pair R d C limit).possible)).getD
          starts) shape ∈ reach d c (R ∩ d.faces.toFinset) limit := by
  have hmem : ∀ {s : Finset Int} {p : Int × Int}, hull s = some p → sel p ∈ s :=
    fun hp => by rcases hsel with rfl | rfl <;> simp [hull_spec hp]
  -- The first die, a result that matches.
  have hx := hmem hst
  simp only [pair, Finset.mem_inter, List.mem_toFinset] at hx
  have hxm := (hC _ hx.1.2).mp hx.2
  -- Each middle die, a face that matches.
  obtain ⟨pm, hpm, -⟩ := exists_hull (s := d.faces.toFinset ∩ C)
    (Finset.mem_inter.mpr ⟨List.mem_toFinset.mpr hx.1.2, hx.2⟩)
  have hy := hmem hpm
  simp only [Finset.mem_inter, List.mem_toFinset] at hy
  have hym := (hC _ hy.1).mp hy.2
  -- Each last die of a spent chain, a face.
  obtain ⟨pf, hpf, -⟩ := exists_hull (s := d.faces.toFinset)
    (List.mem_toFinset.mpr hx.1.2)
  have hf := List.mem_toFinset.mp (hmem hpf)
  -- Each last die that ends a chain, a face that does not match.
  have hend : ∀ {pe : Int × Int}, hull (d.faces.toFinset \ C) = some pe →
      sel pe ∈ d.faces ∧ c.matches (sel pe) = false := fun hpe => by
    have := hmem hpe
    simp only [Finset.mem_sdiff, List.mem_toFinset] at this
    exact ⟨this.1, by simpa [hC _ this.1] using this.2⟩
  have hx' : sel starts ∈ R ∩ d.faces.toFinset :=
    Finset.mem_inter.mpr ⟨hx.1.1, List.mem_toFinset.mpr hx.1.2⟩
  simp only [pair] at hpm hpf hsh ⊢
  simp only [Explosion.extreme, hpm, hpf, Option.getD_some] at hsh ⊢
  cases limit with
  | none =>
    obtain ⟨pe, hpe, rfl | rfl⟩ :=
      (Explosion.mem_shapes_none (e := pair R d C none) rfl pf shape).mp hsh <;>
      exact unlimited_reach _ hx' hxm hy.1 hym (hend hpe).1 (hend hpe).2
  | some b =>
    obtain ⟨lo, hi⟩ := b
    have hlh := hl lo hi rfl
    rcases (Explosion.mem_shapes_some (e := pair R d C (some (lo, hi))) rfl pf
        shape).mp hsh with ⟨h0, rfl⟩ | ⟨pe, hpe, h2, rfl⟩ | ⟨h1, rfl | rfl⟩
    · refine ⟨sel starts, hx', some lo.toNat, ⟨lo, le_rfl, hlh, rfl⟩, _, ?_,
        total_single_eq _ (sel pm)⟩
      simp [Becomes, h0]
    · exact limited_reach hx' hxm hy.1 hym (hend hpe).1 (n := hi) ⟨hlh, le_rfl⟩
        (by omega) (fun _ => (hend hpe).2)
    · by_cases hlo : 1 ≤ lo.toNat
      · exact limited_reach hx' hxm hy.1 hym hf (n := lo) ⟨le_rfl, hlh⟩
          (by omega) (fun h => by omega)
      · have h₁ : lo ≤ 1 := by omega
        have h₂ : 1 ≤ hi := by omega
        have := limited_reach (m := 0) hx' hxm hy.1 hym hf (n := 1) ⟨h₁, h₂⟩
          (by decide) (fun h => by simp at h)
        rwa [show Max.max lo.toNat 1 - 1 = 0 by omega]
    · exact limited_reach hx' hxm hy.1 hym hf (n := hi) ⟨hlh, le_rfl⟩
        (by omega) (fun h => by omega)

/--
The totals of the explosion of one die and one criterion bound the totals of
the chains that the explode may make of a chain of one die exactly in their
hull: their hull covers every such total (`totals_sound`), and its least and
greatest are each the total of some such chain (`Exact.hull`), though the
totals may leave gaps that some chain fills. Exact relative to the results,
which every die still kept is taken to be able to show, and to the bounds of
the limit, every value within which is taken to be possible.

# Parameters
- `R`: The results that a die still kept may show.
- `C`: The results that the criterion matches.
- `d`: The die.
- `c`: The criterion.
- `limit`: The bounds of the limit, or `none` for no limit.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hC`: `C` holds exactly the faces that `c` matches.
- `hl`: The bounds of the limit are ordered.
-/
theorem pair_exact {R C : Finset Int} {d : Die} {c : Criterion}
    {limit : Option (Int × Int)} (hF : ∀ y ∈ d.faces, i32Min ≤ y ∧ y ≤ i32Max)
    (hC : ∀ y ∈ d.faces, y ∈ C ↔ c.matches y = true)
    (hl : ∀ lo hi, limit = some (lo, hi) → lo ≤ hi) :
    Exact (pair R d C limit).totals (reach d c (R ∩ d.faces.toFinset) limit) := by
  refine ⟨?_, fun v hv => ?_⟩
  · rintro _ ⟨x, hx, l, hL, ch, hb, rfl⟩
    have hxF := List.mem_toFinset.mp (Finset.mem_inter.mp hx).2
    refine totals_sound (e := pair R d C limit) hx hxF (hF x hxF)
      (fun y hy => List.mem_toFinset.mpr hy) (fun y hy hm => (hC y hy).mpr hm)
      (fun y hy h => (hC y hy).mp h) (fun hl' hw => ?_) hL hb
    subst hl'
    have hlim : limit = none := by
      cases limit with
      | none => rfl
      | some b => simp [Limits] at hL
    have hsub : d.faces.toFinset ⊆ C := fun y hy => by
      have hy := List.mem_toFinset.mp hy
      exact (hC y hy).mpr (List.all_eq_true.mp hw y hy)
    have hne : (R ∩ d.faces.toFinset).Nonempty := ⟨x, hx⟩
    simp [pair, hlim, hsub, hne]
  · have halone : v ∈ (pair R d C limit).results \ (pair R d C limit).certain →
        (∃ a ∈ reach d c (R ∩ d.faces.toFinset) limit, a ≤ v) ∧
          ∃ b ∈ reach d c (R ∩ d.faces.toFinset) limit, v ≤ b := fun h => by
      simp only [pair, Finset.mem_sdiff] at h
      have hvF := List.mem_toFinset.mp (Finset.mem_inter.mp h.1).2
      have := alone_reach hF hl h.1 (by simpa [hC v hvF] using h.2)
      exact ⟨⟨v, this, le_rfl⟩, ⟨v, this, le_rfl⟩⟩
    cases hst : hull ((pair R d C limit).results ∩ (pair R d C limit).possible)
      with
    | none =>
      simp only [Explosion.totals, hst] at hv
      exact halone hv
    | some starts =>
      rcases (Explosion.mem_totals hst v).mp hv with hv | hv | ⟨shape, hsh, hv⟩
      · exact halone hv
      · simp only [pair] at hv
        split at hv
        · rename_i hcond
          obtain ⟨hnone, hsub, x, hx⟩ := hcond
          rw [Finset.mem_singleton] at hv
          subst hv
          have hlim : limit = none := Option.isNone_iff_eq_none.mp hnone
          have hw : d.facesWithin c = true := List.all_eq_true.mpr fun y hy =>
            (hC y hy).mp (hsub (List.mem_toFinset.mpr hy))
          have : d.saturatedValue ∈ reach d c (R ∩ d.faces.toFinset) limit :=
            ⟨x, hx, none, by simp [Limits, hlim], .saturated d.saturatedValue,
              by simp [Becomes, hw, Chain.single, Chain.saturate], rfl⟩
          exact ⟨⟨_, this, le_rfl⟩, ⟨_, this, le_rfl⟩⟩
        · simp at hv
      · simp only [Explosion.interval, Finset.mem_Icc] at hv
        exact ⟨⟨_, extreme_reach hC hl hst hsh _ (.inl rfl), hv.1⟩,
          ⟨_, extreme_reach hC hl hst hsh _ (.inr rfl), hv.2⟩⟩

/--
The union of the totals of some explosions. Mirrors the union of the answers
for each pair of a die and a criterion in `RollingRecordBounds::explode` in
`evaluator.rs`.

# Parameters
- `es`: The explosions.

# Returns
The union of their totals.
-/
def unite (es : List Explosion) : Finset Int :=
  es.foldl (fun acc e => acc ∪ e.totals) ∅

/--
Uniting explosions whose totals each bound some values exactly bounds the
values united exactly: their hull's least and greatest are the least and
greatest of the values of any of them.

# Parameters
- `is`: The indices of the explosions.
- `f`: The explosion of each index.
- `A`: The values of each index.

# Hypotheses
- `h`: The totals of each explosion bound its values exactly.
-/
theorem unite_exact {ι : Type} (is : List ι) (f : ι → Explosion)
    (A : ι → Set Int) (h : ∀ i ∈ is, Exact (f i).totals (A i)) :
    Exact (unite (is.map f)) {v | ∃ i ∈ is, v ∈ A i} := by
  have : ∀ (acc : Finset Int) (B : Set Int), Exact acc B →
      Exact ((is.map f).foldl (fun acc e => acc ∪ e.totals) acc)
        (B ∪ {v | ∃ i ∈ is, v ∈ A i}) := by
    induction is with
    | nil => intro acc B hB; simpa using hB
    | cons i is ih =>
      intro acc B hB
      rw [List.map_cons, List.foldl_cons]
      have := ih (fun j hj => h j (List.mem_cons_of_mem _ hj)) _ _
        (hB.union (h i List.mem_cons_self))
      convert this using 1
      ext v
      simp only [Set.mem_union, Set.mem_ofPred_eq, List.mem_cons,
        exists_eq_or_imp]
      tauto
  have h₀ : Exact ∅ ∅ := ⟨fun _ h => h.elim, fun _ h => absurd h (by simp)⟩
  simpa [unite] using this ∅ ∅ h₀

/--
The totals of every pair of a die and a criterion, united. Mirrors
`RollingRecordBounds::explode` in `evaluator.rs` when the dice and criteria
are few enough to enumerate.

# Parameters
- `results`: The results that a die still kept may show.
- `dice`: The dice.
- `criteria`: The results that each criterion matches.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The totals.
-/
def explodePairs (results : Finset Int) (dice : List Die)
    (criteria : List (Finset Int)) (limit : Option (Int × Int)) : Finset Int :=
  unite (dice.flatMap fun d => criteria.map fun C => pair results d C limit)

/--
The totals of every pair of a die and a criterion, united, bound the totals
of the chains that the explode of each die by each criterion may make of a
chain of one die exactly in their hull.

# Parameters
- `R`: The results that a die still kept may show.
- `dice`: The dice.
- `criteria`: The results that each criterion matches.
- `crit`: The criterion that each matches.
- `limit`: The bounds of the limit, or `none` for no limit.

# Hypotheses
- `hF`: The faces of every die lie within the `i32` range.
- `hC`: Each criterion's results hold exactly the faces of each die that it
  matches.
- `hl`: The bounds of the limit are ordered.
-/
theorem explodePairs_exact {R : Finset Int} {dice : List Die}
    {criteria : List (Finset Int)} (crit : Finset Int → Criterion)
    {limit : Option (Int × Int)}
    (hF : ∀ d ∈ dice, ∀ y ∈ d.faces, i32Min ≤ y ∧ y ≤ i32Max)
    (hC : ∀ d ∈ dice, ∀ C ∈ criteria, ∀ y ∈ d.faces,
      y ∈ C ↔ (crit C).matches y = true)
    (hl : ∀ lo hi, limit = some (lo, hi) → lo ≤ hi) :
    Exact (explodePairs R dice criteria limit)
      {v | ∃ d ∈ dice, ∃ C ∈ criteria,
        v ∈ reach d (crit C) (R ∩ d.faces.toFinset) limit} := by
  have h := unite_exact (dice.flatMap fun d => criteria.map fun C => (d, C))
    (fun p => pair R p.1 p.2 limit)
    (fun p => reach p.1 (crit p.2) (R ∩ p.1.faces.toFinset) limit)
    (fun p hp => by
      obtain ⟨d, hd, hp⟩ := List.mem_flatMap.mp hp
      obtain ⟨C, hC', rfl⟩ := List.mem_map.mp hp
      exact pair_exact (hF d hd) (hC d hd C hC') hl)
  have hl' : ((dice.flatMap fun d => criteria.map fun C => (d, C)).map
      fun p => pair R p.1 p.2 limit) =
        dice.flatMap fun d => criteria.map fun C => pair R d C limit := by
    simp only [List.map_flatMap, List.map_map]
    rfl
  rw [explodePairs, ← hl']
  convert h using 1
  ext v
  simp only [Set.mem_ofPred_eq, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨d, hd, C, hC', hv⟩
    exact ⟨(d, C), ⟨d, hd, C, hC', rfl⟩, hv⟩
  · rintro ⟨_, ⟨d, hd, C, hC', rfl⟩, hv⟩
    exact ⟨d, hd, C, hC', hv⟩

end Bounds

end Xdy.Spec
