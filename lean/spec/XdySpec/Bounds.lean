import XdySpec.Stacks

/-!
# The bounds of rerolls and explodes

The bounds evaluator in `xdy/src/evaluator.rs` bounds the totals of the chains
of a record by their shapes (`ChainShape`): runs of middle dice, each die of
which may show any of a set of results, and each run any number of dice within
a range, and the results that the last die may show. It tracks the shapes of
the chains of each die that the record may roll (`DieChains`) through every
reroll and explode, never enumerating the chains, and folds each shape at its
least or greatest results (`ChainShape::totals`); every reader of the totals
reads only their hull. This module mirrors the shapes and their transforms,
and proves the hull sound for every chain that a sequence of rerolls,
explodes, drops, and keeps leaves, and exact within the budget, and the counts
of dice that the bounds take of the shapes sound:

- `fold` mirrors `fold`, `min(i32Max, max(i32Min, Σneg) + Σpos)` over the
  negative and other dice, each with its copies, and is the total of every
  chain of those dice, the ascending saturating sum (`fold_eq_total`). It
  never decreases as any die grows (`fold_mono_results`), nor moves against
  the sign of a die as its copies grow (`fold_mono_copies`), `unboundedDice`
  copies of a nonzero die saturate it (`fold_unbounded`), and so copies capped
  at `unboundedDice` fold alike (`fold_eq_of_cap`).
- The tails that the specification's explode may add to a chain whose last
  die it matches are exactly the limited tails (`mem_support_extension_iff`)
  or, without limit, the unlimited tails
  (`mem_support_unlimitedExtension_iff`), so an explode may leave a chain
  exactly when it may make it of some chain still kept (`Becomes`,
  `mem_explode_iff`). A reroll redraws each die of a chain alike (`Rerolled`),
  and may leave a chain exactly when it may make it of some chain still kept
  (`RerollBecomes`, `mem_reroll_iff`).
- `Segment`, `ChainShape`, and their `new`, `totals`, `map`, `unite`,
  `coarsen`, and `uniteShapes` mirror the Rust. A chain has a shape
  (`ChainShape.Has`) if its dice split among the runs, and its total lies
  between the shape's totals (`ChainShape.totals_le`); `new`, `unite`,
  `coarsen`, and `uniteShapes` lose no chain (`ChainShape.has_new`,
  `ChainShape.unite_has`, `ChainShape.coarsen_has`, `uniteShapes_has`), and a
  reroll's `map` loses none that it redraws (`ChainShape.map_has`).
- `DieChains`, with its `reroll` and `explode`, mirrors the Rust, over
  criteria modeled as the results that every criterion and some criterion
  match, and a function `covers` that stands for the least dice of the die and
  the shapes of the criteria. The chains after a clause cover every chain that
  it may make of one that they covered (`DieChains.reroll_covers`,
  `DieChains.explode_covers`), so the chains after the bounds of a sequence of
  clauses cover every chain still kept of every record that the clauses may
  lead to, whose total lies among their totals (`DieChains.run_sound`).
- Every corner of a shape (`ChainShape.Corner`), one result of each run at its
  fewest or most dice, or at any number of an unbounded run beyond a
  threshold, is a chain that the clauses may make (`ChainShape.Realized`),
  through every transform but `coarsen`, and the least and greatest totals of
  a shape are those of corners (`ChainShape.totals_realized`). So for the
  chains of one die (`DieChains.ofDie`), whose clauses enumerate exact
  criteria and never coarsen the shapes, the least and greatest totals are
  those of chains that the clauses may make of a chain of one face
  (`DieChains.run_exact`, `Exact.hull`).
- `ChainShape.dice` and `ChainShape.matching` mirror the Rust, and bound the
  kept dice of every chain of a shape, and those that show one of some results
  (`ChainShape.length_le_dice`, `ChainShape.countP_le_matching`), so
  `lengthOf`, which mirrors `RollingRecordBounds::length`, bounds the kept dice
  of every chain still kept that no saturated die decides after any sequence of
  clauses (`DieChains.run_length`). The specification's reroll and explode
  forget their draws, so `countedReplacement`, `countedUnlimited`,
  `countedRerollChain`, `countedExtension`, and `countedUnlimitedExtension`
  count them beside the specification's laws, which they become once the
  draws are forgotten (`countedRerollChain_fst` and the like), and
  `rerollDraws` and `explodeDraws` answer the law of the dice that a clause
  draws for one chain. `DieChains.rerollRolls` and `DieChains.extensions`
  mirror the dice that `DieChains::reroll` and `DieChains::explode` answer,
  and bound them for every chain that the chains cover
  (`DieChains.rerollDraws_le`, `DieChains.explodeDraws_le`). If no criterion
  may match a kept die (`DieChains.mayMatch`), then a reroll leaves every such
  chain as it was (`DieChains.rerollBecomes_of_not_mayMatch`).

```mermaid
flowchart TD
    F["fold_eq_total,<br/>fold_mono_results, fold_mono_copies,<br/>fold_unbounded"]
    B["mem_explode_iff, mem_reroll_iff:<br/>the chains a clause leaves"]
    H["ChainShape.totals_le:<br/>a shape bounds its chains"]
    U["has_new, unite_has, coarsen_has,<br/>uniteShapes_has, map_has"]
    C["reroll_covers, explode_covers:<br/>the chains after a clause"]
    S["DieChains.run_sound:<br/>the totals hold every total"]
    R["corner_new, unite_realized,<br/>rerollShapes_realized,<br/>explodeShapes_realized"]
    T["ChainShape.totals_realized:<br/>extremes are corners"]
    E["DieChains.run_exact:<br/>the hull is exact"]
    N["length_le_dice,<br/>countP_le_matching:<br/>a shape bounds its dice"]
    D["counted laws, *_fst:<br/>the draws of a clause"]
    L["DieChains.run_length:<br/>the length holds every chain"]
    K["rerollDraws_le, explodeDraws_le,<br/>rerollBecomes_of_not_mayMatch"]
    F --> H
    H --> S
    U --> C
    B --> C
    C --> S
    F --> T
    R --> E
    T --> E
    S --> E
    N --> L
    C --> L
    N --> K
    D --> K
    B --> K
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
shapes of chain that `DieChains::explode` in `evaluator.rs` makes of a chain
of one die.

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
chains, each of which, unless it has saturated, holds only faces of its die,
which lie within the `i32` range: so the totals of the chains that it leaves
are the totals of what it may make of each.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `ch'`: A chain.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hk`: Every kept chain is normal, and holds only faces of the die unless it
  has saturated.
-/
theorem mem_explode_iff {r : Record} {c : Criterion} {l : Option Nat}
    (hF : ∀ x ∈ r.die.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hk : ∀ ch ∈ r.keptChains, NormalChain ch ∧
      (ch.saturates = false → ∀ x ∈ ch.dice, x ∈ r.die.faces))
    (ch' : Chain) :
    (∃ r' ∈ (explode r c l).support, ch' ∈ r'.keptChains) ↔
      ∃ ch ∈ r.keptChains, Becomes r.die c l ch ch' := by
  -- Every kept chain that grows holds only faces, so the die has some.
  have hkg : ∀ ch ∈ r.keptChains, ch.grows c = true →
      ∀ x ∈ ch.dice, x ∈ r.die.faces := fun ch hch hg =>
    (hk ch hch).2 (by simp only [Chain.grows, Bool.and_eq_true,
      Bool.not_eq_true'] at hg; exact hg.1)
  have hne : ∀ ch ∈ r.keptChains, ch.grows c = true → r.die.faces ≠ [] :=
    fun ch hch hg hn => by
      obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ (dice_ne_nil ch)
      simpa [hn] using hkg ch hch hg x hx
  cases l with
  | some limit =>
    have hg : ∀ ch ∈ r.keptChains, (0 < limit && ch.grows c) = true →
        ∀ t ∈ (extension c (customDie r.die.faces) limit).support,
          NormalChain (growChain ch t) := fun ch hch hgr t ht => by
      simp only [Bool.and_eq_true] at hgr
      exact normalChain_growChain hF (hkg ch hch hgr.2)
        (mem_support_extension (hne ch hch hgr.2) ht)
    simp only [explode]
    rw [mem_explodeChains (fun ch h => (hk ch h).1) hg]
    refine exists_congr fun ch => and_congr_right fun hch => ?_
    simp only [Becomes, Bool.and_eq_true, decide_eq_true_eq]
    split
    · rename_i hgr
      refine exists_congr fun t => ?_
      rw [mem_support_extension_iff (hne ch hch hgr.2) hgr.1]
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
            NormalChain (growChain ch t) := fun ch hch hgr t ht => by
        obtain ⟨-, hdm, hy, -⟩ :=
          (mem_support_unlimitedExtension_iff hw).mp ht
        exact normalChain_growChain hF (hkg ch hch hgr)
          ⟨fun x hx => (hdm x hx).1, hy⟩
      simp only [explode, hw, Bool.false_eq_true, ↓reduceDIte]
      rw [mem_explodeChains (fun ch h => (hk ch h).1) hg]
      refine exists_congr fun ch => and_congr_right fun hch => ?_
      simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte]
      split
      · refine exists_congr fun t => ?_
        rw [mem_support_unlimitedExtension_iff hw]
      · rfl

/-! ## What a reroll makes of a chain -/

/--
Whether a reroll may leave one die in place of another: a die that it
rerolls, any replacement of a property, and any other, itself.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The property of a replacement.
- `x`: The die.
- `y`: What the reroll leaves in its place.
-/
def Redraw (rerolled : Int → Bool) (P : Int → Prop) (x y : Int) : Prop :=
  if rerolled x then P y else y = x

/-- Whether a reroll may leave one die in place of another is decided, if
the property of a replacement is. -/
instance (rerolled : Int → Bool) (P : Int → Prop) [DecidablePred P]
    (x y : Int) : Decidable (Redraw rerolled P x y) := by
  unfold Redraw
  infer_instance

/--
Whether a reroll may leave one chain of another: each die of a chain that
may still grow redrawn, the dice besides the last sorted, and a saturated
chain alone. Mirrors `rerollChain`, and the map of the results of every run
and of the last die by which `DieChains::reroll` in `evaluator.rs` transforms
the shapes of a chain.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The property of a replacement.
- `ch`: The chain.
- `ch'`: What the reroll leaves of it.
-/
def Rerolled (rerolled : Int → Bool) (P : Int → Prop) : Chain → Chain → Prop
  | .live body last, ch' => ∃ ys y, List.Forall₂ (Redraw rerolled P) body ys ∧
    Redraw rerolled P last y ∧ ch' = .live ys.mergeSort y
  | .saturated v, ch' => ch' = .saturated v

/--
Whether a reroll may make one chain of another still kept: a reroll limited
to `limit` times redraws each die that it matches, if `limit` is positive, as
any face of the die; an unlimited reroll that matches every face of the die
crosses out every chain that may still grow, and leaves every saturated chain
alone; and any other redraws each die that it matches as a face that it does
not match.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The limit, or `none` for no limit.
- `ch`: The chain still kept.
- `ch'`: What the reroll makes of it.
-/
def RerollBecomes (d : Die) (c : Criterion) : Option Nat → Chain → Chain → Prop
  | some limit, ch, ch' =>
    Rerolled (fun x => 0 < limit && c.matches x) (· ∈ d.faces) ch ch'
  | none, ch, ch' =>
    if d.facesWithin c = true then ch.saturates = true ∧ ch' = ch
    else Rerolled c.matches (fun y => y ∈ d.faces ∧ c.matches y = false) ch ch'

/-- The multisets that draws of independent values may leave are exactly the
draws, one for each value, that each law may answer. -/
private theorem mem_support_bagDraws_iff {α β : Type} {step : α → PMF β} :
    ∀ {xs : List α} {M : Multiset β}, M ∈ (bagDraws step xs).support ↔
      ∃ ys, List.Forall₂ (fun x y => y ∈ (step x).support) xs ys ∧ M = ys
  | [], M => by
    simp only [bagDraws, PMF.mem_support_pure_iff, List.forall₂_nil_left_iff,
      exists_eq_left, Multiset.coe_nil]
  | x :: xs, M => by
    simp only [bagDraws, lift, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff, List.forall₂_cons_left_iff]
    constructor
    · rintro ⟨y, hy, N, hN, rfl⟩
      obtain ⟨ys, hys, rfl⟩ := mem_support_bagDraws_iff.mp hN
      exact ⟨y :: ys, ⟨y, ys, hy, hys, rfl⟩, rfl⟩
    · rintro ⟨_, ⟨y, ys, hy, hys, rfl⟩, rfl⟩
      exact ⟨y, hy, ys, mem_support_bagDraws_iff.mpr ⟨ys, hys, rfl⟩, rfl⟩

/-- A die may be redrawn as exactly what its law may answer. -/
private theorem mem_support_draw_iff {rerolled : Int → Bool} {P : PMF Int}
    {x y : Int} :
    y ∈ (if rerolled x then P else PMF.pure x).support ↔
      Redraw rerolled (· ∈ P.support) x y := by
  unfold Redraw
  split <;> simp

/--
The chains that a reroll may leave of a chain are exactly those that
`Rerolled` describes, for replacements that its law may answer.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The law of the replacement of a rerolled result.
- `ch`: The chain.
- `ch'`: A chain.
-/
theorem mem_support_rerollChain_iff {rerolled : Int → Bool} {P : PMF Int}
    {ch ch' : Chain} :
    ch' ∈ (rerollChain rerolled P ch).support ↔
      Rerolled rerolled (· ∈ P.support) ch ch' := by
  cases ch with
  | saturated v => simp [rerollChain, Rerolled]
  | live body last =>
    rw [rerollChain_live]
    simp only [PMF.mem_support_bind_iff, PMF.mem_support_map_iff, Rerolled]
    constructor
    · rintro ⟨M, hM, y, hy, rfl⟩
      obtain ⟨ys, hys, rfl⟩ := mem_support_bagDraws_iff.mp hM
      refine ⟨ys, y, ?_, mem_support_draw_iff.mp hy, by
        rw [Setup.mergeSort_eq_sortBag]⟩
      exact hys.imp fun _ _ h => mem_support_draw_iff.mp h
    · rintro ⟨ys, y, hys, hy, rfl⟩
      refine ⟨ys, mem_support_bagDraws_iff.mpr ⟨ys, hys.imp fun _ _ h =>
        mem_support_draw_iff.mpr h, rfl⟩, y, mem_support_draw_iff.mpr hy, ?_⟩
      rw [Setup.mergeSort_eq_sortBag]

/-- A relation between lists holds of another relation that follows from it
for every element on the left. -/
private theorem forall₂_imp_mem {R S : Int → Int → Prop} :
    ∀ {l l' : List Int}, (∀ a ∈ l, ∀ b, R a b → S a b) →
      List.Forall₂ R l l' → List.Forall₂ S l l'
  | _, _, _, .nil => .nil
  | a :: _, _, h, .cons hr hl =>
    .cons (h a List.mem_cons_self _ hr)
      (forall₂_imp_mem (fun x hx => h x (List.mem_cons_of_mem _ hx)) hl)

/-- Redrawing by replacements of one property redraws by those of another that
agrees on every replacement that matters. -/
private theorem Rerolled.congr {rerolled : Int → Bool} {P Q : Int → Prop}
    {ch ch' : Chain}
    (h : ∀ body last, ch = .live body last →
      ∀ x ∈ last :: body, rerolled x = true → ∀ y, P y ↔ Q y) :
    Rerolled rerolled P ch ch' ↔ Rerolled rerolled Q ch ch' := by
  cases ch with
  | saturated v => rfl
  | live body last =>
    have hpq := h body last rfl
    have hd : ∀ x ∈ last :: body, ∀ y,
        Redraw rerolled P x y ↔ Redraw rerolled Q x y := fun x hx y => by
      unfold Redraw
      split
      · exact hpq x hx ‹_› y
      · rfl
    simp only [Rerolled]
    refine exists_congr fun ys => exists_congr fun y => ?_
    rw [hd last List.mem_cons_self y]
    refine and_congr_left fun _ => ⟨fun h' => ?_, fun h' => ?_⟩
    · exact forall₂_imp_mem (fun x hx b h => (hd x
        (List.mem_cons_of_mem _ hx) b).mp h) h'
    · exact forall₂_imp_mem (fun x hx b h => (hd x
        (List.mem_cons_of_mem _ hx) b).mpr h) h'

/-- Every replacement of a limited reroll is a value of its die. -/
private theorem mem_support_limitedReplacement {c : Criterion}
    {die : PMF Int} (n : Nat) {y : Int}
    (hy : y ∈ (limitedReplacement c die n).support) : y ∈ die.support := by
  have key : ∀ k : Nat, ∀ y ∈ (k.repeat (redraw c die) die).support,
      y ∈ die.support := by
    intro k
    induction k with
    | zero => exact fun _ h => h
    | succ k ih =>
      intro y hy
      simp only [Nat.repeat, redraw, PMF.mem_support_bind_iff] at hy
      obtain ⟨x, hx, hy⟩ := hy
      split at hy
      · exact hy
      · rw [PMF.mem_support_pure_iff] at hy
        exact hy ▸ ih x hx
  exact key _ y hy

/-- Every face of a die may replace a die that a limited reroll rerolls. -/
private theorem mem_support_limitedReplacement_of_mem {c : Criterion}
    {F : List Int} (n : Nat) {y : Int} (hy : y ∈ F) :
    y ∈ (limitedReplacement c (customDie F) n).support := by
  have key : ∀ k : Nat,
      y ∈ (k.repeat (redraw c (customDie F)) (customDie F)).support := by
    intro k
    induction k with
    | zero => exact mem_support_customDie hy
    | succ k ih =>
      simp only [Nat.repeat, redraw, PMF.mem_support_bind_iff]
      refine ⟨y, ih, ?_⟩
      split
      · exact mem_support_customDie hy
      · exact PMF.mem_support_pure_iff _ _ |>.mpr rfl
  exact key _

/-- A die of faces, if any, shows only its faces, or else only `0`. -/
private theorem range_of_customDie {F : List Int}
    (hF : ∀ x ∈ F, i32Min ≤ x ∧ x ≤ i32Max) {y : Int}
    (hy : y ∈ (customDie F).support) : i32Min ≤ y ∧ y ≤ i32Max := by
  rcases mem_of_mem_support_customDie hy with hy | ⟨-, rfl⟩
  · exact hF y hy
  · simp [i32Min, i32Max]

/-- A reroll of each kept die of a record of normal chains, by replacements
within the `i32` range, may leave a chain among those kept exactly when it may
leave it of some chain kept. -/
private theorem mem_rerollEach_iff {r : Record} {rerolled : Int → Bool}
    {P : PMF Int} (hn : ∀ ch ∈ r.keptChains, NormalChain ch)
    (hP : ∀ y ∈ P.support, i32Min ≤ y ∧ y ≤ i32Max) (ch' : Chain) :
    (∃ r' ∈ (rerollEach r rerolled P).support, ch' ∈ r'.keptChains) ↔
      ∃ ch ∈ r.keptChains, ch' ∈ (rerollChain rerolled P ch).support := by
  -- Every chain that the reroll leaves of a kept chain is normal.
  have hK : ∀ ch ∈ r.keptChains, ∀ x ∈ (rerollChain rerolled P ch).support,
      NormalChain x := fun ch hch x hx => by
    cases ch with
    | saturated v =>
      simp only [rerollChain, PMF.mem_support_pure_iff] at hx
      exact hx ▸ hn _ hch
    | live body last =>
      have hnc := hn _ hch
      exact (mem_support_rerollChain (Q := fun y => i32Min ≤ y ∧ y ≤ i32Max)
        (fun _ _ _ y hy => hP y hy) (fun y hy => hnc.2 y
          ((List.mergeSort_perm _ _).mem_iff.mp (by simpa only [Chain.dice]
            using hy))) hx).2.2 (fun _ h => h) hnc
  -- Which chains some multiset of draws holds.
  have hbag : (∃ M ∈ (bagDraws (rerollChain rerolled P) r.keptChains).support,
      ch' ∈ M) ↔ ∃ ch ∈ r.keptChains,
        ch' ∈ (rerollChain rerolled P ch).support := by
    constructor
    · rintro ⟨M, hM, hch⟩
      exact mem_support_bagDraws hM ch' hch
    · rintro ⟨ch, hch, hy⟩
      exact exists_mem_bagDraws hch hy
  by_cases hc : r.chains.isEmpty
  · -- A record without chains keeps chains of one die, which may grow.
    have hl : ∀ ch ∈ r.keptChains, ch.saturates = false := fun ch hch => by
      simp only [Record.keptChains, hc, ↓reduceIte, List.mem_map] at hch
      obtain ⟨x, -, rfl⟩ := hch
      rfl
    have hv := map_chainView_rerollEach (rerolled := rerolled) hn hl
      fun _ _ _ _ _ y hy => hP y hy
    rw [← hbag]
    constructor
    · rintro ⟨r', hr', hch⟩
      have : (r'.chainView) ∈ ((rerollEach r rerolled P).map
          Record.chainView).support := (PMF.mem_support_map_iff _ _ _).mpr
        ⟨r', hr', rfl⟩
      rw [hv, PMF.mem_support_map_iff] at this
      obtain ⟨M, hM, he⟩ := this
      refine ⟨M, hM, ?_⟩
      have := congrArg Prod.fst he
      simp only [Record.chainView] at this
      rw [this]
      exact Multiset.mem_coe.mpr hch
    · rintro ⟨M, hM, hch⟩
      have : (M, r.die) ∈ ((rerollEach r rerolled P).map
          Record.chainView).support := by
        rw [hv, PMF.mem_support_map_iff]
        exact ⟨M, hM, rfl⟩
      obtain ⟨r', hr', he⟩ := (PMF.mem_support_map_iff _ _ _).mp this
      refine ⟨r', hr', ?_⟩
      have := congrArg Prod.fst he
      simp only [Record.chainView] at this
      rw [← this] at hch
      exact Multiset.mem_coe.mp hch
  · simp only [rerollEach, hc, Bool.false_eq_true, ↓reduceIte]
    rw [rerollChains_eq_bag _ _ _ hK, ← hbag]
    simp only [PMF.support_map, Set.mem_image]
    constructor
    · rintro ⟨_, ⟨M, hM, rfl⟩, hch⟩
      have hnM : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
        obtain ⟨ch, hch, hx⟩ :=
          mem_support_bagDraws hM x (Multiset.mem_toList.mp hx)
        exact hK ch hch x hx
      exact ⟨M, hM, Multiset.mem_toList.mp
        ((mem_keptChains_ofChains hnM).mp hch)⟩
    · rintro ⟨M, hM, hch⟩
      have hnM : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
        obtain ⟨ch, hch, hx⟩ :=
          mem_support_bagDraws hM x (Multiset.mem_toList.mp hx)
        exact hK ch hch x hx
      exact ⟨_, ⟨M, hM, rfl⟩, (mem_keptChains_ofChains hnM).mpr
        (Multiset.mem_toList.mpr hch)⟩

/--
A reroll may leave a chain among the kept chains of some record exactly when
it may make that chain of some chain still kept, for a record of normal
chains, each of which, unless it has saturated, holds only faces of its die,
which lie within the `i32` range: so the totals of the chains that it leaves
are the totals of what it may make of each.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `ch'`: A chain.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hk`: Every kept chain is normal, and holds only faces of the die unless it
  has saturated.
-/
theorem mem_reroll_iff {r : Record} {c : Criterion} {l : Option Nat}
    (hF : ∀ x ∈ r.die.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hk : ∀ ch ∈ r.keptChains, NormalChain ch ∧
      (ch.saturates = false → ∀ x ∈ ch.dice, x ∈ r.die.faces))
    (ch' : Chain) :
    (∃ r' ∈ (reroll r c l).support, ch' ∈ r'.keptChains) ↔
      ∃ ch ∈ r.keptChains, RerollBecomes r.die c l ch ch' := by
  have hn : ∀ ch ∈ r.keptChains, NormalChain ch := fun ch h => (hk ch h).1
  -- Every die of a kept chain that may still grow is a face.
  have hlive : ∀ ch ∈ r.keptChains, ∀ body last, ch = .live body last →
      ∀ x ∈ last :: body, x ∈ r.die.faces := fun ch hch body last he x hx =>
    (hk ch hch).2 (by simp [he, Chain.saturates]) x (by
      rw [he, Chain.dice]; exact (List.mergeSort_perm _ _).mem_iff.mpr hx)
  cases l with
  | some limit =>
    simp only [reroll]
    rw [mem_rerollEach_iff hn (fun y hy => range_of_customDie hF
      (mem_support_limitedReplacement _ hy))]
    refine exists_congr fun ch => and_congr_right fun hch => ?_
    rw [mem_support_rerollChain_iff, RerollBecomes]
    refine Rerolled.congr fun body last he x hx _ y => ?_
    have hxF := hlive ch hch body last he x hx
    exact ⟨fun h => by
      have := mem_support_limitedReplacement _ h
      rcases mem_of_mem_support_customDie this with h | ⟨h, -⟩
      · exact h
      · simp [h] at hxF,
      fun h => mem_support_limitedReplacement_of_mem _ h⟩
  | none =>
    by_cases hw : r.die.facesWithin c = true
    · simp only [reroll, hw, ↓reduceDIte, RerollBecomes, ↓reduceIte]
      by_cases hc : r.chains.isEmpty
      · -- Every kept result matches, so none is kept.
        simp only [hc, ↓reduceIte, PMF.mem_support_pure_iff, exists_eq_left]
        have hkc : r.keptChains = r.kept.map .single := by
          simp [Record.keptChains, hc]
        have hf : r.kept.filter (!c.matches ·) = [] := by
          refine List.filter_eq_nil_iff.mpr fun x hx => ?_
          have hxF := hlive (.single x) (hkc ▸ List.mem_map_of_mem hx) [] x
            rfl x List.mem_cons_self
          simpa using List.all_eq_true.mp hw x hxF
        constructor
        · intro h
          rw [hf] at h
          simp [Record.keptChains, Record.kept, Record.sorted] at h
        · rintro ⟨ch, hch, hs, -⟩
          rw [hkc, List.mem_map] at hch
          obtain ⟨x, -, rfl⟩ := hch
          simp [Chain.single, Chain.saturates] at hs
      · simp only [hc, Bool.false_eq_true, ↓reduceIte,
          PMF.mem_support_pure_iff, exists_eq_left]
        rw [mem_keptChains_ofChains fun x hx =>
          hn x (List.mem_of_mem_filter hx), List.mem_filter]
        constructor
        · rintro ⟨hch, hs⟩
          exact ⟨ch', hch, hs, rfl⟩
        · rintro ⟨ch, hch, hs, rfl⟩
          exact ⟨hch, hs⟩
    · simp only [reroll, hw, Bool.false_eq_true, ↓reduceDIte]
      rw [mem_rerollEach_iff hn (fun y hy => by
        simp only [PMF.mem_support_ofMultiset_iff, Multiset.mem_toFinset,
          Multiset.mem_coe] at hy
        exact hF y (List.mem_of_mem_filter hy))]
      refine exists_congr fun ch => and_congr_right fun hch => ?_
      rw [mem_support_rerollChain_iff]
      simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
      refine Rerolled.congr fun _ _ _ _ _ _ y => ?_
      simp

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
as the bounds in `evaluator.rs` unite their answers for each pair of a die
and a criterion.

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

/-! ## Chain shapes -/

/--
The least of a set of results, or `0` if it is empty, as `ResultSet::hull` in
`evaluator.rs` answers the least of a set that a shape never leaves empty.

# Parameters
- `s`: The results.

# Returns
The least result, or `0`.

# Examples
```lean
#guard Bounds.low {3, -1, 4} == -1
#guard Bounds.low ∅ == 0
```
-/
def low (s : Finset Int) : Int := ((hull s).map Prod.fst).getD 0

/--
The greatest of a set of results, or `0` if it is empty, as `ResultSet::hull`
in `evaluator.rs` answers the greatest of a set that a shape never leaves
empty.

# Parameters
- `s`: The results.

# Returns
The greatest result, or `0`.

# Examples
```lean
#guard Bounds.high {3, -1, 4} == 4
#guard Bounds.high ∅ == 0
```
-/
def high (s : Finset Int) : Int := ((hull s).map Prod.snd).getD 0

/--
The least of a set of results is no more than any of them.

# Parameters
- `s`: The results.
- `x`: A result.

# Hypotheses
- `hx`: `x` lies in `s`.
-/
theorem low_le {s : Finset Int} {x : Int} (hx : x ∈ s) : low s ≤ x := by
  obtain ⟨p, hp, h₁, -⟩ := exists_hull hx
  simpa [low, hp] using h₁

/--
The greatest of a set of results is no less than any of them.

# Parameters
- `s`: The results.
- `x`: A result.

# Hypotheses
- `hx`: `x` lies in `s`.
-/
theorem le_high {s : Finset Int} {x : Int} (hx : x ∈ s) : x ≤ high s := by
  obtain ⟨p, hp, -, h₂⟩ := exists_hull hx
  simpa [high, hp] using h₂

/--
The maximal runs of consecutive values among some values in ascending order,
each as its least and greatest.

# Parameters
- `xs`: The values, ascending, without repetition.

# Returns
The runs, ascending.

# Examples
```lean
#guard Bounds.intervalsOf [1, 2, 3, 5, 7, 8] == [(1, 3), (5, 5), (7, 8)]
#guard Bounds.intervalsOf [] == []
```
-/
def intervalsOf : List Int → List (Int × Int)
  | [] => []
  | x :: xs =>
    match intervalsOf xs with
    | (lo, hi) :: rest =>
      if x + 1 = lo then (x, hi) :: rest else (x, x) :: (lo, hi) :: rest
    | [] => [(x, x)]

/--
The key by which `ResultSet` in `evaluator.rs` orders sets of results: its
disjoint, ascending intervals, compared lexicographically, each by its least
and then its greatest. Comparing the intervals flattened into one list orders
them alike. Only which shapes unite depends on this order, and no proof
here rests on it.

# Parameters
- `s`: The results.

# Returns
The key.

# Examples
```lean
-- `{1, 3}` precedes `{1, 2, 3}`, since `(1, 1)` precedes `(1, 3)`.
#guard Bounds.orderKey {1, 3} == [1, 1, 3, 3]
#guard decide (Bounds.orderKey {1, 3} < Bounds.orderKey {1, 2, 3})
```
-/
def orderKey (s : Finset Int) : List Int :=
  (intervalsOf (s.sort (· ≤ ·))).flatMap fun p => [p.1, p.2]

/-- A run of the middle dice of a chain, each of which may show any of the
same results, independently of the others. Mirrors `Segment` in
`evaluator.rs`, whose greatest number of dice saturates at `u64::MAX`, which
stands for any number; here, `⊤` stands for any number, and a finite number
never saturates, which no program can tell apart, since a run would need more
than `2⁶³` dice. -/
structure Segment where
  /-- The results that each die of the run may show. -/
  results : Finset Int
  /-- The least number of dice of the run. -/
  least : Nat
  /-- The greatest number of dice of the run, or `⊤` for any number. -/
  greatest : ℕ∞
  deriving DecidableEq

/-- A shape of a chain still kept that no saturated die decides: its middle
dice, those that some die extends, in runs, and the results that its last die,
the kept die that no die extends, may show. Mirrors `ChainShape` in
`evaluator.rs`, whose `end` is `last` here. -/
structure ChainShape where
  /-- The runs of middle dice. -/
  middles : List Segment
  /-- The results that the last die may show. -/
  last : Finset Int
  deriving DecidableEq

/--
Merge the adjacent runs that show the same results, whose numbers of dice add,
as `ChainShape::new` in `evaluator.rs` merges them once sorted.

# Parameters
- `runs`: The runs.

# Returns
The merged runs.
-/
def mergeRuns : List Segment → List Segment
  | [] => []
  | run :: runs =>
    match mergeRuns runs with
    | next :: rest =>
      if run.results = next.results then
        { run with
          least := run.least + next.least
          greatest := run.greatest + next.greatest } :: rest
      else run :: next :: rest
    | [] => [run]

/--
The shape of some runs of middle dice and a last die: the runs that may hold
some die, sorted by their results, as `ResultSet` orders them, with the runs
of the same results merged. Mirrors `ChainShape::new` in `evaluator.rs`.

# Parameters
- `middles`: The runs, in any order.
- `last`: The results that the last die may show.

# Returns
The shape.
-/
def ChainShape.new (middles : List Segment) (last : Finset Int) : ChainShape :=
  { middles := mergeRuns ((middles.filter fun run => decide (0 < run.greatest)
      ).mergeSort fun a b => decide (orderKey a.results ≤ orderKey b.results))
    last }

/--
The number of copies of a die by which the totals of a shape fold a run of
some number of dice: that number, or `unboundedDice`, which stands for any
greater number, as `ChainShape::totals` in `evaluator.rs` caps it.

# Parameters
- `count`: The number of dice, or `⊤` for any number.

# Returns
The copies.

# Examples
```lean
#guard Bounds.copies 3 == 3
#guard Bounds.copies ⊤ == Bounds.unboundedDice
```
-/
def copies (count : ℕ∞) : Nat :=
  if count = ⊤ then unboundedDice else Min.min count.toNat unboundedDice

/--
The die and copies by which the least total of a shape folds a run: its least
result, at its most dice if that is negative, and at its fewest otherwise.

# Parameters
- `run`: The run.

# Returns
The die and its copies.
-/
def Segment.lowTerm (run : Segment) : Int × Nat :=
  (low run.results,
    copies (if low run.results < 0 then run.greatest else run.least))

/--
The die and copies by which the greatest total of a shape folds a run: its
greatest result, at its most dice if that is positive, and at its fewest
otherwise.

# Parameters
- `run`: The run.

# Returns
The die and its copies.
-/
def Segment.highTerm (run : Segment) : Int × Nat :=
  (high run.results,
    copies (if 0 < high run.results then run.greatest else run.least))

/--
The least and greatest totals of the chains of a shape: the folds of its last
die and of each run at their least or greatest results, each run at the number
of its dice that moves the total furthest that way. Mirrors
`ChainShape::totals` in `evaluator.rs`. They bound the total of every chain of
the shape (`ChainShape.totals_le`).

# Parameters
- `S`: The shape.

# Returns
The least and greatest totals.

# Examples
```lean
-- A six, then up to two more sixes, and a last die of one to five.
#guard (ChainShape.totals
    { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1, 2, 3, 4, 5} }) == (7, 23)
-- Any number of middle dice of `-1` saturates the least total.
#guard (ChainShape.totals
    { middles := [{ results := {-1, 2}, least := 0, greatest := ⊤ }],
      last := {1} }) == (i32Min + 1, i32Max)
```
-/
def ChainShape.totals (S : ChainShape) : Int × Int :=
  (fold ((low S.last, 1) :: S.middles.map Segment.lowTerm),
    fold ((high S.last, 1) :: S.middles.map Segment.highTerm))

/--
Whether a run may hold some dice: as many as it may, each showing one of its
results.

# Parameters
- `run`: The run.
- `xs`: The dice.
-/
def Segment.Holds (run : Segment) (xs : List Int) : Prop :=
  run.least ≤ xs.length ∧ (xs.length : ℕ∞) ≤ run.greatest ∧
    ∀ x ∈ xs, x ∈ run.results

/--
Whether a chain has a shape: it may still grow, its last die shows one of the
results of the shape's last die, and its other dice split among the runs, each
of which holds its share.

# Parameters
- `S`: The shape.
- `ch`: The chain.
-/
def ChainShape.Has (S : ChainShape) : Chain → Prop
  | .live body last => last ∈ S.last ∧ ∃ parts : List (List Int),
    List.Forall₂ Segment.Holds S.middles parts ∧ body.Perm parts.flatten
  | .saturated _ => False

/-- Whether a run may hold some dice is decided. -/
instance (run : Segment) (xs : List Int) : Decidable (run.Holds xs) := by
  unfold Segment.Holds
  infer_instance

/-- The copies of no more than `unboundedDice` dice are those dice. -/
theorem copies_natCast (n : Nat) :
    copies n = Min.min n unboundedDice := by
  simp [copies]

/-- The copies never exceed `unboundedDice`. -/
theorem copies_le (count : ℕ∞) : copies count ≤ unboundedDice := by
  unfold copies
  split
  · exact le_rfl
  · exact Nat.min_le_right _ _

/-- A run of at least some number of dice folds at least that many copies,
or `unboundedDice`. -/
private theorem min_le_copies {n : Nat} {g : ℕ∞} (h : (n : ℕ∞) ≤ g) :
    Min.min n unboundedDice ≤ copies g := by
  unfold copies
  split
  · exact Nat.min_le_right _ _
  · rename_i hg
    have := ENat.toNat_le_toNat h hg
    rw [ENat.toNat_natCast] at this
    exact min_le_min_right _ this

/-- The fold of some dice is no more than that of others whose negative dice
sum no lower, unless theirs saturate, and whose others sum no lower. -/
private theorem fold_le_of_sums {l l' : List (Int × Nat)}
    (hn : negative l ≤ negative l' ∨ negative l ≤ -(unboundedDice : Int))
    (hp : positive l ≤ positive l') : fold l ≤ fold l' := by
  have := negative_nonpos l'
  have := positive_nonneg l
  simp only [fold, unboundedDice, i32Min, i32Max] at *
  push_cast at hn
  omega

/-- The fold of some dice is no less than that of others whose negative dice
sum no higher, and whose others sum no higher, unless theirs saturate. -/
private theorem le_fold_of_sums {l l' : List (Int × Nat)}
    (hn : negative l' ≤ negative l)
    (hp : positive l' ≤ positive l ∨ (unboundedDice : Int) ≤ positive l) :
    fold l' ≤ fold l := by
  have := negative_nonpos l
  have := positive_nonneg l'
  simp only [fold, unboundedDice, i32Min, i32Max] at *
  push_cast at hp
  omega

/-- The least term of a run bounds the sums of the dice that it holds. -/
private theorem Segment.holds_low {run : Segment} {xs : List Int}
    (h : run.Holds xs) :
    (Min.min run.lowTerm.1 0 * run.lowTerm.2 ≤ negative (xs.map (·, 1)) ∨
      Min.min run.lowTerm.1 0 * run.lowTerm.2 ≤ -(unboundedDice : Int)) ∧
      Max.max run.lowTerm.1 0 * run.lowTerm.2 ≤ positive (xs.map (·, 1)) := by
  obtain ⟨hl, hg, hm⟩ := h
  obtain ⟨a, -, c, -⟩ := singles_bounds (lo := low run.results)
    (hi := high run.results) fun z hz => ⟨low_le (hm z hz), le_high (hm z hz)⟩
  have hpos := positive_nonneg (xs.map (·, 1))
  simp only [Segment.lowTerm]
  split_ifs with hlo
  · rw [min_eq_left hlo.le] at a ⊢
    rw [max_eq_right hlo.le, zero_mul]
    refine ⟨?_, hpos⟩
    have hc := min_le_copies hg
    by_cases hn : xs.length ≤ unboundedDice
    · rw [Nat.min_eq_left hn] at hc
      have : (xs.length : Int) ≤ copies run.greatest := by exact_mod_cast hc
      exact .inl ((mul_le_mul_of_nonpos_left this hlo.le).trans a)
    · rw [Nat.min_eq_right (by omega)] at hc
      have : (unboundedDice : Int) ≤ copies run.greatest := by exact_mod_cast hc
      right
      nlinarith
  · push Not at hlo
    rw [min_eq_right hlo, zero_mul]
    rw [max_eq_left hlo] at c ⊢
    rw [min_eq_right hlo, zero_mul] at a
    refine ⟨.inl a, ?_⟩
    have : (copies run.least : Int) ≤ xs.length := by
      rw [copies_natCast]
      exact_mod_cast (Nat.min_le_left _ _).trans hl
    exact (mul_le_mul_of_nonneg_left this hlo).trans c

/-- The greatest term of a run bounds the sums of the dice that it holds. -/
private theorem Segment.holds_high {run : Segment} {xs : List Int}
    (h : run.Holds xs) :
    negative (xs.map (·, 1)) ≤ Min.min run.highTerm.1 0 * run.highTerm.2 ∧
      (positive (xs.map (·, 1)) ≤ Max.max run.highTerm.1 0 * run.highTerm.2 ∨
        (unboundedDice : Int) ≤
          Max.max run.highTerm.1 0 * run.highTerm.2) := by
  obtain ⟨hl, hg, hm⟩ := h
  obtain ⟨-, b, -, d⟩ := singles_bounds (lo := low run.results)
    (hi := high run.results) fun z hz => ⟨low_le (hm z hz), le_high (hm z hz)⟩
  have hneg := negative_nonpos (xs.map (·, 1))
  simp only [Segment.highTerm]
  split_ifs with hhi
  · rw [max_eq_left hhi.le] at d ⊢
    rw [min_eq_right hhi.le, zero_mul]
    refine ⟨hneg, ?_⟩
    have hc := min_le_copies hg
    by_cases hn : xs.length ≤ unboundedDice
    · rw [Nat.min_eq_left hn] at hc
      have : (xs.length : Int) ≤ copies run.greatest := by exact_mod_cast hc
      exact .inl (d.trans (mul_le_mul_of_nonneg_left this hhi.le))
    · rw [Nat.min_eq_right (by omega)] at hc
      have : (unboundedDice : Int) ≤ copies run.greatest := by exact_mod_cast hc
      right
      nlinarith
  · push Not at hhi
    rw [max_eq_right hhi, zero_mul]
    rw [min_eq_left hhi] at b ⊢
    rw [max_eq_right hhi, zero_mul] at d
    refine ⟨?_, .inl d⟩
    have : (copies run.least : Int) ≤ xs.length := by
      rw [copies_natCast]
      exact_mod_cast (Nat.min_le_left _ _).trans hl
    exact b.trans (mul_le_mul_of_nonpos_left this hhi)

/-- The least terms of some runs bound the sums of the dice that they hold. -/
private theorem sums_low {runs : List Segment} {parts : List (List Int)}
    (h : List.Forall₂ Segment.Holds runs parts) :
    (negative (runs.map Segment.lowTerm) ≤ negative (parts.flatten.map (·, 1)) ∨
      negative (runs.map Segment.lowTerm) ≤ -(unboundedDice : Int)) ∧
      positive (runs.map Segment.lowTerm) ≤
        positive (parts.flatten.map (·, 1)) := by
  induction h with
  | nil => simp
  | @cons run xs runs parts hr _ ih =>
    obtain ⟨h₁, h₂⟩ := run.holds_low hr
    have := negative_nonpos (runs.map Segment.lowTerm)
    have := negative_nonpos (parts.flatten.map (·, 1))
    have := negative_nonpos (xs.map (·, 1))
    have : Min.min run.lowTerm.1 0 * run.lowTerm.2 ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (min_le_right _ _) (by omega)
    simp only [List.map_cons, List.flatten_cons, List.map_append,
      negative_cons, positive_cons, negative_append, positive_append]
    omega

/-- The greatest terms of some runs bound the sums of the dice that they
hold. -/
private theorem sums_high {runs : List Segment} {parts : List (List Int)}
    (h : List.Forall₂ Segment.Holds runs parts) :
    negative (parts.flatten.map (·, 1)) ≤ negative (runs.map Segment.highTerm) ∧
      (positive (parts.flatten.map (·, 1)) ≤
          positive (runs.map Segment.highTerm) ∨
        (unboundedDice : Int) ≤ positive (runs.map Segment.highTerm)) := by
  induction h with
  | nil => simp
  | @cons run xs runs parts hr _ ih =>
    obtain ⟨h₁, h₂⟩ := run.holds_high hr
    have := positive_nonneg (runs.map Segment.highTerm)
    have := positive_nonneg (parts.flatten.map (·, 1))
    have := positive_nonneg (xs.map (·, 1))
    have : 0 ≤ Max.max run.highTerm.1 0 * run.highTerm.2 :=
      mul_nonneg (le_max_right _ _) (by omega)
    simp only [List.map_cons, List.flatten_cons, List.map_append,
      negative_cons, positive_cons, negative_append, positive_append]
    omega

/--
A chain that may still grow totals the fold of its last die and its other
dice, one copy each.

# Parameters
- `body`: The other dice.
- `last`: The last die.
-/
theorem total_live (body : List Int) (last : Int) :
    (Chain.live body last).total = fold ((last, 1) :: body.map (·, 1)) := by
  refine (fold_eq_total ?_).symm
  simp only [Roll.expand, List.flatMap_cons, List.replicate_one,
    List.singleton_append]
  rw [show List.flatMap (fun p : Int × Nat => List.replicate p.2 p.1)
      (body.map ((·, 1))) = body from expand_singles body]

/--
The totals of a shape bound the total of every chain of the shape: the fold
of the least results, each run at the number of dice that lowers it most, is
no more than the total, and the fold of the greatest, each run at the number
that raises it most, is no less, since the fold never decreases as any die
grows (`fold_mono_results`), nor moves against the sign of a die as their
number grows (`fold_mono_copies`), and saturates at `unboundedDice` copies of
a nonzero die (`fold_unbounded`).

# Parameters
- `S`: The shape.
- `ch`: The chain.

# Hypotheses
- `h`: The chain has the shape.

# Examples
```lean
example : (ChainShape.totals
    { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1, 2, 3, 4, 5} }).1 ≤ (Chain.live [6, 6] 3).total :=
  (ChainShape.totals_le
    (S := { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1, 2, 3, 4, 5} })
    (ch := .live [6, 6] 3) ⟨by decide, [[6, 6]], by decide, by decide⟩).1
```
-/
theorem ChainShape.totals_le {S : ChainShape} {ch : Chain} (h : S.Has ch) :
    S.totals.1 ≤ ch.total ∧ ch.total ≤ S.totals.2 := by
  cases ch with
  | saturated v => exact h.elim
  | live body last =>
    obtain ⟨hl, parts, hp, hb⟩ := h
    rw [total_live, fold_perm ((hb.map _).cons _)]
    obtain ⟨hn₁, hp₁⟩ := sums_low hp
    obtain ⟨hn₂, hp₂⟩ := sums_high hp
    have := min_le_min_right 0 (low_le hl)
    have := max_le_max_right 0 (low_le hl)
    have := min_le_min_right 0 (le_high hl)
    have := max_le_max_right 0 (le_high hl)
    have := negative_nonpos (S.middles.map Segment.lowTerm)
    have := positive_nonneg (S.middles.map Segment.highTerm)
    have := min_le_right (low S.last) 0
    have := le_max_right (high S.last) 0
    constructor
    · refine fold_le_of_sums ?_ ?_ <;>
        simp only [negative_cons, positive_cons, Nat.cast_one, mul_one] <;>
        omega
    · refine le_fold_of_sums ?_ ?_ <;>
        simp only [negative_cons, positive_cons, Nat.cast_one, mul_one] <;>
        omega

/-- Runs that may hold no die hold none, so dropping them leaves the dice. -/
private theorem holds_filter {runs : List Segment} {parts : List (List Int)}
    (h : List.Forall₂ Segment.Holds runs parts) :
    ∃ parts', List.Forall₂ Segment.Holds
        (runs.filter fun run => decide (0 < run.greatest)) parts' ∧
      parts'.flatten = parts.flatten := by
  induction h with
  | nil => exact ⟨[], .nil, rfl⟩
  | @cons run xs runs parts hr _ ih =>
    obtain ⟨parts', hp, he⟩ := ih
    by_cases hg : 0 < run.greatest
    · refine ⟨xs :: parts', ?_, by simp [he]⟩
      simpa [List.filter_cons, hg] using List.Forall₂.cons hr hp
    · have : xs = [] := by
        have := hr.2.1
        rw [not_lt, nonpos_iff_eq_zero] at hg
        rw [hg, nonpos_iff_eq_zero, Nat.cast_eq_zero, List.length_eq_zero_iff]
          at this
        exact this
      subst this
      exact ⟨parts', by simpa [List.filter_cons, hg] using hp, by simp [he]⟩

/-- Merging runs that show the same results merges the dice that they
hold. -/
private theorem holds_mergeRuns :
    ∀ {runs : List Segment} {parts : List (List Int)},
      List.Forall₂ Segment.Holds runs parts →
      ∃ parts', List.Forall₂ Segment.Holds (mergeRuns runs) parts' ∧
        parts'.flatten = parts.flatten
  | [], _, h => by
    cases h
    exact ⟨[], .nil, rfl⟩
  | run :: runs, _, .cons (b := xs) (l₂ := parts) hr h => by
    obtain ⟨parts', hp, he⟩ := holds_mergeRuns h
    simp only [mergeRuns]
    split
    · rename_i next rest hm
      rw [hm] at hp
      obtain ⟨ys, parts'', rfl, hy, hp⟩ :
          ∃ ys parts'', parts' = ys :: parts'' ∧ next.Holds ys ∧
            List.Forall₂ Segment.Holds rest parts'' := by
        cases hp with
        | cons hy hp => exact ⟨_, _, rfl, hy, hp⟩
      split
      · rename_i heq
        refine ⟨(xs ++ ys) :: parts'', .cons ⟨?_, ?_, ?_⟩ hp, ?_⟩
        · simp only [List.length_append]
          exact Nat.add_le_add hr.1 hy.1
        · simp only [List.length_append, Nat.cast_add]
          exact add_le_add hr.2.1 hy.2.1
        · intro z hz
          rcases List.mem_append.mp hz with hz | hz
          · exact hr.2.2 z hz
          · exact heq ▸ hy.2.2 z hz
        · simp [← he]
      · exact ⟨xs :: ys :: parts'', .cons hr (.cons hy hp), by simp [← he]⟩
    · rename_i hm
      rw [hm] at hp
      cases hp
      exact ⟨[xs], .cons hr .nil, by simp [← he]⟩

/--
A chain of some runs and a last die has the shape that they make, whose runs
are sorted and merged, so `ChainShape.new` loses no chain.

# Parameters
- `middles`: The runs.
- `last`: The results that the last die may show.
- `ch`: The chain.

# Hypotheses
- `h`: The chain has the shape of the runs as they are.
-/
theorem ChainShape.has_new {middles : List Segment} {last : Finset Int}
    {ch : Chain} (h : ChainShape.Has { middles, last } ch) :
    (ChainShape.new middles last).Has ch := by
  cases ch with
  | saturated v => exact h.elim
  | live body y =>
    obtain ⟨hy, parts, hp, hb⟩ := h
    obtain ⟨parts₁, hp₁, he₁⟩ := holds_filter hp
    obtain ⟨parts₂, hp₂, he₂⟩ := List.perm_comp_forall₂
      (List.mergeSort_perm _ _) hp₁
    obtain ⟨parts₃, hp₃, he₃⟩ := holds_mergeRuns hp₂
    refine ⟨hy, parts₃, hp₃, ?_⟩
    rw [he₃]
    exact hb.trans (he₁ ▸ he₂.flatten.symm)

/-! ## Transforming and uniting shapes -/

/--
The run whose dice show what a function makes of the results of a run's.

# Parameters
- `f`: The function from the results that a die may show to those that it
  may show afterward.
- `run`: The run.

# Returns
The run, of as many dice.
-/
def Segment.mapResults (f : Finset Int → Finset Int) (run : Segment) :
    Segment :=
  { run with results := f run.results }

/--
The shape whose dice show what a function makes of the results of the runs
and the last die of a shape, run by run, as a reroll replaces each die alike.
Mirrors `ChainShape::map` in `evaluator.rs`.

# Parameters
- `S`: The shape.
- `f`: The function from the results that a die may show to those that it
  may show afterward.

# Returns
The shape.

# Examples
```lean
#guard (ChainShape.map
    { middles := [{ results := {6}, least := 1, greatest := 2 }],
      last := {1, 6} } (· ∪ {7})) ==
  { middles := [{ results := {6, 7}, least := 1, greatest := 2 }],
    last := {1, 6, 7} }
```
-/
def ChainShape.map (S : ChainShape) (f : Finset Int → Finset Int) :
    ChainShape :=
  .new (S.middles.map (Segment.mapResults f)) (f S.last)

/--
Whether every die that a run may hold another may hold: its results are among
the other's, and the other holds as few dice and as many.

# Parameters
- `run`, `other`: The runs.
-/
def Segment.Within (run other : Segment) : Prop :=
  run.results ⊆ other.results ∧ other.least ≤ run.least ∧
    run.greatest ≤ other.greatest

/-- Whether a run lies within another is decided. -/
instance (run other : Segment) : Decidable (run.Within other) := by
  unfold Segment.Within
  infer_instance

/--
Whether every chain of a shape has another, as the runs compared in order
decide it: each run lies within the other's at its position, and the results
of the last die are among the other's. Mirrors `ChainShape::within` in
`evaluator.rs`, which may miss some shapes that another covers, but never
claims one that it does not (`ChainShape.Within.has`).

# Parameters
- `S`, `T`: The shapes.
-/
def ChainShape.Within (S T : ChainShape) : Prop :=
  List.Forall₂ Segment.Within S.middles T.middles ∧ S.last ⊆ T.last

/-- Whether a shape lies within another is decided. -/
instance (S T : ChainShape) : Decidable (S.Within T) := by
  unfold ChainShape.Within
  infer_instance

/--
The greater of two numbers of dice, computed, since the order of `ℕ∞` answers
its `max` only noncomputably.

# Parameters
- `a`, `b`: The numbers.

# Returns
The greater.
-/
def emax (a b : ℕ∞) : ℕ∞ := if a ≤ b then b else a

/-- The computed greater of two numbers is their `max`. -/
theorem emax_eq (a b : ℕ∞) : emax a b = Max.max a b := (max_def a b).symm

/--
One run that stands for two, if the bounds find one: the runs of the same
results, with the fewest and the most dice of either, or the runs of the same
numbers of dice, with the results of either. Mirrors the uniting of one run in
`ChainShape::unite` in `evaluator.rs`.

# Parameters
- `run`, `other`: The runs.

# Returns
The united run, or `none` if the runs differ in both.
-/
def Segment.unite (run other : Segment) : Option Segment :=
  if run.results = other.results then
    some { results := run.results, least := Min.min run.least other.least,
           greatest := emax run.greatest other.greatest }
  else if run.least = other.least ∧ run.greatest = other.greatest then
    some { run with results := run.results ∪ other.results }
  else none

/--
The positions at which the runs of two shapes differ.

# Parameters
- `S`, `T`: The shapes.

# Returns
The positions, ascending, among those of `S`'s runs.
-/
def ChainShape.differences (S T : ChainShape) : List Nat :=
  (List.range S.middles.length).filter fun i => S.middles[i]? != T.middles[i]?

/--
Unite the shape at some position of two shapes' runs, if they unite there.

# Parameters
- `S`, `T`: The shapes.
- `i`: The position.

# Returns
The shape of `S`'s runs, with the run at `i` united with `T`'s, and `S`'s last
die, or `none` if the runs do not unite.
-/
def ChainShape.uniteAt (S T : ChainShape) (i : Nat) : Option ChainShape :=
  match S.middles[i]?, T.middles[i]? with
  | some run, some other =>
    (run.unite other).map fun united => .new (S.middles.set i united) S.last
  | _, _ => none

/--
One shape that stands for two, through every later reroll and explode, if
there is one that the bounds find: either, if it covers the other; or, if
they differ only in their last dice, the shape of the same runs and either's
last die; or, if they differ only in one run, the shape of the other runs, the
same last die, and the runs united, if they unite. Mirrors
`ChainShape::unite` in `evaluator.rs`.

# Parameters
- `S`, `T`: The shapes.

# Returns
The united shape, or `none` if the shapes differ too much.

# Examples
```lean
#guard (ChainShape.unite { middles := [], last := {1} }
    { middles := [], last := {3} }) ==
  some { middles := [], last := {1, 3} }
#guard (ChainShape.unite
    { middles := [{ results := {6}, least := 1, greatest := 1 }],
      last := {1} }
    { middles := [{ results := {6}, least := 3, greatest := 3 }],
      last := {1} }) ==
  some { middles := [{ results := {6}, least := 1, greatest := 3 }],
    last := {1} }
```
-/
def ChainShape.unite (S T : ChainShape) : Option ChainShape :=
  if S.Within T then some T
  else if T.Within S then some S
  else if S.middles.length ≠ T.middles.length then none
  else
    match S.differences T with
    | [] => some { middles := S.middles, last := S.last ∪ T.last }
    | [i] => if S.last = T.last then S.uniteAt T i else none
    | _ => none

/--
The union of some sets of results.

# Parameters
- `sets`: The sets.

# Returns
The union.
-/
def unionAll (sets : List (Finset Int)) : Finset Int := sets.foldr (· ∪ ·) ∅

/-- A result lies in the union of some sets exactly when it lies in one. -/
theorem mem_unionAll {sets : List (Finset Int)} {x : Int} :
    x ∈ unionAll sets ↔ ∃ s ∈ sets, x ∈ s := by
  induction sets with
  | nil => simp [unionAll]
  | cons s sets ih =>
    simp only [unionAll, List.foldr_cons, Finset.mem_union] at ih ⊢
    rw [ih]
    simp

/--
The fewest dice that the runs of a shape may hold together.

# Parameters
- `S`: The shape.

# Returns
The number of dice.
-/
def ChainShape.leastMiddles (S : ChainShape) : Nat :=
  (S.middles.map (·.least)).sum

/--
The most dice that the runs of a shape may hold together.

# Parameters
- `S`: The shape.

# Returns
The number of dice, or `⊤` for any number.
-/
def ChainShape.greatestMiddles (S : ChainShape) : ℕ∞ :=
  (S.middles.map (·.greatest)).sum

/--
One shape that covers every chain of each of some shapes: one run, which may
show any result of any of their runs, and holds as few or as many dice as the
fewest or most of theirs, and a last die that may show any result of any of
theirs. Mirrors `ChainShape::coarsen` in `evaluator.rs`.

# Parameters
- `shapes`: The shapes, which are not empty.

# Returns
The coarsened shape, which may admit totals that none of the shapes admits.

# Examples
```lean
#guard ChainShape.coarsen [
    { middles := [{ results := {6}, least := 1, greatest := 1 }],
      last := {1} },
    { middles := [{ results := {5}, least := 2, greatest := 2 },
        { results := {6}, least := 1, greatest := 1 }], last := {2} }] ==
  { middles := [{ results := {5, 6}, least := 1, greatest := 3 }],
    last := {1, 2} }
```
-/
def ChainShape.coarsen (shapes : List ChainShape) : ChainShape :=
  let results := unionAll ((shapes.flatMap (·.middles)).map (·.results))
  let last := unionAll (shapes.map (·.last))
  if results = ∅ then .new [] last
  else
    .new [{ results,
            least := ((shapes.map ChainShape.leastMiddles).min?).getD 0,
            greatest := (shapes.map ChainShape.greatestMiddles).foldr emax 0 }]
      last

/--
The greatest number of shapes of chain that the bounds keep for one die,
beyond which they coarsen them into one. Mirrors `SHAPE_BUDGET` in
`evaluator.rs`.
-/
def shapeBudget : Nat := 64

/--
Remove an element of a list by moving the last element into its place, as
`Vec::swap_remove` in Rust does.

# Parameters
- `l`: The list.
- `i`: The position of the element, less than the length of `l`.

# Returns
The list without the element.

# Examples
```lean
#guard Bounds.swapRemove [1, 2, 3, 4] 1 == [1, 4, 3]
#guard Bounds.swapRemove [1, 2, 3, 4] 3 == [1, 2, 3]
```
-/
def swapRemove {α : Type} (l : List α) (i : Nat) : List α :=
  match l.getLast? with
  | some x => (l.set i x).dropLast
  | none => []

/-- Moving the last element of a list into a place, and dropping it from the
end, leaves the other elements. -/
private theorem perm_set_concat {α : Type} (x : α) :
    ∀ (init : List α) (i : Nat) (h : i < init.length),
      (init ++ [x]).Perm (init[i] :: init.set i x)
  | [], _, h => absurd h (by simp)
  | y :: ys, 0, _ => by simp
  | y :: ys, i + 1, h => by
    have := perm_set_concat x ys i (by simpa using h)
    simp only [List.cons_append, List.getElem_cons_succ, List.set_cons_succ]
    exact (this.cons y).trans (.swap _ _ _)

/--
A list is the element at a position and the list without it, in some order.

# Parameters
- `l`: The list.
- `i`: The position.

# Hypotheses
- `h`: The position lies within the list.
-/
theorem perm_swapRemove {α : Type} {l : List α} {i : Nat} (h : i < l.length) :
    l.Perm (l[i] :: swapRemove l i) := by
  rcases List.eq_nil_or_concat' l with rfl | ⟨init, x, rfl⟩
  · simp at h
  · simp only [swapRemove, List.getLast?_concat]
    by_cases hi : i < init.length
    · rw [List.set_append_left _ _ hi, List.dropLast_concat,
        List.getElem_append_left hi]
      exact perm_set_concat x init i hi
    · have hi : i = init.length := by simp at h; omega
      subst hi
      simp only [List.set_append_right _ _ (le_refl _), Nat.sub_self,
        List.set_cons_zero, List.dropLast_concat, List.getElem_concat_length]
      exact List.perm_append_singleton _ _

/-- Removing an element shortens a list by one. -/
theorem length_swapRemove {α : Type} {l : List α} {i : Nat}
    (h : i < l.length) : (swapRemove l i).length + 1 = l.length := by
  simpa using (perm_swapRemove h).length_eq.symm

/--
Unite a shape with every shape of some shapes already united with which the
bounds find that it unites, from some position on, beginning again after each,
since a shape that unites with another may then unite with a third. Mirrors
the inner loop of `unite_shapes` in `evaluator.rs`.

# Parameters
- `shape`: The shape.
- `united`: The shapes already united.
- `index`: The position from which to try them.

# Returns
The shape, united with every shape with which it unites, and the shapes with
which it did not.
-/
def absorb (shape : ChainShape) (united : List ChainShape) (index : Nat) :
    ChainShape × List ChainShape :=
  if h : index < united.length then
    match united[index].unite shape with
    | some both => absorb both (swapRemove united index) 0
    | none => absorb shape united (index + 1)
  else (shape, united)
termination_by (united.length, united.length - index)
decreasing_by
  · left
    have := length_swapRemove h
    omega
  · right
    omega

/--
Unite one more shape with some shapes already united, as `unite_shapes` in
`evaluator.rs` does for each: absorb it, push it, and coarsen the shapes into
one if they exceed `shapeBudget`.

# Parameters
- `acc`: The shapes already united, and whether they were ever coarsened.
- `shape`: The shape.

# Returns
The shapes united, and whether they were ever coarsened.
-/
def uniteStep (acc : List ChainShape × Bool) (shape : ChainShape) :
    List ChainShape × Bool :=
  let absorbed := absorb shape acc.1 0
  let united := absorbed.2 ++ [absorbed.1]
  if shapeBudget < united.length then ([ChainShape.coarsen united], true)
  else (united, acc.2)

/--
Unite some shapes into as few as the bounds find that stand for all of them,
through every later reroll and explode, coarsening them into one whenever
they exceed `shapeBudget`. Mirrors `unite_shapes` in `evaluator.rs`, which
does not report whether it coarsened; the proofs of exactness ask that it did
not.

# Parameters
- `shapes`: The shapes.

# Returns
The united shapes, and whether they were coarsened.

# Examples
```lean
#guard Bounds.uniteShapes [{ middles := [], last := {1} },
    { middles := [], last := {3} }] ==
  ([{ middles := [], last := {1, 3} }], false)
```
-/
def uniteShapes (shapes : List ChainShape) : List ChainShape × Bool :=
  shapes.foldl uniteStep ([], false)

/-- Every die of a run that lies within another the other may hold. -/
private theorem Segment.Within.holds {run other : Segment} {xs : List Int}
    (h : run.Within other) (hr : run.Holds xs) : other.Holds xs :=
  ⟨h.2.1.trans hr.1, hr.2.1.trans h.2.2, fun x hx => h.1 (hr.2.2 x hx)⟩

/-- The dice that some runs hold, runs that lie within them may hold. -/
private theorem forall₂_within {l l' : List Segment}
    (h : List.Forall₂ Segment.Within l l') :
    ∀ {parts : List (List Int)}, List.Forall₂ Segment.Holds l parts →
      List.Forall₂ Segment.Holds l' parts := by
  induction h with
  | nil => exact id
  | cons hw _ ih =>
    intro parts hp
    cases hp with
    | cons hr hp => exact .cons (hw.holds hr) (ih hp)

/--
Every chain of a shape that lies within another has the other.

# Parameters
- `S`, `T`: The shapes.
- `ch`: The chain.

# Hypotheses
- `h`: `S` lies within `T`.
- `hc`: The chain has `S`.
-/
theorem ChainShape.Within.has {S T : ChainShape} {ch : Chain}
    (h : S.Within T) (hc : S.Has ch) : T.Has ch := by
  cases ch with
  | saturated v => exact hc.elim
  | live body last =>
    obtain ⟨hl, parts, hp, hb⟩ := hc
    exact ⟨h.2 hl, parts, forall₂_within h.1 hp, hb⟩

/-- Every run that one run unites with another stands for both. -/
private theorem Segment.unite_holds {run other united : Segment}
    (h : run.unite other = some united) {xs : List Int}
    (hx : run.Holds xs ∨ other.Holds xs) : united.Holds xs := by
  unfold Segment.unite at h
  by_cases h₁ : run.results = other.results
  · simp only [h₁, ↓reduceIte] at h
    cases h
    rcases hx with ⟨a, b, c⟩ | ⟨a, b, c⟩
    · refine ⟨(Nat.min_le_left _ _).trans a, ?_, fun x hx => h₁ ▸ c x hx⟩
      rw [emax_eq]
      exact b.trans (le_max_left _ _)
    · refine ⟨(Nat.min_le_right _ _).trans a, ?_, c⟩
      rw [emax_eq]
      exact b.trans (le_max_right _ _)
  · simp only [h₁, ↓reduceIte] at h
    by_cases h₂ : run.least = other.least ∧ run.greatest = other.greatest
    · simp only [h₂] at h
      cases h
      rcases hx with ⟨a, b, c⟩ | ⟨a, b, c⟩
      · exact ⟨h₂.1 ▸ a, h₂.2 ▸ b,
          fun x hx => Finset.mem_union_left _ (c x hx)⟩
      · exact ⟨a, b, fun x hx => Finset.mem_union_right _ (c x hx)⟩
    · simp only [h₂, ↓reduceIte] at h
      cases h

/-- The dice that runs hold, with one run replaced by another that may hold
whatever it holds. -/
private theorem forall₂_set {R : Segment → List Int → Prop} {a b : Segment}
    (hab : ∀ xs, R a xs → R b xs) :
    ∀ {l : List Segment} {parts : List (List Int)} {i : Nat},
      List.Forall₂ R (l.set i a) parts → List.Forall₂ R (l.set i b) parts
  | [], _, _, h => by simpa using h
  | _ :: _, _, 0, .cons hr h => .cons (hab _ hr) h
  | _ :: _, _, _ + 1, .cons hr h => .cons hr (forall₂_set hab h)

/-- A list holds the element that it holds at a position, there. -/
private theorem set_self_of_getElem? {l : List Segment} {i : Nat}
    {a : Segment} (h : l[i]? = some a) : l.set i a = l :=
  List.ext_getElem? fun j => by
    by_cases hij : i = j
    · subst hij
      rw [List.getElem?_set_self (List.getElem?_eq_some_iff.mp h).1, h]
    · rw [List.getElem?_set_ne hij]

/--
What two shapes unite into, case by case: either, if it covers the other; the
shape of their runs, which agree, and the last dice of either; or the shape of
their runs, which agree but at one position, with the runs there united, and
the last die, which they share.

# Parameters
- `S`, `T`: The shapes.
- `U`: The united shape.

# Hypotheses
- `h`: `S` and `T` unite into `U`.
-/
theorem ChainShape.unite_cases {S T U : ChainShape} (h : S.unite T = some U) :
    (S.Within T ∧ U = T) ∨ (T.Within S ∧ U = S) ∨
      (S.middles = T.middles ∧
        U = { middles := S.middles, last := S.last ∪ T.last }) ∨
      ∃ i run other united, S.middles[i]? = some run ∧
        S.middles.set i other = T.middles ∧ S.last = T.last ∧
        run.unite other = some united ∧
        U = .new (S.middles.set i united) S.last := by
  unfold ChainShape.unite at h
  by_cases h₁ : S.Within T
  · simp only [h₁, ↓reduceIte, Option.some.injEq] at h
    exact .inl ⟨h₁, h.symm⟩
  simp only [h₁, ↓reduceIte] at h
  by_cases h₂ : T.Within S
  · simp only [h₂, ↓reduceIte, Option.some.injEq] at h
    exact .inr (.inl ⟨h₂, h.symm⟩)
  simp only [h₂, ↓reduceIte] at h
  by_cases h₃ : S.middles.length ≠ T.middles.length
  · simp only [ne_eq, h₃, not_false_eq_true, ↓reduceIte] at h
    cases h
  push Not at h₃
  simp only [h₃, ne_eq, not_true_eq_false, ↓reduceIte] at h
  have hd : ∀ i, i ∈ S.differences T ↔ i < S.middles.length ∧
      S.middles[i]? ≠ T.middles[i]? := fun i => by
    simp [ChainShape.differences]
  -- Runs at no position of difference, or beyond both shapes, agree.
  have hagree : ∀ j, j ∉ S.differences T → S.middles[j]? = T.middles[j]? :=
    fun j hj => by
      by_contra hne
      by_cases hjl : j < S.middles.length
      · exact hj ((hd j).mpr ⟨hjl, hne⟩)
      · rw [List.getElem?_eq_none (by omega),
          List.getElem?_eq_none (by omega)] at hne
        exact hne rfl
  rcases hdiff : S.differences T with _ | ⟨i, _ | ⟨j, rest⟩⟩
  · rw [hdiff] at h hagree
    simp only [Option.some.injEq] at h
    exact .inr (.inr (.inl ⟨List.ext_getElem? fun j => hagree j (by simp),
      h.symm⟩))
  · rw [hdiff] at h hagree
    simp only at h
    by_cases hl : S.last = T.last
    · simp only [hl, ↓reduceIte] at h
      unfold ChainShape.uniteAt at h
      rcases hrun : S.middles[i]? with _ | run
      · simp [hrun] at h
      rcases hother : T.middles[i]? with _ | other
      · simp [hrun, hother] at h
      simp only [hrun, hother, Option.map_eq_some_iff] at h
      obtain ⟨united, hu, rfl⟩ := h
      refine .inr (.inr (.inr ⟨i, run, other, united, hrun, ?_, hl, hu, ?_⟩))
      · refine List.ext_getElem? fun j => ?_
        by_cases hij : i = j
        · subst hij
          rw [List.getElem?_set_self (List.getElem?_eq_some_iff.mp hrun).1,
            hother]
        · rw [List.getElem?_set_ne hij]
          exact hagree j (by simpa using Ne.symm hij)
      · rw [hl]
    · simp only [hl, ↓reduceIte] at h
      cases h
  · rw [hdiff] at h
    cases h

/--
A shape that two shapes unite into has every chain of either.

# Parameters
- `S`, `T`: The shapes.
- `U`: The united shape.
- `ch`: The chain.

# Hypotheses
- `h`: `S` and `T` unite into `U`.
- `hc`: The chain has `S` or `T`.
-/
theorem ChainShape.unite_has {S T U : ChainShape} (h : S.unite T = some U)
    {ch : Chain} (hc : S.Has ch ∨ T.Has ch) : U.Has ch := by
  rcases ChainShape.unite_cases h with ⟨h₁, rfl⟩ | ⟨h₂, rfl⟩ |
      ⟨he, rfl⟩ | ⟨i, run, other, united, hrun, hT, hl, hu, rfl⟩
  · exact hc.elim h₁.has id
  · exact hc.elim id h₂.has
  · cases ch with
    | saturated v => exact hc.elim id id
    | live body last =>
      rcases hc with ⟨hl, hp⟩ | ⟨hl, hp⟩
      · exact ⟨Finset.mem_union_left _ hl, hp⟩
      · exact ⟨Finset.mem_union_right _ hl, he ▸ hp⟩
  · refine ChainShape.has_new ?_
    cases ch with
    | saturated v => exact hc.elim id id
    | live body last =>
      rcases hc with ⟨hl', parts, hp, hb⟩ | ⟨hl', parts, hp, hb⟩
      · refine ⟨hl', parts, ?_, hb⟩
        rw [← set_self_of_getElem? hrun] at hp
        exact forall₂_set (fun xs hr => Segment.unite_holds hu (.inl hr)) hp
      · refine ⟨hl ▸ hl', parts, ?_, hb⟩
        rw [← hT] at hp
        exact forall₂_set (fun xs hr => Segment.unite_holds hu (.inr hr)) hp

/-- Some run holds each of the parts that runs hold. -/
private theorem exists_holds {runs : List Segment} {parts : List (List Int)}
    (h : List.Forall₂ Segment.Holds runs parts) {xs : List Int}
    (hxs : xs ∈ parts) : ∃ run ∈ runs, run.Holds xs := by
  induction h with
  | nil => simp at hxs
  | @cons run ys runs parts hr _ ih =>
    rcases List.mem_cons.mp hxs with rfl | hxs
    · exact ⟨run, List.mem_cons_self, hr⟩
    · obtain ⟨r, hr', h⟩ := ih hxs
      exact ⟨r, List.mem_cons_of_mem _ hr', h⟩

/-- The fewest of some numbers is no more than any of them. -/
private theorem min?_getD_le {l : List Nat} {a : Nat} (h : a ∈ l) :
    l.min?.getD 0 ≤ a := by
  cases hm : l.min? with
  | none =>
    rw [List.min?_eq_none_iff] at hm
    simp [hm] at h
  | some m => exact (List.min?_eq_some_iff.mp hm).2 a h

/-- The greatest of some numbers is no less than any of them. -/
private theorem le_foldr_emax {l : List ℕ∞} {a : ℕ∞} (h : a ∈ l) :
    a ≤ l.foldr emax 0 := by
  induction l with
  | nil => simp at h
  | cons b l ih =>
    rw [List.foldr_cons, emax_eq]
    rcases List.mem_cons.mp h with rfl | h
    · exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

/-- Runs hold together as many dice as they hold each. -/
private theorem holds_sum {runs : List Segment} {parts : List (List Int)}
    (h : List.Forall₂ Segment.Holds runs parts) :
    (runs.map (·.least)).sum ≤ parts.flatten.length ∧
      (parts.flatten.length : ℕ∞) ≤ (runs.map (·.greatest)).sum := by
  induction h with
  | nil => simp
  | cons hr _ ih =>
    simp only [List.map_cons, List.sum_cons, List.flatten_cons,
      List.length_append, Nat.cast_add]
    exact ⟨Nat.add_le_add hr.1 ih.1, add_le_add hr.2.1 ih.2⟩

/--
The coarsened shape of some shapes has every chain of each.

# Parameters
- `shapes`: The shapes.
- `S`: One of them.
- `ch`: The chain.

# Hypotheses
- `hS`: `S` is one of the shapes.
- `hc`: The chain has `S`.
-/
theorem ChainShape.coarsen_has {shapes : List ChainShape} {S : ChainShape}
    (hS : S ∈ shapes) {ch : Chain} (hc : S.Has ch) :
    (ChainShape.coarsen shapes).Has ch := by
  cases ch with
  | saturated v => exact hc.elim
  | live body last =>
    obtain ⟨hl, parts, hp, hb⟩ := hc
    have hlast : last ∈ unionAll (shapes.map (·.last)) :=
      mem_unionAll.mpr ⟨S.last, List.mem_map_of_mem hS, hl⟩
    -- Every die of the chain shows a result of some run of some shape.
    have hmem : ∀ x ∈ parts.flatten,
        x ∈ unionAll ((shapes.flatMap (·.middles)).map (·.results)) := by
      intro x hx
      obtain ⟨xs, hxs, hx⟩ := List.mem_flatten.mp hx
      obtain ⟨run, hrun, hr⟩ := exists_holds hp hxs
      exact mem_unionAll.mpr ⟨run.results, List.mem_map_of_mem
        (List.mem_flatMap.mpr ⟨S, hS, hrun⟩), hr.2.2 x hx⟩
    dsimp only [ChainShape.coarsen]
    split_ifs with hr
    · refine ChainShape.has_new ⟨hlast, [], .nil, ?_⟩
      have : parts.flatten = [] := List.eq_nil_iff_forall_not_mem.mpr
        fun x hx => by simpa [hr] using hmem x hx
      simpa [this] using hb
    · refine ChainShape.has_new ⟨hlast, [parts.flatten],
        .cons ⟨?_, ?_, hmem⟩ .nil, by simpa using hb⟩
      · exact (min?_getD_le (List.mem_map_of_mem hS)).trans (holds_sum hp).1
      · exact (holds_sum hp).2.trans (le_foldr_emax (List.mem_map_of_mem hS))

/--
Absorbing a shape into shapes already united loses no chain of either: every
chain of the shape, or of one of them, has the shape that answers, or one of
the shapes with which it did not unite.

# Parameters
- `shape`: The shape.
- `united`: The shapes already united.
- `index`: The position from which to try them.
- `ch`: The chain.

# Hypotheses
- `h`: The chain has the shape, or one of the shapes already united.
-/
theorem absorb_has {shape : ChainShape} {united : List ChainShape}
    {index : Nat} {ch : Chain} (h : shape.Has ch ∨ ∃ U ∈ united, U.Has ch) :
    (absorb shape united index).1.Has ch ∨
      ∃ U ∈ (absorb shape united index).2, U.Has ch := by
  fun_induction absorb shape united index with
  | case1 shape united index hi both hu ih =>
    refine ih ?_
    rcases h with h | ⟨U, hU, h⟩
    · exact .inl (ChainShape.unite_has hu (.inr h))
    · rcases List.mem_cons.mp ((perm_swapRemove hi).mem_iff.mp hU) with rfl | hU
      · exact .inl (ChainShape.unite_has hu (.inl h))
      · exact .inr ⟨U, hU, h⟩
  | case2 shape united index hi hu ih => exact ih h
  | case3 shape united index hi => exact h

/-- Uniting one more shape loses no chain of it, or of the shapes already
united. -/
private theorem uniteStep_has {acc : List ChainShape × Bool}
    {shape : ChainShape} {ch : Chain}
    (h : (∃ U ∈ acc.1, U.Has ch) ∨ shape.Has ch) :
    ∃ U ∈ (uniteStep acc shape).1, U.Has ch := by
  have ha := absorb_has (index := 0) (h.elim .inr .inl)
  have hu : ∃ U ∈ (absorb shape acc.1 0).2 ++ [(absorb shape acc.1 0).1],
      U.Has ch := by
    rcases ha with ha | ⟨U, hU, ha⟩
    · exact ⟨_, List.mem_append_right _ (List.mem_singleton.mpr rfl), ha⟩
    · exact ⟨U, List.mem_append_left _ hU, ha⟩
  dsimp only [uniteStep]
  split_ifs
  · obtain ⟨U, hU, hc⟩ := hu
    exact ⟨_, List.mem_singleton.mpr rfl, ChainShape.coarsen_has hU hc⟩
  · exact hu

/--
Uniting some shapes loses no chain of any of them: every chain of one has
one of the united shapes, whether or not they were coarsened.

# Parameters
- `shapes`: The shapes.
- `S`: One of them.
- `ch`: The chain.

# Hypotheses
- `hS`: `S` is one of the shapes.
- `hc`: The chain has `S`.
-/
theorem uniteShapes_has {shapes : List ChainShape} {S : ChainShape}
    (hS : S ∈ shapes) {ch : Chain} (hc : S.Has ch) :
    ∃ U ∈ (uniteShapes shapes).1, U.Has ch := by
  have key : ∀ (shapes : List ChainShape) (acc : List ChainShape × Bool),
      ((∃ U ∈ acc.1, U.Has ch) ∨ ∃ S ∈ shapes, S.Has ch) →
        ∃ U ∈ (shapes.foldl uniteStep acc).1, U.Has ch := by
    intro shapes
    induction shapes with
    | nil => intro acc h; simpa using h
    | cons S shapes ih =>
      intro acc h
      refine ih _ ?_
      rcases h with h | ⟨T, hT, h⟩
      · exact .inl (uniteStep_has (.inl h))
      · rcases List.mem_cons.mp hT with rfl | hT
        · exact .inl (uniteStep_has (.inr h))
        · exact .inr ⟨T, hT, h⟩
  exact key shapes ([], false) (.inr ⟨S, hS, hc⟩)

/-- Dice related one by one to the dice of some parts split into parts
related one by one. -/
private theorem forall₂_flatten {R : Int → Int → Prop} :
    ∀ {parts : List (List Int)} {ys : List Int},
      List.Forall₂ R parts.flatten ys →
      ∃ parts', List.Forall₂ (List.Forall₂ R) parts parts' ∧
        ys = parts'.flatten
  | [], ys, h => by
    simp only [List.flatten_nil, List.forall₂_nil_left_iff] at h
    exact ⟨[], .nil, by simp [h]⟩
  | xs :: parts, ys, h => by
    rw [List.flatten_cons] at h
    have h₁ := List.forall₂_take xs.length h
    have h₂ := List.forall₂_drop xs.length h
    rw [List.take_left'] at h₁
    · rw [List.drop_left'] at h₂
      · obtain ⟨parts', hp, he⟩ := forall₂_flatten h₂
        exact ⟨_ :: parts', .cons h₁ hp, by
          rw [List.flatten_cons, ← he, List.take_append_drop]⟩
      · rfl
    · rfl

/-- Dice related one by one to dice of some results show what a function
makes of them. -/
private theorem holds_map {f : Finset Int → Finset Int} {run : Segment}
    {xs ys : List Int} (hr : run.Holds xs)
    (h : List.Forall₂ (fun x y => ∀ R : Finset Int, x ∈ R → y ∈ f R) xs ys) :
    (run.mapResults f).Holds ys := by
  refine ⟨h.length_eq ▸ hr.1, h.length_eq ▸ hr.2.1, ?_⟩
  have hx := hr.2.2
  clear hr
  induction h with
  | nil => simp
  | cons hxy _ ih =>
    intro y hy
    rcases List.mem_cons.mp hy with rfl | hy
    · exact hxy _ (hx _ List.mem_cons_self)
    · exact ih (fun x h => hx x (List.mem_cons_of_mem _ h)) y hy

/-- Dice related one by one to the dice that some runs hold show what a
function makes of their results. -/
private theorem forall₂_holds_map {f : Finset Int → Finset Int}
    {runs : List Segment} {parts parts' : List (List Int)}
    (hp : List.Forall₂ Segment.Holds runs parts)
    (hp' : List.Forall₂ (List.Forall₂ fun x y =>
      ∀ R : Finset Int, x ∈ R → y ∈ f R) parts parts') :
    List.Forall₂ Segment.Holds (runs.map (Segment.mapResults f)) parts' := by
  induction hp generalizing parts' with
  | nil => cases hp'; exact .nil
  | cons hr _ ih =>
    cases hp' with
    | cons hxy hp' => exact .cons (holds_map hr hxy) (ih hp')

/--
A chain whose dice each show what a function makes of the results that the
dice of a chain of a shape may show has the mapped shape, as a reroll
replaces each die of a chain alike.

# Parameters
- `S`: The shape.
- `f`: The function from the results that a die may show to those that it
  may show afterward.
- `body`, `last`: The chain.
- `body'`, `last'`: The chain afterward.

# Hypotheses
- `hc`: The chain has the shape.
- `hb`: Each die of the body becomes the die at its position afterward, which
  shows what `f` makes of any results that the die showed.
- `hl`: So does the last die.
-/
theorem ChainShape.map_has {S : ChainShape} {f : Finset Int → Finset Int}
    {body body' : List Int} {last last' : Int} (hc : S.Has (.live body last))
    (hb : List.Forall₂ (fun x y => ∀ R : Finset Int, x ∈ R → y ∈ f R) body
      body')
    (hl : ∀ R : Finset Int, last ∈ R → last' ∈ f R) :
    (S.map f).Has (.live body' last') := by
  obtain ⟨hlast, parts, hp, hperm⟩ := hc
  obtain ⟨m, hm, hm'⟩ := List.perm_comp_forall₂ hperm.symm hb
  obtain ⟨parts', hp', rfl⟩ := forall₂_flatten hm
  exact ChainShape.has_new
    ⟨hl _ hlast, parts', forall₂_holds_map hp hp', hm'.symm⟩


/-! ## The chains of a die -/

/-- The criteria to which the predicate of a reroll or explode may resolve,
over the bounds of its operands, as the results that every one matches and
those that some one matches. Mirrors `Criteria` in `evaluator.rs`, without its
shape, which decides only whether some criterion may match every face of some
die, and for which `DieChains.covers` stands. -/
structure Criteria where
  /-- The results that every criterion matches. -/
  certain : Finset Int
  /-- The results that some criterion matches. -/
  possible : Finset Int
  deriving DecidableEq

/--
The criteria that hold only one criterion, which matches some results, as the
bounds enumerate the criteria to which a predicate may resolve when they are
few enough. Mirrors `Criteria::exact` in `evaluator.rs`.

# Parameters
- `C`: The results that the criterion matches.

# Returns
The criteria.
-/
def Criteria.exact (C : Finset Int) : Criteria :=
  { certain := C, possible := C }

/-- The chains still kept by a rolling record whose dice are all the same
die, or any die that the record may roll, as the bounds of its rerolls and
explodes track them. Mirrors `DieChains` in `evaluator.rs`, whose least dice,
which decide with each criteria whether some criterion may match every face
of some die, `covers` stands for. -/
structure DieChains where
  /-- The faces of the die, any of which an extension or a replacement may
  show. -/
  faces : Finset Int
  /-- Whether some criterion of some criteria may match every face of some
  die. Mirrors `DieChains::may_be_covered`. -/
  covers : Criteria → Bool
  /-- The values that a saturated die of the die may take. -/
  saturating : Finset Int
  /-- The shapes of the chains still kept that no saturated die decides. -/
  live : List ChainShape
  /-- The values that a chain still kept that a saturated die decides may
  take. -/
  saturated : Finset Int

/--
The chains of a die just rolled, each of which holds one die, which may show
any face. Mirrors `DieChains::rolled` in `evaluator.rs`.

# Parameters
- `faces`: The faces of the die.
- `covers`: Whether some criterion of some criteria may match every face of
  some die.
- `saturating`: The values that a saturated die of the die may take.

# Returns
The chains, of which there are none if the die has no faces.
-/
def DieChains.rolled (faces : Finset Int) (covers : Criteria → Bool)
    (saturating : Finset Int) : DieChains :=
  { faces, covers, saturating,
    live := if faces = ∅ then [] else [.new [] faces], saturated := ∅ }

/--
Whether a reroll or explode under some bounds of its limit may leave a chain
as it was, as a limit that may be zero or less does.

# Parameters
- `limit`: The bounds of the limit, or `none` for no limit.
-/
def stays : Option (Int × Int) → Bool
  | some (lo, _) => decide (lo ≤ 0)
  | none => false

/--
Whether a reroll or explode under some bounds of its limit may reroll or
extend a die, as a limit that may be positive, or none, does.

# Parameters
- `limit`: The bounds of the limit, or `none` for no limit.
-/
def proceeds : Option (Int × Int) → Bool
  | some (_, hi) => decide (0 < hi)
  | none => true

/--
Whether the criteria of a reroll or explode without limit certainly match
every face of the die, so that it crosses out or saturates every chain that
may still grow, rolling nothing. Mirrors `DieChains::covered` in
`evaluator.rs`, conjoined with the absence of a limit.

# Parameters
- `D`: The chains.
- `limit`: The bounds of the limit, or `none` for no limit.
- `cr`: The criteria.
-/
def DieChains.blocked (D : DieChains) (limit : Option (Int × Int))
    (cr : Criteria) : Bool :=
  limit.isNone && decide (D.faces ⊆ cr.certain)

/--
What a reroll by some criteria makes of the results that a die may show: each
that every criterion may fail to match, and, if some criterion may match one,
any replacement.

# Parameters
- `cr`: The criteria.
- `replacements`: The results that a replacement may show.
- `R`: The results that the die may show.

# Returns
The results that it may show afterward.
-/
def rerollResults (cr : Criteria) (replacements R : Finset Int) :
    Finset Int :=
  R \ cr.certain ∪ if (R ∩ cr.possible).Nonempty then replacements else ∅

/--
The shapes that a reroll by some criteria leaves of the chains of a die: none
if it crosses them out; otherwise, for each shape, the shape itself, if the
limit may leave it, and the shape whose dice show what the reroll makes of
theirs, if the limit may reroll some die. Mirrors the shapes that
`DieChains::reroll` in `evaluator.rs` makes for each criteria.

# Parameters
- `D`: The chains.
- `limit`: The bounds of the limit, or `none` for no limit.
- `cr`: The criteria.

# Returns
The shapes.
-/
def DieChains.rerollShapes (D : DieChains) (limit : Option (Int × Int))
    (cr : Criteria) : List ChainShape :=
  if D.blocked limit cr then []
  else
    D.live.flatMap fun S =>
      (if stays limit then [S] else []) ++
        if proceeds limit then
          [S.map (rerollResults cr
            (if limit.isSome then D.faces else D.faces \ cr.certain))]
        else []

/--
The chains after a reroll by any of some criteria, uniting the shapes of
each, and whether their shapes were coarsened. A reroll leaves alone every
chain that a saturated die decides. Mirrors the chains that `DieChains::reroll`
in `evaluator.rs` answers.

# Parameters
- `D`: The chains.
- `criteria`: The criteria, each of which the predicate may resolve to, or one
  that stands for every one.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The chains, and whether their shapes were coarsened.
-/
def DieChains.reroll (D : DieChains) (criteria : List Criteria)
    (limit : Option (Int × Int)) : DieChains × Bool :=
  let united := uniteShapes (criteria.flatMap (D.rerollShapes limit))
  ({ D with live := united.1 }, united.2)

/--
Whether some criterion of some criteria may match the last die of a chain
still kept that no saturated die decides. Mirrors `DieChains::may_extend` in
`evaluator.rs`.

# Parameters
- `D`: The chains.
- `cr`: The criteria.
-/
def DieChains.mayExtend (D : DieChains) (cr : Criteria) : Bool :=
  D.live.any fun S => decide (S.last ∩ cr.possible).Nonempty

/--
The shape of a chain of a shape whose last die an explode extends: the runs of
the shape, then a run of the last die, then a run of middle dice, and a new
last die.

# Parameters
- `S`: The shape.
- `start`: The results that the last die of the shape, which the explode
  matches, may show.
- `middles`: The results that each middle die may show.
- `least`, `greatest`: The fewest and most middle dice.
- `last`: The results that the new last die may show.

# Returns
The grown shape.
-/
def ChainShape.grow (S : ChainShape) (start middles : Finset Int)
    (least : Nat) (greatest : ℕ∞) (last : Finset Int) : ChainShape :=
  .new (S.middles ++ [{ results := start, least := 1, greatest := 1 },
    { results := middles, least, greatest }]) last

/--
The shapes that an explode by some criteria leaves of one shape of the chains
of a die: the shape itself, if the limit may leave it; and, if the limit may
extend a die, the shape of the chains whose last die it does not match, and
the shapes of those that it grows. Under a limit `L`, a chain grows by a last
die that it does not match after `0` to `L - 2` middle dice, or by any face
after `L - 1`, which spends the limit; without limit, by a last die that it
does not match after any number. Mirrors the shapes that `DieChains::explode`
in `evaluator.rs` makes of each shape for each criteria.

# Parameters
- `D`: The chains.
- `limit`: The bounds of the limit, or `none` for no limit.
- `cr`: The criteria.
- `S`: The shape.

# Returns
The shapes.
-/
def DieChains.explodeShape (D : DieChains) (limit : Option (Int × Int))
    (cr : Criteria) (S : ChainShape) : List ChainShape :=
  let middles := D.faces ∩ cr.possible
  let ends := D.faces \ cr.certain
  let rest := S.last \ cr.certain
  let start := S.last ∩ cr.possible
  (if stays limit then [S] else []) ++
    if proceeds limit then
      (if rest.Nonempty then [.new S.middles rest] else []) ++
        if start.Nonempty then
          match limit with
          | none => [S.grow start middles 0 ⊤ ends]
          | some (lo, hi) =>
            (if 2 ≤ hi.toNat ∧ ends.Nonempty then
              [S.grow start middles 0 (hi.toNat - 2 : ℕ) ends] else []) ++
              [S.grow start middles (Max.max lo.toNat 1 - 1)
                (hi.toNat - 1 : ℕ) D.faces]
        else []
    else []

/--
The shapes that an explode by some criteria leaves of the chains of a die:
none if it saturates them; otherwise, the shapes that it leaves of each.

# Parameters
- `D`: The chains.
- `limit`: The bounds of the limit, or `none` for no limit.
- `cr`: The criteria.

# Returns
The shapes.
-/
def DieChains.explodeShapes (D : DieChains) (limit : Option (Int × Int))
    (cr : Criteria) : List ChainShape :=
  if D.blocked limit cr then [] else D.live.flatMap (D.explodeShape limit cr)

/--
The values that an explode by some criteria may saturate a chain at: those
of a saturated die of the die, if the explode has no limit, some criterion
may match the last die of some chain, and some may match every face of some
die.

# Parameters
- `D`: The chains.
- `limit`: The bounds of the limit, or `none` for no limit.
- `cr`: The criteria.

# Returns
The values.
-/
def DieChains.saturates (D : DieChains) (limit : Option (Int × Int))
    (cr : Criteria) : Finset Int :=
  if limit.isNone ∧ D.mayExtend cr ∧ D.covers cr then D.saturating else ∅

/--
The chains after an explode by any of some criteria, uniting the shapes of
each, and whether their shapes were coarsened. An explode leaves alone every
chain that a saturated die decides. Mirrors the chains that
`DieChains::explode` in `evaluator.rs` answers.

# Parameters
- `D`: The chains.
- `criteria`: The criteria, each of which the predicate may resolve to, or one
  that stands for every one.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The chains, and whether their shapes were coarsened.
-/
def DieChains.explode (D : DieChains) (criteria : List Criteria)
    (limit : Option (Int × Int)) : DieChains × Bool :=
  let united := uniteShapes (criteria.flatMap (D.explodeShapes limit))
  ({ D with
      live := united.1
      saturated := D.saturated ∪ unionAll (criteria.map (D.saturates limit)) },
    united.2)

/--
The totals that a chain still kept may show: the values of the saturated
chains, and the totals of each shape, between its extremes. Mirrors
`DieChains::totals` in `evaluator.rs`. They hold the total of every chain
that the chains cover (`DieChains.total_mem_totals`).

# Parameters
- `D`: The chains.

# Returns
The totals.
-/
def DieChains.totals (D : DieChains) : Finset Int :=
  D.live.foldl (fun acc S => acc ∪ Finset.Icc S.totals.1 S.totals.2)
    D.saturated

/--
Whether some chains cover a chain: a chain that may still grow has some
shape of theirs, and a saturated chain takes one of their saturated values.

# Parameters
- `D`: The chains.
- `ch`: The chain.
-/
def DieChains.Covers (D : DieChains) : Chain → Prop
  | .live body last => ∃ S ∈ D.live, S.Has (.live body last)
  | .saturated v => v ∈ D.saturated

/--
Whether some chains stand for a die: they admit every face of the die, and its
saturated value.

# Parameters
- `D`: The chains.
- `d`: The die.
-/
def DieChains.Represents (D : DieChains) (d : Die) : Prop :=
  (∀ y ∈ d.faces, y ∈ D.faces) ∧ d.saturatedValue ∈ D.saturating

/-- Whether some criteria stand for a criterion of a reroll or explode of the
chains of a die: some criterion matches every face of the die that it does,
it matches every face that every criterion does, and the chains find that
some criterion may match every face if it does, by `covers`, which stands for
`DieChains.covers`. -/
structure Admits (covers : Criteria → Bool) (d : Die) (cr : Criteria)
    (c : Criterion) : Prop where
  /-- Some criterion matches every face that `c` does. -/
  possible : ∀ y ∈ d.faces, c.matches y = true → y ∈ cr.possible
  /-- `c` matches every face that every criterion does. -/
  certain : ∀ y ∈ d.faces, y ∈ cr.certain → c.matches y = true
  /-- The chains find that some criterion may match every face, if `c`
  does. -/
  covers : d.facesWithin c = true → covers cr = true

/--
Whether a chain is one that a record of a die may keep: it is normal, and,
unless it has saturated, it holds only faces of the die.

# Parameters
- `d`: The die.
- `ch`: The chain.
-/
def Fine (d : Die) (ch : Chain) : Prop :=
  NormalChain ch ∧ (ch.saturates = false → ∀ x ∈ ch.dice, x ∈ d.faces)

/-- The shape of a chain depends only on its dice, not their order. -/
theorem ChainShape.has_perm {S : ChainShape} {body body' : List Int}
    {last : Int} (h : S.Has (.live body last)) (hp : body.Perm body') :
    S.Has (.live body' last) := by
  obtain ⟨hl, parts, hpa, hb⟩ := h
  exact ⟨hl, parts, hpa, hp.symm.trans hb⟩

/-- A reroll that rerolls no die of a chain leaves its dice. -/
private theorem forall₂_redraw_stays {rerolled : Int → Bool}
    {P : Int → Prop} :
    ∀ {xs ys : List Int}, (∀ x ∈ xs, rerolled x = false) →
      List.Forall₂ (Redraw rerolled P) xs ys → ys = xs
  | _, _, _, .nil => rfl
  | x :: _, _, h, .cons hr hl => by
    have hx := h x List.mem_cons_self
    simp only [Redraw, hx, Bool.false_eq_true, ↓reduceIte] at hr
    rw [hr, forall₂_redraw_stays (fun z hz => h z (List.mem_cons_of_mem _ hz))
      hl]

/-- A chain that a reroll leaves of a chain of a shape, each of whose dice it
redraws as what some function makes of any results that the die showed, has
the mapped shape. -/
private theorem ChainShape.has_rerolled {S : ChainShape}
    {f : Finset Int → Finset Int} {rerolled : Int → Bool} {P : Int → Prop}
    {body : List Int} {last : Int} {ch' : Chain}
    (hc : S.Has (.live body last))
    (hr : Rerolled rerolled P (.live body last) ch')
    (hf : ∀ x ∈ last :: body, ∀ y, Redraw rerolled P x y →
      ∀ R : Finset Int, x ∈ R → y ∈ f R) :
    (S.map f).Has ch' := by
  obtain ⟨ys, y, hys, hy, rfl⟩ := hr
  exact ChainShape.has_perm (ChainShape.map_has hc (forall₂_imp_mem
    (fun x hx b h => hf x (List.mem_cons_of_mem _ hx) b h) hys)
    (hf last List.mem_cons_self y hy)) (List.mergeSort_perm _ _).symm

/-- A chain of a shape, grown by its last die, middle dice, and a new last
die, has the grown shape. -/
private theorem ChainShape.has_grow {S : ChainShape} {body : List Int}
    {last : Int} {start middles E : Finset Int} {least : Nat}
    {greatest : ℕ∞} {t : List Int × Int} (hc : S.Has (.live body last))
    (hs : last ∈ start) (hm : ∀ x ∈ t.1, x ∈ middles)
    (hl : least ≤ t.1.length) (hg : (t.1.length : ℕ∞) ≤ greatest)
    (he : t.2 ∈ E) :
    (S.grow start middles least greatest E).Has
      (growChain (.live body last) t) := by
  obtain ⟨-, parts, hp, hb⟩ := hc
  refine ChainShape.has_new ⟨he, parts ++ [[last], t.1],
    List.rel_append hp (.cons ⟨le_rfl, by simp, by simpa using hs⟩
      (.cons ⟨hl, hg, hm⟩ .nil)), ?_⟩
  refine (List.mergeSort_perm _ _).trans ?_
  simp only [Chain.dice, List.flatten_append, List.flatten_cons,
    List.flatten_nil, List.append_nil]
  refine ((List.mergeSort_perm _ _).append_right _).trans ?_
  refine ((hb.cons last).append_right t.1).trans ?_
  simpa using List.perm_middle.symm

/-- Every chain still kept that a saturated die decides stays as it was
through any reroll. -/
private theorem rerollBecomes_saturated {d : Die} {c : Criterion}
    {l : Option Nat} {v : Int} {ch' : Chain}
    (h : RerollBecomes d c l (.saturated v) ch') : ch' = .saturated v := by
  cases l with
  | some n => exact h
  | none =>
    by_cases hw : d.facesWithin c = true
    · simp only [RerollBecomes, hw, ↓reduceIte] at h
      exact h.2
    · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte] at h
      exact h

/-- Some chains cover every chain of one of their shapes. -/
private theorem DieChains.covers_of_has {D : DieChains} {S : ChainShape}
    (hS : S ∈ D.live) {ch : Chain} (h : S.Has ch) : D.Covers ch := by
  cases ch with
  | live body last => exact ⟨S, hS, h⟩
  | saturated v => exact h.elim

/-- The chains after a reroll cover every chain of a shape that the reroll
makes for one of its criteria. -/
private theorem DieChains.covers_reroll {D : DieChains}
    {criteria : List Criteria} {cr : Criteria} {limit : Option (Int × Int)}
    (hcr : cr ∈ criteria) {S : ChainShape} (hS : S ∈ D.rerollShapes limit cr)
    {ch : Chain} (h : S.Has ch) : (D.reroll criteria limit).1.Covers ch := by
  obtain ⟨U, hU, h⟩ := uniteShapes_has (List.mem_flatMap.mpr ⟨cr, hcr, hS⟩) h
  exact DieChains.covers_of_has hU h

/-- The chains after an explode cover every chain of a shape that the explode
makes for one of its criteria. -/
private theorem DieChains.covers_explode {D : DieChains}
    {criteria : List Criteria} {cr : Criteria} {limit : Option (Int × Int)}
    (hcr : cr ∈ criteria) {S : ChainShape}
    (hS : S ∈ D.explodeShapes limit cr) {ch : Chain} (h : S.Has ch) :
    (D.explode criteria limit).1.Covers ch := by
  obtain ⟨U, hU, h⟩ := uniteShapes_has (List.mem_flatMap.mpr ⟨cr, hcr, hS⟩) h
  exact DieChains.covers_of_has hU h

/-- A die that a reroll redraws, or leaves, shows afterward what the reroll
makes of any results that it showed. -/
private theorem mem_rerollResults_of {cr : Criteria}
    {replacements R : Finset Int} {x y : Int} (hx : x ∈ R)
    (h : (x ∈ cr.possible ∧ y ∈ replacements) ∨ (y = x ∧ x ∉ cr.certain)) :
    y ∈ rerollResults cr replacements R := by
  unfold rerollResults
  rcases h with ⟨hp, hy⟩ | ⟨rfl, hc⟩
  · have hn : (R ∩ cr.possible).Nonempty := ⟨x, Finset.mem_inter.mpr ⟨hx, hp⟩⟩
    simp only [hn, ↓reduceIte]
    exact Finset.mem_union_right _ hy
  · exact Finset.mem_union_left _ (Finset.mem_sdiff.mpr ⟨hx, hc⟩)

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

/-- The bounds of a limit admit a limit of `some`, only from `some` bounds. -/
private theorem limits_some {limit : Option (Int × Int)} {n : Nat}
    (h : Limits limit (some n)) :
    ∃ lo hi k, limit = some (lo, hi) ∧ lo ≤ k ∧ k ≤ hi ∧ n = k.toNat := by
  cases limit with
  | none => simp [Limits] at h
  | some b =>
    obtain ⟨k, hlo, hhi, hk⟩ := h
    exact ⟨b.1, b.2, k, rfl, hlo, hhi, Option.some.inj hk⟩

/-- The bounds of a limit admit no limit only without bounds. -/
private theorem limits_none {limit : Option (Int × Int)}
    (h : Limits limit none) : limit = none := by
  cases limit with
  | none => rfl
  | some b => simp [Limits] at h

/-- Criteria that stand for a criterion that does not match every face of a
die do not certainly match every face of chains that stand for it. -/
private theorem not_blocked {D : DieChains} {d : Die} {cr : Criteria}
    {c : Criterion} (hD : D.Represents d) (ha : Admits D.covers d cr c)
    (hw : ¬d.facesWithin c = true) (limit : Option (Int × Int)) :
    D.blocked limit cr = false := by
  simp only [DieChains.blocked, Bool.and_eq_false_iff, decide_eq_false_iff_not]
  refine .inr fun hsub => hw ?_
  exact List.all_eq_true.mpr fun y hy => ha.certain y hy (hsub (hD.1 y hy))

/--
The chains after a reroll cover every chain that it may make of a chain that
the chains before covered, if they stand for the die, some criteria of the
reroll stand for its criterion, and the bounds of the limit admit its limit.

# Parameters
- `D`: The chains.
- `d`: The die.
- `criteria`: The criteria of the reroll.
- `cr`: The criteria that stand for the criterion.
- `c`: The criterion.
- `limit`: The bounds of the limit.
- `l`: The limit.
- `ch`, `ch'`: The chain, and what the reroll makes of it.

# Hypotheses
- `hD`: The chains stand for the die.
- `hcr`: The criteria are among those of the reroll.
- `ha`: The criteria stand for the criterion.
- `hL`: The bounds admit the limit.
- `hf`: The chain is one that a record of the die may keep.
- `hc`: The chains cover the chain.
- `hb`: The reroll may make `ch'` of the chain.
-/
theorem DieChains.reroll_covers {D : DieChains} {d : Die}
    {criteria : List Criteria} {cr : Criteria} {c : Criterion}
    {limit : Option (Int × Int)} {l : Option Nat} {ch ch' : Chain}
    (hD : D.Represents d) (hcr : cr ∈ criteria) (ha : Admits D.covers d cr c)
    (hL : Limits limit l) (hf : Fine d ch) (hc : D.Covers ch)
    (hb : RerollBecomes d c l ch ch') :
    (D.reroll criteria limit).1.Covers ch' := by
  cases ch with
  | saturated v =>
    rw [rerollBecomes_saturated hb]
    exact hc
  | live body last =>
    obtain ⟨S, hS, hSc⟩ := hc
    have hdice : ∀ x ∈ last :: body, x ∈ d.faces := fun x hx =>
      hf.2 rfl x ((List.mergeSort_perm _ _).mem_iff.mpr hx)
    -- A die that the criterion does not match no criterion certainly does.
    have hcert : ∀ x ∈ last :: body, c.matches x = false →
        x ∉ cr.certain := fun x hx hm hc' => by
      simp [ha.certain x (hdice x hx) hc'] at hm
    cases l with
    | some n =>
      obtain ⟨lo, hi, k, rfl, hlo, hhi, rfl⟩ := limits_some hL
      have hnb : D.blocked (some (lo, hi)) cr = false := by
        simp [DieChains.blocked]
      by_cases hn : k.toNat = 0
      · -- The reroll rerolls nothing, and the limit may leave the shape.
        obtain ⟨ys, y, hys, hy, rfl⟩ := hb
        have hy' : y = last := by simpa [Redraw, hn] using hy
        have hys' := forall₂_redraw_stays (fun x _ => by simp [hn]) hys
        subst hy' hys'
        have hst : stays (some (lo, hi)) = true := by
          simp only [stays, decide_eq_true_eq]
          omega
        refine D.covers_reroll hcr (S := S) ?_
          (ChainShape.has_perm hSc (List.mergeSort_perm _ _).symm)
        simp only [DieChains.rerollShapes, hnb, Bool.false_eq_true, ↓reduceIte,
          List.mem_flatMap]
        exact ⟨S, hS, List.mem_append_left _ (by simp [hst])⟩
      · have hp : proceeds (some (lo, hi)) = true := by
          simp only [proceeds, decide_eq_true_eq]
          omega
        refine D.covers_reroll hcr
          (S := S.map (rerollResults cr D.faces)) ?_
          (ChainShape.has_rerolled hSc hb ?_)
        · simp only [DieChains.rerollShapes, hnb, Bool.false_eq_true,
            ↓reduceIte, List.mem_flatMap, Option.isSome_some]
          exact ⟨S, hS, List.mem_append_right _ (by simp [hp])⟩
        · intro x hx y hy R hR
          unfold Redraw at hy
          split at hy
          · rename_i hm
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
            exact mem_rerollResults_of hR
              (.inl ⟨ha.possible x (hdice x hx) hm.2, hD.1 y hy⟩)
          · rename_i hm
            simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at hm
            exact mem_rerollResults_of hR (.inr ⟨hy, hcert x hx
              (by simpa using hm (by omega))⟩)
    | none =>
      rw [limits_none hL]
      by_cases hw : d.facesWithin c = true
      · simp only [RerollBecomes, hw, ↓reduceIte] at hb
        exact absurd hb.1 (by simp [Chain.saturates])
      · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte] at hb
        have hnb := not_blocked hD ha hw none
        refine D.covers_reroll hcr
          (S := S.map (rerollResults cr (D.faces \ cr.certain))) ?_
          (ChainShape.has_rerolled hSc hb ?_)
        · simp only [DieChains.rerollShapes, hnb, Bool.false_eq_true,
            ↓reduceIte, List.mem_flatMap, Option.isSome_none]
          exact ⟨S, hS, List.mem_append_right _ (by simp [proceeds])⟩
        · intro x hx y hy R hR
          unfold Redraw at hy
          split at hy
          · rename_i hm
            refine mem_rerollResults_of hR (.inl ⟨ha.possible x (hdice x hx) hm,
              Finset.mem_sdiff.mpr ⟨hD.1 y hy.1, fun h => ?_⟩⟩)
            simp [ha.certain y hy.1 h] at hy
          · rename_i hm
            exact mem_rerollResults_of hR (.inr ⟨hy, hcert x hx
              (by simpa using hm)⟩)

/-- Every chain still kept that a saturated die decides stays as it was
through any explode. -/
private theorem becomes_saturated {d : Die} {c : Criterion}
    {l : Option Nat} {v : Int} {ch' : Chain}
    (h : Becomes d c l (.saturated v) ch') : ch' = .saturated v := by
  have hg : (Chain.saturated v).grows c = false := rfl
  cases l with
  | some n => simpa only [Becomes, hg, Bool.false_eq_true, and_false,
      ↓reduceIte] using h
  | none =>
    by_cases hw : d.facesWithin c = true
    · simpa only [Becomes, hw, ↓reduceIte, Chain.saturate] using h
    · simpa only [Becomes, hw, hg, Bool.false_eq_true, ↓reduceIte] using h

/--
The chains after an explode cover every chain that it may make of a chain
that the chains before covered, if they stand for the die, some criteria of
the explode stand for its criterion, and the bounds of the limit admit its
limit.

# Parameters
- `D`: The chains.
- `d`: The die.
- `criteria`: The criteria of the explode.
- `cr`: The criteria that stand for the criterion.
- `c`: The criterion.
- `limit`: The bounds of the limit.
- `l`: The limit.
- `ch`, `ch'`: The chain, and what the explode makes of it.

# Hypotheses
- `hD`: The chains stand for the die.
- `hcr`: The criteria are among those of the explode.
- `ha`: The criteria stand for the criterion.
- `hL`: The bounds admit the limit.
- `hf`: The chain is one that a record of the die may keep.
- `hc`: The chains cover the chain.
- `hb`: The explode may make `ch'` of the chain.
-/
theorem DieChains.explode_covers {D : DieChains} {d : Die}
    {criteria : List Criteria} {cr : Criteria} {c : Criterion}
    {limit : Option (Int × Int)} {l : Option Nat} {ch ch' : Chain}
    (hD : D.Represents d) (hcr : cr ∈ criteria) (ha : Admits D.covers d cr c)
    (hL : Limits limit l) (hf : Fine d ch) (hc : D.Covers ch)
    (hb : Becomes d c l ch ch') :
    (D.explode criteria limit).1.Covers ch' := by
  cases ch with
  | saturated v =>
    rw [becomes_saturated hb]
    exact Finset.mem_union_left _ hc
  | live body last =>
    obtain ⟨S, hS, hSc⟩ := hc
    have hdice : ∀ x ∈ last :: body, x ∈ d.faces := fun x hx =>
      hf.2 rfl x ((List.mergeSort_perm _ _).mem_iff.mpr hx)
    have hlast := hdice last List.mem_cons_self
    have hcert : c.matches last = false → last ∉ cr.certain := fun hm hc' => by
      simp [ha.certain last hlast hc'] at hm
    -- The shape of the chains whose last die the explode does not match.
    have hrest : c.matches last = false →
        (ChainShape.new S.middles (S.last \ cr.certain)).Has
          (.live body last) := fun hm =>
      ChainShape.has_new ⟨Finset.mem_sdiff.mpr ⟨hSc.1, hcert hm⟩, hSc.2⟩
    have hrestn : c.matches last = false → (S.last \ cr.certain).Nonempty :=
      fun hm => ⟨last, Finset.mem_sdiff.mpr ⟨hSc.1, hcert hm⟩⟩
    have hstart : c.matches last = true → (S.last ∩ cr.possible).Nonempty :=
      fun hm => ⟨last, Finset.mem_inter.mpr ⟨hSc.1, ha.possible last hlast hm⟩⟩
    have hgrow : c.matches last = true → last ∈ S.last ∩ cr.possible :=
      fun hm => Finset.mem_inter.mpr ⟨hSc.1, ha.possible last hlast hm⟩
    -- Each middle die of a tail is a face that some criterion matches.
    have hmid : ∀ {xs : List Int}, (∀ x ∈ xs, x ∈ d.faces ∧
        c.matches x = true) → ∀ x ∈ xs, x ∈ D.faces ∩ cr.possible :=
      fun h x hx => Finset.mem_inter.mpr ⟨hD.1 x (h x hx).1,
        ha.possible x (h x hx).1 (h x hx).2⟩
    -- Each last die that the explode does not match ends a chain.
    have hend : ∀ {y : Int}, y ∈ d.faces → c.matches y = false →
        y ∈ D.faces \ cr.certain := fun hy hm =>
      Finset.mem_sdiff.mpr ⟨hD.1 _ hy, fun h => by
        simp [ha.certain _ hy h] at hm⟩
    cases l with
    | some n =>
      obtain ⟨lo, hi, k, rfl, hlo, hhi, rfl⟩ := limits_some hL
      have hnb : D.blocked (some (lo, hi)) cr = false := by
        simp [DieChains.blocked]
      -- Every shape of the explode of `S` for these criteria.
      have hmem : ∀ S' ∈ D.explodeShape (some (lo, hi)) cr S,
          S' ∈ D.explodeShapes (some (lo, hi)) cr := fun S' h => by
        simp only [DieChains.explodeShapes, hnb, Bool.false_eq_true,
          ↓reduceIte, List.mem_flatMap]
        exact ⟨S, hS, h⟩
      by_cases hg : 0 < k.toNat ∧ (Chain.live body last).grows c = true
      · simp only [Becomes] at hb
        rw [ite_eq_left_of_eq_true _ _ (eq_true hg)] at hb
        obtain ⟨hk, hgr⟩ := hg
        have hm : c.matches last = true := by
          simpa [Chain.grows, Chain.saturates, Chain.last] using hgr
        obtain ⟨t, ⟨-, hdm, hy, hlen, htail⟩, rfl⟩ := hb
        have hp : proceeds (some (lo, hi)) = true := by
          simp only [proceeds, decide_eq_true_eq]
          omega
        by_cases hshort : t.1.length + 1 < k.toNat
        · have hyE := hend hy (htail hshort)
          refine D.covers_explode hcr (hmem _ ?_)
            (ChainShape.has_grow hSc (hgrow hm) (hmid hdm) (Nat.zero_le _)
              (least := 0) (greatest := ((hi.toNat - 2 : ℕ) : ℕ∞))
              (Nat.cast_le.mpr (by omega)) hyE)
          have h2 : 2 ≤ hi.toNat := by omega
          have hne : (D.faces \ cr.certain).Nonempty := ⟨_, hyE⟩
          simp [DieChains.explodeShape, hp, hstart hm, h2, hne]
        · refine D.covers_explode hcr (hmem _ ?_)
            (ChainShape.has_grow hSc (hgrow hm) (hmid hdm)
              (least := Max.max lo.toNat 1 - 1)
              (greatest := ((hi.toNat - 1 : ℕ) : ℕ∞)) (by omega)
              (Nat.cast_le.mpr (by omega)) (hD.1 _ hy))
          simp [DieChains.explodeShape, hp, hstart hm]
      · simp only [Becomes] at hb
        rw [ite_eq_right_of_eq_false _ _ (eq_false hg)] at hb
        subst hb
        by_cases hn : k.toNat = 0
        · have hst : stays (some (lo, hi)) = true := by
            simp only [stays, decide_eq_true_eq]
            omega
          refine D.covers_explode hcr (hmem S ?_) hSc
          simp [DieChains.explodeShape, hst]
        · have hm : c.matches last = false := by
            simpa [Chain.grows, Chain.saturates, Chain.last,
              Nat.pos_of_ne_zero hn] using hg
          have hp : proceeds (some (lo, hi)) = true := by
            simp only [proceeds, decide_eq_true_eq]
            omega
          refine D.covers_explode hcr (hmem _ ?_) (hrest hm)
          simp [DieChains.explodeShape, hp, hrestn hm]
    | none =>
      rw [limits_none hL]
      by_cases hw : d.facesWithin c = true
      · simp only [Becomes, hw, ↓reduceIte] at hb
        subst hb
        refine Finset.mem_union_right _ (mem_unionAll.mpr
          ⟨_, List.mem_map_of_mem hcr, ?_⟩)
        have hm : c.matches last = true := List.all_eq_true.mp hw last hlast
        have hext : D.mayExtend cr = true :=
          List.any_eq_true.mpr ⟨S, hS, decide_eq_true (hstart hm)⟩
        simp [DieChains.saturates, hext, ha.covers hw, hD.2]
      · simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte] at hb
        have hnb := not_blocked hD ha hw none
        have hmem : ∀ S' ∈ D.explodeShape none cr S,
            S' ∈ D.explodeShapes none cr := fun S' h => by
          simp only [DieChains.explodeShapes, hnb, Bool.false_eq_true,
            ↓reduceIte, List.mem_flatMap]
          exact ⟨S, hS, h⟩
        by_cases hgr : (Chain.live body last).grows c = true
        · simp only [hgr, ↓reduceIte] at hb
          have hm : c.matches last = true := by
            simpa [Chain.grows, Chain.saturates, Chain.last] using hgr
          obtain ⟨t, ⟨-, hdm, hy, htail⟩, rfl⟩ := hb
          refine D.covers_explode hcr (hmem _ ?_)
            (ChainShape.has_grow hSc (hgrow hm) (hmid hdm) (Nat.zero_le _)
              (least := 0) (greatest := ⊤) le_top (hend hy htail))
          simp [DieChains.explodeShape, proceeds, hstart hm]
        · simp only [hgr, Bool.false_eq_true, ↓reduceIte] at hb
          subst hb
          have hm : c.matches last = false := by
            simpa [Chain.grows, Chain.saturates, Chain.last] using hgr
          refine D.covers_explode hcr (hmem _ ?_) (hrest hm)
          simp [DieChains.explodeShape, proceeds, stays, hrestn hm]

/-- Every value on the right of a relation between lists that only asks a
property of it has the property. -/
private theorem forall₂_mem_right {P : Int → Prop} :
    ∀ {xs ys : List Int}, List.Forall₂ (fun _ z => P z) xs ys → ∀ z ∈ ys, P z
  | _, _, .nil => by simp
  | _, _, .cons h hl => fun z hz => by
    rcases List.mem_cons.mp hz with rfl | hz
    · exact h
    · exact forall₂_mem_right hl z hz

/-- A chain that a reroll leaves of one that a record of a die may keep, by
replacements that are faces, is one that a record of the die may keep. -/
private theorem Fine.rerolled {d : Die} {rerolled : Int → Bool}
    {P : Int → Prop} {ch ch' : Chain}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hP : ∀ y, P y → y ∈ d.faces) (hf : Fine d ch)
    (h : Rerolled rerolled P ch ch') : Fine d ch' := by
  cases ch with
  | saturated v =>
    have : ch' = .saturated v := h
    exact this ▸ hf
  | live body last =>
    obtain ⟨ys, y, hys, hy, rfl⟩ := h
    have hdice : ∀ x ∈ last :: body, x ∈ d.faces := fun x hx =>
      hf.2 rfl x ((List.mergeSort_perm _ _).mem_iff.mpr hx)
    have hdraw : ∀ x ∈ last :: body, ∀ z, Redraw rerolled P x z →
        z ∈ d.faces := fun x hx z hz => by
      unfold Redraw at hz
      split at hz
      · exact hP z hz
      · exact hz ▸ hdice x hx
    have hys' : ∀ z ∈ ys, z ∈ d.faces := forall₂_mem_right
      (forall₂_imp_mem (S := fun _ z => z ∈ d.faces)
        (fun x hx z hz => hdraw x (List.mem_cons_of_mem _ hx) z hz) hys)
    have hall : ∀ z ∈ y :: ys.mergeSort, z ∈ d.faces := fun z hz => by
      rcases List.mem_cons.mp hz with rfl | hz
      · exact hdraw last List.mem_cons_self z hy
      · exact hys' z ((List.mergeSort_perm _ _).mem_iff.mp hz)
    exact ⟨⟨pairwise_mergeSort ys, fun z hz => hF z (hall z hz)⟩,
      fun _ z hz => hall z ((List.mergeSort_perm _ _).mem_iff.mp hz)⟩

/--
A chain that a reroll makes of one that a record of a die may keep is one
that a record of the die may keep, if the faces lie within the `i32` range.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hf`: The chain is one that a record of the die may keep.
- `h`: The reroll may make `ch'` of it.
-/
theorem Fine.rerollBecomes {d : Die} {c : Criterion} {l : Option Nat}
    {ch ch' : Chain} (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hf : Fine d ch) (h : RerollBecomes d c l ch ch') : Fine d ch' := by
  cases l with
  | some n => exact Fine.rerolled hF (fun _ h => h) hf h
  | none =>
    by_cases hw : d.facesWithin c = true
    · simp only [RerollBecomes, hw, ↓reduceIte] at h
      exact h.2 ▸ hf
    · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte] at h
      exact Fine.rerolled hF (fun _ h => h.1) hf h

/--
A chain that an explode makes of one that a record of a die may keep is one
that a record of the die may keep, if the faces lie within the `i32` range.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hf`: The chain is one that a record of the die may keep.
- `h`: The explode may make `ch'` of it.
-/
theorem Fine.becomes {d : Die} {c : Criterion} {l : Option Nat}
    {ch ch' : Chain} (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hf : Fine d ch) (h : Becomes d c l ch ch') : Fine d ch' := by
  -- A chain grown by faces from a chain of faces.
  have hgrown : ∀ t : List Int × Int, ch.saturates = false →
      (∀ x ∈ t.1, x ∈ d.faces) → t.2 ∈ d.faces → Fine d (growChain ch t) :=
    fun t hs ht hy => by
      refine ⟨normalChain_growChain hF (hf.2 hs) ⟨ht, hy⟩, fun _ x hx => ?_⟩
      simp only [growChain, Chain.dice] at hx
      rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
        rfl | hx
      · exact hy
      · rcases List.mem_append.mp ((List.mergeSort_perm _ _).mem_iff.mp hx)
          with hx | hx
        · exact hf.2 hs x hx
        · exact ht x hx
  have hgs : ch.grows c = true → ch.saturates = false := fun hg => by
    simp only [Chain.grows, Bool.and_eq_true, Bool.not_eq_true'] at hg
    exact hg.1
  cases l with
  | some n =>
    by_cases hg : 0 < n ∧ ch.grows c = true
    · simp only [Becomes] at h
      rw [ite_eq_left_of_eq_true _ _ (eq_true hg)] at h
      obtain ⟨t, ⟨-, hdm, hy, -⟩, rfl⟩ := h
      exact hgrown t (hgs hg.2) (fun x hx => (hdm x hx).1) hy
    · simp only [Becomes] at h
      rw [ite_eq_right_of_eq_false _ _ (eq_false hg)] at h
      exact h ▸ hf
  | none =>
    by_cases hw : d.facesWithin c = true
    · simp only [Becomes, hw, ↓reduceIte] at h
      subst h
      cases ch with
      | live => exact ⟨trivial, fun h => by simp [Chain.saturate,
          Chain.saturates] at h⟩
      | saturated v => exact hf
    · simp only [Becomes, hw, Bool.false_eq_true, ↓reduceIte] at h
      by_cases hg : ch.grows c = true
      · simp only [hg, ↓reduceIte] at h
        obtain ⟨t, ⟨-, hdm, hy, -⟩, rfl⟩ := h
        exact hgrown t (hgs hg) (fun x hx => (hdm x hx).1) hy
      · simp only [hg, Bool.false_eq_true, ↓reduceIte] at h
        exact h ▸ hf

/-- Every record that a map of chains into records of a die reaches holds the
die. -/
private theorem die_of_mem_map_ofChains {die : Die} {p : PMF (List Chain)}
    {r : Record} (h : r ∈ (p.map (Record.ofChains die)).support) :
    r.die = die := by
  obtain ⟨cs, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp h
  exact (Record.ofChains_fields die cs).2.2.2

/-- Every record that a reroll of each kept die of a record reaches holds its
die. -/
private theorem die_of_mem_rerollEach {r R : Record} {rerolled : Int → Bool}
    {P : PMF Int} (h : R ∈ (rerollEach r rerolled P).support) :
    R.die = r.die := by
  unfold rerollEach at h
  split at h
  · exact (mem_support_rerollKept h).2.2
  · exact die_of_mem_map_ofChains h

/--
Every record that a reroll reaches holds the record's die.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `R`: The record reached.

# Hypotheses
- `h`: The reroll reaches `R`.
-/
theorem die_of_mem_reroll {r : Record} {c : Criterion} {l : Option Nat}
    {R : Record} (h : R ∈ (reroll r c l).support) : R.die = r.die := by
  cases l with
  | some n => exact die_of_mem_rerollEach h
  | none =>
    simp only [reroll] at h
    split at h
    · split at h
      · rw [PMF.mem_support_pure_iff] at h
        exact h ▸ rfl
      · rw [PMF.mem_support_pure_iff] at h
        exact h ▸ (Record.ofChains_fields _ _).2.2.2
    · exact die_of_mem_rerollEach h

/--
Every record that an explode reaches holds the record's die.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `l`: The limit, or `none` for no limit.
- `R`: The record reached.

# Hypotheses
- `h`: The explode reaches `R`.
-/
theorem die_of_mem_explode {r : Record} {c : Criterion} {l : Option Nat}
    {R : Record} (h : R ∈ (explode r c l).support) : R.die = r.die := by
  cases l with
  | some n => exact die_of_mem_map_ofChains h
  | none =>
    simp only [explode] at h
    split at h
    · rw [PMF.mem_support_pure_iff] at h
      exact h ▸ (Record.ofChains_fields _ _).2.2.2
    · exact die_of_mem_map_ofChains h

/-- An element lies in a window of a list exactly when it lies at some
position within the window. -/
private theorem mem_window_iff {α : Type} {l : List α} {a m : Nat} {x : α} :
    x ∈ (l.drop a).take m ↔
      ∃ i, ∃ h : i < l.length, a ≤ i ∧ i < a + m ∧ l[i] = x := by
  simp only [List.mem_iff_getElem, List.getElem_take, List.getElem_drop,
    List.length_take, List.length_drop, lt_min_iff]
  constructor
  · rintro ⟨i, ⟨h₁, h₂⟩, rfl⟩
    exact ⟨a + i, by omega, by omega, by omega, rfl⟩
  · rintro ⟨i, h, h₁, h₂, rfl⟩
    exact ⟨i - a, ⟨by omega, by omega⟩, by congr 1; omega⟩

/-- A window of a list that drops at least as many at either end, or that
holds nothing, lies within another. -/
private theorem mem_window_of {α : Type} {l : List α} {n a b a' b' : Nat}
    (hw : (a ≤ a' ∧ b ≤ b') ∨ n ≤ a' + b') {x : α}
    (hx : x ∈ (l.drop a').take (n - a' - b')) :
    x ∈ (l.drop a).take (n - a - b) := by
  obtain ⟨i, h, h₁, h₂, rfl⟩ := mem_window_iff.mp hx
  exact mem_window_iff.mpr ⟨i, h, by omega, by omega, rfl⟩

/-- Every chain that a record with the same results and chains keeps, which
drops at least as many results at either end, or keeps none, a record
keeps. -/
private theorem mem_keptChains_of_drops {r r' : Record}
    (hres : r'.results = r.results) (hch : r'.chains = r.chains)
    (hw : (r.lowest ≤ r'.lowest ∧ r.highest ≤ r'.highest) ∨
      r.results.length ≤ r'.lowest + r'.highest)
    {ch : Chain} (h : ch ∈ r'.keptChains) : ch ∈ r.keptChains := by
  unfold Record.keptChains Record.kept Record.sorted at h ⊢
  rw [hres, hch] at h
  split at h
  · rename_i hc
    simp only [hc, ↓reduceIte, List.mem_map] at h ⊢
    obtain ⟨x, hx, rfl⟩ := h
    exact ⟨x, mem_window_of hw hx, rfl⟩
  · rename_i hc
    simp only [hc, Bool.false_eq_true, ↓reduceIte] at ⊢
    exact mem_window_of hw h

/--
Every chain that a drop or keep leaves among the kept chains of a record, the
record kept: a drop or keep only crosses chains out.

# Parameters
- `r`: The record.
- `count`: The count of the drop or keep.
- `ch`: The chain.

# Hypotheses
- `h`: The record after the drop keeps the chain.
-/
theorem mem_keptChains_dropLowest {r : Record} {count : Int} {ch : Chain}
    (h : ch ∈ (r.dropLowest count).keptChains) : ch ∈ r.keptChains := by
  have hl : (r.dropLowest count).lowest =
      Min.min (r.lowest + count.toNat) r.results.length := rfl
  refine mem_keptChains_of_drops (r := r) (r' := r.dropLowest count) rfl rfl
    ?_ h
  have : (r.dropLowest count).highest = r.highest := rfl
  have : (r.dropLowest count).results = r.results := rfl
  omega

/-- Every chain that a drop of the highest results leaves, the record
kept. -/
theorem mem_keptChains_dropHighest {r : Record} {count : Int} {ch : Chain}
    (h : ch ∈ (r.dropHighest count).keptChains) : ch ∈ r.keptChains := by
  have hh : (r.dropHighest count).highest =
      Min.min (r.highest + count.toNat) r.results.length := rfl
  refine mem_keptChains_of_drops (r := r) (r' := r.dropHighest count) rfl rfl
    ?_ h
  have : (r.dropHighest count).lowest = r.lowest := rfl
  omega

/-- Whether some chains track a record of a die: the record holds the die,
and every kept chain is one that a record of the die may keep, and the chains
cover it. -/
structure DieChains.Tracks (D : DieChains) (d : Die) (r : Record) : Prop where
  /-- The record holds the die. -/
  die : r.die = d
  /-- Every kept chain is one that a record of the die may keep. -/
  fine : ∀ ch ∈ r.keptChains, Fine d ch
  /-- The chains cover every kept chain. -/
  covers : ∀ ch ∈ r.keptChains, D.Covers ch

/-- A clause of a roll, which acts on its record after it rolls. -/
inductive Clause where
  /-- A reroll by a criterion, under a limit, or `none` for no limit. -/
  | reroll (c : Criterion) (limit : Option Nat)
  /-- An explode by a criterion, under a limit, or `none` for no limit. -/
  | explode (c : Criterion) (limit : Option Nat)
  /-- A drop of the lowest results. -/
  | dropLowest (count : Int)
  /-- A drop of the highest results. -/
  | dropHighest (count : Int)
  /-- A keep of the lowest results. -/
  | keepLowest (count : Int)
  /-- A keep of the highest results. -/
  | keepHighest (count : Int)

/--
The law of a record after a clause, as `step` in `XdySpec/Semantics.lean`
executes it.

# Parameters
- `cl`: The clause.
- `r`: The record.

# Returns
The law of the record after the clause.
-/
noncomputable def Clause.apply : Clause → Record → PMF Record
  | .reroll c l, r => Spec.reroll r c l
  | .explode c l, r => Spec.explode r c l
  | .dropLowest n, r => PMF.pure (r.dropLowest n)
  | .dropHighest n, r => PMF.pure (r.dropHighest n)
  | .keepLowest n, r => PMF.pure (r.keepLowest n)
  | .keepHighest n, r => PMF.pure (r.keepHighest n)

/--
Whether some clauses may lead from one record to another, each from the
record that the last left.

# Parameters
- `r`: The record before the clauses.
- `cls`: The clauses.
- `r'`: The record after them.
-/
def Reaches : Record → List Clause → Record → Prop
  | r, [], r' => r' = r
  | r, cl :: cls, r'' => ∃ r' ∈ (cl.apply r).support, Reaches r' cls r''

/-- The bounds of a clause, as the bounds evaluator tracks the chains of a
record through it. -/
inductive ClauseBounds where
  /-- A reroll by any of some criteria, under some bounds of its limit, or
  `none` for no limit. -/
  | reroll (criteria : List Criteria) (limit : Option (Int × Int))
  /-- An explode by any of some criteria, under some bounds of its limit, or
  `none` for no limit. -/
  | explode (criteria : List Criteria) (limit : Option (Int × Int))
  /-- A drop or keep, which leaves the shapes of the chains alone. -/
  | drop

/--
The chains after the bounds of a clause, and whether their shapes were
coarsened.

# Parameters
- `b`: The bounds of the clause.
- `D`: The chains.

# Returns
The chains, and whether their shapes were coarsened.
-/
def ClauseBounds.apply : ClauseBounds → DieChains → DieChains × Bool
  | .reroll criteria limit, D => D.reroll criteria limit
  | .explode criteria limit, D => D.explode criteria limit
  | .drop, D => (D, false)

/--
Whether the bounds of a clause stand for a clause of a record of a die: a
reroll or explode by some criteria that stand for its criterion, under bounds
that admit its limit, or a drop or keep.

# Parameters
- `covers`: Whether some criterion of some criteria may match every face of
  some die, as the chains find it.
- `d`: The die.
- `b`: The bounds of the clause.
- `cl`: The clause.
-/
def ClauseBounds.Admits (covers : Criteria → Bool) (d : Die) :
    ClauseBounds → Clause → Prop
  | .reroll criteria limit, .reroll c l =>
    (∃ cr ∈ criteria, Bounds.Admits covers d cr c) ∧ Limits limit l
  | .explode criteria limit, .explode c l =>
    (∃ cr ∈ criteria, Bounds.Admits covers d cr c) ∧ Limits limit l
  | .drop, .dropLowest _ | .drop, .dropHighest _ | .drop, .keepLowest _
  | .drop, .keepHighest _ => True
  | _, _ => False

/--
The chains after the bounds of some clauses, each from the chains that the
last left, and whether their shapes were ever coarsened.

# Parameters
- `D`: The chains before the clauses.
- `bs`: The bounds of the clauses.

# Returns
The chains, and whether their shapes were ever coarsened.
-/
def runBounds (D : DieChains) (bs : List ClauseBounds) : DieChains × Bool :=
  bs.foldl (fun acc b => let next := b.apply acc.1; (next.1, acc.2 || next.2))
    (D, false)

/-- The bounds of a clause leave the faces, the coverage, and the saturating
values of the chains alone. -/
theorem ClauseBounds.apply_fixed (b : ClauseBounds) (D : DieChains) :
    (b.apply D).1.faces = D.faces ∧ (b.apply D).1.covers = D.covers ∧
      (b.apply D).1.saturating = D.saturating := by
  cases b <;> exact ⟨rfl, rfl, rfl⟩

/--
The bounds of a clause that stand for it track every record that it may
leave of a record that the chains tracked.

# Parameters
- `D`: The chains.
- `d`: The die.
- `b`: The bounds of the clause.
- `cl`: The clause.
- `r`, `r'`: The record, and a record that the clause may leave of it.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hD`: The chains stand for the die.
- `hb`: The bounds stand for the clause.
- `ht`: The chains track the record.
- `hr`: The clause may leave `r'`.
-/
theorem ClauseBounds.tracks {D : DieChains} {d : Die} {b : ClauseBounds}
    {cl : Clause} {r r' : Record}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) (hD : D.Represents d)
    (hb : b.Admits D.covers d cl) (ht : D.Tracks d r)
    (hr : r' ∈ (cl.apply r).support) : (b.apply D).1.Tracks d r' := by
  have hk : ∀ ch ∈ r.keptChains, NormalChain ch ∧
      (ch.saturates = false → ∀ x ∈ ch.dice, x ∈ r.die.faces) :=
    fun ch hch => ht.die ▸ ht.fine ch hch
  have hF' : ∀ x ∈ r.die.faces, i32Min ≤ x ∧ x ≤ i32Max := ht.die ▸ hF
  -- A drop or keep keeps a subset of the chains, and the chains alone.
  have hdrop : ∀ r'' : Record, r''.die = r.die →
      (∀ ch ∈ r''.keptChains, ch ∈ r.keptChains) → D.Tracks d r'' :=
    fun r'' hd hsub => ⟨hd.trans ht.die, fun ch h => ht.fine ch (hsub ch h),
      fun ch h => ht.covers ch (hsub ch h)⟩
  cases b with
  | reroll criteria limit =>
    cases cl with
    | reroll c l =>
      obtain ⟨⟨cr, hcr, ha⟩, hL⟩ := hb
      refine ⟨(die_of_mem_reroll hr).trans ht.die, fun ch' hch' => ?_,
        fun ch' hch' => ?_⟩ <;>
      obtain ⟨ch, hch, hbe⟩ :=
        (mem_reroll_iff hF' hk ch').mp ⟨r', hr, hch'⟩ <;>
      rw [ht.die] at hbe
      · exact Fine.rerollBecomes hF (ht.fine ch hch) hbe
      · exact D.reroll_covers hD hcr ha hL (ht.fine ch hch) (ht.covers ch hch)
          hbe
    | _ => exact hb.elim
  | explode criteria limit =>
    cases cl with
    | explode c l =>
      obtain ⟨⟨cr, hcr, ha⟩, hL⟩ := hb
      refine ⟨(die_of_mem_explode hr).trans ht.die, fun ch' hch' => ?_,
        fun ch' hch' => ?_⟩ <;>
      obtain ⟨ch, hch, hbe⟩ :=
        (mem_explode_iff hF' hk ch').mp ⟨r', hr, hch'⟩ <;>
      rw [ht.die] at hbe
      · exact Fine.becomes hF (ht.fine ch hch) hbe
      · exact D.explode_covers hD hcr ha hL (ht.fine ch hch) (ht.covers ch hch)
          hbe
    | _ => exact hb.elim
  | drop =>
    cases cl with
    | dropLowest n =>
      rw [Clause.apply, PMF.mem_support_pure_iff] at hr
      subst hr
      exact hdrop _ rfl fun ch => mem_keptChains_dropLowest
    | dropHighest n =>
      rw [Clause.apply, PMF.mem_support_pure_iff] at hr
      subst hr
      exact hdrop _ rfl fun ch => mem_keptChains_dropHighest
    | keepLowest n =>
      rw [Clause.apply, PMF.mem_support_pure_iff] at hr
      subst hr
      exact hdrop _ rfl fun ch => mem_keptChains_dropHighest
    | keepHighest n =>
      rw [Clause.apply, PMF.mem_support_pure_iff] at hr
      subst hr
      exact hdrop _ rfl fun ch => mem_keptChains_dropLowest
    | _ => exact hb.elim

/-- Running the bounds of some clauses after others runs them after the
chains that the others leave. -/
private theorem runBounds_cons (D : DieChains) (b : ClauseBounds)
    (bs : List ClauseBounds) :
    (runBounds D (b :: bs)).1 = (runBounds (b.apply D).1 bs).1 := by
  have key : ∀ (bs : List ClauseBounds) (D : DieChains) (f f' : Bool),
      (bs.foldl (fun acc b => let next := b.apply acc.1;
          (next.1, acc.2 || next.2)) (D, f)).1 =
        (bs.foldl (fun acc b => let next := b.apply acc.1;
          (next.1, acc.2 || next.2)) (D, f')).1 := by
    intro bs
    induction bs with
    | nil => intros; rfl
    | cons b bs ih => intro D f f'; exact ih _ _ _
  exact key bs _ _ _

/--
The bounds of some clauses, each of which stands for its clause, track every
record that the clauses may lead to from a record that the chains tracked.

# Parameters
- `D`: The chains.
- `d`: The die.
- `bs`: The bounds of the clauses.
- `cls`: The clauses.
- `r`, `r'`: The record, and a record that the clauses may lead to.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hD`: The chains stand for the die.
- `hb`: The bounds of each clause stand for it.
- `ht`: The chains track the record.
- `hr`: The clauses may lead to `r'`.
-/
theorem DieChains.run_tracks {D : DieChains} {d : Die}
    {bs : List ClauseBounds} {cls : List Clause} {r r' : Record}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) (hD : D.Represents d)
    (hb : List.Forall₂ (ClauseBounds.Admits D.covers d) bs cls)
    (ht : D.Tracks d r) (hr : Reaches r cls r') :
    (runBounds D bs).1.Tracks d r' := by
  have key : ∀ {covers : Criteria → Bool} {bs : List ClauseBounds}
      {cls : List Clause}, List.Forall₂ (ClauseBounds.Admits covers d) bs cls →
      ∀ {D : DieChains} {r : Record}, D.covers = covers → D.Represents d →
        D.Tracks d r → Reaches r cls r' → (runBounds D bs).1.Tracks d r' := by
    intro covers bs cls hb
    induction hb with
    | nil =>
      intro D r _ _ ht hr
      rw [show r' = r from hr]
      exact ht
    | @cons b cl bs cls hbc _ ih =>
      intro D r hc hD ht hr
      obtain ⟨r₁, hr₁, hr⟩ := hr
      have hfix := b.apply_fixed D
      rw [runBounds_cons]
      exact ih (hfix.2.1.trans hc) ⟨hfix.1 ▸ hD.1, hfix.2.2 ▸ hD.2⟩
        (ClauseBounds.tracks hF hD (hc ▸ hbc) ht hr₁) hr
  exact key hb rfl hD ht hr

/-- A union of intervals holds what it starts with, and each interval. -/
private theorem mem_foldl_Icc {α : Type} (f : α → Int × Int) (v : Int) :
    ∀ (xs : List α) (init : Finset Int),
      v ∈ xs.foldl (fun acc x => acc ∪ Finset.Icc (f x).1 (f x).2) init ↔
        v ∈ init ∨ ∃ x ∈ xs, (f x).1 ≤ v ∧ v ≤ (f x).2
  | [], init => by simp
  | x :: xs, init => by
    rw [List.foldl_cons, mem_foldl_Icc f v xs, Finset.mem_union,
      Finset.mem_Icc]
    simp only [List.mem_cons, exists_eq_or_imp]
    tauto

/-- The totals of chains hold the totals of each shape and their saturated
values. -/
private theorem mem_totals_iff {D : DieChains} {v : Int} :
    v ∈ D.totals ↔ v ∈ D.saturated ∨
      ∃ S ∈ D.live, S.totals.1 ≤ v ∧ v ≤ S.totals.2 :=
  mem_foldl_Icc ChainShape.totals v D.live D.saturated

/--
The totals of some chains hold the total of every chain that they cover: the
value of a saturated chain, or a total between the extremes of a shape that
the chain has.

# Parameters
- `D`: The chains.
- `ch`: The chain.

# Hypotheses
- `h`: The chains cover the chain.
-/
theorem DieChains.total_mem_totals {D : DieChains} {ch : Chain}
    (h : D.Covers ch) : ch.total ∈ D.totals := by
  rw [mem_totals_iff]
  cases ch with
  | saturated v => exact .inl h
  | live body last =>
    obtain ⟨S, hS, hc⟩ := h
    exact .inr ⟨S, hS, ChainShape.totals_le hc⟩

/--
The chains of a die just rolled track every record of the die each of whose
kept chains holds one face: before any reroll or explode, every chain does.

# Parameters
- `F`: The faces that the chains admit.
- `covers`: Whether some criterion of some criteria may match every face of
  some die.
- `saturating`: The values that a saturated die may take.
- `d`: The die.
- `r`: The record.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range, and the chains admit
  each.
- `hd`: The record holds the die.
- `hk`: Each kept chain holds one face of the die.
-/
theorem DieChains.rolled_tracks {F : Finset Int} {covers : Criteria → Bool}
    {saturating : Finset Int} {d : Die} {r : Record}
    (hF : ∀ x ∈ d.faces, (i32Min ≤ x ∧ x ≤ i32Max) ∧ x ∈ F)
    (hd : r.die = d)
    (hk : ∀ ch ∈ r.keptChains, ∃ x ∈ d.faces, ch = .single x) :
    (DieChains.rolled F covers saturating).Tracks d r := by
  refine ⟨hd, fun ch hch => ?_, fun ch hch => ?_⟩ <;>
    obtain ⟨x, hx, rfl⟩ := hk ch hch
  · refine ⟨⟨by simp, fun y hy => ?_⟩, fun _ y hy => ?_⟩
    · simp only [List.mem_singleton] at hy
      exact hy ▸ (hF x hx).1
    · simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
        List.mem_singleton] at hy
      exact hy ▸ hx
  · have hne : F ≠ ∅ := Finset.ne_empty_of_mem (hF x hx).2
    refine ⟨.new [] F, by simp [DieChains.rolled, hne], ?_⟩
    exact ChainShape.has_new ⟨(hF x hx).2, [], .nil, by simp⟩

/--
The bounds of a rolling record are sound for every sequence of rerolls,
explodes, drops, and keeps: if the bounds of each clause stand for it, then
the total of every chain still kept of every record that the clauses may
lead to from a record that the chains tracked lies among the totals of the
chains after them, and so within their hull, which is all that any reader
reads.

# Parameters
- `D`: The chains.
- `d`: The die.
- `bs`: The bounds of the clauses.
- `cls`: The clauses.
- `r`, `r'`: The record, and a record that the clauses may lead to.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hD`: The chains stand for the die.
- `hb`: The bounds of each clause stand for it.
- `ht`: The chains track the record.
- `hr`: The clauses may lead to `r'`.
-/
theorem DieChains.run_sound {D : DieChains} {d : Die}
    {bs : List ClauseBounds} {cls : List Clause} {r r' : Record}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) (hD : D.Represents d)
    (hb : List.Forall₂ (ClauseBounds.Admits D.covers d) bs cls)
    (ht : D.Tracks d r) (hr : Reaches r cls r') :
    ∀ ch ∈ r'.keptChains, ch.total ∈ (runBounds D bs).1.totals :=
  fun ch hch => DieChains.total_mem_totals
    ((DieChains.run_tracks hF hD hb ht hr).covers ch hch)

/-! ## Exactness of the shapes -/

/--
Whether a number of dice is a corner of a run, under a threshold: its fewest,
its most, or, if it may hold any number, any number at least the threshold.
The totals of a shape fold each run at one of these.

# Parameters
- `T`: The threshold.
- `run`: The run.
- `k`: The number of dice.
-/
def Segment.CornerCount (T : Nat) (run : Segment) (k : Nat) : Prop :=
  k = run.least ∨ run.greatest = (k : ℕ∞) ∨ (run.greatest = ⊤ ∧ T ≤ k)

/--
Whether some dice are a corner of a run, under a threshold: a corner number of
copies of one of its results.

# Parameters
- `T`: The threshold.
- `run`: The run.
- `xs`: The dice.
-/
def Segment.Corner (T : Nat) (run : Segment) (xs : List Int) : Prop :=
  ∃ k, run.CornerCount T k ∧ ∃ ρ ∈ run.results, xs = List.replicate k ρ

/--
Whether a chain that may still grow is a corner of a shape, under a
threshold: its last die shows one of the results of the shape's, and its other
dice split among the runs, each of which holds a corner of its own.

# Parameters
- `T`: The threshold.
- `S`: The shape.
- `body`, `last`: The chain.
-/
def ChainShape.Corner (T : Nat) (S : ChainShape) (body : List Int)
    (last : Int) : Prop :=
  last ∈ S.last ∧ ∃ parts : List (List Int),
    List.Forall₂ (Segment.Corner T) S.middles parts ∧ body.Perm parts.flatten

/--
Whether every corner of a shape, under some threshold, is a chain that some
clauses may make, with its dice in some order. The extreme totals of a shape
that is realized are totals of such chains (`ChainShape.totals_realized`).

# Parameters
- `Reach`: The chains that the clauses may make.
- `S`: The shape.
-/
def ChainShape.Realized (Reach : Chain → Prop) (S : ChainShape) : Prop :=
  ∃ T, ∀ body last, S.Corner T body last →
    ∃ body', body.Perm body' ∧ Reach (.live body' last)

/--
Whether a shape is well formed for a die of some faces: its last die and each
of its runs may show some result, and only faces.

# Parameters
- `F`: The faces.
- `S`: The shape.
-/
def ChainShape.WellFormed (F : Finset Int) (S : ChainShape) : Prop :=
  S.last.Nonempty ∧ S.last ⊆ F ∧
    ∀ run ∈ S.middles, run.results.Nonempty ∧ run.results ⊆ F

/-- A corner number of dice under a threshold is one under any lower. -/
private theorem Segment.CornerCount.mono {T T' : Nat} {run : Segment}
    {k : Nat} (h : run.CornerCount T' k) (hT : T ≤ T') :
    run.CornerCount T k := by
  rcases h with h | h | ⟨h, hk⟩
  · exact .inl h
  · exact .inr (.inl h)
  · exact .inr (.inr ⟨h, by omega⟩)

/-- Corners of runs under a threshold are corners under any lower. -/
private theorem forall₂_corner_mono {T T' : Nat} (hT : T ≤ T') :
    ∀ {runs : List Segment} {parts : List (List Int)},
      List.Forall₂ (Segment.Corner T') runs parts →
      List.Forall₂ (Segment.Corner T) runs parts
  | _, _, .nil => .nil
  | _, _, .cons ⟨k, hk, ρ, hρ, he⟩ h =>
    .cons ⟨k, hk.mono hT, ρ, hρ, he⟩ (forall₂_corner_mono hT h)

/-- Runs that may hold no die hold no dice at a corner, so dropping them loses
no corner, if every run may show some result. -/
private theorem corner_filter {T : Nat} :
    ∀ {runs : List Segment}, (∀ run ∈ runs, run.results.Nonempty) →
      ∀ {parts : List (List Int)}, List.Forall₂ (Segment.Corner T)
        (runs.filter fun run => decide (0 < run.greatest)) parts →
      ∃ parts', List.Forall₂ (Segment.Corner T) runs parts' ∧
        parts'.flatten = parts.flatten
  | [], _, _, h => by
    simp only [List.filter_nil, List.forall₂_nil_left_iff] at h
    exact ⟨[], .nil, by simp [h]⟩
  | run :: runs, hn, parts, h => by
    have hn' := fun r hr => hn r (List.mem_cons_of_mem _ hr)
    by_cases hg : 0 < run.greatest
    · simp only [List.filter_cons, hg, decide_true, ↓reduceIte] at h
      obtain ⟨xs, ps, hc, hps, rfl⟩ : ∃ xs ps, Segment.Corner T run xs ∧
          List.Forall₂ (Segment.Corner T)
            (runs.filter fun run => decide (0 < run.greatest)) ps ∧
          parts = xs :: ps := by
        cases h with
        | cons hc h => exact ⟨_, _, hc, h, rfl⟩
      obtain ⟨ps', hp, he⟩ := corner_filter hn' hps
      exact ⟨xs :: ps', .cons hc hp, by simp [he]⟩
    · simp only [List.filter_cons, hg, decide_false, Bool.false_eq_true,
        ↓reduceIte] at h
      obtain ⟨ps', hp, he⟩ := corner_filter hn' h
      obtain ⟨ρ, hρ⟩ := hn run List.mem_cons_self
      have hz : run.greatest = ((0 : ℕ) : ℕ∞) := by
        rw [not_lt, nonpos_iff_eq_zero] at hg
        simpa using hg
      exact ⟨[] :: ps', .cons ⟨0, .inr (.inl hz), ρ, hρ, rfl⟩ hp,
        by simp [he]⟩

/-- The finite part of a greatest number of dice, or `0` for any number. -/
private def finite : ℕ∞ → Nat
  | ⊤ => 0
  | (n : ℕ) => n

/-- A corner number of dice of two runs merged splits into corner numbers of
each, under thresholds low enough. -/
private theorem Segment.corner_merge {T₁ T₂ : Nat} {run next : Segment}
    {k : Nat}
    (h : Segment.CornerCount (T₁ + T₂ + finite run.greatest +
        finite next.greatest)
      { run with least := run.least + next.least,
                 greatest := run.greatest + next.greatest } k) :
    ∃ k₁ k₂, k = k₁ + k₂ ∧ run.CornerCount T₁ k₁ ∧ next.CornerCount T₂ k₂ := by
  rcases h with h | h | ⟨h, hk⟩
  · exact ⟨_, _, h, .inl rfl, .inl rfl⟩
  · simp only at h
    induction hg₁ : run.greatest using ENat.recTopCoe with
    | top => rw [hg₁, top_add] at h; exact absurd h.symm (ENat.natCast_ne_top _)
    | coe n₁ =>
      induction hg₂ : next.greatest using ENat.recTopCoe with
      | top =>
        rw [hg₁, hg₂, add_top] at h
        exact absurd h.symm (ENat.natCast_ne_top _)
      | coe n₂ =>
        rw [hg₁, hg₂, ← Nat.cast_add, Nat.cast_inj] at h
        exact ⟨n₁, n₂, h.symm, .inr (.inl hg₁), .inr (.inl hg₂)⟩
  · simp only at h hk
    induction hg₁ : run.greatest using ENat.recTopCoe with
    | top =>
      induction hg₂ : next.greatest using ENat.recTopCoe with
      | top =>
        simp only [hg₁, hg₂, finite] at hk
        exact ⟨T₁, k - T₁, by omega, .inr (.inr ⟨hg₁, le_rfl⟩),
          .inr (.inr ⟨hg₂, by omega⟩)⟩
      | coe n₂ =>
        simp only [hg₁, hg₂, finite] at hk
        exact ⟨k - n₂, n₂, by omega, .inr (.inr ⟨hg₁, by omega⟩),
          .inr (.inl hg₂)⟩
    | coe n₁ =>
      induction hg₂ : next.greatest using ENat.recTopCoe with
      | top =>
        simp only [hg₁, hg₂, finite] at hk
        exact ⟨n₁, k - n₁, by omega, .inr (.inl hg₁),
          .inr (.inr ⟨hg₂, by omega⟩)⟩
      | coe n₂ =>
        rw [hg₁, hg₂, ← Nat.cast_add] at h
        exact absurd h (ENat.natCast_ne_top _)

/-- Corners of runs merged split into corners of the runs, under a threshold
low enough. -/
private theorem corner_mergeRuns :
    ∀ (runs : List Segment) (T : Nat), ∃ T', ∀ {parts : List (List Int)},
      List.Forall₂ (Segment.Corner T') (mergeRuns runs) parts →
      ∃ parts', List.Forall₂ (Segment.Corner T) runs parts' ∧
        parts'.flatten.Perm parts.flatten
  | [], T => ⟨T, fun h => ⟨_, h, .refl _⟩⟩
  | run :: runs, T => by
    obtain ⟨T₁, ih⟩ := corner_mergeRuns runs T
    -- A threshold low enough for the run, for the merged runs after it, and
    -- for the run merged with the first of them.
    refine ⟨T + T₁ + finite run.greatest +
      ((mergeRuns runs).head?.map fun next => finite next.greatest).getD 0,
      fun {parts} h => ?_⟩
    simp only [mergeRuns] at h
    split at h
    · rename_i next rest hm
      split at h
      · rename_i heq
        obtain ⟨zs, ps, ⟨k, hk, ρ, hρ, rfl⟩, hp, rfl⟩ :
            ∃ zs ps, Segment.Corner _ _ zs ∧
              List.Forall₂ (Segment.Corner _) rest ps ∧ parts = zs :: ps := by
          cases h with
          | cons hc h => exact ⟨_, _, hc, h, rfl⟩
        simp only [hm, List.head?_cons, Option.map_some, Option.getD_some]
          at hk hp
        obtain ⟨k₁, k₂, rfl, hk₁, hk₂⟩ :=
          Segment.corner_merge (T₁ := T) (T₂ := T₁) hk
        obtain ⟨ps', hp', he⟩ := ih (parts := List.replicate k₂ ρ :: ps)
          (hm ▸ .cons ⟨k₂, hk₂, ρ, heq ▸ hρ, rfl⟩
            (forall₂_corner_mono (T := T₁) (by omega) hp))
        refine ⟨List.replicate k₁ ρ :: ps', .cons ⟨k₁, hk₁, ρ, hρ, rfl⟩ hp', ?_⟩
        simp only [List.flatten_cons, List.replicate_add, List.append_assoc]
        exact he.append_left _
      · obtain ⟨xs, ps, hc, hp, rfl⟩ :
            ∃ xs ps, Segment.Corner _ run xs ∧
              List.Forall₂ (Segment.Corner _) (next :: rest) ps ∧
                parts = xs :: ps := by
          cases h with
          | cons hc h => exact ⟨_, _, hc, h, rfl⟩
        obtain ⟨k, hk, ρ, hρ, rfl⟩ := hc
        obtain ⟨ps', hp', he⟩ :=
          ih (hm ▸ forall₂_corner_mono (T := T₁) (by omega) hp)
        exact ⟨_ :: ps', .cons ⟨k, hk.mono (T := T) (by omega), ρ, hρ, rfl⟩
          hp', he.append_left _⟩
    · rename_i hm
      obtain ⟨xs, rfl⟩ : ∃ xs, parts = [xs] := by
        cases h with
        | cons hc h => cases h; exact ⟨_, rfl⟩
      obtain ⟨k, hk, ρ, hρ, rfl⟩ : Segment.Corner _ run xs := by
        cases h with
        | cons hc _ => exact hc
      obtain ⟨ps', hp', he⟩ := ih (parts := []) (hm ▸ .nil)
      exact ⟨_ :: ps', .cons ⟨k, hk.mono (T := T) (by omega), ρ, hρ, rfl⟩ hp',
        by simpa using he.append_left (List.replicate k ρ)⟩

/--
Every corner of the shape that some runs and a last die make is a corner of
the runs as they are, under a threshold low enough, if every run may show
some result: so `ChainShape.new` makes no corner of its own.

# Parameters
- `middles`: The runs.
- `last`: The results that the last die may show.
- `T`: The threshold of the corners of the runs as they are.

# Hypotheses
- `hn`: Every run may show some result.
-/
theorem ChainShape.corner_new {middles : List Segment} {last : Finset Int}
    (hn : ∀ run ∈ middles, run.results.Nonempty) (T : Nat) :
    ∃ T', ∀ body y, (ChainShape.new middles last).Corner T' body y →
      ChainShape.Corner T { middles, last } body y := by
  obtain ⟨T', hT⟩ := corner_mergeRuns
    ((middles.filter fun run : Segment => decide (0 < run.greatest)).mergeSort
      fun a b => decide (orderKey a.results ≤ orderKey b.results)) T
  refine ⟨T', fun body y ⟨hy, parts, hp, hb⟩ => ?_⟩
  obtain ⟨parts₁, hp₁, he₁⟩ := hT hp
  obtain ⟨parts₂, hp₂, he₂⟩ := List.perm_comp_forall₂
    (List.mergeSort_perm _ _).symm hp₁
  obtain ⟨parts₃, hp₃, he₃⟩ := corner_filter hn hp₂
  refine ⟨hy, parts₃, hp₃, ?_⟩
  rw [he₃]
  exact hb.trans (he₁.symm.trans he₂.flatten.symm)

/--
A shape that some runs and a last die make is realized if the runs as they are
are, and every run may show some result.

# Parameters
- `Reach`: The chains that the clauses may make.
- `middles`: The runs.
- `last`: The results that the last die may show.

# Hypotheses
- `hn`: Every run may show some result.
- `h`: The runs as they are are realized.
-/
theorem ChainShape.Realized.new {Reach : Chain → Prop}
    {middles : List Segment} {last : Finset Int}
    (hn : ∀ run ∈ middles, run.results.Nonempty)
    (h : ChainShape.Realized Reach { middles, last }) :
    (ChainShape.new middles last).Realized Reach := by
  obtain ⟨T, hT⟩ := h
  obtain ⟨T', hT'⟩ := ChainShape.corner_new hn (last := last) T
  exact ⟨T', fun body y hc => hT body y (hT' body y hc)⟩

/-- The negative dice of two lists sum alike, or both saturate, if each die
agrees with its counterpart, and so do their copies unless the die is `0` or
both are at least `unboundedDice`. -/
private theorem sums_cap {l l' : List (Int × Nat)}
    (h : List.Forall₂ (fun a b => a.1 = b.1 ∧ (a.2 = b.2 ∨ a.1 = 0 ∨
      (unboundedDice ≤ a.2 ∧ unboundedDice ≤ b.2))) l l') :
    (negative l = negative l' ∨ (negative l ≤ -(unboundedDice : Int) ∧
        negative l' ≤ -(unboundedDice : Int))) ∧
      (positive l = positive l' ∨ ((unboundedDice : Int) ≤ positive l ∧
        (unboundedDice : Int) ≤ positive l')) := by
  induction h with
  | nil => simp
  | @cons a b l l' hab _ ih =>
    obtain ⟨hn, hp⟩ := ih
    have := negative_nonpos l
    have := negative_nonpos l'
    have := positive_nonneg l
    have := positive_nonneg l'
    obtain ⟨he, hk⟩ := hab
    simp only [negative_cons, positive_cons, ← he]
    have hU : (0 : Int) < unboundedDice := by simp [unboundedDice]
    rcases hk with hk | hk | ⟨hk₁, hk₂⟩
    · rw [hk]
      have : Min.min a.1 0 * (b.2 : Int) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg (min_le_right _ _) (by omega)
      have : 0 ≤ Max.max a.1 0 * (b.2 : Int) :=
        mul_nonneg (le_max_right _ _) (by omega)
      constructor <;> omega
    · rw [hk]
      simp only [min_self, max_self, zero_mul, zero_add]
      exact ⟨hn, hp⟩
    · have hk₁' : (unboundedDice : Int) ≤ a.2 := by exact_mod_cast hk₁
      have hk₂' : (unboundedDice : Int) ≤ b.2 := by exact_mod_cast hk₂
      rcases lt_trichotomy a.1 0 with hx | hx | hx
      · rw [min_eq_left hx.le, max_eq_right hx.le]
        have : a.1 * a.2 ≤ -(unboundedDice : Int) := by nlinarith
        have : a.1 * b.2 ≤ -(unboundedDice : Int) := by nlinarith
        constructor <;> omega
      · rw [hx]
        simp only [min_self, max_self, zero_mul, zero_add]
        exact ⟨hn, hp⟩
      · rw [min_eq_right hx.le, max_eq_left hx.le]
        have : (unboundedDice : Int) ≤ a.1 * a.2 := by nlinarith
        have : (unboundedDice : Int) ≤ a.1 * b.2 := by nlinarith
        constructor <;> omega

/--
The fold of some dice is that of others of the same dice, each of whose copies
agree unless the die is `0` or both are at least `unboundedDice`, since beyond
it every nonzero die has saturated the fold (`fold_unbounded`).

# Parameters
- `l`, `l'`: The dice.

# Hypotheses
- `h`: Each die agrees with its counterpart, and so do their copies, unless
  the die is `0` or both are at least `unboundedDice`.
-/
theorem fold_eq_of_cap {l l' : List (Int × Nat)}
    (h : List.Forall₂ (fun a b => a.1 = b.1 ∧ (a.2 = b.2 ∨ a.1 = 0 ∨
      (unboundedDice ≤ a.2 ∧ unboundedDice ≤ b.2))) l l') :
    fold l = fold l' := by
  obtain ⟨hn, hp⟩ := sums_cap h
  have := negative_nonpos l
  have := negative_nonpos l'
  have := positive_nonneg l
  have := positive_nonneg l'
  simp only [fold, unboundedDice, i32Min, i32Max] at *
  push_cast at hn hp
  omega

/-- The least of a set of results that holds some result is one of them. -/
private theorem low_mem {s : Finset Int} (h : s.Nonempty) : low s ∈ s := by
  obtain ⟨x, hx⟩ := h
  obtain ⟨p, hp, -⟩ := exists_hull hx
  simpa [low, hp] using (hull_spec hp).1

/-- The greatest of a set of results that holds some result is one of
them. -/
private theorem high_mem {s : Finset Int} (h : s.Nonempty) : high s ∈ s := by
  obtain ⟨x, hx⟩ := h
  obtain ⟨p, hp, -⟩ := exists_hull hx
  simpa [high, hp] using (hull_spec hp).2.1

/-- Each run whose term some corner of it agrees with, under the cap, has
corner dice whose copies agree with the terms. -/
private theorem corners_of_terms {T : Nat} {term : Segment → Int × Nat} :
    ∀ {runs : List Segment}, (∀ run ∈ runs, ∃ k, run.CornerCount T k ∧
      (term run).1 ∈ run.results ∧ (k = (term run).2 ∨ (term run).1 = 0 ∨
        (unboundedDice ≤ k ∧ unboundedDice ≤ (term run).2))) →
      ∃ parts pairs, List.Forall₂ (Segment.Corner T) runs parts ∧
        parts.flatten = Roll.expand pairs ∧
        List.Forall₂ (fun a b => a.1 = b.1 ∧ (a.2 = b.2 ∨ a.1 = 0 ∨
          (unboundedDice ≤ a.2 ∧ unboundedDice ≤ b.2))) pairs (runs.map term)
  | [], _ => ⟨[], [], .nil, rfl, .nil⟩
  | run :: runs, h => by
    obtain ⟨k, hk, hρ, hc⟩ := h run List.mem_cons_self
    obtain ⟨parts, pairs, hp, he, hr⟩ :=
      corners_of_terms fun r hr => h r (List.mem_cons_of_mem _ hr)
    refine ⟨List.replicate k (term run).1 :: parts, ((term run).1, k) :: pairs,
      .cons ⟨k, hk, _, hρ, rfl⟩ hp, ?_, .cons ⟨rfl, hc⟩ hr⟩
    simp [Roll.expand, he]

/-- A realized shape's corner whose runs agree with some terms is a chain that
the clauses may make, totaling the fold of the terms. -/
private theorem realized_fold {Reach : Chain → Prop} {S : ChainShape}
    {T : Nat} (hT : ∀ body last, S.Corner T body last →
      ∃ body', body.Perm body' ∧ Reach (.live body' last))
    {term : Segment → Int × Nat}
    (h : ∀ run ∈ S.middles, ∃ k, run.CornerCount T k ∧
      (term run).1 ∈ run.results ∧ (k = (term run).2 ∨ (term run).1 = 0 ∨
        (unboundedDice ≤ k ∧ unboundedDice ≤ (term run).2)))
    {y : Int} (hy : y ∈ S.last) :
    ∃ ch, Reach ch ∧ ch.total = fold ((y, 1) :: S.middles.map term) := by
  obtain ⟨parts, pairs, hp, he, hr⟩ := corners_of_terms h
  obtain ⟨body', hb, hR⟩ := hT parts.flatten y ⟨hy, parts, hp, .refl _⟩
  refine ⟨_, hR, ?_⟩
  rw [total_live, ← fold_eq_of_cap (l := (y, 1) :: pairs)
    (.cons ⟨rfl, .inl rfl⟩ hr)]
  refine fold_congr ?_
  rw [show Roll.expand ((y, 1) :: body'.map (·, 1)) = y :: body' by
      simp only [Roll.expand, List.flatMap_cons, List.replicate_one,
        List.singleton_append]
      rw [show List.flatMap (fun p : Int × Nat => List.replicate p.2 p.1)
          (body'.map ((·, 1))) = body' from expand_singles body'],
    show Roll.expand ((y, 1) :: pairs) = y :: Roll.expand pairs by
      simp [Roll.expand], ← he]
  exact (hb.symm.cons y)

/-- A run may hold a corner number of dice whose copies agree with those of its
least term, under the cap. -/
private theorem Segment.lowTerm_corner (T : Nat) {run : Segment}
    (hn : run.results.Nonempty) :
    ∃ k, run.CornerCount T k ∧ run.lowTerm.1 ∈ run.results ∧
      (k = run.lowTerm.2 ∨ run.lowTerm.1 = 0 ∨
        (unboundedDice ≤ k ∧ unboundedDice ≤ run.lowTerm.2)) := by
  simp only [Segment.lowTerm]
  suffices h : ∃ k, run.CornerCount T k ∧
      (k = copies (if low run.results < 0 then run.greatest else run.least) ∨
        low run.results = 0 ∨ (unboundedDice ≤ k ∧ unboundedDice ≤
          copies (if low run.results < 0 then run.greatest else run.least))) by
    obtain ⟨k, hk, h⟩ := h
    exact ⟨k, hk, low_mem hn, h⟩
  split_ifs with hlo
  · by_cases ht : run.greatest = ⊤
    · refine ⟨Max.max T unboundedDice, .inr (.inr ⟨ht, le_max_left _ _⟩),
        .inr (.inr ⟨le_max_right _ _, by simp [copies, ht]⟩)⟩
    · have hg : run.greatest = (run.greatest.toNat : ℕ∞) :=
        (ENat.natCast_toNat ht).symm
      refine ⟨run.greatest.toNat, .inr (.inl hg), ?_⟩
      simp only [copies, ht, ↓reduceIte]
      by_cases hU : run.greatest.toNat ≤ unboundedDice
      · exact .inl (Nat.min_eq_left hU).symm
      · exact .inr (.inr ⟨by omega, by simp; omega⟩)
  · refine ⟨run.least, .inl rfl, ?_⟩
    rw [copies_natCast]
    by_cases hU : run.least ≤ unboundedDice
    · exact .inl (Nat.min_eq_left hU).symm
    · exact .inr (.inr ⟨by omega, by simp; omega⟩)

/-- A run may hold a corner number of dice whose copies agree with those of its
greatest term, under the cap. -/
private theorem Segment.highTerm_corner (T : Nat) {run : Segment}
    (hn : run.results.Nonempty) :
    ∃ k, run.CornerCount T k ∧ run.highTerm.1 ∈ run.results ∧
      (k = run.highTerm.2 ∨ run.highTerm.1 = 0 ∨
        (unboundedDice ≤ k ∧ unboundedDice ≤ run.highTerm.2)) := by
  simp only [Segment.highTerm]
  suffices h : ∃ k, run.CornerCount T k ∧
      (k = copies (if 0 < high run.results then run.greatest else run.least) ∨
        high run.results = 0 ∨ (unboundedDice ≤ k ∧ unboundedDice ≤
          copies (if 0 < high run.results then run.greatest else run.least))) by
    obtain ⟨k, hk, h⟩ := h
    exact ⟨k, hk, high_mem hn, h⟩
  split_ifs with hhi
  · by_cases ht : run.greatest = ⊤
    · refine ⟨Max.max T unboundedDice, .inr (.inr ⟨ht, le_max_left _ _⟩),
        .inr (.inr ⟨le_max_right _ _, by simp [copies, ht]⟩)⟩
    · have hg : run.greatest = (run.greatest.toNat : ℕ∞) :=
        (ENat.natCast_toNat ht).symm
      refine ⟨run.greatest.toNat, .inr (.inl hg), ?_⟩
      simp only [copies, ht, ↓reduceIte]
      by_cases hU : run.greatest.toNat ≤ unboundedDice
      · exact .inl (Nat.min_eq_left hU).symm
      · exact .inr (.inr ⟨by omega, by simp; omega⟩)
  · refine ⟨run.least, .inl rfl, ?_⟩
    rw [copies_natCast]
    by_cases hU : run.least ≤ unboundedDice
    · exact .inl (Nat.min_eq_left hU).symm
    · exact .inr (.inr ⟨by omega, by simp; omega⟩)

/--
The least and greatest totals of a well formed shape that is realized are
each the total of some chain that the clauses may make, so the totals of the
bounds are exact in their hull.

# Parameters
- `Reach`: The chains that the clauses may make.
- `F`: The faces of the die.
- `S`: The shape.

# Hypotheses
- `hw`: The shape is well formed.
- `hr`: The shape is realized.
-/
theorem ChainShape.totals_realized {Reach : Chain → Prop} {F : Finset Int}
    {S : ChainShape} (hw : S.WellFormed F) (hr : S.Realized Reach) :
    (∃ ch, Reach ch ∧ ch.total = S.totals.1) ∧
      ∃ ch, Reach ch ∧ ch.total = S.totals.2 := by
  obtain ⟨T, hT⟩ := hr
  exact ⟨realized_fold hT (fun run hrun =>
      Segment.lowTerm_corner T (hw.2.2 run hrun).1) (low_mem hw.1),
    realized_fold hT (fun run hrun =>
      Segment.highTerm_corner T (hw.2.2 run hrun).1) (high_mem hw.1)⟩


/-- Every run of runs merged shows the results of one of the runs. -/
private theorem mem_mergeRuns :
    ∀ {runs : List Segment} {run : Segment}, run ∈ mergeRuns runs →
      ∃ r ∈ runs, run.results = r.results
  | [], _, h => by simp [mergeRuns] at h
  | r :: runs, run, h => by
    simp only [mergeRuns] at h
    split at h
    · rename_i next rest hm
      split at h
      · rcases List.mem_cons.mp h with rfl | h
        · exact ⟨r, List.mem_cons_self, rfl⟩
        · obtain ⟨r', hr', he⟩ := mem_mergeRuns (hm ▸ List.mem_cons_of_mem _ h)
          exact ⟨r', List.mem_cons_of_mem _ hr', he⟩
      · rcases List.mem_cons.mp h with rfl | h
        · exact ⟨run, List.mem_cons_self, rfl⟩
        · obtain ⟨r', hr', he⟩ := mem_mergeRuns (hm ▸ h)
          exact ⟨r', List.mem_cons_of_mem _ hr', he⟩
    · rename_i hm
      rcases List.mem_singleton.mp h with rfl
      exact ⟨run, List.mem_cons_self, rfl⟩

/--
The shape that some runs and a last die make is well formed if the runs as
they are are.

# Parameters
- `F`: The faces.
- `middles`: The runs.
- `last`: The results that the last die may show.

# Hypotheses
- `h`: The runs as they are are well formed.
-/
theorem ChainShape.WellFormed.new {F : Finset Int} {middles : List Segment}
    {last : Finset Int} (h : ChainShape.WellFormed F { middles, last }) :
    (ChainShape.new middles last).WellFormed F := by
  refine ⟨h.1, h.2.1, fun run hrun => ?_⟩
  obtain ⟨r, hr, he⟩ := mem_mergeRuns hrun
  have hr := (List.mem_filter.mp ((List.mergeSort_perm _ _).mem_iff.mp hr)).1
  exact he ▸ h.2.2 r hr

/-- A shape realized by some chains is realized by more. -/
theorem ChainShape.Realized.mono {Reach Reach' : Chain → Prop}
    {S : ChainShape} (h : S.Realized Reach) (hR : ∀ ch, Reach ch → Reach' ch) :
    S.Realized Reach' := by
  obtain ⟨T, hT⟩ := h
  exact ⟨T, fun body last hc =>
    (hT body last hc).imp fun _ h => ⟨h.1, hR _ h.2⟩⟩

/-- A shape realized under a threshold is realized under any greater. -/
private theorem realized_of_le {Reach : Chain → Prop} {S : ChainShape}
    {T T' : Nat} (hT : ∀ body last, S.Corner T body last →
      ∃ body', body.Perm body' ∧ Reach (.live body' last)) (hle : T ≤ T') :
    ∀ body last, S.Corner T' body last →
      ∃ body', body.Perm body' ∧ Reach (.live body' last) :=
  fun body last ⟨hl, parts, hp, hb⟩ =>
    hT body last ⟨hl, parts, forall₂_corner_mono hle hp, hb⟩

/-- Every well formed shape has some corner, under any threshold: its fewest
dice of each run, each showing one of its results. -/
theorem ChainShape.exists_corner {F : Finset Int} {S : ChainShape}
    (hw : S.WellFormed F) (T : Nat) : ∃ body last, S.Corner T body last := by
  obtain ⟨y, hy⟩ := hw.1
  have : ∀ runs : List Segment, (∀ run ∈ runs, run.results.Nonempty) →
      ∃ parts, List.Forall₂ (Segment.Corner T) runs parts := by
    intro runs
    induction runs with
    | nil => exact fun _ => ⟨[], .nil⟩
    | cons run runs ih =>
      intro hn
      obtain ⟨ρ, hρ⟩ := hn run List.mem_cons_self
      obtain ⟨parts, hp⟩ := ih fun r hr => hn r (List.mem_cons_of_mem _ hr)
      exact ⟨_ :: parts, .cons ⟨run.least, .inl rfl, ρ, hρ, rfl⟩ hp⟩
  obtain ⟨parts, hp⟩ := this S.middles fun run hrun => (hw.2.2 run hrun).1
  exact ⟨parts.flatten, y, hy, parts, hp, .refl _⟩

/-- A corner of a run united with another is a corner of one of them. -/
private theorem Segment.unite_corner {T : Nat} {run other united : Segment}
    (h : run.unite other = some united) {xs : List Int}
    (hc : united.Corner T xs) : run.Corner T xs ∨ other.Corner T xs := by
  obtain ⟨k, hk, ρ, hρ, rfl⟩ := hc
  unfold Segment.unite at h
  by_cases h₁ : run.results = other.results
  · simp only [h₁, ↓reduceIte, Option.some.injEq] at h
    subst h
    simp only at hρ
    rcases hk with hk | hk | ⟨hk, hT⟩
    · rcases Nat.le_total run.least other.least with hle | hle
      · exact .inl ⟨k, .inl (hk.trans (Nat.min_eq_left hle)), ρ, h₁ ▸ hρ, rfl⟩
      · exact .inr ⟨k, .inl (hk.trans (Nat.min_eq_right hle)), ρ, hρ, rfl⟩
    · simp only [emax_eq] at hk
      rcases max_choice run.greatest other.greatest with he | he <;>
        rw [he] at hk
      · exact .inl ⟨k, .inr (.inl hk), ρ, h₁ ▸ hρ, rfl⟩
      · exact .inr ⟨k, .inr (.inl hk), ρ, hρ, rfl⟩
    · simp only [emax_eq] at hk
      rcases max_choice run.greatest other.greatest with he | he <;>
        rw [he] at hk
      · exact .inl ⟨k, .inr (.inr ⟨hk, hT⟩), ρ, h₁ ▸ hρ, rfl⟩
      · exact .inr ⟨k, .inr (.inr ⟨hk, hT⟩), ρ, hρ, rfl⟩
  · simp only [h₁, ↓reduceIte] at h
    by_cases h₂ : run.least = other.least ∧ run.greatest = other.greatest
    · simp only [h₂, and_self, ↓reduceIte, Option.some.injEq] at h
      subst h
      rcases Finset.mem_union.mp hρ with hρ | hρ
      · refine .inl ⟨k, ?_, ρ, hρ, rfl⟩
        rcases hk with hk | hk | hk
        · exact .inl (hk.trans h₂.1.symm)
        · exact .inr (.inl (h₂.2 ▸ hk))
        · exact .inr (.inr (h₂.2 ▸ hk))
      · exact .inr ⟨k, hk, ρ, hρ, rfl⟩
    · simp only [h₂, ↓reduceIte] at h
      cases h

/-- Corners of runs with one run united with another are corners of the runs
with one or the other. -/
private theorem forall₂_set_corner {T : Nat} {run other united : Segment}
    (h : run.unite other = some united) :
    ∀ {l : List Segment} {parts : List (List Int)} {i : Nat},
      List.Forall₂ (Segment.Corner T) (l.set i united) parts →
      List.Forall₂ (Segment.Corner T) (l.set i run) parts ∨
        List.Forall₂ (Segment.Corner T) (l.set i other) parts
  | [], _, _, hp => .inl (by simpa using hp)
  | _ :: _, _, 0, .cons hc hp =>
    (Segment.unite_corner h hc).imp (fun hc => .cons hc hp)
      (fun hc => .cons hc hp)
  | _ :: _, _, _ + 1, .cons hc hp =>
    (forall₂_set_corner h hp).imp (fun hp => .cons hc hp)
      (fun hp => .cons hc hp)

/--
A shape that two well formed shapes, both realized, unite into is well formed
and realized.

# Parameters
- `Reach`: The chains that the clauses may make.
- `F`: The faces.
- `S`, `T`: The shapes.
- `U`: The united shape.

# Hypotheses
- `hS`, `hT`: The shapes are well formed and realized.
- `h`: They unite into `U`.
-/
theorem ChainShape.unite_realized {Reach : Chain → Prop} {F : Finset Int}
    {S T U : ChainShape} (hS : S.WellFormed F ∧ S.Realized Reach)
    (hT : T.WellFormed F ∧ T.Realized Reach) (h : S.unite T = some U) :
    U.WellFormed F ∧ U.Realized Reach := by
  obtain ⟨TS, hTS⟩ := hS.2
  obtain ⟨TT, hTT⟩ := hT.2
  rcases ChainShape.unite_cases h with ⟨-, rfl⟩ | ⟨-, rfl⟩ |
      ⟨he, rfl⟩ | ⟨i, run, other, united, hrun, hTm, hl, hu, rfl⟩
  · exact hT
  · exact hS
  · refine ⟨⟨hS.1.1.mono Finset.subset_union_left,
      Finset.union_subset hS.1.2.1 hT.1.2.1, hS.1.2.2⟩,
      Max.max TS TT, fun body y ⟨hy, parts, hp, hb⟩ => ?_⟩
    rcases Finset.mem_union.mp hy with hy | hy
    · exact realized_of_le hTS (le_max_left _ _) body y ⟨hy, parts, hp, hb⟩
    · exact realized_of_le hTT (le_max_right _ _) body y
        ⟨hy, parts, he ▸ hp, hb⟩
  · have hi : i < S.middles.length := (List.getElem?_eq_some_iff.mp hrun).1
    have hrun' : run ∈ S.middles := List.mem_of_getElem? hrun
    have hother : other ∈ T.middles := by
      rw [← hTm]
      exact List.mem_set hi other
    -- The united run shows only results of either, which are not empty.
    have hun : united.results.Nonempty ∧ united.results ⊆ F := by
      unfold Segment.unite at hu
      split_ifs at hu <;> cases hu
      · exact hS.1.2.2 run hrun'
      · exact ⟨(hS.1.2.2 run hrun').1.mono Finset.subset_union_left,
          Finset.union_subset (hS.1.2.2 run hrun').2 (hT.1.2.2 other hother).2⟩
    have hw : ChainShape.WellFormed F
        { middles := S.middles.set i united, last := S.last } := by
      refine ⟨hS.1.1, hS.1.2.1, fun r hr => ?_⟩
      rcases List.mem_or_eq_of_mem_set hr with hr | rfl
      · exact hS.1.2.2 r hr
      · exact hun
    refine ⟨hw.new, ChainShape.Realized.new (fun r hr => (hw.2.2 r hr).1) ?_⟩
    refine ⟨Max.max TS TT, fun body y ⟨hy, parts, hp, hb⟩ => ?_⟩
    rcases forall₂_set_corner hu hp with hp | hp
    · rw [set_self_of_getElem? hrun] at hp
      exact realized_of_le hTS (le_max_left _ _) body y ⟨hy, parts, hp, hb⟩
    · rw [hTm] at hp
      exact realized_of_le hTT (le_max_right _ _) body y
        ⟨hl ▸ hy, parts, hp, hb⟩

/-- Absorbing a shape into shapes united keeps a property that uniting
keeps. -/
private theorem absorb_preserves {P : ChainShape → Prop}
    (hP : ∀ {S T U : ChainShape}, P S → P T → S.unite T = some U → P U)
    {shape : ChainShape} {united : List ChainShape} {index : Nat}
    (hs : P shape) (hu : ∀ U ∈ united, P U) :
    P (absorb shape united index).1 ∧
      ∀ U ∈ (absorb shape united index).2, P U := by
  fun_induction absorb shape united index with
  | case1 shape united index hi both hb ih =>
    refine ih (hP (hu _ (List.getElem_mem hi)) hs hb) fun U hU => ?_
    exact hu U ((perm_swapRemove hi).mem_iff.mpr (List.mem_cons_of_mem _ hU))
  | case2 shape united index hi hb ih => exact ih hs hu
  | case3 shape united index hi => exact ⟨hs, hu⟩

/--
Uniting shapes, none of which uniting coarsens, keeps a property that every
one has and that uniting two keeps.

# Parameters
- `P`: The property.
- `shapes`: The shapes.

# Hypotheses
- `hP`: Uniting two shapes of the property makes a shape of it.
- `hs`: Every shape has the property.
- `hc`: Uniting the shapes does not coarsen them.
-/
theorem uniteShapes_preserves {P : ChainShape → Prop}
    (hP : ∀ {S T U : ChainShape}, P S → P T → S.unite T = some U → P U)
    {shapes : List ChainShape} (hs : ∀ S ∈ shapes, P S)
    (hc : (uniteShapes shapes).2 = false) :
    ∀ U ∈ (uniteShapes shapes).1, P U := by
  have key : ∀ (shapes : List ChainShape) (acc : List ChainShape × Bool),
      (acc.2 = false → ∀ U ∈ acc.1, P U) → (∀ S ∈ shapes, P S) →
      (shapes.foldl uniteStep acc).2 = false →
        ∀ U ∈ (shapes.foldl uniteStep acc).1, P U := by
    intro shapes
    induction shapes with
    | nil => intro acc hacc _ hc; exact hacc hc
    | cons S shapes ih =>
      intro acc hacc hs hc
      refine ih _ (fun hc' U hU => ?_) (fun T hT => hs T
        (List.mem_cons_of_mem _ hT)) hc
      unfold uniteStep at hc' hU
      dsimp only at hc' hU
      by_cases hb : shapeBudget < ((absorb S acc.1 0).2 ++
          [(absorb S acc.1 0).1]).length
      · simp only [hb, ↓reduceIte, Bool.true_eq_false] at hc'
      · simp only [hb, ↓reduceIte] at hc' hU
        have ha := absorb_preserves (index := 0) hP (hs S List.mem_cons_self)
          (hacc hc')
        rcases List.mem_append.mp hU with hU | hU
        · exact ha.2 U hU
        · rw [List.mem_singleton.mp hU]
          exact ha.1
  exact key shapes ([], false) (fun _ U hU => by simp at hU) hs hc

/--
Whether some criteria are exact for a criterion of a die: every criterion
matches the same results, which are exactly the faces that the criterion
matches.

# Parameters
- `d`: The die.
- `cr`: The criteria.
- `c`: The criterion.
-/
def Criteria.Matches (d : Die) (cr : Criteria) (c : Criterion) : Prop :=
  cr.certain = cr.possible ∧ ∀ y ∈ d.faces, y ∈ cr.possible ↔ c.matches y = true

/--
Whether some bounds of a limit admit some limit: none, or bounds in order.

# Parameters
- `limit`: The bounds, or `none` for no limit.
-/
def Ordered : Option (Int × Int) → Prop
  | none => True
  | some (lo, hi) => lo ≤ hi

/--
Whether the bounds of a clause are exact for a die: those of a reroll or
explode hold some criteria, each of which is exact for some criterion of the
die, under bounds of the limit in order, as the bounds evaluator enumerates
the criteria of a predicate whose operands are few enough.

# Parameters
- `d`: The die.
- `b`: The bounds of the clause.
-/
def ClauseBounds.Exact (d : Die) : ClauseBounds → Prop
  | .reroll criteria limit | .explode criteria limit =>
    criteria ≠ [] ∧ (∀ cr ∈ criteria, ∃ c, cr.Matches d c) ∧ Ordered limit
  | .drop => True

/--
Whether a clause whose bounds are some bounds may make one chain of another:
a reroll or explode by a criterion for which some of its criteria are exact,
under a limit that its bounds admit, or a drop or keep, which leaves a chain
as it was.

# Parameters
- `d`: The die.
- `b`: The bounds of the clause.
- `ch`, `ch'`: The chain, and what the clause makes of it.
-/
def ChainStep (d : Die) : ClauseBounds → Chain → Chain → Prop
  | .reroll criteria limit, ch, ch' => ∃ cr ∈ criteria, ∃ c,
    cr.Matches d c ∧ ∃ l, Limits limit l ∧ RerollBecomes d c l ch ch'
  | .explode criteria limit, ch, ch' => ∃ cr ∈ criteria, ∃ c,
    cr.Matches d c ∧ ∃ l, Limits limit l ∧ Becomes d c l ch ch'
  | .drop, ch, ch' => ch' = ch

/--
Whether clauses whose bounds are some bounds may make one chain of another,
each of the chain that the last made.

# Parameters
- `d`: The die.
- `bs`: The bounds of the clauses.
- `ch`, `ch'`: The chain, and what the clauses make of it.
-/
def ReachesChain (d : Die) : List ClauseBounds → Chain → Chain → Prop
  | [], ch, ch' => ch' = ch
  | b :: bs, ch, ch'' =>
    ∃ ch', ChainStep d b ch ch' ∧ ReachesChain d bs ch' ch''

/--
The chains of a die just rolled, as the bounds track them once the chains of
a record split by die: the faces of the die, its saturated value, and, as the
least dice of the die, the die itself, some exact criteria of which match
every face exactly when they match each. Mirrors the chains of each die that
`RollingRecordBounds::split` in `evaluator.rs` makes.

# Parameters
- `d`: The die.

# Returns
The chains.
-/
def DieChains.ofDie (d : Die) : DieChains :=
  .rolled d.faces.toFinset (fun cr => decide (d.faces.toFinset ⊆ cr.possible))
    {d.saturatedValue}

/-- Whether some chains of a die are realized by some chains that clauses may
make: the chains are those of the die, as `DieChains.ofDie` makes them, and
every shape is well formed and realized, and every saturated value the value
of some saturated chain. -/
structure DieChains.Realizes (d : Die) (Reach : Chain → Prop) (D : DieChains) :
    Prop where
  /-- The faces are the die's. -/
  faces : D.faces = d.faces.toFinset
  /-- Some criterion may match every face exactly when one matches each. -/
  covers : ∀ cr, D.covers cr = decide (D.faces ⊆ cr.possible)
  /-- A saturated die takes the die's saturated value. -/
  saturating : D.saturating = {d.saturatedValue}
  /-- Every shape is well formed and realized. -/
  shapes : ∀ S ∈ D.live, S.WellFormed D.faces ∧ S.Realized Reach
  /-- Every saturated value is the value of some saturated chain. -/
  saturated : ∀ v ∈ D.saturated, Reach (.saturated v)


/--
The chains that a clause whose bounds are some bounds may make of some chains.

# Parameters
- `d`: The die.
- `Reach`: The chains.
- `b`: The bounds of the clause.
-/
def nextReach (d : Die) (Reach : Chain → Prop) (b : ClauseBounds) :
    Chain → Prop :=
  fun ch' => ∃ ch, Reach ch ∧ ChainStep d b ch ch'

/-- Exact criteria match a face exactly when its criterion does. -/
private theorem Criteria.Matches.mem {d : Die} {cr : Criteria} {c : Criterion}
    (hm : cr.Matches d c) {y : Int} (hy : y ∈ d.faces) :
    (y ∈ cr.possible ↔ c.matches y = true) ∧
      (y ∈ cr.certain ↔ c.matches y = true) :=
  ⟨hm.2 y hy, hm.1 ▸ hm.2 y hy⟩

/-- Exact criteria certainly match every face of a die exactly when their
criterion matches every face. -/
private theorem Criteria.Matches.facesWithin {d : Die} {cr : Criteria}
    {c : Criterion} (hm : cr.Matches d c) :
    d.facesWithin c = true ↔ d.faces.toFinset ⊆ cr.possible := by
  simp only [Die.facesWithin, List.all_eq_true]
  constructor
  · intro h y hy
    have hy := List.mem_toFinset.mp hy
    exact (hm.2 y hy).mpr (h y hy)
  · intro h y hy
    exact (hm.2 y hy).mp (h (List.mem_toFinset.mpr hy))

/-- Exact criteria of chains that a die realizes block a reroll or explode
exactly when their criterion matches every face without limit. -/
private theorem blocked_iff {d : Die} {Reach : Chain → Prop} {D : DieChains}
    (hR : D.Realizes d Reach) {cr : Criteria} {c : Criterion}
    (hm : cr.Matches d c) (limit : Option (Int × Int)) :
    D.blocked limit cr = true ↔
      limit = none ∧ d.facesWithin c = true := by
  rw [hm.facesWithin, ← hR.faces, ← hm.1]
  simp [DieChains.blocked, Option.isNone_iff_eq_none]

/-- A reroll that rerolls no die of a chain leaves it, sorted. -/
private theorem rerolled_self {rerolled : Int → Bool} {P : Int → Prop}
    {body : List Int} {last : Int}
    (h : ∀ x ∈ last :: body, rerolled x = false) :
    Rerolled rerolled P (.live body last) (.live body.mergeSort last) := by
  have hr : ∀ x ∈ last :: body, Redraw rerolled P x x := fun x hx => by
    simp [Redraw, h x hx]
  exact ⟨body, last, List.forall₂_same.mpr fun x hx =>
    hr x (List.mem_cons_of_mem _ hx), hr last List.mem_cons_self, rfl⟩

/-- Corners of runs whose results a reroll maps, each of whose results is a
redraw of a result of its run, are redraws of corners of the runs. -/
private theorem corners_premap {T : Nat} {f : Finset Int → Finset Int}
    {rerolled : Int → Bool} {P : Int → Prop} :
    ∀ {runs : List Segment}, (∀ run ∈ runs, ∀ y ∈ f run.results,
        ∃ x ∈ run.results, Redraw rerolled P x y) →
      ∀ {parts' : List (List Int)},
        List.Forall₂ (Segment.Corner T) (runs.map (Segment.mapResults f))
          parts' →
        ∃ parts, List.Forall₂ (Segment.Corner T) runs parts ∧
          List.Forall₂ (Redraw rerolled P) parts.flatten parts'.flatten
  | [], _, _, h => by
    cases h
    exact ⟨[], .nil, .nil⟩
  | run :: runs, hf, _, .cons ⟨k, hk, y, hy, rfl⟩ h => by
    obtain ⟨x, hx, hxy⟩ := hf run List.mem_cons_self y hy
    obtain ⟨parts, hp, hr⟩ :=
      corners_premap (fun r hr => hf r (List.mem_cons_of_mem _ hr)) h
    refine ⟨List.replicate k x :: parts, .cons ⟨k, hk, x, hx, rfl⟩ hp, ?_⟩
    simp only [List.flatten_cons]
    refine List.rel_append ?_ hr
    clear hk
    induction k with
    | zero => exact .nil
    | succ k ih => exact .cons hxy ih

/-- Every result that a reroll by exact criteria makes of some faces is a
redraw of one of them. -/
private theorem premap_rerollResults {d : Die} {cr : Criteria} {c : Criterion}
    (hm : cr.Matches d c) {rerolled : Int → Bool} {P : Int → Prop}
    {replacements R : Finset Int}
    (hr : ∀ x ∈ d.faces, rerolled x = c.matches x)
    (hP : ∀ y ∈ replacements, P y) (hR : R ⊆ d.faces.toFinset) {y : Int}
    (hy : y ∈ rerollResults cr replacements R) :
    ∃ x ∈ R, Redraw rerolled P x y := by
  unfold rerollResults at hy
  rcases Finset.mem_union.mp hy with hy | hy
  · obtain ⟨hyR, hyc⟩ := Finset.mem_sdiff.mp hy
    have hyF := List.mem_toFinset.mp (hR hyR)
    refine ⟨y, hyR, ?_⟩
    have : c.matches y = false := by
      simpa [(hm.mem hyF).2] using hyc
    simp [Redraw, hr y hyF, this]
  · split at hy
    · rename_i hn
      obtain ⟨x, hx⟩ := hn
      obtain ⟨hxR, hxc⟩ := Finset.mem_inter.mp hx
      have hxF := List.mem_toFinset.mp (hR hxR)
      refine ⟨x, hxR, ?_⟩
      simp [Redraw, hr x hxF, (hm.mem hxF).1.mp hxc, hP y hy]
    · simp at hy

/-- What a reroll by exact criteria makes of some faces that a die may show
are faces, and some, if some faces may replace them. -/
private theorem rerollResults_wf {F : Finset Int} {cr : Criteria}
    (he : cr.certain = cr.possible) {replacements R : Finset Int}
    (hR : R.Nonempty ∧ R ⊆ F) (hrep : replacements.Nonempty)
    (hrepF : replacements ⊆ F) :
    (rerollResults cr replacements R).Nonempty ∧
      rerollResults cr replacements R ⊆ F := by
  unfold rerollResults
  refine ⟨?_, Finset.union_subset (Finset.sdiff_subset.trans hR.2) ?_⟩
  · split
    · exact hrep.mono Finset.subset_union_right
    · rename_i hn
      obtain ⟨x, hx⟩ := hR.1
      refine ⟨x, Finset.mem_union_left _ (Finset.mem_sdiff.mpr ⟨hx, ?_⟩)⟩
      intro hxc
      exact hn ⟨x, Finset.mem_inter.mpr ⟨hx, he ▸ hxc⟩⟩
  · split
    · exact hrepF
    · exact Finset.empty_subset _

/-- A shape whose results a reroll maps is well formed and realized by the
chains that the reroll may make, if the shape is, and its replacements are
faces. -/
private theorem realized_map {d : Die} {Reach : Chain → Prop}
    {D : DieChains} (hR : D.Realizes d Reach) {criteria : List Criteria}
    {limit : Option (Int × Int)} {cr : Criteria} (hcr : cr ∈ criteria)
    {c : Criterion} (hm : cr.Matches d c) {l : Option Nat}
    (hL : Limits limit l) {rerolled : Int → Bool} {P : Int → Prop}
    (hb : ∀ ch ch', Rerolled rerolled P ch ch' → RerollBecomes d c l ch ch')
    (hr : ∀ x ∈ d.faces, rerolled x = c.matches x)
    {replacements : Finset Int} (hP : ∀ y ∈ replacements, P y)
    (hrep : replacements.Nonempty) (hrepF : replacements ⊆ D.faces)
    {S : ChainShape} (hS : S ∈ D.live) :
    (S.map (rerollResults cr replacements)).WellFormed D.faces ∧
      (S.map (rerollResults cr replacements)).Realized
        (nextReach d Reach (.reroll criteria limit)) := by
  obtain ⟨hw, T, hT⟩ := hR.shapes S hS
  have hF : D.faces = d.faces.toFinset := hR.faces
  have hwm : ChainShape.WellFormed D.faces
      { middles := S.middles.map (Segment.mapResults
          (rerollResults cr replacements)),
        last := rerollResults cr replacements S.last } := by
    refine ⟨(rerollResults_wf hm.1 ⟨hw.1, hw.2.1⟩ hrep hrepF).1,
      (rerollResults_wf hm.1 ⟨hw.1, hw.2.1⟩ hrep hrepF).2,
      fun run hrun => ?_⟩
    obtain ⟨r, hr', rfl⟩ := List.mem_map.mp hrun
    exact rerollResults_wf hm.1 (hw.2.2 r hr') hrep hrepF
  refine ⟨hwm.new, ChainShape.Realized.new (fun r hr' => (hwm.2.2 r hr').1)
    ⟨T, fun body y' ⟨hy', parts', hp', hb'⟩ => ?_⟩⟩
  -- Some corner of the shape redraws to this one.
  have hpre : ∀ (R : Finset Int), R ⊆ D.faces → ∀ y ∈ rerollResults cr
      replacements R, ∃ x ∈ R, Redraw rerolled P x y :=
    fun R hRF y hy => premap_rerollResults hm hr hP (hF ▸ hRF) hy
  obtain ⟨y, hyS, hyy⟩ := hpre S.last hw.2.1 y' hy'
  obtain ⟨parts, hp, hpr⟩ := corners_premap
    (fun run hrun => hpre run.results (hw.2.2 run hrun).2) hp'
  obtain ⟨body₀, hb₀, hreach⟩ := hT parts.flatten y ⟨hyS, parts, hp, .refl _⟩
  obtain ⟨ys, hys, hys'⟩ := List.perm_comp_forall₂ hb₀.symm hpr
  refine ⟨ys.mergeSort,
    hb'.trans (hys'.symm.trans (List.mergeSort_perm _ _).symm),
    _, hreach, cr, hcr, c, hm, l, hL, hb _ _ ⟨ys, y', hys, hyy, rfl⟩⟩

/-- Corners of runs and two more split into corners of each. -/
private theorem forall₂_append_two {R : Segment → List Int → Prop} :
    ∀ {l : List Segment} {a b : Segment} {ps : List (List Int)},
      List.Forall₂ R (l ++ [a, b]) ps →
      ∃ ps₀ p q, ps = ps₀ ++ [p, q] ∧ List.Forall₂ R l ps₀ ∧ R a p ∧ R b q
  | [], _, _, _, .cons ha (.cons hb .nil) => ⟨[], _, _, rfl, .nil, ha, hb⟩
  | _ :: _, _, _, _, .cons hx h => by
    obtain ⟨ps₀, p, q, rfl, h₀, ha, hb⟩ := forall₂_append_two h
    exact ⟨_ :: ps₀, p, q, rfl, .cons hx h₀, ha, hb⟩

/-- A shape grown from a realized shape is realized by the chains that some
step makes of each of its corners grown. -/
private theorem realized_grow {Reach Reach' : Chain → Prop} {F : Finset Int}
    {S : ChainShape} {T : Nat} (hw : S.WellFormed F)
    (hT : ∀ body last, S.Corner T body last →
      ∃ body', body.Perm body' ∧ Reach (.live body' last))
    {start middles E : Finset Int} {a : Nat} {b : ℕ∞}
    (hstart : start ⊆ S.last) (hs : start.Nonempty)
    (hmid : middles.Nonempty)
    (hstep : ∀ body₀ s, Reach (.live body₀ s) → s ∈ start → ∀ m μ y,
      Segment.CornerCount T
        { results := middles, least := a, greatest := b } m →
      μ ∈ middles → y ∈ E →
      Reach' (growChain (.live body₀ s) (List.replicate m μ, y))) :
    (S.grow start middles a b E).Realized Reach' := by
  refine ChainShape.Realized.new (fun r hr => ?_) ⟨T, fun body y ⟨hy, parts,
    hp, hb⟩ => ?_⟩
  · rcases List.mem_append.mp hr with hr | hr
    · exact (hw.2.2 r hr).1
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
      rcases hr with rfl | rfl
      · exact hs
      · exact hmid
  · obtain ⟨ps₀, p, q, rfl, hp₀, ⟨k, hk, s, hsS, rfl⟩, ⟨m, hm, μ, hμ, rfl⟩⟩ :=
      forall₂_append_two hp
    have hk1 : k = 1 := by
      rcases hk with hk | hk | ⟨hk, -⟩
      · exact hk
      · simp only at hk
        exact_mod_cast hk.symm
      · simp at hk
    subst hk1
    obtain ⟨body₀, hb₀, hreach⟩ := hT ps₀.flatten s
      ⟨hstart hsS, ps₀, hp₀, .refl _⟩
    refine ⟨_, ?_, hstep body₀ s hreach hsS m μ y hm hμ hy⟩
    refine hb.trans ?_
    simp only [Chain.dice, List.flatten_append, List.flatten_cons,
      List.flatten_nil, List.append_nil, List.replicate_one]
    refine List.Perm.trans ?_ (List.mergeSort_perm _ _).symm
    refine List.Perm.trans ?_ ((List.mergeSort_perm _ _).symm.append_right _)
    refine (hb₀.append_right _).trans ?_
    simp

/-- Every chain of a corner of a realized shape whose last die lies in some of
its results, which a step leaves as it was, is realized by the shape of those
results. -/
private theorem realized_stays {Reach Reach' : Chain → Prop}
    {S : ChainShape} (hr : S.Realized Reach) {E : Finset Int}
    (hE : E ⊆ S.last)
    (hstep : ∀ body last, last ∈ E → Reach (.live body last) →
      Reach' (.live body last)) :
    ChainShape.Realized Reach' { middles := S.middles, last := E } := by
  obtain ⟨T, hT⟩ := hr
  refine ⟨T, fun body y ⟨hy, parts, hp, hb⟩ => ?_⟩
  obtain ⟨body', hb', hreach⟩ := hT body y ⟨hE hy, parts, hp, hb⟩
  exact ⟨body', hb', hstep _ _ hy hreach⟩


/-- Some limit within bounds in order. -/
private theorem exists_limits_of_ordered {limit : Option (Int × Int)}
    (ho : Ordered limit) : ∃ l, Limits limit l := by
  cases limit with
  | none => exact ⟨none, rfl⟩
  | some b => exact ⟨_, b.1, le_rfl, ho, rfl⟩

/-- A reroll leaves every saturated chain as it was. -/
private theorem rerollBecomes_saturated_self (d : Die) (c : Criterion)
    (l : Option Nat) (v : Int) :
    RerollBecomes d c l (.saturated v) (.saturated v) := by
  cases l with
  | some n => rfl
  | none =>
    by_cases hw : d.facesWithin c = true
    · simp [RerollBecomes, hw, Chain.saturates]
    · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
      rfl

/-- An explode leaves every saturated chain as it was. -/
private theorem becomes_saturated_self (d : Die) (c : Criterion)
    (l : Option Nat) (v : Int) :
    Becomes d c l (.saturated v) (.saturated v) := by
  have hg : (Chain.saturated v).grows c = false := rfl
  cases l with
  | some n => simp [Becomes, hg]
  | none =>
    by_cases hw : d.facesWithin c = true
    · simp [Becomes, hw, Chain.saturate]
    · simp [Becomes, hw, hg]

/-- An explode leaves a chain whose last die, a face, it does not match. -/
private theorem becomes_stay {d : Die} {c : Criterion} {body : List Int}
    {last : Int} (hl : last ∈ d.faces) (hm : c.matches last = false)
    (l : Option Nat) :
    Becomes d c l (.live body last) (.live body last) := by
  have hg : (Chain.live body last).grows c = false := by
    simp [Chain.grows, Chain.saturates, Chain.last, hm]
  have hw : d.facesWithin c = false := by
    simp only [Die.facesWithin, List.all_eq_false]
    exact ⟨last, hl, by simp [hm]⟩
  cases l with
  | some n => simp [Becomes, hg]
  | none => simp [Becomes, hw, hg]

/-- An explode limited to no times leaves every chain. -/
private theorem becomes_zero (d : Die) (c : Criterion) (ch : Chain) :
    Becomes d c (some 0) ch ch := by
  simp [Becomes]

/-- A chain grows exactly when it may still grow and its last die matches. -/
private theorem grows_live (body : List Int) (last : Int) (c : Criterion) :
    (Chain.live body last).grows c = c.matches last := by
  simp [Chain.grows, Chain.saturates, Chain.last]

/-- A tail of copies of one middle die is sorted. -/
private theorem pairwise_replicate (m : Nat) (μ : Int) :
    (List.replicate m μ).Pairwise (· ≤ ·) :=
  List.pairwise_replicate.mpr (.inr le_rfl)

/-- The shapes that a reroll by exact criteria makes of chains that a die
realizes are well formed and realized by the chains that it may make. -/
private theorem DieChains.rerollShapes_realized {d : Die}
    {Reach : Chain → Prop} {D : DieChains} (hR : D.Realizes d Reach)
    (hd : d.faces ≠ []) {criteria : List Criteria}
    {limit : Option (Int × Int)} (ho : Ordered limit) {cr : Criteria}
    (hcr : cr ∈ criteria) {c : Criterion} (hm : cr.Matches d c) :
    ∀ S' ∈ D.rerollShapes limit cr, S'.WellFormed D.faces ∧
      S'.Realized (nextReach d Reach (.reroll criteria limit)) := by
  have hF : D.faces = d.faces.toFinset := hR.faces
  intro S' hS'
  unfold DieChains.rerollShapes at hS'
  split at hS'
  · simp at hS'
  rename_i hnb
  obtain ⟨S, hS, hS'⟩ := List.mem_flatMap.mp hS'
  rcases List.mem_append.mp hS' with hst | hpr
  · cases limit with
    | none => simp [stays] at hst
    | some b =>
      obtain ⟨lo, hi⟩ := b
      by_cases hlo : lo ≤ 0
      · simp only [stays, hlo, decide_true, ↓reduceIte,
          List.mem_singleton] at hst
        subst hst
        obtain ⟨hw, T, hT⟩ := hR.shapes S' hS
        refine ⟨hw, T, fun body y hc => ?_⟩
        obtain ⟨body', hb', hreach⟩ := hT body y hc
        refine ⟨body'.mergeSort, hb'.trans (List.mergeSort_perm _ _).symm,
          _, hreach, cr, hcr, c, hm, some lo.toNat, ⟨lo, le_rfl, ho, rfl⟩, ?_⟩
        exact rerolled_self fun x _ => by simp [Int.toNat_eq_zero.mpr hlo]
      · simp [stays, hlo] at hst
  · cases limit with
    | none =>
      simp only [proceeds, ↓reduceIte, Option.isSome_none, Bool.false_eq_true,
        List.mem_singleton] at hpr
      subst hpr
      have hw : ¬d.facesWithin c = true := fun hw =>
        hnb ((blocked_iff hR hm none).mpr ⟨rfl, hw⟩)
      obtain ⟨y, hyF, hyc⟩ : ∃ y ∈ d.faces, c.matches y = false := by
        simpa [Die.facesWithin] using hw
      refine realized_map hR hcr hm (limit := none) (l := none) rfl
        (rerolled := c.matches)
        (P := fun y => y ∈ d.faces ∧ c.matches y = false)
        (fun ch ch' h => by
          simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
          exact h)
        (fun _ _ => rfl) (fun y hy => ?_) ⟨y, ?_⟩ Finset.sdiff_subset hS
      · obtain ⟨hyF, hyc⟩ := Finset.mem_sdiff.mp hy
        have hyF := List.mem_toFinset.mp (hF ▸ hyF)
        exact ⟨hyF, by simpa [(hm.mem hyF).2] using hyc⟩
      · refine Finset.mem_sdiff.mpr ⟨hF ▸ List.mem_toFinset.mpr hyF, ?_⟩
        simp [(hm.mem hyF).2, hyc]
    | some b =>
      obtain ⟨lo, hi⟩ := b
      by_cases hhi : 0 < hi
      · simp only [proceeds, hhi, decide_true, ↓reduceIte, Option.isSome_some,
          List.mem_singleton] at hpr
        subst hpr
        obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hd
        refine realized_map hR hcr hm (limit := some (lo, hi))
          (l := some hi.toNat) ⟨hi, ho, le_rfl, rfl⟩ (fun ch ch' h => h)
          (fun x _ => by simp [show 0 < hi.toNat by omega])
          (fun y hy => List.mem_toFinset.mp (hF ▸ hy))
          ⟨y, hF ▸ List.mem_toFinset.mpr hy⟩ subset_rfl hS
      · simp [proceeds, hhi] at hpr

/-- The shapes that an explode by exact criteria makes of chains that a die
realizes are well formed and realized by the chains that it may make. -/
private theorem DieChains.explodeShapes_realized {d : Die}
    {Reach : Chain → Prop} {D : DieChains} (hR : D.Realizes d Reach)
    (hd : d.faces ≠ []) {criteria : List Criteria}
    {limit : Option (Int × Int)} (ho : Ordered limit) {cr : Criteria}
    (hcr : cr ∈ criteria) {c : Criterion} (hm : cr.Matches d c) :
    ∀ S' ∈ D.explodeShapes limit cr, S'.WellFormed D.faces ∧
      S'.Realized (nextReach d Reach (.explode criteria limit)) := by
  have hF : D.faces = d.faces.toFinset := hR.faces
  have hface : ∀ {y}, y ∈ D.faces → y ∈ d.faces :=
    fun hy => List.mem_toFinset.mp (hF ▸ hy)
  intro S' hS'
  unfold DieChains.explodeShapes at hS'
  split at hS'
  · simp at hS'
  rename_i hnb
  obtain ⟨S, hS, hS'⟩ := List.mem_flatMap.mp hS'
  obtain ⟨hw, T, hT⟩ := hR.shapes S hS
  -- Every die of a shape is a face.
  have hlast : ∀ {y}, y ∈ S.last → y ∈ d.faces := fun hy => hface (hw.2.1 hy)
  -- The step that leaves a chain as it was.
  have hkeep : ∀ (l : Option Nat), Limits limit l → ∀ E ⊆ S.last,
      (∀ body y, y ∈ E → Becomes d c l (.live body y) (.live body y)) →
      ChainShape.Realized (nextReach d Reach (.explode criteria limit))
        { middles := S.middles, last := E } :=
    fun l hL E hE hb => realized_stays ⟨T, hT⟩ hE fun body y hy hreach =>
      ⟨_, hreach, cr, hcr, c, hm, l, hL, hb body y hy⟩
  simp only [DieChains.explodeShape] at hS'
  rcases List.mem_append.mp hS' with hst | hpr
  · -- The limit may leave every chain.
    cases limit with
    | none => simp [stays] at hst
    | some b =>
      obtain ⟨lo, hi⟩ := b
      by_cases hlo : lo ≤ 0
      · simp only [stays, hlo, decide_true, ↓reduceIte,
          List.mem_singleton] at hst
        subst hst
        exact ⟨hw, hkeep (some 0) ⟨lo, le_rfl, ho, by
            rw [Int.toNat_eq_zero.mpr hlo]⟩ _ subset_rfl
          (fun body y _ => becomes_zero d c _)⟩
      · simp [stays, hlo] at hst
  · -- The limit may extend a die.
    split at hpr
    swap
    · simp at hpr
    rename_i hp
    have hw' : ∀ l : Option Nat, Limits limit l → l = none →
        ¬d.facesWithin c = true := fun l hL hl hwc => by
      subst hl
      rw [limits_none hL] at hnb
      exact hnb ((blocked_iff hR hm none).mpr ⟨rfl, hwc⟩)
    rcases List.mem_append.mp hpr with hrest | hgrow
    · split at hrest
      swap
      · simp at hrest
      rename_i hrn
      rw [List.mem_singleton] at hrest
      subst hrest
      have hwr : ChainShape.WellFormed D.faces
          { middles := S.middles, last := S.last \ cr.certain } :=
        ⟨hrn, Finset.sdiff_subset.trans hw.2.1, hw.2.2⟩
      obtain ⟨l, hL⟩ := exists_limits_of_ordered ho
      refine ⟨hwr.new, ChainShape.Realized.new (fun r hr => (hw.2.2 r hr).1)
        (hkeep l hL _ Finset.sdiff_subset fun body y hy => ?_)⟩
      obtain ⟨hyS, hyc⟩ := Finset.mem_sdiff.mp hy
      exact becomes_stay (hlast hyS) (by
        simpa [(hm.mem (hlast hyS)).2] using hyc) l
    · split at hgrow
      swap
      · simp at hgrow
      rename_i hsn
      -- Every grown shape is well formed: its last run, the matched faces,
      -- holds the last die that it grows from.
      have hmid : (D.faces ∩ cr.possible).Nonempty :=
        hsn.mono (Finset.inter_subset_inter hw.2.1 subset_rfl)
      have hwg : ∀ (a : Nat) (b : ℕ∞) (E : Finset Int), E.Nonempty →
          E ⊆ D.faces → (S.grow (S.last ∩ cr.possible) (D.faces ∩ cr.possible)
            a b E).WellFormed D.faces := fun a b E hE hEF => by
        refine ChainShape.WellFormed.new ⟨hE, hEF, fun r hr => ?_⟩
        rcases List.mem_append.mp hr with hr | hr
        · exact hw.2.2 r hr
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
          rcases hr with rfl | rfl
          · exact ⟨hsn, Finset.inter_subset_left.trans hw.2.1⟩
          · exact ⟨hmid, Finset.inter_subset_left⟩
      -- Each middle die is a face that the criterion matches.
      have hμ : ∀ {μ}, μ ∈ D.faces ∩ cr.possible →
          μ ∈ d.faces ∧ c.matches μ = true := fun hμ => by
        obtain ⟨hμF, hμc⟩ := Finset.mem_inter.mp hμ
        exact ⟨hface hμF, (hm.mem (hface hμF)).1.mp hμc⟩
      have hrep : ∀ {m μ}, μ ∈ D.faces ∩ cr.possible →
          ∀ x ∈ List.replicate m μ, x ∈ d.faces ∧ c.matches x = true :=
        fun hm' x hx => List.eq_of_mem_replicate hx ▸ hμ hm'
      -- The last die of a chain that grows matches.
      have hstart : ∀ {s}, s ∈ S.last ∩ cr.possible →
          (Chain.live [] s).grows c = true ∧ s ∈ d.faces := fun hs => by
        obtain ⟨hsS, hsc⟩ := Finset.mem_inter.mp hs
        refine ⟨?_, hlast hsS⟩
        rw [grows_live]
        exact (hm.mem (hlast hsS)).1.mp hsc
      -- Each last die that ends a chain is a face that it does not match.
      have hend : ∀ {y}, y ∈ D.faces \ cr.certain →
          y ∈ d.faces ∧ c.matches y = false := fun hy => by
        obtain ⟨hyF, hyc⟩ := Finset.mem_sdiff.mp hy
        exact ⟨hface hyF, by simpa [(hm.mem (hface hyF)).2] using hyc⟩
      cases limit with
      | none =>
        rw [List.mem_singleton] at hgrow
        subst hgrow
        have hwc := hw' none rfl rfl
        obtain ⟨y, hyF, hyc⟩ : ∃ y ∈ d.faces, c.matches y = false := by
          simpa [Die.facesWithin] using hwc
        have hne : (D.faces \ cr.certain).Nonempty :=
          ⟨y, Finset.mem_sdiff.mpr ⟨hF ▸ List.mem_toFinset.mpr hyF,
            by simp [(hm.mem hyF).2, hyc]⟩⟩
        refine ⟨hwg _ _ _ hne Finset.sdiff_subset, realized_grow hw hT
          Finset.inter_subset_left hsn hmid ?_⟩
        intro body₀ s hreach hs m μ y _ hmμ hy
        have hgr : (Chain.live body₀ s).grows c = true := by
          rw [grows_live]
          simpa [grows_live] using (hstart hs).1
        refine ⟨_, hreach, cr, hcr, c, hm, none, rfl, ?_⟩
        simp only [Becomes, hwc, Bool.false_eq_true, ↓reduceIte, hgr]
        exact ⟨_, ⟨pairwise_replicate m μ, hrep hmμ, (hend hy).1, (hend hy).2⟩,
          rfl⟩
      | some b =>
        obtain ⟨lo, hi⟩ := b
        have hhi : 0 < hi := by simpa [proceeds] using hp
        -- A grown chain, under a limit that admits as many middle dice.
        have hlim : ∀ (n : Int) (m : Nat) (μ y : Int) (body₀ : List Int)
            (s : Int), lo ≤ n → n ≤ hi → m < n.toNat →
            (m + 1 < n.toNat → c.matches y = false) → y ∈ d.faces →
            μ ∈ D.faces ∩ cr.possible → s ∈ S.last ∩ cr.possible →
            Reach (.live body₀ s) →
            nextReach d Reach (.explode criteria (some (lo, hi)))
              (growChain (.live body₀ s) (List.replicate m μ, y)) := by
          intro n m μ y body₀ s hlo hn hmn hend' hy hmμ hs hreach
          have hgr : (Chain.live body₀ s).grows c = true := by
            rw [grows_live]
            simpa [grows_live] using (hstart hs).1
          refine ⟨_, hreach, cr, hcr, c, hm, some n.toNat, ⟨n, hlo, hn, rfl⟩,
            ?_⟩
          simp only [Becomes]
          rw [ite_eq_left_of_eq_true _ _ (eq_true ⟨by omega, hgr⟩)]
          exact ⟨_, ⟨pairwise_replicate m μ, hrep hmμ, hy,
            by simpa using hmn, by simpa using hend'⟩, rfl⟩
        rcases List.mem_append.mp hgrow with hshort | hspent
        · split at hshort
          swap
          · simp at hshort
          rename_i hc2
          rw [List.mem_singleton] at hshort
          subst hshort
          refine ⟨hwg _ _ _ hc2.2 Finset.sdiff_subset, realized_grow hw hT
            Finset.inter_subset_left hsn hmid ?_⟩
          intro body₀ s hreach hs m μ y hmc hmμ hy
          have hm2 : m ≤ hi.toNat - 2 := by
            rcases hmc with hmc | hmc | ⟨hmc, -⟩
            · simp only at hmc
              omega
            · simp only [Nat.cast_inj] at hmc
              omega
            · simp at hmc
          exact hlim hi m μ y body₀ s ho le_rfl (by omega)
            (fun _ => (hend hy).2) (hend hy).1 hmμ hs hreach
        · rw [List.mem_singleton] at hspent
          subst hspent
          obtain ⟨y₀, hy₀⟩ := List.exists_mem_of_ne_nil _ hd
          refine ⟨hwg _ _ _ ⟨y₀, hF ▸ List.mem_toFinset.mpr hy₀⟩ subset_rfl,
            realized_grow hw hT Finset.inter_subset_left hsn hmid ?_⟩
          intro body₀ s hreach hs m μ y hmc hmμ hy
          rcases hmc with hmc | hmc | ⟨hmc, -⟩
          · simp only at hmc
            exact hlim (Max.max lo 1) m μ y body₀ s (le_max_left _ _)
              (max_le ho (by omega)) (by omega) (fun h => by omega)
              (hface hy) hmμ hs hreach
          · simp only [Nat.cast_inj] at hmc
            exact hlim hi m μ y body₀ s ho le_rfl (by omega)
              (fun h => by omega) (hface hy) hmμ hs hreach
          · simp at hmc

/-- The chains that a die realizes are realized by more chains. -/
theorem DieChains.Realizes.mono {d : Die} {Reach Reach' : Chain → Prop}
    {D : DieChains} (h : D.Realizes d Reach)
    (hR : ∀ ch, Reach ch → Reach' ch) : D.Realizes d Reach' :=
  ⟨h.faces, h.covers, h.saturating,
    fun S hS => ⟨(h.shapes S hS).1, (h.shapes S hS).2.mono hR⟩,
    fun v hv => hR _ (h.saturated v hv)⟩

/--
The chains after the bounds of a clause that are exact for a die, which do
not coarsen them, are realized by the chains that the clause may make of the
chains that realized those before.

# Parameters
- `d`: The die.
- `Reach`: The chains that realize the chains before.
- `D`: The chains before.
- `b`: The bounds of the clause.

# Hypotheses
- `hR`: The chains before are realized.
- `hd`: The die has faces.
- `he`: The bounds are exact for the die.
- `hc`: The bounds do not coarsen the shapes.
-/
theorem DieChains.Realizes.step {d : Die} {Reach : Chain → Prop}
    {D : DieChains} {b : ClauseBounds} (hR : D.Realizes d Reach)
    (hd : d.faces ≠ []) (he : b.Exact d) (hc : (b.apply D).2 = false) :
    (b.apply D).1.Realizes d (nextReach d Reach b) := by
  cases b with
  | reroll criteria limit =>
    obtain ⟨hne, hex, ho⟩ := he
    refine ⟨hR.faces, hR.covers, hR.saturating, ?_, fun v hv => ?_⟩
    · refine uniteShapes_preserves
        (P := fun S => S.WellFormed D.faces ∧
          S.Realized (nextReach d Reach (.reroll criteria limit)))
        (fun hS hT h => ChainShape.unite_realized hS hT h) ?_ hc
      intro S hS
      obtain ⟨cr, hcr, hS⟩ := List.mem_flatMap.mp hS
      obtain ⟨c, hm⟩ := hex cr hcr
      exact DieChains.rerollShapes_realized hR hd ho hcr hm S hS
    · obtain ⟨cr, hcr⟩ := List.exists_mem_of_ne_nil _ hne
      obtain ⟨c, hm⟩ := hex cr hcr
      obtain ⟨l, hL⟩ := exists_limits_of_ordered ho
      exact ⟨_, hR.saturated v hv, cr, hcr, c, hm, l, hL,
        rerollBecomes_saturated_self d c l v⟩
  | explode criteria limit =>
    obtain ⟨hne, hex, ho⟩ := he
    refine ⟨hR.faces, hR.covers, hR.saturating, ?_, fun v hv => ?_⟩
    · refine uniteShapes_preserves
        (P := fun S => S.WellFormed D.faces ∧
          S.Realized (nextReach d Reach (.explode criteria limit)))
        (fun hS hT h => ChainShape.unite_realized hS hT h) ?_ hc
      intro S hS
      obtain ⟨cr, hcr, hS⟩ := List.mem_flatMap.mp hS
      obtain ⟨c, hm⟩ := hex cr hcr
      exact DieChains.explodeShapes_realized hR hd ho hcr hm S hS
    · rcases Finset.mem_union.mp hv with hv | hv
      · obtain ⟨cr, hcr⟩ := List.exists_mem_of_ne_nil _ hne
        obtain ⟨c, hm⟩ := hex cr hcr
        obtain ⟨l, hL⟩ := exists_limits_of_ordered ho
        exact ⟨_, hR.saturated v hv, cr, hcr, c, hm, l, hL,
          becomes_saturated_self d c l v⟩
      · -- A chain that saturates: the explode has no limit, and matches
        -- every face, and some chain may still grow.
        obtain ⟨_, hmem, hv⟩ := mem_unionAll.mp hv
        obtain ⟨cr, hcr, rfl⟩ := List.mem_map.mp hmem
        obtain ⟨c, hm⟩ := hex cr hcr
        unfold DieChains.saturates at hv
        split_ifs at hv with hcond
        swap
        · simp at hv
        obtain ⟨hnone, hext, hcov⟩ := hcond
        rw [hR.saturating, Finset.mem_singleton] at hv
        subst hv
        have hlim : limit = none := Option.isNone_iff_eq_none.mp hnone
        subst hlim
        have hwc : d.facesWithin c = true := by
          rw [hm.facesWithin, ← hR.faces]
          simpa [hR.covers] using hcov
        obtain ⟨S, hS, -⟩ := List.any_eq_true.mp hext
        obtain ⟨hw, T, hT⟩ := hR.shapes S hS
        obtain ⟨body, y, hc'⟩ := ChainShape.exists_corner hw T
        obtain ⟨body', -, hreach⟩ := hT body y hc'
        refine ⟨_, hreach, cr, hcr, c, hm, none, rfl, ?_⟩
        simp [Becomes, hwc, Chain.saturate]
  | drop =>
    exact hR.mono fun ch h => ⟨ch, h, rfl⟩

/-- Running the bounds of clauses from some chains and some coarsening flag
answers the chains that they run to, and the flag or theirs. -/
private theorem runBounds_foldl (bs : List ClauseBounds) :
    ∀ (D : DieChains) (f : Bool),
      bs.foldl (fun acc b => let next := b.apply acc.1;
          (next.1, acc.2 || next.2)) (D, f) =
        ((runBounds D bs).1, f || (runBounds D bs).2) := by
  induction bs with
  | nil => intro D f; simp [runBounds]
  | cons b bs ih =>
    intro D f
    simp only [runBounds, List.foldl_cons, Bool.false_or]
    rw [ih, ih]
    simp [Bool.or_assoc]

/-- The bounds of clauses coarsen the shapes if the first or the rest do. -/
private theorem runBounds_cons_flag (D : DieChains) (b : ClauseBounds)
    (bs : List ClauseBounds) :
    (runBounds D (b :: bs)).2 =
      ((b.apply D).2 || (runBounds (b.apply D).1 bs).2) := by
  have := runBounds_foldl bs (b.apply D).1 (b.apply D).2
  simp only [runBounds, List.foldl_cons, Bool.false_or] at this ⊢
  rw [this]

/--
The chains after the bounds of clauses that are exact for a die, which never
coarsen them, are realized by the chains that the clauses may make of those
that realized the chains before.

# Parameters
- `d`: The die.
- `Reach`: The chains that realize the chains before.
- `D`: The chains before.
- `bs`: The bounds of the clauses.

# Hypotheses
- `hR`: The chains before are realized.
- `hd`: The die has faces.
- `he`: The bounds of each clause are exact for the die.
- `hc`: The bounds never coarsen the shapes.
-/
theorem DieChains.Realizes.run {d : Die} {Reach : Chain → Prop}
    {D : DieChains} {bs : List ClauseBounds} (hR : D.Realizes d Reach)
    (hd : d.faces ≠ []) (he : ∀ b ∈ bs, b.Exact d)
    (hc : (runBounds D bs).2 = false) :
    (runBounds D bs).1.Realizes d
      fun ch' => ∃ ch, Reach ch ∧ ReachesChain d bs ch ch' := by
  induction bs generalizing Reach D with
  | nil => exact hR.mono fun ch h => ⟨ch, h, rfl⟩
  | cons b bs ih =>
    rw [runBounds_cons_flag, Bool.or_eq_false_iff] at hc
    rw [runBounds_cons]
    refine (ih (hR.step hd (he b List.mem_cons_self) hc.1)
      (fun b' hb' => he b' (List.mem_cons_of_mem _ hb')) hc.2).mono ?_
    rintro ch'' ⟨ch₁, ⟨ch, hch, hstep⟩, hrest⟩
    exact ⟨ch, hch, ch₁, hstep, hrest⟩

/-- The chains of a die just rolled are realized by its chains of one die. -/
theorem DieChains.realizes_ofDie {d : Die} (hd : d.faces ≠ []) :
    (DieChains.ofDie d).Realizes d fun ch => ∃ x ∈ d.faces, ch = .single x := by
  obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hd
  have hne : d.faces.toFinset ≠ ∅ :=
    Finset.ne_empty_of_mem (List.mem_toFinset.mpr hx)
  have hnew : ChainShape.new [] d.faces.toFinset =
      { middles := [], last := d.faces.toFinset } := by
    simp [ChainShape.new, mergeRuns]
  refine ⟨rfl, fun _ => rfl, rfl, fun S hS => ?_, fun v hv => by
    simp [DieChains.ofDie, DieChains.rolled] at hv⟩
  simp only [DieChains.ofDie, DieChains.rolled, hne, ↓reduceIte,
    List.mem_singleton, hnew] at hS
  subst hS
  refine ⟨⟨⟨x, List.mem_toFinset.mpr hx⟩, subset_rfl, by simp⟩, 0,
    fun body y ⟨hy, parts, hp, hb⟩ => ?_⟩
  cases hp
  exact ⟨[], by simpa using hb, y, List.mem_toFinset.mp hy, rfl⟩

/-- Every chain that clauses whose bounds stand for them may make of a chain
of one face of a die is one that a record of the die may keep, and the chains
after their bounds cover it. -/
private theorem reachesChain_covers {d : Die}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) :
    ∀ {bs : List ClauseBounds} {D : DieChains} {ch ch' : Chain},
      D.faces = d.faces.toFinset →
      (∀ cr, D.covers cr = decide (D.faces ⊆ cr.possible)) →
      D.saturating = {d.saturatedValue} →
      Fine d ch → D.Covers ch → ReachesChain d bs ch ch' →
        Fine d ch' ∧ (runBounds D bs).1.Covers ch'
  | [], D, ch, ch', _, _, _, hf, hc, h => by
    rw [show ch' = ch from h]
    exact ⟨hf, hc⟩
  | b :: bs, D, ch, ch'', hF', hcov, hsat, hf, hc, ⟨ch', hstep, hrest⟩ => by
    have hD : D.Represents d := ⟨fun y hy => hF' ▸ List.mem_toFinset.mpr hy,
      hsat ▸ Finset.mem_singleton_self _⟩
    -- The exact criteria stand for their criterion.
    have hadm : ∀ cr c, cr.Matches d c → Admits D.covers d cr c :=
      fun cr c hm => ⟨fun y hy h => (hm.mem hy).1.mpr h,
        fun y hy h => (hm.mem hy).2.mp h,
        fun hw => by
          rw [hcov, hF']
          exact decide_eq_true (hm.facesWithin.mp hw)⟩
    have hfix := b.apply_fixed D
    rw [runBounds_cons]
    refine reachesChain_covers hF (hfix.1.trans hF')
      (fun cr => by rw [hfix.2.1, hfix.1]; exact hcov cr)
      (hfix.2.2.trans hsat) ?_ ?_ hrest
    · cases b with
      | reroll criteria limit =>
        obtain ⟨cr, -, c, -, l, -, hb⟩ := hstep
        exact Fine.rerollBecomes hF hf hb
      | explode criteria limit =>
        obtain ⟨cr, -, c, -, l, -, hb⟩ := hstep
        exact Fine.becomes hF hf hb
      | drop => exact (show ch' = ch from hstep) ▸ hf
    · cases b with
      | reroll criteria limit =>
        obtain ⟨cr, hcr, c, hm, l, hL, hb⟩ := hstep
        exact D.reroll_covers hD hcr (hadm cr c hm) hL hf hc hb
      | explode criteria limit =>
        obtain ⟨cr, hcr, c, hm, l, hL, hb⟩ := hstep
        exact D.explode_covers hD hcr (hadm cr c hm) hL hf hc hb
      | drop => exact (show ch' = ch from hstep) ▸ hc

/--
The bounds of a rolling record are exact within the budget, for each die
that the record may roll: if the bounds of each clause enumerate exact
criteria, under bounds of their limits in order, and never coarsen the shapes,
then the totals of the chains after them bound the totals of the chains that
the clauses may make of a chain of one face of the die exactly in their hull,
so that its least and greatest are the totals of some such chains
(`Exact.hull`). Exact relative to the criteria and limits, each of which, in
each clause, is taken to be possible, as the bounds of their operands admit.

# Parameters
- `d`: The die.
- `bs`: The bounds of the clauses.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hd`: The die has faces.
- `he`: The bounds of each clause are exact for the die.
- `hc`: The bounds never coarsen the shapes.
-/
theorem DieChains.run_exact {d : Die} {bs : List ClauseBounds}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) (hd : d.faces ≠ [])
    (he : ∀ b ∈ bs, b.Exact d)
    (hc : (runBounds (DieChains.ofDie d) bs).2 = false) :
    Exact (runBounds (DieChains.ofDie d) bs).1.totals
      {v | ∃ x ∈ d.faces, ∃ ch, ReachesChain d bs (.single x) ch ∧
        ch.total = v} := by
  have hR := (DieChains.realizes_ofDie hd).run hd he hc
  refine ⟨?_, fun v hv => ?_⟩
  · rintro _ ⟨x, hx, ch, hch, rfl⟩
    have hx' : x ∈ (DieChains.ofDie d).faces := List.mem_toFinset.mpr hx
    have hne : (DieChains.ofDie d).faces ≠ ∅ := Finset.ne_empty_of_mem hx'
    have hcov : (DieChains.ofDie d).Covers (.single x) := by
      refine ⟨.new [] (DieChains.ofDie d).faces, ?_, ChainShape.has_new
        ⟨hx', [], .nil, by simp⟩⟩
      simp [DieChains.ofDie, DieChains.rolled, hd]
    have hfine : Fine d (.single x) :=
      ⟨⟨by simp, by simpa using hF x hx⟩, fun _ y hy => by
        simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
          List.mem_singleton] at hy
        exact hy ▸ hx⟩
    exact .of_mem (DieChains.total_mem_totals
      (reachesChain_covers hF rfl (fun _ => rfl) rfl hfine hcov hch).2)
  · -- A reachable chain of one face, after the clauses.
    have hreach : ∀ ch, (∃ ch₀, (∃ x ∈ d.faces, ch₀ = .single x) ∧
        ReachesChain d bs ch₀ ch) → ∃ x ∈ d.faces,
          ReachesChain d bs (.single x) ch := by
      rintro ch ⟨_, ⟨x, hx, rfl⟩, h⟩
      exact ⟨x, hx, h⟩
    rcases mem_totals_iff.mp hv with hv | ⟨S, hS, h₁, h₂⟩
    · obtain ⟨x, hx, h⟩ := hreach _ (hR.saturated v hv)
      exact ⟨⟨v, ⟨x, hx, _, h, rfl⟩, le_rfl⟩, ⟨v, ⟨x, hx, _, h, rfl⟩, le_rfl⟩⟩
    · obtain ⟨hw, hr⟩ := hR.shapes S hS
      obtain ⟨⟨ch₁, hc₁, he₁⟩, ⟨ch₂, hc₂, he₂⟩⟩ :=
        ChainShape.totals_realized hw hr
      obtain ⟨x₁, hx₁, h₁'⟩ := hreach _ hc₁
      obtain ⟨x₂, hx₂, h₂'⟩ := hreach _ hc₂
      exact ⟨⟨_, ⟨x₁, hx₁, ch₁, h₁', rfl⟩, he₁ ▸ h₁⟩,
        ⟨_, ⟨x₂, hx₂, ch₂, h₂', rfl⟩, he₂ ▸ h₂⟩⟩

/-! ## Counting dice -/

/--
The most kept dice of a chain of a shape: its last die, and the most dice of
each run. Mirrors `ChainShape::dice` in `evaluator.rs`, which saturates at
`u64::MAX`; here, `⊤` stands for any number. It bounds the kept dice of every
chain of the shape (`ChainShape.length_le_dice`).

# Parameters
- `S`: The shape.

# Returns
The number of dice, or `⊤` for any number.

# Examples
```lean
#guard (ChainShape.dice
    { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1} }) == 4
#guard (ChainShape.dice
    { middles := [{ results := {6}, least := 0, greatest := ⊤ }],
      last := {1} }) == ⊤
```
-/
def ChainShape.dice (S : ChainShape) : ℕ∞ :=
  S.middles.foldl (fun dice run => dice + run.greatest) 1

/--
The most kept dice of a chain of a shape that may show one of some results:
its last die, if it may, and the most dice of each run that may. Mirrors
`ChainShape::matching` in `evaluator.rs`. It bounds the dice of every chain of
the shape that show one of the results (`ChainShape.countP_le_matching`).

# Parameters
- `S`: The shape.
- `R`: The results.

# Returns
The number of dice, or `⊤` for any number.

# Examples
```lean
-- A six, then up to two more sixes, and a last die of one to five: at most
-- three sixes, and one five.
#guard (ChainShape.matching
    { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1, 2, 3, 4, 5} } {6}) == 3
#guard (ChainShape.matching
    { middles := [{ results := {6}, least := 1, greatest := 3 }],
      last := {1, 2, 3, 4, 5} } {5}) == 1
```
-/
def ChainShape.matching (S : ChainShape) (R : Finset Int) : ℕ∞ :=
  (S.middles.filter fun run => decide (run.results ∩ R).Nonempty).foldl
    (fun dice run => dice + run.greatest)
    (if (S.last ∩ R).Nonempty then 1 else 0)

/-- Adding the most dice of each run adds their sum. -/
private theorem foldl_add_greatest :
    ∀ (runs : List Segment) (init : ℕ∞),
      runs.foldl (fun dice run => dice + run.greatest) init =
        init + (runs.map (·.greatest)).sum
  | [], init => by simp
  | run :: runs, init => by
    rw [List.foldl_cons, foldl_add_greatest runs, List.map_cons,
      List.sum_cons, add_assoc]

/-- The kept dice of a chain that may still grow are its last die and its
body. -/
private theorem length_dice_live (body : List Int) (last : Int) :
    (Chain.live body last).dice.length = body.length + 1 := by
  simp [Chain.dice, List.length_mergeSort]

/--
A chain of a shape holds no more kept dice than the shape's most.

# Parameters
- `S`: The shape.
- `ch`: The chain.

# Hypotheses
- `h`: The chain has the shape.
-/
theorem ChainShape.length_le_dice {S : ChainShape} {ch : Chain}
    (h : S.Has ch) : (ch.dice.length : ℕ∞) ≤ S.dice := by
  cases ch with
  | saturated v => exact h.elim
  | live body last =>
    obtain ⟨-, parts, hp, hb⟩ := h
    rw [length_dice_live, ChainShape.dice, foldl_add_greatest, hb.length_eq,
      Nat.cast_add, Nat.cast_one, add_comm]
    exact add_le_add le_rfl (holds_sum hp).2

/-- The dice that runs hold that show one of some results number no more
than the most dice of the runs that may show one. -/
private theorem countP_flatten_le {R : Finset Int} :
    ∀ {runs : List Segment} {parts : List (List Int)},
      List.Forall₂ Segment.Holds runs parts →
        ((parts.flatten.countP fun x => decide (x ∈ R) : ℕ) : ℕ∞) ≤
          ((runs.filter fun run => decide (run.results ∩ R).Nonempty).map
            (·.greatest)).sum
  | _, _, .nil => by simp
  | run :: _, part :: _, .cons hr hl => by
    rw [List.flatten_cons, List.countP_append, Nat.cast_add, List.filter_cons]
    by_cases hm : (run.results ∩ R).Nonempty
    · simp only [hm, decide_true, ↓reduceIte, List.map_cons, List.sum_cons]
      exact add_le_add ((Nat.cast_le.mpr List.countP_le_length).trans hr.2.1)
        (countP_flatten_le hl)
    · simp only [hm, decide_false, Bool.false_eq_true, ↓reduceIte]
      have h0 : part.countP (fun x => decide (x ∈ R)) = 0 :=
        List.countP_eq_zero.mpr fun x hx hxR =>
          hm ⟨x, Finset.mem_inter.mpr ⟨hr.2.2 x hx, of_decide_eq_true hxR⟩⟩
      rw [h0, Nat.cast_zero, zero_add]
      exact countP_flatten_le hl

/--
A chain of a shape holds no more kept dice that show one of some results than
the shape's most that may.

# Parameters
- `S`: The shape.
- `R`: The results.
- `ch`: The chain.

# Hypotheses
- `h`: The chain has the shape.
-/
theorem ChainShape.countP_le_matching {S : ChainShape} {R : Finset Int}
    {ch : Chain} (h : S.Has ch) :
    ((ch.dice.countP fun x => decide (x ∈ R) : ℕ) : ℕ∞) ≤ S.matching R := by
  cases ch with
  | saturated v => exact h.elim
  | live body last =>
    obtain ⟨hl, parts, hp, hb⟩ := h
    rw [Chain.dice, (List.mergeSort_perm _ _).countP_eq, List.countP_cons,
      (hb.countP_eq _), ChainShape.matching, foldl_add_greatest, Nat.cast_add,
      add_comm]
    refine add_le_add ?_ (countP_flatten_le hp)
    by_cases hR : last ∈ R
    · have hn : (S.last ∩ R).Nonempty := ⟨last, Finset.mem_inter.mpr ⟨hl, hR⟩⟩
      simp [hR, hn]
    · simp [hR]

/--
The most times that a reroll or explode under some bounds of its limit may
roll for one die or chain: the greatest limit, or `0` if it is negative, or,
without limit, any number. Mirrors `limit.map_or(u64::MAX, |limit|
dice(limit.max))` in `evaluator.rs`.

# Parameters
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The number of times, or `⊤` for any number.
-/
def times : Option (Int × Int) → ℕ∞
  | some (_, hi) => hi.toNat
  | none => ⊤

/--
The most dice that a reroll by any of some criteria may roll for one chain
still kept: for each criteria that do not block it, if the limit may reroll
some die, the most dice of a chain of each shape that may match, each rolled
the most times. Mirrors the dice that `DieChains::reroll` in `evaluator.rs`
answers. They bound the dice that the reroll draws for every chain that the
chains cover (`DieChains.rerollDraws_le`).

# Parameters
- `D`: The chains.
- `criteria`: The criteria, each of which the predicate may resolve to, or one
  that stands for every one.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The number of dice, or `⊤` for any number.
-/
def DieChains.rerollRolls (D : DieChains) (criteria : List Criteria)
    (limit : Option (Int × Int)) : ℕ∞ :=
  (criteria.flatMap fun cr =>
    if !D.blocked limit cr && proceeds limit then
      D.live.map fun S => S.matching cr.possible * times limit
    else []).foldr emax 0

/--
The most dice that an explode by any of some criteria may roll for one chain
still kept: for each criteria that do not block it, if the limit may extend
some chain, and some criterion may match the last die of some chain, the most
times. Mirrors the extensions that `DieChains::explode` in `evaluator.rs`
answers. They bound the dice that the explode draws for every chain that the
chains cover (`DieChains.explodeDraws_le`).

# Parameters
- `D`: The chains.
- `criteria`: The criteria, each of which the predicate may resolve to, or one
  that stands for every one.
- `limit`: The bounds of the limit, or `none` for no limit.

# Returns
The number of dice, or `⊤` for any number.
-/
def DieChains.extensions (D : DieChains) (criteria : List Criteria)
    (limit : Option (Int × Int)) : ℕ∞ :=
  (criteria.map fun cr =>
    if !D.blocked limit cr && proceeds limit && D.mayExtend cr then times limit
    else 0).foldr emax 0

/--
Whether some criterion of some criteria may match some kept die of a chain
still kept that no saturated die decides. Mirrors `DieChains::may_match` in
`evaluator.rs`. If it may not, then a reroll leaves every chain that the
chains cover as it was (`DieChains.rerollBecomes_of_not_mayMatch`).

# Parameters
- `D`: The chains.
- `cr`: The criteria.
-/
def DieChains.mayMatch (D : DieChains) (cr : Criteria) : Bool :=
  D.live.any fun S => decide (0 < S.matching cr.possible)

/--
The most kept dice of a chain still kept that no saturated die decides, of a
record rolled, by the chains of each die that it may roll: the most of any
shape, and at least `1`. Mirrors `RollingRecordBounds::length` in
`evaluator.rs`, which answers `0` for a record never rolled. It bounds the
kept dice of every such chain that the chains of some die cover
(`le_lengthOf`), and so of every such chain still kept after any sequence of
clauses (`DieChains.run_length`).

# Parameters
- `dice`: The chains of each die.

# Returns
The number of dice, or `⊤` for any number.
-/
def lengthOf (dice : List DieChains) : ℕ∞ :=
  (dice.flatMap fun D => D.live.map ChainShape.dice).foldr emax 1

/-- The greatest of some numbers, and a start, is no less than any of them. -/
private theorem le_foldr_emax_of_mem {l : List ℕ∞} {a : ℕ∞} (init : ℕ∞)
    (h : a ∈ l) : a ≤ l.foldr emax init := by
  induction l with
  | nil => simp at h
  | cons b l ih =>
    rw [List.foldr_cons, emax_eq]
    rcases List.mem_cons.mp h with rfl | h
    · exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

/--
The chains of some die bound the kept dice of every chain that they cover that
no saturated die decides.

# Parameters
- `dice`: The chains of each die.
- `D`: The chains of one die.
- `ch`: The chain.

# Hypotheses
- `hD`: `D` is among the chains of each die.
- `hc`: The chains cover the chain.
- `hs`: No saturated die decides the chain.
-/
theorem le_lengthOf {dice : List DieChains} {D : DieChains} {ch : Chain}
    (hD : D ∈ dice) (hc : D.Covers ch) (hs : ch.saturates = false) :
    (ch.dice.length : ℕ∞) ≤ lengthOf dice := by
  cases ch with
  | saturated v => simp [Chain.saturates] at hs
  | live body last =>
    obtain ⟨S, hS, h⟩ := hc
    exact (ChainShape.length_le_dice h).trans (le_foldr_emax_of_mem _
      (List.mem_flatMap.mpr ⟨D, hD, List.mem_map_of_mem hS⟩))

/--
The bounds of a rolling record bound the kept dice of every chain still kept
that no saturated die decides, after any sequence of rerolls, explodes,
drops, and keeps: if the bounds of each clause stand for it, then every such
chain of every record that the clauses may lead to from a record that the
chains tracked holds no more kept dice than the length of the chains of each
die, among them the chains after the bounds.

# Parameters
- `D`: The chains.
- `d`: The die.
- `bs`: The bounds of the clauses.
- `cls`: The clauses.
- `r`, `r'`: The record, and a record that the clauses may lead to.
- `dice`: The chains of each die.

# Hypotheses
- `hF`: The faces of the die lie within the `i32` range.
- `hD`: The chains stand for the die.
- `hb`: The bounds of each clause stand for it.
- `ht`: The chains track the record.
- `hr`: The clauses may lead to `r'`.
- `hm`: The chains after the bounds are among those of each die.
-/
theorem DieChains.run_length {D : DieChains} {d : Die}
    {bs : List ClauseBounds} {cls : List Clause} {r r' : Record}
    {dice : List DieChains}
    (hF : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) (hD : D.Represents d)
    (hb : List.Forall₂ (ClauseBounds.Admits D.covers d) bs cls)
    (ht : D.Tracks d r) (hr : Reaches r cls r')
    (hm : (runBounds D bs).1 ∈ dice) :
    ∀ ch ∈ r'.keptChains, ch.saturates = false →
      (ch.dice.length : ℕ∞) ≤ lengthOf dice :=
  fun ch hch hs => le_lengthOf hm
    ((DieChains.run_tracks hF hD hb ht hr).covers ch hch) hs

/-! ## Counting draws -/

/--
Redraw a die while it matches, as `redraw` in `XdySpec/Semantics.lean` does,
counting the draws: a die that the criterion matches is replaced by a draw from
the die, one draw more, and any other stays.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `p`: The law of the die before, and of its draws so far.

# Returns
The law of the die after, and of its draws.
-/
noncomputable def countedRedraw (c : Criterion) (die : PMF Int)
    (p : PMF (Int × Nat)) : PMF (Int × Nat) :=
  p.bind fun xk => if c.matches xk.1 then die.map (·, xk.2 + 1) else PMF.pure xk

/--
The law of the replacement of a die that a reroll limited to `limit` times
crosses out, beside the number of draws that it takes, as `reroll_within` in
`primitives.rs` meters them: one draw, and then one more for each redraw.
Mirrors `limitedReplacement` in `XdySpec/Semantics.lean`, which forgets the
draws (`countedReplacement_fst`), and never takes more than `limit` draws
(`countedReplacement_le`).

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The number of times that each die may be rerolled, which is
  positive if any die is rerolled.

# Returns
The law of the replacement, and of its draws.
-/
noncomputable def countedReplacement (c : Criterion) (die : PMF Int)
    (limit : Nat) : PMF (Int × Nat) :=
  (limit - 1).repeat (countedRedraw c die) (die.map (·, 1))

/-- Counting the draws of a redraw changes nothing else. -/
private theorem map_fst_countedRedraw (c : Criterion) (die : PMF Int)
    (p : PMF (Int × Nat)) :
    (countedRedraw c die p).map Prod.fst = redraw c die (p.map Prod.fst) := by
  unfold countedRedraw redraw
  rw [PMF.map_bind, PMF.bind_map]
  congr 1
  funext xk
  simp only [Function.comp]
  split
  · rw [PMF.map_comp]
    exact PMF.map_id die
  · rw [PMF.pure_map]

/--
The law of the counted replacement of a die, its draws forgotten, is the
specification's.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The limit.
-/
theorem countedReplacement_fst (c : Criterion) (die : PMF Int) (limit : Nat) :
    (countedReplacement c die limit).map Prod.fst =
      limitedReplacement c die limit := by
  unfold countedReplacement limitedReplacement
  induction limit - 1 with
  | zero =>
    simp only [Nat.repeat, PMF.map_comp]
    exact PMF.map_id die
  | succ n ih => rw [Nat.repeat, Nat.repeat, map_fst_countedRedraw, ih]

/--
The replacement of a die that a reroll limited to `limit` times crosses out
takes no more than `limit` draws, or one if `limit` is `0`, which rerolls
nothing.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The limit.
- `xk`: A replacement, and its draws.

# Hypotheses
- `h`: The counted replacement may answer `xk`.
-/
theorem countedReplacement_le {c : Criterion} {die : PMF Int} {limit : Nat}
    {xk : Int × Nat} (h : xk ∈ (countedReplacement c die limit).support) :
    xk.2 ≤ Max.max limit 1 := by
  have key : ∀ n : Nat, ∀ xk ∈ (n.repeat (countedRedraw c die)
      (die.map (·, 1))).support, xk.2 ≤ n + 1 := by
    intro n
    induction n with
    | zero =>
      intro xk hxk
      obtain ⟨y, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hxk
      exact le_rfl
    | succ n ih =>
      intro xk hxk
      rw [Nat.repeat, countedRedraw, PMF.mem_support_bind_iff] at hxk
      obtain ⟨xk₀, h₀, hxk⟩ := hxk
      have := ih xk₀ h₀
      split at hxk
      · obtain ⟨y, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hxk
        exact Nat.succ_le_succ this
      · rw [PMF.mem_support_pure_iff] at hxk
        subst hxk
        omega
  have := key _ xk h
  omega

/--
The law of the replacement of a die that an unlimited reroll crosses out, when
its criterion does not match every face of the die, beside the number of draws
that it takes: draws from the die while they match, and then one that does
not. The number of matching draws is `geometric`, in the probability that a
draw matches, `matchedFaces`, and the last draw falls as the die given that it
does not match, independently of their number, as `unlimitedExtension` in
`XdySpec/Semantics.lean` argues of the draws of an explode. Its draws
forgotten, it is the specification's replacement
(`countedUnlimited_fst`).

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`, so some draw ends the reroll.

# Returns
The law of the replacement, and of its draws.
-/
noncomputable def countedUnlimited (c : Criterion) (d : Die)
    (h : ¬d.facesWithin c = true) : PMF (Int × Nat) :=
  (geometric (matchedFaces d c) (matchedFaces_lt_one h)).bind fun k =>
    (PMF.ofMultiset _ (unmatched_ne_zero h)).map (·, k + 1)

/--
The law of the counted replacement of a die that an unlimited reroll crosses
out, its draws forgotten, is the specification's.

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`.
-/
theorem countedUnlimited_fst (c : Criterion) (d : Die)
    (h : ¬d.facesWithin c = true) :
    (countedUnlimited c d h).map Prod.fst =
      PMF.ofMultiset _ (unmatched_ne_zero h) := by
  unfold countedUnlimited
  rw [PMF.map_bind]
  simp only [PMF.map_comp]
  exact (congrArg _ (funext fun _ => PMF.map_id _)).trans (PMF.bind_const _ _)

/--
Reroll each die of a chain, as `rerollChain` in `XdySpec/Semantics.lean` does,
counting the draws: each rerolled die takes the draws of its replacement, and
every other none. A saturated chain draws nothing. Its draws forgotten, it is
`rerollChain` (`countedRerollChain_fst`).

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result, and of its
  draws.
- `ch`: The chain.

# Returns
The law of the chain after the reroll, and of the draws that it took.
-/
noncomputable def countedRerollChain (rerolled : Int → Bool)
    (replacement : PMF (Int × Nat)) : Chain → PMF (Chain × Nat)
  | .live body last =>
    let draw (x : Int) : PMF (Int × Nat) :=
      if rerolled x then replacement else PMF.pure (x, 0)
    (body.foldl (fun (p : PMF (List Int × Nat)) x => p.bind fun bn =>
        (draw x).map fun yk => ((yk.1 :: bn.1).mergeSort, bn.2 + yk.2))
      (PMF.pure ([], 0))).bind fun bn =>
      (draw last).map fun yk => (.live bn.1 yk.1, bn.2 + yk.2)
  | c => PMF.pure (c, 0)

/--
The law of a chain after a counted reroll, its draws forgotten, is the law of
the chain after the specification's reroll.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `R`: The law of the replacement of a rerolled result, and of its draws.
- `ch`: The chain.
-/
theorem countedRerollChain_fst (rerolled : Int → Bool) (R : PMF (Int × Nat))
    (ch : Chain) :
    (countedRerollChain rerolled R ch).map Prod.fst =
      rerollChain rerolled (R.map Prod.fst) ch := by
  have hdraw : ∀ x, (if rerolled x then R else PMF.pure (x, 0)).map
      Prod.fst = if rerolled x then R.map Prod.fst else PMF.pure x := by
    intro x
    split <;> simp [PMF.pure_map]
  have key : ∀ (xs : List Int) (p : PMF (List Int × Nat)),
      (xs.foldl (fun (p : PMF (List Int × Nat)) x => p.bind fun bn =>
          (if rerolled x then R else PMF.pure (x, 0)).map fun yk =>
            ((yk.1 :: bn.1).mergeSort, bn.2 + yk.2)) p).map Prod.fst =
        xs.foldl (fun (p : PMF (List Int)) x => p.bind fun b =>
          (if rerolled x then R.map Prod.fst else PMF.pure x).map fun y =>
            (y :: b).mergeSort) (p.map Prod.fst) := by
    intro xs
    induction xs with
    | nil => intro p; rfl
    | cons x xs ih =>
      intro p
      rw [List.foldl_cons, List.foldl_cons, ih]
      congr 1
      rw [PMF.map_bind, PMF.bind_map]
      congr 1
      funext bn
      rw [PMF.map_comp, ← hdraw, Function.comp_apply, PMF.map_comp]
      rfl
  cases ch with
  | saturated v => simp [countedRerollChain, rerollChain, PMF.pure_map]
  | live body last =>
    simp only [countedRerollChain, rerollChain]
    rw [PMF.map_bind, ← PMF.pure_map Prod.fst ([], 0), ← key, PMF.bind_map]
    congr 1
    funext bn
    rw [PMF.map_comp, ← hdraw]
    exact (PMF.map_comp Prod.fst _ _).symm

/-- A counted reroll of a chain draws no more than the draws of a replacement
for each die that it rerolls. -/
private theorem countedRerollChain_le {rerolled : Int → Bool}
    {R : PMF (Int × Nat)} {B : ℕ∞} (hR : ∀ yk ∈ R.support, (yk.2 : ℕ∞) ≤ B)
    {ch : Chain} {chk : Chain × Nat}
    (h : chk ∈ (countedRerollChain rerolled R ch).support) :
    (chk.2 : ℕ∞) ≤ (ch.dice.countP rerolled : ℕ∞) * B := by
  -- Each die draws no more than a replacement does, if rerolled, and
  -- otherwise none.
  have hdraw : ∀ x, ∀ yk ∈ (if rerolled x then R else PMF.pure (x, 0)).support,
      (yk.2 : ℕ∞) ≤ (([x].countP rerolled : ℕ) : ℕ∞) * B := by
    intro x yk hyk
    by_cases hx : rerolled x = true
    · simp only [hx, ↓reduceIte] at hyk
      simpa [hx] using hR yk hyk
    · simp only [hx, Bool.false_eq_true, ↓reduceIte,
        PMF.mem_support_pure_iff] at hyk
      simp [hyk]
  have key : ∀ (xs : List Int) (p : PMF (List Int × Nat)) (a : ℕ∞),
      (∀ bn ∈ p.support, (bn.2 : ℕ∞) ≤ a) →
      ∀ bn ∈ (xs.foldl (fun (p : PMF (List Int × Nat)) x => p.bind fun bn =>
          (if rerolled x then R else PMF.pure (x, 0)).map fun yk =>
            ((yk.1 :: bn.1).mergeSort, bn.2 + yk.2)) p).support,
        (bn.2 : ℕ∞) ≤ a + (xs.countP rerolled : ℕ∞) * B := by
    intro xs
    induction xs with
    | nil => intro p a hp bn hbn; simpa using hp bn hbn
    | cons x xs ih =>
      intro p a hp bn hbn
      rw [List.foldl_cons] at hbn
      refine (ih _ (a + (([x].countP rerolled : ℕ) : ℕ∞) * B) ?_ bn hbn).trans
        (le_of_eq ?_)
      · intro bn' hbn'
        simp only [PMF.mem_support_bind_iff, PMF.mem_support_map_iff] at hbn'
        obtain ⟨bn₀, h₀, yk, hyk, rfl⟩ := hbn'
        rw [Nat.cast_add]
        exact add_le_add (hp bn₀ h₀) (hdraw x yk hyk)
      · simp only [List.countP_cons, List.countP_nil]
        push_cast
        ring
  cases ch with
  | saturated v =>
    simp only [countedRerollChain, PMF.mem_support_pure_iff] at h
    simp [h]
  | live body last =>
    simp only [countedRerollChain, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff] at h
    obtain ⟨bn, hbn, yk, hyk, rfl⟩ := h
    have hb := key body _ 0 (fun bn h => by
      simp only [PMF.mem_support_pure_iff] at h
      simp [h]) bn hbn
    rw [Chain.dice, (List.mergeSort_perm _ _).countP_eq, List.countP_cons,
      Nat.cast_add]
    refine (add_le_add hb (hdraw last yk hyk)).trans (le_of_eq ?_)
    simp only [List.countP_cons, List.countP_nil]
    push_cast
    ring

/--
The law of the number of dice that a reroll draws for one chain still kept,
as `reroll_within` in `primitives.rs` meters them: those of each replacement,
counted as `countedReplacement` or `countedUnlimited` counts them, of each die
that it rerolls, as `countedRerollChain` rerolls them, whose chains, their
draws forgotten, are those of the specification's `reroll`
(`countedRerollChain_fst`, `countedReplacement_fst`,
`countedUnlimited_fst`). An unlimited reroll that matches every face of the
die draws nothing. The dice that `DieChains.rerollRolls` answers bound them
(`DieChains.rerollDraws_le`).

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The limit, or `none` for no limit.
- `ch`: The chain.

# Returns
The law of the number of dice drawn.
-/
noncomputable def rerollDraws (d : Die) (c : Criterion) :
    Option Nat → Chain → PMF Nat
  | some n, ch =>
    (countedRerollChain (fun x => 0 < n && c.matches x)
      (countedReplacement c (customDie d.faces) n) ch).map Prod.snd
  | none, ch =>
    if h : d.facesWithin c = true then PMF.pure 0
    else (countedRerollChain c.matches (countedUnlimited c d h) ch).map
      Prod.snd

/--
The law of the extension of a chain whose last die an explode limited to
`limit` times matches, as `extension` in `XdySpec/Semantics.lean` draws it,
beside the number of draws that it takes, as `explode_within` in
`primitives.rs` meters them: one draw, and then one more for each further
draw. Its draws forgotten, it is `extension` (`countedExtension_fst`), and it
never takes more than `limit` draws (`countedExtension_le`).

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The number of times that each chain may be exploded, which is
  positive if any chain grows.

# Returns
The law of the dice drawn but the last, sorted, and of the last die drawn, and
of the number of draws.
-/
noncomputable def countedExtension (c : Criterion) (die : PMF Int)
    (limit : Nat) : PMF ((List Int × Int) × Nat) :=
  (limit - 1).repeat
    (fun p => p.bind fun tk =>
      if c.matches tk.1.2 then
        die.map fun y => (((tk.1.2 :: tk.1.1).mergeSort, y), tk.2 + 1)
      else PMF.pure tk)
    (die.map fun y => (([], y), 1))

/--
The law of the counted extension of a chain, its draws forgotten, is the
specification's.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The limit.
-/
theorem countedExtension_fst (c : Criterion) (die : PMF Int) (limit : Nat) :
    (countedExtension c die limit).map Prod.fst = extension c die limit := by
  unfold countedExtension extension
  induction limit - 1 with
  | zero => simp only [Nat.repeat, PMF.map_comp]; rfl
  | succ n ih =>
    rw [Nat.repeat, Nat.repeat, ← ih, PMF.map_bind, PMF.bind_map]
    congr 1
    funext tk
    simp only [Function.comp]
    split
    · rw [PMF.map_comp]
      rfl
    · rw [PMF.pure_map]

/--
The extension of a chain that an explode limited to `limit` times grows takes
no more than `limit` draws, or one if `limit` is `0`, which grows nothing.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The limit.
- `tk`: An extension, and its draws.

# Hypotheses
- `h`: The counted extension may answer `tk`.
-/
theorem countedExtension_le {c : Criterion} {die : PMF Int} {limit : Nat}
    {tk : (List Int × Int) × Nat}
    (h : tk ∈ (countedExtension c die limit).support) :
    tk.2 ≤ Max.max limit 1 := by
  have key : ∀ n : Nat, ∀ tk ∈ (n.repeat
      (fun (p : PMF ((List Int × Int) × Nat)) => p.bind fun tk =>
        if c.matches tk.1.2 then
          die.map fun y => (((tk.1.2 :: tk.1.1).mergeSort, y), tk.2 + 1)
        else PMF.pure tk)
      (die.map fun y => (([], y), 1))).support, tk.2 ≤ n + 1 := by
    intro n
    induction n with
    | zero =>
      intro tk htk
      obtain ⟨y, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp htk
      exact le_rfl
    | succ n ih =>
      intro tk htk
      rw [Nat.repeat, PMF.mem_support_bind_iff] at htk
      obtain ⟨tk₀, h₀, htk⟩ := htk
      have := ih tk₀ h₀
      split at htk
      · obtain ⟨y, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp htk
        exact Nat.succ_le_succ this
      · rw [PMF.mem_support_pure_iff] at htk
        subst htk
        omega
  have := key _ tk h
  omega

/--
The law of the extension of a chain whose last die an unlimited explode
matches, when the explode does not match every face of the die, as
`unlimitedExtension` in `XdySpec/Semantics.lean` draws it, beside the number of
draws that it takes: its matching draws, and the last. Its draws forgotten, it
is `unlimitedExtension` (`countedUnlimitedExtension_fst`).

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`, so some draw ends the chain.

# Returns
The law of the dice drawn but the last, sorted, and of the last die drawn, and
of the number of draws.
-/
noncomputable def countedUnlimitedExtension (c : Criterion) (d : Die)
    (h : ¬d.facesWithin c = true) : PMF ((List Int × Int) × Nat) :=
  (geometric (matchedFaces d c) (matchedFaces_lt_one h)).bind fun k =>
    ((k.repeat
      (fun (p : PMF (List Int)) => p.bind fun drawn =>
        (customDie (d.faces.filter c.matches)).map fun x =>
          (x :: drawn).mergeSort)
      (PMF.pure [])).bind fun drawn =>
      (PMF.ofMultiset _ (unmatched_ne_zero h)).map (drawn, ·)).map (·, k + 1)

/--
The law of the counted extension of a chain that an unlimited explode grows,
its draws forgotten, is the specification's.

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`.
-/
theorem countedUnlimitedExtension_fst (c : Criterion) (d : Die)
    (h : ¬d.facesWithin c = true) :
    (countedUnlimitedExtension c d h).map Prod.fst =
      unlimitedExtension c d h := by
  unfold countedUnlimitedExtension unlimitedExtension
  rw [PMF.map_bind]
  simp only [PMF.map_comp]
  exact congrArg _ (funext fun _ => PMF.map_id _)

/--
The law of the number of dice that an explode draws for one chain still kept,
as `explode_within` in `primitives.rs` meters them: those of its extension,
counted as `countedExtension` or `countedUnlimitedExtension` counts them, if
the explode grows it, whose extensions, their draws forgotten, are those of
the specification's `explode` (`countedExtension_fst`,
`countedUnlimitedExtension_fst`), and otherwise none. An unlimited explode
that matches every face of the die saturates the chain instead, drawing
nothing. The extensions that `DieChains.extensions` answers bound them
(`DieChains.explodeDraws_le`).

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The limit, or `none` for no limit.
- `ch`: The chain.

# Returns
The law of the number of dice drawn.
-/
noncomputable def explodeDraws (d : Die) (c : Criterion) :
    Option Nat → Chain → PMF Nat
  | some n, ch =>
    if 0 < n && ch.grows c then
      (countedExtension c (customDie d.faces) n).map Prod.snd
    else PMF.pure 0
  | none, ch =>
    if h : d.facesWithin c = true then PMF.pure 0
    else if ch.grows c then
      (countedUnlimitedExtension c d h).map Prod.snd
    else PMF.pure 0

/-- The dice of a chain that a reroll by a criterion rerolls number no more
than the most dice of a chain of its shape that criteria that stand for the
criterion may match. -/
private theorem countP_le_matching_of_admits {D : DieChains} {d : Die}
    {cr : Criteria} {c : Criterion} {rerolled : Int → Bool}
    (ha : Admits D.covers d cr c)
    (hr : ∀ x, rerolled x = true → c.matches x = true) {ch : Chain}
    (hf : Fine d ch) {S : ChainShape} (hS : S.Has ch) :
    ((ch.dice.countP rerolled : ℕ) : ℕ∞) ≤ S.matching cr.possible := by
  have hs : ch.saturates = false := by
    cases ch with
    | live => rfl
    | saturated => exact hS.elim
  refine (Nat.cast_le.mpr (List.countP_mono_left fun x hx hx' =>
    decide_eq_true (ha.possible x (hf.2 hs x hx) (hr x hx')))).trans
    (ChainShape.countP_le_matching hS)

/--
The dice that a reroll draws for every chain that some chains cover number no
more than the most that the chains find that it may roll for one, if they
stand for the die, some criteria of the reroll stand for its criterion, and
the bounds of the limit admit its limit.

# Parameters
- `D`: The chains.
- `d`: The die.
- `criteria`: The criteria of the reroll.
- `cr`: The criteria that stand for the criterion.
- `c`: The criterion.
- `limit`: The bounds of the limit.
- `l`: The limit.
- `ch`: The chain.
- `k`: A number of dice that the reroll may draw for it.

# Hypotheses
- `hD`: The chains stand for the die.
- `hcr`: The criteria are among those of the reroll.
- `ha`: The criteria stand for the criterion.
- `hL`: The bounds admit the limit.
- `hf`: The chain is one that a record of the die may keep.
- `hc`: The chains cover the chain.
- `hk`: The reroll may draw `k` dice for the chain.
-/
theorem DieChains.rerollDraws_le {D : DieChains} {d : Die}
    {criteria : List Criteria} {cr : Criteria} {c : Criterion}
    {limit : Option (Int × Int)} {l : Option Nat} {ch : Chain} {k : Nat}
    (hD : D.Represents d) (hcr : cr ∈ criteria) (ha : Admits D.covers d cr c)
    (hL : Limits limit l) (hf : Fine d ch) (hc : D.Covers ch)
    (hk : k ∈ (rerollDraws d c l ch).support) :
    (k : ℕ∞) ≤ D.rerollRolls criteria limit := by
  cases ch with
  | saturated v =>
    -- A reroll leaves a saturated chain alone, drawing nothing.
    have h0 : rerollDraws d c l (.saturated v) = PMF.pure 0 := by
      cases l with
      | some n => simp [rerollDraws, countedRerollChain, PMF.pure_map]
      | none =>
        by_cases hw : d.facesWithin c = true
        · simp only [rerollDraws, hw, ↓reduceDIte]
        · simp [rerollDraws, hw, countedRerollChain, PMF.pure_map]
    rw [h0, PMF.mem_support_pure_iff] at hk
    simp [hk]
  | live body last =>
    obtain ⟨S, hS, hSc⟩ := hc
    -- The chains find that the reroll may roll each die of a chain of the
    -- shape that it may match the most times.
    have hmem : D.blocked limit cr = false → proceeds limit = true →
        S.matching cr.possible * times limit ≤ D.rerollRolls criteria limit :=
      fun hb hp => le_foldr_emax_of_mem _ (List.mem_flatMap.mpr ⟨cr, hcr, by
        simp only [hb, hp, Bool.not_false, Bool.and_self, ↓reduceIte]
        exact List.mem_map_of_mem hS⟩)
    cases l with
    | some n =>
      obtain ⟨lo, hi, j, rfl, -, hhi, rfl⟩ := limits_some hL
      simp only [rerollDraws, PMF.mem_support_map_iff] at hk
      obtain ⟨chk, hchk, rfl⟩ := hk
      have hle := countedRerollChain_le (B := (Max.max j.toNat 1 : ℕ))
        (fun yk hyk => Nat.cast_le.mpr (countedReplacement_le hyk)) hchk
      have hcount := countP_le_matching_of_admits ha
        (rerolled := fun x => 0 < j.toNat && c.matches x)
        (fun x hx => by simp only [Bool.and_eq_true] at hx; exact hx.2) hf hSc
      rcases Nat.eq_zero_or_pos ((Chain.live body last).dice.countP
          fun x => 0 < j.toNat && c.matches x) with h0 | hpos
      · rw [h0, Nat.cast_zero, zero_mul] at hle
        exact hle.trans (by simp)
      · -- Some die is rerolled, so the limit is positive.
        obtain ⟨x, -, hx⟩ := List.countP_pos_iff.mp hpos
        have hj : 0 < j.toNat := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hx
          exact hx.1
        have hp : proceeds (some (lo, hi)) = true := by
          simp only [proceeds, decide_eq_true_eq]
          omega
        refine hle.trans ((mul_le_mul' hcount ?_).trans
          (hmem (by simp [DieChains.blocked]) hp))
        simp only [times, Nat.cast_le]
        omega
    | none =>
      have hlim := limits_none hL
      subst hlim
      by_cases hw : d.facesWithin c = true
      · simp only [rerollDraws, hw, ↓reduceDIte, PMF.mem_support_pure_iff] at hk
        simp [hk]
      · simp only [rerollDraws, hw, Bool.false_eq_true, ↓reduceDIte,
          PMF.mem_support_map_iff] at hk
        obtain ⟨chk, hchk, rfl⟩ := hk
        have hle := countedRerollChain_le (B := ⊤) (fun _ _ => le_top) hchk
        have hcount := countP_le_matching_of_admits ha
          (rerolled := c.matches) (fun _ hx => hx) hf hSc
        exact hle.trans ((mul_le_mul' hcount le_rfl).trans
          (hmem (not_blocked hD ha hw none) rfl))

/--
The dice that an explode draws for every chain that some chains cover number
no more than the most that the chains find that it may roll for one, if they
stand for the die, some criteria of the explode stand for its criterion, and
the bounds of the limit admit its limit.

# Parameters
- `D`: The chains.
- `d`: The die.
- `criteria`: The criteria of the explode.
- `cr`: The criteria that stand for the criterion.
- `c`: The criterion.
- `limit`: The bounds of the limit.
- `l`: The limit.
- `ch`: The chain.
- `k`: A number of dice that the explode may draw for it.

# Hypotheses
- `hD`: The chains stand for the die.
- `hcr`: The criteria are among those of the explode.
- `ha`: The criteria stand for the criterion.
- `hL`: The bounds admit the limit.
- `hf`: The chain is one that a record of the die may keep.
- `hc`: The chains cover the chain.
- `hk`: The explode may draw `k` dice for the chain.
-/
theorem DieChains.explodeDraws_le {D : DieChains} {d : Die}
    {criteria : List Criteria} {cr : Criteria} {c : Criterion}
    {limit : Option (Int × Int)} {l : Option Nat} {ch : Chain} {k : Nat}
    (hD : D.Represents d) (hcr : cr ∈ criteria) (ha : Admits D.covers d cr c)
    (hL : Limits limit l) (hf : Fine d ch) (hc : D.Covers ch)
    (hk : k ∈ (explodeDraws d c l ch).support) :
    (k : ℕ∞) ≤ D.extensions criteria limit := by
  -- The explode draws only for a chain that it grows, whose last die some
  -- criterion may match, so that the chains find that it may extend some.
  have hmem : ch.grows c = true → D.blocked limit cr = false →
      proceeds limit = true → times limit ≤ D.extensions criteria limit := by
    intro hg hb hp
    cases ch with
    | saturated v => simp [Chain.grows, Chain.saturates] at hg
    | live body last =>
      obtain ⟨S, hS, hSc⟩ := hc
      have hlast : last ∈ cr.possible := ha.possible last
        (hf.2 rfl last (by simp [Chain.dice, (List.mergeSort_perm _ _).mem_iff]))
        (by simpa [Chain.grows, Chain.saturates, Chain.last] using hg)
      have he : D.mayExtend cr = true := List.any_eq_true.mpr
        ⟨S, hS, decide_eq_true ⟨last, Finset.mem_inter.mpr ⟨hSc.1, hlast⟩⟩⟩
      exact le_foldr_emax_of_mem _ (List.mem_map.mpr ⟨cr, hcr, by
        simp [hb, hp, he]⟩)
  cases l with
  | some n =>
    obtain ⟨lo, hi, j, rfl, -, hhi, rfl⟩ := limits_some hL
    rw [explodeDraws] at hk
    split at hk
    · rename_i hg
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hg
      rw [PMF.mem_support_map_iff] at hk
      obtain ⟨tk, htk, rfl⟩ := hk
      have hle := countedExtension_le htk
      have hp : proceeds (some (lo, hi)) = true := by
        simp only [proceeds, decide_eq_true_eq]
        omega
      refine le_trans ?_ (hmem hg.2 (by simp [DieChains.blocked]) hp)
      simp only [times, Nat.cast_le]
      omega
    · rw [PMF.mem_support_pure_iff] at hk
      simp [hk]
  | none =>
    have hlim := limits_none hL
    subst hlim
    by_cases hw : d.facesWithin c = true
    · simp only [explodeDraws, hw, ↓reduceDIte, PMF.mem_support_pure_iff] at hk
      simp [hk]
    · by_cases hg : ch.grows c = true
      · exact le_top.trans_eq (hmem hg (not_blocked hD ha hw none) rfl
          |> top_le_iff.mp).symm
      · simp only [explodeDraws, hw, hg, Bool.false_eq_true, ↓reduceDIte,
          ↓reduceIte, PMF.mem_support_pure_iff] at hk
        simp [hk]

/--
A reroll leaves every chain that some chains cover as it was, if no criterion
of criteria that stand for its criterion may match any kept die of a chain of
theirs that no saturated die decides, as the Rust finds them: a chain that may
still grow holds no die that the reroll matches, so it rerolls none, nor
crosses out the chain, and the reroll leaves every saturated chain alone.

# Parameters
- `D`: The chains.
- `d`: The die.
- `cr`: The criteria that stand for the criterion.
- `c`: The criterion.
- `l`: The limit.
- `ch`, `ch'`: The chain, and a chain.

# Hypotheses
- `hm`: No criterion may match any kept die of a chain of the chains.
- `ha`: The criteria stand for the criterion.
- `hf`: The chain is one that a record of the die may keep.
- `hc`: The chains cover the chain.
-/
theorem DieChains.rerollBecomes_of_not_mayMatch {D : DieChains} {d : Die}
    {cr : Criteria} {c : Criterion} {l : Option Nat} {ch ch' : Chain}
    (hm : D.mayMatch cr = false) (ha : Admits D.covers d cr c)
    (hf : Fine d ch) (hc : D.Covers ch) :
    RerollBecomes d c l ch ch' ↔ ch' = ch := by
  cases ch with
  | saturated v =>
    cases l with
    | some n => exact Iff.rfl
    | none =>
      by_cases hw : d.facesWithin c = true
      · simp [RerollBecomes, hw, Chain.saturates]
      · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
        exact Iff.rfl
  | live body last =>
    obtain ⟨S, hS, hSc⟩ := hc
    -- No die of the chain matches the criterion.
    have h0 : S.matching cr.possible = 0 := by
      have := List.any_eq_false.mp hm S hS
      simpa using this
    have hno : ∀ x ∈ last :: body, c.matches x = false := by
      intro x hx
      by_contra hx'
      have hpos := countP_le_matching_of_admits ha (rerolled := c.matches)
        (fun _ h => h) hf hSc
      rw [h0, nonpos_iff_eq_zero, Nat.cast_eq_zero, List.countP_eq_zero] at hpos
      exact hpos x ((List.mergeSort_perm _ _).mem_iff.mpr hx)
        (by simpa using hx')
    have hsorted := mergeSort_eq_self hf.1.1
    -- A reroll that rerolls no die of the chain leaves it.
    have hstay : ∀ {rerolled : Int → Bool} {P : Int → Prop},
        (∀ x ∈ last :: body, rerolled x = false) →
        (Rerolled rerolled P (.live body last) ch' ↔
          ch' = .live body last) := by
      intro rerolled P hr
      constructor
      · rintro ⟨ys, y, hys, hy, rfl⟩
        rw [forall₂_redraw_stays (fun x hx => hr x (List.mem_cons_of_mem _ hx))
          hys, hsorted]
        simp only [Redraw, hr last List.mem_cons_self, Bool.false_eq_true,
          ↓reduceIte] at hy
        rw [hy]
      · rintro rfl
        have := rerolled_self (P := P) hr
        rwa [hsorted] at this
    cases l with
    | some n =>
      exact hstay fun x hx => by simp [hno x hx]
    | none =>
      by_cases hw : d.facesWithin c = true
      · -- Every face matches, so the last die does.
        have hlast : last ∈ d.faces := hf.2 rfl last
          (by simp [Chain.dice, (List.mergeSort_perm _ _).mem_iff])
        have := List.all_eq_true.mp hw last hlast
        rw [hno last List.mem_cons_self] at this
        exact absurd this Bool.false_ne_true
      · simp only [RerollBecomes, hw, Bool.false_eq_true, ↓reduceIte]
        exact hstay hno


end Bounds

end Xdy.Spec
