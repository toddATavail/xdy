import Xdy.Dist

/-!
# Laws of finite distributions

The weight and total of every distribution that `Dist` builds, as sums over
its entries, `d.weights.toList`: the outcomes it holds, each once, with their
weights. Proofs about the oracle can then reason about lists of entries rather
than about the hash map beneath them, as the proof that the oracle meets its
specification does, dividing each weight by the total.

Every operation of `Dist` is a fold that adds weight to one outcome at a time,
so one law, `weight_foldl_add`, gives the weights of `pure`, `uniform` and
`map`, and, applied once per branch, of `bind`. `bind` scales the branches up
to the least common multiple `l` of their totals; its laws say only that `l` is
positive and that every branch's total divides it, which is all that a proof
needs to cancel the scaling. Every operation also keeps every weight positive,
as `Dist` promises, so that the outcomes held are exactly the possible ones.

The laws need `LawfulBEq` outcomes, so that the hash map's lemmas apply: an
outcome is found only under itself. `Record` and `State` decide their equality
for this reason.
-/

namespace Xdy

open Std (HashMap)

namespace Dist

/-! ## Lists of entries -/

/--
Summing the weights of the entries whose outcome is `b` gives `0` if no entry
has that outcome.

# Parameters
- `xs`: The entries.
- `b`: The outcome.

# Hypotheses
- `h`: No entry has outcome `b`.
-/
private theorem sum_beq_eq_zero {α : Type} [BEq α] [LawfulBEq α]
    (xs : List (α × Nat)) (b : α) (h : ∀ p ∈ xs, p.1 ≠ b) :
    (xs.map fun p => if p.1 == b then p.2 else 0).sum = 0 := by
  induction xs with
  | nil => rfl
  | cons p xs ih =>
    simp [h p (by simp), ih fun q hq => h q (by simp [hq])]

/--
Summing the weights of the entries whose outcome is `b` gives the weight of
the one entry that has it, if the entries have distinct outcomes.

# Parameters
- `xs`: The entries.
- `b`: The outcome.
- `v`: Its weight.

# Hypotheses
- `distinct`: No two entries have the same outcome.
- `mem`: `b` has weight `v` among the entries.
-/
private theorem sum_beq_of_mem {α : Type} [BEq α] [LawfulBEq α]
    (xs : List (α × Nat)) (b : α) (v : Nat)
    (distinct : xs.Pairwise fun p q => (p.1 == q.1) = false)
    (mem : (b, v) ∈ xs) :
    (xs.map fun p => if p.1 == b then p.2 else 0).sum = v := by
  induction xs with
  | nil => cases mem
  | cons p xs ih =>
    rw [List.pairwise_cons] at distinct
    rw [List.map_cons, List.sum_cons]
    rcases List.mem_cons.mp mem with rfl | mem
    · rw [sum_beq_eq_zero xs _ fun q hq => by
        have := distinct.1 q hq; simp at this; exact Ne.symm this]
      simp
    · have := distinct.1 _ mem
      simp at this
      simp [this, ih distinct.2 mem]

/--
Split the total weight of some entries into that of the entries with outcome
`a` and that of the rest.

# Parameters
- `xs`: The entries.
- `a`: The outcome.
-/
private theorem sum_snd_split {α : Type} [BEq α] [LawfulBEq α]
    (xs : List (α × Nat)) (a : α) :
    (xs.map (·.2)).sum =
      (xs.map fun p => if p.1 == a then p.2 else 0).sum +
        ((xs.filter fun p => ¬(a == p.1)).map (·.2)).sum := by
  induction xs with
  | nil => rfl
  | cons p xs ih =>
    by_cases h : p.1 = a
    · simp [h, ih]; omega
    · have : ¬(a = p.1) := fun e => h e.symm
      simp [h, this, ih]; omega

/--
Folding addition over the weights of some entries adds their sum.

# Parameters
- `xs`: The entries.
- `n`: The start of the fold.
-/
private theorem foldl_add_snd {α : Type} (xs : List (α × Nat)) (n : Nat) :
    xs.foldl (fun t p => t + p.2) n = n + (xs.map (·.2)).sum := by
  induction xs generalizing n with
  | nil => simp
  | cons p xs ih => simp [ih]; omega

/--
Folding a list onto the front of an accumulator reverses it.

# Parameters
- `xs`: The list.
- `f`: What each element contributes.
- `acc`: The accumulator.
-/
private theorem foldl_cons_eq {γ δ : Type} (xs : List γ) (f : γ → δ)
    (acc : List δ) :
    xs.foldl (fun bs x => f x :: bs) acc = (xs.map f).reverse ++ acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih => simp [ih]

/--
A common factor comes out of a sum.

# Parameters
- `xs`: The list.
- `f`: The terms.
- `s`: The factor.
-/
private theorem sum_map_mul {γ : Type} (xs : List γ) (f : γ → Nat) (s : Nat) :
    (xs.map fun x => s * f x).sum = s * (xs.map f).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Nat.mul_add]

variable {α β : Type} [BEq α] [Hashable α] [LawfulBEq α]
  [BEq β] [Hashable β] [LawfulBEq β]

/-! ## Weights and totals as sums over entries -/

/-- Every weight is positive, as `Dist` promises: an impossible outcome is
absent rather than held with weight `0`. Every operation keeps this. -/
def Positive (d : Dist α) : Prop := ∀ p ∈ d.weights.toList, 0 < p.2

/--
The weight of an outcome is the sum of the weights of the entries with that
outcome, of which there is at most one.

# Parameters
- `d`: The distribution.
- `b`: The outcome.
-/
theorem weight_eq_sum (d : Dist α) (b : α) :
    d.weight b =
      (d.weights.toList.map fun p => if p.1 == b then p.2 else 0).sum := by
  rw [weight, HashMap.getD_eq_getD_getElem?]
  cases h : d.weights[b]? with
  | none =>
    rw [Option.getD_none, sum_beq_eq_zero]
    intro p hp heq
    have := HashMap.mem_toList_iff_getElem?_eq_some.mp hp
    rw [heq, h] at this
    cases this
  | some v =>
    rw [Option.getD_some, sum_beq_of_mem _ _ _ HashMap.distinct_keys_toList
      (HashMap.mem_toList_iff_getElem?_eq_some.mpr h)]

omit [LawfulBEq α] in
/--
The total is the sum of the weights of the entries.

# Parameters
- `d`: The distribution.
-/
theorem total_eq_sum (d : Dist α) :
    d.total = (d.weights.toList.map (·.2)).sum := by
  rw [total, HashMap.fold_eq_foldl_toList, foldl_add_snd, Nat.zero_add]

/--
An entry of a distribution gives the weight of its outcome.

# Parameters
- `d`: The distribution.
- `e`: The entry.

# Hypotheses
- `he`: `e` is an entry of `d`.
-/
theorem weight_of_mem (d : Dist α) {e : α × Nat} (he : e ∈ d.weights.toList) :
    d.weight e.1 = e.2 := by
  rw [weight, HashMap.getD_eq_getD_getElem?,
    HashMap.mem_toList_iff_getElem?_eq_some.mp he, Option.getD_some]

/--
The entries of a distribution have distinct outcomes.

# Parameters
- `d`: The distribution.
-/
theorem nodup_outcomes (d : Dist α) :
    (d.weights.toList.map (·.1)).Nodup := by
  have := HashMap.distinct_keys_toList (m := d.weights)
  rw [List.Nodup, List.pairwise_map]
  exact this.imp fun h e => by simp [e] at h

/-! ## Adding weight -/

/--
Adding weight to an outcome adds it to that outcome's weight alone.

# Parameters
- `d`: The distribution.
- `a`: The outcome given weight.
- `b`: The outcome whose weight is read.
- `w`: The weight added.
-/
theorem weight_add (d : Dist α) (a b : α) (w : Nat) :
    (d.add a w).weight b = d.weight b + if a == b then w else 0 := by
  simp only [add, weight, HashMap.getD_insert]
  by_cases h : a = b
  · subst h; simp
  · simp [h]

/--
Adding weight to an outcome adds it to the total.

# Parameters
- `d`: The distribution.
- `a`: The outcome.
- `w`: The weight added.
-/
theorem total_add (d : Dist α) (a : α) (w : Nat) :
    (d.add a w).total = d.total + w := by
  rw [total_eq_sum, total_eq_sum, sum_snd_split d.weights.toList a,
    ← weight_eq_sum, add,
    ((HashMap.toList_insert_perm (m := d.weights)).map (·.2)).sum_nat]
  simp; omega

/--
Adding positive weight keeps every weight positive.

# Parameters
- `d`: The distribution.
- `a`: The outcome.
- `w`: The weight added.

# Hypotheses
- `hd`: Every weight of `d` is positive.
- `hw`: The weight added is positive.
-/
theorem positive_add (d : Dist α) (a : α) (w : Nat) (hd : d.Positive)
    (hw : 0 < w) : (d.add a w).Positive := by
  intro p hp
  rcases List.mem_cons.mp
    ((HashMap.toList_insert_perm (m := d.weights)).mem_iff.mp hp) with rfl | hp
  · simp; omega
  · exact hd p (List.mem_filter.mp hp).1

omit [LawfulBEq α] in
/--
The empty distribution gives every outcome weight `0`.

# Parameters
- `b`: The outcome.
-/
theorem weight_empty (b : α) : ({} : Dist α).weight b = 0 :=
  HashMap.getD_empty

omit [LawfulBEq α] in
/-- The empty distribution has total `0`. -/
theorem total_empty : ({} : Dist α).total = 0 := by
  rw [total_eq_sum]; simp [HashMap.toList_empty]

omit [LawfulBEq α] in
/-- The empty distribution has no weights, so every one of them is
positive. -/
theorem positive_empty : ({} : Dist α).Positive := by
  intro p hp; simp [HashMap.toList_empty] at hp

/--
Adding weight once per element of a list adds, to each outcome, the weights
of the elements that give it.

# Parameters
- `xs`: The elements.
- `g`: The outcome that each element gives weight.
- `h`: The weight that each element gives.
- `init`: The distribution to add to.
- `b`: The outcome whose weight is read.
-/
theorem weight_foldl_add {γ : Type} (xs : List γ) (g : γ → α) (h : γ → Nat)
    (init : Dist α) (b : α) :
    (xs.foldl (fun e x => e.add (g x) (h x)) init).weight b =
      init.weight b + (xs.map fun x => if g x == b then h x else 0).sum := by
  induction xs generalizing init with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons, ih, weight_add]; simp; omega

/--
Adding weight once per element of a list adds every element's weight to the
total.

# Parameters
- `xs`: The elements.
- `g`: The outcome that each element gives weight.
- `h`: The weight that each element gives.
- `init`: The distribution to add to.
-/
theorem total_foldl_add {γ : Type} (xs : List γ) (g : γ → α) (h : γ → Nat)
    (init : Dist α) :
    (xs.foldl (fun e x => e.add (g x) (h x)) init).total =
      init.total + (xs.map h).sum := by
  induction xs generalizing init with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons, ih, total_add]; simp; omega

/--
Adding positive weight once per element of a list keeps every weight
positive.

# Parameters
- `xs`: The elements.
- `g`: The outcome that each element gives weight.
- `h`: The weight that each element gives.
- `init`: The distribution to add to.

# Hypotheses
- `hinit`: Every weight of `init` is positive.
- `hh`: Every element gives positive weight.
-/
theorem positive_foldl_add {γ : Type} (xs : List γ) (g : γ → α)
    (h : γ → Nat) (init : Dist α) (hinit : init.Positive)
    (hh : ∀ x ∈ xs, 0 < h x) :
    (xs.foldl (fun e x => e.add (g x) (h x)) init).Positive := by
  induction xs generalizing init with
  | nil => exact hinit
  | cons x xs ih =>
    exact ih _ (positive_add _ _ _ hinit (hh x (by simp)))
      fun y hy => hh y (by simp [hy])

/-! ## Pure, uniform and map -/

/--
A certain outcome has weight `1`, and every other `0`.

# Parameters
- `a`: The certain outcome.
- `b`: The outcome whose weight is read.
-/
theorem weight_pure (a b : α) :
    (pure a).weight b = if a == b then 1 else 0 := by
  rw [pure, weight_add, weight_empty, Nat.zero_add]

/--
A certain outcome has total `1`.

# Parameters
- `a`: The certain outcome.
-/
theorem total_pure (a : α) : (pure a).total = 1 := by
  rw [pure, total_add, total_empty]

/--
A certain outcome has positive weight.

# Parameters
- `a`: The certain outcome.
-/
theorem positive_pure (a : α) : (pure a).Positive :=
  positive_add _ _ _ positive_empty Nat.one_pos

/--
Each listed outcome has weight equal to the number of times it is listed.

# Parameters
- `as`: The outcomes.
- `b`: The outcome whose weight is read.
-/
theorem weight_uniform (as : List α) (b : α) :
    (uniform as).weight b = as.count b := by
  have h := weight_foldl_add as (fun a => a) (fun _ => 1) {} b
  rw [uniform, h, weight_empty, Nat.zero_add]
  clear h
  induction as with
  | nil => rfl
  | cons a as ih =>
    simp only [List.map_cons, List.sum_cons, ih, List.count_cons]; omega

/--
The total of a uniform draw is the number of outcomes listed.

# Parameters
- `as`: The outcomes.
-/
theorem total_uniform (as : List α) : (uniform as).total = as.length := by
  have h := total_foldl_add as (fun a => a) (fun _ => 1) {}
  rw [uniform, h, total_empty, Nat.zero_add]
  clear h
  induction as with
  | nil => rfl
  | cons a as ih => simp [ih]; omega

/--
Every listed outcome has positive weight.

# Parameters
- `as`: The outcomes.
-/
theorem positive_uniform (as : List α) : (uniform as).Positive :=
  positive_foldl_add as (fun a => a) (fun _ => 1) _ positive_empty
    fun _ _ => Nat.one_pos

omit [LawfulBEq α] in
/--
Each transformed outcome has the total weight of the outcomes that the
transformation takes to it.

# Parameters
- `f`: The transformation.
- `d`: The distribution.
- `b`: The transformed outcome whose weight is read.
-/
theorem weight_map (f : α → β) (d : Dist α) (b : β) :
    (d.map f).weight b =
      (d.weights.toList.map fun p => if f p.1 == b then p.2 else 0).sum := by
  have h := weight_foldl_add d.weights.toList (fun p => f p.1) (·.2) {} b
  rw [map, HashMap.fold_eq_foldl_toList, h, weight_empty, Nat.zero_add]

omit [LawfulBEq α] in
/--
Transforming the outcomes keeps the total.

# Parameters
- `f`: The transformation.
- `d`: The distribution.
-/
theorem total_map (f : α → β) (d : Dist α) : (d.map f).total = d.total := by
  have h := total_foldl_add d.weights.toList (fun p => f p.1) (·.2) {}
  rw [map, HashMap.fold_eq_foldl_toList, h, total_empty, Nat.zero_add,
    total_eq_sum]

omit [LawfulBEq α] in
/--
Transforming the outcomes keeps every weight positive.

# Parameters
- `f`: The transformation.
- `d`: The distribution.

# Hypotheses
- `hd`: Every weight of `d` is positive.
-/
theorem positive_map (f : α → β) (d : Dist α) (hd : d.Positive) :
    (d.map f).Positive := by
  have h := positive_foldl_add d.weights.toList (fun p => f p.1) (·.2) {}
    positive_empty hd
  rw [map, HashMap.fold_eq_foldl_toList]
  exact h

/-! ## Bind -/

omit [LawfulBEq α] [LawfulBEq β] in
/--
`bind` as folds over lists of entries: the branches are the entries in
reverse, each with its continuation, and each branch adds its continuation's
entries, scaled.

# Parameters
- `d`: The distribution of the first draw.
- `k`: The continuation.
-/
private theorem bind_eq (d : Dist α) (k : α → Dist β) :
    d.bind k =
      let bs := (d.weights.toList.map fun p => (p.2, k p.1)).reverse
      let l := bs.foldl (fun l x => l.lcm x.2.total) 1
      bs.foldl
        (fun acc x => x.2.weights.toList.foldl
          (fun acc p => acc.add p.1 (x.1 * (l / x.2.total) * p.2)) acc)
        {} := by
  simp only [bind, HashMap.fold_eq_foldl_toList, foldl_cons_eq,
    List.append_nil]

/--
Adding every entry of `e`, scaled by `s`, adds `s` times each of its weights.

# Parameters
- `e`: The distribution whose entries are added.
- `acc`: The distribution to add to.
- `s`: The scale.
- `c`: The outcome whose weight is read.
-/
private theorem weight_scaled (e acc : Dist β) (s : Nat) (c : β) :
    (e.weights.toList.foldl (fun acc p => acc.add p.1 (s * p.2)) acc).weight c
      = acc.weight c + s * e.weight c := by
  rw [weight_foldl_add, weight_eq_sum e, ← sum_map_mul]
  congr 2
  apply List.map_congr_left
  intro p _
  split <;> simp

/--
Adding every entry of `e`, scaled by `s`, adds `s` times its total.

# Parameters
- `e`: The distribution whose entries are added.
- `acc`: The distribution to add to.
- `s`: The scale.
-/
private theorem total_scaled (e acc : Dist β) (s : Nat) :
    (e.weights.toList.foldl (fun acc p => acc.add p.1 (s * p.2)) acc).total
      = acc.total + s * e.total := by
  rw [total_foldl_add, total_eq_sum e, ← sum_map_mul]

/--
Adding every entry of `e`, scaled by a positive `s`, keeps every weight
positive.

# Parameters
- `e`: The distribution whose entries are added.
- `acc`: The distribution to add to.
- `s`: The scale.

# Hypotheses
- `he`, `hacc`: Every weight of `e` and of `acc` is positive.
- `hs`: The scale is positive.
-/
private theorem positive_scaled (e acc : Dist β) (s : Nat) (he : e.Positive)
    (hacc : acc.Positive) (hs : 0 < s) :
    (e.weights.toList.foldl (fun acc p => acc.add p.1 (s * p.2)) acc).Positive
    :=
  positive_foldl_add _ _ _ _ hacc fun p hp => Nat.mul_pos hs (he p hp)

omit [LawfulBEq β] in
/--
The least common multiple of the totals of some branches, and a start, is a
multiple of each of them.

# Parameters
- `bs`: The branches, as weights and continuations.
- `l`: The start.
-/
private theorem lcm_foldl_dvd (bs : List (Nat × Dist β)) (l : Nat) :
    l ∣ bs.foldl (fun l x => l.lcm x.2.total) l ∧
      ∀ x ∈ bs, x.2.total ∣ bs.foldl (fun l x => l.lcm x.2.total) l := by
  induction bs generalizing l with
  | nil => simp
  | cons x bs ih =>
    have ⟨hl, hbs⟩ := ih (l.lcm x.2.total)
    refine ⟨Nat.dvd_trans (Nat.dvd_lcm_left _ _) hl, fun y hy => ?_⟩
    rcases List.mem_cons.mp hy with rfl | hy
    · exact Nat.dvd_trans (Nat.dvd_lcm_right _ _) hl
    · exact hbs y hy

omit [LawfulBEq β] in
/--
The least common multiple of positive totals, and a positive start, is
positive.

# Parameters
- `bs`: The branches, as weights and continuations.
- `l`: The start.

# Hypotheses
- `hl`: The start is positive.
- `hbs`: Every branch's total is positive.
-/
private theorem lcm_foldl_pos (bs : List (Nat × Dist β)) (l : Nat)
    (hl : 0 < l) (hbs : ∀ x ∈ bs, 0 < x.2.total) :
    0 < bs.foldl (fun l x => l.lcm x.2.total) l := by
  induction bs generalizing l with
  | nil => exact hl
  | cons x bs ih =>
    exact ih _ (Nat.lcm_pos hl (hbs x (by simp)))
      fun y hy => hbs y (by simp [hy])

section Branches

variable (l : Nat)

/--
Add one branch of a `bind`: every entry of its continuation, scaled by the
branch's weight times `l` over the continuation's total.

# Parameters
- `l`: The common multiple of the continuations' totals.
- `acc`: The distribution to add to.
- `x`: The branch, as its weight and its continuation.
-/
private abbrev branchStep (acc : Dist β) (x : Nat × Dist β) : Dist β :=
  x.2.weights.toList.foldl
    (fun acc p => acc.add p.1 (x.1 * (l / x.2.total) * p.2)) acc

/--
Adding the branches of a `bind` adds, to each outcome, its scaled weight in
each branch.

# Parameters
- `l`: The common multiple of the continuations' totals.
- `bs`: The branches.
- `acc`: The distribution to add to.
- `c`: The outcome whose weight is read.
-/
private theorem weight_branches (bs : List (Nat × Dist β)) (acc : Dist β)
    (c : β) :
    (bs.foldl (branchStep l) acc).weight c =
      acc.weight c +
        (bs.map fun x => x.1 * (l / x.2.total) * x.2.weight c).sum := by
  induction bs generalizing acc with
  | nil => simp
  | cons x bs ih =>
    simp only [List.foldl_cons, ih, branchStep, weight_scaled]; simp; omega

/--
Adding the branches of a `bind` adds each branch's scaled total.

# Parameters
- `l`: The common multiple of the continuations' totals.
- `bs`: The branches.
- `acc`: The distribution to add to.
-/
private theorem total_branches (bs : List (Nat × Dist β)) (acc : Dist β) :
    (bs.foldl (branchStep l) acc).total =
      acc.total + (bs.map fun x => x.1 * (l / x.2.total) * x.2.total).sum := by
  induction bs generalizing acc with
  | nil => simp
  | cons x bs ih =>
    simp only [List.foldl_cons, ih, branchStep, total_scaled]; simp; omega

/--
Adding the branches of a `bind` keeps every weight positive, if every branch
has positive weight and a positive total that divides `l`.

# Parameters
- `l`: The common multiple of the continuations' totals.
- `bs`: The branches.
- `acc`: The distribution to add to.

# Hypotheses
- `hl`: `l` is positive.
- `hacc`: Every weight of `acc` is positive.
- `hbs`: Every branch has positive weight, and its continuation a positive
  total that divides `l` and positive weights.
-/
private theorem positive_branches (bs : List (Nat × Dist β)) (acc : Dist β)
    (hl : 0 < l) (hacc : acc.Positive)
    (hbs : ∀ x ∈ bs,
      0 < x.1 ∧ 0 < x.2.total ∧ x.2.total ∣ l ∧ x.2.Positive) :
    (bs.foldl (branchStep l) acc).Positive := by
  induction bs generalizing acc with
  | nil => exact hacc
  | cons x bs ih =>
    have ⟨hw, ht, hdvd, hx⟩ := hbs x (by simp)
    exact ih _ (positive_scaled _ _ _ hx hacc (Nat.mul_pos hw
        (Nat.div_pos (Nat.le_of_dvd hl hdvd) ht)))
      fun y hy => hbs y (by simp [hy])

end Branches

omit [LawfulBEq α] in
/--
The laws of `bind`: for some positive `l` that every continuation's total
divides, each entry `(a, w)` of `d` contributes `w * (l / t)` times the
weights of its continuation `k a`, whose total is `t`, and the total is `l`
times that of `d`.

# Parameters
- `d`: The distribution of the first draw.
- `k`: The continuation.

# Hypotheses
- `hd`: Every weight of `d` is positive.
- `hk`: The continuation of every outcome of `d` has a positive total and
  positive weights.

# Returns
The least common multiple of the continuations' totals, as `l`, though any
common multiple would satisfy the laws; the probability of each outcome,
its weight over the total, does not depend on `l`.
-/
theorem bind_laws (d : Dist α) (k : α → Dist β) (hd : d.Positive)
    (hk : ∀ p ∈ d.weights.toList, 0 < (k p.1).total ∧ (k p.1).Positive) :
    ∃ l, 0 < l ∧ (∀ p ∈ d.weights.toList, (k p.1).total ∣ l) ∧
      (d.bind k).Positive ∧ (d.bind k).total = l * d.total ∧
      ∀ c, (d.bind k).weight c =
        (d.weights.toList.map fun p =>
          p.2 * (l / (k p.1).total) * (k p.1).weight c).sum := by
  rw [bind_eq]
  let bs := (d.weights.toList.map fun p => (p.2, k p.1)).reverse
  have mem : ∀ x ∈ bs, ∃ p ∈ d.weights.toList, x = (p.2, k p.1) := by
    intro x hx
    have ⟨p, hp, e⟩ := List.mem_map.mp (List.mem_reverse.mp hx)
    exact ⟨p, hp, e.symm⟩
  let l := bs.foldl (fun l x => l.lcm x.2.total) 1
  have hl : 0 < l := lcm_foldl_pos bs 1 Nat.one_pos fun x hx => by
    have ⟨p, hp, e⟩ := mem x hx; subst e; exact (hk p hp).1
  have hdvd : ∀ p ∈ d.weights.toList, (k p.1).total ∣ l := fun p hp =>
    (lcm_foldl_dvd bs 1).2 _
      (List.mem_reverse.mpr (List.mem_map.mpr ⟨p, hp, rfl⟩))
  refine ⟨l, hl, hdvd, ?_, ?_, fun c => ?_⟩
  · exact positive_branches l bs {} hl positive_empty fun x hx => by
      have ⟨p, hp, e⟩ := mem x hx; subst e
      exact ⟨hd p hp, (hk p hp).1, hdvd p hp, (hk p hp).2⟩
  · show (bs.foldl (branchStep l) {}).total = l * d.total
    rw [total_branches, total_empty, Nat.zero_add, total_eq_sum,
      List.map_reverse, List.sum_reverse_nat, List.map_map, ← sum_map_mul]
    congr 1
    apply List.map_congr_left
    intro p hp
    simp only [Function.comp]
    rw [Nat.mul_assoc, Nat.div_mul_cancel (hdvd p hp), Nat.mul_comm]
  · show (bs.foldl (branchStep l) {}).weight c = _
    rw [weight_branches, weight_empty, Nat.zero_add, List.map_reverse,
      List.sum_reverse_nat, List.map_map]
    rfl

end Dist

end Xdy
