import XdySpec.Reroll

/-!
# Explodes in the forward pass

An explode of the forward pass in `xdy/src/distribution/propagation.rs`
explodes every setup of its record's mixture by every joint outcome of its
operands (`Mixture::explode` in `propagation/mixture.rs`), and a setup by what
the explode does to the die from which it draws (`burst` in
`propagation/chain.rs`): nothing, if no face matches or the limit is not
positive; saturating every chain still kept, if an unlimited explode matches
every face; and otherwise, for a limit `n`, growing each chain whose end it
matches by each tail of extensions, weighed by the branches that reach it, so
that every chain weighs `Fⁿ` in all, for a die of `F` faces. The pass refuses
every function that has an unlimited explode that may extend a chain. Unless a
later reroll or explode reads the record, the chains reduce to their totals at
once. This module mirrors those operations and proves that they act on the law
of the record as the specification's explode does:

- `growths` mirrors `extensions`, the program over the tails of a chain, depth
  first rather than stage by stage: its ways weigh `Fⁿ` in all
  (`sum_growths`), and over their weights are `growthLaw`
  (`weighted_growths`), the specification's `extension`
  (`extension_eq_growthLaw`).
- `Fold` mirrors the folds by which the chains reduce to their totals: they
  form a monoid (`Fold.add_ofList`), and total as the ascending saturating sum
  does (`Fold.total_ofList`), so the fold of a chain and of a tail totals the
  grown chain (`total_growChain`), and the program of folds is the program of
  tails, folded (`growthLaw_folds`).
- `Burst.resolve` mirrors `burst`, and what it makes of a chain, over its
  padding, is the specification's: grown by its extension if the explode
  matches its end, or else itself (`Burst.weighted_chains`), or their totals
  (`Burst.weighted_totals`).
- The specification explodes the kept chains of a record one at a time, in
  order. Chains as the Rust keeps them, whose dice besides the last ascend and
  lie within the `i32` range (`NormalChain`), sort alike in any order
  (`normalChain_antisymm`, `ofChains_perm`), so the explode leaves
  independent draws, one for each chain (`explodeChains_eq_bag`), and of
  independent and alike chains, independent draws of the law of each chain,
  exploded (`bind_rollDice_iidBag`, `chainDraws_bind_bagDraws`), as
  `Roll::explode` transforms it (`transformLaw`, in `XdySpec/Chains.lean`,
  without padding: `transformLaw_one`).
- A setup's explode (`Setup.explode`, and `Setup.burst`, which reduces the
  chains to their totals unless they stay whole) describes the specification's
  explode of its records as every later instruction reads them, whether it
  does nothing, saturates, or extends independent and alike chains none of
  which has been dropped (`Setup.law_burst_alike`): if the chains stay whole,
  by their kept chains and die (`Record.chainView`), which a later reroll or
  explode reads; and otherwise by their kept results and die
  (`Record.view`).
- The proofs ask that every setup be well formed, have faces, and be live
  (`Setup.Live`): its die's faces lie within the `i32` range, and its records
  hold only faces of the die, in normal chains that may still grow, whose kept
  chains, after any drop, are a window of them (`Record.Fits.drops`), whether
  the roll is i.i.d. chains, stacks, or any other; every explode of alike
  chains leaves the first two, and the third if its chains stay whole and it
  does not saturate them (`Setup.burst_invariants_alike`). The collapse of
  stacks, which `XdySpec/Stacks.lean` proves, is a hypothesis here.
- An explode that saturates leaves stacks of saturated chains
  (`Setup.Saturated`), every record of which is saturated
  (`Setup.saturated_of_mem`), and which every later explode leaves alone, as
  the specification leaves their records (`Record.Saturated.explode`). A
  grounded setup (`Setup.Grounded`) is live or saturated, and
  `XdySpec/Stacks.lean` proves the laws of both.
- The Rust also explodes the chains of a roll after a drop, or of fixed
  results, a pool, or stacks after a drop, by transforming the multisets of
  the chains still kept, which `Roll.explodeMultisets` mirrors;
  `XdySpec/Stacks.lean` proves its laws, `Setup.law_burst`, which lifts the
  hypothesis that the explode extends alike chains (`Setup.ExtendsAlike`), and
  those of a mixture's explode, independent of its operands
  (`mixtureLaw_burst_whole`, `mixtureLaw_burst_totals`,
  `law_explode_resolved`), and the laws of the rerolls of chains
  (`Setup.law_reroll`). It explodes stacks none of whose chains has been
  dropped by composing the laws that they defer with the explode, which
  `Roll.explodeStacks` mirrors, and whose laws `Setup.law_burst` proves too.

```mermaid
flowchart TD
    S[setup] -->|nothing kept| N[unchanged]
    S --> B{Burst.resolve}
    B -->|nothing| N
    B -->|unbounded, no end matches| N
    B -->|saturate v| T[one multiset of saturated chains]
    B -->|extend Fⁿ tails| A{alike, none dropped?}
    A -->|yes, whole| C[i.i.d. chains: each chain's law, grown]
    A -->|yes, not whole| D[custom dice of the grown chains' totals]
    A -->|no| L[XdySpec/Stacks.lean: each multiset of the kept chains]
```
-/

namespace Xdy.Spec

open Xdy (Die Chain Criterion)
open scoped ENNReal

/-! ## Weighted values -/

/-- A sum over a list of lists sums the sums of the lists. -/
private theorem sum_flatMap' {α : Type} (l : List α) (f : α → List Nat) :
    (l.flatMap f).sum = (l.map fun a => (f a).sum).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]

/--
One value of positive weight is certain.

# Parameters
- `d`: The value if no weight is positive.
- `x`: The value.

# Hypotheses
- `hk`: Its weight is positive.
-/
theorem weighted_single {α : Type} (d x : α) {k : Nat} (hk : k ≠ 0) :
    weighted d [(x, k)] = PMF.pure x := by
  have h : ([(x, k)].flatMap fun p : α × Nat => List.replicate p.2 p.1) =
      List.replicate k x := by simp
  have hne : List.replicate k x ≠ [] := by simpa using hk
  unfold weighted
  simp only [h, hne, ↓reduceDIte]
  ext y
  classical
  rw [PMF.ofMultiset_apply, PMF.pure_apply]
  by_cases hy : y = x
  · subst hy
    simp only [Multiset.coe_count, List.count_replicate_self, Multiset.coe_card,
      List.length_replicate, ↓reduceIte]
    exact ENNReal.div_self (by simpa using hk) (ENNReal.natCast_ne_top _)
  · simp [hy]

/--
Weighted values of positive total weight do not depend on the value that they
would take if none were positive.

# Parameters
- `d`, `d'`: The values if no weight were positive.
- `l`: The values, each with its weight.

# Hypotheses
- `hl`: The total weight is positive.
-/
theorem weighted_congr_default {α : Type} (d d' : α) {l : List (α × Nat)}
    (hl : (l.map (·.2)).sum ≠ 0) : weighted d l = weighted d' l := by
  have hne : (l.flatMap fun p => List.replicate p.2 p.1) ≠ [] := by
    intro he
    have := congrArg List.length he
    simp only [List.length_flatMap, List.length_replicate,
      List.length_nil] at this
    exact hl (by simpa [Function.comp_def] using this)
  unfold weighted
  simp only [hne, ↓reduceDIte]

/--
Weighted values, each sent to another, are the values that they are sent to,
with the same weights.

# Parameters
- `d`: The value if no weight is positive.
- `l`: The values, each with its weight.
- `f`: Where each value is sent.

# Hypotheses
- `hl`: The total weight is positive.
-/
theorem weighted_map {α β : Type} [DecidableEq β] (d : α)
    (l : List (α × Nat)) (f : α → β) (hl : (l.map (·.2)).sum ≠ 0) :
    weighted (f d) (l.map fun e => (f e.1, e.2)) = (weighted d l).map f := by
  have hl' : (l.flatMap fun e => [(f e.1, 1)].map fun o => (o.1, e.2 * 1 * o.2))
      = l.map fun e => (f e.1, e.2) := by
    simp only [List.map_cons, List.map_nil, mul_one]
    clear hl
    induction l with
    | nil => rfl
    | cons e l ih => simp [ih]
  have h := weighted_flatMap (f d) d (fun a => f a) l (fun a => [(f a, 1)])
    (fun _ => 1) (fun _ o => o) 1 one_ne_zero hl (fun _ _ _ => rfl)
  rw [hl'] at h
  rw [h, ← PMF.bind_pure_comp]
  refine congrArg _ (funext fun a => ?_)
  rw [weighted_single _ _ one_ne_zero, PMF.pure_map]
  rfl

/-! ## Extensions -/

/--
Every way in which a chain whose end an explode matches may grow, from a
state of its extensions so far, with `r` more extensions allowed after the
next, each weighed by its branches, padded by the branches of the extensions
that it never needed, as `extensions` in `propagation/chain.rs` answers them.
The next extension shows each face of the die: if the explode matches it and
another extension is allowed, the chain extends from it, with the state
extended by it; otherwise it ends there, padded by `Fʳ⁺¹`, for a die of `F`
branches, which closes the state.

# Parameters
- `faces`: The distinct faces of the die, each with its branches.
- `c`: The criterion.
- `extend`: The state after another matching extension.
- `close`: The way in which the chain grows, from the state and the extension
  that ends it.
- `r`: The number of extensions allowed after the next.
- `s`: The state.

# Returns
The ways, each with its branches, which sum to `Fʳ⁺¹` (`sum_growths`).

# Notes
The Rust rolls the extensions breadth first, one stage for each extension,
merging the states that coincide, and closes the state extended by the last
extension; this follows each face depth first, and closes the state before the
last extension with it, which reaches the same ways with the same weights. Its
law is `growthLaw` (`weighted_growths`).

# Examples
```lean
#guard growths [(1, 1), (2, 1)] (.within 2 2) (fun d x => (x :: d).mergeSort)
  (fun d x => (d, x)) 1 [] ==
    [(([], 1), 2), (([2], 1), 1), (([2], 2), 1)]
```
-/
def growths {S K : Type} (faces : List (Int × Nat)) (c : Criterion)
    (extend : S → Int → S) (close : S → Int → K) :
    Nat → S → List (K × Nat)
  | 0, s => faces.map fun f => (close s f.1, f.2)
  | r + 1, s => faces.flatMap fun f =>
    if c.matches f.1 then
      (growths faces c extend close r (extend s f.1)).map fun g =>
        (g.1, f.2 * g.2)
    else [(close s f.1, f.2 * (faces.map (·.2)).sum ^ (r + 1))]

/--
The law of the way in which a chain whose end an explode matches grows, from a
state of its extensions so far, with `r` more extensions allowed after the
next: a draw from the die, which extends the chain again if the explode
matches it and another extension is allowed, and otherwise ends it.

# Parameters
- `die`: The law of the die.
- `c`: The criterion.
- `extend`: The state after another matching extension.
- `close`: The way in which the chain grows, from the state and the extension
  that ends it.
- `r`: The number of extensions allowed after the next.
- `s`: The state.

# Returns
The law of the way in which the chain grows.

# Examples
```lean
example (die : PMF Int) (c : Criterion) :
    growthLaw die c (fun (d : List Int) x => x :: d) (fun d x => (d, x)) 0 [] =
      die.map ([], ·) := rfl
```
-/
noncomputable def growthLaw {S K : Type} (die : PMF Int) (c : Criterion)
    (extend : S → Int → S) (close : S → Int → K) : Nat → S → PMF K
  | 0, s => die.map (close s)
  | r + 1, s => die.bind fun x =>
    if c.matches x then growthLaw die c extend close r (extend s x)
    else PMF.pure (close s x)

/--
The ways in which a chain may grow weigh `Fʳ⁺¹` in all, for a die of `F`
branches, as `extensions` in `propagation/chain.rs` pads them.

# Parameters
- `faces`: The distinct faces of the die, each with its branches.
- `c`: The criterion.
- `extend`: The state after another matching extension.
- `close`: The way in which the chain grows.
- `r`: The number of extensions allowed after the next.
- `s`: The state.
-/
theorem sum_growths {S K : Type} (faces : List (Int × Nat)) (c : Criterion)
    (extend : S → Int → S) (close : S → Int → K) (r : Nat) (s : S) :
    ((growths faces c extend close r s).map (·.2)).sum =
      (faces.map (·.2)).sum ^ (r + 1) := by
  induction r generalizing s with
  | zero => simp [growths, Function.comp_def]
  | succ r ih =>
    have hf : ∀ f ∈ faces, ((if c.matches f.1 then
        (growths faces c extend close r (extend s f.1)).map fun g =>
          (g.1, f.2 * g.2)
        else [(close s f.1, f.2 * (faces.map (·.2)).sum ^ (r + 1))]).map
          (·.2)).sum = f.2 * (faces.map (·.2)).sum ^ (r + 1) := fun f _ => by
      split
      · simp only [List.map_map, Function.comp_def]
        rw [List.sum_map_mul_left]
        exact congrArg (f.2 * ·) (ih _)
      · simp
    rw [growths, List.map_flatMap, sum_flatMap', List.map_congr_left hf,
      List.sum_map_mul_right, pow_succ _ (r + 1), mul_comm]

/--
The ways in which a chain may grow, over their total weight, are their law:
the Rust's weights are the specification's probabilities.

# Parameters
- `d`: The way if no weight is positive.
- `faces`: The distinct faces of the die, each with its branches.
- `c`: The criterion.
- `extend`: The state after another matching extension.
- `close`: The way in which the chain grows.
- `r`: The number of extensions allowed after the next.
- `s`: The state.

# Hypotheses
- `hF`: The die has branches.
-/
theorem weighted_growths {S K : Type} [DecidableEq K] (d : K)
    {faces : List (Int × Nat)} (c : Criterion) (extend : S → Int → S)
    (close : S → Int → K) (hF : (faces.map (·.2)).sum ≠ 0) (r : Nat)
    (s : S) :
    weighted d (growths faces c extend close r s) =
      growthLaw (weightedDie faces) c extend close r s := by
  induction r generalizing s d with
  | zero =>
    have h := weighted_map (f := close s) 0 faces hF
    simp only [growths, growthLaw, weightedDie_eq_weighted]
    rw [← h]
    have hd : weighted d (faces.map fun f => (close s f.1, f.2)) =
        weighted (close s 0) (faces.map fun f => (close s f.1, f.2)) := by
      unfold weighted
      have hne : ((faces.map fun f => (close s f.1, f.2)).flatMap
          fun p => List.replicate p.2 p.1) ≠ [] := by
        intro he
        have := congrArg List.length he
        simp only [List.length_flatMap, List.length_replicate, List.map_map,
          List.length_nil] at this
        exact hF (by simpa [Function.comp_def] using this)
      simp only [hne, ↓reduceDIte]
    exact hd
  | succ r ih =>
    set F := (faces.map (·.2)).sum
    let h : Int → List (K × Nat) := fun x =>
      if c.matches x then growths faces c extend close r (extend s x)
      else [(close s x, 1)]
    let w : Int → Nat := fun x => if c.matches x then 1 else F ^ (r + 1)
    have hlist : growths faces c extend close (r + 1) s =
        faces.flatMap fun e =>
          (h e.1).map fun o => (o.1, e.2 * w e.1 * o.2) := by
      simp only [growths]
      refine congrArg (fun g => List.flatMap g faces) (funext fun e => ?_)
      simp only [h, w]
      split
      · simp
      · simp only [List.map_cons, List.map_nil, mul_one, F]
    rw [hlist, weighted_flatMap d 0 (fun x => close s x) faces h w
      (fun _ o => o) (F ^ (r + 1)) (pow_ne_zero _ hF) hF ?_]
    · simp only [growthLaw, weightedDie_eq_weighted]
      refine congrArg _ (funext fun x => ?_)
      rw [show (fun o : K => o) = id from rfl, PMF.map_id]
      simp only [h]
      split
      · exact ih _ _
      · exact weighted_single _ _ one_ne_zero
    · intro e _ _
      simp only [h, w]
      split
      · rw [one_mul, sum_growths]
      · simp [F]

/-! ## Tails -/

/--
The state of the extensions of a chain whose end an explode matched, as a
tail tracks it: the extensions so far, in ascending order, each of which
matched. Mirrors the tails of `burst` in `propagation/chain.rs`, which keeps
them as a multiset, with the extension that ends the chain.

# Parameters
- `drawn`: The extensions so far, in ascending order.
- `x`: Another extension, which matched.

# Returns
The extensions, with `x`, in ascending order.

# Examples
```lean
#guard tailExtend [2, 5] 3 == [2, 3, 5]
```
-/
def tailExtend (drawn : List Int) (x : Int) : List Int := (x :: drawn).mergeSort

/--
The tail that a chain whose end an explode matched grows, from the extensions
so far and the extension that ends it: as the specification's `extension`
answers it, the extensions besides the last, in ascending order, and the last.

# Parameters
- `drawn`: The extensions besides the last, in ascending order.
- `y`: The last extension.

# Returns
The tail.
-/
def tailClose (drawn : List Int) (y : Int) : List Int × Int := (drawn, y)

/-- One more stage of the specification's extension, from the law of the
extensions so far, is the law of the tails with one more extension allowed. -/
private theorem growthLaw_bind_step (c : Criterion) (die : PMF Int) :
    ∀ (k : Nat) (s : List Int),
      (growthLaw die c tailExtend tailClose k s).bind
          (fun p => if c.matches p.2 then
            die.map fun y => ((p.2 :: p.1).mergeSort, y)
            else PMF.pure p) =
        growthLaw die c tailExtend tailClose (k + 1) s
  | 0, s => by
    simp only [growthLaw, PMF.bind_map]
    rfl
  | k + 1, s => by
    simp only [growthLaw, PMF.bind_bind]
    refine congrArg _ (funext fun x => ?_)
    split
    · exact growthLaw_bind_step c die k _
    · rename_i hx
      simp [PMF.pure_bind, tailClose, hx]

/--
The specification's extension of a chain whose end an explode limited to
`limit` times matches is the law of the tails, from no extensions, with
`limit - 1` more allowed after the first.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The number of times that each chain may be exploded.
-/
theorem extension_eq_growthLaw (c : Criterion) (die : PMF Int) (limit : Nat) :
    extension c die limit =
      growthLaw die c tailExtend tailClose (limit - 1) [] := by
  unfold extension
  generalize limit - 1 = k
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [Nat.repeat, ih, ← growthLaw_bind_step]

/-! ## Folds -/

/-- The saturating fold of some values in ascending order, by its parts: the
sum of the negative values, which the fold adds first, and so saturates at
most once, at `i32Min`, and the sum of the others, which it adds next, from
there, and so saturates at most once, at `i32Max`. Mirrors `Fold` in
`propagation/chain.rs`. -/
structure Fold where
  /-- The sum of the negative values, clamped below at `i32Min`. -/
  negative : Int := 0
  /-- The sum of the other values, clamped above at `Fold.ceiling`. -/
  positive : Int := 0
  deriving DecidableEq, Repr

namespace Fold

/-- The greatest sum of positive values that a fold keeps, `2³³`, which
exceeds the distance from `i32Min` to `i32Max`, so that clamping there never
changes a total. Mirrors `Fold::CEILING`. -/
def ceiling : Int := 2 ^ 33

/--
The fold of copies of one value. Mirrors `Fold::of`.

# Parameters
- `value`: The value.
- `copies`: The number of copies.

# Returns
The fold.

# Examples
```lean
#guard Fold.of (-3) 2 == { negative := -6 }
#guard Fold.of 5 2 == { positive := 10 }
```
-/
def of (value : Int) (copies : Nat) : Fold :=
  if value < 0 then { negative := Max.max i32Min (value * copies) }
  else { positive := Min.min ceiling (value * copies) }

/--
The fold of the union of the values of two folds. Mirrors `Fold::add`.

# Parameters
- `a`, `b`: The folds.

# Returns
The fold of the union.
-/
def add (a b : Fold) : Fold :=
  { negative := Max.max i32Min (a.negative + b.negative)
    positive := Min.min ceiling (a.positive + b.positive) }

/--
The total of a fold: the saturating sum of its values in ascending order.
Mirrors `Fold::total`.

# Parameters
- `a`: The fold.

# Returns
The total.

# Examples
```lean
#guard (Fold.of 5 2).total == 10
#guard ((Fold.of i32Max 1).add (Fold.of 1 1)).total == i32Max
```
-/
def total (a : Fold) : Int := Min.min i32Max (a.negative + a.positive)

/--
The fold of some values, in any order: the sum of the negative values, and
the sum of the others, each clamped.

# Parameters
- `xs`: The values.

# Returns
The fold.

# Examples
```lean
#guard Fold.ofList [3, -1, 2] == { negative := -1, positive := 5 }
```
-/
def ofList (xs : List Int) : Fold :=
  { negative := Max.max i32Min (xs.filter (· < 0)).sum
    positive := Min.min ceiling (xs.filter (0 ≤ ·)).sum }

/-- Values that are at most `0` sum to at most `0`. -/
private theorem sum_nonpos' :
    ∀ (xs : List Int), (∀ x ∈ xs, x ≤ 0) → xs.sum ≤ 0
  | [], _ => le_rfl
  | x :: xs, h => by
    have := h x List.mem_cons_self
    have := sum_nonpos' xs fun y hy => h y (List.mem_cons_of_mem _ hy)
    simp only [List.sum_cons]
    omega

/-- The negative values sum to at most `0`. -/
private theorem sum_neg_nonpos (xs : List Int) :
    (xs.filter (· < 0)).sum ≤ 0 :=
  sum_nonpos' _ fun x hx => by
    have := (List.mem_filter.mp hx).2
    simp only [decide_eq_true_eq] at this
    omega

/-- The other values sum to at least `0`. -/
private theorem sum_nonneg' (xs : List Int) :
    0 ≤ (xs.filter (0 ≤ ·)).sum :=
  List.sum_nonneg fun x hx => by
    have := (List.mem_filter.mp hx).2
    simpa using this

/--
The fold of some values depends only on their multiset.

# Parameters
- `xs`, `ys`: The values.

# Hypotheses
- `h`: The values are the same, in some order.
-/
theorem ofList_perm {xs ys : List Int} (h : xs.Perm ys) :
    ofList xs = ofList ys := by
  simp only [ofList, (h.filter _).sum_eq]

/--
The union of the values of two folds is the fold of the union, as
`Fold::add` relies on: folds form a monoid.

# Parameters
- `xs`, `ys`: The values of each.
-/
theorem add_ofList (xs ys : List Int) :
    (ofList xs).add (ofList ys) = ofList (xs ++ ys) := by
  have h₁ := sum_neg_nonpos xs
  have h₂ := sum_neg_nonpos ys
  have h₃ := sum_nonneg' xs
  have h₄ := sum_nonneg' ys
  simp only [add, ofList, List.filter_append, List.sum_append, ceiling,
    i32Min, Fold.mk.injEq]
  omega

/--
The fold of copies of a value is the fold of the copies.

# Parameters
- `value`: The value.
- `copies`: The number of copies.
-/
theorem of_eq_ofList (value : Int) (copies : Nat) :
    of value copies = ofList (List.replicate copies value) := by
  unfold of ofList
  by_cases h : value < 0
  · simp [h, show ¬0 ≤ value by omega,
      List.sum_replicate, mul_comm, ceiling]
  · simp [h, show 0 ≤ value by omega,
      List.sum_replicate, mul_comm, i32Min]

/-- Adding negative values to a running sum within the `i32` range saturates
at most once, at `i32Min`. -/
private theorem foldl_add_neg :
    ∀ (xs : List Int), (∀ x ∈ xs, x < 0) → ∀ s : Int, i32Min ≤ s →
      s ≤ 0 →
      xs.foldl Xdy.add s = Max.max i32Min (s + xs.sum)
  | [], _, s, h₁, _ => by simp; omega
  | x :: xs, hx, s, h₁, h₂ => by
    have hx0 := hx x List.mem_cons_self
    have ht : xs.sum ≤ 0 := sum_nonpos' xs fun y hy =>
      (hx y (List.mem_cons_of_mem _ hy)).le
    rw [List.foldl_cons, foldl_add_neg xs (fun y hy => hx y
      (List.mem_cons_of_mem _ hy)) _ (by simp only [Xdy.add, clamp]; omega)
      (by simp only [Xdy.add, clamp, i32Min, i32Max] at h₁ ⊢; omega)]
    simp only [Xdy.add, clamp, List.sum_cons, i32Min, i32Max] at h₁ ⊢
    omega

/-- Adding values that are not negative to a running sum within the `i32`
range saturates at most once, at `i32Max`. -/
private theorem foldl_add_nonneg :
    ∀ (xs : List Int), (∀ x ∈ xs, 0 ≤ x) → ∀ s : Int,
      i32Min ≤ s → s ≤ i32Max → xs.foldl Xdy.add s = Min.min i32Max (s + xs.sum)
  | [], _, s, _, h₂ => by simp; omega
  | x :: xs, hx, s, h₁, h₂ => by
    have hx0 := hx x List.mem_cons_self
    have ht : 0 ≤ xs.sum := List.sum_nonneg fun y hy =>
      hx y (List.mem_cons_of_mem _ hy)
    rw [List.foldl_cons, foldl_add_nonneg xs (fun y hy => hx y
      (List.mem_cons_of_mem _ hy)) _
      (by simp only [Xdy.add, clamp, i32Min, i32Max] at h₁ ⊢; omega)
      (by simp only [Xdy.add, clamp, i32Min, i32Max] at h₁ ⊢; omega)]
    simp only [Xdy.add, clamp, List.sum_cons, i32Min, i32Max] at h₁ h₂ ⊢
    omega

/-- Values in ascending order are their negative values, then the others. -/
private theorem eq_filter_append :
    ∀ (xs : List Int), xs.Pairwise (· ≤ ·) →
      xs = xs.filter (· < 0) ++ xs.filter (0 ≤ ·)
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.pairwise_cons] at h
    by_cases hx : x < 0
    · simp only [List.filter_cons, hx, decide_true, ↓reduceIte,
        show ¬0 ≤ x by omega, decide_false, List.cons_append]
      exact congrArg _ (eq_filter_append xs h.2)
    · have hall : ∀ y ∈ xs, 0 ≤ y := fun y hy => by
        have := h.1 y hy
        omega
      rw [List.filter_eq_nil_iff.mpr (fun y hy => by
          simp only [decide_eq_true_eq]; exact not_lt.mpr
            (by rcases List.mem_cons.mp hy with rfl | hy
                · omega
                · exact hall y hy)),
        List.filter_eq_self.mpr (fun y hy => by
          simp only [decide_eq_true_eq]
          rcases List.mem_cons.mp hy with rfl | hy
          · omega
          · exact hall y hy)]
      rfl

/--
The total of the fold of some values is the saturating sum of the values in
ascending order, as `kept_sum` adds them and as `Xdy.Chain.total` totals the
dice of a chain.

# Parameters
- `xs`: The values, in any order.
-/
theorem total_ofList (xs : List Int) :
    (ofList xs).total = (xs.mergeSort).foldl Xdy.add 0 := by
  set l := xs.mergeSort
  have hp : l.Perm xs := List.mergeSort_perm _ _
  have hs : l.Pairwise (· ≤ ·) := by
    have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
      (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) xs
    simpa using this
  rw [← ofList_perm hp, eq_filter_append l hs, List.foldl_append]
  have hneg : ∀ x ∈ l.filter (· < 0), x < 0 := fun x hx => by
    simpa using (List.mem_filter.mp hx).2
  have hnn : ∀ x ∈ l.filter (0 ≤ ·), 0 ≤ x := fun x hx => by
    simpa using (List.mem_filter.mp hx).2
  rw [foldl_add_neg _ hneg 0 (by simp [i32Min]) le_rfl]
  have h₁ := sum_neg_nonpos l
  have h₂ := sum_nonneg' l
  rw [foldl_add_nonneg _ hnn _ (by omega)
    (by simp only [i32Min, i32Max]; omega)]
  have e₁ : (l.filter (· < 0)).filter (· < 0) = l.filter (· < 0) :=
    List.filter_eq_self.mpr fun x hx => by simpa using hneg x hx
  have e₂ : (l.filter (0 ≤ ·)).filter (· < 0) = [] :=
    List.filter_eq_nil_iff.mpr fun x hx => by
      have := hnn x hx
      simp only [decide_eq_true_eq]
      omega
  have e₃ : (l.filter (· < 0)).filter (0 ≤ ·) = [] :=
    List.filter_eq_nil_iff.mpr fun x hx => by
      have := hneg x hx
      simp only [decide_eq_true_eq]
      omega
  have e₄ : (l.filter (0 ≤ ·)).filter (0 ≤ ·) = l.filter (0 ≤ ·) :=
    List.filter_eq_self.mpr fun x hx => by simpa using hnn x hx
  have hf : (l.filter (· < 0) ++ l.filter (0 ≤ ·)).filter (· < 0) =
      l.filter (· < 0) := by
    rw [List.filter_append, e₁, e₂, List.append_nil]
  have hg : (l.filter (· < 0) ++ l.filter (0 ≤ ·)).filter (0 ≤ ·) =
      l.filter (0 ≤ ·) := by
    rw [List.filter_append, e₃, e₄, List.nil_append]
  simp only [total, ofList, hf, hg, ceiling, i32Min, i32Max]
  omega

end Fold


/-! ## Growing chains -/

/--
The law of the ways in which a chain may grow, mapped, is the law of the ways
of a program whose states and ways are mapped alike: the tails of a chain, by
their folds, are the folds that `burst` in `propagation/chain.rs` grows when
the chains reduce to their totals at once.

# Parameters
- `die`: The law of the die.
- `c`: The criterion.
- `extend`, `close`: The first program.
- `extend'`, `close'`: The second program.
- `φ`: The map of the states.
- `ψ`: The map of the ways.

# Hypotheses
- `he`: The map of the states commutes with extending them.
- `hc`: Closing the mapped states maps the ways.
-/
theorem growthLaw_map {S S' K K' : Type} (die : PMF Int) (c : Criterion)
    {extend : S → Int → S} {close : S → Int → K}
    {extend' : S' → Int → S'} {close' : S' → Int → K'} (φ : S → S') (ψ : K → K')
    (he : ∀ s x, φ (extend s x) = extend' (φ s) x)
    (hc : ∀ s x, close' (φ s) x = ψ (close s x)) :
    ∀ (r : Nat) (s : S), growthLaw die c extend' close' r (φ s) =
      (growthLaw die c extend close r s).map ψ
  | 0, s => by
    simp only [growthLaw, PMF.map_comp]
    exact congrArg (PMF.map · die) (funext fun x => hc s x)
  | r + 1, s => by
    simp only [growthLaw, PMF.map_bind]
    refine congrArg _ (funext fun x => ?_)
    split
    · rw [← he, growthLaw_map die c φ ψ he hc r]
    · rw [PMF.pure_map, hc]

/--
A chain grown by a tail: its kept dice and the extensions besides the last, in
ascending order, and the last extension at its end. Mirrors `Chain::of`,
applied to the union of a chain's dice and a tail's, in `Burst::chains` in
`propagation/chain.rs`, and the growth of a chain in `explodeChains`.

# Parameters
- `ch`: The chain.
- `t`: The tail: the extensions besides the last, in ascending order, and the
  last.

# Returns
The grown chain.

# Examples
```lean
#guard growChain (.single 6) ([6], 2) == .live [6, 6] 2
```
-/
def growChain (ch : Chain) (t : List Int × Int) : Chain :=
  .live (ch.dice ++ t.1).mergeSort t.2

/--
The fold of the kept dice of a chain, distinct value by distinct value. Mirrors
`Chain::fold` in `propagation/chain.rs`.

# Parameters
- `ch`: The chain.

# Returns
The fold.

# Examples
```lean
#guard chainFold (.live [2, 2] 5) == { positive := 9 }
```
-/
def chainFold (ch : Chain) : Fold :=
  (tally ch.dice).foldl (fun f p => f.add (Fold.of p.1 p.2)) {}

/-- The fold of no values is empty. -/
private theorem Fold.ofList_nil : Fold.ofList [] = {} := by
  simp [Fold.ofList, i32Min, Fold.ceiling]

/--
The fold of a chain is the fold of its kept dice.

# Parameters
- `ch`: The chain.
-/
theorem chainFold_eq (ch : Chain) : chainFold ch = Fold.ofList ch.dice := by
  have h : ∀ (l : List (Int × Nat)) (acc : List Int),
      l.foldl (fun f p => f.add (Fold.of p.1 p.2)) (Fold.ofList acc) =
        Fold.ofList (acc ++ Roll.expand l) := by
    intro l
    induction l with
    | nil => simp [Roll.expand]
    | cons p l ih =>
      intro acc
      rw [List.foldl_cons, Fold.of_eq_ofList, Fold.add_ofList, ih]
      simp [Roll.expand, List.append_assoc]
  rw [chainFold, ← Fold.ofList_nil, h, List.nil_append]
  exact Fold.ofList_perm (expand_tally_perm _)

/--
The state of the fold of the extensions of a chain, after another extension,
as `burst` in `propagation/chain.rs` extends it when the chains reduce to their
totals at once.

# Parameters
- `f`: The fold of the extensions so far.
- `x`: Another extension.

# Returns
The fold with `x`.
-/
def foldExtend (f : Fold) (x : Int) : Fold := f.add (Fold.of x 1)

/--
The folds of the tails are the ways of the program of folds: the folds of the
extensions, the last among them.

# Parameters
- `die`: The law of the die.
- `c`: The criterion.
- `r`: The number of extensions allowed after the next.
-/
theorem growthLaw_folds (die : PMF Int) (c : Criterion) (r : Nat) :
    growthLaw die c foldExtend foldExtend r {} =
      (growthLaw die c tailExtend tailClose r []).map
        fun t => Fold.ofList (t.2 :: t.1) := by
  rw [← Fold.ofList_nil]
  refine growthLaw_map die c Fold.ofList _ (fun d x => ?_) (fun d x => ?_) r []
  · rw [foldExtend, Fold.of_eq_ofList, Fold.add_ofList, tailExtend]
    exact Fold.ofList_perm ((List.mergeSort_perm _ _).trans
      (List.perm_append_singleton x d).symm)
  · rw [foldExtend, Fold.of_eq_ofList, Fold.add_ofList, tailClose]
    exact Fold.ofList_perm (List.perm_append_singleton x d)

/--
The total of a chain grown by a tail is the total of the fold of the chain and
the fold of the tail, as `Burst::totals` in `propagation/chain.rs` totals it.

# Parameters
- `ch`: The chain.
- `t`: The tail.
-/
theorem total_growChain (ch : Chain) (t : List Int × Int) :
    ((chainFold ch).add (Fold.ofList (t.2 :: t.1))).total =
      (growChain ch t).total := by
  rw [chainFold_eq, Fold.add_ofList, growChain]
  show _ = ((t.2 :: (ch.dice ++ t.1).mergeSort).mergeSort).foldl Xdy.add 0
  rw [← Fold.total_ofList]
  refine congrArg Fold.total (Fold.ofList_perm ?_)
  exact List.perm_middle.trans ((List.mergeSort_perm _ _).symm.cons _)

/-! ## Bursts -/

/-- The tails that an explode may add to a chain whose end it matches, each
weighed by its branches. Mirrors `Tails` in `propagation/chain.rs`. -/
inductive Tails where
  /-- Their folds alone, when the chains reduce to their totals at once. -/
  | folds (folds : List (Fold × Nat))
  /-- The tails themselves, the extensions besides the last, in ascending
  order, and the last, when a later reroll or explode reads the chains. -/
  | tails (tails : List ((List Int × Int) × Nat))
  deriving DecidableEq, Repr

/-- What an explode does to the chains still kept of a roll, given the die
from which it draws their extensions. Mirrors `Burst` in
`propagation/chain.rs`, which keeps the criterion that every function here
takes. -/
inductive Burst where
  /-- Nothing, since no face matches, or the limit is not positive. -/
  | nothing
  /-- The explode saturates every chain still kept at the value, rolling
  nothing, since it has no limit and matches every face. -/
  | saturate (value : Int)
  /-- The explode has no limit, and does not match every face, so it may
  extend a chain without bound, which the pass refuses. -/
  | unbounded
  /-- The explode may extend the chains still kept, each up to its limit:
  each chain weighs the `padding` in all, and one whose end the explode matches
  grows each of the `tails` along its weight. -/
  | extend (padding : Nat) (tails : Tails)
  deriving DecidableEq, Repr

namespace Burst

/--
Resolve an explode against the die from which it draws: decide whether it may
extend the chains still kept, and if so, how a chain whose end it matches may
grow. Mirrors `burst` in `propagation/chain.rs`.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals at once.

# Returns
The burst: for a positive limit `n`, the padding `Fⁿ` and the tails, or their
folds, that `growths` answers from no extensions.

# Examples
```lean
#guard Burst.resolve (.standard 2) (.within 2 2) (some 1) true ==
  .extend 2 (.tails [(([], 1), 1), (([], 2), 1)])
#guard Burst.resolve (.standard 6) (.within 1 6) none true ==
  .saturate i32Max
#guard Burst.resolve (.standard 6) (.within 7 7) (some 3) true == .nothing
```
-/
def resolve (d : Die) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Burst :=
  let F := d.faces.length
  let m := d.faces.countP c.matches
  if m = 0 then .nothing
  else
    match limit with
    | some 0 => .nothing
    | some (n + 1) =>
      .extend (F ^ (n + 1))
        (if whole then
          .tails (growths (tally d.faces) c tailExtend tailClose n [])
        else .folds (growths (tally d.faces) c foldExtend foldExtend n {}))
    | none => if m = F then .saturate d.saturatedValue else .unbounded

/--
The branches of each chain still kept: `Fⁿ` for a limited explode, and one
for any other. Mirrors `Burst::padding`.

# Parameters
- `b`: The burst.

# Returns
The branches.
-/
def padding : Burst → Nat
  | .extend padding _ => padding
  | _ => 1

/--
Every chain that a chain still kept may become after the explode, each with
its branches, which sum to the padding: the chain grown by each tail if the
explode extends it, or else the chain itself. Mirrors `Burst::chains`.

# Parameters
- `b`: The burst.
- `c`: The criterion.
- `ch`: The chain.

# Returns
The chains, each with its weight.

# Notes
The Rust merges the chains that coincide and sorts them; this lists them as
they come.
-/
def chains (b : Burst) (c : Criterion) (ch : Chain) : List (Chain × Nat) :=
  match b with
  | .extend _ (.tails tails) =>
    if ch.grows c then tails.map fun t => (growChain ch t.1, t.2)
    else [(ch, b.padding)]
  | _ => [(ch, b.padding)]

/--
Every total that a chain still kept may reach after the explode, each with its
branches, which sum to the padding: the total of the fold of the chain and of
each fold of the tails if the explode extends it, or else its own. Mirrors
`Burst::totals`.

# Parameters
- `b`: The burst.
- `c`: The criterion.
- `ch`: The chain.

# Returns
The totals, each with its weight.

# Notes
The Rust merges the totals that coincide and sorts them; this lists them as
they come.
-/
def totals (b : Burst) (c : Criterion) (ch : Chain) : List (Int × Nat) :=
  match b with
  | .extend _ (.folds folds) =>
    if ch.grows c then folds.map fun f => (((chainFold ch).add f.1).total, f.2)
    else [(ch.total, b.padding)]
  | _ => [(ch.total, b.padding)]

end Burst

namespace Burst

/-- A tally holds each value as often as the list. -/
private theorem sum_tally (xs : List Int) :
    ((tally xs).map (·.2)).sum = xs.length := by
  rw [← (expand_tally_perm xs).length_eq]
  simp [Roll.expand, List.length_flatMap]

/-- A criterion that matches some face of a die finds faces. -/
private theorem faces_ne_nil {d : Die} {c : Criterion}
    (hm : d.faces.countP c.matches ≠ 0) : d.faces.length ≠ 0 := by
  intro h
  rw [List.eq_nil_of_length_eq_zero h] at hm
  exact hm rfl

/--
An explode limited to `n + 1` times, that matches some face, extends the
chains whose ends it matches.

# Parameters
- `whole`: Whether the chains stay whole.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem resolve_some {d : Die} {c : Criterion} {n : Nat} (whole : Bool)
    (hm : d.faces.countP c.matches ≠ 0) :
    resolve d c (some (n + 1)) whole = .extend (d.faces.length ^ (n + 1))
      (if whole then
        .tails (growths (tally d.faces) c tailExtend tailClose n [])
      else .folds (growths (tally d.faces) c foldExtend foldExtend n {})) := by
  simp only [resolve, hm, ↓reduceIte]

/--
Every chain that a chain still kept may become after an explode limited to
`n + 1` times weighs `Fⁿ⁺¹` in all, for a die of `F` faces.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem sum_chains {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (ch : Chain) :
    (((resolve d c (some (n + 1)) true).chains c ch).map (·.2)).sum =
      d.faces.length ^ (n + 1) := by
  rw [resolve_some true hm]
  simp only [↓reduceIte, chains]
  split
  · simp only [List.map_map, Function.comp_def]
    rw [sum_growths, sum_tally]
  · simp [padding]

/--
What a chain still kept becomes after an explode limited to `n + 1` times, over
its weight, is the specification's: grown by its extension, if the explode
matches its end, or else itself. Mirrors `Burst::chains` in
`propagation/chain.rs`.

# Parameters
- `dflt`: The chain if no weight were positive.
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem weighted_chains {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (dflt ch : Chain) :
    weighted dflt ((resolve d c (some (n + 1)) true).chains c ch) =
      if ch.grows c then
        (extension c (customDie d.faces) (n + 1)).map (growChain ch)
      else PMF.pure ch := by
  have hF := faces_ne_nil hm
  have hT : ((tally d.faces).map (·.2)).sum ≠ 0 := by rwa [sum_tally]
  rw [resolve_some true hm]
  simp only [↓reduceIte, chains]
  split
  · rw [weighted_congr_default dflt (growChain ch ([], 0)) (by
        simp only [List.map_map, Function.comp_def]
        rw [sum_growths]
        exact pow_ne_zero _ hT),
      weighted_map ([], 0) _ (growChain ch) (by
        rw [sum_growths]; exact pow_ne_zero _ hT),
      weighted_growths _ c _ _ hT, weightedDie_tally,
      extension_eq_growthLaw]
    rfl
  · exact weighted_single _ _ (by simp [padding, hF])

/--
Every total that a chain still kept may reach after an explode limited to
`n + 1` times weighs `Fⁿ⁺¹` in all, for a die of `F` faces.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem sum_totals {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (ch : Chain) :
    (((resolve d c (some (n + 1)) false).totals c ch).map (·.2)).sum =
      d.faces.length ^ (n + 1) := by
  rw [resolve_some false hm]
  simp only [Bool.false_eq_true, ↓reduceIte, totals]
  split
  · simp only [List.map_map, Function.comp_def]
    rw [sum_growths, sum_tally]
  · simp [padding]

/--
The total that a chain still kept reaches after an explode limited to `n + 1`
times, over its weight, is the total of the specification's: of the chain
grown by its extension, if the explode matches its end, or else its own.
Mirrors `Burst::totals` in `propagation/chain.rs`, which folds the tails.

# Parameters
- `dflt`: The total if no weight were positive.
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem weighted_totals {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (dflt : Int) (ch : Chain) :
    weighted dflt ((resolve d c (some (n + 1)) false).totals c ch) =
      if ch.grows c then
        (extension c (customDie d.faces) (n + 1)).map
          fun t => (growChain ch t).total
      else PMF.pure ch.total := by
  have hF := faces_ne_nil hm
  have hT : ((tally d.faces).map (·.2)).sum ≠ 0 := by rwa [sum_tally]
  rw [resolve_some false hm]
  simp only [Bool.false_eq_true, ↓reduceIte, totals]
  split
  · rw [weighted_congr_default dflt (((chainFold ch).add {}).total) (by
        simp only [List.map_map, Function.comp_def]
        rw [sum_growths]
        exact pow_ne_zero _ hT),
      weighted_map {} _ (fun f => ((chainFold ch).add f).total) (by
        rw [sum_growths]; exact pow_ne_zero _ hT),
      weighted_growths _ c _ _ hT, weightedDie_tally, growthLaw_folds,
      extension_eq_growthLaw, PMF.map_comp]
    exact congrArg (PMF.map · _) (funext fun t => total_growChain ch t)
  · exact weighted_single _ _ (by simp [padding, hF])

end Burst

/-! ## Records of chains -/

/--
Whether a chain is as every chain of `propagation/chain.rs` is: the dice of a
chain that may still grow besides its last ascend, as the oracle keeps them,
and every die lies within the `i32` range. Every chain that a roll or an
explode of such a die leaves is (`normalChain_growChain`). Of two such chains,
neither precedes the other unless they are equal (`normalChain_antisymm`), so
that chains sort alike in any order (`ofChains_perm`).

# Parameters
- `ch`: The chain.

# Returns
Whether the chain is normal.
-/
def NormalChain : Chain → Prop
  | .live body last =>
    body.Pairwise (· ≤ ·) ∧
      ∀ x ∈ last :: body, i32Min ≤ x ∧ x ≤ i32Max
  | .saturated _ => True

/--
Every chain precedes itself.

# Parameters
- `a`: The chain.
-/
theorem chain_le_refl (a : Chain) : a.le a = true := by
  rcases Chain.le_total a a with h | h <;> exact h

/--
Of two normal chains, each of which precedes the other, both are the same.

# Hypotheses
- `ha`, `hb`: The chains are normal.
- `hab`, `hba`: Each precedes the other.
-/
theorem normalChain_antisymm {a b : Chain} (ha : NormalChain a)
    (hb : NormalChain b) (hab : a.le b = true) (hba : b.le a = true) :
    a = b := by
  simp only [Chain.le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
    beq_iff_eq, Bool.not_eq_true'] at hab hba
  have irr : ∀ l : List Int, ¬l < l := fun l => List.lt_irrefl l
  rcases hab with h1 | ⟨e1, h1 | ⟨d1, h1 | ⟨l1, s1⟩⟩⟩ <;>
  rcases hba with h2 | ⟨e2, h2 | ⟨d2, h2 | ⟨l2, s2⟩⟩⟩
  all_goals first
    | omega
    | exact absurd (List.lt_trans h1 h2) (irr _)
    | exact absurd (d2 ▸ h1) (irr _)
    | exact absurd (d1 ▸ h2) (irr _)
    | skip
  -- Equal totals, dice, and last dice, and alike saturation.
  have hs : a.saturates = b.saturates := by
    cases hsa : a.saturates <;> cases hsb : b.saturates <;> simp_all
  cases a with
  | live ba la =>
    cases b with
    | live bb lb =>
      simp only [Chain.last] at l1
      subst l1
      simp only [Chain.dice] at d1
      have hp : ba.Perm bb := by
        have hb' := List.mergeSort_perm (la :: bb) (fun a b => decide (a ≤ b))
        rw [← d1] at hb'
        exact ((List.mergeSort_perm _ _).symm.trans hb').cons_inv
      rw [hp.eq_of_pairwise (fun a b _ _ h₁ h₂ => Int.le_antisymm h₁ h₂)
        ha.1 hb.1]
    | saturated v => simp [Chain.saturates] at hs
  | saturated v =>
    cases b with
    | live bb lb => simp [Chain.saturates] at hs
    | saturated w =>
      simp only [Chain.last] at l1
      rw [l1]

/--
Chains sorted by `Xdy.Chain.le` are sorted.

# Parameters
- `l`: The chains.
-/
theorem pairwise_mergeSort_chains (l : List Chain) :
    (l.mergeSort Chain.le).Pairwise (fun a b => a.le b = true) :=
  List.pairwise_mergeSort (le := Chain.le) (fun _ _ _ => Chain.le_trans)
    (fun a b => by simpa only [Bool.or_eq_true] using Chain.le_total a b) l

/--
Normal chains sort alike in any order.

# Parameters
- `l`, `l'`: The chains.

# Hypotheses
- `hp`: The chains are the same, in some order.
- `hn`: Every chain is normal.
-/
theorem mergeSort_chains_perm {l l' : List Chain} (hp : l.Perm l')
    (hn : ∀ x ∈ l, NormalChain x) :
    l.mergeSort Chain.le = l'.mergeSort Chain.le :=
  List.Perm.eq_of_pairwise
    (fun a b ha hb hab hba => normalChain_antisymm
      (hn a ((List.mergeSort_perm _ _).mem_iff.mp ha))
      (hn b (hp.mem_iff.mpr ((List.mergeSort_perm _ _).mem_iff.mp hb))) hab hba)
    (pairwise_mergeSort_chains l) (pairwise_mergeSort_chains l')
    ((List.mergeSort_perm _ _).trans (hp.trans (List.mergeSort_perm _ _).symm))

/--
A record of normal chains depends only on their multiset.

# Parameters
- `die`: The die.
- `l`, `l'`: The chains.

# Hypotheses
- `hp`: The chains are the same, in some order.
- `hn`: Every chain is normal.
-/
theorem ofChains_perm (die : Die) {l l' : List Chain} (hp : l.Perm l')
    (hn : ∀ x ∈ l, NormalChain x) :
    Record.ofChains die l = Record.ofChains die l' := by
  simp only [Record.ofChains, Xdy.Record.ofChains, mergeSort_chains_perm hp hn]

/--
A chain of one die within the `i32` range totals its die.

# Hypotheses
- `h`: The die lies within the `i32` range.
-/
theorem total_single {x : Int} (h : i32Min ≤ x ∧ x ≤ i32Max) :
    (Chain.single x).total = x := by
  simp only [Chain.single, Chain.total, Chain.dice, List.mergeSort_singleton,
    List.foldl_cons, List.foldl_nil, Xdy.add, clamp]
  simp only [i32Min, i32Max] at h ⊢
  omega

/--
The kept results of a record of chains are their totals, in some order.

# Parameters
- `die`: The die.
- `cs`: The chains.
-/
theorem kept_ofChains (die : Die) (cs : List Chain) :
    ((Record.ofChains die cs).kept : Multiset Int) = cs.map Chain.total := by
  simp only [Record.kept, Record.sorted, Record.ofChains, Record.ofOracle,
    Xdy.Record.ofChains, List.drop_zero, Nat.sub_zero, List.length_map]
  rw [show (cs.mergeSort Chain.le).length =
      ((cs.mergeSort Chain.le).map Chain.total).mergeSort.length by
    simp [List.length_mergeSort], List.take_length]
  exact Multiset.coe_eq_coe.mpr ((List.mergeSort_perm _ _).trans
    ((List.mergeSort_perm _ _).map _))

/--
The kept chains of a record of normal chains are the chains, in some order.

# Parameters
- `die`: The die.
- `cs`: The chains.

# Hypotheses
- `hn`: Every chain is normal.
-/
theorem keptChains_ofChains (die : Die) {cs : List Chain}
    (hn : ∀ x ∈ cs, NormalChain x) :
    ((Record.ofChains die cs).keptChains : Multiset Chain) = cs := by
  by_cases hall : (cs.mergeSort Chain.le).all Chain.isSingle
  · have hk : (Record.ofChains die cs).keptChains =
        (Record.ofChains die cs).kept.map .single := by
      simp [Record.keptChains, Record.ofChains, Record.ofOracle,
        Xdy.Record.ofChains, hall]
    rw [hk, ← Multiset.map_coe, kept_ofChains, Multiset.map_coe, List.map_map]
    refine Multiset.coe_eq_coe.mpr (List.Perm.of_eq ?_)
    refine (List.map_congr_left fun x hx => ?_).trans (List.map_id _)
    have hs : x.isSingle = true := List.all_eq_true.mp hall x
      ((List.mergeSort_perm _ _).mem_iff.mpr hx)
    cases x with
    | live body last =>
      cases body with
      | nil =>
        show Chain.single (Chain.single last).total = Chain.single last
        rw [total_single (by
          have := (hn _ hx).2 last List.mem_cons_self
          exact this)]
      | cons => simp [Chain.isSingle] at hs
    | saturated => simp [Chain.isSingle] at hs
  · have hne : (cs.mergeSort Chain.le).isEmpty = false := by
      cases h : cs.mergeSort Chain.le with
      | nil => simp [h] at hall
      | cons => rfl
    simp only [Record.keptChains, Record.ofChains, Record.ofOracle,
      Xdy.Record.ofChains, hall, Bool.false_eq_true, ↓reduceIte, hne,
      List.drop_zero, Nat.sub_zero, List.length_map]
    rw [List.mergeSort_of_pairwise (pairwise_mergeSort_chains cs),
      List.take_length]
    exact Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _)

namespace Record

/--
What a later reroll or explode reads of a record: its kept chains, as a
multiset, and its die. A reroll or explode of chains depends only on the
multiset of the kept chains (`explodeChains_eq_bag`), and the kept results of a
record of chains are their totals (`kept_ofChains`). The Rust forward pass
describes a record after an explode that keeps its chains whole only up to its
chain view.

# Parameters
- `r`: The record.

# Returns
The kept chains, as a multiset, and the die.
-/
def chainView (r : Record) : Multiset Chain × Die := (r.keptChains, r.die)

end Record

/-! ## Explodes of independent chains -/

/-- Exploding chains in place, from any law of chains before, adds the
exploded chains to the multisets of the chains before. -/
private theorem foldl_explode_bag (grows : Chain → Bool)
    (ext : PMF (List Int × Int)) :
    ∀ (kept : List Chain) (p : PMF (List Chain)),
      (kept.foldl
          (fun (p : PMF (List Chain)) c =>
            if grows c then
              p.bind fun cs => ext.map fun (drawn, y) =>
                (.live (c.dice ++ drawn).mergeSort y :: cs).mergeSort Chain.le
            else p.map fun cs => (c :: cs).mergeSort Chain.le) p).map
          (Multiset.ofList : List Chain → Multiset Chain) =
        lift (· + ·) (p.map (Multiset.ofList : List Chain → Multiset Chain))
          (bagDraws (fun ch => if grows ch then ext.map (growChain ch)
            else PMF.pure ch) kept)
  | [], p => by
    simp only [List.foldl_nil, bagDraws, lift, PMF.pure_map, add_zero]
    exact (PMF.bind_pure _).symm
  | c :: kept, p => by
    rw [List.foldl_cons, foldl_explode_bag grows ext kept]
    have hsort : ∀ (x : Chain) (cs : List Chain),
        (((x :: cs).mergeSort Chain.le : List Chain) : Multiset Chain) =
          x ::ₘ (cs : Multiset Chain) := fun x cs =>
      Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _)
    have hstep : ((if grows c then
          p.bind fun cs => ext.map fun (drawn, y) =>
            (.live (c.dice ++ drawn).mergeSort y :: cs).mergeSort Chain.le
        else p.map fun cs => (c :: cs).mergeSort Chain.le).map
          (Multiset.ofList : List Chain → Multiset Chain)) =
        p.bind fun cs => (if grows c then ext.map (growChain c)
          else PMF.pure c).map fun x => x ::ₘ (cs : Multiset Chain) := by
      split
      · simp only [PMF.map_bind, PMF.map_comp]
        refine congrArg _ (funext fun cs => congrArg (PMF.map · ext)
          (funext fun t => ?_))
        simp only [Function.comp_apply, hsort]
        rfl
      · simp only [PMF.map_comp, PMF.pure_map, Function.comp_def, hsort]
        exact (PMF.bind_pure_comp _ _).symm
    rw [hstep]
    simp only [bagDraws, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      PMF.bind_bind, Function.comp_def]
    refine congrArg _ (funext fun cs => congrArg _ (funext fun y =>
      congrArg (PMF.map · _) (funext fun M => ?_)))
    rw [← Multiset.singleton_add, ← Multiset.singleton_add, add_assoc,
      add_left_comm]

/-- Exploding normal chains in place, from normal chains, leaves normal
chains. -/
private theorem foldl_explode_normal (grows : Chain → Bool)
    (ext : PMF (List Int × Int)) :
    ∀ (kept : List Chain) (p : PMF (List Chain)),
      (∀ ch ∈ kept, NormalChain ch) →
      (∀ ch ∈ kept, grows ch → ∀ t ∈ ext.support,
        NormalChain (growChain ch t)) →
      (∀ cs ∈ p.support, ∀ x ∈ cs, NormalChain x) →
      ∀ cs ∈ (kept.foldl
          (fun (p : PMF (List Chain)) c =>
            if grows c then
              p.bind fun cs => ext.map fun (drawn, y) =>
                (.live (c.dice ++ drawn).mergeSort y :: cs).mergeSort Chain.le
            else p.map fun cs => (c :: cs).mergeSort Chain.le) p).support,
        ∀ x ∈ cs, NormalChain x
  | [], _, _, _, hp => hp
  | c :: kept, p, hn, hg, hp => by
    rw [List.foldl_cons]
    refine foldl_explode_normal grows ext kept _
      (fun ch h => hn ch (List.mem_cons_of_mem _ h))
      (fun ch h => hg ch (List.mem_cons_of_mem _ h)) ?_
    intro cs hcs x hx
    split at hcs
    · rename_i hgc
      simp only [PMF.mem_support_bind_iff, PMF.mem_support_map_iff] at hcs
      obtain ⟨cs', hcs', t, ht, rfl⟩ := hcs
      rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
        rfl | hx
      · exact hg c List.mem_cons_self hgc t ht
      · exact hp cs' hcs' x hx
    · simp only [PMF.mem_support_map_iff] at hcs
      obtain ⟨cs', hcs', rfl⟩ := hcs
      rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
        rfl | hx
      · exact hn _ List.mem_cons_self
      · exact hp cs' hcs' x hx

/--
An explode of normal chains, each of which it grows into a normal chain, leaves
independent draws, one for each chain: grown by its extension, if it grows,
or else itself. So the record that it leaves depends only on the multiset of
the chains, which `Burst::chains` in `propagation/chain.rs` transforms one
distinct chain at a time.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `grows`: Whether the explode grows a chain.
- `ext`: The law of the extension of a chain that grows.

# Hypotheses
- `hn`: Every kept chain is normal.
- `hg`: Every chain that the explode grows grows into a normal chain.
-/
theorem explodeChains_eq_bag (die : Die) {kept : List Chain}
    (grows : Chain → Bool) (ext : PMF (List Int × Int))
    (hn : ∀ ch ∈ kept, NormalChain ch)
    (hg : ∀ ch ∈ kept, grows ch → ∀ t ∈ ext.support,
      NormalChain (growChain ch t)) :
    explodeChains die kept grows ext =
      (bagDraws (fun ch => if grows ch then ext.map (growChain ch)
        else PMF.pure ch) kept).map fun M => Record.ofChains die M.toList := by
  have hb := foldl_explode_bag grows ext kept (PMF.pure [])
  simp only [PMF.pure_map, lift, PMF.pure_bind, Multiset.coe_nil,
    zero_add] at hb
  unfold explodeChains
  rw [map_congr_support (g := (fun M : Multiset Chain =>
      Record.ofChains die M.toList) ∘
        (Multiset.ofList : List Chain → Multiset Chain))
    fun cs hcs => ofChains_perm die (Multiset.coe_eq_coe.mp
      (Multiset.coe_toList _).symm)
      (foldl_explode_normal grows ext kept _ hn hg (by simp) cs hcs),
    ← PMF.map_comp, hb, show (fun M : Multiset Chain => M) = id from rfl,
    PMF.map_id]

/-! ## Independent draws -/

/--
Independent results, each drawn anew, are independent draws of the law of the
results drawn anew, as `bind_rollDice_bagDraws` in `XdySpec/Reroll.lean`
finds of rerolls, here of draws of any kind, such as chains.

# Parameters
- `n`: The number of dice.
- `die`: The law of each die.
- `step`: The law of what each result becomes.
-/
theorem bind_rollDice_iidBag {β : Type} (n : Int) (die : PMF Int)
    (step : Int → PMF β) :
    (rollDice n die).bind (fun r => bagDraws step r.results) =
      iidBag n.toNat (die.bind step) := by
  rw [OrderStatistics.rollDice_toNat n die]
  induction n.toNat with
  | zero => simp [rollDice, Nat.repeat, bagDraws, iidBag]
  | succ J ih =>
    rw [OrderStatistics.rollDice_succ, PMF.bind_bind]
    have hpush : ∀ (r : Record) (x : Int), bagDraws step (r.push x).results =
        lift (· ::ₘ ·) (step x) (bagDraws step r.results) := fun r x =>
      bagDraws_perm step (List.perm_append_singleton x r.results)
    calc _ = (rollDice (J : Int) die).bind fun r => (die.bind step).bind
            fun y => (bagDraws step r.results).map (y ::ₘ ·) := by
          refine congrArg _ (funext fun r => ?_)
          simp only [PMF.bind_map, Function.comp_def, hpush, lift,
            PMF.bind_bind]
      _ = (die.bind step).bind fun y => ((rollDice (J : Int) die).bind
            fun r => bagDraws step r.results).map (y ::ₘ ·) := by
          rw [PMF.bind_comm]
          simp only [PMF.map_bind]
      _ = _ := by
          rw [ih]
          show _ = (iidBag J (die.bind step)).bind
            fun M => (die.bind step).map (· ::ₘ M)
          simp only [← PMF.bind_pure_comp]
          exact PMF.bind_comm _ _ _

/--
Results drawn from a list, each by a law of what it becomes, are those drawn
from the list mapped.

# Parameters
- `f`: The law of what each value becomes.
- `g`: The map of the list.
- `l`: The list.
-/
theorem bagDraws_map {α β γ : Type} (f : α → PMF β) (g : γ → α)
    (l : List γ) :
    bagDraws f (l.map g) = bagDraws (f ∘ g) l := by
  induction l with
  | nil => rfl
  | cons x l ih => simp only [List.map_cons, bagDraws, ih, Function.comp_apply]

/-! ## Exploding a setup -/

/--
A law transformed without padding weighs what each chain becomes by the
chain's weight alone, as an explode transforms it.

# Type parameters
- `T`: The type of what a chain becomes.

# Parameters
- `law`: The distinct chains that each may be, each with its weight.
- `h`: What each chain becomes, each with its weight.
-/
theorem transformLaw_one {T : Type} (law : List (Chain × Nat))
    (h : Chain → List (T × Nat)) :
    transformLaw law 1 (fun ch => (h ch, 0)) =
      law.flatMap fun e => (h e.1).map fun o => (o.1, e.2 * o.2) := by
  simp [transformLaw]

namespace Roll

/--
Whether a roll's chains are independent and alike, as `Roll::explode` in
`propagation/chain.rs` asks before it transforms the law of each: a range,
dice, or i.i.d. chains.

# Parameters
- `ρ`: The roll.

# Returns
`true` for a range, dice, or chains, `false` for fixed results, a pool, or
stacks.
-/
def chainLike : Roll → Bool
  | .range .. | .standard .. | .custom .. | .chains .. => true
  | .fixed _ | .pool _ | .stacked .. => false

/--
The law of one chain of a roll whose chains are independent and alike, and
their number. Mirrors `Roll::chain_law` in `propagation/chain.rs`: each face
of the dice is the chain of one die.

# Parameters
- `ρ`: The roll.

# Returns
The number of chains and the distinct chains that each may be, with their
weights, or no chains for fixed results, a pool, or stacks.
-/
def chainLaw : Roll → Int × List (Chain × Nat)
  | .chains count law => (count, law)
  | ρ => (ρ.dice.1, ρ.dice.2.map fun p => (.single p.1, p.2))

/--
Whether the end of some chain of a roll, kept or dropped, may show a result
that a criterion matches. Mirrors `Roll::shows` in `propagation/chain.rs`.

# Parameters
- `ρ`: The roll.
- `c`: The criterion.

# Returns
`true` if some end may match, `false` otherwise.
-/
def shows (ρ : Roll) (c : Criterion) : Bool :=
  match ρ with
  | .range .. | .standard .. => ρ.replacement.faces.any c.matches
  | .custom _ faces =>
    if faces = [] then c.matches 0 else faces.any (c.matches ·.1)
  | .fixed results => results.any (c.matches ·.1)
  | .pool outcomes => outcomes.any fun e => e.1.any (c.matches ·.1)
  | .chains _ law => law.any (·.1.grows c)
  | .stacked multisets laws => multisets.any fun e => e.1.any fun p =>
    match laws.lookup p.1 with
    | some law => law.any (·.1.grows c)
    | none => p.1.grows c

/--
Explode the chains still kept of each multiset of the chains of a roll, after
its lowest `lowest` and its highest `highest` chains are dropped, as the burst
directs. Mirrors the last arm of `Roll::explode` in `propagation/chain.rs`,
which conditions exactly on the chains still kept: if `whole`, into stacks of
the kept pool of the roll's multisets of chains, each of whose chains has
become what the burst makes of it; and otherwise into a pool of the multisets
of the totals of what the chains still kept become.

# Parameters
- `ρ`: The roll.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
- `b`: The burst.
- `c`: The criterion.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Returns
The exploded roll: stacks if `whole`, and a pool otherwise.

# Notes
The Rust joins the totals of each multiset as a sorted map; this tallies
them, which sorts them and merges those that coincide.
-/
def explodeMultisets (ρ : Roll) (lowest highest : Nat) (b : Burst)
    (c : Criterion) (whole : Bool) : Roll :=
  if whole then
    transformStacks (keptPool ρ.stacks lowest highest) [] 1
      fun ch => (b.chains c ch, 0)
  else
    .pool ((transformPool ρ.stacks lowest highest 1
      fun ch => (b.totals c ch, 0)).map fun e => (tally (expand e.1), e.2))

/--
Explode stacks none of whose chains has been dropped, as the burst directs,
without joining them. Mirrors the second arm of `Roll::explode` in
`propagation/chain.rs`: if `whole`, into stacks that compose the law of what
each chain of the pool has become with its explode (`transformStacks`); and
otherwise into a pool of the multisets of the totals of what each copy of
each chain of the pool becomes, each drawn from the law of what it had
become composed with the totals of its explode (`compose`).

# Parameters
- `ms`: The multisets of the stacks, each with its branches.
- `laws`: The law of what each distinct chain of the pool has become, or none
  for a chain that is still itself.
- `b`: The burst.
- `c`: The criterion.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Returns
The exploded roll: stacks if `whole`, and a pool otherwise.

# Notes
The Rust joins the totals of each multiset as a sorted map; this tallies
them, which sorts them and merges those that coincide.
-/
def explodeStacks (ms : List (List (Chain × Nat) × Nat))
    (laws : List (Chain × List (Chain × Nat))) (b : Burst) (c : Criterion)
    (whole : Bool) : Roll :=
  if whole then transformStacks ms laws 1 fun ch => (b.chains c ch, 0)
  else
    .pool ((transformPool ms 0 0 1
      (compose laws 1 fun ch => (b.totals c ch, 0))).map fun e =>
        (tally (expand e.1), e.2))

/--
Explode the chains still kept of a roll, after its lowest `lowest` and its
highest `highest` chains are dropped, as the burst directs, which neither
does nothing nor saturates. Mirrors `Roll::explode` in
`propagation/chain.rs`: if none has been dropped and the roll's chains are
independent and alike, the explode transforms the law of each, into i.i.d.
chains if `whole`, and into custom dice of their totals otherwise; if the
roll is stacks none of whose chains has been dropped, it composes the law of
what each chain of their pool has become with its explode
(`explodeStacks`); and if the roll is fixed results or a pool, or its chains
have been dropped, it transforms each multiset of the chains still kept
(`explodeMultisets`).

# Parameters
- `ρ`: The roll.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
- `b`: The burst.
- `c`: The criterion.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Returns
The exploded roll.
-/
def explode (ρ : Roll) (lowest highest : Nat) (b : Burst) (c : Criterion)
    (whole : Bool) : Roll :=
  if ρ.chainLike ∧ lowest = 0 ∧ highest = 0 then
    if whole then
      .chains ρ.chainLaw.1
        (transformLaw ρ.chainLaw.2 1 fun ch => (b.chains c ch, 0))
    else
      .custom ρ.chainLaw.1
        (tally (expand
          (transformLaw ρ.chainLaw.2 1 fun ch => (b.totals c ch, 0))))
  else
    match ρ with
    | .stacked ms laws =>
      if lowest = 0 ∧ highest = 0 then explodeStacks ms laws b c whole
      else ρ.explodeMultisets lowest highest b c whole
    | _ => ρ.explodeMultisets lowest highest b c whole

/--
Saturate every chain still kept of a roll, which keeps its total weight,
since the explode that saturates them rolls nothing. Mirrors
`Roll::saturate` in `propagation/chain.rs`.

# Parameters
- `ρ`: The roll.
- `value`: The saturated value of the die.
- `kept`: The number of chains still kept.
- `whole`: Whether the chains stay whole, or reduce to their totals.

# Returns
Stacks if `whole`, and a pool otherwise, of one multiset of `kept` copies of
the saturated value, weighing the total of the roll.
-/
def saturate (ρ : Roll) (value : Int) (kept : Nat) (whole : Bool) : Roll :=
  if whole then .stacked [([(.saturated value, kept)], ρ.total)] []
  else .pool [([(value, kept)], ρ.total)]

/--
Reduce the chains of a roll to their totals, which behave as the results of
dice do. Mirrors `Roll::collapse` in `propagation/chain.rs`: i.i.d. chains
become custom dice of their totals, and stacks a pool of the multisets of
their totals (`joinedTotals`).

# Parameters
- `ρ`: The roll.

# Returns
The reduced roll, or the roll itself if it holds no chains.

# Notes
The Rust joins the totals of each multiset as a sorted map; this tallies
them, which sorts them and merges those that coincide.
-/
def collapse : Roll → Roll
  | .chains count law =>
    .custom count (tally (expand (law.map fun e => (e.1.total, e.2))))
  | .stacked ms laws =>
    .pool ((joinedTotals ms laws).map fun e => (tally (expand e.1), e.2))
  | ρ => ρ

end Roll

namespace Setup

/--
Explode the chains still kept of a setup. Mirrors `Setup::explode` in
`propagation/mixture.rs`: without chains still kept, with only saturated
chains, or if the explode does nothing, the setup is left as it is; an
unlimited explode that matches every face saturates every chain still kept;
and otherwise the roll explodes, and the setup keeps its die and drops
nothing.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole, or reduce to their totals.

# Returns
The exploded setup.

# Notes
An unlimited explode that does not match every face, and may match the end of
a chain still kept, the pass refuses before it begins, and the Rust panics
should it reach one; this leaves the setup alone, and the laws ask that it
not arise (`Setup.Explodes`).
-/
def explode (s : Setup) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Setup :=
  match s.roll with
  | none => s
  | some ρ =>
    if s.results - s.lowest - s.highest = 0 ∨ ρ.isSaturated then s
    else
      match Burst.resolve s.replacement c limit whole with
      | .nothing | .unbounded => s
      | .saturate value =>
        { roll := some
            (ρ.saturate value (s.results - s.lowest - s.highest) whole)
          die := some s.replacement }
      | b =>
        { roll := some (ρ.explode s.lowest s.highest b c whole)
          die := some s.replacement }

/--
Reduce the chains of a setup's roll to their totals, as when no later reroll
or explode reads them. Mirrors `Setup::collapse` in `propagation/mixture.rs`.

# Parameters
- `s`: The setup.

# Returns
The reduced setup.
-/
def collapse (s : Setup) : Setup := { s with roll := s.roll.map Roll.collapse }

/--
Explode a setup as `Mixture::explode` in `propagation/mixture.rs` does:
explode it, and reduce its chains to their totals unless they stay whole.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether a later reroll or explode reads the record before a roll
  refills it, so that its chains stay whole.

# Returns
The exploded setup.
-/
def burst (s : Setup) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Setup :=
  if whole then s.explode c limit whole else (s.explode c limit whole).collapse

/--
Whether a setup is one whose chains may still grow, as every grounded setup
but stacks of saturated chains is (`Setup.Grounded`): the faces of its die lie
within the `i32` range; a roll of i.i.d. chains holds the die from which they
were drawn, as every explode leaves it, and each chain that it may be is
normal, may still grow, and holds only faces of the die, whatever its drops;
stacks hold the die too, and each chain of their pool, and each that it may
have become, is likewise; and any other roll holds only faces of the die.

# Parameters
- `s`: The setup.

# Returns
Whether the setup is live.
-/
def Live (s : Setup) : Prop :=
  (∀ x ∈ s.replacement.faces, i32Min ≤ x ∧ x ≤ i32Max) ∧
    ∀ ρ ∈ s.roll, match ρ with
      | .chains _ law => s.die.isSome ∧
        ∀ e ∈ law, NormalChain e.1 ∧ e.1.saturates = false ∧
          ∀ x ∈ e.1.dice, x ∈ s.replacement.faces
      | .stacked ms laws => s.die.isSome ∧
        (∀ e ∈ ms, ∀ p ∈ e.1, NormalChain p.1 ∧ p.1.saturates = false ∧
          ∀ x ∈ p.1.dice, x ∈ s.replacement.faces) ∧
        ∀ e ∈ laws, ∀ q ∈ e.2, NormalChain q.1 ∧ q.1.saturates = false ∧
          ∀ x ∈ q.1.dice, x ∈ s.replacement.faces
      | _ => ∀ r ∈ s.law.support, ∀ x ∈ r.results,
        x ∈ s.replacement.faces

/--
Whether a setup is stacks of saturated chains, as an unlimited explode that
matches every face leaves them, and every later explode and reroll leaves
them: a saturated die decides its chains (`Roll.isSaturated`), and every
chain of their pool, and each that it may have become, has saturated.

# Parameters
- `s`: The setup.

# Returns
Whether the setup is saturated.
-/
def Saturated (s : Setup) : Prop :=
  ∃ ms laws, s.roll = some (.stacked ms laws) ∧
    (Roll.stacked ms laws).isSaturated = true ∧
    (∀ e ∈ ms, ∀ p ∈ e.1, p.1.saturates = true) ∧
    ∀ e ∈ laws, ∀ q ∈ e.2, q.1.saturates = true

/--
Whether a setup is one whose explodes and rerolls the specification
describes: one whose chains may still grow (`Setup.Live`), or stacks of
saturated chains (`Setup.Saturated`), which no explode or reroll changes.
Every explode and reroll that keeps its chains whole leaves a grounded setup
grounded (`Setup.burst_invariants`, `Setup.reroll_invariants`).

# Parameters
- `s`: The setup.

# Returns
Whether the setup is grounded.
-/
def Grounded (s : Setup) : Prop := s.Live ∨ s.Saturated

/--
Whether an explode of a setup is one whose law the pass describes: not an
unlimited explode that may extend a chain still kept, which the pass refuses
before it begins.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.

# Returns
Whether the explode is described.
-/
def Explodes (s : Setup) (c : Criterion) (limit : Option Nat) : Prop :=
  ∀ ρ ∈ s.roll, s.results - s.lowest - s.highest ≠ 0 →
    match Burst.resolve s.replacement c limit true with
    | .unbounded => ρ.shows c = false
    | _ => True

/--
Whether an explode that extends the chains still kept of a setup extends the
independent and alike chains of a range, dice, or i.i.d. chains none of which
has been dropped, which `Roll::explode` in `propagation/chain.rs` transforms
chain by chain, and this module describes; `XdySpec/Stacks.lean` describes
the rest, which it transforms multiset by multiset.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.

# Returns
Whether every explode that extends extends alike chains.
-/
def ExtendsAlike (s : Setup) (c : Criterion) (limit : Option Nat) : Prop :=
  ∀ ρ ∈ s.roll, s.results - s.lowest - s.highest ≠ 0 →
    match Burst.resolve s.replacement c limit true with
    | .extend .. => ρ.chainLike ∧ s.lowest = 0 ∧ s.highest = 0
    | _ => True

end Setup

/-! ## Records that an explode reads -/

/--
Every saturated chain is normal.

# Parameters
- `ch`: The chain.

# Hypotheses
- `h`: The chain has saturated.
-/
theorem normal_of_saturates {ch : Chain} (h : ch.saturates = true) :
    NormalChain ch := by
  cases ch with
  | live => simp [Chain.saturates] at h
  | saturated => trivial

namespace Record

/-- What every record that a live setup describes is, as an explode reads
it (`Setup.fits_of_live`): it holds its die, its kept chains are normal,
may still grow, and hold only faces of the die, and its kept results are their
totals. -/
structure Fits (D : Die) (r : Record) : Prop where
  /-- The record holds the die. -/
  die : r.die = D
  /-- Every kept chain is normal. -/
  normal : ∀ ch ∈ r.keptChains, NormalChain ch
  /-- Every kept chain may still grow. -/
  live : ∀ ch ∈ r.keptChains, ch.saturates = false
  /-- Every kept chain holds only faces of the die. -/
  faces : ∀ ch ∈ r.keptChains, ∀ x ∈ ch.dice, x ∈ D.faces
  /-- The kept results are the totals of the kept chains. -/
  kept : (r.kept : Multiset Int) = r.keptChains.map Chain.total

/-- Sorted lists of the same values are the same. -/
private theorem eq_of_coe_eq {xs ys : List Int} (hx : xs.Pairwise (· ≤ ·))
    (hy : ys.Pairwise (· ≤ ·)) (h : (xs : Multiset Int) = ys) : xs = ys :=
  (Multiset.coe_eq_coe.mp h).eq_of_pairwise
    (fun _ _ _ _ h₁ h₂ => Int.le_antisymm h₁ h₂) hx hy

/-- What every record that a saturated setup describes is
(`Setup.saturated_of_mem`): it holds its die, every kept chain has saturated,
and its kept results are their totals. -/
structure Saturated (D : Die) (r : Record) : Prop where
  /-- The record holds the die. -/
  die : r.die = D
  /-- Every kept chain has saturated. -/
  saturated : ∀ ch ∈ r.keptChains, ch.saturates = true
  /-- The kept results are the totals of the kept chains. -/
  kept : (r.kept : Multiset Int) = r.keptChains.map Chain.total

/-- A record of the kept chains of a record of its die, normal, whose kept
results are their totals, without drops, is the record, as every later
instruction views it. -/
private theorem views_of {D : Die} {r : Record} (hd : r.die = D)
    (hn : ∀ ch ∈ r.keptChains, NormalChain ch)
    (hk : (r.kept : Multiset Int) = r.keptChains.map Chain.total) :
    (ofChains D r.keptChains).chainView = r.chainView ∧
      (ofChains D r.keptChains).view = r.view := by
  have hd' : (ofChains D r.keptChains).die = D :=
    (Record.ofChains_fields D r.keptChains).2.2.2
  refine ⟨?_, ?_⟩
  · simp only [chainView, keptChains_ofChains D hn, hd', hd]
  · simp only [view, hd', hd, Prod.mk.injEq, and_true]
    refine eq_of_coe_eq (kept_pairwise _) (kept_pairwise _) ?_
    rw [kept_ofChains, hk]

/--
A record of the kept chains of a record that fits its die, without drops, is
the record, as every later instruction views it.

# Parameters
- `r`: The record.

# Hypotheses
- `h`: The record fits the die.
-/
theorem Fits.views {D : Die} {r : Record} (h : r.Fits D) :
    (ofChains D r.keptChains).chainView = r.chainView ∧
      (ofChains D r.keptChains).view = r.view :=
  views_of h.die h.normal h.kept

/--
A record of the kept chains of a saturated record, without drops, is the
record, as every later instruction views it.

# Parameters
- `r`: The record.

# Hypotheses
- `h`: The record is saturated.
-/
theorem Saturated.views {D : Die} {r : Record} (h : r.Saturated D) :
    (ofChains D r.keptChains).chainView = r.chainView ∧
      (ofChains D r.keptChains).view = r.view :=
  views_of h.die (fun ch hch => normal_of_saturates (h.saturated ch hch)) h.kept

end Record

/--
An explode of normal chains that grows none leaves the record of the chains.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `grows`: Whether the explode grows a chain.
- `ext`: The law of the extension of a chain that grows.

# Hypotheses
- `hn`: Every kept chain is normal.
- `hg`: The explode grows no kept chain.
-/
theorem explodeChains_stays (die : Die) {kept : List Chain}
    (grows : Chain → Bool) (ext : PMF (List Int × Int))
    (hn : ∀ ch ∈ kept, NormalChain ch)
    (hg : ∀ ch ∈ kept, grows ch = false) :
    explodeChains die kept grows ext = PMF.pure (Record.ofChains die kept) := by
  rw [explodeChains_eq_bag die grows ext hn (fun ch h hgr => by
      rw [hg ch h] at hgr; cases hgr),
    bagDraws_pure (fun ch h => by rw [hg ch h]; rfl), PMF.pure_map]
  exact congrArg PMF.pure (ofChains_perm die
    (Multiset.coe_eq_coe.mp (Multiset.coe_toList _))
    fun x hx => hn x
      ((Multiset.coe_eq_coe.mp (Multiset.coe_toList _)).mem_iff.mp hx))

/--
An explode that grows no kept chain of a record that fits its die, and does
not saturate any, leaves the record, as every later instruction views it.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The limit.

# Hypotheses
- `h`: The record fits the die.
- `hs`: An unlimited explode does not match every face, or nothing is kept.
- `hg`: The explode grows no kept chain, unless it is limited to no times.
-/
theorem explode_stays {D : Die} {r : Record} {c : Criterion}
    {limit : Option Nat} (h : r.Fits D)
    (hs : limit = none → ¬D.facesWithin c = true ∨ r.keptChains = [])
    (hg : ∀ ch ∈ r.keptChains, limit ≠ some 0 → ch.grows c = false) :
    explode r c limit = PMF.pure (Record.ofChains D r.keptChains) := by
  have hd := h.die
  cases limit with
  | none =>
    simp only [explode]
    split
    · rename_i hw
      rcases hs rfl with hs | hs
      · rw [hd] at hw; exact absurd hw hs
      · rw [hs, hd]; rfl
    · rw [explodeChains_stays _ _ _ h.normal fun ch hc => hg ch hc (by simp),
        hd]
  | some l =>
    simp only [explode]
    rw [explodeChains_stays _ _ _ h.normal fun ch hc => by
      by_cases hl : l = 0
      · simp [hl]
      · simp [hg ch hc (by simpa using hl)], hd]

/--
Every explode leaves a saturated record, as every later instruction views it:
it grows no saturated chain, and saturates none anew.

# Parameters
- `c`: The criterion.
- `limit`: The limit.

# Hypotheses
- `h`: The record is saturated.
-/
theorem Record.Saturated.explode {D : Die} {r : Record} (h : r.Saturated D)
    (c : Criterion) (limit : Option Nat) :
    Spec.explode r c limit = PMF.pure (Record.ofChains D r.keptChains) := by
  have hn : ∀ ch ∈ r.keptChains, NormalChain ch := fun ch hch =>
    normal_of_saturates (h.saturated ch hch)
  have hg : ∀ ch ∈ r.keptChains, ch.grows c = false := fun ch hch => by
    simp [Chain.grows, h.saturated ch hch]
  cases limit with
  | none =>
    simp only [Spec.explode]
    split
    · -- Saturating a saturated chain leaves it.
      have hsat : r.keptChains.map (·.saturate r.die.saturatedValue) =
          r.keptChains := by
        refine (List.map_congr_left fun ch hch => ?_).trans (List.map_id _)
        have := h.saturated ch hch
        cases ch with
        | live => simp [Chain.saturates] at this
        | saturated => rfl
      rw [hsat, h.die]
    · rw [explodeChains_stays _ _ _ hn hg, h.die]
  | some l =>
    simp only [Spec.explode]
    rw [explodeChains_stays _ _ _ hn fun ch hch => by simp [hg ch hch], h.die]

/--
An explode that saturates every kept chain of a record that fits its die
leaves as many saturated chains, as every later instruction views them.

# Parameters
- `r`: The record.
- `v`: The saturated value.

# Hypotheses
- `h`: The record fits the die.
-/
theorem Record.Fits.saturates {D : Die} {r : Record} (h : r.Fits D) (v : Int) :
    (Record.ofChains D (r.keptChains.map (·.saturate v))).chainView =
        (Multiset.replicate r.keptChains.length (.saturated v), D) ∧
      (Record.ofChains D (r.keptChains.map (·.saturate v))).view =
        (List.replicate r.keptChains.length v, D) := by
  have hrep : r.keptChains.map (·.saturate v) =
      List.replicate r.keptChains.length (.saturated v) := by
    refine List.eq_replicate_iff.mpr ⟨by simp, fun x hx => ?_⟩
    obtain ⟨ch, hch, rfl⟩ := List.mem_map.mp hx
    have := h.live ch hch
    cases ch with
    | live => rfl
    | saturated => simp [Chain.saturates] at this
  have hd := (Record.ofChains_fields D (r.keptChains.map (·.saturate v))).2.2.2
  rw [hrep] at hd ⊢
  refine ⟨?_, ?_⟩
  · have hk := keptChains_ofChains D
      (cs := List.replicate r.keptChains.length (Chain.saturated v))
      (fun x hx => by rw [List.eq_of_mem_replicate hx]; trivial)
    simp only [Record.chainView, hd, hk, Multiset.coe_replicate]
  · simp only [Record.view, hd, Prod.mk.injEq, and_true]
    have hk := kept_ofChains D (List.replicate r.keptChains.length
      (Chain.saturated v))
    simp only [List.map_replicate] at hk
    refine (Multiset.coe_eq_coe.mp hk).eq_of_pairwise
      (fun _ _ _ _ h₁ h₂ => Int.le_antisymm h₁ h₂)
      (Record.kept_pairwise _) ?_
    exact List.pairwise_replicate.mpr (.inr (by rfl))

/-! ## Windows of chains -/

/--
A chain of one die precedes another exactly when its die is no greater, within
the `i32` range.

# Hypotheses
- `hx`, `hy`: The dice lie within the `i32` range.
- `h`: The first die is no greater than the second.
-/
theorem single_le_single {x y : Int} (hx : i32Min ≤ x ∧ x ≤ i32Max)
    (hy : i32Min ≤ y ∧ y ≤ i32Max) (h : x ≤ y) :
    (Chain.single x).le (Chain.single y) = true := by
  rcases h.lt_or_eq with h | rfl
  · have htx : (Chain.single x).total = x := total_single hx
    have hty : (Chain.single y).total = y := total_single hy
    simp [Chain.le, htx, hty, h]
  · exact chain_le_refl _

namespace Record

/--
The normal kept chains of a record are sorted by `Xdy.Chain.le`.

# Parameters
- `r`: The record.

# Hypotheses
- `hn`: Every kept chain is normal.
-/
theorem keptChains_pairwise (r : Record)
    (hn : ∀ ch ∈ r.keptChains, NormalChain ch) :
    r.keptChains.Pairwise (fun a b => a.le b = true) := by
  unfold keptChains at hn ⊢
  split
  · rename_i hc
    simp only [hc, ↓reduceIte] at hn
    have hk : ∀ x ∈ r.kept, i32Min ≤ x ∧ x ≤ i32Max := fun x hx =>
      (hn _ (List.mem_map_of_mem hx)).2 x List.mem_cons_self
    refine List.pairwise_map.mpr ((kept_pairwise r).imp_of_mem ?_)
    intro a b ha hb hab
    exact single_le_single (hk a ha) (hk b hb) hab
  · exact ((pairwise_mergeSort_chains _).sublist (List.drop_sublist _ _)).sublist
      (List.take_sublist _ _)

/--
A record that drops nothing, and holds one result for each chain if it holds
any, keeps one chain for each result.

# Parameters
- `r`: The record.

# Hypotheses
- `hl`, `hh`: The record drops nothing.
- `hlen`: A record of chains holds one result for each chain.
-/
theorem length_keptChains_undropped {r : Record} (hl : r.lowest = 0)
    (hh : r.highest = 0)
    (hlen : r.chains ≠ [] → r.chains.length = r.results.length) :
    r.keptChains.length = r.results.length := by
  unfold keptChains
  split
  · rw [List.length_map, length_kept, hl, hh]
    rfl
  · rename_i hc
    have hc : r.chains ≠ [] := fun h => hc (by simp [h])
    simp [hl, hh, List.length_mergeSort, hlen hc]

/--
Dropping chains from a record that drops none keeps the window of its kept
chains, sorted by `Xdy.Chain.le`, after the lowest and highest dropped.

# Parameters
- `r`: The record.
- `L`, `H`: The numbers of lowest and highest chains dropped.

# Hypotheses
- `hl`, `hh`: The record drops nothing.
- `hlen`: A record of chains holds one result for each chain.
-/
theorem keptChains_drops {r : Record} (L H : Nat) (hl : r.lowest = 0)
    (hh : r.highest = 0)
    (hlen : r.chains ≠ [] → r.chains.length = r.results.length) :
    ({ r with lowest := L, highest := H } : Record).keptChains =
      (r.keptChains.drop L).take (r.results.length - L - H) := by
  unfold keptChains
  split
  · simp only [kept, sorted, hl, hh, List.drop_zero, Nat.sub_zero,
      List.map_take, List.map_drop]
    rw [List.take_of_length_le (l := List.map Chain.single
      (r.results.mergeSort fun x1 x2 => decide (x1 ≤ x2)))
      (by simp [List.length_mergeSort])]
  · rename_i hc
    have hc : r.chains ≠ [] := fun h => hc (by simp [h])
    simp only [hl, hh, List.drop_zero, Nat.sub_zero]
    rw [List.take_of_length_le (l := r.chains.mergeSort Chain.le)
      (by simp [List.length_mergeSort, hlen hc])]

/--
A record of chains holds no chains, or one result for each chain.

# Parameters
- `die`: The die from which the dice of the chains were drawn.
- `cs`: The chains.
-/
theorem length_chains_ofChains (die : Die) (cs : List Chain) :
    (ofChains die cs).chains ≠ [] →
      (ofChains die cs).chains.length = (ofChains die cs).results.length := by
  intro h
  simp only [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains] at h ⊢
  split at h
  · exact absurd rfl h
  · rename_i hall
    simp only [hall, Bool.false_eq_true, ↓reduceIte, List.length_map]

/-- Dropping chains from a record of normal kept chains, whose kept results
are their totals, and that drops none, keeps a window of its kept chains, whose
totals are its kept results. -/
private theorem drops_of {r : Record} (L H : Nat)
    (hn : ∀ ch ∈ r.keptChains, NormalChain ch)
    (hkt : (r.kept : Multiset Int) = r.keptChains.map Chain.total)
    (hl : r.lowest = 0) (hh : r.highest = 0)
    (hlen : r.chains ≠ [] → r.chains.length = r.results.length) :
    (∀ ch ∈ ({ r with lowest := L, highest := H } : Record).keptChains,
      ch ∈ r.keptChains) ∧
      (({ r with lowest := L, highest := H } : Record).kept : Multiset Int) =
        ({ r with lowest := L, highest := H } : Record).keptChains.map
          Chain.total := by
  have hk := keptChains_drops L H hl hh hlen
  refine ⟨fun ch hch => ?_, ?_⟩
  · rw [hk] at hch
    exact List.mem_of_mem_drop (List.mem_of_mem_take hch)
  -- The kept results of the record are the totals of its kept chains, which
  -- ascend.
  have hsorted : r.kept = r.keptChains.map Chain.total := by
    refine (Multiset.coe_eq_coe.mp hkt).eq_of_pairwise
      (fun _ _ _ _ h₁ h₂ => Int.le_antisymm h₁ h₂) (kept_pairwise r) ?_
    exact List.pairwise_map.mpr ((keptChains_pairwise r hn).imp
      Chain.total_le_of_le)
  have hkept : ({ r with lowest := L, highest := H } : Record).kept =
      (r.kept.drop L).take (r.results.length - L - H) := by
    simp only [Record.kept, Record.sorted, hl, hh, List.drop_zero,
      Nat.sub_zero]
    rw [List.take_of_length_le (l := r.results.mergeSort fun x1 x2 =>
      decide (x1 ≤ x2)) (by simp [List.length_mergeSort])]
  rw [hk, hkept, hsorted, List.map_take, List.map_drop]

/--
Dropping chains from a record that fits its die, and drops none, leaves a
record that fits it: its kept chains are a window of the record's, and its
kept results are their totals.

# Parameters
- `L`, `H`: The numbers of lowest and highest chains dropped.

# Hypotheses
- `h`: The record fits the die.
- `hl`, `hh`: The record drops nothing.
- `hlen`: A record of chains holds one result for each chain.
-/
theorem Fits.drops {D : Die} {r : Record} (h : r.Fits D) (L H : Nat)
    (hl : r.lowest = 0) (hh : r.highest = 0)
    (hlen : r.chains ≠ [] → r.chains.length = r.results.length) :
    ({ r with lowest := L, highest := H } : Record).Fits D := by
  obtain ⟨hsub, hkept⟩ := drops_of L H h.normal h.kept hl hh hlen
  exact ⟨h.die, fun ch hch => h.normal ch (hsub ch hch),
    fun ch hch => h.live ch (hsub ch hch),
    fun ch hch => h.faces ch (hsub ch hch), hkept⟩

/--
Dropping chains from a saturated record that drops none leaves a saturated
record: its kept chains are a window of the record's, and its kept results
are their totals.

# Parameters
- `L`, `H`: The numbers of lowest and highest chains dropped.

# Hypotheses
- `h`: The record is saturated.
- `hl`, `hh`: The record drops nothing.
- `hlen`: A record of chains holds one result for each chain.
-/
theorem Saturated.drops {D : Die} {r : Record} (h : r.Saturated D)
    (L H : Nat) (hl : r.lowest = 0) (hh : r.highest = 0)
    (hlen : r.chains ≠ [] → r.chains.length = r.results.length) :
    ({ r with lowest := L, highest := H } : Record).Saturated D := by
  obtain ⟨hsub, hkept⟩ := drops_of L H
    (fun ch hch => normal_of_saturates (h.saturated ch hch)) h.kept hl hh hlen
  exact ⟨h.die, fun ch hch => h.saturated ch (hsub ch hch), hkept⟩

end Record

/-! ## Live setups -/

/--
Weighted values of positive total weight take only values that they list.

# Parameters
- `x`: A value that they take.

# Hypotheses
- `hl`: The total weight is positive.
- `h`: They take `x`.
-/
theorem mem_of_mem_support_weighted {α : Type} {d : α} {l : List (α × Nat)}
    {x : α} (hl : (l.map (·.2)).sum ≠ 0)
    (h : x ∈ (weighted d l).support) :
    ∃ e ∈ l, e.1 = x := by
  obtain ⟨e₀, he₀⟩ : ∃ e, e ∈ l := by
    cases l with
    | nil => exact absurd rfl hl
    | cons e _ => exact ⟨e, List.mem_cons_self⟩
  rw [weighted_congr_default d e₀.1 hl] at h
  rcases mem_support_weighted h with h | rfl
  · exact h
  · exact ⟨e₀, he₀, rfl⟩

/--
A record of chains without a die, given a die and no drops, is the record of
the chains with the die.

# Parameters
- `D`: The die.
- `cs`: The chains.
-/
theorem ofChains_with (D : Die) (cs : List Chain) :
    ({ Record.ofChains .none cs with lowest := 0, highest := 0, die := D } :
      Record) = Record.ofChains D cs := by
  simp [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains]

/-- The membership of kept chains, as a multiset. -/
private theorem mem_of_coe_eq {l l' : List Chain}
    (h : (l : Multiset Chain) = l')
    {x : Chain} (hx : x ∈ l) : x ∈ l' :=
  (Multiset.coe_eq_coe.mp h).mem_iff.mp hx

/--
A key that a list of keys, each with its value, gives a value is among them,
with that value.

# Parameters
- `l`: The keys, each with its value.
- `x`: The key.
- `v`: The value.

# Hypotheses
- The keys give `x` the value `v`.
-/
theorem mem_of_lookup {β : Type} :
    ∀ {l : List (Chain × β)} {x : Chain} {v : β}, l.lookup x = some v →
      (x, v) ∈ l
  | [], _, _, h => by cases h
  | (y, w) :: l, x, v, h => by
    by_cases hxy : x = y
    · subst hxy
      simp only [List.lookup, beq_self_eq_true, Option.some.injEq] at h
      subst h
      exact List.mem_cons_self
    · have hb : (x == y) = false := by simpa using hxy
      simp only [List.lookup, hb] at h
      exact List.mem_cons_of_mem _ (mem_of_lookup h)

/--
Every chain that a chain of stacks may have become is the chain itself, or a
chain of the law of what some chain has become.

# Parameters
- `laws`: The law of what each distinct chain has become.
- `ch`: The chain.
- `x`: A chain that it may have become.

# Hypotheses
- `hx`: The chain may have become `x`.
-/
theorem mem_support_became {laws : List (Chain × List (Chain × Nat))}
    {ch x : Chain} (hx : x ∈ (became laws ch).support) :
    x = ch ∨ ∃ e ∈ laws, ∃ q ∈ e.2, q.1 = x := by
  unfold became at hx
  split at hx
  · rename_i law heq
    rcases mem_support_weighted hx with ⟨q, hq, rfl⟩ | rfl
    · exact .inr ⟨_, mem_of_lookup heq, q, hq, rfl⟩
    · exact .inl rfl
  · exact .inl ((PMF.mem_support_pure_iff _ _).mp hx)

/--
A multiset of stacks whose laws each hold a chain, each of positive weight,
gains some branches from them.

# Parameters
- `ms`: The multiset.

# Hypotheses
- `hlaws`: The law of what each chain has become holds a chain, and each of
  its chains weighs something.
-/
theorem stackWeight_ne_zero {laws : List (Chain × List (Chain × Nat))}
    (hlaws : ∀ e ∈ laws, e.2 ≠ [] ∧ ∀ p ∈ e.2, 0 < p.2)
    (ms : List (Chain × Nat)) : stackWeight laws ms ≠ 0 := by
  rw [stackWeight, Ne, List.prod_eq_zero_iff]
  simp only [List.mem_map, not_exists, not_and]
  intro p _ h0
  have hb := pow_eq_zero_iff'.mp h0 |>.1
  cases hl : laws.lookup p.1 with
  | none => simp [hl] at hb
  | some law =>
    simp only [hl, Option.map_some, Option.getD_some] at hb
    obtain ⟨hne, hpos⟩ := hlaws _ (mem_of_lookup hl)
    obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp hb q.2 (List.mem_map_of_mem hq)
    exact (hpos q hq).ne' this

/--
A record of normal chains that may still grow, of faces of a die, fits the
die.

# Parameters
- `D`: The die.
- `cs`: The chains.

# Hypotheses
- `h`: Every chain is normal, may still grow, and holds only faces of the die.
-/
theorem fits_ofChains (D : Die) {cs : List Chain}
    (h : ∀ x ∈ cs, NormalChain x ∧ x.saturates = false ∧
      ∀ y ∈ x.dice, y ∈ D.faces) :
    (Record.ofChains D cs).Fits D := by
  have hk := keptChains_ofChains D fun x hx => (h x hx).1
  have hmem : ∀ x ∈ (Record.ofChains D cs).keptChains, x ∈ cs := fun x hx =>
    (Multiset.coe_eq_coe.mp hk).mem_iff.mp hx
  exact ⟨(Record.ofChains_fields _ _).2.2.2, fun x hx => (h x (hmem x hx)).1,
    fun x hx => (h x (hmem x hx)).2.1, fun x hx => (h x (hmem x hx)).2.2,
    by rw [kept_ofChains, ← Multiset.map_coe, ← Multiset.map_coe, hk]⟩

namespace Setup

/-- Every record of a setup of a roll without chains is the roll's, with the
setup's drops and die, and holds no chains. -/
private theorem mem_support_law_plain {s : Setup} {ρ : Roll}
    (hρ : s.roll = some ρ) (hp : ρ.Plain) {r : Record}
    (hr : r ∈ s.law.support) : r.chains = [] := by
  have h : s.Plain := fun ρ' h => by rw [hρ] at h; cases h; exact hp
  exact Setup.chains_of_mem_support_law h hr

/-- Every record of a setup of i.i.d. chains is the record of some draw of its
chains, with its die and its drops. -/
private theorem mem_support_law_chains {s : Setup} {count : Int}
    {cl : List (Chain × Nat)} (hρ : s.roll = some (.chains count cl))
    {r : Record} (hr : r ∈ s.law.support) :
    ∃ cs ∈ (chainDraws count.toNat (weighted (.single 0) cl)).support,
      r = { Record.ofChains s.replacement cs with
        lowest := s.lowest, highest := s.highest } := by
  simp only [law, hρ, Option.map_some, Option.getD_some, Roll.law,
    PMF.map_comp, PMF.support_map, Set.mem_image] at hr
  obtain ⟨cs, hcs, rfl⟩ := hr
  refine ⟨cs, hcs, ?_⟩
  simp [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains]

/-- Every record of a setup of stacks is the record of some draw of chains,
each a chain that some chain of the pool may have become, with its die and its
drops. -/
private theorem mem_support_law_stacked {s : Setup}
    {ms : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))}
    (hρ : s.roll = some (.stacked ms laws)) {r : Record}
    (hr : r ∈ s.law.support) :
    ∃ cs : List Chain, (∀ x ∈ cs, ∃ e ∈ ms, ∃ p ∈ e.1,
        x ∈ (became laws p.1).support) ∧
      r = { Record.ofChains s.replacement cs with
        lowest := s.lowest, highest := s.highest } := by
  simp only [law, hρ, Option.map_some, Option.getD_some, Roll.law] at hr
  obtain ⟨r', hr', rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hr
  obtain ⟨m, hm, hr'⟩ := (PMF.mem_support_bind_iff _ _ _).mp hr'
  obtain ⟨cs, hcs, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hr'
  refine ⟨cs, fun x hx => ?_, by
    simp [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains]⟩
  obtain ⟨p, hp, hx⟩ := (mem_support_stackDraws hcs).2 x hx
  -- The multiset is one of the pool, or the first, or empty.
  rcases mem_support_weighted hm with ⟨e, he, rfl⟩ | rfl
  · obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
    exact ⟨e', he', p, hp, hx⟩
  · cases ms with
    | nil => cases hp
    | cons e ms => exact ⟨e, List.mem_cons_self, p, hp, hx⟩

/--
A live setup's roll is not decided by a saturated die.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hg`: The setup is live.
- `hρ`: The setup's roll is `ρ`.
-/
theorem isSaturated_of_live {s : Setup} {ρ : Roll} (hg : s.Live)
    (hρ : s.roll = some ρ) : ρ.isSaturated = false := by
  have hgρ := hg.2 ρ hρ
  cases ρ with
  | stacked ms laws =>
    obtain ⟨_, hpool, hlaws⟩ := hgρ
    simp only [Roll.isSaturated, List.any_eq_false]
    intro e he
    simp only [List.any_eq_true, not_exists, not_and]
    intro p hp
    split
    · rename_i law hl
      simp only [List.any_eq_true, not_exists, not_and, Bool.not_eq_true]
      exact fun q hq => (hlaws _ (mem_of_lookup hl) q hq).2.1
    · simpa using (hpool e he p hp).2.1
  | _ => rfl

/--
Every record that a well-formed, live setup describes fits its die: its
kept chains are normal, may still grow, and hold only faces of the die, and
its kept results are their totals.

# Parameters
- `r`: The record.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hr`: The setup describes `r`.
-/
theorem fits_of_live {s : Setup} (hs : s.WellFormed) (hg : s.Live)
    {r : Record} (hr : r ∈ s.law.support) : r.Fits s.replacement := by
  have hd := (mem_support_law_die hr).1
  -- A record without chains keeps the chains of one die of its kept results.
  have plain : r.chains = [] →
      (∀ x ∈ r.results, x ∈ s.replacement.faces) → r.Fits s.replacement :=
      fun hc hf => by
    have hk : r.keptChains = r.kept.map .single := by
      simp [Record.keptChains, hc]
    have hkf : ∀ x ∈ r.kept, x ∈ s.replacement.faces := fun x hx =>
      hf x ((List.mergeSort_perm _ _).mem_iff.mp
        (List.mem_of_mem_drop (List.mem_of_mem_take hx)))
    refine ⟨hd, ?_, ?_, ?_, ?_⟩
    · intro ch hch
      rw [hk] at hch
      obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
      refine ⟨.nil, fun y hy => ?_⟩
      rw [List.mem_singleton.mp hy]
      exact hg.1 x (hkf x hx)
    · intro ch hch
      rw [hk] at hch
      obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
      rfl
    · intro ch hch y hy
      rw [hk] at hch
      obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
      simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
        List.mem_singleton] at hy
      rw [hy]
      exact hkf x hx
    · rw [hk, List.map_map]
      refine congrArg _ ((List.map_congr_left fun x hx => ?_).trans
        (List.map_id _)).symm
      exact total_single (hg.1 x (hkf x hx))
  cases hρ : s.roll with
  | none =>
    refine plain ?_ fun x hx => ?_
    · simp only [law, hρ, Option.map_none, Option.getD_none, PMF.pure_map,
        PMF.mem_support_pure_iff] at hr
      rw [hr]
    · simp only [law, hρ, Option.map_none, Option.getD_none, PMF.pure_map,
        PMF.mem_support_pure_iff] at hr
      rw [hr] at hx
      cases hx
  | some ρ =>
    have hgρ := hg.2 ρ hρ
    cases ρ with
    | chains count law =>
      obtain ⟨_, hlaw⟩ := hgρ
      obtain ⟨hne, hpos⟩ := hs _ hρ
      have hW : (law.map (·.2)).sum ≠ 0 := by
        obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
        have := hpos e he
        intro h0
        have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
        omega
      obtain ⟨cs, hcs, rfl⟩ := mem_support_law_chains hρ hr
      -- Every chain drawn is a chain of the law.
      have hmem : ∀ x ∈ cs, ∃ e ∈ law, e.1 = x := fun x hx =>
        mem_of_mem_support_weighted hW ((mem_support_chainDraws hcs).2 x hx)
      have hn : ∀ x ∈ cs, NormalChain x := fun x hx => by
        obtain ⟨e, he, rfl⟩ := hmem x hx
        exact (hlaw e he).1
      have hk := keptChains_ofChains s.replacement hn
      -- The record of the chains, before its drops, fits the die.
      have h0 : (Record.ofChains s.replacement cs).Fits s.replacement := by
        refine ⟨(Record.ofChains_fields _ _).2.2.2, fun ch hch => ?_,
          fun ch hch => ?_, fun ch hch => ?_, ?_⟩
        · exact hn ch (mem_of_coe_eq hk hch)
        · obtain ⟨e, he, rfl⟩ := hmem ch (mem_of_coe_eq hk hch)
          exact (hlaw e he).2.1
        · obtain ⟨e, he, rfl⟩ := hmem ch (mem_of_coe_eq hk hch)
          exact (hlaw e he).2.2
        · rw [kept_ofChains, ← Multiset.map_coe, ← Multiset.map_coe, hk]
      exact h0.drops _ _ (Record.ofChains_fields _ _).2.1
        (Record.ofChains_fields _ _).2.2.1 (Record.length_chains_ofChains _ _)
    | stacked ms laws =>
      obtain ⟨_, hpool, hlaws⟩ := hgρ
      obtain ⟨cs, hcs, rfl⟩ := mem_support_law_stacked hρ hr
      -- Every chain drawn is a chain of the pool, or of a law.
      have h0 := fits_ofChains s.replacement (cs := cs) fun x hx => by
        obtain ⟨e, he, p, hp, hx⟩ := hcs x hx
        rcases mem_support_became hx with rfl | ⟨e', he', q, hq, rfl⟩
        · exact hpool e he p hp
        · exact hlaws e' he' q hq
      exact h0.drops _ _ (Record.ofChains_fields _ _).2.1
        (Record.ofChains_fields _ _).2.2.1 (Record.length_chains_ofChains _ _)
    | range | standard | custom | fixed | pool =>
      exact plain (mem_support_law_plain hρ trivial hr) (hgρ r hr)

/--
Every record that a saturated setup describes is saturated: each of its kept
chains is a chain of the pool, or one that such a chain may have become, and
its kept results are their totals.

# Parameters
- `r`: The record.

# Hypotheses
- `hsat`: The setup is saturated.
- `hr`: The setup describes `r`.
-/
theorem saturated_of_mem {s : Setup} (hsat : s.Saturated) {r : Record}
    (hr : r ∈ s.law.support) : r.Saturated s.replacement := by
  obtain ⟨ms, laws, hρ, _, hpool, hlaws⟩ := hsat
  obtain ⟨cs, hcs, rfl⟩ := mem_support_law_stacked hρ hr
  -- Every chain drawn is a chain of the pool, or of a law.
  have hsc : ∀ x ∈ cs, x.saturates = true := fun x hx => by
    obtain ⟨e, he, p, hp, hx⟩ := hcs x hx
    rcases mem_support_became hx with rfl | ⟨e', he', q, hq, rfl⟩
    · exact hpool e he p hp
    · exact hlaws e' he' q hq
  have hk := keptChains_ofChains s.replacement fun x hx =>
    normal_of_saturates (hsc x hx)
  have h0 : (Record.ofChains s.replacement cs).Saturated s.replacement :=
    ⟨(Record.ofChains_fields _ _).2.2.2,
      fun ch hch => hsc ch (mem_of_coe_eq hk hch),
      by rw [kept_ofChains, ← Multiset.map_coe, ← Multiset.map_coe, hk]⟩
  exact h0.drops _ _ (Record.ofChains_fields _ _).2.1
    (Record.ofChains_fields _ _).2.2.1 (Record.length_chains_ofChains _ _)

/--
Every record that a well-formed, live setup describes keeps as many chains
as the setup keeps results.

# Parameters
- `r`: The record.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hr`: The setup describes `r`.
-/
theorem length_keptChains {s : Setup} (hs : s.WellFormed) (hg : s.Live)
    {r : Record} (hr : r ∈ s.law.support) :
    r.keptChains.length = s.results - s.lowest - s.highest := by
  have hf := fits_of_live hs hg hr
  have hk := congrArg Multiset.card hf.kept
  simp only [Multiset.coe_card, List.length_map] at hk
  rw [← hk, Record.length_kept]
  obtain ⟨hlen, hl, hh⟩ := mem_support_law (fun ρ h => (hs ρ h).even) hr
  rw [hlen, hl, hh]

end Setup

/-! ## The laws of an explode of a setup -/

namespace Setup

/--
Whether exploding a setup, as `Mixture::explode` in `propagation/mixture.rs`
does, describes the specification's explode of the records that it
describes, as every later instruction reads them: if the chains stay whole,
up to their chain view, the kept chains and the die, which a later reroll or
explode reads; and if they reduce to their totals, up to their view, the kept
results and the die, which every later drop, keep, and sum reads.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole.

# Returns
Whether the exploded setup's law is the explode's, viewed.
-/
def Bursts (s : Setup) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Prop :=
  if whole then
    (s.burst c limit true).law.map Record.chainView =
      (s.law.bind fun r => Spec.explode r c limit).map Record.chainView
  else
    (s.burst c limit false).law.map Record.view =
      (s.law.bind fun r => Spec.explode r c limit).map Record.view

/--
A sorted list of values, read from their multiset.

# Parameters
- `M`: The values.

# Returns
The values, in ascending order.
-/
noncomputable def sortBag (M : Multiset Int) : List Int :=
  M.toList.mergeSort

/--
Sorting depends only on the multiset.

# Parameters
- `xs`: The values.
-/
theorem mergeSort_eq_sortBag (xs : List Int) :
    xs.mergeSort = sortBag xs := by
  have hs : ∀ l : List Int, (l.mergeSort).Pairwise (· ≤ ·) := fun l => by
    have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
      (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) l
    simpa using this
  refine List.Perm.eq_of_pairwise
    (fun _ _ _ _ h₁ h₂ => Int.le_antisymm h₁ h₂) (hs xs) (hs _) ?_
  exact (List.mergeSort_perm _ _).trans ((Multiset.coe_eq_coe.mp
    (Multiset.coe_toList (xs : Multiset Int)).symm).trans
      (List.mergeSort_perm _ _).symm)

/--
A record of chains is viewed by the multiset of their totals.

# Parameters
- `D`: The die.
- `cs`: The chains.
-/
theorem view_ofChains (D : Die) (cs : List Chain) :
    (Record.ofChains D cs).view = (sortBag (cs.map Chain.total), D) := by
  have hk := kept_ofChains D cs
  have hd := (Record.ofChains_fields D cs).2.2.2
  simp only [Record.view, hd, Prod.mk.injEq, and_true]
  rw [← mergeSort_eq_sortBag]
  refine (Multiset.coe_eq_coe.mp (hk.trans (Multiset.coe_eq_coe.mpr
    (List.mergeSort_perm _ _).symm))).eq_of_pairwise
    (fun _ _ _ _ h₁ h₂ => Int.le_antisymm h₁ h₂)
    (Record.kept_pairwise _) ?_
  have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
    (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
    (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
    (cs.map Chain.total)
  simpa using this

/--
The view of a record that drops nothing, after the lowest `L` and the highest
`H` of its results are dropped.

# Parameters
- `L`, `H`: The numbers of lowest and highest results dropped.
- `v`: The view of the record: its sorted results, and its die.

# Returns
The view of the record after the drops.
-/
def dropView (L H : Nat) (v : List Int × Die) : List Int × Die :=
  ((v.1.drop L).take (v.1.length - L - H), v.2)

/-- Dropping results from a record that drops none drops them from its view. -/
private theorem view_drops {r : Record} (L H : Nat) (hl : r.lowest = 0)
    (hh : r.highest = 0) :
    ({ r with lowest := L, highest := H } : Record).view =
      dropView L H r.view := by
  rw [Record.view_of_undropped hl hh]
  simp only [Record.view, Record.kept, Record.sorted, dropView,
    List.length_mergeSort]

/--
The records that a setup describes, as every later drop, keep, and sum views
them, depend only on the results of the records that its roll fills, as a
multiset: each view is the multiset sorted, after the setup's drops, with the
setup's die.

# Parameters
- `s`: The setup.
- `ρ`: Its roll.

# Hypotheses
- `hρ`: The setup has the roll.
-/
theorem map_view_law {s : Setup} {ρ : Roll} (hρ : s.roll = some ρ) :
    s.law.map Record.view =
      (ρ.law.map fun r => (r.results : Multiset Int)).map fun M =>
        dropView s.lowest s.highest (sortBag M, s.replacement) := by
  simp only [law, hρ, Option.map_some, Option.getD_some, PMF.map_comp]
  refine map_congr_support fun r _ => ?_
  simp only [Function.comp_apply, Record.view, Record.kept, Record.sorted,
    dropView, mergeSort_eq_sortBag]
  rw [show r.results.length = (sortBag r.results).length by
    rw [← mergeSort_eq_sortBag, List.length_mergeSort]]

/--
The results of independent dice, as a multiset, are independent draws.

# Parameters
- `n`: The number of dice.
- `die`: The law of each.
-/
theorem map_results_rollDice (n : Int) (die : PMF Int) :
    (rollDice n die).map (fun r => (r.results : Multiset Int)) =
      iidBag n.toNat die := by
  have h := bind_rollDice_iidBag n die PMF.pure
  rw [PMF.bind_pure] at h
  rw [← h, ← PMF.bind_pure_comp]
  refine congrArg _ (funext fun r => ?_)
  exact (bagDraws_pure fun _ _ => rfl).symm

/--
Reducing a well-formed, live setup's chains to their totals leaves the law
of the records that it describes, as every later drop, keep, and sum views
them. Mirrors `Setup::collapse` in `propagation/mixture.rs`: the totals of
independent and alike chains are the results of custom dice of their totals.
`XdySpec/Stacks.lean` proves it of stacks (`Roll.map_results_collapse`).

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hns`: The setup's roll is not stacks.
-/
theorem law_collapse {s : Setup} (hs : s.WellFormed) (hg : s.Live)
    (hns : ∀ ms ls, s.roll ≠ some (.stacked ms ls)) :
    s.collapse.law.map Record.view = s.law.map Record.view := by
  have hsame : (∀ ρ ∈ s.roll, ρ.collapse = ρ) → s.collapse = s :=
      fun h => by
    rcases s with ⟨_ | ρ, _, _, _⟩
    · rfl
    · simp only [collapse, Option.map_some, h ρ rfl]
  cases hρ : s.roll with
  | none => rw [hsame fun ρ h => by rw [hρ] at h; cases h]
  | some ρ =>
    cases ρ with
    | chains count cl =>
      have hgρ := hg.2 _ hρ
      obtain ⟨hdie, _⟩ := hgρ
      obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp hdie
      obtain ⟨hne, hpos⟩ := hs _ hρ
      have hW : (cl.map (·.2)).sum ≠ 0 := by
        obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
        have := hpos e he
        intro h0
        have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
        omega
      set D := s.replacement
      -- The setup describes records of independent chains, after its drops.
      have hR : s.law.map Record.view =
          (iidBag count.toNat ((weighted (.single 0) cl).map Chain.total)).map
            fun M => dropView s.lowest s.highest (sortBag M, D) := by
        have hlaw : s.law = (chainDraws count.toNat
            (weighted (.single 0) cl)).map fun cs =>
              ({ Record.ofChains D cs with
                lowest := s.lowest, highest := s.highest } : Record) := by
          simp only [law, hρ, Option.map_some, Option.getD_some, Roll.law,
            PMF.map_comp]
          refine congrArg (PMF.map · _) (funext fun cs => ?_)
          simp [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains, D]
        rw [hlaw, PMF.map_comp, ← iidBag_map, ← map_coe_chainDraws,
          PMF.map_comp, PMF.map_comp]
        refine congrArg (PMF.map · _) (funext fun cs => ?_)
        simp only [Function.comp_apply]
        rw [view_drops _ _ (Record.ofChains_fields _ _).2.1
          (Record.ofChains_fields _ _).2.2.1, view_ofChains, Multiset.map_coe]
      -- The collapsed setup describes records of independent totals, after
      -- its drops.
      have hL : s.collapse.law.map Record.view =
          (iidBag count.toNat ((weighted (.single 0) cl).map Chain.total)).map
            fun M => dropView s.lowest s.highest (sortBag M, D) := by
        simp only [law, collapse, hρ, Option.map_some, Option.getD_some,
          Roll.collapse, Roll.law, PMF.map_comp]
        have hW' : weightedDie (tally (Roll.expand
            (cl.map fun e => (e.1.total, e.2)))) =
            (weighted (.single 0) cl).map Chain.total := by
          rw [weightedDie_tally, ← weightedDie_eq_customDie_expand,
            weightedDie_eq_weighted,
            weighted_congr_default 0 (Chain.single 0).total (by
              simpa [Function.comp_def] using hW),
            weighted_map _ _ _ hW]
        rw [hW', ← map_results_rollDice, PMF.map_comp]
        refine map_congr_support fun r hr => ?_
        obtain ⟨hl', hh', _, _⟩ := mem_support_rollDice hr
        simp only [Function.comp_apply, Record.view, Record.kept,
          Record.sorted, dropView, Prod.mk.injEq]
        rw [← mergeSort_eq_sortBag, List.length_mergeSort]
        exact ⟨rfl, by simp [replacement, hd, D]⟩
      rw [hL, hR]
    | stacked ms ls => exact absurd hρ (hns ms ls)
    | range | standard | custom | fixed | pool =>
      rw [hsame fun ρ' h => by rw [hρ] at h; cases h; rfl]

end Setup

namespace Setup

/-- Exploding a setup that the explode leaves alone describes the explode of
its records, if the explode grows no kept chain of any, and saturates none,
so long as reducing its chains to their totals leaves their law. -/
private theorem bursts_of_stays {s : Setup} {c : Criterion}
    {limit : Option Nat} (whole : Bool) (hs : s.WellFormed) (hg : s.Live)
    (hcol : s.collapse.law.map Record.view = s.law.map Record.view)
    (hst : ∀ w, s.explode c limit w = s)
    (hgrow : ∀ r ∈ s.law.support, ∀ ch ∈ r.keptChains, limit ≠ some 0 →
      ch.grows c = false)
    (hsat : limit = none → ¬s.replacement.facesWithin c = true ∨
      s.results - s.lowest - s.highest = 0) :
    s.Bursts c limit whole := by
  have hspec : ∀ r ∈ s.law.support, Spec.explode r c limit =
      PMF.pure (Record.ofChains s.replacement r.keptChains) := fun r hr =>
    explode_stays (fits_of_live hs hg hr) (fun hl => by
      rcases hsat hl with h | h
      · exact .inl h
      · exact .inr (List.eq_nil_of_length_eq_zero
          ((length_keptChains hs hg hr).trans h)))
      (hgrow r hr)
  rw [Bursts]
  cases whole with
  | true =>
    simp only [↓reduceIte, burst, hst]
    rw [bind_congr_support hspec, PMF.map_bind]
    simp only [PMF.pure_map]
    refine (map_congr_support fun r hr =>
      ((fits_of_live hs hg hr).views).1.symm).trans ?_
    exact (PMF.bind_pure_comp _ _).symm
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte, burst, hst, hcol]
    rw [bind_congr_support hspec, PMF.map_bind]
    simp only [PMF.pure_map]
    refine (map_congr_support fun r hr =>
      ((fits_of_live hs hg hr).views).2.symm).trans ?_
    exact (PMF.bind_pure_comp _ _).symm

/-- Independent draws of a certain chain are copies of it. -/
private theorem chainDraws_pure (x : Chain) :
    ∀ n, chainDraws n (PMF.pure x) = PMF.pure (List.replicate n x)
  | 0 => rfl
  | n + 1 => by
    have := chainDraws_pure x n
    simp only [chainDraws, Nat.repeat] at this ⊢
    rw [this, PMF.pure_bind, PMF.pure_map]
    rfl

/-- A well-formed roll has branches. -/
private theorem total_ne_zero {ρ : Roll} (hρ : ρ.WellFormed) :
    ρ.total ≠ 0 := by
  have hsum : ∀ {α : Type} (l : List (α × Nat)), l ≠ [] →
      (∀ p ∈ l, 0 < p.2) → (l.map (·.2)).sum ≠ 0 := fun l hne hpos h0 => by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    have := hpos e he
    omega
  cases ρ with
  | range start stop =>
    simp only [Roll.total]
    split <;> omega
  | standard count faces =>
    simp only [Roll.total]
    split
    · exact pow_ne_zero _ (by omega)
    · omega
  | custom count faces =>
    simp only [Roll.total]
    split
    · rename_i h
      exact pow_ne_zero _ (hsum faces h.2 hρ.2)
    · omega
  | fixed => simp [Roll.total]
  | pool outcomes =>
    obtain ⟨hne, hmem, _⟩ := hρ
    simp only [Roll.total]
    intro h0
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    have := (hmem e he).2
    omega
  | chains count law =>
    simp only [Roll.total]
    split
    · rename_i h
      exact pow_ne_zero _ (hsum law h.2 hρ.2)
    · omega
  | stacked ms ls =>
    obtain ⟨hne, hmem, _, hlaws⟩ := hρ
    simp only [Roll.total]
    intro h0
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 _ (List.mem_map_of_mem he)
    exact mul_ne_zero (hmem e he).2.ne' (stackWeight_ne_zero hlaws e.1) this

/-- Exploding a setup by an unlimited explode that matches every face of its
die saturates every kept chain of every record that it describes. -/
private theorem bursts_of_saturate {s : Setup} {c : Criterion} {ρ : Roll}
    {v : Int} (whole : Bool) (hs : s.WellFormed) (hg : s.Live)
    (hρ : s.roll = some ρ) (hk : s.results - s.lowest - s.highest ≠ 0)
    (hb : ∀ w, Burst.resolve s.replacement c none w = .saturate v) :
    s.Bursts c none whole := by
  have hsat := isSaturated_of_live hg hρ
  -- The burst saturates at the die's saturated value, and matches every face.
  have hres := hb true
  have hm : s.replacement.faces.countP c.matches ≠ 0 := by
    intro h
    simp only [Burst.resolve, h, ↓reduceIte, reduceCtorEq] at hres
  have hall : s.replacement.faces.countP c.matches =
      s.replacement.faces.length ∧ v = s.replacement.saturatedValue := by
    simp only [Burst.resolve, hm, ↓reduceIte] at hres
    split at hres
    · rename_i h
      exact ⟨h, (Burst.saturate.inj hres).symm⟩
    · cases hres
  have hw : s.replacement.facesWithin c = true := by
    rw [Die.facesWithin, List.all_eq_true]
    exact List.countP_eq_length.mp hall.1
  have hT := total_ne_zero (hs ρ hρ)
  have hexp : ∀ w, s.explode c none w =
      { roll := some (ρ.saturate v (s.results - s.lowest - s.highest) w),
        die := some s.replacement } := fun w => by
    simp only [explode, hρ, hk, hsat, Bool.false_eq_true, or_self, ↓reduceIte,
      hb w]
  -- The specification saturates every kept chain.
  have hspec : ∀ r ∈ s.law.support, Spec.explode r c none =
      PMF.pure (Record.ofChains s.replacement
        (r.keptChains.map (·.saturate v))) :=
    fun r hr => by
      have hd := (fits_of_live hs hg hr).die
      simp only [Spec.explode, hd, hw, ↓reduceDIte, hall.2]
  rw [Bursts]
  cases whole with
  | true =>
    simp only [↓reduceIte, burst, hexp]
    rw [bind_congr_support hspec, PMF.map_bind]
    simp only [PMF.pure_map]
    rw [bind_congr_support (p := s.law) (g := fun _ => PMF.pure
        (Multiset.replicate (s.results - s.lowest - s.highest) (.saturated v),
          s.replacement))
      fun r hr => by
        rw [(fits_of_live hs hg hr).saturates v |>.1,
          length_keptChains hs hg hr], PMF.bind_const]
    simp only [law, Option.map_some, Option.getD_some, Roll.saturate,
      ↓reduceIte, Roll.law, List.map_cons, List.map_nil, stackWeight_nil,
      mul_one, weighted_single _ _ hT, PMF.pure_bind, stackDraws,
      became, List.lookup_nil, chainDraws_pure, PMF.pure_map]
    simp only [List.append_nil, replacement, Option.getD_some]
    rw [ofChains_with]
    have hk := keptChains_ofChains s.replacement
      (cs := List.replicate (s.results - s.lowest - s.highest)
        (Chain.saturated v))
      (fun x hx => by rw [List.eq_of_mem_replicate hx]; trivial)
    simp only [replacement] at hk
    simp only [Record.chainView, (Record.ofChains_fields _ _).2.2.2, hk,
      Multiset.coe_replicate]
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte, burst, hexp]
    rw [bind_congr_support hspec, PMF.map_bind]
    simp only [PMF.pure_map]
    rw [bind_congr_support (p := s.law) (g := fun _ => PMF.pure
        (List.replicate (s.results - s.lowest - s.highest) v, s.replacement))
      fun r hr => by
        rw [(fits_of_live hs hg hr).saturates v |>.2,
          length_keptChains hs hg hr], PMF.bind_const]
    simp only [collapse, Option.map_some, Roll.saturate, Bool.false_eq_true,
      ↓reduceIte, Roll.collapse, law, Option.getD_some, Roll.law,
      weighted_single _ _ hT, PMF.pure_map, replacement]
    rw [Record.view_of_undropped rfl rfl]
    simp only [Record.sorted, Roll.expand, List.flatMap_cons, List.flatMap_nil,
      List.append_nil]
    rw [List.mergeSort_of_pairwise
      (List.pairwise_replicate.mpr (.inr (by simp)))]

end Setup

/-! ## Extending independent chains -/

/--
Every possible extension of a chain draws only faces of the die.

# Parameters
- `c`: The criterion.
- `F`: The faces of the die.
- `l`: The limit.
- `t`: A possible extension.

# Hypotheses
- `hF`: The die has faces.
- `ht`: The extension is possible.
-/
theorem mem_support_extension {c : Criterion} {F : List Int} {l : Nat}
    {t : List Int × Int} (hF : F ≠ [])
    (ht : t ∈ (extension c (customDie F) l).support) :
    (∀ x ∈ t.1, x ∈ F) ∧ t.2 ∈ F := by
  have hd : ∀ x ∈ (customDie F).support, x ∈ F := fun x hx => by
    rcases mem_of_mem_support_customDie hx with h | ⟨h, _⟩
    · exact h
    · exact absurd h hF
  rw [extension_eq_growthLaw] at ht
  have : ∀ (r : Nat) (s : List Int), (∀ x ∈ s, x ∈ F) →
      ∀ t ∈ (growthLaw (customDie F) c tailExtend tailClose r s).support,
        (∀ x ∈ t.1, x ∈ F) ∧ t.2 ∈ F := by
    intro r
    induction r with
    | zero =>
      intro s hs t ht
      simp only [growthLaw, PMF.support_map, Set.mem_image] at ht
      obtain ⟨x, hx, rfl⟩ := ht
      exact ⟨hs, hd x hx⟩
    | succ r ih =>
      intro s hs t ht
      simp only [growthLaw, PMF.mem_support_bind_iff] at ht
      obtain ⟨x, hx, ht⟩ := ht
      split at ht
      · refine ih _ (fun y hy => ?_) t ht
        rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hy) with
          rfl | hy
        · exact hd _ hx
        · exact hs y hy
      · rw [PMF.mem_support_pure_iff] at ht
        subst ht
        exact ⟨hs, hd x hx⟩
  exact this _ [] (by simp) t ht

/--
A chain whose dice are faces of a die within the `i32` range, grown by
extensions that are faces of it too, is normal.

# Parameters
- `ch`: The chain.
- `t`: The extensions.

# Hypotheses
- `hF`: The faces lie within the `i32` range.
- `hch`: The chain's dice are faces.
- `ht`: The extensions are faces.
-/
theorem normalChain_growChain {F : List Int} {ch : Chain} {t : List Int × Int}
    (hF : ∀ x ∈ F, i32Min ≤ x ∧ x ≤ i32Max) (hch : ∀ x ∈ ch.dice, x ∈ F)
    (ht : (∀ x ∈ t.1, x ∈ F) ∧ t.2 ∈ F) : NormalChain (growChain ch t) := by
  refine ⟨?_, fun x hx => hF x ?_⟩
  · have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
      (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
      (ch.dice ++ t.1)
    simpa using this
  · rcases List.mem_cons.mp hx with rfl | hx
    · exact ht.2
    · rcases List.mem_append.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
        hx | hx
      · exact hch x hx
      · exact ht.1 x hx

/--
Every multiset of draws holds only what each value may become.

# Parameters
- `step`: The law of what each value becomes.
- `xs`: The values.
- `M`: A possible multiset.

# Hypotheses
- `hM`: The multiset is possible.
-/
theorem mem_support_bagDraws {α β : Type} {step : α → PMF β} {xs : List α}
    {M : Multiset β} (hM : M ∈ (bagDraws step xs).support) :
    ∀ y ∈ M, ∃ x ∈ xs, y ∈ (step x).support := by
  induction xs generalizing M with
  | nil =>
    simp only [bagDraws, PMF.mem_support_pure_iff] at hM
    subst hM
    simp
  | cons x xs ih =>
    simp only [bagDraws, lift, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff] at hM
    obtain ⟨y, hy, N, hN, rfl⟩ := hM
    intro z hz
    rcases Multiset.mem_cons.mp hz with rfl | hz
    · exact ⟨x, List.mem_cons_self, hy⟩
    · obtain ⟨w, hw, hz⟩ := ih hN z hz
      exact ⟨w, List.mem_cons_of_mem _ hw, hz⟩

/--
Independent draws of chains, each drawn anew, are independent draws of the
chains drawn anew.

# Parameters
- `n`: The number of chains.
- `P`: The law of each.
- `K`: The law of what each becomes.
-/
theorem chainDraws_bind_bagDraws {β : Type} (n : Nat) (P : PMF Chain)
    (K : Chain → PMF β) :
    (chainDraws n P).bind (bagDraws K) = iidBag n (P.bind K) := by
  induction n with
  | zero => simp [chainDraws, Nat.repeat, bagDraws, iidBag]
  | succ n ih =>
    simp only [chainDraws, Nat.repeat] at ih ⊢
    rw [PMF.bind_bind]
    calc _ = (Nat.repeat (fun p => p.bind fun cs => P.map (· :: cs)) n
            (PMF.pure [])).bind fun cs => (P.bind K).bind
              fun y => (bagDraws K cs).map (y ::ₘ ·) := by
          refine congrArg _ (funext fun cs => ?_)
          simp only [PMF.bind_map, Function.comp_def, bagDraws, lift,
            PMF.bind_bind]
      _ = (P.bind K).bind fun y => ((Nat.repeat (fun p => p.bind fun cs =>
            P.map (· :: cs)) n (PMF.pure [])).bind (bagDraws K)).map
              (y ::ₘ ·) := by
          rw [PMF.bind_comm]
          simp only [PMF.map_bind]
      _ = _ := by
          rw [ih]
          show _ = (iidBag n (P.bind K)).bind
            fun M => (P.bind K).map (· ::ₘ M)
          simp only [← PMF.bind_pure_comp]
          exact PMF.bind_comm _ _ _

namespace Setup

/--
A setup of i.i.d. chains, without drops, describes records of independent
draws of them.

# Parameters
- `D`: The die of the setup.
- `count`: The number of chains.
- `cl`: The law of each chain.
-/
theorem law_chains (D : Die) (count : Int) (cl : List (Chain × Nat)) :
    ({ roll := some (.chains count cl), die := some D } : Setup).law =
      (chainDraws count.toNat (weighted (.single 0) cl)).map
        (Record.ofChains D) := by
  simp only [law, Option.map_some, Option.getD_some, Roll.law, PMF.map_comp,
    replacement]
  exact congrArg (PMF.map · _) (funext fun cs => ofChains_with _ _)

/-- A setup of custom dice, without drops, is viewed by the multiset of its
results. -/
private theorem map_view_custom (D : Die) (count : Int)
    (faces : List (Int × Nat)) :
    ({ roll := some (.custom count faces), die := some D } : Setup).law.map
        Record.view =
      (iidBag count.toNat (weightedDie faces)).map fun M => (sortBag M, D) := by
  simp only [law, Option.map_some, Option.getD_some, Roll.law, PMF.map_comp,
    replacement]
  rw [← map_results_rollDice, PMF.map_comp]
  refine map_congr_support fun r hr => ?_
  obtain ⟨hl, hh, _, _⟩ := mem_support_rollDice hr
  simp only [Function.comp_apply]
  rw [Record.view_of_undropped rfl rfl]
  simp [Record.sorted, mergeSort_eq_sortBag]

/-- The kept chains of every record of a setup of alike dice, none dropped,
are independent draws of the chains of one die of the dice, each drawn anew. -/
private theorem bind_bagDraws_keptChains_alike {s : Setup} {ρ : Roll}
    {β : Type} (hfaced : s.Faced) (hρ : s.roll = some ρ)
    (halike : ρ.alike) (hl : s.lowest = 0) (hh : s.highest = 0)
    (hk : s.results - s.lowest - s.highest ≠ 0) (K : Chain → PMF β) :
    s.law.bind (fun r => bagDraws K r.keptChains) =
      iidBag ρ.chainLaw.1.toNat
        ((weighted (.single 0) ρ.chainLaw.2).bind K) := by
  have hcl : ρ.chainLaw =
      (ρ.dice.1, ρ.dice.2.map fun p => (Chain.single p.1, p.2)) := by
    cases ρ <;> first | rfl | simp [Roll.alike] at halike
  have hn := dice_pos hρ halike hk
  have hne := expand_dice_ne_nil halike (hfaced _ hρ) hn
  have hW : (ρ.dice.2.map (·.2)).sum ≠ 0 := by
    intro h0
    apply hne
    simp only [Roll.expand]
    refine List.eq_nil_of_length_eq_zero ?_
    simpa [List.length_flatMap, Function.comp_def] using h0
  rw [hcl]
  show _ = iidBag ρ.dice.1.toNat _
  rw [weighted_map 0 _ Chain.single hW, PMF.bind_map]
  set D := s.replacement
  have hlawρ : s.law = ρ.law.map fun r =>
      { r with lowest := s.lowest, highest := s.highest, die := D } := by
    simp [law, hρ, D]
  rw [hlawρ, PMF.bind_map]
  -- The kept chains are the chains of one die of the results.
  have hkr : ∀ r ∈ ρ.law.support, ((fun r => bagDraws K r.keptChains) ∘
      fun r : Record =>
        ({ r with lowest := s.lowest, highest := s.highest, die := D } :
          Record)) r = bagDraws (K ∘ Chain.single) r.results := fun r hr => by
    have hc := Roll.chains_of_mem_support_law (ρ := ρ)
      (by cases ρ <;> trivial) hr
    simp only [Function.comp_apply, Record.keptChains, hc, List.isEmpty_nil,
      ↓reduceIte, bagDraws_map]
    refine bagDraws_perm _ ?_
    simp only [Record.kept, Record.sorted, hl, hh, List.drop_zero,
      Nat.sub_zero]
    rw [show r.results.length = (r.results.mergeSort).length from
      (List.length_mergeSort _).symm, List.take_length]
    exact List.mergeSort_perm _ _
  rw [bind_congr_support hkr, ← bind_rollDice_iidBag]
  calc _ = (ρ.law.map (·.results)).bind (bagDraws (K ∘ Chain.single)) :=
        (PMF.bind_map _ _ _).symm
    _ = _ := by rw [Roll.map_results_law halike, PMF.bind_map]; rfl

/--
The kept chains of every record of a setup whose chains are independent and
alike, none dropped, are independent draws of the law of each chain, each
drawn anew.

# Type parameters
- `β`: The type of what each chain is drawn anew as.

# Parameters
- `K`: The law of what each chain is drawn anew as.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hfaced`: Every die of the setup has a face.
- `hρ`: The setup's roll is `ρ`.
- `hlike`: Its chains are independent and alike.
- `hl`, `hh`: It drops nothing.
- `hk`: It keeps some chain.
-/
theorem bind_bagDraws_keptChains {s : Setup} {ρ : Roll} {β : Type}
    (hs : s.WellFormed) (hg : s.Live) (hfaced : s.Faced)
    (hρ : s.roll = some ρ) (hlike : ρ.chainLike) (hl : s.lowest = 0)
    (hh : s.highest = 0) (hk : s.results - s.lowest - s.highest ≠ 0)
    (K : Chain → PMF β) :
    s.law.bind (fun r => bagDraws K r.keptChains) =
      iidBag ρ.chainLaw.1.toNat
        ((weighted (.single 0) ρ.chainLaw.2).bind K) := by
  cases ρ with
  | chains count cl =>
    obtain ⟨_, hlaw⟩ := hg.2 _ hρ
    obtain ⟨hne, hpos⟩ := hs _ hρ
    have hW : (cl.map (·.2)).sum ≠ 0 := by
      obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
      have := hpos e he
      intro h0
      have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
      omega
    have hlaw' : s.law = (chainDraws count.toNat
        (weighted (.single 0) cl)).map (Record.ofChains s.replacement) := by
      simp only [law, hρ, Option.map_some, Option.getD_some, Roll.law,
        PMF.map_comp, hl, hh]
      exact congrArg (PMF.map · _) (funext fun cs => ofChains_with _ _)
    rw [hlaw', PMF.bind_map]
    simp only [Roll.chainLaw]
    rw [← chainDraws_bind_bagDraws]
    refine bind_congr_support fun cs hcs => bagDraws_perm K ?_
    refine Multiset.coe_eq_coe.mp (keptChains_ofChains _ fun x hx => ?_)
    obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW
      ((mem_support_chainDraws hcs).2 x hx)
    exact (hlaw e he).1
  | stacked | fixed | pool => simp [Roll.chainLike] at hlike
  | range | standard | custom =>
    exact bind_bagDraws_keptChains_alike hfaced hρ rfl hl hh hk K

end Setup

namespace Setup

/-- The law of each chain of a well-formed, live setup whose chains are
independent and alike, and that keeps some chain, has branches, and each of
its chains is normal and holds only faces of the die. -/
private theorem chainLaw_props {s : Setup} {ρ : Roll} (hs : s.WellFormed)
    (hg : s.Live) (hfaced : s.Faced) (hρ : s.roll = some ρ)
    (hlike : ρ.chainLike) (hk : s.results - s.lowest - s.highest ≠ 0) :
    (ρ.chainLaw.2.map (·.2)).sum ≠ 0 ∧
      ∀ ch ∈ (weighted (.single 0) ρ.chainLaw.2).support,
        NormalChain ch ∧ ch.saturates = false ∧
          ∀ x ∈ ch.dice, x ∈ s.replacement.faces := by
  have hsum : ∀ {α : Type} (l : List (α × Nat)), l ≠ [] →
      (∀ p ∈ l, 0 < p.2) → (l.map (·.2)).sum ≠ 0 := fun l hne hpos h0 => by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    have := hpos e he
    omega
  by_cases hch : ∃ count cl, ρ = .chains count cl
  · obtain ⟨count, cl, rfl⟩ := hch
    obtain ⟨_, hlaw⟩ := hg.2 _ hρ
    obtain ⟨hne, hpos⟩ := hs _ hρ
    have hW := hsum cl hne hpos
    refine ⟨hW, fun ch hch => ?_⟩
    obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW hch
    exact hlaw e he
  · have halike : ρ.alike := by
      cases ρ <;> first | rfl | exact (hch ⟨_, _, rfl⟩).elim |
        simp [Roll.chainLike] at hlike
    have hcl : ρ.chainLaw =
        (ρ.dice.1, ρ.dice.2.map fun p => (Chain.single p.1, p.2)) := by
      cases ρ <;> first | rfl | simp [Roll.alike] at halike
    have hn := dice_pos hρ halike hk
    have hne := expand_dice_ne_nil halike (hfaced _ hρ) hn
    have hW : (ρ.dice.2.map (·.2)).sum ≠ 0 := by
      intro h0
      apply hne
      simp only [Roll.expand]
      refine List.eq_nil_of_length_eq_zero ?_
      simpa [List.length_flatMap, Function.comp_def] using h0
    rw [hcl]
    refine ⟨by simpa [Function.comp_def] using hW, fun ch hch => ?_⟩
    rw [weighted_map 0 _ Chain.single hW, PMF.support_map] at hch
    obtain ⟨f, hf, rfl⟩ := hch
    rw [← weightedDie_eq_weighted, weightedDie_eq_customDie_expand] at hf
    have hf' : f ∈ Roll.expand ρ.dice.2 := by
      rcases mem_of_mem_support_customDie hf with h | ⟨h, _⟩
      · exact h
      · exact absurd h hne
    -- The face is a result of some record, and so a face of the die.
    obtain ⟨r, hr, hfr⟩ := exists_results_alike halike hn hf'
    have hface : f ∈ s.replacement.faces := by
      have hgρ := hg.2 ρ hρ
      have hr' : (⟨r.results, s.lowest, s.highest, s.replacement, r.chains⟩ :
          Record) ∈ s.law.support := by
        simp only [law, hρ, Option.map_some, Option.getD_some, PMF.support_map]
        exact ⟨r, hr, rfl⟩
      cases ρ <;> first | exact hgρ _ hr' f hfr | simp [Roll.alike] at halike
    refine ⟨⟨.nil, fun y hy => ?_⟩, rfl, fun y hy => ?_⟩
    · rw [List.mem_singleton.mp hy]
      exact hg.1 f hface
    · simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
        List.mem_singleton] at hy
      rw [hy]
      exact hface

/-- Exploding a setup whose chains are independent and alike, none dropped,
by a limited explode that matches some face of its die, transforms the law of
each chain as the specification explodes each record's chains. -/
private theorem bursts_of_extend {s : Setup} {c : Criterion} {ρ : Roll}
    {n : Nat} (whole : Bool) (hs : s.WellFormed) (hg : s.Live)
    (hfaced : s.Faced) (hρ : s.roll = some ρ)
    (hk : s.results - s.lowest - s.highest ≠ 0)
    (hm : s.replacement.faces.countP c.matches ≠ 0) (hlike : ρ.chainLike)
    (hl : s.lowest = 0) (hh : s.highest = 0) :
    s.Bursts c (some (n + 1)) whole := by
  have hF : s.replacement.faces ≠ [] := fun h => hm (by rw [h]; rfl)
  let ext := extension c (customDie s.replacement.faces) (n + 1)
  let K : Chain → PMF Chain := fun ch =>
    if ch.grows c then ext.map (growChain ch) else PMF.pure ch
  let P := weighted (.single 0) ρ.chainLaw.2
  obtain ⟨hW, hPn⟩ := chainLaw_props hs hg hfaced hρ hlike hk
  have hsat : ρ.isSaturated = false := by
    cases ρ <;> first | rfl | simp [Roll.chainLike] at hlike
  have hexp : ∀ w, s.explode c (some (n + 1)) w =
      { roll := some (ρ.explode 0 0
          (Burst.resolve s.replacement c (some (n + 1)) w) c w),
        die := some s.replacement } := fun w => by
    have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl, hh] at hk
    simp only [explode, hρ, hsat, Bool.false_eq_true, or_false,
      Burst.resolve_some w hm, hl, hh, hk', ↓reduceIte]
  -- The specification explodes each record's kept chains independently.
  have hspec : ∀ r ∈ s.law.support, Spec.explode r c (some (n + 1)) =
      (bagDraws K r.keptChains).map
        fun M => Record.ofChains s.replacement M.toList := fun r hr => by
    have hf := fits_of_live hs hg hr
    simp only [Spec.explode, hf.die]
    rw [explodeChains_congr _ _ (grows' := fun ch => ch.grows c)
      (fun ch _ => by simp)]
    exact explodeChains_eq_bag _ _ _ hf.normal fun ch hch _ t ht =>
      normalChain_growChain hg.1 (hf.faces ch hch) (mem_support_extension hF ht)
  -- Every chain that a normal chain of faces becomes is normal.
  have hKn' : ∀ ch, NormalChain ch → (∀ x ∈ ch.dice, x ∈ s.replacement.faces) →
      ∀ x ∈ (K ch).support, NormalChain x := by
    intro ch hn hfc x hx
    simp only [K] at hx
    split at hx
    · obtain ⟨t, ht, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hx
      exact normalChain_growChain hg.1 hfc (mem_support_extension hF ht)
    · rw [PMF.mem_support_pure_iff] at hx
      rw [hx]
      exact hn
  have hKn : ∀ x ∈ (P.bind K).support, NormalChain x := fun x hx => by
    obtain ⟨ch, hch, hx⟩ := (PMF.mem_support_bind_iff _ _ _).mp hx
    exact hKn' ch (hPn ch hch).1 (hPn ch hch).2.2 x hx
  -- A record of normal chains is viewed by their multiset.
  have hcv : ∀ L : List Chain, (∀ x ∈ L, NormalChain x) →
      (Record.ofChains s.replacement L).chainView =
        ((L : Multiset Chain), s.replacement) :=
    fun L hn => by
      simp only [Record.chainView, keptChains_ofChains _ hn,
        (Record.ofChains_fields _ _).2.2.2]
  have hX := bind_bagDraws_keptChains hs hg hfaced hρ hlike hl hh hk K
  rw [Bursts]
  cases whole with
  | true =>
    have hP' : weighted (.single 0)
        (transformLaw ρ.chainLaw.2 1 fun ch =>
          ((Burst.resolve s.replacement c (some (n + 1)) true).chains c ch,
            0)) =
        P.bind K := by
      have h := weighted_flatMap (Chain.single 0) (Chain.single 0) id
        ρ.chainLaw.2
        (fun ch =>
          (Burst.resolve s.replacement c (some (n + 1)) true).chains c ch)
        (fun _ => 1) (fun _ o => o) (s.replacement.faces.length ^ (n + 1))
        (pow_ne_zero _ (by simpa using hF)) hW
        (fun e _ _ => by rw [one_mul, Burst.sum_chains hm])
      simp only [mul_one, id] at h
      rw [transformLaw_one, h]
      refine congrArg _ (funext fun ch => ?_)
      rw [show (fun o : Chain => o) = id from rfl, PMF.map_id,
        Burst.weighted_chains hm]
    simp only [↓reduceIte, burst, hexp, Roll.explode, hlike, and_self,
      law_chains]
    rw [hP', bind_congr_support hspec, PMF.map_comp, PMF.map_bind]
    simp only [PMF.map_comp]
    calc _ = (chainDraws ρ.chainLaw.1.toNat (P.bind K)).map
            ((fun M => (M, s.replacement)) ∘
              (Multiset.ofList : List Chain → Multiset Chain)) :=
          map_congr_support fun cs hcs =>
            hcv cs fun x hx => hKn x ((mem_support_chainDraws hcs).2 x hx)
      _ = (iidBag ρ.chainLaw.1.toNat (P.bind K)).map
            fun M => (M, s.replacement) := by
          rw [← PMF.map_comp, map_coe_chainDraws]
      _ = ((s.law.bind fun r => bagDraws K r.keptChains)).map
            fun M => (M, s.replacement) := by rw [hX]
      _ = _ := by
          rw [PMF.map_bind]
          refine bind_congr_support fun r hr => map_congr_support fun M hM => ?_
          have hf := fits_of_live hs hg hr
          have hn : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
            obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x
              (Multiset.mem_toList.mp hx)
            exact hKn' ch (hf.normal ch hch) (hf.faces ch hch) x hx
          rw [Function.comp_apply, hcv _ hn, Multiset.coe_toList]
  | false =>
    have hW' : weightedDie (tally (Roll.expand (transformLaw ρ.chainLaw.2 1
        fun ch =>
          ((Burst.resolve s.replacement c (some (n + 1)) false).totals c ch,
            0)))) =
        (P.bind K).map Chain.total := by
      have h := weighted_flatMap (0 : Int) (Chain.single 0) Chain.total
        ρ.chainLaw.2
        (fun ch =>
          (Burst.resolve s.replacement c (some (n + 1)) false).totals c ch)
        (fun _ => 1) (fun _ o => o) (s.replacement.faces.length ^ (n + 1))
        (pow_ne_zero _ (by simpa using hF)) hW
        (fun e _ _ => by rw [one_mul, Burst.sum_totals hm])
      simp only [mul_one] at h
      rw [weightedDie_tally, ← weightedDie_eq_customDie_expand,
        weightedDie_eq_weighted, transformLaw_one, h, PMF.map_bind]
      refine congrArg _ (funext fun ch => ?_)
      rw [show (fun o : Int => o) = id from rfl, PMF.map_id,
        Burst.weighted_totals hm]
      simp only [K]
      split
      · rw [PMF.map_comp]
        rfl
      · rw [PMF.pure_map]
    simp only [Bool.false_eq_true, ↓reduceIte, burst, hexp, Roll.explode, hlike,
      and_self, collapse, Option.map_some, Roll.collapse, map_view_custom, hW']
    rw [bind_congr_support hspec, ← iidBag_map,
      PMF.map_comp, ← hX, PMF.map_bind, PMF.map_bind]
    refine bind_congr_support fun r _ => ?_
    rw [PMF.map_comp]
    refine map_congr_support fun M _ => ?_
    simp only [Function.comp_apply, view_ofChains]
    rw [← Multiset.map_coe, Multiset.coe_toList]

end Setup

namespace Setup

/-- Every result of a record of fixed results, or of a pool, is a result of
one of its multisets. -/
private theorem mem_results_of_fixed {rs : List (Int × Nat)} {x : Int}
    (hx : x ∈ Roll.expand rs) : ∃ p ∈ rs, p.1 = x := by
  simp only [Roll.expand, List.mem_flatMap, List.mem_replicate] at hx
  obtain ⟨p, hp, _, rfl⟩ := hx
  exact ⟨p, hp, rfl⟩

/-- A die of custom faces shows only its faces, or `0` if it has none. -/
private theorem mem_faces_ofFaces {xs : List Int} {x : Int}
    (hx : x ∈ (Die.ofFaces xs).faces) : x ∈ xs ∨ (xs = [] ∧ x = 0) := by
  simp only [Die.ofFaces] at hx
  cases h : xs.mergeSort (· ≤ ·) with
  | nil =>
    rw [h] at hx
    simp only [Die.faces, List.mem_singleton] at hx
    exact .inr ⟨List.eq_nil_of_length_eq_zero (by
      rw [← List.length_mergeSort (le := fun a b : Int => decide (a ≤ b)), h]
      rfl), hx⟩
  | cons y ys =>
    rw [h] at hx
    have hx' : x ∈ y :: ys := hx
    rw [← h] at hx'
    exact .inl ((List.mergeSort_perm xs _).mem_iff.mp hx')

/--
If no end of a chain of a well-formed, live setup's roll may match a
criterion, then no explode by it grows any kept chain of any record that the
setup describes.

# Parameters
- `c`: The criterion.
- `r`: The record.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hρ`: The setup's roll is `ρ`.
- `hshow`: No end of a chain of the roll may match.
- `hr`: The setup describes `r`.
-/
theorem grows_of_not_shows {s : Setup} {ρ : Roll} {c : Criterion}
    (hs : s.WellFormed) (hg : s.Live) (hρ : s.roll = some ρ)
    (hshow : ρ.shows c = false) {r : Record} (hr : r ∈ s.law.support) :
    ∀ ch ∈ r.keptChains, ch.grows c = false := by
  have hf := fits_of_live hs hg hr
  -- A result that no criterion's match reaches leaves a chain that does not
  -- grow.
  have plain : ρ.Plain → (∀ x ∈ r.results, c.matches x = false) →
      ∀ ch ∈ r.keptChains, ch.grows c = false := fun hp hx ch hch => by
    have hc := mem_support_law_plain hρ hp hr
    simp only [Record.keptChains, hc, List.isEmpty_nil, ↓reduceIte,
      List.mem_map] at hch
    obtain ⟨x, hxk, rfl⟩ := hch
    simp only [Chain.grows, Chain.single, Chain.saturates, Chain.last,
      Bool.not_false, Bool.true_and]
    exact hx x ((List.mergeSort_perm _ _).mem_iff.mp
      (List.mem_of_mem_drop (List.mem_of_mem_take hxk)))
  -- The records of the setup are the roll's, with the setup's drops and die.
  have hlaw : ∃ r' ∈ ρ.law.support, r.results = r'.results := by
    simp only [law, hρ, Option.map_some, Option.getD_some, PMF.support_map,
      Set.mem_image] at hr
    obtain ⟨r', hr', rfl⟩ := hr
    exact ⟨r', hr', rfl⟩
  obtain ⟨r', hr', hres⟩ := hlaw
  -- Alike dice show only faces of the die that the roll shows.
  have alike : ρ.alike → (∀ x ∈ ρ.replacement.faces, c.matches x = false) →
      ∀ ch ∈ r.keptChains, ch.grows c = false := fun ha hx =>
    plain (by cases ρ <;> trivial) fun x hxr => by
      have := faithful_roll ha r' (by rwa [law_roll]) x (hres ▸ hxr)
      exact hx x this
  cases ρ with
  | range start stop =>
    refine alike rfl fun x hx => ?_
    simp only [Roll.shows, List.any_eq_false] at hshow
    simpa using hshow x hx
  | standard count faces =>
    refine alike rfl fun x hx => ?_
    simp only [Roll.shows, List.any_eq_false] at hshow
    simpa using hshow x hx
  | custom count faces =>
    refine alike rfl fun x hx => ?_
    rcases mem_faces_ofFaces hx with hx | ⟨he, rfl⟩
    · obtain ⟨p, hp, rfl⟩ := mem_results_of_fixed hx
      have hne : faces ≠ [] := List.ne_nil_of_mem hp
      simp only [Roll.shows, hne, ↓reduceIte, List.any_eq_false] at hshow
      simpa using hshow p hp
    · have : faces = [] := by
        cases faces with
        | nil => rfl
        | cons p ps =>
          have hp := p.2
          simp only [Roll.expand, List.flatMap_cons, List.append_eq_nil_iff,
            List.replicate_eq_nil_iff] at he
          exfalso
          have hw := (hs _ hρ).2 p List.mem_cons_self
          omega
      simpa [Roll.shows, this] using hshow
  | fixed rs =>
    refine plain trivial fun x hx => ?_
    simp only [Roll.law, PMF.mem_support_pure_iff] at hr'
    rw [hres, hr'] at hx
    obtain ⟨p, hp, rfl⟩ := mem_results_of_fixed hx
    simp only [Roll.shows, List.any_eq_false] at hshow
    simpa using hshow p hp
  | pool outcomes =>
    refine plain trivial fun x hx => ?_
    simp only [Roll.law, PMF.support_map, Set.mem_image] at hr'
    obtain ⟨rs, hrs, rfl⟩ := hr'
    rw [hres] at hx
    obtain ⟨p, hp, rfl⟩ := mem_results_of_fixed hx
    have hrs' : ∃ e ∈ outcomes, e.1 = rs := by
      rcases mem_support_weighted hrs with h | rfl
      · exact h
      · cases outcomes with
        | nil => simp at hp
        | cons e _ => exact ⟨e, List.mem_cons_self, rfl⟩
    obtain ⟨e, he, rfl⟩ := hrs'
    simp only [Roll.shows, List.any_eq_false] at hshow
    have h2 := hshow e he
    simp only [List.any_eq_true, not_exists, not_and] at h2
    simpa using h2 p hp
  | chains count cl =>
    obtain ⟨_, hlaw⟩ := hg.2 _ hρ
    obtain ⟨hne, hpos⟩ := hs _ hρ
    have hW : (cl.map (·.2)).sum ≠ 0 := by
      obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
      have := hpos e he
      intro h0
      have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
      omega
    obtain ⟨cs, hcs, rfl⟩ := mem_support_law_chains hρ hr
    -- Every kept chain is one drawn, and so a chain of the law.
    have hn : ∀ x ∈ cs, NormalChain x := fun x hx => by
      obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW
        ((mem_support_chainDraws hcs).2 x hx)
      exact (hlaw e he).1
    intro ch hch
    rw [Record.keptChains_drops _ _ (Record.ofChains_fields _ _).2.1
      (Record.ofChains_fields _ _).2.2.1 (Record.length_chains_ofChains _ _)]
      at hch
    have hch' := mem_of_coe_eq (keptChains_ofChains _ hn)
      (List.mem_of_mem_drop (List.mem_of_mem_take hch))
    obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW
      ((mem_support_chainDraws hcs).2 _ hch')
    simp only [Roll.shows, List.any_eq_false] at hshow
    simpa using hshow e he
  | stacked ms laws =>
    obtain ⟨_, _, _, hwl⟩ := hs _ hρ
    obtain ⟨cs, hcs, rfl⟩ := mem_support_law_stacked hρ hr
    have hn : ∀ x ∈ cs, NormalChain x := fun x hx => by
      obtain ⟨_, hpool, hlaws⟩ := hg.2 _ hρ
      obtain ⟨e, he, p, hp, hx⟩ := hcs x hx
      rcases mem_support_became hx with rfl | ⟨e', he', q, hq, rfl⟩
      · exact (hpool e he p hp).1
      · exact (hlaws e' he' q hq).1
    intro ch hch
    rw [Record.keptChains_drops _ _ (Record.ofChains_fields _ _).2.1
      (Record.ofChains_fields _ _).2.2.1 (Record.length_chains_ofChains _ _)]
      at hch
    have hch' := mem_of_coe_eq (keptChains_ofChains _ hn)
      (List.mem_of_mem_drop (List.mem_of_mem_take hch))
    -- Every chain drawn is one that no end of the roll's matches.
    obtain ⟨e, he, p, hp, hx⟩ := hcs ch hch'
    simp only [Roll.shows, List.any_eq_false] at hshow
    have h2 := hshow e he
    simp only [List.any_eq_true, not_exists, not_and] at h2
    have h3 := h2 p hp
    unfold became at hx
    split at hx
    · rename_i law heq
      rw [heq] at h3
      simp only [List.any_eq_true, not_exists, not_and] at h3
      obtain ⟨hne, hpos⟩ := hwl _ (mem_of_lookup heq)
      have hW : (law.map (·.2)).sum ≠ 0 := by
        obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ hne
        intro h0
        exact (hpos q hq).ne' (List.sum_eq_zero_iff.mp h0 q.2
          (List.mem_map_of_mem hq))
      obtain ⟨q, hq, rfl⟩ := mem_of_mem_support_weighted hW hx
      simpa using h3 q hq
    · rename_i heq
      rw [heq] at h3
      rw [PMF.mem_support_pure_iff] at hx
      subst hx
      simpa using h3

end Setup

namespace Setup

/-- The last die of a chain that may still grow is one of its dice. -/
private theorem last_mem_dice {ch : Chain} (h : ch.saturates = false) :
    ch.last ∈ ch.dice := by
  cases ch with
  | live body last =>
    exact (List.mergeSort_perm _ _).mem_iff.mpr List.mem_cons_self
  | saturated => simp [Chain.saturates] at h

/--
Exploding a setup, as `Mixture::explode` in `propagation/mixture.rs` explodes
each setup, describes the specification's explode of the records that it
describes, as every later instruction reads them (`Setup.Bursts`), whether the
explode does nothing, saturates every chain still kept, or extends the
independent and alike chains of a roll none of which has been dropped; an
unlimited explode that may extend a chain the pass refuses. Mirrors
`Setup::explode` and `Setup::collapse` in `propagation/mixture.rs`, and
`burst`, `Roll::explode`, `Roll::saturate`, and `Roll::collapse` in
`propagation/chain.rs`. `Setup.law_burst`, in `XdySpec/Stacks.lean`, lifts the
hypothesis that every explode that extends extends alike chains, and that of
the collapse of stacks, which it proves.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hfaced`: Every die of the setup has a face.
- `he`: The explode is not one that the pass refuses.
- `ha`: Every explode that extends extends alike chains.
- `hcol`: If the setup's roll is stacks, reducing its chains to their totals
  leaves the law of its records, as every later drop, keep, and sum views
  them, as `law_collapse` proves of every other roll.
-/
theorem law_burst_alike {s : Setup} (c : Criterion) (limit : Option Nat)
    (whole : Bool) (hs : s.WellFormed) (hg : s.Live) (hfaced : s.Faced)
    (he : s.Explodes c limit) (ha : s.ExtendsAlike c limit)
    (hcol : (∃ ms ls, s.roll = some (.stacked ms ls)) →
      s.collapse.law.map Record.view = s.law.map Record.view) :
    s.Bursts c limit whole := by
  have hcol' : s.collapse.law.map Record.view = s.law.map Record.view := by
    by_cases hst : ∃ ms ls, s.roll = some (.stacked ms ls)
    · exact hcol hst
    · exact law_collapse hs hg fun ms ls h => hst ⟨ms, ls, h⟩
  -- Records that keep no chains leave nothing to explode.
  have hnone : s.results - s.lowest - s.highest = 0 →
      ∀ r ∈ s.law.support, r.keptChains = [] := fun h r hr =>
    List.eq_nil_of_length_eq_zero ((length_keptChains hs hg hr).trans h)
  cases hρ : s.roll with
  | none =>
    have hk : s.results - s.lowest - s.highest = 0 := by simp [results, hρ]
    exact bursts_of_stays whole hs hg hcol' (fun w => by simp [explode, hρ])
      (fun r hr ch hch => by rw [hnone hk r hr] at hch; cases hch)
      fun _ => .inr hk
  | some ρ =>
    by_cases hk : s.results - s.lowest - s.highest = 0
    · exact bursts_of_stays whole hs hg hcol'
        (fun w => by simp [explode, hρ, hk])
        (fun r hr ch hch => by rw [hnone hk r hr] at hch; cases hch)
        fun _ => .inr hk
    have hsat := isSaturated_of_live hg hρ
    have hstays : (∀ w, Burst.resolve s.replacement c limit w = .nothing ∨
        Burst.resolve s.replacement c limit w = .unbounded) →
        ∀ w, s.explode c limit w = s := fun h w => by
      simp only [explode, hρ, hk, hsat, Bool.false_eq_true, or_self,
        ↓reduceIte]
      rcases h w with h | h <;> rw [h]
    -- A chain still kept grows only if the explode matches its last die,
    -- which is a face of the die.
    have hlast : ∀ r ∈ s.law.support, ∀ ch ∈ r.keptChains,
        ch.grows c = c.matches ch.last ∧ ch.last ∈ s.replacement.faces :=
      fun r hr ch hch => by
        have hf := fits_of_live hs hg hr
        refine ⟨by simp [Chain.grows, hf.live ch hch], hf.faces ch hch _
          (last_mem_dice (hf.live ch hch))⟩
    by_cases hm : s.replacement.faces.countP c.matches = 0
    · -- No face matches, so the explode does nothing.
      refine bursts_of_stays whole hs hg hcol'
        (hstays fun w => .inl (by simp [Burst.resolve, hm]))
        (fun r hr ch hch _ => ?_) fun _ => ?_
      · obtain ⟨hgr, hl⟩ := hlast r hr ch hch
        rw [hgr]
        exact Bool.eq_false_iff.mpr fun hc =>
          absurd hm (List.countP_pos_iff.mpr ⟨_, hl, hc⟩).ne'
      · -- Some record keeps a chain, whose last die is a face.
        obtain ⟨r, hr⟩ : ∃ r, r ∈ s.law.support :=
          ⟨_, (PMF.support_nonempty _).some_mem⟩
        have hlen := length_keptChains hs hg hr
        obtain ⟨ch, hch⟩ := List.exists_mem_of_ne_nil r.keptChains
          (fun h => hk (by rw [← hlen, h]; rfl))
        have hfc := (hlast r hr ch hch).2
        refine .inl fun hw => ?_
        rw [Die.facesWithin, List.all_eq_true] at hw
        exact absurd hm (List.countP_pos_iff.mpr ⟨_, hfc, hw _ hfc⟩).ne'
    cases limit with
    | none =>
      by_cases hall : s.replacement.faces.countP c.matches =
          s.replacement.faces.length
      · have hF : s.replacement.faces ≠ [] := fun h => hm (by rw [h]; rfl)
        exact bursts_of_saturate (v := s.replacement.saturatedValue) whole hs hg
          hρ hk fun w => by simp [Burst.resolve, hall, hF]
      · have hu : ∀ w, Burst.resolve s.replacement c none w = .unbounded :=
          fun w => by simp [Burst.resolve, hm, hall]
        have hshow : ρ.shows c = false := by
          have := he ρ hρ hk
          rw [hu true] at this
          exact this
        refine bursts_of_stays whole hs hg hcol' (hstays fun w => .inr (hu w))
          (fun r hr ch hch _ => grows_of_not_shows hs hg hρ hshow hr ch hch)
          fun _ => .inl fun hw => hall ?_
        rw [Die.facesWithin, List.all_eq_true] at hw
        exact List.countP_eq_length.mpr hw
    | some l =>
      cases l with
      | zero =>
        exact bursts_of_stays whole hs hg hcol'
          (hstays fun w => .inl (by simp [Burst.resolve, hm]))
          (fun _ _ _ _ h => absurd rfl h) fun h => by cases h
      | succ n =>
        have := ha ρ hρ hk
        rw [Burst.resolve_some true hm] at this
        obtain ⟨hlike, hl, hh⟩ := this
        exact bursts_of_extend whole hs hg hfaced hρ hk hm hlike hl hh

end Setup

/-! ## What an explode leaves -/

/-- Ways of positive weight from faces of positive weight. -/
private theorem growths_pos {S K : Type} {faces : List (Int × Nat)}
    {c : Criterion} {extend : S → Int → S} {close : S → Int → K}
    (hpos : ∀ f ∈ faces, 0 < f.2) :
    ∀ (r : Nat) (s : S), ∀ g ∈ growths faces c extend close r s, 0 < g.2
  | 0, s, g, hg => by
    simp only [growths, List.mem_map] at hg
    obtain ⟨f, hf, rfl⟩ := hg
    exact hpos f hf
  | r + 1, s, g, hg => by
    simp only [growths, List.mem_flatMap] at hg
    obtain ⟨f, hf, hg⟩ := hg
    have hF : 0 < (faces.map (·.2)).sum :=
      List.sum_pos _ (fun x hx => by
        obtain ⟨f', hf', rfl⟩ := List.mem_map.mp hx
        exact hpos f' hf') (by simpa using List.ne_nil_of_mem hf)
    split at hg
    · obtain ⟨g', hg', rfl⟩ := List.mem_map.mp hg
      exact Nat.mul_pos (hpos f hf) (growths_pos hpos r _ g' hg')
    · rw [List.mem_singleton] at hg
      subst hg
      exact Nat.mul_pos (hpos f hf) (Nat.pow_pos hF)

/-- Ways from faces, if there are any. -/
private theorem growths_ne_nil {S K : Type} {faces : List (Int × Nat)}
    {c : Criterion} {extend : S → Int → S} {close : S → Int → K}
    (hne : faces ≠ []) : ∀ (r : Nat) (s : S),
      growths faces c extend close r s ≠ []
  | 0, s => by simpa [growths] using hne
  | r + 1, s => by
    obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ hne
    intro h
    have := List.flatMap_eq_nil_iff.mp h f hf
    split at this
    · exact growths_ne_nil hne r _ (List.map_eq_nil_iff.mp this)
    · cases this

/-- Tails of faces draw only faces. -/
private theorem growths_tails_faces {faces : List (Int × Nat)} {c : Criterion}
    {F : List Int} (hf : ∀ f ∈ faces, f.1 ∈ F) :
    ∀ (r : Nat) (s : List Int), (∀ x ∈ s, x ∈ F) →
      ∀ g ∈ growths faces c tailExtend tailClose r s,
        (∀ x ∈ g.1.1, x ∈ F) ∧ g.1.2 ∈ F
  | 0, s, hs, g, hg => by
    simp only [growths, List.mem_map] at hg
    obtain ⟨f', hf', rfl⟩ := hg
    exact ⟨hs, hf f' hf'⟩
  | r + 1, s, hs, g, hg => by
    simp only [growths, List.mem_flatMap] at hg
    obtain ⟨f', hf', hg⟩ := hg
    split at hg
    · obtain ⟨g', hg', rfl⟩ := List.mem_map.mp hg
      refine growths_tails_faces hf r _ (fun x hx => ?_) g' hg'
      rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
        rfl | hx
      · exact hf f' hf'
      · exact hs x hx
    · rw [List.mem_singleton] at hg
      subst hg
      exact ⟨hs, hf f' hf'⟩

/-- The tally of some faces holds only them, each with positive weight. -/
private theorem mem_tally {xs : List Int} {p : Int × Nat} (hp : p ∈ tally xs) :
    p.1 ∈ xs ∧ 0 < p.2 := by
  refine ⟨?_, (tally_wellFormed xs).2 p hp⟩
  simp only [tally, List.mem_map, List.mem_mergeSort, List.mem_dedup] at hp
  obtain ⟨x, hx, rfl⟩ := hp
  exact hx

/-- The tally of the faces of a die that a criterion matches somewhere. -/
private theorem tally_ne_nil {d : Die} {c : Criterion}
    (hm : d.faces.countP c.matches ≠ 0) : tally d.faces ≠ [] := fun h => by
  have := (expand_tally_perm d.faces).length_eq
  rw [h] at this
  exact Burst.faces_ne_nil hm this.symm

/--
Every chain that a chain still kept may become after an explode limited to
`n + 1` times weighs something, and there are some.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem Burst.chains_pos {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (ch : Chain) :
    (resolve d c (some (n + 1)) true).chains c ch ≠ [] ∧
      ∀ o ∈ (resolve d c (some (n + 1)) true).chains c ch, 0 < o.2 := by
  rw [resolve_some true hm]
  simp only [↓reduceIte, chains]
  split
  · refine ⟨by simpa using growths_ne_nil (tally_ne_nil hm) n [],
      fun o ho => ?_⟩
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp ho
    exact growths_pos (fun f hf => (mem_tally hf).2) n [] t ht
  · refine ⟨by simp, fun o ho => ?_⟩
    rw [List.mem_singleton] at ho
    subst ho
    simpa [padding] using Nat.pos_of_ne_zero (pow_ne_zero (n + 1)
      (faces_ne_nil hm))

/--
Every total that a chain still kept may reach after an explode limited to
`n + 1` times weighs something, and there are some.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
-/
theorem Burst.totals_pos {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0) (ch : Chain) :
    (resolve d c (some (n + 1)) false).totals c ch ≠ [] ∧
      ∀ o ∈ (resolve d c (some (n + 1)) false).totals c ch, 0 < o.2 := by
  rw [resolve_some false hm]
  simp only [Bool.false_eq_true, ↓reduceIte, totals]
  split
  · refine ⟨by simpa using growths_ne_nil (tally_ne_nil hm) n ({} : Fold),
      fun o ho => ?_⟩
    obtain ⟨f, hf, rfl⟩ := List.mem_map.mp ho
    exact growths_pos (fun f hf => (mem_tally hf).2) n ({} : Fold) f hf
  · refine ⟨by simp, fun o ho => ?_⟩
    rw [List.mem_singleton] at ho
    subst ho
    simpa [padding] using Nat.pos_of_ne_zero (pow_ne_zero (n + 1)
      (faces_ne_nil hm))

/--
Every chain that a chain still kept may become after an explode limited to
`n + 1` times is normal, may still grow, and holds only faces of the die, if
the chain is so and the faces lie within the `i32` range.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hm`: The explode matches some face of the die.
- `hD`: The faces of the die lie within the `i32` range.
- `hch`: The chain is normal, may still grow, and holds only faces of the die.
-/
theorem Burst.chains_live {d : Die} {c : Criterion} {n : Nat}
    (hm : d.faces.countP c.matches ≠ 0)
    (hD : ∀ x ∈ d.faces, i32Min ≤ x ∧ x ≤ i32Max) {ch : Chain}
    (hch : NormalChain ch ∧ ch.saturates = false ∧
      ∀ x ∈ ch.dice, x ∈ d.faces) :
    ∀ o ∈ (resolve d c (some (n + 1)) true).chains c ch,
      NormalChain o.1 ∧ o.1.saturates = false ∧
        ∀ x ∈ o.1.dice, x ∈ d.faces := by
  obtain ⟨hn, hlive, hfc⟩ := hch
  rw [resolve_some true hm]
  simp only [↓reduceIte, chains]
  split
  · intro o ho
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp ho
    have htf := growths_tails_faces (F := d.faces)
      (fun f hf => (mem_tally hf).1) n [] (by simp) t ht
    refine ⟨normalChain_growChain hD hfc htf, rfl, fun x hx => ?_⟩
    simp only [growChain, Chain.dice] at hx
    rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx)
      with rfl | hx
    · exact htf.2
    · rcases List.mem_append.mp
        ((List.mergeSort_perm _ _).mem_iff.mp hx) with hx | hx
      · exact hfc x hx
      · exact htf.1 x hx
  · intro o ho
    rw [List.mem_singleton] at ho
    subst ho
    exact ⟨hn, hlive, hfc⟩

namespace Setup

/-- The law of each chain of a well-formed setup whose chains are independent
and alike weighs each of its chains positively. -/
private theorem chainLaw_pos {s : Setup} {ρ : Roll} (hs : s.WellFormed)
    (hρ : s.roll = some ρ) (hlike : ρ.chainLike) :
    ∀ e ∈ ρ.chainLaw.2, 0 < e.2 := by
  cases ρ with
  | chains count cl => exact (hs _ hρ).2
  | range start stop =>
    intro e he
    simp only [Roll.chainLaw, Roll.dice, List.mem_map] at he
    obtain ⟨p, hp, rfl⟩ := he
    exact (mem_tally hp).2
  | standard count faces =>
    intro e he
    simp only [Roll.chainLaw, Roll.dice, List.mem_map] at he
    obtain ⟨p, hp, rfl⟩ := he
    exact (mem_tally hp).2
  | custom count faces =>
    intro e he
    simp only [Roll.chainLaw, Roll.dice, List.mem_map] at he
    obtain ⟨p, hp, rfl⟩ := he
    exact (hs _ hρ).2 p hp
  | fixed | pool | stacked => simp [Roll.chainLike] at hlike

/--
A burst that extends only alike chains (`Setup.ExtendsAlike`) leaves a
well-formed, live setup with faces well formed and with faces, and, if it
keeps the chains whole and does not saturate them, live too, so that
the law of a later explode holds of it (`Setup.law_burst` in
`XdySpec/Stacks.lean`): an
explode that does nothing leaves the setup, or reduces it to custom dice of
its totals; one that saturates leaves one multiset of the saturated chains, or
of their value; and one that extends leaves i.i.d. chains of the die's faces,
or custom dice of their totals.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is live.
- `hfaced`: Every die of the setup has a face.
- `ha`: Every explode that extends extends alike chains.
- `hcp`: If the setup's roll is stacks, reducing its chains to their totals
  leaves a well-formed setup with faces.
-/
theorem burst_invariants_alike {s : Setup} (c : Criterion)
    (limit : Option Nat) (whole : Bool) (hs : s.WellFormed) (hg : s.Live)
    (hfaced : s.Faced) (ha : s.ExtendsAlike c limit)
    (hcp : (∃ ms ls, s.roll = some (.stacked ms ls)) →
      s.collapse.WellFormed ∧ s.collapse.Faced) :
    (s.burst c limit whole).WellFormed ∧ (s.burst c limit whole).Faced ∧
      (whole = true → (∀ v, Burst.resolve s.replacement c limit true ≠
        .saturate v) → (s.burst c limit true).Live) := by
  -- Reducing the chains to their totals leaves custom dice of them.
  have hcollapse : (s.collapse).WellFormed ∧ (s.collapse).Faced := by
    cases hρ : s.roll with
    | none => exact ⟨fun ρ h => by simp [collapse, hρ] at h,
        fun ρ h => by simp [collapse, hρ] at h⟩
    | some ρ =>
      cases ρ with
      | chains count cl =>
        obtain ⟨hne, hpos⟩ := hs _ hρ
        refine ⟨fun ρ' h => ?_, fun ρ' h => ?_⟩
        · simp only [collapse, hρ, Option.map_some, Option.mem_def,
            Option.some.injEq] at h
          subst h
          exact tally_wellFormed _
        · simp only [collapse, hρ, Option.map_some, Option.mem_def,
            Option.some.injEq] at h
          subst h
          refine .inr fun he' => ?_
          obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
          have hx : e.1.total ∈ Roll.expand
              (cl.map fun e => (e.1.total, e.2)) :=
            List.mem_flatMap.mpr ⟨(e.1.total, e.2), List.mem_map_of_mem he,
              List.mem_replicate.mpr ⟨(hpos e he).ne', rfl⟩⟩
          have := (expand_tally_perm _).mem_iff.mpr hx
          rw [he'] at this
          cases this
      | stacked => exact hcp ⟨_, _, hρ⟩
      | _ =>
        have : s.collapse = s := by
          rcases s with ⟨_ | _, _, _, _⟩ <;> simp_all [collapse, Roll.collapse]
        rw [this]
        exact ⟨hs, hfaced⟩
  -- An explode that leaves the setup alone leaves its invariants.
  have hstay : (∀ w, s.explode c limit w = s) →
      (s.burst c limit whole).WellFormed ∧ (s.burst c limit whole).Faced ∧
        (whole = true → (∀ v, Burst.resolve s.replacement c limit true ≠
          .saturate v) → (s.burst c limit true).Live) :=
      fun h => by
    cases whole with
    | true =>
      simp only [burst, ↓reduceIte, h]
      exact ⟨hs, hfaced, fun _ _ => hg⟩
    | false =>
      simp only [burst, Bool.false_eq_true, ↓reduceIte, h]
      exact ⟨hcollapse.1, hcollapse.2, fun h => by cases h⟩
  cases hρ : s.roll with
  | none => exact hstay fun w => by simp [explode, hρ]
  | some ρ =>
    by_cases hk : s.results - s.lowest - s.highest = 0
    · exact hstay fun w => by simp [explode, hρ, hk]
    have hsat := isSaturated_of_live hg hρ
    have hstays : (∀ w, Burst.resolve s.replacement c limit w = .nothing ∨
        Burst.resolve s.replacement c limit w = .unbounded) →
        ∀ w, s.explode c limit w = s := fun h w => by
      simp only [explode, hρ, hk, hsat, Bool.false_eq_true, or_self,
        ↓reduceIte]
      rcases h w with h | h <;> rw [h]
    by_cases hm : s.replacement.faces.countP c.matches = 0
    · exact hstay (hstays fun w => .inl (by simp [Burst.resolve, hm]))
    have hF : s.replacement.faces ≠ [] := fun h => hm (by rw [h]; rfl)
    cases limit with
    | none =>
      by_cases hall : s.replacement.faces.countP c.matches =
          s.replacement.faces.length
      · -- The explode saturates every chain still kept.
        have hb : ∀ w, Burst.resolve s.replacement c none w =
            .saturate s.replacement.saturatedValue := fun w => by
          simp [Burst.resolve, hall, hF]
        have hT := total_ne_zero (hs ρ hρ)
        have hexp : ∀ w, s.explode c none w =
            { roll := some (ρ.saturate s.replacement.saturatedValue
                (s.results - s.lowest - s.highest) w),
              die := some s.replacement } := fun w => by
          simp only [explode, hρ, hk, hsat, Bool.false_eq_true, or_self,
            ↓reduceIte, hb w]
        cases whole with
        | true =>
          simp only [burst, ↓reduceIte, hexp, Roll.saturate]
          refine ⟨fun ρ' h => ?_, fun ρ' h => ?_,
            fun _ h => absurd (hb true) (h _)⟩
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            refine ⟨by simp, fun e he => ?_, fun e he => ?_, fun e he => ?_⟩
            · simp only [List.mem_singleton] at he
              subst he
              exact ⟨fun p hp => by
                simp only [List.mem_singleton] at hp
                subst hp
                exact Nat.pos_of_ne_zero hk, Nat.pos_of_ne_zero hT⟩
            · simp only [List.mem_singleton] at he
              subst he
              rfl
            · cases he
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            trivial
        | false =>
          simp only [burst, Bool.false_eq_true, ↓reduceIte, hexp,
            Roll.saturate, collapse, Option.map_some, Roll.collapse]
          refine ⟨fun ρ' h => ?_, fun ρ' h => ?_, fun h => by cases h⟩
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            refine ⟨by simp, fun e he => ?_, fun e he => ?_⟩
            · simp only [List.mem_singleton] at he
              subst he
              exact ⟨⟨.cons (fun _ h => by cases h) .nil, fun p hp => by
                simp only [List.mem_singleton] at hp
                subst hp
                exact Nat.pos_of_ne_zero hk⟩, Nat.pos_of_ne_zero hT⟩
            · simp only [List.mem_singleton] at he
              subst he
              rfl
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            trivial
      · exact hstay (hstays fun w => .inr (by simp [Burst.resolve, hm, hall]))
    | some l =>
      cases l with
      | zero => exact hstay (hstays fun w => .inl (by simp [Burst.resolve, hm]))
      | succ n =>
        have := ha ρ hρ hk
        rw [Burst.resolve_some true hm] at this
        obtain ⟨hlike, hl, hh⟩ := this
        obtain ⟨hW, hPn⟩ := chainLaw_props hs hg hfaced hρ hlike hk
        have hpos := chainLaw_pos hs hρ hlike
        have hexp : ∀ w, s.explode c (some (n + 1)) w =
            { roll := some (ρ.explode 0 0 (Burst.resolve s.replacement c
                (some (n + 1)) w) c w),
              die := some s.replacement } := fun w => by
          have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl, hh] at hk
          simp only [explode, hρ, hsat, Bool.false_eq_true, or_false,
            Burst.resolve_some w hm, hl, hh, hk', ↓reduceIte]
        have hlne : ρ.chainLaw.2 ≠ [] := fun h => hW (by rw [h]; rfl)
        have htally : tally s.replacement.faces ≠ [] := fun h => hF (by
          have := (expand_tally_perm s.replacement.faces).length_eq
          rw [h] at this
          exact List.eq_nil_of_length_eq_zero this.symm)
        have htpos : ∀ f ∈ tally s.replacement.faces, 0 < f.2 :=
          fun f hf => (mem_tally hf).2
        have hFpos : 0 < s.replacement.faces.length ^ (n + 1) :=
          Nat.pow_pos (List.length_pos_iff.mpr hF)
        -- The entries of a transformed law, of positive weight.
        have htrans : ∀ {T : Type} (h : Chain → List (T × Nat)),
            (∀ ch ∈ (weighted (.single 0) ρ.chainLaw.2).support, h ch ≠ [] ∧
              ∀ o ∈ h ch, 0 < o.2) →
            transformLaw ρ.chainLaw.2 1 (fun ch => (h ch, 0)) ≠ [] ∧
              ∀ e ∈ transformLaw ρ.chainLaw.2 1 (fun ch => (h ch, 0)),
                0 < e.2 := fun h hh' => by
          rw [transformLaw_one]
          have hsup : ∀ e ∈ ρ.chainLaw.2,
              e.1 ∈ (weighted (.single 0) ρ.chainLaw.2).support :=
            fun e he => mem_support_weighted_of_mem he (hpos e he).ne'
          refine ⟨fun h0 => ?_, fun e he => ?_⟩
          · obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hlne
            have := List.flatMap_eq_nil_iff.mp h0 e he
            exact (hh' e.1 (hsup e he)).1 (List.map_eq_nil_iff.mp this)
          · simp only [List.mem_flatMap, List.mem_map] at he
            obtain ⟨e₀, he₀, o, ho, rfl⟩ := he
            exact Nat.mul_pos (hpos e₀ he₀) ((hh' e₀.1 (hsup e₀ he₀)).2 o ho)
        cases whole with
        | true =>
          have hB := Burst.resolve_some (d := s.replacement) (c := c) (n := n)
            true hm
          simp only [↓reduceIte] at hB
          -- What each chain becomes.
          have hch : ∀ ch ∈ (weighted (.single 0) ρ.chainLaw.2).support,
              (Burst.resolve s.replacement c (some (n + 1)) true).chains c ch ≠
                [] ∧ ∀ o ∈ (Burst.resolve s.replacement c (some (n + 1))
                  true).chains c ch, 0 < o.2 ∧ NormalChain o.1 ∧
                    o.1.saturates = false ∧
                    ∀ x ∈ o.1.dice, x ∈ s.replacement.faces := by
            intro ch hch
            exact ⟨(Burst.chains_pos hm ch).1, fun o ho =>
              ⟨(Burst.chains_pos hm ch).2 o ho,
                Burst.chains_live hm hg.1 (hPn ch hch) o ho⟩⟩
          obtain ⟨hne', hpos'⟩ := htrans _ fun ch hc =>
            ⟨(hch ch hc).1, fun o ho => ((hch ch hc).2 o ho).1⟩
          simp only [burst, ↓reduceIte, hexp, Roll.explode, hlike, and_self]
          refine ⟨fun ρ' h => ?_, fun ρ' h => ?_, fun _ _ => ?_⟩
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            exact ⟨hne', hpos'⟩
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            trivial
          · refine ⟨hg.1, fun ρ' h => ?_⟩
            simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            refine ⟨rfl, fun e he => ?_⟩
            simp only [transformLaw_one, List.mem_flatMap, List.mem_map] at he
            obtain ⟨e₀, he₀, o, ho, rfl⟩ := he
            have hsup := mem_support_weighted_of_mem (d := Chain.single 0) he₀
              (hpos e₀ he₀).ne'
            exact ((hch e₀.1 hsup).2 o ho).2
        | false =>
          have hB := Burst.resolve_some (d := s.replacement) (c := c) (n := n)
            false hm
          simp only [Bool.false_eq_true, ↓reduceIte] at hB
          have hto : ∀ ch ∈ (weighted (.single 0) ρ.chainLaw.2).support,
              (Burst.resolve s.replacement c (some (n + 1)) false).totals c ch ≠
                [] ∧ ∀ o ∈ (Burst.resolve s.replacement c (some (n + 1))
                  false).totals c ch, 0 < o.2 := by
            intro ch _
            have hne0 : growths (tally s.replacement.faces) c foldExtend
                foldExtend n ({} : Fold) ≠ [] := growths_ne_nil htally n _
            rw [hB]
            simp only [Burst.totals]
            split
            · refine ⟨by simpa using hne0, fun o ho => ?_⟩
              obtain ⟨f, hf, rfl⟩ := List.mem_map.mp ho
              exact growths_pos htpos n ({} : Fold) f hf
            · refine ⟨by simp, fun o ho => ?_⟩
              rw [List.mem_singleton] at ho
              subst ho
              simpa [Burst.padding] using hFpos
          obtain ⟨hne', hpos'⟩ := htrans _ hto
          simp only [burst, Bool.false_eq_true, ↓reduceIte, hexp, Roll.explode,
            hlike, and_self, collapse, Option.map_some, Roll.collapse]
          refine ⟨fun ρ' h => ?_, fun ρ' h => ?_, fun h => by cases h⟩
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            exact tally_wellFormed _
          · simp only [Option.mem_def, Option.some.injEq] at h
            subst h
            refine .inr fun h0 => ?_
            obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne'
            have hx : e.1 ∈ Roll.expand (transformLaw ρ.chainLaw.2 1
                fun ch => ((Burst.resolve s.replacement c (some (n + 1))
                  false).totals c ch, 0)) :=
              List.mem_flatMap.mpr ⟨e, he,
                List.mem_replicate.mpr ⟨(hpos' e he).ne', rfl⟩⟩
            have := (expand_tally_perm _).mem_iff.mpr hx
            rw [h0] at this
            cases this

end Setup

end Xdy.Spec
