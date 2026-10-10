import XdySpec.Explode

/-!
# Stacks in the forward pass

Once a drop has made the chains of a record unlike, or a split has fixed them,
the forward pass in `xdy/src/distribution/propagation.rs` keeps not the law of
each chain but the multisets of the chains still kept (`Roll::Stacks` in
`propagation/chain.rs`), which a later reroll or explode transforms. Every
drop, keep, and sum counts each chain still kept as one die, worth its total,
so the chains still kept of a record are a window of its chains sorted by
`Xdy.Chain.le`, whose totals come first. This module proves that the
multisets of `XdySpec/Chains.lean` are the laws of those chains, and so the
laws of the explodes that transform them:

- The kept chains of every record that a roll fills, as a multiset, have the
  law of the roll's multisets of chains (`Roll.stacks`), over their weights
  (`Roll.map_keptChains_law`): of i.i.d. chains, independent draws of the law
  of each (`Outcomes.weighted_outcomeStates`); of stacks whose chains are
  still themselves, their pool; and of any other roll, its outcomes, each
  result the chain of one die.
- Dropping chains from a record that drops none keeps the window of its kept
  chains (`Record.keptChains_drops`, in `XdySpec/Explode.lean`), and the window
  of normal chains sorted by `Xdy.Chain.le` depends only on their multiset,
  which `slice` slices (`coe_window_slice`).
- So the kept chains of every record that a setup describes, after its drops
  and keeps, have the law of the kept pool of its roll's multisets of chains
  (`Setup.map_keptChains_law`), as `kept_pool` makes it.
- A transformed pool, over its weights, draws each multiset of the pool,
  slices it, and draws each copy of each chain still kept independently from
  the law of what it becomes (`weighted_transformPool`), as `transform_pool`
  joins them, if each chain's law weighs alike but for the padding of its kept
  dice, and every multiset keeps as many chains; and the chains of stacks are
  independent draws of what each has become (`map_coe_stackDraws`).
- So an explode of the chains still kept of each multiset, which
  `Roll::explode` makes of a roll after a drop, or of fixed results or a pool,
  into stacks of the kept pool (`transformStacks`) or a pool of totals
  (`transformPool`), describes the specification's explode of the setup's
  records, and `Setup.law_burst` lifts `Setup.law_burst_alike` of
  `XdySpec/Explode.lean` to every explode of a grounded setup that the pass
  does not refuse, and of a mixture (`mixtureLaw_burst_whole`,
  `mixtureLaw_burst_totals`). Every burst leaves a well-formed setup with
  faces, grounded if it keeps its chains whole and leaves no stacks
  (`Setup.burst_invariants`).
- The proofs ask that the roll be well formed, that its chains be normal, and
  its results lie within the `i32` range (`Roll.Normal`), and that stacks hold
  no law of what their chains have become (`Roll.Joined`), which the deferred
  laws of stacks are left to later proofs to lift; the explodes ask that the
  setup be grounded (`Setup.Grounded`), and so hold no stacks.

```mermaid
flowchart TD
    S[grounded setup] --> A{alike, none dropped?}
    A -->|yes| L[Setup.law_burst_alike: each chain's law]
    A -->|no| P[ρ.stacks: multisets of chains]
    P -->|kept_pool, slice| K[kept pool]
    K -->|whole| T[stacks: each chain's explode deferred]
    P -->|totals: transform_pool| Q[pool of the totals of the kept chains]
```
-/
namespace Xdy.Spec

open Xdy (Die Chain)
open scoped ENNReal

/-! ## Weighted values -/

/--
Weighted values, each sent to another, are the values that they are sent to,
with the same weights, even if no weight is positive, so long as the default
is sent to the default.

# Parameters
- `d`: The value if no weight is positive.
- `l`: The values, each with its weight.
- `f`: Where each value is sent.
-/
theorem weighted_map_of {α β : Type} [DecidableEq β] (d : α)
    (l : List (α × Nat)) (f : α → β) :
    weighted (f d) (l.map fun e => (f e.1, e.2)) = (weighted d l).map f := by
  by_cases hl : (l.map (·.2)).sum = 0
  · have hz : ∀ {γ : Type} (m : List (γ × Nat)), (m.map (·.2)).sum = 0 →
        (m.flatMap fun p => List.replicate p.2 p.1) = [] := fun m hm => by
      refine List.flatMap_eq_nil_iff.mpr fun p hp => ?_
      have := List.sum_eq_zero_iff.mp hm p.2 (List.mem_map_of_mem hp)
      simp [this]
    have hl' : ((l.map fun e => (f e.1, e.2)).map (·.2)).sum = 0 := by
      simpa [Function.comp_def] using hl
    unfold weighted
    simp only [hz l hl, hz _ hl', ↓reduceDIte, PMF.pure_map]
  · exact weighted_map d l f hl

/--
Weighted values whose weights all scale alike, by a positive factor, are the
values with their own weights.

# Parameters
- `d`: The value if no weight is positive.
- `l`: The values, each with its weight.
- `f`: The factor by which each value's weight scales.

# Hypotheses
- `hk`: The common factor is positive.
- `hf`: Every value's weight scales by the common factor.
-/
theorem weighted_scale {α : Type} [DecidableEq α] (d : α)
    (l : List (α × Nat)) (f : α → Nat) {k : Nat} (hk : k ≠ 0)
    (hf : ∀ e ∈ l, f e.1 = k) :
    weighted d (l.map fun e => (e.1, e.2 * f e.1)) = weighted d l := by
  by_cases hl : (l.map (·.2)).sum = 0
  · have hz : ∀ {γ : Type} (m : List (γ × Nat)), (m.map (·.2)).sum = 0 →
        (m.flatMap fun p => List.replicate p.2 p.1) = [] := fun m hm => by
      refine List.flatMap_eq_nil_iff.mpr fun p hp => ?_
      have := List.sum_eq_zero_iff.mp hm p.2 (List.mem_map_of_mem hp)
      simp [this]
    have hl' : ((l.map fun e => (e.1, e.2 * f e.1)).map (·.2)).sum = 0 := by
      refine List.sum_eq_zero_iff.mpr fun x hx => ?_
      simp only [List.map_map, Function.comp_def, List.mem_map] at hx
      obtain ⟨e, he, rfl⟩ := hx
      rw [List.sum_eq_zero_iff.mp hl e.2 (List.mem_map_of_mem he), zero_mul]
    unfold weighted
    simp only [hz l hl, hz _ hl', ↓reduceDIte]
  · have h := weighted_flatMap d d (fun _ => ()) l (fun _ => [((), 1)]) f
      (fun a _ => a) k hk hl fun e he _ => by simp [hf e he]
    simp only [List.map_cons, List.map_nil, mul_one] at h
    rw [← List.map_eq_flatMap] at h
    have hc : ∀ a : α, (weighted () [((), 1)]).map (fun _ => a) = PMF.pure a :=
      fun a => PMF.map_const _ a
    rw [h]
    simp only [hc, PMF.bind_pure]

/-! ## Multisets of any faces -/

/--
Every face of every multiset of the program is a face of the die.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The faces, each with its weight.
- `J`: The number of draws.
- `e`: A multiset with its weight.

# Hypotheses
- `e` is a multiset of the state of `J` draws.
-/
theorem mem_multisets {α : Type} :
    ∀ (die : List (α × Nat)) (J : Nat) (e : List (α × Nat) × Nat),
      e ∈ multisets die J → ∀ p ∈ e.1, ∃ q ∈ die, q.1 = p.1 := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro J e he p hp
    simp only [multisets, Outcomes.outcomeStates, List.foldl_nil,
      Outcomes.initial] at he
    split at he
    · simp_all
    · simp at he
  | append_singleton die face ih =>
    intro J e he p hp
    simp only [multisets, Outcomes.outcomeStates_append, Outcomes.outcomeStep,
      List.mem_flatMap, List.mem_range, List.mem_map] at he
    obtain ⟨m, _, e', he', rfl⟩ := he
    unfold Outcomes.grow at hp
    split at hp
    · obtain ⟨q, hq, hqp⟩ := ih _ e' he' p hp
      exact ⟨q, by simp [hq], hqp⟩
    · rcases List.mem_append.mp hp with hp | hp
      · obtain ⟨q, hq, hqp⟩ := ih _ e' he' p hp
        exact ⟨q, by simp [hq], hqp⟩
      · simp only [List.mem_singleton] at hp
        subst hp
        exact ⟨face, by simp, rfl⟩

/--
The multisets of `n` independent draws of a die weigh its total weight raised
to `n` in all.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The faces, each with its weight.
- `n`: The number of draws.
-/
theorem sum_multisets {α : Type} (die : List (α × Nat)) (n : Nat) :
    ((multisets die n).map (·.2)).sum = (die.map (·.2)).sum ^ n := by
  have := Outcomes.integral_outcomeStates_bag die n fun _ => 1
  rw [Outcomes.bagWords_one] at this
  simpa [multisets, OrderStatistics.integral] using this

/--
Weighted values of positive total weight take only values that they list with
a positive weight.

# Parameters
- `x`: A value that they take.

# Hypotheses
- `hl`: The total weight is positive.
- `h`: They take `x`.
-/
theorem exists_pos_of_mem_support_weighted {α : Type} {d : α}
    {l : List (α × Nat)} {x : α} (hl : (l.map (·.2)).sum ≠ 0)
    (h : x ∈ (weighted d l).support) : ∃ e ∈ l, e.1 = x ∧ e.2 ≠ 0 := by
  unfold weighted at h
  split at h
  · rename_i he
    refine absurd (List.sum_eq_zero fun n hn => ?_) hl
    obtain ⟨e, he', rfl⟩ := List.mem_map.mp hn
    simpa using List.flatMap_eq_nil_iff.mp he e he'
  · simp only [PMF.mem_support_ofMultiset_iff, Multiset.mem_toFinset,
      Multiset.mem_coe, List.mem_flatMap] at h
    obtain ⟨e, he, hx⟩ := h
    exact ⟨e, he, (List.eq_of_mem_replicate hx).symm,
      fun h0 => by simp [h0] at hx⟩

/--
Independent draws of the law of each copy of one value are independent draws.

# Parameters
- `step`: The law of what each value becomes.
- `n`: The number of copies.
- `x`: The value.
-/
theorem bagDraws_replicate {α β : Type} (step : α → PMF β) (n : Nat) (x : α) :
    bagDraws step (List.replicate n x) = iidBag n (step x) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.replicate_succ, bagDraws, ih, Outcomes.iidBag_succ]
    rfl

/-- The sum of a list of lists is the sum of their sums. -/
private theorem sum_flatMap_nat {α : Type} (l : List α) (f : α → List Nat) :
    (l.flatMap f).sum = (l.map fun a => (f a).sum).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]

/-- The greatest of values, folded from any start, is at least each. -/
private theorem le_foldl_max :
    ∀ (l : List Nat) (a : Nat), ∀ x ∈ l, x ≤ l.foldl Max.max a
  | [], _, x, hx => by cases hx
  | y :: l, a, x, hx => by
    rw [List.foldl_cons]
    rcases List.mem_cons.mp hx with rfl | hx
    · have : ∀ (l : List Nat) (b : Nat), b ≤ l.foldl Max.max b := by
        intro l
        induction l with
        | nil => exact fun b => le_refl b
        | cons z l ih => exact fun b => (le_max_left b z).trans (ih _)
      exact (le_max_right a x).trans (this l _)
    · exact le_foldl_max l _ x hx

/-! ## Windows of chains -/

/--
The distinct chains of a list, each with its count, repeat to the list, in
some order.

# Parameters
- `cs`: The chains.
-/
theorem expand_group_perm (cs : List Chain) :
    (Roll.expand (group cs)).Perm cs := by
  refine List.perm_iff_count.mpr fun a => ?_
  rw [count_expand]
  simp only [group, List.filter_map, Function.comp_def, List.map_map]
  rw [List.filter_eq, List.count_dedup]
  split
  · simp
  · rename_i ha
    simp [List.count_eq_zero_of_not_mem ha]

/--
The window of chains sorted by `Xdy.Chain.le`, after the lowest `L` and the
highest `H`, depends only on their multiset, if they are normal, and is the
slice of any multiset of them (`slice`).

# Parameters
- `cs`: The chains, sorted.
- `ms`: Their multiset, each distinct chain with its copies.
- `L`, `H`: The numbers of lowest and highest chains dropped.

# Hypotheses
- `hs`: The chains are sorted by `Xdy.Chain.le`.
- `hn`: Every chain is normal.
- `hm`: The chains are those of the multiset.
-/
theorem coe_window_slice {cs : List Chain} {ms : List (Chain × Nat)}
    (L H : Nat) (hs : cs.Pairwise (fun a b => a.le b = true))
    (hn : ∀ c ∈ cs, NormalChain c) (hm : (cs : Multiset Chain) = Roll.expand ms) :
    (((cs.drop L).take (cs.length - L - H) : List Chain) : Multiset Chain) =
      Roll.expand (slice ms L H) := by
  have hp : cs.Perm (Roll.expand ms) := Multiset.coe_eq_coe.mp hm
  have heq : cs = (Roll.expand ms).mergeSort Chain.le := by
    rw [← List.mergeSort_of_pairwise hs]
    exact mergeSort_chains_perm hp hn
  refine Multiset.coe_eq_coe.mpr ?_
  simp only [slice]
  rw [← heq]
  exact (expand_group_perm _).symm

/-! ## Transforming pools -/

/-- Join, group by group, each multiset of independent draws of the law of
what each copy of the chain of each group becomes, as `transformPool` joins
them, from the multisets so far. -/
private def joinGroups {T : Type} (becomes : Chain → List (T × Nat) × Nat)
    (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)) :
    List (List (T × Nat) × Nat) :=
  groups.foldl
    (fun soFar g => soFar.flatMap fun e =>
      (multisets (becomes g.1).1 g.2).map fun o => (e.1 ++ o.1, e.2 * o.2))
    init

/-- Joining one more group joins it first. -/
private theorem joinGroups_cons {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) (g : Chain × Nat)
    (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)) :
    joinGroups becomes (g :: groups) init =
      joinGroups becomes groups (init.flatMap fun e =>
        (multisets (becomes g.1).1 g.2).map fun o => (e.1 ++ o.1, e.2 * o.2)) :=
  rfl

/-- Joining groups commutes with scaling the weights of the multisets so far. -/
private theorem joinGroups_scale {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) (w : Nat) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)),
      joinGroups becomes groups (init.map fun e => (e.1, w * e.2)) =
        (joinGroups becomes groups init).map fun e => (e.1, w * e.2)
  | [], _ => rfl
  | g :: groups, init => by
    rw [joinGroups_cons, joinGroups_cons, ← joinGroups_scale becomes w groups]
    congr 1
    simp only [List.flatMap_map, List.map_flatMap, List.map_map]
    refine List.flatMap_congr fun e _ => List.map_congr_left fun o _ => ?_
    simp [Nat.mul_assoc]

/-- The multisets of joined groups weigh those so far times the weights of
the laws of the groups, each raised to its copies, in all. -/
private theorem sum_joinGroups {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) (Q padding : Nat) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)),
      (∀ g ∈ groups,
        ((becomes g.1).1.map (·.2)).sum = Q * padding ^ (becomes g.1).2) →
      ((joinGroups becomes groups init).map (·.2)).sum =
        (init.map (·.2)).sum * Q ^ (groups.map (·.2)).sum *
          padding ^ (groups.map fun g => (becomes g.1).2 * g.2).sum
  | [], init, _ => by simp [joinGroups]
  | g :: groups, init, h => by
    rw [joinGroups_cons, sum_joinGroups becomes Q padding groups _
      fun g' hg' => h g' (List.mem_cons_of_mem _ hg')]
    have hstep : ((init.flatMap fun e =>
        (multisets (becomes g.1).1 g.2).map fun o =>
          (e.1 ++ o.1, e.2 * o.2)).map (·.2)).sum =
        (init.map (·.2)).sum * (Q * padding ^ (becomes g.1).2) ^ g.2 := by
      rw [← h g List.mem_cons_self, ← sum_multisets, List.map_flatMap,
        sum_flatMap_nat, ← List.sum_map_mul_right]
      refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
      simp only [List.map_map, Function.comp_def]
      exact List.sum_map_mul_left _ _ _
    rw [hstep]
    simp only [List.map_cons, List.sum_cons, mul_pow, ← pow_mul, pow_add]
    ring

/-- The multisets of joined groups, over their weights, are the multisets so
far, each joined with independent draws of the law of each chain of the
groups, one for each copy. -/
private theorem weighted_joinGroups {T : Type} [DecidableEq T]
    (becomes : Chain → List (T × Nat) × Nat) (dc : Chain → T) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat))
      (d d₀ : List (T × Nat)),
      (init.map (·.2)).sum ≠ 0 →
      (∀ g ∈ groups, ((becomes g.1).1.map (·.2)).sum ≠ 0) →
      (weighted d (joinGroups becomes groups init)).map
          (fun ms => (Roll.expand ms : Multiset T)) =
        (weighted d₀ init).bind fun a =>
          (bagDraws (fun ch => weighted (dc ch) (becomes ch).1)
            (Roll.expand groups)).map ((Roll.expand a : Multiset T) + ·)
  | [], init, d, d₀, hi, _ => by
    simp only [joinGroups, List.foldl_nil, Roll.expand, List.flatMap_nil,
      bagDraws, PMF.pure_map, add_zero]
    rw [weighted_congr_default d d₀ hi]
    exact (PMF.bind_pure_comp _ _).symm
  | g :: groups, init, d, d₀, hi, hg => by
    have hW := hg g List.mem_cons_self
    let M := multisets (becomes g.1).1 g.2
    have hstep := weighted_flatMap ([] : List (T × Nat)) d₀ (fun _ => [])
      init (fun _ => M) (fun _ => 1) (fun a o => a ++ o)
      (((becomes g.1).1.map (·.2)).sum ^ g.2) (pow_ne_zero _ hW) hi
      (fun e _ _ => by rw [one_mul, sum_multisets])
    simp only [mul_one] at hstep
    have hi' : ((init.flatMap fun e =>
        M.map fun o => (e.1 ++ o.1, e.2 * o.2)).map (·.2)).sum ≠ 0 := by
      rw [List.map_flatMap, sum_flatMap_nat]
      intro h0
      obtain ⟨e, he, he0⟩ : ∃ e ∈ init, e.2 ≠ 0 := by
        by_contra hc
        push Not at hc
        exact hi (List.sum_eq_zero fun n hn => by
          obtain ⟨e, he, rfl⟩ := List.mem_map.mp hn
          exact hc e he)
      have := List.sum_eq_zero_iff.mp h0 _ (List.mem_map_of_mem he)
      simp only [List.map_map, Function.comp_def] at this
      rw [List.sum_map_mul_left, sum_multisets] at this
      exact mul_ne_zero he0 (pow_ne_zero _ hW) this
    have ih := weighted_joinGroups becomes dc groups _ d [] hi'
      fun g' hg' => hg g' (List.mem_cons_of_mem _ hg')
    rw [joinGroups_cons, ih, hstep, PMF.bind_bind]
    refine congrArg _ (funext fun a => ?_)
    have hM := Outcomes.weighted_outcomeStates (dc g.1) [] (becomes g.1).1 hW
      g.2
    simp only [Roll.expand, List.flatMap_cons, bagDraws_append,
      bagDraws_replicate] at hM ⊢
    rw [← hM]
    simp only [lift, PMF.bind_map, PMF.map_bind, PMF.map_comp,
      Function.comp_def, multisets, M]
    refine congrArg _ (funext fun o => congrArg (PMF.map · _)
      (funext fun N => ?_))
    simp only [List.flatMap_append, ← Multiset.coe_add, add_assoc]

/-- The greatest number of kept dice of the chains still kept of any multiset
of a pool, by which `transformPool` pads each multiset. -/
private def mostDice {T : Type} (outcomes : List (List (Chain × Nat) × Nat))
    (lowest highest : Nat) (becomes : Chain → List (T × Nat) × Nat) : Nat :=
  (outcomes.map fun e =>
    ((slice e.1 lowest highest).map fun g => (becomes g.1).2 * g.2).sum).foldl
      Max.max 0

/-- A transformed pool joins, for each multiset, the groups of its chains
still kept, padded. -/
private theorem transformPool_eq {T : Type}
    (outcomes : List (List (Chain × Nat) × Nat)) (lowest highest padding : Nat)
    (becomes : Chain → List (T × Nat) × Nat) :
    transformPool outcomes lowest highest padding becomes =
      outcomes.flatMap fun e =>
        (joinGroups becomes (slice e.1 lowest highest) [([], 1)]).map fun o =>
          (o.1, e.2 * padding ^ (mostDice outcomes lowest highest becomes -
            ((slice e.1 lowest highest).map fun g =>
              (becomes g.1).2 * g.2).sum) * o.2) := by
  simp only [transformPool, List.flatMap_map, List.map_map, Function.comp_def,
    mostDice]
  refine List.flatMap_congr fun e _ => ?_
  have h := joinGroups_scale becomes
    (e.2 * padding ^ ((outcomes.map fun e =>
      ((slice e.1 lowest highest).map fun g =>
        (becomes g.1).2 * g.2).sum).foldl Max.max 0 -
      ((slice e.1 lowest highest).map fun g => (becomes g.1).2 * g.2).sum))
    (slice e.1 lowest highest) [([], 1)]
  simp only [List.map_cons, List.map_nil, mul_one] at h
  exact h

/--
A transformed pool, over its weights, is each multiset of the pool, over its
weights, after its lowest `lowest` and its highest `highest` chains are
dropped, each copy of each chain still kept drawn independently from the law
of what it becomes, as `transform_pool` in `propagation/chain.rs` transforms
it: each multiset weighs alike in all, if each chain's law weighs `Q` times the
padding raised to its kept dice, and every multiset keeps as many chains.

# Type parameters
- `T`: The type of what a chain becomes: a chain, or a total.

# Parameters
- `d`, `d₀`: The multisets if no weight were positive.
- `dc`: What a chain becomes if no weight of its law were positive.
- `outcomes`: The multisets of the chains, each with its branches.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
- `padding`: The branches that a reroll adds to each kept die, or one for an
  explode.
- `becomes`: The law of what a chain becomes, and its kept dice.
- `Q`: The weight of each law besides its padding.
- `k`: The number of chains still kept of each multiset.

# Hypotheses
- `hQ`, `hp`: Neither `Q` nor the padding is zero.
- `hl`: The pool weighs something.
- `hlaw`: The law of each chain still kept of each multiset of positive weight
  weighs `Q` times the padding raised to its kept dice.
- `hk`: Every multiset of positive weight keeps `k` chains.
-/
theorem weighted_transformPool {T : Type} [DecidableEq T]
    (d : List (T × Nat)) (d₀ : List (Chain × Nat)) (dc : Chain → T)
    (outcomes : List (List (Chain × Nat) × Nat)) (lowest highest padding : Nat)
    (becomes : Chain → List (T × Nat) × Nat) (Q k : Nat) (hQ : Q ≠ 0)
    (hp : padding ≠ 0) (hl : (outcomes.map (·.2)).sum ≠ 0)
    (hlaw : ∀ e ∈ outcomes, e.2 ≠ 0 → ∀ g ∈ slice e.1 lowest highest,
      ((becomes g.1).1.map (·.2)).sum = Q * padding ^ (becomes g.1).2)
    (hk : ∀ e ∈ outcomes, e.2 ≠ 0 →
      ((slice e.1 lowest highest).map (·.2)).sum = k) :
    (weighted d (transformPool outcomes lowest highest padding becomes)).map
        (fun ms => (Roll.expand ms : Multiset T)) =
      (weighted d₀ outcomes).bind fun ms =>
        bagDraws (fun ch => weighted (dc ch) (becomes ch).1)
          (Roll.expand (slice ms lowest highest)) := by
  set most := mostDice outcomes lowest highest becomes
  let D : List (Chain × Nat) → Nat := fun ms =>
    ((slice ms lowest highest).map fun g => (becomes g.1).2 * g.2).sum
  have hD : ∀ e ∈ outcomes, D e.1 ≤ most := fun e he =>
    le_foldl_max _ 0 _ (List.mem_map_of_mem (f := fun e : List (Chain × Nat) ×
      Nat => D e.1) he)
  have h := weighted_flatMap d d₀ (fun _ => [])
    outcomes (fun ms => joinGroups becomes (slice ms lowest highest) [([], 1)])
    (fun ms => padding ^ (most - D ms)) (fun _ o => o) (Q ^ k * padding ^ most)
    (mul_ne_zero (pow_ne_zero _ hQ) (pow_ne_zero _ hp)) hl
    (fun e he he0 => by
      rw [sum_joinGroups becomes Q padding _ _ (hlaw e he he0), hk e he he0]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
        add_zero, one_mul]
      rw [mul_comm, mul_assoc, ← pow_add, Nat.add_sub_cancel' (hD e he)])
  rw [transformPool_eq, h, PMF.map_bind]
  refine bind_congr_support fun ms hms => ?_
  obtain ⟨e, he, rfl, he0⟩ := exists_pos_of_mem_support_weighted hl hms
  rw [show (fun o : List (T × Nat) => o) = id from rfl, PMF.map_id,
    weighted_joinGroups becomes dc _ _ _ [] (by simp)
      fun g hg => by
        rw [hlaw e he he0 g hg]
        exact mul_ne_zero hQ (pow_ne_zero _ hp)]
  simp only [weighted_single [] [] one_ne_zero, PMF.pure_bind, Roll.expand,
    List.flatMap_nil, Multiset.coe_nil, zero_add]
  exact PMF.map_id _

/--
Every multiset of `n` independent draws of a die holds `n` faces, and, if
every face weighs something, weighs something.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The faces, each with its weight.
- `n`: The number of draws.
- `e`: A multiset with its weight.

# Hypotheses
- `he`: `e` is a multiset of the draws.
-/
theorem multisets_props {α : Type} :
    ∀ (die : List (α × Nat)) (n : Nat) (e : List (α × Nat) × Nat),
      e ∈ multisets die n → (e.1.map (·.2)).sum = n ∧
        ((∀ f ∈ die, 0 < f.2) → 0 < e.2) := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro n e he
    simp only [multisets, Outcomes.outcomeStates, List.foldl_nil,
      Outcomes.initial] at he
    split at he
    · simp_all
    · simp at he
  | append_singleton die face ih =>
    intro n e he
    simp only [multisets, Outcomes.outcomeStates_append, Outcomes.outcomeStep,
      List.mem_flatMap, List.mem_range, List.mem_map] at he
    obtain ⟨m, hm, e', he', rfl⟩ := he
    obtain ⟨hc, hp⟩ := ih _ e' he'
    refine ⟨?_, fun hpos => Nat.mul_pos (Nat.mul_pos
      (Nat.choose_pos (by omega)) (Nat.pow_pos (hpos face (by simp))))
      (hp fun f hf => hpos f (by simp [hf]))⟩
    have hg : ((Outcomes.grow e'.1 face.1 m).map (·.2)).sum =
        (e'.1.map (·.2)).sum + m := by
      unfold Outcomes.grow
      split <;> simp_all
    rw [hg, hc]
    omega

/-- Every multiset of joined groups holds the faces of the multisets so far
and the copies of the groups, and weighs something if they do. -/
private theorem joinGroups_props {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) (a : Nat) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)),
      (∀ e ∈ init, (e.1.map (·.2)).sum = a ∧ 0 < e.2) →
      (∀ g ∈ groups, ∀ o ∈ (becomes g.1).1, 0 < o.2) →
      ∀ o ∈ joinGroups becomes groups init,
        (o.1.map (·.2)).sum = a + (groups.map (·.2)).sum ∧ 0 < o.2
  | [], init, hi, _, o, ho => by simpa using hi o ho
  | g :: groups, init, hi, hg, o, ho => by
    rw [joinGroups_cons] at ho
    have h := joinGroups_props becomes (a + g.2) groups _ (fun e he => by
        simp only [List.mem_flatMap, List.mem_map] at he
        obtain ⟨e', he', o', ho', rfl⟩ := he
        obtain ⟨hc, hp⟩ := multisets_props _ _ _ ho'
        obtain ⟨hc', hp'⟩ := hi e' he'
        refine ⟨by simp [hc, hc'], Nat.mul_pos hp' (hp (hg g
          List.mem_cons_self))⟩)
      (fun g' hg' => hg g' (List.mem_cons_of_mem _ hg')) o ho
    simp only [List.map_cons, List.sum_cons] at h ⊢
    rw [h.1]
    exact ⟨by omega, h.2⟩

/-- Every multiset of a transformed pool joins the groups of the chains still
kept of some multiset of the pool, padded. -/
private theorem mem_transformPool {T : Type}
    {outcomes : List (List (Chain × Nat) × Nat)} {lowest highest padding : Nat}
    {becomes : Chain → List (T × Nat) × Nat} {o : List (T × Nat) × Nat}
    (ho : o ∈ transformPool outcomes lowest highest padding becomes) :
    ∃ e ∈ outcomes, ∃ o' ∈ joinGroups becomes (slice e.1 lowest highest)
      [([], 1)], o.1 = o'.1 ∧ ∃ m, o.2 = e.2 * padding ^ m * o'.2 := by
  rw [transformPool_eq] at ho
  simp only [List.mem_flatMap, List.mem_map] at ho
  obtain ⟨e, he, o', ho', rfl⟩ := ho
  exact ⟨e, he, o', ho', rfl, _, rfl⟩

/--
A transformed pool weighs as much as the pool times what each multiset weighs
after it, under the hypotheses of `weighted_transformPool`.

# Parameters
- `outcomes`: The multisets of the chains, each with its branches.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
- `padding`: The branches that a reroll adds to each kept die, or one for an
  explode.
- `becomes`: The law of what a chain becomes, and its kept dice.
- `Q`: The weight of each law besides its padding.
- `k`: The number of chains still kept of each multiset.

# Hypotheses
- `hQ`, `hp`: Neither `Q` nor the padding is zero.
- `hl`: The pool weighs something.
- `hlaw`: The law of each chain still kept of each multiset of positive weight
  weighs `Q` times the padding raised to its kept dice.
- `hk`: Every multiset of positive weight keeps `k` chains.
-/
theorem sum_transformPool_ne_zero {T : Type}
    (outcomes : List (List (Chain × Nat) × Nat)) (lowest highest padding : Nat)
    (becomes : Chain → List (T × Nat) × Nat) (Q k : Nat) (hQ : Q ≠ 0)
    (hp : padding ≠ 0) (hl : (outcomes.map (·.2)).sum ≠ 0)
    (hlaw : ∀ e ∈ outcomes, e.2 ≠ 0 → ∀ g ∈ slice e.1 lowest highest,
      ((becomes g.1).1.map (·.2)).sum = Q * padding ^ (becomes g.1).2)
    (hk : ∀ e ∈ outcomes, e.2 ≠ 0 →
      ((slice e.1 lowest highest).map (·.2)).sum = k) :
    ((transformPool outcomes lowest highest padding becomes).map (·.2)).sum ≠
      0 := by
  obtain ⟨e, he, he0⟩ : ∃ e ∈ outcomes, e.2 ≠ 0 := by
    by_contra hc
    push Not at hc
    exact hl (List.sum_eq_zero fun n hn => by
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hn
      exact hc e he)
  have hJ := sum_joinGroups becomes Q padding (slice e.1 lowest highest)
    [([], 1)] (hlaw e he he0)
  rw [hk e he he0] at hJ
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero, one_mul] at hJ
  obtain ⟨o, ho, ho0⟩ : ∃ o ∈ joinGroups becomes (slice e.1 lowest highest)
      [([], 1)], o.2 ≠ 0 := by
    by_contra hc
    push Not at hc
    refine mul_ne_zero (pow_ne_zero _ hQ) (pow_ne_zero _ hp)
      (hJ ▸ List.sum_eq_zero fun n hn => ?_)
    obtain ⟨o, ho, rfl⟩ := List.mem_map.mp hn
    exact hc o ho
  rw [transformPool_eq]
  intro h0
  have hmem : (o.1, e.2 * padding ^ (mostDice outcomes lowest highest becomes -
      ((slice e.1 lowest highest).map fun g => (becomes g.1).2 * g.2).sum) *
        o.2) ∈ outcomes.flatMap fun e =>
      (joinGroups becomes (slice e.1 lowest highest) [([], 1)]).map fun o =>
        (o.1, e.2 * padding ^ (mostDice outcomes lowest highest becomes -
          ((slice e.1 lowest highest).map fun g =>
            (becomes g.1).2 * g.2).sum) * o.2) :=
    List.mem_flatMap.mpr ⟨e, he, List.mem_map.mpr ⟨o, ho, rfl⟩⟩
  have := List.sum_eq_zero_iff.mp h0 _ (List.mem_map_of_mem hmem)
  exact mul_ne_zero (mul_ne_zero he0 (pow_ne_zero _ hp)) ho0 this

/-- Values repeated by their copies number as many as their copies. -/
private theorem length_expand {α : Type} (ms : List (α × Nat)) :
    (Roll.expand ms).length = (ms.map (·.2)).sum := by
  induction ms with
  | nil => rfl
  | cons p ms ih =>
    simp only [Roll.expand, List.flatMap_cons, List.length_append,
      List.length_replicate, List.map_cons, List.sum_cons] at ih ⊢
    omega

/--
The chains still kept of a multiset of chains number as many as its chains,
less those dropped.

# Parameters
- `ms`: The multiset, each distinct chain with its copies.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
-/
theorem count_slice (ms : List (Chain × Nat)) (lowest highest : Nat) :
    ((slice ms lowest highest).map (·.2)).sum =
      (ms.map (·.2)).sum - lowest - highest := by
  rw [← length_expand (slice ms lowest highest), ← length_expand ms]
  have hg : ∀ l, (Roll.expand (group l)).length = l.length := fun l =>
    (expand_group_perm l).length_eq
  simp only [slice]
  rw [hg]
  simp only [List.length_take, List.length_drop, List.length_mergeSort]
  omega

/--
Every chain still kept of a multiset of chains, after its drops, is one of its
chains, and each holds copies.

# Parameters
- `ms`: The multiset, each distinct chain with its copies.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
- `g`: A chain still kept, with its copies.

# Hypotheses
- `hg`: `g` is kept.
-/
theorem mem_slice {ms : List (Chain × Nat)} {lowest highest : Nat}
    {g : Chain × Nat} (hg : g ∈ slice ms lowest highest) :
    g.1 ∈ Roll.expand ms ∧ 0 < g.2 := by
  simp only [slice, group, List.mem_map, List.mem_dedup] at hg
  obtain ⟨x, hx, rfl⟩ := hg
  have hx' := (List.mergeSort_perm _ _).mem_iff.mp
    (List.mem_of_mem_drop (List.mem_of_mem_take hx))
  exact ⟨hx', List.count_pos_iff.mpr hx⟩

/-- Draws of what each value becomes, each mapped, are draws of the mapped
laws. -/
private theorem bagDraws_map_map {α β γ : Type} (step : α → PMF β)
    (g : β → γ) :
    ∀ xs : List α, (bagDraws step xs).map (Multiset.map g) =
      bagDraws (fun x => (step x).map g) xs
  | [] => by simp [bagDraws, PMF.pure_map]
  | x :: xs => by
    simp only [bagDraws, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      ← bagDraws_map_map step g xs, Function.comp_def, Multiset.map_cons]

/-- The value that a list of keys, each with its value, gives a key among
them. -/
private theorem lookup_graph {β : Type} (f : Chain → β) :
    ∀ (l : List Chain) (x : Chain), x ∈ l →
      (l.map fun y => (y, f y)).lookup x = some (f x)
  | [], _, h => by cases h
  | y :: l, x, h => by
    by_cases hxy : x = y
    · subst hxy
      simp
    · have hx : x ∈ l := by simpa [hxy] using h
      have hb : (x == y) = false := by simpa using hxy
      simp only [List.map_cons, List.lookup, hb]
      exact lookup_graph f l x hx

/--
A multiset of stacks, each of whose chains has become what a law of the same
total weight gives it, gains that weight for each of its chains.

# Parameters
- `laws`: The law of what each distinct chain has become.
- `ms`: The multiset, each distinct chain with its copies.
- `w`: The common total weight.

# Hypotheses
- `h`: The law of what every chain of the multiset has become weighs `w`.
-/
theorem stackWeight_of_const {laws : List (Chain × List (Chain × Nat))}
    {w : Nat} : ∀ {ms : List (Chain × Nat)},
      (∀ p ∈ ms, (((laws.lookup p.1).map fun l => (l.map (·.2)).sum).getD 1) =
        w) →
      stackWeight laws ms = w ^ (ms.map (·.2)).sum
  | [], _ => rfl
  | p :: ms, h => by
    have ih := stackWeight_of_const (ms := ms) fun q hq => h q (by simp [hq])
    simp only [stackWeight, List.map_cons, List.prod_cons, List.sum_cons] at ih ⊢
    rw [ih, h p List.mem_cons_self, pow_add]

/--
The kept pool of multisets, over its weights, is the slice of each multiset.

# Parameters
- `outcomes`: The multisets of the chains, each with its branches.
- `lowest`, `highest`: The numbers of lowest and highest chains dropped.
-/
theorem weighted_keptPool (outcomes : List (List (Chain × Nat) × Nat))
    (lowest highest : Nat) :
    weighted [] (keptPool outcomes lowest highest) =
      (weighted [] outcomes).map (fun ms => slice ms lowest highest) := by
  rw [← weighted_map_of]
  exact congrArg₂ weighted (by simp [slice, group, Roll.expand]) rfl

/--
The chains of one multiset of stacks, as a multiset, are independent draws of
what each copy of each of its chains has become.

# Parameters
- `laws`: The law of what each distinct chain has become.
- `ms`: The multiset, each distinct chain with its copies.
-/
theorem map_coe_stackDraws (laws : List (Chain × List (Chain × Nat))) :
    ∀ ms : List (Chain × Nat),
      (stackDraws laws ms).map (Multiset.ofList : List Chain → Multiset Chain) =
        bagDraws (became laws) (Roll.expand ms)
  | [] => by simp [stackDraws, bagDraws, Roll.expand, PMF.pure_map]
  | (ch, k) :: ms => by
    simp only [stackDraws, Roll.expand, List.flatMap_cons]
    rw [bagDraws_append, bagDraws_replicate, ← Roll.expand,
      ← map_coe_stackDraws laws ms,
      ← map_coe_chainDraws, PMF.map_bind]
    simp only [lift, PMF.bind_map, PMF.map_comp, Function.comp_def,
      ← Multiset.coe_add]

/-! ## The chains of a roll -/

namespace Roll

/--
Whether the chains of a roll are normal, and its results lie within the `i32`
range, as every chain and result of `propagation/chain.rs` does: each chain
that i.i.d. chains may be, or that stacks may hold or have become, is normal;
and every result of every record that any other roll fills lies within the
`i32` range, so that its chain of one die is normal.

# Parameters
- `ρ`: The roll.

# Returns
Whether the roll is normal.
-/
def Normal : Roll → Prop
  | .chains _ law => ∀ e ∈ law, NormalChain e.1
  | .stacked ms laws => (∀ e ∈ ms, ∀ p ∈ e.1, NormalChain p.1) ∧
    ∀ e ∈ laws, ∀ p ∈ e.2, NormalChain p.1
  | ρ => ∀ r ∈ ρ.law.support, ∀ x ∈ r.results, i32Min ≤ x ∧ x ≤ i32Max

/--
Whether every chain of a roll's stacks is still itself, as `kept_pool` leaves
them; every other roll is joined. The deferred laws of stacks are left to
later proofs.

# Parameters
- `ρ`: The roll.

# Returns
Whether the roll is joined.
-/
def Joined : Roll → Prop
  | .stacked _ laws => laws = []
  | _ => True

/-- Draws of one chain each time are copies of it. -/
private theorem chainDraws_pure' (x : Chain) :
    ∀ k : Nat, chainDraws k (PMF.pure x) = PMF.pure (List.replicate k x)
  | 0 => rfl
  | k + 1 => by
    simp only [chainDraws, Nat.repeat] at chainDraws_pure' ⊢
    rw [chainDraws_pure' x k, PMF.pure_bind, PMF.pure_map]
    rfl

/--
The chains of a multiset of stacks that are still themselves are the
multiset's chains.

# Parameters
- `ms`: The multiset.
-/
theorem stackDraws_nil (ms : List (Chain × Nat)) :
    stackDraws [] ms = PMF.pure (Roll.expand ms) := by
  induction ms with
  | nil => rfl
  | cons e ms ih =>
    obtain ⟨ch, k⟩ := e
    simp only [stackDraws, became, List.lookup_nil, chainDraws_pure', ih,
      PMF.pure_bind, PMF.pure_map]
    rfl

/-- The kept chains of a record without chains or drops are the chains of one
die of its results, sorted. -/
private theorem keptChains_plain {r : Record} (hc : r.chains = [])
    (hl : r.lowest = 0) (hh : r.highest = 0) :
    r.keptChains = r.sorted.map .single := by
  have hv := Record.view_of_undropped hl hh
  simp only [Record.keptChains, hc, List.isEmpty_nil, ↓reduceIte]
  rw [show r.kept = r.sorted from (Prod.mk.inj hv).1]

/--
The kept chains of every record that a roll fills, as a multiset, have the law
of the roll's multisets of chains, over their weights, as `Roll::stacks`
weighs them: independent draws of the law of each of i.i.d. chains, the pool
of stacks whose chains are still themselves, or the outcomes of any other
roll, each result the chain of one die.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
- `hn`: The roll is normal.
- `hj`: The roll is joined.
-/
theorem map_keptChains_law {ρ : Roll} (hρ : ρ.WellFormed) (hn : ρ.Normal)
    (hj : ρ.Joined) :
    ρ.law.map (fun r => (r.keptChains : Multiset Chain)) =
      (weighted [] ρ.stacks).map
        (fun ms => (Roll.expand ms : Multiset Chain)) := by
  -- Any roll without chains: its outcomes, each result a chain of one die.
  have plain : ρ.Plain → ρ.law.map (fun r => (r.keptChains : Multiset Chain)) =
      (weighted [] ρ.stacks).map
        (fun ms => (Roll.expand ms : Multiset Chain)) := fun hp => by
    have hs : ρ.stacks = ρ.outcomes.map fun e =>
        (e.1.map fun p => (Chain.single p.1, p.2), e.2) := by
      cases ρ <;> first | rfl | exact hp.elim
    have hl : ρ.law.map (fun r => (r.keptChains : Multiset Chain)) =
        (ρ.law.map Record.sorted).map
          (fun ys => ((ys.map Chain.single : List Chain) : Multiset Chain)) := by
      rw [PMF.map_comp]
      refine map_congr_support fun r hr => ?_
      obtain ⟨hl, hh, _⟩ := mem_support_law_die hr
      simp only [Function.comp_apply,
        keptChains_plain (chains_of_mem_support_law hp hr) hl hh]
    rw [hl, map_sorted_law_weighted ρ hρ hp, hs,
      show ([] : List (Chain × Nat)) =
        (fun rs : List (Int × Nat) => rs.map fun p => (Chain.single p.1, p.2))
          [] from rfl,
      weighted_map_of, PMF.map_comp, PMF.map_comp]
    refine congrArg (PMF.map · _) (funext fun rs => ?_)
    simp only [Function.comp_apply, expand, List.flatMap_map, List.map_flatMap,
      List.map_replicate]
  cases ρ with
  | range => exact plain trivial
  | standard => exact plain trivial
  | custom => exact plain trivial
  | fixed => exact plain trivial
  | pool => exact plain trivial
  | chains count cl =>
    obtain ⟨hne, hpos⟩ := hρ
    have hW : (cl.map (·.2)).sum ≠ 0 := by
      obtain ⟨e, es, rfl⟩ := List.exists_cons_of_ne_nil hne
      have := hpos e (by simp)
      simp only [List.map_cons, List.sum_cons]
      omega
    simp only [law, Roll.stacks, multisets, PMF.map_comp]
    rw [Outcomes.weighted_outcomeStates (.single 0) [] cl hW,
      ← map_coe_chainDraws]
    refine map_congr_support fun cs hcs => ?_
    simp only [Function.comp_apply]
    refine keptChains_ofChains _ fun x hx => ?_
    rcases mem_support_weighted ((mem_support_chainDraws hcs).2 x hx) with
      ⟨e, he, rfl⟩ | rfl
    · exact hn e he
    · exact ⟨.nil, by simp [i32Min, i32Max]⟩
  | stacked ms laws =>
    have hl : laws = [] := hj
    subst hl
    obtain ⟨hne, hpos, _⟩ := hρ
    have hW : (ms.map (·.2)).sum ≠ 0 := by
      obtain ⟨e, es, rfl⟩ := List.exists_cons_of_ne_nil hne
      have := (hpos e (by simp)).2
      simp only [List.map_cons, List.sum_cons]
      omega
    simp only [law, Roll.stacks, joined, List.isEmpty_nil, ↓reduceIte,
      stackDraws_nil, PMF.pure_map, PMF.map_bind, stackWeight_nil, mul_one,
      List.map_id']
    rw [weighted_congr_default _ [] hW, ← PMF.bind_pure_comp]
    refine bind_congr_support fun m hm => ?_
    rw [Function.comp_apply]
    refine congrArg PMF.pure (keptChains_ofChains _ fun x hx => ?_)
    obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hx
    rw [List.eq_of_mem_replicate hpx]
    rcases mem_support_weighted hm with ⟨e, he, rfl⟩ | rfl
    · exact hn.1 e he p hp
    · simp at hp

/--
Every chain of every multiset of chains of a well-formed, normal, joined roll
is normal.

# Parameters
- `ρ`: The roll.
- `e`: A multiset with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `hn`: The roll is normal.
- `hj`: The roll is joined.
- `he`: `e` is a multiset of chains of the roll.
-/
theorem normal_of_mem_stacks {ρ : Roll} {e : List (Chain × Nat) × Nat}
    (hρ : ρ.WellFormed) (hn : ρ.Normal) (hj : ρ.Joined) (he : e ∈ ρ.stacks) :
    ∀ p ∈ e.1, NormalChain p.1 := by
  have plain : ρ.Plain → ρ.stacks = ρ.outcomes.map fun e =>
      (e.1.map fun p => (Chain.single p.1, p.2), e.2) := fun hp => by
    cases ρ <;> first | rfl | exact hp.elim
  have hplain : ρ.Plain → (∀ r ∈ ρ.law.support, ∀ x ∈ r.results,
      i32Min ≤ x ∧ x ≤ i32Max) → ∀ p ∈ e.1, NormalChain p.1 := fun hp hres => by
    rw [plain hp, List.mem_map] at he
    obtain ⟨o, ho, rfl⟩ := he
    intro p hp'
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hp'
    obtain ⟨r, hr, hsort⟩ := exists_sorted_of_mem_outcomes hρ ho
    have hq0 : 0 < q.2 := (wellFormed_of_mem_outcomes hρ ho).2 q hq
    have hx : q.1 ∈ r.results := by
      have : q.1 ∈ expand o.1 :=
        List.mem_flatMap.mpr ⟨q, hq, List.mem_replicate.mpr ⟨hq0.ne', rfl⟩⟩
      rw [← hsort] at this
      exact (List.mergeSort_perm _ _).mem_iff.mp this
    have := hres r hr q.1 hx
    exact ⟨.nil, fun y hy => by rw [List.mem_singleton.mp hy]; exact this⟩
  cases ρ with
  | range => exact hplain trivial hn
  | standard => exact hplain trivial hn
  | custom => exact hplain trivial hn
  | fixed => exact hplain trivial hn
  | pool => exact hplain trivial hn
  | chains count law =>
    intro p hp
    obtain ⟨q, hq, hqp⟩ := mem_multisets law count.toNat e he p hp
    exact hqp ▸ hn q hq
  | stacked ms laws =>
    have hl : laws = [] := hj
    subst hl
    simp only [Roll.stacks, joined, List.isEmpty_nil, ↓reduceIte] at he
    exact hn.1 e he

end Roll

/--
Every record that a roll fills holds no chains, or one result for each chain.

# Parameters
- `ρ`: The roll.
- `r`: The record.

# Hypotheses
- `hr`: The roll can fill `r`.
-/
theorem Roll.length_chains_of_mem_support_law {ρ : Roll} {r : Record}
    (hr : r ∈ ρ.law.support) :
    r.chains ≠ [] → r.chains.length = r.results.length := by
  have hof : ∀ cs : List Chain, (Record.ofChains .none cs).chains ≠ [] →
      (Record.ofChains .none cs).chains.length =
        (Record.ofChains .none cs).results.length := fun cs h => by
    simp only [Record.ofChains, Record.ofOracle, Xdy.Record.ofChains] at h ⊢
    split at h
    · exact absurd rfl h
    · rename_i hall
      simp only [hall, Bool.false_eq_true, ↓reduceIte, List.length_map]
  have plain : ρ.Plain → r.chains ≠ [] → r.chains.length = r.results.length :=
    fun hp h => absurd (Roll.chains_of_mem_support_law hp hr) h
  cases ρ with
  | chains count cl =>
    simp only [Roll.law, PMF.support_map, Set.mem_image] at hr
    obtain ⟨cs, _, rfl⟩ := hr
    exact hof cs
  | stacked ms laws =>
    simp only [Roll.law, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff] at hr
    obtain ⟨_, _, cs, _, rfl⟩ := hr
    exact hof cs
  | range => exact plain trivial
  | standard => exact plain trivial
  | custom => exact plain trivial
  | fixed => exact plain trivial
  | pool => exact plain trivial

/--
Every kept chain of every record that a well-formed, normal, joined roll
fills is normal.

# Parameters
- `ρ`: The roll.
- `r`: The record.

# Hypotheses
- `hw`: The roll is well formed.
- `hn`: The roll is normal.
- `hj`: The roll is joined.
- `hr`: The roll can fill `r`.
-/
theorem Roll.normal_of_mem_support_law {ρ : Roll} {r : Record}
    (hw : ρ.WellFormed) (hn : ρ.Normal) (hj : ρ.Joined)
    (hr : r ∈ ρ.law.support) : ∀ ch ∈ r.keptChains, NormalChain ch := by
  have hm : (r.keptChains : Multiset Chain) ∈
      (ρ.law.map fun r => (r.keptChains : Multiset Chain)).support :=
    (PMF.mem_support_map_iff _ _ _).mpr ⟨r, hr, rfl⟩
  rw [ρ.map_keptChains_law hw hn hj, PMF.mem_support_map_iff] at hm
  obtain ⟨ms, hms, hcoe⟩ := hm
  intro ch hch
  have hch' : ch ∈ Roll.expand ms := by
    rw [← Multiset.mem_coe, hcoe]
    exact hch
  obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hch'
  rw [List.eq_of_mem_replicate hpx]
  rcases mem_support_weighted hms with ⟨e, he, rfl⟩ | rfl
  · exact ρ.normal_of_mem_stacks hw hn hj he p hp
  · simp at hp

/--
Every multiset of chains of a well-formed roll but stacks holds as many chains
as the roll's records hold results, and weighs something.

# Parameters
- `ρ`: The roll.
- `e`: A multiset with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `hns`: The roll is not stacks.
- `he`: `e` is a multiset of chains of the roll.
-/
theorem Roll.stacks_props {ρ : Roll} {e : List (Chain × Nat) × Nat}
    (hρ : ρ.WellFormed) (hns : ∀ ms ls, ρ ≠ .stacked ms ls)
    (he : e ∈ ρ.stacks) : (e.1.map (·.2)).sum = ρ.results ∧ 0 < e.2 := by
  have plain : ρ.Plain → ρ.stacks = ρ.outcomes.map fun e =>
      (e.1.map fun p => (Chain.single p.1, p.2), e.2) := fun hp => by
    cases ρ <;> first | rfl | exact hp.elim
  have hplain : ρ.Plain → (e.1.map (·.2)).sum = ρ.results ∧ 0 < e.2 :=
      fun hp => by
    rw [plain hp, List.mem_map] at he
    obtain ⟨o, ho, rfl⟩ := he
    obtain ⟨r, hr, hsort⟩ := exists_sorted_of_mem_outcomes hρ ho
    have hlen := congrArg List.length hsort
    rw [Record.sorted, List.length_mergeSort, length_expand,
      (mem_support_law hρ.even hr).1] at hlen
    simp only [List.map_map, Function.comp_def]
    exact ⟨hlen.symm, weight_pos_of_mem_outcomes hρ ho⟩
  cases ρ with
  | chains count law =>
    obtain ⟨hc, hp⟩ := multisets_props law count.toNat e he
    exact ⟨hc, hp hρ.2⟩
  | stacked ms ls => exact (hns ms ls rfl).elim
  | range | standard | custom | fixed | pool => exact hplain trivial

/--
The multisets of chains of a well-formed roll but stacks weigh something.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
- `hns`: The roll is not stacks.
-/
theorem Roll.sum_stacks_ne_zero {ρ : Roll} (hρ : ρ.WellFormed)
    (hns : ∀ ms ls, ρ ≠ .stacked ms ls) : (ρ.stacks.map (·.2)).sum ≠ 0 := by
  cases ρ with
  | chains count law =>
    obtain ⟨hne, hpos⟩ := hρ
    rw [Roll.stacks, sum_multisets]
    refine pow_ne_zero _ fun h0 => ?_
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    exact (hpos e he).ne' this
  | stacked ms ls => exact (hns ms ls rfl).elim
  | range | standard | custom | fixed | pool =>
    simp only [Roll.stacks, List.map_map, Function.comp_def]
    exact (sum_outcomes _ hρ trivial).2 ∘ ((sum_outcomes _ hρ trivial).1 ▸ ·)

/-! ## The chains of a setup -/

namespace Setup

/--
The kept chains of every record that a setup describes, after its drops and
keeps, as a multiset, have the law of the kept pool of its roll's multisets of
chains, over their weights, as `kept_pool` in `propagation/chain.rs` keeps
them: every drop and keep counts each chain still kept as one die, worth its
total.

# Parameters
- `s`: The setup.
- `ρ`: Its roll.

# Hypotheses
- `hρ`: The setup has the roll.
- `hw`: The roll is well formed.
- `hn`: The roll is normal.
- `hj`: The roll is joined.
-/
theorem map_keptChains_law {s : Setup} {ρ : Roll} (hρ : s.roll = some ρ)
    (hw : ρ.WellFormed) (hn : ρ.Normal) (hj : ρ.Joined) :
    s.law.map (fun r => (r.keptChains : Multiset Chain)) =
      (weighted [] (keptPool ρ.stacks s.lowest s.highest)).map
        (fun ms => (Roll.expand ms : Multiset Chain)) := by
  set L := s.lowest
  set H := s.highest
  -- The window of the sorted chains of a multiset.
  let G : Multiset Chain → Multiset Chain := fun M =>
    let cs := M.toList.mergeSort Chain.le
    ((cs.drop L).take (cs.length - L - H) : List Chain)
  -- Normal chains sorted are the sorted chains of their multiset.
  have hsort : ∀ cs : List Chain, cs.Pairwise (fun a b => a.le b = true) →
      (∀ c ∈ cs, NormalChain c) →
      (cs : Multiset Chain).toList.mergeSort Chain.le = cs := fun cs hs hn' => by
    have hp : (cs : Multiset Chain).toList.Perm cs :=
      Multiset.coe_eq_coe.mp (Multiset.coe_toList _)
    rw [mergeSort_chains_perm hp fun c hc => hn' c (hp.mem_iff.mp hc),
      List.mergeSort_of_pairwise hs]
  have hlhs : s.law.map (fun r => (r.keptChains : Multiset Chain)) =
      (ρ.law.map fun r => (r.keptChains : Multiset Chain)).map G := by
    simp only [law, hρ, Option.map_some, Option.getD_some, PMF.map_comp]
    refine map_congr_support fun r hr => ?_
    obtain ⟨hl, hh, _⟩ := Roll.mem_support_law_die hr
    have hlen := Roll.length_chains_of_mem_support_law hr
    have hnr := ρ.normal_of_mem_support_law hw hn hj hr
    simp only [Function.comp_apply, G]
    rw [hsort _ (r.keptChains_pairwise hnr) hnr,
      Record.length_keptChains_undropped hl hh hlen,
      ← Record.keptChains_drops L H hl hh hlen]
    rfl
  have hk : weighted [] (keptPool ρ.stacks L H) =
      (weighted [] ρ.stacks).map (fun ms => slice ms L H) := by
    rw [← weighted_map_of]
    exact congrArg₂ weighted (by simp [slice, group, Roll.expand]) rfl
  rw [hlhs, ρ.map_keptChains_law hw hn hj, PMF.map_comp, hk, PMF.map_comp]
  refine map_congr_support fun ms hms => ?_
  simp only [Function.comp_apply, G]
  have hnm : ∀ c ∈ Roll.expand ms, NormalChain c := fun c hc => by
    obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hc
    rw [List.eq_of_mem_replicate hpx]
    rcases mem_support_weighted hms with ⟨e, he, rfl⟩ | rfl
    · exact ρ.normal_of_mem_stacks hw hn hj he p hp
    · simp at hp
  refine coe_window_slice L H (pairwise_mergeSort_chains _) (fun c hc => ?_) ?_
  · exact hnm c (Multiset.mem_coe.mp
      (Multiset.mem_toList.mp ((List.mergeSort_perm _ _).mem_iff.mp hc)))
  · rw [Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _), Multiset.coe_toList]

end Setup

/-! ## Explodes of multisets of chains -/

namespace Setup

/-- The roll of a grounded setup is normal and joined, and is not stacks. -/
private theorem roll_props {s : Setup} {ρ : Roll} (hg : s.Grounded)
    (hρ : s.roll = some ρ) :
    ρ.Normal ∧ ρ.Joined ∧ ∀ ms ls, ρ ≠ .stacked ms ls := by
  have hgρ := hg.2 ρ hρ
  -- The results of any other roll are faces of the die.
  have plain : (∀ r ∈ s.law.support, ∀ x ∈ r.results,
      x ∈ s.replacement.faces) →
      ∀ r ∈ ρ.law.support, ∀ x ∈ r.results, i32Min ≤ x ∧ x ≤ i32Max :=
    fun h r hr x hx => by
      let r' : Record := { r with
        lowest := s.lowest, highest := s.highest, die := s.replacement }
      have hr' : r' ∈ s.law.support := by
        simp only [law, hρ, Option.map_some, Option.getD_some, PMF.support_map]
        exact ⟨r, hr, rfl⟩
      exact hg.1 x (h r' hr' x hx)
  cases ρ with
  | chains count law =>
    exact ⟨fun e he => (hgρ.2 e he).1, trivial, fun _ _ h => by cases h⟩
  | stacked => exact hgρ.elim
  | range | standard | custom | fixed | pool =>
    exact ⟨plain hgρ, trivial, fun _ _ h => by cases h⟩

/-- A setup whose chains are not alike, or have been dropped, explodes the
multisets of its chains still kept, by a limited explode that matches some
face of its die. -/
private theorem explode_multisets {s : Setup} {c : Criterion} {ρ : Roll}
    {n : Nat} (hρ : s.roll = some ρ) (hk : s.results - s.lowest - s.highest ≠ 0)
    (hm : s.replacement.faces.countP c.matches ≠ 0)
    (hna : ¬(ρ.chainLike = true ∧ s.lowest = 0 ∧ s.highest = 0))
    (hns : ∀ ms ls, ρ ≠ .stacked ms ls) (w : Bool) :
    s.explode c (some (n + 1)) w =
      { roll := some (ρ.explodeMultisets s.lowest s.highest
          (Burst.resolve s.replacement c (some (n + 1)) w) c w),
        die := some s.replacement } := by
  have hsat : ρ.isSaturated = false := by
    cases ρ <;> first | rfl | exact (hns _ _ rfl).elim
  have hre : ∀ b, ρ.explode s.lowest s.highest b c w =
      ρ.explodeMultisets s.lowest s.highest b c w := fun b => by
    unfold Roll.explode
    split
    · exact absurd ‹_› hna
    · cases ρ <;> first | rfl | exact (hns _ _ rfl).elim
  simp only [explode, hρ, hk, hsat, Bool.false_eq_true, or_false,
    ↓reduceIte, Burst.resolve_some w hm, hre]

/-- Stacks without laws, transformed without padding, keep their pool, and
hold the law of what each distinct chain of it becomes. -/
private theorem transformStacks_one (pool : List (List (Chain × Nat) × Nat))
    (f : Chain → List (Chain × Nat) × Nat) :
    transformStacks pool 1 f = .stacked pool
      ((pool.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
        (ch, (f ch).1)) := by
  simp [transformStacks]

/-- Exploding a setup whose chains are not independent and alike, or have
been dropped, by a limited explode that matches some face of its die,
transforms each multiset of its chains still kept as the specification
explodes each record's chains. -/
private theorem bursts_of_multisets {s : Setup} {c : Criterion} {ρ : Roll}
    {n : Nat} (whole : Bool) (hs : s.WellFormed) (hg : s.Grounded)
    (hρ : s.roll = some ρ) (hk : s.results - s.lowest - s.highest ≠ 0)
    (hm : s.replacement.faces.countP c.matches ≠ 0)
    (hna : ¬(ρ.chainLike = true ∧ s.lowest = 0 ∧ s.highest = 0)) :
    s.Bursts c (some (n + 1)) whole := by
  have hF : s.replacement.faces ≠ [] := fun h => hm (by rw [h]; rfl)
  let ext := extension c (customDie s.replacement.faces) (n + 1)
  let K : Chain → PMF Chain := fun ch =>
    if ch.grows c then ext.map (growChain ch) else PMF.pure ch
  obtain ⟨hn, hj, hns⟩ := roll_props hg hρ
  have hw := hs ρ hρ
  have hexp := explode_multisets (n := n) (c := c) hρ hk hm hna hns
  -- The specification explodes each record's kept chains independently.
  have hspec : ∀ r ∈ s.law.support, Spec.explode r c (some (n + 1)) =
      (bagDraws K r.keptChains).map
        fun M => Record.ofChains s.replacement M.toList := fun r hr => by
    have hf := fits_of_grounded hs hg hr
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
  -- A record of normal chains is viewed by their multiset.
  have hcv : ∀ L : List Chain, (∀ x ∈ L, NormalChain x) →
      (Record.ofChains s.replacement L).chainView =
        ((L : Multiset Chain), s.replacement) :=
    fun L hn => by
      simp only [Record.chainView, keptChains_ofChains _ hn,
        (Record.ofChains_fields _ _).2.2.2]
  set kp := keptPool ρ.stacks s.lowest s.highest with hkp
  have hkc := map_keptChains_law hρ hw hn hj
  have hWs := Roll.sum_stacks_ne_zero hw hns
  have hW : (kp.map (·.2)).sum ≠ 0 := by
    simpa [hkp, keptPool, Function.comp_def] using hWs
  -- Every chain still kept of the pool is a normal chain of faces of the die.
  have hpool : ∀ ms ∈ (weighted [] kp).support, ∀ ch ∈ Roll.expand ms,
      NormalChain ch ∧ ∀ x ∈ ch.dice, x ∈ s.replacement.faces := by
    intro ms hms ch hch
    have hm' : ((Roll.expand ms : List Chain) : Multiset Chain) ∈
        (s.law.map fun r => (r.keptChains : Multiset Chain)).support := by
      rw [hkc]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨ms, hms, rfl⟩
    obtain ⟨r, hr, hre⟩ := (PMF.mem_support_map_iff _ _ _).mp hm'
    have hch' : ch ∈ r.keptChains := by
      rw [← Multiset.mem_coe, hre]
      exact hch
    have hf := fits_of_grounded hs hg hr
    exact ⟨hf.normal ch hch', hf.faces ch hch'⟩
  -- The specification's explodes, as the multisets of the chains still kept
  -- weigh them.
  have hR : ∀ {β : Type} (V : Multiset Chain → β),
      (s.law.bind fun r => (bagDraws K r.keptChains).map V) =
        (weighted [] kp).bind fun ms =>
          (bagDraws K (Roll.expand ms)).map V := by
    intro β V
    have hperm : ∀ l : List Chain,
        bagDraws K ((l : Multiset Chain)).toList = bagDraws K l := fun l =>
      bagDraws_perm K (Multiset.coe_eq_coe.mp (Multiset.coe_toList _))
    calc _ = (s.law.map fun r => (r.keptChains : Multiset Chain)).bind
            fun M => (bagDraws K M.toList).map V := by
          rw [PMF.bind_map]
          exact congrArg _ (funext fun r => by
            simp only [Function.comp_apply, hperm])
      _ = _ := by
          rw [hkc, PMF.bind_map]
          exact congrArg _ (funext fun ms => by
            simp only [Function.comp_apply, hperm])
  -- Each record's explode, viewed by its kept chains or its totals.
  have hspec' : ∀ {β : Type} (V : Record → β) (U : Multiset Chain → β),
      (∀ M : Multiset Chain, (∀ x ∈ M, NormalChain x) →
        V (Record.ofChains s.replacement M.toList) = U M) →
      (s.law.bind fun r => Spec.explode r c (some (n + 1))).map V =
        (weighted [] kp).bind fun ms =>
          (bagDraws K (Roll.expand ms)).map U := by
    intro β V U hVU
    rw [← hR, PMF.map_bind]
    refine bind_congr_support fun r hr => ?_
    rw [hspec r hr, PMF.map_comp]
    refine map_congr_support fun M hM => hVU M fun x hx => ?_
    have hf := fits_of_grounded hs hg hr
    obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x hx
    exact hKn' ch (hf.normal ch hch) (hf.faces ch hch) x hx
  have hrep : ∀ ρ' : Roll,
      ({ roll := some ρ', die := some s.replacement } : Setup).replacement =
        s.replacement := fun _ => rfl
  rw [Bursts]
  cases whole with
  | true =>
    let bc := (Burst.resolve s.replacement c (some (n + 1)) true).chains c
    let laws := (kp.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
      (ch, bc ch)
    -- What each chain of the pool has become is its explode.
    have hbec : ∀ ms ∈ (weighted [] kp).support, ∀ ch ∈ Roll.expand ms,
        became laws ch = K ch := by
      intro ms hms ch hch
      obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW hms
      have hin : ch ∈ (kp.flatMap fun e => e.1.map (·.1)).dedup := by
        obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hch
        rw [List.eq_of_mem_replicate hpx]
        exact List.mem_dedup.mpr
          (List.mem_flatMap.mpr ⟨e, he, List.mem_map_of_mem hp⟩)
      simp only [became, laws, lookup_graph bc _ ch hin, bc,
        Burst.weighted_chains hm]
      rfl
    have hlaw : (s.burst c (some (n + 1)) true).law =
        ((weighted [] kp).bind fun ms =>
          (stackDraws laws ms).map (Record.ofChains .none)).map fun r =>
            { r with lowest := 0, highest := 0, die := s.replacement } := by
      simp only [burst, ↓reduceIte, hexp, Roll.explodeMultisets,
        transformStacks_one]
      simp only [law, Option.map_some, Option.getD_some, Roll.law, hrep]
      -- Every multiset keeps as many chains, each of whose explodes weighs
      -- alike, so that every multiset gains as many branches.
      have hFn : s.replacement.faces.length ^ (n + 1) ≠ 0 :=
        pow_ne_zero _ (by simpa using hF)
      rw [weighted_scale _ kp (stackWeight laws) (pow_ne_zero
          (ρ.results - s.lowest - s.highest) hFn) fun e he => by
        refine (stackWeight_of_const (w := s.replacement.faces.length ^ (n + 1))
          fun p hp => ?_).trans ?_
        · have hin : p.1 ∈ (kp.flatMap fun e => e.1.map (·.1)).dedup :=
            List.mem_dedup.mpr
              (List.mem_flatMap.mpr ⟨e, he, List.mem_map_of_mem hp⟩)
          simp only [laws, lookup_graph bc _ p.1 hin, Option.map_some,
            Option.getD_some, bc, Burst.sum_chains hm]
        · simp only [hkp, keptPool, List.mem_map] at he
          obtain ⟨e', he', rfl⟩ := he
          rw [count_slice, (Roll.stacks_props hw hns he').1],
        weighted_congr_default _ [] hW]
    simp only [↓reduceIte]
    rw [hlaw, hspec' Record.chainView (fun M => (M, s.replacement))
      fun M hM => by
        rw [hcv _ fun x hx => hM x (Multiset.mem_toList.mp hx),
          Multiset.coe_toList],
      PMF.map_comp, PMF.map_bind]
    refine bind_congr_support fun ms hms => ?_
    rw [← bagDraws_congr (hbec ms hms), ← map_coe_stackDraws, PMF.map_comp,
      PMF.map_comp]
    refine map_congr_support fun cs hcs => ?_
    have hn : ∀ x ∈ cs, NormalChain x := fun x hx => by
      have hM : (cs : Multiset Chain) ∈
          (bagDraws (became laws) (Roll.expand ms)).support := by
        rw [← map_coe_stackDraws]
        exact (PMF.mem_support_map_iff _ _ _).mpr ⟨cs, hcs, rfl⟩
      obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x hx
      rw [hbec ms hms ch hch] at hx
      obtain ⟨hnc, hfc⟩ := hpool ms hms ch hch
      exact hKn' ch hnc hfc x hx
    simp only [Function.comp_apply]
    rw [ofChains_with, hcv cs hn]
  | false =>
    obtain ⟨bt, hbtd⟩ : ∃ f,
        f = (Burst.resolve s.replacement c (some (n + 1)) false).totals c :=
      ⟨_, rfl⟩
    let T := transformPool ρ.stacks s.lowest s.highest 1 fun ch => (bt ch, 0)
    have hlawT : ∀ e ∈ ρ.stacks, e.2 ≠ 0 →
        ∀ g ∈ slice e.1 s.lowest s.highest,
          (((fun ch => (bt ch, 0)) g.1 : List (Int × Nat) × Nat).1.map
            (·.2)).sum =
            s.replacement.faces.length ^ (n + 1) *
              1 ^ ((fun ch => (bt ch, 0)) g.1 : List (Int × Nat) × Nat).2 :=
      fun _ _ _ g _ => by simp [hbtd, Burst.sum_totals hm]
    have hkT : ∀ e ∈ ρ.stacks, e.2 ≠ 0 →
        ((slice e.1 s.lowest s.highest).map (·.2)).sum =
          ρ.results - s.lowest - s.highest := fun e he _ => by
      rw [count_slice, (Roll.stacks_props hw hns he).1]
    have hFn : s.replacement.faces.length ^ (n + 1) ≠ 0 :=
      pow_ne_zero _ (by simpa using hF)
    have hT := sum_transformPool_ne_zero ρ.stacks s.lowest s.highest 1
      (fun ch => (bt ch, 0)) _ _ hFn one_ne_zero hWs hlawT hkT
    have hP := weighted_transformPool [] [] (fun _ => 0) ρ.stacks s.lowest
      s.highest 1 (fun ch => (bt ch, 0)) _ _ hFn one_ne_zero hWs hlawT hkT
    -- What each chain still kept totals is the total of its explode.
    have hbt : ∀ ch, weighted 0 (bt ch) = (K ch).map Chain.total := fun ch => by
      simp only [hbtd, Burst.weighted_totals hm, K]
      split
      · rw [PMF.map_comp]
        rfl
      · rw [PMF.pure_map]
    have hlaw : (s.burst c (some (n + 1)) false).law.map Record.view =
        (weighted [] T).map fun rs =>
          (sortBag (Roll.expand rs : Multiset Int), s.replacement) := by
      simp only [burst, Bool.false_eq_true, ↓reduceIte, hexp,
        Roll.explodeMultisets, collapse, Option.map_some, Roll.collapse,
        ← hbtd]
      simp only [law, Option.map_some, Option.getD_some, Roll.law, hrep,
        PMF.map_comp]
      have hmw := weighted_map ([] : List (Int × Nat)) T
        (fun rs => tally (Roll.expand rs)) hT
      rw [weighted_congr_default _ (tally (Roll.expand ([] : List (Int × Nat))))
          (by simpa [Function.comp_def] using hT), hmw, PMF.map_comp]
      refine congrArg (PMF.map · _) (funext fun rs => ?_)
      simp only [Function.comp_apply]
      rw [Record.view_of_undropped rfl rfl]
      simp only [Record.sorted, Prod.mk.injEq, and_true]
      rw [mergeSort_eq_sortBag, Multiset.coe_eq_coe.mpr
        (expand_tally_perm (Roll.expand rs))]
    simp only [Bool.false_eq_true, ↓reduceIte]
    rw [hlaw, hspec' Record.view (fun M => (sortBag (M.map Chain.total),
        s.replacement)) fun M _ => by
          rw [view_ofChains, ← Multiset.map_coe, Multiset.coe_toList],
      show (fun rs : List (Int × Nat) =>
          (sortBag (Roll.expand rs : Multiset Int), s.replacement)) =
        (fun M => (sortBag M, s.replacement)) ∘
          fun rs => (Roll.expand rs : Multiset Int) from rfl,
      ← PMF.map_comp, hP, hkp, weighted_keptPool, PMF.bind_map, PMF.map_bind]
    refine bind_congr_support fun ms _ => ?_
    simp only [Function.comp_apply]
    rw [show (fun M : Multiset Chain => (sortBag (M.map Chain.total),
        s.replacement)) = (fun M => (sortBag M, s.replacement)) ∘
          Multiset.map Chain.total from rfl,
      ← PMF.map_comp, bagDraws_map_map]
    simp only [hbt]

/-- An explode that is not one that extends alike chains extends the chains of
some roll by a limited explode that matches some face of its die. -/
private theorem extends_multisets {s : Setup} {c : Criterion}
    {limit : Option Nat} (ha : ¬s.ExtendsAlike c limit) :
    ∃ ρ n, s.roll = some ρ ∧ limit = some (n + 1) ∧
      s.results - s.lowest - s.highest ≠ 0 ∧
      s.replacement.faces.countP c.matches ≠ 0 ∧
      ¬(ρ.chainLike = true ∧ s.lowest = 0 ∧ s.highest = 0) := by
  simp only [ExtendsAlike, not_forall] at ha
  obtain ⟨ρ, hρ, hk, hna⟩ := ha
  -- Only a limited explode that matches some face extends.
  have hext : ∃ n, limit = some (n + 1) ∧
      s.replacement.faces.countP c.matches ≠ 0 := by
    by_contra hc
    refine hna ?_
    cases limit with
    | none =>
      simp only [Burst.resolve]
      split_ifs <;> trivial
    | some l =>
      cases l with
      | zero => simp [Burst.resolve]
      | succ n =>
        have hm : s.replacement.faces.countP c.matches = 0 := by
          by_contra hm
          exact hc ⟨n, rfl, hm⟩
        unfold Burst.resolve
        simp [hm]
  obtain ⟨n, rfl, hm⟩ := hext
  rw [Burst.resolve_some true hm] at hna
  exact ⟨ρ, n, hρ, rfl, hk, hm, hna⟩

/--
Exploding a setup, as `Mixture::explode` in `propagation/mixture.rs` explodes
each setup, describes the specification's explode of the records that it
describes, as every later instruction reads them (`Setup.Bursts`), whether the
explode does nothing, saturates every chain still kept, extends the
independent and alike chains of a range, dice, or i.i.d. chains none of which
has been dropped (`Setup.law_burst_alike`), or extends the chains still kept
of each multiset of the chains of any other roll but stacks, or after a drop,
which it transforms multiset by multiset; an unlimited explode that may extend
a chain the pass refuses. Mirrors `Setup::explode` and `Setup::collapse` in
`propagation/mixture.rs`, and `burst`, `Roll::explode`, `transform_pool`,
`transform_stacks`, `kept_pool`, `Roll::saturate`, and `Roll::collapse` in
`propagation/chain.rs`.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is grounded.
- `hfaced`: Every die of the setup has a face.
- `he`: The explode is not one that the pass refuses.
-/
theorem law_burst {s : Setup} (c : Criterion) (limit : Option Nat)
    (whole : Bool) (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced)
    (he : s.Explodes c limit) : s.Bursts c limit whole := by
  by_cases ha : s.ExtendsAlike c limit
  · exact law_burst_alike c limit whole hs hg hfaced he ha
  · obtain ⟨ρ, n, hρ, rfl, hk, hm, hna⟩ := extends_multisets ha
    exact bursts_of_multisets whole hs hg hρ hk hm hna

/-- Weighted multisets that each hold `k` values hold as many as the first. -/
private theorem even_of_count {α : Type} :
    ∀ (ms : List (List (α × Nat) × Nat)) (k : Nat),
      (∀ e ∈ ms, (e.1.map (·.2)).sum = k) →
      ∀ e ∈ ms, (e.1.map (·.2)).sum =
        ((ms.head?.map fun e => (e.1.map (·.2)).sum).getD 0)
  | [], _, _, e, he => by cases he
  | m :: ms, k, h, e, he => by
    rw [h e he]
    simp [h m List.mem_cons_self]

/--
A burst leaves a well-formed, grounded setup with faces well formed and with
faces, and, if it keeps the chains whole and leaves no stacks, grounded too,
so that the law of a later explode holds of it (`law_burst`): an explode that
does nothing leaves the setup, or reduces it to custom dice of its totals; one
that saturates leaves one multiset of the saturated chains, or of their value;
one that extends alike chains leaves i.i.d. chains of the die's faces, or
custom dice of their totals (`Setup.burst_invariants_alike`); and one that
extends the chains of each multiset leaves stacks of the kept pool, or a pool
of the multisets of their totals.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.
- `whole`: Whether the chains stay whole.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is grounded.
- `hfaced`: Every die of the setup has a face.
-/
theorem burst_invariants {s : Setup} (c : Criterion) (limit : Option Nat)
    (whole : Bool) (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced) :
    (s.burst c limit whole).WellFormed ∧ (s.burst c limit whole).Faced ∧
      (whole = true → (∀ ms ls, (s.burst c limit true).roll ≠
        some (.stacked ms ls)) → (s.burst c limit true).Grounded) := by
  by_cases ha : s.ExtendsAlike c limit
  · exact burst_invariants_alike c limit whole hs hg hfaced ha
  obtain ⟨ρ, n, hρ, rfl, hk, hm, hna⟩ := extends_multisets ha
  obtain ⟨_, _, hns⟩ := roll_props hg hρ
  have hw := hs ρ hρ
  have hexp := explode_multisets (n := n) (c := c) hρ hk hm hna hns
  have hWs := Roll.sum_stacks_ne_zero hw hns
  -- Every multiset of chains keeps as many chains, and weighs something.
  have hkept : ∀ e ∈ ρ.stacks,
      ((slice e.1 s.lowest s.highest).map (·.2)).sum =
        ρ.results - s.lowest - s.highest ∧ 0 < e.2 := fun e he => by
    obtain ⟨hc, hp⟩ := Roll.stacks_props hw hns he
    exact ⟨by rw [count_slice, hc], hp⟩
  cases whole with
  | true =>
    simp only [burst, ↓reduceIte, hexp, Roll.explodeMultisets,
      transformStacks_one]
    refine ⟨fun ρ' h => ?_, fun ρ' h => ?_, fun _ h => absurd rfl (h _ _)⟩
    · simp only [Option.mem_def, Option.some.injEq] at h
      subst h
      have hcount : ∀ e ∈ keptPool ρ.stacks s.lowest s.highest,
          (e.1.map (·.2)).sum = ρ.results - s.lowest - s.highest :=
        fun e he => by
          simp only [keptPool, List.mem_map] at he
          obtain ⟨e', he', rfl⟩ := he
          exact (hkept e' he').1
      refine ⟨fun h0 => hWs ?_, fun e he => ?_, fun e he => ?_,
        fun e he => ?_⟩
      · simp only [keptPool, List.map_eq_nil_iff] at h0
        simp [h0]
      · simp only [keptPool, List.mem_map] at he
        obtain ⟨e', he', rfl⟩ := he
        exact ⟨fun p hp => (mem_slice hp).2, (hkept e' he').2⟩
      · rw [even_of_count _ _ hcount e he]
        cases keptPool ρ.stacks s.lowest s.highest with
        | nil => rfl
        | cons m ms => rfl
      · simp only [List.mem_map] at he
        obtain ⟨ch, _, rfl⟩ := he
        exact Burst.chains_pos hm ch
    · simp only [Option.mem_def, Option.some.injEq] at h
      subst h
      trivial
  | false =>
    simp only [burst, Bool.false_eq_true, ↓reduceIte, hexp,
      Roll.explodeMultisets, collapse, Option.map_some, Roll.collapse]
    refine ⟨fun ρ' h => ?_, fun ρ' h => ?_, fun h => by cases h⟩
    · simp only [Option.mem_def, Option.some.injEq] at h
      subst h
      let bt := (Burst.resolve s.replacement c (some (n + 1)) false).totals c
      have hFn : s.replacement.faces.length ^ (n + 1) ≠ 0 :=
        pow_ne_zero _ fun h => hm (by rw [List.eq_nil_of_length_eq_zero h]; rfl)
      have hT := sum_transformPool_ne_zero ρ.stacks s.lowest s.highest 1
        (fun ch => (bt ch, 0)) _ (ρ.results - s.lowest - s.highest) hFn
        one_ne_zero hWs (fun _ _ _ g _ => by simp [bt, Burst.sum_totals hm])
        fun e he _ => (hkept e he).1
      -- Every multiset of totals holds as many totals, and weighs something.
      have hmem : ∀ o ∈ transformPool ρ.stacks s.lowest s.highest 1
          (fun ch => (bt ch, 0)), (o.1.map (·.2)).sum =
            ρ.results - s.lowest - s.highest ∧ 0 < o.2 := fun o ho => by
        obtain ⟨e, he, o', ho', h1, m, h2⟩ := mem_transformPool ho
        have hj := joinGroups_props (fun ch => (bt ch, 0)) 0 _ [([], 1)]
          (by simp) (fun g _ o ho => (Burst.totals_pos hm g.1).2 o ho) o' ho'
        rw [h1, h2, hj.1, (hkept e he).1, one_pow, mul_one, zero_add]
        exact ⟨rfl, Nat.mul_pos (hkept e he).2 hj.2⟩
      have hcount : ∀ e ∈ (transformPool ρ.stacks s.lowest s.highest 1
          fun ch => (bt ch, 0)).map fun e => (tally (Roll.expand e.1), e.2),
          (e.1.map (·.2)).sum = ρ.results - s.lowest - s.highest :=
        fun e he => by
          simp only [List.mem_map] at he
          obtain ⟨o, ho, rfl⟩ := he
          rw [← length_expand, (expand_tally_perm _).length_eq, length_expand]
          exact (hmem o ho).1
      refine ⟨fun h0 => hT ?_, fun e he => ?_, fun e he => ?_⟩
      · simp only [List.map_eq_nil_iff] at h0
        simp [bt, h0]
      · simp only [List.mem_map] at he
        obtain ⟨o, ho, rfl⟩ := he
        exact ⟨tally_wellFormed _, (hmem o ho).2⟩
      · rw [even_of_count _ _ hcount e he]
        cases (transformPool ρ.stacks s.lowest s.highest 1
          fun ch => (bt ch, 0)).map fun e => (tally (Roll.expand e.1), e.2) with
        | nil => rfl
        | cons m ms => rfl
    · simp only [Option.mem_def, Option.some.injEq] at h
      subst h
      trivial

end Setup

/-! ## Mixtures -/

/-- Transforming every setup of a mixture by every resolution of the
operands describes, as a view reads it, the specification's explode of the
mixture's law, if it does so of each setup. -/
private theorem mixtureLaw_of_setups {β : Type} {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)} (V : Record → β)
    (F : Setup → Criterion × Option Nat → Setup)
    (h : ∀ s ∈ μ.support, ∀ q ∈ κ.support, (F s q).law.map V =
      (s.law.bind fun r => explode r q.1 q.2).map V) :
    (mixtureLaw (lift F μ κ)).map V =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        explode p.1 p.2.1 p.2.2).map V := by
  simp only [mixtureLaw, lift, PMF.bind_bind, PMF.bind_map, PMF.map_bind,
    Function.comp_def]
  refine bind_congr_support fun s hs => ?_
  refine (bind_congr_support (p := κ)
    (f := fun q : Criterion × Option Nat => (F s q).law.map V)
    (g := fun q : Criterion × Option Nat =>
      (s.law.bind fun r => explode r q.1 q.2).map V)
    fun q hq => h s hs q hq).trans ?_
  simp only [PMF.map_bind]
  exact PMF.bind_comm _ _ _

/--
Exploding every setup of a mixture by every resolution of the explode's
operands, keeping the chains whole, describes the specification's explode of
the mixture's law, independent of the resolutions, as a later reroll or
explode reads it (`Record.chainView`), as `Mixture::explode` in
`propagation/mixture.rs` explodes a record whose operands are independent of
it (`law_explode_resolved`).

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the criterion and the limit, as the operands resolve them.

# Hypotheses
- `hμ`: Every setup is well formed, grounded, and has faces, and every
  resolution's explode of it is not one that the pass refuses.
-/
theorem mixtureLaw_burst_whole {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.WellFormed ∧ s.Grounded ∧ s.Faced ∧
      ∀ q ∈ κ.support, s.Explodes q.1 q.2) :
    (mixtureLaw (lift (fun s q => s.burst q.1 q.2 true) μ κ)).map
        Record.chainView =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        explode p.1 p.2.1 p.2.2).map Record.chainView :=
  mixtureLaw_of_setups _ _ fun s hs q hq => by
    obtain ⟨hw, hg, hf, he⟩ := hμ s hs
    have h := Setup.law_burst q.1 q.2 true hw hg hf (he q hq)
    simpa [Setup.Bursts] using h

/--
Exploding every setup of a mixture by every resolution of the explode's
operands, reducing the chains to their totals, describes the specification's
explode of the mixture's law, independent of the resolutions, as every later
drop, keep, and sum reads it (`Record.view`), as `Mixture::explode` in
`propagation/mixture.rs` explodes a record that no later reroll or explode
reads.

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the criterion and the limit, as the operands resolve them.

# Hypotheses
- `hμ`: Every setup is well formed, grounded, and has faces, and every
  resolution's explode of it is not one that the pass refuses.
-/
theorem mixtureLaw_burst_totals {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.WellFormed ∧ s.Grounded ∧ s.Faced ∧
      ∀ q ∈ κ.support, s.Explodes q.1 q.2) :
    (mixtureLaw (lift (fun s q => s.burst q.1 q.2 false) μ κ)).map
        Record.view =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        explode p.1 p.2.1 p.2.2).map Record.view :=
  mixtureLaw_of_setups _ _ fun s hs q hq => by
    obtain ⟨hw, hg, hf, he⟩ := hμ s hs
    have h := Setup.law_burst q.1 q.2 false hw hg hf (he q hq)
    simpa [Setup.Bursts] using h

/--
The law that `law_explode` in `XdySpec/World.lean` gives an exploded record is
the explode of the record's law, independent of the criterion and the limit
to which the joint law of the operands resolves, as `visit_explode` in
`propagation.rs` resolves each joint outcome before `Mixture::explode`
explodes by it.

# Parameters
- `ρ`: The law of the record.
- `ν`: The joint law of the operands.
- `p`: The predicate.
- `l`: The limit.
-/
theorem law_explode_resolved (ρ : PMF Record)
    (ν : PMF (AddressingMode → Int)) (p : Predicate) (l : Limit) :
    (lift Prod.mk ρ ν).bind
        (fun q => explode q.1 (p.resolve q.2) (l.resolve q.2)) =
      (lift Prod.mk ρ (ν.map fun v => (p.resolve v, l.resolve v))).bind
        fun q => explode q.1 q.2.1 q.2.2 := by
  simp only [lift, PMF.bind_bind, PMF.bind_map, PMF.map_comp, Function.comp_def]

end Xdy.Spec
