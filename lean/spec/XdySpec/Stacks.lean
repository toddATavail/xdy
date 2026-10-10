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
  of each (`Outcomes.weighted_outcomeStates`); of stacks, their joins, which
  draw each copy of each chain of each multiset of the pool from what it has
  become, weighing each multiset by the branches of those draws
  (`weighted_joined`), so that the joins that `joined` defers until a drop
  or a split needs them are the eager ones; and of any other roll, its
  outcomes, each result the chain of one die.
- Stacks reduced to their totals (`joinedTotals`), as `Roll::collapse` reduces
  them, leave the results of the records that they fill, as a multiset
  (`Roll.map_results_collapse`), and so the records of a setup as every later
  drop, keep, and sum views them.
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
- A reroll of a chain rerolls each of its kept dice that the criterion
  matches, the replacement of its last die becoming its last die
  (`rerollChain`), and what `rerolled` lists of it, over its weights, is that
  reroll (`weighted_rerolled`), weighing the padding raised to the chain's
  kept dice in all (`sum_rerolled`). The specification rerolls the kept chains
  of a record independently (`rerollChains_eq_bag`), whether the record holds
  chains or not (`map_chainView_rerollEach`), so a reroll of i.i.d. chains none
  of which has been dropped is the reroll of the law of each, padded
  (`weighted_transformLaw_rerolled`), and one after a drop is the reroll of
  each chain still kept of each multiset, into stacks whose multisets
  `transformStacks` pads, and whose deferred laws weigh them back
  (`stackWeight_of_pow`). `Setup.law_reroll` lifts `Setup.law_reroll_plain` of
  `XdySpec/Reroll.lean` to every grounded setup, by its kept chains if its
  chains stay whole and by its kept results if they reduce to their totals
  (`Setup.rerollAs`, `Setup.RerollsAs`), as every record that either reroll
  leaves fits the die (`fits_reroll`, `view_of_fits`), and so to a mixture
  (`mixtureLaw_reroll_whole`, `mixtureLaw_reroll_totals`). Every reroll leaves
  a well-formed setup with faces, grounded if it keeps its chains whole and
  leaves no stacks (`Setup.reroll_invariants`).
- The proofs ask that the roll be well formed, that its chains be normal, and
  its results lie within the `i32` range (`Roll.Normal`); the explodes and
  rerolls ask that the setup be grounded (`Setup.Grounded`), and so hold no
  stacks: the rerolls and explodes of stacks none of whose chains has been
  dropped, which compose the laws that they defer, are left to later proofs.

```mermaid
flowchart TD
    S[grounded setup] --> A{alike, none dropped?}
    A -->|yes| L[Setup.law_burst_alike: each chain's law]
    A -->|no| P[ρ.stacks: multisets of chains]
    P -->|kept_pool, slice| K[kept pool]
    K -->|whole| T[stacks: each chain's explode deferred]
    P -->|totals: transform_pool| Q[pool of the totals of the kept chains]
    S -->|reroll, none dropped| R[transformLaw: each chain's law, rerolled]
    K -->|reroll| U[stacks: each chain's reroll deferred, padded]
```
-/
namespace Xdy.Spec

open Xdy (Die Chain Criterion)
open scoped ENNReal
open Setup (sortBag mergeSort_eq_sortBag view_ofChains)

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

/--
Weighted values that list, for each outer value, the inner values that it
leads to, each weighed by the outer value's weight and its own, are the outer
values, each weighed by its weight times the total of its inner values,
followed by the inner: the law of joined stacks, whose multisets `joined` in
`propagation/chain.rs` weighs by their branches times those of the draws of
their chains, which need not weigh alike.

# Type parameters
- `α`: The type of the outer values.
- `β`: The type of the inner values.
- `γ`: The type of the values that they lead to.

# Parameters
- `d`, `d₀`, `dh`: The defaults of the flattened, outer, and inner values.
- `l`: The outer values, each with its weight.
- `h`: The inner values of each outer value, each with its weight.
- `f`: The value that each outer and inner value lead to.

# Hypotheses
- `hl`: The outer values, each weighed by its weight times the total of its
  inner values, weigh something.
-/
theorem weighted_flatMap_sum {α β γ : Type} [DecidableEq γ] (d : γ) (d₀ : α)
    (dh : α → β) (l : List (α × Nat)) (h : α → List (β × Nat))
    (f : α → β → γ)
    (hl : ((l.map fun e => (e.1, e.2 * ((h e.1).map (·.2)).sum)).map
      (·.2)).sum ≠ 0) :
    weighted d (l.flatMap fun e =>
        (h e.1).map fun o => (f e.1 o.1, e.2 * o.2)) =
      (weighted d₀ (l.map fun e => (e.1, e.2 * ((h e.1).map (·.2)).sum))).bind
        fun a => (weighted (dh a) (h a)).map (f a) := by
  have hsum : ∀ {δ M : Type} [AddMonoid M] (l : List δ) (g : δ → List M),
      (l.flatMap g).sum = (l.map fun e => (g e).sum).sum := by
    intro δ M _ l g
    induction l with
    | nil => rfl
    | cons a l ih => simp [List.flatMap_cons, ih]
  -- The total of the flattened weights.
  have htot : ((l.flatMap fun e =>
      (h e.1).map fun o => (f e.1 o.1, e.2 * o.2)).map (·.2)).sum =
        ((l.map fun e => (e.1, e.2 * ((h e.1).map (·.2)).sum)).map
          (·.2)).sum := by
    rw [List.map_flatMap, hsum]
    simp only [List.map_map, Function.comp_def]
    refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
    exact List.sum_map_mul_left _ _ _
  ext z
  rw [weighted_apply d _ (htot ▸ hl), htot, PMF.bind_apply,
    Outcomes.tsum_weighted d₀ _ hl]
  congr 1
  rw [List.map_flatMap, hsum]
  simp only [List.map_map, Function.comp_def]
  refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
  -- Each outer value's share.
  by_cases hS : ((h e.1).map (·.2)).sum = 0
  · have hz : ∀ o ∈ h e.1, o.2 = 0 := fun o ho =>
      List.sum_eq_zero_iff.mp hS o.2 (List.mem_map_of_mem ho)
    rw [hS, mul_zero, Nat.cast_zero, zero_mul]
    refine List.sum_eq_zero fun x hx => ?_
    obtain ⟨o, ho, rfl⟩ := List.mem_map.mp hx
    simp [hz o ho]
  rw [Outcomes.map_apply_eq_tsum, Outcomes.tsum_weighted _ _ hS, Nat.cast_mul,
    mul_assoc, ENNReal.mul_div_cancel (Nat.cast_ne_zero.mpr hS)
      (ENNReal.natCast_ne_top _), ← List.sum_map_mul_left]
  refine congrArg List.sum (List.map_congr_left fun o _ => ?_)
  push_cast
  split_ifs <;> ring

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

/--
The greatest of values, folded from any start, is at least each.

# Parameters
- `l`: The values.
- `a`: The start.
- `x`: A value.

# Hypotheses
- `x` is among the values.
-/
theorem le_foldl_max :
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

/--
The branches of the draws that a multiset of chains joins, each copy of each
chain drawn from the law of what it becomes: the total weight of the law of
each copy, multiplied, as `transform_pool` in `propagation/chain.rs`
multiplies the weights of the draws that it joins.

# Type parameters
- `T`: The type of what a chain becomes: a chain, or a total.

# Parameters
- `becomes`: The law of what a chain becomes, and its kept dice.
- `ms`: The multiset, each distinct chain with its copies.

# Returns
The branches.

# Examples
```lean
#guard drawWeight (fun _ => ([((0 : Int), 2), (1, 1)], 0))
  [(.single 1, 2), (.single 2, 1)] == 27
```
-/
def drawWeight {T : Type} (becomes : Chain → List (T × Nat) × Nat)
    (ms : List (Chain × Nat)) : Nat :=
  (ms.map fun g => (((becomes g.1).1.map (·.2)).sum) ^ g.2).prod

/-- The branches of the draws of a multiset are those of each of its chains,
copied. -/
private theorem drawWeight_eq_prod {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) :
    ∀ ms : List (Chain × Nat), drawWeight becomes ms =
      ((Roll.expand ms).map fun ch => ((becomes ch).1.map (·.2)).sum).prod
  | [] => rfl
  | g :: ms => by
    have ih := drawWeight_eq_prod becomes ms
    simp only [drawWeight, Roll.expand, List.map_cons, List.prod_cons,
      List.flatMap_cons, List.map_append, List.prod_append, List.map_replicate,
      List.prod_replicate] at ih ⊢
    rw [ih]

/-- The branches of the draws of a multiset do not depend on the order of its
chains. -/
private theorem drawWeight_perm {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) {ms ms' : List (Chain × Nat)}
    (h : (Roll.expand ms).Perm (Roll.expand ms')) :
    drawWeight becomes ms = drawWeight becomes ms' := by
  rw [drawWeight_eq_prod, drawWeight_eq_prod]
  exact (h.map _).prod_eq

/-- A slice that drops nothing keeps every chain. -/
private theorem expand_slice_zero (ms : List (Chain × Nat)) :
    (Roll.expand (slice ms 0 0)).Perm (Roll.expand ms) := by
  simp only [slice, List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by rw [List.length_mergeSort])]
  exact (expand_group_perm _).trans (List.mergeSort_perm _ _)

/-- The multisets of joined groups weigh those so far times the branches of
the draws of the groups. -/
private theorem sum_joinGroups_draws {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)),
      ((joinGroups becomes groups init).map (·.2)).sum =
        (init.map (·.2)).sum * drawWeight becomes groups
  | [], init => by simp [joinGroups, drawWeight]
  | g :: groups, init => by
    rw [joinGroups_cons, sum_joinGroups_draws becomes groups]
    have hstep : ((init.flatMap fun e =>
        (multisets (becomes g.1).1 g.2).map fun o =>
          (e.1 ++ o.1, e.2 * o.2)).map (·.2)).sum =
        (init.map (·.2)).sum * ((becomes g.1).1.map (·.2)).sum ^ g.2 := by
      rw [← sum_multisets, List.map_flatMap, sum_flatMap_nat,
        ← List.sum_map_mul_right]
      refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
      simp only [List.map_map, Function.comp_def]
      exact List.sum_map_mul_left _ _ _
    rw [hstep]
    simp only [drawWeight, List.map_cons, List.prod_cons]
    ring

/--
A pool transformed without drops or padding, over its weights, is each
multiset of the pool, weighed by its weight times the branches of the draws
of its chains (`drawWeight`), each copy of each chain drawn independently from
the law of what it becomes, as `transform_pool` in `propagation/chain.rs`
joins them for `joined` and `joined_totals`, whose laws need not weigh alike.

# Type parameters
- `T`: The type of what a chain becomes: a chain, or a total.

# Parameters
- `d`, `d₀`: The multisets if no weight were positive.
- `dc`: What a chain becomes if no weight of its law were positive.
- `outcomes`: The multisets of the chains, each with its branches.
- `becomes`: The law of what a chain becomes, and its kept dice.

# Hypotheses
- `hl`: The multisets, so weighed, weigh something.
-/
theorem weighted_transformPool_one {T : Type} [DecidableEq T]
    (d : List (T × Nat)) (d₀ : List (Chain × Nat)) (dc : Chain → T)
    (outcomes : List (List (Chain × Nat) × Nat))
    (becomes : Chain → List (T × Nat) × Nat)
    (hl : ((outcomes.map fun e => (e.1, e.2 * drawWeight becomes e.1)).map
      (·.2)).sum ≠ 0) :
    (weighted d (transformPool outcomes 0 0 1 becomes)).map
        (fun ms => (Roll.expand ms : Multiset T)) =
      (weighted d₀ (outcomes.map fun e =>
          (e.1, e.2 * drawWeight becomes e.1))).bind fun ms =>
        bagDraws (fun ch => weighted (dc ch) (becomes ch).1)
          (Roll.expand ms) := by
  let J : List (Chain × Nat) → List (List (T × Nat) × Nat) := fun ms =>
    joinGroups becomes (slice ms 0 0) [([], 1)]
  have hJ : ∀ ms, ((J ms).map (·.2)).sum = drawWeight becomes ms := fun ms => by
    show ((joinGroups becomes (slice ms 0 0) [([], 1)]).map (·.2)).sum = _
    simp only [sum_joinGroups_draws, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, add_zero, one_mul]
    exact drawWeight_perm becomes (expand_slice_zero ms)
  have hout : (outcomes.map fun e => (e.1, e.2 * ((J e.1).map (·.2)).sum)) =
      outcomes.map fun e => (e.1, e.2 * drawWeight becomes e.1) :=
    List.map_congr_left fun e _ => by rw [hJ]
  have h := weighted_flatMap_sum d d₀ (fun _ => []) outcomes J (fun _ o => o)
    (hout ▸ hl)
  rw [hout] at h
  rw [transformPool_eq]
  simp only [one_pow, mul_one]
  rw [show (outcomes.flatMap fun e =>
      (joinGroups becomes (slice e.1 0 0) [([], 1)]).map fun o =>
        (o.1, e.2 * o.2)) = outcomes.flatMap fun e =>
      (J e.1).map fun o => ((fun _ o => o) e.1 o.1, e.2 * o.2) from rfl,
    h, PMF.map_bind]
  refine bind_congr_support fun ms hms => ?_
  obtain ⟨e', he', rfl, he0⟩ := exists_pos_of_mem_support_weighted hl hms
  obtain ⟨e, _, rfl⟩ := List.mem_map.mp he'
  have hW : drawWeight becomes e.1 ≠ 0 := fun h0 => he0 (by simp [h0])
  rw [drawWeight_eq_prod] at hW
  rw [show (fun o : List (T × Nat) => o) = id from rfl, PMF.map_id,
    weighted_joinGroups becomes dc _ _ _ [] (by simp) fun g hg h0 => by
      obtain ⟨hg1, _⟩ := mem_slice hg
      exact hW (List.prod_eq_zero (List.mem_map.mpr
        ⟨g.1, hg1, h0⟩)),
    weighted_single [] [] one_ne_zero, PMF.pure_bind]
  simp only [Roll.expand, List.flatMap_nil, Multiset.coe_nil, zero_add]
  exact (PMF.map_id _).trans (bagDraws_perm _ (expand_slice_zero e.1))

/-- A pool transformed without drops or padding weighs each multiset of the
pool by its weight times the branches of the draws of its chains. -/
private theorem sum_transformPool_one {T : Type}
    (outcomes : List (List (Chain × Nat) × Nat))
    (becomes : Chain → List (T × Nat) × Nat) :
    ((transformPool outcomes 0 0 1 becomes).map (·.2)).sum =
      ((outcomes.map fun e => (e.1, e.2 * drawWeight becomes e.1)).map
        (·.2)).sum := by
  rw [transformPool_eq]
  simp only [one_pow, mul_one, List.map_flatMap, sum_flatMap_nat, List.map_map,
    Function.comp_def]
  refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
  rw [List.sum_map_mul_left, sum_joinGroups_draws,
    drawWeight_perm _ (expand_slice_zero e.1)]
  simp

/-- Every multiset of a pool transformed without drops or padding, by laws
each of whose values weighs something, holds as many values as some multiset
of the pool holds chains, and weighs something if that multiset does. -/
private theorem transformPool_one_props {T : Type}
    {outcomes : List (List (Chain × Nat) × Nat)}
    {becomes : Chain → List (T × Nat) × Nat} {o : List (T × Nat) × Nat}
    (hpos : ∀ ch, ∀ v ∈ (becomes ch).1, 0 < v.2)
    (ho : o ∈ transformPool outcomes 0 0 1 becomes) :
    ∃ e ∈ outcomes, (o.1.map (·.2)).sum = (e.1.map (·.2)).sum ∧
      (0 < e.2 → 0 < o.2) := by
  obtain ⟨e, he, o', ho', h1, m, h2⟩ := mem_transformPool ho
  obtain ⟨hc, hp⟩ := joinGroups_props becomes 0 _ [([], 1)] (by simp)
    (fun g _ => hpos g.1) o' ho'
  rw [h1, h2, one_pow, mul_one, hc, count_slice]
  exact ⟨e, he, by simp, fun h => Nat.mul_pos h hp⟩

/--
Draws of what each value becomes, each mapped, are draws of the mapped laws.

# Parameters
- `step`: The law of what each value becomes.
- `g`: The map.
- `xs`: The values.
-/
theorem bagDraws_map_map {α β γ : Type} (step : α → PMF β)
    (g : β → γ) :
    ∀ xs : List α, (bagDraws step xs).map (Multiset.map g) =
      bagDraws (fun x => (step x).map g) xs
  | [] => by simp [bagDraws, PMF.pure_map]
  | x :: xs => by
    simp only [bagDraws, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      ← bagDraws_map_map step g xs, Function.comp_def, Multiset.map_cons]

/--
The value that a list of keys, each with its value, gives a key among them.

# Parameters
- `f`: The value of each key.
- `l`: The keys.
- `x`: A key.

# Hypotheses
- `x` is among the keys.
-/
theorem lookup_graph {β : Type} (f : Chain → β) :
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
A multiset of stacks, each of whose chains has become what a law that weighs
`G` raised to some power of it gives it, gains `G` raised to the sum of those
powers over its chains, each counted as often as it is copied.

# Parameters
- `laws`: The law of what each distinct chain has become.
- `G`: The base.
- `f`: The power of each chain.
- `ms`: The multiset, each distinct chain with its copies.

# Hypotheses
- The law of what every chain of the multiset has become weighs `G` raised to
  its power.
-/
theorem stackWeight_of_pow {laws : List (Chain × List (Chain × Nat))}
    {G : Nat} {f : Chain → Nat} : ∀ {ms : List (Chain × Nat)},
      (∀ p ∈ ms, (((laws.lookup p.1).map fun l => (l.map (·.2)).sum).getD 1) =
        G ^ f p.1) →
      stackWeight laws ms = G ^ (ms.map fun p => f p.1 * p.2).sum
  | [], _ => rfl
  | p :: ms, h => by
    have ih := stackWeight_of_pow (ms := ms) fun q hq => h q (by simp [hq])
    simp only [stackWeight, List.map_cons, List.prod_cons,
      List.sum_cons] at ih ⊢
    rw [ih, h p List.mem_cons_self, ← pow_mul, ← pow_add]

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

/-- The branches that the laws of stacks add to a multiset are those of the
draws that `joined` joins. -/
private theorem stackWeight_eq_drawWeight
    (laws : List (Chain × List (Chain × Nat))) (ms : List (Chain × Nat)) :
    stackWeight laws ms =
      drawWeight (fun ch => ((laws.lookup ch).getD [(ch, 1)], 0)) ms := by
  simp only [stackWeight, drawWeight]
  refine congrArg List.prod (List.map_congr_left fun p _ => ?_)
  cases laws.lookup p.1 <;> simp

/--
Joined stacks, over their weights, are the law of the chains of the stacks,
as a multiset: each multiset of the pool, weighed by its weight times the
branches that the laws of what its chains have become add to it
(`stackWeight`), each copy of each chain drawn independently from what it has
become (`became`). So the deferred joins of stacks, which `joined` in
`propagation/chain.rs` performs only when a drop or a split needs them, are
the eager ones.

# Parameters
- `d₀`: The multiset if no weight were positive.
- `pool`: The multisets of the stacks, each with its branches.
- `laws`: The law of what each distinct chain of the pool has become.

# Hypotheses
- `hl`: The multisets, so weighed, weigh something.
-/
theorem weighted_joined (d₀ : List (Chain × Nat))
    (pool : List (List (Chain × Nat) × Nat))
    (laws : List (Chain × List (Chain × Nat)))
    (hl : ((pool.map fun e => (e.1, e.2 * stackWeight laws e.1)).map
      (·.2)).sum ≠ 0) :
    (weighted [] (joined pool laws)).map
        (fun ms => (Roll.expand ms : Multiset Chain)) =
      (weighted d₀ (pool.map fun e => (e.1, e.2 * stackWeight laws e.1))).bind
        fun ms => bagDraws (became laws) (Roll.expand ms) := by
  unfold joined
  split
  · rename_i he
    have hnil : laws = [] := List.isEmpty_iff.mp he
    subst hnil
    simp only [stackWeight_nil, mul_one] at hl ⊢
    have hp : (pool.map fun e => (e.1, e.2)) = pool := by simp
    rw [hp] at hl ⊢
    rw [weighted_congr_default d₀ [] hl, ← PMF.bind_pure_comp]
    refine congrArg _ (funext fun ms => ?_)
    exact (bagDraws_pure fun x _ => rfl).symm
  · simp only [stackWeight_eq_drawWeight] at hl ⊢
    rw [weighted_transformPool_one [] d₀ id pool _ hl]
    refine congrArg _ (funext fun ms => bagDraws_congr fun ch _ => ?_)
    simp only [became, id]
    cases laws.lookup ch with
    | some law => rfl
    | none => exact weighted_single ch ch one_ne_zero

/-- A key that a list of keys, each with its value, gives a value is among
them, with that value. -/
private theorem mem_of_lookup {β : Type} :
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

/-- A multiset of stacks whose laws each hold a chain, each of positive
weight, gains some branches from them. -/
theorem stackWeight_ne_zero {laws : List (Chain × List (Chain × Nat))}
    (hlaws : ∀ e ∈ laws, e.2 ≠ [] ∧ ∀ p ∈ e.2, 0 < p.2)
    (ms : List (Chain × Nat)) : stackWeight laws ms ≠ 0 := by
  rw [stackWeight_eq_drawWeight, drawWeight, Ne, List.prod_eq_zero_iff]
  simp only [List.mem_map, not_exists, not_and]
  intro p _ h0
  have hb := pow_eq_zero_iff'.mp h0 |>.1
  cases hl : laws.lookup p.1 with
  | none => simp [hl] at hb
  | some law =>
    simp only [hl, Option.getD_some] at hb
    obtain ⟨hne, hpos⟩ := hlaws _ (mem_of_lookup hl)
    obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp hb q.2 (List.mem_map_of_mem hq)
    exact (hpos q hq).ne' this

/-- The multisets of well-formed stacks, each weighed by its weight times the
branches that their laws add to it, weigh something. -/
private theorem sum_stackWeights_ne_zero
    {ms : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))}
    (hρ : (Roll.stacked ms laws).WellFormed) :
    ((ms.map fun e => (e.1, e.2 * stackWeight laws e.1)).map (·.2)).sum ≠ 0 := by
  obtain ⟨hne, hpos, _, hlaws⟩ := hρ
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
  intro h0
  have := List.sum_eq_zero_iff.mp h0 (e.2 * stackWeight laws e.1)
    (List.mem_map.mpr ⟨_, List.mem_map.mpr ⟨e, he, rfl⟩, rfl⟩)
  exact mul_ne_zero (hpos e he).2.ne' (stackWeight_ne_zero hlaws e.1) this

/-- Joined stacks weigh their multisets, each weighed by its weight times the
branches that their laws add to it. -/
private theorem sum_joined (pool : List (List (Chain × Nat) × Nat))
    (laws : List (Chain × List (Chain × Nat))) :
    ((joined pool laws).map (·.2)).sum =
      ((pool.map fun e => (e.1, e.2 * stackWeight laws e.1)).map (·.2)).sum := by
  unfold joined
  split
  · rename_i he
    rw [List.isEmpty_iff.mp he]
    simp
  · rw [transformPool_eq]
    simp only [one_pow, mul_one, List.map_flatMap, sum_flatMap_nat, List.map_map,
      Function.comp_def]
    refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
    rw [List.sum_map_mul_left, sum_joinGroups_draws, stackWeight_eq_drawWeight,
      drawWeight_perm _ (expand_slice_zero e.1)]
    simp

/-- Every value of every multiset of joined groups is one of the multisets so
far, or what some chain of the groups may become. -/
private theorem joinGroups_mem {T : Type}
    (becomes : Chain → List (T × Nat) × Nat) (P : T → Prop) :
    ∀ (groups : List (Chain × Nat)) (init : List (List (T × Nat) × Nat)),
      (∀ e ∈ init, ∀ p ∈ e.1, P p.1) →
      (∀ g ∈ groups, ∀ q ∈ (becomes g.1).1, P q.1) →
      ∀ o ∈ joinGroups becomes groups init, ∀ p ∈ o.1, P p.1
  | [], init, hi, _, o, ho => hi o ho
  | g :: groups, init, hi, hg, o, ho => by
    rw [joinGroups_cons] at ho
    refine joinGroups_mem becomes P groups _ (fun e he p hp => ?_)
      (fun g' hg' => hg g' (List.mem_cons_of_mem _ hg')) o ho
    simp only [List.mem_flatMap, List.mem_map] at he
    obtain ⟨e', he', o', ho', rfl⟩ := he
    rcases List.mem_append.mp hp with hp | hp
    · exact hi e' he' p hp
    · obtain ⟨q, hq, hqp⟩ := mem_multisets _ _ _ ho' p hp
      exact hqp ▸ hg g List.mem_cons_self q hq

/--
Every multiset of joined stacks joins draws of what the chains of some
multiset of the pool have become: each of its chains is a chain of the law of
what some chain of the multiset has become, and, if every chain of every law
weighs something, it holds as many chains as the multiset, and weighs
something if the multiset does.

# Parameters
- `pool`: The multisets of the stacks, each with its branches.
- `laws`: The law of what each distinct chain of the pool has become.
- `e`: A multiset of the joined stacks, with its branches.

# Hypotheses
- `he`: `e` is a multiset of the joined stacks.
-/
theorem mem_joined {pool : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))} {e : List (Chain × Nat) × Nat}
    (he : e ∈ joined pool laws) :
    ∃ e' ∈ pool, (∀ p ∈ e.1, ∃ q ∈ e'.1,
        ∃ r ∈ (laws.lookup q.1).getD [(q.1, 1)], r.1 = p.1) ∧
      ((∀ l ∈ laws, ∀ r ∈ l.2, 0 < r.2) →
        (e.1.map (·.2)).sum = (e'.1.map (·.2)).sum ∧ (0 < e'.2 → 0 < e.2)) := by
  unfold joined at he
  split at he
  · rename_i hl
    rw [List.isEmpty_iff.mp hl]
    exact ⟨e, he, fun p hp => ⟨p, hp, (p.1, 1), by simp, rfl⟩,
      fun _ => ⟨rfl, id⟩⟩
  · obtain ⟨e', he', o', ho', h1, m, h2⟩ := mem_transformPool he
    let J : Chain → List (Chain × Nat) × Nat := fun ch =>
      ((laws.lookup ch).getD [(ch, 1)], 0)
    have hJ : ∀ (hl : ∀ l ∈ laws, ∀ r ∈ l.2, 0 < r.2),
        ∀ g ∈ slice e'.1 0 0, ∀ o ∈ (J g.1).1, 0 < o.2 := fun hl g _ o ho => by
      cases hg : laws.lookup g.1 with
      | none =>
        simp only [J, hg, Option.getD_none, List.mem_singleton] at ho
        subst ho
        exact Nat.one_pos
      | some l =>
        simp only [J, hg, Option.getD_some] at ho
        exact hl _ (mem_of_lookup hg) o ho
    refine ⟨e', he', fun p hp => ?_, fun hl => ?_⟩
    · rw [h1] at hp
      refine joinGroups_mem J (fun x => ∃ q ∈ e'.1,
        ∃ r ∈ (laws.lookup q.1).getD [(q.1, 1)], r.1 = x) _ _ (by simp)
        (fun g hg r hr => ?_) o' ho' p hp
      obtain ⟨hg1, _⟩ := mem_slice hg
      obtain ⟨q, hq, hqg⟩ := List.mem_flatMap.mp hg1
      have hr' : r ∈ (laws.lookup g.1).getD [(g.1, 1)] := hr
      rw [List.eq_of_mem_replicate hqg] at hr'
      exact ⟨q, hq, r, hr', rfl⟩
    · obtain ⟨hc, hpos⟩ := joinGroups_props J 0 _ [([], 1)] (by simp)
        (hJ hl) o' ho'
      rw [h1, h2, one_pow, mul_one, hc, count_slice]
      exact ⟨by simp, fun hp => Nat.mul_pos hp hpos⟩

/-- The results of a record of chains, as a multiset, are their totals. -/
private theorem results_ofChains (die : Die) (cs : List Chain) :
    ((Record.ofChains die cs).results : Multiset Int) = cs.map Chain.total := by
  obtain ⟨_, hl, hh, _⟩ := Record.ofChains_fields die cs
  rw [← kept_ofChains die cs, Record.kept, hl, hh, List.drop_zero,
    Nat.sub_zero, Nat.sub_zero, show (Record.ofChains die cs).results.length =
      (Record.ofChains die cs).sorted.length from
        (List.length_mergeSort _).symm, List.take_length]
  exact Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _).symm

/-- The multisets of the totals of well-formed stacks weigh something, and
each holds as many totals as the stacks hold chains, and weighs something. -/
private theorem joinedTotals_props {ms : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))}
    (hρ : (Roll.stacked ms laws).WellFormed) :
    ((joinedTotals ms laws).map (·.2)).sum ≠ 0 ∧
      ∀ o ∈ joinedTotals ms laws,
        (o.1.map (·.2)).sum = (Roll.stacked ms laws).results ∧ 0 < o.2 := by
  have hW := sum_stackWeights_ne_zero hρ
  obtain ⟨_, hpos, heven, hlaws⟩ := hρ
  refine ⟨?_, fun o ho => ?_⟩
  · rw [joinedTotals, sum_transformPool_one]
    refine ne_of_eq_of_ne (congrArg List.sum
      (congrArg (List.map fun e : List (Chain × Nat) × Nat => e.2)
        (List.map_congr_left fun e _ => ?_))) hW
    rw [stackWeight_eq_drawWeight]
    simp [drawWeight, Function.comp_def]
  · obtain ⟨e, he, hc, hp⟩ := transformPool_one_props (fun ch v hv => by
      obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hv
      cases hl : laws.lookup ch with
      | none =>
        simp only [hl, Option.getD_none, List.mem_singleton] at hr
        subst hr
        exact Nat.one_pos
      | some law =>
        simp only [hl, Option.getD_some] at hr
        exact (hlaws _ (mem_of_lookup hl)).2 r hr) ho
    exact ⟨hc.trans (heven e he), hp (hpos e he).2⟩

/--
Reducing stacks to a pool of the multisets of their totals, as
`Roll::collapse` in `propagation/chain.rs` reduces them by `joined_totals`,
leaves the results of the records that they fill, as a multiset: each draws
the total of what each copy of each chain of each multiset of the pool has
become, as the chains themselves draw what each has become.

# Parameters
- `ms`: The multisets of the stacks, each with its branches.
- `laws`: The law of what each distinct chain of the pool has become.

# Hypotheses
- `hρ`: The stacks are well formed.
-/
theorem Roll.map_results_collapse {ms : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))}
    (hρ : (Roll.stacked ms laws).WellFormed) :
    (Roll.stacked ms laws).collapse.law.map
        (fun r => (r.results : Multiset Int)) =
      (Roll.stacked ms laws).law.map fun r => (r.results : Multiset Int) := by
  let JT : Chain → List (Int × Nat) × Nat := fun ch =>
    (((laws.lookup ch).getD [(ch, 1)]).map fun e => (e.1.total, e.2), 0)
  have hdw : ∀ m, drawWeight JT m = stackWeight laws m := fun m => by
    rw [stackWeight_eq_drawWeight]
    simp only [drawWeight, JT, List.map_map, Function.comp_def]
  have hout : (ms.map fun e => (e.1, e.2 * drawWeight JT e.1)) =
      ms.map fun e => (e.1, e.2 * stackWeight laws e.1) :=
    List.map_congr_left fun e _ => by rw [hdw]
  have hW := sum_stackWeights_ne_zero hρ
  have hJT := (joinedTotals_props hρ).1
  have hP := weighted_transformPool_one [] ((ms.head?.map (·.1)).getD [])
    Chain.total ms JT (by rw [hout]; exact hW)
  rw [hout] at hP
  -- What each chain totals is the total of what it has become.
  have hbt : ∀ ch, weighted ch.total (JT ch).1 = (became laws ch).map
      Chain.total := fun ch => by
    simp only [JT, became]
    rw [weighted_map_of]
    cases laws.lookup ch with
    | some law => rfl
    | none => rw [Option.getD_none, weighted_single ch ch one_ne_zero]
  simp only [Roll.collapse, Roll.law, PMF.map_comp, PMF.map_bind]
  rw [weighted_congr_default _ (tally (Roll.expand ([] : List (Int × Nat))))
      (by simpa [Function.comp_def] using hJT),
    weighted_map ([] : List (Int × Nat)) _ (fun rs => tally (Roll.expand rs))
      hJT, PMF.map_comp]
  have hc : (((fun r : Record => (r.results : Multiset Int)) ∘
      (fun rs => ({ results := Roll.expand rs } : Record))) ∘
        fun rs => tally (Roll.expand rs)) =
      fun rs : List (Int × Nat) => (Roll.expand rs : Multiset Int) :=
    funext fun rs => Multiset.coe_eq_coe.mpr (expand_tally_perm _)
  rw [hc]
  change (weighted [] (transformPool ms 0 0 1 JT)).map _ = _
  rw [hP]
  refine bind_congr_support fun m _ => ?_
  rw [bagDraws_congr fun ch _ => hbt ch, ← bagDraws_map_map,
    ← map_coe_stackDraws, PMF.map_comp]
  refine congrArg (PMF.map · _) (funext fun cs => ?_)
  simp only [Function.comp_apply, results_ofChains, Multiset.map_coe]

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

/-- Every chain of the law of what a chain of normal stacks has become is
normal. -/
private theorem normal_of_mem_lawOf {ms : List (List (Chain × Nat) × Nat)}
    {laws : List (Chain × List (Chain × Nat))} {e : List (Chain × Nat) × Nat}
    {q r : Chain × Nat} (hn : (Roll.stacked ms laws).Normal) (he : e ∈ ms)
    (hq : q ∈ e.1) (hr : r ∈ (laws.lookup q.1).getD [(q.1, 1)]) :
    NormalChain r.1 := by
  cases hl : laws.lookup q.1 with
  | none =>
    simp only [hl, Option.getD_none, List.mem_singleton] at hr
    subst hr
    exact hn.1 e he q hq
  | some law =>
    simp only [hl, Option.getD_some] at hr
    exact hn.2 _ (mem_of_lookup hl) r hr

/-- Every chain that a normal chain of stacks may have become is normal, if
every chain of their laws is. -/
private theorem normal_of_mem_support_became
    {laws : List (Chain × List (Chain × Nat))} {ch x : Chain}
    (hl : ∀ e ∈ laws, ∀ p ∈ e.2, NormalChain p.1) (hch : NormalChain ch)
    (hx : x ∈ (became laws ch).support) : NormalChain x := by
  unfold became at hx
  split at hx
  · rename_i law heq
    rcases mem_support_weighted hx with ⟨q, hq, rfl⟩ | rfl
    · exact hl _ (mem_of_lookup heq) q hq
    · exact hch
  · rw [PMF.mem_support_pure_iff] at hx
    exact hx ▸ hch

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
weighs them: independent draws of the law of each of i.i.d. chains, the
joined stacks, whose deferred joins are the eager ones (`weighted_joined`), or
the outcomes of any other roll, each result the chain of one die.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
- `hn`: The roll is normal.
-/
theorem map_keptChains_law {ρ : Roll} (hρ : ρ.WellFormed) (hn : ρ.Normal) :
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
    have hW := sum_stackWeights_ne_zero hρ
    simp only [law, Roll.stacks, PMF.map_bind]
    rw [weighted_joined ((ms.head?.map (·.1)).getD []) ms laws hW]
    refine bind_congr_support fun m hm => ?_
    -- Every chain of the multiset is normal.
    have hm' : ∀ p ∈ m, NormalChain p.1 := by
      rcases mem_support_weighted hm with ⟨e, he, rfl⟩ | rfl
      · obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
        exact hn.1 e' he'
      · cases ms with
        | nil => simp
        | cons e ms => exact hn.1 e List.mem_cons_self
    rw [PMF.map_comp, ← map_coe_stackDraws]
    refine map_congr_support fun cs hcs => ?_
    simp only [Function.comp_apply]
    refine keptChains_ofChains _ fun x hx => ?_
    obtain ⟨p, hp, hx⟩ := (mem_support_stackDraws hcs).2 x hx
    exact normal_of_mem_support_became hn.2 (hm' p hp) hx

/--
Every chain of every multiset of chains of a well-formed, normal roll is
normal.

# Parameters
- `ρ`: The roll.
- `e`: A multiset with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `hn`: The roll is normal.
- `he`: `e` is a multiset of chains of the roll.
-/
theorem normal_of_mem_stacks {ρ : Roll} {e : List (Chain × Nat) × Nat}
    (hρ : ρ.WellFormed) (hn : ρ.Normal) (he : e ∈ ρ.stacks) :
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
    intro p hp
    obtain ⟨e', he', hmem, _⟩ := mem_joined he
    obtain ⟨q, hq, r, hr, hrp⟩ := hmem p hp
    rw [← hrp]
    exact normal_of_mem_lawOf hn he' hq hr

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
Every kept chain of every record that a well-formed, normal roll fills is
normal.

# Parameters
- `ρ`: The roll.
- `r`: The record.

# Hypotheses
- `hw`: The roll is well formed.
- `hn`: The roll is normal.
- `hr`: The roll can fill `r`.
-/
theorem Roll.normal_of_mem_support_law {ρ : Roll} {r : Record}
    (hw : ρ.WellFormed) (hn : ρ.Normal)
    (hr : r ∈ ρ.law.support) : ∀ ch ∈ r.keptChains, NormalChain ch := by
  have hm : (r.keptChains : Multiset Chain) ∈
      (ρ.law.map fun r => (r.keptChains : Multiset Chain)).support :=
    (PMF.mem_support_map_iff _ _ _).mpr ⟨r, hr, rfl⟩
  rw [ρ.map_keptChains_law hw hn, PMF.mem_support_map_iff] at hm
  obtain ⟨ms, hms, hcoe⟩ := hm
  intro ch hch
  have hch' : ch ∈ Roll.expand ms := by
    rw [← Multiset.mem_coe, hcoe]
    exact hch
  obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hch'
  rw [List.eq_of_mem_replicate hpx]
  rcases mem_support_weighted hms with ⟨e, he, rfl⟩ | rfl
  · exact ρ.normal_of_mem_stacks hw hn he p hp
  · simp at hp

/--
Every multiset of chains of a well-formed roll holds as many chains as the
roll's records hold results, and weighs something.

# Parameters
- `ρ`: The roll.
- `e`: A multiset with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is a multiset of chains of the roll.
-/
theorem Roll.stacks_props {ρ : Roll} {e : List (Chain × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.stacks) : (e.1.map (·.2)).sum = ρ.results ∧ 0 < e.2 := by
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
  | stacked ms ls =>
    obtain ⟨e', he', _, h⟩ := mem_joined he
    obtain ⟨_, hpos, heven, hlaws⟩ := hρ
    obtain ⟨hc, hp⟩ := h fun l hl r hr => (hlaws l hl).2 r hr
    exact ⟨hc.trans (heven e' he'), hp (hpos e' he').2⟩
  | range | standard | custom | fixed | pool => exact hplain trivial

/--
The multisets of chains of a well-formed roll weigh something.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
-/
theorem Roll.sum_stacks_ne_zero {ρ : Roll} (hρ : ρ.WellFormed) :
    (ρ.stacks.map (·.2)).sum ≠ 0 := by
  cases ρ with
  | chains count law =>
    obtain ⟨hne, hpos⟩ := hρ
    rw [Roll.stacks, sum_multisets]
    refine pow_ne_zero _ fun h0 => ?_
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    exact (hpos e he).ne' this
  | stacked ms ls =>
    rw [Roll.stacks, sum_joined]
    exact sum_stackWeights_ne_zero hρ
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
-/
theorem map_keptChains_law {s : Setup} {ρ : Roll} (hρ : s.roll = some ρ)
    (hw : ρ.WellFormed) (hn : ρ.Normal) :
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
    have hnr := ρ.normal_of_mem_support_law hw hn hr
    simp only [Function.comp_apply, G]
    rw [hsort _ (r.keptChains_pairwise hnr) hnr,
      Record.length_keptChains_undropped hl hh hlen,
      ← Record.keptChains_drops L H hl hh hlen]
    rfl
  have hk : weighted [] (keptPool ρ.stacks L H) =
      (weighted [] ρ.stacks).map (fun ms => slice ms L H) := by
    rw [← weighted_map_of]
    exact congrArg₂ weighted (by simp [slice, group, Roll.expand]) rfl
  rw [hlhs, ρ.map_keptChains_law hw hn, PMF.map_comp, hk, PMF.map_comp]
  refine map_congr_support fun ms hms => ?_
  simp only [Function.comp_apply, G]
  have hnm : ∀ c ∈ Roll.expand ms, NormalChain c := fun c hc => by
    obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hc
    rw [List.eq_of_mem_replicate hpx]
    rcases mem_support_weighted hms with ⟨e, he, rfl⟩ | rfl
    · exact ρ.normal_of_mem_stacks hw hn he p hp
    · simp at hp
  refine coe_window_slice L H (pairwise_mergeSort_chains _) (fun c hc => ?_) ?_
  · exact hnm c (Multiset.mem_coe.mp
      (Multiset.mem_toList.mp ((List.mergeSort_perm _ _).mem_iff.mp hc)))
  · rw [Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _), Multiset.coe_toList]

end Setup

/-! ## Explodes of multisets of chains -/

namespace Setup

/-- The roll of a grounded setup is normal, and is not stacks. -/
private theorem roll_props {s : Setup} {ρ : Roll} (hg : s.Grounded)
    (hρ : s.roll = some ρ) :
    ρ.Normal ∧ ∀ ms ls, ρ ≠ .stacked ms ls := by
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
    exact ⟨fun e he => (hgρ.2 e he).1, fun _ _ h => by cases h⟩
  | stacked => exact hgρ.elim
  | range | standard | custom | fixed | pool =>
    exact ⟨plain hgρ, fun _ _ h => by cases h⟩

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
    transformStacks pool [] 1 f = .stacked pool
      ((pool.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
        (ch, (f ch).1)) := by
  simp [transformStacks, compose]

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
  obtain ⟨hn, hns⟩ := roll_props hg hρ
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
  have hkc := map_keptChains_law hρ hw hn
  have hWs := Roll.sum_stacks_ne_zero hw
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
          rw [count_slice, (Roll.stacks_props hw he').1],
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
      rw [count_slice, (Roll.stacks_props hw he).1]
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
  obtain ⟨_, hns⟩ := roll_props hg hρ
  have hw := hs ρ hρ
  have hexp := explode_multisets (n := n) (c := c) hρ hk hm hna hns
  have hWs := Roll.sum_stacks_ne_zero hw
  -- Every multiset of chains keeps as many chains, and weighs something.
  have hkept : ∀ e ∈ ρ.stacks,
      ((slice e.1 s.lowest s.highest).map (·.2)).sum =
        ρ.results - s.lowest - s.highest ∧ 0 < e.2 := fun e he => by
    obtain ⟨hc, hp⟩ := Roll.stacks_props hw he
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

/-! ## Rerolls of chains -/

/-- A sorted list of values, as a multiset, is the values. -/
theorem coe_sortBag (M : Multiset Int) :
    ((sortBag M : List Int) : Multiset Int) = M := by
  rw [sortBag, Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _),
    Multiset.coe_toList]

/-- Rerolling the dice of a body one at a time, from sorted bodies, sorts the
draws into them. -/
private theorem foldl_rerollChain_bag (draw : Int → PMF Int) :
    ∀ (xs : List Int) (q : PMF (Multiset Int)),
      xs.foldl (fun (p : PMF (List Int)) x => p.bind fun b =>
          (draw x).map fun y => (y :: b).mergeSort) (q.map sortBag) =
        (lift (· + ·) q (bagDraws draw xs)).map sortBag
  | [], q => by
    simp only [List.foldl_nil, bagDraws, lift, PMF.pure_map, add_zero]
    rw [PMF.bind_pure]
  | x :: xs, q => by
    rw [List.foldl_cons]
    have hstep : ((q.map sortBag).bind fun b =>
        (draw x).map fun y => (y :: b).mergeSort) =
        (q.bind fun M => (draw x).map fun y => y ::ₘ M).map sortBag := by
      rw [PMF.bind_map, PMF.map_bind]
      refine congrArg _ (funext fun M => ?_)
      simp only [Function.comp_apply, PMF.map_comp]
      refine congrArg (PMF.map · _) (funext fun y => ?_)
      simp only [Function.comp_apply]
      rw [mergeSort_eq_sortBag, ← Multiset.cons_coe, coe_sortBag]
    rw [hstep, foldl_rerollChain_bag draw xs]
    simp only [bagDraws, lift, PMF.bind_bind, PMF.map_bind, PMF.bind_map,
      PMF.map_comp, Function.comp_def]
    refine congrArg _ (funext fun M => congrArg _ (funext fun y =>
      congrArg (PMF.map · _) (funext fun N => ?_)))
    rw [Multiset.cons_add, Multiset.add_cons]

/--
Rerolling a chain that may still grow draws each of its dice besides the last
independently, sorted, and then its last.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The law of the replacement of a rerolled result.
- `body`: The dice of the chain besides its last.
- `last`: Its last die.
-/
theorem rerollChain_live (rerolled : Int → Bool) (P : PMF Int)
    (body : List Int) (last : Int) :
    rerollChain rerolled P (.live body last) =
      (bagDraws (fun x => if rerolled x then P else PMF.pure x) body).bind
        fun M => (if rerolled last then P else PMF.pure last).map
          (.live (sortBag M) ·) := by
  have h := foldl_rerollChain_bag
    (fun x => if rerolled x then P else PMF.pure x) body (PMF.pure 0)
  simp only [lift, PMF.pure_bind, zero_add, PMF.pure_map] at h
  rw [show sortBag 0 = [] by simp [sortBag]] at h
  rw [rerollChain]
  simp only
  rw [h, PMF.bind_map, show (fun x : Multiset Int => x) = id from rfl,
    PMF.map_id]
  rfl

/--
Draws of one law, as a multiset, one for each value, are independent draws.

# Parameters
- `P`: The law of each draw.
- `xs`: The values.
-/
theorem bagDraws_const_iidBag {α β : Type} (P : PMF β) :
    ∀ xs : List α, bagDraws (fun _ => P) xs = iidBag xs.length P
  | [] => rfl
  | x :: xs => by
    rw [bagDraws, bagDraws_const_iidBag P xs, List.length_cons]
    simp only [iidBag, Nat.repeat, lift, ← PMF.bind_pure_comp]
    exact PMF.bind_comm _ _ _

/-- Rerolling dice draws the replacement of each that the criterion matches
and leaves the others: the survivors, joined with independent draws of the
replacement, one for each matched die. -/
private theorem bagDraws_survivors (c : Criterion) (P : PMF Int)
    (xs : List Int) :
    bagDraws (fun x => if c.matches x then P else PMF.pure x) xs =
      (iidBag (xs.countP c.matches) P).map
        (((xs.filter fun x => !c.matches x : List Int) : Multiset Int) +
          ·) := by
  rw [bagDraws_perm _ ((List.filter_append_perm (!c.matches ·) xs).symm),
    bagDraws_append, bagDraws_pure fun x hx => by
      have := (List.mem_filter.mp hx).2
      simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
      simp [this],
    bagDraws_congr (step' := fun _ => P) fun x hx => by
      have := (List.mem_filter.mp hx).2
      simp only [Bool.not_not] at this
      simp [this],
    bagDraws_const_iidBag, lift, PMF.pure_bind, List.countP_eq_length_filter]
  simp

/--
What a chain becomes after a reroll, over its weights, is the specification's
reroll of the chain: each die that the criterion matches replaced by a draw of
the final results of a matched die, and each other die left. Mirrors
`rerolled` in `propagation/chain.rs`.

# Parameters
- `d`: The chain if no weight were positive.
- `ch`: The chain.

# Hypotheses
- `hg`: The final results of a matched die weigh the padding in all.
- `hG`: The padding is positive.
-/
theorem weighted_rerolled {c : Criterion} {G : Nat} {g : List (Int × Nat)}
    (hg : (g.map (·.2)).sum = G) (hG : G ≠ 0) (d ch : Chain) :
    weighted d (rerolled c G g ch).1 =
      rerollChain c.matches (weightedDie g) ch := by
  cases ch with
  | saturated v =>
    simp only [rerolled, rerollChain]
    exact weighted_single _ _ one_ne_zero
  | live body last =>
    set S : List Int := body.filter fun x => !c.matches x
    set m := body.countP c.matches
    have hW : ((multisets g m).map (·.2)).sum ≠ 0 := by
      rw [sum_multisets, hg]
      exact pow_ne_zero _ hG
    have h := weighted_flatMap d [] (fun _ => (0 : Int)) (multisets g m)
      (fun _ => if c.matches last then g else [(last, 1)])
      (fun _ => G ^ (S.length + if c.matches last then 0 else 1))
      (fun o y => Chain.live (S ++ Roll.expand o).mergeSort y)
      (G ^ (S.length + if c.matches last then 0 else 1) *
        ((if c.matches last then g else [(last, 1)]).map (·.2)).sum)
      (Nat.mul_ne_zero (pow_ne_zero _ hG) (by split <;> simp [hg, hG]))
      hW fun _ _ _ => rfl
    simp only [rerolled]
    rw [h, rerollChain_live, bagDraws_survivors, PMF.bind_map,
      weightedDie_eq_weighted,
      ← Outcomes.weighted_outcomeStates 0 [] g (by rw [hg]; exact hG),
      PMF.bind_map]
    refine congrArg _ (funext fun o => ?_)
    simp only [Function.comp_apply]
    rw [mergeSort_eq_sortBag, Multiset.coe_add]
    split
    · rfl
    · rw [weighted_single _ _ one_ne_zero]

/--
What a chain becomes after a reroll weighs the padding raised to its kept
dice in all, as `rerolled` in `propagation/chain.rs` pads it.

# Parameters
- `ch`: The chain.

# Hypotheses
- `hg`: The final results of a matched die weigh the padding in all.
-/
theorem sum_rerolled {c : Criterion} {G : Nat} {g : List (Int × Nat)}
    (hg : (g.map (·.2)).sum = G) (ch : Chain) :
    ((rerolled c G g ch).1.map (·.2)).sum = G ^ (rerolled c G g ch).2 := by
  cases ch with
  | saturated v => simp [rerolled]
  | live body last =>
    have hsum : ∀ (l : List (List (Int × Nat) × Nat))
        (f : List (Int × Nat) × Nat → List Nat),
        (l.flatMap f).sum = (l.map fun e => (f e).sum).sum := fun l f => by
      induction l with
      | nil => rfl
      | cons a l ih => simp [List.flatMap_cons, ih]
    have hlen : (body.filter fun x => !c.matches x).length +
        body.countP c.matches = body.length := by
      have := List.length_eq_length_filter_add (l := body) c.matches
      rw [List.countP_eq_length_filter]
      omega
    simp only [rerolled, List.map_flatMap, List.map_map, Function.comp_def]
    rw [hsum]
    simp only [mul_assoc, List.sum_map_mul_left]
    rw [List.sum_map_mul_right, sum_multisets, hg, ← hlen]
    split <;> simp [hg, ← pow_add, ← pow_succ] <;> ring

/-- Transforming chains in place, each by its law, from any law of chains
before, adds the transformed chains to the multisets of the chains before. -/
private theorem foldl_transform_bag (K : Chain → PMF Chain) :
    ∀ (kept : List Chain) (p : PMF (List Chain)),
      (kept.foldl (fun (p : PMF (List Chain)) c => p.bind fun cs =>
          (K c).map fun c' => (c' :: cs).mergeSort Chain.le) p).map
          (Multiset.ofList : List Chain → Multiset Chain) =
        lift (· + ·) (p.map (Multiset.ofList : List Chain → Multiset Chain))
          (bagDraws K kept)
  | [], p => by
    simp only [List.foldl_nil, bagDraws, lift, PMF.pure_map, add_zero]
    exact (PMF.bind_pure _).symm
  | c :: kept, p => by
    rw [List.foldl_cons, foldl_transform_bag K kept]
    have hstep : (p.bind fun cs => (K c).map fun c' =>
          (c' :: cs).mergeSort Chain.le).map
          (Multiset.ofList : List Chain → Multiset Chain) =
        p.bind fun cs => (K c).map fun x => x ::ₘ (cs : Multiset Chain) := by
      simp only [PMF.map_bind, PMF.map_comp]
      refine congrArg _ (funext fun cs => congrArg (PMF.map · _)
        (funext fun x => ?_))
      exact Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _)
    rw [hstep]
    simp only [bagDraws, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      PMF.bind_bind, Function.comp_def]
    refine congrArg _ (funext fun cs => congrArg _ (funext fun y =>
      congrArg (PMF.map · _) (funext fun M => ?_)))
    rw [← Multiset.singleton_add, ← Multiset.singleton_add, add_assoc,
      add_left_comm]

/-- Transforming chains in place, each into normal chains, from normal chains,
leaves normal chains. -/
private theorem foldl_transform_normal (K : Chain → PMF Chain) :
    ∀ (kept : List Chain) (p : PMF (List Chain)),
      (∀ ch ∈ kept, ∀ x ∈ (K ch).support, NormalChain x) →
      (∀ cs ∈ p.support, ∀ x ∈ cs, NormalChain x) →
      ∀ cs ∈ (kept.foldl (fun (p : PMF (List Chain)) c => p.bind fun cs =>
          (K c).map fun c' => (c' :: cs).mergeSort Chain.le) p).support,
        ∀ x ∈ cs, NormalChain x
  | [], _, _, hp => hp
  | c :: kept, p, hK, hp => by
    rw [List.foldl_cons]
    refine foldl_transform_normal K kept _
      (fun ch h => hK ch (List.mem_cons_of_mem _ h)) ?_
    intro cs hcs x hx
    simp only [PMF.mem_support_bind_iff, PMF.mem_support_map_iff] at hcs
    obtain ⟨cs', hcs', y, hy, rfl⟩ := hcs
    rcases List.mem_cons.mp ((List.mergeSort_perm _ _).mem_iff.mp hx) with
      rfl | hx
    · exact hK c List.mem_cons_self x hy
    · exact hp cs' hcs' x hx

/--
A reroll of chains, each of which it rerolls into normal chains, leaves
independent draws, one for each chain, each its reroll. So the record that it
leaves depends only on the multiset of the chains, which `rerolled` in
`propagation/chain.rs` transforms one distinct chain at a time.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The law of the replacement of a rerolled result.

# Hypotheses
- `hn`: Every chain that the reroll leaves of a kept chain is normal.
-/
theorem rerollChains_eq_bag (die : Die) {kept : List Chain}
    (rerolled : Int → Bool) (P : PMF Int)
    (hn : ∀ ch ∈ kept, ∀ x ∈ (rerollChain rerolled P ch).support,
      NormalChain x) :
    rerollChains die kept rerolled P =
      (bagDraws (rerollChain rerolled P) kept).map
        fun M => Record.ofChains die M.toList := by
  have hb := foldl_transform_bag (rerollChain rerolled P) kept (PMF.pure [])
  simp only [PMF.pure_map, lift, PMF.pure_bind, Multiset.coe_nil,
    zero_add] at hb
  unfold rerollChains
  rw [map_congr_support (g := (fun M : Multiset Chain =>
      Record.ofChains die M.toList) ∘
        (Multiset.ofList : List Chain → Multiset Chain))
    fun cs hcs => ofChains_perm die (Multiset.coe_eq_coe.mp
      (Multiset.coe_toList _).symm)
      (foldl_transform_normal _ kept _ hn (by simp) cs hcs),
    ← PMF.map_comp, hb, show (fun M : Multiset Chain => M) = id from rfl,
    PMF.map_id]

/-- A sorted list of values is sorted. -/
theorem pairwise_sortBag (M : Multiset Int) : (sortBag M).Pairwise (· ≤ ·) := by
  have := List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
    (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
    (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
    M.toList
  simpa [sortBag] using this

/--
Every chain that a reroll leaves of a chain holds only values of a property
that every die of the chain and every replacement has, may still grow if the
chain may, and is normal if the chain is, whenever the property keeps values
within the `i32` range.

# Parameters
- `Q`: The property.
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The law of the replacement of a rerolled result.
- `ch`: The chain.
- `x`: A chain that the reroll leaves.

# Hypotheses
- `hP`: Every replacement has the property, if the reroll rerolls some die
  of the chain.
- `hch`: Every die of the chain has the property.
- `hx`: The reroll leaves `x`.
-/
theorem mem_support_rerollChain {Q : Int → Prop} {rerolled : Int → Bool}
    {P : PMF Int} {ch x : Chain}
    (hP : ∀ z ∈ ch.dice, rerolled z = true → ∀ y ∈ P.support, Q y)
    (hch : ∀ y ∈ ch.dice, Q y) (hx : x ∈ (rerollChain rerolled P ch).support) :
    (∀ y ∈ x.dice, Q y) ∧ x.saturates = ch.saturates ∧
      ((∀ y, Q y → i32Min ≤ y ∧ y ≤ i32Max) → NormalChain ch →
        NormalChain x) := by
  cases ch with
  | saturated v =>
    simp only [rerollChain, PMF.mem_support_pure_iff] at hx
    subst hx
    exact ⟨hch, rfl, fun _ h => h⟩
  | live body last =>
    have hdice : ∀ y ∈ last :: body, Q y := fun y hy =>
      hch y ((List.mergeSort_perm _ _).mem_iff.mpr hy)
    -- Every die drawn has the property.
    have hdraw : ∀ z ∈ last :: body, ∀ y ∈
        (if rerolled z then P else PMF.pure z).support, Q y :=
        fun z hz y hy => by
      split at hy
      · exact hP z ((List.mergeSort_perm _ _).mem_iff.mpr hz) ‹_› y hy
      · rw [PMF.mem_support_pure_iff] at hy
        exact hy ▸ hdice z hz
    rw [rerollChain_live] at hx
    simp only [PMF.mem_support_bind_iff, PMF.mem_support_map_iff] at hx
    obtain ⟨M, hM, y, hy, rfl⟩ := hx
    have hMQ : ∀ z ∈ sortBag M, Q z := fun z hz => by
      rw [← Multiset.mem_coe, coe_sortBag] at hz
      obtain ⟨w, hw, hz⟩ := mem_support_bagDraws hM z hz
      exact hdraw w (List.mem_cons_of_mem _ hw) z hz
    have hyQ := hdraw last List.mem_cons_self y hy
    have hall : ∀ z ∈ y :: sortBag M, Q z := fun z hz => by
      rcases List.mem_cons.mp hz with rfl | hz
      · exact hyQ
      · exact hMQ z hz
    refine ⟨fun z hz => hall z ((List.mergeSort_perm _ _).mem_iff.mp hz), rfl,
      fun hi _ => ⟨pairwise_sortBag M, fun z hz => hi z (hall z hz)⟩⟩


/-- Every record that a reroll of kept results reaches holds no chains. -/
private theorem chains_of_mem_support_rerollKept {die : Die}
    {kept : List Int} {rerolled : Int → Bool} {replacement : PMF Int}
    {r : Record} (h : r ∈ (rerollKept die kept rerolled replacement).support) :
    r.chains = [] := by
  have key : ∀ (xs : List Int) (p : PMF Record),
      (∀ r ∈ p.support, r.chains = []) →
      ∀ r ∈ (xs.foldl (fun p x =>
          if rerolled x then p.bind fun r => replacement.map r.push
          else p.map (·.push x)) p).support, r.chains = [] := by
    intro xs
    induction xs with
    | nil => exact fun p hp => hp
    | cons x xs ih =>
      intro p hp
      refine ih _ fun r hr => ?_
      dsimp only at hr
      split at hr
      · simp only [PMF.support_bind, PMF.support_map, Set.mem_iUnion,
          Set.mem_image] at hr
        obtain ⟨r', hr', _, _, rfl⟩ := hr
        exact hp r' hr'
      · simp only [PMF.support_map, Set.mem_image] at hr
        obtain ⟨r', hr', rfl⟩ := hr
        exact hp r' hr'
  exact key kept _ (fun r hr => by
    simp only [PMF.support_pure, Set.mem_singleton_iff] at hr
    subst hr
    rfl) r h

/-- A reroll of a chain of one die rerolls the die. -/
theorem rerollChain_single (rerolled : Int → Bool) (P : PMF Int) (x : Int) :
    rerollChain rerolled P (.single x) =
      (if rerolled x then P else PMF.pure x).map Chain.single := by
  rw [Chain.single, rerollChain_live]
  simp only [bagDraws, PMF.pure_bind, show sortBag 0 = [] by simp [sortBag]]
  rfl

/--
A reroll of each kept die of a record whose kept chains are normal and may
still grow, by replacements within the `i32` range, leaves, as a later reroll
or explode reads it, independent draws, one for each kept chain, each its
reroll, and the record's die, whether the record holds chains or not.

# Parameters
- `r`: The record.
- `rerolled`: Whether the reroll rerolls a result.
- `P`: The law of the replacement of a rerolled result.

# Hypotheses
- `hn`: Every kept chain is normal.
- `hl`: Every kept chain may still grow.
- `hP`: Every replacement lies within the `i32` range, if the reroll rerolls
  some kept die.
-/
theorem map_chainView_rerollEach {r : Record} {rerolled : Int → Bool}
    {P : PMF Int} (hn : ∀ ch ∈ r.keptChains, NormalChain ch)
    (hl : ∀ ch ∈ r.keptChains, ch.saturates = false)
    (hP : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice, rerolled z = true →
      ∀ y ∈ P.support, i32Min ≤ y ∧ y ≤ i32Max) :
    (rerollEach r rerolled P).map Record.chainView =
      (bagDraws (rerollChain rerolled P) r.keptChains).map
        fun M => (M, r.die) := by
  -- Every chain that the reroll leaves is normal.
  have hK : ∀ ch ∈ r.keptChains, ∀ x ∈ (rerollChain rerolled P ch).support,
      NormalChain x := fun ch hch x hx => by
    cases ch with
    | saturated v => exact absurd (hl _ hch) (by simp [Chain.saturates])
    | live body last =>
      have hnc := hn _ hch
      refine (mem_support_rerollChain (Q := fun y => i32Min ≤ y ∧ y ≤ i32Max)
        (hP _ hch) (fun y hy => hnc.2 y ((List.mergeSort_perm _ _).mem_iff.mp
          (by simpa only [Chain.dice] using hy)))
        hx).2.2 (fun _ h => h) hnc
  unfold rerollEach
  split
  · rename_i hc
    have hc' : r.chains = [] := List.isEmpty_iff.mp hc
    have hkc : r.keptChains = r.kept.map .single := by
      simp [Record.keptChains, hc']
    rw [hkc, bagDraws_map, show (rerollChain rerolled P ∘ Chain.single) =
        fun x => (if rerolled x then P else PMF.pure x).map Chain.single from
      funext (rerollChain_single rerolled P),
      ← bagDraws_map_map, ← map_bag_rerollKept, PMF.map_comp, PMF.map_comp]
    refine map_congr_support fun R hR => ?_
    obtain ⟨hl0, hh0, hd⟩ := mem_support_rerollKept hR
    have hRc := chains_of_mem_support_rerollKept hR
    simp only [Function.comp_apply, Record.chainView, Record.keptChains, hRc,
      List.isEmpty_nil, ↓reduceIte, hd, Prod.mk.injEq, and_true]
    have hk : R.kept = R.sorted :=
      congrArg Prod.fst (Record.view_of_undropped hl0 hh0)
    rw [hk, Record.sorted, ← Multiset.map_coe,
      Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _)]
  · rw [rerollChains_eq_bag _ _ _ hK, PMF.map_comp]
    refine map_congr_support fun M hM => ?_
    have hnM : ∀ x ∈ M.toList, NormalChain x := fun x hx => by
      obtain ⟨ch, hch, hx⟩ :=
      mem_support_bagDraws hM x (Multiset.mem_toList.mp hx)
      exact hK ch hch x hx
    simp only [Function.comp_apply, Record.chainView,
      keptChains_ofChains _ hnM, Multiset.coe_toList,
      (Record.ofChains_fields _ _).2.2.2]

/--
A reroll that may change the dice still kept rerolls each kept die of a record
that it matches by the final results of a matched die, whether the record
holds chains or not.

# Hypotheses
- `he`: The reroll may change the dice still kept, padding each by `G` and
  weighing the final results of a matched die by `g`.
-/
theorem reroll_eq_rerollEach {r : Record} {c : Criterion}
    {limit : Option Nat} {G : Nat} {g : List (Int × Nat)}
    (he : Effect.resolve r.die c limit = .redraw G g) :
    Spec.reroll r c limit = rerollEach r c.matches (weightedDie g) := by
  obtain ⟨hm, hcase⟩ := Effect.resolve_redraw he
  have hF : r.die.faces ≠ [] := fun h => hm (by simp [h])
  rcases hcase with ⟨n, rfl, hn, _, rfl⟩ | ⟨rfl, hmF, _, rfl⟩
  · simp only [reroll]
    rw [weightedDie_limitedLaw c n hF]
    congr 1
    funext x
    simp [Nat.pos_of_ne_zero hn]
  · have hw : ¬r.die.facesWithin c = true := by
      rw [Die.facesWithin, List.all_eq_true]
      exact fun h => hmF (List.countP_eq_length.mpr h)
    simp only [reroll]
    simp only [hw, Bool.false_eq_true, ↓reduceDIte]
    rw [weightedDie_unlimitedLaw hw]

/--
A reroll of a normal chain that draws each of its dice as itself leaves it.

# Hypotheses
- `hn`: The chain is normal.
- `h`: Each die of the chain draws itself.
-/
theorem rerollChain_pure {rerolled : Int → Bool} {P : PMF Int} {ch : Chain}
    (hn : NormalChain ch)
    (h : ∀ z ∈ ch.dice, (if rerolled z then P else PMF.pure z) = PMF.pure z) :
    rerollChain rerolled P ch = PMF.pure ch := by
  cases ch with
  | saturated v => rfl
  | live body last =>
    have hd : ∀ z ∈ last :: body,
        (if rerolled z then P else PMF.pure z) = PMF.pure z := fun z hz =>
      h z ((List.mergeSort_perm _ _).mem_iff.mpr hz)
    rw [rerollChain_live,
      bagDraws_pure fun z hz => hd z (List.mem_cons_of_mem _ hz), PMF.pure_bind,
      hd last List.mem_cons_self, PMF.pure_map]
    have hs : body.Pairwise (fun a b => decide (a ≤ b) = true) := by
      simpa using hn.1
    rw [← mergeSort_eq_sortBag, List.mergeSort_of_pairwise hs]

/-- A reroll of each kept die of a record that fits its die, each drawn as
itself, leaves the record, as a later reroll or explode reads it. -/
private theorem rerollEach_stays {r : Record} {D : Die}
    {rerolled : Int → Bool} {P : PMF Int} (hf : r.Fits D)
    (h : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice,
      (if rerolled z then P else PMF.pure z) = PMF.pure z) :
    (rerollEach r rerolled P).map Record.chainView = PMF.pure r.chainView := by
  have hi : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice, i32Min ≤ z ∧ z ≤ i32Max :=
    fun ch hch z hz => by
      have hnc := hf.normal ch hch
      cases ch with
      | saturated v =>
        exact absurd (hf.live _ hch) (by simp [Chain.saturates])
      | live body last =>
        exact hnc.2 z ((List.mergeSort_perm _ _).mem_iff.mp
          (by simpa only [Chain.dice] using hz))
  rw [map_chainView_rerollEach hf.normal hf.live fun ch hch z hz hr y hy => by
      have := h ch hch z hz
      simp only [hr, ↓reduceIte] at this
      rw [this, PMF.mem_support_pure_iff] at hy
      exact hy ▸ hi ch hch z hz,
    bagDraws_pure fun ch hch => rerollChain_pure (hf.normal ch hch) (h ch hch),
    PMF.pure_map]
  rfl

/--
A reroll that does nothing to the dice still kept leaves every record that
fits its die, as a later reroll or explode reads it, whether the record holds
chains or not.

# Hypotheses
- `hf`: The record fits the die.
- `he`: The reroll does nothing.
-/
theorem reroll_chainView_of_nothing {r : Record} {D : Die} {c : Criterion}
    {limit : Option Nat} (hf : r.Fits D)
    (he : Effect.resolve D c limit = .nothing) :
    (Spec.reroll r c limit).map Record.chainView = PMF.pure r.chainView := by
  have hd := hf.die
  rcases Effect.resolve_nothing he with hm | rfl | ⟨n, rfl, hF⟩
  · -- No face matches, so no kept die does.
    have hnm : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice, c.matches z = false :=
      fun ch hch z hz => by
        have := List.countP_eq_zero.mp hm z (hf.faces ch hch z hz)
        simpa using this
    cases limit with
    | none =>
      by_cases hw : r.die.facesWithin c
      · -- Every face matches, so there are none, and no kept chain.
        have hw' : D.facesWithin c := hd ▸ hw
        have hF : D.faces = [] := by
          refine List.eq_nil_iff_forall_not_mem.mpr fun z hz => ?_
          have h1 := List.all_eq_true.mp hw' z hz
          have h2 := List.countP_eq_zero.mp hm z hz
          simp_all
        have hk : r.keptChains = [] := by
          refine List.eq_nil_iff_forall_not_mem.mpr fun ch hch => ?_
          cases ch with
          | saturated v =>
            exact absurd (hf.live _ hch) (by simp [Chain.saturates])
          | live body last =>
            have := hf.faces _ hch last
              ((List.mergeSort_perm _ _).mem_iff.mpr List.mem_cons_self)
            rw [hF] at this
            cases this
        have hkr : r.kept = [] := by
          have := hf.kept
          rw [hk] at this
          simpa using this
        have hv : r.chainView = (0, r.die) := by
          simp [Record.chainView, hk]
        simp only [reroll, hw, ↓reduceDIte]
        rw [hv]
        split
        · rw [PMF.pure_map, hkr]
          simp [Record.chainView, Record.keptChains, Record.kept,
            Record.sorted]
        · rw [PMF.pure_map, hk]
          simp [Record.chainView, keptChains_ofChains _
            (fun x (hx : x ∈ []) => absurd hx (by simp)),
            (Record.ofChains_fields _ _).2.2.2]
      · simp only [reroll, hw]
        exact rerollEach_stays hf fun ch hch z hz => by simp [hnm ch hch z hz]
    | some n =>
      simp only [reroll]
      exact rerollEach_stays hf fun ch hch z hz => by simp [hnm ch hch z hz]
  · simp only [reroll]
    exact rerollEach_stays hf fun ch hch z hz => by simp
  · obtain ⟨y, hy⟩ := List.length_eq_one_iff.mp hF
    simp only [reroll]
    refine rerollEach_stays hf fun ch hch z hz => ?_
    have hzy : z = y := by
      have := hf.faces ch hch z hz
      rw [hy, List.mem_singleton] at this
      exact this
    rw [hd, hy, customDie_singleton, limitedReplacement, repeat_redraw_pure,
      hzy]
    simp

/--
An unlimited reroll that matches every face of the die crosses out every kept
chain of a record that fits the die, whether the record holds chains or not.

# Hypotheses
- `hf`: The record fits the die.
- `hw`: The criterion matches every face of the die.
-/
theorem reroll_chainView_of_crossOut {r : Record} {D : Die} {c : Criterion}
    (hf : r.Fits D) (hw : D.facesWithin c) :
    (Spec.reroll r c none).map Record.chainView = PMF.pure (0, D) := by
  have hd := hf.die
  simp only [reroll, hd ▸ hw, ↓reduceDIte]
  split
  · rename_i hc
    have hc' : r.chains = [] := List.isEmpty_iff.mp hc
    have hkc : r.keptChains = r.kept.map .single := by
      simp [Record.keptChains, hc']
    have : r.kept.filter (!c.matches ·) = [] := by
      refine List.filter_eq_nil_iff.mpr fun x hx => ?_
      have hx' := hf.faces (.single x) (hkc ▸ List.mem_map_of_mem hx) x
        (by simp [Chain.single, Chain.dice])
      have := List.all_eq_true.mp hw x hx'
      simp [this]
    rw [PMF.pure_map, this]
    simp [Record.chainView, Record.keptChains, Record.kept, Record.sorted, hd]
  · have : r.keptChains.filter (·.saturates) = [] :=
      List.filter_eq_nil_iff.mpr fun ch hch => by simp [hf.live ch hch]
    rw [PMF.pure_map, this]
    simp [Record.chainView, keptChains_ofChains _
      (fun x (hx : x ∈ []) => absurd hx (by simp)),
      (Record.ofChains_fields _ _).2.2.2, hd]

/-- Every record that a reroll of a record without chains reaches holds no
chains. -/
theorem chains_of_mem_support_reroll {r : Record} {c : Criterion}
    {limit : Option Nat} (hc : r.chains = []) {R : Record}
    (h : R ∈ (Spec.reroll r c limit).support) : R.chains = [] := by
  have hce : r.chains.isEmpty = true := by simp [hc]
  cases limit with
  | none =>
    simp only [reroll] at h
    split at h
    · simp only [PMF.mem_support_pure_iff] at h
      subst h
      rfl
    · simp only [rerollEach, hce, ↓reduceIte] at h
      exact chains_of_mem_support_rerollKept h
  | some n =>
    simp only [reroll, rerollEach, hce, ↓reduceIte] at h
    exact chains_of_mem_support_rerollKept h

/-- A record without chains is read by a later reroll or explode as the
chains of one die of its kept results. -/
theorem chainView_of_plain {r : Record} (hc : r.chains = []) :
    r.chainView = (((r.view.1.map Chain.single : List Chain) : Multiset Chain),
      r.view.2) := by
  simp [Record.chainView, Record.keptChains, hc, Record.view]

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

/--
A record without chains, of faces of a die within the `i32` range, fits the
die.

# Parameters
- `D`: The die.
- `r`: The record.

# Hypotheses
- `hD`: The faces of the die lie within the `i32` range.
- `hc`: The record holds no chains.
- `hd`: The record holds the die.
- `hf`: Every kept result is a face of the die.
-/
theorem fits_of_plain {D : Die} {r : Record}
    (hD : ∀ x ∈ D.faces, i32Min ≤ x ∧ x ≤ i32Max) (hc : r.chains = [])
    (hd : r.die = D) (hf : ∀ x ∈ r.kept, x ∈ D.faces) : r.Fits D := by
  have hk : r.keptChains = r.kept.map .single := by
    simp [Record.keptChains, hc]
  refine ⟨hd, fun ch hch => ?_, fun ch hch => ?_, fun ch hch y hy => ?_, ?_⟩
  · rw [hk] at hch
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
    refine ⟨.nil, fun y hy => ?_⟩
    rw [List.mem_singleton.mp hy]
    exact hD x (hf x hx)
  · rw [hk] at hch
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
    rfl
  · rw [hk] at hch
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hch
    simp only [Chain.single, Chain.dice, List.mergeSort_singleton,
      List.mem_singleton] at hy
    rw [hy]
    exact hf x hx
  · rw [hk, List.map_map]
    refine congrArg _ ((List.map_congr_left fun x hx => ?_).trans
      (List.map_id _)).symm
    exact total_single (hD x (hf x hx))

/--
A record that fits its die is viewed by the sorted totals of its kept chains:
every later drop, keep, and sum reads what a later reroll or explode reads,
reduced to the totals.

# Hypotheses
- `h`: The record fits the die.
-/
theorem view_of_fits {D : Die} {r : Record} (h : r.Fits D) :
    r.view = (sortBag (r.chainView.1.map Chain.total), r.chainView.2) := by
  rw [← h.views.2, view_ofChains]
  simp [Record.chainView, h.die]

/--
Two laws of records that fit a die, alike as a later reroll or explode reads
them, are alike as every later drop, keep, and sum reads them.

# Hypotheses
- `hX`, `hY`: Every record of each law fits the die.
- `h`: The laws are alike as a later reroll or explode reads them.
-/
theorem map_view_of_chainView {D : Die} {X Y : PMF Record}
    (hX : ∀ r ∈ X.support, r.Fits D) (hY : ∀ r ∈ Y.support, r.Fits D)
    (h : X.map Record.chainView = Y.map Record.chainView) :
    X.map Record.view = Y.map Record.view := by
  let F : Multiset Chain × Die → List Int × Die := fun v =>
    (sortBag (v.1.map Chain.total), v.2)
  have hF : ∀ (Z : PMF Record), (∀ r ∈ Z.support, r.Fits D) →
      Z.map Record.view = (Z.map Record.chainView).map F := fun Z hZ => by
    rw [PMF.map_comp]
    exact map_congr_support fun r hr => view_of_fits (hZ r hr)
  rw [hF X hX, hF Y hY, h]

/--
Every record that a reroll of each kept die of a record that fits a die
leaves fits the die, if every replacement of a die that it rerolls is a face
of the die.

# Hypotheses
- `hf`: The record fits the die.
- `hD`: The faces of the die lie within the `i32` range.
- `hP`: Every replacement is a face of the die, if the reroll rerolls some
  kept die.
- `hR`: The reroll leaves `R`.
-/
theorem fits_rerollEach {r R : Record} {D : Die} {rerolled : Int → Bool}
    {P : PMF Int} (hf : r.Fits D)
    (hD : ∀ x ∈ D.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hP : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice, rerolled z = true →
      ∀ y ∈ P.support, y ∈ D.faces)
    (hR : R ∈ (rerollEach r rerolled P).support) : R.Fits D := by
  unfold rerollEach at hR
  split at hR
  · rename_i hc
    have hc' : r.chains = [] := List.isEmpty_iff.mp hc
    have hkc : r.keptChains = r.kept.map .single := by
      simp [Record.keptChains, hc']
    obtain ⟨hl0, hh0, hd⟩ := mem_support_rerollKept hR
    have hM : (R.results : Multiset Int) ∈ (bagDraws
        (fun x => if rerolled x then P else PMF.pure x) r.kept).support := by
      rw [← map_bag_rerollKept]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨R, hR, rfl⟩
    refine fits_of_plain hD (chains_of_mem_support_rerollKept hR)
      (hd.trans hf.die) fun y hy => ?_
    have hy' : y ∈ R.results := (List.mergeSort_perm _ _).mem_iff.mp
      (List.mem_of_mem_drop (List.mem_of_mem_take hy))
    obtain ⟨x, hx, hyx⟩ := mem_support_bagDraws hM y hy'
    have hxf : x ∈ D.faces :=
      hf.faces (.single x) (hkc ▸ List.mem_map_of_mem hx) x
        (by simp [Chain.single, Chain.dice])
    split at hyx
    · exact hP (.single x) (hkc ▸ List.mem_map_of_mem hx) x
        (by simp [Chain.single, Chain.dice]) ‹_› y hyx
    · rw [PMF.mem_support_pure_iff] at hyx
      exact hyx ▸ hxf
  · have hK : ∀ ch ∈ r.keptChains, ∀ x ∈ (rerollChain rerolled P ch).support,
        NormalChain x ∧ x.saturates = false ∧ ∀ y ∈ x.dice, y ∈ D.faces :=
      fun ch hch x hx => by
        have h := mem_support_rerollChain (Q := fun y => y ∈ D.faces)
          (hP ch hch) (hf.faces ch hch) hx
        exact ⟨h.2.2 hD (hf.normal ch hch), h.2.1.trans (hf.live ch hch), h.1⟩
    rw [rerollChains_eq_bag _ _ _ fun ch hch x hx => (hK ch hch x hx).1] at hR
    obtain ⟨M, hM, rfl⟩ := (PMF.mem_support_map_iff _ _ _).mp hR
    rw [← hf.die]
    refine hf.die ▸ fits_ofChains _ fun x hx => ?_
    obtain ⟨ch, hch, hx⟩ :=
      mem_support_bagDraws hM x (Multiset.mem_toList.mp hx)
    exact hf.die ▸ hK ch hch x hx

/-- Every replacement of a limited reroll from a custom die of faces is one of
them. -/
private theorem mem_faces_limitedReplacement {c : Criterion} {F : List Int}
    (hF : F ≠ []) (n : Nat) {y : Int}
    (hy : y ∈ (limitedReplacement c (customDie F) n).support) : y ∈ F := by
  have hd : ∀ y ∈ (customDie F).support, y ∈ F := fun y hy => by
    rcases mem_of_mem_support_customDie hy with hy | ⟨h, _⟩
    · exact hy
    · exact absurd h hF
  have key : ∀ k : Nat, ∀ y ∈ (k.repeat (redraw c (customDie F))
      (customDie F)).support, y ∈ F := by
    intro k
    induction k with
    | zero => exact hd
    | succ k ih =>
      intro y hy
      simp only [Nat.repeat, redraw, PMF.mem_support_bind_iff] at hy
      obtain ⟨x, hx, hy⟩ := hy
      split at hy
      · exact hd y hy
      · rw [PMF.mem_support_pure_iff] at hy
        exact hy ▸ ih x hx
  exact key _ y hy

/--
Every record that the specification's reroll of a record that fits its die
leaves fits the die, if the die's faces lie within the `i32` range.

# Hypotheses
- `hf`: The record fits the die.
- `hD`: The faces of the die lie within the `i32` range.
- `hR`: The reroll leaves `R`.
-/
theorem fits_reroll {r R : Record} {D : Die} {c : Criterion}
    {limit : Option Nat} (hf : r.Fits D)
    (hD : ∀ x ∈ D.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hR : R ∈ (Spec.reroll r c limit).support) : R.Fits D := by
  have hd := hf.die
  -- Every kept die is a face of the die.
  have hkf : ∀ ch ∈ r.keptChains, ∀ z ∈ ch.dice, z ∈ D.faces := hf.faces
  cases limit with
  | none =>
    simp only [reroll] at hR
    split at hR
    · split at hR
      · rename_i hc
        have hc' : r.chains = [] := List.isEmpty_iff.mp hc
        have hkc : r.keptChains = r.kept.map .single := by
          simp [Record.keptChains, hc']
        rw [PMF.mem_support_pure_iff] at hR
        subst hR
        refine fits_of_plain hD rfl hd fun y hy => ?_
        have hy' : y ∈ r.kept.filter (!c.matches ·) :=
          (List.mergeSort_perm _ _).mem_iff.mp
            (List.mem_of_mem_drop (List.mem_of_mem_take hy))
        have hy'' := (List.mem_filter.mp hy').1
        exact hkf (.single y) (hkc ▸ List.mem_map_of_mem hy'') y
          (by simp [Chain.single, Chain.dice])
      · rw [PMF.mem_support_pure_iff] at hR
        subst hR
        rw [← hd]
        refine hd ▸ fits_ofChains D fun x hx => ?_
        have hx' := (List.mem_filter.mp hx).1
        exact ⟨hf.normal x hx', hf.live x hx', hf.faces x hx'⟩
    · refine fits_rerollEach hf hD (fun _ _ _ _ _ y hy => ?_) hR
      simp only [PMF.mem_support_ofMultiset_iff, Multiset.mem_toFinset,
        Multiset.mem_coe] at hy
      exact hd ▸ (List.mem_filter.mp hy).1
  | some n =>
    simp only [reroll] at hR
    refine fits_rerollEach hf hD (fun ch hch z hz _ y hy => ?_) hR
    have hF : r.die.faces ≠ [] := fun h => by
      have := hkf ch hch z hz
      rw [← hd, h] at this
      cases this
    exact hd ▸ mem_faces_limitedReplacement hF n hy

/-! ## Rerolling a setup's chains -/

namespace Setup

/--
Reroll a setup as `Mixture::reroll` in `propagation/mixture.rs` does: reroll
it, and reduce its chains to their totals unless they stay whole.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.
- `whole`: Whether a later reroll or explode reads the record before a roll
  refills it, so that its chains stay whole.

# Returns
The rerolled setup.
-/
def rerollAs (s : Setup) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Setup :=
  if whole then s.reroll c limit else (s.reroll c limit).collapse

/--
Whether rerolling a setup, as `Mixture::reroll` in `propagation/mixture.rs`
does, describes the specification's reroll of the records that it describes,
as every later instruction reads them: if the chains stay whole, up to their
chain view, the kept chains and the die, which a later reroll or explode
reads; and if they reduce to their totals, up to their view, the kept results
and the die, which every later drop, keep, and sum reads.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.
- `whole`: Whether the chains stay whole.

# Returns
Whether the rerolled setup's law is the reroll's, viewed.
-/
def RerollsAs (s : Setup) (c : Criterion) (limit : Option Nat) (whole : Bool) :
    Prop :=
  if whole then
    (s.rerollAs c limit true).law.map Record.chainView =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView
  else
    (s.rerollAs c limit false).law.map Record.view =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.view

/-- Every record of a grounded setup without chains is faithful to its die. -/
theorem faithful_of_grounded {s : Setup} (hg : s.Grounded) (hp : s.Plain) :
    s.Faithful := by
  intro r hr x hx
  cases hρ : s.roll with
  | none =>
    simp only [law, hρ, Option.map_none, Option.getD_none, PMF.support_map,
      PMF.support_pure, Set.image_singleton, Set.mem_singleton_iff] at hr
    subst hr
    cases hx
  | some ρ =>
    have hgρ := hg.2 ρ hρ
    have hpρ := hp ρ hρ
    cases ρ with
    | chains | stacked => exact hpρ.elim
    | range | standard | custom | fixed | pool => exact hgρ r hr x hx

/-- A reroll of a grounded setup without chains describes the specification's
reroll of its records, as a later reroll or explode reads them. -/
private theorem rerolls_whole_plain {s : Setup} (c : Criterion)
    (limit : Option Nat) (hs : s.WellFormed) (hg : s.Grounded)
    (hfaced : s.Faced) (hp : s.Plain) :
    (s.reroll c limit).law.map Record.chainView =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView := by
  let G : List Int × Die → Multiset Chain × Die := fun v =>
    (((v.1.map Chain.single : List Chain) : Multiset Chain), v.2)
  have hp' := (reroll_invariants_plain c limit hs
    (faithful_of_grounded hg hp) hfaced hp).2.2.2
  have hl : (s.reroll c limit).law.map Record.chainView =
      ((s.reroll c limit).law.map Record.view).map G := by
    rw [PMF.map_comp]
    exact map_congr_support fun R hR =>
      chainView_of_plain (Setup.chains_of_mem_support_law hp' hR)
  have hr : (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView =
      ((s.law.bind fun r => Spec.reroll r c limit).map Record.view).map G := by
    rw [PMF.map_comp]
    refine map_congr_support fun R hR => ?_
    obtain ⟨r, hr, hR⟩ := (PMF.mem_support_bind_iff _ _ _).mp hR
    exact chainView_of_plain (chains_of_mem_support_reroll
      (Setup.chains_of_mem_support_law hp hr) hR)
  rw [hl, hr,
    law_reroll_plain c limit hs (faithful_of_grounded hg hp) hfaced hp]

/-- Every final result of a matched die that a reroll draws is a face of the
die. -/
theorem mem_faces_of_redraw {D : Die} {c : Criterion} {limit : Option Nat}
    {G : Nat} {g : List (Int × Nat)}
    (he : Effect.resolve D c limit = .redraw G g) {y : Int}
    (hy : y ∈ (weightedDie g).support) : y ∈ D.faces := by
  obtain ⟨hglen, hG, hgf⟩ := redraw_law he
  rcases mem_of_mem_support_customDie hy with hy | ⟨h0, _⟩
  · exact hgf y hy
  · exact absurd (by rw [← hglen, Roll.expand]; simp [h0]) hG

/-- The specification rerolls each record of a grounded setup, by a reroll that
may change its dice still kept, as independent draws of the reroll of each of
its kept chains. -/
private theorem reroll_chainView_of_redraw {s : Setup} {c : Criterion}
    {limit : Option Nat} {G : Nat} {g : List (Int × Nat)} (hs : s.WellFormed)
    (hg : s.Grounded) (he : Effect.resolve s.replacement c limit = .redraw G g)
    {r : Record} (hr : r ∈ s.law.support) :
    (Spec.reroll r c limit).map Record.chainView =
      (bagDraws (rerollChain c.matches (weightedDie g)) r.keptChains).map
        fun M => (M, s.replacement) := by
  have hf := fits_of_grounded hs hg hr
  rw [reroll_eq_rerollEach (hf.die ▸ he), map_chainView_rerollEach hf.normal
    hf.live fun _ _ _ _ _ y hy => hg.1 y (mem_faces_of_redraw he hy), hf.die]

/-- The specification's reroll of every record of a grounded setup that keeps
no chain leaves it, as a later reroll or explode reads it. -/
private theorem reroll_chainView_of_empty {s : Setup} {c : Criterion}
    {limit : Option Nat} (hs : s.WellFormed) (hg : s.Grounded)
    (hk : s.results - s.lowest - s.highest = 0) {r : Record}
    (hr : r ∈ s.law.support) :
    (Spec.reroll r c limit).map Record.chainView = PMF.pure r.chainView := by
  have hf := fits_of_grounded hs hg hr
  have hkc : r.keptChains = [] := List.eq_nil_of_length_eq_zero
    ((length_keptChains hs hg hr).trans hk)
  have hv : r.chainView = (0, s.replacement) := by
    simp [Record.chainView, hkc, hf.die]
  cases he : Effect.resolve s.replacement c limit with
  | nothing => exact reroll_chainView_of_nothing hf he
  | crossOut =>
    obtain ⟨rfl, hm⟩ := Effect.resolve_crossOut he
    rw [hv]
    exact reroll_chainView_of_crossOut hf (by
      rw [Die.facesWithin, List.all_eq_true]
      exact List.countP_eq_length.mp hm)
  | redraw G g =>
    rw [reroll_chainView_of_redraw hs hg he hr, hkc, hv]
    simp [bagDraws, PMF.pure_map]

/-- A reroll of a grounded setup of i.i.d. chains describes the
specification's reroll of its records, as a later reroll or explode reads
them, if it keeps no chain, does nothing, or crosses out every chain. -/
private theorem rerolls_whole_stays {s : Setup} {c : Criterion}
    {limit : Option Nat} (hst : s.reroll c limit = s)
    (h : ∀ r ∈ s.law.support,
      (Spec.reroll r c limit).map Record.chainView = PMF.pure r.chainView) :
    (s.reroll c limit).law.map Record.chainView =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView := by
  rw [hst, PMF.map_bind, bind_congr_support h]
  exact (PMF.bind_pure_comp _ _).symm

/-- The final results of a matched die weigh the padding in all. -/
theorem sum_of_redraw {D : Die} {c : Criterion} {limit : Option Nat}
    {G : Nat} {g : List (Int × Nat)}
    (he : Effect.resolve D c limit = .redraw G g) : (g.map (·.2)).sum = G := by
  rw [← (redraw_law he).1]
  simp [Roll.expand, List.length_flatMap]

/-- A law of chains, each transformed by its reroll and padded, is the law of
each chain's reroll. -/
theorem weighted_transformLaw_rerolled {c : Criterion} {G : Nat}
    {g : List (Int × Nat)} (hg : (g.map (·.2)).sum = G) (hG : G ≠ 0)
    {cl : List (Chain × Nat)} (hW : (cl.map (·.2)).sum ≠ 0) :
    weighted (.single 0) (transformLaw cl G (rerolled c G g)) =
      (weighted (.single 0) cl).bind
        (rerollChain c.matches (weightedDie g)) := by
  set most := (cl.map fun e => (rerolled c G g e.1).2).foldl Max.max 0
  have h := weighted_flatMap (Chain.single 0) (Chain.single 0) id cl
    (fun ch => (rerolled c G g ch).1)
    (fun ch => G ^ (most - (rerolled c G g ch).2)) (fun _ o => o) (G ^ most)
    (pow_ne_zero _ hG) hW fun e he _ => by
      rw [sum_rerolled hg, ← pow_add, Nat.sub_add_cancel
        (le_foldl_max _ 0 _ (List.mem_map_of_mem (f := fun e =>
          (rerolled c G g e.1).2) he))]
  rw [transformLaw, h]
  refine congrArg _ (funext fun ch => ?_)
  rw [show (fun o : Chain => o) = id from rfl, PMF.map_id, id,
    weighted_rerolled hg hG]

/-- A reroll of a grounded setup of i.i.d. chains, none dropped, that may
change its chains still kept, describes the specification's reroll of its
records, as a later reroll or explode reads them. -/
private theorem rerolls_whole_iid {s : Setup} {c : Criterion}
    {limit : Option Nat} {count : Int} {cl : List (Chain × Nat)} {G : Nat}
    {g : List (Int × Nat)} (hs : s.WellFormed) (hg : s.Grounded)
    (hfaced : s.Faced) (hρ : s.roll = some (.chains count cl))
    (hk : s.results - s.lowest - s.highest ≠ 0) (hl : s.lowest = 0)
    (hh : s.highest = 0)
    (he : Effect.resolve s.replacement c limit = .redraw G g) :
    (s.reroll c limit).law.map Record.chainView =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView := by
  let K := rerollChain c.matches (weightedDie g)
  let P := weighted (.single 0) cl
  have hG := (redraw_law he).2.1
  have hgG := sum_of_redraw he
  obtain ⟨hne, hpos⟩ := hs _ hρ
  have hgρ := (hg.2 _ hρ).2
  have hW : (cl.map (·.2)).sum ≠ 0 := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
    have := hpos e he
    intro h0
    have := List.sum_eq_zero_iff.mp h0 e.2 (List.mem_map_of_mem he)
    omega
  have hs' : s.reroll c limit = { roll := some (.chains count
      (transformLaw cl G (rerolled c G g))), die := some s.replacement } := by
    have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl, hh] at hk
    simp only [reroll, hρ, hk', he, Roll.alike, Bool.false_eq_true, false_and,
      ↓reduceIte, Roll.chained, Roll.rerollChains, hl, hh, and_self]
  -- Every chain that the reroll leaves is normal.
  have hKn : ∀ x ∈ (P.bind K).support, NormalChain x := fun x hx => by
    obtain ⟨ch, hch, hx⟩ := (PMF.mem_support_bind_iff _ _ _).mp hx
    obtain ⟨e, he', rfl⟩ := mem_of_mem_support_weighted hW hch
    obtain ⟨hn, _, hfc⟩ := hgρ e he'
    exact (mem_support_rerollChain (Q := fun y => y ∈ s.replacement.faces)
      (fun _ _ _ y hy => mem_faces_of_redraw he hy) hfc hx).2.2
      (fun y hy => hg.1 y hy) hn
  have hL : (s.reroll c limit).law.map Record.chainView =
      (iidBag count.toNat (P.bind K)).map fun M => (M, s.replacement) := by
    rw [hs', law_chains, weighted_transformLaw_rerolled hgG hG hW, PMF.map_comp,
      ← map_coe_chainDraws, PMF.map_comp]
    refine map_congr_support fun cs hcs => ?_
    simp only [Function.comp_apply, Record.chainView,
      (Record.ofChains_fields _ _).2.2.2, keptChains_ofChains _
        fun x hx => hKn x ((mem_support_chainDraws hcs).2 x hx)]
  rw [hL, PMF.map_bind,
    bind_congr_support fun r hr => reroll_chainView_of_redraw hs hg he hr,
    ← PMF.map_bind, bind_bagDraws_keptChains hs hg hfaced hρ rfl hl hh hk K]
  rfl

/-- A reroll of a grounded setup of i.i.d. chains, some dropped, that may
change its chains still kept, describes the specification's reroll of its
records, as a later reroll or explode reads them. -/
private theorem rerolls_whole_dropped {s : Setup} {c : Criterion}
    {limit : Option Nat} {count : Int} {cl : List (Chain × Nat)} {G : Nat}
    {g : List (Int × Nat)} (hs : s.WellFormed) (hg : s.Grounded)
    (hρ : s.roll = some (.chains count cl))
    (hk : s.results - s.lowest - s.highest ≠ 0)
    (hd : ¬(s.lowest = 0 ∧ s.highest = 0))
    (he : Effect.resolve s.replacement c limit = .redraw G g) :
    (s.reroll c limit).law.map Record.chainView =
        (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView ∧
      ∀ R ∈ (s.reroll c limit).law.support, R.Fits s.replacement := by
  set ρ : Roll := .chains count cl with hρdef
  let K := rerollChain c.matches (weightedDie g)
  let rer := rerolled c G g
  have hG := (redraw_law he).2.1
  have hgG := sum_of_redraw he
  have hw := hs ρ hρ
  have hgρ := (hg.2 _ hρ).2
  have hn : ρ.Normal := fun e he => (hgρ e he).1
  set kp := keptPool ρ.stacks s.lowest s.highest with hkp
  have hkc := map_keptChains_law hρ hw hn
  have hWs := Roll.sum_stacks_ne_zero hw
  have hW : (kp.map (·.2)).sum ≠ 0 := by
    simpa [hkp, keptPool, Function.comp_def] using hWs
  -- Every chain still kept of the pool is a normal chain of faces of the die,
  -- which may still grow.
  have hpool : ∀ ms ∈ (weighted [] kp).support, ∀ ch ∈ Roll.expand ms,
      NormalChain ch ∧ ch.saturates = false ∧
        ∀ x ∈ ch.dice, x ∈ s.replacement.faces := by
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
    exact ⟨hf.normal ch hch', hf.live ch hch', hf.faces ch hch'⟩
  -- The specification's rerolls, as the multisets of the chains still kept
  -- weigh them.
  have hR : (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView =
      (weighted [] kp).bind fun ms =>
        (bagDraws K (Roll.expand ms)).map fun M => (M, s.replacement) := by
    have hperm : ∀ l : List Chain,
        bagDraws K ((l : Multiset Chain)).toList = bagDraws K l := fun l =>
      bagDraws_perm K (Multiset.coe_eq_coe.mp (Multiset.coe_toList _))
    rw [PMF.map_bind,
      bind_congr_support fun r hr => reroll_chainView_of_redraw hs hg he hr]
    calc _ = (s.law.map fun r => (r.keptChains : Multiset Chain)).bind
            fun M => (bagDraws K M.toList).map fun M => (M, s.replacement) := by
          rw [PMF.bind_map]
          exact congrArg _ (funext fun r => by
            simp only [Function.comp_apply, hperm]
            rfl)
      _ = _ := by
          rw [hkc, PMF.bind_map]
          exact congrArg _ (funext fun ms => by
            simp only [Function.comp_apply, hperm])
  -- The stacks that the reroll leaves.
  let dice := fun (chains : List (Chain × Nat)) =>
    (chains.map fun g => (rer g.1).2 * g.2).sum
  let most := (kp.map (dice ·.1)).foldl Max.max 0
  let laws := (kp.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
    (ch, (rer ch).1)
  let pool := kp.map fun e => (e.1, e.2 * G ^ (most - dice e.1))
  have hs' : s.reroll c limit =
      { roll := some (Roll.stacked pool laws), die := some s.replacement } := by
    have hk' : ¬s.results - s.lowest - s.highest = 0 := hk
    have hal : ρ.alike = false := by rw [hρdef]; rfl
    have hch : ρ.chained = true := by rw [hρdef]; rfl
    simp only [reroll, hρ, hk', he, hal, Bool.false_eq_true, false_and,
      ↓reduceIte, hch, Roll.rerollChains, hd, ← hkp, transformStacks]
    rfl
  -- What each chain of the pool has become is its reroll.
  have hin : ∀ e ∈ kp, ∀ p ∈ e.1,
      p.1 ∈ (kp.flatMap fun e => e.1.map (·.1)).dedup := fun e he p hp =>
    List.mem_dedup.mpr (List.mem_flatMap.mpr ⟨e, he, List.mem_map_of_mem hp⟩)
  have hbec : ∀ ms ∈ (weighted [] kp).support, ∀ ch ∈ Roll.expand ms,
      became laws ch = K ch := by
    intro ms hms ch hch
    obtain ⟨e, he, rfl⟩ := mem_of_mem_support_weighted hW hms
    obtain ⟨p, hp, hpx⟩ := List.mem_flatMap.mp hch
    rw [List.eq_of_mem_replicate hpx]
    simp only [became, laws, lookup_graph (fun ch => (rer ch).1) _ p.1
      (hin e he p hp)]
    exact weighted_rerolled hgG hG _ _
  -- Every multiset weighs alike, padded and joined.
  have hweights : weighted ((pool.head?.map (·.1)).getD [])
      (pool.map fun e => (e.1, e.2 * stackWeight laws e.1)) =
        weighted [] kp := by
    have hmap : (pool.map fun e => (e.1, e.2 * stackWeight laws e.1)) =
        kp.map fun e =>
          (e.1, e.2 * (G ^ (most - dice e.1) * stackWeight laws e.1)) := by
      simp [pool, mul_assoc]
    rw [hmap, weighted_scale _ kp
        (fun ms => G ^ (most - dice ms) * stackWeight laws ms)
        (pow_ne_zero most hG) fun e he => by
      rw [stackWeight_of_pow (f := fun ch => (rer ch).2) fun p hp => by
          simp only [laws, lookup_graph (fun ch => (rer ch).1) _ p.1
            (hin e he p hp), Option.map_some, Option.getD_some]
          exact sum_rerolled hgG p.1,
        ← pow_add, Nat.sub_add_cancel (le_foldl_max _ 0 _
          (List.mem_map_of_mem (f := fun e => dice e.1) he))],
      weighted_congr_default _ [] hW]
  have hlaw : (s.reroll c limit).law = ((weighted [] kp).bind fun ms =>
      (stackDraws laws ms).map (Record.ofChains .none)).map fun r =>
        { r with lowest := 0, highest := 0, die := s.replacement } := by
    rw [hs']
    simp only [law, Option.map_some, Option.getD_some, Roll.law]
    rw [hweights]
    rfl
  -- Every chain that the stacks draw is a normal chain of faces of the die,
  -- which may still grow.
  have hcs : ∀ ms ∈ (weighted [] kp).support,
      ∀ cs ∈ (stackDraws laws ms).support, ∀ x ∈ cs,
        NormalChain x ∧ x.saturates = false ∧
          ∀ y ∈ x.dice, y ∈ s.replacement.faces := fun ms hms cs hcs x hx => by
    have hM : (cs : Multiset Chain) ∈
        (bagDraws (became laws) (Roll.expand ms)).support := by
      rw [← map_coe_stackDraws]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨cs, hcs, rfl⟩
    obtain ⟨ch, hch, hx⟩ := mem_support_bagDraws hM x hx
    rw [hbec ms hms ch hch] at hx
    obtain ⟨hnc, hlc, hfc⟩ := hpool ms hms ch hch
    have h := mem_support_rerollChain (Q := fun y => y ∈ s.replacement.faces)
      (fun _ _ _ y hy => mem_faces_of_redraw he hy) hfc hx
    exact ⟨h.2.2 (fun y hy => hg.1 y hy) hnc, h.2.1.trans hlc, h.1⟩
  refine ⟨?_, fun R hR => ?_⟩
  · rw [hlaw, hR, PMF.map_comp, PMF.map_bind]
    refine bind_congr_support fun ms hms => ?_
    rw [← bagDraws_congr (hbec ms hms), ← map_coe_stackDraws, PMF.map_comp,
      PMF.map_comp]
    refine map_congr_support fun cs hcs' => ?_
    simp only [Function.comp_apply]
    rw [ofChains_with]
    simp only [Record.chainView, keptChains_ofChains _
      fun x hx => (hcs ms hms cs hcs' x hx).1,
      (Record.ofChains_fields _ _).2.2.2]
  · rw [hlaw] at hR
    simp only [PMF.mem_support_map_iff, PMF.mem_support_bind_iff] at hR
    obtain ⟨_, ⟨ms, hms, cs, hcs', rfl⟩, rfl⟩ := hR
    rw [ofChains_with]
    exact fits_ofChains _ (hcs ms hms cs hcs')

/-- What a chain becomes after a reroll that weighs the final results of a
matched die positively, and pads positively, is something, each of positive
weight. -/
theorem rerolled_pos {c : Criterion} {G : Nat} {g : List (Int × Nat)}
    (hg : ∀ f ∈ g, 0 < f.2) (hgne : g ≠ []) (hG : G ≠ 0) (ch : Chain) :
    (rerolled c G g ch).1 ≠ [] ∧ ∀ o ∈ (rerolled c G g ch).1, 0 < o.2 := by
  cases ch with
  | saturated v => simp [rerolled]
  | live body last =>
    have hms : multisets g (body.countP c.matches) ≠ [] := fun h => by
      have := sum_multisets g (body.countP c.matches)
      rw [h] at this
      obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ hgne
      have hpos : 0 < (g.map (·.2)).sum :=
        Nat.lt_of_lt_of_le (hg f hf)
          (List.le_sum_of_mem (List.mem_map_of_mem hf))
      simp only [List.map_nil, List.sum_nil] at this
      exact absurd this.symm (pow_ne_zero _ hpos.ne')
    have hends : (if c.matches last then g else [(last, 1)]) ≠ [] := by
      split
      · exact hgne
      · simp
    simp only [rerolled]
    refine ⟨fun h => ?_, fun o ho => ?_⟩
    · obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hms
      have := List.flatMap_eq_nil_iff.mp h e he
      exact hends (List.map_eq_nil_iff.mp this)
    · simp only [List.mem_flatMap, List.mem_map] at ho
      obtain ⟨e, he, f, hf, rfl⟩ := ho
      refine Nat.mul_pos (Nat.mul_pos ((multisets_props g _ e he).2 hg)
        (Nat.pow_pos (Nat.pos_of_ne_zero hG))) ?_
      split at hf
      · exact hg f hf
      · rw [List.mem_singleton] at hf
        rw [hf]
        exact Nat.one_pos

/-- A reroll changes neither the die of a setup nor the die from which it
draws. -/
theorem replacement_reroll (s : Setup) (c : Criterion) (limit : Option Nat) :
    (s.reroll c limit).replacement = s.replacement := by
  unfold reroll
  split
  · rfl
  · split
    · rfl
    · split
      · rfl
      · split <;> rfl
      · rfl

/-- A grounded setup without chains is faithful to its die, and conversely. -/
private theorem grounded_of_faithful {s : Setup}
    (hD : ∀ x ∈ s.replacement.faces, i32Min ≤ x ∧ x ≤ i32Max)
    (hp : s.Plain) (hf : s.Faithful) : s.Grounded := by
  refine ⟨hD, fun ρ hρ => ?_⟩
  have hpρ := hp ρ hρ
  cases ρ with
  | chains | stacked => exact hpρ.elim
  | range | standard | custom | fixed | pool => exact hf

/-- What a reroll leaves of a well-formed, grounded setup with faces: a
well-formed setup with faces, grounded unless it leaves stacks, each of whose
records fits the die. -/
private theorem reroll_props {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced) :
    (s.reroll c limit).WellFormed ∧ (s.reroll c limit).Faced ∧
      ((∀ ms ls, (s.reroll c limit).roll ≠ some (.stacked ms ls)) →
        (s.reroll c limit).Grounded) ∧
      ∀ R ∈ (s.reroll c limit).law.support, R.Fits s.replacement := by
  have hrep := replacement_reroll s c limit
  -- A grounded setup's records fit its die.
  have fits : ∀ {s' : Setup}, s'.WellFormed → s'.Grounded →
      s'.replacement = s.replacement →
      ∀ R ∈ s'.law.support, R.Fits s.replacement := fun hs' hg' hr' R hR =>
    hr' ▸ fits_of_grounded hs' hg' hR
  let Goal : Setup → Prop := fun s' => s'.WellFormed ∧ s'.Faced ∧
    ((∀ ms ls, s'.roll ≠ some (.stacked ms ls)) → s'.Grounded) ∧
    ∀ R ∈ s'.law.support, R.Fits s.replacement
  show Goal (s.reroll c limit)
  -- A reroll that leaves the setup.
  have stays : s.reroll c limit = s → Goal (s.reroll c limit) := fun h => by
    rw [h]
    exact ⟨hs, hfaced, fun _ => hg, fits hs hg rfl⟩
  by_cases hp : s.Plain
  · obtain ⟨hs', hf', hc', hp'⟩ := reroll_invariants_plain c limit hs
      (faithful_of_grounded hg hp) hfaced hp
    have hg' := grounded_of_faithful (hrep ▸ hg.1) hp' hf'
    exact ⟨hs', hc', fun _ => hg', fits hs' hg' hrep⟩
  obtain ⟨ρ, hρ⟩ : ∃ ρ, s.roll = some ρ := by
    cases h : s.roll with
    | none => exact absurd (fun ρ hρ => by rw [h] at hρ; cases hρ) hp
    | some ρ => exact ⟨ρ, rfl⟩
  obtain ⟨count, cl, rfl⟩ : ∃ count cl, ρ = .chains count cl := by
    have hgρ := hg.2 ρ hρ
    cases ρ with
    | chains count cl => exact ⟨count, cl, rfl⟩
    | stacked => exact hgρ.elim
    | range | standard | custom | fixed | pool =>
      exact absurd (fun ρ' h => by rw [hρ] at h; cases h; trivial) hp
  by_cases hk : s.results - s.lowest - s.highest = 0
  · exact stays (by simp [reroll, hρ, hk])
  cases he : Effect.resolve s.replacement c limit with
  | nothing => exact stays (by simp [reroll, hρ, hk, he])
  | crossOut =>
    have hs' : s.reroll c limit = { s with lowest := s.results } := by
      simp [reroll, hρ, hk, he, Roll.isSaturated]
    have hg' : ({ s with lowest := s.results } : Setup).Grounded := by
      refine ⟨hg.1, fun ρ' h => ?_⟩
      simp only [Option.mem_def, hρ, Option.some.injEq] at h
      subst h
      exact hg.2 _ hρ
    rw [hs']
    exact ⟨hs, hfaced, fun _ => hg', fits hs hg' rfl⟩
  | redraw G g =>
    obtain ⟨hglen, hG, _⟩ := redraw_law he
    have hgpos : ∀ f ∈ g, 0 < f.2 := (wellFormed_of_redraw he 0).2
    have hgne : g ≠ [] := fun h => hG (by rw [← hglen, h]; rfl)
    have hgG := sum_of_redraw he
    by_cases hd : s.lowest = 0 ∧ s.highest = 0
    · obtain ⟨hl, hh⟩ := hd
      obtain ⟨hne, hpos⟩ := hs _ hρ
      have hgρ := (hg.2 _ hρ).2
      have hW : (cl.map (·.2)).sum ≠ 0 := by
        obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
        exact (Nat.lt_of_lt_of_le (hpos e he)
          (List.le_sum_of_mem (List.mem_map_of_mem he))).ne'
      set TL := transformLaw cl G (rerolled c G g) with hTL
      let s'' : Setup :=
        { roll := some (.chains count TL), die := some s.replacement }
      have hs' : s.reroll c limit = s'' := by
        have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl, hh] at hk
        simp only [reroll, hρ, hk', he, Roll.alike, Bool.false_eq_true,
          false_and, ↓reduceIte, Roll.chained, Roll.rerollChains, hl, hh,
          and_self]
        rfl
      -- Every entry of the law of each chain weighs something.
      have hTLpos : ∀ e ∈ TL, 0 < e.2 := fun e he' => by
        simp only [hTL, transformLaw, List.mem_flatMap, List.mem_map] at he'
        obtain ⟨e₀, he₀, o, ho, rfl⟩ := he'
        exact Nat.mul_pos (Nat.mul_pos (hpos e₀ he₀)
          (Nat.pow_pos (Nat.pos_of_ne_zero hG)))
          ((rerolled_pos hgpos hgne hG e₀.1).2 o ho)
      have hTLne : TL ≠ [] := fun h => by
        obtain ⟨e, he'⟩ := List.exists_mem_of_ne_nil _ hne
        have := List.flatMap_eq_nil_iff.mp (by rw [← h, hTL, transformLaw])
          e he'
        exact (rerolled_pos hgpos hgne hG e.1).1 (List.map_eq_nil_iff.mp this)
      have hg' : s''.Grounded := by
        refine ⟨hg.1, fun ρ' h => ?_⟩
        simp only [s'', Option.mem_def, Option.some.injEq] at h
        subst h
        refine ⟨rfl, fun e he' => ?_⟩
        have hx : e.1 ∈ (weighted (.single 0) TL).support :=
          mem_support_weighted_of_mem he' (hTLpos e he').ne'
        rw [hTL, weighted_transformLaw_rerolled hgG hG hW] at hx
        obtain ⟨ch, hch, hx⟩ := (PMF.mem_support_bind_iff _ _ _).mp hx
        obtain ⟨e₀, he₀, rfl⟩ := mem_of_mem_support_weighted hW hch
        obtain ⟨hn, hlv, hfc⟩ := hgρ e₀ he₀
        have h := mem_support_rerollChain
          (Q := fun y => y ∈ s.replacement.faces)
          (fun _ _ _ y hy => mem_faces_of_redraw he hy) hfc hx
        exact ⟨h.2.2 (fun y hy => hg.1 y hy) hn, h.2.1.trans hlv, h.1⟩
      have hw' : s''.WellFormed := fun ρ' h => by
        simp only [s'', Option.mem_def, Option.some.injEq] at h
        subst h
        exact ⟨hTLne, hTLpos⟩
      rw [hs']
      exact ⟨hw', fun ρ' h => by cases h; trivial, fun _ => hg',
        fits hw' hg' rfl⟩
    · obtain ⟨hls, hfit⟩ := rerolls_whole_dropped (c := c) (limit := limit) hs
        hg hρ hk hd he
      have hw := hs _ hρ
      have hWs := Roll.sum_stacks_ne_zero hw
      set kp := keptPool (Roll.chains count cl).stacks s.lowest s.highest
        with hkp
      let dice := fun (chains : List (Chain × Nat)) =>
        (chains.map fun g' => (rerolled c G g g'.1).2 * g'.2).sum
      let most := (kp.map (dice ·.1)).foldl Max.max 0
      let pool := kp.map fun e => (e.1, e.2 * G ^ (most - dice e.1))
      let laws := (kp.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
        (ch, (rerolled c G g ch).1)
      let s'' : Setup :=
        { roll := some (Roll.stacked pool laws), die := some s.replacement }
      have hs' : s.reroll c limit = s'' := by
        have hk' : ¬s.results - s.lowest - s.highest = 0 := hk
        simp only [reroll, hρ, hk', he, Roll.alike, Bool.false_eq_true,
          false_and, ↓reduceIte, Roll.chained, Roll.rerollChains, hd, ← hkp,
          transformStacks]
        rfl
      -- Every multiset of chains keeps as many chains, and weighs something.
      have hkept : ∀ e ∈ pool,
          (e.1.map (·.2)).sum = s.results - s.lowest - s.highest ∧
            (∀ p ∈ e.1, 0 < p.2) ∧ 0 < e.2 := fun e he' => by
        simp only [pool, hkp, keptPool, List.map_map, List.mem_map,
          Function.comp_apply] at he'
        obtain ⟨e', he'', rfl⟩ := he'
        obtain ⟨hc, hpos⟩ := Roll.stacks_props hw he''
        refine ⟨by rw [count_slice, hc]; simp [results, hρ],
          fun p hp => (mem_slice hp).2,
          Nat.mul_pos hpos (Nat.pow_pos (Nat.pos_of_ne_zero hG))⟩
      have hw' : s''.WellFormed := fun ρ' h => by
        simp only [s'', Option.mem_def, Option.some.injEq] at h
        subst h
        refine ⟨fun h0 => hWs ?_, fun e he' => ⟨(hkept e he').2.1,
          (hkept e he').2.2⟩, fun e he' => ?_, fun e he' => ?_⟩
        · simp only [pool, hkp, keptPool, List.map_eq_nil_iff] at h0
          simp [h0]
        · rw [even_of_count _ _ (fun e he => (hkept e he).1) e he']
          cases pool with
          | nil => rfl
          | cons m ms => rfl
        · simp only [laws, List.mem_map] at he'
          obtain ⟨ch, _, rfl⟩ := he'
          exact rerolled_pos hgpos hgne hG ch
      refine ⟨hs' ▸ hw', hs' ▸ fun ρ' h => by cases h; trivial,
        fun h => absurd hs' fun h' => h _ _ (by rw [h']), hfit⟩

/-- A reroll of a grounded setup describes the specification's reroll of its
records, as a later reroll or explode reads them. -/
private theorem rerolls_whole {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced) :
    (s.reroll c limit).law.map Record.chainView =
      (s.law.bind fun r => Spec.reroll r c limit).map Record.chainView := by
  by_cases hp : s.Plain
  · exact rerolls_whole_plain c limit hs hg hfaced hp
  obtain ⟨ρ, hρ⟩ : ∃ ρ, s.roll = some ρ := by
    cases h : s.roll with
    | none => exact absurd (fun ρ hρ => by rw [h] at hρ; cases hρ) hp
    | some ρ => exact ⟨ρ, rfl⟩
  obtain ⟨count, cl, rfl⟩ : ∃ count cl, ρ = .chains count cl := by
    have hgρ := hg.2 ρ hρ
    cases ρ with
    | chains count cl => exact ⟨count, cl, rfl⟩
    | stacked => exact hgρ.elim
    | range | standard | custom | fixed | pool =>
      exact absurd (fun ρ' h => by rw [hρ] at h; cases h; trivial) hp
  by_cases hk : s.results - s.lowest - s.highest = 0
  · exact rerolls_whole_stays (by simp [reroll, hρ, hk])
      fun r hr => reroll_chainView_of_empty hs hg hk hr
  cases he : Effect.resolve s.replacement c limit with
  | nothing =>
    exact rerolls_whole_stays (by simp [reroll, hρ, hk, he])
      fun r hr => reroll_chainView_of_nothing (fits_of_grounded hs hg hr) he
  | crossOut =>
    obtain ⟨rfl, hm⟩ := Effect.resolve_crossOut he
    have hw : s.replacement.facesWithin c := by
      rw [Die.facesWithin, List.all_eq_true]
      exact List.countP_eq_length.mp hm
    have hs' : s.reroll c none = { s with lowest := s.results } := by
      simp [reroll, hρ, hk, he, Roll.isSaturated]
    -- Every record keeps no chain.
    have hg' : ({ s with lowest := s.results } : Setup).Grounded := by
      refine ⟨hg.1, fun ρ' h => ?_⟩
      simp only [Option.mem_def, hρ, Option.some.injEq] at h
      subst h
      exact hg.2 _ hρ
    rw [hs', PMF.map_bind, bind_congr_support fun r hr =>
        reroll_chainView_of_crossOut (fits_of_grounded hs hg hr) hw,
      PMF.bind_const, map_congr_support (g := fun _ => (0, s.replacement))
        fun R hR => ?_]
    · exact PMF.map_const _ _
    have hf := fits_of_grounded (s := { s with lowest := s.results }) hs hg' hR
    have hlen :=
      length_keptChains (s := { s with lowest := s.results }) hs hg' hR
    have hkc : R.keptChains = [] := List.eq_nil_of_length_eq_zero (by
      rw [hlen]
      simp [results])
    simp [Record.chainView, hkc, hf.die, replacement]
  | redraw G g =>
    by_cases hd : s.lowest = 0 ∧ s.highest = 0
    · exact rerolls_whole_iid hs hg hfaced hρ hk hd.1 hd.2 he
    · exact (rerolls_whole_dropped hs hg hρ hk hd he).1

/-- Reducing the chains of a well-formed setup to their totals leaves it well
formed, with faces. -/
private theorem collapse_props {s : Setup} (hs : s.WellFormed)
    (hfaced : s.Faced) : s.collapse.WellFormed ∧ s.collapse.Faced := by
  cases hρ : s.roll with
  | none =>
    have : s.collapse = s := by cases s; simp_all [collapse]
    rw [this]
    exact ⟨hs, hfaced⟩
  | some ρ =>
    cases ρ with
    | chains count law =>
      obtain ⟨hne, hpos⟩ := hs _ hρ
      refine ⟨fun ρ' h => ?_, fun ρ' h => ?_⟩
      · simp only [collapse, hρ, Option.map_some, Option.mem_def,
          Option.some.injEq, Roll.collapse] at h
        subst h
        exact tally_wellFormed _
      · simp only [collapse, hρ, Option.map_some, Option.mem_def,
          Option.some.injEq, Roll.collapse] at h
        subst h
        refine .inr fun h0 => ?_
        obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hne
        have hx : e.1.total ∈ Roll.expand (law.map fun e => (e.1.total, e.2)) :=
          List.mem_flatMap.mpr ⟨(e.1.total, e.2), List.mem_map_of_mem he,
            List.mem_replicate.mpr ⟨(hpos e he).ne', rfl⟩⟩
        have := (expand_tally_perm _).mem_iff.mpr hx
        rw [h0] at this
        cases this
    | stacked ms laws =>
      obtain ⟨hT, hmem⟩ := joinedTotals_props (hs _ hρ)
      have hcount : ∀ e ∈ (joinedTotals ms laws).map fun e =>
          (tally (Roll.expand e.1), e.2),
          (e.1.map (·.2)).sum = (Roll.stacked ms laws).results := fun e he => by
        simp only [List.mem_map] at he
        obtain ⟨o, ho, rfl⟩ := he
        rw [← length_expand, (expand_tally_perm _).length_eq, length_expand]
        exact (hmem o ho).1
      refine ⟨fun ρ' h => ?_, fun ρ' h => ?_⟩
      · simp only [collapse, hρ, Option.map_some, Option.mem_def,
          Option.some.injEq, Roll.collapse] at h
        subst h
        refine ⟨fun h0 => hT ?_, fun e he => ?_, fun e he => ?_⟩
        · simp only [List.map_eq_nil_iff] at h0
          simp [h0]
        · simp only [List.mem_map] at he
          obtain ⟨o, ho, rfl⟩ := he
          exact ⟨tally_wellFormed _, (hmem o ho).2⟩
        · rw [even_of_count _ _ hcount e he]
          cases (joinedTotals ms laws).map fun e =>
              (tally (Roll.expand e.1), e.2) with
          | nil => rfl
          | cons m ms => rfl
      · simp only [collapse, hρ, Option.map_some, Option.mem_def,
          Option.some.injEq, Roll.collapse] at h
        subst h
        trivial
    | range | standard | custom | fixed | pool =>
      have : s.collapse = s := by cases s; simp_all [collapse, Roll.collapse]
      rw [this]
      exact ⟨hs, hfaced⟩

/-- Reducing the chains of a well-formed setup, grounded unless it holds
stacks, to their totals leaves its records as every later drop, keep, and sum
views them. -/
private theorem law_collapse_of_props {s : Setup} (hs : s.WellFormed)
    (hg : (∀ ms ls, s.roll ≠ some (.stacked ms ls)) → s.Grounded) :
    s.collapse.law.map Record.view = s.law.map Record.view := by
  by_cases hst : ∃ ms ls, s.roll = some (.stacked ms ls)
  · obtain ⟨ms, ls, hρ⟩ := hst
    have hρ' : s.collapse.roll = some (Roll.stacked ms ls).collapse := by
      simp [collapse, hρ]
    have hrep : s.collapse.replacement = s.replacement := by
      simp only [collapse, replacement, hρ, Option.map_some, Option.getD_some,
        Roll.collapse, Roll.replacement]
    rw [map_view_law hρ', map_view_law hρ, Roll.map_results_collapse (hs _ hρ),
      hrep]
    rfl
  · simp only [not_exists] at hst
    exact law_collapse hs (hg hst)

/--
Rerolling a setup, as `Mixture::reroll` in `propagation/mixture.rs` rerolls
each setup, describes the specification's reroll of the records that it
describes, as every later instruction reads them (`Setup.RerollsAs`): by their
kept chains and die if the chains stay whole, and by their kept results and
die if they reduce to their totals. A setup without chains rerolls as
`Setup.law_reroll_plain` proves; one of i.i.d. chains rerolls the dice of each
chain, transforming the law of each if none has been dropped
(`weighted_transformLaw_rerolled`), and the kept chains of each multiset of
its chains, into stacks, if some have (`transformStacks`), whose deferred laws
weigh each multiset as `joined` does (`stackWeight`). Mirrors `Setup::reroll`
and `Setup::collapse` in `propagation/mixture.rs`, and `Roll::reroll_chains`,
`rerolled`, `transform_law`, `transform_stacks`, and `kept_pool` in
`propagation/chain.rs`.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.
- `whole`: Whether the chains stay whole, for a later reroll or explode, or
  reduce to their totals.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is grounded.
- `hfaced`: Every die of the setup has a face.
-/
theorem law_reroll {s : Setup} (c : Criterion) (limit : Option Nat)
    (whole : Bool) (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced) :
    s.RerollsAs c limit whole := by
  have hw := rerolls_whole c limit hs hg hfaced
  cases whole with
  | true => simpa [RerollsAs, rerollAs] using hw
  | false =>
    obtain ⟨hs', _, hg', hfit⟩ := reroll_props c limit hs hg hfaced
    simp only [RerollsAs, rerollAs, Bool.false_eq_true, ↓reduceIte]
    rw [law_collapse_of_props hs' hg']
    refine map_view_of_chainView hfit (fun R hR => ?_) hw
    obtain ⟨r, hr, hR⟩ := (PMF.mem_support_bind_iff _ _ _).mp hR
    exact fits_reroll (fits_of_grounded hs hg hr) hg.1 hR

/--
A reroll leaves a well-formed, grounded setup with faces well formed and with
faces, and, if it keeps the chains whole and leaves no stacks, grounded too,
so that the laws of a later reroll (`law_reroll`) and explode (`law_burst`)
hold of it: a reroll of a setup without chains leaves one, faithful to its
die (`Setup.reroll_invariants_plain`); one of i.i.d. chains none of which has
been dropped leaves i.i.d. chains of the die's faces, or custom dice of their
totals; and one of chains some of which have been dropped leaves stacks of
the kept pool.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.
- `whole`: Whether the chains stay whole.

# Hypotheses
- `hs`: The setup is well formed.
- `hg`: The setup is grounded.
- `hfaced`: Every die of the setup has a face.
-/
theorem reroll_invariants {s : Setup} (c : Criterion) (limit : Option Nat)
    (whole : Bool) (hs : s.WellFormed) (hg : s.Grounded) (hfaced : s.Faced) :
    (s.rerollAs c limit whole).WellFormed ∧ (s.rerollAs c limit whole).Faced ∧
      (whole = true → (∀ ms ls, (s.rerollAs c limit true).roll ≠
        some (.stacked ms ls)) → (s.rerollAs c limit true).Grounded) := by
  obtain ⟨hs', hc', hg', _⟩ := reroll_props c limit hs hg hfaced
  cases whole with
  | true => exact ⟨hs', hc', fun _ => hg'⟩
  | false =>
    obtain ⟨h1, h2⟩ := collapse_props hs' hc'
    exact ⟨h1, h2, fun h => by cases h⟩

end Setup

/-! ## Mixtures -/

/-- Transforming every setup of a mixture by every resolution of the
operands describes, as a view reads it, the specification's reroll or explode
of the mixture's law, if it does so of each setup. -/
private theorem mixtureLaw_of_setups {β : Type} {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)} (V : Record → β)
    (F : Setup → Criterion × Option Nat → Setup)
    (op : Record → Criterion → Option Nat → PMF Record)
    (h : ∀ s ∈ μ.support, ∀ q ∈ κ.support, (F s q).law.map V =
      (s.law.bind fun r => op r q.1 q.2).map V) :
    (mixtureLaw (lift F μ κ)).map V =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        op p.1 p.2.1 p.2.2).map V := by
  simp only [mixtureLaw, lift, PMF.bind_bind, PMF.bind_map, PMF.map_bind,
    Function.comp_def]
  refine bind_congr_support fun s hs => ?_
  refine (bind_congr_support (p := κ)
    (f := fun q : Criterion × Option Nat => (F s q).law.map V)
    (g := fun q : Criterion × Option Nat =>
      (s.law.bind fun r => op r q.1 q.2).map V)
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
  mixtureLaw_of_setups _ _ _ fun s hs q hq => by
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
  mixtureLaw_of_setups _ _ _ fun s hs q hq => by
    obtain ⟨hw, hg, hf, he⟩ := hμ s hs
    have h := Setup.law_burst q.1 q.2 false hw hg hf (he q hq)
    simpa [Setup.Bursts] using h

/--
Rerolling every setup of a mixture by every resolution of the reroll's
operands, keeping the chains whole, describes the specification's reroll of
the mixture's law, independent of the resolutions, as a later reroll or
explode reads it (`Record.chainView`), as `Mixture::reroll` in
`propagation/mixture.rs` rerolls a record whose operands are independent of
it (`law_reroll_resolved`).

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the criterion and the limit, as the operands resolve them.

# Hypotheses
- `hμ`: Every setup is well formed, grounded, and has faces.
-/
theorem mixtureLaw_reroll_whole {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.WellFormed ∧ s.Grounded ∧ s.Faced) :
    (mixtureLaw (lift (fun s q => s.rerollAs q.1 q.2 true) μ κ)).map
        Record.chainView =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        reroll p.1 p.2.1 p.2.2).map Record.chainView :=
  mixtureLaw_of_setups _ _ _ fun s hs q _ => by
    obtain ⟨hw, hg, hf⟩ := hμ s hs
    have h := Setup.law_reroll q.1 q.2 true hw hg hf
    simpa [Setup.RerollsAs] using h

/--
Rerolling every setup of a mixture by every resolution of the reroll's
operands, reducing the chains to their totals, describes the specification's
reroll of the mixture's law, independent of the resolutions, as every later
drop, keep, and sum reads it (`Record.view`), as `Mixture::reroll` in
`propagation/mixture.rs` rerolls a record that no later reroll or explode
reads.

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the criterion and the limit, as the operands resolve them.

# Hypotheses
- `hμ`: Every setup is well formed, grounded, and has faces.
-/
theorem mixtureLaw_reroll_totals {μ : PMF Setup}
    {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.WellFormed ∧ s.Grounded ∧ s.Faced) :
    (mixtureLaw (lift (fun s q => s.rerollAs q.1 q.2 false) μ κ)).map
        Record.view =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        reroll p.1 p.2.1 p.2.2).map Record.view :=
  mixtureLaw_of_setups _ _ _ fun s hs q _ => by
    obtain ⟨hw, hg, hf⟩ := hμ s hs
    have h := Setup.law_reroll q.1 q.2 false hw hg hf
    simpa [Setup.RerollsAs] using h

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
