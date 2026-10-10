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
multisets of `XdySpec/Chains.lean` are the laws of those chains:

- The kept chains of every record that a roll fills, as a multiset, have the
  law of the roll's multisets of chains (`Roll.stacks`), over their weights
  (`Roll.map_keptChains_law`): of i.i.d. chains, independent draws of the law
  of each (`Outcomes.weighted_outcomeStates`); of stacks whose chains are
  still themselves, their pool; and of any other roll, its outcomes, each
  result the chain of one die.
- Dropping chains from a record that drops none keeps the window of its kept
  chains (`Record.keptChains_drops`), and the window of normal chains sorted by
  `Xdy.Chain.le` depends only on their multiset, which `slice` slices
  (`coe_window_slice`).
- So the kept chains of every record that a setup describes, after its drops
  and keeps, have the law of the kept pool of its roll's multisets of chains
  (`Setup.map_keptChains_law`), as `kept_pool` makes it.
- The proofs ask that the roll be well formed, that its chains be normal, and
  its results lie within the `i32` range (`Roll.Normal`), and that stacks hold
  no law of what their chains have become (`Roll.Joined`), which the deferred
  laws of stacks are left to later proofs to lift.
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

end Record

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
      stackDraws_nil, PMF.pure_map, PMF.map_bind]
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

end Xdy.Spec
