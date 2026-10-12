import XdySpec.Stacks

/-!
# The outcomes of setups

The forward pass in `xdy/src/distribution/propagation.rs` splits a rolling
record by `Mixture::outcomes` in `propagation/mixture.rs`, which fixes each
setup at each outcome of `Roll::outcomes` in `propagation/record.rs`, keeping
its drops and its die. A roll without chains fixes the multisets of its
results, as `XdySpec/Outcomes.lean` proves; a roll of i.i.d. chains or of
stacks fixes the multisets of its chains, each as stacks of one multiset
(`chain_outcomes` in `propagation/chain.rs`), since a later reroll or explode
reads its chains, not their totals. This module proves that either way the
outcomes are the law of the record read as the oracle reads it, sorted:

- A setup's outcomes (`Setup.outcomes`) fix its roll at each outcome of
  `Roll.outcomes`, as fixed results, or, for chains or stacks, at each of
  `Roll.chainOutcomes`, and keep its die and its drops; `Setup.held` reads the
  one record that each describes.
- The multisets of chains weigh the roll's total (`Roll.sum_stacks`). A record
  of normal chains, read sorted, depends only on the multiset of its chains,
  which `Roll.map_keptChains_law` draws from the roll's multisets of chains,
  over their weights; so the outcomes of chains or stacks, over the total, are
  the law of the record read sorted, as the outcomes of every other roll are
  by `Roll.tsum_law_sorted` (`Setup.tsum_law_toOracle`,
  `Setup.map_toOracle_law`).
- Each outcome describes a certain record, already sorted
  (`Setup.law_of_mem_outcomes`), and is a well-formed setup
  (`Setup.wellFormed_of_mem_outcomes`), whose stacks are normal if the roll's
  chains are (`Roll.normal_of_mem_chainOutcomes`).

```mermaid
flowchart TD
    P["Roll.tsum_law_sorted:<br/>results, over the total,<br/>are the record's law, sorted"]
    K["Roll.map_keptChains_law:<br/>multisets of chains are<br/>the kept chains' law"]
    O["a record of normal chains,<br/>read sorted, is a function<br/>of its multiset of chains"]
    S["Setup.tsum_law_toOracle:<br/>outcomes, over the total,<br/>are the record's law, read sorted"]
    X["exec_split_sorted:<br/>fixing each world at its<br/>outcome keeps the law"]
    P --> S
    K --> S
    O --> S
    S --> X
```

The Rust's outcomes keep no law of what each chain has become, so the chains
of each are still themselves; the model's do likewise.
-/

open scoped ENNReal

namespace Xdy.Spec

open OrderStatistics (tsum_map_mul)

/-! ## The outcomes of a setup -/

namespace Setup

/--
Every outcome of a setup: the setup with each multiset of its roll's results
fixed, or, for chains or stacks, each multiset of its chains, as stacks of one
multiset, keeping its die and its drops, or the setup itself if it has no
roll. Mirrors the outcomes of each setup in `Mixture::outcomes` in
`mixture.rs`, before it scales them to the common denominator.

# Parameters
- `s`: The setup.

# Returns
The outcomes, each with its weight, whose weights sum to the
[total](Setup.total).

# Notes
The Rust keeps the die only if a later reroll reads the record before a roll
refills it (`rerolled` in `propagation/plan.rs`), and otherwise merges the
outcomes that differ only in their dice. Keeping it always splits no less
finely, and is as sound.

# Examples
```lean
#guard ({ roll := some (.range 1 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 1)]), die := some (.range 1 2),
      lowest := 1 }, 1),
    ({ roll := some (.fixed [(2, 1)]), die := some (.range 1 2),
       lowest := 1 }, 1)]
```
-/
def outcomes (s : Setup) : List (Setup × Nat) :=
  match s.roll with
  | some ρ =>
    let rolls := if ρ.chained then ρ.chainOutcomes
      else ρ.outcomes.map fun e => (.fixed e.1, e.2)
    rolls.map fun e =>
      ({ s with roll := some e.1, die := some s.replacement }, e.2)
  | none => [(s, 1)]

/--
The record that an outcome of a setup holds, as the oracle reads it: the
results of its fixed roll, or the chains of its one multiset of stacks, or
none if it has no roll, with its drops and its die.

# Parameters
- `s`: The outcome, a setup of fixed results, of stacks of one multiset, or
  of no roll.

# Returns
The record. Any other setup holds no one record, and reads as empty, or as
the chains of its first multiset.

# Examples
```lean
#guard ({ roll := some (.fixed [(1, 2), (3, 1)]), highest := 1 } : Setup).held
  == { results := [1, 1, 3], highest := 1 }
#guard ({ roll := some (.stacked [([(.live [6] 2, 1), (.single 3, 1)], 1)] []),
          lowest := 1 } : Setup).held
  == { results := [3, 8], lowest := 1, chains := [.single 3, .live [6] 2] }
```
-/
def held (s : Setup) : Xdy.Record :=
  match s.roll with
  | some (.fixed rs) =>
    { results := Roll.expand rs, lowest := s.lowest, highest := s.highest,
      die := s.replacement }
  | some (.stacked ((ms, _) :: _) _) =>
    { Xdy.Record.ofChains s.replacement (Roll.expand ms) with
      lowest := s.lowest, highest := s.highest }
  | _ => { lowest := s.lowest, highest := s.highest, die := s.replacement }

end Setup

/-! ## The outcomes of chains -/

namespace Roll

/--
Every outcome of a well-formed roll of chains or stacks is well formed, stacks
of one multiset whose chains are all still themselves, and weighs something.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem wellFormed_of_mem_chainOutcomes {ρ : Roll} {e : Roll × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.chainOutcomes) :
    e.1.WellFormed ∧ 0 < e.2 := by
  simp only [chainOutcomes, List.mem_map] at he
  obtain ⟨m, hm, rfl⟩ := he
  refine ⟨⟨List.cons_ne_nil _ _, fun e he => ?_, by simp [Even, results],
    by simp⟩, (stacks_props hρ hm).2⟩
  rw [List.mem_singleton.mp he]
  exact ⟨copies_pos_of_mem_stacks (e := m) hρ hm, Nat.one_pos⟩

/--
Every outcome of a well-formed, normal roll of chains or stacks is normal.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `hn`: The roll is normal.
- `he`: `e` is an outcome of the roll.
-/
theorem normal_of_mem_chainOutcomes {ρ : Roll} {e : Roll × Nat}
    (hρ : ρ.WellFormed) (hn : ρ.Normal) (he : e ∈ ρ.chainOutcomes) :
    e.1.Normal := by
  simp only [chainOutcomes, List.mem_map] at he
  obtain ⟨m, hm, rfl⟩ := he
  refine ⟨fun e he => ?_, by simp⟩
  rw [List.mem_singleton.mp he]
  exact normal_of_mem_stacks (e := m) hρ hn hm

/-- A roll holds chains exactly when it is chained. -/
private theorem chained_of_not_plain {ρ : Roll} (h : ¬ρ.Plain) :
    ρ.chained = true := by
  cases ρ <;> first | rfl | exact absurd trivial h

end Roll

/-! ## The law of a setup, read sorted -/

namespace Setup

/-- `Setup.tsum_law_toOracle` for a roll of chains or stacks: each record,
read sorted, is the oracle's record of its kept chains, with the setup's
drops, which depends only on their multiset, and their multisets have the
law of the roll's multisets of chains. -/
private theorem tsum_law_toOracle_chained {s : Setup} {ρ : Roll}
    (hρ : s.roll = some ρ) (hw : ρ.WellFormed) (hc : ¬ρ.Plain)
    (hn : ρ.Normal) (ψ : Xdy.Record → Nat) :
    ∑' r, s.law r * (ψ r.toOracle : ℝ≥0∞) =
      (OrderStatistics.integral s.outcomes (fun o => ψ o.held) : ℝ≥0∞) /
        (s.total : ℝ≥0∞) := by
  set D := s.replacement with hD
  -- The oracle's record of chains, with the setup's drops.
  let out : List Chain → Xdy.Record := fun cs =>
    { Xdy.Record.ofChains D cs with lowest := s.lowest, highest := s.highest }
  let F : Multiset Chain → Xdy.Record := fun M => out M.toList
  have hF : ∀ cs : List Chain, (∀ x ∈ cs, NormalChain x) →
      F cs = out cs := fun cs hn' => by
    have hp : (cs : Multiset Chain).toList.Perm cs :=
      Multiset.coe_eq_coe.mp (Multiset.coe_toList _)
    simp only [F, out, Xdy.Record.ofChains,
      mergeSort_chains_perm hp fun x hx => hn' x (hp.mem_iff.mp hx)]
  have hor : ∀ cs : List Chain,
      ({ Record.ofChains .none cs with
          lowest := s.lowest, highest := s.highest, die := D } :
        Record).toOracle = out cs :=
    fun cs => by
      change { (Record.ofChains D cs).toOracle with
        lowest := s.lowest, highest := s.highest } = out cs
      rw [Record.toOracle_ofChains]
  -- Each record, read sorted, is a function of its kept chains.
  have hlhs : ∑' r, s.law r * (ψ r.toOracle : ℝ≥0∞) =
      ∑' r, ρ.law r * (ψ (F r.keptChains) : ℝ≥0∞) := by
    simp only [law, hρ, Option.map_some, Option.getD_some, tsum_map_mul]
    refine tsum_congr fun r => ?_
    by_cases hr : r ∈ ρ.law.support
    · obtain ⟨cs, rfl, hcs⟩ := ρ.exists_ofChains_of_mem_support_law hc hn hr
      rw [keptChains_ofChains .none hcs, hF cs hcs, ← hor cs]
    · rw [PMF.mem_support_iff, not_not] at hr
      simp [hr]
  rw [hlhs, ← tsum_map_mul ρ.law (fun r => (r.keptChains : Multiset Chain))
      fun M => (ψ (F M) : ℝ≥0∞), ρ.map_keptChains_law hw hn, tsum_map_mul,
    Outcomes.tsum_weighted [] ρ.stacks (Roll.sum_stacks_ne_zero hw),
    Roll.sum_stacks hw]
  simp only [outcomes, total, hρ, Roll.chained_of_not_plain hc, ↓reduceIte,
    Roll.chainOutcomes, List.map_map, Option.map_some, Option.getD_some,
    OrderStatistics.integral, Nat.cast_list_sum]
  congr 1
  refine congrArg List.sum (List.map_congr_left fun e he => ?_)
  have hne : ∀ x ∈ Roll.expand e.1, NormalChain x := fun x hx => by
    obtain ⟨p, hp, hx⟩ := List.mem_flatMap.mp hx
    rw [List.eq_of_mem_replicate hx]
    exact Roll.normal_of_mem_stacks hw hn he p hp
  simp only [Function.comp_apply, Nat.cast_mul, hF _ hne]
  rfl

/--
A setup's outcomes, over its total, are the law of the record that it
describes, read as the oracle reads it, sorted: a sum of `ψ` of the sorted
record against the law is the outcomes' integral of `ψ` of the records that
they hold, over the total.

# Parameters
- `s`: The setup.
- `ψ`: The function of the sorted record.

# Hypotheses
- `hs`: The setup is well formed.
- `hn`: The setup's roll holds no chains, or its chains are normal.
-/
theorem tsum_law_toOracle (s : Setup) (hs : s.WellFormed)
    (hn : ∀ ρ ∈ s.roll, ρ.Plain ∨ ρ.Normal) (ψ : Xdy.Record → Nat) :
    ∑' r, s.law r * (ψ r.toOracle : ℝ≥0∞) =
      (OrderStatistics.integral s.outcomes (fun o => ψ o.held) : ℝ≥0∞) /
        (s.total : ℝ≥0∞) := by
  rcases s with ⟨_ | ρ, die, lowest, highest⟩
  · simp only [law, Option.map_none, Option.getD_none, PMF.pure_map,
      Outcomes.tsum_pure_mul, outcomes, total]
    simp [OrderStatistics.integral, held, Record.toOracle, Record.sorted]
  by_cases hp : ρ.Plain
  · have h := ρ.tsum_law_sorted (hs ρ rfl) hp
      fun ys => ψ ⟨ys, lowest, highest, die.getD ρ.replacement, []⟩
    simp only [law, Option.map_some, Option.getD_some, tsum_map_mul]
    refine (tsum_congr fun r => ?_).trans (h.trans ?_)
    · by_cases hr : r ∈ ρ.law.support
      · simp [Roll.chains_of_mem_support_law hp hr, Record.toOracle,
          Record.sorted, replacement]
      · rw [PMF.mem_support_iff, not_not] at hr
        simp [hr]
    · simp [outcomes, total, held, OrderStatistics.integral, Function.comp_def,
        replacement, Roll.chained_of_plain hp]
  · exact tsum_law_toOracle_chained rfl (hs ρ rfl) hp
      ((hn ρ rfl).resolve_left hp) ψ

/--
The law of the record that a setup describes, read as the oracle reads it,
sorted, gives each sorted record the weight of the outcomes that hold it, over
the total, as `Mixture::outcomes` weighs them.

# Parameters
- `s`: The setup.
- `v`: The sorted record.

# Hypotheses
- `hs`: The setup is well formed.
- `hn`: The setup's roll holds no chains, or its chains are normal.
-/
theorem map_toOracle_law (s : Setup) (hs : s.WellFormed)
    (hn : ∀ ρ ∈ s.roll, ρ.Plain ∨ ρ.Normal) (v : Xdy.Record) :
    (s.law.map Record.toOracle) v =
      (OrderStatistics.integral s.outcomes
          (fun o => if o.held = v then 1 else 0) : ℝ≥0∞) /
        (s.total : ℝ≥0∞) := by
  rw [Outcomes.map_apply_eq_tsum]
  exact s.tsum_law_toOracle hs hn fun r => if r = v then 1 else 0

/--
Every outcome of a well-formed setup describes a certain record, the one that
it holds, whose results are sorted, so that the oracle reads it as it is: the
world that a split fixes at the outcome holds that record.

# Parameters
- `s`: The setup.
- `o`: An outcome with its weight.

# Hypotheses
- `hs`: The setup is well formed.
- `ho`: `o` is an outcome of the setup.
-/
theorem law_of_mem_outcomes {s : Setup} (hs : s.WellFormed)
    {o : Setup × Nat} (ho : o ∈ s.outcomes) :
    o.1.law = PMF.pure (Record.ofOracle o.1.held) ∧
      o.1.held.results.Pairwise (· ≤ ·) := by
  rcases s with ⟨_ | ρ, die, lowest, highest⟩
  · simp only [outcomes, List.mem_singleton] at ho
    subst ho
    simp [law, PMF.pure_map, held, Record.ofOracle]
  by_cases hp : ρ.Plain
  · simp only [outcomes, Roll.chained_of_plain hp, Bool.false_eq_true,
      ↓reduceIte, List.map_map, List.mem_map] at ho
    obtain ⟨e, he, rfl⟩ := ho
    have hw := Roll.wellFormed_of_mem_outcomes (hs ρ rfl) he
    refine ⟨?_, Roll.expand_pairwise hw.1⟩
    simp [law, Roll.law, PMF.pure_map, held, Record.ofOracle]
  · simp only [outcomes, Roll.chained_of_not_plain hp, ↓reduceIte,
      Roll.chainOutcomes, List.map_map, List.mem_map] at ho
    obtain ⟨e, _, rfl⟩ := ho
    refine ⟨?_, ?_⟩
    · simp only [Function.comp_apply, law, Roll.law, Option.map_some,
        Option.getD_some, List.map_cons, List.map_nil, stackWeight_nil,
        mul_one, weighted_single _ _ one_ne_zero, PMF.pure_bind,
        Roll.stackDraws_nil, PMF.pure_map]
      rfl
    · exact List.pairwise_map.mpr
        ((pairwise_mergeSort_chains _).imp Chain.total_le_of_le)

/--
Every outcome of a well-formed setup is well formed.

# Parameters
- `s`: The setup.
- `o`: An outcome with its weight.

# Hypotheses
- `hs`: The setup is well formed.
- `ho`: `o` is an outcome of the setup.
-/
theorem wellFormed_of_mem_outcomes {s : Setup} (hs : s.WellFormed)
    {o : Setup × Nat} (ho : o ∈ s.outcomes) : o.1.WellFormed := by
  rcases s with ⟨_ | ρ, die, lowest, highest⟩
  · simp only [outcomes, List.mem_singleton] at ho
    subst ho
    exact hs
  intro τ hτ
  by_cases hp : ρ.Plain
  · simp only [outcomes, Roll.chained_of_plain hp, Bool.false_eq_true,
      ↓reduceIte, List.map_map, List.mem_map] at ho
    obtain ⟨e, he, rfl⟩ := ho
    cases hτ
    exact Roll.wellFormed_of_mem_outcomes (hs ρ rfl) he
  · simp only [outcomes, Roll.chained_of_not_plain hp, ↓reduceIte,
      List.mem_map] at ho
    obtain ⟨e, he, rfl⟩ := ho
    cases hτ
    exact (Roll.wellFormed_of_mem_chainOutcomes (hs ρ rfl) he).1

end Setup

end Xdy.Spec
