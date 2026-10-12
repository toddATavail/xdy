import XdySpec.Conditioning
import XdySpec.Oracle
import XdySpec.OrderStatistics

/-!
# The outcomes of a rolling record

The forward pass in `xdy/src/distribution/propagation.rs` splits a rolling
record not by the records that it can hold, in the order rolled, but by the
multisets of its results: `split` fixes the record at each outcome of
`Mixture::outcomes` in `propagation/mixture.rs`, each setup with each multiset
of `Roll::outcomes` in `propagation/record.rs`. `exec_split` proves splitting
on any value sound, so the weights of the multisets are sound as the law of a
value; this module proves that they are the law of the record read sorted, and
that fixing each world's record at its multiset, rather than conditioning it
on one, changes nothing that any later instruction can tell:

- The multisets of a roll of dice grow one distinct face at a time, in
  ascending order (`outcomeStates`): placing `m` copies of the next face after
  `j` results weighs `C(j + m, m) · wᵐ`. As the order statistics' states are
  the weighed rolls read through the kept window, these states are the weighed
  rolls read whole (`integral_outcomeStates`), by the same peeling of the
  greatest face (`words_peel`).
- The program is `multisets` in `propagation/record.rs`, which the chains of
  `propagation/chain.rs` share, and its faces may be of any type: its states,
  integrated against any function of each multiset, are the weighed draws of
  the die (`integral_outcomeStates_bag`), since a multiset needs no order to
  peel its last face (`bagWords_peel`); so its multisets, over their weights,
  are the multiset of independent draws (`weighted_outcomeStates`).
- So every roll's outcomes (`Roll.outcomes`), over its total (`Roll.total`),
  are the law of the record that it fills, sorted
  (`Roll.tsum_law_sorted`, `Roll.map_sorted_law`), and each outcome is fixed
  results in ascending order (`Roll.wellFormed_of_mem_outcomes`). A pool's
  outcomes are its multisets, whose weights are its law
  (`Outcomes.tsum_weighted`). These are the multisets of the results of a
  roll without chains (`Roll.Plain`); the Rust splits chains on the
  multisets of the chains instead, which `XdySpec/Splits.lean` proves, with
  the outcomes of setups, which keep their die and their drops.
- A mixture read sorted mixes its setups read sorted
  (`map_toOracle_mixtureLaw`), and `Dist.hasLaw_mixture` cancels the scaling
  of each setup's outcomes to the common denominator.
- Sorting commutes with every instruction (`Oracle.lean`'s `hasLaw_step`), so
  the law of states read sorted after any instructions depends only on the law
  read sorted before them (`map_toOracle_exec`). Splitting on a record read
  sorted, and fixing each world's record at its outcome, therefore answers the
  law of the program read sorted (`exec_split_sorted`), and with it the law of
  every register (`exec_split_sorted_value`).

```mermaid
flowchart TD
    W["integral_outcomeStates:<br/>the states are the weighed rolls"]
    R["Roll.tsum_law_sorted:<br/>outcomes over the total are<br/>the record's law, sorted"]
    S["Setup.tsum_law_toOracle, in Splits.lean:<br/>with the drops kept,<br/>and chains split by chains"]
    C["map_toOracle_exec:<br/>sorting commutes with execution"]
    X["exec_split_sorted:<br/>fixing each world at its<br/>outcome keeps the law"]
    W --> R --> S --> X
    C --> X
```

The Rust places only all the remaining copies of the greatest face, since the
multisets that it leaves short of `n` results are never read; the model
computes them anyway, and reads the multisets of `n` results, which are the
same, as `OrderStatistics.lean` does. It keeps its outcomes in lists, merging
nothing, where `Mixture::outcomes` merges equal setups; an outcome's weight is
the total of its entries.
-/

open scoped ENNReal

namespace Xdy.Spec

open OrderStatistics (words words_zero words_succ words_peel integral
  integral_flatMap integral_map list_range_sum tsum_map_mul tsum_rollDice
  tsum_weightedDie rollDice_toNat standardFaces standardFaces_sorted
  total_standardFaces expand_standardFaces)

namespace Outcomes

/-! ## The program -/

/--
Place copies of a face after a multiset, as `multisets` in `record.rs` pushes
them only if there are any.

# Type parameters
- `α`: The type of the faces: results, or chains.

# Parameters
- `rs`: The multiset: each distinct face, in the order placed, with its
  copies.
- `v`: The face, which `rs` does not hold.
- `m`: The copies.

# Returns
The multiset with `m` copies of `v` after its results, or `rs` if `m = 0`.

# Examples
```lean
#guard Outcomes.grow [(1, 2)] 3 0 == [(1, 2)]
#guard Outcomes.grow [(1, 2)] 3 2 == [(1, 2), (3, 2)]
```
-/
def grow {α : Type} (rs : List (α × Nat)) (v : α) (m : Nat) :
    List (α × Nat) :=
  if m = 0 then rs else rs ++ [(v, m)]

/--
Growing a multiset by copies of a face appends them to its results.

# Parameters
- `rs`: The multiset.
- `v`: The face.
- `m`: The copies.
-/
theorem expand_grow {α : Type} (rs : List (α × Nat)) (v : α) (m : Nat) :
    Roll.expand (grow rs v m) = Roll.expand rs ++ List.replicate m v := by
  unfold grow
  split <;> simp_all [Roll.expand]

/--
Place every number of copies of one face, of value `v` and weight `w`. Mirrors
the loop over the faces of `multisets` in `record.rs`: the state of `J`
results gathers, for each number `m` of copies, the multisets of `J - m`
results, each grown by the copies and weighed by `C(J, m) · wᵐ`, which is the
Rust's `C(j + m, m) · wᵐ` with `j = J - m`.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `states`: The states before the face: for each number of results, the
  multisets and their weights.
- `face`: The face and its weight.

# Returns
The states after the face.
-/
def outcomeStep {α : Type} (states : Nat → List (List (α × Nat) × Nat))
    (face : α × Nat) : Nat → List (List (α × Nat) × Nat) :=
  fun J => (List.range (J + 1)).flatMap fun m =>
    (states (J - m)).map fun e =>
      (grow e.1 face.1 m, J.choose m * face.2 ^ m * e.2)

/-- The states before any face: no results, the empty multiset once. -/
def initial {α : Type} : Nat → List (List (α × Nat) × Nat) :=
  fun j => if j = 0 then [([], 1)] else []

/--
The states of the program after every face of a die, in order. Mirrors
`multisets` in `record.rs`, which `Roll::outcomes` calls for dice, and
`propagation/chain.rs` for chains.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The distinct faces, in ascending order, each with its weight.

# Returns
The states: for each number of results, the multisets of the faces and their
weights.

# Examples
```lean
-- Two dice of `[1, 2, 2]` roll `{1, 1}` once, `{1, 2}` four ways, and
-- `{2, 2}` four ways.
#guard Outcomes.outcomeStates [(1, 1), (2, 2)] 2 ==
  [([(1, 2)], 1), ([(1, 1), (2, 1)], 4), ([(2, 2)], 4)]
```
-/
def outcomeStates {α : Type} (die : List (α × Nat)) :
    Nat → List (List (α × Nat) × Nat) :=
  die.foldl outcomeStep initial

/--
One more face takes one more step.

# Parameters
- `die`: The faces before it.
- `face`: The face.
-/
theorem outcomeStates_append {α : Type} (die : List (α × Nat))
    (face : α × Nat) :
    outcomeStates (die ++ [face]) = outcomeStep (outcomeStates die) face := by
  simp [outcomeStates, List.foldl_append]

/--
The program's state of `J` results, integrated against `ψ` of each multiset's
results, is the weighed rolls of `J` dice: each step peels the greatest face
of the die so far, as `words_peel` does.

# Parameters
- `die`: The faces.

# Hypotheses
- The faces ascend strictly.
-/
theorem integral_outcomeStates :
    ∀ (die : List (Int × Nat)), die.Pairwise (fun a b => a.1 < b.1) →
      ∀ (J : Nat) (ψ : List Int → Nat),
        integral (outcomeStates die J) (fun rs => ψ (Roll.expand rs)) =
          words die J ψ := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro _ J ψ
    cases J with
    | zero => simp [outcomeStates, initial, integral, words_zero, Roll.expand]
    | succ J => simp [outcomeStates, initial, integral, words_succ]
  | append_singleton die face ih =>
    intro hsorted J ψ
    rw [List.pairwise_append] at hsorted
    obtain ⟨hdie, _, hlt⟩ := hsorted
    have hv : ∀ p ∈ die, p.1 < face.1 := fun p hp => hlt p hp face (by simp)
    obtain ⟨v, w⟩ := face
    rw [outcomeStates_append, words_peel w hv]
    simp only [outcomeStep, integral_flatMap, list_range_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [integral_map _ (fun rs => grow rs v m)]
    simp only [expand_grow]
    exact congrArg _ (ih hdie (J - m) fun ys => ψ (ys ++ List.replicate m v))

/--
Every multiset of the program ascends strictly, holds each result at least
once, and holds only faces of the die.

# Parameters
- `die`: The faces.
- `J`: The number of results.
- `e`: A multiset with its weight.

# Hypotheses
- The faces ascend strictly.
- `e` is a multiset of the state of `J` results.
-/
theorem mem_outcomeStates :
    ∀ (die : List (Int × Nat)), die.Pairwise (fun a b => a.1 < b.1) →
      ∀ (J : Nat) (e : List (Int × Nat) × Nat), e ∈ outcomeStates die J →
        e.1.Pairwise (fun a b => a.1 < b.1) ∧
          ∀ p ∈ e.1, 0 < p.2 ∧ ∃ q ∈ die, q.1 = p.1 := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro _ J e he
    simp only [outcomeStates, List.foldl_nil, initial] at he
    split at he
    · simp_all
    · simp at he
  | append_singleton die face ih =>
    intro hsorted J e he
    rw [List.pairwise_append] at hsorted
    obtain ⟨hdie, _, hlt⟩ := hsorted
    have hv : ∀ p ∈ die, p.1 < face.1 := fun p hp => hlt p hp face (by simp)
    rw [outcomeStates_append] at he
    simp only [outcomeStep, List.mem_flatMap, List.mem_range, List.mem_map]
      at he
    obtain ⟨m, _, e', he', rfl⟩ := he
    obtain ⟨hpw, hmem⟩ := ih hdie _ e' he'
    unfold grow
    split
    · exact ⟨hpw, fun p hp => by
        obtain ⟨hp0, q, hq, hqp⟩ := hmem p hp
        exact ⟨hp0, q, by simp [hq], hqp⟩⟩
    · rename_i hm
      refine ⟨List.pairwise_append.mpr ⟨hpw, by simp, fun a ha b hb => ?_⟩,
        fun p hp => ?_⟩
      · obtain ⟨_, q, hq, hqa⟩ := hmem a ha
        simp only [List.mem_singleton] at hb
        subst hb
        rw [← hqa]
        exact hv q hq
      · rcases List.mem_append.mp hp with hp | hp
        · obtain ⟨hp0, q, hq, hqp⟩ := hmem p hp
          exact ⟨hp0, q, by simp [hq], hqp⟩
        · simp only [List.mem_singleton] at hp
          subst hp
          exact ⟨Nat.pos_of_ne_zero hm, face, by simp, rfl⟩

end Outcomes

/-! ## Rolls -/

namespace Roll

/--
A roll as `propagation/record.rs` keeps it: a custom die's distinct faces, as
`distinct_faces` counts them, and fixed results, as `Roll::outcomes` fixes
them, ascend strictly, each at least once; and a pool, as `Roll::pool` makes
it, holds at least one multiset, each of fixed results so formed, of positive
weight, and each of as many results. The law of i.i.d. chains holds at least
one chain, each of positive weight; stacks, as `propagation/chain.rs` makes
them, hold at least one multiset, each of positive weight and of as many
chains, each at least once, and the law of what each chain has become holds
at least one chain, each of positive weight.

# Parameters
- `ρ`: The roll.

# Returns
Whether the roll is well formed. Ranges and standard dice always are.
-/
def WellFormed : Roll → Prop
  | .custom _ faces | .fixed faces =>
    faces.Pairwise (fun a b => a.1 < b.1) ∧ ∀ p ∈ faces, 0 < p.2
  | .pool outcomes =>
    outcomes ≠ [] ∧
      (∀ e ∈ outcomes, (e.1.Pairwise (fun a b => a.1 < b.1) ∧
        ∀ p ∈ e.1, 0 < p.2) ∧ 0 < e.2) ∧
      (Roll.pool outcomes).Even
  | .chains _ law => law ≠ [] ∧ ∀ p ∈ law, 0 < p.2
  | .stacked multisets laws =>
    multisets ≠ [] ∧
      (∀ e ∈ multisets, (∀ p ∈ e.1, 0 < p.2) ∧ 0 < e.2) ∧
      (Roll.stacked multisets laws).Even ∧
      ∀ e ∈ laws, e.2 ≠ [] ∧ ∀ p ∈ e.2, 0 < p.2
  | _ => True

/--
A well-formed roll is even.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
-/
theorem WellFormed.even {ρ : Roll} (hρ : ρ.WellFormed) : ρ.Even := by
  cases ρ with
  | pool outcomes => exact hρ.2.2
  | stacked multisets laws => exact hρ.2.2.1
  | _ => trivial

/--
The total weight of a roll: the number of its branches. Mirrors `total` in
`record.rs`.

# Parameters
- `ρ`: The roll.

# Returns
The width of a range, or the total weight of the die raised to the number of
dice, or `1` for an empty range, no dice, dice without faces, and fixed
results, or the total weight of the multisets of a pool, or the total weight
of the law of a chain raised to the number of chains, or the weight of each
multiset of stacks, times the branches that the laws of what its chains have
become add to it (`stackWeight`), summed.

# Examples
```lean
#guard (Roll.standard 3 6).total == 216
#guard (Roll.custom 2 [(1, 1), (2, 2)]).total == 9
#guard (Roll.range 4 3).total == 1
```
-/
def total : Roll → Nat
  | .range start stop => if stop < start then 1 else (stop - start + 1).toNat
  | .standard count faces =>
    if 0 < count ∧ 0 < faces then faces.toNat ^ count.toNat else 1
  | .custom count faces =>
    if 0 < count ∧ faces ≠ [] then OrderStatistics.total faces ^ count.toNat
    else 1
  | .fixed _ => 1
  | .pool outcomes => (outcomes.map (·.2)).sum
  | .chains count law =>
    if 0 < count ∧ law ≠ [] then (law.map (·.2)).sum ^ count.toNat else 1
  | .stacked multisets laws =>
    (multisets.map fun e => e.2 * stackWeight laws e.1).sum

/--
The outcomes of `n` dice of a die: the multisets of the program, unless there
are no dice, which roll nothing once, or the die has no faces, which rolls
zeros once. Mirrors the dice of `Roll::outcomes` in `record.rs`.

# Parameters
- `n`: The number of dice.
- `die`: The distinct faces, in ascending order, each with its weight.

# Returns
The multisets, each with its weight.
-/
def diceOutcomes (n : Nat) (die : List (Int × Nat)) :
    List (List (Int × Nat) × Nat) :=
  if n = 0 then [([], 1)]
  else if die = [] then [([(0, n)], 1)]
  else Outcomes.outcomeStates die n

/--
Every outcome of a roll: each multiset of its results, in ascending order,
with the number of branches of the roll that reach it. Mirrors `outcomes` in
`record.rs`.

# Parameters
- `ρ`: The roll.

# Returns
The multisets, each with its weight, whose weights sum to the
[total](Roll.total), or none for chains or stacks.

# Notes
The Rust splits chains or stacks on the multisets of their chains, as stacks
of one multiset each (`chain_outcomes` in `propagation/chain.rs`), not on the
multisets of their results, so these are the outcomes of a roll without
chains (`Roll.Plain`), which every proof about them asks; `Roll.chainOutcomes`
mirrors those of chains, and `Setup.outcomes` splits on either.

# Examples
```lean
#guard (Roll.range 4 3).outcomes == [([(0, 1)], 1)]
#guard (Roll.range 1 2).outcomes == [([(1, 1)], 1), ([(2, 1)], 1)]
#guard (Roll.custom 2 []).outcomes == [([(0, 2)], 1)]
```
-/
def outcomes : Roll → List (List (Int × Nat) × Nat)
  | .range start stop =>
    if stop < start then [([(0, 1)], 1)]
    else (List.range (stop - start + 1).toNat).map fun (i : Nat) =>
      ([(start + i, 1)], 1)
  | .standard count faces => diceOutcomes count.toNat (standardFaces faces)
  | .custom count faces => diceOutcomes count.toNat faces
  | .fixed results => [(results, 1)]
  | .pool outcomes => outcomes
  | .chains .. | .stacked .. => []

/--
The results of a multiset that ascends strictly are sorted.

# Parameters
- `rs`: The multiset.

# Hypotheses
- `h`: Its distinct results ascend strictly.
-/
theorem expand_pairwise {rs : List (Int × Nat)}
    (h : rs.Pairwise (fun a b => a.1 < b.1)) :
    (expand rs).Pairwise (· ≤ ·) := by
  unfold expand
  rw [List.pairwise_flatMap]
  refine ⟨fun p _ => ?_, h.imp fun hab x hx y hy => ?_⟩
  · exact List.pairwise_replicate.mpr (.inr le_rfl)
  · rw [List.eq_of_mem_replicate hx, List.eq_of_mem_replicate hy]
    exact hab.le

/--
Every outcome of a well-formed roll is a well-formed roll of fixed results, as
`Roll::outcomes` gives them.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem wellFormed_of_mem_outcomes {ρ : Roll} {e : List (Int × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.outcomes) : (Roll.fixed e.1).WellFormed := by
  have hdice : ∀ (n : Nat) (die : List (Int × Nat)),
      die.Pairwise (fun a b => a.1 < b.1) → e ∈ diceOutcomes n die →
        (Roll.fixed e.1).WellFormed := by
    intro n die hdie he
    unfold diceOutcomes at he
    split at he
    · simp_all [WellFormed]
    split at he
    · simp_all [WellFormed]; omega
    obtain ⟨hpw, hmem⟩ := Outcomes.mem_outcomeStates die hdie n e he
    exact ⟨hpw, fun p hp => (hmem p hp).1⟩
  cases ρ with
  | range start stop =>
    simp only [outcomes] at he
    split at he
    · simp_all [WellFormed]
    · simp only [List.mem_map, List.mem_range] at he
      obtain ⟨i, _, rfl⟩ := he
      simp [WellFormed]
  | standard count faces => exact hdice _ _ (standardFaces_sorted faces) he
  | custom count faces => exact hdice _ _ hρ.1 he
  | fixed rs =>
    simp only [outcomes, List.mem_singleton] at he
    subst he
    exact hρ
  | pool outcomes => exact (hρ.2.1 e he).1
  | chains | stacked => simp [outcomes] at he

end Roll

/-! ## The law of a roll, sorted -/

namespace Outcomes

/--
A sum against a point mass reads its point.

# Parameters
- `a`: The point.
- `f`: The function summed.
-/
theorem tsum_pure_mul {α : Type} (a : α) (f : α → ℝ≥0∞) :
    ∑' x, PMF.pure a x * f x = f a := by
  rw [tsum_eq_single a fun x hx => by simp [PMF.pure_apply, hx]]
  simp

/--
A sorted record sorts to its own results.

# Parameters
- `r`: The record.

# Hypotheses
- `h`: Its results are sorted.
-/
theorem sorted_of_pairwise {r : Record} (h : r.results.Pairwise (· ≤ ·)) :
    r.sorted = r.results :=
  List.mergeSort_of_pairwise (by simpa using h)

/--
Dice of one face roll it every time.

# Parameters
- `x`: The face.
- `J`: The number of dice.
-/
theorem rollDice_pure (x : Int) :
    ∀ J : Nat, rollDice (J : Int) (PMF.pure x) =
      PMF.pure { results := List.replicate J x }
  | 0 => rfl
  | J + 1 => by
    rw [OrderStatistics.rollDice_succ, rollDice_pure x J, PMF.pure_bind,
      PMF.pure_map]
    simp [Record.push, List.replicate_succ']

/--
A standard die is the die of its faces, each of weight one, even if it has
none.

# Parameters
- `faces`: The number of faces.
-/
theorem standardDie_eq_weightedDie (faces : Int) :
    standardDie faces = weightedDie (standardFaces faces) := by
  by_cases h : 0 < faces
  · exact OrderStatistics.standardDie_eq h
  · have : standardFaces faces = [] := by
      simp [standardFaces, show faces.toNat = 0 by omega]
    rw [this]
    simp [standardDie, weightedDie, customDie, show faces ≤ 0 by omega]

/--
The law of `n` dice of a die, sorted, is their outcomes over the total weight
of the die raised to `n`, or over `1` if there are no dice or the die has no
faces.

# Parameters
- `n`: The number of dice.
- `ψ`: The function of the sorted roll.

# Hypotheses
- `sorted`: The faces ascend strictly.
- `hpos`: Every face weighs at least one.
-/
theorem tsum_rollDice_sorted {die : List (Int × Nat)}
    (sorted : die.Pairwise (fun a b => a.1 < b.1)) (hpos : ∀ p ∈ die, 0 < p.2)
    (n : Nat) (ψ : List Int → Nat) :
    ∑' r, rollDice (n : Int) (weightedDie die) r * (ψ r.sorted : ℝ≥0∞) =
      (integral (Roll.diceOutcomes n die) (fun rs => ψ (Roll.expand rs)) :
          ℝ≥0∞) /
        ((if 0 < n ∧ die ≠ [] then OrderStatistics.total die ^ n else 1 :
          Nat) : ℝ≥0∞) := by
  unfold Roll.diceOutcomes
  by_cases hn : n = 0
  · subst hn
    rw [show rollDice ((0 : Nat) : Int) (weightedDie die) = PMF.pure {}
      from rfl, tsum_pure_mul]
    simp [integral, Roll.expand, Record.sorted]
  by_cases hd : die = []
  · subst hd
    rw [show weightedDie [] = PMF.pure 0 by simp [weightedDie, customDie],
      rollDice_pure, tsum_pure_mul,
      sorted_of_pairwise (List.pairwise_replicate.mpr (.inr le_rfl))]
    simp [hn, integral, Roll.expand]
  have hW : OrderStatistics.total die ≠ 0 := by
    obtain ⟨p, ps, rfl⟩ := List.exists_cons_of_ne_nil hd
    have := hpos p (by simp)
    simp [OrderStatistics.total]
    omega
  simp only [hn, hd, ↓reduceIte, Nat.pos_of_ne_zero hn, ne_eq,
    not_false_eq_true, and_self]
  rw [tsum_rollDice die hW n ψ, integral_outcomeStates die sorted n ψ,
    Nat.cast_pow]

/--
The law of a nonempty range, sorted, is its outcomes over its width.

# Parameters
- `ψ`: The function of the sorted roll.

# Hypotheses
- `h`: The range is not empty.
-/
theorem tsum_rollRange_sorted {start stop : Int} (h : start ≤ stop)
    (ψ : List Int → Nat) :
    ∑' r, rollRange start stop r * (ψ r.sorted : ℝ≥0∞) =
      (((List.range (stop - start + 1).toNat).map fun (i : Nat) =>
          ψ [start + i]).sum : ℝ≥0∞) / ((stop - start + 1).toNat : ℝ≥0∞) := by
  set faces : List (Int × Nat) :=
    (List.range (stop - start + 1).toNat).map fun (i : Nat) => (start + i, 1)
  have hexpand : Roll.expand faces =
      (List.range (stop - start + 1).toNat).map fun (i : Nat) => start + i := by
    simp only [faces, Roll.expand, List.flatMap_map]
    induction (List.range (stop - start + 1).toNat) with
    | nil => rfl
    | cons i l ih => simp_all
  have hne : Roll.expand faces ≠ [] := by
    rw [hexpand]
    simp only [ne_eq, List.map_eq_nil_iff, List.range_eq_nil]
    omega
  have hdie : PMF.ofMultiset (Finset.Icc start stop).val
      (by simp; omega) = weightedDie faces := by
    unfold weightedDie customDie
    simp only [show List.flatMap _ faces ≠ [] from hne, ↓reduceDIte]
    congr 1
    rw [show List.flatMap _ faces = Roll.expand faces from rfl, hexpand,
      icc_val]
  have htotal : OrderStatistics.total faces = (stop - start + 1).toNat := by
    simp [OrderStatistics.total, faces, Function.comp_def]
  unfold rollRange
  simp only [show ¬ stop < start by omega, ↓reduceDIte]
  rw [hdie, tsum_map_mul,
    tsum_weightedDie faces (by rw [htotal]; omega), htotal]
  simp only [Record.sorted, List.mergeSort_singleton]
  simp [faces, Function.comp_def]

/--
A sum against weighted values weighs each value by its weight over the total.

# Parameters
- `default`: The default.
- `l`: The values, each with its weight.
- `f`: The function summed.

# Hypotheses
- `hW`: The total weight is positive.
-/
theorem tsum_weighted {α : Type} (default : α)
    (l : List (α × Nat)) (hW : (l.map (·.2)).sum ≠ 0)
    (f : α → ℝ≥0∞) :
    ∑' x, weighted default l x * f x =
      (l.map fun e => (e.2 : ℝ≥0∞) * f e.1).sum /
        (((l.map (·.2)).sum : Nat) : ℝ≥0∞) := by
  classical
  have hsum : ∀ l : List (α × Nat),
      ((l.flatMap fun p => List.replicate p.2 p.1).map f).sum =
        (l.map fun e => (e.2 : ℝ≥0∞) * f e.1).sum := by
    intro l
    induction l with
    | nil => rfl
    | cons e l ih => simp [List.flatMap_cons, List.sum_append, ← ih]
  rw [← hsum]
  set xs := l.flatMap fun p => List.replicate p.2 p.1 with hxs
  have hlen : xs.length = (l.map (·.2)).sum := by
    simp [xs, List.length_flatMap]
  have hne : xs ≠ [] := fun h => hW (by rw [← hlen, h]; rfl)
  have hlaw : weighted default l = PMF.ofMultiset (xs : Multiset α)
      (by simpa using hne) := by
    unfold weighted
    simp only [← hxs, hne, ↓reduceDIte]
  rw [hlaw, ← hlen]
  rw [tsum_eq_sum (s := (xs : Multiset α).toFinset) fun x hx => by
    simp only [PMF.ofMultiset_apply]
    rw [Multiset.count_eq_zero.mpr (by simpa using hx)]
    simp]
  simp only [PMF.ofMultiset_apply, ENNReal.div_eq_inv_mul]
  have : (xs.map f).sum = ((xs : Multiset α).map f).sum := by simp
  rw [this, Finset.sum_multiset_map_count, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [nsmul_eq_mul, Multiset.coe_card]
  exact mul_assoc _ _ _

/--
A transformed law gives each value the mass of the outcomes that it comes
from.

# Parameters
- `p`: The law.
- `f`: The transformation.
- `v`: The value.
-/
theorem map_apply_eq_tsum {α β : Type} [DecidableEq β] (p : PMF α)
    (f : α → β) (v : β) :
    (p.map f) v = ∑' a, p a * ((if f a = v then 1 else 0 : Nat) : ℝ≥0∞) := by
  rw [PMF.map_apply]
  refine tsum_congr fun a => ?_
  by_cases h : f a = v <;> simp [h, Ne.symm]

/-! ## Multisets of any faces -/

/--
The weighed draws of a die of faces of any type: the sum, over every sequence
of `J` draws, of the product of their faces' weights times `ψ` of the multiset
of the draws, as `OrderStatistics.words` weighs the rolls of a die of results
by `φ` of the roll sorted.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The faces of the die, each with its weight.
- `J`: The number of draws.
- `ψ`: The function of the multiset of the draws.

# Returns
The weighed sum.

# Examples
```lean
#guard Outcomes.bagWords [((1 : Int), 1), (2, 3)] 2
  (fun M => (M.map Int.toNat).sum) == 56
```
-/
def bagWords {α : Type} (die : List (α × Nat)) :
    Nat → (Multiset α → Nat) → Nat
  | 0, ψ => ψ 0
  | J + 1, ψ =>
    (die.map fun p => p.2 * bagWords die J (fun M => ψ (p.1 ::ₘ M))).sum

/--
No draws weigh the empty multiset once.

# Parameters
- `die`: The die.
- `ψ`: The function of the multiset.
-/
theorem bagWords_zero {α : Type} (die : List (α × Nat))
    (ψ : Multiset α → Nat) : bagWords die 0 ψ = ψ 0 := rfl

/--
One more draw shows each face beside the others.

# Parameters
- `die`: The die.
- `J`: The number of draws before it.
- `ψ`: The function of the multiset.
-/
theorem bagWords_succ {α : Type} (die : List (α × Nat)) (J : Nat)
    (ψ : Multiset α → Nat) :
    bagWords die (J + 1) ψ =
      (die.map fun p => p.2 * bagWords die J (fun M => ψ (p.1 ::ₘ M))).sum :=
  rfl

/--
The draws of a die weigh its total weight raised to their number in all.

# Parameters
- `die`: The die.
- `J`: The number of draws.
-/
theorem bagWords_one {α : Type} (die : List (α × Nat)) :
    ∀ J : Nat, bagWords die J (fun _ => 1) = (die.map (·.2)).sum ^ J
  | 0 => rfl
  | J + 1 => by
    rw [bagWords_succ, pow_succ, mul_comm, ← List.sum_map_mul_right]
    simp only [bagWords_one die J]

/--
Peeling the last face: drawing from a die whose last face is `v`, of weight
`w`, draws `m` copies of `v` at `C(J, m) · wᵐ` ways, and the rest from the
other faces. This is the step of the program, from the faces before `v` to
`v`, as `OrderStatistics.words_peel` is for results; a multiset needs no order,
so `v` may be any face.

# Parameters
- `w`: The weight of the last face.
- `J`: The number of draws.
- `ψ`: The function of the multiset.
-/
theorem bagWords_peel {α : Type} (die : List (α × Nat)) (v : α) (w : Nat) :
    ∀ (J : Nat) (ψ : Multiset α → Nat),
      bagWords (die ++ [(v, w)]) J ψ =
        ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          bagWords die (J - m) (fun M => ψ (M + Multiset.replicate m v))
  | 0, ψ => by simp [bagWords_zero]
  | J + 1, ψ => by
    -- Each other face draws beside the copies of `v`, and `v` joins them.
    have hlow : ∀ p ∈ die, bagWords (die ++ [(v, w)]) J
        (fun M => ψ (p.1 ::ₘ M)) =
        ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          bagWords die (J - m)
            (fun M => ψ (p.1 ::ₘ M + Multiset.replicate m v)) := by
      intro p _
      rw [bagWords_peel die v w J]
      refine Finset.sum_congr rfl fun m _ => ?_
      simp only [Multiset.cons_add]
    have htop : bagWords (die ++ [(v, w)]) J (fun M => ψ (v ::ₘ M)) =
        ∑ m ∈ Finset.range (J + 1), J.choose m * w ^ m *
          bagWords die (J - m)
            (fun M => ψ (M + Multiset.replicate (m + 1) v)) := by
      rw [bagWords_peel die v w J]
      refine Finset.sum_congr rfl fun m _ => ?_
      simp only [Multiset.replicate_succ, Multiset.add_cons]
    rw [bagWords_succ, List.map_append, List.sum_append, List.map_singleton,
      List.sum_singleton, htop]
    rw [List.map_congr_left fun p hp => congrArg (p.2 * ·) (hlow p hp)]
    simp only [Finset.mul_sum]
    rw [OrderStatistics.list_sum_finset_sum]
    -- The other faces fold back into the draws of one more.
    have hfold : ∀ i ∈ Finset.range (J + 1),
        (die.map fun p => p.2 * (J.choose i * w ^ i *
          bagWords die (J - i) fun M =>
            ψ (p.1 ::ₘ M + Multiset.replicate i v))).sum
        = J.choose i * w ^ i * bagWords die (J + 1 - i)
            (fun M => ψ (M + Multiset.replicate i v)) := by
      intro i hi
      have hi : i ≤ J := Nat.lt_succ_iff.mp (Finset.mem_range.mp hi)
      rw [show J + 1 - i = (J - i) + 1 by omega, bagWords_succ,
        ← List.sum_map_mul_left]
      refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
      ring
    rw [Finset.sum_congr rfl hfold]
    have hshift : ∀ i ∈ Finset.range (J + 1),
        w * (J.choose i * w ^ i * bagWords die (J - i)
          fun M => ψ (M + Multiset.replicate (i + 1) v))
        = J.choose i * w ^ (i + 1) * bagWords die (J + 1 - (i + 1))
            (fun M => ψ (M + Multiset.replicate (i + 1) v)) := by
      intro i _
      rw [show J + 1 - (i + 1) = J - i by omega]
      ring
    rw [Finset.sum_congr rfl hshift]
    exact OrderStatistics.pascal_sum w J fun m =>
      bagWords die (J + 1 - m) (fun M => ψ (M + Multiset.replicate m v))

/--
The program's state of `J` draws, integrated against `ψ` of each multiset, is
the weighed draws of `J` faces of the die, for faces of any type, as
`integral_outcomeStates` finds of results read sorted.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `die`: The faces, each with its weight.
- `J`: The number of draws.
- `ψ`: The function of the multiset.
-/
theorem integral_outcomeStates_bag {α : Type} (die : List (α × Nat))
    (J : Nat) (ψ : Multiset α → Nat) :
    integral (outcomeStates die J) (fun rs => ψ (Roll.expand rs)) =
      bagWords die J ψ := by
  induction die using List.reverseRecOn generalizing J ψ with
  | nil =>
    cases J with
    | zero => simp [outcomeStates, initial, integral, bagWords_zero, Roll.expand]
    | succ J => simp [outcomeStates, initial, integral, bagWords_succ]
  | append_singleton die face ih =>
    obtain ⟨v, w⟩ := face
    rw [outcomeStates_append, bagWords_peel]
    simp only [outcomeStep, integral_flatMap, list_range_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [integral_map _ (fun rs => grow rs v m)]
    simp only [expand_grow, ← Multiset.coe_add, Multiset.coe_replicate]
    exact congrArg _ (ih (J - m) fun M => ψ (M + Multiset.replicate m v))

/--
One more independent draw is a draw beside the others.

# Parameters
- `n`: The number of draws before it.
- `P`: The law of each.
-/
theorem iidBag_succ {β : Type} (n : Nat) (P : PMF β) :
    iidBag (n + 1) P = P.bind fun x => (iidBag n P).map (x ::ₘ ·) := by
  show (iidBag n P).bind (fun M => P.map (· ::ₘ M)) = _
  simp only [← PMF.bind_pure_comp]
  exact PMF.bind_comm _ _ _

/--
Independent draws of weighted faces weigh each multiset by the weighed draws
that reach it, over the total weight raised to their number.

# Type parameters
- `α`: The type of the faces.

# Parameters
- `d`: The face if no weight were positive.
- `die`: The faces, each with its weight.
- `J`: The number of draws.
- `ψ`: The function of the multiset.

# Hypotheses
- `hW`: The total weight is positive.
-/
theorem tsum_iidBag {α : Type} (d : α) (die : List (α × Nat))
    (hW : (die.map (·.2)).sum ≠ 0) :
    ∀ (J : Nat) (ψ : Multiset α → Nat),
      ∑' M, iidBag J (weighted d die) M * (ψ M : ℝ≥0∞) =
        (bagWords die J ψ : ℝ≥0∞) / (((die.map (·.2)).sum : Nat) : ℝ≥0∞) ^ J
  | 0, ψ => by
    rw [show iidBag 0 (weighted d die) = PMF.pure 0 from rfl, tsum_pure_mul]
    simp [bagWords_zero]
  | J + 1, ψ => by
    set W := (die.map (·.2)).sum
    have hW' : (W : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW
    rw [iidBag_succ, OrderStatistics.tsum_bind_mul, tsum_weighted d die hW]
    simp only [tsum_map_mul, tsum_iidBag d die hW J, bagWords_succ,
      Nat.cast_list_sum, List.map_map, Function.comp_def, Nat.cast_mul,
      div_eq_mul_inv, ← List.sum_map_mul_right]
    refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
    have hc : (die.map fun x => (x.2 : ℝ≥0∞)).sum = (W : ℝ≥0∞) := by
      simp [W, Nat.cast_list_sum, Function.comp_def]
    rw [hc, pow_succ, ENNReal.mul_inv (a := (W : ℝ≥0∞) ^ J) (b := W)
      (.inr (ENNReal.natCast_ne_top _)) (.inr hW')]
    ring

/--
The multisets of `n` draws of weighted faces of any type, over their weights,
are the multiset of `n` independent draws of the faces, as `multisets` in
`record.rs` weighs them, so that each holds the multinomial coefficient of
its copies times the product of their weights.

# Type parameters
- `α`: The type of the faces: results, or chains.

# Parameters
- `d`: The face if no weight were positive.
- `d₀`: The multiset if no weight were positive.
- `die`: The faces, each with its weight.
- `n`: The number of draws.

# Hypotheses
- `hW`: The total weight of the faces is positive.
-/
theorem weighted_outcomeStates {α : Type} [DecidableEq α] (d : α)
    (d₀ : List (α × Nat)) (die : List (α × Nat))
    (hW : (die.map (·.2)).sum ≠ 0) (n : Nat) :
    (weighted d₀ (outcomeStates die n)).map
        (fun rs => (Roll.expand rs : Multiset α)) =
      iidBag n (weighted d die) := by
  -- The multisets weigh the total weight raised to `n` in all.
  have htot : ((outcomeStates die n).map (·.2)).sum =
      (die.map (·.2)).sum ^ n := by
    have := integral_outcomeStates_bag die n fun _ => 1
    rw [bagWords_one] at this
    simpa [integral] using this
  have hT : ((outcomeStates die n).map (·.2)).sum ≠ 0 := by
    rw [htot]
    exact pow_ne_zero _ hW
  ext M
  rw [map_apply_eq_tsum, tsum_weighted _ _ hT]
  have hrhs : iidBag n (weighted d die) M =
      ∑' M', iidBag n (weighted d die) M' *
        ((if M' = M then 1 else 0 : Nat) : ℝ≥0∞) := by
    rw [tsum_eq_single M fun M' h => by simp [h]]
    simp
  rw [hrhs, tsum_iidBag d die hW, ← integral_outcomeStates_bag, htot,
    Nat.cast_pow]
  congr 1
  simp [integral, Function.comp_def]

end Outcomes

namespace Roll

/--
A roll's outcomes, over its total, are the law of the record that it fills,
sorted: a sum of `ψ` of the sorted record against the law is the outcomes'
integral of `ψ` of their results, over the total.

# Parameters
- `ρ`: The roll.
- `ψ`: The function of the sorted record.

# Hypotheses
- `hρ`: The roll is well formed.
- `hp`: The roll holds no chains.
-/
theorem tsum_law_sorted (ρ : Roll) (hρ : ρ.WellFormed) (hp : ρ.Plain)
    (ψ : List Int → Nat) :
    ∑' r, ρ.law r * (ψ r.sorted : ℝ≥0∞) =
      (OrderStatistics.integral ρ.outcomes (fun rs => ψ (expand rs)) : ℝ≥0∞) /
        (ρ.total : ℝ≥0∞) := by
  cases ρ with
  | range start stop =>
    by_cases h : stop < start
    · simp only [law, rollRange, h, ↓reduceDIte, Outcomes.tsum_pure_mul,
        outcomes, total, ↓reduceIte]
      simp [OrderStatistics.integral, expand, Record.sorted]
    · simp only [law, outcomes, total, h, ↓reduceIte]
      rw [Outcomes.tsum_rollRange_sorted (by omega)]
      simp [OrderStatistics.integral, expand, Function.comp_def]
  | standard count faces =>
    simp only [law, outcomes, total, Outcomes.standardDie_eq_weightedDie,
      tsum_map_mul, Record.sorted_die]
    rw [rollDice_toNat,
      Outcomes.tsum_rollDice_sorted (standardFaces_sorted faces)
        (by simp [standardFaces]) count.toNat ψ]
    congr 2
    have hne : standardFaces faces ≠ [] ↔ 0 < faces := by
      simp [standardFaces]
    simp only [hne, total_standardFaces]
    split_ifs <;> omega
  | custom count faces =>
    simp only [law, outcomes, total, tsum_map_mul, Record.sorted_die]
    rw [rollDice_toNat, Outcomes.tsum_rollDice_sorted hρ.1 hρ.2 count.toNat ψ]
    simp only [show 0 < count.toNat ↔ 0 < count by omega]
  | fixed rs =>
    simp only [law, Outcomes.tsum_pure_mul, outcomes, total,
      Outcomes.sorted_of_pairwise (r := { results := expand rs })
        (expand_pairwise hρ.1)]
    simp [OrderStatistics.integral]
  | pool es =>
    obtain ⟨hne, hmem, _⟩ := hρ
    have hW : (es.map (·.2)).sum ≠ 0 := by
      obtain ⟨e, es, rfl⟩ := List.exists_cons_of_ne_nil hne
      have := (hmem e (by simp)).2
      simp only [List.map_cons, List.sum_cons]
      omega
    simp only [law, outcomes, total, tsum_map_mul]
    rw [Outcomes.tsum_weighted _ es hW]
    congr 1
    simp only [OrderStatistics.integral, Nat.cast_list_sum, List.map_map]
    refine congrArg List.sum (List.map_congr_left fun e he => ?_)
    simp [Outcomes.sorted_of_pairwise (r := { results := expand e.1 })
      (expand_pairwise (hmem e he).1.1)]
  | chains | stacked => exact hp.elim

/--
The law of the results of the record that a roll fills, sorted, gives each
sorted list of results the weight of the outcomes that hold it, over the
total, as `Roll::outcomes` weighs them.

# Parameters
- `ρ`: The roll.
- `ys`: The sorted results.

# Hypotheses
- `hρ`: The roll is well formed.
- `hp`: The roll holds no chains.
-/
theorem map_sorted_law (ρ : Roll) (hρ : ρ.WellFormed) (hp : ρ.Plain)
    (ys : List Int) :
    (ρ.law.map Record.sorted) ys =
      (OrderStatistics.integral ρ.outcomes
          (fun rs => if expand rs = ys then 1 else 0) : ℝ≥0∞) /
        (ρ.total : ℝ≥0∞) := by
  rw [Outcomes.map_apply_eq_tsum]
  exact ρ.tsum_law_sorted hρ hp fun xs => if xs = ys then 1 else 0

end Roll

/-! ## Setups -/

namespace Setup

/--
A setup as `propagation/mixture.rs` keeps it: its roll, if any, is well
formed.

# Parameters
- `s`: The setup.

# Returns
Whether the setup is well formed.
-/
def WellFormed (s : Setup) : Prop := ∀ ρ ∈ s.roll, ρ.WellFormed

/--
The total weight of a setup: that of its roll, or `1` if it has none. Mirrors
`total` of `Setup` in `mixture.rs`.

# Parameters
- `s`: The setup.

# Returns
The total.
-/
def total (s : Setup) : Nat := (s.roll.map Roll.total).getD 1

end Setup

/--
The law of the record that a mixture describes, read sorted, mixes the laws
of its setups, read sorted. With `Setup.map_toOracle_law`, in
`XdySpec/Splits.lean`, the outcomes of every setup, each scaled to the common
denominator and weighed by its setup, are the law of the mixture read sorted,
as `Dist.hasLaw_mixture` cancels the scaling.

# Parameters
- `μ`: The law of the setups.
-/
theorem map_toOracle_mixtureLaw (μ : PMF Setup) :
    (mixtureLaw μ).map Record.toOracle =
      μ.bind fun s => s.law.map Record.toOracle :=
  PMF.map_bind _ _ _

/-! ## Splitting on a record read sorted -/

namespace Outcomes

/--
Replacing a record of the oracle's state by itself changes nothing.

# Parameters
- `s`: The state.
- `d`: The rolling record.
-/
theorem setRecord_record (s : Xdy.State) (d : Nat) :
    s.setRecord d (s.record d) = s := by
  rcases s with ⟨registers, records⟩
  simp only [Xdy.State.setRecord, Xdy.State.record, Xdy.State.mk.injEq,
    true_and]
  apply Array.ext
  · simp
  · intro i _ _
    simp only [Array.getElem_modify]
    split <;> simp_all

/--
Every record of a state read sorted is sorted, its results in ascending order
and its chains by `Xdy.Chain.le`.

# Parameters
- `s`: The state.
-/
theorem toOracle_sorted (s : State) :
    ∀ r ∈ s.toOracle.records, r.results.Pairwise (· ≤ ·) ∧
      r.chains.Pairwise (fun a b => a.le b = true) := by
  intro r hr
  simp only [State.toOracle, Array.mem_map] at hr
  obtain ⟨a, _, rfl⟩ := hr
  exact ⟨Record.sorted_pairwise a, Record.chains_pairwise a⟩

/--
An explode depends on its record only through the record read sorted: its
kept chains and its die.

# Parameters
- `r`, `r'`: The records.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none`
  for no limit.

# Hypotheses
- `h`: The records read alike sorted.
-/
private theorem explode_congr {r r' : Record}
    (h : r.toOracle = r'.toOracle) (c : Criterion) (limit : Option Nat) :
    explode r c limit = explode r' c limit := by
  have hk : r.keptChains = r'.keptChains := by
    rw [← Record.keptChains_toOracle, h, Record.keptChains_toOracle]
  have hd : r.die = r'.die := congrArg Xdy.Record.die h
  cases limit <;> simp only [explode, hk, hd]

/--
A step's law, read sorted, depends only on the state read sorted: it is the
step's law from the sorted state, read sorted. Both have the law of the
oracle's step from the sorted state, by `hasLaw_step`, and a distribution
has one law.

# Parameters
- `s`: The state.
- `i`: The instruction.
-/
theorem map_toOracle_step (s : State) (i : Instruction) :
    (step s i).map State.toOracle =
      (step (State.ofOracle s.toOracle) i).map State.toOracle := by
  have hs : (State.ofOracle s.toOracle).toOracle = s.toOracle :=
    State.toOracle_ofOracle _ (toOracle_sorted s)
  cases hb : unbounded s i
  · have h₁ := hasLaw_step s i hb
    have h₂ := hasLaw_step (State.ofOracle s.toOracle) i (by
      rw [← State.unbounded_toOracle, hs, State.unbounded_toOracle, hb])
    rw [hs] at h₂
    exact PMF.ext fun t => (h₁.2.2 t).trans (h₂.2.2 t).symm
  · -- Only an explode may draw without bound, and it reads its record only
    -- through its kept chains and its die, which sorting leaves alone.
    cases i <;> simp only [unbounded, Bool.false_eq_true] at hb
    case explode d p l =>
      have hv : (State.ofOracle s.toOracle).value = s.value := funext fun op =>
        by rw [← State.value_toOracle, hs, State.value_toOracle]
      have hr : ((State.ofOracle s.toOracle).record d).toOracle =
          (s.record d).toOracle := by
        rw [← State.record_toOracle, hs, State.record_toOracle]
      simp only [step, PMF.map_comp, hv, explode_congr hr]
      congr 1
      funext x
      simp only [Function.comp_apply, State.setRecord_toOracle, hs]

/--
A step's law of states, read sorted, is a function of the law of states
before it, read sorted.

# Parameters
- `μ`: The law of states before the step.
- `i`: The instruction.
-/
theorem map_toOracle_bind_step (μ : PMF State) (i : Instruction) :
    (μ.bind (step · i)).map State.toOracle =
      (μ.map State.toOracle).bind fun o =>
        (step (State.ofOracle o) i).map State.toOracle := by
  rw [PMF.map_bind, PMF.bind_map]
  exact congrArg μ.bind (funext fun s => map_toOracle_step s i)

end Outcomes

/--
Sorting commutes with execution: two laws of states that agree read sorted
agree read sorted after any instructions, since no instruction can tell the
order of a record's results.

# Parameters
- `is`: The instructions.

# Hypotheses
- `h`: The laws agree read sorted before the instructions.
-/
theorem map_toOracle_exec (is : List Instruction) {μ ν : PMF State}
    (h : μ.map State.toOracle = ν.map State.toOracle) :
    (exec is μ).map State.toOracle = (exec is ν).map State.toOracle := by
  induction is generalizing μ ν with
  | nil => exact h
  | cons i is ih =>
    rw [exec_cons, exec_cons]
    exact ih (by rw [Outcomes.map_toOracle_bind_step,
      Outcomes.map_toOracle_bind_step, h])

/--
Splitting a rolling record on its results read sorted, with its drops, and
fixing the record of each world at its outcome, as `split` in
`propagation.rs` does, answers the law of the program read sorted: each world
weighs its outcome's probability, and, though its states hold the record in
every order that sorts to the outcome, fixing it at the sorted outcome
changes nothing that any later instruction can tell.

# Parameters
- `is`: The instructions after the split.
- `μ`: The law of states at the split.
- `d`: The rolling record.
-/
theorem exec_split_sorted (is : List Instruction) (μ : PMF State) (d : Nat) :
    (exec is μ).map State.toOracle =
      (μ.map fun s => (s.record d).toOracle).bind fun v =>
        (exec is ((cond μ (fun s => (s.record d).toOracle) v).map
          fun s => s.setRecord d (Record.ofOracle v))).map State.toOracle := by
  rw [exec_split is μ (fun s => (s.record d).toOracle), PMF.map_bind]
  refine bind_congr_support fun v hv => map_toOracle_exec is ?_
  rw [PMF.map_comp]
  refine map_congr_support fun s hs => ?_
  have hsv := eq_of_mem_support_cond hv hs
  subst hsv
  rw [Function.comp_apply, State.setRecord_toOracle,
    Record.toOracle_ofOracle _ (Record.sorted_pairwise (s.record d))
      (Record.chains_pairwise (s.record d)),
    ← State.record_toOracle, Outcomes.setRecord_record]

/--
Splitting a rolling record on its results read sorted, and fixing each
world's record at its outcome, answers the law of every operand, and so of
the program's answer, since sorting leaves every register alone.

# Parameters
- `is`: The instructions after the split.
- `μ`: The law of states at the split.
- `d`: The rolling record.
- `op`: The operand.
-/
theorem exec_split_sorted_value (is : List Instruction) (μ : PMF State)
    (d : Nat) (op : AddressingMode) :
    (exec is μ).map (·.value op) =
      (μ.map fun s => (s.record d).toOracle).bind fun v =>
        (exec is ((cond μ (fun s => (s.record d).toOracle) v).map
          fun s => s.setRecord d (Record.ofOracle v))).map (·.value op) := by
  have hv : (fun s : State => s.value op) =
      (fun s : Xdy.State => s.value op) ∘ State.toOracle :=
    funext fun s => (State.value_toOracle s op).symm
  rw [hv, ← PMF.map_comp, exec_split_sorted is μ d, PMF.map_bind]
  simp only [PMF.map_comp]

end Xdy.Spec
