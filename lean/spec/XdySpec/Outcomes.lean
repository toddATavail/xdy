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
- So every roll's outcomes (`Roll.outcomes`), over its total (`Roll.total`),
  are the law of the record that it fills, sorted
  (`Roll.tsum_law_sorted`, `Roll.map_sorted_law`), and each outcome is fixed
  results in ascending order (`Roll.wellFormed_of_mem_outcomes`).
- A setup's outcomes keep its drops, and are the law of its record read as
  the oracle reads it, sorted (`Setup.tsum_law_toOracle`,
  `Setup.map_toOracle_law`); each describes a certain record, already sorted
  (`Setup.law_of_mem_outcomes`). A mixture read sorted mixes its setups read
  sorted (`map_toOracle_mixtureLaw`), and `Dist.hasLaw_mixture` cancels the
  scaling of each setup's outcomes to the common denominator.
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
    S["Setup.tsum_law_toOracle:<br/>with the drops kept"]
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
Place copies of a face after a multiset, as `Roll::outcomes` pushes them only
if there are any.

# Parameters
- `rs`: The multiset: each distinct result, in ascending order, with its
  copies.
- `v`: The face, greater than every result of `rs`.
- `m`: The copies.

# Returns
The multiset with `m` copies of `v` after its results, or `rs` if `m = 0`.

# Examples
```lean
#guard Outcomes.grow [(1, 2)] 3 0 == [(1, 2)]
#guard Outcomes.grow [(1, 2)] 3 2 == [(1, 2), (3, 2)]
```
-/
def grow (rs : List (Int × Nat)) (v : Int) (m : Nat) : List (Int × Nat) :=
  if m = 0 then rs else rs ++ [(v, m)]

/--
Growing a multiset by copies of a face appends them to its results.

# Parameters
- `rs`: The multiset.
- `v`: The face.
- `m`: The copies.
-/
theorem expand_grow (rs : List (Int × Nat)) (v : Int) (m : Nat) :
    Roll.expand (grow rs v m) = Roll.expand rs ++ List.replicate m v := by
  unfold grow
  split <;> simp_all [Roll.expand]

/--
Place every number of copies of one face, of value `v` and weight `w`. Mirrors
the loop over the faces of `Roll::outcomes` in `record.rs`: the state of `J`
results gathers, for each number `m` of copies, the multisets of `J - m`
results, each grown by the copies and weighed by `C(J, m) · wᵐ`, which is the
Rust's `C(j + m, m) · wᵐ` with `j = J - m`.

# Parameters
- `states`: The states before the face: for each number of results, the
  multisets and their weights.
- `face`: The face and its weight.

# Returns
The states after the face.
-/
def outcomeStep (states : Nat → List (List (Int × Nat) × Nat))
    (face : Int × Nat) : Nat → List (List (Int × Nat) × Nat) :=
  fun J => (List.range (J + 1)).flatMap fun m =>
    (states (J - m)).map fun e =>
      (grow e.1 face.1 m, J.choose m * face.2 ^ m * e.2)

/-- The states before any face: no results, the empty multiset once. -/
def initial : Nat → List (List (Int × Nat) × Nat) :=
  fun j => if j = 0 then [([], 1)] else []

/--
The states of the program after every face of a die, in order.

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
def outcomeStates (die : List (Int × Nat)) :
    Nat → List (List (Int × Nat) × Nat) :=
  die.foldl outcomeStep initial

/--
One more face takes one more step.

# Parameters
- `die`: The faces before it.
- `face`: The face.
-/
theorem outcomeStates_append (die : List (Int × Nat)) (face : Int × Nat) :
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
them, ascend strictly, each at least once.

# Parameters
- `ρ`: The roll.

# Returns
Whether the roll is well formed. Ranges and standard dice always are.
-/
def WellFormed : Roll → Prop
  | .custom _ faces | .fixed faces =>
    faces.Pairwise (fun a b => a.1 < b.1) ∧ ∀ p ∈ faces, 0 < p.2
  | _ => True

/--
The total weight of a roll: the number of its branches. Mirrors `total` in
`record.rs`.

# Parameters
- `ρ`: The roll.

# Returns
The width of a range, or the total weight of the die raised to the number of
dice, or `1` for an empty range, no dice, dice without faces, and fixed
results.

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
[total](Roll.total).

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
-/
theorem tsum_law_sorted (ρ : Roll) (hρ : ρ.WellFormed) (ψ : List Int → Nat) :
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
    simp only [law, outcomes, total, Outcomes.standardDie_eq_weightedDie]
    rw [rollDice_toNat,
      Outcomes.tsum_rollDice_sorted (standardFaces_sorted faces)
        (by simp [standardFaces]) count.toNat ψ]
    congr 2
    have hne : standardFaces faces ≠ [] ↔ 0 < faces := by
      simp [standardFaces]
    simp only [hne, total_standardFaces]
    split_ifs <;> omega
  | custom count faces =>
    simp only [law, outcomes, total]
    rw [rollDice_toNat, Outcomes.tsum_rollDice_sorted hρ.1 hρ.2 count.toNat ψ]
    simp only [show 0 < count.toNat ↔ 0 < count by omega]
  | fixed rs =>
    simp only [law, Outcomes.tsum_pure_mul, outcomes, total,
      Outcomes.sorted_of_pairwise (r := { results := expand rs })
        (expand_pairwise hρ.1)]
    simp [OrderStatistics.integral]

/--
The law of the results of the record that a roll fills, sorted, gives each
sorted list of results the weight of the outcomes that hold it, over the
total, as `Roll::outcomes` weighs them.

# Parameters
- `ρ`: The roll.
- `ys`: The sorted results.

# Hypotheses
- `hρ`: The roll is well formed.
-/
theorem map_sorted_law (ρ : Roll) (hρ : ρ.WellFormed) (ys : List Int) :
    (ρ.law.map Record.sorted) ys =
      (OrderStatistics.integral ρ.outcomes
          (fun rs => if expand rs = ys then 1 else 0) : ℝ≥0∞) /
        (ρ.total : ℝ≥0∞) := by
  rw [Outcomes.map_apply_eq_tsum]
  exact ρ.tsum_law_sorted hρ fun xs => if xs = ys then 1 else 0

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

/--
Every outcome of a setup: the setup with each multiset of its roll's results
fixed, keeping its drops, or the setup itself if it has no roll. Mirrors the
outcomes of each setup in `Mixture::outcomes` in `mixture.rs`, before it
scales them to the common denominator.

# Parameters
- `s`: The setup.

# Returns
The outcomes, each with its weight, whose weights sum to the
[total](Setup.total).

# Examples
```lean
#guard ({ roll := some (.range 1 2), lowest := 1 } : Setup).outcomes ==
  [({ roll := some (.fixed [(1, 1)]), lowest := 1 }, 1),
    ({ roll := some (.fixed [(2, 1)]), lowest := 1 }, 1)]
```
-/
def outcomes (s : Setup) : List (Setup × Nat) :=
  match s.roll with
  | some ρ => ρ.outcomes.map fun e => ({ s with roll := some (.fixed e.1) }, e.2)
  | none => [(s, 1)]

/--
The record that an outcome of a setup holds, as the oracle reads it: the
results of its fixed roll, or none if it has no roll, with its drops.

# Parameters
- `s`: The outcome, a setup of fixed results or of no roll.

# Returns
The record. Any other setup holds no one record, and reads as empty.

# Examples
```lean
#guard ({ roll := some (.fixed [(1, 2), (3, 1)]), highest := 1 } : Setup).held
  == { results := [1, 1, 3], highest := 1 }
```
-/
def held (s : Setup) : Xdy.Record :=
  { results :=
      match s.roll with
      | some (.fixed rs) => Roll.expand rs
      | _ => []
    lowest := s.lowest
    highest := s.highest }

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
-/
theorem tsum_law_toOracle (s : Setup) (hs : s.WellFormed)
    (ψ : Xdy.Record → Nat) :
    ∑' r, s.law r * (ψ r.toOracle : ℝ≥0∞) =
      (OrderStatistics.integral s.outcomes (fun o => ψ o.held) : ℝ≥0∞) /
        (s.total : ℝ≥0∞) := by
  rcases s with ⟨_ | ρ, lowest, highest⟩
  · simp only [law, Option.map_none, Option.getD_none, PMF.pure_map,
      Outcomes.tsum_pure_mul, outcomes, total]
    simp [OrderStatistics.integral, held, Record.toOracle, Record.sorted]
  · have h := ρ.tsum_law_sorted (hs ρ rfl)
      fun ys => ψ { results := ys, lowest, highest }
    simp only [law, Option.map_some, Option.getD_some, tsum_map_mul]
    refine h.trans ?_
    simp [outcomes, total, held, OrderStatistics.integral, Function.comp_def]

/--
The law of the record that a setup describes, read as the oracle reads it,
sorted, gives each sorted record the weight of the outcomes that hold it, over
the total, as `Mixture::outcomes` weighs them.

# Parameters
- `s`: The setup.
- `v`: The sorted record.

# Hypotheses
- `hs`: The setup is well formed.
-/
theorem map_toOracle_law (s : Setup) (hs : s.WellFormed) (v : Xdy.Record) :
    (s.law.map Record.toOracle) v =
      (OrderStatistics.integral s.outcomes
          (fun o => if o.held = v then 1 else 0) : ℝ≥0∞) /
        (s.total : ℝ≥0∞) := by
  rw [Outcomes.map_apply_eq_tsum]
  exact s.tsum_law_toOracle hs fun r => if r = v then 1 else 0

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
  rcases s with ⟨_ | ρ, lowest, highest⟩
  · simp only [outcomes, List.mem_singleton] at ho
    subst ho
    simp [law, PMF.pure_map, held, Record.ofOracle]
  · simp only [outcomes, List.mem_map] at ho
    obtain ⟨e, he, rfl⟩ := ho
    have hw := Roll.wellFormed_of_mem_outcomes (hs ρ rfl) he
    refine ⟨?_, Roll.expand_pairwise hw.1⟩
    simp [law, Roll.law, PMF.pure_map, held, Record.ofOracle]

end Setup

/--
The law of the record that a mixture describes, read sorted, mixes the laws
of its setups, read sorted. With `Setup.map_toOracle_law`, the outcomes of
every setup, each scaled to the common denominator and weighed by its setup,
are the law of the mixture read sorted, as `Dist.hasLaw_mixture` cancels the
scaling.

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
Every record of a state read sorted is sorted.

# Parameters
- `s`: The state.
-/
theorem toOracle_sorted (s : State) :
    ∀ r ∈ s.toOracle.records, r.results.Pairwise (· ≤ ·) := by
  intro r hr
  simp only [State.toOracle, Array.mem_map] at hr
  obtain ⟨a, _, rfl⟩ := hr
  exact Record.sorted_pairwise a

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
  have h₁ := hasLaw_step s i
  have h₂ := hasLaw_step (State.ofOracle s.toOracle) i
  rw [State.toOracle_ofOracle _ (toOracle_sorted s)] at h₂
  exact PMF.ext fun t => (h₁.2.2 t).trans (h₂.2.2 t).symm

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
    Record.toOracle_ofOracle _ (Record.sorted_pairwise (s.record d)),
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
