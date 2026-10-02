import XdySpec.World

/-!
# Mixtures of setups

A rolling record of the forward pass in `xdy/src/distribution/propagation.rs`
holds not its law but a mixture of setups, `Mixture` in
`propagation/mixture.rs`: each setup a roll with its operands resolved to
values, and its drops, weighed by the branches of the random operands that
chose it. This module gives a setup its law, the law of the record that it
fills, and a mixture the mixture of its setups' laws, and proves that the
mixture's operations act on that law as the instructions do:

- A record never rolled is a mixture of one setup without a roll, whose law is
  the empty record's (`Setup.law_empty`), as `Mixture::default` makes it.
- Rolling mixes the roll's law over the law of its operands
  (`mixtureLaw_rolls`), as `Mixture::rolls` makes one setup of each, so the
  law that `law_rollRange_operands`, `law_rollStandardDice` and
  `law_rollCustomDice` give a roll is the law of the mixture that it fills.
  A custom die holds its distinct faces, each with its multiplicity, which
  have the law of the die (`weightedDie_eq_customDie`), as `distinct_faces`
  in `propagation/record.rs` counts them.
- A drop drops every setup by every count (`mixtureLaw_dropLowest`,
  `mixtureLaw_dropHighest`), as `Mixture::drop` does, which is lemma 1's lift
  of the drop over the record's law and the count's, as `law_dropLowest` and
  `law_dropHighest` give it. A setup can clamp its drops itself, since it
  fixes the number of results of every record that it fills
  (`Setup.mem_support_roll`).
- The sum of a mixture is the mixture of its setups' sums
  (`map_sum_mixtureLaw`), as `Mixture::sum` mixes them, and a setup without a
  roll sums to `0` (`Setup.map_sum_law_of_none`). The sums of ranges without
  drops accumulate in a difference array, `Ranges`, whose sweep weighs every
  outcome as adding each range's outcomes one by one would
  (`Span.sweep_eq_cover`).
- A setup's sum is what `Roll::sum` builds for it, branch by branch: fixed
  results sum to `keptSum`, which adds each distinct result's kept copies with
  one clamp, as `kept_sum` does (`Setup.map_sum_law_fixed`); a range that is
  empty or drops anything, and dice that are none, have no faces, or are all
  dropped, sum to a certain `0` (`Setup.map_sum_law_range_empty`,
  `Setup.map_sum_law_range_dropped`, `Setup.map_sum_law_standard_eq_zero`,
  `Setup.map_sum_law_custom_eq_zero`); and a range without drops sums
  uniformly over its values (`Setup.map_sum_law_range`). Dice without drops
  whose faces pass `folds_to_clamp` sum to their convolution power, clamped
  (`rollDice_saturating_sum`), and the rest by order statistics, as
  `XdySpec/OrderStatistics.lean` proves.

The Rust weighs its setups with natural numbers over their total; a mixture
here is a `PMF` of setups, which is those weights over their total.
-/

namespace Xdy.Spec

/-! ## Dice of weighted faces -/

/--
The law of a die of weighted faces: each face equally likely by position once
every face is repeated as many times as its weight, or a certain `0` if it
has no faces, as `customDie` rolls the faces repeated.

# Parameters
- `faces`: The faces, each with its weight.

# Returns
The law of the die.

# Examples
```lean
example : weightedDie [] = PMF.pure 0 := by simp [weightedDie, customDie]
```
-/
noncomputable def weightedDie (faces : List (Int × Nat)) : PMF Int :=
  customDie (faces.flatMap fun p => List.replicate p.2 p.1)

/--
A custom die's faces, weighed by multiplicity, have the law of the die, as a
custom roll of `propagation/record.rs` holds its die's `distinct_faces`.

# Parameters
- `ws`: The faces, each with its weight.
- `faces`: The faces of the die, in any order, possibly repeated.

# Hypotheses
- `h`: Repeating each face of `ws` as many times as its weight lists the die's
  faces, in some order.
-/
theorem weightedDie_eq_customDie {ws : List (Int × Nat)} {faces : List Int}
    (h : (ws.flatMap fun p => List.replicate p.2 p.1).Perm faces) :
    weightedDie ws = customDie faces := by
  unfold weightedDie customDie
  by_cases hf : faces = []
  · subst hf
    rw [List.perm_nil.mp h]
  · have hw : (ws.flatMap fun p => List.replicate p.2 p.1) ≠ [] :=
      fun he => hf (List.perm_nil.mp (he ▸ h.symm))
    simp only [hw, hf, ↓reduceDIte]
    congr 1
    exact Quot.sound h

/-! ## Rolls -/

/-- A roll that fills a rolling record, with its operands resolved to values.
Mirrors `Roll` in `propagation/record.rs`. -/
inductive Roll where
  /-- A roll of the range from `start` to `stop`. -/
  | range (start stop : Int)
  /-- A roll of `count` standard dice of `faces` faces. -/
  | standard (count faces : Int)
  /-- A roll of `count` custom dice, each of the given faces with their
  weights, as `distinct_faces` counts them. -/
  | custom (count : Int) (faces : List (Int × Nat))
  /-- The results of a roll, fixed by conditioning on them: each distinct
  result, in ascending order, with its number of copies. -/
  | fixed (results : List (Int × Nat))
  deriving DecidableEq, Repr

namespace Roll

/--
Repeat each result as many times as its copies.

# Parameters
- `results`: The results, each with its number of copies.

# Returns
The results, repeated, in order.

# Examples
```lean
#guard Roll.expand [(1, 2), (4, 1)] == [1, 1, 4]
```
-/
def expand (results : List (Int × Nat)) : List Int :=
  results.flatMap fun p => List.replicate p.2 p.1

/--
The law of the record that a roll fills.

# Parameters
- `ρ`: The roll.

# Returns
The law of the range or of the dice, as `step` rolls them, or a certain
record of the fixed results, in ascending order.
-/
noncomputable def law : Roll → PMF Record
  | .range start stop => rollRange start stop
  | .standard count faces => rollDice count (standardDie faces)
  | .custom count faces => rollDice count (weightedDie faces)
  | .fixed results => PMF.pure { results := expand results }

/--
The number of results in the record that a roll fills. Mirrors `results` in
`propagation/record.rs`.

# Parameters
- `ρ`: The roll.

# Returns
One for a range, even an empty one, the number of dice, which is `0` if it is
negative, or the number of fixed results.

# Examples
```lean
#guard (Roll.standard (-2) 6).results == 0
#guard (Roll.fixed [(1, 2), (4, 1)]).results == 3
```
-/
def results : Roll → Nat
  | .range .. => 1
  | .standard count _ | .custom count _ => count.toNat
  | .fixed results => (results.map (·.2)).sum

/--
Every record that a roll can fill holds its number of results, and drops
none of them.

# Parameters
- `ρ`: The roll.
- `r`: The record.

# Hypotheses
- `h`: The roll can fill `r`.
-/
theorem mem_support_law {ρ : Roll} {r : Record} (h : r ∈ ρ.law.support) :
    r.results.length = ρ.results ∧ r.lowest = 0 ∧ r.highest = 0 := by
  cases ρ with
  | range start stop =>
    simp only [law, rollRange] at h
    split at h
    · simp only [PMF.mem_support_pure_iff] at h
      subst h
      simp [results]
    · simp only [PMF.support_map, Set.mem_image] at h
      obtain ⟨x, _, rfl⟩ := h
      simp [results]
  | standard count faces =>
    simp only [law] at h
    obtain ⟨hl, hh, hlen, _⟩ := mem_support_rollDice h
    exact ⟨hlen, hl, hh⟩
  | custom count faces =>
    simp only [law] at h
    obtain ⟨hl, hh, hlen, _⟩ := mem_support_rollDice h
    exact ⟨hlen, hl, hh⟩
  | fixed rs =>
    simp only [law, PMF.mem_support_pure_iff] at h
    subst h
    simp [results, expand, List.length_flatMap]

end Roll

/-! ## Setups -/

/-- One setup of a rolling record: its roll, if any, and its drops. Mirrors
`Setup` in `propagation/mixture.rs`. -/
structure Setup where
  /-- The roll that filled the record, if any. -/
  roll : Option Roll := none
  /-- The number of lowest results dropped, at most the number of results. -/
  lowest : Nat := 0
  /-- The number of highest results dropped, at most the number of results. -/
  highest : Nat := 0
  deriving DecidableEq, Repr

namespace Setup

/--
The number of results in the record. Mirrors `results` in `mixture.rs`.

# Parameters
- `s`: The setup.

# Returns
The number of results of its roll, or `0` if it has none.
-/
def results (s : Setup) : Nat := (s.roll.map Roll.results).getD 0

/--
The law of the record that a setup describes.

# Parameters
- `s`: The setup.

# Returns
The law of its roll, or the empty record if it has none, with its drops.
-/
noncomputable def law (s : Setup) : PMF Record :=
  ((s.roll.map Roll.law).getD (PMF.pure {})).map
    fun r => { r with lowest := s.lowest, highest := s.highest }

/--
Drop the lowest results of a setup, in addition to any already dropped,
clamping to the number of results, as `Mixture::drop` does.

# Parameters
- `s`: The setup.
- `count`: The number of results to drop. A negative count drops nothing.

# Returns
The setup with its drop count updated.
-/
def dropLowest (s : Setup) (count : Int) : Setup :=
  { s with lowest := min (s.lowest + count.toNat) s.results }

/--
Drop the highest results of a setup, in addition to any already dropped,
clamping to the number of results, as `Mixture::drop` does.

# Parameters
- `s`: The setup.
- `count`: The number of results to drop. A negative count drops nothing.

# Returns
The setup with its drop count updated.
-/
def dropHighest (s : Setup) (count : Int) : Setup :=
  { s with highest := min (s.highest + count.toNat) s.results }

/--
Every record that a setup's roll can fill holds the setup's number of
results.

# Parameters
- `s`: The setup.
- `r`: The record.

# Hypotheses
- `h`: The setup's roll, or its absence, can fill `r`.
-/
theorem mem_support_roll {s : Setup} {r : Record}
    (h : r ∈ ((s.roll.map Roll.law).getD (PMF.pure {})).support) :
    r.results.length = s.results := by
  cases hs : s.roll with
  | none =>
    simp only [hs, Option.map_none, Option.getD_none,
      PMF.mem_support_pure_iff] at h
    subst h
    simp [results, hs]
  | some ρ =>
    simp only [hs, Option.map_some, Option.getD_some] at h
    simp [results, hs, (Roll.mem_support_law h).1]

/--
Dropping the lowest results of the records that a setup describes describes
the setup with its drop updated, since it clamps to the same number of
results.

# Parameters
- `s`: The setup.
- `count`: The number of results to drop.
-/
theorem law_dropLowest (s : Setup) (count : Int) :
    s.law.map (·.dropLowest count) = (s.dropLowest count).law := by
  simp only [law, PMF.map_comp]
  refine map_congr_support fun r hr => ?_
  have hlen := mem_support_roll hr
  simp only [Function.comp_apply, dropLowest, ← hlen]
  rfl

/--
Dropping the highest results of the records that a setup describes describes
the setup with its drop updated, since it clamps to the same number of
results.

# Parameters
- `s`: The setup.
- `count`: The number of results to drop.
-/
theorem law_dropHighest (s : Setup) (count : Int) :
    s.law.map (·.dropHighest count) = (s.dropHighest count).law := by
  simp only [law, PMF.map_comp]
  refine map_congr_support fun r hr => ?_
  have hlen := mem_support_roll hr
  simp only [Function.comp_apply, dropHighest, ← hlen]
  rfl

/--
A setup of a roll without drops describes the roll's law, as
`Mixture::rolls` makes it.

# Parameters
- `ρ`: The roll.
-/
theorem law_roll (ρ : Roll) : ({ roll := some ρ } : Setup).law = ρ.law := by
  simp only [law, Option.map_some, Option.getD_some]
  conv_rhs => rw [← PMF.map_id ρ.law]
  refine map_congr_support fun r hr => ?_
  obtain ⟨_, hl, hh⟩ := Roll.mem_support_law hr
  cases r
  simp_all

/-- The setup of a record never rolled describes the empty record, as
`Mixture::default` makes it. -/
theorem law_empty : ({} : Setup).law = PMF.pure {} := by
  simp [law, PMF.pure_map]

/--
A setup without a roll sums to `0`, as `setup_sum` in `mixture.rs` sums it.

# Parameters
- `s`: The setup.

# Hypotheses
- `h`: The setup has no roll.
-/
theorem map_sum_law_of_none {s : Setup} (h : s.roll = none) :
    s.law.map Record.sum = PMF.pure 0 := by
  simp [law, h, PMF.pure_map, Record.sum, Record.kept, Record.sorted]

end Setup

/-! ## Mixtures -/

/--
The law of the record that a mixture of setups describes.

# Parameters
- `μ`: The law of the setups, each the weight of its setup over the total, as
  `Mixture` in `mixture.rs` weighs them.

# Returns
The mixture of the setups' laws.
-/
noncomputable def mixtureLaw (μ : PMF Setup) : PMF Record := μ.bind Setup.law

/--
A mixture of one setup describes the setup's law.

# Parameters
- `s`: The setup.
-/
theorem mixtureLaw_pure (s : Setup) : mixtureLaw (PMF.pure s) = s.law :=
  PMF.pure_bind _ _

/--
A mixture of rolls, one for each outcome of the operands, describes the
rolls' laws mixed over the operands' law, as `Mixture::rolls` fills a record
with one setup for each roll that the operands choose, merging repeated
rolls.

# Parameters
- `π`: The law of the operands.
- `roll`: The roll at each outcome of the operands.
-/
theorem mixtureLaw_rolls {α : Type} (π : PMF α) (roll : α → Roll) :
    mixtureLaw (π.map fun a => { roll := some (roll a) }) =
      π.bind fun a => (roll a).law := by
  simp only [mixtureLaw, PMF.bind_map, Function.comp_def, Setup.law_roll]

/--
Dropping the lowest results of every setup by every count describes lemma 1's
lift of the drop over the mixture's law and the count's, as `Mixture::drop`
drops a record whose count is independent of it.

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the count.
-/
theorem mixtureLaw_dropLowest (μ : PMF Setup) (κ : PMF Int) :
    mixtureLaw (lift Setup.dropLowest μ κ) =
      lift Record.dropLowest (mixtureLaw μ) κ := by
  simp only [mixtureLaw, lift, PMF.bind_bind, PMF.bind_map]
  refine congrArg μ.bind (funext fun s => ?_)
  calc κ.bind (Setup.law ∘ s.dropLowest)
      _ = κ.bind fun c => s.law.map (·.dropLowest c) :=
        congrArg κ.bind (funext fun c => (Setup.law_dropLowest s c).symm)
      _ = _ := by
        simp only [← PMF.bind_pure_comp]
        exact PMF.bind_comm _ _ _

/--
Dropping the highest results of every setup by every count describes lemma
1's lift of the drop over the mixture's law and the count's, as
`Mixture::drop` drops a record whose count is independent of it.

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the count.
-/
theorem mixtureLaw_dropHighest (μ : PMF Setup) (κ : PMF Int) :
    mixtureLaw (lift Setup.dropHighest μ κ) =
      lift Record.dropHighest (mixtureLaw μ) κ := by
  simp only [mixtureLaw, lift, PMF.bind_bind, PMF.bind_map]
  refine congrArg μ.bind (funext fun s => ?_)
  calc κ.bind (Setup.law ∘ s.dropHighest)
      _ = κ.bind fun c => s.law.map (·.dropHighest c) :=
        congrArg κ.bind (funext fun c => (Setup.law_dropHighest s c).symm)
      _ = _ := by
        simp only [← PMF.bind_pure_comp]
        exact PMF.bind_comm _ _ _

/--
The sum of a mixture is the mixture of its setups' sums, by the law of total
probability, as `Mixture::sum` mixes them.

# Parameters
- `μ`: The law of the setups.
-/
theorem map_sum_mixtureLaw (μ : PMF Setup) :
    (mixtureLaw μ).map Record.sum = μ.bind fun s => s.law.map Record.sum :=
  PMF.map_bind _ _ _

/-! ## Sums of setups -/

/--
The saturating sum of the kept results of fixed results, from a position on,
as `kept_sum` in `propagation/record.rs` folds them: each distinct result adds
the copies of it that lie within the kept positions with one clamp.

# Parameters
- `lowest`: The first kept position.
- `stop`: The position just past the last kept one.
- `j`: The position of the first of the results.
- `sum`: The sum so far.
- `results`: The distinct results, each with its number of copies.

# Returns
The sum.
-/
def keptSumFrom (lowest stop : Int) : Int → Int → List (Int × Nat) → Int
  | _, sum, [] => sum
  | j, sum, (v, m) :: rest =>
    let placed := Max.max 0 (Min.min stop (j + m) - Max.max lowest j)
    keptSumFrom lowest stop (j + m) (clamp (sum + placed * v)) rest

/--
The saturating sum of the kept results of fixed results. Mirrors `kept_sum`
in `propagation/record.rs`.

# Parameters
- `results`: The distinct results, in ascending order, each with its number of
  copies.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.

# Returns
The sum.

# Examples
```lean
#guard keptSum [(1, 2), (4, 1)] 1 0 == 5
#guard keptSum [(1, 1), (i32Max, 2)] 0 0 == i32Max
```
-/
def keptSum (results : List (Int × Nat)) (lowest highest : Nat) : Int :=
  keptSumFrom lowest ((results.map (·.2)).sum - highest) 0 0 results

/-- Keeping a window of copies of one value followed by the rest keeps some
of the copies, then a window of the rest. -/
private theorem keep_replicate_append (m : Nat) (v : Int) (rest : List Int)
    (lo k : Nat) :
    ((List.replicate m v ++ rest).drop lo).take k =
      List.replicate (min (m - lo) k) v ++
        (rest.drop (lo - m)).take (k - (m - lo)) := by
  rw [List.drop_append, List.drop_replicate, List.length_replicate,
    List.take_append, List.take_replicate, List.length_replicate,
    Nat.min_comm]

/-- Adding copies of one value saturates at most once, so the copies add with
one clamp. -/
private theorem foldl_add_replicate {s : Int} (lo : i32Min ≤ s)
    (hi : s ≤ i32Max) (p : Nat) (v : Int) :
    (List.replicate p v).foldl add s = clamp (s + p * v) := by
  rw [foldl_add_from _ s (by
    by_cases h : 0 ≤ v
    · exact .inl fun x hx => (List.eq_of_mem_replicate hx) ▸ h
    · exact .inr fun x hx => (List.eq_of_mem_replicate hx) ▸ by omega) lo hi,
    List.sum_replicate, nsmul_eq_mul]

/-- A clamp lies within the `i32` range. -/
private theorem clamp_bounds (x : Int) :
    i32Min ≤ clamp x ∧ clamp x ≤ i32Max := by
  simp only [clamp, i32Min, i32Max]; omega

/-- Folding the kept window of fixed results, from a position on, is
`keptSumFrom`, which counts each result's copies in the window. -/
private theorem foldl_keep (lowest stop : Int) :
    ∀ (rs : List (Int × Nat)) (j s : Int) (lo k : Nat),
      i32Min ≤ s → s ≤ i32Max → 0 ≤ j →
      (lo : Int) = Max.max 0 (lowest - j) →
      (k : Int) = Max.max 0 (stop - Max.max lowest j) →
      (((Roll.expand rs).drop lo).take k).foldl add s =
        keptSumFrom lowest stop j s rs
  | [], j, s, lo, k, _, _, _, _, _ => by simp [Roll.expand, keptSumFrom]
  | (v, m) :: rest, j, s, lo, k, hlo, hhi, hj, hl, hk => by
    have hexp : Roll.expand ((v, m) :: rest) =
        List.replicate m v ++ Roll.expand rest := rfl
    rw [hexp, keep_replicate_append, List.foldl_append,
      foldl_add_replicate hlo hhi]
    simp only [keptSumFrom]
    have hp : ((min (m - lo) k : Nat) : Int) =
        Max.max 0 (Min.min stop (j + m) - Max.max lowest j) := by omega
    rw [hp]
    exact foldl_keep lowest stop rest (j + m) _ (lo - m) (k - (m - lo))
      (clamp_bounds _).1 (clamp_bounds _).2 (by omega) (by omega) (by omega)

/--
The sum of a record of fixed results, sorted, is their `keptSum`, which adds
each distinct result's kept copies with one clamp rather than one at a time.

# Parameters
- `rs`: The distinct results, each with its number of copies.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.

# Hypotheses
- `sorted`: The results, repeated, ascend, as the fixed results of a split
  do.
-/
theorem Record.sum_fixed (rs : List (Int × Nat)) (lowest highest : Nat)
    (sorted : (Roll.expand rs).Pairwise (· ≤ ·)) :
    ({ results := Roll.expand rs, lowest, highest } : Record).sum =
      keptSum rs lowest highest := by
  have hsort : (Roll.expand rs).mergeSort (fun a b => decide (a ≤ b)) =
      Roll.expand rs := List.mergeSort_of_pairwise (by simpa using sorted)
  have hlen :
      ((Roll.expand rs).length : Int) = ((rs.map (·.2)).sum : Nat) := by
    simp [Roll.expand, List.length_flatMap]
  simp only [Record.sum, Record.kept, Record.sorted, hsort]
  exact foldl_keep _ _ rs 0 0 _ _ (by decide) (by decide) le_rfl (by omega)
    (by push_cast at hlen ⊢; omega)

/--
A record that drops every result sums to `0`.

# Parameters
- `r`: The record.

# Hypotheses
- `h`: The drops cover every result.
-/
theorem Record.sum_eq_zero_of_dropped (r : Record)
    (h : r.results.length ≤ r.lowest + r.highest) : r.sum = 0 := by
  simp only [Record.sum, Record.kept]
  rw [show r.results.length - r.lowest - r.highest = 0 by omega,
    List.take_zero]
  rfl

/-- Adding zeros keeps `0`. -/
private theorem foldl_add_zeros (xs : List Int) (h : ∀ x ∈ xs, x = 0) :
    xs.foldl add 0 = 0 := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.foldl_cons, h x List.mem_cons_self,
      show add 0 0 = 0 by decide]
    exact ih fun y hy => h y (List.mem_cons_of_mem _ hy)

/--
A record whose results are all `0` sums to `0`, whatever it drops.

# Parameters
- `r`: The record.

# Hypotheses
- `h`: Every result is `0`.
-/
theorem Record.sum_eq_zero_of_zeros (r : Record)
    (h : ∀ x ∈ r.results, x = 0) : r.sum = 0 := by
  refine foldl_add_zeros _ fun x hx => h x ?_
  have := List.mem_of_mem_drop (List.mem_of_mem_take hx)
  exact (List.mergeSort_perm _ _).mem_iff.mp this

/--
A setup of a roll describes the roll's law with the setup's drops.

# Parameters
- `ρ`: The roll.
- `lowest`, `highest`: The drops.
-/
theorem Setup.law_some (ρ : Roll) (lowest highest : Nat) :
    ({ roll := some ρ, lowest, highest } : Setup).law =
      ρ.law.map fun r => { r with lowest, highest } := rfl

/--
Fixed results sum to their `keptSum`, as `Roll::sum` sums them.

# Parameters
- `rs`: The distinct results, each with its number of copies.
- `lowest`, `highest`: The drops.

# Hypotheses
- `sorted`: The results, repeated, ascend.
-/
theorem Setup.map_sum_law_fixed (rs : List (Int × Nat)) (lowest highest : Nat)
    (sorted : (Roll.expand rs).Pairwise (· ≤ ·)) :
    ({ roll := some (.fixed rs), lowest, highest } : Setup).law.map
        Record.sum = PMF.pure (keptSum rs lowest highest) := by
  simp only [Setup.law_some, Roll.law, PMF.pure_map]
  exact congrArg PMF.pure (Record.sum_fixed rs lowest highest sorted)

/-- A law of records each of which sums to `0`, transformed, sums to a certain
`0`. -/
private theorem map_sum_const {p : PMF Record} (f : Record → Record)
    (h : ∀ r ∈ p.support, (f r).sum = 0) :
    (p.map f).map Record.sum = PMF.pure 0 := by
  rw [PMF.map_comp, map_congr_support (f := Record.sum ∘ f) (g := fun _ => 0) h]
  exact PMF.map_const _ _

/--
A range that drops anything sums to a certain `0`, as `Roll::sum` sums it,
since it has one result.

# Parameters
- `start`, `stop`: The endpoints.
- `lowest`, `highest`: The drops.

# Hypotheses
- `h`: The range drops its lowest or its highest result.
-/
theorem Setup.map_sum_law_range_dropped (start stop : Int)
    (lowest highest : Nat) (h : 0 < lowest ∨ 0 < highest) :
    ({ roll := some (.range start stop), lowest, highest } : Setup).law.map
        Record.sum = PMF.pure 0 := by
  rw [Setup.law_some]
  refine map_sum_const _ fun r hr => Record.sum_eq_zero_of_dropped _ ?_
  have := (Roll.mem_support_law hr).1
  simp only [Roll.results] at this
  show r.results.length ≤ lowest + highest
  omega

/--
An empty range sums to a certain `0`, as `Roll::sum` sums it, since its one
result is `0`.

# Parameters
- `lowest`, `highest`: The drops.

# Hypotheses
- `h`: The range is empty.
-/
theorem Setup.map_sum_law_range_empty {start stop : Int} (lowest highest : Nat)
    (h : stop < start) :
    ({ roll := some (.range start stop), lowest, highest } : Setup).law.map
        Record.sum = PMF.pure 0 := by
  rw [Setup.law_some]
  refine map_sum_const _ fun r hr => Record.sum_eq_zero_of_zeros _ ?_
  simp only [Roll.law, rollRange, h, ↓reduceDIte,
    PMF.mem_support_pure_iff] at hr
  subst hr
  simp

/--
A range within the `i32` range, without drops, sums uniformly over its
values, as `Roll::sum` and `Ranges` weigh it, since its one result sums to
itself.

# Hypotheses
- `hle`: The range is not empty.
- `hstart`, `hstop`: Its endpoints lie within the `i32` range, as they do in
  Rust.
-/
theorem Setup.map_sum_law_range {start stop : Int} (hle : start ≤ stop)
    (hstart : i32Min ≤ start) (hstop : stop ≤ i32Max) :
    ({ roll := some (.range start stop) } : Setup).law.map Record.sum =
      PMF.ofMultiset (Finset.Icc start stop).val (by simp; omega) := by
  rw [Setup.law_roll]
  simp only [Roll.law, rollRange, show ¬ stop < start by omega, ↓reduceDIte,
    PMF.map_comp]
  conv_rhs => rw [← PMF.map_id (PMF.ofMultiset _ _)]
  refine map_congr_support fun x hx => ?_
  have hx : start ≤ x ∧ x ≤ stop := by simpa using hx
  show Record.sum { results := [x] } = x
  have : Record.sum { results := [x] } = add 0 x := by
    simp [Record.sum, Record.kept, Record.sorted]
  rw [this]
  simp only [add, clamp, i32Min, i32Max] at hstart hstop ⊢
  omega

/--
Dice sum to a certain `0` if they drop every die, or if every face is `0`.

# Parameters
- `count`: The number of dice.
- `die`: The law of one die.
- `lowest`, `highest`: The drops.

# Hypotheses
- `h`: The drops cover every die, or the die is a certain `0`.
-/
theorem map_sum_rollDice_eq_zero (count : Int) (die : PMF Int)
    (lowest highest : Nat)
    (h : count.toNat ≤ lowest + highest ∨ die = PMF.pure 0) :
    ((rollDice count die).map fun r => { r with lowest, highest }).map
      Record.sum = PMF.pure 0 := by
  refine map_sum_const _ fun r hr => ?_
  obtain ⟨_, _, hlen, hmem⟩ := mem_support_rollDice hr
  rcases h with h | h
  · exact Record.sum_eq_zero_of_dropped _ (by
      show r.results.length ≤ lowest + highest
      omega)
  · subst h
    exact Record.sum_eq_zero_of_zeros _ fun x hx => by
      simpa using hmem x hx

/--
Standard dice sum to a certain `0` if there are none, if they have no faces,
or if every die is dropped, as `Roll::sum` sums them.

# Parameters
- `lowest`, `highest`: The drops.

# Hypotheses
- `h`: The count or the faces are not positive, or the drops cover every die.
-/
theorem Setup.map_sum_law_standard_eq_zero {count faces : Int}
    (lowest highest : Nat)
    (h : count ≤ 0 ∨ faces ≤ 0 ∨ count ≤ lowest + highest) :
    ({ roll := some (.standard count faces), lowest, highest } :
        Setup).law.map Record.sum = PMF.pure 0 := by
  rw [Setup.law_some]
  exact map_sum_rollDice_eq_zero _ _ lowest highest (by
    rcases h with h | h | h
    · exact .inl (by omega)
    · exact .inr (by simp [standardDie, h])
    · exact .inl (by omega))

/--
Custom dice sum to a certain `0` if there are none, if they have no faces,
or if every die is dropped, as `Roll::sum` sums them.

# Parameters
- `lowest`, `highest`: The drops.

# Hypotheses
- `h`: The count is not positive, the die has no faces, or the drops cover
  every die.
-/
theorem Setup.map_sum_law_custom_eq_zero {count : Int}
    {faces : List (Int × Nat)} (lowest highest : Nat)
    (h : count ≤ 0 ∨ faces = [] ∨ count ≤ lowest + highest) :
    ({ roll := some (.custom count faces), lowest, highest } :
        Setup).law.map Record.sum = PMF.pure 0 := by
  rw [Setup.law_some]
  exact map_sum_rollDice_eq_zero _ _ lowest highest (by
    rcases h with h | h | h
    · exact .inl (by omega)
    · exact .inr (by simp [h, weightedDie, customDie])
    · exact .inl (by omega))

/-! ## Ranges -/

/-- A range of outcomes, each of the same weight, as `Ranges::add` in
`mixture.rs` weighs the uniform sum of a range without drops. -/
structure Span where
  /-- The least outcome. -/
  start : Int
  /-- The greatest outcome. -/
  stop : Int
  /-- The weight of each outcome. -/
  weight : Nat
  deriving DecidableEq, Repr

namespace Span

/--
The weight that the spans that start at an outcome add to the level there.
Mirrors `rises` in `Ranges`.

# Parameters
- `spans`: The spans.
- `k`: The outcome.

# Returns
The total weight of the spans that start at `k`.
-/
def rise (spans : List Span) (k : Int) : Nat :=
  (spans.map fun s => if s.start = k then s.weight else 0).sum

/--
The weight that the spans that end just before an outcome remove from the
level there. Mirrors `falls` in `Ranges`.

# Parameters
- `spans`: The spans.
- `k`: The outcome.

# Returns
The total weight of the spans that end at `k - 1`.
-/
def fall (spans : List Span) (k : Int) : Nat :=
  (spans.map fun s => if s.stop + 1 = k then s.weight else 0).sum

/--
The weight of an outcome: that of every span that covers it, as adding each
span's outcomes one by one would weigh it.

# Parameters
- `spans`: The spans.
- `v`: The outcome.

# Returns
The total weight of the spans that cover `v`.
-/
def cover (spans : List Span) (v : Int) : Nat :=
  (spans.map fun s => if s.start ≤ v ∧ v ≤ s.stop then s.weight else 0).sum

/--
The weight that `Ranges::sweep` in `mixture.rs` gives an outcome: it walks the
breaks in ascending order, adds each break's rise to a running level and
subtracts its fall, and gives the level to every outcome from that break to
the next.

# Parameters
- `spans`: The spans.
- `level`: The level before the first break.
- `breaks`: The breaks, in ascending order.
- `v`: The outcome.

# Returns
The level of the window between consecutive breaks that holds `v`, or `0` if
none does.

# Examples
```lean
#guard ([-1, 0, 1, 2, 3, 4, 5, 6] : List Int).map
    (Span.sweep [⟨1, 3, 2⟩, ⟨2, 5, 1⟩] 0 [1, 2, 4, 6])
  == [0, 0, 2, 3, 3, 1, 1, 0]
```
-/
def sweep (spans : List Span) : Int → List Int → Int → Int
  | level, here :: next :: rest, v =>
    let level := level + rise spans here - fall spans here
    if here ≤ v ∧ v < next then level else sweep spans level (next :: rest) v
  | _, _, _ => 0

/-- The weight of the spans that start before `x`, less that of those that
end before `x - 1`: the level just before the break at `x`. -/
private def below (spans : List Span) (x : Int) : Int :=
  (spans.map fun s => (if s.start < x then (s.weight : Int) else 0) -
    if s.stop + 1 < x then (s.weight : Int) else 0).sum

/-- The weight of the spans that start by `x`, less that of those that end
before `x`: the level just after the break at `x`. -/
private def upTo (spans : List Span) (x : Int) : Int :=
  (spans.map fun s => (if s.start ≤ x then (s.weight : Int) else 0) -
    if s.stop + 1 ≤ x then (s.weight : Int) else 0).sum

/-- Sums of functions that agree on a list agree. -/
private theorem sum_map_congr {l : List Span} {f g : Span → Int}
    (h : ∀ s ∈ l, f s = g s) : (l.map f).sum = (l.map g).sum :=
  congrArg List.sum (List.map_congr_left h)

/-- The level just after a break is the weight of the spans that cover it,
since every span ends after it starts. -/
private theorem upTo_eq_cover {spans : List Span}
    (valid : ∀ s ∈ spans, s.start ≤ s.stop) (x : Int) :
    upTo spans x = cover spans x := by
  unfold upTo cover
  rw [Nat.cast_list_sum, List.map_map]
  refine sum_map_congr fun s hs => ?_
  have := valid s hs
  simp only [Function.comp_apply]
  split_ifs <;> omega

/-- A break's rise and fall take the level from just before it to just after
it. -/
private theorem below_add (spans : List Span) (x : Int) :
    below spans x + rise spans x - fall spans x = upTo spans x := by
  unfold below upTo rise fall
  rw [Nat.cast_list_sum, Nat.cast_list_sum, List.map_map, List.map_map,
    ← List.sum_map_add, sub_eq_iff_eq_add, ← List.sum_map_add]
  refine sum_map_congr fun s _ => ?_
  simp only [Function.comp_apply]
  split_ifs <;> push_cast <;> omega

/-- The keys of the spans: where each starts, and just past where each
ends. -/
private def Key (spans : List Span) (k : Int) : Prop :=
  ∃ s ∈ spans, s.start = k ∨ s.stop + 1 = k

/-- No key lies strictly between consecutive breaks, once every key from the
first on is a break. -/
private theorem gap {spans : List Span} {here next : Int} {rest : List Int}
    (sorted : (here :: next :: rest).Pairwise (· < ·))
    (keys : ∀ k, Key spans k → here ≤ k → k ∈ here :: next :: rest)
    {k : Int} (hk : Key spans k) : k ≤ here ∨ next ≤ k := by
  by_cases h : k ≤ here
  · exact .inl h
  · right
    have hm := keys k hk (by omega)
    simp only [List.mem_cons] at hm
    rcases hm with rfl | rfl | hm
    · omega
    · exact le_refl _
    · have := List.rel_of_pairwise_cons (List.Pairwise.of_cons sorted) hm
      omega

/-- The sweep from a break, with the level just before it, gives every
outcome from that break on the weight of the spans that cover it, and every
outcome before it nothing. -/
private theorem sweep_eq {spans : List Span}
    (valid : ∀ s ∈ spans, s.start ≤ s.stop) :
    ∀ (rest : List Int) (here level : Int),
      (here :: rest).Pairwise (· < ·) →
      (∀ k, Key spans k → here ≤ k → k ∈ here :: rest) →
      level = below spans here →
      ∀ v, sweep spans level (here :: rest) v =
        if here ≤ v then (cover spans v : Int) else 0
  | [], here, level, _, keys, _, v => by
    simp only [sweep]
    split_ifs with hv
    · -- Every span ends by `here`.
      refine (congrArg Nat.cast (List.sum_eq_zero fun x hx => ?_)).symm
      obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hx
      have := keys _ ⟨s, hs, .inr rfl⟩
      have := valid s hs
      have : s.stop + 1 ≤ here := by
        by_cases h : here ≤ s.stop + 1
        · have := List.mem_singleton.mp (keys _ ⟨s, hs, .inr rfl⟩ h)
          omega
        · omega
      split_ifs <;> omega
    · rfl
  | next :: rest, here, level, sorted, keys, hlevel, v => by
    have hstep : level + rise spans here - fall spans here =
        cover spans here := by
      rw [hlevel, below_add, upTo_eq_cover valid]
    have hgap := fun k hk => gap sorted keys (k := k) hk
    -- Nothing starts or ends between `here` and `next`.
    have hbelow : below spans next = cover spans here := by
      rw [← upTo_eq_cover valid]
      refine sum_map_congr fun s hs => ?_
      rcases hgap _ ⟨s, hs, .inl rfl⟩ with h1 | h1 <;>
        rcases hgap _ ⟨s, hs, .inr rfl⟩ with h2 | h2 <;>
        have := List.rel_of_pairwise_cons sorted (List.mem_cons_self) <;>
        split_ifs <;> omega
    have hcover : ∀ v, here ≤ v → v < next →
        cover spans v = cover spans here := by
      intro v h1 h2
      refine Nat.cast_injective (R := Int) ?_
      simp only [← upTo_eq_cover valid]
      refine sum_map_congr fun s hs => ?_
      rcases hgap _ ⟨s, hs, .inl rfl⟩ with h3 | h3 <;>
        rcases hgap _ ⟨s, hs, .inr rfl⟩ with h4 | h4 <;>
        split_ifs <;> omega
    have hnext := List.rel_of_pairwise_cons sorted List.mem_cons_self
    have ih := sweep_eq valid rest next _ (List.Pairwise.of_cons sorted)
      (fun k hk hle => by
        have := keys k hk (by omega)
        simp only [List.mem_cons] at this ⊢
        rcases this with rfl | h
        · omega
        · exact h)
      (hstep.trans hbelow.symm) v
    simp only [sweep]
    split_ifs with hw hv hv
    · rw [hstep, hcover v hw.1 hw.2]
    · omega
    · rw [ih]
      split_ifs <;> first | rfl | omega
    · rw [ih]
      split_ifs <;> first | rfl | omega

/--
The sweep of `Ranges::sweep` in `mixture.rs` gives every outcome the weight of
the spans that cover it, as adding each span's outcomes one by one would. The
level of every window is that weight at the window's first outcome, so it
never falls below zero, as `sweep` expects of it.

# Parameters
- `spans`: The spans.
- `breaks`: The breaks.
- `v`: The outcome.

# Hypotheses
- `valid`: Every span ends where or after it starts, as `Mixture::sum` adds
  only ranges that are not empty.
- `sorted`: The breaks ascend strictly, as `sweep` collects them into a set.
- `starts`: Every span starts at a break, the key of its rise.
- `stops`: Every span ends just before a break, the key of its fall.
-/
theorem sweep_eq_cover {spans : List Span} {breaks : List Int}
    (valid : ∀ s ∈ spans, s.start ≤ s.stop)
    (sorted : breaks.Pairwise (· < ·))
    (starts : ∀ s ∈ spans, s.start ∈ breaks)
    (stops : ∀ s ∈ spans, s.stop + 1 ∈ breaks) (v : Int) :
    sweep spans 0 breaks v = cover spans v := by
  cases breaks with
  | nil =>
    cases spans with
    | nil => rfl
    | cons s _ => exact absurd (starts s List.mem_cons_self) (by simp)
  | cons here rest =>
    have keys : ∀ k, Key spans k → k ∈ here :: rest := by
      rintro k ⟨s, hs, rfl | rfl⟩
      · exact starts s hs
      · exact stops s hs
    have least : ∀ k, Key spans k → here ≤ k := fun k hk => by
      rcases List.mem_cons.mp (keys k hk) with rfl | h
      · exact le_refl _
      · exact le_of_lt (List.rel_of_pairwise_cons sorted h)
    rw [sweep_eq valid rest here 0 sorted (fun k hk _ => keys k hk) ?_ v]
    · split_ifs with h
      · rfl
      · -- Nothing starts by `v`.
        refine (congrArg Nat.cast (List.sum_eq_zero fun x hx => ?_)).symm
        obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hx
        have := least _ ⟨s, hs, .inl rfl⟩
        split_ifs <;> omega
    · refine (List.sum_eq_zero fun x hx => ?_).symm
      obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hx
      have := least _ ⟨s, hs, .inl rfl⟩
      have := least _ ⟨s, hs, .inr rfl⟩
      split_ifs <;> omega

end Span

end Xdy.Spec
