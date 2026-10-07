import XdySpec.Outcomes

/-!
# Rerolls in the forward pass

A reroll of the forward pass in `xdy/src/distribution/propagation.rs` rerolls
every setup of its record's mixture by every joint outcome of its operands
(`Mixture::reroll` in `propagation/mixture.rs`), and a setup by what the
reroll does to the die from which it draws (`redraw` in
`propagation/record.rs`): nothing, if no face matches, the limit is not
positive, or a die of one face is limited; crossing out every die still kept,
if an unlimited reroll matches every face; and otherwise a redraw, which
weighs the final results of a matched die by the branches that reach them,
`g`, and pads every die still kept by `G` branches. This module mirrors those
operations and proves that they act on the law of the record as the
specification's reroll does:

- A redraw's weights are the specification's laws of a replacement: limited
  to `n` times, the `limitedLaw`, whose closed form is the Rust's
  (`unmatchedPaths_mul`, `unmatchedPaths_self`), is `limitedReplacement`
  (`weightedDie_limitedLaw`), counted position by position through the
  draws; unlimited, the `unlimitedLaw` is the die's conditional law, given
  that it does not match (`weightedDie_unlimitedLaw`). Each weighs the
  padding in all (`length_expand_limitedLaw`, `length_expand_unlimitedLaw`).
- Rerolling dice that are alike, without drops, rewrites the law of each die
  (`Roll.rewrite`, `weightedDie_rewriteFaces`), since a custom die whose every
  face chooses a die of as many faces is the die of all of their faces
  (`customDie_bind`).
- The specification rerolls the kept results in ascending order, in place;
  the Rust rewrites each die in roll order. Only the multiset of the results
  that a reroll leaves reaches anything later, and it does not depend on the
  order (`bagDraws_perm`), so independent dice, each rerolled, are the dice
  of the rerolled law (`bind_rollDice_bagDraws`).
- A setup's reroll (`Setup.reroll`) describes the specification's reroll of
  its records as every later instruction views them (`Record.view`, the kept
  results and the die), whether it does nothing, crosses out, rewrites, or
  pools (`Setup.law_reroll`); and so a mixture's reroll describes the reroll
  of the mixture's law, independent of the operands' (`mixtureLaw_reroll`,
  `law_reroll_resolved`). The view, not the record, since the Rust leaves a
  setup that the reroll cannot change with its drops, and crosses out by
  dropping, where the specification's record keeps only its kept results.
- Dice after a drop, or fixed results, are not alike, so the reroll pools the
  multisets of the dice still kept (`Roll.rerollPool`, `Roll::pool`). For
  each outcome of the roll, the specification rerolls the kept results that
  the criterion matches and leaves the survivors (`survivors_spec`), so the
  multiset that it leaves is the survivors joined with each outcome of the
  replacements (`bagDraws_kept`); the Rust weighs each by `G^(K − k)`, so that
  every outcome weighs `Gᴷ` in all, and the pool is the outcomes followed by
  their replacements (`weighted_flatMap`, `Setup.rerolls_of_pools`).
- The proofs ask that every setup be well formed, that every result be a face
  of its die, and that every die have a face (`Setup.Faithful`,
  `Setup.Faced`), as every roll (`Setup.faithful_roll`), drop and keep
  (`Setup.Faithful.of_eq`), and reroll (`Setup.reroll_invariants`) leaves
  them.

```mermaid
flowchart TD
    S[setup] -->|no dice still kept| N[unchanged]
    S --> E{Effect.resolve}
    E -->|nothing| N
    E -->|crossOut| X[every die still kept dropped]
    E -->|redraw G g| A{alike, no drops?}
    A -->|yes| R[Roll.rewrite: each die's law]
    A -->|no| P[Roll.rerollPool: multisets kept]
```
-/

namespace Xdy.Spec

open Xdy (Die Criterion)
open scoped ENNReal

/-! ## Weighted faces -/

/--
Each distinct value of a list, in ascending order, with the number of times
that it appears, as `distinct_faces` and `Die::faces` in
`propagation/record.rs` count the faces of a die.

# Parameters
- `xs`: The values, in any order, possibly repeated.

# Returns
The distinct values, each with its positive count.

# Examples
```lean
#guard tally [3, 1, 3] == [(1, 1), (3, 2)]
```
-/
def tally (xs : List Int) : List (Int × Nat) :=
  (xs.dedup.mergeSort (· ≤ ·)).map fun x => (x, xs.count x)

/--
The copies of a value among weighted values, repeated by their weights, are
the weights of its entries.

# Parameters
- `l`: The values, each with its weight.
- `v`: The value.
-/
theorem count_expand (l : List (Int × Nat)) (v : Int) :
    (Roll.expand l).count v = ((l.filter (·.1 = v)).map (·.2)).sum := by
  induction l with
  | nil => rfl
  | cons e l ih =>
    simp only [Roll.expand, List.flatMap_cons, List.count_append,
      List.count_replicate] at ih ⊢
    rw [ih]
    by_cases h : e.1 = v <;> simp [h]

/-- Weighted values listed once each count their weights. -/
private theorem count_expand_keys {K : List Int} (hK : K.Nodup)
    (w : Int → Nat) (v : Int) :
    (Roll.expand (K.map fun x => (x, w x))).count v =
      if v ∈ K then w v else 0 := by
  rw [count_expand]
  induction K with
  | nil => simp
  | cons x K ih =>
    rw [List.nodup_cons] at hK
    by_cases h : x = v
    · subst h
      have : (K.map fun x => (x, w x)).filter (·.1 = x) = [] := by
        simp only [List.filter_eq_nil_iff, List.mem_map, decide_eq_true_eq]
        rintro _ ⟨y, hy, rfl⟩ rfl
        exact hK.1 hy
      simp [this]
    · simp [h, ih hK.2, Ne.symm h]

/--
A tally counts every value as often as the list does.

# Parameters
- `xs`: The values.
- `v`: The value counted.
-/
theorem count_expand_tally (xs : List Int) (v : Int) :
    (Roll.expand (tally xs)).count v = xs.count v := by
  rw [tally, count_expand_keys ((List.nodup_dedup xs).perm
    (List.mergeSort_perm _ _).symm)]
  simp only [List.mem_mergeSort, List.mem_dedup]
  split_ifs with h
  · rfl
  · exact (List.count_eq_zero.mpr h).symm

/--
A tally, repeated by its counts, is the list in some order.

# Parameters
- `xs`: The values.
-/
theorem expand_tally_perm (xs : List Int) : (Roll.expand (tally xs)).Perm xs :=
  List.perm_iff_count.mpr fun v => count_expand_tally xs v

/--
Weighted values, each weight scaled, repeat each value as many more times.

# Parameters
- `l`: The values, each with its weight.
- `k`: The scale.
- `v`: The value counted.
-/
theorem count_expand_scale (l : List (Int × Nat)) (k : Nat) (v : Int) :
    (Roll.expand (l.map fun e => (e.1, e.2 * k))).count v =
      (Roll.expand l).count v * k := by
  rw [count_expand, count_expand]
  induction l with
  | nil => simp
  | cons e l ih => by_cases h : e.1 = v <;> simp_all [Nat.add_mul]

/--
Weighted values, filtered by their values, keep the copies of the values that
pass.

# Parameters
- `l`: The values, each with its weight.
- `p`: The filter.
- `v`: The value counted.
-/
theorem count_expand_filter (l : List (Int × Nat)) (p : Int → Bool)
    (v : Int) :
    (Roll.expand (l.filter (p ·.1))).count v =
      if p v then (Roll.expand l).count v else 0 := by
  rw [count_expand, count_expand]
  induction l with
  | nil => simp
  | cons e l ih =>
    by_cases h : e.1 = v
    · subst h
      by_cases hp : p e.1 <;> simp_all
    · by_cases hp : p e.1 <;> simp_all

/--
The weight of the values that a criterion matches is the number of copies
that it matches once they are repeated by their weights.

# Parameters
- `l`: The values, each with its weight.
- `c`: The criterion.
-/
theorem countP_expand (l : List (Int × Nat)) (c : Criterion) :
    (Roll.expand l).countP c.matches =
      ((l.filter (c.matches ·.1)).map (·.2)).sum := by
  induction l with
  | nil => rfl
  | cons e l ih =>
    simp only [Roll.expand, List.flatMap_cons, List.countP_append,
      List.countP_replicate] at ih ⊢
    rw [ih]
    by_cases h : c.matches e.1 <;> simp [h]

/-! ## Uniform laws -/

/--
A custom die rolls alike whatever the order of its faces.

# Parameters
- `xs`, `ys`: The faces.

# Hypotheses
- `h`: The faces are the same, in some order.
-/
theorem customDie_perm {xs ys : List Int} (h : xs.Perm ys) :
    customDie xs = customDie ys := by
  unfold customDie
  by_cases hx : xs = []
  · subst hx
    rw [List.perm_nil.mp h.symm]
  · have hy : ys ≠ [] := fun he => hx (List.perm_nil.mp (he ▸ h))
    simp only [hx, hy, ↓reduceDIte]
    congr 1
    exact Quot.sound h

/-- Repeating weighted values repeats each list of them in turn. -/
private theorem expand_append (l l' : List (Int × Nat)) :
    Roll.expand (l ++ l') = Roll.expand l ++ Roll.expand l' :=
  List.flatMap_append

/--
A custom die of copies of one face always shows it.

# Parameters
- `k`: The number of copies.
- `x`: The face.

# Hypotheses
- `hk`: There is at least one copy.
-/
theorem customDie_replicate {k : Nat} (hk : k ≠ 0) (x : Int) :
    customDie (List.replicate k x) = PMF.pure x := by
  have hne : List.replicate k x ≠ [] := by simpa using hk
  unfold customDie
  simp only [hne, ↓reduceDIte]
  ext y
  rw [PMF.ofMultiset_apply, PMF.pure_apply]
  by_cases h : y = x
  · subst h
    simp only [Multiset.coe_count, List.count_replicate_self, Multiset.coe_card,
      List.length_replicate, ↓reduceIte]
    exact ENNReal.div_self (by simpa using hk) (by simp)
  · simp [h]

/--
A custom die of one face always shows it.

# Parameters
- `x`: The face.
-/
theorem customDie_singleton (x : Int) : customDie [x] = PMF.pure x :=
  customDie_replicate one_ne_zero x

/--
A die of weighted faces is the custom die of its faces repeated.

# Parameters
- `l`: The faces, each with its weight.
-/
theorem weightedDie_eq_customDie_expand (l : List (Int × Nat)) :
    weightedDie l = customDie (Roll.expand l) := rfl

/--
A sum against a custom die averages over its faces.

# Parameters
- `xs`: The faces.
- `f`: The function summed.

# Hypotheses
- `h`: The die has faces.
-/
theorem tsum_customDie {xs : List Int} (h : xs ≠ []) (f : Int → ℝ≥0∞) :
    ∑' x, customDie xs x * f x =
      (xs.map f).sum / (xs.length : ℝ≥0∞) := by
  have hx : Roll.expand (xs.map fun x => (x, 1)) = xs := by
    induction xs with
    | nil => rfl
    | cons x xs ih =>
      by_cases hxs : xs = []
      · subst hxs
        rfl
      · simp [Roll.expand] at ih ⊢
        exact ih hxs
  have hlen : ((xs.map fun x => ((x, 1) : Int × Nat)).map (·.2)).sum =
      xs.length := by
    simp [Function.comp_def]
  rw [show customDie xs = weighted 0 (xs.map fun x => (x, 1)) by
      rw [← weightedDie_eq_weighted, weightedDie_eq_customDie_expand, hx],
    Outcomes.tsum_weighted _ _ (by rw [hlen]; simpa using h), hlen]
  simp [Function.comp_def]

/--
A custom die gives each face its share of the positions.

# Parameters
- `xs`: The faces.
- `v`: The face.

# Hypotheses
- `h`: The die has faces.
-/
theorem customDie_apply {xs : List Int} (h : xs ≠ []) (v : Int) :
    customDie xs v = (xs.count v : ℝ≥0∞) / (xs.length : ℝ≥0∞) := by
  unfold customDie
  simp only [h, ↓reduceDIte, PMF.ofMultiset_apply, Multiset.coe_count,
    Multiset.coe_card]
  congr
  exact Subsingleton.elim _ _

/--
A custom die whose every face chooses a custom die of as many faces as every
other is the custom die of all of their faces: every position of the first,
and then of the second, is equally likely.

# Parameters
- `xs`: The faces of the first die.
- `ys`: The faces of the die that each face chooses.
- `K`: The number of faces of each chosen die.

# Hypotheses
- `hxs`: The first die has faces.
- `hK`: Every chosen die has `K` faces.
- `hK0`: `K` is positive.
-/
theorem customDie_bind {xs : List Int} {ys : Int → List Int} {K : Nat}
    (hxs : xs ≠ []) (hK : ∀ x ∈ xs, (ys x).length = K) (hK0 : K ≠ 0) :
    (customDie xs).bind (fun x => customDie (ys x)) =
      customDie (xs.flatMap ys) := by
  have hlen : (xs.flatMap ys).length = xs.length * K := by
    rw [List.length_flatMap, List.map_congr_left hK]
    simp
  have hne : xs.flatMap ys ≠ [] := by
    intro h
    have := congrArg List.length h
    rw [hlen] at this
    simp only [List.length_nil, Nat.mul_eq_zero] at this
    rcases this with h | h
    · exact hxs (List.eq_nil_of_length_eq_zero h)
    · exact hK0 h
  ext v
  rw [PMF.bind_apply, tsum_customDie hxs, customDie_apply hne,
    List.count_flatMap, hlen]
  have hys : ∀ x ∈ xs, customDie (ys x) v =
      ((ys x).count v : ℝ≥0∞) / (K : ℝ≥0∞) := fun x hx => by
    rw [customDie_apply (fun h => hK0 (by rw [← hK x hx, h]; rfl)), hK x hx]
  rw [List.map_congr_left hys]
  simp only [ENNReal.div_eq_inv_mul, ← List.sum_map_mul_left,
    Function.comp_apply, Nat.cast_list_sum, List.map_map, Nat.cast_mul]
  rw [ENNReal.mul_inv (by simp) (by simp)]
  refine congrArg List.sum (List.map_congr_left fun x _ => ?_)
  ring

/--
Weighted values from a tally, each weight scaled by a function of its value,
repeat each value as many times as the list holds it, scaled.

# Parameters
- `xs`: The values.
- `φ`: The scale of each value.
- `v`: The value counted.
-/
theorem count_expand_tally_map (xs : List Int) (φ : Int → Nat) (v : Int) :
    (Roll.expand ((tally xs).map fun p => (p.1, p.2 * φ p.1))).count v =
      xs.count v * φ v := by
  rw [tally, List.map_map]
  rw [show ((fun p : Int × Nat => (p.1, p.2 * φ p.1)) ∘
      fun x => (x, xs.count x)) = fun x => (x, xs.count x * φ x) from rfl,
    count_expand_keys ((List.nodup_dedup xs).perm
      (List.mergeSort_perm _ _).symm)]
  simp only [List.mem_mergeSort, List.mem_dedup]
  split_ifs with h
  · rfl
  · rw [List.count_eq_zero.mpr h, Nat.zero_mul]

/-! ## The effect of a reroll -/

/--
The number of branches along which a die, limited to `j + 1` rerolls and
matched by a criterion, first shows a result that it does not match, of a
face of weight one: `Σ_{k<j} mᵏ · Fʲ⁻ᵏ`, for a die of `F` branches of which
the criterion matches `m`. Mirrors `unmatched` in `redraw` in
`propagation/record.rs`, which computes it in closed form
(`unmatchedPaths_mul`, `unmatchedPaths_self`).

# Parameters
- `F`: The branches of the die.
- `m`: The branches that the criterion matches.
- `j`: The number of rerolls after the first.

# Returns
The number of branches.

# Examples
```lean
#guard unmatchedPaths 6 1 2 == 36 + 6
```
-/
def unmatchedPaths (F m j : Nat) : Nat :=
  ((List.range j).map fun k => m ^ k * F ^ (j - k)).sum

/--
One more reroll multiplies every earlier path by the die, and adds the path
that matched every time before.

# Parameters
- `F`, `m`: The branches of the die, and those that the criterion matches.
- `j`: The number of rerolls after the first.
-/
theorem unmatchedPaths_succ (F m j : Nat) :
    unmatchedPaths F m (j + 1) = F * unmatchedPaths F m j + m ^ j * F := by
  simp only [unmatchedPaths, List.range_succ, List.map_append, List.sum_append,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    show j + 1 - j = 1 by omega, pow_one, ← List.sum_map_mul_left]
  congr 1
  refine congrArg List.sum (List.map_congr_left fun k hk => ?_)
  rw [List.mem_range] at hk
  rw [show j + 1 - k = (j - k) + 1 by omega, pow_succ]
  ring

/-- The paths of a die, padded, telescope: `S · (F − m) + F · mʲ = Fʲ⁺¹`,
with `F = m + t`. -/
private theorem unmatchedPaths_telescope (m t : Nat) :
    ∀ j, unmatchedPaths (m + t) m j * t + (m + t) * m ^ j = (m + t) ^ (j + 1)
  | 0 => by simp [unmatchedPaths]
  | j + 1 => by
    rw [unmatchedPaths_succ, pow_succ (m + t) (j + 1),
      ← unmatchedPaths_telescope m t j]
    ring

/--
The closed form of `redraw` in `propagation/record.rs`, if some face does not
match: `F · (Fʲ − mʲ) / (F − m)`, here without the division.

# Parameters
- `F`, `m`: The branches of the die, and those that the criterion matches.
- `j`: The number of rerolls after the first.

# Hypotheses
- `h`: The criterion matches no more branches than the die has.
-/
theorem unmatchedPaths_mul {F m : Nat} (h : m ≤ F) (j : Nat) :
    unmatchedPaths F m j * (F - m) = (F ^ j - m ^ j) * F := by
  obtain ⟨t, rfl⟩ := Nat.exists_eq_add_of_le h
  have := unmatchedPaths_telescope m t j
  rw [Nat.add_sub_cancel_left, Nat.sub_mul, mul_comm (m ^ j), ← pow_succ,
    ← this]
  rw [mul_comm (m + t) (m ^ j)]
  omega

/--
The closed form of `redraw` in `propagation/record.rs`, if every face
matches: `j · Fʲ`.

# Parameters
- `F`: The branches of the die, every one of which the criterion matches.
- `j`: The number of rerolls after the first.
-/
theorem unmatchedPaths_self (F j : Nat) :
    unmatchedPaths F F j = j * F ^ j := by
  induction j with
  | zero => simp [unmatchedPaths]
  | succ j ih =>
    rw [unmatchedPaths_succ, ih, pow_succ]
    ring

/--
The final results of a die that a reroll limited to `limit` times matched,
each weighed by the branches that reach it, `g` in `redraw` in
`propagation/record.rs`: each face of the die, weighed by
`mʲ` if it matches and by `unmatchedPaths F m j + mʲ` if not, times its
weight, where `j = limit - 1`.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled.

# Returns
The distinct faces, in ascending order, each with its weight.
-/
def limitedLaw (d : Die) (c : Criterion) (limit : Nat) : List (Int × Nat) :=
  let F := d.faces.length
  let m := d.faces.countP c.matches
  let j := limit - 1
  (tally d.faces).map fun p =>
    (p.1, p.2 * if c.matches p.1 then m ^ j else unmatchedPaths F m j + m ^ j)

/--
The final results of a die that an unlimited reroll matched, each weighed by
its branches, `g` in `redraw` in `propagation/record.rs`: the faces of the die
that the criterion does not match.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Returns
The distinct faces, in ascending order, each with its weight.
-/
def unlimitedLaw (d : Die) (c : Criterion) : List (Int × Nat) :=
  (tally d.faces).filter (!c.matches ·.1)

/-- What a reroll does to the dice still kept of a roll, given the die from
which it draws their replacements. Mirrors `Effect` in
`propagation/record.rs`. -/
inductive Effect where
  /-- Nothing, since no face matches, the limit is not positive, or the die
  has one face and the limit is finite, so that every die shows what it
  showed. -/
  | nothing
  /-- The reroll crosses out every die still kept, rolling nothing, since it
  has no limit and matches every face. -/
  | crossOut
  /-- The reroll may change the dice still kept: each die still kept gains
  `padding` branches, `G`, and one that the criterion matched finally shows
  each result of `law` along its weight, `g`. -/
  | redraw (padding : Nat) (law : List (Int × Nat))
  deriving DecidableEq, Repr

/--
Resolve a reroll against the die from which it draws: decide whether it may
change the dice still kept, and if so, how a die that it matches finally
falls. Mirrors `redraw` in `propagation/record.rs`.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Returns
The effect: for a limited reroll, the padding `Fⁿ` and the `limitedLaw`; for
an unlimited one, the padding `F − m` and the `unlimitedLaw`.

# Examples
```lean
#guard Effect.resolve (.standard 6) (.within 1 1) (some 2) ==
  .redraw 36 [(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
#guard Effect.resolve (.standard 6) (.within 1 6) none == .crossOut
#guard Effect.resolve (.standard 6) (.within 7 7) none == .nothing
```
-/
def Effect.resolve (d : Die) (c : Criterion) (limit : Option Nat) : Effect :=
  let F := d.faces.length
  let m := d.faces.countP c.matches
  if m = 0 then .nothing
  else
    match limit with
    | some limit =>
      if limit = 0 ∨ F = 1 then .nothing
      else .redraw (F ^ (limit - 1) * F) (limitedLaw d c limit)
    | none => if m = F then .crossOut else .redraw (F - m) (unlimitedLaw d c)

/-! ## The laws of the effect -/

/-- Replacing each position that matches by the positions of `ys`, and
repeating each other `G` times, counts each value by the positions that
matched and by its own. -/
private theorem count_flatMap_redraw (c : Criterion) (ys : List Int) (G : Nat)
    (v : Int) (xs : List Int) :
    (xs.flatMap fun x =>
        if c.matches x then ys else List.replicate G x).count v =
      xs.countP c.matches * ys.count v +
        if c.matches v then 0 else G * xs.count v := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    rw [List.flatMap_cons, List.count_append, ih, List.countP_cons,
      List.count_cons]
    by_cases hxv : x = v
    · subst hxv
      by_cases hx : c.matches x <;> simp [hx] <;> ring
    · by_cases hx : c.matches x <;> by_cases hv : c.matches v <;>
        simp [hx, hv, hxv, List.count_replicate] <;> ring

/-- The positions of `k + 1` draws of a limited reroll: the faces, and then,
for each further draw, each position that matched replaced by the faces, and
each other repeated as many times. -/
private def draws (c : Criterion) (faces : List Int) : Nat → List Int
  | 0 => faces
  | k + 1 => (draws c faces k).flatMap fun x =>
    if c.matches x then faces else List.replicate faces.length x

/-- Every draw multiplies the positions by the faces. -/
private theorem length_draws (c : Criterion) (faces : List Int) :
    ∀ k, (draws c faces k).length = faces.length ^ (k + 1)
  | 0 => by simp [draws]
  | k + 1 => by
    have hlen : ∀ x, (if c.matches x then faces
        else List.replicate faces.length x).length = faces.length :=
      fun x => by split <;> simp
    rw [draws, List.length_flatMap, pow_succ, ← length_draws c faces k]
    simp only [hlen, List.map_const', List.sum_replicate, smul_eq_mul]

/-- The positions that still match multiply by the matching faces. -/
private theorem countP_draws (c : Criterion) (faces : List Int) :
    ∀ k, (draws c faces k).countP c.matches =
      faces.countP c.matches ^ (k + 1)
  | 0 => by simp [draws]
  | k + 1 => by
    rw [draws, List.countP_flatMap, pow_succ, ← countP_draws c faces k]
    induction draws c faces k with
    | nil => simp
    | cons x xs ih =>
      by_cases h : c.matches x
      · simp_all [Nat.add_mul]
        ring
      · simp_all [List.countP_replicate]

/-- The positions that show each value, as `redraw` weighs them. -/
private theorem count_draws (c : Criterion) (faces : List Int) (v : Int) :
    ∀ k, (draws c faces k).count v = faces.count v *
      if c.matches v then faces.countP c.matches ^ k
      else unmatchedPaths faces.length (faces.countP c.matches) k +
        faces.countP c.matches ^ k
  | 0 => by simp [draws, unmatchedPaths]
  | k + 1 => by
    rw [draws, count_flatMap_redraw, countP_draws, count_draws c faces v k,
      unmatchedPaths_succ]
    by_cases hv : c.matches v
    · simp only [hv, ↓reduceIte, Nat.add_zero]
      ring
    · simp only [hv, Bool.false_eq_true, ↓reduceIte]
      ring

/-- The positions of the draws are the law of the redraws, as
`limitedReplacement` repeats them. -/
private theorem customDie_draws (c : Criterion) {faces : List Int}
    (h : faces ≠ []) :
    ∀ k, customDie (draws c faces k) =
      k.repeat (redraw c (customDie faces)) (customDie faces)
  | 0 => rfl
  | k + 1 => by
    have hF : faces.length ≠ 0 := by simpa using h
    rw [Nat.repeat, ← customDie_draws c h k, redraw, draws,
      ← customDie_bind (K := faces.length)
        (fun he => by
          have := length_draws c faces k
          rw [he] at this
          simp only [List.length_nil] at this
          exact absurd this.symm (pow_ne_zero _ hF))
        (fun x _ => by split <;> simp) hF]
    refine congrArg _ (funext fun x => ?_)
    split
    · rfl
    · exact customDie_replicate hF x

/-- The faces of the `limitedLaw` are the positions of the draws. -/
private theorem expand_limitedLaw_perm (d : Die) (c : Criterion) (limit : Nat) :
    (Roll.expand (limitedLaw d c limit)).Perm (draws c d.faces (limit - 1)) :=
  List.perm_iff_count.mpr fun v => by
    rw [count_draws]
    unfold limitedLaw
    exact count_expand_tally_map d.faces (fun x =>
      if c.matches x then d.faces.countP c.matches ^ (limit - 1)
      else unmatchedPaths d.faces.length (d.faces.countP c.matches) (limit - 1)
        + d.faces.countP c.matches ^ (limit - 1)) v

/-- Filtering a list keeps every copy of a value that passes, and none of
one that does not. -/
private theorem count_filter_ite (l : List Int) (p : Int → Bool) (v : Int) :
    (l.filter p).count v = if p v then l.count v else 0 := by
  split_ifs with h
  · exact List.count_filter h
  · exact List.count_eq_zero.mpr fun hm => h (List.mem_filter.mp hm).2

/-- The faces of the `unlimitedLaw` are those that the criterion does not
match. -/
private theorem expand_unlimitedLaw_perm (d : Die) (c : Criterion) :
    (Roll.expand (unlimitedLaw d c)).Perm (d.faces.filter (!c.matches ·)) :=
  List.perm_iff_count.mpr fun v => by
    rw [unlimitedLaw, count_expand_filter (p := fun x => !c.matches x),
      count_expand_tally, count_filter_ite]

/--
A die that a reroll limited to `limit` times matched finally falls as the
`limitedLaw` weighs its faces, which is the specification's
`limitedReplacement`: a draw from the die, then up to `limit - 1` redraws,
each while the result matches. Mirrors the limited law of `redraw` in
`propagation/record.rs`.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled.

# Hypotheses
- `h`: The die has faces.
-/
theorem weightedDie_limitedLaw {d : Die} (c : Criterion) (limit : Nat)
    (h : d.faces ≠ []) :
    weightedDie (limitedLaw d c limit) =
      limitedReplacement c (customDie d.faces) limit := by
  rw [limitedReplacement, ← customDie_draws c h,
    weightedDie_eq_customDie_expand]
  exact customDie_perm (expand_limitedLaw_perm d c limit)

/--
The final results of a die that a reroll limited to `limit` times matched
weigh, in all, the padding `Fⁿ`, as `redraw` in `propagation/record.rs` pads
each die still kept.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled.
-/
theorem length_expand_limitedLaw (d : Die) (c : Criterion) (limit : Nat) :
    (Roll.expand (limitedLaw d c limit)).length =
      d.faces.length ^ (limit - 1) * d.faces.length := by
  rw [← pow_succ, (expand_limitedLaw_perm d c limit).length_eq, length_draws]

/--
A die that an unlimited reroll matched finally falls as the `unlimitedLaw`
weighs its faces, which is the specification's conditional law of the die,
given that it does not match. Mirrors the unlimited law of `redraw` in
`propagation/record.rs`.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Hypotheses
- `h`: The criterion does not match every face.
-/
theorem weightedDie_unlimitedLaw {d : Die} {c : Criterion}
    (h : ¬d.facesWithin c = true) :
    weightedDie (unlimitedLaw d c) =
      PMF.ofMultiset _ (unmatched_ne_zero h) := by
  have hp := expand_unlimitedLaw_perm d c
  have hne : d.faces.filter (!c.matches ·) ≠ [] := by
    simpa using unmatched_ne_zero h
  rw [weightedDie_eq_customDie_expand, customDie_perm hp]
  unfold customDie
  simp only [hne, ↓reduceDIte]

/--
The final results of a die that an unlimited reroll matched weigh, in all,
the padding `F − m`, as `redraw` in `propagation/record.rs` pads each die
still kept.

# Parameters
- `d`: The die.
- `c`: The criterion.
-/
theorem length_expand_unlimitedLaw (d : Die) (c : Criterion) :
    (Roll.expand (unlimitedLaw d c)).length =
      d.faces.length - d.faces.countP c.matches := by
  have hp := expand_unlimitedLaw_perm d c
  rw [hp.length_eq, ← List.countP_eq_length_filter,
    List.length_eq_countP_add_countP c.matches (l := d.faces)]
  simp

/-! ## Rewriting the dice of a roll -/

namespace Roll

/--
Whether a roll's dice are alike: a range, which is one die, or standard or
custom dice, as `Setup::reroll` in `propagation/mixture.rs` asks before it
rewrites them.

# Parameters
- `ρ`: The roll.

# Returns
`true` for a range or dice, `false` for fixed results or a pool.
-/
def alike : Roll → Bool
  | .range .. | .standard .. | .custom .. => true
  | .fixed _ | .pool _ => false

/--
The dice of a roll whose dice are alike, as `Roll::rewrite` in
`propagation/record.rs` reads them: their number, and the distinct faces of
each, with their weights.

# Parameters
- `ρ`: The roll.

# Returns
One die of the range's faces; the standard dice of the standard die's faces;
the custom dice of their faces; or no dice for fixed results or a pool.

# Examples
```lean
#guard (Roll.standard 2 3).dice == (2, [(1, 1), (2, 1), (3, 1)])
#guard (Roll.range 4 3).dice == (1, [(0, 1)])
```
-/
def dice : Roll → Int × List (Int × Nat)
  | .range start stop => (1, tally (Die.range start stop).faces)
  | .standard count faces => (count, tally (Die.standard faces).faces)
  | .custom count faces => (count, faces)
  | .fixed _ | .pool _ => (0, [])

/--
Whether every die of a roll has a face, as every roll of `propagation/record.rs`
does: custom dice, if there are any, have faces. A range or standard die
without faces shows `0`, which is a face of its die.

# Parameters
- `ρ`: The roll.

# Returns
Whether every die has a face.
-/
def Faced : Roll → Prop
  | .custom count faces => count ≤ 0 ∨ expand faces ≠ []
  | _ => True

end Roll

/--
The faces of dice after a reroll rewrites them, each weighed by the branches
that reach it, `f′` in `Roll::rewrite` in `propagation/record.rs`: each face
that the criterion does not match keeps its weight, padded, and the final
results of a die that it matched join them, weighed by the weight of the
faces that it matches.

# Parameters
- `faces`: The faces of each die, with their weights, `f`.
- `c`: The criterion.
- `padding`: The branches that the reroll adds to each die, `G`.
- `law`: The final results of a die that the criterion matched, `g`.

# Returns
The distinct faces, in ascending order, each with its weight
`f(v) · [v ∉ R] · G + f(R) · g(v)`.

# Examples
```lean
#guard rewriteFaces [(1, 1), (2, 1)] (.within 1 1) 2 [(1, 1), (2, 1)] ==
  [(1, 1), (2, 3)]
```
-/
def rewriteFaces (faces : List (Int × Nat)) (c : Criterion) (padding : Nat)
    (law : List (Int × Nat)) : List (Int × Nat) :=
  let matched := ((faces.filter (c.matches ·.1)).map (·.2)).sum
  tally (Roll.expand
    ((faces.filter (!c.matches ·.1)).map (fun p => (p.1, p.2 * padding)) ++
      if matched = 0 then [] else law.map fun p => (p.1, p.2 * matched)))

/--
Reroll the dice of a roll whose dice are alike, none of which has been
dropped, as the effect of the reroll directs. Mirrors `Roll::rewrite` in
`propagation/record.rs`.

# Parameters
- `ρ`: The roll.
- `c`: The criterion.
- `padding`: The branches that the reroll adds to each die.
- `law`: The final results of a die that the criterion matched.

# Returns
The rerolled dice, as custom dice of the same number, or one for a range.
-/
def Roll.rewrite (ρ : Roll) (c : Criterion) (padding : Nat)
    (law : List (Int × Nat)) : Roll :=
  .custom ρ.dice.1 (rewriteFaces ρ.dice.2 c padding law)

/-- The faces of a rewrite are those of the die, each matched one replaced by
the final results of a die that the criterion matched, and each other
repeated as many times as the padding. -/
private theorem expand_rewriteFaces_perm (faces : List (Int × Nat))
    (c : Criterion) (padding : Nat) (law : List (Int × Nat)) :
    (Roll.expand (rewriteFaces faces c padding law)).Perm
      ((Roll.expand faces).flatMap fun x =>
        if c.matches x then Roll.expand law else List.replicate padding x) := by
  refine List.perm_iff_count.mpr fun v => ?_
  have hB : ∀ M : Nat, (Roll.expand (if M = 0 then []
      else law.map fun p => (p.1, p.2 * M))).count v =
      (Roll.expand law).count v * M := fun M => by
    split_ifs with h
    · simp [h, Roll.expand]
    · exact count_expand_scale law M v
  rw [rewriteFaces, count_expand_tally, count_flatMap_redraw, expand_append,
    List.count_append, count_expand_scale,
    count_expand_filter (p := fun x => !c.matches x), countP_expand, hB]
  cases hv : c.matches v <;> simp <;> ring

/--
A die that a reroll rewrites falls as the die, rerolled: each face that the
criterion matches replaced by a draw from the final results of a die that it
matched, and every other face kept.

# Parameters
- `faces`: The faces of the die, with their weights.
- `c`: The criterion.
- `padding`: The branches that the reroll adds to each die.
- `law`: The final results of a die that the criterion matched.

# Hypotheses
- `hf`: The die has faces.
- `hlaw`: The final results weigh the padding in all.
- `hG`: The padding is positive.
-/
theorem weightedDie_rewriteFaces {faces : List (Int × Nat)} (c : Criterion)
    {padding : Nat} {law : List (Int × Nat)} (hf : Roll.expand faces ≠ [])
    (hlaw : (Roll.expand law).length = padding) (hG : padding ≠ 0) :
    weightedDie (rewriteFaces faces c padding law) =
      (weightedDie faces).bind fun x =>
        if c.matches x then weightedDie law else PMF.pure x := by
  have hstep : (fun x => if c.matches x then weightedDie law else PMF.pure x) =
      fun x => customDie (if c.matches x then Roll.expand law
        else List.replicate padding x) := funext fun x => by
    split
    · rfl
    · exact (customDie_replicate hG x).symm
  rw [hstep, weightedDie_eq_customDie_expand faces,
    customDie_bind (K := padding) hf (fun x _ => by split <;> simp [hlaw]) hG,
    weightedDie_eq_customDie_expand]
  exact customDie_perm (expand_rewriteFaces_perm faces c padding law)

/-- A range of values is the interval of integers between its ends. -/
private theorem customDie_range {start stop : Int} (h : ¬stop < start) :
    customDie (Die.range start stop).faces =
      PMF.ofMultiset (Finset.Icc start stop).val (by simp; omega) := by
  have hne : (Die.range start stop).faces ≠ [] := by
    simp only [Die.faces, h, ↓reduceIte, ne_eq, List.map_eq_nil_iff,
      List.range_eq_nil]
    omega
  unfold customDie
  simp only [hne, ↓reduceDIte]
  congr 1
  rw [icc_val]
  simp [Die.faces, h]

/-- A die of the tally of some faces is the custom die of the faces. -/
private theorem weightedDie_tally (xs : List Int) :
    weightedDie (tally xs) = customDie xs :=
  customDie_perm (expand_tally_perm xs)

/--
The results of a roll whose dice are alike are those of its dice: the
records that it fills hold the results of as many independent draws of the
die as it has dice, as `Roll::rewrite` reads them.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `h`: The roll's dice are alike.
-/
theorem Roll.map_results_law {ρ : Roll} (h : ρ.alike) :
    ρ.law.map (·.results) =
      (rollDice ρ.dice.1 (weightedDie ρ.dice.2)).map (·.results) := by
  cases ρ with
  | range start stop =>
    simp only [law, dice, weightedDie_tally]
    have h1 : rollDice 1 (customDie (Die.range start stop).faces) =
        (customDie (Die.range start stop).faces).map fun x =>
          ({ results := [x] } : Record) := by
      simp [rollDice, Nat.repeat, PMF.pure_bind]
      rfl
    rw [h1, PMF.map_comp]
    unfold rollRange
    split
    · rename_i hlt
      rw [show (Die.range start stop).faces = [0] by simp [Die.faces, hlt],
        customDie_singleton, PMF.pure_map, PMF.pure_map]
      rfl
    · rename_i hle
      rw [customDie_range hle, PMF.map_comp]
      rfl
  | standard count faces =>
    simp only [law, dice, weightedDie_tally, PMF.map_comp]
    by_cases hf : faces ≤ 0
    · have : (Die.standard faces).faces = [0] := by
        simp [Die.standard, Die.faces, show faces < 1 by omega]
      rw [this, customDie_singleton]
      simp [standardDie, hf, Function.comp_def]
    · rw [Die.standard, customDie_range (by omega)]
      simp [standardDie, hf, Function.comp_def]
  | custom count faces =>
    simp only [law, dice, PMF.map_comp]
    rfl
  | fixed => simp [alike] at h
  | pool => simp [alike] at h

/-! ## Rerolling a bag of results -/

/--
The law of the results that a reroll leaves, as a multiset, from independent
results: each `x` replaced by a draw of `step x`.

# Parameters
- `step`: The law of the result that each result leaves.
- `xs`: The results.

# Returns
The law of the multiset of the results left.
-/
noncomputable def bagDraws (step : Int → PMF Int) :
    List Int → PMF (Multiset Int)
  | [] => PMF.pure 0
  | x :: xs => lift (· ::ₘ ·) (step x) (bagDraws step xs)

/--
The results that a reroll leaves, as a multiset, do not depend on the order
in which it rerolls them.

# Parameters
- `step`: The law of the result that each result leaves.

# Hypotheses
- `h`: The results are the same, in some order.
-/
theorem bagDraws_perm (step : Int → PMF Int) {xs ys : List Int}
    (h : xs.Perm ys) : bagDraws step xs = bagDraws step ys := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [bagDraws, ih]
  | swap x y l =>
    simp only [bagDraws, lift, PMF.map_bind, PMF.map_comp]
    rw [PMF.bind_comm]
    refine congrArg _ (funext fun a => congrArg _ (funext fun b =>
      congrArg (PMF.map · _) (funext fun M => ?_)))
    exact Multiset.cons_swap _ _ _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- One more result after the rest is one more result before them. -/
private theorem bagDraws_concat (step : Int → PMF Int) (xs : List Int)
    (x : Int) :
    bagDraws step (xs ++ [x]) =
      lift (· ::ₘ ·) (step x) (bagDraws step xs) :=
  bagDraws_perm step (List.perm_append_singleton x xs)

/-- The multiset of a record's results. -/
private def bag (r : Record) : Multiset Int := r.results

/-- Rerolling in place, from any law of records before, adds the rerolled
results to the records' bags. -/
private theorem foldl_rerollKept_bag (rerolled : Int → Bool)
    (replacement : PMF Int) :
    ∀ (xs : List Int) (p : PMF Record),
      (xs.foldl (fun p x =>
          if rerolled x then p.bind fun r => replacement.map r.push
          else p.map (·.push x)) p).map bag =
        lift (· + ·) (p.map bag) (bagDraws
          (fun x => if rerolled x then replacement else PMF.pure x) xs)
  | [], p => by
    simp only [List.foldl_nil, bagDraws, lift, PMF.pure_map, add_zero]
    exact (PMF.bind_pure _).symm
  | x :: xs, p => by
    rw [List.foldl_cons, foldl_rerollKept_bag rerolled replacement xs]
    have hstep : (if rerolled x then p.bind fun r => replacement.map r.push
        else p.map (·.push x)) =
        p.bind fun r => (if rerolled x then replacement else PMF.pure x).map
          r.push := by
      split
      · rfl
      · simp only [PMF.pure_map]
        exact (PMF.bind_pure_comp _ _).symm
    rw [hstep]
    simp only [bagDraws, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      PMF.bind_bind, Function.comp_def]
    refine congrArg _ (funext fun r => congrArg _ (funext fun y =>
      congrArg (PMF.map · _) (funext fun M => ?_)))
    simp only [bag, Record.push, ← Multiset.coe_add, Multiset.coe_singleton]
    rw [add_assoc, Multiset.singleton_add]

/--
The results that a reroll of kept results leaves, as a multiset, are
independent draws, one for each result: the replacement if the reroll
rerolls it, and the result itself if not.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.
-/
theorem map_bag_rerollKept (die : Die) (kept : List Int)
    (rerolled : Int → Bool) (replacement : PMF Int) :
    (rerollKept die kept rerolled replacement).map
        (fun r => (r.results : Multiset Int)) =
      bagDraws (fun x => if rerolled x then replacement else PMF.pure x)
        kept := by
  have := foldl_rerollKept_bag rerolled replacement kept (PMF.pure { die })
  simp only [PMF.pure_map, lift, PMF.pure_bind] at this
  rw [show (fun r : Record => (r.results : Multiset Int)) = bag from rfl,
    rerollKept, this]
  simp only [bag, Multiset.coe_nil, zero_add]
  exact PMF.map_id _

/--
Rerolling independent results, one at a time, leaves independent results:
the dice of a law, each rerolled, are the dice of the law rerolled.

# Parameters
- `n`: The number of dice.
- `die`: The law of each die.
- `step`: The law of the result that each result leaves.
-/
theorem bind_rollDice_bagDraws (n : Int) (die : PMF Int)
    (step : Int → PMF Int) :
    (rollDice n die).bind (fun r => bagDraws step r.results) =
      (rollDice n (die.bind step)).map fun r => (r.results : Multiset Int) := by
  rw [OrderStatistics.rollDice_toNat n die,
    OrderStatistics.rollDice_toNat n (die.bind step)]
  induction n.toNat with
  | zero => simp [rollDice, Nat.repeat, PMF.pure_map, bagDraws]
  | succ J ih =>
    have hpush : ∀ (a : Record) (y : Int),
        ((a.push y).results : Multiset Int) =
          y ::ₘ (a.results : Multiset Int) :=
      fun a y => by
        simp only [Record.push, ← Multiset.coe_add, Multiset.coe_singleton,
          add_comm _ {y}, Multiset.singleton_add]
    rw [OrderStatistics.rollDice_succ, OrderStatistics.rollDice_succ,
      PMF.bind_bind]
    calc _ = (rollDice (J : Int) die).bind fun r => (die.bind step).bind
            fun y => (bagDraws step r.results).map (y ::ₘ ·) := by
          refine congrArg _ (funext fun r => ?_)
          simp only [PMF.bind_map, Function.comp_def, Record.push,
            bagDraws_concat, lift, PMF.bind_bind]
      _ = (die.bind step).bind fun y => ((rollDice (J : Int) die).bind
            fun r => bagDraws step r.results).map (y ::ₘ ·) := by
          rw [PMF.bind_comm]
          simp only [PMF.map_bind]
      _ = _ := by
          rw [ih, PMF.map_bind]
          simp only [PMF.map_comp, Function.comp_def, hpush]
          simp only [← PMF.bind_pure_comp]
          exact PMF.bind_comm _ _ _

/-! ## Pools -/

/--
The results still kept of fixed results that a criterion does not match, and
the number that it does, from a position on, as `survivors` in
`propagation/record.rs` walks them.

# Parameters
- `lowest`: The first kept position.
- `stop`: The position just past the last kept one.
- `c`: The criterion.
- `j`: The position of the first of the results.
- `results`: The distinct results, each with its number of copies.

# Returns
The distinct results still kept that `c` does not match, each with its kept
copies, and the number of kept results that it matches.
-/
def survivorsFrom (lowest stop : Int) (c : Criterion) :
    Int → List (Int × Nat) → List (Int × Nat) × Nat
  | _, [] => ([], 0)
  | j, (v, m) :: rest =>
    let placed := (Max.max 0 (Min.min stop (j + m) - Max.max lowest j)).toNat
    let (survivors, matched) := survivorsFrom lowest stop c (j + m) rest
    if placed = 0 then (survivors, matched)
    else if c.matches v then (survivors, matched + placed)
    else ((v, placed) :: survivors, matched)

/--
The results still kept of fixed results, after the lowest `lowest` and the
highest `highest` are dropped, that a criterion does not match, and the number
that it does. Mirrors `survivors` in `propagation/record.rs`.

# Parameters
- `results`: The distinct results, in ascending order, each with its number
  of copies.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.

# Returns
The distinct results still kept that `c` does not match, in ascending order,
each with its kept copies, and the number of kept results that it matches.

# Examples
```lean
#guard survivors [(1, 2), (3, 1), (5, 2)] 1 1 (.within 3 3) ==
  ([(1, 1), (5, 1)], 1)
```
-/
def survivors (results : List (Int × Nat)) (lowest highest : Nat)
    (c : Criterion) : List (Int × Nat) × Nat :=
  survivorsFrom lowest (((results.map (·.2)).sum : Nat) - (highest : Int)) c 0
    results

/--
The union of two multisets of fixed results. Mirrors `join` in
`propagation/record.rs`.

# Parameters
- `xs`, `ys`: The distinct results of each, each with its number of copies.

# Returns
The distinct results of the union, in ascending order, each with its number of
copies.

# Examples
```lean
#guard join [(1, 1), (3, 2)] [(2, 1), (3, 1)] == [(1, 1), (2, 1), (3, 3)]
```
-/
def join (xs ys : List (Int × Nat)) : List (Int × Nat) :=
  tally (Roll.expand (xs ++ ys))

/--
Reroll the dice still kept of a roll, after its lowest `lowest` and highest
`highest` results are dropped, as the effect of the reroll directs, and answer
the pool of the multisets of the dice that remain kept. Mirrors `Roll::pool`
in `propagation/record.rs`: for each outcome of the roll, reached along `b`
branches, the survivors join each outcome of the replacements of the `k` dice
still kept that the criterion matches, reached along `v`, weighed
`b · v · G^(K − k)`, where `K` is the number of dice still kept.

# Parameters
- `ρ`: The roll.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.
- `padding`: The branches that the reroll adds to each die still kept, `G`.
- `law`: The final results of a die that the criterion matched.

# Returns
The pool.

# Notes
The Rust merges the multisets that coincide, adding their weights; this lists
them as they come, which weighs every multiset alike.
-/
def Roll.rerollPool (ρ : Roll) (lowest highest : Nat) (c : Criterion)
    (padding : Nat) (law : List (Int × Nat)) : Roll :=
  let kept := ρ.results - lowest - highest
  .pool (ρ.outcomes.flatMap fun e =>
    let (rest, matched) := survivors e.1 lowest highest c
    (Roll.custom matched law).outcomes.map fun o =>
      (join rest o.1, e.2 * padding ^ (kept - matched) * o.2))

/-! ## The law of a pool -/

/--
Weighted values, if some weight is positive, give each value its share of the
total weight.

# Parameters
- `default`: The value if no weight is positive.
- `l`: The values, each with its weight.
- `x`: The value.

# Hypotheses
- `hW`: The total weight is positive.
-/
theorem weighted_apply {α : Type} [DecidableEq α] (default : α)
    (l : List (α × Nat)) (hW : (l.map (·.2)).sum ≠ 0) (x : α) :
    weighted default l x =
      (l.map fun e => (e.2 : ℝ≥0∞) * if e.1 = x then 1 else 0).sum /
        (((l.map (·.2)).sum : Nat) : ℝ≥0∞) := by
  rw [← Outcomes.tsum_weighted default l hW (fun y => if y = x then 1 else 0)]
  rw [tsum_eq_single x fun y hy => by simp [hy]]
  simp

/--
Weighted values can take each value of positive weight.

# Parameters
- `e`: A value with its weight.

# Hypotheses
- `he`: `e` is among the values.
- `h0`: Its weight is positive.
-/
theorem mem_support_weighted_of_mem {α : Type} {d : α} {l : List (α × Nat)}
    {e : α × Nat} (he : e ∈ l) (h0 : e.2 ≠ 0) :
    e.1 ∈ (weighted d l).support := by
  have hx : e.1 ∈ l.flatMap fun p => List.replicate p.2 p.1 :=
    List.mem_flatMap.mpr ⟨e, he, List.mem_replicate.mpr ⟨h0, rfl⟩⟩
  have hne := List.ne_nil_of_mem hx
  unfold weighted
  simp only [hne, ↓reduceDIte, PMF.mem_support_ofMultiset_iff,
    Multiset.mem_toFinset, Multiset.mem_coe]
  exact hx

/-- A sum over a list of lists sums the sums of the lists. -/
private theorem sum_flatMap {α : Type} (l : List α) (f : α → List Nat) :
    (l.flatMap f).sum = (l.map fun a => (f a).sum).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]

/--
Weighted values that list, for each outer value, the inner values that it
leads to, each weighed by the outer value's weight, a scale, and its own
weight, are the outer values followed by the inner, if every outer value of
positive weight scales its inner values to the same total: the law of a pool,
whose multisets `Roll::pool` in `propagation/record.rs` weighs by
`b · G^(K − k) · v`, so that the survivors and replacements of every outcome
of the roll weigh `Gᴷ` in all.

# Parameters
- `d`, `d₀`, `dh`: The defaults of the flattened, outer, and inner values.
- `l`: The outer values, each with its weight.
- `h`: The inner values of each outer value, each with its weight.
- `c`: The scale of each outer value's inner values.
- `f`: The value that each outer and inner value lead to.
- `T`: The common total.

# Hypotheses
- `hT`: The common total is positive.
- `hl`: The total weight of the outer values is positive.
- `hc`: Every outer value of positive weight scales its inner values to the
  common total.
-/
theorem weighted_flatMap {α β γ : Type} [DecidableEq γ] (d : γ) (d₀ : α)
    (dh : α → β) (l : List (α × Nat)) (h : α → List (β × Nat))
    (c : α → Nat) (f : α → β → γ) (T : Nat) (hT : T ≠ 0)
    (hl : (l.map (·.2)).sum ≠ 0)
    (hc : ∀ e ∈ l, e.2 ≠ 0 → c e.1 * ((h e.1).map (·.2)).sum = T) :
    weighted d (l.flatMap fun e =>
        (h e.1).map fun o => (f e.1 o.1, e.2 * c e.1 * o.2)) =
      (weighted d₀ l).bind fun a => (weighted (dh a) (h a)).map (f a) := by
  -- The total of the flattened weights.
  have htot : ((l.flatMap fun e =>
      (h e.1).map fun o => (f e.1 o.1, e.2 * c e.1 * o.2)).map (·.2)).sum =
        (l.map (·.2)).sum * T := by
    rw [List.map_flatMap, sum_flatMap, ← List.sum_map_mul_right]
    refine congrArg List.sum (List.map_congr_left fun e he => ?_)
    simp only [List.map_map, Function.comp_def]
    rw [show (fun o : β × Nat => e.2 * c e.1 * o.2) =
      fun o => (e.2 * c e.1) * o.2 from rfl, List.sum_map_mul_left]
    by_cases h0 : e.2 = 0
    · simp [h0]
    · rw [mul_assoc, hc e he h0]
  have hST : (l.map (·.2)).sum * T ≠ 0 := Nat.mul_ne_zero hl hT
  ext z
  rw [weighted_apply d _ (htot ▸ hST), htot, PMF.bind_apply,
    Outcomes.tsum_weighted d₀ l hl]
  -- Each element's share.
  have hterm : ∀ e ∈ l, (e.2 : ℝ≥0∞) * ((weighted (dh e.1) (h e.1)).map
      (f e.1)) z =
        (((h e.1).map fun o => (f e.1 o.1, e.2 * c e.1 * o.2)).map
          fun e' => (e'.2 : ℝ≥0∞) * if e'.1 = z then 1 else 0).sum /
          (T : ℝ≥0∞) := by
    intro e he
    by_cases h0 : e.2 = 0
    · simp [h0, Function.comp_def]
    have hH : ((h e.1).map (·.2)).sum ≠ 0 := fun hH =>
      hT (by rw [← hc e he h0, hH, mul_zero])
    have hc0 : c e.1 ≠ 0 := fun hc0 =>
      hT (by rw [← hc e he h0, hc0, zero_mul])
    rw [Outcomes.map_apply_eq_tsum, Outcomes.tsum_weighted _ _ hH,
      ← hc e he h0, ← mul_div_assoc, Nat.cast_mul,
      ← ENNReal.mul_div_mul_left _ _ (Nat.cast_ne_zero.mpr hc0)
        (ENNReal.natCast_ne_top _)]
    congr 1
    simp only [List.map_map, Function.comp_def, ← List.sum_map_mul_left]
    refine congrArg List.sum (List.map_congr_left fun o _ => ?_)
    push_cast
    split_ifs <;> ring
  rw [List.map_congr_left hterm, List.map_flatMap]
  have hsum : ∀ (l : List (α × Nat)) (g : α × Nat → List ℝ≥0∞),
      (l.flatMap g).sum = (l.map fun e => (g e).sum).sum := by
    intro l g
    induction l with
    | nil => rfl
    | cons a l ih => simp [List.flatMap_cons, ih]
  rw [hsum]
  simp only [ENNReal.div_eq_inv_mul, ← List.sum_map_mul_left]
  refine congrArg List.sum (List.map_congr_left fun e _ =>
    congrArg List.sum (List.map_congr_left fun o _ => ?_))
  rw [Nat.cast_mul, ENNReal.mul_inv (.inr (ENNReal.natCast_ne_top _))
    (.inl (ENNReal.natCast_ne_top _))]
  ring

namespace Roll

/--
The weights of a well-formed roll's outcomes sum to its total, which is
positive.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
-/
theorem sum_outcomes (ρ : Roll) (hρ : ρ.WellFormed) :
    (ρ.outcomes.map (·.2)).sum = ρ.total ∧ ρ.total ≠ 0 := by
  have h := ρ.tsum_law_sorted hρ fun _ => 1
  simp only [Nat.cast_one, mul_one, PMF.tsum_coe] at h
  have hi : OrderStatistics.integral ρ.outcomes
      (fun rs => (fun _ => 1) (expand rs)) =
      (ρ.outcomes.map (·.2)).sum := by
    simp [OrderStatistics.integral]
  rw [hi] at h
  have hT : ρ.total ≠ 0 := fun hT => by
    rw [hT, Nat.cast_zero] at h
    generalize (ρ.outcomes.map (·.2)).sum = n at h
    by_cases h0 : (n : ℝ≥0∞) = 0
    · rw [h0, ENNReal.zero_div] at h
      exact one_ne_zero h
    · rw [ENNReal.div_zero h0] at h
      exact ENNReal.one_ne_top h
  refine ⟨?_, hT⟩
  have := (ENNReal.div_eq_one_iff (Nat.cast_ne_zero.mpr hT)
    (ENNReal.natCast_ne_top _)).mp h.symm
  exact_mod_cast this

/--
The law of the record that a well-formed roll fills, sorted, is its outcomes,
weighed and expanded.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `hρ`: The roll is well formed.
-/
theorem map_sorted_law_weighted (ρ : Roll) (hρ : ρ.WellFormed) :
    ρ.law.map Record.sorted = (weighted [] ρ.outcomes).map expand := by
  obtain ⟨hS, hT⟩ := ρ.sum_outcomes hρ
  ext ys
  rw [map_sorted_law ρ hρ, Outcomes.map_apply_eq_tsum,
    Outcomes.tsum_weighted _ _ (hS ▸ hT), hS]
  congr 1
  simp only [OrderStatistics.integral, Nat.cast_list_sum, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun e _ => ?_)
  simp only [Function.comp_apply]
  split_ifs <;> simp

/-- Every multiset of the program, from faces of positive weight, has positive
weight. -/
private theorem weight_pos_outcomeStates :
    ∀ (die : List (Int × Nat)), (∀ q ∈ die, 0 < q.2) →
      ∀ (J : Nat) (e : List (Int × Nat) × Nat),
        e ∈ Outcomes.outcomeStates die J → 0 < e.2 := by
  intro die
  induction die using List.reverseRecOn with
  | nil =>
    intro _ J e he
    simp only [Outcomes.outcomeStates, List.foldl_nil, Outcomes.initial] at he
    split at he
    · simp_all
    · simp at he
  | append_singleton die face ih =>
    intro hpos J e he
    rw [Outcomes.outcomeStates_append] at he
    simp only [Outcomes.outcomeStep, List.mem_flatMap, List.mem_range,
      List.mem_map] at he
    obtain ⟨m, hm, e', he', rfl⟩ := he
    exact Nat.mul_pos (Nat.mul_pos (Nat.choose_pos (by omega))
      (Nat.pow_pos (hpos face (by simp))))
      (ih (fun q hq => hpos q (by simp [hq])) _ e' he')

/--
Every outcome of a well-formed roll has positive weight, as `Roll::outcomes`
in `propagation/record.rs` weighs them.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem weight_pos_of_mem_outcomes {ρ : Roll} {e : List (Int × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.outcomes) : 0 < e.2 := by
  have hdice : ∀ (n : Nat) (die : List (Int × Nat)), (∀ q ∈ die, 0 < q.2) →
      e ∈ diceOutcomes n die → 0 < e.2 := by
    intro n die hpos he
    unfold diceOutcomes at he
    split at he
    · simp_all
    split at he
    · simp_all
    exact weight_pos_outcomeStates die hpos n e he
  cases ρ with
  | range start stop =>
    simp only [outcomes] at he
    split at he
    · simp_all
    · simp only [List.mem_map] at he
      obtain ⟨i, _, rfl⟩ := he
      exact Nat.one_pos
  | standard count faces =>
    exact hdice _ _ (fun q hq => by
      simp only [OrderStatistics.standardFaces, List.mem_map] at hq
      obtain ⟨i, _, rfl⟩ := hq
      exact Nat.one_pos) he
  | custom count faces => exact hdice _ _ hρ.2 he
  | fixed rs =>
    simp only [outcomes, List.mem_singleton] at he
    subst he
    exact Nat.one_pos
  | pool outcomes => exact (hρ.2.1 e he).2

/--
Every outcome of a well-formed roll holds the sorted results of a record that
the roll can fill.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem exists_sorted_of_mem_outcomes {ρ : Roll} {e : List (Int × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.outcomes) :
    ∃ r ∈ ρ.law.support, r.sorted = expand e.1 := by
  have hm : expand e.1 ∈ (ρ.law.map Record.sorted).support := by
    rw [ρ.map_sorted_law_weighted hρ, PMF.support_map]
    exact ⟨e.1, mem_support_weighted_of_mem he
      (weight_pos_of_mem_outcomes hρ he).ne', rfl⟩
  rw [PMF.support_map] at hm
  exact hm

/--
Every outcome of a well-formed roll holds its number of results.

# Parameters
- `ρ`: The roll.
- `e`: An outcome with its weight.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem length_of_mem_outcomes {ρ : Roll} {e : List (Int × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.outcomes) :
    (expand e.1).length = ρ.results := by
  obtain ⟨r, hr, hre⟩ := exists_sorted_of_mem_outcomes hρ he
  rw [← hre, Record.sorted, List.length_mergeSort]
  exact (mem_support_law hρ.even hr).1

end Roll

/-- The values of a list at positions from `lo` to just before `st`, the
first at position `j`, as `survivors` in `propagation/record.rs` places
results. -/
private def window (lo st : Int) : Int → List Int → List Int
  | _, [] => []
  | j, x :: xs =>
    (if lo ≤ j ∧ j < st then [x] else []) ++ window lo st (j + 1) xs

/-- The values at positions from `lo` to just before `st` drop those before
`lo` and take those before `st`. -/
private theorem window_eq (lo st : Int) :
    ∀ (j : Int) (xs : List Int), window lo st j xs =
      (xs.drop (lo - j).toNat).take (st - Max.max lo j).toNat
  | _, [] => by simp [window]
  | j, x :: xs => by
    rw [window, window_eq lo st (j + 1) xs]
    split_ifs with h
    · rw [show (lo - j).toNat = 0 by omega,
        show (lo - (j + 1)).toNat = 0 by omega,
        show (st - Max.max lo j).toNat =
          (st - Max.max lo (j + 1)).toNat + 1 by omega]
      simp
    · by_cases hj : j < lo
      · rw [show (lo - j).toNat = (lo - (j + 1)).toNat + 1 by omega,
          show Max.max lo (j + 1) = Max.max lo j by omega]
        simp
      · rw [show (st - Max.max lo (j + 1)).toNat = 0 by omega,
          show (st - Max.max lo j).toNat = 0 by omega]
        simp

/-- The positions from `j` to just before `j + m` that lie from `lo` to just
before `st`, as `survivors` in `propagation/record.rs` places copies. -/
private def placed (lo st j : Int) (m : Nat) : Nat :=
  (Max.max 0 (Min.min st (j + m) - Max.max lo j)).toNat

/-- Copies of a value, before the rest, place as many of their positions as
the window holds. -/
private theorem window_replicate_append (lo st v : Int) (ys : List Int) :
    ∀ (m : Nat) (j : Int), window lo st j (List.replicate m v ++ ys) =
      List.replicate (placed lo st j m) v ++ window lo st (j + m) ys
  | 0, j => by
    rw [show placed lo st j 0 = 0 by simp only [placed]; omega]
    simp
  | m + 1, j => by
    rw [List.replicate_succ, List.cons_append, window,
      window_replicate_append lo st v ys m (j + 1)]
    rw [show j + 1 + (m : Int) = j + ((m + 1 : Nat) : Int) by push_cast; ring]
    split_ifs with h
    · rw [show placed lo st j (m + 1) = placed lo st (j + 1) m + 1 by
        simp only [placed]; push_cast; omega]
      simp [List.replicate_succ]
    · rw [show placed lo st j (m + 1) = placed lo st (j + 1) m by
        simp only [placed]; push_cast; omega]
      simp

/-- The survivors from a position on are the window's values that a criterion
does not match, and the matched, the number that it does. -/
private theorem survivorsFrom_spec (lo st : Int) (c : Criterion) :
    ∀ (j : Int) (rs : List (Int × Nat)),
      Roll.expand (survivorsFrom lo st c j rs).1 =
          (window lo st j (Roll.expand rs)).filter (!c.matches ·) ∧
        (survivorsFrom lo st c j rs).2 =
          (window lo st j (Roll.expand rs)).countP c.matches
  | _, [] => by simp [survivorsFrom, window, Roll.expand]
  | j, (v, m) :: rest => by
    obtain ⟨ih₁, ih₂⟩ := survivorsFrom_spec lo st c (j + m) rest
    have hx : Roll.expand ((v, m) :: rest) =
        List.replicate m v ++ Roll.expand rest := by
      simp [Roll.expand]
    rw [hx, window_replicate_append]
    simp only [survivorsFrom, placed]
    generalize (Max.max 0 (Min.min st (j + m) - Max.max lo j)).toNat = n
    rcases hs : survivorsFrom lo st c (j + m) rest with ⟨sv, mt⟩
    rw [hs] at ih₁ ih₂
    dsimp only at ih₁ ih₂ ⊢
    split_ifs with h0 hm
    · subst h0
      simp [ih₁, ih₂]
    · simp [List.filter_append, List.countP_append, ih₁, ih₂, hm,
        List.countP_replicate]
      omega
    · simp only [Roll.expand] at ih₁
      simp [List.filter_append, List.countP_append, ih₁, ih₂, hm,
        List.countP_replicate, Roll.expand]

/--
The results that `survivors` leaves are the kept results that the criterion
does not match, and the number that it answers is the number of kept results
that it does, as a record of the results views them (`Record.kept`).

# Parameters
- `rs`: The distinct results, each with its number of copies.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.
-/
theorem survivors_spec (rs : List (Int × Nat)) (lowest highest : Nat)
    (c : Criterion) :
    Roll.expand (survivors rs lowest highest c).1 =
        (((Roll.expand rs).drop lowest).take
          ((Roll.expand rs).length - lowest - highest)).filter
            (!c.matches ·) ∧
      (survivors rs lowest highest c).2 =
        (((Roll.expand rs).drop lowest).take
          ((Roll.expand rs).length - lowest - highest)).countP c.matches := by
  have hlen : (Roll.expand rs).length = ((rs.map (·.2)).sum : Nat) := by
    simp [Roll.expand, List.length_flatMap]
  have hw : window lowest (((rs.map (·.2)).sum : Nat) - (highest : Int)) 0
      (Roll.expand rs) = ((Roll.expand rs).drop lowest).take
        ((Roll.expand rs).length - lowest - highest) := by
    rw [window_eq, hlen]
    congr 2
    omega
  rw [← hw]
  exact survivorsFrom_spec _ _ c 0 rs

/--
The union of two multisets of fixed results holds the results of both.

# Parameters
- `xs`, `ys`: The distinct results of each, each with its number of copies.
-/
theorem perm_expand_join (xs ys : List (Int × Nat)) :
    (Roll.expand (join xs ys)).Perm (Roll.expand xs ++ Roll.expand ys) := by
  rw [join]
  exact (expand_tally_perm _).trans
    (by rw [Roll.expand, List.flatMap_append]; rfl)

/--
The results that a reroll leaves of two lists of results are the results that
it leaves of each, independently.

# Parameters
- `step`: The law of the result that each result leaves.
- `xs`, `ys`: The results.
-/
theorem bagDraws_append (step : Int → PMF Int) (xs ys : List Int) :
    bagDraws step (xs ++ ys) =
      lift (· + ·) (bagDraws step xs) (bagDraws step ys) := by
  induction xs with
  | nil =>
    simp only [List.nil_append, bagDraws, lift, PMF.pure_bind, zero_add]
    exact (PMF.map_id _).symm
  | cons x xs ih =>
    simp only [List.cons_append, bagDraws, ih, lift, PMF.bind_bind,
      PMF.map_bind, PMF.bind_map, PMF.map_comp, Function.comp_def]
    refine congrArg _ (funext fun a => congrArg _ (funext fun M =>
      congrArg (PMF.map · _) (funext fun N => ?_)))
    exact (Multiset.cons_add _ _ _).symm

/--
The results that a reroll leaves depend only on the laws of the results that
it leaves of the results present.

# Hypotheses
- `h`: The laws agree on every result present.
-/
theorem bagDraws_congr {step step' : Int → PMF Int} {xs : List Int}
    (h : ∀ x ∈ xs, step x = step' x) :
    bagDraws step xs = bagDraws step' xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [bagDraws, h x List.mem_cons_self,
      ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

/--
A reroll that leaves every result as it was leaves them.

# Hypotheses
- `h`: Every result leaves itself.
-/
theorem bagDraws_pure {step : Int → PMF Int} {xs : List Int}
    (h : ∀ x ∈ xs, step x = PMF.pure x) :
    bagDraws step xs = PMF.pure (xs : Multiset Int) := by
  rw [bagDraws_congr h]
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [bagDraws, ih fun y hy => h y (List.mem_cons_of_mem _ hy)]
    simp [lift, PMF.pure_bind, PMF.pure_map]

/--
Results that a reroll replaces alike are as many dice of the replacement.

# Parameters
- `P`: The law of every replacement.
- `xs`: The results.
-/
theorem bagDraws_const (P : PMF Int) (xs : List Int) :
    bagDraws (fun _ => P) xs =
      (rollDice (xs.length : Int) P).map
        fun r => (r.results : Multiset Int) := by
  have hc : ∀ ys : List Int, ys.length = xs.length →
      bagDraws (fun _ => P) ys = bagDraws (fun _ => P) xs := by
    intro ys
    induction ys generalizing xs with
    | nil => intro h; rw [List.eq_nil_of_length_eq_zero h.symm]
    | cons y ys ih =>
      intro h
      cases xs with
      | nil => cases h
      | cons x xs =>
        simp only [bagDraws, ih xs (by simpa using h)]
  have := bind_rollDice_bagDraws (xs.length : Int) (PMF.pure 0) fun _ => P
  rw [Outcomes.rollDice_pure, PMF.pure_bind, PMF.pure_bind] at this
  rw [← this]
  exact (hc _ (by simp)).symm

/--
The multiset of the results of dice of weighted faces is the outcomes of the
custom roll of those dice, weighed and expanded.

# Parameters
- `k`: The number of dice.

# Hypotheses
- `hg`: The custom roll of the dice is well formed.
-/
theorem map_bag_rollDice_weighted (k : Int) {g : List (Int × Nat)}
    (hg : (Roll.custom k g).WellFormed) :
    (rollDice k (weightedDie g)).map (fun r => (r.results : Multiset Int)) =
      (weighted [] (Roll.custom k g).outcomes).map
        fun o => (Roll.expand o : Multiset Int) := by
  have h := congrArg (PMF.map (fun xs : List Int => (xs : Multiset Int)))
    (Roll.map_sorted_law_weighted _ hg)
  simp only [PMF.map_comp, Roll.law] at h
  refine Eq.trans ?_ (h.trans (congrArg (PMF.map · _) rfl))
  refine map_congr_support fun r _ => ?_
  simp only [Function.comp_apply, Record.sorted]
  exact (Multiset.coe_eq_coe.mpr (List.mergeSort_perm _ _)).symm

/--
Rerolling the kept results of a multiset of fixed results leaves, as a
multiset, the survivors joined with each outcome of the replacements of the
results that the criterion matches, as `Roll::pool` in
`propagation/record.rs` joins them, weighed as those outcomes are.

# Parameters
- `rs`: The distinct results, each with its number of copies.
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.

# Hypotheses
- `hg`: Every custom roll of dice of the final results of a matched die, `g`,
  is well formed.
-/
theorem bagDraws_kept (rs : List (Int × Nat)) (lowest highest : Nat)
    (c : Criterion) {g : List (Int × Nat)}
    (hg : ∀ k : Int, (Roll.custom k g).WellFormed) :
    bagDraws (fun x => if c.matches x then weightedDie g else PMF.pure x)
        (((Roll.expand rs).drop lowest).take
          ((Roll.expand rs).length - lowest - highest)) =
      (weighted [] (Roll.custom ((survivors rs lowest highest c).2 : Int)
          g).outcomes).map
        fun o => (Roll.expand (join (survivors rs lowest highest c).1 o) :
          Multiset Int) := by
  obtain ⟨h₁, h₂⟩ := survivors_spec rs lowest highest c
  generalize ((Roll.expand rs).drop lowest).take
    ((Roll.expand rs).length - lowest - highest) = W at h₁ h₂
  rw [bagDraws_perm _ ((List.filter_append_perm (!c.matches ·) W).symm),
    bagDraws_append, bagDraws_pure fun x hx => by
      have := (List.mem_filter.mp hx).2
      simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
      simp [this],
    bagDraws_congr (step' := fun _ => weightedDie g) fun x hx => by
      have := (List.mem_filter.mp hx).2
      simp only [Bool.not_not] at this
      simp [this],
    bagDraws_const]
  have hlen : ((W.filter fun x => !!c.matches x).length : Int) =
      ((survivors rs lowest highest c).2 : Int) := by
    rw [h₂, List.countP_eq_length_filter]
    simp
  rw [hlen, map_bag_rollDice_weighted _ (hg _), lift, PMF.pure_bind,
    PMF.map_comp, ← h₁]
  refine congrArg (PMF.map · _) (funext fun o => ?_)
  simp only [Function.comp_apply, Multiset.coe_add]
  exact (Multiset.coe_eq_coe.mpr (perm_expand_join _ _)).symm

/--
A tally ascends strictly, and counts each value at least once.

# Parameters
- `xs`: The values.
-/
theorem tally_wellFormed (xs : List Int) :
    (tally xs).Pairwise (fun a b => a.1 < b.1) ∧ ∀ p ∈ tally xs, 0 < p.2 := by
  have hnd : (xs.dedup.mergeSort (· ≤ ·)).Nodup :=
    (List.nodup_dedup xs).perm (List.mergeSort_perm _ _).symm
  have hs : (xs.dedup.mergeSort (· ≤ ·)).Pairwise (· ≤ ·) := by
    simpa using List.pairwise_mergeSort (le := fun a b : Int => decide (a ≤ b))
      (fun a b c hab hbc => by simp at *; omega)
      (fun a b => by simp; omega) xs.dedup
  refine ⟨?_, fun p hp => ?_⟩
  · rw [tally, List.pairwise_map]
    exact (hs.and hnd).imp fun ⟨hle, hne⟩ => lt_of_le_of_ne hle hne
  · simp only [tally, List.mem_map, List.mem_mergeSort, List.mem_dedup] at hp
    obtain ⟨x, hx, rfl⟩ := hp
    exact List.count_pos_iff.mpr hx

/--
The survivors of an outcome of a well-formed roll and the results that the
criterion matches are, together, the results still kept.

# Parameters
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.

# Hypotheses
- `hρ`: The roll is well formed.
- `he`: `e` is an outcome of the roll.
-/
theorem survivors_count {ρ : Roll} {e : List (Int × Nat) × Nat}
    (hρ : ρ.WellFormed) (he : e ∈ ρ.outcomes) (lowest highest : Nat)
    (c : Criterion) :
    (Roll.expand (survivors e.1 lowest highest c).1).length +
        (survivors e.1 lowest highest c).2 =
      ρ.results - lowest - highest := by
  obtain ⟨h₁, h₂⟩ := survivors_spec e.1 lowest highest c
  rw [h₁, h₂, ← List.countP_eq_length_filter]
  generalize hW : ((Roll.expand e.1).drop lowest).take
    ((Roll.expand e.1).length - lowest - highest) = W
  have hlen : W.length = ρ.results - lowest - highest := by
    rw [← hW]
    simp only [List.length_take, List.length_drop,
      Roll.length_of_mem_outcomes hρ he]
    omega
  rw [← hlen, List.length_eq_countP_add_countP c.matches, Nat.add_comm]
  simp

/-- The replacements of `k` dice still kept that a criterion matches, and the
padding of the other `K − k`, weigh `Gᴷ` in all. -/
private theorem sum_replacements {g : List (Int × Nat)} {G : Nat}
    (hglen : (Roll.expand g).length = G) (hG : G ≠ 0)
    (hgw : ∀ k : Int, (Roll.custom k g).WellFormed) {k K : Nat}
    (hkK : k ≤ K) :
    G ^ (K - k) * ((Roll.custom (k : Int) g).outcomes.map (·.2)).sum =
      G ^ K := by
  have hgne : g ≠ [] := fun h => hG (by rw [← hglen, h]; rfl)
  have htot : OrderStatistics.total g = G := by
    rw [← hglen, OrderStatistics.total]
    simp [Roll.expand, List.length_flatMap]
  rw [(Roll.sum_outcomes _ (hgw _)).1, Roll.total]
  by_cases hk0 : k = 0
  · simp [hk0]
  · split_ifs with hpos
    · rw [htot, Int.toNat_natCast, ← pow_add, Nat.sub_add_cancel hkK]
    · exact absurd ⟨by omega, hgne⟩ hpos

/-- Weighted results hold as many results as their weights. -/
private theorem length_expand (rs : List (Int × Nat)) :
    (Roll.expand rs).length = (rs.map (·.2)).sum := by
  simp [Roll.expand, List.length_flatMap]

/--
The pool that a reroll leaves of a well-formed roll is well formed, as
`Roll::pool` in `propagation/record.rs` makes it: it holds a multiset, each
multiset ascends strictly, with positive copies and positive weight, and each
holds the results still kept.

# Parameters
- `lowest`: The number of lowest results dropped.
- `highest`: The number of highest results dropped.
- `c`: The criterion.

# Hypotheses
- `hρ`: The roll is well formed.
- `hG`: The padding is positive.
- `hgw`: Every custom roll of dice of the final results is well formed.
-/
theorem Roll.wellFormed_rerollPool {ρ : Roll} {lowest highest : Nat}
    {c : Criterion} {G : Nat} {g : List (Int × Nat)} (hρ : ρ.WellFormed)
    (hG : G ≠ 0)
    (hgw : ∀ k : Int, (Roll.custom k g).WellFormed) :
    (ρ.rerollPool lowest highest c G g).WellFormed := by
  set K := ρ.results - lowest - highest
  let sv : List (Int × Nat) → List (Int × Nat) × Nat := fun rs =>
    survivors rs lowest highest c
  have hne : ∀ τ : Roll, τ.WellFormed → τ.outcomes ≠ [] := fun τ hτ h => by
    have := τ.sum_outcomes hτ
    rw [h] at this
    exact this.2 this.1.symm
  -- Every multiset of the pool is well formed and holds the dice still kept.
  have hmem : ∀ x ∈ (ρ.rerollPool lowest highest c G g).outcomes,
      ((x.1.Pairwise (fun a b => a.1 < b.1) ∧ ∀ p ∈ x.1, 0 < p.2) ∧
        0 < x.2) ∧ (x.1.map (·.2)).sum = K := by
    intro x hx
    simp only [Roll.rerollPool, Roll.outcomes, List.mem_flatMap,
      List.mem_map] at hx
    obtain ⟨e, he, o, ho, rfl⟩ := hx
    have hcount := survivors_count hρ he lowest highest c
    have ho' : o ∈ (Roll.custom ((sv e.1).2 : Int) g).outcomes := ho
    refine ⟨⟨tally_wellFormed _, Nat.mul_pos (Nat.mul_pos
      (Roll.weight_pos_of_mem_outcomes hρ he) (Nat.pow_pos
        (Nat.pos_of_ne_zero hG)))
      (Roll.weight_pos_of_mem_outcomes (hgw _) ho')⟩, ?_⟩
    rw [← length_expand, (perm_expand_join _ _).length_eq, List.length_append,
      Roll.length_of_mem_outcomes (hgw _) ho']
    simp only [Roll.results, Int.toNat_natCast]
    exact hcount
  refine ⟨?_, fun x hx => (hmem x hx).1, ?_⟩
  · obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ (hne ρ hρ)
    obtain ⟨o, ho⟩ := List.exists_mem_of_ne_nil _
      (hne _ (hgw ((sv e.1).2 : Int)))
    exact List.ne_nil_of_mem (List.mem_flatMap.mpr ⟨e, he,
      List.mem_map.mpr ⟨o, ho, rfl⟩⟩)
  · intro x hx
    rw [(hmem x hx).2]
    change x ∈ (ρ.rerollPool lowest highest c G g).outcomes at hx
    change K = (Roll.pool (ρ.rerollPool lowest highest c G g).outcomes).results
    generalize (ρ.rerollPool lowest highest c G g).outcomes = es at hmem hx ⊢
    cases es with
    | nil => cases hx
    | cons e₀ es =>
      exact ((hmem e₀ List.mem_cons_self).2).symm

/-! ## Rerolling a setup -/

namespace Setup

/--
Reroll the dice still kept of a setup. Mirrors `Setup::reroll` in
`propagation/mixture.rs`.

A reroll that does nothing leaves the setup as it was, and one that crosses
out every die still kept drops them all. Otherwise, if no result has been
dropped and the roll's dice are alike, the reroll rewrites the law of each
(`Roll.rewrite`); if not, it pools the multisets of the dice still kept
(`Roll.rerollPool`). Either way, the setup keeps its die, and drops nothing.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Returns
The rerolled setup.
-/
def reroll (s : Setup) (c : Criterion) (limit : Option Nat) : Setup :=
  match s.roll with
  | none => s
  | some ρ =>
    if s.results - s.lowest - s.highest = 0 then s
    else
      match Effect.resolve s.replacement c limit with
      | .nothing => s
      | .crossOut => { s with lowest := s.results }
      | .redraw padding law =>
        { roll := some
            (if ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0 then
              ρ.rewrite c padding law
            else ρ.rerollPool s.lowest s.highest c padding law)
          die := some s.replacement }

/--
Whether every result of every record that a setup describes is a face of its
die, as every record that `propagation.rs` rolls is. A reroll that rerolls no
face of the die, or that rerolls every face without limit, leaves a record
whose results are faces of its die as the setup leaves it.

# Parameters
- `s`: The setup.

# Returns
Whether the setup is faithful to its die.
-/
def Faithful (s : Setup) : Prop :=
  ∀ r ∈ s.law.support, ∀ x ∈ r.results, x ∈ s.replacement.faces

/--
Whether every die of a setup's roll, if any, has a face (`Roll.Faced`).

# Parameters
- `s`: The setup.

# Returns
Whether every die has a face.
-/
def Faced (s : Setup) : Prop := ∀ ρ ∈ s.roll, ρ.Faced

/--
Whether a reroll pools a setup's dice still kept: whether it may change them,
and they are not alike, or some have been dropped.

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Returns
Whether the reroll pools the setup.
-/
def Pools (s : Setup) (c : Criterion) (limit : Option Nat) : Prop :=
  ∃ ρ padding law, s.roll = some ρ ∧
    s.results - s.lowest - s.highest ≠ 0 ∧
    Effect.resolve s.replacement c limit = .redraw padding law ∧
    ¬(ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0)

/--
Whether rerolling a setup describes the specification's reroll of the records
that it describes, as every later instruction reads them (`Record.view`).

# Parameters
- `s`: The setup.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Returns
Whether the rerolled setup's law is the reroll's, viewed.
-/
def Rerolls (s : Setup) (c : Criterion) (limit : Option Nat) : Prop :=
  (s.reroll c limit).law.map Record.view =
    (s.law.bind fun r => Spec.reroll r c limit).map Record.view

end Setup

/-! ## What a reroll resolves to -/

namespace Effect

/--
A reroll does nothing only if no face matches, the limit is zero, or the die
has one face and the limit is finite.

# Hypotheses
- `h`: The reroll does nothing.
-/
theorem resolve_nothing {d : Die} {c : Criterion} {limit : Option Nat}
    (h : resolve d c limit = .nothing) :
    d.faces.countP c.matches = 0 ∨ limit = some 0 ∨
      ∃ n, limit = some n ∧ d.faces.length = 1 := by
  unfold resolve at h
  dsimp only at h
  by_cases hm : d.faces.countP c.matches = 0
  · exact .inl hm
  simp only [hm, ↓reduceIte] at h
  cases limit with
  | none =>
    dsimp only at h
    split at h <;> cases h
  | some n =>
    dsimp only at h
    split at h
    · rename_i hn
      rcases hn with rfl | hF
      · exact .inr (.inl rfl)
      · exact .inr (.inr ⟨n, rfl, hF⟩)
    · cases h

/--
A reroll crosses out every die still kept only if it has no limit and matches
every face.

# Hypotheses
- `h`: The reroll crosses out every die still kept.
-/
theorem resolve_crossOut {d : Die} {c : Criterion} {limit : Option Nat}
    (h : resolve d c limit = .crossOut) :
    limit = none ∧ d.faces.countP c.matches = d.faces.length := by
  unfold resolve at h
  dsimp only at h
  by_cases hm : d.faces.countP c.matches = 0
  · simp [hm] at h
  simp only [hm, ↓reduceIte] at h
  cases limit with
  | none =>
    dsimp only at h
    split at h
    · rename_i hF
      exact ⟨rfl, hF⟩
    · cases h
  | some n =>
    dsimp only at h
    split at h <;> cases h

/--
A reroll that may change the dice still kept matches some face, and pads each
die by `Fⁿ` and redraws a matched die by the `limitedLaw` if it is limited to
`n` times, or pads each die by `F − m` and redraws a matched die by the
`unlimitedLaw` if it is not limited and does not match every face.

# Hypotheses
- `h`: The reroll may change the dice still kept.
-/
theorem resolve_redraw {d : Die} {c : Criterion} {limit : Option Nat}
    {padding : Nat} {law : List (Int × Nat)}
    (h : resolve d c limit = .redraw padding law) :
    d.faces.countP c.matches ≠ 0 ∧
      ((∃ n, limit = some n ∧ n ≠ 0 ∧
          padding = d.faces.length ^ (n - 1) * d.faces.length ∧
          law = limitedLaw d c n) ∨
        (limit = none ∧ d.faces.countP c.matches ≠ d.faces.length ∧
          padding = d.faces.length - d.faces.countP c.matches ∧
          law = unlimitedLaw d c)) := by
  unfold resolve at h
  dsimp only at h
  by_cases hm : d.faces.countP c.matches = 0
  · simp [hm] at h
  simp only [hm, ↓reduceIte] at h
  refine ⟨hm, ?_⟩
  cases limit with
  | none =>
    dsimp only at h
    split at h
    · cases h
    · rename_i hF
      cases h
      exact .inr ⟨rfl, hF, rfl, rfl⟩
  | some n =>
    dsimp only at h
    split at h
    · cases h
    · rename_i hn
      cases h
      exact .inl ⟨n, rfl, fun h0 => hn (.inl h0), rfl, rfl⟩

end Effect

/-- The positions of the draws are faces of the die. -/
private theorem mem_draws (c : Criterion) (faces : List Int) {x : Int} :
    ∀ k, x ∈ draws c faces k → x ∈ faces
  | 0, h => h
  | k + 1, h => by
    rw [draws, List.mem_flatMap] at h
    obtain ⟨y, hy, hx⟩ := h
    split at hx
    · exact hx
    · rw [List.eq_of_mem_replicate hx]
      exact mem_draws c faces k hy

/-- The final results of a die that a reroll may change are faces of the die,
and weigh the padding, which is positive. -/
private theorem redraw_law {d : Die} {c : Criterion} {limit : Option Nat}
    {padding : Nat} {law : List (Int × Nat)}
    (h : Effect.resolve d c limit = .redraw padding law) :
    (Roll.expand law).length = padding ∧ padding ≠ 0 ∧
      ∀ x ∈ Roll.expand law, x ∈ d.faces := by
  obtain ⟨hm, hcase⟩ := Effect.resolve_redraw h
  have hFlen : d.faces.length ≠ 0 := fun h0 =>
    hm (Nat.eq_zero_of_le_zero (h0 ▸ List.countP_le_length))
  have hmle := List.countP_le_length (p := c.matches) (l := d.faces)
  rcases hcase with ⟨n, rfl, _, rfl, rfl⟩ | ⟨rfl, hmF, rfl, rfl⟩
  · exact ⟨length_expand_limitedLaw _ _ _,
      Nat.mul_ne_zero (pow_ne_zero _ hFlen) hFlen, fun x hx =>
        mem_draws c _ _ ((expand_limitedLaw_perm d c n).mem_iff.mp hx)⟩
  · exact ⟨length_expand_unlimitedLaw _ _, by omega, fun x hx =>
      List.mem_of_mem_filter ((expand_unlimitedLaw_perm d c).mem_iff.mp hx)⟩

/--
The final results of a die that a reroll may change ascend strictly, each of
positive weight, so that every custom roll of them is well formed.

# Parameters
- `k`: The number of dice.

# Hypotheses
- `h`: The reroll may change the dice still kept.
-/
theorem wellFormed_of_redraw {d : Die} {c : Criterion} {limit : Option Nat}
    {padding : Nat} {law : List (Int × Nat)}
    (h : Effect.resolve d c limit = .redraw padding law) (k : Int) :
    (Roll.custom k law).WellFormed := by
  obtain ⟨hm, hcase⟩ := Effect.resolve_redraw h
  obtain ⟨hpw, hpos⟩ := tally_wellFormed d.faces
  rcases hcase with ⟨n, rfl, _, rfl, rfl⟩ | ⟨rfl, _, rfl, rfl⟩
  · refine ⟨by rw [limitedLaw, List.pairwise_map]; exact hpw, fun p hp => ?_⟩
    simp only [limitedLaw, List.mem_map] at hp
    obtain ⟨q, hq, rfl⟩ := hp
    refine Nat.mul_pos (hpos q hq) ?_
    split
    · exact Nat.pow_pos (Nat.pos_of_ne_zero hm)
    · exact Nat.add_pos_right _ (Nat.pow_pos (Nat.pos_of_ne_zero hm))
  · exact ⟨hpw.filter _, fun p hp => hpos p (List.mem_of_mem_filter hp)⟩

/-! ## Rerolls that leave the kept results -/

/--
A reroll of kept results, each of which it leaves as it was, leaves them in
place.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.

# Hypotheses
- `h`: Every kept result that the reroll rerolls is replaced by itself.
-/
theorem rerollKept_of_pure {die : Die} {kept : List Int}
    {rerolled : Int → Bool} {replacement : PMF Int}
    (h : ∀ x ∈ kept, rerolled x = true → replacement = PMF.pure x) :
    rerollKept die kept rerolled replacement =
      PMF.pure { results := kept, die } := by
  have key : ∀ (xs : List Int) (r₀ : Record),
      (∀ x ∈ xs, rerolled x = true → replacement = PMF.pure x) →
      xs.foldl (fun p x =>
          if rerolled x then p.bind fun r => replacement.map r.push
          else p.map (·.push x)) (PMF.pure r₀) =
        PMF.pure { r₀ with results := r₀.results ++ xs } := by
    intro xs
    induction xs with
    | nil => intro r₀ _; simp
    | cons x xs ih =>
      intro r₀ hx
      rw [List.foldl_cons]
      have hstep : (if rerolled x then
            (PMF.pure r₀).bind fun r => replacement.map r.push
          else (PMF.pure r₀).map (·.push x)) = PMF.pure (r₀.push x) := by
        split
        · rename_i hr
          rw [hx x List.mem_cons_self hr, PMF.pure_bind, PMF.pure_map]
        · rw [PMF.pure_map]
      rw [hstep, ih _ fun y hy => hx y (List.mem_cons_of_mem _ hy)]
      simp [Record.push]
  rw [rerollKept, key kept _ h]
  rfl

/--
Every record that a reroll of kept results reaches drops nothing and holds the
die.

# Parameters
- `r`: The record reached.

# Hypotheses
- `h`: The reroll reaches `r`.
-/
theorem mem_support_rerollKept {die : Die} {kept : List Int}
    {rerolled : Int → Bool} {replacement : PMF Int} {r : Record}
    (h : r ∈ (rerollKept die kept rerolled replacement).support) :
    r.lowest = 0 ∧ r.highest = 0 ∧ r.die = die := by
  let P : Record → Prop := fun r =>
    r.lowest = 0 ∧ r.highest = 0 ∧ r.die = die
  have key : ∀ (xs : List Int) (p : PMF Record),
      (∀ r ∈ p.support, P r) →
      ∀ r ∈ (xs.foldl (fun p x =>
          if rerolled x then p.bind fun r => replacement.map r.push
          else p.map (·.push x)) p).support, P r := by
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
    exact ⟨rfl, rfl, rfl⟩) r h

/--
The view of a bag of results without drops: the results sorted, and the die.

# Parameters
- `die`: The die.
- `M`: The results.

# Returns
The view.
-/
def bagView (die : Die) (M : Multiset Int) : List Int × Die :=
  (M.sort (· ≤ ·), die)

/--
A reroll of kept results, viewed, is the view of the multiset of results that
it leaves.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.
-/
theorem map_view_rerollKept (die : Die) (kept : List Int)
    (rerolled : Int → Bool) (replacement : PMF Int) :
    (rerollKept die kept rerolled replacement).map Record.view =
      (bagDraws (fun x => if rerolled x then replacement else PMF.pure x)
        kept).map (bagView die) := by
  rw [← map_bag_rerollKept, PMF.map_comp]
  refine map_congr_support fun r hr => ?_
  obtain ⟨hl, hh, hd⟩ := mem_support_rerollKept hr
  rw [Record.view_of_undropped hl hh, hd]
  simp [bagView, Record.sorted]

/-- A record of sorted results, without drops, views them. -/
private theorem view_sorted {xs : List Int} (h : xs.Pairwise (· ≤ ·))
    (die : Die) : ({ results := xs, die } : Record).view = (xs, die) := by
  rw [Record.view_of_undropped rfl rfl,
    Outcomes.sorted_of_pairwise (r := { results := xs, die }) h]

/--
A reroll that matches none of the kept results leaves them.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Hypotheses
- `h`: The criterion matches no kept result.
-/
theorem reroll_view_of_unmatched {r : Record} {c : Criterion}
    (limit : Option Nat) (h : ∀ x ∈ r.kept, c.matches x = false) :
    (reroll r c limit).map Record.view = PMF.pure r.view := by
  have hpure : ∀ (rerolled : Int → Bool) (replacement : PMF Int),
      (∀ x ∈ r.kept, rerolled x = true → c.matches x = true) →
      (rerollKept r.die r.kept rerolled replacement).map Record.view =
        PMF.pure r.view := fun rr rep hrr => by
    rw [rerollKept_of_pure fun x hx hr =>
        absurd (hrr x hx hr) (by simp [h x hx]),
      PMF.pure_map, view_sorted (Record.kept_pairwise r)]
    rfl
  cases limit with
  | none =>
    by_cases hw : r.die.facesWithin c
    · simp only [reroll, hw, ↓reduceDIte]
      rw [PMF.pure_map, List.filter_eq_self.mpr fun x hx => by simp [h x hx],
        view_sorted (Record.kept_pairwise r)]
      rfl
    · simp only [reroll, hw]
      exact hpure _ _ fun _ _ h => h
  | some n =>
    simp only [reroll]
    exact hpure _ _ fun _ _ h => by
      simp only [Bool.and_eq_true] at h
      exact h.2

/--
A reroll limited to no times leaves the kept results.

# Parameters
- `r`: The record.
- `c`: The criterion.
-/
theorem reroll_view_of_zero (r : Record) (c : Criterion) :
    (reroll r c (some 0)).map Record.view = PMF.pure r.view := by
  simp only [reroll]
  rw [rerollKept_of_pure fun _ _ h => by simp at h, PMF.pure_map,
    view_sorted (Record.kept_pairwise r)]
  rfl

/-- Redrawing from a die of one face, from that face, leaves it. -/
private theorem repeat_redraw_pure (c : Criterion) (y : Int) :
    ∀ k : Nat, k.repeat (redraw c (PMF.pure y)) (PMF.pure y) = PMF.pure y
  | 0 => rfl
  | k + 1 => by
    rw [Nat.repeat, repeat_redraw_pure c y k, redraw, PMF.pure_bind]
    split <;> rfl

/--
A limited reroll of a die of one face, whose kept results are that face,
leaves them.

# Parameters
- `n`: The limit.

# Hypotheses
- `hy`: The die has one face.
- `hk`: Every kept result is a face of the die.
-/
theorem reroll_view_of_one_face {r : Record} {c : Criterion} {y : Int}
    (n : Nat) (hy : r.die.faces = [y])
    (hk : ∀ x ∈ r.kept, x ∈ r.die.faces) :
    (reroll r c (some n)).map Record.view = PMF.pure r.view := by
  simp only [reroll]
  rw [rerollKept_of_pure fun x hx _ => by
      rw [hy, customDie_singleton, limitedReplacement, repeat_redraw_pure]
      have := hk x hx
      rw [hy, List.mem_singleton] at this
      rw [this],
    PMF.pure_map, view_sorted (Record.kept_pairwise r)]
  rfl

/--
An unlimited reroll that matches every face of the die crosses out every kept
result, if each is a face.

# Hypotheses
- `hw`: The criterion matches every face of the die.
- `hk`: Every kept result is a face of the die.
-/
theorem reroll_view_of_crossOut {r : Record} {c : Criterion}
    (hw : r.die.facesWithin c) (hk : ∀ x ∈ r.kept, x ∈ r.die.faces) :
    (reroll r c none).map Record.view = PMF.pure ([], r.die) := by
  simp only [reroll, hw, ↓reduceDIte]
  rw [PMF.pure_map]
  have : r.kept.filter (!c.matches ·) = [] := by
    refine List.filter_eq_nil_iff.mpr fun x hx => ?_
    have := List.all_eq_true.mp hw x (hk x hx)
    simp [this]
  rw [this, view_sorted List.Pairwise.nil]

/--
A reroll limited to a positive number of times, from a die that has faces,
rerolls each kept result that it matches by the `limitedLaw`.

# Parameters
- `r`: The record.
- `c`: The criterion.

# Hypotheses
- `hn`: The limit is positive.
- `hF`: The die has faces.
-/
theorem reroll_some {r : Record} (c : Criterion) {n : Nat} (hn : n ≠ 0)
    (hF : r.die.faces ≠ []) :
    reroll r c (some n) =
      rerollKept r.die r.kept c.matches
        (weightedDie (limitedLaw r.die c n)) := by
  simp only [reroll]
  rw [weightedDie_limitedLaw c n hF]
  congr 1
  funext x
  simp [Nat.pos_of_ne_zero hn]

/--
An unlimited reroll that does not match every face rerolls each kept result
that it matches by the `unlimitedLaw`.

# Parameters
- `r`: The record.
- `c`: The criterion.

# Hypotheses
- `hw`: The criterion does not match every face.
-/
theorem reroll_none {r : Record} {c : Criterion}
    (hw : ¬r.die.facesWithin c = true) :
    reroll r c none =
      rerollKept r.die r.kept c.matches
        (weightedDie (unlimitedLaw r.die c)) := by
  simp only [reroll, hw]
  rw [weightedDie_unlimitedLaw hw]
  rfl

/-- The faces of a die that has them, tallied, are not empty. -/
private theorem expand_tally_ne_nil {xs : List Int} (h : xs ≠ []) :
    Roll.expand (tally xs) ≠ [] := fun he =>
  h (List.perm_nil.mp (he ▸ (expand_tally_perm xs).symm))

/-! ## The laws of a setup's reroll -/

namespace Setup

/--
Every record that a setup describes holds its drops and its die.

# Parameters
- `s`: The setup.
- `r`: The record.

# Hypotheses
- `h`: The setup describes `r`.
-/
theorem mem_support_law_die {s : Setup} {r : Record} (h : r ∈ s.law.support) :
    r.die = s.replacement ∧ r.lowest = s.lowest ∧
      r.highest = s.highest := by
  rw [law, PMF.support_map] at h
  obtain ⟨_, _, rfl⟩ := h
  exact ⟨rfl, rfl, rfl⟩

/-- A reroll that leaves a setup alone, and every record that it describes as
viewed, rerolls as specified. -/
private theorem rerolls_of_stays {s : Setup} {c : Criterion}
    {limit : Option Nat} (hs : s.reroll c limit = s)
    (h : ∀ r ∈ s.law.support,
      (Spec.reroll r c limit).map Record.view = PMF.pure r.view) :
    s.Rerolls c limit := by
  rw [Rerolls, hs, PMF.map_bind, bind_congr_support h]
  exact (PMF.bind_pure_comp _ _).symm

/-- A custom die of faces lists them. -/
private theorem faces_custom {xs : List Int} (h : xs ≠ []) :
    (Die.custom xs).faces = xs := by
  cases xs with
  | nil => exact absurd rfl h
  | cons x xs => rfl

/-- The faces of a range are not empty. -/
private theorem faces_range_ne_nil (start stop : Int) :
    (Die.range start stop).faces ≠ [] := by
  simp only [Die.faces]
  split
  · simp
  · simp only [ne_eq, List.map_eq_nil_iff, List.range_eq_nil]
    omega

/-- A setup of alike dice that keeps some result has dice. -/
private theorem dice_pos {s : Setup} {ρ : Roll} (hρ : s.roll = some ρ)
    (halike : ρ.alike) (hk : ¬s.results - s.lowest - s.highest = 0) :
    0 < ρ.dice.1 := by
  simp only [results, hρ, Option.map_some, Option.getD_some] at hk
  cases ρ with
  | range => simp [Roll.dice]
  | standard count faces =>
    simp only [Roll.results] at hk
    simp only [Roll.dice]
    omega
  | custom count faces =>
    simp only [Roll.results] at hk
    simp only [Roll.dice]
    omega
  | fixed => simp [Roll.alike] at halike
  | pool => simp [Roll.alike] at halike

/-- Alike dice, if there are any, have faces. -/
private theorem expand_dice_ne_nil {ρ : Roll} (halike : ρ.alike)
    (hfaced : ρ.Faced) (hn : 0 < ρ.dice.1) :
    Roll.expand ρ.dice.2 ≠ [] := by
  cases ρ with
  | range start stop => exact expand_tally_ne_nil (faces_range_ne_nil _ _)
  | standard count faces => exact expand_tally_ne_nil (faces_range_ne_nil _ _)
  | custom count faces =>
    rcases hfaced with h | h
    · simp only [Roll.dice] at hn
      omega
    · exact h
  | fixed => simp [Roll.alike] at halike
  | pool => simp [Roll.alike] at halike

/--
If a reroll may change the dice still kept of a setup, the specification
rerolls each record that it describes by the final results of a matched die.

# Hypotheses
- `he`: The reroll may change the dice still kept.
-/
theorem reroll_of_redraw {s : Setup} {c : Criterion} {limit : Option Nat}
    {G : Nat} {g : List (Int × Nat)}
    (he : Effect.resolve s.replacement c limit = .redraw G g) :
    ∀ r ∈ s.law.support, Spec.reroll r c limit =
      rerollKept r.die r.kept c.matches (weightedDie g) := by
  obtain ⟨hm, hcase⟩ := Effect.resolve_redraw he
  have hF : s.replacement.faces ≠ [] := fun h => hm (by simp [h])
  intro r hr
  have hdie := (mem_support_law_die hr).1
  rcases hcase with ⟨n, rfl, hn, rfl, rfl⟩ | ⟨rfl, hmF, rfl, rfl⟩
  · rw [reroll_some c hn (by rw [hdie]; exact hF), hdie]
  · have hw : ¬r.die.facesWithin c = true := by
      rw [hdie, Die.facesWithin, List.all_eq_true]
      exact fun h => hmF (List.countP_eq_length.mpr h)
    rw [reroll_none hw, hdie]

/--
A reroll that pools a well-formed setup describes the specification's reroll
of the records that it describes, as every later instruction views them.
Mirrors `Roll::pool` in `propagation/record.rs`: the specification rerolls
the kept results of each record in place, and the multiset that it leaves,
by `bagDraws_kept`, is the survivors of the record's outcome joined with each
outcome of the replacements, whose weights `weighted_flatMap` gathers, since
every outcome weighs `Gᴷ` in all.

# Hypotheses
- `hs`: The setup is well formed.
- `hp`: The reroll pools the setup.
-/
theorem rerolls_of_pools {s : Setup} {c : Criterion} {limit : Option Nat}
    (hs : s.WellFormed) (hp : s.Pools c limit) : s.Rerolls c limit := by
  obtain ⟨ρ, G, g, hρ, hk, he, hal⟩ := hp
  have hρw := hs ρ hρ
  obtain ⟨hglen, hG, _⟩ := redraw_law he
  have hgw := wellFormed_of_redraw he
  set D := s.replacement with hD
  set L := s.lowest
  set H := s.highest
  set K := ρ.results - L - H
  let step : Int → PMF Int := fun x =>
    if c.matches x then weightedDie g else PMF.pure x
  let F : List Int → PMF (List Int × Die) := fun ys =>
    (bagDraws step ((ys.drop L).take (ys.length - L - H))).map (bagView D)
  -- The specification's reroll, viewed.
  have hrhs : (s.law.bind fun r => Spec.reroll r c limit).map Record.view =
      (weighted [] ρ.outcomes).bind fun rs => F (Roll.expand rs) := by
    rw [PMF.map_bind, bind_congr_support (g := fun r =>
      (bagDraws step r.kept).map (bagView D)) fun r hr => by
        rw [reroll_of_redraw he r hr, map_view_rerollKept,
          (mem_support_law_die hr).1]]
    have hlaw : s.law = ρ.law.map fun r =>
        { r with lowest := L, highest := H, die := D } := by
      simp [law, hρ, hD, L, H]
    rw [hlaw, PMF.bind_map]
    have : ∀ r : Record,
        (bagDraws step ({ r with lowest := L, highest := H, die := D } :
          Record).kept).map (bagView D) = F r.sorted := fun r => by
      simp only [F, Record.kept, Record.sorted, List.length_mergeSort]
    simp only [Function.comp_def, this]
    rw [show (ρ.law.bind fun r => F r.sorted) =
        (ρ.law.map Record.sorted).bind F from
          (PMF.bind_map ρ.law Record.sorted F).symm,
      ρ.map_sorted_law_weighted hρw, PMF.bind_map]
    rfl
  -- The pool, as `weighted_flatMap` reads it.
  let sv : List (Int × Nat) → List (Int × Nat) × Nat := fun rs =>
    survivors rs L H c
  let hh : List (Int × Nat) → List (List (Int × Nat) × Nat) := fun rs =>
    (Roll.custom ((sv rs).2 : Int) g).outcomes
  let P := ρ.outcomes.flatMap fun e => (hh e.1).map fun o =>
    (join (sv e.1).1 o.1, e.2 * G ^ (K - (sv e.1).2) * o.2)
  have hs' : s.reroll c limit = { roll := some (.pool P), die := some D } := by
    have hk' : ¬s.results - s.lowest - s.highest = 0 := hk
    have he' : Effect.resolve s.replacement c limit = .redraw G g := he
    have hal' : ¬(ρ.alike = true ∧ s.lowest = 0 ∧ s.highest = 0) := hal
    simp only [reroll, hρ, hk', he', hal', ↓reduceIte]
    rfl
  -- Every outcome weighs `Gᴷ` in all.
  have hc : ∀ e ∈ ρ.outcomes, e.2 ≠ 0 →
      G ^ (K - (sv e.1).2) * ((hh e.1).map (·.2)).sum = G ^ K :=
    fun e he0 _ => sum_replacements hglen hG hgw (by
      have := survivors_count hρw he0 L H c
      show (survivors e.1 L H c).2 ≤ ρ.results - L - H
      omega)
  rw [Rerolls, hrhs, hs']
  simp only [law, Roll.law, Option.map_some, Option.getD_some, replacement,
    PMF.map_comp, P]
  rw [weighted_flatMap _ [] (fun _ => []) ρ.outcomes hh
      (fun a => G ^ (K - (sv a).2)) (fun a o => join (sv a).1 o) (G ^ K)
      (pow_ne_zero _ hG)
      ((ρ.sum_outcomes hρw).1 ▸ (ρ.sum_outcomes hρw).2) hc,
    PMF.map_bind]
  refine congrArg _ (funext fun rs => ?_)
  simp only [F, step, bagDraws_kept rs L H c hgw, PMF.map_comp]
  refine map_congr_support fun o _ => ?_
  simp only [Function.comp_apply]
  rw [Record.view_of_undropped rfl rfl]
  simp [bagView, Record.sorted, sv]

/--
A reroll of a setup describes the specification's reroll of the records that
it describes, as every later instruction views them, whether it does nothing,
crosses out every die still kept, rewrites the dice, or pools them
(`rerolls_of_pools`). Mirrors `Setup::reroll` in `propagation/mixture.rs`.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Hypotheses
- `hs`: The setup is well formed.
- `hfaith`: Every result of the setup is a face of its die.
- `hfaced`: Every die of the setup has a face.
-/
theorem law_reroll {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.WellFormed) (hfaith : s.Faithful) (hfaced : s.Faced) :
    s.Rerolls c limit := by
  have hev : s.Even := fun ρ h => (hs ρ h).even
  have hkf : ∀ r ∈ s.law.support, ∀ x ∈ r.kept,
      x ∈ s.replacement.faces :=
    fun r hr x hx => hfaith r hr x (by
      have := List.mem_of_mem_drop (List.mem_of_mem_take hx)
      exact (List.mergeSort_perm _ _).mem_iff.mp this)
  have hdie := fun r (hr : r ∈ s.law.support) => (mem_support_law_die hr).1
  have hempty : s.results - s.lowest - s.highest = 0 →
      ∀ r ∈ s.law.support, r.kept = [] := fun h r hr => by
    obtain ⟨hlen, hl, hh⟩ := mem_support_law hev hr
    have := Record.length_kept r
    rw [hlen, hl, hh, h] at this
    exact List.eq_nil_of_length_eq_zero this
  cases hρ : s.roll with
  | none =>
    have hk : s.results - s.lowest - s.highest = 0 := by simp [results, hρ]
    exact rerolls_of_stays (by simp [reroll, hρ]) fun r hr =>
      reroll_view_of_unmatched limit fun x hx => by
        rw [hempty hk r hr] at hx
        cases hx
  | some ρ =>
    by_cases hk : s.results - s.lowest - s.highest = 0
    · exact rerolls_of_stays (by simp [reroll, hρ, hk]) fun r hr =>
        reroll_view_of_unmatched limit fun x hx => by
          rw [hempty hk r hr] at hx
          cases hx
    cases he : Effect.resolve s.replacement c limit with
    | nothing =>
      refine rerolls_of_stays (by simp [reroll, hρ, hk, he]) fun r hr => ?_
      rcases Effect.resolve_nothing he with hm | rfl | ⟨n, rfl, hF⟩
      · exact reroll_view_of_unmatched limit fun x hx => by
          have := List.countP_eq_zero.mp hm x (hkf r hr x hx)
          simpa using this
      · exact reroll_view_of_zero r c
      · obtain ⟨y, hy⟩ := List.length_eq_one_iff.mp hF
        exact reroll_view_of_one_face n (by rw [hdie r hr, hy])
          fun x hx => by rw [hdie r hr]; exact hkf r hr x hx
    | crossOut =>
      obtain ⟨rfl, hm⟩ := Effect.resolve_crossOut he
      have hw : s.replacement.facesWithin c := by
        rw [Die.facesWithin, List.all_eq_true]
        exact List.countP_eq_length.mp hm
      have hs' : s.reroll c none = { s with lowest := s.results } := by
        simp [reroll, hρ, hk, he]
      rw [Rerolls, hs', PMF.map_bind]
      rw [bind_congr_support fun r hr => (hdie r hr) ▸
        reroll_view_of_crossOut ((hdie r hr) ▸ hw)
          fun x hx => (hdie r hr) ▸ hkf r hr x hx]
      rw [PMF.bind_const, map_congr_support (g := fun _ => ([], s.replacement))
        fun r hr => ?_]
      · exact PMF.map_const _ _
      obtain ⟨hd, hl, _⟩ := mem_support_law_die hr
      obtain ⟨hlen, _, _⟩ := mem_support_law
        (s := { s with lowest := s.results }) (fun ρ' h => hev ρ' h) hr
      have := Record.length_kept r
      rw [hlen, hl] at this
      simp only [Record.view, hd, List.eq_nil_of_length_eq_zero
        (by rw [this]; simp [results])]
      rfl
    | redraw G g =>
      by_cases hal : ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0
      · obtain ⟨halike, hl0, hh0⟩ := hal
        -- The law of a matched die, its padding, and the specification's
        -- reroll of each record by it.
        obtain ⟨hglen, hG, _⟩ := redraw_law he
        have hreroll := reroll_of_redraw he
        set D := s.replacement with hD
        let step : Int → PMF Int := fun x =>
          if c.matches x then weightedDie g else PMF.pure x
        -- The dice have faces.
        have hf := expand_dice_ne_nil halike (hfaced ρ hρ)
          (dice_pos hρ halike hk)
        have hlawρ : s.law = ρ.law.map fun r =>
            { r with lowest := s.lowest, highest := s.highest, die := D } := by
          simp [law, hρ, hD]
        rw [Rerolls, PMF.map_bind, bind_congr_support
          (g := fun r => (bagDraws step r.results).map (bagView D))
          fun r hr => by
          rw [hreroll r hr, map_view_rerollKept]
          have hkr : r.kept.Perm r.results := by
            obtain ⟨_, hl, hh⟩ := mem_support_law_die hr
            have := Record.view_of_undropped (hl.trans hl0) (hh.trans hh0)
            rw [show r.kept = r.sorted from (Prod.mk.inj this).1]
            exact List.mergeSort_perm _ _
          rw [bagDraws_perm _ hkr, hdie r hr]]
        rw [hlawρ, PMF.bind_map]
        show _ = ρ.law.bind fun r => (bagDraws step r.results).map (bagView D)
        have hb : (ρ.law.bind fun r => bagDraws step r.results) =
            (rollDice ρ.dice.1 (weightedDie ρ.dice.2)).bind
              fun r => bagDraws step r.results :=
          calc _ = (ρ.law.map (·.results)).bind (bagDraws step) :=
                (PMF.bind_map _ _ _).symm
            _ = _ := by rw [Roll.map_results_law halike, PMF.bind_map]; rfl
        rw [← PMF.map_bind, hb, bind_rollDice_bagDraws,
          ← weightedDie_rewriteFaces c hf hglen hG, PMF.map_comp]
        have hs' : s.reroll c limit =
            { roll := some (ρ.rewrite c G g), die := some D } := by
          have he' : Effect.resolve s.replacement c limit = .redraw G g := he
          have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl0, hh0] at hk
          simp only [reroll, hρ, he', halike, hl0, hh0, and_self, hD, hk',
            ↓reduceIte]
        rw [hs']
        simp only [law, Option.map_some, Option.getD_some, Roll.rewrite,
          Roll.law, PMF.map_comp, replacement, Option.getD_some]
        refine map_congr_support fun r hr => ?_
        obtain ⟨hl, hh, _, _⟩ := mem_support_rollDice hr
        simp only [Function.comp_apply]
        rw [Record.view_of_undropped rfl rfl]
        simp [bagView, Record.sorted]
      · exact rerolls_of_pools hs ⟨ρ, G, g, hρ, hk, he, hal⟩

end Setup

/-! ## Mixtures -/

/--
Rerolling every setup of a mixture by every resolution of the reroll's
operands describes, as every later instruction views it, the specification's
reroll of the mixture's law, independent of the resolutions, as
`Mixture::reroll` in `propagation/mixture.rs` rerolls a record whose operands
are independent of it (`law_reroll_resolved`).

# Parameters
- `μ`: The law of the setups.
- `κ`: The law of the criterion and the limit, as the operands resolve them.

# Hypotheses
- `hμ`: Every setup is well formed, faithful to its die, and has faces.
-/
theorem mixtureLaw_reroll {μ : PMF Setup} {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.WellFormed ∧ s.Faithful ∧ s.Faced) :
    (mixtureLaw (lift (fun s q => s.reroll q.1 q.2) μ κ)).map Record.view =
      ((lift Prod.mk (mixtureLaw μ) κ).bind fun p =>
        reroll p.1 p.2.1 p.2.2).map Record.view := by
  simp only [mixtureLaw, lift, PMF.bind_bind, PMF.bind_map, PMF.map_bind,
    Function.comp_def]
  refine bind_congr_support fun s hs => ?_
  obtain ⟨he, hf, hc⟩ := hμ s hs
  refine (bind_congr_support (p := κ)
    (f := fun q : Criterion × Option Nat =>
      (s.reroll q.1 q.2).law.map Record.view)
    (g := fun q : Criterion × Option Nat =>
      (s.law.bind fun r => reroll r q.1 q.2).map Record.view)
    fun q hq => by
      have h := Setup.law_reroll q.1 q.2 he hf hc
      rw [Setup.Rerolls] at h
      exact h).trans ?_
  simp only [PMF.map_bind]
  exact PMF.bind_comm _ _ _

/--
The law that `law_reroll` in `XdySpec/World.lean` gives a rerolled record is
the reroll of the record's law, independent of the criterion and the limit to
which the joint law of the operands resolves, as `visit_reroll` in
`propagation.rs` resolves each joint outcome before `Mixture::reroll` rerolls
by it.

# Parameters
- `ρ`: The law of the record.
- `ν`: The joint law of the operands.
- `p`: The predicate.
- `l`: The limit.
-/
theorem law_reroll_resolved (ρ : PMF Record)
    (ν : PMF (AddressingMode → Int)) (p : Predicate) (l : Limit) :
    (lift Prod.mk ρ ν).bind
        (fun q => reroll q.1 (p.resolve q.2) (l.resolve q.2)) =
      (lift Prod.mk ρ (ν.map fun v => (p.resolve v, l.resolve v))).bind
        fun q => reroll q.1 q.2.1 q.2.2 := by
  simp only [lift, PMF.bind_bind, PMF.bind_map, PMF.map_comp, Function.comp_def]

/-! ## Faithful setups -/

/--
A custom die shows only its faces, or `0` if it has none.

# Parameters
- `xs`: The faces.
- `x`: A value that it shows.

# Hypotheses
- `h`: The die can show `x`.
-/
theorem mem_of_mem_support_customDie {xs : List Int} {x : Int}
    (h : x ∈ (customDie xs).support) : x ∈ xs ∨ (xs = [] ∧ x = 0) := by
  unfold customDie at h
  split at h
  · rename_i he
    exact .inr ⟨he, (PMF.mem_support_pure_iff _ _).mp h⟩
  · simp only [PMF.mem_support_ofMultiset_iff, Multiset.mem_toFinset,
      Multiset.mem_coe] at h
    exact .inl h

/--
A custom die can show each of its faces.

# Parameters
- `xs`: The faces.
- `x`: A face.

# Hypotheses
- `h`: `x` is a face.
-/
theorem mem_support_customDie {xs : List Int} {x : Int} (h : x ∈ xs) :
    x ∈ (customDie xs).support := by
  have hne : xs ≠ [] := List.ne_nil_of_mem h
  unfold customDie
  simp only [hne, ↓reduceDIte, PMF.mem_support_ofMultiset_iff,
    Multiset.mem_toFinset, Multiset.mem_coe]
  exact h

/-- Dice of a law can all show any one value that it can. -/
private theorem replicate_mem_support_rollDice {P : PMF Int} {x : Int}
    (hx : x ∈ P.support) :
    ∀ k : Nat, ({ results := List.replicate k x } : Record) ∈
      (rollDice (k : Int) P).support
  | 0 => by simp [rollDice, Nat.repeat]
  | k + 1 => by
    rw [OrderStatistics.rollDice_succ, PMF.mem_support_bind_iff]
    refine ⟨_, replicate_mem_support_rollDice hx k, ?_⟩
    rw [PMF.support_map]
    exact ⟨x, hx, by simp [Record.push, List.replicate_succ']⟩

/-- Every result of a roll whose dice are alike is a face of its dice, or `0`
if they have none. -/
private theorem mem_results_alike {ρ : Roll} (h : ρ.alike) {r : Record}
    (hr : r ∈ ρ.law.support) {x : Int} (hx : x ∈ r.results) :
    x ∈ Roll.expand ρ.dice.2 ∨ (Roll.expand ρ.dice.2 = [] ∧ x = 0) := by
  have hm : r.results ∈ ((rollDice ρ.dice.1 (weightedDie ρ.dice.2)).map
      (·.results)).support := by
    rw [← Roll.map_results_law h, PMF.support_map]
    exact ⟨r, hr, rfl⟩
  rw [PMF.support_map] at hm
  obtain ⟨r', hr', he⟩ := hm
  have he' : r'.results = r.results := he
  have := (mem_support_rollDice hr').2.2.2 x (he' ▸ hx)
  exact mem_of_mem_support_customDie this

/-- Every face of the dice of a roll whose dice are alike, if there are any,
is a result of some record that the roll fills. -/
private theorem exists_results_alike {ρ : Roll} (h : ρ.alike)
    (hn : 0 < ρ.dice.1) {x : Int} (hx : x ∈ Roll.expand ρ.dice.2) :
    ∃ r ∈ ρ.law.support, x ∈ r.results := by
  have hm := replicate_mem_support_rollDice
    (P := weightedDie ρ.dice.2) (mem_support_customDie hx) ρ.dice.1.toNat
  rw [← OrderStatistics.rollDice_toNat] at hm
  have hm' : List.replicate ρ.dice.1.toNat x ∈
      (ρ.law.map (·.results)).support := by
    rw [Roll.map_results_law h, PMF.support_map]
    exact ⟨_, hm, rfl⟩
  rw [PMF.support_map] at hm'
  obtain ⟨r, hr, he⟩ := hm'
  have he' : r.results = List.replicate ρ.dice.1.toNat x := he
  exact ⟨r, hr, he' ▸ List.mem_replicate.mpr ⟨by omega, rfl⟩⟩

namespace Setup

/--
A setup is faithful to its die exactly when every result of its roll is a face
of the die: its drops play no part.

# Parameters
- `s`, `s'`: The setups.

# Hypotheses
- `hr`: They hold the same roll.
- `hd`: They hold the same die.
- `h`: `s` is faithful to its die.
-/
theorem Faithful.of_eq {s s' : Setup} (hr : s.roll = s'.roll)
    (hd : s.die = s'.die) (h : s.Faithful) : s'.Faithful := by
  intro r hr' x hx
  rw [law, PMF.support_map] at hr'
  obtain ⟨r₀, h₀, rfl⟩ := hr'
  have hrep : s.replacement = s'.replacement := by simp [replacement, hr, hd]
  rw [← hrep]
  refine h ⟨r₀.results, s.lowest, s.highest, s.replacement⟩ ?_ x hx
  rw [law, PMF.support_map, hr]
  exact ⟨r₀, h₀, rfl⟩

/--
A setup of a roll whose dice are alike, without drops, is faithful to its
die, as every setup that `Mixture::rolls` makes is.

# Parameters
- `ρ`: The roll.

# Hypotheses
- `h`: The roll's dice are alike.
-/
theorem faithful_roll {ρ : Roll} (h : ρ.alike) :
    ({ roll := some ρ } : Setup).Faithful := by
  intro r hr x hx
  rw [law_roll] at hr
  have hrep : ({ roll := some ρ } : Setup).replacement = ρ.replacement := rfl
  rw [hrep]
  rcases mem_results_alike h hr hx with hm | ⟨he, rfl⟩
  · cases ρ with
    | range start stop =>
      exact (expand_tally_perm _).mem_iff.mp hm
    | standard count faces =>
      exact (expand_tally_perm _).mem_iff.mp hm
    | custom count faces =>
      simp only [Roll.dice] at hm
      have hne : (Roll.expand faces).mergeSort (· ≤ ·) ≠ [] := fun he =>
        List.ne_nil_of_mem hm (List.perm_nil.mp (he ▸ (List.mergeSort_perm
          (Roll.expand faces) (· ≤ ·)).symm))
      simp only [Roll.replacement, Die.ofFaces]
      rw [faces_custom hne]
      exact (List.mergeSort_perm _ _).mem_iff.mpr hm
    | fixed => simp [Roll.alike] at h
    | pool => simp [Roll.alike] at h
  · cases ρ with
    | range start stop =>
      exact absurd he (expand_tally_ne_nil (faces_range_ne_nil _ _))
    | standard count faces =>
      exact absurd he (expand_tally_ne_nil (faces_range_ne_nil _ _))
    | custom count faces =>
      simp only [Roll.dice] at he
      simp [Roll.replacement, Die.ofFaces, Die.faces, he]
    | fixed => simp [Roll.alike] at h
    | pool => simp [Roll.alike] at h

end Setup

namespace Setup

/-- The pool that a reroll leaves of a setup faithful to its die is faithful
to it too: each survivor is a result of the setup, and each replacement is a
final result of a matched die, which is a face. -/
private theorem faithful_rerollPool {s : Setup} {ρ : Roll} {c : Criterion}
    {limit : Option Nat} {G : Nat} {g : List (Int × Nat)}
    (hρ : s.roll = some ρ) (hs : s.WellFormed) (hfaith : s.Faithful)
    (he : Effect.resolve s.replacement c limit = .redraw G g) :
    ({ roll := some (ρ.rerollPool s.lowest s.highest c G g),
       die := some s.replacement } : Setup).Faithful := by
  have hρw := hs ρ hρ
  obtain ⟨hglen, hG, hgf⟩ := redraw_law he
  have hgw := wellFormed_of_redraw he
  have hgne : Roll.expand g ≠ [] := fun h => hG (by rw [← hglen, h]; rfl)
  -- Every multiset of the pool holds only faces of the die.
  have hfaces : ∀ e ∈ (ρ.rerollPool s.lowest s.highest c G g).outcomes,
      ∀ x ∈ Roll.expand e.1, x ∈ s.replacement.faces := by
    intro e he x hx
    simp only [Roll.rerollPool, Roll.outcomes, List.mem_flatMap,
      List.mem_map] at he
    obtain ⟨a, ha, o, ho, rfl⟩ := he
    have ho' : o ∈ (Roll.custom
        ((survivors a.1 s.lowest s.highest c).2 : Int) g).outcomes := ho
    rcases List.mem_append.mp ((perm_expand_join _ _).mem_iff.mp hx) with
      hx | hx
    · -- A survivor is a kept result of the setup.
      rw [(survivors_spec a.1 s.lowest s.highest c).1] at hx
      have hx := List.mem_of_mem_drop (List.mem_of_mem_take
        (List.mem_of_mem_filter hx))
      obtain ⟨r, hr, hre⟩ := Roll.exists_sorted_of_mem_outcomes hρw ha
      rw [← hre, Record.sorted] at hx
      refine hfaith ⟨r.results, s.lowest, s.highest, s.replacement⟩ ?_ x
        ((List.mergeSort_perm _ _).mem_iff.mp hx)
      rw [law, PMF.support_map, hρ]
      exact ⟨r, hr, rfl⟩
    · -- A replacement is a final result of a matched die.
      obtain ⟨r, hr, hre⟩ := Roll.exists_sorted_of_mem_outcomes (hgw _) ho'
      rw [← hre, Record.sorted] at hx
      simp only [Roll.law, PMF.support_map, Set.mem_image] at hr
      obtain ⟨r', hr', rfl⟩ := hr
      have := (mem_support_rollDice hr').2.2.2 x
        ((List.mergeSort_perm _ _).mem_iff.mp hx)
      rcases mem_of_mem_support_customDie this with h | ⟨h, _⟩
      · exact hgf x h
      · exact absurd h hgne
  intro r hr x hx
  simp only [law, Option.map_some, Option.getD_some, Roll.rerollPool,
    Roll.law, PMF.map_comp, PMF.support_map, Set.mem_image] at hr
  obtain ⟨rs, hrs, rfl⟩ := hr
  simp only [Function.comp_apply] at hx
  simp only [replacement, Option.getD_some]
  rcases mem_support_weighted hrs with ⟨e, he, rfl⟩ | rfl
  · exact hfaces e he x hx
  · generalize hP : (ρ.rerollPool s.lowest s.highest c G g).outcomes = P
      at hfaces
    change x ∈ Roll.expand
      (((ρ.rerollPool s.lowest s.highest c G g).outcomes.head?.map
        (·.1)).getD []) at hx
    rw [hP] at hx
    cases P with
    | nil => simp [Roll.expand] at hx
    | cons e P => exact hfaces e List.mem_cons_self x hx

/--
A reroll leaves a setup well formed, faithful to its die, and with faces, so
that the law of a later reroll holds of it too (`law_reroll`): a rewrite
tallies the faces of its dice (`tally_wellFormed`), and a pool holds
multisets of the results still kept (`Roll.wellFormed_rerollPool`), each a
survivor or a final result of a matched die.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Hypotheses
- `hs`: The setup is well formed.
- `hfaith`: Every result of the setup is a face of its die.
- `hfaced`: Every die of the setup has a face.
-/
theorem reroll_invariants {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.WellFormed) (hfaith : s.Faithful) (hfaced : s.Faced) :
    (s.reroll c limit).WellFormed ∧ (s.reroll c limit).Faithful ∧
      (s.reroll c limit).Faced := by
  cases hρ : s.roll with
  | none =>
    rw [show s.reroll c limit = s by simp [reroll, hρ]]
    exact ⟨hs, hfaith, hfaced⟩
  | some ρ =>
    by_cases hk : s.results - s.lowest - s.highest = 0
    · rw [show s.reroll c limit = s by simp [reroll, hρ, hk]]
      exact ⟨hs, hfaith, hfaced⟩
    cases he : Effect.resolve s.replacement c limit with
    | nothing =>
      rw [show s.reroll c limit = s by simp [reroll, hρ, hk, he]]
      exact ⟨hs, hfaith, hfaced⟩
    | crossOut =>
      rw [show s.reroll c limit = { s with lowest := s.results } by
        simp [reroll, hρ, hk, he]]
      exact ⟨hs, Faithful.of_eq (s := s)
        (s' := { s with lowest := s.results }) rfl rfl hfaith, hfaced⟩
    | redraw G g =>
      obtain ⟨hglen, hG, hgf⟩ := redraw_law he
      by_cases hal : ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0
      swap
      · have hs' : s.reroll c limit =
            { roll := some (ρ.rerollPool s.lowest s.highest c G g),
              die := some s.replacement } := by
          simp only [reroll, hρ, hk, he, hal, ↓reduceIte]
        rw [hs']
        exact ⟨fun ρ' h => by
            cases h
            exact Roll.wellFormed_rerollPool (hs ρ hρ) hG
              (wellFormed_of_redraw he),
          faithful_rerollPool hρ hs hfaith he, fun ρ' h => by cases h; trivial⟩
      obtain ⟨halike, hl0, hh0⟩ := hal
      have hs' : s.reroll c limit =
          { roll := some (ρ.rewrite c G g), die := some s.replacement } := by
        have hk' : ¬s.results - 0 - 0 = 0 := by rwa [hl0, hh0] at hk
        simp only [reroll, hρ, he, halike, hl0, hh0, and_self, hk',
          ↓reduceIte]
      have hn := dice_pos hρ halike hk
      -- Every face of the dice is a face of the die.
      have hff : ∀ x ∈ Roll.expand ρ.dice.2, x ∈ s.replacement.faces :=
        fun x hx => by
          obtain ⟨r, hr, hxr⟩ := exists_results_alike halike hn hx
          refine hfaith ⟨r.results, s.lowest, s.highest, s.replacement⟩ ?_ x
            hxr
          rw [law, PMF.support_map, hρ]
          exact ⟨r, hr, rfl⟩
      have hmem : ∀ x ∈ Roll.expand (rewriteFaces ρ.dice.2 c G g),
          x ∈ s.replacement.faces := fun x hx => by
        rw [(expand_rewriteFaces_perm _ c G g).mem_iff, List.mem_flatMap] at hx
        obtain ⟨y, hy, hx⟩ := hx
        split at hx
        · exact hgf x hx
        · rw [List.eq_of_mem_replicate hx]
          exact hff y hy
      have hne : Roll.expand (rewriteFaces ρ.dice.2 c G g) ≠ [] := by
        obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _
          (expand_dice_ne_nil halike (hfaced ρ hρ) hn)
        have hz : (if c.matches y then Roll.expand g
            else List.replicate G y) ≠ [] := by
          split
          · intro h
            rw [h] at hglen
            exact hG hglen.symm
          · simpa using hG
        obtain ⟨z, hz⟩ := List.exists_mem_of_ne_nil _ hz
        exact List.ne_nil_of_mem ((expand_rewriteFaces_perm _ c G g).mem_iff.mpr
          (List.mem_flatMap.mpr ⟨y, hy, hz⟩))
      rw [hs']
      refine ⟨fun ρ' h => by cases h; exact tally_wellFormed _, ?_,
        fun ρ' h => ?_⟩
      · intro r hr x hx
        simp only [law, Option.map_some, Option.getD_some, Roll.rewrite,
          Roll.law, PMF.map_comp, PMF.support_map, Set.mem_image] at hr
        obtain ⟨r', hr', rfl⟩ := hr
        have := (mem_support_rollDice hr').2.2.2 x hx
        rcases mem_of_mem_support_customDie this with h | ⟨h, _⟩
        · exact hmem x h
        · exact absurd h hne
      · cases h
        exact .inr hne

end Setup

end Xdy.Spec
