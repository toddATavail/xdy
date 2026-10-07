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
  results and the die), whether it does nothing, crosses out, or rewrites
  (`Setup.law_reroll`); and so a mixture's reroll describes the reroll of the
  mixture's law, independent of the operands' (`mixtureLaw_reroll`,
  `law_reroll_resolved`). The view, not the record, since the Rust leaves a
  setup that the reroll cannot change with its drops, and crosses out by
  dropping, where the specification's record keeps only its kept results.
- The proofs ask that every setup be even, that every result be a face of its
  die, and that every die have a face (`Setup.Faithful`, `Setup.Faced`), as
  every roll (`Setup.faithful_roll`), drop and keep (`Setup.Faithful.of_eq`),
  and reroll that does not pool (`Setup.reroll_of_not_pools`) leaves them.
- Dice after a drop, or fixed results, are not alike, so the reroll pools the
  multisets of the dice still kept (`Roll.rerollPool`, `Roll::pool`), whose
  law `Setup.law_reroll` takes as a hypothesis, `Setup.Pools`.

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
A reroll of a setup describes the specification's reroll of the records that
it describes, as every later instruction views them, if it does nothing,
crosses out every die still kept, or rewrites the dice; and if it pools them,
as `hpool` grants. Mirrors `Setup::reroll` in `propagation/mixture.rs`.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Hypotheses
- `hs`: The setup is even.
- `hfaith`: Every result of the setup is a face of its die.
- `hfaced`: Every die of the setup has a face.
- `hpool`: If the reroll pools the setup, it rerolls as specified.
-/
theorem law_reroll {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.Even) (hfaith : s.Faithful) (hfaced : s.Faced)
    (hpool : s.Pools c limit → s.Rerolls c limit) : s.Rerolls c limit := by
  have hkf : ∀ r ∈ s.law.support, ∀ x ∈ r.kept,
      x ∈ s.replacement.faces :=
    fun r hr x hx => hfaith r hr x (by
      have := List.mem_of_mem_drop (List.mem_of_mem_take hx)
      exact (List.mergeSort_perm _ _).mem_iff.mp this)
  have hdie := fun r (hr : r ∈ s.law.support) => (mem_support_law_die hr).1
  have hempty : s.results - s.lowest - s.highest = 0 →
      ∀ r ∈ s.law.support, r.kept = [] := fun h r hr => by
    obtain ⟨hlen, hl, hh⟩ := mem_support_law hs hr
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
        (s := { s with lowest := s.results }) (fun ρ' h => hs ρ' h) hr
      have := Record.length_kept r
      rw [hlen, hl] at this
      simp only [Record.view, hd, List.eq_nil_of_length_eq_zero
        (by rw [this]; simp [results])]
      rfl
    | redraw G g =>
      by_cases hal : ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0
      · obtain ⟨halike, hl0, hh0⟩ := hal
        obtain ⟨hm, hcase⟩ := Effect.resolve_redraw he
        set D := s.replacement with hD
        have hF : D.faces ≠ [] := fun h => hm (by simp [h])
        have hFlen : D.faces.length ≠ 0 := by simpa using hF
        have hmle := List.countP_le_length (p := c.matches) (l := D.faces)
        -- The law of a matched die, its padding, and the specification's
        -- reroll of each record by it.
        have hlaw : (Roll.expand g).length = G ∧ G ≠ 0 ∧
            ∀ r ∈ s.law.support, Spec.reroll r c limit =
              rerollKept r.die r.kept c.matches (weightedDie g) := by
          rcases hcase with
            ⟨n, rfl, hn, rfl, rfl⟩ | ⟨rfl, hmF, rfl, rfl⟩
          · refine ⟨length_expand_limitedLaw _ _ _,
              Nat.mul_ne_zero (pow_ne_zero _ hFlen) hFlen, fun r hr => ?_⟩
            rw [reroll_some c hn (by rw [hdie r hr]; exact hF), hdie r hr]
          · refine ⟨length_expand_unlimitedLaw _ _, by omega,
              fun r hr => ?_⟩
            have hw : ¬r.die.facesWithin c = true := by
              rw [hdie r hr, Die.facesWithin, List.all_eq_true]
              exact fun h => hmF (List.countP_eq_length.mpr h)
            rw [reroll_none hw, hdie r hr]
        obtain ⟨hglen, hG, hreroll⟩ := hlaw
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
      · exact hpool ⟨ρ, G, g, hρ, hk, he, hal⟩

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
- `hμ`: Every setup is even, faithful to its die, and has faces.
- `hpool`: Every reroll that pools a setup rerolls as specified.
-/
theorem mixtureLaw_reroll {μ : PMF Setup} {κ : PMF (Criterion × Option Nat)}
    (hμ : ∀ s ∈ μ.support, s.Even ∧ s.Faithful ∧ s.Faced)
    (hpool : ∀ s ∈ μ.support, ∀ q ∈ κ.support,
      s.Pools q.1 q.2 → s.Rerolls q.1 q.2) :
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
      have h := Setup.law_reroll q.1 q.2 he hf hc (hpool s hs q hq)
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

namespace Setup

/--
A reroll that does not pool a setup leaves it even, faithful to its die, and
with faces, so that the law of a later reroll holds of it too
(`law_reroll`).

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for no
  limit.

# Hypotheses
- `hs`: The setup is even.
- `hfaith`: Every result of the setup is a face of its die.
- `hfaced`: Every die of the setup has a face.
- `hnp`: The reroll does not pool the setup.
-/
theorem reroll_of_not_pools {s : Setup} (c : Criterion) (limit : Option Nat)
    (hs : s.Even) (hfaith : s.Faithful) (hfaced : s.Faced)
    (hnp : ¬s.Pools c limit) :
    (s.reroll c limit).Even ∧ (s.reroll c limit).Faithful ∧
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
      have hal : ρ.alike ∧ s.lowest = 0 ∧ s.highest = 0 :=
        Classical.byContradiction fun hal =>
          hnp ⟨ρ, G, g, hρ, hk, he, hal⟩
      obtain ⟨halike, hl0, hh0⟩ := hal
      obtain ⟨hglen, hG, hgf⟩ := redraw_law he
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
      refine ⟨fun ρ' h => by cases h; trivial, ?_, fun ρ' h => ?_⟩
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
