import Mathlib.Analysis.SpecificLimits.Basic
import XdySpec.Reroll

/-!
# Unlimited rerolls as limits

The specification gives an unlimited reroll the conditional law of the die,
given that it does not match (`reroll` in `XdySpec/Semantics.lean`), where the
Rust, `reroll_within` in `xdy/src/primitives.rs`, loops until the die no
longer matches. This module justifies the one against the other: the loop that
stops after `n` draws is the specification's limited reroll, the unlimited
loop agrees with it on every draw before the `n`th, and the limited laws
converge to the conditional law as `n` grows.

- A limited reroll has a closed form (`limitedReplacement_apply`): limited to
  `n + 1` draws, it shows a result that the criterion matches with `qⁿ` times
  its probability, since every draw must match, and any other with
  `1 + q + ⋯ + qⁿ` times its probability, once for each draw that might first
  show it, where `q` is the probability that the die matches (`matched`),
  for a custom die the share of its faces that match (`matched_customDie`).
- The loop stops with probability one: after `n` draws, it would draw again
  only with probability `qⁿ` (`matched_limitedReplacement`), which vanishes
  if `q < 1`, and the probability of each result that it stops on only grows
  with `n` (`monotone_limitedReplacement`).
- So the limited laws converge, result by result, to the conditional law,
  `PMF.filter` (`tendsto_limitedReplacement`), which for a custom die is the
  die of the faces that do not match (`customDie_filter`), as the
  specification draws it (`tendsto_limitedReplacement_faces`). Rerolling a
  record's kept results is continuous in the law of the replacement
  (`tendsto_rerollKept`), so the limited rerolls of a record that holds no
  chains converge to its unlimited reroll (`tendsto_reroll`).
- If every face matches, then `q = 1`, and no limit gives the loop a meaning:
  every limited reroll leaves the die's law alone
  (`limitedReplacement_of_matches`, `limitedReplacement_faces_of_facesWithin`)
  and the loop never stops. By convention, the Rust and the specification
  roll nothing but cross out every kept result that matches
  (`reroll_crossOut`, `kept_crossOut`), so that each contributes zero to the
  sum (`sum_reroll_crossOut`).
-/

namespace Xdy.Spec

open Xdy (Die Criterion)
open Filter Topology
open scoped ENNReal

/-! ## The closed form of a limited reroll -/

/--
The probability that a law shows a result that a criterion matches: for the
law of a die, the probability that a reroll redraws it.

# Parameters
- `c`: The criterion.
- `p`: The law.

# Returns
The probability.

# Examples
```lean
example : matched (.within 1 1) (PMF.pure 1) = 1 := by
  rw [matched, tsum_eq_single 1 fun x hx => by simp [hx]]
  simp [Criterion.matches]
```
-/
noncomputable def matched (c : Criterion) (p : PMF Int) : ℝ≥0∞ :=
  ∑' x, if c.matches x then p x else 0

/--
A custom die matches with the share of its faces that match, by position:
`m / F`, as `redraw` in `propagation/record.rs` counts them.

# Parameters
- `c`: The criterion.
- `xs`: The faces.

# Hypotheses
- `h`: The die has faces.
-/
theorem matched_customDie (c : Criterion) {xs : List Int} (h : xs ≠ []) :
    matched c (customDie xs) =
      (xs.countP c.matches : ℝ≥0∞) / (xs.length : ℝ≥0∞) := by
  have hsum : (xs.map fun y => if c.matches y then (1 : ℝ≥0∞) else 0).sum =
      (xs.countP c.matches : ℝ≥0∞) := by
    clear h
    induction xs with
    | nil => simp
    | cons x xs ih =>
      by_cases hx : c.matches x <;> simp [hx, ih, add_comm]
  rw [← hsum, ← tsum_customDie h, matched]
  refine tsum_congr fun y => ?_
  by_cases hy : c.matches y <;> simp [hy]

/--
A redraw shows a result by redrawing it, from the probability that it matched,
or, if the criterion does not match it, by keeping it.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `p`: The law of the die's current result.
- `x`: The result.
-/
theorem redraw_apply (c : Criterion) (die p : PMF Int) (x : Int) :
    redraw c die p x =
      matched c p * die x + if c.matches x then 0 else p x := by
  have hterm : ∀ y, p y * (if c.matches y then die else PMF.pure y) x =
      (if c.matches y then p y else 0) * die x +
        if c.matches y then 0 else p y * (PMF.pure y) x := fun y => by
    split <;> simp
  have hkept : (∑' y, if c.matches y then 0 else p y * (PMF.pure y) x) =
      if c.matches x then 0 else p x := by
    rw [tsum_eq_single x fun y hy => by simp [PMF.pure_apply, Ne.symm hy]]
    simp
  rw [redraw, PMF.bind_apply, tsum_congr hterm, ENNReal.tsum_add,
    ENNReal.tsum_mul_right, hkept]
  rfl

/--
A redraw still matches only if the result matched and the redraw matches too.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `p`: The law of the die's current result.
-/
theorem matched_redraw (c : Criterion) (die p : PMF Int) :
    matched c (redraw c die p) = matched c p * matched c die := by
  have hterm : ∀ x, (if c.matches x then redraw c die p x else 0) =
      matched c p * if c.matches x then die x else 0 := fun x => by
    rw [redraw_apply]
    split <;> simp
  rw [matched, tsum_congr hterm, ENNReal.tsum_mul_left]
  rfl

/-- After `k` redraws, a die still matches with probability `qᵏ⁺¹`, and shows
a result that matches with `qᵏ` times its probability, and one that does not
with `1 + q + ⋯ + qᵏ` times its probability. -/
private theorem repeat_redraw (c : Criterion) (die : PMF Int) :
    ∀ k, matched c (k.repeat (redraw c die) die) = matched c die ^ (k + 1) ∧
      ∀ x, k.repeat (redraw c die) die x =
        if c.matches x then matched c die ^ k * die x
        else (∑ i ∈ Finset.range (k + 1), matched c die ^ i) * die x
  | 0 => ⟨by simp [Nat.repeat], fun x => by split <;> simp [Nat.repeat]⟩
  | k + 1 => by
    obtain ⟨hm, hx⟩ := repeat_redraw c die k
    refine ⟨by rw [Nat.repeat, matched_redraw, hm, ← pow_succ], fun x => ?_⟩
    rw [Nat.repeat, redraw_apply, hm, hx]
    split
    · simp [pow_succ']
    · rw [Finset.sum_range_succ _ (k + 1), add_mul, add_comm]

/--
The closed form of a limited reroll: a die rerolled up to `n + 1` times shows
a result that the criterion matches only if every draw matched, with
probability `qⁿ` times its own, and any other result with probability
`1 + q + ⋯ + qⁿ` times its own, once for each draw that might first show it,
where `q` is the probability that the die matches.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `n`: The number of redraws after the first draw.
- `x`: The result.
-/
theorem limitedReplacement_apply (c : Criterion) (die : PMF Int) (n : Nat)
    (x : Int) :
    limitedReplacement c die (n + 1) x =
      if c.matches x then matched c die ^ n * die x
      else (∑ i ∈ Finset.range (n + 1), matched c die ^ i) * die x :=
  (repeat_redraw c die n).2 x

/--
A die rerolled up to `n` times still matches with probability `qⁿ`: the
probability that the Rust's loop, `reroll_within` in `primitives.rs`, would
roll again were its limit greater.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `n`: The limit, which is positive.

# Hypotheses
- `hn`: The limit is positive.
-/
theorem matched_limitedReplacement (c : Criterion) (die : PMF Int) {n : Nat}
    (hn : n ≠ 0) :
    matched c (limitedReplacement c die n) = matched c die ^ n := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn
  exact (repeat_redraw c die k).1

/--
Raising the limit of a reroll only adds to the probability of each result that
the criterion does not match: the draws up to the old limit fall as before, and
those after it may show the result too.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `x`: A result that the criterion does not match.

# Hypotheses
- `hx`: The criterion does not match `x`.
-/
theorem monotone_limitedReplacement (c : Criterion) (die : PMF Int) {x : Int}
    (hx : c.matches x = false) :
    Monotone fun n => limitedReplacement c die n x := by
  refine monotone_nat_of_le_succ fun n => ?_
  cases n with
  | zero => exact le_rfl
  | succ n =>
    simp only [limitedReplacement_apply, hx, Bool.false_eq_true, ↓reduceIte]
    rw [Finset.sum_range_succ _ (n + 1)]
    exact mul_le_mul_left le_self_add _

/-! ## The limit -/

/-- A die that may show a result that the criterion does not match matches
with probability `q < 1`, and does not with probability `1 - q`. -/
private theorem matched_lt_one {c : Criterion} {die : PMF Int}
    (h : ∃ x ∈ {x | c.matches x = false}, x ∈ die.support) :
    matched c die < 1 ∧
      ∑' x, {x | c.matches x = false}.indicator die x = 1 - matched c die := by
  have hsum : ∑' x, {x | c.matches x = false}.indicator die x +
      matched c die = 1 := by
    rw [matched, ← ENNReal.tsum_add, ← PMF.tsum_coe die]
    refine tsum_congr fun x => ?_
    by_cases hx : c.matches x <;> simp [Set.indicator, hx]
  have hne : matched c die ≠ ⊤ := by
    intro htop
    rw [htop] at hsum
    simp at hsum
  obtain ⟨x, hx, hsupp⟩ := h
  have hu : ∑' x, {x | c.matches x = false}.indicator die x ≠ 0 := by
    intro h0
    have := ENNReal.tsum_eq_zero.mp h0 x
    rw [Set.indicator_of_mem hx] at this
    exact hsupp this
  refine ⟨?_, ENNReal.eq_sub_of_add_eq hne hsum⟩
  calc matched c die
      < matched c die + ∑' x, {x | c.matches x = false}.indicator die x :=
        ENNReal.lt_add_right hne hu
    _ = 1 := by rw [add_comm, hsum]

/--
As its limit grows, a limited reroll converges to the conditional law of the
die, given that it does not match: the law that the specification gives an
unlimited reroll. A result that the criterion matches is shown only if every
draw matched, with `qⁿ` times its probability, which vanishes, and any other
result sums its first appearance over every draw,
`(1 + q + q² + ⋯) · p = p / (1 − q)`.

So an unlimited reroll is the limit of limited ones, as the Rust's loop,
`reroll_within` in `primitives.rs`, which lifts the limit, is the limit of its
loops that stop after `n` draws: they agree on every draw before the `n`th.
The loop stops with probability one, since it rolls again after `n` draws
only with probability `qⁿ` (`matched_limitedReplacement`), and the
probability of each result that it stops on grows toward the conditional law
(`monotone_limitedReplacement`).

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `x`: The result.

# Hypotheses
- `h`: The die may show a result that the criterion does not match.
-/
theorem tendsto_limitedReplacement {c : Criterion} {die : PMF Int}
    (h : ∃ x ∈ {x | c.matches x = false}, x ∈ die.support) (x : Int) :
    Tendsto (fun n => limitedReplacement c die n x) atTop
      (𝓝 (die.filter {x | c.matches x = false} h x)) := by
  obtain ⟨hq, hu⟩ := matched_lt_one h
  refine (tendsto_add_atTop_iff_nat 1).mp ?_
  simp only [limitedReplacement_apply]
  by_cases hx : c.matches x
  · rw [PMF.filter_apply_eq_zero_of_notMem h (by simp [hx])]
    simp only [hx, ↓reduceIte]
    simpa using ENNReal.Tendsto.mul_const
      (ENNReal.tendsto_pow_atTop_nhds_zero_iff.mpr hq)
      (.inr (PMF.apply_ne_top die x))
  · simp only [hx, Bool.false_eq_true, ↓reduceIte, PMF.filter_apply, hu,
      Set.indicator_of_mem
        (show x ∈ {x | c.matches x = false} by simpa using hx)]
    rw [mul_comm (die x)]
    refine ENNReal.Tendsto.mul_const ?_ (.inr (PMF.apply_ne_top die x))
    rw [← ENNReal.tsum_geometric]
    exact (tendsto_add_atTop_iff_nat 1).mpr
      (ENNReal.tendsto_nat_tsum fun i => matched c die ^ i)

/-- Counting each face that the criterion does not match counts those faces. -/
private theorem sum_map_unmatched (c : Criterion) (xs : List Int) :
    (xs.map fun y => if c.matches y then (0 : ℝ≥0∞) else 1).sum =
      ((xs.filter (!c.matches ·)).length : ℝ≥0∞) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    by_cases hx : c.matches x <;> simp [hx, ih, add_comm]

/--
The conditional law of a custom die, given that a criterion does not match,
is the die of the faces that it does not match, each equally likely by
position, as `reroll` in `XdySpec/Semantics.lean` draws the replacement of an
unlimited reroll.

# Parameters
- `xs`: The faces.
- `c`: The criterion.

# Hypotheses
- `h`: The die may show a result that the criterion does not match.
- `hne`: Some face does not match.
-/
theorem customDie_filter {xs : List Int} {c : Criterion}
    (h : ∃ x ∈ {x | c.matches x = false}, x ∈ (customDie xs).support)
    (hne : ((xs.filter (!c.matches ·) : List Int) : Multiset Int) ≠ 0) :
    (customDie xs).filter {x | c.matches x = false} h =
      PMF.ofMultiset _ hne := by
  have hxs : xs ≠ [] := by
    rintro rfl
    simp at hne
  have hL : (xs.length : ℝ≥0∞) ≠ 0 := by simpa using hxs
  have hF : ((xs.filter (!c.matches ·)).length : ℝ≥0∞) ≠ 0 := by
    simpa using hne
  have hsum : ∑' y, {y | c.matches y = false}.indicator (customDie xs) y =
      ((xs.filter (!c.matches ·)).length : ℝ≥0∞) / xs.length := by
    rw [← sum_map_unmatched, ← tsum_customDie hxs]
    refine tsum_congr fun y => ?_
    by_cases hy : c.matches y <;> simp [hy]
  have hfne : xs.filter (!c.matches ·) ≠ [] := by simpa using hne
  rw [show PMF.ofMultiset _ hne = customDie (xs.filter (!c.matches ·)) by
    unfold customDie
    simp only [hfne, ↓reduceDIte]]
  ext x
  rw [PMF.filter_apply, hsum, customDie_apply hfne]
  by_cases hx : c.matches x
  · rw [List.count_eq_zero.mpr (by simp [hx])]
    simp [hx]
  · rw [Set.indicator_of_mem (by simpa using hx), customDie_apply hxs,
      List.count_filter (by simpa using hx),
      ENNReal.inv_div (.inl (by simp)) (.inl hL), div_eq_mul_inv,
      div_eq_mul_inv, div_eq_mul_inv, mul_assoc,
      ← mul_assoc _ (xs.length : ℝ≥0∞), ENNReal.inv_mul_cancel hL (by simp),
      one_mul]

/--
As its limit grows, a limited reroll of a die converges to the replacement that
the specification gives an unlimited reroll of the die, if some face does not
match: `tendsto_limitedReplacement`, of the die's faces.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `x`: The result.

# Hypotheses
- `h`: The criterion does not match every face.
-/
theorem tendsto_limitedReplacement_faces {d : Die} {c : Criterion}
    (h : ¬d.facesWithin c = true) (x : Int) :
    Tendsto (fun n => limitedReplacement c (customDie d.faces) n x) atTop
      (𝓝 (PMF.ofMultiset _ (unmatched_ne_zero h) x)) := by
  have hs : ∃ y ∈ {y | c.matches y = false},
      y ∈ (customDie d.faces).support := by
    simp only [Die.facesWithin, List.all_eq_true, not_forall] at h
    obtain ⟨y, hy, hm⟩ := h
    exact ⟨y, by simpa using hm, mem_support_customDie hy⟩
  rw [← customDie_filter hs]
  exact tendsto_limitedReplacement hs x

/-! ## The limit of a record's reroll -/

/-- Appending a result determines both the record and the result. -/
private theorem push_inj {r r' : Record} {x x' : Int} :
    r.push x = r'.push x' ↔ r = r' ∧ x = x' := by
  refine ⟨fun h => ?_, fun ⟨hr, hx⟩ => hr ▸ hx ▸ rfl⟩
  cases r
  cases r'
  simp only [Record.push, Record.mk.injEq] at h ⊢
  obtain ⟨hres, hl, hh, hd⟩ := h
  obtain ⟨hres, hx⟩ := List.append_inj' hres rfl
  exact ⟨⟨hres, hl, hh, hd⟩, by simpa using hx⟩

/-- A record drawn and then extended by a draw shows each extension by the
product of the probabilities of the record and of the draw. -/
private theorem bind_push_apply (p : PMF Record) (P : PMF Int) (r : Record)
    (x : Int) : (p.bind fun r' => P.map r'.push) (r.push x) = p r * P x := by
  rw [PMF.bind_apply, tsum_eq_single r fun r' hr' => by
    simp [PMF.map_apply, push_inj, Ne.symm hr']]
  simp [PMF.map_apply, push_inj]

/-- A record that extends no record has no probability of being drawn by an
extension. -/
private theorem bind_push_apply_of_not (p : PMF Record) (P : PMF Int)
    {R : Record} (h : ∀ (r : Record) (x : Int), R ≠ r.push x) :
    (p.bind fun r' => P.map r'.push) R = 0 := by
  simp [PMF.bind_apply, PMF.map_apply, h]

/-- Extending records by draws is continuous in the laws of both. -/
private theorem tendsto_bind_push {p : ℕ → PMF Record} {p' : PMF Record}
    {P : ℕ → PMF Int} {P' : PMF Int}
    (hp : ∀ R, Tendsto (fun n => p n R) atTop (𝓝 (p' R)))
    (hP : ∀ x, Tendsto (fun n => P n x) atTop (𝓝 (P' x))) (R : Record) :
    Tendsto (fun n => ((p n).bind fun r : Record => (P n).map r.push) R)
      atTop (𝓝 ((p'.bind fun r : Record => P'.map r.push) R)) := by
  by_cases hR : ∃ (r : Record) (x : Int), R = r.push x
  · obtain ⟨r, x, rfl⟩ := hR
    simp only [bind_push_apply]
    exact ENNReal.Tendsto.mul (hp r) (.inr (PMF.apply_ne_top _ _)) (hP x)
      (.inr (PMF.apply_ne_top _ _))
  · push Not at hR
    simp only [bind_push_apply_of_not _ _ hR]
    exact tendsto_const_nhds

/-- Rerolling kept results is continuous in the law of the replacement, from
any law of the record so far. -/
private theorem tendsto_foldl_reroll (die : Die) (rerolled : Int → Bool)
    {P : ℕ → PMF Int} {P' : PMF Int}
    (hP : ∀ x, Tendsto (fun n => P n x) atTop (𝓝 (P' x))) :
    ∀ (kept : List Int) {p : ℕ → PMF Record} {p' : PMF Record},
      (∀ R, Tendsto (fun n => p n R) atTop (𝓝 (p' R))) →
      ∀ R, Tendsto
        (fun n => kept.foldl
          (fun p x =>
            if rerolled x then p.bind fun r => (P n).map r.push
            else p.map (·.push x))
          (p n) R) atTop
        (𝓝 (kept.foldl
          (fun p x =>
            if rerolled x then p.bind fun r => P'.map r.push
            else p.map (·.push x))
          p' R))
  | [], _, _, hp => hp
  | x :: kept, p, p', hp => by
    simp only [List.foldl_cons]
    refine tendsto_foldl_reroll die rerolled hP kept fun R => ?_
    by_cases hx : rerolled x
    · simp only [hx, ↓reduceIte]
      exact tendsto_bind_push hp hP R
    · have hmap : ∀ q : PMF Record,
          q.map (·.push x) = q.bind fun r => (PMF.pure x).map r.push :=
        fun q => by
          simp only [PMF.pure_map]
          exact (PMF.bind_pure_comp _ _).symm
      simp only [hx, Bool.false_eq_true, ↓reduceIte, hmap]
      exact tendsto_bind_push hp (fun _ => tendsto_const_nhds) R

/--
Rerolling the kept results of a record is continuous in the law of the
replacement: if the laws of the replacements converge, then so do the laws of
the records that they leave.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `R`: The record after the reroll.

# Hypotheses
- `hP`: The laws of the replacement converge to `P'`.
-/
theorem tendsto_rerollKept (die : Die) (kept : List Int)
    (rerolled : Int → Bool) {P : ℕ → PMF Int} {P' : PMF Int}
    (hP : ∀ x, Tendsto (fun n => P n x) atTop (𝓝 (P' x))) (R : Record) :
    Tendsto (fun n => rerollKept die kept rerolled (P n) R) atTop
      (𝓝 (rerollKept die kept rerolled P' R)) :=
  tendsto_foldl_reroll die rerolled hP kept (fun _ => tendsto_const_nhds) R

/--
As its limit grows, the specification's limited reroll of a record converges
to its unlimited reroll, if some face of the die does not match: an unlimited
reroll is the limit of limited ones, record by record, as
`tendsto_limitedReplacement_faces` shows of each die.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `R`: The record after the reroll.

# Hypotheses
- `hc`: The record holds no chains.
- `h`: The criterion does not match every face.

# Notes
A record that holds chains rerolls each die of each chain alone, by the same
replacement, so its limited rerolls converge alike; this proves it of the
records that the forward pass describes, which hold no chains.
-/
theorem tendsto_reroll {r : Record} {c : Criterion} (hc : r.chains = [])
    (h : ¬r.die.facesWithin c = true) (R : Record) :
    Tendsto (fun n => reroll r c (some n) R) atTop
      (𝓝 (reroll r c none R)) := by
  have hlim : ∀ n, n ≠ 0 → reroll r c (some n) =
      rerollKept r.die r.kept c.matches
        (limitedReplacement c (customDie r.die.faces) n) := fun n hn => by
    simp only [reroll, rerollEach, hc, List.isEmpty_nil, ↓reduceIte]
    congr 1
    funext x
    simp [Nat.pos_of_ne_zero hn]
  rw [show reroll r c none = rerollKept r.die r.kept c.matches
      (PMF.ofMultiset _ (unmatched_ne_zero h)) by
    simp [reroll, h, rerollEach, hc]]
  refine Tendsto.congr' ?_
    (tendsto_rerollKept r.die r.kept c.matches
      (tendsto_limitedReplacement_faces h) R)
  filter_upwards [eventually_ne_atTop 0] with n hn
  rw [hlim n hn]

/-! ## Rerolls that match every face -/

/--
A die whose every possible result matches is redrawn from itself, however
many times: a limited reroll leaves its law alone. So no limit of limited
rerolls gives an unlimited reroll of such a die a meaning, and the Rust's
loop would never end. Instead, by convention, an unlimited reroll crosses
out every matching die (`reroll_crossOut`).

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `n`: The limit.

# Hypotheses
- `h`: The criterion matches every result that the die may show.
-/
theorem limitedReplacement_of_matches {c : Criterion} {die : PMF Int}
    (h : ∀ x ∈ die.support, c.matches x = true) (n : Nat) :
    limitedReplacement c die n = die := by
  have hnot : ∀ x, c.matches x = false → die x = 0 := fun x hx =>
    (PMF.apply_eq_zero_iff die x).mpr fun hs => by simp [h x hs] at hx
  have hm : matched c die = 1 := by
    rw [matched, ← PMF.tsum_coe die]
    refine tsum_congr fun x => ?_
    cases hx : c.matches x
    · exact (hnot x hx).symm
    · rfl
  have hredraw : redraw c die die = die := by
    ext x
    rw [redraw_apply, hm, one_mul]
    cases hx : c.matches x
    · simp [hnot x hx]
    · simp
  rw [limitedReplacement]
  induction n - 1 with
  | zero => rfl
  | succ k ih => rw [Nat.repeat, ih, hredraw]

/--
A die whose every face matches is redrawn from itself, however many times:
`limitedReplacement_of_matches`, of the die's faces.

# Parameters
- `d`: The die.
- `c`: The criterion.
- `n`: The limit.

# Hypotheses
- `h`: The criterion matches every face.
- `hF`: The die has faces.
-/
theorem limitedReplacement_faces_of_facesWithin {d : Die} {c : Criterion}
    (h : d.facesWithin c = true) (hF : d.faces ≠ []) (n : Nat) :
    limitedReplacement c (customDie d.faces) n = customDie d.faces := by
  refine limitedReplacement_of_matches (fun x hx => ?_) n
  simp only [Die.facesWithin, List.all_eq_true] at h
  rcases mem_of_mem_support_customDie hx with hx | ⟨he, _⟩
  · exact h x hx
  · exact absurd he hF

/--
An unlimited reroll whose criterion matches every face of the die rolls
nothing, but crosses out every kept result that it matches, leaving the others
kept, in ascending order, as `reroll` in `primitives.rs` does.

# Parameters
- `r`: The record.
- `c`: The criterion.

# Hypotheses
- `hc`: The record holds no chains.
- `h`: The criterion matches every face.
-/
theorem reroll_crossOut {r : Record} {c : Criterion} (hc : r.chains = [])
    (h : r.die.facesWithin c = true) :
    reroll r c none = PMF.pure
      { results := r.kept.filter (!c.matches ·), die := r.die } := by
  simp [reroll, h, hc]

/--
The record that an unlimited reroll crosses out keeps exactly the kept results
that its criterion does not match, still in ascending order.

# Parameters
- `r`: The record.
- `c`: The criterion.
-/
theorem kept_crossOut (r : Record) (c : Criterion) :
    ({ results := r.kept.filter (!c.matches ·), die := r.die } : Record).kept =
      r.kept.filter (!c.matches ·) := by
  have hsorted : (r.kept.filter (!c.matches ·)).Pairwise (· ≤ ·) :=
    (Record.kept_pairwise r).sublist List.filter_sublist
  generalize r.kept.filter (!c.matches ·) = l at hsorted ⊢
  have hms : l.mergeSort (fun a b => decide (a ≤ b)) = l :=
    List.mergeSort_of_pairwise (by simpa using hsorted)
  simp [Record.kept, Record.sorted, hms]

/-- A zero adds nothing to a saturating sum that has already saturated. -/
private theorem foldl_add_map_crossOut (c : Criterion) :
    ∀ (xs : List Int) (acc : Int), clamp acc = acc →
      (xs.map fun x => if c.matches x then 0 else x).foldl add acc =
        (xs.filter (!c.matches ·)).foldl add acc
  | [], _, _ => rfl
  | x :: xs, acc, hacc => by
    have hclamp : ∀ y, clamp (add acc y) = add acc y := fun y => by
      simp only [add, clamp, i32Min, i32Max]
      omega
    cases hx : c.matches x
    · simp only [List.map_cons, hx, Bool.false_eq_true, ↓reduceIte,
        List.foldl_cons, List.filter_cons, Bool.not_false]
      exact foldl_add_map_crossOut c xs _ (hclamp x)
    · simp only [List.map_cons, hx, ↓reduceIte, List.foldl_cons,
        List.filter_cons, Bool.not_true, Bool.false_eq_true]
      rw [show add acc 0 = acc by rw [add, Int.add_zero, hacc]]
      exact foldl_add_map_crossOut c xs acc hacc

/--
An unlimited reroll whose criterion matches every face of the die crosses out
every kept result that it matches, so that each contributes zero to the sum:
the record sums its kept results with each that matches replaced by `0`.

# Parameters
- `r`: The record.
- `c`: The criterion.

# Hypotheses
- `hc`: The record holds no chains.
- `h`: The criterion matches every face.
-/
theorem sum_reroll_crossOut {r : Record} {c : Criterion} (hc : r.chains = [])
    (h : r.die.facesWithin c = true) :
    (reroll r c none).map Record.sum = PMF.pure
      ((r.kept.map fun x => if c.matches x then 0 else x).foldl add 0) := by
  rw [reroll_crossOut hc h, PMF.pure_map, Record.sum, kept_crossOut,
    foldl_add_map_crossOut c _ 0 (by decide)]

end Xdy.Spec
