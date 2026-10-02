import Mathlib.Probability.Distributions.Uniform
import Xdy.DistLaws

/-!
# The law of a finite distribution

The oracle's `Dist` weighs outcomes with natural numbers over one total; the
specification's `PMF` gives them probabilities. A distribution has law `p`
when every one of its outcomes has probability, in `p`, of its weight over
its total. This module proves that each operation of `Dist` has the law of
the corresponding operation of `PMF`: `pure` of `PMF.pure`, `uniform` of
`PMF.ofMultiset`, `map` of `PMF.map`, and `bind` of `PMF.bind`, so that the
proof that the oracle meets the specification can follow the oracle
operation by operation.

`bind` scales its branches to a common total, and its law shows that the
scaling cancels, whatever common multiple of the branches' totals it picks.
Its law is stated through a transformation of the outcomes: the oracle
merges outcomes that the specification keeps apart, such as rolling records
that differ only in the order of their results, so the oracle's distribution
has the law of the specification's, transformed. With the identity for that
transformation, it is the plain law of `bind`.

The forward pass of `xdy/src/distribution/propagation.rs` weighs its
distributions likewise, and two of its weighings are proved here as well. A
mixture over any common multiple of its branches' totals has the law of
`bind` (`hasLaw_mixture`), as a merge and `Mixture::sum` mix; and scaling
every weight by one positive factor keeps the law (`hasLaw_scale`), as a
world scales its answer by its common factor, the totals of the values that
it never read.

The laws of the weights themselves, as sums over the entries of the hash
map, need no Mathlib, and live beside the oracle in `Xdy/DistLaws.lean`.
-/

open scoped ENNReal

namespace Xdy

open Std (HashMap)

namespace Dist

/-! ## Reindexing -/

/--
Summing over a transformed law sums over the original, transformed.

# Parameters
- `p`: The law.
- `g`: The transformation.
- `F`: The function summed, weighted by the probability of its argument.
-/
private theorem tsum_map_mul {σ α : Type} (p : PMF σ) (g : σ → α)
    (F : α → ℝ≥0∞) :
    ∑' a, (p.map g) a * F a = ∑' s, p s * F (g s) := by
  simp_rw [PMF.map_apply, ← ENNReal.tsum_mul_right]
  rw [ENNReal.tsum_comm]
  congr 1
  ext s
  rw [tsum_eq_single (g s) fun a ha => by simp [ha]]
  simp

/--
The scaling of a branch of `bind` cancels: a weight `w` over total `T`, then a
weight `v` over total `t`, scaled up to a common multiple `l` of the totals
of the branches, is the product of the two probabilities.

# Parameters
- `w`: The weight of the branch.
- `v`: The weight of an outcome of its continuation.
- `t`: The total of its continuation.
- `l`: The common multiple.
- `T`: The total of the first draw.

# Hypotheses
- `ht`, `hl`, `hT`: The totals and their common multiple are positive.
- `hdvd`: The continuation's total divides the common multiple.
-/
private theorem scale_div (w v t l T : ℕ) (ht : 0 < t) (hdvd : t ∣ l)
    (hl : 0 < l) (hT : 0 < T) :
    ((w * (l / t) * v : ℕ) : ℝ≥0∞) / ((l * T : ℕ) : ℝ≥0∞) =
      (w / T) * (v / t) := by
  obtain ⟨m, rfl⟩ := hdvd
  rw [Nat.mul_div_cancel_left m ht]
  have hm : 0 < m := Nat.pos_of_mul_pos_left hl
  rw [eq_comm, ENNReal.eq_div_iff (by simp; omega) (ENNReal.natCast_ne_top _)]
  push_cast
  have hT' : (T : ℝ≥0∞) * (w / T) = w :=
    ENNReal.mul_div_cancel (by simp; omega) (by simp)
  have ht' : (t : ℝ≥0∞) * (v / t) = v :=
    ENNReal.mul_div_cancel (by simp; omega) (by simp)
  calc (t : ℝ≥0∞) * m * T * (w / T * (v / t))
      = m * (T * (w / T)) * (t * (v / t)) := by ring
    _ = w * m * v := by rw [hT', ht']; ring

variable {α β σ τ : Type} [DecidableEq α] [Hashable α] [DecidableEq β]
  [Hashable β]

/-! ## Probabilities -/

/--
The probability of an outcome: its weight over the total.

# Parameters
- `d`: The distribution.
- `a`: The outcome.

# Returns
The probability of `a`, or `0` if it is impossible.

# Examples
```lean
example : (Dist.uniform ([1, 2, 2] : List Int)).prob 2 = 2 / 3 := by
  rw [Dist.prob, Dist.weight_uniform, Dist.total_uniform]; rfl
```
-/
noncomputable def prob (d : Dist α) (a : α) : ℝ≥0∞ := d.weight a / d.total

/-- A distribution has law `p` when every outcome has probability, in `p`, of
its weight over the total, which is positive, and every weight is positive,
as every operation of `Dist` keeps them. -/
def HasLaw (d : Dist α) (p : PMF α) : Prop :=
  d.Positive ∧ 0 < d.total ∧ ∀ a, p a = d.prob a

/--
A sum over every outcome, of a function of its weight that vanishes on weight
`0`, is a sum over the entries, since every other outcome has weight `0`.

# Parameters
- `d`: The distribution.
- `g`: The function summed, of an outcome and its weight.

# Hypotheses
- `hg`: `g` vanishes on weight `0`.
-/
theorem tsum_eq_sum_entries (d : Dist α) (g : α → ℕ → ℝ≥0∞)
    (hg : ∀ a, g a 0 = 0) :
    ∑' a, g a (d.weight a) =
      (d.weights.toList.map fun e => g e.1 e.2).sum := by
  rw [tsum_eq_sum (s := (d.weights.toList.map (·.1)).toFinset) fun a ha => by
      have : d.weight a = 0 := by
        rw [weight, HashMap.getD_eq_getD_getElem?]
        cases h : d.weights[a]? with
        | none => rfl
        | some v =>
          exact absurd (List.mem_toFinset.mpr (List.mem_map.mpr
            ⟨(a, v), HashMap.mem_toList_iff_getElem?_eq_some.mpr h, rfl⟩)) ha
      rw [this, hg],
    List.sum_toFinset _ d.nodup_outcomes, List.map_map]
  congr 1
  apply List.map_congr_left
  intro e he
  simp [weight_of_mem d he]

/-! ## Laws of the operations -/

/--
A certain outcome has the law of `PMF.pure`.

# Parameters
- `a`: The outcome.
-/
theorem hasLaw_pure (a : α) : (pure a).HasLaw (PMF.pure a) := by
  refine ⟨positive_pure a, by rw [total_pure]; exact Nat.one_pos, fun b => ?_⟩
  rw [prob, weight_pure, total_pure, PMF.pure_apply]
  by_cases h : b = a
  · subst h; simp
  · simp [h, Ne.symm h]

/--
A uniform draw from a list has the law of `PMF.ofMultiset` of the list:
each position equally likely.

# Parameters
- `as`: The outcomes.

# Hypotheses
- `h`: The list is not empty.
-/
theorem hasLaw_uniform (as : List α) (h : as ≠ []) :
    (uniform as).HasLaw (PMF.ofMultiset as (by simpa using h)) := by
  have hlen : 0 < as.length := List.length_pos_iff.mpr h
  refine ⟨positive_uniform as, by rw [total_uniform]; exact hlen, fun b => ?_⟩
  rw [PMF.ofMultiset_apply, prob, weight_uniform, total_uniform,
    Multiset.coe_card]
  congr
  convert Multiset.coe_count b as

/--
Transforming the outcomes transforms the law.

# Parameters
- `f`: The transformation.

# Hypotheses
- `h`: `d` has law `p`.
-/
theorem hasLaw_map (f : α → β) {d : Dist α} {p : PMF α} (h : d.HasLaw p) :
    (d.map f).HasLaw (p.map f) := by
  obtain ⟨hpos, htot, hp⟩ := h
  refine ⟨positive_map f d hpos, by rw [total_map]; exact htot, fun b => ?_⟩
  rw [PMF.map_apply, prob, weight_map, total_map]
  simp_rw [hp, prob]
  rw [tsum_eq_sum_entries d
      (fun a w => if b = f a then (w : ℝ≥0∞) / d.total else 0) (by simp),
    Nat.cast_list_sum, List.map_map]
  simp only [div_eq_mul_inv]
  rw [← List.sum_map_mul_right]
  congr 1
  apply List.map_congr_left
  intro e _
  by_cases h : b = f e.1
  · simp [h]
  · simp [h, Ne.symm h]

/--
A mixture over any common multiple of the continuations' totals has the law
of `PMF.bind`, through transformations of the outcomes, as `bind` has over the
least: if the first draw has the law of `p`, transformed by `g`, and each
continuation, at the transformation of an outcome of `p`, has the law of the
corresponding continuation of `p`, transformed by `h`, then a distribution
that adds, for each entry `(a, w)` of the first draw, the weights of `k a`,
of total `t`, scaled by `w * (l / t)`, has the law of `PMF.bind`, transformed
by `h`, whatever positive common multiple `l` of the totals it scales to.

# Parameters
- `d`: The distribution of the first draw.
- `k`: The continuation.
- `e`: The mixture.
- `l`: The common multiple to which it scales the continuations.
- `p`: The law of the first draw, before transformation.
- `q`: The law of the continuation, before transformation.
- `g`: The transformation of the first draw.
- `h`: The transformation of the continuation's outcomes.

# Hypotheses
- `hd`: `d` has the law of `p`, transformed by `g`.
- `hk`: For each possible outcome `s` of `p`, `k (g s)` has the law of
  `q s`, transformed by `h`.
- `hl`: `l` is positive.
- `hdvd`: The continuation of every entry of `d` has a total that divides `l`.
- `hepos`: Every weight of the mixture is positive.
- `hetot`: The mixture's total is `l` times that of `d`.
- `hew`: Each outcome of the mixture weighs its scaled weights in the
  continuations.

# Notes
`merge` and `Mixture::sum` in `xdy/src/distribution/propagation.rs` and
`propagation/mixture.rs` mix so, the merge over the least common multiple of
its survivors' totals, each times its world's common factor, which therefore
cancels.
-/
theorem hasLaw_mixture {d : Dist α} {k : α → Dist β} {e : Dist β} {l : ℕ}
    {p : PMF σ} {q : σ → PMF τ} {g : σ → α} {h : τ → β}
    (hd : d.HasLaw (p.map g))
    (hk : ∀ s ∈ p.support, (k (g s)).HasLaw ((q s).map h)) (hl : 0 < l)
    (hdvd : ∀ x ∈ d.weights.toList, (k x.1).total ∣ l) (hepos : e.Positive)
    (hetot : e.total = l * d.total)
    (hew : ∀ c, e.weight c =
      (d.weights.toList.map fun x => x.2 * (l / (k x.1).total) *
        (k x.1).weight c).sum) :
    e.HasLaw ((p.bind q).map h) := by
  obtain ⟨hpos, htot, hp⟩ := hd
  -- Every outcome of `d` transforms a possible outcome of `p`, so its
  -- continuation has a law, and with it a positive total.
  have reach : ∀ x ∈ d.weights.toList, ∃ s ∈ p.support, g s = x.1 := by
    intro x hx
    have : (p.map g) x.1 ≠ 0 := by
      rw [hp, prob, weight_of_mem d hx, ne_eq, ENNReal.div_eq_zero_iff]
      have := hpos x hx
      simp; omega
    rw [← PMF.mem_support_iff, PMF.support_map] at this
    exact this
  have hk' : ∀ x ∈ d.weights.toList, 0 < (k x.1).total := by
    intro x hx
    obtain ⟨s, hs, hsx⟩ := reach x hx
    rw [← hsx]
    exact (hk s hs).2.1
  refine ⟨hepos, by rw [hetot]; exact Nat.mul_pos hl htot, fun c => ?_⟩
  have step : ∀ s, p s * ((q s).map h) c = p s * (k (g s)).prob c := by
    intro s
    by_cases hs : s ∈ p.support
    · rw [(hk s hs).2.2 c]
    · rw [PMF.mem_support_iff, not_not] at hs; simp [hs]
  -- Sum over `p`, then over the transformed law, then over the entries of
  -- `d`, and cancel each branch's scaling.
  rw [PMF.map_bind, PMF.bind_apply]
  simp_rw [step]
  rw [← tsum_map_mul p g fun a => (k a).prob c]
  simp_rw [hp, prob]
  rw [tsum_eq_sum_entries d
      (fun a w => (w : ℝ≥0∞) / d.total * ((k a).weight c / (k a).total))
      (by simp),
    hew c, hetot, Nat.cast_list_sum, List.map_map, div_eq_mul_inv,
    ← List.sum_map_mul_right]
  congr 1
  apply List.map_congr_left
  intro x hx
  simp only [Function.comp]
  rw [← div_eq_mul_inv, scale_div _ _ _ _ _ (hk' x hx) (hdvd x hx) hl htot]

/--
`bind` has the law of `PMF.bind`, through transformations of the outcomes:
if the first draw has the law of `p`, transformed by `g`, and each
continuation, at the transformation of an outcome of `p`, has the law of the
corresponding continuation of `p`, transformed by `h`, then `bind` has the
law of `PMF.bind`, transformed by `h`.

# Parameters
- `d`: The distribution of the first draw.
- `k`: The continuation.
- `p`: The law of the first draw, before transformation.
- `q`: The law of the continuation, before transformation.
- `g`: The transformation of the first draw.
- `h`: The transformation of the continuation's outcomes.

# Hypotheses
- `hd`: `d` has the law of `p`, transformed by `g`.
- `hk`: For each possible outcome `s` of `p`, `k (g s)` has the law of
  `q s`, transformed by `h`.

# Notes
The law of `k` is needed only at the transformation of a possible outcome,
and it must not depend on which outcome was transformed, which the
hypothesis ensures, since `k (g s)` depends on `s` only through `g s`. With
identities for `g` and `h`, this is the plain law of `bind`. `bind` scales to
the least common multiple of the continuations' totals; `hasLaw_mixture`
proves the law for any common multiple.
-/
theorem hasLaw_bind {d : Dist α} {k : α → Dist β} {p : PMF σ}
    {q : σ → PMF τ} {g : σ → α} {h : τ → β} (hd : d.HasLaw (p.map g))
    (hk : ∀ s ∈ p.support, (k (g s)).HasLaw ((q s).map h)) :
    (d.bind k).HasLaw ((p.bind q).map h) := by
  have hd' := hd
  obtain ⟨hpos, _, hp⟩ := hd
  -- Every outcome of `d` transforms a possible outcome of `p`, so its
  -- continuation has a law, and with it a positive total.
  have hk' : ∀ x ∈ d.weights.toList, 0 < (k x.1).total ∧ (k x.1).Positive := by
    intro x hx
    have : (p.map g) x.1 ≠ 0 := by
      rw [hp, prob, weight_of_mem d hx, ne_eq, ENNReal.div_eq_zero_iff]
      have := hpos x hx
      simp; omega
    rw [← PMF.mem_support_iff, PMF.support_map] at this
    obtain ⟨s, hs, hsx⟩ := this
    rw [← hsx]
    exact ⟨(hk s hs).2.1, (hk s hs).1⟩
  obtain ⟨l, hl, hdvd, hbpos, hbtot, hbw⟩ := bind_laws d k hpos hk'
  exact hasLaw_mixture hd' hk hl hdvd hbpos hbtot hbw

/--
The total of a distribution is the sum of its weights.

# Parameters
- `d`: The distribution.
-/
theorem total_eq_tsum (d : Dist α) :
    (d.total : ℝ≥0∞) = ∑' a, (d.weight a : ℝ≥0∞) := by
  rw [tsum_eq_sum_entries d (fun _ w => (w : ℝ≥0∞)) (by simp), total_eq_sum,
    Nat.cast_list_sum, List.map_map]
  rfl

/--
Scaling every weight by the same positive factor keeps the law, since every
probability is a weight over the total, which scales alike. `World::finish`
in `xdy/src/distribution/propagation.rs` scales the answer so by the world's
common factor, the product of the totals of the values that it never read.

# Parameters
- `d`: The distribution.
- `e`: The distribution scaled.
- `k`: The factor.

# Hypotheses
- `hk`: The factor is positive.
- `hepos`: Every weight of `e` is positive.
- `hw`: Every weight of `e` is `k` times that of `d`.
- `hd`: `d` has law `p`.
-/
theorem hasLaw_scale {d e : Dist α} {k : ℕ} {p : PMF α} (hk : 0 < k)
    (hepos : e.Positive) (hw : ∀ a, e.weight a = k * d.weight a)
    (hd : d.HasLaw p) : e.HasLaw p := by
  obtain ⟨_, htot, hp⟩ := hd
  have hk0 : (k : ℝ≥0∞) ≠ 0 := by exact_mod_cast hk.ne'
  have hetot : (e.total : ℝ≥0∞) = k * d.total := by
    simp only [total_eq_tsum, hw, Nat.cast_mul, ENNReal.tsum_mul_left]
  have hetot' : e.total = k * d.total := by exact_mod_cast hetot
  refine ⟨hepos, by rw [hetot']; exact Nat.mul_pos hk htot, fun a => ?_⟩
  rw [hp, prob, prob, hetot, hw, Nat.cast_mul,
    ENNReal.mul_div_mul_left _ _ hk0 (ENNReal.natCast_ne_top k)]

end Dist

end Xdy
