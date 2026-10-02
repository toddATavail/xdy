import Xdy.DistLaws

/-!
# Laws of finite distributions tests

The laws applied to concrete distributions, and the law of `bind` checked
numerically where the continuations have unequal totals, so that the scaling
to their least common multiple shows.
-/

namespace Xdy.Test

open Xdy

/-- The distribution of one die with faces `1` through `n`. -/
private def die (n : Nat) : Dist Nat := Dist.uniform (List.range' 1 n)

/-- A `1D1` half the time and a `1D2` the other half: continuations with
totals `1` and `2`, so `bind` scales the first by `2`. -/
private def dynamicFaces : Dist Nat := (die 2).bind die

/-! ## Weights and totals as sums over entries -/

#guard (die 6).weight 3 ==
  ((die 6).weights.toList.map fun p => if p.1 == 3 then p.2 else 0).sum
#guard (die 6).total == ((die 6).weights.toList.map (·.2)).sum

/-! ## Pure, uniform and map -/

example : (Dist.pure (7 : Int)).total = 1 := Dist.total_pure 7
example : (Dist.uniform ([1, 2, 2] : List Int)).weight 2 = 2 := by
  rw [Dist.weight_uniform]; decide
example : (Dist.uniform ([1, 2, 2] : List Int)).total = 3 := by
  rw [Dist.total_uniform]; rfl
example (d : Dist Int) : (d.map (· % 2)).total = d.total :=
  Dist.total_map _ d
example : (Dist.uniform ([1, 2, 2] : List Int)).Positive :=
  Dist.positive_uniform _

/-! ## Bind -/

-- With `l = 2`, the `1D1` contributes `1 * (2 / 1)` times its weights, and
-- the `1D2` `1 * (2 / 2)` times its weights, over a total of `2 * 2`.
#guard dynamicFaces.weight 1 == 1 * (2 / 1) * 1 + 1 * (2 / 2) * 1
#guard dynamicFaces.weight 2 == 1 * (2 / 2) * 1
#guard dynamicFaces.total == 2 * (die 2).total

-- The laws hold of every `bind` whose continuations are drawn from `uniform`
-- over a list that is never empty.
example (d : Dist Int) (hd : d.Positive) (faces : Int → List Int)
    (nonempty : ∀ a, faces a ≠ []) :
    ∃ l, 0 < l ∧ (d.bind fun a => Dist.uniform (faces a)).total =
      l * d.total := by
  have ⟨l, hl, _, _, htotal, _⟩ := Dist.bind_laws d
    (fun a => Dist.uniform (faces a)) hd fun p _ => by
      refine ⟨?_, Dist.positive_uniform _⟩
      rw [Dist.total_uniform]
      exact List.length_pos_iff.mpr (nonempty p.1)
  exact ⟨l, hl, htotal⟩

end Xdy.Test
