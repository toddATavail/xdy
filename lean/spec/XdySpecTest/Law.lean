import XdySpec.Law

/-!
# Law tests

The laws of the operations of `Dist`, applied: the plain law of `bind`, as
the special case of its law through transformations, laws composed from
`uniform` and `map`, and a scaled die.
-/

namespace Xdy.Spec.Test

open Xdy

/-! ## Laws of the operations -/

-- With identities for the transformations, `bind` has the plain law of
-- `PMF.bind`.
example (d : Dist Int) (k : Int → Dist Int) (p : PMF Int)
    (q : Int → PMF Int) (hd : d.HasLaw p)
    (hk : ∀ a ∈ p.support, (k a).HasLaw (q a)) :
    (d.bind k).HasLaw (p.bind q) := by
  have := Dist.hasLaw_bind (g := id) (h := id) (by rwa [PMF.map_id])
    fun s hs => by rw [PMF.map_id]; exact hk s hs
  rwa [PMF.map_id] at this

-- A die with faces `1` through `6`, halved and rounded down, has the law of
-- the die, halved and rounded down.
example :
    ((Dist.uniform ([1, 2, 3, 4, 5, 6] : List Int)).map (· / 2)).HasLaw
      ((PMF.ofMultiset ([1, 2, 3, 4, 5, 6] : List Int) (by simp)).map
        (· / 2)) :=
  Dist.hasLaw_map _ (Dist.hasLaw_uniform _ (by simp))

-- Every face of a die doubled has the law of the die, as a world's common
-- factor scales its answer.
example :
    (Dist.uniform ([1, 1, 2, 2] : List Int)).HasLaw
      (PMF.ofMultiset ([1, 2] : List Int) (by simp)) :=
  Dist.hasLaw_scale (d := Dist.uniform ([1, 2] : List Int)) (k := 2)
    (by decide) (Dist.positive_uniform _)
    (fun a => by
      simp only [Dist.weight_uniform, List.count_cons, List.count_nil]
      split_ifs <;> simp_all)
    (Dist.hasLaw_uniform _ (by simp))

/-! ## Docstring examples, verbatim -/

example : (Dist.uniform ([1, 2, 2] : List Int)).prob 2 = 2 / 3 := by
  rw [Dist.prob, Dist.weight_uniform, Dist.total_uniform]; rfl

end Xdy.Spec.Test
