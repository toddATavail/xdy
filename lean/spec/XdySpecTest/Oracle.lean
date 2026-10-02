import XdySpec.Oracle

/-!
# Oracle tests

Consequences of the theorem that the oracle meets the specification, drawn
for any function, since neither the oracle's hash maps nor a `PMF` reduce in
proofs about a particular one; `just oracle` runs the oracle on real
programs.
-/

namespace Xdy.Spec.Test

open Xdy

-- The specification fails, with the oracle's message, when the oracle does.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (message : String)
    (h : Xdy.Function.run f args externals = .error message) :
    run f args externals = .error message := by
  have := run_eq f args externals
  rw [h] at this
  cases hr : run f args externals <;> simp_all [Except.map]

-- Where the oracle answers, the probability of each result is its weight
-- over the total.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (d : Dist Int) (h : Xdy.Function.run f args externals = .ok d) (x : Int) :
    ∃ p, run f args externals = .ok p ∧ p x = d.weight x / d.total := by
  have ⟨p, hp, hlaw⟩ := run_hasLaw h
  exact ⟨p, hp, hlaw.2.2 x⟩

-- An outcome that the oracle holds is possible in the specification.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (d : Dist Int) (h : Xdy.Function.run f args externals = .ok d)
    (e : Int × Nat) (he : e ∈ d.weights.toList) :
    ∃ p, run f args externals = .ok p ∧ e.1 ∈ p.support := by
  have ⟨p, hp, hpos, htot, hprob⟩ := run_hasLaw h
  refine ⟨p, hp, ?_⟩
  rw [PMF.mem_support_iff, hprob, Dist.prob, Dist.weight_of_mem d he,
    ne_eq, ENNReal.div_eq_zero_iff]
  have := hpos e he
  simp; omega

/-! ## Sorting states -/

-- Every record of an initial state is empty, so sorted.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (s : Xdy.State) (h : f.initialState args externals = .ok s) :
    (State.ofOracle s).toOracle = s :=
  State.toOracle_ofOracle s fun r hr => by
    rw [initialState_records h] at hr
    rw [(Array.mem_replicate.mp hr).2]
    exact .nil

end Xdy.Spec.Test
