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

-- The specification fails, with the oracle's message, when the oracle does,
-- unless the oracle refuses an explode that may draw without bound.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (message : String) (hm : message ≠ Xdy.unboundedMessage)
    (h : Xdy.Function.run f args externals = .error message) :
    run f args externals = .error message := by
  have := run_eq f args externals (by
    rw [h]; exact fun he => hm (Except.error.inj he))
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

/-! ## Rerolls -/

-- The oracle rerolls the sorted record, whose kept results and kept chains
-- are the specification's, to the law of the specification's reroll, sorted,
-- limited or not.
example (r : Spec.Record) (c : Criterion) (l : Option Nat) :
    (Xdy.reroll r.toOracle c l).HasLaw
      ((Spec.reroll r c l).map Spec.Record.toOracle) :=
  hasLaw_reroll r c l
-- A limited replacement has the law of the specification's, as a die does.
example (c : Criterion) (faces : List Int) (n : Nat) :
    (Xdy.limitedReplacement c (Xdy.customDie faces) n).HasLaw
      (limitedReplacement c (Spec.customDie faces) n) :=
  hasLaw_limitedReplacement c n (hasLaw_customDie faces)

/-! ## Explodes -/

-- The oracle explodes the sorted record to the law of the specification's
-- explode, sorted, if limited...
example (r : Spec.Record) (c : Criterion) (n : Nat) :
    (Xdy.explode r.toOracle c (some n)).HasLaw
      ((Spec.explode r c (some n)).map Spec.Record.toOracle) :=
  hasLaw_explode r c (some n) fun h => absurd h (by simp)
-- ...or unlimited, if it matches every face, and so saturates...
example (r : Spec.Record) (c : Criterion) (h : r.die.facesWithin c = true) :
    (Xdy.explode r.toOracle c none).HasLaw
      ((Spec.explode r c none).map Spec.Record.toOracle) :=
  hasLaw_explode r c none fun _ hw => absurd h hw
-- ...or if it grows no chain still kept.
example (r : Spec.Record) (c : Criterion)
    (h : ∀ ch ∈ r.keptChains, ch.grows c = false) :
    (Xdy.explode r.toOracle c none).HasLaw
      ((Spec.explode r c none).map Spec.Record.toOracle) :=
  hasLaw_explode r c none fun _ _ => h
-- The oracle refuses exactly the explodes that the specification finds
-- unbounded, reading the state sorted.
example (s : State) (i : Instruction) :
    Xdy.unbounded s.toOracle i = unbounded s i :=
  State.unbounded_toOracle s i
-- A distribution answers no outcome that a law that it has cannot reach.
example (d : Dist Int) (p : PMF Int) (h : d.HasLaw p) (x : Int)
    (hx : x ∈ p.support) : d.any (· == x) = true := by
  by_contra hn
  have := Dist.any_eq_false_of_hasLaw (P := (· == x)) (g := id)
    (by rwa [PMF.map_id]) (by simpa using hn) hx
  simp at this

/-! ## Sorting states -/

-- Every record of an initial state is empty, so sorted.
example (f : Function) (args : List Int) (externals : List (String × Int))
    (s : Xdy.State) (h : f.initialState args externals = .ok s) :
    (State.ofOracle s).toOracle = s :=
  State.toOracle_ofOracle s fun r hr => by
    rw [initialState_records h] at hr
    rw [(Array.mem_replicate.mp hr).2]
    exact ⟨.nil, .nil⟩

end Xdy.Spec.Test
