import XdySpec.Invariant
import XdySpecTest.Fixtures

/-!
# Invariant tests

The plan's invariant, applied to the function of `XdySpecTest/Conditioning.lean`
whose split on `b` merges beneath its split on `a`: its hypotheses hold, so
the invariant holds before every instruction, and before the return, once
both splits have merged, the law of states, read sorted, is one world, in
which the only
live location, the answer's register, is independent of everything buried.
So the function answers the law of that register in that world. A function
that returns a split, which stays open before the return, answers the law of
the split's outcome. The plan's facts about the functions are decided by the
kernel, since they recur on well-founded measures.
-/

namespace Xdy.Spec.Test.Invariant

-- `nested`, from `XdySpecTest/Fixtures.lean`: the plan splits on `a` and on
-- `b = a + x`, merges `b` into `@5` beneath `a`, and then `a` into `@6`.

/-- The values of the function. -/
private abbrev vs : Values := ⟨nested.instructions.toList⟩

-- The invariant holds before every instruction, and after the last, in the
-- worlds of the forward pass.
example (pc : Nat) (h : pc ≤ 10) :
    (forward vs (.ofOracle nestedInitial) pc).Holds pc (vs.liveBefore pc)
      (vs.openSplits pc)
      (exec (nested.instructions.toList.take pc)
        (PMF.pure (.ofOracle nestedInitial))) :=
  invariant_run nested_valid nested_initialState h

-- Both splits are open while `b` is live, and `a`'s is open while `b`'s is.
example : vs.openSplits 5 = [1, 4] := by decide +kernel
example : vs.openSplits 7 = [1, 4] := by decide +kernel

-- Before the return, no split is open, and only `@6` is live, so the law of
-- states, with the dead locations buried, read sorted, is one world of `@6`
-- alone.
example : ∃ (buried : List Location) (base : State) (law : PMF Int),
    ((exec (nested.instructions.toList.take 9)
      (PMF.pure (.ofOracle nestedInitial))).map (·.buryAll buried)).map
        State.toOracle =
      (product base [.register 6] (Function.update
        (fun ℓ => PMF.pure (blank ℓ)) (.register 6) law)).map
          State.toOracle := by
  have h := invariant_run nested_valid nested_initialState (pc := 9)
    (by decide)
  generalize forward _ _ _ = F at h
  rcases F with ⟨buried, base, K, laws⟩
  have hsplits : vs.openSplits 9 = [] := by decide +kernel
  have hlive : (vs.liveBefore 9).map vs.location = [.register 6] := by
    decide +kernel
  refine ⟨buried, base, laws (fun v => blank (vs.location v)) (.register 6),
    ?_⟩
  rw [h.law, hsplits, hlive, splitOutcomes, List.foldl_nil, PMF.pure_bind]
  exact congrArg _ <| product_congr fun ℓ hℓ => by
    rw [List.mem_singleton.mp hℓ, Function.update_self]

-- It answers the law of `@6` in that one world.
example {p : PMF Int} (h : run nested [] [] = .ok p) :
    ∃ laws : WorldLaws vs,
      p = laws (fun v => blank (vs.location v)) (.register 6) := by
  obtain ⟨_, _, hp⟩ := run_eq_forward h nested_initialState rfl
    (show vs.writer 9 (.register 6) = some 8 by decide +kernel)
  have hsplits : vs.openSplits 9 = [] := by decide +kernel
  exact ⟨_, by
    rw [hp, show nested.instructions.size - 1 = 9 from rfl, hsplits,
      splitOutcomes, List.foldl_nil, PMF.pure_bind]⟩

/-! ## A returned split

A function that returns a split, which another instruction also reads:

```text
0  ⚅0 <- roll range 1:2
1  @0 <- sum rolling record ⚅0
2  @1 <- @0 * 2
3  return @0
```

`@0` fans out, so the plan splits on it after instruction 1, and it stays live
until the return reads it, so its split is still open there: only after the
return does it merge, into the answer. In each world of the split, `@0` is a
point mass at its outcome, so the answer is the law of the split.
-/

/-- The function. -/
private def returned : Function where
  parameters := #[]
  externals := #[]
  registerCount := 2
  rollingRecordCount := 1
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 2), .sumRollingRecord 0 0,
    .binary .mul 1 (.register 0) (.immediate 2),
    .return (.register 0)]

/-- The values of the function. -/
private abbrev rvs : Values := ⟨returned.instructions.toList⟩

-- The split on `@0` merges only after the return.
#guard (Plan.new returned.instructions.toList).steps == [[],
  [.split (.register 0)], [],
  [.merge 0 (some .answer) [.register 0, .register 1]]]

-- The answer is the law of the split's outcome.
example {p : PMF Int} (h : run returned [] [] = .ok p) :
    ∃ K : SplitLaws rvs, p = K 1 fun v => blank (rvs.location v) := by
  obtain ⟨hl, hw, hp⟩ := run_eq_forward h rfl rfl
    (show rvs.writer 3 (.register 0) = some 1 by decide +kernel)
  generalize forward _ _ _ = F at hw hp
  rcases F with ⟨_, _, K, laws⟩
  have hsplits : rvs.openSplits (returned.instructions.size - 1) = [1] := by
    decide +kernel
  have hpure : ∀ o, laws o (.register 0) = PMF.pure (o 1) := fun o =>
    hw.split_pure 1 hl (by rw [hsplits]; exact .head _) o
  refine ⟨K, ?_⟩
  rw [hp, hsplits, splitOutcomes, List.foldl_cons, List.foldl_nil, drawSplit,
    PMF.pure_bind, PMF.bind_map]
  simp only [hpure]
  conv_rhs => rw [← PMF.bind_pure (K 1 _)]
  rfl

/-! ## Docstring examples, verbatim -/

example (K : SplitLaws vs) :
    splitOutcomes K [] = PMF.pure fun v => blank (vs.location v) := rfl

end Xdy.Spec.Test.Invariant
