import XdySpec.Ghost
import XdySpecTest.Fixtures

/-!
# Ghost tests

The ghost's invariant, applied to the function of
`XdySpecTest/Conditioning.lean` whose split on `b` merges beneath its split on
`a`: its hypotheses hold, so before every instruction the ghost's states are
the function's, and its outcomes have the law of the forward pass's worlds.
-/

namespace Xdy.Spec.Test.Ghost

-- `nested`, from `XdySpecTest/Fixtures.lean`: the plan splits on `a` and on
-- `b = a + x`, merges `b` into `@5` beneath `a`, and then `a` into `@6`.

/-- The values of the function. -/
private abbrev vs : Values := ⟨nested.instructions.toList⟩

-- The ghost starts with every outcome blank, in the pass's one world.
example : (Forward.initial vs (.ofOracle nestedInitial)).Traces [] []
    (PMF.pure (.ofOracle nestedInitial, blanks vs)) :=
  traces_zero

-- The ghost's invariant holds before every instruction, and after the last.
example (pc : Nat) (h : pc ≤ 10) :
    (forward vs (.ofOracle nestedInitial) pc).Traces (vs.liveBefore pc)
      (vs.openSplits pc) (ghost vs (.ofOracle nestedInitial) pc) :=
  traces_forward (operandsAreValues_of_validate nested_valid)
    (has_initialState nested_valid nested_initialState) h

-- Its states are the function's.
example (pc : Nat) (h : pc ≤ 10) :
    (ghost vs (.ofOracle nestedInitial) pc).map Prod.fst =
      exec (vs.instructions.take pc) (PMF.pure (.ofOracle nestedInitial)) :=
  ghost_fst h

-- Its outcomes have the law of the pass's worlds: both splits' while `b` is
-- live, and `a`'s alone once `b` merges.
example :
    (ghost vs (.ofOracle nestedInitial) 5).map Prod.snd =
      splitOutcomes (forward vs (.ofOracle nestedInitial) 5).K [1, 4] := by
  refine (ghost_snd (vs := vs) (operandsAreValues_of_validate nested_valid)
    (has_initialState nested_valid nested_initialState) (by decide)).trans ?_
  congr 1
  decide +kernel
example :
    (ghost vs (.ofOracle nestedInitial) 8).map Prod.snd =
      splitOutcomes (forward vs (.ofOracle nestedInitial) 8).K [1] := by
  refine (ghost_snd (vs := vs) (operandsAreValues_of_validate nested_valid)
    (has_initialState nested_valid nested_initialState) (by decide)).trans ?_
  congr 1
  decide +kernel

end Xdy.Spec.Test.Ghost
