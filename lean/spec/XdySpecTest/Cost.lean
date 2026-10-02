import XdySpec.Cost
import XdySpecTest.Fixtures

/-!
# Lemma 7 tests

The ghost along each path, and lemma 7, on the function of
`XdySpecTest/Conditioning.lean` whose split on `b` merges beneath its split on
`a`, and on a roll of `2D2` that fans out. The ghost's entries are computed,
so the tests count the pass's worlds among them and compare them with the
paths: the bound is tight while both of `nested`'s splits are open, and loose
where distinct paths reach the same world.
-/

namespace Xdy.Spec.Test.Cost

/--
Read an outcome as an integer, if its location is a register.

# Parameters
- `ℓ`: The location.
- `x`: The outcome.

# Returns
The outcome, if `ℓ` is a register.
-/
private def toInt? : (ℓ : Location) → Content ℓ → Option Int
  | .register _, x => some x
  | _, _ => none

/--
The worlds of the open splits that the paths reach before an instruction:
the distinct outcomes, beside the states, at the end of the paths.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.
- `pc`: The program point.

# Returns
The number of distinct outcomes of the open splits, each a register.
-/
private def worlds (vs : Values) (s : State) (pc : Nat) : Nat :=
  ((ghostPaths vs s pc).map fun p => (vs.openSplits pc).map fun v =>
    toInt? (vs.location v) (p.2 v)).eraseDups.length

/-! ## A split that merges beneath another -/

-- `nested`, from `XdySpecTest/Fixtures.lean`: the plan splits on `a` and on
-- `b = a + x`, merges `b` into `@5` beneath `a`, and then `a` into `@6`.

/-- The values of the function. -/
private abbrev vs : Values := ⟨nested.instructions.toList⟩

/-- Its initial state. -/
private def s₀ : State := .ofOracle nestedInitial

-- The ghost has one entry for each path that reaches each instruction.
#guard (List.range 11).all fun pc =>
  (ghostPaths vs s₀ pc).length == (paths s₀ (vs.instructions.take pc)).length
#guard (List.range 11).map (fun pc => (ghostPaths vs s₀ pc).length) ==
  [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4]

-- The worlds: one, then `a`'s two, then four while `b`'s split is open too,
-- as many as the paths, then `a`'s two, then one.
#guard (List.range 11).map (worlds vs s₀) == [1, 1, 2, 2, 2, 4, 4, 4, 2, 1, 1]

-- Along each path, `a`'s outcome is the value in `@0` when it splits.
#guard (ghostPaths vs s₀ 2).all fun p =>
  toInt? (vs.location 1) (p.2 1) == some (p.1.value (.register 0))

-- Lemma 7, before each instruction and after each step.
example (pc : Nat) (h : pc ≤ 10) :
    (splitOutcomes (forward vs s₀ pc).K (vs.openSplits pc)).support.encard ≤
      (paths s₀ nested.instructions.toList).length :=
  worlds_le_paths nested_valid nested_initialState h
example (pc : Nat) (h : pc < 10) (n : Nat) :
    let r := ((vs.steps pc).take n).foldl (Forward.perform pc)
      ((forward vs s₀ pc).exec pc vs.instructions[pc], vs.openSplits pc)
    (splitOutcomes r.1.K r.2).support.encard ≤
      (paths s₀ nested.instructions.toList).length :=
  worlds_le_paths_of_step nested_valid nested_initialState h n

-- `nested` has four paths.
#guard (paths s₀ nested.instructions.toList).length == 4

/-! ## Paths that reach the same world -/

/-- `{x}@(2D2) + {x} * 2`, which splits on the sum of `2D2`: four paths reach
three worlds, since two of them sum to `3`. -/
private abbrev twoDice : Values := ⟨[
  .rollStandardDice 0 (.immediate 2) (.immediate 2), .sumRollingRecord 0 0,
  .binary .mul 1 (.register 0) (.immediate 2),
  .binary .add 0 (.register 0) (.register 1), .return (.register 0)]⟩

/-- Its initial state. -/
private def s₁ : State := { registers := #[0, 0], records := #[{}] }

#guard twoDice.openSplits 2 == [1]
#guard (ghostPaths twoDice s₁ 2).length == 4
#guard worlds twoDice s₁ 2 == 3

end Xdy.Spec.Test.Cost
