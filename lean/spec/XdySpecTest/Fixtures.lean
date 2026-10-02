import XdySpec.Semantics

/-!
# Test fixtures

Functions and laws that more than one test module uses: the law of a roll of
`1D2`, and the function whose split on `b` merges beneath its split on `a`.
-/

namespace Xdy.Spec.Test

/-! ## A roll of `1D2` -/

/-- The law of a roll of `1D2`. -/
noncomputable def roll : PMF Int :=
  PMF.ofMultiset (Finset.Icc (1 : Int) 2).val (by simp)

/-- The rolls are `1` and `2`. -/
theorem mem_roll {x : Int} (hx : x ∈ roll.support) : x = 1 ∨ x = 2 := by
  simp [roll] at hx
  omega

/-- The roll of a range from `1` to `2`, which is how the compiler writes
`1D2`. -/
theorem rollRange_one_two :
    rollRange 1 2 = roll.map fun x => { results := [x] } := by
  simp [rollRange, roll]

/-- A record of one result sums to it, clamped. -/
theorem sum_single (x : Int) : ({ results := [x] } : Record).sum = add 0 x := by
  simp [Record.sum, Record.kept, Record.sorted]

/-! ## A split that merges beneath another

`a + b` and friends, where `a` and `b` each roll `1D2`, as
`XdySpecTest/Conditioning.lean` writes them:

```text
0  ⚅0 <- roll range 1:2
1  @0 <- sum rolling record ⚅0
2  ⚅1 <- roll range 1:2
3  @1 <- sum rolling record ⚅1
4  @2 <- @0 + @1
5  @3 <- @0 * 2
6  @4 <- @2 * 3
7  @5 <- @2 + @4
8  @6 <- @3 + @5
9  return @6
```

The plan splits on `a` and on `b = a + x`, where `x` is the second roll,
merges `b` into `@5` beneath `a`, and then `a` into `@6`.
-/

/-- The function. -/
def nested : Function where
  parameters := #[]
  externals := #[]
  registerCount := 7
  rollingRecordCount := 2
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 2), .sumRollingRecord 0 0,
    .rollRange 1 (.immediate 1) (.immediate 2), .sumRollingRecord 1 1,
    .binary .add 2 (.register 0) (.register 1),
    .binary .mul 3 (.register 0) (.immediate 2),
    .binary .mul 4 (.register 2) (.immediate 3),
    .binary .add 5 (.register 2) (.register 4),
    .binary .add 6 (.register 3) (.register 5),
    .return (.register 6)]

/-- Its initial state, with every register `0` and every record empty. -/
def nestedInitial : Xdy.State :=
  { registers := #[0, 0, 0, 0, 0, 0, 0], records := #[{}, {}] }

/-- The function is well formed. -/
theorem nested_valid : nested.validate = .ok () := by decide +kernel

/-- Its initial state, without arguments or external variables. -/
theorem nested_initialState : nested.initialState [] [] = .ok nestedInitial :=
  rfl

end Xdy.Spec.Test
