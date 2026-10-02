import Xdy.Plan

/-!
# Plan tests

Which locations are dead, and the plans of small functions, transcribed from
the plan tests in `xdy/src/tests/propagation.rs`, so that the Lean plan and
the Rust plan agree step for step.
-/

namespace Xdy.Test

open Xdy

/-! ## Helpers -/

/-- An immediate operand. -/
private def imm (x : Int) : AddressingMode := .immediate x

/-- A register operand. -/
private def reg (r : Nat) : AddressingMode := .register r

/-- Roll `count` standard dice with `faces` faces into rolling record `d`. -/
private def dice (d : Nat) (count faces : AddressingMode) : Instruction :=
  .rollStandardDice d count faces

/--
The steps of the plan of instructions, as the Rust test helper `schedule`
answers them: each instruction that has steps, by program counter, with its
steps.
-/
private def schedule (is : List Instruction) : List (Nat × List Step) :=
  (List.range is.length).zip (Plan.new is).steps |>.filter (!·.2.isEmpty)

/-- Whether every split of the plan of instructions merges by the end, as the
Rust asserts in debug builds. -/
private def allMerge (is : List Instruction) : Bool :=
  (Values.mk is).openSplits is.length == []

/-! ## Dead locations -/

-- A register that no instruction reads, and that the function does not
-- return, is dead.
example : (Location.register 0).dead (.register 1)
    [.binary .add 1 (.immediate 1) (.immediate 2)] := rfl
-- A register that an instruction overwrites before any reads it is dead,
-- even if the function returns it.
example : (Location.register 0).dead (.register 0)
    [.binary .add 0 (.immediate 1) (.immediate 2)] := rfl
-- A register that an instruction reads is live, even as it overwrites it.
example : (Location.register 0).dead (.register 0)
    [.binary .add 0 (.register 0) (.immediate 2)] = false := rfl
-- A rolling record that a sum reads is live.
example : (Location.record 0).dead (.register 0)
    [.sumRollingRecord 0 0] = false := rfl
-- A rolling record that a roll replaces is dead.
example : (Location.record 0).dead (.register 0)
    [.rollStandardDice 0 (.immediate 1) (.immediate 6),
      .sumRollingRecord 0 0] := rfl
-- A register that the function returns is live.
example : (Location.register 0).dead (.register 0) [] = false := rfl

/-! ## Fan-out -/

-- A random value that two instructions read splits right after it is
-- written, and merges once one value carries its influence.
#guard schedule [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .mul 1 (reg 0) (imm 3),
    .binary .add 2 (reg 0) (reg 1),
    .return (reg 2)] == [
  (1, [.split (.register 0)]),
  (3, [.merge 0 (some (.register 2)) [.register 0, .register 1]])]

-- A value that one instruction reads twice plans nothing.
#guard schedule [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (reg 0),
    .return (reg 1)] == []

-- A fixed value plans nothing.
#guard schedule [
    .binary .add 0 (imm 1) (imm 2),
    .binary .mul 1 (reg 0) (imm 3),
    .binary .add 2 (reg 0) (reg 1),
    .return (reg 2)] == []

/-! ## Coalesced registers -/

-- A plan follows values rather than registers.
#guard schedule [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .mul 1 (reg 0) (imm 3),
    .binary .add 0 (reg 0) (reg 1),
    .return (reg 0)] == [
  (1, [.split (.register 0)]),
  (3, [.merge 0 (some (.register 0)) [.register 1]])]

#guard schedule [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 0 (reg 0) (reg 0),
    .return (reg 0)] == []

/-! ## Rolling records -/

-- A rolling record that two sums read splits.
#guard schedule [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .sumRollingRecord 1 0,
    .binary .add 2 (reg 0) (reg 1),
    .return (reg 2)] == [
  (0, [.split (.record 0)]),
  (3, [.merge 0 (some (.register 2))
    [.record 0, .register 0, .register 1]])]

-- ...even across a drop, which rewrites the record.
#guard schedule [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .dropLowest 0 (imm 1),
    .sumRollingRecord 1 0,
    .binary .sub 2 (reg 0) (reg 1),
    .return (reg 2)] == [
  (0, [.split (.record 0)]),
  (4, [.merge 0 (some (.register 2))
    [.register 0, .record 0, .register 1]])]

-- A plan never merges into a rolling record, whose influence is not yet one
-- value.
#guard schedule [
    dice 0 (imm 1) (imm 4),
    .sumRollingRecord 0 0,
    dice 1 (reg 0) (imm 6),
    .dropLowest 1 (reg 0),
    .sumRollingRecord 1 1,
    .return (reg 1)] == [
  (1, [.split (.register 0)]),
  (4, [.merge 0 (some (.register 1)) [.register 0, .record 1]])]

/-! ## Nested splits -/

-- A split merges beneath a later split that is independent of it. This is
-- the worked example of `propagate_worlds` in `propagation.rs`, with other
-- dice, and of the documentation of `Plan` in `propagation/plan.rs`.
#guard schedule [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    dice 1 (imm 1) (imm 4),
    .sumRollingRecord 1 1,
    .binary .mul 2 (reg 0) (reg 1),
    .binary .add 3 (reg 0) (reg 2),
    .binary .add 4 (reg 1) (reg 3),
    .return (reg 4)] == [
  (1, [.split (.register 0)]),
  (3, [.split (.register 1)]),
  (5, [.merge 0 (some (.register 3)) [.register 0, .register 2]]),
  (6, [.merge 0 (some (.register 4))
    [.register 1, .register 2, .register 3]])]

-- A split waits for a later split that depends on it to merge first.
#guard schedule [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (imm 2),
    .binary .add 3 (reg 1) (imm 1),
    .binary .mul 4 (reg 1) (reg 3),
    .return (reg 4)] == [
  (1, [.split (.register 0)]),
  (2, [.split (.register 1)]),
  (5, [.merge 1 (some (.register 4)) [.register 1, .register 3],
    .merge 0 (some (.register 4))
      [.register 0, .register 1, .register 2, .register 3]])]

/-! ## Final merges -/

-- The split on `@1` may not merge into itself, and the split on `@0` must
-- wait for it, until the answer carries both.
#guard schedule [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (reg 1),
    .return (reg 1)] == [
  (1, [.split (.register 0)]),
  (2, [.split (.register 1)]),
  (4, [.merge 1 (some .answer) [.register 1, .register 2],
    .merge 0 (some .answer) [.register 0, .register 1, .register 2]])]

-- A split on which no live value depends merges into nothing.
#guard schedule [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (imm 2),
    .return (imm 0)] == [
  (1, [.split (.register 0)]),
  (3, [.merge 0 none [.register 0, .register 1, .register 2]])]

-- Every split of every plan above merges by the end.
#guard allMerge [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (reg 1),
    .return (reg 1)]
#guard allMerge [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (imm 2),
    .binary .add 3 (reg 1) (imm 1),
    .binary .mul 4 (reg 1) (reg 3),
    .return (reg 4)]

/-! ## Docstring examples, verbatim -/

example : (Location.register 0).dead (.register 1)
    [.binary .add 0 (.immediate 1) (.immediate 2)] := rfl
example : (Location.register 0).dead (.register 0) [] = false := rfl
#guard (Plan.new [.return (.immediate 7)]).steps = [[]]

end Xdy.Test
