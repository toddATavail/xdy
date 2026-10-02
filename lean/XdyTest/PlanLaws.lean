import Xdy.PlanLaws

/-!
# Plan law tests

The conclusions of the laws of plans, checked on every instruction and value
of the functions whose plans `XdyTest/Plan.lean` tests, and the two places
where `dead_eq` needs its hypotheses.
-/

namespace Xdy.Test.PlanLaws

open Xdy

/-! ## Helpers -/

/-- An immediate operand. -/
private def imm (x : Int) : AddressingMode := .immediate x

/-- A register operand. -/
private def reg (r : Nat) : AddressingMode := .register r

/-- Roll `count` standard dice with `faces` faces into rolling record `d`. -/
private def dice (d : Nat) (count faces : AddressingMode) : Instruction :=
  .rollStandardDice d count faces

/-- Every live value occupies its location, and every live value that fans
out is an open split. -/
private def liveLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range is.length).all fun pc =>
    (vs.live pc).all fun v =>
      vs.writer (pc + 1) (vs.location v) == some v &&
        (!vs.fansOut v || (vs.openSplits (pc + 1)).contains v)

/-- Every random value that an instruction reads is an open split before it,
or is read for the last time. -/
private def operandLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range is.length).all fun pc =>
    (vs.reads pc).all fun u =>
      !(vs.random u && vs.isLiveAfter u pc) ||
        (vs.openSplits pc).contains u

/-- An instruction that overwrites a live value reads it. -/
private def overwriteLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range (is.length - 1)).all fun pc =>
    match vs.writer (pc + 1) (vs.location (pc + 1)) with
    | some v => !vs.isLiveAfter v pc || (vs.reads (pc + 1)).contains v
    | none => true

/-- Every value depends only on earlier values that fan out. -/
private def dependsOnLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range is.length).all fun v =>
    (vs.dependsOn v).all fun w => vs.fansOut w && w ≤ v

/-- A register or rolling record is dead exactly when its value is not
live, after every instruction but the last. -/
private def deadLaws (is : List Instruction) (src : AddressingMode) : Bool :=
  let vs := Values.mk is
  (List.range (is.length - 1)).all fun pc =>
    (List.range (pc + 1)).all fun v =>
      let ℓ := vs.location v
      ℓ == .answer || vs.writer (pc + 1) ℓ != some v ||
        ℓ.dead src (is.drop (pc + 1)) == !vs.isLiveAfter v pc

/-- Every value live before an instruction occupies its location, alone; the
instruction reads only such values; those that it does not read stay live
after it; and every value live after it, but its own, was live before it,
and is not in the location that it writes. -/
private def liveBeforeLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range (is.length + 1)).all (fun pc =>
    ((vs.liveBefore pc).map vs.location).Nodup &&
      (vs.liveBefore pc).all fun v =>
        vs.writer pc (vs.location v) == some v) &&
  (List.range is.length).all fun pc =>
    (vs.reads pc).all (vs.liveBefore pc).contains &&
      (vs.liveBefore pc).all (fun v =>
        (vs.reads pc).contains v || (vs.live pc).contains v) &&
      (vs.live pc).all (fun v =>
        v == pc || (vs.liveBefore pc).contains v &&
          vs.location v != vs.location pc) &&
      (!vs.fansOut pc || (vs.live pc).contains pc)

/-- Dependence is transitive, and the open splits ascend, none depending on a
split that may merge but itself. -/
private def stackLaws (is : List Instruction) : Bool :=
  let vs := Values.mk is
  (List.range is.length).all (fun v =>
    (vs.dependsOn v).all fun u =>
      (vs.dependsOn u).all (vs.dependsOn v).contains) &&
  (List.range (is.length + 1)).all (fun pc =>
    (vs.openSplits pc).Pairwise (· < ·)) &&
  (List.range is.length).all fun pc =>
    let splits :=
      if vs.fansOut pc then vs.openSplits pc ++ [pc] else vs.openSplits pc
    (List.range splits.length).all fun k =>
      (vs.survivor splits k (vs.live pc)).isNone ||
        splits.all fun j =>
          !(vs.dependsOn j).contains (splits.getD k 0) || j == splits.getD k 0

/-- Every law above holds of a function that returns `src`. -/
private def allLaws (is : List Instruction) (src : AddressingMode) : Bool :=
  is.all (·.operandsAreValues) && liveLaws is && operandLaws is &&
    overwriteLaws is && dependsOnLaws is && deadLaws is src &&
    liveBeforeLaws is && stackLaws is

/-! ## The functions of the plan tests -/

#guard allLaws [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .mul 1 (reg 0) (imm 3),
    .binary .add 2 (reg 0) (reg 1),
    .return (reg 2)] (reg 2)
#guard allLaws [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .binary .mul 1 (reg 0) (imm 3),
    .binary .add 0 (reg 0) (reg 1),
    .return (reg 0)] (reg 0)
#guard allLaws [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .sumRollingRecord 1 0,
    .binary .add 2 (reg 0) (reg 1),
    .return (reg 2)] (reg 2)
#guard allLaws [
    dice 0 (imm 3) (imm 6),
    .sumRollingRecord 0 0,
    .dropLowest 0 (imm 1),
    .sumRollingRecord 1 0,
    .binary .sub 2 (reg 0) (reg 1),
    .return (reg 2)] (reg 2)
#guard allLaws [
    dice 0 (imm 1) (imm 4),
    .sumRollingRecord 0 0,
    dice 1 (reg 0) (imm 6),
    .dropLowest 1 (reg 0),
    .sumRollingRecord 1 1,
    .return (reg 1)] (reg 1)
#guard allLaws [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    dice 1 (imm 1) (imm 4),
    .sumRollingRecord 1 1,
    .binary .mul 2 (reg 0) (reg 1),
    .binary .add 3 (reg 0) (reg 2),
    .binary .add 4 (reg 1) (reg 3),
    .return (reg 4)] (reg 4)
#guard allLaws [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (imm 2),
    .binary .add 3 (reg 1) (imm 1),
    .binary .mul 4 (reg 1) (reg 3),
    .return (reg 4)] (reg 4)
#guard allLaws [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (reg 1),
    .return (reg 1)] (reg 1)
#guard allLaws [
    dice 0 (imm 1) (imm 6),
    .sumRollingRecord 0 0,
    .binary .add 1 (reg 0) (imm 1),
    .binary .mul 2 (reg 0) (imm 2),
    .return (imm 0)] (imm 0)

/-! ## Where `dead_eq` needs its hypotheses -/

-- After the last instruction, the returned register is not live, since no
-- instruction reads it, but `Location.dead` reads it once more.
#guard
  let is := [dice 0 (imm 1) (imm 6), .sumRollingRecord 0 0, .return (reg 0)]
  !(Values.mk is).isLiveAfter 1 2 &&
    !(Location.register 0).dead (reg 0) (is.drop 3)

-- A rolling record as an operand is a source, as the Rust reads it, but
-- `Location.readBy` does not read it, as the specification reads it as `0`.
#guard
  let i : Instruction := .binary .add 0 (.rollingRecord 0) (imm 1)
  !i.operandsAreValues && (i.sources.contains (.record 0)) &&
    !(Location.record 0).readBy i

end Xdy.Test.PlanLaws
