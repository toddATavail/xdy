import Xdy.Interpreter

/-!
# Interpreter tests

Every instruction and its edge cases, argument and external binding, and the
exact distributions of the compiled fixtures in `Fixtures/`, checked against
distributions known in closed form or composed independently from `Dist`.
-/

namespace Xdy.Test

open Xdy

/-! ## Helpers -/

/-- Compare results and errors, for which core provides no `BEq`. -/
private instance {ε α : Type} [BEq ε] [BEq α] : BEq (Except ε α) where
  beq
    | .ok a, .ok b => a == b
    | .error e, .error f => e == f
    | _, _ => false

/--
Run a function, answering its distribution as a sorted list of outcomes and
weights, or the error message.
-/
private def dist (f : Function) (args : List Int := [])
    (externals : List (String × Int) := []) :
    Except String (List (Int × Nat)) :=
  (f.run args externals).map Dist.toList

/-- Run a function, as `dist` does, but reduce the weights to lowest terms. -/
private def reduced (f : Function) (args : List Int := []) :
    Except String (List (Int × Nat)) :=
  (f.run args).map (·.lowestTerms.toList)

/-- Read a fixture's JSON text as a function, or the empty function if it
fails to read, which no test expects. -/
private def fixture (json : String) : Function :=
  (Function.ofJson json).toOption.getD default

/--
A function over two registers and two rolling records, whose parameters are
the first `arity` registers, with the given body.
-/
private def prog (body : List Instruction) (arity : Nat := 0) : Function := {
  parameters := (List.range arity).map (s!"p{·}") |>.toArray
  externals := #[]
  registerCount := 2
  rollingRecordCount := 2
  instructions := body.toArray }

/-- The distribution of the sum of rolling record `0` after `roll`. -/
private def sumOf (roll : Instruction) (arity : Nat := 0) : Function :=
  prog [roll, .sumRollingRecord 0 0, .return (.register 0)] arity

/-- An immediate operand. -/
private def imm (x : Int) : AddressingMode := .immediate x

/-- A register operand. -/
private def reg (r : Nat) : AddressingMode := .register r

/-- The distribution of the sum of `n` dice drawn from `die`, composed from
`Dist` alone. -/
private def sumDice (n : Nat) (die : Dist Int) : Dist Int :=
  n.repeat (fun d => d.bind fun t => die.map (t + ·)) (.pure 0)

/-- The distribution of one die with faces `1` through `k`. -/
private def d (k : Int) : Dist Int := standardDie k

/-- `3D6`, in closed form: the number of ways to roll each total. -/
private def threeD6 : List (Int × Nat) :=
  [(3, 1), (4, 3), (5, 6), (6, 10), (7, 15), (8, 21), (9, 25), (10, 27),
   (11, 27), (12, 25), (13, 21), (14, 15), (15, 10), (16, 6), (17, 3),
   (18, 1)]

/-! ## Lawful equality of states -/

-- Equality is decided structurally, so it is lawful, as the laws of `Dist`
-- over states require.
example : LawfulBEq State := inferInstance
example : LawfulHashable State := inferInstance

/-! ## Ranges -/

#guard dist (sumOf (.rollRange 0 (imm 1) (imm 6))) ==
  .ok [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
#guard dist (sumOf (.rollRange 0 (imm (-2)) (imm 0))) ==
  .ok [(-2, 1), (-1, 1), (0, 1)]
#guard dist (sumOf (.rollRange 0 (imm 4) (imm 4))) == .ok [(4, 1)]
-- An empty range rolls a certain `0`.
#guard dist (sumOf (.rollRange 0 (imm 4) (imm 3))) == .ok [(0, 1)]

/-! ## Standard dice -/

#guard dist (sumOf (.rollStandardDice 0 (imm 3) (imm 6))) == .ok threeD6
#guard dist (sumOf (.rollStandardDice 0 (imm 1) (imm 1))) == .ok [(1, 1)]
-- A count of `0` or less rolls no dice, which sum to `0`.
#guard dist (sumOf (.rollStandardDice 0 (imm 0) (imm 6))) == .ok [(0, 1)]
#guard dist (sumOf (.rollStandardDice 0 (imm (-1)) (imm 6))) == .ok [(0, 1)]
#guard dist (sumOf (.rollStandardDice 0 (imm i32Min) (imm 6))) == .ok [(0, 1)]
-- Dice with `0` or fewer faces each roll `0`.
#guard dist (sumOf (.rollStandardDice 0 (imm 3) (imm 0))) == .ok [(0, 1)]
#guard dist (sumOf (.rollStandardDice 0 (imm 3) (imm (-6)))) == .ok [(0, 1)]
-- Faceless dice still count as results, so dropping them matters.
#guard dist (prog [.rollStandardDice 0 (imm 3) (imm 0),
    .dropLowest 0 (imm 1), .sumRollingRecord 0 0, .return (reg 0)]) ==
  .ok [(0, 1)]

/-! ## Custom dice -/

#guard dist (sumOf (.rollCustomDice 0 (imm 1) [-5, 0, 7])) ==
  .ok [(-5, 1), (0, 1), (7, 1)]
-- Faces are equally likely by position, so a repeated face is likelier.
#guard dist (sumOf (.rollCustomDice 0 (imm 1) [1, 1, 2])) ==
  .ok [(1, 2), (2, 1)]
#guard dist (sumOf (.rollCustomDice 0 (imm 2) [1, 1, 2])) ==
  .ok [(2, 4), (3, 4), (4, 1)]
-- No dice, and dice with no faces.
#guard dist (sumOf (.rollCustomDice 0 (imm 0) [1, 2])) == .ok [(0, 1)]
#guard dist (sumOf (.rollCustomDice 0 (imm 3) [])) == .ok [(0, 1)]

/-! ## Drops -/

/-- Roll `3D4` into rolling record `0`, apply the drops, and sum. -/
private def dropping (drops : List Instruction) : Function :=
  prog ([.rollStandardDice 0 (imm 3) (imm 4)] ++ drops ++
    [.sumRollingRecord 0 0, .return (reg 0)])

-- The highest of `3D4`: `k³ - (k - 1)³` ways for each `k`.
#guard dist (dropping [.dropLowest 0 (imm 2)]) ==
  .ok [(1, 1), (2, 7), (3, 19), (4, 37)]
-- The lowest of `3D4`, the mirror image.
#guard dist (dropping [.dropHighest 0 (imm 2)]) ==
  .ok [(1, 37), (2, 19), (3, 7), (4, 1)]
-- Drops accumulate: two drops of one are one drop of two.
#guard dist (dropping [.dropLowest 0 (imm 1), .dropLowest 0 (imm 1)]) ==
  dist (dropping [.dropLowest 0 (imm 2)])
-- A negative drop drops nothing.
#guard dist (dropping [.dropLowest 0 (imm (-1))]) == dist (dropping [])
-- Dropping more than every result, or every result from both ends, keeps
-- nothing, which sums to `0` in all 64 ways.
#guard dist (dropping [.dropHighest 0 (imm i32Max)]) == .ok [(0, 64)]
#guard dist (dropping [.dropLowest 0 (imm 2), .dropHighest 0 (imm 2)]) ==
  .ok [(0, 64)]
-- Rolling again replaces the record, discarding its drops. Both rolls branch,
-- so the total is 64 times larger, but the probabilities are the same.
#guard reduced (prog [.rollStandardDice 0 (imm 3) (imm 4),
    .dropLowest 0 (imm 3), .rollStandardDice 0 (imm 3) (imm 4),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  reduced (dropping [])

/-! ## Sums, arithmetic and registers -/

-- A rolling record that was never rolled sums to `0`.
#guard dist (prog [.sumRollingRecord 0 1, .return (reg 0)]) == .ok [(0, 1)]
-- A register that was never written reads as `0`.
#guard dist (prog [.return (reg 1)]) == .ok [(0, 1)]
-- Summing leaves the rolling record intact, so it can be summed again.
#guard dist (prog [.rollRange 0 (imm 1) (imm 2), .sumRollingRecord 0 0,
    .sumRollingRecord 1 0, .binary .add 0 (reg 0) (reg 1),
    .return (reg 0)]) ==
  .ok [(2, 1), (4, 1)]

/-- The result of one binary operation on two immediates. -/
private def binary (op : BinaryOp) (a b : Int) :
    Except String (List (Int × Nat)) :=
  dist (prog [.binary op 0 (imm a) (imm b), .return (reg 0)])

#guard binary .add i32Max 1 == .ok [(i32Max, 1)]
#guard binary .sub i32Min 1 == .ok [(i32Min, 1)]
#guard binary .mul i32Max (-2) == .ok [(i32Min, 1)]
#guard binary .div 7 0 == .ok [(0, 1)]
#guard binary .div (-7) 2 == .ok [(-3, 1)]
#guard binary .mod (-7) 2 == .ok [(-1, 1)]
#guard binary .exp (-2) 3 == .ok [(-8, 1)]
#guard binary .max (-2) 3 == .ok [(3, 1)]
#guard dist (prog [.neg 0 (imm i32Min), .return (reg 0)]) == .ok [(i32Max, 1)]
-- Arithmetic applies to every state: `1D3 * 1D3`.
#guard dist (prog [.rollRange 0 (imm 1) (imm 3), .sumRollingRecord 0 0,
    .rollRange 1 (imm 1) (imm 3), .sumRollingRecord 1 1,
    .binary .mul 0 (reg 0) (reg 1), .return (reg 0)]) ==
  .ok [(1, 1), (2, 2), (3, 2), (4, 1), (6, 2), (9, 1)]
-- Saturation happens in every state: `1D2 + i32Max`.
#guard dist (prog [.rollRange 0 (imm 1) (imm 2), .sumRollingRecord 0 0,
    .binary .add 0 (reg 0) (imm i32Max), .return (reg 0)]) ==
  .ok [(i32Max, 2)]

/-! ## Arguments and external variables -/

#guard dist (prog [.return (reg 0)] 1) [42] == .ok [(42, 1)]
#guard dist (prog [.return (reg 1)] 2) [1, 2] == .ok [(2, 1)]
#guard dist (prog [.return (reg 0)] 1) [] ==
  .error "expected 1 arguments, found 0"
#guard dist (prog [.return (reg 0)] 1) [1, 2] ==
  .error "expected 1 arguments, found 2"
#guard dist (prog [.return (reg 0)] 1) [i32Max + 1] ==
  .error "2147483648 is outside the i32 range"
#guard dist (prog [.return (reg 0)]) [] [("z", 1)] ==
  .error "z is not an external variable"
-- An ill-formed function is rejected before it runs.
#guard dist (prog []) == .error "the function has no return"

/-! ## Fixtures -/

-- `7`
#guard dist (fixture (include_str "Fixtures/constant.json")) == .ok [(7, 1)]

-- `[1:6]`
#guard dist (fixture (include_str "Fixtures/range.json")) ==
  .ok [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]

-- `3D6`
#guard dist (fixture (include_str "Fixtures/standard_dice.json")) ==
  .ok threeD6

-- `(1D3)D3`: a dice count from a register, over 81.
#guard dist (fixture (include_str "Fixtures/dynamic_count.json")) ==
  .ok [(1, 9), (2, 12), (3, 16), (4, 12), (5, 12), (6, 10), (7, 6), (8, 3),
       (9, 1)]

-- `{x}@(3D6) + {x}`: one roll, read twice, is twice `3D6`, not `6D6`.
#guard dist (fixture (include_str "Fixtures/shared_register.json")) ==
  .ok (threeD6.map fun (x, w) => (2 * x, w))

-- `4D6 drop lowest`, over 1296.
#guard dist (fixture (include_str "Fixtures/drop_lowest.json")) ==
  .ok [(3, 1), (4, 4), (5, 10), (6, 21), (7, 38), (8, 62), (9, 91),
       (10, 122), (11, 148), (12, 167), (13, 172), (14, 160), (15, 131),
       (16, 94), (17, 54), (18, 21)]

-- `2D[-1, 0, 1, 3, 5]`, over 25.
#guard dist (fixture (include_str "Fixtures/custom_dice.json")) ==
  .ok [(-2, 1), (-1, 2), (0, 3), (1, 2), (2, 3), (3, 2), (4, 4), (5, 2),
       (6, 3), (8, 2), (10, 1)]

-- `{x}: {x}D[1, 3]`, for several counts.
#guard dist (fixture (include_str "Fixtures/dynamic_custom_count.json")) [2]
  == .ok [(2, 1), (4, 2), (6, 1)]
#guard dist (fixture (include_str "Fixtures/dynamic_custom_count.json")) [0]
  == .ok [(0, 1)]
#guard dist (fixture (include_str "Fixtures/dynamic_custom_count.json")) [-3]
  == .ok [(0, 1)]

-- `{n}, {m}: 5D6 drop lowest {n} drop highest {m}`.
/-- The `5D6` drop fixture, with the given drops. -/
private def drops (n m : Int) : Except String (List (Int × Nat)) :=
  dist (fixture (include_str "Fixtures/dynamic_drops.json")) [n, m]
-- The highest of `5D6`: `k⁵ - (k - 1)⁵` ways for each `k`.
#guard drops 4 0 == .ok [(1, 1), (2, 31), (3, 211), (4, 781), (5, 2101),
  (6, 4651)]
-- The lowest of `5D6`, the mirror image.
#guard drops 0 4 == .ok [(1, 4651), (2, 2101), (3, 781), (4, 211), (5, 31),
  (6, 1)]
#guard drops 0 0 == .ok (sumDice 5 (d 6)).toList
#guard drops (-1) (-1) == drops 0 0
#guard drops 3 3 == .ok [(0, 7776)]

-- `3D(2D6)`: a face count from a register.
#guard dist (fixture (include_str "Fixtures/dynamic_faces.json")) ==
  .ok ((sumDice 2 (d 6)).bind fun k => sumDice 3 (d k)).toList

-- `[1D3:2D6]`: range endpoints from registers, empty when `2D6` is less.
#guard dist (fixture (include_str "Fixtures/dynamic_range.json")) ==
  .ok ((d 3).bind fun a => (sumDice 2 (d 6)).bind fun b =>
    if b < a then .pure 0
    else .uniform ((List.range (b - a + 1).toNat).map fun (i : Nat) => a + i)
  ).toList

-- `1D6 + {y}`: bound, rebound, and unbound, which is `0`.
#guard dist (fixture (include_str "Fixtures/external.json")) [] [("y", 10)] ==
  .ok [(11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1)]
#guard dist (fixture (include_str "Fixtures/external.json")) []
    [("y", 10), ("y", -1)] ==
  .ok [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
#guard dist (fixture (include_str "Fixtures/external.json")) ==
  .ok [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]

-- `{x}: {x}D1`, which the optimizer turns into `max(0, {x})`.
#guard dist (fixture (include_str "Fixtures/max.json")) [5] == .ok [(5, 1)]
#guard dist (fixture (include_str "Fixtures/max.json")) [-3] == .ok [(0, 1)]

-- `{a}, {b}: -({a} * {b} - {a} / {b} % 3 ^ {b})`.
#guard dist (fixture (include_str "Fixtures/arithmetic.json")) [7, 2] ==
  .ok [(-11, 1)]
#guard dist (fixture (include_str "Fixtures/arithmetic.json")) [7, 0] ==
  .ok [(0, 1)]
#guard dist (fixture (include_str "Fixtures/arithmetic.json"))
    [i32Min, -1] ==
  .ok [(i32Min + 1, 1)]

/-! ## Docstring examples, verbatim -/

#guard (rollRange 1 3).total == 3
#guard rollRange 3 1 == Dist.pure { results := [0] }
#guard (rollDice 2 fun _ => Dist.uniform [1, 2]).total == 4
#guard rollDice (-1) (fun _ => Dist.uniform [1, 2]) == Dist.pure {}
#guard rollDice 0 (fun _ => standardDie i32Max) == Dist.pure {}
#guard (standardDie 6).total == 6
#guard standardDie 0 == Dist.pure 0
#guard (customDie [1, 1, 2]).weight 1 == 2
#guard customDie [] == Dist.pure 0
#guard BinaryOp.sub.apply 1 2 == -1
#guard BinaryOp.add.apply i32Max 1 == i32Max
-- `1D6`.
#guard (Function.run
    { parameters := #[], externals := #[]
      registerCount := 1, rollingRecordCount := 1
      instructions := #[.rollRange 0 (.immediate 1) (.immediate 6),
        .sumRollingRecord 0 0, .return (.register 0)] }
    [] |>.map Dist.toList)
  matches .ok [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]

end Xdy.Test
