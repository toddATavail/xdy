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

/-! ## Dice -/

/-- A fresh state with two registers and two rolling records. -/
private def fresh : State := { registers := #[0, 0], records := #[{}, {}] }

-- A roll records the die from which it drew, even if it drew nothing.
#guard step fresh (.rollRange 0 (imm 4) (imm 3)) ==
  .pure (fresh.setRecord 0 { results := [0], die := .range 4 3 })
#guard step fresh (.rollStandardDice 0 (imm 0) (imm 6)) ==
  .pure (fresh.setRecord 0 { die := .standard 6 })
#guard step fresh (.rollStandardDice 1 (imm 1) (imm 1)) ==
  .pure (fresh.setRecord 1 { results := [1], die := .range 1 1 })
#guard step fresh (.rollCustomDice 0 (imm 0) [3, 1, 3]) ==
  .pure (fresh.setRecord 0 { die := .custom [1, 3, 3] })
-- Rolling again replaces the die.
#guard ((step fresh (.rollRange 0 (imm 1) (imm 1))).bind
    (step · (.rollStandardDice 0 (imm 1) (imm 1)))) ==
  .pure (fresh.setRecord 0 { results := [1], die := .standard 1 })

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

/-! ## Keeps -/

-- Keeping one of `3D4` drops the other two from the opposite end.
#guard dist (dropping [.keepLowest 0 (imm 1)]) ==
  dist (dropping [.dropHighest 0 (imm 2)])
#guard dist (dropping [.keepHighest 0 (imm 1)]) ==
  dist (dropping [.dropLowest 0 (imm 2)])
-- A keep counts from the results still kept, so this keeps the middle die,
-- as does keeping the lower of the higher two.
#guard dist (dropping [.dropLowest 0 (imm 1), .keepLowest 0 (imm 1)]) ==
  dist (dropping [.dropLowest 0 (imm 1), .dropHighest 0 (imm 1)])
#guard dist (dropping [.keepHighest 0 (imm 2), .keepLowest 0 (imm 1)]) ==
  dist (dropping [.dropLowest 0 (imm 1), .dropHighest 0 (imm 1)])
-- So keeps do not commute with drops: dropping after keeping the highest two
-- keeps one die, but keeping the highest two after dropping keeps two.
#guard dist (dropping [.keepHighest 0 (imm 2), .dropLowest 0 (imm 1)]) ==
  dist (dropping [.dropLowest 0 (imm 2)])
#guard dist (dropping [.dropLowest 0 (imm 1), .keepHighest 0 (imm 2)]) ==
  dist (dropping [.dropLowest 0 (imm 1)])
-- Keeping no more than are kept keeps them all.
#guard dist (dropping [.keepLowest 0 (imm 3)]) == dist (dropping [])
#guard dist (dropping [.keepHighest 0 (imm i32Max)]) == dist (dropping [])
-- A count of `0` or less keeps nothing, which sums to `0` in all 64 ways.
#guard dist (dropping [.keepLowest 0 (imm 0)]) == .ok [(0, 64)]
#guard dist (dropping [.keepHighest 0 (imm i32Min)]) == .ok [(0, 64)]
-- Keeping from a record whose drops cover every result keeps nothing.
#guard dist (dropping [.dropLowest 0 (imm 2), .dropHighest 0 (imm 2),
    .keepLowest 0 (imm 1)]) ==
  .ok [(0, 64)]
-- Rolling again replaces the record, discarding its keeps.
#guard reduced (prog [.rollStandardDice 0 (imm 3) (imm 4),
    .keepLowest 0 (imm 0), .rollStandardDice 0 (imm 3) (imm 4),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  reduced (dropping [])

/-! ## Rerolls -/

/-- Roll `count` dice with `faces` faces into rolling record `0`, apply the
reroll, and sum. -/
private def rerolling (count faces : Int) (p : Predicate) (l : Limit) :
    Function :=
  prog [.rollStandardDice 0 (imm count) (imm faces), .reroll 0 p l,
    .sumRollingRecord 0 0, .return (reg 0)]

/-- The law of a die that a reroll rerolls once if `rerolled` matches it,
composed from `Dist` alone. -/
private def rerolledOnce (die : Dist Int) (rerolled : Int → Bool) :
    Dist Int :=
  die.bind fun x => if rerolled x then die else .pure x

/-- The distribution of `d` reduced to lowest terms, as `reduced` answers. -/
private def lowest (d : Dist Int) : Except String (List (Int × Nat)) :=
  .ok d.lowestTerms.toList

-- `1D6 reroll = 1`: a 1 survives only by matching twice.
#guard reduced (rerolling 1 6 (.eq (imm 1)) (.times (imm 1))) ==
  .ok [(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
-- `1D6 reroll = 1 2 times`: each other face in `36 + 6 + 1` ways of 216.
#guard reduced (rerolling 1 6 (.eq (imm 1)) (.times (imm 2))) ==
  .ok [(1, 1), (2, 43), (3, 43), (4, 43), (5, 43), (6, 43)]
-- A limit of `0` or less rerolls nothing.
#guard reduced (rerolling 1 6 (.eq (imm 1)) (.times (imm 0))) ==
  reduced (sumOf (.rollStandardDice 0 (imm 1) (imm 6)))
#guard reduced (rerolling 1 6 (.eq (imm 1)) (.times (imm i32Min))) ==
  reduced (sumOf (.rollStandardDice 0 (imm 1) (imm 6)))
-- Each die rerolls independently: `2D6 reroll = 1`.
#guard reduced (rerolling 2 6 (.eq (imm 1)) (.times (imm 1))) ==
  lowest (sumDice 2 (rerolledOnce (d 6) (· == 1)))
-- An unlimited reroll answers the conditional law of the die, given that it
-- does not match: `1D6 reroll = 1 ∞ times` and `1D6 reroll ≠ 6 ∞ times`.
#guard reduced (rerolling 1 6 (.eq (imm 1)) .unlimited) ==
  .ok [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
#guard reduced (rerolling 1 6 (.ne (imm 6)) .unlimited) == .ok [(6, 1)]
#guard reduced (rerolling 3 6 (.lt (imm 4)) .unlimited) ==
  lowest (sumDice 3 (.uniform [4, 5, 6]))
-- Its faces are equally likely by position, as the die's are.
#guard reduced (prog [.rollCustomDice 0 (imm 1) [1, 2, 2, 3],
    .reroll 0 (.eq (imm 1)) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(2, 2), (3, 1)]
-- An unlimited reroll whose predicate matches every face crosses out every
-- matching die, which then sums to `0`: `3D1 reroll = 1 ∞ times`, dice
-- without faces, which show only `0`, and an empty range.
#guard dist (rerolling 3 1 (.eq (imm 1)) .unlimited) == .ok [(0, 1)]
#guard dist (rerolling 2 6 (.le (imm 6)) .unlimited) == .ok [(0, 36)]
#guard dist (rerolling 2 0 (.in [0]) .unlimited) == .ok [(0, 1)]
#guard dist (prog [.rollRange 0 (imm 4) (imm 3),
    .reroll 0 (.ge (imm 0)) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(0, 1)]
-- A limited reroll of such dice draws them again, to the same law.
#guard reduced (rerolling 2 1 (.eq (imm 1)) (.times (imm 3))) ==
  .ok [(2, 1)]
#guard reduced (rerolling 2 6 (.le (imm 6)) (.times (imm 2))) ==
  reduced (sumOf (.rollStandardDice 0 (imm 2) (imm 6)))
-- An unlimited reroll that matches nothing rerolls nothing.
#guard reduced (rerolling 2 6 (.in []) .unlimited) ==
  reduced (sumOf (.rollStandardDice 0 (imm 2) (imm 6)))
-- Ranges are half-open or inclusive: `1D6 reroll in 2..5 ∞ times` and
-- `1D6 reroll in 2..=5 ∞ times`; an inverted range matches nothing.
#guard reduced (rerolling 1 6 (.inRange (imm 2) (imm 5)) .unlimited) ==
  .ok [(1, 1), (5, 1), (6, 1)]
#guard reduced (rerolling 1 6 (.inRangeInclusive (imm 2) (imm 5))
    .unlimited) ==
  .ok [(1, 1), (6, 1)]
#guard reduced (rerolling 1 6 (.inRangeInclusive (imm 5) (imm 2))
    .unlimited) ==
  reduced (sumOf (.rollStandardDice 0 (imm 1) (imm 6)))
-- Thresholds and counts may be read from registers.
#guard reduced (prog [.rollStandardDice 0 (imm 1) (imm 6),
    .reroll 0 (.gt (reg 0)) (.times (reg 1)), .sumRollingRecord 0 0,
    .return (reg 0)] 2) [4, 1] ==
  reduced (rerolling 1 6 (.gt (imm 4)) (.times (imm 1)))
-- A reroll considers only the kept dice: `2D2 keep highest 1 reroll = 1`
-- rerolls the higher die, a 1 only if both dice are.
#guard reduced (prog [.rollStandardDice 0 (imm 2) (imm 2),
    .keepHighest 0 (imm 1), .reroll 0 (.eq (imm 1)) (.times (imm 1)),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  .ok [(1, 1), (2, 7)]
-- A drop after a reroll drops from the dice that it left.
#guard reduced (prog [.rollStandardDice 0 (imm 2) (imm 2),
    .reroll 0 (.eq (imm 1)) .unlimited, .dropLowest 0 (imm 1),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  .ok [(2, 1)]

/-- A state whose rolling record `0` holds `[1, 2, 3]`, the lowest dropped,
from a three-sided die. -/
private def dropped : State :=
  fresh.setRecord 0 { results := [1, 2, 3], lowest := 1, die := .standard 3 }

-- A reroll leaves the kept dice, without drops, and its die.
#guard step dropped (.reroll 0 (.in []) .unlimited) ==
  .pure (fresh.setRecord 0 { results := [2, 3], die := .standard 3 })
#guard step dropped (.reroll 0 (.eq (imm 1)) (.times (imm 5))) ==
  .pure (fresh.setRecord 0 { results := [2, 3], die := .standard 3 })
#guard step dropped (.reroll 0 (.ge (imm 1)) .unlimited) ==
  .pure (fresh.setRecord 0 { die := .standard 3 })
-- A record never rolled holds no dice, so a reroll changes nothing.
#guard step fresh (.reroll 0 (.eq (imm 0)) (.times (imm 1))) == .pure fresh
#guard step fresh (.reroll 1 (.eq (imm 0)) .unlimited) == .pure fresh

/-! ## Explodes -/

/-- Roll `count` dice with `faces` faces into rolling record `0`, apply the
explode, and sum. -/
private def exploding (count faces : Int) (p : Predicate) (l : Limit) :
    Function :=
  prog [.rollStandardDice 0 (imm count) (imm faces), .explode 0 p l,
    .sumRollingRecord 0 0, .return (reg 0)]

/-- The law of the dice of a chain that begins with one die, and that an
explode grows up to `limit` times while `grows` matches its last die, in the
order drawn, composed from `Dist` alone. -/
private def chainDice (die : Dist Int) (grows : Int → Bool) :
    Nat → Dist (List Int)
  | 0 => die.map ([·])
  | n + 1 => die.bind fun x =>
    if grows x then (chainDice die grows n).map (x :: ·) else .pure [x]

/-- The law of the total of a chain that begins with one die, as
`chainDice` grows it, which never saturates in these tests. -/
private def chainTotal (die : Dist Int) (grows : Int → Bool) (limit : Nat) :
    Dist Int :=
  (chainDice die grows limit).map (·.foldl (· + ·) 0)

-- `1D6 explode = 6`, in closed form over 36.
#guard reduced (exploding 1 6 (.eq (imm 6)) (.times (imm 1))) ==
  .ok [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (7, 1), (8, 1), (9, 1),
    (10, 1), (11, 1), (12, 1)]
-- Each chain grows independently: `2D6 explode = 6 2 times`.
#guard reduced (exploding 2 6 (.eq (imm 6)) (.times (imm 2))) ==
  lowest (sumDice 2 (chainTotal (d 6) (· == 6) 2))
-- Custom faces are equally likely by position: `1D[1, 2, 2] explode = 2`.
#guard reduced (prog [.rollCustomDice 0 (imm 1) [1, 2, 2],
    .explode 0 (.eq (imm 2)) (.times (imm 1)), .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  lowest (chainTotal (customDie [1, 2, 2]) (· == 2) 1)
-- A limit of `0` or less explodes nothing, and neither does an unlimited
-- explode that matches no die.
#guard reduced (exploding 2 6 (.eq (imm 6)) (.times (imm 0))) ==
  reduced (sumOf (.rollStandardDice 0 (imm 2) (imm 6)))
#guard reduced (exploding 2 6 (.eq (imm 6)) (.times (imm i32Min))) ==
  reduced (sumOf (.rollStandardDice 0 (imm 2) (imm 6)))
#guard reduced (exploding 2 6 (.gt (imm 6)) .unlimited) ==
  reduced (sumOf (.rollStandardDice 0 (imm 2) (imm 6)))
-- An unlimited explode whose predicate matches every face saturates every
-- chain, toward the sign of the mean face, or at `0` if that mean is `0`.
#guard dist (exploding 2 6 (.ge (imm 1)) .unlimited) == .ok [(i32Max, 36)]
#guard dist (prog [.rollCustomDice 0 (imm 2) [-1, -2],
    .explode 0 (.le (imm (-1))) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(i32Min, 4)]
#guard dist (prog [.rollCustomDice 0 (imm 1) [-1, 1],
    .explode 0 (.in [-1, 1]) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(0, 2)]
#guard dist (exploding 2 0 (.in [0]) .unlimited) == .ok [(0, 1)]
-- Any other unlimited explode that grows a chain may draw without bound, so
-- the oracle refuses the function...
#guard dist (exploding 1 6 (.eq (imm 6)) .unlimited) == .error unboundedMessage
-- ...if any state that it reaches grows a chain, here only when `1D2` shows
-- `1`, and the die then shows `6`...
#guard dist (prog [.rollRange 1 (imm 1) (imm 2), .sumRollingRecord 0 1,
    .binary .add 0 (reg 0) (imm 4), .rollStandardDice 0 (imm 1) (imm 6),
    .explode 0 (.gt (reg 0)) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .error unboundedMessage
-- ...but not if no chain that it reaches grows.
#guard dist (prog [.rollStandardDice 0 (imm 2) (imm 6),
    .dropLowest 0 (imm 2), .explode 0 (.eq (imm 6)) .unlimited,
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  .ok [(0, 36)]
-- An explode grows a chain only from its end: `1D2 explode = 2 explode = 2`
-- grows `[2, 1]` no further, though its `2` matches, and so is
-- `1D2 explode = 2 2 times`.
#guard reduced (prog [.rollStandardDice 0 (imm 1) (imm 2),
    .explode 0 (.eq (imm 2)) (.times (imm 1)),
    .explode 0 (.eq (imm 2)) (.times (imm 1)), .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  lowest (chainTotal (d 2) (· == 2) 2)
-- Later drops and keeps count a chain as one die, worth its total:
-- `2D6 explode = 6 keep highest 1`.
#guard reduced (prog [.rollStandardDice 0 (imm 2) (imm 6),
    .explode 0 (.eq (imm 6)) (.times (imm 1)), .keepHighest 0 (imm 1),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  lowest ((chainTotal (d 6) (· == 6) 1).bind fun a =>
    (chainTotal (d 6) (· == 6) 1).map (Max.max a ·))
-- An explode grows only the chains still kept: `2D2 drop highest explode = 2`.
#guard reduced (prog [.rollStandardDice 0 (imm 2) (imm 2),
    .dropHighest 0 (imm 1), .explode 0 (.eq (imm 2)) (.times (imm 1)),
    .sumRollingRecord 0 0, .return (reg 0)]) ==
  lowest ((d 2).bind fun a => (d 2).bind fun b =>
    let x := Min.min a b
    if x == 2 then (d 2).map (x + ·) else .pure x)
-- A reroll rerolls each die of a chain alone: `1D2 explode = 2 2 times
-- reroll = 2` rerolls every `2` of the chain.
#guard reduced (prog [.rollStandardDice 0 (imm 1) (imm 2),
    .explode 0 (.eq (imm 2)) (.times (imm 2)),
    .reroll 0 (.eq (imm 2)) (.times (imm 1)), .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  lowest ((chainDice (d 2) (· == 2) 2).bind fun xs =>
    xs.foldl (fun t x => t.bind fun t =>
        (if x == 2 then d 2 else .pure x).map (t + ·))
      (.pure 0))
-- The replacement of the last die of a chain becomes its last die, from which
-- a later explode grows it: `1D2 explode = 2 reroll = 1 explode = 2`.
#guard reduced (prog [.rollStandardDice 0 (imm 1) (imm 2),
    .explode 0 (.eq (imm 2)) (.times (imm 1)),
    .reroll 0 (.eq (imm 1)) (.times (imm 1)),
    .explode 0 (.eq (imm 2)) (.times (imm 1)), .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  lowest (
    let rerolled (y : Int) : Dist Int := if y == 1 then d 2 else .pure y
    let grown (t y : Int) : Dist Int :=
      if y == 2 then (d 2).map (t + y + ·) else .pure (t + y)
    (d 2).bind fun x =>
      if x == 2 then (d 2).bind fun e => (rerolled e).bind (grown 2)
      else (rerolled x).bind (grown 0))
-- An unlimited reroll that matches every face crosses out every chain that
-- may still grow, whole.
#guard dist (prog [.rollStandardDice 0 (imm 2) (imm 2),
    .explode 0 (.eq (imm 2)) (.times (imm 1)),
    .reroll 0 (.ge (imm 1)) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(0, 16)]
-- Rerolls and explodes leave saturated chains alone.
#guard dist (prog [.rollStandardDice 0 (imm 2) (imm 6),
    .explode 0 (.ge (imm 1)) .unlimited,
    .reroll 0 (.ge (imm 1)) .unlimited,
    .reroll 0 (.ge (imm i32Min)) (.times (imm 1)),
    .explode 0 (.ge (imm i32Min)) (.times (imm 1)),
    .explode 0 (.eq (imm i32Max)) .unlimited, .sumRollingRecord 0 0,
    .return (reg 0)]) ==
  .ok [(i32Max, 36)]

-- An explode leaves the kept chains, without drops, and its die, and a record
-- whose chains are all single dice holds no chains.
#guard step dropped (.explode 0 (.in []) .unlimited) ==
  .pure (fresh.setRecord 0 { results := [2, 3], die := .standard 3 })
#guard step dropped (.explode 0 (.eq (imm 2)) (.times (imm 0))) ==
  .pure (fresh.setRecord 0 { results := [2, 3], die := .standard 3 })
#guard step dropped (.explode 0 (.ge (imm 1)) .unlimited) ==
  .pure (fresh.setRecord 0 {
    results := [i32Max, i32Max], die := .standard 3
    chains := [.saturated i32Max, .saturated i32Max] })
-- A record never rolled holds no dice, so an explode changes nothing.
#guard step fresh (.explode 0 (.eq (imm 0)) (.times (imm 1))) == .pure fresh
#guard step fresh (.explode 1 (.eq (imm 0)) .unlimited) == .pure fresh
-- Only an unlimited explode that does not saturate, but grows some chain
-- still kept, may draw without bound.
#guard unbounded dropped (.explode 0 (.eq (imm 2)) .unlimited)
#guard !unbounded dropped (.explode 0 (.eq (imm 1)) .unlimited)
#guard !unbounded dropped (.explode 0 (.ge (imm 1)) .unlimited)
#guard !unbounded dropped (.explode 0 (.eq (imm 2)) (.times (imm 9)))
#guard !unbounded dropped (.reroll 0 (.eq (imm 2)) .unlimited)

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
#guard binary .min (-2) 3 == .ok [(-2, 1)]
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

-- `4D6 keep highest 3`, which is `4D6 drop lowest`.
#guard dist (fixture (include_str "Fixtures/keep_highest.json")) ==
  dist (fixture (include_str "Fixtures/drop_lowest.json"))

-- `{n}, {m}: 5D6 keep highest {n} drop lowest {m}`.
/-- The `5D6` keep fixture, with the given keep and drop. -/
private def keeps (n m : Int) : Except String (List (Int × Nat)) :=
  dist (fixture (include_str "Fixtures/dynamic_keeps.json")) [n, m]
-- The highest of `5D6`, kept, or kept as the higher of two and then dropped.
#guard keeps 1 0 == drops 4 0
#guard keeps 2 1 == drops 4 0
-- The drop counts from the dice that the keep left kept.
#guard keeps 4 2 == drops 3 0
#guard keeps 5 0 == .ok (sumDice 5 (d 6)).toList
#guard keeps i32Max 0 == keeps 5 0
#guard keeps 0 0 == .ok [(0, 7776)]
#guard keeps (-1) 0 == .ok [(0, 7776)]
#guard keeps 3 3 == .ok [(0, 7776)]

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

-- `{x}: lowest({x}, 3)`, which caps `{x}` at `3`.
#guard dist (fixture (include_str "Fixtures/min.json")) [5] == .ok [(3, 1)]
#guard dist (fixture (include_str "Fixtures/min.json")) [-3] == .ok [(-3, 1)]
#guard dist (fixture (include_str "Fixtures/min.json")) [i32Min] ==
  .ok [(i32Min, 1)]

-- `lowest(highest(1D4, 1D6), 1D8)`: `Max` and `Min` of random registers, over
-- 192.
#guard dist (fixture (include_str "Fixtures/extrema.json")) ==
  .ok [(1, 31), (2, 41), (3, 45), (4, 43), (5, 20), (6, 12)]

-- `{a}, {b}: -({a} * {b} - {a} / {b} % 3 ^ {b})`.
#guard dist (fixture (include_str "Fixtures/arithmetic.json")) [7, 2] ==
  .ok [(-11, 1)]
#guard dist (fixture (include_str "Fixtures/arithmetic.json")) [7, 0] ==
  .ok [(0, 1)]
#guard dist (fixture (include_str "Fixtures/arithmetic.json"))
    [i32Min, -1] ==
  .ok [(i32Min + 1, 1)]

-- `3D4 reroll = 1`.
#guard reduced (fixture (include_str "Fixtures/reroll.json")) ==
  lowest (sumDice 3 (rerolledOnce (d 4) (· == 1)))
-- `2D[1, 3, 5] reroll in [1, 5]`.
#guard reduced (fixture (include_str "Fixtures/reroll_list.json")) ==
  lowest (sumDice 2 (rerolledOnce (customDie [1, 3, 5]) ([1, 5].contains ·)))
-- `1D6 reroll in (1D2)..=(1D3)`.
#guard reduced (fixture (include_str "Fixtures/reroll_range.json")) ==
  lowest ((d 2).bind fun a => (d 3).bind fun b =>
    rerolledOnce (d 6) fun x => a ≤ x && x ≤ b)
-- `{x}: 2D6 reroll ≤ {x} ∞ times`, which crosses out every die once every
-- face matches.
#guard (reduced (fixture (include_str "Fixtures/reroll_unlimited.json")) ·)
    [2] ==
  lowest (sumDice 2 (.uniform [3, 4, 5, 6]))
#guard (reduced (fixture (include_str "Fixtures/reroll_unlimited.json")) ·)
    [0] ==
  lowest (sumDice 2 (d 6))
#guard (reduced (fixture (include_str "Fixtures/reroll_unlimited.json")) ·)
    [6] ==
  .ok [(0, 1)]
-- `{x}: 2D{x} reroll ≤ 6 ∞ times`, which crosses out every die unless the die
-- has a face above 6.
#guard (reduced (fixture (include_str "Fixtures/reroll_cross_out.json")) ·)
    [8] ==
  lowest (sumDice 2 (.uniform [7, 8]))
#guard (reduced (fixture (include_str "Fixtures/reroll_cross_out.json")) ·)
    [6] ==
  .ok [(0, 1)]
#guard (reduced (fixture (include_str "Fixtures/reroll_cross_out.json")) ·)
    [0] ==
  .ok [(0, 1)]

/-- The distribution of `n` dice drawn from `die`, each as a sorted list,
composed from `Dist` alone. -/
private def sortedDice (n : Nat) (die : Dist Int) : Dist (List Int) :=
  n.repeat (fun d => d.bind fun xs => die.map fun x =>
    (x :: xs).mergeSort (· ≤ ·)) (.pure [])

-- `4D6 keep highest 3 reroll < 3 2 times`: drop the lowest of four sorted
-- dice, then reroll each of the rest independently, composed from `Dist`.
#guard reduced (fixture (include_str "Fixtures/reroll_after_keep.json")) ==
  lowest ((sortedDice 4 (d 6)).bind fun xs =>
    (xs.drop 1).foldl (fun t x =>
        t.bind fun t =>
          (if x < 3 then rerolledOnce (d 6) (· < 3) else .pure x).map (t + ·))
      (.pure 0))

-- `2D4 explode = 4`.
#guard reduced (fixture (include_str "Fixtures/explode.json")) ==
  lowest (sumDice 2 (chainTotal (d 4) (· == 4) 1))
-- `1D4 explode = 4 3 times`, a range, in closed form over 256: a chain ends
-- below `4` after one, two or three `4`s, or after three `4`s with any face.
#guard reduced (fixture (include_str "Fixtures/explode_limited.json")) ==
  .ok [(1, 64), (2, 64), (3, 64), (5, 16), (6, 16), (7, 16), (9, 4), (10, 4),
    (11, 4), (13, 1), (14, 1), (15, 1), (16, 1)]
-- `{x}: 2D6 explode ≥ {x} ∞ times`, which saturates if every face matches,
-- grows nothing if none does, and is refused otherwise.
#guard (reduced (fixture (include_str "Fixtures/explode_saturate.json")) ·)
    [1] ==
  .ok [(i32Max, 1)]
#guard (reduced (fixture (include_str "Fixtures/explode_saturate.json")) ·)
    [i32Min] ==
  .ok [(i32Max, 1)]
#guard (reduced (fixture (include_str "Fixtures/explode_saturate.json")) ·)
    [7] ==
  lowest (sumDice 2 (d 6))
#guard (reduced (fixture (include_str "Fixtures/explode_saturate.json")) ·)
    [6] ==
  .error unboundedMessage
-- `3D4 keep highest 2 explode = 4`: drop the lowest of three sorted dice, then
-- explode each of the rest independently.
#guard reduced (fixture (include_str "Fixtures/explode_after_keep.json")) ==
  lowest ((sortedDice 3 (d 4)).bind fun xs =>
    (xs.drop 1).foldl (fun t x =>
        t.bind fun t =>
          (if x == 4 then (d 4).map (x + ·) else .pure x).map (t + ·))
      (.pure 0))
-- `2D3 explode = 3 reroll = 1`: reroll each die of each chain independently.
#guard reduced (fixture (include_str "Fixtures/reroll_after_explode.json")) ==
  lowest (sumDice 2 (
    let rerolled (y : Int) : Dist Int := if y == 1 then d 3 else .pure y
    (d 3).bind fun x =>
      if x == 3 then (d 3).bind fun e => (rerolled e).map (3 + ·)
      else rerolled x))

/-- Whether a chain, as its dice in the order drawn, precedes another in the
order of drops and keeps, or equals it: by total, then by its dice in
ascending order, compared lexicographically, then by its last die, as
`cross_out` in `primitives.rs` orders them, written apart from `Chain.le`. -/
private def precedes (a b : List Int) : Bool :=
  let key (c : List Int) := (c.foldl (· + ·) 0, c.mergeSort (· ≤ ·), c.getLast?)
  let (s, xs, x) := key a
  let (t, ys, y) := key b
  s < t || s == t && (xs < ys || xs == ys && (x == y || compare x y == .lt))

-- `2D4 explode ≥ 3 drop lowest reroll = 2`: chains of equal totals, such as
-- `[3, 2]` and `[4, 1]`, are ordered by their dice, so the drop crosses out
-- `[4, 1]`, and the reroll then rerolls the `2` of `[3, 2]`.
#guard reduced (fixture (include_str "Fixtures/explode_ties.json")) ==
  lowest ((chainDice (d 4) (· ≥ 3) 1).bind fun a =>
    (chainDice (d 4) (· ≥ 3) 1).bind fun b =>
      (if precedes a b then b else a).foldl (fun t x =>
          t.bind fun t => (if x == 2 then d 4 else .pure x).map (t + ·))
        (.pure 0))

/-! ## Docstring examples, verbatim -/

#guard (rollRange 1 3).total == 3
#guard rollRange 3 1 == Dist.pure { results := [0], die := .range 3 1 }
#guard (rollDice 2 fun _ => Dist.uniform [1, 2]).total == 4
#guard rollDice (-1) (fun _ => Dist.uniform [1, 2]) == Dist.pure {}
#guard rollDice 0 (fun _ => standardDie i32Max) == Dist.pure {}
#guard (standardDie 6).total == 6
#guard standardDie 0 == Dist.pure 0
#guard (customDie [1, 1, 2]).weight 1 == 2
#guard customDie [] == Dist.pure 0
#guard redraw (.within 1 1) (.uniform [1, 2]) (.pure 1) == .uniform [1, 2]
#guard redraw (.within 1 1) (.uniform [1, 2]) (.pure 2) == .pure 2
#guard (limitedReplacement (.within 1 1) (.uniform [1, 2]) 2).toList
  == [(1, 1), (2, 3)]
#guard (rerollKept (.standard 2) [1, 2] (· == 1) (.pure (.uniform [1, 2]))
  |>.map (·.results)).toList == [([1, 2], 1), ([2, 2], 1)]
#guard ((reroll { results := [1, 2], die := .standard 2 } (.within 1 1)
    none).map (·.results)).toList == [([2, 2], 1)]
#guard reroll { results := [1, 1], die := .standard 1 } (.within 1 1) none
  == .pure { die := .standard 1 }
#guard reroll { results := [1, 2], lowest := 1, die := .standard 2 }
    (.within 1 1) (some 3)
  == .pure { results := [2], die := .standard 2 }
#guard (rerollChain (· == 1) (.pure (.uniform [1, 2])) (.live [1, 3] 1)
  |>.map (·.total)).toList == [(5, 1), (6, 2), (7, 1)]
#guard rerollChain (fun _ => true) (.pure (.uniform [1, 2])) (.saturated 0)
  == .pure (.saturated 0)
#guard ((rerollChains (.standard 2) [.live [2] 1] (· == 1)
    (.pure (.uniform [1, 2]))).map (·.results)).toList == [([3], 1), ([4], 1)]
#guard (rerollEach { results := [1, 2], die := .standard 2 } (· == 1)
    (.pure (.uniform [1, 2])) |>.map (·.results)).toList
  == [([1, 2], 1), ([2, 2], 1)]
#guard ((extension (.within 2 2) (.uniform [1, 2]) 2).map
    fun (drawn, y) => drawn ++ [y]).toList
  == [([1], 2), ([2, 1], 1), ([2, 2], 1)]
#guard explodeChains (.standard 6) [.single 6, .single 3] (·.last == 6)
    (.pure (.pure ([], 2)))
  == .pure { results := [3, 8], die := .standard 6
             chains := [.single 3, .live [6] 2] }
#guard explode { results := [6, 6], lowest := 1, die := .standard 6 }
    (.within 1 6) none
  == .pure { results := [i32Max], die := .standard 6
             chains := [.saturated i32Max] }
#guard ((explode { results := [1, 2], die := .standard 2 } (.within 2 2)
    (some 1)).map (·.results)).toList == [([1, 3], 1), ([1, 4], 1)]
#guard unbounded
    { registers := #[], records := #[{ results := [6], die := .standard 6 }] }
    (.explode 0 (.eq (.immediate 6)) .unlimited)
#guard !unbounded
    { registers := #[], records := #[{ results := [5], die := .standard 6 }] }
    (.explode 0 (.eq (.immediate 6)) .unlimited)
#guard (stepAll (.pure { registers := #[0], records := #[{}] })
    (.rollRange 0 (.immediate 1) (.immediate 2))).toOption.map (·.total)
  == some 2
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
