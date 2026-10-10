import Xdy.Ir

/-!
# IR tests

Reading the JSON that Rust's `serde` feature writes for a compiled `Function`,
and rejecting JSON that describes no well-formed function. The fixtures in
`Fixtures/` are real compiler output; see `Fixtures/README.md`.
-/

namespace Xdy.Test

open Xdy

/-- Read the JSON text as a function, or answer `none` if it is rejected. -/
private def read (s : String) : Option Function := (Function.ofJson s).toOption

/-- Answer whether the JSON text reads as a well-formed function. -/
private def reads (s : String) : Bool := Function.ofJson s matches .ok _

/--
Answer whether the JSON text is rejected with a message containing the given
text.
-/
private def rejects (s : String) (message : String) : Bool :=
  match Function.ofJson s with
  | .ok _ => false
  | .error e => (e.splitOn message).length > 1

/--
Wrap a list of instructions, as JSON text, in a function with two registers
and one rolling record, as JSON text.
-/
private def body (instructions : String) : String :=
  "{\"parameters\": [], \"externals\": [], \"register_count\": 2, " ++
  "\"rolling_record_count\": 1, \"instructions\": [" ++ instructions ++ "]}"

/-- JSON text for `return 0`. -/
private def ret : String := "{\"Return\": {\"src\": {\"Immediate\": 0}}}"

/-! ## Every fixture reads -/

#guard reads (include_str "Fixtures/arithmetic.json")
#guard reads (include_str "Fixtures/constant.json")
#guard reads (include_str "Fixtures/custom_dice.json")
#guard reads (include_str "Fixtures/drop_lowest.json")
#guard reads (include_str "Fixtures/dynamic_count.json")
#guard reads (include_str "Fixtures/dynamic_custom_count.json")
#guard reads (include_str "Fixtures/dynamic_drops.json")
#guard reads (include_str "Fixtures/dynamic_faces.json")
#guard reads (include_str "Fixtures/dynamic_keeps.json")
#guard reads (include_str "Fixtures/dynamic_range.json")
#guard reads (include_str "Fixtures/external.json")
#guard reads (include_str "Fixtures/extrema.json")
#guard reads (include_str "Fixtures/keep_highest.json")
#guard reads (include_str "Fixtures/max.json")
#guard reads (include_str "Fixtures/min.json")
#guard reads (include_str "Fixtures/range.json")
#guard reads (include_str "Fixtures/explode.json")
#guard reads (include_str "Fixtures/explode_after_keep.json")
#guard reads (include_str "Fixtures/explode_limited.json")
#guard reads (include_str "Fixtures/explode_saturate.json")
#guard reads (include_str "Fixtures/explode_ties.json")
#guard reads (include_str "Fixtures/reroll.json")
#guard reads (include_str "Fixtures/reroll_after_explode.json")
#guard reads (include_str "Fixtures/reroll_after_keep.json")
#guard reads (include_str "Fixtures/reroll_cross_out.json")
#guard reads (include_str "Fixtures/reroll_list.json")
#guard reads (include_str "Fixtures/reroll_range.json")
#guard reads (include_str "Fixtures/reroll_unlimited.json")
#guard reads (include_str "Fixtures/shared_register.json")
#guard reads (include_str "Fixtures/standard_dice.json")

/-! ## Fixtures read exactly -/

-- `{a}, {b}: -({a} * {b} - {a} / {b} % 3 ^ {b})`: every arithmetic operation
-- but `Max`, and parameters.
#guard read (include_str "Fixtures/arithmetic.json") == some {
  parameters := #["a", "b"], externals := #[]
  registerCount := 3, rollingRecordCount := 0
  instructions := #[
    .binary .mul 2 (.register 0) (.register 1),
    .binary .div 0 (.register 0) (.register 1),
    .binary .exp 1 (.immediate 3) (.register 1),
    .binary .mod 0 (.register 0) (.register 1),
    .binary .sub 0 (.register 2) (.register 0),
    .neg 0 (.register 0),
    .return (.register 0)] }

-- `(1D3)D3`: a range, standard dice with a dynamic count, and sums.
#guard read (include_str "Fixtures/dynamic_count.json") == some {
  parameters := #[], externals := #[]
  registerCount := 1, rollingRecordCount := 2
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 3),
    .sumRollingRecord 0 0,
    .rollStandardDice 1 (.register 0) (.immediate 3),
    .sumRollingRecord 0 1,
    .return (.register 0)] }

-- `2D[-1, 0, 1, 3, 5]`: custom dice with negative faces.
#guard read (include_str "Fixtures/custom_dice.json") == some {
  parameters := #[], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollCustomDice 0 (.immediate 2) [-1, 0, 1, 3, 5],
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `{n}, {m}: 5D6 drop lowest {n} drop highest {m}`: dynamic drops.
#guard read (include_str "Fixtures/dynamic_drops.json") == some {
  parameters := #["n", "m"], externals := #[]
  registerCount := 2, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 5) (.immediate 6),
    .dropLowest 0 (.register 0),
    .dropHighest 0 (.register 1),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `{n}, {m}: 5D6 keep highest {n} drop lowest {m}`: a dynamic keep, then a
-- dynamic drop.
#guard read (include_str "Fixtures/dynamic_keeps.json") == some {
  parameters := #["n", "m"], externals := #[]
  registerCount := 2, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 5) (.immediate 6),
    .keepHighest 0 (.register 0),
    .dropLowest 0 (.register 1),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- A keep of the lowest, which no fixture holds.
#guard read
    (body ("{\"KeepLowest\": {\"dest\": 0, \"count\": {\"Register\": 1}}}, " ++
      ret)) == some {
  parameters := #[], externals := #[]
  registerCount := 2, rollingRecordCount := 1
  instructions := #[.keepLowest 0 (.register 1), .return (.immediate 0)] }

-- `{x}: {x}D1`: the optimizer's `Max`.
#guard read (include_str "Fixtures/max.json") == some {
  parameters := #["x"], externals := #[]
  registerCount := 1, rollingRecordCount := 0
  instructions := #[
    .binary .max 0 (.immediate 0) (.register 0),
    .return (.register 0)] }

-- `{x}: lowest({x}, 3)`: `Min`.
#guard read (include_str "Fixtures/min.json") == some {
  parameters := #["x"], externals := #[]
  registerCount := 1, rollingRecordCount := 0
  instructions := #[
    .binary .min 0 (.immediate 3) (.register 0),
    .return (.register 0)] }

-- `1D6 + {y}`: an external variable, in the register after the parameters.
#guard read (include_str "Fixtures/external.json") == some {
  parameters := #[], externals := #["y"]
  registerCount := 2, rollingRecordCount := 1
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 6),
    .sumRollingRecord 1 0,
    .binary .add 0 (.register 0) (.register 1),
    .return (.register 0)] }

-- `3D4 reroll = 1`: a reroll with a threshold and a count.
#guard read (include_str "Fixtures/reroll.json") == some {
  parameters := #[], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 3) (.immediate 4),
    .reroll 0 (.eq (.immediate 1)) (.times (.immediate 1)),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `{x}: 2D{x} reroll ≤ 6 ∞ times`: an unlimited reroll, which `serde` writes
-- as a bare string.
#guard read (include_str "Fixtures/reroll_cross_out.json") == some {
  parameters := #["x"], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 2) (.register 0),
    .reroll 0 (.le (.immediate 6)) .unlimited,
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `2D[1, 3, 5] reroll in [1, 5]`: a list.
#guard read (include_str "Fixtures/reroll_list.json") == some {
  parameters := #[], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollCustomDice 0 (.immediate 2) [1, 3, 5],
    .reroll 0 (.in [1, 5]) (.times (.immediate 1)),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `1D6 reroll in (1D2)..=(1D3)`: an inclusive range whose endpoints are
-- registers.
#guard read (include_str "Fixtures/reroll_range.json") == some {
  parameters := #[], externals := #[]
  registerCount := 2, rollingRecordCount := 3
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 6),
    .rollRange 1 (.immediate 1) (.immediate 2),
    .sumRollingRecord 0 1,
    .rollRange 2 (.immediate 1) (.immediate 3),
    .sumRollingRecord 1 2,
    .reroll 0 (.inRangeInclusive (.register 0) (.register 1))
      (.times (.immediate 1)),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `2D4 explode = 4`: an explode with a threshold and a count.
#guard read (include_str "Fixtures/explode.json") == some {
  parameters := #[], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 2) (.immediate 4),
    .explode 0 (.eq (.immediate 4)) (.times (.immediate 1)),
    .sumRollingRecord 0 0,
    .return (.register 0)] }

-- `{x}: 2D6 explode ≥ {x} ∞ times`: an unlimited explode, whose threshold is a
-- parameter.
#guard read (include_str "Fixtures/explode_saturate.json") == some {
  parameters := #["x"], externals := #[]
  registerCount := 1, rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 2) (.immediate 6),
    .explode 0 (.ge (.register 0)) .unlimited,
    .sumRollingRecord 0 0,
    .return (.register 0)] }

/--
JSON text for a reroll of rolling record `0` with the given predicate and
limit, as JSON text, then `return 0`.
-/
private def rerolling (predicate limit : String) : String :=
  body ("{\"Reroll\": {\"dest\": 0, \"predicate\": " ++ predicate ++
    ", \"limit\": " ++ limit ++ "}}, " ++ ret)

/-- The instructions of a function read from JSON text, if it reads. -/
private def instructions (s : String) : Option (Array Instruction) :=
  (read s).map (·.instructions)

-- Every predicate and limit, which no fixture holds.
#guard instructions (rerolling "{\"Ne\": {\"Register\": 1}}"
    "{\"Times\": {\"Register\": 0}}") ==
  some #[.reroll 0 (.ne (.register 1)) (.times (.register 0)),
    .return (.immediate 0)]
#guard instructions (rerolling "{\"Lt\": {\"Immediate\": -2}}"
    "\"Unlimited\"") ==
  some #[.reroll 0 (.lt (.immediate (-2))) .unlimited, .return (.immediate 0)]
#guard instructions (rerolling "{\"Gt\": {\"Immediate\": 4}}"
    "\"Unlimited\"") ==
  some #[.reroll 0 (.gt (.immediate 4)) .unlimited, .return (.immediate 0)]
#guard instructions (rerolling "{\"Ge\": {\"Immediate\": 4}}"
    "\"Unlimited\"") ==
  some #[.reroll 0 (.ge (.immediate 4)) .unlimited, .return (.immediate 0)]
#guard instructions (rerolling "{\"In\": []}" "\"Unlimited\"") ==
  some #[.reroll 0 (.in []) .unlimited, .return (.immediate 0)]
#guard instructions (rerolling
    ("{\"InRange\": {\"start\": {\"Immediate\": 1}, " ++
      "\"end\": {\"Register\": 1}}}")
    "{\"Times\": {\"Immediate\": 2}}") ==
  some #[.reroll 0 (.inRange (.immediate 1) (.register 1))
      (.times (.immediate 2)),
    .return (.immediate 0)]

/--
JSON text for an explode of rolling record `0` with the given predicate and
limit, as JSON text, then `return 0`.
-/
private def exploding (predicate limit : String) : String :=
  body ("{\"Explode\": {\"dest\": 0, \"predicate\": " ++ predicate ++
    ", \"limit\": " ++ limit ++ "}}, " ++ ret)

-- An explode takes every predicate and limit that a reroll does.
#guard instructions (exploding "{\"Le\": {\"Register\": 1}}"
    "{\"Times\": {\"Register\": 0}}") ==
  some #[.explode 0 (.le (.register 1)) (.times (.register 0)),
    .return (.immediate 0)]
#guard instructions (exploding
    ("{\"InRangeInclusive\": {\"start\": {\"Immediate\": -1}, " ++
      "\"end\": {\"Register\": 1}}}")
    "\"Unlimited\"") ==
  some #[.explode 0 (.inRangeInclusive (.immediate (-1)) (.register 1))
      .unlimited,
    .return (.immediate 0)]
#guard instructions (exploding "{\"In\": [6, -2]}" "\"Unlimited\"") ==
  some #[.explode 0 (.in [6, -2]) .unlimited, .return (.immediate 0)]

/-! ## Malformed JSON is rejected -/

#guard !reads "not json"
#guard rejects (body ("{\"Nop\": {}}, " ++ ret)) "unknown instruction Nop"
#guard rejects (body "{\"Return\": {\"src\": {\"Stack\": 0}}}")
  "unknown addressing mode Stack"
#guard rejects (body "{\"Return\": {}, \"Neg\": {}}")
  "expected an object with one field"
#guard !reads (body "{\"Return\": {}}")
#guard rejects (rerolling "{\"Odd\": {\"Immediate\": 1}}" "\"Unlimited\"")
  "unknown predicate Odd"
#guard !reads (rerolling "{\"Eq\": {\"Immediate\": 1}}" "\"Forever\"")
#guard rejects (rerolling "{\"Eq\": {\"Immediate\": 1}}"
    "{\"Never\": {\"Immediate\": 1}}")
  "unknown limit Never"
#guard rejects (rerolling "{\"In\": [2147483648]}" "\"Unlimited\"")
  "outside the i32 range"
#guard rejects (exploding "{\"Odd\": {\"Immediate\": 1}}" "\"Unlimited\"")
  "unknown predicate Odd"
#guard rejects (exploding "{\"Eq\": {\"Immediate\": 1}}"
    "{\"Never\": {\"Immediate\": 1}}")
  "unknown limit Never"

/-! ## Immediates must be `i32`s -/

#guard reads (body "{\"Return\": {\"src\": {\"Immediate\": 2147483647}}}")
#guard reads (body "{\"Return\": {\"src\": {\"Immediate\": -2147483648}}}")
#guard rejects (body "{\"Return\": {\"src\": {\"Immediate\": 2147483648}}}")
  "outside the i32 range"
#guard rejects (body "{\"Return\": {\"src\": {\"Immediate\": -2147483649}}}")
  "outside the i32 range"
#guard rejects (body
  ("{\"RollCustomDice\": {\"dest\": 0, \"count\": {\"Immediate\": 1}, " ++
   "\"faces\": [1, 2147483648]}}, " ++ ret))
  "outside the i32 range"

/-! ## Registers and rolling records must exist -/

#guard rejects (body "{\"Return\": {\"src\": {\"Register\": 2}}}")
  "register @2 does not exist"
#guard rejects (body
  ("{\"Neg\": {\"dest\": 2, \"op\": {\"Immediate\": 1}}}, " ++ ret))
  "register @2 does not exist"
#guard rejects
  (body ("{\"SumRollingRecord\": {\"dest\": 0, \"src\": 1}}, " ++ ret))
  "rolling record ⚅1 does not exist"
#guard rejects (body
  ("{\"DropLowest\": {\"dest\": 1, \"count\": {\"Immediate\": 1}}}, " ++ ret))
  "rolling record ⚅1 does not exist"
#guard rejects (body
  ("{\"KeepLowest\": {\"dest\": 1, \"count\": {\"Immediate\": 1}}}, " ++ ret))
  "rolling record ⚅1 does not exist"
#guard rejects (body
  ("{\"KeepHighest\": {\"dest\": 0, \"count\": {\"Register\": 2}}}, " ++
   ret))
  "register @2 does not exist"
#guard rejects (body "{\"Return\": {\"src\": {\"RollingRecord\": 0}}}")
  "rolling record ⚅0 used as a value"
#guard rejects (body
  ("{\"Reroll\": {\"dest\": 1, \"predicate\": {\"In\": []}, " ++
   "\"limit\": \"Unlimited\"}}, " ++ ret))
  "rolling record ⚅1 does not exist"
#guard rejects (rerolling
    ("{\"InRangeInclusive\": {\"start\": {\"Immediate\": 1}, " ++
      "\"end\": {\"Register\": 2}}}")
    "\"Unlimited\"")
  "register @2 does not exist"
#guard rejects
  (rerolling "{\"Eq\": {\"RollingRecord\": 0}}" "\"Unlimited\"")
  "rolling record ⚅0 used as a value"
#guard rejects (rerolling "{\"Eq\": {\"Immediate\": 1}}"
    "{\"Times\": {\"Register\": 2}}")
  "register @2 does not exist"
#guard rejects (body
  ("{\"Explode\": {\"dest\": 1, \"predicate\": {\"In\": []}, " ++
   "\"limit\": \"Unlimited\"}}, " ++ ret))
  "rolling record ⚅1 does not exist"
#guard rejects (exploding "{\"Ne\": {\"Register\": 2}}" "\"Unlimited\"")
  "register @2 does not exist"
#guard rejects (exploding "{\"Eq\": {\"Immediate\": 1}}"
    "{\"Times\": {\"RollingRecord\": 0}}")
  "rolling record ⚅0 used as a value"
#guard rejects
  ("{\"parameters\": [\"a\", \"b\"], \"externals\": [\"c\"], " ++
   "\"register_count\": 2, \"rolling_record_count\": 0, " ++
   "\"instructions\": [" ++ ret ++ "]}")
  "exceed the registers"

/-! ## A function ends with its only `return` -/

#guard rejects (body "") "the function has no return"
#guard rejects (body
  ("{\"Neg\": {\"dest\": 0, \"op\": {\"Immediate\": 1}}}"))
  "the function has no return"
#guard rejects (body (ret ++ ", " ++ ret))
  "instruction 0: return is not the last instruction"
#guard rejects (body
  (ret ++ ", {\"Neg\": {\"dest\": 0, \"op\": {\"Immediate\": 1}}}"))
  "instruction 0: return is not the last instruction"

/-! ## Resolving rerolls -/

/-- Read an operand as its immediate, or `10` times its register. -/
private def value : AddressingMode → Int
  | .immediate v => v
  | .register r => 10 * r
  | .rollingRecord _ => 0

#guard (Predicate.eq (.register 1)).resolve value == .within 10 10
#guard (Predicate.ne (.immediate 3)).resolve value == .outside 3 3
#guard (Predicate.lt (.immediate 3)).resolve value == .within i32Min 2
#guard (Predicate.le (.immediate 3)).resolve value == .within i32Min 3
#guard (Predicate.gt (.immediate 3)).resolve value == .within 4 i32Max
#guard (Predicate.ge (.immediate 3)).resolve value == .within 3 i32Max
#guard (Predicate.in [5, 1, 5]).resolve value == .among [5, 1, 5]
#guard (Predicate.inRange (.immediate 2) (.register 1)).resolve value
  == .within 2 9
#guard (Predicate.inRangeInclusive (.immediate 2) (.register 1)).resolve value
  == .within 2 10
#guard (Predicate.inRange (.register 1) (.immediate 2)).operands
  == [.register 1, .immediate 2]
#guard (Predicate.in [1]).operands == []
#guard (Limit.times (.register 1)).resolve value == some 10
#guard (Limit.times (.immediate 0)).resolve value == some 0
#guard (Limit.times (.register 1)).operands == [.register 1]
#guard Limit.unlimited.operands == []

/-! ## Docstring examples, verbatim -/

#guard (Function.ofJson "{\"parameters\": [], \"externals\": [],
  \"register_count\": 0, \"rolling_record_count\": 0,
  \"instructions\": [{\"Return\": {\"src\": {\"Immediate\": 7}}}]}")
  matches .ok _
#guard ((Predicate.lt (.immediate 3)).resolve
  fun | .immediate v => v | _ => 0).matches 2
#guard !((Predicate.inRange (.immediate 2) (.immediate 2)).resolve
  fun | .immediate v => v | _ => 0).matches 2
#guard (Limit.times (.immediate (-2))).resolve (fun _ => 0) == some 0
#guard Limit.unlimited.resolve (fun _ => 0) == none

end Xdy.Test
