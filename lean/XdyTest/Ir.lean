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
#guard reads (include_str "Fixtures/dynamic_range.json")
#guard reads (include_str "Fixtures/external.json")
#guard reads (include_str "Fixtures/max.json")
#guard reads (include_str "Fixtures/range.json")
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

-- `{x}: {x}D1`: the optimizer's `Max`.
#guard read (include_str "Fixtures/max.json") == some {
  parameters := #["x"], externals := #[]
  registerCount := 1, rollingRecordCount := 0
  instructions := #[
    .binary .max 0 (.immediate 0) (.register 0),
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

/-! ## Malformed JSON is rejected -/

#guard !reads "not json"
#guard rejects (body ("{\"Nop\": {}}, " ++ ret)) "unknown instruction Nop"
#guard rejects (body "{\"Return\": {\"src\": {\"Stack\": 0}}}")
  "unknown addressing mode Stack"
#guard rejects (body "{\"Return\": {}, \"Neg\": {}}")
  "expected an object with one field"
#guard !reads (body "{\"Return\": {}}")

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
#guard rejects (body "{\"Return\": {\"src\": {\"RollingRecord\": 0}}}")
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

/-! ## Docstring examples, verbatim -/

#guard (Function.ofJson "{\"parameters\": [], \"externals\": [],
  \"register_count\": 0, \"rolling_record_count\": 0,
  \"instructions\": [{\"Return\": {\"src\": {\"Immediate\": 7}}}]}")
  matches .ok _

end Xdy.Test
