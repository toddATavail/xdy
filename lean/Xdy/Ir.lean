import Lean.Data.Json
import Xdy.Record

/-!
# Intermediate representation

The xDy intermediate representation (IR), mirroring `xdy/src/ir.rs` and
`Function` in `xdy/src/compiler.rs`, and a reader for the JSON that Rust's
`serde` feature writes for a compiled `Function`.

A function is a straight-line list of instructions over a bank of registers,
which hold integers, and a set of rolling records, which hold the results of
ranges and dice. The first registers hold the function's parameters, in
order, and the next hold its external variables, in order. Only `RollRange`,
`RollStandardDice`, `RollCustomDice` and `Reroll` introduce randomness.

## The JSON form

`serde` writes enums in its default, externally tagged form: a variant is an
object with a single field, named for the variant, whose value is the
variant's payload, or just its name, as a string, if it has no payload, as
`Limit::Unlimited` has none. Structs are objects, and newtypes such as
`Immediate` and `RegisterIndex` are their inner values. For example, the IR
of `(1D3)D3`:

```json
{"parameters": [], "externals": [],
 "register_count": 1, "rolling_record_count": 2,
 "instructions": [
   {"RollRange": {"dest": 0,
                  "start": {"Immediate": 1}, "end": {"Immediate": 3}}},
   {"SumRollingRecord": {"dest": 0, "src": 0}},
   {"RollStandardDice": {"dest": 1,
                         "count": {"Register": 0},
                         "faces": {"Immediate": 3}}},
   {"SumRollingRecord": {"dest": 0, "src": 1}},
   {"Return": {"src": {"Register": 0}}}]}
```

The optimizer reuses registers, so the IR is not in SSA form: an instruction
may overwrite a register that an earlier instruction wrote.
-/

namespace Xdy

open Lean (Json FromJson fromJson?)

/-! ## Instructions -/

/-- An operand of an instruction. Mirrors `AddressingMode` in `ir.rs`. -/
inductive AddressingMode where
  /-- A constant, within the `i32` range. -/
  | immediate (value : Int)
  /-- The current value of a register. -/
  | register (index : Nat)
  /-- A rolling record. No instruction reads a rolling record through an
  operand; they name their records directly. It is here only because Rust
  can write it. -/
  | rollingRecord (index : Nat)
  deriving Repr, BEq, Inhabited

/-- The operation of a binary arithmetic instruction. Rust gives each its own
instruction type (`Add`, `Sub`, and so on), all with the same fields; here
they share one constructor, `Instruction.binary`. -/
inductive BinaryOp where
  /-- Saturating addition, `Add`. -/
  | add
  /-- Saturating subtraction, `Sub`. -/
  | sub
  /-- Saturating multiplication, `Mul`. -/
  | mul
  /-- Division toward zero, `Div`. -/
  | div
  /-- Remainder with the sign of the dividend, `Mod`. -/
  | «mod»
  /-- Saturating exponentiation, `Exp`. -/
  | exp
  /-- The greater operand, `Max`. -/
  | «max»
  /-- The lesser operand, `Min`. -/
  | «min»
  deriving Repr, BEq, Inhabited

/-- The predicate of a reroll: the test that a die must pass to be rerolled,
whose thresholds and endpoints are operands, read once, before the first
reroll. Mirrors `Predicate` in `ir.rs`. -/
inductive Predicate where
  /-- Matches a result equal to the threshold. -/
  | eq (threshold : AddressingMode)
  /-- Matches a result not equal to the threshold. -/
  | ne (threshold : AddressingMode)
  /-- Matches a result less than the threshold. -/
  | lt (threshold : AddressingMode)
  /-- Matches a result less than or equal to the threshold. -/
  | le (threshold : AddressingMode)
  /-- Matches a result greater than the threshold. -/
  | gt (threshold : AddressingMode)
  /-- Matches a result greater than or equal to the threshold. -/
  | ge (threshold : AddressingMode)
  /-- Matches a result among the listed values, which are literals. An empty
  list matches nothing. -/
  | «in» (values : List Int)
  /-- Matches a result within the half-open range `start..stop`, which is
  empty if `stop` does not exceed `start`. The JSON names `stop` `end`, a Lean
  keyword. -/
  | inRange (start stop : AddressingMode)
  /-- Matches a result within the inclusive range `start..=stop`, which is
  empty if `stop` is less than `start`. -/
  | inRangeInclusive (start stop : AddressingMode)
  deriving Repr, BEq, Inhabited

/-- The limit of a reroll: the number of times that each die may be rerolled.
Mirrors `Limit` in `ir.rs`. -/
inductive Limit where
  /-- Reroll each die at most `count` times, or not at all if the count is
  zero or less. -/
  | times (count : AddressingMode)
  /-- Reroll each die until it no longer matches, or not at all if every face
  of the die matches. -/
  | unlimited
  deriving Repr, BEq, Inhabited

namespace Predicate

/--
The operands of a predicate, in the order written. Mirrors
`Predicate::operands`.

# Parameters
- `p`: The predicate.

# Returns
The threshold of a comparison, the start and end of a range, or nothing for a
list.
-/
def operands : Predicate → List AddressingMode
  | .eq t | .ne t | .lt t | .le t | .gt t | .ge t => [t]
  | .in _ => []
  | .inRange a b | .inRangeInclusive a b => [a, b]

/--
Resolve a predicate to the set of results that it matches, given the values of
its operands. Mirrors `Predicate::resolve`.

# Parameters
- `p`: The predicate.
- `value`: Answers the value of an operand.

# Returns
The set of results that `p` matches.

# Notes
Rust clamps the bounds of each range to the `i32` range, after computing them
in `i64`. Every result lies within the `i32` range, so the clamp changes
nothing that a criterion matches, and here the bounds are left unclamped.

# Examples
```lean
#guard ((Predicate.lt (.immediate 3)).resolve
  fun | .immediate v => v | _ => 0).matches 2
#guard !((Predicate.inRange (.immediate 2) (.immediate 2)).resolve
  fun | .immediate v => v | _ => 0).matches 2
```
-/
def resolve (p : Predicate) (value : AddressingMode → Int) : Criterion :=
  match p with
  | .eq t => .within (value t) (value t)
  | .ne t => .outside (value t) (value t)
  | .lt t => .within i32Min (value t - 1)
  | .le t => .within i32Min (value t)
  | .gt t => .within (value t + 1) i32Max
  | .ge t => .within (value t) i32Max
  | .in values => .among values
  | .inRange a b => .within (value a) (value b - 1)
  | .inRangeInclusive a b => .within (value a) (value b)

/--
A predicate resolves alike under any two readings of operands that agree on
its own.

# Parameters
- `p`: The predicate.
- `f`, `g`: The readings of operands.

# Hypotheses
- `h`: `f` and `g` agree on every operand of `p`.
-/
theorem resolve_congr {p : Predicate} {f g : AddressingMode → Int}
    (h : ∀ op ∈ p.operands, f op = g op) : p.resolve f = p.resolve g := by
  cases p <;> simp_all [resolve, operands]

end Predicate

namespace Limit

/--
The operand of a limit, if any, as a list. Mirrors `Limit::operand`.

# Parameters
- `l`: The limit.

# Returns
The count of a finite limit, or nothing if the limit is unlimited.
-/
def operands : Limit → List AddressingMode
  | .times c => [c]
  | .unlimited => []

/--
Resolve a limit to the number of times that each die may be rerolled, given
the value of its operand, as `evaluator.rs` and `reroll_within` in
`primitives.rs` resolve it together.

# Parameters
- `l`: The limit.
- `value`: Answers the value of an operand.

# Returns
The count, or `0` if it is negative, or `none` if `l` is unlimited.

# Examples
```lean
#guard (Limit.times (.immediate (-2))).resolve (fun _ => 0) == some 0
#guard Limit.unlimited.resolve (fun _ => 0) == none
```
-/
def resolve (l : Limit) (value : AddressingMode → Int) : Option Nat :=
  match l with
  | .times c => some (value c).toNat
  | .unlimited => none

/--
A limit resolves alike under any two readings of operands that agree on its
own.

# Parameters
- `l`: The limit.
- `f`, `g`: The readings of operands.

# Hypotheses
- `h`: `f` and `g` agree on every operand of `l`.
-/
theorem resolve_congr {l : Limit} {f g : AddressingMode → Int}
    (h : ∀ op ∈ l.operands, f op = g op) : l.resolve f = l.resolve g := by
  cases l <;> simp_all [resolve, operands]

end Limit

/-- An instruction. Mirrors `Instruction` in `ir.rs`. -/
inductive Instruction where
  /-- Roll one value uniformly from the inclusive range `[start, stop]` into
  rolling record `dest`; roll `0` if the range is empty. The JSON names `stop`
  `end`, a Lean keyword. -/
  | rollRange (dest : Nat) (start stop : AddressingMode)
  /-- Roll `count` dice with faces `1` through `faces` into rolling record
  `dest`. -/
  | rollStandardDice (dest : Nat) (count faces : AddressingMode)
  /-- Roll `count` dice with the listed faces, each equally likely by position,
  into rolling record `dest`. -/
  | rollCustomDice (dest : Nat) (count : AddressingMode) (faces : List Int)
  /-- Drop `count` more of the lowest results of rolling record `dest`. -/
  | dropLowest (dest : Nat) (count : AddressingMode)
  /-- Drop `count` more of the highest results of rolling record `dest`. -/
  | dropHighest (dest : Nat) (count : AddressingMode)
  /-- Keep the lowest `count` of the results of rolling record `dest` that are
  still kept, dropping the rest of them from the high end. -/
  | keepLowest (dest : Nat) (count : AddressingMode)
  /-- Keep the highest `count` of the results of rolling record `dest` that are
  still kept, dropping the rest of them from the low end. -/
  | keepHighest (dest : Nat) (count : AddressingMode)
  /-- Reroll the dice still kept in rolling record `dest` that satisfy the
  predicate, each up to the limit of times: cross out a matching die and draw
  its replacement from the record's die, repeating while the replacement
  matches and the limit allows. An unlimited reroll whose predicate matches
  every face of the die draws nothing, but crosses out every matching die. -/
  | reroll (dest : Nat) (predicate : Predicate) (limit : Limit)
  /-- Write the sum of the kept results of rolling record `src` to register
  `dest`. -/
  | sumRollingRecord (dest src : Nat)
  /-- Write `op1 ⊕ op2` to register `dest`, for the operation `op`. -/
  | binary (op : BinaryOp) (dest : Nat) (op1 op2 : AddressingMode)
  /-- Write the saturating negation of `op` to register `dest`. -/
  | neg (dest : Nat) (op : AddressingMode)
  /-- End the function, answering `src`. -/
  | «return» (src : AddressingMode)
  deriving Repr, BEq, Inhabited

/-- A compiled function. Mirrors `Function` in `compiler.rs`. -/
structure Function where
  /-- The names of the formal parameters, in order. -/
  parameters : Array String
  /-- The names of the external variables, in order. -/
  externals : Array String
  /-- The number of registers. -/
  registerCount : Nat
  /-- The number of rolling records. -/
  rollingRecordCount : Nat
  /-- The body. -/
  instructions : Array Instruction
  deriving Repr, BEq, Inhabited

/-! ## Reading JSON -/

/--
Split an externally tagged `serde` variant into its tag and payload.

# Parameters
- `j`: A JSON object with exactly one field.

# Returns
The name of the field and its value.

# Errors
If `j` is not an object with exactly one field.
-/
private def variant (j : Json) : Except String (String × Json) := do
  let fields ← j.getObj?
  match fields.toArray with
  | #[⟨tag, payload⟩] => pure (tag, payload)
  | _ => throw s!"expected an object with one field, found {j.compress}"

/--
Read an `i32` from JSON, rejecting integers outside the range.

# Parameters
- `j`: A JSON number.

# Returns
The integer.

# Errors
If `j` is not an integer, or lies outside the `i32` range.
-/
private def getI32 (j : Json) : Except String Int := do
  let x ← j.getInt?
  unless i32Min ≤ x ∧ x ≤ i32Max do throw s!"{x} is outside the i32 range"
  pure x

/-- Read an addressing mode from its `serde` form, e.g., `{"Immediate": 3}` or
`{"Register": 0}`. -/
instance : FromJson AddressingMode where
  fromJson? j := do
    match ← variant j with
    | ("Immediate", v) => .immediate <$> getI32 v
    | ("Register", v) => .register <$> v.getNat?
    | ("RollingRecord", v) => .rollingRecord <$> v.getNat?
    | (tag, _) => throw s!"unknown addressing mode {tag}"

/--
Read the operation of a binary instruction from its `serde` tag.

# Parameters
- `tag`: The name of a Rust instruction type.

# Returns
The operation, or `none` if `tag` names no binary instruction.
-/
private def binaryOp? : String → Option BinaryOp
  | "Add" => some .add
  | "Sub" => some .sub
  | "Mul" => some .mul
  | "Div" => some .div
  | "Mod" => some .mod
  | "Exp" => some .exp
  | "Max" => some .max
  | "Min" => some .min
  | _ => none

/-- Read a predicate from its `serde` form, e.g., `{"Le": {"Immediate": 2}}`,
`{"In": [1, 5]}` or `{"InRange": {"start": {"Immediate": 1},
"end": {"Register": 0}}}`. -/
instance : FromJson Predicate where
  fromJson? j := do
    let (tag, p) ← variant j
    let mode (key : String) : Except String AddressingMode :=
      p.getObjValAs? AddressingMode key
    let threshold : Except String AddressingMode := fromJson? p
    match tag with
    | "Eq" => .eq <$> threshold
    | "Ne" => .ne <$> threshold
    | "Lt" => .lt <$> threshold
    | "Le" => .le <$> threshold
    | "Gt" => .gt <$> threshold
    | "Ge" => .ge <$> threshold
    | "In" => return .in (← (← p.getArr?).toList.mapM getI32)
    | "InRange" => return .inRange (← mode "start") (← mode "end")
    | "InRangeInclusive" =>
      return .inRangeInclusive (← mode "start") (← mode "end")
    | _ => throw s!"unknown predicate {tag}"

/-- Read a limit from its `serde` form, `{"Times": {"Immediate": 2}}` or
`"Unlimited"`, since `serde` writes a variant without a payload as its bare
name. -/
instance : FromJson Limit where
  fromJson? j := do
    if j == "Unlimited" then return .unlimited
    match ← variant j with
    | ("Times", c) => .times <$> fromJson? c
    | (tag, _) => throw s!"unknown limit {tag}"

/-- Read an instruction from its `serde` form, e.g.,
`{"SumRollingRecord": {"dest": 0, "src": 1}}`. -/
instance : FromJson Instruction where
  fromJson? j := do
    let (tag, p) ← variant j
    let dest : Except String Nat := p.getObjValAs? Nat "dest"
    let mode (key : String) : Except String AddressingMode :=
      p.getObjValAs? AddressingMode key
    match tag with
    | "RollRange" =>
      return .rollRange (← dest) (← mode "start") (← mode "end")
    | "RollStandardDice" =>
      return .rollStandardDice (← dest) (← mode "count") (← mode "faces")
    | "RollCustomDice" =>
      let faces ← (← p.getObjVal? "faces").getArr?
      return .rollCustomDice (← dest) (← mode "count")
        (← faces.toList.mapM getI32)
    | "DropLowest" => return .dropLowest (← dest) (← mode "count")
    | "DropHighest" => return .dropHighest (← dest) (← mode "count")
    | "KeepLowest" => return .keepLowest (← dest) (← mode "count")
    | "KeepHighest" => return .keepHighest (← dest) (← mode "count")
    | "Reroll" =>
      return .reroll (← dest) (← p.getObjValAs? Predicate "predicate")
        (← p.getObjValAs? Limit "limit")
    | "SumRollingRecord" =>
      return .sumRollingRecord (← dest) (← p.getObjValAs? Nat "src")
    | "Neg" => return .neg (← dest) (← mode "op")
    | "Return" => return .return (← mode "src")
    | _ =>
      match binaryOp? tag with
      | some op =>
        return .binary op (← dest) (← mode "op1") (← mode "op2")
      | none => throw s!"unknown instruction {tag}"

/-- Read a function from its `serde` form. -/
instance : FromJson Function where
  fromJson? j := do
    return {
      parameters := ← j.getObjValAs? (Array String) "parameters"
      externals := ← j.getObjValAs? (Array String) "externals"
      registerCount := ← j.getObjValAs? Nat "register_count"
      rollingRecordCount := ← j.getObjValAs? Nat "rolling_record_count"
      instructions := ← j.getObjValAs? (Array Instruction) "instructions"
    }

/-! ## Validation -/

namespace Function

/--
Check that an operand can be read as a value: an immediate, or a register
that exists.

# Parameters
- `f`: The function.
- `op`: The operand.

# Errors
If `op` names a register beyond the bank, or a rolling record.
-/
private def checkValue (f : Function) : AddressingMode → Except String Unit
  | .immediate _ => pure ()
  | .register r =>
    unless r < f.registerCount do throw s!"register @{r} does not exist"
  | .rollingRecord r => throw s!"rolling record ⚅{r} used as a value"

/--
Check that a register exists, as the destination of an instruction.

# Parameters
- `f`: The function.
- `r`: The register.

# Errors
If `r` lies beyond the bank.
-/
private def checkRegister (f : Function) (r : Nat) : Except String Unit :=
  unless r < f.registerCount do throw s!"register @{r} does not exist"

/--
Check that a rolling record exists.

# Parameters
- `f`: The function.
- `r`: The rolling record.

# Errors
If `r` lies beyond the set of rolling records.
-/
private def checkRecord (f : Function) (r : Nat) : Except String Unit :=
  unless r < f.rollingRecordCount do
    throw s!"rolling record ⚅{r} does not exist"

/--
Check that every operand and destination of an instruction exists.

# Parameters
- `f`: The function.
- `inst`: The instruction.

# Errors
If the instruction names a register or rolling record that does not exist,
or reads a rolling record as a value.
-/
private def checkInstruction (f : Function) :
    Instruction → Except String Unit
  | .rollRange d s e => do
    f.checkRecord d; f.checkValue s; f.checkValue e
  | .rollStandardDice d c n => do
    f.checkRecord d; f.checkValue c; f.checkValue n
  | .rollCustomDice d c _ => do f.checkRecord d; f.checkValue c
  | .dropLowest d c | .dropHighest d c | .keepLowest d c | .keepHighest d c =>
    do f.checkRecord d; f.checkValue c
  | .reroll d p l => do
    f.checkRecord d; (p.operands ++ l.operands).forM f.checkValue
  | .sumRollingRecord d s => do f.checkRegister d; f.checkRecord s
  | .binary _ d a b => do
    f.checkRegister d; f.checkValue a; f.checkValue b
  | .neg d a => do f.checkRegister d; f.checkValue a
  | .return s => f.checkValue s

/--
Check that a function is well formed, so that the interpreter never reads
outside its registers or rolling records: its parameters and external
variables fit in its registers, every instruction names only registers and
rolling records that exist, and it ends with its only `return`.

# Parameters
- `f`: The function.

# Errors
The first defect found, as a message.
-/
def validate (f : Function) : Except String Unit := do
  unless f.parameters.size + f.externals.size ≤ f.registerCount do
    throw "the parameters and external variables exceed the registers"
  f.instructions.forM f.checkInstruction
  -- Blame a return that anything follows before the absence of a return at
  -- the end, since the former pinpoints the defect.
  match f.instructions.findIdx? fun | .return _ => true | _ => false with
  | none => throw "the function has no return"
  | some i =>
    unless i + 1 == f.instructions.size do
      throw s!"instruction {i}: return is not the last instruction (a \
        function ends with its only return)"

/--
Read and validate a function from the JSON that Rust's `serde` feature writes.

# Parameters
- `s`: The JSON text of a compiled `Function`.

# Returns
The function.

# Errors
If `s` is not JSON, does not describe a function, or describes one that is
not well formed.

# Examples
```lean
#guard (Function.ofJson "{\"parameters\": [], \"externals\": [],
  \"register_count\": 0, \"rolling_record_count\": 0,
  \"instructions\": [{\"Return\": {\"src\": {\"Immediate\": 7}}}]}")
  matches .ok _
```
-/
def ofJson (s : String) : Except String Function := do
  let f ← fromJson? (← Json.parse s)
  f.validate
  pure f

end Function

end Xdy
