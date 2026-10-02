import Xdy.Dist
import Xdy.Ir
import Xdy.Record

/-!
# Interpreter

The executable oracle's interpreter: it runs a compiled xDy function once over
a distribution of whole machine states, and answers the exact distribution of
the function's result. Mirrors `Evaluator` in `xdy/src/evaluator.rs`, with
every draw from the pseudo-random number generator replaced by a branch into
every outcome.

A machine state holds every register and every rolling record. Each
instruction maps one state to a distribution of next states: arithmetic,
drops and sums to a single certain state, and the rolls to one state per
outcome. Rolls draw one die at a time, and since records keep their results
sorted, states that differ only in the order of their rolls merge as they
arise. The instructions never branch or loop, so the interpreter binds them
in order and then reads the returned operand from every final state.

The interpreter is exponential in the worst case: `nDk` has as many states as
there are multisets of `n` faces drawn from `k`. It is meant only for small
programs.
-/

namespace Xdy

/-! ## Machine states -/

/-- The state of the machine: every register and every rolling record. Mirrors
`EvaluatorState` in `evaluator.rs`, less its pseudo-random number generator,
program counter, result and dice budget. Equality is decided, not merely a
`BEq`, so that it is lawful, as proofs about a `Dist` of states require. -/
structure State where
  /-- The registers. The first hold the arguments, in order, and the next the
  external variables, in order. -/
  registers : Array Int
  /-- The rolling records. -/
  records : Array Record
  deriving Repr, DecidableEq, Hashable, Inhabited

namespace State

/--
Read an operand as a value.

# Parameters
- `s`: The state.
- `op`: The operand.

# Returns
The immediate, or the current value of the register. A register that does not
exist, and a rolling record, read as `0`; `Function.validate` rules out both.
-/
def value (s : State) : AddressingMode → Int
  | .immediate v => v
  | .register r => s.registers.getD r 0
  | .rollingRecord _ => 0

/--
Write a value to a register.

# Parameters
- `s`: The state.
- `r`: The register, which must exist.
- `v`: The value.

# Returns
The state with `v` in register `r`.
-/
def setRegister (s : State) (r : Nat) (v : Int) : State :=
  { s with registers := s.registers.modify r fun _ => v }

/--
Read a rolling record.

# Parameters
- `s`: The state.
- `r`: The rolling record, which must exist.

# Returns
The rolling record.
-/
def record (s : State) (r : Nat) : Record := s.records.getD r {}

/--
Replace a rolling record.

# Parameters
- `s`: The state.
- `r`: The rolling record, which must exist.
- `rec`: The new contents of the rolling record.

# Returns
The state with `rec` in rolling record `r`.
-/
def setRecord (s : State) (r : Nat) (rec : Record) : State :=
  { s with records := s.records.modify r fun _ => rec }

end State

/-! ## Draws -/

/--
The distribution of a range, as `roll_range` in `primitives.rs` rolls it: one
result, uniform over `[start, stop]`, or a certain `0` if the range is empty.

# Parameters
- `start`: The least result.
- `stop`: The greatest result.

# Returns
The distribution of the rolling record.

# Examples
```lean
#guard (rollRange 1 3).total == 3
#guard rollRange 3 1 == Dist.pure { results := [0] }
```
-/
def rollRange (start stop : Int) : Dist Record :=
  if stop < start then .pure { results := [0] }
  else
    let results :=
      (List.range (stop - start + 1).toNat).map fun (i : Nat) => start + i
    (Dist.uniform results).map fun x => { results := [x] }

/--
The distribution of a set of dice: `count` independent draws from `die`, one
at a time.

# Parameters
- `count`: The number of dice. A count of `0` or less rolls no dice.
- `die`: Answers the distribution of one die. It is called only if `count` is
  positive, since a die with many faces is costly to build, and rolling no
  dice must not build it, as rolling none of them costs nothing in Rust.

# Returns
The distribution of the rolling record.

# Examples
```lean
#guard (rollDice 2 fun _ => Dist.uniform [1, 2]).total == 4
#guard rollDice (-1) (fun _ => Dist.uniform [1, 2]) == Dist.pure {}
#guard rollDice 0 (fun _ => standardDie i32Max) == Dist.pure {}
```
-/
def rollDice (count : Int) (die : Unit → Dist Int) : Dist Record :=
  if count ≤ 0 then .pure {}
  else
    let die := die ()
    count.toNat.repeat (fun d => d.bind fun r => die.map r.push) (.pure {})

/--
The distribution of one standard die, as `roll_standard_dice` in
`primitives.rs` rolls it: uniform over `1` through `faces`, or a certain `0`
if `faces` is `0` or less.

# Parameters
- `faces`: The number of faces.

# Returns
The distribution of the die.

# Examples
```lean
#guard (standardDie 6).total == 6
#guard standardDie 0 == Dist.pure 0
```
-/
def standardDie (faces : Int) : Dist Int :=
  if faces ≤ 0 then .pure 0
  else .uniform ((List.range faces.toNat).map fun (i : Nat) => (i : Int) + 1)

/--
The distribution of one custom die, as `roll_custom_dice` in `primitives.rs`
rolls it: each listed face equally likely by position, or a certain `0` if
there are no faces.

# Parameters
- `faces`: The faces, which may repeat.

# Returns
The distribution of the die.

# Examples
```lean
#guard (customDie [1, 1, 2]).weight 1 == 2
#guard customDie [] == Dist.pure 0
```
-/
def customDie (faces : List Int) : Dist Int :=
  if faces.isEmpty then .pure 0 else .uniform faces

/-! ## Instructions -/

/--
Apply the operation of a binary instruction.

# Parameters
- `op`: The operation.
- `op1`: The first operand.
- `op2`: The second operand.

# Returns
The result, as the corresponding primitive in `primitives.rs` computes it.

# Examples
```lean
#guard BinaryOp.sub.apply 1 2 == -1
#guard BinaryOp.add.apply i32Max 1 == i32Max
```
-/
def BinaryOp.apply : BinaryOp → Int → Int → Int
  | .add => Xdy.add
  | .sub => Xdy.sub
  | .mul => Xdy.mul
  | .div => Xdy.div
  | .mod => Xdy.mod
  | .exp => Xdy.exp
  | .max => Xdy.max

/--
Execute one instruction in one state.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The distribution of states after the instruction. A roll replaces its rolling
record outright, discarding any drops, as `evaluator.rs` does; every other
instruction answers a single certain state. `return` changes nothing, since
`Function.run` reads its operand afterward.
-/
def step (s : State) : Instruction → Dist State
  | .rollRange d a b =>
    (rollRange (s.value a) (s.value b)).map (s.setRecord d)
  | .rollStandardDice d c n =>
    (rollDice (s.value c) fun _ => standardDie (s.value n)).map
      (s.setRecord d)
  | .rollCustomDice d c faces =>
    (rollDice (s.value c) fun _ => customDie faces).map (s.setRecord d)
  | .dropLowest d c =>
    .pure (s.setRecord d ((s.record d).dropLowest (s.value c)))
  | .dropHighest d c =>
    .pure (s.setRecord d ((s.record d).dropHighest (s.value c)))
  | .sumRollingRecord d r => .pure (s.setRegister d (s.record r).sum)
  | .binary op d a b =>
    .pure (s.setRegister d (op.apply (s.value a) (s.value b)))
  | .neg d a => .pure (s.setRegister d (neg (s.value a)))
  | .return _ => .pure s

/-! ## Functions -/

namespace Function

/--
Bind the arguments and external variables of a function in a fresh state, as
`evaluate` in `evaluator.rs` does: every register and rolling record starts
empty, then the arguments fill the first registers and each bound external
variable fills its own register after them.

# Parameters
- `f`: The function.
- `args`: The arguments, one per parameter, in order.
- `externals`: Values for external variables, by name. A later binding of the
  same name replaces an earlier one, and an external variable left unbound is
  `0`.

# Returns
The initial state.

# Errors
If the number of arguments differs from the number of parameters, a value is
outside the `i32` range, or a name is not an external variable of `f`.
-/
def initialState (f : Function) (args : List Int)
    (externals : List (String × Int)) : Except String State := do
  unless args.length == f.parameters.size do
    throw s!"expected {f.parameters.size} arguments, found {args.length}"
  let checkI32 (x : Int) : Except String Unit :=
    unless i32Min ≤ x ∧ x ≤ i32Max do
      throw s!"{x} is outside the i32 range"
  let blank : State := {
    registers := .replicate f.registerCount 0
    records := .replicate f.rollingRecordCount {} }
  let s ← args.zipIdx.foldlM (init := blank) fun s (x, i) => do
    checkI32 x
    pure (s.setRegister i x)
  externals.foldlM (init := s) fun s (name, x) => do
    checkI32 x
    match f.externals.toList.idxOf? name with
    | some i => pure (s.setRegister (f.parameters.size + i) x)
    | none => throw s!"{name} is not an external variable"

/--
Compute the exact distribution of a function's result.

# Parameters
- `f`: The function.
- `args`: The arguments, one per parameter, in order.
- `externals`: Values for external variables, by name, as for
  `initialState`.

# Returns
The distribution of the result, over the product of the totals of every
draw, scaled as `Dist.bind` describes. It is not reduced to lowest terms.

# Errors
If `f` is not well formed (see `validate`), or the arguments or external
variables do not fit it (see `initialState`).

# Examples
```lean
-- `1D6`.
#guard (Function.run
    { parameters := #[], externals := #[]
      registerCount := 1, rollingRecordCount := 1
      instructions := #[.rollRange 0 (.immediate 1) (.immediate 6),
        .sumRollingRecord 0 0, .return (.register 0)] }
    [] |>.map Dist.toList)
  matches .ok [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
```
-/
def run (f : Function) (args : List Int)
    (externals : List (String × Int) := []) : Except String (Dist Int) := do
  f.validate
  let s ← f.initialState args externals
  let final := f.instructions.foldl
    (fun (d : Dist State) i => d.bind (step · i)) (.pure s)
  match f.instructions.back? with
  | some (.return src) => pure (final.map (·.value src))
  | _ => throw "the function has no return"

end Function

end Xdy
