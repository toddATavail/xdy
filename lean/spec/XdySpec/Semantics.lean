import Mathlib.Probability.Distributions.Uniform
import Xdy.Interpreter
import XdySpec.Record

/-!
# Semantics

The meaning of a compiled xDy function, as a Mathlib `PMF ℤ`: the probability
of each result that it can answer. Mirrors `Evaluator` in
`xdy/src/evaluator.rs`, with every draw from the pseudo-random number
generator replaced by a draw from a probability mass function.

A machine state holds every register and every rolling record. Each
instruction maps one state to a `PMF` of next states: arithmetic, drops and
sums to a certain state, and the rolls to a draw. A die is `PMF.ofMultiset`
of its faces, so each face is equally likely by position, and a set of dice is
that many independent draws, recorded in the order rolled. The instructions
never branch or loop, so a function's meaning binds them in order and then
reads the returned operand.

This is the specification that the oracle, `Xdy.Function.run`, must meet, and
does, by `run_eq` in `XdySpec/Oracle.lean`. It
has the same shape, instruction by instruction, but it differs in how it
represents what it computes: `PMF` probabilities are extended nonnegative
reals rather than natural weights over one total, nothing merges, since equal
outcomes are equal as elements of the support, and records keep their results
in roll order rather than sorted. It cannot be computed, only reasoned about.
Validation and the initial state are deterministic, so it reuses the oracle's.
-/

namespace Xdy.Spec

/-! ## Machine states -/

/-- The state of the machine: every register and every rolling record. Mirrors
`EvaluatorState` in `evaluator.rs`, less its pseudo-random number generator,
program counter, result and dice budget, as the oracle's `Xdy.State` does. -/
structure State where
  /-- The registers. The first hold the arguments, in order, and the next the
  external variables, in order. -/
  registers : Array Int
  /-- The rolling records, whose results are in roll order. -/
  records : Array Record
  deriving Inhabited

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

/--
Read a state of the oracle as a state of the specification.

# Parameters
- `s`: The oracle's state.

# Returns
The state with the same registers, and every rolling record read by
`Record.ofOracle`.
-/
def ofOracle (s : Xdy.State) : State :=
  { registers := s.registers, records := s.records.map Record.ofOracle }

/--
Read a state of the specification as a state of the oracle.

# Parameters
- `s`: The specification's state.

# Returns
The state with the same registers, and every rolling record read by
`Record.toOracle`, which sorts its results. Inverse to `ofOracle` on states
whose records are sorted.
-/
def toOracle (s : State) : Xdy.State :=
  { registers := s.registers, records := s.records.map Record.toOracle }

end State

/-! ## Draws -/

/--
The law of a range, as `roll_range` in `primitives.rs` rolls it: one result,
uniform over `[start, stop]`, or a certain `0` if the range is empty.

# Parameters
- `start`: The least result.
- `stop`: The greatest result.

# Returns
The law of the rolling record.

# Examples
```lean
example : rollRange 3 1 = PMF.pure { results := [0] } := by
  simp [rollRange]
```
-/
noncomputable def rollRange (start stop : Int) : PMF Record :=
  if h : stop < start then PMF.pure { results := [0] }
  else
    (PMF.ofMultiset (Finset.Icc start stop).val (by simp; omega)).map
      fun x => { results := [x] }

/--
The law of one standard die, as `roll_standard_dice` in `primitives.rs` rolls
it: uniform over `1` through `faces`, or a certain `0` if `faces` is `0` or
less.

# Parameters
- `faces`: The number of faces.

# Returns
The law of the die.

# Examples
```lean
example : standardDie 0 = PMF.pure 0 := by simp [standardDie]
```
-/
noncomputable def standardDie (faces : Int) : PMF Int :=
  if h : faces ≤ 0 then PMF.pure 0
  else PMF.ofMultiset (Finset.Icc 1 faces).val (by simp; omega)

/--
The law of one custom die, as `roll_custom_dice` in `primitives.rs` rolls it:
each listed face equally likely by position, or a certain `0` if there are no
faces.

# Parameters
- `faces`: The faces, which may repeat.

# Returns
The law of the die.

# Examples
```lean
example : customDie [] = PMF.pure 0 := by simp [customDie]
example : customDie [1, 1, 2] 1 = 2 / 3 := by
  simp [customDie, PMF.ofMultiset_apply]
```
-/
noncomputable def customDie (faces : List Int) : PMF Int :=
  if h : faces = [] then PMF.pure 0
  else PMF.ofMultiset faces (by simpa using h)

/--
The law of a set of dice: `count` independent draws from `die`, recorded in
the order rolled.

# Parameters
- `count`: The number of dice. A count of `0` or less rolls no dice.
- `die`: The law of one die.

# Returns
The law of the rolling record.

# Notes
The oracle takes its die as a thunk, so that rolling no dice never builds a
die with many faces. The specification builds nothing, so it takes the law
itself.

# Examples
```lean
example (die : PMF Int) : rollDice (-1) die = PMF.pure {} := rfl
```
-/
noncomputable def rollDice (count : Int) (die : PMF Int) : PMF Record :=
  count.toNat.repeat (fun d => d.bind fun r => die.map r.push) (PMF.pure {})

/-! ## Instructions -/

/--
Execute one instruction in one state.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The law of the state after the instruction. A roll replaces its rolling record
outright, discarding any drops, as `evaluator.rs` does; every other
instruction answers a certain state. `return` changes nothing, since `run`
reads its operand afterward.
-/
noncomputable def step (s : State) : Instruction → PMF State
  | .rollRange d a b =>
    (rollRange (s.value a) (s.value b)).map (s.setRecord d)
  | .rollStandardDice d c n =>
    (rollDice (s.value c) (standardDie (s.value n))).map (s.setRecord d)
  | .rollCustomDice d c faces =>
    (rollDice (s.value c) (customDie faces)).map (s.setRecord d)
  | .dropLowest d c =>
    PMF.pure (s.setRecord d ((s.record d).dropLowest (s.value c)))
  | .dropHighest d c =>
    PMF.pure (s.setRecord d ((s.record d).dropHighest (s.value c)))
  | .sumRollingRecord d r => PMF.pure (s.setRegister d (s.record r).sum)
  | .binary op d a b =>
    PMF.pure (s.setRegister d (op.apply (s.value a) (s.value b)))
  | .neg d a => PMF.pure (s.setRegister d (neg (s.value a)))
  | .return _ => PMF.pure s

/-! ## Functions -/

/--
The meaning of a function: the law of its result.

# Parameters
- `f`: The function.
- `args`: The arguments, one per parameter, in order.
- `externals`: Values for external variables, by name, as for
  `Xdy.Function.initialState`.

# Returns
The law of the result.

# Errors
If `f` is not well formed (see `Xdy.Function.validate`), or the arguments or
external variables do not fit it (see `Xdy.Function.initialState`). These are
the oracle's checks, so the specification fails exactly when the oracle does.
-/
noncomputable def run (f : Function) (args : List Int)
    (externals : List (String × Int) := []) : Except String (PMF Int) := do
  f.validate
  let s ← f.initialState args externals
  let final := f.instructions.foldl
    (fun (d : PMF State) i => d.bind (step · i)) (PMF.pure (.ofOracle s))
  match f.instructions.back? with
  | some (.return src) => pure (final.map (·.value src))
  | _ => throw "the function has no return"

end Xdy.Spec
