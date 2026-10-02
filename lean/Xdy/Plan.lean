import Xdy.Ir

/-!
# Fan-out plans

The splits and merges that the Rust forward pass performs, mirroring
`xdy/src/distribution/propagation/plan.rs`, so that proofs can show that they
preserve the law of a function.

Combining the operands of an instruction pair by pair is exact only if the
operands are independent, and they are unless some random value reaches both.
A random value that two or more instructions read _fans out_. The pass
conditions on such a value: it splits its state into one world for each
outcome of the value, in which the value is fixed, and later merges those
worlds into one, at the first instruction after which at most one live value
depends on it. A plan schedules these splits and merges before the pass runs.

A plan reasons about _values_ rather than locations, since register
coalescence reuses registers: every write to a location, including every drop,
which rewrites its record, makes a new value, and every read reads the value
most recently written there. Every instruction writes exactly one value, so a
value is numbered by the program counter of the instruction that writes it.

The Rust numbers the values in sweeps over the instructions, filling in their
facts as it goes. Here each fact is a function of the instructions, defined
directly, so that proofs unfold one definition at a time rather than reason
about the invariants of loops: the value in a location before an instruction
is the most recent write there, a value's readers are the later instructions
that read it, and whether a value is random, and which values it depends on,
recur over the values that its instruction reads. Only the stack of open
splits, which each merge changes, is threaded from one instruction to the
next, as the Rust threads it.
-/

namespace Xdy

/-! ## Locations -/

/-- A location that holds a value. Mirrors `Location` in
`propagation/plan.rs`. -/
inductive Location where
  /-- A register. -/
  | register (index : Nat)
  /-- A rolling record. -/
  | record (index : Nat)
  /-- The answer, which `return` writes. -/
  | answer
  deriving DecidableEq, Repr

namespace Location

/--
Whether an operand reads a location.

# Parameters
- `ℓ`: The location.
- `op`: The operand.

# Returns
`true` if `op` is the register at `ℓ`. An operand never reads a rolling
record; the specification's `State.value` reads one as `0`.
-/
def isOperand (ℓ : Location) : AddressingMode → Bool
  | .register r => ℓ == .register r
  | _ => false

/--
Whether an instruction reads a location.

# Parameters
- `ℓ`: The location.
- `i`: The instruction.

# Returns
`true` if `i` reads the value at `ℓ`: an operand that is its register, the
record that a drop drops from, or the record that a sum sums. No instruction
reads the answer.
-/
def readBy (ℓ : Location) : Instruction → Bool
  | .rollRange _ a b => ℓ.isOperand a || ℓ.isOperand b
  | .rollStandardDice _ c n => ℓ.isOperand c || ℓ.isOperand n
  | .rollCustomDice _ c _ => ℓ.isOperand c
  | .dropLowest d c => ℓ == .record d || ℓ.isOperand c
  | .dropHighest d c => ℓ == .record d || ℓ.isOperand c
  | .sumRollingRecord _ r => ℓ == .record r
  | .binary _ _ a b => ℓ.isOperand a || ℓ.isOperand b
  | .neg _ a => ℓ.isOperand a
  | .return a => ℓ.isOperand a

/--
Whether an instruction writes a location.

# Parameters
- `ℓ`: The location.
- `i`: The instruction.

# Returns
`true` if `i` writes `ℓ`: the record that a roll or a drop writes, the
register that a sum or an arithmetic instruction writes, or the answer that
`return` writes.
-/
def writtenBy (ℓ : Location) : Instruction → Bool
  | .rollRange d _ _ | .rollStandardDice d _ _ | .rollCustomDice d _ _
  | .dropLowest d _ | .dropHighest d _ => ℓ == .record d
  | .sumRollingRecord d _ | .binary _ d _ _ | .neg d _ => ℓ == .register d
  | .return _ => ℓ == .answer

/--
Whether a location is dead before instructions: whether they, and then the
reading of the returned operand, never read the value that it holds.

# Parameters
- `ℓ`: The location.
- `src`: The returned operand, which the specification's `run` reads after
  the instructions.
- `is`: The instructions.

# Returns
`true` if some instruction writes `ℓ` before any reads it, or none reads it
and neither does `src`.

# Examples
```lean
example : (Location.register 0).dead (.register 1)
    [.binary .add 0 (.immediate 1) (.immediate 2)] := rfl
example : (Location.register 0).dead (.register 0) [] = false := rfl
```
-/
def dead (ℓ : Location) (src : AddressingMode) : List Instruction → Bool
  | [] => !ℓ.isOperand src
  | i :: is => !ℓ.readBy i && (ℓ.writtenBy i || ℓ.dead src is)

/--
The location that an operand reads, if any, as `Values::number` in
`propagation/plan.rs` reads it.

# Parameters
- `op`: The operand.

# Returns
`none` for an immediate, and otherwise the register or rolling record that it
names. Unlike `isOperand`, this follows the Rust, which reads a rolling record
named as an operand; `Function.validate` rules out such operands, so the two
agree on every valid function.
-/
def ofOperand : AddressingMode → Option Location
  | .immediate _ => none
  | .register r => some (.register r)
  | .rollingRecord r => some (.record r)

end Location

/-! ## Instructions -/

namespace Instruction

/--
The locations that an instruction reads, in order. Mirrors `sources` in
`ir.rs`, less its immediates.

# Parameters
- `i`: The instruction.

# Returns
The locations of its operands, preceded by the record that a drop drops from,
or the record that a sum sums.
-/
def sources : Instruction → List Location
  | .rollRange _ a b => [a, b].filterMap Location.ofOperand
  | .rollStandardDice _ c n => [c, n].filterMap Location.ofOperand
  | .rollCustomDice _ c _ => [c].filterMap Location.ofOperand
  | .dropLowest d c | .dropHighest d c =>
    .record d :: [c].filterMap Location.ofOperand
  | .sumRollingRecord _ r => [.record r]
  | .binary _ _ a b => [a, b].filterMap Location.ofOperand
  | .neg _ a => [a].filterMap Location.ofOperand
  | .return a => [a].filterMap Location.ofOperand

/--
The location that an instruction writes. Mirrors `destination` in `ir.rs`,
with the answer for `return`, as `Values::number` in `propagation/plan.rs`
reads it.

# Parameters
- `i`: The instruction.

# Returns
The record that a roll or a drop writes, the register that a sum or an
arithmetic instruction writes, or the answer.
-/
def destination : Instruction → Location
  | .rollRange d _ _ | .rollStandardDice d _ _ | .rollCustomDice d _ _
  | .dropLowest d _ | .dropHighest d _ => .record d
  | .sumRollingRecord d _ | .binary _ d _ _ | .neg d _ => .register d
  | .return _ => .answer

/--
Whether an instruction rolls: whether it draws its result at random.

# Parameters
- `i`: The instruction.

# Returns
`true` for a range and for standard and custom dice, `false` otherwise.
-/
def rolls : Instruction → Bool
  | .rollRange .. | .rollStandardDice .. | .rollCustomDice .. => true
  | _ => false

end Instruction

/-! ## Values -/

/-- The values of a function, each numbered by the program counter of the
instruction that writes it. Mirrors `Values` in `propagation/plan.rs`, which
stores the facts that these functions compute. -/
structure Values where
  /-- The instructions of the function. -/
  instructions : List Instruction

namespace Values

variable (vs : Values)

/--
The location that holds a value. Mirrors `Value::location`.

# Parameters
- `v`: The value.

# Returns
The destination of the instruction that writes `v`, or the answer if there is
no such instruction.
-/
def location (v : Nat) : Location :=
  (vs.instructions[v]?.map Instruction.destination).getD .answer

/--
The value in a location just before an instruction: the one most recently
written there. Mirrors the map `current` of `Values::number`.

# Parameters
- `pc`: The program counter of the instruction.
- `ℓ`: The location.

# Returns
The last value before `pc` whose location is `ℓ`, or `none` if no instruction
before `pc` writes `ℓ`, whose value is then fixed by the arguments and
external variables.
-/
def writer (pc : Nat) (ℓ : Location) : Option Nat :=
  (List.range pc).reverse.find? (vs.location · == ℓ)

/--
Every value in a location before an instruction was written before it.

# Parameters
- `pc`: The program counter of the instruction.
- `ℓ`: The location.
- `v`: The value.

# Hypotheses
- `h`: `v` is in `ℓ` before `pc`.
-/
theorem lt_of_writer {pc : Nat} {ℓ : Location} {v : Nat}
    (h : vs.writer pc ℓ = some v) : v < pc := by
  have := List.mem_of_find?_eq_some h
  simpa using this

/--
The values that an instruction reads, each once. Mirrors `Values::reads`.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
The values in the locations that the instruction reads, in the order of its
sources. Locations never written hold fixed values, which are omitted.
-/
def reads (pc : Nat) : List Nat :=
  (((vs.instructions[pc]?.map Instruction.sources).getD []).filterMap
    (vs.writer pc)).eraseDups

/--
Every value that an instruction reads was written before it.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.

# Hypotheses
- `h`: The instruction reads `v`.
-/
theorem lt_of_mem_reads {pc v : Nat} (h : v ∈ vs.reads pc) : v < pc := by
  simp only [reads, List.mem_eraseDups, List.mem_filterMap] at h
  obtain ⟨_, _, h⟩ := h
  exact vs.lt_of_writer h

/--
The instructions that read a value. Mirrors `Value::readers`.

# Parameters
- `v`: The value.

# Returns
The program counters of the instructions that read `v`, in ascending order,
each once, even if it reads `v` as more than one operand.
-/
def readers (v : Nat) : List Nat :=
  (List.range vs.instructions.length).filter fun pc => (vs.reads pc).contains v

/--
Whether a value may be random: whether some roll reaches it. Mirrors
`Value::random`.

# Parameters
- `v`: The value.

# Returns
`true` if the instruction that writes `v` rolls, or reads a value that may be
random; `false` if `v` is fixed by the arguments and external variables.
-/
def random (v : Nat) : Bool :=
  (vs.instructions[v]?.any Instruction.rolls) ||
    (vs.reads v).attach.any fun ⟨u, _⟩ => random u
termination_by v
decreasing_by exact vs.lt_of_mem_reads ‹_›

/--
Whether a value lives until the end of the function, as the last answer does.
Mirrors `Value::lasting`.

# Parameters
- `v`: The value.

# Returns
`true` if `v` is the answer after the last instruction, `false` otherwise.
-/
def lasting (v : Nat) : Bool :=
  vs.writer vs.instructions.length .answer == some v

/--
Whether a value fans out: whether it may be random, and more than one
instruction reads it. Mirrors `Value::fans_out`.

# Parameters
- `v`: The value.

# Returns
`true` if `v` fans out, `false` otherwise.
-/
def fansOut (v : Nat) : Bool := vs.random v && 1 < (vs.readers v).length

/--
Whether a value is live after an instruction: whether a later instruction
reads it, or it lasts until the end. Mirrors `Value::is_live_after`, which asks
only whether its last reader is later, as its readers ascend.

# Parameters
- `v`: The value.
- `pc`: The program counter of the instruction.

# Returns
`true` if `v` is live after `pc`, `false` otherwise.
-/
def isLiveAfter (v pc : Nat) : Bool :=
  vs.lasting v || (vs.readers v).any (pc < ·)

/--
The values that fan out on which a value depends, including itself if it fans
out. Mirrors `Value::depends_on`.

# Parameters
- `v`: The value.

# Returns
The values, each once: those on which the values that `v`'s instruction reads
depend, and `v` itself if it fans out.
-/
def dependsOn (v : Nat) : List Nat :=
  ((vs.reads v).attach.flatMap (fun ⟨u, _⟩ => dependsOn u) ++
    if vs.fansOut v then [v] else []).eraseDups
termination_by v
decreasing_by exact vs.lt_of_mem_reads ‹_›

/--
The values live after an instruction. Mirrors the set `live` of
`Values::schedule`.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
The values written at or before `pc` that are live after it, in ascending
order. The Rust adds each value as it is written and removes it at its last
reader, which is the same set.
-/
def live (pc : Nat) : List Nat :=
  (List.range (pc + 1)).filter (vs.isLiveAfter · pc)

/--
The values that occupy their locations after an instruction. Mirrors the set
`current` of `Values::schedule`.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
The values written at or before `pc` that no later write at or before `pc`
displaced, in ascending order.
-/
def current (pc : Nat) : List Nat :=
  (List.range (pc + 1)).filter fun v =>
    vs.writer (pc + 1) (vs.location v) == some v

/-! ## Scheduling -/

/--
Decide whether a split may merge, and if so, answer the location of the value
that survives it. A split may merge once at most one live value depends on it,
other than itself, and that value is not a rolling record, whose influence is
not yet one value; and once every split above it is independent of it.
Mirrors `Values::survivor`.

# Parameters
- `splits`: The stack of splits, by value, from the bottom.
- `split`: The position of the split in the stack.
- `live`: The live values.

# Returns
`none` if the split may not merge yet, `some none` if it may merge and no live
value depends on it, or `some (some ℓ)` if it may merge into the value at `ℓ`.
-/
def survivor (splits : List Nat) (split : Nat) (live : List Nat) :
    Option (Option Location) :=
  let value := splits.getD split 0
  let dependent v := (vs.dependsOn v).contains value
  if (splits.drop (split + 1)).any dependent then none
  else
    match live.filter dependent with
    | [] => some none
    | [survivor] =>
      if survivor == value then none
      else
        match vs.location survivor with
        | .record _ => none
        | ℓ => some (some ℓ)
    | _ => none

/--
The locations of the values, other than the survivor, that depend on a split
and still occupy their locations. Mirrors `Values::dead`.

# Parameters
- `split`: The split, by value.
- `survivor`: The location of the survivor, if any.
- `current`: The values that occupy their locations, in ascending order.

# Returns
The locations, in the order that their values were written.
-/
def dead (split : Nat) (survivor : Option Location) (current : List Nat) :
    List Location :=
  (current.filter fun v =>
    (vs.dependsOn v).contains split && some (vs.location v) != survivor).map
      vs.location

end Values

/-- What the forward pass does after an instruction, beyond executing it.
Mirrors `Step` in `propagation/plan.rs`. -/
inductive Step where
  /-- Condition on the value that the instruction just wrote to the location:
  split every world into one for each of its outcomes, in which it is fixed,
  and push the split atop the stack. -/
  | split (location : Location)
  /-- Merge the worlds of the split at position `split` in the stack, counting
  from the bottom, and remove it from the stack. Every split above it is
  independent of it. `survivor` is the location of the only live value that
  depends on the split, if any, which is never a rolling record. `dead` holds
  the locations of the other values that depend on the split and still occupy
  their locations, in the order written, including the split value itself
  unless a later write displaced it. -/
  | merge (split : Nat) (survivor : Option Location) (dead : List Location)
  deriving DecidableEq, Repr

namespace Values

variable (vs : Values)

/--
Merge every split that may merge after an instruction, topmost first, until
none may. Mirrors the loop `'merge` of `Values::schedule`, which restarts from
the top after each merge.

# Parameters
- `live`: The values live after the instruction.
- `current`: The values that occupy their locations after it.
- `splits`: The stack of splits, by value, from the bottom.

# Returns
The merges, in order, and the splits that remain open.
-/
def merges (live current splits : List Nat) : List Step × List Nat :=
  match (List.range splits.length).reverse.findSome? fun split =>
      (vs.survivor splits split live).map (split, ·) with
  | some (split, survivor) =>
    if h : split < splits.length then
      let (steps, rest) := merges live current (splits.eraseIdx split)
      (.merge split survivor (vs.dead splits[split] survivor current) :: steps,
        rest)
    else ([], splits)
  | none => ([], splits)
termination_by splits.length
decreasing_by rw [List.length_eraseIdx_of_lt h]; omega

/--
The steps that follow an instruction, given the splits open before it: split
the value that it writes, if it fans out, then merge every split that may.

# Parameters
- `splits`: The splits open before the instruction, by value, from the
  bottom.
- `pc`: The program counter of the instruction.

# Returns
The steps, in order, and the splits open after them.
-/
def after (splits : List Nat) (pc : Nat) : List Step × List Nat :=
  let (split, splits) :=
    if vs.fansOut pc then ([Step.split (vs.location pc)], splits ++ [pc])
    else ([], splits)
  let (merges, splits) := vs.merges (vs.live pc) (vs.current pc) splits
  (split ++ merges, splits)

/--
The splits open before an instruction. Mirrors the stack `splits` of
`Values::schedule`.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
The splits, by value, from the bottom.
-/
def openSplits : Nat → List Nat
  | 0 => []
  | pc + 1 => (vs.after (openSplits pc) pc).2

/--
The steps that follow an instruction.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
The steps, in the order that the pass performs them.
-/
def steps (pc : Nat) : List Step := (vs.after (vs.openSplits pc) pc).1

end Values

/-! ## Plans -/

/-- The splits and merges that the forward pass performs after each
instruction, so that every random value that fans out is conditioned on until
its influence reconverges. Mirrors `Plan` in `propagation/plan.rs`. -/
structure Plan where
  /-- The steps that follow each instruction, by program counter. -/
  steps : List (List Step)
  deriving DecidableEq, Repr

namespace Plan

/--
Plan the splits and merges of instructions. Mirrors `Plan::new`.

# Parameters
- `is`: The instructions of a function.

# Returns
The plan.

# Examples
```lean
#guard (Plan.new [.return (.immediate 7)]).steps = [[]]
```
-/
def new (is : List Instruction) : Plan :=
  ⟨(List.range is.length).map (Values.mk is).steps⟩

end Plan

end Xdy
