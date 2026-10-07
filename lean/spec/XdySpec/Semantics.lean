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
instruction maps one state to a `PMF` of next states: arithmetic, drops,
keeps and sums to a certain state, and the rolls and rerolls to a draw. A die
is `PMF.ofMultiset` of its faces, so each face is equally likely by position,
and a set of dice is that many independent draws, recorded in the order
rolled. A reroll replaces each kept result that it rerolls by an independent
draw: limited to `n` times, from the die and then up to `n - 1` redraws, each
while the result matches, as the Rust loops; and unlimited, from the
conditional law of the die, given that it does not match, rather than from a
loop that may never end, unless every face matches, when it crosses out every
matching result instead. The conditional law is the limit of the limited
rerolls as their limit grows, which `XdySpec/Unlimited.lean` proves
(`tendsto_reroll`). The instructions never branch or loop, so a function's
meaning binds them in order and then reads the returned operand.

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
uniform over `[start, stop]`, or a certain `0` if the range is empty, drawn
from the range as its die.

# Parameters
- `start`: The least result.
- `stop`: The greatest result.

# Returns
The law of the rolling record.

# Examples
```lean
example : rollRange 3 1 = PMF.pure { results := [0], die := .range 3 1 } := by
  simp [rollRange]
```
-/
noncomputable def rollRange (start stop : Int) : PMF Record :=
  if h : stop < start then PMF.pure { results := [0], die := .range start stop }
  else
    (PMF.ofMultiset (Finset.Icc start stop).val (by simp; omega)).map
      fun x => { results := [x], die := .range start stop }

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
the order rolled, in a record without a die, which `step` then records.

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

/-! ## Rerolls -/

/--
Redraw a die that matches a criterion: one more step of a limited reroll,
applied to the law of a die's current result.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `p`: The law of the die's current result.

# Returns
The law of the result after redrawing it from `die` if `c` matches it, and
keeping it otherwise.

# Examples
```lean
example (die : PMF Int) :
    redraw (.within 1 1) die (PMF.pure 2) = PMF.pure 2 := by
  simp [redraw, Criterion.matches]
```
-/
noncomputable def redraw (c : Criterion) (die : PMF Int) (p : PMF Int) :
    PMF Int :=
  p.bind fun x => if c.matches x then die else PMF.pure x

/--
The law of the replacement of a die that a reroll limited to `limit` times
crosses out: a draw from the die, then up to `limit - 1` more redraws, each
while the result matches, as `reroll_within` in `primitives.rs` loops. Its
closed form is `limitedReplacement_apply`, in `XdySpec/Unlimited.lean`.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The number of times that each die may be rerolled, which is
  positive if any die is rerolled.

# Returns
The law of the replacement.

# Examples
```lean
example (c : Criterion) (die : PMF Int) :
    limitedReplacement c die 1 = die := rfl
```
-/
noncomputable def limitedReplacement (c : Criterion) (die : PMF Int)
    (limit : Nat) : PMF Int :=
  (limit - 1).repeat (redraw c die) die

/--
Reroll the kept results of a record, one at a time, in ascending order: each
that the reroll rerolls is replaced by an independent draw of the
replacement, in its place, and every other is kept. The record that answers
holds the results in that order, without drops, since the dice that the drops
crossed out reach nothing after a reroll.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.

# Returns
The law of the record after the reroll.

# Notes
The Rust record keeps the dice that the reroll leaves alone in place and
appends the replacements, so its kept dice are in another order. Only their
multiset reaches anything later, so the order is the specification's to
choose, and replacing in place is the simplest.

# Examples
```lean
example (p : PMF Int) : rerollKept .none [] (fun _ => true) p =
    PMF.pure { die := .none } := rfl
```
-/
noncomputable def rerollKept (die : Die) (kept : List Int)
    (rerolled : Int → Bool) (replacement : PMF Int) : PMF Record :=
  kept.foldl
    (fun p x =>
      if rerolled x then p.bind fun r => replacement.map r.push
      else p.map (·.push x))
    (PMF.pure { die })

/--
If a criterion does not match every face of a die, then the faces that it does
not match are not empty, so they have a law: the die's, given that it does not
match.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Hypotheses
- `h`: `c` does not match every face of `d`.
-/
theorem unmatched_ne_zero {d : Die} {c : Criterion}
    (h : ¬d.facesWithin c = true) :
    ((d.faces.filter (!c.matches ·) : List Int) : Multiset Int) ≠ 0 := by
  simp only [Die.facesWithin, List.all_eq_true, not_forall] at h
  obtain ⟨x, hx, hm⟩ := h
  simp only [ne_eq, Multiset.coe_eq_zero, List.filter_eq_nil_iff, not_forall]
  exact ⟨x, hx, by simpa using hm⟩

/--
The law of a record after a reroll, as `reroll` in `primitives.rs` rerolls
it. A reroll limited to `limit` times replaces each matching kept result by
its limited replacement, and rerolls nothing if `limit` is `0`. An unlimited
reroll whose criterion matches every face of the die rolls nothing, but
crosses out every matching result; otherwise, it replaces each matching
result by a draw from the faces of the die that the criterion does not
match: the conditional law of the die, given that it does not match, which
is the law of a die rerolled until it no longer matches.

# Notes
`XdySpec/Unlimited.lean` justifies the unlimited reroll against the Rust's
loop: the limited rerolls converge to it as their limit grows
(`tendsto_reroll`), since the loop stops with probability one and its first
`n` draws are the reroll limited to `n` times. If every face matches, then no
limited reroll changes the die's law (`limitedReplacement_of_matches`), so the
loop has no limit and would never stop; crossing out is a convention, under
which each matching die contributes zero (`sum_reroll_crossOut`).

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Returns
The law of the record after the reroll, which holds its kept results, without
drops, and its die.

# Examples
```lean
-- A record never rolled holds no dice to reroll.
example (c : Criterion) (l : Option Nat) : reroll {} c l = PMF.pure {} := by
  cases l <;> simp [reroll, rerollKept, Record.kept, Record.sorted,
    Die.facesWithin, Die.faces]
```
-/
noncomputable def reroll (r : Record) (c : Criterion) :
    Option Nat → PMF Record
  | none =>
    if h : r.die.facesWithin c then
      PMF.pure { results := r.kept.filter (!c.matches ·), die := r.die }
    else
      rerollKept r.die r.kept c.matches
        (PMF.ofMultiset _ (unmatched_ne_zero h))
  | some limit =>
    rerollKept r.die r.kept (fun x => 0 < limit && c.matches x)
      (limitedReplacement c (customDie r.die.faces) limit)

/-! ## Instructions -/

/--
Execute one instruction in one state.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The law of the state after the instruction. A roll replaces its rolling record
outright, discarding any drops, and records its die, as `evaluator.rs` does;
every other
instruction answers a certain state. `return` changes nothing, since `run`
reads its operand afterward.
-/
noncomputable def step (s : State) : Instruction → PMF State
  | .rollRange d a b =>
    (rollRange (s.value a) (s.value b)).map (s.setRecord d)
  | .rollStandardDice d c n =>
    (rollDice (s.value c) (standardDie (s.value n))).map fun r =>
      s.setRecord d { r with die := .standard (s.value n) }
  | .rollCustomDice d c faces =>
    (rollDice (s.value c) (customDie faces)).map fun r =>
      s.setRecord d { r with die := .ofFaces faces }
  | .dropLowest d c =>
    PMF.pure (s.setRecord d ((s.record d).dropLowest (s.value c)))
  | .dropHighest d c =>
    PMF.pure (s.setRecord d ((s.record d).dropHighest (s.value c)))
  | .keepLowest d c =>
    PMF.pure (s.setRecord d ((s.record d).keepLowest (s.value c)))
  | .keepHighest d c =>
    PMF.pure (s.setRecord d ((s.record d).keepHighest (s.value c)))
  | .reroll d p l =>
    (reroll (s.record d) (p.resolve s.value) (l.resolve s.value)).map
      (s.setRecord d)
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
