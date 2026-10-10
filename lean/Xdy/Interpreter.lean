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
drops, keeps and sums to a single certain state, and the rolls, rerolls and
explodes to one state per outcome. Rolls draw one die at a time, and since
records keep their results sorted, states that differ only in the order of
their rolls merge as they arise. The instructions never branch or loop, so
the interpreter binds them in order and then reads the returned operand from
every final state.

A reroll replaces each kept result that it rerolls by a draw of one
replacement, which is the same for every rerolled result, so the interpreter
computes it once. A reroll limited to `n` times draws its replacement
naively, from the die and then up to `n - 1` redraws, as the Rust loops, but
merges outcomes after each redraw. An unlimited reroll may loop forever in
Rust, with probability zero, so the interpreter draws its replacement from
the conditional law of the die instead, given that it does not match, which is
the law to which the limited replacement converges, as
`tendsto_limitedReplacement` in `lean/spec/XdySpec/Unlimited.lean` proves.
A reroll of a record that holds chains rerolls each die of each chain alone,
and the replacement of the last die of a chain becomes its last die.

An explode grows each chain still kept whose last die matches by an
extension, which is the same for every chain that grows, so the interpreter
computes it once: limited to `n` times, a draw from the die and then up to
`n - 1` more, each while the last draw matches, merging outcomes after each.
An unlimited explode whose criterion matches every face of the die saturates
every chain still kept. Any other unlimited explode that grows a chain may
draw without bound, and its law has infinitely many outcomes, so the oracle
refuses the function, as the Rust forward pass refuses to build it
(`BuildError::Unbounded`), but exactly: only if some state that the function
reaches runs such an explode on a chain whose last die matches.

The interpreter is exponential in the worst case: `nDk` has as many states as
there are multisets of `n` faces drawn from `k`. A limited reroll or explode
costs time proportional to its limit. It is meant only for small programs.
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
result, uniform over `[start, stop]`, or a certain `0` if the range is empty,
drawn from the range as its die.

# Parameters
- `start`: The least result.
- `stop`: The greatest result.

# Returns
The distribution of the rolling record.

# Examples
```lean
#guard (rollRange 1 3).total == 3
#guard rollRange 3 1 == Dist.pure { results := [0], die := .range 3 1 }
```
-/
def rollRange (start stop : Int) : Dist Record :=
  if stop < start then .pure { results := [0], die := .range start stop }
  else
    let results :=
      (List.range (stop - start + 1).toNat).map fun (i : Nat) => start + i
    (Dist.uniform results).map fun x =>
      { results := [x], die := .range start stop }

/--
The distribution of a set of dice: `count` independent draws from `die`, one
at a time, into a record without a die, which `step` then records.

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

/-! ## Rerolls -/

/--
Redraw a die that matches a criterion: one more step of a limited reroll,
applied to the distribution of a die's current result.

# Parameters
- `c`: The criterion.
- `die`: The distribution of the die.
- `d`: The distribution of the die's current result.

# Returns
The distribution of the result after redrawing it from `die` if `c` matches
it, and keeping it otherwise.

# Examples
```lean
#guard redraw (.within 1 1) (.uniform [1, 2]) (.pure 1) == .uniform [1, 2]
#guard redraw (.within 1 1) (.uniform [1, 2]) (.pure 2) == .pure 2
```
-/
def redraw (c : Criterion) (die : Dist Int) (d : Dist Int) : Dist Int :=
  d.bind fun x => if c.matches x then die else .pure x

/--
The distribution of the replacement of a die that a reroll limited to
`limit` times crosses out: a draw from the die, then up to `limit - 1` more
redraws, each while the result matches, as `reroll_within` in `primitives.rs`
loops. Every matching die has the same replacement, so the oracle computes it
once for them all, merging outcomes after every redraw, rather than following
each chain of redraws apart.

# Parameters
- `c`: The criterion.
- `die`: The distribution of the die.
- `limit`: The number of times that each die may be rerolled, which is
  positive if any die is rerolled.

# Returns
The distribution of the replacement.

# Examples
```lean
-- `1D2 reroll = 1 2 times`: a 1 survives only by matching twice.
#guard (limitedReplacement (.within 1 1) (.uniform [1, 2]) 2).toList
  == [(1, 1), (2, 3)]
```
-/
def limitedReplacement (c : Criterion) (die : Dist Int) (limit : Nat) :
    Dist Int :=
  (limit - 1).repeat (redraw c die) die

/--
Reroll the kept results of a record, one at a time, in ascending order: each
that the reroll rerolls is replaced by a draw of the replacement, and every
other is kept. The record that answers holds the results, sorted, without
drops, since the dice that the drops crossed out reach nothing after a
reroll, as only the kept dice reach a later clause or the sum.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The distribution of the replacement of a rerolled result,
  computed only if some result is rerolled.

# Returns
The distribution of the record after the reroll.

# Examples
```lean
#guard (rerollKept (.standard 2) [1, 2] (· == 1) (.pure (.uniform [1, 2]))
  |>.map (·.results)).toList == [([1, 2], 1), ([2, 2], 1)]
```
-/
def rerollKept (die : Die) (kept : List Int) (rerolled : Int → Bool)
    (replacement : Thunk (Dist Int)) : Dist Record :=
  kept.foldl
    (fun d x =>
      if rerolled x then d.bind fun r => replacement.get.map r.push
      else d.map (·.push x))
    (.pure { die })

/--
Reroll each die of a chain, one at a time: each that the reroll rerolls is
replaced by a draw of the replacement, in its place, so the replacement of the
last die of a chain becomes its last die, and every other is kept. A saturated
chain is left alone.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The distribution of the replacement of a rerolled result,
  computed only if some result is rerolled.
- `c`: The chain.

# Returns
The distribution of the chain after the reroll.

# Examples
```lean
#guard (rerollChain (· == 1) (.pure (.uniform [1, 2])) (.live [1, 3] 1)
  |>.map (·.total)).toList == [(5, 1), (6, 2), (7, 1)]
#guard rerollChain (fun _ => true) (.pure (.uniform [1, 2])) (.saturated 0)
  == .pure (.saturated 0)
```
-/
def rerollChain (rerolled : Int → Bool) (replacement : Thunk (Dist Int)) :
    Chain → Dist Chain
  | .live body last =>
    let draw (x : Int) : Dist Int :=
      if rerolled x then replacement.get else .pure x
    (body.foldl (fun (d : Dist (List Int)) x => d.bind fun b =>
        (draw x).map fun y => (y :: b).mergeSort)
      (.pure [])).bind fun b => (draw last).map (.live b ·)
  | c => .pure c

/--
Reroll the kept chains of a record, one at a time, in order, each as
`rerollChain` rerolls it. The record that answers holds the chains, sorted,
without drops, as the kept results of a record that holds no chains are
rerolled.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The distribution of the replacement of a rerolled result,
  computed only if some result is rerolled.

# Returns
The distribution of the record after the reroll.

# Examples
```lean
#guard ((rerollChains (.standard 2) [.live [2] 1] (· == 1)
    (.pure (.uniform [1, 2]))).map (·.results)).toList == [([3], 1), ([4], 1)]
```
-/
def rerollChains (die : Die) (kept : List Chain) (rerolled : Int → Bool)
    (replacement : Thunk (Dist Int)) : Dist Record :=
  (kept.foldl
    (fun (d : Dist (List Chain)) c => d.bind fun cs =>
      (rerollChain rerolled replacement c).map fun c' =>
        (c' :: cs).mergeSort Chain.le)
    (.pure [])).map (Record.ofChains die)

/--
Reroll each kept die of a record: its kept results, as `rerollKept` rerolls
them, if it holds no chains, or else each die of its kept chains, as
`rerollChains` rerolls them.

# Parameters
- `r`: The record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The distribution of the replacement of a rerolled result,
  computed only if some result is rerolled.

# Returns
The distribution of the record after the reroll.

# Examples
```lean
#guard (rerollEach { results := [1, 2], die := .standard 2 } (· == 1)
    (.pure (.uniform [1, 2])) |>.map (·.results)).toList
  == [([1, 2], 1), ([2, 2], 1)]
```
-/
def rerollEach (r : Record) (rerolled : Int → Bool)
    (replacement : Thunk (Dist Int)) : Dist Record :=
  if r.chains.isEmpty then rerollKept r.die r.kept rerolled replacement
  else rerollChains r.die r.keptChains rerolled replacement

/--
The distribution of a record after a reroll, as `reroll` in `primitives.rs`
rerolls it. A reroll limited to `limit` times replaces each matching kept
result by its limited replacement, and rerolls nothing if `limit` is `0`. An
unlimited reroll whose criterion matches every face of the die rolls nothing,
but crosses out every matching result; otherwise, it replaces each matching
result by a draw from the faces of the die that the criterion does not match,
the law of a die rerolled until it no longer matches. A record that holds
chains is rerolled die by die, as `rerollChains` rerolls it.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Returns
The distribution of the record after the reroll, which holds its kept
results, or chains, sorted, without drops, and its die.

# Notes
If an unlimited reroll matches every face of the die, it crosses out every die
of each chain that may still grow, since each shows a face, and so crosses
out the chain whole, but leaves every saturated chain alone. The Rust record
then holds no chain that is only partly crossed out, so neither does this.

# Examples
```lean
#guard ((reroll { results := [1, 2], die := .standard 2 } (.within 1 1)
    none).map (·.results)).toList == [([2, 2], 1)]
#guard reroll { results := [1, 1], die := .standard 1 } (.within 1 1) none
  == .pure { die := .standard 1 }
#guard reroll { results := [1, 2], lowest := 1, die := .standard 2 }
    (.within 1 1) (some 3)
  == .pure { results := [2], die := .standard 2 }
```
-/
def reroll (r : Record) (c : Criterion) : Option Nat → Dist Record
  | none =>
    if r.die.facesWithin c then
      if r.chains.isEmpty then
        .pure { results := r.kept.filter (!c.matches ·), die := r.die }
      else .pure (.ofChains r.die (r.keptChains.filter (·.saturates)))
    else
      rerollEach r c.matches
        (Thunk.mk fun _ => .uniform (r.die.faces.filter (!c.matches ·)))
  | some limit =>
    rerollEach r (fun x => 0 < limit && c.matches x)
      (Thunk.mk fun _ => limitedReplacement c (customDie r.die.faces) limit)

/-! ## Explodes -/

/--
The distribution of the extension of a chain whose last die an explode
limited to `limit` times matches: a draw from the die, then up to `limit - 1`
more, each while the last draw matches, as `explode_within` in
`primitives.rs` loops. Every chain that grows has the same extension, so the
oracle computes it once for them all, merging outcomes after every draw,
rather than following each chain apart.

# Parameters
- `c`: The criterion.
- `die`: The distribution of the die.
- `limit`: The number of times that each chain may be exploded, which is
  positive if any chain grows.

# Returns
The distribution of the dice drawn but the last, sorted in ascending order,
each of which matched, and of the last die drawn, which ends the chain.

# Examples
```lean
-- `1D2 explode = 2 2 times`: a second draw only after a 2.
#guard ((extension (.within 2 2) (.uniform [1, 2]) 2).map
    fun (drawn, y) => drawn ++ [y]).toList
  == [([1], 2), ([2, 1], 1), ([2, 2], 1)]
```
-/
def extension (c : Criterion) (die : Dist Int) (limit : Nat) :
    Dist (List Int × Int) :=
  (limit - 1).repeat
    (fun d => d.bind fun (drawn, x) =>
      if c.matches x then die.map fun y => ((x :: drawn).mergeSort, y)
      else .pure (drawn, x))
    (die.map ([], ·))

/--
Explode the kept chains of a record, one at a time, in order: each that grows
gains a draw of the extension, after its last die, whose last draw becomes its
last die, and every other is kept. The record that answers holds the chains,
sorted, without drops, or its results alone, if no chain grew.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains of the record.
- `grows`: Whether the explode grows a chain.
- `extension`: The distribution of the extension of a chain that grows,
  computed only if some chain grows.

# Returns
The distribution of the record after the explode.

# Examples
```lean
#guard explodeChains (.standard 6) [.single 6, .single 3] (·.last == 6)
    (.pure (.pure ([], 2)))
  == .pure { results := [3, 8], die := .standard 6
             chains := [.single 3, .live [6] 2] }
```
-/
def explodeChains (die : Die) (kept : List Chain) (grows : Chain → Bool)
    (extension : Thunk (Dist (List Int × Int))) : Dist Record :=
  (kept.foldl
    (fun (d : Dist (List Chain)) c =>
      if grows c then
        d.bind fun cs => extension.get.map fun (drawn, y) =>
          (.live (c.dice ++ drawn).mergeSort y :: cs).mergeSort Chain.le
      else d.map fun cs => (c :: cs).mergeSort Chain.le)
    (.pure [])).map (Record.ofChains die)

/--
The distribution of a record after an explode, as `explode` in
`primitives.rs` explodes it. An explode limited to `limit` times grows each
kept chain whose last die matches by its extension, and grows nothing if
`limit` is `0`. An unlimited explode whose criterion matches every face of the
die rolls nothing, but saturates every chain still kept, at the saturated
value of the die.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.

# Returns
The distribution of the record after the explode, which holds its kept
chains, sorted, without drops, and its die.

# Notes
Any other unlimited explode that grows a chain draws without bound, and
`Function.run` refuses the function before it would explode (see
`unbounded`). So an unlimited explode answers here only if it saturates or
grows no chain, and if it does not saturate, it explodes the chains as
`explodeChains` does, growing none of them.

# Examples
```lean
#guard explode { results := [6, 6], lowest := 1, die := .standard 6 }
    (.within 1 6) none
  == .pure { results := [i32Max], die := .standard 6
             chains := [.saturated i32Max] }
#guard ((explode { results := [1, 2], die := .standard 2 } (.within 2 2)
    (some 1)).map (·.results)).toList == [([1, 3], 1), ([1, 4], 1)]
```
-/
def explode (r : Record) (c : Criterion) : Option Nat → Dist Record
  | none =>
    if r.die.facesWithin c then
      .pure (.ofChains r.die
        (r.keptChains.map (·.saturate r.die.saturatedValue)))
    else
      explodeChains r.die r.keptChains (fun _ => false)
        (Thunk.mk fun _ => .pure ([], 0))
  | some limit =>
    explodeChains r.die r.keptChains (fun ch => 0 < limit && ch.grows c)
      (Thunk.mk fun _ => extension c (customDie r.die.faces) limit)

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
  | .min => Xdy.min

/--
Execute one instruction in one state.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The distribution of states after the instruction. A roll replaces its rolling
record outright, discarding any drops, and records its die, as `evaluator.rs`
does; a reroll or explode draws too; every other
instruction answers a single certain state. `return` changes nothing, since
`Function.run` reads its operand afterward.
-/
def step (s : State) : Instruction → Dist State
  | .rollRange d a b =>
    (rollRange (s.value a) (s.value b)).map (s.setRecord d)
  | .rollStandardDice d c n =>
    (rollDice (s.value c) fun _ => standardDie (s.value n)).map fun r =>
      s.setRecord d { r with die := .standard (s.value n) }
  | .rollCustomDice d c faces =>
    (rollDice (s.value c) fun _ => customDie faces).map fun r =>
      s.setRecord d { r with die := .ofFaces faces }
  | .dropLowest d c =>
    .pure (s.setRecord d ((s.record d).dropLowest (s.value c)))
  | .dropHighest d c =>
    .pure (s.setRecord d ((s.record d).dropHighest (s.value c)))
  | .keepLowest d c =>
    .pure (s.setRecord d ((s.record d).keepLowest (s.value c)))
  | .keepHighest d c =>
    .pure (s.setRecord d ((s.record d).keepHighest (s.value c)))
  | .reroll d p l =>
    (reroll (s.record d) (p.resolve s.value) (l.resolve s.value)).map
      (s.setRecord d)
  | .explode d p l =>
    (explode (s.record d) (p.resolve s.value) (l.resolve s.value)).map
      (s.setRecord d)
  | .sumRollingRecord d r => .pure (s.setRegister d (s.record r).sum)
  | .binary op d a b =>
    .pure (s.setRegister d (op.apply (s.value a) (s.value b)))
  | .neg d a => .pure (s.setRegister d (neg (s.value a)))
  | .return _ => .pure s

/--
Whether an instruction, in a state, is an unlimited explode that may draw
without bound: whether its criterion does not match every face of its
record's die, so that it does not saturate, and yet matches the last die of
some chain still kept, which it would grow by an extension that may never end.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
`true` if `inst` is such an explode in `s`, `false` otherwise.

# Examples
```lean
#guard unbounded
    { registers := #[], records := #[{ results := [6], die := .standard 6 }] }
    (.explode 0 (.eq (.immediate 6)) .unlimited)
#guard !unbounded
    { registers := #[], records := #[{ results := [5], die := .standard 6 }] }
    (.explode 0 (.eq (.immediate 6)) .unlimited)
```
-/
def unbounded (s : State) : Instruction → Bool
  | .explode d p l =>
    let r := s.record d
    let c := p.resolve s.value
    l.resolve s.value == none && !r.die.facesWithin c
      && r.keptChains.any (·.grows c)
  | _ => false

/-- The message with which the oracle refuses a function that may explode
without bound, as `BuildError::Unbounded` in
`xdy/src/distribution/propagation.rs` describes it. -/
def unboundedMessage : String :=
  "no exact distribution: an explode without limit may grow a chain without "
    ++ "bound, so sample the function instead"

/--
Execute one instruction in every state of a distribution, unless some state
runs it as an explode that may draw without bound.

# Parameters
- `d`: The distribution of states before the instruction.
- `inst`: The instruction.

# Returns
The distribution of states after the instruction.

# Errors
With `unboundedMessage` if `inst` is `unbounded` in some state of `d`.

# Examples
```lean
#guard (stepAll (.pure { registers := #[0], records := #[{}] })
    (.rollRange 0 (.immediate 1) (.immediate 2))).toOption.map (·.total)
  == some 2
```
-/
def stepAll (d : Dist State) (inst : Instruction) :
    Except String (Dist State) :=
  if d.any (unbounded · inst) then .error unboundedMessage
  else .ok (d.bind (step · inst))

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
variables do not fit it (see `initialState`), or with `unboundedMessage` if
some state that `f` reaches with positive weight runs an explode that may
draw without bound (see `unbounded` and `stepAll`), whose distribution has
infinitely many outcomes.

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
  let final ← f.instructions.foldlM stepAll (.pure s)
  match f.instructions.back? with
  | some (.return src) => pure (final.map (·.value src))
  | _ => throw "the function has no return"

end Function

end Xdy
