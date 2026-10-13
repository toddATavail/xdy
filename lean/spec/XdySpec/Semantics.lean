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
(`tendsto_reroll`). A reroll of a record that holds chains rerolls each die of
each chain alone, and the replacement of the last die of a chain becomes its
last die.

An explode grows each chain still kept whose last die matches by an
independent extension: limited to `n` times, a draw from the die and then up
to `n - 1` more, each while the last draw matches, as the Rust loops; and
unlimited, a number of matching draws that is geometric, in the probability
that a draw does not match, each from the die given that it matches, and then
one from the die given that it does not, unless every face matches, when it
saturates every chain instead. An unlimited explode that grows a chain has
infinitely many outcomes, which the oracle refuses to enumerate, so the
oracle meets this specification only where it answers. The instructions never
branch or loop, so a function's meaning binds them in order and then reads
the returned operand.

This is the specification that the oracle, `Xdy.Function.run`, must meet, and
does, by `run_eq` in `XdySpec/Oracle.lean`. It
has the same shape, instruction by instruction, but it differs in how it
represents what it computes: `PMF` probabilities are extended nonnegative
reals rather than natural weights over one total, nothing merges, since equal
outcomes are equal as elements of the support, and records keep their results
in roll order rather than sorted. It cannot be computed, only reasoned about.
Validation and the initial state are deterministic, so it reuses the oracle's,
and so are the chains of a record, its order of drops and keeps, and which
chains an explode may grow without bound, so it reuses the oracle's chains.
-/

open scoped ENNReal

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
Reroll each die of a chain, one at a time: each that the reroll rerolls is
replaced by an independent draw of the replacement, in its place, so the
replacement of the last die of a chain becomes its last die, and every other
is kept. A saturated chain is left alone. Mirrors the oracle's
`Xdy.rerollChain`, which keeps the dice of each chain besides its last
sorted, as this does.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.
- `c`: The chain.

# Returns
The law of the chain after the reroll.

# Examples
```lean
example (p : PMF Int) : rerollChain (fun _ => true) p (.saturated 0) =
    PMF.pure (.saturated 0) := rfl
```
-/
noncomputable def rerollChain (rerolled : Int → Bool) (replacement : PMF Int) :
    Chain → PMF Chain
  | .live body last =>
    let draw (x : Int) : PMF Int :=
      if rerolled x then replacement else PMF.pure x
    (body.foldl (fun (p : PMF (List Int)) x => p.bind fun b =>
        (draw x).map fun y => (y :: b).mergeSort)
      (PMF.pure [])).bind fun b => (draw last).map (.live b ·)
  | c => PMF.pure c

/--
Reroll the kept chains of a record, one at a time, in order, each as
`rerollChain` rerolls it. The record that answers holds the chains, without
drops. Mirrors the oracle's `Xdy.rerollChains`.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.

# Returns
The law of the record after the reroll.

# Examples
```lean
example (p : PMF Int) : rerollChains .none [] (fun _ => true) p =
    PMF.pure (Record.ofChains .none []) := by
  simp [rerollChains, PMF.pure_map]
```
-/
noncomputable def rerollChains (die : Die) (kept : List Chain)
    (rerolled : Int → Bool) (replacement : PMF Int) : PMF Record :=
  (kept.foldl
    (fun (p : PMF (List Chain)) c => p.bind fun cs =>
      (rerollChain rerolled replacement c).map fun c' =>
        (c' :: cs).mergeSort Chain.le)
    (PMF.pure [])).map (Record.ofChains die)

/--
Reroll each kept die of a record: its kept results, as `rerollKept` rerolls
them, if it holds no chains, or else each die of its kept chains, as
`rerollChains` rerolls them.

# Parameters
- `r`: The record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement of a rerolled result.

# Returns
The law of the record after the reroll.

# Examples
```lean
example (rerolled : Int → Bool) (p : PMF Int) :
    rerollEach {} rerolled p = PMF.pure {} := by
  simp [rerollEach, rerollKept, Record.kept, Record.sorted]
```
-/
noncomputable def rerollEach (r : Record) (rerolled : Int → Bool)
    (replacement : PMF Int) : PMF Record :=
  if r.chains.isEmpty then rerollKept r.die r.kept rerolled replacement
  else rerollChains r.die r.keptChains rerolled replacement

/--
The law of a record after a reroll, as `reroll` in `primitives.rs` rerolls
it. A reroll limited to `limit` times replaces each matching kept result by
its limited replacement, and rerolls nothing if `limit` is `0`. An unlimited
reroll whose criterion matches every face of the die rolls nothing, but
crosses out every matching result; otherwise, it replaces each matching
result by a draw from the faces of the die that the criterion does not
match: the conditional law of the die, given that it does not match, which
is the law of a die rerolled until it no longer matches. A record that holds
chains is rerolled die by die, as `rerollChains` rerolls it.

# Notes
`XdySpec/Unlimited.lean` justifies the unlimited reroll of a record without
chains against the Rust's loop: the limited rerolls converge to it as their
limit grows (`tendsto_reroll`), since the loop stops with probability one and
its first `n` draws are the reroll limited to `n` times. If every face
matches, then no limited reroll changes the die's law
(`limitedReplacement_of_matches`), so the loop has no limit and would never
stop; crossing out is a convention, under which each matching die contributes
zero (`sum_reroll_crossOut`). Every die of a chain that may still grow shows
a face, so such a reroll crosses out each such chain whole, and leaves every
saturated chain alone, as the Rust does.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Returns
The law of the record after the reroll, which holds its kept results, or
chains, without drops, and its die.

# Examples
```lean
-- A record never rolled holds no dice to reroll.
example (c : Criterion) (l : Option Nat) : reroll {} c l = PMF.pure {} := by
  cases l <;> simp [reroll, rerollEach, rerollKept, Record.kept,
    Record.sorted, Die.facesWithin, Die.faces]
```
-/
noncomputable def reroll (r : Record) (c : Criterion) :
    Option Nat → PMF Record
  | none =>
    if h : r.die.facesWithin c then
      if r.chains.isEmpty then
        PMF.pure { results := r.kept.filter (!c.matches ·), die := r.die }
      else PMF.pure (.ofChains r.die (r.keptChains.filter (·.saturates)))
    else
      rerollEach r c.matches (PMF.ofMultiset _ (unmatched_ne_zero h))
  | some limit =>
    rerollEach r (fun x => 0 < limit && c.matches x)
      (limitedReplacement c (customDie r.die.faces) limit)

/-! ## Explodes -/

/--
The law of the extension of a chain whose last die an explode limited to
`limit` times matches: a draw from the die, then up to `limit - 1` more, each
while the last draw matches, as `explode_within` in `primitives.rs` loops.
Mirrors the oracle's `Xdy.extension`.

# Parameters
- `c`: The criterion.
- `die`: The law of the die.
- `limit`: The number of times that each chain may be exploded, which is
  positive if any chain grows.

# Returns
The law of the dice drawn but the last, sorted in ascending order, each of
which matched, and of the last die drawn, which ends the chain.

# Examples
```lean
example (c : Criterion) (die : PMF Int) :
    extension c die 1 = die.map ([], ·) := rfl
```
-/
noncomputable def extension (c : Criterion) (die : PMF Int) (limit : Nat) :
    PMF (List Int × Int) :=
  (limit - 1).repeat
    (fun p => p.bind fun (drawn, x) =>
      if c.matches x then die.map fun y => ((x :: drawn).mergeSort, y)
      else PMF.pure (drawn, x))
    (die.map ([], ·))

/--
The geometric law of the number of draws that match before the first that
does not: `k` with probability `q ^ k * (1 - q)`, for `q` the probability that
a draw matches.

# Parameters
- `q`: The probability that a draw matches.
- `hq`: `q` is less than one, so some draw fails to match.

# Returns
The law of the number of matching draws.

# Notes
Mathlib's `ProbabilityTheory.geometricPMF`, which counts the same way over
real probabilities, is deprecated in favor of a measure, so this builds the
law directly, from the sum of the geometric series in `ℝ≥0∞`.

# Examples
```lean
example : geometric 0 (by norm_num) 0 = 1 := by simp [geometric_apply]
```
-/
noncomputable def geometric (q : ℝ≥0∞) (hq : q < 1) : PMF ℕ :=
  ⟨fun k => q ^ k * (1 - q), by
    rw [ENNReal.summable.hasSum_iff, ENNReal.tsum_mul_right,
      ENNReal.tsum_geometric]
    exact ENNReal.inv_mul_cancel (tsub_pos_iff_lt.mpr hq).ne'
      (ENNReal.sub_ne_top ENNReal.one_ne_top)⟩

/--
The probability of each number of matching draws.

# Parameters
- `q`: The probability that a draw matches.
- `hq`: `q` is less than one.
- `k`: The number of matching draws.
-/
@[simp]
theorem geometric_apply (q : ℝ≥0∞) (hq : q < 1) (k : ℕ) :
    geometric q hq k = q ^ k * (1 - q) := rfl

/--
The probability that a draw from a die matches a criterion: the fraction of
its faces, by position, that the criterion matches.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Returns
The probability.
-/
noncomputable def matchedFaces (d : Die) (c : Criterion) : ℝ≥0∞ :=
  ((d.faces.filter c.matches).length : ℝ≥0∞) / d.faces.length

/--
If a criterion does not match every face of a die, then a draw from the die
matches it with probability less than one.

# Parameters
- `d`: The die.
- `c`: The criterion.

# Hypotheses
- `h`: `c` does not match every face of `d`.
-/
theorem matchedFaces_lt_one {d : Die} {c : Criterion}
    (h : ¬d.facesWithin c = true) : matchedFaces d c < 1 := by
  simp only [Die.facesWithin, List.all_eq_true, not_forall] at h
  obtain ⟨x, hx, hm⟩ := h
  have hlt : (d.faces.filter c.matches).length < d.faces.length :=
    List.length_filter_lt_length_iff_exists.mpr ⟨x, hx, by simpa using hm⟩
  have hpos : d.faces.length ≠ 0 := by omega
  unfold matchedFaces
  rw [ENNReal.div_lt_iff (.inl (by exact_mod_cast hpos)) (.inl (by simp)),
    one_mul]
  exact_mod_cast hlt

/--
The law of the extension of a chain whose last die an unlimited explode
matches, when the explode does not match every face of the die: draws from
the die while they match, and then one that does not. The number of matching
draws is `geometric`, in the probability that a draw matches,
`matchedFaces`; each matching draw falls as the die given that it matches,
and the last as the die given that it does not.

# Parameters
- `c`: The criterion.
- `d`: The die.
- `h`: `c` does not match every face of `d`, so some draw ends the chain.

# Returns
The law of the dice drawn but the last, sorted in ascending order, each of
which matched, and of the last die drawn, which ends the chain.

# Notes
This is the law of the Rust's loop, which stops with probability one: a
chain of `k` matching draws and then one that does not, each a face by
position, has probability `F^-(k+1)`, for a die of `F` faces, which is the
probability of `k` here, `(m/F)^k · (1 - m/F)`, for `m` matching and `u`
other faces, so `1 - m/F = u/F`, times `m^-k · u^-1`, the probability of the
draws given that they match or do not. This module does not prove that the
limited explodes converge to it as their limit grows, as
`XdySpec/Unlimited.lean` proves of rerolls. The law has infinitely many
outcomes whenever some face matches, and the oracle refuses every function
that reaches such an explode, so `run_eq` in `XdySpec/Oracle.lean` does not
rest on it.
-/
noncomputable def unlimitedExtension (c : Criterion) (d : Die)
    (h : ¬d.facesWithin c = true) : PMF (List Int × Int) :=
  (geometric (matchedFaces d c) (matchedFaces_lt_one h)).bind fun k =>
    (k.repeat
      (fun (p : PMF (List Int)) => p.bind fun drawn =>
        (customDie (d.faces.filter c.matches)).map fun x =>
          (x :: drawn).mergeSort)
      (PMF.pure [])).bind fun drawn =>
      (PMF.ofMultiset _ (unmatched_ne_zero h)).map (drawn, ·)

/--
Explode the kept chains of a record, one at a time, in order: each that grows
gains an independent draw of the extension, after its last die, whose last
draw becomes its last die, and every other is kept. The record that answers
holds the chains, without drops, or its results alone, if no chain grew.
Mirrors the oracle's `Xdy.explodeChains`.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains of the record.
- `grows`: Whether the explode grows a chain.
- `extension`: The law of the extension of a chain that grows.

# Returns
The law of the record after the explode.

# Examples
```lean
example (p : PMF (List Int × Int)) :
    explodeChains .none [] (fun _ => true) p =
      PMF.pure (Record.ofChains .none []) := by
  simp [explodeChains, PMF.pure_map]
```
-/
noncomputable def explodeChains (die : Die) (kept : List Chain)
    (grows : Chain → Bool) (extension : PMF (List Int × Int)) : PMF Record :=
  (kept.foldl
    (fun (p : PMF (List Chain)) c =>
      if grows c then
        p.bind fun cs => extension.map fun (drawn, y) =>
          (.live (c.dice ++ drawn).mergeSort y :: cs).mergeSort Chain.le
      else p.map fun cs => (c :: cs).mergeSort Chain.le)
    (PMF.pure [])).map (Record.ofChains die)

/--
An explode of chains grows the same chains, and so explodes them alike,
under any two tests that agree on every chain.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `grows`, `grows'`: The tests of whether the explode grows a chain.
- `ext`: The extension.

# Hypotheses
- `h`: The tests agree on every kept chain.
-/
theorem explodeChains_congr (die : Die) {kept : List Chain}
    {grows grows' : Chain → Bool} (ext : PMF (List Int × Int))
    (h : ∀ c ∈ kept, grows c = grows' c) :
    explodeChains die kept grows ext = explodeChains die kept grows' ext := by
  unfold explodeChains
  congr 1
  generalize (PMF.pure [] : PMF (List Chain)) = p
  induction kept generalizing p with
  | nil => rfl
  | cons c kept ih =>
    simp only [List.foldl_cons]
    rw [h c List.mem_cons_self]
    exact ih (fun c hc => h c (List.mem_cons_of_mem _ hc)) _

/--
The law of a record after an explode, as `explode` in `primitives.rs`
explodes it. An explode limited to `limit` times grows each kept chain whose
last die matches by its extension, and grows nothing if `limit` is `0`. An
unlimited explode whose criterion matches every face of the die rolls
nothing, but saturates every chain still kept, at the saturated value of the
die; any other grows each kept chain whose last die matches by its unlimited
extension.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.

# Returns
The law of the record after the explode, which holds its kept chains, without
drops, and its die.

# Examples
```lean
-- A record never rolled holds no dice to explode.
example (c : Criterion) (l : Option Nat) :
    explode {} c l = PMF.pure (Record.ofChains .none []) := by
  cases l <;> simp [explode, explodeChains, Record.keptChains,
    Record.kept, Record.sorted, Die.facesWithin, Die.faces, PMF.pure_map]
```
-/
noncomputable def explode (r : Record) (c : Criterion) :
    Option Nat → PMF Record
  | none =>
    if h : r.die.facesWithin c then
      PMF.pure (.ofChains r.die
        (r.keptChains.map (·.saturate r.die.saturatedValue)))
    else
      explodeChains r.die r.keptChains (·.grows c)
        (unlimitedExtension c r.die h)
  | some limit =>
    explodeChains r.die r.keptChains (fun ch => 0 < limit && ch.grows c)
      (extension c (customDie r.die.faces) limit)

/-! ## Instructions -/

/--
Execute one instruction in one state.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The law of the state after the instruction. A roll replaces its rolling record
outright, discarding any drops, and records its die, as `evaluator.rs` does;
a reroll or explode draws too; every other
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
  | .explode d p l =>
    (explode (s.record d) (p.resolve s.value) (l.resolve s.value)).map
      (s.setRecord d)
  | .sumRollingRecord d r => PMF.pure (s.setRegister d (s.record r).sum)
  | .binary op d a b =>
    PMF.pure (s.setRegister d (op.apply (s.value a) (s.value b)))
  | .neg d a => PMF.pure (s.setRegister d (neg (s.value a)))
  | .return _ => PMF.pure s

/--
Whether an instruction, in a state, is an unlimited explode that may draw
without bound: whether its criterion does not match every face of its
record's die, so that it does not saturate, and yet matches the last die of
some chain still kept, whose unlimited extension has infinitely many
outcomes. Mirrors the oracle's `Xdy.unbounded`, which the oracle refuses.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
`true` if `inst` is such an explode in `s`, `false` otherwise.
-/
def unbounded (s : State) : Instruction → Bool
  | .explode d p l =>
    let r := s.record d
    let c := p.resolve s.value
    l.resolve s.value == none && !r.die.facesWithin c
      && r.keptChains.any (·.grows c)
  | _ => false

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
