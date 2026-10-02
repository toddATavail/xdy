import Xdy.PlanLaws
import XdySpec.World

/-!
# The forward pass

The forward pass of `propagate_worlds` in `xdy/src/distribution/propagation.rs`,
as the specification sees it: in each world of the open splits, a law of each
live location, over a base state that holds everything else. The pass runs
each instruction in every world, then performs the plan's steps after it, as
the Rust does:

```mermaid
flowchart LR
    Before["forward pc"] --> Exec["run instruction pc<br/>(Forward.exec)"]
    Exec --> Steps{"next step of<br/>vs.steps pc"}
    Steps -- split --> Split["Forward.split"] --> Steps
    Steps -- merge --> Merge["Forward.merge"] --> Steps
    Steps -- none left --> After["forward (pc + 1)"]
```

The pass holds, for every world `o` of the open splits:

- `laws o ℓ`, the law of each live location `ℓ` in that world;
- `K v o`, the law of the outcome of each split `v`, given the outcomes of the
  splits beneath it;
- `base`, the state of every location whose value is not live;
- `buried`, the locations whose dead values it has buried.

An instruction writes, to its destination, the law of the instruction run on the
world of the values that it reads (`Forward.exec`); a split fixes its value at
its outcome in each world and records the value's law, read sorted, as the
split's (`Forward.split`), so that a rolling record's outcomes are the multisets
of its results, as the Rust's are; and a merge mixes the survivor's law over the
split's outcome (`Forward.merge`). The laws of the locations that are not live
do not matter, and are left as they fall.

The pass is not computable, since its laws are `PMF`s, but it is explicit, so
its laws on a concrete function unfold step by step. `XdySpec/Invariant.lean`
proves that it holds the law of states, read sorted, before every
instruction, so that a function answers the forward pass's law of its
returned register.

Unlike the Rust, the pass buries a dead value as soon as the instruction that
reads it last has run, as the invariant does, rather than when a merge's list
of the dead names it. Which it buries, and when, changes the law of no live
location.
-/

namespace Xdy.Spec

/-! ## Worlds -/

/-- An outcome for every value of a function, in its location's type, of which
only those of the open splits matter: one world of the open splits. -/
abbrev Outcomes (vs : Values) : Type := (v : Nat) → Content (vs.location v)

/-- The law of each split's outcome, given the outcomes of the splits beneath
it. -/
abbrev SplitLaws (vs : Values) : Type :=
  (v : Nat) → Outcomes vs → PMF (Content (vs.location v))

/-- The law of each live location in each world of the open splits. -/
abbrev WorldLaws (vs : Values) : Type :=
  Outcomes vs → (ℓ : Location) → PMF (Content ℓ)

variable {vs : Values}

/--
The laws of the live locations once a split merges.

# Parameters
- `K`: The laws of the splits' outcomes.
- `laws`: The laws of the live locations in each world before the merge.
- `s`: The split.
- `survivor`: The location of the only live value that depends on it, if any.

# Returns
The survivor's law mixed over the split's outcome, in each world of the other
splits, and every other law unchanged. Mirrors `merge` in `propagation.rs`.
-/
noncomputable def mergeLaws (K : SplitLaws vs) (laws : WorldLaws vs) (s : Nat) :
    Option Location → WorldLaws vs
  | none => laws
  | some ℓ => fun o => Function.update (laws o) ℓ
      ((K s o).bind fun x => laws (Function.update o s x) ℓ)

/-! ## The values that an instruction reads -/

/--
The values live before an instruction that it reads and that stay live after
it. Each is an open split or reached by no roll, so a point mass in every
world.

# Parameters
- `vs`: The values of the function.
- `pc`: The program counter of the instruction.

# Returns
The values, in ascending order.
-/
def readLive (vs : Values) (pc : Nat) : List Nat :=
  ((vs.liveBefore pc).filter fun v => (vs.reads pc).contains v).filter
    fun v => (vs.live pc).contains v

/--
The values live before an instruction that it reads for the last time, which
the pass buries once it runs.

# Parameters
- `vs`: The values of the function.
- `pc`: The program counter of the instruction.

# Returns
The values, in ascending order.
-/
def readLast (vs : Values) (pc : Nat) : List Nat :=
  ((vs.liveBefore pc).filter fun v => (vs.reads pc).contains v).filter
    fun v => !(vs.live pc).contains v

/-! ## The pass -/

/-- The state of the forward pass at a program point: the witnesses of the
plan's invariant there. -/
structure Forward (vs : Values) where
  /-- The locations buried, each of which holds a value that is not live. -/
  buried : List Location
  /-- The state of every location whose value is not live. -/
  base : State
  /-- The law of each split's outcome, given the outcomes of the splits
  beneath it. -/
  K : SplitLaws vs
  /-- The law of each live location in each world of the open splits. -/
  laws : WorldLaws vs

namespace Forward

/--
The forward pass before the first instruction: one world, the initial state,
with nothing live and no split open.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.

# Returns
The pass, which has buried nothing, and whose laws are point masses at the
blanks, which no live location reads.
-/
noncomputable def initial (vs : Values) (s : State) : Forward vs where
  buried := []
  base := s
  K := fun v _ => PMF.pure (blank (vs.location v))
  laws := fun _ ℓ => PMF.pure (blank ℓ)

/--
The locations to bury once an instruction runs, besides those of the values
that it reads for the last time: its own value's, unless it is live, and
every location buried before, but its destination, which it overwrites.

# Parameters
- `F`: The pass before the instruction.
- `pc`: The program counter of the instruction.

# Returns
The locations.
-/
def reburied (F : Forward vs) (pc : Nat) : List Location :=
  (if pc ∈ vs.live pc then [] else [vs.location pc]) ++
    F.buried.filter (· != vs.location pc)

/--
The law of an instruction's value, in a world: the law of the instruction run
on the world of the values that it reads, over the base.

# Parameters
- `F`: The pass before the instruction.
- `pc`: The program counter of the instruction.
- `i`: The instruction.
- `o`: The world.

# Returns
The law of its destination.

# Notes
This is the specification's law, not the recipe by which `propagation.rs`
computes it; the laws of `XdySpec/World.lean`, such as `law_binary` and
`law_neg`, tie the two for each instruction.
-/
noncomputable def destination (F : Forward vs) (pc : Nat) (i : Instruction)
    (o : Outcomes vs) : PMF (Content (vs.location pc)) :=
  ((product F.base ((readLive vs pc ++ readLast vs pc).map vs.location)
    (F.laws o)).bind (step · i)).map (·.get (vs.location pc))

/--
Run an instruction in every world. Mirrors the visit of each world in
`propagate_worlds`.

# Parameters
- `F`: The pass before the instruction.
- `pc`: The program counter of the instruction.
- `i`: The instruction.

# Returns
The pass after it, which buries the values that it reads for the last time,
and its own, unless it is live, and gives its destination its law in each
world (`destination`). The splits' laws are unchanged.
-/
noncomputable def exec (F : Forward vs) (pc : Nat) (i : Instruction) :
    Forward vs where
  buried := ((readLast vs pc).map vs.location).erase (vs.location pc) ++
    F.reburied pc
  base := (F.base.buryAll
    (((readLast vs pc).map vs.location).erase (vs.location pc))).buryAll
      (F.reburied pc)
  K := F.K
  laws := fun o => Function.update (F.laws o) (vs.location pc)
    (F.destination pc i o)

/--
An instruction gives its destination its law in each world.

# Parameters
- `F`: The pass before the instruction.
- `pc`: The program counter of the instruction.
- `i`: The instruction.
- `o`: The world.
-/
theorem exec_laws_self (F : Forward vs) (pc : Nat) (i : Instruction)
    (o : Outcomes vs) :
    (F.exec pc i).laws o (vs.location pc) = F.destination pc i o :=
  Function.update_self ..

/--
An instruction leaves the law of every other location alone.

# Parameters
- `F`: The pass before the instruction.
- `i`: The instruction.
- `o`: The world.

# Hypotheses
- `h`: The location is not the instruction's destination.
-/
theorem exec_laws_of_ne (F : Forward vs) {pc : Nat} (i : Instruction)
    (o : Outcomes vs) {ℓ : Location} (h : ℓ ≠ vs.location pc) :
    (F.exec pc i).laws o ℓ = F.laws o ℓ :=
  Function.update_of_ne h ..

/--
Split on a value. Mirrors `split` in `propagation.rs`.

# Parameters
- `F`: The pass before the split.
- `v`: The value.

# Returns
The pass after it, whose law of `v`'s outcome is `v`'s law read sorted
(`sortContent`), and in each world of which `v` is a point mass at its
outcome. So a rolling record splits by the multisets of its results, as
`Roll::outcomes` in `propagation/record.rs` grows them, and each world fixes
it at its results sorted; a register splits by its values.
-/
noncomputable def split (F : Forward vs) (v : Nat) : Forward vs :=
  { F with
    K := Function.update F.K v fun o =>
      (F.laws o (vs.location v)).map (sortContent (vs.location v))
    laws := fun o => Function.update (F.laws o) (vs.location v)
      (PMF.pure (o v)) }

/--
A split's value is a point mass at its outcome in each world.

# Parameters
- `F`: The pass before the split.
- `v`: The split.
- `o`: The world.
-/
theorem split_laws_self (F : Forward vs) (v : Nat) (o : Outcomes vs) :
    (F.split v).laws o (vs.location v) = PMF.pure (o v) :=
  Function.update_self ..

/--
A split leaves the law of every other location alone.

# Parameters
- `F`: The pass before the split.
- `o`: The world.

# Hypotheses
- `h`: The location is not the split's.
-/
theorem split_laws_of_ne (F : Forward vs) {v : Nat} (o : Outcomes vs)
    {ℓ : Location} (h : ℓ ≠ vs.location v) :
    (F.split v).laws o ℓ = F.laws o ℓ :=
  Function.update_of_ne h ..

/--
A split's outcome takes, in each world, the law of its value there, read
sorted.

# Parameters
- `F`: The pass before the split.
- `v`: The split.
-/
theorem split_K_self (F : Forward vs) (v : Nat) :
    (F.split v).K v = fun o =>
      (F.laws o (vs.location v)).map (sortContent (vs.location v)) :=
  Function.update_self ..

/--
A split on a register's value takes, in each world, the law of its value
there, which reading sorted leaves alone.

# Parameters
- `F`: The pass before the split.
- `v`: The split.

# Hypotheses
- `h`: The split's value is in a register.
-/
theorem split_K_self_of_register (F : Forward vs) (v : Nat) {r : Nat}
    (h : vs.location v = .register r) :
    (F.split v).K v = fun o => F.laws o (vs.location v) := by
  rw [split_K_self]
  exact funext fun o => map_sortContent_of_register h _

/--
A split leaves the law of every other split's outcome alone.

# Parameters
- `F`: The pass before the split.

# Hypotheses
- `h`: The other split is not this one.
-/
theorem split_K_of_ne (F : Forward vs) {v u : Nat} (h : u ≠ v) :
    (F.split v).K u = F.K u :=
  Function.update_of_ne h ..

/--
Merge a split. Mirrors `merge` in `propagation.rs`.

# Parameters
- `F`: The pass before the merge.
- `v`: The split.
- `survivor`: The location of the only live value that depends on it, if any.

# Returns
The pass after it, whose survivor's law is mixed over `v`'s outcome
(`mergeLaws`).
-/
noncomputable def merge (F : Forward vs) (v : Nat)
    (survivor : Option Location) : Forward vs :=
  { F with laws := mergeLaws F.K F.laws v survivor }

/--
A merge mixes its survivor's law over the split's outcome.

# Parameters
- `F`: The pass before the merge.
- `v`: The split.
- `ℓ`: The survivor's location.
- `o`: The world.
-/
theorem merge_laws_self (F : Forward vs) (v : Nat) (ℓ : Location)
    (o : Outcomes vs) :
    (F.merge v (some ℓ)).laws o ℓ =
      (F.K v o).bind fun x => F.laws (Function.update o v x) ℓ :=
  Function.update_self ..

/--
A merge leaves the law of every location but its survivor's alone.

# Parameters
- `F`: The pass before the merge.
- `v`: The split.
- `o`: The world.

# Hypotheses
- `h`: The location is not the survivor's.
-/
theorem merge_laws_of_ne (F : Forward vs) (v : Nat) {ℓ ℓ' : Location}
    (o : Outcomes vs) (h : ℓ' ≠ ℓ) :
    (F.merge v (some ℓ)).laws o ℓ' = F.laws o ℓ' :=
  Function.update_of_ne h ..

/--
Perform one step of the plan after an instruction. Mirrors the loop over
`plan.steps(pc)` in `propagate_worlds`.

# Parameters
- `pc`: The program counter of the instruction, whose value any split splits.
- `F`: The pass before the step.
- `splits`: The open splits, by value, from the bottom.
- `step`: The step.

# Returns
The pass and the open splits after it. A merge's list of the dead is ignored,
since the pass has already buried them.
-/
noncomputable def perform (pc : Nat) : Forward vs × List Nat → Step →
    Forward vs × List Nat
  | (F, splits), .split _ => (F.split pc, splits ++ [pc])
  | (F, splits), .merge k survivor _ =>
    (F.merge (splits.getD k 0) survivor, splits.eraseIdx k)

end Forward

/--
The forward pass before an instruction. Mirrors `propagate_worlds`.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.
- `pc`: The program point: the number of instructions run.

# Returns
The pass once the first `pc` instructions have run, each followed by the
plan's steps after it. Beyond the last instruction, the pass is unchanged.
-/
noncomputable def forward (vs : Values) (s : State) : Nat → Forward vs
  | 0 => .initial vs s
  | pc + 1 =>
    match vs.instructions[pc]? with
    | some i =>
      ((vs.steps pc).foldl (Forward.perform pc)
        ((forward vs s pc).exec pc i, vs.openSplits pc)).1
    | none => forward vs s pc

/--
The forward pass after an instruction: the pass before it, the instruction run
in every world, and the plan's steps after it performed in turn.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.
- `pc`: The program counter of the instruction.
- `i`: The instruction.

# Hypotheses
- `h`: `i` is the instruction at `pc`.
-/
theorem forward_succ (vs : Values) (s : State) {pc : Nat} {i : Instruction}
    (h : vs.instructions[pc]? = some i) :
    forward vs s (pc + 1) = ((vs.steps pc).foldl (Forward.perform pc)
      ((forward vs s pc).exec pc i, vs.openSplits pc)).1 := by
  rw [forward, h]

end Xdy.Spec
