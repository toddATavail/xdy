import XdySpec.Ghost
import XdySpec.Paths

/-!
# Lemma 7: the worlds never outnumber the paths

Lemma 7 of the design: conditioning never costs more than enumerating paths.
The forward pass in `xdy/src/distribution/propagation.rs` keeps one world for
each assignment of outcomes to its open splits that it can reach, and the
space that it needs grows with them. The enumerating builder that it
replaced walked one path for each sequence of draws (`XdySpec/Paths.lean`).
At every program point, and after every step of the plan, the worlds of the
pass number no more than the paths that reach that point, and so no more than
the paths through the whole function (`worlds_le_paths`,
`worlds_le_paths_of_step`).

A path fixes every draw, so it fixes the state at every point, and each split's
outcome, which is the value that the state holds when the split opens, read
sorted. So the ghost of `XdySpec/Ghost.lean`, the state beside the outcomes of
the open splits, is a function of the path: `ghostPaths` follows it along each
path of `XdySpec/Paths.lean`, one entry per path. Every state and outcomes that
the ghost can reach is among them (`mem_ghostPaths`), since every state that an
instruction can reach is among its forks. The pass's worlds are the outcomes
that the ghost can reach (`ghost_snd`), so each is the outcomes of some path,
and distinct worlds are the outcomes of distinct paths:

```mermaid
flowchart LR
    Paths["paths that reach<br/>the point"]
    Entries["ghostPaths"]
    Worlds["worlds of the pass"]
    Paths -->|"follow the ghost<br/>(one entry each)"| Entries
    Entries -->|"outcomes<br/>(onto)"| Worlds
```

The worlds are counted as the support of `splitOutcomes`: the assignments of
outcomes that the pass reaches with positive probability. A rolling record's
outcomes are the multisets of its results, as the Rust's are, and the Rust keeps
one world for each outcome of positive weight at each split, and merges the
worlds that agree on every other split, so its count is that support's size.
That correspondence is argued, not proved, here.
-/

namespace Xdy.Spec

variable {vs : Values}

/-! ## The ghost along each path -/

namespace Ghost

/--
Run an instruction on each path, beside the outcomes of the open splits,
which it leaves alone. Mirrors `Ghost.exec`, with each draw's law replaced by
its forks.

# Parameters
- `i`: The instruction.
- `E`: The state and the outcomes at the end of each path before it.

# Returns
The state and the outcomes at the end of each path after it.
-/
def execPaths (i : Instruction) (E : List (State × Outcomes vs)) :
    List (State × Outcomes vs) :=
  E.flatMap fun p => (forks p.1 i).map (·, p.2)

/--
Perform one step of the plan on the outcomes beside the state at the end of
each path. Mirrors `Ghost.perform`.

# Parameters
- `pc`: The program counter of the instruction, whose value any split splits.
- `E`: The state and the outcomes at the end of each path before the step.
- `splits`: The open splits, by value, from the bottom.
- `step`: The step.

# Returns
The state and the outcomes at the end of each path after it, and the open
splits after it.
-/
def performPaths (pc : Nat) :
    List (State × Outcomes vs) × List Nat → Step →
      List (State × Outcomes vs) × List Nat
  | (E, splits), .split _ => (E.map (split pc), splits ++ [pc])
  | (E, splits), .merge k _ _ =>
    (E.map (merge (splits.getD k 0)), splits.eraseIdx k)

end Ghost

/--
The state and the outcomes of the open splits at the end of each path that
reaches an instruction. Mirrors `ghost`.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.
- `pc`: The program point: the number of instructions run.

# Returns
One entry for each path of the first `pc` instructions, each followed by the
plan's steps after it. Beyond the last instruction, the entries are
unchanged.
-/
def ghostPaths (vs : Values) (s : State) : Nat → List (State × Outcomes vs)
  | 0 => [(s, blanks vs)]
  | pc + 1 =>
    match vs.instructions[pc]? with
    | some i =>
      ((vs.steps pc).foldl (Ghost.performPaths pc)
        (Ghost.execPaths i (ghostPaths vs s pc), vs.openSplits pc)).1
    | none => ghostPaths vs s pc

/--
The plan's steps change only the outcomes beside the states.

# Parameters
- `pc`: The program counter of the instruction.
- `steps`: The steps.
- `E`: The state and the outcomes at the end of each path before them.
- `splits`: The open splits before them.
-/
theorem map_fst_foldl_performPaths (pc : Nat) (steps : List Step)
    (E : List (State × Outcomes vs)) (splits : List Nat) :
    (steps.foldl (Ghost.performPaths pc) (E, splits)).1.map Prod.fst =
      E.map Prod.fst := by
  induction steps generalizing E splits with
  | nil => rfl
  | cons st steps ih =>
    cases st <;> simp only [List.foldl_cons, Ghost.performPaths] <;> rw [ih] <;>
      simp [Ghost.split, Ghost.merge, Function.comp_def]

/--
Running an instruction on each path beside the outcomes forks the states as
`enumerate` does.

# Parameters
- `i`: The instruction.
- `E`: The state and the outcomes at the end of each path before it.
-/
theorem map_fst_execPaths (i : Instruction) (E : List (State × Outcomes vs)) :
    (Ghost.execPaths i E).map Prod.fst = (E.map Prod.fst).flatMap (forks · i) :=
  by simp [Ghost.execPaths, List.map_flatMap, List.flatMap_map,
    Function.comp_def]

/--
The ghost has one entry for each path: at the end of each path that reaches
an instruction, its state is the path's.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hpc`: `pc` is at most the number of instructions.
-/
theorem map_fst_ghostPaths {s : State} {pc : Nat}
    (hpc : pc ≤ vs.instructions.length) :
    (ghostPaths vs s pc).map Prod.fst = paths s (vs.instructions.take pc) := by
  induction pc with
  | zero => rfl
  | succ pc ih =>
    have hlt : pc < vs.instructions.length := by omega
    rw [List.take_add_one, List.getElem?_eq_getElem hlt, Option.toList_some,
      paths, enumerate_append, ← paths, ← ih (by omega)]
    simp only [ghostPaths, List.getElem?_eq_getElem hlt]
    rw [map_fst_foldl_performPaths, map_fst_execPaths]
    rfl

/-! ## Every reachable ghost is on a path -/

/--
Every state and outcomes that running an instruction can reach, beside the
outcomes, is at the end of some path, if every one before it is.

# Parameters
- `i`: The instruction.
- `γ`: The law of states and outcomes before it.
- `E`: The state and the outcomes at the end of each path before it.

# Hypotheses
- `h`: Every state and outcomes that `γ` can reach is in `E`.
-/
theorem mem_execPaths {i : Instruction} {γ : PMF (State × Outcomes vs)}
    {E : List (State × Outcomes vs)} (h : ∀ p ∈ γ.support, p ∈ E) :
    ∀ p ∈ (Ghost.exec i γ).support, p ∈ Ghost.execPaths i E := by
  intro p hp
  simp only [Ghost.exec, PMF.mem_support_bind_iff, PMF.support_map,
    Set.mem_image] at hp
  obtain ⟨q, hq, t, ht, rfl⟩ := hp
  exact List.mem_flatMap.mpr ⟨q, h q hq,
    List.mem_map_of_mem (mem_forks_of_mem_support ht)⟩

/--
Every state and outcomes that the plan's steps can reach is at the end of
some path, if every one before them is.

# Parameters
- `pc`: The program counter of the instruction.
- `steps`: The steps.
- `γ`: The law of states and outcomes before them.
- `E`: The state and the outcomes at the end of each path before them.
- `splits`: The open splits before them.

# Hypotheses
- `h`: Every state and outcomes that `γ` can reach is in `E`.
-/
theorem mem_foldl_performPaths {pc : Nat} :
    ∀ {steps : List Step} {γ : PMF (State × Outcomes vs)}
      {E : List (State × Outcomes vs)} {splits : List Nat},
      (∀ p ∈ γ.support, p ∈ E) →
        ∀ p ∈ (steps.foldl (Ghost.perform pc) (γ, splits)).1.support,
          p ∈ (steps.foldl (Ghost.performPaths pc) (E, splits)).1
  | [], _, _, _, h => h
  | .split _ :: _, _, _, _, h => by
    simp only [List.foldl_cons, Ghost.perform, Ghost.performPaths]
    refine mem_foldl_performPaths fun p hp => ?_
    simp only [PMF.support_map, Set.mem_image] at hp
    obtain ⟨q, hq, rfl⟩ := hp
    exact List.mem_map_of_mem (h q hq)
  | .merge _ _ _ :: _, _, _, _, h => by
    simp only [List.foldl_cons, Ghost.perform, Ghost.performPaths]
    refine mem_foldl_performPaths fun p hp => ?_
    simp only [PMF.support_map, Set.mem_image] at hp
    obtain ⟨q, hq, rfl⟩ := hp
    exact List.mem_map_of_mem (h q hq)

/--
Every state and outcomes that the ghost can reach before an instruction is at
the end of some path that reaches it.

# Parameters
- `s`: The initial state.
- `pc`: The program point.
- `p`: The state and outcomes.

# Hypotheses
- `h`: The ghost can reach `p`.
-/
theorem mem_ghostPaths {s : State} :
    ∀ {pc : Nat} {p : State × Outcomes vs}, p ∈ (ghost vs s pc).support →
      p ∈ ghostPaths vs s pc
  | 0, _, h => by simpa [ghost, ghostPaths] using h
  | pc + 1, p, h => by
    simp only [ghost, ghostPaths] at h ⊢
    cases hi : vs.instructions[pc]? with
    | some i =>
      simp only [hi] at h ⊢
      exact mem_foldl_performPaths (mem_execPaths fun _ hq =>
        mem_ghostPaths hq) p h
    | none =>
      simp only [hi] at h ⊢
      exact mem_ghostPaths h

/-! ## Counting -/

/--
A law each of whose outcomes is the second component of some entry of a list
has no more outcomes than the list has entries.

# Parameters
- `W`: The law.
- `E`: The list.

# Hypotheses
- `h`: Each outcome of `W` is the second component of some entry.
-/
theorem encard_support_le_length {α β : Type} {W : PMF α}
    {E : List (β × α)} (h : ∀ o ∈ W.support, ∃ p ∈ E, p.2 = o) :
    W.support.encard ≤ E.length := by
  classical
  calc W.support.encard
      ≤ ((E.map Prod.snd).toFinset : Set α).encard := Set.encard_mono
        fun o ho => by
          obtain ⟨p, hp, rfl⟩ := h o ho
          simpa using ⟨p.1, hp⟩
    _ = ((E.map Prod.snd).toFinset.card : ℕ∞) :=
        Set.encard_coe_eq_coe_finsetCard _
    _ ≤ E.length := by
        exact_mod_cast (List.toFinset_card_le _).trans (by simp)

/--
The outcomes that a law of states and outcomes can reach, each of whose
entries is in a list, number no more than the list's entries.

# Parameters
- `γ`: The law.
- `E`: The list.

# Hypotheses
- `h`: Every state and outcomes that `γ` can reach is in `E`.
-/
theorem encard_support_map_snd_le {γ : PMF (State × Outcomes vs)}
    {E : List (State × Outcomes vs)} (h : ∀ p ∈ γ.support, p ∈ E) :
    (γ.map Prod.snd).support.encard ≤ E.length :=
  encard_support_le_length fun o ho => by
    simp only [PMF.support_map, Set.mem_image] at ho
    obtain ⟨p, hp, rfl⟩ := ho
    exact ⟨p, h p hp, rfl⟩

/-! ## Lemma 7 -/

/--
Before each instruction, the worlds of the forward pass number no more than
the paths that reach it.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hhas`: The initial state has the destination of every instruction.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem worlds_le_paths_forward
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true) {s : State}
    (hhas : ∀ i ∈ vs.instructions, s.Has i.destination) {pc : Nat}
    (hpc : pc ≤ vs.instructions.length) :
    (splitOutcomes (forward vs s pc).K (vs.openSplits pc)).support.encard ≤
      (paths s (vs.instructions.take pc)).length := by
  rw [← ghost_snd hvals hhas hpc, ← map_fst_ghostPaths hpc, List.length_map]
  exact encard_support_map_snd_le fun _ => mem_ghostPaths

/--
After each step of the plan that follows an instruction, the worlds of the
forward pass number no more than the paths that reach the instruction's end.

# Parameters
- `s`: The initial state.
- `pc`: The program counter of the instruction.
- `n`: The number of steps performed.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hhas`: The initial state has the destination of every instruction.
- `hpc`: `pc` is an instruction of the function.
-/
theorem worlds_le_paths_steps
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true) {s : State}
    (hhas : ∀ i ∈ vs.instructions, s.Has i.destination) {pc : Nat}
    (hpc : pc < vs.instructions.length) (n : Nat) :
    let r := ((vs.steps pc).take n).foldl (Forward.perform pc)
      ((forward vs s pc).exec pc vs.instructions[pc], vs.openSplits pc)
    (splitOutcomes r.1.K r.2).support.encard ≤
      (paths s (vs.instructions.take (pc + 1))).length := by
  obtain ⟨h1, -, -, h4⟩ := traces_steps hvals hpc
    (invariant_exec hvals hhas (Nat.le_of_lt hpc))
    (traces_forward hvals hhas (Nat.le_of_lt hpc)) n
  intro r
  rw [← h4.map_snd, List.take_add_one, List.getElem?_eq_getElem hpc,
    Option.toList_some, paths, enumerate_append, ← paths,
    ← map_fst_ghostPaths (Nat.le_of_lt hpc)]
  show _ ≤ ((((ghostPaths vs s pc).map Prod.fst).flatMap
    (forks · vs.instructions[pc])).length : ℕ∞)
  rw [← map_fst_execPaths, ← map_fst_foldl_performPaths pc
    ((vs.steps pc).take n) _ (vs.openSplits pc), List.length_map]
  exact encard_support_map_snd_le
    (mem_foldl_performPaths (mem_execPaths fun _ => mem_ghostPaths))

/--
Lemma 7, before each instruction: for a well-formed function, the worlds of
the forward pass never outnumber the paths through the function.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hv`: The function is well formed.
- `hs`: Its initial state is `s`.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem worlds_le_paths {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (hv : f.validate = .ok ()) (hs : f.initialState args externals = .ok s)
    {pc : Nat} (hpc : pc ≤ f.instructions.size) :
    (splitOutcomes (forward ⟨f.instructions.toList⟩ (.ofOracle s) pc).K
        ((Values.mk f.instructions.toList).openSplits pc)).support.encard ≤
      (paths (.ofOracle s) f.instructions.toList).length :=
  (worlds_le_paths_forward (operandsAreValues_of_validate hv)
    (has_initialState hv hs) (by simpa using hpc)).trans
      (by exact_mod_cast length_paths_take_le _ _ _)

/--
Lemma 7, after each step of the plan: for a well-formed function, the worlds
of the forward pass never outnumber the paths through the function, after any
number of the steps that follow any instruction, which is when
`propagate_worlds` counts them.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.
- `pc`: The program counter of the instruction.
- `n`: The number of steps performed.

# Hypotheses
- `hv`: The function is well formed.
- `hs`: Its initial state is `s`.
- `hpc`: `pc` is an instruction of the function.
-/
theorem worlds_le_paths_of_step {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (hv : f.validate = .ok ()) (hs : f.initialState args externals = .ok s)
    {pc : Nat} (hpc : pc < f.instructions.toList.length) (n : Nat) :
    let vs : Values := ⟨f.instructions.toList⟩
    let r := ((vs.steps pc).take n).foldl (Forward.perform pc)
      ((forward vs (.ofOracle s) pc).exec pc vs.instructions[pc],
        vs.openSplits pc)
    (splitOutcomes r.1.K r.2).support.encard ≤
      (paths (.ofOracle s) f.instructions.toList).length :=
  (worlds_le_paths_steps (vs := ⟨f.instructions.toList⟩)
    (operandsAreValues_of_validate hv) (has_initialState hv hs) hpc n).trans
      (by exact_mod_cast length_paths_take_le _ _ _)

end Xdy.Spec
