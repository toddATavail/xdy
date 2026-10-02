import XdySpec.Invariant

/-!
# The worlds' outcomes

The forward pass draws each world of its open splits with the probability that
`splitOutcomes` gives it. This module proves that this is the true joint law of
the splits' outcomes: the law with which running the function reaches them.

A split's outcome is the value that its instruction wrote, read sorted, but the
state need not keep it: once the split's value is dead, a later instruction may
overwrite its location while the split stays open, since live values still
depend on it. So the proof follows a _ghost_ beside the state: the outcomes of
the open splits, which a split copies from the state, sorted, as the pass fixes
it (`Ghost.split`), and a merge forgets (`Ghost.merge`). The ghost's law
(`ghost`) runs every instruction on the state and leaves the ghost alone, then
performs the plan's steps on the ghost, as the forward pass does on its worlds.

The invariant, `Traces`, holds of a law of states and ghosts when, once the
pass's dead locations are buried, it is, with its states read sorted, the
mixture, over the worlds of the open splits, of each world's law of states, read
sorted, tagged with its outcomes, as `Worlds` holds of the states alone. It
holds of the initial state (`traces_zero`), and each instruction, split and
merge keeps it (`traces_step`, `Worlds.traces_split`, `Worlds.traces_merge`),
with the facts from which `XdySpec/Invariant.lean` proves the plan's invariant,
so it holds throughout (`traces_steps`, `traces_forward`). Forgetting the
states, the ghost's law is the law of the pass's worlds (`Traces.map_snd`):

```mermaid
flowchart LR
    Run["run the function,<br/>copying each split's outcome"]
    Ghost["law of states<br/>and outcomes"]
    Worlds["law of the outcomes<br/>of the open splits"]
    Pass["forward pass"]
    Run --> Ghost
    Ghost -->|"forget the states"| Worlds
    Pass -->|"splitOutcomes"| Worlds
```

`XdySpec/Cost.lean` uses this to bound the worlds of the pass by the paths
through the function, since every world that the pass draws is one that some
path reaches.
-/

namespace Xdy.Spec

variable {vs : Values}

/-! ## The ghost -/

/--
The outcomes of no split, as `splitOutcomes` draws them from: every value
blank.

# Parameters
- `vs`: The values of the function.

# Returns
The blank of each value's location.
-/
def blanks (vs : Values) : Outcomes vs := fun v => blank (vs.location v)

namespace Ghost

/--
Split a value, beside a state: its outcome is the value in its location, read
sorted, as `Forward.split` reads it.

# Parameters
- `v`: The value.
- `p`: The state and the outcomes of the open splits.

# Returns
The state, and the outcomes with `v`'s read from the state, sorted.
-/
def split (v : Nat) (p : State × Outcomes vs) : State × Outcomes vs :=
  (p.1, Function.update p.2 v
    (sortContent (vs.location v) (p.1.get (vs.location v))))

/--
Merge a split, beside a state, which may be read sorted: its outcome is
forgotten.

# Parameters
- `v`: The split.
- `p`: The state and the outcomes of the open splits.

# Returns
The state, and the outcomes with `v`'s blank.
-/
def merge {σ : Type} (v : Nat) (p : σ × Outcomes vs) : σ × Outcomes vs :=
  (p.1, Function.update p.2 v (blank (vs.location v)))

/--
Run an instruction beside the outcomes of the open splits, which it leaves
alone.

# Parameters
- `i`: The instruction.
- `γ`: The law of the state and the outcomes before it.

# Returns
The law of the state and the outcomes after it.
-/
noncomputable def exec (i : Instruction) (γ : PMF (State × Outcomes vs)) :
    PMF (State × Outcomes vs) :=
  γ.bind fun p => (step p.1 i).map (·, p.2)

/--
Perform one step of the plan on the outcomes beside the states. Mirrors
`Forward.perform`.

# Parameters
- `pc`: The program counter of the instruction, whose value any split splits.
- `γ`: The law of the states and the outcomes before the step.
- `splits`: The open splits, by value, from the bottom.
- `step`: The step.

# Returns
The law of the states and the outcomes after it, and the open splits after it.
-/
noncomputable def perform (pc : Nat) :
    PMF (State × Outcomes vs) × List Nat → Step →
      PMF (State × Outcomes vs) × List Nat
  | (γ, splits), .split _ => (γ.map (split pc), splits ++ [pc])
  | (γ, splits), .merge k _ _ =>
    (γ.map (merge (splits.getD k 0)), splits.eraseIdx k)

end Ghost

/--
The law of the states and of the outcomes of the open splits before an
instruction. Mirrors `forward`.

# Parameters
- `vs`: The values of the function.
- `s`: The initial state.
- `pc`: The program point: the number of instructions run.

# Returns
The law once the first `pc` instructions have run, each followed by the plan's
steps after it. Beyond the last instruction, the law is unchanged.
-/
noncomputable def ghost (vs : Values) (s : State) :
    Nat → PMF (State × Outcomes vs)
  | 0 => PMF.pure (s, blanks vs)
  | pc + 1 =>
    match vs.instructions[pc]? with
    | some i =>
      ((vs.steps pc).foldl (Ghost.perform pc)
        (Ghost.exec i (ghost vs s pc), vs.openSplits pc)).1
    | none => ghost vs s pc

/-! ## The invariant -/

/--
The ghost's invariant at a program point: once the dead locations are buried,
a law of states and outcomes, with the states read sorted, is the mixture,
over the worlds of the open splits, of each world's law of states, read
sorted, tagged with the world's outcomes.

# Parameters
- `vs`: The values of the function.
- `live`: The live values.
- `splits`: The open splits, by value, from the bottom.
- `γ`: The law of states and outcomes.
- `buried`: The locations buried.
- `base`: The state of every location whose value is not live.
- `K`: The law of each split's outcome.
- `laws`: The law of each live location in each world.
-/
def Traces (vs : Values) (live splits : List Nat)
    (γ : PMF (State × Outcomes vs)) (buried : List Location) (base : State)
    (K : SplitLaws vs) (laws : WorldLaws vs) : Prop :=
  γ.map (fun p => ((p.1.buryAll buried).toOracle, p.2)) =
    (splitOutcomes K splits).bind fun o =>
      (product base (live.map vs.location) (laws o)).map fun t =>
        (t.toOracle, o)

/--
The ghost's invariant holds of a law of states and outcomes in the worlds of
a forward pass: `Traces`, with the pass's witnesses.

# Parameters
- `F`: The forward pass.
- `live`: The live values.
- `splits`: The open splits, by value, from the bottom.
- `γ`: The law of states and outcomes.
-/
abbrev Forward.Traces (F : Forward vs) (live splits : List Nat)
    (γ : PMF (State × Outcomes vs)) : Prop :=
  Spec.Traces vs live splits γ F.buried F.base F.K F.laws

variable {pc : Nat} {live splits : List Nat} {μ : PMF State}
  {γ : PMF (State × Outcomes vs)} {buried : List Location} {base : State}
  {K : SplitLaws vs} {laws : WorldLaws vs}

/--
Where the invariant holds, the outcomes have the law of the pass's worlds.

# Hypotheses
- `h`: The invariant holds.
-/
theorem Traces.map_snd (h : Traces vs live splits γ buried base K laws) :
    γ.map Prod.snd = splitOutcomes K splits := by
  have := congrArg (PMF.map Prod.snd) h
  simp only [PMF.map_comp, Function.comp_def, PMF.map_bind] at this
  rw [this]
  conv_rhs => rw [← PMF.bind_pure (splitOutcomes K splits)]
  exact congrArg _ (funext fun o => PMF.map_const _ _)

/--
Burying locations leaves every other alone.

# Parameters
- `s`: The state.
- `ℓ`: The location read.
- `ℓs`: The locations buried.

# Hypotheses
- `h`: `ℓ` is not buried.
-/
private theorem get_buryAll {s : State} {ℓ : Location} {ℓs : List Location}
    (h : ℓ ∉ ℓs) : (s.buryAll ℓs).get ℓ = s.get ℓ := by
  induction ℓs generalizing s with
  | nil => rfl
  | cons ℓ' ℓs ih =>
    rw [State.buryAll_cons, ih fun h' => h (.tail _ h'), State.bury_eq_set,
      State.get_set_ne fun he => h (by simp [he])]

/--
Drawing splits leaves blank the outcome of every value not among them.

# Parameters
- `K`: The laws of the splits' outcomes.
- `splits`: The splits drawn.
- `W`: The law of the outcomes before them.
- `u`: A value not among them.

# Hypotheses
- `hu`: `u` is not among the splits.
- `hW`: `u`'s outcome is blank in every world before them.
-/
private theorem blank_of_foldl_drawSplit {K : SplitLaws vs} {u : Nat} :
    ∀ {splits : List Nat} {W : PMF (Outcomes vs)}, u ∉ splits →
      (∀ o ∈ W.support, o u = blank (vs.location u)) →
        ∀ o ∈ (splits.foldl (drawSplit K) W).support,
          o u = blank (vs.location u)
  | [], _, _, hW => hW
  | v :: splits, W, hu, hW => by
    refine blank_of_foldl_drawSplit (splits := splits) (W := drawSplit K W v)
      (fun h => hu (.tail _ h)) fun o ho => ?_
    simp only [drawSplit, PMF.mem_support_bind_iff, PMF.support_map,
      Set.mem_image] at ho
    obtain ⟨o', ho', x, _, rfl⟩ := ho
    rw [Function.update_of_ne fun he => hu (by simp [he])]
    exact hW o' ho'

/--
In every world of a stack of splits, a value not in the stack is blank.

# Parameters
- `K`: The laws of the splits' outcomes.
- `splits`: The stack.
- `o`: The world.
- `u`: The value.

# Hypotheses
- `ho`: `o` is a world of the stack.
- `hu`: `u` is not in the stack.
-/
theorem eq_blank_of_mem_support_splitOutcomes {K : SplitLaws vs}
    {splits : List Nat} {o : Outcomes vs}
    (ho : o ∈ (splitOutcomes K splits).support) {u : Nat} (hu : u ∉ splits) :
    o u = blank (vs.location u) :=
  blank_of_foldl_drawSplit hu (fun o ho => by simp_all) o ho

/--
The invariant holds of the initial state, with nothing live and no split open,
in the forward pass's one world, and with every outcome blank.

# Parameters
- `s`: The initial state.
-/
theorem traces_zero {s : State} :
    (Forward.initial vs s).Traces [] [] (PMF.pure (s, blanks vs)) := by
  unfold Forward.Traces Traces
  rw [PMF.pure_map]
  show PMF.pure (s.toOracle, blanks vs) =
    (PMF.pure (blanks vs)).bind fun o => (PMF.pure s).map fun t =>
      (t.toOracle, o)
  rw [PMF.pure_bind, PMF.pure_map]

/-! ## Instructions -/

/--
Two laws of states and outcomes that agree with the states read sorted agree
bound to any kernel that cannot tell apart states that read alike sorted.

# Parameters
- `G`: The kernel.

# Hypotheses
- `hG`: States that read alike sorted, beside the same outcomes, have the same
  law.
- `h`: The laws agree with the states read sorted.
-/
private theorem bind_congr_toOracle_fst {ω β : Type} {G : State × ω → PMF β}
    (hG : ∀ s t o, s.toOracle = t.toOracle → G (s, o) = G (t, o))
    {γ γ' : PMF (State × ω)}
    (h : γ.map (Prod.map State.toOracle id) =
      γ'.map (Prod.map State.toOracle id)) :
    γ.bind G = γ'.bind G := by
  have hc : (G ∘ Prod.map State.ofOracle id) ∘ Prod.map State.toOracle id =
      G :=
    funext fun p => hG _ _ p.2 (toOracle_ofOracle_toOracle p.1)
  calc γ.bind G = (γ.map (Prod.map State.toOracle id)).bind
        (G ∘ Prod.map State.ofOracle id) := by rw [PMF.bind_map, hc]
    _ = γ'.bind G := by rw [h, PMF.bind_map, hc]

/--
An instruction keeps the invariant: it runs on the state and leaves the
outcomes alone, so in each world it takes the world's law of states, tagged
with its outcomes, to the pass's law of states after it, tagged alike
(`invariant_step_world`).

# Parameters
- `pc`: The program counter of the instruction.
- `F`: The forward pass before the instruction.
- `γ`: The law of states and outcomes before it.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hpc`: `pc` is an instruction of the function.
- `h`: The plan's invariant holds before the instruction, in the worlds of
  `F`.
- `ht`: The ghost's invariant holds before it, in the worlds of `F`.
-/
theorem traces_step
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hpc : pc < vs.instructions.length) {F : Forward vs}
    (h : F.Holds pc (vs.liveBefore pc) (vs.openSplits pc) μ)
    (ht : F.Traces (vs.liveBefore pc) (vs.openSplits pc) γ) :
    (F.exec pc vs.instructions[pc]).Traces (vs.live pc) (vs.openSplits pc)
      (Ghost.exec vs.instructions[pc] γ) := by
  obtain ⟨-, hbury, hworld⟩ := invariant_step_world hvals hpc h
  unfold Forward.Traces Traces at ht ⊢
  -- Bury what was buried before the instruction, then run it in each world.
  let G : State × Outcomes vs → PMF (Xdy.State × Outcomes vs) := fun q =>
    (((step q.1 vs.instructions[pc]).map
      (·.buryAll (F.exec pc vs.instructions[pc]).buried)).map
        State.toOracle).map (·, q.2)
  have hG : (Ghost.exec vs.instructions[pc] γ).map
      (fun p => ((p.1.buryAll (F.exec pc vs.instructions[pc]).buried).toOracle,
        p.2)) = (γ.map fun p => (p.1.buryAll F.buried, p.2)).bind G := by
    rw [Ghost.exec, PMF.map_bind, PMF.bind_map]
    refine congrArg γ.bind (funext fun p => ?_)
    show _ = G (p.1.buryAll F.buried, p.2)
    simp only [G]
    rw [hbury]
    simp only [PMF.map_comp]
    rfl
  -- The kernel reads its state only sorted.
  have hGs : ∀ s t o, s.toOracle = t.toOracle → G (s, o) = G (t, o) := by
    intro s t o hst
    have := map_buryAll_bind_step_toOracle (μ := PMF.pure s) (ν := PMF.pure t)
      (by rw [PMF.pure_map, PMF.pure_map, hst]) vs.instructions[pc]
      (F.exec pc vs.instructions[pc]).buried
    simp only [PMF.pure_bind] at this
    exact congrArg (PMF.map _) this
  rw [hG, bind_congr_toOracle_fst hGs (γ' := (splitOutcomes F.K
    (vs.openSplits pc)).bind fun o => (product F.base
      ((vs.liveBefore pc).map vs.location) (F.laws o)).map (·, o)) (by
    rw [PMF.map_comp, PMF.map_bind]
    simp only [PMF.map_comp]
    exact ht), PMF.bind_bind]
  refine congrArg _ (funext fun o => ?_)
  rw [PMF.bind_map, ← hworld o]
  simp only [G, PMF.map_comp, PMF.map_bind]
  rfl

/-! ## Splits and merges -/

/--
A split keeps the invariant: in each world, the split's value is live, so the
state holds it, and the ghost copies it, sorted, as the world's outcome, which
the split draws from the value's law in that world, read sorted
(`bind_pure_product`, `map_toOracle_product_pure`).

# Parameters
- `v`: The value.

# Hypotheses
- `h`: The plan's invariant holds before the split.
- `hl`: `v` is live.
- `hv`: `v` is not already an open split.
- `hlen`: `v` is an instruction of the function.
- `hw`: `v` occupies its location.
- `hnd`: The live values occupy distinct locations.
- `ht`: The ghost's invariant holds before the split.
-/
theorem Worlds.traces_split (h : Worlds vs pc live splits μ buried base K laws)
    {v : Nat} (hl : v ∈ live) (hv : v ∉ splits)
    (hlen : v < vs.instructions.length)
    (hw : vs.writer pc (vs.location v) = some v)
    (hnd : (live.map vs.location).Nodup)
    (ht : Traces vs live splits γ buried base K laws) :
    Traces vs live (splits ++ [v]) (γ.map (Ghost.split v)) buried base
      (Function.update K v fun o =>
        (laws o (vs.location v)).map (sortContent (vs.location v)))
      (fun o => Function.update (laws o) (vs.location v) (PMF.pure (o v))) := by
  -- The split's location is not buried, and the base has it.
  have hb : vs.location v ∉ buried := fun hm => by
    obtain ⟨w, hw', hwl⟩ := h.buried_dead _ hm
    rw [hw] at hw'
    cases hw'
    exact hwl hl
  have hh : base.Has (vs.location v) := by
    rw [vs.location_eq hlen]
    exact h.has _ (List.getElem_mem hlen)
  have hmem : vs.location v ∈ live.map vs.location := List.mem_map_of_mem hl
  -- The law of a live location other than `v`'s ignores `v`'s outcome.
  have hlaws : ∀ ℓ ∈ live.map vs.location, ℓ ≠ vs.location v → ∀ o x,
      laws (Function.update o v x) ℓ = laws o ℓ := by
    intro ℓ hℓ _ o x
    obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
    exact h.laws_dependsOn w hw _ _ fun u _ hu =>
      Function.update_of_ne (fun he : u = v => hv (he ▸ hu)) _ _
  unfold Traces at ht ⊢
  -- The ghost copies the outcome, sorted, from the state as buried.
  have hcopy : (γ.map (Ghost.split v)).map
      (fun p => ((p.1.buryAll buried).toOracle, p.2)) =
        (γ.map fun p => ((p.1.buryAll buried).toOracle, p.2)).map fun q =>
          (q.1, Function.update q.2 v (sortContent (vs.location v)
            ((State.ofOracle q.1).get (vs.location v)))) := by
    rw [PMF.map_comp, PMF.map_comp]
    refine congrArg (PMF.map · γ) (funext fun p => ?_)
    simp only [Function.comp_def, Ghost.split]
    rw [sortContent_get_congr
      (toOracle_ofOracle_toOracle (p.1.buryAll buried)), get_buryAll hb]
  rw [hcopy, ht, PMF.map_bind, splitOutcomes_append_singleton,
    splitOutcomes_congr (K := Function.update K v fun o =>
      (laws o (vs.location v)).map (sortContent (vs.location v))) (K' := K)
      fun u hu => Function.update_of_ne
        (fun he : u = v => hv (he ▸ hu)) _ _, drawSplit, PMF.bind_bind]
  refine congrArg _ (funext fun o => ?_)
  -- In each world, draw the split's outcome, then the rest.
  rw [PMF.bind_map, Function.update_self, PMF.bind_map, PMF.map_comp]
  conv_lhs => rw [← bind_pure_product hnd hmem]
  rw [PMF.map_bind]
  refine congrArg _ (funext fun x => ?_)
  simp only [Function.comp_apply, Function.update_self]
  rw [product_congr (laws := Function.update
    (laws (Function.update o v (sortContent _ x))) (vs.location v)
      (PMF.pure (sortContent _ x))) (laws' := Function.update (laws o)
        (vs.location v) (PMF.pure (sortContent _ x))) fun ℓ hℓ => by
          by_cases he : ℓ = vs.location v
          · rw [he, Function.update_self, Function.update_self]
          · rw [Function.update_of_ne he, Function.update_of_ne he,
              hlaws ℓ hℓ he]]
  -- Read sorted, the world of the outcome sorted is the world of the outcome.
  have hpure := congrArg (PMF.map fun y : Xdy.State =>
      (y, Function.update o v (sortContent (vs.location v) x)))
    (map_toOracle_product_pure (base := base) (laws := laws o) x hnd hmem)
  simp only [PMF.map_comp, Function.comp_def] at hpure
  rw [hpure]
  -- The state holds the outcome in each world.
  refine map_congr_support fun t htx => ?_
  have hget := map_get_product (laws := Function.update (laws o)
    (vs.location v) (PMF.pure x)) hnd hmem hh
  rw [Function.update_self] at hget
  have : t.get (vs.location v) ∈ (PMF.pure x).support :=
    hget ▸ (PMF.mem_support_map_iff _ _ _).mpr ⟨t, htx, rfl⟩
  rw [PMF.support_pure, Set.mem_singleton_iff] at this
  simp only [Function.comp_apply]
  rw [sortContent_get_congr (toOracle_ofOracle_toOracle t), this]

/--
A merge keeps the invariant: the ghost forgets the split's outcome, which the
merge draws last and then mixes into the survivor's law in each world of the
splits that remain (`Worlds.merge_world`), in which it was blank.

# Parameters
- `k`: The position of the split in the stack.
- `sv`: The location of the survivor, if any.

# Hypotheses
- `h`: The plan's invariant holds before the merge.
- `hs`: The splits ascend.
- `hk`: `k` is in range.
- `hsv`: The split may merge, into `sv`.
- `hnd`: The live values occupy distinct locations.
- `ht`: The ghost's invariant holds before the merge.
-/
theorem Worlds.traces_merge (h : Worlds vs pc live splits μ buried base K laws)
    (hs : splits.Pairwise (· < ·)) {k : Nat} (hk : k < splits.length)
    {sv : Option Location} (hsv : vs.survivor splits k live = some sv)
    (hnd : (live.map vs.location).Nodup)
    (ht : Traces vs live splits γ buried base K laws) :
    Traces vs live (splits.eraseIdx k) (γ.map (Ghost.merge splits[k])) buried
      base K (mergeLaws K laws splits[k] sv) := by
  obtain ⟨hW, hworld⟩ := h.merge_world hs hk hsv hnd
  have hout : splits[k] ∉ splits.eraseIdx k := fun hm => by
    obtain ⟨i, hi, hik, he⟩ := List.mem_eraseIdx_iff_getElem.mp hm
    exact hik ((List.Nodup.getElem_inj_iff (hs.imp Nat.ne_of_lt)).mp he)
  unfold Traces at ht ⊢
  -- The ghost forgets the outcome whatever the state.
  have hforget : (γ.map (Ghost.merge splits[k])).map
      (fun p => ((p.1.buryAll buried).toOracle, p.2)) =
        (γ.map fun p => ((p.1.buryAll buried).toOracle, p.2)).map
          (Ghost.merge splits[k]) := by
    rw [PMF.map_comp, PMF.map_comp]
    rfl
  rw [hforget, ht, hW, drawSplit, PMF.bind_bind, PMF.map_bind]
  refine bind_congr_support fun o ho => ?_
  -- In each world of the splits that remain, mix over the outcome.
  rw [PMF.bind_map, PMF.map_bind, ← hworld o, PMF.map_bind]
  refine congrArg _ (funext fun x => ?_)
  simp only [Function.comp_def, PMF.map_comp]
  refine congrArg (PMF.map · _) (funext fun t => ?_)
  simp only [Ghost.merge, Function.update_idem,
    ← eq_blank_of_mem_support_splitOutcomes ho hout, Function.update_eq_self]

/-! ## Every step of the plan -/

/--
Every merge that the plan schedules after an instruction keeps both
invariants, in the worlds of the forward pass, which performs them in turn,
beside the ghost, which performs them alike. So does every prefix of them,
since the pass counts its worlds after each.

# Parameters
- `pc'`: The program counter of the instruction.
- `current`: The values that occupy their locations.
- `splits`: The stack before the merges.
- `F`: The forward pass before the merges.
- `γ`: The law of states and outcomes before them.
- `n`: The number of merges performed.

# Hypotheses
- `hnd`: The live values occupy distinct locations.
- `hs`: The splits ascend.
- `h`: The plan's invariant holds before the merges, in the worlds of `F`.
- `ht`: The ghost's invariant holds before them, in the worlds of `F`.

# Returns
That the ghost's stack is the pass's; that it is the stack that the plan
leaves, once every merge is performed; and that both invariants hold after the
first `n` merges.
-/
theorem traces_merges {current : List Nat} {pc' : Nat}
    (hnd : (live.map vs.location).Nodup) (splits : List Nat)
    (hs : splits.Pairwise (· < ·)) (F : Forward vs)
    (γ : PMF (State × Outcomes vs)) (h : F.Holds pc live splits μ)
    (ht : F.Traces live splits γ) (n : Nat) :
    let ms := (vs.merges live current splits).1.take n
    let r := ms.foldl (Forward.perform pc') (F, splits)
    let g := ms.foldl (Ghost.perform pc') (γ, splits)
    g.2 = r.2 ∧
      ((vs.merges live current splits).1.length ≤ n →
        r.2 = (vs.merges live current splits).2) ∧
      r.1.Holds pc live r.2 μ ∧ r.1.Traces live r.2 g.1 := by
  rcases vs.merges_cases live current splits with he | ⟨k, hk, sv, hsv, he⟩
  · simp only [he, List.take_nil, List.foldl_nil]
    exact ⟨by simp, by simp, h, ht⟩
  · cases n with
    | zero =>
      simp only [he, List.take_zero, List.foldl_nil, List.length_cons]
      exact ⟨by simp, by simp, h, ht⟩
    | succ n =>
      simp only [he, List.take_succ_cons, List.foldl_cons, Forward.perform,
        Ghost.perform, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk,
        Option.getD_some, List.length_cons]
      obtain ⟨h1, h2, h3, h4⟩ := traces_merges (current := current)
        (pc' := pc') hnd (splits.eraseIdx k)
        (hs.sublist (List.eraseIdx_sublist _ _)) (F.merge splits[k] sv)
        (γ.map (Ghost.merge splits[k])) (h.merge hs hk hsv hnd)
        (h.traces_merge hs hk hsv hnd ht) n
      exact ⟨h1, fun hle => h2 (by omega), h3, h4⟩
termination_by splits.length
decreasing_by rw [List.length_eraseIdx_of_lt hk]; omega

/--
The steps that follow an instruction keep both invariants: running it,
splitting its value if it fans out, and merging every split that may, beside
the ghost, which runs it on the state and performs the steps alike. So does
every prefix of them.

# Parameters
- `pc`: The program counter of the instruction.
- `F`: The forward pass before the instruction.
- `γ`: The law of states and outcomes before it.
- `n`: The number of steps performed.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hpc`: `pc` is an instruction of the function.
- `h`: The plan's invariant holds before the instruction, in the worlds of
  `F`.
- `ht`: The ghost's invariant holds before it, in the worlds of `F`.

# Returns
That the ghost's stack is the pass's; that it is the stack open before the
next instruction, once every step is performed; and that both invariants hold
after the first `n` steps.
-/
theorem traces_steps
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hpc : pc < vs.instructions.length) {F : Forward vs}
    (h : F.Holds pc (vs.liveBefore pc) (vs.openSplits pc) μ)
    (ht : F.Traces (vs.liveBefore pc) (vs.openSplits pc) γ) (n : Nat) :
    let ss := (vs.steps pc).take n
    let r := ss.foldl (Forward.perform pc)
      (F.exec pc vs.instructions[pc], vs.openSplits pc)
    let g := ss.foldl (Ghost.perform pc)
      (Ghost.exec vs.instructions[pc] γ, vs.openSplits pc)
    g.2 = r.2 ∧ ((vs.steps pc).length ≤ n → r.2 = vs.openSplits (pc + 1)) ∧
      r.1.Holds (pc + 1) (vs.live pc) r.2
        (μ.bind (step · vs.instructions[pc])) ∧
      r.1.Traces (vs.live pc) r.2 g.1 := by
  have hstep := invariant_step hvals hpc h
  have htstep := traces_step hvals hpc h ht
  have hnd : ((vs.live pc).map vs.location).Nodup :=
    vs.nodup_map_location_liveBefore (pc := pc + 1) hpc
  simp only [vs.steps_eq, vs.openSplits_succ]
  cases hf : vs.fansOut pc
  · simp only [Bool.false_eq_true, ↓reduceIte, List.nil_append]
    exact traces_merges hnd _ (vs.pairwise_openSplits pc)
      (F.exec pc vs.instructions[pc]) (Ghost.exec vs.instructions[pc] γ)
      hstep htstep n
  · simp only [↓reduceIte, List.singleton_append]
    cases n with
    | zero =>
      simp only [List.take_zero, List.foldl_nil, List.length_cons]
      exact ⟨by simp, by simp, hstep, htstep⟩
    | succ n =>
      simp only [List.take_succ_cons, List.foldl_cons, Forward.perform,
        Ghost.perform, List.length_cons]
      have hl := vs.mem_live_self hf
      have hv : pc ∉ vs.openSplits pc := fun h =>
        Nat.lt_irrefl _ (vs.mem_openSplits h).2
      obtain ⟨h1, h2, h3, h4⟩ := traces_merges (current := vs.current pc)
        (pc' := pc) hnd _ (List.pairwise_append.mpr
          ⟨vs.pairwise_openSplits pc, by simp, fun a ha b hb => by
            rw [List.mem_singleton.mp hb]
            exact (vs.mem_openSplits ha).2⟩)
        ((F.exec pc vs.instructions[pc]).split pc)
        ((Ghost.exec vs.instructions[pc] γ).map (Ghost.split pc))
        (hstep.split hf hl hv hnd)
        (hstep.traces_split hl hv hpc
          (vs.writer_of_mem_liveBefore (pc := pc + 1) hpc hl) hnd htstep) n
      exact ⟨h1, fun hle => h2 (by omega), h3, h4⟩

/--
The ghost's invariant holds throughout, in the worlds of the forward pass:
before each instruction, the law of states and outcomes, once the dead
locations are buried, is the mixture of the pass's worlds, each tagged with
its outcomes.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hhas`: The initial state has the destination of every instruction.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem traces_forward
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true) {s : State}
    (hhas : ∀ i ∈ vs.instructions, s.Has i.destination)
    (hpc : pc ≤ vs.instructions.length) :
    (forward vs s pc).Traces (vs.liveBefore pc) (vs.openSplits pc)
      (ghost vs s pc) := by
  induction pc with
  | zero => exact traces_zero
  | succ pc ih =>
    have hlt : pc < vs.instructions.length := by omega
    obtain ⟨h1, h2, -, h4⟩ := traces_steps hvals hlt
      (invariant_exec hvals hhas (by omega)) (ih (by omega))
      (vs.steps pc).length
    simp only [List.take_length] at h1 h2 h4
    simp only [forward, ghost, List.getElem?_eq_getElem hlt]
    rw [vs.liveBefore_succ, ← h2 (Nat.le_refl _)]
    exact h4

/-! ## The worlds' law -/

/--
The plan's steps change only the outcomes beside the states.

# Parameters
- `pc`: The program counter of the instruction.
- `steps`: The steps.
- `γ`: The law of states and outcomes before them.
- `splits`: The open splits before them.
-/
theorem map_fst_foldl_perform (pc : Nat) (steps : List Step)
    (γ : PMF (State × Outcomes vs)) (splits : List Nat) :
    (steps.foldl (Ghost.perform pc) (γ, splits)).1.map Prod.fst =
      γ.map Prod.fst := by
  induction steps generalizing γ splits with
  | nil => rfl
  | cons st steps ih =>
    cases st <;> simp only [List.foldl_cons, Ghost.perform] <;> rw [ih] <;>
      rw [PMF.map_comp] <;> rfl

/--
The ghost's states are the function's: before each instruction, forgetting
the outcomes, the ghost's law is the law of states that running the
instructions so far answers.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hpc`: `pc` is at most the number of instructions.
-/
theorem ghost_fst {s : State} (hpc : pc ≤ vs.instructions.length) :
    (ghost vs s pc).map Prod.fst =
      exec (vs.instructions.take pc) (PMF.pure s) := by
  induction pc with
  | zero => simp [ghost, PMF.pure_map]
  | succ pc ih =>
    have hlt : pc < vs.instructions.length := by omega
    rw [List.take_add_one, List.getElem?_eq_getElem hlt, Option.toList_some,
      exec_append, exec_cons, exec_nil, ← ih (by omega)]
    simp only [ghost, List.getElem?_eq_getElem hlt]
    rw [map_fst_foldl_perform, Ghost.exec, PMF.map_bind, PMF.bind_map]
    refine congrArg _ (funext fun p => ?_)
    rw [PMF.map_comp]
    exact PMF.map_id _

/--
The forward pass draws its worlds with their true joint law: before each
instruction, the law with which running the function reaches each assignment
of outcomes to the open splits (`ghost_fst`) is the law of the pass's worlds.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hhas`: The initial state has the destination of every instruction.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem ghost_snd
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true) {s : State}
    (hhas : ∀ i ∈ vs.instructions, s.Has i.destination)
    (hpc : pc ≤ vs.instructions.length) :
    (ghost vs s pc).map Prod.snd =
      splitOutcomes (forward vs s pc).K (vs.openSplits pc) :=
  (traces_forward hvals hhas hpc).map_snd

end Xdy.Spec
