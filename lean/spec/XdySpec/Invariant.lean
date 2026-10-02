import XdySpec.Forward
import XdySpec.Outcomes

/-!
# The plan's invariant

The forward pass in `xdy/src/distribution/propagation.rs` holds, in each world
of its open splits, every live value as its own distribution, and treats them
as independent. This module proves that the specification's joint law of
states agrees with that picture at every step of the plan in
`Xdy/Plan.lean`: before each instruction, and after each split and merge.

The invariant, `Worlds`, holds of a law of states `μ` at a program point when,
once every location whose value is dead is buried, `μ`, read sorted
(`State.toOracle`), is a mixture of worlds (`product`), read sorted, one for
each assignment of outcomes to the open splits, drawn split by split from the
bottom of the stack (`splitOutcomes`), in which:

- each live value's law depends only on the outcomes of the open splits on
  which the value depends;
- each split's outcome depends only on the outcomes of the open splits on
  which it depends, which lie beneath it;
- each split's every outcome is sorted, so that a rolling record's outcomes
  are the multisets of its results, as in the worlds of `propagation.rs`
  (`Worlds.sortContent_outcome`);
- each live open split is a point mass at its outcome, and each live value
  that no roll reaches is a point mass.

The law holds read sorted, not exactly, since a split fixes a rolling record
at its results sorted, while the states hold them in every order that sorts
to them. No instruction can tell the order (`Outcomes.map_toOracle_step`), so
laws that agree read sorted agree after any instruction
(`map_buryAll_bind_step_toOracle`), and so do a world in which a record is a
point mass and the world in which it is a point mass at its results sorted
(`map_toOracle_product_pure`): `exec_split_sorted`, in
`XdySpec/Outcomes.lean`, carried into the pass.

Its witnesses are the forward pass of `XdySpec/Forward.lean`: the locations
that it has buried, its base, and its laws of the splits and of the live
locations (`Forward.Holds`). Each step of the plan keeps it
(`invariant_step`, `Worlds.split`, `Worlds.merge`), with the witnesses that
the pass's step gives, so it holds throughout, of the pass before each
instruction (`invariant_exec`, `invariant_run`):

- An instruction reads live values of three kinds: those that stay live, which
  are point masses, since each is an open split or reached by no roll
  (`mem_openSplits_of_random`); those that it reads for the last time, which
  it buries; and it leaves the rest alone, since it neither reads nor
  overwrites a live value that it does not read. So `bind_step_product_pure`
  gives its value a law of its own, which depends only on the laws of the
  values that it reads.
- A split rewrites its value's world as the mixture of point masses at each of
  its values (`bind_pure_product`), each, read sorted, the point mass at its
  value sorted (`map_toOracle_product_pure`), and draws that outcome atop the
  stack.
- A merge draws its split's outcome last, since no split above depends on it,
  and then mixes the survivor's law over it (`bind_product`), since no other
  live value depends on it.

Reading a register cannot tell the order of any record's results, so a
function answers the mixture, over the worlds of the splits open before its
return, of the forward pass's law of the returned register in each
(`run_eq_forward`). No merge goes through lemma 6's `exec_merge`, whose
hypotheses are laws of states given a split: a merge changes only how the law
of states is written as a mixture, never the law itself.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Before: initial state
    Before --> Executed: instruction
    Executed --> Split: value fans out
    Executed --> Merged: merges
    Split --> Merged: merges
    Merged --> Before: next instruction
```
-/

namespace Xdy.Spec

open private Xdy.Function.checkInstruction Xdy.Function.checkValue
  Xdy.Function.checkRegister Xdy.Function.checkRecord from Xdy.Ir

/-! ## Well-formed functions -/

/--
An action run for each element of a list succeeds only if each succeeds.

# Parameters
- `g`: The action.

# Hypotheses
- `h`: Running it for each element succeeds.
-/
private theorem forM_ok {α : Type} {g : α → Except String Unit} :
    ∀ {l : List α}, forM l g = .ok () → ∀ a ∈ l, g a = .ok ()
  | [], _ => by simp
  | a :: l, h => by
    rw [List.forM_cons] at h
    cases ha : g a with
    | error e => rw [ha] at h; cases h
    | ok u =>
      rw [ha] at h
      intro b hb
      rcases List.mem_cons.mp hb with rfl | hb
      · exact ha
      · exact forM_ok h b hb

/--
An action run for each element of an array succeeds only if each succeeds.

# Parameters
- `xs`: The array.

# Hypotheses
- `h`: Running the action for each element succeeds.
-/
private theorem forM_array_ok {α : Type} {g : α → Except String Unit}
    (xs : Array α) (h : forM xs g = .ok ()) : ∀ a ∈ xs, g a = .ok () := by
  rcases xs with ⟨xs⟩
  simp only [List.forM_toArray] at h
  intro a ha
  exact forM_ok h a (by simpa using ha)

/--
Two checks in sequence succeed only if both do.

# Hypotheses
- `h`: The sequence succeeds.
-/
private theorem bind_ok {a : Except String Unit} {b : Unit → Except String Unit}
    (h : (a >>= b) = .ok ()) : a = .ok () ∧ b () = .ok () := by
  cases a with
  | error => cases h
  | ok u => exact ⟨rfl, h⟩

/--
An operand that passes validation is a value.

# Hypotheses
- `h`: The operand passes validation.
-/
private theorem checkValue_ok {f : Function} {op : AddressingMode}
    (h : Xdy.Function.checkValue f op = .ok ()) : op.isValue = true := by
  cases op
  · rfl
  · rfl
  · cases h

/--
A register that passes validation exists.

# Hypotheses
- `h`: The register passes validation.
-/
private theorem checkRegister_ok {f : Function} {r : Nat}
    (h : Xdy.Function.checkRegister f r = .ok ()) : r < f.registerCount := by
  simp only [Xdy.Function.checkRegister] at h
  by_contra hr
  simp only [hr, ↓reduceIte] at h
  cases h

/--
A rolling record that passes validation exists.

# Hypotheses
- `h`: The rolling record passes validation.
-/
private theorem checkRecord_ok {f : Function} {r : Nat}
    (h : Xdy.Function.checkRecord f r = .ok ()) :
    r < f.rollingRecordCount := by
  simp only [Xdy.Function.checkRecord] at h
  by_contra hr
  simp only [hr, ↓reduceIte] at h
  cases h

/--
An instruction that passes validation has operands that are values, and a
destination that every state of the function's shape has.

# Parameters
- `s`: A state.

# Hypotheses
- `h`: The instruction passes validation.
- `hr`: `s` has as many registers as the function.
- `hd`: `s` has as many rolling records as the function.
-/
private theorem checkInstruction_ok {f : Function} {i : Instruction}
    (h : Xdy.Function.checkInstruction f i = .ok ()) {s : State}
    (hr : s.registers.size = f.registerCount)
    (hd : s.records.size = f.rollingRecordCount) :
    i.operandsAreValues = true ∧ s.Has i.destination := by
  cases i <;> simp only [Xdy.Function.checkInstruction] at h
  case sumRollingRecord =>
    exact ⟨rfl, by simpa [State.Has, Instruction.destination, hr] using
      checkRegister_ok (bind_ok h).1⟩
  case «return» =>
    exact ⟨by simp [Instruction.operandsAreValues, checkValue_ok h], trivial⟩
  case binary =>
    obtain ⟨hd', h⟩ := bind_ok h
    obtain ⟨h₁, h₂⟩ := bind_ok h
    exact ⟨by simp [Instruction.operandsAreValues, checkValue_ok h₁,
        checkValue_ok h₂],
      by simpa [State.Has, Instruction.destination, hr] using
        checkRegister_ok hd'⟩
  case neg =>
    obtain ⟨hd', h⟩ := bind_ok h
    exact ⟨by simp [Instruction.operandsAreValues, checkValue_ok h],
      by simpa [State.Has, Instruction.destination, hr] using
        checkRegister_ok hd'⟩
  case rollRange | rollStandardDice =>
    obtain ⟨hd', h⟩ := bind_ok h
    obtain ⟨h₁, h₂⟩ := bind_ok h
    exact ⟨by simp [Instruction.operandsAreValues, checkValue_ok h₁,
        checkValue_ok h₂],
      by simpa [State.Has, Instruction.destination, hd] using
        checkRecord_ok hd'⟩
  all_goals
    obtain ⟨hd', h⟩ := bind_ok h
    exact ⟨by simp [Instruction.operandsAreValues, checkValue_ok h],
      by simpa [State.Has, Instruction.destination, hd] using
        checkRecord_ok hd'⟩

/--
A well-formed function passes validation instruction by instruction.

# Parameters
- `i`: An instruction.

# Hypotheses
- `h`: The function is well formed.
- `hi`: `i` is one of its instructions.
-/
private theorem checkInstruction_of_validate {f : Function}
    (h : f.validate = .ok ()) {i : Instruction} (hi : i ∈ f.instructions) :
    Xdy.Function.checkInstruction f i = .ok () := by
  unfold Xdy.Function.validate at h
  simp only [bind, Except.bind] at h
  by_cases hc : f.parameters.size + f.externals.size ≤ f.registerCount
  · simp only [hc, ↓reduceIte, Array.forM_eq_forM] at h
    cases hf : forM f.instructions (Xdy.Function.checkInstruction f) with
    | error e => rw [hf] at h; cases h
    | ok u => exact forM_array_ok _ hf i hi
  · simp only [hc, ↓reduceIte] at h
    cases h

/--
A fold whose every step keeps a property keeps it.

# Parameters
- `P`: The property.
- `g`: The step.

# Hypotheses
- `hg`: Every step that succeeds keeps `P`.
-/
private theorem foldlM_ok {α β : Type} {P : β → Prop}
    {g : β → α → Except String β}
    (hg : ∀ b a b', g b a = .ok b' → P b → P b') :
    ∀ {l : List α} {b b' : β}, l.foldlM g b = .ok b' → P b → P b'
  | [], b, b', h, hb => by cases h; exact hb
  | a :: l, b, b', h, hb => by
    rw [List.foldlM_cons] at h
    cases ha : g b a with
    | error => rw [ha] at h; cases h
    | ok c => rw [ha] at h; exact foldlM_ok hg h (hg b a c ha hb)

/--
A function's initial state has as many registers and rolling records as the
function.

# Parameters
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.

# Hypotheses
- `h`: The initial state is `s`.
-/
private theorem initialState_size {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (h : f.initialState args externals = .ok s) :
    s.registers.size = f.registerCount ∧
      s.records.size = f.rollingRecordCount := by
  unfold Xdy.Function.initialState at h
  simp only [bind, Except.bind] at h
  by_cases hc : (args.length == f.parameters.size) = true
  · simp only [hc, ↓reduceIte] at h
    let P : Xdy.State → Prop := fun t =>
      t.registers.size = f.registerCount ∧
        t.records.size = f.rollingRecordCount
    have hset : ∀ (t : Xdy.State) r x, P t → P (t.setRegister r x) :=
      fun t r x ht => by simpa [P, Xdy.State.setRegister] using ht
    split at h
    · cases h
    · rename_i t h₁
      have ht : P t := foldlM_ok (fun b a b' hb hP => by
        split at hb
        · cases hb
        · cases hb; exact hset _ _ _ hP) h₁ (by simp [P])
      refine foldlM_ok (P := P) (fun b a b' hb hP => ?_) h ht
      split at hb
      · cases hb
      · split at hb
        · cases hb; exact hset _ _ _ hP
        · cases hb
  · simp only [hc] at h
    cases h

/--
Every operand of a well-formed function is a value.

# Parameters
- `f`: The function.

# Hypotheses
- `h`: The function is well formed.
-/
theorem operandsAreValues_of_validate {f : Function}
    (h : f.validate = .ok ()) :
    ∀ i ∈ f.instructions.toList, i.operandsAreValues = true := fun _ hi =>
  (checkInstruction_ok (s := ⟨.replicate f.registerCount 0,
      .replicate f.rollingRecordCount {}⟩)
    (checkInstruction_of_validate h (Array.mem_toList_iff.mp hi))
    Array.size_replicate Array.size_replicate).1

/--
The initial state of a well-formed function has the destination of each of
its instructions.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.

# Hypotheses
- `h`: The function is well formed.
- `hs`: Its initial state is `s`.
-/
theorem has_initialState {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (h : f.validate = .ok ()) (hs : f.initialState args externals = .ok s) :
    ∀ i ∈ f.instructions.toList, (State.ofOracle s).Has i.destination :=
  fun _ hi =>
    (checkInstruction_ok
      (checkInstruction_of_validate h (Array.mem_toList_iff.mp hi))
      (initialState_size hs).1
      (by simpa [State.ofOracle] using (initialState_size hs).2)).2

/-! ## Reading states sorted -/

/--
Reading a state sorted, as a state of the specification, and sorting it again
reads the same.

# Parameters
- `s`: The state.
-/
theorem toOracle_ofOracle_toOracle (s : State) :
    (State.ofOracle s.toOracle).toOracle = s.toOracle :=
  State.toOracle_ofOracle _ (Outcomes.toOracle_sorted s)

/--
A function of states that cannot tell two states apart when they read alike
sorted is a function of the state read sorted.

# Parameters
- `g`: The function.

# Hypotheses
- `hg`: States that read alike sorted have the same image.
-/
theorem comp_ofOracle_toOracle {β : Type} {g : State → β}
    (hg : ∀ s t : State, s.toOracle = t.toOracle → g s = g t) :
    (g ∘ State.ofOracle) ∘ State.toOracle = g :=
  funext fun s => hg _ _ (toOracle_ofOracle_toOracle s)

/--
Two laws of states that agree read sorted agree through any function that
cannot tell apart states that read alike sorted.

# Parameters
- `g`: The function.

# Hypotheses
- `hg`: States that read alike sorted have the same image.
- `h`: The laws agree read sorted.
-/
theorem map_congr_toOracle {β : Type} {g : State → β}
    (hg : ∀ s t : State, s.toOracle = t.toOracle → g s = g t)
    {μ ν : PMF State} (h : μ.map State.toOracle = ν.map State.toOracle) :
    μ.map g = ν.map g := by
  have hc := comp_ofOracle_toOracle hg
  calc μ.map g = (μ.map State.toOracle).map (g ∘ State.ofOracle) := by
        rw [PMF.map_comp, hc]
    _ = ν.map g := by rw [h, PMF.map_comp, hc]

/--
Two laws of states that agree read sorted agree bound to any kernel that
cannot tell apart states that read alike sorted.

# Parameters
- `g`: The kernel.

# Hypotheses
- `hg`: States that read alike sorted have the same law.
- `h`: The laws agree read sorted.
-/
theorem bind_congr_toOracle {β : Type} {g : State → PMF β}
    (hg : ∀ s t : State, s.toOracle = t.toOracle → g s = g t)
    {μ ν : PMF State} (h : μ.map State.toOracle = ν.map State.toOracle) :
    μ.bind g = ν.bind g := by
  have hc := comp_ofOracle_toOracle hg
  calc μ.bind g = (μ.map State.toOracle).bind (g ∘ State.ofOracle) := by
        rw [PMF.bind_map, hc]
    _ = ν.bind g := by rw [h, PMF.bind_map, hc]

/--
Writing a location keeps two states that read alike sorted reading alike, if
the values written read alike sorted: no order of a rolling record's results
shows once it is read sorted.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `h`: The states read alike sorted.
- `hxy`: The values read alike sorted.
-/
theorem set_toOracle_congr {s t : State} (h : s.toOracle = t.toOracle)
    (ℓ : Location) {x y : Content ℓ}
    (hxy : sortContent ℓ x = sortContent ℓ y) :
    (s.set ℓ x).toOracle = (t.set ℓ y).toOracle := by
  cases ℓ with
  | register r =>
    change x = y at hxy
    subst hxy
    show (s.setRegister r x).toOracle = (t.setRegister r x).toOracle
    rw [State.setRegister_toOracle, State.setRegister_toOracle, h]
  | record r =>
    show (s.setRecord r x).toOracle = (t.setRecord r y).toOracle
    rw [State.setRecord_toOracle, State.setRecord_toOracle, h,
      Record.sort_eq_sort_iff.mp hxy]
  | answer => exact h

/--
Burying locations keeps two states that read alike sorted reading alike.

# Parameters
- `ℓs`: The locations.

# Hypotheses
- `h`: The states read alike sorted.
-/
theorem buryAll_toOracle_congr {s t : State} (h : s.toOracle = t.toOracle)
    (ℓs : List Location) :
    (s.buryAll ℓs).toOracle = (t.buryAll ℓs).toOracle := by
  induction ℓs generalizing s t with
  | nil => exact h
  | cons ℓ ℓs ih =>
    rw [State.buryAll_cons, State.buryAll_cons]
    refine ih ?_
    rw [State.bury_eq_set, State.bury_eq_set]
    exact set_toOracle_congr h ℓ rfl

/--
A location of two states that read alike sorted reads alike sorted.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `h`: The states read alike sorted.
-/
theorem sortContent_get_congr {s t : State} (h : s.toOracle = t.toOracle)
    (ℓ : Location) : sortContent ℓ (s.get ℓ) = sortContent ℓ (t.get ℓ) := by
  cases ℓ with
  | register r =>
    show s.value (.register r) = t.value (.register r)
    rw [← State.value_toOracle, ← State.value_toOracle t, h]
  | record r =>
    exact Record.sort_eq_sort_iff.mpr (by
      show (s.record r).toOracle = (t.record r).toOracle
      rw [← State.record_toOracle, ← State.record_toOracle, h])
  | answer => rfl

/--
A step from two states that read alike sorted has one law, read sorted.

# Parameters
- `i`: The instruction.

# Hypotheses
- `h`: The states read alike sorted.
-/
theorem step_toOracle_congr {s t : State} (h : s.toOracle = t.toOracle)
    (i : Instruction) :
    (step s i).map State.toOracle = (step t i).map State.toOracle := by
  rw [Outcomes.map_toOracle_step s, Outcomes.map_toOracle_step t, h]

/--
Running an instruction and then burying locations keeps two laws of states
that agree read sorted agreeing.

# Parameters
- `i`: The instruction.
- `B`: The locations buried after it.

# Hypotheses
- `h`: The laws agree read sorted.
-/
theorem map_buryAll_bind_step_toOracle {μ ν : PMF State}
    (h : μ.map State.toOracle = ν.map State.toOracle) (i : Instruction)
    (B : List Location) :
    ((μ.bind (step · i)).map (·.buryAll B)).map State.toOracle =
      ((ν.bind (step · i)).map (·.buryAll B)).map State.toOracle := by
  simp only [PMF.map_comp, PMF.map_bind]
  refine bind_congr_toOracle (fun s t hst => ?_) h
  exact map_congr_toOracle (g := State.toOracle ∘ (·.buryAll B))
    (fun s t hst => buryAll_toOracle_congr hst B) (step_toOracle_congr hst i)

/--
A world over two bases that read alike sorted reads alike sorted.

# Parameters
- `live`: The live locations.
- `laws`: Their laws.

# Hypotheses
- `h`: The bases read alike sorted.
-/
theorem map_toOracle_product_congr {base base' : State}
    (h : base.toOracle = base'.toOracle) (live : List Location)
    (laws : (ℓ : Location) → PMF (Content ℓ)) :
    (product base live laws).map State.toOracle =
      (product base' live laws).map State.toOracle := by
  induction live with
  | nil => simp only [product_nil, PMF.pure_map, h]
  | cons ℓ live ih =>
    rw [product_cons, product_cons, lift, lift, PMF.map_bind, PMF.map_bind]
    refine congrArg _ (funext fun x => ?_)
    rw [PMF.map_comp, PMF.map_comp]
    exact map_congr_toOracle (g := State.toOracle ∘ (·.set ℓ x))
      (fun s t hst => set_toOracle_congr hst ℓ rfl) ih

/--
A world in which a live location is a point mass reads sorted as the world in
which it is a point mass at its value sorted. So a split may fix a rolling
record at its sorted reading.

# Parameters
- `ℓ`: The location.
- `x`: The value that it certainly holds.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
-/
theorem map_toOracle_product_pure {base : State} {live : List Location}
    {laws : (ℓ : Location) → PMF (Content ℓ)} {ℓ : Location} (x : Content ℓ)
    (hnd : live.Nodup) (h : ℓ ∈ live) :
    (product base live
        (Function.update laws ℓ (PMF.pure (sortContent ℓ x)))).map
          State.toOracle =
      (product base live (Function.update laws ℓ (PMF.pure x))).map
        State.toOracle := by
  rw [product_pure _ hnd h, product_pure _ hnd h]
  exact map_toOracle_product_congr
    (set_toOracle_congr rfl ℓ (sortContent_sortContent ℓ x)) _ _

/-! ## Outcomes of the open splits -/

variable {vs : Values}

/--
Draw one more split's outcome in each world.

# Parameters
- `K`: The laws of the splits' outcomes.
- `W`: The law of the worlds.
- `v`: The split.

# Returns
The law of the worlds, each extended by an outcome of `v` drawn from its law
in that world.
-/
noncomputable def drawSplit (K : SplitLaws vs) (W : PMF (Outcomes vs))
    (v : Nat) : PMF (Outcomes vs) :=
  W.bind fun o => (K v o).map (Function.update o v)

/--
The law of the worlds of a stack of splits: each split's outcome drawn in
turn, from the bottom, given those beneath it.

# Parameters
- `K`: The laws of the splits' outcomes.
- `splits`: The stack of splits, from the bottom.

# Returns
The law of the outcomes. Values that are not in the stack keep their blanks.

# Examples
```lean
example (K : SplitLaws vs) :
    splitOutcomes K [] = PMF.pure fun v => blank (vs.location v) := rfl
```
-/
noncomputable def splitOutcomes (K : SplitLaws vs) (splits : List Nat) :
    PMF (Outcomes vs) :=
  splits.foldl (drawSplit K) (PMF.pure fun v => blank (vs.location v))

/--
Drawing one more split atop a stack draws it last.

# Parameters
- `K`: The laws of the splits' outcomes.
- `splits`: The stack.
- `v`: The split.
-/
theorem splitOutcomes_append_singleton (K : SplitLaws vs)
    (splits : List Nat) (v : Nat) :
    splitOutcomes K (splits ++ [v]) =
      drawSplit K (splitOutcomes K splits) v := by
  simp [splitOutcomes, List.foldl_append]

/--
The worlds of a stack depend only on the laws of its splits.

# Parameters
- `K`, `K'`: The laws of the splits' outcomes.
- `splits`: The stack.

# Hypotheses
- `h`: The laws agree on every split in the stack.
-/
theorem splitOutcomes_congr {K K' : SplitLaws vs} {splits : List Nat}
    (h : ∀ v ∈ splits, K v = K' v) :
    splitOutcomes K splits = splitOutcomes K' splits := by
  unfold splitOutcomes
  generalize PMF.pure (fun v => blank (vs.location v)) = W
  induction splits generalizing W with
  | nil => rfl
  | cons v splits ih =>
    simp only [List.foldl_cons, drawSplit, h v (.head _)]
    exact ih (fun u hu => h u (.tail _ hu)) _

/--
Two splits, neither of whose laws depends on the other's outcome, may be drawn
in either order.

# Parameters
- `K`: The laws of the splits' outcomes.
- `W`: The law of the worlds before them.
- `a`, `b`: The splits.

# Hypotheses
- `hne`: The splits differ.
- `hab`: `b`'s law does not depend on `a`'s outcome.
- `hba`: `a`'s law does not depend on `b`'s outcome.
-/
theorem drawSplit_comm {K : SplitLaws vs} (W : PMF (Outcomes vs)) {a b : Nat}
    (hne : a ≠ b) (hab : ∀ o x, K b (Function.update o a x) = K b o)
    (hba : ∀ o y, K a (Function.update o b y) = K a o) :
    drawSplit K (drawSplit K W a) b = drawSplit K (drawSplit K W b) a := by
  simp only [drawSplit, PMF.bind_bind, PMF.bind_map, Function.comp_def, hab,
    hba]
  refine congrArg W.bind (funext fun o => ?_)
  simp only [← PMF.bind_pure_comp, Function.comp_def]
  rw [PMF.bind_comm]
  simp only [Function.update_comm hne]

/--
A split whose law depends on none of the outcomes of the splits above it, and
on whose outcome none of theirs depends, may be drawn after them.

# Parameters
- `K`: The laws of the splits' outcomes.
- `l`: The splits beneath it.
- `a`: The split.
- `r`: The splits above it.

# Hypotheses
- `h`: Each split above differs from `a`, its law does not depend on `a`'s
  outcome, and `a`'s does not depend on its.
-/
theorem splitOutcomes_move {K : SplitLaws vs} (l : List Nat) {a : Nat}
    {r : List Nat}
    (h : ∀ b ∈ r, a ≠ b ∧ (∀ o x, K b (Function.update o a x) = K b o) ∧
      ∀ o y, K a (Function.update o b y) = K a o) :
    splitOutcomes K (l ++ a :: r) = splitOutcomes K (l ++ r ++ [a]) := by
  unfold splitOutcomes
  simp only [List.foldl_append, List.foldl_cons, List.foldl_nil]
  generalize List.foldl (drawSplit K) _ l = W
  induction r generalizing W with
  | nil => rfl
  | cons b r ih =>
    obtain ⟨hne, hab, hba⟩ := h b (.head _)
    rw [List.foldl_cons, List.foldl_cons, drawSplit_comm W hne hab hba]
    exact ih (fun b' hb' => h b' (.tail _ hb')) _

/-! ## The invariant -/

/--
The invariant of a plan at a program point: once its dead locations are
buried, a law of states, read sorted, is the mixture of the worlds of the open
splits, each independent laws of the live values' locations, as the forward
pass holds them, read sorted.

# Parameters
- `vs`: The values of the function.
- `pc`: The program point: the number of instructions run.
- `live`: The live values.
- `splits`: The open splits, by value, from the bottom.
- `μ`: The law of states.
- `buried`: The locations buried.
- `base`: The state of every location whose value is not live.
- `K`: The law of each split's outcome.
- `laws`: The law of each live location in each world.
-/
structure Worlds (vs : Values) (pc : Nat) (live splits : List Nat)
    (μ : PMF State) (buried : List Location) (base : State) (K : SplitLaws vs)
    (laws : WorldLaws vs) : Prop where
  /-- Once `buried` is buried, the law of states is, read sorted, the
  mixture, over the worlds of the open splits, of independent laws of the live
  locations. -/
  law : (μ.map (·.buryAll buried)).map State.toOracle =
    ((splitOutcomes K splits).bind fun o =>
      product base (live.map vs.location) (laws o)).map State.toOracle
  /-- Every location buried holds a value that is not live. -/
  buried_dead : ∀ ℓ ∈ buried, ∃ v, vs.writer pc ℓ = some v ∧ v ∉ live
  /-- A live value's law depends only on the outcomes of the open splits on
  which it depends. -/
  laws_dependsOn : ∀ v ∈ live, ∀ o o' : Outcomes vs,
    (∀ u ∈ vs.dependsOn v, u ∈ splits → o u = o' u) →
      laws o (vs.location v) = laws o' (vs.location v)
  /-- A split's law depends only on the outcomes of the other open splits on
  which it depends. -/
  splitLaws_dependsOn : ∀ v ∈ splits, ∀ o o' : Outcomes vs,
    (∀ u ∈ vs.dependsOn v, u ≠ v → u ∈ splits → o u = o' u) →
      K v o = K v o'
  /-- A split's every outcome is sorted, as the multisets of the results of a
  rolling record are. -/
  splitLaws_sorted : ∀ v ∈ splits, ∀ o : Outcomes vs, ∀ x ∈ (K v o).support,
    sortContent (vs.location v) x = x
  /-- A live open split is a point mass at its outcome. -/
  split_pure : ∀ v ∈ live, v ∈ splits → ∀ o : Outcomes vs,
    laws o (vs.location v) = PMF.pure (o v)
  /-- A live value that no roll reaches is a point mass. -/
  fixed_pure : ∀ v ∈ live, vs.random v = false → ∀ o : Outcomes vs,
    ∃ x, laws o (vs.location v) = PMF.pure x
  /-- The base has the destination of every instruction. -/
  has : ∀ i ∈ vs.instructions, base.Has i.destination

/--
The invariant of a plan at a program point holds of a law of states in the
worlds of a forward pass: `Worlds`, with the pass's witnesses.

# Parameters
- `F`: The forward pass.
- `pc`: The program point.
- `live`: The live values.
- `splits`: The open splits, by value, from the bottom.
- `μ`: The law of states.
-/
abbrev Forward.Holds (F : Forward vs) (pc : Nat) (live splits : List Nat)
    (μ : PMF State) : Prop :=
  Worlds vs pc live splits μ F.buried F.base F.K F.laws

/--
The invariant holds of the initial state, with nothing live and no split open,
in the forward pass's one world.

# Parameters
- `s`: The initial state.

# Hypotheses
- `h`: The state has the destination of every instruction.
-/
theorem invariant_zero {s : State}
    (h : ∀ i ∈ vs.instructions, s.Has i.destination) :
    (Forward.initial vs s).Holds 0 [] [] (PMF.pure s) where
  law := by
    show (PMF.map (fun t : State => t.buryAll []) (PMF.pure s)).map _ = _
    rw [show (fun t : State => t.buryAll []) = id from rfl, PMF.map_id,
      splitOutcomes, List.foldl_nil, PMF.pure_bind]
    rfl
  buried_dead := by simp [Forward.initial]
  laws_dependsOn := by simp
  splitLaws_dependsOn := by simp
  splitLaws_sorted := by simp
  split_pure := by simp
  fixed_pure := by simp
  has := h

variable {pc : Nat} {live splits : List Nat} {μ : PMF State}
  {buried : List Location} {base : State} {K : SplitLaws vs}
  {laws : WorldLaws vs}

/--
In every world of the open splits, every outcome is sorted: a rolling
record's is the multiset of its results, as in the worlds of `propagation.rs`,
and that of a value not in the stack is blank.

# Parameters
- `o`: The world.
- `u`: A value.

# Hypotheses
- `h`: The invariant holds.
- `ho`: `o` is a world of the open splits.
-/
theorem Worlds.sortContent_outcome
    (h : Worlds vs pc live splits μ buried base K laws) {o : Outcomes vs}
    (ho : o ∈ (splitOutcomes K splits).support) (u : Nat) :
    sortContent (vs.location u) (o u) = o u := by
  unfold splitOutcomes at ho
  suffices ∀ l : List Nat, (∀ v ∈ l, v ∈ splits) → ∀ W : PMF (Outcomes vs),
      (∀ o ∈ W.support, ∀ u, sortContent (vs.location u) (o u) = o u) →
        ∀ o ∈ (l.foldl (drawSplit K) W).support, ∀ u,
          sortContent (vs.location u) (o u) = o u from
    this splits (fun _ hv => hv) _ (by simp) o ho u
  intro l
  induction l with
  | nil => exact fun _ _ hW => hW
  | cons v l ih =>
    intro hl W hW
    refine ih (fun w hw => hl w (.tail _ hw)) _ fun o ho u => ?_
    simp only [drawSplit, PMF.mem_support_bind_iff, PMF.support_map,
      Set.mem_image] at ho
    obtain ⟨o', ho', x, hx, rfl⟩ := ho
    by_cases hu : u = v
    · subst hu
      simp only [Function.update_self]
      exact h.splitLaws_sorted u (hl u (.head _)) o' x hx
    · rw [Function.update_of_ne hu]
      exact hW o' ho' u

/-! ## Splits -/

/--
A split keeps the invariant: its value's world is the mixture of those in which
it is a point mass at each of its outcomes, and, read sorted, of those in which
it is a point mass at each outcome sorted (`map_toOracle_product_pure`), which
is drawn atop the stack.

# Parameters
- `v`: The split.

# Hypotheses
- `h`: The invariant holds before the split.
- `hf`: `v` fans out.
- `hl`: `v` is live.
- `hv`: `v` is not open.
- `hnd`: The live values occupy distinct locations.
-/
theorem Worlds.split (h : Worlds vs pc live splits μ buried base K laws)
    {v : Nat} (hf : vs.fansOut v) (hl : v ∈ live) (hv : v ∉ splits)
    (hnd : (live.map vs.location).Nodup) :
    Worlds vs pc live (splits ++ [v]) μ buried base
      (Function.update K v fun o =>
        (laws o (vs.location v)).map (sortContent (vs.location v)))
      (fun o => Function.update (laws o) (vs.location v) (PMF.pure (o v))) := by
  have hloc : ∀ w ∈ live, w ≠ v → vs.location w ≠ vs.location v :=
    fun w hw hne he => hne (List.inj_on_of_nodup_map hnd hw hl he)
  -- The law of a live value other than `v` ignores `v`'s outcome.
  have hlaws : ∀ w ∈ live, ∀ o x, laws (Function.update o v x) (vs.location w) =
      laws o (vs.location w) := fun w hw o x =>
    h.laws_dependsOn w hw _ _ fun u _ hu => Function.update_of_ne
      (fun he : u = v => hv (he ▸ hu)) _ _
  constructor
  · rw [h.law, splitOutcomes_append_singleton,
      splitOutcomes_congr (K := Function.update K v fun o =>
        (laws o (vs.location v)).map (sortContent (vs.location v))) (K' := K)
        fun u hu => Function.update_of_ne
          (fun he : u = v => hv (he ▸ hu)) _ _, drawSplit, PMF.bind_bind,
      PMF.map_bind, PMF.map_bind]
    refine congrArg _ (funext fun o => ?_)
    rw [PMF.bind_map, Function.update_self, PMF.bind_map, PMF.map_bind,
      ← bind_pure_product (hnd := hnd) (List.mem_map_of_mem hl), PMF.map_bind]
    refine congrArg _ (funext fun x => ?_)
    simp only [Function.comp_apply, Function.update_self]
    rw [← map_toOracle_product_pure x hnd (List.mem_map_of_mem hl)]
    refine congrArg _ (product_congr fun ℓ hℓ => ?_)
    by_cases he : ℓ = vs.location v
    · subst he
      simp only [Function.update_self]
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
      rw [Function.update_of_ne he, Function.update_of_ne he, hlaws w hw]
  · exact fun ℓ hℓ => h.buried_dead ℓ hℓ
  · intro w hw o o' ho
    by_cases he : w = v
    · subst he
      simp only [Function.update_self]
      rw [ho w (vs.self_mem_dependsOn hf) (List.mem_append_right _ (.head _))]
    · rw [Function.update_of_ne (hloc w hw he), Function.update_of_ne
        (hloc w hw he)]
      exact h.laws_dependsOn w hw o o' fun u hu hs =>
        ho u hu (List.mem_append_left _ hs)
  · intro w hw o o' ho
    by_cases he : w = v
    · subst he
      simp only [Function.update_self]
      exact congrArg (PMF.map _) (h.laws_dependsOn w hl o o' fun u hu hs =>
        ho u hu (fun he => hv (he ▸ hs)) (List.mem_append_left _ hs))
    · have hw' : w ∈ splits := by
        rcases List.mem_append.mp hw with hw | hw
        · exact hw
        · exact absurd (List.mem_singleton.mp hw) he
      rw [Function.update_of_ne he]
      exact h.splitLaws_dependsOn w hw' o o' fun u hu hne hs =>
        ho u hu hne (List.mem_append_left _ hs)
  · intro w hw o x hx
    by_cases he : w = v
    · subst he
      simp only [Function.update_self, PMF.support_map, Set.mem_image] at hx
      obtain ⟨y, _, rfl⟩ := hx
      exact sortContent_sortContent _ y
    · have hw' : w ∈ splits := by
        rcases List.mem_append.mp hw with hw | hw
        · exact hw
        · exact absurd (List.mem_singleton.mp hw) he
      rw [Function.update_of_ne he] at hx
      exact h.splitLaws_sorted w hw' o x hx
  · intro w hw hs o
    by_cases he : w = v
    · subst he
      simp only [Function.update_self]
    · have hw' : w ∈ splits := by
        rcases List.mem_append.mp hs with hs | hs
        · exact hs
        · exact absurd (List.mem_singleton.mp hs) he
      rw [Function.update_of_ne (hloc w hw he)]
      exact h.split_pure w hw hw' o
  · intro w hw hr o
    by_cases he : w = v
    · subst he
      exact ⟨o w, Function.update_self ..⟩
    · rw [Function.update_of_ne (hloc w hw he)]
      exact h.fixed_pure w hw hr o
  · exact h.has

/-! ## Merges -/

/--
A stack without the split at a position holds every other split in it, if no
split is in it twice.

# Parameters
- `splits`: The stack.
- `k`: The position.
- `u`: A split.

# Hypotheses
- `hnd`: No split is in the stack twice.
- `hk`: `k` is in range.
-/
private theorem mem_eraseIdx_iff {splits : List Nat} {k u : Nat}
    (hnd : splits.Nodup) (hk : k < splits.length) :
    u ∈ splits.eraseIdx k ↔ u ∈ splits ∧ u ≠ splits[k] := by
  rw [List.mem_eraseIdx_iff_getElem]
  constructor
  · rintro ⟨i, hi, hik, rfl⟩
    exact ⟨List.getElem_mem hi, fun he =>
      hik ((List.Nodup.getElem_inj_iff hnd).mp he)⟩
  · rintro ⟨hu, hne⟩
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hu
    exact ⟨i, hi, fun he => hne (by simp [he]), rfl⟩

/--
A merge's split may be drawn last, since no split above depends on it, and
then mixed into the survivor's law in each world, since no other live value
depends on it. These are the two facts from which a merge's law follows, so
that `XdySpec/Ghost.lean` can follow the worlds' outcomes through it too.

# Parameters
- `k`: The position of the split in the stack.
- `sv`: The location of the survivor, if any.

# Hypotheses
- `h`: The invariant holds before the merge.
- `hs`: The splits ascend.
- `hk`: `k` is in range.
- `hsv`: The split may merge, into `sv`.
- `hnd`: The live values occupy distinct locations.

# Returns
That the worlds of the stack are those of the stack without the split, each
extended by the split's outcome drawn last; and that mixing each world over
that outcome gives the world whose survivor's law the merge mixes.
-/
theorem Worlds.merge_world (h : Worlds vs pc live splits μ buried base K laws)
    (hs : splits.Pairwise (· < ·)) {k : Nat} (hk : k < splits.length)
    {sv : Option Location} (hsv : vs.survivor splits k live = some sv)
    (hnd : (live.map vs.location).Nodup) :
    splitOutcomes K splits =
        drawSplit K (splitOutcomes K (splits.eraseIdx k)) splits[k] ∧
      ∀ o, ((K splits[k] o).bind fun x => product base (live.map vs.location)
          (laws (Function.update o splits[k] x))) =
        product base (live.map vs.location)
          (mergeLaws K laws splits[k] sv o) := by
  have hsplit : splits =
      splits.take k ++ splits[k] :: splits.drop (k + 1) := by
    rw [List.getElem_cons_drop, List.take_append_drop]
  -- The splits above the merging one were written after it.
  have habove : ∀ j ∈ splits.drop (k + 1), splits[k] < j := by
    have := hs
    rw [hsplit, List.pairwise_append] at this
    exact fun j hj => List.rel_of_pairwise_cons this.2.1 hj
  -- Every live value other than the survivor ignores the split's outcome.
  have hlaws : ∀ w ∈ live, splits[k] ∉ vs.dependsOn w →
      ∀ (o : Outcomes vs) (x : Content (vs.location splits[k])),
        laws (Function.update o splits[k] x) (vs.location w) =
          laws o (vs.location w) := fun w hw hdep o x =>
    h.laws_dependsOn w hw _ _ fun u hu _ => Function.update_of_ne
      (fun he : u = splits[k] => hdep (he ▸ hu)) _ _
  refine ⟨?_, fun o => ?_⟩
  · -- Draw the split last.
    rw [List.eraseIdx_eq_take_drop_succ, ← splitOutcomes_append_singleton,
      ← splitOutcomes_move]
    · exact congrArg _ hsplit
    · intro j hj
      have hlt := habove j hj
      refine ⟨Nat.ne_of_lt hlt, fun o x => (h.splitLaws_dependsOn j
        (List.mem_of_mem_drop hj) _ _ fun u hu _ _ => ?_).symm,
          fun o y => h.splitLaws_dependsOn _ (List.getElem_mem hk) _ _
            fun u hu _ _ => ?_⟩
      · refine (Function.update_of_ne (fun he : u = splits[k] => ?_) _ _).symm
        exact vs.not_mem_dependsOn_of_survivor hk hsv j hj (he ▸ hu)
      · refine (Function.update_of_ne (fun he : u = j => ?_) _ _)
        have := (vs.fansOut_of_mem_dependsOn hu).2
        omega
  · -- Then mix it into the survivor.
    cases sv with
    | none =>
      have hnone := vs.not_mem_dependsOn_of_survivor_none hk hsv
      simp only [mergeLaws]
      rw [bind_congr_support fun x _ => product_congr fun ℓ hℓ => by
        obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
        exact hlaws w hw (hnone w hw) o x]
      exact PMF.bind_const _ _
    | some ℓ =>
      obtain ⟨w, hw, hdw, _, hwℓ, _, huniq⟩ := vs.survivor_some hk hsv
      refine (bind_product hnd (hwℓ ▸ List.mem_map_of_mem hw) (K splits[k] o)
        (fun x => laws (Function.update o splits[k] x)) (laws o)
        fun x _ ℓ' hℓ' hne => ?_).trans rfl
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hℓ'
      exact hlaws v hv (fun hdep => hne (by rw [huniq v hv hdep, hwℓ])) o x

/--
A merge keeps the invariant: its split's outcome may be drawn last, since no
split above depends on it, and then mixed into the survivor's law, since no
other live value depends on it (`Worlds.merge_world`).

# Parameters
- `k`: The position of the split in the stack.
- `sv`: The location of the survivor, if any.

# Hypotheses
- `h`: The invariant holds before the merge.
- `hs`: The splits ascend.
- `hk`: `k` is in range.
- `hsv`: The split may merge, into `sv`.
- `hnd`: The live values occupy distinct locations.
-/
theorem Worlds.merge (h : Worlds vs pc live splits μ buried base K laws)
    (hs : splits.Pairwise (· < ·)) {k : Nat} (hk : k < splits.length)
    {sv : Option Location} (hsv : vs.survivor splits k live = some sv)
    (hnd : (live.map vs.location).Nodup) :
    Worlds vs pc live (splits.eraseIdx k) μ buried base K
      (mergeLaws K laws splits[k] sv) := by
  have hsnd : splits.Nodup := hs.imp Nat.ne_of_lt
  have hmem : ∀ {u}, u ∈ splits.eraseIdx k ↔ u ∈ splits ∧ u ≠ splits[k] :=
    mem_eraseIdx_iff hsnd hk
  -- A value that ignores the split's outcome, and depends only on the other
  -- open splits, depends only on those that remain.
  have hrest : ∀ {w} {o o' : Outcomes vs}, splits[k] ∉ vs.dependsOn w →
      (∀ u ∈ vs.dependsOn w, u ∈ splits.eraseIdx k → o u = o' u) →
        ∀ u ∈ vs.dependsOn w, u ∈ splits → o u = o' u :=
    fun hdep ho u hu hs' => ho u hu (hmem.mpr ⟨hs', fun he => hdep (he ▸ hu)⟩)
  constructor
  · obtain ⟨hW, hworld⟩ := h.merge_world hs hk hsv hnd
    rw [h.law, hW, drawSplit, PMF.bind_bind]
    refine congrArg (PMF.map _) (congrArg _ (funext fun o => ?_))
    rw [PMF.bind_map]
    exact hworld o
  · exact h.buried_dead
  · intro v hv o o' ho
    cases sv with
    | none =>
      have hdep := vs.not_mem_dependsOn_of_survivor_none hk hsv v hv
      exact h.laws_dependsOn v hv o o' (hrest hdep ho)
    | some ℓ =>
      obtain ⟨w, hw, hdw, _, hwℓ, _, huniq⟩ := vs.survivor_some hk hsv
      simp only [mergeLaws]
      by_cases he : vs.location v = ℓ
      · have hvw : v = w :=
          List.inj_on_of_nodup_map hnd hv hw (he.trans hwℓ.symm)
        subst hvw
        rw [he]
        simp only [Function.update_self]
        rw [h.splitLaws_dependsOn _ (List.getElem_mem hk) o o'
          fun u hu hne hs' => ho u (vs.mem_dependsOn_trans hdw hu)
            (hmem.mpr ⟨hs', hne⟩)]
        refine congrArg _ (funext fun x => ?_)
        rw [← he]
        refine h.laws_dependsOn v hv _ _ fun u hu hs' => ?_
        by_cases hu' : u = splits[k]
        · subst hu'
          simp only [Function.update_self]
        · rw [Function.update_of_ne hu', Function.update_of_ne hu']
          exact ho u hu (hmem.mpr ⟨hs', hu'⟩)
      · rw [Function.update_of_ne he, Function.update_of_ne he]
        have hdep : splits[k] ∉ vs.dependsOn v := fun hdep =>
          he (by rw [huniq v hv hdep, hwℓ])
        exact h.laws_dependsOn v hv o o' (hrest hdep ho)
  · intro j hj o o' ho
    obtain ⟨hj, hjk⟩ := hmem.mp hj
    have hdep : splits[k] ∉ vs.dependsOn j := fun hdep =>
      hjk (vs.eq_of_mem_dependsOn_of_survivor hs hk hsv hj hdep)
    exact h.splitLaws_dependsOn j hj o o' fun u hu hne hs' =>
      ho u hu hne (hmem.mpr ⟨hs', fun he => hdep (he ▸ hu)⟩)
  · exact fun j hj => h.splitLaws_sorted j (hmem.mp hj).1
  · intro v hv hs' o
    obtain ⟨hs', _⟩ := hmem.mp hs'
    cases sv with
    | none => exact h.split_pure v hv hs' o
    | some ℓ =>
      obtain ⟨w, hw, hdw, hwk, hwℓ, _, _⟩ := vs.survivor_some hk hsv
      have he : vs.location v ≠ ℓ := fun he => by
        have hvw : v = w :=
          List.inj_on_of_nodup_map hnd hv hw (he.trans hwℓ.symm)
        subst hvw
        exact hwk (vs.eq_of_mem_dependsOn_of_survivor hs hk hsv hs' hdw)
      simp only [mergeLaws]
      rw [Function.update_of_ne he]
      exact h.split_pure v hv hs' o
  · intro v hv hr o
    cases sv with
    | none => exact h.fixed_pure v hv hr o
    | some ℓ =>
      obtain ⟨w, hw, hdw, _, hwℓ, _, _⟩ := vs.survivor_some hk hsv
      have he : vs.location v ≠ ℓ := fun he => by
        have hvw : v = w :=
          List.inj_on_of_nodup_map hnd hv hw (he.trans hwℓ.symm)
        subst hvw
        rw [vs.random_of_mem_dependsOn hdw] at hr
        cases hr
      simp only [mergeLaws]
      rw [Function.update_of_ne he]
      exact h.fixed_pure v hv hr o
  · exact h.has

/--
Every merge that the plan schedules after an instruction keeps the invariant,
in the worlds of the forward pass, which performs them in turn.

# Parameters
- `pc'`: The program counter of the instruction.
- `current`: The values that occupy their locations.
- `splits`: The stack before the merges.
- `F`: The forward pass before the merges.

# Hypotheses
- `hnd`: The live values occupy distinct locations.
- `hs`: The splits ascend.
- `h`: The invariant holds before the merges, in the worlds of `F`.

# Returns
That performing the merges leaves the stack that the plan leaves, and the
invariant holds in the worlds of the pass that they leave.
-/
theorem invariant_merges {current : List Nat} {pc' : Nat}
    (hnd : (live.map vs.location).Nodup) (splits : List Nat)
    (hs : splits.Pairwise (· < ·)) (F : Forward vs)
    (h : F.Holds pc live splits μ) :
    let r := (vs.merges live current splits).1.foldl (Forward.perform pc')
      (F, splits)
    r.2 = (vs.merges live current splits).2 ∧ r.1.Holds pc live r.2 μ := by
  rcases vs.merges_cases live current splits with he | ⟨k, hk, sv, hsv, he⟩
  · simp only [he, List.foldl_nil]
    exact ⟨trivial, h⟩
  · simp only [he, List.foldl_cons, Forward.perform,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk, Option.getD_some]
    exact invariant_merges hnd (splits.eraseIdx k)
      (hs.sublist (List.eraseIdx_sublist _ _)) _ (h.merge hs hk hsv hnd)
termination_by splits.length
decreasing_by rw [List.length_eraseIdx_of_lt hk]; omega

/-! ## Instructions -/

/--
An instruction keeps the invariant. The values live before it split three
ways: `P`, which it reads and which stay live, and so are point masses; `D`,
which it reads for the last time, and which are buried once it runs; and `T`,
which it does not read, and so neither reads nor writes. Its value takes the
law of the instruction run on the world of `P` and `D`, as the forward pass
gives it (`Forward.exec`).

Besides the invariant after it, it answers the two facts from which the law
follows, which hold of any state and of each world alone, so that
`XdySpec/Ghost.lean` can follow the worlds' outcomes through it too.

# Parameters
- `pc`: The program counter of the instruction.
- `F`: The forward pass before the instruction.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hpc`: `pc` is an instruction of the function.
- `h`: The invariant holds before the instruction, in the worlds of `F`.

# Returns
That the invariant holds after the instruction, in the worlds of the pass
that runs it; that burying what the pass has buried before it changes no state
that it can reach, once the pass buries what it buries after; and that it
takes each world, so buried, to the pass's world after it.
-/
theorem invariant_step_world
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hpc : pc < vs.instructions.length) {F : Forward vs}
    (h : F.Holds pc (vs.liveBefore pc) (vs.openSplits pc) μ) :
    let F' := F.exec pc vs.instructions[pc]
    F'.Holds (pc + 1) (vs.live pc) (vs.openSplits pc)
        (μ.bind (step · vs.instructions[pc])) ∧
      (∀ t : State, (step (t.buryAll F.buried) vs.instructions[pc]).map
          (·.buryAll F'.buried) =
        (step t vs.instructions[pc]).map (·.buryAll F'.buried)) ∧
      ∀ o, ((product F.base ((vs.liveBefore pc).map vs.location)
          (F.laws o)).bind (step · vs.instructions[pc])).map
            (·.buryAll F'.buried) =
        product F'.base ((vs.live pc).map vs.location) (F'.laws o) := by
  rcases F with ⟨B, base, K, laws⟩
  have hdest : vs.instructions[pc].destination = vs.location pc :=
    (vs.location_eq hpc).symm
  have hval := hvals _ (List.getElem_mem hpc)
  -- The values live before the instruction, split three ways.
  let r : Nat → Bool := fun v => (vs.reads pc).contains v
  let a : Nat → Bool := fun v => (vs.live pc).contains v
  let P := ((vs.liveBefore pc).filter r).filter a
  let D := ((vs.liveBefore pc).filter r).filter fun v => !a v
  let T := (vs.liveBefore pc).filter fun v => !r v
  have hP : ∀ {v}, v ∈ P ↔ v ∈ vs.liveBefore pc ∧ v ∈ vs.reads pc ∧
      v ∈ vs.live pc := by
    intro v; simp only [P, r, a, List.mem_filter, List.contains_iff_mem]
    tauto
  have hD : ∀ {v}, v ∈ D ↔ v ∈ vs.liveBefore pc ∧ v ∈ vs.reads pc ∧
      v ∉ vs.live pc := by
    intro v
    simp only [D, r, a, List.mem_filter, List.contains_iff_mem,
      Bool.not_eq_true', ← Bool.not_eq_true]
    tauto
  have hT : ∀ {v}, v ∈ T ↔ v ∈ vs.liveBefore pc ∧ v ∉ vs.reads pc := by
    intro v; simp [T, r]
  have hperm : (vs.liveBefore pc).Perm (P ++ D ++ T) :=
    ((List.filter_append_perm a _).append_right T |>.trans
      (List.filter_append_perm r _)).symm
  have hlt : ∀ {v}, v ∈ vs.liveBefore pc → v < pc :=
    fun hv => ((vs.mem_liveBefore).mp hv).1
  have hnodup : (P ++ D ++ T).Nodup :=
    hperm.nodup_iff.mp (vs.nodup_liveBefore pc)
  -- The live values after it are `P` and `T`, and its own if it is live.
  have hnodupPT : (P ++ T).Nodup :=
    hnodup.sublist ((List.sublist_append_left P D).append_right T)
  have hpcPT : pc ∉ P ++ T := fun hm => by
    rcases List.mem_append.mp hm with hm | hm
    · exact Nat.lt_irrefl _ (hlt (hP.mp hm).1)
    · exact Nat.lt_irrefl _ (hlt (hT.mp hm).1)
  have hlive : ∀ {v}, v ∈ vs.live pc ↔ v ∈ P ∨ v ∈ T ∨
      (v = pc ∧ pc ∈ vs.live pc) := by
    intro v
    constructor
    · intro hv
      by_cases he : v = pc
      · exact .inr (.inr ⟨he, he ▸ hv⟩)
      · have hb := vs.mem_liveBefore_of_mem_live hv he
        by_cases hr : v ∈ vs.reads pc
        · exact .inl (hP.mpr ⟨hb, hr, hv⟩)
        · exact .inr (.inl (hT.mpr ⟨hb, hr⟩))
    · rintro (hv | hv | ⟨rfl, hv⟩)
      · exact (hP.mp hv).2.2
      · exact vs.mem_live_of_mem_liveBefore (hT.mp hv).1 (hT.mp hv).2
      · exact hv
  have hnodupLive : (vs.live pc).Nodup := vs.nodup_liveBefore (pc + 1)
  -- Their locations.
  have hlocB : ((vs.liveBefore pc).map vs.location).Nodup :=
    vs.nodup_map_location_liveBefore (Nat.le_of_lt hpc)
  have hlocA : ((vs.live pc).map vs.location).Nodup :=
    vs.nodup_map_location_liveBefore (pc := pc + 1) hpc
  have hlocPDT : ((P ++ D ++ T).map vs.location).Nodup :=
    (hperm.map _).nodup_iff.mp hlocB
  have hocc : ∀ {v}, v ∈ vs.liveBefore pc →
      vs.writer pc (vs.location v) = some v :=
    vs.writer_of_mem_liveBefore (Nat.le_of_lt hpc)
  have hned : ∀ {v}, v ∈ vs.live pc → v ≠ pc →
      vs.location v ≠ vs.location pc :=
    fun hv hne => vs.location_ne_of_mem_live hpc hv hne
  -- The instruction neither reads nor writes the locations of `T`.
  have hTr : ∀ ℓ ∈ T.map vs.location,
      ℓ.readBy vs.instructions[pc] = false := by
    intro ℓ hℓ
    obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
    obtain ⟨hwb, hwr⟩ := hT.mp hw
    cases hrb : (vs.location w).readBy vs.instructions[pc]
    · rfl
    · exact absurd (vs.mem_reads_of_readBy hpc hval hrb (hocc hwb)) hwr
  have hTw : ∀ ℓ ∈ T.map vs.location,
      ℓ.writtenBy vs.instructions[pc] = false := by
    intro ℓ hℓ
    obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
    obtain ⟨hwb, hwr⟩ := hT.mp hw
    cases hwb' : (vs.location w).writtenBy vs.instructions[pc]
    · rfl
    · exact absurd ((Location.writtenBy_iff.mp hwb').trans hdest)
        (hned (vs.mem_live_of_mem_liveBefore hwb hwr)
          (Nat.ne_of_lt (hlt hwb)))
  -- It does not write the locations of `P`, which are point masses.
  have hPd : vs.location pc ∉ P.map vs.location := fun hm => by
    obtain ⟨w, hw, he⟩ := List.mem_map.mp hm
    obtain ⟨hwb, _, hwl⟩ := hP.mp hw
    exact hned hwl (Nat.ne_of_lt (hlt hwb)) he
  have hPpure : ∀ o, ∀ ℓ ∈ P.map vs.location, ∃ x, laws o ℓ = PMF.pure x := by
    intro o ℓ hℓ
    obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hℓ
    obtain ⟨hwb, hwr, hwl⟩ := hP.mp hw
    cases hrand : vs.random w
    · exact h.fixed_pure w hwb hrand o
    · exact ⟨o w, h.split_pure w hwb (vs.mem_openSplits_of_random hwr hrand
        ((vs.mem_live).mp hwl).2) o⟩
  -- What to bury after it: `D`, what was buried before, but for what it
  -- writes, and its own value, unless it is live.
  have hBoff : ∀ ℓ ∈ B, ∀ w ∈ vs.liveBefore pc, ℓ ≠ vs.location w :=
    fun ℓ hℓ w hw he => by
      obtain ⟨v, hv, hvl⟩ := h.buried_dead ℓ hℓ
      rw [he, hocc hw] at hv
      exact hvl (Option.some.inj hv ▸ hw)
  let E := (if pc ∈ vs.live pc then [] else [vs.location pc]) ++
    B.filter (· != vs.location pc)
  let B' := (D.map vs.location).erase (vs.location pc) ++ E
  have hBr : ∀ ℓ ∈ B, ℓ.readBy vs.instructions[pc] = false := by
    intro ℓ hℓ
    obtain ⟨v, hv, hvl⟩ := h.buried_dead ℓ hℓ
    cases hrb : ℓ.readBy vs.instructions[pc]
    · rfl
    · exact absurd (vs.mem_liveBefore_of_mem_reads
        (vs.mem_reads_of_readBy hpc hval hrb hv)) hvl
  have hBw : ∀ ℓ ∈ B, ℓ.writtenBy vs.instructions[pc] = true ∨ ℓ ∈ B' := by
    intro ℓ hℓ
    by_cases he : ℓ = vs.location pc
    · left
      rw [he, ← hdest]
      exact writtenBy_destination _
    · right
      simp [B', E, hℓ, he]
  -- The law of its value, in each world.
  let N : Outcomes vs → PMF (Content (vs.location pc)) := fun o =>
    ((product base ((P ++ D).map vs.location) (laws o)).bind
      (step · vs.instructions[pc])).map (·.get (vs.location pc))
  let laws' : WorldLaws vs := fun o =>
    Function.update (laws o) (vs.location pc) (N o)
  let base' :=
    (base.buryAll ((D.map vs.location).erase (vs.location pc))).buryAll E
  have hworld : ∀ o, ((product base ((vs.liveBefore pc).map vs.location)
      (laws o)).bind (step · vs.instructions[pc])).map (·.buryAll B') =
        product base' ((vs.live pc).map vs.location) (laws' o) := by
    intro o
    have hstep := bind_step_product_pure (base := base) (laws := laws o)
      (P := P.map vs.location) (S := D.map vs.location)
      (T := T.map vs.location) (by simpa using hlocPDT) (hPpure o)
      (hdest ▸ hPd) hTr hTw (h.has _ (List.getElem_mem hpc))
    rw [hdest] at hstep
    rw [product_perm (hperm.map _) hlocB,
      show (fun s : State => s.buryAll B') =
        (fun s => s.buryAll E) ∘ (fun s => s.buryAll
          ((D.map vs.location).erase (vs.location pc))) from
        funext fun s => by simp [B', State.buryAll, List.foldl_append],
      ← PMF.map_comp]
    simp only [List.map_append] at hstep ⊢
    rw [hstep]
    -- Every other location buried holds no live value.
    have hdisj : ∀ ℓ ∈ B.filter (· != vs.location pc),
        ℓ ∉ P.map vs.location ++ vs.location pc :: T.map vs.location := by
      intro ℓ hℓ hm
      simp only [List.mem_filter, bne_iff_ne, ne_eq] at hℓ
      simp only [List.mem_append, List.mem_cons, List.mem_map] at hm
      rcases hm with ⟨w, hw, rfl⟩ | rfl | ⟨w, hw, rfl⟩
      · exact hBoff _ hℓ.1 w (hP.mp hw).1 rfl
      · exact hℓ.2 rfl
      · exact hBoff _ hℓ.1 w (hT.mp hw).1 rfl
    have hsub : ∀ {l : List Nat}, (∀ v ∈ l, v ∈ P ∨ v ∈ T) → l.Nodup →
        (∀ ℓ ∈ B.filter (· != vs.location pc), ℓ ∉ l.map vs.location) := by
      intro l hl _ ℓ hℓ hm
      obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hm
      refine hdisj _ hℓ ?_
      rcases hl w hw with hw | hw
      · exact List.mem_append_left _ (List.mem_map_of_mem hw)
      · exact List.mem_append_right _ (.tail _ (List.mem_map_of_mem hw))
    by_cases hpl : pc ∈ vs.live pc
    · -- Its value is live: bury only what was buried before.
      have hp : (vs.live pc).Perm (P ++ pc :: T) :=
        (List.perm_ext_iff_of_nodup hnodupLive
          (List.nodup_middle.mpr (List.nodup_cons.mpr ⟨hpcPT, hnodupPT⟩))).mpr
          fun v => by
            rw [hlive]
            simp only [List.mem_append, List.mem_cons]
            constructor
            · rintro (hv | hv | ⟨rfl, _⟩)
              · exact .inl hv
              · exact .inr (.inr hv)
              · exact .inr (.inl rfl)
            · rintro (hv | rfl | hv)
              · exact .inl hv
              · exact .inr (.inr ⟨rfl, hpl⟩)
              · exact .inr (.inl hv)
      have hl : (P.map vs.location ++ vs.location pc :: T.map vs.location) =
          (P ++ pc :: T).map vs.location := by simp
      simp only [E, hpl, ↓reduceIte, List.nil_append]
      rw [map_buryAll_product_of_disjoint hdisj, hl,
        product_perm (hp.map _).symm ((hp.map _).nodup_iff.mp hlocA)]
      simp only [base', laws', N, E, hpl, ↓reduceIte, List.nil_append,
        List.map_append]
    · -- Its value is dead: bury it too.
      have hp : (vs.live pc).Perm (P ++ T) :=
        (List.perm_ext_iff_of_nodup hnodupLive hnodupPT).mpr fun v => by
          rw [hlive, List.mem_append]
          constructor
          · rintro (hv | hv | ⟨_, hv⟩)
            · exact .inl hv
            · exact .inr hv
            · exact absurd hv hpl
          · rintro (hv | hv)
            · exact .inl hv
            · exact .inr (.inl hv)
      have hndd : (P.map vs.location ++ vs.location pc ::
          T.map vs.location).Nodup := by
        rw [List.nodup_middle, List.nodup_cons, ← List.map_append]
        refine ⟨fun hm => ?_, (hp.map _).nodup_iff.mp hlocA⟩
        obtain ⟨w, hw, he⟩ := List.mem_map.mp hm
        rcases List.mem_append.mp hw with hw | hw
        · exact hPd (he ▸ List.mem_map_of_mem hw)
        · have ⟨hwb, hwr⟩ := hT.mp hw
          exact hned (vs.mem_live_of_mem_liveBefore hwb hwr)
            (Nat.ne_of_lt (hlt hwb)) he
      have herase : (P.map vs.location ++ vs.location pc ::
          T.map vs.location).erase (vs.location pc) =
            (P ++ T).map vs.location := by
        rw [List.erase_append_right _ hPd, List.erase_cons_head,
          List.map_append]
      simp only [E, hpl, ↓reduceIte, List.singleton_append]
      rw [show (fun s : State => s.buryAll (vs.location pc ::
          B.filter (· != vs.location pc))) =
            (fun s => s.buryAll (B.filter (· != vs.location pc))) ∘
              (fun s => s.bury (vs.location pc)) from rfl,
        ← PMF.map_comp, map_bury_product hndd, herase,
        map_buryAll_product_of_disjoint
          (hsub (fun v hv => List.mem_append.mp hv) hnodupPT),
        product_perm (hp.map _).symm ((hp.map _).nodup_iff.mp hlocA)]
      simp only [base', laws', N, E, hpl, ↓reduceIte, List.singleton_append,
        List.map_append]
      rfl
  -- The values that the instruction reads.
  have hPD : ∀ {u}, u ∈ P ++ D → u ∈ vs.liveBefore pc ∧ u ∈ vs.reads pc :=
    fun hu => by
      rcases List.mem_append.mp hu with hu | hu
      · exact ⟨(hP.mp hu).1, (hP.mp hu).2.1⟩
      · exact ⟨(hD.mp hu).1, (hD.mp hu).2.1⟩
  have hlaws' : ∀ v ∈ vs.live pc, v ≠ pc → ∀ o,
      laws' o (vs.location v) = laws o (vs.location v) :=
    fun v hv hne _ => Function.update_of_ne (hned hv hne) _ _
  show Worlds vs (pc + 1) (vs.live pc) (vs.openSplits pc) _ B' base' K laws' ∧
    (∀ t : State, (step (t.buryAll B) vs.instructions[pc]).map
        (·.buryAll B') = (step t vs.instructions[pc]).map (·.buryAll B')) ∧
    ∀ o, ((product base ((vs.liveBefore pc).map vs.location)
        (laws o)).bind (step · vs.instructions[pc])).map (·.buryAll B') =
      product base' ((vs.live pc).map vs.location) (laws' o)
  refine ⟨?_, fun t => by
    simpa [PMF.pure_map, PMF.pure_bind] using
      map_buryAll_bind_step (μ := PMF.pure t) hBr hBw, hworld⟩
  refine ⟨?_, ?_, ?_, h.splitLaws_dependsOn, h.splitLaws_sorted, ?_, ?_, ?_⟩
  · rw [← map_buryAll_bind_step hBr hBw,
      map_buryAll_bind_step_toOracle h.law, PMF.bind_bind, PMF.map_bind]
    exact congrArg (PMF.map _) (congrArg _ (funext hworld))
  · intro ℓ hℓ
    rw [List.mem_append] at hℓ
    rcases hℓ with hℓ | hℓ
    · -- A value that it read for the last time.
      have hDnd : (D.map vs.location).Nodup := hlocPDT.sublist (by
        rw [List.map_append, List.map_append]
        exact (List.sublist_append_right _ _).trans
          (List.sublist_append_left _ _))
      obtain ⟨hne, hℓ⟩ := (hDnd.mem_erase_iff).mp hℓ
      obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hℓ
      obtain ⟨hub, _, hul⟩ := hD.mp hu
      exact ⟨u, by
        simp only [vs.writer_succ, Ne.symm hne, ↓reduceIte, hocc hub], hul⟩
    · simp only [E, List.mem_append, List.mem_filter, bne_iff_ne, ne_eq] at hℓ
      rcases hℓ with hℓ | ⟨hℓ, hne⟩
      · -- Its own value, which is dead.
        by_cases hpl : pc ∈ vs.live pc
        · simp [hpl] at hℓ
        · simp only [hpl, ↓reduceIte, List.mem_singleton] at hℓ
          subst hℓ
          exact ⟨pc, by simp [vs.writer_succ], hpl⟩
      · -- A value buried before, which is still dead.
        obtain ⟨v, hv, hvl⟩ := h.buried_dead ℓ hℓ
        refine ⟨v, by simp only [vs.writer_succ, Ne.symm hne, ↓reduceIte, hv],
          fun hva => hvl (vs.mem_liveBefore_of_mem_live hva ?_)⟩
        exact Nat.ne_of_lt (vs.lt_of_writer hv)
  · intro v hv o o' ho
    by_cases he : v = pc
    · subst he
      simp only [laws', Function.update_self, N]
      refine congrArg (PMF.map _) (congrArg (PMF.bind · _)
        (product_congr fun ℓ hℓ => ?_))
      obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hℓ
      obtain ⟨hub, hur⟩ := hPD hu
      exact h.laws_dependsOn u hub o o' fun w hw hs =>
        ho w (vs.mem_dependsOn_of_mem_reads hur hw) hs
    · rw [hlaws' v hv he, hlaws' v hv he]
      exact h.laws_dependsOn v (vs.mem_liveBefore_of_mem_live hv he) o o' ho
  · intro v hv hs o
    have he : v ≠ pc := Nat.ne_of_lt (vs.mem_openSplits hs).2
    rw [hlaws' v hv he]
    exact h.split_pure v (vs.mem_liveBefore_of_mem_live hv he) hs o
  · intro v hv hr o
    by_cases he : v = pc
    · subst he
      obtain ⟨hroll, hreads⟩ := vs.not_random hr
      obtain ⟨t, ht⟩ := product_eq_pure (base := base) (laws := laws o)
        (live := (P ++ D).map vs.location) fun ℓ hℓ => by
          obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hℓ
          obtain ⟨hub, hur⟩ := hPD hu
          exact h.fixed_pure u hub (hreads u hur) o
      obtain ⟨t', ht'⟩ := step_eq_pure (s := t) (i := vs.instructions[v])
        (by simpa [List.getElem?_eq_getElem hpc] using hroll)
      exact ⟨t'.get (vs.location v), by
        simp only [laws', Function.update_self, N]
        rw [ht, PMF.pure_bind, ht', PMF.pure_map]⟩
    · rw [hlaws' v hv he]
      exact h.fixed_pure v (vs.mem_liveBefore_of_mem_live hv he) hr o
  · exact fun i hi => (State.has_buryAll _ _ _).mpr
      ((State.has_buryAll _ _ _).mpr (h.has i hi))

/--
An instruction keeps the invariant: `invariant_step_world`, without the facts
from which its law follows.

# Parameters
- `pc`: The program counter of the instruction.
- `F`: The forward pass before the instruction.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hpc`: `pc` is an instruction of the function.
- `h`: The invariant holds before the instruction, in the worlds of `F`.
-/
theorem invariant_step
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hpc : pc < vs.instructions.length) {F : Forward vs}
    (h : F.Holds pc (vs.liveBefore pc) (vs.openSplits pc) μ) :
    (F.exec pc vs.instructions[pc]).Holds (pc + 1) (vs.live pc)
      (vs.openSplits pc) (μ.bind (step · vs.instructions[pc])) :=
  (invariant_step_world hvals hpc h).1

/-! ## Every step of the plan -/

/--
The steps that follow an instruction keep the invariant: running it, splitting
its value if it fans out, and merging every split that may, lead from the
values live and the splits open before it to those before the next, in the
worlds of the forward pass that performs them.

# Parameters
- `pc`: The program counter of the instruction.
- `F`: The forward pass before the instruction.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hpc`: `pc` is an instruction of the function.
- `h`: The invariant holds before the instruction, in the worlds of `F`.
-/
theorem invariant_succ
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hpc : pc < vs.instructions.length) {F : Forward vs}
    (h : F.Holds pc (vs.liveBefore pc) (vs.openSplits pc) μ) :
    ((vs.steps pc).foldl (Forward.perform pc)
        (F.exec pc vs.instructions[pc], vs.openSplits pc)).1.Holds (pc + 1)
      (vs.liveBefore (pc + 1)) (vs.openSplits (pc + 1))
      (μ.bind (step · vs.instructions[pc])) := by
  have hstep := invariant_step hvals hpc h
  have hnd : ((vs.live pc).map vs.location).Nodup :=
    vs.nodup_map_location_liveBefore (pc := pc + 1) hpc
  simp only [vs.steps_eq, List.foldl_append]
  rw [vs.liveBefore_succ, vs.openSplits_succ]
  cases hf : vs.fansOut pc
  · simp only [Bool.false_eq_true, ↓reduceIte, List.foldl_nil]
    obtain ⟨he, hw⟩ := invariant_merges (current := vs.current pc) (pc' := pc)
      hnd _ (vs.pairwise_openSplits pc) (F.exec pc vs.instructions[pc]) hstep
    rw [← he]
    exact hw
  · simp only [↓reduceIte, List.foldl_cons, List.foldl_nil, Forward.perform]
    obtain ⟨he, hw⟩ := invariant_merges (current := vs.current pc) (pc' := pc)
      hnd _ (List.pairwise_append.mpr ⟨vs.pairwise_openSplits pc, by simp,
        fun a ha b hb => by
          rw [List.mem_singleton.mp hb]
          exact (vs.mem_openSplits ha).2⟩)
      ((F.exec pc vs.instructions[pc]).split pc)
      (hstep.split hf (vs.mem_live_self hf)
        (fun h => Nat.lt_irrefl _ (vs.mem_openSplits h).2) hnd)
    rw [← he]
    exact hw

/--
The invariant holds throughout, in the worlds of the forward pass: before each
instruction, the law of states that running the function's instructions so
far answers is, read sorted, the mixture of the worlds of the splits open
there, as the pass holds them.

# Parameters
- `s`: The initial state.
- `pc`: The program point.

# Hypotheses
- `hvals`: Every operand of every instruction is a value.
- `hhas`: The initial state has the destination of every instruction.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem invariant_exec
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true) {s : State}
    (hhas : ∀ i ∈ vs.instructions, s.Has i.destination)
    (hpc : pc ≤ vs.instructions.length) :
    (forward vs s pc).Holds pc (vs.liveBefore pc) (vs.openSplits pc)
      (exec (vs.instructions.take pc) (PMF.pure s)) := by
  induction pc with
  | zero => exact invariant_zero hhas
  | succ pc ih =>
    rw [List.take_add_one, List.getElem?_eq_getElem (by omega),
      Option.toList_some, exec_append, exec_cons, exec_nil]
    simp only [forward, List.getElem?_eq_getElem (Nat.lt_of_succ_le hpc)]
    exact invariant_succ hvals (by omega) (ih (by omega))

/--
The invariant holds throughout a well-formed function, from its initial
state, in the worlds of the forward pass. The law of states before each
instruction, as `run` computes it, is, read sorted, the mixture of the worlds
of the splits open there, as the pass holds them.

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
theorem invariant_run {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (hv : f.validate = .ok ()) (hs : f.initialState args externals = .ok s)
    {pc : Nat} (hpc : pc ≤ f.instructions.size) :
    (forward ⟨f.instructions.toList⟩ (.ofOracle s) pc).Holds pc
      ((Values.mk f.instructions.toList).liveBefore pc)
      ((Values.mk f.instructions.toList).openSplits pc)
      (exec (f.instructions.toList.take pc) (PMF.pure (.ofOracle s))) :=
  invariant_exec (operandsAreValues_of_validate hv) (has_initialState hv hs)
    (by simpa using hpc)

/--
The forward pass's base, at every program point of a well-formed function,
has the location of every value: the destination of its instruction, or the
answer.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.
- `pc`: The program point.
- `v`: The value.

# Hypotheses
- `hv`: The function is well formed.
- `hs`: Its initial state is `s`.
- `hpc`: `pc` is at most the number of instructions.
-/
theorem has_forward {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (hv : f.validate = .ok ()) (hs : f.initialState args externals = .ok s)
    {pc : Nat} (hpc : pc ≤ f.instructions.size) (v : Nat) :
    (forward ⟨f.instructions.toList⟩ (.ofOracle s) pc).base.Has
      ((Values.mk f.instructions.toList).location v) := by
  cases h : (Values.mk f.instructions.toList).instructions[v]? with
  | none => simp [Values.location, h, State.Has]
  | some i =>
    rw [Values.location, h, Option.map_some, Option.getD_some]
    exact (invariant_run hv hs hpc).has i (List.mem_of_getElem? h)

/-! ## The answer -/

/--
An array whose last element is `x` is its elements before the last, then `x`.

# Parameters
- `a`: The array.
- `x`: Its last element.

# Hypotheses
- `h`: The last element of `a` is `x`.
-/
private theorem toList_eq_of_back? {α : Type} {a : Array α} {x : α}
    (h : a.back? = some x) : a.toList = a.toList.take (a.size - 1) ++ [x] := by
  rw [Array.back?_eq_getElem?] at h
  have hlt : a.size - 1 < a.size := by
    rcases Nat.eq_zero_or_pos a.size with hz | hz
    · simp [hz] at h
    · omega
  rw [Array.getElem?_eq_getElem hlt, Option.some.injEq] at h
  conv_lhs => rw [← List.take_append_drop (a.size - 1) a.toList]
  congr 1
  rw [List.drop_eq_getElem_cons (by simpa using hlt)]
  simp [← h, show a.size - 1 + 1 = a.size by omega]

/--
Burying locations that an operand does not read leaves its value alone.

# Parameters
- `s`: The state.
- `ℓs`: The locations.
- `op`: The operand.

# Hypotheses
- `h`: The operand reads none of the locations.
-/
private theorem value_buryAll {s : State} {ℓs : List Location}
    {op : AddressingMode} (h : ∀ ℓ ∈ ℓs, ℓ.isOperand op = false) :
    (s.buryAll ℓs).value op = s.value op := by
  induction ℓs generalizing s with
  | nil => rfl
  | cons ℓ ℓs ih =>
    rw [State.buryAll_cons, ih fun ℓ' h' => h ℓ' (.tail _ h'),
      State.value_bury (h ℓ (.head _))]

/--
A function that answers a law is well formed.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `p`: The law of the result.

# Hypotheses
- `h`: The function answers `p`.
-/
private theorem validate_of_run {f : Function} {args : List Int}
    {externals : List (String × Int)} {p : PMF Int}
    (h : run f args externals = .ok p) : f.validate = .ok () := by
  unfold run at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i u hu
    cases u
    exact hu

/--
The answer of a function that returns a register is the mixture, over the
worlds of the splits open before the return, of the forward pass's law of the
register in each, as the pass mixes the returned register over the worlds that
remain. Its value is live before the return, so the worlds of the pass there,
in which the invariant holds (`invariant_run`), hold its law, since reading a
register cannot tell the order of any record's results; in particular, if
it is an open split, it is a point mass at its outcome in each world
(`Worlds.split_pure`), and if no roll reaches it, a point mass
(`Worlds.fixed_pure`).

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `p`: The law of the result.
- `s`: The initial state.
- `r`: The returned register.
- `v`: The value in it before the return.

# Hypotheses
- `hrun`: The function answers `p`.
- `hs`: Its initial state is `s`.
- `hsrc`: It returns `r`.
- `hw`: `v` is in `r` before the return.

# Notes
Not every split has merged before the return: one whose value is itself
returned, and read before, stays open until the return has read it, as the
tests show. A function that returns a constant, or a register that no
instruction writes, answers a point mass, which needs no invariant.
-/
theorem run_eq_forward {f : Function} {args : List Int}
    {externals : List (String × Int)} {p : PMF Int} {s : Xdy.State} {r v : Nat}
    (hrun : run f args externals = .ok p)
    (hs : f.initialState args externals = .ok s)
    (hsrc : f.instructions.back? = some (.return (.register r)))
    (hw : (Values.mk f.instructions.toList).writer (f.instructions.size - 1)
      (.register r) = some v) :
    let F := forward ⟨f.instructions.toList⟩ (.ofOracle s)
      (f.instructions.size - 1)
    v ∈ (Values.mk f.instructions.toList).liveBefore (f.instructions.size - 1) ∧
      F.Holds (f.instructions.size - 1)
        ((Values.mk f.instructions.toList).liveBefore (f.instructions.size - 1))
        ((Values.mk f.instructions.toList).openSplits (f.instructions.size - 1))
        (exec (f.instructions.toList.take (f.instructions.size - 1))
          (PMF.pure (.ofOracle s))) ∧
      p = (splitOutcomes F.K ((Values.mk f.instructions.toList).openSplits
        (f.instructions.size - 1))).bind fun o => F.laws o (.register r) := by
  let vs : Values := ⟨f.instructions.toList⟩
  have hv := validate_of_run hrun
  have hvals := operandsAreValues_of_validate hv
  have hlist := toList_eq_of_back? hsrc
  have hn : f.instructions.size - 1 < vs.instructions.length := by
    have := congrArg List.length hlist
    simp only [List.length_append, List.length_take, List.length_singleton,
      Array.length_toList] at this
    simp [vs]
    omega
  have hret : vs.instructions[f.instructions.size - 1] =
      .return (.register r) := by
    simp only [vs, List.getElem_of_eq hlist]
    simp
  -- The return reads `v`, so it is live before it.
  have hread : v ∈ vs.reads (f.instructions.size - 1) :=
    vs.mem_reads_of_readBy hn (hvals _ (List.getElem_mem hn))
      (by rw [hret]; simp [Location.readBy, Location.isOperand]) hw
  have hlive := vs.mem_liveBefore_of_mem_reads hread
  have h := invariant_run hv hs (pc := f.instructions.size - 1) (by omega)
  dsimp only at h ⊢
  generalize forward _ _ _ = F at h ⊢
  rcases F with ⟨buried, base, K, laws⟩
  refine ⟨hlive, h, ?_⟩
  obtain ⟨s', src, hs', hsrc', hp⟩ := run_eq_exec hrun
  rw [hs] at hs'
  cases hs'
  rw [hsrc] at hsrc'
  cases hsrc'
  have hloc : vs.location v = .register r := vs.location_of_writer hw
  -- No location buried is the returned register, whose value is live.
  have hunb : ∀ ℓ ∈ buried, ℓ.isOperand (.register r) = false := by
    intro ℓ hℓ
    obtain ⟨w, hw', hwl⟩ := h.buried_dead ℓ hℓ
    cases hb : ℓ.isOperand (.register r)
    · rfl
    · have : ℓ = .register r := by simpa [Location.isOperand] using hb
      subst this
      rw [hw] at hw'
      cases hw'
      exact absurd hlive hwl
  have hvlt : v < vs.instructions.length :=
    Nat.lt_trans ((vs.mem_liveBefore).mp hlive).1 hn
  have hhas : base.Has (.register r) := by
    have := h.has _ (List.getElem_mem hvlt)
    rwa [← vs.location_eq hvlt, hloc] at this
  -- The return changes nothing, and reads the register from the worlds.
  rw [hp]
  conv_lhs => rw [hlist]
  rw [exec_append, exec_cons, exec_nil]
  simp only [step, PMF.bind_pure]
  -- Reading a register cannot tell the order of a record's results.
  have hget : (fun t : State => t.get (.register r)) =
      (fun t : Xdy.State => t.value (.register r)) ∘ State.toOracle :=
    funext fun t => (State.value_toOracle t (.register r)).symm
  rw [show (fun t : State => t.value (.register r)) =
      (fun t : State => t.get (.register r)) ∘ (·.buryAll buried) from
    funext fun t => (value_buryAll hunb).symm, ← PMF.map_comp, hget,
    ← PMF.map_comp, h.law, PMF.map_comp, ← hget, PMF.map_bind]
  refine congrArg _ (funext fun o => ?_)
  exact map_get_product (vs.nodup_map_location_liveBefore (Nat.le_of_lt hn))
    (hloc ▸ List.mem_map_of_mem hlive) hhas

end Xdy.Spec
