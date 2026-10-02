import Xdy.Plan
import XdySpec.Convolution

/-!
# Conditioning and merging at the elimination point

Lemma 6 of the design: conditioning on a value that fans out, and merging its
worlds at its elimination point, preserves the law of the program.

The forward pass in `xdy/src/distribution/propagation.rs` holds each register
as its own distribution, and combines the operands of an instruction pair by
pair, which lemma 1 justifies only when they are independent. They are not
when one random value reaches both. So the pass conditions on such a value, as
the plan in `propagation/plan.rs` directs: it splits each of its worlds into
one for each outcome of the value, in which the value is fixed, and later
merges them, once one value carries the whole influence of the conditioned
one, by mixing that value's distributions over the worlds.

This module proves the laws on which that rests, over the specification's
joint law of machine states:

- Splitting never changes the law (`exec_split`): a law of states is the
  mixture, over the outcomes of any value, of its law given each outcome
  (`bind_cond`), in which the value is fixed (`map_cond`), and running the
  rest of the program distributes over the mixture. Where the value
  determines the state, each world is a point mass (`cond_map`).
- Burying a dead location never changes the law of the result
  (`exec_buryAll`): a location that the rest of the program overwrites before
  it reads it, or never reads, may be reset to a fixed value, as the pass
  folds dead values away before it merges.
- Merging mixes the survivor over a common factor (`merge`): if every world's
  law is the survivor's law, in that world, combined independently with one
  law of everything else, the same in every world, then the mixture of the
  worlds is the mixture of the survivor's laws, combined independently with
  that common law. That is the merged world that the pass keeps.
- A split may merge beneath a later split whose law it does not change
  (`merge_beneath`), which is why the pass may keep its splits as a stack.
  So may one above an earlier split, on which the rest may depend, by merging
  within each of its worlds.

`exec_merge` puts them together: a program split on a value after one run of
instructions, whose worlds factor as above after the next, once their dead
locations are buried, answers the same law when its worlds merge there. It
merges within the worlds of the other open splits, as the pass merges the
worlds that agree on the outcome of every other split: the rest may differ
between the worlds of the splits beneath, and the survivor and the rest may
both depend on the outcomes of the splits above, provided that those splits
are independent of the merging one.

The hypotheses of `merge` and `exec_merge` are laws of states: that each
world factors, and that the rest is the same in every world. The plan
establishes them from its analysis of which values depend on which.
`XdySpec/Invariant.lean` proves that its merges preserve the law without
them: before every instruction, the law of states is a mixture of worlds of
independent laws of the live locations (`XdySpec/World.lean`), and a merge
mixes the survivor's law over the split's outcome within them.
-/

namespace Xdy.Spec

/-! ## Running instructions on a law of states -/

/--
Run instructions on a law of states.

# Parameters
- `is`: The instructions, in order.
- `μ`: The law of the state before them.

# Returns
The law of the state after them. `run` runs a function's instructions this
way, from its initial state.

# Examples
```lean
example (μ : PMF State) : exec [] μ = μ := rfl
```
-/
noncomputable def exec (is : List Instruction) (μ : PMF State) : PMF State :=
  is.foldl (fun d i => d.bind (step · i)) μ

/--
Running no instructions leaves the law alone.

# Parameters
- `μ`: The law of the state.
-/
@[simp] theorem exec_nil (μ : PMF State) : exec [] μ = μ := rfl

/--
Running instructions runs the first, then the rest.

# Parameters
- `i`: The first instruction.
- `is`: The rest.
- `μ`: The law of the state before them.
-/
theorem exec_cons (i : Instruction) (is : List Instruction) (μ : PMF State) :
    exec (i :: is) μ = exec is (μ.bind (step · i)) := rfl

/--
Running two runs of instructions runs the first, then the second.

# Parameters
- `is`: The first run.
- `js`: The second.
- `μ`: The law of the state before them.
-/
theorem exec_append (is js : List Instruction) (μ : PMF State) :
    exec (is ++ js) μ = exec js (exec is μ) := by
  simp [exec, List.foldl_append]

/--
Running instructions on a mixture of laws mixes the laws of their runs.

# Parameters
- `is`: The instructions.
- `μ`: The law of the mixture's component.
- `k`: The law of the state in each component.
-/
theorem exec_bind {α : Type} (is : List Instruction) (μ : PMF α)
    (k : α → PMF State) :
    exec is (μ.bind k) = μ.bind fun a => exec is (k a) := by
  induction is generalizing k with
  | nil => rfl
  | cons i is ih =>
    simp only [exec_cons, PMF.bind_bind]
    exact ih _

/--
A function that succeeds answers the law of its returned operand after it
runs its instructions from its initial state.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `p`: The law of the result.

# Hypotheses
- `h`: The function answers `p`.
-/
theorem run_eq_exec {f : Function} {args : List Int}
    {externals : List (String × Int)} {p : PMF Int}
    (h : run f args externals = .ok p) :
    ∃ s src, f.initialState args externals = .ok s ∧
      f.instructions.back? = some (.return src) ∧
      p = (exec f.instructions.toList (PMF.pure (.ofOracle s))).map
        (·.value src) := by
  unfold run at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i s hs
      refine ⟨s, ?_⟩
      cases hb : f.instructions.back? with
      | none => rw [hb] at h; cases h
      | some i =>
        rw [hb] at h
        cases i
        case «return» src =>
          refine ⟨src, hs, rfl, ?_⟩
          cases h
          rw [exec, Array.foldl_toList]
        all_goals cases h

/-! ## Conditioning -/

variable {σ α : Type}

open scoped Classical in
/--
The conditional law of states given the outcome of a value.

# Parameters
- `μ`: The law of states.
- `g`: The value, read from a state.
- `v`: The outcome.

# Returns
The law of states given that `g` reads `v`, i.e., `μ` restricted to those
states and renormalized: the law of one world of a split on `g`.

# Notes
An outcome that `g` reads with probability `0` has no conditional law, so this
answers `μ` itself. Every mixture over the outcomes of `g` weighs it by `0`,
so any law would do.
-/
noncomputable def cond (μ : PMF σ) (g : σ → α) (v : α) : PMF σ :=
  if h : ∃ s ∈ {s | g s = v}, s ∈ μ.support then μ.filter _ h else μ

/--
Every state that a world of a split can reach reads the world's outcome.

# Parameters
- `μ`: The law of states.
- `g`: The value.
- `v`: The world's outcome.
- `s`: A state.

# Hypotheses
- `hv`: The outcome is possible.
- `hs`: The state is possible in the world.
-/
theorem eq_of_mem_support_cond {μ : PMF σ} {g : σ → α} {v : α} {s : σ}
    (hv : v ∈ (μ.map g).support) (hs : s ∈ (cond μ g v).support) :
    g s = v := by
  rw [PMF.support_map] at hv
  obtain ⟨t, ht, rfl⟩ := hv
  have h : ∃ s ∈ {s | g s = g t}, s ∈ μ.support := ⟨t, rfl, ht⟩
  rw [cond, dite_eq_left h, PMF.mem_support_filter_iff] at hs
  exact hs.1

/--
Every state that a world of a split can reach is possible before the split.

# Parameters
- `μ`: The law of states.
- `g`: The value.
- `v`: The world's outcome.
- `s`: A state.

# Hypotheses
- `hv`: The outcome is possible.
- `hs`: The state is possible in the world.
-/
theorem mem_support_of_mem_support_cond {μ : PMF σ} {g : σ → α} {v : α}
    {s : σ} (hv : v ∈ (μ.map g).support) (hs : s ∈ (cond μ g v).support) :
    s ∈ μ.support := by
  rw [PMF.support_map] at hv
  obtain ⟨t, ht, rfl⟩ := hv
  have h : ∃ s ∈ {s | g s = g t}, s ∈ μ.support := ⟨t, rfl, ht⟩
  rw [cond, dite_eq_left h, PMF.mem_support_filter_iff] at hs
  exact hs.2

/--
In each world of a split, the value is fixed at the world's outcome, as the
pass fixes it.

# Parameters
- `μ`: The law of states.
- `g`: The value.
- `v`: The world's outcome.

# Hypotheses
- `hv`: The outcome is possible.
-/
theorem map_cond {μ : PMF σ} {g : σ → α} {v : α}
    (hv : v ∈ (μ.map g).support) : (cond μ g v).map g = PMF.pure v := by
  rw [map_congr_support fun s hs => eq_of_mem_support_cond hv hs]
  exact PMF.map_const _ _

/--
Splitting a law of states that a value determines: if every state comes from
an outcome that the value reads back, then each world of a split on the value
is the point mass at the state of its outcome.

# Parameters
- `w`: The law of the outcome.
- `h`: The state of each outcome.
- `g`: The value.
- `v`: The world's outcome.

# Hypotheses
- `hg`: The value reads back the outcome of every state.
- `hv`: The outcome is possible.
-/
theorem cond_map {β : Type} {w : PMF β} {h : β → σ} {g : σ → β}
    (hg : ∀ x, g (h x) = x) {v : β} (hv : v ∈ w.support) :
    cond (w.map h) g v = PMF.pure (h v) := by
  have hv' : v ∈ ((w.map h).map g).support := by
    rwa [PMF.map_comp, show g ∘ h = id from funext hg, PMF.map_id]
  -- Every state possible in the world is the state of its outcome.
  have hsupp : ∀ s ∈ (cond (w.map h) g v).support, s = h v := by
    intro s hs
    have hgs := eq_of_mem_support_cond hv' hs
    have hmem := mem_support_of_mem_support_cond hv' hs
    rw [PMF.support_map] at hmem
    obtain ⟨x, _, rfl⟩ := hmem
    rw [hg] at hgs
    rw [hgs]
  ext t
  by_cases ht : t = h v
  · subst ht
    have := (cond (w.map h) g v).tsum_coe
    rwa [tsum_eq_single (h v) fun t ht => by
      by_contra hne
      exact ht (hsupp t ((PMF.mem_support_iff _ _).mpr hne)),
      ← PMF.pure_apply_self (h v)] at this
  · rw [PMF.pure_apply_of_ne _ _ ht]
    by_contra hne
    exact ht (hsupp t ((PMF.mem_support_iff _ _).mpr hne))

/--
The law of total probability: a law of states is the mixture, over the
outcomes of any value, of its conditional laws, each weighted by the
probability of its outcome.

# Parameters
- `μ`: The law of states.
- `g`: The value.
-/
theorem bind_cond (μ : PMF σ) (g : σ → α) : (μ.map g).bind (cond μ g) = μ := by
  ext s
  -- The probability of an outcome is the mass of the states that read it.
  have hmap : ∀ v, (μ.map g) v = ∑' t, {t | g t = v}.indicator μ t := by
    intro v
    rw [PMF.map_apply]
    refine tsum_congr fun t => ?_
    by_cases h : g t = v <;> simp [Set.indicator, h, Ne.symm]
  rw [PMF.bind_apply, tsum_eq_single (g s) fun v hv => ?_]
  · by_cases h : ∃ t ∈ {t | g t = g s}, t ∈ μ.support
    · rw [cond, dite_eq_left h, PMF.filter_apply, hmap,
        Set.indicator_of_mem (show s ∈ {t | g t = g s} from rfl),
        mul_comm (μ s), ← mul_assoc, ENNReal.mul_inv_cancel _ _, one_mul]
      · obtain ⟨t, ht, hts⟩ := h
        exact ENNReal.summable.tsum_ne_zero_iff.mpr ⟨t, by
          rwa [Set.indicator_of_mem ht, ← PMF.mem_support_iff]⟩
      · rw [← hmap]
        exact PMF.apply_ne_top _ _
    · have hs : μ s = 0 := by
        rw [PMF.apply_eq_zero_iff]
        exact fun hs => h ⟨s, rfl, hs⟩
      have hg : (μ.map g) (g s) = 0 := by
        rw [PMF.apply_eq_zero_iff, PMF.support_map]
        rintro ⟨t, ht, hts⟩
        exact h ⟨t, hts, ht⟩
      rw [hg, hs, zero_mul]
  · by_cases h : ∃ t ∈ {t | g t = v}, t ∈ μ.support
    · rw [cond, dite_eq_left h,
        PMF.filter_apply_eq_zero_of_notMem h fun hs => hv hs.symm, mul_zero]
    · have hg : (μ.map g) v = 0 := by
        rw [PMF.apply_eq_zero_iff, PMF.support_map]
        rintro ⟨t, ht, htv⟩
        exact h ⟨t, htv, ht⟩
      rw [hg, zero_mul]

/--
Splitting never changes the law: running instructions on a law of states is
running them in each world of a split on any value, weighted by the
probability of the world's outcome, and mixing the results.

# Parameters
- `is`: The instructions after the split.
- `μ`: The law of states at the split.
- `g`: The value.
-/
theorem exec_split (is : List Instruction) (μ : PMF State) (g : State → α) :
    exec is μ = (μ.map g).bind fun v => exec is (cond μ g v) := by
  rw [← exec_bind, bind_cond]

/-! ## Burying dead locations

A location is a register or a rolling record, or the answer, which the
specification reads from the returned operand instead; `Location` and the
instructions that read and write each, with `Location.dead`, live beside the
plan in `Xdy/Plan.lean`, which needs only core Lean.
-/

/-! ### Arrays modified at an index -/

/--
Modifying an element twice modifies it once, by the composite.

# Parameters
- `a`: The array.
- `i`: The index.
- `f`: The first modification.
- `g`: The second.
-/
theorem modify_modify {β : Type} (a : Array β) (i : Nat)
    (f g : β → β) : (a.modify i f).modify i g = a.modify i (g ∘ f) := by
  apply Array.ext
  · simp
  · intro j _ _
    simp only [Array.getElem_modify]
    split <;> rfl

/--
Modifying elements at distinct indices commutes.

# Parameters
- `a`: The array.
- `i`, `j`: The indices.
- `f`, `g`: The modifications.

# Hypotheses
- `h`: The indices differ.
-/
theorem modify_comm {β : Type} (a : Array β) {i j : Nat}
    (h : i ≠ j) (f g : β → β) :
    (a.modify i f).modify j g = (a.modify j g).modify i f := by
  apply Array.ext
  · simp
  · intro k _ _
    simp only [Array.getElem_modify]
    split <;> split <;> simp_all

/-! ### Burying locations of a state -/

namespace State

/--
Bury a location: reset it to a fixed value, forgetting what it held, as the
pass buries a dead value before a merge.

# Parameters
- `s`: The state.
- `ℓ`: The location.

# Returns
The state with `0` in the register, or an empty rolling record. The state
holds no answer, which the specification reads from the returned operand, so
burying it changes nothing.
-/
def bury (s : State) : Location → State
  | .register r => s.setRegister r 0
  | .record r => s.setRecord r {}
  | .answer => s

/--
Burying the answer changes nothing.

# Parameters
- `s`: The state.
-/
@[simp] theorem bury_answer (s : State) : s.bury .answer = s := rfl

/--
Bury locations, in order.

# Parameters
- `s`: The state.
- `ℓs`: The locations.

# Returns
The state with every location in `ℓs` buried.
-/
def buryAll (s : State) (ℓs : List Location) : State := ℓs.foldl bury s

/--
Burying a location leaves alone every operand that does not read it.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `op`: The operand.

# Hypotheses
- `h`: `op` does not read `ℓ`.
-/
theorem value_bury {s : State} {ℓ : Location} {op : AddressingMode}
    (h : ℓ.isOperand op = false) : (s.bury ℓ).value op = s.value op := by
  cases ℓ <;> cases op <;>
    simp_all [bury, value, setRegister, setRecord, Location.isOperand,
      Array.getElem?_modify]

/--
Burying a location leaves alone every other rolling record.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `r`: The rolling record.

# Hypotheses
- `h`: `ℓ` is not the rolling record.
-/
theorem record_bury {s : State} {ℓ : Location} {r : Nat}
    (h : ℓ ≠ .record r) : (s.bury ℓ).record r = s.record r := by
  cases ℓ <;>
    simp_all [bury, record, setRegister, setRecord, Array.getElem?_modify]

/--
Burying a location commutes with writing another register.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `d`: The register.
- `v`: The value.

# Hypotheses
- `h`: `ℓ` is not the register.
-/
theorem bury_setRegister {s : State} {ℓ : Location} {d : Nat} {v : Int}
    (h : ℓ ≠ .register d) :
    (s.setRegister d v).bury ℓ = (s.bury ℓ).setRegister d v := by
  cases ℓ <;> simp_all [bury, setRegister, setRecord, modify_comm]

/--
Burying a location commutes with replacing another rolling record.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `d`: The rolling record.
- `rec`: Its new contents.

# Hypotheses
- `h`: `ℓ` is not the rolling record.
-/
theorem bury_setRecord {s : State} {ℓ : Location} {d : Nat} {rec : Record}
    (h : ℓ ≠ .record d) :
    (s.setRecord d rec).bury ℓ = (s.bury ℓ).setRecord d rec := by
  cases ℓ <;> simp_all [bury, setRegister, setRecord, modify_comm]

/--
Writing a register forgets that it was buried.

# Parameters
- `s`: The state.
- `d`: The register.
- `v`: The value.
-/
theorem setRegister_bury (s : State) (d : Nat) (v : Int) :
    (s.bury (.register d)).setRegister d v = s.setRegister d v := by
  simp only [bury, setRegister, modify_modify]; rfl

/--
Replacing a rolling record forgets that it was buried.

# Parameters
- `s`: The state.
- `d`: The rolling record.
- `rec`: Its new contents.
-/
theorem setRecord_bury (s : State) (d : Nat) (rec : Record) :
    (s.bury (.record d)).setRecord d rec = s.setRecord d rec := by
  simp only [bury, setRecord, modify_modify]; rfl

end State

/--
An instruction that writes a location without reading it forgets that it was
buried.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `i`: The instruction.

# Hypotheses
- `hr`: `i` does not read `ℓ`.
- `hw`: `i` writes `ℓ`.
-/
theorem step_bury_of_written {s : State} {ℓ : Location} {i : Instruction}
    (hr : ℓ.readBy i = false) (hw : ℓ.writtenBy i = true) :
    step (s.bury ℓ) i = step s i := by
  cases i <;> cases ℓ <;>
    simp_all [Location.readBy, Location.writtenBy, step, State.value_bury,
      State.record_bury, State.setRegister_bury] <;>
    (subst_vars
     exact congrArg (PMF.map · _) (funext (State.setRecord_bury s _)))

/--
An instruction that neither reads nor writes a location runs alike whether or
not it was buried, and leaves it buried.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `i`: The instruction.

# Hypotheses
- `hr`: `i` does not read `ℓ`.
- `hw`: `i` does not write `ℓ`.
-/
theorem step_bury {s : State} {ℓ : Location} {i : Instruction}
    (hr : ℓ.readBy i = false) (hw : ℓ.writtenBy i = false) :
    step (s.bury ℓ) i = (step s i).map (·.bury ℓ) := by
  cases i <;> cases ℓ <;>
    simp_all [Location.readBy, Location.writtenBy, step, State.value_bury,
      State.record_bury, State.bury_setRegister, State.bury_setRecord,
      PMF.map_comp, PMF.pure_map, Function.comp_def]

/--
Burying a dead location never changes the law of the result: running the
rest of a function from a law of states answers the same law of its returned
operand whether or not a location that the rest never reads was buried.

# Parameters
- `ℓ`: The location.
- `src`: The returned operand.
- `is`: The rest of the function's instructions.
- `μ`: The law of states before them.

# Hypotheses
- `h`: `ℓ` is dead before `is`.
-/
theorem exec_bury {ℓ : Location} {src : AddressingMode}
    {is : List Instruction} (h : ℓ.dead src is) (μ : PMF State) :
    (exec is (μ.map (·.bury ℓ))).map (·.value src) =
      (exec is μ).map (·.value src) := by
  induction is generalizing μ with
  | nil =>
    simp only [Location.dead, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    rw [exec_nil, exec_nil, PMF.map_comp]
    exact congrArg (PMF.map · μ) (funext fun s => State.value_bury h)
  | cons i is ih =>
    simp only [Location.dead, Bool.and_eq_true, Bool.not_eq_eq_eq_not,
      Bool.not_true, Bool.or_eq_true] at h
    obtain ⟨hr, hw | hd⟩ := h
    · rw [exec_cons, exec_cons, PMF.bind_map]
      exact congrArg (fun μ => (exec is μ).map (·.value src))
        (congrArg μ.bind (funext fun s => step_bury_of_written hr hw))
    · by_cases hw : ℓ.writtenBy i
      · rw [exec_cons, exec_cons, PMF.bind_map]
        exact congrArg (fun μ => (exec is μ).map (·.value src))
          (congrArg μ.bind (funext fun s => step_bury_of_written hr hw))
      · rw [exec_cons, exec_cons, PMF.bind_map,
          show ((step · i) ∘ (·.bury ℓ)) = fun s => (step s i).map (·.bury ℓ)
            from funext fun s => step_bury hr (by simpa using hw),
          ← PMF.map_bind, ih hd]

/--
Burying dead locations never changes the law of the result, as `exec_bury`.

# Parameters
- `ℓs`: The locations.
- `src`: The returned operand.
- `is`: The rest of the function's instructions.
- `μ`: The law of states before them.

# Hypotheses
- `h`: Every location in `ℓs` is dead before `is`.
-/
theorem exec_buryAll {ℓs : List Location} {src : AddressingMode}
    {is : List Instruction} (h : ∀ ℓ ∈ ℓs, ℓ.dead src is)
    (μ : PMF State) :
    (exec is (μ.map (·.buryAll ℓs))).map (·.value src) =
      (exec is μ).map (·.value src) := by
  induction ℓs generalizing μ with
  | nil =>
    rw [show (fun s : State => s.buryAll []) = id from rfl, PMF.map_id]
  | cons ℓ ℓs ih =>
    rw [show (fun s : State => s.buryAll (ℓ :: ℓs)) =
        (fun s : State => s.buryAll ℓs) ∘ (fun s : State => s.bury ℓ) from rfl,
      ← PMF.map_comp, ih (fun ℓ' h' => h ℓ' (.tail _ h')),
      exec_bury (h ℓ (.head _))]

/-! ## Merging -/

variable {β γ δ : Type}

/--
Mixing laws that agree on every possible component mixes alike.

# Parameters
- `p`: The law of the component.
- `f`, `g`: The laws in each component.

# Hypotheses
- `h`: `f` and `g` agree on every possible component.
-/
theorem bind_congr_support {p : PMF α} {f g : α → PMF β}
    (h : ∀ a ∈ p.support, f a = g a) : p.bind f = p.bind g := by
  ext b
  simp only [PMF.bind_apply]
  refine tsum_congr fun a => ?_
  by_cases ha : a ∈ p.support
  · rw [h a ha]
  · rw [PMF.mem_support_iff, not_not] at ha
    simp [ha]

/--
Merging mixes the survivor over a common factor: if every world's law
combines the survivor's law, in that world, independently with one law of
everything else, the same in every world, then the mixture of the worlds
combines the mixture of the survivor's laws independently with that law.
Mirrors `mix` in `propagation.rs`.

# Parameters
- `w`: The law of the outcome of the split, which weighs each world.
- `d`: The law of the survivor in each world.
- `R`: The law of everything else, the same in every world.
- `c`: How a survivor and everything else make a whole.

# Notes
A merge without a survivor is the case in which `d` answers `PMF.pure ()` in
every world: every world is then `R`, and so is their mixture.
-/
theorem merge (w : PMF α) (d : α → PMF β) (R : PMF γ) (c : β → γ → δ) :
    w.bind (fun v => lift c (d v) R) = lift c (w.bind d) R := by
  simp only [lift, PMF.bind_bind]

/--
A split may merge beneath a later split whose law it does not change: mixing
over the earlier split within each world of the later one is mixing over the
later split within each world of the earlier one. So the pass may keep its
splits as a stack, merging one beneath any later split that does not depend on
it.

# Parameters
- `w`: The law of the outcome of the earlier split.
- `u`: The law of the outcome of the later split, the same in every world of
  the earlier.
- `k`: The law in each world of both.
-/
theorem merge_beneath (w : PMF α) (u : PMF β) (k : α → β → PMF γ) :
    (w.bind fun v => u.bind (k v)) = u.bind fun x => w.bind fun v => k v x :=
  PMF.bind_comm w u k

/-! ## Lemma 6 -/

/--
Lemma 6: conditioning on a value and merging at its elimination point
preserves the law of the result, within the worlds of the other open splits.
After instructions `is`, the law of states is a mixture `W` of worlds `ν`, one
for each outcome of the splits open beneath the merging one. Split each on
`g`, and run `js` in each world. If, once the locations `ℓs` that the rest
`ks` never reads are buried, each world is a mixture `U`, over the outcomes of
the splits above the merging one, of worlds that each combine the survivor's
law independently with one law `R` of everything else, and neither `U` nor
`R` depends on the outcome of `g`, then running `ks` from the merged worlds,
whose survivor has the mixture of its laws over the outcomes of `g`, answers
the law of running the whole. Mirrors `merge` in `propagation.rs`, which
merges the worlds that agree on the outcome of every other split.

# Parameters
- `is`: The instructions before the split.
- `js`: The instructions between the split and the merge.
- `ks`: The instructions after the merge.
- `μ`: The law of states before `is`.
- `W`: The law of the outcomes of the splits beneath, which weighs each of
  their worlds.
- `ν`: The law of states in each world of the splits beneath, at the split.
- `g`: The value of the split.
- `src`: The returned operand.
- `ℓs`: The locations to bury before the merge.
- `c`: How a survivor and everything else make a state.
- `U`: The law of the outcomes of the splits above, in each world of the
  splits beneath.
- `d`: The law of the survivor in each world of the splits beneath, of the
  merging split, and of the splits above.
- `R`: The law of everything else in each world of the splits beneath and
  above, the same in every world of the merging split.

# Hypotheses
- `hworlds`: The law of states after `is` is the mixture of the worlds of the
  splits beneath.
- `hdead`: Every location in `ℓs` is dead before `ks`.
- `hfactor`: In every possible world of the splits beneath and of the merging
  split, the law of states after `js`, with `ℓs` buried, is the mixture over
  `U` of `d` combined independently with `R`.

# Notes
The specification's states keep every correlation, even with values that the
pass has already read and forgotten, such as the results of a record already
summed. So `ℓs` must include every location that depends on the split and that
the rest never reads, not only those that the plan buries.

The worlds of the splits beneath are a mixture, not the conditional laws given
the values of those splits, since a later write may displace a split value
from the state, as `fixed` in `propagation.rs` keeps each outcome apart from
the state. A merge with no split open beneath takes `W` and `U` to be
`PMF.pure ()`, and a merge with none open above takes `U` to be.

That neither `U` nor `R` depends on the outcome of `g` is what lets the merge
mix the survivor alone: every split above the merging one is independent of
it, so the mixture over its outcomes may move beneath them (`merge_beneath`),
and everything else is common to its worlds (`merge`).
-/
theorem exec_merge {ω υ : Type} (is js ks : List Instruction) (μ : PMF State)
    (W : PMF ω) (ν : ω → PMF State) (g : State → α) (src : AddressingMode)
    (ℓs : List Location) (c : β → γ → State) (U : ω → PMF υ)
    (d : ω → α → υ → PMF β) (R : ω → υ → PMF γ)
    (hworlds : exec is μ = W.bind ν)
    (hdead : ∀ ℓ ∈ ℓs, ℓ.dead src ks)
    (hfactor : ∀ o ∈ W.support, ∀ v ∈ ((ν o).map g).support,
      (exec js (cond (ν o) g v)).map (·.buryAll ℓs) =
        (U o).bind fun u => lift c (d o v u) (R o u)) :
    (exec (is ++ js ++ ks) μ).map (·.value src) =
      (exec ks (W.bind fun o => (U o).bind fun u =>
        lift c (((ν o).map g).bind fun v => d o v u) (R o u))).map
          (·.value src) := by
  rw [exec_append, exec_append, hworlds, exec_bind, exec_bind, PMF.map_bind,
    exec_bind, PMF.map_bind]
  refine bind_congr_support fun o ho => ?_
  -- Within a world of the splits beneath, mix over the outcomes of `g` last.
  have hmix : ((U o).bind fun u =>
      lift c (((ν o).map g).bind fun v => d o v u) (R o u)) =
      ((ν o).map g).bind fun v => (U o).bind fun u =>
        lift c (d o v u) (R o u) := by
    simp only [← merge]
    exact merge_beneath _ _ _
  rw [hmix, exec_split js _ g, exec_bind, PMF.map_bind, exec_bind,
    PMF.map_bind]
  refine bind_congr_support fun v hv => ?_
  rw [← hfactor o ho v hv, exec_buryAll hdead]

/--
Lemma 6 for a function: a function that splits on `g` after instructions
`is`, within the worlds of the splits beneath, and merges after `js` as
`exec_merge` describes, answers the law of running the rest `ks` from the
merged worlds.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `p`: The law of the result.
- `s`: The initial state.
- `src`: The returned operand.
- `is`, `js`, `ks`: The instructions before the split, between the split and
  the merge, and after the merge.
- `W`: The law of the outcomes of the splits beneath.
- `ν`: The law of states in each world of the splits beneath, at the split.
- `g`: The value of the split.
- `ℓs`: The locations to bury before the merge.
- `c`: How a survivor and everything else make a state.
- `U`: The law of the outcomes of the splits above.
- `d`: The law of the survivor in each world.
- `R`: The law of everything else in each world of the splits beneath and
  above.

# Hypotheses
- `hrun`: The function answers `p`.
- `hs`: Its initial state is `s`.
- `hsrc`: It returns `src`.
- `his`: Its instructions are `is`, then `js`, then `ks`.
- `hworlds`, `hdead`, `hfactor`: As for `exec_merge`, from the initial state.
-/
theorem run_merge {ω υ : Type} {f : Function} {args : List Int}
    {externals : List (String × Int)} {p : PMF Int} {s : Xdy.State}
    {src : AddressingMode} {is js ks : List Instruction} {W : PMF ω}
    {ν : ω → PMF State} {g : State → α} {ℓs : List Location}
    {c : β → γ → State} {U : ω → PMF υ} {d : ω → α → υ → PMF β}
    {R : ω → υ → PMF γ}
    (hrun : run f args externals = .ok p)
    (hs : f.initialState args externals = .ok s)
    (hsrc : f.instructions.back? = some (.return src))
    (his : f.instructions.toList = is ++ js ++ ks)
    (hworlds : exec is (PMF.pure (.ofOracle s)) = W.bind ν)
    (hdead : ∀ ℓ ∈ ℓs, ℓ.dead src ks)
    (hfactor : ∀ o ∈ W.support, ∀ v ∈ ((ν o).map g).support,
      (exec js (cond (ν o) g v)).map (·.buryAll ℓs) =
        (U o).bind fun u => lift c (d o v u) (R o u)) :
    p = (exec ks (W.bind fun o => (U o).bind fun u =>
      lift c (((ν o).map g).bind fun v => d o v u) (R o u))).map
        (·.value src) := by
  obtain ⟨s', src', hs', hsrc', hp⟩ := run_eq_exec hrun
  rw [hs] at hs'
  cases hs'
  rw [hsrc] at hsrc'
  cases hsrc'
  rw [hp, his]
  exact exec_merge is js ks _ W ν g src ℓs c U d R hworlds hdead hfactor

end Xdy.Spec
