import XdySpec.Conditioning

/-!
# Product laws of worlds

The vocabulary in which the plan's invariant is stated: a world of the open
splits as independent laws, one per live location, assembled over a base state
that holds everything else.

The forward pass in `xdy/src/distribution/propagation.rs` holds each live
value as its own distribution, and treats them as independent within each
world of its open splits. This module gives that picture a meaning in the
specification's joint law of states, `product`, and proves the laws by which
the pass's operations act on it:

- The order of the locations never matters (`product_perm`), and any one of
  them splits off, combined independently with the rest (`product_eq_lift`),
  which is the form in which lemma 6, `exec_merge`, takes the worlds that it
  merges.
- Conditioning on a location makes it a point mass and leaves the others
  alone (`cond_product`), and a point mass folds into the base
  (`product_pure`, `product_cons_pure`); a world of point masses is one
  (`product_eq_pure`).
- Burying a location marginalizes it out (`map_bury_product`), and burying
  locations that an instruction does not read commutes with it
  (`map_buryAll_bind_step`).
- Mixing worlds that differ in one location's law mixes that law
  (`bind_product`); read the other way, a world is the mixture of those in
  which a live location is a point mass at each of its outcomes
  (`bind_pure_product`), which is how a split leaves it.
- An instruction on independent operands leaves a world a world
  (`bind_step_product`): its destination takes the law of the instruction on
  its operands, and every other location is untouched. So it does beside live
  point masses that it reads, which stay live (`bind_step_product_pure`), and
  one that does not roll answers a point mass (`step_eq_pure`).
- The law of an instruction on its operands is, for an arithmetic instruction
  on two registers, lemma 1's `lift` (`law_binary`); with one immediate
  operand, the other operand's law, mapped (`law_binary_immediate_left`,
  `law_binary_immediate_right`); for a range of immediates, the range's law
  (`law_rollRange`); and for a sum, the law of the record's sum
  (`law_sumRollingRecord`).
- More generally, an instruction's law binds its operands' joint law, as
  `read` and `read_pair` in `propagation.rs` read them: independent, unless
  both are the same register, whose value they then share
  (`map_value_product_pair`). So a negation maps its operand's law
  (`law_neg`), arithmetic maps the operation over the joint law
  (`law_binary_operands`), and over the same register twice maps it over the
  register's law (`law_binary_self`), rather than lifting it; a roll mixes the
  roll's law over its operands' (`law_rollRange_operands`,
  `law_rollStandardDice`, `law_rollCustomDice`), and a drop is lemma 1's
  `lift` of the drop over the record's law and the count's (`law_dropLowest`,
  `law_dropHighest`). Each is the law that the visit of its instruction in
  `propagation.rs` computes.

That the plan's analysis keeps the law of states such a world, in each world
of its open splits, is the invariant that these laws serve, proved in
`XdySpec/Invariant.lean`.
-/

namespace Xdy.Spec

/-! ## Reading and writing a location -/

/--
The type of the value that a location holds.

# Parameters
- `ℓ`: The location.

# Returns
`Int` for a register, `Record` for a rolling record, and `Unit` for the
answer, which the state does not hold.
-/
abbrev Content : Location → Type
  | .register _ => Int
  | .record _ => Record
  | .answer => Unit

/--
The value that a buried location holds.

# Parameters
- `ℓ`: The location.

# Returns
`0` for a register, and an empty rolling record, as `State.bury` leaves them.
-/
def blank : (ℓ : Location) → Content ℓ
  | .register _ => (0 : Int)
  | .record _ => ({} : Record)
  | .answer => ()

/--
Read a value sorted, as the forward pass fixes a split's outcome.

# Parameters
- `ℓ`: The location.
- `x`: The value.

# Returns
A rolling record with its results sorted (`Record.sort`), and any other value
unchanged.
-/
def sortContent : (ℓ : Location) → Content ℓ → Content ℓ
  | .register _, x => x
  | .record _, x => Record.sort x
  | .answer, x => x

/--
Sorting a sorted value leaves it alone.

# Parameters
- `ℓ`: The location.
- `x`: The value.
-/
@[simp] theorem sortContent_sortContent (ℓ : Location) (x : Content ℓ) :
    sortContent ℓ (sortContent ℓ x) = sortContent ℓ x := by
  cases ℓ
  · rfl
  · exact Record.sort_sort x
  · rfl

/--
Reading a register's law sorted leaves it alone.

# Parameters
- `p`: The law.

# Hypotheses
- `h`: The location is a register.
-/
theorem map_sortContent_of_register {ℓ : Location} {r : Nat}
    (h : ℓ = .register r) (p : PMF (Content ℓ)) :
    p.map (sortContent ℓ) = p := by
  subst h
  exact PMF.map_id p

/--
A blank is sorted.

# Parameters
- `ℓ`: The location.
-/
@[simp] theorem sortContent_blank (ℓ : Location) :
    sortContent ℓ (blank ℓ) = blank ℓ := by
  cases ℓ
  · rfl
  · simp [sortContent, blank, Record.sort, Record.toOracle_empty,
      Record.ofOracle]
  · rfl

namespace State

/--
Read a location.

# Parameters
- `s`: The state.
- `ℓ`: The location.

# Returns
The value in the register, or the rolling record, as `value` and `record`
read them.
-/
def get (s : State) : (ℓ : Location) → Content ℓ
  | .register r => s.value (.register r)
  | .record r => s.record r
  | .answer => ()

/--
Write a location.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.

# Returns
The state with `x` at `ℓ`. The state holds no answer, so writing it changes
nothing.
-/
def set (s : State) : (ℓ : Location) → Content ℓ → State
  | .register r, x => s.setRegister r x
  | .record r, x => s.setRecord r x
  | .answer, _ => s

/--
Whether a state has a location: whether its register or rolling record
exists, so that writing it is not lost.

# Parameters
- `s`: The state.
- `ℓ`: The location.

# Returns
`True` if the index is in range, and always for the answer.
-/
def Has (s : State) : Location → Prop
  | .register r => r < s.registers.size
  | .record r => r < s.records.size
  | .answer => True

/--
Burying a location writes its blank.

# Parameters
- `s`: The state.
- `ℓ`: The location.
-/
theorem bury_eq_set (s : State) (ℓ : Location) :
    s.bury ℓ = s.set ℓ (blank ℓ) := by
  cases ℓ <;> rfl

/--
Reading a location that a state has reads what was last written there.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.

# Hypotheses
- `h`: The state has `ℓ`.
-/
@[simp] theorem get_set_self {s : State} {ℓ : Location} (h : s.Has ℓ)
    (x : Content ℓ) : (s.set ℓ x).get ℓ = x := by
  cases ℓ <;>
    simp_all [Has, get, set, value, record, setRegister, setRecord,
      Array.getElem_modify]

/--
Writing a location leaves every other alone.

# Parameters
- `s`: The state.
- `ℓ`: The location written.
- `ℓ'`: The location read.
- `x`: The value.

# Hypotheses
- `h`: The locations differ.
-/
theorem get_set_ne {s : State} {ℓ ℓ' : Location} (h : ℓ ≠ ℓ')
    (x : Content ℓ) : (s.set ℓ x).get ℓ' = s.get ℓ' := by
  cases ℓ <;> cases ℓ' <;>
    simp_all [get, set, value, record, setRegister, setRecord,
      Array.getElem?_modify]

/--
Writing a location twice writes it once, with the second value.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`, `y`: The values, in order.
-/
@[simp] theorem set_set (s : State) (ℓ : Location) (x y : Content ℓ) :
    (s.set ℓ x).set ℓ y = s.set ℓ y := by
  cases ℓ <;> simp only [set, setRegister, setRecord, modify_modify] <;> rfl

/--
Writing distinct locations commutes.

# Parameters
- `s`: The state.
- `ℓ`, `ℓ'`: The locations.
- `x`, `y`: Their values.

# Hypotheses
- `h`: The locations differ.
-/
theorem set_comm (s : State) {ℓ ℓ' : Location} (h : ℓ ≠ ℓ') (x : Content ℓ)
    (y : Content ℓ') : (s.set ℓ x).set ℓ' y = (s.set ℓ' y).set ℓ x := by
  cases ℓ <;> cases ℓ' <;> simp_all [set, setRegister, setRecord] <;>
    exact modify_comm _ h _ _

/--
Writing a location keeps every location that the state has.

# Parameters
- `s`: The state.
- `ℓ`: The location written.
- `x`: The value.
- `ℓ'`: The location had.
-/
@[simp] theorem has_set (s : State) (ℓ : Location) (x : Content ℓ)
    (ℓ' : Location) : (s.set ℓ x).Has ℓ' ↔ s.Has ℓ' := by
  cases ℓ <;> cases ℓ' <;> simp [Has, set, setRegister, setRecord]

/--
Burying a location keeps every location that the state has.

# Parameters
- `s`: The state.
- `ℓ`: The location buried.
- `ℓ'`: The location had.
-/
@[simp] theorem has_bury (s : State) (ℓ ℓ' : Location) :
    (s.bury ℓ).Has ℓ' ↔ s.Has ℓ' := by
  rw [bury_eq_set, has_set]

/--
Writing the answer changes nothing.

# Parameters
- `s`: The state.
- `x`: The value.
-/
@[simp] theorem set_answer (s : State) (x : Content .answer) :
    s.set .answer x = s := rfl

/--
Writing a location leaves alone every operand that does not read it.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.
- `op`: The operand.

# Hypotheses
- `h`: `op` does not read `ℓ`.
-/
theorem value_set {s : State} {ℓ : Location} {op : AddressingMode}
    (h : ℓ.isOperand op = false) (x : Content ℓ) :
    (s.set ℓ x).value op = s.value op := by
  cases ℓ <;> cases op <;>
    simp_all [set, value, setRegister, setRecord, Location.isOperand,
      Array.getElem?_modify]

/--
Writing a location leaves alone every other rolling record.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.
- `r`: The rolling record.

# Hypotheses
- `h`: `ℓ` is not the rolling record.
-/
theorem record_set {s : State} {ℓ : Location} {r : Nat} (h : ℓ ≠ .record r)
    (x : Content ℓ) : (s.set ℓ x).record r = s.record r :=
  get_set_ne (ℓ' := .record r) h x

/--
Writing a location commutes with writing another register.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: Its value.
- `d`: The register.
- `v`: Its value.

# Hypotheses
- `h`: `ℓ` is not the register.
-/
theorem set_setRegister {s : State} {ℓ : Location} {d : Nat} {v : Int}
    (h : ℓ ≠ .register d) (x : Content ℓ) :
    (s.setRegister d v).set ℓ x = (s.set ℓ x).setRegister d v :=
  set_comm s (ℓ := .register d) (Ne.symm h) v x

/--
Writing a location commutes with replacing another rolling record.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: Its value.
- `d`: The rolling record.
- `rec`: Its new contents.

# Hypotheses
- `h`: `ℓ` is not the rolling record.
-/
theorem set_setRecord {s : State} {ℓ : Location} {d : Nat} {rec : Record}
    (h : ℓ ≠ .record d) (x : Content ℓ) :
    (s.setRecord d rec).set ℓ x = (s.set ℓ x).setRecord d rec :=
  set_comm s (ℓ := .record d) (Ne.symm h) rec x

/--
Burying a location after writing it buries it.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.
-/
@[simp] theorem bury_set_self (s : State) (ℓ : Location) (x : Content ℓ) :
    (s.set ℓ x).bury ℓ = s.bury ℓ := by
  rw [bury_eq_set, bury_eq_set, set_set]

/--
Burying a location commutes with writing another.

# Parameters
- `s`: The state.
- `ℓ`: The location written.
- `ℓ'`: The location buried.
- `x`: The value.

# Hypotheses
- `h`: The locations differ.
-/
theorem bury_set_ne (s : State) {ℓ ℓ' : Location} (h : ℓ ≠ ℓ')
    (x : Content ℓ) : (s.set ℓ x).bury ℓ' = (s.bury ℓ').set ℓ x := by
  rw [bury_eq_set, bury_eq_set, set_comm _ h]

/--
Burying locations commutes.

# Parameters
- `s`: The state.
- `ℓ`, `ℓ'`: The locations.
-/
theorem bury_comm (s : State) (ℓ ℓ' : Location) :
    (s.bury ℓ).bury ℓ' = (s.bury ℓ').bury ℓ := by
  by_cases h : ℓ = ℓ'
  · subst h; rfl
  · rw [bury_eq_set s ℓ, bury_set_ne _ h, ← bury_eq_set]

/--
Burying a location twice buries it once.

# Parameters
- `s`: The state.
- `ℓ`: The location.
-/
@[simp] theorem bury_bury (s : State) (ℓ : Location) :
    (s.bury ℓ).bury ℓ = s.bury ℓ := by
  rw [bury_eq_set s ℓ, bury_set_self, bury_eq_set]

/--
Burying locations buries the first, then the rest.

# Parameters
- `s`: The state.
- `ℓ`: The first location.
- `ℓs`: The rest.
-/
theorem buryAll_cons (s : State) (ℓ : Location) (ℓs : List Location) :
    s.buryAll (ℓ :: ℓs) = (s.bury ℓ).buryAll ℓs := rfl

/--
Burying a location commutes with burying others.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `ℓs`: The others.
-/
theorem buryAll_bury (s : State) (ℓ : Location) (ℓs : List Location) :
    (s.bury ℓ).buryAll ℓs = (s.buryAll ℓs).bury ℓ := by
  induction ℓs generalizing s with
  | nil => rfl
  | cons ℓ' ℓs ih => rw [buryAll_cons, bury_comm, ih, buryAll_cons]

/--
Burying locations commutes with writing another.

# Parameters
- `s`: The state.
- `ℓ`: The location written.
- `x`: The value.
- `ℓs`: The locations buried.

# Hypotheses
- `h`: `ℓ` is not among them.
-/
theorem buryAll_set_of_not_mem {s : State} {ℓ : Location} (x : Content ℓ)
    {ℓs : List Location} (h : ℓ ∉ ℓs) :
    (s.set ℓ x).buryAll ℓs = (s.buryAll ℓs).set ℓ x := by
  induction ℓs generalizing s with
  | nil => rfl
  | cons ℓ' ℓs ih =>
    rw [List.mem_cons, not_or] at h
    rw [buryAll_cons, bury_set_ne _ h.1, ih h.2, buryAll_cons]

/--
The order in which locations are buried never matters.

# Parameters
- `s`: The state.
- `ℓs'`: The locations, reordered.

# Hypotheses
- `h`: `ℓs'` reorders `ℓs`.
-/
theorem buryAll_perm (s : State) {ℓs ℓs' : List Location}
    (h : ℓs.Perm ℓs') : s.buryAll ℓs = s.buryAll ℓs' := by
  induction h generalizing s with
  | nil => rfl
  | cons ℓ _ ih => rw [buryAll_cons, buryAll_cons, ih]
  | swap a b rest => rw [buryAll_cons, buryAll_cons, buryAll_cons,
      buryAll_cons, bury_comm]
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

/--
Writing a location forgets whether it was buried, even among others.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value.
- `ℓs`: The locations buried.
-/
theorem set_buryAll_erase (s : State) (ℓ : Location) (x : Content ℓ)
    (ℓs : List Location) :
    (s.buryAll (ℓs.erase ℓ)).set ℓ x = ((s.buryAll ℓs).bury ℓ).set ℓ x := by
  have hset : ∀ u : State, (u.bury ℓ).set ℓ x = u.set ℓ x := fun u => by
    rw [bury_eq_set, set_set]
  rw [hset]
  by_cases h : ℓ ∈ ℓs
  · rw [buryAll_perm s (List.perm_cons_erase h), buryAll_cons, buryAll_bury,
      hset]
  · rw [List.erase_of_not_mem h]

/--
Burying a location before burying others among which it is changes nothing.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `ℓs`: The locations buried.

# Hypotheses
- `h`: `ℓ` is among them.
-/
theorem buryAll_bury_of_mem (s : State) {ℓ : Location} {ℓs : List Location}
    (h : ℓ ∈ ℓs) : (s.bury ℓ).buryAll ℓs = s.buryAll ℓs := by
  rw [buryAll_perm _ (List.perm_cons_erase h), buryAll_cons, bury_bury,
    ← buryAll_cons, ← buryAll_perm _ (List.perm_cons_erase h)]

/--
Burying locations keeps every location that the state has.

# Parameters
- `s`: The state.
- `ℓs`: The locations buried.
- `ℓ`: The location had.
-/
@[simp] theorem has_buryAll (s : State) (ℓs : List Location) (ℓ : Location) :
    (s.buryAll ℓs).Has ℓ ↔ s.Has ℓ := by
  induction ℓs generalizing s with
  | nil => rfl
  | cons ℓ' ℓs ih => rw [buryAll_cons, ih, has_bury]

end State

/-! ## Instructions -/

/--
An instruction writes its destination.

# Parameters
- `i`: The instruction.
-/
@[simp] theorem writtenBy_destination (i : Instruction) :
    i.destination.writtenBy i = true := by
  cases i <;> simp [Instruction.destination, Location.writtenBy]

/--
An instruction that neither reads nor writes a location runs alike whatever it
holds, and leaves it alone. Generalizes `step_bury`.

# Parameters
- `s`: The state.
- `ℓ`: The location.
- `x`: The value written there.
- `i`: The instruction.

# Hypotheses
- `hr`: `i` does not read `ℓ`.
- `hw`: `i` does not write `ℓ`.
-/
theorem step_set {s : State} {ℓ : Location} (x : Content ℓ) {i : Instruction}
    (hr : ℓ.readBy i = false) (hw : ℓ.writtenBy i = false) :
    step (s.set ℓ x) i = (step s i).map (·.set ℓ x) := by
  cases i <;> cases ℓ <;>
    simp_all [Location.readBy, Location.writtenBy, step, State.value_set,
      State.record_set, State.set_setRegister, State.set_setRecord,
      PMF.map_comp, PMF.pure_map, Function.comp_def]

/--
An instruction writes nothing but its destination: every state that it can
reach is the state before it with its destination written.

# Parameters
- `s`: The state before the instruction.
- `i`: The instruction.
- `t`: A state after it.

# Hypotheses
- `h`: `t` is possible after `i`.
-/
theorem exists_set_of_mem_support_step {s : State} {i : Instruction}
    {t : State} (h : t ∈ (step s i).support) :
    ∃ x, t = s.set i.destination x := by
  cases i <;> simp only [step, PMF.support_map, PMF.support_pure,
    Set.mem_image, Set.mem_singleton_iff] at h
  all_goals first
    | (obtain ⟨x, _, rfl⟩ := h; exact ⟨x, rfl⟩)
    | (subst h; exact ⟨_, rfl⟩)
    | (subst h; exact ⟨(), rfl⟩)

/--
An instruction that does not roll answers a certain state.

# Parameters
- `s`: The state before the instruction.
- `i`: The instruction.

# Hypotheses
- `h`: `i` does not roll.
-/
theorem step_eq_pure {s : State} {i : Instruction} (h : i.rolls = false) :
    ∃ t, step s i = PMF.pure t := by
  cases i <;> simp_all [Instruction.rolls, step]

/--
Burying locations that an instruction does not read, before it runs, changes
nothing once they are buried again after it, unless it writes them.

# Parameters
- `μ`: The law of the state before the instruction.
- `B`: The locations buried before it.
- `B'`: The locations buried after it.
- `i`: The instruction.

# Hypotheses
- `hr`: `i` reads no location in `B`.
- `hw`: `i` writes every location in `B` that is not in `B'`.
-/
theorem map_buryAll_bind_step {μ : PMF State} {B B' : List Location}
    {i : Instruction} (hr : ∀ ℓ ∈ B, ℓ.readBy i = false)
    (hw : ∀ ℓ ∈ B, ℓ.writtenBy i = true ∨ ℓ ∈ B') :
    ((μ.map (·.buryAll B)).bind (step · i)).map (·.buryAll B') =
      (μ.bind (step · i)).map (·.buryAll B') := by
  rw [PMF.bind_map, PMF.map_bind, PMF.map_bind]
  refine congrArg μ.bind (funext fun s => ?_)
  show (step (s.buryAll B) i).map _ = (step s i).map _
  induction B generalizing s with
  | nil => rfl
  | cons ℓ B ih =>
    rw [State.buryAll_cons, ih (fun ℓ' h => hr ℓ' (.tail _ h))
      (fun ℓ' h => hw ℓ' (.tail _ h))]
    rcases hw ℓ (.head _) with hw' | hB
    · rw [step_bury_of_written (hr ℓ (.head _)) hw']
    · by_cases hw' : ℓ.writtenBy i
      · rw [step_bury_of_written (hr ℓ (.head _)) hw']
      · rw [step_bury (hr ℓ (.head _)) (by simpa using hw'), PMF.map_comp]
        exact congrArg (PMF.map · _)
          (funext fun t => State.buryAll_bury_of_mem t hB)

/-! ## Worlds -/

/--
Combining a law independently with a mixture mixes the combinations.

# Parameters
- `c`: How the two combine.
- `p`: The law combined.
- `μ`: The law of the mixture's component.
- `k`: The law in each component.
-/
private theorem lift_bind {α β γ δ : Type} (c : α → γ → δ) (p : PMF α)
    (μ : PMF β) (k : β → PMF γ) :
    lift c p (μ.bind k) = μ.bind fun b => lift c p (k b) := by
  simp only [lift, PMF.map_bind]
  exact PMF.bind_comm _ _ _

/--
A world: independent laws of the live locations, assembled over a base state
that holds everything else, as the pass holds the live values of each world of
its open splits.

# Parameters
- `base`: The state of every location that is not live.
- `live`: The live locations, without repetition.
- `laws`: The law of each live location.

# Returns
The law of the state that writes, over `base`, an independent draw from
`laws ℓ` at each `ℓ` in `live`. Only the laws of the live locations matter
(`product_congr`), and not their order (`product_perm`).

# Examples
```lean
example (base : State) (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base [] laws = PMF.pure base := rfl
```
-/
noncomputable def product (base : State) (live : List Location)
    (laws : (ℓ : Location) → PMF (Content ℓ)) : PMF State :=
  live.foldr (fun ℓ R => lift (fun x s => s.set ℓ x) (laws ℓ) R)
    (PMF.pure base)

variable {base : State} {live : List Location}
  {laws : (ℓ : Location) → PMF (Content ℓ)}

/--
A world with nothing live is its base.

# Parameters
- `base`: The base state.
- `laws`: The laws.
-/
@[simp] theorem product_nil (base : State)
    (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base [] laws = PMF.pure base := rfl

/--
A world combines its first live location independently with the rest.

# Parameters
- `base`: The base state.
- `ℓ`: The first live location.
- `live`: The rest.
- `laws`: The laws.
-/
theorem product_cons (base : State) (ℓ : Location) (live : List Location)
    (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base (ℓ :: live) laws =
      lift (fun x s => s.set ℓ x) (laws ℓ) (product base live laws) := rfl

/--
A world depends only on the laws of its live locations.

# Parameters
- `laws'`: Other laws.

# Hypotheses
- `h`: The laws agree on every live location.
-/
theorem product_congr {laws' : (ℓ : Location) → PMF (Content ℓ)}
    (h : ∀ ℓ ∈ live, laws ℓ = laws' ℓ) :
    product base live laws = product base live laws' := by
  induction live with
  | nil => rfl
  | cons ℓ live ih =>
    rw [product_cons, product_cons, h ℓ (.head _),
      ih fun ℓ' h' => h ℓ' (.tail _ h')]

/--
A world over live locations `T ++ S` is the world over `S`, with the world
over `T` assembled over each of its states.

# Parameters
- `base`: The base state.
- `S`, `T`: The live locations.
- `laws`: The laws.
-/
theorem product_append (base : State) (S T : List Location)
    (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base (T ++ S) laws =
      (product base S laws).bind fun s => product s T laws := by
  induction T with
  | nil => simp only [List.nil_append, product_nil, PMF.bind_pure]
  | cons ℓ T ih => rw [List.cons_append, product_cons, ih, lift_bind]; rfl

/--
Distinct live locations may swap.

# Parameters
- `a`, `b`: The locations.
- `rest`: The other live locations.

# Hypotheses
- `h`: The locations differ.
-/
private theorem product_swap {a b : Location} (rest : List Location)
    (h : a ≠ b) :
    product base (a :: b :: rest) laws =
      product base (b :: a :: rest) laws := by
  simp only [product_cons, lift, PMF.map_bind, PMF.map_comp, Function.comp_def]
  rw [PMF.bind_comm]
  simp only [State.set_comm _ h]

/--
The order of the live locations never matters.

# Parameters
- `live'`: The live locations, reordered.

# Hypotheses
- `hp`: `live'` reorders `live`.
- `hnd`: No location is live twice.
-/
theorem product_perm {live' : List Location} (hp : live.Perm live')
    (hnd : live.Nodup) : product base live laws = product base live' laws := by
  revert hnd
  induction hp with
  | nil => intro _; rfl
  | cons ℓ _ ih =>
    intro hnd
    rw [product_cons, product_cons, ih hnd.of_cons]
  | swap a b rest =>
    intro hnd
    exact product_swap rest fun h => by subst h; simp at hnd
  | trans h₁ _ ih₁ ih₂ =>
    intro hnd
    rw [ih₁ hnd, ih₂ (h₁.nodup_iff.mp hnd)]

/--
Any live location splits off a world, combined independently with the world
of the rest. This is the form in which `exec_merge` takes each world that it
merges, with `ℓ` the survivor.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
-/
theorem product_eq_lift {ℓ : Location} (hnd : live.Nodup) (h : ℓ ∈ live) :
    product base live laws =
      lift (fun x s => s.set ℓ x) (laws ℓ)
        (product base (live.erase ℓ) laws) := by
  rw [product_perm (List.perm_cons_erase h) hnd, product_cons]

/--
Every state that a world can reach agrees with the base on every location
that is not live.

# Parameters
- `t`: A state.

# Hypotheses
- `h`: `t` is possible in the world.
-/
theorem buryAll_of_mem_support_product {t : State}
    (h : t ∈ (product base live laws).support) :
    t.buryAll live = base.buryAll live := by
  induction live generalizing t with
  | nil =>
    rw [product_nil, PMF.support_pure, Set.mem_singleton_iff] at h
    rw [h]
  | cons ℓ live ih =>
    simp only [product_cons, lift, PMF.support_bind, PMF.support_map,
      Set.mem_iUnion, Set.mem_image] at h
    obtain ⟨x, _, t', ht', rfl⟩ := h
    rw [State.buryAll_cons, State.bury_set_self, State.buryAll_bury, ih ht',
      ← State.buryAll_bury, ← State.buryAll_cons]

/--
Every state that a world can reach has the locations that its base has.

# Parameters
- `t`: A state.
- `ℓ`: A location.

# Hypotheses
- `h`: `t` is possible in the world.
-/
theorem has_of_mem_support_product {t : State}
    (h : t ∈ (product base live laws).support) (ℓ : Location) :
    t.Has ℓ ↔ base.Has ℓ := by
  rw [← State.has_buryAll t live, buryAll_of_mem_support_product h,
    State.has_buryAll]

/--
Writing a location that is not live writes it in the base.

# Parameters
- `x`: The value.

# Hypotheses
- `h`: `ℓ` is not live.
-/
theorem map_set_product {ℓ : Location} (x : Content ℓ) (h : ℓ ∉ live) :
    (product base live laws).map (·.set ℓ x) =
      product (base.set ℓ x) live laws := by
  induction live with
  | nil => rw [product_nil, product_nil, PMF.pure_map]
  | cons a live ih =>
    rw [List.mem_cons, not_or] at h
    rw [product_cons, product_cons, map_lift, ← ih h.2]
    simp only [lift, PMF.map_comp, Function.comp_def,
      State.set_comm _ (Ne.symm h.1)]

/-! ## Burying -/

/--
Burying a location marginalizes it out of a world: it is no longer live, and
its base holds its blank.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `hnd`: No location is live twice.

# Notes
Burying a location that is not live buries it in the base.
-/
theorem map_bury_product {ℓ : Location} (hnd : live.Nodup) :
    (product base live laws).map (·.bury ℓ) =
      product (base.bury ℓ) (live.erase ℓ) laws := by
  have hnot : ∀ {live : List Location}, ℓ ∉ live →
      (product base live laws).map (·.bury ℓ) =
        product (base.bury ℓ) live laws := fun h => by
    simp only [State.bury_eq_set]
    exact map_set_product _ h
  by_cases h : ℓ ∈ live
  · rw [product_eq_lift hnd h, map_lift]
    simp only [State.bury_set_self]
    rw [← hnot (hnd.not_mem_erase)]
    simp only [lift, PMF.bind_const]
  · rw [List.erase_of_not_mem h]
    exact hnot h

/--
Burying locations marginalizes them out of a world, as `map_bury_product`.

# Parameters
- `ℓs`: The locations.

# Hypotheses
- `hnd`: No location is live twice.
-/
theorem map_buryAll_product (ℓs : List Location) (hnd : live.Nodup) :
    (product base live laws).map (·.buryAll ℓs) =
      product (base.buryAll ℓs) (live.diff ℓs) laws := by
  induction ℓs generalizing base live with
  | nil =>
    rw [show (fun s : State => s.buryAll []) = id from rfl, PMF.map_id]
    rfl
  | cons ℓ ℓs ih =>
    rw [show (fun s : State => s.buryAll (ℓ :: ℓs)) =
        (fun s : State => s.buryAll ℓs) ∘ (fun s : State => s.bury ℓ) from rfl,
      ← PMF.map_comp, map_bury_product hnd, ih (hnd.erase ℓ), List.diff_cons]
    rfl

/-! ## Conditioning -/

/--
Conditioning a mixture whose every component reads back its outcome picks out
a component: if each state possible in the component of an outcome reads that
outcome, then the world of an outcome is its component. Generalizes
`cond_map`, whose components are point masses.

# Parameters
- `w`: The law of the outcome.
- `k`: The law of states in the component of each outcome.
- `g`: The value.
- `v`: The world's outcome.

# Hypotheses
- `hk`: Every state possible in the component of a possible outcome reads it.
- `hv`: The outcome is possible.
-/
theorem cond_bind {σ α : Type} {w : PMF α} {k : α → PMF σ} {g : σ → α}
    (hk : ∀ x ∈ w.support, ∀ s ∈ (k x).support, g s = x) {v : α}
    (hv : v ∈ w.support) : cond (w.bind k) g v = k v := by
  -- The value reads back the outcome, so its law is the outcome's.
  have hw : (w.bind k).map g = w := by
    rw [PMF.map_bind]
    conv_rhs => rw [← PMF.bind_pure w]
    refine bind_congr_support fun x hx => ?_
    rw [map_congr_support fun s hs => hk x hx s hs]
    exact PMF.map_const _ _
  -- In any such mixture, a state weighs its outcome's probability times its
  -- probability in that outcome's component.
  have happly : ∀ k' : α → PMF σ,
      (∀ x ∈ w.support, ∀ s ∈ (k' x).support, g s = x) →
        ∀ s, (w.bind k') s = w (g s) * k' (g s) s := by
    intro k' hk' s
    rw [PMF.bind_apply, tsum_eq_single (g s) fun x hx => ?_]
    by_cases hwx : x ∈ w.support
    · have : k' x s = 0 := by
        by_contra hne
        exact hx (hk' x hwx s ((PMF.mem_support_iff _ _).mpr hne)).symm
      rw [this, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hwx
      rw [hwx, zero_mul]
  -- The worlds of the split are such components too.
  have hc : ∀ x ∈ w.support, ∀ s ∈ (cond (w.bind k) g x).support, g s = x :=
    fun x hx s hs => eq_of_mem_support_cond (by rwa [hw]) hs
  have hmix : w.bind (cond (w.bind k) g) = w.bind k := by
    have := bind_cond (w.bind k) g
    rwa [hw] at this
  ext s
  by_cases hs : g s = v
  · subst hs
    have h₁ := happly k hk s
    rw [← hmix, happly _ hc s] at h₁
    exact (ENNReal.mul_right_inj ((PMF.mem_support_iff _ _).mp hv)
      (PMF.apply_ne_top _ _)).mp h₁
  · have h₁ : cond (w.bind k) g v s = 0 := by
      by_contra hne
      exact hs (hc v hv s ((PMF.mem_support_iff _ _).mpr hne))
    have h₂ : k v s = 0 := by
      by_contra hne
      exact hs (hk v hv s ((PMF.mem_support_iff _ _).mpr hne))
    rw [h₁, h₂]

/--
Replacing the law of a live location leaves the laws of the others alone.

# Parameters
- `ℓ`: The location.
- `p`: Its new law.

# Hypotheses
- `hnd`: No location is live twice.
-/
private theorem product_update_erase {ℓ : Location} (p : PMF (Content ℓ))
    (hnd : live.Nodup) :
    product base (live.erase ℓ) (Function.update laws ℓ p) =
      product base (live.erase ℓ) laws :=
  product_congr fun _ h => Function.update_of_ne
    (fun heq => hnd.not_mem_erase (heq ▸ h)) _ _

/--
A world holds each live location at its law, if its base has the location.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
- `hh`: The base has `ℓ`.
-/
theorem map_get_product {ℓ : Location} (hnd : live.Nodup) (h : ℓ ∈ live)
    (hh : base.Has ℓ) : (product base live laws).map (·.get ℓ) = laws ℓ := by
  rw [product_eq_lift hnd h, map_lift, lift]
  conv_rhs => rw [← PMF.bind_pure (laws ℓ)]
  refine bind_congr_support fun x _ => ?_
  rw [map_congr_support fun s hs => State.get_set_self
    ((has_of_mem_support_product hs ℓ).mpr hh) x]
  exact PMF.map_const _ _

/--
Conditioning a world on a live location makes it a point mass at the world's
outcome, and leaves the laws of the others alone, as the pass fixes the value
of a split in each of its worlds.

# Parameters
- `ℓ`: The location.
- `v`: The outcome.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
- `hh`: The base has `ℓ`.
- `hv`: The outcome is possible.
-/
theorem cond_product {ℓ : Location} (hnd : live.Nodup) (h : ℓ ∈ live)
    (hh : base.Has ℓ) {v : Content ℓ} (hv : v ∈ (laws ℓ).support) :
    cond (product base live laws) (·.get ℓ) v =
      product base live (Function.update laws ℓ (PMF.pure v)) := by
  have hk : ∀ x ∈ (laws ℓ).support, ∀ s ∈ ((product base (live.erase ℓ)
      laws).map (·.set ℓ x)).support, s.get ℓ = x := by
    intro x _ s hs
    rw [PMF.support_map] at hs
    obtain ⟨t, ht, rfl⟩ := hs
    exact State.get_set_self ((has_of_mem_support_product ht ℓ).mpr hh) x
  rw [product_eq_lift hnd h, lift, cond_bind hk hv, product_eq_lift hnd h,
    Function.update_self, product_update_erase _ hnd, lift, PMF.pure_bind]

/--
A live location that is a point mass folds into the base.

# Parameters
- `ℓ`: The location.
- `x`: The value that it certainly holds.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
-/
theorem product_pure {ℓ : Location} (x : Content ℓ) (hnd : live.Nodup)
    (h : ℓ ∈ live) :
    product base live (Function.update laws ℓ (PMF.pure x)) =
      product (base.set ℓ x) (live.erase ℓ) laws := by
  rw [product_eq_lift hnd h, Function.update_self, product_update_erase _ hnd,
    lift, PMF.pure_bind, map_set_product x hnd.not_mem_erase]

/--
A live location that is a point mass folds into the base, as `product_pure`,
from the front of the live locations.

# Parameters
- `ℓ`: The location.
- `R`: The other live locations.
- `x`: The value that it certainly holds.

# Hypotheses
- `hx`: `ℓ` certainly holds `x`.
- `hℓ`: `ℓ` is not among the others.
-/
theorem product_cons_pure {ℓ : Location} {R : List Location} {x : Content ℓ}
    (hx : laws ℓ = PMF.pure x) (hℓ : ℓ ∉ R) :
    product base (ℓ :: R) laws = product (base.set ℓ x) R laws := by
  rw [product_cons, hx, lift, PMF.pure_bind, map_set_product x hℓ]

/--
A world whose every live location is a point mass is a point mass.

# Hypotheses
- `h`: Every live location certainly holds some value.
-/
theorem product_eq_pure (h : ∀ ℓ ∈ live, ∃ x, laws ℓ = PMF.pure x) :
    ∃ t, product base live laws = PMF.pure t := by
  induction live with
  | nil => exact ⟨base, rfl⟩
  | cons ℓ live ih =>
    obtain ⟨x, hx⟩ := h ℓ (.head _)
    obtain ⟨t, ht⟩ := ih fun ℓ' h' => h ℓ' (.tail _ h')
    exact ⟨t.set ℓ x, by rw [product_cons, hx, ht, lift, PMF.pure_bind,
      PMF.pure_map]⟩

/-! ## Mixing -/

/--
Mixing worlds that differ only in the law of one live location mixes that
law: the mixture is the world whose location has the mixture of its laws, and
whose other locations keep their common laws. Lifts `merge` to worlds; read
the other way, with a point mass at each outcome, it undoes a split.

# Parameters
- `ℓ`: The location.
- `w`: The law of the outcome, which weighs each world.
- `laws`: The laws in the world of each outcome.
- `rest`: The common laws of the other live locations.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
- `hrest`: In every possible world, every other live location has its common
  law.
-/
theorem bind_product {ω : Type} {ℓ : Location} (hnd : live.Nodup)
    (h : ℓ ∈ live) (w : PMF ω) (laws : ω → (ℓ : Location) → PMF (Content ℓ))
    (rest : (ℓ : Location) → PMF (Content ℓ))
    (hrest : ∀ o ∈ w.support, ∀ ℓ' ∈ live, ℓ' ≠ ℓ → laws o ℓ' = rest ℓ') :
    (w.bind fun o => product base live (laws o)) =
      product base live
        (Function.update rest ℓ (w.bind fun o => laws o ℓ)) := by
  have hworld : ∀ o ∈ w.support, product base live (laws o) =
      lift (fun x s => s.set ℓ x) (laws o ℓ)
        (product base (live.erase ℓ) rest) := by
    intro o ho
    rw [product_eq_lift hnd h, product_congr fun ℓ' h' => hrest o ho ℓ'
      (List.mem_of_mem_erase h') fun heq => hnd.not_mem_erase (heq ▸ h')]
  rw [bind_congr_support hworld, merge, product_eq_lift hnd h,
    Function.update_self, product_update_erase _ hnd]

/--
A world is the mixture, over the outcomes of any live location, of the worlds
in which that location is a point mass at its outcome: `bind_product` read the
other way, which is how a split leaves the law of states alone.

# Parameters
- `ℓ`: The location.

# Hypotheses
- `hnd`: No location is live twice.
- `h`: `ℓ` is live.
-/
theorem bind_pure_product {ℓ : Location} (hnd : live.Nodup) (h : ℓ ∈ live) :
    ((laws ℓ).bind fun x =>
        product base live (Function.update laws ℓ (PMF.pure x))) =
      product base live laws := by
  rw [bind_product hnd h (laws ℓ) (fun x => Function.update laws ℓ (PMF.pure x))
    laws fun _ _ ℓ' _ hne => Function.update_of_ne hne _ _]
  simp only [Function.update_self, PMF.bind_pure, Function.update_eq_self]

/-! ## Instructions on independent operands -/

/--
Burying locations that are not live buries them in the base.

# Parameters
- `ℓs`: The locations.

# Hypotheses
- `h`: None of them is live.
-/
theorem map_buryAll_product_of_disjoint {ℓs : List Location}
    (h : ∀ ℓ ∈ ℓs, ℓ ∉ live) :
    (product base live laws).map (·.buryAll ℓs) =
      product (base.buryAll ℓs) live laws := by
  induction ℓs generalizing base with
  | nil =>
    rw [show (fun s : State => s.buryAll []) = id from rfl, PMF.map_id]
    rfl
  | cons ℓ ℓs ih =>
    rw [show (fun s : State => s.buryAll (ℓ :: ℓs)) =
        (fun s : State => s.buryAll ℓs) ∘ (fun s : State => s.bury ℓ) from rfl,
      ← PMF.map_comp]
    simp only [State.bury_eq_set]
    rw [map_set_product _ (h ℓ (.head _)), ← State.bury_eq_set,
      ih fun ℓ' h' => h ℓ' (.tail _ h')]
    rfl

/--
An instruction runs beside the live locations that it neither reads nor
writes: running it on a world runs it on the base, and leaves them
independent of what it writes.

# Parameters
- `s`: The base state.
- `T`: The live locations.
- `i`: The instruction.

# Hypotheses
- `hr`: `i` reads no location in `T`.
- `hw`: `i` writes no location in `T`.
-/
theorem bind_step_product_frame {s : State} {T : List Location}
    {i : Instruction} (hr : ∀ ℓ ∈ T, ℓ.readBy i = false)
    (hw : ∀ ℓ ∈ T, ℓ.writtenBy i = false) :
    (product s T laws).bind (step · i) =
      (step s i).bind fun t => product t T laws := by
  induction T with
  | nil =>
    simp only [product_nil, PMF.pure_bind]
    exact (PMF.bind_pure _).symm
  | cons ℓ T ih =>
    have hstep : ∀ x, ((product s T laws).map (·.set ℓ x)).bind (step · i) =
        ((step s i).bind fun t => product t T laws).map (·.set ℓ x) := by
      intro x
      rw [PMF.bind_map, ← ih (fun ℓ' h => hr ℓ' (.tail _ h))
        (fun ℓ' h => hw ℓ' (.tail _ h)), PMF.map_bind]
      exact congrArg _ (funext fun t =>
        step_set x (hr ℓ (.head _)) (hw ℓ (.head _)))
    simp only [product_cons, lift, PMF.bind_bind, hstep, PMF.map_bind]
    exact PMF.bind_comm _ _ _

/--
An instruction on independent operands leaves a world a world. Split the live
locations into `S`, which holds the random operands that the instruction
reads, and `T`, which it neither reads nor writes. Once it runs, and every
location in `S` but its destination is buried, its destination is live, with
the law of the instruction run on the world of `S`, independent of `T`, whose
laws are untouched. Generalizes lemma 1 to states: see `law_binary`.

# Parameters
- `S`: The live locations that the instruction may read.
- `T`: The rest.
- `i`: The instruction.

# Hypotheses
- `hnd`: No location is live twice.
- `hr`: `i` reads no location in `T`.
- `hw`: `i` writes no location in `T`.
- `hd`: The base has the destination of `i`.

# Notes
The operands of `i` that are not in `S` are read from the base, and so are
fixed. The destination may be in `S`, as for a drop, which reads the record
that it writes; its old law is then forgotten.
-/
theorem bind_step_product {S T : List Location} {i : Instruction}
    (hnd : (S ++ T).Nodup) (hr : ∀ ℓ ∈ T, ℓ.readBy i = false)
    (hw : ∀ ℓ ∈ T, ℓ.writtenBy i = false) (hd : base.Has i.destination) :
    ((product base (S ++ T) laws).bind (step · i)).map
        (·.buryAll (S.erase i.destination)) =
      product (base.buryAll (S.erase i.destination)) (i.destination :: T)
        (Function.update laws i.destination
          (((product base S laws).bind (step · i)).map
            (·.get i.destination))) := by
  obtain ⟨hS, _, hST⟩ := List.nodup_append.mp hnd
  have hdT : i.destination ∉ T := fun h => by
    have := hw _ h
    rw [writtenBy_destination] at this
    cases this
  have hBT : ∀ ℓ ∈ S.erase i.destination, ℓ ∉ T :=
    fun ℓ h h' => hST ℓ (List.mem_of_mem_erase h) ℓ h' rfl
  -- Run the instruction on the world of `S`, beside that of `T`, and bury
  -- what it read there.
  rw [product_perm List.perm_append_comm hnd, product_append base S T,
    PMF.bind_bind]
  simp only [bind_step_product_frame hr hw]
  rw [← PMF.bind_bind, PMF.map_bind]
  simp only [map_buryAll_product_of_disjoint hBT]
  -- What remains of the world of `S` varies only at the destination.
  have hcollapse : ((product base S laws).bind (step · i)).map
      (·.buryAll (S.erase i.destination)) =
      (((product base S laws).bind (step · i)).map (·.get i.destination)).map
        ((base.buryAll (S.erase i.destination)).set i.destination) := by
    rw [PMF.map_comp]
    refine map_congr_support fun t ht => ?_
    rw [PMF.mem_support_bind_iff] at ht
    obtain ⟨s, hs, ht⟩ := ht
    obtain ⟨x, rfl⟩ := exists_set_of_mem_support_step ht
    simp only [Function.comp,
      State.get_set_self ((has_of_mem_support_product hs _).mpr hd),
      State.buryAll_set_of_not_mem x hS.not_mem_erase,
      State.set_buryAll_erase, buryAll_of_mem_support_product hs]
  -- That is a world of the destination alone, beneath the world of `T`.
  have hT : ∀ (t : State) (p : PMF (Content i.destination)),
      product t T (Function.update laws i.destination p) =
        product t T laws := fun t p => product_congr fun ℓ h =>
    Function.update_of_ne (fun heq : ℓ = i.destination => hdT (heq ▸ h)) _ _
  rw [product_perm (List.perm_append_singleton _ _).symm
      (List.nodup_cons.mpr ⟨hdT, (List.nodup_append.mp hnd).2.1⟩),
    product_append _ [i.destination] T]
  simp only [hT]
  rw [product_cons, product_nil, Function.update_self, lift]
  simp only [PMF.pure_map]
  rw [← Function.comp_def PMF.pure, PMF.bind_pure_comp, ← hcollapse,
    PMF.bind_map]
  rfl

/--
An instruction on independent operands leaves a world a world, even if it also
reads live locations that are point masses, which stay live. As
`bind_step_product`, with the live locations split further: `P`, which the
instruction may read, but which are point masses; `S`, which it may read, and
which are buried once it runs; and `T`, which it neither reads nor writes.

# Parameters
- `P`: The live point masses that the instruction may read.
- `S`: The other live locations that it may read.
- `T`: The rest.
- `i`: The instruction.

# Hypotheses
- `hnd`: No location is live twice.
- `hP`: Every location in `P` certainly holds some value.
- `hdP`: `i` does not write a location in `P`.
- `hr`: `i` reads no location in `T`.
- `hw`: `i` writes no location in `T`.
- `hd`: The base has the destination of `i`.
-/
theorem bind_step_product_pure {P S T : List Location} {i : Instruction}
    (hnd : (P ++ S ++ T).Nodup) (hP : ∀ ℓ ∈ P, ∃ x, laws ℓ = PMF.pure x)
    (hdP : i.destination ∉ P) (hr : ∀ ℓ ∈ T, ℓ.readBy i = false)
    (hw : ∀ ℓ ∈ T, ℓ.writtenBy i = false) (hd : base.Has i.destination) :
    ((product base (P ++ S ++ T) laws).bind (step · i)).map
        (·.buryAll (S.erase i.destination)) =
      product (base.buryAll (S.erase i.destination)) (P ++ i.destination :: T)
        (Function.update laws i.destination
          (((product base (P ++ S) laws).bind (step · i)).map
            (·.get i.destination))) := by
  induction P generalizing base with
  | nil => exact bind_step_product hnd hr hw hd
  | cons ℓ P ih =>
    obtain ⟨x, hx⟩ := hP ℓ (.head _)
    rw [List.cons_append, List.cons_append, List.nodup_cons] at hnd
    have hℓS : ℓ ∉ S := fun h => hnd.1 (List.mem_append_left _
      (List.mem_append_right _ h))
    have hℓd : ℓ ≠ i.destination := fun h => hdP (h ▸ .head _)
    have hℓT : ℓ ∉ T := fun h => hnd.1 (List.mem_append_right _ h)
    have hℓP : ℓ ∉ P := fun h => hnd.1 (List.mem_append_left _
      (List.mem_append_left _ h))
    -- Fold the point mass into the base, run the instruction there, and
    -- unfold it again.
    rw [List.cons_append, List.cons_append, product_cons_pure hx hnd.1,
      ih hnd.2 (fun ℓ' h => hP ℓ' (.tail _ h)) (fun h => hdP (.tail _ h))
        ((State.has_set _ _ _ _).mpr hd),
      List.cons_append, product_cons_pure (base := base) hx
        (fun h => hnd.1 (by simp_all)),
      product_cons_pure (x := x) (by rw [Function.update_of_ne hℓd, hx])
        (by simp [hℓP, hℓd, hℓT]),
      State.buryAll_set_of_not_mem x
        (fun h => hℓS (List.mem_of_mem_erase h))]

/--
Lemma 1, on states: an arithmetic instruction on two registers that are live
and distinct in a world writes the lift of its operation over their laws, as
`pairwise` in `propagation.rs` combines them. With `bind_step_product`, the
world after the instruction has its destination at this law.

# Parameters
- `op`: The operation.
- `d`: The destination register.
- `a`, `b`: The operand registers.

# Hypotheses
- `hab`: The operands differ.
- `hd`, `ha`, `hb`: The base has the destination and the operands.
-/
theorem law_binary {op : BinaryOp} {d a b : Nat} (hab : a ≠ b)
    (hd : base.Has (.register d)) (ha : base.Has (.register a))
    (hb : base.Has (.register b)) :
    ((product base [.register a, .register b] laws).bind
        (step · (.binary op d (.register a) (.register b)))).map
          (·.get (.register d)) =
      lift op.apply (laws (.register a)) (laws (.register b)) := by
  have hne : Location.register b ≠ .register a := fun h => by
    cases h
    exact hab rfl
  -- Each state of the world has the lift's outcome in the destination.
  have hstate : ∀ (x : Int) (y : Int),
      (step ((base.set (.register b) y).set (.register a) x)
        (.binary op d (.register a) (.register b))).map
          (·.get (.register d)) = PMF.pure (op.apply x y) := by
    intro x y
    have hx : ((base.set (.register b) y).set (.register a) x).get
        (.register a) = x := State.get_set_self (by simpa using ha) _
    have hy : ((base.set (.register b) y).set (.register a) x).get
        (.register b) = y := by
      rw [State.get_set_ne hne.symm, State.get_set_self hb]
    simp only [step, PMF.pure_map]
    exact congrArg PMF.pure ((State.get_set_self (ℓ := .register d)
      (by simpa using hd) _).trans (congrArg₂ op.apply hx hy))
  simp only [product_cons, product_nil, lift, PMF.pure_map, PMF.map_bind,
    PMF.bind_bind]
  refine bind_congr_support fun x _ => ?_
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun y _ => ?_
  rw [PMF.pure_bind]
  exact hstate x y

/--
A range of immediates, which reads no location, writes its law to its rolling
record.

# Parameters
- `d`: The rolling record.
- `a`, `b`: The endpoints.

# Hypotheses
- `hd`: The base has the rolling record.
-/
theorem law_rollRange {d : Nat} {a b : Int} (hd : base.Has (.record d)) :
    ((product base [] laws).bind
        (step · (.rollRange d (.immediate a) (.immediate b)))).map
          (·.get (.record d)) = rollRange a b := by
  simp only [product_nil, PMF.pure_bind, step, State.value, PMF.map_comp]
  conv_rhs => rw [← PMF.map_id (rollRange a b)]
  exact congrArg (PMF.map · _) (funext fun x => State.get_set_self hd x)

/--
A sum of a rolling record that is live in a world writes the law of its sum to
its register.

# Parameters
- `d`: The register.
- `r`: The rolling record.

# Hypotheses
- `hd`: The base has the register.
- `hr`: The base has the rolling record.
-/
theorem law_sumRollingRecord {d r : Nat} (hd : base.Has (.register d))
    (hr : base.Has (.record r)) :
    ((product base [.record r] laws).bind (step · (.sumRollingRecord d r))).map
        (·.get (.register d)) = (laws (.record r)).map Record.sum := by
  simp only [product_cons, product_nil, lift, PMF.pure_map, PMF.map_bind,
    PMF.bind_bind, PMF.pure_bind, step]
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun x _ => ?_
  have hx : (base.set (.record r) x).record r = x := State.get_set_self hr x
  simp only [Function.comp_apply]
  exact congrArg PMF.pure ((State.get_set_self (ℓ := .register d)
    (by simpa using hd) _).trans (congrArg Record.sum hx))

/--
An arithmetic instruction on a register that is live in a world, and an
immediate on its right, writes the register's law mapped by the operation.

# Parameters
- `op`: The operation.
- `d`: The destination register.
- `a`: The operand register.
- `c`: The immediate.

# Hypotheses
- `hd`, `ha`: The base has the destination and the operand.
-/
theorem law_binary_immediate_right {op : BinaryOp} {d a : Nat} {c : Int}
    (hd : base.Has (.register d)) (ha : base.Has (.register a)) :
    ((product base [.register a] laws).bind
        (step · (.binary op d (.register a) (.immediate c)))).map
          (·.get (.register d)) = (laws (.register a)).map (op.apply · c) := by
  simp only [product_cons, product_nil, lift, PMF.pure_map, PMF.map_bind,
    PMF.bind_bind, PMF.pure_bind, step]
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun x _ => ?_
  have hx : (base.set (.register a) x).get (.register a) = x :=
    State.get_set_self ha x
  simp only [Function.comp_apply]
  exact congrArg PMF.pure ((State.get_set_self (ℓ := .register d)
    (by simpa using hd) _).trans (congrArg (op.apply · c) hx))

/--
An arithmetic instruction on an immediate, and a register on its right that is
live in a world, writes the register's law mapped by the operation.

# Parameters
- `op`: The operation.
- `d`: The destination register.
- `c`: The immediate.
- `b`: The operand register.

# Hypotheses
- `hd`, `hb`: The base has the destination and the operand.
-/
theorem law_binary_immediate_left {op : BinaryOp} {d b : Nat} {c : Int}
    (hd : base.Has (.register d)) (hb : base.Has (.register b)) :
    ((product base [.register b] laws).bind
        (step · (.binary op d (.immediate c) (.register b)))).map
          (·.get (.register d)) = (laws (.register b)).map (op.apply c) := by
  simp only [product_cons, product_nil, lift, PMF.pure_map, PMF.map_bind,
    PMF.bind_bind, PMF.pure_bind, step]
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun x _ => ?_
  have hx : (base.set (.register b) x).get (.register b) = x :=
    State.get_set_self hb x
  simp only [Function.comp_apply]
  exact congrArg PMF.pure ((State.get_set_self (ℓ := .register d)
    (by simpa using hd) _).trans (congrArg (op.apply c) hx))

/-! ## Operands -/

/--
A world of one live location writes a draw from its law over the base.

# Parameters
- `base`: The base state.
- `ℓ`: The live location.
- `laws`: The laws.
-/
theorem product_singleton (base : State) (ℓ : Location)
    (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base [ℓ] laws = (laws ℓ).map (base.set ℓ) := by
  simp only [product_cons, product_nil, lift, PMF.pure_map]
  exact PMF.bind_pure_comp _ _

/--
The live locations that an operand reads: its register, if it names one.

# Parameters
- `op`: The operand.

# Returns
The register, or nothing for an immediate, and for a rolling record, which
`State.value` reads as `0`.

# Examples
```lean
#guard operandLocations (.register 3) == [.register 3]
#guard operandLocations (.immediate 3) == []
```
-/
def operandLocations : AddressingMode → List Location
  | .register r => [.register r]
  | _ => []

/--
The law of an operand in a world, as `read` in `propagation.rs` reads it.

# Parameters
- `laws`: The laws of the live locations.
- `op`: The operand.

# Returns
A point mass at an immediate, the law of a register, or a point mass at `0`
for a rolling record, as `State.value` reads one.

# Examples
```lean
example (laws : (ℓ : Location) → PMF (Content ℓ)) :
    operandLaw laws (.immediate 3) = PMF.pure 3 := rfl
```
-/
noncomputable def operandLaw (laws : (ℓ : Location) → PMF (Content ℓ)) :
    AddressingMode → PMF Int
  | .immediate c => PMF.pure c
  | .register r => laws (.register r)
  | .rollingRecord _ => PMF.pure 0

/--
The live locations that two operands read, each once.

# Parameters
- `a`, `b`: The operands.

# Returns
Their registers, once if they name the same one.

# Examples
```lean
#guard pairLocations (.register 1) (.register 1) == [.register 1]
#guard pairLocations (.register 1) (.register 2) == [.register 1, .register 2]
#guard pairLocations (.immediate 1) (.register 2) == [.register 2]
```
-/
def pairLocations : AddressingMode → AddressingMode → List Location
  | .register a, .register b =>
    if a = b then [.register a] else [.register a, .register b]
  | a, b => operandLocations a ++ operandLocations b

/--
The joint law of two operands in a world, as `read_pair` in `propagation.rs`
reads them: independent, unless they are the same register, whose value they
then share.

# Parameters
- `laws`: The laws of the live locations.
- `a`, `b`: The operands.

# Returns
The law of the pair of their values: the register's law, paired with itself,
if they name the same register, and otherwise lemma 1's `lift` of their laws.

# Examples
```lean
example (laws : (ℓ : Location) → PMF (Content ℓ)) :
    pairLaw laws (.register 1) (.register 1) =
      (laws (.register 1)).map fun x => (x, x) := by
  simp [pairLaw]
```
-/
noncomputable def pairLaw (laws : (ℓ : Location) → PMF (Content ℓ)) :
    AddressingMode → AddressingMode → PMF (Int × Int)
  | .register a, .register b =>
    if a = b then (laws (.register a)).map fun x => (x, x)
    else lift Prod.mk (laws (.register a)) (laws (.register b))
  | a, b => lift Prod.mk (operandLaw laws a) (operandLaw laws b)

/--
An operand's value in a world has its law. Mirrors `read` in
`propagation.rs`.

# Parameters
- `a`: The operand.

# Hypotheses
- `ha`: The base has its register, if any.
-/
theorem map_value_product_operand {a : AddressingMode}
    (ha : ∀ ℓ ∈ operandLocations a, base.Has ℓ) :
    (product base (operandLocations a) laws).map (·.value a) =
      operandLaw laws a := by
  cases a with
  | immediate c =>
    simp [operandLocations, operandLaw, State.value, PMF.pure_map]
  | rollingRecord r =>
    simp [operandLocations, operandLaw, State.value, PMF.pure_map]
  | register r =>
    simp only [operandLocations, product_singleton, PMF.map_comp, operandLaw]
    conv_rhs => rw [← PMF.map_id (laws _)]
    exact congrArg (PMF.map · _) (funext fun x =>
      State.get_set_self (ℓ := .register r) (ha _ (.head _)) x)

/--
Two operands' values in a world have their joint law. Mirrors `read_pair` in
`propagation.rs`.

# Parameters
- `a`, `b`: The operands.

# Hypotheses
- `hab`: The base has their registers.
-/
theorem map_value_product_pair {a b : AddressingMode}
    (hab : ∀ ℓ ∈ pairLocations a b, base.Has ℓ) :
    (product base (pairLocations a b) laws).map
        (fun s => (s.value a, s.value b)) = pairLaw laws a b := by
  have hget : ∀ r, base.Has (.register r) → ∀ x : Int,
      (base.set (.register r) x).value (.register r) = x :=
    fun r h x => State.get_set_self (ℓ := .register r) h x
  have himm : ∀ (s : State) c, s.value (.immediate c) = c := fun _ _ => rfl
  have hrec : ∀ (s : State) r, s.value (.rollingRecord r) = 0 :=
    fun _ _ => rfl
  cases a <;> cases b
  case register.register a b =>
    by_cases h : a = b
    · subst h
      simp only [pairLocations, pairLaw, ↓reduceIte, product_singleton,
        PMF.map_comp] at hab ⊢
      exact congrArg (PMF.map · _) (funext fun x => by
        simp only [Function.comp_apply, hget a (hab _ (.head _))])
    · have hne : Location.register a ≠ .register b := fun h' => by
        cases h'
        exact h rfl
      simp only [pairLocations, pairLaw, h, ↓reduceIte] at hab ⊢
      rw [product_cons, product_singleton]
      simp only [lift, PMF.map_bind, PMF.map_comp]
      refine congrArg _ (funext fun x => congrArg (PMF.map · _) (funext
        fun y => ?_))
      have hx := State.get_set_self (s := base.set (.register b) y)
        (ℓ := .register a) ((State.has_set _ _ _ _).mpr (hab _ (.head _))) x
      have hy : ((base.set (.register b) y).set (.register a) x).get
          (.register b) = y := by
        rw [State.get_set_ne hne,
          State.get_set_self (hab _ (.tail _ (.head _)))]
      exact Prod.ext hx hy
  all_goals simp only [pairLocations, operandLocations, pairLaw, operandLaw,
    List.nil_append, List.append_nil, product_nil, product_singleton,
    PMF.pure_map, PMF.map_comp, lift, PMF.pure_bind, himm, hrec] at hab ⊢
  all_goals first
    | rfl
    | exact congrArg (PMF.map · _) (funext fun x => by
        simp only [Function.comp_apply, hget _ (hab _ (.head _))])

/--
A rolling record and an operand, live in a world, are independent: their
joint law is lemma 1's `lift` of their laws.

# Parameters
- `d`: The rolling record.
- `c`: The operand.

# Hypotheses
- `hd`: The base has the rolling record.
- `hc`: The base has the operand's register, if any.
-/
theorem map_record_value_product {d : Nat} {c : AddressingMode}
    (hd : base.Has (.record d)) (hc : ∀ ℓ ∈ operandLocations c, base.Has ℓ) :
    (product base (.record d :: operandLocations c) laws).map
        (fun s => (s.record d, s.value c)) =
      lift Prod.mk (laws (.record d)) (operandLaw laws c) := by
  rw [product_cons, ← map_value_product_operand hc]
  simp only [lift, PMF.map_bind, PMF.map_comp]
  refine congrArg _ (funext fun r => map_congr_support fun s hs => ?_)
  have hsd : s.Has (.record d) := (has_of_mem_support_product hs _).mpr hd
  exact Prod.ext (State.get_set_self hsd r)
    (State.value_set (by cases c <;> rfl) r)

/--
An instruction whose destination depends only on one operand's value writes,
in a world of that operand, its law bound to the instruction's.

# Parameters
- `i`: The instruction.
- `a`: The operand.
- `k`: The law of the destination at each value of the operand.

# Hypotheses
- `hk`: In every state that has the destination, `i` writes it with law
  `k` of the operand's value.
- `hd`: The base has the destination.
- `ha`: The base has the operand's register, if any.
-/
theorem law_of_operand {i : Instruction} {a : AddressingMode}
    {k : Int → PMF (Content i.destination)}
    (hk : ∀ s : State, s.Has i.destination →
      (step s i).map (·.get i.destination) = k (s.value a))
    (hd : base.Has i.destination)
    (ha : ∀ ℓ ∈ operandLocations a, base.Has ℓ) :
    ((product base (operandLocations a) laws).bind (step · i)).map
        (·.get i.destination) = (operandLaw laws a).bind k := by
  rw [PMF.map_bind, ← map_value_product_operand ha, PMF.bind_map]
  exact bind_congr_support fun s hs =>
    hk s ((has_of_mem_support_product hs _).mpr hd)

/--
An instruction whose destination depends only on two operands' values writes,
in a world of those operands, their joint law bound to the instruction's.

# Parameters
- `i`: The instruction.
- `a`, `b`: The operands.
- `k`: The law of the destination at each pair of values of the operands.

# Hypotheses
- `hk`: In every state that has the destination, `i` writes it with law
  `k` of the operands' values.
- `hd`: The base has the destination.
- `hab`: The base has the operands' registers.
-/
theorem law_of_pair {i : Instruction} {a b : AddressingMode}
    {k : Int → Int → PMF (Content i.destination)}
    (hk : ∀ s : State, s.Has i.destination →
      (step s i).map (·.get i.destination) = k (s.value a) (s.value b))
    (hd : base.Has i.destination)
    (hab : ∀ ℓ ∈ pairLocations a b, base.Has ℓ) :
    ((product base (pairLocations a b) laws).bind (step · i)).map
        (·.get i.destination) =
      (pairLaw laws a b).bind fun p => k p.1 p.2 := by
  rw [PMF.map_bind, ← map_value_product_pair hab, PMF.bind_map]
  exact bind_congr_support fun s hs =>
    hk s ((has_of_mem_support_product hs _).mpr hd)

/--
An instruction whose destination depends only on a rolling record and an
operand writes, in a world of the two, their joint law bound to the
instruction's.

# Parameters
- `i`: The instruction.
- `d`: The rolling record.
- `c`: The operand.
- `k`: The law of the destination at each record and value of the operand.

# Hypotheses
- `hk`: In every state that has the destination, `i` writes it with law
  `k` of the record and the operand's value.
- `hi`: The base has the destination.
- `hd`: The base has the rolling record.
- `hc`: The base has the operand's register, if any.
-/
theorem law_of_record_operand {i : Instruction} {d : Nat}
    {c : AddressingMode} {k : Record → Int → PMF (Content i.destination)}
    (hk : ∀ s : State, s.Has i.destination →
      (step s i).map (·.get i.destination) = k (s.record d) (s.value c))
    (hi : base.Has i.destination) (hd : base.Has (.record d))
    (hc : ∀ ℓ ∈ operandLocations c, base.Has ℓ) :
    ((product base (.record d :: operandLocations c) laws).bind
        (step · i)).map (·.get i.destination) =
      (lift Prod.mk (laws (.record d)) (operandLaw laws c)).bind
        fun p => k p.1 p.2 := by
  rw [PMF.map_bind, ← map_record_value_product hd hc, PMF.bind_map]
  exact bind_congr_support fun s hs =>
    hk s ((has_of_mem_support_product hs _).mpr hi)

/-! ## The law of each instruction -/

/--
A negation writes its operand's law, mapped, as `visit_neg` in
`propagation.rs` computes it with `map`.

# Parameters
- `d`: The destination register.
- `a`: The operand.

# Hypotheses
- `hd`: The base has the destination.
- `ha`: The base has the operand's register, if any.
-/
theorem law_neg {d : Nat} {a : AddressingMode} (hd : base.Has (.register d))
    (ha : ∀ ℓ ∈ operandLocations a, base.Has ℓ) :
    ((product base (operandLocations a) laws).bind (step · (.neg d a))).map
        (·.get (.register d)) = (operandLaw laws a).map neg := by
  rw [← PMF.bind_pure_comp]
  exact law_of_operand (i := .neg d a) (k := fun x => PMF.pure (neg x))
    (fun s hs => by
      simp only [step, PMF.pure_map]
      exact congrArg PMF.pure (State.get_set_self (ℓ := .register d) hs _))
    hd ha

/--
An arithmetic instruction writes its operation mapped over its operands'
joint law, as `binary` in `propagation.rs` computes it: lemma 1's `lift` for
distinct operands (`law_binary`), which `pairwise` and `saturating_sum`
compute, and a mapped law for the same register twice (`law_binary_self`) or
an immediate (`law_binary_immediate_left`, `law_binary_immediate_right`),
which `map` computes.

# Parameters
- `op`: The operation.
- `d`: The destination register.
- `a`, `b`: The operands.

# Hypotheses
- `hd`: The base has the destination.
- `hab`: The base has the operands' registers.
-/
theorem law_binary_operands {op : BinaryOp} {d : Nat} {a b : AddressingMode}
    (hd : base.Has (.register d))
    (hab : ∀ ℓ ∈ pairLocations a b, base.Has ℓ) :
    ((product base (pairLocations a b) laws).bind
        (step · (.binary op d a b))).map (·.get (.register d)) =
      (pairLaw laws a b).map fun p => op.apply p.1 p.2 := by
  rw [← PMF.bind_pure_comp]
  exact law_of_pair (i := .binary op d a b)
    (k := fun x y => PMF.pure (op.apply x y)) (fun s hs => by
      simp only [step, PMF.pure_map]
      exact congrArg PMF.pure (State.get_set_self (ℓ := .register d) hs _))
    hd hab

/--
An arithmetic instruction on the same register twice, as the strength reducer
writes `x + x` for `x * 2`, writes the register's law mapped by the operation
on each value paired with itself, not lemma 1's `lift`, which would pair it
with an independent copy. Mirrors `binary` in `propagation.rs`, which reads
the register once and computes this with `map`.

# Parameters
- `op`: The operation.
- `d`: The destination register.
- `a`: The operand register.

# Hypotheses
- `hd`, `ha`: The base has the destination and the operand.
-/
theorem law_binary_self {op : BinaryOp} {d a : Nat}
    (hd : base.Has (.register d)) (ha : base.Has (.register a)) :
    ((product base [.register a] laws).bind
        (step · (.binary op d (.register a) (.register a)))).map
          (·.get (.register d)) =
      (laws (.register a)).map fun x => op.apply x x := by
  have h := law_binary_operands (laws := laws) (op := op)
    (a := .register a) (b := .register a) hd
    (by simpa [pairLocations] using ha)
  simp only [pairLocations, pairLaw, ↓reduceIte, PMF.map_comp] at h
  exact h

/--
A range writes the range's law at each pair of its endpoints, mixed over
their joint law, as `visit_roll_range` in `propagation.rs` makes one setup of
each pair. Generalizes `law_rollRange` to endpoints in registers.

# Parameters
- `d`: The rolling record.
- `a`, `b`: The endpoints.

# Hypotheses
- `hd`: The base has the rolling record.
- `hab`: The base has the endpoints' registers.
-/
theorem law_rollRange_operands {d : Nat} {a b : AddressingMode}
    (hd : base.Has (.record d)) (hab : ∀ ℓ ∈ pairLocations a b, base.Has ℓ) :
    ((product base (pairLocations a b) laws).bind
        (step · (.rollRange d a b))).map (·.get (.record d)) =
      (pairLaw laws a b).bind fun p => rollRange p.1 p.2 :=
  law_of_pair (i := .rollRange d a b) (k := rollRange) (fun s hs => by
    simp only [step, PMF.map_comp]
    conv_rhs => rw [← PMF.map_id (rollRange _ _)]
    exact congrArg (PMF.map · _) (funext fun x => State.get_set_self hs x))
    hd hab

/--
Standard dice write the law of the dice at each pair of count and faces,
mixed over their joint law, as `visit_roll_standard_dice` in `propagation.rs`
makes one setup of each pair.

# Parameters
- `d`: The rolling record.
- `c`: The count.
- `n`: The faces.

# Hypotheses
- `hd`: The base has the rolling record.
- `hcn`: The base has the count's and the faces' registers.
-/
theorem law_rollStandardDice {d : Nat} {c n : AddressingMode}
    (hd : base.Has (.record d)) (hcn : ∀ ℓ ∈ pairLocations c n, base.Has ℓ) :
    ((product base (pairLocations c n) laws).bind
        (step · (.rollStandardDice d c n))).map (·.get (.record d)) =
      (pairLaw laws c n).bind fun p => rollDice p.1 (standardDie p.2) :=
  law_of_pair (i := .rollStandardDice d c n)
    (k := fun x y => rollDice x (standardDie y)) (fun s hs => by
      simp only [step, PMF.map_comp]
      conv_rhs => rw [← PMF.map_id (rollDice _ _)]
      exact congrArg (PMF.map · _) (funext fun x => State.get_set_self hs x))
    hd hcn

/--
Custom dice write the law of the dice at each count, mixed over its law, as
`visit_roll_custom_dice` in `propagation.rs` makes one setup of each count.

# Parameters
- `d`: The rolling record.
- `c`: The count.
- `faces`: The faces of each die.

# Hypotheses
- `hd`: The base has the rolling record.
- `hc`: The base has the count's register, if any.
-/
theorem law_rollCustomDice {d : Nat} {c : AddressingMode} {faces : List Int}
    (hd : base.Has (.record d)) (hc : ∀ ℓ ∈ operandLocations c, base.Has ℓ) :
    ((product base (operandLocations c) laws).bind
        (step · (.rollCustomDice d c faces))).map (·.get (.record d)) =
      (operandLaw laws c).bind fun x => rollDice x (customDie faces) :=
  law_of_operand (i := .rollCustomDice d c faces)
    (k := fun x => rollDice x (customDie faces)) (fun s hs => by
      simp only [step, PMF.map_comp]
      conv_rhs => rw [← PMF.map_id (rollDice _ _)]
      exact congrArg (PMF.map · _) (funext fun x => State.get_set_self hs x))
    hd hc

/--
Dropping the lowest results writes lemma 1's `lift` of the drop over the
record's law and the count's, as `visit_drop_lowest` in `propagation.rs`
drops every setup of the record by every count.

# Parameters
- `d`: The rolling record.
- `c`: The count.

# Hypotheses
- `hd`: The base has the rolling record.
- `hc`: The base has the count's register, if any.
-/
theorem law_dropLowest {d : Nat} {c : AddressingMode}
    (hd : base.Has (.record d)) (hc : ∀ ℓ ∈ operandLocations c, base.Has ℓ) :
    ((product base (.record d :: operandLocations c) laws).bind
        (step · (.dropLowest d c))).map (·.get (.record d)) =
      lift Record.dropLowest (laws (.record d)) (operandLaw laws c) := by
  refine (law_of_record_operand (i := .dropLowest d c)
    (k := fun r x => PMF.pure (r.dropLowest x)) (fun s hs => by
      simp only [step, PMF.pure_map]
      exact congrArg PMF.pure (State.get_set_self hs _)) hd hd hc).trans ?_
  simp only [lift, PMF.bind_bind, PMF.bind_map, Function.comp_def]
  exact congrArg _ (funext fun r => PMF.bind_pure_comp _ _)

/--
Dropping the highest results writes lemma 1's `lift` of the drop over the
record's law and the count's, as `visit_drop_highest` in `propagation.rs`
drops every setup of the record by every count.

# Parameters
- `d`: The rolling record.
- `c`: The count.

# Hypotheses
- `hd`: The base has the rolling record.
- `hc`: The base has the count's register, if any.
-/
theorem law_dropHighest {d : Nat} {c : AddressingMode}
    (hd : base.Has (.record d)) (hc : ∀ ℓ ∈ operandLocations c, base.Has ℓ) :
    ((product base (.record d :: operandLocations c) laws).bind
        (step · (.dropHighest d c))).map (·.get (.record d)) =
      lift Record.dropHighest (laws (.record d)) (operandLaw laws c) := by
  refine (law_of_record_operand (i := .dropHighest d c)
    (k := fun r x => PMF.pure (r.dropHighest x)) (fun s hs => by
      simp only [step, PMF.pure_map]
      exact congrArg PMF.pure (State.get_set_self hs _)) hd hd hc).trans ?_
  simp only [lift, PMF.bind_bind, PMF.bind_map, Function.comp_def]
  exact congrArg _ (funext fun r => PMF.bind_pure_comp _ _)

end Xdy.Spec
