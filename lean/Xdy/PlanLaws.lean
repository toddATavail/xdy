import Xdy.Plan

/-!
# Laws of fan-out plans

The structural facts about a plan on which the proof that it preserves the law
of a function rests. They are pure combinatorics over `Xdy/Plan.lean`, with no
probability, so they need only core Lean.

The proof keeps, in each world of the open splits, the law of the live values
as independent laws, one per location. An instruction keeps that form if every
random value that it reads is either fixed in the world, as an open split, or
read for the last time, so that it may be buried once the instruction has
combined it. So:

- A value that an instruction reads is live just before it, and occupies its
  location; a live value occupies its location, and an instruction that
  overwrites a live value reads it.
- A live value that fans out is an open split (`mem_openSplits_of_live`), so
  every random value that an instruction reads is an open split, or is read
  for the last time (`mem_openSplits_of_random`).
- Each merge meets the survivor's conditions against the stack as it stands:
  no split above it depends on it, and no live value but the survivor does.
- A value is live exactly when its location is not dead before the rest of the
  function (`dead_eq`), which is how lemma 6 asks which locations it may bury.

The dependence of each value on the splits follows `dependsOn`, which grows
along the values that an instruction reads, and holds only earlier values that
fan out.
-/

namespace Xdy

/-! ## Operands -/

/--
Whether an operand is a value: an immediate or a register, not a rolling
record, as `Function.validate` requires.

# Parameters
- `op`: The operand.

# Returns
`false` for a rolling record, `true` otherwise.
-/
def AddressingMode.isValue : AddressingMode → Bool
  | .rollingRecord _ => false
  | _ => true

/--
Whether every operand of an instruction is a value, as `Function.validate`
requires. Where they are, `Location.readBy` and `Instruction.sources` agree.

# Parameters
- `i`: The instruction.

# Returns
`true` if no operand is a rolling record.
-/
def Instruction.operandsAreValues : Instruction → Bool
  | .rollRange _ a b | .rollStandardDice _ a b | .binary _ _ a b =>
    a.isValue && b.isValue
  | .rollCustomDice _ c _ | .dropLowest _ c | .dropHighest _ c | .neg _ c
  | .return c => c.isValue
  | .sumRollingRecord .. => true

/--
An operand that is a value reads a location exactly when it names it.

# Parameters
- `ℓ`: The location.
- `op`: The operand.

# Hypotheses
- `h`: `op` is a value.
-/
theorem Location.isOperand_iff {ℓ : Location} {op : AddressingMode}
    (h : op.isValue = true) :
    ℓ.isOperand op = true ↔ Location.ofOperand op = some ℓ := by
  cases op <;> cases ℓ <;>
    simp_all [Location.isOperand, Location.ofOperand,
      AddressingMode.isValue] <;>
    exact eq_comm

/--
An instruction whose operands are values reads a location exactly when it is
one of its sources.

# Parameters
- `ℓ`: The location.
- `i`: The instruction.

# Hypotheses
- `h`: Every operand of `i` is a value.
-/
theorem Location.readBy_iff {ℓ : Location} {i : Instruction}
    (h : i.operandsAreValues = true) : ℓ.readBy i = true ↔ ℓ ∈ i.sources := by
  cases i <;>
    simp_all [Location.readBy, Instruction.sources,
      Instruction.operandsAreValues, Location.isOperand_iff]

/--
An instruction writes a location exactly when it is its destination.

# Parameters
- `ℓ`: The location.
- `i`: The instruction.
-/
theorem Location.writtenBy_iff {ℓ : Location} {i : Instruction} :
    ℓ.writtenBy i = true ↔ ℓ = i.destination := by
  cases i <;> simp [Location.writtenBy, Instruction.destination]

/--
A location is not dead before instructions that end by returning an operand
exactly when one of them reads it before any writes it.

# Parameters
- `ℓ`: The location.
- `src`: The returned operand.
- `l`: The instructions.

# Hypotheses
- `h`: The last instruction returns `src`.
-/
theorem Location.dead_eq_false_iff {ℓ : Location} {src : AddressingMode} :
    ∀ {l : List Instruction}, l.getLast? = some (.return src) →
      (ℓ.dead src l = false ↔ ∃ j, ∃ hj : j < l.length,
        ℓ.readBy l[j] = true ∧ ∀ i (hi : i < j), ℓ.writtenBy l[i] = false)
  | [], h => by simp at h
  | [i], h => by
    simp only [List.getLast?_singleton, Option.some.injEq] at h
    subst h
    constructor
    · intro hd
      refine ⟨0, Nat.zero_lt_one, ?_, fun _ hi => absurd hi (Nat.not_lt_zero _)⟩
      cases ho : ℓ.isOperand src <;>
        simp_all [Location.dead, Location.readBy]
    · rintro ⟨j, hj, hr, _⟩
      have : j = 0 := by simp at hj; omega
      subst this
      simp_all [Location.dead, Location.readBy]
  | i :: i' :: l, h => by
    have ih := Location.dead_eq_false_iff (ℓ := ℓ) (l := i' :: l)
      (by simpa [List.getLast?_cons_cons] using h)
    simp only [Location.dead, Bool.and_eq_false_iff, Bool.not_eq_false',
      Bool.or_eq_false_iff] at ih ⊢
    rw [ih]
    constructor
    · rintro (hr | ⟨hw, j, hj, hr, hb⟩)
      · exact ⟨0, by simp, hr, fun _ hi => absurd hi (Nat.not_lt_zero _)⟩
      · refine ⟨j + 1, by simp at hj ⊢; omega, hr, fun k hk => ?_⟩
        cases k with
        | zero => exact hw
        | succ k => exact hb k (by omega)
    · rintro ⟨j, hj, hr, hb⟩
      cases j with
      | zero => exact .inl hr
      | succ j =>
        cases hri : ℓ.readBy i
        · exact .inr ⟨hb 0 (by omega), j, by simp at hj ⊢; omega, hr,
            fun k hk => hb (k + 1) (by omega)⟩
        · exact .inl rfl

namespace Values

variable (vs : Values)

/-! ## Writers -/

/--
No value precedes the first instruction.

# Parameters
- `ℓ`: The location.
-/
@[simp] theorem writer_zero (ℓ : Location) : vs.writer 0 ℓ = none := rfl

/--
The value in a location just after an instruction is the instruction's, if it
writes the location, and otherwise the value just before it.

# Parameters
- `pc`: The program counter of the instruction.
- `ℓ`: The location.
-/
theorem writer_succ (pc : Nat) (ℓ : Location) :
    vs.writer (pc + 1) ℓ =
      if vs.location pc = ℓ then some pc else vs.writer pc ℓ := by
  by_cases h : vs.location pc = ℓ <;>
    simp [writer, List.range_succ, h]

/--
The value in a location before an instruction is the last write there: it was
written earlier, to that location, and nothing written since was written
there.

# Parameters
- `pc`: The program counter of the instruction.
- `ℓ`: The location.
- `v`: The value.
-/
theorem writer_eq_some_iff {pc : Nat} {ℓ : Location} {v : Nat} :
    vs.writer pc ℓ = some v ↔
      v < pc ∧ vs.location v = ℓ ∧
        ∀ w, v < w → w < pc → vs.location w ≠ ℓ := by
  induction pc with
  | zero => simp
  | succ pc ih =>
    rw [writer_succ]
    by_cases h : vs.location pc = ℓ
    · simp only [h, ↓reduceIte]
      constructor
      · rintro ⟨⟩
        exact ⟨Nat.lt_succ_self _, h, fun w h₁ h₂ => by omega⟩
      · rintro ⟨h₁, _, h₃⟩
        by_cases hv : v = pc
        · rw [hv]
        · exact absurd h (h₃ pc (by omega) (Nat.lt_succ_self _))
    · simp only [h, ↓reduceIte, ih]
      constructor
      · rintro ⟨h₁, h₂, h₃⟩
        refine ⟨by omega, h₂, fun w hw₁ hw₂ => ?_⟩
        by_cases hw : w = pc
        · rw [hw]; exact h
        · exact h₃ w hw₁ (by omega)
      · rintro ⟨h₁, h₂, h₃⟩
        have : v ≠ pc := fun hv => h (hv ▸ h₂)
        exact ⟨by omega, h₂, fun w hw₁ hw₂ => h₃ w hw₁ (by omega)⟩

/--
The value in a location is at that location.

# Parameters
- `pc`: The program counter.
- `ℓ`: The location.
- `v`: The value.

# Hypotheses
- `h`: `v` is in `ℓ` before `pc`.
-/
theorem location_of_writer {pc : Nat} {ℓ : Location} {v : Nat}
    (h : vs.writer pc ℓ = some v) : vs.location v = ℓ :=
  ((vs.writer_eq_some_iff).mp h).2.1

/--
A value stays in its location until an instruction writes there: if nothing
from `pc` up to `pc'` writes the location, the value in it before `pc'` is the
value before `pc`.

# Parameters
- `pc`, `pc'`: The program counters, in order.
- `ℓ`: The location.
- `v`: The value.

# Hypotheses
- `h`: `v` is in `ℓ` before `pc`.
- `hle`: `pc` is at most `pc'`.
- `hw`: No instruction from `pc` up to `pc'` writes `ℓ`.
-/
theorem writer_of_le {pc pc' : Nat} {ℓ : Location} {v : Nat}
    (h : vs.writer pc ℓ = some v) (hle : pc ≤ pc')
    (hw : ∀ w, pc ≤ w → w < pc' → vs.location w ≠ ℓ) :
    vs.writer pc' ℓ = some v := by
  obtain ⟨h₁, h₂, h₃⟩ := (vs.writer_eq_some_iff).mp h
  refine (vs.writer_eq_some_iff).mpr ⟨by omega, h₂, fun w hw₁ hw₂ => ?_⟩
  by_cases hw' : w < pc
  · exact h₃ w hw₁ hw'
  · exact hw w (by omega) hw₂

/--
A value is in its location from when it is written until an instruction
writes there again.

# Parameters
- `pc`, `pc'`: The program counters, in order.
- `ℓ`: The location.
- `v`: The value.

# Hypotheses
- `h`: `v` is in `ℓ` before `pc'`.
- `hv`: `v` was written before `pc`.
- `hle`: `pc` is at most `pc'`.
-/
theorem writer_of_ge {pc pc' : Nat} {ℓ : Location} {v : Nat}
    (h : vs.writer pc' ℓ = some v) (hv : v < pc) (hle : pc ≤ pc') :
    vs.writer pc ℓ = some v := by
  obtain ⟨_, h₂, h₃⟩ := (vs.writer_eq_some_iff).mp h
  exact (vs.writer_eq_some_iff).mpr
    ⟨hv, h₂, fun w hw₁ hw₂ => h₃ w hw₁ (by omega)⟩

/-! ## Reads and readers -/

/--
An instruction's sources, or none past the end of the function.

# Parameters
- `pc`: The program counter.

# Returns
The sources of the instruction at `pc`, or `[]` if there is none.
-/
def sourcesAt (pc : Nat) : List Location :=
  (vs.instructions[pc]?.map Instruction.sources).getD []

/--
No instruction reads the answer.

# Parameters
- `i`: The instruction.
-/
theorem answer_not_mem_sources (i : Instruction) :
    Location.answer ∉ i.sources := by
  have hop : ∀ op : AddressingMode, Location.ofOperand op ≠ some .answer :=
    fun op => by cases op <;> simp [Location.ofOperand]
  cases i <;> simp [Instruction.sources, List.mem_filterMap, hop]

/--
An instruction reads a value exactly when the value is in one of its sources.

# Parameters
- `pc`: The program counter of the instruction.
- `u`: The value.
-/
theorem mem_reads {pc u : Nat} :
    u ∈ vs.reads pc ↔ ∃ ℓ ∈ vs.sourcesAt pc, vs.writer pc ℓ = some u := by
  simp [reads, sourcesAt, List.mem_filterMap]

/--
A value that an instruction reads is not the answer, which no instruction
reads.

# Parameters
- `pc`: The program counter of the instruction.
- `u`: The value.

# Hypotheses
- `h`: The instruction reads `u`.
-/
theorem location_ne_answer_of_mem_reads {pc u : Nat} (h : u ∈ vs.reads pc) :
    vs.location u ≠ .answer := by
  obtain ⟨ℓ, hℓ, hw⟩ := (vs.mem_reads).mp h
  rw [vs.location_of_writer hw]
  rintro rfl
  simp only [sourcesAt] at hℓ
  cases hi : vs.instructions[pc]? with
  | none => simp [hi] at hℓ
  | some i =>
    rw [hi] at hℓ
    exact answer_not_mem_sources i hℓ

/--
Only an instruction of the function reads a value.

# Parameters
- `pc`: The program counter.
- `u`: The value.

# Hypotheses
- `h`: The instruction at `pc` reads `u`.
-/
theorem lt_length_of_mem_reads {pc u : Nat} (h : u ∈ vs.reads pc) :
    pc < vs.instructions.length := by
  obtain ⟨ℓ, hℓ, _⟩ := (vs.mem_reads).mp h
  refine Nat.lt_of_not_le fun hn => ?_
  simp [sourcesAt, List.getElem?_eq_none hn] at hℓ

/--
The readers of a value are the instructions that read it.

# Parameters
- `v`: The value.
- `pc`: The program counter.
-/
theorem mem_readers {v pc : Nat} : pc ∈ vs.readers v ↔ v ∈ vs.reads pc := by
  simp only [readers, List.mem_filter, List.mem_range, List.contains_iff_mem]
  exact ⟨fun h => h.2, fun h => ⟨vs.lt_length_of_mem_reads h, h⟩⟩

/--
A value that an instruction reads is live before it.

# Parameters
- `pc`, `pc'`: The program counters.
- `u`: The value.

# Hypotheses
- `h`: The instruction at `pc` reads `u`.
- `hlt`: `pc'` precedes `pc`.
-/
theorem isLiveAfter_of_mem_reads {pc pc' u : Nat} (h : u ∈ vs.reads pc)
    (hlt : pc' < pc) : vs.isLiveAfter u pc' := by
  simp only [isLiveAfter, Bool.or_eq_true, List.any_eq_true, decide_eq_true_eq]
  exact .inr ⟨pc, (vs.mem_readers).mpr h, hlt⟩

/--
A value that an instruction reads is in its location before the instruction.

# Parameters
- `pc`: The program counter of the instruction.
- `u`: The value.

# Hypotheses
- `h`: The instruction reads `u`.
-/
theorem writer_of_mem_reads {pc u : Nat} (h : u ∈ vs.reads pc) :
    vs.writer pc (vs.location u) = some u := by
  obtain ⟨ℓ, _, hw⟩ := (vs.mem_reads).mp h
  rwa [vs.location_of_writer hw]

/--
An instruction whose operands are values reads the value in every location
that it reads.

# Parameters
- `pc`: The program counter of the instruction.
- `ℓ`: The location.
- `u`: The value in it before the instruction.

# Hypotheses
- `hvals`: The instruction's operands are values.
- `hpc`: `pc` is an instruction of the function.
- `hr`: The instruction reads `ℓ`.
- `hw`: `u` is in `ℓ` before the instruction.
-/
theorem mem_reads_of_readBy {pc u : Nat} {ℓ : Location}
    (hpc : pc < vs.instructions.length)
    (hvals : vs.instructions[pc].operandsAreValues = true)
    (hr : ℓ.readBy vs.instructions[pc] = true)
    (hw : vs.writer pc ℓ = some u) : u ∈ vs.reads pc :=
  (vs.mem_reads).mpr ⟨ℓ, by
    simpa [sourcesAt, List.getElem?_eq_getElem hpc] using
      (Location.readBy_iff hvals).mp hr, hw⟩

/-! ## Live values -/

/--
The live values after an instruction are those written at or before it that
are live after it.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.
-/
theorem mem_live {pc v : Nat} :
    v ∈ vs.live pc ↔ v ≤ pc ∧ vs.isLiveAfter v pc := by
  simp [live, List.mem_filter, Nat.lt_succ_iff]

/--
A value is live after an instruction only if it is live after every earlier
one.

# Parameters
- `v`: The value.
- `pc`, `pc'`: The program counters, in order.

# Hypotheses
- `h`: `v` is live after `pc`.
- `hle`: `pc'` is at most `pc`.
-/
theorem isLiveAfter_of_le {v pc pc' : Nat} (h : vs.isLiveAfter v pc)
    (hle : pc' ≤ pc) : vs.isLiveAfter v pc' := by
  simp only [isLiveAfter, Bool.or_eq_true, List.any_eq_true,
    decide_eq_true_eq] at h ⊢
  rcases h with h | ⟨r, hr, hlt⟩
  · exact .inl h
  · exact .inr ⟨r, hr, by omega⟩

/--
A live value occupies its location: no instruction since it was written has
written there.

# Parameters
- `v`: The value.
- `pc`: The program counter.

# Hypotheses
- `hv`: `v` was written at or before `pc`.
- `hpc`: `pc` is an instruction of the function.
- `h`: `v` is live after `pc`.
-/
theorem writer_of_isLiveAfter {v pc : Nat} (hv : v ≤ pc)
    (hpc : pc < vs.instructions.length) (h : vs.isLiveAfter v pc) :
    vs.writer (pc + 1) (vs.location v) = some v := by
  simp only [isLiveAfter, lasting, Bool.or_eq_true, beq_iff_eq,
    List.any_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨r, hr, hlt⟩
  · rw [vs.location_of_writer h]
    exact vs.writer_of_ge h (by omega) (by omega)
  · exact vs.writer_of_ge (vs.writer_of_mem_reads ((vs.mem_readers).mp hr))
      (by omega) hlt

/--
An instruction that overwrites a live value reads it: the value's last reader
must precede the write, and it is live after the instruction before.

# Parameters
- `pc`: The program counter of the instruction before.
- `v`: The value.

# Hypotheses
- `hpc`: The instruction after `pc` is in the function.
- `hw`: `v` is in the location of the instruction after `pc`, before it.
- `h`: `v` is live after `pc`.
-/
theorem mem_reads_of_overwritten {pc v : Nat}
    (hpc : pc + 1 < vs.instructions.length)
    (hw : vs.writer (pc + 1) (vs.location (pc + 1)) = some v)
    (h : vs.isLiveAfter v pc) : v ∈ vs.reads (pc + 1) := by
  obtain ⟨hv, hℓ, _⟩ := (vs.writer_eq_some_iff).mp hw
  simp only [isLiveAfter, lasting, Bool.or_eq_true, beq_iff_eq,
    List.any_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨r, hr, hlt⟩
  · -- The answer lasts only if nothing after it writes the answer.
    obtain ⟨_, h₂, h₃⟩ := (vs.writer_eq_some_iff).mp h
    exact absurd (hℓ.symm.trans h₂) (h₃ (pc + 1) hv hpc)
  · have hu := (vs.mem_readers).mp hr
    by_cases hr' : r = pc + 1
    · rwa [hr'] at hu
    · obtain ⟨_, _, h₃⟩ := (vs.writer_eq_some_iff).mp
        (vs.writer_of_mem_reads hu)
      exact absurd hℓ.symm (h₃ (pc + 1) hv (by omega))

/-! ## Live values before an instruction -/

/--
The values live just before an instruction.

# Parameters
- `pc`: The program counter of the instruction.

# Returns
None before the first instruction, and otherwise the values live after the
instruction before, in ascending order.
-/
def liveBefore : Nat → List Nat
  | 0 => []
  | pc + 1 => vs.live pc

/--
No value is live before the first instruction.
-/
@[simp] theorem liveBefore_zero : vs.liveBefore 0 = [] := rfl

/--
The values live before an instruction are those live after the one before.

# Parameters
- `pc`: The program counter of the instruction before.
-/
theorem liveBefore_succ (pc : Nat) : vs.liveBefore (pc + 1) = vs.live pc :=
  rfl

/--
The values live before an instruction are those written before it that it or
a later instruction reads, or that last until the end.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.
-/
theorem mem_liveBefore {pc v : Nat} :
    v ∈ vs.liveBefore pc ↔
      v < pc ∧ (vs.lasting v ∨ ∃ r ∈ vs.readers v, pc ≤ r) := by
  cases pc with
  | zero => simp
  | succ pc =>
    simp only [liveBefore_succ, mem_live, isLiveAfter, Bool.or_eq_true,
      List.any_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨hv, h | ⟨r, hr, hlt⟩⟩
      · exact ⟨by omega, .inl h⟩
      · exact ⟨by omega, .inr ⟨r, hr, hlt⟩⟩
    · rintro ⟨hv, h | ⟨r, hr, hle⟩⟩
      · exact ⟨by omega, .inl h⟩
      · exact ⟨by omega, .inr ⟨r, hr, hle⟩⟩

/--
No value is live twice before an instruction.

# Parameters
- `pc`: The program counter of the instruction.
-/
theorem nodup_liveBefore (pc : Nat) : (vs.liveBefore pc).Nodup := by
  cases pc with
  | zero => exact List.nodup_nil
  | succ pc => exact List.nodup_range.filter _

/--
A value live before an instruction occupies its location.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.

# Hypotheses
- `hpc`: `pc` is at most the number of instructions.
- `h`: `v` is live before `pc`.
-/
theorem writer_of_mem_liveBefore {pc v : Nat}
    (hpc : pc ≤ vs.instructions.length) (h : v ∈ vs.liveBefore pc) :
    vs.writer pc (vs.location v) = some v := by
  cases pc with
  | zero => simp at h
  | succ pc =>
    obtain ⟨hv, hl⟩ := (vs.mem_live).mp h
    exact vs.writer_of_isLiveAfter hv (by omega) hl

/--
The values live before an instruction occupy distinct locations.

# Parameters
- `pc`: The program counter of the instruction.

# Hypotheses
- `hpc`: `pc` is at most the number of instructions.
-/
theorem nodup_map_location_liveBefore {pc : Nat}
    (hpc : pc ≤ vs.instructions.length) :
    ((vs.liveBefore pc).map vs.location).Nodup :=
  List.pairwise_map.mpr ((vs.nodup_liveBefore pc).imp_of_mem
    fun hv hw hne he => by
      have := vs.writer_of_mem_liveBefore hpc hv
      rw [he, vs.writer_of_mem_liveBefore hpc hw] at this
      exact hne (Option.some.inj this).symm)

/--
Every value that an instruction reads is live before it.

# Parameters
- `pc`: The program counter of the instruction.
- `u`: The value.

# Hypotheses
- `h`: The instruction reads `u`.
-/
theorem mem_liveBefore_of_mem_reads {pc u : Nat} (h : u ∈ vs.reads pc) :
    u ∈ vs.liveBefore pc :=
  (vs.mem_liveBefore).mpr ⟨vs.lt_of_mem_reads h,
    .inr ⟨pc, (vs.mem_readers).mpr h, Nat.le_refl _⟩⟩

/--
A value live before an instruction that does not read it is live after it.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.

# Hypotheses
- `h`: `v` is live before `pc`.
- `hr`: The instruction does not read `v`.
-/
theorem mem_live_of_mem_liveBefore {pc v : Nat} (h : v ∈ vs.liveBefore pc)
    (hr : v ∉ vs.reads pc) : v ∈ vs.live pc := by
  obtain ⟨hv, hl⟩ := (vs.mem_liveBefore).mp h
  refine (vs.mem_live).mpr ⟨Nat.le_of_lt hv, ?_⟩
  simp only [isLiveAfter, Bool.or_eq_true, List.any_eq_true,
    decide_eq_true_eq]
  rcases hl with hl | ⟨r, hr', hle⟩
  · exact .inl hl
  · refine .inr ⟨r, hr', Nat.lt_of_le_of_ne hle fun he => hr ?_⟩
    subst he
    exact (vs.mem_readers).mp hr'

/--
A value live after an instruction, other than its own, was live before it.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.

# Hypotheses
- `h`: `v` is live after `pc`.
- `hv`: `v` is not the instruction's value.
-/
theorem mem_liveBefore_of_mem_live {pc v : Nat} (h : v ∈ vs.live pc)
    (hv : v ≠ pc) : v ∈ vs.liveBefore pc := by
  obtain ⟨hle, hl⟩ := (vs.mem_live).mp h
  simp only [isLiveAfter, Bool.or_eq_true, List.any_eq_true,
    decide_eq_true_eq] at hl
  refine (vs.mem_liveBefore).mpr ⟨by omega, ?_⟩
  rcases hl with hl | ⟨r, hr, hlt⟩
  · exact .inl hl
  · exact .inr ⟨r, hr, Nat.le_of_lt hlt⟩

/--
A value that an instruction writes is live after it if it fans out, since a
later instruction reads it.

# Parameters
- `pc`: The program counter of the instruction.

# Hypotheses
- `hf`: The instruction's value fans out.
-/
theorem mem_live_self {pc : Nat} (hf : vs.fansOut pc) : pc ∈ vs.live pc := by
  simp only [fansOut, Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨r, hr⟩ :=
    List.exists_mem_of_length_pos (l := vs.readers pc) (by omega)
  refine (vs.mem_live).mpr ⟨Nat.le_refl _, ?_⟩
  simp only [isLiveAfter, Bool.or_eq_true, List.any_eq_true,
    decide_eq_true_eq]
  exact .inr ⟨r, hr, vs.lt_of_mem_reads ((vs.mem_readers).mp hr)⟩

/--
A value live after an instruction, other than its own, is not in the location
that the instruction writes, which then holds the instruction's value.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The value.

# Hypotheses
- `hpc`: `pc` is an instruction of the function.
- `h`: `v` is live after `pc`.
- `hv`: `v` is not the instruction's value.
-/
theorem location_ne_of_mem_live {pc v : Nat}
    (hpc : pc < vs.instructions.length) (h : v ∈ vs.live pc) (hv : v ≠ pc) :
    vs.location v ≠ vs.location pc := fun he => by
  have hw := vs.writer_of_mem_liveBefore (pc := pc + 1) hpc h
  rw [he, writer_succ] at hw
  simp only [ite_true, Option.some.injEq] at hw
  exact hv hw.symm

/-! ## Randomness and dependence -/

/--
A value may be random exactly when its instruction rolls or it reads a value
that may be random.

# Parameters
- `v`: The value.
-/
theorem random_iff {v : Nat} :
    vs.random v ↔ vs.instructions[v]?.any Instruction.rolls ∨
      ∃ u ∈ vs.reads v, vs.random u := by
  rw [random]
  simp

/--
A value that reads a value that may be random may be random.

# Parameters
- `u`: The value read.
- `v`: The value that reads it.

# Hypotheses
- `hu`: `v`'s instruction reads `u`.
- `h`: `u` may be random.
-/
theorem random_of_mem_reads {u v : Nat} (hu : u ∈ vs.reads v)
    (h : vs.random u) : vs.random v :=
  (vs.random_iff).mpr (.inr ⟨u, hu, h⟩)

/--
A fixed value neither rolls nor reads a value that may be random.

# Parameters
- `v`: The value.

# Hypotheses
- `h`: `v` is fixed.
-/
theorem not_random {v : Nat} (h : vs.random v = false) :
    vs.instructions[v]?.any Instruction.rolls = false ∧
      ∀ u ∈ vs.reads v, vs.random u = false := by
  refine ⟨?_, fun u hu => ?_⟩
  · cases hr : vs.instructions[v]?.any Instruction.rolls
    · rfl
    · have := (vs.random_iff).mpr (.inl hr)
      rw [h] at this
      cases this
  · cases hu' : vs.random u
    · rfl
    · have := vs.random_of_mem_reads hu hu'
      rw [h] at this
      cases this

/--
A value that fans out may be random.

# Parameters
- `v`: The value.

# Hypotheses
- `h`: `v` fans out.
-/
theorem random_of_fansOut {v : Nat} (h : vs.fansOut v) : vs.random v := by
  simp only [fansOut, Bool.and_eq_true] at h
  exact h.1

/--
The values that fan out on which a value depends: those on which the values
that it reads depend, and itself if it fans out.

# Parameters
- `v`: The value.
- `w`: A value that fans out.
-/
theorem mem_dependsOn {v w : Nat} :
    w ∈ vs.dependsOn v ↔
      (∃ u ∈ vs.reads v, w ∈ vs.dependsOn u) ∨ (vs.fansOut v ∧ w = v) := by
  rw [dependsOn]
  by_cases h : vs.fansOut v <;> simp [h]

/--
A value depends on every split on which a value that it reads depends.

# Parameters
- `u`: The value read.
- `v`: The value that reads it.
- `w`: The split.

# Hypotheses
- `hu`: `v`'s instruction reads `u`.
- `hw`: `u` depends on `w`.
-/
theorem mem_dependsOn_of_mem_reads {u v w : Nat} (hu : u ∈ vs.reads v)
    (hw : w ∈ vs.dependsOn u) : w ∈ vs.dependsOn v :=
  (vs.mem_dependsOn).mpr (.inl ⟨u, hu, hw⟩)

/--
A value that fans out depends on itself.

# Parameters
- `v`: The value.

# Hypotheses
- `h`: `v` fans out.
-/
theorem self_mem_dependsOn {v : Nat} (h : vs.fansOut v) :
    v ∈ vs.dependsOn v :=
  (vs.mem_dependsOn).mpr (.inr ⟨h, rfl⟩)

/--
A value depends only on values that fan out and were written no later.

# Parameters
- `v`: The value.
- `w`: A value on which it depends.

# Hypotheses
- `h`: `v` depends on `w`.
-/
theorem fansOut_of_mem_dependsOn {v w : Nat} (h : w ∈ vs.dependsOn v) :
    vs.fansOut w ∧ w ≤ v := by
  induction v using Nat.strongRecOn with
  | ind v ih =>
    rcases (vs.mem_dependsOn).mp h with ⟨u, hu, hw⟩ | ⟨hf, rfl⟩
    · have hlt := vs.lt_of_mem_reads hu
      have := ih u hlt hw
      exact ⟨this.1, by omega⟩
    · exact ⟨hf, Nat.le_refl _⟩

/--
A value that depends on a split may be random.

# Parameters
- `v`: The value.
- `w`: The split.

# Hypotheses
- `h`: `v` depends on `w`.
-/
theorem random_of_mem_dependsOn {v w : Nat} (h : w ∈ vs.dependsOn v) :
    vs.random v := by
  induction v using Nat.strongRecOn with
  | ind v ih =>
    rcases (vs.mem_dependsOn).mp h with ⟨u, hu, hw⟩ | ⟨hf, rfl⟩
    · exact vs.random_of_mem_reads hu (ih u (vs.lt_of_mem_reads hu) hw)
    · exact vs.random_of_fansOut hf

/--
Dependence is transitive: a value depends on every split on which a split
that it depends on depends.

# Parameters
- `u`: The split on which `v` depends.
- `v`: The value.
- `w`: The split on which `u` depends.

# Hypotheses
- `hu`: `v` depends on `u`.
- `hw`: `u` depends on `w`.
-/
theorem mem_dependsOn_trans {u v w : Nat} (hu : u ∈ vs.dependsOn v)
    (hw : w ∈ vs.dependsOn u) : w ∈ vs.dependsOn v := by
  induction v using Nat.strongRecOn with
  | ind v ih =>
    rcases (vs.mem_dependsOn).mp hu with ⟨r, hr, hu⟩ | ⟨_, rfl⟩
    · exact vs.mem_dependsOn_of_mem_reads hr (ih r (vs.lt_of_mem_reads hr) hu)
    · exact hw

/-! ## Survivors -/

/--
The split at a position in range of the stack.

# Parameters
- `splits`: The stack.
- `k`: The position.

# Hypotheses
- `hk`: `k` is in range.
-/
private theorem getD_splits {splits : List Nat} {k : Nat}
    (hk : k < splits.length) : splits.getD k 0 = splits[k] := by
  simp [List.getD_eq_getElem?_getD, hk]

/--
A split may merge only if no split above it depends on it.

# Parameters
- `splits`: The stack of splits, from the bottom.
- `k`: The position of the split.
- `live`: The live values.
- `s`: The survivor's location, if any.

# Hypotheses
- `hk`: `k` is in range.
- `h`: The split may merge, into `s`.
-/
theorem not_mem_dependsOn_of_survivor {splits : List Nat} {k : Nat}
    {live : List Nat} {s : Option Location} (hk : k < splits.length)
    (h : vs.survivor splits k live = some s) :
    ∀ j ∈ splits.drop (k + 1), splits[k] ∉ vs.dependsOn j := by
  intro j hj hdep
  simp only [survivor, getD_splits hk] at h
  split at h
  · cases h
  · rename_i hn
    exact hn (List.any_eq_true.mpr ⟨j, hj, List.elem_eq_true_of_mem hdep⟩)

/--
A split that merges without a survivor has no live value that depends on it.

# Parameters
- `splits`: The stack of splits, from the bottom.
- `k`: The position of the split.
- `live`: The live values.

# Hypotheses
- `hk`: `k` is in range.
- `h`: The split may merge, without a survivor.
-/
theorem not_mem_dependsOn_of_survivor_none {splits : List Nat} {k : Nat}
    {live : List Nat} (hk : k < splits.length)
    (h : vs.survivor splits k live = some none) :
    ∀ v ∈ live, splits[k] ∉ vs.dependsOn v := by
  intro v hv hdep
  simp only [survivor, getD_splits hk] at h
  split at h
  · cases h
  · have hmem : v ∈ live.filter fun v => (vs.dependsOn v).contains splits[k] :=
      List.mem_filter.mpr ⟨hv, List.elem_eq_true_of_mem hdep⟩
    split at h
    · rename_i hf; rw [hf] at hmem; cases hmem
    · split at h
      · cases h
      · split at h <;> cases h
    · cases h

/--
A split that merges into a survivor has exactly one live value that depends on
it: the survivor, which depends on it but is not the split value, and whose
location is a register or the answer, never a rolling record.

# Parameters
- `splits`: The stack of splits, from the bottom.
- `k`: The position of the split.
- `live`: The live values.
- `ℓ`: The survivor's location.

# Hypotheses
- `hk`: `k` is in range.
- `h`: The split may merge, into `ℓ`.
-/
theorem survivor_some {splits : List Nat} {k : Nat} {live : List Nat}
    {ℓ : Location} (hk : k < splits.length)
    (h : vs.survivor splits k live = some (some ℓ)) :
    ∃ w ∈ live, splits[k] ∈ vs.dependsOn w ∧ w ≠ splits[k] ∧
      vs.location w = ℓ ∧ (∀ r, ℓ ≠ .record r) ∧
        ∀ v ∈ live, splits[k] ∈ vs.dependsOn v → v = w := by
  simp only [survivor, getD_splits hk] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i w hf
      have hw : w ∈ live.filter fun v => (vs.dependsOn v).contains splits[k] :=
        hf ▸ List.mem_singleton_self w
      split at h
      · cases h
      · rename_i hne
        split at h
        · cases h
        · rename_i ℓ' hrec
          cases h
          refine ⟨w, (List.mem_filter.mp hw).1,
            List.contains_iff_mem.mp (List.mem_filter.mp hw).2,
            fun he => hne (by simp [he]), rfl, fun r he => hrec r he,
            fun v hv hdep => ?_⟩
          have : v ∈ live.filter fun v =>
              (vs.dependsOn v).contains splits[k] :=
            List.mem_filter.mpr ⟨hv, List.elem_eq_true_of_mem hdep⟩
          rw [hf] at this
          exact List.mem_singleton.mp this
    · cases h

/--
A split whose value is live may not merge, since the value depends on itself.

# Parameters
- `splits`: The stack of splits, from the bottom.
- `k`: The position of the split.
- `live`: The live values.

# Hypotheses
- `hk`: `k` is in range.
- `hl`: The split value is live.
- `hf`: It fans out.
-/
theorem survivor_eq_none_of_live {splits : List Nat} {k : Nat}
    {live : List Nat} (hk : k < splits.length) (hl : splits[k] ∈ live)
    (hf : vs.fansOut splits[k]) : vs.survivor splits k live = none := by
  have hmem : splits[k] ∈ live.filter fun v =>
      (vs.dependsOn v).contains splits[k] :=
    List.mem_filter.mpr
      ⟨hl, List.elem_eq_true_of_mem (vs.self_mem_dependsOn hf)⟩
  simp only [survivor, getD_splits hk]
  split
  · rfl
  · split
    · rename_i hf'; rw [hf'] at hmem; cases hmem
    · rename_i w hf'
      rw [hf'] at hmem
      rw [List.mem_singleton.mp hmem]
      simp
    · rfl

/--
In a stack whose splits ascend, no split but itself depends on a split that
may merge: none above it does, since it may merge, and none beneath it does,
since each was written before it.

# Parameters
- `splits`: The stack of splits, from the bottom.
- `k`: The position of the split.
- `j`: A split in the stack.
- `live`: The live values.
- `s`: The survivor's location, if any.

# Hypotheses
- `hs`: The splits ascend.
- `hk`: `k` is in range.
- `h`: The split may merge, into `s`.
- `hj`: `j` is in the stack.
- `hdep`: `j` depends on the split.
-/
theorem eq_of_mem_dependsOn_of_survivor {splits : List Nat} {k j : Nat}
    {live : List Nat} {s : Option Location}
    (hs : splits.Pairwise (· < ·)) (hk : k < splits.length)
    (h : vs.survivor splits k live = some s) (hj : j ∈ splits)
    (hdep : splits[k] ∈ vs.dependsOn j) : j = splits[k] := by
  obtain ⟨m, hm, rfl⟩ := List.getElem_of_mem hj
  have hle := (vs.fansOut_of_mem_dependsOn hdep).2
  rcases Nat.lt_trichotomy m k with hlt | rfl | hgt
  · have := List.pairwise_iff_getElem.mp hs m k hm hk hlt
    omega
  · rfl
  · refine absurd hdep (vs.not_mem_dependsOn_of_survivor hk h _ ?_)
    have hidx : k + 1 + (m - (k + 1)) = m := by omega
    rw [List.mem_iff_getElem]
    exact ⟨m - (k + 1), by simp; omega, by simp only [List.getElem_drop, hidx]⟩

/-! ## Merges -/

/--
Merging either stops, or merges a split that may merge and then merges the
rest.

# Parameters
- `live`: The live values.
- `current`: The values that occupy their locations.
- `splits`: The stack of splits, from the bottom.
-/
theorem merges_cases (live current splits : List Nat) :
    vs.merges live current splits = ([], splits) ∨
      ∃ k, ∃ hk : k < splits.length, ∃ s,
        vs.survivor splits k live = some s ∧
          vs.merges live current splits =
            (.merge k s (vs.dead splits[k] s current) ::
                (vs.merges live current (splits.eraseIdx k)).1,
              (vs.merges live current (splits.eraseIdx k)).2) := by
  rw [merges]
  split
  · rename_i k s hfind
    obtain ⟨k', _, hk'⟩ := List.exists_of_findSome?_eq_some hfind
    rw [Option.map_eq_some_iff] at hk'
    obtain ⟨s', hs', he⟩ := hk'
    cases he
    split
    · rename_i hk
      exact .inr ⟨_, hk, _, hs', rfl⟩
    · exact .inl rfl
  · exact .inl rfl

/--
The splits that remain open after merging are among those open before, in
order.

# Parameters
- `live`: The live values.
- `current`: The values that occupy their locations.
- `splits`: The stack of splits, from the bottom.
-/
theorem merges_sublist (live current splits : List Nat) :
    (vs.merges live current splits).2.Sublist splits := by
  rcases vs.merges_cases live current splits with h | ⟨k, hk, s, _, h⟩
  · rw [h]
    exact List.Sublist.refl _
  · rw [h]
    exact (merges_sublist live current (splits.eraseIdx k)).trans
      (List.eraseIdx_sublist _ _)
termination_by splits.length
decreasing_by rw [List.length_eraseIdx_of_lt hk]; omega

/--
Merging nothing leaves every split open.

# Parameters
- `live`: The live values.
- `current`: The values that occupy their locations.
- `splits`: The stack of splits, from the bottom.

# Hypotheses
- `h`: Nothing merges.
-/
theorem merges_eq_nil {live current splits : List Nat}
    (h : (vs.merges live current splits).1 = []) :
    (vs.merges live current splits).2 = splits := by
  rcases vs.merges_cases live current splits with he | ⟨k, hk, s, _, he⟩
  · rw [he]
  · rw [he] at h
    cases h

/--
The first merge merges a split that may merge, into its survivor, burying the
other values that depend on it, and the merges after it are those of the stack
without it.

# Parameters
- `live`: The live values.
- `current`: The values that occupy their locations.
- `splits`: The stack of splits, from the bottom.
- `step`: The first merge.
- `steps`: The merges after it.
- `rest`: The splits open after them.

# Hypotheses
- `h`: Merging answers `step`, then `steps`, leaving `rest` open.
-/
theorem merges_eq_cons {live current splits : List Nat} {step : Step}
    {steps : List Step} {rest : List Nat}
    (h : vs.merges live current splits = (step :: steps, rest)) :
    ∃ k, ∃ hk : k < splits.length, ∃ s,
      vs.survivor splits k live = some s ∧
        step = .merge k s (vs.dead splits[k] s current) ∧
          vs.merges live current (splits.eraseIdx k) = (steps, rest) := by
  rcases vs.merges_cases live current splits with he | ⟨k, hk, s, hs, he⟩
  · rw [he] at h
    cases h
  · rw [he] at h
    injection h with h₁ h₂
    injection h₁ with h₁ h₃
    exact ⟨k, hk, s, hs, h₁.symm, Prod.ext h₃ h₂⟩

/--
A live split that fans out never merges.

# Parameters
- `live`: The live values.
- `current`: The values that occupy their locations.
- `splits`: The stack of splits, from the bottom.
- `v`: The split value.

# Hypotheses
- `hv`: `v` is open.
- `hl`: `v` is live.
- `hf`: `v` fans out.
-/
theorem mem_merges {live current splits : List Nat} {v : Nat}
    (hv : v ∈ splits) (hl : v ∈ live) (hf : vs.fansOut v) :
    v ∈ (vs.merges live current splits).2 := by
  rcases vs.merges_cases live current splits with h | ⟨k, hk, s, hs, h⟩
  · rw [h]
    exact hv
  · rw [h]
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hv
    have hik : i ≠ k := fun he => by
      subst he
      rw [vs.survivor_eq_none_of_live hk hl hf] at hs
      cases hs
    exact mem_merges
      (List.mem_eraseIdx_iff_getElem.mpr ⟨i, hi, hik, rfl⟩) hl hf
termination_by splits.length
decreasing_by rw [List.length_eraseIdx_of_lt hk]; omega

/-! ## Open splits -/

/--
The splits open after an instruction: those open before it, with its value
atop them if it fans out, less those that merge after it.

# Parameters
- `pc`: The program counter of the instruction.
-/
theorem openSplits_succ (pc : Nat) :
    vs.openSplits (pc + 1) =
      (vs.merges (vs.live pc) (vs.current pc)
        (if vs.fansOut pc then vs.openSplits pc ++ [pc]
          else vs.openSplits pc)).2 := by
  cases h : vs.fansOut pc <;> simp [openSplits, after, h]

/--
The steps after an instruction: a split on its value if it fans out, then the
merges of the stack that it leaves.

# Parameters
- `pc`: The program counter of the instruction.
-/
theorem steps_eq (pc : Nat) :
    vs.steps pc =
      (if vs.fansOut pc then [.split (vs.location pc)] else []) ++
        (vs.merges (vs.live pc) (vs.current pc)
          (if vs.fansOut pc then vs.openSplits pc ++ [pc]
            else vs.openSplits pc)).1 := by
  cases h : vs.fansOut pc <;> simp [steps, after, h]

/--
Every open split fans out, and was written before the instruction.

# Parameters
- `pc`: The program counter of the instruction.
- `v`: The split.

# Hypotheses
- `h`: `v` is open before `pc`.
-/
theorem mem_openSplits {pc v : Nat} (h : v ∈ vs.openSplits pc) :
    vs.fansOut v ∧ v < pc := by
  induction pc with
  | zero => simp [openSplits] at h
  | succ pc ih =>
    rw [openSplits_succ] at h
    have h' := (vs.merges_sublist _ _ _).subset h
    split at h'
    · rename_i hf
      rcases List.mem_append.mp h' with h' | h'
      · exact ⟨(ih h').1, by have := (ih h').2; omega⟩
      · rw [List.mem_singleton.mp h']
        exact ⟨hf, Nat.lt_succ_self _⟩
    · exact ⟨(ih h').1, by have := (ih h').2; omega⟩

/--
No split is open twice.

# Parameters
- `pc`: The program counter of the instruction.
-/
theorem nodup_openSplits (pc : Nat) : (vs.openSplits pc).Nodup := by
  induction pc with
  | zero => exact List.nodup_nil
  | succ pc ih =>
    rw [openSplits_succ]
    refine (vs.merges_sublist _ _ _).nodup ?_
    split
    · exact List.nodup_append.mpr ⟨ih, by simp,
        fun a ha b hb => by
          rw [List.mem_singleton.mp hb]
          exact Nat.ne_of_lt (vs.mem_openSplits ha).2⟩
    · exact ih

/--
The open splits ascend from the bottom of the stack, since each is pushed
after those written before it.

# Parameters
- `pc`: The program counter of the instruction.
-/
theorem pairwise_openSplits (pc : Nat) :
    (vs.openSplits pc).Pairwise (· < ·) := by
  induction pc with
  | zero => exact List.Pairwise.nil
  | succ pc ih =>
    rw [openSplits_succ]
    refine List.Pairwise.sublist (vs.merges_sublist _ _ _) ?_
    split
    · exact List.pairwise_append.mpr ⟨ih, by simp,
        fun a ha b hb => by
          rw [List.mem_singleton.mp hb]
          exact (vs.mem_openSplits ha).2⟩
    · exact ih

/--
A live value that fans out is an open split: it splits after its instruction,
and may not merge while it is live.

# Parameters
- `v`: The value.
- `pc`: The program counter of the instruction.

# Hypotheses
- `hf`: `v` fans out.
- `hv`: `v` was written at or before `pc`.
- `h`: `v` is live after `pc`.
-/
theorem mem_openSplits_of_live {v pc : Nat} (hf : vs.fansOut v)
    (hv : v ≤ pc) (h : vs.isLiveAfter v pc) : v ∈ vs.openSplits (pc + 1) := by
  induction pc with
  | zero =>
    rw [Nat.le_zero.mp hv] at hf h ⊢
    rw [openSplits_succ]
    exact vs.mem_merges (by simp [hf]) ((vs.mem_live).mpr ⟨Nat.le_refl _, h⟩) hf
  | succ pc ih =>
    have hs : v ∈ (if vs.fansOut (pc + 1)
        then vs.openSplits (pc + 1) ++ [pc + 1]
        else vs.openSplits (pc + 1)) := by
      by_cases hv' : v = pc + 1
      · subst hv'
        simp [hf]
      · have := ih (by omega) (vs.isLiveAfter_of_le h (Nat.le_succ _))
        split <;> simp [this]
    rw [openSplits_succ]
    exact vs.mem_merges hs ((vs.mem_live).mpr ⟨hv, h⟩) hf

/--
Two distinct members make a list longer than one.

# Parameters
- `l`: The list.
- `a`, `b`: The members.

# Hypotheses
- `ha`, `hb`: They are members.
- `hab`: They differ.
-/
private theorem one_lt_length {l : List Nat} {a b : Nat} (ha : a ∈ l)
    (hb : b ∈ l) (hab : a ≠ b) : 1 < l.length := by
  match l with
  | [] => simp at ha
  | [_] =>
    simp only [List.mem_singleton] at ha hb
    exact absurd (ha.trans hb.symm) hab
  | _ :: _ :: _ => simp

/--
Every random value that an instruction reads is an open split before it, or
is read for the last time: so the instruction combines values that are fixed
in each world of the open splits with values that may be buried after it.

# Parameters
- `pc`: The program counter of the instruction.
- `u`: The value.

# Hypotheses
- `hu`: The instruction reads `u`.
- `hr`: `u` may be random.
- `h`: `u` is live after the instruction.
-/
theorem mem_openSplits_of_random {pc u : Nat} (hu : u ∈ vs.reads pc)
    (hr : vs.random u) (h : vs.isLiveAfter u pc) : u ∈ vs.openSplits pc := by
  have hlt := vs.lt_of_mem_reads hu
  simp only [isLiveAfter, lasting, Bool.or_eq_true, beq_iff_eq,
    List.any_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨r, hr', hlt'⟩
  · exact absurd (vs.location_of_writer h)
      (vs.location_ne_answer_of_mem_reads hu)
  · have hf : vs.fansOut u := by
      simp only [fansOut, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hr, one_lt_length ((vs.mem_readers).mpr hu) hr'
        (Nat.ne_of_lt hlt')⟩
    obtain ⟨p, rfl⟩ : ∃ p, pc = p + 1 := ⟨pc - 1, by omega⟩
    exact vs.mem_openSplits_of_live hf (by omega)
      (vs.isLiveAfter_of_mem_reads hu (Nat.lt_succ_self _))

/-! ## Liveness and dead locations -/

/--
The location of an instruction's value is its destination.

# Parameters
- `w`: The program counter.

# Hypotheses
- `hw`: `w` is an instruction of the function.
-/
theorem location_eq {w : Nat} (hw : w < vs.instructions.length) :
    vs.location w = vs.instructions[w].destination := by
  simp [location, List.getElem?_eq_getElem hw]

/--
A register or rolling record is live after an instruction exactly when its
location is not dead before the rest of the function: a value is live when a
later instruction reads it, which it does exactly when the instruction reads
its location before any other writes there. This is how lemma 6 asks which
locations a merge may bury.

# Parameters
- `src`: The returned operand.
- `pc`: The program counter of the instruction.
- `ℓ`: The location.
- `v`: The value in it after the instruction.

# Hypotheses
- `hvals`: Every operand is a value.
- `hret`: The last instruction returns `src`.
- `hpc`: The instruction is not the last.
- `hℓ`: `ℓ` is not the answer.
- `hw`: `v` is in `ℓ` after the instruction.

# Notes
After the last instruction, the two differ at `src`: `Location.dead` reads
`src` once more after the instructions, as the specification's `run` does,
though the `return` has already read it.
-/
theorem dead_eq {src : AddressingMode}
    (hvals : ∀ i ∈ vs.instructions, i.operandsAreValues = true)
    (hret : vs.instructions.getLast? = some (.return src)) {pc v : Nat}
    {ℓ : Location} (hpc : pc + 1 < vs.instructions.length)
    (hℓ : ℓ ≠ .answer) (hw : vs.writer (pc + 1) ℓ = some v) :
    ℓ.dead src (vs.instructions.drop (pc + 1)) = !vs.isLiveAfter v pc := by
  have hv := vs.location_of_writer hw
  have hvpc := ((vs.writer_eq_some_iff).mp hw).1
  -- A later instruction reads `v` exactly when it reads `ℓ` before any other
  -- instruction writes there.
  have hlive : vs.isLiveAfter v pc = true ↔
      ∃ r, ∃ hr : r < vs.instructions.length, pc < r ∧
        ℓ.readBy vs.instructions[r] = true ∧
          ∀ w (hw : w < vs.instructions.length), pc < w → w < r →
            ℓ.writtenBy vs.instructions[w] = false := by
    simp only [isLiveAfter, lasting, Bool.or_eq_true, beq_iff_eq,
      List.any_eq_true, decide_eq_true_eq]
    constructor
    · rintro (hl | ⟨r, hr, hlt⟩)
      · exact absurd (hv.symm.trans (vs.location_of_writer hl)) hℓ
      · have hu := (vs.mem_readers).mp hr
        have hrn := vs.lt_length_of_mem_reads hu
        obtain ⟨ℓ', hℓ', hw'⟩ := (vs.mem_reads).mp hu
        rw [← vs.location_of_writer hw', hv] at hℓ'
        rw [← vs.location_of_writer hw', hv] at hw'
        simp only [sourcesAt, List.getElem?_eq_getElem hrn, Option.map_some,
          Option.getD_some] at hℓ'
        obtain ⟨_, _, h₃⟩ := (vs.writer_eq_some_iff).mp hw'
        refine ⟨r, hrn, hlt,
          (Location.readBy_iff (hvals _ (List.getElem_mem hrn))).mpr hℓ',
          fun w hwn hw₁ hw₂ => ?_⟩
        cases hwr : ℓ.writtenBy vs.instructions[w]
        · rfl
        · rw [Location.writtenBy_iff, ← vs.location_eq hwn] at hwr
          exact absurd hwr.symm (h₃ w (by omega) hw₂)
    · rintro ⟨r, hrn, hlt, hr, hb⟩
      refine .inr ⟨r, (vs.mem_readers).mpr ((vs.mem_reads).mpr ⟨ℓ, ?_, ?_⟩),
        hlt⟩
      · simp only [sourcesAt, List.getElem?_eq_getElem hrn, Option.map_some,
          Option.getD_some]
        exact (Location.readBy_iff (hvals _ (List.getElem_mem hrn))).mp hr
      · refine vs.writer_of_le hw hlt fun w hw₁ hw₂ hwℓ => ?_
        have hwn : w < vs.instructions.length := by omega
        have := hb w hwn (by omega) hw₂
        rw [vs.location_eq hwn] at hwℓ
        rw [Location.writtenBy_iff.mpr hwℓ.symm] at this
        cases this
  have hlast : (vs.instructions.drop (pc + 1)).getLast? =
      some (.return src) := by
    rw [List.getLast?_drop]
    split
    · exact absurd ‹_› (by omega)
    · exact hret
  have hdead := Location.dead_eq_false_iff (ℓ := ℓ) hlast
  simp only [List.getElem_drop, List.length_drop] at hdead
  have hiff : ℓ.dead src (vs.instructions.drop (pc + 1)) = false ↔
      vs.isLiveAfter v pc = true := by
    rw [hdead, hlive]
    constructor
    · rintro ⟨j, hj, hr, hb⟩
      exact ⟨pc + 1 + j, by omega, by omega, hr, fun w hwn hw₁ hw₂ => by
        have := hb (w - (pc + 1)) (by omega)
        simpa [Nat.add_sub_cancel' (show pc + 1 ≤ w by omega)] using this⟩
    · rintro ⟨r, hrn, hlt, hr, hb⟩
      refine ⟨r - (pc + 1), by omega, ?_, fun i hi => hb _ _ (by omega)
        (by omega)⟩
      simpa [Nat.add_sub_cancel' (show pc + 1 ≤ r by omega)] using hr
  cases hd : ℓ.dead src (vs.instructions.drop (pc + 1)) <;>
    cases hl : vs.isLiveAfter v pc <;> simp_all

end Values

end Xdy
