import XdySpec.Semantics

/-!
# Path enumeration

The paths through a function, as the specification sees them, and as the
enumerating builder walked them before the forward pass replaced it in 0.14.0
(`xdy/src/distribution/builder.rs`). The paths fork at every draw, one
successor for each outcome of the draw: one for each value of a range, or one
if the range is empty, and one for each face of each die in turn, or one if
the die has no faces. A custom die forks by the position of each face, so two
faces that show the same number fork apart. Every other instruction has one
successor. A path runs from the initial state to the end of the function.

`forks` answers the successors of one instruction, and `enumerate` runs a list
of instructions over a list of states, forking each in turn:

```mermaid
flowchart LR
    S["[s]"] -->|"forks · i₀"| A["states after i₀"]
    A -->|"forks · i₁"| B["states after i₁"]
    B -->|"…"| C["paths s is"]
```

Every state that an instruction can reach is among its forks
(`mem_forks_of_mem_support`), and every instruction forks at least once
(`forks_ne_nil`), so the paths that reach a program point never outnumber
those that reach the end (`length_paths_take_le`). `XdySpec/Cost.lean` uses
both to bound the worlds of the forward pass by the paths.
-/

namespace Xdy.Spec

/-! ## Forks -/

/--
The results of a range, as the paths fork them: each value from `start`
to `stop`, or a single `0` if the range is empty.

# Parameters
- `start`: The least result.
- `stop`: The greatest result.

# Returns
The rolling record of each fork, in ascending order.

# Examples
```lean
#guard (rangeForks 1 3).map (·.results) == [[1], [2], [3]]
#guard (rangeForks 3 1).map (·.results) == [[0]]
```
-/
def rangeForks (start stop : Int) : List Record :=
  if stop < start then [{ results := [0] }]
  else
    (List.range (stop - start + 1).toNat).map fun (i : Nat) =>
      { results := [start + i] }

/--
The faces of a standard die, as the paths fork them: `1` through `faces`,
or a single `0` if `faces` is `0` or less.

# Parameters
- `faces`: The number of faces.

# Returns
The faces, in ascending order.

# Examples
```lean
#guard standardFaces 3 == [1, 2, 3]
#guard standardFaces 0 == [0]
```
-/
def standardFaces (faces : Int) : List Int :=
  if faces ≤ 0 then [0]
  else (List.range faces.toNat).map fun (i : Nat) => (i : Int) + 1

/--
The faces of a custom die, as the paths fork them: each listed face, by
position, or a single `0` if there are none.

# Parameters
- `faces`: The faces, which may repeat.

# Returns
The faces, in the order listed.

# Examples
```lean
#guard customFaces [1, 1, 2] == [1, 1, 2]
#guard customFaces [] == [0]
```
-/
def customFaces (faces : List Int) : List Int :=
  if faces.isEmpty then [0] else faces

/--
The results of a set of dice, as the paths fork them: one fork for each
face of the first die, then each of those forks once for each face of the
next, and so on.

# Parameters
- `count`: The number of dice. A count of `0` or less rolls no dice, and forks
  once.
- `faces`: The faces of one die, as the paths fork them.

# Returns
The rolling record of each fork, whose results are in the order rolled.

# Examples
```lean
#guard (diceForks 2 [1, 2]).map (·.results) == [[1, 1], [1, 2], [2, 1], [2, 2]]
#guard (diceForks 0 [1, 2]).map (·.results) == [[]]
```
-/
def diceForks (count : Int) (faces : List Int) : List Record :=
  count.toNat.repeat (fun rs => rs.flatMap fun r => faces.map r.push) [{}]

/--
The successors of one instruction in one state, as the paths fork them.
Mirrors `step`, with each draw's law replaced by the list of its forks.

# Parameters
- `s`: The state before the instruction.
- `inst`: The instruction.

# Returns
The state after the instruction in each fork: one for each outcome of a
roll's draws, and one for every other instruction.

# Examples
```lean
#guard (forks { registers := #[0], records := #[{}] }
  (.rollStandardDice 0 (.immediate 2) (.immediate 6))).length == 36
```
-/
def forks (s : State) : Instruction → List State
  | .rollRange d a b =>
    (rangeForks (s.value a) (s.value b)).map (s.setRecord d)
  | .rollStandardDice d c n =>
    (diceForks (s.value c) (standardFaces (s.value n))).map (s.setRecord d)
  | .rollCustomDice d c faces =>
    (diceForks (s.value c) (customFaces faces)).map (s.setRecord d)
  | .dropLowest d c => [s.setRecord d ((s.record d).dropLowest (s.value c))]
  | .dropHighest d c => [s.setRecord d ((s.record d).dropHighest (s.value c))]
  | .sumRollingRecord d r => [s.setRegister d (s.record r).sum]
  | .binary op d a b => [s.setRegister d (op.apply (s.value a) (s.value b))]
  | .neg d a => [s.setRegister d (neg (s.value a))]
  | .return _ => [s]

/-! ## Paths -/

/--
Run instructions over states, forking each state at each instruction.

# Parameters
- `is`: The instructions.
- `ts`: The states before them.

# Returns
The state at the end of each path, in the order that the forks list them.
-/
def enumerate (is : List Instruction) (ts : List State) : List State :=
  is.foldl (fun ts i => ts.flatMap (forks · i)) ts

/--
The paths of instructions from a state: the states at their ends, one for
each path.

# Parameters
- `s`: The state before the instructions.
- `is`: The instructions.

# Returns
The state at the end of each path. Its length is the number of paths.

# Examples
```lean
-- `3D6` forks 6 × 6 × 6 ways.
#guard (paths { registers := #[0], records := #[{}] }
  [.rollStandardDice 0 (.immediate 3) (.immediate 6),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 216
```
-/
def paths (s : State) (is : List Instruction) : List State := enumerate is [s]

/--
Enumerating a list of instructions, then another, enumerates them together.

# Parameters
- `is`: The first instructions.
- `js`: The instructions after them.
- `ts`: The states before them.
-/
theorem enumerate_append (is js : List Instruction) (ts : List State) :
    enumerate (is ++ js) ts = enumerate js (enumerate is ts) := by
  simp [enumerate, List.foldl_append]

/-! ## Every instruction forks -/

/--
A set of dice forks at least once, whatever its faces, if a die forks at least
once.

# Parameters
- `count`: The number of dice.
- `faces`: The faces of one die.

# Hypotheses
- `h`: A die forks at least once.
-/
theorem diceForks_ne_nil (count : Int) {faces : List Int} (h : faces ≠ []) :
    diceForks count faces ≠ [] := by
  unfold diceForks
  induction count.toNat with
  | zero => simp [Nat.repeat]
  | succ n ih =>
    simp only [Nat.repeat, ne_eq, List.flatMap_eq_nil_iff, List.map_eq_nil_iff,
      not_forall]
    obtain ⟨r, hr⟩ := List.exists_mem_of_ne_nil _ ih
    exact ⟨r, hr, h⟩

/--
Every instruction forks at least once.

# Parameters
- `s`: The state before the instruction.
- `i`: The instruction.
-/
theorem forks_ne_nil (s : State) (i : Instruction) : forks s i ≠ [] := by
  have hstd : ∀ n : Int, standardFaces n ≠ [] := by
    intro n
    unfold standardFaces
    split
    · simp
    · simp only [ne_eq, List.map_eq_nil_iff, List.range_eq_nil]
      omega
  have hcus : ∀ fs : List Int, customFaces fs ≠ [] := by
    intro fs
    unfold customFaces
    cases fs <;> simp
  cases i with
  | rollRange d a b =>
    simp only [forks, rangeForks, ne_eq, List.map_eq_nil_iff]
    split
    · simp
    · simp only [List.map_eq_nil_iff, List.range_eq_nil]
      omega
  | rollStandardDice d c n =>
    simpa [forks] using diceForks_ne_nil _ (hstd _)
  | rollCustomDice d c fs =>
    simpa [forks] using diceForks_ne_nil _ (hcus _)
  | _ => simp [forks]

/--
Enumerating instructions leaves at least as many paths as it starts with,
since every state forks at least once at each instruction.

# Parameters
- `is`: The instructions.
- `ts`: The states before them.
-/
theorem length_le_enumerate (is : List Instruction) (ts : List State) :
    ts.length ≤ (enumerate is ts).length := by
  induction is generalizing ts with
  | nil => exact Nat.le_refl _
  | cons i is ih =>
    refine Nat.le_trans ?_ (ih _)
    simp only [List.length_flatMap]
    induction ts with
    | nil => exact Nat.le_refl _
    | cons t ts ih' =>
      simp only [List.length_cons, List.map_cons, List.sum_cons]
      have := List.length_pos_of_ne_nil (forks_ne_nil t i)
      omega

/--
The paths that reach a program point never outnumber those that reach the
end: each of them continues along at least one.

# Parameters
- `s`: The state before the instructions.
- `is`: The instructions.
- `n`: The program point: the number of instructions run.
-/
theorem length_paths_take_le (s : State) (is : List Instruction) (n : Nat) :
    (paths s (is.take n)).length ≤ (paths s is).length := by
  conv_rhs => rw [← List.take_append_drop n is]
  rw [paths, paths, enumerate_append]
  exact length_le_enumerate _ _

/-! ## Every outcome is a fork -/

/--
Every face that a standard die can show is among its forks.

# Parameters
- `n`: The number of faces.
- `x`: The face.

# Hypotheses
- `h`: The die can show `x`.
-/
theorem mem_standardFaces {n x : Int} (h : x ∈ (standardDie n).support) :
    x ∈ standardFaces n := by
  unfold standardDie at h
  unfold standardFaces
  split at h
  · rename_i hn
    simp only [hn, ↓reduceIte, List.mem_singleton]
    simpa using h
  · rename_i hn
    simp only [hn, ↓reduceIte, List.mem_map, List.mem_range]
    simp only [PMF.mem_support_ofMultiset_iff, Finset.val_toFinset,
      Finset.mem_Icc] at h
    exact ⟨(x - 1).toNat, by omega, by omega⟩

/--
Every face that a custom die can show is among its forks.

# Parameters
- `fs`: The faces.
- `x`: The face.

# Hypotheses
- `h`: The die can show `x`.
-/
theorem mem_customFaces {fs : List Int} {x : Int}
    (h : x ∈ (customDie fs).support) : x ∈ customFaces fs := by
  unfold customDie at h
  unfold customFaces
  split at h
  · rename_i hfs
    subst hfs
    simpa using h
  · rename_i hfs
    simp only [List.isEmpty_iff, hfs, ↓reduceIte]
    simpa using h

/--
Every record that a set of dice can roll is among its forks, if every face of
one die is.

# Parameters
- `count`: The number of dice.
- `die`: The law of one die.
- `faces`: The faces of one die, as the paths fork them.
- `r`: The record.

# Hypotheses
- `hf`: Every face that the die can show is among `faces`.
- `h`: The dice can roll `r`.
-/
theorem mem_diceForks {count : Int} {die : PMF Int} {faces : List Int}
    {r : Record} (hf : ∀ x ∈ die.support, x ∈ faces)
    (h : r ∈ (rollDice count die).support) : r ∈ diceForks count faces := by
  unfold rollDice at h
  unfold diceForks
  generalize count.toNat = n at h ⊢
  induction n generalizing r with
  | zero => simpa [Nat.repeat] using h
  | succ n ih =>
    simp only [Nat.repeat, PMF.mem_support_bind_iff, PMF.support_map,
      Set.mem_image] at h
    obtain ⟨r', hr', x, hx, rfl⟩ := h
    simp only [Nat.repeat, List.mem_flatMap, List.mem_map]
    exact ⟨r', ih hr', x, hf x hx, rfl⟩

/--
Every record that a range can roll is among its forks.

# Parameters
- `a`: The least result.
- `b`: The greatest result.
- `r`: The record.

# Hypotheses
- `h`: The range can roll `r`.
-/
theorem mem_rangeForks {a b : Int} {r : Record}
    (h : r ∈ (rollRange a b).support) : r ∈ rangeForks a b := by
  unfold rollRange at h
  unfold rangeForks
  split at h
  · rename_i hab
    simp only [hab, ↓reduceIte, List.mem_singleton]
    simpa using h
  · rename_i hab
    simp only [hab, ↓reduceIte, List.mem_map, List.mem_range]
    simp only [PMF.support_map, Set.mem_image, PMF.mem_support_ofMultiset_iff,
      Finset.val_toFinset, Finset.mem_Icc] at h
    obtain ⟨x, hx, rfl⟩ := h
    exact ⟨(x - a).toNat, by omega, by congr; omega⟩

/--
Every state that an instruction can reach is among its forks.

# Parameters
- `s`: The state before the instruction.
- `i`: The instruction.
- `t`: The state after it.

# Hypotheses
- `h`: `i` can reach `t` from `s`.
-/
theorem mem_forks_of_mem_support {s : State} {i : Instruction} {t : State}
    (h : t ∈ (step s i).support) : t ∈ forks s i := by
  cases i with
  | rollRange d a b =>
    simp only [step, PMF.support_map, Set.mem_image] at h
    obtain ⟨r, hr, rfl⟩ := h
    exact List.mem_map_of_mem (mem_rangeForks hr)
  | rollStandardDice d c n =>
    simp only [step, PMF.support_map, Set.mem_image] at h
    obtain ⟨r, hr, rfl⟩ := h
    exact List.mem_map_of_mem
      (mem_diceForks (fun _ hx => mem_standardFaces hx) hr)
  | rollCustomDice d c fs =>
    simp only [step, PMF.support_map, Set.mem_image] at h
    obtain ⟨r, hr, rfl⟩ := h
    exact List.mem_map_of_mem
      (mem_diceForks (fun _ hx => mem_customFaces hx) hr)
  | _ => simpa [step, forks] using h

end Xdy.Spec
