import XdySpec.Semantics

/-!
# Path enumeration

The paths through a function, as the specification sees them, and as the
enumerating builder walked them before the forward pass replaced it in 0.14.0
(`xdy/src/distribution/builder.rs`). The paths fork at every draw, one
successor for each outcome of the draw: one for each value of a range, or one
if the range is empty, and one for each face of each die in turn, or one if
the die has no faces. A custom die forks by the position of each face, so two
faces that show the same number fork apart. A limited reroll forks each die
that it rerolls by each face of each draw, until a draw no longer matches or
the limit runs out; the draws of an unlimited reroll are unbounded, so it
forks each such die by its conditional law instead, once for each face, by
position, that does not match. Every other instruction has one successor. A
path runs from the initial state to the end of the function.

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
to `stop`, or a single `0` if the range is empty, drawn from the range as its
die.

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
  if stop < start then [{ results := [0], die := .range start stop }]
  else
    (List.range (stop - start + 1).toNat).map fun (i : Nat) =>
      { results := [start + i], die := .range start stop }

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
The rolling record of each fork, whose results are in the order rolled, and
which has no die, which `forks` then records.

# Examples
```lean
#guard (diceForks 2 [1, 2]).map (·.results) == [[1, 1], [1, 2], [2, 1], [2, 2]]
#guard (diceForks 0 [1, 2]).map (·.results) == [[]]
```
-/
def diceForks (count : Int) (faces : List Int) : List Record :=
  count.toNat.repeat (fun rs => rs.flatMap fun r => faces.map r.push) [{}]

/--
The replacements of a die that a reroll limited to `limit` times crosses out,
as the paths fork them: one fork for each face of the first draw, then each
fork that still matches once for each face of the next draw, up to `limit - 1`
more draws, as `limitedReplacement` redraws.

# Parameters
- `c`: The criterion.
- `faces`: The faces of the die, as the paths fork them.
- `limit`: The number of times that each die may be rerolled.

# Returns
The replacement in each fork.

# Examples
```lean
#guard limitedForks (.within 1 1) [1, 2] 2 == [1, 2, 2]
```
-/
def limitedForks (c : Criterion) (faces : List Int) (limit : Nat) :
    List Int :=
  (limit - 1).repeat
    (fun ys => ys.flatMap fun y => if c.matches y then faces else [y]) faces

/--
The results of a reroll of kept results, as the paths fork them: each
rerolled result once for each replacement, in turn, and every other result
once. Mirrors `rerollKept`.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results of the record.
- `rerolled`: Whether the reroll rerolls a result.
- `replacements`: The replacement of a rerolled result in each fork.

# Returns
The rolling record of each fork.

# Examples
```lean
#guard (keptForks .none [1, 2, 1] (· == 1) [3, 4]).map (·.results)
  == [[3, 2, 3], [3, 2, 4], [4, 2, 3], [4, 2, 4]]
```
-/
def keptForks (die : Die) (kept : List Int) (rerolled : Int → Bool)
    (replacements : List Int) : List Record :=
  kept.foldl
    (fun rs x =>
      if rerolled x then rs.flatMap fun r => replacements.map r.push
      else rs.map (·.push x))
    [{ die }]

/--
The results of a reroll of a record, as the paths fork them. Mirrors
`reroll`. A limited reroll forks each matching result by its limited
replacement. An unlimited reroll whose criterion matches every face forks
once; otherwise it forks each matching result once for each face, by
position, that the criterion does not match. The paths of a die rerolled
without limit are unbounded, but its conditional law is the specification's
law, so the paths fork by its outcomes alone.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.

# Returns
The rolling record of each fork.

# Examples
```lean
#guard (rerollForks { results := [1, 2], die := .standard 3 } (.within 1 1)
  none).map (·.results) == [[2, 2], [3, 2]]
```
-/
def rerollForks (r : Record) (c : Criterion) : Option Nat → List Record
  | none =>
    if r.die.facesWithin c then
      [{ results := r.kept.filter (!c.matches ·), die := r.die }]
    else keptForks r.die r.kept c.matches (r.die.faces.filter (!c.matches ·))
  | some limit =>
    keptForks r.die r.kept (fun x => 0 < limit && c.matches x)
      (limitedForks c (customFaces r.die.faces) limit)

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
    (diceForks (s.value c) (standardFaces (s.value n))).map fun r =>
      s.setRecord d { r with die := .standard (s.value n) }
  | .rollCustomDice d c faces =>
    (diceForks (s.value c) (customFaces faces)).map fun r =>
      s.setRecord d { r with die := .ofFaces faces }
  | .dropLowest d c => [s.setRecord d ((s.record d).dropLowest (s.value c))]
  | .dropHighest d c => [s.setRecord d ((s.record d).dropHighest (s.value c))]
  | .keepLowest d c => [s.setRecord d ((s.record d).keepLowest (s.value c))]
  | .keepHighest d c => [s.setRecord d ((s.record d).keepHighest (s.value c))]
  | .reroll d p l =>
    (rerollForks (s.record d) (p.resolve s.value) (l.resolve s.value)).map
      (s.setRecord d)
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
The replacements of a limited reroll fork at least once, if a die does.

# Parameters
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled.

# Hypotheses
- `h`: A die forks at least once.
-/
theorem limitedForks_ne_nil (c : Criterion) {faces : List Int}
    (limit : Nat) (h : faces ≠ []) : limitedForks c faces limit ≠ [] := by
  unfold limitedForks
  induction limit - 1 with
  | zero => exact h
  | succ n ih =>
    simp only [Nat.repeat, ne_eq, List.flatMap_eq_nil_iff, not_forall]
    obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ ih
    refine ⟨y, hy, ?_⟩
    split
    · exact h
    · simp

/--
A reroll of kept results forks at least once, if a replacement does.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.

# Hypotheses
- `h`: A replacement forks at least once.
-/
theorem keptForks_ne_nil (die : Die) (kept : List Int)
    (rerolled : Int → Bool) {replacements : List Int}
    (h : replacements ≠ []) :
    keptForks die kept rerolled replacements ≠ [] := by
  unfold keptForks
  suffices ∀ rs : List Record, rs ≠ [] → kept.foldl
      (fun rs x =>
        if rerolled x then rs.flatMap fun r => replacements.map r.push
        else rs.map (·.push x)) rs ≠ [] from this _ (by simp)
  induction kept with
  | nil => exact fun _ hrs => hrs
  | cons x kept ih =>
    intro rs hrs
    refine ih _ ?_
    obtain ⟨r, hr⟩ := List.exists_mem_of_ne_nil _ hrs
    by_cases hx : rerolled x
    · simp only [hx, ↓reduceIte, ne_eq, List.flatMap_eq_nil_iff,
        List.map_eq_nil_iff, not_forall]
      exact ⟨r, hr, h⟩
    · simpa [hx] using hrs

/--
A reroll forks at least once.

# Parameters
- `r`: The record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.
-/
theorem rerollForks_ne_nil (r : Record) (c : Criterion) (limit : Option Nat) :
    rerollForks r c limit ≠ [] := by
  cases limit with
  | none =>
    by_cases hw : r.die.facesWithin c
    · simp [rerollForks, hw]
    · simp only [rerollForks, hw, Bool.false_eq_true, ↓reduceIte]
      exact keptForks_ne_nil _ _ _ (by simpa using unmatched_ne_zero hw)
  | some limit =>
    simp only [rerollForks]
    refine keptForks_ne_nil _ _ _ (limitedForks_ne_nil _ _ ?_)
    unfold customFaces
    split
    · simp
    · rename_i h
      simpa using h

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
  | reroll d p l => simpa [forks] using rerollForks_ne_nil _ _ _
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
Every replacement that a limited reroll can draw is among its forks, if every
face of one die is.

# Parameters
- `c`: The criterion.
- `die`: The law of one die.
- `faces`: The faces of one die, as the paths fork them.
- `limit`: The number of times that each die may be rerolled.
- `x`: The replacement.

# Hypotheses
- `hf`: Every face that the die can show is among `faces`.
- `h`: The reroll can draw `x`.
-/
theorem mem_limitedForks {c : Criterion} {die : PMF Int} {faces : List Int}
    {limit : Nat} {x : Int} (hf : ∀ x ∈ die.support, x ∈ faces)
    (h : x ∈ (limitedReplacement c die limit).support) :
    x ∈ limitedForks c faces limit := by
  unfold limitedReplacement at h
  unfold limitedForks
  generalize limit - 1 = n at h ⊢
  induction n generalizing x with
  | zero => exact hf x (by simpa [Nat.repeat] using h)
  | succ n ih =>
    simp only [Nat.repeat, redraw, PMF.mem_support_bind_iff] at h
    obtain ⟨y, hy, hx⟩ := h
    simp only [Nat.repeat, List.mem_flatMap]
    refine ⟨y, ih hy, ?_⟩
    split at hx
    · rename_i hm
      simp only [hm, ↓reduceIte]
      exact hf x hx
    · rename_i hm
      simp only [hm, Bool.false_eq_true, ↓reduceIte, List.mem_singleton]
      simpa using hx

/--
Every record that a reroll of kept results can reach is among its forks, from
any records before them among their forks, if every replacement is.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement.
- `replacements`: The replacements, as the paths fork them.
- `kept`: The results to reroll.
- `p`: The law of the records before them.
- `rs`: The forks of the records before them.

# Hypotheses
- `hf`: Every replacement that can be drawn is among `replacements`.
- `hp`: Every record that `p` can hold is among `rs`.
-/
theorem mem_foldl_keptForks {rerolled : Int → Bool} {replacement : PMF Int}
    {replacements : List Int}
    (hf : ∀ x ∈ replacement.support, x ∈ replacements) (kept : List Int)
    {p : PMF Record} {rs : List Record} (hp : ∀ r ∈ p.support, r ∈ rs) :
    ∀ r ∈ (kept.foldl
        (fun p x =>
          if rerolled x then p.bind fun r => replacement.map r.push
          else p.map (·.push x)) p).support,
      r ∈ kept.foldl
        (fun rs x =>
          if rerolled x then rs.flatMap fun r => replacements.map r.push
          else rs.map (·.push x)) rs := by
  induction kept generalizing p rs with
  | nil => exact hp
  | cons x kept ih =>
    refine ih fun r hr => ?_
    by_cases hx : rerolled x
    · simp only [hx, ↓reduceIte, PMF.mem_support_bind_iff, PMF.support_map,
        Set.mem_image] at hr
      obtain ⟨r', hr', y, hy, rfl⟩ := hr
      simp only [hx, ↓reduceIte, List.mem_flatMap, List.mem_map]
      exact ⟨r', hp r' hr', y, hf y hy, rfl⟩
    · simp only [hx, Bool.false_eq_true, ↓reduceIte, PMF.support_map,
        Set.mem_image] at hr
      obtain ⟨r', hr', rfl⟩ := hr
      simp only [hx, Bool.false_eq_true, ↓reduceIte]
      exact List.mem_map_of_mem (hp r' hr')

/--
Every record that a reroll of kept results can reach is among its forks, if
every replacement is.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.
- `replacement`: The law of the replacement.
- `replacements`: The replacements, as the paths fork them.
- `r`: The record.

# Hypotheses
- `hf`: Every replacement that can be drawn is among `replacements`.
- `h`: The reroll can reach `r`.
-/
theorem mem_keptForks {die : Die} {kept : List Int} {rerolled : Int → Bool}
    {replacement : PMF Int} {replacements : List Int} {r : Record}
    (hf : ∀ x ∈ replacement.support, x ∈ replacements)
    (h : r ∈ (rerollKept die kept rerolled replacement).support) :
    r ∈ keptForks die kept rerolled replacements :=
  mem_foldl_keptForks hf kept (by simp) r h

/--
Every record that a reroll can reach is among its forks.

# Parameters
- `r`: The record before the reroll.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.
- `r'`: The record after it.

# Hypotheses
- `h`: The reroll can reach `r'`.
-/
theorem mem_rerollForks {r : Record} {c : Criterion} {limit : Option Nat}
    {r' : Record} (h : r' ∈ (reroll r c limit).support) :
    r' ∈ rerollForks r c limit := by
  cases limit with
  | none =>
    by_cases hw : r.die.facesWithin c
    · simp only [reroll, hw, ↓reduceDIte, PMF.support_pure,
        Set.mem_singleton_iff] at h
      simp only [rerollForks, hw, ↓reduceIte, List.mem_singleton]
      exact h
    · simp only [reroll, hw, Bool.false_eq_true, ↓reduceDIte] at h
      simp only [rerollForks, hw, Bool.false_eq_true, ↓reduceIte]
      refine mem_keptForks (fun x hx => ?_) h
      rw [PMF.mem_support_ofMultiset_iff] at hx
      simpa using hx
  | some limit =>
    simp only [reroll] at h
    exact mem_keptForks (fun x hx => mem_limitedForks
      (fun y hy => mem_customFaces hy) hx) h

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
  | reroll d p l =>
    simp only [step, PMF.support_map, Set.mem_image] at h
    obtain ⟨r, hr, rfl⟩ := h
    exact List.mem_map_of_mem (mem_rerollForks hr)
  | _ => simpa [step, forks] using h

end Xdy.Spec
