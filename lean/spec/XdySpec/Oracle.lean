import XdySpec.Law
import XdySpec.Semantics

/-!
# The oracle meets the specification

The executable oracle, `Xdy.Function.run`, answers exactly the specification's
law of a function's result: `run_eq` says that the oracle's weights over its
total are the specification's probabilities, and that each fails, with the
same message, exactly when the other does.

The two differ in how they hold rolling records. The specification keeps
results in the order rolled, as Rust does; the oracle sorts them as they
arrive, so that states differing only in the order of their rolls merge. So
the proof relates each oracle distribution of states to the specification's
law of states with every record sorted, read by `State.toOracle`. That
relation holds of the initial state, whose records are empty; each
instruction keeps it, since sorting commutes with rolling, dropping and
summing, by `hasLaw_step`; so it holds of the final states, by
`hasLaw_foldl`; and the returned operand reads only registers, which sorting
leaves alone.

Each instruction keeps the relation because `Dist.hasLaw_bind` needs the law
of each step only at the sorted reading of a state, and the oracle's step
from the sorted state has the law of the specification's step from the
unsorted one, sorted: the order of a record's results changes nothing that
any instruction can observe.
-/

namespace Xdy.Spec

/-! ## Reading states sorted -/

namespace State

/--
Sorting leaves every operand's value alone.

# Parameters
- `s`: The state.
- `op`: The operand.
-/
theorem value_toOracle (s : State) (op : AddressingMode) :
    s.toOracle.value op = s.value op := by
  cases op <;> rfl

/--
Reading a record of the sorted state reads the record sorted.

# Parameters
- `s`: The state.
- `r`: The rolling record.
-/
theorem record_toOracle (s : State) (r : Nat) :
    s.toOracle.record r = (s.record r).toOracle := by
  simp only [Xdy.State.record, record, toOracle, Array.getD_eq_getD_getElem?,
    Array.getElem?_map]
  cases s.records[r]? <;> simp [Record.toOracle_empty]

/--
Sorting commutes with replacing a record.

# Parameters
- `s`: The state.
- `r`: The rolling record.
- `rec`: Its new contents.
-/
theorem setRecord_toOracle (s : State) (r : Nat) (rec : Record) :
    (s.setRecord r rec).toOracle = s.toOracle.setRecord r rec.toOracle := by
  simp only [toOracle, setRecord, Xdy.State.setRecord, Xdy.State.mk.injEq,
    true_and]
  apply Array.ext
  · simp
  · intro i _ _
    simp only [Array.getElem_map, Array.getElem_modify]
    split <;> rfl

/--
Sorting commutes with writing a register.

# Parameters
- `s`: The state.
- `r`: The register.
- `v`: The value.
-/
theorem setRegister_toOracle (s : State) (r : Nat) (v : Int) :
    (s.setRegister r v).toOracle = s.toOracle.setRegister r v := rfl

/--
Sorting the specification's reading of an oracle state gives it back, if its
records are sorted, as the oracle keeps them.

# Parameters
- `s`: The oracle's state.

# Hypotheses
- `sorted`: The results of every record are sorted in ascending order.
-/
theorem toOracle_ofOracle (s : Xdy.State)
    (sorted : ∀ r ∈ s.records, r.results.Pairwise (· ≤ ·)) :
    (ofOracle s).toOracle = s := by
  cases s with
  | mk registers records =>
  simp only [toOracle, ofOracle, Xdy.State.mk.injEq, true_and]
  apply Array.ext
  · simp
  · intro i _ _
    simp only [Array.getElem_map]
    exact Record.toOracle_ofOracle _ (sorted _ (Array.getElem_mem _))

end State

/-! ## Draws -/

/--
The integers from `a` to `b`, as `Finset.Icc` holds them, are those that the
oracle lists: `a` plus each natural below the width of the range.

# Parameters
- `a`: The least integer.
- `b`: The greatest integer.
-/
theorem icc_val (a b : ℤ) :
    (Finset.Icc a b).val =
      ((List.range (b - a + 1).toNat).map fun (i : ℕ) => a + i : List ℤ) := by
  rw [Int.Icc_eq_finset_map, Finset.map_val, Finset.range_val, Multiset.range,
    Multiset.map_coe]
  congr 2
  rw [show b + 1 - a = b - a + 1 by omega]

/--
The oracle's range has the law of the specification's, sorted.

# Parameters
- `a`: The least result.
- `b`: The greatest result.
-/
theorem hasLaw_rollRange (a b : Int) :
    (Xdy.rollRange a b).HasLaw ((rollRange a b).map Record.toOracle) := by
  unfold Xdy.rollRange rollRange
  by_cases h : b < a
  · simp only [h, ↓reduceIte, ↓reduceDIte, PMF.pure_map]
    convert Dist.hasLaw_pure _ <;> simp [Record.toOracle, Record.sorted]
  · simp only [h, ↓reduceIte, ↓reduceDIte, PMF.map_comp]
    have hne :
        ((List.range (b - a + 1).toNat).map fun (i : ℕ) => a + i) ≠ [] := by
      simp; omega
    convert Dist.hasLaw_map _ (Dist.hasLaw_uniform _ hne) using 2
    all_goals first
      | (congr 1; exact icc_val a b)
      | (funext x; simp [Record.toOracle, Record.sorted])

/--
The oracle's standard die has the specification's law.

# Parameters
- `n`: The number of faces.
-/
theorem hasLaw_standardDie (n : Int) :
    (Xdy.standardDie n).HasLaw (standardDie n) := by
  unfold Xdy.standardDie standardDie
  by_cases h : n ≤ 0
  · simp only [h, ↓reduceIte, ↓reduceDIte]
    exact Dist.hasLaw_pure _
  · simp only [h, ↓reduceIte, ↓reduceDIte]
    have hne :
        ((List.range n.toNat).map fun (i : ℕ) => (i : Int) + 1) ≠ [] := by
      simp; omega
    convert Dist.hasLaw_uniform _ hne using 2
    rw [icc_val, show n - 1 + 1 = n by omega]
    congr 2
    funext i; omega

/--
The oracle's custom die has the specification's law.

# Parameters
- `faces`: The faces.
-/
theorem hasLaw_customDie (faces : List Int) :
    (Xdy.customDie faces).HasLaw (customDie faces) := by
  unfold Xdy.customDie customDie
  by_cases h : faces = []
  · subst h; exact Dist.hasLaw_pure _
  · simp only [h, ↓reduceIte, ↓reduceDIte, List.isEmpty_eq_false_iff.mpr h,
      Bool.false_eq_true]
    exact Dist.hasLaw_uniform _ h

/--
The oracle's set of dice has the law of the specification's, sorted: rolling
in order and sorting at the end is sorting as each die arrives.

# Parameters
- `c`: The number of dice.
- `die`: The oracle's die.
- `pdie`: The specification's die.

# Hypotheses
- `h`: The oracle's die has the specification's law.
-/
theorem hasLaw_rollDice (c : Int) (die : Unit → Dist Int) (pdie : PMF Int)
    (h : (die ()).HasLaw pdie) :
    (Xdy.rollDice c die).HasLaw ((rollDice c pdie).map Record.toOracle) := by
  unfold Xdy.rollDice rollDice
  by_cases hc : c ≤ 0
  · have : c.toNat = 0 := by omega
    simp only [hc, ↓reduceIte, this, Nat.repeat, PMF.pure_map,
      Record.toOracle_empty]
    exact Dist.hasLaw_pure _
  · simp only [hc, ↓reduceIte]
    generalize c.toNat = n
    induction n with
    | zero =>
      simp only [Nat.repeat, PMF.pure_map, Record.toOracle_empty]
      exact Dist.hasLaw_pure _
    | succ n ih =>
      simp only [Nat.repeat]
      refine Dist.hasLaw_bind ih fun r _ => ?_
      rw [PMF.map_comp, show Record.toOracle ∘ r.push = r.toOracle.push from
        funext (Record.toOracle_push r)]
      exact Dist.hasLaw_map _ h

/-! ## Instructions -/

/--
A roll into a record of the sorted state has the law of the specification's
roll, sorted, if the rolls themselves do.

# Parameters
- `s`: The specification's state.
- `d`: The rolling record.
- `oracle`: The oracle's roll.
- `spec`: The specification's roll.

# Hypotheses
- `h`: The oracle's roll has the law of the specification's, sorted.
-/
private theorem hasLaw_roll (s : State) (d : Nat) (oracle : Dist Xdy.Record)
    (spec : PMF Record) (h : oracle.HasLaw (spec.map Record.toOracle)) :
    (oracle.map (s.toOracle.setRecord d)).HasLaw
      ((spec.map (s.setRecord d)).map State.toOracle) := by
  rw [PMF.map_comp, show State.toOracle ∘ s.setRecord d =
      s.toOracle.setRecord d ∘ Record.toOracle from
    funext fun r => State.setRecord_toOracle s d r, ← PMF.map_comp]
  exact Dist.hasLaw_map _ h

/--
The oracle's step from the sorted state has the law of the specification's
step, sorted.

# Parameters
- `s`: The specification's state.
- `i`: The instruction.
-/
theorem hasLaw_step (s : State) (i : Instruction) :
    (Xdy.step s.toOracle i).HasLaw ((step s i).map State.toOracle) := by
  cases i with
  | rollRange d a b =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ (hasLaw_rollRange _ _)
  | rollStandardDice d c n =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ (hasLaw_rollDice _ _ _ (hasLaw_standardDie _))
  | rollCustomDice d c faces =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ (hasLaw_rollDice _ _ _ (hasLaw_customDie _))
  | dropLowest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_dropLowest, State.record_toOracle, State.value_toOracle]
    exact Dist.hasLaw_pure _
  | dropHighest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_dropHighest, State.record_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | sumRollingRecord d r =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.record_toOracle, Record.sum_toOracle]
    exact Dist.hasLaw_pure _
  | binary op d a b =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | neg d a =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | «return» _ =>
    simp only [Xdy.step, step, PMF.pure_map]
    exact Dist.hasLaw_pure _

/--
Running instructions keeps the relation: if the oracle's states have the law
of the specification's, sorted, then so do they after the instructions.

# Parameters
- `is`: The instructions.
- `d`: The oracle's distribution of states before them.
- `p`: The specification's law of states before them.

# Hypotheses
- `h`: `d` has the law of `p`, sorted.
-/
theorem hasLaw_foldl (is : List Instruction) (d : Dist Xdy.State)
    (p : PMF State) (h : d.HasLaw (p.map State.toOracle)) :
    (is.foldl (fun d i => d.bind (Xdy.step · i)) d).HasLaw
      ((is.foldl (fun p i => p.bind (step · i)) p).map State.toOracle) := by
  induction is generalizing d p with
  | nil => exact h
  | cons i is ih =>
    exact ih _ _ (Dist.hasLaw_bind h fun s _ => hasLaw_step s i)

/-! ## Functions -/

/--
A property that every step of a fold in `Except` keeps holds of the fold's
result, if it succeeds.

# Parameters
- `P`: The property.
- `f`: The step.
- `l`: The list folded.
- `b`: The start.
- `b'`: The result.

# Hypotheses
- `hf`: Each step that succeeds keeps `P`.
- `hb`: The start satisfies `P`.
- `h`: The fold succeeds with `b'`.
-/
private theorem foldlM_inv {α β ε : Type} (P : β → Prop)
    (f : β → α → Except ε β) (hf : ∀ b a b', P b → f b a = .ok b' → P b') :
    ∀ (l : List α) (b b' : β), P b → l.foldlM f b = .ok b' → P b'
  | [], b, b', hb, h => by
    simp only [List.foldlM_nil] at h; cases h; exact hb
  | a :: l, b, b', hb, h => by
    rw [List.foldlM_cons] at h
    cases hfa : f b a with
    | error e => rw [hfa] at h; cases h
    | ok c => rw [hfa] at h; exact foldlM_inv P f hf l c b' (hf b a c hb hfa) h

/--
Every rolling record of the initial state is empty, since binding arguments
and external variables writes only registers.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.

# Hypotheses
- `h`: `s` is the initial state.
-/
theorem initialState_records {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (h : f.initialState args externals = .ok s) :
    s.records = .replicate f.rollingRecordCount {} := by
  unfold Xdy.Function.initialState at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · cases h
    · rename_i v hv
      refine foldlM_inv (fun s : Xdy.State => s.records = _) _ ?_ _ v s
        (foldlM_inv (fun s : Xdy.State => s.records = _) _ ?_ _ _ v rfl hv) h
      all_goals
        intro b x b' hb hx
        repeat' split at hx
        all_goals first
          | (cases hx; done)
          | (cases hx; exact hb)
  · cases h

/--
The oracle's run and the specification's agree: both fail with the same
message, or both succeed, the oracle's distribution having the
specification's law.
-/
private def RunsAgree : Except String (Dist Int) → Except String (PMF Int) →
    Prop
  | .ok d, .ok p => d.HasLaw p
  | .error e, .error e' => e = e'
  | _, _ => False

/--
The oracle and the specification agree on every run.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
-/
private theorem runsAgree (f : Function) (args : List Int)
    (externals : List (String × Int)) :
    RunsAgree (Xdy.Function.run f args externals) (run f args externals) := by
  unfold run Xdy.Function.run
  simp only [bind, Except.bind]
  split
  · rfl
  · split
    · rfl
    · rename_i s hs
      have sorted : ∀ r ∈ s.records, r.results.Pairwise (· ≤ ·) := by
        rw [initialState_records hs]
        intro r hr
        rw [(Array.mem_replicate.mp hr).2]
        exact .nil
      have hlaw := hasLaw_foldl f.instructions.toList (Dist.pure s)
        (PMF.pure (.ofOracle s)) (by
          rw [PMF.pure_map, State.toOracle_ofOracle s sorted]
          exact Dist.hasLaw_pure s)
      rw [Array.foldl_toList, Array.foldl_toList] at hlaw
      cases hb : f.instructions.back? with
      | none => rfl
      | some i =>
        cases i
        case «return» src =>
          have := Dist.hasLaw_map (·.value src) hlaw
          rwa [PMF.map_comp, show (·.value src) ∘ State.toOracle =
            (·.value src) from funext fun s => State.value_toOracle s src]
            at this
        all_goals rfl

/--
If the oracle answers a distribution, then the specification answers a law,
and the distribution has it.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `d`: The oracle's distribution.

# Hypotheses
- `h`: The oracle answers `d`.
-/
theorem run_hasLaw {f : Function} {args : List Int}
    {externals : List (String × Int)} {d : Dist Int}
    (h : Xdy.Function.run f args externals = .ok d) :
    ∃ p, run f args externals = .ok p ∧ d.HasLaw p := by
  have := runsAgree f args externals
  rw [h] at this
  cases hr : run f args externals with
  | error e => rw [hr] at this; cases this
  | ok p => rw [hr] at this; exact ⟨p, rfl, this⟩

/--
The oracle meets the specification: its weights over its total are the
specification's probabilities, and it fails, with the same message, exactly
when the specification does.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.

# Notes
A `PMF` is determined by its probabilities, so this determines the
specification's law from the oracle's distribution. `render` in
`Xdy/Oracle.lean` reduces the weights to lowest terms, which leaves every
probability unchanged.
-/
theorem run_eq (f : Function) (args : List Int)
    (externals : List (String × Int)) :
    (run f args externals).map (⇑) =
      (Xdy.Function.run f args externals).map Dist.prob := by
  have := runsAgree f args externals
  cases ho : Xdy.Function.run f args externals with
  | error e =>
    cases hr : run f args externals with
    | error e' => rw [ho, hr] at this; cases this; rfl
    | ok p => rw [ho, hr] at this; cases this
  | ok d =>
    cases hr : run f args externals with
    | error e' => rw [ho, hr] at this; cases this
    | ok p =>
      rw [ho, hr] at this
      simp only [Except.map]
      congr 1
      funext x
      exact this.2.2 x

end Xdy.Spec
