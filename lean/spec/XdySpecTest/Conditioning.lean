import XdySpec.Conditioning
import XdySpecTest.Fixtures

/-!
# Conditioning tests

The laws of burying and merging, applied: burying a register that the next
instruction overwrites, merging with and without a survivor, and lemma 6 on a
compiled function whose random register fans out, and on a function whose
split merges beneath another, whose hypotheses it shows can hold. Which locations are dead is tested beside the plan, in
`XdyTest/Plan.lean`.
-/

namespace Xdy.Spec.Test

/-! ## Burying -/

-- Burying a register that the next instruction overwrites leaves the law of
-- the result alone.
example (μ : PMF State) :
    (exec [.binary .add 0 (.immediate 1) (.immediate 2)]
      (μ.map (·.bury (.register 0)))).map (·.value (.register 0)) =
    (exec [.binary .add 0 (.immediate 1) (.immediate 2)] μ).map
      (·.value (.register 0)) :=
  exec_bury rfl μ

/-! ## Merging -/

-- Merging worlds that each fix the survivor at their outcome gives the
-- survivor the law of the split.
example (w R : PMF Int) :
    w.bind (fun v => lift (· + ·) (PMF.pure v) R) = lift (· + ·) w R := by
  rw [merge, PMF.bind_pure]
-- A merge without a survivor leaves the common law of everything else.
example (w : PMF Int) (R : PMF State) :
    w.bind (fun _ => lift (fun _ s => s) (PMF.pure ()) R) =
      lift (fun _ s => s) (PMF.pure ()) R := by
  rw [merge, PMF.bind_const]

/-! ## A worked example

`{x}@(1D2) + {x} * 3`, as the compiler writes it, which reduces `1D2` to a
range. Two instructions read `@0`, so the plan splits on it after the sum, and
merges into the new `@0` after the add:

```text
⚅0 <- roll range 1:2
@0 <- sum rolling record ⚅0
@1 <- 3 * @0
@0 <- @0 + @1
return @0
```

Each world's state is certain, so it factors, but only once the rolling
record is buried: it holds the world's roll, so the rest of the state would
otherwise differ between the worlds. Only the return follows the merge, so
the record and `@1` are dead. So the law is that of `4 × 1D2`, not
`1D2 + 3 × 1D2`.
-/

/-- The instructions before the split. -/
private def beforeSplit : List Instruction :=
  [.rollRange 0 (.immediate 1) (.immediate 2), .sumRollingRecord 0 0]

/-- The instructions between the split and the merge. -/
private def beforeMerge : List Instruction :=
  [.binary .mul 1 (.immediate 3) (.register 0),
    .binary .add 0 (.register 0) (.register 1)]

/-- The instructions after the merge. -/
private def afterMerge : List Instruction := [.return (.register 0)]

/-- The initial state, with two registers and one rolling record. -/
private def initial : State := { registers := #[0, 0], records := #[{}] }

/-- The state after the sum, given the roll. -/
private def rolled (x : Int) : State :=
  { registers := #[x, 0], records := #[{ results := [x] }] }

/-- The value of the split, `@0`. -/
private def split₀ (s : State) : Int := s.value (.register 0)

/-- After the split's instructions, the state is the roll's. -/
private theorem exec_beforeSplit :
    exec beforeSplit (PMF.pure initial) = roll.map rolled := by
  simp only [beforeSplit, exec_cons, exec_nil, PMF.pure_bind, step,
    State.value, rollRange_one_two, PMF.bind_map, PMF.map_comp]
  refine bind_congr_support fun x hx => ?_
  have hrec : ∀ x, (initial.setRecord 0 { results := [x] }).record 0 =
      { results := [x] } := fun _ => rfl
  rcases mem_roll hx with rfl | rfl
  all_goals
    simp only [Function.comp, hrec, sum_single]
    rfl

/-- In every world, the state after the merge's instructions, with the dead
record and `@1` buried, is the initial state with `4 × x` in `@0`. -/
private theorem factor :
    ∀ v ∈ ((exec beforeSplit (PMF.pure initial)).map split₀).support,
      (exec beforeMerge (cond (exec beforeSplit (PMF.pure initial)) split₀
        v)).map (·.buryAll [.record 0, .register 1]) =
      lift (fun x (s : State) => s.setRegister 0 x) (PMF.pure (4 * v))
        (PMF.pure initial) := by
  intro v hv
  rw [exec_beforeSplit, PMF.map_comp, show split₀ ∘ rolled = id from rfl,
    PMF.map_id] at hv
  rw [exec_beforeSplit, cond_map (fun _ => rfl) hv]
  simp only [beforeMerge, exec_cons, exec_nil, PMF.pure_bind, PMF.pure_map,
    lift, step]
  rcases mem_roll hv with rfl | rfl <;> rfl

-- The worked example has the law of `4 × 1D2`. No other split is open, so the
-- worlds of the splits beneath and above are one each.
example :
    (exec (beforeSplit ++ beforeMerge ++ afterMerge) (PMF.pure initial)).map
      (·.value (.register 0)) = roll.map (4 * ·) := by
  rw [exec_merge beforeSplit beforeMerge afterMerge _ (PMF.pure ())
    (fun _ => exec beforeSplit (PMF.pure initial)) split₀ (.register 0)
    [.record 0, .register 1] (fun x (s : State) => s.setRegister 0 x)
    (fun _ => PMF.pure ()) (fun _ v _ => PMF.pure (4 * v))
    (fun _ _ => PMF.pure initial)
    (by rw [PMF.pure_bind]) (by decide)
    (fun _ _ v hv => by rw [PMF.pure_bind]; exact factor v hv),
    exec_beforeSplit, PMF.map_comp, show split₀ ∘ rolled = id from rfl,
    PMF.map_id]
  simp only [afterMerge, exec_cons, exec_nil, step, PMF.bind_pure, lift,
    PMF.pure_map, PMF.map_bind, PMF.bind_bind, PMF.pure_bind]
  rfl

/-! ## Merging beneath an open split

A function in which one split merges while another stays open beneath it, and
the rest of the state depends on the one beneath:

```text
0  ⚅0 <- roll range 1:2
1  @0 <- sum rolling record ⚅0
2  ⚅1 <- roll range 1:2
3  @1 <- sum rolling record ⚅1
4  @2 <- @0 + @1
5  @3 <- @0 * 2
6  @4 <- @2 * 3
7  @5 <- @2 + @4
8  @6 <- @3 + @5
9  return @6
```

Call the first roll `a`, and `@2`, which is `a` plus the second roll, `b`. Two
instructions read each of `@0` and `@2`, so the plan splits on `a` after
instruction 1 and on `b` after instruction 4. After instruction 7, `@5` is the
only live value that depends on `b`, so the split on `b` merges into `@5`,
while the split on `a` stays open beneath it, since `@3` and `@5` both depend
on `a`.

Merging over the whole law of states would require everything but `@5` to have
one law in every world of `b`. But `@3` is `2a`, and `a` and `b` are
correlated: `b = 4` forces `a = 2`. Within each world of `a`, though, `@3` is
certain, so the worlds of `b` differ only in `@5` and the dead locations, and
they merge there. So the law is that of `2a + 4b`, i.e., `6a + 4x` for the
second roll `x`.

The plan buries `@2` and `@4` at the merge, but the specification must also
bury `⚅1` and `@1`, which hold the second roll: within a world of `a`, it
determines `b`, and the rest reads neither.
-/

/-- The instructions of the function. -/
private def beneath : List Instruction :=
  [.rollRange 0 (.immediate 1) (.immediate 2), .sumRollingRecord 0 0,
    .rollRange 1 (.immediate 1) (.immediate 2), .sumRollingRecord 1 1,
    .binary .add 2 (.register 0) (.register 1),
    .binary .mul 3 (.register 0) (.immediate 2),
    .binary .mul 4 (.register 2) (.immediate 3),
    .binary .add 5 (.register 2) (.register 4),
    .binary .add 6 (.register 3) (.register 5),
    .return (.register 6)]

-- The plan splits on `a` and `b`, merges `b` into `@5` beneath `a`, and then
-- `a` into `@6`.
#guard ((List.range beneath.length).zip (Plan.new beneath).steps).filter
    (!·.2.isEmpty) == [
  (1, [.split (.register 0)]),
  (4, [.split (.register 2)]),
  (7, [.merge 1 (some (.register 5)) [.register 2, .register 4]]),
  (8, [.merge 0 (some (.register 6)) [.register 0, .register 2, .register 3,
    .register 4, .register 5]])]

/-- The instructions before the split on `b`. -/
private def beforeB : List Instruction := beneath.take 5

/-- The instructions between the split on `b` and its merge. -/
private def beforeMergeB : List Instruction := (beneath.drop 5).take 3

/-- The instructions after the merge of `b`. -/
private def afterMergeB : List Instruction := beneath.drop 8

/-- The initial state, with seven registers and two rolling records. -/
private def initialB : State :=
  { registers := #[0, 0, 0, 0, 0, 0, 0], records := #[{}, {}] }

/-- The state at the split on `b`, given `a` and `b`. -/
private def rolledB (a b : Int) : State :=
  { registers := #[a, b - a, b, 0, 0, 0, 0],
    records := #[{ results := [a] }, { results := [b - a] }] }

/-- The value of the split on `b`, `@2`. -/
private def splitB (s : State) : Int := s.value (.register 2)

/-- The world of `a` at the split on `b`. -/
private noncomputable def worldA (a : Int) : PMF State :=
  (roll.map (a + ·)).map (rolledB a)

/-- The state of the world of `a` once the split on `b` merges, less the
survivor `@5`: `@3` holds `2a`, and every other location that depends on `b`
is buried. -/
private def restB (a : Int) : State :=
  { registers := #[a, 0, 0, 2 * a, 0, 0, 0],
    records := #[{ results := [a] }, {}] }

/-- At the split on `b`, the law of states is the mixture over `a` of its
worlds. -/
private theorem exec_beforeB :
    exec beforeB (PMF.pure initialB) = roll.bind worldA := by
  simp only [beforeB, beneath, List.take, exec_cons, exec_nil, PMF.pure_bind,
    step, State.value, rollRange_one_two, PMF.bind_map, PMF.map_comp,
    PMF.bind_bind, Function.comp_def]
  refine bind_congr_support fun a ha => ?_
  rw [worldA, PMF.map_comp, ← PMF.bind_pure_comp]
  refine bind_congr_support fun x hx => ?_
  rcases mem_roll ha with rfl | rfl <;> rcases mem_roll hx with rfl | rfl
  all_goals
    simp [initialB, State.setRecord, State.setRegister, State.record,
      sum_single]
    rfl

/-- In the world of `a`, `b` is `a` plus the second roll. -/
private theorem map_worldA (a : Int) :
    (worldA a).map splitB = roll.map (a + ·) := by
  rw [worldA, PMF.map_comp, show splitB ∘ rolledB a = id from rfl, PMF.map_id]

/-- In every world of `a` and of `b`, the state after the merge's instructions,
with the locations that depend on `b` buried, is `restB a` with `4b` in `@5`. -/
private theorem factorB :
    ∀ a ∈ roll.support, ∀ v ∈ ((worldA a).map splitB).support,
      (exec beforeMergeB (cond (worldA a) splitB v)).map
          (·.buryAll [.record 1, .register 1, .register 2, .register 4]) =
        (PMF.pure ()).bind fun _ =>
          lift (fun x (s : State) => s.setRegister 5 x) (PMF.pure (4 * v))
            (PMF.pure (restB a)) := by
  intro a ha v hv
  rw [map_worldA] at hv
  rw [worldA, cond_map (fun _ => rfl) hv]
  rw [PMF.support_map] at hv
  obtain ⟨x, hx, rfl⟩ := hv
  simp only [beforeMergeB, beneath, List.drop, List.take, exec_cons, exec_nil,
    PMF.pure_bind, PMF.pure_map, lift, step]
  rcases mem_roll ha with rfl | rfl <;> rcases mem_roll hx with rfl | rfl <;>
    rfl

-- The function has the law of `6a + 4x`.
example :
    (exec beneath (PMF.pure initialB)).map (·.value (.register 6)) =
      roll.bind fun a => roll.map fun x => 6 * a + 4 * x := by
  rw [show beneath = beforeB ++ beforeMergeB ++ afterMergeB from rfl,
    exec_merge beforeB beforeMergeB afterMergeB _ roll worldA splitB
      (.register 6) [.record 1, .register 1, .register 2, .register 4]
      (fun x (s : State) => s.setRegister 5 x) (fun _ => PMF.pure ())
      (fun _ v _ => PMF.pure (4 * v)) (fun a _ => PMF.pure (restB a))
      exec_beforeB (by decide) factorB]
  simp only [afterMergeB, beneath, List.drop, exec_cons, exec_nil, step,
    PMF.pure_bind, lift, PMF.pure_map, PMF.map_bind, PMF.bind_bind,
    PMF.bind_pure, map_worldA, PMF.bind_map, Function.comp_def]
  refine bind_congr_support fun a ha => ?_
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun x hx => ?_
  rcases mem_roll ha with rfl | rfl <;> rcases mem_roll hx with rfl | rfl <;>
    rfl

/-! ## Docstring examples, verbatim -/

example (μ : PMF State) : exec [] μ = μ := rfl

end Xdy.Spec.Test
