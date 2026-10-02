import XdySpec.Invariant
import XdySpecTest.Fixtures

/-!
# Forward pass tests

The forward pass replayed on two functions, to the laws that they answer, and
on a third, which splits a rolling record, to the law of its worlds: the
multisets of its results. Each
replay unfolds the pass instruction by instruction (`forward_succ`), with the
plan's facts decided by the kernel, and reads each law on which the answer
depends from the laws before it, back to the rolls. `run_eq_forward` then
gives the answer as the pass's law of the returned register.

The laws that the pass gives the values of these functions are those of the
instruction on the world of its operands (`Forward.destination`), which the
laws of `XdySpec/World.lean` compute: `law_rollRange`, `law_sumRollingRecord`,
`law_binary`, `law_binary_immediate_left` and `law_binary_immediate_right`.
-/

namespace Xdy.Spec.Test.Forward

/-! ## The sum of a roll -/

/-- The law of the sum of a rolling record of `1D2`, which the compiler
reduces to a range. -/
private noncomputable abbrev sumLaw : PMF Int := (rollRange 1 2).map Record.sum

/-- The sum of a rolling record of `1D2` is the roll, clamped. -/
private theorem sumLaw_eq : sumLaw = roll.map (add 0) := by
  rw [sumLaw, rollRange_one_two, PMF.map_comp]
  exact congrArg (PMF.map · _) (funext sum_single)

/-! ## A split that merges into its own location

`{x}@(1D2) + {x} * 3`, as the compiler writes it, as in
`XdySpecTest/Conditioning.lean`:

```text
0  ⚅0 <- roll range 1:2
1  @0 <- sum rolling record ⚅0
2  @1 <- 3 * @0
3  @0 <- @0 + @1
4  return @0
```

The pass splits on `@0` after the sum, and merges into the new `@0` after the
add, so it answers `4 × 1D2`: in the world of each outcome `x` of the split,
`@0` is `x + 3x`, a point mass, and the merge mixes it over the law of `x`.
-/

namespace Tripled

/-- The function. -/
private def tripled : Function where
  parameters := #[]
  externals := #[]
  registerCount := 2
  rollingRecordCount := 1
  instructions := #[
    .rollRange 0 (.immediate 1) (.immediate 2), .sumRollingRecord 0 0,
    .binary .mul 1 (.immediate 3) (.register 0),
    .binary .add 0 (.register 0) (.register 1),
    .return (.register 0)]

/-- The values of the function. -/
private abbrev vs : Values := ⟨tripled.instructions.toList⟩

/-- Its initial state. -/
private def initial : Xdy.State := { registers := #[0, 0], records := #[{}] }

/-- The function is well formed. -/
private theorem valid : tripled.validate = .ok () := by decide +kernel

/-- Its initial state is `initial`. -/
private theorem hinitial : tripled.initialState [] [] = .ok initial := rfl

/-- The pass before instruction `n`. -/
private noncomputable abbrev F (n : Nat) : Forward vs :=
  forward vs (.ofOracle initial) n

/-- The pass's base has every location. -/
private theorem has {n : Nat} (hn : n ≤ 5) (k : Nat) :
    (F n).base.Has (vs.location k) :=
  has_forward valid hinitial hn k

-- The plan splits on `@0` after instruction 1, and merges it into `@0` after
-- instruction 3.
#guard (Plan.new tripled.instructions.toList).steps == [[],
  [.split (.register 0)], [],
  [.merge 0 (some (.register 0)) [.register 1]], []]

/-- The pass after the roll. -/
private theorem F₁ : F 1 = (F 0).exec 0 (vs.instructions[0]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 0 = [] by decide +kernel]
  rfl

/-- The pass after the sum, which it splits. -/
private theorem F₂ :
    F 2 = ((F 1).exec 1 (vs.instructions[1]'(by decide))).split 1 := by
  rw [F, forward_succ _ _ rfl, show vs.steps 1 = [.split (.register 0)] by
    decide +kernel]
  rfl

/-- The pass after the product. -/
private theorem F₃ : F 3 = (F 2).exec 2 (vs.instructions[2]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 2 = [] by decide +kernel]
  rfl

/-- The pass after the add, which merges the split into it. -/
private theorem F₄ : F 4 =
    ((F 3).exec 3 (vs.instructions[3]'(by decide))).merge 1
      (some (.register 0)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 3 = [.merge 0
      (some (.register 0)) [.register 1]] by decide +kernel,
    show vs.openSplits 3 = [1] by decide +kernel]
  rfl

/-- The roll's law. -/
private theorem roll₁ (o : Outcomes vs) :
    (F 1).laws o (.record 0) = rollRange 1 2 := by
  rw [F₁]
  refine (Forward.exec_laws_self _ 0 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 0 ++ readLast vs 0).map
    vs.location = [] by decide +kernel]
  exact law_rollRange (has (by decide) 0)

/-- The split's law is the sum's. -/
private theorem K_x (o : Outcomes vs) : (F 2).K 1 o = sumLaw := by
  rw [F₂, Forward.split_K_self_of_register _ 1
    (show vs.location 1 = .register 0 by decide +kernel)]
  refine (Forward.exec_laws_self _ 1 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 1 ++ readLast vs 1).map
    vs.location = [.record 0] by decide +kernel]
  refine (law_sumRollingRecord (has (by decide) 1) (has (by decide) 0)).trans ?_
  rw [roll₁]

/-- In each world, `@0` is the split's outcome. -/
private theorem x₂ (o : Outcomes vs) :
    (F 2).laws o (.register 0) = PMF.pure (o 1) := by
  rw [F₂]
  exact Forward.split_laws_self _ 1 o

/-- The product leaves `@0` alone. -/
private theorem x₃ (o : Outcomes vs) :
    (F 3).laws o (.register 0) = PMF.pure (o 1) := by
  rw [F₃, Forward.exec_laws_of_ne _ _ _ (by decide), x₂]

/-- In each world, `@1` is three times the split's outcome. -/
private theorem threeX₃ (o : Outcomes vs) :
    (F 3).laws o (.register 1) = PMF.pure (mul 3 (o 1)) := by
  rw [F₃]
  refine (Forward.exec_laws_self _ 2 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 2 ++ readLast vs 2).map
    vs.location = [.register 0] by decide +kernel]
  refine (law_binary_immediate_left (has (by decide) 2)
    (has (by decide) 1)).trans ?_
  rw [x₂]
  exact PMF.pure_map _ _

/-- The merge mixes `x + 3x` over the split's law. -/
private theorem answer₄ (o : Outcomes vs) : (F 4).laws o (.register 0) =
    sumLaw.bind fun x => PMF.pure (add x (mul 3 x)) := by
  rw [F₄, Forward.merge_laws_self]
  change ((F 3).K 1 o).bind _ = _
  rw [show (F 3).K 1 o = (F 2).K 1 o by rw [F₃]; rfl, K_x]
  refine congrArg _ (funext fun x => ?_)
  refine (Forward.exec_laws_self _ 3 _ _).trans ?_
  rw [Forward.destination, show (readLive vs 3 ++ readLast vs 3).map
    vs.location = [.register 0, .register 1] by decide +kernel]
  refine (law_binary (by decide) (has (by decide) 3) (has (by decide) 1)
    (has (by decide) 2)).trans ?_
  rw [x₃, threeX₃]
  simp only [lift, Function.update_self]
  exact (PMF.pure_bind _ _).trans (PMF.pure_map _ _)

-- The function answers `4 × 1D2`.
example {p : PMF Int} (h : run tripled [] [] = .ok p) :
    p = roll.map (4 * ·) := by
  obtain ⟨_, _, hp⟩ := run_eq_forward h hinitial rfl
    (show vs.writer 4 (.register 0) = some 3 by decide +kernel)
  rw [hp, show tripled.instructions.size - 1 = 4 from rfl,
    show vs.openSplits 4 = [] by decide +kernel, splitOutcomes,
    List.foldl_nil, PMF.pure_bind]
  refine (answer₄ _).trans ?_
  rw [sumLaw_eq, PMF.bind_map, ← PMF.bind_pure_comp]
  refine bind_congr_support fun x hx => ?_
  rcases mem_roll hx with rfl | rfl <;> rfl

end Tripled

/-! ## A split that merges beneath another

The function of `XdySpecTest/Conditioning.lean` whose split on `b` merges
beneath its split on `a`:

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

Call the first roll `a`, the second `x`, and `@2`, which is `a + x`, `b`. The
pass splits on `a` after instruction 1 and on `b` after instruction 4. In each
world of both, `@3` is `2a` and `@5` is `b + 3b`, point masses; the merge
after instruction 7 mixes `@5` over `b`'s law in the world of `a`, which is
`x`'s shifted by `a`; and the merge after instruction 8 mixes `@6`, `2a` plus
that law, over `a`'s law. So the function answers `6a + 4x`.
-/

namespace Nested

/-- The values of `nested`, from `XdySpecTest/Fixtures.lean`. -/
private abbrev vs : Values := ⟨nested.instructions.toList⟩

/-- The pass before instruction `n`. -/
private noncomputable abbrev F (n : Nat) : Forward vs :=
  forward vs (.ofOracle nestedInitial) n

/-- The pass's base has every location. -/
private theorem has {n : Nat} (hn : n ≤ 10) (k : Nat) :
    (F n).base.Has (vs.location k) :=
  has_forward nested_valid nested_initialState hn k

/-- The pass after the first roll. -/
private theorem F₁ : F 1 = (F 0).exec 0 (vs.instructions[0]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 0 = [] by decide +kernel]
  rfl

/-- The pass after the first sum, `a`, which it splits. -/
private theorem F₂ :
    F 2 = ((F 1).exec 1 (vs.instructions[1]'(by decide))).split 1 := by
  rw [F, forward_succ _ _ rfl, show vs.steps 1 = [.split (.register 0)] by
    decide +kernel]
  rfl

/-- The pass after the second roll. -/
private theorem F₃ : F 3 = (F 2).exec 2 (vs.instructions[2]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 2 = [] by decide +kernel]
  rfl

/-- The pass after the second sum, `x`. -/
private theorem F₄ : F 4 = (F 3).exec 3 (vs.instructions[3]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 3 = [] by decide +kernel]
  rfl

/-- The pass after `b`, which it splits. -/
private theorem F₅ :
    F 5 = ((F 4).exec 4 (vs.instructions[4]'(by decide))).split 4 := by
  rw [F, forward_succ _ _ rfl, show vs.steps 4 = [.split (.register 2)] by
    decide +kernel]
  rfl

/-- The pass after `2a`. -/
private theorem F₆ : F 6 = (F 5).exec 5 (vs.instructions[5]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 5 = [] by decide +kernel]
  rfl

/-- The pass after `3b`. -/
private theorem F₇ : F 7 = (F 6).exec 6 (vs.instructions[6]'(by decide)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 6 = [] by decide +kernel]
  rfl

/-- The pass after `b + 3b`, which merges `b` into it. -/
private theorem F₈ : F 8 =
    ((F 7).exec 7 (vs.instructions[7]'(by decide))).merge 4
      (some (.register 5)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 7 = [.merge 1
      (some (.register 5)) [.register 2, .register 4]] by decide +kernel,
    show vs.openSplits 7 = [1, 4] by decide +kernel]
  rfl

/-- The pass after the answer, which merges `a` into it. -/
private theorem F₉ : F 9 =
    ((F 8).exec 8 (vs.instructions[8]'(by decide))).merge 1
      (some (.register 6)) := by
  rw [F, forward_succ _ _ rfl, show vs.steps 8 = [.merge 0
      (some (.register 6)) [.register 0, .register 2, .register 3,
        .register 4, .register 5]] by decide +kernel,
    show vs.openSplits 8 = [1] by decide +kernel]
  rfl

/-- The first roll's law. -/
private theorem rollA₁ (o : Outcomes vs) :
    (F 1).laws o (.record 0) = rollRange 1 2 := by
  rw [F₁]
  refine (Forward.exec_laws_self _ 0 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 0 ++ readLast vs 0).map
    vs.location = [] by decide +kernel]
  exact law_rollRange (has (by decide) 0)

/-- `a`'s law is the first sum's. -/
private theorem K_a (o : Outcomes vs) : (F 2).K 1 o = sumLaw := by
  rw [F₂, Forward.split_K_self_of_register _ 1
    (show vs.location 1 = .register 0 by decide +kernel)]
  refine (Forward.exec_laws_self _ 1 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 1 ++ readLast vs 1).map
    vs.location = [.record 0] by decide +kernel]
  refine (law_sumRollingRecord (has (by decide) 1) (has (by decide) 0)).trans ?_
  rw [rollA₁]

/-- In each world, `@0` is `a`'s outcome. -/
private theorem a₂ (o : Outcomes vs) :
    (F 2).laws o (.register 0) = PMF.pure (o 1) := by
  rw [F₂]
  exact Forward.split_laws_self _ 1 o

/-- The second roll's law. -/
private theorem rollX₃ (o : Outcomes vs) :
    (F 3).laws o (.record 1) = rollRange 1 2 := by
  rw [F₃]
  refine (Forward.exec_laws_self _ 2 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 2 ++ readLast vs 2).map
    vs.location = [] by decide +kernel]
  exact law_rollRange (has (by decide) 2)

/-- The second roll and its sum leave `@0` alone. -/
private theorem a₄ (o : Outcomes vs) :
    (F 4).laws o (.register 0) = PMF.pure (o 1) := by
  rw [F₄, Forward.exec_laws_of_ne _ _ _ (by decide), F₃,
    Forward.exec_laws_of_ne _ _ _ (by decide), a₂]

/-- `x`'s law is the second sum's, in every world. -/
private theorem x₄ (o : Outcomes vs) : (F 4).laws o (.register 1) = sumLaw := by
  rw [F₄]
  refine (Forward.exec_laws_self _ 3 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 3 ++ readLast vs 3).map
    vs.location = [.record 1] by decide +kernel]
  refine (law_sumRollingRecord (has (by decide) 3) (has (by decide) 2)).trans ?_
  rw [rollX₃]

/-- `b`'s law, in each world of `a`, is `x`'s shifted by `a`. -/
private theorem K_b (o : Outcomes vs) :
    (F 5).K 4 o = sumLaw.map (add (o 1)) := by
  rw [F₅, Forward.split_K_self_of_register _ 4
    (show vs.location 4 = .register 2 by decide +kernel)]
  refine (Forward.exec_laws_self _ 4 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 4 ++ readLast vs 4).map
    vs.location = [.register 0, .register 1] by decide +kernel]
  refine (law_binary (by decide) (has (by decide) 4) (has (by decide) 1)
    (has (by decide) 3)).trans ?_
  rw [a₄, x₄]
  exact PMF.pure_bind _ _

/-- Splitting on `b` leaves `@0` alone. -/
private theorem a₅ (o : Outcomes vs) :
    (F 5).laws o (.register 0) = PMF.pure (o 1) := by
  rw [F₅, Forward.split_laws_of_ne _ _ (by decide),
    Forward.exec_laws_of_ne _ _ _ (by decide), a₄]

/-- In each world, `@2` is `b`'s outcome. -/
private theorem b₅ (o : Outcomes vs) :
    (F 5).laws o (.register 2) = PMF.pure (o 4) := by
  rw [F₅]
  exact Forward.split_laws_self _ 4 o

/-- In each world, `@3` is `2a`. -/
private theorem twoA₆ (o : Outcomes vs) :
    (F 6).laws o (.register 3) = PMF.pure (mul (o 1) 2) := by
  rw [F₆]
  refine (Forward.exec_laws_self _ 5 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 5 ++ readLast vs 5).map
    vs.location = [.register 0] by decide +kernel]
  refine (law_binary_immediate_right (has (by decide) 5)
    (has (by decide) 1)).trans ?_
  rw [a₅]
  exact PMF.pure_map _ _

/-- `2a` leaves `@2` alone. -/
private theorem b₆ (o : Outcomes vs) :
    (F 6).laws o (.register 2) = PMF.pure (o 4) := by
  rw [F₆, Forward.exec_laws_of_ne _ _ _ (by decide), b₅]

/-- `3b` leaves `@3` alone. -/
private theorem twoA₇ (o : Outcomes vs) :
    (F 7).laws o (.register 3) = PMF.pure (mul (o 1) 2) := by
  rw [F₇, Forward.exec_laws_of_ne _ _ _ (by decide), twoA₆]

/-- `3b` leaves `@2` alone. -/
private theorem b₇ (o : Outcomes vs) :
    (F 7).laws o (.register 2) = PMF.pure (o 4) := by
  rw [F₇, Forward.exec_laws_of_ne _ _ _ (by decide), b₆]

/-- In each world, `@4` is `3b`. -/
private theorem threeB₇ (o : Outcomes vs) :
    (F 7).laws o (.register 4) = PMF.pure (mul (o 4) 3) := by
  rw [F₇]
  refine (Forward.exec_laws_self _ 6 _ o).trans ?_
  rw [Forward.destination, show (readLive vs 6 ++ readLast vs 6).map
    vs.location = [.register 2] by decide +kernel]
  refine (law_binary_immediate_right (has (by decide) 6)
    (has (by decide) 4)).trans ?_
  rw [b₆]
  exact PMF.pure_map _ _

/-- `b + 3b` and its merge leave `@3` alone. -/
private theorem twoA₈ (o : Outcomes vs) :
    (F 8).laws o (.register 3) = PMF.pure (mul (o 1) 2) := by
  rw [F₈, Forward.merge_laws_of_ne _ _ _ (by decide),
    Forward.exec_laws_of_ne _ _ _ (by decide), twoA₇]

/-- The merge of `b` mixes `b + 3b` over `b`'s law in the world of `a`. -/
private theorem fourB₈ (o : Outcomes vs) : (F 8).laws o (.register 5) =
    (sumLaw.map (add (o 1))).bind fun b => PMF.pure (add b (mul b 3)) := by
  rw [F₈, Forward.merge_laws_self]
  change ((F 7).K 4 o).bind _ = _
  rw [show (F 7).K 4 o = (F 5).K 4 o by rw [F₇, F₆]; rfl, K_b]
  refine congrArg _ (funext fun b => ?_)
  refine (Forward.exec_laws_self _ 7 _ _).trans ?_
  rw [Forward.destination, show (readLive vs 7 ++ readLast vs 7).map
    vs.location = [.register 2, .register 4] by decide +kernel]
  refine (law_binary (by decide) (has (by decide) 7) (has (by decide) 4)
    (has (by decide) 6)).trans ?_
  rw [b₇, threeB₇]
  simp only [lift, Function.update_self]
  exact (PMF.pure_bind _ _).trans (PMF.pure_map _ _)

/-- The merge of `a` mixes `2a` plus `@5`'s law over `a`'s law. -/
private theorem answer₉ (o : Outcomes vs) : (F 9).laws o (.register 6) =
    sumLaw.bind fun a => ((sumLaw.map (add a)).bind fun b =>
      PMF.pure (add b (mul b 3))).map (add (mul a 2)) := by
  rw [F₉, Forward.merge_laws_self]
  change ((F 8).K 1 o).bind _ = _
  rw [show (F 8).K 1 o = (F 2).K 1 o by
      rw [F₈, F₇, F₆, F₅]
      refine (congrFun (Forward.split_K_of_ne (u := 1) (v := 4) _
        (by decide)) o).trans ?_
      rw [F₄, F₃]
      rfl,
    K_a]
  refine congrArg _ (funext fun a => ?_)
  refine (Forward.exec_laws_self _ 8 _ _).trans ?_
  rw [Forward.destination, show (readLive vs 8 ++ readLast vs 8).map
    vs.location = [.register 3, .register 5] by decide +kernel]
  refine (law_binary (by decide) (has (by decide) 8) (has (by decide) 5)
    (has (by decide) 7)).trans ?_
  rw [twoA₈, fourB₈]
  simp only [lift, Function.update_self]
  exact PMF.pure_bind _ _

-- The function answers `6a + 4x`, for independent rolls `a` and `x` of `1D2`.
example {p : PMF Int} (h : run nested [] [] = .ok p) :
    p = roll.bind fun a => roll.map fun x => 6 * a + 4 * x := by
  obtain ⟨_, _, hp⟩ := run_eq_forward h nested_initialState rfl
    (show vs.writer 9 (.register 6) = some 8 by decide +kernel)
  rw [hp, show nested.instructions.size - 1 = 9 from rfl,
    show vs.openSplits 9 = [] by decide +kernel, splitOutcomes,
    List.foldl_nil, PMF.pure_bind]
  refine (answer₉ _).trans ?_
  rw [sumLaw_eq, PMF.bind_map]
  refine bind_congr_support fun a ha => ?_
  simp only [Function.comp_apply, PMF.map_comp, PMF.bind_map, PMF.map_bind,
    PMF.pure_map]
  rw [← PMF.bind_pure_comp]
  refine bind_congr_support fun x hx => ?_
  rcases mem_roll ha with rfl | rfl <;> rcases mem_roll hx with rfl | rfl <;>
    rfl

end Nested

/-! ## A split rolling record

`2D2`, which two sums read:

```text
0  ⚅0 <- roll standard dice 2D2
1  @0 <- sum rolling record ⚅0
2  @1 <- sum rolling record ⚅0
3  @0 <- @0 + @1
4  return @0
```

Two instructions read the record, so the plan splits on it after the roll. The
pass's worlds are the multisets of its results, as the Rust's are: three, not
the four rolls in order. `[2, 1]` is no world, and `[1, 2]` weighs what the
rolls `[1, 2]` and `[2, 1]` weigh together.
-/

namespace Doubled

/-- The function. -/
private def doubled : Function where
  parameters := #[]
  externals := #[]
  registerCount := 2
  rollingRecordCount := 1
  instructions := #[
    .rollStandardDice 0 (.immediate 2) (.immediate 2), .sumRollingRecord 0 0,
    .sumRollingRecord 1 0, .binary .add 0 (.register 0) (.register 1),
    .return (.register 0)]

/-- The values of the function. -/
private abbrev vs : Values := ⟨doubled.instructions.toList⟩

/-- Its initial state. -/
private def initial : Xdy.State := { registers := #[0, 0], records := #[{}] }

/-- The function is well formed. -/
private theorem valid : doubled.validate = .ok () := by decide +kernel

/-- Its initial state is `initial`. -/
private theorem hinitial : doubled.initialState [] [] = .ok initial := rfl

/-- The pass before instruction `n`. -/
private noncomputable abbrev F (n : Nat) : Forward vs :=
  forward vs (.ofOracle initial) n

/-- The pass's base has every location. -/
private theorem has {n : Nat} (hn : n ≤ 5) (k : Nat) :
    (F n).base.Has (vs.location k) :=
  has_forward valid hinitial hn k

-- The plan splits on the record after the roll.
#guard (Plan.new doubled.instructions.toList).steps.head? ==
  some [.split (.record 0)]

/-- The pass after the roll, which it splits. -/
private theorem F₁ :
    F 1 = ((F 0).exec 0 (vs.instructions[0]'(by decide))).split 0 := by
  rw [F, forward_succ _ _ rfl, show vs.steps 0 = [.split (.record 0)] by
    decide +kernel]
  rfl

/-- The split's law is the roll's, sorted. -/
private theorem K_roll (o : Outcomes vs) :
    (F 1).K 0 o = (rollDice 2 (standardDie 2)).map Record.sort := by
  rw [F₁, Forward.split_K_self]
  refine congrArg (PMF.map _) ((Forward.exec_laws_self _ 0 _ o).trans ?_)
  rw [Forward.destination, show (readLive vs 0 ++ readLast vs 0).map
    vs.location = [] by decide +kernel]
  refine (law_rollStandardDice (c := .immediate 2) (n := .immediate 2)
    (has (n := 0) (by decide) 0) (by simp [pairLocations,
      operandLocations])).trans ?_
  simp [pairLaw, operandLaw, lift, PMF.pure_bind, PMF.pure_map]

-- The law of the results of each world's record: the law of the results of
-- the roll, sorted.
private theorem results_K (o : Outcomes vs) :
    ((F 1).K 0 o).map Record.results =
      (Roll.standard 2 2).law.map Record.sorted := by
  rw [K_roll, PMF.map_comp]
  rfl

-- `[1, 2]` weighs two of the four rolls: `[1, 2]` and `[2, 1]`.
example (o : Outcomes vs) : (((F 1).K 0 o).map Record.results) [1, 2] =
    2 / 4 := by
  rw [results_K, Roll.map_sorted_law (.standard 2 2) trivial,
    show OrderStatistics.integral (Roll.standard 2 2).outcomes
      (fun rs => if Roll.expand rs = [1, 2] then 1 else 0) = 2 by decide,
    show (Roll.standard 2 2).total = 4 by decide]
  rfl

-- `[2, 1]` is no world.
example (o : Outcomes vs) : (((F 1).K 0 o).map Record.results) [2, 1] = 0 := by
  rw [results_K, Roll.map_sorted_law (.standard 2 2) trivial,
    show OrderStatistics.integral (Roll.standard 2 2).outcomes
      (fun rs => if Roll.expand rs = [2, 1] then 1 else 0) = 0 by decide]
  simp

-- In every world, the invariant holds, so the record's outcome is sorted.
example {o : Outcomes vs}
    (ho : o ∈ (splitOutcomes (F 1).K (vs.openSplits 1)).support) :
    Record.sort (o 0) = o 0 :=
  (invariant_run valid hinitial (pc := 1) (by decide)).sortContent_outcome ho 0

end Doubled

end Xdy.Spec.Test.Forward
