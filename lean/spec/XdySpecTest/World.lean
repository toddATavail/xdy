import XdySpec.World

/-!
# World tests

Reading and writing locations, and the laws of worlds applied to two
independent rolls: reading one back, conditioning on one, mixing a split back
together, splitting one off in the form that lemma 6 takes, burying the record
that a sum has read, and adding the two, which is lemma 1 on states. Then the
law of each instruction on its operands, as `propagation.rs` computes it.
-/

namespace Xdy.Spec.Test.World

/-! ## Reading and writing -/

/-- A state with three registers and one rolling record. -/
private def base : State := { registers := #[0, 0, 0], records := #[{}] }

#guard (base.set (.register 1) 5).get (.register 1) == 5
#guard (base.set (.register 1) 5).get (.register 0) == 0
#guard ((base.set (.record 0) { results := [3] }).get (.record 0)).results ==
  [3]
-- A register out of range is not written, as `setRegister` leaves it.
#guard (base.set (.register 3) 5).get (.register 3) == 0
#guard (base.bury (.register 1)).get (.register 1) == 0

/-! ## Two independent rolls -/

/-- The law of a roll of a range from `1` to `2`. -/
private noncomputable def roll : PMF Int :=
  PMF.ofMultiset (Finset.Icc (1 : Int) 2).val (by simp)

/-- Every register holds a roll; every record is empty. -/
private noncomputable def rolls : (ℓ : Location) → PMF (Content ℓ)
  | .register _ => roll
  | .record _ => PMF.pure {}
  | .answer => PMF.pure ()

/-- The base has every register. -/
private theorem has (r : Nat) (h : r < 3) : base.Has (.register r) := h

-- Each register of a world of two rolls holds a roll.
example :
    (product base [.register 0, .register 1] rolls).map
      (·.get (.register 1)) = roll :=
  map_get_product (by decide) (by simp) (has 1 (by decide))

-- Conditioning on the first roll makes it a point mass, and leaves the
-- second alone.
example :
    cond (product base [.register 0, .register 1] rolls)
        (·.get (.register 0)) 2 =
      product base [.register 0, .register 1]
        (Function.update rolls (.register 0) (PMF.pure 2)) :=
  cond_product (by decide) (by simp) (has 0 (by decide))
    (by simp [rolls, roll])

-- A point mass folds into the base.
example :
    product base [.register 0, .register 1]
        (Function.update rolls (.register 0) (PMF.pure 2)) =
      product (base.set (.register 0) 2) [.register 1] rolls :=
  product_pure (live := [.register 0, .register 1]) 2 (by decide) (by simp)

-- Mixing the worlds of a split on the first roll undoes the split.
example :
    (roll.bind fun v => product base [.register 0, .register 1]
      (Function.update rolls (.register 0) (PMF.pure v))) =
      product base [.register 0, .register 1] rolls := by
  rw [bind_product (by decide) (by simp) roll _ rolls
    fun _ _ _ _ h => Function.update_of_ne h _ _]
  simp only [Function.update_self, PMF.bind_pure]
  exact congrArg _ (Function.update_eq_self _ _)

-- Either roll splits off, combined independently with the other, as lemma 6
-- takes the survivor of a merge.
example :
    product base [.register 0, .register 1] rolls =
      lift (fun x s => s.set (.register 1) x) roll
        (product base [.register 0] rolls) :=
  product_eq_lift (live := [.register 0, .register 1]) (ℓ := .register 1)
    (by decide) (by simp)

-- Burying a record that holds a roll's result, as the sum of `1D2` leaves
-- it, forgets how the result was rolled: only the register is live.
example :
    (roll.map fun x => (base.set (.record 0) { results := [x] }).set
      (.register 0) x).map (·.bury (.record 0)) =
      product (base.bury (.record 0)) [.register 0] rolls := by
  simp only [PMF.map_comp, Function.comp_def, product_cons, product_nil,
    lift, PMF.pure_map, State.bury_set_ne (s := base.set _ _)
      (show Location.register 0 ≠ .record 0 by decide),
    State.bury_set_self]
  rw [← PMF.bind_pure_comp]
  rfl

-- Adding the two rolls, and burying them, leaves a world whose only live
-- location is the sum, with the law of lemma 1's lift.
example :
    ((product base [.register 0, .register 1] rolls).bind
        (step · (.binary .add 2 (.register 0) (.register 1)))).map
          (·.buryAll [.register 0, .register 1]) =
      product (base.buryAll [.register 0, .register 1]) [.register 2]
        (Function.update rolls (.register 2)
          (lift BinaryOp.add.apply roll roll)) := by
  have h := bind_step_product (base := base) (laws := rolls)
    (S := [.register 0, .register 1]) (T := [])
    (i := .binary .add 2 (.register 0) (.register 1)) (by decide) (by simp)
    (by simp) (has 2 (by decide))
  simp only [Instruction.destination, List.append_nil] at h
  rw [law_binary (by decide) (has 2 (by decide)) (has 0 (by decide))
    (has 1 (by decide))] at h
  exact h

-- Mixing a world over the outcomes of its first roll, each a point mass,
-- leaves it alone, as a split does.
example :
    (roll.bind fun v => product base [.register 0, .register 1]
      (Function.update rolls (.register 0) (PMF.pure v))) =
      product base [.register 0, .register 1] rolls :=
  bind_pure_product (base := base) (laws := rolls)
    (live := [.register 0, .register 1]) (ℓ := .register 0) (by decide)
    (by simp)

-- A world of point masses is a point mass.
example : ∃ t, product base [.register 0, .record 0] (Function.update rolls
    (.register 0) (PMF.pure 2)) = PMF.pure t :=
  product_eq_pure fun ℓ h => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl
    · exact ⟨2, Function.update_self ..⟩
    · exact ⟨{}, rfl⟩

-- Adding a split value, fixed at `2`, to a roll, and burying the roll,
-- leaves the split live beside the sum.
example :
    ((product base [.register 0, .register 1]
        (Function.update rolls (.register 0) (PMF.pure 2))).bind
          (step · (.binary .add 2 (.register 0) (.register 1)))).map
            (·.buryAll [.register 1]) =
      product (base.buryAll [.register 1]) [.register 0, .register 2]
        (Function.update (Function.update rolls (.register 0) (PMF.pure 2))
          (.register 2)
          (((product base [.register 0, .register 1]
            (Function.update rolls (.register 0) (PMF.pure 2))).bind
              (step · (.binary .add 2 (.register 0) (.register 1)))).map
                (·.get (.register 2)))) :=
  bind_step_product_pure (P := [.register 0]) (S := [.register 1]) (T := [])
    (by decide) (by simp) (by decide) (by simp) (by simp) (has 2 (by decide))

-- Burying a register that an instruction overwrites, without reading it,
-- changes nothing.
example (μ : PMF State) :
    ((μ.map (·.buryAll [.register 1])).bind
        (step · (.neg 1 (.register 0)))).map (·.buryAll []) =
      (μ.bind (step · (.neg 1 (.register 0)))).map (·.buryAll []) :=
  map_buryAll_bind_step (by decide) (by decide)

/-! ## The law of each instruction -/

/-- The base has its rolling record. -/
private theorem hasRecord : base.Has (.record 0) :=
  show 0 < base.records.size by decide

-- Both operands of a pair are one register: the pair shares its value.
example : pairLaw rolls (.register 0) (.register 0) =
    roll.map fun x => (x, x) := by
  simp [pairLaw, rolls]

-- Distinct registers are independent.
example : pairLaw rolls (.register 0) (.register 1) = lift Prod.mk roll roll :=
  by simp [pairLaw, rolls]

-- An immediate is a point mass.
example : pairLaw rolls (.immediate 3) (.register 1) =
    lift Prod.mk (PMF.pure 3) roll := rfl

-- Reading two registers reads their joint law.
example :
    (product base (pairLocations (.register 0) (.register 1)) rolls).map
        (fun s => (s.value (.register 0), s.value (.register 1))) =
      lift Prod.mk roll roll := by
  rw [map_value_product_pair (by
    simp [pairLocations, has 0 (by decide), has 1 (by decide)])]
  simp [pairLaw, rolls]

-- Negating a roll negates its law.
example :
    ((product base (operandLocations (.register 0)) rolls).bind
        (step · (.neg 1 (.register 0)))).map (·.get (.register 1)) =
      roll.map neg :=
  law_neg (has 1 (by decide)) (by simp [operandLocations, has 0 (by decide)])

-- Doubling a roll as `x + x` maps it, rather than adding an independent copy.
example :
    ((product base [.register 0] rolls).bind
        (step · (.binary .add 2 (.register 0) (.register 0)))).map
          (·.get (.register 2)) = roll.map fun x => add x x :=
  law_binary_self (has 2 (by decide)) (has 0 (by decide))

-- A range from a roll to `6` mixes the ranges from each of its outcomes.
example :
    ((product base (pairLocations (.register 0) (.immediate 6)) rolls).bind
        (step · (.rollRange 0 (.register 0) (.immediate 6)))).map
          (·.get (.record 0)) = roll.bind fun x => rollRange x 6 := by
  rw [law_rollRange_operands hasRecord
    (by simp [pairLocations, operandLocations, has 0 (by decide)])]
  simp [pairLaw, operandLaw, rolls, lift, PMF.bind_map, Function.comp_def]

-- A roll of dice of a roll's count mixes the dice of each count.
example :
    ((product base (pairLocations (.register 0) (.immediate 6)) rolls).bind
        (step · (.rollStandardDice 0 (.register 0) (.immediate 6)))).map
          (·.get (.record 0)) =
      roll.bind fun x => rollDice x (standardDie 6) := by
  rw [law_rollStandardDice hasRecord
    (by simp [pairLocations, operandLocations, has 0 (by decide)])]
  simp [pairLaw, operandLaw, rolls, lift, PMF.bind_map, Function.comp_def]

-- So does a roll of custom dice.
example :
    ((product base (operandLocations (.register 0)) rolls).bind
        (step · (.rollCustomDice 0 (.register 0) [1, 1, 2]))).map
          (·.get (.record 0)) =
      roll.bind fun x => rollDice x (customDie [1, 1, 2]) :=
  law_rollCustomDice hasRecord
    (by simp [operandLocations, has 0 (by decide)])

-- Dropping a roll's count of the lowest results of an empty record drops each
-- count independently of the record.
example :
    ((product base (.record 0 :: operandLocations (.register 0)) rolls).bind
        (step · (.dropLowest 0 (.register 0)))).map (·.get (.record 0)) =
      lift Record.dropLowest (PMF.pure {}) roll :=
  law_dropLowest hasRecord
    (by simp [operandLocations, has 0 (by decide)])

-- So does dropping the highest.
example :
    ((product base (.record 0 :: operandLocations (.register 0)) rolls).bind
        (step · (.dropHighest 0 (.register 0)))).map (·.get (.record 0)) =
      lift Record.dropHighest (PMF.pure {}) roll :=
  law_dropHighest hasRecord
    (by simp [operandLocations, has 0 (by decide)])

/-! ## Docstring examples, verbatim -/

example (base : State) (laws : (ℓ : Location) → PMF (Content ℓ)) :
    product base [] laws = PMF.pure base := rfl

#guard operandLocations (.register 3) == [.register 3]
#guard operandLocations (.immediate 3) == []

example (laws : (ℓ : Location) → PMF (Content ℓ)) :
    operandLaw laws (.immediate 3) = PMF.pure 3 := rfl

#guard pairLocations (.register 1) (.register 1) == [.register 1]
#guard pairLocations (.register 1) (.register 2) == [.register 1, .register 2]
#guard pairLocations (.immediate 1) (.register 2) == [.register 2]

example (laws : (ℓ : Location) → PMF (Content ℓ)) :
    pairLaw laws (.register 1) (.register 1) =
      (laws (.register 1)).map fun x => (x, x) := by
  simp [pairLaw]

end Xdy.Spec.Test.World
