import XdySpec.Semantics

/-!
# Semantics tests

The laws of ranges and dice, including their degenerate cases, proved of the
definitions, since a `PMF` cannot be computed.
-/

namespace Xdy.Spec.Test

/-! ## Ranges -/

example : rollRange 3 1 = PMF.pure { results := [0] } := by simp [rollRange]
example : rollRange 1 3 { results := [2] } = 1 / 3 := by
  simp [rollRange, PMF.map_apply, PMF.ofMultiset_apply,
    Multiset.count_eq_one_of_mem (Finset.nodup _)]

/-! ## Dice -/

example : standardDie 0 = PMF.pure 0 := by simp [standardDie]
example : standardDie (-3) = PMF.pure 0 := by simp [standardDie]
example : standardDie 6 3 = 1 / 6 := by
  simp [standardDie, PMF.ofMultiset_apply,
    Multiset.count_eq_one_of_mem (Finset.nodup _)]
example : customDie [] = PMF.pure 0 := by simp [customDie]
example : customDie [1, 1, 2] 1 = 2 / 3 := by
  simp [customDie, PMF.ofMultiset_apply]
example : customDie [1, 1, 2] 3 = 0 := by
  simp [customDie, PMF.ofMultiset_apply]

/-! ## Sets of dice -/

example (die : PMF Int) : rollDice 0 die = PMF.pure {} := rfl
example (die : PMF Int) : rollDice (-1) die = PMF.pure {} := rfl
-- One die records its face as the only result.
example (die : PMF Int) :
    rollDice 1 die = die.map fun x => { results := [x] } := by
  simp [rollDice, Nat.repeat, PMF.pure_bind]; rfl

/-! ## Docstring examples, verbatim -/

example : rollRange 3 1 = PMF.pure { results := [0] } := by
  simp [rollRange]
example : standardDie 0 = PMF.pure 0 := by simp [standardDie]
example : customDie [] = PMF.pure 0 := by simp [customDie]
example : customDie [1, 1, 2] 1 = 2 / 3 := by
  simp [customDie, PMF.ofMultiset_apply]
example (die : PMF Int) : rollDice (-1) die = PMF.pure {} := rfl

end Xdy.Spec.Test
