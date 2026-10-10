import XdySpec.Stacks

/-!
# Stacks tests

The laws of the kept chains of setups after drops and keeps, as the kept pool
of their roll's multisets of chains weighs them: of dice, whose chains are of
one die, and of i.i.d. chains that an explode grew.
-/

namespace Xdy.Spec.Test.Stacks

open Xdy (Chain)

/-! ## Dice -/

/-- Three two-sided dice, the lowest dropped. -/
private def threeD2 : Setup := { roll := some (.standard 3 2), lowest := 1 }

/-- The results of two-sided dice lie within the `i32` range. -/
private theorem standard_normal : (Roll.standard 3 2).Normal := by
  intro r hr x hx
  have hf : x ∈ (Die.standard 2).faces := Setup.faithful_roll
    (ρ := .standard 3 2) rfl r (by rw [Setup.law_roll]; exact hr) x hx
  rw [show (Die.standard 2).faces = [1, 2] by decide] at hf
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> simp [i32Min, i32Max]

-- Dropping the lowest of three two-sided dice keeps the higher two of each
-- outcome.
#guard keptPool (Roll.standard 3 2).stacks 1 0 ==
  [([(.single 1, 2)], 1), ([(.single 1, 1), (.single 2, 1)], 3),
    ([(.single 2, 2)], 3), ([(.single 2, 2)], 1)]

-- Their kept chains have the law of the kept pool.
example : threeD2.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] (keptPool (Roll.standard 3 2).stacks 1 0)).map
      (fun ms => (Roll.expand ms : Multiset Chain)) :=
  Setup.map_keptChains_law rfl trivial standard_normal trivial

/-! ## Chains -/

/-- The law of a chain of a two-sided die exploded on `2` once: a `1`, or a
`2` extended by a `2`, as the explode leaves the chains that a later reroll
reads, the extension by a `1` aside. -/
private def law : List (Chain × Nat) := [(.single 1, 1), (.live [2] 2, 1)]

/-- Two such chains, the lowest dropped. -/
private def twoChains : Setup :=
  { roll := some (.chains 2 law), die := some (.standard 2), lowest := 1 }

-- The higher chain is the chain of two dice unless both are of one die.
#guard keptPool (Roll.chains 2 law).stacks 1 0 ==
  [([(.single 1, 1)], 1), ([(.live [2] 2, 1)], 2), ([(.live [2] 2, 1)], 1)]

/-- The chains are normal. -/
private theorem law_normal : (Roll.chains 2 law).Normal := by
  intro e he
  simp only [law, List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl <;>
    simp [NormalChain, Chain.single, i32Min, i32Max]

-- Their kept chains have the law of the kept pool.
example : twoChains.law.map (fun r => (r.keptChains : Multiset Chain)) =
    (weighted [] (keptPool (Roll.chains 2 law).stacks 1 0)).map
      (fun ms => (Roll.expand ms : Multiset Chain)) :=
  Setup.map_keptChains_law rfl ⟨by decide, by decide⟩ law_normal trivial

end Xdy.Spec.Test.Stacks
