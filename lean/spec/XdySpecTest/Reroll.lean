import XdySpec.Reroll

/-!
# Reroll tests

The effect of a reroll on a die, with its weights and padding against the Rust's
closed form; the rewrite of alike dice, the survivors and pools of dice after a
drop, and each branch of a setup's reroll; and the laws of a setup's and a
mixture's reroll, as every later instruction views them.
-/

namespace Xdy.Spec.Test.Reroll

/-! ## Effects -/

-- A limited reroll pads each die by `Fⁿ`, and weighs a face that matches by
-- `mʲ` and one that does not by `unmatchedPaths F m j + mʲ`, `j = n - 1`.
#guard Effect.resolve (.standard 6) (.within 1 1) (some 1) ==
  .redraw 6 [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
#guard Effect.resolve (.standard 6) (.within 1 2) (some 3) ==
  .redraw 216 [(1, 4), (2, 4), (3, 52), (4, 52), (5, 52), (6, 52)]
-- An unlimited reroll pads each die by `F − m`, and keeps the faces that do
-- not match.
#guard Effect.resolve (.standard 6) (.within 1 2) none ==
  .redraw 4 [(3, 1), (4, 1), (5, 1), (6, 1)]
-- A custom die counts its faces by position.
#guard Effect.resolve (.ofFaces [1, 1, 2]) (.within 1 1) none ==
  .redraw 1 [(2, 1)]
-- No match, no limit, or one face does nothing; matching every face without
-- limit crosses out.
#guard Effect.resolve (.standard 6) (.within 7 9) (some 2) == .nothing
#guard Effect.resolve (.standard 6) (.within 1 1) (some 0) == .nothing
#guard Effect.resolve (.standard 1) (.within 1 1) (some 3) == .nothing
#guard Effect.resolve (.standard 0) (.among [0]) none == .crossOut
#guard Effect.resolve (.standard 6) (.outside 7 9) none == .crossOut
#guard Effect.resolve .none (.within 1 1) none == .nothing

-- The final results of a matched die weigh the padding in all.
#guard (List.range 5).all fun n =>
  (Roll.expand (limitedLaw (.standard 6) (.within 1 2) n)).length ==
    6 ^ (n - 1) * 6
#guard (Roll.expand (unlimitedLaw (.ofFaces [1, 1, 2, 5]) (.within 1 1))).length
  == 2

-- The paths agree with the Rust's closed forms.
#guard (List.range 6).all fun j =>
  unmatchedPaths 6 2 j * (6 - 2) == (6 ^ j - 2 ^ j) * 6
#guard (List.range 6).all fun j => unmatchedPaths 4 4 j == j * 4 ^ j

/-! ## Rewrites -/

-- Two dice rerolled once on `1`: the face `1` survives only by rerolling it.
#guard (Roll.standard 2 6).rewrite (.within 1 1) 6
    [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)] ==
  .custom 2 [(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
-- A range is one die, which an unlimited reroll conditions.
#guard (Roll.range 1 4).rewrite (.outside 2 3) 2 [(2, 1), (3, 1)] ==
  .custom 1 [(2, 4), (3, 4)]
-- Faces that no die shows contribute nothing of the law.
#guard rewriteFaces [(2, 1), (3, 1)] (.within 1 1) 6 [(1, 1), (2, 5)] ==
  [(2, 6), (3, 6)]

/-! ## Pools -/

#guard survivors [(1, 2), (3, 1), (5, 2)] 0 0 (.within 3 3) ==
  ([(1, 2), (5, 2)], 1)
#guard survivors [(1, 2), (3, 1), (5, 2)] 2 2 (.within 3 3) == ([], 1)
#guard survivors [(1, 2), (3, 1), (5, 2)] 3 0 (.within 1 1) == ([(5, 2)], 0)

/-! ## Setups -/

/-- Two two-sided dice, the lowest dropped. -/
private def keepHigh : Setup := { roll := some (.standard 2 2), lowest := 1 }

-- A reroll after a drop pools the multisets of the die still kept, of the
-- total `2² · 2`, of which `1` shows `1`, as `2d2kh1 reroll = 1` does.
#guard keepHigh.reroll (.within 1 1) (some 1) ==
  { roll := some (.pool [([(1, 1)], 1), ([(2, 1)], 1), ([(2, 1)], 4),
      ([(2, 1)], 2)]),
    die := some (.range 1 2) }
-- A reroll of alike dice without drops rewrites them, keeping their die.
#guard ({ roll := some (.standard 2 6) } : Setup).reroll (.within 1 1)
    (some 1) ==
  { roll := some (.custom 2 [(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]),
    die := some (.standard 6) }
-- A reroll that matches every face without limit drops every die still kept.
#guard ({ roll := some (.standard 2 6), highest := 1 } : Setup).reroll
    (.within 1 6) none ==
  { roll := some (.standard 2 6), lowest := 2, highest := 1 }
-- A reroll that does nothing, or a setup that keeps nothing, stays.
#guard ({ roll := some (.standard 2 6) } : Setup).reroll (.within 7 7) none ==
  { roll := some (.standard 2 6) }
#guard ({ roll := some (.standard 2 6), lowest := 2 } : Setup).reroll
    (.within 1 1) (some 1) == { roll := some (.standard 2 6), lowest := 2 }
#guard ({} : Setup).reroll (.within 1 1) none == {}

/-! ## Laws -/

/-- Two six-sided dice. -/
private def twoD6 : Setup := { roll := some (.standard 2 6) }

/-- Two six-sided dice reroll as specified by any criterion and limit, since
they are alike and drop nothing. -/
private theorem twoD6_rerolls (c : Criterion) (limit : Option Nat) :
    twoD6.Rerolls c limit :=
  Setup.law_reroll c limit (fun _ h => by cases h; trivial)
    (Setup.faithful_roll rfl) (fun _ h => by cases h; trivial)
    fun ⟨_, _, _, hρ, _, _, hal⟩ =>
      absurd (by cases hρ; exact ⟨rfl, rfl, rfl⟩) hal

-- Rerolling two dice once on `1` describes the specification's reroll.
example : twoD6.Rerolls (.within 1 1) (some 1) := twoD6_rerolls _ _

-- Rerolling every face of three dice without limit, after a drop, describes
-- the specification's cross out, which leaves nothing kept.
example : ({ roll := some (.standard 3 6), highest := 1 } : Setup).Rerolls
    (.within 1 6) none := by
  have hx : Effect.resolve
      ({ roll := some (.standard 3 6), highest := 1 } : Setup).replacement
      (.within 1 6) none = .crossOut := by decide
  exact Setup.law_reroll _ _ (fun _ h => by cases h; trivial)
    (Setup.Faithful.of_eq (s := { roll := some (.standard 3 6) }) rfl rfl
      (Setup.faithful_roll rfl))
    (fun _ h => by cases h; trivial)
    fun ⟨_, _, _, _, _, he, _⟩ => by
      rw [hx] at he
      cases he

-- A mixture of two dice, rerolled by independent criteria and limits,
-- describes the specification's reroll of the dice.
example (κ : PMF (Criterion × Option Nat)) :
    (mixtureLaw (lift (fun s q => s.reroll q.1 q.2) (PMF.pure twoD6) κ)).map
        Record.view =
      ((lift Prod.mk ((rollDice 2 (standardDie 6)).map fun r =>
          { r with die := .standard 6 }) κ).bind fun p =>
        reroll p.1 p.2.1 p.2.2).map Record.view := by
  rw [mixtureLaw_reroll (fun s hs => by
      rw [PMF.support_pure, Set.mem_singleton_iff] at hs
      subst hs
      exact ⟨fun _ h => by cases h; trivial, Setup.faithful_roll rfl,
        fun _ h => by cases h; trivial⟩)
    (fun s hs q _ _ => by
      rw [PMF.support_pure, Set.mem_singleton_iff] at hs
      subst hs
      exact twoD6_rerolls q.1 q.2),
    mixtureLaw_pure, twoD6, Setup.law_roll]
  rfl

-- A rewrite leaves its setup ready for another reroll.
example : (twoD6.reroll (.within 1 1) (some 1)).Faithful :=
  (Setup.reroll_of_not_pools _ _ (fun _ h => by cases h; trivial)
    (Setup.faithful_roll rfl) (fun _ h => by cases h; trivial)
    fun ⟨_, _, _, hρ, _, _, hal⟩ =>
      absurd (by cases hρ; exact ⟨rfl, rfl, rfl⟩) hal).2.1

-- A die of the weighted faces of a limited reroll's law is the limited
-- replacement.
example : weightedDie (limitedLaw (.standard 6) (.within 1 1) 2) =
    limitedReplacement (.within 1 1) (customDie (Die.standard 6).faces) 2 :=
  weightedDie_limitedLaw _ _ (by decide)

/-! ## Docstring examples, verbatim -/

#guard tally [3, 1, 3] == [(1, 1), (3, 2)]
#guard unmatchedPaths 6 1 2 == 36 + 6
#guard Effect.resolve (.standard 6) (.within 1 1) (some 2) ==
  .redraw 36 [(1, 1), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
#guard Effect.resolve (.standard 6) (.within 1 6) none == .crossOut
#guard Effect.resolve (.standard 6) (.within 7 7) none == .nothing
#guard (Roll.standard 2 3).dice == (2, [(1, 1), (2, 1), (3, 1)])
#guard (Roll.range 4 3).dice == (1, [(0, 1)])
#guard rewriteFaces [(1, 1), (2, 1)] (.within 1 1) 2 [(1, 1), (2, 1)] ==
  [(1, 1), (2, 3)]
#guard survivors [(1, 2), (3, 1), (5, 2)] 1 1 (.within 3 3) ==
  ([(1, 1), (5, 1)], 1)
#guard join [(1, 1), (3, 2)] [(2, 1), (3, 1)] == [(1, 1), (2, 1), (3, 3)]

end Xdy.Spec.Test.Reroll
