import XdySpec.Paths

/-!
# Path enumeration tests

The forks of ranges, dice, rerolls and explodes, including their degenerate
cases and records that hold chains, and the number of paths of whole
functions.
-/

namespace Xdy.Spec.Test

/-- A state with three registers and two empty rolling records. -/
private def blankState : State :=
  { registers := #[0, 0, 0], records := #[{}, {}] }

/-! ## Forks -/

-- An empty range forks once, at `0`.
#guard (rangeForks 3 1).map (·.results) == [[0]]
#guard (rangeForks (-1) 1).map (·.results) == [[-1], [0], [1]]
-- A die without faces forks once, at `0`.
#guard standardFaces (-2) == [0]
#guard customFaces [] == [0]
-- Repeated faces of a custom die fork apart.
#guard (diceForks 2 (customFaces [1, 1])).length == 4
-- A count of `0` or less forks once, with no results.
#guard (diceForks (-3) [1, 2]).map (·.results) == [[]]
-- Every instruction but a roll forks once.
#guard (forks blankState (.binary .add 0 (.immediate 1) (.immediate 2))).length
  == 1
#guard (forks blankState (.return (.register 0))).length == 1
#guard (forks blankState (.keepLowest 0 (.immediate 1))).length == 1
#guard (forks blankState (.keepHighest 1 (.register 2))).length == 1

/-! ## Rerolls -/

-- A limited reroll forks a matching die by each face of each draw, until a
-- draw no longer matches or the limit runs out.
#guard limitedForks (.within 1 1) [1, 2, 3] 1 == [1, 2, 3]
#guard limitedForks (.within 1 1) [1, 2, 3] 3 == [1, 2, 3, 2, 3, 2, 3]
-- A reroll forks only the dice that it rerolls.
#guard (rerollForks { results := [1, 2, 1], die := .standard 2 } (.within 1 1)
  (some 1)).length == 4
#guard (rerollForks { results := [1, 2, 1], die := .standard 2 } (.within 1 1)
  (some 0)).map (·.results) == [[1, 1, 2]]
-- An unlimited reroll forks a matching die by each face that does not match,
-- by position, and once if every face matches.
#guard (rerollForks { results := [1], die := .ofFaces [1, 2, 2] }
  (.within 1 1) none).map (·.results) == [[2], [2]]
#guard (rerollForks { results := [1, 1], die := .standard 1 } (.within 1 1)
  none).map (·.results) == [[]]
-- A reroll forks the dice that the drops kept, and forgets the drops.
#guard (rerollForks { results := [1, 2, 3], lowest := 1, die := .standard 3 }
  (.within 2 2) none).map (fun r => (r.lowest, r.results))
  == [(0, [1, 3]), (0, [3, 3])]
-- A record never rolled forks once.
#guard (forks blankState (.reroll 0 (.eq (.immediate 0)) .unlimited)).length
  == 1

/-! ## Rerolls of chains -/

-- A reroll of a chain forks each die that it rerolls by each replacement, in
-- its place, and leaves a saturated chain alone.
#guard (chainForks (· == 1) [1, 2] (.live [1, 3] 1)).length == 4
#guard chainForks (fun _ => true) [1, 2] (.saturated 0) == [.saturated 0]
/-- `2D2 explode = 2`, as the chain `[2, 1]` beside a `2`. -/
private def exploded : Record :=
  { results := [3, 2], die := .standard 2
    chains := [.live [2] 1, .single 2] }

-- A reroll of a record that holds chains forks each die of each kept chain.
#guard (rerollForks exploded (.within 1 1) (some 1)).map (·.results)
  == [[2, 3], [2, 4]]
-- An unlimited reroll that matches every face crosses out every chain that may
-- still grow.
#guard (rerollForks exploded (.within 1 2) none).map (·.results) == [[]]

/-! ## Explodes -/

-- A limited explode forks a chain that it grows by each face of each draw,
-- until a draw no longer matches or the limit runs out.
#guard extensionForks (.within 6 6) [1, 2, 3, 4, 5, 6] 1
  == [1, 2, 3, 4, 5, 6].map ([], ·)
#guard (extensionForks (.within 2 2) [1, 2] 3).length == 4
-- It forks only the chains whose last dice match.
#guard (explodeForks { results := [1, 2, 2], die := .standard 2 }
  (.within 2 2) (some 1)).length == 4
#guard (explodeForks { results := [1, 2, 2], die := .standard 2 }
  (.within 2 2) (some 0)).map (·.results) == [[1, 2, 2]]
-- An unlimited explode that matches every face forks once, saturating every
-- chain, and one that grows nothing forks once.
#guard (explodeForks { results := [1, 2], die := .standard 2 } (.within 1 2)
  none).map (·.results) == [[i32Max, i32Max]]
#guard (explodeForks { results := [1, 2], die := .standard 2 } (.within 3 3)
  none).map (·.results) == [[1, 2]]
-- An explode forks the chains that the drops kept, and forgets the drops.
#guard (explodeForks { results := [1, 2], highest := 1, die := .standard 2 }
  (.within 2 2) (some 1)).map (fun r => (r.lowest, r.highest, r.results))
  == [(0, 0, [1])]
-- A record never rolled forks once.
#guard (forks blankState (.explode 0 (.eq (.immediate 0)) .unlimited)).length
  == 1

/-! ## Paths -/

-- `3D6`: 6 × 6 × 6 paths.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 3) (.immediate 6),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 216

-- `(1D3)D3`: 3 + 9 + 27 paths, one for each count's dice.
#guard (paths blankState
  [.rollRange 0 (.immediate 1) (.immediate 3), .sumRollingRecord 0 0,
   .rollStandardDice 1 (.register 0) (.immediate 3),
   .sumRollingRecord 1 1, .return (.register 1)]).length == 39

-- `1D6 + 1D6`: two ranges, 6 × 6 paths.
#guard (paths blankState
  [.rollRange 0 (.immediate 1) (.immediate 6), .sumRollingRecord 0 0,
   .rollRange 1 (.immediate 1) (.immediate 6), .sumRollingRecord 1 1,
   .binary .add 2 (.register 0) (.register 1),
   .return (.register 2)]).length == 36

-- The paths that reach a program point: 3 after the range, 39 at the end.
#guard (paths blankState
  ([.rollRange 0 (.immediate 1) (.immediate 3), .sumRollingRecord 0 0,
   .rollStandardDice 1 (.register 0) (.immediate 3),
   .sumRollingRecord 1 1, .return (.register 1)].take 2)).length == 3

-- `1D6 reroll = 1`: 5 paths that keep their die and 6 that reroll it.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 6),
   .reroll 0 (.eq (.immediate 1)) (.times (.immediate 1)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 11
-- `1D6 reroll = 1 2 times`: 5 + 5 + 6 paths.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 6),
   .reroll 0 (.eq (.immediate 1)) (.times (.immediate 2)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 16
-- `1D6 reroll = 1 ∞ times`: 5 + 5 paths, by the conditional law.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 6),
   .reroll 0 (.eq (.immediate 1)) .unlimited,
   .sumRollingRecord 0 0, .return (.register 0)]).length == 10
-- `3D4 reroll = 1`: each die 3 ways not to match and 4 to reroll, so
-- `(3 + 4)³` paths.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 3) (.immediate 4),
   .reroll 0 (.eq (.immediate 1)) (.times (.immediate 1)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 343
-- `2D1 reroll = 1 ∞ times`: one path, which crosses out both dice.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 2) (.immediate 1),
   .reroll 0 (.eq (.immediate 1)) .unlimited,
   .sumRollingRecord 0 0, .return (.register 0)]).length == 1

-- `1D6 explode = 6`: 5 paths that end at once and 6 that explode.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 6),
   .explode 0 (.eq (.immediate 6)) (.times (.immediate 1)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 11
-- `1D6 explode = 6 2 times`: 5 + 5 + 6 paths.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 6),
   .explode 0 (.eq (.immediate 6)) (.times (.immediate 2)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 16
-- `2D6 explode ≥ 1 ∞ times`: 36 paths, each of which saturates.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 2) (.immediate 6),
   .explode 0 (.ge (.immediate 1)) .unlimited,
   .sumRollingRecord 0 0, .return (.register 0)]).length == 36
-- `1D2 explode = 2 reroll = 1`: a `1` rerolled 2 ways, a `2` extended by a
-- `1` rerolled 2 ways, and a `2` extended by a `2`.
#guard (paths blankState
  [.rollStandardDice 0 (.immediate 1) (.immediate 2),
   .explode 0 (.eq (.immediate 2)) (.times (.immediate 1)),
   .reroll 0 (.eq (.immediate 1)) (.times (.immediate 1)),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 5

/-! ## Docstring examples, verbatim -/

#guard (rangeForks 1 3).map (·.results) == [[1], [2], [3]]
#guard (rangeForks 3 1).map (·.results) == [[0]]
#guard standardFaces 3 == [1, 2, 3]
#guard standardFaces 0 == [0]
#guard customFaces [1, 1, 2] == [1, 1, 2]
#guard customFaces [] == [0]
#guard (diceForks 2 [1, 2]).map (·.results) == [[1, 1], [1, 2], [2, 1], [2, 2]]
#guard (diceForks 0 [1, 2]).map (·.results) == [[]]
#guard limitedForks (.within 1 1) [1, 2] 2 == [1, 2, 2]
#guard (keptForks .none [1, 2, 1] (· == 1) [3, 4]).map (·.results)
  == [[3, 2, 3], [3, 2, 4], [4, 2, 3], [4, 2, 4]]
#guard chainForks (· == 1) [3, 4] (.live [1, 2] 1)
  == [.live [2, 3] 3, .live [2, 3] 4, .live [2, 4] 3, .live [2, 4] 4]
#guard (chainsForks .none [.live [2] 1] (· == 1) [3, 4]).map (·.results)
  == [[5], [6]]
#guard (rerollForks { results := [1, 2], die := .standard 3 } (.within 1 1)
  none).map (·.results) == [[2, 2], [3, 2]]
#guard extensionForks (.within 2 2) [1, 2] 2
  == [([], 1), ([2], 1), ([2], 2)]
#guard (explodeChainsForks .none [.single 2, .single 1] (·.last == 2)
  [([], 1), ([], 2)]).map (·.results) == [[1, 3], [1, 4]]
#guard (explodeForks { results := [1, 2], die := .standard 2 } (.within 2 2)
  (some 1)).map (·.results) == [[1, 3], [1, 4]]
#guard (forks { registers := #[0], records := #[{}] }
  (.rollStandardDice 0 (.immediate 2) (.immediate 6))).length == 36
-- `3D6` forks 6 × 6 × 6 ways.
#guard (paths { registers := #[0], records := #[{}] }
  [.rollStandardDice 0 (.immediate 3) (.immediate 6),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 216

end Xdy.Spec.Test
