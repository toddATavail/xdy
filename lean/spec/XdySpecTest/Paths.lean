import XdySpec.Paths

/-!
# Path enumeration tests

The forks of ranges and dice, including their degenerate cases, and the number
of paths of whole functions.
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

/-! ## Docstring examples, verbatim -/

#guard (rangeForks 1 3).map (·.results) == [[1], [2], [3]]
#guard (rangeForks 3 1).map (·.results) == [[0]]
#guard standardFaces 3 == [1, 2, 3]
#guard standardFaces 0 == [0]
#guard customFaces [1, 1, 2] == [1, 1, 2]
#guard customFaces [] == [0]
#guard (diceForks 2 [1, 2]).map (·.results) == [[1, 1], [1, 2], [2, 1], [2, 2]]
#guard (diceForks 0 [1, 2]).map (·.results) == [[]]
#guard (forks { registers := #[0], records := #[{}] }
  (.rollStandardDice 0 (.immediate 2) (.immediate 6))).length == 36
-- `3D6` forks 6 × 6 × 6 ways.
#guard (paths { registers := #[0], records := #[{}] }
  [.rollStandardDice 0 (.immediate 3) (.immediate 6),
   .sumRollingRecord 0 0, .return (.register 0)]).length == 216

end Xdy.Spec.Test
