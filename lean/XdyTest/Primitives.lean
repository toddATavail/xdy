import Xdy.Primitives

/-!
# Primitive tests

Every assertion in the doctests of `xdy/src/primitives.rs`, transcribed, plus
the edge cases that distinguish these definitions from Lean's own operators.
-/

namespace Xdy.Test

open Xdy

/-! ## `clamp` -/

#guard clamp 7 == 7
#guard clamp i32Max == i32Max
#guard clamp i32Min == i32Min
#guard clamp (i32Max + 1) == i32Max
#guard clamp (i32Min - 1) == i32Min

/-! ## `add` -/

#guard add 1 2 == 3
#guard add i32Max 1 == i32Max
#guard add i32Min (-1) == i32Min

/-! ## `sub` -/

#guard sub 1 2 == -1
#guard sub i32Max (-1) == i32Max
#guard sub i32Min 1 == i32Min

/-! ## `mul` -/

#guard mul 2 3 == 6
#guard mul i32Max 2 == i32Max
#guard mul i32Min 2 == i32Min
#guard mul i32Max (-1) == i32Min + 1
#guard mul i32Min (-1) == i32Max

/-! ## `div` -/

#guard div 6 2 == 3
#guard div 6 0 == 0
#guard div i32Min (-1) == i32Max
-- Rounds toward zero, unlike Lean's `/` on `Int`.
#guard div (-7) 2 == -3
#guard div 7 (-2) == -3
#guard div (-7) (-2) == 3
#guard div 0 0 == 0

/-! ## `mod` -/

#guard «mod» 6 4 == 2
#guard «mod» 6 0 == 0
#guard «mod» i32Min (-1) == 0
-- Takes the sign of the dividend, unlike Lean's `%` on `Int`.
#guard «mod» (-7) 2 == -1
#guard «mod» 7 (-2) == 1
#guard «mod» (-7) (-2) == -1

/-! ## `exp` -/

#guard exp 2 3 == 8
#guard exp 2 0 == 1
#guard exp 0 0 == 1
#guard exp 0 1 == 0
#guard exp 0 (-1) == 0
#guard exp 1 0 == 1
#guard exp 1 1 == 1
#guard exp 1 (-1) == 1
#guard exp (-1) 0 == 1
#guard exp (-1) 1 == -1
#guard exp (-1) 2 == 1
#guard exp (-1) 3 == -1
#guard exp (-1) (-1) == -1
#guard exp (-1) (-2) == 1
#guard exp (-1) (-3) == -1
#guard exp 2 (-1) == 0
#guard exp 10 20 == i32Max
#guard exp (-10) 20 == i32Max
#guard exp (-10) 21 == i32Min
-- The boundary between computing exactly and saturating by sign.
#guard exp 2 30 == 1073741824
#guard exp 2 31 == i32Max
#guard exp (-2) 31 == i32Min
#guard exp 2 32 == i32Max
#guard exp (-2) 32 == i32Max
#guard exp (-2) 33 == i32Min
#guard exp 2 i32Max == i32Max
#guard exp i32Min i32Max == i32Min

/-! ## `neg` -/

#guard neg 1 == -1
#guard neg i32Max == i32Min + 1
#guard neg i32Min == i32Max

/-! ## `max` -/

#guard «max» 1 2 == 2
#guard «max» 0 (-3) == 0
#guard «max» i32Min i32Max == i32Max

end Xdy.Test
