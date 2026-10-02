import Xdy.Dist
import Xdy.DistLaws
import Xdy.Interpreter
import Xdy.Ir
import Xdy.Oracle
import Xdy.Plan
import Xdy.PlanLaws
import Xdy.Primitives
import Xdy.Record
import Xdy.Saturation

/-!
# xDy in Lean

The executable oracle for xDy's exact distributions: a deliberately naive
interpreter of the xDy intermediate representation that computes the exact
distribution of a dice expression's outcomes. See `README.md`.
-/
