import XdyTest.Dist
import XdyTest.DistLaws
import XdyTest.Interpreter
import XdyTest.Ir
import XdyTest.Oracle
import XdyTest.Plan
import XdyTest.PlanLaws
import XdyTest.Primitives
import XdyTest.Record
import XdyTest.Saturation

/-!
# xDy Lean tests

The test driver for `lake test`. Every test is a `#guard` or `example`, checked
when its module builds, so a failing test fails the build.
-/
