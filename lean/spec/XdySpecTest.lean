import XdySpecTest.Conditioning
import XdySpecTest.Convolution
import XdySpecTest.Cost
import XdySpecTest.Forward
import XdySpecTest.Ghost
import XdySpecTest.Invariant
import XdySpecTest.Law
import XdySpecTest.Mixture
import XdySpecTest.Oracle
import XdySpecTest.OrderStatistics
import XdySpecTest.Outcomes
import XdySpecTest.Paths
import XdySpecTest.Record
import XdySpecTest.Semantics
import XdySpecTest.World

/-!
# xDy specification tests

The test driver for `lake test`. Every test is a `#guard` or `example`, checked
when its module builds, so a failing test fails the build.
-/
