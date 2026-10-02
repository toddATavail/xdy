import Xdy.Oracle

/-!
# Oracle protocol tests

Reading requests, writing responses in lowest terms with weights beyond `u64`,
and every way a request can fail, all through `respond`, as the `xdy-oracle`
executable drives it.
-/

namespace Xdy.Test

open Xdy Oracle

/-- A request for the function with the given JSON text, followed by the given
further fields, as JSON text. -/
private def req (function : String) (fields : String := "") : String :=
  "{\"function\": " ++ function ++ fields ++ "}"

/-- Answer whether the request succeeds with exactly the given response. -/
private def answers (request response : String) : Bool :=
  (respond request).toOption == some response

/-- Answer whether the request succeeds with a response containing the given
text. -/
private def answersWith (request text : String) : Bool :=
  match respond request with
  | .ok response => (response.splitOn text).length > 1
  | .error _ => false

/-- Answer whether the request fails with a message containing the given
text. -/
private def fails (request message : String) : Bool :=
  match respond request with
  | .ok _ => false
  | .error e => (e.splitOn message).length > 1

/-- `7` -/
private def constant : String := include_str "Fixtures/constant.json"

/-- `{x}: {x}D1`, which has one parameter. -/
private def maxFn : String := include_str "Fixtures/max.json"

/-- `1D6 + {y}`, which has one external variable. -/
private def external : String := include_str "Fixtures/external.json"

/-- `{n}, {m}: 5D6 drop lowest {n} drop highest {m}`. -/
private def drops : String := include_str "Fixtures/dynamic_drops.json"

/-- `65D2`, whose total, `2⁶⁵`, exceeds `u64`, though it has only 66
states. -/
private def sixtyFiveD2 : String :=
  "{\"parameters\": [], \"externals\": [], \"register_count\": 1, " ++
  "\"rolling_record_count\": 1, \"instructions\": [" ++
  "{\"RollStandardDice\": {\"dest\": 0, \"count\": {\"Immediate\": 65}, " ++
  "\"faces\": {\"Immediate\": 2}}}, " ++
  "{\"SumRollingRecord\": {\"dest\": 0, \"src\": 0}}, " ++
  "{\"Return\": {\"src\": {\"Register\": 0}}}]}"

/-! ## Responses -/

#guard answers (req constant) "{\"outcomes\":[[7,\"1\"]],\"total\":\"1\"}"
#guard answers (req (include_str "Fixtures/dynamic_count.json"))
  ("{\"outcomes\":[[1,\"9\"],[2,\"12\"],[3,\"16\"],[4,\"12\"],[5,\"12\"]," ++
   "[6,\"10\"],[7,\"6\"],[8,\"3\"],[9,\"1\"]],\"total\":\"81\"}")
-- Negative outcomes come first.
#guard answers (req (include_str "Fixtures/custom_dice.json"))
  ("{\"outcomes\":[[-2,\"1\"],[-1,\"2\"],[0,\"3\"],[1,\"2\"],[2,\"3\"]," ++
   "[3,\"2\"],[4,\"4\"],[5,\"2\"],[6,\"3\"],[8,\"2\"],[10,\"1\"]]," ++
   "\"total\":\"25\"}")

/-! ## Weights are in lowest terms and unbounded -/

-- Keeping none of `5D6` is certain, not 7776 ways to roll `0`.
#guard answers (req drops ", \"arguments\": [3, 3]")
  "{\"outcomes\":[[0,\"1\"]],\"total\":\"1\"}"
#guard answersWith (req sixtyFiveD2) "\"total\":\"36893488147419103232\""
#guard answersWith (req sixtyFiveD2) "[65,\"1\"],[66,\"65\"]"
-- The central weight, `C(65, 32)`.
#guard answersWith (req sixtyFiveD2) "[97,\"3609714217008132870\"]"

/-! ## Arguments and external variables -/

#guard answers (req drops ", \"arguments\": [4, 0]")
  ("{\"outcomes\":[[1,\"1\"],[2,\"31\"],[3,\"211\"],[4,\"781\"]," ++
   "[5,\"2101\"],[6,\"4651\"]],\"total\":\"7776\"}")
#guard answers (req external ", \"externals\": {\"y\": 10}")
  ("{\"outcomes\":[[11,\"1\"],[12,\"1\"],[13,\"1\"],[14,\"1\"],[15,\"1\"]," ++
   "[16,\"1\"]],\"total\":\"6\"}")
-- An unbound external variable is `0`.
#guard answersWith (req external) "[[1,\"1\"],"
-- `null` is the same as absent.
#guard answers (req constant ", \"arguments\": null, \"externals\": null")
  "{\"outcomes\":[[7,\"1\"]],\"total\":\"1\"}"
#guard answers (req constant ", \"arguments\": [], \"externals\": {}")
  "{\"outcomes\":[[7,\"1\"]],\"total\":\"1\"}"

/-! ## Malformed requests -/

#guard fails "not json" "the request is not JSON"
#guard fails "[1]" "the request is not an object"
#guard fails "{}" "the request has no function"
#guard fails (req constant ", \"args\": []") "unknown request field args"
#guard fails (req "{\"parameters\": []}") "function: "
#guard fails (req maxFn ", \"arguments\": [1.5]") "arguments: "
#guard fails (req maxFn ", \"arguments\": 1") "arguments: "
#guard fails (req external ", \"externals\": [10]") "externals: "
#guard fails (req external ", \"externals\": {\"y\": \"10\"}") "externals: "

/-! ## Requests that do not fit their functions -/

#guard fails (req maxFn) "expected 1 arguments, found 0"
#guard fails (req maxFn ", \"arguments\": [2147483648]")
  "2147483648 is outside the i32 range"
#guard fails (req constant ", \"externals\": {\"z\": 1}")
  "z is not an external variable"
#guard fails (req ("{\"parameters\": [], \"externals\": [], " ++
    "\"register_count\": 0, \"rolling_record_count\": 0, " ++
    "\"instructions\": []}"))
  "the function has no return"

/-! ## Docstring examples, verbatim -/

#guard render (Dist.uniform [2, 1, 2, 1]) ==
  "{\"outcomes\":[[1,\"1\"],[2,\"1\"]],\"total\":\"2\"}"
#guard respond "{\"function\": {\"parameters\": [], \"externals\": [],
  \"register_count\": 0, \"rolling_record_count\": 0,
  \"instructions\": [{\"Return\": {\"src\": {\"Immediate\": 7}}}]}}"
  matches .ok "{\"outcomes\":[[7,\"1\"]],\"total\":\"1\"}"

end Xdy.Test
