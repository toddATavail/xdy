import Lean.Data.Json
import Xdy.Interpreter

/-!
# Oracle protocol

The protocol of the `xdy-oracle` executable, as pure functions from request
text to response text, so that tests can drive it without running a process.
`Main.lean` only moves text between them and the standard streams.

## The request

One JSON object on standard input, or, with `--lifeline`, on the first line of
standard input, which then stays open until the oracle should exit (see
`Main.lean`):

```json
{"function": {…}, "arguments": [3], "externals": {"y": 2}}
```

- `function`: a compiled xDy `Function`, as Rust's `serde` feature writes it;
  see `Xdy/Ir.lean`.
- `arguments`: the arguments, one per parameter, in order. Optional; absent
  or `null` means none.
- `externals`: values for external variables, by name. Optional; absent or
  `null` means none, and an external variable left unbound is `0`.

## The response

One line of JSON on standard output, giving the exact distribution of the
function's result in lowest terms:

```json
{"outcomes": [[-2, "1"], [-1, "2"], …], "total": "25"}
```

- `outcomes`: each possible result with its weight, least result first. The
  probability of a result is its weight over `total`.
- `total`: the sum of the weights.

Weights are decimal strings, since they can exceed every fixed-width integer:
`30D6` has a total of `6³⁰`, beyond `u64`. On failure, the executable writes a
message to standard error instead, and exits with status `1`.
-/

namespace Xdy

open Lean (Json FromJson fromJson? toJson)

namespace Oracle

/-- A request to the oracle: a function and what to bind before running it. -/
structure Request where
  /-- The function to run. -/
  function : Function
  /-- The arguments, one per parameter, in order. -/
  arguments : List Int := []
  /-- Values for external variables, by name. -/
  externals : List (String × Int) := []
  deriving Inhabited

/--
Read the arguments of a request.

# Parameters
- `j`: The value of the `arguments` field, or `null` if it is absent.

# Returns
The arguments, in order.

# Errors
If `j` is neither `null` nor an array of integers.
-/
private def readArguments : Json → Except String (List Int)
  | .null => pure []
  | j => fromJson? j

/--
Read the external variable bindings of a request.

# Parameters
- `j`: The value of the `externals` field, or `null` if it is absent.

# Returns
The bindings, as names and values.

# Errors
If `j` is neither `null` nor an object whose values are integers.
-/
private def readExternals : Json → Except String (List (String × Int))
  | .null => pure []
  | j => do
    let bindings ← j.getObj?
    bindings.toList.mapM fun (name, x) => return (name, ← x.getInt?)

/--
Read a request.

# Parameters
- `s`: The JSON text of a request.

# Returns
The request.

# Errors
If `s` is not JSON, is not an object, has a field other than `function`,
`arguments` and `externals`, lacks `function`, or has a field of the wrong
shape. The function is not validated here; `Function.run` validates it.
-/
def Request.ofJson (s : String) : Except String Request := do
  let j ← Json.parse s |>.mapError (s!"the request is not JSON: {·}")
  let fields ← j.getObj? |>.mapError fun _ => "the request is not an object"
  fields.toList.forM fun (key, _) =>
    unless ["function", "arguments", "externals"].contains key do
      throw s!"unknown request field {key}"
  let function ← match j.getObjVal? "function" with
    | .ok f => fromJson? f |>.mapError (s!"function: {·}")
    | .error _ => throw "the request has no function"
  let arguments ← readArguments (j.getObjValD "arguments")
    |>.mapError (s!"arguments: {·}")
  let externals ← readExternals (j.getObjValD "externals")
    |>.mapError (s!"externals: {·}")
  pure { function, arguments, externals }

/--
Write a distribution as a response.

# Parameters
- `d`: The distribution of a function's result.

# Returns
The JSON text of the response, on one line, with the weights reduced to
lowest terms.

# Examples
```lean
#guard render (Dist.uniform [2, 1, 2, 1]) ==
  "{\"outcomes\":[[1,\"1\"],[2,\"1\"]],\"total\":\"2\"}"
```
-/
def render (d : Dist Int) : String :=
  let d := d.lowestTerms
  let outcomes := d.toList.map fun (x, w) => Json.arr #[toJson x, toString w]
  (Json.mkObj [
    ("outcomes", .arr outcomes.toArray),
    ("total", toString d.total)]).compress

/--
Answer a request: read it, run its function, and write the distribution of
the result.

# Parameters
- `s`: The JSON text of a request.

# Returns
The JSON text of the response.

# Errors
If the request cannot be read (see `Request.ofJson`), or its function cannot
be run with its arguments and external variables (see `Function.run`).

# Examples
```lean
#guard respond "{\"function\": {\"parameters\": [], \"externals\": [],
  \"register_count\": 0, \"rolling_record_count\": 0,
  \"instructions\": [{\"Return\": {\"src\": {\"Immediate\": 7}}}]}}"
  matches .ok "{\"outcomes\":[[7,\"1\"]],\"total\":\"1\"}"
```
-/
def respond (s : String) : Except String String := do
  let request ← Request.ofJson s
  let d ← request.function.run request.arguments request.externals
  pure (render d)

end Oracle

end Xdy
