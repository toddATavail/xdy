import Xdy

/-!
# `xdy-oracle`

The executable oracle: reads one request from standard input, and writes the
exact distribution of the requested function's result to standard output. See
`Xdy/Oracle.lean` for the protocol.

With `--lifeline`, the oracle reads the request from the first line of
standard input rather than all of it, then holds standard input as a lifeline:
as soon as it closes, the oracle exits with status `2`, answered or not. The
kernel closes the oracle's standard input when whoever writes to it dies, even
by `SIGKILL`, so an oracle on a lifeline never outlives its parent.
-/

open Xdy.Oracle in
/--
Answer one request from standard input.

# Parameters
- `args`: The command line, which is empty or `--lifeline`.

# Returns
`0` after writing the response to standard output, or `1` after writing an
error message to standard error. With `--lifeline`, exits with status `2` as
soon as standard input closes, if it has not answered by then.

# Notes
With `--lifeline`, the oracle exits explicitly once it answers, rather than
returning, lest the runtime wait for the task that watches standard input. It
exits with `IO.Process.forceExit`, after flushing standard output and standard
error itself, rather than with `IO.Process.exit`: the task that watches
standard input blocks in a read that holds the lock of standard input, and
glibc's `exit` locks every stream to flush it, so it would wait for the
lifeline to close, and the parent, waiting for the oracle to exit, would never
close it. The task exits likewise, since the main thread may still be running.
-/
def main (args : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  match args with
  | [] =>
    let request ← stdin.readToEnd
    answer request
  | ["--lifeline"] =>
    let request ← stdin.getLine
    let _ ← IO.asTask (prio := .dedicated) do
      let _ ← stdin.readToEnd
      (IO.Process.forceExit 2 : IO Unit)
    let code ← answer request
    (← IO.getStdout).flush
    (← IO.getStderr).flush
    IO.Process.forceExit code.toUInt8
  | _ =>
    IO.eprintln "usage: xdy-oracle [--lifeline]"
    pure 1
where
  /--
  Answer a request.

  # Parameters
  - `request`: The text of the request.

  # Returns
  `0` after writing the response to standard output, or `1` after writing an
  error message to standard error.
  -/
  answer (request : String) : IO UInt32 := do
    match respond request with
    | .ok response =>
      IO.println response
      pure 0
    | .error message =>
      IO.eprintln message
      pure 1
