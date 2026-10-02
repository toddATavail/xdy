# xDy in Lean

A Lean 4 model of the semantics of the xDy intermediate representation (IR). It
serves as the executable oracle for exact distributions: given a compiled xDy
function and its arguments, it computes the exact distribution of the function's
outcomes, which the Rust forward pass is checked against.

## Why a naive oracle

The oracle is deliberately naive. It runs the program once over a distribution
of whole machine states (every register and rolling record): each range or die
draw branches into one state per outcome, and states that become equal merge,
adding their weights. It shares no algorithm with the Rust forward pass, which
relies on convolution, order statistics, mixtures and conditioning, so it cannot
share its bugs. It is exponential in the worst case and intended only for small
programs.

The oracle mirrors the roll-time evaluator, so the primitives in
[`Xdy/Primitives.lean`](Xdy/Primitives.lean) follow
[`xdy/src/primitives.rs`](../xdy/src/primitives.rs) case by case, including
saturation, division and remainder by zero, and the special cases of
exponentiation.

## Layout

| Module | Contents |
|--------|----------|
| [`Xdy/Primitives.lean`](Xdy/Primitives.lean) | The `i32` arithmetic primitives, as saturating operations on `Int`. |
| [`Xdy/Record.lean`](Xdy/Record.lean) | Rolling records: results, drops, the kept window and the saturating sum. |
| [`Xdy/Saturation.lean`](Xdy/Saturation.lean) | Proofs of closed forms for the saturating fold that sums a record, and of the soundness of the test by which the Rust forward pass replaces it with one clamp, `folds_to_clamp`. |
| [`Xdy/Dist.lean`](Xdy/Dist.lean) | Finite distributions: exact natural weights over one total, merging equal outcomes. |
| [`Xdy/DistLaws.lean`](Xdy/DistLaws.lean) | Proofs of the weight and total of every distribution that `Dist` builds, as sums over its entries, for the proof that the oracle meets the specification. |
| [`Xdy/Ir.lean`](Xdy/Ir.lean) | The IR, and a reader and validator for the JSON that Rust's `serde` feature writes. |
| [`Xdy/Interpreter.lean`](Xdy/Interpreter.lean) | The interpreter: machine states, one step per instruction, and the exact distribution of a function's result. |
| [`Xdy/Oracle.lean`](Xdy/Oracle.lean) | The oracle's protocol: reading a request and writing the distribution as a response. |
| [`Xdy/Plan.lean`](Xdy/Plan.lean) | The Rust forward pass's fan-out plans: locations, which instructions read and write them, and when each is dead; the values of a function, with their readers, whether each may be random, and the values that fan out on which each depends; and the splits and merges scheduled after each instruction. It mirrors `propagation/plan.rs`, so that the specification can prove that the plan preserves the law. |
| [`Xdy/PlanLaws.lean`](Xdy/PlanLaws.lean) | Proofs of the structural facts about plans on which the proof that they preserve the law rests: a live value occupies its location, and an instruction that overwrites a live value reads it; a live value that fans out is an open split, so every random value that an instruction reads is an open split or is read for the last time; each merge meets the survivor's conditions; a register or record is live exactly when its location is not dead; and what is live before an instruction, and that the open splits ascend, so that none depends on a split that may merge but itself. |
| [`Main.lean`](Main.lean) | The `xdy-oracle` executable, which answers one request from standard input. |
| [`XdyTest/`](XdyTest/) | Tests, as `#guard` checks that fail the build when false, and [IR fixtures](XdyTest/Fixtures/README.md) compiled by Rust. |

## Building

Install [elan](https://github.com/leanprover/elan), which provisions the
toolchain pinned in [`lean-toolchain`](lean-toolchain), then from the
repository root:

```sh
just lean
```

This builds the library and the `xdy-oracle` executable, then runs the tests
(`lake build`, then `lake test`). Every warning is an error, and every public
declaration must be documented.

## Running the oracle

The oracle reads one JSON request from standard input and writes the exact
distribution of the function's result to standard output, in lowest terms:

```sh
$ echo '{"function": '"$(cat XdyTest/Fixtures/dynamic_count.json)"'}' \
    | .lake/build/bin/xdy-oracle
{"outcomes":[[1,"9"],[2,"12"],[3,"16"],[4,"12"],[5,"12"],[6,"10"],[7,"6"],[8,"3"],[9,"1"]],"total":"81"}
```

A request may also bind `"arguments"`, as an array of integers, and
`"externals"`, as an object from names to integers. Weights are decimal strings,
since they can exceed every fixed-width integer. On failure, the oracle writes a
message to standard error and exits with status `1`. See
[`Xdy/Oracle.lean`](Xdy/Oracle.lean) for the full protocol.

With `--lifeline`, the oracle reads the request from the first line of standard
input rather than all of it, then holds standard input as a lifeline: as soon as
it closes, the oracle exits with status `2`, answered or not. The kernel closes
it when whoever writes to it dies, even by `SIGKILL`, so an oracle on a lifeline
never outlives its parent. The Rust oracle tests run it so, lest an interrupted
test leave it computing for hours.

The oracle is exponential in the worst case, and intended only for small
programs: `30D6` takes tens of seconds.

## Checking the Rust forward pass

From the repository root:

```sh
just oracle
```

This builds and tests the Lean, then runs four ignored Rust tests against the
oracle, so neither `just verify` nor `just stress` needs Lean. CI does the same
on every push to `main` and every pull request against it. Set `XDY_ORACLE` to
run an oracle built elsewhere.

The first runs every case of the Rust distribution corpus,
[`xdy/tests/test_distributions.txt`](../xdy/tests/test_distributions.txt),
through the oracle. Every case must agree with the oracle exactly, as
probabilities, including every case with a dynamic roll, i.e., a roll whose
count, faces or range depends on an earlier roll, whose paths are not equally
likely.

The test writes the corpus, with the oracle's distribution in place of each
case that disagrees, to `target/oracle/test_distributions.txt`, so a diff
against the corpus shows exactly the cases to correct.

The second is a property test: it generates random small programs that favor
dynamic rolls, builds each distribution, optimized and not, with the forward
pass, and holds it to the oracle exactly, as probabilities, as well as to the
bounds that the evaluator computes. A program abstains if the forward pass
exceeds a budget of steps and cells, or if the oracle, which has no budget of
its own, has not answered within two seconds. The third does the same after
resumming each program, so that it reads its rolling records more than once,
as the compiler never does. `proptest` records each failure, shrunk, in
[`xdy/proptest-regressions/tests/oracle.txt`](../xdy/proptest-regressions/tests/oracle.txt),
so later runs try it first.

The fourth kills a test process that has asked the oracle a question that would
take it ages, and expects the oracle to exit at once, rather than outlive it.

The IR fixtures that the Lean tests read are held to the current compiler by an
ordinary Rust test; see the [fixtures README](XdyTest/Fixtures/README.md).

## Conventions

The Lean follows the Rust's documentation and quality rubric, adapted where the
languages differ:

- Docstrings use the same sections as the Rust: `# Parameters`, `# Returns`, `#
  Errors`, `# Notes` and `# Examples`. Theorems, which the Rust lacks, list
  their assumptions under `# Hypotheses`, apart from their `# Parameters`.
- Lines stay within 80 columns. Indentation is 2 spaces, the Lean convention,
  since Lean forbids tabs.
- Sections are marked with `/-! ## Heading -/` module docs rather than banners.
- Examples in docstrings are `#guard` checks, repeated in the tests.

## Status

The executable oracle is complete. The specification of the IR as a Mathlib `PMF
ℤ` lives in a separate package, [`spec/`](spec/README.md), so that the oracle
never needs Mathlib. The oracle is proved to meet the specification (`run_eq`),
and lemmas 1 to 3 of the design are proved, with Case 1, which puts lemmas 2 and
3 together for rolls without drops, and lemma 6, that conditioning and merging
at the elimination point preserve the law, given that each world factors as the
plan assumes, and lemma 7, that the forward pass's worlds never outnumber the
paths through the function. Beyond those, the forward pass itself is proved to
answer the specification's law (`run_eq_forward`): the plan's invariant holds
before every instruction, so the merges preserve the law with no assumption
about how worlds factor, and splitting a rolling record on the multisets of its
results, as the Rust does, keeps the law (`exec_split_sorted`). Proofs that need
no Mathlib live beside the oracle: lemma 3, on saturating folds, and the laws of
`Dist`'s weights and totals.
