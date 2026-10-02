# The xDy specification in Lean

The meaning of the xDy intermediate representation (IR) as a Mathlib `PMF ℤ`:
for a compiled xDy function and its arguments, the probability of each result
that it can answer. It is the specification that the executable oracle in the
parent package must meet, and the statement and proof of the lemmas that justify
the Rust forward pass.

## Why a separate package

The specification needs Mathlib, for `PMF` and the measure theory beneath it;
the oracle needs only core Lean. Keeping them apart means that building and
running the oracle, which `just oracle` does often, never waits on Mathlib. The
specification requires the oracle by path, and reuses its IR, its validator and
its initial state, which are deterministic.

## How it differs from the oracle

The specification mirrors the oracle instruction by instruction, but it
describes rather than computes:

| | Oracle ([`../Xdy/`](../Xdy/)) | Specification ([`XdySpec/`](XdySpec/)) |
|---|---|---|
| Distribution | `Dist`: natural weights over one total | `PMF`: probabilities in `ℝ≥0∞` |
| Equal outcomes | Merged as they arise | Equal as points of the support |
| Rolling records | Results sorted as they arrive | Results in roll order, sorted only to drop and sum, as in Rust |
| A die | `Dist.uniform` of its faces | `PMF.ofMultiset` of its faces |
| Computable | Yes | No |

The oracle's normalized weights equal the specification: `run_eq` in
[`XdySpec/Oracle.lean`](XdySpec/Oracle.lean) proves that, for every function,
arguments and external variables, the oracle's weights over its total are the
specification's probabilities, and that the two fail, with the same message,
alike. So sorting early and merging change nothing. The proof relates each
distribution of the oracle's states to the specification's law of states read
with their records sorted, and shows that every instruction keeps that relation.

## Layout

| Module | Contents |
|--------|----------|
| [`XdySpec/Record.lean`](XdySpec/Record.lean) | Rolling records in roll order: results, drops, the kept window of the sorted results and the saturating sum; and reading them as the oracle's, sorted. |
| [`XdySpec/Semantics.lean`](XdySpec/Semantics.lean) | Machine states, the laws of ranges and dice, one step per instruction, and the law of a function's result. |
| [`XdySpec/Law.lean`](XdySpec/Law.lean) | The law of an oracle `Dist`, its weights over its total, and the proofs that `pure`, `uniform`, `map` and `bind` have the laws of their `PMF` counterparts. A mixture over any common multiple of its branches' totals has the law of `bind`, as the forward pass's merges and mixtures of setups mix, and scaling every weight by one positive factor keeps the law, as a world's common factor scales its answer. |
| [`XdySpec/Oracle.lean`](XdySpec/Oracle.lean) | The proof that the oracle meets the specification: sorting commutes with every instruction, so each oracle distribution of states has the specification's law, sorted. |
| [`XdySpec/Conditioning.lean`](XdySpec/Conditioning.lean) | Lemma 6 of the design: conditioning on a value and merging its worlds at its elimination point preserves the law. Splitting is the law of total probability, burying a location that the rest never reads leaves the result's law alone, and worlds that each combine the survivor independently with one common law merge into the mixture of the survivor's laws, combined with it; `exec_merge` and `run_merge` put these together, within the worlds of the other open splits, as the pass merges the worlds that agree on every other split's outcome. The plan's merges preserve the law without these hypotheses, as `XdySpec/Invariant.lean` proves. |
| [`XdySpec/World.lean`](XdySpec/World.lean) | Product laws of worlds, the vocabulary of the plan's invariant: a world of the open splits as independent laws of its live locations over a base state. Any live location splits off in the form that `exec_merge` takes; conditioning on one makes it a point mass, burying one marginalizes it out, mixing worlds that differ in one location mixes its law, and an instruction on independent operands leaves a world a world, its destination at the law of the instruction on its operands: lemma 1's lift for arithmetic on two registers, a mapped law for arithmetic with an immediate or on the same register twice, the negated law for a negation, the roll's law mixed over its operands' joint law for a range or dice, the lift of the drop over the record and the count for a drop, and the law of the sum for a sum, each the law that `propagation.rs` computes for its instruction. So it does beside live point masses that it reads, which stay live. |
| [`XdySpec/Mixture.lean`](XdySpec/Mixture.lean) | The rolling records of `propagation/mixture.rs`, as mixtures of setups: a roll with its operands resolved to values, and its drops. A setup has the law of the record that it fills, and a mixture the mixture of its setups' laws. Rolling mixes the roll's law over its operands', dropping drops every setup by every count, which is the lift of the drop over the record and the count, and the sum of a mixture is the mixture of its setups' sums. A custom die's distinct faces, weighed by multiplicity, have the law of the die. The difference array in which the sums of ranges accumulate weighs every outcome as adding each range's outcomes one by one would. A setup's sum is what `Roll::sum` builds for each of its branches but the order statistics: fixed results add each distinct result's kept copies with one clamp, a range that is empty or drops anything and dice that are none, have no faces, or are all dropped sum to `0`, and a range without drops sums uniformly. |
| [`XdySpec/Forward.lean`](XdySpec/Forward.lean) | The forward pass of `propagation.rs`, as the specification sees it: in each world of the open splits, a law of each live location over a base state, and the law of each split's outcome. An instruction gives its destination the law of the instruction run on the world of the values that it reads, a split fixes its value in each world at its outcome, drawn from its law read sorted, so that a rolling record's outcomes are the multisets of its results, as the Rust's are, and a merge mixes the survivor's law over the split's outcome, following the plan's steps as the Rust does. Not computable, since its laws are `PMF`s, but explicit, so its laws on a concrete function unfold step by step. |
| [`XdySpec/Invariant.lean`](XdySpec/Invariant.lean) | The plan's invariant, proved to hold of the forward pass before every instruction of a well-formed function: once every dead location is buried, the law of states, read sorted, is the mixture, over the outcomes of the open splits, drawn from the bottom of the stack, of worlds of independent laws of the live locations, read sorted, in which each live value's law, and each split's, depends only on the open splits on which it depends, each split's outcomes are sorted, and each live open split is a point mass at its outcome. No instruction can tell the order of a record's results, so fixing a split record at its results sorted keeps it, as `exec_split_sorted` does the law of the program. Each instruction, split and merge keeps it, so a function answers the mixture, over the worlds of the splits open before its return, of the forward pass's law of the returned register in each. The forward pass tests (`XdySpecTest/Forward.lean`) replay the pass on two functions to the laws that they answer, and on a third, which splits a rolling record, to worlds that are the multisets of its results. |
| [`XdySpec/Ghost.lean`](XdySpec/Ghost.lean) | The forward pass draws its worlds with their true joint law. A ghost beside the state holds the outcomes of the open splits, which a split copies from the state, sorted, and a merge forgets, since a split's value may be overwritten while it stays open. Once the dead locations are buried, the law of states and outcomes, with the states read sorted, is the mixture of the pass's worlds, each tagged with its outcomes; each instruction, split and merge keeps this, so the outcomes have the law of the pass's worlds (`ghost_snd`), and the states the function's (`ghost_fst`). |
| [`XdySpec/Paths.lean`](XdySpec/Paths.lean) | The paths through a function, as the enumerating builder that the forward pass replaced walked them: each instruction's forks, one for each value of a range and each face of each die, by position, or one if there are none, and one for every other instruction. Every state that an instruction can reach is among its forks, and every instruction forks at least once, so the paths that reach a program point never outnumber those that reach the end. |
| [`XdySpec/Cost.lean`](XdySpec/Cost.lean) | Lemma 7 of the design: the worlds of the forward pass never outnumber the paths through the function, before each instruction (`worlds_le_paths`) and after each step of the plan (`worlds_le_paths_of_step`). A path fixes the ghost, so following it along each path gives one entry per path, among which are all the outcomes that the ghost can reach, and so every world. The worlds are counted as the support of their law, which the Rust's count of worlds is argued, not proved, to be. |
| [`XdySpec/Convolution.lean`](XdySpec/Convolution.lean) | Lemmas 1 and 2 of the design: the law of a binary operation over independent values, convolution, and convolution powers as the law of the sum of dice; and Case 1, that the saturating sum of a roll without drops that passes `folds_to_clamp`'s test has the law of the convolution power, clamped once. |
| [`XdySpec/OrderStatistics.lean`](XdySpec/OrderStatistics.lean) | The sum of dice with drops, as `order_statistics` in `propagation/record.rs` computes it: a dynamic program over the distinct faces in ascending order, which places every number of copies of each face after the positions filled so far, weighed by a binomial and the face's weight raised to the copies. Weighing every roll by its faces' weights, sorted die by die, peeling the greatest face is Pascal's rule, so the program's states are the weighed rolls read through the kept window, and its weights, over the total weight of the die raised to the number of dice, are the law of the sum of the kept dice, for custom and standard dice alike. |
| [`XdySpec/Outcomes.lean`](XdySpec/Outcomes.lean) | The outcomes on which `split` in `propagation.rs` fixes a rolling record: the multisets of its results, as `Roll::outcomes` in `propagation/record.rs` grows them one distinct face at a time, each copy count weighed by a binomial and the face's weight raised to the copies. Peeling the greatest face, as for the order statistics, the program's states are the weighed rolls read whole, so every roll's outcomes, over its total, are the law of its record sorted, and every setup's, which keep its drops, the law of its record as the oracle reads it. Sorting commutes with every instruction, so splitting a record on its sorted reading and fixing each world's record at its outcome answers the law of the program read sorted, and of every register. |
| [`XdySpecTest/`](XdySpecTest/) | Tests, as `#guard` checks of the records and `example` proofs of the laws, since a `PMF` cannot be computed. |

## Building

From the repository root:

```sh
just spec
```

This fetches Mathlib's prebuilt files, several gigabytes the first time, then
builds the specification and runs its tests (`lake exe cache get`, then `lake
build`, then `lake test`). The toolchain in [`lean-toolchain`](lean-toolchain)
must match both the oracle's and the one that the pinned Mathlib release
requires.

## Conventions

As for the oracle; see its [README](../README.md#conventions). Examples in
docstrings that concern a `PMF` are `example` proofs rather than `#guard`
checks, and are repeated in the tests. The specification's names match the
oracle's, in the namespace `Xdy.Spec`, so opening both `Xdy` and `Xdy.Spec`
makes names such as `rollRange` ambiguous.
