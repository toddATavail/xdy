# xDy: Once-and-for-all dice expression compiler

[<img alt="github" src="https://img.shields.io/badge/github-toddATavail/xdy-abda724?logo=github" height="20">](https://github.com/toddATavail/xdy)
[![Latest version](https://img.shields.io/crates/v/xdy.svg)](https://crates.io/crates/xdy)
[![Documentation](https://docs.rs/xdy/badge.svg)](https://docs.rs/xdy)
[![Build Status](https://github.com/toddATavail/xdy/workflows/Rust/badge.svg)](https://github.com/toddATavail/xdy/actions/workflows/rust.yml)
[![BSD](https://img.shields.io/badge/license-BSD3-blue.svg)](https://github.com/toddATavail/xdy/blob/main/LICENSE)

`xDy` is an extremely fast dice expression compiler that can be used to generate pseudorandom numbers. It is designed for use within a variety of applications, such as role-playing games (RPGs) and simulations, but also suits other applications that must introduce controlled randomness or generate specific probability distributions. It is written in Rust for maximum performance, safety, and portability.

* [Overview](#overview)
* [Installation](#installation)
* [Upgrading from 0.13](#upgrading-from-013)
* [Language features](#language-features)
* [Examples](#examples)
	* [One-shot evaluation](#one-shot-evaluation)
	* [Repeated evaluation](#repeated-evaluation)
	* [Bounds analysis](#bounds-analysis)
	* [Dice budgets](#dice-budgets)
	* [Probability distributions](#probability-distributions)
	* [Diagnostics](#diagnostics)
* [Performance](#performance)
* [Safety](#safety)
* [Cargo features](#cargo-features)
* [Project structure](#project-structure)
* [Planned work](#planned-work)

## Overview

`xDy` compiles dice expressions into reusable functions that can be applied to user-supplied pseudorandom number generators (pRNGs) to simulate dice rolls. Functions may define formal parameters or bind values supplied through an environment. A six-pass optimizing compiler produces efficient code for each dice expression, so the resulting functions can be used to generate dice rolls with minimal overhead. The generated code targets a dedicated intermediate representation (IR) that is designed specifically for dice expressions. An efficient evaluator interprets the IR to produce the final result.

Beyond generating just a final tally, `xDy` also provides detailed information about the individual dice rolls that contributed to the total. So `3D6 + 1D8` might produce a total of `18`, but it also tells you that the six-sided dice produced [`3`, `5`, `4`] and the eight-sided die produced `6`.

`xDy` also provides analytical capabilities. `xDy` can compute the bounds and probability distribution of a dice expression. For example, `xDy` can determine that `3D6 + 1D8` has a minimum value of `4`, has a maximum value of `26`, and puts the probability of rolling a `10` at `0.046875` (`81/1728`). Bounds are computed by interval arithmetic over a static analysis, so a parameter or environmental variable need not be pinned to a single value: it may be given an interval, or left unsupplied entirely, in which case it is bounded by the whole of `i32` and the answer remains sound. Probability distributions are exact, and computed in one forward pass over the expression, without enumerating its rolls, within a budget that the caller can weigh against an estimate before the pass begins.

## Installation

Add `xDy` to your `Cargo.toml`, together with version `0.10` of `rand`, whose `Rng` trait supplies every roll:

```toml
[dependencies]
xdy = "0.14"
rand = "0.10"
```

Or run `cargo add xdy rand@0.10`. See [Cargo features](#cargo-features) for the optional features.

## Upgrading from 0.13

`0.14.0` replaces histograms with exact probability distributions. Through `0.13.0`, a histogram counted the paths through an expression's rolls, as though every path were equally likely. That holds when the dice are fixed, but not when a roll decides how many dice to roll, how many faces they have, or the bounds of a range: in `(1D3)D3`, the count, `1D3`, decides whether one, two, or three `D3`s follow. Each count is equally likely, but leads to `3`, `9`, or `27` paths, so each path after a count of `1` is three times as likely as each path after a count of `2`, and nine times as likely as each path after a count of `3`. `0.13.0` weighed all `39` paths alike, and so put the probability of a total of `1` at `1/39`, where it is `1/9`. Distributions now weigh every outcome exactly, and a Lean specification, with proofs, an executable Lean oracle, and property tests hold them to the semantics of the language. Beyond that fix:

* `Distribution` replaces `Histogram`. Its weights are unbounded integers (`Weight`), and it answers exact `Probability`s, its cumulative distribution and quantiles, and its `mean` as a signed `Rational`.
* `Evaluator::plan_distribution` replaces the serial and parallel `HistogramBuilder`s and their `EvaluationStateIterator`s. It answers a `DistributionPlan`, whose `estimate` prices the build before it runs, and whose `build` computes the distribution within a `Budget` of steps and cells, reports its progress, and may be cancelled. `BuildError::BudgetExhausted` replaces `EvaluationError::HistogramBudgetExhausted`.
* `Evaluator::sample` estimates a distribution by sampling, for when the exact one costs too much.
* The `parallel-histogram` feature, and with it the dependency on `rayon`, is gone: the forward pass that computes distributions is serial, and far faster than the parallel builder was (see [Performance](#performance)).
* `Validator` takes two lifetimes, as `Validator<'a, 'src>`, and is no longer `Copy`, since driving it as an `ASTVisitor` now performs every check.
* The parser combinator `constant` reads only an unsigned constant, as the grammar's `CONSTANT` does, and the new `integer` reads a signed one, as the grammar's `INTEGER` does.
* `tree-sitter-xdy` names every face of a die `integer`, and every other literal `constant`; `negative_constant` is gone.

## Language features

`xDy` aims to achieve sufficient expressiveness to cover the gamut of use cases established by popular RPG systems. The following features are supported:

* Constants: `-1`, `0`, `1`, `2`, `3`, …
* Standard dice, `xDy`, where the count `x` and the faces `y` are each a constant, a name, a binding, or a group:
  `1D6`, `2D8`, `3D10`, `4D17`, `{x}D{y}`, `(1D4)D6`, …
* Custom dice, e.g., Fudge dice:
  `1D[-1, 0, 1]`, `2D[-1, 0, 1, 3, 5]`, `3D[1, 1, 2, 3, 5, 8]`, …
* Arithmetic
  * Negation: `-1`, `-1D4`, …
  * Addition: `1 + 1`, `1 + 1D4`, `1D6 + 1D8`, …
  * Subtraction: `3 - 2`, `1 - 1D4`, `1D6 - 1D8`, …
  * Multiplication: `2 * 3`, `2 × 3`, `2 * 1D4`, `2D6 * 1D8`, …
  * Division: `6 / 2`, `6 ÷ 2`, `6 / 1D4`, `1D6 / 1D8`, …
  * Modulus: `6 % 2`, `6 % 1D4`, `1D6 % 1D8`, …
  * Exponentiation: `2 ^ 3`, `2 ^ 1D4`, `2D6 ^ 1D8`, …
* Grouping: `(1 + 2) * 3`, `1 + (2 * 3D4)`, `(3 * 2)D5`, `4D(5 - 2)`, …
* Ranges, i.e., a single value chosen uniformly between two bounds, inclusive: `[1:6]`, `[3:18]`, `[{x}:2 * {x}]`, …
* Drop lowest: `4D6 drop lowest`, `4D6 drop lowest 1`, `2D8 drop lowest 2`, …
* Drop highest: `4D6 drop highest`, `4D6 drop highest 1`, `2D8 drop highest 2`, …
* Chained drops, applied in the order written: `4D6 drop lowest drop highest`, `5D4 drop lowest 2 drop highest 2`, …
* Formal parameters: `{x}: 1D6 + {x}`, `{y}: 1D6 + {y}`, `{x}, {y}: {x}D{y}`, …
* Environmental variables: `1D6 + {x}`, `1D6 + {y}`, `{x}D{y}`, …
* Subexpression naming: `{x}@(3D6) + {x}`, `{y}: {x}@(3D6) + {x} + {y}`, `{x}@(2 + 3)D6`, …
* Dynamic expressions: `3D(2D6)`, `{x}: ({x}D3)D8`, …

A standard die with zero or fewer faces always rolls `0`, so `3D-6` is `0`. A drop clause without a count drops a single die, and a count of zero or less drops nothing. A drop count never begins with a minus, so `4D6 drop lowest -1` is `(4D6 drop lowest) - 1`, just as it reads. Integer division truncates toward zero, and a division or remainder by zero answers zero.

Environmental variables permit background state to be associated with an evaluator, while formal parameters permit dynamic state to be passed into each evaluation. Both features can be used to parameterize dice expressions and make them more flexible. Environmental variables could easily cover, for example, the attributes and skills of a character in an RPG, while formal parameters could cover the situational modifiers applied to a particular roll.

Subexpression naming binds the integer result of a parenthesized subexpression to a name with `{name}@(expr)`, making it available as `{name}` at any later position. Every name is written in braces, whether it names a formal parameter, an environmental variable, or a binding. The braces free a name to contain any visible character but a brace, so `{weapon: 2/3}`, `{$env|weapon}`, and `{🎲}` are all names. A name may contain whitespace of any kind, so a long name may be broken over lines, but a name is never distinguished by its whitespace: whitespace just inside the braces is not part of it, so `{ x }` is `{x}`, and every run of whitespace within it collapses to a single space, so `{a  b}`, and `{a` and `b}` on separate lines, are both `{a b}`. Supply an environmental variable by this canonical name, e.g., `a b`. The bound expression is evaluated exactly once — even when it contains dice — so `{x}@(3D6)` rolls its dice a single time and reuses the total wherever `{x}` appears. A name must be referenced only after it is bound, and binding names share a single namespace with formal parameters and environmental variables, so a binding may not reuse any of their names.

Names have been braced everywhere since `0.13.0`. Sources in the earlier syntax, such as `x, y: {x} + {y}` and `x@(3D6) + {x}`, no longer parse, but [diagnostics](#diagnostics) migrate them.

## Examples

### One-shot evaluation

Rolling one standard six-sided die:

```rust
use xdy::evaluate;
use rand::rng;

let result = evaluate("1d6", vec![], vec![], &mut rng()).unwrap();

assert!(1 <= result.result && result.result <= 6);
assert!(result.records.len() == 1);
assert!(result.records[0].results.len() == 1);
assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
```

Rolling a variable number of dice, using an argument named `x` to supply the number of dice to roll:

```rust
use xdy::evaluate;
use rand::rng;

let result = evaluate("{x}: {x}D6", vec![3], vec![], &mut rng()).unwrap();

assert!(3 <= result.result && result.result <= 18);
assert!(result.records.len() == 1);
assert!(result.records[0].results.len() == 3);
assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
assert!(1 <= result.records[0].results[1] && result.records[0].results[1] <= 6);
assert!(1 <= result.records[0].results[2] && result.records[0].results[2] <= 6);
```

Rolling a variable number of dice, using an external variable named `x` to supply the number of dice to roll:

```rust
use xdy::evaluate;
use rand::rng;

let result =
    evaluate("{x}D6", vec![], vec![("x", 3)], &mut rng()).unwrap();

assert!(3 <= result.result && result.result <= 18);
assert!(result.records.len() == 1);
assert!(result.records[0].results.len() == 3);
assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
assert!(1 <= result.records[0].results[1] && result.records[0].results[1] <= 6);
assert!(1 <= result.records[0].results[2] && result.records[0].results[2] <= 6);
```

Evaluating an arithmetic expression involving different types of dice:

```rust
use xdy::evaluate;
use rand::rng;

let result = evaluate("1d6 + 2d8 - 1d10", vec![], vec![], &mut rng()).unwrap();

assert!(-7 <= result.result && result.result <= 21);
assert!(result.records.len() == 3);
assert!(result.records[0].results.len() == 1);
assert!(result.records[1].results.len() == 2);
assert!(result.records[2].results.len() == 1);
assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
assert!(1 <= result.records[1].results[0] && result.records[1].results[0] <= 8);
assert!(1 <= result.records[1].results[1] && result.records[1].results[1] <= 8);
assert!(
    1 <= result.records[2].results[0]
        && result.records[2].results[0] <= 10
);
```

Binding a subexpression to a name with `{name}@(expr)` and reusing it as `{name}`. The bound `3D6` is rolled exactly once and its total is reused, so the result is twice that total and only a single rolling record is produced:

```rust
use xdy::evaluate;
use rand::rng;

let result = evaluate("{x}@(3D6) + {x}", vec![], vec![], &mut rng()).unwrap();

assert!(result.records.len() == 1);
assert!(result.records[0].results.len() == 3);

let total = result.records[0].results.iter().sum::<i32>();
assert!(3 <= total && total <= 18);
assert!(result.result == 2 * total);
```

### Repeated evaluation

Compiling and optimizing a dice expression and evaluating it multiple times:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("3D6").unwrap();
let mut evaluator = Evaluator::new(function);
let results = (0..10)
    .flat_map(|_| evaluator.evaluate(vec![], &mut rng()))
    .collect::<Vec<_>>();

assert!(results.len() == 10);
assert!(results.iter().all(|result| 3 <= result.result && result.result <= 18));
```

Compiling and optimizing a dice expression with formal parameters and evaluating it multiple times with different arguments:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("{x}: 1D6 + {x}").unwrap();
let mut evaluator = Evaluator::new(function);
let results = (0..10)
   .flat_map(|x| evaluator.evaluate(vec![x], &mut rng()))
   .collect::<Vec<_>>();

assert!(results.len() == 10);
(0..10).for_each(|i| {
   let x = i as i32;
   assert!(1 + x <= results[i].result && results[i].result <= 6 + x);
});
```

Compiling and optimizing a dice expression with environmental variables and evaluating it multiple times:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("1D6 + {x}").unwrap();
let mut evaluator = Evaluator::new(function);
evaluator.bind("x", 3).unwrap();
let results = (0..10)
   .flat_map(|_| evaluator.evaluate(vec![], &mut rng()))
   .collect::<Vec<_>>();

assert!(results.len() == 10);
assert!(
    results.iter().all(|result| 4 <= result.result && result.result <= 9)
);
```

### Bounds analysis

Computing the bounds of a dice expression, without rolling anything:

```rust
use xdy::{compile, Evaluator};

let function = compile("3D6 + 1D8").unwrap();
let evaluator = Evaluator::new(function);
let bounds = evaluator.bounds_over([], []).unwrap();

assert_eq!(bounds.value, (4, 26).into());
assert_eq!(bounds.count, Some(1728));
```

Bindings need not be single values. Each may be an interval, and each may be omitted; an omitted binding is bounded by the whole of `i32`, so the bounds remain sound no matter how little the caller knows:

```rust
use xdy::{compile, Evaluator};

let function = compile("{x}: {x}D6 + {y}").unwrap();
let evaluator = Evaluator::new(function);

// Roll between 1 and 20 dice, and add a `y` known to be 3.
let bounds =
    evaluator.bounds_over([Some((1, 20).into())], [("y", 3.into())]).unwrap();
assert_eq!(bounds.value, (4, 123).into());
// The outcome count is exact only when every binding is a single value, so an
// interval binding withdraws it rather than inventing one.
assert_eq!(bounds.count, None);

// Say nothing about `y`, and the bounds say so in turn.
let bounds = evaluator.bounds_over([Some((1, 20).into())], []).unwrap();
assert_eq!(bounds.value, (i32::MIN + 1, i32::MAX).into());
```

### Dice budgets

Evaluation costs time and memory in proportion to the dice it rolls, and an expression like `{x}D6` rolls as many dice as `x` says, which may be over two billion. When the expression or its bindings are untrusted, evaluate within a dice budget. A roll that would exceed what remains of the budget is refused before it rolls anything, and a successful evaluation reports the dice it rolled, so one budget can be carried across several evaluations:

```rust
use rand::rng;
use xdy::{compile, EvaluationError, Evaluator};

let mut evaluator = Evaluator::new(compile("{x}: {x}D6").unwrap());

let evaluation = evaluator.evaluate_metered([3], &mut rng(), 10).unwrap();
assert_eq!(evaluation.dice, 3);

let error = evaluator.evaluate_metered([i32::MAX], &mut rng(), 10);
assert_eq!(
    error,
    Err(EvaluationError::DiceBudgetExhausted {
        requested: i32::MAX as u64,
        remaining: 10,
        consumed: 0
    })
);
```

Bounds analysis reports the worst case, so a caller can compare it against a budget without evaluating anything:

```rust
use xdy::{compile, Evaluator};

let evaluator = Evaluator::new(compile("{x}: ({x}D4)D6").unwrap());
let bounds = evaluator.bounds_over([Some((1, 3).into())], []).unwrap();
assert_eq!(bounds.dice, 15);
```

### Probability distributions

Computing the exact probability distribution of a dice expression costs far more than rolling it, and how much more depends on the expression and its bindings. So a distribution is built in two stages. First plan it, which estimates its cost without computing any weight. Then build it, within a budget of _steps_, the operations on weights, and _cells_, the entries of distributions alive at once. Every operation of the build charges its maximum before it runs, so one that would exceed the budget is refused before it expands anything, and a budget that covers the estimate always runs to completion:

```rust
use xdy::{BuildError, Budget, Evaluator, Unobserved, Weight, compile};

let evaluator = Evaluator::new(compile("{n}: {n}D6").unwrap());
let budget = Budget { steps: 1_000_000, cells: 1_000_000 };

let plan = evaluator.plan_distribution([10]).unwrap();
let estimate = plan.estimate();
assert!(estimate.steps <= budget.steps && estimate.cells <= budget.cells);
let distribution = plan.build(budget, &Unobserved).unwrap();
assert_eq!(distribution.total(), &Weight::from(6u64.pow(10)));

// Over two billion dice cost more than the budget allows. A caller may refuse
// the function by its estimate, before building anything; otherwise the build
// refuses it, and the refusal carries the estimate.
let plan = evaluator.plan_distribution([i32::MAX]).unwrap();
assert!(plan.estimate().steps > budget.steps);
assert!(matches!(
    plan.build(budget, &Unobserved),
    Err(BuildError::BudgetExhausted { .. })
));
```

The build reports its usage so far, and the estimate, at every charge, so a caller may show its progress, and cancel it, as when a user presses `Cancel`. A cancelled build answers no distribution:

```rust
use std::{cell::Cell, ops::ControlFlow};

use xdy::{BuildError, Budget, Cost, Evaluator, Usage, compile};

let evaluator = Evaluator::new(compile("100D6").unwrap());
let plan = evaluator.plan_distribution([]).unwrap();
let done = Cell::new(0.0);
let progress = |consumed: Usage, estimate: &Cost| {
    // Show the fraction of the estimated steps taken so far, and cancel the
    // build halfway through.
    done.set(consumed.steps as f64 / estimate.steps as f64);
    if done.get() < 0.5 { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
};
assert_eq!(plan.build(Budget::UNLIMITED, &progress), Err(BuildError::Cancelled));
assert!(done.get() >= 0.5);
```

When the estimate exceeds what a caller will spend, sampling estimates the distribution instead, by evaluating the function many times within one dice budget across every evaluation. The estimate also classifies the function, which explains its cost: the function below multiplies random values pairwise, so the build enumerates their pairs. The counts of a sample are exact, but the distribution that they describe is only an estimate, so they are `Sampled`, never a bare `Distribution`. The Dvoretzky–Kiefer–Wolfowitz inequality bounds the error of the whole estimate: at confidence `1 - δ`, after `n` samples, the cumulative distribution function of the counts is within `ε = √(ln(2/δ) / 2n)` of the exact one at every outcome at once:

```rust
use std::num::NonZeroU64;

use rand::rng;
use xdy::{Budget, Evaluator, Probability, Weight, compile};

let mut evaluator = Evaluator::new(compile("1D1000 * 1D1000 * 1D1000").unwrap());
let budget = Budget { steps: 1_000_000, cells: 1_000_000 };
let estimate = evaluator.plan_distribution([]).unwrap().estimate();
assert!(estimate.steps > budget.steps);
assert!(estimate.class.pairwise);

let n = NonZeroU64::new(10_000).unwrap();
let sampled = evaluator.sample([], &mut rng(), n, 3 * 10_000).unwrap();
assert_eq!(sampled.counts().total(), &Weight::from(10_000u32));

// At 95% confidence, every cumulative probability is within 0.0136 of exact.
let confidence = Probability::new(Weight::from(19u8), Weight::from(20u8)).unwrap();
assert_eq!(format!("{:.4}", sampled.error_bound(&confidence)), "0.0136");
```

### Diagnostics

`compile` and `evaluate` report errors as typed values. For editor-style feedback — rich error reports, suggested fixes, and caret-precise source spans — use `diagnostics::diagnose` instead. It diagnoses every syntax error in a single recovering parse, in time linear in the length of the source, and suggests a fix for each that it can; or, if the source parses cleanly, it validates the function, reporting duplicate parameters and misused bindings. Each suggestion is a list of edits of the original source, which `Suggestion::apply` applies in the manner of an editor's quick fix, and the fully corrected source is ready when every error was fixable. The doctor also migrates sources written in the syntax that preceded `0.13.0`:

```rust
use xdy::diagnostics::{DiagnosticKind, diagnose};

let result = diagnose("x, y: {x}D{y}");
assert!(result.diagnostics.iter().all(|diagnostic| matches!(
    diagnostic.kind,
    DiagnosticKind::BareIdentifier
)));
assert_eq!(result.corrected_source.as_deref(), Some("{x}, {y}: {x}D{y}"));
```

## Performance

`xDy` is _very fast_. Consider the following dice expression, where `x = 5`, `y = 2`, and `z = 2`:

```text
{x}, {y}, {z}: {x}D[-1, 0, 1, 3, 5] drop lowest {y} drop highest {z}
```

This dice expression involves argument binding, custom dice, and dropping values: roll `5` custom dice, each with faces `[-1, 0, 1, 3, 5]`, drop the `2` lowest results, and drop the `2` highest results, leaving only `1` die. On a 2026 MacBook Pro, `xDy` compiled and optimized this expression with mean time `5.304 µs` and evaluated it with mean time `115.92 ns`. Furthermore, `xDy` estimated the cost of its exact probability distribution with mean time `1.192 µs` and computed the distribution with mean time `4.593 µs`, where the enumerating builders of `0.13.0` took `1038.18 µs` serially and `702.56 µs` in parallel.

`xDy` provides a six-pass optimizer that rewrites IR into more efficient forms. The optimizer folds constant expressions, performs strength-reducing operations, eliminates common subexpressions, eliminates dead code, and coalesces registers. The optimizer also puts the operands of commutative operations in canonical order and merges chained constants, to improve opportunities for constant folding and strength reduction. Every pass is exact: an optimized function answers exactly what the unoptimized one does, even where arithmetic saturates. The optimizer runs its passes repeatedly, in predefined order, until a fixed point is reached, and then coalesces registers once. It takes time linear in the length of the function. `compile` and `evaluate` optimize fully; `compile_unoptimized` and `evaluate_unoptimized` skip the optimizer, and `StandardOptimizer` runs any chosen set of `Passes`.

### `nom` > `tree-sitter`

The switch from `tree-sitter` to `nom` for parsing delivered a substantial performance improvement across the full compilation pipeline (parse + compile + optimize). Benchmarking 316 expressions across all language features:

* **99% of expressions are faster** (313 of 316 cases improved by more than 1%)
* **Median improvement: 62%**, with simple expressions up to 89% faster
* **Aggregate pipeline time cut by 48%** (about 1.9× faster) across the entire benchmark corpus
* Constants improved 76%, variables 66%, dice 62%, ranges 60%, drop expressions 58%

Only 3 pathological cases (deeply right-nested with repeated subexpressions) showed regressions, attributable to optimizer behavior on the different AST structure. See [`benches/reports/comparison.md`](https://github.com/toddATavail/xdy/blob/main/xdy/benches/reports/comparison.md) for the full comparison.

### Linear time, constant stack

In `0.13.0`, the parser became an explicit-stack engine, and every other stage followed, so parsing, compilation, optimization, and diagnosis take time linear in the length of their input, and none of them consumes more of the machine stack for deeply nested input than for shallow input. Across 1,078 benchmark cases, the geometric mean time fell by 18%, and nestings that had parsed in exponential time now parse in microseconds. See [`benches/reports/iterative-comparison.md`](https://github.com/toddATavail/xdy/blob/main/xdy/benches/reports/iterative-comparison.md) for the full comparison.

## Safety

`xDy` is designed to be well-behaved for all inputs. Dice expression values are `i32` and all arithmetic operations saturate on overflow or underflow. Nothing recurses in proportion to its input: the parser, the validator, the compiler, and the diagnostics handle expressions nested up to a million deep, or as deep as fits in a source of several megabytes, on a 2 MiB stack, and the optimizer handles functions compiled from expressions nested up to 100,000 deep. When expressions or their bindings are untrusted, [dice budgets](#dice-budgets) bound the work of evaluation, and budgets of steps and cells bound the work of [probability distributions](#probability-distributions). Neither the compiler nor evaluator should panic or cause undefined behavior, even for invalid dice expressions and inputs, though client misuse of vector results can lead to panics. The main crate contains `unsafe` code in one place: the abstract syntax tree's `Drop` implementation dismantles stacked drop clauses (e.g., `4D6 drop lowest drop highest`) without recursion or allocation, by moving values out of place and back with `ptr::read` and `ptr::write`; this is sound because every value has exactly one owner throughout, and nothing can unwind while a value is out of place. Dedicated tests check this code under Miri: `cargo +nightly miri test -p xdy --lib tests::ast::test_miri`. No foreign function interfaces are involved.

## Cargo features

`xDy` provides one optional feature that can be enabled or disabled in your `Cargo.toml`:

* `serde`: Implements the `Serialize` and `Deserialize` traits for various types. Deserialization refuses a value that breaks its type's invariants, such as a `Function` that is not well formed (see `Function::validate`). This feature requires the `serde` crate, and is enabled by default.

## Project structure

The workspace comprises two crates and two Lean projects:

| Component | Description |
|-------|-------------|
| [`xdy`](https://github.com/toddATavail/xdy/tree/main/xdy) | The main crate: parser, compiler, optimizer, evaluator, distribution engine, and diagnostics. See the [crate README](https://github.com/toddATavail/xdy/blob/main/xdy/README.md) for an architectural deep-dive, including the instruction set reference, virtual machine model, and optimizer internals. |
| [`tree-sitter-xdy`](https://github.com/toddATavail/xdy/tree/main/tree-sitter-xdy) | A tree-sitter grammar for the xDy language, for editors and other tools. It is not used by the compilation pipeline, and ships no highlighting or folding queries yet. |
| [`lean`](https://github.com/toddATavail/xdy/tree/main/lean) | A Lean 4 model of the IR's semantics, which serves as an executable oracle for exact distributions: `just oracle` checks the corpus, and random programs, against it, as CI does. It is not part of the Rust build. See the [Lean README](https://github.com/toddATavail/xdy/blob/main/lean/README.md). |
| [`lean/spec`](https://github.com/toddATavail/xdy/tree/main/lean/spec) | A Lean 4 specification of the IR's semantics as probability distributions, with Mathlib, and the proofs that the forward pass that builds exact distributions answers it: `just spec` checks them. See the [specification README](https://github.com/toddATavail/xdy/blob/main/lean/spec/README.md). |

## Planned work

### Macros

I plan to introduce an `xdy!` macro for compiling statically known dice expressions. This macro will generate a Rust function that leverages the low-level IR primitives directly, thereby completely eliminating the (already very low) runtime overhead of the compiler and evaluator.

### More language features

Other language features that I plan to add include:

* Keep lowest: `1D6 keep lowest 1`, `2D8 keep lowest 2`, …
* Keep highest: `1D6 keep highest 1`, `2D8 keep highest 2`, …
* Reroll: `1D6 reroll =1`, `2D8 reroll >=5`, `2D8 reroll >=5 1 time`,
  `2D8 reroll <=2 3 times`, `1D20 reroll in [1, 2]`, `1D6 reroll =1 unlimited times`, …
* Explosion, e.g., rolling additional dice under certain conditions: `1D6 explode =6`, `2D8 explode >=7`, `1D6 explode =6 2 times`, `1D10 explode in 9..=10 unlimited times`, …
* Ranges in predicates: `in 1..3` (half-open) and `in 1..=3` (inclusive), alongside lists such as `in [1, 2]`.

### Foreign function interface (FFI)

I plan to expose an FFI that allows `xDy` to be used from other programming languages. This FFI will be designed to be safe and efficient, and will aim to minimize the amount of marshaling and manual resource management required by clients.
