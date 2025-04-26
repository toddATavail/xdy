# xDy: Once-and-for-all dice expression compiler

`xDy` is an extremely fast dice expression compiler that can be used to generate pseudorandom numbers. It is designed for use within a variety of applications, such as role-playing games (RPGs) and simulations, but also suits other applications that must introduce controlled randomness or generate specific probability distributions. It is written in Rust for maximum performance, safety, and portability.

* [Examples](#examples)
	* [One-shot evaluation](#one-shot-evaluation)
	* [Repeated evaluation](#repeated-evaluation)
* [Performance](#performance)
* [Safety](#safety)
* [Language features](#language-features)
* [Cargo features](#cargo-features)
* [Planned work](#planned-work)

## Overview

`xDy` compiles dice expressions into reusable functions that can be applied to user-supplied pseudorandom number generators (pRNGs) to simulate dice rolls. Functions may define formal parameters or bind values supplied through an environment. A six-pass optimizing compiler produces efficient code for each dice expression, so the resulting functions can be used to generate dice rolls with minimal overhead. The generated code targets a dedicated intermediate representation (IR) that is designed specifically for dice expressions. An efficient evaluator interprets the IR to produce the final result.

Beyond generating just a final tally, `xDy` also provides detailed information about the individual dice rolls that contributed to the total. So `3D6 + 1D8` might produce a total of `18`, but it also tells you that the six-sided dice produced [`3`, `5`, `4`] and the eight-sided die produced `6`.

`xDy` also provides analytical capabilities. `xDy` can compute the bounds and probability distribution of a dice expression. For example, `xDy` can determine that `3D6 + 1D8` has a minimum value of `4`, has a maximum value of `26`, and puts the probability of rolling a `10` at `0.125` (`27/216`). Probability distributions can be computed serially or in parallel (with the `parallel-histogram` feature). For static dice expressions, `xDy` can calculate the total number of outcomes without iterating through the state space.

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

Rolling a variable number of dice, using an argument named `x` to supply the
number of dice to roll:

```rust
use xdy::evaluate;
use rand::rng;

let result = evaluate("x: {x}D6", vec![3], vec![], &mut rng()).unwrap();

assert!(3 <= result.result && result.result <= 18);
assert!(result.records.len() == 1);
assert!(result.records[0].results.len() == 3);
assert!(1 <= result.records[0].results[0] && result.records[0].results[0] <= 6);
assert!(1 <= result.records[0].results[1] && result.records[0].results[1] <= 6);
assert!(1 <= result.records[0].results[2] && result.records[0].results[2] <= 6);
```

Rolling a variable number of dice, using an external variable named `x` to
supply the number of dice to roll:

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

### Repeated evaluation

Compiling and optimizing a dice expression and evaluating it multiple times:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("3D6")?;
let mut evaluator = Evaluator::new(function);
let results = (0..10)
    .flat_map(|_| evaluator.evaluate(vec![], &mut rng()))
    .collect::<Vec<_>>();

assert!(results.len() == 10);
assert!(results.iter().all(|result| 3 <= result.result && result.result <= 18));
```

Compiling and optimizing a dice expression with formal parameters and evaluating
it multiple times with different arguments:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("x: 1D6 + {x}")?;
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

Compiling and optimizing a dice expression with environmental variables and
evaluating it multiple times:

```rust
use xdy::{compile, Evaluator};
use rand::rng;

let function = compile("1D6 + {x}")?;
let mut evaluator = Evaluator::new(function);
evaluator.bind("x", 3)?;
let results = (0..10)
   .flat_map(|x| evaluator.evaluate(vec![], &mut rng()))
   .collect::<Vec<_>>();

assert!(results.len() == 10);
assert!(
    results.iter().all(|result| 4 <= result.result && result.result <= 9)
);
```

## Performance

`xDy` is _very fast_. Consider the following dice expression, where `x = 4`, `y = 2`, and `z = 2`:

```text
x, y, z: {x}D[-1, 0, 1, 3, 5] drop lowest {y} drop highest {z}
```

This dice expression involves argument binding, custom dice, and dropping values. On a 2023 MacBook Pro, `xDy` compiled and optimized this expression with mean time `22.664 µs` and evaluated it with mean time `216.89 ns`. Furthermore, the serial probability distribution calculation took mean time `394.85 µs` and the parallel calculation took mean time `291.46 µs`.

`xDy` provides a six-pass optimizer that rewrites IR into more efficient forms. The optimizer folds constant expressions, performs strength-reducing operations, eliminates common subexpressions, eliminates dead code, and coalesces registers. The optimizer can also reorder commutative operations and operands to improve opportunities for constant folding and strength reduction. The optimizer runs all passes repeatedly, in predefined order, until a fixed point is reached. The optimizer is designed to be fast and effective, but it can be disabled if desired.

## Safety

`xDy` is designed to be well-behaved for all inputs. Dice expression values are `i32` and all arithmetic operations saturate on overflow or underflow. Neither the compiler nor evaluator should panic or cause undefined behavior, even for invalid dice expressions and inputs, though client misuse of vector results can lead to panics. `unsafe` code is limited to the `tree-sitter` dependency, and pertains exclusively to foreign function interfaces (FFIs) that are not exposed to users of the main crate.

## Language features

`xDy` aims to achieve sufficient expressiveness to cover the gamut of use cases established by popular RPG systems. The following features are supported:

* Constants: `-1`, `0`, `1`, `2`, `3`, …
* Standard dice, e.g., `xDy` for all `x` and `y`:
  `1D6`, `2D8`, `3D10`, `4D17`, …
* Custom dice, e.g., Fudge dice:
  `1D[-1, 0, 1]`, `2D[-1, 0, 1, 3, 5]`, `3D[1, 1, 2, 3, 5, 8]`, …
* Arithmetic
  * Negation: `-1`, `-1D4`, …
  * Addition: `1 + 1`, `1 + 1D4`, `1D6 + 1D8`, …
  * Subtraction: `3 - 2`, `1 - 1D4`, `1D6 - 1D8`, …
  * Multiplication: `2 * 3`, `2 * 1D4`, `2D6 * 1D8`, …
  * Division: `6 / 2`, `6 / 1D4`, `1D6 / 1D8`, …
  * Modulus: `6 % 2`, `6 % 1D4`, `1D6 % 1D8`, …
  * Exponentiation: `2 ^ 3`, `2 ^ 1D4`, `2D6 ^ 1D8`, …
* Grouping: `(1 + 2) * 3`, `1 + (2 * 3D4)`, `(3 * 2)D5`, `4D(5 - 2)`, …
* Drop lowest: `1D6 drop lowest 1`, `2D8 drop lowest 2`, …
* Drop highest: `1D6 drop highest 1`, `2D8 drop highest 2`, …
* Formal parameters: `x: 1D6 + {x}`, `y: 1D6 + {y}`, `x, y: {x}D{y}`, …
* Environmental variables: `1D6 + {x}`, `1D6 + {y}`, `{x}D{y}`, …
* Dynamic expressions: `3D(2D6)`, `x: ({x}D3)D8`, …

Environmental variables permit background state to be associated with an evaluator, while formal parameters permit dynamic state to be passed into each evaluation. Both features can be used to parameterize dice expressions and make them more flexible. Environmental variables could easily cover, for example, the attributes and skills of a character in an RPG, while formal parameters could cover the situational modifiers applied to a particular roll.

## Cargo features

`xDy` provides several optional features that can be enabled or disabled in your `Cargo.toml`:

* `parallel-histogram`: Enables parallel computation of probability distributions. This feature requires the `rayon` crate, and is enabled by default.
* `serde`: Implements the `Serialize` and `Deserialize` traits for various types. This feature requires the `serde` crate, and is enabled by default.

## Planned work

### Error reporting

Sadly, I discovered that `tree-sitter` does not support good syntax error reporting until _after_ all of the happy paths were working. I plan to introduce another parser that can provide better error messages if and only if `tree-sitter` fails to parse a dice expression.

### Macros

I plan to introduce an `xdy!` macro for compiling statically known dice expressions. This macro will generate a Rust function that leverages the low-level IR primitives directly, thereby completely eliminating the (already very low) runtime overhead of the compiler and evaluator.

### More language features

Other language features that I plan to add include:

* Subexpression naming: `{x}@(1D6) + {x}`, `x, y: {z}@({x} + {y}) + {z}D6`, …
* Keep lowest: `1D6 keep lowest 1`, `2D8 keep lowest 2`, …
* Keep highest: `1D6 keep highest 1`, `2D8 keep highest 2`, …
* Reroll: `1D6 reroll =1`, `2D8 reroll >=5`, `2D8 reroll >=5 1 time`,
  `2D8 reroll <=2 3 times`, …
* Explosion, e.g., rolling additional dice under certain conditions: `1D6 explode =6`, `2D8 explode >=7`, `1D6 explode =6 2 times`, …

### Foreign function interface (FFI)

I plan to expose an FFI that allows `xDy` to be used from other programming languages. This FFI will be designed to be safe and efficient, and will aim to minimize the amount of marshaling and manual resource management required by clients.
