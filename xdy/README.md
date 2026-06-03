# xDy crate

The `xdy` crate is the core library of the xDy project. It compiles dice expressions into reusable functions, optimizes them, and evaluates them against client-supplied pseudorandom number generators. For a high-level overview of the project, language features, and usage examples, see the [project README](../README.md).

This document covers the internal architecture for contributors and curious power users.

* [Architecture](#architecture)
	* [Compilation pipeline](#compilation-pipeline)
	* [Source spans](#source-spans)
	* [Semantic validation](#semantic-validation)
	* [Intermediate representation](#intermediate-representation)
	* [Evaluator](#evaluator)
	* [Optimizer](#optimizer)
	* [Histogram engine](#histogram-engine)
	* [Diagnostics](#diagnostics)
* [Instruction set reference](#instruction-set-reference)
	* [Addressing modes](#addressing-modes)
	* [Roll instructions](#roll-instructions)
	* [Drop instructions](#drop-instructions)
	* [Reduce instructions](#reduce-instructions)
	* [Arithmetic instructions](#arithmetic-instructions)
	* [Control instructions](#control-instructions)
* [Cargo features](#cargo-features)
* [Safety](#safety)

## Architecture

### Compilation pipeline

Source code flows through a four-stage happy path: parse, validate, compile, optimize. The `compile()` convenience function drives the full pipeline; `compile_unoptimized()` skips the optimizer; the sad path (syntactic and semantic errors, rich diagnostics) is described in [Diagnostics](#diagnostics).

```mermaid
graph LR
    A["Source Code<br/><code>&str</code>"] --> B["Parser<br/><code>Parser::parse</code>"]
    B --> C["AST<br/><code>ast::Function</code>"]
    C --> V["Validator<br/><code>Validator::validate</code>"]
    V --> D["Compiler<br/><code>Compiler::compile</code>"]
    D --> E["Unoptimized IR<br/><code>Function</code>"]
    E --> F["Optimizer<br/><code>StandardOptimizer</code>"]
    F --> G["Optimized IR<br/><code>Function</code>"]
    style A fill:#f9f,stroke:#333,color:#000
    style G fill:#9f9,stroke:#333,color:#000
```

**Parser.** A [nom](https://docs.rs/nom)-based combinator parser recognizes the xDy grammar and produces an abstract syntax tree (`ast::Function`). The grammar supports operator precedence via recursive descent: `add_sub` → `mul_div_mod` → `unary` → `exponent` → `primary`. See `parser::combinators` for the full set of production rules; the Rustdoc includes a railroad diagram generated from the EBNF grammar. Every AST node carries a `SourceSpan` referencing its byte range in the original input — see [Source spans](#source-spans) below.

**Validator.** The `Validator` performs semantic checks on the parsed AST before code generation. It catches duplicate formal parameter names and enforces the invariants of local bindings — use-before-bind (including self-reference), rebinding, and collision with a formal parameter — with room for further checks as downstream language features demand them. See [Semantic validation](#semantic-validation).

**Compiler.** The `Compiler` implements the `ASTVisitor` trait and walks the AST in a single pass to emit IR instructions. It uses static single assignment (SSA) form — every instruction writes to a fresh register or rolling record. Parameters are allocated first (in declaration order), then external variables (depth-first, left-to-right), ensuring deterministic register layout. Keeping validation as a separate pass lets the compiler's `ASTVisitor::Error` be `Infallible`.

**Optimizer.** The `StandardOptimizer` applies five transformation passes in a fixed-point loop, then a final register coalescing pass. See [Optimizer](#optimizer) below.

### Source spans

Every AST node carries a `SourceSpan { start, end }` byte range referencing the original source text. The `Spanned` trait provides uniform access to these spans (`span()`) and an `untethered()` operation that zeroes every span in a value recursively, enabling position-independent structural comparison for tests and round-trip fidelity checks.

Spans flow end-to-end: parser combinators compute them from `nom_locate::LocatedSpan` positions; semantic errors (`CompilationError::DuplicateParameter`) carry the spans of both the first and duplicate occurrences; diagnostics (`Diagnostic`, `RelatedLabel`) surface them as byte ranges in the original source. This layer is a prerequisite for the planned `xdy!` procedural macro (which translates compiler errors to `compile_error!` at token-precise spans) and for editor integrations that want to underline or gutter-highlight errors.

### Semantic validation

The `Validator` pass sits between parsing and code generation and rejects ASTs that are syntactically well-formed but semantically invalid. It implements `ASTVisitor` so callers can drive it directly (e.g., to interleave it with other analyses), but `Validator::validate()` is the ordinary entry point.

The current checks:

| Error | Example | Payload |
|-------|---------|---------|
| `DuplicateParameter` | `x, x: {x} + 1` | name, first and duplicate occurrence spans |
| `BindingCollidesWithParameter` | `x: x@(3D6) + {x}` | name, parameter span, binding-site span |
| `DuplicateBinding` | `x@(3D6) + x@(1D4)` | name, first and duplicate binding-site spans |
| `UseBeforeBind` | `{x} + x@(3D6)` | name, reference span, binding-site span |

The binding checks (`BindingCollidesWithParameter`, `DuplicateBinding`, `UseBeforeBind`) enforce the single flat namespace and forward-only reference rules of subexpression naming; `UseBeforeBind` also rejects self-reference inside a bound expression, such as `x@(1 + {x})`. Additional checks will be added here as further language features land.

### Intermediate representation

The IR is a simple register transfer language (RTL) with no control flow — all instructions reside in a single basic block. The machine model provides two register files:

- **Register bank** (`@0`, `@1`, …): holds `i32` values for parameters, external variables, and computed intermediates.
- **Rolling record bank** (`⚅0`, `⚅1`, …): holds the individual results of dice rolls and range selections, along with drop counters.

Each operand uses one of three addressing modes:

| Mode | Notation | Description |
|------|----------|-------------|
| Immediate | `#N` | Constant `i32` embedded in the instruction |
| Register | `@N` | Index into the register bank |
| RollingRecord | `⚅N` | Index into the rolling record bank |

See [Instruction set reference](#instruction-set-reference) for the complete ISA.

### Evaluator

The evaluator is a linear instruction interpreter. For each evaluation it:

1. Allocates a register bank (sized at compile time) and a rolling record bank.
2. Loads arguments into parameter registers and environment bindings into external variable registers.
3. Walks the instruction stream sequentially — there is no branching, so the program counter simply increments.
4. Returns the final `Evaluation`, which includes both the `i32` result and the complete `Vec<RollingRecord>` for display.

```mermaid
graph TD
    subgraph VM["Evaluator VM"]
        direction TB
        PC["Program Counter"]
        subgraph RF["Register Bank (i32)"]
            R0["@0: param"]
            R1["@1: extern"]
            RN["@N: computed"]
        end
        subgraph RR["Rolling Record Bank"]
            RR0["⚅0: dice results"]
            RR1["⚅1: range results"]
        end
    end
    F["Function (IR)"] --> PC
    RNG["pRNG"] --> RR
    ARGS["Arguments"] --> RF
    ENV["Environment"] --> RF
    VM --> OUT["Evaluation<br/>result + records"]
    style VM fill:#e8f4fd,stroke:#333,color:#000
    style RF fill:#d4edda,stroke:#333,color:#000
    style RR fill:#fff3cd,stroke:#333,color:#000
```

All arithmetic saturates to `i32::MIN`/`i32::MAX`. Division by zero yields zero. `0^0 = 1`. Negative exponents yield zero.

The evaluator also provides `bounds()`, which computes static `min`/`max` bounds and (when possible) the total outcome count — without requiring an RNG.

### Optimizer

The `StandardOptimizer` applies six passes. The first five run in a fixed-point loop; the sixth runs once at the end:

```mermaid
graph TD
    A["Input Function (SSA)"] --> B["Common Subexpression Elimination"]
    B --> C["Constant Commuting"]
    C --> D["Constant Folding"]
    D --> E["Strength Reduction"]
    E --> F["Dead Code Elimination"]
    F --> G{"Changed?"}
    G -- Yes --> B
    G -- No --> H["Register Coalescing"]
    H --> I["Optimized Function"]
    style A fill:#f9f,stroke:#333,color:#000
    style I fill:#9f9,stroke:#333,color:#000
    style G fill:#ff9,stroke:#333,color:#000
```

| Pass | Effect |
|------|--------|
| **CSE** | Identifies identical instructions and replaces duplicates with references to the first occurrence |
| **Constant commuting** | Moves immediate operands to the left side of commutative operators, creating folding opportunities |
| **Constant folding** | Evaluates instructions whose operands are all immediates at compile time |
| **Strength reduction** | Replaces expensive operations with cheaper equivalents (e.g., multiply by power of two → shift) |
| **Dead code elimination** | Removes instructions whose results are never consumed |
| **Register coalescing** | Merges equivalent registers to reduce the register bank size; breaks SSA form, so it runs last |

### Histogram engine

The histogram engine computes the complete probability distribution of a dice expression by exhaustive enumeration. It uses a state-machine approach:

1. Each `EvaluationState` suspends at roll/range instructions.
2. For each possible outcome of the suspended instruction, a successor state is generated.
3. States that reach `Return` contribute their result to the histogram.
4. A configurable gas tank (`MAX_SUCCESSORS = 30`) bounds the branching factor per instruction.

Two builders are provided: `SerialHistogramBuilder` and `ParallelHistogramBuilder` (the latter requires the `parallel-histogram` feature and uses `rayon`).

### Diagnostics

The `diagnostics` module is the sad-path counterpart to the [compilation pipeline](#compilation-pipeline): it produces rich `Diagnostic` values — each carrying an error kind, a source span, a human-readable message, optional secondary `RelatedLabel`s, and one or more `Suggestion`s (complete corrected source strings, with placeholder regions for ambiguous fixes) — designed to power IDE-style error reporting on every keystroke. It runs as two sequential passes: a **syntactic fix-and-retry loop** over parse errors, followed by a **semantic validator pass** that runs only on clean parses.

```mermaid
graph TD
    S["Source Code<br/><code>&str</code>"] --> P["Parser::parse"]
    P -->|OK| V["Validator::validate"]
    P -->|Err| A["Analyze parse error"]
    A --> F["Apply fix to source"]
    F --> OM["OffsetMap records edit"]
    OM --> P
    V -->|OK| DONE["DiagnoseResult<br/>diagnostics + corrected_source"]
    V -->|Err| SEM["Build semantic Diagnostic"]
    SEM --> DONE
    A -.unfixable.-> DONE
    style S fill:#f9f,stroke:#333,color:#000
    style DONE fill:#9f9,stroke:#333,color:#000
    style P fill:#ffd,stroke:#333,color:#000
    style V fill:#ffd,stroke:#333,color:#000
```

**Syntactic pass.** Each fix inserts, removes, or wraps characters in the source string, making progress toward a clean parse. An `OffsetMap` tracks cumulative adjustments so that diagnostic spans are mapped back to positions in the original source. The loop terminates when parsing succeeds or when an unfixable error is encountered.

**Semantic pass.** After the loop, the doctor runs the `Validator` on the parsed AST — but **only when the original source parsed cleanly**, that is, when no fixes were applied. Semantic diagnostics point at spans the user actually typed; running the validator on a fix-synthesized source would attach diagnostics to characters the user never wrote, which is confusing, so semantic checks wait for a syntactically valid source.

## Instruction set reference

### Addressing modes

| Mode | Syntax | Description |
|------|--------|-------------|
| `Immediate(N)` | `#N` | Constant `i32` value |
| `Register(N)` | `@N` | General-purpose register |
| `RollingRecord(N)` | `⚅N` | Rolling record (dice/range results) |

### Roll instructions

| Instruction | Syntax | Semantics |
|-------------|--------|-----------|
| `RollRange` | `⚅N ← roll range S:E` | Select a random value from the inclusive range `[S, E]` and store it in rolling record `N` |
| `RollStandardDice` | `⚅N ← roll standard dice CDF` | Roll `C` standard dice with `F` faces each, storing all results in rolling record `N` |
| `RollCustomDice` | `⚅N ← roll custom dice CD[f₁, f₂, …]` | Roll `C` custom dice with the specified face values, storing all results in rolling record `N` |

### Drop instructions

| Instruction | Syntax | Semantics |
|-------------|--------|-----------|
| `DropLowest` | `⚅N ← drop lowest K from ⚅N` | Mark the `K` lowest results in rolling record `N` as dropped |
| `DropHighest` | `⚅N ← drop highest K from ⚅N` | Mark the `K` highest results in rolling record `N` as dropped |

### Reduce instructions

| Instruction | Syntax | Semantics |
|-------------|--------|-----------|
| `SumRollingRecord` | `@N ← sum rolling record ⚅M` | Sum the non-dropped results of rolling record `M` into register `N` (saturating) |

### Arithmetic instructions

All arithmetic instructions saturate on overflow/underflow.

| Instruction | Syntax | Semantics |
|-------------|--------|-----------|
| `Add` | `@N ← A + B` | Saturating addition |
| `Sub` | `@N ← A - B` | Saturating subtraction |
| `Mul` | `@N ← A * B` | Saturating multiplication |
| `Div` | `@N ← A / B` | Saturating division; `x / 0 = 0` |
| `Mod` | `@N ← A % B` | Saturating remainder; `x % 0 = 0` |
| `Exp` | `@N ← A ^ B` | Saturating exponentiation; `0^0 = 1`, `x^(-n) = 0` |
| `Neg` | `@N ← -A` | Saturating negation |

### Control instructions

| Instruction | Syntax | Semantics |
|-------------|--------|-----------|
| `Return` | `return A` | Set the function result to `A` and terminate |

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel-histogram` | Yes | Parallel probability distribution computation via `rayon` |
| `serde` | Yes | `Serialize`/`Deserialize` for IR, evaluation, and histogram types |

## Safety

`xDy` is designed to be well-behaved for all inputs. Dice expression values are `i32` and all arithmetic operations saturate on overflow or underflow. Neither the compiler nor evaluator should panic or cause undefined behavior, even for invalid dice expressions and inputs, though client misuse of vector results can lead to panics. The main crate contains a single small block of `unsafe` code in the parser, where it reconstructs a source span after trimming trailing whitespace from an identifier; it is sound because every value it uses derives from a span that `nom` has already validated against the same source text. No foreign function interfaces are involved.
