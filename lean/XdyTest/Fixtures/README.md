# IR fixtures

Compiled xDy functions, as the JSON that Rust's `serde` feature writes. Each is
the output of `xdy::compile` (so fully optimized) for the source below, written
with `serde_json::to_string`.

The Rust test `test_lean_fixtures_are_current`, in
[`xdy/src/tests/oracle.rs`](../../../xdy/src/tests/oracle.rs), fails whenever a
fixture differs from what the current compiler writes, so the fixtures cannot
fall behind the IR or the optimizer. To rewrite them, then check the Lean
against them, run from the repository root:

```sh
XDY_BLESS_FIXTURES=1 cargo test -p xdy --lib lean_fixtures
just lean
```

The test's `FIXTURES` table mirrors the one below; add a fixture to both.

| Fixture | Source | Exercises |
|---------|--------|-----------|
| `arithmetic` | `{a}, {b}: -({a} * {b} - {a} / {b} % 3 ^ {b})` | parameters; `Mul`, `Div`, `Exp`, `Mod`, `Sub`, `Neg` |
| `constant` | `7` | an immediate `Return` |
| `custom_dice` | `2D[-1, 0, 1, 3, 5]` | `RollCustomDice` |
| `drop_lowest` | `4D6 drop lowest` | `DropLowest` |
| `dynamic_count` | `(1D3)D3` | a dice count from a register |
| `dynamic_custom_count` | `{x}: {x}D[1, 3]` | a custom dice count from a parameter |
| `dynamic_drops` | `{n}, {m}: 5D6 drop lowest {n} drop highest {m}` | `DropHighest`; drop counts from parameters |
| `dynamic_faces` | `3D(2D6)` | a face count from a register |
| `dynamic_range` | `[1D3:2D6]` | range endpoints from registers |
| `external` | `1D6 + {y}` | an external variable |
| `max` | `{x}: {x}D1` | the optimizer's `Max` |
| `range` | `[1:6]` | `RollRange` |
| `shared_register` | `{x}@(3D6) + {x}` | one random register read twice |
| `standard_dice` | `3D6` | `RollStandardDice` |

The optimizer turns a custom die with contiguous faces, e.g., `1D[-1, 0, 1]`,
into a range, so `custom_dice` needs gaps between its faces to stay custom.
