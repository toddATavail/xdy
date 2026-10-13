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
| `dynamic_keeps` | `{n}, {m}: 5D6 keep highest {n} drop lowest {m}` | a keep count from a parameter; a drop after a keep |
| `dynamic_range` | `[1D3:2D6]` | range endpoints from registers |
| `explode` | `2D4 explode = 4` | `Explode`, limited to once |
| `explode_after_keep` | `3D4 keep highest 2 explode = 4` | an explode of the dice that a keep kept |
| `explode_limited` | `1D4 explode = 4 3 times` | an explode of a range, three times |
| `explode_list` | `2D[1, 3, 5] explode in [1, 5]` | an explode of custom dice by a list |
| `explode_ne` | `2D4 explode ≠ 2` | an explode by `≠`, which matches faces on both sides of its threshold |
| `explode_parameter` | `{x}: 2D6 explode ≥ {x}` | a limited explode, whose threshold is a parameter |
| `explode_range` | `1D6 explode in (1D2)..=(1D3)` | an inclusive range whose endpoints are registers |
| `explode_saturate` | `{x}: 2D6 explode ≥ {x} ∞ times` | an unlimited explode, whose threshold is a parameter: it saturates if every face matches, grows nothing if none does, and is refused otherwise |
| `explode_ties` | `2D4 explode ≥ 3 drop lowest reroll = 2` | a drop between chains of equal totals, which their contents order, and a reroll of the chain that it keeps |
| `external` | `1D6 + {y}` | an external variable |
| `extrema` | `lowest(highest(1D4, 1D6), 1D8)` | `Max` and `Min` of random registers |
| `keep_highest` | `4D6 keep highest 3` | `KeepHighest` |
| `max` | `{x}: {x}D1` | the optimizer's `Max` |
| `min` | `{x}: lowest({x}, 3)` | `Min` |
| `range` | `[1:6]` | `RollRange` |
| `reroll` | `3D4 reroll = 1` | `Reroll`, limited to once |
| `reroll_after_explode` | `2D3 explode = 3 reroll = 1` | a reroll of each die of the chains that an explode grew |
| `reroll_after_keep` | `4D6 keep highest 3 reroll < 3 2 times` | a reroll of the dice that a keep kept, twice |
| `reroll_cross_out` | `{x}: 2D{x} reroll ≤ 6 ∞ times` | an unlimited reroll whose predicate matches every face of some dice, known only at runtime |
| `reroll_list` | `2D[1, 3, 5] reroll in [1, 5]` | a reroll of custom dice by a list |
| `reroll_range` | `1D6 reroll in (1D2)..=(1D3)` | an inclusive range whose endpoints are registers |
| `reroll_unlimited` | `{x}: 2D6 reroll ≤ {x} ∞ times` | an unlimited reroll, whose threshold is a parameter |
| `shared_register` | `{x}@(3D6) + {x}` | one random register read twice |
| `standard_dice` | `3D6` | `RollStandardDice` |

The optimizer turns a custom die with contiguous faces, e.g., `1D[-1, 0, 1]`,
into a range, so `custom_dice` needs gaps between its faces to stay custom.
