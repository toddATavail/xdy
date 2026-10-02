# Iterative engine: before/after benchmark comparison

Benchmarks of xDy before the work that made parsing iterative and after it,
at the release of 0.13.0, measured by Criterion 0.8.2 with its default
warm-up and 100 samples (the nesting and diagnose groups use flat sampling
and a 1 s measurement time).

- **Platform**: macOS 26.6.2, Apple M5 Max (18 cores), 128 GB
- **Rust**: 1.97.1 (stable), release profile
- **Before**: commit `bbd568c` (xDy 0.12.0 plus the benchmarks), with the lockfile of `5c30edd`
- **After**: the 0.13.0 release candidate; the optimization, nesting compile, and histogram groups at `92a87ea`, the others at `5552937`, which differs from it only in the optimizer and the histogram meter
- **Runs**: separate worktrees and target directories, one after the other on an otherwise idle machine, on 2026-09-27 and 2026-09-28

This is a snapshot of 0.13.0. In 0.14.0, a forward pass that builds exact
distributions replaced the enumerating serial and parallel histogram builders
measured here, so the serial and parallel histogram groups no longer exist;
the benchmarks now estimate and build exact distributions instead.

The raw medians are in `iterative-before.txt`, `iterative-before-parallel.txt`,
`iterative-after.txt`, and `iterative-after-optimizer.txt`. Every case of the
optimization, parse, and nesting groups ran on both sides. The
evaluation and histogram groups hold thousands of cases, so the after side ran
every tenth case of each, and the parallel histograms ran every tenth case on
both sides. Cases are matched by their source text with the braces of names
removed, since 0.13.0 braces every name; a few corpus entries changed beyond
their braces and are unmatched. Δ is the change in median time; a negative Δ is
faster.

## Summary

| Group | Cases | Faster (>2%) | ±2% | Slower (>2%) | Median Δ | Geometric mean ratio | Aggregate before | Aggregate after | Aggregate Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimizations | 333 | 252 | 33 | 48 | -12.4% | 0.852 | 1.81 ms | 1.34 ms | -26.1% |
| parse | 338 | 193 | 19 | 126 | -9.7% | 0.890 | 721.57 µs | 484.87 µs | -32.8% |
| nesting parse | 38 | 33 | 0 | 5 | -36.8% | 0.147 | 190.04 ms | 596.27 µs | -99.7% |
| nesting compile | 38 | 18 | 0 | 20 | +11.6% | 0.276 | 191.36 ms | 1.53 ms | -99.2% |
| evaluations | 200 | 116 | 73 | 11 | -2.6% | 0.971 | 16.80 µs | 16.38 µs | -2.5% |
| serial histograms | 56 | 3 | 29 | 24 | +1.5% | 1.021 | 13.16 ms | 13.22 ms | +0.5% |
| parallel histograms | 75 | 26 | 11 | 38 | +2.5% | 1.074 | 17.36 ms | 19.13 ms | +10.2% |

Across all 1078 matched cases, the geometric mean ratio of after to before is 0.820.

## Distribution

| Range | optimizations | parse | nesting parse | nesting compile | evaluations | serial histograms | parallel histograms |
|---|---:|---:|---:|---:|---:|---:|---:|
| < −50% | 11 | 4 | 17 | 13 | 0 | 0 | 0 |
| −50% to −30% | 48 | 32 | 5 | 3 | 0 | 0 | 0 |
| −30% to −10% | 132 | 133 | 9 | 0 | 7 | 0 | 6 |
| −10% to −2% | 61 | 24 | 2 | 2 | 109 | 3 | 20 |
| ±2% | 33 | 19 | 0 | 0 | 73 | 29 | 11 |
| +2% to +10% | 27 | 55 | 1 | 1 | 11 | 24 | 14 |
| +10% to +30% | 3 | 68 | 1 | 4 | 0 | 0 | 12 |
| > +30% | 18 | 3 | 3 | 15 | 0 | 0 | 12 |

## Pathological nesting: time against depth

Each family nests one construct around a constant. In 0.12.0, groups, groups
followed by dice, and bindings parsed in time exponential in depth; the other
families recursed once per level, and overflow the stack at depths beyond
those benchmarked. After the epic, every family parses and compiles in time
linear in depth. The compile group includes optimization after the epic, which
compile() skipped before.

### group

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.54 µs | 977.9 ns | 1.62 µs | 1.56 µs |
| 2 | 3.50 µs | 1.53 µs | 3.61 µs | 2.12 µs |
| 4 | 15.40 µs | 2.52 µs | 15.62 µs | 3.32 µs |
| 8 | 252.96 µs | 4.52 µs | 254.76 µs | 5.50 µs |
| 12 | 4.06 ms | 6.34 µs | 4.08 ms | 7.43 µs |
| 16 | 64.77 ms | 8.45 µs | 65.19 ms | 9.68 µs |

### group then dice

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.17 µs | 1.10 µs | 1.25 µs | 3.34 µs |
| 2 | 2.18 µs | 1.63 µs | 2.28 µs | 4.00 µs |
| 4 | 8.23 µs | 2.63 µs | 8.36 µs | 5.07 µs |
| 8 | 127.21 µs | 4.63 µs | 128.70 µs | 7.29 µs |
| 12 | 2.03 ms | 6.45 µs | 2.03 ms | 9.22 µs |
| 16 | 32.45 ms | 8.48 µs | 33.10 ms | 11.73 µs |

### binding

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.72 µs | 1.06 µs | 1.90 µs | 1.75 µs |
| 2 | 3.96 µs | 1.63 µs | 4.21 µs | 2.37 µs |
| 4 | 17.59 µs | 2.68 µs | 18.08 µs | 3.86 µs |
| 8 | 289.03 µs | 4.61 µs | 291.78 µs | 6.57 µs |
| 12 | 4.63 ms | 6.69 µs | 4.68 ms | 8.90 µs |
| 16 | 74.45 ms | 8.61 µs | 74.50 ms | 11.96 µs |

### dice count

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.16 µs | 1.11 µs | 1.26 µs | 3.31 µs |
| 4 | 3.68 µs | 2.96 µs | 3.94 µs | 11.40 µs |
| 16 | 23.13 µs | 10.05 µs | 23.99 µs | 41.19 µs |
| 64 | 360.84 µs | 38.35 µs | 364.80 µs | 159.39 µs |
| 256 | 6.12 ms | 145.93 µs | 6.20 ms | 624.68 µs |

### negation

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 221.2 ns | 241.7 ns | 277.6 ns | 665.8 ns |
| 4 | 371.9 ns | 498.0 ns | 447.2 ns | 1.59 µs |
| 16 | 878.0 ns | 1.40 µs | 1.13 µs | 4.90 µs |
| 64 | 3.41 µs | 4.65 µs | 4.90 µs | 18.46 µs |
| 256 | 14.15 µs | 17.70 µs | 19.86 µs | 68.52 µs |

### exponent

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.01 µs | 787.4 ns | 1.08 µs | 1.80 µs |
| 4 | 2.24 µs | 1.62 µs | 2.42 µs | 3.54 µs |
| 16 | 7.25 µs | 5.00 µs | 7.71 µs | 10.48 µs |
| 64 | 27.66 µs | 18.07 µs | 29.61 µs | 35.73 µs |
| 256 | 112.14 µs | 70.64 µs | 118.78 µs | 134.66 µs |

### range

| Depth | Parse before | Parse after | Compile before | Compile after |
|---:|---:|---:|---:|---:|
| 1 | 1.28 µs | 1.09 µs | 1.35 µs | 2.11 µs |
| 4 | 3.53 µs | 2.93 µs | 3.73 µs | 5.09 µs |
| 16 | 12.16 µs | 10.02 µs | 12.66 µs | 16.19 µs |
| 64 | 47.69 µs | 38.59 µs | 50.20 µs | 58.58 µs |
| 256 | 193.32 µs | 150.11 µs | 203.05 µs | 223.17 µs |

## Regressions

The cases that slowed by more than 10%, with the causes identified during the epic:

| Group | Case | Before | After | Δ |
|---|---|---:|---:|---:|
| nesting compile | `negation/16` | 1.13 µs | 4.90 µs | +332.8% |
| nesting compile | `negation/64` | 4.90 µs | 18.46 µs | +276.6% |
| nesting compile | `negation/4` | 447.2 ns | 1.59 µs | +255.0% |
| nesting compile | `negation/256` | 19.86 µs | 68.52 µs | +245.1% |
| nesting compile | `dice count/4` | 3.94 µs | 11.40 µs | +189.6% |
| nesting compile | `group then dice/1` | 1.25 µs | 3.34 µs | +166.5% |
| nesting compile | `dice count/1` | 1.26 µs | 3.31 µs | +163.8% |
| nesting compile | `negation/1` | 277.6 ns | 665.8 ns | +139.8% |
| optimizations | `case 231: {x}, {y}: {x}D1 drop lowest {y}` | 3.89 µs | 8.31 µs | +113.5% |
| optimizations | `case 240: {x}, {y}: {x}D1 drop highest {y}` | 3.94 µs | 8.35 µs | +112.0% |
| parallel histograms | `case 390: ("{x}, {y}: {x}D{y} * {x}D{y}", [3, 4], [])` | 362.95 µs | 739.96 µs | +103.9% |
| parallel histograms | `case 730: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [5, 2], [])` | 363.24 µs | 662.11 µs | +82.3% |
| parallel histograms | `case 810: ("(1D3)D(1D3) + (1D3)D[-1, 0, 1]", [], [])` | 387.16 µs | 684.93 µs | +76.9% |
| nesting compile | `group then dice/2` | 2.28 µs | 4.00 µs | +75.1% |
| optimizations | `case 233: {x}: {x}D1 drop lowest 1` | 3.36 µs | 5.88 µs | +75.0% |
| nesting compile | `dice count/16` | 23.99 µs | 41.19 µs | +71.7% |
| optimizations | `case 242: {x}: {x}D1 drop highest 1` | 3.43 µs | 5.87 µs | +71.5% |
| parallel histograms | `case 440: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 6], [])` | 266.92 µs | 450.83 µs | +68.9% |
| optimizations | `case 264: {x}: {x}D[1] drop highest 1 drop lowest 1` | 6.03 µs | 10.17 µs | +68.7% |
| optimizations | `case 230: {x}: 10D1 drop lowest {x}` | 3.44 µs | 5.79 µs | +68.6% |
| nesting compile | `exponent/1` | 1.08 µs | 1.80 µs | +66.9% |
| optimizations | `case 265: {x}: {x}D[1, 1, 1] drop highest 1 drop lowest 1` | 6.17 µs | 10.29 µs | +66.9% |
| optimizations | `case 239: {x}: 10D1 drop highest {x}` | 3.54 µs | 5.84 µs | +65.2% |
| optimizations | `case 163: {x}D1` | 2.05 µs | 3.35 µs | +62.8% |
| nesting parse | `negation/16` | 878.0 ns | 1.40 µs | +58.9% |
| optimizations | `case 257: {x}: {x}D[1] drop highest 3` | 3.90 µs | 6.18 µs | +58.7% |
| optimizations | `case 258: {x}: {x}D[1, 1, 1] drop highest 3` | 4.05 µs | 6.34 µs | +56.6% |
| nesting compile | `range/1` | 1.35 µs | 2.11 µs | +55.8% |
| parallel histograms | `case 690: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [4, 2], [])` | 208.03 µs | 321.89 µs | +54.7% |
| optimizations | `case 218: {x}: {x}D1` | 2.23 µs | 3.44 µs | +54.6% |
| optimizations | `case 250: {x}: {x}D[1] drop lowest 3` | 3.95 µs | 6.09 µs | +54.1% |
| optimizations | `case 251: {x}: {x}D[1, 1, 1] drop lowest 3` | 4.09 µs | 6.24 µs | +52.5% |
| parallel histograms | `case 630: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [3, 10, 2], [])` | 274.35 µs | 403.47 µs | +47.1% |
| parallel histograms | `case 560: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [3, 10, 2], [])` | 276.22 µs | 405.30 µs | +46.7% |
| nesting compile | `exponent/4` | 2.42 µs | 3.54 µs | +46.1% |
| parallel histograms | `case 320: ("{x}D{y}", [], [("x", 4), ("y", 10)])` | 752.50 µs | 1.10 ms | +45.6% |
| optimizations | `case 223: {x}: {x}D[1]` | 2.65 µs | 3.86 µs | +45.4% |
| optimizations | `case 248: {x}: 10D1 drop lowest {x} drop highest 1` | 6.62 µs | 9.58 µs | +44.7% |
| optimizations | `case 224: {x}: {x}D[1, 1, 1]` | 2.79 µs | 3.98 µs | +42.8% |
| optimizations | `case 266: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 1 drop highest 1` | 8.13 µs | 11.25 µs | +38.3% |
| parallel histograms | `case 480: ("{x}, {y}: -{x}D{y}", [4, 4], [])` | 156.99 µs | 216.74 µs | +38.1% |
| parallel histograms | `case 700: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [6, 2], [])` | 1.03 ms | 1.41 ms | +37.3% |
| nesting compile | `range/4` | 3.73 µs | 5.09 µs | +36.6% |
| nesting parse | `negation/64` | 3.41 µs | 4.65 µs | +36.3% |
| nesting compile | `exponent/16` | 7.71 µs | 10.48 µs | +36.0% |
| parallel histograms | `case 460: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 4], [])` | 166.53 µs | 225.24 µs | +35.3% |
| nesting parse | `negation/4` | 371.9 ns | 498.0 ns | +33.9% |
| parallel histograms | `case 290: ("3D6 drop highest 0", [], [])` | 170.03 µs | 224.73 µs | +32.2% |
| parse | `case 163: {x}D1` | 582.0 ns | 763.4 ns | +31.2% |
| parse | `case 161: {x}D0` | 584.0 ns | 765.5 ns | +31.1% |
| parse | `case 27: {x}D{y}` | 684.1 ns | 892.2 ns | +30.4% |
| parallel histograms | `case 600: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 12, 1], [])` | 1.26 ms | 1.64 ms | +29.4% |
| parallel histograms | `case 530: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 12, 1], [])` | 1.28 ms | 1.65 ms | +28.8% |
| nesting compile | `range/16` | 12.66 µs | 16.19 µs | +27.8% |
| optimizations | `case 259: {x}: {x}D[5, 5, 5, 5, 5] drop highest 5` | 6.81 µs | 8.64 µs | +26.9% |
| optimizations | `case 214: {x}: ----{x}` | 3.23 µs | 4.06 µs | +26.0% |
| nesting parse | `negation/256` | 14.15 µs | 17.70 µs | +25.1% |
| optimizations | `case 252: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 5` | 6.87 µs | 8.58 µs | +25.0% |
| parallel histograms | `case 720: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [3, 2], [])` | 153.24 µs | 188.95 µs | +23.3% |
| parse | `case 162: {x}D(-1)` | 904.9 ns | 1.11 µs | +22.9% |
| parse | `case 221: {x}, {y}: {x}D{y}` | 837.4 ns | 1.01 µs | +20.9% |
| parallel histograms | `case 520: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 8, 1], [])` | 116.39 µs | 140.69 µs | +20.9% |
| nesting compile | `exponent/64` | 29.61 µs | 35.73 µs | +20.7% |
| parse | `case 155: 6D0 drop highest 3` | 596.4 ns | 719.6 ns | +20.7% |
| parse | `case 240: {x}, {y}: {x}D1 drop highest {y}` | 951.4 ns | 1.15 µs | +20.5% |
| parse | `case 231: {x}, {y}: {x}D1 drop lowest {y}` | 947.9 ns | 1.14 µs | +20.3% |
| parse | `case 234: {x}, {y}: 1D{x} drop lowest {y}` | 944.7 ns | 1.14 µs | +20.3% |
| parse | `case 243: {x}, {y}: 1D{x} drop highest {y}` | 948.8 ns | 1.14 µs | +20.2% |
| parse | `case 144: 3D6 drop lowest 1` | 593.4 ns | 713.4 ns | +20.2% |
| parse | `case 150: 3D6 drop highest 1` | 596.2 ns | 716.8 ns | +20.2% |
| parse | `case 235: {x}, {y}: 3D{x} drop lowest {y}` | 945.8 ns | 1.14 µs | +20.2% |
| parse | `case 242: {x}: {x}D1 drop highest 1` | 771.4 ns | 926.8 ns | +20.2% |
| parse | `case 156: 8D1 drop highest 5` | 595.7 ns | 715.5 ns | +20.1% |
| parse | `case 218: {x}: {x}D1` | 660.1 ns | 792.8 ns | +20.1% |
| parse | `case 230: {x}: 10D1 drop lowest {x}` | 765.9 ns | 919.7 ns | +20.1% |
| parse | `case 151: 3D6 drop highest 3` | 595.4 ns | 714.7 ns | +20.0% |
| parse | `case 244: {x}, {y}: 3D{x} drop highest {y}` | 950.8 ns | 1.14 µs | +20.0% |
| parse | `case 149: 8D1 drop lowest 5` | 592.4 ns | 710.9 ns | +20.0% |
| parse | `case 233: {x}: {x}D1 drop lowest 1` | 768.6 ns | 922.1 ns | +20.0% |
| parse | `case 152: 3D6 drop highest 5` | 595.7 ns | 714.3 ns | +19.9% |
| parse | `case 148: 6D0 drop lowest 3` | 592.0 ns | 709.5 ns | +19.9% |
| parse | `case 147: 0D6 drop lowest 3` | 592.6 ns | 709.9 ns | +19.8% |
| parse | `case 154: 0D6 drop highest 3` | 596.6 ns | 714.6 ns | +19.8% |
| parse | `case 239: {x}: 10D1 drop highest {x}` | 770.5 ns | 922.0 ns | +19.7% |
| parse | `case 232: {x}: 1D{x} drop lowest 1` | 767.2 ns | 917.2 ns | +19.5% |
| parse | `case 146: 3D6 drop lowest 5` | 593.8 ns | 708.4 ns | +19.3% |
| parse | `case 145: 3D6 drop lowest 3` | 594.6 ns | 709.2 ns | +19.3% |
| parse | `case 249: {x}: 10D1 drop lowest 1 drop highest {x}` | 878.5 ns | 1.05 µs | +19.3% |
| parse | `case 245: 3D6 drop highest 0` | 597.1 ns | 711.5 ns | +19.1% |
| parse | `case 241: {x}: 1D{x} drop highest 1` | 773.1 ns | 921.0 ns | +19.1% |
| parse | `case 220: {x}: 1D{x}` | 658.5 ns | 784.2 ns | +19.1% |
| parse | `case 236: 3D6 drop lowest 0` | 593.5 ns | 706.6 ns | +19.1% |
| parse | `case 159: 8D1 drop highest 3 drop lowest 3` | 705.2 ns | 839.0 ns | +19.0% |
| parse | `case 158: 8D1 drop lowest 3 drop highest 3` | 705.8 ns | 838.8 ns | +18.8% |
| parse | `case 273: {x}, {y}: 20D10 drop highest {x} drop lowest {y} drop highest {x} drop lowest {y}` | 1.46 µs | 1.74 µs | +18.8% |
| parse | `case 274: {x}, {y}: 20D10 drop lowest {x} drop highest {y} drop lowest {x} drop highest {y}` | 1.46 µs | 1.73 µs | +18.8% |
| parse | `case 248: {x}: 10D1 drop lowest {x} drop highest 1` | 877.7 ns | 1.04 µs | +18.4% |
| parse | `case 36: 3D10 drop highest 2` | 596.4 ns | 705.6 ns | +18.3% |
| parse | `case 33: 3D10 drop lowest 2` | 594.4 ns | 702.8 ns | +18.2% |
| parse | `case 38: 10D10 drop highest 2 drop lowest 5` | 706.4 ns | 832.5 ns | +17.8% |
| parse | `case 141: 0D100` | 485.4 ns | 571.9 ns | +17.8% |
| parse | `case 272: {x}, {y}: 20D10 drop highest {x} drop highest {y} drop lowest {x} drop lowest {y}` | 1.48 µs | 1.74 µs | +17.7% |
| parse | `case 60: 3D6 + 2D12` | 883.4 ns | 1.04 µs | +17.5% |
| parse | `case 143: 100D1` | 486.1 ns | 570.8 ns | +17.4% |
| parse | `case 237: 3D6 drop lowest 1 drop lowest 1` | 707.4 ns | 830.4 ns | +17.4% |
| parse | `case 25: 2d8` | 488.2 ns | 572.9 ns | +17.4% |
| parse | `case 222: 10D1` | 486.5 ns | 570.7 ns | +17.3% |
| parallel histograms | `case 350: ("{x}, {y}: {x}D{y} + {x}D{y}", [1, 12], [])` | 180.55 µs | 211.79 µs | +17.3% |
| parse | `case 278: 3D6 - 3D6` | 878.1 ns | 1.03 µs | +17.3% |
| parse | `case 140: 3D6` | 486.1 ns | 569.9 ns | +17.2% |
| parallel histograms | `case 490: ("{x}, {y}: -{x}D{y}", [2, 10], [])` | 144.93 µs | 169.89 µs | +17.2% |
| parse | `case 164: 0D{x}` | 581.5 ns | 681.5 ns | +17.2% |
| parse | `case 282: 3D6 drop highest 1 + 3D6 drop highest 1` | 1.10 µs | 1.29 µs | +17.1% |
| parse | `case 271: {x}, {y}: 20D10 drop lowest {x} drop lowest {y} drop highest {x} drop highest {y}` | 1.48 µs | 1.73 µs | +17.1% |
| parse | `case 219: 1D6` | 484.8 ns | 567.8 ns | +17.1% |
| parse | `case 281: 3D6 drop lowest 1 + 3D6 drop lowest 1` | 1.10 µs | 1.29 µs | +17.1% |
| parallel histograms | `case 590: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 8, 1], [])` | 120.58 µs | 141.07 µs | +17.0% |
| parse | `case 277: 3D6 + 3D6` | 880.4 ns | 1.03 µs | +17.0% |
| parse | `case 142: 100D0` | 487.4 ns | 569.3 ns | +16.8% |
| parallel histograms | `case 420: ("{x}, {y}: {x}D{y} / {x}D{y}", [1, 10], [])` | 157.27 µs | 183.68 µs | +16.8% |
| nesting compile | `range/64` | 50.20 µs | 58.58 µs | +16.7% |
| parse | `case 26: 20d10` | 488.9 ns | 570.4 ns | +16.7% |
| parse | `case 214: {x}: ----{x}` | 882.0 ns | 1.02 µs | +16.1% |
| parse | `case 16: {an external variable}` | 640.8 ns | 743.9 ns | +16.1% |
| parse | `case 246: 3D6 drop highest 1 drop highest 1` | 722.8 ns | 837.4 ns | +15.9% |
| parse | `case 213: {x}: ---{x}` | 837.1 ns | 954.5 ns | +14.0% |
| nesting compile | `exponent/256` | 118.78 µs | 134.66 µs | +13.4% |
| parallel histograms | `case 370: ("{x}, {y}: {x}D{y} - {x}D{y}", [1, 8], [])` | 132.05 µs | 149.59 µs | +13.3% |
| parallel histograms | `case 50: ("{x}: {x} * 0", [2147483647], [])` | 60.28 µs | 67.72 µs | +12.3% |
| parse | `case 212: {x}: --{x}` | 779.9 ns | 875.7 ns | +12.3% |
| parse | `case 247: 3D6 drop highest 1 drop highest (-1)` | 1.03 µs | 1.15 µs | +12.1% |
| parse | `case 13: {snake_case}` | 616.6 ns | 688.8 ns | +11.7% |
| parse | `case 12: {kebab-case}` | 616.9 ns | 689.1 ns | +11.7% |
| parse | `case 165: (-1)D{x}` | 921.3 ns | 1.03 µs | +11.7% |
| parse | `case 14: {qualified.access}` | 637.4 ns | 710.9 ns | +11.5% |
| parallel histograms | `case 210: ("{x}, {y}: [{x}:{y}] / [{x}:{y}]", [1, 1], [])` | 61.80 µs | 68.78 µs | +11.3% |
| parse | `case 238: 3D6 drop lowest 1 drop lowest (-1)` | 1.02 µs | 1.14 µs | +11.1% |
| parse | `case 314: 1D(1D3)` | 1.09 µs | 1.21 µs | +11.0% |
| parallel histograms | `case 60: ("{x}: {x} * -1", [2147483647], [])` | 60.35 µs | 66.58 µs | +10.3% |

The regressions have four causes:

- **Compilation now optimizes.** The nesting compile group times compile(),
  which ran no optimizer passes before 0.13.0 and runs all of them now, so its
  small cases pay for optimization that they skipped before. It grows linearly
  with depth: the passes that grew quadratically with the length of a chain were
  made linear before release, and test_compile_is_linear holds them to it.
- **Exact reduction of drops.** Rolls of one face with drops, e.g., `{x}D1 drop
  lowest {y}`, take longer to optimize, because the optimizer now clamps their
  counts with Max, which a negative count requires to stay exact.
- **The iterative parser's constant factor.** Small dice expressions and drop
  clauses parse about 0.1 to 0.2 µs slower, and negation chains about a third
  slower, in exchange for parsing any depth in linear time; the parse group as a
  whole is a third faster.
- **The iterative parallel histogram driver.** Mid-sized parallel histograms, of
  a few hundred microseconds, run slower: the crew of workers that replaced the
  recursive split recruits helpers gradually, which costs most where a build is
  too large for one thread and too small to amortize the recruitment (an earlier
  measurement found a geometric mean ratio of 1.06). Unmetered builds charge the
  histogram meter nothing, and the interval at which workers recruit helpers
  makes no difference.

The last two are accepted costs of 0.13.0. The parallel driver was removed in
0.14.0, and the parser's constant factor on small expressions remains to be
profiled.

## New in 0.13.0: failing diagnose

0.12.0 diagnosed a failing source by parsing it again after every fix, in time
quadratic in the number of errors (2,000 unclosed groups took about 2 s), so this
group has no before. After the epic, diagnosis is linear:

| Case | Time |
|---|---:|
| `unclosed groups/10` | 14.37 µs |
| `unclosed groups/100` | 121.87 µs |
| `unclosed groups/1000` | 1.14 ms |
| `unclosed groups/10000` | 13.54 ms |
| `unclosed groups/100000` | 142.12 ms |
| `bare identifiers/10` | 30.91 µs |
| `bare identifiers/100` | 289.39 µs |
| `bare identifiers/1000` | 2.92 ms |
| `bare identifiers/10000` | 29.17 ms |
| `bare identifiers/100000` | 304.19 ms |
| `leading closers/10` | 26.81 µs |
| `leading closers/100` | 251.03 µs |
| `leading closers/1000` | 2.52 ms |
| `leading closers/10000` | 25.19 ms |
| `leading closers/100000` | 257.38 ms |
| `missing operands/10` | 30.26 µs |
| `missing operands/100` | 282.71 µs |
| `missing operands/1000` | 2.81 ms |
| `missing operands/10000` | 28.38 ms |
| `missing operands/100000` | 291.58 ms |

## Full results

### optimizations

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `case 0: 0` | 1.04 µs | 935.6 ns | -10.4% |
| `case 100: {x}: 1 / 2 / 3 / 4 / {x} / 5` | 13.48 µs | 5.00 µs | -62.9% |
| `case 101: {x}: 1 / 2 / 3 / 4 / 5 / {x}` | 13.43 µs | 4.95 µs | -63.1% |
| `case 102: 0 + 1` | 2.36 µs | 1.84 µs | -22.1% |
| `case 103: 2147483647 + 1` | 2.39 µs | 1.86 µs | -22.2% |
| `case 104: -2147483648 + -1` | 1.66 µs | 1.43 µs | -14.0% |
| `case 105: 0 - 1` | 2.27 µs | 1.84 µs | -19.1% |
| `case 106: 2147483647 - -1` | 1.91 µs | 1.64 µs | -14.0% |
| `case 107: -2147483648 - 1` | 1.90 µs | 1.66 µs | -12.8% |
| `case 108: 100 * 100` | 2.32 µs | 1.82 µs | -21.5% |
| `case 109: 32768 * 65536` | 2.33 µs | 1.82 µs | -22.2% |
| `case 10: {alloy}` | 1.34 µs | 1.33 µs | -0.8% |
| `case 110: 32768 * 65537` | 2.34 µs | 1.82 µs | -22.2% |
| `case 111: 200 / 100` | 2.21 µs | 1.97 µs | -10.9% |
| `case 112: 100 / 0` | 2.21 µs | 1.83 µs | -17.3% |
| `case 113: -2147483648 / -1` | 1.48 µs | 1.42 µs | -4.2% |
| `case 114: 200 % 100` | 2.08 µs | 1.83 µs | -12.0% |
| `case 115: 100 % 0` | 2.08 µs | 1.82 µs | -12.6% |
| `case 116: 0 ^ 0` | 2.05 µs | 1.82 µs | -11.0% |
| `case 117: 0 ^ 1` | 2.06 µs | 1.83 µs | -11.5% |
| `case 118: 0 ^ -1` | 1.71 µs | 1.58 µs | -7.7% |
| `case 119: 1 ^ 0` | 2.06 µs | 1.83 µs | -11.5% |
| `case 11: {a0}` | 1.33 µs | 1.32 µs | -1.0% |
| `case 120: 1 ^ 1` | 2.04 µs | 1.83 µs | -10.2% |
| `case 121: 1 ^ -1` | 1.71 µs | 1.58 µs | -7.3% |
| `case 122: 1 ^ -2` | 1.70 µs | 1.58 µs | -7.1% |
| `case 123: (-1) ^ 0` | 2.38 µs | 2.15 µs | -9.8% |
| `case 124: (-1) ^ 1` | 2.38 µs | 2.16 µs | -9.6% |
| `case 125: (-1) ^ 2` | 2.40 µs | 2.15 µs | -10.2% |
| `case 126: (-1) ^ -1` | 2.02 µs | 1.95 µs | -3.6% |
| `case 127: (-1) ^ -2` | 2.03 µs | 1.94 µs | -4.2% |
| `case 128: 10 ^ 0` | 2.07 µs | 1.82 µs | -11.8% |
| `case 129: 10 ^ 3` | 2.08 µs | 1.82 µs | -12.5% |
| `case 12: {kebab-case}` | 1.37 µs | 1.35 µs | -1.4% |
| `case 130: 10 ^ 10000` | 2.07 µs | 1.82 µs | -12.2% |
| `case 131: 10 ^ -1` | 1.72 µs | 1.59 µs | -7.7% |
| `case 132: -0` | 669.9 ns | 695.5 ns | +3.8% |
| `case 133: -(-2147483648)` | 2.02 µs | 1.91 µs | -5.7% |
| `case 134: 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10` | 13.33 µs | 7.27 µs | -45.5% |
| `case 135: {x}: 1 + 4 + {x} + 4 + 9` | 10.25 µs | 7.16 µs | -30.2% |
| `case 136: 1 + 4 + {x} + 4 + 9` | 10.10 µs | 6.95 µs | -31.1% |
| `case 137: [5:10]` | 2.90 µs | 2.49 µs | -14.3% |
| `case 138: [7:7]` | 2.43 µs | 2.12 µs | -12.6% |
| `case 139: [5:4]` | 2.50 µs | 2.15 µs | -14.1% |
| `case 13: {snake_case}` | 1.35 µs | 1.35 µs | -0.4% |
| `case 140: 3D6` | 2.02 µs | 1.94 µs | -3.6% |
| `case 141: 0D100` | 1.64 µs | 1.61 µs | -1.9% |
| `case 142: 100D0` | 1.64 µs | 1.61 µs | -1.9% |
| `case 143: 100D1` | 1.65 µs | 1.62 µs | -1.6% |
| `case 144: 3D6 drop lowest 1` | 2.50 µs | 2.56 µs | +2.1% |
| `case 145: 3D6 drop lowest 3` | 1.88 µs | 2.01 µs | +6.6% |
| `case 146: 3D6 drop lowest 5` | 1.90 µs | 1.99 µs | +4.2% |
| `case 147: 0D6 drop lowest 3` | 1.89 µs | 1.97 µs | +4.4% |
| `case 148: 6D0 drop lowest 3` | 1.89 µs | 1.96 µs | +4.1% |
| `case 149: 8D1 drop lowest 5` | 1.89 µs | 2.00 µs | +5.5% |
| `case 14: {qualified.access}` | 1.36 µs | 1.37 µs | +1.2% |
| `case 150: 3D6 drop highest 1` | 2.48 µs | 2.56 µs | +3.0% |
| `case 151: 3D6 drop highest 3` | 1.88 µs | 2.02 µs | +7.4% |
| `case 152: 3D6 drop highest 5` | 1.90 µs | 2.00 µs | +5.1% |
| `case 153: 10D10 drop lowest (1 + 3)` | 5.58 µs | 4.75 µs | -14.9% |
| `case 154: 0D6 drop highest 3` | 1.89 µs | 1.98 µs | +4.8% |
| `case 155: 6D0 drop highest 3` | 1.89 µs | 1.97 µs | +4.3% |
| `case 156: 8D1 drop highest 5` | 1.90 µs | 2.00 µs | +5.4% |
| `case 157: 10D10 drop highest (1 + 3)` | 5.59 µs | 4.75 µs | -15.0% |
| `case 158: 8D1 drop lowest 3 drop highest 3` | 2.17 µs | 2.30 µs | +6.0% |
| `case 159: 8D1 drop highest 3 drop lowest 3` | 2.17 µs | 2.30 µs | +5.9% |
| `case 15: {dés}` | 1.35 µs | 1.33 µs | -1.1% |
| `case 160: {x}D(3 + 3)` | 5.27 µs | 4.40 µs | -16.5% |
| `case 161: {x}D0` | 1.99 µs | 2.03 µs | +2.2% |
| `case 162: {x}D(-1)` | 2.34 µs | 2.38 µs | +2.1% |
| `case 163: {x}D1` | 2.05 µs | 3.35 µs | +62.8% |
| `case 164: 0D{x}` | 1.99 µs | 1.94 µs | -2.5% |
| `case 165: (-1)D{x}` | 2.35 µs | 2.32 µs | -1.4% |
| `case 166: 1D[0, 0, 0, 1, 1, 2]` | 2.72 µs | 2.64 µs | -2.8% |
| `case 167: 0D[0, 0, 0, 1, 1, 2]` | 2.32 µs | 2.27 µs | -2.5% |
| `case 168: 5D[1]` | 2.02 µs | 1.98 µs | -1.9% |
| `case 169: 8D[3, 3, 3]` | 2.16 µs | 2.12 µs | -1.8% |
| `case 16: {an external variable}` | 1.38 µs | 1.40 µs | +1.6% |
| `case 170: 5D[1, 2, 3] drop lowest 5` | 2.34 µs | 2.35 µs | +0.6% |
| `case 171: 5D[1, 2, 3] drop lowest 6` | 2.38 µs | 2.35 µs | -1.4% |
| `case 172: 5D[1, 1, 1] drop lowest 3` | 2.43 µs | 2.38 µs | -2.0% |
| `case 173: 5D[1, 2, 3] drop highest 5` | 2.34 µs | 2.36 µs | +0.9% |
| `case 174: 5D[1, 2, 3] drop highest 6` | 2.40 µs | 2.36 µs | -1.8% |
| `case 175: 5D[1, 1, 1] drop highest 3` | 2.43 µs | 2.40 µs | -1.5% |
| `case 176: 5D[5, 5, 5, 5] drop highest 3 drop lowest 1` | 2.79 µs | 2.86 µs | +2.5% |
| `case 177: 5D[5, 5, 5, 5] drop lowest 3 drop highest 1` | 2.78 µs | 2.84 µs | +1.9% |
| `case 178: 3D1 + 7` | 2.72 µs | 2.26 µs | -17.1% |
| `case 179: 7 + 3D1` | 2.75 µs | 2.16 µs | -21.5% |
| `case 17: {выражение в кости}` | 1.74 µs | 1.41 µs | -18.6% |
| `case 180: {x}: 0 + {x}` | 2.91 µs | 2.39 µs | -17.9% |
| `case 181: {x}: {x} + 0` | 2.90 µs | 2.40 µs | -17.4% |
| `case 182: {x}: {x} + 1` | 3.96 µs | 2.99 µs | -24.3% |
| `case 183: {x}: 0 - {x}` | 3.46 µs | 2.97 µs | -14.0% |
| `case 184: {x}: {x} - 0` | 2.88 µs | 2.40 µs | -16.6% |
| `case 185: {x}: {x} - {x}` | 2.80 µs | 2.53 µs | -9.8% |
| `case 186: {x}: {x} - 1` | 2.78 µs | 2.41 µs | -13.2% |
| `case 187: {x}: 0 * {x}` | 2.85 µs | 2.35 µs | -17.7% |
| `case 188: {x}: {x} * 0` | 2.84 µs | 2.35 µs | -17.3% |
| `case 189: {x}: -1 * {x}` | 3.14 µs | 2.72 µs | -13.3% |
| `case 18: {έκφραση ζαριών}` | 1.66 µs | 1.39 µs | -16.2% |
| `case 190: {x}: {x} * -1` | 3.14 µs | 2.72 µs | -13.4% |
| `case 191: {x}: {x} * 2` | 3.86 µs | 3.01 µs | -22.0% |
| `case 192: {x}: 2 * {x}` | 3.86 µs | 2.99 µs | -22.4% |
| `case 193: {x}: {x} * 3` | 3.84 µs | 3.00 µs | -21.8% |
| `case 194: {x}: 0 / {x}` | 2.72 µs | 2.36 µs | -13.5% |
| `case 195: {x}: {x} / 0` | 2.72 µs | 2.38 µs | -12.4% |
| `case 196: {x}: {x} / 1` | 2.74 µs | 2.42 µs | -11.5% |
| `case 197: {x}: {x} / -1` | 3.02 µs | 2.78 µs | -8.1% |
| `case 198: {x}: {x} / {x}` | 2.75 µs | 2.51 µs | -8.8% |
| `case 199: {x}: {x} / 2` | 2.73 µs | 2.42 µs | -11.3% |
| `case 19: {a}: {a}` | 1.48 µs | 1.40 µs | -4.9% |
| `case 1: 1` | 1.04 µs | 920.2 ns | -11.5% |
| `case 200: {x}: 0 % {x}` | 2.67 µs | 2.36 µs | -11.5% |
| `case 201: {x}: {x} % 0` | 2.59 µs | 2.35 µs | -9.0% |
| `case 202: {x}: {x} % 1` | 2.58 µs | 2.36 µs | -8.6% |
| `case 203: {x}: {x} % 2` | 2.60 µs | 2.39 µs | -7.9% |
| `case 204: {x}: 0 ^ {x}` | 2.60 µs | 2.41 µs | -7.5% |
| `case 205: {x}: 1 ^ {x}` | 2.59 µs | 2.42 µs | -6.6% |
| `case 206: {x}: {x} ^ 0` | 2.58 µs | 2.36 µs | -8.4% |
| `case 207: {x}: {x} ^ 1` | 2.60 µs | 2.37 µs | -8.7% |
| `case 208: {x}: {x} ^ 2` | 3.60 µs | 3.01 µs | -16.3% |
| `case 209: {x}: {x} ^ 3` | 2.58 µs | 2.39 µs | -7.3% |
| `case 20: {a}, {an external variable}: {an external variable}` | 1.82 µs | 1.74 µs | -4.6% |
| `case 210: {x}: {x} ^ -1` | 2.22 µs | 2.15 µs | -2.9% |
| `case 211: {x}: -{x}` | 2.03 µs | 2.01 µs | -0.9% |
| `case 212: {x}: --{x}` | 2.31 µs | 2.49 µs | +7.7% |
| `case 213: {x}: ---{x}` | 3.27 µs | 3.04 µs | -7.0% |
| `case 214: {x}: ----{x}` | 3.23 µs | 4.06 µs | +26.0% |
| `case 215: {x}: --{x} + 2` | 4.50 µs | 4.69 µs | +4.2% |
| `case 216: {x}: [{x}:{x}]` | 3.00 µs | 2.87 µs | -4.2% |
| `case 217: {x}, {y}: [{x}:{y}]` | 3.63 µs | 3.39 µs | -6.8% |
| `case 218: {x}: {x}D1` | 2.23 µs | 3.44 µs | +54.6% |
| `case 219: 1D6` | 3.06 µs | 2.78 µs | -9.1% |
| `case 21: ({a})` | 2.40 µs | 1.90 µs | -20.9% |
| `case 220: {x}: 1D{x}` | 3.63 µs | 3.33 µs | -8.4% |
| `case 221: {x}, {y}: {x}D{y}` | 2.92 µs | 2.86 µs | -2.2% |
| `case 222: 10D1` | 1.66 µs | 1.61 µs | -3.0% |
| `case 223: {x}: {x}D[1]` | 2.65 µs | 3.86 µs | +45.4% |
| `case 224: {x}: {x}D[1, 1, 1]` | 2.79 µs | 3.98 µs | +42.8% |
| `case 225: {x}: {x}D[5, 5, 5, 5, 5]` | 5.00 µs | 5.17 µs | +3.4% |
| `case 226: {x}: 1D[3, 4, 5, 6]` | 4.11 µs | 3.72 µs | -9.4% |
| `case 227: {x}: {x}D[1, 2, 3]` | 4.19 µs | 3.84 µs | -8.4% |
| `case 228: {x}: {x}D[3, 4, 5, 6, 7, 8]` | 3.18 µs | 3.10 µs | -2.6% |
| `case 229: {x}: {x}D[1, 1, 2, 3, 5, 8]` | 3.17 µs | 3.10 µs | -2.2% |
| `case 22: [0:5]` | 2.92 µs | 2.50 µs | -14.4% |
| `case 230: {x}: 10D1 drop lowest {x}` | 3.44 µs | 5.79 µs | +68.6% |
| `case 231: {x}, {y}: {x}D1 drop lowest {y}` | 3.89 µs | 8.31 µs | +113.5% |
| `case 232: {x}: 1D{x} drop lowest 1` | 4.16 µs | 4.16 µs | -0.1% |
| `case 233: {x}: {x}D1 drop lowest 1` | 3.36 µs | 5.88 µs | +75.0% |
| `case 234: {x}, {y}: 1D{x} drop lowest {y}` | 4.77 µs | 4.65 µs | -2.5% |
| `case 235: {x}, {y}: 3D{x} drop lowest {y}` | 3.38 µs | 3.46 µs | +2.5% |
| `case 236: 3D6 drop lowest 0` | 3.35 µs | 3.26 µs | -2.6% |
| `case 237: 3D6 drop lowest 1 drop lowest 1` | 6.35 µs | 4.04 µs | -36.5% |
| `case 238: 3D6 drop lowest 1 drop lowest (-1)` | 6.45 µs | 4.41 µs | -31.6% |
| `case 239: {x}: 10D1 drop highest {x}` | 3.54 µs | 5.84 µs | +65.2% |
| `case 23: [-10:10]` | 2.59 µs | 2.31 µs | -10.9% |
| `case 240: {x}, {y}: {x}D1 drop highest {y}` | 3.94 µs | 8.35 µs | +112.0% |
| `case 241: {x}: 1D{x} drop highest 1` | 4.24 µs | 4.14 µs | -2.4% |
| `case 242: {x}: {x}D1 drop highest 1` | 3.43 µs | 5.87 µs | +71.5% |
| `case 243: {x}, {y}: 1D{x} drop highest {y}` | 4.83 µs | 4.64 µs | -4.0% |
| `case 244: {x}, {y}: 3D{x} drop highest {y}` | 3.41 µs | 3.45 µs | +1.4% |
| `case 245: 3D6 drop highest 0` | 3.39 µs | 3.23 µs | -4.7% |
| `case 246: 3D6 drop highest 1 drop highest 1` | 6.36 µs | 4.05 µs | -36.3% |
| `case 247: 3D6 drop highest 1 drop highest (-1)` | 6.47 µs | 4.46 µs | -31.1% |
| `case 248: {x}: 10D1 drop lowest {x} drop highest 1` | 6.62 µs | 9.58 µs | +44.7% |
| `case 249: {x}: 10D1 drop lowest 1 drop highest {x}` | 6.59 µs | 6.22 µs | -5.6% |
| `case 24: [1d3 + 5 : 55 - 8D4]` | 14.15 µs | 10.93 µs | -22.8% |
| `case 250: {x}: {x}D[1] drop lowest 3` | 3.95 µs | 6.09 µs | +54.1% |
| `case 251: {x}: {x}D[1, 1, 1] drop lowest 3` | 4.09 µs | 6.24 µs | +52.5% |
| `case 252: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 5` | 6.87 µs | 8.58 µs | +25.0% |
| `case 253: {x}: 1D[3, 4, 5, 6] drop lowest {x}` | 5.01 µs | 4.75 µs | -5.1% |
| `case 254: {x}, {y}: {x}D[1, 2, 3] drop lowest {y}` | 5.41 µs | 5.12 µs | -5.3% |
| `case 255: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop lowest {y}` | 4.17 µs | 4.14 µs | -0.9% |
| `case 256: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop lowest {y}` | 4.15 µs | 4.13 µs | -0.5% |
| `case 257: {x}: {x}D[1] drop highest 3` | 3.90 µs | 6.18 µs | +58.7% |
| `case 258: {x}: {x}D[1, 1, 1] drop highest 3` | 4.05 µs | 6.34 µs | +56.6% |
| `case 259: {x}: {x}D[5, 5, 5, 5, 5] drop highest 5` | 6.81 µs | 8.64 µs | +26.9% |
| `case 25: 2d8` | 2.02 µs | 1.96 µs | -2.8% |
| `case 260: {x}: 1D[3, 4, 5, 6] drop highest {x}` | 5.00 µs | 4.88 µs | -2.4% |
| `case 261: {x}, {y}: {x}D[1, 2, 3] drop highest {y}` | 5.40 µs | 5.22 µs | -3.3% |
| `case 262: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop highest {y}` | 4.20 µs | 4.18 µs | -0.5% |
| `case 263: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop highest {y}` | 4.21 µs | 4.15 µs | -1.4% |
| `case 264: {x}: {x}D[1] drop highest 1 drop lowest 1` | 6.03 µs | 10.17 µs | +68.7% |
| `case 265: {x}: {x}D[1, 1, 1] drop highest 1 drop lowest 1` | 6.17 µs | 10.29 µs | +66.9% |
| `case 266: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 1 drop highest 1` | 8.13 µs | 11.25 µs | +38.3% |
| `case 267: {x}: 1D[3, 4, 5, 6] drop highest {x} drop lowest {x}` | 6.09 µs | 6.03 µs | -1.1% |
| `case 268: {x}, {y}: {x}D[1, 2, 3] drop highest {y} drop lowest {y}` | 6.47 µs | 6.38 µs | -1.4% |
| `case 269: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop highest {y} drop lowest {y}` | 4.92 µs | 5.04 µs | +2.6% |
| `case 26: 20d10` | 2.05 µs | 1.96 µs | -4.5% |
| `case 270: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop lowest {y} drop highest {y}` | 4.92 µs | 5.05 µs | +2.6% |
| `case 275: [2:8] + [2:8]` | 6.39 µs | 5.13 µs | -19.7% |
| `case 276: [2:8] - [2:8]` | 6.21 µs | 5.14 µs | -17.3% |
| `case 277: 3D6 + 3D6` | 4.68 µs | 4.13 µs | -11.7% |
| `case 278: 3D6 - 3D6` | 4.50 µs | 4.12 µs | -8.5% |
| `case 279: 2D[-1, 0, 1] + 2D[-1, 0, 1]` | 6.01 µs | 5.19 µs | -13.6% |
| `case 27: {x}D{y}` | 2.67 µs | 2.62 µs | -2.1% |
| `case 280: 2D[-1, 0, 1] - 2D[-1, 0, 1]` | 5.85 µs | 5.21 µs | -11.1% |
| `case 281: 3D6 drop lowest 1 + 3D6 drop lowest 1` | 5.90 µs | 5.31 µs | -10.0% |
| `case 282: 3D6 drop highest 1 + 3D6 drop highest 1` | 5.92 µs | 5.32 µs | -10.3% |
| `case 283: {x}: 9D6 drop lowest ({x} + 1) + 9D6 drop lowest ({x} + 1)` | 15.00 µs | 12.05 µs | -19.7% |
| `case 284: (1 + 1) + (1 + 1)` | 7.03 µs | 4.17 µs | -40.8% |
| `case 285: (2 + 1) + (1 + 2)` | 7.37 µs | 4.10 µs | -44.4% |
| `case 286: {x}: (1 + {x}) - (1 + {x})` | 7.97 µs | 5.11 µs | -35.8% |
| `case 287: {x}: ({x} - 1) + ({x} - 1)` | 9.77 µs | 6.20 µs | -36.5% |
| `case 288: {x}: (1 - {x}) + (1 - {x})` | 9.80 µs | 6.17 µs | -37.0% |
| `case 289: {x}: ({x} / 3) + ({x} / 3)` | 9.58 µs | 6.16 µs | -35.7% |
| `case 28: 1D[-1, 0, 1]` | 3.69 µs | 3.33 µs | -9.6% |
| `case 290: {x}: (3 / {x}) + (3 / {x})` | 9.63 µs | 6.15 µs | -36.1% |
| `case 291: {x}: ({x} % 3) + ({x} % 3)` | 9.30 µs | 6.10 µs | -34.4% |
| `case 292: {x}: (3 % {x}) + (3 % {x})` | 9.32 µs | 6.11 µs | -34.5% |
| `case 293: {x}: ({x} ^ 3) + ({x} ^ 3)` | 9.24 µs | 6.11 µs | -33.8% |
| `case 294: {x}: (3 ^ {x}) + (3 ^ {x})` | 9.25 µs | 6.12 µs | -33.8% |
| `case 295: {x}: -{x} + -{x}` | 4.83 µs | 4.05 µs | -16.1% |
| `case 296: {x}: ({x} + 1) + -({x} + 1)` | 10.88 µs | 7.10 µs | -34.8% |
| `case 298: {x}, {y}: (({x} + {y}) + ({x} + {y})) + (({x} + {y}) + ({x} + {y}))` | 29.64 µs | 12.53 µs | -57.7% |
| `case 299: {x}: {x}` | 1.48 µs | 1.41 µs | -4.2% |
| `case 29: (1d6)` | 4.11 µs | 3.33 µs | -18.9% |
| `case 2: -1` | 671.5 ns | 679.4 ns | +1.2% |
| `case 300: {x}: {x} + 1 + 1 + 1 + 1` | 10.05 µs | 6.05 µs | -39.8% |
| `case 301: {x}, {y}: {x} + {y}` | 3.27 µs | 2.82 µs | -13.6% |
| `case 302: {u}, {v}, {w}, {x}, {y}, {z}: {u} + {v} + {w} + {x} + {y} + {z}` | 17.54 µs | 11.65 µs | -33.5% |
| `case 303: {x}` | 1.33 µs | 1.33 µs | +0.4% |
| `case 304: {x} + 1` | 3.73 µs | 2.96 µs | -20.8% |
| `case 305: {x} + 1 + 1 + 1 + 1` | 9.86 µs | 6.01 µs | -39.0% |
| `case 306: {x} + {y}` | 3.01 µs | 2.62 µs | -12.9% |
| `case 307: {u} + {v} + {w} + {x} + {y} + {z}` | 16.74 µs | 10.89 µs | -34.9% |
| `case 308: [1 + 2 : 3 + 4]` | 6.27 µs | 4.63 µs | -26.2% |
| `case 309: (4 - 2)D(3 - 1)` | 6.43 µs | 4.99 µs | -22.3% |
| `case 30: (1)d(6)` | 4.55 µs | 3.85 µs | -15.4% |
| `case 310: (4 * 2)D[-1, 0, 1]` | 5.54 µs | 4.69 µs | -15.4% |
| `case 311: (8 * 2)D(24 / 3) drop lowest (10 % 7)` | 9.08 µs | 7.17 µs | -21.1% |
| `case 312: (3 ^ 3)D(24 / 3) drop highest (3 + -1)` | 8.75 µs | 7.06 µs | -19.3% |
| `case 313: (1D3)D3` | 6.12 µs | 5.54 µs | -9.5% |
| `case 314: 1D(1D3)` | 6.01 µs | 5.48 µs | -8.7% |
| `case 315: (1D3)D(1D3)` | 8.76 µs | 7.56 µs | -13.7% |
| `case 316: {x}@(3D6)` | 3.30 µs | 2.75 µs | -16.7% |
| `case 317: {x}@(3D6) + {x}` | 4.71 µs | 3.83 µs | -18.7% |
| `case 318: {x}@(3D6) + {x} + {x}` | 9.26 µs | 6.84 µs | -26.1% |
| `case 319: {a}@({b}@(3D6) + {b}) + {a} + {b}` | 14.44 µs | 9.21 µs | -36.2% |
| `case 31: (1d[1, 1, 2, 3, 5, 8])D(1D8)` | 10.05 µs | 8.69 µs | -13.6% |
| `case 320: {y}: {x}@(3D6) + {x} + {y}` | 9.75 µs | 7.23 µs | -25.9% |
| `case 321: {x}@(2+3)D6` | 5.13 µs | 4.17 µs | -18.7% |
| `case 322: {x}@(7)` | 2.33 µs | 1.76 µs | -24.2% |
| `case 323: {x}@(7) + {x}` | 3.77 µs | 2.78 µs | -26.1% |
| `case 324: {x}@(1D6) + {x}` | 6.49 µs | 4.92 µs | -24.1% |
| `case 325: {x}@([1:6]) + {x}` | 6.31 µs | 4.31 µs | -31.7% |
| `case 326: {x}@(2D4) * {x}` | 4.62 µs | 3.76 µs | -18.4% |
| `case 327: {hp}@(3D6) + {hp}` | 4.67 µs | 4.06 µs | -13.1% |
| `case 328: {score}@(1D20) + {score} + {score}` | 9.44 µs | 7.03 µs | -25.5% |
| `case 329: {x}@(1D6) + {y}@(1D4) + {x} * {y}` | 15.10 µs | 10.56 µs | -30.1% |
| `case 32: 3D10 drop lowest` | 2.77 µs | 2.63 µs | -5.2% |
| `case 330: {a}@({b}@(1D4) + {b}) + {a}` | 12.06 µs | 7.88 µs | -34.7% |
| `case 331: {x}@(2) ^ {x}` | 3.47 µs | 2.73 µs | -21.5% |
| `case 332: {n}@(1D3)D6 + {n}` | 9.26 µs | 7.38 µs | -20.3% |
| `case 333: 3D{f}@(6) + {f}` | 6.00 µs | 4.79 µs | -20.2% |
| `case 334: 4D6 drop lowest {k}@(1) + {k}` | 7.08 µs | 5.93 µs | -16.3% |
| `case 335: {y}: {x}@(1D6) + {x} + {y}` | 9.92 µs | 7.37 µs | -25.7% |
| `case 336: {x}@(2+3)` | 4.17 µs | 2.58 µs | -38.1% |
| `case 337: {x}@(3) + {y}@(4) + {x} + {y}` | 7.13 µs | 4.69 µs | -34.3% |
| `case 33: 3D10 drop lowest 2` | 2.48 µs | 2.56 µs | +3.3% |
| `case 34: 10D10 drop lowest (3 + 3)` | 5.56 µs | 4.75 µs | -14.6% |
| `case 35: 3D10 drop highest` | 2.73 µs | 2.62 µs | -4.0% |
| `case 36: 3D10 drop highest 2` | 2.46 µs | 2.59 µs | +5.0% |
| `case 37: 10D10 drop highest (3 + 3)` | 5.58 µs | 4.74 µs | -15.1% |
| `case 38: 10D10 drop highest 2 drop lowest 5` | 3.01 µs | 3.12 µs | +3.7% |
| `case 39: 10D10 drop lowest (1 + 3) drop highest {x}` | 6.97 µs | 6.04 µs | -13.3% |
| `case 3: 10` | 1.04 µs | 922.3 ns | -11.1% |
| `case 40: 0 + 0` | 2.36 µs | 1.84 µs | -22.1% |
| `case 41: 0 - 0` | 2.24 µs | 1.82 µs | -18.9% |
| `case 42: 0 * 0` | 2.31 µs | 1.79 µs | -22.7% |
| `case 43: {a} × {b}` | 2.94 µs | 2.55 µs | -13.5% |
| `case 44: 0 / 0` | 2.22 µs | 1.79 µs | -19.2% |
| `case 45: {a} ÷ {b}` | 2.83 µs | 2.54 µs | -10.3% |
| `case 46: {a} % {b}` | 2.72 µs | 2.54 µs | -6.8% |
| `case 47: -10` | 677.1 ns | 684.3 ns | +1.1% |
| `case 48: 0 + 1 + 2` | 3.41 µs | 2.47 µs | -27.4% |
| `case 49: (0 + 1) + 2` | 4.95 µs | 3.04 µs | -38.6% |
| `case 4: 100` | 1.04 µs | 922.5 ns | -11.0% |
| `case 50: 0 + (1 + 2)` | 4.95 µs | 3.03 µs | -38.7% |
| `case 51: 0 + 1 - 2` | 3.37 µs | 2.47 µs | -26.7% |
| `case 52: 0 + 1 * 2` | 3.39 µs | 2.34 µs | -31.0% |
| `case 53: 0 + 1 / 2` | 3.29 µs | 2.34 µs | -28.9% |
| `case 54: 0 + 1 % 2` | 3.18 µs | 2.37 µs | -25.2% |
| `case 55: 0 + 1 ^ 2 ^ {a}` | 4.46 µs | 3.60 µs | -19.1% |
| `case 56: 0 ^ {a} - {a} ^ 2` | 7.13 µs | 5.55 µs | -22.2% |
| `case 57: 0 ^ -{a} - -{a} ^ 2` | 10.18 µs | 8.00 µs | -21.5% |
| `case 58: 3D6 + 0` | 4.22 µs | 3.50 µs | -17.1% |
| `case 59: 5 + 3D6` | 3.43 µs | 2.84 µs | -17.0% |
| `case 5: 123456789123456789123456789123456789` | 1.06 µs | 937.7 ns | -11.9% |
| `case 60: 3D6 + 2D12` | 5.09 µs | 4.36 µs | -14.4% |
| `case 61: 3 ^ 3D6 + -2D12` | 6.79 µs | 5.78 µs | -15.0% |
| `case 62: (5 + {a})d({b})` | 5.50 µs | 4.69 µs | -14.8% |
| `case 63: (5 + {a})d({b}) drop lowest 1` | 6.24 µs | 5.35 µs | -14.2% |
| `case 64: (5 + {a})d({b}) drop highest 1 * 3` | 11.42 µs | 8.96 µs | -21.5% |
| `case 65: (5 + {a})d({b}) drop highest 1 % [2:3]` | 9.79 µs | 8.22 µs | -16.1% |
| `case 66: {x}: ({x} + 1) - ({x} + 1)` | 7.96 µs | 5.11 µs | -35.8% |
| `case 67: {x}: 1 + {x}` | 2.92 µs | 2.45 µs | -15.8% |
| `case 68: {x}: {x} + 1 + 1` | 6.03 µs | 4.13 µs | -31.5% |
| `case 69: {x}: 1 + 1 + {x} + 1 + 1` | 10.11 µs | 7.15 µs | -29.3% |
| `case 6: 987654321987654321987654321987654321` | 1.06 µs | 935.8 ns | -11.8% |
| `case 70: {x}: {x} + {x} + {x} + 1 + 1 + 1` | 16.35 µs | 8.24 µs | -49.6% |
| `case 71: {x}: {x} + 1 + {x} + 1 + {x} + 1` | 16.41 µs | 9.18 µs | -44.1% |
| `case 72: {x}: 1 + 1 + 1 + 1 - 1 + 1 + 1 + 1` | 9.48 µs | 5.99 µs | -36.8% |
| `case 73: {x}: ({x} + {x} + {x}) + (1 + 1 + 1)` | 20.80 µs | 9.62 µs | -53.7% |
| `case 74: {x}: ({x} + {x} + {x}) - (1 + 1 + 1)` | 15.01 µs | 8.41 µs | -43.9% |
| `case 75: {x}: ({x} + 1 + {x}) - (1 + {x} + 1)` | 20.52 µs | 10.40 µs | -49.3% |
| `case 76: {x}: ({x} + ({x} + ({x} + (1 + (1 + 1)))))` | 77.34 µs | 11.46 µs | -85.2% |
| `case 77: {x}: {x} * 1` | 2.84 µs | 2.40 µs | -15.5% |
| `case 78: {x}: 1 * {x}` | 2.84 µs | 2.39 µs | -15.7% |
| `case 79: {x}: {x} * 1 * 1` | 3.82 µs | 3.06 µs | -19.9% |
| `case 7: 09` | 1.03 µs | 918.3 ns | -11.2% |
| `case 80: {x}: 1 * 1 * {x} * 1 * 1` | 6.04 µs | 4.65 µs | -23.0% |
| `case 81: {x}: {x} * {x} * {x} * 1 * 1 * 1` | 10.92 µs | 6.66 µs | -39.0% |
| `case 82: {x}: {x} * 1 * {x} * 1 * {x} * 1` | 10.92 µs | 6.59 µs | -39.7% |
| `case 83: {x}: 1 * 1 * 1 * 1 - 1 * 1 * 1 * 1` | 8.35 µs | 6.33 µs | -24.3% |
| `case 84: {x}: ({x} * {x} * {x}) * (1 * 1 * 1)` | 15.22 µs | 7.59 µs | -50.1% |
| `case 85: {x}: ({x} * {x} * {x}) - (1 * 1 * 1)` | 14.76 µs | 8.31 µs | -43.7% |
| `case 86: {x}: ({x} * 1 * {x}) - (1 * {x} * 1)` | 13.55 µs | 7.68 µs | -43.3% |
| `case 87: {x}: ({x} * ({x} * ({x} * (1 * (1 * 1)))))` | 68.51 µs | 9.30 µs | -86.4% |
| `case 88: {x}: {x} + {x} + 1 * {x} * 1 * 1` | 11.05 µs | 7.43 µs | -32.8% |
| `case 89: {x}: {x} + {x} + 1 * {x} * 1 * 1 + 1 + 1` | 14.92 µs | 10.68 µs | -28.4% |
| `case 8: (0)` | 2.04 µs | 1.60 µs | -21.7% |
| `case 90: {x}: {x} - 1 - 2 - 3 - 4 - 5` | 8.60 µs | 6.71 µs | -21.9% |
| `case 91: {x}: 1 - {x} - 2 - 3 - 4 - 5` | 13.74 µs | 7.59 µs | -44.7% |
| `case 92: {x}: 1 - 2 - {x} - 3 - 4 - 5` | 13.74 µs | 7.41 µs | -46.1% |
| `case 93: {x}: 1 - 2 - 3 - {x} - 4 - 5` | 13.77 µs | 7.09 µs | -48.5% |
| `case 94: {x}: 1 - 2 - 3 - 4 - {x} - 5` | 13.75 µs | 6.53 µs | -52.5% |
| `case 95: {x}: 1 - 2 - 3 - 4 - 5 - {x}` | 13.72 µs | 5.74 µs | -58.1% |
| `case 96: {x}: {x} / 1 / 2 / 3 / 4 / 5` | 8.34 µs | 6.59 µs | -20.9% |
| `case 97: {x}: 1 / {x} / 2 / 3 / 4 / 5` | 13.47 µs | 7.36 µs | -45.3% |
| `case 98: {x}: 1 / 2 / {x} / 3 / 4 / 5` | 13.53 µs | 5.30 µs | -60.8% |
| `case 99: {x}: 1 / 2 / 3 / {x} / 4 / 5` | 13.47 µs | 5.10 µs | -62.2% |
| `case 9: {a}` | 1.33 µs | 1.32 µs | -0.9% |

### parse

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `case 0: 0` | 573.5 ns | 471.7 ns | -17.8% |
| `case 100: {x}: 1 / 2 / 3 / 4 / {x} / 5` | 2.93 µs | 2.21 µs | -24.6% |
| `case 101: {x}: 1 / 2 / 3 / 4 / 5 / {x}` | 2.90 µs | 2.20 µs | -24.3% |
| `case 102: 0 + 1` | 1.05 µs | 805.8 ns | -23.5% |
| `case 103: 2147483647 + 1` | 1.06 µs | 814.1 ns | -23.3% |
| `case 104: -2147483648 + -1` | 382.0 ns | 400.9 ns | +4.9% |
| `case 105: 0 - 1` | 1.06 µs | 805.3 ns | -23.9% |
| `case 106: 2147483647 - -1` | 731.8 ns | 619.2 ns | -15.4% |
| `case 107: -2147483648 - 1` | 729.1 ns | 626.6 ns | -14.1% |
| `case 108: 100 * 100` | 1.02 µs | 777.9 ns | -23.4% |
| `case 109: 32768 * 65536` | 1.02 µs | 781.4 ns | -23.5% |
| `case 10: {alloy}` | 614.1 ns | 672.7 ns | +9.5% |
| `case 110: 32768 * 65537` | 1.02 µs | 781.2 ns | -23.3% |
| `case 111: 200 / 100` | 1.02 µs | 777.6 ns | -23.8% |
| `case 112: 100 / 0` | 1.01 µs | 778.4 ns | -22.8% |
| `case 113: -2147483648 / -1` | 343.6 ns | 370.1 ns | +7.7% |
| `case 114: 200 % 100` | 1.01 µs | 777.5 ns | -23.3% |
| `case 115: 100 % 0` | 1.02 µs | 775.8 ns | -23.7% |
| `case 116: 0 ^ 0` | 1.00 µs | 787.5 ns | -21.5% |
| `case 117: 0 ^ 1` | 1.01 µs | 789.6 ns | -21.6% |
| `case 118: 0 ^ -1` | 671.9 ns | 560.8 ns | -16.5% |
| `case 119: 1 ^ 0` | 1.00 µs | 788.4 ns | -21.3% |
| `case 11: {a0}` | 607.3 ns | 662.2 ns | +9.0% |
| `case 120: 1 ^ 1` | 1.00 µs | 788.3 ns | -21.2% |
| `case 121: 1 ^ -1` | 669.4 ns | 562.0 ns | -16.0% |
| `case 122: 1 ^ -2` | 670.5 ns | 560.6 ns | -16.4% |
| `case 123: (-1) ^ 0` | 1.32 µs | 1.10 µs | -17.0% |
| `case 124: (-1) ^ 1` | 1.32 µs | 1.09 µs | -17.2% |
| `case 125: (-1) ^ 2` | 1.32 µs | 1.09 µs | -17.4% |
| `case 126: (-1) ^ -1` | 976.2 ns | 893.9 ns | -8.4% |
| `case 127: (-1) ^ -2` | 977.8 ns | 895.0 ns | -8.5% |
| `case 128: 10 ^ 0` | 1.01 µs | 786.9 ns | -21.7% |
| `case 129: 10 ^ 3` | 1.01 µs | 788.8 ns | -21.7% |
| `case 12: {kebab-case}` | 616.9 ns | 689.1 ns | +11.7% |
| `case 130: 10 ^ 10000` | 1.01 µs | 789.4 ns | -22.0% |
| `case 131: 10 ^ -1` | 672.9 ns | 562.6 ns | -16.4% |
| `case 132: -0` | 221.3 ns | 242.3 ns | +9.5% |
| `case 133: -(-2147483648)` | 964.8 ns | 894.6 ns | -7.3% |
| `case 134: 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10` | 4.97 µs | 3.44 µs | -30.8% |
| `case 135: {x}: 1 + 4 + {x} + 4 + 9` | 2.62 µs | 2.04 µs | -22.0% |
| `case 136: 1 + 4 + {x} + 4 + 9` | 2.53 µs | 1.93 µs | -23.6% |
| `case 137: [5:10]` | 1.31 µs | 1.08 µs | -17.0% |
| `case 138: [7:7]` | 1.29 µs | 1.08 µs | -16.1% |
| `case 139: [5:4]` | 1.29 µs | 1.08 µs | -16.2% |
| `case 13: {snake_case}` | 616.6 ns | 688.8 ns | +11.7% |
| `case 140: 3D6` | 486.1 ns | 569.9 ns | +17.2% |
| `case 141: 0D100` | 485.4 ns | 571.9 ns | +17.8% |
| `case 142: 100D0` | 487.4 ns | 569.3 ns | +16.8% |
| `case 143: 100D1` | 486.1 ns | 570.8 ns | +17.4% |
| `case 144: 3D6 drop lowest 1` | 593.4 ns | 713.4 ns | +20.2% |
| `case 145: 3D6 drop lowest 3` | 594.6 ns | 709.2 ns | +19.3% |
| `case 146: 3D6 drop lowest 5` | 593.8 ns | 708.4 ns | +19.3% |
| `case 147: 0D6 drop lowest 3` | 592.6 ns | 709.9 ns | +19.8% |
| `case 148: 6D0 drop lowest 3` | 592.0 ns | 709.5 ns | +19.9% |
| `case 149: 8D1 drop lowest 5` | 592.4 ns | 710.9 ns | +20.0% |
| `case 14: {qualified.access}` | 637.4 ns | 710.9 ns | +11.5% |
| `case 150: 3D6 drop highest 1` | 596.2 ns | 716.8 ns | +20.2% |
| `case 151: 3D6 drop highest 3` | 595.4 ns | 714.7 ns | +20.0% |
| `case 152: 3D6 drop highest 5` | 595.7 ns | 714.3 ns | +19.9% |
| `case 153: 10D10 drop lowest (1 + 3)` | 1.74 µs | 1.58 µs | -9.0% |
| `case 154: 0D6 drop highest 3` | 596.6 ns | 714.6 ns | +19.8% |
| `case 155: 6D0 drop highest 3` | 596.4 ns | 719.6 ns | +20.7% |
| `case 156: 8D1 drop highest 5` | 595.7 ns | 715.5 ns | +20.1% |
| `case 157: 10D10 drop highest (1 + 3)` | 1.75 µs | 1.59 µs | -8.9% |
| `case 158: 8D1 drop lowest 3 drop highest 3` | 705.8 ns | 838.8 ns | +18.8% |
| `case 159: 8D1 drop highest 3 drop lowest 3` | 705.2 ns | 839.0 ns | +19.0% |
| `case 15: {dés}` | 617.2 ns | 664.9 ns | +7.7% |
| `case 160: {x}D(3 + 3)` | 1.78 µs | 1.66 µs | -6.6% |
| `case 161: {x}D0` | 584.0 ns | 765.5 ns | +31.1% |
| `case 162: {x}D(-1)` | 904.9 ns | 1.11 µs | +22.9% |
| `case 163: {x}D1` | 582.0 ns | 763.4 ns | +31.2% |
| `case 164: 0D{x}` | 581.5 ns | 681.5 ns | +17.2% |
| `case 165: (-1)D{x}` | 921.3 ns | 1.03 µs | +11.7% |
| `case 166: 1D[0, 0, 0, 1, 1, 2]` | 1.17 µs | 1.17 µs | +0.3% |
| `case 167: 0D[0, 0, 0, 1, 1, 2]` | 1.16 µs | 1.18 µs | +1.4% |
| `case 168: 5D[1]` | 836.6 ns | 856.0 ns | +2.3% |
| `case 169: 8D[3, 3, 3]` | 962.5 ns | 967.4 ns | +0.5% |
| `case 16: {an external variable}` | 640.8 ns | 743.9 ns | +16.1% |
| `case 170: 5D[1, 2, 3] drop lowest 5` | 1.08 µs | 1.11 µs | +2.9% |
| `case 171: 5D[1, 2, 3] drop lowest 6` | 1.08 µs | 1.11 µs | +2.9% |
| `case 172: 5D[1, 1, 1] drop lowest 3` | 1.08 µs | 1.12 µs | +3.6% |
| `case 173: 5D[1, 2, 3] drop highest 5` | 1.10 µs | 1.12 µs | +2.5% |
| `case 174: 5D[1, 2, 3] drop highest 6` | 1.08 µs | 1.12 µs | +3.6% |
| `case 175: 5D[1, 1, 1] drop highest 3` | 1.08 µs | 1.12 µs | +3.8% |
| `case 176: 5D[5, 5, 5, 5] drop highest 3 drop lowest 1` | 1.25 µs | 1.30 µs | +4.1% |
| `case 177: 5D[5, 5, 5, 5] drop lowest 3 drop highest 1` | 1.26 µs | 1.30 µs | +3.4% |
| `case 178: 3D1 + 7` | 970.0 ns | 930.1 ns | -4.1% |
| `case 179: 7 + 3D1` | 984.6 ns | 932.6 ns | -5.3% |
| `case 17: {выражение в кости}` | 955.9 ns | 749.7 ns | -21.6% |
| `case 180: {x}: 0 + {x}` | 1.19 µs | 1.04 µs | -12.7% |
| `case 181: {x}: {x} + 0` | 1.18 µs | 1.03 µs | -12.7% |
| `case 182: {x}: {x} + 1` | 1.18 µs | 1.04 µs | -12.1% |
| `case 183: {x}: 0 - {x}` | 1.18 µs | 1.03 µs | -12.7% |
| `case 184: {x}: {x} - 0` | 1.18 µs | 1.03 µs | -12.1% |
| `case 185: {x}: {x} - {x}` | 1.22 µs | 1.16 µs | -5.5% |
| `case 186: {x}: {x} - 1` | 1.18 µs | 1.03 µs | -12.8% |
| `case 187: {x}: 0 * {x}` | 1.14 µs | 997.8 ns | -12.4% |
| `case 188: {x}: {x} * 0` | 1.13 µs | 997.8 ns | -12.0% |
| `case 189: {x}: -1 * {x}` | 784.2 ns | 792.8 ns | +1.1% |
| `case 18: {έκφραση ζαριών}` | 919.9 ns | 738.4 ns | -19.7% |
| `case 190: {x}: {x} * -1` | 786.2 ns | 803.2 ns | +2.2% |
| `case 191: {x}: {x} * 2` | 1.14 µs | 998.5 ns | -12.2% |
| `case 192: {x}: 2 * {x}` | 1.14 µs | 999.7 ns | -12.4% |
| `case 193: {x}: {x} * 3` | 1.14 µs | 1.00 µs | -12.3% |
| `case 194: {x}: 0 / {x}` | 1.14 µs | 996.0 ns | -12.6% |
| `case 195: {x}: {x} / 0` | 1.14 µs | 998.7 ns | -12.2% |
| `case 196: {x}: {x} / 1` | 1.13 µs | 1.00 µs | -11.8% |
| `case 197: {x}: {x} / -1` | 787.8 ns | 801.3 ns | +1.7% |
| `case 198: {x}: {x} / {x}` | 1.18 µs | 1.12 µs | -5.0% |
| `case 199: {x}: {x} / 2` | 1.14 µs | 1.00 µs | -11.8% |
| `case 19: {a}: {a}` | 677.8 ns | 687.0 ns | +1.4% |
| `case 1: 1` | 564.2 ns | 471.1 ns | -16.5% |
| `case 200: {x}: 0 % {x}` | 1.14 µs | 1.00 µs | -12.1% |
| `case 201: {x}: {x} % 0` | 1.14 µs | 1.01 µs | -11.6% |
| `case 202: {x}: {x} % 1` | 1.14 µs | 1.01 µs | -11.5% |
| `case 203: {x}: {x} % 2` | 1.14 µs | 1.01 µs | -11.7% |
| `case 204: {x}: 0 ^ {x}` | 1.13 µs | 1.01 µs | -10.6% |
| `case 205: {x}: 1 ^ {x}` | 1.14 µs | 1.01 µs | -11.1% |
| `case 206: {x}: {x} ^ 0` | 1.13 µs | 1.01 µs | -10.5% |
| `case 207: {x}: {x} ^ 1` | 1.13 µs | 1.02 µs | -10.0% |
| `case 208: {x}: {x} ^ 2` | 1.13 µs | 1.02 µs | -10.0% |
| `case 209: {x}: {x} ^ 3` | 1.13 µs | 1.02 µs | -10.1% |
| `case 20: {a}, {an external variable}: {an external variable}` | 814.1 ns | 875.1 ns | +7.5% |
| `case 210: {x}: {x} ^ -1` | 783.5 ns | 784.6 ns | +0.1% |
| `case 211: {x}: -{x}` | 734.5 ns | 752.0 ns | +2.4% |
| `case 212: {x}: --{x}` | 779.9 ns | 875.7 ns | +12.3% |
| `case 213: {x}: ---{x}` | 837.1 ns | 954.5 ns | +14.0% |
| `case 214: {x}: ----{x}` | 882.0 ns | 1.02 µs | +16.1% |
| `case 215: {x}: --{x} + 2` | 1.29 µs | 1.23 µs | -4.7% |
| `case 216: {x}: [{x}:{x}]` | 1.44 µs | 1.43 µs | -0.4% |
| `case 217: {x}, {y}: [{x}:{y}]` | 1.52 µs | 1.52 µs | +0.1% |
| `case 218: {x}: {x}D1` | 660.1 ns | 792.8 ns | +20.1% |
| `case 219: 1D6` | 484.8 ns | 567.8 ns | +17.1% |
| `case 21: ({a})` | 1.62 µs | 1.10 µs | -32.6% |
| `case 220: {x}: 1D{x}` | 658.5 ns | 784.2 ns | +19.1% |
| `case 221: {x}, {y}: {x}D{y}` | 837.4 ns | 1.01 µs | +20.9% |
| `case 222: 10D1` | 486.5 ns | 570.7 ns | +17.3% |
| `case 223: {x}: {x}D[1]` | 1.02 µs | 1.08 µs | +5.2% |
| `case 224: {x}: {x}D[1, 1, 1]` | 1.15 µs | 1.20 µs | +4.2% |
| `case 225: {x}: {x}D[5, 5, 5, 5, 5]` | 1.27 µs | 1.32 µs | +4.1% |
| `case 226: {x}: 1D[3, 4, 5, 6]` | 1.10 µs | 1.12 µs | +1.6% |
| `case 227: {x}: {x}D[1, 2, 3]` | 1.14 µs | 1.19 µs | +4.1% |
| `case 228: {x}: {x}D[3, 4, 5, 6, 7, 8]` | 1.34 µs | 1.38 µs | +2.7% |
| `case 229: {x}: {x}D[1, 1, 2, 3, 5, 8]` | 1.34 µs | 1.38 µs | +3.0% |
| `case 22: [0:5]` | 1.29 µs | 1.08 µs | -16.5% |
| `case 230: {x}: 10D1 drop lowest {x}` | 765.9 ns | 919.7 ns | +20.1% |
| `case 231: {x}, {y}: {x}D1 drop lowest {y}` | 947.9 ns | 1.14 µs | +20.3% |
| `case 232: {x}: 1D{x} drop lowest 1` | 767.2 ns | 917.2 ns | +19.5% |
| `case 233: {x}: {x}D1 drop lowest 1` | 768.6 ns | 922.1 ns | +20.0% |
| `case 234: {x}, {y}: 1D{x} drop lowest {y}` | 944.7 ns | 1.14 µs | +20.3% |
| `case 235: {x}, {y}: 3D{x} drop lowest {y}` | 945.8 ns | 1.14 µs | +20.2% |
| `case 236: 3D6 drop lowest 0` | 593.5 ns | 706.6 ns | +19.1% |
| `case 237: 3D6 drop lowest 1 drop lowest 1` | 707.4 ns | 830.4 ns | +17.4% |
| `case 238: 3D6 drop lowest 1 drop lowest (-1)` | 1.02 µs | 1.14 µs | +11.1% |
| `case 239: {x}: 10D1 drop highest {x}` | 770.5 ns | 922.0 ns | +19.7% |
| `case 23: [-10:10]` | 973.3 ns | 902.6 ns | -7.3% |
| `case 240: {x}, {y}: {x}D1 drop highest {y}` | 951.4 ns | 1.15 µs | +20.5% |
| `case 241: {x}: 1D{x} drop highest 1` | 773.1 ns | 921.0 ns | +19.1% |
| `case 242: {x}: {x}D1 drop highest 1` | 771.4 ns | 926.8 ns | +20.2% |
| `case 243: {x}, {y}: 1D{x} drop highest {y}` | 948.8 ns | 1.14 µs | +20.2% |
| `case 244: {x}, {y}: 3D{x} drop highest {y}` | 950.8 ns | 1.14 µs | +20.0% |
| `case 245: 3D6 drop highest 0` | 597.1 ns | 711.5 ns | +19.1% |
| `case 246: 3D6 drop highest 1 drop highest 1` | 722.8 ns | 837.4 ns | +15.9% |
| `case 247: 3D6 drop highest 1 drop highest (-1)` | 1.03 µs | 1.15 µs | +12.1% |
| `case 248: {x}: 10D1 drop lowest {x} drop highest 1` | 877.7 ns | 1.04 µs | +18.4% |
| `case 249: {x}: 10D1 drop lowest 1 drop highest {x}` | 878.5 ns | 1.05 µs | +19.3% |
| `case 24: [1d3 + 5 : 55 - 8D4]` | 2.16 µs | 2.03 µs | -6.1% |
| `case 250: {x}: {x}D[1] drop lowest 3` | 1.13 µs | 1.22 µs | +7.4% |
| `case 251: {x}: {x}D[1, 1, 1] drop lowest 3` | 1.26 µs | 1.34 µs | +6.1% |
| `case 252: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 5` | 1.40 µs | 1.46 µs | +4.9% |
| `case 253: {x}: 1D[3, 4, 5, 6] drop lowest {x}` | 1.32 µs | 1.39 µs | +5.6% |
| `case 254: {x}, {y}: {x}D[1, 2, 3] drop lowest {y}` | 1.46 µs | 1.56 µs | +6.7% |
| `case 255: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop lowest {y}` | 1.65 µs | 1.75 µs | +6.0% |
| `case 256: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop lowest {y}` | 1.64 µs | 1.75 µs | +6.4% |
| `case 257: {x}: {x}D[1] drop highest 3` | 1.14 µs | 1.22 µs | +7.0% |
| `case 258: {x}: {x}D[1, 1, 1] drop highest 3` | 1.26 µs | 1.34 µs | +6.0% |
| `case 259: {x}: {x}D[5, 5, 5, 5, 5] drop highest 5` | 1.40 µs | 1.47 µs | +5.3% |
| `case 25: 2d8` | 488.2 ns | 572.9 ns | +17.4% |
| `case 260: {x}: 1D[3, 4, 5, 6] drop highest {x}` | 1.32 µs | 1.39 µs | +5.4% |
| `case 261: {x}, {y}: {x}D[1, 2, 3] drop highest {y}` | 1.45 µs | 1.56 µs | +7.9% |
| `case 262: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop highest {y}` | 1.65 µs | 1.76 µs | +6.5% |
| `case 263: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop highest {y}` | 1.65 µs | 1.75 µs | +6.5% |
| `case 264: {x}: {x}D[1] drop highest 1 drop lowest 1` | 1.27 µs | 1.34 µs | +5.9% |
| `case 265: {x}: {x}D[1, 1, 1] drop highest 1 drop lowest 1` | 1.38 µs | 1.46 µs | +6.4% |
| `case 266: {x}: {x}D[5, 5, 5, 5, 5] drop lowest 1 drop highest 1` | 1.51 µs | 1.58 µs | +4.8% |
| `case 267: {x}: 1D[3, 4, 5, 6] drop highest {x} drop lowest {x}` | 1.54 µs | 1.63 µs | +6.1% |
| `case 268: {x}, {y}: {x}D[1, 2, 3] drop highest {y} drop lowest {y}` | 1.65 µs | 1.80 µs | +9.4% |
| `case 269: {x}, {y}: {x}D[3, 4, 5, 6, 7, 8] drop highest {y} drop lowest {y}` | 1.84 µs | 1.99 µs | +8.0% |
| `case 26: 20d10` | 488.9 ns | 570.4 ns | +16.7% |
| `case 270: {x}, {y}: {x}D[1, 1, 2, 3, 5, 8] drop lowest {y} drop highest {y}` | 1.85 µs | 2.01 µs | +8.9% |
| `case 271: {x}, {y}: 20D10 drop lowest {x} drop lowest {y} drop highest {x} drop highest {y}` | 1.48 µs | 1.73 µs | +17.1% |
| `case 272: {x}, {y}: 20D10 drop highest {x} drop highest {y} drop lowest {x} drop lowest {y}` | 1.48 µs | 1.74 µs | +17.7% |
| `case 273: {x}, {y}: 20D10 drop highest {x} drop lowest {y} drop highest {x} drop lowest {y}` | 1.46 µs | 1.74 µs | +18.8% |
| `case 274: {x}, {y}: 20D10 drop lowest {x} drop highest {y} drop lowest {x} drop highest {y}` | 1.46 µs | 1.73 µs | +18.8% |
| `case 275: [2:8] + [2:8]` | 2.48 µs | 2.02 µs | -18.5% |
| `case 276: [2:8] - [2:8]` | 2.51 µs | 2.05 µs | -18.4% |
| `case 277: 3D6 + 3D6` | 880.4 ns | 1.03 µs | +17.0% |
| `case 278: 3D6 - 3D6` | 878.1 ns | 1.03 µs | +17.3% |
| `case 279: 2D[-1, 0, 1] + 2D[-1, 0, 1]` | 1.85 µs | 1.82 µs | -1.7% |
| `case 27: {x}D{y}` | 684.1 ns | 892.2 ns | +30.4% |
| `case 280: 2D[-1, 0, 1] - 2D[-1, 0, 1]` | 1.86 µs | 1.85 µs | -0.5% |
| `case 281: 3D6 drop lowest 1 + 3D6 drop lowest 1` | 1.10 µs | 1.29 µs | +17.1% |
| `case 282: 3D6 drop highest 1 + 3D6 drop highest 1` | 1.10 µs | 1.29 µs | +17.1% |
| `case 283: {x}: 9D6 drop lowest ({x} + 1) + 9D6 drop lowest ({x} + 1)` | 3.60 µs | 3.33 µs | -7.6% |
| `case 284: (1 + 1) + (1 + 1)` | 5.00 µs | 2.52 µs | -49.5% |
| `case 285: (2 + 1) + (1 + 2)` | 4.99 µs | 2.52 µs | -49.6% |
| `case 286: {x}: (1 + {x}) - (1 + {x})` | 5.24 µs | 2.85 µs | -45.6% |
| `case 287: {x}: ({x} - 1) + ({x} - 1)` | 5.22 µs | 2.84 µs | -45.6% |
| `case 288: {x}: (1 - {x}) + (1 - {x})` | 5.23 µs | 2.85 µs | -45.5% |
| `case 289: {x}: ({x} / 3) + ({x} / 3)` | 5.06 µs | 2.80 µs | -44.7% |
| `case 28: 1D[-1, 0, 1]` | 961.8 ns | 969.5 ns | +0.8% |
| `case 290: {x}: (3 / {x}) + (3 / {x})` | 5.07 µs | 2.79 µs | -45.0% |
| `case 291: {x}: ({x} % 3) + ({x} % 3)` | 5.08 µs | 2.79 µs | -45.0% |
| `case 292: {x}: (3 % {x}) + (3 % {x})` | 5.08 µs | 2.79 µs | -45.0% |
| `case 293: {x}: ({x} ^ 3) + ({x} ^ 3)` | 5.02 µs | 2.81 µs | -44.1% |
| `case 294: {x}: (3 ^ {x}) + (3 ^ {x})` | 5.05 µs | 2.82 µs | -44.3% |
| `case 295: {x}: -{x} + -{x}` | 1.33 µs | 1.31 µs | -1.7% |
| `case 296: {x}: ({x} + 1) + -({x} + 1)` | 5.28 µs | 2.92 µs | -44.6% |
| `case 297: {x}, {y}: ({x} + {y}) - ({x} + {y}) * ({x} + {y}) / ({x} + {y}) % ({x} + {y})` | 13.12 µs | 7.24 µs | -44.8% |
| `case 298: {x}, {y}: (({x} + {y}) + ({x} + {y})) + (({x} + {y}) + ({x} + {y}))` | 22.18 µs | 6.96 µs | -68.6% |
| `case 299: {x}: {x}` | 681.1 ns | 682.1 ns | +0.1% |
| `case 29: (1d6)` | 1.40 µs | 1.11 µs | -20.7% |
| `case 2: -1` | 220.9 ns | 242.5 ns | +9.7% |
| `case 300: {x}: {x} + 1 + 1 + 1 + 1` | 2.62 µs | 2.03 µs | -22.2% |
| `case 301: {x}, {y}: {x} + {y}` | 1.30 µs | 1.25 µs | -3.8% |
| `case 302: {u}, {v}, {w}, {x}, {y}, {z}: {u} + {v} + {w} + {x} + {y} + {z}` | 3.74 µs | 3.47 µs | -7.1% |
| `case 303: {x}` | 603.9 ns | 654.6 ns | +8.4% |
| `case 304: {x} + 1` | 1.10 µs | 997.0 ns | -9.3% |
| `case 305: {x} + 1 + 1 + 1 + 1` | 2.51 µs | 2.00 µs | -20.2% |
| `case 306: {x} + {y}` | 1.14 µs | 1.13 µs | -1.3% |
| `case 307: {u} + {v} + {w} + {x} + {y} + {z}` | 3.21 µs | 2.91 µs | -9.3% |
| `case 308: [1 + 2 : 3 + 4]` | 2.26 µs | 1.77 µs | -21.9% |
| `case 309: (4 - 2)D(3 - 1)` | 2.89 µs | 2.28 µs | -21.0% |
| `case 30: (1)d(6)` | 1.82 µs | 1.59 µs | -12.8% |
| `case 310: (4 * 2)D[-1, 0, 1]` | 2.18 µs | 1.82 µs | -16.3% |
| `case 311: (8 * 2)D(24 / 3) drop lowest (10 % 7)` | 4.04 µs | 3.12 µs | -22.8% |
| `case 312: (3 ^ 3)D(24 / 3) drop highest (3 + -1)` | 3.74 µs | 3.00 µs | -20.0% |
| `case 313: (1D3)D3` | 1.15 µs | 1.21 µs | +5.2% |
| `case 314: 1D(1D3)` | 1.09 µs | 1.21 µs | +11.0% |
| `case 315: (1D3)D(1D3)` | 1.75 µs | 1.80 µs | +3.3% |
| `case 316: {x}@(3D6)` | 1.56 µs | 1.19 µs | -23.7% |
| `case 317: {x}@(3D6) + {x}` | 2.11 µs | 1.67 µs | -20.9% |
| `case 318: {x}@(3D6) + {x} + {x}` | 2.65 µs | 2.13 µs | -19.6% |
| `case 319: {a}@({b}@(3D6) + {b}) + {a} + {b}` | 5.84 µs | 3.14 µs | -46.2% |
| `case 31: (1d[1, 1, 2, 3, 5, 8])D(1D8)` | 2.47 µs | 2.42 µs | -2.0% |
| `case 320: {y}: {x}@(3D6) + {x} + {y}` | 2.72 µs | 2.16 µs | -20.7% |
| `case 321: {x}@(2+3)D6` | 1.75 µs | 1.52 µs | -13.4% |
| `case 322: {x}@(7)` | 1.73 µs | 1.06 µs | -38.6% |
| `case 323: {x}@(7) + {x}` | 2.27 µs | 1.55 µs | -31.8% |
| `case 324: {x}@(1D6) + {x}` | 2.10 µs | 1.66 µs | -21.0% |
| `case 325: {x}@([1:6]) + {x}` | 3.69 µs | 2.19 µs | -40.5% |
| `case 326: {x}@(2D4) * {x}` | 2.06 µs | 1.63 µs | -20.9% |
| `case 327: {hp}@(3D6) + {hp}` | 2.11 µs | 1.66 µs | -21.3% |
| `case 328: {score}@(1D20) + {score} + {score}` | 2.69 µs | 2.13 µs | -20.7% |
| `case 329: {x}@(1D6) + {y}@(1D4) + {x} * {y}` | 4.10 µs | 3.04 µs | -25.9% |
| `case 32: 3D10 drop lowest` | 847.6 ns | 885.1 ns | +4.4% |
| `case 330: {a}@({b}@(1D4) + {b}) + {a}` | 5.38 µs | 2.67 µs | -50.3% |
| `case 331: {x}@(2) ^ {x}` | 2.21 µs | 1.49 µs | -32.7% |
| `case 332: {n}@(1D3)D6 + {n}` | 1.71 µs | 1.76 µs | +3.2% |
| `case 333: 3D{f}@(6) + {f}` | 1.71 µs | 1.59 µs | -6.8% |
| `case 334: 4D6 drop lowest {k}@(1) + {k}` | 1.81 µs | 1.71 µs | -5.7% |
| `case 335: {y}: {x}@(1D6) + {x} + {y}` | 2.75 µs | 2.15 µs | -21.9% |
| `case 336: {x}@(2+3)` | 2.67 µs | 1.41 µs | -47.2% |
| `case 337: {x}@(3) + {y}@(4) + {x} + {y}` | 4.48 µs | 2.85 µs | -36.4% |
| `case 33: 3D10 drop lowest 2` | 594.4 ns | 702.8 ns | +18.2% |
| `case 34: 10D10 drop lowest (3 + 3)` | 1.75 µs | 1.57 µs | -10.8% |
| `case 35: 3D10 drop highest` | 854.7 ns | 889.7 ns | +4.1% |
| `case 36: 3D10 drop highest 2` | 596.4 ns | 705.6 ns | +18.3% |
| `case 37: 10D10 drop highest (3 + 3)` | 1.75 µs | 1.57 µs | -10.4% |
| `case 38: 10D10 drop highest 2 drop lowest 5` | 706.4 ns | 832.5 ns | +17.8% |
| `case 39: 10D10 drop lowest (1 + 3) drop highest {x}` | 1.97 µs | 1.82 µs | -7.8% |
| `case 3: 10` | 565.9 ns | 473.4 ns | -16.3% |
| `case 40: 0 + 0` | 1.06 µs | 802.1 ns | -24.3% |
| `case 41: 0 - 0` | 1.06 µs | 801.9 ns | -24.2% |
| `case 42: 0 * 0` | 1.01 µs | 770.9 ns | -24.0% |
| `case 43: {a} × {b}` | 1.10 µs | 1.10 µs | -0.4% |
| `case 44: 0 / 0` | 1.01 µs | 776.0 ns | -23.5% |
| `case 45: {a} ÷ {b}` | 1.10 µs | 1.10 µs | -0.2% |
| `case 46: {a} % {b}` | 1.10 µs | 1.09 µs | -0.5% |
| `case 47: -10` | 234.8 ns | 251.7 ns | +7.2% |
| `case 48: 0 + 1 + 2` | 1.53 µs | 1.15 µs | -25.0% |
| `case 49: (0 + 1) + 2` | 3.04 µs | 1.67 µs | -44.9% |
| `case 4: 100` | 568.1 ns | 477.4 ns | -16.0% |
| `case 50: 0 + (1 + 2)` | 3.04 µs | 1.66 µs | -45.3% |
| `case 51: 0 + 1 - 2` | 1.53 µs | 1.15 µs | -25.0% |
| `case 52: 0 + 1 * 2` | 1.48 µs | 1.12 µs | -24.0% |
| `case 53: 0 + 1 / 2` | 1.48 µs | 1.13 µs | -24.0% |
| `case 54: 0 + 1 % 2` | 1.49 µs | 1.13 µs | -24.1% |
| `case 55: 0 + 1 ^ 2 ^ {a}` | 1.97 µs | 1.57 µs | -20.3% |
| `case 56: 0 ^ {a} - {a} ^ 2` | 2.04 µs | 1.71 µs | -16.2% |
| `case 57: 0 ^ -{a} - -{a} ^ 2` | 2.17 µs | 1.88 µs | -13.3% |
| `case 58: 3D6 + 0` | 970.0 ns | 926.1 ns | -4.5% |
| `case 59: 5 + 3D6` | 1.00 µs | 931.8 ns | -7.1% |
| `case 5: 123456789123456789123456789123456789` | 598.0 ns | 485.9 ns | -18.7% |
| `case 60: 3D6 + 2D12` | 883.4 ns | 1.04 µs | +17.5% |
| `case 61: 3 ^ 3D6 + -2D12` | 1.40 µs | 1.47 µs | +4.5% |
| `case 62: (5 + {a})d({b})` | 2.47 µs | 2.18 µs | -11.9% |
| `case 63: (5 + {a})d({b}) drop lowest 1` | 2.59 µs | 2.29 µs | -11.6% |
| `case 64: (5 + {a})d({b}) drop highest 1 * 3` | 3.07 µs | 2.64 µs | -14.0% |
| `case 65: (5 + {a})d({b}) drop highest 1 % [2:3]` | 3.89 µs | 3.28 µs | -15.8% |
| `case 66: {x}: ({x} + 1) - ({x} + 1)` | 5.25 µs | 2.88 µs | -45.2% |
| `case 67: {x}: 1 + {x}` | 1.18 µs | 1.03 µs | -12.6% |
| `case 68: {x}: {x} + 1 + 1` | 1.69 µs | 1.40 µs | -17.1% |
| `case 69: {x}: 1 + 1 + {x} + 1 + 1` | 2.62 µs | 2.04 µs | -22.1% |
| `case 6: 987654321987654321987654321987654321` | 596.3 ns | 488.8 ns | -18.0% |
| `case 70: {x}: {x} + {x} + {x} + 1 + 1 + 1` | 3.25 µs | 2.63 µs | -19.2% |
| `case 71: {x}: {x} + 1 + {x} + 1 + {x} + 1` | 3.27 µs | 2.62 µs | -19.9% |
| `case 72: {x}: 1 + 1 + 1 + 1 - 1 + 1 + 1 + 1` | 4.01 µs | 2.90 µs | -27.7% |
| `case 73: {x}: ({x} + {x} + {x}) + (1 + 1 + 1)` | 7.33 µs | 3.67 µs | -49.9% |
| `case 74: {x}: ({x} + {x} + {x}) - (1 + 1 + 1)` | 7.33 µs | 3.67 µs | -49.9% |
| `case 75: {x}: ({x} + 1 + {x}) - (1 + {x} + 1)` | 7.32 µs | 3.67 µs | -49.8% |
| `case 76: {x}: ({x} + ({x} + ({x} + (1 + (1 + 1)))))` | 62.39 µs | 5.22 µs | -91.6% |
| `case 77: {x}: {x} * 1` | 1.14 µs | 1.00 µs | -11.8% |
| `case 78: {x}: 1 * {x}` | 1.14 µs | 1.02 µs | -10.6% |
| `case 79: {x}: {x} * 1 * 1` | 1.59 µs | 1.31 µs | -17.1% |
| `case 7: 09` | 567.1 ns | 473.2 ns | -16.6% |
| `case 80: {x}: 1 * 1 * {x} * 1 * 1` | 2.43 µs | 1.92 µs | -21.2% |
| `case 81: {x}: {x} * {x} * {x} * 1 * 1 * 1` | 3.02 µs | 2.45 µs | -18.8% |
| `case 82: {x}: {x} * 1 * {x} * 1 * {x} * 1` | 3.03 µs | 2.45 µs | -19.0% |
| `case 83: {x}: 1 * 1 * 1 * 1 - 1 * 1 * 1 * 1` | 3.70 µs | 2.75 µs | -25.8% |
| `case 84: {x}: ({x} * {x} * {x}) * (1 * 1 * 1)` | 6.95 µs | 3.54 µs | -49.1% |
| `case 85: {x}: ({x} * {x} * {x}) - (1 * 1 * 1)` | 7.02 µs | 3.59 µs | -48.9% |
| `case 86: {x}: ({x} * 1 * {x}) - (1 * {x} * 1)` | 7.03 µs | 3.60 µs | -48.8% |
| `case 87: {x}: ({x} * ({x} * ({x} * (1 * (1 * 1)))))` | 59.59 µs | 5.02 µs | -91.6% |
| `case 88: {x}: {x} + {x} + 1 * {x} * 1 * 1` | 3.09 µs | 2.56 µs | -17.3% |
| `case 89: {x}: {x} + {x} + 1 * {x} * 1 * 1 + 1 + 1` | 4.06 µs | 3.22 µs | -20.7% |
| `case 8: (0)` | 1.56 µs | 990.9 ns | -36.5% |
| `case 90: {x}: {x} - 1 - 2 - 3 - 4 - 5` | 3.12 µs | 2.38 µs | -23.9% |
| `case 91: {x}: 1 - {x} - 2 - 3 - 4 - 5` | 3.13 µs | 2.38 µs | -24.2% |
| `case 92: {x}: 1 - 2 - {x} - 3 - 4 - 5` | 3.13 µs | 2.37 µs | -24.3% |
| `case 93: {x}: 1 - 2 - 3 - {x} - 4 - 5` | 3.12 µs | 2.37 µs | -24.0% |
| `case 94: {x}: 1 - 2 - 3 - 4 - {x} - 5` | 3.13 µs | 2.37 µs | -24.2% |
| `case 95: {x}: 1 - 2 - 3 - 4 - 5 - {x}` | 3.11 µs | 2.36 µs | -24.1% |
| `case 96: {x}: {x} / 1 / 2 / 3 / 4 / 5` | 2.90 µs | 2.22 µs | -23.7% |
| `case 97: {x}: 1 / {x} / 2 / 3 / 4 / 5` | 2.92 µs | 2.21 µs | -24.2% |
| `case 98: {x}: 1 / 2 / {x} / 3 / 4 / 5` | 2.91 µs | 2.21 µs | -24.1% |
| `case 99: {x}: 1 / 2 / 3 / {x} / 4 / 5` | 2.91 µs | 2.20 µs | -24.4% |
| `case 9: {a}` | 605.9 ns | 660.3 ns | +9.0% |

### nesting parse

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `binding/1` | 1.72 µs | 1.06 µs | -38.2% |
| `binding/12` | 4.63 ms | 6.69 µs | -99.9% |
| `binding/16` | 74.45 ms | 8.61 µs | -100.0% |
| `binding/2` | 3.96 µs | 1.63 µs | -59.0% |
| `binding/4` | 17.59 µs | 2.68 µs | -84.8% |
| `binding/8` | 289.03 µs | 4.61 µs | -98.4% |
| `dice count/1` | 1.16 µs | 1.11 µs | -4.9% |
| `dice count/16` | 23.13 µs | 10.05 µs | -56.6% |
| `dice count/256` | 6.12 ms | 145.93 µs | -97.6% |
| `dice count/4` | 3.68 µs | 2.96 µs | -19.5% |
| `dice count/64` | 360.84 µs | 38.35 µs | -89.4% |
| `exponent/1` | 1.01 µs | 787.4 ns | -22.0% |
| `exponent/16` | 7.25 µs | 5.00 µs | -31.1% |
| `exponent/256` | 112.14 µs | 70.64 µs | -37.0% |
| `exponent/4` | 2.24 µs | 1.62 µs | -27.5% |
| `exponent/64` | 27.66 µs | 18.07 µs | -34.7% |
| `group then dice/1` | 1.17 µs | 1.10 µs | -5.9% |
| `group then dice/12` | 2.03 ms | 6.45 µs | -99.7% |
| `group then dice/16` | 32.45 ms | 8.48 µs | -100.0% |
| `group then dice/2` | 2.18 µs | 1.63 µs | -25.5% |
| `group then dice/4` | 8.23 µs | 2.63 µs | -68.1% |
| `group then dice/8` | 127.21 µs | 4.63 µs | -96.4% |
| `group/1` | 1.54 µs | 977.9 ns | -36.5% |
| `group/12` | 4.06 ms | 6.34 µs | -99.8% |
| `group/16` | 64.77 ms | 8.45 µs | -100.0% |
| `group/2` | 3.50 µs | 1.53 µs | -56.4% |
| `group/4` | 15.40 µs | 2.52 µs | -83.6% |
| `group/8` | 252.96 µs | 4.52 µs | -98.2% |
| `negation/1` | 221.2 ns | 241.7 ns | +9.3% |
| `negation/16` | 878.0 ns | 1.40 µs | +58.9% |
| `negation/256` | 14.15 µs | 17.70 µs | +25.1% |
| `negation/4` | 371.9 ns | 498.0 ns | +33.9% |
| `negation/64` | 3.41 µs | 4.65 µs | +36.3% |
| `range/1` | 1.28 µs | 1.09 µs | -14.6% |
| `range/16` | 12.16 µs | 10.02 µs | -17.6% |
| `range/256` | 193.32 µs | 150.11 µs | -22.4% |
| `range/4` | 3.53 µs | 2.93 µs | -17.0% |
| `range/64` | 47.69 µs | 38.59 µs | -19.1% |

### nesting compile

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `binding/1` | 1.90 µs | 1.75 µs | -7.7% |
| `binding/12` | 4.68 ms | 8.90 µs | -99.8% |
| `binding/16` | 74.50 ms | 11.96 µs | -100.0% |
| `binding/2` | 4.21 µs | 2.37 µs | -43.7% |
| `binding/4` | 18.08 µs | 3.86 µs | -78.6% |
| `binding/8` | 291.78 µs | 6.57 µs | -97.7% |
| `dice count/1` | 1.26 µs | 3.31 µs | +163.8% |
| `dice count/16` | 23.99 µs | 41.19 µs | +71.7% |
| `dice count/256` | 6.20 ms | 624.68 µs | -89.9% |
| `dice count/4` | 3.94 µs | 11.40 µs | +189.6% |
| `dice count/64` | 364.80 µs | 159.39 µs | -56.3% |
| `exponent/1` | 1.08 µs | 1.80 µs | +66.9% |
| `exponent/16` | 7.71 µs | 10.48 µs | +36.0% |
| `exponent/256` | 118.78 µs | 134.66 µs | +13.4% |
| `exponent/4` | 2.42 µs | 3.54 µs | +46.1% |
| `exponent/64` | 29.61 µs | 35.73 µs | +20.7% |
| `group then dice/1` | 1.25 µs | 3.34 µs | +166.5% |
| `group then dice/12` | 2.03 ms | 9.22 µs | -99.5% |
| `group then dice/16` | 33.10 ms | 11.73 µs | -100.0% |
| `group then dice/2` | 2.28 µs | 4.00 µs | +75.1% |
| `group then dice/4` | 8.36 µs | 5.07 µs | -39.3% |
| `group then dice/8` | 128.70 µs | 7.29 µs | -94.3% |
| `group/1` | 1.62 µs | 1.56 µs | -3.3% |
| `group/12` | 4.08 ms | 7.43 µs | -99.8% |
| `group/16` | 65.19 ms | 9.68 µs | -100.0% |
| `group/2` | 3.61 µs | 2.12 µs | -41.2% |
| `group/4` | 15.62 µs | 3.32 µs | -78.7% |
| `group/8` | 254.76 µs | 5.50 µs | -97.8% |
| `negation/1` | 277.6 ns | 665.8 ns | +139.8% |
| `negation/16` | 1.13 µs | 4.90 µs | +332.8% |
| `negation/256` | 19.86 µs | 68.52 µs | +245.1% |
| `negation/4` | 447.2 ns | 1.59 µs | +255.0% |
| `negation/64` | 4.90 µs | 18.46 µs | +276.6% |
| `range/1` | 1.35 µs | 2.11 µs | +55.8% |
| `range/16` | 12.66 µs | 16.19 µs | +27.8% |
| `range/256` | 203.05 µs | 223.17 µs | +9.9% |
| `range/4` | 3.73 µs | 5.09 µs | +36.6% |
| `range/64` | 50.20 µs | 58.58 µs | +16.7% |

### evaluations

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `case 0: ("0", [], [])` | 7.6 ns | 6.6 ns | -13.1% |
| `case 0: ("0", [], []): bounds` | 16.8 ns | 16.0 ns | -4.8% |
| `case 100: ("{x}: {x} % -1", [2147483647], [])` | 24.6 ns | 23.4 ns | -5.1% |
| `case 100: ("{x}: {x} % -1", [2147483647], []): bounds` | 49.9 ns | 49.6 ns | -0.5% |
| `case 1020: ("(1D3)D3", [], [])` | 113.2 ns | 108.0 ns | -4.5% |
| `case 1020: ("(1D3)D3", [], []): bounds` | 50.1 ns | 50.6 ns | +1.1% |
| `case 1030: ("(1D3)D(1D3) - (1D3)D[-1, 0, 1]", [], [])` | 276.9 ns | 274.3 ns | -0.9% |
| `case 1030: ("(1D3)D(1D3) - (1D3)D[-1, 0, 1]", [], []): bounds` | 86.0 ns | 87.5 ns | +1.7% |
| `case 1040: ("{x}@([1:6]) + {x}", [], [])` | 61.2 ns | 58.3 ns | -4.8% |
| `case 1040: ("{x}@([1:6]) + {x}", [], []): bounds` | 46.3 ns | 46.2 ns | -0.1% |
| `case 1050: ("{y}: {x}@(1D6) + {x} + {y}", [3], [])` | 72.1 ns | 70.0 ns | -2.9% |
| `case 1050: ("{y}: {x}@(1D6) + {x} + {y}", [3], []): bounds` | 69.2 ns | 69.2 ns | +0.1% |
| `case 10: ("{x}", [], [("x", -1)])` | 13.8 ns | 13.5 ns | -1.9% |
| `case 10: ("{x}", [], [("x", -1)]): bounds` | 54.0 ns | 54.4 ns | +0.6% |
| `case 110: ("{x}: 0 ^ {x}", [2147483647], [])` | 25.1 ns | 24.1 ns | -4.0% |
| `case 110: ("{x}: 0 ^ {x}", [2147483647], []): bounds` | 167.7 ns | 174.8 ns | +4.2% |
| `case 120: ("{x}: 1 ^ {x}", [-2], [])` | 23.3 ns | 22.5 ns | -3.7% |
| `case 120: ("{x}: 1 ^ {x}", [-2], []): bounds` | 46.5 ns | 46.4 ns | -0.2% |
| `case 130: ("{x}: (-1) ^ {x}", [-1], [])` | 25.0 ns | 23.8 ns | -4.9% |
| `case 130: ("{x}: (-1) ^ {x}", [-1], []): bounds` | 163.0 ns | 168.4 ns | +3.3% |
| `case 140: ("{x}: {x} ^ -1", [2147483647], [])` | 25.1 ns | 24.0 ns | -4.5% |
| `case 140: ("{x}: {x} ^ -1", [2147483647], []): bounds` | 164.9 ns | 165.8 ns | +0.6% |
| `case 150: ("{x}: {x} ^ 3", [-2147483648], [])` | 25.6 ns | 24.1 ns | -5.8% |
| `case 150: ("{x}: {x} ^ 3", [-2147483648], []): bounds` | 173.6 ns | 170.8 ns | -1.6% |
| `case 160: ("{x}: -{x}", [0], [])` | 24.7 ns | 23.1 ns | -6.2% |
| `case 160: ("{x}: -{x}", [0], []): bounds` | 47.7 ns | 47.4 ns | -0.6% |
| `case 170: ("{x}: [1:{x}]", [10], [])` | 69.2 ns | 66.7 ns | -3.6% |
| `case 170: ("{x}: [1:{x}]", [10], []): bounds` | 64.1 ns | 63.0 ns | -1.7% |
| `case 180: ("{x}: [{x}:1]", [-2147483648], [])` | 74.0 ns | 70.9 ns | -4.3% |
| `case 180: ("{x}: [{x}:1]", [-2147483648], []): bounds` | 64.0 ns | 63.4 ns | -0.9% |
| `case 190: ("{x}, {y}: [{x}:{y}] + 1", [0, 0], [])` | 70.9 ns | 69.3 ns | -2.2% |
| `case 190: ("{x}, {y}: [{x}:{y}] + 1", [0, 0], []): bounds` | 67.6 ns | 66.8 ns | -1.2% |
| `case 200: ("{x}, {y}: [{x}:{y}] + [{x}:{y}]", [0, 1], [])` | 119.1 ns | 112.0 ns | -5.9% |
| `case 200: ("{x}, {y}: [{x}:{y}] + [{x}:{y}]", [0, 1], []): bounds` | 75.6 ns | 76.5 ns | +1.2% |
| `case 20: ("{x}: {x} + 1", [-2147483648], [])` | 24.4 ns | 23.9 ns | -2.3% |
| `case 20: ("{x}: {x} + 1", [-2147483648], []): bounds` | 47.8 ns | 47.9 ns | +0.3% |
| `case 210: ("{x}, {y}: [{x}:{y}] - [{x}:{y}]", [1, 0], [])` | 113.2 ns | 109.0 ns | -3.7% |
| `case 210: ("{x}, {y}: [{x}:{y}] - [{x}:{y}]", [1, 0], []): bounds` | 73.5 ns | 75.2 ns | +2.2% |
| `case 220: ("{x}, {y}: [{x}:{y}] * [{x}:{y}]", [1, 1], [])` | 120.3 ns | 112.2 ns | -6.8% |
| `case 220: ("{x}, {y}: [{x}:{y}] * [{x}:{y}]", [1, 1], []): bounds` | 76.5 ns | 77.2 ns | +0.9% |
| `case 230: ("{x}, {y}: [{x}:{y}] / [{x}:{y}]", [0, 5], [])` | 120.0 ns | 112.2 ns | -6.5% |
| `case 230: ("{x}, {y}: [{x}:{y}] / [{x}:{y}]", [0, 5], []): bounds` | 91.4 ns | 91.3 ns | -0.1% |
| `case 240: ("{x}, {y}, {z}: [{x}:{y}] % {z}", [3, 9, 10], [])` | 71.3 ns | 67.4 ns | -5.5% |
| `case 240: ("{x}, {y}, {z}: [{x}:{y}] % {z}", [3, 9, 10], []): bounds` | 76.9 ns | 76.5 ns | -0.5% |
| `case 250: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [0, 0], [])` | 119.7 ns | 114.2 ns | -4.5% |
| `case 250: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [0, 0], []): bounds` | 75.8 ns | 76.2 ns | +0.4% |
| `case 260: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [-2147483648, 2147483647], [])` | 117.5 ns | 110.9 ns | -5.6% |
| `case 260: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [-2147483648, 2147483647], []): bounds` | 76.8 ns | 78.5 ns | +2.1% |
| `case 270: ("{x}, {y}: [{x}:{y}] ^ [{x}:{y}]", [-2147483648, 0], [])` | 129.6 ns | 121.5 ns | -6.2% |
| `case 270: ("{x}, {y}: [{x}:{y}] ^ [{x}:{y}]", [-2147483648, 0], []): bounds` | 237.1 ns | 229.2 ns | -3.3% |
| `case 280: ("{x}, {y}: -[{x}:{y}]", [-2147483648, 2147483647], [])` | 70.9 ns | 67.7 ns | -4.5% |
| `case 280: ("{x}, {y}: -[{x}:{y}]", [-2147483648, 2147483647], []): bounds` | 67.3 ns | 65.6 ns | -2.5% |
| `case 290: ("1D0 drop highest 1", [], [])` | 7.6 ns | 6.5 ns | -14.7% |
| `case 290: ("1D0 drop highest 1", [], []): bounds` | 16.8 ns | 15.8 ns | -6.3% |
| `case 300: ("1D1 drop highest 0", [], [])` | 7.7 ns | 6.5 ns | -15.2% |
| `case 300: ("1D1 drop highest 0", [], []): bounds` | 16.8 ns | 16.0 ns | -4.6% |
| `case 30: ("{x}: {x} - 0", [1], [])` | 22.8 ns | 22.6 ns | -0.8% |
| `case 30: ("{x}: {x} - 0", [1], []): bounds` | 46.4 ns | 46.1 ns | -0.6% |
| `case 310: ("1D(-1) drop lowest 1", [], [])` | 7.6 ns | 6.5 ns | -14.6% |
| `case 310: ("1D(-1) drop lowest 1", [], []): bounds` | 16.8 ns | 15.7 ns | -6.4% |
| `case 320: ("3D6", [], [])` | 76.6 ns | 75.2 ns | -1.8% |
| `case 320: ("3D6", [], []): bounds` | 46.4 ns | 46.6 ns | +0.4% |
| `case 330: ("3D6 drop highest 4", [], [])` | 7.6 ns | 6.5 ns | -14.9% |
| `case 330: ("3D6 drop highest 4", [], []): bounds` | 16.8 ns | 15.7 ns | -6.5% |
| `case 340: ("{x}D{y}", [], [("x", 4), ("y", 2)])` | 82.6 ns | 79.3 ns | -4.0% |
| `case 340: ("{x}D{y}", [], [("x", 4), ("y", 2)]): bounds` | 81.7 ns | 82.2 ns | +0.6% |
| `case 350: ("{x}D{y}", [], [("x", 2), ("y", 8)])` | 71.9 ns | 70.8 ns | -1.6% |
| `case 350: ("{x}D{y}", [], [("x", 2), ("y", 8)]): bounds` | 81.4 ns | 82.3 ns | +1.2% |
| `case 360: ("{x}D{y}", [], [("x", 4), ("y", 12)])` | 86.4 ns | 84.1 ns | -2.7% |
| `case 360: ("{x}D{y}", [], [("x", 4), ("y", 12)]): bounds` | 81.9 ns | 82.3 ns | +0.5% |
| `case 370: ("{x}D{y}", [], [("x", 2), ("y", 2147483647)])` | 77.8 ns | 82.8 ns | +6.5% |
| `case 370: ("{x}D{y}", [], [("x", 2), ("y", 2147483647)]): bounds` | 81.1 ns | 81.2 ns | +0.1% |
| `case 380: ("{x}, {y}: {x}D{y} + {x}D{y}", [-1, -1], [])` | 52.3 ns | 48.1 ns | -8.0% |
| `case 380: ("{x}, {y}: {x}D{y} + {x}D{y}", [-1, -1], []): bounds` | 76.1 ns | 76.9 ns | +1.0% |
| `case 390: ("{x}, {y}: {x}D{y} + {x}D{y}", [2, 6], [])` | 132.2 ns | 129.9 ns | -1.8% |
| `case 390: ("{x}, {y}: {x}D{y} + {x}D{y}", [2, 6], []): bounds` | 79.7 ns | 79.1 ns | -0.7% |
| `case 400: ("{x}, {y}: {x}D{y} + {x}D{y}", [4, 10], [])` | 162.8 ns | 161.0 ns | -1.1% |
| `case 400: ("{x}, {y}: {x}D{y} + {x}D{y}", [4, 10], []): bounds` | 80.2 ns | 80.1 ns | -0.1% |
| `case 40: ("{x}: {x} - -1", [1], [])` | 24.7 ns | 23.7 ns | -4.1% |
| `case 40: ("{x}: {x} - -1", [1], []): bounds` | 49.0 ns | 47.9 ns | -2.3% |
| `case 410: ("{x}, {y}: {x}D{y} + {x}D{y}", [2, 100], [])` | 134.7 ns | 129.3 ns | -4.0% |
| `case 410: ("{x}, {y}: {x}D{y} + {x}D{y}", [2, 100], []): bounds` | 78.7 ns | 78.0 ns | -0.9% |
| `case 420: ("{x}, {y}: {x}D{y} - {x}D{y}", [-1, 1], [])` | 52.8 ns | 49.3 ns | -6.6% |
| `case 420: ("{x}, {y}: {x}D{y} - {x}D{y}", [-1, 1], []): bounds` | 71.6 ns | 76.2 ns | +6.4% |
| `case 430: ("{x}, {y}: {x}D{y} - {x}D{y}", [4, 4], [])` | 167.7 ns | 158.3 ns | -5.6% |
| `case 430: ("{x}, {y}: {x}D{y} - {x}D{y}", [4, 4], []): bounds` | 75.2 ns | 78.4 ns | +4.2% |
| `case 440: ("{x}, {y}: {x}D{y} - {x}D{y}", [2, 10], [])` | 135.4 ns | 128.7 ns | -5.0% |
| `case 440: ("{x}, {y}: {x}D{y} - {x}D{y}", [2, 10], []): bounds` | 74.4 ns | 78.4 ns | +5.3% |
| `case 450: ("{x}, {y}: {x}D{y} - {x}D{y}", [4, 20], [])` | 163.7 ns | 159.8 ns | -2.4% |
| `case 450: ("{x}, {y}: {x}D{y} - {x}D{y}", [4, 20], []): bounds` | 75.0 ns | 78.8 ns | +5.1% |
| `case 460: ("{x}, {y}: {x}D{y} * {x}D{y}", [0, 1], [])` | 52.6 ns | 48.8 ns | -7.1% |
| `case 460: ("{x}, {y}: {x}D{y} * {x}D{y}", [0, 1], []): bounds` | 77.6 ns | 76.6 ns | -1.2% |
| `case 470: ("{x}, {y}: {x}D{y} * {x}D{y}", [2, 4], [])` | 135.0 ns | 129.6 ns | -4.0% |
| `case 470: ("{x}, {y}: {x}D{y} * {x}D{y}", [2, 4], []): bounds` | 81.1 ns | 79.1 ns | -2.4% |
| `case 480: ("{x}, {y}: {x}D{y} * {x}D{y}", [4, 8], [])` | 165.1 ns | 159.1 ns | -3.6% |
| `case 480: ("{x}, {y}: {x}D{y} * {x}D{y}", [4, 8], []): bounds` | 81.3 ns | 79.9 ns | -1.7% |
| `case 490: ("{x}, {y}: {x}D{y} * {x}D{y}", [2, 20], [])` | 135.2 ns | 130.3 ns | -3.6% |
| `case 490: ("{x}, {y}: {x}D{y} * {x}D{y}", [2, 20], []): bounds` | 79.3 ns | 79.9 ns | +0.7% |
| `case 500: ("{x}, {y}: {x}D{y} * {x}D{y}", [4, 2147483647], [])` | 196.6 ns | 191.1 ns | -2.8% |
| `case 500: ("{x}, {y}: {x}D{y} * {x}D{y}", [4, 2147483647], []): bounds` | 81.7 ns | 81.6 ns | -0.2% |
| `case 50: ("{x}: {x} * 0", [2147483647], [])` | 22.6 ns | 22.5 ns | -0.6% |
| `case 50: ("{x}: {x} * 0", [2147483647], []): bounds` | 45.7 ns | 45.4 ns | -0.8% |
| `case 510: ("{x}, {y}: {x}D{y} / {x}D{y}", [4, 2], [])` | 155.3 ns | 149.0 ns | -4.1% |
| `case 510: ("{x}, {y}: {x}D{y} / {x}D{y}", [4, 2], []): bounds` | 83.0 ns | 81.2 ns | -2.1% |
| `case 520: ("{x}, {y}: {x}D{y} / {x}D{y}", [2, 8], [])` | 133.2 ns | 128.1 ns | -3.8% |
| `case 520: ("{x}, {y}: {x}D{y} / {x}D{y}", [2, 8], []): bounds` | 80.7 ns | 80.4 ns | -0.3% |
| `case 530: ("{x}, {y}: {x}D{y} / {x}D{y}", [4, 12], [])` | 164.3 ns | 158.8 ns | -3.3% |
| `case 530: ("{x}, {y}: {x}D{y} / {x}D{y}", [4, 12], []): bounds` | 83.1 ns | 81.9 ns | -1.4% |
| `case 540: ("{x}, {y}: {x}D{y} / {x}D{y}", [2, 2147483647], [])` | 149.2 ns | 144.2 ns | -3.4% |
| `case 540: ("{x}, {y}: {x}D{y} / {x}D{y}", [2, 2147483647], []): bounds` | 80.9 ns | 80.2 ns | -0.9% |
| `case 550: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 2], [])` | 129.5 ns | 124.7 ns | -3.7% |
| `case 550: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 2], []): bounds` | 81.7 ns | 80.9 ns | -1.0% |
| `case 560: ("{x}, {y}: {x}D{y} % {x}D{y}", [4, 6], [])` | 163.1 ns | 156.9 ns | -3.8% |
| `case 560: ("{x}, {y}: {x}D{y} % {x}D{y}", [4, 6], []): bounds` | 95.9 ns | 92.9 ns | -3.2% |
| `case 570: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 12], [])` | 134.4 ns | 129.3 ns | -3.8% |
| `case 570: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 12], []): bounds` | 93.6 ns | 95.1 ns | +1.6% |
| `case 580: ("{x}, {y}: {x}D{y} % {x}D{y}", [4, 100], [])` | 164.2 ns | 160.1 ns | -2.5% |
| `case 580: ("{x}, {y}: {x}D{y} % {x}D{y}", [4, 100], []): bounds` | 81.7 ns | 81.7 ns | -0.1% |
| `case 590: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [-1, -1], [])` | 53.1 ns | 48.8 ns | -8.0% |
| `case 590: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [-1, -1], []): bounds` | 185.0 ns | 189.5 ns | +2.4% |
| `case 600: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 6], [])` | 141.2 ns | 135.7 ns | -3.9% |
| `case 600: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 6], []): bounds` | 267.7 ns | 257.9 ns | -3.7% |
| `case 60: ("{x}: {x} * -1", [2147483647], [])` | 24.4 ns | 23.7 ns | -3.1% |
| `case 60: ("{x}: {x} * -1", [2147483647], []): bounds` | 47.4 ns | 47.2 ns | -0.5% |
| `case 610: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [4, 10], [])` | 176.0 ns | 170.2 ns | -3.3% |
| `case 610: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [4, 10], []): bounds` | 272.9 ns | 264.7 ns | -3.0% |
| `case 620: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 100], [])` | 142.5 ns | 137.3 ns | -3.7% |
| `case 620: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 100], []): bounds` | 271.5 ns | 260.9 ns | -3.9% |
| `case 630: ("{x}, {y}: -{x}D{y}", [-1, 1], [])` | 42.4 ns | 40.0 ns | -5.5% |
| `case 630: ("{x}, {y}: -{x}D{y}", [-1, 1], []): bounds` | 68.5 ns | 66.6 ns | -2.8% |
| `case 640: ("{x}, {y}: -{x}D{y}", [4, 4], [])` | 92.5 ns | 90.8 ns | -1.8% |
| `case 640: ("{x}, {y}: -{x}D{y}", [4, 4], []): bounds` | 68.9 ns | 68.0 ns | -1.4% |
| `case 650: ("{x}, {y}: -{x}D{y}", [2, 10], [])` | 78.7 ns | 75.8 ns | -3.7% |
| `case 650: ("{x}, {y}: -{x}D{y}", [2, 10], []): bounds` | 70.1 ns | 67.6 ns | -3.6% |
| `case 660: ("{x}, {y}: -{x}D{y}", [4, 20], [])` | 93.4 ns | 91.0 ns | -2.5% |
| `case 660: ("{x}, {y}: -{x}D{y}", [4, 20], []): bounds` | 68.7 ns | 68.9 ns | +0.2% |
| `case 670: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [0, 1, 1], [])` | 43.6 ns | 40.9 ns | -6.3% |
| `case 670: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [0, 1, 1], []): bounds` | 73.8 ns | 73.5 ns | -0.5% |
| `case 680: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 4, 1], [])` | 78.1 ns | 77.2 ns | -1.1% |
| `case 680: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 4, 1], []): bounds` | 74.8 ns | 74.6 ns | -0.1% |
| `case 690: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 8, 1], [])` | 95.8 ns | 91.9 ns | -4.0% |
| `case 690: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 8, 1], []): bounds` | 76.4 ns | 75.2 ns | -1.6% |
| `case 700: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 20, 1], [])` | 78.2 ns | 76.9 ns | -1.7% |
| `case 700: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 20, 1], []): bounds` | 74.9 ns | 75.1 ns | +0.2% |
| `case 70: ("{x}: {x} / 0", [2147483647], [])` | 23.2 ns | 22.5 ns | -3.2% |
| `case 70: ("{x}: {x} / 0", [2147483647], []): bounds` | 46.5 ns | 46.1 ns | -1.0% |
| `case 710: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2147483647, 1], [])` | 110.9 ns | 108.0 ns | -2.6% |
| `case 710: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2147483647, 1], []): bounds` | 75.8 ns | 76.2 ns | +0.5% |
| `case 720: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2, 2], [])` | 88.4 ns | 86.3 ns | -2.3% |
| `case 720: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2, 2], []): bounds` | 75.5 ns | 75.7 ns | +0.3% |
| `case 730: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 8, 2], [])` | 69.9 ns | 68.1 ns | -2.6% |
| `case 730: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 8, 2], []): bounds` | 75.5 ns | 74.5 ns | -1.4% |
| `case 740: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 12, 2], [])` | 96.5 ns | 90.8 ns | -5.9% |
| `case 740: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 12, 2], []): bounds` | 77.0 ns | 75.4 ns | -2.1% |
| `case 750: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 2147483647, 2], [])` | 78.4 ns | 75.4 ns | -3.8% |
| `case 750: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 2147483647, 2], []): bounds` | 75.2 ns | 74.2 ns | -1.2% |
| `case 760: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 2, 1], [])` | 77.4 ns | 75.0 ns | -3.1% |
| `case 760: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 2, 1], []): bounds` | 75.5 ns | 73.9 ns | -2.0% |
| `case 770: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 6, 1], [])` | 96.4 ns | 91.2 ns | -5.3% |
| `case 770: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 6, 1], []): bounds` | 77.3 ns | 75.7 ns | -2.0% |
| `case 780: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 12, 1], [])` | 81.2 ns | 76.8 ns | -5.4% |
| `case 780: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 12, 1], []): bounds` | 76.0 ns | 74.5 ns | -2.1% |
| `case 790: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 100, 1], [])` | 98.2 ns | 91.1 ns | -7.3% |
| `case 790: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 100, 1], []): bounds` | 76.2 ns | 74.9 ns | -1.8% |
| `case 800: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [-1, -1, 2], [])` | 42.9 ns | 40.8 ns | -4.8% |
| `case 800: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [-1, -1, 2], []): bounds` | 73.5 ns | 73.1 ns | -0.4% |
| `case 80: ("{x}: {x} / -1", [2147483647], [])` | 24.6 ns | 23.7 ns | -3.8% |
| `case 80: ("{x}: {x} / -1", [2147483647], []): bounds` | 47.5 ns | 47.5 ns | -0.1% |
| `case 810: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 6, 2], [])` | 69.9 ns | 68.5 ns | -1.9% |
| `case 810: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 6, 2], []): bounds` | 76.1 ns | 74.7 ns | -1.9% |
| `case 820: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 10, 2], [])` | 95.0 ns | 90.2 ns | -5.0% |
| `case 820: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 10, 2], []): bounds` | 77.5 ns | 75.8 ns | -2.1% |
| `case 830: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 100, 2], [])` | 76.7 ns | 68.7 ns | -10.5% |
| `case 830: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 100, 2], []): bounds` | 76.5 ns | 74.5 ns | -2.6% |
| `case 840: ("0D[-1, 0, 1] drop highest 0", [], [])` | 7.8 ns | 6.5 ns | -16.5% |
| `case 840: ("0D[-1, 0, 1] drop highest 0", [], []): bounds` | 16.5 ns | 15.6 ns | -5.2% |
| `case 850: ("1D[-2, -3, -4] drop highest 0", [], [])` | 58.5 ns | 56.2 ns | -4.0% |
| `case 850: ("1D[-2, -3, -4] drop highest 0", [], []): bounds` | 46.0 ns | 44.3 ns | -3.9% |
| `case 860: ("{x}: {x}D[-1, 0, 1, 3, 5]", [3], [])` | 97.7 ns | 96.0 ns | -1.7% |
| `case 860: ("{x}: {x}D[-1, 0, 1, 3, 5]", [3], []): bounds` | 79.1 ns | 80.1 ns | +1.3% |
| `case 870: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [0, 4], [])` | 55.3 ns | 52.8 ns | -4.6% |
| `case 870: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [0, 4], []): bounds` | 80.9 ns | 74.6 ns | -7.8% |
| `case 880: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [2, 4], [])` | 81.3 ns | 79.8 ns | -1.9% |
| `case 880: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [2, 4], []): bounds` | 81.4 ns | 75.7 ns | -6.9% |
| `case 890: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [4, 4], [])` | 94.8 ns | 93.0 ns | -1.9% |
| `case 890: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [4, 4], []): bounds` | 80.8 ns | 75.3 ns | -6.9% |
| `case 900: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [6, 4], [])` | 125.4 ns | 120.2 ns | -4.2% |
| `case 900: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [6, 4], []): bounds` | 81.8 ns | 75.7 ns | -7.4% |
| `case 90: ("{x}: {x} % 0", [2147483647], [])` | 22.6 ns | 22.6 ns | -0.1% |
| `case 90: ("{x}: {x} % 0", [2147483647], []): bounds` | 45.9 ns | 49.5 ns | +7.8% |
| `case 910: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [8, 4], [])` | 143.6 ns | 135.6 ns | -5.6% |
| `case 910: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [8, 4], []): bounds` | 81.0 ns | 75.4 ns | -6.9% |
| `case 920: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [1, 4], [])` | 74.7 ns | 72.7 ns | -2.7% |
| `case 920: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [1, 4], []): bounds` | 78.6 ns | 74.4 ns | -5.3% |
| `case 930: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [3, 4], [])` | 89.6 ns | 86.3 ns | -3.8% |
| `case 930: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [3, 4], []): bounds` | 80.7 ns | 74.5 ns | -7.7% |
| `case 940: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [5, 4], [])` | 116.5 ns | 111.8 ns | -4.1% |
| `case 940: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [5, 4], []): bounds` | 81.3 ns | 74.8 ns | -8.0% |
| `case 950: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [7, 4], [])` | 132.4 ns | 128.1 ns | -3.3% |
| `case 950: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [7, 4], []): bounds` | 81.9 ns | 75.0 ns | -8.4% |

### serial histograms

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `case 0: ("0", [], [])` | 47.9 ns | 50.7 ns | +5.7% |
| `case 100: ("{x}: {x} % -1", [2147483647], [])` | 65.8 ns | 70.7 ns | +7.5% |
| `case 10: ("{x}", [], [("x", -1)])` | 54.7 ns | 57.1 ns | +4.4% |
| `case 110: ("{x}: 0 ^ {x}", [2147483647], [])` | 67.0 ns | 67.7 ns | +1.1% |
| `case 120: ("{x}: 1 ^ {x}", [-2], [])` | 60.9 ns | 62.5 ns | +2.7% |
| `case 130: ("{x}: (-1) ^ {x}", [-1], [])` | 66.6 ns | 67.9 ns | +1.9% |
| `case 140: ("{x}: {x} ^ -1", [2147483647], [])` | 64.7 ns | 68.1 ns | +5.2% |
| `case 150: ("{x}: {x} ^ 3", [-2147483648], [])` | 64.5 ns | 69.5 ns | +7.8% |
| `case 160: ("{x}: -{x}", [0], [])` | 67.9 ns | 67.9 ns | +0.1% |
| `case 170: ("{x}: [1:{x}]", [10], [])` | 2.84 µs | 2.92 µs | +3.0% |
| `case 180: ("{x}, {y}: [{x}:{y}]", [0, 1], [])` | 339.6 ns | 330.7 ns | -2.6% |
| `case 190: ("{x}, {y}: [{x}:{y}] + 1", [-5, 10], [])` | 6.05 µs | 6.20 µs | +2.4% |
| `case 200: ("{x}, {y}: [{x}:{y}] - [{x}:{y}]", [0, 5], [])` | 10.41 µs | 10.46 µs | +0.5% |
| `case 20: ("{x}: {x} + 1", [-2147483648], [])` | 69.1 ns | 70.7 ns | +2.4% |
| `case 210: ("{x}, {y}: [{x}:{y}] / [{x}:{y}]", [1, 1], [])` | 265.2 ns | 265.8 ns | +0.2% |
| `case 220: ("{x}, {y}, {z}: [{x}:{y}] % {z}", [10, 20, 4], [])` | 3.29 µs | 3.33 µs | +1.3% |
| `case 230: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [-2147483648, -2147483645], [])` | 3.94 µs | 3.96 µs | +0.5% |
| `case 240: ("{x}, {y}: -[{x}:{y}]", [0, 1], [])` | 344.3 ns | 328.6 ns | -4.5% |
| `case 250: ("1D0", [], [])` | 48.3 ns | 50.3 ns | +4.1% |
| `case 260: ("1D1", [], [])` | 48.4 ns | 50.1 ns | +3.5% |
| `case 270: ("(-1)D1 drop highest 0", [], [])` | 48.7 ns | 50.5 ns | +3.6% |
| `case 280: ("1D6 drop lowest 2", [], [])` | 48.9 ns | 50.5 ns | +3.3% |
| `case 290: ("3D6 drop highest 0", [], [])` | 54.77 µs | 55.55 µs | +1.4% |
| `case 300: ("{x}D{y}", [], [("x", -1), ("y", -1)])` | 97.6 ns | 102.3 ns | +4.9% |
| `case 30: ("{x}: {x} - 0", [1], [])` | 60.9 ns | 62.9 ns | +3.3% |
| `case 310: ("{x}D{y}", [], [("x", 2), ("y", 6)])` | 8.91 µs | 9.05 µs | +1.6% |
| `case 320: ("{x}D{y}", [], [("x", 4), ("y", 10)])` | 3.30 ms | 3.34 ms | +1.3% |
| `case 330: ("{x}, {y}: {x}D{y} + {x}D{y}", [0, 0], [])` | 158.2 ns | 153.1 ns | -3.3% |
| `case 340: ("{x}, {y}: {x}D{y} + {x}D{y}", [1, 4], [])` | 3.95 µs | 4.02 µs | +1.8% |
| `case 350: ("{x}, {y}: {x}D{y} + {x}D{y}", [1, 12], [])` | 59.22 µs | 60.30 µs | +1.8% |
| `case 360: ("{x}, {y}: {x}D{y} - {x}D{y}", [1, 2], [])` | 912.7 ns | 920.4 ns | +0.8% |
| `case 370: ("{x}, {y}: {x}D{y} - {x}D{y}", [1, 8], [])` | 20.90 µs | 20.88 µs | -0.1% |
| `case 380: ("{x}, {y}: {x}D{y} * {x}D{y}", [1, 0], [])` | 257.4 ns | 266.9 ns | +3.7% |
| `case 390: ("{x}, {y}: {x}D{y} * {x}D{y}", [3, 4], [])` | 1.12 ms | 1.17 ms | +4.7% |
| `case 400: ("{x}, {y}: {x}D{y} * {x}D{y}", [1, 20], [])` | 241.06 µs | 243.44 µs | +1.0% |
| `case 40: ("{x}: {x} - -1", [1], [])` | 65.6 ns | 71.9 ns | +9.6% |
| `case 410: ("{x}, {y}: {x}D{y} / {x}D{y}", [3, 2], [])` | 17.49 µs | 18.02 µs | +3.0% |
| `case 420: ("{x}, {y}: {x}D{y} / {x}D{y}", [1, 10], [])` | 36.50 µs | 36.84 µs | +0.9% |
| `case 430: ("{x}, {y}: {x}D{y} % {x}D{y}", [1, -1], [])` | 261.5 ns | 268.0 ns | +2.5% |
| `case 440: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 6], [])` | 407.58 µs | 407.13 µs | -0.1% |
| `case 450: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [0, 1], [])` | 153.4 ns | 153.8 ns | +0.3% |
| `case 460: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 4], [])` | 67.90 µs | 68.59 µs | +1.0% |
| `case 470: ("{x}, {y}: -{x}D{y}", [-1, 1], [])` | 104.4 ns | 106.7 ns | +2.2% |
| `case 480: ("{x}, {y}: -{x}D{y}", [4, 4], [])` | 59.42 µs | 60.13 µs | +1.2% |
| `case 490: ("{x}, {y}: -{x}D{y}", [2, 10], [])` | 31.17 µs | 31.14 µs | -0.1% |
| `case 500: ("{x}, {y}: -{x}D{y}", [1, 100], [])` | 58.25 µs | 58.72 µs | +0.8% |
| `case 50: ("{x}: {x} * 0", [2147483647], [])` | 60.6 ns | 64.3 ns | +6.1% |
| `case 510: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2, 1], [])` | 4.07 µs | 4.10 µs | +0.7% |
| `case 520: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 8, 1], [])` | 17.60 µs | 17.68 µs | +0.5% |
| `case 530: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 12, 1], [])` | 7.62 ms | 7.58 ms | -0.4% |
| `case 540: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [1, -1, 2], [])` | 145.3 ns | 146.4 ns | +0.8% |
| `case 550: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [1, 6, 2], [])` | 1.21 µs | 1.22 µs | +1.3% |
| `case 60: ("{x}: {x} * -1", [2147483647], [])` | 67.4 ns | 67.8 ns | +0.7% |
| `case 70: ("{x}: {x} / 0", [2147483647], [])` | 61.2 ns | 62.8 ns | +2.6% |
| `case 80: ("{x}: {x} / -1", [2147483647], [])` | 67.4 ns | 68.3 ns | +1.5% |
| `case 90: ("{x}: {x} % 0", [2147483647], [])` | 60.7 ns | 64.4 ns | +6.1% |

### parallel histograms

| Case | Before | After | Δ |
|---|---:|---:|---:|
| `case 0: ("0", [], [])` | 59.72 µs | 65.42 µs | +9.5% |
| `case 100: ("{x}: {x} % -1", [2147483647], [])` | 60.79 µs | 65.01 µs | +7.0% |
| `case 10: ("{x}", [], [("x", -1)])` | 65.68 µs | 64.52 µs | -1.8% |
| `case 110: ("{x}: 0 ^ {x}", [2147483647], [])` | 62.43 µs | 66.08 µs | +5.9% |
| `case 120: ("{x}: 1 ^ {x}", [-2], [])` | 62.52 µs | 59.65 µs | -4.6% |
| `case 130: ("{x}: (-1) ^ {x}", [-1], [])` | 66.50 µs | 59.55 µs | -10.5% |
| `case 140: ("{x}: {x} ^ -1", [2147483647], [])` | 64.83 µs | 59.58 µs | -8.1% |
| `case 150: ("{x}: {x} ^ 3", [-2147483648], [])` | 66.90 µs | 59.91 µs | -10.4% |
| `case 160: ("{x}: -{x}", [0], [])` | 60.67 µs | 62.70 µs | +3.3% |
| `case 170: ("{x}: [1:{x}]", [10], [])` | 76.78 µs | 74.16 µs | -3.4% |
| `case 180: ("{x}, {y}: [{x}:{y}]", [0, 1], [])` | 68.44 µs | 63.64 µs | -7.0% |
| `case 190: ("{x}, {y}: [{x}:{y}] + 1", [-5, 10], [])` | 91.08 µs | 89.30 µs | -2.0% |
| `case 200: ("{x}, {y}: [{x}:{y}] - [{x}:{y}]", [0, 5], [])` | 107.09 µs | 85.67 µs | -20.0% |
| `case 20: ("{x}: {x} + 1", [-2147483648], [])` | 64.85 µs | 62.69 µs | -3.3% |
| `case 210: ("{x}, {y}: [{x}:{y}] / [{x}:{y}]", [1, 1], [])` | 61.80 µs | 68.78 µs | +11.3% |
| `case 220: ("{x}, {y}, {z}: [{x}:{y}] % {z}", [10, 20, 4], [])` | 76.17 µs | 79.92 µs | +4.9% |
| `case 230: ("{x}, {y}: [{x}:{y}] % [{x}:{y}]", [-2147483648, -2147483645], [])` | 80.11 µs | 73.95 µs | -7.7% |
| `case 240: ("{x}, {y}: -[{x}:{y}]", [0, 1], [])` | 62.67 µs | 61.12 µs | -2.5% |
| `case 250: ("1D0", [], [])` | 59.96 µs | 62.79 µs | +4.7% |
| `case 260: ("1D1", [], [])` | 59.75 µs | 59.45 µs | -0.5% |
| `case 270: ("(-1)D1 drop highest 0", [], [])` | 67.82 µs | 65.19 µs | -3.9% |
| `case 280: ("1D6 drop lowest 2", [], [])` | 61.79 µs | 59.02 µs | -4.5% |
| `case 290: ("3D6 drop highest 0", [], [])` | 170.03 µs | 224.73 µs | +32.2% |
| `case 300: ("{x}D{y}", [], [("x", -1), ("y", -1)])` | 65.09 µs | 58.68 µs | -9.9% |
| `case 30: ("{x}: {x} - 0", [1], [])` | 64.08 µs | 67.33 µs | +5.1% |
| `case 310: ("{x}D{y}", [], [("x", 2), ("y", 6)])` | 109.83 µs | 94.38 µs | -14.1% |
| `case 320: ("{x}D{y}", [], [("x", 4), ("y", 10)])` | 752.50 µs | 1.10 ms | +45.6% |
| `case 330: ("{x}, {y}: {x}D{y} + {x}D{y}", [0, 0], [])` | 68.14 µs | 67.46 µs | -1.0% |
| `case 340: ("{x}, {y}: {x}D{y} + {x}D{y}", [1, 4], [])` | 76.99 µs | 78.49 µs | +1.9% |
| `case 350: ("{x}, {y}: {x}D{y} + {x}D{y}", [1, 12], [])` | 180.55 µs | 211.79 µs | +17.3% |
| `case 360: ("{x}, {y}: {x}D{y} - {x}D{y}", [1, 2], [])` | 69.47 µs | 65.37 µs | -5.9% |
| `case 370: ("{x}, {y}: {x}D{y} - {x}D{y}", [1, 8], [])` | 132.05 µs | 149.59 µs | +13.3% |
| `case 380: ("{x}, {y}: {x}D{y} * {x}D{y}", [1, 0], [])` | 58.68 µs | 60.80 µs | +3.6% |
| `case 390: ("{x}, {y}: {x}D{y} * {x}D{y}", [3, 4], [])` | 362.95 µs | 739.96 µs | +103.9% |
| `case 400: ("{x}, {y}: {x}D{y} * {x}D{y}", [1, 20], [])` | 384.40 µs | 375.00 µs | -2.4% |
| `case 40: ("{x}: {x} - -1", [1], [])` | 63.91 µs | 59.44 µs | -7.0% |
| `case 410: ("{x}, {y}: {x}D{y} / {x}D{y}", [3, 2], [])` | 137.87 µs | 136.77 µs | -0.8% |
| `case 420: ("{x}, {y}: {x}D{y} / {x}D{y}", [1, 10], [])` | 157.27 µs | 183.68 µs | +16.8% |
| `case 430: ("{x}, {y}: {x}D{y} % {x}D{y}", [1, -1], [])` | 65.65 µs | 67.48 µs | +2.8% |
| `case 440: ("{x}, {y}: {x}D{y} % {x}D{y}", [2, 6], [])` | 266.92 µs | 450.83 µs | +68.9% |
| `case 450: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [0, 1], [])` | 60.43 µs | 65.78 µs | +8.9% |
| `case 460: ("{x}, {y}: {x}D{y} ^ {x}D{y}", [2, 4], [])` | 166.53 µs | 225.24 µs | +35.3% |
| `case 470: ("{x}, {y}: -{x}D{y}", [-1, 1], [])` | 60.78 µs | 61.19 µs | +0.7% |
| `case 480: ("{x}, {y}: -{x}D{y}", [4, 4], [])` | 156.99 µs | 216.74 µs | +38.1% |
| `case 490: ("{x}, {y}: -{x}D{y}", [2, 10], [])` | 144.93 µs | 169.89 µs | +17.2% |
| `case 500: ("{x}, {y}: -{x}D{y}", [1, 100], [])` | 213.34 µs | 218.76 µs | +2.5% |
| `case 50: ("{x}: {x} * 0", [2147483647], [])` | 60.28 µs | 67.72 µs | +12.3% |
| `case 510: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 2, 1], [])` | 80.06 µs | 72.75 µs | -9.1% |
| `case 520: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 8, 1], [])` | 116.39 µs | 140.69 µs | +20.9% |
| `case 530: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [4, 12, 1], [])` | 1.28 ms | 1.65 ms | +28.8% |
| `case 540: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [1, -1, 2], [])` | 60.67 µs | 65.54 µs | +8.0% |
| `case 550: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [1, 6, 2], [])` | 73.59 µs | 66.92 µs | -9.1% |
| `case 560: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [3, 10, 2], [])` | 276.22 µs | 405.30 µs | +46.7% |
| `case 570: ("{x}, {y}, {z}: {x}D{y} drop lowest {z}", [2, 100, 2], [])` | 2.67 ms | 1.89 ms | -29.2% |
| `case 580: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 2, 1], [])` | 76.81 µs | 78.24 µs | +1.9% |
| `case 590: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 8, 1], [])` | 120.58 µs | 141.07 µs | +17.0% |
| `case 600: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [4, 12, 1], [])` | 1.26 ms | 1.64 ms | +29.4% |
| `case 60: ("{x}: {x} * -1", [2147483647], [])` | 60.35 µs | 66.58 µs | +10.3% |
| `case 610: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [1, -1, 2], [])` | 65.76 µs | 65.30 µs | -0.7% |
| `case 620: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [1, 6, 2], [])` | 67.49 µs | 66.00 µs | -2.2% |
| `case 630: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [3, 10, 2], [])` | 274.35 µs | 403.47 µs | +47.1% |
| `case 640: ("{x}, {y}, {z}: {x}D{y} drop highest {z}", [2, 100, 2], [])` | 2.62 ms | 1.89 ms | -27.8% |
| `case 650: ("1D[-1, 0, 1] drop highest 1", [], [])` | 64.78 µs | 65.57 µs | +1.2% |
| `case 660: ("2D[-1, 0, 1] drop highest 1", [], [])` | 70.72 µs | 74.41 µs | +5.2% |
| `case 670: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [0, 2], [])` | 64.86 µs | 59.36 µs | -8.5% |
| `case 680: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [2, 2], [])` | 86.86 µs | 83.88 µs | -3.4% |
| `case 690: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [4, 2], [])` | 208.03 µs | 321.89 µs | +54.7% |
| `case 700: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop lowest {y}", [6, 2], [])` | 1.03 ms | 1.41 ms | +37.3% |
| `case 70: ("{x}: {x} / 0", [2147483647], [])` | 65.60 µs | 59.18 µs | -9.8% |
| `case 710: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [1, 2], [])` | 69.12 µs | 72.50 µs | +4.9% |
| `case 720: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [3, 2], [])` | 153.24 µs | 188.95 µs | +23.3% |
| `case 730: ("{x}, {y}: {x}D[-1, 0, 1, 3, 5] drop highest {y}", [5, 2], [])` | 363.24 µs | 662.11 µs | +82.3% |
| `case 80: ("{x}: {x} / -1", [2147483647], [])` | 60.12 µs | 59.42 µs | -1.2% |
| `case 810: ("(1D3)D(1D3) + (1D3)D[-1, 0, 1]", [], [])` | 387.16 µs | 684.93 µs | +76.9% |
| `case 90: ("{x}: {x} % 0", [2147483647], [])` | 65.74 µs | 59.30 µs | -9.8% |
