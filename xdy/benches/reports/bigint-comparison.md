# Big-Integer Comparison: num-bigint vs dashu for `Weight`

Benchmark: building the exact distribution of each case of the `big weights`
group in `benches/benchmarks.rs`, whose totals exceed `u128::MAX`, so that
`Weight` promotes to its arbitrary-precision representation.

- **Platform**: macOS 26.6.2, Apple M5 Max (aarch64), on a quiet machine
- **Rust**: 1.98.1 (stable), release profile
- **Crates**: num-bigint 0.5.1 (plan of record), dashu-int 0.6.1 (trial)
- **Date**: 2026-10-01

## Summary

| Metric | Value |
|--------|-------|
| Cases | 12, totals of 174 to 959 bits |
| Geometric mean, dashu / num-bigint (interleaved) | 0.93 |
| Geometric mean, dashu / num-bigint (Criterion) | 0.95 |
| Faster with dashu (>3%) | 7 |
| Slower with dashu (>3%) | 3 |
| Largest gain | `50D100`, 22–25% faster |
| Largest loss | `(1D100)D6`, 25–28% slower |
| Share of a build in the allocator, either crate | ~60% (`200D6`), before cutting the allocations; 5% after |
| Results | Identical weights, probabilities and means |

This summary describes the first comparison, before the allocations of big
weights were cut; the rerun afterward is reported below. dashu is
somewhat faster on balance, but not uniformly, and the difference is small
beside the time that both spend allocating. **Recommendation: keep
num-bigint**, cut the allocations of big weights first, and rerun this
comparison afterward, when arithmetic is a larger share of the work.

## Method

`weight.rs` confines its big integers to a private module, `big`, with one
implementation per crate; the trial selects dashu with a `dashu` feature. The
trial lives on the local branch `bigint-trial`, not on the main line. Only
the big representation differs: weights that fit in a `u128` take the same
path either way, and `Weight` is 32 bytes either way, so the small-weight
histograms were not rerun.

Correctness, under dashu: every xdy test passes, the histogram corpus and
random programs agree with the Lean oracle (`just oracle -F dashu`), and the
twelve cases build byte-identical distributions under both crates, down to
each outcome's probability as an `f64` and the mean.

Timing, measured twice:

1. **Criterion**: the `big weights` group with num-bigint, saved as a
   baseline, then with dashu against it, then with num-bigint again to gauge
   drift: within 3.5% for nine cases, but 19–30% slower for the first three,
   whose Criterion comparisons are the least certain.
2. **Interleaved**: ten rounds alternating the crates, in alternating order,
   each the median of 15 builds of every case, by the `big_weights` example
   on the trial branch, which times `DistributionPlan::build` alone. Every
   round ranked the cases alike.

An earlier attempt, on a machine loaded by other builds (load average 11–17),
swung from 0.5× to 1.9× between rounds and was discarded.

## Results

Times are medians per build: across rounds for the interleaved runs, and
Criterion's point estimates.

| Case | num-bigint | dashu | Ratio | Range over rounds | Criterion num-bigint | Criterion dashu | Ratio |
|------|-----------:|------:|------:|------------------:|---------------------:|----------------:|------:|
| `100D6` | 2.32 ms | 1.80 ms | 0.77 | 0.72–0.84 | 2.06 ms | 1.74 ms | 0.85 |
| `200D6` | 12.92 ms | 13.61 ms | 1.05 | 0.99–1.10 | 12.38 ms | 13.23 ms | 1.07 |
| `60D20` | 10.96 ms | 8.53 ms | 0.78 | 0.72–0.80 | 10.43 ms | 8.43 ms | 0.81 |
| `50D100` | 270.86 ms | 211.38 ms | 0.78 | 0.70–0.80 | 274.75 ms | 206.08 ms | 0.75 |
| `100D6 drop lowest 10` | 21.60 ms | 17.33 ms | 0.80 | 0.76–0.83 | 21.30 ms | 17.38 ms | 0.82 |
| `(1D100)D6` | 3.53 ms | 4.41 ms | 1.25 | 1.18–1.29 | 3.44 ms | 4.41 ms | 1.28 |
| `(2D20)D(3D6)` | 19.62 ms | 19.53 ms | 1.00 | 0.92–1.01 | 19.69 ms | 19.71 ms | 1.00 |
| `(3D6)D(3D6) drop lowest 3` | 31.04 ms | 29.21 ms | 0.94 | 0.92–0.96 | 30.68 ms | 28.85 ms | 0.94 |
| `(1D20)D(1D20) drop lowest 1` | 91.84 ms | 86.79 ms | 0.95 | 0.92–0.98 | 91.25 ms | 87.36 ms | 0.96 |
| `(3D6)D(1D20) + (2D10)D(1D12)` | 9.93 ms | 10.33 ms | 1.04 | 1.00–1.09 | 9.91 ms | 10.67 ms | 1.08 |
| `{x}@(5D20) + {x}D6` | 49.62 ms | 45.00 ms | 0.91 | 0.83–0.94 | 50.07 ms | 46.32 ms | 0.92 |
| `{x}@(1D6)D(1D20) + {x} * 2` | 652 µs | 634 µs | 0.97 | 0.90–1.01 | 643 µs | 634 µs | 0.98 |

dashu wins most where long convolution powers multiply and add weights of a
few hundred bits (`100D6`, `60D20`, `50D100`, and the order statistics of
`100D6 drop lowest 10`). It loses on `(1D100)D6`, whose mixture of a hundred
setups adds weights that straddle 128 bits, where its profile shows the
additions building their results from fresh buffers (`add_large`,
`Repr::from_buffer`) and dropping them; and slightly on `200D6`, whose
weights reach 512 bits.

## Where the time goes

Sampling `200D6` with `sample(1)` attributes, of the samples at the top of the
stack:

| Work | num-bigint | dashu |
|------|-----------:|------:|
| Allocation, freeing and zeroing | 59% | 58% |
| Big-integer arithmetic | 19% | 28% |
| `Weight`'s own arithmetic, inlined | 14% | 8% |
| Copying and the rest | 8% | 6% |

Before the allocations were cut, most of a big-weight build was spent
allocating and freeing the integers' storage, under either crate: every
promotion, every `&Weight op &Weight`, which cloned its left operand, and every
remainder of the gcd allocated. That was the larger opportunity, and the
crate's to take whichever big-integer crate it used.

## Footprint

dashu adds five crates to the dependency graph: dashu-int, dashu-base,
num-order, num-modular, and static_assertions. Swapping would drop num-bigint
and num-integer from the library, though the `Weight` tests would keep
num-bigint as their independent model. Both crates compile in seconds; build
time does not decide between them.

## Recommendation

Keep num-bigint for now:

- dashu's advantage is a geometric mean of 5–7%, with a 25% loss on one
  realistic case, against five more dependencies.
- Allocation dominates both, so reducing it (follow-up work) should gain more
  than either crate, and will change the balance between them.
- The `big weights` group stays, and the trial branch keeps the backend
  switch, so the comparison can be rerun cheaply after that work.

## After cutting the allocations

Two changes followed the recommendation, on the same quiet machine and by the
same interleaved method, eight rounds of each:

1. **Borrowed products**: `&Weight * &Weight` multiplies its operands
   directly, rather than copying the left one and multiplying the copy, so
   that a product of big weights allocates once, not twice.
2. **Kronecker substitution**: a dense convolution whose products may exceed
   `u128` packs each operand into one integer, a slot of whole 32-bit digits
   per outcome, wide enough that no sum carries into the next, multiplies the
   two once, and unpacks the sums (`convolve_packed`, in
   `propagation/record.rs`). One multiplication of big integers replaces one
   for every pair of outcomes, and num-bigint multiplies big operands by
   Karatsuba and Toom-3 rather than by schoolbook.

Each speedup is against the original code in the same interleaved runs, so the
first change's runs had their own baseline.

| Case | Before | Borrowed products | Speedup | Both | Speedup |
|------|-------:|------------------:|--------:|-----:|--------:|
| `100D6` | 2.28 ms | 2.16 ms | 1.04× | 391 µs | 5.8× |
| `200D6` | 13.21 ms | 10.51 ms | 1.22× | 2.31 ms | 5.7× |
| `60D20` | 10.85 ms | 10.41 ms | 1.06× | 1.43 ms | 7.6× |
| `50D100` | 269.20 ms | 265.18 ms | 1.03× | 15.32 ms | 17.6× |
| `100D6 drop lowest 10` | 21.43 ms | 21.15 ms | 1.05× | 20.63 ms | 1.0× |
| `(1D100)D6` | 3.52 ms | 3.59 ms | 0.99× | 2.72 ms | 1.3× |
| `(2D20)D(3D6)` | 19.45 ms | 18.99 ms | 1.06× | 16.90 ms | 1.2× |
| `(3D6)D(3D6) drop lowest 3` | 30.57 ms | 28.64 ms | 1.10× | 27.79 ms | 1.1× |
| `(1D20)D(1D20) drop lowest 1` | 90.61 ms | 85.14 ms | 1.10× | 82.85 ms | 1.1× |
| `(3D6)D(1D20) + (2D10)D(1D12)` | 9.82 ms | 8.62 ms | 1.17× | 6.71 ms | 1.5× |
| `{x}@(5D20) + {x}D6` | 49.34 ms | 47.31 ms | 1.07× | 13.86 ms | 3.6× |
| `{x}@(1D6)D(1D20) + {x} * 2` | 647 µs | 597 µs | 1.11× | 586 µs | 1.1× |
| **Geometric mean** | | | 1.08× | | 2.45× |

The order statistics (`drop`) and the mixtures' sums do not convolve big
operands, so they gain little. Small weights never take the packed path, and a
dozen ordinary expressions, from `3D6` to `(2D6)D(1D8)`, took 0.91 times as
long by the geometric mean, none slower. The big weights build identical
distributions, and the property tests hold the packed convolution to the
pairwise one.

Sampling `200D6` again, allocation fell from 59% to 5% of the samples, and
num-bigint's multiplication rose to 90%: the arithmetic is now the work, so the
crate's multiplication matters more. The trial backend, rebased onto both
changes, answers:

| Case | num-bigint | dashu | Ratio |
|------|-----------:|------:|------:|
| `100D6` | 412 µs | 307 µs | 0.74 |
| `200D6` | 2.40 ms | 1.90 ms | 0.79 |
| `60D20` | 1.48 ms | 1.50 ms | 1.01 |
| `50D100` | 16.02 ms | 9.95 ms | 0.62 |
| `100D6 drop lowest 10` | 21.56 ms | 17.53 ms | 0.81 |
| `(1D100)D6` | 2.80 ms | 3.44 ms | 1.23 |
| `(2D20)D(3D6)` | 17.39 ms | 17.02 ms | 0.98 |
| `(3D6)D(3D6) drop lowest 3` | 28.75 ms | 28.89 ms | 1.00 |
| `(1D20)D(1D20) drop lowest 1` | 85.70 ms | 86.31 ms | 1.01 |
| `(3D6)D(1D20) + (2D10)D(1D12)` | 6.96 ms | 6.66 ms | 0.96 |
| `{x}@(5D20) + {x}D6` | 14.39 ms | 12.65 ms | 0.88 |
| `{x}@(1D6)D(1D20) + {x} * 2` | 603 µs | 629 µs | 1.04 |
| **Geometric mean** | | | 0.91 |

dashu now leads by more where convolutions are largest (`50D100`, 1.6× as fast),
presumably by multiplying the packed operands faster, but still trails on
`(1D100)D6` by 23%, and the absolute differences are a few milliseconds at most
beside the up to 18× that the two changes gained. The recommendation stands:
keep num-bigint.
