# xdy workspace build recipes.
#
# Requires: just (https://github.com/casey/just), cargo, cargo +nightly (for
# fmt).

# Default recipe: verify everything.
default: verify

# Run clippy on the entire workspace.
clippy *args:
    cargo clippy --workspace --all-targets {{args}}

# Format with nightly rustfmt (required for unstable options in rustfmt.toml).
fmt *args:
    cargo +nightly fmt {{args}}

# Check formatting without modifying files.
fmt-check:
    cargo +nightly fmt -- --check

# Run all workspace tests.
test *args:
    cargo test --workspace {{args}}

# Run the stress tests, which are ignored by default: deep nesting, which needs
# gigabytes of memory, and wall-clock linearity. Pass --release to stress the
# release build. Skips the oracle tests, which need Lean; see `oracle`.
stress *args:
    cargo test --workspace {{args}} -- --ignored --skip oracle

# Run doc-tests only.
doc-test *args:
    cargo test --workspace --doc {{args}}

# Run benchmarks, which requires the `bench` feature.
bench *args:
    cargo bench -p xdy -F bench {{args}}

# Full verification: fmt check, clippy, tests, doc links.
verify:
    @just fmt-check
    @just clippy
    @just test
    @just doc-check

# Build the Lean oracle and run its tests, which are checked as they build.
# Requires elan (https://github.com/leanprover/elan); not part of `verify`.
[working-directory: 'lean']
lean:
    lake build
    lake test

# Build the Lean specification, which needs Mathlib, and run its tests. Fetches
# Mathlib's prebuilt files first, several gigabytes the first time. Requires
# elan (https://github.com/leanprover/elan); not part of `verify` or `lean`.
[working-directory: 'lean/spec']
spec:
    lake exe cache get
    lake build
    lake test

# Check the distribution corpus and random programs, some of them resummed,
# against the Lean oracle, which it builds first, check that the oracle dies
# with the test process, and write the corpus, corrected by the oracle, to
# target/oracle/test_distributions.txt. Requires elan
# (https://github.com/leanprover/elan); not part of `verify`.
oracle *args: lean
    cargo test -p xdy --lib {{args}} oracle -- --ignored --nocapture

# Regenerate the railroad diagrams of the grammars, xdy/doc/*.svg, from their
# EBNF sources, xdy/grammar/*.ebnf. A test fails while they disagree.
railroad:
    XDY_REGENERATE_RAILROAD=1 cargo test -p xdy --lib tests::railroad

# Build documentation.
doc *args:
    cargo doc --workspace --no-deps {{args}}

# Check documentation for broken intra-doc links and other rustdoc warnings.
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
