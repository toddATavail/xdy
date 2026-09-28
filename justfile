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
# release build.
stress *args:
    cargo test --workspace {{args}} -- --ignored

# Run doc-tests only.
doc-test *args:
    cargo test --workspace --doc {{args}}

# Run benchmarks.
bench *args:
    cargo bench -p xdy {{args}}

# Full verification: fmt check, clippy, tests, doc links.
verify:
    @just fmt-check
    @just clippy
    @just test
    @just doc-check

# Build documentation.
doc *args:
    cargo doc --workspace --no-deps {{args}}

# Check documentation for broken intra-doc links and other rustdoc warnings.
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
