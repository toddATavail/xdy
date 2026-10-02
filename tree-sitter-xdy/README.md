# xDy: Tree-sitter

This is the [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) grammar
for the xDy dice expression language, for editors and other tools that consume
Tree-sitter grammars. The [`xdy`](../xdy) crate does not use it: its parser is
built on [`nom`](https://crates.io/crates/nom), and is the reference for the
language, which this grammar follows. Most of the mainline documentation is in
that crate, where the parser, compiler, optimizer, and evaluator reside.

The grammar produces syntax trees, but ships no queries yet, so it provides no
syntax highlighting or code folding of its own.

## Building

The generated parser (`src/parser.c`) is checked in, so the Rust bindings build
without the `tree-sitter` command-line tool. The Rust bindings are the only
bindings in this repository. You need the tool only to modify the grammar, or
to generate bindings for another language yourself.

### Generating the parser

To generate the parser, you need the `tree-sitter` command-line tool, which
`cargo install tree-sitter-cli` installs; see
[Creating parsers](https://tree-sitter.github.io/tree-sitter/creating-parsers)
for other ways. Then, from this directory:

```shell
$ tree-sitter generate
```

## Testing

Run the grammar's corpus tests, under `test/corpus`, and the Rust integration
test:

```shell
$ tree-sitter test
$ cargo test
```
