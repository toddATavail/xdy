# xDy: Tree-sitter

This is the [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) grammar
for the xDy dice expression language. Most of the mainline documentation is in
the [compiler](../xdy) crate, where the compiler, optimizer, evaluator, and
application reside.

## Building

The Rust bindings are checked in, so you can build the [compiler](../xdy)
without doing anything special. The relevant build instructions are
[here](../xdy/README.md), and you can safely ignore the directions hereinafter
unless you want to build bindings for a different language or modify the
grammar.

### Generating the parser

To generate the parser, you need to have the `tree-sitter` command-line tool
installed. There's a Rust version and a Node.js version, and you can find the
pertinent installation instructions
[here](https://tree-sitter.github.io/tree-sitter/creating-parsers).
Once you have the tool installed, you can generate the parser using `npm`:

```shell
$ npm run build
```

Or using `tree-sitter` directly:

```shell
$ tree-sitter generate
$ tree-sitter build
```

## Testing

After the parser has been generated, you can run the tests using `npm`:

```shell
$ tree-sitter test
```

Or you can run the grammar tests and the Rust integration test piecewise:

```shell
$ tree-sitter test
$ cargo test
```
