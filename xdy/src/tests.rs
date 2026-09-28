//! # Tests
//!
//! Herein are test cases for the compiler. The test cases are stored in
//! files in the `../tests` directory. Each file contains a series of test
//! cases, each of which consists of a source dice expression and an expected
//! print rendition.

mod assembler;
mod ast;
mod bounds;
mod compiler;
mod corpus;
mod diagnostics;
mod evaluator;
mod histogram;
mod optimizer;
mod parser;
mod property;
mod recovery;
mod recovery_parity;
mod small_stack;
mod validator;
mod visitor;
