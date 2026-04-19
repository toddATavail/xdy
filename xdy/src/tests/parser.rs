//! # Parser test cases
//!
//! Herein are the test cases for the various parser functions. The suite is
//! partitioned into submodules by combinator cluster, with the large
//! data-driven, span-metadata, internal, and source-span tests broken out
//! separately to keep each file navigable.

mod combinators_add_sub;
mod combinators_atoms;
mod combinators_dice;
mod combinators_expression;
mod combinators_function;
mod combinators_mul_div_mod;
mod combinators_primary;
mod data_driven;
mod internals;
mod s_expr_format;
mod spans;
