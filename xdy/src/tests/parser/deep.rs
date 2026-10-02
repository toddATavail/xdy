//! # Deep parser test cases
//!
//! Herein are the tests that hold the parser to its promise of constant stack
//! depth and linear time. Every [nesting construct](Nesting) is parsed at its
//! [depth](Nesting::depth) on a [small stack](on_small_stack), both as a valid
//! source and as a source that fails at its innermost level. The default time
//! budget of the small stack also bounds the parse time: the recursive parser
//! took time exponential in the nesting of groups and bindings, so it would
//! not finish. The deep parses are ignored by default; `just stress` runs them.
//!
//! A failing parse builds a [`ParseError`] whose list keeps only the errors at
//! the rightmost position, and one beyond them, so its length does not grow
//! with the depth of the input. The [`Synthetic`](NomErrorKind::Synthetic)
//! errors nest no deeper than the grammar allows, regardless of the input, so
//! the derived implementations of [`Clone`], [`PartialEq`], [`Hash`], and
//! [`Drop`] recurse only to that bound.

use std::{
	collections::hash_map::DefaultHasher,
	hash::{Hash, Hasher}
};

use crate::{
	Parser,
	parser::{NomErrorKind, ParseError},
	support::on_small_stack,
	tests::ast::{LEAF, Nesting, nest_parsable}
};

////////////////////////////////////////////////////////////////////////////////
//                                Deep parses.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a deep chain of groups parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_group() { parse_deep(Nesting::Group) }

/// Ensure that a deep chain of bindings parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_binding() { parse_deep(Nesting::Binding) }

/// Ensure that a deep chain of range starts parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_range_start() { parse_deep(Nesting::RangeStart) }

/// Ensure that a deep chain of range ends parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_range_end() { parse_deep(Nesting::RangeEnd) }

/// Ensure that a deep chain of negations parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_negation() { parse_deep(Nesting::Negation) }

/// Ensure that a deep chain of exponents parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_exponent() { parse_deep(Nesting::Exponent) }

/// Ensure that a deep chain of additions parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_addition() { parse_deep(Nesting::Addition) }

/// Ensure that a deep chain of subtractions parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_subtraction() { parse_deep(Nesting::Subtraction) }

/// Ensure that a deep chain of multiplications parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_multiplication() { parse_deep(Nesting::Multiplication) }

/// Ensure that a deep chain of divisions parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_division() { parse_deep(Nesting::Division) }

/// Ensure that a deep chain of modulos parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_modulo() { parse_deep(Nesting::Modulo) }

/// Ensure that a deep chain of dice counts parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_dice_count() { parse_deep(Nesting::DiceCount) }

/// Ensure that a deep chain of dice faces parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_dice_faces() { parse_deep(Nesting::DiceFaces) }

/// Ensure that a deep chain of custom dice counts parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_custom_count() { parse_deep(Nesting::CustomCount) }

/// Ensure that a deep chain of drop expressions parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_drop_expression() { parse_deep(Nesting::DropExpression) }

/// Ensure that a deep stack of drop clauses parses on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_drop_clauses() { parse_deep(Nesting::DropClauses) }

/// Ensure that a deep mixture of every nesting construct parses on a small
/// stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_parse_deep_mixed() { parse_deep(Nesting::Mixed) }

/// Parse a [deep](Nesting::depth) source on a small stack, and ensure that the
/// result renders as the source. Then replace its innermost constant with a
/// stray `@`, and ensure that parsing fails there without trouble: the error
/// keeps at most one entry beyond those at its position; it renders, clones,
/// compares equal to its clone, and hashes alike; and its
/// [`Synthetic`](NomErrorKind::Synthetic) errors nest no deeper than
/// [`SYNTHETIC_NESTING_BOUND`]. Uses `assert!` rather than `assert_eq!` on
/// sources and errors, since a failure would otherwise try to render them in
/// full.
///
/// # Parameters
/// - `nesting`: The way to nest.
fn parse_deep(nesting: Nesting)
{
	on_small_stack(|| {
		let source = nest_parsable(nesting, nesting.depth(), LEAF).to_string();
		let function = Parser::parse(&source).expect("deep source failed");
		assert!(function.parameters.is_none(), "{:?}", nesting);
		assert!(
			function.body.to_string() == source,
			"{:?}: parsed expression renders differently from source",
			nesting
		);
		drop(function);

		let leaf = LEAF.to_string();
		assert_eq!(source.matches(&leaf).count(), 1, "{:?}", nesting);
		let failing = source.replacen(&leaf, "@", 1);
		let error = Parser::parse(&failing).expect_err("failing source parsed");
		let offset = source.find(&leaf).unwrap();
		assert_eq!(
			error.errors[0].0.location_offset(),
			offset,
			"{:?}",
			nesting
		);
		assert!(
			synthetic_nesting(&error) <= SYNTHETIC_NESTING_BOUND,
			"{:?}: synthetic errors nest too deeply",
			nesting
		);
		let leading = error
			.errors
			.iter()
			.take_while(|(span, _)| span.location_offset() == offset)
			.count();
		assert!(
			error.errors.len() <= leading + 1,
			"{:?}: error keeps {} entries, of which {} are leading",
			nesting,
			error.errors.len(),
			leading
		);
		assert!(error.to_string().starts_with("Parse error @ 1:"));
		let copy = error.clone();
		assert!(copy == error, "{:?}: clone compared unequal", nesting);
		assert_eq!(hash_of(&copy), hash_of(&error), "{:?}", nesting);
	});
}

////////////////////////////////////////////////////////////////////////////////
//                            Synthetic errors.                               //
////////////////////////////////////////////////////////////////////////////////

/// The greatest nesting of [`Synthetic`](NomErrorKind::Synthetic) errors in any
/// [`ParseError`]. A synthetic error merges the errors of alternatives that
/// failed at the same position, so synthetic errors nest only where
/// alternatives nest without consuming input between them, which the grammar
/// bounds. The bound was measured, not derived: it is the greatest nesting
/// found in 400,000 random failing inputs of up to 24 tokens, and it did not
/// grow with their length.
const SYNTHETIC_NESTING_BOUND: usize = 11;

/// Ensure that [`Synthetic`](NomErrorKind::Synthetic) errors nest no deeper
/// than [`SYNTHETIC_NESTING_BOUND`], however deep the input: for every nesting
/// construct at every depth up to 40, neither any prefix of the source nor the
/// source with its innermost constant replaced by a stray `@` produces deeper
/// nesting. Every error must also clone, compare equal to its clone, and hash
/// alike, which the derived implementations can do only if the nesting is
/// bounded.
#[test]
fn test_synthetic_nesting_is_bounded()
{
	let leaf = LEAF.to_string();
	for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
	{
		for depth in 1..=40
		{
			let source = nest_parsable(nesting, depth, LEAF).to_string();
			let failing = source.replacen(&leaf, "@", 1);
			for input in source
				.char_indices()
				.map(|(i, _)| &source[..i])
				.chain([failing.as_str()])
			{
				if let Err(error) = Parser::parse(input)
				{
					let copy = error.clone();
					assert!(
						copy == error,
						"clone compared unequal: {:?}",
						input
					);
					assert_eq!(hash_of(&copy), hash_of(&error), "{:?}", input);
					assert!(
						synthetic_nesting(&error) <= SYNTHETIC_NESTING_BOUND,
						"{:?} at depth {}: synthetic errors nest too deeply \
						 for {:?}",
						nesting,
						depth,
						input
					);
				}
			}
		}
	}
}

/// Answer the greatest nesting of [`Synthetic`](NomErrorKind::Synthetic)
/// errors within a parse error, without recursion.
///
/// # Parameters
/// - `error`: The parse error.
///
/// # Returns
/// The greatest nesting, which is `0` if there are no synthetic errors.
fn synthetic_nesting(error: &ParseError<'_>) -> usize
{
	let mut greatest = 0;
	let mut pending = vec![(error, 0)];
	while let Some((error, depth)) = pending.pop()
	{
		for (_, kind) in &error.errors
		{
			if let NomErrorKind::Synthetic(merged) = kind
			{
				greatest = greatest.max(depth + 1);
				pending.extend(merged.iter().map(|error| (error, depth + 1)));
			}
		}
	}
	greatest
}

/// Answer the hash of a value.
///
/// # Parameters
/// - `value`: The value.
///
/// # Returns
/// The hash.
fn hash_of<T: Hash>(value: &T) -> u64
{
	let mut hasher = DefaultHasher::new();
	value.hash(&mut hasher);
	hasher.finish()
}
