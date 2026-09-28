//! # Corpus of sources
//!
//! Herein is a corpus of sources for differential tests, mostly malformed. It
//! comprises every source in the test files, whether or not it parses; every
//! prefix of each, every deletion of one character from each, and every
//! insertion of one [token](TOKENS) into each; shallow textual nests of every
//! [nesting construct](Nesting), and their prefixes; and a fixed sample of
//! random strings of tokens.

use std::{collections::BTreeSet, sync::OnceLock};

use crate::{
	support::{
		read_compilation_test_cases, read_error_test_cases,
		read_evaluation_test_cases, read_histogram_test_cases
	},
	tests::ast::{Nesting, nest}
};

////////////////////////////////////////////////////////////////////////////////
//                                  Corpus.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The tokens that the corpus inserts into sources, and from which it builds
/// random strings. The braced names draw on the whole identifier set, e.g.,
/// spaces, `:`, `/`, non-ASCII characters, and characters that would be syntax
/// outside of braces, and one holds a bidi control, which the identifier set
/// excludes. The whitespace, inserted into the braced names of the sources,
/// forms runs that [canonicalize](crate::parser::canonical_name) away.
pub(super) const TOKENS: &[&str] = &[
	"(",
	")",
	"[",
	"]",
	"{",
	"}",
	":",
	",",
	"@",
	"-",
	"+",
	"*",
	"/",
	"%",
	"^",
	"×",
	"÷",
	"d",
	"D",
	" ",
	"\t",
	"\n",
	"\u{A0}",
	"x",
	"{x}",
	"{hit points}",
	"{weapon: 2/3}",
	"{1d6 drop lowest}",
	"{Ω-(1)}",
	"{a\u{202E}b}",
	"{a}@(",
	"{ a\t\u{A0}b }@(",
	"1",
	"6",
	"[1,2]",
	"drop",
	"lowest",
	"highest"
];

/// The number of random strings of [tokens](TOKENS) in the corpus.
const RANDOM_INPUTS: usize = 20_000;

/// The greatest number of [tokens](TOKENS) in a random string.
const MAX_RANDOM_TOKENS: usize = 24;

/// The greatest depth of the textual nests in the corpus.
const MAX_NEST_DEPTH: usize = 6;

/// Answer the corpus, building it on first use.
///
/// # Returns
/// The corpus, in lexicographic order.
pub(super) fn corpus() -> &'static [String]
{
	static CORPUS: OnceLock<Vec<String>> = OnceLock::new();
	CORPUS.get_or_init(|| {
		let mut corpus = BTreeSet::new();
		let mut seeds = sources();
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			for depth in 0..=MAX_NEST_DEPTH
			{
				let source = nest(nesting, depth, 1).to_string();
				corpus.extend(prefixes(&source));
				seeds.insert(source);
			}
		}
		for source in &seeds
		{
			corpus.insert(source.clone());
			corpus.extend(prefixes(source));
			corpus.extend(deletions(source));
			corpus.extend(insertions(source, TOKENS));
		}
		corpus.extend(random_inputs());
		let corpus = corpus.into_iter().collect::<Vec<_>>();
		assert!(corpus.len() > 500_000, "corpus shrank to {}", corpus.len());
		corpus
	})
}

/// Answer every distinct source in the test files, whether or not it parses.
///
/// # Returns
/// The sources.
fn sources() -> BTreeSet<String>
{
	let mut sources = BTreeSet::new();
	for text in [
		include_str!("../../tests/test_ast_debug.txt"),
		include_str!("../../tests/test_ast_debug_pretty.txt"),
		include_str!("../../tests/test_common_subexpression_elimination.txt"),
		include_str!("../../tests/test_compile_unoptimized.txt"),
		include_str!("../../tests/test_constant_commuting.txt"),
		include_str!("../../tests/test_constant_folding.txt"),
		include_str!("../../tests/test_full_optimization.txt"),
		include_str!("../../tests/test_parse.txt"),
		include_str!("../../tests/test_register_coalescence.txt"),
		include_str!("../../tests/test_strength_reduction.txt")
	]
	{
		sources.extend(
			read_compilation_test_cases(text)
				.into_iter()
				.map(|(source, _)| source.to_owned())
		);
	}
	sources.extend(
		read_evaluation_test_cases(include_str!(
			"../../tests/test_evaluation.txt"
		))
		.into_iter()
		.map(|case| case.0.to_owned())
	);
	sources.extend(
		read_histogram_test_cases(include_str!(
			"../../tests/test_histograms.txt"
		))
		.into_iter()
		.map(|case| case.0.to_owned())
	);
	sources.extend(
		read_error_test_cases(include_str!(
			"../../tests/test_parser_errors.txt"
		))
		.into_iter()
		.map(|case| case.source.to_owned())
	);
	assert!(sources.len() > 500, "sources shrank to {}", sources.len());
	sources
}

/// Answer every proper prefix of a source that ends on a character boundary.
///
/// # Parameters
/// - `source`: The source.
///
/// # Returns
/// The prefixes.
fn prefixes(source: &str) -> impl Iterator<Item = String> + '_
{
	source.char_indices().map(|(i, _)| source[..i].to_owned())
}

/// Answer every string obtained by deleting one character from a source.
///
/// # Parameters
/// - `source`: The source.
///
/// # Returns
/// The strings.
fn deletions(source: &str) -> impl Iterator<Item = String> + '_
{
	source.char_indices().map(|(i, c)| {
		let mut deleted = source.to_owned();
		deleted.replace_range(i..i + c.len_utf8(), "");
		deleted
	})
}

/// Answer every string obtained by inserting one token into a source, at any
/// character boundary.
///
/// # Parameters
/// - `source`: The source.
/// - `tokens`: The tokens to insert, e.g., [`TOKENS`].
///
/// # Returns
/// The strings.
fn insertions<'a>(
	source: &'a str,
	tokens: &'a [&'a str]
) -> impl Iterator<Item = String> + 'a
{
	source
		.char_indices()
		.map(|(i, _)| i)
		.chain([source.len()])
		.flat_map(move |i| {
			tokens.iter().map(move |token| {
				let mut inserted = source.to_owned();
				inserted.insert_str(i, token);
				inserted
			})
		})
}

/// Answer a fixed sample of random strings of [tokens](TOKENS). The generator
/// is a xorshift with a fixed seed, so the sample is the same on every run.
///
/// # Returns
/// The strings.
fn random_inputs() -> impl Iterator<Item = String>
{
	let mut state = 0x1234_5678_9abc_def0_u64;
	let mut next = move || {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		state as usize
	};
	(0..RANDOM_INPUTS).map(move |_| {
		let tokens = next() % MAX_RANDOM_TOKENS + 1;
		(0..tokens)
			.map(|_| TOKENS[next() % TOKENS.len()])
			.collect::<String>()
	})
}
