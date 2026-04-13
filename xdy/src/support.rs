//! # Testing and benchmarking support
//!
//! This module provides support for testing and benchmarking.

use std::num::IntErrorKind;

use crate::{Function, Optimizer as _, Passes, StandardOptimizer, compile};

////////////////////////////////////////////////////////////////////////////////
//                            Compilation support.                            //
////////////////////////////////////////////////////////////////////////////////

/// A test case is a pair of a source code string and an expected output
/// string. This type is shared by compilation tests, parse tests, and any
/// other test suite that uses the `{source}\n=\n{expected}` file format.
pub type TestCase = (&'static str, &'static str);

/// A compilation test case.
pub type CompilationTestCase = TestCase;

/// Parse the compilation test cases from a test case file. The file is expected
/// to contain a series of blocks of the form:
///
/// ```text
/// {source}
/// =
/// {expected}
/// ```
///
/// Where `{source}` is the source code to compile and `{expected}` is the
/// expected output of the compiler. Blocks are separated by a blank line.
///
/// # Parameters
/// - `source`: The contents of a test case file.
///
/// # Returns
/// The test cases.
///
/// # Panics
/// If the test case file is incorrectly formatted.
pub fn read_compilation_test_cases(
	source: &'static str
) -> Vec<CompilationTestCase>
{
	let mut test_cases = Vec::new();
	let blocks = source.split("\n\n");
	for block in blocks
	{
		let parts = block.split("\n=\n").collect::<Vec<_>>();
		for part in parts[..].chunks(2)
		{
			let source = part[0].trim();
			let expected = part[1].trim();
			test_cases.push((source, expected));
		}
	}
	test_cases
}

/// Compile the specified valid dice source code into a [function](Function).
///
/// # Parameters
/// - `source`: The source code to compile.
///
/// # Returns
/// The compiled function.
pub fn compile_valid(source: &str) -> Function
{
	match compile(source)
	{
		Ok(function) => function,
		Err(e) => panic!("compilation error: {e}")
	}
}

/// Optimize the specified function using the standard optimizer and the
/// specified passes.
///
/// # Parameters
/// - `function`: The function to optimize.
///
/// # Returns
/// The optimized function.
pub fn optimize(function: Function, passes: Passes) -> Function
{
	StandardOptimizer::new(passes).optimize(function).unwrap()
}

////////////////////////////////////////////////////////////////////////////////
//                            Evaluation support.                             //
////////////////////////////////////////////////////////////////////////////////

/// An evaluation test case is a source code string, a set of arguments, a set
/// of externs, and an expected output string.
pub type EvaluationTestCase = (
	&'static str,
	Vec<i32>,
	Vec<(&'static str, i32)>,
	&'static str
);

/// Parse the evaluation test cases from a test case file. The file is expected
/// to conform to the following grammar:
///
/// ```text
/// file ::= cases? ;
/// cases ::= case ("\n\n" case)* ;
/// case ::= source "\n=\n" args? externs? expected ;
/// source ::= [^\n] "\n" ;
/// args ::= "args:" I32 ("," I32)* "\n" ;
/// externs ::= "externs:" I32 ("," I32)* "\n" ;
/// extern ::= name "="
/// expected ::= min "," max "\n" ;
/// min ::= I32 ;
/// max ::= I32 ;
/// I32 ::= /[0-9]+/ ;
/// WS ::= /[ \t]*/ ;
/// ```
///
/// `WS` is permitted to occur between any two tokens.
///
/// # Parameters
/// - `source`: The contents of a test case file.
///
/// # Returns
/// The test cases.
///
/// # Panics
/// If the test case file is incorrectly formatted.
pub fn read_evaluation_test_cases(
	source: &'static str
) -> Vec<EvaluationTestCase>
{
	let mut test_cases = Vec::new();
	let blocks = source.split("\n\n");
	for block in blocks
	{
		let parts: Vec<&str> = block.splitn(2, "\n=\n").collect();
		let source = parts[0].trim();
		let cases: Vec<&str> = parts[1].trim().split("\n=\n").collect();
		for case in cases
		{
			let mut lines: Vec<&str> = case.lines().collect();
			let expected = lines.pop().expect("missing expected result");
			let expected = expected.trim();
			let mut args = Vec::new();
			let mut externs = Vec::new();
			for line in lines
			{
				let line = line.trim();
				if let Some(line) = line.strip_prefix("args:")
				{
					args = line
						.split(',')
						.map(|s| match s.trim().parse::<i32>()
						{
							Ok(i) => i,
							Err(e)
								if e.kind() == &IntErrorKind::PosOverflow =>
							{
								i32::MAX
							},
							Err(e)
								if e.kind() == &IntErrorKind::NegOverflow =>
							{
								i32::MIN
							},
							_ => unreachable!()
						})
						.collect();
				}
				else if let Some(line) = line.strip_prefix("externs:")
				{
					externs = line
						.split(',')
						.map(|s| {
							let parts: Vec<&str> =
								s.trim().splitn(2, '=').collect();
							(parts[0].trim(), parts[1].trim().parse().unwrap())
						})
						.collect();
				}
			}
			test_cases.push((source, args, externs, expected));
		}
	}
	test_cases
}

////////////////////////////////////////////////////////////////////////////////
//                             Histogram support.                             //
////////////////////////////////////////////////////////////////////////////////

/// A histogram test case is a source code string, a set of arguments, a set of
/// externs, and a histogram of expected outcomes as a vector of pairs of
/// outcome and count.
pub type HistogramTestCase = (
	&'static str,
	Vec<i32>,
	Vec<(&'static str, i32)>,
	Vec<(i32, usize)>
);

/// Parse the histogram test cases from a test case file. The file is expected
/// to conform to the following grammar:
///
/// ```text
/// file ::= cases? ;
/// cases ::= case ("\n\n" case)* ;
/// case ::= source "\n=\n" args? externs? expected ;
/// source ::= [^\n] "\n" ;
/// args ::= "args:" I32 ("," I32)* "\n" ;
/// externs ::= "externs:" I32 ("," I32)* "\n" ;
/// extern ::= name "="
/// expected ::= outcome_pair ("\n" outcome_pair)* "\n";
/// outcome_pair ::= outcome ":" count ;
/// outcome ::= I32 ;
/// count ::= USIZE ;
/// min ::= I32 ;
/// max ::= I32 ;
/// I32 ::= /[0-9]+/ ;
/// WS ::= /[ \t]*/ ;
/// ```
///
/// `WS` is permitted to occur between any two tokens.
///
/// # Parameters
/// - `source`: The contents of a test case file.
///
/// # Returns
/// The test cases.
///
/// # Panics
/// If the test case file is incorrectly formatted.
pub fn read_histogram_test_cases(source: &'static str)
-> Vec<HistogramTestCase>
{
	let mut test_cases = Vec::new();
	let blocks = source.split("\n\n");
	for block in blocks
	{
		let parts: Vec<&str> = block.splitn(2, "\n=\n").collect();
		let source = parts[0].trim();
		let cases: Vec<&str> = parts[1].trim().split("\n=\n").collect();
		for case in cases
		{
			let mut expected = Vec::new();
			let mut args = Vec::new();
			let mut externs = Vec::new();
			for line in case.lines()
			{
				let line = line.trim();
				if let Some(line) = line.strip_prefix("args:")
				{
					args = line
						.split(',')
						.map(|s| match s.trim().parse::<i32>()
						{
							Ok(i) => i,
							Err(e)
								if e.kind() == &IntErrorKind::PosOverflow =>
							{
								i32::MAX
							},
							Err(e)
								if e.kind() == &IntErrorKind::NegOverflow =>
							{
								i32::MIN
							},
							_ => unreachable!()
						})
						.collect();
				}
				else if let Some(line) = line.strip_prefix("externs:")
				{
					externs = line
						.split(',')
						.map(|s| {
							let parts: Vec<&str> =
								s.trim().splitn(2, '=').collect();
							(parts[0].trim(), parts[1].trim().parse().unwrap())
						})
						.collect();
				}
				else
				{
					let parts: Vec<&str> = line.splitn(2, ':').collect();
					let outcome = parts[0].trim().parse::<i32>().unwrap();
					let count = parts[1].trim().parse::<usize>().unwrap();
					expected.push((outcome, count));
				}
			}
			test_cases.push((source, args, externs, expected));
		}
	}
	test_cases
}

////////////////////////////////////////////////////////////////////////////////
//                         Error diagnostic support.                          //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
/// An expected placeholder in an error test case.
#[derive(Debug)]
pub struct ExpectedPlaceholder
{
	/// The byte range in the corrected source.
	pub span: (usize, usize),

	/// The description of the placeholder.
	pub description: &'static str,

	/// The valid kinds for this placeholder.
	pub valid_kinds: Vec<&'static str>
}

/// An expected suggestion in an error test case.
#[derive(Debug)]
pub struct ExpectedSuggestion
{
	/// The corrected source string.
	pub corrected_source: &'static str,

	/// The expected placeholders.
	pub placeholders: Vec<ExpectedPlaceholder>
}

/// An expected diagnostic in an error test case.
#[derive(Debug)]
pub struct ExpectedDiagnostic
{
	/// The diagnostic kind name (e.g., "MissingRightOperand").
	pub kind: &'static str,

	/// The byte range in the original source.
	pub span: (usize, usize),

	/// The expected message.
	pub message: &'static str,

	/// The expected suggestions.
	pub suggestions: Vec<ExpectedSuggestion>
}

/// An error test case.
#[derive(Debug)]
pub struct ErrorTestCase
{
	/// The broken source input.
	pub source: &'static str,

	/// The expected diagnostics.
	pub expected_diagnostics: Vec<ExpectedDiagnostic>
}

/// Parse error test cases from a test case file. The file is expected to
/// conform to the following grammar:
///
/// ```text
/// file ::= cases? ;
/// cases ::= case ("\n\n" case)* ;
/// case ::= source "\n=\n" diagnostics ;
/// source ::= [^\n]* ;
/// diagnostics ::= diagnostic ("---\n" diagnostic)* ;
/// diagnostic ::= kind span message suggestion? ;
/// kind ::= "kind:" IDENTIFIER "\n" ;
/// span ::= "span:" USIZE ".." USIZE "\n" ;
/// message ::= "message:" [^\n]* "\n" ;
/// suggestion ::= "suggestion:" [^\n]* "\n" placeholder* ;
/// placeholder ::= "placeholder:" USIZE ".." USIZE
///     '"' [^"]* '"' "[" kinds "]" "\n" ;
/// kinds ::= IDENTIFIER ("," IDENTIFIER)* ;
/// ```
///
/// # Parameters
/// - `source`: The contents of a test case file.
///
/// # Returns
/// The test cases.
///
/// # Panics
/// If the test case file is incorrectly formatted.
pub fn read_error_test_cases(source: &'static str) -> Vec<ErrorTestCase>
{
	let mut test_cases = Vec::new();
	for block in source.split("\n\n")
	{
		let block = block.trim();
		if block.is_empty()
		{
			continue;
		}
		let parts: Vec<&str> = block.splitn(2, "\n=\n").collect();
		assert!(parts.len() == 2, "malformed test case block: {:?}", block);
		let test_source = match parts[0]
		{
			"<empty>" => "",
			other => other
		};
		let diagnostics_text = parts[1];

		let mut expected_diagnostics = Vec::new();
		for diag_block in diagnostics_text.split("\n---\n")
		{
			expected_diagnostics
				.push(parse_expected_diagnostic(diag_block.trim()));
		}

		test_cases.push(ErrorTestCase {
			source: test_source,
			expected_diagnostics
		});
	}
	test_cases
}

/// Parse a single expected diagnostic from its text representation.
///
/// # Parameters
/// - `text`: The text of one diagnostic section.
///
/// # Returns
/// The parsed expected diagnostic.
///
/// # Panics
/// If the text is malformed.
fn parse_expected_diagnostic(text: &'static str) -> ExpectedDiagnostic
{
	let mut kind = None;
	let mut span = None;
	let mut message = None;
	let mut suggestions: Vec<ExpectedSuggestion> = Vec::new();
	let mut current_suggestion: Option<&'static str> = None;
	let mut current_placeholders: Vec<ExpectedPlaceholder> = Vec::new();

	for line in text.lines()
	{
		let line = line.trim();
		if let Some(rest) = line.strip_prefix("kind:")
		{
			kind = Some(rest.trim());
		}
		else if let Some(rest) = line.strip_prefix("span:")
		{
			let parts: Vec<&str> = rest.trim().split("..").collect();
			span = Some((
				parts[0].parse::<usize>().unwrap(),
				parts[1].parse::<usize>().unwrap()
			));
		}
		else if let Some(rest) = line.strip_prefix("message:")
		{
			message = Some(rest.trim());
		}
		else if let Some(rest) = line.strip_prefix("suggestion:")
		{
			// Flush the previous suggestion, if any.
			if let Some(src) = current_suggestion
			{
				suggestions.push(ExpectedSuggestion {
					corrected_source: src,
					placeholders: std::mem::take(&mut current_placeholders)
				});
			}
			current_suggestion = Some(rest.trim());
		}
		else if let Some(rest) = line.strip_prefix("placeholder:")
		{
			current_placeholders.push(parse_expected_placeholder(rest.trim()));
		}
	}
	// Flush the last suggestion.
	if let Some(src) = current_suggestion
	{
		suggestions.push(ExpectedSuggestion {
			corrected_source: src,
			placeholders: std::mem::take(&mut current_placeholders)
		});
	}

	ExpectedDiagnostic {
		kind: kind.expect("missing kind"),
		span: span.expect("missing span"),
		message: message.expect("missing message"),
		suggestions
	}
}

/// Parse a placeholder specification from its text representation.
///
/// Expected format: `start..end "description" [kind1, kind2, ...]`
///
/// # Parameters
/// - `text`: The placeholder text.
///
/// # Returns
/// The parsed expected placeholder.
///
/// # Panics
/// If the text is malformed.
fn parse_expected_placeholder(text: &'static str) -> ExpectedPlaceholder
{
	// Parse span: "start..end"
	let span_end = text.find(' ').unwrap();
	let span_parts: Vec<&str> = text[..span_end].split("..").collect();
	let span = (
		span_parts[0].parse::<usize>().unwrap(),
		span_parts[1].parse::<usize>().unwrap()
	);

	// Parse description: "..."
	let rest = text[span_end..].trim();
	let desc_start = rest.find('"').unwrap() + 1;
	let desc_end = rest[desc_start..].find('"').unwrap() + desc_start;
	let description = &rest[desc_start..desc_end];

	// Parse valid kinds: [kind1, kind2, ...]
	let kinds_start = rest.find('[').unwrap() + 1;
	let kinds_end = rest.find(']').unwrap();
	let valid_kinds: Vec<&str> = rest[kinds_start..kinds_end]
		.split(',')
		.map(|s| s.trim())
		.collect();

	ExpectedPlaceholder {
		span,
		description,
		valid_kinds
	}
}
