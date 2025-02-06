//! # Testing and benchmarking support
//!
//! This module provides support for testing and benchmarking.

use std::num::IntErrorKind;

use crate::{Function, Optimizer as _, Passes, StandardOptimizer, compile};

////////////////////////////////////////////////////////////////////////////////
//                            Compilation support.                            //
////////////////////////////////////////////////////////////////////////////////

/// A compilation test case is a pair of a source code string and an expected
/// output string.
pub type CompilationTestCase = (&'static str, &'static str);

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
		Some(function) => function,
		None => panic!("compilation error")
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
