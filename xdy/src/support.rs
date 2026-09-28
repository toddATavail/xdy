//! # Testing and benchmarking support
//!
//! This module provides support for testing and benchmarking.

use std::num::IntErrorKind;
#[cfg(test)]
use std::{
	env,
	fmt::{self, Display, Formatter},
	io::Read,
	panic,
	process::{Command, ExitStatus, Stdio},
	thread::{self, JoinHandle},
	time::{Duration, Instant}
};

use crate::{
	Function, Optimizer as _, Passes, StandardOptimizer, compile_unoptimized
};

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
/// Do not optimize the function, so that tests and benchmarks can apply
/// exactly the passes that they exercise.
///
/// # Parameters
/// - `source`: The source code to compile.
///
/// # Returns
/// The compiled, unoptimized function.
pub fn compile_valid(source: &str) -> Function
{
	match compile_unoptimized(source)
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

#[cfg(test)]
/// An expected suggestion in an error test case.
#[derive(Debug)]
pub struct ExpectedSuggestion
{
	/// The corrected source string.
	pub corrected_source: &'static str,

	/// The expected placeholders.
	pub placeholders: Vec<ExpectedPlaceholder>
}

#[cfg(test)]
/// An expected related label attached to a diagnostic.
#[derive(Debug)]
pub struct ExpectedRelated
{
	/// The byte range in the original source.
	pub span: (usize, usize),

	/// The expected message.
	pub message: &'static str
}

#[cfg(test)]
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

	/// The expected full [`Display`](std::fmt::Display) rendering of the
	/// diagnostic, as produced by `format!("{}", diagnostic)`. This is an
	/// end-to-end check on the `Display` assembly of `DiagnosticKind`,
	/// `SourceSpan`, and the message, distinct from the component-wise
	/// assertions on `kind`/`span`/`message`. Keeping it alongside the
	/// decomposed fields means a regression in any of the three `Display`
	/// impls — or in the assembly itself — is caught with a human-readable
	/// diff.
	pub rendered: &'static str,

	/// The expected related labels, in the order they should appear.
	pub related: Vec<ExpectedRelated>,

	/// The expected suggestions.
	pub suggestions: Vec<ExpectedSuggestion>
}

#[cfg(test)]
/// An error test case.
#[derive(Debug)]
pub struct ErrorTestCase
{
	/// The broken source input.
	pub source: &'static str,

	/// The expected diagnostics.
	pub expected_diagnostics: Vec<ExpectedDiagnostic>
}

#[cfg(test)]
/// Parse error test cases from a test case file. The file is expected to
/// conform to the following grammar:
///
/// ```text
/// file ::= cases? ;
/// cases ::= case ("\n\n" case)* ;
/// case ::= source "\n=\n" diagnostics ;
/// source ::= [^\n]* ;
/// diagnostics ::= diagnostic ("---\n" diagnostic)* ;
/// diagnostic ::= kind span message rendered related* suggestion* ;
/// kind ::= "kind:" IDENTIFIER "\n" ;
/// span ::= "span:" USIZE ".." USIZE "\n" ;
/// message ::= "message:" [^\n]* "\n" ;
/// rendered ::= "rendered:" [^\n]* "\n" ;
/// related ::= "related:" USIZE ".." USIZE '"' [^"]* '"' "\n" ;
/// suggestion ::= "suggestion:" [^\n]* "\n" placeholder* ;
/// placeholder ::= "placeholder:" USIZE ".." USIZE
///     '"' [^"]* '"' "[" kinds "]" "\n" ;
/// kinds ::= IDENTIFIER ("," IDENTIFIER)* ;
/// ```
///
/// The `rendered:` field captures the full
/// [`Display`](std::fmt::Display) output of the diagnostic, as produced by
/// `format!("{}", diag)`; the harness asserts this directly against the
/// `diagnose()` output. All `related:` lines must appear before the first
/// `suggestion:` line.
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

#[cfg(test)]
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
	let mut rendered = None;
	let mut related: Vec<ExpectedRelated> = Vec::new();
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
		else if let Some(rest) = line.strip_prefix("rendered:")
		{
			rendered = Some(rest.trim());
		}
		else if let Some(rest) = line.strip_prefix("related:")
		{
			assert!(
				current_suggestion.is_none(),
				"`related:` line must appear before any `suggestion:` line: {:?}",
				line
			);
			related.push(parse_expected_related(rest.trim()));
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
		rendered: rendered.expect("missing rendered"),
		related,
		suggestions
	}
}

#[cfg(test)]
/// Parse a related-label specification from its text representation.
///
/// Expected format: `start..end "message"`
///
/// # Parameters
/// - `text`: The related-label text.
///
/// # Returns
/// The parsed expected related label.
///
/// # Panics
/// If the text is malformed.
fn parse_expected_related(text: &'static str) -> ExpectedRelated
{
	// Parse span: "start..end"
	let span_end = text.find(' ').unwrap();
	let span_parts: Vec<&str> = text[..span_end].split("..").collect();
	let span = (
		span_parts[0].parse::<usize>().unwrap(),
		span_parts[1].parse::<usize>().unwrap()
	);

	// Parse message: "..."
	let rest = text[span_end..].trim();
	let msg_start = rest.find('"').unwrap() + 1;
	let msg_end = rest[msg_start..].rfind('"').unwrap() + msg_start;
	let message = &rest[msg_start..msg_end];

	ExpectedRelated { span, message }
}

#[cfg(test)]
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

////////////////////////////////////////////////////////////////////////////////
//                            Small-stack support.                            //
////////////////////////////////////////////////////////////////////////////////

/// The stack size, in bytes, of the thread on which [`on_small_stack`] runs its
/// closure: 2 MiB, the default stack size of a spawned Rust thread. Code that
/// passes on a stack this small does not lean on the larger stack of a typical
/// main thread.
#[cfg(test)]
pub const SMALL_STACK_SIZE: usize = 2 * 1024 * 1024;

/// The default time budget of [`on_small_stack`]. It is a safety net for the
/// test suite, not a limit on `xDy`: it turns a hang into a failure.
#[cfg(test)]
pub const SMALL_STACK_TIMEOUT: Duration = Duration::from_secs(60);

/// The environment variable that marks a child process of
/// [`try_on_small_stack`]. Its value is the name of the test that the child
/// runs.
#[cfg(test)]
const SMALL_STACK_CHILD: &str = "XDY_SMALL_STACK_CHILD";

/// The line that a child process prints to its standard output after its
/// closure returns normally. Its absence means that the child never ran the
/// closure, e.g., because the test filter matched nothing.
#[cfg(test)]
const SMALL_STACK_MARKER: &str = "xdy-small-stack: closure returned";

/// The ways that [`try_on_small_stack`] can fail. Each variant carries the
/// transcript of the child process's standard output and standard error.
#[cfg(test)]
#[derive(Debug)]
pub enum SmallStackFailure
{
	/// The closure panicked, e.g., because an assertion failed.
	Panicked
	{
		/// The child's output.
		transcript: String
	},

	/// The closure overflowed the small stack.
	Overflowed
	{
		/// The child's output.
		transcript: String
	},

	/// The child did not finish within the time budget, so it was killed.
	TimedOut
	{
		/// The time budget.
		timeout: Duration,

		/// The child's output.
		transcript: String
	},

	/// The child terminated in some other abnormal way.
	Abnormal
	{
		/// The child's exit status.
		status: ExitStatus,

		/// The child's output.
		transcript: String
	},

	/// The child exited successfully without running the closure.
	NotRun
	{
		/// The child's output.
		transcript: String
	}
}

#[cfg(test)]
impl Display for SmallStackFailure
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		let transcript = match self
		{
			SmallStackFailure::Panicked { transcript } =>
			{
				writeln!(f, "the closure panicked on the small stack")?;
				transcript
			},
			SmallStackFailure::Overflowed { transcript } =>
			{
				writeln!(
					f,
					"the closure overflowed the small stack ({} bytes)",
					SMALL_STACK_SIZE
				)?;
				transcript
			},
			SmallStackFailure::TimedOut {
				timeout,
				transcript
			} =>
			{
				writeln!(
					f,
					"the closure did not finish within {:?}: either it runs in \
					 super-linear time, or it overflowed the stack on a host \
					 where an overflow hangs instead of aborting",
					timeout
				)?;
				transcript
			},
			SmallStackFailure::Abnormal { status, transcript } =>
			{
				writeln!(f, "the child terminated abnormally: {}", status)?;
				transcript
			},
			SmallStackFailure::NotRun { transcript } =>
			{
				writeln!(
					f,
					"the child exited successfully without running the \
					 closure; did the test filter match nothing?"
				)?;
				transcript
			}
		};
		write!(f, "{}", transcript)
	}
}

/// Run the specified closure on a [small stack](SMALL_STACK_SIZE) in a child
/// process, within the [default time budget](SMALL_STACK_TIMEOUT).
///
/// # Parameters
/// - `f`: The closure to run.
///
/// # Panics
/// If the closure panics, overflows the stack, or does not finish in time. See
/// [`try_on_small_stack`] for details.
///
/// # Examples
/// ```ignore
/// #[test]
/// fn test_deep_group()
/// {
///     let source = format!("{}1{}", "(".repeat(100_000), ")".repeat(100_000));
///     on_small_stack(move || assert!(Parser::parse(&source).is_ok()));
/// }
/// ```
#[cfg(test)]
pub fn on_small_stack<F>(f: F)
where
	F: FnOnce() + Send
{
	on_small_stack_within(SMALL_STACK_TIMEOUT, f)
}

/// Run the specified closure on a [small stack](SMALL_STACK_SIZE) in a child
/// process, within the specified time budget.
///
/// # Parameters
/// - `timeout`: The time budget.
/// - `f`: The closure to run.
///
/// # Panics
/// If the closure panics, overflows the stack, or does not finish in time. See
/// [`try_on_small_stack`] for details.
#[cfg(test)]
pub fn on_small_stack_within<F>(timeout: Duration, f: F)
where
	F: FnOnce() + Send
{
	if let Err(failure) = try_on_small_stack(timeout, f)
	{
		panic!("{}", failure)
	}
}

/// Run the specified closure on a [small stack](SMALL_STACK_SIZE) in a child
/// process, within the specified time budget, and report how it fared.
///
/// A stack overflow cannot be caught: it aborts the whole process on Linux and
/// Windows, and on macOS it hangs forever if an ancestor process installed a
/// buggy Mach exception handler. So the closure runs in a child process, and
/// only the child dies or hangs. The child is the current test binary,
/// restricted to the calling test. It recognizes itself by an environment
/// variable, runs the closure on a thread with a small stack, and prints a
/// marker if the closure returns. The parent classifies the child's fate.
///
/// ```mermaid
/// sequenceDiagram
///     participant P as Parent test
///     participant C as Child test binary
///     participant T as Small-stack thread
///     P->>C: spawn(current_exe, --exact <test>, XDY_SMALL_STACK_CHILD)
///     C->>T: spawn with SMALL_STACK_SIZE
///     T->>T: run the closure
///     alt the closure returns
///         T-->>C: Ok
///         C-->>P: print the marker, exit 0
///     else the closure panics
///         T-->>C: Err(payload)
///         C-->>P: resume the panic, exit 101
///     else the closure overflows
///         T-->>P: abort, "has overflowed its stack"
///     else the time budget expires
///         P->>C: kill
///     end
///     P->>P: classify the exit status and the transcript
/// ```
///
/// # Parameters
/// - `timeout`: The time budget.
/// - `f`: The closure to run. It runs only in the child.
///
/// # Returns
/// `Ok(())` if the closure returned normally.
///
/// # Errors
/// * [`Panicked`](SmallStackFailure::Panicked) if the closure panicked.
/// * [`Overflowed`](SmallStackFailure::Overflowed) if the closure overflowed
///   the stack.
/// * [`TimedOut`](SmallStackFailure::TimedOut) if the child did not finish
///   within the time budget.
/// * [`Abnormal`](SmallStackFailure::Abnormal) if the child terminated in some
///   other abnormal way.
/// * [`NotRun`](SmallStackFailure::NotRun) if the child never ran the closure.
///
/// # Panics
/// If the caller is not a `libtest` test thread, whose name is the path of the
/// test, or if the child process cannot be spawned.
///
/// # Notes
/// In the child, this function returns only if the closure returned normally.
#[cfg(test)]
#[cfg_attr(doc, aquamarine::aquamarine)]
pub fn try_on_small_stack<F>(
	timeout: Duration,
	f: F
) -> Result<(), SmallStackFailure>
where
	F: FnOnce() + Send
{
	let test = thread::current()
		.name()
		.filter(|name| *name != "main")
		.expect("the caller must be a libtest test thread")
		.to_owned();
	match env::var(SMALL_STACK_CHILD)
	{
		Ok(child) if child == test =>
		{
			run_small_stack_child(f);
			Ok(())
		},
		Ok(child) => panic!(
			"small-stack child for `{}` reached `{}` instead",
			child, test
		),
		Err(_) => run_small_stack_parent(&test, timeout)
	}
}

/// Run the specified closure on a [small stack](SMALL_STACK_SIZE), as the
/// child process of [`try_on_small_stack`]. Print the
/// [marker](SMALL_STACK_MARKER) if the closure returns normally, and resume
/// its panic otherwise.
///
/// # Parameters
/// - `f`: The closure to run.
///
/// # Panics
/// If the closure panics, or if the thread cannot be spawned.
#[cfg(test)]
fn run_small_stack_child<F>(f: F)
where
	F: FnOnce() + Send
{
	let outcome = thread::scope(|scope| {
		thread::Builder::new()
			.name("xdy-small-stack".to_owned())
			.stack_size(SMALL_STACK_SIZE)
			.spawn_scoped(scope, f)
			.expect("failed to spawn the small-stack thread")
			.join()
	});
	match outcome
	{
		Ok(()) => println!("{}", SMALL_STACK_MARKER),
		Err(payload) => panic::resume_unwind(payload)
	}
}

/// Run the specified test in a child process, as the parent side of
/// [`try_on_small_stack`], and classify the child's fate.
///
/// # Parameters
/// - `test`: The path of the test.
/// - `timeout`: The time budget.
///
/// # Returns
/// `Ok(())` if the child's closure returned normally.
///
/// # Errors
/// See [`try_on_small_stack`].
///
/// # Panics
/// If the child process cannot be spawned or awaited.
#[cfg(test)]
fn run_small_stack_parent(
	test: &str,
	timeout: Duration
) -> Result<(), SmallStackFailure>
{
	let executable =
		env::current_exe().expect("failed to locate the test executable");
	let mut child = Command::new(executable)
		.args([
			test,
			"--exact",
			"--nocapture",
			"--include-ignored",
			"--test-threads=1"
		])
		.env(SMALL_STACK_CHILD, test)
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.expect("failed to spawn the small-stack child");
	let stdout = drain(child.stdout.take().expect("stdout is piped"));
	let stderr = drain(child.stderr.take().expect("stderr is piped"));
	let deadline = Instant::now() + timeout;
	let status = loop
	{
		if let Some(status) = child
			.try_wait()
			.expect("failed to await the small-stack child")
		{
			break Some(status)
		}
		if Instant::now() >= deadline
		{
			// The child may exit on its own before the kill lands, so ignore
			// the error; the wait reaps it either way.
			let _ = child.kill();
			let _ = child.wait();
			break None
		}
		thread::sleep(Duration::from_millis(10));
	};
	let stdout = stdout.join().expect("the stdout reader panicked");
	let stderr = stderr.join().expect("the stderr reader panicked");
	let transcript = format!(
		"--- child stdout ---\n{}\n--- child stderr ---\n{}",
		stdout, stderr
	);
	match status
	{
		None => Err(SmallStackFailure::TimedOut {
			timeout,
			transcript
		}),
		Some(status) if status.success() =>
		{
			// libtest's progress line, e.g., "test name ... ", has no newline
			// yet when the child prints the marker, so search rather than
			// match whole lines.
			if stdout.contains(SMALL_STACK_MARKER)
			{
				Ok(())
			}
			else
			{
				Err(SmallStackFailure::NotRun { transcript })
			}
		},
		Some(_) if stderr.contains("has overflowed its stack") =>
		{
			Err(SmallStackFailure::Overflowed { transcript })
		},
		// 101 is the exit code of a test binary whose test failed.
		Some(status) if status.code() == Some(101) =>
		{
			Err(SmallStackFailure::Panicked { transcript })
		},
		Some(status) => Err(SmallStackFailure::Abnormal { status, transcript })
	}
}

/// Read the specified pipe to its end on a new thread, so that a chatty child
/// cannot block on a full pipe.
///
/// # Parameters
/// - `pipe`: The pipe to read.
///
/// # Returns
/// A handle whose result is the pipe's contents, decoded lossily as UTF-8.
#[cfg(test)]
fn drain(mut pipe: impl Read + Send + 'static) -> JoinHandle<String>
{
	thread::spawn(move || {
		let mut bytes = Vec::new();
		// A read error just truncates the transcript, which is diagnostic.
		let _ = pipe.read_to_end(&mut bytes);
		String::from_utf8_lossy(&bytes).into_owned()
	})
}
