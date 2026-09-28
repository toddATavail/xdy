//! # Atom combinator test cases
//!
//! Herein are the test cases for the atom-level combinators: [`constant`],
//! [`d_operator`], [`identifier`], [`alpha`], and [`alphanumeric1`], and the
//! identifier character set.

use std::borrow::Cow;

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                           Atom combinator tests.                           //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`constant`] behaves as expected.
#[test]
fn test_constant()
{
	// Happy paths.
	for (input, expected_str, expected_ast) in [
		(
			"0",
			"0",
			Constant {
				value: 0,
				span: SourceSpan::default()
			}
		),
		(
			"42",
			"42",
			Constant {
				value: 42,
				span: SourceSpan::default()
			}
		),
		(
			"-42",
			"-42",
			Constant {
				value: -42,
				span: SourceSpan::default()
			}
		),
		(
			"9999",
			"9999",
			Constant {
				value: 9999,
				span: SourceSpan::default()
			}
		),
		(
			"-9999",
			"-9999",
			Constant {
				value: -9999,
				span: SourceSpan::default()
			}
		),
		(
			"2147483647",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		), // i32::MAX
		(
			"-2147483648",
			"-2147483648",
			Constant {
				value: -2147483648,
				span: SourceSpan::default()
			}
		), // i32::MIN
		(
			"2147483648",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		), // Saturates
		(
			"-2147483649",
			"-2147483648",
			Constant {
				value: -2147483648,
				span: SourceSpan::default()
			}
		), // Saturates
		(
			"123456789123456789123456789123456789",
			"2147483647",
			Constant {
				value: 2147483647,
				span: SourceSpan::default()
			}
		) // massive overflow saturates
	]
	{
		let span = Span::new(input);
		match constant(span)
		{
			Ok((residue, result)) =>
			{
				assert!(
					residue.is_empty(),
					"Residue not empty for input: {}",
					input
				);
				assert_eq!(
					result.to_string(),
					expected_str,
					"Failed for input: {}",
					input
				);
				assert_eq!(
					result.untethered(),
					expected_ast.untethered(),
					"AST mismatch for input: {}",
					input
				);
			},
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", " 42", "a", "+42", "--42", "a42", "42a"]
	{
		let span = Span::new(input);
		let result = constant(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`d_operator`] behaves as expected.
#[test]
fn test_d_operator()
{
	// Happy paths.
	for input in ["d", "D"]
	{
		let span = Span::new(input);
		match d_operator(span)
		{
			Ok(result) => assert_eq!(
				result.1,
				input.chars().next().unwrap(),
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}

	// Invalid inputs.
	for input in ["", " ", "a", "3", " d", " D"]
	{
		let span = Span::new(input);
		let result = d_operator(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
	}
}

/// Ensure that [`identifier`] behaves as expected.
#[test]
fn test_identifier()
{
	for (input, expected) in [
		("hello", "hello"),
		("hello123", "hello123"),
		("hello_world", "hello_world"),
		("hello-world", "hello-world"),
		("_", "_"),
		("αβγ", "αβγ"),
		("a", "a"),
		("hello world", "hello world"),
		("an external variable", "an external variable"),
		// Interior whitespace is exactly as written.
		("a  b", "a  b"),
		("a\tb", "a\tb"),
		("a\n   b", "a\n   b"),
		("a\u{00A0}b", "a\u{00A0}b"),
		("a\u{2028}b", "a\u{2028}b"),
		// Trailing whitespace is not part of the identifier.
		("a b ", "a b"),
		("a b\t\n", "a b"),
		("a\u{00A0}", "a"),
		// Any visible character, other than a brace.
		("123hello", "123hello"),
		("-1", "-1"),
		("0", "0"),
		("weapon: 2/3", "weapon: 2/3"),
		("hello@world", "hello@world"),
		("f(x), g[y]", "f(x), g[y]"),
		("1d6 drop lowest", "1d6 drop lowest"),
		("$env|weapon", "$env|weapon"),
		("|selector", "|selector"),
		("日本語", "日本語"),
		("🎲", "🎲"),
		// Joiners join visible characters.
		("👨\u{200D}👩\u{200D}👧", "👨\u{200D}👩\u{200D}👧"),
		("می\u{200C}خواهم", "می\u{200C}خواهم"),
		// An excluded character ends the identifier.
		("a}", "a"),
		("a{", "a"),
		("a\u{0}b", "a"),
		("a\u{1B}b", "a"),
		("a\u{200B}b", "a"),
		("a\u{202E}b", "a")
	]
	{
		let span = Span::new(input);
		match identifier(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {:?}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {:?}: {}", input, e)
		}
	}

	// Test cases that should fail
	for input in [
		"",
		" ",
		" hello",
		"\nhello",
		"\u{00A0}hello",
		"}",
		"{",
		"\t",
		"\u{00A0}",
		"\u{0085}",
		"\u{0}",
		"\u{200B}",
		"\u{FEFF}",
		"\u{2066}"
	]
	{
		let span = Span::new(input);
		assert!(
			identifier(span).is_err(),
			"Failed to reject invalid input: {:?}",
			input
		);
	}
}

/// Ensure that [`is_identifier_char`] classifies characters as the shared
/// source of truth for the identifier grammar, and that [`is_canonical_name`]
/// accepts exactly the canonical identifiers.
#[test]
fn test_identifier_char_classes()
{
	// Letters, digits, sigils, operators, delimiters other than braces,
	// symbols, the joiners, and whitespace of every kind.
	for c in [
		'a', 'Z', 'δ', 'Ⅻ', '0', '²', '_', '$', '#', '\'', '-', '.', '|', '?',
		'!', '~', '@', ',', ':', '/', '(', ')', '[', ']', '+', '*', '^', '%',
		'×', '÷', '🎲', '\u{200C}', '\u{200D}', '\u{FE0F}',
		// Whitespace, including the whitespace control characters.
		' ', '\t', '\n', '\r', '\u{0B}', '\u{0C}', '\u{85}', '\u{A0}',
		'\u{1680}', '\u{2000}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}',
		'\u{205F}', '\u{3000}'
	]
	{
		assert!(
			is_identifier_char(c),
			"expected {:?} to be an identifier character",
			c
		);
	}

	for c in [
		// The braces.
		'{', '}', // Control characters (Cc) other than whitespace.
		'\0', '\u{1B}', '\u{7F}', '\u{9F}',
		// Bidirectional formatting controls.
		'\u{061C}', '\u{200E}', '\u{200F}', '\u{202A}', '\u{202B}', '\u{202C}',
		'\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}',
		// Other invisible characters.
		'\u{00AD}', '\u{115F}', '\u{1160}', '\u{180E}', '\u{200B}', '\u{2060}',
		'\u{2061}', '\u{2062}', '\u{2063}', '\u{2064}', '\u{3164}', '\u{FEFF}',
		'\u{FFA0}'
	]
	{
		assert!(
			!is_identifier_char(c),
			"expected {:?} to be excluded from identifiers",
			c
		);
	}

	for name in [
		"a",
		"a b",
		"a b c",
		"weapon: 2/3",
		"-1",
		"🎲",
		"👨\u{200D}👩"
	]
	{
		assert!(is_canonical_name(name), "Failed to accept {:?}", name);
	}
	for name in [
		"",
		" ",
		" a",
		"a ",
		"a  b",
		"a\tb",
		"a\nb",
		"a\u{00A0}b",
		"a}",
		"{x",
		"a\u{0}b",
		"\u{200B}"
	]
	{
		assert!(!is_canonical_name(name), "Failed to reject {:?}", name);
	}
}

/// Ensure that [`canonical_name`] trims a name and collapses every run of
/// whitespace within it to a single space, borrowing the name whenever it is
/// already canonical.
#[test]
fn test_canonical_name()
{
	for (name, expected, borrowed) in [
		("a", "a", true),
		("a b", "a b", true),
		("a b c", "a b c", true),
		("weapon: 2/3", "weapon: 2/3", true),
		("a  b", "a b", false),
		("a\tb", "a b", false),
		("a\n   b", "a b", false),
		("a\r\nb c", "a b c", false),
		("a\u{00A0}b", "a b", false),
		("a \u{2028} b", "a b", false),
		(" a ", "a", false)
	]
	{
		let canonical = canonical_name(name);
		assert_eq!(canonical, expected, "Failed for {:?}", name);
		assert_eq!(
			matches!(canonical, Cow::Borrowed(_)),
			borrowed,
			"Wrong ownership for {:?}",
			name
		);
		assert!(is_canonical_name(&canonical), "Failed for {:?}", name);
	}
}

/// Ensure that [`alpha`] behaves as expected.
#[test]
fn test_alpha()
{
	let span = Span::new("h");
	assert_eq!(alpha(span).unwrap().1.fragment(), &"h");

	let span = Span::new("1");
	assert!(alpha(span).is_err());

	let span = Span::new("");
	assert!(alpha(span).is_err());

	let span = Span::new("_");
	assert!(alpha(span).is_err());

	for (input, expected) in [
		("H", "H"),
		("a", "a"),
		("é", "é"),
		("こ", "こ"),
		("П", "П"),
		("Γ", "Γ")
	]
	{
		let span = Span::new(input);
		match alpha(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}
}

/// Ensure that [`alphanumeric1`] behaves as expected.
#[test]
fn test_alphanumeric1()
{
	let span = Span::new("hello123");
	assert_eq!(alphanumeric1(span).unwrap().1.fragment(), &"hello123");

	let span = Span::new("123hello");
	assert_eq!(alphanumeric1(span).unwrap().1.fragment(), &"123hello");

	let span = Span::new("");
	assert!(alphanumeric1(span).is_err());

	let span = Span::new("_hello");
	assert!(alphanumeric1(span).is_err());

	for (input, expected) in [
		("Hello", "Hello"),
		("hElLo", "hElLo"),
		("héllo", "héllo"),
		("こんにちは", "こんにちは"),
		("Привет", "Привет"),
		("Γειά σου", "Γειά"),
		("hello world", "hello"),
		("1234567890", "1234567890"),
		("1234567890hello", "1234567890hello"),
		("hello1234567890", "hello1234567890"),
		("hello1234567890world", "hello1234567890world")
	]
	{
		let span = Span::new(input);
		match alphanumeric1(span)
		{
			Ok(result) => assert_eq!(
				result.1.fragment(),
				&expected,
				"Failed for input: {}",
				input
			),
			Err(e) => panic!("Parsing failed for input: {}: {}", input, e)
		}
	}
}
