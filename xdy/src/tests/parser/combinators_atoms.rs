//! # Atom combinator test cases
//!
//! Herein are the test cases for the atom-level combinators: [`constant`],
//! [`d_operator`], [`identifier`], [`alpha`], and [`alphanumeric1`].

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
		("_hello", "_hello"),
		("_", "_"),
		("h3llo_w0rld-123_test", "h3llo_w0rld-123_test"),
		("αβγ", "αβγ"),
		("_123", "_123"),
		("a", "a"),
		("hello world", "hello world"),
		("hello.world", "hello.world"),
		("an external variable", "an external variable"),
		("a.b.c", "a.b.c")
	]
	{
		let span = Span::new(input);
		match identifier(span)
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

	// Test cases that should fail
	for input in [
		"123hello",
		"-hello",
		" hello",
		"",
		"!hello world",
		"hello!world",
		"hello@world",
		"x "
	]
	{
		let span = Span::new(input);
		let result = identifier(span);
		assert!(
			result.is_err() || !result.unwrap().0.fragment().is_empty(),
			"Failed to reject invalid input: {}",
			input
		);
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
