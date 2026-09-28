//! # Internal parser mechanism test cases
//!
//! Herein are the test cases for internal parser mechanisms that are not
//! externally-facing combinators: token classification for error reporting
//! and identifier and bare word trailing-whitespace handling.

use crate::parser::*;
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                         Internal mechanism tests.                          //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that [`token`] correctly identifies the next token for error
/// reporting.
#[test]
fn test_token()
{
	// Empty input — no token (EOF).
	assert_eq!(token(Span::new("")).map(|s| *s.fragment()), None);

	// Integer constant.
	assert_eq!(token(Span::new("42")).map(|s| *s.fragment()), Some("42"));

	// Negative constant.
	assert_eq!(token(Span::new("-5")).map(|s| *s.fragment()), Some("-5"));

	// Bare word.
	assert_eq!(
		token(Span::new("hello")).map(|s| *s.fragment()),
		Some("hello")
	);

	// A bare word stops at an operator, though an identifier would not.
	assert_eq!(
		token(Span::new("hello + 1")).map(|s| *s.fragment()),
		Some("hello")
	);

	// Single punctuation character.
	assert_eq!(token(Span::new("+")).map(|s| *s.fragment()), Some("+"));
	assert_eq!(token(Span::new("*")).map(|s| *s.fragment()), Some("*"));
	assert_eq!(token(Span::new("(")).map(|s| *s.fragment()), Some("("));
	assert_eq!(token(Span::new(")")).map(|s| *s.fragment()), Some(")"));
	assert_eq!(token(Span::new("{")).map(|s| *s.fragment()), Some("{"));
	assert_eq!(token(Span::new("}")).map(|s| *s.fragment()), Some("}"));
	assert_eq!(token(Span::new("[")).map(|s| *s.fragment()), Some("["));
	assert_eq!(token(Span::new("]")).map(|s| *s.fragment()), Some("]"));
	assert_eq!(token(Span::new("@")).map(|s| *s.fragment()), Some("@"));
}

/// Ensure that [`identifier`] correctly handles trailing spaces, returning the
/// trimmed identifier and giving back the spaces to the remaining input.
#[test]
fn test_identifier_trailing_whitespace()
{
	// Trailing spaces should be trimmed from the identifier and returned as
	// part of the remaining input.
	let span = Span::new("hello }");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "hello");
	assert_eq!(*residue.fragment(), " }");

	// Multiple trailing spaces — the exact number must be preserved.
	let span = Span::new("foo   }");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "foo");
	assert_eq!(*residue.fragment(), "   }");
	assert_eq!(residue.location_offset(), 3, "Residue offset after 'foo'");

	// Two trailing spaces — distinguishes subtraction from division in the
	// trail calculation.
	let span = Span::new("ab  }");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "ab");
	assert_eq!(*residue.fragment(), "  }");
	assert_eq!(residue.location_offset(), 2, "Residue offset after 'ab'");

	// A tab is not an identifier character, so it ends the identifier.
	let span = Span::new("bar\t}");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "bar");
	assert_eq!(*residue.fragment(), "\t}");

	// Operators and delimiters other than braces are identifier characters.
	let span = Span::new("a + (b) }");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "a + (b)");
	assert_eq!(*residue.fragment(), " }");

	// No trailing spaces — residue should be what follows immediately.
	let span = Span::new("baz}");
	let (residue, result) = identifier(span).unwrap();
	assert_eq!(*result.fragment(), "baz");
	assert_eq!(*residue.fragment(), "}");
}

/// Ensure that [`bare_word`] correctly handles trailing whitespace, returning
/// the trimmed bare word and giving back the whitespace to the remaining
/// input.
#[test]
fn test_bare_word_trailing_whitespace()
{
	// Trailing whitespace should be trimmed from the bare word and returned as
	// part of the remaining input.
	let span = Span::new("hello )");
	let (residue, result) = bare_word(span).unwrap();
	assert_eq!(*result.fragment(), "hello");
	assert_eq!(*residue.fragment(), " )");

	// Multiple trailing spaces — the exact number must be preserved.
	let span = Span::new("foo   +");
	let (residue, result) = bare_word(span).unwrap();
	assert_eq!(*result.fragment(), "foo");
	assert_eq!(*residue.fragment(), "   +");
	assert_eq!(residue.location_offset(), 3, "Residue offset after 'foo'");

	// Tab as trailing whitespace.
	let span = Span::new("bar\t}");
	let (residue, result) = bare_word(span).unwrap();
	assert_eq!(*result.fragment(), "bar");
	assert_eq!(*residue.fragment(), "\t}");

	// Inline whitespace within the bare word is kept.
	let span = Span::new("an external variable + 1");
	let (residue, result) = bare_word(span).unwrap();
	assert_eq!(*result.fragment(), "an external variable");
	assert_eq!(*residue.fragment(), " + 1");

	// No trailing whitespace — residue should be what follows immediately.
	let span = Span::new("baz)");
	let (residue, result) = bare_word(span).unwrap();
	assert_eq!(*result.fragment(), "baz");
	assert_eq!(*residue.fragment(), ")");
}
