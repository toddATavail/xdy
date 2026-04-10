//! Parse errors
//!
//! Herein is error handling support for the parser, including types and
//! utilities that facilitate the creation of user-friendly error messages.

use std::{
	borrow::Cow,
	error::Error,
	fmt::{self, Display, Formatter}
};

use nom::error::{ErrorKind, ParseError as _};

use crate::parser::token;

use super::Span;

////////////////////////////////////////////////////////////////////////////////
//                               Parse errors.                                //
////////////////////////////////////////////////////////////////////////////////

/// A parse error, compatible with `nom`'s requirements. The error type is
/// designed as an improvement over
/// [VerboseError](nom_language::error::VerboseError), at least because it
/// preserves the rightmost _parse error_ of an [`alt`](nom::branch::alt)
/// combinator rather than the one produced by the rightmost _alternative_.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParseError<'a>
{
	/// The errors accumulated during parsing, in reverse order, i.e., the last
	/// error is the outermost error.
	pub errors: Vec<(Span<'a>, NomErrorKind<'a>)>
}

/// The error kind for [`ParseError`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NomErrorKind<'a>
{
	/// The label provided by the [`context`](nom::error::context) combinator,
	/// which is used to understand the context of the failed parse. Each label
	/// encodes a comma-separated list of classifiers, e.g., "constant",
	/// "variable", "identifier", etc., from denote the expectations of the
	/// parser at the corresponding parse position.
	Context(&'static str),

	/// A synthetic error, produced by the [`alt`](nom::branch::alt) combinator
	/// when merging errors at equivalent parse positions. Each element of the
	/// vector represents an aborted parse that failed at the same parse
	/// position.
	Synthetic(Vec<ParseError<'a>>),

	/// The character expected by the [`char`](nom::character::complete::char)
	/// combinator.
	Char(char),

	/// The `nom` [error kind](ErrorKind), which indicates which of the
	/// standard combinators produced the error.
	Nom(ErrorKind)
}

impl NomErrorKind<'_>
{
	/// Compute the parse expectations for the receiver.
	///
	/// # Returns
	/// The requested parse expectations.
	pub fn expectations(&self) -> Vec<Cow<'_, str>>
	{
		let mut expectations = Vec::new();
		match self
		{
			NomErrorKind::Context(c) =>
			{
				expected_for_classifier(c).iter().for_each(|c| {
					expectations.push(Cow::Borrowed(*c));
				});
			},
			NomErrorKind::Synthetic(merged) =>
			{
				// For each error that occurs at the rightmost parse position,
				// we recurse.
				let rightmost = &merged[0].errors[0].0;
				merged.iter().for_each(|error| {
					error
						.errors
						.iter()
						.take_while(|(span, _)| {
							span.location_offset()
								== rightmost.location_offset()
						})
						.for_each(|(_, kind)| {
							expectations.append(&mut kind.expectations())
						});
				})
			},
			NomErrorKind::Char(c) =>
			{
				// We can't push the reference to the string directly to
				// `expectations` here because of sharing rules.
				expectations.push(Cow::Owned(format!("`{}`", c)));
			},
			NomErrorKind::Nom(e) => match e
			{
				ErrorKind::Eof =>
				{
					// TODO: This isn't right yet; may need to add handling to
					// `function` combinator.
					expected_for_classifier(INCOMPLETE_FUNCTION_BODY_CONTEXT)
						.iter()
						.for_each(|c| {
							expectations.push(Cow::Borrowed(*c));
						})
				},
				_ =>
				{
					// Ignore other `nom` errors, as we handle most others
					// through context only.
				}
			}
		}
		expectations
	}
}

impl Error for ParseError<'_> {}

impl<'a> nom::error::ParseError<Span<'a>> for ParseError<'a>
{
	fn from_error_kind(input: Span<'a>, kind: ErrorKind) -> Self
	{
		Self {
			errors: vec![(input, NomErrorKind::Nom(kind))]
		}
	}

	fn append(input: Span<'a>, kind: ErrorKind, mut other: Self) -> Self
	{
		other.errors.push((input, NomErrorKind::Nom(kind)));
		other
	}

	fn from_char(input: Span<'a>, c: char) -> Self
	{
		Self {
			errors: vec![(input, NomErrorKind::Char(c))]
		}
	}

	fn or(self, other: Self) -> Self
	{
		assert!(!self.errors.is_empty());
		assert!(!other.errors.is_empty());

		// The alternatives were supplied by the `alt` combinator, and we only
		// want to keep the errors associated with rightmost parses. We compare
		// the first error of each alternative, as it represents the rightmost
		// parse position.
		match (&self.errors[0], &other.errors[0])
		{
			((span, _), (other_span, _))
				if span.location_offset() > other_span.location_offset() =>
			{
				self
			},
			((span, _), (other_span, _))
				if span.location_offset() < other_span.location_offset() =>
			{
				other
			},
			((input, _), _) => Self {
				errors: vec![(
					*input,
					NomErrorKind::Synthetic(vec![self, other])
				)]
			}
		}
	}
}

impl<'a> nom::error::ContextError<Span<'a>> for ParseError<'a>
{
	fn add_context(input: Span<'a>, ctx: &'static str, mut other: Self)
	-> Self
	{
		other.errors.push((input, NomErrorKind::Context(ctx)));
		other
	}
}

impl<'a, E> nom::error::FromExternalError<Span<'a>, E> for ParseError<'a>
{
	fn from_external_error(input: Span<'a>, kind: ErrorKind, _e: E) -> Self
	{
		Self::from_error_kind(input, kind)
	}
}

impl Display for ParseError<'_>
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		assert!(!self.errors.is_empty());

		let rightmost = self.errors[0].0;
		let column = rightmost.get_column();
		write!(
			f,
			"Parse error @ {}:{} (byte {}): expected ",
			rightmost.location_line(),
			column,
			rightmost.location_offset()
		)?;
		let expectations = self
			.errors
			.iter()
			.take_while(|(span, _)| {
				span.location_offset() == rightmost.location_offset()
			})
			.flat_map(|(_, kind)| kind.expectations())
			.collect::<Vec<_>>();
		writeln!(
			f,
			"{}, but found {}",
			// The `unwrap` is safe because `expectations` is guaranteed to be
			// nonempty.
			expected(&expectations).unwrap(),
			token(rightmost).map_or(Cow::Borrowed("end of input"), |t| {
				Cow::Owned(format!("`{}`", t.fragment()))
			})
		)?;
		// Output the full line containing the error.
		let line = String::from_utf8_lossy(rightmost.get_line_beginning());
		writeln!(f, "  {}", line)?;
		// Output the visual locator for the error.
		write!(f, "  {}^", " ".repeat(column - 1))
	}
}

////////////////////////////////////////////////////////////////////////////////
//                     Communicating parser expectations.                     //
////////////////////////////////////////////////////////////////////////////////

/// Answer the aggregated lexical expectations for the given parse position
/// classifiers.
///
/// # Parameters
/// - `classifiers`: The parse position classifiers.
///
/// # Returns
/// A string suitable for presentation to the user, or `None` if the classifiers
/// are empty.
fn expected<T>(classifiers: &[T]) -> Option<Cow<'_, str>>
where
	T: AsRef<str> + Ord
{
	if classifiers.is_empty()
	{
		None
	}
	else if classifiers.len() == 1
	{
		// There's only one classifier, so we can avoid a lot of work.
		Some(Cow::Borrowed(classifiers[0].as_ref()))
	}
	else
	{
		let mut expectations = classifiers.iter().collect::<Vec<_>>();
		expectations.sort();
		expectations.dedup();
		Some(match expectations.len()
		{
			3.. =>
			{
				let mut s = String::new();
				for expectation in &expectations[..expectations.len() - 1]
				{
					s.push_str(expectation.as_ref());
					s.push_str(", ");
				}
				s.push_str("or ");
				let last = expectations.last().unwrap().as_ref();
				s.push_str(last);
				Cow::Owned(s)
			},
			2 => Cow::Owned(format!(
				"{} or {}",
				expectations[0].as_ref(),
				expectations[1].as_ref()
			)),
			1 => Cow::Borrowed(expectations[0].as_ref()),
			_ => unreachable!()
		})
	}
}

/// Answer the lexical expectations for a given parse position classifier.
///
/// # Parameters
/// - `classifier`: The parse position classifier.
///
/// # Returns
/// A slice of static strings representing the expected lexical tokens at any
/// parse position labeled by the given classifier.
fn expected_for_classifier(classifier: &str) -> &'static [&'static str]
{
	match classifier
	{
		FUNCTION_CONTEXT => &[
			"parameter definition",
			"`{`",
			"`[`",
			"`(`",
			"`-`",
			"dice expression",
			"integer"
		],
		PARAMETER_CONTEXT => &["parameter definition"],
		NEXT_PARAMETER_CONTEXT => &["`,`"],
		FUNCTION_BODY_CONTEXT =>
		{
			&["`{`", "`[`", "`(`", "`-`", "dice expression", "integer"]
		},
		INCOMPLETE_FUNCTION_BODY_CONTEXT => &[
			"end of input",
			"`+`",
			"`-`",
			"`*`",
			"`×`",
			"`/`",
			"`÷`",
			"`%`",
			"`^`"
		],
		EXPRESSION_CONTEXT =>
		{
			&["`{`", "`[`", "`(`", "`-`", "dice expression", "integer"]
		},
		RANGE_CONTEXT => &["`[`"],
		RANGE_START_CONTEXT =>
		{
			&["`{`", "`[`", "`(`", "`-`", "dice expression", "integer"]
		},
		RANGE_END_CONTEXT =>
		{
			&["`{`", "`[`", "`(`", "`-`", "dice expression", "integer"]
		},
		DICE_CONTEXT => &["`{`", "`(`", "integer"],
		STANDARD_DICE_CONTEXT => &["`{`", "`(`", "integer"],
		CUSTOM_DICE_CONTEXT => &["`{`", "`(`", "integer"],
		DICE_COUNT_CONTEXT => &["`{`", "`(`", "integer"],
		STANDARD_FACES_CONTEXT => &["`{`", "`(`", "integer"],
		CUSTOM_FACES_CONTEXT => &["`[`"],
		DROP_EXPRESSION_CONTEXT => &["`{`", "`(`", "integer"],
		GROUP_CONTEXT => &["`(`"],
		VARIABLE_CONTEXT => &["`{`"],
		IDENTIFIER_CONTEXT => &["identifier"],
		CONSTANT_CONTEXT => &["integer"],
		unexpected =>
		{
			unreachable!("unexpected context classifier: {}", unexpected)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                            Context classifiers.                            //
////////////////////////////////////////////////////////////////////////////////

/// Context classifier for the `function` production rule.
pub(crate) const FUNCTION_CONTEXT: &str = "function";

/// Context classifier for the `parameter` production rule.
pub(crate) const PARAMETER_CONTEXT: &str = "parameter";

/// Context classifier for the next parameter in a function definition.
pub(crate) const NEXT_PARAMETER_CONTEXT: &str = "next parameter";

/// Context classifier for a function body.
pub(crate) const FUNCTION_BODY_CONTEXT: &str = "function body";

/// Context classifier for an incomplete function body.
pub(crate) const INCOMPLETE_FUNCTION_BODY_CONTEXT: &str =
	"incomplete function body";

/// Context classifier for the `expression` production rule.
pub(crate) const EXPRESSION_CONTEXT: &str = "expression";

/// Context classifier for the `range` production rule.
pub(crate) const RANGE_CONTEXT: &str = "range";

/// Context classifier for the start of a range.
pub(crate) const RANGE_START_CONTEXT: &str = "range start";

/// Context classifier for the end of a range.
pub(crate) const RANGE_END_CONTEXT: &str = "range end";

/// Context classifier for the `dice` production rule.
pub(crate) const DICE_CONTEXT: &str = "dice";

/// Context classifier for the `standard_dice` production rule.
pub(crate) const STANDARD_DICE_CONTEXT: &str = "standard dice";

/// Context classifier for the `custom_dice` production rule.
pub(crate) const CUSTOM_DICE_CONTEXT: &str = "custom dice";

/// Context classifier for the `dice_count` production rule.
pub(crate) const DICE_COUNT_CONTEXT: &str = "dice count";

/// Context classifier for the `standard_faces` production rule.
pub(crate) const STANDARD_FACES_CONTEXT: &str = "standard faces";

/// Context classifier for the `custom_faces` production rule.
pub(crate) const CUSTOM_FACES_CONTEXT: &str = "custom faces";

/// Context classifier for the `drop_expression` production rule.
pub(crate) const DROP_EXPRESSION_CONTEXT: &str = "drop expression";

/// Context classifier for the `group` production rule.
pub(crate) const GROUP_CONTEXT: &str = "primary";

/// Context classifier for the `variable` production rule.
pub(crate) const VARIABLE_CONTEXT: &str = "variable";

/// Context classifier for the `IDENTIFIER` production rule.
pub(crate) const IDENTIFIER_CONTEXT: &str = "identifier";

/// Context classifier for the `CONSTANT` production rule.
pub(crate) const CONSTANT_CONTEXT: &str = "constant";
