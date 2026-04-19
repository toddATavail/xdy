//! # Diagnostics
//!
//! Herein is the diagnostic layer for the `xDy` parser and
//! [validator](Validator). The [`diagnose`] function analyzes a source string
//! by repeatedly parsing, detecting errors, applying fixes, and re-parsing
//! until all syntactic errors are collected; it then runs the semantic
//! validator on the final AST whenever the original source parsed cleanly. This
//! produces structured [`Diagnostic`]s with specific error kinds, source spans
//! highlighting, human-readable messages, secondary [related labels](
//! RelatedLabel) for cross-referenced positions, and suggested fixes —
//! including complete corrected source strings for unambiguous errors and
//! templates with highlighted placeholder regions for ambiguous ones.
//!
//! # Architecture
//!
//! The diagnostic layer is a consumer of the parser's [`ParseError`] and the
//! validator's [`CompilationError`], not part of either. The
//! [`compile`](crate::compile) and
//! [`compile_unoptimized`](crate::compile_unoptimized) functions continue to
//! deal in the typed errors directly; diagnostics are a separate concern for
//! clients that want richer error information (e.g., for IDE integration,
//! syntax highlighting on every keystroke, or autocompletion).
//!
//! # Error Recovery Strategy
//!
//! The doctor runs two sequential passes. Syntactic errors go through a
//! **fix-and-retry loop**; once the source parses cleanly, a **semantic pass**
//! runs the [`Validator`] against the final AST. The [rustdoc for
//! `diagnose`](diagnose) carries a Mermaid rendering of the complete pipeline;
//! the ASCII sketch below summarizes the flow for casual skimming.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ Parser::parse(current_source)                        │
//! │   ├─ success → Validator::validate(ast) → diagnostics│
//! │   └─ error   → analyze → apply fix → loop            │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! Each fix inserts, removes, or wraps characters in the source string, making
//! progress toward a clean parse. An `OffsetMap` tracks cumulative
//! adjustments so that diagnostic spans are mapped back to positions in the
//! original source. The loop terminates when parsing succeeds or when an
//! unfixable error is encountered.
//!
//! The semantic pass runs **only when the original source parsed cleanly** —
//! that is, when no fixes were applied. Semantic diagnostics (e.g., duplicate
//! parameters) point at spans the user actually typed; running the validator on
//! a fix-synthesized source would attach diagnostics to characters the user
//! never wrote, so semantic checks wait for a syntactically valid source.

use std::{
	collections::HashSet,
	fmt::{self, Display, Formatter}
};

use crate::{
	CompilationError, Parser, Validator,
	ast::{Function, Parameter},
	parser::ParseError,
	span::SourceSpan
};

////////////////////////////////////////////////////////////////////////////////
//                                   Types.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The specific kind of diagnostic, indicating what went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticKind
{
	/// An opening delimiter has no matching close.
	///
	/// # Examples
	/// `(3D6`, `{x`, `[1:6`
	UnclosedDelimiter
	{
		/// The opening delimiter character.
		opener: char,
		/// The expected closing delimiter character.
		expected_closer: char
	},

	/// A closing delimiter has no matching open.
	///
	/// # Examples
	/// `3D6)`, `x}`
	UnopenedDelimiter
	{
		/// The unmatched closing delimiter character.
		closer: char
	},

	/// A binary operator is missing its right operand.
	///
	/// # Examples
	/// `3D6 +`, `1 + * 2`
	MissingRightOperand
	{
		/// The binary operator character.
		operator: char
	},

	/// A binary operator is missing its left operand.
	///
	/// # Examples
	/// `+ 3D6`
	MissingLeftOperand
	{
		/// The binary operator character.
		operator: char
	},

	/// A bare identifier appears where only `{identifier}` is legal.
	///
	/// # Examples
	/// `xD6`, `abcD12`
	BareIdentifier,

	/// A dice operator has no face specification.
	///
	/// # Examples
	/// `3D`, `3d`
	MissingDiceFaces,

	/// A `drop` keyword is not followed by `lowest` or `highest`.
	///
	/// # Examples
	/// `4D6 drop`
	IncompleteDropClause,

	/// A parameter definition is missing the `:` separator and body.
	///
	/// # Examples
	/// `hello`, `x, y`
	IncompleteParameterDefinition,

	/// A valid expression is followed by unparseable input.
	///
	/// # Examples
	/// `3D6)`, `3D6 hello`
	TrailingInput,

	/// The input is empty or contains only whitespace.
	EmptyExpression,

	/// An unexpected token was encountered.
	UnexpectedToken,

	/// The input ended unexpectedly.
	UnexpectedEof,

	/// A formal parameter name was declared more than once. Unlike the kinds
	/// above — which are produced from [`ParseError`] during the fix-and-retry
	/// loop — this kind is produced by the [`Validator`] semantic pass after a
	/// clean parse. The diagnostic's primary [`span`](Diagnostic::span) points
	/// at the duplicate occurrence; the [`related`](Diagnostic::related) list
	/// carries the span of the first occurrence.
	///
	/// # Examples
	/// `x, x: {x}`, `a, b, b: {a}`
	DuplicateParameter
	{
		/// The duplicated parameter name.
		name: String
	}
}

impl Display for DiagnosticKind
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			Self::UnclosedDelimiter { opener, .. } =>
			{
				write!(f, "unclosed `{}`", opener)
			},
			Self::UnopenedDelimiter { closer } =>
			{
				write!(f, "unexpected `{}`", closer)
			},
			Self::MissingRightOperand { operator } =>
			{
				write!(f, "missing right operand of `{}`", operator)
			},
			Self::MissingLeftOperand { operator } =>
			{
				write!(f, "missing left operand of `{}`", operator)
			},
			Self::BareIdentifier => write!(f, "bare identifier"),
			Self::MissingDiceFaces => write!(f, "missing dice faces"),
			Self::IncompleteDropClause =>
			{
				write!(f, "incomplete drop clause")
			},
			Self::IncompleteParameterDefinition =>
			{
				write!(f, "incomplete parameter definition")
			},
			Self::TrailingInput => write!(f, "trailing input"),
			Self::EmptyExpression => write!(f, "empty expression"),
			Self::UnexpectedToken => write!(f, "unexpected token"),
			Self::UnexpectedEof => write!(f, "unexpected end of input"),
			Self::DuplicateParameter { name } =>
			{
				write!(f, "duplicate parameter `{}`", name)
			}
		}
	}
}

/// A diagnostic produced by the [`doctor`](diagnose).
///
/// # Notes
/// The `span` field refers to byte offsets in the **original** source string
/// passed to [`diagnose`], not in the corrected source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic
{
	/// The specific kind of error.
	pub kind: DiagnosticKind,

	/// The region of source text that is erroneous, as byte offsets in the
	/// original source.
	pub span: SourceSpan,

	/// A human-readable error message.
	pub message: String,

	/// Secondary spans that enrich the diagnostic with related context, such
	/// as the prior occurrence of a duplicate name. Each entry is a
	/// [`RelatedLabel`] whose span is in the **original** source coordinates.
	/// Empty for most parse-error diagnostics; populated by semantic
	/// diagnostics that naturally reference multiple source positions.
	pub related: Vec<RelatedLabel>,

	/// Suggested fixes, from most to least specific.
	pub suggestions: Vec<Suggestion>
}

/// A secondary source label attached to a [`Diagnostic`]. Used to point at
/// related positions in the source — for example, the prior declaration of a
/// duplicated parameter — without expanding the primary span of the
/// diagnostic.
///
/// # Notes
/// The `span` refers to byte offsets in the **original** source string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedLabel
{
	/// The byte range of the related region in the original source.
	pub span: SourceSpan,

	/// A short human-readable explanation of the relationship, such as
	/// `"first declared here"`.
	pub message: String
}

impl Display for Diagnostic
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{} ({}): {}", self.kind, self.span, self.message)
	}
}

/// A suggested fix for a diagnostic.
///
/// # Notes
/// The `corrected_source` is a complete, parseable source string. For
/// unambiguous fixes (e.g., `{x` → `{x}`), `placeholders` is empty. For
/// ambiguous fixes (e.g., `3D` → `3D6`), `placeholders` identifies the regions
/// that are defaults and can be replaced by the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion
{
	/// A human-readable description of the fix.
	pub description: String,

	/// The complete corrected source string.
	pub corrected_source: String,

	/// Placeholder regions in the corrected source that the user may want
	/// to replace. Empty for unambiguous fixes.
	pub placeholders: Vec<Placeholder>
}

/// A placeholder region in a corrected source template.
///
/// A sophisticated UI can highlight these regions and present the
/// `valid_kinds` as a picker, allowing the user to type over the default
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placeholder
{
	/// The byte range of the placeholder in the corrected source string.
	pub span: SourceSpan,

	/// A description of what belongs here (e.g., "face count", "operand").
	pub description: &'static str,

	/// The kinds of expressions that are valid at this position.
	pub valid_kinds: &'static [&'static str]
}

/// The result of diagnosing a source string.
///
/// # Notes
/// If all errors are fixable, `corrected_source` contains a parseable source
/// string with all fixes applied. If any error is unfixable, `corrected_source`
/// is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnoseResult
{
	/// The diagnostics collected during analysis, in source order.
	pub diagnostics: Vec<Diagnostic>,

	/// The fully corrected source string, if all errors were fixable. This
	/// string will parse cleanly via [`Parser::parse`].
	pub corrected_source: Option<String>
}

/// Valid expression kinds for operand positions.
const OPERAND_KINDS: &[&str] =
	&["integer", "{variable}", "(expression)", "dice expression"];

/// Valid expression kinds for face count positions.
const FACE_COUNT_KINDS: &[&str] = &["integer", "{variable}", "(expression)"];

/// Valid kinds for drop direction.
const DROP_DIRECTION_KINDS: &[&str] = &["lowest", "highest"];

////////////////////////////////////////////////////////////////////////////////
//                              Error analysis.                               //
////////////////////////////////////////////////////////////////////////////////

/// Analyze a [`ParseError`] and the current source text to produce a
/// [`Diagnostic`] with an appropriate fix.
///
/// # Parameters
/// - `source`: The current (possibly already partially fixed) source text.
/// - `error`: The parse error to analyze.
/// - `offset_map`: Tracks cumulative byte adjustments from prior fixes, used to
///   map positions back to the original source.
///
/// # Returns
/// A diagnostic describing the error, with suggestions for fixing it.
fn analyze_error(
	source: &str,
	error: &ParseError<'_>,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let rightmost = &error.errors[0];
	let pos = rightmost.0.location_offset();
	let at_eof =
		pos >= source.len() || source[pos..].chars().all(char::is_whitespace);

	// Collect all expectations at the rightmost parse position.
	let expectations = error
		.errors
		.iter()
		.take_while(|(span, _)| span.location_offset() == pos)
		.flat_map(|(_, kind)| kind.expectations())
		.collect::<Vec<_>>();

	// Check for specific patterns based on error structure and source context.
	let expects_expression = expectations.iter().any(|e| {
		let s = e.as_ref();
		s == "integer"
			|| s == "dice expression"
			|| s == "`(`"
			|| s == "`{`"
			|| s == "`[`"
			|| s == "`-`"
	});
	let expects_delimiter =
		expectations.iter().find_map(|e| match e.as_ref()
		{
			"`)`" => Some(('(', ')')),
			"`]`" => Some(('[', ']')),
			"`}`" => Some(('{', '}')),
			_ => None
		});

	// Pattern: closing delimiter expected at EOF → unclosed delimiter.
	if at_eof && let Some((opener, closer)) = expects_delimiter
	{
		let opener_pos = find_unmatched_opener(source, opener, closer, pos);
		return make_unclosed_delimiter(
			source, opener, closer, opener_pos, pos, offset_map
		);
	}

	// Pattern: expression expected, and the text at the error position is an
	// identifier → bare identifier in expression context.
	if expects_expression
		&& !at_eof
		&& let Some(diag) =
			detect_bare_identifier_at_pos(source, pos, offset_map)
	{
		return diag;
	}

	// Pattern: expression or identifier expected after delimiter/operator.
	let expects_identifier =
		expectations.iter().any(|e| e.as_ref() == "identifier");
	if (expects_expression || expects_identifier)
		&& let Some((prev, prev_pos)) = find_preceding_char(source, pos)
	{
		if prev == 'd' || prev == 'D'
		{
			return make_missing_dice_faces(source, prev_pos, pos, offset_map);
		}
		if "+-*×/÷%^".contains(prev)
		{
			return make_missing_right_operand(
				source, prev, prev_pos, pos, offset_map
			);
		}
		// Opening delimiter with no expression inside:
		// e.g., `(` → `(0)`, `[` → `[0:0]`, `{` → `{x}`.
		if prev == '(' || prev == '[' || prev == '{'
		{
			return make_incomplete_delimited(
				source, prev, prev_pos, pos, offset_map
			);
		}
		// After `:` inside a range, e.g., `[1:` → insert end expression and
		// `]`.
		if prev == ':'
			&& let Some(bracket_pos) = source[..prev_pos].rfind('[')
		{
			let orig_bracket = offset_map.to_original(bracket_pos);
			let mut corrected = source[..pos].to_string();
			let p_start = corrected.len();
			corrected.push('0');
			let p_end = corrected.len();
			corrected.push(']');
			corrected.push_str(&source[pos..]);
			return Diagnostic {
				kind: DiagnosticKind::UnclosedDelimiter {
					opener: '[',
					expected_closer: ']'
				},
				span: SourceSpan {
					start: orig_bracket,
					end: orig_bracket + 1
				},
				message: "expected `]` to close `[`".into(),
				related: vec![],
				suggestions: vec![Suggestion {
					description: "insert range end \
								and `]`"
						.into(),
					corrected_source: corrected,
					placeholders: vec![Placeholder {
						span: SourceSpan {
							start: p_start,
							end: p_end
						},
						description: "range end",
						valid_kinds: OPERAND_KINDS
					}]
				}]
			};
		}
	}

	// Pattern: drop direction expected → incomplete drop clause.
	if expectations
		.iter()
		.any(|e| e.as_ref() == "`lowest`" || e.as_ref() == "`highest`")
	{
		let drop_pos =
			source[..pos].rfind("drop").unwrap_or(pos.saturating_sub(5));
		let orig_drop = offset_map.to_original(drop_pos);
		let orig_end = offset_map.to_original((drop_pos + 4).min(source.len()));
		return make_incomplete_drop(
			source, orig_drop, orig_end, pos, offset_map
		);
	}

	// Pattern: operator at start of input → missing left operand.
	if !at_eof
	{
		let ch = source[pos..].chars().next().unwrap_or('\0');
		if "+-*×/÷%^".contains(ch) && pos == 0
		{
			return make_missing_left_operand(source, ch, offset_map);
		}
	}

	// Pattern: empty or whitespace-only input.
	if source.trim().is_empty()
	{
		return make_empty_expression(source);
	}

	// Pattern: bare identifier followed by `d`/`D` → BareIdentifier.
	// Must run before incomplete parameter detection, because `xD6` would
	// otherwise be interpreted as parameter `xD6` missing `:`.
	if let Some(diag) = detect_bare_identifier(source, pos, offset_map)
	{
		return diag;
	}

	// Pattern: bare identifier parsed as parameter, missing `:` and body.
	if expectations
		.iter()
		.any(|e| e.as_ref() == "`,`" || e.as_ref() == "`:`")
		&& let Some(diag) = detect_incomplete_parameter(source, pos, offset_map)
	{
		return diag;
	}

	// Pattern: bare closing delimiter where an expression was expected → the
	// user has a spurious `)`, `]`, or `}` with no matching opener. This is
	// more specific than the catch-all `UnexpectedToken` and produces a
	// `remove the unopened` suggestion. Only fires when an expression was
	// expected at `pos`; if the parser had already consumed a complete
	// expression, the trailing-input branch below handles it instead.
	if expects_expression && !at_eof
	{
		let ch = source[pos..].chars().next().unwrap_or('\0');
		if ch == ')' || ch == ']' || ch == '}'
		{
			return make_unopened_delimiter(source, ch, pos, offset_map);
		}
	}

	// Pattern: end of input from all_consuming → trailing input.
	if expectations.iter().any(|e| e.as_ref() == "end of input") && !at_eof
	{
		let orig_pos = offset_map.to_original(pos);
		let orig_end = offset_map.to_original(source.len());
		return Diagnostic {
			kind: DiagnosticKind::TrailingInput,
			span: SourceSpan {
				start: orig_pos,
				end: orig_end
			},
			message: format!(
				"unexpected `{}` after expression",
				&source[pos..].split_whitespace().next().unwrap_or("")
			),
			related: vec![],
			suggestions: vec![Suggestion {
				description: "remove trailing input".into(),
				corrected_source: source[..pos].trim_end().to_string(),
				placeholders: vec![]
			}]
		};
	}

	// Catch-all: unexpected token or EOF.
	let orig_pos = offset_map.to_original(pos);
	if at_eof
	{
		Diagnostic {
			kind: DiagnosticKind::UnexpectedEof,
			span: SourceSpan {
				start: orig_pos,
				end: orig_pos
			},
			message: "unexpected end of input".into(),
			related: vec![],
			suggestions: vec![]
		}
	}
	else
	{
		let token_end = source[pos..]
			.find(|c: char| c.is_whitespace())
			.map_or(source.len(), |i| pos + i);
		let orig_end = offset_map.to_original(token_end);
		Diagnostic {
			kind: DiagnosticKind::UnexpectedToken,
			span: SourceSpan {
				start: orig_pos,
				end: orig_end
			},
			message: format!("unexpected `{}`", &source[pos..token_end]),
			related: vec![],
			suggestions: vec![]
		}
	}
}

/// Search backward from `pos` for the nearest non-whitespace character.
///
/// # Parameters
/// - `source`: The source text.
/// - `pos`: The position to search backward from.
///
/// # Returns
/// The character and its byte offset, if found.
fn find_preceding_char(source: &str, pos: usize) -> Option<(char, usize)>
{
	let before = &source[..pos];
	for (i, c) in before.char_indices().rev()
	{
		if !c.is_whitespace()
		{
			return Some((c, i));
		}
	}
	None
}

/// Find the byte offset of an unmatched opening delimiter before `pos`.
///
/// # Parameters
/// - `source`: The source text.
/// - `opener`: The opening delimiter character.
/// - `closer`: The closing delimiter character.
/// - `pos`: The position to search before.
///
/// # Returns
/// The byte offset of the unmatched opener, if found.
fn find_unmatched_opener(
	source: &str,
	opener: char,
	closer: char,
	pos: usize
) -> Option<usize>
{
	let mut depth = 0i32;
	let mut last_opener = None;
	for (i, c) in source[..pos].char_indices()
	{
		if c == opener
		{
			depth += 1;
			last_opener = Some(i);
		}
		else if c == closer
		{
			depth -= 1;
		}
	}
	if depth > 0 { last_opener } else { None }
}

/// Detect a bare identifier at the error position in an expression context.
///
/// When the parser expects an expression operand (e.g., after `+`) but the
/// text at the error position is an identifier, the user almost certainly
/// forgot to wrap it in `{}`.
///
/// # Parameters
/// - `source`: The source text.
/// - `pos`: The error position.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A `BareIdentifier` diagnostic if the text at `pos` is an identifier.
fn detect_bare_identifier_at_pos(
	source: &str,
	pos: usize,
	offset_map: &OffsetMap
) -> Option<Diagnostic>
{
	let remaining = &source[pos..];
	// Check if the text starts with an identifier character.
	let first = remaining.chars().next()?;
	if !first.is_alphabetic() && first != '_'
	{
		return None;
	}
	// Find the end of the identifier.
	let ident_end = remaining
		.find(|c: char| {
			!(c.is_alphanumeric()
				|| c == '_' || c == '-'
				|| c == '.' || (c.is_whitespace() && !matches!(c, '\n' | '\r')))
		})
		.unwrap_or(remaining.len());
	let name = remaining[..ident_end].trim_end();
	if name.is_empty()
	{
		return None;
	}
	let orig_start = offset_map.to_original(pos);
	let orig_end = offset_map.to_original(pos + name.len());
	let after = &source[pos + ident_end..];
	let mut suggestions = Vec::new();

	// Check if the identifier contains `d`/`D` — if so, offer both a split fix
	// (e.g., `{x}D6`) and a whole-variable fix (e.g., `{xD6}`).
	let split_pos = name.char_indices().find(|&(i, c)| {
		(c == 'd' || c == 'D')
			&& i > 0 && name[..i]
			.starts_with(|c: char| c.is_alphabetic() || c == '_')
	});
	if let Some((d_offset, _)) = split_pos
	{
		let prefix = &name[..d_offset];
		let suffix = &name[d_offset..];
		let mut split_fix = source[..pos].to_string();
		split_fix.push('{');
		split_fix.push_str(prefix);
		split_fix.push('}');
		split_fix.push_str(suffix);
		split_fix.push_str(after);
		suggestions.push(Suggestion {
			description: format!("wrap `{}` in braces", prefix),
			corrected_source: split_fix,
			placeholders: vec![]
		});
	}
	// Always offer wrapping the whole identifier.
	let mut whole_fix = source[..pos].to_string();
	whole_fix.push('{');
	whole_fix.push_str(name);
	whole_fix.push('}');
	whole_fix.push_str(after);
	suggestions.push(Suggestion {
		description: format!("use `{}` as a variable name", name),
		corrected_source: whole_fix,
		placeholders: vec![]
	});

	// Use the first (most specific) suggestion's description for the message.
	// If there's a `d`/`D` split, the bare name is the prefix.
	let bare_name = split_pos.map(|(i, _)| &name[..i]).unwrap_or(name);
	Some(Diagnostic {
		kind: DiagnosticKind::BareIdentifier,
		span: SourceSpan {
			start: orig_start,
			end: orig_end
		},
		message: format!(
			"bare identifier `{}` is not valid here; \
			 variables must be wrapped in `{{}}`",
			bare_name
		),
		related: vec![],
		suggestions
	})
}

/// Detect an incomplete parameter definition.
///
/// When the parser sees identifiers followed by `,` or at EOF without `:`,
/// and the fix `{ident}` would produce a valid expression, offer both
/// interpretations: completing the parameter definition or wrapping in `{}`.
///
/// # Parameters
/// - `source`: The source text.
/// - `pos`: The error position.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// An `IncompleteParameterDefinition` diagnostic if the pattern matches.
fn detect_incomplete_parameter(
	source: &str,
	pos: usize,
	offset_map: &OffsetMap
) -> Option<Diagnostic>
{
	let trimmed = source.trim();
	if trimmed.is_empty()
	{
		return None;
	}
	// Extract the parameter names from the source (everything before pos).
	let params_text = source[..pos].trim();
	if params_text.is_empty()
	{
		return None;
	}
	// Verify this looks like parameter names (identifiers separated by `,`).
	let params: Vec<&str> = params_text
		.split(',')
		.map(|s| s.trim())
		.filter(|s| !s.is_empty())
		.collect();
	if params.is_empty()
		|| !params
			.iter()
			.all(|p| p.starts_with(|c: char| c.is_alphabetic() || c == '_'))
	{
		return None;
	}
	let orig_start = offset_map.to_original(0);
	let orig_end = offset_map.to_original(pos);

	let mut suggestions = Vec::new();
	let trailing = source[pos..].trim_start();

	// Build the variable reference suggestion (single parameter only).
	let var_suggestion = if params.len() == 1
	{
		let name = params[0];
		let mut var_fix = format!("{{{}}}", name);
		if !trailing.is_empty()
		{
			var_fix.push(' ');
			var_fix.push_str(trailing);
		}
		Some(Suggestion {
			description: format!("use `{}` as a variable reference", name),
			corrected_source: var_fix,
			placeholders: vec![]
		})
	}
	else
	{
		None
	};

	// Build the parameter definition suggestion.
	let param_suggestion = {
		let mut fix = source[..pos].to_string();
		fix.push_str(": ");
		// Try incorporating trailing content as the body. Only use it if the
		// result actually parses; otherwise fall back to a placeholder.
		let trailing_works = if !trailing.is_empty()
		{
			let mut candidate = fix.clone();
			candidate.push_str(trailing);
			Parser::parse(&candidate).is_ok()
		}
		else
		{
			false
		};
		if trailing_works
		{
			fix.push_str(trailing);
			Suggestion {
				description: "complete parameter definition".into(),
				corrected_source: fix,
				placeholders: vec![]
			}
		}
		else
		{
			let placeholder_start = fix.len();
			fix.push('0');
			let placeholder_end = fix.len();
			Suggestion {
				description: "complete parameter definition".into(),
				corrected_source: fix,
				placeholders: vec![Placeholder {
					span: SourceSpan {
						start: placeholder_start,
						end: placeholder_end
					},
					description: "expression",
					valid_kinds: OPERAND_KINDS
				}]
			}
		}
	};

	// When there's trailing content (like `* 3`), the variable reference
	// interpretation is more likely — put it first so the retry loop uses it.
	if !trailing.is_empty()
	{
		if let Some(vs) = var_suggestion
		{
			suggestions.push(vs);
		}
		suggestions.push(param_suggestion);
	}
	else
	{
		suggestions.push(param_suggestion);
		if let Some(vs) = var_suggestion
		{
			suggestions.push(vs);
		}
	}

	Some(Diagnostic {
		kind: DiagnosticKind::IncompleteParameterDefinition,
		span: SourceSpan {
			start: orig_start,
			end: orig_end
		},
		message: format!(
			"expected `:` and expression body after parameter{}",
			if params.len() > 1 { "s" } else { "" }
		),
		related: vec![],
		suggestions
	})
}

/// Detect a bare identifier followed by `d`/`D` in the source.
///
/// # Notes
/// Bare identifiers are only legal as parameter definitions. When the parser
/// sees `xD6`, it interprets `x` as a parameter name and then expects `:` or
/// `,`. If the character after the identifier is `d`/`D`, the user almost
/// certainly meant `{x}D6`.
///
/// # Parameters
/// - `source`: The source text.
/// - `error_pos`: Where the parse error occurred.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A `BareIdentifier` diagnostic if the pattern matches.
fn detect_bare_identifier(
	source: &str,
	error_pos: usize,
	offset_map: &OffsetMap
) -> Option<Diagnostic>
{
	// The error position is after the identifier (where `:` or `,` was
	// expected). Check if there's a `d`/`D` somewhere in the remaining source
	// that's part of what was parsed as an identifier. Walk backward from
	// error_pos to find the start of the identifier.
	let before = &source[..error_pos];

	// Find the last contiguous identifier-like region ending at error_pos.
	let ident_start = before
		.char_indices()
		.rev()
		.take_while(|(_, c)| {
			c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.'
		})
		.last()
		.map(|(i, _)| i)?;

	let ident = &source[ident_start..error_pos];

	// Check if the identifier contains `d` or `D` followed by digits or `[`.
	// E.g., "xD6" → identifier is "xD6", but the `D` is a dice operator.
	for (i, c) in ident.char_indices()
	{
		if (c == 'd' || c == 'D') && i > 0
		{
			let after_d = &ident[i + 1..];
			if after_d.is_empty()
				|| after_d.starts_with(|c: char| c.is_ascii_digit())
				|| after_d.starts_with('[')
				|| after_d.starts_with('(')
				|| after_d.starts_with('{')
			{
				let name = &ident[..i];
				// Verify the name part is a valid identifier start.
				if name.starts_with(|c: char| c.is_alphabetic() || c == '_')
				{
					let orig_start = offset_map.to_original(ident_start);
					let orig_end = offset_map.to_original(ident_start + i);
					// Suggestion 1: split at `d`/`D` → `{name}D...`
					let after = &source[ident_start + i..];
					let mut split_fix = source[..ident_start].to_string();
					split_fix.push('{');
					split_fix.push_str(name);
					split_fix.push('}');
					split_fix.push_str(after);
					// Suggestion 2: whole identifier as variable
					// → `{xD6}`
					let mut whole_fix = source[..ident_start].to_string();
					whole_fix.push('{');
					whole_fix.push_str(ident);
					whole_fix.push('}');
					whole_fix.push_str(&source[error_pos..]);
					return Some(Diagnostic {
						kind: DiagnosticKind::BareIdentifier,
						span: SourceSpan {
							start: orig_start,
							end: orig_end
						},
						message: format!(
							"bare identifier `{}` is not valid here; \
							 variables must be wrapped in `{{}}`",
							name
						),
						related: vec![],
						suggestions: vec![
							Suggestion {
								description: format!(
									"wrap `{}` in braces",
									name
								),
								corrected_source: split_fix,
								placeholders: vec![]
							},
							Suggestion {
								description: format!(
									"use `{}` as a variable \
									 name",
									ident
								),
								corrected_source: whole_fix,
								placeholders: vec![]
							},
						]
					});
				}
			}
		}
	}
	None
}

////////////////////////////////////////////////////////////////////////////////
//                              Fix generation.                               //
////////////////////////////////////////////////////////////////////////////////

/// Create a diagnostic for an unclosed delimiter. If the delimiter contains
/// no expression, a placeholder `0` is inserted alongside the closing
/// delimiter.
///
/// # Parameters
/// - `source`: The current source text.
/// - `opener`: The opening delimiter character (e.g., `(`).
/// - `closer`: The expected closing delimiter character (e.g., `)`).
/// - `opener_pos`: The byte offset of the opening delimiter, or `None` if
///   unknown.
/// - `error_pos`: The byte offset where the error was detected.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with an `UnclosedDelimiter` kind and a suggestion that
/// inserts the closing delimiter (and a placeholder expression if needed).
fn make_unclosed_delimiter(
	source: &str,
	opener: char,
	closer: char,
	opener_pos: Option<usize>,
	error_pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	// The parser only raises a closer expectation after it has already
	// consumed an expression inside the delimited construct, so by the time
	// this function is called the content between `opener` and `error_pos` is
	// always non-empty. The zero-content path (which would need a placeholder
	// expression as well as a closer) is handled by
	// [`make_incomplete_delimited`] from the `expects_expression` branch of
	// [`analyze_error`] — a separate helper because the diagnostic kind and
	// placeholder set differ. See the doc-comment on [`analyze_error`] for
	// the full pattern-matching order.
	let actual_opener = opener_pos.unwrap_or(0);
	let orig_opener = offset_map.to_original(actual_opener);
	let mut corrected = source[..error_pos].to_string();
	corrected.push(closer);
	corrected.push_str(&source[error_pos..]);
	Diagnostic {
		kind: DiagnosticKind::UnclosedDelimiter {
			opener,
			expected_closer: closer
		},
		span: SourceSpan {
			start: orig_opener,
			end: orig_opener + opener.len_utf8()
		},
		message: format!("expected `{}` to close `{}`", closer, opener),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("insert `{}`", closer),
			corrected_source: corrected,
			placeholders: vec![]
		}]
	}
}

/// Create a diagnostic for a missing right operand. Offers two suggestions:
/// inserting a placeholder operand, or dropping the operator.
///
/// # Parameters
/// - `source`: The current source text.
/// - `operator`: The binary operator character (e.g., `+`).
/// - `op_pos`: The byte offset of the operator in the current source.
/// - `error_pos`: The byte offset where the right operand was expected.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with a `MissingRightOperand` kind and two suggestions.
fn make_missing_right_operand(
	source: &str,
	operator: char,
	op_pos: usize,
	error_pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let orig_error_pos = offset_map.to_original(error_pos);
	let orig_op_pos = offset_map.to_original(op_pos);
	// Suggestion 1: insert `0` at the error position with spacing.
	let mut insert_fix = source[..error_pos].to_string();
	if !insert_fix.is_empty() && !insert_fix.ends_with(' ')
	{
		insert_fix.push(' ');
	}
	let placeholder_start = insert_fix.len();
	insert_fix.push('0');
	let placeholder_end = insert_fix.len();
	if !source[error_pos..].is_empty() && !source[error_pos..].starts_with(' ')
	{
		insert_fix.push(' ');
	}
	insert_fix.push_str(&source[error_pos..]);

	// Suggestion 2: drop the operator.
	let before_op = source[..op_pos].trim_end();
	let mut drop_fix = before_op.to_string();
	if !source[error_pos..].is_empty()
	{
		if !drop_fix.is_empty()
		{
			drop_fix.push(' ');
		}
		drop_fix.push_str(source[error_pos..].trim_start());
	}

	Diagnostic {
		kind: DiagnosticKind::MissingRightOperand { operator },
		span: SourceSpan {
			start: orig_op_pos,
			end: orig_error_pos
		},
		message: format!("expected operand after `{}`", operator),
		related: vec![],
		suggestions: vec![
			Suggestion {
				description: format!("insert operand after `{}`", operator),
				corrected_source: insert_fix,
				placeholders: vec![Placeholder {
					span: SourceSpan {
						start: placeholder_start,
						end: placeholder_end
					},
					description: "operand",
					valid_kinds: OPERAND_KINDS
				}]
			},
			Suggestion {
				description: format!("remove `{}`", operator),
				corrected_source: drop_fix,
				placeholders: vec![]
			},
		]
	}
}

/// Create a diagnostic for a missing left operand. Offers two suggestions:
/// inserting a placeholder operand, or dropping the operator.
///
/// # Parameters
/// - `source`: The current source text.
/// - `operator`: The binary operator character at position 0.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with a `MissingLeftOperand` kind and two suggestions.
fn make_missing_left_operand(
	source: &str,
	operator: char,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let orig_pos = offset_map.to_original(0);
	// Suggestion 1: insert `0` before the operator.
	let mut insert_fix = String::from("0 ");
	insert_fix.push_str(source);

	// Suggestion 2: drop the operator.
	let after_op = source[operator.len_utf8()..].trim_start();
	let drop_fix = after_op.to_string();

	Diagnostic {
		kind: DiagnosticKind::MissingLeftOperand { operator },
		span: SourceSpan {
			start: orig_pos,
			end: orig_pos + operator.len_utf8()
		},
		message: format!("expected operand before `{}`", operator),
		related: vec![],
		suggestions: vec![
			Suggestion {
				description: format!("insert operand before `{}`", operator),
				corrected_source: insert_fix,
				placeholders: vec![Placeholder {
					span: SourceSpan { start: 0, end: 1 },
					description: "operand",
					valid_kinds: OPERAND_KINDS
				}]
			},
			Suggestion {
				description: format!("remove `{}`", operator),
				corrected_source: drop_fix,
				placeholders: vec![]
			},
		]
	}
}

/// Create a diagnostic for an incomplete delimited construct where the
/// expression inside is missing (e.g., `(` alone, `[` alone). The fix inserts
/// context-appropriate content: `0)` for groups, `0:0]` for ranges, `0]` for
/// custom faces, or `x}` for variables.
///
/// # Parameters
/// - `source`: The current source text.
/// - `opener`: The opening delimiter character.
/// - `opener_pos`: The byte offset of the opener in the current source.
/// - `error_pos`: The byte offset where the error was detected.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with an `UnclosedDelimiter` kind and a suggestion that
/// inserts the appropriate placeholder content and closing delimiter.
fn make_incomplete_delimited(
	source: &str,
	opener: char,
	opener_pos: usize,
	error_pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let orig_opener = offset_map.to_original(opener_pos);
	let closer = match opener
	{
		'(' => ')',
		'[' => ']',
		'{' => '}',
		_ => ')'
	};
	let mut corrected = source[..error_pos].to_string();
	let mut placeholders = Vec::new();
	// Determine if `[` is a custom faces bracket (after `d`/`D`) or a range.
	let is_custom_faces = opener == '['
		&& opener_pos > 0
		&& source[..opener_pos]
			.chars()
			.last()
			.is_some_and(|c| c == 'd' || c == 'D');
	match opener
	{
		'[' if is_custom_faces =>
		{
			// Custom faces need a comma-separated list, e.g., `[1,2,3]`.
			let p_start = corrected.len();
			corrected.push('0');
			let p_end = corrected.len();
			placeholders.push(Placeholder {
				span: SourceSpan {
					start: p_start,
					end: p_end
				},
				description: "face value",
				valid_kinds: &["integer"]
			});
		},
		'[' =>
		{
			// Ranges need `start:end`, so insert `0:0`.
			let p1_start = corrected.len();
			corrected.push('0');
			let p1_end = corrected.len();
			corrected.push(':');
			let p2_start = corrected.len();
			corrected.push('0');
			let p2_end = corrected.len();
			placeholders.push(Placeholder {
				span: SourceSpan {
					start: p1_start,
					end: p1_end
				},
				description: "range start",
				valid_kinds: OPERAND_KINDS
			});
			placeholders.push(Placeholder {
				span: SourceSpan {
					start: p2_start,
					end: p2_end
				},
				description: "range end",
				valid_kinds: OPERAND_KINDS
			});
		},
		'{' =>
		{
			// Variables need an identifier.
			let p_start = corrected.len();
			corrected.push('x');
			let p_end = corrected.len();
			placeholders.push(Placeholder {
				span: SourceSpan {
					start: p_start,
					end: p_end
				},
				description: "identifier",
				valid_kinds: &["identifier"]
			});
		},
		_ =>
		{
			// Groups just need an expression.
			let p_start = corrected.len();
			corrected.push('0');
			let p_end = corrected.len();
			placeholders.push(Placeholder {
				span: SourceSpan {
					start: p_start,
					end: p_end
				},
				description: "expression",
				valid_kinds: OPERAND_KINDS
			});
		}
	}
	corrected.push(closer);
	corrected.push_str(&source[error_pos..]);
	Diagnostic {
		kind: DiagnosticKind::UnclosedDelimiter {
			opener,
			expected_closer: closer
		},
		span: SourceSpan {
			start: orig_opener,
			end: orig_opener + opener.len_utf8()
		},
		message: format!("expected `{}` to close `{}`", closer, opener),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("insert expression and `{}`", closer),
			corrected_source: corrected,
			placeholders
		}]
	}
}

/// Create a diagnostic for missing dice faces after `d`/`D`. The fix inserts
/// `6` as a placeholder face count immediately after the dice operator.
///
/// # Parameters
/// - `source`: The current source text.
/// - `d_pos`: The byte offset of the `d`/`D` operator.
/// - `_error_pos`: The byte offset where the face count was expected (unused
///   because we insert at `d_pos + 1`).
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with a `MissingDiceFaces` kind and a suggestion with a
/// placeholder for the face count.
fn make_missing_dice_faces(
	source: &str,
	d_pos: usize,
	_error_pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let orig_d_pos = offset_map.to_original(d_pos);
	// Insert `6` as a default face count right after the `d`/`D`, preserving
	// any whitespace between the operator and the error.
	let insert_pos = d_pos + 1;
	let mut corrected = source[..insert_pos].to_string();
	let placeholder_start = corrected.len();
	corrected.push('6');
	let placeholder_end = corrected.len();
	// Preserve content from after the `d`/`D` onward (including any whitespace
	// that was between the operator and the error position).
	corrected.push_str(&source[insert_pos..]);
	Diagnostic {
		kind: DiagnosticKind::MissingDiceFaces,
		span: SourceSpan {
			start: orig_d_pos,
			end: orig_d_pos + 1
		},
		message: format!(
			"expected face count after `{}`",
			&source[d_pos..d_pos + 1]
		),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert face count".into(),
			corrected_source: corrected,
			placeholders: vec![Placeholder {
				span: SourceSpan {
					start: placeholder_start,
					end: placeholder_end
				},
				description: "face count",
				valid_kinds: FACE_COUNT_KINDS
			}]
		}]
	}
}

/// Create a diagnostic for an incomplete drop clause. The fix inserts
/// `lowest` as a placeholder direction after the `drop` keyword.
///
/// # Parameters
/// - `source`: The current source text.
/// - `orig_drop_start`: The byte offset of `drop` in the original source.
/// - `orig_drop_end`: The byte offset of the end of `drop` in the original
///   source.
/// - `error_pos`: The byte offset where the direction was expected.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with an `IncompleteDropClause` kind and a suggestion with a
/// placeholder for the drop direction.
fn make_incomplete_drop(
	source: &str,
	orig_drop_start: usize,
	orig_drop_end: usize,
	error_pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let _ = offset_map;
	// Insert ` lowest` after `drop`.
	let insert_pos = source[..error_pos]
		.rfind("drop")
		.map_or(error_pos, |p| p + 4);
	let mut corrected = source[..insert_pos].to_string();
	corrected.push(' ');
	let placeholder_start = corrected.len();
	corrected.push_str("lowest");
	let placeholder_end = corrected.len();
	corrected.push_str(&source[insert_pos..]);
	Diagnostic {
		kind: DiagnosticKind::IncompleteDropClause,
		span: SourceSpan {
			start: orig_drop_start,
			end: orig_drop_end
		},
		message: "expected `lowest` or `highest` after `drop`".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert drop direction".into(),
			corrected_source: corrected,
			placeholders: vec![Placeholder {
				span: SourceSpan {
					start: placeholder_start,
					end: placeholder_end
				},
				description: "direction",
				valid_kinds: DROP_DIRECTION_KINDS
			}]
		}]
	}
}

/// Create a diagnostic for a bare closing delimiter with no matching opener.
/// The fix removes the spurious closer. Emitted from
/// [`analyze_error`](analyze_error) when an expression was expected at the
/// error position but the source has `)`, `]`, or `}` there instead.
///
/// # Parameters
/// - `source`: The current source text.
/// - `closer`: The unmatched closing delimiter character.
/// - `pos`: The byte offset of the closer in the current source.
/// - `offset_map`: For mapping positions back to the original source.
///
/// # Returns
/// A diagnostic with an
/// [`UnopenedDelimiter`](DiagnosticKind::UnopenedDelimiter) kind and a single
/// suggestion that removes the offending character.
fn make_unopened_delimiter(
	source: &str,
	closer: char,
	pos: usize,
	offset_map: &OffsetMap
) -> Diagnostic
{
	let orig_pos = offset_map.to_original(pos);
	let mut corrected = source[..pos].to_string();
	corrected.push_str(&source[pos + closer.len_utf8()..]);
	Diagnostic {
		kind: DiagnosticKind::UnopenedDelimiter { closer },
		span: SourceSpan {
			start: orig_pos,
			end: orig_pos + closer.len_utf8()
		},
		message: format!("unexpected `{}` with no matching opener", closer),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("remove the unopened `{}`", closer),
			corrected_source: corrected,
			placeholders: vec![]
		}]
	}
}

/// Create a diagnostic for an empty or whitespace-only expression. The fix
/// inserts `0` as a placeholder.
///
/// # Parameters
/// - `source`: The current source text (empty or whitespace-only).
///
/// # Returns
/// A diagnostic with an `EmptyExpression` kind and a suggestion with a
/// placeholder for the expression.
fn make_empty_expression(source: &str) -> Diagnostic
{
	let _ = source;
	Diagnostic {
		kind: DiagnosticKind::EmptyExpression,
		span: SourceSpan { start: 0, end: 0 },
		message: "expected expression".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert expression".into(),
			corrected_source: "0".into(),
			placeholders: vec![Placeholder {
				span: SourceSpan { start: 0, end: 1 },
				description: "expression",
				valid_kinds: OPERAND_KINDS
			}]
		}]
	}
}

////////////////////////////////////////////////////////////////////////////////
//                          Semantic error analysis.                          //
////////////////////////////////////////////////////////////////////////////////

/// Run the [validator](Validator) over a parsed [function](Function) and
/// translate any [semantic error](CompilationError) into a [`Diagnostic`].
///
/// Invoked from [`diagnose`] only after the original source parses cleanly —
/// that is, when no fix-and-retry edits have been applied — so that semantic
/// diagnostics reference the user's actual source text rather than a
/// fix-synthesized variant. Returns an empty vector when the validator
/// accepts the function, or a single-element vector when it rejects it.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `source`: The original source text (same lifetime as the AST).
/// - `ast`: The parsed function to validate.
///
/// # Returns
/// The semantic diagnostics produced by the validator, in source order.
fn run_validator<'src>(
	source: &'src str,
	ast: &Function<'src>
) -> Vec<Diagnostic>
{
	match Validator::validate(ast)
	{
		Ok(()) => Vec::new(),
		Err(error) => vec![analyze_semantic_error(source, ast, error)]
	}
}

/// Translate a [semantic error](CompilationError) into a [`Diagnostic`].
///
/// Exhaustively matches the variants [`Validator::validate`] is permitted to
/// produce today. The `ParseError` and `OptimizationFailed` arms are statically
/// unreachable — the parser has already succeeded by the time this helper runs,
/// and the optimizer is not invoked from the diagnostic pipeline — but listing
/// them explicitly (rather than using a wildcard) forces a compile-time
/// consideration whenever a new [`CompilationError`] variant is introduced.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `source`: The original source text.
/// - `ast`: The parsed function that produced the error.
/// - `error`: The semantic error returned by [`Validator::validate`].
///
/// # Returns
/// A [`Diagnostic`] describing the semantic error, with a rename suggestion
/// whose corrected source parses and validates cleanly.
fn analyze_semantic_error<'src>(
	source: &'src str,
	ast: &Function<'src>,
	error: CompilationError<'src>
) -> Diagnostic
{
	match error
	{
		CompilationError::DuplicateParameter {
			name,
			first,
			duplicate
		} => make_duplicate_parameter(source, ast, name, first, duplicate),
		CompilationError::ParseError(_)
		| CompilationError::OptimizationFailed =>
		{
			unreachable!(
				"Validator::validate produces only DuplicateParameter; \
				 ParseError is produced by the parser and OptimizationFailed \
				 by the optimizer, neither of which is reached from \
				 diagnose() at this point"
			)
		}
	}
}

/// Build a [`DuplicateParameter`](DiagnosticKind::DuplicateParameter)
/// diagnostic with a rename suggestion whose corrected source is guaranteed to
/// parse and validate cleanly.
///
/// The primary [`span`](Diagnostic::span) points at the duplicate
/// occurrence; a single [`RelatedLabel`] on [`related`](Diagnostic::related)
/// points at the first occurrence. The suggestion splices a fresh name
/// produced by [`suggest_rename`] into the duplicate span — references in the
/// body are deliberately left alone because there is no safe way to pick
/// which one of them (if any) was intended to refer to a different binding.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `source`: The original source text.
/// - `ast`: The parsed function (used to enumerate in-use parameter names).
/// - `name`: The duplicated parameter name, borrowed from `source`.
/// - `first`: The span of the first occurrence of `name`.
/// - `duplicate`: The span of the duplicate occurrence that triggered the
///   error.
///
/// # Returns
/// A fully-formed [`Diagnostic`] with one rename [`Suggestion`].
fn make_duplicate_parameter<'src>(
	source: &'src str,
	ast: &Function<'src>,
	name: &'src str,
	first: SourceSpan,
	duplicate: SourceSpan
) -> Diagnostic
{
	let parameters = ast.parameters.as_deref().unwrap_or(&[]);
	let fresh = suggest_rename(name, parameters);
	let mut corrected =
		String::with_capacity(source.len() + fresh.len() - name.len());
	corrected.push_str(&source[..duplicate.start]);
	corrected.push_str(&fresh);
	corrected.push_str(&source[duplicate.end..]);
	Diagnostic {
		kind: DiagnosticKind::DuplicateParameter {
			name: name.to_string()
		},
		span: duplicate,
		message: format!(
			"parameter `{}` is declared more than once; review references \
			 to `{}` in the body — one may have meant a different parameter \
			 or an external variable",
			name, name
		),
		related: vec![RelatedLabel {
			span: first,
			message: "first declared here".into()
		}],
		suggestions: vec![Suggestion {
			description: format!("rename duplicate parameter to `{}`", fresh),
			corrected_source: corrected,
			placeholders: vec![]
		}]
	}
}

/// Suggest a fresh parameter name that does not collide with any existing name
/// in `parameters`.
///
/// When the duplicate is a single ASCII letter — the common case, given dice
/// expressions tend to use terse names like `x`, `y`, `a`, `b` — the
/// replacement is the first unused letter encountered by **bumping from the
/// highest attested same-case single-letter parameter** and wrapping through
/// the alphabet. So `x, x` becomes `x, y` (bump `x` → `y`), `a, b, b, c`
/// becomes `a, b, d, c` (bump highest `c` → `d`), `a, x, x` becomes `a, x, y`
/// (bump highest `x` → `y`), and `a, z, z` becomes `a, z, b` (bump `z` wraps
/// past unused `a` which is taken, lands on `b`). Continuing from the user's
/// apparent "trajectory" is usually more natural than filling the earliest
/// alphabetic gap, which would produce `a, x, b` in the third example.
///
/// When the duplicate is a multi-character name, or all 26 letters in the
/// matching case are already in use, the replacement falls through to a `newN`
/// form (`new0`, `new1`, …), where `N` is the first non-negative integer for
/// which the resulting name is unused. The `newN` loop likewise skips over
/// names the user has already adopted, so `abc, new0, abc` becomes `abc, new0,
/// new1` and `abc, new0, new1, abc` becomes `abc, new0, new1, new2`.
///
/// # Parameters
/// - `duplicate`: The duplicated parameter name.
/// - `parameters`: The function's parameter list, used to determine which names
///   are already in use.
///
/// # Returns
/// A fresh parameter name that does not appear in `parameters`.
fn suggest_rename(duplicate: &str, parameters: &[Parameter<'_>]) -> String
{
	let used: HashSet<&str> = parameters.iter().map(|p| p.name).collect();
	let mut chars = duplicate.chars();
	if let (Some(c), None) = (chars.next(), chars.next())
		&& c.is_ascii_alphabetic()
	{
		let start = if c.is_ascii_uppercase() { b'A' } else { b'a' };
		// The highest attested same-case single-letter parameter. The duplicate
		// itself is always attested, so `highest` is at least `c`.
		let highest = parameters
			.iter()
			.filter_map(|p| {
				let mut cs = p.name.chars();
				match (cs.next(), cs.next())
				{
					(Some(ch), None)
						if ch.is_ascii_alphabetic()
							&& ch.is_ascii_uppercase()
								== c.is_ascii_uppercase() =>
					{
						Some(ch as u8)
					},
					_ => None
				}
			})
			.max()
			.unwrap_or(c as u8);
		// Iterate the 25 other letters by bumping forward from `highest` and
		// wrapping through the alphabet. `offset == 26` would loop back to
		// `highest`, which is attested, so 1..26 is sufficient.
		for offset in 1u8..26
		{
			let code = start + ((highest - start + offset) % 26);
			let candidate = (code as char).to_string();
			if !used.contains(candidate.as_str())
			{
				return candidate;
			}
		}
	}
	for i in 0usize..
	{
		let candidate = format!("new{}", i);
		if !used.contains(candidate.as_str())
		{
			return candidate;
		}
	}
	unreachable!("cannot exhaust usize worth of `newN` candidates")
}

////////////////////////////////////////////////////////////////////////////////
//                             Position mapping.                              //
////////////////////////////////////////////////////////////////////////////////

/// Tracks cumulative byte offset adjustments from applied fixes, enabling
/// positions in a modified source string to be mapped back to positions in the
/// original source.
#[derive(Debug, Clone)]
struct OffsetMap
{
	/// Each entry is `(modified_pos, delta)`, where `delta` is the number of
	/// bytes inserted (positive) or removed (negative) at `modified_pos`.
	adjustments: Vec<(usize, isize)>
}

impl OffsetMap
{
	/// Create a new, empty offset map with no adjustments recorded.
	///
	/// # Returns
	/// An offset map that maps all positions to themselves.
	fn new() -> Self
	{
		Self {
			adjustments: Vec::new()
		}
	}

	/// Record an adjustment: `delta` bytes were inserted (positive) or removed
	/// (negative) at `modified_pos` in the current source.
	///
	/// # Parameters
	/// - `modified_pos`: The byte offset in the current (modified) source where
	///   the edit occurred.
	/// - `delta`: The number of bytes inserted (positive) or removed
	///   (negative).
	fn record(&mut self, modified_pos: usize, delta: isize)
	{
		self.adjustments.push((modified_pos, delta));
	}

	/// Map a position in the current (modified) source back to the
	/// corresponding position in the original source.
	///
	/// # Parameters
	/// - `modified_pos`: A byte offset in the current source.
	///
	/// # Returns
	/// The corresponding byte offset in the original source.
	fn to_original(&self, modified_pos: usize) -> usize
	{
		let mut pos = modified_pos as isize;
		// Walk adjustments in reverse to undo them.
		for &(adj_pos, delta) in self.adjustments.iter().rev()
		{
			if modified_pos >= adj_pos
			{
				pos -= delta;
			}
		}
		pos.max(0) as usize
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                The doctor.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Diagnose a source string, collecting all parse errors, semantic errors, and
/// suggested fixes.
///
/// The doctor runs two sequential passes. Syntactic errors go through a
/// fix-and-retry loop: parse, detect the error, generate a fix, apply the fix,
/// and re-parse. This continues until parsing succeeds (all errors collected)
/// or an unfixable error is encountered. When — and only when — the original
/// source parsed cleanly, the doctor then runs the semantic [`Validator`]
/// against the AST and translates any error into a corresponding
/// [`Diagnostic`]. The semantic pass is skipped after any fix, because
/// attaching semantic diagnostics to fix-synthesized characters the user never
/// typed would mislead them about what their source actually says.
///
/// ```mermaid
/// graph TD
///     S["Source Code"] --> P["Parser::parse"]
///     P -->|OK| V["Validator::validate"]
///     P -->|Err| A["analyze_error"]
///     A --> F["Apply fix"]
///     F --> OM["OffsetMap records edit"]
///     OM --> P
///     V -->|OK| DONE["DiagnoseResult"]
///     V -->|Err| SEM["analyze_semantic_error"]
///     SEM --> DONE
///     A -.unfixable.-> DONE
///     style S fill:#f9f,stroke:#333,color:#000
///     style DONE fill:#9f9,stroke:#333,color:#000
///     style P fill:#ffd,stroke:#333,color:#000
///     style V fill:#ffd,stroke:#333,color:#000
/// ```
///
/// # Parameters
/// - `source`: The source string to diagnose.
///
/// # Returns
/// A [`DiagnoseResult`] containing all diagnostics and, if all parse errors
/// were fixable, the corrected source string. Semantic diagnostics never
/// prevent the corrected-source return: a duplicate-parameter program
/// produces a diagnostic *and* a `corrected_source` equal to the original
/// input, because the source already parses.
///
/// # Performance
/// Designed for keystroke-speed execution. Each iteration involves a single
/// `nom` parse (microseconds for typical dice expressions) and a string fixup.
/// The loop runs at most `O(n)` times where `n` is the number of errors, and
/// the semantic pass runs at most once per call.
///
/// # Examples
/// Single error with an unambiguous fix:
///
/// ```rust
/// use xdy::diagnostics::{DiagnosticKind, diagnose};
///
/// let result = diagnose("{x");
/// assert_eq!(result.diagnostics.len(), 1);
/// assert!(matches!(
///     result.diagnostics[0].kind,
///     DiagnosticKind::UnclosedDelimiter { opener: '{', .. }
/// ));
/// assert_eq!(
///     result.corrected_source.as_deref(),
///     Some("{x}")
/// );
/// ```
///
/// Multiple errors collected via fix-and-retry:
///
/// ```rust
/// use xdy::diagnostics::{DiagnosticKind, diagnose};
///
/// let result = diagnose("3D6 + + 1D3");
/// assert_eq!(result.diagnostics.len(), 1);
/// assert!(matches!(
///     result.diagnostics[0].kind,
///     DiagnosticKind::MissingRightOperand { operator: '+' }
/// ));
/// // The corrected source inserts `0` as a placeholder operand.
/// assert!(result.corrected_source.is_some());
/// // The corrected source parses cleanly.
/// assert!(
///     xdy::Parser::parse(
///         result.corrected_source.as_ref().unwrap()
///     ).is_ok()
/// );
/// ```
///
/// A semantic diagnostic on a program that otherwise parses cleanly:
///
/// ```rust
/// use xdy::diagnostics::{DiagnosticKind, diagnose};
///
/// let result = diagnose("x, x: {x}");
/// assert_eq!(result.diagnostics.len(), 1);
/// assert!(matches!(
///     result.diagnostics[0].kind,
///     DiagnosticKind::DuplicateParameter { .. }
/// ));
/// // Semantic diagnostics don't prevent `corrected_source`; the original
/// // source already parses, so it is returned verbatim.
/// assert_eq!(result.corrected_source.as_deref(), Some("x, x: {x}"));
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub fn diagnose(source: &str) -> DiagnoseResult
{
	// Fast path: if the source parses cleanly, skip the fix-and-retry loop and
	// run the semantic pass directly on the AST. Fixing up a syntactically
	// invalid source first and then reporting a semantic error against the
	// fix-synthesized text would attach diagnostics to characters the user
	// never typed, which is confusing — so semantic checks run only on the
	// unmodified original.
	if let Ok(ast) = Parser::parse(source)
	{
		return DiagnoseResult {
			diagnostics: run_validator(source, &ast),
			corrected_source: Some(source.to_string())
		};
	}

	let mut current_source = source.to_string();
	let mut diagnostics = Vec::new();
	let mut offset_map = OffsetMap::new();
	// Safety valve: at most one fix per character, plus a generous baseline for
	// empty and very short inputs.
	let max_iterations = source.len() + 16;

	for _ in 0..max_iterations
	{
		match Parser::parse(&current_source)
		{
			Ok(_) =>
			{
				return DiagnoseResult {
					diagnostics,
					corrected_source: Some(current_source)
				};
			},
			Err(error) =>
			{
				let diag = analyze_error(&current_source, &error, &offset_map);
				if let Some(suggestion) = diag.suggestions.first()
				{
					let old_len = current_source.len() as isize;
					let new_source = suggestion.corrected_source.clone();
					let new_len = new_source.len() as isize;
					let delta = new_len - old_len;
					// Record the adjustment at position 0 (whole-string
					// replacement). This is a simplification; a more precise
					// approach would record the specific edit position.
					let error_pos = error.errors[0].0.location_offset();
					offset_map.record(error_pos, delta);
					current_source = new_source;
					diagnostics.push(diag);
				}
				else
				{
					// Unfixable error.
					diagnostics.push(diag);
					return DiagnoseResult {
						diagnostics,
						corrected_source: None
					};
				}
			}
		}
	}

	// Should not reach here, but if we do, return what we have.
	DiagnoseResult {
		diagnostics,
		corrected_source: None
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           OffsetMap unit tests.                            //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests
{
	use super::OffsetMap;

	/// No adjustments → identity mapping.
	#[test]
	fn test_offset_map_identity()
	{
		let map = OffsetMap::new();
		assert_eq!(map.to_original(0), 0);
		assert_eq!(map.to_original(5), 5);
		assert_eq!(map.to_original(100), 100);
	}

	/// Single insertion: 2 bytes inserted at position 3.
	/// Original: `abc_efg` (7 bytes)
	/// Modified: `abc00_efg` (9 bytes)
	/// Position 0–2 in modified → 0–2 in original (before insertion).
	/// Position 3+ in modified → shifted back by 2.
	#[test]
	fn test_offset_map_single_insertion()
	{
		let mut map = OffsetMap::new();
		map.record(3, 2); // inserted 2 bytes at position 3
		// Positions before the insertion are unchanged.
		assert_eq!(map.to_original(0), 0);
		assert_eq!(map.to_original(2), 2);
		// Positions at or after the insertion shift back by 2.
		assert_eq!(map.to_original(3), 1);
		assert_eq!(map.to_original(5), 3);
		assert_eq!(map.to_original(8), 6);
	}

	/// Single deletion: 2 bytes removed at position 3.
	/// Original: `abcXX_efg` (9 bytes)
	/// Modified: `abc_efg` (7 bytes)
	/// Position 3+ in modified → shifted forward by 2.
	#[test]
	fn test_offset_map_single_deletion()
	{
		let mut map = OffsetMap::new();
		map.record(3, -2); // removed 2 bytes at position 3
		// Positions before the deletion are unchanged.
		assert_eq!(map.to_original(0), 0);
		assert_eq!(map.to_original(2), 2);
		// Positions at or after the deletion shift forward by 2.
		assert_eq!(map.to_original(3), 5);
		assert_eq!(map.to_original(5), 7);
	}

	/// Multiple insertions: simulates the fix-and-retry loop inserting `0`
	/// at two different positions.
	/// Original: `a + + b -` (9 bytes)
	/// After fix 1: `a + 0 + b -` (11 bytes, inserted 2 at pos 4)
	/// After fix 2: `a + 0 + b - 0` (13 bytes, inserted 2 at pos 12)
	#[test]
	fn test_offset_map_multiple_insertions()
	{
		let mut map = OffsetMap::new();
		map.record(4, 2); // first fix: inserted 2 bytes at position 4
		map.record(12, 2); // second fix: inserted 2 bytes at position 12
		// Position before both insertions.
		assert_eq!(map.to_original(0), 0);
		assert_eq!(map.to_original(3), 3);
		// Position after first insertion but before second.
		assert_eq!(map.to_original(6), 4);
		// Position after both insertions.
		assert_eq!(map.to_original(13), 9);
	}

	/// Position before any adjustment is unaffected.
	#[test]
	fn test_offset_map_position_before_adjustment()
	{
		let mut map = OffsetMap::new();
		map.record(10, 5); // inserted 5 bytes at position 10
		// Positions 0–9 are before the adjustment.
		for i in 0..10
		{
			assert_eq!(map.to_original(i), i);
		}
	}

	/// Mapping clamps to zero for extreme negative adjustments.
	#[test]
	fn test_offset_map_clamp_to_zero()
	{
		let mut map = OffsetMap::new();
		map.record(0, 10); // inserted 10 bytes at position 0
		// Position 0 in modified maps to original 0 (clamped, not -10).
		assert_eq!(map.to_original(0), 0);
		// Position 5 in modified would be -5, clamped to 0.
		assert_eq!(map.to_original(5), 0);
		// Position 10 in modified maps to original 0.
		assert_eq!(map.to_original(10), 0);
		// Position 11 in modified maps to original 1.
		assert_eq!(map.to_original(11), 1);
	}
}
