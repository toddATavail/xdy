//! # Diagnostics
//!
//! Herein is the diagnostic layer for the `xDy` parser and
//! [validator](Validator). The [`diagnose`] function analyzes a source string
//! by parsing it once in the parser's recovery mode, which diagnoses and
//! repairs every syntactic error in turn; it runs the semantic validator
//! instead whenever the source parses cleanly. This produces structured
//! [`Diagnostic`]s with specific error kinds, source spans highlighting,
//! human-readable messages, secondary [related labels](RelatedLabel) for
//! cross-referenced positions, and suggested fixes — edits of the original
//! source, in the manner of an editor's quick fixes, with highlighted
//! placeholder regions for the defaults of ambiguous ones.
//!
//! # Architecture
//!
//! The diagnostic layer is a consumer of the parser's
//! [`ParseError`](crate::parser::ParseError) and the validator's
//! [`CompilationError`], not part of either. The [`compile`](crate::compile)
//! and [`compile_unoptimized`](crate::compile_unoptimized) functions continue
//! to deal in the typed errors directly; diagnostics are a separate concern for
//! clients that want richer error information (e.g., for IDE integration,
//! syntax highlighting on every keystroke, or autocompletion).
//!
//! # Error Recovery Strategy
//!
//! The doctor runs one of two passes. If the source parses cleanly, a
//! **semantic pass** runs the [`Validator`] against the AST. Otherwise, a
//! **recovering parse** collects the syntactic errors. The [rustdoc for
//! `diagnose`](diagnose) carries a Mermaid rendering of the complete pipeline;
//! the ASCII sketch below summarizes the flow for casual skimming.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ Parser::parse(source)                                     │
//! │   ├─ success → Validator::validate(ast) → diagnostics     │
//! │   └─ error   → Parser::parse_recovering(blanked, doctor)  │
//! │                  └─ each failure → diagnose → repair      │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! The parser never edits the source. Wherever the recovering parse cannot
//! continue, it consults the doctor, which diagnoses the failure from where
//! it lies in the grammar, the expectations of its error, and the text around
//! it. If the diagnostic has a fix, the doctor answers the repair that makes
//! the parse continue as though the source had been edited by the fix, e.g.,
//! as though a missing `)` were present. The parser reads stray whitespace,
//! which may not separate tokens, e.g., U+00A0 NO-BREAK SPACE, as spaces of
//! the same length, and the doctor diagnoses each stray itself. Every position
//! is a position of the original source, so the spans of diagnostics and the
//! edits of suggestions need no mapping, and the whole diagnosis takes time
//! linear in the length of the source, however many errors it holds. The parse
//! ends when it succeeds or when an unfixable error is encountered.
//!
//! The semantic pass runs **only when the original source parsed cleanly** —
//! that is, when no fixes were applied. Semantic diagnostics (e.g., duplicate
//! parameters) point at spans the user actually typed; running the validator on
//! a fix-synthesized source would attach diagnostics to characters the user
//! never wrote, so semantic checks wait for a syntactically valid source.

use std::{
	borrow::Cow,
	collections::HashSet,
	fmt::{self, Display, Formatter},
	iter
};

use crate::{
	CompilationError, Parser, Validator,
	ast::{Event, Expression, Function, Node, Walk},
	parser::{
		FailureSite, Recovery, Repair, Site, canonical_name,
		is_bare_word_continue, is_bare_word_start, is_token_space,
		unclosed_name_ends
	},
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

	/// A delimited construct is closed, but lacks the expression, name, or
	/// face within it.
	///
	/// # Examples
	/// `( )`, `{ }`, `[1: ]`, `3D[ ]`
	MissingExpression
	{
		/// The opening delimiter character.
		opener: char
	},

	/// A closing delimiter has no matching open.
	///
	/// # Examples
	/// `)3D6`, `]3D6`, `}3D6`
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

	/// A drop clause follows something other than a dice expression.
	///
	/// # Examples
	/// `3 drop lowest`, `1 + drop highest`
	MisplacedDropClause,

	/// A parameter definition is missing the `:` separator and body.
	///
	/// # Examples
	/// `{x}, {y}`, `{x}, {y} {x} + {y}`
	IncompleteParameterDefinition,

	/// A comma in the formal parameters is followed by another comma or by
	/// the `:`, rather than by a parameter.
	///
	/// # Examples
	/// `{x},: 1`, `{x}, , {y}: 1`
	MissingParameter,

	/// A valid expression is followed by unparsable input.
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
	/// above — which are produced from
	/// [`ParseError`](crate::parser::ParseError) during the recovering parse —
	/// this kind is produced by the [`Validator`] semantic pass after a
	/// clean parse. The diagnostic's primary [`span`](Diagnostic::span) points
	/// at the duplicate occurrence; the [`related`](Diagnostic::related) list
	/// carries the span of the first occurrence.
	///
	/// # Examples
	/// `{x}, {x}: {x}`, `{a}, {b}, {b}: {a}`
	DuplicateParameter
	{
		/// The duplicated parameter name.
		name: String
	},

	/// A [local binding](crate::ast::Binding) uses a name that is already
	/// declared as a formal parameter. Not produced by the recovering parse —
	/// this kind is produced by the [`Validator`] semantic pass after a clean
	/// parse. The diagnostic's primary [`span`](Diagnostic::span) points at the
	/// binding-site name; the [`related`](Diagnostic::related) list carries the
	/// span of the colliding parameter declaration.
	///
	/// # Examples
	/// `{x}: {x}@(3D6) + {x}`
	BindingCollidesWithParameter
	{
		/// The colliding name.
		name: String
	},

	/// The same name is bound more than once by
	/// [local bindings](crate::ast::Binding) within a single function body.
	/// Not produced by the recovering parse — this kind is produced by the
	/// [`Validator`] semantic pass after a clean parse. The diagnostic's
	/// primary [`span`](Diagnostic::span) points at the duplicate binding-site
	/// name; the [`related`](Diagnostic::related) list carries the span of the
	/// first binding.
	///
	/// # Examples
	/// `{x}@(3D6) + {x}@(1D4)`, `{x}@({x}@(1))` (a binding nested within the
	/// bound expression of a binding of the same name)
	DuplicateBinding
	{
		/// The rebound name.
		name: String
	},

	/// A [variable reference](crate::ast::Variable) appears lexically before
	/// the [binding](crate::ast::Binding) that introduces its name. Not
	/// produced by the recovering parse — this kind is produced by the
	/// [`Validator`] semantic pass after a clean parse. The diagnostic's
	/// primary [`span`](Diagnostic::span) points at the offending reference;
	/// the [`related`](Diagnostic::related) list carries the span of the later
	/// binding site.
	///
	/// # Examples
	/// `{x} + {x}@(3D6)`, `{x}@({x})` (self-reference inside the bound
	/// expression)
	UseBeforeBind
	{
		/// The name that was referenced before being bound.
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
			Self::MissingExpression { opener } =>
			{
				write!(f, "missing expression in `{}`", opener)
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
			Self::MisplacedDropClause => write!(f, "misplaced drop clause"),
			Self::IncompleteParameterDefinition =>
			{
				write!(f, "incomplete parameter definition")
			},
			Self::MissingParameter => write!(f, "missing parameter"),
			Self::TrailingInput => write!(f, "trailing input"),
			Self::EmptyExpression => write!(f, "empty expression"),
			Self::UnexpectedToken => write!(f, "unexpected token"),
			Self::UnexpectedEof => write!(f, "unexpected end of input"),
			Self::DuplicateParameter { name } =>
			{
				write!(f, "duplicate parameter `{}`", name)
			},
			Self::BindingCollidesWithParameter { name } =>
			{
				write!(
					f,
					"local binding `{}` collides with formal parameter",
					name
				)
			},
			Self::DuplicateBinding { name } =>
			{
				write!(f, "duplicate local binding `{}`", name)
			},
			Self::UseBeforeBind { name } =>
			{
				write!(f, "reference to `{}` precedes its binding", name)
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

/// A suggested fix for a diagnostic: a set of [edits](Edit) to the original
/// source, in the manner of a quick fix in an editor.
///
/// # Notes
/// Each suggestion stands alone: its edits apply to the original source passed
/// to [`diagnose`], whether or not the fixes of other diagnostics are applied
/// too. The [`corrected_source`](DiagnoseResult::corrected_source) of the
/// [result](DiagnoseResult) is the source with the first suggestion of every
/// diagnostic applied. Where two fixes interact, e.g., where one removes
/// trailing input that begins with text that the other inserted, their edits
/// may overlap or abut, and `corrected_source` resolves the interaction.
///
/// For unambiguous fixes (e.g., `{x` → `{x}`), no edit has
/// [placeholders](Edit::placeholders). For ambiguous fixes (e.g., `3D` →
/// `3D6`), placeholders identify the inserted defaults that the user may want
/// to replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion
{
	/// A human-readable description of the fix.
	pub description: String,

	/// The edits that make the fix, in ascending order of position. They do
	/// not overlap, though insertions may abut.
	pub edits: Vec<Edit>
}

impl Suggestion
{
	/// Apply the suggestion to the original source.
	///
	/// # Parameters
	/// - `source`: The source string originally passed to [`diagnose`].
	///
	/// # Returns
	/// The source with the [edits](Suggestion::edits) applied.
	///
	/// # Panics
	/// If an edit's [span](Edit::span) is out of bounds or does not fall on
	/// character boundaries of `source`, i.e., if `source` is not the source
	/// that was diagnosed.
	///
	/// # Examples
	/// ```rust
	/// use xdy::diagnostics::diagnose;
	///
	/// let result = diagnose("3D");
	/// let suggestion = &result.diagnostics[0].suggestions[0];
	/// assert_eq!(suggestion.apply("3D"), "3D6");
	/// ```
	pub fn apply(&self, source: &str) -> String
	{
		let mut corrected = String::with_capacity(source.len());
		let mut copied = 0;
		for edit in &self.edits
		{
			corrected.push_str(&source[copied..edit.span.start]);
			corrected.push_str(&edit.replacement);
			copied = edit.span.end;
		}
		corrected.push_str(&source[copied..]);
		corrected
	}
}

/// A replacement of one region of the original source, as part of a
/// [suggestion](Suggestion).
///
/// # Examples
/// Inserting `)` at the end of `(1`:
///
/// ```rust
/// use xdy::{SourceSpan, diagnostics::diagnose};
///
/// let result = diagnose("(1");
/// let edit = &result.diagnostics[0].suggestions[0].edits[0];
/// assert_eq!(edit.span, SourceSpan { start: 2, end: 2 });
/// assert_eq!(edit.replacement, ")");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit
{
	/// The byte range of the original source to replace. It is empty for an
	/// insertion.
	pub span: SourceSpan,

	/// The replacement text. It is empty for a deletion.
	pub replacement: String,

	/// The regions of the [replacement](Edit::replacement) that are defaults,
	/// which the user may want to replace. Empty for unambiguous fixes.
	pub placeholders: Vec<Placeholder>
}

/// A placeholder region in the [replacement](Edit::replacement) text of an
/// [edit](Edit).
///
/// A sophisticated UI can highlight these regions and present the
/// `valid_kinds` as a picker, allowing the user to type over the default
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placeholder
{
	/// The byte range of the placeholder, relative to the start of the
	/// [replacement](Edit::replacement) text that contains it.
	pub span: SourceSpan,

	/// A description of what belongs here (e.g., "face count", "operand").
	pub description: &'static str,

	/// The kinds of expressions that are valid at this position.
	pub valid_kinds: &'static [&'static str]
}

/// The result of diagnosing a source string.
///
/// # Notes
/// If all syntactic errors are fixable, `corrected_source` contains a
/// parseable source string with the first suggestion of each applied. If any
/// syntactic error is unfixable, `corrected_source` is `None`. Semantic
/// diagnostics arise only from a source that already parses, and their
/// suggestions are never applied, so `corrected_source` is then the original
/// source.
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

/// Valid expression kinds for positions that only a dice expression may fill.
const DICE_KINDS: &[&str] = &["dice expression"];

/// Valid kinds for drop direction.
const DROP_DIRECTION_KINDS: &[&str] = &["lowest", "highest"];

////////////////////////////////////////////////////////////////////////////////
//                              Error analysis.                               //
////////////////////////////////////////////////////////////////////////////////

/// The doctor's [recovery policy](Recovery). The parser consults it at each
/// [failure](FailureSite) of a single recovering parse of the source; it
/// diagnoses the failure and, if it can, answers the [repair](Repair) that
/// makes the parse continue as though the source had been edited by the first
/// suggestion of the diagnostic.
///
/// The parser reads the source with its [stray whitespace](stray_whitespace)
/// replaced by spaces of the same length, which the doctor diagnoses itself, so
/// every position is a position of the original source, and the diagnostics
/// and their edits need no mapping. To keep the whole diagnosis linear in the
/// length of the source, the doctor inspects the text around a failure in
/// constant amortized time: the whitespace before the failure, which it
/// remembers for the next failure at the same position, and the token at the
/// failure, which the repair consumes. Only a failure after formal parameters,
/// which happens at most once, inspects the parameters in full, and only a
/// variable without its `}` inspects its name in full, which ends where the
/// next name can begin.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
#[derive(Debug)]
struct Doctor<'src>
{
	/// The source text that the parser reads: the original source, with its
	/// stray whitespace replaced by spaces.
	source: &'src str,

	/// The original source.
	original: &'src str,

	/// The byte offsets of the start and end of each character of [stray
	/// whitespace](stray_whitespace) in the original source, in order.
	strays: Vec<(usize, usize)>,

	/// The byte offset of the end of the source without its trailing
	/// whitespace. A failure at or beyond it is at the end of the input.
	content_end: usize,

	/// The diagnostics, in the order of the failures.
	diagnostics: Vec<Diagnostic>,

	/// The edits of the first suggestions of the diagnostics, in the order of
	/// the failures.
	fixes: Vec<Edit>,

	/// The compound fix whose tokens the parser is still supplying, if any.
	compound: Option<Compound>,

	/// The state of the diagnosis.
	state: State,

	/// The last position whose preceding character the doctor looked for, and
	/// that character and its byte offset, if any.
	preceding: Option<(usize, Option<(char, usize)>)>,

	/// The byte offsets of the start and end of the last [bare
	/// word](crate::parser::bare_word) that the doctor scanned.
	bare_word: Option<(usize, usize)>,

	/// Whether a fix removes trailing input.
	trailing: bool,

	/// The byte offset of a misplaced `drop`, before which a fix supplied a
	/// dice expression, if the parser has yet to read the `d` of the `drop`
	/// as its dice operator, and fail at its faces.
	reread: Option<usize>,

	/// The byte offsets of the start and end of the last run of unopened
	/// closing delimiters that fixes discarded, separated by whitespace only.
	skipped: Option<(usize, usize)>
}

/// The state of a [diagnosis](Doctor).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum State
{
	/// Every failure so far has a fix.
	Fixing,

	/// A fix replaced the rest of the source, so no later failure matters.
	Supplanted,

	/// A failure has no fix.
	Unfixable
}

/// Where a [bare identifier](DiagnosticKind::BareIdentifier) lies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Place
{
	/// At the start of the input, where formal parameters may begin.
	Start,

	/// After the `,` of a formal parameter.
	Parameters,

	/// Within an expression.
	Expression
}

/// The [doctor's](Doctor) analysis of a failure.
#[derive(Debug)]
enum Analysis
{
	/// A diagnostic, and the repair that corresponds to its first suggestion,
	/// or [`Stop`](Repair::Stop) if it has none.
	Diagnosed(Diagnostic, Repair),

	/// A diagnostic whose first suggestion replaces the rest of the source,
	/// and whose repair is therefore [`Fix`](Repair::Fix), which supplies the
	/// first token of the replacement.
	Supplanting(Diagnostic),

	/// The first token of a compound fix, whose repair is
	/// [`Fix`](Repair::Fix).
	Compound(Compound)
}

impl<'src> Doctor<'src>
{
	/// Create a doctor for the specified source.
	///
	/// # Parameters
	/// - `source`: The source text that the parser reads: the original source,
	///   with its stray whitespace replaced by spaces of the same length.
	/// - `original`: The original source.
	/// - `strays`: The byte offsets of the start and end of each character of
	///   [stray whitespace](stray_whitespace) in the original source, in order.
	///
	/// # Returns
	/// The doctor.
	fn new(
		source: &'src str,
		original: &'src str,
		strays: Vec<(usize, usize)>
	) -> Self
	{
		Self {
			source,
			original,
			strays,
			content_end: source.trim_end_matches(is_token_space).len(),
			diagnostics: Vec::new(),
			fixes: Vec::new(),
			compound: None,
			state: State::Fixing,
			preceding: None,
			bare_word: None,
			trailing: false,
			reread: None,
			skipped: None
		}
	}

	/// Finish the diagnosis, once the recovering parse is over.
	///
	/// # Parameters
	/// - `recovered`: Whether the recovering parse succeeded.
	///
	/// # Returns
	/// The result of the diagnosis.
	fn finish(mut self, recovered: bool) -> DiagnoseResult
	{
		self.finish_compound();
		// The parser declines a repair that it cannot make, in which case the
		// parse fails although the doctor fixed every failure.
		let fixed = match self.state
		{
			State::Fixing => recovered,
			State::Supplanted => true,
			State::Unfixable => false
		};
		self.diagnose_strays();
		let corrected_source = fixed.then(|| {
			// An insertion precedes a replacement at the same position, whose
			// text it therefore cannot have followed.
			self.fixes
				.sort_by_key(|edit| (edit.span.start, edit.span.end));
			let mut corrected = apply(self.original, &self.fixes);
			// The removal of trailing input leaves no trailing whitespace, not
			// even whitespace that an earlier fix inserted.
			if self.trailing
			{
				corrected
					.truncate(corrected.trim_end_matches(is_token_space).len());
			}
			corrected
		});
		DiagnoseResult {
			diagnostics: self.diagnostics,
			corrected_source
		}
	}

	/// Diagnose the [stray whitespace](stray_whitespace), and record the fixes,
	/// but not where another fix replaces it, e.g., within trailing input that
	/// a fix removes. The diagnostics of the strays join the others in source
	/// order: each precedes the first other diagnostic that begins after it.
	fn diagnose_strays(&mut self)
	{
		let mut replaced = self
			.fixes
			.iter()
			.filter(|edit| edit.span.start < edit.span.end)
			.map(|edit| (edit.span.start, edit.span.end))
			.collect::<Vec<_>>();
		replaced.sort_unstable();
		let mut replaced = replaced.into_iter().peekable();
		// The greatest end of the regions that begin at or before a stray.
		let mut reach = 0;
		let mut strays = Vec::new();
		for &(start, end) in &self.strays
		{
			while let Some((_, region_end)) =
				replaced.next_if(|&(region_start, _)| region_start <= start)
			{
				reach = reach.max(region_end);
			}
			if reach < end
			{
				let diagnostic =
					make_stray_whitespace(self.original, start, end);
				self.fixes.extend(diagnostic.suggestions[0].edits.clone());
				strays.push(diagnostic);
			}
		}
		let mut strays = strays.into_iter().peekable();
		let others = std::mem::take(&mut self.diagnostics);
		for diagnostic in others
		{
			self.diagnostics.extend(iter::from_fn(|| {
				strays.next_if(|stray| stray.span.start < diagnostic.span.start)
			}));
			self.diagnostics.push(diagnostic);
		}
		self.diagnostics.extend(strays);
	}

	/// Record the diagnostic of the pending compound fix, if any.
	fn finish_compound(&mut self)
	{
		if let Some(compound) = self.compound.take()
		{
			self.record(compound.diagnostic(self.source));
		}
	}

	/// Record a diagnostic, and the edits of its first suggestion, if any.
	///
	/// # Parameters
	/// - `diagnostic`: The diagnostic.
	fn record(&mut self, diagnostic: Diagnostic)
	{
		if let Some(first) = diagnostic.suggestions.first()
		{
			self.fixes.extend(first.edits.iter().cloned());
		}
		self.diagnostics.push(diagnostic);
	}

	/// Answer the nearest character before a position that is not whitespace,
	/// and its byte offset. The answer for the last position asked about is
	/// remembered, since several failures may lie at the same position.
	/// Otherwise, since every failure but one at the end of the input lies at
	/// a character that is not whitespace, the whitespace scanned for
	/// different failures does not overlap.
	///
	/// # Parameters
	/// - `pos`: The byte offset of the position.
	///
	/// # Returns
	/// The character and its byte offset, if any.
	fn preceding(&mut self, pos: usize) -> Option<(char, usize)>
	{
		if let Some((at, preceding)) = self.preceding
			&& at == pos
		{
			return preceding;
		}
		let before = |end: usize| {
			self.source[..end]
				.char_indices()
				.rev()
				.find(|(_, c)| !is_token_space(*c))
				.map(|(i, c)| (c, i))
		};
		// Look past discarded delimiters, as though they were whitespace.
		let preceding = match (before(pos), self.skipped)
		{
			(Some((_, i)), Some((start, end))) if start <= i && i < end =>
			{
				before(start)
			},
			(preceding, _) => preceding
		};
		self.preceding = Some((pos, preceding));
		preceding
	}

	/// Record that a fix discards an unopened closing delimiter.
	///
	/// # Parameters
	/// - `start`: The byte offset of the delimiter.
	/// - `end`: The byte offset of the end of the delimiter.
	fn skip(&mut self, start: usize, end: usize)
	{
		self.skipped = match self.skipped
		{
			Some((first, last))
				if self.source[last..start]
					.trim_matches(is_token_space)
					.is_empty() =>
			{
				Some((first, end))
			},
			_ => Some((start, end))
		};
	}

	/// Answer the byte offset of the end of the [bare
	/// word](crate::parser::bare_word) that begins at a position, if any. A
	/// bare word that begins within the last bare word scanned ends where that
	/// one does, so its end is not scanned again.
	///
	/// # Parameters
	/// - `pos`: The byte offset of the position.
	///
	/// # Returns
	/// The byte offset of the end of the bare word, if one begins at `pos`.
	fn bare_word_end(&mut self, pos: usize) -> Option<usize>
	{
		if !self.source[pos..].starts_with(is_bare_word_start)
		{
			return None;
		}
		if let Some((start, end)) = self.bare_word
			&& start <= pos
			&& pos < end
		{
			return Some(end);
		}
		let remaining = &self.source[pos..];
		let end = pos
			+ remaining
				.find(|c: char| !is_bare_word_continue(c))
				.unwrap_or(remaining.len());
		self.bare_word = Some((pos, end));
		Some(end)
	}

	/// Analyze a failure, choosing the diagnostic and repair from the site of
	/// the failure, the expectations of its error, and the text around it.
	///
	/// # Parameters
	/// - `site`: The failure.
	///
	/// # Returns
	/// The analysis.
	fn analyze(&mut self, site: &FailureSite<'src>) -> Analysis
	{
		let source = self.source;
		let pos = site.position();
		let at_eof = pos >= self.content_end;
		let is_goal = matches!(site.site, Site::Goal { .. });

		// Collect all expectations at the position of the failure.
		let expectations = site
			.error
			.errors
			.iter()
			.flat_map(|(_, kind)| kind.expectations())
			.collect::<Vec<_>>();

		// Check for specific patterns based on the failure and the source.
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

		// Pattern: closing brace of a variable expected → unclosed delimiter.
		// The name reads everything up to the next brace or the end of the
		// input, so break it early, before an operator or delimiter, if it has
		// one; otherwise, close it at the failure. The name ends at the
		// failure, and the next one begins at or after it, so no two scans of
		// names overlap.
		if let Site::Closer {
			opener: opener_pos,
			closer: '}'
		} = site.site
		{
			let name = source[opener_pos + 1..pos].trim_start();
			let start = pos - name.len();
			let ends = unclosed_name_ends(name)
				.into_iter()
				.flatten()
				.map(|length| start + length)
				.collect::<Vec<_>>();
			let repair = ends
				.first()
				.map_or(Repair::Fix, |&end| Repair::Break { end });
			return Analysis::Diagnosed(
				make_unclosed_brace(source, opener_pos, &ends, pos, at_eof),
				repair
			);
		}

		// Pattern: closing delimiter expected at EOF → unclosed delimiter.
		if at_eof
			&& let Some((opener, closer)) = expects_delimiter
			&& let Site::Closer {
				opener: opener_pos, ..
			} = site.site
		{
			return Analysis::Diagnosed(
				make_unclosed_delimiter(opener, closer, opener_pos, pos),
				Repair::Fix
			);
		}

		// Pattern: a drop clause that follows no dice expression, e.g.,
		// `3 drop lowest`, whose `d` the parser read as a dice operator, or
		// `1 + drop lowest`, where an expression was expected, rather than a
		// bare name that splits `drop` at its `d`. After an operand, the parse
		// supplies faces and reads the `drop` again. Where an expression was
		// expected, the parse supplies an operand, which it reads as the count
		// of a dice expression whose operator is the `d` of `drop`, and the
		// faces then fail, as after an operand.
		let misplaced_drop = match site.site
		{
			Site::Faces => self
				.preceding(pos)
				.filter(|&(c, d_pos)| c == 'd' && begins_drop(&source[d_pos..]))
				.map(|(_, d_pos)| d_pos),
			Site::Goal { .. }
				if expects_expression && begins_drop(&source[pos..]) =>
			{
				Some(pos)
			},
			_ => None
		};
		if let Some(drop_pos) = misplaced_drop
		{
			let after_operand = site.site == Site::Faces;
			let repair = if after_operand
			{
				Repair::Reread
			}
			else
			{
				self.reread = Some(drop_pos);
				Repair::Fix
			};
			return Analysis::Diagnosed(
				make_misplaced_drop(source, drop_pos, after_operand),
				repair
			);
		}

		// Pattern: expression expected, and the text at the error position is
		// an identifier → bare identifier in expression context. The
		// identifier must begin the goal or the faces, which a `-` might
		// begin instead.
		let reads_variable = match site.site
		{
			Site::Goal { .. } => true,
			Site::Faces => self.preceding(pos).is_none_or(|(c, _)| c != '-'),
			_ => false
		};
		if expects_expression && !at_eof && reads_variable
		{
			// Only a name at the start of the input may begin formal
			// parameters.
			let place = if is_goal && self.preceding(pos).is_none()
			{
				Place::Start
			}
			else
			{
				Place::Expression
			};
			if let Some((diagnostic, end)) = self.bare_identifier_at(pos, place)
			{
				return Analysis::Diagnosed(
					diagnostic,
					Repair::Variable { end }
				);
			}
		}

		// Pattern: formal parameter expected after `,`, and the text after
		// the `,` is an identifier → bare identifier as formal parameter; or
		// the text after the `,` is another `,` or the `:` → a stray comma.
		if site.site == Site::ParameterName
		{
			let start = source.len()
				- source[pos..].trim_start_matches(is_token_space).len();
			if let Some((diagnostic, end)) =
				self.bare_identifier_at(start, Place::Parameters)
			{
				return Analysis::Diagnosed(
					diagnostic,
					Repair::Variable { end }
				);
			}
			if source[start..].starts_with([',', ':'])
				&& source[..pos].ends_with(',')
			{
				return Analysis::Diagnosed(
					make_missing_parameter(pos - 1, start),
					Repair::Retract
				);
			}
		}

		// Pattern: expression or identifier expected after delimiter/operator.
		let expects_identifier =
			expectations.iter().any(|e| e.as_ref() == "identifier");
		if (expects_expression || expects_identifier)
			&& let Some((prev, prev_pos)) = self.preceding(pos)
		{
			if (prev == 'd' || prev == 'D')
				&& (is_goal || site.site == Site::Faces)
			{
				return Analysis::Diagnosed(
					make_missing_dice_faces(source, prev_pos),
					Repair::Fix
				);
			}
			if "+-*×/÷%^".contains(prev) && is_goal
			{
				return Analysis::Diagnosed(
					make_missing_right_operand(source, prev, prev_pos, pos),
					Repair::Fix
				);
			}
			// Opening delimiter with no expression inside:
			// e.g., `(` → `(0)`, `[` → `[0:0]`, `{` → `{x}`.
			let shape = match (prev, site.site)
			{
				('(', Site::Goal { .. }) => Some(Shape::Group),
				('[', Site::Goal { .. }) => Some(Shape::Range),
				('[', Site::FaceValue { .. }) => Some(Shape::CustomFaces),
				('{', Site::VariableName { .. }) => Some(Shape::Variable),
				_ => None
			};
			if let Some(shape) = shape
			{
				return Analysis::Compound(Compound::new(shape, prev_pos, pos));
			}
			// After `:` inside a range, e.g., `[1:` → insert end expression and
			// `]`.
			if prev == ':'
				&& let Site::Goal {
					range: Some(bracket),
					..
				} = site.site
			{
				return Analysis::Compound(Compound::new(
					Shape::RangeEnd,
					bracket,
					pos
				));
			}
		}

		// Pattern: drop direction expected → incomplete drop clause. The drop
		// keyword ends just before the whitespace before the direction.
		if site.site == Site::Direction
			&& expectations
				.iter()
				.any(|e| e.as_ref() == "`lowest`" || e.as_ref() == "`highest`")
			&& let Some((last, last_pos)) = self.preceding(pos)
		{
			let drop_end = last_pos + last.len_utf8();
			return Analysis::Diagnosed(
				make_incomplete_drop(drop_end.saturating_sub(4), drop_end),
				Repair::Fix
			);
		}

		// Pattern: operator at start of input, perhaps after unopened closing
		// delimiters that earlier fixes discarded → missing left operand.
		if !at_eof && (pos == 0 || self.skipped == Some((0, pos))) && is_goal
		{
			let ch = source[pos..].chars().next().unwrap_or('\0');
			if "+-*×/÷%^".contains(ch)
			{
				return Analysis::Diagnosed(
					make_missing_left_operand(source, ch, pos),
					Repair::Fix
				);
			}
		}

		// Pattern: empty or whitespace-only input.
		if at_eof && is_goal && self.preceding(pos).is_none()
		{
			// The expression supplants the whitespace, but not the delimiters
			// that earlier fixes discarded.
			let start = self.skipped.map_or(0, |(_, end)| end);
			return Analysis::Diagnosed(
				make_empty_expression(start, source.len()),
				Repair::Fix
			);
		}

		// Pattern: formal parameters, missing `:` and perhaps body.
		if site.site == Site::ParameterColon
			&& expectations
				.iter()
				.any(|e| e.as_ref() == "`,`" || e.as_ref() == "`:`")
			&& let Some(analysis) = self.incomplete_parameters(pos)
		{
			return analysis;
		}

		// Pattern: bare closing delimiter where an expression was expected →
		// the user has a spurious `)`, `]`, or `}` with no matching opener.
		// This is more specific than the catch-all `UnexpectedToken` and
		// produces a `remove the unopened` suggestion. Only fires when an
		// expression was expected at `pos`; if the parser had already
		// consumed a complete expression, the trailing-input branch below
		// handles it instead.
		if expects_expression && !at_eof && is_goal
		{
			let ch = source[pos..].chars().next().unwrap_or('\0');
			if ch == ')' || ch == ']' || ch == '}'
			{
				let end = pos + ch.len_utf8();
				self.skip(pos, end);
				return Analysis::Diagnosed(
					make_unopened_delimiter(ch, pos),
					Repair::Skip { end }
				);
			}
		}

		// Pattern: end of input from all_consuming → trailing input.
		if site.site == Site::TrailingInput
			&& expectations.iter().any(|e| e.as_ref() == "end of input")
			&& !at_eof
		{
			self.trailing = true;
			return Analysis::Diagnosed(self.trailing_input(pos), Repair::Fix);
		}

		// Catch-all: unexpected token or EOF.
		let diagnostic = if at_eof
		{
			Diagnostic {
				kind: DiagnosticKind::UnexpectedEof,
				span: SourceSpan {
					start: pos,
					end: pos
				},
				message: "unexpected end of input".into(),
				related: vec![],
				suggestions: vec![]
			}
		}
		else
		{
			let (start, end, token) = unexpected_token(source, pos);
			Diagnostic {
				kind: DiagnosticKind::UnexpectedToken,
				span: SourceSpan { start, end },
				message: format!("unexpected `{}`", token),
				related: vec![],
				suggestions: vec![]
			}
		};
		Analysis::Diagnosed(diagnostic, Repair::Stop)
	}

	/// Detect a bare identifier at the position of a failure where an
	/// expression or a formal parameter was expected. The user almost
	/// certainly forgot to wrap it in `{}`, or wrote a formal parameter or a
	/// local binding as the language once did, without braces, e.g., `x, y: x
	/// + y` or `x@(3D6)`. A formal parameter or a local binding is wrapped
	/// whole; a variable may instead be split before a `d` or `D`, e.g., `xD6`.
	///
	/// # Parameters
	/// - `pos`: The position of the identifier.
	/// - `place`: Where the identifier lies.
	///
	/// # Returns
	/// A `BareIdentifier` diagnostic, if the text at `pos` is a [bare
	/// word](crate::parser::bare_word), and the byte offset of the end of the
	/// name that its first suggestion makes.
	fn bare_identifier_at(
		&mut self,
		pos: usize,
		place: Place
	) -> Option<(Diagnostic, usize)>
	{
		let ident_end = self.bare_word_end(pos)?;
		let name = self.source[pos..ident_end].trim_end_matches(is_token_space);
		if name.is_empty()
		{
			return None;
		}
		// The identifier may have swallowed inline whitespace after the name,
		// which both fixes leave where it is, e.g., `{hello} * 3`.
		let name_end = pos + name.len();
		let mut suggestions = Vec::new();

		// A formal parameter precedes a `,` or `:`, and the name of a local
		// binding precedes its `@`.
		let follows =
			self.source[name_end..].trim_start_matches(is_token_space);
		let what = match place
		{
			Place::Parameters => "parameter",
			Place::Start if follows.starts_with([',', ':']) => "parameter",
			_ if follows.starts_with('@') => "binding",
			_ => "variable"
		};

		// Check if a variable contains `d`/`D` — if so, offer both a split
		// fix (e.g., `{x}D6`) and a whole-variable fix (e.g., `{xD6}`). The
		// split fix leaves the whitespace before the `d` outside the braces,
		// e.g., `{abc} d6`.
		let split_pos = name.char_indices().find(|&(i, c)| {
			what == "variable"
				&& (c == 'd' || c == 'D')
				&& i > 0
				&& name[..i].starts_with(is_bare_word_start)
		});
		let prefix =
			split_pos.map(|(i, _)| name[..i].trim_end_matches(is_token_space));
		if let Some(prefix) = prefix
		{
			suggestions.push(Suggestion {
				description: format!(
					"wrap `{}` in braces",
					canonical_name(prefix)
				),
				edits: vec![
					insertion(pos, "{"),
					insertion(pos + prefix.len(), "}"),
				]
			});
		}
		// Always offer wrapping the whole identifier.
		suggestions.push(Suggestion {
			description: format!(
				"use `{}` as a {} name",
				canonical_name(name),
				what
			),
			edits: vec![insertion(pos, "{"), insertion(name_end, "}")]
		});

		// Use the first (most specific) suggestion's description for the
		// message. If there's a `d`/`D` split, the bare name is the prefix.
		// Text shows the name that the braced fix would denote, e.g., not the
		// U+00A0 NO-BREAK SPACE that the fix braces after `hit points`.
		let bare_name = prefix.unwrap_or(name);
		let end = pos + bare_name.len();
		Some((
			Diagnostic {
				kind: DiagnosticKind::BareIdentifier,
				span: SourceSpan {
					start: pos,
					end: name_end
				},
				message: format!(
					"bare identifier `{}` is not valid here; \
					 {}s must be wrapped in `{{}}`",
					canonical_name(bare_name),
					what
				),
				related: vec![],
				suggestions
			},
			end
		))
	}

	/// Detect an incomplete parameter definition, where formal parameters lack
	/// their `:`, and offer to complete it. Only two or more formal parameters
	/// can lack their `:`, since a lone braced name without one begins the
	/// body instead.
	///
	/// # Parameters
	/// - `pos`: The position of the failure.
	///
	/// # Returns
	/// An `IncompleteParameterDefinition` analysis, if the pattern matches.
	fn incomplete_parameters(&self, pos: usize) -> Option<Analysis>
	{
		let source = self.source;
		// The formal parameters lie before `pos`, past any delimiters that
		// earlier fixes discarded. Earlier fixes may have edited them, e.g.,
		// by bracing bare names, so edit them likewise.
		let start = self.skipped.map_or(0, |(_, end)| end);
		let mut edits = self
			.fixes
			.iter()
			.filter(|edit| start <= edit.span.start && edit.span.end <= pos)
			.map(|edit| {
				replacement(
					edit.span.start - start,
					edit.span.end - start,
					&edit.replacement
				)
			})
			.collect::<Vec<_>>();
		edits.sort_by_key(|edit| (edit.span.start, edit.span.end));
		let params = apply(&source[start..pos], &edits);
		if params.trim_matches(is_token_space).is_empty()
		{
			return None;
		}
		// The formal parameters end before any whitespace before `pos`, where
		// the `:` belongs.
		let end =
			start + source[start..pos].trim_end_matches(is_token_space).len();

		let trailing = source[pos..].trim_start_matches(is_token_space);
		// The start of the trailing content, past any whitespace after `pos`.
		let trailing_start = source.len() - trailing.len();

		// Try incorporating trailing content as the body. Only use it if the
		// result actually parses; otherwise fall back to a placeholder, which
		// supplants the trailing content. This parses the source once more, but
		// only once, since the formal parameters fail only once.
		let trailing_works = !trailing.is_empty() && {
			let mut candidate = params;
			candidate.push_str(": ");
			candidate.push_str(trailing);
			Parser::parse(&candidate).is_ok()
		};
		let suggestion = if trailing_works
		{
			Suggestion {
				description: "complete parameter definition".into(),
				edits: vec![replacement(end, trailing_start, ": ")]
			}
		}
		else
		{
			Suggestion {
				description: "complete parameter definition".into(),
				edits: vec![Edit {
					span: SourceSpan {
						start: end,
						end: source.len()
					},
					replacement: ": 0".into(),
					placeholders: vec![Placeholder {
						span: SourceSpan { start: 2, end: 3 },
						description: "expression",
						valid_kinds: OPERAND_KINDS
					}]
				}]
			}
		};
		let diagnostic = Diagnostic {
			kind: DiagnosticKind::IncompleteParameterDefinition,
			span: SourceSpan { start, end },
			message: "expected `:` and expression body after parameters".into(),
			related: vec![],
			suggestions: vec![suggestion]
		};
		Some(
			if trailing_works
			{
				Analysis::Diagnosed(diagnostic, Repair::Fix)
			}
			else
			{
				Analysis::Supplanting(diagnostic)
			}
		)
	}

	/// Create a diagnostic for trailing input after a complete function. The
	/// fix removes the trailing input and the whitespace before it, but not
	/// text that an earlier fix inserted.
	///
	/// # Parameters
	/// - `pos`: The byte offset of the trailing input.
	///
	/// # Returns
	/// A diagnostic with a `TrailingInput` kind.
	fn trailing_input(&self, pos: usize) -> Diagnostic
	{
		let source = self.source;
		let kept = self.fixes.iter().map(|edit| edit.span.end).fold(
			source[..pos].trim_end_matches(is_token_space).len(),
			usize::max
		);
		Diagnostic {
			kind: DiagnosticKind::TrailingInput,
			span: SourceSpan {
				start: pos,
				end: source.len()
			},
			message: format!(
				"unexpected `{}` after expression",
				unexpected_token(source, pos).2
			),
			related: vec![],
			suggestions: vec![Suggestion {
				description: "remove trailing input".into(),
				edits: vec![replacement(kept, source.len(), "")]
			}]
		}
	}
}

impl<'src> Recovery<'src> for Doctor<'src>
{
	fn repair(&mut self, site: &FailureSite<'src>) -> Repair
	{
		if self.state == State::Supplanted
		{
			return Repair::Stop;
		}
		if let Some(compound) = &mut self.compound
		{
			if compound.absorbs(site)
			{
				return Repair::Fix;
			}
			self.finish_compound();
		}
		// The faces of the dice expression that a fix supplied before a
		// misplaced `drop` fail at once, and the fix already accounts for them.
		if let Some(drop_pos) = self.reread.take()
			&& site.site == Site::Faces
			&& self
				.preceding(site.position())
				.is_some_and(|(_, operator)| operator == drop_pos)
		{
			return Repair::Reread;
		}
		match self.analyze(site)
		{
			Analysis::Diagnosed(diagnostic, repair) =>
			{
				if repair == Repair::Stop
				{
					self.state = State::Unfixable;
				}
				self.record(diagnostic);
				repair
			},
			Analysis::Supplanting(diagnostic) =>
			{
				self.state = State::Supplanted;
				self.record(diagnostic);
				Repair::Fix
			},
			Analysis::Compound(compound) =>
			{
				self.compound = Some(compound);
				Repair::Fix
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Fix generation.                               //
////////////////////////////////////////////////////////////////////////////////

/// Apply edits to a source.
///
/// # Parameters
/// - `source`: The source.
/// - `edits`: The edits, in ascending order of position. They must not overlap,
///   though insertions may abut.
///
/// # Returns
/// The source with the edits applied.
///
/// # Panics
/// If an edit's [span](Edit::span) is out of bounds or does not fall on
/// character boundaries of `source`.
fn apply(source: &str, edits: &[Edit]) -> String
{
	let mut corrected = String::with_capacity(source.len());
	let mut copied = 0;
	for edit in edits
	{
		corrected.push_str(&source[copied..edit.span.start]);
		corrected.push_str(&edit.replacement);
		copied = edit.span.end;
	}
	corrected.push_str(&source[copied..]);
	corrected
}

/// Create an [edit](Edit) that inserts text, with no placeholders.
///
/// # Parameters
/// - `pos`: The byte offset at which to insert.
/// - `text`: The text to insert.
///
/// # Returns
/// The edit.
fn insertion(pos: usize, text: &str) -> Edit { replacement(pos, pos, text) }

/// Create an [edit](Edit) that replaces a region with text, with no
/// placeholders.
///
/// # Parameters
/// - `start`: The byte offset of the start of the region.
/// - `end`: The byte offset of the end of the region.
/// - `text`: The replacement text, or `""` to delete the region.
///
/// # Returns
/// The edit.
fn replacement(start: usize, end: usize, text: &str) -> Edit
{
	Edit {
		span: SourceSpan { start, end },
		replacement: text.to_string(),
		placeholders: vec![]
	}
}

/// Answer the unexpected token at or after a position, past any [whitespace
/// between tokens](is_token_space), to show in a message. The token is the
/// text up to the next whitespace, but at least one character. Whitespace
/// that may not separate tokens, e.g., U+00A0 NO-BREAK SPACE, is a token by
/// itself, and is shown by its code point, e.g., `U+00A0`, since it would be
/// invisible.
///
/// # Parameters
/// - `source`: The source.
/// - `pos`: The byte offset of the position.
///
/// # Returns
/// The byte offsets of the start and end of the token, and the token as
/// shown, which is empty at the end of the source.
fn unexpected_token(source: &str, pos: usize) -> (usize, usize, Cow<'_, str>)
{
	let rest = source[pos..].trim_start_matches(is_token_space);
	let start = source.len() - rest.len();
	match rest.chars().next()
	{
		None => (start, start, Cow::Borrowed("")),
		Some(c) if c.is_whitespace() =>
		{
			let token = format!("U+{:04X}", u32::from(c));
			(start, start + c.len_utf8(), Cow::Owned(token))
		},
		Some(c) =>
		{
			let len = rest[c.len_utf8()..]
				.find(char::is_whitespace)
				.map_or(rest.len(), |i| c.len_utf8() + i);
			(start, start + len, Cow::Borrowed(&rest[..len]))
		}
	}
}

/// Find the stray whitespace of a source: each character of whitespace outside
/// of braces that may not [separate tokens](is_token_space), e.g., U+00A0
/// NO-BREAK SPACE, which text pasted from a word processor often carries.
/// Inside braces, any whitespace belongs to the name. The name of a variable
/// whose `}` is missing ends at the next brace or the end of the source, or
/// where the doctor [breaks](unclosed_name_ends) it, if it does, since the
/// text after the break lies between tokens. The scan takes time linear in the
/// length of the source.
///
/// # Parameters
/// - `source`: The source.
///
/// # Returns
/// The byte offsets of the start and end of each character of stray
/// whitespace, in order.
fn stray_whitespace(source: &str) -> Vec<(usize, usize)>
{
	let mut strays = Vec::new();
	let mut at = 0;
	while let Some(c) = source[at..].chars().next()
	{
		if c == '{'
		{
			let opened = at + 1;
			let end = source[opened..]
				.find(['{', '}'])
				.map_or(source.len(), |i| opened + i);
			at = if source[end..].starts_with('}')
			{
				end + 1
			}
			else
			{
				let name = source[opened..end].trim_start();
				match unclosed_name_ends(name)
				{
					[Some(length), _] => end - name.len() + length,
					_ => end
				}
			};
			continue;
		}
		if c.is_whitespace() && !is_token_space(c)
		{
			strays.push((at, at + c.len_utf8()));
		}
		at += c.len_utf8();
	}
	strays
}

/// The shape of a [compound fix](Compound): the construct whose contents it
/// supplies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Shape
{
	/// A group or the expression of a binding, after its `(`: `0)`.
	Group,

	/// A range, after its `[`: `0:0]`.
	Range,

	/// Custom faces, after their `[`: `0]`.
	CustomFaces,

	/// A variable, after its `{`: `x}`.
	Variable,

	/// The end of a range, after its `:`: `0]`.
	RangeEnd
}

/// A [token](Token) that a [compound fix](Compound) may supply, and the
/// [site](Site) of the failure at which the parser needs it.
#[derive(Debug)]
struct Token
{
	/// The text of the token.
	text: &'static str,

	/// Whether the parser needs the token at the specified site.
	needed_at: fn(Site) -> bool,

	/// The description and valid kinds of the placeholder that the token is,
	/// if any.
	placeholder: Option<(&'static str, &'static [&'static str])>
}

impl Token
{
	/// An operand, needed at a [goal](Site::Goal).
	///
	/// # Parameters
	/// - `description`: The description of the placeholder.
	///
	/// # Returns
	/// The token.
	const fn operand(description: &'static str) -> Self
	{
		Self {
			text: "0",
			needed_at: |site| matches!(site, Site::Goal { .. }),
			placeholder: Some((description, OPERAND_KINDS))
		}
	}

	/// A closing delimiter, needed at a [closer](Site::Closer).
	///
	/// # Parameters
	/// - `text`: The delimiter.
	///
	/// # Returns
	/// The token.
	const fn closer(text: &'static str) -> Self
	{
		Self {
			text,
			needed_at: |site| matches!(site, Site::Closer { .. }),
			placeholder: None
		}
	}
}

impl Shape
{
	/// Answer the tokens that a compound fix of this shape supplies, in order.
	///
	/// # Returns
	/// The tokens.
	fn tokens(self) -> &'static [Token]
	{
		const GROUP: &[Token] =
			&[Token::operand("expression"), Token::closer(")")];
		const RANGE: &[Token] = &[
			Token::operand("range start"),
			Token {
				text: ":",
				needed_at: |site| matches!(site, Site::Colon { .. }),
				placeholder: None
			},
			Token::operand("range end"),
			Token::closer("]")
		];
		const CUSTOM_FACES: &[Token] = &[
			Token {
				text: "0",
				needed_at: |site| matches!(site, Site::FaceValue { .. }),
				placeholder: Some(("face value", &["integer"]))
			},
			Token::closer("]")
		];
		const VARIABLE: &[Token] = &[
			Token {
				text: "x",
				needed_at: |site| matches!(site, Site::VariableName { .. }),
				placeholder: Some(("identifier", &["identifier"]))
			},
			Token::closer("}")
		];
		const RANGE_END: &[Token] =
			&[Token::operand("range end"), Token::closer("]")];
		match self
		{
			Shape::Group => GROUP,
			Shape::Range => RANGE,
			Shape::CustomFaces => CUSTOM_FACES,
			Shape::Variable => VARIABLE,
			Shape::RangeEnd => RANGE_END
		}
	}

	/// Answer what a compound fix of this shape supplies, when it supplies
	/// only the specified number of its tokens, because the source supplies
	/// the rest.
	///
	/// # Parameters
	/// - `supplied`: The number of tokens supplied.
	///
	/// # Returns
	/// A description of the supplied tokens.
	fn what(self, supplied: usize) -> &'static str
	{
		match (self, supplied)
		{
			(Shape::Group, _) => "expression",
			(Shape::Range, 1) => "range start",
			(Shape::Range, _) => "range start and end",
			(Shape::CustomFaces, _) => "face value",
			(Shape::Variable, _) => "identifier",
			(Shape::RangeEnd, _) => "range end"
		}
	}
}

/// A fix that supplies the missing contents of a delimited construct, and
/// perhaps its closing delimiter, e.g., `0)` after `(`. The parser needs the
/// tokens of the fix one at a time, each at a failure at the same position;
/// the fix supplies only those tokens that the parser needs, since the source
/// may supply the rest, e.g., the `)` of `( )`.
#[derive(Debug)]
struct Compound
{
	/// The shape of the fix.
	shape: Shape,

	/// The byte offset of the opening delimiter of the construct.
	opener_pos: usize,

	/// The byte offset at which the fix inserts its tokens.
	at: usize,

	/// The number of tokens supplied so far.
	supplied: usize
}

impl Compound
{
	/// Begin a compound fix, supplying its first token.
	///
	/// # Parameters
	/// - `shape`: The shape of the fix.
	/// - `opener_pos`: The byte offset of the opening delimiter.
	/// - `at`: The byte offset of the failure, where the fix inserts its
	///   tokens.
	///
	/// # Returns
	/// The fix.
	fn new(shape: Shape, opener_pos: usize, at: usize) -> Self
	{
		Self {
			shape,
			opener_pos,
			at,
			supplied: 1
		}
	}

	/// Supply the next token, if the parser needs it at the specified
	/// failure.
	///
	/// # Parameters
	/// - `site`: The failure.
	///
	/// # Returns
	/// `true` if the fix supplied the token, `false` otherwise.
	fn absorbs(&mut self, site: &FailureSite<'_>) -> bool
	{
		let absorbs = site.position() == self.at
			&& self
				.shape
				.tokens()
				.get(self.supplied)
				.is_some_and(|token| (token.needed_at)(site.site));
		if absorbs
		{
			self.supplied += 1;
		}
		absorbs
	}

	/// Answer the diagnostic of the fix, as supplied so far. If the fix
	/// supplied every token, it closes an unclosed delimiter; otherwise, it
	/// supplies only the missing contents of a closed one.
	///
	/// # Parameters
	/// - `source`: The source text.
	///
	/// # Returns
	/// The diagnostic.
	fn diagnostic(&self, source: &str) -> Diagnostic
	{
		let tokens = self.shape.tokens();
		let mut replacement = String::new();
		let mut placeholders = Vec::new();
		for token in &tokens[..self.supplied]
		{
			if let Some((description, valid_kinds)) = token.placeholder
			{
				placeholders.push(Placeholder {
					span: SourceSpan {
						start: replacement.len(),
						end: replacement.len() + token.text.len()
					},
					description,
					valid_kinds
				});
			}
			replacement.push_str(token.text);
		}
		let edits = vec![Edit {
			span: SourceSpan {
				start: self.at,
				end: self.at
			},
			replacement,
			placeholders
		}];
		let opener = source[self.opener_pos..].chars().next().unwrap_or('(');
		let span = SourceSpan {
			start: self.opener_pos,
			end: self.opener_pos + opener.len_utf8()
		};
		if self.supplied == tokens.len()
		{
			let closer = tokens[tokens.len() - 1].text;
			let description = match self.shape
			{
				Shape::RangeEnd => "insert range end and `]`".to_string(),
				_ => format!("insert expression and `{}`", closer)
			};
			Diagnostic {
				kind: DiagnosticKind::UnclosedDelimiter {
					opener,
					expected_closer: closer.chars().next().unwrap_or(')')
				},
				span,
				message: format!("expected `{}` to close `{}`", closer, opener),
				related: vec![],
				suggestions: vec![Suggestion { description, edits }]
			}
		}
		else
		{
			let what = self.shape.what(self.supplied);
			let next = source[self.at..].chars().next().unwrap_or(' ');
			Diagnostic {
				kind: DiagnosticKind::MissingExpression { opener },
				span,
				message: format!("expected {} before `{}`", what, next),
				related: vec![],
				suggestions: vec![Suggestion {
					description: format!("insert {}", what),
					edits
				}]
			}
		}
	}
}

/// Create a diagnostic for an unclosed delimiter whose contents are complete.
/// The fix inserts the closing delimiter.
///
/// # Parameters
/// - `opener`: The opening delimiter character (e.g., `(`).
/// - `closer`: The expected closing delimiter character (e.g., `)`).
/// - `opener_pos`: The byte offset of the opening delimiter.
/// - `error_pos`: The byte offset where the error was detected.
///
/// # Returns
/// A diagnostic with an `UnclosedDelimiter` kind and a suggestion that
/// inserts the closing delimiter.
fn make_unclosed_delimiter(
	opener: char,
	closer: char,
	opener_pos: usize,
	error_pos: usize
) -> Diagnostic
{
	// The parser only raises a closer expectation after it has already
	// consumed the contents of the delimited construct. A construct without
	// contents is a [`Compound`] fix instead.
	Diagnostic {
		kind: DiagnosticKind::UnclosedDelimiter {
			opener,
			expected_closer: closer
		},
		span: SourceSpan {
			start: opener_pos,
			end: opener_pos + opener.len_utf8()
		},
		message: format!("expected `{}` to close `{}`", closer, opener),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("insert `{}`", closer),
			edits: vec![insertion(error_pos, &closer.to_string())]
		}]
	}
}

/// Create a diagnostic for a variable whose `}` is missing. The fixes insert
/// the `}` at each place where the name may [break](unclosed_name_ends), the
/// more likely first, and then at the failure, which keeps the whole name as
/// written. A less likely break is offered only if no bare word follows its
/// break character, since the fix would not parse, e.g., `{hit}-points + 2`
/// for `{hit-points + 2`. Unless it is the only fix, the last is offered only
/// at the end of the input, since a brace follows the failure otherwise, and
/// only where the name holds no `)` or `]` that nothing in the name opened,
/// since the whole name would swallow the closing delimiter of a construct
/// around the variable, e.g., the `)` of `({x + 2)`.
///
/// # Parameters
/// - `source`: The source text.
/// - `opener_pos`: The byte offset of the `{`.
/// - `ends`: The byte offsets at which the name may break, the more likely
///   first.
/// - `error_pos`: The byte offset where the `}` was expected.
/// - `at_eof`: Whether `error_pos` is at the end of the input.
///
/// # Returns
/// A diagnostic with an `UnclosedDelimiter` kind and a suggestion for each
/// fix.
fn make_unclosed_brace(
	source: &str,
	opener_pos: usize,
	ends: &[usize],
	error_pos: usize,
	at_eof: bool
) -> Diagnostic
{
	let mut diagnostic =
		make_unclosed_delimiter('{', '}', opener_pos, error_pos);
	let whole = diagnostic.suggestions.pop();
	diagnostic.suggestions = ends
		.iter()
		.enumerate()
		.filter_map(|(i, &end)| {
			let mut after =
				source[end..].trim_start_matches(is_token_space).chars();
			let before = after.next().unwrap_or(' ');
			// A less likely break must not leave a bare word after the break
			// character, e.g., `points` in `{hit}-points + 2`.
			let bare = after
				.as_str()
				.trim_start_matches(is_token_space)
				.starts_with(is_bare_word_start);
			(i == 0 || !bare).then(|| Suggestion {
				description: format!("insert `}}` before `{}`", before),
				edits: vec![insertion(end, "}")]
			})
		})
		.collect();
	if ends.is_empty()
		|| at_eof && !swallows_closer(&source[opener_pos + 1..error_pos])
	{
		diagnostic.suggestions.extend(whole);
	}
	diagnostic
}

/// Answer whether a name holds a `)` or `]` that nothing in the name opened.
///
/// # Parameters
/// - `name`: The name, as written.
///
/// # Returns
/// `true` if the name holds an unopened closing delimiter, `false` otherwise.
fn swallows_closer(name: &str) -> bool
{
	let (mut parentheses, mut brackets) = (0usize, 0usize);
	for c in name.chars()
	{
		let depth = match c
		{
			'(' | ')' => &mut parentheses,
			'[' | ']' => &mut brackets,
			_ => continue
		};
		if matches!(c, '(' | '[')
		{
			*depth += 1;
		}
		else if *depth == 0
		{
			return true;
		}
		else
		{
			*depth -= 1;
		}
	}
	false
}

/// Create a diagnostic for a missing right operand. Offers two suggestions:
/// inserting a placeholder operand, or dropping the operator.
///
/// # Parameters
/// - `source`: The source text.
/// - `operator`: The binary operator character (e.g., `+`).
/// - `op_pos`: The byte offset of the operator.
/// - `error_pos`: The byte offset where the right operand was expected.
///
/// # Returns
/// A diagnostic with a `MissingRightOperand` kind and two suggestions.
fn make_missing_right_operand(
	source: &str,
	operator: char,
	op_pos: usize,
	error_pos: usize
) -> Diagnostic
{
	// Suggestion 1: insert `0` at the error position, separated by a space
	// from the operator before it and from the token after it, but not from
	// a closing delimiter or separator, e.g., `(1 + 0)` for `(1 + )`.
	let before = &source[..error_pos];
	let after = &source[error_pos..];
	let mut operand = String::new();
	if !before.is_empty() && !before.ends_with(is_token_space)
	{
		operand.push(' ');
	}
	let placeholder_start = operand.len();
	operand.push('0');
	let placeholder_end = operand.len();
	if !after.is_empty()
		&& !after.starts_with(is_token_space)
		&& !after.starts_with([')', ']', '}', ',', ':'])
	{
		operand.push(' ');
	}

	// Suggestion 2: drop the operator, and the whitespace around it, keeping
	// a single space between the surviving neighbors, but none before a
	// closing delimiter or separator, e.g., `(1)` for `(1 + )`.
	let before_op = source[..op_pos].trim_end_matches(is_token_space);
	let rest = after.trim_start_matches(is_token_space);
	let separator = if !after.is_empty()
		&& !before_op.is_empty()
		&& !rest.starts_with([')', ']', '}', ',', ':'])
	{
		" "
	}
	else
	{
		""
	};

	Diagnostic {
		kind: DiagnosticKind::MissingRightOperand { operator },
		span: SourceSpan {
			start: op_pos,
			end: error_pos
		},
		message: format!("expected operand after `{}`", operator),
		related: vec![],
		suggestions: vec![
			Suggestion {
				description: format!("insert operand after `{}`", operator),
				edits: vec![Edit {
					span: SourceSpan {
						start: error_pos,
						end: error_pos
					},
					replacement: operand,
					placeholders: vec![Placeholder {
						span: SourceSpan {
							start: placeholder_start,
							end: placeholder_end
						},
						description: "operand",
						valid_kinds: OPERAND_KINDS
					}]
				}]
			},
			Suggestion {
				description: format!("remove `{}`", operator),
				edits: vec![replacement(
					before_op.len(),
					source.len() - rest.len(),
					separator
				)]
			},
		]
	}
}

/// Create a diagnostic for a missing left operand. Offers two suggestions:
/// inserting a placeholder operand, or dropping the operator.
///
/// # Parameters
/// - `source`: The source text.
/// - `operator`: The binary operator character at the start of the input.
/// - `op_pos`: The byte offset of the operator.
///
/// # Returns
/// A diagnostic with a `MissingLeftOperand` kind and two suggestions.
fn make_missing_left_operand(
	source: &str,
	operator: char,
	op_pos: usize
) -> Diagnostic
{
	// Suggestion 2 drops the operator and the whitespace after it.
	let op_end = op_pos + operator.len_utf8();
	let after_op = source[op_end..].trim_start_matches(is_token_space);

	Diagnostic {
		kind: DiagnosticKind::MissingLeftOperand { operator },
		span: SourceSpan {
			start: op_pos,
			end: op_end
		},
		message: format!("expected operand before `{}`", operator),
		related: vec![],
		suggestions: vec![
			Suggestion {
				description: format!("insert operand before `{}`", operator),
				edits: vec![Edit {
					span: SourceSpan {
						start: op_pos,
						end: op_pos
					},
					replacement: "0 ".into(),
					placeholders: vec![Placeholder {
						span: SourceSpan { start: 0, end: 1 },
						description: "operand",
						valid_kinds: OPERAND_KINDS
					}]
				}]
			},
			Suggestion {
				description: format!("remove `{}`", operator),
				edits: vec![replacement(
					op_pos,
					source.len() - after_op.len(),
					""
				)]
			},
		]
	}
}

/// Create a diagnostic for a comma in the formal parameters that another
/// comma or the `:` follows. The fix removes the comma, and the whitespace
/// after it.
///
/// # Parameters
/// - `comma`: The byte offset of the comma.
/// - `next`: The byte offset of the `,` or `:` that follows the comma.
///
/// # Returns
/// A diagnostic with a `MissingParameter` kind and a suggestion that removes
/// the comma.
fn make_missing_parameter(comma: usize, next: usize) -> Diagnostic
{
	Diagnostic {
		kind: DiagnosticKind::MissingParameter,
		span: SourceSpan {
			start: comma,
			end: comma + 1
		},
		message: "expected a formal parameter after `,`".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "remove `,`".into(),
			edits: vec![replacement(comma, next, "")]
		}]
	}
}

/// Create a diagnostic for missing dice faces after `d`/`D`. The fix inserts
/// `6` as a placeholder face count immediately after the dice operator.
///
/// # Parameters
/// - `source`: The source text.
/// - `d_pos`: The byte offset of the `d`/`D` operator.
///
/// # Returns
/// A diagnostic with a `MissingDiceFaces` kind and a suggestion with a
/// placeholder for the face count.
fn make_missing_dice_faces(source: &str, d_pos: usize) -> Diagnostic
{
	// Insert `6` as a default face count right after the `d`/`D`, preserving
	// any whitespace between the operator and the error.
	let insert_pos = d_pos + 1;
	Diagnostic {
		kind: DiagnosticKind::MissingDiceFaces,
		span: SourceSpan {
			start: d_pos,
			end: d_pos + 1
		},
		message: format!(
			"expected face count after `{}`",
			&source[d_pos..d_pos + 1]
		),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert face count".into(),
			edits: vec![Edit {
				span: SourceSpan {
					start: insert_pos,
					end: insert_pos
				},
				replacement: "6".into(),
				placeholders: vec![Placeholder {
					span: SourceSpan { start: 0, end: 1 },
					description: "face count",
					valid_kinds: FACE_COUNT_KINDS
				}]
			}]
		}]
	}
}

/// Answer whether text begins with the keyword `drop`, as a drop clause would,
/// rather than a name that begins with `drop`, e.g., `dropout`: `drop` must
/// be followed by a direction, or by something that cannot continue a [bare
/// word](crate::parser::bare_word), e.g., whitespace or the end of the text.
///
/// # Parameters
/// - `text`: The text.
///
/// # Returns
/// `true` if `text` begins with the keyword `drop`, `false` otherwise.
fn begins_drop(text: &str) -> bool
{
	text.strip_prefix("drop").is_some_and(|after| {
		after.starts_with("lowest")
			|| after.starts_with("highest")
			|| after
				.chars()
				.next()
				.is_none_or(|c| c.is_whitespace() || !is_bare_word_continue(c))
	})
}

/// Create a diagnostic for a drop clause that follows something other than a
/// dice expression. After an operand, which the parser read as the count of a
/// dice expression whose operator is the `d` of `drop`, the fix inserts
/// placeholder faces after the operand, e.g., `3D6 drop lowest` for
/// `3 drop lowest`. Otherwise, it inserts a placeholder dice expression
/// before `drop`, e.g., `1 + 1D6 drop lowest` for `1 + drop lowest`.
///
/// # Parameters
/// - `source`: The source.
/// - `drop_pos`: The byte offset of `drop`.
/// - `after_operand`: Whether an operand precedes `drop`.
///
/// # Returns
/// A diagnostic with a `MisplacedDropClause` kind and a suggestion with a
/// placeholder for the faces or the dice expression.
fn make_misplaced_drop(
	source: &str,
	drop_pos: usize,
	after_operand: bool
) -> Diagnostic
{
	let edit = if after_operand
	{
		// Separate the faces from `drop`, unless whitespace already does.
		let operand_end =
			source[..drop_pos].trim_end_matches(is_token_space).len();
		Edit {
			span: SourceSpan {
				start: operand_end,
				end: operand_end
			},
			replacement: if operand_end == drop_pos { "D6 " } else { "D6" }
				.into(),
			placeholders: vec![Placeholder {
				span: SourceSpan { start: 1, end: 2 },
				description: "face count",
				valid_kinds: FACE_COUNT_KINDS
			}]
		}
	}
	else
	{
		// Separate the dice expression from a `-` just before it, which
		// would otherwise make its count a negative constant.
		let separator = if source[..drop_pos].ends_with('-')
		{
			" "
		}
		else
		{
			""
		};
		Edit {
			span: SourceSpan {
				start: drop_pos,
				end: drop_pos
			},
			replacement: format!("{}1D6 ", separator),
			placeholders: vec![Placeholder {
				span: SourceSpan {
					start: separator.len(),
					end: separator.len() + 3
				},
				description: "dice expression",
				valid_kinds: DICE_KINDS
			}]
		}
	};
	Diagnostic {
		kind: DiagnosticKind::MisplacedDropClause,
		span: SourceSpan {
			start: drop_pos,
			end: drop_pos + "drop".len()
		},
		message: "`drop` must follow a dice expression".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: if after_operand
			{
				"insert dice faces before `drop`"
			}
			else
			{
				"insert a dice expression before `drop`"
			}
			.into(),
			edits: vec![edit]
		}]
	}
}

/// Create a diagnostic for an incomplete drop clause. The fix inserts
/// `lowest` as a placeholder direction after the `drop` keyword.
///
/// # Parameters
/// - `drop_start`: The byte offset of `drop`.
/// - `drop_end`: The byte offset of the end of `drop`.
///
/// # Returns
/// A diagnostic with an `IncompleteDropClause` kind and a suggestion with a
/// placeholder for the drop direction.
fn make_incomplete_drop(drop_start: usize, drop_end: usize) -> Diagnostic
{
	// Insert ` lowest` after `drop`.
	Diagnostic {
		kind: DiagnosticKind::IncompleteDropClause,
		span: SourceSpan {
			start: drop_start,
			end: drop_end
		},
		message: "expected `lowest` or `highest` after `drop`".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert drop direction".into(),
			edits: vec![Edit {
				span: SourceSpan {
					start: drop_end,
					end: drop_end
				},
				replacement: " lowest".into(),
				placeholders: vec![Placeholder {
					span: SourceSpan { start: 1, end: 7 },
					description: "direction",
					valid_kinds: DROP_DIRECTION_KINDS
				}]
			}]
		}]
	}
}

/// Create a diagnostic for a bare closing delimiter with no matching opener.
/// The fix removes the spurious closer. Emitted from
/// [`analyze`](Doctor::analyze) when an expression was expected at the error
/// position but the source has `)`, `]`, or `}` there instead.
///
/// # Parameters
/// - `closer`: The unmatched closing delimiter character.
/// - `pos`: The byte offset of the closer.
///
/// # Returns
/// A diagnostic with an
/// [`UnopenedDelimiter`](DiagnosticKind::UnopenedDelimiter) kind and a single
/// suggestion that removes the offending character.
fn make_unopened_delimiter(closer: char, pos: usize) -> Diagnostic
{
	Diagnostic {
		kind: DiagnosticKind::UnopenedDelimiter { closer },
		span: SourceSpan {
			start: pos,
			end: pos + closer.len_utf8()
		},
		message: format!("unexpected `{}` with no matching opener", closer),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("remove the unopened `{}`", closer),
			edits: vec![replacement(pos, pos + closer.len_utf8(), "")]
		}]
	}
}

/// Create a diagnostic for a character of [stray whitespace](stray_whitespace).
/// The fix replaces it with a space.
///
/// # Parameters
/// - `source`: The original source.
/// - `start`: The byte offset of the character.
/// - `end`: The byte offset of the end of the character.
///
/// # Returns
/// A diagnostic with an `UnexpectedToken` kind that shows the character by its
/// code point, e.g., `U+00A0`, since it would be invisible.
fn make_stray_whitespace(source: &str, start: usize, end: usize) -> Diagnostic
{
	let (_, _, token) = unexpected_token(source, start);
	Diagnostic {
		kind: DiagnosticKind::UnexpectedToken,
		span: SourceSpan { start, end },
		message: format!(
			"unexpected `{}`; only spaces, tabs, and line breaks may separate \
			 tokens",
			token
		),
		related: vec![],
		suggestions: vec![Suggestion {
			description: format!("replace `{}` with a space", token),
			edits: vec![replacement(start, end, " ")]
		}]
	}
}

/// Create a diagnostic for an empty or whitespace-only expression. The fix
/// inserts `0` as a placeholder.
///
/// # Parameters
/// - `start`: The byte offset of the whitespace that the expression supplants.
/// - `end`: The byte offset of the end of the whitespace, i.e., of the source.
///
/// # Returns
/// A diagnostic with an `EmptyExpression` kind and a suggestion with a
/// placeholder for the expression.
fn make_empty_expression(start: usize, end: usize) -> Diagnostic
{
	Diagnostic {
		kind: DiagnosticKind::EmptyExpression,
		span: SourceSpan { start: 0, end: 0 },
		message: "expected expression".into(),
		related: vec![],
		suggestions: vec![Suggestion {
			description: "insert expression".into(),
			edits: vec![Edit {
				span: SourceSpan { start, end },
				replacement: "0".into(),
				placeholders: vec![Placeholder {
					span: SourceSpan { start: 0, end: 1 },
					description: "expression",
					valid_kinds: OPERAND_KINDS
				}]
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
/// that is, when no fixes have been applied — so that semantic
/// diagnostics reference the user's actual source text rather than a
/// fix-synthesized variant. Returns an empty vector when the validator
/// accepts the function, or a single-element vector when it rejects it.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `ast`: The parsed function to validate.
///
/// # Returns
/// The semantic diagnostics produced by the validator, in source order.
pub(crate) fn run_validator<'src>(ast: &Function<'src>) -> Vec<Diagnostic>
{
	match Validator::validate(ast)
	{
		Ok(()) => Vec::new(),
		Err(error) => vec![analyze_semantic_error(ast, error)]
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
/// - `ast`: The parsed function that produced the error.
/// - `error`: The semantic error returned by [`Validator::validate`].
///
/// # Returns
/// A [`Diagnostic`] describing the semantic error, with a rename suggestion
/// whose corrected source parses and validates cleanly.
fn analyze_semantic_error<'src>(
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
		} => make_duplicate_parameter(ast, &name, first, duplicate),
		CompilationError::BindingCollidesWithParameter {
			name,
			parameter,
			binding
		} =>
		{
			make_binding_collides_with_parameter(ast, &name, parameter, binding)
		},
		CompilationError::DuplicateBinding {
			name,
			first,
			duplicate
		} => make_duplicate_binding(ast, &name, first, duplicate),
		CompilationError::UseBeforeBind {
			name,
			reference,
			binding
		} => make_use_before_bind(ast, &name, reference, binding),
		CompilationError::ParseError(_)
		| CompilationError::OptimizationFailed =>
		{
			unreachable!(
				"Validator::validate produces only semantic errors; \
				 ParseError is produced by the parser and OptimizationFailed \
				 by the optimizer, neither of which is reached from \
				 diagnose() at this point"
			)
		}
	}
}

/// Build a
/// [`BindingCollidesWithParameter`](DiagnosticKind::BindingCollidesWithParameter)
/// diagnostic with a rename [`Suggestion`] whose fresh name collides with no
/// name in the function, so that the suggestion resolves this error without
/// introducing another, though a source with several errors needs the fixes of
/// all of them.
///
/// The primary [`span`](Diagnostic::span) points at the binding-site name; a
/// single [`RelatedLabel`] points at the colliding parameter declaration. The
/// suggestion splices a fresh name into the binding span — references in the
/// body are deliberately left alone because there is no safe way to pick which
/// of them (if any) was intended to refer to the parameter rather than the
/// binding.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `ast`: The parsed function (used to enumerate in-use names).
/// - `name`: The colliding name.
/// - `parameter`: The span of the colliding parameter declaration.
/// - `binding`: The span of the binding-site name.
///
/// # Returns
/// A fully-formed [`Diagnostic`] with one rename [`Suggestion`].
fn make_binding_collides_with_parameter<'src>(
	ast: &Function<'src>,
	name: &str,
	parameter: SourceSpan,
	binding: SourceSpan
) -> Diagnostic
{
	let used = collect_in_use_names(ast);
	let fresh = suggest_rename_with_pool(name, &used);
	Diagnostic {
		kind: DiagnosticKind::BindingCollidesWithParameter {
			name: name.to_string()
		},
		span: binding,
		message: format!(
			"local binding `{}` collides with a formal parameter of the same \
			 name; bindings, parameters, and environment variables share one \
			 flat namespace per function",
			name
		),
		related: vec![RelatedLabel {
			span: parameter,
			message: "declared as a parameter here".into()
		}],
		suggestions: vec![Suggestion {
			description: format!("rename local binding to `{}`", fresh),
			edits: vec![replacement(binding.start, binding.end, &fresh)]
		}]
	}
}

/// Build a [`DuplicateBinding`](DiagnosticKind::DuplicateBinding) diagnostic
/// with a rename [`Suggestion`] whose fresh name collides with no name in the
/// function, so that the suggestion resolves this error without introducing
/// another, though a source with several errors needs the fixes of all of
/// them.
///
/// The primary [`span`](Diagnostic::span) points at the duplicate binding-site
/// name; a single [`RelatedLabel`] points at the first binding. The suggestion
/// splices a fresh name into the duplicate binding span — references in the
/// body are deliberately left alone because there is no safe way to pick which
/// one (if any) was intended to refer to the rebinding rather than the
/// original.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `ast`: The parsed function (used to enumerate in-use names).
/// - `name`: The rebound name.
/// - `first`: The span of the first binding-site name.
/// - `duplicate`: The span of the duplicate binding-site name.
///
/// # Returns
/// A fully-formed [`Diagnostic`] with one rename [`Suggestion`].
fn make_duplicate_binding<'src>(
	ast: &Function<'src>,
	name: &str,
	first: SourceSpan,
	duplicate: SourceSpan
) -> Diagnostic
{
	let used = collect_in_use_names(ast);
	let fresh = suggest_rename_with_pool(name, &used);
	Diagnostic {
		kind: DiagnosticKind::DuplicateBinding {
			name: name.to_string()
		},
		span: duplicate,
		message: format!(
			"local binding `{}` is bound more than once; a function body \
			 provides a single flat namespace, so rebinding is not permitted",
			name
		),
		related: vec![RelatedLabel {
			span: first,
			message: "first bound here".into()
		}],
		suggestions: vec![Suggestion {
			description: format!("rename duplicate binding to `{}`", fresh),
			edits: vec![replacement(duplicate.start, duplicate.end, &fresh)]
		}]
	}
}

/// Build a [`UseBeforeBind`](DiagnosticKind::UseBeforeBind) diagnostic with a
/// rename [`Suggestion`] whose fresh name collides with no name in the
/// function, so that the suggestion resolves this error without introducing
/// another, though a source with several errors needs the fixes of all of
/// them.
///
/// The primary [`span`](Diagnostic::span) points at the offending reference; a
/// single [`RelatedLabel`] points at the later binding site. The suggestion
/// splices a fresh name into the binding site — the forward reference is then
/// reinterpreted as an external variable. Renaming the binding rather than the
/// reference is consistent with [`make_binding_collides_with_parameter`] and
/// [`make_duplicate_binding`]: always rewrite the declaration, never the
/// references.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `ast`: The parsed function (used to enumerate in-use names).
/// - `name`: The name referenced before its binding.
/// - `reference`: The span of the offending reference.
/// - `binding`: The span of the binding-site name that appears later in the
///   source.
///
/// # Returns
/// A fully-formed [`Diagnostic`] with one rename [`Suggestion`].
fn make_use_before_bind<'src>(
	ast: &Function<'src>,
	name: &str,
	reference: SourceSpan,
	binding: SourceSpan
) -> Diagnostic
{
	let used = collect_in_use_names(ast);
	let fresh = suggest_rename_with_pool(name, &used);
	Diagnostic {
		kind: DiagnosticKind::UseBeforeBind {
			name: name.to_string()
		},
		span: reference,
		message: format!(
			"reference to `{}` precedes its binding; references to a local \
			 binding must lexically follow the binding site, including \
			 references inside the bound expression itself (self-reference \
			 is not permitted)",
			name
		),
		related: vec![RelatedLabel {
			span: binding,
			message: "bound here".into()
		}],
		suggestions: vec![Suggestion {
			description: format!("rename local binding to `{}`", fresh),
			edits: vec![replacement(binding.start, binding.end, &fresh)]
		}]
	}
}

/// Collect every name that is in use anywhere in `ast`: formal parameters,
/// [local-binding](crate::ast::Binding) names, and
/// [variable references](crate::ast::Variable). A fresh name suggested against
/// this pool is guaranteed to collide with nothing currently in the function.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the function.
///
/// # Parameters
/// - `ast`: The parsed function.
///
/// # Returns
/// A set containing every parameter, binding, and variable-reference name.
pub(crate) fn collect_in_use_names<'a>(
	ast: &'a Function<'_>
) -> HashSet<&'a str>
{
	let mut names: HashSet<&'a str> = HashSet::new();
	if let Some(ref parameters) = ast.parameters
	{
		for param in parameters
		{
			names.insert(&param.name);
		}
	}
	for event in Walk::new(Node::Expression(&ast.body))
	{
		match event
		{
			Event::Enter(Node::Expression(Expression::Variable(v))) =>
			{
				names.insert(&v.name);
			},
			Event::Enter(Node::Expression(Expression::Binding(b))) =>
			{
				names.insert(&b.name);
			},
			_ =>
			{}
		}
	}
	names
}

/// Build a [`DuplicateParameter`](DiagnosticKind::DuplicateParameter)
/// diagnostic with a rename suggestion whose fresh name collides with no name
/// in the function, so that the suggestion resolves this error without
/// introducing another, though a source with several errors needs the fixes of
/// all of them.
///
/// The primary [`span`](Diagnostic::span) points at the duplicate
/// occurrence; a single [`RelatedLabel`] on [`related`](Diagnostic::related)
/// points at the first occurrence. The suggestion splices a fresh name
/// produced by [`suggest_rename_with_pool`] into the duplicate span —
/// references in the body are deliberately left alone because there is no safe
/// way to pick which one of them (if any) was intended to refer to a different
/// binding. The fresh name avoids every name in use, not just the parameters,
/// so that it neither collides with a local binding, as `{y}` would in
/// `{x}, {x}: {y}@(1)`, nor captures a reference to an external variable.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text from which `ast` was parsed.
///
/// # Parameters
/// - `ast`: The parsed function (used to enumerate the names in use).
/// - `name`: The duplicated parameter name.
/// - `first`: The span of the first occurrence of `name`.
/// - `duplicate`: The span of the duplicate occurrence that triggered the
///   error.
///
/// # Returns
/// A fully-formed [`Diagnostic`] with one rename [`Suggestion`].
fn make_duplicate_parameter<'src>(
	ast: &Function<'src>,
	name: &str,
	first: SourceSpan,
	duplicate: SourceSpan
) -> Diagnostic
{
	let used = collect_in_use_names(ast);
	let fresh = suggest_rename_with_pool(name, &used);
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
			edits: vec![replacement(duplicate.start, duplicate.end, &fresh)]
		}]
	}
}

/// Suggest a fresh name that does not collide with any name in `used`, for
/// the rename suggestions of duplicate parameters and of
/// [local-binding](crate::ast::Binding) errors alike.
///
/// When the name is a single ASCII letter — the common case, given dice
/// expressions tend to use terse names like `x`, `y`, `a`, `b` — the
/// replacement is the first unused letter encountered by **bumping from the
/// highest attested same-case single-letter name** and wrapping through the
/// alphabet. So `{x}, {x}` becomes `{x}, {y}` (bump `x` → `y`),
/// `{a}, {b}, {b}, {c}` becomes `{a}, {b}, {d}, {c}` (bump highest `c` → `d`),
/// `{a}, {x}, {x}` becomes `{a}, {x}, {y}` (bump highest `x` → `y`), and
/// `{a}, {z}, {z}` becomes `{a}, {z}, {b}` (bump `z` wraps past `a`, which is
/// taken, and lands on `b`). Continuing from the user's apparent "trajectory"
/// is usually more natural than filling the earliest alphabetic gap, which
/// would produce `{a}, {x}, {b}` in the third example.
///
/// When the name has more than one character, or all 26 letters in the
/// matching case are already in use, the replacement falls through to a `newN`
/// form (`new0`, `new1`, …), where `N` is the first non-negative integer for
/// which the resulting name is unused. The `newN` loop likewise skips over
/// names the user has already adopted, so `{abc}, {new0}, {abc}` becomes
/// `{abc}, {new0}, {new1}`.
///
/// # Parameters
/// - `duplicate`: The name to rename.
/// - `used`: The set of names already in use, against which the candidate must
///   not collide.
///
/// # Returns
/// A fresh name that does not appear in `used`.
fn suggest_rename_with_pool(duplicate: &str, used: &HashSet<&str>) -> String
{
	let mut chars = duplicate.chars();
	if let (Some(c), None) = (chars.next(), chars.next())
		&& c.is_ascii_alphabetic()
	{
		let start = if c.is_ascii_uppercase() { b'A' } else { b'a' };
		// The highest attested same-case single-letter name. The duplicate
		// itself is always attested, so `highest` is at least `c`.
		let highest = used
			.iter()
			.filter_map(|n| {
				let mut cs = n.chars();
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
//                                The doctor.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Diagnose a source string, collecting all parse errors, semantic errors, and
/// suggested fixes.
///
/// The doctor runs two alternative passes. If the source parses cleanly, the
/// doctor runs the semantic [`Validator`] against the AST and translates any
/// error into a corresponding [`Diagnostic`]. Otherwise, it parses the source
/// once more, in the parser's recovery mode: wherever the parse cannot
/// continue, the parser consults the doctor, which diagnoses the failure and,
/// if it can, answers a repair that corresponds to the first suggestion of the
/// diagnostic, so that the parse continues as though the source had been
/// edited. This continues until the parse succeeds (all errors collected) or
/// an unfixable error is encountered. The semantic pass is skipped after any
/// fix, because attaching semantic diagnostics to fix-synthesized characters
/// the user never typed would mislead them about what their source actually
/// says.
///
/// ```mermaid
/// sequenceDiagram
///     participant D as diagnose
///     participant P as Parser
///     participant R as Doctor
///     participant V as Validator
///     D->>P: parse(source)
///     alt parses cleanly
///         P-->>D: AST
///         D->>V: validate(AST)
///         V-->>D: semantic error, if any
///     else fails
///         P-->>D: error
///         Note over D,R: blank the stray whitespace of the source to spaces
///         D->>P: parse_recovering(text, doctor)
///         loop each failure
///             P->>R: repair(failure)
///             R->>R: diagnose, record the fix
///             R-->>P: repair, or stop if unfixable
///         end
///         P-->>D: recovered AST, or error
///         D->>R: finish(recovered)
///         R-->>D: diagnostics, corrected source
///     end
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
/// Designed for keystroke-speed execution. The doctor parses the source at most
/// three times, whatever the number of errors, and takes time linear in the
/// length of the source: the clean parse, the recovering parse, and, for
/// formal parameters without a `:`, a parse of the completed definition. The
/// semantic pass runs at most once per call.
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
/// An error collected by the recovering parse:
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
/// let result = diagnose("{x}, {x}: {x}");
/// assert_eq!(result.diagnostics.len(), 1);
/// assert!(matches!(
///     result.diagnostics[0].kind,
///     DiagnosticKind::DuplicateParameter { .. }
/// ));
/// // Semantic diagnostics don't prevent `corrected_source`; the original
/// // source already parses, so it is returned verbatim.
/// assert_eq!(result.corrected_source.as_deref(), Some("{x}, {x}: {x}"));
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub fn diagnose(source: &str) -> DiagnoseResult
{
	// Fast path: if the source parses cleanly, skip the recovering parse and
	// run the semantic pass directly on the AST. Fixing up a syntactically
	// invalid source first and then reporting a semantic error against the
	// fix-synthesized text would attach diagnostics to characters the user
	// never typed, which is confusing — so semantic checks run only on the
	// unmodified original.
	if let Ok(ast) = Parser::parse(source)
	{
		return DiagnoseResult {
			diagnostics: run_validator(&ast),
			corrected_source: Some(source.to_string())
		};
	}

	// The parser reads the stray whitespace as spaces, and the doctor
	// diagnoses it instead.
	let strays = stray_whitespace(source);
	let mut text = source.as_bytes().to_vec();
	for &(start, end) in &strays
	{
		text[start..end].fill(b' ');
	}
	let text = String::from_utf8(text)
		.expect("replacing whole characters with spaces preserves UTF-8");
	let mut doctor = Doctor::new(&text, source, strays);
	let recovered = Parser::parse_recovering(&text, &mut doctor).is_ok();
	doctor.finish(recovered)
}
