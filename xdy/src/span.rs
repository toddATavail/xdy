//! # Source spans
//!
//! Herein are the [`SourceSpan`] type — a byte range in source text suitable
//! for highlighting a region — and the [`Spanned`] trait, which provides
//! uniform access to spans across heterogeneous abstract syntax tree (AST)
//! and [diagnostic](crate::diagnostics) node types. `SourceSpan` is a
//! cross-cutting primitive shared by the parser, AST, compiler, and
//! diagnostic layers, so it lives at the crate root rather than within any
//! single pass.

use std::fmt::{self, Display, Formatter};

////////////////////////////////////////////////////////////////////////////////
//                                Source span.                                //
////////////////////////////////////////////////////////////////////////////////

/// A byte range in source text, suitable for highlighting a region.
///
/// # Notes
/// Both `start` and `end` are byte offsets. `end` is exclusive, following the
/// standard Rust range convention. The default span is `0..0`, which
/// conventionally represents a synthetic or placeholder location; see
/// [`Spanned::untethered`] for the motivating use case.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SourceSpan
{
	/// The start byte offset (inclusive).
	pub start: usize,

	/// The end byte offset (exclusive).
	pub end: usize
}

impl SourceSpan
{
	/// The synthetic span — the `0..0` placeholder used for nodes that have
	/// no real source location, including AST nodes round-tripped through a
	/// format that does not preserve position (e.g., S-expressions without
	/// span metadata) and ASTs produced by [`Spanned::untethered`].
	///
	/// This is identical to `<SourceSpan as Default>::default()`, but is
	/// available in `const` contexts (such as promoted temporaries), which
	/// the derived [`Default`] implementation is not.
	pub const SYNTHETIC: SourceSpan = SourceSpan { start: 0, end: 0 };
}

impl Display for SourceSpan
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}..{}", self.start, self.end)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Spanned trait.                               //
////////////////////////////////////////////////////////////////////////////////

/// Uniform access to the [source span](SourceSpan) of an abstract syntax tree
/// (AST) node, plus a facility for producing a position-independent copy of a
/// value.
///
/// Every AST node carries a span referencing the original source text, so
/// compile-time diagnostics (andthe `xdy!` procedural macro) can report errors
/// with caret-level precision. Span metadata participates in equality and
/// hashing by default, so two parses of the same expression at different
/// positions are *not* equal — this keeps the derived traits honest.
///
/// When structural comparison without regard to position is desired — for
/// instance, in tests that care about the shape of an AST but not the byte
/// ranges that produced it — use [`untethered`](Self::untethered) on both
/// operands: `a.untethered() == b.untethered()`. The name captures the
/// metaphor: a spanned value is *tethered* to a specific lexical location, and
/// `untethered()` produces an equivalent value freed from that location, with
/// every span zeroed out.
pub trait Spanned
{
	/// Answer the [source span](SourceSpan) of this value.
	///
	/// # Returns
	/// The source span of this value. For enum node types such as
	/// [`Expression`](crate::ast::Expression),
	/// [`DiceExpression`](crate::ast::DiceExpression), and
	/// [`ArithmeticExpression`](crate::ast::ArithmeticExpression), this
	/// dispatches to the span of the active variant.
	fn span(&self) -> SourceSpan;

	/// Answer a structurally identical copy of this value with every
	/// [source span](SourceSpan) replaced with [`SourceSpan::default`].
	///
	/// # Returns
	/// A value equal to `self` in every respect except that all spans —
	/// including those nested inside children — are zeroed out. This enables
	/// position-independent structural equality via
	/// `a.untethered() == b.untethered()`.
	fn untethered(&self) -> Self
	where
		Self: Sized;
}
