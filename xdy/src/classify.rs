//! # Source classification
//!
//! Herein are the types of source classification, which tags the tokens of an
//! expression's source text with their semantic [roles](Role), for editors,
//! syntax highlighters, and hosts that embed `xDy` expressions in text of
//! their own. A classification is an ordered sequence of non-overlapping
//! [tokens](Token), each a [span](SourceSpan) of the source and its role.
//!
//! A role says what a token _is_, never how to show it: a host maps roles into
//! its own vocabulary of styles. The roles are as fine as the grammar is, so
//! that the kinds of literal, the clauses, and the kinds of name are told
//! apart, and the arithmetic operators are grouped by precedence, so that a
//! stylesheet may show precedence at a glance.
//!
//! Whitespace that may separate tokens, i.e., spaces, horizontal tabs, line
//! feeds, and carriage returns, belongs to no token. Every other byte of the
//! source belongs to exactly one, if only as [`Role::Unrecognized`].

use crate::span::SourceSpan;

////////////////////////////////////////////////////////////////////////////////
//                                   Roles.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The semantic role of a [token](Token) of source text.
///
/// # Notes
/// New productions of the grammar may add roles in minor releases, so the
/// enumeration is non-exhaustive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Role
{
	/// An unsigned literal: an arithmetic constant, the count of a dice
	/// expression, the count of a drop or keep clause, or the count of the
	/// limit of a reroll or explode clause.
	///
	/// # Examples
	/// `2` in `1D6 + 2`, `3` in `3D6`, `1` in `4D6 drop lowest 1`, `2` in
	/// `3D6 reroll = 1 2 times`
	Constant,

	/// A signed literal: standard faces, a custom face, the threshold or an
	/// endpoint of the predicate of a reroll or explode clause, or a member of
	/// its list. A minus sign that begins the literal belongs to it.
	///
	/// # Examples
	/// `6` in `3D6`, `-6` in `3D-6`, `-1` in `1D[-1, 0, 1]`, `1` in
	/// `3D6 reroll = 1`, `5` in `2D6 explode in [5, 6]`
	Face,

	/// The dice operator.
	///
	/// # Examples
	/// `D` in `3D6`, `d` in `3d6`
	DiceOperator,

	/// The keyword of a drop or keep clause.
	///
	/// # Examples
	/// `drop` in `4D6 drop lowest`, `keep` in `4D6 keep highest 3`
	Selection,

	/// The keyword of a reroll clause.
	///
	/// # Examples
	/// `reroll` in `3D6 reroll = 1`
	Reroll,

	/// The keyword of an explode clause.
	///
	/// # Examples
	/// `explode` in `3D6 explode = 6`
	Explode,

	/// The direction of a drop or keep clause.
	///
	/// # Examples
	/// `lowest` in `4D6 drop lowest`, `highest` in `4D6 keep highest 3`
	Direction,

	/// The keyword of a call of the greatest or the least of expressions.
	///
	/// # Examples
	/// `highest` in `highest(1D20, 1D20)`, `lowest` in `lowest({x}, 99)`
	Call,

	/// The operator of the predicate of a reroll or explode clause: a
	/// comparator, or the keyword `in`.
	///
	/// # Examples
	/// `=` in `3D6 reroll = 1`, `≥` in `3D6 explode ≥ 5`, `in` in
	/// `3D6 reroll in 1..3`
	Predicate,

	/// A range operator: `..` or `..=` in the predicate of a reroll or explode
	/// clause, or the `:` of a range expression.
	///
	/// # Examples
	/// `..` in `3D6 reroll in 1..3`, `..=` in `3D6 explode in 5..=6`, `:` in
	/// `[1:6]`
	RangeOperator,

	/// A word of the limit of a reroll or explode clause, other than its count.
	///
	/// # Examples
	/// `times` in `3D6 reroll = 1 2 times`, `unlimited` and `times` in
	/// `3D6 reroll = 1 unlimited times`, `∞` and `times` in
	/// `3D6 explode = 6 ∞ times`
	Limit,

	/// An additive operator: addition or subtraction.
	///
	/// # Examples
	/// `+` in `1D6 + 2`, `-` in `1D6 - 2`
	Additive,

	/// A multiplicative operator: multiplication, division, or remainder.
	///
	/// # Examples
	/// `*` in `2 * 1D6`, `×` in `2 × 1D6`, `/` in `1D6 / 2`, `÷` in
	/// `1D6 ÷ 2`, `%` in `1D6 % 2`
	Multiplicative,

	/// The exponentiation operator.
	///
	/// # Examples
	/// `^` in `2 ^ 1D4`
	Exponent,

	/// The negation operator, unary minus. A constant is unsigned, so a minus
	/// before one is negation, even where the parser folds the two into a
	/// single negative constant.
	///
	/// # Examples
	/// `-` in `-1D6`, `-(1D6 + 2)`, and `-5`
	Negation,

	/// The binding operator of subexpression naming.
	///
	/// # Examples
	/// `@` in `{x}@(1D6) + {x}`
	Bind,

	/// A separator: a comma, or the colon that ends the formal parameters.
	///
	/// # Examples
	/// `,` in `{x}, {y}: {x} + {y}`, `1D[1, 2, 3]`, and `highest(1D20, 1D20)`;
	/// `:` in `{x}: {x}D6`
	Separator,

	/// A parenthesis of a group, of a binding, or of a call.
	///
	/// # Examples
	/// `(` and `)` in `(1D6 + 2) * 2`, `{x}@(1D6)`, and `highest(1D20, 1D20)`
	Parenthesis,

	/// A bracket of a range expression.
	///
	/// # Examples
	/// `[` and `]` in `[1:6]`
	RangeBracket,

	/// A bracket of the faces of custom dice.
	///
	/// # Examples
	/// `[` and `]` in `1D[-1, 0, 1]`
	FacesBracket,

	/// A bracket of the list of the predicate of a reroll or explode clause.
	///
	/// # Examples
	/// `[` and `]` in `2D6 explode in [5, 6]`
	ListBracket,

	/// A brace of a name.
	///
	/// # Examples
	/// `{` and `}` in `{x}`
	NameBrace,

	/// The identifier of a formal parameter, where the function declares it.
	///
	/// # Examples
	/// The first `x` in `{x}: {x}D6`
	ParameterDeclaration,

	/// The identifier of a formal parameter, where the body refers to it.
	///
	/// # Examples
	/// The second `x` in `{x}: {x}D6`
	ParameterReference,

	/// The identifier of a local binding, where subexpression naming binds it.
	///
	/// # Examples
	/// The first `x` in `{x}@(1D6) + {x}`
	BindingDeclaration,

	/// The identifier of a local binding, where the expression refers to it.
	///
	/// # Examples
	/// The second `x` in `{x}@(1D6) + {x}`
	BindingReference,

	/// The identifier of an external variable, which the expression neither
	/// declares nor binds, and so only refers to.
	///
	/// # Examples
	/// `STR` in `1D20 + {STR}`
	External,

	/// The identifier of a name that the scoping rules of the compiler cannot
	/// place, e.g., a reference to a binding that precedes the binding.
	///
	/// # Examples
	/// The first `x` in `{x} + {x}@(1D6)`
	Unresolved,

	/// Source text that is not a token of the grammar where it occurs, e.g.,
	/// a closing delimiter that nothing opened, input that follows a complete
	/// expression, whitespace that may not separate tokens, such as U+00A0
	/// NO-BREAK SPACE, or a name without its braces. A role describes the
	/// source as written, so text that the parser could make sense of only by
	/// reading it as something else is unrecognized too.
	///
	/// # Examples
	/// `)` in `)3D6`, `x` in `{y} + x`
	Unrecognized
}

////////////////////////////////////////////////////////////////////////////////
//                                  Tokens.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A token of source text, tagged with its semantic [role](Role).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Token
{
	/// The span of the token, as byte offsets into the classified source.
	pub span: SourceSpan,

	/// The role of the token.
	pub role: Role,

	/// The nesting depth of a [parenthesis](Role::Parenthesis) or a bracket
	/// ([range](Role::RangeBracket), [faces](Role::FacesBracket), or
	/// [list](Role::ListBracket)), counted across all of them, from `0` for
	/// an outermost pair, so that a surface may color delimiters by depth.
	/// Both delimiters of a pair have the same depth. `None` for every other
	/// role, including [braces](Role::NameBrace), which never nest.
	pub depth: Option<usize>
}
