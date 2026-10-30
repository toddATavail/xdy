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
//!
//! [`classify`] is the entry point. It is total: it classifies any text, even
//! malformed or unfinished text, such as an editor's buffer mid-keystroke, and
//! it derives every token from the parser itself, so that it never disagrees
//! with the language.

use std::{borrow::Cow, collections::HashSet};

use crate::{
	Parser, diagnostics,
	parser::{Strict, canonical_name, is_token_space},
	span::SourceSpan
};

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
	/// place: a reference to a binding that precedes the end of the binding,
	/// whether it precedes the binding or lies within its bound expression; a
	/// formal parameter declared again; or a binding of the name of a formal
	/// parameter or of an earlier binding. The compiler rejects every function
	/// with an unresolved name. Only the later of two clashing declarations
	/// is unresolved, and references to the name resolve to the earlier.
	///
	/// # Examples
	/// The first `x` in `{x} + {x}@(1D6)`, the second `x` in `{x}@({x})`,
	/// the second `x` in `{x}, {x}: {x}`, and the second `x` in
	/// `{x}@(1) + {x}@(2)`
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
	///
	/// In malformed text, delimiters pair as the text nests them: a closer
	/// pairs with the nearest open delimiter of its own role, and closes every
	/// delimiter opened since, whose closers are missing. An opener whose
	/// closer is missing has a depth all the same. A closer that nothing
	/// opened is [unrecognized](Role::Unrecognized), and has no depth.
	pub depth: Option<usize>
}

////////////////////////////////////////////////////////////////////////////////
//                              Classification.                               //
////////////////////////////////////////////////////////////////////////////////

/// Classify the tokens of an expression's source text.
///
/// The parser reports each token as it reads it. If the source parses cleanly,
/// the classification is exactly the tokens of the parse. Otherwise, the
/// parser reads the source in the recovery mode of the
/// [doctor](diagnostics::diagnose), which repairs each failure that it can
/// fix as though the source had been edited by the fix, e.g., as though a
/// missing `)` were present, and parses on. A token that a repair supplies is
/// not in the source, so it is not classified. Every region of the source that
/// the parse could not make sense of is [unrecognized](Role::Unrecognized):
/// text that a repair discards, such as a closing delimiter that nothing
/// opened, or input after a complete expression; text that a repair reads as
/// something else, such as a name without its braces; and everything after a
/// failure that the doctor cannot fix. Each run of such text between
/// whitespace that may separate tokens is a token of its own.
///
/// Every name is resolved by the scoping rules of the compiler, as a formal
/// parameter, a local binding, or an external variable, and as a declaration
/// or a reference, or else is [unresolved](Role::Unresolved). Resolution of a
/// function that compiles agrees with the compiler. Every parenthesis and
/// bracket has its [depth](Token::depth), even in malformed text.
///
/// # Parameters
/// - `source`: The source text: the whole of an expression, which a host that
///   embeds expressions has already delimited.
///
/// # Returns
/// The tokens, in source order, which do not overlap, and which cover every
/// byte of the source but whitespace between tokens and the whitespace that
/// pads a name within its braces.
///
/// # Performance
/// Classification parses the source at most twice, cleanly and then
/// recovering, and neither compiles nor optimizes, and then resolves names and
/// counts depths in two passes over the tokens. It takes time linear in the
/// length of the source, and consumes no more of the machine stack for deeply
/// nested input than for shallow input, so it may run on every keystroke.
///
/// # Examples
/// ```rust
/// use xdy::classify::{Role, classify};
///
/// let source = "3D6 + )";
/// let roles = classify(source)
///     .iter()
///     .map(|token| (&source[token.span.start..token.span.end], token.role))
///     .collect::<Vec<_>>();
/// assert_eq!(
///     roles,
///     [
///         ("3", Role::Constant),
///         ("D", Role::DiceOperator),
///         ("6", Role::Face),
///         ("+", Role::Additive),
///         (")", Role::Unrecognized)
///     ]
/// );
/// ```
pub fn classify(source: &str) -> Vec<Token>
{
	let mut tokens = Vec::new();
	if Parser::parse_recovering(source, &mut Strict, &mut tokens).is_ok()
	{
		resolve(source, &mut tokens);
		return tokens;
	}
	tokens.clear();
	diagnostics::recover(source, &mut tokens);
	resolve(source, &mut tokens);
	fill(source, tokens)
}

/// Resolve the names among the tokens that a parse reported, and give each
/// delimiter its depth. The engine reports every reference as
/// [unresolved](Role::Unresolved), and every declaration as it reads it, so
/// resolution places each by the scoping rules of the
/// [validator](crate::Validator), in source order:
///
/// - The first declaration of a formal parameter is a [parameter
///   declaration](Role::ParameterDeclaration), and every later one is
///   unresolved.
/// - A binding is a [binding declaration](Role::BindingDeclaration), unless its
///   name is that of a formal parameter, or of an earlier binding, when it is
///   unresolved.
/// - A reference to a formal parameter is a [parameter
///   reference](Role::ParameterReference).
/// - A reference to a binding is a [binding reference](Role::BindingReference)
///   once the binding is complete, i.e., after the `)` that matches the `(`
///   just after its `@`, and is unresolved before then, whether it precedes the
///   binding or lies within its bound expression. A binding whose `(` or `)` a
///   repair supplied is never complete.
/// - Every other reference is [external](Role::External).
///
/// Names are compared [canonically](canonical_name).
///
/// A parenthesis or bracket that opens takes the count of those that are open,
/// and one that closes matches the nearest open delimiter of its own role,
/// which it closes, along with every delimiter opened since, whose closers a
/// repair supplied. A closer that matches no open delimiter, which the engine
/// never reports, takes depth `0`.
///
/// # Parameters
/// - `source`: The source.
/// - `tokens`: The tokens that the parse reported, in source order.
pub(crate) fn resolve(source: &str, tokens: &mut [Token])
{
	let name = |token: &Token| {
		canonical_name(&source[token.span.start..token.span.end])
	};
	// Gather the names that the declarations declare, and the declarations
	// that resolution cannot place.
	let mut parameters = HashSet::<Cow<str>>::new();
	let mut bindings = HashSet::<Cow<str>>::new();
	let mut rejected = HashSet::<usize>::new();
	for (index, token) in tokens.iter().enumerate()
	{
		let declared = match token.role
		{
			Role::ParameterDeclaration => parameters.insert(name(token)),
			Role::BindingDeclaration =>
			{
				let name = name(token);
				!parameters.contains(&name) && bindings.insert(name)
			},
			_ => continue
		};
		if !declared
		{
			rejected.insert(index);
		}
	}
	// Place the names, and give the delimiters their depths. The name of a
	// binding awaits its `@`, and then its `(`, just after the `@`.
	let mut open = Vec::<Delimiter>::new();
	let mut counts = [0usize; 4];
	let mut bound = HashSet::<Cow<str>>::new();
	let mut binding = None;
	let mut awaiting = None;
	for (index, token) in tokens.iter_mut().enumerate()
	{
		let awaited = awaiting.take();
		match token.role
		{
			Role::ParameterDeclaration | Role::BindingDeclaration
				if rejected.contains(&index) =>
			{
				token.role = Role::Unresolved;
				binding = None;
			},
			Role::BindingDeclaration => binding = Some(name(token)),
			Role::Bind => awaiting = binding.take(),
			Role::Unresolved =>
			{
				let name = name(token);
				token.role = if parameters.contains(&name)
				{
					Role::ParameterReference
				}
				else if bound.contains(&name)
				{
					Role::BindingReference
				}
				else if bindings.contains(&name)
				{
					Role::Unresolved
				}
				else
				{
					Role::External
				};
			},
			Role::Parenthesis
			| Role::RangeBracket
			| Role::FacesBracket
			| Role::ListBracket =>
			{
				let slot = kind(token.role);
				if matches!(source.as_bytes()[token.span.start], b'(' | b'[')
				{
					token.depth = Some(open.len());
					counts[slot] += 1;
					open.push(Delimiter {
						role: token.role,
						binding: awaited
							.filter(|_| token.role == Role::Parenthesis)
					});
				}
				else if counts[slot] == 0
				{
					token.depth = Some(0);
				}
				else
				{
					// Close every delimiter opened since the match, whose
					// closers a repair supplied, and then the match.
					while let Some(delimiter) = open.pop()
					{
						counts[kind(delimiter.role)] -= 1;
						if delimiter.role == token.role
						{
							token.depth = Some(open.len());
							bound.extend(delimiter.binding);
							break;
						}
					}
				}
			},
			_ =>
			{}
		}
	}
}

/// An open delimiter, during [resolution](resolve).
struct Delimiter<'src>
{
	/// The role of the delimiter.
	role: Role,

	/// The name of the binding whose bound expression the delimiter opens, if
	/// any. The binding is complete once the delimiter closes.
	binding: Option<Cow<'src, str>>
}

/// Answer the index of the kind of a delimiter, for counting the open
/// delimiters of each kind.
///
/// # Parameters
/// - `role`: The role of the delimiter, a parenthesis or a bracket.
///
/// # Returns
/// The index, below `4`.
fn kind(role: Role) -> usize
{
	match role
	{
		Role::Parenthesis => 0,
		Role::RangeBracket => 1,
		Role::FacesBracket => 2,
		_ => 3
	}
}

/// Fill the gaps between the tokens that a recovering parse reported with
/// [unrecognized](Role::Unrecognized) tokens, one for each run of text between
/// whitespace that may separate tokens. Between a `{` and the identifier after
/// it, or between an identifier and the `}` after it, any whitespace pads the
/// name.
///
/// # Parameters
/// - `source`: The source.
/// - `parsed`: The tokens that the parse reported, in source order.
///
/// # Returns
/// The tokens, gaps filled.
///
/// # Panics
/// In debug builds only, if a parsed token is empty, out of order, overlaps
/// another, or splits a character, which the engine must never report. In
/// release builds, such a token is dropped instead.
fn fill(source: &str, parsed: Vec<Token>) -> Vec<Token>
{
	let mut tokens = Vec::with_capacity(parsed.len() + 1);
	let mut end = 0;
	let mut opened = false;
	for token in parsed
	{
		let SourceSpan {
			start,
			end: token_end
		} = token.span;
		if start < end
			|| token_end <= start
			|| !source.is_char_boundary(start)
			|| !source.is_char_boundary(token_end)
		{
			debug_assert!(
				false,
				"token {:?} is empty, out of order, or not whole in {:?}",
				token, source
			);
			continue;
		}
		let brace =
			|b| token.role == Role::NameBrace && source.as_bytes()[start] == b;
		unrecognized(source, end, start, opened || brace(b'}'), &mut tokens);
		opened = brace(b'{');
		tokens.push(token);
		end = token_end;
	}
	unrecognized(source, end, source.len(), false, &mut tokens);
	tokens
}

/// Classify each run of text within a gap between tokens, between whitespace
/// that may separate tokens, as [unrecognized](Role::Unrecognized).
///
/// # Parameters
/// - `source`: The source.
/// - `start`: The byte offset of the start of the gap.
/// - `end`: The byte offset of the end of the gap.
/// - `padding`: Whether the gap pads a name within its braces, where any
///   whitespace may separate the braces from the identifier.
/// - `tokens`: The tokens, to which the unrecognized tokens are appended.
fn unrecognized(
	source: &str,
	start: usize,
	end: usize,
	padding: bool,
	tokens: &mut Vec<Token>
)
{
	let space = |c: char| {
		if padding
		{
			c.is_whitespace()
		}
		else
		{
			is_token_space(c)
		}
	};
	let mut run = None;
	for (at, c) in source[start..end].char_indices()
	{
		let at = start + at;
		match (space(c), run)
		{
			(true, Some(run_start)) =>
			{
				tokens.push(Token {
					span: SourceSpan {
						start: run_start,
						end: at
					},
					role: Role::Unrecognized,
					depth: None
				});
				run = None;
			},
			(false, None) => run = Some(at),
			_ =>
			{}
		}
	}
	if let Some(run_start) = run
	{
		tokens.push(Token {
			span: SourceSpan {
				start: run_start,
				end
			},
			role: Role::Unrecognized,
			depth: None
		});
	}
}
