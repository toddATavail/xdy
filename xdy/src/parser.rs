//! # Parser
//!
//! Herein is the parser for the `xDy` language. [`Parser::parse`] is the main
//! entry point, which discards leading and trailing whitespace around a
//! [function](Function). The recognized grammar is as follows, in Extended
//! Backus-Naur Form (EBNF), where non-terminals are in lowercase and terminals
//! are in uppercase:
//!
//! ```text
//! function       ::= parameters? expression
//! parameters     ::= parameter (',' parameter)* ':'
//! parameter      ::= name
//! name           ::= '{' WHITESPACE* IDENTIFIER WHITESPACE* '}'
//! expression     ::= add_sub
//! add_sub        ::= mul_div_mod (('+' | '-') mul_div_mod)*
//! mul_div_mod    ::= unary (('*' | '×' | '/' | '÷' | '%') unary)*
//! unary          ::= '-' unary | exponent
//! exponent       ::= primary ('^' unary)?
//! primary        ::= range | call | dice | group | variable | binding
//!                  | CONSTANT
//! call           ::= ('highest' | 'lowest') '(' expression (',' expression)+
//!                    ')'
//! group          ::= '(' expression ')'
//! variable       ::= name
//! binding        ::= name '@' '(' expression ')'
//! range          ::= '[' expression ':' expression ']'
//! dice           ::= base_dice (drop_clause | keep_clause | reroll_clause
//!                    | explode_clause)*
//! base_dice      ::= dice_count D_OPERATOR (standard_faces | custom_faces)
//! dice_count     ::= CONSTANT | variable | binding | group
//! standard_faces ::= INTEGER | variable | binding | group
//! custom_faces   ::= '[' INTEGER (',' INTEGER)* ']'
//! drop_clause    ::= 'drop' ('lowest' | 'highest') drop_expression?
//! keep_clause    ::= 'keep' ('lowest' | 'highest') drop_expression?
//! reroll_clause  ::= 'reroll' predicate limit?
//! explode_clause ::= 'explode' predicate limit?
//! predicate      ::= comparator standard_faces
//!                  | 'in' '[' (INTEGER (',' INTEGER)*)? ']'
//!                  | 'in' standard_faces ('..' | '..=') standard_faces
//! comparator     ::= '=' | '!=' | '≠' | '<' | '<=' | '≤' | '>' | '>=' | '≥'
//! limit          ::= (drop_expression | 'unlimited' | '∞') ('time' | 'times')
//! drop_expression::= CONSTANT | variable | binding | group
//! CONSTANT       ::= DIGIT+
//! INTEGER        ::= '-'? DIGIT+
//! D_OPERATOR     ::= 'd' | 'D'
//! IDENTIFIER     ::= NAME_CHAR (NAME_CHAR | WHITESPACE+ NAME_CHAR)*
//! NAME_CHAR      ::= any identifier character (is_identifier_char) except
//!                    whitespace
//! WHITESPACE     ::= any whitespace (char::is_whitespace), e.g., ' ', '\t',
//!                    '\n', or U+00A0
//! ```
//!
//! A `CONSTANT` is unsigned, so a minus before one is the unary operator:
//! `-2 ^ 3` is `-(2 ^ 3)`, and `-3D6` is `-(3D6)`. The parser folds a negated
//! literal into a single [constant](crate::ast::Constant), as in `-5`, to the
//! same effect. Only the faces of dice take a signed `INTEGER`, as in `3D-6`
//! and `1D[-1, 0, 1]`, where nothing but a face may follow. Each combinator
//! accepts just what its rule does, so [`constant`], [`primary`],
//! [`dice_count`], and [`dice`] reject `-3`, while [`integer`] and
//! [`standard_faces`] accept it, as do the thresholds and endpoints of the
//! predicates of rerolls and explodes, since custom dice may have negative
//! faces, e.g., `3D[-2, -1, 0, 1] reroll in -2..0`. In particular, a drop
//! expression, which is the count of a drop or keep clause, or of the limit of
//! a reroll or explode clause, never begins with `-`, so after a clause, a `-`
//! is subtraction: `4D6 drop lowest -1` is `(4D6 drop lowest) - 1`, and `4D6
//! reroll = 1 -1` is `(4D6 reroll = 1) - 1`. A count of zero or less drops
//! nothing, keeps nothing, rerolls nothing, or explodes nothing, so no one
//! means to write a negative count, but one may be grouped, as in `4D6 drop
//! lowest (-1)`.
//!
//! The clauses of a dice expression apply in the order written. A keep counts
//! from the dice that the clauses before it left kept, so `4D6 drop lowest
//! keep highest 2` keeps the highest 2 of the 3 dice that the drop left, and a
//! reroll rerolls only those dice, so `4D6 drop lowest reroll = 1` rerolls the
//! ones among the 3 dice that the drop left.
//!
//! A reroll clause rerolls each die that matches its predicate once, or as many
//! times as its limit says: `n time` or `n times`, which read alike, or `∞
//! times` or `unlimited times`, until the die no longer matches. Its threshold
//! or the endpoints of its range are each evaluated once, before any reroll,
//! and a list holds literals only. A range `start..end` is half-open, and
//! `start..=end` inclusive; either is empty if its end precedes its start. The
//! comparators `!=`, `<=`, and `>=` may also be written `≠`, `≤`, and `≥`.
//!
//! An explode clause takes the same predicates and limits, but rolls each die
//! that matches again, and adds the new die to it, as one die of a chain: `3D6
//! explode = 6` adds another die to each six, and `3D6 explode = 6 ∞ times`
//! keeps adding while the new die is a six. Later drops and keeps count each
//! chain as one die, worth its total, so `4D6 explode = 6 keep highest 3`
//! keeps the three highest chains, while a later reroll considers each die of
//! a chain, and a later explode only its end. If an unlimited explode matches
//! every face of its die, the chain never ends, and saturates instead, as
//! [`Explode`](crate::ast::Explode) describes.
//!
//! A call of `highest` or `lowest` answers the greatest or the least of the
//! values of its arguments, two or more full expressions, each evaluated in
//! turn, left to right, e.g., `highest(1D20, 1D20)` or `lowest({POW} * 5,
//! 99)`. An argument is a value, not a pool of dice, so `highest(4D6, 10)`
//! compares the sum of `4D6` with `10`; `4D6 keep highest` keeps the highest
//! die. A call is a primary expression, which may stand wherever an expression
//! may begin, but nowhere that only a constant, a variable, a binding, or a
//! group may: as a dice count, faces, the count of a clause, or an operand of a
//! reroll or explode, it must be grouped, as a range must, e.g., `4D6 drop
//! lowest (lowest({x}, 2))`. So a drop or keep clause still reads `lowest` or
//! `highest` as its direction, whatever follows it: `4D6 drop lowest (2)` and
//! `4D6 drop lowest(2)` both drop the lowest 2 dice. The keyword is
//! recognized only where a `(` follows it, after any whitespace, so a name that
//! begins with it, such as `lowest roll` without its braces, is no call.
//!
//! Tokens may be separated by spaces, horizontal tabs, line feeds, and carriage
//! returns, but by no other whitespace; the [doctor](crate::diagnostics)
//! replaces stray whitespace between tokens, such as U+00A0 NO-BREAK SPACE,
//! with a space. The keywords `drop`, `keep`, `lowest`, `highest`, `reroll`,
//! `explode`, `in`, `time`, `times`, and `unlimited` are lowercase, but the
//! dice operator may be `d` or `D`. The keywords of a reroll or explode clause
//! are contextual: each must not run into a word after it, so `rerolled` is
//! not `reroll` and `ed`, nor `exploded` `explode` and `d`, but it may abut a
//! digit or a symbol, as in `reroll=1`, `explode>=5`, or `in1..3`.
//!
//! Every name is braced, whether it names a parameter, a variable, or a
//! binding, so no name can run into the text around it. A lone braced name
//! without a `:` after it begins the expression, as a variable or a binding,
//! rather than declaring a parameter.
//!
//! An identifier may contain any visible character but a brace, e.g.,
//! `{weapon: 2/3}` or `{$env|weapon}`, and whitespace of any kind, so a long
//! name may be broken over lines. But a name is never distinguished by its
//! whitespace: the whitespace inside the braces, around the identifier, is not
//! part of it, and every run of whitespace within it collapses to a single
//! space, so `{ a b }`, `{a  b}`, and `{a`, a line feed, and `   b}` all name
//! `a b`. The identifier character set is defined once, by
//! [`is_identifier_char`], which the parser, the [S-expression](crate::s_expr)
//! reader, and the assembler share, and a name is canonicalized once, by
//! [`canonical_name`]. The S-expression reader and the assembler accept only
//! [canonical](is_canonical_name) names, as their writers emit. The
//! [diagnostics](crate::diagnostics) also look for names that lack their
//! braces, but by a narrower character set that stops at operators and
//! delimiters.
//!
//! The following railroad diagram is generated from the EBNF grammar above:
#![doc = include_str!("../doc/xdy.svg")]
//! Parsing is based on the [`nom`] library, which provides a combinator-based
//! approach to parsing.
//!
//! The grammar is recursive, but the parser is not. The combinators for the
//! recursive productions, from [`function`] down to [`group`] and [`binding`],
//! share a private engine that performs recursive descent with an explicit
//! stack on the heap, so parsing consumes no more of the machine stack for
//! deeply nested input than for shallow input. Parse time is linear in the
//! length of the input.

mod combinators;
mod engine;
mod errors;

pub use combinators::*;
pub(crate) use engine::{
	FailureSite, LoneArgument, Operand, PLACEHOLDER_OPERAND, PredicatedKind,
	Recovery, Repair, Silent, Site, Strict, TokenSink
};
#[cfg(test)]
pub(crate) use engine::{
	PLACEHOLDER_FACE, PLACEHOLDER_FACES, PLACEHOLDER_NAME, steps
};
pub use errors::*;

use nom::{
	IResult, Parser as _,
	character::complete::multispace0,
	combinator::all_consuming,
	error::{ContextError as _, ErrorKind, ParseError as _, context},
	sequence::delimited
};

use crate::ast::Function;

////////////////////////////////////////////////////////////////////////////////
//                                  Parser.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The `xDy` parser. Use [`Parser::parse`] as the high-level entry point; the
/// individual parser combinators exposed by this module are available for
/// low-level uses but are not recommended for most clients.
#[derive(Copy, Clone, Debug, Default)]
pub struct Parser;

impl Parser
{
	/// Parse a function definition, discarding leading and trailing
	/// whitespace. This is the intended high-level entry point for the parser.
	/// The individual parser combinators are available for low-level uses,
	/// but not recommended for most clients.
	///
	/// # Parameters
	/// - `input`: The input text to parse.
	///
	/// # Returns
	/// The parsed function definition.
	///
	/// # Errors
	/// * [`ParseError`] if the input could not be parsed, including if anything
	///   but whitespace follows the function definition.
	pub fn parse(input: &str) -> Result<Function<'_>, ParseError<'_>>
	{
		let input = Span::new(input);
		all_consuming(delimited(
			multispace0,
			context(FUNCTION_CONTEXT, function),
			multispace0
		))
		.parse_complete(input)
		.map_err(|e| match e
		{
			nom::Err::Error(e) => e,
			nom::Err::Failure(e) => e,
			nom::Err::Incomplete(_) => unreachable!()
		})
		.map(|(_, f)| f)
	}

	/// Parse a function definition, discarding leading and trailing
	/// whitespace, as [`parse`](Parser::parse) does, but in _recovery mode_:
	/// wherever the parse cannot continue, consult a [policy](Recovery),
	/// which may [repair](Repair) the failure so that the parse continues
	/// as though the source had been edited. The source is parsed only
	/// once, whatever the number of repairs.
	///
	/// With a policy that repairs nothing, the result is exactly that of
	/// [`parse`](Parser::parse).
	///
	/// The parse reports every token of the source that it reads to a
	/// [sink](TokenSink), in source order, but none that a repair supplies.
	///
	/// # Type parameters
	/// - `'src`: The lifetime of the source text.
	/// - `R`: The type of the policy.
	/// - `S`: The type of the sink.
	///
	/// # Parameters
	/// - `input`: The input text to parse.
	/// - `policy`: The recovery policy.
	/// - `sink`: The sink of the tokens that the parse reads.
	///
	/// # Returns
	/// The parsed function definition, in which every repair appears as the
	/// token that it supplied.
	///
	/// # Errors
	/// * [`ParseError`] if the policy declined to repair a failure.
	pub(crate) fn parse_recovering<'src, R: Recovery<'src>, S: TokenSink>(
		input: &'src str,
		policy: &mut R,
		sink: &mut S
	) -> Result<Function<'src>, ParseError<'src>>
	{
		let input = Span::new(input);
		let skipped: IResult<Span, Span, ParseError> = multispace0(input);
		// `multispace0` accepts the empty string, so it cannot fail.
		let input = skipped.map_or(input, |(rest, _)| rest);
		let (rest, function) = match engine::run_recovering(input, policy, sink)
		{
			Ok((rest, value)) => (rest, value.into_function()),
			Err(nom::Err::Error(e) | nom::Err::Failure(e)) =>
			{
				return Err(ParseError::add_context(input, FUNCTION_CONTEXT, e));
			},
			Err(nom::Err::Incomplete(_)) => unreachable!()
		};
		let skipped: IResult<Span, Span, ParseError> = multispace0(rest);
		let rest = skipped.map_or(rest, |(rest, _)| rest);
		if rest.fragment().is_empty()
		{
			return Ok(function);
		}
		// As `all_consuming`.
		let site = FailureSite {
			site: Site::TrailingInput,
			error: ParseError::from_error_kind(rest, ErrorKind::Eof)
		};
		match policy.repair(&site)
		{
			Repair::Fix => Ok(function),
			_ => Err(site.error)
		}
	}
}
