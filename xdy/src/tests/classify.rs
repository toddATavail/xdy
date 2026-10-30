//! # Source classification
//!
//! Herein are the tests of the tokens that the parsing engine reports, with
//! their roles, as it parses: a table of sources that, between them, report
//! every role that the engine reports, and the retraction of the tokens of an
//! alternative that a catching frame abandons. That the tokens of every clean
//! parse tile the source is a [property](super::property) test.

use pretty_assertions::assert_eq;

use crate::{
	Parser,
	classify::{Role, Token},
	parser::{Strict, TokenSink as _},
	span::SourceSpan
};

////////////////////////////////////////////////////////////////////////////////
//                                   Roles.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the engine reports every token of each source with the expected
/// role.
#[test]
fn test_roles()
{
	use Role::*;
	let cases: &[(&str, &[(&str, Role)])] = &[
		(
			"{x}, { y }: {x}D6 + -2",
			&[
				("{", NameBrace),
				("x", ParameterDeclaration),
				("}", NameBrace),
				(",", Separator),
				("{", NameBrace),
				("y", ParameterDeclaration),
				("}", NameBrace),
				(":", Separator),
				("{", NameBrace),
				("x", Unresolved),
				("}", NameBrace),
				("D", DiceOperator),
				("6", Face),
				("+", Additive),
				("-", Negation),
				("2", Constant)
			]
		),
		(
			"3d[-1, 0,1] * 2 × 3 / 4 ÷ 5 % 6 - 3D-6",
			&[
				("3", Constant),
				("d", DiceOperator),
				("[", FacesBracket),
				("-1", Face),
				(",", Separator),
				("0", Face),
				(",", Separator),
				("1", Face),
				("]", FacesBracket),
				("*", Multiplicative),
				("2", Constant),
				("×", Multiplicative),
				("3", Constant),
				("/", Multiplicative),
				("4", Constant),
				("÷", Multiplicative),
				("5", Constant),
				("%", Multiplicative),
				("6", Constant),
				("-", Additive),
				("3", Constant),
				("D", DiceOperator),
				("-6", Face)
			]
		),
		(
			"- 2 ^ -(1D4) - - 7",
			&[
				("-", Negation),
				("2", Constant),
				("^", Exponent),
				("-", Negation),
				("(", Parenthesis),
				("1", Constant),
				("D", DiceOperator),
				("4", Face),
				(")", Parenthesis),
				("-", Additive),
				("-", Negation),
				("7", Constant)
			]
		),
		(
			"4D6 drop lowest 1 keep highest",
			&[
				("4", Constant),
				("D", DiceOperator),
				("6", Face),
				("drop", Selection),
				("lowest", Direction),
				("1", Constant),
				("keep", Selection),
				("highest", Direction)
			]
		),
		(
			"3D6 reroll = 1 2 times explode ≥ 6 ∞ times",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("reroll", Reroll),
				("=", Predicate),
				("1", Face),
				("2", Constant),
				("times", Limit),
				("explode", Explode),
				("≥", Predicate),
				("6", Face),
				("∞", Limit),
				("times", Limit)
			]
		),
		(
			"2D6 explode in [5, 6] unlimited time reroll in -1..=2 \
			 reroll in 1..3 1 time reroll in []",
			&[
				("2", Constant),
				("D", DiceOperator),
				("6", Face),
				("explode", Explode),
				("in", Predicate),
				("[", ListBracket),
				("5", Face),
				(",", Separator),
				("6", Face),
				("]", ListBracket),
				("unlimited", Limit),
				("time", Limit),
				("reroll", Reroll),
				("in", Predicate),
				("-1", Face),
				("..=", RangeOperator),
				("2", Face),
				("reroll", Reroll),
				("in", Predicate),
				("1", Face),
				("..", RangeOperator),
				("3", Face),
				("1", Constant),
				("time", Limit),
				("reroll", Reroll),
				("in", Predicate),
				("[", ListBracket),
				("]", ListBracket)
			]
		),
		(
			"{x} @ ( 1D20 ) + highest (lowest({x}, 2), 3) - [1 : 6]",
			&[
				("{", NameBrace),
				("x", BindingDeclaration),
				("}", NameBrace),
				("@", Bind),
				("(", Parenthesis),
				("1", Constant),
				("D", DiceOperator),
				("20", Face),
				(")", Parenthesis),
				("+", Additive),
				("highest", Call),
				("(", Parenthesis),
				("lowest", Call),
				("(", Parenthesis),
				("{", NameBrace),
				("x", Unresolved),
				("}", NameBrace),
				(",", Separator),
				("2", Constant),
				(")", Parenthesis),
				(",", Separator),
				("3", Constant),
				(")", Parenthesis),
				("-", Additive),
				("[", RangeBracket),
				("1", Constant),
				(":", RangeOperator),
				("6", Constant),
				("]", RangeBracket)
			]
		),
		(
			"{\u{A0}hit\tpoints }D({y}@(2)) drop highest {y} explode = {y}",
			&[
				("{", NameBrace),
				("hit\tpoints", Unresolved),
				("}", NameBrace),
				("D", DiceOperator),
				("(", Parenthesis),
				("{", NameBrace),
				("y", BindingDeclaration),
				("}", NameBrace),
				("@", Bind),
				("(", Parenthesis),
				("2", Constant),
				(")", Parenthesis),
				(")", Parenthesis),
				("drop", Selection),
				("highest", Direction),
				("{", NameBrace),
				("y", Unresolved),
				("}", NameBrace),
				("explode", Explode),
				("=", Predicate),
				("{", NameBrace),
				("y", Unresolved),
				("}", NameBrace)
			]
		)
	];
	for &(source, expected) in cases
	{
		assert_eq!(roles(source), expected, "{:?}", source);
	}
}

/// Answer the tokens that the engine reports as it parses a source, as their
/// text and role.
///
/// # Parameters
/// - `source`: The source.
///
/// # Returns
/// The tokens.
///
/// # Panics
/// If the source fails to parse.
fn roles(source: &str) -> Vec<(&str, Role)>
{
	tokens(source)
		.expect("the source parses")
		.iter()
		.map(|token| (&source[token.span.start..token.span.end], token.role))
		.collect()
}

/// Answer the tokens that the engine reports as it parses a source.
///
/// # Parameters
/// - `source`: The source.
///
/// # Returns
/// The tokens, or `None` if the source fails to parse.
fn tokens(source: &str) -> Option<Vec<Token>>
{
	let mut tokens = Vec::new();
	let parsed = Parser::parse_recovering(source, &mut Strict, &mut tokens);
	parsed.ok().map(|_| tokens)
}

////////////////////////////////////////////////////////////////////////////////
//                                Retraction.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a sink of tokens retracts exactly the tokens that begin at or
/// after an offset.
#[test]
fn test_retract()
{
	let token = |start, end, role| Token {
		span: SourceSpan { start, end },
		role,
		depth: None
	};
	let mut tokens = Vec::new();
	tokens.token(SourceSpan { start: 0, end: 1 }, Role::Negation);
	tokens.token(SourceSpan { start: 2, end: 3 }, Role::Constant);
	tokens.token(SourceSpan { start: 4, end: 5 }, Role::Additive);
	tokens.retract(3);
	assert_eq!(
		tokens,
		[token(0, 1, Role::Negation), token(2, 3, Role::Constant)]
	);
	tokens.retract(2);
	assert_eq!(tokens, [token(0, 1, Role::Negation)]);
	tokens.retract(5);
	assert_eq!(tokens, [token(0, 1, Role::Negation)]);
	tokens.retract(0);
	assert_eq!(tokens, []);
}

/// Ensure that the engine retracts the sign of a negation whose operand fails
/// recoverably, as the frame of the negation abandons it, however deeply the
/// negations nest. In a clean parse, no catching frame abandons an alternative
/// after its first token, so only a failing parse shows the retraction.
#[test]
fn test_retract_negation()
{
	for source in ["- x", "--x", "- - - x"]
	{
		let mut tokens = Vec::new();
		let parsed = Parser::parse_recovering(source, &mut Strict, &mut tokens);
		assert!(parsed.is_err(), "{:?}", source);
		assert_eq!(tokens, [], "{:?}", source);
	}
}
