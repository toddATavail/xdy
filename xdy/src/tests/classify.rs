//! # Source classification
//!
//! Herein are the tests of the tokens that the parsing engine reports, with
//! their roles, as it parses: a table of sources that, between them, report
//! every role that the engine reports, and the retraction of the tokens of an
//! alternative that a catching frame abandons. That the tokens of every clean
//! parse tile the source is a [property](super::property) test.
//!
//! Herein too are the tests of [classification](classify) of malformed
//! sources, through the recovering parse: a table of sources that, between
//! them, take every path of recovery that the doctor's repairs reach, and the
//! totality of classification over the whole [corpus](corpus), which `just
//! stress` runs. That classification is total over random sources, and over
//! every prefix of a random program, is a [property](super::property) test.

use pretty_assertions::assert_eq;

use crate::{
	Parser,
	classify::{Role, Token, classify},
	parser::{Strict, TokenSink as _},
	span::SourceSpan,
	tests::{corpus::corpus, property::check_classification}
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

////////////////////////////////////////////////////////////////////////////////
//                                 Recovery.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that classification of each malformed source answers the expected
/// tokens: the tokens of the source that the recovering parse read, but none
/// that a repair supplied, and [unrecognized](Role::Unrecognized) text wherever
/// a repair discarded the source, read it as something else, or declined to
/// repair it.
#[test]
fn test_recovered_roles()
{
	use Role::*;
	let cases: &[(&str, &[(&str, Role)])] = &[
		// An unopened delimiter, skipped where an operand is missing, and at
		// the start of the body.
		(
			"3D6 + )",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("+", Additive),
				(")", Unrecognized)
			]
		),
		(
			")))3D6",
			&[
				(")))", Unrecognized),
				("3", Constant),
				("D", DiceOperator),
				("6", Face)
			]
		),
		// A missing `)`, and a missing operand.
		(
			"(3D6 + + 1",
			&[
				("(", Parenthesis),
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("+", Additive),
				("+", Additive),
				("1", Constant)
			]
		),
		// A missing `}`, at the end, and where the name breaks early, before
		// stray whitespace.
		("{x", &[("{", NameBrace), ("x", Unresolved)]),
		(
			"{x\u{a0}+ 2",
			&[
				("{", NameBrace),
				("x", Unresolved),
				("\u{a0}", Unrecognized),
				("+", Additive),
				("2", Constant)
			]
		),
		// A missing name.
		(
			"{ }D6",
			&[
				("{", NameBrace),
				("}", NameBrace),
				("D", DiceOperator),
				("6", Face)
			]
		),
		// A broken name that becomes the first formal parameter, so that the
		// function is parsed again.
		(
			"{x, {y}: {x}",
			&[
				("{", NameBrace),
				("x", ParameterDeclaration),
				(",", Separator),
				("{", NameBrace),
				("y", ParameterDeclaration),
				("}", NameBrace),
				(":", Separator),
				("{", NameBrace),
				("x", Unresolved),
				("}", NameBrace)
			]
		),
		// Bare words, read as names: as a variable, as the first formal
		// parameter, and as a later one.
		(
			"{y} + x",
			&[
				("{", NameBrace),
				("y", Unresolved),
				("}", NameBrace),
				("+", Additive),
				("x", Unrecognized)
			]
		),
		(
			"x, y: 1",
			&[
				("x", Unrecognized),
				(",", Separator),
				("y", Unrecognized),
				(":", Separator),
				("1", Constant)
			]
		),
		(
			"{x}, y: 1",
			&[
				("{", NameBrace),
				("x", ParameterDeclaration),
				("}", NameBrace),
				(",", Separator),
				("y", Unrecognized),
				(":", Separator),
				("1", Constant)
			]
		),
		// Retracted commas of formal parameters.
		(
			"{x}, , {y}: 1",
			&[
				("{", NameBrace),
				("x", ParameterDeclaration),
				("}", NameBrace),
				(",", Unrecognized),
				(",", Separator),
				("{", NameBrace),
				("y", ParameterDeclaration),
				("}", NameBrace),
				(":", Separator),
				("1", Constant)
			]
		),
		// Formal parameters without their `:`.
		(
			"{x}, {y}",
			&[
				("{", NameBrace),
				("x", ParameterDeclaration),
				("}", NameBrace),
				(",", Separator),
				("{", NameBrace),
				("y", ParameterDeclaration),
				("}", NameBrace)
			]
		),
		// A misnamed call, a lone argument, unwrapped or kept, a trailing
		// comma, and a missing comma.
		(
			"max(1, 2)",
			&[
				("max", Unrecognized),
				("(", Parenthesis),
				("1", Constant),
				(",", Separator),
				("2", Constant),
				(")", Parenthesis)
			]
		),
		(
			"highest(1 + 2) * 3",
			&[
				("highest", Call),
				("(", Parenthesis),
				("1", Constant),
				("+", Additive),
				("2", Constant),
				(")", Parenthesis),
				("*", Multiplicative),
				("3", Constant)
			]
		),
		(
			"highest(4D6)",
			&[
				("highest", Call),
				("(", Parenthesis),
				("4", Constant),
				("D", DiceOperator),
				("6", Face),
				(")", Parenthesis)
			]
		),
		(
			"highest(1, 2,)",
			&[
				("highest", Call),
				("(", Parenthesis),
				("1", Constant),
				(",", Separator),
				("2", Constant),
				(",", Unrecognized),
				(")", Parenthesis)
			]
		),
		(
			"highest(1 2)",
			&[
				("highest", Call),
				("(", Parenthesis),
				("1", Constant),
				("2", Constant),
				(")", Parenthesis)
			]
		),
		// Ungrouped calls, as faces, as a dice count, and as the count of a
		// drop clause.
		(
			"3Dhighest(1, 2)",
			&[
				("3", Constant),
				("D", DiceOperator),
				("highest", Call),
				("(", Parenthesis),
				("1", Constant),
				(",", Separator),
				("2", Constant),
				(")", Parenthesis)
			]
		),
		(
			"highest(1, 2)D6",
			&[
				("highest", Call),
				("(", Parenthesis),
				("1", Constant),
				(",", Separator),
				("2", Constant),
				(")", Parenthesis),
				("D", DiceOperator),
				("6", Face)
			]
		),
		(
			"4D6 drop lowest lowest({x}, 2)",
			&[
				("4", Constant),
				("D", DiceOperator),
				("6", Face),
				("drop", Selection),
				("lowest", Direction),
				("lowest", Call),
				("(", Parenthesis),
				("{", NameBrace),
				("x", Unresolved),
				("}", NameBrace),
				(",", Separator),
				("2", Constant),
				(")", Parenthesis)
			]
		),
		// Misplaced clauses: a drop, whose `d` is read again, and a reroll.
		(
			"3 drop lowest",
			&[("3", Constant), ("drop", Selection), ("lowest", Direction)]
		),
		(
			"3 reroll = 1",
			&[
				("3", Constant),
				("reroll", Reroll),
				("=", Predicate),
				("1", Face)
			]
		),
		// Incomplete clauses: a missing direction, predicate, range
		// operator, range end, and `times`.
		(
			"3D6 drop",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("drop", Selection)
			]
		),
		(
			"3D6 reroll",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("reroll", Reroll)
			]
		),
		(
			"3D6 reroll in 1 3",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("reroll", Reroll),
				("in", Predicate),
				("1", Face),
				("3", Face)
			]
		),
		(
			"3D6 explode in 5..",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("explode", Explode),
				("in", Predicate),
				("5", Face),
				("..", RangeOperator)
			]
		),
		(
			"3D6 reroll = 1 unlimited",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("reroll", Reroll),
				("=", Predicate),
				("1", Face),
				("unlimited", Limit)
			]
		),
		// Unclosed lists, of custom faces and of a predicate, and a missing
		// first face.
		(
			"3D[",
			&[("3", Constant), ("D", DiceOperator), ("[", FacesBracket)]
		),
		(
			"3D6 reroll in [1, 2",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("reroll", Reroll),
				("in", Predicate),
				("[", ListBracket),
				("1", Face),
				(",", Separator),
				("2", Face)
			]
		),
		// Unfixable failures, after which nothing is recognized, but what was
		// read before them is.
		(
			"3D[1,",
			&[
				("3", Constant),
				("D", DiceOperator),
				("[", FacesBracket),
				("1", Face),
				(",", Unrecognized)
			]
		),
		(
			"3D6 $ 2",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("$", Unrecognized),
				("2", Unrecognized)
			]
		),
		// Trailing input, and stray whitespace.
		(
			"3D6 + 1 2",
			&[
				("3", Constant),
				("D", DiceOperator),
				("6", Face),
				("+", Additive),
				("1", Constant),
				("2", Unrecognized)
			]
		),
		(
			"1\u{a0}+ 2",
			&[
				("1", Constant),
				("\u{a0}", Unrecognized),
				("+", Additive),
				("2", Constant)
			]
		),
		// A negation whose operand is missing, or is a bare word.
		(
			"1 + - ",
			&[("1", Constant), ("+", Additive), ("-", Negation)]
		),
		("- x", &[("-", Negation), ("x", Unrecognized)])
	];
	for &(source, expected) in cases
	{
		let tokens = classify(source);
		let actual = tokens
			.iter()
			.map(|token| {
				(&source[token.span.start..token.span.end], token.role)
			})
			.collect::<Vec<_>>();
		assert_eq!(actual, expected, "{:?}", source);
	}
}

/// Ensure that classification of every input of the [corpus](corpus) is
/// [total](check_classification). The corpus is large, so the test is ignored
/// by default; `just stress` runs it.
#[test]
#[ignore = "stress: run with just stress"]
fn test_classified_corpus()
{
	for input in corpus()
	{
		if let Err(e) = check_classification(input)
		{
			panic!("{}", e);
		}
	}
}
