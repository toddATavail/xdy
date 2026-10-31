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
//!
//! Herein too are the tests of name resolution and delimiter depth: a table of
//! sources that, between them, reach every outcome of resolution, a table of
//! depths, in clean and malformed sources, the closing of delimiters whose
//! closers are missing, and resolution past deep nesting, which `just stress`
//! runs. That resolution and depth agree with the tree of every random program
//! is a [property](super::property) test.

use pretty_assertions::assert_eq;

use crate::{
	Parser,
	classify::{Role, Token, classify, resolve},
	parser::{Strict, TokenSink as _},
	span::SourceSpan,
	support::on_small_stack,
	tests::{
		ast::{Nesting, nest_parsable},
		corpus::corpus,
		property::{check_classification, check_resolution}
	}
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
		("{x", &[("{", NameBrace), ("x", External)]),
		(
			"{x\u{a0}+ 2",
			&[
				("{", NameBrace),
				("x", External),
				("\u{a0}", Unrecognized),
				("+", Additive),
				("2", Constant)
			]
		),
		// A name that begins with `#`, which no repair reads, so that the
		// recovering parse stops at its `{`.
		(
			"{#x} + 1",
			&[
				("{#x}", Unrecognized),
				("+", Unrecognized),
				("1", Unrecognized)
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
				("x", ParameterReference),
				("}", NameBrace)
			]
		),
		// Bare words, read as names: as a variable, as the first formal
		// parameter, and as a later one.
		(
			"{y} + x",
			&[
				("{", NameBrace),
				("y", External),
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
				("x", External),
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

////////////////////////////////////////////////////////////////////////////////
//                                Resolution.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that classification resolves each name of each source as expected,
/// by the scoping rules of the compiler.
#[test]
fn test_resolved_names()
{
	use Role::*;
	let cases: &[(&str, &[(&str, Role)])] = &[
		// Parameters, and an external variable.
		(
			"{x}, {y}: {x} + {y} + {z}",
			&[
				("x", ParameterDeclaration),
				("y", ParameterDeclaration),
				("x", ParameterReference),
				("y", ParameterReference),
				("z", External)
			]
		),
		// A binding, referred to before, within, and after it.
		(
			"{x} + {x}@(1D6 + {x}) + {x}",
			&[
				("x", Unresolved),
				("x", BindingDeclaration),
				("x", Unresolved),
				("x", BindingReference)
			]
		),
		// Bindings as the count and the faces of dice, and as the count of a
		// drop clause.
		(
			"{n}@(2)D{f}@(6) drop lowest {n}",
			&[
				("n", BindingDeclaration),
				("f", BindingDeclaration),
				("n", BindingReference)
			]
		),
		// Names that differ only in their whitespace.
		(
			"{hit points}@(1) + { hit\tpoints }",
			&[
				("hit points", BindingDeclaration),
				("hit\tpoints", BindingReference)
			]
		),
		// A parameter declared again.
		(
			"{x}, {x}: {x}",
			&[
				("x", ParameterDeclaration),
				("x", Unresolved),
				("x", ParameterReference)
			]
		),
		// A binding of the name of a parameter.
		(
			"{x}: {x}@(1) + {x}",
			&[
				("x", ParameterDeclaration),
				("x", Unresolved),
				("x", ParameterReference)
			]
		),
		// A binding of the name of an earlier binding, after it and within
		// it. The later binding never binds the name.
		(
			"{x}@(1) + {x}@(2) + {x}",
			&[
				("x", BindingDeclaration),
				("x", Unresolved),
				("x", BindingReference)
			]
		),
		(
			"{x}@({x}@(1) + {x}) + {x}",
			&[
				("x", BindingDeclaration),
				("x", Unresolved),
				("x", Unresolved),
				("x", BindingReference)
			]
		),
		// Malformed sources: a binding that lacks its `)`, within a group that
		// lacks its own, and on its own.
		(
			"({x}@(1D6) + {x}",
			&[("x", BindingDeclaration), ("x", BindingReference)]
		),
		(
			"{x}@(1D6 + {x}",
			&[("x", BindingDeclaration), ("x", Unresolved)]
		),
		// A broken name that becomes the first formal parameter.
		(
			"{x, {y}: {x} + {y}",
			&[
				("x", ParameterDeclaration),
				("y", ParameterDeclaration),
				("x", ParameterReference),
				("y", ParameterReference)
			]
		)
	];
	for &(source, expected) in cases
	{
		let actual = classify(source)
			.iter()
			.filter(|token| {
				matches!(
					token.role,
					ParameterDeclaration
						| ParameterReference
						| BindingDeclaration
						| BindingReference
						| External
						| Unresolved
				)
			})
			.map(|token| {
				(&source[token.span.start..token.span.end], token.role)
			})
			.collect::<Vec<_>>();
		assert_eq!(actual, expected, "{:?}", source);
	}
}

/// Ensure that classification gives each delimiter of each source the expected
/// depth, and no other token any.
#[test]
fn test_depths()
{
	let cases: &[(&str, &[(&str, usize)])] = &[
		// Every kind of delimiter, nested and in sequence.
		(
			"((1) + [1:(2)]) * highest(1D[1, 2], 3D6 reroll in [1, 2])",
			&[
				("(", 0),
				("(", 1),
				(")", 1),
				("[", 1),
				("(", 2),
				(")", 2),
				("]", 1),
				(")", 0),
				("(", 0),
				("[", 1),
				("]", 1),
				("[", 1),
				("]", 1),
				(")", 0)
			]
		),
		// A binding.
		("{x}@((1)) + {x}", &[("(", 0), ("(", 1), (")", 1), (")", 0)]),
		// Openers whose closers are missing.
		("((1D6", &[("(", 0), ("(", 1)]),
		("(3D[1, 2", &[("(", 0), ("[", 1)]),
		// A closer that nothing opened has no depth.
		(")(1)", &[("(", 0), (")", 0)]),
		("[1:(2]", &[("[", 0), ("(", 1)])
	];
	for &(source, expected) in cases
	{
		let tokens = classify(source);
		let actual = tokens
			.iter()
			.filter_map(|token| {
				token.depth.map(|depth| {
					(&source[token.span.start..token.span.end], depth)
				})
			})
			.collect::<Vec<_>>();
		assert_eq!(actual, expected, "{:?}", source);
	}
}

/// Ensure that resolution closes every delimiter opened since the delimiter
/// that a closer pairs with, as though a repair had supplied their closers,
/// and gives a closer that pairs with nothing depth `0`. The engine reports
/// neither, so the tokens are made by hand.
#[test]
fn test_resolve_unbalanced()
{
	let source = "([(]) )";
	let roles = [
		Role::Parenthesis,
		Role::FacesBracket,
		Role::Parenthesis,
		Role::FacesBracket,
		Role::Parenthesis,
		Role::RangeBracket
	];
	let offsets = [0, 1, 2, 3, 4, 6];
	let mut tokens = offsets
		.iter()
		.zip(roles)
		.map(|(&start, role)| Token {
			span: SourceSpan {
				start,
				end: start + 1
			},
			role,
			depth: None
		})
		.collect::<Vec<_>>();
	resolve(source, &mut tokens);
	let depths = tokens.iter().map(|token| token.depth).collect::<Vec<_>>();
	assert_eq!(
		depths,
		[Some(0), Some(1), Some(2), Some(1), Some(0), Some(0)]
	);
}

/// Ensure that classification [resolves](check_resolution) past a deep chain of
/// groups on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_resolve_deep_group() { resolve_deep(Nesting::Group) }

/// Ensure that classification [resolves](check_resolution) past a deep chain of
/// bindings, each of the same name, on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_resolve_deep_binding() { resolve_deep(Nesting::Binding) }

/// Ensure that classification [resolves](check_resolution) past a deep chain of
/// range starts on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_resolve_deep_range_start() { resolve_deep(Nesting::RangeStart) }

/// Ensure that classification [resolves](check_resolution) past a deep chain of
/// every nesting construct on a small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_resolve_deep_mixed() { resolve_deep(Nesting::Mixed) }

/// Check that classification [resolves names and counts
/// depths](check_resolution) past a deep chain of a nesting construct, between
/// a binding and references to it and to a parameter, on a small stack.
///
/// # Parameters
/// - `nesting`: The way to nest.
///
/// # Panics
/// If a property fails.
fn resolve_deep(nesting: Nesting)
{
	on_small_stack(|| {
		let nest = nest_parsable(nesting, nesting.depth(), 1).to_string();
		let source = format!("{{p}}: {{b}}@({{p}}) + {} + {{b}} + {{p}}", nest);
		if let Err(e) = check_resolution(&source)
		{
			panic!("{:?}: {}", nesting, e);
		}
	});
}
