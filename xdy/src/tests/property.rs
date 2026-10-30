//! # Property tests
//!
//! Herein are the property tests of the front end. They generate random
//! [programs](program) from the grammar, and random [mutations](Mutation) of
//! them, which are mostly malformed, and hold every program and mutation to
//! these properties:
//!
//! - [`compile`](crate::compile) and [`diagnose`] return, without panicking,
//!   overflowing the stack, or hanging.
//! - [`compile`](crate::compile) fails to parse just when [`Parser::parse`]
//!   does, with the same error.
//! - [`diagnose`] reports a diagnostic for every source that fails to parse,
//!   leaves every source that parses as it is, and corrects sources only to
//!   sources that parse.
//!
//! Every program must also parse, render as a source that parses and renders
//! the same, and write as an S-expression that reads back to it. The tokens
//! that the parser reports as it parses a program, or a mutation that still
//! parses, must tile the source. The classification of every program must
//! resolve its names by the scoping rules of the validator, and give its
//! delimiters the depths of the nesting of its tree. Every program
//! that compiles must also [evaluate](Evaluator::evaluate_metered), over
//! arguments and externals that include the extremes of [`i32`], within a
//! [dice budget](EvaluationError::DiceBudgetExhausted), and so return promptly
//! however many dice it asks for; and its distribution must
//! [build](crate::DistributionPlan::build) within a budget of steps and cells,
//! or be refused, and contain the result of every evaluation within budget.
//!
//! Optimization must never change a program's distribution: the exact
//! [distributions](crate::DistributionPlan::build) of every program that
//! compiles, optimized and not, over the same arguments and externals, must
//! agree outcome for outcome, as probabilities, whenever both fit a budget. A
//! program that rolls no dice has a single outcome, so for it the distributions
//! agree just when the results do, even at the extremes of [`i32`].
//!
//! Each property runs [apart](on_small_stack), whose time budget turns a hang
//! into a failure, and runs each case on its own thread with a [small
//! stack](SMALL_STACK_SIZE). `proptest` itself, whose value trees for deep
//! programs are deep, runs on a [larger stack](RUNNER_STACK_SIZE), so that the
//! small stack measures only the code under test. A failure reports the input,
//! shrunk by `proptest`, and records it under `proptest-regressions`, so that
//! later runs try it first.

use std::{
	collections::{HashMap, HashSet},
	panic::resume_unwind,
	thread
};

use proptest::{
	option,
	prelude::*,
	sample::{Index, select, subsequence},
	test_runner::{Config, FileFailurePersistence, TestCaseError, TestRunner}
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	Budget, BuildError, Distribution, EvaluationError, Evaluator, Explode,
	Function as IrFunction, Instruction, Limit, Optimizer as _, Parser, Passes,
	StandardOptimizer, Unobserved, Validator,
	ast::{
		DiceExpression, Event, Expression, Extremum, Function, Group, Node,
		Range, Walk
	},
	classify::{Role, Token, classify},
	compiler::{CompilationError, compile, compile_unoptimized},
	diagnostics::diagnose,
	parser::{Strict, is_token_space},
	s_expr::{SExpressible, SExpressibleOptions, read_s_expr},
	support::{SMALL_STACK_SIZE, on_small_stack},
	tests::corpus::{TOKENS, sources}
};

////////////////////////////////////////////////////////////////////////////////
//                                Properties.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The number of cases of each property.
const CASES: u32 = 2_000;

/// The stack size, in bytes, of the thread on which `proptest` generates,
/// shrinks, and drops the inputs of a property: 64 MiB, ample for value trees
/// as deep as [`MAX_DEPTH`] allows.
const RUNNER_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Ensure that every random [program](program) parses, round-trips through
/// its rendering and its S-expression, and satisfies the properties of every
/// source.
#[test]
fn test_programs()
{
	on_small_stack(|| check(program, |source| check_program(&source)));
}

/// Ensure that every random [mutation](Mutation) of a random
/// [program](program) satisfies the properties of every source.
#[test]
fn test_mutated_programs()
{
	on_small_stack(|| check(mutated_program, |source| check_source(&source)));
}

/// Ensure that the tokens that the parser reports, as it parses every random
/// [program](program), and every random [mutation](Mutation) of one that still
/// parses, [tile](check_tokens) the source.
#[test]
fn test_program_tokens()
{
	on_small_stack(|| {
		check(
			|| prop_oneof![program(), mutated_program()],
			|source| check_tokens(&source)
		)
	});
}

/// Ensure that the [classification](check_classification) of every random
/// [program](program), [mutation](Mutation) of one, string of
/// [tokens](token_string), and arbitrary string is total, and exactly the
/// tokens of the parse where the source parses.
#[test]
fn test_classified_sources()
{
	on_small_stack(|| {
		check(
			|| {
				prop_oneof![
					program(),
					mutated_program(),
					token_string(),
					any::<String>()
				]
			},
			|source| check_classification(&source)
		)
	});
}

/// Ensure that the [classification](check_classification) of every prefix of
/// every random [program](program) is total, as an editor's buffer is
/// classified while the program is typed.
#[test]
fn test_classified_prefixes()
{
	on_small_stack(|| {
		check(program, |source| {
			source
				.char_indices()
				.try_for_each(|(at, _)| check_classification(&source[..at]))
		})
	});
}

/// Ensure that the classification of every random [program](program), and of
/// every random [mutation](Mutation) of one that still parses,
/// [resolves its names and counts its depths](check_resolution) as the tree of
/// the parse does.
#[test]
fn test_resolved_programs()
{
	on_small_stack(|| {
		check(
			|| prop_oneof![program(), mutated_program()],
			|source| check_resolution(&source)
		)
	});
}

/// Ensure that every random [program](program) that compiles satisfies the
/// [properties of metered evaluation](check_evaluation), over random bindings
/// and budgets.
#[test]
fn test_evaluated_programs()
{
	on_small_stack(|| {
		check(
			|| {
				(
					program(),
					[binding(), binding()],
					[binding(), binding(), binding(), binding(), binding()],
					budget(),
					any::<u64>()
				)
			},
			|(source, args, externals, budget, seed)| {
				check_evaluation(&source, &args, &externals, budget, seed)
			}
		)
	});
}

/// The budget of each distribution built by [`check_optimization`]: small
/// enough that both builds are prompt, and large enough for most random
/// programs.
const DISTRIBUTION_BUDGET: Budget = Budget {
	steps: 200_000,
	cells: 100_000
};

/// Ensure that optimization never changes the distribution of any random
/// [program](program) that compiles, over random bindings.
#[test]
fn test_optimized_programs()
{
	on_small_stack(|| {
		check(
			|| {
				(
					program(),
					[binding(), binding()],
					[binding(), binding(), binding(), binding(), binding()]
				)
			},
			|(source, args, externals)| {
				check_optimization(&source, &args, &externals)
			}
		)
	});
}

/// Ensure that optimization never changes the distribution of a reroll, for
/// every combination of a die, ordinary or with one face, constant or not,
/// whose faces are positive, negative, or of mean zero; a predicate of every
/// form, constant or not, that matches some faces, every face, or none; a limit
/// of every form; and the clauses around it, of every kind, before it or after
/// it, over values of the parameter that make every operand that is not
/// constant match differently.
#[test]
fn test_optimized_rerolls()
{
	let dice = [
		"3D1",
		"{x}D1",
		"2D[5, 5]",
		"3D[2, 4]",
		"2D3",
		"{x}D2",
		"2D{x}",
		"[2:2]",
		"2D[-2, -1]",
		"2D[-1, 0, 1]",
		"2D[0, 0]"
	];
	let predicates = [
		"= 1",
		"≠ 1",
		"≤ 3",
		"≥ 1",
		"< 9",
		"in []",
		"in [2, 4]",
		"in 2..2",
		"in 1..=3",
		"= {x}",
		"in {x}..=4"
	];
	let limits = ["", " 0 times", " 2 times", " ∞ times", " {x} times"];
	let clauses = [
		"{dice} reroll {reroll}",
		"{dice} drop lowest reroll {reroll}",
		"{dice} reroll {reroll} keep highest",
		"{dice} reroll {reroll} reroll {reroll}",
		"{dice} reroll ≥ 2 ∞ times reroll {reroll}"
	];
	for dice in dice
	{
		for predicate in predicates
		{
			for limit in limits
			{
				let reroll = format!("{}{}", predicate, limit);
				for clauses in clauses
				{
					let source = format!(
						"{{x}}: {}",
						clauses
							.replace("{dice}", dice)
							.replace("{reroll}", &reroll)
					);
					for x in -1..=4
					{
						check_optimization(&source, &[x], &[])
							.map_err(|e| {
								format!("{}, x = {}: {}", source, x, e)
							})
							.unwrap();
					}
				}
			}
		}
	}
}

/// Ensure that optimization never widens the bounds of an explode, for every
/// combination of a die, ordinary or with one face, constant or not, whose
/// faces are positive, negative, or of mean zero, so that a chain may saturate
/// toward either bound or to zero; a predicate of every form, constant or not,
/// that matches some faces, every face, or none; a limit of every form; and the
/// clauses around it, of every kind, before it or after it, among them rerolls
/// and other explodes, and after an explode that saturates every chain of a die
/// whose faces are positive, over values of the parameter that make every
/// operand that is not constant match differently, and that their distributions
/// agree, unless the forward pass refuses an explode without limit.
#[test]
fn test_optimized_explodes()
{
	let dice = [
		"3D1",
		"{x}D1",
		"2D[5, 5]",
		"3D[2, 4]",
		"2D3",
		"{x}D2",
		"2D{x}",
		"[2:2]",
		"2D[-2, -1]",
		"2D[-1, 0, 1]",
		"2D[0, 0]"
	];
	let predicates = [
		"= 1",
		"≠ 1",
		"≤ 3",
		"≥ 1",
		"< 9",
		"in []",
		"in [2, 4]",
		"in 2..2",
		"in 1..=3",
		"= {x}",
		"in {x}..=4"
	];
	let limits = ["", " 0 times", " 2 times", " ∞ times", " {x} times"];
	let clauses = [
		"{dice} explode {explode}",
		"{dice} drop lowest explode {explode}",
		"{dice} explode {explode} keep highest",
		"{dice} explode {explode} explode {explode}",
		"{dice} reroll = 1 explode {explode}",
		"{dice} explode {explode} reroll ≥ 2 ∞ times",
		"{dice} explode ≥ 1 ∞ times reroll {explode}",
		"{dice} explode ≥ 1 ∞ times drop lowest explode {explode}"
	];
	for dice in dice
	{
		for predicate in predicates
		{
			for limit in limits
			{
				let explode = format!("{}{}", predicate, limit);
				for clauses in clauses
				{
					let source = format!(
						"{{x}}: {}",
						clauses
							.replace("{dice}", dice)
							.replace("{explode}", &explode)
					);
					for x in -1..=4
					{
						check_optimization(&source, &[x], &[])
							.map_err(|e| {
								format!("{}, x = {}: {}", source, x, e)
							})
							.unwrap();
					}
				}
			}
		}
	}
}

/// Ensure that optimization never changes the distribution of any source of
/// the test files that compiles, over values of its parameters and externals
/// that are negative, zero, one, and greater.
#[test]
fn test_optimized_corpus()
{
	for source in sources()
	{
		let Ok(function) = compile_unoptimized(&source)
		else
		{
			continue
		};
		for x in [-1, 0, 1, 3]
		{
			let args = vec![x; function.arity()];
			let externals = function
				.externals
				.iter()
				.map(|name| (name.as_str(), x))
				.collect::<Vec<_>>();
			check_optimization_with(&source, &args, &externals)
				.map_err(|e| format!("{}, x = {}: {}", source, x, e))
				.unwrap();
		}
	}
}

/// Run a property on [`CASES`] random inputs, recording failures under
/// `proptest-regressions`, in the file that parallels this one. See
/// [`check_within`].
///
/// # Parameters
/// - `strategy`: The constructor of the strategy that generates the inputs.
/// - `property`: The property.
///
/// # Panics
/// If the property fails on any input.
fn check<S>(
	strategy: impl FnOnce() -> S + Send,
	property: impl Fn(S::Value) -> Result<(), TestCaseError> + Sync
) where
	S: Strategy,
	S::Value: Send
{
	check_within(CASES, file!(), strategy, property)
}

/// Run a property on random inputs, recording failures under
/// `proptest-regressions`, in the file that parallels the specified source
/// file. `proptest` runs on a [large stack](RUNNER_STACK_SIZE), and each case
/// on a [small stack](SMALL_STACK_SIZE), so that a case that overflows the
/// small stack aborts the process, as [`on_small_stack`] expects.
///
/// # Parameters
/// - `cases`: The number of cases.
/// - `source_file`: The source file of the property, as [`file!`] gives it,
///   whose regressions file records its failures.
/// - `strategy`: The constructor of the strategy that generates the inputs,
///   which runs on the large stack, since a recursive strategy cannot move
///   between threads.
/// - `property`: The property.
///
/// # Panics
/// If the property fails on any input.
pub(super) fn check_within<S>(
	cases: u32,
	source_file: &'static str,
	strategy: impl FnOnce() -> S + Send,
	property: impl Fn(S::Value) -> Result<(), TestCaseError> + Sync
) where
	S: Strategy,
	S::Value: Send
{
	let config = Config {
		cases,
		source_file: Some(source_file),
		failure_persistence: Some(Box::new(
			FileFailurePersistence::SourceParallel("proptest-regressions")
		)),
		..Config::default()
	};
	on_stack(RUNNER_STACK_SIZE, || {
		let result = TestRunner::new(config).run(&strategy(), |value| {
			on_stack(SMALL_STACK_SIZE, || property(value))
		});
		if let Err(e) = result
		{
			panic!("{}", e);
		}
	})
}

/// Run a closure on a new thread with the specified stack size, and answer its
/// result. A panic in the closure resumes on the calling thread, where
/// `proptest` can catch it and shrink its input.
///
/// # Parameters
/// - `size`: The stack size, in bytes.
/// - `f`: The closure to run.
///
/// # Returns
/// The result of the closure.
///
/// # Panics
/// If the closure panics, or the thread cannot be spawned.
fn on_stack<R: Send>(size: usize, f: impl FnOnce() -> R + Send) -> R
{
	thread::scope(|scope| {
		thread::Builder::new()
			.stack_size(size)
			.spawn_scoped(scope, f)
			.unwrap()
			.join()
			.unwrap_or_else(|payload| resume_unwind(payload))
	})
}

/// Check the properties of a program: it parses; it renders as a source that
/// parses and renders the same; it writes as an S-expression that reads back
/// to it; and it satisfies the [properties of every source](check_source).
///
/// # Parameters
/// - `source`: The program.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
fn check_program(source: &str) -> Result<(), TestCaseError>
{
	let function = Parser::parse(source)
		.map_err(|e| TestCaseError::fail(format!("program fails:\n{}", e)))?;
	let rendered = function.to_string();
	prop_assert_eq!(reparse(&rendered)?.to_string(), rendered.as_str());
	let options = SExpressibleOptions::default()
		.with_spans(true)
		.with_groups(true);
	let s_expr = function.to_s_expr(options);
	let read = read_s_expr(&s_expr).map_err(|e| {
		TestCaseError::fail(format!("S-expression {:?} fails: {}", s_expr, e))
	})?;
	prop_assert!(
		read == function,
		"S-expression {:?} reads back differently",
		s_expr
	);
	check_source(source)
}

/// Parse a rendering.
///
/// # Parameters
/// - `rendered`: The rendering.
///
/// # Returns
/// The function.
///
/// # Errors
/// [`TestCaseError`] if the rendering fails to parse.
fn reparse(rendered: &str) -> Result<Function<'_>, TestCaseError>
{
	Parser::parse(rendered).map_err(|e| {
		TestCaseError::fail(format!("rendering {:?} fails:\n{}", rendered, e))
	})
}

/// Check that the tokens that the parser reports, as it parses a source, tile
/// it, if it parses: they are non-empty, in source order, and disjoint, and
/// only [whitespace between tokens](is_token_space) lies between them, or any
/// whitespace between the braces of a name and its identifier.
///
/// # Parameters
/// - `source`: The source.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
fn check_tokens(source: &str) -> Result<(), TestCaseError>
{
	if Parser::parse(source).is_err()
	{
		return Ok(());
	}
	let mut tokens = Vec::<Token>::new();
	prop_assert!(
		Parser::parse_recovering(source, &mut Strict, &mut tokens).is_ok(),
		"recovery mode fails a source that parses"
	);
	let mut end = 0;
	let mut braced = false;
	for token in &tokens
	{
		let span = token.span;
		prop_assert!(
			end <= span.start && span.start < span.end,
			"token {:?} is empty or out of order in {:?}",
			token,
			source
		);
		let gap = source.get(end..span.start);
		let text = source.get(span.start..span.end);
		let (Some(gap), Some(text)) = (gap, text)
		else
		{
			return Err(TestCaseError::fail(format!(
				"token {:?} splits a character of {:?}",
				token, source
			)));
		};
		prop_assert!(
			gap.chars().all(|c| {
				if braced
				{
					c.is_whitespace()
				}
				else
				{
					is_token_space(c)
				}
			}),
			"{:?} before token {:?} belongs to no token in {:?}",
			gap,
			token,
			source
		);
		if token.role == Role::NameBrace
		{
			braced = text == "{";
		}
		end = span.end;
	}
	prop_assert!(
		source[end..].chars().all(is_token_space),
		"{:?} after the last token belongs to no token",
		&source[end..]
	);
	Ok(())
}

/// Check that the [classification](classify) of a source is total: its tokens
/// are non-empty, in source order, and disjoint, each fits its
/// [role](fits_role), and only [whitespace between tokens](is_token_space)
/// lies between them, or any whitespace between the braces of a name and its
/// identifier. Just the parentheses and brackets have depths, none greater than
/// the number of those that open before them. If the source parses, they must
/// be exactly the tokens that the parse reports, but for the roles of names,
/// which only classification resolves, and depths, which only it counts.
///
/// # Parameters
/// - `source`: The source.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
pub(super) fn check_classification(source: &str) -> Result<(), TestCaseError>
{
	let tokens = classify(source);
	let mut parsed = Vec::<Token>::new();
	if Parser::parse_recovering(source, &mut Strict, &mut parsed).is_ok()
	{
		let unresolved = |tokens: &[Token]| {
			tokens
				.iter()
				.map(|token| {
					let role = if is_name(token.role)
					{
						Role::Unresolved
					}
					else
					{
						token.role
					};
					(token.span, role)
				})
				.collect::<Vec<_>>()
		};
		prop_assert_eq!(
			unresolved(&tokens),
			unresolved(&parsed),
			"classification is not the parse of {:?}",
			source
		);
	}
	let mut end = 0;
	let mut opened = false;
	let mut openers = 0;
	for token in &tokens
	{
		let span = token.span;
		prop_assert!(
			end <= span.start && span.start < span.end,
			"token {:?} is empty or out of order in {:?}",
			token,
			source
		);
		let gap = source.get(end..span.start);
		let text = source.get(span.start..span.end);
		let (Some(gap), Some(text)) = (gap, text)
		else
		{
			return Err(TestCaseError::fail(format!(
				"token {:?} splits a character of {:?}",
				token, source
			)));
		};
		let padding = opened || (token.role == Role::NameBrace && text == "}");
		prop_assert!(
			gap.chars().all(|c| {
				if padding
				{
					c.is_whitespace()
				}
				else
				{
					is_token_space(c)
				}
			}),
			"{:?} before token {:?} belongs to no token in {:?}",
			gap,
			token,
			source
		);
		prop_assert!(
			fits_role(text, token.role),
			"{:?} is no {:?} in {:?}",
			text,
			token.role,
			source
		);
		let delimiter = matches!(
			token.role,
			Role::Parenthesis
				| Role::RangeBracket
				| Role::FacesBracket
				| Role::ListBracket
		);
		prop_assert_eq!(
			token.depth.is_some(),
			delimiter,
			"depth of {:?} in {:?}",
			token,
			source
		);
		prop_assert!(
			token.depth.is_none_or(|depth| depth <= openers),
			"{:?} is deeper than the delimiters before it in {:?}",
			token,
			source
		);
		if delimiter && matches!(text, "(" | "[")
		{
			openers += 1;
		}
		opened = token.role == Role::NameBrace && text == "{";
		end = span.end;
	}
	prop_assert!(
		source[end..].chars().all(is_token_space),
		"{:?} after the last token belongs to no token in {:?}",
		&source[end..],
		source
	);
	Ok(())
}

/// Answer whether a role is that of the identifier of a name.
///
/// # Parameters
/// - `role`: The role.
///
/// # Returns
/// `true` if the role is that of an identifier, `false` otherwise.
fn is_name(role: Role) -> bool
{
	matches!(
		role,
		Role::ParameterDeclaration
			| Role::ParameterReference
			| Role::BindingDeclaration
			| Role::BindingReference
			| Role::External
			| Role::Unresolved
	)
}

/// Check that the [classification](classify) of a source that parses resolves
/// every name, and counts the depth of every delimiter, as an independent walk
/// of the tree of the parse does. The walk resolves names by the rules of the
/// [validator](Validator): the first declaration of a parameter, or of a
/// binding whose name no parameter has, is a declaration, and every other is
/// unresolved; a reference to a parameter is a parameter reference, one to a
/// binding that it follows the end of is a binding reference, one to another
/// binding is unresolved, and every other reference is external. A name is
/// unresolved just when the validator rejects the function. The walk counts
/// as the depth of the closing delimiter of each group, binding, call, range,
/// and custom dice the number of groups, bindings, calls, and ranges that
/// enclose it, and each opening delimiter must have the depth of the closing
/// delimiter that it pairs with. A source that fails to parse passes.
///
/// # Parameters
/// - `source`: The source.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
pub(super) fn check_resolution(source: &str) -> Result<(), TestCaseError>
{
	let Ok(function) = Parser::parse(source)
	else
	{
		return Ok(());
	};
	let tokens = classify(source);
	// The roles of the identifiers, by their offsets, and the depths of the
	// closing delimiters, by theirs.
	let mut names = HashMap::<usize, Role>::new();
	let mut depths = HashMap::<usize, usize>::new();
	let mut parameters = HashSet::<&str>::new();
	for parameter in function.parameters.iter().flatten()
	{
		let role = if parameters.insert(&parameter.name)
		{
			Role::ParameterDeclaration
		}
		else
		{
			Role::Unresolved
		};
		names.insert(parameter.span.start, role);
	}
	let mut bindings = HashSet::<&str>::new();
	let mut declarations = HashSet::<usize>::new();
	for event in Walk::new(Node::Expression(&function.body))
	{
		if let Event::Enter(Node::Expression(Expression::Binding(binding))) =
			event
			&& !parameters.contains(&*binding.name)
			&& bindings.insert(&binding.name)
		{
			declarations.insert(binding.name_span.start);
		}
	}
	let mut bound = HashSet::<&str>::new();
	let mut depth = 0;
	for event in Walk::new(Node::Expression(&function.body))
	{
		match event
		{
			Event::Enter(Node::Expression(Expression::Variable(variable))) =>
			{
				let name = &*variable.name;
				let role = if parameters.contains(name)
				{
					Role::ParameterReference
				}
				else if bound.contains(name)
				{
					Role::BindingReference
				}
				else if bindings.contains(name)
				{
					Role::Unresolved
				}
				else
				{
					Role::External
				};
				let braced = &source[variable.span.start + 1..];
				let padding = braced.len() - braced.trim_start().len();
				names.insert(variable.span.start + 1 + padding, role);
			},
			Event::Enter(Node::Expression(Expression::Binding(binding))) =>
			{
				let start = binding.name_span.start;
				let role = if declarations.contains(&start)
				{
					Role::BindingDeclaration
				}
				else
				{
					Role::Unresolved
				};
				names.insert(start, role);
				depths.insert(binding.span.end - 1, depth);
				depth += 1;
			},
			Event::Leave(Node::Expression(Expression::Binding(binding))) =>
			{
				depth -= 1;
				if declarations.contains(&binding.name_span.start)
				{
					bound.insert(&binding.name);
				}
			},
			Event::Enter(Node::Expression(
				Expression::Group(Group { span, .. })
				| Expression::Extremum(Extremum { span, .. })
				| Expression::Range(Range { span, .. })
			)) =>
			{
				depths.insert(span.end - 1, depth);
				depth += 1;
			},
			Event::Leave(Node::Expression(
				Expression::Group(_)
				| Expression::Extremum(_)
				| Expression::Range(_)
			)) => depth -= 1,
			Event::Enter(Node::Dice(DiceExpression::Custom(custom))) =>
			{
				depths.insert(custom.span.end - 1, depth);
			},
			_ =>
			{}
		}
	}
	let mut resolved = HashMap::<usize, Role>::new();
	let mut counted = HashMap::<usize, usize>::new();
	let mut open = Vec::new();
	for token in &tokens
	{
		if is_name(token.role)
		{
			resolved.insert(token.span.start, token.role);
		}
		else if let Some(depth) = token.depth
		{
			// The source parses, so its delimiters nest.
			if matches!(source.as_bytes()[token.span.start], b'(' | b'[')
			{
				open.push(token);
				continue;
			}
			let opener = open.pop();
			prop_assert_eq!(
				opener.map(|opener| (opener.role, opener.depth)),
				Some((token.role, token.depth)),
				"{:?} pairs with no opener of its depth in {:?}",
				token,
				source
			);
			if depths.contains_key(&token.span.start)
			{
				counted.insert(token.span.start, depth);
			}
		}
	}
	prop_assert!(open.is_empty(), "unclosed delimiters in {:?}", source);
	prop_assert_eq!(&resolved, &names, "names of {:?}", source);
	prop_assert_eq!(&counted, &depths, "depths of {:?}", source);
	prop_assert_eq!(
		Validator::validate(&function).is_ok(),
		!names.values().any(|&role| role == Role::Unresolved),
		"the validator disagrees with resolution of {:?}",
		source
	);
	Ok(())
}

/// Answer whether the text of a token could be a token of its role, by its
/// spelling alone. A face may be a lone sign, whose digits a repair supplied.
/// An identifier is any text without braces that neither
/// begins nor ends with whitespace, and unrecognized text any text without
/// whitespace between tokens.
///
/// # Parameters
/// - `text`: The text of the token.
/// - `role`: The role of the token.
///
/// # Returns
/// `true` if the text fits the role, `false` otherwise.
fn fits_role(text: &str, role: Role) -> bool
{
	let digits = |text: &str| {
		!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
	};
	match role
	{
		Role::Constant => digits(text),
		// The sign of a face whose digits a repair supplied stands alone.
		Role::Face =>
		{
			text == "-" || digits(text.strip_prefix('-').unwrap_or(text))
		},
		Role::DiceOperator => matches!(text, "d" | "D"),
		Role::Selection => matches!(text, "drop" | "keep"),
		Role::Reroll => text == "reroll",
		Role::Explode => text == "explode",
		Role::Direction | Role::Call => matches!(text, "lowest" | "highest"),
		Role::Predicate => matches!(
			text,
			"=" | "!=" | "≠" | "<" | "<=" | "≤" | ">" | ">=" | "≥" | "in"
		),
		Role::RangeOperator => matches!(text, ".." | "..=" | ":"),
		Role::Limit => matches!(text, "time" | "times" | "unlimited" | "∞"),
		Role::Additive => matches!(text, "+" | "-"),
		Role::Multiplicative => matches!(text, "*" | "×" | "/" | "÷" | "%"),
		Role::Exponent => text == "^",
		Role::Negation => text == "-",
		Role::Bind => text == "@",
		Role::Separator => matches!(text, "," | ":"),
		Role::Parenthesis => matches!(text, "(" | ")"),
		Role::RangeBracket | Role::FacesBracket | Role::ListBracket =>
		{
			matches!(text, "[" | "]")
		},
		Role::NameBrace => matches!(text, "{" | "}"),
		Role::ParameterDeclaration
		| Role::ParameterReference
		| Role::BindingDeclaration
		| Role::BindingReference
		| Role::External
		| Role::Unresolved =>
		{
			!text.contains(['{', '}'])
				&& !text.starts_with(char::is_whitespace)
				&& !text.ends_with(char::is_whitespace)
		},
		Role::Unrecognized => !text.contains(is_token_space)
	}
}

/// Check the properties of every source: [`compile`] fails to parse just when
/// [`Parser::parse`] does, with the same error; and [`diagnose`] reports a
/// diagnostic if the source fails to parse, leaves it as it is if it parses,
/// and corrects it only to a source that parses.
///
/// # Parameters
/// - `source`: The source.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
fn check_source(source: &str) -> Result<(), TestCaseError>
{
	let parsed = Parser::parse(source);
	match (&parsed, compile(source))
	{
		(Err(expected), Err(CompilationError::ParseError(actual))) =>
		{
			prop_assert_eq!(&actual, expected)
		},
		(Err(_), _) => prop_assert!(false, "compile parsed a failing source"),
		(Ok(_), Err(CompilationError::ParseError(e))) =>
		{
			prop_assert!(false, "compile failed to parse:\n{}", e)
		},
		(Ok(_), _) =>
		{}
	}
	let diagnosis = diagnose(source);
	match &parsed
	{
		Ok(_) =>
		{
			prop_assert_eq!(diagnosis.corrected_source.as_deref(), Some(source))
		},
		Err(_) => prop_assert!(
			!diagnosis.diagnostics.is_empty(),
			"no diagnostics for a failing source"
		)
	}
	if let Some(corrected) = &diagnosis.corrected_source
	{
		prop_assert!(
			Parser::parse(corrected).is_ok(),
			"corrected source {:?} fails",
			corrected
		);
	}
	Ok(())
}

/// Check the properties of metered evaluation of a source, if it compiles: it
/// evaluates within its budget, or else is refused, and never charges more
/// than the budget; it is refused only if its worst case,
/// [`dice`](crate::Bounds::dice), exceeds the budget; it rolls no more dice
/// than its worst case; and within its budget it agrees with unmetered
/// evaluation from the same seed. Likewise, its distribution builds within the
/// same budget, now of steps and of cells, or else is refused, and never
/// charges more than the budget; it is refused only if its
/// [estimate](crate::DistributionPlan::estimate) exceeds the budget; its
/// outcomes lie within the bounds of the value; and it contains the result of
/// the evaluation, if both fit their budgets.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments, of which the function takes as many as its arity.
/// - `externals`: The values of the externals, by index into [`NAMES`].
/// - `budget`: The dice budget, and the budget of steps and of cells of the
///   distribution.
/// - `seed`: The seed of the pRNG.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
fn check_evaluation(
	source: &str,
	args: &[i32],
	externals: &[i32],
	budget: u64,
	seed: u64
) -> Result<(), TestCaseError>
{
	let Ok(function) = compile(source)
	else
	{
		return Ok(())
	};
	let mut evaluator = Evaluator::new(function);
	let args = &args[..evaluator.function.arity()];
	let externals = NAMES
		.iter()
		.zip(externals)
		.filter(|(name, _)| {
			evaluator.function.externals.contains(&name.to_string())
		})
		.map(|(name, value)| (*name, *value))
		.collect::<Vec<_>>();
	for (name, value) in &externals
	{
		evaluator.bind(name, *value).unwrap();
	}
	let bounds = evaluator
		.bounds_over(
			args.iter().map(|arg| Some((*arg).into())),
			externals
				.iter()
				.map(|(name, value)| (*name, (*value).into()))
		)
		.unwrap();
	let metered = evaluator.evaluate_metered(
		args.iter().copied(),
		&mut StdRng::seed_from_u64(seed),
		budget
	);
	match &metered
	{
		Ok(evaluation) =>
		{
			prop_assert!(evaluation.dice <= budget, "over budget");
			prop_assert!(evaluation.dice <= bounds.dice, "over worst case");
			// Having fit the budget, the evaluation is cheap enough to repeat
			// without a meter.
			let unmetered = evaluator
				.evaluate(
					args.iter().copied(),
					&mut StdRng::seed_from_u64(seed)
				)
				.unwrap();
			prop_assert_eq!(evaluation, &unmetered);
		},
		Err(EvaluationError::DiceBudgetExhausted {
			requested,
			remaining,
			consumed
		}) =>
		{
			prop_assert!(requested > remaining, "refused an affordable roll");
			prop_assert_eq!(consumed.checked_add(*remaining), Some(budget));
			prop_assert!(bounds.dice > budget, "refused within the worst case");
		},
		Err(e) => prop_assert!(false, "evaluation failed: {}", e)
	}
	let plan = evaluator.plan_distribution(args.iter().copied()).unwrap();
	let within = Budget {
		steps: budget,
		cells: budget
	};
	match plan.build(within, &Unobserved)
	{
		Ok(distribution) =>
		{
			for (outcome, _) in &distribution
			{
				prop_assert!(
					bounds.value.contains(outcome),
					"outcome {} out of bounds {}",
					outcome,
					bounds.value
				);
			}
			if let Ok(evaluation) = &metered
			{
				prop_assert!(
					!distribution.get(evaluation.result).is_zero(),
					"distribution lacks the evaluated result {}",
					evaluation.result
				);
			}
		},
		Err(BuildError::BudgetExhausted {
			estimate,
			requested,
			remaining,
			consumed,
			..
		}) =>
		{
			prop_assert!(requested > remaining, "refused an affordable charge");
			prop_assert_eq!(consumed.checked_add(remaining), Some(budget));
			prop_assert!(
				estimate.steps > budget || estimate.cells > budget,
				"refused within the estimate"
			);
		},
		// The forward pass refuses only explodes without limit that may grow
		// a chain without bound.
		Err(BuildError::Unbounded) => prop_assert!(
			evaluator.function.instructions.iter().any(|inst| matches!(
				inst,
				Instruction::Explode(Explode {
					limit: Limit::Unlimited,
					..
				})
			)),
			"refused a function that explodes no dice without limit"
		),
		Err(e) => prop_assert!(false, "distribution failed: {}", e)
	}
	Ok(())
}

/// Check that optimizing a source, if it compiles, never changes its
/// distribution: its exact [distributions](crate::DistributionPlan::build),
/// unoptimized and fully optimized, agree outcome for outcome, as
/// probabilities, whenever both fit [`DISTRIBUTION_BUDGET`]; and the
/// [bounds](Evaluator::bounds_over) of its optimized value lie within those
/// of its unoptimized value. Optimization may
/// change the number of paths to each outcome, e.g., by eliminating a roll
/// whose result is never used, which multiplies the paths to every outcome
/// alike, so the distributions are compared as probabilities rather than as
/// counts. Optimization changes the work of the build, which may fit the
/// budget one way and not the other, so a refusal of either build abstains.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments, of which the function takes as many as its arity.
/// - `externals`: The values of the externals, by index into [`NAMES`].
///
/// # Errors
/// [`TestCaseError`] if the property fails.
fn check_optimization(
	source: &str,
	args: &[i32],
	externals: &[i32]
) -> Result<(), TestCaseError>
{
	let externals = NAMES
		.iter()
		.copied()
		.zip(externals.iter().copied())
		.collect::<Vec<_>>();
	check_optimization_with(source, args, &externals)
}

/// Check that optimizing a source, if it compiles, never changes its
/// distribution, as [`check_optimization`] does, but with externals of any
/// names.
///
/// # Parameters
/// - `source`: The source.
/// - `args`: The arguments, of which the function takes as many as its arity.
/// - `externals`: The names and values of the externals, of which the function
///   binds those that it reads.
///
/// # Errors
/// [`TestCaseError`] if the property fails.
fn check_optimization_with(
	source: &str,
	args: &[i32],
	externals: &[(&str, i32)]
) -> Result<(), TestCaseError>
{
	let Ok(unoptimized) = compile_unoptimized(source)
	else
	{
		return Ok(())
	};
	let optimized = StandardOptimizer::new(Passes::all())
		.optimize(unoptimized.clone())
		.unwrap();
	let args = &args[..unoptimized.arity()];
	let evaluator = |function: &IrFunction| {
		let mut evaluator = Evaluator::new(function.clone());
		let externals = externals
			.iter()
			.filter(|(name, _)| function.externals.contains(&name.to_string()))
			.copied()
			.collect::<Vec<_>>();
		for (name, value) in &externals
		{
			evaluator.bind(name, *value).unwrap();
		}
		let bounds = evaluator
			.bounds_over(
				args.iter().map(|arg| Some((*arg).into())),
				externals
					.iter()
					.map(|(name, value)| (*name, (*value).into()))
			)
			.unwrap();
		(evaluator, bounds.value)
	};
	let (unoptimized_evaluator, unoptimized_bounds) = evaluator(&unoptimized);
	let (optimized_evaluator, optimized_bounds) = evaluator(&optimized);
	prop_assert!(
		unoptimized_bounds.min <= optimized_bounds.min
			&& optimized_bounds.max <= unoptimized_bounds.max,
		"optimized bounds {} escape unoptimized bounds {}\nunoptimized:\n{}\n\
		 optimized:\n{}",
		optimized_bounds,
		unoptimized_bounds,
		unoptimized,
		optimized
	);
	let distribution = |evaluator: Evaluator| {
		evaluator
			.plan_distribution(args.iter().copied())
			.unwrap()
			.build(DISTRIBUTION_BUDGET, &Unobserved)
	};
	let (Ok(before), Ok(after)) = (
		distribution(unoptimized_evaluator),
		distribution(optimized_evaluator)
	)
	else
	{
		return Ok(())
	};
	let outcomes = |distribution: &Distribution| {
		distribution
			.iter()
			.map(|(outcome, _)| outcome)
			.collect::<Vec<_>>()
	};
	prop_assert_eq!(
		outcomes(&before),
		outcomes(&after),
		"outcomes differ\nunoptimized:\n{}\noptimized:\n{}",
		unoptimized,
		optimized
	);
	for outcome in outcomes(&before)
	{
		prop_assert_eq!(
			before.probability(outcome),
			after.probability(outcome),
			"probability of {} differs\nunoptimized:\n{}\noptimized:\n{}",
			outcome,
			unoptimized,
			optimized
		);
	}
	Ok(())
}

////////////////////////////////////////////////////////////////////////////////
//                                 Programs.                                  //
////////////////////////////////////////////////////////////////////////////////

/// The canonical names of parameters, variables, and bindings. Parameters come
/// from the first two, so that programs reference both parameters and
/// externals, and may bind names that collide with either. The names draw on
/// the whole identifier set, e.g., spaces, `:`, `/`, non-ASCII characters, and
/// characters that would be syntax outside of braces.
pub(super) const NAMES: &[&str] =
	&["x", "hit points", "weapon: 2/3", "1d6 drop lowest", "Ω-(1)"];

/// The runs of whitespace that may pad a name or stand for a space within it,
/// all of which [canonicalize](crate::parser::canonical_name) away.
const WHITESPACE: &[&str] = &[" ", "  ", "\t", "\n", "\u{A0}"];

/// The binary operators, including the alternate glyphs.
pub(super) const OPERATORS: &[&str] = &["+", "-", "*", "/", "%", "^", "×", "÷"];

/// The greatest depth of the expressions in a program.
const MAX_DEPTH: u32 = 8;

/// The number of nodes that the expressions in a program aim for.
const TARGET_SIZE: u32 = 128;

/// Answer a strategy that generates programs: syntactically valid functions,
/// with or without parameters, whose bodies nest every construct of the
/// grammar. Programs need not be semantically valid, e.g., they may bind a
/// name twice.
///
/// # Returns
/// The strategy.
fn program() -> impl Strategy<Value = String> { with_parameters(expression()) }

/// Answer a strategy that generates functions, with or without parameters,
/// whose bodies come from the specified strategy. The parameters are one or
/// both of the first two [names](NAMES), in random [spellings](spelling).
///
/// # Parameters
/// - `body`: The strategy for the bodies.
///
/// # Returns
/// The strategy.
pub(super) fn with_parameters(
	body: impl Strategy<Value = String>
) -> impl Strategy<Value = String>
{
	let parameters =
		(subsequence(&NAMES[..2], 1..=2), [spelling(), spelling()]).prop_map(
			|(names, spellings)| {
				names
					.iter()
					.zip(&spellings)
					.map(|(name, spelling)| spelling.braced(name))
					.collect::<Vec<_>>()
					.join(", ")
			}
		);
	(option::of(parameters), body).prop_map(|(parameters, body)| {
		match parameters
		{
			Some(parameters) => format!("{}: {}", parameters, body),
			None => body
		}
	})
}

/// Answer a strategy that generates expressions.
///
/// # Returns
/// The strategy.
fn expression() -> impl Strategy<Value = String>
{
	let leaf = prop_oneof![constant(), variable()];
	leaf.prop_recursive(MAX_DEPTH, TARGET_SIZE, 3, |inner| {
		prop_oneof![
			(inner.clone(), select(OPERATORS), inner.clone()).prop_map(
				|(left, op, right)| format!("{} {} {}", left, op, right)
			),
			inner.clone().prop_map(|operand| format!("-{}", operand)),
			inner.clone().prop_map(|e| format!("({})", e)),
			(variable(), inner.clone())
				.prop_map(|(name, e)| format!("{}@({})", name, e)),
			(inner.clone(), inner.clone())
				.prop_map(|(start, end)| format!("[{}:{}]", start, end)),
			extremum(inner.clone()),
			dice(inner)
		]
	})
}

/// Answer a strategy that generates calls of `highest` and `lowest`, whose
/// two or more arguments come from the specified strategy. Whitespace may
/// separate the keyword from the `(`.
///
/// # Parameters
/// - `inner`: The strategy for nested expressions.
///
/// # Returns
/// The strategy.
pub(super) fn extremum(
	inner: BoxedStrategy<String>
) -> impl Strategy<Value = String>
{
	(
		select(&["highest", "lowest"][..]),
		select(&["", " "][..]),
		prop::collection::vec(inner, 2..5)
	)
		.prop_map(|(keyword, space, arguments)| {
			format!("{}{}({})", keyword, space, arguments.join(", "))
		})
}

/// Answer a strategy that generates dice expressions, whose operands come
/// from the specified strategy, with drop, keep, reroll, and explode clauses in
/// any order. Any drop or keep clause may lack a drop expression, and any
/// reroll or explode clause its limit.
///
/// # Parameters
/// - `inner`: The strategy for nested expressions.
///
/// # Returns
/// The strategy.
fn dice(inner: BoxedStrategy<String>) -> impl Strategy<Value = String>
{
	let faces = prop_oneof![
		atom(inner.clone()),
		prop::collection::vec(-3i32..=20, 1..6).prop_map(|faces| {
			let faces = faces.iter().map(i32::to_string).collect::<Vec<_>>();
			format!("[{}]", faces.join(", "))
		})
	];
	// A drop expression never begins with `-`, so a negative constant must be
	// grouped.
	let drop = atom(inner.clone())
		.prop_map(|drop| {
			if drop.starts_with('-')
			{
				format!("({})", drop)
			}
			else
			{
				drop
			}
		})
		.boxed();
	let clause = (
		select(&["drop", "keep"][..]),
		select(&["lowest", "highest"][..]),
		option::of(drop.clone())
	)
		.prop_map(|(verb, direction, drop)| match drop
		{
			Some(drop) => format!(" {} {} {}", verb, direction, drop),
			None => format!(" {} {}", verb, direction)
		});
	let clause = prop_oneof![
		2 => clause,
		1 => predicated("reroll", inner.clone(), drop.clone()),
		1 => predicated("explode", inner.clone(), drop)
	];
	(
		atom(inner),
		select(&["d", "D"][..]),
		faces,
		prop::collection::vec(clause, 0..3)
	)
		.prop_map(|(count, operator, faces, clauses)| {
			format!("{}{}{}{}", count, operator, faces, clauses.concat())
		})
}

/// Answer a strategy that generates reroll or explode clauses, whose
/// thresholds and endpoints, which may be negative constants, come from the
/// specified strategy, of every form of predicate and limit.
///
/// # Parameters
/// - `verb`: The verb of the clauses, `reroll` or `explode`.
/// - `inner`: The strategy for nested expressions.
/// - `count`: The strategy for the counts of limits, which never begin with
///   `-`.
///
/// # Returns
/// The strategy.
fn predicated(
	verb: &'static str,
	inner: BoxedStrategy<String>,
	count: impl Strategy<Value = String> + 'static
) -> impl Strategy<Value = String>
{
	let comparator =
		select(&["=", "!=", "≠", "<", "<=", "≤", ">", ">=", "≥"][..]);
	let predicate = prop_oneof![
		(comparator, atom(inner.clone())).prop_map(
			|(comparator, threshold)| {
				format!("{} {}", comparator, threshold)
			}
		),
		prop::collection::vec(-3i32..=20, 0..4).prop_map(|values| {
			let values = values.iter().map(i32::to_string).collect::<Vec<_>>();
			format!("in [{}]", values.join(", "))
		}),
		(atom(inner.clone()), select(&["..", "..="][..]), atom(inner))
			.prop_map(|(start, operator, end)| {
				format!("in {}{}{}", start, operator, end)
			})
	];
	let limit = prop_oneof![
		(count, select(&["time", "times"][..]))
			.prop_map(|(count, word)| format!(" {} {}", count, word)),
		(
			select(&["∞", "unlimited"][..]),
			select(&["time", "times"][..])
		)
			.prop_map(|(keyword, word)| format!(" {} {}", keyword, word))
	];
	(predicate, option::of(limit)).prop_map(move |(predicate, limit)| {
		format!(" {} {}{}", verb, predicate, limit.unwrap_or_default())
	})
}

/// Answer a strategy that generates the operands that the grammar admits as
/// dice counts, standard faces, and drop expressions: constants, variables,
/// groups, and bindings.
///
/// # Parameters
/// - `inner`: The strategy for nested expressions.
///
/// # Returns
/// The strategy.
fn atom(inner: BoxedStrategy<String>) -> impl Strategy<Value = String>
{
	prop_oneof![
		constant(),
		variable(),
		inner.clone().prop_map(|e| format!("({})", e)),
		(variable(), inner).prop_map(|(name, e)| format!("{}@({})", name, e))
	]
}

/// Answer a strategy that generates constants, mostly small ones, but
/// sometimes any [`i32`].
///
/// # Returns
/// The strategy.
fn constant() -> impl Strategy<Value = String>
{
	prop_oneof![
		4 => (-3i32..=20).prop_map(|n| n.to_string()),
		1 => any::<i32>().prop_map(|n| n.to_string())
	]
}

/// Answer a strategy that generates variables: braced [names](NAMES), in
/// random [spellings](spelling).
///
/// # Returns
/// The strategy.
pub(super) fn variable() -> impl Strategy<Value = String>
{
	(select(NAMES), spelling())
		.prop_map(|(name, spelling)| spelling.braced(name))
}

/// How to spell a canonical name in a source: mostly as the name itself, but
/// sometimes padded at either end, or with some spaces replaced, by runs of
/// [whitespace](WHITESPACE), which denote the same name.
///
/// A spelling is drawn independently of the name that it spells, rather than
/// flat-mapped from it, because flat-mapped value trees at every leaf of a
/// deep [expression](expression) overflow the [small stack](on_small_stack).
#[derive(Clone, Debug)]
struct Spelling
{
	/// The run of whitespace before the name.
	leading: &'static str,

	/// The run of whitespace for each space in the name, in order. There are
	/// enough for any of [`NAMES`]; a name with fewer spaces ignores the rest.
	gaps: Vec<&'static str>,

	/// The run of whitespace after the name.
	trailing: &'static str
}

impl Spelling
{
	/// Spell a canonical name, in braces.
	///
	/// # Parameters
	/// - `name`: The canonical name.
	///
	/// # Returns
	/// The braced spelling.
	fn braced(&self, name: &str) -> String
	{
		let mut words = name.split(' ');
		let mut spelling = format!("{{{}", self.leading);
		spelling.extend(words.next());
		for (gap, word) in self.gaps.iter().zip(words)
		{
			spelling.push_str(gap);
			spelling.push_str(word);
		}
		spelling.push_str(self.trailing);
		spelling.push('}');
		spelling
	}
}

/// Answer a strategy that generates [spellings](Spelling).
///
/// # Returns
/// The strategy.
fn spelling() -> impl Strategy<Value = Spelling>
{
	let run = || select(WHITESPACE);
	let gap = prop_oneof![3 => Just(" "), 1 => run()];
	let gaps = NAMES
		.iter()
		.map(|name| name.matches(' ').count())
		.max()
		.unwrap_or_default();
	(
		option::weighted(0.25, run()),
		prop::collection::vec(gap, gaps),
		option::weighted(0.25, run())
	)
		.prop_map(|(leading, gaps, trailing)| Spelling {
			leading: leading.unwrap_or_default(),
			gaps,
			trailing: trailing.unwrap_or_default()
		})
}

////////////////////////////////////////////////////////////////////////////////
//                               Distributions.                               //
////////////////////////////////////////////////////////////////////////////////

/// Answer whether two distributions agree: they have the same outcomes, each
/// with the same probability, though their totals may differ.
///
/// # Parameters
/// - `a`: One distribution.
/// - `b`: The other distribution.
///
/// # Returns
/// `true` if the distributions agree, `false` otherwise.
pub(super) fn agree(a: &Distribution, b: &Distribution) -> bool
{
	a.len() == b.len()
		&& a.iter().zip(b).all(|((x, _), (y, _))| {
			x == y && a.probability(x) == b.probability(y)
		})
}

////////////////////////////////////////////////////////////////////////////////
//                                 Bindings.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Answer a strategy that generates the values of arguments and externals,
/// favoring the extremes of [`i32`] and its neighborhood of zero.
///
/// # Returns
/// The strategy.
pub(super) fn binding() -> impl Strategy<Value = i32>
{
	prop_oneof![
		select(&[i32::MIN, -1, 0, 1, i32::MAX][..]),
		-3i32..=20,
		any::<i32>()
	]
}

/// Answer a strategy that generates dice budgets, which are small enough that
/// every evaluation within one is prompt.
///
/// # Returns
/// The strategy.
fn budget() -> impl Strategy<Value = u64>
{
	prop_oneof![0u64..=16, 0u64..=10_000]
}

////////////////////////////////////////////////////////////////////////////////
//                                 Mutations.                                 //
////////////////////////////////////////////////////////////////////////////////

/// An edit of a source at a character boundary, chosen by an [`Index`] into
/// its boundaries.
#[derive(Clone, Debug)]
enum Mutation
{
	/// Insert a token.
	Insert(Index, &'static str),

	/// Delete a character.
	Delete(Index),

	/// Replace a character with a token.
	Replace(Index, &'static str),

	/// Delete everything from the boundary onward.
	Truncate(Index)
}

impl Mutation
{
	/// Apply the mutation to a source.
	///
	/// # Parameters
	/// - `source`: The source, which the mutation edits in place.
	fn apply(&self, source: &mut String)
	{
		let boundaries = source
			.char_indices()
			.map(|(i, _)| i)
			.chain([source.len()])
			.collect::<Vec<_>>();
		let at = |index: &Index| boundaries[index.index(boundaries.len())];
		// The character that begins at a boundary, if any.
		let end = |start: usize| {
			source[start..]
				.chars()
				.next()
				.map_or(start, |c| start + c.len_utf8())
		};
		match self
		{
			Mutation::Insert(index, token) =>
			{
				source.insert_str(at(index), token)
			},
			Mutation::Delete(index) =>
			{
				let start = at(index);
				source.replace_range(start..end(start), "");
			},
			Mutation::Replace(index, token) =>
			{
				let start = at(index);
				source.replace_range(start..end(start), token);
			},
			Mutation::Truncate(index) => source.truncate(at(index))
		}
	}
}

/// Answer a strategy that generates mutations, mostly insertions, deletions,
/// and replacements, whose tokens come from the [corpus](TOKENS).
///
/// # Returns
/// The strategy.
fn mutation() -> impl Strategy<Value = Mutation>
{
	prop_oneof![
		3 => (any::<Index>(), select(TOKENS))
			.prop_map(|(index, token)| Mutation::Insert(index, token)),
		3 => any::<Index>().prop_map(Mutation::Delete),
		3 => (any::<Index>(), select(TOKENS))
			.prop_map(|(index, token)| Mutation::Replace(index, token)),
		1 => any::<Index>().prop_map(Mutation::Truncate)
	]
}

/// Answer a strategy that generates strings of up to 24 [tokens](TOKENS) of
/// the corpus, each perhaps followed by a space.
///
/// # Returns
/// The strategy.
fn token_string() -> impl Strategy<Value = String>
{
	prop::collection::vec((select(TOKENS), select(&["", " "][..])), 0..=24)
		.prop_map(|parts| {
			parts
				.into_iter()
				.flat_map(|(token, space)| [token, space])
				.collect()
		})
}

/// Answer a strategy that generates [programs](program) with one to three
/// [mutations](mutation) applied in turn.
///
/// # Returns
/// The strategy.
fn mutated_program() -> impl Strategy<Value = String>
{
	(program(), prop::collection::vec(mutation(), 1..=3)).prop_map(
		|(mut source, mutations)| {
			for mutation in &mutations
			{
				mutation.apply(&mut source);
			}
			source
		}
	)
}
