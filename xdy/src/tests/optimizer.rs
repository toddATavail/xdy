//! # Optimizer tests
//!
//! Herein are tests for the optimizer. The actual test cases are stored in
//! `../../tests`, variously among:
//!
//! * `test_common_subexpression_elimination.txt`
//! * `test_constant_commuting.txt`
//! * `test_constant_folding.txt`
//! * `test_full_optimization.txt`
//! * `test_register_coalescence.txt`
//! * `test_strength_reduction.txt`
//!
//! Each file comprises a series of test cases, each of which consists of a
//! source dice expression and an expected print rendition.

use std::collections::HashSet;

use pretty_assertions::assert_eq;

use rand::{SeedableRng, rngs::StdRng};

use crate::{
	Evaluator, Pass, Passes,
	support::{compile_valid, optimize, read_compilation_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                             Other test cases.                              //
////////////////////////////////////////////////////////////////////////////////

/// Test that the common subexpression eliminator generates the expected output
/// for the test cases.
#[test]
fn test_common_subexpression_elimination()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_common_subexpression_elimination.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(
			function.clone(),
			Pass::CommonSubexpressionElimination.into()
		);
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant commuter generates the expected output for the test
/// cases.
#[test]
fn test_constant_commuting()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_constant_commuting.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantCommuting.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the constant folder generates the expected output for the test
/// cases.
#[test]
fn test_constant_folding()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_constant_folding.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::ConstantFolding.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the strength reducer generates the expected output for the test
/// cases.
#[test]
fn test_strength_reduction()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_strength_reduction.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::StrengthReduction.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the register coalescer generates the expected output for the test
/// cases.
#[test]
fn test_register_coalescence()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_register_coalescence.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized =
			optimize(function.clone(), Pass::RegisterCoalescing.into());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the fully-loaded optimizer produces the expected output for the
/// test cases.
#[test]
fn test_full_optimization()
{
	let mut seen = HashSet::new();
	for (index, (source, expected)) in read_compilation_test_cases(
		include_str!("../../tests/test_full_optimization.txt")
	)
	.iter()
	.enumerate()
	{
		assert!(seen.insert(source), "duplicate test case: {}", source);
		let function = compile_valid(source);
		let optimized = optimize(function.clone(), Passes::all());
		let actual = format!("{}", optimized);
		assert_eq!(
			actual.trim(),
			*expected,
			"case {}: {}\nunoptimized:\n{}",
			index + 1,
			source,
			function
		);
	}
}

/// Test that the optimizer folds a constant [maximum](crate::Max), which only
/// the optimizer emits, applies its identities, and gathers the constants of a
/// chain of maxima so that they fold.
#[test]
fn test_optimize_max()
{
	for (body, expected) in [
		// Constant folding.
		("@1 <- 3 max 5\n\t\treturn @1", "return 5"),
		// The least value is the identity.
		("@1 <- @0 max -2147483648\n\t\treturn @1", "return @0"),
		// The greatest value absorbs.
		(
			"@1 <- 2147483647 max @0\n\t\treturn @1",
			"return 2147483647"
		),
		// The maximum is idempotent.
		("@1 <- @0 max @0\n\t\treturn @1", "return @0"),
		// Commutation gathers the constants of a chain, which then fold.
		(
			"@1 <- @0 max 3\n\t\t@2 <- 5 max @1\n\t\treturn @2",
			"@0 <- 5 max @0\n\t\treturn @0"
		)
	]
	{
		let registers = body.matches("<-").count() + 1;
		let text = format!(
			"Function({{x}}@0) r#{} ⚅#0\n\textern[]\n\tbody:\n\t\t{}\n",
			registers, body
		);
		let function = crate::Assembler::assemble(&text).unwrap();
		let optimized = optimize(function, Passes::all()).to_string();
		let actual = optimized.trim().split_once("body:\n\t\t").unwrap().1;
		assert_eq!(actual, expected, "{}", body);
	}
}

/// Test that register coalescing never lets a write that nothing reads clobber
/// a register that is live across it.
#[test]
fn test_coalescing_keeps_dead_writes_apart()
{
	let function = crate::Assembler::assemble(
		"\
Function({x}@0) r#4 ⚅#0
\textern[]
\tbody:
\t\t@1 <- @0 + 1
\t\t@2 <- @0 + 2
\t\t@3 <- @1 + @0
\t\treturn @3
"
	)
	.unwrap();
	let coalesced = optimize(function.clone(), Pass::RegisterCoalescing.into());
	let evaluate = |function| {
		Evaluator::new(function)
			.evaluate([10], &mut StdRng::seed_from_u64(0))
			.unwrap()
			.result
	};
	assert_eq!(evaluate(coalesced.clone()), 21, "{}", coalesced);
	assert_eq!(evaluate(function), 21);
}

/// Test that no pass folds or reduces a roll that keeps as though its keeps
/// were drops: every die of `4D1` shows 1, but a keep counts from the dice
/// still kept, so `4D1 keep highest 2` is 2, not the 4 that counting only the
/// drops would answer, and `5D1 drop lowest 1 keep highest 2 drop lowest 1` is
/// 1.
#[test]
fn test_keeps_are_not_folded_as_drops()
{
	for (clauses, expected) in [
		("\t\t⚅0 <- keep highest 2 from ⚅0\n", 2),
		("\t\t⚅0 <- keep lowest 0 from ⚅0\n", 0),
		(
			"\t\t⚅0 <- drop lowest 1 from ⚅0\n\t\t⚅0 <- keep highest 2 from \
			 ⚅0\n\t\t⚅0 <- drop lowest 1 from ⚅0\n",
			1
		)
	]
	{
		for die in ["standard dice 5D1", "custom dice 5D[1, 1]"]
		{
			let function = crate::Assembler::assemble(&format!(
				"Function() r#1 ⚅#1
	extern[]
	body:
		⚅0 <- roll {die}
{clauses}		@0 <- sum rolling record ⚅0
		return @0"
			))
			.unwrap();
			for passes in [
				Pass::ConstantFolding.into(),
				Pass::StrengthReduction.into(),
				Passes::all()
			]
			{
				let optimized = optimize(function.clone(), passes);
				let result = Evaluator::new(optimized.clone())
					.evaluate([], &mut StdRng::seed_from_u64(0))
					.unwrap()
					.result;
				assert_eq!(result, expected, "{:?}\n{}", passes, optimized);
			}
		}
	}
}

/// Test that constant folding applies the drops and keeps of a roll, in order,
/// to its count of dice, when every count is constant, and folds the roll if
/// they keep none of its dice or its dice show only one face. A keep counts
/// from the dice still kept, so `4D1 drop lowest 1 keep highest 2` keeps 2
/// dice, but `4D1 keep highest 2 drop lowest 1` keeps 1. A roll that keeps
/// some of its dice, which show several faces, does not fold, nor does a roll
/// with a keep whose count is not constant.
#[test]
fn test_folding_applies_keeps_in_order()
{
	let body = |roll: &str, clauses: &[&str]| {
		let clauses = clauses
			.iter()
			.map(|clause| format!("\t\t⚅0 <- {clause} from ⚅0\n"))
			.collect::<String>();
		format!(
			"\t\t⚅0 <- roll {roll}\n{clauses}\t\t@0 <- sum rolling record \
			 ⚅0\n\t\treturn @0\n"
		)
	};
	for (roll, clauses, expected) in [
		("standard dice 4D1", &["keep highest 2"][..], Some(2)),
		(
			"standard dice 4D1",
			&["drop lowest 1", "keep highest 2"],
			Some(2)
		),
		(
			"standard dice 4D1",
			&["keep highest 2", "drop lowest 1"],
			Some(1)
		),
		("standard dice 3D1", &["keep lowest 7"], Some(3)),
		(
			"custom dice 5D[2, 2]",
			&["drop lowest 1", "keep highest 2", "drop lowest 1"],
			Some(2)
		),
		(
			"custom dice 4D[-3, -3]",
			&["keep lowest 3", "keep highest 2"],
			Some(-6)
		),
		("standard dice 4D6", &["keep lowest 0"], Some(0)),
		("standard dice 4D6", &["keep highest -1"], Some(0)),
		(
			"standard dice 4D6",
			&["keep highest 2", "drop lowest 2"],
			Some(0)
		),
		(
			"custom dice 4D[1, 5]",
			&["drop lowest 1", "keep highest 5", "drop highest 3"],
			Some(0)
		),
		(
			"standard dice 4D6",
			&["keep highest 3", "drop lowest 1"],
			None
		),
		("custom dice 4D[1, 5]", &["keep lowest 1"], None)
	]
	{
		let body = body(roll, clauses);
		let function = crate::Assembler::assemble(&format!(
			"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n{body}"
		))
		.unwrap();
		let folded = optimize(function.clone(), Pass::ConstantFolding.into())
			.to_string();
		let actual = folded.split_once("body:\n").unwrap().1;
		match expected
		{
			Some(expected) =>
			{
				assert_eq!(
					actual,
					format!("\t\treturn {expected}\n"),
					"{body}"
				);
				let result = Evaluator::new(function)
					.evaluate([], &mut StdRng::seed_from_u64(0))
					.unwrap()
					.result;
				assert_eq!(result, expected, "{body}");
			},
			None => assert_eq!(actual, body)
		}
	}
	let body = "\
\t\t⚅0 <- roll range 1:2
\t\t@0 <- sum rolling record ⚅0
\t\t⚅1 <- roll standard dice 4D1
\t\t⚅1 <- keep highest @0 from ⚅1
\t\t@1 <- sum rolling record ⚅1
\t\treturn @1
";
	let function = crate::Assembler::assemble(&format!(
		"Function() r#2 ⚅#2\n\textern[]\n\tbody:\n{body}"
	))
	.unwrap();
	let folded = optimize(function, Pass::ConstantFolding.into()).to_string();
	assert_eq!(folded.split_once("body:\n").unwrap().1, body);
}

/// Test that strength reduction never combines two drops of a record across a
/// keep of it, since the keep counts from the dice that the drops before it
/// left: of `4D6`, dropping the lowest, keeping the lowest 2 of the other 3,
/// and dropping the lowest of those keeps the third lowest die, whereas
/// dropping the lowest 2 after keeping the lowest 2 of all 4 keeps nothing.
#[test]
fn test_drops_do_not_combine_across_keeps()
{
	let body = "\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- keep lowest 2 from ⚅0
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = crate::Assembler::assemble(&format!(
		"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n{body}"
	))
	.unwrap();
	let optimized = optimize(function, Passes::all()).to_string();
	assert_eq!(optimized.split_once("body:\n").unwrap().1, body);
}

/// Assemble a function that has no parameters from the specified body.
///
/// # Parameters
/// - `registers`: The number of registers.
/// - `records`: The number of rolling records.
/// - `body`: The instructions, each indented by two tabs and ending with a
///   newline.
///
/// # Returns
/// The function.
fn assemble_body(
	registers: usize,
	records: usize,
	body: &str
) -> crate::Function
{
	crate::Assembler::assemble(&format!(
		"Function() r#{registers} ⚅#{records}\n\textern[]\n\tbody:\n{body}"
	))
	.unwrap()
}

/// Test that neither constant folding nor strength reduction folds a roll
/// that a reroll targets, even when its dice show only one face, since an
/// unlimited reroll that matches that face crosses out every die, and that the
/// rolls that they still fold, which roll no dice or only zeros, sum to zero
/// whatever their rerolls.
#[test]
fn test_rerolls_are_not_folded()
{
	for (roll, clauses, expected) in [
		("standard dice 3D1", &["reroll = 1 unlimited times"][..], 0),
		("standard dice 3D1", &["reroll = 1 up to 2 times"], 3),
		("standard dice 3D1", &["reroll != 1 unlimited times"], 3),
		(
			"custom dice 3D[5, 5]",
			&["reroll in [5] unlimited times"],
			0
		),
		("custom dice 3D[5, 5]", &["reroll < 9 up to 1 times"], 15),
		("range 2:2", &["reroll in 2..=2 unlimited times"], 0),
		("range 2:2", &["reroll in 2..3 up to 4 times"], 2),
		(
			"standard dice 4D1",
			&["drop lowest 1", "reroll >= 1 unlimited times"],
			0
		),
		(
			"standard dice 4D1",
			&["reroll >= 1 unlimited times", "drop lowest 1"],
			0
		),
		(
			"standard dice 4D1",
			&["keep highest 2", "reroll = 1 up to 1 times"],
			2
		),
		("standard dice 0D6", &["reroll < 9 unlimited times"], 0),
		("standard dice 3D0", &["reroll = 0 up to 3 times"], 0),
		("range 3:1", &["reroll = 0 unlimited times"], 0)
	]
	{
		let clauses = clauses
			.iter()
			.map(|clause| format!("\t\t⚅0 <- {clause} from ⚅0\n"))
			.collect::<String>();
		let function = assemble_body(
			1,
			1,
			&format!(
				"\t\t⚅0 <- roll {roll}\n{clauses}\t\t@0 <- sum rolling record \
				 ⚅0\n\t\treturn @0\n"
			)
		);
		for passes in [
			Passes::none(),
			Pass::ConstantFolding.into(),
			Pass::StrengthReduction.into(),
			Passes::all()
		]
		{
			let optimized = optimize(function.clone(), passes);
			assert_eq!(optimized.validate(), Ok(()), "{}", optimized);
			let result = Evaluator::new(optimized.clone())
				.evaluate([], &mut StdRng::seed_from_u64(0))
				.unwrap()
				.result;
			assert_eq!(result, expected, "{:?}\n{}", passes, optimized);
		}
	}
}

/// Test that strength reduction never combines two drops of a record across a
/// reroll of it, since the reroll changes the dice that the later drop sees.
#[test]
fn test_drops_do_not_combine_across_rerolls()
{
	let body = "\
\t\t⚅0 <- roll standard dice 4D6
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- reroll < 3 unlimited times from ⚅0
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let optimized = optimize(assemble_body(1, 1, body), Passes::all());
	assert_eq!(optimized.to_string().split_once("body:\n").unwrap().1, body);
}

/// Test that every pass renames the operands of a reroll, and that dead code
/// elimination removes a reroll of a record that nothing sums, so that the
/// optimized function is well formed and evaluates as the original does.
#[test]
fn test_optimizer_rewrites_rerolls()
{
	let function = crate::Assembler::assemble(
		"\
Function({x}@0) r#7 ⚅#2
\textern[]
\tbody:
\t\t@1 <- @0 + 1
\t\t@2 <- @0 + 1
\t\t@3 <- @2 * 1
\t\t@4 <- 2 + 1
\t\t⚅0 <- roll standard dice 5D6
\t\t⚅0 <- reroll in @1..=@2 up to @3 times from ⚅0
\t\t⚅0 <- reroll != @4 up to @4 times from ⚅0
\t\t@5 <- sum rolling record ⚅0
\t\t⚅1 <- roll standard dice 2D6
\t\t⚅1 <- reroll = @1 unlimited times from ⚅1
\t\t@6 <- @5 + 0
\t\treturn @6
"
	)
	.unwrap();
	let optimized = optimize(function.clone(), Passes::all());
	assert_eq!(optimized.validate(), Ok(()), "{}", optimized);
	assert_eq!(
		optimized.to_string(),
		"\
Function({x}@0) r#1 ⚅#1
\textern[]
\tbody:
\t\t@0 <- 1 + @0
\t\t⚅0 <- roll standard dice 5D6
\t\t⚅0 <- reroll in @0..=@0 up to @0 times from ⚅0
\t\t⚅0 <- reroll != 3 up to 3 times from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
"
	);
	// The dead roll follows every live one, so removing it changes no draw.
	for x in -1..=6
	{
		for seed in 0..20
		{
			let expected = Evaluator::new(function.clone())
				.evaluate([x], &mut StdRng::seed_from_u64(seed))
				.unwrap()
				.result;
			let actual = Evaluator::new(optimized.clone())
				.evaluate([x], &mut StdRng::seed_from_u64(seed))
				.unwrap()
				.result;
			assert_eq!(actual, expected, "x = {}, seed = {}", x, seed);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Exactness.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Values of arguments that exercise the edges of the arithmetic: zero, the
/// units, and the extremes of [`i32`], where arithmetic saturates.
const EDGES: [i32; 9] = [
	i32::MIN,
	i32::MIN + 1,
	-65536,
	-1,
	0,
	1,
	65536,
	i32::MAX - 1,
	i32::MAX
];

/// The greatest number of combinations of [edge](EDGES) values over which
/// [`assert_exact`] evaluates a function. Functions with many variables take
/// their values from a prefix of the edges, so that the combinations stay
/// within it.
const COMBINATIONS: usize = 10_000;

/// Assert that optimizing the specified dice-free function with the specified
/// passes never changes its result, for every combination of [edge](EDGES)
/// values of its formal parameters and external variables.
///
/// # Parameters
/// - `source`: The source of the function, which rolls no dice.
/// - `passes`: The passes to apply.
fn assert_exact(source: &str, passes: Passes)
{
	let unoptimized = compile_valid(source);
	let optimized = optimize(unoptimized.clone(), passes);
	let arity = unoptimized.arity();
	let variables = arity + unoptimized.externals.len();
	// Take as many edges as keep the combinations within bounds.
	let mut edges = EDGES.len();
	while edges > 2 && edges.pow(variables as u32) > COMBINATIONS
	{
		edges -= 1;
	}
	let edges = &EDGES[EDGES.len() - edges..];
	for mut combination in 0..edges.len().pow(variables as u32)
	{
		let values = (0..variables)
			.map(|_| {
				let value = edges[combination % edges.len()];
				combination /= edges.len();
				value
			})
			.collect::<Vec<_>>();
		let evaluate = |function| {
			let mut evaluator = Evaluator::new(function);
			for (name, value) in
				unoptimized.externals.iter().zip(&values[arity..])
			{
				evaluator.bind(name, *value).unwrap();
			}
			evaluator
				.evaluate(
					values[..arity].iter().copied(),
					&mut StdRng::seed_from_u64(0)
				)
				.unwrap()
				.result
		};
		assert_eq!(
			evaluate(optimized.clone()),
			evaluate(unoptimized.clone()),
			"{} at {:?}\noptimized:\n{}",
			source,
			values,
			optimized
		);
	}
}

/// Test that no pass changes the result of any dice-free function of the
/// optimizer's test cases, at the edges of the arithmetic, alone or together
/// with the others.
#[test]
fn test_test_cases_are_exact()
{
	let files = [
		(
			include_str!(
				"../../tests/test_common_subexpression_elimination.txt"
			),
			Passes::from(Pass::CommonSubexpressionElimination)
		),
		(
			include_str!("../../tests/test_constant_commuting.txt"),
			Pass::ConstantCommuting.into()
		),
		(
			include_str!("../../tests/test_constant_folding.txt"),
			Pass::ConstantFolding.into()
		),
		(
			include_str!("../../tests/test_strength_reduction.txt"),
			Pass::StrengthReduction.into()
		),
		(
			include_str!("../../tests/test_register_coalescence.txt"),
			Pass::RegisterCoalescing.into()
		),
		(
			include_str!("../../tests/test_full_optimization.txt"),
			Passes::all()
		)
	];
	for (file, passes) in files
	{
		for (source, _) in read_compilation_test_cases(file)
		{
			if compile_valid(source).rolling_record_count == 0
			{
				assert_exact(source, passes);
				assert_exact(source, Passes::all());
			}
		}
	}
}

/// Test that strength reduction never changes a result: a value divided by
/// itself is zero when the value is zero, and a double negation of
/// [`i32::MIN`] is `-i32::MAX`, so neither reduces. The last
/// source nests a double negation around another instruction, which once
/// tripped an assertion.
#[test]
fn test_strength_reduction_is_exact()
{
	for source in [
		"{x}: {x} / {x}",
		"{x}: --{x}",
		"{x}: ---{x}",
		"{x}: --{x} + 2",
		"{x}, {y}: {a}@(-{x}) + {y} * 2 + -{a}",
		"{x}: {x} * -1",
		"{x}: {x} / -1",
		"{x}: 0 - {x}",
		"{x}: {x} * 2",
		"{x}: {x} ^ 2",
		"{x}: {x} % 1",
		"{x}: {x} ^ 0",
		"{x}: 1 ^ {x}"
	]
	{
		assert_exact(source, Pass::StrengthReduction.into());
		assert_exact(source, Passes::all());
	}
}
