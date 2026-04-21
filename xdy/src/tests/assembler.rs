//! # Assembler tests
//!
//! Herein are tests for the [`Assembler`](crate::Assembler), covering the full
//! cross-product of instruction shapes, addressing modes, and error paths. The
//! core of the suite is a pair of round-trip harnesses driven by
//! `../../tests/test_full_optimization.txt`, which exercises every
//! compiler-produced IR shape across the test corpus:
//!
//! 1. **Text round-trip.** For each expected print form in the corpus,
//!    [`Assembler::assemble`](crate::Assembler::assemble) produces a
//!    [`Function`](crate::Function) whose [`Display`](std::fmt::Display) output
//!    matches the original text byte-for-byte.
//! 2. **Struct round-trip.** For each source expression in the corpus,
//!    compiling and then assembling the compiler's own [`Display`] output
//!    reconstructs a structurally identical [`Function`].
//!
//! The remainder of the suite covers cases the corpus does not reach:
//! per-instruction positive coverage for each addressing-mode slot, negative
//! coverage for each [`AssemblyError`](crate::AssemblyError) path, and
//! [`FromStr`](std::str::FromStr) ergonomics.

use pretty_assertions::assert_eq;

use crate::{
	AddressingMode, Assembler, AssemblyError, DropKind, Function, Immediate,
	Instruction, RegisterIndex, RollingRecordIndex,
	support::{compile_valid, read_compilation_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                         Gold-standard round-trip.                          //
////////////////////////////////////////////////////////////////////////////////

/// Text round-trip: for every expected print form in the corpus, assemble it,
/// re-emit it, and assert equality.
#[test]
fn test_text_round_trip_full_optimization()
{
	let corpus = include_str!("../../tests/test_full_optimization.txt");
	for (index, (source, expected)) in
		read_compilation_test_cases(corpus).iter().enumerate()
	{
		let function = match Assembler::assemble(expected)
		{
			Ok(f) => f,
			Err(e) =>
			{
				panic!(
					"case {}: assemble failed for source `{}`:\n{}\nexpected:\n{}",
					index + 1,
					source,
					e,
					expected
				);
			}
		};
		let actual = format!("{}", function);
		assert_eq!(
			actual.trim(),
			*expected,
			"text round-trip mismatch for case {} source `{}`",
			index + 1,
			source
		);
	}
}

/// Struct round-trip: for every source in the corpus, compile it,
/// [`Display`]-format the result, assemble that text, and assert the assembled
/// [`Function`] equals the compiled one.
#[test]
fn test_struct_round_trip_full_optimization()
{
	let corpus = include_str!("../../tests/test_full_optimization.txt");
	for (index, (source, _expected)) in
		read_compilation_test_cases(corpus).iter().enumerate()
	{
		let compiled = compile_valid(source);
		let text = format!("{}", compiled);
		let reassembled = match Assembler::assemble(&text)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {} source `{}`: re-assemble failed: {}\ntext:\n{}",
				index + 1,
				source,
				e,
				text
			)
		};
		assert_eq!(
			reassembled,
			compiled,
			"struct round-trip mismatch for case {} source `{}`",
			index + 1,
			source
		);
	}
}

/// Text round-trip over the unoptimized corpus, for completeness and to cover
/// IR shapes the optimizer may remove.
#[test]
fn test_text_round_trip_compile_unoptimized()
{
	let corpus = include_str!("../../tests/test_compile_unoptimized.txt");
	for (index, (source, expected)) in
		read_compilation_test_cases(corpus).iter().enumerate()
	{
		let function = match Assembler::assemble(expected)
		{
			Ok(f) => f,
			Err(e) => panic!(
				"case {} source `{}`: assemble failed: {}\nexpected:\n{}",
				index + 1,
				source,
				e,
				expected
			)
		};
		let actual = format!("{}", function);
		assert_eq!(
			actual.trim(),
			*expected,
			"text round-trip mismatch for case {} source `{}`",
			index + 1,
			source
		);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Positive tests.                               //
////////////////////////////////////////////////////////////////////////////////

/// Assemble the bare-minimum function: no parameters, no externs, no registers,
/// no rolling records, and a single [`Return`] instruction.
#[test]
fn test_assemble_constant_return()
{
	let text = "\
Function() r#0 ⚅#0
\textern[]
\tbody:
\t\treturn 0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(function.parameters, Vec::<String>::new());
	assert_eq!(function.externals, Vec::<String>::new());
	assert_eq!(function.register_count, 0);
	assert_eq!(function.rolling_record_count, 0);
	assert_eq!(
		function.instructions,
		vec![Instruction::r#return(AddressingMode::Immediate(Immediate(
			0
		)))]
	);
}

/// Assemble a function with a single parameter, exercising the parameter header
/// and the `@0` register reference that satisfies the no-gaps rule.
#[test]
fn test_assemble_single_parameter()
{
	let text = "\
Function(x@0) r#1 ⚅#0
\textern[]
\tbody:
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(function.parameters, vec!["x".to_string()]);
	assert_eq!(function.register_count, 1);
}

/// Assemble a function with a single extern, exercising the extern
/// header and the `@0` register reference.
#[test]
fn test_assemble_single_extern()
{
	let text = "\
Function() r#1 ⚅#0
\textern[y@0]
\tbody:
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(function.parameters, Vec::<String>::new());
	assert_eq!(function.externals, vec!["y".to_string()]);
	assert_eq!(function.register_count, 1);
}

/// Assemble parameter and extern names that contain non-ASCII characters,
/// internal spaces, dots, and hyphens — all of which the source grammar permits
/// and [`Function`]'s [`Display`] emits verbatim.
#[test]
fn test_assemble_unicode_and_whitespace_names()
{
	let text = "\
Function(an argument@0, qualified.access@1, kebab-case@2) r#4 ⚅#0
\textern[выражение в кости@3]
\tbody:
\t\t@3 <- @0 + @3
\t\treturn @3
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.parameters,
		vec![
			"an argument".to_string(),
			"qualified.access".to_string(),
			"kebab-case".to_string()
		]
	);
	assert_eq!(function.externals, vec!["выражение в кости".to_string()]);
	assert_eq!(function.register_count, 4);
}

/// Cover every binary arithmetic sigil.
#[test]
fn test_assemble_binary_arithmetic()
{
	for (sigil, constructor) in [
		(
			"+",
			Instruction::add
				as fn(
					RegisterIndex,
					AddressingMode,
					AddressingMode
				) -> Instruction
		),
		("-", Instruction::sub),
		("*", Instruction::mul),
		("/", Instruction::div),
		("%", Instruction::r#mod),
		("^", Instruction::exp)
	]
	{
		let text = format!(
			"\
Function() r#1 ⚅#0
\textern[]
\tbody:
\t\t@0 <- 3 {} 4
\t\treturn @0
",
			sigil
		);
		let function = Assembler::assemble(&text).unwrap();
		assert_eq!(
			function.instructions[0],
			constructor(
				RegisterIndex(0),
				AddressingMode::Immediate(Immediate(3)),
				AddressingMode::Immediate(Immediate(4))
			),
			"binary op {} mis-parsed",
			sigil
		);
	}
}

/// Unary negation, with both an immediate and a register operand.
#[test]
fn test_assemble_unary_negation()
{
	let text = "\
Function(x@0) r#2 ⚅#0
\textern[]
\tbody:
\t\t@1 <- -@0
\t\treturn @1
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::neg(
			RegisterIndex(1),
			AddressingMode::Register(RegisterIndex(0))
		)
	);

	let text = "\
Function() r#1 ⚅#0
\textern[]
\tbody:
\t\t@0 <- -5
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::neg(
			RegisterIndex(0),
			AddressingMode::Immediate(Immediate(5))
		)
	);
}

/// Roll range with both immediate and register bounds.
#[test]
fn test_assemble_roll_range()
{
	let text = "\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll range -10:10
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::roll_range(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(-10)),
			AddressingMode::Immediate(Immediate(10))
		)
	);

	let text = "\
Function(a@0, b@1) r#3 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll range @0:@1
\t\t@2 <- sum rolling record ⚅0
\t\treturn @2
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::roll_range(
			RollingRecordIndex(0),
			AddressingMode::Register(RegisterIndex(0)),
			AddressingMode::Register(RegisterIndex(1))
		)
	);
}

/// Roll standard dice.
#[test]
fn test_assemble_roll_standard_dice()
{
	let text = "\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll standard dice 3D6
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::roll_standard_dice(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(3)),
			AddressingMode::Immediate(Immediate(6))
		)
	);
}

/// Roll custom dice with a one-face list, a multi-face list, and negative
/// faces.
#[test]
fn test_assemble_roll_custom_dice()
{
	let text = "\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll custom dice 1D[-1, 0, 1]
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::roll_custom_dice(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(1)),
			vec![-1, 0, 1]
		)
	);

	let text = "\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll custom dice 2D[42]
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::roll_custom_dice(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(2)),
			vec![42]
		)
	);
}

/// Drop lowest and drop highest, with both immediate and register counts.
#[test]
fn test_assemble_drop_clauses()
{
	let text = "\
Function() r#1 ⚅#1
\textern[]
\tbody:
\t\t⚅0 <- roll standard dice 3D6
\t\t⚅0 <- drop lowest 1 from ⚅0
\t\t⚅0 <- drop highest 1 from ⚅0
\t\t@0 <- sum rolling record ⚅0
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[1],
		Instruction::drop_lowest(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(1))
		)
	);
	assert_eq!(
		function.instructions[2],
		Instruction::drop_highest(
			RollingRecordIndex(0),
			AddressingMode::Immediate(Immediate(1))
		)
	);
}

/// Negative integer operands at every supported slot — regression check for the
/// `@N <- -OP` vs `@N <- OP1 - OP2` tokenization corner.
#[test]
fn test_assemble_negative_operands()
{
	let text = "\
Function() r#1 ⚅#0
\textern[]
\tbody:
\t\t@0 <- -5 + -3
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::add(
			RegisterIndex(0),
			AddressingMode::Immediate(Immediate(-5)),
			AddressingMode::Immediate(Immediate(-3))
		)
	);

	let text = "\
Function() r#1 ⚅#0
\textern[]
\tbody:
\t\t@0 <- 10 - -3
\t\treturn @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::sub(
			RegisterIndex(0),
			AddressingMode::Immediate(Immediate(10)),
			AddressingMode::Immediate(Immediate(-3))
		)
	);
}

/// Permissive horizontal whitespace: accept any run of spaces/tabs where
/// the [`Display`] impl emits a single space or a pair of tabs.
#[test]
fn test_assemble_permissive_whitespace()
{
	let text = "\
Function(x@0,   y@1) r#2 ⚅#0
    extern[]
  body:
          @0 <- @0  +  @1
     return @0
";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(function.parameters, vec!["x".to_string(), "y".to_string()]);
	assert_eq!(
		function.instructions[0],
		Instruction::add(
			RegisterIndex(0),
			AddressingMode::Register(RegisterIndex(0)),
			AddressingMode::Register(RegisterIndex(1))
		)
	);
}

/// Accept `\r\n` line terminators as readily as `\n`.
#[test]
fn test_assemble_crlf_line_endings()
{
	let text =
		"Function() r#0 ⚅#0\r\n\textern[]\r\n\tbody:\r\n\t\treturn 1\r\n";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::r#return(AddressingMode::Immediate(Immediate(1)))
	);
}

/// Accept leading and trailing blank lines around the input.
#[test]
fn test_assemble_surrounding_blank_lines()
{
	let text = "\n\n\
Function() r#0 ⚅#0
\textern[]
\tbody:
\t\treturn 0
\n\n";
	let function = Assembler::assemble(text).unwrap();
	assert_eq!(function.register_count, 0);
}

////////////////////////////////////////////////////////////////////////////////
//                              Negative tests.                               //
////////////////////////////////////////////////////////////////////////////////

/// Assert that [`Assembler::assemble`] rejects `input` and return the resulting
/// [`AssemblyError`] for caller-side inspection of its variant and fields.
///
/// # Parameters
/// - `input`: The malformed input.
///
/// # Returns
/// The rejected [`AssemblyError`].
#[track_caller]
fn assert_rejects(input: &str) -> AssemblyError
{
	match Assembler::assemble(input)
	{
		Ok(function) => panic!(
			"expected rejection but assembled successfully:\n{:#?}",
			function
		),
		Err(e) => e
	}
}

/// The header must begin with the literal `Function(`. Any other prefix yields
/// an [`AssemblyError::Syntax`].
#[test]
fn test_reject_missing_function_keyword()
{
	assert!(matches!(
		assert_rejects(
			"NotFunction() r#0 ⚅#0\n\textern[]\n\tbody:\n\t\treturn 0\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// The register-count field must carry the `r#` prefix. A bare integer is
/// rejected as [`AssemblyError::Syntax`].
#[test]
fn test_reject_missing_register_count_prefix()
{
	assert!(matches!(
		assert_rejects("Function() 0 ⚅#0\n\textern[]\n\tbody:\n\t\treturn 0\n"),
		AssemblyError::Syntax { .. }
	));
}

/// The rolling-record-count field must carry the `⚅#` prefix. A bare integer is
/// rejected as [`AssemblyError::Syntax`].
#[test]
fn test_reject_missing_rolling_record_count_prefix()
{
	assert!(matches!(
		assert_rejects("Function() r#0 0\n\textern[]\n\tbody:\n\t\treturn 0\n"),
		AssemblyError::Syntax { .. }
	));
}

/// The `extern[...]` line is mandatory even when the extern list is empty.
/// Omitting it yields [`AssemblyError::Syntax`].
#[test]
fn test_reject_missing_extern_line()
{
	assert!(matches!(
		assert_rejects("Function() r#0 ⚅#0\n\tbody:\n\t\treturn 0\n"),
		AssemblyError::Syntax { .. }
	));
}

/// The `body:` separator between the extern line and the instruction list is
/// mandatory; omitting it yields [`AssemblyError::Syntax`].
#[test]
fn test_reject_missing_body_header()
{
	assert!(matches!(
		assert_rejects("Function() r#0 ⚅#0\n\textern[]\n\t\treturn 0\n"),
		AssemblyError::Syntax { .. }
	));
}

/// A body instruction must begin with a recognized mnemonic or destination
/// sigil. An unknown leading token (`wibble` here) yields
/// [`AssemblyError::Syntax`].
#[test]
fn test_reject_unknown_opcode()
{
	assert!(matches!(
		assert_rejects(
			"Function() r#1 ⚅#0\n\textern[]\n\tbody:\n\t\t@0 <- wibble 1 + 2\n\t\treturn @0\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// The destination sigil (`@N` vs `⚅N`) and the opcode keyword must agree on
/// the result's kind. `@0 <- roll range …` pairs a register destination with a
/// rolling instruction and is rejected as [`AssemblyError::Syntax`].
#[test]
fn test_reject_wrong_destination_kind()
{
	assert!(matches!(
		assert_rejects(
			"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n\t\t@0 <- roll range 0:5\n\t\treturn @0\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// The grammar forbids `⚅N` as an arithmetic operand. This closes the panic
/// surface in `evaluator::EvaluatorState::value`, where an
/// [`AddressingMode::RollingRecord`](crate::AddressingMode::RollingRecord) in a
/// source slot reaches `unreachable!()`. The input is rejected as
/// [`AssemblyError::Syntax`] rather than reaching the validator's
/// [`AssemblyError::UnexpectedRollingRecordOperand`] branch, because the parser
/// catches it first.
#[test]
fn test_reject_rolling_record_as_arithmetic_operand()
{
	assert!(matches!(
		assert_rejects(
			"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll standard dice 3D6\n\t\t@0 <- ⚅0 + 1\n\t\treturn @0\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// A register reference whose index exceeds the header-declared register count
/// yields [`AssemblyError::RegisterOutOfBounds`], with the offending index and
/// the declared count surfaced on the variant.
#[test]
fn test_reject_register_out_of_bounds()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#0\n\textern[]\n\tbody:\n\t\t@5 <- 1 + 2\n\t\treturn @5\n"
	);
	let AssemblyError::RegisterOutOfBounds {
		index,
		register_count,
		..
	} = e
	else
	{
		panic!("expected RegisterOutOfBounds, got: {:?}", e);
	};
	assert_eq!(index, 5);
	assert_eq!(register_count, 1);
}

/// A rolling record reference whose index exceeds the header-declared rolling
/// record count yields [`AssemblyError::RollingRecordOutOfBounds`], with the
/// offending index and the declared count surfaced on the variant.
#[test]
fn test_reject_rolling_record_out_of_bounds()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅5 <- roll standard dice 3D6\n\t\t@0 <- sum rolling record ⚅5\n\t\treturn @0\n"
	);
	let AssemblyError::RollingRecordOutOfBounds {
		index,
		rolling_record_count,
		..
	} = e
	else
	{
		panic!("expected RollingRecordOutOfBounds, got: {:?}", e);
	};
	assert_eq!(index, 5);
	assert_eq!(rolling_record_count, 1);
}

/// A register declared by the header but never referenced in the body violates
/// the no-gaps rule and yields [`AssemblyError::RegisterGap`]. This input
/// declares `r#3` but references only `@0` and `@2`, leaving `@1` unreferenced.
#[test]
fn test_reject_register_gap()
{
	let e = assert_rejects(
		"Function() r#3 ⚅#0\n\textern[]\n\tbody:\n\t\t@0 <- 1 + 2\n\t\t@2 <- @0 + 3\n\t\treturn @2\n"
	);
	let AssemblyError::RegisterGap {
		index,
		register_count,
		..
	} = e
	else
	{
		panic!("expected RegisterGap, got: {:?}", e);
	};
	assert_eq!(index, 1);
	assert_eq!(register_count, 3);
}

/// A rolling record declared by the header but never referenced in the body
/// violates the no-gaps rule and yields [`AssemblyError::RollingRecordGap`].
/// This input declares `⚅#3` but references only `⚅0` and `⚅2`, leaving `⚅1`
/// unreferenced.
#[test]
fn test_reject_rolling_record_gap()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#3\n\textern[]\n\tbody:\n\t\t⚅0 <- roll standard dice 3D6\n\t\t⚅2 <- roll standard dice 3D6\n\t\t@0 <- sum rolling record ⚅2\n\t\treturn @0\n"
	);
	let AssemblyError::RollingRecordGap {
		index,
		rolling_record_count,
		..
	} = e
	else
	{
		panic!("expected RollingRecordGap, got: {:?}", e);
	};
	assert_eq!(index, 1);
	assert_eq!(rolling_record_count, 3);
}

/// Parameters must be numbered contiguously from `@0`. When a parameter's `@N`
/// suffix disagrees with its list position, the assembler yields
/// [`AssemblyError::NonContiguousParameter`] with the offending name, claimed
/// index, and actual position surfaced on the variant. Here parameter `y`
/// claims `@2` but appears at position `1`.
#[test]
fn test_reject_non_contiguous_parameter_indices()
{
	let e = assert_rejects(
		"Function(x@0, y@2) r#2 ⚅#0\n\textern[]\n\tbody:\n\t\t@0 <- @0 + @1\n\t\treturn @0\n"
	);
	let AssemblyError::NonContiguousParameter {
		name,
		index,
		position,
		..
	} = e
	else
	{
		panic!("expected NonContiguousParameter, got: {:?}", e);
	};
	assert_eq!(name, "y");
	assert_eq!(index, 2);
	assert_eq!(position, 1);
}

/// Externals must be numbered contiguously from `@parameters.len()`. When an
/// external's `@N` suffix disagrees with its expected position, the assembler
/// yields [`AssemblyError::NonContiguousExternal`] with the offending name,
/// claimed index, expected index, and arity surfaced on the variant. Here
/// extern `b` claims `@5` but should be `@2` (arity `1` plus list position
/// `1`).
#[test]
fn test_reject_non_contiguous_extern_indices()
{
	let e = assert_rejects(
		"Function(x@0) r#3 ⚅#0\n\textern[a@1, b@5]\n\tbody:\n\t\t@2 <- @1 + @0\n\t\treturn @2\n"
	);
	let AssemblyError::NonContiguousExternal {
		name,
		index,
		expected,
		arity,
		..
	} = e
	else
	{
		panic!("expected NonContiguousExternal, got: {:?}", e);
	};
	assert_eq!(name, "b");
	assert_eq!(index, 5);
	assert_eq!(expected, 2);
	assert_eq!(arity, 1);
}

/// The first parameter claiming any suffix other than `@0` also trips the
/// contiguity rule, yielding [`AssemblyError::NonContiguousParameter`]. This
/// complements [`test_reject_non_contiguous_parameter_indices`] by exercising
/// the boundary case at position `0`.
#[test]
fn test_reject_parameter_index_disagrees_with_position()
{
	let e = assert_rejects(
		"Function(x@1) r#1 ⚅#0\n\textern[]\n\tbody:\n\t\treturn @0\n"
	);
	let AssemblyError::NonContiguousParameter {
		name,
		index,
		position,
		..
	} = e
	else
	{
		panic!("expected NonContiguousParameter, got: {:?}", e);
	};
	assert_eq!(name, "x");
	assert_eq!(index, 1);
	assert_eq!(position, 0);
}

/// Custom dice must carry at least one face. The parser rejects `D[]` via
/// `separated_list1` before the validator's
/// [`AssemblyError::FacelessCustomDice`] branch is reached, so the surfaced
/// variant is [`AssemblyError::Syntax`]. Both paths converge on the same safety
/// invariant (no faceless custom dice reach the evaluator).
#[test]
fn test_reject_faceless_custom_dice()
{
	assert!(matches!(
		assert_rejects(
			"Function() r#1 ⚅#1\n\textern[]\n\tbody:\n\t\t⚅0 <- roll custom dice 1D[]\n\t\t@0 <- sum rolling record ⚅0\n\t\treturn @0\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// A `drop lowest` instruction whose destination disagrees with its `from ⚅N`
/// source yields [`AssemblyError::DropSourceMismatch`] with
/// [`DropKind::Lowest`] and the mismatched indices surfaced on the variant.
/// Here the destination is `⚅0` but the source is `⚅1`.
#[test]
fn test_reject_drop_source_disagrees_with_destination()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#2\n\textern[]\n\tbody:\n\t\t⚅0 <- roll standard dice 3D6\n\t\t⚅1 <- roll standard dice 3D6\n\t\t⚅0 <- drop lowest 1 from ⚅1\n\t\t@0 <- sum rolling record ⚅0\n\t\treturn @0\n"
	);
	let AssemblyError::DropSourceMismatch {
		kind,
		destination,
		source,
		..
	} = e
	else
	{
		panic!("expected DropSourceMismatch, got: {:?}", e);
	};
	assert_eq!(kind, DropKind::Lowest);
	assert_eq!(destination, 0);
	assert_eq!(source, 1);
}

/// The `drop highest` counterpart to
/// [`test_reject_drop_source_disagrees_with_destination`], confirming that
/// [`DropKind::Highest`] is surfaced on [`AssemblyError::DropSourceMismatch`]
/// for the other direction.
#[test]
fn test_reject_drop_highest_source_disagrees_with_destination()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#2\n\textern[]\n\tbody:\n\t\t⚅0 <- roll standard dice 3D6\n\t\t⚅1 <- roll standard dice 3D6\n\t\t⚅0 <- drop highest 1 from ⚅1\n\t\t@0 <- sum rolling record ⚅0\n\t\treturn @0\n"
	);
	let AssemblyError::DropSourceMismatch { kind, .. } = e
	else
	{
		panic!("expected DropSourceMismatch, got: {:?}", e);
	};
	assert_eq!(kind, DropKind::Highest);
}

/// Non-whitespace content following the final instruction yields
/// [`AssemblyError::Syntax`]. The trailing-garbage check catches stray tokens
/// that might otherwise be silently discarded, which would hide regressions in
/// callers that format [`Function`]s back into text.
#[test]
fn test_reject_trailing_garbage()
{
	assert!(matches!(
		assert_rejects(
			"Function() r#0 ⚅#0\n\textern[]\n\tbody:\n\t\treturn 0\nxxx\n"
		),
		AssemblyError::Syntax { .. }
	));
}

/// When the header-declared register count cannot accommodate the parameter and
/// extern slots, the assembler yields
/// [`AssemblyError::InsufficientRegisterCount`] with the declared count and the
/// required count (`parameters.len() + externals.len()`) surfaced on the
/// variant. Here two parameters demand at least `r#2` but the header declares
/// only `r#1`.
#[test]
fn test_reject_declared_args_exceed_register_count()
{
	let e = assert_rejects(
		"Function(x@0, y@1) r#1 ⚅#0\n\textern[]\n\tbody:\n\t\treturn @0\n"
	);
	let AssemblyError::InsufficientRegisterCount {
		register_count,
		required,
		..
	} = e
	else
	{
		panic!("expected InsufficientRegisterCount, got: {:?}", e);
	};
	assert_eq!(register_count, 1);
	assert_eq!(required, 2);
}

/// Location accessor surfaces a line number matching the offending instruction,
/// not the header.
#[test]
fn test_error_location_points_at_offending_line()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#0\n\textern[]\n\tbody:\n\t\t@9 <- 1 + 2\n\t\treturn @9\n"
	);
	let location = e.location();
	assert_eq!(
		location.line, 4,
		"expected line 4 (body instruction), got {}",
		location.line
	);
	assert!(matches!(e, AssemblyError::RegisterOutOfBounds { .. }));
}

/// The [`Display`](std::fmt::Display) impl renders a human-readable message
/// that includes the line, column, byte offset, and a variant-specific
/// description.
#[test]
fn test_display_renders_location_and_description()
{
	let e = assert_rejects(
		"Function() r#1 ⚅#0\n\textern[]\n\tbody:\n\t\t@5 <- 1 + 2\n\t\treturn @5\n"
	);
	let rendered = format!("{}", e);
	assert!(
		rendered.contains("line 4"),
		"missing line number: {}",
		rendered
	);
	assert!(
		rendered.contains("register @5"),
		"missing register token: {}",
		rendered
	);
	assert!(
		rendered.contains("exceeds declared register count r#1"),
		"missing descriptive tail: {}",
		rendered
	);
}

////////////////////////////////////////////////////////////////////////////////
//                               FromStr tests.                               //
////////////////////////////////////////////////////////////////////////////////

/// [`FromStr`](std::str::FromStr) for [`Function`] is a thin delegate to
/// [`Assembler::assemble`], so `parse::<Function>()` is available as an
/// ergonomic alternative at every call site that already takes a `&str`. The
/// happy path produces the same [`Function`] as a direct call through
/// [`Assembler`].
#[test]
fn test_from_str_delegates_to_assembler()
{
	let text = "\
Function() r#0 ⚅#0
\textern[]
\tbody:
\t\treturn 42
";
	let function: Function = text.parse().unwrap();
	assert_eq!(
		function.instructions[0],
		Instruction::r#return(AddressingMode::Immediate(Immediate(42)))
	);
}

/// The [`FromStr`](std::str::FromStr) path surfaces [`AssemblyError`]s with the
/// same fidelity as a direct [`Assembler::assemble`] call; malformed input
/// reaches the caller as an `Err`, not as a panic or a silently-empty
/// [`Function`].
#[test]
fn test_from_str_propagates_errors()
{
	let result: Result<Function, _> = "not a function".parse();
	assert!(result.is_err());
}
