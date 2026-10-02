//! # Function tests
//!
//! Herein are tests for the [validation](crate::Function::validate) of a
//! [`Function`], which both the [`Assembler`] and deserialization rely upon,
//! and for the round trip of a [`Function`] through `serde`. The core of the
//! suite is a table of functions, each of which breaks exactly one invariant of
//! a well-formed base function, paired with the [`FunctionError`] that it
//! earns.

use pretty_assertions::assert_eq;

use crate::{
	AddressingMode, Assembler, Function, FunctionError, Instruction,
	RegisterIndex, RollingRecordIndex, compile,
	support::{compile_valid, read_compilation_test_cases}
};

////////////////////////////////////////////////////////////////////////////////
//                                 Fixtures.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Answer a well-formed function, with a parameter, an external variable, and
/// one instruction of each shape that the table of [malformed] functions
/// needs to break.
///
/// # Returns
/// The function.
fn well_formed() -> Function
{
	Assembler::assemble(
		"\
Function({x}@0) r#3 ⚅#1
\textern[{y}@1]
\tbody:
\t\t⚅0 <- roll custom dice @0D[1, 2]
\t\t@2 <- sum rolling record ⚅0
\t\t@2 <- @2 + @1
\t\treturn @2
"
	)
	.unwrap()
}

/// A way to break exactly one invariant of the [well-formed](well_formed)
/// function.
type Breakage = fn(&mut Function);

/// Answer functions that each break exactly one invariant of the
/// [well-formed](well_formed) function, beside the error that each earns and
/// its rendering.
///
/// # Returns
/// The malformed functions, their errors, and the renderings of their errors.
fn malformed() -> Vec<(Function, FunctionError, &'static str)>
{
	let cases: Vec<(Breakage, FunctionError, &'static str)> = vec![
		(
			|f| f.parameters[0] = "x ".to_string(),
			FunctionError::NonCanonicalName {
				name: "x ".to_string(),
				index: 0
			},
			"variable @0 is named `x `, which is not canonical"
		),
		(
			|f| f.externals[0] = "x".to_string(),
			FunctionError::DuplicateName {
				name: "x".to_string(),
				first: 0,
				index: 1
			},
			"variables @0 and @1 are both named `x` (names must be distinct)"
		),
		(
			|f| f.register_count = 1,
			FunctionError::InsufficientRegisterCount {
				register_count: 1,
				required: 2
			},
			"r#1 registers cannot hold the 2 parameters and external variables"
		),
		(
			|f| {
				f.instructions[2] = Instruction::add(
					RegisterIndex(2),
					AddressingMode::Register(RegisterIndex(5)),
					AddressingMode::Register(RegisterIndex(1))
				)
			},
			FunctionError::RegisterOutOfBounds {
				index: 5,
				register_count: 3,
				instruction: 2
			},
			"instruction 2: register @5 exceeds declared register count r#3"
		),
		(
			|f| {
				f.instructions[1] = Instruction::sum_rolling_record(
					RegisterIndex(2),
					RollingRecordIndex(4)
				)
			},
			FunctionError::RollingRecordOutOfBounds {
				index: 4,
				rolling_record_count: 1,
				instruction: 1
			},
			"instruction 1: rolling record ⚅4 exceeds declared rolling record \
			 count ⚅#1"
		),
		(
			|f| {
				f.instructions[0] = Instruction::roll_custom_dice(
					RollingRecordIndex(0),
					AddressingMode::Register(RegisterIndex(0)),
					vec![]
				)
			},
			FunctionError::FacelessCustomDice { instruction: 0 },
			"instruction 0: custom dice must have at least one face"
		),
		(
			|f| {
				f.instructions[2] = Instruction::add(
					RegisterIndex(2),
					AddressingMode::RollingRecord(RollingRecordIndex(0)),
					AddressingMode::Register(RegisterIndex(1))
				)
			},
			FunctionError::UnexpectedRollingRecordOperand { instruction: 2 },
			"instruction 2: rolling record operand is not permitted here"
		),
		(
			|f| {
				f.instructions.pop();
			},
			FunctionError::MissingReturn,
			"the function has no return"
		),
		(
			|f| f.instructions.clear(),
			FunctionError::MissingReturn,
			"the function has no return"
		),
		(
			|f| {
				f.instructions.push(Instruction::add(
					RegisterIndex(2),
					AddressingMode::Register(RegisterIndex(2)),
					AddressingMode::Register(RegisterIndex(1))
				))
			},
			FunctionError::EarlyReturn { instruction: 3 },
			"instruction 3: return is not the last instruction (a function \
			 ends with its only return)"
		),
		(
			|f| {
				f.instructions.insert(
					2,
					Instruction::r#return(AddressingMode::Register(
						RegisterIndex(2)
					))
				)
			},
			FunctionError::EarlyReturn { instruction: 2 },
			"instruction 2: return is not the last instruction (a function \
			 ends with its only return)"
		),
		(
			|f| f.register_count = 4,
			FunctionError::RegisterGap {
				index: 3,
				register_count: 4
			},
			"register @3 is declared by r#4 but is never referenced (no gaps \
			 are permitted in the register file)"
		),
		(
			|f| f.rolling_record_count = 2,
			FunctionError::RollingRecordGap {
				index: 1,
				rolling_record_count: 2
			},
			"rolling record ⚅1 is declared by ⚅#2 but is never referenced (no \
			 gaps are permitted in the rolling record file)"
		),
	];
	cases
		.into_iter()
		.map(|(breaks, error, rendering)| {
			let mut function = well_formed();
			breaks(&mut function);
			(function, error, rendering)
		})
		.collect()
}

////////////////////////////////////////////////////////////////////////////////
//                                Validation.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Test that every function that the compiler makes, with or without
/// optimization, is well formed.
#[test]
fn test_compiled_functions_are_well_formed()
{
	for (source, _) in read_compilation_test_cases(include_str!(
		"../../tests/test_full_optimization.txt"
	))
	{
		assert_eq!(compile_valid(source).validate(), Ok(()), "{}", source);
		assert_eq!(compile(source).unwrap().validate(), Ok(()), "{}", source);
	}
}

/// Test that [`Function::validate`] rejects each malformed function with the
/// expected error, which renders as expected and answers the expected
/// offending instruction.
#[test]
fn test_validate_rejects_malformed_functions()
{
	assert_eq!(well_formed().validate(), Ok(()));
	for (function, error, rendering) in malformed()
	{
		let actual = function.validate().unwrap_err();
		assert_eq!(actual, error, "{}", function);
		assert_eq!(actual.to_string(), rendering, "{}", function);
		let instruction = match error
		{
			FunctionError::RegisterOutOfBounds { instruction, .. }
			| FunctionError::RollingRecordOutOfBounds { instruction, .. }
			| FunctionError::FacelessCustomDice { instruction }
			| FunctionError::UnexpectedRollingRecordOperand { instruction }
			| FunctionError::EarlyReturn { instruction } => Some(instruction),
			_ => None
		};
		assert_eq!(actual.instruction(), instruction, "{}", function);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// Test that every function that the compiler makes, with or without
/// optimization, survives the round trip through `serde`.
#[cfg(feature = "serde")]
#[test]
fn test_serde_round_trip()
{
	for (source, _) in read_compilation_test_cases(include_str!(
		"../../tests/test_full_optimization.txt"
	))
	{
		for function in [compile_valid(source), compile(source).unwrap()]
		{
			let json = serde_json::to_string(&function).unwrap();
			let actual: Function = serde_json::from_str(&json).unwrap();
			assert_eq!(actual, function, "{}", source);
		}
	}
}

/// Test that deserialization refuses each malformed function, reporting the
/// error that [`Function::validate`] finds.
#[cfg(feature = "serde")]
#[test]
fn test_serde_rejects_malformed_functions()
{
	for (function, _, rendering) in malformed()
	{
		let json = serde_json::to_string(&function).unwrap();
		let error = serde_json::from_str::<Function>(&json).unwrap_err();
		assert!(
			error.to_string().contains(rendering),
			"{}: {}",
			function,
			error
		);
	}
}

/// Test that deserialization refuses a function with a field that
/// [`Function`] does not have, rather than silently dropping it.
#[cfg(feature = "serde")]
#[test]
fn test_serde_rejects_unknown_fields()
{
	let mut json = serde_json::to_value(well_formed()).unwrap();
	json.as_object_mut()
		.unwrap()
		.insert("answer".to_string(), 42.into());
	let error = serde_json::from_value::<Function>(json).unwrap_err();
	assert!(error.to_string().contains("answer"), "{}", error);
}
