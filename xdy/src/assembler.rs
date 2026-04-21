//! # Assembler
//!
//! The assembler is the inverse of [`Function`]'s [`Display`] impl: it reads
//! the print representation of a [`Function`] back into a [`Function`] struct.
//! This completes the round trip that the text corpus in
//! `tests/test_full_optimization.txt` was already assuming existed in one
//! direction; with the assembler in place, any regression in either direction
//! can be caught by round-trip tests over the corpus.
//!
//! The assembler targets the [`Evaluator`](crate::Evaluator) as its downstream
//! consumer. Index-bounds and register-file contiguity are enforced so that the
//! evaluator's unchecked vector indexing cannot panic on an assembled
//! [`Function`]. Hand-authored IR that would survive the assembler but
//! nonetheless trip the [`StandardOptimizer`](crate::StandardOptimizer) (which
//! has internal invariants the compiler establishes — e.g., a trailing
//! [`Return`](crate::Return) instruction, single-assignment forms in some
//! passes) is out of scope.
//!
//! # Grammar
//!
//! The recognized grammar is the following, in Extended Backus-Naur Form
//! (EBNF), where non-terminals are in lowercase and terminals are in uppercase:
//!
//! ```text
//! function          ::= header extern_line body_header instruction_line*
//! header            ::= "Function(" parameters? ") r#" INDEX " ⚅#" INDEX EOL
//! parameters        ::= named_reg ("," named_reg)*
//! extern_line       ::= "extern[" externals? "]" EOL
//! externals         ::= named_reg ("," named_reg)*
//! named_reg         ::= NAME "@" INDEX
//! body_header       ::= "body:" EOL
//! instruction_line  ::= instruction EOL
//! instruction       ::= roll_range | roll_standard_dice | roll_custom_dice
//!                     | drop_lowest | drop_highest | sum_rolling_record
//!                     | binary_op | unary_neg | return_inst
//! roll_range        ::= "⚅" INDEX "<-" "roll range" op ":" op
//! roll_standard_dice::= "⚅" INDEX "<-" "roll standard dice" op "D" op
//! roll_custom_dice  ::= "⚅" INDEX "<-" "roll custom dice" op "D[" INTEGER
//!                         ("," INTEGER)* "]"
//! drop_lowest       ::= "⚅" INDEX "<-" "drop lowest" op "from" "⚅" INDEX
//! drop_highest      ::= "⚅" INDEX "<-" "drop highest" op "from" "⚅" INDEX
//! sum_rolling_record::= "@" INDEX "<-" "sum rolling record" "⚅" INDEX
//! binary_op         ::= "@" INDEX "<-" op BINARY_OP op
//! unary_neg         ::= "@" INDEX "<-" "-" op
//! return_inst       ::= "return" op
//! op                ::= INTEGER | "@" INDEX
//! NAME              ::= any run of characters other than "@", ",", ")",
//!                         "]", "\t", "\r", "\n"; trimmed of leading and
//!                         trailing ASCII whitespace
//! INDEX             ::= DIGIT+
//! INTEGER           ::= "-"? DIGIT+
//! BINARY_OP         ::= "+" | "-" | "*" | "/" | "%" | "^"
//! EOL               ::= "\n" | "\r\n"
//! ```
//!
//! Whitespace (horizontal: space and tab) between tokens is permissive — any
//! run of spaces and tabs is accepted wherever [`Function`]'s [`Display`] emits
//! indentation or a separator. The structural tokens (`:`, `<-`, `D`, `r#`,
//! `⚅#`, the digraph `[-`, and so on) are matched literally.
//!
//! Rolling records (`⚅N`) are syntactically restricted to the slots where the
//! intermediate representation expects them: the destination of every
//! rolling instruction, the source of the `from` clause on the drop
//! instructions, and the source of `sum rolling record`. They are not
//! permitted anywhere else — in particular, not as operands of arithmetic,
//! roll-count, drop-count, or return instructions — because
//! [`Evaluator`](crate::Evaluator)'s operand-dispatch would otherwise
//! unreachable-panic on an unmapped rolling record read.
//!
//! The following railroad diagram is generated from the EBNF grammar above:
#![doc = include_str!("../doc/xdy_asm.svg")]
//! # Examples
//!
//! Assemble a trivial function that returns a constant:
//!
//! ```rust
//! use xdy::{Assembler, Function};
//!
//! let text = "\
//! Function() r#0 ⚅#0
//! \textern[]
//! \tbody:
//! \t\treturn 42
//! ";
//! let function = Assembler::assemble(text).unwrap();
//! assert_eq!(function.arity(), 0);
//! assert_eq!(function.register_count, 0);
//! assert_eq!(format!("{}", function).trim(), text.trim());
//! ```
//!
//! Round-trip [`FromStr`](std::str::FromStr) via [`parse`](str::parse):
//!
//! ```rust
//! use xdy::Function;
//!
//! let text = "\
//! Function(x@0) r#1 ⚅#0
//! \textern[]
//! \tbody:
//! \t\treturn @0
//! ";
//! let function: Function = text.parse().unwrap();
//! assert_eq!(function.parameters, vec!["x".to_string()]);
//! ```

use std::{
	error::Error,
	fmt::{self, Display, Formatter},
	str::FromStr
};

use nom::{
	IResult, Parser,
	branch::alt,
	bytes::complete::{tag, take_while, take_while1},
	character::complete::{char, digit1, line_ending, one_of},
	combinator::{eof, map, opt, recognize, value},
	error::Error as NomError,
	multi::{many0, separated_list0, separated_list1},
	sequence::{delimited, pair, terminated}
};
use nom_locate::LocatedSpan;

use crate::{
	AddressingMode, Function, Immediate, Instruction, RegisterIndex,
	RollingRecordIndex
};

////////////////////////////////////////////////////////////////////////////////
//                                 Assembler.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The `xDy` intermediate-representation assembler. Use [`Assembler::assemble`]
/// as the high-level entry point; the assembler holds no persistent state and
/// does not need to be constructed. For ergonomic access, [`Function`]
/// implements [`FromStr`] with the same semantics.
///
/// # Examples
///
/// ```rust
/// use xdy::Assembler;
///
/// let text = "\
/// Function() r#1 ⚅#1
/// \textern[]
/// \tbody:
/// \t\t⚅0 <- roll standard dice 3D6
/// \t\t@0 <- sum rolling record ⚅0
/// \t\treturn @0
/// ";
/// let function = Assembler::assemble(text).unwrap();
/// assert_eq!(function.register_count, 1);
/// assert_eq!(function.rolling_record_count, 1);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct Assembler;

impl Assembler
{
	/// Assemble a [`Function`] from its print representation. The input must
	/// match the format produced by [`Function`]'s [`Display`] impl, modulo
	/// permissive horizontal whitespace between tokens. Leading and trailing
	/// whitespace around the whole input are tolerated.
	///
	/// # Parameters
	/// - `input`: The text form of a compiled function.
	///
	/// # Returns
	/// The assembled [`Function`].
	///
	/// # Errors
	/// [`AssemblyError`] if the input is syntactically malformed or if the
	/// assembled form would violate any of the safety invariants the assembler
	/// enforces: contiguous numbering of the register and rolling record files,
	/// agreement between header counts and body references, non-empty custom
	/// dice face lists, and so on.
	pub fn assemble(input: &str) -> Result<Function, AssemblyError>
	{
		let raw = Self::parse(input)?;
		Self::validate(raw)
	}

	/// Parse the input into a [`Raw`] assemblage, without applying any
	/// semantic validation. Separated from [`Self::validate`] so that the
	/// syntactic parser can be exercised in isolation by the tests and so
	/// that validation can report errors against the byte offsets collected
	/// during parsing.
	///
	/// # Parameters
	/// - `input`: The text form of a compiled function.
	///
	/// # Returns
	/// The unvalidated [`Raw`] assemblage.
	///
	/// # Errors
	/// [`AssemblyError`] if the input is syntactically malformed.
	fn parse(input: &str) -> Result<Raw, AssemblyError>
	{
		let span = Span::new(input);
		let (_, raw) = delimited(skip_ws_and_newlines, function, trailing)
			.parse_complete(span)
			.map_err(|e| match e
			{
				nom::Err::Error(e) | nom::Err::Failure(e) =>
				{
					AssemblyError::from_nom(e)
				},
				nom::Err::Incomplete(_) => unreachable!()
			})?;
		Ok(raw)
	}

	/// Validate a [`Raw`] assemblage and convert it into a [`Function`]. The
	/// validation enforces every invariant that the downstream
	/// [`Evaluator`](crate::Evaluator) relies on to avoid panicking:
	///
	/// 1. Parameter indices are exactly `0..parameters.len()` in order.
	/// 2. External-variable indices are exactly
	///    `parameters.len()..parameters.len() + externals.len()` in order.
	/// 3. Every register reference in the body satisfies `index <
	///    register_count`, and every rolling record reference satisfies `index
	///    < rolling_record_count`.
	/// 4. The register file has no gaps: every index in `0..register_count`
	///    appears at least once as a parameter slot, an extern slot, or a body
	///    reference (destination or source).
	/// 5. The rolling record file has no gaps: every index in
	///    `0..rolling_record_count` appears at least once as a destination or
	///    source.
	/// 6. Every [`RollCustomDice`](crate::RollCustomDice) carries a non-empty
	///    face list.
	/// 7. Every [`DropLowest`](crate::DropLowest) and
	///    [`DropHighest`](crate::DropHighest) names the same rolling record as
	///    its destination and its `from` clause.
	///
	/// # Parameters
	/// - `raw`: The unvalidated [`Raw`] assemblage.
	///
	/// # Returns
	/// The validated [`Function`].
	///
	/// # Errors
	/// An [`AssemblyError`] variant describing the specific invariant that
	/// the input violated, carrying the [`AssemblyLocation`] of the
	/// offending token.
	fn validate(raw: Raw) -> Result<Function, AssemblyError>
	{
		// Parameters occupy @0..@P in order.
		for (position, named) in raw.parameters.iter().enumerate()
		{
			if named.index != position
			{
				return Err(AssemblyError::NonContiguousParameter {
					name: named.name.clone(),
					index: named.index,
					position,
					location: named.location
				})
			}
		}
		let arity = raw.parameters.len();
		// Externals occupy @P..@P+E in order.
		for (position, named) in raw.externals.iter().enumerate()
		{
			let expected = arity + position;
			if named.index != expected
			{
				return Err(AssemblyError::NonContiguousExternal {
					name: named.name.clone(),
					index: named.index,
					expected,
					arity,
					location: named.location
				})
			}
		}
		let declared_args = arity + raw.externals.len();
		if declared_args > raw.register_count
		{
			return Err(AssemblyError::InsufficientRegisterCount {
				register_count: raw.register_count,
				required: declared_args,
				location: raw.header_location
			})
		}
		// Collect referenced register and rolling record indices; fault on any
		// out-of-bounds index.
		let mut register_seen = vec![false; raw.register_count];
		register_seen
			.iter_mut()
			.take(declared_args)
			.for_each(|slot| *slot = true);
		let mut record_seen = vec![false; raw.rolling_record_count];
		for raw_inst in &raw.instructions
		{
			let check_register = |idx: RegisterIndex,
			                      seen: &mut [bool]|
			 -> Result<(), AssemblyError> {
				if idx.0 >= raw.register_count
				{
					return Err(AssemblyError::RegisterOutOfBounds {
						index: idx.0,
						register_count: raw.register_count,
						location: raw_inst.location
					})
				}
				seen[idx.0] = true;
				Ok(())
			};
			let check_record = |idx: RollingRecordIndex,
			                    seen: &mut [bool]|
			 -> Result<(), AssemblyError> {
				if idx.0 >= raw.rolling_record_count
				{
					return Err(AssemblyError::RollingRecordOutOfBounds {
						index: idx.0,
						rolling_record_count: raw.rolling_record_count,
						location: raw_inst.location
					})
				}
				seen[idx.0] = true;
				Ok(())
			};
			let check_mode = |mode: AddressingMode,
			                  regs: &mut [bool]|
			 -> Result<(), AssemblyError> {
				match mode
				{
					AddressingMode::Immediate(_) => Ok(()),
					AddressingMode::Register(reg) =>
					{
						if reg.0 >= raw.register_count
						{
							return Err(AssemblyError::RegisterOutOfBounds {
								index: reg.0,
								register_count: raw.register_count,
								location: raw_inst.location
							})
						}
						regs[reg.0] = true;
						Ok(())
					},
					AddressingMode::RollingRecord(_) =>
					{
						// The parser rejects rolling records in these
						// slots grammatically, so this branch is
						// defensive. A [`Raw`] produced by
						// [`Self::parse`] cannot reach it; the variant
						// exists purely to keep the validator total.
						Err(AssemblyError::UnexpectedRollingRecordOperand {
							location: raw_inst.location
						})
					}
				}
			};
			match &raw_inst.instruction
			{
				Instruction::RollRange(inst) =>
				{
					check_record(inst.dest, &mut record_seen)?;
					check_mode(inst.start, &mut register_seen)?;
					check_mode(inst.end, &mut register_seen)?;
				},
				Instruction::RollStandardDice(inst) =>
				{
					check_record(inst.dest, &mut record_seen)?;
					check_mode(inst.count, &mut register_seen)?;
					check_mode(inst.faces, &mut register_seen)?;
				},
				Instruction::RollCustomDice(inst) =>
				{
					check_record(inst.dest, &mut record_seen)?;
					check_mode(inst.count, &mut register_seen)?;
					if inst.faces.is_empty()
					{
						return Err(AssemblyError::FacelessCustomDice {
							location: raw_inst.location
						})
					}
				},
				Instruction::DropLowest(inst) =>
				{
					check_record(inst.dest, &mut record_seen)?;
					check_mode(inst.count, &mut register_seen)?;
				},
				Instruction::DropHighest(inst) =>
				{
					check_record(inst.dest, &mut record_seen)?;
					check_mode(inst.count, &mut register_seen)?;
				},
				Instruction::SumRollingRecord(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_record(inst.src, &mut record_seen)?;
				},
				Instruction::Add(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Sub(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Mul(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Div(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Mod(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Exp(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op1, &mut register_seen)?;
					check_mode(inst.op2, &mut register_seen)?;
				},
				Instruction::Neg(inst) =>
				{
					check_register(inst.dest, &mut register_seen)?;
					check_mode(inst.op, &mut register_seen)?;
				},
				Instruction::Return(inst) =>
				{
					check_mode(inst.src, &mut register_seen)?;
				}
			}
		}
		// Drop instructions must read back from their own destination. The
		// parser accepts two independent rolling record indices; the
		// validator confirms they agree.
		for raw_inst in &raw.instructions
		{
			match (&raw_inst.instruction, raw_inst.drop_source)
			{
				(Instruction::DropLowest(inst), Some(src))
					if inst.dest != src =>
				{
					return Err(AssemblyError::DropSourceMismatch {
						kind: DropKind::Lowest,
						destination: inst.dest.0,
						source: src.0,
						location: raw_inst.location
					})
				},
				(Instruction::DropHighest(inst), Some(src))
					if inst.dest != src =>
				{
					return Err(AssemblyError::DropSourceMismatch {
						kind: DropKind::Highest,
						destination: inst.dest.0,
						source: src.0,
						location: raw_inst.location
					})
				},
				_ =>
				{}
			}
		}
		// No gaps in the register file.
		if let Some(gap) = register_seen.iter().position(|seen| !*seen)
		{
			return Err(AssemblyError::RegisterGap {
				index: gap,
				register_count: raw.register_count,
				location: raw.header_location
			})
		}
		// No gaps in the rolling record file.
		if let Some(gap) = record_seen.iter().position(|seen| !*seen)
		{
			return Err(AssemblyError::RollingRecordGap {
				index: gap,
				rolling_record_count: raw.rolling_record_count,
				location: raw.header_location
			})
		}
		Ok(Function {
			parameters: raw.parameters.into_iter().map(|n| n.name).collect(),
			externals: raw.externals.into_iter().map(|n| n.name).collect(),
			register_count: raw.register_count,
			rolling_record_count: raw.rolling_record_count,
			instructions: raw
				.instructions
				.into_iter()
				.map(|r| r.instruction)
				.collect()
		})
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Assembly errors.                              //
////////////////////////////////////////////////////////////////////////////////

/// An error produced by the [`Assembler`]. Each variant identifies a specific
/// failure mode — syntactic or semantic — so callers can mechanically
/// distinguish cases without parsing human-readable messages. Every variant
/// carries an [`AssemblyLocation`] locating the offending token within the
/// input.
///
/// The assembler targets the [`Evaluator`](crate::Evaluator) as its downstream
/// consumer, so most variants correspond to invariants that, if violated, would
/// otherwise lead to evaluator panics: index out of bounds, gaps in the
/// register or rolling record file, custom dice with no faces, and so on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssemblyError
{
	/// The input was syntactically malformed. The `description` is a short,
	/// human-readable hint derived from the `nom` combinator that rejected the
	/// input. Callers interested only in "is this a parse failure vs. a
	/// validation failure?" should match this variant.
	Syntax
	{
		/// A short, human-readable hint describing the syntactic failure.
		description: String,

		/// The source position at which the parse was rejected.
		location: AssemblyLocation
	},

	/// A formal parameter's `@N` suffix disagrees with its position in the
	/// parameter list. Parameters must be numbered contiguously from `@0`.
	NonContiguousParameter
	{
		/// The offending parameter's name.
		name: String,

		/// The `@N` suffix asserted by the input.
		index: usize,

		/// The 0-based position at which the parameter appears.
		position: usize,

		/// The source position of the parameter.
		location: AssemblyLocation
	},

	/// An external variable's `@N` suffix disagrees with its expected position.
	/// Externals must be numbered contiguously from `@parameters.len()`.
	NonContiguousExternal
	{
		/// The offending external's name.
		name: String,

		/// The `@N` suffix asserted by the input.
		index: usize,

		/// The expected index (`arity + position`).
		expected: usize,

		/// The arity of the enclosing function.
		arity: usize,

		/// The source position of the external.
		location: AssemblyLocation
	},

	/// The header's declared register count is too small to accommodate the
	/// parameter and extern slots.
	InsufficientRegisterCount
	{
		/// The declared register count (`r#N`).
		register_count: usize,

		/// The number of registers required by the parameter and extern slots
		/// (`parameters.len() + externals.len()`).
		required: usize,

		/// The source position of the header.
		location: AssemblyLocation
	},

	/// A body reference to a register index exceeds the declared register
	/// count.
	RegisterOutOfBounds
	{
		/// The offending register index.
		index: usize,

		/// The declared register count.
		register_count: usize,

		/// The source position of the offending instruction.
		location: AssemblyLocation
	},

	/// A body reference to a rolling record index exceeds the declared rolling
	/// record count.
	RollingRecordOutOfBounds
	{
		/// The offending rolling record index.
		index: usize,

		/// The declared rolling record count.
		rolling_record_count: usize,

		/// The source position of the offending instruction.
		location: AssemblyLocation
	},

	/// A register declared by the header is never referenced by any parameter
	/// slot, extern slot, or body instruction — a gap in the register file.
	RegisterGap
	{
		/// The index of the unreferenced register.
		index: usize,

		/// The declared register count.
		register_count: usize,

		/// The source position of the header.
		location: AssemblyLocation
	},

	/// A rolling record declared by the header is never referenced by any body
	/// instruction — a gap in the rolling record file.
	RollingRecordGap
	{
		/// The index of the unreferenced rolling record.
		index: usize,

		/// The declared rolling record count.
		rolling_record_count: usize,

		/// The source position of the header.
		location: AssemblyLocation
	},

	/// A custom dice instruction has an empty face list. Although the primitive
	/// `roll_custom_dice` defensively treats empty faces as `0` per die, the
	/// source grammar forbids faceless dice and the assembler preserves that
	/// contract.
	FacelessCustomDice
	{
		/// The source position of the offending instruction.
		location: AssemblyLocation
	},

	/// A drop instruction's destination disagrees with its `from ⚅N` source.
	/// [`Function`]'s [`Display`] impl always emits the same rolling record on
	/// both sides; the assembler enforces the invariant so hand-authored IR
	/// cannot desynchronize.
	DropSourceMismatch
	{
		/// Whether the mismatch is in a `drop lowest` or `drop highest`
		/// instruction.
		kind: DropKind,

		/// The destination rolling record index.
		destination: usize,

		/// The `from` source rolling record index.
		source: usize,

		/// The source position of the offending instruction.
		location: AssemblyLocation
	},

	/// The validator encountered an [`AddressingMode::RollingRecord`] in a slot
	/// where the grammar prohibits it. This variant is defensive — the parser
	/// cannot produce such a form — but keeps the validator total and the
	/// assembler panic-free even under future refactoring that might relax the
	/// grammar.
	UnexpectedRollingRecordOperand
	{
		/// The source position of the offending instruction.
		location: AssemblyLocation
	}
}

impl AssemblyError
{
	/// Answer the source position at which the error was detected.
	///
	/// # Returns
	/// The offending token's [`AssemblyLocation`].
	pub fn location(&self) -> AssemblyLocation
	{
		match self
		{
			Self::Syntax { location, .. }
			| Self::NonContiguousParameter { location, .. }
			| Self::NonContiguousExternal { location, .. }
			| Self::InsufficientRegisterCount { location, .. }
			| Self::RegisterOutOfBounds { location, .. }
			| Self::RollingRecordOutOfBounds { location, .. }
			| Self::RegisterGap { location, .. }
			| Self::RollingRecordGap { location, .. }
			| Self::FacelessCustomDice { location }
			| Self::DropSourceMismatch { location, .. }
			| Self::UnexpectedRollingRecordOperand { location } => *location
		}
	}

	/// Construct a [`Self::Syntax`] variant from a `nom` error, borrowing the
	/// source position from the offending span.
	///
	/// # Parameters
	/// - `error`: The `nom` error.
	///
	/// # Returns
	/// The constructed error.
	fn from_nom(error: NomError<Span>) -> Self
	{
		Self::Syntax {
			description: format!(
				"nom::{:?} near `{}`",
				error.code,
				token(error.input)
			),
			location: AssemblyLocation::of(error.input)
		}
	}
}

impl Display for AssemblyError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		let location = self.location();
		write!(
			f,
			"assembly error at line {}, column {} (byte {}): ",
			location.line, location.column, location.offset
		)?;
		match self
		{
			Self::Syntax { description, .. } =>
			{
				write!(f, "syntax error ({})", description)
			},
			Self::NonContiguousParameter {
				name,
				index,
				position,
				..
			} => write!(
				f,
				"parameter `{}` has index {} but appears in position {}; \
				 parameters must be contiguous from @0",
				name, index, position
			),
			Self::NonContiguousExternal {
				name,
				index,
				expected,
				arity,
				..
			} => write!(
				f,
				"external `{}` has index {} but expected @{} (externs must \
				 be contiguous, starting at @{})",
				name, index, expected, arity
			),
			Self::InsufficientRegisterCount {
				register_count,
				required,
				..
			} => write!(
				f,
				"header declares r#{} registers but parameters and externs \
				 together require at least @{}",
				register_count,
				required - 1
			),
			Self::RegisterOutOfBounds {
				index,
				register_count,
				..
			} => write!(
				f,
				"register @{} exceeds declared register count r#{}",
				index, register_count
			),
			Self::RollingRecordOutOfBounds {
				index,
				rolling_record_count,
				..
			} => write!(
				f,
				"rolling record ⚅{} exceeds declared rolling record count \
				 ⚅#{}",
				index, rolling_record_count
			),
			Self::RegisterGap {
				index,
				register_count,
				..
			} => write!(
				f,
				"register @{} is declared by r#{} but is never referenced \
				 (no gaps are permitted in the register file)",
				index, register_count
			),
			Self::RollingRecordGap {
				index,
				rolling_record_count,
				..
			} => write!(
				f,
				"rolling record ⚅{} is declared by ⚅#{} but is never \
				 referenced (no gaps are permitted in the rolling record \
				 file)",
				index, rolling_record_count
			),
			Self::FacelessCustomDice { .. } =>
			{
				write!(f, "custom dice must have at least one face")
			},
			Self::DropSourceMismatch {
				kind,
				destination,
				source,
				..
			} => write!(
				f,
				"drop {} destination ⚅{} disagrees with its `from` source \
				 ⚅{}",
				kind, destination, source
			),
			Self::UnexpectedRollingRecordOperand { .. } =>
			{
				write!(f, "rolling record operand is not permitted here")
			}
		}
	}
}

impl Error for AssemblyError {}

/// Which drop-direction triggered an [`AssemblyError::DropSourceMismatch`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DropKind
{
	/// The offending instruction was a `drop lowest`.
	Lowest,

	/// The offending instruction was a `drop highest`.
	Highest
}

impl Display for DropKind
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		match self
		{
			Self::Lowest => write!(f, "lowest"),
			Self::Highest => write!(f, "highest")
		}
	}
}

/// A source-position locator carried by every [`AssemblyError`] variant. Byte
/// offset plus 1-based line and column, the latter measured in `char`s —
/// matching the conventions of `nom_locate`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AssemblyLocation
{
	/// The byte offset within the input.
	pub offset: usize,

	/// The 1-based line number.
	pub line: u32,

	/// The 1-based column, measured in `char`s.
	pub column: usize
}

impl AssemblyLocation
{
	/// Capture the position of the given span's start.
	///
	/// # Parameters
	/// - `span`: The span whose start to capture.
	///
	/// # Returns
	/// The captured location.
	fn of(span: Span) -> Self
	{
		Self {
			offset: span.location_offset(),
			line: span.location_line(),
			column: span.get_utf8_column()
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           FromStr for Function.                            //
////////////////////////////////////////////////////////////////////////////////

impl FromStr for Function
{
	type Err = AssemblyError;

	fn from_str(s: &str) -> Result<Self, Self::Err> { Assembler::assemble(s) }
}

////////////////////////////////////////////////////////////////////////////////
//                           Raw, unvalidated form.                           //
////////////////////////////////////////////////////////////////////////////////

/// A named register reference from the header or extern line, paired with the
/// source position at which it was parsed.
#[derive(Clone, Debug)]
struct RawNamedReg
{
	/// The variable's name, as written in the input.
	name: String,

	/// The register index asserted by the input's `@N` suffix.
	index: usize,

	/// The source position of the reference.
	location: AssemblyLocation
}

/// A body instruction, paired with its source position and any auxiliary
/// information needed by the validator.
#[derive(Clone, Debug)]
struct RawInstruction
{
	/// The parsed instruction.
	instruction: Instruction,

	/// The source position of the instruction.
	location: AssemblyLocation,

	/// For [`Instruction::DropLowest`] and [`Instruction::DropHighest`], the
	/// `from ⚅N` clause's rolling record index. Validator compares this
	/// against the destination. `None` for any other instruction.
	drop_source: Option<RollingRecordIndex>
}

/// The unvalidated result of parsing. The [`Assembler`] walks a [`Raw`] to
/// produce a [`Function`].
#[derive(Clone, Debug)]
struct Raw
{
	/// The function's formal parameters.
	parameters: Vec<RawNamedReg>,

	/// The function's external variables.
	externals: Vec<RawNamedReg>,

	/// The register count declared in the header.
	register_count: usize,

	/// The rolling record count declared in the header.
	rolling_record_count: usize,

	/// The instructions parsed from the body.
	instructions: Vec<RawInstruction>,

	/// The source position of the header, used for diagnostics that refer to
	/// the whole function.
	header_location: AssemblyLocation
}

////////////////////////////////////////////////////////////////////////////////
//                            Parser combinators.                             //
////////////////////////////////////////////////////////////////////////////////

/// The type of a span of text, as threaded through the `nom` combinators.
type Span<'a> = LocatedSpan<&'a str>;

/// The result type of every combinator in this module.
type AsmResult<'a, T> = IResult<Span<'a>, T, NomError<Span<'a>>>;

/// Extract a short preview of the input at the given span, for use in error
/// messages. Returns at most the first 16 `char`s of the first line.
///
/// # Parameters
/// - `span`: The span to preview.
///
/// # Returns
/// A short preview string.
fn token(span: Span) -> String
{
	let line = span.fragment().lines().next().unwrap_or("");
	line.chars().take(16).collect()
}

/// Recognize a run of horizontal whitespace (spaces and tabs). Does not match
/// line terminators. Matches zero or more characters.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The remaining input after skipping horizontal whitespace.
///
/// # Errors
/// Never fails; returns `Ok` with an empty match if no horizontal whitespace is
/// present.
fn hws(input: Span) -> AsmResult<Span>
{
	take_while(|c: char| c == ' ' || c == '\t')(input)
}

/// Skip any run of horizontal whitespace and line terminators. Used to tolerate
/// leading blank lines before the header.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The remaining input after skipping whitespace.
///
/// # Errors
/// Never fails.
fn skip_ws_and_newlines(input: Span) -> AsmResult<()>
{
	let (input, _) = many0(alt((
		recognize(one_of::<_, _, NomError<Span>>(" \t")),
		recognize(line_ending)
	)))
	.parse_complete(input)?;
	Ok((input, ()))
}

/// Match the trailing whitespace that can follow the final instruction line.
/// Accepts any run of horizontal whitespace and line terminators, followed by
/// end of input.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The remaining input (empty on success).
///
/// # Errors
/// `nom` error if anything other than whitespace remains before end of
/// input.
fn trailing(input: Span) -> AsmResult<()>
{
	let (input, _) = skip_ws_and_newlines(input)?;
	let (input, _) = eof(input)?;
	Ok((input, ()))
}

/// Parse a non-negative decimal integer as a `usize`, saturating at
/// [`usize::MAX`] on overflow.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed index.
///
/// # Errors
/// `nom` error if no digits are present.
fn decimal_index(input: Span) -> AsmResult<usize>
{
	let (input, digits) = digit1(input)?;
	let parsed = digits.fragment().parse::<usize>().unwrap_or(usize::MAX);
	Ok((input, parsed))
}

/// Parse a signed decimal integer as an [`i32`], saturating at [`i32::MIN`] or
/// [`i32::MAX`] on overflow.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed integer.
///
/// # Errors
/// `nom` error if no digits are present.
fn signed_i32(input: Span) -> AsmResult<i32>
{
	let (input, text) = recognize(pair(opt(char('-')), digit1)).parse(input)?;
	let parsed = text.fragment().parse::<i32>().unwrap_or_else(|_| match text
		.fragment()
		.starts_with('-')
	{
		true => i32::MIN,
		false => i32::MAX
	});
	Ok((input, parsed))
}

/// Parse a register reference (`@N`).
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed register index.
///
/// # Errors
/// `nom` error if the input does not start with `@` followed by digits.
fn register_ref(input: Span) -> AsmResult<RegisterIndex>
{
	let (input, _) = char('@')(input)?;
	let (input, idx) = decimal_index(input)?;
	Ok((input, RegisterIndex(idx)))
}

/// Parse a rolling record reference (`⚅N`).
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed rolling record index.
///
/// # Errors
/// `nom` error if the input does not start with `⚅` followed by digits.
fn record_ref(input: Span) -> AsmResult<RollingRecordIndex>
{
	let (input, _) = char('⚅')(input)?;
	let (input, idx) = decimal_index(input)?;
	Ok((input, RollingRecordIndex(idx)))
}

/// Parse a `value_operand` (integer or register reference). Rolling record
/// references are syntactically forbidden here so that
/// [`Evaluator`](crate::Evaluator)'s unchecked operand dispatch cannot
/// encounter one.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed addressing mode.
///
/// # Errors
/// `nom` error if the input matches neither form.
fn value_operand(input: Span) -> AsmResult<AddressingMode>
{
	alt((
		map(register_ref, AddressingMode::Register),
		map(signed_i32, |v| AddressingMode::Immediate(Immediate(v)))
	))
	.parse(input)
}

/// Parse the `NAME` portion of a `named_reg`. Accepts any run of characters
/// other than the delimiters that bound a `named_reg`, trimmed of leading and
/// trailing ASCII whitespace.
///
/// The grammar for source-level identifiers permits inline whitespace
/// (e.g., `an external variable`), hyphens, dots, and Unicode, but not
/// `@`. Names are thus unambiguously terminated by `@` in the print
/// format.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The trimmed name.
///
/// # Errors
/// `nom` error if the matched run is empty after trimming.
fn named_reg_name(input: Span) -> AsmResult<String>
{
	let (rest, raw) = take_while1(|c: char| {
		!matches!(c, '@' | ',' | ')' | ']' | '\t' | '\r' | '\n')
	})(input)?;
	let trimmed = raw.fragment().trim();
	if trimmed.is_empty()
	{
		return Err(nom::Err::Error(NomError::new(
			input,
			nom::error::ErrorKind::TakeWhile1
		)))
	}
	Ok((rest, trimmed.to_string()))
}

/// Parse a `named_reg` (name followed by `@N`).
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed named register reference.
///
/// # Errors
/// `nom` error if the input does not match.
fn named_reg(input: Span) -> AsmResult<RawNamedReg>
{
	let location = AssemblyLocation::of(input);
	let (input, name) = named_reg_name(input)?;
	let (input, _) = char('@')(input)?;
	let (input, index) = decimal_index(input)?;
	Ok((
		input,
		RawNamedReg {
			name,
			index,
			location
		}
	))
}

/// Parse a comma-separated list of `named_reg`s, permissive on surrounding
/// horizontal whitespace. Returns an empty vector if the immediate input begins
/// with the closing delimiter.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed list.
///
/// # Errors
/// `nom` error if a non-terminal `,` is not followed by a valid `named_reg`.
fn named_reg_list(input: Span) -> AsmResult<Vec<RawNamedReg>>
{
	separated_list0(
		delimited(hws, char(','), hws),
		delimited(hws, named_reg, hws)
	)
	.parse(input)
}

/// Parse the `header` line.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed header: `(parameters, register_count, rolling_record_count,
/// location)`.
///
/// # Errors
/// `nom` error if the header is malformed.
fn header(
	input: Span
) -> AsmResult<(Vec<RawNamedReg>, usize, usize, AssemblyLocation)>
{
	let location = AssemblyLocation::of(input);
	let (input, _) = tag("Function(")(input)?;
	let (input, parameters) = named_reg_list(input)?;
	let (input, _) = char(')')(input)?;
	let (input, _) = hws(input)?;
	let (input, _) = tag("r#")(input)?;
	let (input, register_count) = decimal_index(input)?;
	let (input, _) = hws(input)?;
	let (input, _) = tag("⚅#")(input)?;
	let (input, rolling_record_count) = decimal_index(input)?;
	let (input, _) = hws(input)?;
	let (input, _) = line_ending(input)?;
	Ok((
		input,
		(parameters, register_count, rolling_record_count, location)
	))
}

/// Parse the `extern_line`.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed external variables.
///
/// # Errors
/// `nom` error if the line is malformed.
fn extern_line(input: Span) -> AsmResult<Vec<RawNamedReg>>
{
	let (input, _) = hws(input)?;
	let (input, _) = tag("extern[")(input)?;
	let (input, externals) = named_reg_list(input)?;
	let (input, _) = char(']')(input)?;
	let (input, _) = hws(input)?;
	let (input, _) = line_ending(input)?;
	Ok((input, externals))
}

/// Parse the `body:` header line.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// Unit.
///
/// # Errors
/// `nom` error if the line is malformed.
fn body_header(input: Span) -> AsmResult<()>
{
	let (input, _) = hws(input)?;
	let (input, _) = tag("body:")(input)?;
	let (input, _) = hws(input)?;
	let (input, _) = line_ending(input)?;
	Ok((input, ()))
}

/// Match a literal token surrounded by optional horizontal whitespace on the
/// right. Useful for keywords that appear adjacent to operands.
///
/// # Parameters
/// - `kw`: The literal token to match.
///
/// # Returns
/// A combinator that matches the token.
fn kw<'a>(kw: &'static str) -> impl Fn(Span<'a>) -> AsmResult<'a, ()>
{
	move |input| {
		let (input, _) = tag(kw)(input)?;
		let (input, _) = hws(input)?;
		Ok((input, ()))
	}
}

/// Parse a `roll_range` instruction.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_roll_range(input: Span) -> AsmResult<Instruction>
{
	let (input, dest) = record_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	let (input, _) = kw("roll range")(input)?;
	let (input, start) = value_operand(input)?;
	let (input, _) = delimited(hws, char(':'), hws).parse(input)?;
	let (input, end) = value_operand(input)?;
	Ok((input, Instruction::roll_range(dest, start, end)))
}

/// Parse a `roll_standard_dice` instruction.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_roll_standard_dice(input: Span) -> AsmResult<Instruction>
{
	let (input, dest) = record_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	let (input, _) = kw("roll standard dice")(input)?;
	let (input, count) = value_operand(input)?;
	let (input, _) = char('D')(input)?;
	let (input, faces) = value_operand(input)?;
	Ok((input, Instruction::roll_standard_dice(dest, count, faces)))
}

/// Parse a `roll_custom_dice` instruction.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_roll_custom_dice(input: Span) -> AsmResult<Instruction>
{
	let (input, dest) = record_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	let (input, _) = kw("roll custom dice")(input)?;
	let (input, count) = value_operand(input)?;
	let (input, _) = tag("D[")(input)?;
	let (input, faces) = separated_list1(
		delimited(hws, char(','), hws),
		delimited(hws, signed_i32, hws)
	)
	.parse(input)?;
	let (input, _) = char(']')(input)?;
	Ok((input, Instruction::roll_custom_dice(dest, count, faces)))
}

/// Parse a `drop_lowest` instruction, returning both the parsed [`Instruction`]
/// and the `from ⚅N` source. The validator compares the source against the
/// destination.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction and its `from` source.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_drop_lowest(input: Span)
-> AsmResult<(Instruction, RollingRecordIndex)>
{
	let (input, dest) = record_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	let (input, _) = kw("drop lowest")(input)?;
	let (input, count) = value_operand(input)?;
	let (input, _) = delimited(hws, tag("from"), hws).parse(input)?;
	let (input, src) = record_ref(input)?;
	Ok((input, (Instruction::drop_lowest(dest, count), src)))
}

/// Parse a `drop_highest` instruction, returning both the parsed
/// [`Instruction`] and the `from ⚅N` source.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction and its `from` source.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_drop_highest(
	input: Span
) -> AsmResult<(Instruction, RollingRecordIndex)>
{
	let (input, dest) = record_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	let (input, _) = kw("drop highest")(input)?;
	let (input, count) = value_operand(input)?;
	let (input, _) = delimited(hws, tag("from"), hws).parse(input)?;
	let (input, src) = record_ref(input)?;
	Ok((input, (Instruction::drop_highest(dest, count), src)))
}

/// A binary arithmetic operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BinaryOp
{
	// Addition.
	Add,

	// Subtraction.
	Sub,

	// Multiplication.
	Mul,

	// Division.
	Div,

	// Modulus.
	Mod,

	// Exponentiation.
	Exp
}

/// Parse the sigil for a binary arithmetic operator.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed operator.
///
/// # Errors
/// `nom` error if the input does not begin with a recognized sigil.
fn binary_op(input: Span) -> AsmResult<BinaryOp>
{
	alt((
		value(BinaryOp::Add, char('+')),
		value(BinaryOp::Sub, char('-')),
		value(BinaryOp::Mul, char('*')),
		value(BinaryOp::Div, char('/')),
		value(BinaryOp::Mod, char('%')),
		value(BinaryOp::Exp, char('^'))
	))
	.parse(input)
}

/// Parse the right-hand side of a register-destined instruction: the `<-` arrow
/// followed by either a binary-operator form or a unary-negation form. The
/// caller has already parsed the destination register.
///
/// Binary is attempted first because its leading `value_operand` can consume a
/// negative immediate (e.g., `-5` in `@N <- -5 + -3`). Trying unary first would
/// eagerly eat the `-5` as `Neg(Immediate(5))` and leave the remainder of the
/// line as a parse failure.
///
/// # Parameters
/// - `input`: The input text to parse.
/// - `dest`: The destination register already parsed by the caller.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the tail does not match.
fn register_tail(input: Span, dest: RegisterIndex) -> AsmResult<Instruction>
{
	let binary_attempt = (|| -> AsmResult<Instruction> {
		let (tail, op1) = value_operand(input)?;
		let (tail, _) = hws(tail)?;
		let (tail, op) = binary_op(tail)?;
		let (tail, _) = hws(tail)?;
		let (tail, op2) = value_operand(tail)?;
		let inst = match op
		{
			BinaryOp::Add => Instruction::add(dest, op1, op2),
			BinaryOp::Sub => Instruction::sub(dest, op1, op2),
			BinaryOp::Mul => Instruction::mul(dest, op1, op2),
			BinaryOp::Div => Instruction::div(dest, op1, op2),
			BinaryOp::Mod => Instruction::r#mod(dest, op1, op2),
			BinaryOp::Exp => Instruction::exp(dest, op1, op2)
		};
		Ok((tail, inst))
	})();
	if let Ok(parsed) = binary_attempt
	{
		return Ok(parsed)
	}
	// Unary form.
	let (input, _) = char('-')(input)?;
	let (input, _) = hws(input)?;
	let (input, op) = value_operand(input)?;
	Ok((input, Instruction::neg(dest, op)))
}

/// Parse an instruction whose destination is a register: either
/// `sum rolling record`, a binary arithmetic op, or a unary negation. The
/// caller peeks the leading `@` to know that this branch applies.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the instruction does not match any of the supported forms.
fn register_instruction(input: Span) -> AsmResult<Instruction>
{
	let (input, dest) = register_ref(input)?;
	let (input, _) = delimited(hws, tag("<-"), hws).parse(input)?;
	// `sum rolling record` is the only register-destined instruction that
	// is not a binary op or unary negation, and it is prefixed by the
	// literal "sum rolling record" keyword, so try it first.
	if let Ok((rest, _)) =
		tag::<_, _, NomError<Span>>("sum rolling record")(input)
	{
		let (rest, _) = hws(rest)?;
		let (rest, src) = record_ref(rest)?;
		return Ok((rest, Instruction::sum_rolling_record(dest, src)))
	}
	register_tail(input, dest)
}

/// Parse a `return` instruction.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction.
///
/// # Errors
/// `nom` error if the input does not match.
fn inst_return(input: Span) -> AsmResult<Instruction>
{
	let (input, _) = tag("return")(input)?;
	let (input, _) = hws(input)?;
	let (input, src) = value_operand(input)?;
	Ok((input, Instruction::r#return(src)))
}

/// Parse a rolling-record-destined instruction. The caller peeks the leading
/// `⚅` to know that this branch applies.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed instruction, paired with the `from ⚅N` source for drop
/// instructions (or `None` for the other forms).
///
/// # Errors
/// `nom` error if the instruction does not match any of the supported forms.
fn record_instruction(
	input: Span
) -> AsmResult<(Instruction, Option<RollingRecordIndex>)>
{
	// Speculate the parse of the destination + arrow to avoid duplicating it
	// across each sub-parser. We still fail the outer parse if any sub-parser
	// fails.
	let alt1 = map(inst_roll_range, |i| (i, None::<RollingRecordIndex>));
	let alt2 = map(inst_roll_standard_dice, |i| (i, None));
	let alt3 = map(inst_roll_custom_dice, |i| (i, None));
	let alt4 = map(inst_drop_lowest, |(i, src)| (i, Some(src)));
	let alt5 = map(inst_drop_highest, |(i, src)| (i, Some(src)));
	alt((alt1, alt2, alt3, alt4, alt5)).parse(input)
}

/// Parse a single body instruction, discarding the leading horizontal
/// whitespace and the trailing end-of-line.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed raw instruction.
///
/// # Errors
/// `nom` error if the line is malformed.
fn instruction_line(input: Span) -> AsmResult<RawInstruction>
{
	let (input, _) = hws(input)?;
	let location = AssemblyLocation::of(input);
	let (input, (instruction, drop_source)) = alt((
		record_instruction,
		map(register_instruction, |i| (i, None)),
		map(inst_return, |i| (i, None))
	))
	.parse(input)?;
	let (input, _) = hws(input)?;
	// The final instruction need not be newline-terminated; upstream callers
	// routinely `trim()` the text, which strips the trailing `\n` that
	// `Function::fmt` emits.
	let (input, _) = alt((line_ending, eof)).parse(input)?;
	Ok((
		input,
		RawInstruction {
			instruction,
			location,
			drop_source
		}
	))
}

/// Parse an entire function in print form into a [`Raw`] assemblage.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed raw assemblage.
///
/// # Errors
/// `nom` error if the input is malformed.
fn function(input: Span) -> AsmResult<Raw>
{
	let (input, (parameters, register_count, rolling_record_count, location)) =
		header(input)?;
	let (input, externals) = extern_line(input)?;
	let (input, _) = body_header(input)?;
	let (input, instructions) =
		terminated(many0(instruction_line), hws).parse(input)?;
	Ok((
		input,
		Raw {
			parameters,
			externals,
			register_count,
			rolling_record_count,
			instructions,
			header_location: location
		}
	))
}

////////////////////////////////////////////////////////////////////////////////
//                                   Tests.                                   //
////////////////////////////////////////////////////////////////////////////////

// Unit tests live alongside the rest of the assembler test suite in
// `src/tests/assembler.rs`, which additionally exercises the
// gold-standard round-trip corpus.
