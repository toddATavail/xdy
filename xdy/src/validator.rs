//! # Validator
//!
//! The validator performs semantic analysis on a parsed
//! [abstract syntax tree](crate::ast) (AST) before code generation. It sits
//! between the [parser](crate::parser) and the [compiler](crate::Compiler) in
//! the compilation pipeline, rejecting ASTs that are syntactically well-formed
//! but semantically invalid.
//!
//! Keeping validation as a distinct pass keeps the code generator infallible
//! (its [`ASTVisitor::Error`](crate::ast::ASTVisitor::Error) type is
//! [`Infallible`](std::convert::Infallible)) while allowing semantic errors to
//! carry typed, span-bearing payloads suitable for rich diagnostics.
//!
//! # Current checks
//!
//! - **Duplicate parameters.** A function may not declare the same formal
//!   parameter name more than once. `x, x: {x} + 1` is rejected as
//!   [`DuplicateParameter`](CompilationError::DuplicateParameter), carrying the
//!   spans of both occurrences for caret-level reporting.
//!
//! Additional semantic checks (e.g., use-before-bind, rebinding) will be added
//! here as downstream consumers need them.

use std::collections::HashMap;

use crate::{
	CompilationError, SourceSpan,
	ast::{
		self, ASTVisitor, Add, Constant, CustomDice, Div, DropHighest,
		DropLowest, Exp, Group, Mod, Mul, Neg, Range, StandardDice, Sub,
		Variable
	}
};

////////////////////////////////////////////////////////////////////////////////
//                              Validator pass.                               //
////////////////////////////////////////////////////////////////////////////////

/// A semantic-validation pass over an [abstract syntax tree](crate::ast)
/// (AST). Use [`Validator::validate`] as the high-level entry point; the
/// [`Validator`] type also implements [`ASTVisitor`](ast::ASTVisitor) for
/// callers who want to drive validation directly — e.g., to interleave it
/// with other AST analyses.
#[derive(Copy, Clone, Debug, Default)]
pub struct Validator;

impl Validator
{
	/// Construct a new [`Validator`].
	///
	/// # Returns
	/// A fresh [`Validator`], ready to validate an AST.
	#[inline]
	pub const fn new() -> Self { Self }

	/// Validate the semantic well-formedness of a parsed
	/// [function](ast::Function). Runs between parsing and code generation.
	///
	/// # Type parameters
	/// - `'src`: The lifetime of the source text from which the AST was parsed.
	///   Parameter names and spans reported in errors are borrowed directly
	///   from the source, so the error borrows for `'src`.
	///
	/// # Parameters
	/// - `ast`: The parsed function definition.
	///
	/// # Returns
	/// `Ok(())` if the function passes all semantic checks.
	///
	/// # Errors
	/// [`DuplicateParameter`](CompilationError::DuplicateParameter) if the
	/// function declares the same formal parameter name more than once.
	pub fn validate<'src>(
		ast: &ast::Function<'src>
	) -> Result<(), CompilationError<'src>>
	{
		check_duplicate_parameters(ast)
	}
}

/// The [duplicate-parameter](CompilationError::DuplicateParameter) check,
/// factored out so both [`Validator::validate`] and the [`ASTVisitor`]
/// implementation share a single source of truth. Writing the check here —
/// rather than inside [`Validator::visit_function`] — lets the reported error
/// borrow for the full source-text lifetime `'src`, independent of the AST
/// reference's (possibly shorter) borrow lifetime. The [`ASTVisitor`] trait
/// ties [`Error`](ASTVisitor::Error) to a single lifetime shared with the
/// node reference, so calling through the trait would constrain the error to
/// the reborrow lifetime.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `ast`: The parsed function definition.
///
/// # Returns
/// `Ok(())` if no parameter name is declared more than once.
///
/// # Errors
/// [`DuplicateParameter`](CompilationError::DuplicateParameter) with the spans
/// of the first and duplicate occurrences of the repeated name.
fn check_duplicate_parameters<'src>(
	ast: &ast::Function<'src>
) -> Result<(), CompilationError<'src>>
{
	if let Some(ref parameters) = ast.parameters
	{
		let mut seen: HashMap<&'src str, SourceSpan> =
			HashMap::with_capacity(parameters.len());
		for param in parameters
		{
			if let Some(&first) = seen.get(param.name)
			{
				return Err(CompilationError::DuplicateParameter {
					name: param.name,
					first,
					duplicate: param.span
				});
			}
			seen.insert(param.name, param.span);
		}
	}
	Ok(())
}

////////////////////////////////////////////////////////////////////////////////
//                         ASTVisitor for Validator.                          //
////////////////////////////////////////////////////////////////////////////////

impl<'src> ASTVisitor<'src> for Validator
{
	type Error = CompilationError<'src>;
	type Output = ();

	fn visit_function(
		&mut self,
		node: &'src ast::Function<'src>
	) -> Result<(), Self::Error>
	{
		check_duplicate_parameters(node)
	}

	fn visit_group(
		&mut self,
		_node: &'src Group<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_constant(&mut self, _node: &Constant) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_variable(
		&mut self,
		_node: &'src Variable<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_range(
		&mut self,
		_node: &'src Range<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_standard_dice(
		&mut self,
		_node: &'src StandardDice<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_custom_dice(
		&mut self,
		_node: &'src CustomDice<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_drop_lowest(
		&mut self,
		_node: &'src DropLowest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_drop_highest(
		&mut self,
		_node: &'src DropHighest<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_add(&mut self, _node: &'src Add<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_sub(&mut self, _node: &'src Sub<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_mul(&mut self, _node: &'src Mul<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_div(&mut self, _node: &'src Div<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_mod(&mut self, _node: &'src Mod<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_exp(&mut self, _node: &'src Exp<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_neg(&mut self, _node: &'src Neg<'src>) -> Result<(), Self::Error>
	{
		Ok(())
	}
}
