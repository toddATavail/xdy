//! # Validator
//!
//! The validator performs semantic analysis on a parsed
//! [abstract syntax tree](crate::ast) (AST) before code generation. It sits
//! between the [parser](crate::parser) and the [compiler](crate::Compiler) in
//! the compilation pipeline, rejecting ASTs that are syntactically well-formed
//! but semantically invalid.
//!
//! Keeping validation as a distinct pass keeps the code generator infallible
//! (its [`ASTVisitor::Error`] type is
//! [`Infallible`](std::convert::Infallible)) while allowing semantic errors to
//! carry typed, span-bearing payloads suitable for rich diagnostics.
//!
//! # Current checks
//!
//! - **Duplicate parameters.** A function may not declare the same formal
//!   parameter name more than once. `{x}, {x}: {x} + 1` is rejected as
//!   [`DuplicateParameter`](CompilationError::DuplicateParameter), carrying the
//!   spans of both occurrences for caret-level reporting.
//! - **Binding collides with parameter.** A [local binding](ast::Binding) may
//!   not reuse a formal-parameter name. `{x}: {x}@(3D6) + {x}` is rejected as
//!   [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter),
//!   since binding, parameter, and environment-variable names share a single
//!   flat namespace per function.
//! - **Duplicate binding.** The same name may not be bound twice within a
//!   function body, even when one binding is nested within the other's bound
//!   expression. `{x}@(3D6) + {x}@(1D4)` and `{x}@({x}@(1))` are both rejected
//!   as [`DuplicateBinding`](CompilationError::DuplicateBinding).
//! - **Use before bind.** A [variable reference](ast::Variable) must lexically
//!   follow the [binding](ast::Binding) that introduces its name, including any
//!   reference inside the bound expression itself (self-reference). `{x} +
//!   {x}@(3D6)` and `{x}@({x})` are both rejected as
//!   [`UseBeforeBind`](CompilationError::UseBeforeBind).

use std::collections::{HashMap, HashSet};

use crate::{
	CompilationError, SourceSpan,
	ast::{
		self, ASTVisitor, Add, Binding, Constant, CustomDice, Div, DropHighest,
		DropLowest, Exp, Group, Mod, Mul, Neg, Range, StandardDice, Sub,
		Variable
	}
};

////////////////////////////////////////////////////////////////////////////////
//                              Validator pass.                               //
////////////////////////////////////////////////////////////////////////////////

/// A semantic-validation pass over an [abstract syntax tree](crate::ast)
/// (AST). Use [`Validator::validate`] as the entry point; it performs every
/// semantic check.
///
/// The [`Validator`] performs its checks as an [`ASTVisitor`], in a single
/// walk, so a caller that drives it through [`accept`](ast::Function::accept)
/// gets the same validation as [`Validator::validate`]. The validator resets
/// itself upon entering a [function](ast::Function), so it may validate many
/// functions in turn.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the AST. Names are borrowed from the
///   AST during the walk.
/// - `'src`: The lifetime of the source text from which the AST was parsed.
///
/// # Notes
/// The walk reports the first error by kind, and only then by position: a
/// [`DuplicateParameter`](CompilationError::DuplicateParameter), before the
/// body is walked; then the first
/// [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
/// or [`DuplicateBinding`](CompilationError::DuplicateBinding) in pre-order;
/// and then the first [`UseBeforeBind`](CompilationError::UseBeforeBind). A
/// collision anywhere in the body outranks a use before bind that precedes it,
/// so the walk records each reference that may prove to be a use before bind,
/// and resolves them only upon [visiting](ASTVisitor::visit_function) the
/// function, when every binding is known.
#[derive(Clone, Debug, Default)]
pub struct Validator<'a, 'src>
{
	/// The formal parameters of the function, mapped to the spans of their
	/// declarations.
	parameters: HashMap<&'a str, SourceSpan>,

	/// The [local bindings](Binding) entered so far, mapped to the spans of
	/// their names. A binding enters this map before its bound expression, so
	/// that a binding of the same name nested within the bound expression is
	/// a duplicate of it.
	bindings: HashMap<&'a str, SourceSpan>,

	/// The names of the [local bindings](Binding) whose bound expressions are
	/// complete. A reference to a name is a use before bind unless the name is
	/// here, so a self-reference within a bound expression is one.
	bound: HashSet<&'a str>,

	/// The first reference to each name that was not yet [bound](Self::bound)
	/// when the walk reached it, in the order reached. Each is a use before
	/// bind if its name proves to be that of a [binding](Self::bindings);
	/// otherwise it refers to a parameter or an external variable.
	forward: Vec<&'a Variable<'src>>,

	/// The names of the references in [`forward`](Self::forward). Only the
	/// first reference to a name can be the first use before bind, so later
	/// ones are not recorded.
	referenced: HashSet<&'a str>
}

impl Validator<'_, '_>
{
	/// Construct a new [`Validator`].
	///
	/// # Returns
	/// A fresh [`Validator`], ready to validate an AST.
	#[inline]
	pub fn new() -> Self { Self::default() }

	/// Validate the semantic well-formedness of a parsed
	/// [function](ast::Function). Runs between parsing and code generation.
	///
	/// # Type parameters
	/// - `'src`: The lifetime of the source text from which the AST was parsed.
	///   Names reported in errors are borrowed from the source wherever it
	///   spells them [canonically](crate::parser::canonical_name), so the error
	///   borrows for `'src`, but not from the AST.
	///
	/// # Parameters
	/// - `ast`: The parsed function definition.
	///
	/// # Returns
	/// `Ok(())` if the function passes all semantic checks.
	///
	/// # Errors
	/// * [`DuplicateParameter`](CompilationError::DuplicateParameter) if the
	///   function declares the same formal parameter name more than once.
	/// * [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
	///   if a [local binding](ast::Binding) uses a name that is already
	///   declared as a formal parameter.
	/// * [`DuplicateBinding`](CompilationError::DuplicateBinding) if the same
	///   name is bound more than once within the same function body, including
	///   a binding nested within the bound expression of a binding of the same
	///   name, e.g., `{x}@({x}@(1))`.
	/// * [`UseBeforeBind`](CompilationError::UseBeforeBind) if a [variable
	///   reference](ast::Variable) appears lexically before the
	///   [binding](ast::Binding) that introduces its name, including a
	///   reference within the binding's own bound expression, e.g.,
	///   `{x}@({x})`.
	///
	/// The [`Validator`] describes which error is reported when there are
	/// several.
	pub fn validate<'src>(
		ast: &ast::Function<'src>
	) -> Result<(), CompilationError<'src>>
	{
		ast.accept(&mut Validator::new())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                         ASTVisitor for Validator.                          //
////////////////////////////////////////////////////////////////////////////////

impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for Validator<'a, 'src>
{
	type Error = CompilationError<'src>;
	type Output = ();

	fn enter_function(
		&mut self,
		node: &'a ast::Function<'src>
	) -> Result<(), Self::Error>
	{
		*self = Self::default();
		if let Some(ref parameters) = node.parameters
		{
			for param in parameters
			{
				if let Some(&first) = self.parameters.get(&*param.name)
				{
					return Err(CompilationError::DuplicateParameter {
						name: param.name.clone(),
						first,
						duplicate: param.span
					});
				}
				self.parameters.insert(&param.name, param.span);
			}
		}
		Ok(())
	}

	fn enter_binding(
		&mut self,
		node: &'a Binding<'src>
	) -> Result<(), Self::Error>
	{
		if let Some(&parameter) = self.parameters.get(&*node.name)
		{
			return Err(CompilationError::BindingCollidesWithParameter {
				name: node.name.clone(),
				parameter,
				binding: node.name_span
			});
		}
		if let Some(&first) = self.bindings.get(&*node.name)
		{
			return Err(CompilationError::DuplicateBinding {
				name: node.name.clone(),
				first,
				duplicate: node.name_span
			});
		}
		self.bindings.insert(&node.name, node.name_span);
		Ok(())
	}

	fn visit_function(
		&mut self,
		_node: &'a ast::Function<'src>,
		_body: ()
	) -> Result<(), Self::Error>
	{
		// Every binding is now known, so the first recorded reference to the
		// name of a binding is the first use before bind.
		match self.forward.iter().find_map(|reference| {
			self.bindings
				.get(&*reference.name)
				.map(|&binding| (reference, binding))
		})
		{
			Some((reference, binding)) =>
			{
				Err(CompilationError::UseBeforeBind {
					name: reference.name.clone(),
					reference: reference.span,
					binding
				})
			},
			None => Ok(())
		}
	}

	fn visit_group(
		&mut self,
		_node: &'a Group<'src>,
		_expression: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_constant(&mut self, _node: &'a Constant)
	-> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_variable(
		&mut self,
		node: &'a Variable<'src>
	) -> Result<(), Self::Error>
	{
		if !self.bound.contains(&*node.name)
			&& self.referenced.insert(&node.name)
		{
			self.forward.push(node);
		}
		Ok(())
	}

	fn visit_binding(
		&mut self,
		node: &'a Binding<'src>,
		_expression: ()
	) -> Result<(), Self::Error>
	{
		self.bound.insert(&node.name);
		Ok(())
	}

	fn visit_range(
		&mut self,
		_node: &'a Range<'src>,
		_start: (),
		_end: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>,
		_count: (),
		_faces: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_custom_dice(
		&mut self,
		_node: &'a CustomDice<'src>,
		_count: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>,
		_dice: (),
		_drop: Option<()>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>,
		_dice: (),
		_drop: Option<()>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_add(
		&mut self,
		_node: &'a Add<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_sub(
		&mut self,
		_node: &'a Sub<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_mul(
		&mut self,
		_node: &'a Mul<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_div(
		&mut self,
		_node: &'a Div<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_mod(
		&mut self,
		_node: &'a Mod<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_exp(
		&mut self,
		_node: &'a Exp<'src>,
		_left: (),
		_right: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_neg(
		&mut self,
		_node: &'a Neg<'src>,
		_operand: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
	}
}
