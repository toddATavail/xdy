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
//!   function body. `{x}@(3D6) + {x}@(1D4)` is rejected as
//!   [`DuplicateBinding`](CompilationError::DuplicateBinding).
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
		DropLowest, Event, Exp, Expression, Group, Mod, Mul, Neg, Node, Range,
		StandardDice, Sub, Variable, Walk
	}
};

////////////////////////////////////////////////////////////////////////////////
//                              Validator pass.                               //
////////////////////////////////////////////////////////////////////////////////

/// A semantic-validation pass over an [abstract syntax tree](crate::ast)
/// (AST). Use [`Validator::validate`] as the high-level entry point; the
/// [`Validator`] type also implements [`ASTVisitor`] for
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
	/// [`DuplicateParameter`](CompilationError::DuplicateParameter) if the
	/// function declares the same formal parameter name more than once.
	pub fn validate<'src>(
		ast: &ast::Function<'src>
	) -> Result<(), CompilationError<'src>>
	{
		check_duplicate_parameters(ast)?;
		let bindings = collect_bindings_and_check_collisions(ast)?;
		check_use_before_bind(&ast.body, &bindings)
	}
}

/// The [duplicate-parameter](CompilationError::DuplicateParameter) check,
/// factored out so both [`Validator::validate`] and the [`ASTVisitor`]
/// implementation share a single source of truth.
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
		let mut seen: HashMap<&str, SourceSpan> =
			HashMap::with_capacity(parameters.len());
		for param in parameters
		{
			if let Some(&first) = seen.get(&*param.name)
			{
				return Err(CompilationError::DuplicateParameter {
					name: param.name.clone(),
					first,
					duplicate: param.span
				});
			}
			seen.insert(&param.name, param.span);
		}
	}
	Ok(())
}

/// Walk the AST collecting every [local binding](Binding) by name, and in the
/// same pass reject duplicate bindings and bindings whose names collide with
/// formal parameters. The returned map records the binding-site
/// [`name_span`](Binding::name_span) for each binding, so the companion
/// [use-before-bind check](check_use_before_bind) can cite the binding site
/// when reporting an offending reference.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the AST. Binding names are borrowed
///   from the AST.
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `ast`: The parsed function definition.
///
/// # Returns
/// A map from binding name to the [span](SourceSpan) of its binding site.
///
/// # Errors
/// * [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
///   if a binding name matches a formal parameter name.
/// * [`DuplicateBinding`](CompilationError::DuplicateBinding) if the same name
///   appears as a binding more than once in the function body.
fn collect_bindings_and_check_collisions<'a, 'src>(
	ast: &'a ast::Function<'src>
) -> Result<HashMap<&'a str, SourceSpan>, CompilationError<'src>>
{
	let parameter_spans = match ast.parameters
	{
		Some(ref parameters) => parameters
			.iter()
			.map(|p| (&*p.name, p.span))
			.collect::<HashMap<_, _>>(),
		None => HashMap::new()
	};
	let mut bindings: HashMap<&'a str, SourceSpan> = HashMap::new();
	for event in Walk::new(Node::Expression(&ast.body))
	{
		if let Event::Enter(Node::Expression(Expression::Binding(b))) = event
		{
			if let Some(&parameter) = parameter_spans.get(&*b.name)
			{
				return Err(CompilationError::BindingCollidesWithParameter {
					name: b.name.clone(),
					parameter,
					binding: b.name_span
				});
			}
			if let Some(&first) = bindings.get(&*b.name)
			{
				return Err(CompilationError::DuplicateBinding {
					name: b.name.clone(),
					first,
					duplicate: b.name_span
				});
			}
			// Record the binding on entering it, before its bound
			// expression, so that a binding of the same name nested
			// within the bound expression is a duplicate of this one. A
			// self-reference within the bound expression is a variable,
			// not a binding, and surfaces as a use-before-bind instead.
			bindings.insert(&b.name, b.name_span);
		}
	}
	Ok(bindings)
}

/// Walk the body of a function in lexical order, rejecting any
/// [variable reference](Variable) whose name is introduced by a
/// [local binding](Binding) that has not yet been reached. Because every
/// binding is visited _after_ its own bound expression, a self-reference inside
/// the RHS surfaces here as [`UseBeforeBind`](CompilationError::UseBeforeBind)
/// — the sole mechanism by which self-reference is rejected.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the function body.
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `body`: The function body to check.
/// - `bindings`: The complete set of bindings in the body, keyed by name, with
///   the binding-site [span](SourceSpan) used to cite the binding in errors.
///
/// # Errors
/// [`UseBeforeBind`](CompilationError::UseBeforeBind) at the first offending
/// reference encountered in a left-to-right, depth-first walk.
fn check_use_before_bind<'a, 'src>(
	body: &'a Expression<'src>,
	bindings: &HashMap<&'a str, SourceSpan>
) -> Result<(), CompilationError<'src>>
{
	// The names of the bindings reached so far. A reference to a name that
	// appears in `bindings` but not yet here is a use-before-bind error. Names
	// not in `bindings` at all are not local bindings, and flow through to the
	// compiler as external variables or parameters.
	let mut seen: HashSet<&'a str> = HashSet::new();
	for event in Walk::new(Node::Expression(body))
	{
		match event
		{
			Event::Enter(Node::Expression(Expression::Variable(v))) =>
			{
				if let Some(&binding_span) = bindings.get(&*v.name)
					&& !seen.contains(&*v.name)
				{
					return Err(CompilationError::UseBeforeBind {
						name: v.name.clone(),
						reference: v.span,
						binding: binding_span
					});
				}
			},
			Event::Leave(Node::Expression(Expression::Binding(b))) =>
			{
				seen.insert(&b.name);
			},
			_ =>
			{}
		}
	}
	Ok(())
}

////////////////////////////////////////////////////////////////////////////////
//                         ASTVisitor for Validator.                          //
////////////////////////////////////////////////////////////////////////////////

impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for Validator
{
	type Error = CompilationError<'src>;
	type Output = ();

	fn enter_function(
		&mut self,
		node: &'a ast::Function<'src>
	) -> Result<(), Self::Error>
	{
		check_duplicate_parameters(node)
	}

	fn visit_function(
		&mut self,
		_node: &'a ast::Function<'src>,
		_body: ()
	) -> Result<(), Self::Error>
	{
		Ok(())
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
		_node: &'a Variable<'src>
	) -> Result<(), Self::Error>
	{
		Ok(())
	}

	fn visit_binding(
		&mut self,
		_node: &'a Binding<'src>,
		_expression: ()
	) -> Result<(), Self::Error>
	{
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
