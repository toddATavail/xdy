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
//! - **Binding collides with parameter.** A [local binding](ast::Binding) may
//!   not reuse a formal-parameter name. `x: x@(3D6) + {x}` is rejected as
//!   [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter),
//!   since binding, parameter, and environment-variable names share a single
//!   flat namespace per function.
//! - **Duplicate binding.** The same name may not be bound twice within a
//!   function body. `x@(3D6) + x@(1D4)` is rejected as
//!   [`DuplicateBinding`](CompilationError::DuplicateBinding).
//! - **Use before bind.** A [variable reference](ast::Variable) must lexically
//!   follow the [binding](ast::Binding) that introduces its name, including any
//!   reference inside the bound expression itself (self-reference). `{x} +
//!   x@(3D6)` and `x@({x})` are both rejected as
//!   [`UseBeforeBind`](CompilationError::UseBeforeBind).

use std::collections::{HashMap, HashSet};

use crate::{
	CompilationError, SourceSpan,
	ast::{
		self, ASTVisitor, Add, ArithmeticExpression, Binding, Constant,
		CustomDice, DiceExpression, Div, DropHighest, DropLowest, Exp,
		Expression, Group, Mod, Mul, Neg, Range, StandardDice, Sub, Variable
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
		check_duplicate_parameters(ast)?;
		let bindings = collect_bindings_and_check_collisions(ast)?;
		check_use_before_bind(&ast.body, &bindings)
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

/// Walk the AST collecting every [local binding](Binding) by name, and in the
/// same pass reject duplicate bindings and bindings whose names collide with
/// formal parameters. The returned map records the binding-site
/// [`name_span`](Binding::name_span) for each binding, so the companion
/// [use-before-bind check](check_use_before_bind) can cite the binding site
/// when reporting an offending reference.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text. Binding names and spans are
///   borrowed directly from the source.
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
fn collect_bindings_and_check_collisions<'src>(
	ast: &ast::Function<'src>
) -> Result<HashMap<&'src str, SourceSpan>, CompilationError<'src>>
{
	let parameter_spans = match ast.parameters
	{
		Some(ref parameters) => parameters
			.iter()
			.map(|p| (p.name, p.span))
			.collect::<HashMap<_, _>>(),
		None => HashMap::new()
	};
	let mut bindings: HashMap<&'src str, SourceSpan> = HashMap::new();
	gather_bindings(&ast.body, &parameter_spans, &mut bindings)?;
	Ok(bindings)
}

/// Recursively walk `expr` accumulating [binding](Binding) name spans into
/// `bindings`, producing the collision errors described by
/// [`collect_bindings_and_check_collisions`].
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `expr`: The expression to walk.
/// - `parameters`: Formal-parameter names mapped to their declaration spans.
/// - `bindings`: Accumulator for binding-site name spans, keyed by name.
///
/// # Errors
/// See [`collect_bindings_and_check_collisions`].
fn gather_bindings<'src>(
	expr: &Expression<'src>,
	parameters: &HashMap<&'src str, SourceSpan>,
	bindings: &mut HashMap<&'src str, SourceSpan>
) -> Result<(), CompilationError<'src>>
{
	match expr
	{
		Expression::Binding(b) =>
		{
			if let Some(&parameter) = parameters.get(b.name)
			{
				return Err(CompilationError::BindingCollidesWithParameter {
					name: b.name,
					parameter,
					binding: b.name_span
				});
			}
			if let Some(&first) = bindings.get(b.name)
			{
				return Err(CompilationError::DuplicateBinding {
					name: b.name,
					first,
					duplicate: b.name_span
				});
			}
			// Walk the bound expression first so that a self-reference inside
			// the RHS is recorded as a [binding](Binding) of the same name (via
			// nested binding) or surfaces later as a use-before-bind, never as
			// a duplicate against the binding we are about to add.
			gather_bindings(&b.expression, parameters, bindings)?;
			bindings.insert(b.name, b.name_span);
		},
		Expression::Group(g) =>
		{
			gather_bindings(&g.expression, parameters, bindings)?
		},
		Expression::Range(r) =>
		{
			gather_bindings(&r.start, parameters, bindings)?;
			gather_bindings(&r.end, parameters, bindings)?;
		},
		Expression::Dice(d) => gather_dice_bindings(d, parameters, bindings)?,
		Expression::Arithmetic(a) =>
		{
			gather_arithmetic_bindings(a, parameters, bindings)?
		},
		Expression::Variable(_) | Expression::Constant(_) =>
		{}
	}
	Ok(())
}

/// Recursively collect [binding](Binding) name spans from a dice expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `dice`: The dice expression to walk.
/// - `parameters`: Formal-parameter names mapped to their declaration spans.
/// - `bindings`: Accumulator for binding-site name spans, keyed by name.
///
/// # Errors
/// See [`collect_bindings_and_check_collisions`].
fn gather_dice_bindings<'src>(
	dice: &DiceExpression<'src>,
	parameters: &HashMap<&'src str, SourceSpan>,
	bindings: &mut HashMap<&'src str, SourceSpan>
) -> Result<(), CompilationError<'src>>
{
	match dice
	{
		DiceExpression::Standard(d) =>
		{
			gather_bindings(&d.count, parameters, bindings)?;
			gather_bindings(&d.faces, parameters, bindings)?;
		},
		DiceExpression::Custom(d) =>
		{
			gather_bindings(&d.count, parameters, bindings)?;
		},
		DiceExpression::DropLowest(d) =>
		{
			gather_dice_bindings(&d.dice, parameters, bindings)?;
			if let Some(ref drop) = d.drop
			{
				gather_bindings(drop, parameters, bindings)?;
			}
		},
		DiceExpression::DropHighest(d) =>
		{
			gather_dice_bindings(&d.dice, parameters, bindings)?;
			if let Some(ref drop) = d.drop
			{
				gather_bindings(drop, parameters, bindings)?;
			}
		}
	}
	Ok(())
}

/// Recursively collect [binding](Binding) name spans from an arithmetic
/// expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `arith`: The arithmetic expression to walk.
/// - `parameters`: Formal-parameter names mapped to their declaration spans.
/// - `bindings`: Accumulator for binding-site name spans, keyed by name.
///
/// # Errors
/// See [`collect_bindings_and_check_collisions`].
fn gather_arithmetic_bindings<'src>(
	arith: &ArithmeticExpression<'src>,
	parameters: &HashMap<&'src str, SourceSpan>,
	bindings: &mut HashMap<&'src str, SourceSpan>
) -> Result<(), CompilationError<'src>>
{
	match arith
	{
		ArithmeticExpression::Add(a) =>
		{
			gather_bindings(&a.left, parameters, bindings)?;
			gather_bindings(&a.right, parameters, bindings)?;
		},
		ArithmeticExpression::Sub(s) =>
		{
			gather_bindings(&s.left, parameters, bindings)?;
			gather_bindings(&s.right, parameters, bindings)?;
		},
		ArithmeticExpression::Mul(m) =>
		{
			gather_bindings(&m.left, parameters, bindings)?;
			gather_bindings(&m.right, parameters, bindings)?;
		},
		ArithmeticExpression::Div(d) =>
		{
			gather_bindings(&d.left, parameters, bindings)?;
			gather_bindings(&d.right, parameters, bindings)?;
		},
		ArithmeticExpression::Mod(m) =>
		{
			gather_bindings(&m.left, parameters, bindings)?;
			gather_bindings(&m.right, parameters, bindings)?;
		},
		ArithmeticExpression::Exp(e) =>
		{
			gather_bindings(&e.left, parameters, bindings)?;
			gather_bindings(&e.right, parameters, bindings)?;
		},
		ArithmeticExpression::Neg(n) =>
		{
			gather_bindings(&n.operand, parameters, bindings)?
		},
	}
	Ok(())
}

/// Walk the body of a function in lexical order, rejecting any
/// [variable reference](Variable) whose name is introduced by a
/// [local binding](Binding) that has not yet been reached. Because every
/// binding is visited _after_ its own bound expression, a self-reference inside
/// the RHS surfaces here as [`UseBeforeBind`](CompilationError::UseBeforeBind)
/// — the sole mechanism by which self-reference is rejected.
///
/// # Type parameters
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
fn check_use_before_bind<'src>(
	body: &Expression<'src>,
	bindings: &HashMap<&'src str, SourceSpan>
) -> Result<(), CompilationError<'src>>
{
	let mut seen: HashSet<&'src str> = HashSet::new();
	walk_use_before_bind(body, bindings, &mut seen)
}

/// Recursive implementation of [`check_use_before_bind`]. `seen` grows as
/// bindings are encountered during the left-to-right walk; a reference to a
/// name that appears in `bindings` but not yet in `seen` is a use-before-bind
/// error. Names not in `bindings` at all are not local bindings and flow
/// through to the compiler as external variables or parameters.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `expr`: The expression to walk.
/// - `bindings`: Binding-site name spans, keyed by name.
/// - `seen`: The set of binding names encountered so far in the walk.
///
/// # Errors
/// See [`check_use_before_bind`].
fn walk_use_before_bind<'src>(
	expr: &Expression<'src>,
	bindings: &HashMap<&'src str, SourceSpan>,
	seen: &mut HashSet<&'src str>
) -> Result<(), CompilationError<'src>>
{
	match expr
	{
		Expression::Variable(v) =>
		{
			if let Some(&binding_span) = bindings.get(v.name)
				&& !seen.contains(v.name)
			{
				return Err(CompilationError::UseBeforeBind {
					name: v.name,
					reference: v.span,
					binding: binding_span
				});
			}
		},
		Expression::Binding(b) =>
		{
			walk_use_before_bind(&b.expression, bindings, seen)?;
			seen.insert(b.name);
		},
		Expression::Group(g) =>
		{
			walk_use_before_bind(&g.expression, bindings, seen)?
		},
		Expression::Range(r) =>
		{
			walk_use_before_bind(&r.start, bindings, seen)?;
			walk_use_before_bind(&r.end, bindings, seen)?;
		},
		Expression::Dice(d) => walk_dice_use_before_bind(d, bindings, seen)?,
		Expression::Arithmetic(a) =>
		{
			walk_arithmetic_use_before_bind(a, bindings, seen)?
		},
		Expression::Constant(_) =>
		{}
	}
	Ok(())
}

/// Recursively check use-before-bind in a dice expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `dice`: The dice expression to walk.
/// - `bindings`: Binding-site name spans, keyed by name.
/// - `seen`: The set of binding names encountered so far in the walk.
///
/// # Errors
/// See [`check_use_before_bind`].
fn walk_dice_use_before_bind<'src>(
	dice: &DiceExpression<'src>,
	bindings: &HashMap<&'src str, SourceSpan>,
	seen: &mut HashSet<&'src str>
) -> Result<(), CompilationError<'src>>
{
	match dice
	{
		DiceExpression::Standard(d) =>
		{
			walk_use_before_bind(&d.count, bindings, seen)?;
			walk_use_before_bind(&d.faces, bindings, seen)?;
		},
		DiceExpression::Custom(d) =>
		{
			walk_use_before_bind(&d.count, bindings, seen)?;
		},
		DiceExpression::DropLowest(d) =>
		{
			walk_dice_use_before_bind(&d.dice, bindings, seen)?;
			if let Some(ref drop) = d.drop
			{
				walk_use_before_bind(drop, bindings, seen)?;
			}
		},
		DiceExpression::DropHighest(d) =>
		{
			walk_dice_use_before_bind(&d.dice, bindings, seen)?;
			if let Some(ref drop) = d.drop
			{
				walk_use_before_bind(drop, bindings, seen)?;
			}
		}
	}
	Ok(())
}

/// Recursively check use-before-bind in an arithmetic expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
///
/// # Parameters
/// - `arith`: The arithmetic expression to walk.
/// - `bindings`: Binding-site name spans, keyed by name.
/// - `seen`: The set of binding names encountered so far in the walk.
///
/// # Errors
/// See [`check_use_before_bind`].
fn walk_arithmetic_use_before_bind<'src>(
	arith: &ArithmeticExpression<'src>,
	bindings: &HashMap<&'src str, SourceSpan>,
	seen: &mut HashSet<&'src str>
) -> Result<(), CompilationError<'src>>
{
	match arith
	{
		ArithmeticExpression::Add(a) =>
		{
			walk_use_before_bind(&a.left, bindings, seen)?;
			walk_use_before_bind(&a.right, bindings, seen)?;
		},
		ArithmeticExpression::Sub(s) =>
		{
			walk_use_before_bind(&s.left, bindings, seen)?;
			walk_use_before_bind(&s.right, bindings, seen)?;
		},
		ArithmeticExpression::Mul(m) =>
		{
			walk_use_before_bind(&m.left, bindings, seen)?;
			walk_use_before_bind(&m.right, bindings, seen)?;
		},
		ArithmeticExpression::Div(d) =>
		{
			walk_use_before_bind(&d.left, bindings, seen)?;
			walk_use_before_bind(&d.right, bindings, seen)?;
		},
		ArithmeticExpression::Mod(m) =>
		{
			walk_use_before_bind(&m.left, bindings, seen)?;
			walk_use_before_bind(&m.right, bindings, seen)?;
		},
		ArithmeticExpression::Exp(e) =>
		{
			walk_use_before_bind(&e.left, bindings, seen)?;
			walk_use_before_bind(&e.right, bindings, seen)?;
		},
		ArithmeticExpression::Neg(n) =>
		{
			walk_use_before_bind(&n.operand, bindings, seen)?
		},
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

	fn visit_binding(
		&mut self,
		_node: &'src Binding<'src>
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
