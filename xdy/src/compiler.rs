//! # Compiler
//!
//! The dice language compiles to an [intermediate representation](crate::ir)
//! (IR). The IR is generated using static single assignment (SSA) form, where
//! all registers have a uniform type, `i32`.
//!
//! The [`Compiler`] implements the [`ASTVisitor`]
//! trait, producing an [`AddressingMode`] from each node. Users who want to
//! walk the AST for their own purposes can implement [`ASTVisitor`] directly;
//! the [`Compiler`] serves as the reference implementation.
//!
//! # Usage
//!
//! Most users should use the top-level [`compile()`] and
//! [`evaluate()`](crate::evaluate)
//! functions, which handle the full pipeline automatically. The compiler is
//! exposed for advanced users who want to drive the pipeline manually — e.g.,
//! to insert custom AST analysis or transformation passes between parsing and
//! code generation.
//!
//! ```
//! use xdy::{
//!     Compiler, Evaluator, Optimizer, Parser, Passes, StandardOptimizer,
//!     Validator
//! };
//!
//! let ast = Parser::parse("2d6 + 3").unwrap();
//! Validator::validate(&ast).unwrap();
//! let function = Compiler::compile(&ast);
//! let optimized = StandardOptimizer::new(Passes::all())
//!     .optimize(function)
//!     .unwrap();
//! let mut rng = rand::rng();
//! let evaluation = Evaluator::new(optimized).evaluate([], &mut rng).unwrap();
//! ```

use std::{
	borrow::Cow,
	collections::{HashMap, HashSet},
	convert::Infallible,
	error::Error,
	fmt::{Display, Formatter}
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
	CanAllocate as _, Optimizer as _, Parser, Passes, SourceSpan,
	StandardOptimizer, Validator,
	ast::{self, ASTVisitor, Binding, Constant, Event, Expression, Node, Walk},
	ir::{
		AddressingMode, Immediate, Instruction, RegisterIndex,
		RollingRecordIndex
	},
	parser::{ParseError, is_canonical_name}
};

////////////////////////////////////////////////////////////////////////////////
//                          Convenient compilation.                           //
////////////////////////////////////////////////////////////////////////////////

/// Compile an alleged dice expression into a [function](Function). Do not
/// optimize the function.
///
/// # Parameters
/// - `source`: The source code to compile.
///
/// # Returns
/// The compiled function.
///
/// # Errors
/// * [`ParseError`](CompilationError::ParseError) if the source code could not
///   be parsed.
/// * [`DuplicateParameter`](CompilationError::DuplicateParameter) if the
///   function declares the same formal parameter name more than once.
/// * [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
///   if a [local binding](crate::ast::Binding) uses a name that is already
///   declared as a formal parameter.
/// * [`DuplicateBinding`](CompilationError::DuplicateBinding) if the same name
///   is bound more than once within the same function body.
/// * [`UseBeforeBind`](CompilationError::UseBeforeBind) if a [variable
///   reference](crate::ast::Variable) appears lexically before the
///   [binding](crate::ast::Binding) that introduces its name.
pub fn compile_unoptimized(
	source: &str
) -> Result<Function, CompilationError<'_>>
{
	let ast = Parser::parse(source).map_err(CompilationError::ParseError)?;
	Validator::validate(&ast)?;
	Ok(Compiler::compile(&ast))
}

/// Compile an alleged dice expression into a [function](Function). Optimize the
/// function using the [standard optimizer](StandardOptimizer).
///
/// # Pipeline
///
/// This function drives the full compilation pipeline — the happy path.
/// Syntactic and semantic errors produce typed [`CompilationError`] values;
/// for the rich-diagnostic sad path used by editor integrations, see
/// [`diagnose`](crate::diagnostics::diagnose).
///
/// ```mermaid
/// graph LR
///     A["Source Code<br/><code>&amp;str</code>"] --> B["Parser<br/><code>Parser::parse</code>"]
///     B --> C["AST<br/><code>ast::Function</code>"]
///     C --> V["Validator<br/><code>Validator::validate</code>"]
///     V --> D["Compiler<br/><code>Compiler::compile</code>"]
///     D --> E["IR<br/><code>Function</code>"]
///     E --> F["Optimizer<br/><code>StandardOptimizer</code><br/>every pass"]
///     F --> G["Optimized IR<br/><code>Function</code>"]
///     style A fill:#f9f,stroke:#333,color:#000
///     style G fill:#9f9,stroke:#333,color:#000
/// ```
///
/// # Parameters
/// - `source`: The source code to compile.
///
/// # Returns
/// The compiled function.
///
/// # Errors
/// * [`ParseError`](CompilationError::ParseError) if the source code could not
///   be parsed.
/// * [`DuplicateParameter`](CompilationError::DuplicateParameter) if the
///   function declares the same formal parameter name more than once.
/// * [`BindingCollidesWithParameter`](CompilationError::BindingCollidesWithParameter)
///   if a [local binding](crate::ast::Binding) uses a name that is already
///   declared as a formal parameter.
/// * [`DuplicateBinding`](CompilationError::DuplicateBinding) if the same name
///   is bound more than once within the same function body.
/// * [`UseBeforeBind`](CompilationError::UseBeforeBind) if a [variable
///   reference](crate::ast::Variable) appears lexically before the
///   [binding](crate::ast::Binding) that introduces its name.
/// * [`OptimizationFailed`](CompilationError::OptimizationFailed) if the
///   function could not be optimized.
///
/// # Examples
/// Compile and optimize a dice expression and evaluate it multiple times:
///
/// ```rust
/// use xdy::{compile, CompilationError, Evaluator};
/// use rand::rng;
///
/// # fn main() -> Result<(), CompilationError<'static>> {
/// let function = compile("3D6")?;
/// let mut evaluator = Evaluator::new(function);
/// let results = (0..10)
///     .flat_map(|_| evaluator.evaluate(vec![], &mut rng()))
///     .collect::<Vec<_>>();
/// assert!(results.len() == 10);
/// assert!(
///     results.iter().all(|result| 3 <= result.result && result.result <= 18)
/// );
/// # Ok(())
/// # }
/// ```
///
/// Compile and optimize a dice expression with formal parameters and evaluate
/// it multiple times with different arguments:
///
/// ```rust
/// use xdy::{compile, CompilationError, Evaluator};
/// use rand::rng;
///
/// # fn main() -> Result<(), CompilationError<'static>> {
/// let function = compile("{x}: 1D6 + {x}")?;
/// let mut evaluator = Evaluator::new(function);
/// let results = (0..10)
///    .flat_map(|x| evaluator.evaluate(vec![x], &mut rng()))
///    .collect::<Vec<_>>();
/// assert!(results.len() == 10);
/// (0..10).for_each(|i| {
///    let x = i as i32;
///    assert!(1 + x <= results[i].result && results[i].result <= 6 + x);
/// });
/// # Ok(())
/// # }
/// ```
///
/// Compile and optimize a dice expression with environmental variables and
/// evaluate it multiple times:
///
/// ```rust
/// use xdy::{compile, EvaluationError, Evaluator};
/// use rand::rng;
///
/// # fn main() -> Result<(), EvaluationError<'static>> {
/// let function = compile("1D6 + {x}")?;
/// let mut evaluator = Evaluator::new(function);
/// evaluator.bind("x", 3)?;
/// let results = (0..10)
///    .flat_map(|x| evaluator.evaluate(vec![], &mut rng()))
///    .collect::<Vec<_>>();
/// assert!(results.len() == 10);
/// assert!(
///     results.iter().all(|result| 4 <= result.result && result.result <= 9)
/// );
/// # Ok(())
/// # }
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub fn compile(source: &str) -> Result<Function, CompilationError<'_>>
{
	let ast = Parser::parse(source).map_err(CompilationError::ParseError)?;
	Validator::validate(&ast)?;
	let function = Compiler::compile(&ast);
	let optimizer = StandardOptimizer::new(Passes::all());
	let function = optimizer
		.optimize(function)
		.map_err(|_| CompilationError::OptimizationFailed)?;
	Ok(function)
}

/// An error that may occur during compilation of a dice expression.
///
/// # Type parameters
/// - `'src`: The lifetime of the source code that was being compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError<'src>
{
	/// The source code could not be parsed.
	ParseError(ParseError<'src>),

	/// A formal parameter name was declared more than once in the function
	/// signature.
	DuplicateParameter
	{
		/// The [canonical](crate::parser::canonical_name) duplicated parameter
		/// name, borrowed from the source text wherever it is spelled
		/// canonically there.
		name: Cow<'src, str>,

		/// The span of the first occurrence of the name in the parameter list.
		first: SourceSpan,

		/// The span of the duplicate occurrence that triggered the error.
		duplicate: SourceSpan
	},

	/// A [local binding](crate::ast::Binding) uses a name that is already
	/// declared as a formal parameter. Local bindings, formal parameters, and
	/// environment variables all share one namespace per function;
	/// cross-category collisions are not permitted.
	BindingCollidesWithParameter
	{
		/// The [canonical](crate::parser::canonical_name) colliding name,
		/// borrowed from the source text wherever it is spelled canonically
		/// there.
		name: Cow<'src, str>,

		/// The span of the parameter declaration in the function signature.
		parameter: SourceSpan,

		/// The span of the binding-site name that triggered the error.
		binding: SourceSpan
	},

	/// The same name is bound more than once by
	/// [local bindings](crate::ast::Binding) within a single function body. The
	/// language provides a single flat namespace per function, so rebinding is
	/// not permitted.
	DuplicateBinding
	{
		/// The [canonical](crate::parser::canonical_name) rebound name,
		/// borrowed from the source text wherever it is spelled canonically
		/// there.
		name: Cow<'src, str>,

		/// The span of the first binding-site name.
		first: SourceSpan,

		/// The span of the duplicate binding-site name that triggered the
		/// error.
		duplicate: SourceSpan
	},

	/// A [variable reference](crate::ast::Variable) appears lexically before
	/// the [binding](crate::ast::Binding) that introduces its name. References
	/// to a local binding are forward-only, so the binding must precede every
	/// use — including any use inside its own bound expression (i.e.,
	/// self-reference is rejected as use-before-bind).
	UseBeforeBind
	{
		/// The [canonical](crate::parser::canonical_name) name that was
		/// referenced before being bound, borrowed from the source text
		/// wherever it is spelled canonically there.
		name: Cow<'src, str>,

		/// The span of the offending reference.
		reference: SourceSpan,

		/// The span of the binding-site name that appears later in the source.
		binding: SourceSpan
	},

	/// The function could not be optimized.
	OptimizationFailed
}

impl Display for CompilationError<'_>
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		match self
		{
			CompilationError::ParseError(e) =>
			{
				write!(f, "{}", e)
			},
			CompilationError::DuplicateParameter {
				name,
				first,
				duplicate
			} =>
			{
				write!(
					f,
					"duplicate parameter '{}' at {} (first declared at {})",
					name, duplicate, first
				)
			},
			CompilationError::BindingCollidesWithParameter {
				name,
				parameter,
				binding
			} =>
			{
				write!(
					f,
					"local binding '{}' at {} collides with formal \
					 parameter declared at {}",
					name, binding, parameter
				)
			},
			CompilationError::DuplicateBinding {
				name,
				first,
				duplicate
			} =>
			{
				write!(
					f,
					"duplicate local binding '{}' at {} (first bound at {})",
					name, duplicate, first
				)
			},
			CompilationError::UseBeforeBind {
				name,
				reference,
				binding
			} =>
			{
				write!(
					f,
					"reference to '{}' at {} precedes its binding at {}",
					name, reference, binding
				)
			},
			CompilationError::OptimizationFailed =>
			{
				write!(f, "optimization failed")
			}
		}
	}
}

impl Error for CompilationError<'_> {}

////////////////////////////////////////////////////////////////////////////////
//                                 Compiler.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A compiler that walks the abstract syntax tree (AST) and emits intermediate
/// representation (IR) code. The IR represents the body of a single
/// [function](Function).
///
/// The compiler borrows variable names from the AST during generation, then
/// copies them into owned strings when assembling the output [`Function`]. This
/// keeps the generation phase zero-copy while producing a self-contained
/// output.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the AST. Variable names are borrowed
///   from the AST during compilation, then copied into owned strings when
///   assembling the output [`Function`].
pub struct Compiler<'a>
{
	/// The instructions emitted by the compiler.
	instructions: Vec<Instruction>,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The next rolling record to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The arity of the function, i.e., the number of formal parameters.
	arity: usize,

	/// The parameters and external variables, mapped to their register
	/// indices. Local bindings are tracked separately in
	/// [`bindings`](Self::bindings).
	variables: HashMap<&'a str, RegisterIndex>,

	/// [Local bindings](crate::ast::Binding) introduced by `{name}@(expr)`
	/// forms, mapped to the [addressing mode](AddressingMode) of the bound
	/// expression. A binding is stored as whatever
	/// [`visit_expression`](ASTVisitor::visit_expression) produced for its
	/// right-hand side — an [`Immediate`] for a constant RHS, a register for
	/// everything else — so subsequent [references](ast::Variable) resolve to
	/// the same value without reallocating or re-emitting the bound
	/// expression. The [`Validator`] guarantees that binding names are
	/// disjoint from parameter and external names, so the two tables never
	/// need to be consulted together during name resolution.
	bindings: HashMap<&'a str, AddressingMode>
}

impl<'a> Compiler<'a>
{
	/// Compile the specified AST into a [`Function`] in intermediate
	/// representation (IR).
	///
	/// Together with parsing and [validation](Validator::validate), this is
	/// equivalent to calling [`compile_unoptimized()`], but gives the caller
	/// access to the AST between parsing and compilation.
	///
	/// # Parameters
	/// - `ast`: The parsed function definition, which must already have been
	///   [validated](Validator::validate). The compiler does not check it, so
	///   the function that it answers for an AST that fails validation is
	///   meaningless.
	///
	/// # Returns
	/// The compiled function in intermediate representation.
	///
	/// # Examples
	///
	/// ```
	/// use xdy::{Compiler, Parser, Validator};
	///
	/// let ast = Parser::parse("2d6 + 3").unwrap();
	/// Validator::validate(&ast).unwrap();
	/// let function = Compiler::compile(&ast);
	/// assert_eq!(function.arity(), 0);
	/// ```
	pub fn compile(ast: &'a ast::Function<'_>) -> Function
	{
		let mut compiler = Self {
			instructions: Vec::new(),
			next_register: RegisterIndex(0),
			next_rolling_record: RollingRecordIndex(0),
			arity: 0,
			variables: HashMap::new(),
			bindings: HashMap::new()
		};
		let _ = ast.accept(&mut compiler);
		compiler.finish()
	}

	/// Assemble the output [`Function`] from the accumulated state.
	///
	/// # Returns
	/// The compiled function.
	fn finish(self) -> Function
	{
		let mut parameters = Vec::new();
		let mut externals = Vec::new();
		for (name, register) in &self.variables
		{
			match register.0 >= self.arity
			{
				false => parameters.push((name, register)),
				true => externals.push((name, register))
			}
		}
		parameters.sort_by_key(|(_, register)| register.0);
		externals.sort_by_key(|(_, register)| register.0);
		let parameters = parameters
			.into_iter()
			.map(|(name, _)| name.to_string())
			.collect();
		let externals = externals
			.into_iter()
			.map(|(name, _)| name.to_string())
			.collect();
		Function {
			parameters,
			externals,
			register_count: self.next_register.0,
			rolling_record_count: self.next_rolling_record.0,
			instructions: self.instructions
		}
	}

	/// Get the register index for the specified variable, allocating a new
	/// register if necessary.
	///
	/// # Parameters
	/// - `name`: The variable name.
	///
	/// # Returns
	/// The register index for the variable.
	fn variable(&mut self, name: &'a str) -> RegisterIndex
	{
		match self.variables.get(name)
		{
			Some(&register) => register,
			None =>
			{
				let register = self.allocate_register();
				self.variables.insert(name, register);
				register
			}
		}
	}

	/// Allocate a new register.
	///
	/// # Returns
	/// The index of the newly allocated register.
	#[inline]
	fn allocate_register(&mut self) -> RegisterIndex
	{
		self.next_register.allocate()
	}

	/// Allocate a new rolling record.
	///
	/// # Returns
	/// The index of the newly allocated rolling record.
	#[inline]
	fn allocate_rolling_record(&mut self) -> RollingRecordIndex
	{
		self.next_rolling_record.allocate()
	}

	/// Emit an instruction.
	///
	/// # Parameters
	/// - `instruction`: The instruction to emit.
	#[inline]
	fn emit(&mut self, instruction: Instruction)
	{
		self.instructions.push(instruction);
	}

	/// Generate IR for a binary arithmetic expression.
	///
	/// # Parameters
	/// - `op1`: The left operand.
	/// - `op2`: The right operand.
	/// - `constructor`: The instruction constructor.
	///
	/// # Returns
	/// The [`AddressingMode`] for the result register.
	fn generate_binary(
		&mut self,
		op1: AddressingMode,
		op2: AddressingMode,
		constructor: fn(
			RegisterIndex,
			AddressingMode,
			AddressingMode
		) -> Instruction
	) -> AddressingMode
	{
		let dest = self.allocate_register();
		self.emit(constructor(dest, op1, op2));
		dest.into()
	}
}

////////////////////////////////////////////////////////////////////////////////
//                          ASTVisitor for Compiler.                          //
////////////////////////////////////////////////////////////////////////////////

impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for Compiler<'a>
{
	type Output = AddressingMode;
	type Error = Infallible;

	fn enter_function(
		&mut self,
		node: &'a ast::Function<'src>
	) -> Result<(), Infallible>
	{
		// Register formal parameters first, in declaration order.
		if let Some(ref parameters) = node.parameters
		{
			for param in parameters
			{
				self.variable(&param.name);
			}
			self.arity = self.variables.len();
		}
		// Discover local binding names before external discovery so that
		// `{x}` inside `{x}@(...) + {x}` is not misclassified as an external.
		// The [`Validator`] has already guaranteed that binding names are
		// disjoint from parameter names and that every reference lexically
		// follows its binding, so a name present in this set belongs to a
		// local binding and not to the external environment.
		let binding_names = collect_binding_names(&node.body);
		// Discover and register external variables before generating the body,
		// so that their register allocation order is deterministic
		// (depth-first, left-to-right through the AST).
		let externals = discover_externals(&node.body);
		for external in externals
		{
			if !binding_names.contains(external)
			{
				self.variable(external);
			}
		}
		Ok(())
	}

	fn visit_function(
		&mut self,
		_node: &'a ast::Function<'src>,
		body: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		self.emit(Instruction::r#return(body));
		Ok(body)
	}

	fn visit_group(
		&mut self,
		_node: &'a ast::Group<'src>,
		expression: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(expression)
	}

	fn visit_constant(
		&mut self,
		node: &'a Constant
	) -> Result<AddressingMode, Infallible>
	{
		Ok(Immediate(node.value).into())
	}

	fn visit_variable(
		&mut self,
		node: &'a ast::Variable<'src>
	) -> Result<AddressingMode, Infallible>
	{
		// Local bindings take precedence over the parameter/external table:
		// the [`Validator`] guarantees disjoint namespaces, so at most one
		// match is possible, and consulting bindings first avoids allocating
		// a spurious register for a name that has already been bound.
		if let Some(&addr) = self.bindings.get(&*node.name)
		{
			return Ok(addr);
		}
		let register = self.variable(&node.name);
		Ok(register.into())
	}

	fn visit_binding(
		&mut self,
		node: &'a Binding<'src>,
		expression: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		// The bound expression has already been compiled, and a rolling record
		// coerced to its sum register, so the binding always captures the
		// integer main effect. The resulting [addressing mode](AddressingMode)
		// — a register for derived values, an immediate for a constant RHS —
		// becomes the single shared source for every subsequent
		// [reference](ast::Variable), which is how "single-evaluation"
		// semantics fall out of linear IR without an explicit marker.
		self.bindings.insert(&node.name, expression);
		Ok(expression)
	}

	fn visit_range(
		&mut self,
		_node: &'a ast::Range<'src>,
		start: AddressingMode,
		end: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_range(dest, start, end));
		let sum = self.allocate_register();
		self.emit(Instruction::sum_rolling_record(sum, dest));
		Ok(sum.into())
	}

	fn visit_standard_dice(
		&mut self,
		_node: &'a ast::StandardDice<'src>,
		count: AddressingMode,
		faces: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_standard_dice(dest, count, faces));
		Ok(dest.into())
	}

	fn visit_custom_dice(
		&mut self,
		node: &'a ast::CustomDice<'src>,
		count: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_custom_dice(
			dest,
			count,
			node.faces.clone()
		));
		Ok(dest.into())
	}

	fn visit_drop_lowest(
		&mut self,
		_node: &'a ast::DropLowest<'src>,
		dice: AddressingMode,
		drop: Option<AddressingMode>
	) -> Result<AddressingMode, Infallible>
	{
		let record: RollingRecordIndex = dice
			.try_into()
			.expect("dice visitor must return RollingRecord");
		let count = drop.unwrap_or(Immediate(1).into());
		self.emit(Instruction::drop_lowest(record, count));
		Ok(record.into())
	}

	fn visit_drop_highest(
		&mut self,
		_node: &'a ast::DropHighest<'src>,
		dice: AddressingMode,
		drop: Option<AddressingMode>
	) -> Result<AddressingMode, Infallible>
	{
		let record: RollingRecordIndex = dice
			.try_into()
			.expect("dice visitor must return RollingRecord");
		let count = drop.unwrap_or(Immediate(1).into());
		self.emit(Instruction::drop_highest(record, count));
		Ok(record.into())
	}

	fn visit_add(
		&mut self,
		_node: &'a ast::Add<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::add))
	}

	fn visit_sub(
		&mut self,
		_node: &'a ast::Sub<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::sub))
	}

	fn visit_mul(
		&mut self,
		_node: &'a ast::Mul<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::mul))
	}

	fn visit_div(
		&mut self,
		_node: &'a ast::Div<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::div))
	}

	fn visit_mod(
		&mut self,
		_node: &'a ast::Mod<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::r#mod))
	}

	fn visit_exp(
		&mut self,
		_node: &'a ast::Exp<'src>,
		left: AddressingMode,
		right: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(left, right, Instruction::exp))
	}

	fn visit_neg(
		&mut self,
		node: &'a ast::Neg<'src>,
		operand: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		// Fold negation of constants into a single immediate. Visiting the
		// constant emitted nothing, so there is nothing to discard.
		if let Expression::Constant(Constant { value, .. }) =
			node.operand.as_ref()
		{
			return Ok(Immediate(value.saturating_neg()).into());
		}
		let dest = self.allocate_register();
		self.emit(Instruction::neg(dest, operand));
		Ok(dest.into())
	}

	fn visit_expression(
		&mut self,
		_node: &'a Expression<'src>,
		output: AddressingMode
	) -> Result<AddressingMode, Infallible>
	{
		// Every expression slot consumes an integer, so reduce an unsummed
		// rolling record to its sum at once, before any sibling is compiled.
		// Only the dice beneath a drop clause, which is not an expression slot,
		// keeps its rolling record, for the drop to operate upon.
		match output
		{
			AddressingMode::RollingRecord(record) =>
			{
				let sum = self.allocate_register();
				self.emit(Instruction::sum_rolling_record(sum, record));
				Ok(sum.into())
			},
			other => Ok(other)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        External variable discovery.                        //
////////////////////////////////////////////////////////////////////////////////

/// Discover all external variable references in the given expression, in
/// depth-first, left-to-right order. This ensures deterministic register
/// allocation.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the expression.
///
/// # Parameters
/// - `expr`: The expression to search.
///
/// # Returns
/// The variable names, in discovery order, with duplicates included (the caller
/// is expected to deduplicate via the variable map).
fn discover_externals<'a>(expr: &'a Expression<'_>) -> Vec<&'a str>
{
	let mut externals: Vec<&'a str> = Vec::new();
	for event in Walk::new(Node::Expression(expr))
	{
		// The name of a binding is not a free variable — it is introduced by
		// the binding, not referenced from outside — but any
		// [references](ast::Variable) inside the bound expression may still be
		// externals, and the walk reaches them in turn.
		if let Event::Enter(Node::Expression(Expression::Variable(v))) = event
		{
			externals.push(&v.name);
		}
	}
	externals
}

/// Collect the names introduced by every [local binding](crate::ast::Binding)
/// anywhere in `expr`. Used by the [compiler](Compiler) to exclude binding
/// names from external-variable discovery — the [`Validator`] has already
/// rejected duplicate bindings, so duplicates here would indicate an
/// inconsistency in the pipeline and are merely deduplicated by the set.
///
/// # Type parameters
/// - `'a`: The lifetime of the borrow of the expression.
///
/// # Parameters
/// - `expr`: The expression to walk.
///
/// # Returns
/// The set of binding names discovered in `expr`.
fn collect_binding_names<'a>(expr: &'a Expression<'_>) -> HashSet<&'a str>
{
	let mut names: HashSet<&'a str> = HashSet::new();
	for event in Walk::new(Node::Expression(expr))
	{
		if let Event::Enter(Node::Expression(Expression::Binding(b))) = event
		{
			names.insert(&b.name);
		}
	}
	names
}

/// A function in the intermediate representation. This is the output of the
/// [compiler](Compiler). A function that the compiler makes is always
/// [well formed](Function::validate).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Function
{
	/// The parameters that the function takes.
	pub parameters: Vec<String>,

	/// The external variables that the function uses.
	pub externals: Vec<String>,

	/// The number of registers the function uses.
	pub register_count: usize,

	/// The number of rolling records the function uses.
	pub rolling_record_count: usize,

	/// The instructions that make up the function.
	pub instructions: Vec<Instruction>
}

impl Function
{
	/// Answer the arity of the function.
	///
	/// # Returns
	/// The number of parameters that the function requires.
	#[inline]
	pub fn arity(&self) -> usize { self.parameters.len() }

	/// Check that the function is well formed, as the [compiler](Compiler)
	/// always makes it, so that the [`Evaluator`](crate::Evaluator) never reads
	/// outside its registers or rolling records and every analysis sees the one
	/// answer it expects. The [assembler](crate::Assembler) and deserialization
	/// check every function that they build; check here any function built by
	/// hand.
	///
	/// A function is well formed if it satisfies each of the following, checked
	/// in order:
	///
	/// 1. Every parameter and external variable has a
	///    [canonical](crate::parser::is_canonical_name) name, and no two of
	///    them share a name.
	/// 2. The parameters and external variables, which occupy the first
	///    registers, fit in `register_count` registers.
	/// 3. Every register reference is less than `register_count`, and every
	///    rolling record reference is less than `rolling_record_count`.
	/// 4. Every [`RollCustomDice`](crate::RollCustomDice) has at least one
	///    face.
	/// 5. No operand is a rolling record: rolling records appear only as the
	///    destinations of rolls and drops, the sources of drops, and the
	///    sources of [sums](crate::SumRollingRecord).
	/// 6. The last instruction is a [`Return`](crate::Return), and it is the
	///    only one. Nothing may follow a return, since the answer of the
	///    function is its one return.
	/// 7. The register file has no gaps: every register holds a parameter or an
	///    external variable, or some instruction mentions it.
	/// 8. The rolling record file has no gaps: some instruction mentions every
	///    rolling record.
	///
	/// # Errors
	/// The first [`FunctionError`] found.
	///
	/// # Examples
	/// ```rust
	/// use xdy::{FunctionError, compile};
	///
	/// let mut function = compile("3D6")?;
	/// assert_eq!(function.validate(), Ok(()));
	/// function.instructions.pop();
	/// assert_eq!(function.validate(), Err(FunctionError::MissingReturn));
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn validate(&self) -> Result<(), FunctionError>
	{
		// Names are canonical and distinct.
		let mut first_by_name = HashMap::new();
		for (index, name) in
			self.parameters.iter().chain(&self.externals).enumerate()
		{
			if !is_canonical_name(name)
			{
				return Err(FunctionError::NonCanonicalName {
					name: name.clone(),
					index
				})
			}
			if let Some(&first) = first_by_name.get(name.as_str())
			{
				return Err(FunctionError::DuplicateName {
					name: name.clone(),
					first,
					index
				})
			}
			first_by_name.insert(name.as_str(), index);
		}
		// Parameters and externals occupy the first registers.
		let declared_args = self.parameters.len() + self.externals.len();
		if declared_args > self.register_count
		{
			return Err(FunctionError::InsufficientRegisterCount {
				register_count: self.register_count,
				required: declared_args
			})
		}
		// Collect referenced register and rolling record indices; fault on any
		// out-of-bounds index.
		let mut register_seen = vec![false; self.register_count];
		register_seen[..declared_args].fill(true);
		let mut record_seen = vec![false; self.rolling_record_count];
		for (instruction, inst) in self.instructions.iter().enumerate()
		{
			self.check_instruction(
				inst,
				instruction,
				&mut register_seen,
				&mut record_seen
			)?;
		}
		// The function ends with its only return. Blame a return that anything
		// follows before the absence of a return at the end, since the former
		// pinpoints the defect.
		if let Some(instruction) = self
			.instructions
			.iter()
			.position(|inst| matches!(inst, Instruction::Return(_)))
			&& instruction + 1 < self.instructions.len()
		{
			return Err(FunctionError::EarlyReturn { instruction })
		}
		if !matches!(self.instructions.last(), Some(Instruction::Return(_)))
		{
			return Err(FunctionError::MissingReturn)
		}
		// No gaps in the register file.
		if let Some(gap) = register_seen.iter().position(|seen| !*seen)
		{
			return Err(FunctionError::RegisterGap {
				index: gap,
				register_count: self.register_count
			})
		}
		// No gaps in the rolling record file.
		if let Some(gap) = record_seen.iter().position(|seen| !*seen)
		{
			return Err(FunctionError::RollingRecordGap {
				index: gap,
				rolling_record_count: self.rolling_record_count
			})
		}
		Ok(())
	}

	/// Check that one of the function's instructions mentions only registers
	/// and rolling records that exist, rolls custom dice that have faces, and
	/// reads no rolling record as an operand; and note every register and
	/// rolling record that it mentions.
	///
	/// # Parameters
	/// - `inst`: The instruction.
	/// - `instruction`: The index of the instruction within the function.
	/// - `register_seen`: Whether each register has been mentioned, updated
	///   with the registers that the instruction mentions.
	/// - `record_seen`: Whether each rolling record has been mentioned, updated
	///   with the rolling records that the instruction mentions.
	///
	/// # Errors
	/// * [`RegisterOutOfBounds`](FunctionError::RegisterOutOfBounds) if the
	///   instruction mentions a register that does not exist.
	/// * [`RollingRecordOutOfBounds`](FunctionError::RollingRecordOutOfBounds)
	///   if the instruction mentions a rolling record that does not exist.
	/// * [`FacelessCustomDice`](FunctionError::FacelessCustomDice) if the
	///   instruction rolls custom dice without faces.
	/// * [`UnexpectedRollingRecordOperand`](FunctionError::UnexpectedRollingRecordOperand)
	///   if an operand of the instruction is a rolling record.
	fn check_instruction(
		&self,
		inst: &Instruction,
		instruction: usize,
		register_seen: &mut [bool],
		record_seen: &mut [bool]
	) -> Result<(), FunctionError>
	{
		let check_register = |idx: RegisterIndex,
		                      seen: &mut [bool]|
		 -> Result<(), FunctionError> {
			if idx.0 >= self.register_count
			{
				return Err(FunctionError::RegisterOutOfBounds {
					index: idx.0,
					register_count: self.register_count,
					instruction
				})
			}
			seen[idx.0] = true;
			Ok(())
		};
		let check_record = |idx: RollingRecordIndex,
		                    seen: &mut [bool]|
		 -> Result<(), FunctionError> {
			if idx.0 >= self.rolling_record_count
			{
				return Err(FunctionError::RollingRecordOutOfBounds {
					index: idx.0,
					rolling_record_count: self.rolling_record_count,
					instruction
				})
			}
			seen[idx.0] = true;
			Ok(())
		};
		let check_mode = |mode: AddressingMode,
		                  seen: &mut [bool]|
		 -> Result<(), FunctionError> {
			match mode
			{
				AddressingMode::Immediate(_) => Ok(()),
				AddressingMode::Register(reg) => check_register(reg, seen),
				AddressingMode::RollingRecord(_) =>
				{
					Err(FunctionError::UnexpectedRollingRecordOperand {
						instruction
					})
				},
			}
		};
		match inst
		{
			Instruction::RollRange(inst) =>
			{
				check_record(inst.dest, record_seen)?;
				check_mode(inst.start, register_seen)?;
				check_mode(inst.end, register_seen)?;
			},
			Instruction::RollStandardDice(inst) =>
			{
				check_record(inst.dest, record_seen)?;
				check_mode(inst.count, register_seen)?;
				check_mode(inst.faces, register_seen)?;
			},
			Instruction::RollCustomDice(inst) =>
			{
				check_record(inst.dest, record_seen)?;
				check_mode(inst.count, register_seen)?;
				if inst.faces.is_empty()
				{
					return Err(FunctionError::FacelessCustomDice {
						instruction
					})
				}
			},
			Instruction::DropLowest(inst) =>
			{
				check_record(inst.dest, record_seen)?;
				check_mode(inst.count, register_seen)?;
			},
			Instruction::DropHighest(inst) =>
			{
				check_record(inst.dest, record_seen)?;
				check_mode(inst.count, register_seen)?;
			},
			Instruction::SumRollingRecord(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_record(inst.src, record_seen)?;
			},
			Instruction::Add(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Sub(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Mul(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Div(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Mod(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Exp(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Max(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op1, register_seen)?;
				check_mode(inst.op2, register_seen)?;
			},
			Instruction::Neg(inst) =>
			{
				check_register(inst.dest, register_seen)?;
				check_mode(inst.op, register_seen)?;
			},
			Instruction::Return(inst) =>
			{
				check_mode(inst.src, register_seen)?;
			}
		}
		Ok(())
	}
}

impl Display for Function
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		write!(f, "Function(")?;
		for (i, parameter) in self.parameters.iter().enumerate()
		{
			if i != 0
			{
				write!(f, ", ")?;
			}
			write!(f, "{{{}}}@{}", parameter, i)?;
		}
		writeln!(
			f,
			") r#{} ⚅#{}",
			self.register_count, self.rolling_record_count
		)?;
		write!(f, "\textern[")?;
		for (i, external) in self.externals.iter().enumerate()
		{
			if i != 0
			{
				write!(f, ", ")?;
			}
			write!(f, "{{{}}}@{}", external, i + self.parameters.len())?;
		}
		writeln!(f, "]")?;
		writeln!(f, "\tbody:")?;
		for instruction in &self.instructions
		{
			writeln!(f, "\t\t{}", instruction)?;
		}
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Function errors.                              //
////////////////////////////////////////////////////////////////////////////////

/// The reason that a [`Function`] is not [well formed](Function::validate).
/// Each variant names the invariant that the function violates, and those about
/// a single instruction name its index within the function.
///
/// Any of these would otherwise mislead the [`Evaluator`](crate::Evaluator) or
/// the analyses built on it: an index out of bounds panics, a rolling record
/// operand is unreachable, and an instruction after a return overwrites the
/// answer that the analyses have already weighed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FunctionError
{
	/// A parameter or external variable has a name that is not
	/// [canonical](crate::parser::is_canonical_name), so the function's print
	/// form would not [assemble](crate::Assembler) back.
	NonCanonicalName
	{
		/// The offending name.
		name: String,

		/// The register of the parameter or external variable.
		index: usize
	},

	/// Two parameters or external variables share a name.
	DuplicateName
	{
		/// The shared name.
		name: String,

		/// The register of the first parameter or external variable with the
		/// name.
		first: usize,

		/// The register of the later parameter or external variable with the
		/// name.
		index: usize
	},

	/// The declared register count is too small to hold the parameters and
	/// external variables.
	InsufficientRegisterCount
	{
		/// The declared register count.
		register_count: usize,

		/// The number of registers that the parameters and external variables
		/// require (`parameters.len() + externals.len()`).
		required: usize
	},

	/// An instruction mentions a register beyond the declared register count.
	RegisterOutOfBounds
	{
		/// The offending register index.
		index: usize,

		/// The declared register count.
		register_count: usize,

		/// The index of the offending instruction.
		instruction: usize
	},

	/// An instruction mentions a rolling record beyond the declared rolling
	/// record count.
	RollingRecordOutOfBounds
	{
		/// The offending rolling record index.
		index: usize,

		/// The declared rolling record count.
		rolling_record_count: usize,

		/// The index of the offending instruction.
		instruction: usize
	},

	/// A custom dice instruction has an empty face list. Although the primitive
	/// `roll_custom_dice` defensively treats empty faces as `0` per die, the
	/// source grammar forbids faceless dice, and a function preserves that
	/// contract.
	FacelessCustomDice
	{
		/// The index of the offending instruction.
		instruction: usize
	},

	/// An operand is an [`AddressingMode::RollingRecord`], which the
	/// [`Evaluator`](crate::Evaluator) cannot read as a value. Rolling records
	/// appear only as the destinations of rolls and drops, the sources of
	/// drops, and the sources of [sums](crate::SumRollingRecord).
	UnexpectedRollingRecordOperand
	{
		/// The index of the offending instruction.
		instruction: usize
	},

	/// The function has no [`Return`](crate::Return), perhaps because it has
	/// no instructions at all.
	MissingReturn,

	/// A [`Return`](crate::Return) is not the last instruction. The function
	/// ends with its only return, so instructions that would overwrite its
	/// answer follow this return, whether or not another return ends the
	/// function. Only the first such return is reported.
	EarlyReturn
	{
		/// The index of the offending return.
		instruction: usize
	},

	/// A register within the declared register count holds no parameter or
	/// external variable, and no instruction mentions it — a gap in the
	/// register file.
	RegisterGap
	{
		/// The index of the unmentioned register.
		index: usize,

		/// The declared register count.
		register_count: usize
	},

	/// A rolling record within the declared rolling record count is mentioned
	/// by no instruction — a gap in the rolling record file.
	RollingRecordGap
	{
		/// The index of the unmentioned rolling record.
		index: usize,

		/// The declared rolling record count.
		rolling_record_count: usize
	}
}

impl FunctionError
{
	/// Answer the index of the offending instruction, if the error is about a
	/// single instruction.
	///
	/// # Returns
	/// The index of the offending instruction within the function, or `None`
	/// if the error is about the function as a whole.
	pub fn instruction(&self) -> Option<usize>
	{
		match self
		{
			Self::RegisterOutOfBounds { instruction, .. }
			| Self::RollingRecordOutOfBounds { instruction, .. }
			| Self::FacelessCustomDice { instruction }
			| Self::UnexpectedRollingRecordOperand { instruction }
			| Self::EarlyReturn { instruction } => Some(*instruction),
			Self::NonCanonicalName { .. }
			| Self::DuplicateName { .. }
			| Self::InsufficientRegisterCount { .. }
			| Self::MissingReturn
			| Self::RegisterGap { .. }
			| Self::RollingRecordGap { .. } => None
		}
	}

	/// Describe the error, without the index of the offending instruction.
	/// The [assembler](crate::Assembler) locates the instruction in the source
	/// instead.
	///
	/// # Parameters
	/// - `f`: The formatter.
	///
	/// # Errors
	/// Any error of the formatter.
	pub(crate) fn describe(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			Self::NonCanonicalName { name, index } => write!(
				f,
				"variable @{} is named `{}`, which is not canonical",
				index, name
			),
			Self::DuplicateName { name, first, index } => write!(
				f,
				"variables @{} and @{} are both named `{}` (names must be \
				 distinct)",
				first, index, name
			),
			Self::InsufficientRegisterCount {
				register_count,
				required
			} => write!(
				f,
				"r#{} registers cannot hold the {} parameters and external \
				 variables",
				register_count, required
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
			Self::FacelessCustomDice { .. } =>
			{
				write!(f, "custom dice must have at least one face")
			},
			Self::UnexpectedRollingRecordOperand { .. } =>
			{
				write!(f, "rolling record operand is not permitted here")
			},
			Self::MissingReturn =>
			{
				write!(f, "the function has no return")
			},
			Self::EarlyReturn { .. } => write!(
				f,
				"return is not the last instruction (a function ends with \
				 its only return)"
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
			)
		}
	}
}

impl Display for FunctionError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
	{
		if let Some(instruction) = self.instruction()
		{
			write!(f, "instruction {}: ", instruction)?;
		}
		self.describe(f)
	}
}

impl Error for FunctionError {}

////////////////////////////////////////////////////////////////////////////////
//                               Serialization.                               //
////////////////////////////////////////////////////////////////////////////////

/// A [`Function`] as it deserializes, before it is
/// [validated](Function::validate).
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedFunction
{
	/// The parameters that the function takes.
	parameters: Vec<String>,

	/// The external variables that the function uses.
	externals: Vec<String>,

	/// The number of registers the function uses.
	register_count: usize,

	/// The number of rolling records the function uses.
	rolling_record_count: usize,

	/// The instructions that make up the function.
	instructions: Vec<Instruction>
}

/// Deserializes a function, refusing one that is not
/// [well formed](Function::validate).
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Function
{
	fn deserialize<D: Deserializer<'de>>(
		deserializer: D
	) -> Result<Self, D::Error>
	{
		use serde::de::Error as _;
		let UncheckedFunction {
			parameters,
			externals,
			register_count,
			rolling_record_count,
			instructions
		} = UncheckedFunction::deserialize(deserializer)?;
		let function = Self {
			parameters,
			externals,
			register_count,
			rolling_record_count,
			instructions
		};
		function.validate().map_err(D::Error::custom)?;
		Ok(function)
	}
}
