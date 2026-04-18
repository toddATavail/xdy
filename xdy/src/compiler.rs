//! # Compiler
//!
//! The dice language compiles to an [intermediate representation](crate::ir)
//! (IR). The IR is generated using static single assignment (SSA) form, where
//! all registers have a uniform type, `i32`.
//!
//! The [`Compiler`] implements the [`ASTVisitor`](crate::ast::ASTVisitor)
//! trait, producing an [`AddressingMode`] from each node. Users who want to
//! walk the AST for their own purposes can implement [`ASTVisitor`] directly;
//! the [`Compiler`] serves as the reference implementation.
//!
//! # Usage
//!
//! Most users should use the top-level [`compile()`] and [`evaluate()`]
//! functions, which handle the full pipeline automatically. The compiler is
//! exposed for advanced users who want to drive the pipeline manually — e.g.,
//! to insert custom AST analysis or transformation passes between parsing and
//! code generation.
//!
//! ```
//! use xdy::{Compiler, Evaluator, Optimizer, Passes, StandardOptimizer};
//!
//! let ast = xdy::parser::parse("2d6 + 3").unwrap();
//! let function = Compiler::compile(&ast);
//! let optimized = StandardOptimizer::new(Passes::all())
//!     .optimize(function)
//!     .unwrap();
//! let mut rng = rand::rng();
//! let evaluation = Evaluator::new(optimized).evaluate([], &mut rng).unwrap();
//! ```

use std::{
	collections::HashMap,
	convert::Infallible,
	error::Error,
	fmt::{Display, Formatter}
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
	CanAllocate as _, Optimizer as _, StandardOptimizer,
	ast::{
		self, ASTVisitor, ArithmeticExpression, Constant, DiceExpression,
		Expression
	},
	ir::{
		AddressingMode, Immediate, Instruction, RegisterIndex,
		RollingRecordIndex
	},
	parser::{self, ParseError}
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
/// [`CompilationFailed`](CompilationError::CompilationFailed) if the source
/// code could not be parsed.
pub fn compile_unoptimized(
	source: &str
) -> Result<Function, CompilationError<'_>>
{
	let ast =
		parser::parse(source).map_err(CompilationError::CompilationFailed)?;
	Ok(Compiler::compile(&ast))
}

/// Compile an alleged dice expression into a [function](Function). Optimize the
/// function using the [standard optimizer](StandardOptimizer).
///
/// # Pipeline
///
/// This function drives the full compilation pipeline:
///
/// ```mermaid
/// graph LR
///     A["Source Code<br/><code>&amp;str</code>"] --> B["Parser<br/><code>parser::parse</code>"]
///     B --> C["AST<br/><code>ast::Function</code>"]
///     C --> D["Compiler<br/><code>Compiler::compile</code>"]
///     D --> E["IR<br/><code>Function</code>"]
///     E --> F["Optimizer<br/><code>StandardOptimizer</code>"]
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
/// * [`CompilationFailed`](CompilationError::CompilationFailed) if the function
///   could not be compiled.
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
/// let function = compile("x: 1D6 + {x}")?;
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
	let ast =
		parser::parse(source).map_err(CompilationError::CompilationFailed)?;
	let function = Compiler::compile(&ast);
	let optimizer = StandardOptimizer::new(Default::default());
	let function = optimizer
		.optimize(function)
		.map_err(|_| CompilationError::OptimizationFailed)?;
	Ok(function)
}

/// An error that may occur during compilation of a dice expression.
///
/// # Lifetimes
/// - `'src`: The lifetime of the source code that was being compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError<'src>
{
	/// The source code could not be parsed.
	CompilationFailed(ParseError<'src>),

	/// The function could not be optimized.
	OptimizationFailed
}

impl Display for CompilationError<'_>
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		match self
		{
			CompilationError::CompilationFailed(e) =>
			{
				write!(f, "{}", e)
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
/// # Lifetimes
/// - `'src`: The lifetime of the source text from which the AST was parsed.
///   Variable names are borrowed from the AST (and transitively from the source
///   text) during compilation, then copied into owned strings when assembling
///   the output [`Function`].
pub struct Compiler<'src>
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
	/// indices.
	variables: HashMap<&'src str, RegisterIndex>
}

impl<'src> Compiler<'src>
{
	/// Compile the specified AST into a [`Function`] in intermediate
	/// representation (IR).
	///
	/// This is equivalent to calling [`compile_unoptimized()`] after parsing,
	/// but gives the caller access to the AST between parsing and compilation.
	///
	/// # Parameters
	/// - `ast`: The parsed function definition.
	///
	/// # Returns
	/// The compiled function in intermediate representation.
	///
	/// # Examples
	///
	/// ```
	/// use xdy::Compiler;
	///
	/// let ast = xdy::parser::parse("2d6 + 3").unwrap();
	/// let function = Compiler::compile(&ast);
	/// assert_eq!(function.arity(), 0);
	/// ```
	pub fn compile(ast: &'src ast::Function<'src>) -> Function
	{
		let mut compiler = Self {
			instructions: Vec::new(),
			next_register: RegisterIndex(0),
			next_rolling_record: RollingRecordIndex(0),
			arity: 0,
			variables: HashMap::new()
		};
		let _ = compiler.visit_function(ast);
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
	fn variable(&mut self, name: &'src str) -> RegisterIndex
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

	/// Accept an expression and ensure the result is not an unsummed rolling
	/// record. If the expression produces a
	/// [`RollingRecord`](AddressingMode::RollingRecord), emit a
	/// [`SumRollingRecord`](crate::ir::SumRollingRecord) instruction to reduce
	/// it to a register.
	///
	/// # Parameters
	/// - `expr`: The expression to accept.
	///
	/// # Returns
	/// The [`AddressingMode`] for the result.
	fn accept_expression(
		&mut self,
		expr: &'src Expression<'src>
	) -> AddressingMode
	{
		let value = expr.accept(self).unwrap();
		match value
		{
			AddressingMode::RollingRecord(record) =>
			{
				let sum = self.allocate_register();
				self.emit(Instruction::sum_rolling_record(sum, record));
				sum.into()
			},
			other => other
		}
	}

	/// Generate IR for a binary arithmetic expression.
	///
	/// # Parameters
	/// - `left`: The left operand.
	/// - `right`: The right operand.
	/// - `constructor`: The instruction constructor.
	///
	/// # Returns
	/// The [`AddressingMode`] for the result register.
	fn generate_binary(
		&mut self,
		left: &'src Expression<'src>,
		right: &'src Expression<'src>,
		constructor: fn(
			RegisterIndex,
			AddressingMode,
			AddressingMode
		) -> Instruction
	) -> AddressingMode
	{
		let op1 = self.accept_expression(left);
		let op2 = self.accept_expression(right);
		let dest = self.allocate_register();
		self.emit(constructor(dest, op1, op2));
		dest.into()
	}
}

////////////////////////////////////////////////////////////////////////////////
//                          ASTVisitor for Compiler.                          //
////////////////////////////////////////////////////////////////////////////////

impl<'src> ASTVisitor<'src> for Compiler<'src>
{
	type Output = AddressingMode;
	type Error = Infallible;

	fn visit_function(
		&mut self,
		node: &'src ast::Function<'src>
	) -> Result<AddressingMode, Infallible>
	{
		// Register formal parameters first, in declaration order.
		if let Some(ref parameters) = node.parameters
		{
			for &param in parameters
			{
				self.variable(param);
			}
			self.arity = self.variables.len();
		}
		// Discover and register external variables before generating the body,
		// so that their register allocation order is deterministic
		// (depth-first, left-to-right through the AST).
		let externals = discover_externals(&node.body);
		for external in externals
		{
			self.variable(external);
		}
		// Generate the body.
		let return_value = self.accept_expression(&node.body);
		self.emit(Instruction::r#return(return_value));
		Ok(return_value)
	}

	fn visit_group(
		&mut self,
		node: &'src ast::Group<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.accept_expression(&node.expression))
	}

	fn visit_constant(
		&mut self,
		node: &Constant
	) -> Result<AddressingMode, Infallible>
	{
		Ok(Immediate(node.0).into())
	}

	fn visit_variable(
		&mut self,
		node: &'src ast::Variable<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let register = self.variable(node.0);
		Ok(register.into())
	}

	fn visit_range(
		&mut self,
		node: &'src ast::Range<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let start = self.accept_expression(&node.start);
		let end = self.accept_expression(&node.end);
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_range(dest, start, end));
		let sum = self.allocate_register();
		self.emit(Instruction::sum_rolling_record(sum, dest));
		Ok(sum.into())
	}

	fn visit_standard_dice(
		&mut self,
		node: &'src ast::StandardDice<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let count = self.accept_expression(&node.count);
		let faces = self.accept_expression(&node.faces);
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_standard_dice(dest, count, faces));
		Ok(dest.into())
	}

	fn visit_custom_dice(
		&mut self,
		node: &'src ast::CustomDice<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let count = self.accept_expression(&node.count);
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
		node: &'src ast::DropLowest<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let record: RollingRecordIndex = node
			.dice
			.accept(self)
			.unwrap()
			.try_into()
			.expect("dice visitor must return RollingRecord");
		let count = match &node.drop
		{
			Some(expr) => self.accept_expression(expr),
			None => Immediate(1).into()
		};
		self.emit(Instruction::drop_lowest(record, count));
		Ok(record.into())
	}

	fn visit_drop_highest(
		&mut self,
		node: &'src ast::DropHighest<'src>
	) -> Result<AddressingMode, Infallible>
	{
		let record: RollingRecordIndex = node
			.dice
			.accept(self)
			.unwrap()
			.try_into()
			.expect("dice visitor must return RollingRecord");
		let count = match &node.drop
		{
			Some(expr) => self.accept_expression(expr),
			None => Immediate(1).into()
		};
		self.emit(Instruction::drop_highest(record, count));
		Ok(record.into())
	}

	fn visit_add(
		&mut self,
		node: &'src ast::Add<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::add))
	}

	fn visit_sub(
		&mut self,
		node: &'src ast::Sub<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::sub))
	}

	fn visit_mul(
		&mut self,
		node: &'src ast::Mul<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::mul))
	}

	fn visit_div(
		&mut self,
		node: &'src ast::Div<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::div))
	}

	fn visit_mod(
		&mut self,
		node: &'src ast::Mod<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::r#mod))
	}

	fn visit_exp(
		&mut self,
		node: &'src ast::Exp<'src>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::exp))
	}

	fn visit_neg(
		&mut self,
		node: &'src ast::Neg<'src>
	) -> Result<AddressingMode, Infallible>
	{
		// Fold negation of constants into a single immediate.
		if let Expression::Constant(Constant(n)) = node.operand.as_ref()
		{
			return Ok(Immediate(n.saturating_neg()).into());
		}
		let op = self.accept_expression(&node.operand);
		let dest = self.allocate_register();
		self.emit(Instruction::neg(dest, op));
		Ok(dest.into())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        External variable discovery.                        //
////////////////////////////////////////////////////////////////////////////////

/// Discover all external variable references in the given expression, in
/// depth-first, left-to-right order. This ensures deterministic register
/// allocation.
///
/// # Parameters
/// - `expr`: The expression to search.
///
/// # Returns
/// The variable names, in discovery order, with duplicates included (the caller
/// is expected to deduplicate via the variable map).
fn discover_externals<'src>(expr: &'src Expression<'src>) -> Vec<&'src str>
{
	let mut externals = Vec::new();
	collect_variables(expr, &mut externals);
	externals
}

/// Recursively collect variable references from the given expression.
///
/// # Parameters
/// - `expr`: The expression to search.
/// - `out`: The accumulator for variable names.
fn collect_variables<'src>(
	expr: &'src Expression<'src>,
	out: &mut Vec<&'src str>
)
{
	match expr
	{
		Expression::Variable(v) =>
		{
			out.push(v.0);
		},
		Expression::Group(g) =>
		{
			collect_variables(&g.expression, out);
		},
		Expression::Range(r) =>
		{
			collect_variables(&r.start, out);
			collect_variables(&r.end, out);
		},
		Expression::Dice(d) =>
		{
			collect_dice_variables(d, out);
		},
		Expression::Arithmetic(a) =>
		{
			collect_arithmetic_variables(a, out);
		},
		Expression::Constant(_) =>
		{}
	}
}

/// Recursively collect variable references from a dice expression.
///
/// # Parameters
/// - `dice`: The dice expression to search.
/// - `out`: The accumulator for variable names.
fn collect_dice_variables<'src>(
	dice: &'src DiceExpression<'src>,
	out: &mut Vec<&'src str>
)
{
	match dice
	{
		DiceExpression::Standard(d) =>
		{
			collect_variables(&d.count, out);
			collect_variables(&d.faces, out);
		},
		DiceExpression::Custom(d) =>
		{
			collect_variables(&d.count, out);
		},
		DiceExpression::DropLowest(d) =>
		{
			collect_dice_variables(&d.dice, out);
			if let Some(ref drop) = d.drop
			{
				collect_variables(drop, out);
			}
		},
		DiceExpression::DropHighest(d) =>
		{
			collect_dice_variables(&d.dice, out);
			if let Some(ref drop) = d.drop
			{
				collect_variables(drop, out);
			}
		}
	}
}

/// Recursively collect variable references from an arithmetic expression.
///
/// # Parameters
/// - `arith`: The arithmetic expression to search.
/// - `out`: The accumulator for variable names.
fn collect_arithmetic_variables<'src>(
	arith: &'src ArithmeticExpression<'src>,
	out: &mut Vec<&'src str>
)
{
	match arith
	{
		ArithmeticExpression::Add(a) =>
		{
			collect_variables(&a.left, out);
			collect_variables(&a.right, out);
		},
		ArithmeticExpression::Sub(s) =>
		{
			collect_variables(&s.left, out);
			collect_variables(&s.right, out);
		},
		ArithmeticExpression::Mul(m) =>
		{
			collect_variables(&m.left, out);
			collect_variables(&m.right, out);
		},
		ArithmeticExpression::Div(d) =>
		{
			collect_variables(&d.left, out);
			collect_variables(&d.right, out);
		},
		ArithmeticExpression::Mod(m) =>
		{
			collect_variables(&m.left, out);
			collect_variables(&m.right, out);
		},
		ArithmeticExpression::Exp(e) =>
		{
			collect_variables(&e.left, out);
			collect_variables(&e.right, out);
		},
		ArithmeticExpression::Neg(n) =>
		{
			collect_variables(&n.operand, out);
		}
	}
}

/// A function in the intermediate representation. This is the output of the
/// [compiler](Compiler).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
			write!(f, "{}@{}", parameter, i)?;
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
			write!(f, "{}@{}", external, i + self.parameters.len())?;
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
