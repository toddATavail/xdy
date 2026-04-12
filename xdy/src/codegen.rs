//! # Code generator
//!
//! The code generator walks the abstract syntax tree (AST) produced by the
//! [parser](crate::parser) and emits the
//! [intermediate&#32;representation](crate::ir) (IR) consumed by the rest of
//! the compilation pipeline.
//!
//! The [`CodeGenerator`] implements the [`ASTVisitor`] trait, producing an
//! [`AddressingMode`] from each node. Users who want to walk the AST for their
//! own purposes can implement [`ASTVisitor`] directly; the [`CodeGenerator`]
//! serves as the reference implementation.
//!
//! # Usage
//!
//! Most users should use the top-level [`compile()`](crate::compile) and
//! [`evaluate()`](crate::evaluate) functions, which handle the full pipeline
//! automatically. The code generator is exposed for advanced users who want to
//! drive the pipeline manually — e.g., to insert custom AST analysis or
//! transformation passes between parsing and code generation.
//!
//! ```
//! use xdy::{Evaluator, Optimizer, Passes, StandardOptimizer};
//! use xdy::codegen::CodeGenerator;
//!
//! let ast = xdy::parser::parse("2d6 + 3").unwrap();
//! let function = CodeGenerator::generate(&ast);
//! let optimized = StandardOptimizer::new(Passes::all())
//!     .optimize(function)
//!     .unwrap();
//! let mut rng = rand::rng();
//! let evaluation = Evaluator::new(optimized).evaluate([], &mut rng).unwrap();
//! ```

use std::{collections::HashMap, convert::Infallible};

use crate::{
	CanAllocate as _,
	ast::{
		self, ASTVisitor, ArithmeticExpression, Constant, DiceExpression,
		Expression
	},
	ir::{
		AddressingMode, Immediate, Instruction, RegisterIndex,
		RollingRecordIndex
	}
};

////////////////////////////////////////////////////////////////////////////////
//                              Code generator.                               //
////////////////////////////////////////////////////////////////////////////////

/// A code generator that walks the abstract syntax tree (AST) and emits
/// intermediate representation (IR) code. The IR represents the body of a
/// single [function](crate::Function).
///
/// The code generator borrows variable names from the AST during generation,
/// then copies them into owned strings when assembling the output
/// [`Function`](crate::Function). This keeps the generation phase zero-copy
/// while producing a self-contained output.
pub struct CodeGenerator<'a>
{
	/// The instructions emitted by the code generator.
	instructions: Vec<Instruction>,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The next rolling record to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The arity of the function, i.e., the number of formal parameters.
	arity: usize,

	/// The parameters and external variables, mapped to their register
	/// indices.
	variables: HashMap<&'a str, RegisterIndex>
}

impl<'a> CodeGenerator<'a>
{
	/// Generate IR for the specified AST function.
	///
	/// This is equivalent to calling
	/// [`compile_unoptimized()`](crate::compile_unoptimized) after parsing, but
	/// gives the caller access to the AST between parsing and code generation.
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
	/// use xdy::codegen::CodeGenerator;
	///
	/// let ast = xdy::parser::parse("2d6 + 3").unwrap();
	/// let function = CodeGenerator::generate(&ast);
	/// assert_eq!(function.arity(), 0);
	/// ```
	pub fn generate(ast: &'a ast::Function<'a>) -> crate::Function
	{
		let mut codegen = Self {
			instructions: Vec::new(),
			next_register: RegisterIndex(0),
			next_rolling_record: RollingRecordIndex(0),
			arity: 0,
			variables: HashMap::new()
		};
		let _ = codegen.visit_function(ast);
		codegen.finish()
	}

	/// Assemble the output [`Function`](crate::Function) from the accumulated
	/// state.
	///
	/// # Returns
	/// The compiled function.
	fn finish(self) -> crate::Function
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
		crate::Function {
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
	fn accept_expression(&mut self, expr: &'a Expression<'a>)
	-> AddressingMode
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
		left: &'a Expression<'a>,
		right: &'a Expression<'a>,
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
//                       ASTVisitor for CodeGenerator.                        //
////////////////////////////////////////////////////////////////////////////////

impl<'a> ASTVisitor<'a> for CodeGenerator<'a>
{
	type Output = AddressingMode;
	type Error = Infallible;

	fn visit_function(
		&mut self,
		node: &'a ast::Function<'a>
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
		node: &'a ast::Group<'a>
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
		node: &'a ast::Variable<'a>
	) -> Result<AddressingMode, Infallible>
	{
		let register = self.variable(node.0);
		Ok(register.into())
	}

	fn visit_range(
		&mut self,
		node: &'a ast::Range<'a>
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
		node: &'a ast::StandardDice<'a>
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
		node: &'a ast::CustomDice<'a>
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
		node: &'a ast::DropLowest<'a>
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
		node: &'a ast::DropHighest<'a>
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
		node: &'a ast::Add<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::add))
	}

	fn visit_sub(
		&mut self,
		node: &'a ast::Sub<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::sub))
	}

	fn visit_mul(
		&mut self,
		node: &'a ast::Mul<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::mul))
	}

	fn visit_div(
		&mut self,
		node: &'a ast::Div<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::div))
	}

	fn visit_mod(
		&mut self,
		node: &'a ast::Mod<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::r#mod))
	}

	fn visit_exp(
		&mut self,
		node: &'a ast::Exp<'a>
	) -> Result<AddressingMode, Infallible>
	{
		Ok(self.generate_binary(&node.left, &node.right, Instruction::exp))
	}

	fn visit_neg(
		&mut self,
		node: &'a ast::Neg<'a>
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
/// depth-first, left-to-right order. This matches the traversal order of
/// `tree_sitter::QueryCursor`, ensuring deterministic register allocation.
///
/// # Parameters
/// - `expr`: The expression to search.
///
/// # Returns
/// The variable names, in discovery order, with duplicates included (the caller
/// is expected to deduplicate via the variable map).
fn discover_externals<'a>(expr: &'a Expression<'a>) -> Vec<&'a str>
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
fn collect_variables<'a>(expr: &'a Expression<'a>, out: &mut Vec<&'a str>)
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
fn collect_dice_variables<'a>(
	dice: &'a DiceExpression<'a>,
	out: &mut Vec<&'a str>
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

/// Recursively collect variable references from an arithmetic
/// expression.
///
/// # Parameters
/// - `arith`: The arithmetic expression to search.
/// - `out`: The accumulator for variable names.
fn collect_arithmetic_variables<'a>(
	arith: &'a ArithmeticExpression<'a>,
	out: &mut Vec<&'a str>
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
