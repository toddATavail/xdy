//! # Code generator
//!
//! The code generator walks the abstract syntax tree (AST) produced by the
//! [parser](crate::parser) and emits the
//! [intermediate&#32;representation](crate::ir) (IR) consumed by the rest of
//! the compilation pipeline. It is an internal component of the
//! [`Compiler`](crate::Compiler) and is not exposed as part of the public API.

use std::collections::HashMap;

use crate::{
	CanAllocate as _,
	ast::{self, ArithmeticExpression, Constant, DiceExpression, Expression},
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
pub(crate) struct CodeGenerator<'a>
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
	/// # Parameters
	/// - `ast`: The parsed function definition.
	///
	/// # Returns
	/// The compiled function in intermediate representation.
	pub(crate) fn generate(ast: &'a ast::Function<'a>) -> crate::Function
	{
		let mut codegen = Self {
			instructions: Vec::new(),
			next_register: RegisterIndex(0),
			next_rolling_record: RollingRecordIndex(0),
			arity: 0,
			variables: HashMap::new()
		};
		// Register formal parameters first, in declaration order.
		if let Some(ref parameters) = ast.parameters
		{
			for &param in parameters
			{
				codegen.variable(param);
			}
			codegen.arity = codegen.variables.len();
		}
		// Discover and register external variables before generating the body,
		// so that their register allocation order is deterministic
		// (depth-first, left-to-right through the AST).
		let externals = discover_externals(&ast.body);
		for external in externals
		{
			codegen.variable(external);
		}
		// Generate the body.
		let return_value = codegen.generate_expression(&ast.body);
		codegen.emit(Instruction::r#return(return_value));
		// Assemble the output function.
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

	/// Generate IR for an expression.
	///
	/// # Parameters
	/// - `expr`: The expression.
	///
	/// # Returns
	/// The addressing mode for the result of the expression.
	fn generate_expression(
		&mut self,
		expr: &'a Expression<'a>
	) -> AddressingMode
	{
		match expr
		{
			Expression::Constant(c) => self.generate_constant(c),
			Expression::Variable(v) => self.generate_variable(v),
			Expression::Group(g) => self.generate_expression(&g.expression),
			Expression::Range(r) => self.generate_range(r),
			Expression::Dice(d) =>
			{
				let record = self.generate_dice_expression(d);
				let sum = self.allocate_register();
				self.emit(Instruction::sum_rolling_record(sum, record));
				sum.into()
			},
			Expression::Arithmetic(a) => self.generate_arithmetic(a)
		}
	}

	/// Generate IR for a constant.
	///
	/// # Parameters
	/// - `constant`: The constant.
	///
	/// # Returns
	/// The addressing mode for the constant value.
	#[inline]
	fn generate_constant(&self, constant: &Constant) -> AddressingMode
	{
		Immediate(constant.0).into()
	}

	/// Generate IR for a variable reference.
	///
	/// # Parameters
	/// - `variable`: The variable reference.
	///
	/// # Returns
	/// The addressing mode for the variable's register.
	fn generate_variable(
		&mut self,
		variable: &'a ast::Variable<'a>
	) -> AddressingMode
	{
		let register = self.variable(variable.0);
		AddressingMode::Register(register)
	}

	/// Generate IR for a range expression.
	///
	/// # Parameters
	/// - `range`: The range expression.
	///
	/// # Returns
	/// The addressing mode for the result of the range.
	fn generate_range(&mut self, range: &'a ast::Range<'a>) -> AddressingMode
	{
		let start = self.generate_expression(&range.start);
		let end = self.generate_expression(&range.end);
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_range(dest, start, end));
		let sum = self.allocate_register();
		self.emit(Instruction::sum_rolling_record(sum, dest));
		sum.into()
	}

	/// Generate IR for a dice expression. Returns the rolling record index,
	/// leaving it to the caller to decide whether to sum.
	///
	/// # Parameters
	/// - `dice`: The dice expression.
	///
	/// # Returns
	/// The rolling record index holding the dice results.
	fn generate_dice_expression(
		&mut self,
		dice: &'a DiceExpression<'a>
	) -> RollingRecordIndex
	{
		match dice
		{
			DiceExpression::Standard(d) => self.generate_standard_dice(d),
			DiceExpression::Custom(d) => self.generate_custom_dice(d),
			DiceExpression::DropLowest(d) => self.generate_drop_lowest(d),
			DiceExpression::DropHighest(d) => self.generate_drop_highest(d)
		}
	}

	/// Generate IR for a standard dice expression.
	///
	/// # Parameters
	/// - `dice`: The standard dice expression.
	///
	/// # Returns
	/// The rolling record index holding the dice results.
	fn generate_standard_dice(
		&mut self,
		dice: &'a ast::StandardDice<'a>
	) -> RollingRecordIndex
	{
		let count = self.generate_expression(&dice.count);
		let faces = self.generate_expression(&dice.faces);
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_standard_dice(dest, count, faces));
		dest
	}

	/// Generate IR for a custom dice expression.
	///
	/// # Parameters
	/// - `dice`: The custom dice expression.
	///
	/// # Returns
	/// The rolling record index holding the dice results.
	fn generate_custom_dice(
		&mut self,
		dice: &'a ast::CustomDice<'a>
	) -> RollingRecordIndex
	{
		let count = self.generate_expression(&dice.count);
		let dest = self.allocate_rolling_record();
		self.emit(Instruction::roll_custom_dice(
			dest,
			count,
			dice.faces.clone()
		));
		dest
	}

	/// Generate IR for a drop-lowest expression.
	///
	/// # Parameters
	/// - `drop`: The drop-lowest expression.
	///
	/// # Returns
	/// The rolling record index holding the dice results after dropping.
	fn generate_drop_lowest(
		&mut self,
		drop: &'a ast::DropLowest<'a>
	) -> RollingRecordIndex
	{
		let record = self.generate_dice_expression(&drop.dice);
		let count = match &drop.drop
		{
			Some(expr) => self.generate_expression(expr),
			None => Immediate(1).into()
		};
		self.emit(Instruction::drop_lowest(record, count));
		record
	}

	/// Generate IR for a drop-highest expression.
	///
	/// # Parameters
	/// - `drop`: The drop-highest expression.
	///
	/// # Returns
	/// The rolling record index holding the dice results after dropping.
	fn generate_drop_highest(
		&mut self,
		drop: &'a ast::DropHighest<'a>
	) -> RollingRecordIndex
	{
		let record = self.generate_dice_expression(&drop.dice);
		let count = match &drop.drop
		{
			Some(expr) => self.generate_expression(expr),
			None => Immediate(1).into()
		};
		self.emit(Instruction::drop_highest(record, count));
		record
	}

	/// Generate IR for an arithmetic expression.
	///
	/// # Parameters
	/// - `arith`: The arithmetic expression.
	///
	/// # Returns
	/// The addressing mode for the result.
	fn generate_arithmetic(
		&mut self,
		arith: &'a ArithmeticExpression<'a>
	) -> AddressingMode
	{
		match arith
		{
			ArithmeticExpression::Add(a) =>
			{
				self.generate_binary(&a.left, &a.right, Instruction::add)
			},
			ArithmeticExpression::Sub(s) =>
			{
				self.generate_binary(&s.left, &s.right, Instruction::sub)
			},
			ArithmeticExpression::Mul(m) =>
			{
				self.generate_binary(&m.left, &m.right, Instruction::mul)
			},
			ArithmeticExpression::Div(d) =>
			{
				self.generate_binary(&d.left, &d.right, Instruction::div)
			},
			ArithmeticExpression::Mod(m) =>
			{
				self.generate_binary(&m.left, &m.right, Instruction::r#mod)
			},
			ArithmeticExpression::Exp(e) =>
			{
				self.generate_binary(&e.left, &e.right, Instruction::exp)
			},
			ArithmeticExpression::Neg(n) => self.generate_neg(n)
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
	/// The addressing mode for the result.
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
		let op1 = self.generate_expression(left);
		let op2 = self.generate_expression(right);
		let dest = self.allocate_register();
		self.emit(constructor(dest, op1, op2));
		dest.into()
	}

	/// Generate IR for a negation expression. If the operand is a constant,
	/// fold the negation into a single immediate to preserve the ability to
	/// represent `i32::MIN`.
	///
	/// # Parameters
	/// - `neg`: The negation expression.
	///
	/// # Returns
	/// The addressing mode for the result.
	fn generate_neg(&mut self, neg: &'a ast::Neg<'a>) -> AddressingMode
	{
		// Fold negation of constants into a single immediate.
		if let Expression::Constant(Constant(n)) = neg.operand.as_ref()
		{
			return Immediate(n.saturating_neg()).into();
		}
		let op = self.generate_expression(&neg.operand);
		let dest = self.allocate_register();
		self.emit(Instruction::neg(dest, op));
		dest.into()
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
