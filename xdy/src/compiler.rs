//! # Compiler
//!
//! The dice language compiles to an [intermediate representation](crate::ir)
//! (IR). The IR is generated using static single assignment (SSA) form, where
//! all registers have a uniform type, `i32`.

use std::{
	collections::HashMap,
	error::Error,
	fmt::{Display, Formatter},
	num::IntErrorKind
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Query, QueryCursor, StreamingIterator, Tree};

use crate::{
	CanAllocate as _, Optimizer as _, StandardOptimizer,
	ir::{
		AddressingMode, Immediate, Instruction, RegisterIndex,
		RollingRecordIndex
	}
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
/// [`CompilationFailed`](CompilationError::CompilationFailed) if the function
/// could not be compiled.
pub fn compile_unoptimized(source: &str) -> Result<Function, CompilationError>
{
	let language = tree_sitter_xdy::language();
	let mut parser = tree_sitter::Parser::new();
	parser.set_language(&language).unwrap();
	let tree = parser.parse(source, None).unwrap();
	let root = tree.root_node();
	if root.has_error()
	{
		return Err(CompilationError::CompilationFailed)
	}
	Ok(Compiler::new(source.as_bytes()).compile(tree))
}

/// Compile an alleged dice expression into a [function](Function). Optimize the
/// function using the [standard optimizer](StandardOptimizer).
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
///
/// # fn main() -> Result<(), CompilationError> {
/// let function = compile("3D6")?;
/// let mut evaluator = Evaluator::new(function);
/// let results = (0..10)
///     .flat_map(|_| evaluator.evaluate(vec![], &mut rand::thread_rng()))
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
///
/// # fn main() -> Result<(), CompilationError> {
/// let function = compile("x: 1D6 + {x}")?;
/// let mut evaluator = Evaluator::new(function);
/// let results = (0..10)
///    .flat_map(|x| evaluator.evaluate(vec![x], &mut rand::thread_rng()))
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
///
/// # fn main() -> Result<(), EvaluationError<'static>> {
/// let function = compile("1D6 + {x}")?;
/// let mut evaluator = Evaluator::new(function);
/// evaluator.bind("x", 3)?;
/// let results = (0..10)
///    .flat_map(|x| evaluator.evaluate(vec![], &mut rand::thread_rng()))
///    .collect::<Vec<_>>();
/// assert!(results.len() == 10);
/// assert!(
///     results.iter().all(|result| 4 <= result.result && result.result <= 9)
/// );
/// # Ok(())
/// # }
/// ```
pub fn compile(source: &str) -> Result<Function, CompilationError>
{
	let language = tree_sitter_xdy::language();
	let mut parser = tree_sitter::Parser::new();
	parser.set_language(&language).unwrap();
	let tree = parser.parse(source, None).unwrap();
	let root = tree.root_node();
	if root.has_error()
	{
		return Err(CompilationError::CompilationFailed)
	}
	let code_generator = Compiler::new(source.as_bytes());
	let function = code_generator.compile(tree);
	let optimizer = StandardOptimizer::new(Default::default());
	let function = optimizer
		.optimize(function)
		.map_err(|_| CompilationError::OptimizationFailed)?;
	Ok(function)
}

/// An error that may occur during the evaluation of a dice expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationError
{
	/// The function could not be compiled.
	CompilationFailed,

	/// The function could not be optimized.
	OptimizationFailed
}

impl Display for CompilationError
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		match self
		{
			CompilationError::CompilationFailed =>
			{
				write!(f, "compilation failed")
			},
			CompilationError::OptimizationFailed =>
			{
				write!(f, "optimization failed")
			}
		}
	}
}

impl Error for CompilationError {}

////////////////////////////////////////////////////////////////////////////////
//                          Syntax tree visitation.                           //
////////////////////////////////////////////////////////////////////////////////

/// A visitor for the syntax tree produced by the dice language parser. Each
/// method visits a specific type of node in the syntax tree, and is responsible
/// for visiting any interesting children of that node.
///
/// # Type Parameters
/// - `E`: The type of error that can occur while visiting the syntax tree.
pub trait SyntaxTreeVisitor<E>
{
	/// Visit a "function" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "function" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "function" node.
	fn visit_function(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "parameters" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "parameters" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "parameters" node.
	fn visit_parameters(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "parameter" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "parameter" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "parameter" node.
	fn visit_parameter(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "variable" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "variable" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "variable" node.
	fn visit_variable(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "group" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "group" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "group" node.
	fn visit_group(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "range" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "range" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "range" node.
	fn visit_range(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "standard_dice" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "standard_dice" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "standard_dice" node.
	fn visit_standard_dice(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "custom_dice" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "custom_dice" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "custom_dice" node.
	fn visit_custom_dice(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "custom_faces" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "custom_faces" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "custom_faces" node.
	fn visit_custom_faces(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "custom_face" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "custom_face" node.
	///
	/// Visit a "drop_lowest" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "drop_lowest" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "drop_lowest" node.
	fn visit_drop_lowest(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "drop_highest" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "drop_highest" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "drop_highest" node.
	fn visit_drop_highest(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "add" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "add" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "add" node.
	fn visit_add(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "sub" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "sub" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "sub" node.
	fn visit_sub(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "mul" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "mul" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "mul" node.
	fn visit_mul(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "div" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "div" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "div" node.
	fn visit_div(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "mod" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "mod" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "mod" node.
	fn visit_mod(&mut self, node: &Node) -> Result<(), E>;

	/// Visit an "exp" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "exp" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "exp" node.
	fn visit_exp(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "neg" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "neg" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "neg" node.
	fn visit_neg(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "constant" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "constant" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "constant" node.
	fn visit_constant(&mut self, node: &Node) -> Result<(), E>;

	/// Visit a "negative_constant" node in the syntax tree.
	///
	/// # Parameters
	/// - `node`: The "negative_constant" node.
	///
	/// # Errors
	/// An error if one occurs while visiting the "negative_constant" node.
	fn visit_negative_constant(&mut self, node: &Node) -> Result<(), E>;
}

/// Provides the ability to apply a [visitor](SyntaxTreeVisitor).
///
/// # Type Parameters
/// - `V`: The type of visitor to apply.
/// - `E`: The type of error that can occur while visiting.
pub trait CanVisitSyntaxTree<V, E>
where
	V: SyntaxTreeVisitor<E>
{
	/// Apply the specified [visitor](SyntaxTreeVisitor) to the receiver.
	///
	/// # Parameters
	/// - `visitor`: The visitor to apply.
	///
	/// # Errors
	/// An error if one occurs while visiting the receiver.
	fn visit(&self, node: &mut V) -> Result<(), E>;
}

impl<V, E> CanVisitSyntaxTree<V, E> for Node<'_>
where
	V: SyntaxTreeVisitor<E>
{
	fn visit(&self, visitor: &mut V) -> Result<(), E>
	{
		match self.kind()
		{
			"function" => visitor.visit_function(self),
			"parameters" => visitor.visit_parameters(self),
			"parameter" => visitor.visit_parameter(self),
			"variable" => visitor.visit_variable(self),
			"range" => visitor.visit_range(self),
			"group" => visitor.visit_group(self),
			"standard_dice" => visitor.visit_standard_dice(self),
			"custom_dice" => visitor.visit_custom_dice(self),
			"custom_faces" => visitor.visit_custom_faces(self),
			"drop_lowest" => visitor.visit_drop_lowest(self),
			"drop_highest" => visitor.visit_drop_highest(self),
			"add" => visitor.visit_add(self),
			"sub" => visitor.visit_sub(self),
			"mul" => visitor.visit_mul(self),
			"div" => visitor.visit_div(self),
			"mod" => visitor.visit_mod(self),
			"exp" => visitor.visit_exp(self),
			"neg" => visitor.visit_neg(self),
			"constant" => visitor.visit_constant(self),
			"negative_constant" => visitor.visit_negative_constant(self),
			kind => unreachable!("Unexpected node kind: {}", kind)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                              Code generation.                              //
////////////////////////////////////////////////////////////////////////////////

/// A compiler for the dice syntax tree, capable of walking the syntax tree and
/// generating intermediate representation (IR) code. The IR code represents the
/// body of a single function.
#[derive(Debug)]
pub struct Compiler<'src>
{
	/// The source code.
	source: &'src [u8],

	/// The instructions emitted by the code generator.
	instructions: Vec<Instruction>,

	/// The next register to allocate.
	next_register: RegisterIndex,

	/// The next rolling record to allocate.
	next_rolling_record: RollingRecordIndex,

	/// The arity of the function.
	arity: usize,

	/// The parameters and external variables.
	variables: HashMap<String, RegisterIndex>,

	/// The expression map, mapping node identifiers to registers.
	expressions: HashMap<usize, AddressingMode>
}

impl<'src> Compiler<'src>
{
	/// Create a new code generator.
	///
	/// # Parameters
	/// - `source`: The source code.
	///
	/// # Returns
	/// The new code generator.
	pub fn new(source: &'src [u8]) -> Self
	{
		Self {
			source,
			instructions: Vec::new(),
			next_register: RegisterIndex(0),
			next_rolling_record: RollingRecordIndex(0),
			arity: 0,
			variables: HashMap::new(),
			expressions: HashMap::new()
		}
	}

	/// Compile the specified syntax tree into a [function](Function) in
	/// intermediate representation (IR).
	///
	/// # Parameters
	/// - `tree`: The root of the syntax tree.
	///
	/// # Returns
	/// The generated code.
	pub fn compile(mut self, tree: Tree) -> Function
	{
		tree.root_node().visit(&mut self).unwrap();
		let mut parameters = Vec::new();
		let mut externals = Vec::new();
		for binding in &self.variables
		{
			match binding.1.0 >= self.arity
			{
				false => parameters.push(binding),
				true => externals.push(binding)
			}
		}
		parameters.sort_by_key(|(_, register)| **register);
		externals.sort_by_key(|(_, register)| **register);
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

	/// Get the register index for the specified parameter or external variable,
	/// allocating a new register if necessary.
	///
	/// # Parameters
	/// - `parameter`: The parameter.
	///
	/// # Returns
	/// The register index for the specified parameter.
	fn variable(&mut self, parameter: &'_ str) -> RegisterIndex
	{
		match self.variables.get(parameter)
		{
			Some(&register) => register,
			None =>
			{
				let register = self.allocate_register();
				self.variables.insert(parameter.to_string(), register);
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

	/// Memoize the addressing mode for the specified expression.
	///
	/// # Parameters
	/// - `expression`: The expression to memoize.
	/// - `mode`: The addressing mode to push.
	#[inline]
	fn memoize(&mut self, expression: &Node<'_>, mode: AddressingMode)
	{
		let _ = self.expressions.insert(expression.id(), mode);
	}

	/// Get the addressing mode for the specified expression.
	///
	/// # Parameters
	/// - `expression`: The expression.
	///
	/// # Returns
	/// The addressing mode for the specified expression.
	///
	/// # Panics
	/// Panics if the addressing mode for the specified expression is not
	/// memoized.
	#[inline]
	fn expression(&self, expression: &Node<'_>) -> AddressingMode
	{
		self.expressions[&expression.id()]
	}

	/// Emit an instruction.
	///
	/// # Parameters
	/// - `instruction`: The instruction to emit.
	fn emit(&mut self, instruction: Instruction)
	{
		self.instructions.push(instruction);
	}

	/// Find all external variables that are descendants of the specified node.
	///
	/// # Parameters
	/// - `node`: The node.
	///
	/// # Returns
	/// The external variables.
	fn find_externals(&self, node: Node<'_>) -> Vec<&str>
	{
		let language = tree_sitter_xdy::language();
		let query = Query::new(
			&language,
			"(variable identifier: (identifier) @external)"
		)
		.unwrap();
		let mut cursor = QueryCursor::new();
		cursor
			.captures(&query, node, self.source)
			.map(|(capture, _)| {
				assert_eq!(capture.captures.len(), 1);
				let capture = capture.captures[0].node;
				let identifier = capture.utf8_text(self.source).unwrap();
				identifier
			})
			.copied()
			.collect()
	}
}

impl SyntaxTreeVisitor<()> for Compiler<'_>
{
	fn visit_function(&mut self, node: &Node) -> Result<(), ()>
	{
		let parameters = node.child_by_field_name("parameters");
		if let Some(parameters) = parameters
		{
			parameters.visit(self).unwrap();
		}
		let body = node.child_by_field_name("body").unwrap();
		// Register all of the external variables before visiting the body.
		let externals = self
			.find_externals(body)
			.iter()
			.map(|s| s.to_string())
			.collect::<Vec<_>>();
		for external in externals
		{
			self.variable(&external);
		}
		body.visit(self).unwrap();
		// The last instruction in the function is always a return instruction.
		let return_value = self.expression(&body);
		self.emit(Instruction::r#return(return_value));
		Ok(())
	}

	fn visit_parameters(&mut self, node: &Node) -> Result<(), ()>
	{
		for parameter in
			node.children_by_field_name("parameter", &mut node.walk())
		{
			parameter.visit(self).unwrap();
		}
		// The arity of the function is the number of parameters, and the
		// variable map can only contain parameters at this point.
		self.arity = self.variables.len();
		Ok(())
	}

	fn visit_parameter(&mut self, node: &Node) -> Result<(), ()>
	{
		let parameter = node.utf8_text(self.source).unwrap();
		// Allocate a register for the parameter. We don't need the register
		// index, so we're just doing this for the side effect.
		self.variable(parameter);
		Ok(())
	}

	fn visit_variable(&mut self, node: &Node) -> Result<(), ()>
	{
		let identifier = node.child_by_field_name("identifier").unwrap();
		let identifier = identifier.utf8_text(self.source).unwrap();
		let register = self.variable(identifier);
		self.memoize(node, AddressingMode::Register(register));
		Ok(())
	}

	fn visit_group(&mut self, node: &Node) -> Result<(), ()>
	{
		let expression = node.child_by_field_name("expression").unwrap();
		expression.visit(self).unwrap();
		let expression = self.expression(&expression);
		self.memoize(node, expression);
		Ok(())
	}

	fn visit_range(&mut self, node: &Node) -> Result<(), ()>
	{
		// Visit the children first.
		let start = node.child_by_field_name("start").unwrap();
		start.visit(self).unwrap();
		let end = node.child_by_field_name("end").unwrap();
		end.visit(self).unwrap();
		// Emit the instruction to calculate the range. Ranges aren't subject
		// to drop expressions, so we can compute the sum unconditionally.
		let dest = self.allocate_rolling_record();
		let start = self.expression(&start);
		let end = self.expression(&end);
		self.emit(Instruction::roll_range(dest, start, end));
		let sum = self.allocate_register();
		self.emit(Instruction::sum_rolling_record(sum, dest));
		// Record the sum register as the result of the whole range expression.
		self.memoize(node, sum.into());
		Ok(())
	}

	fn visit_standard_dice(&mut self, node: &Node) -> Result<(), ()>
	{
		// Visit the children first.
		let count = node.child_by_field_name("count").unwrap();
		count.visit(self).unwrap();
		let faces = node.child_by_field_name("faces").unwrap();
		faces.visit(self).unwrap();
		// Emit the instruction to roll the dice.
		let dest = self.allocate_rolling_record();
		let count = self.expression(&count);
		let faces = self.expression(&faces);
		self.emit(Instruction::roll_standard_dice(dest, count, faces));
		self.emit_dice_expression_epilogue(node, dest)
	}

	fn visit_custom_dice(&mut self, node: &Node) -> Result<(), ()>
	{
		// Visit the children first.
		let count = node.child_by_field_name("count").unwrap();
		count.visit(self).unwrap();
		let faces = node.child_by_field_name("faces").unwrap();
		faces.visit(self).unwrap();
		// Collect the custom faces.
		let faces = faces
			.children_by_field_name("face", &mut node.walk())
			.map(|face| Immediate::try_from(self.expression(&face)).unwrap().0)
			.collect();
		// Emit the instruction to roll the dice.
		let dest = self.allocate_rolling_record();
		let count = self.expression(&count);
		self.emit(Instruction::roll_custom_dice(dest, count, faces));
		self.emit_dice_expression_epilogue(node, dest)
	}

	fn visit_custom_faces(&mut self, node: &Node) -> Result<(), ()>
	{
		for face in node.children_by_field_name("face", &mut node.walk())
		{
			face.visit(self).unwrap();
		}
		Ok(())
	}

	fn visit_drop_lowest(&mut self, node: &Node) -> Result<(), ()>
	{
		// Visit the children first.
		let dice = node.child_by_field_name("dice").unwrap();
		dice.visit(self).unwrap();
		let drop = if let Some(drop) = node.child_by_field_name("drop")
		{
			drop.visit(self).unwrap();
			self.expression(&drop)
		}
		else
		{
			// If no drop expression is provided, drop one die.
			Immediate(1).into()
		};
		// Emit the instruction to drop the lowest dice.
		let dice = self.expression(&dice).try_into().unwrap();
		self.emit(Instruction::drop_lowest(dice, drop));
		self.emit_dice_expression_epilogue(node, dice)
	}

	fn visit_drop_highest(&mut self, node: &Node) -> Result<(), ()>
	{
		// Visit the children first.
		let dice = node.child_by_field_name("dice").unwrap();
		dice.visit(self).unwrap();
		let drop = if let Some(drop) = node.child_by_field_name("drop")
		{
			drop.visit(self).unwrap();
			self.expression(&drop)
		}
		else
		{
			// If no drop expression is provided, drop one die.
			Immediate(1).into()
		};
		// Emit the instruction to drop the highest dice.
		let dice = self.expression(&dice).try_into().unwrap();
		self.emit(Instruction::drop_highest(dice, drop));
		self.emit_dice_expression_epilogue(node, dice)
	}

	#[inline]
	fn visit_add(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::add)
	}

	#[inline]
	fn visit_sub(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::sub)
	}

	#[inline]
	fn visit_mul(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::mul)
	}

	#[inline]
	fn visit_div(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::div)
	}

	#[inline]
	fn visit_mod(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::r#mod)
	}

	#[inline]
	fn visit_exp(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_binary_expression(node, Instruction::exp)
	}

	fn visit_neg(&mut self, node: &Node) -> Result<(), ()>
	{
		// If the negation is applied to a constant, we must simplify it here,
		// otherwise we lose the opportunity to capture i32::MIN.
		let op = node.child_by_field_name("op").unwrap();
		if op.kind() == "constant"
		{
			// Be sure to include the sign in the constant lexeme.
			self.visit_any_constant(node).unwrap();
			return Ok(());
		}
		// Otherwise, visit the operand normally.
		op.visit(self).unwrap();
		let dest = self.allocate_register();
		let op = self.expression(&op);
		self.emit(Instruction::neg(dest, op));
		self.memoize(node, dest.into());
		Ok(())
	}

	#[inline]
	fn visit_constant(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_any_constant(node)
	}

	#[inline]
	fn visit_negative_constant(&mut self, node: &Node) -> Result<(), ()>
	{
		self.visit_any_constant(node)
	}
}

impl Compiler<'_>
{
	/// Visits any kind of constant node to generate code. Saturate the constant
	/// to the maximum or minimum value if it overflows.
	///
	/// # Parameters
	/// - `node`: The constant node to visit.
	fn visit_any_constant(&mut self, node: &Node) -> Result<(), ()>
	{
		let constant = node.utf8_text(self.source).unwrap();
		let constant = match constant.parse::<i32>()
		{
			Ok(constant) => constant,
			Err(e) if e.kind() == &IntErrorKind::PosOverflow => i32::MAX,
			Err(e) if e.kind() == &IntErrorKind::NegOverflow => i32::MIN,
			Err(_) => unreachable!()
		};
		let constant = Immediate(constant);
		self.memoize(node, constant.into());
		Ok(())
	}

	/// Emit an appropriate epilogue for a dice expression. If the specified
	/// node is inside a drop expression, the result of the dice expression is
	/// the rolling record itself. Otherwise, the result is the sum of the
	/// rolling record.
	fn emit_dice_expression_epilogue(
		&mut self,
		node: &Node,
		record: RollingRecordIndex
	) -> Result<(), ()>
	{
		// Unless we're inside a drop expression, we need to sum the result.
		match node.parent().unwrap().kind()
		{
			"drop_lowest" | "drop_highest" =>
			{
				// Record the rolling record as the result of the dice
				// expression.
				self.memoize(node, record.into());
			},
			_ =>
			{
				let sum = self.allocate_register();
				self.emit(Instruction::sum_rolling_record(sum, record));
				// Record the sum register as the result of the dice expression.
				self.memoize(node, sum.into());
			}
		}
		Ok(())
	}

	/// Visits a binary expression node to generate code.
	///
	/// # Parameters
	/// - `node`: The binary expression node to visit.
	fn visit_binary_expression(
		&mut self,
		node: &Node,
		selector: impl FnOnce(
			RegisterIndex,
			AddressingMode,
			AddressingMode
		) -> Instruction
	) -> Result<(), ()>
	{
		// Visit the children first.
		let op1 = node.child_by_field_name("op1").unwrap();
		op1.visit(self).unwrap();
		let op2 = node.child_by_field_name("op2").unwrap();
		op2.visit(self).unwrap();
		// Emit the instruction to add the values.
		let dest = self.allocate_register();
		let op1 = self.expression(&op1);
		let op2 = self.expression(&op2);
		self.emit(selector(dest, op1, op2));
		// Record the destination register.
		self.memoize(node, dest.into());
		Ok(())
	}
}

impl Display for Compiler<'_>
{
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result
	{
		writeln!(f, "CodeGenerator[")?;
		writeln!(f, "\t{:?},", String::from_utf8_lossy(self.source))?;
		write!(f, "Function(")?;
		let mut first = true;
		for (parameter, register) in self.variables.iter()
		{
			let register = register.0;
			if register < self.arity
			{
				if !first
				{
					write!(f, ", ")?;
				}
				write!(f, "{}@{}", parameter, register)?;
				first = false;
			}
		}
		writeln!(
			f,
			"] r#{} ⚅#{}",
			self.next_register.0, self.next_rolling_record.0
		)?;
		write!(f, "\textern[")?;
		let mut first = true;
		for (external, register) in self.variables.iter()
		{
			let register = register.0;
			if register < self.arity
			{
				if !first
				{
					write!(f, ", ")?;
				}
				write!(f, "{}@{}", external, register)?;
				first = false;
			}
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

/// A function in the intermediate representation. This is the output of a
/// [compiler](Compiler).
#[derive(Debug, Clone, PartialEq, Eq)]
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
