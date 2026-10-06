//! # AST visitor tests
//!
//! Herein are tests for the iterative driver behind
//! [`ASTVisitor`]: the order in which it calls a visitor, the outputs that it
//! hands each method, where it calls
//! [`visit_expression`](ASTVisitor::visit_expression), how it stops on error,
//! and that it survives deep nesting on a small stack.

use pretty_assertions::assert_eq;

use super::ast::{Nesting, nest};
use crate::{Parser, ast::*, support::on_small_stack};

////////////////////////////////////////////////////////////////////////////////
//                                Call order.                                 //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a walk enters every node before its children, visits every
/// node after its children with their outputs in source order, and finishes
/// every expression slot, but not the dice beneath a drop clause, with
/// [`visit_expression`](ASTVisitor::visit_expression).
#[test]
fn test_visitor_order()
{
	let ast = Parser::parse(
		"{x}: {a}@([1:{x}]) + (2D6 drop lowest 1 drop highest) * -3D[1, 2] / 4 \
		 - 5 % 6 ^ 7"
	)
	.unwrap();
	let mut tracer = Tracer::default();
	let output = ast.accept(&mut tracer).unwrap();
	assert_eq!(
		output,
		"fn <<<{a}@(<[<1>:<{x}>]>)> + <<<(<<2>D<6> drop lowest <1> drop \
		 highest>)> * <-<<3>D[1, 2]>>> / <4>>> - <<5> % <<6> ^ <7>>>>"
	);
	assert_eq!(
		tracer.entered,
		[
			"function",
			"sub",
			"add",
			"binding",
			"range",
			"constant",
			"variable",
			"div",
			"mul",
			"group",
			"drop_highest",
			"drop_lowest",
			"standard_dice",
			"constant",
			"constant",
			"constant",
			"neg",
			"custom_dice",
			"constant",
			"constant",
			"mod",
			"constant",
			"exp",
			"constant",
			"constant"
		]
	);
}

/// Ensure that a walk enters and visits keep clauses as it does drop clauses,
/// in order with them, the outer clause first and its count last, and does not
/// finish the dice beneath a keep clause with
/// [`visit_expression`](ASTVisitor::visit_expression).
#[test]
fn test_visitor_order_keeps()
{
	let ast = crate::s_expr::read_s_expr(
		"(function [{x}] (add (keep-highest (drop-lowest (keep-lowest \
		 (standard-dice 5 6) 3)) {x}) 1))"
	)
	.unwrap();
	let mut tracer = Tracer::default();
	let output = ast.accept(&mut tracer).unwrap();
	assert_eq!(
		output,
		"fn <<<5>D<6> keep lowest <3> drop lowest keep highest <{x}>> + <1>>"
	);
	assert_eq!(
		tracer.entered,
		[
			"function",
			"add",
			"keep_highest",
			"drop_lowest",
			"keep_lowest",
			"standard_dice",
			"constant",
			"constant",
			"constant",
			"variable",
			"constant"
		]
	);
}

/// Ensure that a walk enters and visits reroll clauses as it does drop and keep
/// clauses, in order with them, the outer clause first, then its dice, then the
/// operands of its predicate in the order written, and then its count; that it
/// hands [`visit_reroll`](ASTVisitor::visit_reroll) the outputs of those
/// operands in the shape of the predicate and the limit; and that it does not
/// finish the dice beneath a reroll clause with
/// [`visit_expression`](ASTVisitor::visit_expression).
#[test]
fn test_visitor_order_rerolls()
{
	let ast = crate::s_expr::read_s_expr(
		"(function [{x}] (add (reroll (keep-highest (reroll (reroll \
		 (standard-dice 5 6) (eq 1)) (in-range-inclusive (neg 2) {x}) (times \
		 3)) 4) (in [1 2]) (unlimited)) (reroll (custom-dice 2 [-1 1]) (in-range \
		 {x} 7) (unlimited))))"
	)
	.unwrap();
	let mut tracer = Tracer::default();
	let output = ast.accept(&mut tracer).unwrap();
	assert_eq!(
		output,
		"fn <<<5>D<6> reroll = <1> reroll in <-<2>>..=<{x}> <3> times keep \
		 highest <4> reroll in [1, 2] ∞ times> + <<2>D[-1, 1] reroll in \
		 <{x}>..<7> ∞ times>>"
	);
	assert_eq!(
		tracer.entered,
		[
			"function",
			"add",
			"reroll",
			"keep_highest",
			"reroll",
			"reroll",
			"standard_dice",
			"constant",
			"constant",
			"constant",
			"neg",
			"constant",
			"variable",
			"constant",
			"constant",
			"reroll",
			"custom_dice",
			"constant",
			"variable",
			"constant"
		]
	);
}

/// Ensure that a walk from a [`DiceExpression`] or an
/// [`ArithmeticExpression`] does not finish its root with
/// [`visit_expression`](ASTVisitor::visit_expression), since neither fills a
/// slot of type [`Expression`], but that a walk from an [`Expression`] does.
#[test]
fn test_visitor_roots()
{
	let ast = Parser::parse("3D6 drop lowest").unwrap();
	let Expression::Dice(dice) = &ast.body
	else
	{
		unreachable!()
	};
	assert_eq!(
		dice.accept(&mut Tracer::default()).unwrap(),
		"<3>D<6> drop lowest"
	);
	assert_eq!(
		ast.body.accept(&mut Tracer::default()).unwrap(),
		"<<3>D<6> drop lowest>"
	);
	let ast = Parser::parse("1 + 2").unwrap();
	let Expression::Arithmetic(arithmetic) = &ast.body
	else
	{
		unreachable!()
	};
	assert_eq!(
		arithmetic.accept(&mut Tracer::default()).unwrap(),
		"<1> + <2>"
	);
}

/// Ensure that a walk stops at the first error, whether from an `enter_*`
/// hook, a `visit_*` method, or
/// [`visit_expression`](ASTVisitor::visit_expression), and makes no further
/// calls.
#[test]
fn test_visitor_stops_on_error()
{
	let ast = Parser::parse("{a}@(1) + {b} + 2").unwrap();
	for (fail, entered) in [
		("enter binding", &["function", "add", "add", "binding"][..]),
		(
			"visit variable",
			&["function", "add", "add", "binding", "constant", "variable"][..]
		),
		(
			"visit expression",
			&["function", "add", "add", "binding", "constant"][..]
		)
	]
	{
		let mut tracer = Tracer {
			fail: Some(fail),
			..Tracer::default()
		};
		assert_eq!(ast.accept(&mut tracer), Err(fail));
		assert_eq!(tracer.entered, entered, "{}", fail);
		assert_eq!(tracer.calls.last(), Some(&fail));
	}
}

////////////////////////////////////////////////////////////////////////////////
//                               Deep nesting.                                //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that a walk survives a deep chain of every nesting construct on a
/// small stack.
#[test]
#[ignore = "stress: run with just stress"]
fn test_visitor_deep()
{
	on_small_stack(|| {
		for nesting in Nesting::ROTATION.into_iter().chain([Nesting::Mixed])
		{
			let depth = nesting.depth();
			let expression = nest(nesting, depth, 1);
			let mut counter = Counter::default();
			let nodes = expression.accept(&mut counter).unwrap();
			assert!(nodes > depth, "{:?}", nesting);
			assert_eq!(counter.entered, nodes, "{:?}", nesting);
			assert_eq!(counter.visited, nodes, "{:?}", nesting);
		}
	});
}

////////////////////////////////////////////////////////////////////////////////
//                                  Helpers.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A visitor that renders an AST, wrapping in angle brackets the output of
/// every node that [`visit_expression`](ASTVisitor::visit_expression)
/// finishes, and records its calls. It fails at the call named by
/// [`fail`](Self::fail), if any.
#[derive(Default)]
struct Tracer
{
	/// The node types entered, in order.
	entered: Vec<&'static str>,

	/// Every call, in order, e.g., `enter binding` or `visit expression`.
	calls: Vec<&'static str>,

	/// The call at which to fail, if any.
	fail: Option<&'static str>
}

impl Tracer
{
	/// Record an `enter_*` call.
	///
	/// # Parameters
	/// - `node`: The type of node entered.
	/// - `call`: The name of the call.
	///
	/// # Errors
	/// The name of the call, if it is the one at which to fail.
	fn enter(
		&mut self,
		node: &'static str,
		call: &'static str
	) -> Result<(), &'static str>
	{
		self.entered.push(node);
		self.call(call)
	}

	/// Record a call.
	///
	/// # Parameters
	/// - `call`: The name of the call.
	///
	/// # Errors
	/// The name of the call, if it is the one at which to fail.
	fn call(&mut self, call: &'static str) -> Result<(), &'static str>
	{
		self.calls.push(call);
		match self.fail
		{
			Some(fail) if fail == call => Err(call),
			_ => Ok(())
		}
	}

	/// Record a `visit_*` call and render a drop or keep clause.
	///
	/// # Parameters
	/// - `call`: The name of the call.
	/// - `keywords`: The keywords of the clause, e.g., `drop lowest`.
	/// - `dice`: The output of the dice.
	/// - `count`: The output of the count, if any.
	///
	/// # Returns
	/// The rendering.
	///
	/// # Errors
	/// The name of the call, if it is the one at which to fail.
	fn clause(
		&mut self,
		call: &'static str,
		keywords: &str,
		dice: String,
		count: Option<String>
	) -> Result<String, &'static str>
	{
		self.call(call)?;
		Ok(match count
		{
			Some(count) => format!("{} {} {}", dice, keywords, count),
			None => format!("{} {}", dice, keywords)
		})
	}

	/// Record a `visit_*` call and render a reroll clause.
	///
	/// # Parameters
	/// - `dice`: The output of the dice.
	/// - `predicate`: The predicate, holding the outputs of its operands.
	/// - `limit`: The limit, holding the output of its count, if any.
	///
	/// # Returns
	/// The rendering.
	///
	/// # Errors
	/// The name of the call, if it is the one at which to fail.
	fn reroll(
		&mut self,
		dice: String,
		predicate: Predicate<String>,
		limit: Option<Limit<String>>
	) -> Result<String, &'static str>
	{
		self.call("visit reroll")?;
		let predicate = match predicate
		{
			Predicate::Eq(t) => format!("= {}", t),
			Predicate::Ne(t) => format!("≠ {}", t),
			Predicate::Lt(t) => format!("< {}", t),
			Predicate::Le(t) => format!("≤ {}", t),
			Predicate::Gt(t) => format!("> {}", t),
			Predicate::Ge(t) => format!("≥ {}", t),
			Predicate::In(values) =>
			{
				let values =
					values.iter().map(i32::to_string).collect::<Vec<_>>();
				format!("in [{}]", values.join(", "))
			},
			Predicate::InRange { start, end } =>
			{
				format!("in {}..{}", start, end)
			},
			Predicate::InRangeInclusive { start, end } =>
			{
				format!("in {}..={}", start, end)
			}
		};
		let limit = match limit
		{
			None => String::new(),
			Some(Limit::Times(count)) => format!(" {} times", count),
			Some(Limit::Unlimited) => " ∞ times".to_string()
		};
		Ok(format!("{} reroll {}{}", dice, predicate, limit))
	}

	/// Record a `visit_*` call and render a binary operation.
	///
	/// # Parameters
	/// - `call`: The name of the call.
	/// - `operator`: The operator.
	/// - `left`: The output of the left operand.
	/// - `right`: The output of the right operand.
	///
	/// # Returns
	/// The rendering.
	///
	/// # Errors
	/// The name of the call, if it is the one at which to fail.
	fn binary(
		&mut self,
		call: &'static str,
		operator: &str,
		left: String,
		right: String
	) -> Result<String, &'static str>
	{
		self.call(call)?;
		Ok(format!("{} {} {}", left, operator, right))
	}
}

impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for Tracer
{
	type Output = String;
	type Error = &'static str;

	fn enter_function(
		&mut self,
		_node: &'a Function<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("function", "enter function")
	}

	fn enter_group(&mut self, _node: &'a Group<'src>)
	-> Result<(), Self::Error>
	{
		self.enter("group", "enter group")
	}

	fn enter_constant(&mut self, _node: &'a Constant)
	-> Result<(), Self::Error>
	{
		self.enter("constant", "enter constant")
	}

	fn enter_variable(
		&mut self,
		_node: &'a Variable<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("variable", "enter variable")
	}

	fn enter_binding(
		&mut self,
		_node: &'a Binding<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("binding", "enter binding")
	}

	fn enter_range(&mut self, _node: &'a Range<'src>)
	-> Result<(), Self::Error>
	{
		self.enter("range", "enter range")
	}

	fn enter_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("standard_dice", "enter standard_dice")
	}

	fn enter_custom_dice(
		&mut self,
		_node: &'a CustomDice<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("custom_dice", "enter custom_dice")
	}

	fn enter_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("drop_lowest", "enter drop_lowest")
	}

	fn enter_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("drop_highest", "enter drop_highest")
	}

	fn enter_keep_lowest(
		&mut self,
		_node: &'a KeepLowest<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("keep_lowest", "enter keep_lowest")
	}

	fn enter_keep_highest(
		&mut self,
		_node: &'a KeepHighest<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("keep_highest", "enter keep_highest")
	}

	fn enter_reroll(
		&mut self,
		_node: &'a Reroll<'src>
	) -> Result<(), Self::Error>
	{
		self.enter("reroll", "enter reroll")
	}

	fn enter_add(&mut self, _node: &'a Add<'src>) -> Result<(), Self::Error>
	{
		self.enter("add", "enter add")
	}

	fn enter_sub(&mut self, _node: &'a Sub<'src>) -> Result<(), Self::Error>
	{
		self.enter("sub", "enter sub")
	}

	fn enter_mul(&mut self, _node: &'a Mul<'src>) -> Result<(), Self::Error>
	{
		self.enter("mul", "enter mul")
	}

	fn enter_div(&mut self, _node: &'a Div<'src>) -> Result<(), Self::Error>
	{
		self.enter("div", "enter div")
	}

	fn enter_mod(&mut self, _node: &'a Mod<'src>) -> Result<(), Self::Error>
	{
		self.enter("mod", "enter mod")
	}

	fn enter_exp(&mut self, _node: &'a Exp<'src>) -> Result<(), Self::Error>
	{
		self.enter("exp", "enter exp")
	}

	fn enter_neg(&mut self, _node: &'a Neg<'src>) -> Result<(), Self::Error>
	{
		self.enter("neg", "enter neg")
	}

	fn visit_function(
		&mut self,
		_node: &'a Function<'src>,
		body: String
	) -> Result<String, Self::Error>
	{
		self.call("visit function")?;
		Ok(format!("fn {}", body))
	}

	fn visit_group(
		&mut self,
		_node: &'a Group<'src>,
		expression: String
	) -> Result<String, Self::Error>
	{
		self.call("visit group")?;
		Ok(format!("({})", expression))
	}

	fn visit_constant(
		&mut self,
		node: &'a Constant
	) -> Result<String, Self::Error>
	{
		self.call("visit constant")?;
		Ok(node.value.to_string())
	}

	fn visit_variable(
		&mut self,
		node: &'a Variable<'src>
	) -> Result<String, Self::Error>
	{
		self.call("visit variable")?;
		Ok(format!("{{{}}}", node.name))
	}

	fn visit_binding(
		&mut self,
		node: &'a Binding<'src>,
		expression: String
	) -> Result<String, Self::Error>
	{
		self.call("visit binding")?;
		Ok(format!("{{{}}}@({})", node.name, expression))
	}

	fn visit_range(
		&mut self,
		_node: &'a Range<'src>,
		start: String,
		end: String
	) -> Result<String, Self::Error>
	{
		self.call("visit range")?;
		Ok(format!("[{}:{}]", start, end))
	}

	fn visit_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>,
		count: String,
		faces: String
	) -> Result<String, Self::Error>
	{
		self.call("visit standard_dice")?;
		Ok(format!("{}D{}", count, faces))
	}

	fn visit_custom_dice(
		&mut self,
		node: &'a CustomDice<'src>,
		count: String
	) -> Result<String, Self::Error>
	{
		self.call("visit custom_dice")?;
		Ok(format!("{}D{:?}", count, node.faces))
	}

	fn visit_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>,
		dice: String,
		drop: Option<String>
	) -> Result<String, Self::Error>
	{
		self.clause("visit drop_lowest", "drop lowest", dice, drop)
	}

	fn visit_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>,
		dice: String,
		drop: Option<String>
	) -> Result<String, Self::Error>
	{
		self.clause("visit drop_highest", "drop highest", dice, drop)
	}

	fn visit_keep_lowest(
		&mut self,
		_node: &'a KeepLowest<'src>,
		dice: String,
		keep: Option<String>
	) -> Result<String, Self::Error>
	{
		self.clause("visit keep_lowest", "keep lowest", dice, keep)
	}

	fn visit_keep_highest(
		&mut self,
		_node: &'a KeepHighest<'src>,
		dice: String,
		keep: Option<String>
	) -> Result<String, Self::Error>
	{
		self.clause("visit keep_highest", "keep highest", dice, keep)
	}

	fn visit_reroll(
		&mut self,
		_node: &'a Reroll<'src>,
		dice: String,
		predicate: Predicate<String>,
		limit: Option<Limit<String>>
	) -> Result<String, Self::Error>
	{
		self.reroll(dice, predicate, limit)
	}

	fn visit_add(
		&mut self,
		_node: &'a Add<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit add", "+", left, right)
	}

	fn visit_sub(
		&mut self,
		_node: &'a Sub<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit sub", "-", left, right)
	}

	fn visit_mul(
		&mut self,
		_node: &'a Mul<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit mul", "*", left, right)
	}

	fn visit_div(
		&mut self,
		_node: &'a Div<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit div", "/", left, right)
	}

	fn visit_mod(
		&mut self,
		_node: &'a Mod<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit mod", "%", left, right)
	}

	fn visit_exp(
		&mut self,
		_node: &'a Exp<'src>,
		left: String,
		right: String
	) -> Result<String, Self::Error>
	{
		self.binary("visit exp", "^", left, right)
	}

	fn visit_neg(
		&mut self,
		_node: &'a Neg<'src>,
		operand: String
	) -> Result<String, Self::Error>
	{
		self.call("visit neg")?;
		Ok(format!("-{}", operand))
	}

	fn visit_expression(
		&mut self,
		_node: &'a Expression<'src>,
		output: String
	) -> Result<String, Self::Error>
	{
		self.call("visit expression")?;
		Ok(format!("<{}>", output))
	}
}

/// A visitor that counts the nodes of an AST, both as it enters them and as it
/// visits them. The output of each node is the number of nodes in its subtree.
#[derive(Default)]
struct Counter
{
	/// The number of nodes entered.
	entered: usize,

	/// The number of nodes visited.
	visited: usize
}

impl Counter
{
	/// Record an `enter_*` call.
	///
	/// # Returns
	/// `Ok(())`.
	fn enter(&mut self) -> Result<(), ()>
	{
		self.entered += 1;
		Ok(())
	}

	/// Record a `visit_*` call.
	///
	/// # Parameters
	/// - `children`: The outputs of the node's children.
	///
	/// # Returns
	/// The number of nodes in the node's subtree.
	fn visit(&mut self, children: &[usize]) -> Result<usize, ()>
	{
		self.visited += 1;
		Ok(1 + children.iter().sum::<usize>())
	}
}

impl<'a, 'src: 'a> ASTVisitor<'a, 'src> for Counter
{
	type Output = usize;
	type Error = ();

	fn enter_function(&mut self, _node: &'a Function<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_group(&mut self, _node: &'a Group<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_constant(&mut self, _node: &'a Constant) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_variable(&mut self, _node: &'a Variable<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_binding(&mut self, _node: &'a Binding<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_range(&mut self, _node: &'a Range<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_custom_dice(
		&mut self,
		_node: &'a CustomDice<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_keep_lowest(
		&mut self,
		_node: &'a KeepLowest<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_keep_highest(
		&mut self,
		_node: &'a KeepHighest<'src>
	) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_reroll(&mut self, _node: &'a Reroll<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_add(&mut self, _node: &'a Add<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_sub(&mut self, _node: &'a Sub<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_mul(&mut self, _node: &'a Mul<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_div(&mut self, _node: &'a Div<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_mod(&mut self, _node: &'a Mod<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_exp(&mut self, _node: &'a Exp<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn enter_neg(&mut self, _node: &'a Neg<'src>) -> Result<(), ()>
	{
		self.enter()
	}

	fn visit_function(
		&mut self,
		_node: &'a Function<'src>,
		body: usize
	) -> Result<usize, ()>
	{
		self.visit(&[body])
	}

	fn visit_group(
		&mut self,
		_node: &'a Group<'src>,
		expression: usize
	) -> Result<usize, ()>
	{
		self.visit(&[expression])
	}

	fn visit_constant(&mut self, _node: &'a Constant) -> Result<usize, ()>
	{
		self.visit(&[])
	}

	fn visit_variable(&mut self, _node: &'a Variable<'src>)
	-> Result<usize, ()>
	{
		self.visit(&[])
	}

	fn visit_binding(
		&mut self,
		_node: &'a Binding<'src>,
		expression: usize
	) -> Result<usize, ()>
	{
		self.visit(&[expression])
	}

	fn visit_range(
		&mut self,
		_node: &'a Range<'src>,
		start: usize,
		end: usize
	) -> Result<usize, ()>
	{
		self.visit(&[start, end])
	}

	fn visit_standard_dice(
		&mut self,
		_node: &'a StandardDice<'src>,
		count: usize,
		faces: usize
	) -> Result<usize, ()>
	{
		self.visit(&[count, faces])
	}

	fn visit_custom_dice(
		&mut self,
		_node: &'a CustomDice<'src>,
		count: usize
	) -> Result<usize, ()>
	{
		self.visit(&[count])
	}

	fn visit_drop_lowest(
		&mut self,
		_node: &'a DropLowest<'src>,
		dice: usize,
		drop: Option<usize>
	) -> Result<usize, ()>
	{
		self.visit(&[dice, drop.unwrap_or(0)])
	}

	fn visit_drop_highest(
		&mut self,
		_node: &'a DropHighest<'src>,
		dice: usize,
		drop: Option<usize>
	) -> Result<usize, ()>
	{
		self.visit(&[dice, drop.unwrap_or(0)])
	}

	fn visit_keep_lowest(
		&mut self,
		_node: &'a KeepLowest<'src>,
		dice: usize,
		keep: Option<usize>
	) -> Result<usize, ()>
	{
		self.visit(&[dice, keep.unwrap_or(0)])
	}

	fn visit_keep_highest(
		&mut self,
		_node: &'a KeepHighest<'src>,
		dice: usize,
		keep: Option<usize>
	) -> Result<usize, ()>
	{
		self.visit(&[dice, keep.unwrap_or(0)])
	}

	fn visit_reroll(
		&mut self,
		_node: &'a Reroll<'src>,
		dice: usize,
		predicate: Predicate<usize>,
		limit: Option<Limit<usize>>
	) -> Result<usize, ()>
	{
		let operands = predicate.operands().sum::<usize>();
		let count = limit.as_ref().and_then(Limit::operand).unwrap_or(&0);
		self.visit(&[dice, operands, *count])
	}

	fn visit_add(
		&mut self,
		_node: &'a Add<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_sub(
		&mut self,
		_node: &'a Sub<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_mul(
		&mut self,
		_node: &'a Mul<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_div(
		&mut self,
		_node: &'a Div<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_mod(
		&mut self,
		_node: &'a Mod<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_exp(
		&mut self,
		_node: &'a Exp<'src>,
		left: usize,
		right: usize
	) -> Result<usize, ()>
	{
		self.visit(&[left, right])
	}

	fn visit_neg(
		&mut self,
		_node: &'a Neg<'src>,
		operand: usize
	) -> Result<usize, ()>
	{
		self.visit(&[operand])
	}
}
