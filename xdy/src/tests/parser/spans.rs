//! # Source span position test cases
//!
//! Herein are the test cases that verify the source-span positions computed
//! by the parser for every AST node, including primary and dice expressions,
//! arithmetic operators, function and parameter nodes, [`Spanned::span`]
//! variant dispatch, and [`Spanned::untethered`] normalisation.

use crate::{
	ast::*,
	parser::*,
	span::{SourceSpan, Spanned}
};
use pretty_assertions::assert_eq;

////////////////////////////////////////////////////////////////////////////////
//                        Source span position tests.                         //
////////////////////////////////////////////////////////////////////////////////

/// Shorthand for building a [`SourceSpan`] from a pair of offsets.
fn span(start: usize, end: usize) -> SourceSpan { SourceSpan { start, end } }

////////////////////////////////////////////////////////////////////////////////
//                         Primary-expression spans.                          //
////////////////////////////////////////////////////////////////////////////////

/// A [`Constant`]'s span covers exactly its numeric literal.
#[test]
fn test_constant_span()
{
	let (_, result) = constant(Span::new("42")).unwrap();
	assert_eq!(result.span, span(0, 2));
}

/// A negative numeric literal parses directly to a [`Constant`] with a negative
/// value (rather than a [`Neg`] wrapping a positive [`Constant`]), so that
/// `i32::MIN` is representable. The span covers the leading `-` as well as the
/// digits.
#[test]
fn test_negative_constant_span()
{
	let (_, result) = expression(Span::new("-123")).unwrap();
	assert_eq!(result.span(), span(0, 4));
	match result
	{
		Expression::Constant(c) =>
		{
			assert_eq!(c.value, -123);
			assert_eq!(c.span, span(0, 4));
		},
		other => panic!("expected Constant, got {:?}", other)
	}
}

/// A [`Variable`]'s span includes the surrounding braces.
#[test]
fn test_variable_span_includes_braces()
{
	let (_, result) = variable(Span::new("{abc}")).unwrap();
	assert_eq!(result.name, "abc");
	assert_eq!(result.span, span(0, 5));
}

/// A [`Variable`] embedded in a larger expression preserves its brace-inclusive
/// span, and the enclosing binary op takes its span from its children.
#[test]
fn test_variable_span_inside_expression()
{
	let (_, result) = expression(Span::new("{x} + 1")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.span, span(0, 7));
			match add.left.as_ref()
			{
				Expression::Variable(v) => assert_eq!(v.span, span(0, 3)),
				other => panic!("expected Variable, got {:?}", other)
			}
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// A [`Range`]'s span includes both `[` and `]`; the endpoint expressions'
/// spans do not.
#[test]
fn test_range_span_includes_brackets()
{
	let (_, result) = range(Span::new("[1:20]")).unwrap();
	assert_eq!(result.span, span(0, 6));
	assert_eq!(result.start.span(), span(1, 2));
	assert_eq!(result.end.span(), span(3, 5));
}

/// A [`Group`]'s span includes its delimiters; the inner expression's span does
/// not.
#[test]
fn test_group_span_includes_delimiters()
{
	let (_, result) = expression(Span::new("(1 + 2)")).unwrap();
	match result
	{
		Expression::Group(group) =>
		{
			assert_eq!(group.span, span(0, 7));
			assert_eq!(group.expression.span(), span(1, 6));
		},
		other => panic!("expected Group, got {:?}", other)
	}
}

/// Nested [`Group`]s each carry their own brace-inclusive span.
#[test]
fn test_nested_group_spans()
{
	let (_, result) = expression(Span::new("((1+2))")).unwrap();
	match result
	{
		Expression::Group(outer) =>
		{
			assert_eq!(outer.span, span(0, 7));
			match outer.expression.as_ref()
			{
				Expression::Group(inner) =>
				{
					assert_eq!(inner.span, span(1, 6));
					assert_eq!(inner.expression.span(), span(2, 5));
				},
				other => panic!("expected inner Group, got {:?}", other)
			}
		},
		other => panic!("expected outer Group, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                           Dice-expression spans.                           //
////////////////////////////////////////////////////////////////////////////////

/// A bare [`StandardDice`]'s span covers the count, the `d`/`D` operator, and
/// the face count — exactly the source text of the dice expression.
#[test]
fn test_standard_dice_span()
{
	let (_, result) = dice(Span::new("3D6")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 3));
			assert_eq!(sd.count.span(), span(0, 1));
			assert_eq!(sd.faces.span(), span(2, 3));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`StandardDice`] with variable count and variable faces takes its span
/// from the start of the count through the closing brace of the faces.
#[test]
fn test_standard_dice_with_variable_operands_span()
{
	let (_, result) = dice(Span::new("{x}D{y}")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 7));
			assert_eq!(sd.count.span(), span(0, 3));
			assert_eq!(sd.faces.span(), span(4, 7));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`StandardDice`] with a parenthesized count takes its span from the
/// opening `(` of the group through the end of the faces.
#[test]
fn test_standard_dice_with_group_count_span()
{
	let (_, result) = dice(Span::new("(1+2)D6")).unwrap();
	match result
	{
		DiceExpression::Standard(sd) =>
		{
			assert_eq!(sd.span, span(0, 7));
			assert_eq!(sd.count.span(), span(0, 5));
			assert_eq!(sd.faces.span(), span(6, 7));
		},
		other => panic!("expected Standard, got {:?}", other)
	}
}

/// A [`CustomDice`]'s span starts at the count and extends through the closing
/// `]` of the face list.
#[test]
fn test_custom_dice_span()
{
	let (_, result) = dice(Span::new("2D[1,2,3]")).unwrap();
	match result
	{
		DiceExpression::Custom(cd) =>
		{
			assert_eq!(cd.span, span(0, 9));
			assert_eq!(cd.count.span(), span(0, 1));
			assert_eq!(cd.faces, vec![1, 2, 3]);
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// A [`CustomDice`] with negative face values still covers through the closing
/// `]`; face values do not have their own spans but are recorded verbatim.
#[test]
fn test_custom_dice_with_negative_faces_span()
{
	let (_, result) = dice(Span::new("3D[-1,0,1]")).unwrap();
	match result
	{
		DiceExpression::Custom(cd) =>
		{
			assert_eq!(cd.span, span(0, 10));
			assert_eq!(cd.faces, vec![-1, 0, 1]);
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// A [`DropLowest`]'s span extends from the start of the inner dice to the end
/// of the drop clause. The inner [`StandardDice`]'s own span covers only the
/// dice, not the drop clause.
#[test]
fn test_drop_lowest_no_count_span()
{
	let (_, result) = dice(Span::new("3d6 drop lowest")).unwrap();
	match result
	{
		DiceExpression::DropLowest(drop) =>
		{
			assert_eq!(drop.span, span(0, 15));
			match drop.dice.as_ref()
			{
				DiceExpression::Standard(std_dice) =>
				{
					assert_eq!(std_dice.span, span(0, 3));
				},
				other => panic!("expected StandardDice, got {:?}", other)
			}
			assert!(drop.drop.is_none());
		},
		other => panic!("expected DropLowest, got {:?}", other)
	}
}

/// A [`DropLowest`] with an explicit drop count extends its span through that
/// drop count, while the inner dice retains its pre-clause span.
#[test]
fn test_drop_lowest_with_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop lowest 2")).unwrap();
	match result
	{
		DiceExpression::DropLowest(drop) =>
		{
			assert_eq!(drop.span, span(0, 17));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert_eq!(
				drop.drop.as_ref().expect("drop count").span(),
				span(16, 17)
			);
		},
		other => panic!("expected DropLowest, got {:?}", other)
	}
}

/// A [`DropHighest`] without an explicit drop count carries a span extending
/// through the `highest` keyword.
#[test]
fn test_drop_highest_no_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop highest")).unwrap();
	match result
	{
		DiceExpression::DropHighest(drop) =>
		{
			assert_eq!(drop.span, span(0, 16));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert!(drop.drop.is_none());
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// A [`DropHighest`] with an explicit drop count extends its span through that
/// drop count.
#[test]
fn test_drop_highest_with_count_span()
{
	let (_, result) = dice(Span::new("4D6 drop highest 1")).unwrap();
	match result
	{
		DiceExpression::DropHighest(drop) =>
		{
			assert_eq!(drop.span, span(0, 18));
			assert_eq!(drop.dice.span(), span(0, 3));
			assert_eq!(
				drop.drop.as_ref().expect("drop count").span(),
				span(17, 18)
			);
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// Stacked drop clauses produce nested wrappers, each with a span extending
/// through its own clause; inner wrappers keep the pre-outer-clause span.
#[test]
fn test_stacked_drop_clauses_span()
{
	let (_, result) =
		dice(Span::new("8D6 drop lowest 3 drop highest 1")).unwrap();
	match result
	{
		DiceExpression::DropHighest(outer) =>
		{
			assert_eq!(outer.span, span(0, 32));
			match outer.dice.as_ref()
			{
				DiceExpression::DropLowest(inner) =>
				{
					assert_eq!(inner.span, span(0, 17));
					assert_eq!(inner.dice.span(), span(0, 3));
				},
				other => panic!("expected inner DropLowest, got {:?}", other)
			}
		},
		other => panic!("expected outer DropHighest, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        Arithmetic-expression spans.                        //
////////////////////////////////////////////////////////////////////////////////

/// A binary [`Add`] derives its span from its children: start of `left` through
/// end of `right`. Whitespace inside the expression is *included* (because
/// `right.span().end` is past the last operand character), but whitespace
/// *outside* the binary op is not.
#[test]
fn test_add_span_from_children()
{
	let (_, result) = expression(Span::new("1 + 22")).unwrap();
	assert_eq!(result.span(), span(0, 6));
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.left.span(), span(0, 1));
			assert_eq!(add.right.span(), span(4, 6));
			assert_eq!(add.span, span(0, 6));
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// A binary [`Sub`] takes its span from its operands.
#[test]
fn test_sub_span_from_children()
{
	let (_, result) = expression(Span::new("10 - 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Sub(sub)) =>
		{
			assert_eq!(sub.span, span(0, 6));
			assert_eq!(sub.left.span(), span(0, 2));
			assert_eq!(sub.right.span(), span(5, 6));
		},
		other => panic!("expected Sub, got {:?}", other)
	}
}

/// A binary [`Mul`] takes its span from its operands.
#[test]
fn test_mul_span_from_children()
{
	let (_, result) = expression(Span::new("2 * 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Mul(mul)) =>
		{
			assert_eq!(mul.span, span(0, 5));
			assert_eq!(mul.left.span(), span(0, 1));
			assert_eq!(mul.right.span(), span(4, 5));
		},
		other => panic!("expected Mul, got {:?}", other)
	}
}

/// A binary [`Div`] takes its span from its operands.
#[test]
fn test_div_span_from_children()
{
	let (_, result) = expression(Span::new("10 / 2")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Div(div)) =>
		{
			assert_eq!(div.span, span(0, 6));
			assert_eq!(div.left.span(), span(0, 2));
			assert_eq!(div.right.span(), span(5, 6));
		},
		other => panic!("expected Div, got {:?}", other)
	}
}

/// A binary [`Mod`] takes its span from its operands.
#[test]
fn test_mod_span_from_children()
{
	let (_, result) = expression(Span::new("10 % 3")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Mod(r#mod)) =>
		{
			assert_eq!(r#mod.span, span(0, 6));
			assert_eq!(r#mod.left.span(), span(0, 2));
			assert_eq!(r#mod.right.span(), span(5, 6));
		},
		other => panic!("expected Mod, got {:?}", other)
	}
}

/// A binary [`Exp`] takes its span from its operands.
#[test]
fn test_exp_span_from_children()
{
	let (_, result) = expression(Span::new("2 ^ 10")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Exp(exp)) =>
		{
			assert_eq!(exp.span, span(0, 6));
			assert_eq!(exp.left.span(), span(0, 1));
			assert_eq!(exp.right.span(), span(4, 6));
		},
		other => panic!("expected Exp, got {:?}", other)
	}
}

/// Exponentiation is right-associative: `2^3^2` parses as `2 ^ (3 ^ 2)`. The
/// outer [`Exp`] covers the whole expression; the inner [`Exp`] is the right
/// child and carries the trailing sub-span.
#[test]
fn test_exp_right_associative_span()
{
	let (_, result) = expression(Span::new("2^3^2")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Exp(outer)) =>
		{
			assert_eq!(outer.span, span(0, 5));
			assert_eq!(outer.left.span(), span(0, 1));
			match outer.right.as_ref()
			{
				Expression::Arithmetic(ArithmeticExpression::Exp(inner)) =>
				{
					assert_eq!(inner.span, span(2, 5));
					assert_eq!(inner.left.span(), span(2, 3));
					assert_eq!(inner.right.span(), span(4, 5));
				},
				other => panic!("expected inner Exp, got {:?}", other)
			}
		},
		other => panic!("expected outer Exp, got {:?}", other)
	}
}

/// A [`Neg`] of a variable takes its span from the leading `-` through the
/// closing brace of the variable.
#[test]
fn test_neg_of_variable_span()
{
	let (_, result) = expression(Span::new("-{x}")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			assert_eq!(neg.operand.span(), span(1, 4));
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

/// A [`Neg`] of a parenthesized expression takes its span from the leading `-`
/// through the closing `)` of the group.
#[test]
fn test_neg_of_group_span()
{
	let (_, result) = expression(Span::new("-(1+2)")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 6));
			assert_eq!(neg.operand.span(), span(1, 6));
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

/// Double-negation (`-` applied to a negative constant) produces a [`Neg`]
/// wrapping a [`Constant`]. The outer [`Neg`] begins at the leading `-` and
/// extends through the constant.
#[test]
fn test_neg_of_negative_constant_span()
{
	let (_, result) = expression(Span::new("- -1")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			match neg.operand.as_ref()
			{
				Expression::Constant(c) =>
				{
					assert_eq!(c.value, -1);
					assert_eq!(c.span, span(2, 4));
				},
				other => panic!("expected Constant, got {:?}", other)
			}
		},
		other => panic!("expected outer Neg, got {:?}", other)
	}
}

/// A [`Neg`] of a dice expression (`-3D6`) takes its span from the leading `-`
/// through the end of the dice, using the general negation path because
/// [`negative_constant`] bails when followed by `d`/`D`.
#[test]
fn test_neg_of_dice_span()
{
	let (_, result) = expression(Span::new("-3D6")).unwrap();
	match result
	{
		Expression::Arithmetic(ArithmeticExpression::Neg(neg)) =>
		{
			assert_eq!(neg.span, span(0, 4));
			match neg.operand.as_ref()
			{
				Expression::Dice(DiceExpression::Standard(sd)) =>
				{
					assert_eq!(sd.span, span(1, 4));
				},
				other => panic!("expected StandardDice, got {:?}", other)
			}
		},
		other => panic!("expected Neg, got {:?}", other)
	}
}

////////////////////////////////////////////////////////////////////////////////
//                       Function and parameter spans.                        //
////////////////////////////////////////////////////////////////////////////////

/// Each [`Parameter`] carries the span of its identifier, and the enclosing
/// [`Function`] covers the whole source.
#[test]
fn test_parameter_and_function_spans()
{
	let source = "x, y: 1";
	let (_, result) = function(Span::new(source)).unwrap();
	assert_eq!(result.span, span(0, 7));
	let parameters = result.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].span, span(0, 1));
	assert_eq!(parameters[1].span, span(3, 4));
	assert_eq!(result.body.span(), span(6, 7));
}

/// A [`Function`] with no parameters has its span anchored at the start of the
/// body.
#[test]
fn test_function_no_parameters_span()
{
	let (_, result) = function(Span::new("42")).unwrap();
	assert!(result.parameters.is_none());
	assert_eq!(result.span, span(0, 2));
	assert_eq!(result.body.span(), span(0, 2));
}

/// [`Parameter`] spans cover only the identifier, not surrounding whitespace or
/// separators, regardless of how much whitespace the source contains.
#[test]
fn test_parameter_span_excludes_whitespace()
{
	let source = "  abc  ,  def  : 1";
	let (_, result) = function(Span::new(source)).unwrap();
	let parameters = result.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].name, "abc");
	assert_eq!(parameters[0].span, span(2, 5));
	assert_eq!(parameters[1].name, "def");
	assert_eq!(parameters[1].span, span(10, 13));
}

/// [`Parser::parse`] strips leading whitespace before invoking [`function`], so
/// the resulting [`Function`] span begins at the first non-whitespace byte.
/// Trailing whitespace is stripped outside the body, so the span end matches
/// the body end.
#[test]
fn test_parser_parse_strips_outer_whitespace_from_function_span()
{
	let ast = Parser::parse("  1D6   ").unwrap();
	assert_eq!(ast.span, span(2, 5));
	assert_eq!(ast.body.span(), span(2, 5));
}

////////////////////////////////////////////////////////////////////////////////
//                          Spanned::span dispatch.                           //
////////////////////////////////////////////////////////////////////////////////

/// [`Expression::span`] dispatches to the active variant, returning the same
/// span as the contained node would.
#[test]
fn test_expression_span_dispatches_to_variant()
{
	// Group.
	let group = expression(Span::new("(1)")).unwrap().1;
	assert_eq!(group.span(), span(0, 3));
	assert!(matches!(group, Expression::Group(_)));

	// Constant.
	let constant_expr = expression(Span::new("7")).unwrap().1;
	assert_eq!(constant_expr.span(), span(0, 1));
	assert!(matches!(constant_expr, Expression::Constant(_)));

	// Variable.
	let variable_expr = expression(Span::new("{x}")).unwrap().1;
	assert_eq!(variable_expr.span(), span(0, 3));
	assert!(matches!(variable_expr, Expression::Variable(_)));

	// Range.
	let range_expr = expression(Span::new("[1:6]")).unwrap().1;
	assert_eq!(range_expr.span(), span(0, 5));
	assert!(matches!(range_expr, Expression::Range(_)));

	// Dice.
	let dice_expr = expression(Span::new("2D8")).unwrap().1;
	assert_eq!(dice_expr.span(), span(0, 3));
	assert!(matches!(dice_expr, Expression::Dice(_)));

	// Arithmetic.
	let arith_expr = expression(Span::new("1+2")).unwrap().1;
	assert_eq!(arith_expr.span(), span(0, 3));
	assert!(matches!(arith_expr, Expression::Arithmetic(_)));
}

/// [`DiceExpression::span`] dispatches to the active variant.
#[test]
fn test_dice_expression_span_dispatches_to_variant()
{
	let standard = dice(Span::new("2D8")).unwrap().1;
	assert_eq!(standard.span(), span(0, 3));
	assert!(matches!(standard, DiceExpression::Standard(_)));

	let custom = dice(Span::new("2D[1,2]")).unwrap().1;
	assert_eq!(custom.span(), span(0, 7));
	assert!(matches!(custom, DiceExpression::Custom(_)));

	let drop_lowest = dice(Span::new("4D6 drop lowest")).unwrap().1;
	assert_eq!(drop_lowest.span(), span(0, 15));
	assert!(matches!(drop_lowest, DiceExpression::DropLowest(_)));

	let drop_highest = dice(Span::new("4D6 drop highest")).unwrap().1;
	assert_eq!(drop_highest.span(), span(0, 16));
	assert!(matches!(drop_highest, DiceExpression::DropHighest(_)));
}

/// [`ArithmeticExpression::span`] dispatches to the active variant. The parser
/// produces these only wrapped inside [`Expression::Arithmetic`], so we unwrap
/// once to exercise the inner dispatch.
#[test]
fn test_arithmetic_expression_span_dispatches_to_variant()
{
	fn unwrap_arith(expr: Expression<'_>) -> ArithmeticExpression<'_>
	{
		match expr
		{
			Expression::Arithmetic(a) => a,
			other => panic!("expected Arithmetic, got {:?}", other)
		}
	}

	let add = unwrap_arith(expression(Span::new("1+2")).unwrap().1);
	assert_eq!(add.span(), span(0, 3));
	assert!(matches!(add, ArithmeticExpression::Add(_)));

	let sub = unwrap_arith(expression(Span::new("3-1")).unwrap().1);
	assert_eq!(sub.span(), span(0, 3));
	assert!(matches!(sub, ArithmeticExpression::Sub(_)));

	let mul = unwrap_arith(expression(Span::new("2*3")).unwrap().1);
	assert_eq!(mul.span(), span(0, 3));
	assert!(matches!(mul, ArithmeticExpression::Mul(_)));

	let div = unwrap_arith(expression(Span::new("6/2")).unwrap().1);
	assert_eq!(div.span(), span(0, 3));
	assert!(matches!(div, ArithmeticExpression::Div(_)));

	let r#mod = unwrap_arith(expression(Span::new("7%3")).unwrap().1);
	assert_eq!(r#mod.span(), span(0, 3));
	assert!(matches!(r#mod, ArithmeticExpression::Mod(_)));

	let exp = unwrap_arith(expression(Span::new("2^3")).unwrap().1);
	assert_eq!(exp.span(), span(0, 3));
	assert!(matches!(exp, ArithmeticExpression::Exp(_)));

	let neg = unwrap_arith(expression(Span::new("-{x}")).unwrap().1);
	assert_eq!(neg.span(), span(0, 4));
	assert!(matches!(neg, ArithmeticExpression::Neg(_)));
}

/// [`Function::span`] and [`Parameter::span`] expose the struct field through
/// the [`Spanned`] trait. Other tests access the field directly; this one
/// exercises the trait dispatch specifically.
#[test]
fn test_function_and_parameter_spanned_trait_dispatch()
{
	let ast = Parser::parse("a, b: 1").unwrap();
	// Exercise the trait method, not the struct field.
	assert_eq!(Spanned::span(&ast), span(0, 7));
	let params = ast.parameters.as_ref().unwrap();
	assert_eq!(Spanned::span(&params[0]), span(0, 1));
	assert_eq!(Spanned::span(&params[1]), span(3, 4));
}

////////////////////////////////////////////////////////////////////////////////
//                       Spanned::untethered coverage.                        //
////////////////////////////////////////////////////////////////////////////////

/// [`Spanned::untethered`] zeroes every span in a complex AST — including
/// parameter spans, arithmetic operand spans, dice wrapper spans, and nested
/// variable spans — without altering the structure or identifiers.
#[test]
fn test_untethered_zeroes_all_spans()
{
	let ast = Parser::parse("a, b: {a}D6 drop lowest 1 + {b}").unwrap();
	assert_ne!(ast.span, SourceSpan::default());
	let untethered = ast.untethered();

	assert_eq!(untethered.span, SourceSpan::default());
	let parameters = untethered.parameters.as_ref().unwrap();
	assert_eq!(parameters[0].name, "a");
	assert_eq!(parameters[0].span, SourceSpan::default());
	assert_eq!(parameters[1].name, "b");
	assert_eq!(parameters[1].span, SourceSpan::default());

	// Walk the body: Add(DropLowest(Standard(Variable, Constant), Some(1)),
	// Variable).
	match untethered.body
	{
		Expression::Arithmetic(ArithmeticExpression::Add(add)) =>
		{
			assert_eq!(add.span, SourceSpan::default());
			match *add.left
			{
				Expression::Dice(DiceExpression::DropLowest(drop)) =>
				{
					assert_eq!(drop.span, SourceSpan::default());
					let drop_count =
						drop.drop.as_ref().expect("explicit drop count");
					assert_eq!(drop_count.span(), SourceSpan::default());
					match *drop.dice
					{
						DiceExpression::Standard(sd) =>
						{
							assert_eq!(sd.span, SourceSpan::default());
							assert_eq!(sd.count.span(), SourceSpan::default());
							assert_eq!(sd.faces.span(), SourceSpan::default());
						},
						other =>
						{
							panic!("expected Standard, got {:?}", other)
						}
					}
				},
				other => panic!("expected DropLowest, got {:?}", other)
			}
			match *add.right
			{
				Expression::Variable(v) =>
				{
					assert_eq!(v.name, "b");
					assert_eq!(v.span, SourceSpan::default());
				},
				other => panic!("expected Variable, got {:?}", other)
			}
		},
		other => panic!("expected Add, got {:?}", other)
	}
}

/// [`Spanned::untethered`] on a parameter-less function leaves `parameters` as
/// `None` rather than materializing an empty vector.
#[test]
fn test_untethered_preserves_none_parameters()
{
	let ast = Parser::parse("1D6").unwrap();
	let untethered = ast.untethered();
	assert!(untethered.parameters.is_none());
	assert_eq!(untethered.span, SourceSpan::default());
}

/// [`Spanned::untethered`] on a [`CustomDice`] preserves the face vector
/// exactly (faces are raw `i32` values with no span of their own).
#[test]
fn test_untethered_custom_dice_preserves_faces()
{
	let ast = Parser::parse("2D[-1,0,1]").unwrap();
	let untethered = ast.untethered();
	match untethered.body
	{
		Expression::Dice(DiceExpression::Custom(cd)) =>
		{
			assert_eq!(cd.faces, vec![-1, 0, 1]);
			assert_eq!(cd.span, SourceSpan::default());
			assert_eq!(cd.count.span(), SourceSpan::default());
		},
		other => panic!("expected Custom, got {:?}", other)
	}
}

/// [`Spanned::untethered`] on a [`DropHighest`] without an explicit count keeps
/// the `drop` field as `None` (the optional payload is preserved verbatim).
#[test]
fn test_untethered_drop_highest_no_count()
{
	let ast = Parser::parse("4D6 drop highest").unwrap();
	let untethered = ast.untethered();
	match untethered.body
	{
		Expression::Dice(DiceExpression::DropHighest(drop)) =>
		{
			assert!(drop.drop.is_none());
			assert_eq!(drop.span, SourceSpan::default());
		},
		other => panic!("expected DropHighest, got {:?}", other)
	}
}

/// [`Spanned::untethered`] applied twice is idempotent: the second application
/// is a no-op on already-zeroed spans.
#[test]
fn test_untethered_is_idempotent()
{
	let ast = Parser::parse("1 + {x}").unwrap();
	let once = ast.untethered();
	let twice = once.untethered();
	assert_eq!(once, twice);
}

/// A structurally identical AST built by hand with default spans compares equal
/// to the [`Spanned::untethered`] form of the parser-originated AST.
#[test]
fn test_untethered_matches_hand_built_equivalent()
{
	let untethered = Parser::parse("{x} + 1").unwrap().untethered();
	let handbuilt = Function {
		parameters: None,
		body: Expression::Arithmetic(ArithmeticExpression::Add(Add {
			left: Box::new(Expression::Variable(Variable {
				name: "x",
				span: SourceSpan::default()
			})),
			right: Box::new(Expression::Constant(Constant {
				value: 1,
				span: SourceSpan::default()
			})),
			span: SourceSpan::default()
		})),
		span: SourceSpan::default()
	};
	assert_eq!(untethered, handbuilt);
}
