//! # Parser
//!
//! Herein is the parser for the `xDy` language. [`parse`](crate::parser::parse)
//! is the main entry point, which discards leading whitespace before parsing a
//! [function](Function). The recognized grammar is as follows, in Extended
//! Backus-Naur Form (EBNF), where non-terminals are in lowercase and terminals
//! are in uppercase:
//!
//! ```text
//! function ::= [parameters] expression
//! parameters ::= parameter {',' parameter} ':'
//! parameter ::= IDENTIFIER
//! expression ::= add_sub
//! add_sub ::= mul_div_mod {('+' | '-') mul_div_mod}
//! mul_div_mod ::= exponent {('*' | '×' | '/' | '÷' | '%') exponent}
//! exponent ::= unary {'^' unary}
//! unary ::= ('-' unary | primary)
//! primary ::= range | dice | group | CONSTANT | variable
//! group ::= '(' expression ')'
//! variable ::= '{' IDENTIFIER '}'
//! range ::= '[' expression ':' expression ']'
//! dice ::= (standard_dice | custom_dice) (drop_lowest | drop_highest)*
//! standard_dice ::= dice_count D_OPERATOR standard_faces
//! custom_dice ::= dice_count D_OPERATOR custom_faces
//! dice_count ::= CONSTANT | variable | group
//! standard_faces ::= CONSTANT | variable | group
//! custom_faces ::= '[' CONSTANT {',' CONSTANT} ']'
//! drop_lowest ::= 'drop' 'lowest' [drop_expression]
//! drop_highest ::= 'drop' 'highest' [drop_expression]
//! drop_expression ::= CONSTANT | variable | group
//! CONSTANT ::= ['-'] {DIGIT}
//! D_OPERATOR ::= 'd' | 'D'
//! IDENTIFIER ::= {ALPHA1 | '_'} {ALPHANUMBERIC1 | '_' | '-'}
//! DIGIT ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
//! ALPHA1 ::= ? any alphabetic character ?
//! ALPHANUMBERIC1 ::= ? any alphanumeric character ?
//! WHITESPACE ::= ? any whitespace, permitted between tokens ?
//! ```
//!
//! Parsing is based on the [`nom`] library, which provides a combinator-based
//! approach to parsing.

mod combinators;
mod errors;

pub use combinators::*;
pub use errors::*;

use nom::{
	Parser as _, character::complete::multispace0, combinator::all_consuming,
	error::context, sequence::delimited
};

use crate::ast::Function;

////////////////////////////////////////////////////////////////////////////////
//                                  Parser.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Parse a function definition, discarding leading whitespace. This is the
/// intended high-level entry point for the parser. The individual parser
/// combinators are available for low-level uses, but not recommended for
/// most clients.
///
/// # Parameters
/// - `input`: The input text to parse.
///
/// # Returns
/// The parsed function definition.
///
/// # Errors
/// * [`Err`](nom::Err) if the input could not be parsed.
pub fn parse(input: &str) -> Result<Function, ParseError>
{
	let input = Span::new(input);
	all_consuming(delimited(
		multispace0,
		context(FUNCTION_CONTEXT, function),
		multispace0
	))
	.parse_complete(input)
	.map_err(|e| match e
	{
		nom::Err::Error(e) => e.into(),
		nom::Err::Failure(e) => e.into(),
		nom::Err::Incomplete(_) => unreachable!()
	})
	.map(|(_, f)| f)
}
