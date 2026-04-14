//! # Parser
//!
//! Herein is the parser for the `xDy` language. [`parse`]
//! is the main entry point, which discards leading whitespace before parsing a
//! [function](Function). The recognized grammar is as follows, in Extended
//! Backus-Naur Form (EBNF), where non-terminals are in lowercase and terminals
//! are in uppercase:
//!
//! ```text
//! function       ::= parameters? expression
//! parameters     ::= parameter (',' parameter)* ':'
//! parameter      ::= IDENTIFIER
//! expression     ::= add_sub
//! add_sub        ::= mul_div_mod (('+' | '-') mul_div_mod)*
//! mul_div_mod    ::= unary (('*' | '/' | '%') unary)*
//! unary          ::= '-' unary | exponent
//! exponent       ::= primary ('^' unary)?
//! primary        ::= range | dice | group | variable | CONSTANT
//! group          ::= '(' expression ')'
//! variable       ::= '{' IDENTIFIER '}'
//! range          ::= '[' expression ':' expression ']'
//! dice           ::= base_dice drop_clause*
//! base_dice      ::= dice_count D_OPERATOR (standard_faces | custom_faces)
//! dice_count     ::= CONSTANT | variable | group
//! standard_faces ::= CONSTANT | variable | group
//! custom_faces   ::= '[' CONSTANT (',' CONSTANT)* ']'
//! drop_clause    ::= 'drop' ('lowest' | 'highest') drop_expression?
//! drop_expression::= CONSTANT | variable | group
//! CONSTANT       ::= '-'? DIGIT+
//! D_OPERATOR     ::= 'd' | 'D'
//! IDENTIFIER     ::= (ALPHA | '_') (ALPHANUMERIC | '_' | '-')*
//! ```
//!
//! The following railroad diagram is generated from the EBNF grammar above:
#![doc = include_str!("../doc/xdy.svg")]
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
pub fn parse(input: &str) -> Result<Function<'_>, ParseError<'_>>
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
		nom::Err::Error(e) => e,
		nom::Err::Failure(e) => e,
		nom::Err::Incomplete(_) => unreachable!()
	})
	.map(|(_, f)| f)
}
