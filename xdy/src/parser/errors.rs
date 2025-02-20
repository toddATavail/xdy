//! Parse errors
//!
//! Herein is error handling support for the parser, including types and
//! utilities that facilitate the creation of user-friendly error messages.

use std::fmt::{self, Display, Formatter};

////////////////////////////////////////////////////////////////////////////////
//                                  Errors.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The public error type for the parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError
{
	/// The error message.
	pub message: String
}

impl Display for ParseError
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result
	{
		write!(f, "{}", self.message)
	}
}
