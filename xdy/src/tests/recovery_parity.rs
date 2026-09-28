//! # Recovery parity with the parser
//!
//! Herein are the differential tests of the parser's recovery mode over the
//! [corpus](super::corpus), which hold it to the ordinary parser.
//!
//! - With a policy that declines every repair, recovery mode must produce
//!   exactly the result of [`Parser::parse`], and must present exactly one
//!   failure for each input that fails to parse, whose error comprises the
//!   leading entries of the parser's error. This holds the detection of
//!   failures and the computation of their errors to the ordinary parse.
//! - With the [canonical](Canonical) policy, which repairs every failure that
//!   it can, the source with the edits of the repairs applied must parse, to
//!   the function that recovery mode produced. This holds the engine's
//!   continuation after each repair to what a parse of the edited source does.

use pretty_assertions::assert_eq;

use crate::{
	Parser,
	parser::{FailureSite, Recovery, Repair, Site},
	tests::{corpus::corpus, recovery::Canonical}
};

////////////////////////////////////////////////////////////////////////////////
//                                  Parity.                                   //
////////////////////////////////////////////////////////////////////////////////

/// A policy that declines every repair, and records the failures that it was
/// asked to repair.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
#[derive(Debug, Default)]
struct Decline<'src>
{
	/// The failures.
	sites: Vec<FailureSite<'src>>
}

impl<'src> Recovery<'src> for Decline<'src>
{
	fn repair(&mut self, site: &FailureSite<'src>) -> Repair
	{
		self.sites.push(site.clone());
		Repair::Stop
	}
}

/// Ensure that, when its policy declines every repair, recovery mode matches
/// [`Parser::parse`] on every input in the corpus, and presents exactly one
/// failure for each input that fails, with the leading entries of the error.
#[test]
fn test_declining_recovery_matches_parse()
{
	for input in corpus()
	{
		let expected = Parser::parse(input);
		let mut policy = Decline::default();
		let actual = Parser::parse_recovering(input, &mut policy);
		assert_eq!(actual, expected, "input: {:?}", input);
		match expected
		{
			Ok(_) => assert_eq!(policy.sites, vec![], "input: {:?}", input),
			Err(error) =>
			{
				let position = error.errors[0].0.location_offset();
				let leading = error
					.errors
					.iter()
					.take_while(|(span, _)| span.location_offset() == position)
					.cloned()
					.collect::<Vec<_>>();
				assert_eq!(policy.sites.len(), 1, "input: {:?}", input);
				assert_eq!(
					policy.sites[0].error.errors, leading,
					"input: {:?}",
					input
				);
			}
		}
	}
}

/// Ensure that, for every input in the corpus, the source with the edits of
/// the [canonical](Canonical) policy's repairs applied parses to the function
/// that recovery mode produced. Only an input that begins with a comma, which
/// has no repair, may fail to recover.
#[test]
fn test_canonical_recovery_matches_edited_parse()
{
	for input in corpus()
	{
		let mut policy = Canonical::new(input);
		match Parser::parse_recovering(input, &mut policy)
		{
			Ok(recovered) =>
			{
				let corrected = policy.corrected();
				let parsed = Parser::parse(&corrected).unwrap_or_else(|e| {
					panic!(
						"input {:?} corrected to {:?}, which fails: {}",
						input, corrected, e
					)
				});
				assert_eq!(
					recovered.to_string(),
					parsed.to_string(),
					"input {:?} corrected to {:?}",
					input,
					corrected
				);
			},
			Err(_) => assert!(
				matches!(
					policy.sites.last(),
					Some(FailureSite {
						site: Site::LeadingComma,
						..
					})
				),
				"input {:?} failed to recover at {:?}",
				input,
				policy.sites.last()
			)
		}
	}
}
