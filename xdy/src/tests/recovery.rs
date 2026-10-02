//! # Recovery mode
//!
//! Herein are the tests of the parser's recovery mode, in which the engine
//! consults a [policy](Recovery) wherever the parse cannot continue, and
//! continues as though the source had been edited. The [canonical](Canonical)
//! policy repairs every failure that it can, and records the edits of the
//! source that its repairs correspond to, so the tests can compare the
//! recovered function with the function that the ordinary parser makes of the
//! edited source. The differential tests over the [corpus](super::corpus) are
//! in [`recovery_parity`](super::recovery_parity).
//!
//! Recovery must also take time linear in the length of the input, however
//! many repairs it makes, which the tests check by counting the steps of the
//! engine.

use nom::Input;
use pretty_assertions::assert_eq;

use crate::{
	Parser,
	parser::{
		FailureSite, PLACEHOLDER_FACE, PLACEHOLDER_FACES, PLACEHOLDER_NAME,
		PLACEHOLDER_OPERAND, Recovery, Repair, Site, bare_word, is_token_space,
		steps, unclosed_name_ends
	}
};

////////////////////////////////////////////////////////////////////////////////
//                                 Policies.                                  //
////////////////////////////////////////////////////////////////////////////////

/// A policy that repairs every failure that it can, in the canonical way, and
/// records the edit of the source that each repair corresponds to.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
#[derive(Debug)]
pub(super) struct Canonical<'src>
{
	/// The source text.
	source: &'src str,

	/// The edits of the source, in the order of the repairs: the byte offset
	/// of each, the number of bytes that it removes, and the text that it
	/// inserts.
	edits: Vec<(usize, usize, String)>,

	/// The failures, in the order of the repairs.
	pub(super) sites: Vec<FailureSite<'src>>
}

impl<'src> Canonical<'src>
{
	/// Create a canonical policy for the specified source.
	///
	/// # Parameters
	/// - `source`: The source text.
	///
	/// # Returns
	/// The policy.
	pub(super) fn new(source: &'src str) -> Self
	{
		Self {
			source,
			edits: Vec::new(),
			sites: Vec::new()
		}
	}

	/// Answer the source with the edits of every repair applied.
	///
	/// # Returns
	/// The edited source.
	pub(super) fn corrected(&self) -> String
	{
		// The repairs happen in source order, except that the closing brace of
		// a variable is recorded with its opening brace, so sort stably.
		let mut edits = self.edits.iter().collect::<Vec<_>>();
		edits.sort_by_key(|(at, _, _)| *at);
		let mut corrected = String::with_capacity(self.source.len());
		let mut copied = 0;
		for (at, removed, inserted) in edits
		{
			corrected.push_str(&self.source[copied..*at]);
			corrected.push_str(inserted);
			copied = at + removed;
		}
		corrected.push_str(&self.source[copied..]);
		corrected
	}

	/// Record an insertion.
	///
	/// # Parameters
	/// - `at`: The byte offset of the insertion.
	/// - `text`: The inserted text.
	fn insert(&mut self, at: usize, text: impl Into<String>)
	{
		self.edits.push((at, 0, text.into()));
	}
}

impl<'src> Recovery<'src> for Canonical<'src>
{
	fn repair(&mut self, site: &FailureSite<'src>) -> Repair
	{
		self.sites.push(site.clone());
		let at = site.position();
		// At a goal or faces, read a bare word as a variable, unless it
		// follows the `-` that begins the faces; after the comma of a formal
		// parameter, read it as the parameter.
		let variable = match site.site
		{
			Site::Goal { .. } | Site::ParameterName => true,
			Site::Faces => !self.source[..at].trim_end().ends_with('-'),
			_ => false
		};
		let start = at + self.source[at..].len()
			- self.source[at..].trim_start_matches(is_token_space).len();
		let input = match site.site
		{
			Site::ParameterName => site.input().take_from(start - at),
			_ => site.input()
		};
		if variable && let Ok((_, name)) = bare_word(input)
		{
			let end = name.location_offset() + name.fragment().len();
			self.insert(name.location_offset(), "{");
			self.insert(end, "}");
			return Repair::Variable { end };
		}
		match site.site
		{
			Site::Goal { .. } =>
			{
				self.insert(at, format!(" {}", PLACEHOLDER_OPERAND))
			},
			Site::ParameterName =>
			{
				self.insert(at, format!("{{{}}}", PLACEHOLDER_NAME))
			},
			Site::VariableName { .. } => self.insert(at, PLACEHOLDER_NAME),
			Site::ParameterColon | Site::Colon { .. } => self.insert(at, ":"),
			Site::LeadingComma => return Repair::Stop,
			// Break the name of a variable early, where it most likely breaks,
			// if it has an operator or a delimiter to break before.
			Site::Closer {
				opener,
				closer: '}'
			} =>
			{
				let name = self.source[opener + 1..at].trim_start();
				if let [Some(length), _] = unclosed_name_ends(name)
				{
					let end = at - name.len() + length;
					self.insert(end, "}");
					return Repair::Break { end };
				}
				self.insert(at, '}')
			},
			Site::Closer { closer, .. } => self.insert(at, closer),
			Site::BindingParen => self.insert(at, "("),
			Site::Faces => self.insert(at, PLACEHOLDER_FACES.to_string()),
			Site::FaceValue { .. } =>
			{
				self.insert(at, PLACEHOLDER_FACE.to_string())
			},
			Site::Direction => self.insert(at, "lowest"),
			Site::TrailingInput =>
			{
				self.edits.push((at, self.source.len() - at, String::new()))
			},
		}
		Repair::Fix
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Examples.                                  //
////////////////////////////////////////////////////////////////////////////////

/// Ensure that the canonical policy repairs representative failures of every
/// [site](Site), producing the expected source, and that the recovered function
/// is the one that the ordinary parser makes of it.
#[test]
fn test_canonical_repairs()
{
	for (source, expected) in [
		("", " 0"),
		("(", "( 0)"),
		("(1", "(1)"),
		("((1", "((1))"),
		("1 +", "1 + 0"),
		("1 + * 2", "1 +  0* 2"),
		("1 + x", "1 + {x}"),
		("1 + xD6", "1 + {xD6}"),
		("- - x", "- - {x}"),
		("{", "{x}"),
		("{x", "{x}"),
		("{x + 2", "{x} + 2"),
		("({x + 2) * 3", "({x} + 2) * 3"),
		("{hit-points + 2", "{hit-points} + 2"),
		("{x+2", "{x}+2"),
		("{x@(3)", "{x}@(3)"),
		("{x: 1 + 2", "{x: 1} + 2"),
		("{x + {y}", "{x} + {y}"),
		("{x {y}", "{x }"),
		("{-3", "{-3}"),
		("x@", "{x}@( 0)"),
		("x@(1", "{x}@(1)"),
		("{x}@", "{x}@( 0)"),
		("{x}@(1", "{x}@(1)"),
		("[", "[ 0: 0]"),
		("[1", "[1: 0]"),
		("[1:", "[1: 0]"),
		("[1:2", "[1:2]"),
		("3d", "3d6"),
		("3d[", "3d[0]"),
		("3d[1,", "3d[1]"),
		("3d6 drop", "3d6 droplowest"),
		("x", "{x}"),
		("x: x", "{x}: {x}"),
		("x, y: x + y", "{x}, {y}: {x} + {y}"),
		("x , y : 1", "{x} , {y} : 1"),
		("{x}, y: {x}", "{x}, {y}: {x}"),
		("x, {y}: {y}", "{x}, {y}: {y}"),
		("hit points: {hit points}", "{hit points}: {hit points}"),
		("x@(1) + {x}", "{x}@(1) + {x}"),
		("x, y", "{x}, {y}: 0"),
		("{x, {y}: {x}", "{x}, {y}: {x}"),
		("{x: {x}", "{x}: {x}"),
		("{x : 1", "{x} : 1"),
		("{x},", "{x},{x}: 0"),
		("{x}, {y}", "{x}, {y}: 0"),
		("{x}, , {y}: 1", "{x},{x} , {y}: 1"),
		("1 2", "1 "),
		("(1 x", "(1 )"),
		("- -", "- - 0"),
		("1 + -", "1 + - 0"),
		("3d6 drop lowest-", "3d6 drop lowest- 0"),
		("3d-", "3d-6"),
		("3d-^2", "3d-6^2"),
		("3dx", "3d{x}"),
		("3d x + 1", "3d {x} + 1"),
		("3d-x", "3d-6")
	]
	{
		let mut policy = Canonical::new(source);
		let recovered = Parser::parse_recovering(source, &mut policy)
			.unwrap_or_else(|e| panic!("{:?} failed: {}", source, e));
		let corrected = policy.corrected();
		assert_eq!(corrected, expected, "source: {:?}", source);
		let parsed = Parser::parse(&corrected).unwrap_or_else(|e| {
			panic!(
				"{:?} corrected to {:?}, which fails: {}",
				source, corrected, e
			)
		});
		assert_eq!(
			recovered.to_string(),
			parsed.to_string(),
			"source: {:?}",
			source
		);
	}
}

/// Ensure that a leading comma, which has no fix, ends recovery with the error
/// of the ordinary parse.
#[test]
fn test_leading_comma_stops()
{
	let mut policy = Canonical::new(", 1");
	assert_eq!(
		Parser::parse_recovering(", 1", &mut policy),
		Parser::parse(", 1")
	);
	assert!(matches!(
		policy.sites.as_slice(),
		[FailureSite {
			site: Site::LeadingComma,
			..
		}]
	));
}

/// A policy that answers each failure by a function, and records the
/// failures.
///
/// # Type parameters
/// - `'src`: The lifetime of the source text.
/// - `F`: The type of the function.
struct Scripted<'src, F>
{
	/// The function that decides each repair.
	decide: F,

	/// The failures, in the order of the repairs.
	sites: Vec<FailureSite<'src>>
}

impl<'src, F: FnMut(&FailureSite<'src>) -> Repair> Recovery<'src>
	for Scripted<'src, F>
{
	fn repair(&mut self, site: &FailureSite<'src>) -> Repair
	{
		self.sites.push(site.clone());
		(self.decide)(site)
	}
}

/// Recover a source with a [scripted](Scripted) policy.
///
/// # Parameters
/// - `source`: The source.
/// - `decide`: The function that decides each repair.
///
/// # Returns
/// The [`Display`](std::fmt::Display) of the recovered function, if recovery
/// succeeded, and the failures.
fn recover<'src>(
	source: &'src str,
	decide: impl FnMut(&FailureSite<'src>) -> Repair
) -> (Option<String>, Vec<FailureSite<'src>>)
{
	let mut policy = Scripted {
		decide,
		sites: Vec::new()
	};
	let recovered = Parser::parse_recovering(source, &mut policy)
		.ok()
		.map(|f| f.to_string());
	(recovered, policy.sites)
}

/// Ensure that a [skip](Repair::Skip) discards closing delimiters at the start
/// of the body, before formal parameters too, and that it stops recovery where
/// the failure does not lie at the start of the goal that the engine would
/// parse again.
#[test]
fn test_skip()
{
	let skip = |site: &FailureSite<'_>| {
		let at = site.position();
		match site.site
		{
			Site::Goal { .. } => Repair::Skip { end: at + 1 },
			_ => Repair::Stop
		}
	};
	let (recovered, sites) = recover(" ) ]}  1 + 2", skip);
	assert_eq!(
		recovered,
		Parser::parse("1 + 2").ok().map(|f| f.to_string())
	);
	assert_eq!(
		sites.iter().map(FailureSite::position).collect::<Vec<_>>(),
		vec![1, 3, 4]
	);
	let (recovered, sites) = recover("{x}: )1", skip);
	assert_eq!(
		recovered,
		Parser::parse("{x}: 1").ok().map(|f| f.to_string())
	);
	assert_eq!(sites.len(), 1);
	// Formal parameters may follow closing delimiters at the start.
	let (recovered, sites) = recover("){x}: {x}", skip);
	assert_eq!(
		recovered,
		Parser::parse("{x}: {x}").ok().map(|f| f.to_string())
	);
	assert_eq!(sites.len(), 1);
	// The operand of `-` fails at `)`, but the engine would parse the body
	// again from `-`.
	let (recovered, sites) = recover("- )1", skip);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
}

/// Ensure that a [retraction](Repair::Retract) continues the formal parameters
/// as though the source lacked a comma that another comma or the `:` follows,
/// and that it stops recovery elsewhere.
#[test]
fn test_retract()
{
	let retract = |_: &FailureSite<'_>| Repair::Retract;
	for (source, expected) in [
		("{x},: 1", "{x}: 1"),
		("{x}, , {y}: 1", "{x}, {y}: 1"),
		("{x},,{y}: {x}", "{x}, {y}: {x}"),
		("{x}, {y},: {y}", "{x}, {y}: {y}"),
		("{x},,: 1", "{x}: 1")
	]
	{
		let (recovered, _) = recover(source, retract);
		assert_eq!(
			recovered,
			Parser::parse(expected).ok().map(|f| f.to_string()),
			"{}",
			source
		);
	}
	// Neither a `,` nor the `:` follows the comma.
	let (recovered, sites) = recover("{x}, 3: 1", retract);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
	// Elsewhere, a retraction stops recovery.
	let (recovered, sites) = recover("1 +", retract);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
}

/// Ensure that a [reread](Repair::Reread) supplies faces before the dice
/// operator that the source supplied, and reads the source again from it, e.g.,
/// as the `d` of a misplaced drop clause, and that it stops recovery elsewhere,
/// or where the faces do not fail at their start.
#[test]
fn test_reread()
{
	let reread = |site: &FailureSite<'_>| match site.site
	{
		Site::Faces => Repair::Reread,
		_ => Repair::Stop
	};
	for (source, expected) in [
		("3 drop lowest", "3D6 drop lowest"),
		("(1)drop highest 2", "(1)D6 drop highest 2"),
		(
			"3 drop lowest + {x} drop highest",
			"3D6 drop lowest + {x}D6 drop highest"
		),
		(
			"{a}@(2 drop lowest drop highest)",
			"{a}@(2D6 drop lowest drop highest)"
		)
	]
	{
		let (recovered, sites) = recover(source, reread);
		assert_eq!(
			recovered,
			Parser::parse(expected).ok().map(|f| f.to_string()),
			"source: {:?}",
			source
		);
		assert!(sites.iter().all(|site| site.site == Site::Faces));
	}
	// A reread supplies faces only where they fail at their start.
	let (recovered, sites) = recover("3d-x", reread);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
	// Elsewhere, a reread stops recovery.
	let (recovered, sites) = recover("1 +", |_| Repair::Reread);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
}

/// Ensure that a failure in the end of a range reports the `[` of the range,
/// and that a failure elsewhere reports none.
#[test]
fn test_goal_range()
{
	for (source, range) in [
		("[1:", Some(0)),
		("1 + [[1:2]:", Some(4)),
		("[1:-", Some(0)),
		("[1:(", None),
		("[", None),
		("1 +", None)
	]
	{
		let (_, sites) = recover(source, |_| Repair::Stop);
		assert!(
			matches!(sites[0].site, Site::Goal { range: r, .. } if r == range),
			"source: {:?}, site: {:?}",
			source,
			sites[0]
		);
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                 Linearity.                                 //
////////////////////////////////////////////////////////////////////////////////

/// The number of repetitions in the smaller inputs of the linearity tests.
const WIDTH: usize = 10_000;

/// The factor by which the larger inputs of the linearity tests repeat more
/// than the smaller ones.
pub(super) const SCALE: usize = 10;

/// A family of inputs for the linearity tests, parameterized by the number of
/// repetitions of the part that needs repair.
pub(super) struct Family
{
	/// The name of the family.
	pub(super) name: &'static str,

	/// Build the member of the family with the given number of repetitions.
	pub(super) source: fn(usize) -> String,

	/// The number of repairs that the member with the given number of
	/// repetitions needs.
	repairs: fn(usize) -> usize
}

/// The families of inputs for the linearity tests, deep and wide.
pub(super) const FAMILIES: &[Family] = &[
	Family {
		name: "unclosed groups",
		source: |n| format!("{}1", "(".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "empty groups",
		source: |n| "(".repeat(n),
		repairs: |n| n + 1
	},
	Family {
		name: "unclosed bindings",
		source: |n| format!("{}1", "{x}@(".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "unclosed ranges",
		source: |n| "[".repeat(n),
		repairs: |n| 3 * n + 1
	},
	Family {
		name: "negations",
		source: |n| format!("{}x", "-".repeat(n)),
		repairs: |_| 1
	},
	Family {
		name: "missing operands",
		source: |n| format!("1{}", " +".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "bare identifiers",
		source: |n| format!("1{}", " + x".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing faces",
		source: |n| format!("1{}", " + 3d".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing directions",
		source: |n| format!("3d6{}", " drop".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "bare faces",
		source: |n| format!("1{}", " + 3dx".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "bare parameters",
		source: |n| format!("{}: 1", vec!["x"; n].join(", ")),
		repairs: |n| n
	},
	// An unclosed brace breaks its name before the operator after it.
	Family {
		name: "unclosed variables",
		source: |n| format!("1{}", " + {x\n".repeat(n)),
		repairs: |n| n
	},
	// An unclosed brace without an operator or delimiter in its name reads
	// everything up to the next brace, or the end of the input, as its name,
	// even across lines, so it needs one repair, however long its name.
	Family {
		name: "unclosed variable",
		source: |n| format!("1 + {{x{}", " y\n".repeat(n)),
		repairs: |_| 1
	}
];

/// Ensure that recovery takes a number of steps linear in the length of the
/// input, for each of the [families](FAMILIES) of inputs that need a repair
/// per repetition: that [`SCALE`] times as many repetitions take at most one
/// more than [`SCALE`] times as many steps, where quadratic time would take
/// [`SCALE`] times more again.
#[test]
fn test_recovery_is_linear()
{
	for family in FAMILIES
	{
		let taken = [WIDTH, SCALE * WIDTH].map(|width| {
			let source = (family.source)(width);
			let mut policy = Canonical::new(&source);
			let before = steps();
			let recovered = Parser::parse_recovering(&source, &mut policy);
			let taken = steps() - before;
			assert!(recovered.is_ok(), "{} failed", family.name);
			assert_eq!(
				policy.sites.len(),
				(family.repairs)(width),
				"{} repairs",
				family.name
			);
			taken
		});
		assert!(
			taken[1] <= (SCALE + 1) * taken[0],
			"{} took {} steps, then {} for {} times the input",
			family.name,
			taken[0],
			taken[1],
			SCALE
		);
	}
}
