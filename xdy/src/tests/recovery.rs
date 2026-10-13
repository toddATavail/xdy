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
	ast::Direction,
	parser::{
		FailureSite, LoneArgument, Operand, PLACEHOLDER_FACE,
		PLACEHOLDER_FACES, PLACEHOLDER_NAME, PLACEHOLDER_OPERAND,
		PredicatedKind, Recovery, Repair, Site, bare_word, is_token_space,
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
		let signed = self.source[..at].trim_end().ends_with('-');
		let variable = match site.site
		{
			// The faces of dice, or an operand of a reroll or explode, may
			// begin with a `-`, as its sign.
			Site::Faces
			| Site::Goal {
				operand: Some(_), ..
			} => !signed,
			Site::Goal { .. } | Site::ParameterName => true,
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
			// The operand of a reroll is the constant that begins at its sign.
			Site::Goal {
				operand: Some(_), ..
			} if signed => self.insert(at, PLACEHOLDER_OPERAND.to_string()),
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
			Site::Direction { keep: false } => self.insert(at, "lowest"),
			Site::Direction { keep: true } => self.insert(at, "highest"),
			// Supply the dice operator and faces just after the dice count,
			// which an earlier repair may have supplied at the `keep`, and
			// separate them from the `keep`, unless whitespace already does.
			Site::DiceOperator =>
			{
				let count_end = match self.edits.last()
				{
					Some(&(edit, 0, _)) if edit == at => at,
					_ =>
					{
						self.source[..at].trim_end_matches(is_token_space).len()
					},
				};
				let faces = format!("D{}", PLACEHOLDER_FACES);
				match count_end == at
				{
					true => self.insert(count_end, format!("{faces} ")),
					false => self.insert(count_end, faces)
				}
			},
			// Separate the supplied tokens from those after them, lest they run
			// together, as `= 12` would for `= 1` before `2 times`.
			Site::Predicate { threshold, .. } =>
			{
				self.insert(at, format!("= {} ", threshold))
			},
			Site::RangeOperator { .. } => self.insert(at, "..="),
			// Supply the `,`, before another argument; a second after a lone
			// argument, before its `)`.
			Site::Comma { .. } => self.insert(at, ","),
			// Retract the comma.
			Site::TrailingComma => self.edits.push((at, 1, String::new())),
			// Supply the `(` before the call, whose `)` a closer supplies.
			Site::UngroupedCall { .. } => self.insert(at, "("),
			Site::Times { singular, .. } =>
			{
				self.insert(at, if singular { " time " } else { " times " })
			},
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
		("3d6 keep", "3d6 keephighest"),
		("3d6 keep lowest-", "3d6 keep lowest- 0"),
		("3 keep lowest", "3D6 keep lowest"),
		("-3 keep lowest 2", "-3D6 keep lowest 2"),
		("(1)keep highest 2", "(1)D6 keep highest 2"),
		("{x} keep", "{x}D6 keephighest"),
		("[3 keep lowest:4]", "[3D6 keep lowest:4]"),
		("(2 keep highest", "(2D6 keep highest)"),
		(
			"1 + 2 keep lowest drop highest",
			"1 + 2D6 keep lowest drop highest"
		),
		("3d6 reroll", "3d6 reroll= 1 "),
		("3d6 reroll 2 times", "3d6 reroll = 1 2 times"),
		("3d6 reroll + 1", "3d6 reroll = 1 + 1"),
		("3d6 reroll =", "3d6 reroll = 0"),
		("3d6 reroll ≤ + 1", "3d6 reroll ≤  0+ 1"),
		("3d6 reroll < -", "3d6 reroll < -0"),
		("3d6 reroll < x", "3d6 reroll < {x}"),
		("3d6 reroll in", "3d6 reroll in 0..= 0"),
		("3d6 reroll in ..3", "3d6 reroll in  0..3"),
		("3d6 reroll in 1 3", "3d6 reroll in 1 ..=3"),
		("3d6 reroll in 1..", "3d6 reroll in 1.. 0"),
		("3d6 reroll in -..2", "3d6 reroll in -0..2"),
		("3d6 reroll in [1, 2", "3d6 reroll in [1, 2]"),
		("3d6 reroll = 1 2", "3d6 reroll = 1 2 times "),
		("3d6 reroll = 1 1", "3d6 reroll = 1 1 time "),
		("3d6 reroll = 1 ∞", "3d6 reroll = 1 ∞ times "),
		(
			"3d6 reroll = 1 unlimited + 1",
			"3d6 reroll = 1 unlimited  times + 1"
		),
		("3 reroll = 1", "3D6 reroll = 1"),
		("3 reroll!=1", "3D6 reroll!=1"),
		("1 + rerollhighest", "1 + {rerollhighest}"),
		("highest(1, rerollhighest)", "highest(1, {rerollhighest})"),
		("-3 reroll in [1]", "-3D6 reroll in [1]"),
		("(2 reroll ≥ 2", "(2D6 reroll ≥ 2)"),
		("3d6 explode", "3d6 explode= 6 "),
		("3d6 explode 2 times", "3d6 explode = 6 2 times"),
		("3d6 explode + 1", "3d6 explode = 6 + 1"),
		(
			"4d10 keep highest 3 explode",
			"4d10 keep highest 3 explode= 10 "
		),
		(
			"3d[1, 4] drop lowest explode",
			"3d[1, 4] drop lowest explode= 4 "
		),
		("3d{x} explode", "3d{x} explode= 6 "),
		("3d6 explode =", "3d6 explode = 0"),
		("3d6 explode ≤ + 1", "3d6 explode ≤  0+ 1"),
		("3d6 explode < -", "3d6 explode < -0"),
		("3d6 explode < x", "3d6 explode < {x}"),
		("3d6 explode in", "3d6 explode in 0..= 0"),
		("3d6 explode in ..3", "3d6 explode in  0..3"),
		("3d6 explode in 1 3", "3d6 explode in 1 ..=3"),
		("3d6 explode in 1..", "3d6 explode in 1.. 0"),
		("3d6 explode in -..2", "3d6 explode in -0..2"),
		("3d6 explode in [1, 2", "3d6 explode in [1, 2]"),
		("3d6 explode = 1 2", "3d6 explode = 1 2 times "),
		("3d6 explode = 1 1", "3d6 explode = 1 1 time "),
		("3d6 explode = 1 ∞", "3d6 explode = 1 ∞ times "),
		(
			"3d6 explode = 1 unlimited + 1",
			"3d6 explode = 1 unlimited  times + 1"
		),
		("3 explode = 1", "3D6 explode = 1"),
		("3 explode!=1", "3D6 explode!=1"),
		("1 + explodehighest", "1 + {explodehighest}"),
		("highest(1, explodehighest)", "highest(1, {explodehighest})"),
		("-3 explode in [1]", "-3D6 explode in [1]"),
		("(2 explode ≥ 2", "(2D6 explode ≥ 2)"),
		("3d-", "3d-6"),
		("3d-^2", "3d-6^2"),
		("3dx", "3d{x}"),
		("3d x + 1", "3d {x} + 1"),
		("3d-x", "3d-6"),
		("highest(", "highest( 0, 0)"),
		("highest()", "highest( 0, 0)"),
		("lowest(1", "lowest(1, 0)"),
		("lowest(1 2)", "lowest(1 ,2)"),
		("highest(1, 2", "highest(1, 2)"),
		("highest(1, 2 3)", "highest(1, 2 ,3)"),
		("highest(1, x)", "highest(1, {x})"),
		("highest(1, 2,)", "highest(1, 2)"),
		("highest(1, 2 , )", "highest(1, 2  )"),
		("lowest(1,)", "lowest(1, 0)"),
		("highest(4D6)", "highest(4D6, 0)"),
		("3dhighest(1, 2)", "3d(highest(1, 2))"),
		("3d highest(1 2", "3d (highest(1 ,2))"),
		("highest(1, 2)d6", "(highest(1, 2))d6"),
		("lowest (1, 2) d6 + 1", "(lowest (1, 2)) d6 + 1"),
		(
			"3d6 drop lowest lowest(1, 2)",
			"3d6 drop lowest (lowest(1, 2))"
		),
		(
			"3d6 keep highest highest(1, 2) + 1",
			"3d6 keep highest (highest(1, 2)) + 1"
		),
		("3d6 reroll ≤ highest(1, 2)", "3d6 reroll ≤ (highest(1, 2))"),
		(
			"3d6 reroll in lowest(1, 2)..3",
			"3d6 reroll in (lowest(1, 2))..3"
		),
		(
			"3d6 reroll in 1..lowest(2, 3)",
			"3d6 reroll in 1..(lowest(2, 3))"
		),
		(
			"3d6 reroll = 1 highest(1, 2) times",
			"3d6 reroll = 1 (highest(1, 2)) times"
		),
		(
			"3d6 explode ≤ highest(1, 2)",
			"3d6 explode ≤ (highest(1, 2))"
		),
		(
			"3d6 explode in lowest(1, 2)..3",
			"3d6 explode in (lowest(1, 2))..3"
		),
		(
			"3d6 explode in 1..lowest(2, 3)",
			"3d6 explode in 1..(lowest(2, 3))"
		),
		(
			"3d6 explode = 1 highest(1, 2) times",
			"3d6 explode = 1 (highest(1, 2)) times"
		),
		(
			"3dhighest(1, 2dlowest(3, 4))",
			"3d(highest(1, 2d(lowest(3, 4))))"
		)
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

/// Ensure that declining a [`Site::DiceOperator`], which is not a failure of
/// the ordinary parse, leaves the parse as the ordinary parse is, and does not
/// end recovery: the trailing keep clause after it is still presented, and
/// fixed, as trailing input.
#[test]
fn test_declining_dice_operator_continues()
{
	let (recovered, sites) = recover("3 keep lowest", |site| match site.site
	{
		Site::DiceOperator => Repair::Stop,
		_ => Repair::Fix
	});
	assert_eq!(recovered.as_deref(), Some("3"));
	assert_eq!(
		sites.iter().map(|site| site.site).collect::<Vec<_>>(),
		[Site::DiceOperator, Site::TrailingInput]
	);
	assert_eq!(sites[0].position(), 2);
	let (recovered, sites) = recover("(3 keep lowest) + 1", |site| match site
		.site
	{
		Site::DiceOperator => Repair::Fix,
		_ => Repair::Stop
	});
	assert_eq!(recovered.as_deref(), Some("(3D6 keep lowest) + 1"));
	assert_eq!(sites.len(), 1);
}

/// Ensure that a reroll or explode clause after a dice count offers a repair
/// at [`Site::DiceOperator`] as a keep clause does, and that declining it does
/// not end recovery.
#[test]
fn test_declining_dice_operator_continues_predicated()
{
	for (source, grouped, expected) in [
		("3 reroll = 1", "(3 reroll = 1) + 1", "(3D6 reroll = 1) + 1"),
		(
			"3 explode = 6",
			"(3 explode = 6) + 1",
			"(3D6 explode = 6) + 1"
		)
	]
	{
		let (recovered, sites) = recover(source, |site| match site.site
		{
			Site::DiceOperator => Repair::Stop,
			_ => Repair::Fix
		});
		assert_eq!(recovered.as_deref(), Some("3"), "{source:?}");
		assert_eq!(
			sites.iter().map(|site| site.site).collect::<Vec<_>>(),
			[Site::DiceOperator, Site::TrailingInput],
			"{source:?}"
		);
		assert_eq!(sites[0].position(), 2, "{source:?}");
		let (recovered, sites) = recover(grouped, |site| match site.site
		{
			Site::DiceOperator => Repair::Fix,
			_ => Repair::Stop
		});
		assert_eq!(recovered.as_deref(), Some(expected), "{grouped:?}");
		assert_eq!(sites.len(), 1, "{grouped:?}");
	}
}

/// Ensure that declining a [`Site::UngroupedCall`] or a
/// [`Site::TrailingComma`], which are not failures of the ordinary parse,
/// leaves the parse as the ordinary parse is, and does not end recovery, and
/// that accepting one alone recovers.
#[test]
fn test_declining_call_offers_continues()
{
	let decline = |site: &FailureSite<'_>| match site.site
	{
		Site::UngroupedCall { .. } | Site::TrailingComma => Repair::Stop,
		_ => Repair::Fix
	};
	let (recovered, sites) = recover("3dhighest(1, 2)", decline);
	assert_eq!(recovered.as_deref(), Some("3D6"));
	assert_eq!(
		sites.iter().map(|site| site.site).collect::<Vec<_>>(),
		[
			Site::UngroupedCall {
				operand: Operand::Faces
			},
			Site::Faces,
			Site::TrailingInput
		]
	);
	assert_eq!(sites[0].position(), 2);
	let (recovered, sites) = recover("highest(1, 2,)", decline);
	assert_eq!(recovered.as_deref(), Some("highest(1, 2, 0)"));
	assert_eq!(sites[0].site, Site::TrailingComma);
	assert_eq!(sites[0].position(), 12);
	let accept = |site: &FailureSite<'_>| match site.site
	{
		Site::UngroupedCall { .. }
		| Site::TrailingComma
		| Site::Closer { .. } => Repair::Fix,
		_ => Repair::Stop
	};
	for (source, expected) in [
		(
			"3d6 drop lowest lowest(1, 2,)",
			"3D6 drop lowest (lowest(1, 2))"
		),
		("highest(1, 2)d6 + 1", "(highest(1, 2))D6 + 1")
	]
	{
		let (recovered, _) = recover(source, accept);
		assert_eq!(recovered.as_deref(), Some(expected), "{source:?}");
	}
}

/// Ensure that each ungrouped call reports the operand that it stands as.
#[test]
fn test_ungrouped_call_operands()
{
	for (source, operand) in [
		("highest(1, 2)d6", Operand::DiceCount),
		("3dhighest(1, 2)", Operand::Faces),
		("3d6 drop lowest lowest(1, 2)", Operand::DropCount),
		("3d6 keep lowest lowest(1, 2)", Operand::KeepCount),
		(
			"3d6 reroll < lowest(1, 2)",
			Operand::Threshold(PredicatedKind::Reroll)
		),
		(
			"3d6 reroll in lowest(1, 2)..3",
			Operand::RangeStart(PredicatedKind::Reroll)
		),
		(
			"3d6 reroll in 1..=lowest(1, 2)",
			Operand::RangeEnd(PredicatedKind::Reroll)
		),
		(
			"3d6 reroll = 1 lowest(1, 2) times",
			Operand::Limit(PredicatedKind::Reroll)
		),
		(
			"3d6 explode > lowest(1, 2)",
			Operand::Threshold(PredicatedKind::Explode)
		),
		(
			"3d6 explode in lowest(1, 2)..7",
			Operand::RangeStart(PredicatedKind::Explode)
		),
		(
			"3d6 explode in 5..=lowest(1, 2)",
			Operand::RangeEnd(PredicatedKind::Explode)
		),
		(
			"3d6 explode = 6 lowest(1, 2) times",
			Operand::Limit(PredicatedKind::Explode)
		)
	]
	{
		let (_, sites) = recover(source, |_| Repair::Fix);
		assert_eq!(
			sites[0].site,
			Site::UngroupedCall { operand },
			"{source:?}"
		);
	}
}

/// Ensure that a [call](Repair::Call) reads a word as the keyword of a call,
/// where an expression may begin and a `(` follows the word, and that it stops
/// recovery otherwise.
#[test]
fn test_call()
{
	let call = |site: &FailureSite<'_>| {
		let at = site.position();
		match site.site
		{
			Site::Goal { .. } => Repair::Call {
				end: at + 3,
				direction: Direction::Lowest
			},
			_ => Repair::Stop
		}
	};
	for (source, expected) in [
		("min(1, 2)", "lowest(1, 2)"),
		("1 + min (2, 3)", "1 + lowest(2, 3)"),
		("-min(1, 2)", "-lowest(1, 2)"),
		("lowest(min(1, 2), 3)", "lowest(lowest(1, 2), 3)"),
		("(min(1, 2))d6", "(lowest(1, 2))D6")
	]
	{
		let (recovered, sites) = recover(source, call);
		assert_eq!(recovered.as_deref(), Some(expected), "{source:?}");
		assert_eq!(sites.len(), 1, "{source:?}");
	}
	// No `(` follows the word.
	let (recovered, sites) = recover("min + 1", call);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
	// Only an expression may begin a call.
	let (recovered, sites) = recover("3dmin(1, 2)", call);
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
}

/// Ensure that an [unwrap](Repair::Unwrap) replaces a call of one argument
/// with that argument, grouped unless it may stand bare, and that a
/// [keep](Repair::Keep) replaces one of a dice argument with the argument and
/// a keep clause, and that each stops recovery elsewhere.
#[test]
fn test_unwrap_and_keep()
{
	let lone = |repair: Repair| {
		move |site: &FailureSite<'_>| match site.site
		{
			Site::Comma { lone: Some(_), .. } => repair,
			_ => Repair::Stop
		}
	};
	for (source, repair, expected) in [
		("highest(1 + 2) * 3", Repair::Unwrap, "(1 + 2) * 3"),
		("lowest(-1) ^ 2", Repair::Unwrap, "(-1) ^ 2"),
		("highest( {x} ) + 1", Repair::Unwrap, "{x} + 1"),
		("highest(4D6)", Repair::Unwrap, "4D6"),
		("highest(4D6)", Repair::Keep, "4D6 keep highest"),
		(
			"lowest(4D6 drop lowest) - 1",
			Repair::Keep,
			"4D6 drop lowest keep lowest - 1"
		),
		(
			"2 ^ highest(highest(3d6))",
			Repair::Keep,
			"2 ^ 3D6 keep highest keep highest"
		)
	]
	{
		let (recovered, _) = recover(source, lone(repair));
		assert_eq!(recovered.as_deref(), Some(expected), "{source:?}");
		let parsed = Parser::parse(expected).map(|f| f.to_string());
		assert_eq!(parsed.as_deref(), Ok(expected), "{source:?}");
	}
	let LoneArgument { argument, .. } =
		match recover("lowest( 1 + 2 )", lone(Repair::Stop)).1[0].site
		{
			Site::Comma {
				opener: 6,
				lone: Some(lone)
			} => lone,
			site => panic!("{site:?}")
		};
	assert_eq!((argument.start, argument.end), (8, 13));
	// A keep needs a dice argument.
	let (recovered, sites) = recover("highest({x})", lone(Repair::Keep));
	assert_eq!(recovered, None);
	assert_eq!(sites.len(), 1);
	// Neither applies where the call goes on.
	for repair in [Repair::Unwrap, Repair::Keep]
	{
		let (recovered, sites) = recover("highest(4D6 2)", |_| repair);
		assert_eq!(recovered, None);
		assert_eq!(sites.len(), 1);
	}
}

/// Ensure that recovery stops, rather than supply `,` after `,` forever,
/// wherever the argument after a supplied `,` fails without reading anything,
/// as where a policy supplies an operand before a bare word rather than read
/// the word as a variable, so that the word is never consumed; and that it
/// does not stop where a call within a call lacks its `,` at the same place.
#[test]
fn test_repeated_comma_stops()
{
	for source in ["highest(1, rerollhighest", "lowest(1 rerolllowest 2)"]
	{
		let (recovered, sites) = recover(source, |_| Repair::Fix);
		assert_eq!(recovered, None, "{source:?}");
		let commas = sites
			.iter()
			.filter(|site| matches!(site.site, Site::Comma { .. }))
			.count();
		assert_eq!(commas, 1, "{source:?}");
	}
	let (recovered, _) = recover("lowest(lowest(1", |_| Repair::Fix);
	assert_eq!(recovered.as_deref(), Some("lowest(lowest(1, 0), 0)"));
}

/// Ensure that a failure at the start of an argument of a call reports the
/// `(` of the call, and that a failure elsewhere reports none.
#[test]
fn test_goal_call()
{
	for (source, call) in [
		("highest(", Some(7)),
		("1 + lowest (1, ", Some(11)),
		("highest(1, lowest(", Some(17)),
		("highest(1 +", None),
		("(", None)
	]
	{
		let (_, sites) = recover(source, |_| Repair::Stop);
		assert!(
			matches!(sites[0].site, Site::Goal { call: c, .. } if c == call),
			"source: {:?}, site: {:?}",
			source,
			sites[0]
		);
	}
}

/// Ensure that a goal that fails at an operand of the predicate of a reroll
/// or explode clause says so, and which kind of clause it is, and that every
/// other goal does not.
#[test]
fn test_goal_of_predicated_operand()
{
	let reroll = Some(PredicatedKind::Reroll);
	let explode = Some(PredicatedKind::Explode);
	for (source, operand) in [
		("3d6 reroll =", reroll),
		("3d6 reroll ≤ -", reroll),
		("3d6 reroll in", reroll),
		("3d6 reroll in 1..", reroll),
		("3d6 explode =", explode),
		("3d6 explode ≥ -", explode),
		("3d6 explode in", explode),
		("3d6 explode in 5..", explode),
		("3d6 explode = 6 reroll in", reroll),
		("3d6 reroll = 1 explode in", explode),
		("3d6 reroll = (", None),
		("3d6 reroll = 1 (", None),
		("3d6 explode = 6 (", None),
		("3d6 drop lowest (", None),
		("1 +", None)
	]
	{
		let (_, sites) = recover(source, |_| Repair::Fix);
		let goals = sites
			.iter()
			.filter_map(|site| match site.site
			{
				Site::Goal { operand, .. } => Some(operand),
				_ => None
			})
			.collect::<Vec<_>>();
		assert_eq!(goals.first(), Some(&operand), "{source:?}");
	}
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
		name: "missing keep directions",
		source: |n| format!("3d6{}", " keep".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "misplaced keeps",
		source: |n| format!("1{}", " + 3 keep lowest".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing reroll predicates",
		source: |n| format!("3d6{}", " reroll".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing reroll thresholds",
		source: |n| format!("1{}", " + 3d6 reroll ≤".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing range operators",
		source: |n| format!("3d6{}", " reroll in 1 2".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing times",
		source: |n| format!("3d6{}", " reroll = 1 2".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "misplaced rerolls",
		source: |n| format!("1{}", " + 3 reroll = 1".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing explode predicates",
		source: |n| format!("3d6{}", " explode".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing explode thresholds",
		source: |n| format!("1{}", " + 3d6 explode ≥".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing explode range operators",
		source: |n| format!("3d6{}", " explode in 5 6".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing explode times",
		source: |n| format!("3d6{}", " explode = 6 2".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "misplaced explodes",
		source: |n| format!("1{}", " + 3 explode = 6".repeat(n)),
		repairs: |n| n
	},
	Family {
		name: "missing commas",
		source: |n| format!("highest(1{})", " 2".repeat(n)),
		repairs: |n| n
	},
	// A lone argument gains a `,` and then a second argument.
	Family {
		name: "lone arguments",
		source: |n| format!("1{}", " + highest(1)".repeat(n)),
		repairs: |n| 2 * n
	},
	Family {
		name: "nested lone arguments",
		source: |n| format!("{}1{}", "lowest(".repeat(n), ")".repeat(n)),
		repairs: |n| 2 * n
	},
	// An empty call gains an argument, a `,`, and another argument.
	Family {
		name: "empty calls",
		source: |n| format!("1{}", " + highest()".repeat(n)),
		repairs: |n| 3 * n
	},
	Family {
		name: "trailing commas",
		source: |n| format!("1{}", " + lowest(1, 2,)".repeat(n)),
		repairs: |n| n
	},
	// An ungrouped call gains its `(` and its `)`.
	Family {
		name: "ungrouped faces",
		source: |n| format!("1{}", " + 3dhighest(1, 2)".repeat(n)),
		repairs: |n| 2 * n
	},
	Family {
		name: "ungrouped dice counts",
		source: |n| format!("1{}", " + lowest(1, 2)d6".repeat(n)),
		repairs: |n| 2 * n
	},
	Family {
		name: "nested ungrouped faces",
		source: |n| format!("{}2{}", "3dhighest(1, ".repeat(n), ")".repeat(n)),
		repairs: |n| 2 * n
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
