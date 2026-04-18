//! Build script for the xdy crate.
//!
//! Generates SVG railroad diagrams from EBNF grammar files in the `grammar/`
//! directory. The generated SVGs are written to the `doc/` directory and
//! inlined into Rustdoc via `include_str!`.
//!
//! The SVGs are post-processed to be Rustdoc-compatible:
//!
//! * The `<style>` block is removed and its rules are applied as inline `style`
//!   attributes. Rustdoc's markdown parser mangles `<style>` blocks inside
//!   `include_str!` doc attributes because empty lines and `{}` braces look
//!   like markdown constructs.
//!
//! * Square brackets are escaped as XML character references (`&#x5B;`,
//!   `&#x5D;`) so that TOML headers like `[metadata]` in railroad diagram text
//!   nodes are not resolved as intra-doc links.

use std::fs;
use std::path::Path;

fn main()
{
	let grammar_dir = Path::new("grammar");
	let doc_dir = Path::new("doc");
	fs::create_dir_all(doc_dir).expect("failed to create doc/ directory");

	for entry in
		fs::read_dir(grammar_dir).expect("failed to read grammar/ directory")
	{
		let entry = entry.expect("failed to read directory entry");
		let path = entry.path();
		if path.extension().is_some_and(|ext| ext == "ebnf")
		{
			let stem = path
				.file_stem()
				.expect("no file stem")
				.to_str()
				.expect("non-UTF-8 filename");
			let source =
				fs::read_to_string(&path).expect("failed to read EBNF file");
			let diagram = ebnsf::parse_ebnf(&source)
				.unwrap_or_else(|e| panic!("failed to parse {}: {e}", stem));
			let svg = diagram.to_string();
			let svg = inline_styles(&svg);
			let svg = escape_brackets(&svg);
			let output_path = doc_dir.join(format!("{stem}.svg"));
			fs::write(&output_path, &svg).unwrap_or_else(|e| {
				panic!("failed to write {}: {e}", output_path.display())
			});
			println!("cargo::rerun-if-changed={}", path.display());
		}
	}
}

/// Remove the `<style>` block and apply its rules as inline `style` attributes
/// on the corresponding SVG elements.
///
/// The railroad crate's CSS is simple and predictable — a small fixed set of
/// class-based rules. We map each rule to inline styles on the elements that
/// match, then strip the `<style>` block entirely.
///
/// # Parameters
///
/// * `svg` — the SVG's content.
///
/// # Returns
///
/// The same SVG, with inlined styles.
fn inline_styles(svg: &str) -> String
{
	// Step 1: strip the <style>...</style> block.
	let svg = if let Some(start) = svg.find("<style")
	{
		if let Some(end) = svg.find("</style>")
		{
			let end = end + "</style>".len();
			// Also consume the trailing newline if present.
			let end = if svg.as_bytes().get(end) == Some(&b'\n')
			{
				end + 1
			}
			else
			{
				end
			};
			format!("{}{}", &svg[..start], &svg[end..])
		}
		else
		{
			svg.to_string()
		}
	}
	else
	{
		svg.to_string()
	};

	// Step 2: apply inline styles based on the railroad crate's CSS rules.
	//
	// The mapping below matches the default light stylesheet from the
	// `railroad` crate (v0.3.x). Each replacement targets a specific
	// element+class combination.

	let svg = svg
		// svg.railroad — background styles on the root <svg> element.
		.replace(
			"class=\"railroad\"",
			"class=\"railroad\" style=\"\
			 background-color:hsl(30,20%,95%);\
			 background-size:15px 15px;\
			 background-image:\
			 linear-gradient(to right,rgba(30,30,30,.05) 1px,transparent 1px),\
			 linear-gradient(to bottom,rgba(30,30,30,.05) 1px,transparent 1px)\"",
		)
		// rect.railroad_canvas — invisible canvas rect.
		.replace(
			"class=\"railroad_canvas\"",
			"class=\"railroad_canvas\" style=\"stroke-width:0;fill:none\"",
		);

	// Step 3: apply styles that require context-aware replacement.
	//
	// We process line by line to handle rules that depend on parent context
	// (e.g., text inside .nonterminal groups gets bold).
	let mut result = String::with_capacity(svg.len());
	let mut in_nonterminal = false;
	let mut nonterminal_depth: usize = 0;

	for line in svg.lines()
	{
		let trimmed = line.trim();
		let mut styled_line = line.to_string();

		// Track nonterminal group nesting.
		if trimmed.contains("class=\"nonterminal\"")
		{
			in_nonterminal = true;
			nonterminal_depth = 1;
		}
		else if in_nonterminal
		{
			// Count nested <g> opens and </g> closes.
			if trimmed.starts_with("<g ")
			{
				nonterminal_depth += 1;
			}
			if trimmed == "</g>"
			{
				nonterminal_depth = nonterminal_depth.saturating_sub(1);
				if nonterminal_depth == 0
				{
					in_nonterminal = false;
				}
			}
		}

		// Apply element-level styles.
		if trimmed.starts_with("<path ")
		{
			styled_line = styled_line.replacen(
				"<path ",
				"<path style=\"stroke-width:3px;stroke:black;fill:none\" ",
				1
			);
		}
		else if trimmed.starts_with("<rect ")
			&& !trimmed.contains("class=\"railroad_canvas\"")
		{
			styled_line = styled_line.replacen(
				"<rect ",
				"<rect style=\"stroke-width:3px;stroke:black;\
				 fill:hsl(-290,70%,90%)\" ",
				1
			);
		}
		else if trimmed.starts_with("<text ")
		{
			if trimmed.contains("class=\"comment\"")
			{
				// text.comment — italic.
				styled_line = styled_line.replacen(
					"<text ",
					"<text style=\"font:italic 12px monospace;\
					 text-anchor:middle\" ",
					1
				);
			}
			else if in_nonterminal
			{
				// .nonterminal text — bold.
				styled_line = styled_line.replacen(
					"<text ",
					"<text style=\"font:bold 14px monospace;\
					 text-anchor:middle\" ",
					1
				);
			}
			else
			{
				// Regular text.
				styled_line = styled_line.replacen(
					"<text ",
					"<text style=\"font:14px monospace;\
					 text-anchor:middle\" ",
					1
				);
			}
		}

		result.push_str(&styled_line);
		result.push('\n');
	}

	result
}

/// Escape square brackets as XML character references so Rustdoc does not try
/// to resolve TOML headers like `[metadata]` as intra-doc links.
///
/// # Parameters
///
/// * `svg` — the SVG's content.
///
/// # Returns
///
/// The same content, but with square brackets escaped.
fn escape_brackets(svg: &str) -> String
{
	svg.replace('[', "&#x5B;").replace(']', "&#x5D;")
}
