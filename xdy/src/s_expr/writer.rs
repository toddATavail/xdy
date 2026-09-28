//! # Iterative S-expression writer and sizer
//!
//! Herein is the engine behind the [`SExpressible`] implementations of the AST.
//! Each AST type describes its S-expression form as a [`Shape`] — a leaf, a
//! keyword form of one or two subexpressions, a binding, or a function —
//! through its [`Shaped`] implementation, which is the only statement of that
//! form. The engine walks the shapes with explicit stacks, so its stack depth
//! is constant however deep the tree.
//!
//! The size of a form is its own contribution — span prefix, keyword,
//! delimiters, spaces, names, and lists — plus the sizes of its subexpressions.
//! Sizing is therefore a single sum over the tree. Writing needs the size of
//! every form that it might wrap, so it first computes all of them in one
//! post-order pass, then writes in pre-order, deciding at each form whether it
//! fits on the current line. Both passes are linear in the size of the tree.
//!
//! A [`Group`] is transparent unless
//! [`with_groups`](SExpressibleOptions::with_groups) is set, in which case it
//! forwards to its subexpression. The engine resolves such forwarding before it
//! counts a form, so both passes agree on which forms exist.

use std::fmt::{self, Write};

use super::{
	Layout, SExpressible, SExpressibleOptions, ident_size, layout,
	span_prefix_size, write_ident, write_newline, write_span_prefix
};
use crate::{
	ast::{
		Add, ArithmeticExpression, Binding, Constant, CustomDice,
		DiceExpression, Div, DropHighest, DropLowest, Exp, Expression,
		Function, Group, Mod, Mul, Neg, Parameter, Range, StandardDice, Sub,
		Variable
	},
	span::SourceSpan
};

////////////////////////////////////////////////////////////////////////////////
//                                  Shapes.                                   //
////////////////////////////////////////////////////////////////////////////////

/// The S-expression form of an AST node. The lifetime `'a` bounds both the
/// borrow of the tree and its source text, which outlives it.
#[derive(Copy, Clone)]
pub(super) enum Shape<'a>
{
	/// A constant, which cannot be split.
	Constant(&'a Constant),

	/// A variable reference, which cannot be split.
	Variable(&'a Variable<'a>),

	/// `(keyword child)`, preceded by the span prefix of `span`.
	Pair
	{
		/// The span of the node.
		span: SourceSpan,

		/// The keyword.
		keyword: &'static str,

		/// The subexpression.
		child: Child<'a>
	},

	/// `(keyword first second)`, preceded by the span prefix of `span`.
	Triple
	{
		/// The span of the node.
		span: SourceSpan,

		/// The keyword.
		keyword: &'static str,

		/// The first subexpression.
		first: Child<'a>,

		/// The second part: a subexpression, or the faces of a custom die.
		second: Part<'a>
	},

	/// `(binding name expression)`, preceded by the span prefix of `span`, with
	/// the name preceded by the span prefix of `name_span`.
	Binding
	{
		/// The span of the binding.
		span: SourceSpan,

		/// The span of the bound name.
		name_span: SourceSpan,

		/// The bound name.
		name: &'a str,

		/// The bound expression.
		expression: &'a Expression<'a>
	},

	/// `(function [parameters] body)`, preceded by the span prefix of `span`.
	Function
	{
		/// The span of the function.
		span: SourceSpan,

		/// The formal parameters.
		parameters: &'a Option<Vec<Parameter<'a>>>,

		/// The body.
		body: &'a Expression<'a>
	},

	/// A transparent [`Group`], which forwards to its subexpression.
	Transparent(&'a Expression<'a>)
}

/// A subexpression of a [`Shape`].
#[derive(Copy, Clone)]
pub(super) enum Child<'a>
{
	/// An arbitrary expression.
	Expression(&'a Expression<'a>),

	/// A dice expression.
	Dice(&'a DiceExpression<'a>)
}

/// The second part of a [triple](Shape::Triple).
#[derive(Copy, Clone)]
pub(super) enum Part<'a>
{
	/// A subexpression.
	Child(Child<'a>),

	/// The faces of a custom die.
	Faces(&'a [i32])
}

/// An AST type that can describe its S-expression form as a [`Shape`].
pub(super) trait Shaped
{
	/// Answer the S-expression form of the receiver.
	///
	/// # Parameters
	/// - `options`: The formatting options, which decide whether a [`Group`] is
	///   transparent.
	///
	/// # Returns
	/// The shape.
	fn shape<'a>(&'a self, options: SExpressibleOptions) -> Shape<'a>;
}

impl<'a> Child<'a>
{
	/// Answer the shape of the subexpression, looking through any transparent
	/// groups.
	///
	/// # Parameters
	/// - `options`: The formatting options.
	///
	/// # Returns
	/// The shape, which is never [`Transparent`](Shape::Transparent).
	fn shape(self, options: SExpressibleOptions) -> Shape<'a>
	{
		let shape = match self
		{
			Child::Expression(expression) => expression.shape(options),
			Child::Dice(dice) => dice.shape(options)
		};
		resolve(shape, options)
	}
}

/// Look through any transparent groups, without recursion.
///
/// # Parameters
/// - `shape`: The shape.
/// - `options`: The formatting options.
///
/// # Returns
/// The first shape that is not [`Transparent`](Shape::Transparent).
fn resolve<'a>(mut shape: Shape<'a>, options: SExpressibleOptions)
-> Shape<'a>
{
	while let Shape::Transparent(expression) = shape
	{
		shape = expression.shape(options);
	}
	shape
}

impl<'a> Shape<'a>
{
	/// Answer the size of the form, excluding the sizes of its subexpressions.
	///
	/// # Parameters
	/// - `options`: The formatting options.
	///
	/// # Returns
	/// The size of the form's own contribution.
	///
	/// # Panics
	/// If the shape is [`Transparent`](Shape::Transparent), which must be
	/// [resolved](resolve) first.
	fn own_size(&self, options: SExpressibleOptions) -> usize
	{
		match *self
		{
			Shape::Constant(constant) => constant.size_s_expr(options),
			Shape::Variable(variable) => variable.size_s_expr(options),
			// Account for two parentheses, the keyword, and one interposing
			// space.
			Shape::Pair { span, keyword, .. } =>
			{
				span_prefix_size(span, options) + 3 + keyword.len()
			},
			// Account for two parentheses, the keyword, two interposing
			// spaces, and the faces of a custom die.
			Shape::Triple {
				span,
				keyword,
				second,
				..
			} =>
			{
				span_prefix_size(span, options)
					+ 4 + keyword.len()
					+ match second
					{
						Part::Child(_) => 0,
						Part::Faces(faces) => faces.size_s_expr(options)
					}
			},
			// Account for two parentheses, the keyword `binding`, two
			// interposing spaces, and the bound name with its span prefix.
			Shape::Binding {
				span,
				name_span,
				name,
				..
			} =>
			{
				span_prefix_size(span, options)
					+ 2 + "binding".len()
					+ 2 + span_prefix_size(name_span, options)
					+ ident_size(name)
			},
			// Account for two parentheses, the keyword, two spaces, and the
			// parameters.
			Shape::Function {
				span, parameters, ..
			} =>
			{
				span_prefix_size(span, options)
					+ 12 + parameters.size_s_expr(options)
			},
			Shape::Transparent(_) =>
			{
				unreachable!("transparent groups are resolved first")
			}
		}
	}

	/// Answer the subexpressions of the form, in order.
	///
	/// # Returns
	/// The subexpressions, of which there are at most two.
	fn children(&self) -> [Option<Child<'a>>; 2]
	{
		match *self
		{
			Shape::Constant(_) | Shape::Variable(_) => [None, None],
			Shape::Pair { child, .. } => [Some(child), None],
			Shape::Triple { first, second, .. } => [
				Some(first),
				match second
				{
					Part::Child(child) => Some(child),
					Part::Faces(_) => None
				}
			],
			Shape::Binding { expression, .. } =>
			{
				[Some(Child::Expression(expression)), None]
			},
			Shape::Function { body, .. } =>
			{
				[Some(Child::Expression(body)), None]
			},
			Shape::Transparent(expression) =>
			{
				[Some(Child::Expression(expression)), None]
			},
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//                                  Sizing.                                   //
////////////////////////////////////////////////////////////////////////////////

/// Size the S-expression representation of an AST node, without recursion.
///
/// # Parameters
/// - `root`: The node.
/// - `options`: The formatting options.
///
/// # Returns
/// The size, as if no indentation were applied.
pub(super) fn size(
	root: &(impl Shaped + ?Sized),
	options: SExpressibleOptions
) -> usize
{
	let mut total = 0;
	let mut pending = vec![resolve(root.shape(options), options)];
	while let Some(shape) = pending.pop()
	{
		total += shape.own_size(options);
		pending.extend(
			shape
				.children()
				.into_iter()
				.flatten()
				.map(|child| child.shape(options))
		);
	}
	total
}

/// A unit of work for [`sizes`].
enum SizeTask<'a>
{
	/// Count a form, and schedule its subexpressions.
	Enter
	{
		/// The form, [resolved](resolve).
		shape: Shape<'a>,

		/// The index of the enclosing form, if any.
		parent: Option<usize>
	},

	/// Add the size of a finished form to that of the enclosing form.
	Exit
	{
		/// The index of the form.
		index: usize,

		/// The index of the enclosing form, if any.
		parent: Option<usize>
	}
}

/// Compute the size of every form in a tree, without recursion.
///
/// # Parameters
/// - `root`: The root form, [resolved](resolve).
/// - `options`: The formatting options.
///
/// # Returns
/// The sizes, indexed by the pre-order position of each form.
fn sizes(root: Shape<'_>, options: SExpressibleOptions) -> Vec<usize>
{
	let mut sizes = Vec::new();
	let mut tasks = vec![SizeTask::Enter {
		shape: root,
		parent: None
	}];
	while let Some(task) = tasks.pop()
	{
		match task
		{
			SizeTask::Enter { shape, parent } =>
			{
				let index = sizes.len();
				sizes.push(shape.own_size(options));
				tasks.push(SizeTask::Exit { index, parent });
				// Schedule the subexpressions in reverse, so that they are
				// entered, and therefore indexed, in order.
				let [first, second] = shape.children();
				for child in [second, first].into_iter().flatten()
				{
					tasks.push(SizeTask::Enter {
						shape: child.shape(options),
						parent: Some(index)
					});
				}
			},
			SizeTask::Exit { index, parent } =>
			{
				if let Some(parent) = parent
				{
					sizes[parent] += sizes[index];
				}
			}
		}
	}
	sizes
}

////////////////////////////////////////////////////////////////////////////////
//                                  Writing.                                  //
////////////////////////////////////////////////////////////////////////////////

/// An entry on the stack of a [`Writer`].
enum Item<'a>
{
	/// Literal text.
	Text(&'static str),

	/// A line break, followed by indentation to the specified level.
	Newline(usize),

	/// A subexpression to write.
	Child
	{
		/// The subexpression.
		child: Child<'a>,

		/// The remaining space on the current line.
		remaining_space: usize,

		/// The indentation level.
		indent: usize
	},

	/// The faces of a custom die to write.
	Faces
	{
		/// The faces.
		faces: &'a [i32],

		/// The remaining space on the current line.
		remaining_space: usize,

		/// The indentation level.
		indent: usize
	}
}

/// A pre-order writer of S-expressions. Every decision to wrap a form is made
/// when the form is expanded; what follows it is scheduled on the stack.
struct Writer<'a, 'f>
{
	/// The destination.
	f: &'f mut dyn Write,

	/// The formatting options. Only the indentation varies during a write,
	/// and each [item](Item) carries its own.
	options: SExpressibleOptions,

	/// The size of every form, indexed by pre-order position.
	sizes: Vec<usize>,

	/// The pre-order position of the next form to expand.
	next: usize,

	/// The pending items, the next of which is on top.
	stack: Vec<Item<'a>>
}

/// Write the S-expression representation of an AST node, without recursion.
///
/// ```mermaid
/// flowchart LR
///     S["shape of root"] --> P1["post-order pass:<br/>size of every form"]
///     P1 --> P2["pre-order pass:<br/>fit or wrap each form,<br/>write"]
/// ```
///
/// # Parameters
/// - `root`: The node.
/// - `f`: The destination.
/// - `remaining_space`: The remaining space on the current line.
/// - `options`: The formatting options.
///
/// # Returns
/// The result of writing to the destination.
///
/// # Errors
/// Propagates any error from the destination.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub(super) fn write(
	root: &(impl Shaped + ?Sized),
	f: &mut dyn Write,
	remaining_space: usize,
	options: SExpressibleOptions
) -> fmt::Result
{
	let root = resolve(root.shape(options), options);
	let mut writer = Writer {
		f,
		options,
		sizes: sizes(root, options),
		next: 0,
		stack: Vec::new()
	};
	writer.expand(root, remaining_space, options.indent)?;
	writer.run()
}

impl<'a> Writer<'a, '_>
{
	/// Answer the formatting options at the specified indentation level.
	///
	/// # Parameters
	/// - `indent`: The indentation level.
	///
	/// # Returns
	/// The options.
	fn at(&self, indent: usize) -> SExpressibleOptions
	{
		SExpressibleOptions {
			indent,
			..self.options
		}
	}

	/// Schedule the specified items, the first of which will be next.
	///
	/// # Parameters
	/// - `items`: The items, in order of writing.
	fn schedule(&mut self, items: impl DoubleEndedIterator<Item = Item<'a>>)
	{
		self.stack.extend(items.rev());
	}

	/// Write every pending item.
	///
	/// # Returns
	/// The result of writing to the destination.
	///
	/// # Errors
	/// Propagates any error from the destination.
	fn run(&mut self) -> fmt::Result
	{
		while let Some(item) = self.stack.pop()
		{
			match item
			{
				Item::Text(text) => self.f.write_str(text)?,
				Item::Newline(indent) => write_newline(self.f, indent)?,
				Item::Child {
					child,
					remaining_space,
					indent
				} =>
				{
					let shape = child.shape(self.options);
					self.expand(shape, remaining_space, indent)?
				},
				Item::Faces {
					faces,
					remaining_space,
					indent
				} =>
				{
					let options = self.at(indent);
					faces.write_s_expr(self.f, remaining_space, options)?
				}
			}
		}
		Ok(())
	}

	/// Write the beginning of a form, deciding whether it fits on the current
	/// line, and schedule the rest.
	///
	/// # Parameters
	/// - `shape`: The form, [resolved](resolve).
	/// - `remaining_space`: The remaining space on the current line.
	/// - `indent`: The indentation level.
	///
	/// # Returns
	/// The result of writing to the destination.
	///
	/// # Errors
	/// Propagates any error from the destination.
	fn expand(
		&mut self,
		shape: Shape<'a>,
		remaining_space: usize,
		indent: usize
	) -> fmt::Result
	{
		let size = self.sizes[self.next];
		self.next += 1;
		let options = self.at(indent);
		match shape
		{
			Shape::Constant(constant) =>
			{
				constant.write_s_expr(self.f, remaining_space, options)
			},
			Shape::Variable(variable) =>
			{
				variable.write_s_expr(self.f, remaining_space, options)
			},
			Shape::Pair {
				span,
				keyword,
				child
			} =>
			{
				let (remaining_space, size) =
					self.begin_form(span, keyword, remaining_space, size)?;
				// We need enough space for ourselves plus the trailing
				// parentheses of enclosing expressions, of which there are as
				// many as the indentation level.
				let (remaining_space, indent) =
					match layout(remaining_space, size, options)
					{
						Layout::Inline =>
						{
							self.f.write_str(" ")?;
							(usize::MAX, indent)
						},
						Layout::Wrapped(options) =>
						{
							write_newline(self.f, options.indent)?;
							(options.available_space(), options.indent)
						}
					};
				self.schedule(
					[
						Item::Child {
							child,
							remaining_space,
							indent
						},
						Item::Text(")")
					]
					.into_iter()
				);
				Ok(())
			},
			Shape::Triple {
				span,
				keyword,
				first,
				second
			} =>
			{
				let (remaining_space, size) =
					self.begin_form(span, keyword, remaining_space, size)?;
				let (remaining_space, indent, separator) =
					match layout(remaining_space, size, options)
					{
						Layout::Inline =>
						{
							// We know that the whole expression fits, so use an
							// effectively infinite value for the remaining
							// space of the subexpressions.
							self.f.write_str(" ")?;
							(usize::MAX, indent, Item::Text(" "))
						},
						Layout::Wrapped(options) =>
						{
							write_newline(self.f, options.indent)?;
							(
								options.available_space(),
								options.indent,
								Item::Newline(options.indent)
							)
						}
					};
				let second = match second
				{
					Part::Child(child) => Item::Child {
						child,
						remaining_space,
						indent
					},
					Part::Faces(faces) => Item::Faces {
						faces,
						remaining_space,
						indent
					}
				};
				self.schedule(
					[
						Item::Child {
							child: first,
							remaining_space,
							indent
						},
						separator,
						second,
						Item::Text(")")
					]
					.into_iter()
				);
				Ok(())
			},
			Shape::Binding {
				span,
				name_span,
				name,
				expression
			} =>
			{
				let body_size = size - shape.own_size(options);
				write_span_prefix(self.f, span, options)?;
				let remaining_space = remaining_space
					.saturating_sub(span_prefix_size(span, options));
				let keyword = "binding";
				write!(self.f, "({}", keyword)?;
				// Account for the open paren and the keyword already emitted.
				let remaining_space =
					remaining_space.saturating_sub(1 + keyword.len());
				let name_size =
					span_prefix_size(name_span, options) + ident_size(name);
				// Two interposing spaces (after keyword, between name and body)
				// and the closing parenthesis, plus the indentation required by
				// enclosing expressions.
				let space_needed =
					1 + name_size + 1 + body_size + 1 + options.indent;
				let (remaining_space, indent) = if remaining_space
					>= space_needed
				{
					self.f.write_str(" ")?;
					write_span_prefix(self.f, name_span, options)?;
					write_ident(self.f, name)?;
					self.f.write_str(" ")?;
					(usize::MAX, indent)
				}
				else
				{
					let options = options.increase_indent();
					write_newline(self.f, options.indent)?;
					write_span_prefix(self.f, name_span, options)?;
					write_ident(self.f, name)?;
					write_newline(self.f, options.indent)?;
					(options.available_space(), options.indent)
				};
				self.schedule(
					[
						Item::Child {
							child: Child::Expression(expression),
							remaining_space,
							indent
						},
						Item::Text(")")
					]
					.into_iter()
				);
				Ok(())
			},
			Shape::Function {
				span,
				parameters,
				body
			} =>
			{
				let body_size = size - shape.own_size(options);
				self.begin_function(
					span,
					parameters,
					body,
					body_size,
					remaining_space,
					options
				)
			},
			Shape::Transparent(_) =>
			{
				unreachable!("transparent groups are resolved first")
			}
		}
	}

	/// Write the span prefix, the open parenthesis, and the keyword of a pair
	/// or triple.
	///
	/// # Parameters
	/// - `span`: The span of the form.
	/// - `keyword`: The keyword.
	/// - `remaining_space`: The remaining space on the current line.
	/// - `size`: The size of the form, including its span prefix.
	///
	/// # Returns
	/// The remaining space before the span prefix, and the size of the form
	/// without its span prefix, i.e., the quantities that decide whether the
	/// form fits.
	///
	/// # Errors
	/// Propagates any error from the destination.
	fn begin_form(
		&mut self,
		span: SourceSpan,
		keyword: &str,
		remaining_space: usize,
		size: usize
	) -> Result<(usize, usize), fmt::Error>
	{
		write_span_prefix(self.f, span, self.options)?;
		let prefix_size = span_prefix_size(span, self.options);
		write!(self.f, "({}", keyword)?;
		Ok((
			remaining_space.saturating_sub(prefix_size),
			size - prefix_size
		))
	}

	/// Write a function up to its body, and schedule the body.
	///
	/// # Parameters
	/// - `span`: The span of the function.
	/// - `parameters`: The formal parameters.
	/// - `body`: The body.
	/// - `body_size`: The size of the body.
	/// - `remaining_space`: The remaining space on the current line.
	/// - `options`: The formatting options.
	///
	/// # Returns
	/// The result of writing to the destination.
	///
	/// # Errors
	/// Propagates any error from the destination.
	fn begin_function(
		&mut self,
		span: SourceSpan,
		parameters: &'a Option<Vec<Parameter<'a>>>,
		body: &'a Expression<'a>,
		body_size: usize,
		remaining_space: usize,
		options: SExpressibleOptions
	) -> fmt::Result
	{
		write_span_prefix(self.f, span, options)?;
		let remaining_space =
			remaining_space.saturating_sub(span_prefix_size(span, options));
		let keyword = "function";
		write!(self.f, "({}", keyword)?;
		// The remaining space discounts the open parenthesis and the keyword.
		// Saturate at zero to tolerate ambitious soft limits without panicking
		// — the remaining space drives a fits-or-wraps heuristic, not a hard
		// budget.
		let remaining_space = remaining_space.saturating_sub(9);
		// We want to write the parameters on the same line as the keyword if
		// at all possible. They need an interposing space, but not the closing
		// parenthesis, which follows the body.
		let params_size = parameters.size_s_expr(options);
		let space_needed = 1 + params_size;
		let remaining_space = if remaining_space >= space_needed
		{
			self.f.write_str(" ")?;
			// We know that the parameters fit, so use an effectively infinite
			// value for their remaining space.
			parameters.write_s_expr(self.f, usize::MAX, options)?;
			remaining_space - space_needed
		}
		else
		{
			let options = options.increase_indent();
			write_newline(self.f, options.indent)?;
			let available_space = options.available_space();
			parameters.write_s_expr(self.f, available_space, options)?;
			// The body may follow the parameters on their last line, so
			// discount whatever the parameters left there: either the whole
			// list, if it fit on one line, or its closing bracket. The test
			// mirrors the one in the parameter-list writer.
			let is_one_line = parameters.as_ref().is_none_or(Vec::is_empty)
				|| available_space >= params_size + options.indent;
			available_space
				.saturating_sub(if is_one_line { params_size } else { 1 })
		};
		// Write out the body. It needs an interposing space and the closing
		// parenthesis, plus the trailing parentheses of enclosing expressions,
		// of which there are as many as the indentation level.
		let space_needed = 1 + body_size + 1 + options.indent;
		let (remaining_space, indent) = if remaining_space >= space_needed
		{
			self.f.write_str(" ")?;
			// We know that the body fits, so use an effectively infinite value
			// for its remaining space.
			(usize::MAX, options.indent)
		}
		else
		{
			let options = options.increase_indent();
			write_newline(self.f, options.indent)?;
			(options.available_space(), options.indent)
		};
		self.schedule(
			[
				Item::Child {
					child: Child::Expression(body),
					remaining_space,
					indent
				},
				Item::Text(")")
			]
			.into_iter()
		);
		Ok(())
	}
}

////////////////////////////////////////////////////////////////////////////////
//                        Abstract syntax tree (AST).                         //
////////////////////////////////////////////////////////////////////////////////

impl Shaped for Function<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		Shape::Function {
			span: self.span,
			parameters: &self.parameters,
			body: &self.body
		}
	}
}

impl Shaped for Group<'_>
{
	fn shape<'a>(&'a self, options: SExpressibleOptions) -> Shape<'a>
	{
		if options.with_groups
		{
			// Opaque rendering: `(group <expr>)` so the group's span and
			// structural identity survive the round-trip.
			pair(self.span, "group", Child::Expression(&self.expression))
		}
		else
		{
			// Transparent rendering: forward directly to the subexpression.
			Shape::Transparent(&self.expression)
		}
	}
}

impl Shaped for Constant
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		Shape::Constant(self)
	}
}

impl Shaped for Variable<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		Shape::Variable(self)
	}
}

impl Shaped for Range<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "range", &self.start, &self.end)
	}
}

impl Shaped for Binding<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		Shape::Binding {
			span: self.span,
			name_span: self.name_span,
			name: &self.name,
			expression: &self.expression
		}
	}
}

impl Shaped for Expression<'_>
{
	fn shape<'a>(&'a self, options: SExpressibleOptions) -> Shape<'a>
	{
		// Expressions are transparent in the S-expression representation, so
		// we simply answer the shape of the appropriate variant.
		match self
		{
			Expression::Group(group) => group.shape(options),
			Expression::Constant(constant) => constant.shape(options),
			Expression::Variable(variable) => variable.shape(options),
			Expression::Binding(binding) => binding.shape(options),
			Expression::Range(range) => range.shape(options),
			Expression::Dice(dice) => dice.shape(options),
			Expression::Arithmetic(arithmetic) => arithmetic.shape(options)
		}
	}
}

impl Shaped for StandardDice<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "standard-dice", &self.count, &self.faces)
	}
}

impl Shaped for CustomDice<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		Shape::Triple {
			span: self.span,
			keyword: "custom-dice",
			first: Child::Expression(&self.count),
			second: Part::Faces(&self.faces)
		}
	}
}

impl Shaped for DropLowest<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		drop_clause(self.span, "drop-lowest", &self.dice, self.drop.as_deref())
	}
}

impl Shaped for DropHighest<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		drop_clause(self.span, "drop-highest", &self.dice, self.drop.as_deref())
	}
}

impl Shaped for DiceExpression<'_>
{
	fn shape<'a>(&'a self, options: SExpressibleOptions) -> Shape<'a>
	{
		// Dice expressions are transparent in the S-expression representation,
		// so we simply answer the shape of the underlying variant.
		match self
		{
			DiceExpression::Standard(dice) => dice.shape(options),
			DiceExpression::Custom(dice) => dice.shape(options),
			DiceExpression::DropLowest(clause) => clause.shape(options),
			DiceExpression::DropHighest(clause) => clause.shape(options)
		}
	}
}

impl Shaped for Add<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "add", &self.left, &self.right)
	}
}

impl Shaped for Sub<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "sub", &self.left, &self.right)
	}
}

impl Shaped for Mul<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "mul", &self.left, &self.right)
	}
}

impl Shaped for Div<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "div", &self.left, &self.right)
	}
}

impl Shaped for Mod<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "mod", &self.left, &self.right)
	}
}

impl Shaped for Exp<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		triple(self.span, "exp", &self.left, &self.right)
	}
}

impl Shaped for Neg<'_>
{
	fn shape<'a>(&'a self, _options: SExpressibleOptions) -> Shape<'a>
	{
		pair(self.span, "neg", Child::Expression(&self.operand))
	}
}

impl Shaped for ArithmeticExpression<'_>
{
	fn shape<'a>(&'a self, options: SExpressibleOptions) -> Shape<'a>
	{
		// Arithmetic expressions are transparent in the S-expression
		// representation, so we simply answer the shape of the underlying
		// variant.
		match self
		{
			ArithmeticExpression::Add(add) => add.shape(options),
			ArithmeticExpression::Sub(sub) => sub.shape(options),
			ArithmeticExpression::Mul(mul) => mul.shape(options),
			ArithmeticExpression::Div(div) => div.shape(options),
			ArithmeticExpression::Mod(r#mod) => r#mod.shape(options),
			ArithmeticExpression::Exp(exp) => exp.shape(options),
			ArithmeticExpression::Neg(neg) => neg.shape(options)
		}
	}
}

/// Answer the shape of a keyword form of one subexpression.
///
/// # Parameters
/// - `span`: The span of the node.
/// - `keyword`: The keyword.
/// - `child`: The subexpression.
///
/// # Returns
/// The shape.
fn pair<'a>(
	span: SourceSpan,
	keyword: &'static str,
	child: Child<'a>
) -> Shape<'a>
{
	Shape::Pair {
		span,
		keyword,
		child
	}
}

/// Answer the shape of a keyword form of two expressions.
///
/// # Parameters
/// - `span`: The span of the node.
/// - `keyword`: The keyword.
/// - `first`: The first expression.
/// - `second`: The second expression.
///
/// # Returns
/// The shape.
fn triple<'a>(
	span: SourceSpan,
	keyword: &'static str,
	first: &'a Expression<'a>,
	second: &'a Expression<'a>
) -> Shape<'a>
{
	Shape::Triple {
		span,
		keyword,
		first: Child::Expression(first),
		second: Part::Child(Child::Expression(second))
	}
}

/// Answer the shape of a drop clause: a triple if it has a drop count, or a
/// pair if not.
///
/// # Parameters
/// - `span`: The span of the clause.
/// - `keyword`: The keyword.
/// - `dice`: The dice expression from which to drop.
/// - `drop`: The number of dice to drop, if given.
///
/// # Returns
/// The shape.
fn drop_clause<'a>(
	span: SourceSpan,
	keyword: &'static str,
	dice: &'a DiceExpression<'a>,
	drop: Option<&'a Expression<'a>>
) -> Shape<'a>
{
	match drop
	{
		Some(drop) => Shape::Triple {
			span,
			keyword,
			first: Child::Dice(dice),
			second: Part::Child(Child::Expression(drop))
		},
		None => pair(span, keyword, Child::Dice(dice))
	}
}
