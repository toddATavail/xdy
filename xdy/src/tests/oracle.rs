//! # Lean oracle tests
//!
//! Herein are the tests that connect the compiler to the Lean model of the IR
//! in `../../../lean`, whose executable oracle computes the exact distribution
//! of a compiled function by brute force, sharing no algorithm with the
//! [forward pass](crate::DistributionPlan::build) that builds distributions.
//!
//! The Lean tests read real compiler output, the [fixtures](FIXTURES) in
//! `lean/XdyTest/Fixtures`, as the JSON that the `serde` feature writes. An
//! ordinary test holds every fixture to the current compiler, so that a change
//! to the IR or the optimizer cannot leave the Lean tests checking stale
//! programs. Setting [`XDY_BLESS_FIXTURES`](BLESS) rewrites the fixtures
//! instead, after which `just lean` checks the Lean against them.
//!
//! The oracle tests need the Lean toolchain, so they are ignored by default;
//! `just oracle` builds the oracle and runs them, as CI does on every push to
//! `main`, and every pull request against it, that changes the oracle or the
//! code whose distributions it checks. The [corpus
//! test](test_oracle_distribution_corpus) runs every case of the distribution
//! corpus through the oracle. Every case must agree with the oracle exactly, as
//! probabilities, including those whose paths are not equally likely because
//! the number of branches of some roll depends on an earlier roll. The test
//! writes the corpus, with the oracle's distribution in place of every case
//! that disagrees, to `target/oracle/test_distributions.txt`, so that a diff
//! against the corpus shows exactly the cases to correct. The forward pass
//! answers every case of the corpus weight for weight, as its own tests insist,
//! so it agrees with the oracle wherever the corpus does.
//!
//! The [property test](test_oracle_random_programs) runs random [small
//! programs](small_program), which favor dynamic rolls, optimized and not,
//! through the forward pass and the oracle, and holds each to the [properties
//! of exact distributions](check_oracle), above all exact agreement with the
//! oracle. The [resummed
//! test](test_oracle_resummed_programs) does the same after
//! [resumming](resum) each program, so that it reads its rolling records more
//! than once, as the compiler never does. A failure reports the input, shrunk
//! by `proptest`, and records it under `proptest-regressions`, so that later
//! runs try it first.
//!
//! The oracle runs [on a lifeline](RunningOracle), so that it dies with the
//! test process that asked it, however that dies, rather than compute for
//! hours with no one to read its answer. The [orphan
//! test](test_oracle_dies_with_its_parent) kills a test process mid-request,
//! and expects the oracle to exit at once.

#[cfg(unix)]
use std::io::{BufRead as _, BufReader};
use std::{
	env, fs,
	io::{Read, Write as _},
	path::{Path, PathBuf},
	process::{Child, ChildStdin, Command, Stdio},
	thread,
	time::{Duration, Instant}
};

use pretty_assertions::assert_eq;
use proptest::{
	option, prelude::*, sample::select, test_runner::TestCaseError
};
use serde_json::{Map, Value, json};

use crate::{
	Budget, BuildError, Distribution, Evaluator, Function, Passes, Unobserved,
	Weight, compile,
	compiler::compile_unoptimized,
	support::{
		compile_valid, on_small_stack_within, optimize,
		read_distribution_test_cases
	},
	tests::{
		propagation::resum,
		property::{
			NAMES, OPERATORS, agree, binding, check_within, variable,
			with_parameters
		}
	}
};

////////////////////////////////////////////////////////////////////////////////
//                               Lean fixtures.                               //
////////////////////////////////////////////////////////////////////////////////

/// The environment variable that, when set, makes
/// [`test_lean_fixtures_are_current`] rewrite the fixtures rather than check
/// them.
const BLESS: &str = "XDY_BLESS_FIXTURES";

/// The Lean fixtures, as the name of each file, less its `.json` extension,
/// and the source compiled into it. Mirrors the table in
/// `lean/XdyTest/Fixtures/README.md`, which says what each exercises.
const FIXTURES: &[(&str, &str)] = &[
	("arithmetic", "{a}, {b}: -({a} * {b} - {a} / {b} % 3 ^ {b})"),
	("constant", "7"),
	("custom_dice", "2D[-1, 0, 1, 3, 5]"),
	("drop_lowest", "4D6 drop lowest"),
	("dynamic_count", "(1D3)D3"),
	("dynamic_custom_count", "{x}: {x}D[1, 3]"),
	(
		"dynamic_drops",
		"{n}, {m}: 5D6 drop lowest {n} drop highest {m}"
	),
	("dynamic_faces", "3D(2D6)"),
	("dynamic_range", "[1D3:2D6]"),
	("external", "1D6 + {y}"),
	("max", "{x}: {x}D1"),
	("range", "[1:6]"),
	("shared_register", "{x}@(3D6) + {x}"),
	("standard_dice", "3D6")
];

/// Answer the directory of the Lean fixtures.
///
/// # Returns
/// The path of `lean/XdyTest/Fixtures`.
fn fixture_directory() -> PathBuf
{
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.join("..")
		.join("lean")
		.join("XdyTest")
		.join("Fixtures")
}

/// Answer the JSON text of a fixture, as the current compiler writes it: the
/// fully optimized function, as `serde_json` writes it, on one line.
///
/// # Parameters
/// - `source`: The source to compile.
///
/// # Returns
/// The contents of the fixture file.
///
/// # Panics
/// If the source does not compile.
fn fixture_json(source: &str) -> String
{
	let function = compile(source)
		.unwrap_or_else(|e| panic!("{source}: compilation error: {e}"));
	serde_json::to_string(&function).unwrap() + "\n"
}

/// Test that every Lean fixture is exactly what the current compiler writes
/// for its source, and that the fixture directory holds no JSON file that
/// [`FIXTURES`] does not list. With [`XDY_BLESS_FIXTURES`](BLESS) set, rewrite
/// the fixtures instead.
#[test]
fn test_lean_fixtures_are_current()
{
	let directory = fixture_directory();
	if env::var_os(BLESS).is_some()
	{
		for (name, source) in FIXTURES
		{
			fs::write(
				directory.join(format!("{name}.json")),
				fixture_json(source)
			)
			.unwrap();
		}
		return
	}
	let stale = FIXTURES
		.iter()
		.filter(|(name, source)| {
			let path = directory.join(format!("{name}.json"));
			fs::read_to_string(path).ok() != Some(fixture_json(source))
		})
		.map(|(name, _)| *name)
		.collect::<Vec<_>>();
	assert!(
		stale.is_empty(),
		"stale Lean fixtures: {stale:?}; rerun with {BLESS}=1 to rewrite \
		 them, then run `just lean`"
	);
	let unlisted = fs::read_dir(&directory)
		.unwrap()
		.map(|entry| entry.unwrap().path())
		.filter(|path| path.extension().is_some_and(|e| e == "json"))
		.filter(|path| {
			let stem = path.file_stem().unwrap().to_str().unwrap();
			FIXTURES.iter().all(|(name, _)| *name != stem)
		})
		.collect::<Vec<_>>();
	assert!(
		unlisted.is_empty(),
		"Lean fixtures missing from FIXTURES: {unlisted:?}"
	);
}

////////////////////////////////////////////////////////////////////////////////
//                              Oracle support.                               //
////////////////////////////////////////////////////////////////////////////////

/// The environment variable that, when set, names the oracle executable in
/// place of the [default](oracle_path).
const ORACLE: &str = "XDY_ORACLE";

/// The distribution corpus.
const DISTRIBUTION_TEST_SOURCE: &str =
	include_str!("../../tests/test_distributions.txt");

/// Answer the path of the oracle executable: the value of [`XDY_ORACLE`]
/// (ORACLE) if it is set, otherwise where `just lean` builds it.
///
/// # Returns
/// The path of the oracle.
fn oracle_path() -> PathBuf
{
	env::var_os(ORACLE).map(PathBuf::from).unwrap_or_else(|| {
		PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.join("..")
			.join("lean")
			.join(".lake")
			.join("build")
			.join("bin")
			.join(format!("xdy-oracle{}", env::consts::EXE_SUFFIX))
	})
}

/// Answer the directory that receives the oracle's report: `oracle` within
/// the target directory.
///
/// # Returns
/// The path of the directory, which may not exist yet.
fn report_directory() -> PathBuf
{
	env::var_os("CARGO_TARGET_DIR")
		.map(PathBuf::from)
		.unwrap_or_else(|| {
			PathBuf::from(env!("CARGO_MANIFEST_DIR"))
				.join("..")
				.join("target")
		})
		.join("oracle")
}

/// Why the oracle answered no distribution.
#[derive(Debug)]
enum Unanswered
{
	/// The oracle refused the request, with the specified message.
	Refused(String),

	/// The oracle had not answered by the deadline, and was killed.
	Late
}

/// The [oracle](oracle_path), running on a lifeline: the guard holds the
/// oracle's standard input open, and the oracle exits as soon as it closes,
/// whether because the guard drops or because the test process dies, even by
/// `SIGKILL`, whereupon the kernel closes it. So no oracle outlives the test
/// process that asked it. The guard also kills and reaps the oracle on drop,
/// rather than leave it to notice its lifeline closing.
struct RunningOracle
{
	/// The oracle's process.
	child: Child,

	/// The oracle's standard input, which it reads to its end only to learn
	/// when it closes.
	_lifeline: ChildStdin
}

impl RunningOracle
{
	/// Start the oracle on a lifeline, and ask it for the exact distribution of
	/// a function's result.
	///
	/// # Parameters
	/// - `oracle`: The path of the oracle executable.
	/// - `function`: The function.
	/// - `args`: The arguments.
	/// - `externs`: The values of the external variables, by name.
	///
	/// # Returns
	/// The running oracle, whose standard output and standard error are piped.
	///
	/// # Panics
	/// If the oracle cannot be run, or the request cannot be written.
	fn ask(
		oracle: &Path,
		function: &Function,
		args: &[i32],
		externs: &[(&str, i32)]
	) -> Self
	{
		let externs = externs
			.iter()
			.map(|(name, value)| (name.to_string(), json!(value)))
			.collect::<Map<_, _>>();
		let request = json!({
			"function": function,
			"arguments": args,
			"externals": externs
		});
		let mut child = Command::new(oracle)
			.arg("--lifeline")
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.unwrap_or_else(|e| {
				panic!(
					"cannot run the Lean oracle at {}: {e}; build it with \
					 `just lean`, or name it with {ORACLE}",
					oracle.display()
				)
			});
		// Own the lifeline before writing to it, so that the oracle dies even
		// if the write fails.
		let mut oracle = Self {
			_lifeline: child.stdin.take().unwrap(),
			child
		};
		// On a lifeline, the request is the first line of standard input.
		writeln!(oracle._lifeline, "{request}").unwrap();
		oracle
	}
}

impl Drop for RunningOracle
{
	fn drop(&mut self)
	{
		// The oracle may have exited already, so ignore the error; the wait
		// reaps it either way.
		let _ = self.child.kill();
		let _ = self.child.wait();
	}
}

/// Ask the oracle for the exact distribution of a function's result, killing
/// it if it has not answered by the specified deadline. The oracle runs [on a
/// lifeline](RunningOracle), so it dies with the test process, even if that
/// dies before the deadline passes.
///
/// # Parameters
/// - `oracle`: The path of the oracle executable.
/// - `function`: The function.
/// - `args`: The arguments.
/// - `externs`: The values of the external variables, by name.
/// - `deadline`: How long the oracle may take, which may be [`Duration::MAX`].
///
/// # Returns
/// The distribution, in lowest terms.
///
/// # Errors
/// [`Unanswered`] if the oracle refuses the request, or misses the deadline.
///
/// # Panics
/// If the oracle cannot be run, or its response is malformed, has no outcome,
/// or has weights that do not sum to its total.
fn ask_oracle(
	oracle: &Path,
	function: &Function,
	args: &[i32],
	externs: &[(&str, i32)],
	deadline: Duration
) -> Result<Distribution, Unanswered>
{
	let mut running = RunningOracle::ask(oracle, function, args, externs);
	// Drain the pipes while waiting, lest a full pipe stall the oracle.
	let drain = |mut pipe: Box<dyn Read + Send>| {
		thread::spawn(move || {
			let mut bytes = Vec::new();
			pipe.read_to_end(&mut bytes).unwrap();
			bytes
		})
	};
	let stdout = drain(Box::new(running.child.stdout.take().unwrap()));
	let stderr = drain(Box::new(running.child.stderr.take().unwrap()));
	let start = Instant::now();
	let status = loop
	{
		if let Some(status) = running.child.try_wait().unwrap()
		{
			break status
		}
		if start.elapsed() >= deadline
		{
			// Dropping the oracle kills it.
			return Err(Unanswered::Late)
		}
		thread::sleep(Duration::from_millis(1));
	};
	let (stdout, stderr) = (stdout.join().unwrap(), stderr.join().unwrap());
	if !status.success()
	{
		return Err(Unanswered::Refused(
			String::from_utf8_lossy(&stderr).trim().to_owned()
		))
	}
	let response: Value = serde_json::from_slice(&stdout).unwrap();
	let weight = |value: &Value| {
		let text = value.as_str().unwrap();
		text.parse::<Weight>()
			.unwrap_or_else(|e| panic!("malformed weight {text}: {e}"))
	};
	let distribution = Distribution::from_weights(
		response["outcomes"].as_array().unwrap().iter().map(|pair| {
			let outcome = i32::try_from(pair[0].as_i64().unwrap()).unwrap();
			(outcome, weight(&pair[1]))
		})
	)
	.expect("the oracle answered no outcomes");
	assert_eq!(distribution.total(), &weight(&response["total"]));
	Ok(distribution)
}

/// Append a test case to a report in the format of the distribution corpus,
/// opening a new block unless the case has the same source as the one before
/// it.
///
/// # Parameters
/// - `report`: The report.
/// - `previous`: The source of the case before this one, if any.
/// - `source`: The source of the case.
/// - `args`: The arguments.
/// - `externs`: The values of the external variables, by name.
/// - `distribution`: The distribution to record.
fn report_case(
	report: &mut String,
	previous: Option<&str>,
	source: &str,
	args: &[i32],
	externs: &[(&str, i32)],
	distribution: &Distribution
)
{
	if previous != Some(source)
	{
		if previous.is_some()
		{
			report.push('\n');
		}
		report.push_str(source);
		report.push('\n');
	}
	report.push_str("=\n");
	if !args.is_empty()
	{
		let args = args.iter().map(i32::to_string).collect::<Vec<_>>();
		report.push_str(&format!("args: {}\n", args.join(", ")));
	}
	if !externs.is_empty()
	{
		let externs = externs
			.iter()
			.map(|(name, value)| format!("{name}={value}"))
			.collect::<Vec<_>>();
		report.push_str(&format!("externs: {}\n", externs.join(", ")));
	}
	report.push_str(&distribution.to_string());
}

////////////////////////////////////////////////////////////////////////////////
//                               Oracle tests.                                //
////////////////////////////////////////////////////////////////////////////////

/// Test every case of the distribution corpus against the Lean oracle: each
/// must agree with it exactly, as probabilities. Write the corpus, with the
/// oracle's distribution in place of each case that disagrees, to
/// `target/oracle/test_distributions.txt`, and list the cases that disagree.
#[test]
#[ignore = "oracle: run with just oracle"]
fn test_oracle_distribution_corpus()
{
	let oracle = oracle_path();
	let cases = read_distribution_test_cases(DISTRIBUTION_TEST_SOURCE);
	let mut report = String::new();
	let mut previous = None;
	let mut disagreeing = Vec::new();
	for (index, (source, args, externs, expected)) in cases.iter().enumerate()
	{
		let case = format!("case {}: {source} {args:?} {externs:?}", index + 1);
		// Compile exactly as the distribution tests do.
		let function = optimize(compile_valid(source), Passes::all());
		let actual =
			ask_oracle(&oracle, &function, args, externs, Duration::MAX)
				.unwrap_or_else(|e| panic!("{case}: unanswered: {e:?}"));
		let expected = Distribution::from_weights(
			expected
				.iter()
				.map(|&(outcome, count)| (outcome, Weight::from(count)))
		)
		.unwrap();
		let agrees = agree(&expected, &actual);
		if !agrees
		{
			disagreeing.push(case);
		}
		let recorded = if agrees { &expected } else { &actual };
		report_case(&mut report, previous, source, args, externs, recorded);
		previous = Some(source);
	}
	let directory = report_directory();
	fs::create_dir_all(&directory).unwrap();
	let path = directory.join("test_distributions.txt");
	fs::write(&path, &report).unwrap();
	println!(
		"{} cases, of which {} disagree with the oracle:",
		cases.len(),
		disagreeing.len()
	);
	disagreeing.iter().for_each(|case| println!("  {case}"));
	println!("the corpus, corrected by the oracle: {}", path.display());
	assert!(
		disagreeing.is_empty(),
		"{} cases disagree with the oracle; see {}",
		disagreeing.len(),
		path.display()
	);
}

/// The environment variable that marks the process that
/// [`test_oracle_dies_with_its_parent`] kills.
#[cfg(unix)]
const DOOMED: &str = "XDY_ORACLE_DOOMED";

/// The marker that precedes the process ID of the oracle whose parent
/// [`test_oracle_dies_with_its_parent`] kills.
#[cfg(unix)]
const ORPHAN_MARKER: &str = "xdy-oracle-orphan:";

/// How long the oracle may outlive its parent.
#[cfg(unix)]
const ORPHAN_DEADLINE: Duration = Duration::from_secs(10);

/// Ensure that an oracle dies with the test process that asked it, even if that
/// dies by `SIGKILL` while the oracle is still computing. The test runs itself
/// in a child process, marked by [`XDY_ORACLE_DOOMED`](DOOMED), which asks the
/// oracle for `20D20`, which would take it ages, prints the oracle's process
/// ID, and waits. The test kills the child, and expects the oracle to exit
/// within [`ORPHAN_DEADLINE`]; if it does not, the test kills the oracle itself
/// before failing, so that it leaves no orphan behind.
#[test]
#[ignore = "oracle: run with just oracle"]
#[cfg(unix)]
fn test_oracle_dies_with_its_parent()
{
	let oracle = oracle_path();
	if env::var_os(DOOMED).is_some()
	{
		let function = compile("20D20").unwrap();
		let running = RunningOracle::ask(&oracle, &function, &[], &[]);
		println!("{ORPHAN_MARKER}{}", running.child.id());
		loop
		{
			thread::sleep(Duration::from_secs(60));
		}
	}
	assert!(
		oracle.exists(),
		"no Lean oracle at {}; build it with `just lean`, or name it with \
		 {ORACLE}",
		oracle.display()
	);
	let test = thread::current().name().unwrap().to_owned();
	let mut parent = Command::new(env::current_exe().unwrap())
		.args([
			&test,
			"--exact",
			"--nocapture",
			"--include-ignored",
			"--test-threads=1"
		])
		.env(DOOMED, "1")
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.spawn()
		.unwrap();
	// libtest's progress line, e.g., "test name ... ", has no newline yet when
	// the child prints the marker, so search rather than match whole lines.
	let pid = BufReader::new(parent.stdout.take().unwrap())
		.lines()
		.find_map(|line| {
			let line = line.unwrap();
			let (_, pid) = line.split_once(ORPHAN_MARKER)?;
			Some(pid.trim().to_owned())
		});
	parent.kill().unwrap();
	parent.wait().unwrap();
	let pid = pid.expect("the doomed parent never started the oracle");
	// `kill -0` succeeds exactly while the process exists.
	let alive = || {
		Command::new("kill")
			.args(["-0", &pid])
			.stderr(Stdio::null())
			.status()
			.unwrap()
			.success()
	};
	let start = Instant::now();
	while alive()
	{
		if start.elapsed() >= ORPHAN_DEADLINE
		{
			let _ = Command::new("kill").args(["-9", &pid]).status();
			panic!(
				"the oracle (pid {pid}) outlived its parent by {:?}",
				ORPHAN_DEADLINE
			);
		}
		thread::sleep(Duration::from_millis(10));
	}
}

/// The number of cases of [`test_oracle_random_programs`] and
/// [`test_oracle_resummed_programs`]. Each case runs the oracle once for each
/// of its functions, optimized and not, in about 17 ms.
const ORACLE_CASES: u32 = 1_000;

/// The budget of each distribution built by [`check_oracle`], which is a
/// safety net: the oracle answers promptly only far smaller functions.
const ORACLE_BUDGET: Budget = Budget {
	steps: 1_000_000,
	cells: 1_000_000
};

/// How long [`check_oracle`] waits for each answer of the oracle, which
/// computes naively and so may take minutes over a function that the pass
/// answers at once. A function that the oracle has not answered by then
/// abstains. The oracle answers most functions in about 17 ms, and every one
/// that the enumerating builders admitted in under 130 ms.
const ORACLE_DEADLINE: Duration = Duration::from_secs(2);

/// The time budget of [`test_oracle_random_programs`] and
/// [`test_oracle_resummed_programs`], which is a safety net: it turns a hang
/// into a failure.
const ORACLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Ensure that the distribution of every random [small program](small_program)
/// that compiles, optimized and not, over random bindings, satisfies the
/// [properties of exact distributions](check_oracle), and above all agrees
/// with the Lean oracle. Failures are recorded under `proptest-regressions`,
/// in the file that parallels this one.
#[test]
#[ignore = "oracle: run with just oracle"]
fn test_oracle_random_programs()
{
	let oracle = oracle_path();
	assert!(
		oracle.exists(),
		"no Lean oracle at {}; build it with `just lean`, or name it with \
		 {ORACLE}",
		oracle.display()
	);
	on_small_stack_within(ORACLE_TIMEOUT, || {
		check_within(
			ORACLE_CASES,
			file!(),
			|| {
				(
					small_program(),
					[small_binding(), small_binding()],
					[
						small_binding(),
						small_binding(),
						small_binding(),
						small_binding(),
						small_binding()
					]
				)
			},
			|(source, args, externals)| {
				check_oracle(&oracle, &source, &args, &externals, |f| f)
			}
		)
	});
}

/// Ensure that the distribution of every random [small program](small_program)
/// that compiles, optimized and not, [resummed](resum), over random bindings,
/// satisfies the [properties of exact distributions](check_oracle), and above
/// all agrees with the Lean oracle. The compiler never sums a rolling record
/// more than once, so only these programs hold the forward pass to the oracle
/// where it does. Failures are
/// recorded under `proptest-regressions`, in the file that parallels this one.
#[test]
#[ignore = "oracle: run with just oracle"]
fn test_oracle_resummed_programs()
{
	let oracle = oracle_path();
	assert!(
		oracle.exists(),
		"no Lean oracle at {}; build it with `just lean`, or name it with \
		 {ORACLE}",
		oracle.display()
	);
	on_small_stack_within(ORACLE_TIMEOUT, || {
		check_within(
			ORACLE_CASES,
			file!(),
			|| {
				(
					small_program(),
					[small_binding(), small_binding()],
					[
						small_binding(),
						small_binding(),
						small_binding(),
						small_binding(),
						small_binding()
					],
					prop::collection::vec(any::<bool>(), 1..8)
				)
			},
			|(source, args, externals, chosen)| {
				check_oracle(&oracle, &source, &args, &externals, |f| {
					resum(f, &chosen)
				})
			}
		)
	});
}

/// Check the properties of the exact distribution of a source, if it compiles,
/// optimized and not, as the [forward pass](crate::DistributionPlan::build)
/// builds it within [`ORACLE_BUDGET`]: its outcomes lie within the
/// [bounds](Evaluator::bounds_over) of its value; its total is the number of
/// outcomes, whenever the bounds know that number; and it agrees with the Lean
/// oracle, outcome for outcome, as probabilities, and so in its
/// exact [mean](Distribution::mean). The oracle
/// answers in lowest terms, but the pass weighs mixtures and merges over a
/// common denominator, so they agree as probabilities rather than as weights.
///
/// A source that does not compile, or whose distribution does not fit the
/// budget, or that the oracle has not answered by [`ORACLE_DEADLINE`],
/// abstains. The oracle refuses only malformed requests, so its refusal fails.
///
/// # Parameters
/// - `oracle`: The path of the oracle executable.
/// - `source`: The source.
/// - `args`: The arguments, of which the function takes as many as its arity.
/// - `externals`: The values of the externals, by index into [`NAMES`].
/// - `rewrite`: The rewrite of each function, optimized and not, before the
///   pass and the oracle see it.
///
/// # Errors
/// [`TestCaseError`] if a property fails.
fn check_oracle(
	oracle: &Path,
	source: &str,
	args: &[i32],
	externals: &[i32],
	rewrite: impl Fn(Function) -> Function
) -> Result<(), TestCaseError>
{
	let Ok(unoptimized) = compile_unoptimized(source)
	else
	{
		return Ok(())
	};
	let optimized = optimize(unoptimized.clone(), Passes::all());
	let args = &args[..unoptimized.arity()];
	for function in [unoptimized, optimized]
	{
		let function = rewrite(function);
		let externals = NAMES
			.iter()
			.zip(externals)
			.filter(|(name, _)| function.externals.contains(&name.to_string()))
			.map(|(name, value)| (*name, *value))
			.collect::<Vec<_>>();
		let mut evaluator = Evaluator::new(function.clone());
		for (name, value) in &externals
		{
			evaluator.bind(name, *value).unwrap();
		}
		let bounds = evaluator
			.bounds_over(
				args.iter().map(|arg| Some((*arg).into())),
				externals
					.iter()
					.map(|(name, value)| (*name, (*value).into()))
			)
			.unwrap();
		let propagated = match evaluator
			.plan_distribution(args.iter().copied())
			.unwrap()
			.build(ORACLE_BUDGET, &Unobserved)
		{
			Ok(distribution) => distribution,
			Err(BuildError::BudgetExhausted { .. }) => continue,
			Err(e) =>
			{
				return Err(TestCaseError::fail(format!("build failed: {e}")))
			},
		};
		for (outcome, _) in &propagated
		{
			prop_assert!(
				bounds.value.contains(outcome),
				"outcome {} out of bounds {}\nfunction:\n{}",
				outcome,
				bounds.value,
				function
			);
		}
		// The count saturates, and so counts nothing once it reaches its limit.
		if let Some(count) = bounds.count
			&& count < u128::MAX
		{
			prop_assert_eq!(
				propagated.total(),
				&Weight::from(count),
				"total is not the count\nfunction:\n{}",
				function
			);
		}
		let answer = match ask_oracle(
			oracle,
			&function,
			args,
			&externals,
			ORACLE_DEADLINE
		)
		{
			Ok(answer) => answer,
			Err(Unanswered::Late) => continue,
			Err(Unanswered::Refused(e)) =>
			{
				return Err(TestCaseError::fail(format!(
					"the oracle refused: {e}"
				)))
			},
		};
		prop_assert!(
			agree(&propagated, &answer),
			"disagrees with the oracle\nfunction:\n{}\npropagated:\n{}\n\
				oracle:\n{}",
			function,
			propagated,
			answer
		);
		prop_assert_eq!(propagated.mean(), answer.mean(), "means differ");
	}
	Ok(())
}

////////////////////////////////////////////////////////////////////////////////
//                              Small programs.                               //
////////////////////////////////////////////////////////////////////////////////

/// The greatest depth of the expressions in a small program.
const SMALL_MAX_DEPTH: u32 = 4;

/// The number of nodes that the expressions in a small program aim for.
const SMALL_TARGET_SIZE: u32 = 16;

/// Answer a strategy that generates small programs: functions, with or
/// without [parameters](with_parameters), whose bodies are [small
/// expressions](small_expression), so that the oracle often answers them by
/// [`ORACLE_DEADLINE`]. Most bodies are [dice](small_dice) or ranges, whose
/// operands are small expressions, since a small expression is often a lone
/// constant or variable, which rolls nothing.
///
/// # Returns
/// The strategy.
fn small_program() -> impl Strategy<Value = String>
{
	let expression = || small_expression().boxed();
	let body = prop_oneof![
		1 => expression(),
		2 => small_dice(expression()),
		1 => (expression(), expression())
			.prop_map(|(start, end)| format!("[{}:{}]", start, end))
	];
	with_parameters(body)
}

/// Answer a strategy that generates small expressions over small
/// [constants](small_constant), which favor dynamic rolls: dice whose counts,
/// faces, and drops are themselves rolled, ranges whose ends are rolled, and
/// rolls bound to variables that later rolls read. These are the rolls whose
/// paths are not equally likely, because the number of branches of one roll
/// depends on an earlier roll.
///
/// # Returns
/// The strategy.
fn small_expression() -> impl Strategy<Value = String>
{
	let leaf = prop_oneof![small_constant(), variable()];
	leaf.prop_recursive(SMALL_MAX_DEPTH, SMALL_TARGET_SIZE, 3, |inner| {
		prop_oneof![
			2 => (inner.clone(), select(OPERATORS), inner.clone()).prop_map(
				|(left, op, right)| format!("{} {} {}", left, op, right)
			),
			1 => inner.clone().prop_map(|operand| format!("-{}", operand)),
			2 => (variable(), inner.clone())
				.prop_map(|(name, e)| format!("{}@({})", name, e)),
			2 => (inner.clone(), inner.clone())
				.prop_map(|(start, end)| format!("[{}:{}]", start, end)),
			4 => small_dice(inner)
		]
	})
}

/// Answer a strategy that generates small dice expressions, whose operands
/// come from the specified strategy. Any drop clause may lack a drop
/// expression.
///
/// # Parameters
/// - `inner`: The strategy for nested expressions.
///
/// # Returns
/// The strategy.
fn small_dice(inner: BoxedStrategy<String>) -> impl Strategy<Value = String>
{
	let faces = prop_oneof![
		3 => small_atom(inner.clone(), small_size()),
		1 => prop::collection::vec(-2i32..=4, 1..4).prop_map(|faces| {
			let faces = faces.iter().map(i32::to_string).collect::<Vec<_>>();
			format!("[{}]", faces.join(", "))
		})
	];
	// A drop expression never begins with `-`, so a negative constant must be
	// grouped.
	let drop = small_atom(inner.clone(), small_constant()).prop_map(|drop| {
		if drop.starts_with('-')
		{
			format!("({})", drop)
		}
		else
		{
			drop
		}
	});
	let clause = (select(&["lowest", "highest"][..]), option::of(drop))
		.prop_map(|(direction, drop)| match drop
		{
			Some(drop) => format!(" drop {} {}", direction, drop),
			None => format!(" drop {}", direction)
		});
	(
		small_atom(inner, small_size()),
		faces,
		prop::collection::vec(clause, 0..3)
	)
		.prop_map(|(count, faces, clauses)| {
			format!("{}D{}{}", count, faces, clauses.concat())
		})
}

/// Answer a strategy that generates the operands that the grammar admits as
/// dice counts, standard faces, and drop expressions, favoring those that
/// roll: constants, variables, groups, and bindings.
///
/// # Parameters
/// - `inner`: The strategy for nested expressions.
/// - `constant`: The strategy for constants.
///
/// # Returns
/// The strategy.
fn small_atom(
	inner: BoxedStrategy<String>,
	constant: impl Strategy<Value = String> + 'static
) -> impl Strategy<Value = String>
{
	prop_oneof![
		2 => constant,
		1 => variable(),
		2 => inner.clone().prop_map(|e| format!("({})", e)),
		1 => (variable(), inner).prop_map(|(name, e)| format!("{}@({})", name, e))
	]
}

/// Answer a strategy that generates small constants.
///
/// # Returns
/// The strategy.
fn small_constant() -> impl Strategy<Value = String>
{
	(-2i32..=4).prop_map(|n| n.to_string())
}

/// Answer a strategy that generates small constant dice counts and faces,
/// mostly positive ones, since a count or faces of `0` or less rolls nothing,
/// and the optimizer folds such dice away.
///
/// # Returns
/// The strategy.
fn small_size() -> impl Strategy<Value = String>
{
	prop_oneof![4 => 1i32..=3, 1 => -2i32..=0].prop_map(|n| n.to_string())
}

/// Answer a strategy that generates the values of arguments and externals,
/// mostly small ones, but sometimes [any](binding), so that the extremes of
/// [`i32`] reach the oracle whenever the distribution fits the budget.
///
/// # Returns
/// The strategy.
fn small_binding() -> impl Strategy<Value = i32>
{
	prop_oneof![3 => -2i32..=4, 1 => binding()]
}
