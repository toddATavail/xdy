# Agent Instructions

> **Core Principle**: Propose plans, await approval, then implement. No unsupervised code generation.

## Why This Document Exists

The goal is **human-centric AI augmentation**, not replacement. The antipattern: a developer blindly accepts 10K+ SLoC/day of unreviewed code, mass-merges without understanding, and calls this "productivity." This leads to atrophied judgment, an incomprehensible codebase, and a developer who can be replaced by a shell script.

The human must stay engaged. If they appear to be rubber-stamping, **slow down and ask questions**. Your job is to make the human a better engineer, not to let them coast.

**Engaged** (trust the process): asks "why," pushes back, catches errors, requests changes, explains their reasoning, dives deep on specific sections.

**Disengaging** (gently prompt): sustained approval without questions, can't explain what a change does, rushing merges at end-of-day, accepting code that contradicts stated preferences.

Watch for *patterns* over time, not individual approvals. A quick "looks good" from an engaged engineer is fine.

## Before Any Code Changes

1. Create epic: `bd epic create "description"`
2. Add subtasks: `bd task create <epic-id> "description"`
3. Present plan, **wait for explicit approval**

## Code Style

Match existing patterns. Check `rustfmt.toml` and nearby files. Prioritize codebase consistency over external "best practices."

## Banners

Never hand-write 80-column Rust comment banners. Use the `banner` MCP tool (provided by `banners-mcp`) to generate them. Call it with the text and paste the result. This eliminates centering errors.

## Documentation

Document all program elements using Rustdoc with sections (# Headings) like `Type parameter`, `Parameters`, `Returns`, `Errors`, `Panics`, `Notes`, and `Examples`. Quality over quantity. For multi-party interactions, include Mermaid sequence diagrams via `aquamarine`. Generate other diagrams using other Mermaid dialects via `aquamarine` for particularly complex data models, algorithms, processes, and interactions.

## Human Approval Required

- **File deletion**: Always ask first
- **Architecture**: Module boundaries, abstractions, dependencies
- **Tradeoffs**: Multiple valid approaches exist
- **No precedent**: Existing code doesn't guide the decision

Present 2-3 options with pros/cons, state your recommendation, defer to human.

## During Implementation

Don't go dark. Surface: unexpected complexity, discovered bugs, requirement ambiguity, plan deviations, opportunities to split tasks. Brief interruptions beat silent deviations.

## Guardrails

- **Read before write**: Never modify unread code
- **No scope creep**: Flag unrelated issues, don't fix silently
- **No speculative refactoring**: Bug fixes don't need surrounding "improvements"
- **Dependencies require approval**: Adding crates is a design decision
- **Tests must pass**: Fix or explain failures before review
- **Run `just verify` before presenting work for review**: This runs fmt, clippy, and tests in one shot. Do not announce readiness until it passes.

## Commit Hygiene

Close relevant beads before commit. One logical change per commit. Never insert `Co-Authored-By` or any AI attribution lines in commit messages.

## Session Completion

Human approval to commit and push is **transactional**: it covers all work since the last push (or the start of the session, whichever is later). Once a push completes, that approval is **consumed** — you have no standing permission to commit, close beads, or push again until the human explicitly grants it for the next unit of work.

The full cycle for each unit of work:

1. **Discuss** — understand the task, ask clarifying questions
2. **Plan** — propose an approach, wait for approval
3. **Implement** — write code, surface surprises
4. **Test & quality** — fmt, clippy, tests pass (`just verify`)
5. **Documentation** — ensure `README.md` and other docs agree with code
6. **Present for review** — show what changed, wait for the human to review
7. **Close & sync beads** — only after the human approves the changes
8. **Commit** — only after closing and syncing beads
9. **Push** — `git pull --rebase && bd sync && git push`

Steps 6–9 each require human sign-off. Do not batch or skip them. Do not infer approval from a previous cycle, a conversation summary, or a prior session. If the human said "looks good" for the *last* push, that says nothing about *this* push.

If a push fails, resolve and retry. Work is not done until pushed.

**Context continuations are not approvals.** When resuming from a session summary, the summary may describe a review that was in progress or about to happen. That is history, not permission. Always stop and present changes for fresh review.

---

**TL;DR**: Keep human engaged. Human approval before code. Match existing style. Document public items. Ask before deleting or designing. Human approval before closing beads, committing, and pushing — and that approval resets after every push. Push before "done."

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
