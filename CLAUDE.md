# CLAUDE.md

## Project
The full spec is in SPEC.md. Read it before planning anything.
Work through its milestones in order, one at a time. Do not start the next
milestone until I say the current one is done.

## Commands
(Claude: fill these in as they come to exist, and keep them current.)
- Install: `make install`
- Run: `make data` (download bars; needs `.env`), `make earnings` (needs `SEC_USER_AGENT`), `make train`, `make compare`, `make analysis`
- Pipeline: `make up` (dashboard at http://localhost:8501), `make replay`, `make down`
- Test: `make test` (unit), `make integration` (Docker Compose, about 30s once built)
- Lint: `make lint`

## 1. Think before coding
- Before each milestone, give me a short plan: the files you will create or
  change, and how each step will be verified. Wait for my OK.
- State your assumptions. If there are several reasonable approaches, show me
  the options and tradeoffs instead of picking silently.
- Ask only when the answer changes the design. For small choices, pick the
  simplest option and tell me what you picked.
- If a simpler approach exists than the one I asked for, say so.

## 2. Simplicity first
- Write the minimum code that solves the problem. No extra features, no
  abstractions for single-use code, no configurability I didn't ask for.
- If 200 lines could be 50, rewrite it.
- Do not add a dependency without telling me what it is for and what the
  alternative was.

## 3. Surgical changes
- Change only what the task needs. Match the existing style.
- Remove only the unused code your own change created. Mention other dead
  code, don't delete it.

## 4. Verify
- Turn each task into a check. For a bug, write a test that reproduces it,
  then fix it.
- Run the tests after every change and show me the result.
- Never edit or delete a test just to make it pass. If a test is wrong, tell
  me why first.

## 5. Teach me
- I'm a student. I have to be able to explain every line of this project in
  an interview.
- When you use something I may not know (a library, a pattern, a language
  feature), explain it in two or three plain sentences.
- After each milestone, explain what you built and why. Then ask me three
  questions an interviewer could ask about it, and correct my answers.
- Record every major decision in DECISIONS.md: what we chose, what the
  alternatives were, and why.
- When I say "my turn", give me the function signature and the tests, and let
  me write the body myself. Review it after.

## 6. Honesty
- Never invent metrics, results, or sources. Every number in the README must
  come from a script in this repo that I can rerun.
- Report sample sizes with every percentage.
- If something doesn't work or a result is bad, say so plainly. Don't hide it
  with a workaround.

## 7. Safety and cost
- No secrets or API keys in the repo. Use a .env file, and keep a
  .env.example with the variable names only.
- Files in data/raw/ are never modified. Write cleaned data to data/processed/.
- Before any command that calls a paid API or runs longer than a few minutes,
  tell me the estimated cost or time and wait for my OK.

## Git
- Commit after each verified step, with a message that says what changed and
  why. Never force-push or rewrite history.

## Done means
- Tests pass locally and in CI.
- README has setup steps, an architecture overview, real metrics with sample
  sizes, and a Limitations section.
- DECISIONS.md is up to date.
