<!-- AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md. -->

# Decision log and AI use

This project was done with an AI assistant. This file records how the AI was used and who made each decision. The AI keeps this log, and the author confirms its entries at each step.

## How AI is used

- **Assistant:** Claude Code (model Claude Opus 5.5, by Anthropic). It runs exploratory analysis, explains methods and options, drafts code and text, and runs checks.
- **Supervision:** the author supervises every step. For each step the AI explains what it proposes and why; the author makes the call; only then is the step built, and the author reviews the result before the next step starts.
- **Claude as a test subject:** separately, part of the project tests Claude itself (`claude-opus-5` through the Claude API) on the same problem, as a blind comparison against the reference solution. That part is an experiment, not assistance.

## Who decided

- **Author:** the author's own call, including a choice among options the AI laid out.
- **Author, on AI proposal:** the AI recommended it; the author reviewed and approved it.
- **AI:** done or found by the AI and not yet reviewed by the author. These are checked again in a later step.

## Labels in files

- Every file the AI drafts starts with the label `AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.`, written as a comment in that file's syntax.
- In `README.md`, every section is split into "AI-assisted work" and "Author's decisions".
- The original task files (`task.md`, `items.json`, `packages.json`) are unchanged and carry no label.
- Each step is committed only after the author's review. The commit message says what the AI drafted and what the author decided.

## Step review

| Step | Content | Drafted by | Author review |
|---|---|---|---|
| 0 | Setup: git, Python 3.13 venv, `requirements.txt`, `.gitignore`, `README.md` skeleton, this log, `CLAUDE.md` | AI | Approved 2026-09-23 |

## Log

| # | Date | Step | Decision or finding | Who decided | Notes |
|---|---|---|---|---|---|
| 1 | 2026-09-23 | Setup | Use Python. | Author | Chosen from languages the AI listed. |
| 2 | 2026-09-23 | Setup | Build two solutions and compare them: a direct reference solver, and a script where Claude solves the problem through the API. | Author | Chosen from three options the AI offered (Claude only / direct only / both). |
| 3 | 2026-09-23 | Planning | Exploratory analysis (not yet reviewed) found that the data fit a noise variance of about 3.86 rather than the stated 2 (likely meaning a standard deviation of 2), and that the best-priced set (757, 39.97 L) has only about a 50% chance of truly fitting in 40 L. | AI | Finding only. The analysis ran outside the repo; its results will be rebuilt with reviewed code in the estimation and uncertainty steps. |
| 4 | 2026-09-23 | Planning | Headline answer: recommend the 735 set (the most expensive set with at least a 95% chance of fitting), and show the 757 set as the exact optimum with its risk. | Author, on AI proposal | |
| 5 | 2026-09-23 | Planning | Use the estimated noise variance (about 3.86) as the main one for the uncertainty maths, with the stated 2 alongside. | Author, on AI proposal | |
| 6 | 2026-09-23 | Planning | Run the Claude solver 8 times, blind, at an estimated $3–8 in total. | Author, on AI proposal | Paid runs start only after a further go-ahead. |
| 7 | 2026-09-23 | Planning | Submit a repository containing the write-up, the saved results and the Claude run logs. | Author, on AI proposal | |
| 8 | 2026-09-23 | Planning | Rejected the AI's proposed outline for the solution. | Author | This is a job-interview test, not just a problem to be solved, so the AI does not set the structure. |
| 9 | 2026-09-23 | Planning | Work one step at a time: the author supervises and decides each step, and AI use and decision ownership are recorded here. | Author | |
| 10 | 2026-09-23 | Planning | Write-up structure: Problem statement, Methodology, Results and discussion. Each section has two parts: the AI-assisted work and the author's decisions. | Author | |
| 11 | 2026-09-23 | 0 Setup | Create the `README.md` skeleton now and fill it in as the steps are built. | Author | The AI asked "now or at the end" with no recommendation. |
| 12 | 2026-09-23 | 0 Setup | The AI writes the code; the author reviews every step. | Author, on AI proposal | |
| 13 | 2026-09-23 | 0 Setup | Use git, with one commit per step. Each commit message says what the AI drafted and what the author decided. | Author, on AI proposal | |
| 14 | 2026-09-23 | 0 Setup | Put an AI label in every file the AI drafts. | Author | The AI had recommended relying on this log and the README only; the author chose file labels instead. |
| 15 | 2026-09-23 | 0 Setup | Use a Python 3.13 virtual environment (`.venv`). | Author, on AI proposal | Needed for the Claude SDK 1.x (Python 3.10 or later). Versions are pinned in `requirements.txt`. |
