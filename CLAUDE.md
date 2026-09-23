# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md. -->

## Commands

- Python: `.venv/Scripts/python.exe` (Python 3.13). Install dependencies with `.venv/Scripts/python.exe -m pip install -r requirements.txt`.
- Tests: `.venv/Scripts/python.exe -m pytest`. For a single test, add `path/to/test_file.py::test_name`.

## Repository state

There is no code, dependency file, or test suite yet. The repo has only the problem statement and its data:

- `task.md`: the Zalando "Gift Problem". It is a job-application exercise for an Applied Scientist role, submitted in place of a cover letter, so the reasoning and write-up count as much as the final answer.
- `items.json`: 60 items (`A1`–`A60`), each with an integer `price`. No volumes.
- `packages.json`: 1000 past deliveries, each with a measured `total_volume` (litres) and the list of item names it contained.

`task.md` links the datasets on Google Drive. The local JSON files are those datasets, so don't fetch them.

When code is added, record its run and test commands here.

## The problem, as a model

1. **Estimate item volumes (regression).** Each package's `total_volume` is the sum of its items' true volumes plus measurement error ~ N(0, σ²) with **variance 2** (σ = √2 ≈ 1.41, not 2). An item appears at most once per package, so this is a linear model `A v + ε = b`: `A` is a 1000×60 binary incidence matrix (package × item), `v` holds the unknown volumes, and `b` holds the measured totals. Ordinary least squares is the MLE under this i.i.d. Gaussian noise. Volumes are physically non-negative (NNLS is an option), and `σ²(AᵀA)⁻¹` gives the uncertainty of the estimates.
2. **Choose the gift (knapsack).** Maximise total price subject to total volume ≤ 40 L. The natural reading is a 0/1 knapsack (each item at most once), but `task.md` does not say this outright, so state the assumption. Volumes are real-valued and there are 60 items, which rules out brute force. Use an exact ILP/MILP, or a DP over discretised volume.
3. **Robustness.** The volumes are estimates. If the optimal selection lands close to 40 L, check how likely it is to actually fit, given the estimation uncertainty.

## Working mode (set by the user)

This is the user's job-interview submission, so the user supervises every step:

- Work one step at a time. Explain the idea and the options, give a recommendation, and wait for the user's decision. Build only that step, then stop for review. Never start a later step early.
- Log every decision and every unreviewed AI finding in `DECISIONS.md`, labelled with who decided. An AI proposal that the user approved is logged as "Author, on AI proposal", never as the user's own idea.
- The user rejected the AI's proposed solution outline. The structure of the solution and of the write-up is the user's call.
- The user's write-up structure: Problem statement, Methodology, Results and discussion. Each section has two parts: the AI-assisted work and the author's decisions.
- The AI writes the code and the user reviews it. Every file the AI drafts starts with the label `AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.`, written as a comment. Update the step's row in the "Step review" table of `DECISIONS.md` when the user signs it off.
- Commit each step only after the user's review, one commit per step. The message says what the AI drafted and what the user decided, citing the log entry numbers. Never commit unreviewed work.

## Decided so far

See `DECISIONS.md`. In short:

- Python.
- Two solutions and a comparison: a direct reference solver (regression, then knapsack), and a script where Claude solves the same problem through the API. Before writing any Claude API code, load the `claude-api` skill so model IDs and SDK usage are current.
- Python environment: a venv on Python 3.13 (`py -3.13`). The Claude SDK 1.x needs Python 3.10 or later, and the default `python` in Git Bash is Anaconda 3.8.
