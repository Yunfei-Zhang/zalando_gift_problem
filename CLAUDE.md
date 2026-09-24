# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md. -->

## Commands

- Python: `.venv/Scripts/python.exe` (Python 3.13). Install dependencies with `.venv/Scripts/python.exe -m pip install -r requirements.txt`.
- Tests: `.venv/Scripts/python.exe -m pytest`. For a single test, add `path/to/test_file.py::test_name`.
- Data summary and checks: `.venv/Scripts/python.exe -m gift.data`.
- Volume estimation report, which rewrites `results/volumes.csv`, `results/estimation.json` and `results/figures/*.png`: `.venv/Scripts/python.exe -m gift.estimate`.
- Gift selection report, which rewrites `results/knapsack.json`: `.venv/Scripts/python.exe -m gift.knapsack`. The timings in it change from run to run.
- Uncertainty report, which rewrites `results/uncertainty.json`, `results/figures/price_vs_chance.png` and `results/figures/item_stability.png` and takes about a minute: `.venv/Scripts/python.exe -m gift.robust`. The full test suite takes about 1.5 minutes, mostly for the winner's-curse check. It is seeded, so the numbers repeat.
- Git remote: `origin` is `https://github.com/Yunfei-Zhang/zalando_gift_problem` (branch `main`).

When code is added, record its run and test commands here.

## Repository state

The code lives in the `gift/` package, with tests in `tests/`. `gift/data.py` loads and checks the JSON and returns `GiftData` (`names`, `prices`, `A`, `b`), which every later step uses. `gift/estimate.py` fits the volumes by least squares and returns a `Fit`, whose `covariance(variance)` and `standard_errors(variance)` default to the estimated noise variance; pass `STATED_VARIANCE` only for comparison. `gift/figures.py` holds the plotting helpers. `gift/knapsack.py` solves the 0/1 knapsack on given volumes with three methods: `exact_table` (the proposed method), `integer_program` and `greedy`, each returning `(Selection, work)`. It also provides `fractional_bound` and `near_optimal`, which step 4 builds on. `gift/robust.py` computes each near-optimal set's chance of fitting (normal and positive-only, under both variances), applies the 95% rule, and runs the winner's-curse check and the item-stability diagnostic. Generated outputs go in `results/` and are committed. `DECISIONS.md` has the "Step review" table showing which steps are done. The original problem statement and data:

- `task.md`: the Zalando "Gift Problem". It is a job-application exercise for an Applied Scientist role, submitted in place of a cover letter, so the reasoning and write-up count as much as the final answer.
- `items.json`: 60 items (`A1`–`A60`), each with an integer `price`. No volumes.
- `packages.json`: 1000 past deliveries, each with a measured `total_volume` (litres) and the list of item names it contained.

`task.md` links the datasets on Google Drive. The local JSON files are those datasets, so don't fetch them.

## The problem, as a model

1. **Estimate item volumes (regression).** Each package's `total_volume` is the sum of its items' true volumes plus Gaussian measurement error. An item appears at most once per package, so this is a linear model `A v + ε = b`: `A` is the 1000×60 0/1 table (package × item), `v` holds the unknown volumes, and `b` holds the measured totals. Ordinary least squares is the maximum-likelihood estimate, and `σ²(AᵀA)⁻¹` gives the uncertainty of the estimates.
   - **Noise variance:** `task.md` states variance 2, but the residual variance of the fit is about 3.86 (95% CI 3.53–4.23). The author rejects the stated value (DECISIONS #19), so the estimated variance is the main one and the stated 2 is shown only for comparison (#5). The reading "2 was meant as the standard deviation" goes in the write-up labelled as the AI's interpretation, not the author's (#23).
2. **Choose the gift (knapsack).** Maximise total price subject to total volume ≤ 40 L, with each item at most once (#31) and exactly 40 L allowed (#35). The exact table over total price is the proposed method, integer programming is the second exact method and cross-check, and greedy by unit price is only a comparison (#32, #37). On the estimated volumes, the best set is priced 757 at 39.972 L.
3. **Robustness.** The volumes are estimates, and the best-priced set sits very close to 40 L. The headline is the most expensive set with at least a 95% chance of fitting, and the exact optimum is shown alongside with its risk (#4, #42). The results: the 735 set (A6 A8 A9 A32 A35 A38 A39 A44 A49) at a nominal 99.1%, against the 757 optimum at 51.6%. The 734 set (A6 A9 A23 A32 A35 A38 A39 A44 A48, 99.75%) is mentioned as a near-equal, safer alternative (#50).

## Working mode (set by the user)

This is the user's job-interview submission, so the user supervises every step:

- Work one step at a time. Explain the idea and the options, give a recommendation, and wait for the user's decision. Build only that step, then stop for review. Never start a later step early.
- Log every decision and every unreviewed AI finding in `DECISIONS.md`, labelled with who decided. An AI proposal that the user approved is logged as "Author, on AI proposal", never as the user's own idea.
- The user rejected the AI's proposed solution outline. The structure of the solution and of the write-up is the user's call.
- The user's write-up structure: Problem statement, Methodology, Results and discussion. Each section has two parts: the AI-assisted work and the author's decisions.
- The AI writes the code and the user reviews it. Every file the AI drafts carries the label `AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.` at the top, written as a comment. In `CLAUDE.md` it sits right after the required header. Generated files in `results/` carry no label; their origin is logged in `DECISIONS.md`. Update the step's row in the "Step review" table of `DECISIONS.md` when the user signs it off.
- The README's "Author's decisions" parts must show attribution for each item, for example "(log #16, AI proposal approved by the author)". Findings computed by the AI belong under "AI-assisted work".
- Any check or behaviour the AI adds beyond what was agreed is logged in `DECISIONS.md` as "AI", pending the author's review.
- Commit each step only after the user's review, one commit per step. The message says what the AI drafted and what the user decided, citing the log entry numbers. Never commit unreviewed work. Push `main` to `origin` after each approved step commit (#22).

## Decided so far

See `DECISIONS.md`. In short:

- Python.
- Two solutions and a comparison: a direct reference solver (regression, then knapsack), and a script where Claude solves the same problem through the API. Before writing any Claude API code, load the `claude-api` skill so model IDs and SDK usage are current.
- Python environment: a venv on Python 3.13 (`py -3.13`). The Claude SDK 1.x needs Python 3.10 or later, and the default `python` in Git Bash is Anaconda 3.8.
