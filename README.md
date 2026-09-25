<!-- AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md. -->

# Zalando Gift Problem

> **AI use.** Done with Claude Code (Claude Opus 5.5) under the author's step-by-step supervision; see [DECISIONS.md](DECISIONS.md).

> **Answer.** Recommended gift: **A6, A8, A9, A32, A35, A38, A39, A44, A49**, with a total price of **735** and an estimated volume of 38.2 L. It is the most expensive set with at least a 95% chance of fitting in the 40 L backpack; its nominal chance is 99.1%. The exact optimum on the estimated volumes is priced 757 (39.97 L), but it fits with only about a 52% chance. A near-equal, safer alternative is the 734 set (A6, A9, A23, A32, A35, A38, A39, A44, A48), with a nominal 99.75% chance. (The 95% rule: log #4 and #42, AI proposals approved by the author. Mentioning the 734 set: log #50, the author's. Details in section 3.)

## 1. Problem statement

### AI-assisted work

**The task, restated** (drafted by the AI, reviewed by the author, log #55):

- **Goal (the task's words):** "the most expensive present" that fits in Ahmad's 40-litre backpack.
- **Data:** prices of 60 items; their volumes are unknown. The only information is 1,000 past packages, each with its items and its measured total volume.
- **Noise:** normal, with mean 0 and, as stated, variance 2.
- **So:** estimate the item volumes, then choose the gift. The task does not say how to handle the uncertainty of the estimates; this write-up also takes it into account (section 3).

**Other readings of the task** (log #39): if items could repeat, the answer would be 106 × A6 (11,554), which rests entirely on A6's very uncertain volume; if the present had to be one item, it would be A14 or A49 (119).

### Author's decisions

- "The most expensive present" means the set of items with the highest total price (log #38, AI proposal approved by the author).
- Each item can be picked at most once (log #31, AI proposal approved by the author).
- The gift fits if its total volume is at most 40 L; exactly 40 L is allowed. Volumes simply add up, with no packing loss (log #35, AI proposal approved by the author).
- The assumptions are stated here, in the problem statement (log #36, AI proposal approved by the author).

## 2. Methodology

### AI-assisted work

- **Data (step 1)**, `gift/data.py`: builds the 1000 × 60 package-by-item table and stops on any data error, including missing items, duplicates and a table that cannot separate all 60 volumes (plus the approved extra checks, log #20). Findings: 15 repeated item combinations and two negative package volumes (#663, #790).
- **Volume estimation (step 2)**, `gift/estimate.py`: least squares on "package volume = sum of item volumes + Gaussian noise". The noise variance comes from the residuals, with a χ² test of the stated value; standard errors come from σ²(AᵀA)⁻¹; four model checks cover a fixed volume per package, scatter against package size, normality and outliers (additions: log #29).
- **Choosing the gift (step 3)**, `gift/knapsack.py`: a 0/1 knapsack, solved exactly by a table over total price (the proposed method) and by integer programming. Greedy by unit price is a comparison, and the fractional upper bound caps what any set could reach. Every set within 40 of the best price is listed for step 4.
- **Uncertainty (step 4)**, `gift/robust.py`: each set's chance of fitting, from the covariance of the volume estimates (a normal version, and a positive-only version that excludes negative volumes). The rule picks the most expensive set with at least a 95% chance. A winner's-curse check reruns the whole pipeline on 2,000 simulated data sets, and item stability serves as a diagnostic (method details: log #47).
- **Tests** (`tests/`, 85 tests): each method is checked against known answers, brute force over all subsets, or simulation.

### Author's decisions

**Data preparation (step 1):**

- Item combinations that appear more than once are kept as separate measurements (log #16, AI proposal approved by the author).
- Any failed data check stops the program with a clear error (log #17, AI proposal approved by the author).
- The two negative volumes are kept as measurement noise (log #18, AI proposal approved by the author). A negative volume cannot happen in reality and would not appear in a real data set (log #18, the author's addition).

**Volume estimation (step 2):**

- The item volumes are estimated with plain least squares, and the program stops if any estimate is 0 or below (log #24, AI proposal approved by the author).
- A fixed volume per package is estimated, tested and reported, but left out of the model, because the task defines a package's volume as the sum of its items' volumes (log #25, AI proposal approved by the author).
- All four model checks are reported (log #26, AI proposal approved by the author).
- Outputs: the volume table and the residual histogram (log #27, AI proposal approved by the author), plus histograms of item price, estimated item volume and unit price (log #27, the author's addition).

**Choosing the gift (step 3):**

- The exact table over total price is the proposed method (log #32, AI proposal approved by the author).
- Integer programming and greedy are run as well, and all three are compared on their final result and computational effort (log #32, the author's).
- The author raised greedy by unit price (log #30). It serves only as a comparison to the proposed method, with the fractional upper bound and the gap explained (log #37, AI proposal approved by the author).
- Cross-checks: integer programming against the table on the real data, and brute force on small cases in the tests (log #33, AI proposal approved by the author).
- Every set within 40 of the best price is listed, for step 4 (log #34, AI proposal approved by the author).

**Handling the uncertainty (step 4):**

- The chance of fitting is reported in both versions, normal and positive-only (log #41, AI proposal approved by the author).
- The required chance is 95%, with a table from 50% to 99.9% (log #42, AI proposal approved by the author; the 95% rule itself dates from planning, log #4).
- The winner's-curse check is included (log #43, AI proposal approved by the author).
- Item stability is kept in the write-up as a diagnostic (log #44, the author's, decided after seeing the results; the AI had recommended building it as a diagnostic).
- The figure of price against chance of fitting (log #45) and the one-line expected-price view (log #46) are included (both AI proposals approved by the author).

**Reproducibility:**

- The repository is set up so that the results can be reproduced as they are, apart from the computing times, which vary between runs: the code, the pinned requirements (`requirements.txt`), the tests and every generated result are included, and the random parts are seeded (log #57, the author's).

## 3. Results and discussion

### AI-assisted work

**Item volumes.** All 60 estimates are positive, from 0.38 L (A6) to 29.67 L (A14), with standard errors of 0.23–0.29 L ([`results/volumes.csv`](results/volumes.csv)).

![Estimated item volumes](results/figures/volumes.png)

![Item prices](results/figures/prices.png)

**Noise variance.** The data give **3.86** (standard deviation 1.96 L; 95% interval 3.53–4.23). The stated variance of 2 is rejected by a χ² test (p ≈ 9 × 10⁻⁵⁸).

![Package residuals against the two noise levels](results/figures/residuals.png)

*The AI's interpretation (log #23):* the data fit a **standard deviation** of 2, which a call such as numpy's `np.random.normal(0, 2)` would produce. The data cannot show how they were made.

**Model checks.** None points to another source of variation: no fixed volume per package (−0.06 L, p = 0.71), no growth of the scatter with package size (p = 0.62), normal residuals (p = 0.88), and no outliers beyond chance.

**Unit price.** A6 (290 per litre), A32 (87) and A9 (67) stand far apart from the rest (1–20).

![Unit price of items](results/figures/unit_prices.png)

It is not needed for an exact method, which compares whole sets. Choosing greedily by unit price is only guaranteed to be optimal when items can be cut into fractions. It is also fragile for small items: within one standard error, A6's unit price could be anywhere from 165 to 1,225 per litre.

**The gift on the estimated volumes (step 3)**, from `results/knapsack.json`. The times are one run on the author's laptop and vary between runs:

| Method | Price | Volume | Time |
|---|---|---|---|
| Exact table (proposed) | **757** | 39.97 L | 1.5 ms |
| Integer programming | **757** | 39.97 L | 46 ms |
| Greedy by unit price | 751 | 39.94 L | 0.09 ms |
| Fractional upper bound | 770.8 | – | – |

- Both exact methods find the same set (A6 A8 A9 A23 A32 A35 A38 A44 A48), which leaves only 0.028 L to spare.
- Greedy ends 6 below the optimum: it keeps A39 early and later fits A53, where the optimum uses A48.
- All three methods are fast at this size.

**The gift under uncertainty (step 4)**, from `results/uncertainty.json`:

| Set | Price | Volume | Chance of fitting |
|---|---|---|---|
| Exact optimum on the estimates | 757 | 39.97 L | **51.6%** |
| Recommended: A6 A8 A9 A32 A35 A38 A39 A44 A49 | 735 | 38.22 L | **99.1%** (nominal) |

![Price against the chance of fitting](results/figures/price_vs_chance.png)

- **By required chance:** 80% gives 737, 90–99% gives 735, and 99.9% gives 730. At 95%, both versions and both noise variances give the same 735 set.
- **Winner's curse:** the 95% rule delivers 95.2% in the 2,000 simulated data sets (94.8% counting the 190 skipped ones). The printed chance of a chosen set is optimistic, mainly just above the threshold. Sets printed at 99–99.5%, like the recommended one, actually fit 97.7% of the time.
- **Expected price** (log #46): under variance 3.86, a 734 set edges out the 735 set on price × chance (732.2 against 728.3).

**Item stability** (a diagnostic, log #44): A6, A9 and A32 are in every best set, A35 in 99.9% and A44 in 97%. The most frequent winner is a 760 set that does not fit on the point estimates.

![How often each item is in the best set](results/figures/item_stability.png)

**Limitations of the data:**

- Volumes are never observed directly; they are inferred from noisy package totals (±0.23–0.29 L each).
- The stated noise (variance 2) does not match the data (about 3.86).
- The smallest items are poorly determined: A6 is 0.38 ± 0.29 L, and it is in every good gift.
- The data contain only package totals, so volumes are assumed to simply add up.

### Author's decisions

- **The stated noise variance of 2 is rejected** on the basis of the calculation above (log #19, the author's). The estimated variance is used for all uncertainty calculations, and the stated value is shown only for comparison (log #5, AI proposal approved by the author).
- The reading that 2 was meant as the standard deviation is included, labelled as the AI's interpretation (log #23, the author's decision to include it).
- **Unit price is not needed to solve this problem, but it can be useful for other problems** (log #28, the author's).
- The answer on the estimated volumes comes from the exact methods, and greedy is reported only for comparison (log #37, AI proposal approved by the author). All three methods are compared on result and computational effort (log #32, the author's).
- **Recommended gift: A6 A8 A9 A32 A35 A38 A39 A44 A49, total price 735, estimated volume 38.22 L, with a nominal 99.1% chance of fitting in 40 L.** It is the most expensive set with at least a 95% chance. The exact optimum on the estimates (757, 39.97 L) is shown alongside with its risk: a 51.6% chance (log #4, #42, AI proposals approved by the author).
- The recommended set's 99.1% is reported as a nominal chance, because the winner's-curse check shows that a chosen set's nominal chance is optimistic, even though the 95% rule itself delivers about 95% (log #49, AI interpretation approved by the author).
- **A near-equal, safer alternative: A6 A9 A23 A32 A35 A38 A39 A44 A48, price 734, estimated volume 37.98 L, with a nominal 99.75% chance of fitting.** It is mentioned alongside, and the recommendation stays with 735 (log #50, the author's).
- The limitations are stated briefly and cover only the data set (log #58, the author's).
- **Computational time is not a problem at this size. For industry-size problems, though, it is always a good start to estimate the computational time of different solutions on a small data set, and then choose the most accurate and efficient one** (log #40, the author's).
