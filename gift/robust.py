# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Handle the uncertainty in the volume estimates (step 4).

Uncertainty model: with a flat prior and the noise variance plugged in as if known,
the true item volumes are distributed N(v_hat, Cov), Cov = sigma^2 (A^T A)^-1, from
step 2. The main sigma^2 is the estimated one; the stated 2 is for comparison
(#5, #19).

Chance that a set S fits in 40 L (#41), in two versions:
    normal         Phi((40 - sum_S v_hat) / sd_S), with sd_S^2 = x^T Cov x. The full
                   covariance is used, so how the item estimates move together counts.
    positive-only  The same chance, but volumes cannot be negative: draws from
                   N(v_hat, Cov) in which any item volume is 0 or below are thrown
                   out (Monte Carlo).
Headline rule (#4, #42): the most expensive set with a chance of at least 95%
(normal version, estimated variance).

Which sets are searched: every set whose estimated volume fits and whose price is
within WINDOW of the best (step 3). A set whose estimated volume is over 40 L has
a normal chance below 50%, so at levels of 50% and above only sets that fit on the
estimates can qualify; the list holds every such set within the window, so the
most expensive qualifying set is in it as long as the list holds at least one.
Otherwise the window is widened. This argument covers the normal version (under
either variance). For the positive-only version it is not a proof; on the real
data no set outside the list reaches 50% positive-only.

Winner's-curse check (#43): simulate new package data from the fitted model
(truth = v_hat, noise variance = the estimated one), rerun estimation and
selection, and count how often the chosen set really fits. At the headline level
the result is also split by the chosen set's nominal chance, because the
optimism sits mostly in sets chosen just above the threshold.

Item stability (#44, a diagnostic built as the AI recommended; the author kept it
in the write-up after seeing the results): draw plausible volumes
(positive-only, estimated variance), solve the exact knapsack for each draw, and
count how often each item is in the best set. Draws in which several sets tie on
the best price are counted; the exact method then picks the smallest volume.

Decisions applied here (DECISIONS.md records who made each one):
    #41, #42, #43, #45, #46  AI proposals, approved by the author.
    #44  Kept by the author after seeing the results.
    #47, #49  AI method choices and interpretation, approved by the author.

Run `python -m gift.robust` to print the report and write results/uncertainty.json
and two figures.
"""

import json
import math
from collections import Counter

import numpy as np
from scipy import stats

from gift.data import load
from gift.estimate import RESULTS_DIR, STATED_VARIANCE, estimate, fit
from gift.knapsack import CAPACITY, TOLERANCE, WINDOW, exact_table, near_optimal

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
HEADLINE_LEVEL = 0.95
SEED = 2026
POSITIVE_DRAWS = 100_000   # draws kept for the positive-only chance
STABILITY_DRAWS = 2_000    # draws solved exactly for item stability
CALIBRATION_REPS = 2_000   # simulated data sets for the winner's-curse check
NOMINAL_BINS = (0.95, 0.97, 0.99, 0.995, 1.0)  # split of the headline-level check by nominal chance


def indicator(sets, n_items):
    """0/1 matrix with one row per set."""
    X = np.zeros((len(sets), n_items))
    for k, items in enumerate(sets):
        X[k, list(items)] = 1.0
    return X


def chance_normal(X, volumes, cov, capacity=CAPACITY):
    """Normal chance that each set (row of X) fits, with its mean and sd of total volume."""
    mean = X @ volumes
    sd = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
    return stats.norm.cdf((capacity - mean) / sd), mean, sd


def positive_draws(volumes, cov, n, rng, batch=50_000):
    """Draws from N(volumes, cov) with every volume positive (rejection).

    Returns the draws and the share of raw draws that were kept.
    """
    L = np.linalg.cholesky(cov)
    kept, n_kept, n_drawn = [], 0, 0
    while n_kept < n:
        draws = volumes + rng.standard_normal((batch, len(volumes))) @ L.T
        draws = draws[(draws > 0).all(axis=1)]
        kept.append(draws)
        n_kept += len(draws)
        n_drawn += batch
    return np.vstack(kept)[:n], n_kept / n_drawn


def chance_positive(X, draws, capacity=CAPACITY):
    """Share of the (positive-only) draws in which each set fits."""
    return ((draws @ X.T) <= capacity + TOLERANCE).mean(axis=0)


def pick(prices, chances, level):
    """Index of the most expensive set with chance >= level (ties: higher chance), or None."""
    ok = np.flatnonzero(chances >= level)
    if len(ok) == 0:
        return None
    return int(max(ok, key=lambda k: (prices[k], chances[k])))


def candidate_sets(prices, volumes, cov, levels=LEVELS):
    """The near-optimal sets, with the window widened until every level has a qualifying set."""
    best, _ = exact_table(prices, volumes)
    window = WINDOW
    while True:
        sets = [s.items for s in near_optimal(prices, volumes, best.price, window)]
        chances, _, _ = chance_normal(indicator(sets, len(volumes)), volumes, cov)
        if all(chances.max() >= level for level in levels) or window >= best.price:
            return sets, window
        window *= 2


def calibrate(data, result, reps=CALIBRATION_REPS, seed=SEED, levels=LEVELS):
    """Winner's-curse check (#43) for the rule "most expensive set with normal chance >= level".

    Truth: v_hat from the real data and the estimated noise variance. For each
    simulated data set the whole pipeline is rerun. A simulated data set in which
    some estimated volume is 0 or below would stop the real pipeline (#24), so it
    is skipped and counted.
    """
    rng = np.random.default_rng(seed)
    truth, sd = result.volumes, math.sqrt(result.variance)
    prices = [int(p) for p in data.prices]
    nominal = {level: [] for level in levels}
    fits = {level: [] for level in levels}
    skipped, skipped_fits = 0, []

    def choose(volumes, cov, level_list):
        sets, _ = candidate_sets(prices, volumes, cov, level_list)
        X = indicator(sets, len(prices))
        chances, _, _ = chance_normal(X, volumes, cov)
        set_prices, true_volumes = X @ np.array(prices), X @ truth
        for level in level_list:
            k = pick(set_prices, chances, level)
            assert k is not None, "candidate_sets should always give a qualifying set"
            yield level, chances[k], true_volumes[k] <= CAPACITY + TOLERANCE

    for _ in range(reps):
        b = data.A @ truth + rng.normal(0, sd, size=len(data.b))
        sim = fit(data.A, b)
        if (sim.volumes <= 0).any():
            # The real pipeline would stop here (#24). For a sensitivity check only, the
            # headline rule is also run with the non-positive estimates set to 0.0001 L.
            skipped += 1
            clipped = np.where(sim.volumes <= 0, 1e-4, sim.volumes)
            for _, _, ok in choose(clipped, sim.covariance(), [HEADLINE_LEVEL]):
                skipped_fits.append(ok)
            continue
        for level, chance, ok in choose(sim.volumes, sim.covariance(), levels):
            nominal[level].append(chance)
            fits[level].append(ok)

    def summary(nominal_values, fit_values):
        rate = float(np.mean(fit_values))
        n = len(fit_values)
        return {
            "mean_nominal_chance": float(np.mean(nominal_values)),
            "actual_fit_rate": rate,
            "standard_error": math.sqrt(rate * (1 - rate) / n),
            "data_sets": n,
        }

    rows = [{"level": level, **summary(nominal[level], fits[level])} for level in levels]
    by_nominal = []
    if HEADLINE_LEVEL in levels:
        values, hits = np.array(nominal[HEADLINE_LEVEL]), np.array(fits[HEADLINE_LEVEL])
        for low, high in zip(NOMINAL_BINS, NOMINAL_BINS[1:]):
            inside = (values >= low) & ((values < high) if high < 1.0 else (values <= high))
            if inside.any():
                by_nominal.append({"from": low, "to": high, **summary(values[inside], hits[inside])})
    headline_with_skipped = None
    if HEADLINE_LEVEL in levels:
        all_fits = fits[HEADLINE_LEVEL] + skipped_fits
        rate = float(np.mean(all_fits))
        headline_with_skipped = {
            "actual_fit_rate": rate,
            "standard_error": math.sqrt(rate * (1 - rate) / len(all_fits)),
            "data_sets": len(all_fits),
            "skipped_that_fit": int(sum(skipped_fits)),
        }
    return {
        "reps": reps,
        "skipped_non_positive_estimate": skipped,
        "rows": rows,
        "headline_by_nominal_chance": by_nominal,
        "headline_including_skipped": headline_with_skipped,
    }


def item_stability(prices, draws, names, n_draws=STABILITY_DRAWS):
    """How often each item, and each set, is the exact optimum across plausible volumes (#44).

    When several sets tie on the best price, the exact method picks the smallest
    volume; `tie_counts` also credits every set that reaches the best price.
    """
    item_counts, set_counts, tie_counts = Counter(), Counter(), Counter()
    draws_with_ties = 0
    for volumes in draws[:n_draws]:
        best, _ = exact_table(prices, volumes)
        item_counts.update(best.items)
        set_counts[best.items] += 1
        tied = [s.items for s in near_optimal(prices, volumes, best.price, window=0)]
        draws_with_ties += len(tied) > 1
        tie_counts.update(tied)
    return {
        "draws": n_draws,
        "draws_with_ties": draws_with_ties,
        "tie_share": {s: c / n_draws for s, c in tie_counts.items()},
        "item_share": {names[i]: item_counts[i] / n_draws for i in sorted(item_counts, key=lambda i: -item_counts[i])},
        "top_sets": [
            {"items": [names[i] for i in s], "price": int(sum(prices[i] for i in s)), "share": c / n_draws}
            for s, c in set_counts.most_common(8)
        ],
        "distinct_sets": len(set_counts),
        "set_share": {s: c / n_draws for s, c in set_counts.items()},
    }


def analyse(data, result, seed=SEED):
    """Everything in step 4 except the winner's-curse check."""
    rng = np.random.default_rng(seed)
    prices = [int(p) for p in data.prices]
    names, volumes = data.names, result.volumes
    variances = {"estimated": result.variance, "stated": STATED_VARIANCE}
    sets, window = candidate_sets(prices, volumes, result.covariance())
    X = indicator(sets, len(prices))
    set_prices = X @ np.array(prices)

    table, chances, draws_by_variance, acceptance = [], {}, {}, {}
    for key, variance in variances.items():
        cov = result.covariance(variance)
        normal, mean, sd = chance_normal(X, volumes, cov)
        draws, acceptance[key] = positive_draws(volumes, cov, POSITIVE_DRAWS, rng)
        draws_by_variance[key] = draws
        chances[key] = {"normal": normal, "positive_only": chance_positive(X, draws), "sd": sd, "mean": mean}
    for level in LEVELS:
        row = {"level": level}
        for key in variances:
            for version in ("normal", "positive_only"):
                k = pick(set_prices, chances[key][version], level)
                row[f"{key}_{version}"] = None if k is None else {
                    "price": int(set_prices[k]),
                    "items": [names[i] for i in sets[k]],
                    "chance": float(chances[key][version][k]),
                }
        table.append(row)

    def describe(k):
        return {
            "items": [names[i] for i in sets[k]],
            "price": int(set_prices[k]),
            "volume_l": float(chances["estimated"]["mean"][k]),
            "sd_l": {key: float(chances[key]["sd"][k]) for key in variances},
            "chance": {
                f"{key}_{version}": float(chances[key][version][k])
                for key in variances for version in ("normal", "positive_only")
            },
        }

    main = chances["estimated"]["normal"]
    recommended = pick(set_prices, main, HEADLINE_LEVEL)
    optimum = int(np.argmax(set_prices))  # the step 3 optimum is the most expensive set in the list
    expected = {key: int(np.argmax(set_prices * chances[key]["normal"])) for key in variances}
    stability = item_stability(prices, draws_by_variance["estimated"], names)
    set_share = stability.pop("set_share")
    tie_share = stability.pop("tie_share")
    return {
        "levels": list(LEVELS),
        "headline_level": HEADLINE_LEVEL,
        "variances": variances,
        "window": window,
        "positive_only_acceptance": acceptance,
        "optimum_on_estimates": describe(optimum),
        "recommended": describe(recommended),
        "best_by_level": table,
        "expected_price": {
            key: {**describe(k), "price_times_chance": float(set_prices[k] * chances[key]["normal"][k])}
            for key, k in expected.items()
        },
        "sets": [describe(k) for k in range(len(sets))],
        "stability": {
            **stability,
            "share_optimum_set": set_share.get(tuple(sets[optimum]), 0.0),
            "share_recommended_set": set_share.get(tuple(sets[recommended]), 0.0),
            "share_optimum_set_including_ties": tie_share.get(tuple(sets[optimum]), 0.0),
            "share_recommended_set_including_ties": tie_share.get(tuple(sets[recommended]), 0.0),
        },
    }


def write_figures(report, figure_dir):
    from gift import figures

    figure_dir.mkdir(parents=True, exist_ok=True)
    figures.price_vs_chance(
        [s["chance"]["estimated_normal"] for s in report["sets"]],
        [s["price"] for s in report["sets"]],
        optimum=(report["optimum_on_estimates"]["chance"]["estimated_normal"], report["optimum_on_estimates"]["price"]),
        recommended=(report["recommended"]["chance"]["estimated_normal"], report["recommended"]["price"]),
        level=report["headline_level"],
        path=figure_dir / "price_vs_chance.png",
    )
    figures.item_share(report["stability"]["item_share"], report["stability"]["draws"],
                       figure_dir / "item_stability.png")


def print_report(report):
    o, r = report["optimum_on_estimates"], report["recommended"]
    for label, s in [("Exact optimum on the estimates", o), ("Recommended (>= 95%)", r)]:
        c = s["chance"]
        print(f"{label}: {s['price']} at {s['volume_l']:.3f} L (sd {s['sd_l']['estimated']:.3f} L): "
              f"chance {c['estimated_normal']:.3f} normal, {c['estimated_positive_only']:.3f} positive-only; "
              f"stated variance {c['stated_normal']:.3f} / {c['stated_positive_only']:.3f}")
        print(f"  {' '.join(s['items'])}")
    print(f"Sets searched: {len(report['sets'])} (window {report['window']}); positive-only draws kept: "
          + ", ".join(f"{k} {v:.1%}" for k, v in report["positive_only_acceptance"].items()))
    print("Best price by required chance (estimated normal / estimated positive-only / stated normal / stated positive-only):")
    for row in report["best_by_level"]:
        cells = [row[k] for k in ("estimated_normal", "estimated_positive_only", "stated_normal", "stated_positive_only")]
        print(f"  {row['level']:>6.1%}: " + " / ".join("-" if c is None else f"{c['price']} ({c['chance']:.3f})" for c in cells))
    for key, e in report["expected_price"].items():
        print(f"Best price x chance ({key} variance, normal): {e['price']} set, {e['price_times_chance']:.1f}")
    st = report["stability"]
    print(f"Item stability over {st['draws']} draws: {st['distinct_sets']} distinct optimal sets; "
          f"exact optimum wins {st['share_optimum_set']:.1%}, recommended set {st['share_recommended_set']:.1%}; "
          f"{st['draws_with_ties']} draws with a tie on the best price (counting every tied set: optimum "
          f"{st['share_optimum_set_including_ties']:.1%}, recommended {st['share_recommended_set_including_ties']:.1%})")
    print("  " + ", ".join(f"{k} {v:.0%}" for k, v in st["item_share"].items()))
    if "calibration" in report:
        cal = report["calibration"]
        print(f"Winner's curse ({cal['reps']} simulated data sets, {cal['skipped_non_positive_estimate']} skipped):")
        for row in cal["rows"]:
            print(f"  level {row['level']:.1%}: nominal {row['mean_nominal_chance']:.3f}, "
                  f"actual {row['actual_fit_rate']:.3f} +/- {row['standard_error']:.3f} (n = {row['data_sets']})")
        w = cal["headline_including_skipped"]
        print(f"  at {report['headline_level']:.0%}, keeping the skipped data sets (non-positive estimates set to "
              f"0.0001 L): actual {w['actual_fit_rate']:.3f} +/- {w['standard_error']:.3f} (n = {w['data_sets']}; "
              f"{w['skipped_that_fit']} of the {cal['skipped_non_positive_estimate']} skipped fit)")
        print(f"  at {report['headline_level']:.0%}, split by the chosen set's nominal chance:")
        for row in cal["headline_by_nominal_chance"]:
            print(f"    {row['from']:.1%} to {row['to']:.1%}: nominal {row['mean_nominal_chance']:.4f}, "
                  f"actual {row['actual_fit_rate']:.3f} +/- {row['standard_error']:.3f} (n = {row['data_sets']})")


def main(out_dir=RESULTS_DIR, reps=CALIBRATION_REPS):
    data = load()
    result = estimate(data)
    report = analyse(data, result)
    report["calibration"] = calibrate(data, result, reps=reps)
    print_report(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "uncertainty.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    write_figures(report, out_dir / "figures")
    print(f"Wrote {out_dir / 'uncertainty.json'} and figures in {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
