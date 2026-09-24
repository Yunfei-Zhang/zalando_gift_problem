# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Choose the gift on the estimated volumes: a 0/1 knapsack (step 3).

Problem: maximise sum(price_i * x_i) subject to sum(volume_i * x_i) <= 40 L,
with x_i in {0, 1}. The volumes are the step 2 estimates; how their uncertainty
changes the choice is step 4.

Methods (#32):
    exact_table      For every total price P, the smallest total volume of a set
                     with price exactly P. Prices are whole numbers, so this is
                     exact and never rounds a volume. The proposed method.
    integer_program  The same problem solved by an integer-programming solver
                     (scipy.optimize.milp, HiGHS). Exact; an independent check (#33).
    greedy           Take items in order of price per litre while they fit.
                     A heuristic, used only as a comparison (#37).
Also computed: the fractional upper bound (items may be cut, so no set can do
better) and every set within 40 of the best price (#34).

Decisions applied here (DECISIONS.md records who made each one):
    #31  Each item at most once. (AI proposal, approved by the author.)
    #32  Three methods, with results and computational effort. (Exact table as
         the proposed method: AI proposal, approved. Integer programming, greedy
         and the effort comparison: the author's.)
    #33  Cross-checks. (AI proposal, approved by the author.)
    #34  Near-optimal list within 40 of the best price. (AI proposal, approved.)
    #35  Capacity <= 40 L inclusive; volumes add up. (AI proposal, approved.)
    #37  Greedy as a comparison. (AI proposal, approved by the author; the
         author raised greedy in #30.)
    #38  "The most expensive present" = the set with the highest total price.
         (AI reading, approved by the author.)
    #39  alternative_readings(), results/knapsack.json, and the tie-break
         "smallest volume among sets with the best price". (AI additions,
         approved by the author.)

Run `python -m gift.knapsack` to print the report and write results/knapsack.json.
"""

import json
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from gift.data import load
from gift.estimate import RESULTS_DIR, estimate

CAPACITY = 40.0
TOLERANCE = 1e-9  # a set fits if its volume is at most CAPACITY + TOLERANCE (float rounding only)
WINDOW = 40       # near-optimal list: every set within this much of the best price (#34)


@dataclass(frozen=True)
class Selection:
    items: tuple    # item indices, ascending
    price: int
    volume: float


def _check_volumes(volumes):
    """Every method assumes positive volumes (unit prices and the fractional bound need it)."""
    if any(v <= 0 for v in volumes):
        raise ValueError("every volume must be positive")


def _selection(indices, prices, volumes):
    items = tuple(sorted(int(i) for i in indices))
    return Selection(items, int(sum(int(prices[i]) for i in items)), math.fsum(float(volumes[i]) for i in items))


def exact_table(prices, volumes, capacity=CAPACITY):
    """Exact 0/1 knapsack by a table over total price.

    smallest[P] is the smallest total volume of any set whose prices add up to
    exactly P. Adding item k with price p and volume v, a set with price P can
    either skip k (smallest[P]) or use it (smallest[P - p] + v). The answer is the
    largest P whose smallest volume fits. Among sets with that price, the one with
    the smallest volume is returned.
    """
    _check_volumes(volumes)
    prices = [int(p) for p in prices]
    total = sum(prices)
    smallest = np.full(total + 1, np.inf)
    smallest[0] = 0.0
    used = np.zeros((len(prices), total + 1), dtype=bool)  # used[k, P]: item k improved P
    for k, (p, v) in enumerate(zip(prices, volumes)):
        with_item = smallest[: total + 1 - p] + v  # computed from the table before item k
        better = with_item < smallest[p:]
        used[k, p:] = better
        smallest[p:] = np.where(better, with_item, smallest[p:])
    P = int(np.flatnonzero(smallest <= capacity + TOLERANCE).max())
    chosen = []
    for k in range(len(prices) - 1, -1, -1):
        if used[k, P]:
            chosen.append(k)
            P -= prices[k]
    return _selection(chosen, prices, volumes), {"table_cells": len(prices) * (total + 1)}


def integer_program(prices, volumes, capacity=CAPACITY):
    """Exact 0/1 knapsack by integer programming (scipy.optimize.milp, HiGHS).

    The solver's own feasibility tolerance (about 1e-7) is looser than TOLERANCE
    and cannot be set through scipy, so the returned set is re-checked here.
    Among several sets with the best price it may return any one of them.
    """
    _check_volumes(volumes)
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(prices)
    result = milp(
        c=-prices,
        constraints=LinearConstraint(volumes[None, :], -np.inf, capacity),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"integer program failed: {result.message}")
    selection = _selection(np.flatnonzero(result.x > 0.5), prices, volumes)
    if selection.volume > capacity + TOLERANCE:
        raise RuntimeError(
            f"integer program returned a set of {selection.volume!r} L, over the capacity "
            "(the solver's feasibility tolerance is looser than this project's)"
        )
    return selection, {"branch_and_bound_nodes": int(result.mip_node_count)}


def _unit_price_order(prices, volumes):
    """Item indices by price per litre, highest first (ties by item order)."""
    return sorted(range(len(prices)), key=lambda i: (-prices[i] / volumes[i], i))


def greedy(prices, volumes, capacity=CAPACITY):
    """Heuristic: take items in order of price per litre, each one that still fits."""
    _check_volumes(volumes)
    chosen, used = [], 0.0
    order = _unit_price_order(prices, volumes)
    for i in order:
        if used + volumes[i] <= capacity + TOLERANCE:
            chosen.append(i)
            used += volumes[i]
    return _selection(chosen, prices, volumes), {"items_sorted": len(order), "fit_checks": len(order)}


def fractional_bound(prices, volumes, capacity=CAPACITY):
    """Best price if items could be cut into fractions. No real set can beat it."""
    _check_volumes(volumes)
    total, room = 0.0, capacity
    for i in _unit_price_order(prices, volumes):
        if volumes[i] <= room:
            total += prices[i]
            room -= volumes[i]
        else:
            return total + prices[i] * room / volumes[i]
    return total


def near_optimal(prices, volumes, best_price, window=WINDOW, capacity=CAPACITY):
    """Every set that fits and has price >= best_price - window (#34).

    Depth-first search over items in unit-price order. A branch is cut when even
    the fractional bound of the remaining items cannot reach the threshold, so no
    qualifying set is missed. Sorted by price (high first), then volume.
    """
    _check_volumes(volumes)
    order = _unit_price_order(prices, volumes)
    p = [int(prices[i]) for i in order]
    v = [float(volumes[i]) for i in order]
    n, threshold = len(order), best_price - window
    found = []

    def bound(k, room):
        total = 0.0
        for j in range(k, n):
            if v[j] <= room:
                total += p[j]
                room -= v[j]
            else:
                return total + p[j] * room / v[j]
        return total

    def search(k, price, volume, chosen):
        # The bound gets the same tolerance as the fit check, or a set exactly at the
        # threshold with a volume just over 40 L (within TOLERANCE) could be cut.
        if price + bound(k, capacity + TOLERANCE - volume) < threshold - 1e-9:
            return
        if k == n:
            found.append(_selection([order[j] for j in chosen], prices, volumes))
            return
        if volume + v[k] <= capacity + TOLERANCE:
            search(k + 1, price + p[k], volume + v[k], chosen + [k])
        search(k + 1, price, volume, chosen)

    search(0, 0, 0.0, [])
    return sorted(found, key=lambda s: (-s.price, s.volume, s.items))


def alternative_readings(prices, volumes, capacity=CAPACITY):
    """What the other readings of the task would give (for the assumption in #31).

    Repeats allowed: an unbounded knapsack, solved exactly by a table over total
    price. Single item: every item that fits and has the highest price among
    those that fit, smallest volume first (an empty list if none fits).
    """
    _check_volumes(volumes)
    prices = [int(p) for p in prices]
    top = (int((capacity + TOLERANCE) / min(volumes)) + 1) * max(prices)
    smallest = np.full(top + 1, np.inf)
    smallest[0] = 0.0
    last = np.full(top + 1, -1)
    for P in range(1, top + 1):
        for k, (p, v) in enumerate(zip(prices, volumes)):
            if p <= P and smallest[P - p] + v < smallest[P]:
                smallest[P] = smallest[P - p] + v
                last[P] = k
    P = int(np.flatnonzero(smallest <= capacity + TOLERANCE).max())
    counts = {}
    while P > 0:
        k = int(last[P])
        counts[k] = counts.get(k, 0) + 1
        P -= prices[k]
    fitting = [i for i in range(len(prices)) if volumes[i] <= capacity + TOLERANCE]
    single = []
    if fitting:
        best = max(prices[i] for i in fitting)
        single = sorted((i for i in fitting if prices[i] == best), key=lambda i: (volumes[i], i))
    return {"repeats_allowed": counts, "single_item": single}


def _timed(method, prices, volumes, repeats):
    """Run a method `repeats` times; return its result and the median run time in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = method(prices, volumes)
        times.append(time.perf_counter() - start)
    return out, float(np.median(times))


def solve(data, volumes):
    """All three methods, the bound, the near-optimal list and the alternative readings."""
    prices = [int(p) for p in data.prices]
    names = data.names

    def describe(selection):
        return {
            "items": [names[i] for i in selection.items],
            "price": selection.price,
            "volume_l": selection.volume,
            "slack_l": CAPACITY - selection.volume,
        }

    methods = {}
    for name, method, repeats in [
        ("exact_table", exact_table, 20),
        ("integer_program", integer_program, 20),
        ("greedy", greedy, 1000),
    ]:
        (selection, work), seconds = _timed(method, prices, volumes, repeats)
        methods[name] = {**describe(selection), "median_seconds": seconds, "runs_timed": repeats, "work": work}

    best = methods["exact_table"]["price"]
    near = near_optimal(prices, volumes, best)
    alt = alternative_readings(prices, volumes)
    return {
        "capacity_l": CAPACITY,
        "assumptions": ["each item at most once (#31)", "total volume <= 40 L, 40 allowed; volumes add up (#35)"],
        "methods": methods,
        # Both exact methods must reach the same best price. On a tie they may pick
        # different sets, so the item lists are compared separately.
        "exact_methods_agree": methods["exact_table"]["price"] == methods["integer_program"]["price"],
        "exact_methods_same_items": methods["exact_table"]["items"] == methods["integer_program"]["items"],
        "greedy_gap": best - methods["greedy"]["price"],
        "fractional_upper_bound": fractional_bound(prices, volumes),
        "near_optimal_window": WINDOW,
        "near_optimal": [describe(s) for s in near],
        "sets_at_best_price": sum(1 for s in near if s.price == best),
        "alternative_readings": {
            "repeats_allowed": {
                "items": {names[k]: c for k, c in alt["repeats_allowed"].items()},
                "price": sum(prices[k] * c for k, c in alt["repeats_allowed"].items()),
            },
            "single_item": {
                "items": [names[i] for i in alt["single_item"]],
                "price": prices[alt["single_item"][0]] if alt["single_item"] else None,
            },
        },
    }


def print_report(report):
    m = report["methods"]
    labels = {"exact_table": "Exact table (proposed)", "integer_program": "Integer programming", "greedy": "Greedy by unit price"}
    print(f"Capacity {report['capacity_l']:g} L; each item at most once")
    print(f"{'Method':26} {'Price':>5} {'Volume':>9} {'Time':>10}  Work")
    for key, label in labels.items():
        r = m[key]
        work = ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in r["work"].items())
        print(f"{label:26} {r['price']:>5} {r['volume_l']:>7.3f} L {r['median_seconds'] * 1000:>7.3f} ms  {work}")
        print(f"{'':26} items: {' '.join(r['items'])}")
    print(f"Exact methods agree: {report['exact_methods_agree']} (same items: {report['exact_methods_same_items']}); "
          f"greedy is {report['greedy_gap']} below the optimum")
    print(f"Fractional upper bound: {report['fractional_upper_bound']:.1f}")
    near = report["near_optimal"]
    print(f"Sets within {report['near_optimal_window']} of the best price: {len(near)} "
          f"({report['sets_at_best_price']} at the best price)")
    for s in near[:10]:
        print(f"  {s['price']}  {s['volume_l']:.3f} L  {' '.join(s['items'])}")
    alt = report["alternative_readings"]
    rep = ", ".join(f"{c} x {k}" for k, c in alt["repeats_allowed"]["items"].items())
    single = " or ".join(alt["single_item"]["items"]) or "none fits"
    print(f"Other readings: repeats allowed -> {rep} = {alt['repeats_allowed']['price']}; "
          f"single item -> {single} at {alt['single_item']['price']}")


def main(out_dir=RESULTS_DIR):
    data = load()
    report = solve(data, estimate(data).volumes)
    print_report(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "knapsack.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_dir / 'knapsack.json'}")


if __name__ == "__main__":
    main()
