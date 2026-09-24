# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from gift.data import load
from gift.estimate import STATED_VARIANCE, estimate
from gift.knapsack import exact_table
from gift.robust import (
    analyse,
    calibrate,
    candidate_sets,
    chance_normal,
    chance_positive,
    indicator,
    item_stability,
    pick,
    positive_draws,
    write_figures,
)

OPTIMUM = ["A6", "A8", "A9", "A23", "A32", "A35", "A38", "A44", "A48"]
RECOMMENDED = ["A6", "A8", "A9", "A32", "A35", "A38", "A39", "A44", "A49"]


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def result(data):
    return estimate(data)


@pytest.fixture(scope="module")
def report(data, result):
    return analyse(data, result)


@pytest.fixture(scope="module")
def calibration(data, result):
    return calibrate(data, result)  # the full run behind the README numbers (about 40 s)


def items(data, names):
    return tuple(data.names.index(n) for n in names)


class StandInFit:
    """A small stand-in for gift.estimate.Fit: volumes, variance and covariance(variance)."""

    def __init__(self, volumes, base, variance):
        self.volumes, self._base, self.variance = np.asarray(volumes, float), np.asarray(base, float), variance

    def covariance(self, variance=None):
        return (self.variance if variance is None else variance) * self._base


# Item a is tiny and often negative in draws, like A6. {a, d} has a normal chance of 0.960 but a
# positive-only chance of about 0.937; {c} has 0.92 under the estimated variance and 0.977 under 2.
STAND_IN_VOLUMES = [0.1, 39.8, 39.719, 39.346]
STAND_IN_BASE = np.diag([0.09, 0.04, 0.04, 0.01]) / 4


# --- The chance of fitting ---------------------------------------------------------------------

def test_single_item_chance_is_the_textbook_formula():
    cov = np.diag([0.04, 0.09])
    chance, mean, sd = chance_normal(indicator([(1,)], 2), np.array([10.0, 39.5]), cov)
    assert sd[0] == pytest.approx(0.3)
    assert chance[0] == pytest.approx(stats.norm.cdf((40 - 39.5) / 0.3))


def test_covariance_between_items_counts():
    # Two items whose estimates move together: the sd of the sum includes the cross term.
    cov = np.array([[0.04, 0.03], [0.03, 0.04]])
    _, _, sd = chance_normal(indicator([(0, 1)], 2), np.array([19.0, 20.0]), cov)
    assert sd[0] == pytest.approx(np.sqrt(0.04 + 0.04 + 2 * 0.03))


def test_normal_chance_matches_simulation(data, result):
    X = indicator([items(data, OPTIMUM)], len(data.names))
    chance, _, _ = chance_normal(X, result.volumes, result.covariance())
    rng = np.random.default_rng(1)
    draws = rng.multivariate_normal(result.volumes, result.covariance(), size=200_000)
    assert chance[0] == pytest.approx(((draws @ X.T) <= 40).mean(), abs=0.005)


def test_positive_draws_are_positive(result):
    draws, kept = positive_draws(result.volumes, result.covariance(), 20_000, np.random.default_rng(2))
    assert draws.shape == (20_000, 60)
    assert (draws > 0).all()
    # A6 (0.38 +/- 0.29 L) is negative in about 10% of normal draws.
    assert 0.85 < kept < 0.95


def test_positive_draws_have_the_right_covariance():
    # Far from zero, so nothing is rejected: the draws must reproduce the covariance.
    cov = np.array([[4.0, 3.0], [3.0, 4.0]])
    draws, kept = positive_draws(np.array([50.0, 50.0]), cov, 50_000, np.random.default_rng(0))
    assert kept == 1.0
    assert np.cov(draws.T) == pytest.approx(cov, abs=0.15)


def test_positive_only_chance_is_lower_for_a_set_with_a6(data, result):
    X = indicator([items(data, OPTIMUM)], len(data.names))
    normal, _, _ = chance_normal(X, result.volumes, result.covariance())
    draws, _ = positive_draws(result.volumes, result.covariance(), 100_000, np.random.default_rng(3))
    assert chance_positive(X, draws)[0] < normal[0] - 0.01


# --- Choosing a set by required chance ------------------------------------------------------------

def test_pick():
    prices = np.array([757, 751, 735, 735])
    chances = np.array([0.52, 0.55, 0.97, 0.99])
    assert pick(prices, chances, 0.5) == 0
    assert pick(prices, chances, 0.95) == 3  # tie on price: the higher chance wins
    assert pick(prices, chances, 0.999) is None
    # A chance exactly at the level counts (positive-only chances are shares, so 0.95 can occur).
    assert pick(np.array([100, 90]), np.array([0.95, 0.99]), 0.95) == 0


def test_headline_rule_is_normal_estimated_at_95():
    data = SimpleNamespace(names=["a", "b", "c", "d"], prices=np.array([3, 100, 90, 80]))
    rep = analyse(data, StandInFit(STAND_IN_VOLUMES, STAND_IN_BASE, 4.0))
    row = {r["level"]: r for r in rep["best_by_level"]}
    assert rep["recommended"]["items"] == ["a", "d"]
    assert row[0.95]["estimated_normal"]["items"] == ["a", "d"]
    assert row[0.95]["estimated_positive_only"]["items"] == ["d"]
    assert row[0.95]["stated_normal"]["items"] == ["c"]
    assert row[0.9]["estimated_normal"]["items"] == ["c"]


def test_expected_price_uses_normal_chance():
    # {a, d}: 110 x 0.960 = 105.6 (normal) but 110 x 0.937 = 103.1 (positive-only);
    # {b}: 124 x 0.841 = 104.3. So the normal version picks {a, d}; positive-only would pick {b}.
    data = SimpleNamespace(names=["a", "b", "c", "d"], prices=np.array([30, 124, 90, 80]))
    rep = analyse(data, StandInFit(STAND_IN_VOLUMES, STAND_IN_BASE, 4.0))
    assert rep["expected_price"]["estimated"]["items"] == ["a", "d"]


def test_window_is_widened_when_no_set_qualifies():
    # The best set (item 0) has about a 52% chance. The only safe set (item 1) is 50 cheaper,
    # outside the default window of 40, so the window must grow to find it.
    prices, volumes = [100, 50], np.array([39.9, 10.0])
    cov = np.diag([4.0, 0.01])
    sets, window = candidate_sets(prices, volumes, cov)
    assert window > 40
    assert (1,) in sets


def test_item_stability_without_uncertainty(data, result):
    draws = np.tile(result.volumes, (5, 1))
    st = item_stability([int(p) for p in data.prices], draws, data.names, n_draws=5)
    best, _ = exact_table(data.prices, result.volumes)
    assert st["distinct_sets"] == 1
    assert st["top_sets"][0]["items"] == [data.names[i] for i in best.items]
    assert st["top_sets"][0]["share"] == 1.0


def test_calibration_mechanics(data, result):
    cal = calibrate(data, result, reps=20, seed=5)
    assert len(cal["rows"]) == 6
    assert cal["skipped_non_positive_estimate"] + cal["rows"][0]["data_sets"] == 20
    for row in cal["rows"]:
        assert row["mean_nominal_chance"] >= row["level"]
        assert 0 <= row["actual_fit_rate"] <= 1
        p, n = row["actual_fit_rate"], row["data_sets"]
        assert row["standard_error"] == pytest.approx(math.sqrt(p * (1 - p) / n))
    headline = next(r for r in cal["rows"] if r["level"] == 0.95)
    assert sum(b["data_sets"] for b in cal["headline_by_nominal_chance"]) == headline["data_sets"]
    for b in cal["headline_by_nominal_chance"]:
        assert b["from"] <= b["mean_nominal_chance"] <= b["to"]
    assert cal["headline_including_skipped"]["data_sets"] == 20


# --- The real data --------------------------------------------------------------------------------

def test_real_optimum_and_recommendation(report):
    o, r = report["optimum_on_estimates"], report["recommended"]
    assert o["items"] == OPTIMUM and o["price"] == 757
    assert o["chance"]["estimated_normal"] == pytest.approx(0.516, abs=0.001)
    assert o["chance"]["estimated_positive_only"] == pytest.approx(0.491, abs=0.005)
    assert r["items"] == RECOMMENDED and r["price"] == 735
    assert r["volume_l"] == pytest.approx(38.221, abs=1e-3)
    assert r["chance"]["estimated_normal"] == pytest.approx(0.991, abs=0.001)
    assert r["chance"]["stated_normal"] == pytest.approx(0.999, abs=0.001)


def test_real_sd_of_total(report, result):
    o, r = report["optimum_on_estimates"], report["recommended"]
    assert o["sd_l"]["estimated"] == pytest.approx(0.706, abs=0.001)
    assert r["sd_l"]["estimated"] == pytest.approx(0.754, abs=0.001)
    ratio = math.sqrt(STATED_VARIANCE / result.variance)
    for s in (o, r):
        assert s["sd_l"]["stated"] == pytest.approx(s["sd_l"]["estimated"] * ratio)


def test_real_best_price_by_level(report):
    main = [row["estimated_normal"]["price"] for row in report["best_by_level"]]
    stated = [row["stated_normal"]["price"] for row in report["best_by_level"]]
    assert main == [757, 737, 735, 735, 735, 730]
    assert stated == [757, 737, 737, 735, 735, 735]
    assert main == sorted(main, reverse=True)
    assert report["best_by_level"][0]["estimated_positive_only"]["price"] == 751


def test_real_stated_positive_only(report):
    assert report["positive_only_acceptance"]["estimated"] == pytest.approx(0.904, abs=0.005)
    assert report["positive_only_acceptance"]["stated"] == pytest.approx(0.965, abs=0.005)
    assert report["optimum_on_estimates"]["chance"]["stated_positive_only"] == pytest.approx(0.512, abs=0.005)
    assert [r["stated_positive_only"]["price"] for r in report["best_by_level"]] == [757, 737, 737, 735, 735, 735]


def test_real_expected_price(report):
    assert report["expected_price"]["estimated"]["price"] == 734
    assert report["expected_price"]["estimated"]["price_times_chance"] == pytest.approx(732.19, abs=0.01)
    assert report["expected_price"]["stated"]["price"] == 735


def test_real_item_stability(report):
    st = report["stability"]
    for name in ["A6", "A9", "A32"]:
        assert st["item_share"][name] == 1.0
    assert st["item_share"]["A35"] == pytest.approx(0.9985)
    assert st["distinct_sets"] == 28
    assert st["top_sets"][0]["price"] == 760
    assert st["top_sets"][0]["share"] == pytest.approx(0.2855)
    assert st["share_optimum_set"] == pytest.approx(0.184)
    assert st["share_recommended_set"] == pytest.approx(0.114)
    assert st["draws_with_ties"] == 101
    assert st["share_optimum_set_including_ties"] == pytest.approx(0.1845)
    assert st["share_recommended_set_including_ties"] == pytest.approx(0.115)


def test_real_calibration(calibration):
    rows = {r["level"]: r for r in calibration["rows"]}
    assert calibration["skipped_non_positive_estimate"] == 190
    assert rows[0.95]["data_sets"] == 1810
    assert [round(rows[lv]["mean_nominal_chance"], 3) for lv in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)] == \
        [0.672, 0.903, 0.958, 0.981, 0.997, 1.000]
    assert [round(rows[lv]["actual_fit_rate"], 3) for lv in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)] == \
        [0.522, 0.818, 0.902, 0.952, 0.990, 0.999]
    assert [round(b["actual_fit_rate"], 3) for b in calibration["headline_by_nominal_chance"]] == \
        [0.901, 0.953, 0.977, 0.993]
    kept = calibration["headline_including_skipped"]
    assert kept["skipped_that_fit"] == 172
    assert kept["actual_fit_rate"] == pytest.approx(0.948, abs=0.001)


def test_write_figures(tmp_path, report):
    write_figures(report, tmp_path)
    assert (tmp_path / "price_vs_chance.png").stat().st_size > 0
    assert (tmp_path / "item_stability.png").stat().st_size > 0


def test_stated_variance_constant_is_used(report):
    assert report["variances"]["stated"] == STATED_VARIANCE
