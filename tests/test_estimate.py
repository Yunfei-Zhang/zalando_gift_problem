# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
import csv
import json

import numpy as np
import pytest

from gift.data import GiftData, load
from gift.estimate import (
    STATED_VARIANCE,
    EstimationError,
    checks,
    estimate,
    fit,
    intercept_test,
    normality_test,
    outlier_check,
    size_test,
    variance_test,
    write_outputs,
)


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def result(data):
    return estimate(data)


@pytest.fixture(scope="module")
def report(data, result):
    return checks(data, result)


def made_up_volumes(p, seed):
    return np.random.default_rng(seed).uniform(0.5, 30, size=p)


# --- The estimator on made-up data with known volumes --------------------------------------

def test_exact_recovery_without_noise(data):
    v = made_up_volumes(60, seed=0)
    r = fit(data.A, data.A @ v)
    assert np.allclose(r.volumes, v, atol=1e-9)
    assert r.rss < 1e-12


def test_recovery_with_noise_of_sd_2(data):
    rng = np.random.default_rng(1)
    v = made_up_volumes(60, seed=1)
    r = fit(data.A, data.A @ v + rng.normal(0, 2, size=len(data.b)))
    errors_in_se = (r.volumes - v) / r.standard_errors(4.0)
    assert np.max(np.abs(errors_in_se)) < 4


def rejection_rate(A, noise_sd, reps=200, seed=3):
    """How often the variance test rejects the stated variance 2 at the 5% level."""
    rng = np.random.default_rng(seed)
    v = made_up_volumes(A.shape[1], seed)
    rejected = 0
    for _ in range(reps):
        b = A @ v + rng.normal(0, noise_sd, size=A.shape[0])
        rejected += variance_test(fit(A, b))["p_value"] < 0.05
    return rejected / reps


def test_variance_test_keeps_its_level_when_variance_2_is_true(data):
    # When the stated variance is right, a 5% test should reject about 5% of the time.
    assert 0.01 <= rejection_rate(data.A, np.sqrt(2)) <= 0.10


def test_variance_test_always_rejects_when_sd_is_2(data):
    assert rejection_rate(data.A, 2.0) == 1.0


def test_variance_test_is_two_sided(data):
    # Noise that is too small must be rejected as well as noise that is too large.
    rng = np.random.default_rng(8)
    b = data.A @ made_up_volumes(60, seed=8) + rng.normal(0, 0.7, size=len(data.b))
    assert variance_test(fit(data.A, b))["p_value"] < 1e-6


def test_intercept_test_finds_a_fixed_volume(data):
    rng = np.random.default_rng(4)
    b = 1.0 + data.A @ made_up_volumes(60, seed=4) + rng.normal(0, 2, size=len(data.b))
    r = intercept_test(data.A, b, fit(data.A, b).volumes)
    assert r["p_value"] < 1e-3
    assert r["fixed_volume"] == pytest.approx(1.0, abs=0.5)


def test_size_test_finds_scatter_growing_with_package_size(data):
    rng = np.random.default_rng(5)
    sizes = data.A.sum(axis=1)
    b = data.A @ made_up_volumes(60, seed=5) + rng.normal(0, 1, size=len(data.b)) * np.sqrt(sizes)
    assert size_test(fit(data.A, b), data.A)["p_value"] < 1e-3


def test_normality_test_finds_heavy_tailed_noise(data):
    rng = np.random.default_rng(7)
    b = data.A @ made_up_volumes(60, seed=7) + 2 * rng.standard_t(3, size=len(data.b))
    assert normality_test(fit(data.A, b))["shapiro_p"] < 1e-3


def test_outlier_check_finds_a_planted_outlier(data):
    rng = np.random.default_rng(6)
    b = data.A @ made_up_volumes(60, seed=6) + rng.normal(0, 2, size=len(data.b))
    b[9] += 15.0
    out = outlier_check(fit(data.A, b))
    assert out["worst_package"] == 10
    assert out["bonferroni_p"] < 1e-6


def test_non_positive_estimate_stops_with_clear_error():
    tiny = GiftData(
        names=["X", "Y"],
        prices=np.array([1, 1]),
        A=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        b=np.array([-1.0, 2.0, 1.0]),
    )
    with pytest.raises(EstimationError, match="0 or below for X"):
        estimate(tiny)


def test_zero_estimate_also_stops():
    tiny = GiftData(names=["X", "Y"], prices=np.array([1, 1]), A=np.eye(2), b=np.array([0.0, 2.0]))
    with pytest.raises(EstimationError, match="0 or below for X"):
        estimate(tiny)


def test_fit_internals(data, result):
    # The leverages of a least-squares fit add up to the number of parameters.
    assert result.leverage.sum() == pytest.approx(60)
    # Standard errors scale with the square root of the variance.
    assert np.allclose(result.standard_errors(8.0), 2 * result.standard_errors(2.0))
    # They match the textbook formula computed independently.
    direct = np.sqrt(np.diag(STATED_VARIANCE * np.linalg.inv(data.A.T @ data.A)))
    assert np.allclose(result.standard_errors(STATED_VARIANCE), direct)


# --- The real data (values checked against the earlier analysis, log #3) ------------------

def test_real_estimates(data, result):
    assert result.dof == 940
    assert np.all(result.volumes > 0)
    assert result.volumes[data.names.index("A6")] == pytest.approx(0.376, abs=1e-3)
    assert result.volumes[data.names.index("A14")] == pytest.approx(29.670, abs=1e-3)


def test_real_variance_test(report):
    v = report["variance"]
    assert v["estimated_variance"] == pytest.approx(3.861, abs=1e-3)
    assert v["ci"][0] == pytest.approx(3.53, abs=0.01)
    assert v["ci"][1] == pytest.approx(4.23, abs=0.01)
    assert v["z"] == pytest.approx(20.2, abs=0.1)
    assert v["chi2_sd"] == pytest.approx(43.36, abs=0.01)
    assert v["p_value"] == pytest.approx(8.87e-58, rel=0.01)


def test_real_model_checks(report):
    assert report["fixed_volume"]["fixed_volume"] == pytest.approx(-0.060, abs=1e-3)
    assert report["fixed_volume"]["p_value"] > 0.5
    assert report["fixed_volume"]["max_volume_change"] < 0.02
    assert report["scatter_vs_size"]["p_value"] == pytest.approx(0.624, abs=0.001)
    assert report["normality"]["shapiro_p"] == pytest.approx(0.876, abs=0.001)
    assert report["normality"]["skewness"] == pytest.approx(0.052, abs=0.001)
    assert report["normality"]["excess_kurtosis"] == pytest.approx(-0.031, abs=0.001)
    assert report["outliers"]["beyond"] == 3
    assert report["outliers"]["expected_beyond"] == pytest.approx(2.70, abs=0.01)
    assert report["outliers"]["worst_package"] == 44
    assert report["outliers"]["bonferroni_p"] > 0.05
    assert report["outliers_under_stated_variance"]["beyond"] == 32


def test_write_outputs(tmp_path, data, result, report):
    write_outputs(data, result, report, tmp_path)
    with open(tmp_path / "volumes.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    assert list(rows[0]) == ["item", "price", "volume_l", "se_l", "se_l_stated_variance", "unit_price_per_l"]
    a6 = rows[data.names.index("A6")]
    assert float(a6["unit_price_per_l"]) == pytest.approx(109 / float(a6["volume_l"]), rel=1e-5)
    # The main standard error uses the estimated variance (about 3.86), so it is the larger one.
    ratio = (result.variance / STATED_VARIANCE) ** 0.5
    for row in rows:
        assert float(row["se_l"]) == pytest.approx(ratio * float(row["se_l_stated_variance"]), rel=1e-4)
    saved = json.loads((tmp_path / "estimation.json").read_text(encoding="utf-8"))
    assert saved["variance"]["dof"] == 940
    for name in ["residuals", "prices", "volumes", "unit_prices"]:
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
