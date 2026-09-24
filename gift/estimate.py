# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Estimate the item volumes from the package measurements (step 2).

Model: b = A v + e
    b  measured package volumes (litres)
    A  0/1 package-by-item table from gift.data
    v  unknown item volumes
    e  independent Gaussian measurement noise with variance sigma^2
There is no fixed volume per package: the task defines a package's volume as the
sum of its items' volumes (#25). Plain least squares (#24) gives v_hat, which is
the maximum-likelihood estimate under this noise model.

Uncertainty: Cov(v_hat) = sigma^2 (A^T A)^-1. The main sigma^2 is the one
estimated from the residuals, RSS / (n - p). The stated value 2 is kept only for
comparison, because the data reject it (#5, #19).

Decisions applied here (DECISIONS.md records who made each one):
    #24  Plain least squares; stop if any estimated volume is 0 or below.
         (AI proposal, approved by the author.)
    #25  The fixed volume per package is estimated, tested and reported, but not
         modelled. (AI proposal, approved by the author.)
    #26  Four model checks, one line each. (AI proposal, approved by the author.)
    #27  Outputs: results/volumes.csv and figures. (CSV and residual figure: AI
         proposal, approved by the author. Price, volume and unit-price
         histograms: the author's.)
    #29  estimation.json, the outlier count under the stated variance, the
         leverage-scaled residuals in the figure, the unit-price column and the
         unit-price bar labels. (AI additions, approved by the author.)

Run `python -m gift.estimate` to print the report and write the files in results/.
"""

import csv
import json
from dataclasses import dataclass

import numpy as np
from scipy import stats

from gift.data import ROOT, load

STATED_VARIANCE = 2.0
RESULTS_DIR = ROOT / "results"


class EstimationError(ValueError):
    """The estimates break an assumption of the method."""


@dataclass(frozen=True)
class Fit:
    volumes: np.ndarray    # v_hat, one per item
    residuals: np.ndarray  # b - A v_hat, one per package
    leverage: np.ndarray   # diagonal of A (A^T A)^-1 A^T, one per package
    xtx_inv: np.ndarray    # (A^T A)^-1
    rss: float             # residual sum of squares
    dof: int               # degrees of freedom, n - p

    @property
    def variance(self):
        """Estimated noise variance, RSS / (n - p)."""
        return self.rss / self.dof

    def covariance(self, variance=None):
        """Cov(v_hat) under the given noise variance (default: the estimated one)."""
        return (self.variance if variance is None else variance) * self.xtx_inv

    def standard_errors(self, variance=None):
        return np.sqrt(np.diag(self.covariance(variance)))

    def scaled_residuals(self):
        """Residuals divided by sqrt(1 - leverage), in litres.

        A raw residual has variance sigma^2 (1 - leverage), a little less than the
        noise itself. After this scaling each has variance sigma^2, so they can be
        compared directly with the noise distribution.
        """
        return self.residuals / np.sqrt(1 - self.leverage)

    def standardized_residuals(self, variance=None):
        """Scaled residuals in units of the noise standard deviation."""
        return self.scaled_residuals() / np.sqrt(self.variance if variance is None else variance)


def fit(A, b):
    """Plain least-squares fit of b = A v + e."""
    volumes, *_ = np.linalg.lstsq(A, b, rcond=None)
    residuals = b - A @ volumes
    xtx_inv = np.linalg.inv(A.T @ A)
    leverage = np.einsum("ij,jk,ik->i", A, xtx_inv, A)
    n, p = A.shape
    return Fit(volumes, residuals, leverage, xtx_inv, float(residuals @ residuals), n - p)


def estimate(data):
    """Fit the real data and check that every estimated volume is positive (#24)."""
    result = fit(data.A, data.b)
    bad = [f"{name} ({v:.3f} L)" for name, v in zip(data.names, result.volumes) if v <= 0]
    if bad:
        raise EstimationError(
            "estimated volume is 0 or below for " + ", ".join(bad)
            + "; plain least squares no longer suits the problem (#24)"
        )
    return result


def variance_test(result, stated=STATED_VARIANCE, level=0.95):
    """Test the stated noise variance against the residuals.

    If the stated variance were right, RSS / stated would follow a chi-square
    distribution with n - p degrees of freedom. The p-value is two-sided. The
    confidence interval for the variance comes from the same distribution.
    """
    dof = result.dof
    statistic = result.rss / stated
    p_value = 2 * min(stats.chi2.cdf(statistic, dof), stats.chi2.sf(statistic, dof))
    tail = (1 - level) / 2
    return {
        "stated_variance": stated,
        "estimated_variance": result.variance,
        "estimated_sd": result.variance ** 0.5,
        "rss": result.rss,
        "dof": dof,
        "statistic": statistic,
        "chi2_mean": dof,
        "chi2_sd": (2 * dof) ** 0.5,
        "z": (statistic - dof) / (2 * dof) ** 0.5,
        "p_value": min(1.0, float(p_value)),
        "ci_level": level,
        "ci": [result.rss / stats.chi2.ppf(1 - tail, dof), result.rss / stats.chi2.ppf(tail, dof)],
    }


def intercept_test(A, b, volumes):
    """Check 1: is there a fixed volume per package (#25)?

    Fits b = c + A v + e and runs a t-test of c = 0. `volumes` are the estimates
    of the main model, used to report how much adding c would move them.
    """
    n, p = A.shape
    X = np.column_stack([np.ones(n), A])
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    residuals = b - X @ coef
    dof = n - p - 1
    se = np.sqrt(float(residuals @ residuals) / dof * np.linalg.inv(X.T @ X)[0, 0])
    t = coef[0] / se
    return {
        "fixed_volume": float(coef[0]),
        "se": float(se),
        "t": float(t),
        "p_value": float(2 * stats.t.sf(abs(t), dof)),
        "max_volume_change": float(np.max(np.abs(coef[1:] - volumes))),
    }


def size_test(result, A):
    """Check 2: does the scatter grow with the number of items in a package?

    If item volumes varied from one unit to the next, packages with more items
    would scatter more. Koenker's version of the Breusch-Pagan test: regress the
    squared standardized residuals on package size; n R^2 ~ chi-square(1) if the
    scatter does not depend on size.
    """
    sizes = A.sum(axis=1)
    squared = result.standardized_residuals() ** 2
    reg = stats.linregress(sizes, squared)
    lm = len(sizes) * reg.rvalue ** 2
    variance_by_size = {
        int(k): float(result.variance * np.mean(squared[sizes == k])) for k in np.unique(sizes)
    }
    return {
        "slope": float(reg.slope),
        "p_value": float(stats.chi2.sf(lm, 1)),
        "variance_by_size": variance_by_size,
    }


def normality_test(result):
    """Check 3: are the residuals normally distributed (Shapiro-Wilk)?"""
    z = result.standardized_residuals()
    return {
        "shapiro_p": float(stats.shapiro(z).pvalue),
        "skewness": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z)),
    }


def outlier_check(result, variance=None, threshold=3.0):
    """Check 4: are there outlying packages?

    Counts standardized residuals beyond `threshold` and tests the largest one
    with a Bonferroni correction over all packages. Package numbers are 1-based.
    """
    z = result.standardized_residuals(variance)
    n = len(z)
    worst = int(np.argmax(np.abs(z)))
    return {
        "variance": result.variance if variance is None else variance,
        "threshold": threshold,
        "beyond": int(np.sum(np.abs(z) > threshold)),
        "expected_beyond": float(n * 2 * stats.norm.sf(threshold)),
        "max_abs": float(abs(z[worst])),
        "worst_package": worst + 1,
        "bonferroni_p": float(min(1.0, n * 2 * stats.norm.sf(abs(z[worst])))),
    }


def checks(data, result):
    """The variance test and the four model checks (#19, #26)."""
    return {
        "variance": variance_test(result),
        "fixed_volume": intercept_test(data.A, data.b, result.volumes),
        "scatter_vs_size": size_test(result, data.A),
        "normality": normality_test(result),
        "outliers": outlier_check(result),
        "outliers_under_stated_variance": outlier_check(result, STATED_VARIANCE),
    }


def volume_table(data, result):
    """One row per item: price, volume, standard errors and unit price."""
    se = result.standard_errors()
    se_stated = result.standard_errors(STATED_VARIANCE)
    return [
        {
            "item": name,
            "price": int(price),
            "volume_l": float(v),
            "se_l": float(s),
            "se_l_stated_variance": float(s2),
            "unit_price_per_l": float(price / v),
        }
        for name, price, v, s, s2 in zip(data.names, data.prices, result.volumes, se, se_stated)
    ]


def write_outputs(data, result, report, out_dir=RESULTS_DIR):
    """Write volumes.csv, estimation.json and the figures (#27) into out_dir."""
    from gift import figures

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = volume_table(data, result)
    with open(out_dir / "volumes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    with open(out_dir / "estimation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures.residual_histogram(result.scaled_residuals(), result.variance, STATED_VARIANCE,
                               figure_dir / "residuals.png")
    figures.histogram([r["price"] for r in rows], "Item prices", "Price", figure_dir / "prices.png")
    figures.histogram([r["volume_l"] for r in rows], "Estimated item volumes", "Volume (L)",
                      figure_dir / "volumes.png")
    extreme = {f"{r['item']} ({r['unit_price_per_l']:.0f}/L)": r["unit_price_per_l"]
               for r in rows if r["unit_price_per_l"] > 50}
    figures.histogram([r["unit_price_per_l"] for r in rows], "Unit price of items",
                      "Price per litre (log scale)", figure_dir / "unit_prices.png", log=True,
                      labels=extreme)


def print_report(data, result, report):
    v, fx, sz, nm = report["variance"], report["fixed_volume"], report["scatter_vs_size"], report["normality"]
    out, out2 = report["outliers"], report["outliers_under_stated_variance"]
    se, se2 = result.standard_errors(), result.standard_errors(STATED_VARIANCE)
    lo, hi = int(np.argmin(result.volumes)), int(np.argmax(result.volumes))
    print(f"Least squares: {len(result.volumes)} item volumes from {len(result.residuals)} packages "
          f"({result.dof} degrees of freedom)")
    print(f"Volumes: min {result.volumes[lo]:.3f} L ({data.names[lo]}), "
          f"max {result.volumes[hi]:.3f} L ({data.names[hi]}), all positive")
    print(f"Noise variance: estimated {v['estimated_variance']:.3f} (sd {v['estimated_sd']:.3f} L), "
          f"95% CI {v['ci'][0]:.2f} to {v['ci'][1]:.2f}")
    print(f"  Stated variance {v['stated_variance']:g}: RSS/{v['stated_variance']:g} = {v['statistic']:.1f}, "
          f"chi-square({v['dof']}) mean {v['chi2_mean']}, sd {v['chi2_sd']:.1f}; "
          f"z = {v['z']:.1f}, p = {v['p_value']:.1e}")
    print(f"Standard errors: {se.min():.3f} to {se.max():.3f} L (estimated variance), "
          f"{se2.min():.3f} to {se2.max():.3f} L (stated variance)")
    print(f"Check 1, fixed volume per package (not in the model): {fx['fixed_volume']:+.3f} L "
          f"(SE {fx['se']:.3f}), p = {fx['p_value']:.2f}; item volumes would move by at most "
          f"{fx['max_volume_change']:.3f} L")
    print(f"Check 2, scatter vs number of items: slope {sz['slope']:+.3f} per item, p = {sz['p_value']:.2f}")
    print(f"Check 3, normality: Shapiro-Wilk p = {nm['shapiro_p']:.2f}, skewness {nm['skewness']:+.3f}, "
          f"excess kurtosis {nm['excess_kurtosis']:+.3f}")
    print(f"Check 4, outliers: {out['beyond']} beyond {out['threshold']:g} sd "
          f"(expected {out['expected_beyond']:.1f}); largest {out['max_abs']:.2f} sd "
          f"(package #{out['worst_package']}), Bonferroni p = {out['bonferroni_p']:.2f}")
    print(f"  Under the stated variance: {out2['beyond']} beyond {out2['threshold']:g} sd, "
          f"largest {out2['max_abs']:.2f} sd, Bonferroni p = {out2['bonferroni_p']:.1e}")


def main(out_dir=RESULTS_DIR):
    data = load()
    result = estimate(data)
    report = checks(data, result)
    print_report(data, result, report)
    write_outputs(data, result, report, out_dir)
    print(f"Wrote {out_dir / 'volumes.csv'}, {out_dir / 'estimation.json'} and figures in {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
