# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Figures for the write-up, saved as PNG files.

Colours follow a colour-blind-safe categorical order: blue for the data, then
orange and aqua for reference curves. The reference curves also differ in line
style (solid and dashed), so no curve is identified by colour alone.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def _axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def residual_histogram(residuals, estimated_variance, stated_variance, path):
    """Histogram of the scaled residuals with the two candidate noise curves."""
    fig, ax = _axes(
        "Package residuals against the two noise levels",
        "Residual, scaled for leverage (L)",
        "Density",
    )
    ax.hist(residuals, bins=40, density=True, color=BLUE, edgecolor=SURFACE, linewidth=1,
            label="Residuals (1000 packages)")
    x = np.linspace(-8, 8, 400)
    for variance, colour, style, name in [
        (estimated_variance, ORANGE, "-", f"N(0, {estimated_variance:.2f}): estimated from the data"),
        (stated_variance, AQUA, "--", f"N(0, {stated_variance:g}): stated in the task"),
    ]:
        ax.plot(x, stats.norm.pdf(x, scale=np.sqrt(variance)), color=colour, linestyle=style,
                linewidth=2, label=name)
    ax.set_xlim(-8, 8)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left", fontsize=9)
    _save(fig, path)


def price_vs_chance(chances, prices, optimum, recommended, level, path):
    """Each near-optimal set as a point (chance of fitting, price), with the frontier.

    The frontier is the best price available at each required chance. `optimum` and
    `recommended` are (chance, price) pairs that get a label.
    """
    fig, ax = _axes("Price against the chance of fitting in 40 L",
                    "Chance of fitting (normal model, estimated noise variance)", "Total price")
    chances, prices = np.asarray(chances), np.asarray(prices)
    ax.scatter(chances, prices, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3,
               label=f"Sets within 40 of the best price ({len(prices)})")
    grid = np.linspace(0, 1, 1001)
    frontier = [prices[chances >= g].max() if (chances >= g).any() else np.nan for g in grid]
    ax.step(grid, frontier, where="post", color=ORANGE, linewidth=2, zorder=2,
            label="Best price at this chance or higher")
    ax.axvline(level, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
    ax.annotate(f"required {level:.0%}", (level, prices.max()), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", color=TEXT_SECONDARY, fontsize=9)
    for (x, y), text in [(optimum, "Exact optimum"), (recommended, "Recommended")]:
        ax.scatter([x], [y], s=90, facecolor="none", edgecolor=TEXT, linewidth=1.5, zorder=4)
    x, y = optimum
    ax.annotate(f"Exact optimum: {y} ({x:.0%})", (x, y), xytext=(8, 4), textcoords="offset points",
                ha="left", color=TEXT, fontsize=9)
    # The recommended set sits among other points and the frontier, so its label goes in
    # the empty area above the frontier's middle steps, with a thin leader line.
    x, y = recommended
    ax.annotate(f"Recommended: {y} ({x:.0%})", (x, y), xytext=(0.62, (y + prices.max()) / 2),
                textcoords="data", ha="left", color=TEXT, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8})
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower left", fontsize=9)
    _save(fig, path)


def item_share(shares, draws, path):
    """Horizontal bars: how often each item is in the best set across simulated volumes."""
    names, values = list(shares), np.array(list(shares.values()))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(names) + 1.2), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"How often each item is in the best set\n{draws:,} simulated volume sets (positive-only, "
                 "estimated noise variance)\nOnly items that are ever chosen are shown",
                 loc="left", color=TEXT, fontsize=10)
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=BLUE, edgecolor=SURFACE, linewidth=1, height=0.8)
    for yi, v in zip(y, values):
        # Show one decimal near 100%, so that e.g. 99.85% is not printed as 100%.
        text = "100%" if v == 1.0 else (f"{v:.1%}" if v >= 0.995 else f"{v:.0%}")
        ax.annotate(text, (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    color=TEXT, fontsize=8)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    _save(fig, path)


def histogram(values, title, xlabel, path, log=False, labels=None):
    """Plain histogram of one variable across the 60 items.

    `labels` maps a text label to a value; each is written above the bar that
    holds the value.
    """
    fig, ax = _axes(title, xlabel, "Number of items")
    values = np.asarray(values, dtype=float)
    if log:
        bins = np.geomspace(values.min(), values.max(), 16)
        ax.set_xscale("log")
    else:
        bins = 12
    counts, edges, _ = ax.hist(values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    stacked = {}
    for text, value in (labels or {}).items():
        k = min(int(np.searchsorted(edges, value, side="right")) - 1, len(counts) - 1)
        height = counts[k] + 0.3 + 0.7 * stacked.get(k, 0)
        stacked[k] = stacked.get(k, 0) + 1
        ax.annotate(text, ((edges[k] * edges[k + 1]) ** 0.5 if log else (edges[k] + edges[k + 1]) / 2,
                           height), ha="center", color=TEXT, fontsize=9)
    if labels:
        ax.set_ylim(top=max(ax.get_ylim()[1], counts.max() + 1))
    _save(fig, path)
