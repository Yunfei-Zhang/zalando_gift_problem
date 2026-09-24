# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Figures for the write-up, saved as PNG files.

Colours follow a colour-blind-safe categorical order: blue for the data, then
orange and aqua for reference curves. The reference curves also differ in line
style (solid and dashed), so no curve is identified by colour alone.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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
