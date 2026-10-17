"""Shared figure styling for every chart this project produces.

Colour choices follow a validated palette rather than taste:

  - Categorical slots are assigned in FIXED order, never cycled. Slots 1 and 2
    (blue, orange) clear every colour-vision-deficiency gate as a pair:
    CVD Delta E 24.7, normal-vision Delta E 33.6, both well above the 8 / 15 floors.
  - Magnitude (the confusion matrix) uses ONE hue, light to dark - never a
    rainbow, because a rainbow implies category where the data has order.
  - Text always wears ink colours, never a series colour; a coloured mark
    beside a label carries the identity instead.
  - Grid and axes are recessive so the data is the most prominent thing.

These figures are rendered for print in a report, so they target the light
surface only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a headless/CLI run
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# --- palette ------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8980"
GRID = "#e4e3df"

# Categorical slots, in fixed assignment order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Status colours are reserved and never reused as "another series".
GOOD = "#008300"
CRITICAL = "#e34948"

# Single-hue sequential ramp (blue, light -> dark) for magnitude encoding.
SEQUENTIAL_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                    "#184f95", "#104281", "#0d366b"]
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#ffffff"] + SEQUENTIAL_STEPS)


def apply_style() -> None:
    """Recessive chrome, thin marks, ink-coloured text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 14,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.labelcolor": TEXT_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "lines.solid_capstyle": "round",
        "font.size": 10.5,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def subtitle(ax, text: str, title_pad: float = 26) -> None:
    """A muted one-line note under the title, for context the title can't hold.

    Positioned in offset points rather than axes fractions: an axes-fraction
    offset shrinks in absolute terms as the figure grows, which collides with
    the title on tall figures.
    """
    # get_title() reads the CENTER title; a left-aligned title is a separate
    # Text object, so reading the wrong one would silently blank the title.
    current = ax.get_title(loc="left") or ax.get_title()
    ax.set_title(current, loc="left", pad=title_pad)
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 7), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, color=TEXT_MUTED)


def save(fig, name: str, close: bool = True) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    if close:
        plt.close(fig)
    print("  wrote %s" % path.relative_to(config.ROOT))
    return path
