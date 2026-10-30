"""Publication figure style (Nature-like) and small drawing helpers.

Conventions: figures are sized in millimetres to journal column widths, text is Arial 5-7 pt,
panels carry bold lowercase letters, axes are thin with outward ticks and no grid, and
explanations live in the figure legend (see ``results/README.md``), not in plot titles.
Colours are the Okabe-Ito set, which stays distinguishable with colour-vision deficiency.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.transforms import blended_transform_factory
from scipy import stats

MM = 1 / 25.4  # inches per millimetre
SINGLE_COLUMN = 89 * MM
DOUBLE_COLUMN = 183 * MM

BLACK = "#000000"
GREY = "#808080"
LIGHT_GREY = "#E6E6E6"
VERMILION = "#D55E00"  # Okabe-Ito
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
DIVERGING = "RdBu_r"


def nature_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "axes.titleweight": "normal",
            "axes.titlepad": 3,
            "axes.labelpad": 2,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.5,
            "axes.edgecolor": BLACK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 3,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.borderaxespad": 0.2,
            "mathtext.default": "regular",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, letter: str, x: float = -0.28, y: float = 1.12) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left"
    )


def mean_sem(
    ax: plt.Axes, x: np.ndarray, data: np.ndarray, color: str, label: str | None = None
) -> None:
    """Mean line with a ±1 s.e.m. band over the first axis of ``data``."""
    mean, sem = data.mean(axis=0), stats.sem(data, axis=0)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.22, lw=0)
    ax.plot(x, mean, color=color, label=label)


def significance_bar(
    ax: plt.Axes, x: np.ndarray, mask: np.ndarray, y: float = 0.04, color: str = BLACK
) -> None:
    """Thick horizontal bar under contiguous significant runs; ``y`` in axes fraction."""
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        ax.plot(
            [x[start], x[stop - 1]],
            [y, y],
            color=color,
            lw=1.6,
            solid_capstyle="butt",
            transform=transform,
            clip_on=False,
        )


def reference_lines(ax: plt.Axes, x0: float | None = 0.0, y0: float | None = 0.0) -> None:
    if y0 is not None:
        ax.axhline(y0, color=GREY, lw=0.4, zorder=0)
    if x0 is not None:
        ax.axvline(x0, color=GREY, lw=0.4, ls=(0, (2, 2)), zorder=0)


def topomap(
    ax: plt.Axes,
    values: np.ndarray,
    info: mne.Info,
    vlim: tuple[float, float],
    mask: np.ndarray | None = None,
    cmap: str = DIVERGING,
):
    """Small scalp map: thin outline, faint contours, tiny sensors, white-dot cluster mask."""
    im, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        cmap=cmap,
        vlim=vlim,
        contours=4,
        sensors=True,
        mask=mask,
        mask_params={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": BLACK,
            "markeredgewidth": 0.4,
            "markersize": 2.6,
        },
        show=False,
    )
    for line in ax.lines:
        if line.get_marker() in ("None", "", None):
            line.set_linewidth(0.5)
            line.set_color(BLACK)
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            collection.set_sizes([0.6])
        elif hasattr(collection, "set_linewidth"):
            collection.set_linewidth(0.25)
    return im


def colorbar(fig: plt.Figure, image, cax: plt.Axes, label: str, ticks: Sequence[float]) -> None:
    bar = fig.colorbar(image, cax=cax, ticks=ticks)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(width=0.4, length=1.5, labelsize=5)
    bar.set_label(label, fontsize=5.5, labelpad=2)


def save_figure(fig: plt.Figure, stem: Path) -> Path:
    """Write ``stem.png`` at 600 dpi."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".png")
    fig.savefig(path)
    plt.close(fig)
    return path
