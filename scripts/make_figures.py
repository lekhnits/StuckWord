"""Draw the publication figures from ``results/cache`` (run ``run_analyses.py`` first).

Fig. 1  Missing-word listening task
Fig. 2  ERP CORE N400
Fig. 3  Kiloword

Each figure is saved as a 600-dpi PNG in ``results/figures``. The figure legends are in
``results/README.md``.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from stuckword.plotting import (
    BLACK,
    BLUE,
    DOUBLE_COLUMN,
    GREEN,
    GREY,
    LIGHT_GREY,
    MM,
    VERMILION,
    colorbar,
    mean_sem,
    nature_style,
    panel_label,
    reference_lines,
    save_figure,
    significance_bar,
    topomap,
)

ROOT = Path(__file__).resolve().parents[1]
SHOWN_EPOCH_WINDOW = (-0.2, 1.2)  # seconds shown in the time-course panels
MINUS = "\u2212"
EN_DASH = "\u2013"


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return r"$\mathit{P}$ < 0.001"
    return rf"$\mathit{{P}}$ = {p_value:.3f}"


def format_time_axis(
    ax: plt.Axes, x_limits: tuple[float, float] = SHOWN_EPOCH_WINDOW, tick_step: float = 0.4
) -> None:
    ax.set_xlim(*x_limits)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_xlabel("Time (s)")


def plot_scores_with_mean(ax: plt.Axes, scores: np.ndarray, color: str) -> None:
    """One dot per recording/participant (jittered horizontally) plus mean ± s.e.m."""
    horizontal_jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(scores))
    ax.scatter(
        horizontal_jitter, scores, s=7, color=color, edgecolor="white", linewidth=0.3, zorder=3
    )
    standard_error = scores.std(ddof=1) / np.sqrt(len(scores))
    ax.errorbar(
        0.42,
        scores.mean(),
        yerr=standard_error,
        fmt="o",
        color=BLACK,
        ms=3,
        elinewidth=0.6,
        capsize=0,
    )
    ax.axhline(0.5, color=GREY, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(-0.45, 0.65)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


# ---------------------------------------------------------------------------------------------
def plot_missing_words_figure(results: dict, figure_dir: Path) -> None:
    times, info = results["times"], results["info"]
    channel_names = info.ch_names
    is_shown = (times >= SHOWN_EPOCH_WINDOW[0]) & (times <= SHOWN_EPOCH_WINDOW[1])
    shown_times = times[is_shown]

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 118 * MM))
    rows = fig.add_gridspec(
        3,
        1,
        height_ratios=[1, 0.66, 1],
        hspace=0.38,
        left=0.065,
        right=0.985,
        top=0.965,
        bottom=0.08,
    )

    # a: ERPs at three midline electrodes
    top_row = rows[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.32)
    erp_axes = [fig.add_subplot(top_row[0, column]) for column in range(3)]
    for column, (ax, electrode) in enumerate(zip(erp_axes, ("Fz", "Cz", "Pz"), strict=True)):
        electrode_index = channel_names.index(electrode)
        reference_lines(ax)
        mean_sem(ax, shown_times, results["speech_erp"][:, electrode_index, is_shown], GREY)
        mean_sem(ax, shown_times, results["missing_erp"][:, electrode_index, is_shown], VERMILION)
        significance_bar(
            ax, shown_times, results["significant_mask"][is_shown, electrode_index], y=0.03
        )
        ax.text(0.04, 0.97, electrode, transform=ax.transAxes, va="top", fontweight="bold")
        format_time_axis(ax)
        ax.set_ylim(-2.2, 3.2)
        if column > 0:
            ax.tick_params(labelleft=False)
    erp_axes[0].set_ylabel("Amplitude (a.u.)")
    erp_axes[1].text(0.62, 0.86, "Missing word", color=VERMILION, transform=erp_axes[1].transAxes)
    erp_axes[1].text(0.62, 0.76, "Speech", color=GREY, transform=erp_axes[1].transAxes)
    panel_label(erp_axes[0], "a", x=-0.38)

    # b: timing check - response strength for shifted event times, as z relative to random times
    ax = fig.add_subplot(top_row[0, 3])
    null_mean, null_sd = (
        results["random_times_strength"].mean(),
        results["random_times_strength"].std(),
    )
    onset_z = (results["onset_locked_strength"] - null_mean) / null_sd
    offset_z = (results["offset_locked_strength"] - null_mean) / null_sd
    null_95th_z = (np.percentile(results["random_times_strength"], 95) - null_mean) / null_sd
    ax.set_ylim(min(onset_z.min(), offset_z.min()) - 0.5, onset_z.max() + 0.8)
    ax.axhspan(ax.get_ylim()[0], null_95th_z, color=LIGHT_GREY, lw=0, zorder=0)
    reference_lines(ax, y0=None)
    ax.plot(results["shifts"], offset_z, "-o", color=BLUE, ms=2.2, lw=0.8)
    ax.plot(results["shifts"], onset_z, "-o", color=VERMILION, ms=2.2, lw=0.8)
    ax.text(0.03, 0.96, "Locked to gap onset", color=VERMILION, transform=ax.transAxes, va="top")
    ax.text(0.03, 0.86, "Locked to gap end", color=BLUE, transform=ax.transAxes, va="top")
    ax.text(
        0.97,
        0.05,
        "95% of random\nevent times",
        color=GREY,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5,
    )
    ax.set_xlabel("Shift added to timecodes (s)")
    ax.set_ylabel("Evoked response (z)")
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    panel_label(ax, "b", x=-0.3)

    # c: scalp maps of the missing - speech difference
    map_windows = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2)]
    mean_difference = (results["missing_erp"] - results["speech_erp"]).mean(0)
    color_limit = 2.0
    map_row = rows[1].subgridspec(1, 8, width_ratios=[1] * 6 + [0.12, 0.05], wspace=0.08)
    for column, (start, stop) in enumerate(map_windows):
        ax = fig.add_subplot(map_row[0, column])
        in_window = (times >= start) & (times < stop)
        image = topomap(
            ax,
            mean_difference[:, in_window].mean(1),
            info,
            (-color_limit, color_limit),
            mask=results["significant_mask"][in_window].any(0),
        )
        ax.set_title(f"{start:.1f}{EN_DASH}{stop:.1f} s", pad=1)
        if column == 0:
            panel_label(ax, "c", x=-0.12, y=1.2)
    colorbar(
        fig,
        image,
        fig.add_subplot(map_row[0, 7]),
        f"Missing {MINUS} speech (a.u.)",
        [-color_limit, 0, color_limit],
    )

    # d-f: decoding, always tested on a recording left out of training
    bottom_row = rows[2].subgridspec(1, 3, width_ratios=[2.2, 0.8, 1.1], wspace=0.5)
    ax = fig.add_subplot(bottom_row[0, 0])
    reference_lines(ax, y0=0.5)
    mean_sem(ax, shown_times, results["auc_over_time"][:, is_shown], VERMILION)
    significance_bar(ax, shown_times, results["auc_above_chance"][is_shown], y=0.04)
    format_time_axis(ax)
    ax.set_ylabel("Decoding (AUC)")
    panel_label(ax, "d", x=-0.13)

    ax = fig.add_subplot(bottom_row[0, 1])
    plot_scores_with_mean(ax, results["auc_per_recording"], VERMILION)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel(f"AUC, 0{EN_DASH}1.2 s")
    ax.set_xlabel(f"Recordings (n = {len(results['auc_per_recording'])})")
    panel_label(ax, "e", x=-0.55)

    ax = fig.add_subplot(bottom_row[0, 2])
    ax.hist(
        results["auc_shuffled_labels"], bins=30, color=LIGHT_GREY, edgecolor=GREY, linewidth=0.3
    )
    observed_auc = results["auc_per_recording"].mean()
    ax.axvline(observed_auc, color=VERMILION, lw=1)
    ax.text(
        observed_auc,
        ax.get_ylim()[1] * 0.98,
        f" Observed\n {format_p_value(results['permutation_p'])}",
        color=VERMILION,
        va="top",
        ha="left",
        fontsize=5.5,
    )
    ax.set_xlabel("AUC, shuffled labels")
    ax.set_ylabel("Permutations")
    panel_label(ax, "f", x=-0.38)

    save_figure(fig, figure_dir / "Fig1_missing_words")


# ---------------------------------------------------------------------------------------------
def plot_n400_figure(results: dict, figure_dir: Path) -> None:
    times, info = results["times"], results["info"]
    cpz_index = info.ch_names.index("CPz")
    n400_x_limits = (-0.2, 0.8)

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 50 * MM))
    # Narrow empty column (index 3) keeps the colorbar label clear of the next panel.
    columns = fig.add_gridspec(
        1,
        6,
        width_ratios=[1.35, 0.85, 0.05, 0.28, 1.35, 0.6],
        wspace=0.4,
        left=0.06,
        right=0.985,
        top=0.88,
        bottom=0.2,
    )

    ax = fig.add_subplot(columns[0, 0])
    ax.axvspan(0.3, 0.5, color=LIGHT_GREY, lw=0, zorder=0)
    reference_lines(ax)
    mean_sem(ax, times, results["related_erp"][:, cpz_index], GREY)
    mean_sem(ax, times, results["unrelated_erp"][:, cpz_index], VERMILION)
    format_time_axis(ax, n400_x_limits, tick_step=0.2)
    ax.set_ylabel("Amplitude at CPz (µV)")
    ax.text(0.03, 0.97, "Related", color=GREY, transform=ax.transAxes, va="top")
    ax.text(0.03, 0.87, "Unrelated", color=VERMILION, transform=ax.transAxes, va="top")
    summary = results["summary"]
    effect_size = f"{summary['n400_effect_uv_cpz_300_500']:.1f}".replace("-", MINUS)
    effect_text = f"{effect_size} µV, {format_p_value(summary['n400_effect_p'])}"
    ax.text(0.4, 0.06, effect_text, transform=ax.transAxes, fontsize=5.5)
    panel_label(ax, "a", x=-0.22)

    ax = fig.add_subplot(columns[0, 1])
    in_n400_window = (times >= 0.3) & (times < 0.5)
    mean_difference = (results["unrelated_erp"] - results["related_erp"]).mean(0)
    image = topomap(ax, mean_difference[:, in_n400_window].mean(1), info, (-6, 6))
    ax.set_title(f"0.3{EN_DASH}0.5 s", pad=1)
    colorbar(
        fig, image, fig.add_subplot(columns[0, 2]), f"Unrelated {MINUS} related (µV)", [-6, 0, 6]
    )
    panel_label(ax, "b", x=-0.1, y=1.18)

    ax = fig.add_subplot(columns[0, 4])
    reference_lines(ax, y0=0.5)
    mean_sem(ax, times, results["within_person_auc"], VERMILION)
    significance_bar(ax, times, results["within_person_above_chance"], y=0.04)
    format_time_axis(ax, n400_x_limits, tick_step=0.2)
    ax.set_ylabel("Decoding within person (AUC)")
    panel_label(ax, "c", x=-0.22)

    ax = fig.add_subplot(columns[0, 5])
    plot_scores_with_mean(ax, results["cross_person_auc"], VERMILION)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("Decoding new person (AUC)")
    ax.set_xlabel(f"Participants (n = {len(results['cross_person_auc'])})")
    panel_label(ax, "d", x=-0.7)

    save_figure(fig, figure_dir / "Fig2_n400")


# ---------------------------------------------------------------------------------------------
def plot_kiloword_figure(results: dict, figure_dir: Path) -> None:
    times, info = results["times"], results["info"]
    channel_names = info.ch_names
    pz_index = channel_names.index("Pz") if "Pz" in channel_names else len(channel_names) // 2
    property_colors = {"WordFrequency": VERMILION, "Concreteness": BLUE, "NumberOfLetters": GREEN}
    property_labels = {
        "WordFrequency": "Frequency",
        "Concreteness": "Concreteness",
        "NumberOfLetters": "Length",
    }

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 52 * MM))
    # Narrow empty column (index 3) keeps the colorbar label clear of the scatter plot.
    columns = fig.add_gridspec(
        1,
        6,
        width_ratios=[1.45, 0.62, 0.05, 0.22, 1, 1],
        wspace=0.45,
        left=0.055,
        right=0.985,
        top=0.86,
        bottom=0.2,
    )

    ax = fig.add_subplot(columns[0, 0])
    ax.axhspan(
        -results["bonferroni_t_threshold"],
        results["bonferroni_t_threshold"],
        color=LIGHT_GREY,
        lw=0,
        zorder=0,
    )
    reference_lines(ax)
    for row, (word_property, color) in enumerate(property_colors.items()):
        ax.plot(times, results["t_values"][word_property][pz_index], color=color)
        ax.text(
            0.02,
            0.97 - 0.1 * row,
            property_labels[word_property],
            color=color,
            transform=ax.transAxes,
            va="top",
        )
    format_time_axis(ax, (-0.1, 0.9), tick_step=0.2)
    ax.set_ylabel(r"Effect at Pz ($\mathit{t}$)")
    panel_label(ax, "a", x=-0.18)

    map_cells = columns[0, 1].subgridspec(2, 1, hspace=0.35)
    effect_windows = {"WordFrequency": (0.55, 0.70), "Concreteness": (0.30, 0.45)}
    for row, (word_property, (start, stop)) in enumerate(effect_windows.items()):
        ax = fig.add_subplot(map_cells[row, 0])
        in_window = (times >= start) & (times < stop)
        mean_t = results["t_values"][word_property][:, in_window].mean(1)
        image = topomap(ax, mean_t, info, (-6, 6))
        ax.set_title(
            f"{property_labels[word_property]}, {start:.2f}{EN_DASH}{stop:.2f} s",
            pad=1,
            fontsize=5.5,
        )
        if row == 0:
            panel_label(ax, "b", x=-0.25, y=1.35)
    colorbar(fig, image, fig.add_subplot(columns[0, 2]), r"$\mathit{t}$", [-6, 0, 6])

    for column, word_property in enumerate(("WordFrequency", "Concreteness")):
        ax = fig.add_subplot(columns[0, 4 + column])
        prediction = results["predictions"][word_property]
        actual, predicted = prediction["actual"], prediction["predicted"]
        ax.scatter(actual, predicted, s=1.5, color=property_colors[word_property], alpha=0.35, lw=0)
        slope, intercept = np.polyfit(actual, predicted, 1)
        line_x = np.array([actual.min(), actual.max()])
        ax.plot(line_x, slope * line_x + intercept, color=BLACK, lw=0.7)
        ax.text(
            0.03,
            0.97,
            rf"$\mathit{{r}}$ = {prediction['r']:.2f}" + "\n" + format_p_value(prediction["p"]),
            transform=ax.transAxes,
            va="top",
            fontsize=5.5,
        )
        label = property_labels[word_property]
        ax.set_xlabel(f"{label} (actual)")
        ax.set_ylabel(f"{label} (predicted from ERP)")
        panel_label(ax, "cd"[column], x=-0.3)

    save_figure(fig, figure_dir / "Fig3_kiloword")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    nature_style()
    cache_dir, figure_dir = args.results / "cache", args.results / "figures"
    figure_builders = {
        "missingwords": plot_missing_words_figure,
        "n400": plot_n400_figure,
        "kiloword": plot_kiloword_figure,
    }
    for analysis_name, plot_figure in figure_builders.items():
        cache_file = cache_dir / f"{analysis_name}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:  # written by run_analyses.py in this repo
                plot_figure(pickle.load(fh), figure_dir)
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
