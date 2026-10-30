"""Which few channels carry the difference? Used to decide where the headband electrodes go."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stuckword.config import FREQ_BANDS
from stuckword.features import compute_band_power
from stuckword.model import build_classifier, evaluate_classifier


def rank_channels(
    data: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    band: str = "theta",
) -> pd.DataFrame:
    """Univariate per-channel separability of relative log power in one band.

    AUC > 0.5 means higher band power on positive (e.g. tip-of-the-tongue) trials.
    """
    # Relative power needs all bands for the denominator; then keep the one of interest.
    band_power = compute_band_power(data, sfreq, FREQ_BANDS)[:, :, list(FREQ_BANDS).index(band)]
    positive, negative = band_power[y == 1], band_power[y == 0]
    pooled_sd = np.sqrt((positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)) / 2)
    table = pd.DataFrame(
        {
            "channel": list(ch_names),
            "band": band,
            "auc": [
                roc_auc_score(y, band_power[:, channel]) for channel in range(band_power.shape[1])
            ],
            "cohens_d": (positive.mean(axis=0) - negative.mean(axis=0)) / pooled_sd,
        }
    )
    order = (table["auc"] - 0.5).abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def forward_select_channels(
    data: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    groups: np.ndarray | None = None,
    *,
    max_channels: int = 4,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Greedy forward selection of channels by cross-validated score.

    Note: the reported scores are optimistic, because channels are chosen using the same data
    they are scored on. Treat the result as a *placement hypothesis* and confirm it on a session
    recorded after the selection.
    """
    selected: list[int] = []
    remaining = list(range(len(ch_names)))
    rows = []
    for step in range(1, max_channels + 1):
        scores = {
            ch: evaluate_classifier(
                build_classifier(sfreq, picks=[*selected, ch]), data, y, groups, metric=metric
            ).mean
            for ch in remaining
        }
        best = max(scores, key=scores.__getitem__)
        selected.append(best)
        remaining.remove(best)
        rows.append(
            {
                "step": step,
                "added": ch_names[best],
                "channels": ", ".join(ch_names[i] for i in selected),
                metric: scores[best],
            }
        )
    return pd.DataFrame(rows)
