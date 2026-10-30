"""Event-related potential statistics and decoding shared by all datasets."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mne
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterResult:
    t_obs: np.ndarray  # (n_times, n_channels)
    significant: np.ndarray  # bool mask, same shape: inside a cluster with p < alpha
    cluster_p: np.ndarray  # p-value of every cluster found

    @property
    def any_significant(self) -> bool:
        return bool(self.significant.any())


def cluster_test_condition_difference(
    differences: np.ndarray,
    info: mne.Info,
    *,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ClusterResult:
    """Spatio-temporal cluster permutation test of condition differences against zero.

    Parameters
    ----------
    differences
        ``(n_observations, n_times, n_channels)``: one difference wave per subject/session.
    """
    n_obs = differences.shape[0]
    threshold = stats.t.ppf(1 - 0.025, df=n_obs - 1)  # cluster-forming, two-sided p < .05
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    t_obs, clusters, cluster_p, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        differences,
        threshold=threshold,
        n_permutations=n_permutations,
        adjacency=adjacency,
        seed=seed,
        out_type="mask",
        verbose="error",
    )
    significant = np.zeros_like(t_obs, dtype=bool)
    for cluster_mask, p_value in zip(clusters, cluster_p, strict=True):
        if p_value < alpha:
            significant |= cluster_mask
    return ClusterResult(t_obs, significant, np.asarray(cluster_p))


def mean_amplitudes_in_time_bins(
    data: np.ndarray, times: np.ndarray, tmin: float, tmax: float, bin_s: float = 0.1
) -> np.ndarray:
    """Mean amplitude per channel in consecutive time bins -> ``(n_epochs, n_channels * n_bins)``.

    A standard, low-dimensional ERP feature set for linear classifiers.
    """
    edges = np.arange(tmin, tmax + 1e-9, bin_s)
    bins = [
        data[:, :, (times >= start) & (times < stop)].mean(axis=-1)
        for start, stop in itertools.pairwise(edges)
    ]
    return np.concatenate(bins, axis=1)


def make_erp_classifier(C: float = 0.1):
    """Standardise -> L2 logistic regression; small C because features >> trials."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    )


def decode_each_time_point(
    data: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    groups: np.ndarray | None = None,
    *,
    C: float = 1.0,
) -> np.ndarray:
    """Train/test one classifier per time point on the channel pattern.

    Returns ROC AUC of shape ``(n_folds, n_times)``.
    """
    sliding = SlidingEstimator(make_erp_classifier(C), scoring="roc_auc", n_jobs=1, verbose=False)
    return cross_val_multiscore(sliding, data, y, cv=cv, groups=groups, n_jobs=1, verbose=False)


def find_above_chance_times(
    scores: np.ndarray, chance: float = 0.5, *, n_permutations: int = 1000, seed: int = 0
) -> np.ndarray:
    """Time points where fold/subject scores exceed chance (1-D cluster permutation, p < .05)."""
    n_folds = scores.shape[0]
    threshold = stats.t.ppf(0.95, df=n_folds - 1)  # one-sided: only above-chance decoding counts
    _, clusters, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
        scores - chance,
        threshold=threshold,
        tail=1,
        n_permutations=n_permutations,
        seed=seed,
        out_type="indices",
        verbose="error",
    )
    significant = np.zeros(scores.shape[1], dtype=bool)
    for (indices,), p_value in zip(clusters, cluster_p, strict=True):
        if p_value < 0.05:
            significant[indices] = True
    return significant
