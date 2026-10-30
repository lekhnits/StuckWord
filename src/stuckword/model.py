"""Classifier construction and honest evaluation.

Trials recorded in the same session share electrode contact, fatigue and drift, so random
K-fold splits overestimate accuracy. When more than one session is available we hold out whole
sessions (leave-one-session-out); the permutation test gives a single-subject chance level.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stuckword.config import FREQ_BANDS
from stuckword.features import BandPowerTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    metric: str
    fold_scores: np.ndarray
    permutation_scores: np.ndarray | None = None
    p_value: float | None = None

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.nanstd(self.fold_scores))

    def summary(self) -> str:
        text = f"{self.metric}: {self.mean:.3f} ± {self.std:.3f} over {len(self.fold_scores)} folds"
        if self.p_value is not None and self.permutation_scores is not None:
            text += (
                f" (chance {np.mean(self.permutation_scores):.3f}, "
                f"permutation p = {self.p_value:.4f})"
            )
        return text


def build_classifier(
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]] = FREQ_BANDS,
    picks: Sequence[int] | None = None,
    ch_names: Sequence[str] | None = None,
    C: float = 1.0,
) -> Pipeline:
    """Band power -> standardise -> L2 logistic regression.

    Deliberately simple: with ~100 trials per class a linear model on a few dozen features is
    about as much as the data can support, and its coefficients stay interpretable.
    """
    return Pipeline(
        [
            ("bandpower", BandPowerTransformer(sfreq, bands=bands, picks=picks, ch_names=ch_names)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000)),
        ]
    )


def make_session_cross_validator(
    groups: np.ndarray | None, n_splits: int = 5, seed: int = 0
) -> BaseCrossValidator:
    if groups is not None and len(np.unique(groups)) >= 2:
        return LeaveOneGroupOut()
    logger.warning("Only one session: falling back to stratified K-fold (optimistic).")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_classifier(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    metric: str = "roc_auc",
    n_permutations: int = 0,
    seed: int = 0,
) -> EvaluationResult:
    """Cross-validate ``estimator``; optionally add a label-permutation test.

    With ``groups`` the permutations are done within sessions, preserving each session's class
    balance.
    """
    cross_validator = make_session_cross_validator(groups, seed=seed)
    fold_scores = cross_val_score(
        estimator, X, y, groups=groups, cv=cross_validator, scoring=metric, error_score=np.nan
    )
    if not n_permutations:
        return EvaluationResult(metric, fold_scores)

    _, perm_scores, p_value = permutation_test_score(
        estimator,
        X,
        y,
        groups=groups,
        cv=cross_validator,
        scoring=metric,
        n_permutations=n_permutations,
        random_state=seed,
    )
    return EvaluationResult(metric, fold_scores, perm_scores, float(p_value))
