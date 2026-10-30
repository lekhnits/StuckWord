"""Spectral band-power features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

from stuckword.config import FREQ_BANDS

_EPS = 1e-30


def compute_band_power(
    data: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] = FREQ_BANDS,
    *,
    relative: bool = True,
    log: bool = True,
) -> np.ndarray:
    """Welch power integrated over each band.

    Parameters
    ----------
    data
        Array of shape ``(n_epochs, n_channels, n_times)``.
    sfreq
        Sampling frequency in Hz.
    bands
        Mapping of band name to ``(low, high)`` edges in Hz (low inclusive, high exclusive).
    relative
        Divide by total power from the lowest to the highest band edge, giving the fraction of
        power in each band. This removes per-channel gain differences
        (useful with dry/taped electrodes whose impedance drifts between sessions).
    log
        Return ``log10`` power; band power is roughly log-normal, which suits linear models.

    Returns
    -------
    np.ndarray
        Shape ``(n_epochs, n_channels, n_bands)``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected (n_epochs, n_channels, n_times), got shape {data.shape}")

    # 1-s segments give 1 Hz resolution; shorter epochs fall back to a single segment.
    nperseg = min(data.shape[-1], round(sfreq))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)

    powers = np.stack([_band_power(psd, freqs, low, high) for low, high in bands.values()], axis=-1)
    if relative:
        lowest = min(edges[0] for edges in bands.values())
        highest = max(edges[1] for edges in bands.values())
        total_power = _band_power(psd, freqs, lowest, highest)
        powers = powers / (total_power[..., np.newaxis] + _EPS)
    if log:
        powers = np.log10(powers + _EPS)
    return powers


def _band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    """Sum of PSD bins in ``[low, high)`` times bin width: power, not density."""
    in_band = (freqs >= low) & (freqs < high)
    if not in_band.any():
        raise ValueError(f"No frequency bins in [{low}, {high}) Hz; epoch too short?")
    return psd[..., in_band].sum(axis=-1) * (freqs[1] - freqs[0])


class BandPowerTransformer(TransformerMixin, BaseEstimator):
    """scikit-learn transformer: raw epochs -> flattened (channel x band) log band power.

    Keeping feature extraction inside the pipeline means the exact same code runs offline, in
    cross-validation and in the live loop, and channel subsets are just a ``picks`` parameter.
    """

    def __init__(
        self,
        sfreq: float,
        bands: Mapping[str, tuple[float, float]] = FREQ_BANDS,
        picks: Sequence[int] | None = None,
        ch_names: Sequence[str] | None = None,
        relative: bool = True,
        log: bool = True,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.picks = picks
        self.ch_names = ch_names
        self.relative = relative
        self.log = log

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BandPowerTransformer:
        self.n_channels_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.shape[1] != self.n_channels_in_:
            raise ValueError(f"Fitted on {self.n_channels_in_} channels, got {X.shape[1]}")
        if self.picks is not None:
            X = X[:, list(self.picks), :]
        band_power = compute_band_power(
            X, self.sfreq, self.bands, relative=self.relative, log=self.log
        )
        return band_power.reshape(len(band_power), -1)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        n_channels = self.n_channels_in_
        names = (
            list(self.ch_names)
            if self.ch_names is not None
            else [f"ch{index}" for index in range(n_channels)]
        )
        if self.picks is not None:
            names = [names[i] for i in self.picks]
        return np.array([f"{ch}_{band}" for ch in names for band in self.bands], dtype=object)
