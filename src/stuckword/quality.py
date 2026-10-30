"""Recording-quality metrics that work on uncalibrated data (raw ADC counts)."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from stuckword.acquisition.edf import ADC_MAX, ADC_MIN


def clipped_fraction(raw: mne.io.BaseRaw, lo: float = ADC_MIN, hi: float = ADC_MAX) -> pd.Series:
    """Fraction of samples sitting at the ADC rails (amplifier saturated)."""
    data = raw.get_data(picks="eeg")
    return pd.Series(((data <= lo) | (data >= hi)).mean(axis=1), index=_eeg_channel_names(raw))


def _peak_to_peak_per_window(raw: mne.io.BaseRaw, window_s: float = 1.0) -> np.ndarray:
    """Peak-to-peak per non-overlapping window: shape ``(n_channels, n_windows)``."""
    data = raw.get_data(picks="eeg")
    step = round(window_s * raw.info["sfreq"])
    n_windows = data.shape[1] // step
    windows = data[:, : n_windows * step].reshape(data.shape[0], n_windows, step)
    return np.ptp(windows, axis=2)


def flat_window_fraction(raw: mne.io.BaseRaw, min_ptp: float = 1.0) -> pd.Series:
    """Fraction of 1-s windows with (almost) no signal - a disconnected electrode.

    Counting repeated samples is not enough here: at ~1 count resolution and ~7 counts of
    typical deviation, ties between consecutive samples are common in perfectly good EEG.
    """
    return pd.Series(
        (_peak_to_peak_per_window(raw) < min_ptp).mean(axis=1), index=_eeg_channel_names(raw)
    )


def artifact_window_fraction(raw: mne.io.BaseRaw, max_ptp: float = 200.0) -> pd.Series:
    """Fraction of 1-s windows whose peak-to-peak exceeds ``max_ptp`` (movement, pops, EMG).

    The default assumes the data are roughly in microvolts (1 count ~ 1 uV); adjust if the
    amplifier gain turns out differently.
    """
    return pd.Series(
        (_peak_to_peak_per_window(raw) > max_ptp).mean(axis=1), index=_eeg_channel_names(raw)
    )


def robust_amplitude(raw: mne.io.BaseRaw) -> pd.Series:
    """Median absolute deviation per channel, in the recording's own units."""
    data = raw.get_data(picks="eeg")
    mad = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1)
    return pd.Series(mad, index=_eeg_channel_names(raw))


def find_bad_channels(
    raw: mne.io.BaseRaw,
    *,
    max_clipped: float = 0.01,
    max_flat: float = 0.2,
    max_artifact: float = 0.2,
    amplitude_ratio: float = 5.0,
) -> list[str]:
    """Channels that saturate, go flat, are mostly artefact, or whose amplitude is more than
    ``amplitude_ratio`` times above/below the median channel."""
    mad = robust_amplitude(raw)
    ratio = mad / mad.median()
    bad = (
        (clipped_fraction(raw) > max_clipped)
        | (flat_window_fraction(raw) > max_flat)
        | (artifact_window_fraction(raw) > max_artifact)
        | (ratio > amplitude_ratio)
        | (ratio < 1 / amplitude_ratio)
    )
    return list(bad[bad].index)


@dataclass(frozen=True)
class EmgBursts:
    times: np.ndarray  # envelope time axis (s)
    envelope: np.ndarray  # log envelope, median-centred
    threshold: float
    onsets: np.ndarray  # burst onset times (s)


def detect_emg_bursts(
    raw: mne.io.BaseRaw,
    picks: list[str],
    *,
    band: tuple[float, float] = (30.0, 90.0),
    envelope_hz: float = 10.0,
    n_mad: float = 3.0,
    min_gap_s: float = 1.0,
) -> EmgBursts:
    """Find bursts of high-frequency muscle activity (speech, jaw clench, swallowing).

    Without event markers these are the best available *candidate* moments of speaking. They
    are not trial markers: blinks, swallowing and movement trigger them too.
    """
    sfreq = raw.info["sfreq"]
    high = min(band[1], sfreq / 2 - 1)
    data = raw.copy().pick(picks).filter(band[0], high, verbose="error").get_data()
    envelope = np.abs(hilbert(data, axis=1)).mean(axis=0)

    step = max(1, round(sfreq / envelope_hz))
    n_used = len(envelope) // step * step
    envelope = np.log(envelope[:n_used].reshape(-1, step).mean(axis=1) + 1e-12)
    envelope -= np.median(envelope)
    times = (np.arange(len(envelope)) + 0.5) * step / sfreq

    threshold = n_mad * np.median(np.abs(envelope)) * 1.4826
    crossings = np.flatnonzero((envelope[1:] > threshold) & (envelope[:-1] <= threshold)) + 1
    onsets: list[float] = []
    for t in times[crossings]:
        if not onsets or t - onsets[-1] >= min_gap_s:
            onsets.append(float(t))
    return EmgBursts(times, envelope, float(threshold), np.array(onsets))


def _eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    # exclude=[] keeps channels marked bad, matching get_data(picks="eeg").
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
