"""Days 1-2: signal checks on effects that must be there before trusting anything subtler."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from stuckword.config import OCCIPITAL_CHANNELS

# PhysioNet EEG Motor Movement/Imagery: run 1 = eyes-open baseline, run 2 = eyes-closed baseline.
_EEGBCI_EYES_OPEN_RUN = 1
_EEGBCI_EYES_CLOSED_RUN = 2


def band_power_per_channel(
    raw: mne.io.BaseRaw, fmin: float, fmax: float, picks: Sequence[str]
) -> pd.Series:
    """Absolute mean PSD (µV²/Hz) in ``[fmin, fmax]`` per channel."""
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, picks=list(picks), verbose="error")
    psd = spectrum.get_data()  # V²/Hz
    return pd.Series(psd.mean(axis=-1) * 1e12, index=spectrum.ch_names, name=f"{fmin}-{fmax} Hz")


def alpha_reactivity(
    raw_eyes_open: mne.io.BaseRaw,
    raw_eyes_closed: mne.io.BaseRaw,
    picks: Sequence[str] = OCCIPITAL_CHANNELS,
) -> pd.Series:
    """Eyes-closed / eyes-open alpha (8-13 Hz) power ratio per channel.

    A working occipital electrode typically shows a ratio well above 2. A ratio near 1 usually
    means poor contact or a misplaced electrode.
    """
    open_ = band_power_per_channel(raw_eyes_open, 8.0, 13.0, picks)
    closed = band_power_per_channel(raw_eyes_closed, 8.0, 13.0, picks)
    return (closed / open_).rename("alpha_closed_over_open")


def count_blinks(raw: mne.io.BaseRaw, ch_name: str = "Fp1") -> int:
    """Number of blinks found on a frontopolar channel (compare with the number performed)."""
    events = mne.preprocessing.find_eog_events(raw, ch_name=ch_name, verbose="error")
    return len(events)


def emg_power(raw: mne.io.BaseRaw, picks: Sequence[str] | None = None) -> pd.Series:
    """High-frequency (30-45 Hz) power per channel; jaw clench should raise it sharply,
    especially on temporal/frontal sites. Compare clench and rest segments."""
    picks = picks if picks is not None else raw.ch_names
    return band_power_per_channel(raw, 30.0, 45.0, picks)


def load_eegbci_eyes(subject: int = 1, path: Path | None = None) -> tuple[mne.io.Raw, mne.io.Raw]:
    """Download (~5 MB) and load eyes-open / eyes-closed baselines from PhysioNet EEGBCI."""
    raws = []
    for run in (_EEGBCI_EYES_OPEN_RUN, _EEGBCI_EYES_CLOSED_RUN):
        (fname,) = mne.datasets.eegbci.load_data(subject, [run], path=path, update_path=False)
        raw = mne.io.read_raw_edf(fname, preload=True, verbose="error")
        mne.datasets.eegbci.standardize(raw)  # "O1.." -> "O1"
        raws.append(raw)
    return raws[0], raws[1]


def eegbci_alpha_check(subject: int = 1) -> pd.Series:
    """Run :func:`alpha_reactivity` on a public recording to validate the code path."""
    raw_open, raw_closed = load_eegbci_eyes(subject)
    for raw in (raw_open, raw_closed):
        raw.filter(1.0, 40.0, verbose="error")
    return alpha_reactivity(raw_open, raw_closed)


def is_alpha_reactive(ratios: pd.Series, min_ratio: float = 1.5) -> bool:
    return bool(np.nanmedian(ratios.to_numpy()) >= min_ratio)
