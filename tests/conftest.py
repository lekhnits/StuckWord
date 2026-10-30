"""Shared fixtures. All EEG used in tests is real: PhysioNet EEGBCI eyes-open/eyes-closed
baselines (subjects 1-2, ~5 MB, cached by MNE in ~/mne_data)."""

from dataclasses import dataclass

import mne
import numpy as np
import pytest

from stuckword.config import OCTOPUS_16_MONTAGE


@dataclass(frozen=True)
class EyesData:
    """2-s epochs; y = 1 for eyes closed, groups = subject."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sfreq: float
    ch_names: list[str]
    raw_open: mne.io.BaseRaw


@pytest.fixture(scope="session")
def eyes() -> EyesData:
    from stuckword.sanity import load_eegbci_eyes

    X, y, groups = [], [], []
    raw_open = None
    for subject in (1, 2):
        try:
            raws = load_eegbci_eyes(subject)
        except Exception as err:  # offline and not cached
            pytest.skip(f"PhysioNet EEGBCI data unavailable: {err}")
        for label, raw in enumerate(raws):
            raw = raw.pick(list(OCTOPUS_16_MONTAGE)).filter(1.0, 40.0, verbose="error")
            raw_open = raw_open or raw
            data = mne.make_fixed_length_epochs(raw, duration=2.0, verbose="error").get_data()
            X.append(data)
            y.append(np.full(len(data), label))
            groups.append(np.full(len(data), subject))
    return EyesData(
        np.concatenate(X),
        np.concatenate(y),
        np.concatenate(groups),
        raw_open.info["sfreq"],
        list(OCTOPUS_16_MONTAGE),
        raw_open,
    )
