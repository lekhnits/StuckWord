"""Offline preprocessing and epoching of recall sessions."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import mne
import numpy as np

from stuckword.config import EpochConfig, PreprocessingConfig
from stuckword.markers import RecallLabel

logger = logging.getLogger(__name__)


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig | None = None,
) -> mne.io.BaseRaw:
    """Notch, band-pass and re-reference a copy of ``raw``."""
    config = config or PreprocessingConfig()
    raw = raw.copy().load_data()

    nyquist = raw.info["sfreq"] / 2
    notch_freqs = [freq for freq in config.notch_freqs if freq < nyquist]
    if notch_freqs:
        raw.notch_filter(notch_freqs, verbose="error")
    raw.filter(config.l_freq, min(config.h_freq, nyquist - 1), verbose="error")

    reference = config.reference if config.reference == "average" else list(config.reference)
    raw.set_eeg_reference(reference, projection=False, verbose="error")

    if config.resample_sfreq is not None:
        raw.resample(config.resample_sfreq, verbose="error")
    return raw


def make_recall_epochs(
    raw: mne.io.BaseRaw,
    config: EpochConfig | None = None,
    labels: Sequence[RecallLabel] = (RecallLabel.KNEW, RecallLabel.TIP_OF_TONGUE),
) -> mne.Epochs:
    """Cut epochs around ``cue`` or ``retrieved`` annotations for the requested labels.

    Annotations must follow the ``"<kind>/<label>"`` scheme from
    :func:`stuckword.markers.markers_to_annotations`.
    """
    config = config or EpochConfig()
    wanted = {f"{config.lock}/{label.value}" for label in labels}
    present = set(raw.annotations.description) & wanted
    if not present:
        found = sorted(set(raw.annotations.description))
        raise ValueError(f"No annotations matching {sorted(wanted)}; found {found}")
    # Stable integer codes independent of which labels happen to be present in this file.
    event_id = {f"{config.lock}/{label.value}": i + 1 for i, label in enumerate(RecallLabel)}
    event_id = {name: code for name, code in event_id.items() if name in present}
    events, event_id = mne.events_from_annotations(raw, event_id=event_id, verbose="error")

    reject = (
        {"eeg": config.reject_peak_to_peak_uv * 1e-6}
        if config.reject_peak_to_peak_uv is not None
        else None
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        reject=reject,
        preload=True,
        verbose="error",
    )
    logger.info("Kept %d/%d epochs after rejection", len(epochs), len(events))
    return epochs


def binary_labels(
    epochs: mne.Epochs, positive: RecallLabel = RecallLabel.TIP_OF_TONGUE
) -> np.ndarray:
    """Return 1 for epochs whose event name carries the ``positive`` tag, else 0."""
    code_to_name = {code: name for name, code in epochs.event_id.items()}
    names = [code_to_name[code] for code in epochs.events[:, 2]]
    return np.array([name.split("/")[-1] == positive.value for name in names], dtype=int)
