"""Load a LabRecorder XDF file (EEG + marker streams) into an annotated :class:`mne.io.RawArray`.

Recording setup: ``pieeg-server --lsl`` streams the Octopus EEG, the flashcard app streams
markers, and LabRecorder writes both to one XDF file on a shared clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np

from stuckword.config import DEFAULT_SFREQ, OCTOPUS_16_MONTAGE
from stuckword.markers import markers_to_annotations

MICROVOLTS_TO_VOLTS = 1e-6


def _standard_montage() -> str:
    # MNE 1.13 renamed "standard_1005" to "colin27_1005"; support both.
    names = mne.channels.get_builtin_montages()
    return "colin27_1005" if "colin27_1005" in names else "standard_1005"


def _find_stream(streams: list[dict[str, Any]], stream_type: str) -> dict[str, Any]:
    matches = [s for s in streams if s["info"]["type"][0] == stream_type]
    if len(matches) != 1:
        found = [(s["info"]["name"][0], s["info"]["type"][0]) for s in streams]
        raise ValueError(f"Expected exactly one {stream_type!r} stream, found {found}")
    return matches[0]


def _channel_labels(stream: dict[str, Any]) -> list[str] | None:
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        return [ch["label"][0] for ch in channels]
    except (KeyError, IndexError, TypeError):
        return None


def load_xdf_session(
    path: Path | str,
    *,
    eeg_stream_type: str = "EEG",
    marker_stream_type: str = "Markers",
    ch_names: Sequence[str] | None = None,
    scale_to_volts: float = MICROVOLTS_TO_VOLTS,
) -> mne.io.RawArray:
    """Read one session.

    Parameters
    ----------
    ch_names
        Override channel names (defaults to labels in the stream, then
        :data:`stuckword.config.OCTOPUS_16_MONTAGE`).
    scale_to_volts
        Multiplier from stream units to volts. PiEEG server streams microvolts.
    """
    import pyxdf  # optional dependency, imported lazily

    streams, _ = pyxdf.load_xdf(str(path))
    eeg = _find_stream(streams, eeg_stream_type)
    markers = _find_stream(streams, marker_stream_type)

    data = np.asarray(eeg["time_series"], dtype=float).T * scale_to_volts
    sfreq = float(eeg["info"]["nominal_srate"][0]) or DEFAULT_SFREQ
    names = list(ch_names or _channel_labels(eeg) or OCTOPUS_16_MONTAGE[: data.shape[0]])
    if len(names) != data.shape[0]:
        raise ValueError(f"{len(names)} channel names for {data.shape[0]} channels")

    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_montage(_standard_montage(), on_missing="warn")

    t0 = float(eeg["time_stamps"][0])
    pairs = [
        (float(ts), sample[0])
        for ts, sample in zip(markers["time_stamps"], markers["time_series"], strict=True)
    ]
    raw.set_annotations(markers_to_annotations(pairs, t0=t0))
    return raw
