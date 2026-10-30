"""Event-marker protocol shared by the flashcard app (writer) and the loader (reader).

Markers are plain strings pushed to an LSL marker stream, so LabRecorder stores them in the same
XDF file and on the same clock as the EEG::

    cue|12|perro          word shown
    hint|12|p             first-letter hint shown
    retrieved|12|space    participant found the word / gave up (or "timeout")
    label|12|tot          self-report given after answering

The self-report arrives last, so annotations are only built once a trial is complete.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import mne

SEPARATOR = "|"


class MarkerKind(str, Enum):
    SESSION_START = "session_start"
    CUE = "cue"
    HINT = "hint"
    RETRIEVED = "retrieved"
    LABEL = "label"
    SESSION_END = "session_end"


class RecallLabel(str, Enum):
    """Self-reported retrieval difficulty."""

    KNEW = "knew"  # knew it straight away
    TIP_OF_TONGUE = "tot"  # "it's right there"
    NO_IDEA = "no_idea"


# Keys pressed in the flashcard app after each answer.
KEY_TO_LABEL: dict[str, RecallLabel] = {
    "1": RecallLabel.KNEW,
    "2": RecallLabel.TIP_OF_TONGUE,
    "3": RecallLabel.NO_IDEA,
}

# Marker kinds that become time-locked annotations ("<kind>/<label>").
_TIMED_KINDS = (MarkerKind.CUE, MarkerKind.HINT, MarkerKind.RETRIEVED)


@dataclass(frozen=True)
class Marker:
    kind: MarkerKind
    trial: int
    payload: str = ""


def format_marker(kind: MarkerKind, trial: int, payload: str = "") -> str:
    if SEPARATOR in payload:
        raise ValueError(f"Marker payload must not contain {SEPARATOR!r}: {payload!r}")
    return SEPARATOR.join((kind.value, str(trial), payload))


def parse_marker(text: str) -> Marker:
    kind, trial, payload = text.split(SEPARATOR, maxsplit=2)
    return Marker(MarkerKind(kind), int(trial), payload)


def markers_to_annotations(
    markers: Iterable[tuple[float, str]],
    t0: float,
) -> mne.Annotations:
    """Turn ``(timestamp, marker_text)`` pairs into MNE annotations.

    Each timed marker of a labelled trial becomes ``"<kind>/<label>"`` (e.g. ``"retrieved/tot"``),
    so ``epochs["tot"]`` selects all tip-of-the-tongue trials via MNE's tag matching. Trials
    without a self-report (session aborted mid-trial) are dropped.

    Parameters
    ----------
    markers
        Pairs of LSL timestamp (seconds) and marker string.
    t0
        Timestamp of the first EEG sample; annotation onsets are relative to it.
    """
    timed: dict[int, list[tuple[float, MarkerKind]]] = {}
    labels: dict[int, RecallLabel] = {}
    for timestamp, text in markers:
        marker = parse_marker(text)
        if marker.kind in _TIMED_KINDS:
            timed.setdefault(marker.trial, []).append((timestamp, marker.kind))
        elif marker.kind is MarkerKind.LABEL:
            labels[marker.trial] = RecallLabel(marker.payload)

    onsets, descriptions = [], []
    for trial, events in sorted(timed.items()):
        label = labels.get(trial)
        if label is None:
            continue
        for timestamp, kind in events:
            onsets.append(timestamp - t0)
            descriptions.append(f"{kind.value}/{label.value}")
    return mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descriptions)
