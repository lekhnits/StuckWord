"""Days 6-7: online struggle detection for the flashcard loop.

The pieces are kept separate so each can be tested without hardware:

* :class:`CausalFilter` - streaming band-pass (offline filters are zero-phase and see the future).
* :class:`RingBuffer` - last N seconds of EEG.
* :class:`OnlineStruggleEstimator` - filter + buffer + a fitted pipeline -> P(struggling).
* :class:`StruggleDetector` - turns a noisy probability into a hint decision with hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.pipeline import Pipeline


class CausalFilter:
    """Butterworth band-pass applied sample-by-sample, keeping state across chunks."""

    def __init__(
        self, n_channels: int, sfreq: float, l_freq: float = 1.0, h_freq: float = 45.0
    ) -> None:
        self._sections = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
        # Filter state per (section, channel); starts at rest. Expect ~1 s of start-up transient.
        self._filter_state = np.zeros((self._sections.shape[0], n_channels, 2))

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a ``(n_channels, n_samples)`` chunk."""
        filtered, self._filter_state = sosfilt(
            self._sections, chunk, axis=-1, zi=self._filter_state
        )
        return filtered


class RingBuffer:
    """Fixed-length ``(n_channels, n_samples)`` buffer of the most recent samples."""

    def __init__(self, n_channels: int, n_samples: int) -> None:
        self._data = np.zeros((n_channels, n_samples))
        self._n_seen = 0

    @property
    def is_full(self) -> bool:
        return self._n_seen >= self._data.shape[1]

    def extend(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk)
        n_new = min(chunk.shape[1], self._data.shape[1])
        self._data = np.roll(self._data, -n_new, axis=1)
        self._data[:, -n_new:] = chunk[:, -n_new:]
        self._n_seen += chunk.shape[1]

    def get(self) -> np.ndarray:
        return self._data.copy()


class OnlineStruggleEstimator:
    """Probability that the most recent window looks like a tip-of-the-tongue window.

    The pipeline must have been trained on windows of the same length, same channels and
    *cue-locked* (see :class:`stuckword.config.EpochConfig`), since that is what is available live.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        n_channels: int,
        sfreq: float,
        window_s: float = 2.0,
        positive_class: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self._filter = CausalFilter(n_channels, sfreq)
        self._buffer = RingBuffer(n_channels, round(window_s * sfreq))
        self._class_index = list(pipeline.classes_).index(positive_class)

    def push(self, chunk_volts: np.ndarray) -> None:
        filtered = self._filter(chunk_volts)
        self._buffer.extend(filtered - filtered.mean(axis=0, keepdims=True))  # average reference

    def predict(self) -> float | None:
        if not self._buffer.is_full:
            return None
        window = self._buffer.get()[np.newaxis]
        return float(self.pipeline.predict_proba(window)[0, self._class_index])


class DetectorState(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    HINT = "hint"


@dataclass
class StruggleDetector:
    """Hysteresis + minimum duration, so one noisy window does not trigger a hint.

    IDLE -> SEARCHING when p >= ``on_threshold``; back to IDLE when p < ``off_threshold``;
    SEARCHING -> HINT after ``min_duration_s`` of continuous searching. HINT is latched until
    :meth:`reset` (call it at the start of each trial).
    """

    on_threshold: float = 0.65
    off_threshold: float = 0.45
    min_duration_s: float = 3.0
    state: DetectorState = field(default=DetectorState.IDLE, init=False)
    _since: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.off_threshold <= self.on_threshold <= 1:
            raise ValueError("Need 0 <= off_threshold <= on_threshold <= 1")

    def reset(self) -> None:
        self.state = DetectorState.IDLE
        self._since = None

    def update(self, p_struggle: float, timestamp: float) -> DetectorState:
        if self.state is DetectorState.IDLE and p_struggle >= self.on_threshold:
            self.state, self._since = DetectorState.SEARCHING, timestamp
        elif self.state is DetectorState.SEARCHING:
            if p_struggle < self.off_threshold:
                self.reset()
            elif self._since is not None and timestamp - self._since >= self.min_duration_s:
                self.state = DetectorState.HINT
        return self.state
