"""Connect the live EEG stream to the flashcard app's hint logic."""

from __future__ import annotations

import numpy as np

from stuckword.acquisition.xdf import MICROVOLTS_TO_VOLTS
from stuckword.realtime import DetectorState, OnlineStruggleEstimator, StruggleDetector


class LslHintProvider:
    """Pulls EEG from LSL, feeds the estimator, and asks the detector whether to hint.

    Implements :class:`stuckword.acquisition.flashcards.HintProvider`.
    """

    def __init__(
        self,
        estimator: OnlineStruggleEstimator,
        detector: StruggleDetector,
        stream_type: str = "EEG",
        timeout_s: float = 10.0,
    ) -> None:
        import pylsl

        found = pylsl.resolve_byprop("type", stream_type, timeout=timeout_s)
        if not found:
            raise RuntimeError(
                f"No LSL stream of type {stream_type!r}; is pieeg-server --lsl running?"
            )
        self._pylsl = pylsl
        self._inlet = pylsl.StreamInlet(found[0])
        self.estimator = estimator
        self.detector = detector
        self.last_probability: float | None = None

    def reset(self) -> None:
        self.detector.reset()

    def should_hint(self) -> bool:
        samples, _ = self._inlet.pull_chunk(timeout=0.0)
        if samples:
            self.estimator.push(np.asarray(samples).T * MICROVOLTS_TO_VOLTS)
        self.last_probability = self.estimator.predict()
        if self.last_probability is None:
            return False
        state = self.detector.update(self.last_probability, self._pylsl.local_clock())
        return state is DetectorState.HINT
