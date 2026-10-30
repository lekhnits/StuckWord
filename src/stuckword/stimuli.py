"""Missing-word listening task: stimulus timing and event construction.

The stimulus is a ~5-min Russian popular-science video about the psychology of long space
flights, with 48 words removed (``data/stimuli/missing_words_text.txt``, one ``[MASK]`` per
removed word). ``missing_words_timecodes.csv`` gives each gap as video timecodes
``M:SS:FF`` at 25 frames per second - not ``H:MM:SS``: the frame field never exceeds 24, the
gaps last 0.24-1.04 s (one spoken word) and the last one ends at 4:58, inside the 303-s
recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_FPS = 25
MASK = "[MASK]"
STIMULI_DIR = Path(__file__).resolve().parents[2] / "data" / "stimuli"


def timecode_to_seconds(timecode: str, fps: int = VIDEO_FPS) -> float:
    """``"2:26:24"`` (minutes:seconds:frames) -> 146.96 s."""
    minutes, seconds, frames = (int(part) for part in timecode.split(":"))
    if not (0 <= seconds < 60 and 0 <= frames < fps):
        raise ValueError(f"Not an M:SS:FF timecode at {fps} fps: {timecode!r}")
    return minutes * 60 + seconds + frames / fps


@dataclass(frozen=True)
class MissingWordStimulus:
    gaps: pd.DataFrame  # one row per removed word: onset_s, offset_s, duration_s, sentence

    @property
    def onsets(self) -> np.ndarray:
        return self.gaps["onset_s"].to_numpy()

    @property
    def offsets(self) -> np.ndarray:
        return self.gaps["offset_s"].to_numpy()


def load_missing_word_stimulus(stimuli_dir: Path = STIMULI_DIR) -> MissingWordStimulus:
    """Gap times plus the sentence each gap belongs to (masks are matched in reading order)."""
    timing = pd.read_csv(stimuli_dir / "missing_words_timecodes.csv")
    sentences = [
        line.strip()
        for line in (stimuli_dir / "missing_words_text.txt").read_text("utf-8").splitlines()
        if line.strip()
    ]
    sentence_of_each_mask = [
        index for index, sentence in enumerate(sentences) for _ in range(sentence.count(MASK))
    ]
    if len(sentence_of_each_mask) != len(timing):
        n_masks, n_gaps = len(sentence_of_each_mask), len(timing)
        raise ValueError(f"{n_masks} masks in the text but {n_gaps} gaps in the timecodes")

    gaps = timing.assign(
        onset_s=timing["start_tc"].map(timecode_to_seconds),
        offset_s=timing["end_tc"].map(timecode_to_seconds),
        sentence_index=sentence_of_each_mask,
        sentence=[sentences[index] for index in sentence_of_each_mask],
    )
    gaps["duration_s"] = gaps["offset_s"] - gaps["onset_s"]
    if (gaps["duration_s"] <= 0).any() or not gaps["onset_s"].is_monotonic_increasing:
        raise ValueError("Gaps must have positive duration and be in chronological order")
    return MissingWordStimulus(gaps)


def sample_speech_times(
    gap_onsets: np.ndarray,
    gap_offsets: np.ndarray,
    n: int,
    *,
    t_min: float,
    t_max: float,
    min_distance_s: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Times during uninterrupted speech, at least ``min_distance_s`` from any gap.

    These moments of the recording are the comparison condition for the missing words; the
    seed only makes the choice reproducible.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(t_min, t_max, 0.01)
    distance = np.min(
        np.abs(grid[:, None] - np.concatenate([gap_onsets, gap_offsets])[None, :]), axis=1
    )
    candidates = grid[distance >= min_distance_s]
    chosen: list[float] = []
    for t in rng.permutation(candidates):
        if all(abs(t - c) >= 1.0 for c in chosen):  # controls must not overlap each other
            chosen.append(float(t))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"Only {len(chosen)} control times fit, wanted {n}")
    return np.sort(chosen)
