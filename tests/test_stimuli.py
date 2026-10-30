import numpy as np
import pytest

from stuckword.stimuli import load_missing_word_stimulus, sample_speech_times, timecode_to_seconds


@pytest.mark.parametrize(
    ("timecode", "seconds"), [("0:25:00", 25.0), ("2:26:24", 146.96), ("4:58:09", 298.36)]
)
def test_timecodes_are_minutes_seconds_frames(timecode, seconds):
    assert timecode_to_seconds(timecode) == pytest.approx(seconds)


def test_invalid_frame_number_is_rejected():
    with pytest.raises(ValueError, match="M:SS:FF"):
        timecode_to_seconds("0:10:25")


def test_every_mask_has_a_gap_and_gaps_are_word_length():
    gaps = load_missing_word_stimulus().gaps
    assert len(gaps) == 48
    assert gaps["duration_s"].between(0.2, 1.1).all()
    assert gaps["offset_s"].max() < 303.0  # inside the recordings
    assert "нервных [MASK]" in gaps.loc[0, "sentence"]


def test_speech_times_avoid_gaps():
    stimulus = load_missing_word_stimulus()
    speech_times = sample_speech_times(
        stimulus.onsets, stimulus.offsets, 48, t_min=5.0, t_max=300.0
    )
    gap_edges = np.concatenate([stimulus.onsets, stimulus.offsets])
    assert len(speech_times) == 48
    assert np.min(np.abs(speech_times[:, None] - gap_edges[None, :])) >= 1.5
    assert np.all(np.diff(speech_times) >= 1.0)
