import numpy as np
import pytest

from stuckword.acquisition.edf import ADC_MAX, recording_start_from_filename
from stuckword.quality import (
    clipped_fraction,
    detect_emg_bursts,
    find_bad_channels,
    flat_window_fraction,
)


@pytest.fixture
def raw_counts(eyes):
    """Real EEG rescaled to ~ADC counts (1 count ~ 1 uV, like the EDF exports), with two
    electrode faults: Cz disconnected, Pz saturated for 10 s."""
    raw = eyes.raw_open.copy()
    raw._data *= 1e6
    raw._data[raw.ch_names.index("Cz")] = 0.0
    sfreq = int(raw.info["sfreq"])
    raw._data[raw.ch_names.index("Pz"), : 10 * sfreq] = ADC_MAX
    return raw


def test_quality_metrics_flag_the_faulty_channels(raw_counts):
    assert flat_window_fraction(raw_counts)["Cz"] == 1.0
    assert clipped_fraction(raw_counts)["Pz"] == pytest.approx(10 / raw_counts.times[-1], rel=0.02)
    assert {"Cz", "Pz"} <= set(find_bad_channels(raw_counts))


def test_metrics_stay_aligned_after_marking_bads(raw_counts):
    raw_counts.info["bads"] = ["Cz"]
    assert list(flat_window_fraction(raw_counts).index) == raw_counts.ch_names


def test_emg_bursts_are_ordered_and_spaced(eyes):
    bursts = detect_emg_bursts(eyes.raw_open, ["F7", "F8"], min_gap_s=1.0)
    assert bursts.threshold > 0
    assert np.all(np.diff(bursts.onsets) >= 1.0)
    assert len(bursts.times) == len(bursts.envelope)


def test_recording_start_from_filename():
    assert recording_start_from_filename("result_31.03.2023_14.12.41.edf").hour == 14
    assert recording_start_from_filename("5.edf") is None
