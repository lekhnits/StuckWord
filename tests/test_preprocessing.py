import numpy as np
import pytest

from stuckword.config import EpochConfig
from stuckword.markers import MarkerKind, RecallLabel, format_marker, markers_to_annotations
from stuckword.preprocessing import binary_labels, make_recall_epochs, preprocess_raw


@pytest.fixture
def labelled_raw(eyes):
    """Real eyes-open EEG with the flashcard marker protocol attached."""
    raw = eyes.raw_open.copy()
    markers = []
    labels = [RecallLabel.KNEW, RecallLabel.TIP_OF_TONGUE, RecallLabel.NO_IDEA] * 3
    for trial, label in enumerate(labels):
        cue = 3.0 + trial * 6.0
        markers += [
            (cue, format_marker(MarkerKind.CUE, trial, "w")),
            (cue + 3.0, format_marker(MarkerKind.RETRIEVED, trial, "space")),
            (cue + 5.0, format_marker(MarkerKind.LABEL, trial, label.value)),
        ]
    raw.set_annotations(markers_to_annotations(markers, t0=0.0))
    return raw


def test_preprocess_returns_filtered_average_referenced_copy(labelled_raw):
    clean = preprocess_raw(labelled_raw)
    assert clean is not labelled_raw
    assert clean.info["highpass"] == pytest.approx(1.0)
    assert np.abs(clean.get_data().sum(axis=0)).max() < 1e-12


def test_retrieved_locked_epochs_exclude_no_idea(labelled_raw):
    config = EpochConfig(tmin=-2.0, tmax=0.0, reject_peak_to_peak_uv=None)
    epochs = make_recall_epochs(preprocess_raw(labelled_raw), config)
    assert len(epochs) == 6
    assert binary_labels(epochs).sum() == 3
    assert epochs.tmin == pytest.approx(-2.0)


def test_missing_annotations_raise(labelled_raw):
    with pytest.raises(ValueError, match="No annotations"):
        make_recall_epochs(labelled_raw.copy().set_annotations(None))
