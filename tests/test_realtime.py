import numpy as np
import pytest

from stuckword.model import build_classifier
from stuckword.realtime import (
    CausalFilter,
    DetectorState,
    OnlineStruggleEstimator,
    RingBuffer,
    StruggleDetector,
)


def test_ring_buffer_keeps_latest_samples():
    buffer = RingBuffer(n_channels=1, n_samples=4)
    buffer.extend(np.array([[1, 2, 3]]))
    assert not buffer.is_full
    buffer.extend(np.array([[4, 5]]))
    assert buffer.is_full
    np.testing.assert_array_equal(buffer.get(), [[2, 3, 4, 5]])


def test_causal_filter_is_chunk_invariant(eyes):
    signal = eyes.raw_open.get_data(picks=["Fz", "Oz"], stop=1000)
    whole = CausalFilter(2, eyes.sfreq)(signal)
    chunked_filter = CausalFilter(2, eyes.sfreq)
    chunked = np.hstack([chunked_filter(c) for c in np.array_split(signal, 7, axis=1)])
    np.testing.assert_allclose(whole, chunked)


def test_detector_needs_sustained_struggle():
    detector = StruggleDetector(on_threshold=0.6, off_threshold=0.4, min_duration_s=3.0)
    assert detector.update(0.7, 0.0) is DetectorState.SEARCHING
    assert detector.update(0.3, 1.0) is DetectorState.IDLE  # dropped -> wait
    assert detector.update(0.7, 2.0) is DetectorState.SEARCHING
    assert detector.update(0.5, 4.0) is DetectorState.SEARCHING  # inside hysteresis band
    assert detector.update(0.7, 5.0) is DetectorState.HINT
    assert detector.update(0.0, 6.0) is DetectorState.HINT  # latched until reset
    detector.reset()
    assert detector.state is DetectorState.IDLE


def test_detector_validates_thresholds():
    with pytest.raises(ValueError):
        StruggleDetector(on_threshold=0.3, off_threshold=0.5)


def test_online_estimator_tracks_eyes_closing(eyes):
    """Fit on subject 1, stream subject 2: P(eyes closed) is higher while the eyes are closed."""
    train = eyes.groups == 1
    pipeline = build_classifier(eyes.sfreq).fit(eyes.X[train], eyes.y[train])

    def stream(epochs):
        estimator = OnlineStruggleEstimator(pipeline, n_channels=eyes.X.shape[1], sfreq=eyes.sfreq)
        assert estimator.predict() is None
        probabilities = []
        for epoch in epochs:
            estimator.push(epoch)
            probabilities.append(estimator.predict())
        return np.array(probabilities[2:])  # skip the filter start-up transient

    test = eyes.groups == 2
    p_open = stream(eyes.X[test & (eyes.y == 0)][:10])
    p_closed = stream(eyes.X[test & (eyes.y == 1)][:10])
    assert ((p_open >= 0) & (p_open <= 1)).all()
    assert p_closed.mean() > p_open.mean()
