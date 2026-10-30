import numpy as np
import pytest

from stuckword.config import FREQ_BANDS
from stuckword.features import BandPowerTransformer, compute_band_power


def test_eyes_closed_raise_occipital_alpha(eyes):
    bp = compute_band_power(eyes.X, eyes.sfreq, log=False)
    alpha = bp[:, :, list(FREQ_BANDS).index("alpha")]
    occipital = [eyes.ch_names.index(ch) for ch in ("O1", "Oz", "O2")]
    closed = alpha[eyes.y == 1][:, occipital].mean()
    open_ = alpha[eyes.y == 0][:, occipital].mean()
    assert closed > 1.5 * open_


def test_relative_power_is_a_fraction_of_total(eyes):
    bp = compute_band_power(eyes.X[:5], eyes.sfreq, log=False)
    assert bp.shape == (5, len(eyes.ch_names), len(FREQ_BANDS))
    np.testing.assert_allclose(bp.sum(axis=-1), 1.0)


def test_rejects_wrong_dimensionality(eyes):
    with pytest.raises(ValueError, match="n_epochs"):
        compute_band_power(eyes.X[0], eyes.sfreq)


def test_transformer_picks_and_feature_names(eyes):
    transformer = BandPowerTransformer(
        eyes.sfreq, bands={"alpha": (8.0, 13.0)}, picks=[0, 13], ch_names=eyes.ch_names
    ).fit(eyes.X)
    assert transformer.transform(eyes.X[:3]).shape == (3, 2)
    assert list(transformer.get_feature_names_out()) == ["Fp1_alpha", "O1_alpha"]


def test_transformer_checks_channel_count(eyes):
    transformer = BandPowerTransformer(eyes.sfreq).fit(eyes.X)
    with pytest.raises(ValueError, match="channels"):
        transformer.transform(eyes.X[:, :3])
