import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from stuckword.channels import forward_select_channels, rank_channels
from stuckword.model import build_classifier, evaluate_classifier, make_session_cross_validator

POSTERIOR = {"P3", "Pz", "P4", "O1", "Oz", "O2"}


def test_cross_validator_holds_out_whole_sessions():
    assert isinstance(make_session_cross_validator(np.array([0, 0, 1, 1])), LeaveOneGroupOut)
    assert isinstance(make_session_cross_validator(np.zeros(4)), StratifiedKFold)
    assert isinstance(make_session_cross_validator(None), StratifiedKFold)


def test_classifier_generalises_to_a_new_person(eyes):
    result = evaluate_classifier(build_classifier(eyes.sfreq), eyes.X, eyes.y, eyes.groups)
    assert result.mean > 0.8


def test_classifier_is_at_chance_with_shuffled_labels(eyes):
    shuffled = np.random.default_rng(0).permutation(eyes.y)
    result = evaluate_classifier(
        build_classifier(eyes.sfreq), eyes.X, shuffled, eyes.groups, n_permutations=20
    )
    assert 0.3 < result.mean < 0.7
    assert result.p_value is not None and result.p_value > 0.05


def test_alpha_ranking_puts_posterior_channels_first(eyes):
    table = rank_channels(eyes.X, eyes.y, eyes.sfreq, eyes.ch_names, band="alpha")
    assert set(table["channel"].head(3)) <= POSTERIOR
    assert (table["auc"].head(3) > 0.5).all()


def test_forward_selection_starts_posterior(eyes):
    table = forward_select_channels(
        eyes.X, eyes.y, eyes.sfreq, eyes.ch_names, eyes.groups, max_channels=2
    )
    assert list(table["step"]) == [1, 2]
    assert table.loc[0, "added"] in POSTERIOR
