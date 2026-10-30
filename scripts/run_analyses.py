"""Run every analysis on real EEG and cache the results for ``make_figures.py``.

1. Missing-word listening task (``data/wordlists/MissingWords``, 14 recordings): ERP to the
   removed words vs ongoing speech, a stimulus/EEG timing check, and decoding tested on
   recordings left out of training.
2. ERP CORE N400 (Kappenman et al., 2021): related vs unrelated targets, using whichever
   participants are already in ``data/external/erpcore_n400``. Nothing is downloaded.
3. Kiloword (Dufau et al., 2015; 25 MB via MNE): word-property regression ERPs and prediction.

Outputs: ``results/cache/*.pkl`` (arrays for plotting), ``results/summary.csv`` and
``results/missingwords_recordings.csv``.

Usage::

    python scripts/run_analyses.py [--skip-n400] [--skip-kiloword]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stuckword.acquisition.edf import load_edf_recording, recording_start_from_filename
from stuckword.datasets import DEFAULT_DATA_DIR, load_erpcore_n400_epochs, load_kiloword_epochs
from stuckword.erp import (
    cluster_test_condition_difference,
    decode_each_time_point,
    find_above_chance_times,
    make_erp_classifier,
    mean_amplitudes_in_time_bins,
)
from stuckword.quality import find_bad_channels
from stuckword.stimuli import load_missing_word_stimulus, sample_speech_times

logger = logging.getLogger("run_analyses")

ROOT = Path(__file__).resolve().parents[1]
N400_DIR = ROOT / DEFAULT_DATA_DIR / "erpcore_n400"
N_LABEL_PERMUTATIONS = 1000
MISSING_WORD_CODE, SPEECH_CODE = 1, 2


# ---------------------------------------------------------------------------------------------
# 1. Missing-word listening task
# ---------------------------------------------------------------------------------------------
@dataclass
class Recording:
    name: str  # start time, e.g. "31 Mar 14:12"
    raw: mne.io.BaseRaw  # 0.3-30 Hz, bad channels interpolated, average reference
    interpolated_channels: list[str]


def load_missing_word_recordings(data_dir: Path) -> list[Recording]:
    recordings = []
    for edf_path in sorted(data_dir.glob("result_*.edf")):  # 5.edf duplicates 31 Mar 14:12
        raw = load_edf_recording(edf_path).pick("eeg")
        # Each file starts with a large ramp and ends with a step; judge quality without them.
        raw_without_edges = raw.copy().crop(2.0, raw.times[-1] - 1.0)
        raw.info["bads"] = bad_channels = find_bad_channels(raw_without_edges)
        raw.filter(0.3, 30.0, verbose="error")
        if bad_channels:
            raw.interpolate_bads(reset_bads=True, verbose="error")
        raw.set_eeg_reference("average", verbose="error")
        name = recording_start_from_filename(edf_path).strftime("%d %b %H:%M")
        recordings.append(Recording(name, raw, bad_channels))
    return recordings


def epoch_missing_words_and_speech(
    recording: Recording, gap_onsets: np.ndarray, gap_offsets: np.ndarray, seed: int
) -> mne.Epochs:
    """Epochs at the 48 gap onsets ("missing") and at 48 moments of uninterrupted speech."""
    raw = recording.raw
    speech_times = sample_speech_times(
        gap_onsets, gap_offsets, len(gap_onsets), t_min=5.0, t_max=raw.times[-1] - 2.0, seed=seed
    )
    event_times = np.r_[gap_onsets, speech_times]
    event_codes = np.r_[
        np.full(len(gap_onsets), MISSING_WORD_CODE), np.full(len(speech_times), SPEECH_CODE)
    ]
    chronological = np.argsort(event_times)
    event_samples = np.round(event_times[chronological] * raw.info["sfreq"]).astype(int)
    events = np.column_stack(
        [event_samples, np.zeros(len(event_samples), int), event_codes[chronological]]
    )
    return mne.Epochs(
        raw,
        events,
        {"missing": MISSING_WORD_CODE, "speech": SPEECH_CODE},
        tmin=-0.5,
        tmax=1.5,
        baseline=(-0.2, 0.0),
        reject={"eeg": 200.0},  # data are ADC counts (~uV): ~200 uV peak-to-peak
        decim=2,
        preload=True,
        verbose="error",
    )


def evoked_strength_for_shifted_events(recordings, stimulus, shifts, n_random_sets=200, seed=0):
    """How strongly the EEG responds to the gaps if the timecodes were off by ``shifts``.

    Returns the response strength for onset-locked and offset-locked epochs at every shift,
    and for ``n_random_sets`` sets of random event times (the null distribution).
    """
    sfreq = recordings[0].raw.info["sfreq"]
    lags = np.arange(int(-0.2 * sfreq), int(0.6 * sfreq))  # samples relative to the event
    in_response_window = (lags / sfreq > 0.05) & (lags / sfreq < 0.35)
    standardised_recordings = []
    for recording in recordings:
        data = recording.raw.get_data()
        standardised_recordings.append(
            (data - data.mean(1, keepdims=True)) / data.std(1, keepdims=True)
        )

    def response_strength(event_times: np.ndarray) -> float:
        """Mean |grand-average ERP| 50-350 ms after the events, across channels."""
        evoked_per_recording = []
        for data in standardised_recordings:
            event_samples = np.round(event_times * sfreq).astype(int)
            fits = (event_samples + lags[0] > 0) & (event_samples + lags[-1] < data.shape[1])
            segments = data[:, event_samples[fits, None] + lags[None, :]]
            baseline = segments[:, :, lags < 0].mean(2, keepdims=True)
            evoked_per_recording.append(np.clip(segments - baseline, -6, 6).mean(1))
        grand_average = np.mean(evoked_per_recording, axis=0)
        return float(np.abs(grand_average[:, in_response_window]).mean())

    rng = np.random.default_rng(seed)
    n_events = len(stimulus.onsets)
    random_times_strength = np.array(
        [response_strength(np.sort(rng.uniform(5, 298, n_events))) for _ in range(n_random_sets)]
    )
    onset_locked = np.array([response_strength(stimulus.onsets + shift) for shift in shifts])
    offset_locked = np.array([response_strength(stimulus.offsets + shift) for shift in shifts])
    return onset_locked, offset_locked, random_times_strength


def analyse_missing_words(data_dir: Path) -> dict:
    stimulus = load_missing_word_stimulus()
    recordings = load_missing_word_recordings(data_dir)
    epochs_per_recording = [
        epoch_missing_words_and_speech(recording, stimulus.onsets, stimulus.offsets, seed=index)
        for index, recording in enumerate(recordings)
    ]
    info, times = epochs_per_recording[0].info, epochs_per_recording[0].times

    # ERPs: recording x channel x time
    missing_erp = np.stack([e["missing"].average().get_data() for e in epochs_per_recording])
    speech_erp = np.stack([e["speech"].average().get_data() for e in epochs_per_recording])
    after_onset = times >= 0
    cluster = cluster_test_condition_difference(
        (missing_erp - speech_erp)[:, :, after_onset].transpose(0, 2, 1),
        info,
        n_permutations=2000,
    )
    significant_mask = np.zeros((len(times), len(info.ch_names)), bool)
    significant_mask[after_onset] = cluster.significant

    shifts = np.round(np.arange(-0.6, 1.21, 0.1), 2)
    onset_locked, offset_locked, random_times_strength = evoked_strength_for_shifted_events(
        recordings, stimulus, shifts
    )

    epoch_data = np.concatenate([e.get_data(copy=False) for e in epochs_per_recording])
    is_missing_word = np.concatenate(
        [(e.events[:, 2] == MISSING_WORD_CODE).astype(int) for e in epochs_per_recording]
    )
    recording_index = np.concatenate(
        [np.full(len(e), index) for index, e in enumerate(epochs_per_recording)]
    )
    leave_one_recording_out = LeaveOneGroupOut()
    auc_over_time = decode_each_time_point(
        epoch_data, is_missing_word, leave_one_recording_out, recording_index
    )
    amplitude_features = mean_amplitudes_in_time_bins(epoch_data, times, 0.0, 1.2, 0.1)
    classifier = make_erp_classifier(C=0.05)
    auc_per_recording = cross_val_score(
        classifier,
        amplitude_features,
        is_missing_word,
        groups=recording_index,
        cv=leave_one_recording_out,
        scoring="roc_auc",
    )
    _, auc_shuffled_labels, permutation_p = permutation_test_score(
        classifier,
        amplitude_features,
        is_missing_word,
        groups=recording_index,
        cv=leave_one_recording_out,
        scoring="roc_auc",
        n_permutations=N_LABEL_PERMUTATIONS,
        random_state=0,
        n_jobs=-1,
    )

    recording_table = pd.DataFrame(
        {
            "recording": [r.name for r in recordings],
            "bad_channels_interpolated": [" ".join(r.interpolated_channels) for r in recordings],
            "missing_epochs_kept": [len(e["missing"]) for e in epochs_per_recording],
            "speech_epochs_kept": [len(e["speech"]) for e in epochs_per_recording],
            "heldout_auc_0_1200ms": auc_per_recording.round(3),
        }
    )
    significant_times = times[significant_mask.any(1)]
    summary = {
        "n_recordings": len(recordings),
        "erp_cluster_p_min": float(cluster.cluster_p.min()),
        "erp_significant_from_s": float(significant_times.min()),
        "erp_significant_to_s": float(significant_times.max()),
        "decoding_peak_auc": float(auc_over_time.mean(0).max()),
        "decoding_peak_time_s": float(times[auc_over_time.mean(0).argmax()]),
        "window_model_auc": float(auc_per_recording.mean()),
        "window_model_p": float(permutation_p),
        "strongest_onset_shift_s": float(shifts[np.argmax(onset_locked)]),
    }
    return {
        "info": info,
        "times": times,
        "missing_erp": missing_erp,
        "speech_erp": speech_erp,
        "significant_mask": significant_mask,
        "shifts": shifts,
        "onset_locked_strength": onset_locked,
        "offset_locked_strength": offset_locked,
        "random_times_strength": random_times_strength,
        "auc_over_time": auc_over_time,
        "auc_above_chance": find_above_chance_times(auc_over_time),
        "auc_per_recording": auc_per_recording,
        "auc_shuffled_labels": auc_shuffled_labels,
        "permutation_p": permutation_p,
        "recording_table": recording_table,
        "summary": summary,
    }


# ---------------------------------------------------------------------------------------------
# 2. ERP CORE N400
# ---------------------------------------------------------------------------------------------
def find_local_n400_participants(n400_dir: Path = N400_DIR) -> list[int]:
    """Participants whose .set/.fdt/events files are all present (no downloading here)."""
    participants = []
    for participant_dir in sorted(n400_dir.glob("sub-*")):
        file_stem = participant_dir / "eeg" / f"{participant_dir.name}_task-N400"
        required_files = [
            Path(f"{file_stem}_eeg.set"),
            Path(f"{file_stem}_eeg.fdt"),
            Path(f"{file_stem}_events.tsv"),
        ]
        if all(path.exists() for path in required_files):
            participants.append(int(participant_dir.name.split("-")[1]))
    return participants


def analyse_n400(participants: Sequence[int]) -> dict:
    epochs_per_participant = [load_erpcore_n400_epochs(p, N400_DIR) for p in participants]
    info, times = epochs_per_participant[0].info, epochs_per_participant[0].times
    related_erp = np.stack([e["related"].average().get_data() for e in epochs_per_participant])
    unrelated_erp = np.stack([e["unrelated"].average().get_data() for e in epochs_per_participant])
    related_erp, unrelated_erp = related_erp * 1e6, unrelated_erp * 1e6  # volts -> µV
    after_onset = times >= 0
    cluster = cluster_test_condition_difference(
        (unrelated_erp - related_erp)[:, :, after_onset].transpose(0, 2, 1),
        info,
        n_permutations=2000,
    )
    significant_mask = np.zeros((len(times), len(info.ch_names)), bool)
    significant_mask[after_onset] = cluster.significant

    def is_unrelated(epochs: mne.Epochs) -> np.ndarray:
        return (epochs.events[:, 2] == epochs.event_id["unrelated"]).astype(int)

    within_person_auc = np.stack(
        [
            decode_each_time_point(
                e.get_data(copy=False),
                is_unrelated(e),
                StratifiedKFold(5, shuffle=True, random_state=0),
            ).mean(0)
            for e in epochs_per_participant
        ]
    )

    features_per_participant, labels_per_participant, participant_index = [], [], []
    for index, epochs in enumerate(epochs_per_participant):
        features = mean_amplitudes_in_time_bins(
            epochs.get_data(copy=False) * 1e6, times, 0.2, 0.6, 0.1
        )
        # z-score within each person so differences in overall amplitude do not dominate
        features_per_participant.append((features - features.mean(0)) / features.std(0))
        labels_per_participant.append(is_unrelated(epochs))
        participant_index.append(np.full(len(epochs), index))
    cross_person_auc = cross_val_score(
        make_erp_classifier(0.05),
        np.concatenate(features_per_participant),
        np.concatenate(labels_per_participant),
        groups=np.concatenate(participant_index),
        cv=LeaveOneGroupOut(),
        scoring="roc_auc",
    )

    cpz_index = info.ch_names.index("CPz")
    in_n400_window = (times >= 0.3) & (times < 0.5)
    n400_effect = (unrelated_erp - related_erp)[:, cpz_index][:, in_n400_window].mean(1)
    n400_test = stats.ttest_1samp(n400_effect, 0)
    summary = {
        "n_participants": len(participants),
        "trials_kept_per_participant": " ".join(str(len(e)) for e in epochs_per_participant),
        "n400_effect_uv_cpz_300_500": float(n400_effect.mean()),
        "n400_effect_t": float(n400_test.statistic),
        "n400_effect_p": float(n400_test.pvalue),
        "erp_cluster_p_min": float(cluster.cluster_p.min()),
        "within_peak_auc": float(within_person_auc.mean(0).max()),
        "within_peak_time_s": float(times[within_person_auc.mean(0).argmax()]),
        "cross_person_auc": float(cross_person_auc.mean()),
    }
    return {
        "info": info,
        "times": times,
        "related_erp": related_erp,
        "unrelated_erp": unrelated_erp,
        "significant_mask": significant_mask,
        "within_person_auc": within_person_auc,
        "within_person_above_chance": find_above_chance_times(within_person_auc),
        "cross_person_auc": cross_person_auc,
        "participants": list(participants),
        "summary": summary,
    }


# ---------------------------------------------------------------------------------------------
# 3. Kiloword
# ---------------------------------------------------------------------------------------------
def analyse_kiloword(n_label_permutations: int = 200) -> dict:
    epochs = load_kiloword_epochs()
    word_properties, times = epochs.metadata, epochs.times
    predictors = ["WordFrequency", "Concreteness", "NumberOfLetters"]
    design = word_properties[predictors].apply(
        lambda column: (column - column.mean()) / column.std()
    )
    design.insert(0, "Intercept", 1.0)
    regression = mne.stats.linear_regression(epochs, design, names=list(design.columns))
    t_values = {name: regression[name].t_val.data for name in predictors}
    bonferroni_t_threshold = stats.t.ppf(
        1 - 0.025 / len(times), df=len(word_properties) - design.shape[1]
    )

    amplitude_features = mean_amplitudes_in_time_bins(
        epochs.get_data(copy=False) * 1e6, times, 0.1, 0.7, 0.05
    )
    ridge_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
    ten_fold = KFold(10, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)
    predictions = {}
    for name in ("WordFrequency", "Concreteness"):
        actual = word_properties[name].to_numpy()
        predicted = cross_val_predict(ridge_model, amplitude_features, actual, cv=ten_fold)
        correlation = stats.pearsonr(actual, predicted).statistic
        shuffled_correlations = np.array(
            [
                stats.pearsonr(
                    shuffled,
                    cross_val_predict(ridge_model, amplitude_features, shuffled, cv=ten_fold),
                ).statistic
                for shuffled in (rng.permutation(actual) for _ in range(n_label_permutations))
            ]
        )
        p_value = (np.sum(shuffled_correlations >= correlation) + 1) / (n_label_permutations + 1)
        predictions[name] = {
            "actual": actual,
            "predicted": predicted,
            "r": correlation,
            "p": p_value,
        }

    summary = {f"{name}_prediction_r": p["r"] for name, p in predictions.items()} | {
        f"{name}_prediction_p": p["p"] for name, p in predictions.items()
    }
    return {
        "info": epochs.info,
        "times": times,
        "t_values": t_values,
        "bonferroni_t_threshold": bonferroni_t_threshold,
        "n_words": len(word_properties),
        "predictions": predictions,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/wordlists/MissingWords")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-n400", action="store_true")
    parser.add_argument("--skip-kiloword", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mne.set_log_level("ERROR")
    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {"missingwords": analyse_missing_words(args.data_dir)}
    results["missingwords"]["recording_table"].to_csv(
        args.out / "missingwords_recordings.csv", index=False
    )
    participants = find_local_n400_participants()
    if args.skip_n400 or len(participants) < 3:
        logger.info("Skipping N400: %d local participants (need >= 3)", len(participants))
    else:
        results["n400"] = analyse_n400(participants)
    if not args.skip_kiloword:
        results["kiloword"] = analyse_kiloword()

    for analysis_name, analysis_results in results.items():
        with open(cache_dir / f"{analysis_name}.pkl", "wb") as fh:
            pickle.dump(analysis_results, fh)
    summary_table = pd.DataFrame(
        [
            (analysis_name, metric, value)
            for analysis_name, analysis_results in results.items()
            for metric, value in analysis_results["summary"].items()
        ],
        columns=["analysis", "metric", "value"],
    )
    summary_table.to_csv(args.out / "summary.csv", index=False)
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
