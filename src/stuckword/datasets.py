"""Registry of public datasets relevant to StuckWord, plus download helpers.

No public dataset labels trials by retrieval *difficulty*, which is why the project records its
own. These are used to validate the pipeline, estimate artefacts and borrow priors.
See ``docs/DATASETS.md`` for the reasoning behind each entry.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/external")


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    name: str
    url: str
    modality: str
    use_for: str
    openneuro_id: str | None = None


DATASETS: dict[str, DatasetInfo] = {
    info.key: info
    for info in (
        DatasetInfo(
            key="eegbci",
            name="PhysioNet EEG Motor Movement/Imagery (baseline runs)",
            url="https://physionet.org/content/eegmmidb/1.0.0/",
            modality="EEG 64 ch, 160 Hz, 109 subjects",
            use_for="Day 1-2 sanity: eyes-open vs eyes-closed alpha (via mne.datasets.eegbci)",
        ),
        DatasetInfo(
            key="erpcore-n400",
            name="ERP CORE - N400 word-pair task (Kappenman et al., 2021)",
            url="https://osf.io/29xpq/",
            modality="EEG 30 ch, 40 subjects, BIDS",
            use_for="Validate epoching/features on a well-known lexical-semantic effect",
        ),
        DatasetInfo(
            key="pavlov-digit-span",
            name="EEG + pupillometry in the digit span task (Pavlov et al., 2022)",
            url="https://openneuro.org/datasets/ds003838",
            modality="EEG 64 ch, ~65-86 subjects, BIDS",
            use_for="Frontal theta vs memory load: prior for the 'effort' signature",
            openneuro_id="ds003838",
        ),
        DatasetInfo(
            key="inner-speech",
            name="'Thinking out loud' inner speech dataset (Nieto et al., 2022)",
            url="https://openneuro.org/datasets/ds003626",
            modality="EEG 128+8 ch, 10 subjects, inner / pronounced / visualised",
            use_for="Pronounced vs inner speech: size and spread of speech EMG artefact",
            openneuro_id="ds003626",
        ),
        DatasetInfo(
            key="rennes-naming",
            name="HD-EEG picture naming & spelling (Mheich et al., 2021)",
            url="https://doi.org/10.1038/s41597-021-00821-1",
            modality="EEG 256 ch, 23 subjects (naming), BIDS on OpenNeuro",
            use_for="Overt naming with speech-onset times: closest analogue of a retrieval window",
        ),
        DatasetInfo(
            key="kara-one",
            name="KaraOne imagined & spoken phonemes/words (Zhao & Rudzicz, 2015)",
            url="http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html",
            modality="EEG 64 ch, 14 subjects",
            use_for="Imagined vs spoken speech benchmark for existing decoding code",
        ),
        DatasetInfo(
            key="anomia-tacs",
            name="Post-stroke anomia, picture naming during tACS program (Martzoukou et al.)",
            url="https://openneuro.org/datasets/ds007315",
            modality="EEG 32 ch (Starstim), 2 patients, longitudinal, BIDS",
            use_for="Rehab direction: what naming EEG looks like in chronic anomia",
            openneuro_id="ds007315",
        ),
    )
}


def download_openneuro(
    key: str,
    target_dir: Path = DEFAULT_DATA_DIR,
    include: Sequence[str] | None = None,
) -> Path:
    """Download (part of) an OpenNeuro dataset with ``openneuro-py``.

    Use ``include`` to avoid pulling tens of GB, e.g. ``include=["sub-01/*"]``.
    """
    info = DATASETS[key]
    if info.openneuro_id is None:
        raise ValueError(f"{key!r} is not on OpenNeuro via this helper; download from {info.url}")
    try:
        import openneuro
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'datasets' extra: pip install -e '.[datasets]'") from err

    dataset_dir = Path(target_dir) / info.openneuro_id
    logger.info("Downloading %s to %s", info.openneuro_id, dataset_dir)
    openneuro.download(
        dataset=info.openneuro_id,
        target_dir=dataset_dir,
        include=list(include) if include else None,
    )
    return dataset_dir


# ERP CORE N400, BIDS-compatible raw data folder on OSF (project 29xpq).
_ERPCORE_N400_OSF_FOLDER = (
    "https://api.osf.io/v2/nodes/29xpq/files/osfstorage/6007857286541a092614c5d3/"
)


def _osf_list(url: str) -> list[dict]:
    import requests

    items: list[dict] = []
    while url:
        page = requests.get(url, timeout=60)
        page.raise_for_status()
        body = page.json()
        items += body["data"]
        url = body["links"].get("next")
    return items


def download_erpcore_n400(
    target_dir: Path = DEFAULT_DATA_DIR / "erpcore_n400",
    subjects: Sequence[int] = range(1, 13),
) -> Path:
    """Download BIDS EEG files of selected ERP CORE N400 subjects (~70 MB each) from OSF.

    Files that already exist with the right size are skipped, so the call is resumable.
    """
    import requests

    target_dir = Path(target_dir)
    wanted = {f"sub-{s:03d}" for s in subjects}
    for sub in _osf_list(_ERPCORE_N400_OSF_FOLDER):
        name = sub["attributes"]["name"]
        if name not in wanted:
            continue
        for eeg_dir in _osf_list(sub["relationships"]["files"]["links"]["related"]["href"]):
            for item in _osf_list(eeg_dir["relationships"]["files"]["links"]["related"]["href"]):
                attrs = item["attributes"]
                destination = target_dir / name / "eeg" / attrs["name"]
                if destination.exists() and destination.stat().st_size == attrs["size"]:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading %s (%.1f MB)", attrs["name"], attrs["size"] / 1e6)
                with requests.get(item["links"]["download"], stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    partial_file = destination.with_suffix(destination.suffix + ".part")
                    with open(partial_file, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    partial_file.rename(destination)
    return target_dir


# ERP CORE N400 target-word event codes (Kappenman et al., 2021): 2xy = target,
# x = 1 related / 2 unrelated to the prime, y = stimulus list.
N400_TARGET_CODES = {"related": (211, 212), "unrelated": (221, 222)}


def load_erpcore_n400_epochs(
    subject: int,
    root: Path = DEFAULT_DATA_DIR / "erpcore_n400",
    *,
    tmin: float = -0.2,
    tmax: float = 0.8,
    sfreq: float = 128.0,
    reject_uv: float = 150.0,
):
    """Target-word epochs, related vs unrelated, preprocessed as in ERP CORE.

    0.1-30 Hz band-pass, re-referenced to the average of P9/P10 (mastoid-like sites used by
    ERP CORE), EOG regressed out, -200..0 ms baseline, +-``reject_uv`` peak-to-peak rejection,
    decimated to ``sfreq``.
    """
    import mne
    import pandas as pd

    eeg_dir = Path(root) / f"sub-{subject:03d}" / "eeg"
    stem = f"sub-{subject:03d}_task-N400"
    raw = mne.io.read_raw_eeglab(eeg_dir / f"{stem}_eeg.set", preload=True, verbose="error")
    raw.rename_channels({"FP1": "Fp1", "FP2": "Fp2"})
    raw.set_channel_types(
        {ch: "eog" for ch in raw.ch_names if "EOG" in ch.upper()}, verbose="error"
    )
    raw.set_montage("standard_1020", on_missing="ignore", verbose="error")
    raw.filter(0.1, 30.0, picks=["eeg", "eog"], verbose="error")
    raw.set_eeg_reference(["P9", "P10"], verbose="error")
    # Remove blink/eye-movement activity by regressing the EOG channels out of the EEG
    # (ERP CORE corrects these with ICA); otherwise rejection discards most trials of some people.
    raw = mne.set_bipolar_reference(
        raw,
        ["HEOG_left", "VEOG_lower"],
        ["HEOG_right", "Fp1"],
        ["HEOG", "VEOG"],
        drop_refs=False,
        verbose="error",
    ).drop_channels(["HEOG_left", "HEOG_right", "VEOG_lower"])
    raw.set_channel_types({"HEOG": "eog", "VEOG": "eog"}, verbose="error")
    eog = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(raw)
    raw = eog.apply(raw)

    events_tsv = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")
    codes = {code: i + 1 for i, codes_ in enumerate(N400_TARGET_CODES.values()) for code in codes_}
    targets = events_tsv[events_tsv["value"].isin(codes)]
    events = np.column_stack(
        [targets["sample"].to_numpy(), np.zeros(len(targets), int), targets["value"].map(codes)]
    )
    decim = max(1, round(raw.info["sfreq"] / sfreq))
    return mne.Epochs(
        raw,
        events,
        {name: i + 1 for i, name in enumerate(N400_TARGET_CODES)},
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        reject={"eeg": reject_uv * 1e-6},
        decim=decim,
        picks="eeg",
        preload=True,
        verbose="error",
    )


def load_kiloword_epochs():
    """MNE's Kiloword data: ERPs to 960 English words averaged over 75 readers
    (Dufau et al., 2015), with word properties in ``epochs.metadata``. ~25 MB download."""
    import mne

    path = mne.datasets.kiloword.data_path(verbose="error")
    return mne.read_epochs(Path(path) / "kword_metadata-epo.fif", verbose="error")
