"""Load the 24-channel EDF+ exports in ``data/wordlists/MissingWords``.

What the files contain (checked, not assumed):

* 21 scalp channels in old 10-20 names (T3/T4/T5/T6), plus ``A1``, ``A2`` and ``Ref``.
* Physical range == digital range (-32768..32767) and a blank unit field, so values are raw
  16-bit ADC counts with an unknown gain. Only relative/unitless measures are meaningful.
* No event annotations apart from "Recording starts" / "Recording end".
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import mne

ADC_MIN, ADC_MAX = -32768, 32767
DEVICE_SFREQ = 250.0
NON_SCALP_CHANNELS = ("A1", "A2", "Ref")
# Old 10-20 names -> current 10-10 names used by MNE's template montages.
OLD_TO_NEW_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_FILENAME_TIME = re.compile(r"result_(\d{2}\.\d{2}\.\d{4})_(\d{2}\.\d{2}\.\d{2})")


def recording_start_from_filename(path: Path | str) -> datetime | None:
    """Session start time encoded in ``result_DD.MM.YYYY_HH.MM.SS.edf``.

    The EDF header's start date is the *export* time (all files say 3 April 2023), so the
    filename is the only record of when each session was actually recorded.
    """
    match = _FILENAME_TIME.search(Path(path).name)
    if match is None:
        return None
    return datetime.strptime(" ".join(match.groups()), "%d.%m.%Y %H.%M.%S")


def _template_montage() -> mne.channels.DigMontage:
    builtin = mne.channels.get_builtin_montages()
    name = "colin27_1020" if "colin27_1020" in builtin else "standard_1020"
    return mne.channels.make_standard_montage(name)


def load_edf_recording(path: Path | str, sfreq: float | None = DEVICE_SFREQ) -> mne.io.BaseRaw:
    """Read one file, mark non-scalp channels as ``misc``, rename to 10-10 and set positions.

    Parameters
    ----------
    sfreq
        True sampling rate. Some exports declare 249 Hz (304 records x 249 samples), yet hold
        75 696 samples where the 250-Hz files hold 75 750 for the same ~303 s: the amplifier ran
        at 250 Hz and only the record size differs. Keeping 249 Hz would shift events near the
        end of the recording by ~1.2 s. Pass ``None`` to keep the header value.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    if sfreq is not None and raw.info["sfreq"] != sfreq:
        info = mne.create_info(raw.ch_names, sfreq, "eeg")
        raw = mne.io.RawArray(raw.get_data(), info, verbose="error")
    raw.set_channel_types(
        {ch: "misc" for ch in NON_SCALP_CHANNELS if ch in raw.ch_names}, verbose="error"
    )
    raw.rename_channels({old: new for old, new in OLD_TO_NEW_NAMES.items() if old in raw.ch_names})
    raw.set_montage(_template_montage(), on_missing="ignore", verbose="error")
    return raw
