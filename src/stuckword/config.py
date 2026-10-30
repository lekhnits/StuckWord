"""Shared constants and configuration objects.

Everything that is a *choice* (filter edges, window lengths, montage) lives here so that
analysis scripts and the live loop use exactly the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Placeholder 10-20 layout for the 16 Octopus channels. The board itself has no fixed montage:
# edit this to match the actual electrode placement (order = ADC channel order).
OCTOPUS_16_MONTAGE: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3",
    "Cz", "C4", "P3", "Pz", "P4", "O1", "Oz", "O2",
)  # fmt: skip

FRONTAL_CHANNELS: tuple[str, ...] = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")
OCCIPITAL_CHANNELS: tuple[str, ...] = ("O1", "Oz", "O2")

# PiEEG server streams 250 Hz; change if the firmware runs the ADS131M08 faster.
DEFAULT_SFREQ: float = 250.0


@dataclass(frozen=True)
class PreprocessingConfig:
    """Offline filtering and referencing."""

    l_freq: float = 1.0
    h_freq: float = 45.0
    notch_freqs: tuple[float, ...] = (50.0,)  # mains frequency in Europe; use 60.0 in the US
    reference: Literal["average"] | tuple[str, ...] = "average"
    resample_sfreq: float | None = None


@dataclass(frozen=True)
class EpochConfig:
    """Which part of each trial is analysed.

    ``lock="retrieved"`` with ``tmin=-2, tmax=0`` takes the 2 s *before* the participant signals
    that they found the word (or gave up) - i.e. before any speech, so jaw/tongue EMG is excluded.
    ``lock="cue"`` takes a window after the word appears; this is what a live detector sees,
    because it cannot know when retrieval will end.
    """

    lock: Literal["cue", "retrieved"] = "retrieved"
    tmin: float = -2.0
    tmax: float = 0.0
    reject_peak_to_peak_uv: float | None = 150.0
