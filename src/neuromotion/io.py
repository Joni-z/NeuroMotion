"""Loading WAY-EEG-GAL.

The release ships three kinds of MATLAB file per participant:

  ``HS_P{p}_S{s}.mat``   continuous recording for one series, all modalities
  ``WS_P{p}_S{s}.mat``   the same data cut into per-trial windows
  ``P{p}_AllLifts.mat``  one row per trial, 43 derived behavioural measures

Series 1--9 are released in full. A tenth weight series per participant was
stripped of its non-EEG streams for the Kaggle competition and appears only as
``HS_P{p}_ST.mat``; ``AllLifts`` therefore has 294 rows, not 328.

Everything is MATLAB <= v7.2, so ``scipy.io.loadmat`` is enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# The 43 AllLifts columns, in file order. Kept here so callers index by name.
ALLLIFTS_COLUMNS = (
    "Part", "Run", "Lift", "CurW", "CurS", "PrevW", "PrevS", "StartTime",
    "LEDOn", "LEDOff", "BlockType", "tIndTouch", "tThumbTouch",
    "tFirstDigitTouch", "tBothDigitTouch", "tIndStartLoadPhase",
    "tThuStartLoadPhase", "tBothStartLoadPhase", "tLiftOff", "tReplace",
    "tIndRelease", "tThuRelease", "tBothReleased", "GF_Max", "LF_Max",
    "dGF_Max", "dLF_Max", "tGF_Max", "tLF_Max", "tdGF_Max", "tdLF_Max",
    "GF_Hold", "LF_Hold", "tHandStart", "tHandStop", "tPeakVelHandReach",
    "tPeakVelHandRetract", "GripAparture_Max", "tGripAparture_Max",
    "Dur_Reach", "Dur_Preload", "Dur_LoadPhase", "Dur_Release",
)

# FASTRAK sensor assignment, from the data paper: 1 object, 2 index, 3 thumb, 4 wrist.
SENSOR = {"object": 1, "index": 2, "thumb": 3, "wrist": 4}

# Event columns in AllLifts are timed from the start of a per-trial window that
# opens 2 s before the LED, which is why ``LEDOn`` is 2.000 in every row.
# ``StartTime`` is the absolute LED-on time within the series. So:
#
#     absolute_time = StartTime + t_event - WINDOW_PRE_LED_S
#
# Verified empirically on P1 S1: mean wrist speed over the 300 ms after
# ``StartTime + tHandStart - 2`` is 28x the 300 ms before it, whereas without
# the offset the ratio is 0.78 (i.e. no movement there).
WINDOW_PRE_LED_S = 2.0

# Position channels are in centimetres. Established against three known scales on
# P1 S1: reach displacement median 26.1 (a natural reach is 25-40 cm), peak reach
# speed median 75.4 (50-100 cm/s), object vertical excursion after lift-off 5.8
# ("a few centimetres" in the data paper).
CM_TO_MM = 10.0

# Despite its 't' prefix, ``tPeakVelHandReach`` is a speed, not a timestamp: its
# median of 73.4 matches the 75.4 cm/s we measure from the wrist trace, whereas a
# time relative to trial start would be ~2.5-3.5 s. Compute peak-velocity *timing*
# from the trace instead of reading this column.
SPEED_COLUMNS = ("tPeakVelHandReach", "tPeakVelHandRetract")


@dataclass(frozen=True)
class Series:
    """One continuous series: every modality on its own sampling grid."""

    participant: int
    series: int
    eeg: np.ndarray          # (n, 32) at 500 Hz
    eeg_names: tuple[str, ...]
    emg: np.ndarray          # (m, 5) at 4000 Hz
    emg_names: tuple[str, ...]
    kin: np.ndarray          # (n, 36) at 500 Hz -- 12 angle, 12 position, 12 force/torque
    kin_names: tuple[str, ...]
    env: np.ndarray          # (n, 2)  surface, weight
    env_names: tuple[str, ...]
    misc: np.ndarray         # (n, 6)  button, magnet, LEDs, temperatures
    misc_names: tuple[str, ...]
    fs_eeg: int
    fs_emg: int
    fs_kin: int

    def kin_channel(self, short: str) -> np.ndarray:
        """Column of ``kin`` by its short name, e.g. ``Px4`` or ``FZ1``."""
        return self.kin[:, self.kin_names.index(short)]

    def position(self, part: str = "wrist") -> np.ndarray:
        """(n, 3) position of one tracked point, in file units."""
        s = SENSOR[part]
        return np.column_stack([self.kin_channel(f"P{ax}{s}") for ax in "xyz"])

    @property
    def duration_s(self) -> float:
        return len(self.eeg) / self.fs_eeg


def _names(cell) -> tuple[str, ...]:
    return tuple(str(x).strip() for x in np.atleast_1d(cell).ravel())


def _short(name: str) -> str:
    """'Px4 - position x sensor 4' -> 'Px4'."""
    return name.split(" ")[0]


def load_series(path: str | Path) -> Series:
    hs = loadmat(str(path), squeeze_me=True)["hs"]
    g = {k: hs[k].item() for k in ("eeg", "emg", "kin", "env", "misc")}
    return Series(
        participant=int(hs["participant"].item()),
        series=int(hs["series"].item()),
        eeg=np.asarray(g["eeg"]["sig"].item()),
        eeg_names=_names(g["eeg"]["names"].item()),
        emg=np.asarray(g["emg"]["sig"].item()),
        emg_names=_names(g["emg"]["names"].item()),
        kin=np.asarray(g["kin"]["sig"].item()),
        kin_names=tuple(_short(n) for n in _names(g["kin"]["names"].item())),
        env=np.asarray(g["env"]["sig"].item()),
        env_names=_names(g["env"]["names"].item()),
        misc=np.asarray(g["misc"]["sig"].item()),
        misc_names=_names(g["misc"]["names"].item()),
        fs_eeg=int(g["eeg"]["samplingrate"].item()),
        fs_emg=int(g["emg"]["samplingrate"].item()),
        fs_kin=int(g["kin"]["samplingrate"].item()),
    )


def load_alllifts(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """(n_trials, 43) table plus its column names."""
    P = loadmat(str(path), squeeze_me=True)["P"]
    table = np.asarray(P["AllLifts"].item(), dtype=float)
    cols = _names(P["ColNames"].item())
    return table, cols


def alllifts_column(table: np.ndarray, cols: tuple[str, ...], name: str) -> np.ndarray:
    return table[:, cols.index(name)]


def absolute_event_time(table: np.ndarray, cols: tuple[str, ...], event: str) -> np.ndarray:
    """Absolute time of an event within its series, in seconds.

    See ``WINDOW_PRE_LED_S``: event columns are relative to a window opening 2 s
    before the LED, and ``StartTime`` is the absolute LED-on time.
    """
    return (alllifts_column(table, cols, "StartTime")
            + alllifts_column(table, cols, event)
            - WINDOW_PRE_LED_S)


def participants(root: str | Path) -> list[int]:
    """Participants whose AllLifts file is present."""
    root = Path(root)
    found = {int(m.group(1))
             for p in root.glob("P*_AllLifts.mat")
             if (m := re.fullmatch(r"P(\d+)_AllLifts", p.stem))}
    return sorted(found)


def series_paths(root: str | Path, participant: int, released_only: bool = True) -> list[Path]:
    """HS files for one participant, series order.

    ``released_only`` drops ``HS_P*_ST.mat``, the series whose non-EEG streams
    were withheld; it has no kinematics and no AllLifts rows.
    """
    root = Path(root)
    numbered = [(int(m.group(1)), p)
                for p in root.glob(f"HS_P{participant}_S*.mat")
                if (m := re.search(r"_S(\d+)$", p.stem))]
    paths = [p for _, p in sorted(numbered)]
    if not released_only:
        st = root / f"HS_P{participant}_ST.mat"
        if st.exists():
            paths.append(st)
    return paths
