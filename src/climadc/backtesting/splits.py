from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    calibration_end: pd.Timestamp
    test_end: pd.Timestamp


def _validate_index(index: object) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    if index.empty:
        raise ValueError("index must be non-empty")
    if index.hasnans:
        raise ValueError("index must not contain NaT")
    if (
        index.tz is None
        or str(index.tz) != "UTC"
        or any(timestamp.utcoffset() != pd.Timedelta(0) for timestamp in index)
    ):
        raise ValueError("index must contain timezone-aware exact UTC timestamps")
    if not index.is_unique:
        raise ValueError("index must contain unique timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing")
    return index


def _validate_positions(name: str, positions: np.ndarray, index_size: int) -> None:
    if not isinstance(positions, np.ndarray) or positions.ndim != 1:
        raise TypeError(f"{name} positions must be a one-dimensional NumPy array")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError(f"{name} positions must have an integer dtype")
    if positions.size == 0:
        raise ValueError(f"{name} positions must be non-empty")
    if np.any(positions < 0) or np.any(positions >= index_size):
        raise ValueError(f"{name} positions must be in range")
    if np.any(np.diff(positions) <= 0):
        raise ValueError(f"{name} positions must be strictly increasing and unique")


def _validate_temporal_split(index: pd.DatetimeIndex, split: TemporalSplit) -> None:
    checked_index = _validate_index(index)
    partitions = {
        "train": split.train,
        "calibration": split.calibration,
        "test": split.test,
    }
    for name, positions in partitions.items():
        _validate_positions(name, positions, len(checked_index))

    if any(
        np.intersect1d(partitions[left], partitions[right]).size
        for left, right in (
            ("train", "calibration"),
            ("train", "test"),
            ("calibration", "test"),
        )
    ):
        raise ValueError("temporal split partitions must be pairwise disjoint")
    if not (split.train[-1] < split.calibration[0] and split.calibration[-1] < split.test[0]):
        raise ValueError("temporal split partitions must be strictly ordered")

    expected_boundaries = (
        checked_index[split.train[-1]],
        checked_index[split.calibration[-1]],
        checked_index[split.test[-1]],
    )
    actual_boundaries = (split.train_end, split.calibration_end, split.test_end)
    if actual_boundaries != expected_boundaries:
        raise ValueError("temporal split boundaries must match partition end timestamps")


def _make_split(
    index: pd.DatetimeIndex,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> TemporalSplit:
    split = TemporalSplit(
        train=train,
        calibration=calibration,
        test=test,
        train_end=index[train[-1]],
        calibration_end=index[calibration[-1]],
        test_end=index[test[-1]],
    )
    _validate_temporal_split(index, split)
    return split


def _require_fraction(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} fraction must be a real number")
    fraction = float(value)
    if not isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(f"{name} fraction must be strictly between 0 and 1")
    return fraction


def blocked_split(
    index: pd.DatetimeIndex,
    train_fraction: float,
    calibration_fraction: float,
) -> TemporalSplit:
    checked_index = _validate_index(index)
    checked_train = _require_fraction("train", train_fraction)
    checked_calibration = _require_fraction("calibration", calibration_fraction)
    if checked_train + checked_calibration >= 1:
        raise ValueError("train and calibration fraction sum must be strictly less than 1")

    train_size = int(len(checked_index) * checked_train)
    calibration_size = int(len(checked_index) * checked_calibration)
    test_size = len(checked_index) - train_size - calibration_size
    if min(train_size, calibration_size, test_size) < 1:
        raise ValueError("blocked split fractions produced an empty partition")

    calibration_end = train_size + calibration_size
    return _make_split(
        checked_index,
        train=np.arange(0, train_size, dtype=np.int64),
        calibration=np.arange(train_size, calibration_end, dtype=np.int64),
        test=np.arange(calibration_end, len(checked_index), dtype=np.int64),
    )


def _require_rolling_parameter(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < minimum:
        if name == "min_train":
            raise ValueError("min_train must be at least 2")
        raise ValueError(f"{name} must be a positive integer")
    return value


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    min_train: int,
    calibration_size: int,
    test_size: int,
    step: int,
) -> list[TemporalSplit]:
    checked_index = _validate_index(index)
    checked_min_train = _require_rolling_parameter("min_train", min_train, 2)
    checked_calibration = _require_rolling_parameter("calibration_size", calibration_size, 1)
    checked_test = _require_rolling_parameter("test_size", test_size, 1)
    checked_step = _require_rolling_parameter("step", step, 1)

    splits: list[TemporalSplit] = []
    train_size = checked_min_train
    while train_size + checked_calibration + checked_test <= len(checked_index):
        calibration_end = train_size + checked_calibration
        test_end = calibration_end + checked_test
        splits.append(
            _make_split(
                checked_index,
                train=np.arange(0, train_size, dtype=np.int64),
                calibration=np.arange(train_size, calibration_end, dtype=np.int64),
                test=np.arange(calibration_end, test_end, dtype=np.int64),
            )
        )
        train_size += checked_step
    return splits
