from dataclasses import replace
from datetime import timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from climadc.backtesting.splits import (
    TemporalSplit,
    _validate_temporal_split,
    blocked_split,
    rolling_origin_splits,
)


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")


def test_blocked_split_is_ordered_and_disjoint(hourly_index: pd.DatetimeIndex) -> None:
    split = blocked_split(hourly_index, train_fraction=0.6, calibration_fraction=0.2)

    assert split.train.max() < split.calibration.min()
    assert split.calibration.max() < split.test.min()
    assert not (set(split.train) & set(split.test))


def test_blocked_split_has_exact_positions_and_utc_boundaries(
    hourly_index: pd.DatetimeIndex,
) -> None:
    split = blocked_split(hourly_index, train_fraction=0.6, calibration_fraction=0.2)

    np.testing.assert_array_equal(split.train, np.arange(0, 6))
    np.testing.assert_array_equal(split.calibration, np.arange(6, 8))
    np.testing.assert_array_equal(split.test, np.arange(8, 10))
    assert split.train.dtype.kind == "i"
    assert split.calibration.dtype.kind == "i"
    assert split.test.dtype.kind == "i"
    assert split.train_end == hourly_index[5]
    assert split.calibration_end == hourly_index[7]
    assert split.test_end == hourly_index[9]


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([0, 1, 2]),
        pd.DatetimeIndex([]),
        pd.date_range("2026-01-01", periods=5, freq="h"),
        pd.date_range("2026-01-01", periods=5, freq="h", tz="Asia/Shanghai"),
        pd.date_range(
            "2026-01-01",
            periods=5,
            freq="h",
            tz=timezone(timedelta(hours=1), name="UTC"),
        ),
        pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.NaT,
                pd.Timestamp("2026-01-01 02:00", tz="UTC"),
            ]
        ),
        pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 02:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
            ]
        ),
        pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
            ]
        ),
    ],
)
def test_temporal_split_rejects_invalid_indexes(index: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="index"):
        blocked_split(index, train_fraction=0.6, calibration_fraction=0.2)


@pytest.mark.parametrize(
    ("train_fraction", "calibration_fraction"),
    [
        (0.0, 0.2),
        (1.0, 0.2),
        (-0.1, 0.2),
        (0.6, 0.0),
        (0.6, 1.0),
        (0.8, 0.2),
        (0.8, 0.3),
        (float("nan"), 0.2),
        ("0.6", 0.2),
        (True, 0.2),
    ],
)
def test_blocked_split_rejects_invalid_fractions(
    hourly_index: pd.DatetimeIndex,
    train_fraction: Any,
    calibration_fraction: Any,
) -> None:
    with pytest.raises((TypeError, ValueError), match="fraction"):
        blocked_split(hourly_index, train_fraction, calibration_fraction)


def test_blocked_split_rejects_fraction_sizes_that_empty_a_partition() -> None:
    short_index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="empty partition"):
        blocked_split(short_index, train_fraction=0.2, calibration_fraction=0.2)


def test_rolling_splits_expand_training_with_immediate_disjoint_blocks() -> None:
    index = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")

    splits = rolling_origin_splits(
        index,
        min_train=4,
        calibration_size=2,
        test_size=2,
        step=3,
    )

    assert len(splits) == 2
    np.testing.assert_array_equal(splits[0].train, np.arange(0, 4))
    np.testing.assert_array_equal(splits[0].calibration, np.arange(4, 6))
    np.testing.assert_array_equal(splits[0].test, np.arange(6, 8))
    np.testing.assert_array_equal(splits[1].train, np.arange(0, 7))
    np.testing.assert_array_equal(splits[1].calibration, np.arange(7, 9))
    np.testing.assert_array_equal(splits[1].test, np.arange(9, 11))
    assert splits[0].train_end == index[3]
    assert splits[0].calibration_end == index[5]
    assert splits[0].test_end == index[7]
    assert splits[1].train_end == index[6]
    assert splits[1].calibration_end == index[8]
    assert splits[1].test_end == index[10]


@pytest.mark.parametrize(
    ("min_train", "calibration_size", "test_size", "step"),
    [
        (1, 1, 1, 1),
        (2, 0, 1, 1),
        (2, 1, 0, 1),
        (2, 1, 1, 0),
        (2.0, 1, 1, 1),
        (True, 1, 1, 1),
    ],
)
def test_rolling_splits_reject_invalid_parameters(
    hourly_index: pd.DatetimeIndex,
    min_train: Any,
    calibration_size: Any,
    test_size: Any,
    step: Any,
) -> None:
    with pytest.raises((TypeError, ValueError), match="min_train|positive integer"):
        rolling_origin_splits(
            hourly_index,
            min_train=min_train,
            calibration_size=calibration_size,
            test_size=test_size,
            step=step,
        )


def test_rolling_splits_return_empty_when_valid_data_is_insufficient() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")

    assert (
        rolling_origin_splits(
            index,
            min_train=3,
            calibration_size=2,
            test_size=2,
            step=1,
        )
        == []
    )


def test_validate_temporal_split_rejects_invalid_positions_and_boundaries(
    hourly_index: pd.DatetimeIndex,
) -> None:
    valid = blocked_split(hourly_index, train_fraction=0.6, calibration_fraction=0.2)
    invalid_splits = [
        replace(valid, train=np.array([], dtype=int)),
        replace(valid, train=np.array([0.0, 1.0])),
        replace(valid, train=np.array([0, 2, 1])),
        replace(valid, train=np.array([0, 1, 1])),
        replace(valid, test=np.array([8, 10])),
        replace(valid, calibration=np.array([5, 6, 7])),
        TemporalSplit(
            train=np.array([0, 2]),
            calibration=np.array([1]),
            test=np.array([3]),
            train_end=hourly_index[2],
            calibration_end=hourly_index[1],
            test_end=hourly_index[3],
        ),
        TemporalSplit(
            train=np.array([0]),
            calibration=np.array([1, 3]),
            test=np.array([2, 4]),
            train_end=hourly_index[0],
            calibration_end=hourly_index[3],
            test_end=hourly_index[4],
        ),
        replace(valid, train_end=hourly_index[4]),
    ]

    for invalid in invalid_splits:
        with pytest.raises((TypeError, ValueError)):
            _validate_temporal_split(hourly_index, invalid)
