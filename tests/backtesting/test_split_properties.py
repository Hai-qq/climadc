import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from climadc.backtesting.splits import blocked_split, rolling_origin_splits


@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    offsets=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=5,
        max_size=40,
        unique=True,
    )
)
def test_splits_preserve_generated_strictly_increasing_utc_indexes(
    offsets: list[int],
) -> None:
    ordered_offsets = sorted(offsets)
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(seconds=offset)
            for offset in ordered_offsets
        ]
    )

    blocked = blocked_split(index, train_fraction=0.5, calibration_fraction=0.25)
    np.testing.assert_array_equal(
        np.concatenate([blocked.train, blocked.calibration, blocked.test]),
        np.arange(len(index)),
    )
    assert blocked.train_end == index[blocked.train[-1]]
    assert blocked.calibration_end == index[blocked.calibration[-1]]
    assert blocked.test_end == index[blocked.test[-1]]

    rolling = rolling_origin_splits(
        index,
        min_train=2,
        calibration_size=1,
        test_size=1,
        step=1,
    )
    assert len(rolling) == len(index) - 3
    for expected_train_size, split in enumerate(rolling, start=2):
        np.testing.assert_array_equal(split.train, np.arange(expected_train_size))
        assert split.train[-1] < split.calibration[0] < split.test[0]
        assert split.test_end == index[split.test[-1]]
