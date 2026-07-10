import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from climadc.validation.leakage import LeakageGuard


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    legal_offsets=st.lists(
        st.integers(min_value=-86_400, max_value=0),
        min_size=1,
        max_size=20,
    ),
    illegal_offsets=st.lists(
        st.integers(min_value=1, max_value=86_400),
        min_size=1,
        max_size=20,
    ),
)
def test_safe_subset_never_crosses_decision_time(
    legal_offsets: list[int],
    illegal_offsets: list[int],
) -> None:
    decision_time = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    offsets = legal_offsets + illegal_offsets
    frame = pd.DataFrame(
        {
            "available_at": [decision_time + pd.Timedelta(seconds=offset) for offset in offsets],
            "value": list(range(len(offsets))),
        }
    )

    subset, audit = LeakageGuard().safe_subset(frame, decision_time)

    assert (subset["available_at"] <= decision_time).all()
    assert audit.accepted_rows + audit.rejected_rows == len(frame)
    assert len(subset) == audit.accepted_rows
    assert len(audit.violations) == audit.rejected_rows
