from typing import Any

import pandas as pd
import pytest

from climadc.errors import LeakageError
from climadc.validation.leakage import LeakageAudit, LeakageGuard


DECISION_TIME = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def test_guard_rejects_rows_published_after_decision() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 00:01", tz="UTC"),
            ],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(LeakageError, match="1 row"):
        LeakageGuard().require_safe(frame, DECISION_TIME)


def test_equality_at_decision_time_is_safe_and_returns_deep_copy() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                pd.Timestamp("2025-12-31 23:59", tz="UTC"),
                DECISION_TIME,
            ],
            "value": [1.0, 2.0],
        }
    )
    original = frame.copy(deep=True)

    audit = LeakageGuard().audit(frame, DECISION_TIME)
    safe = LeakageGuard().require_safe(frame, DECISION_TIME)
    safe.loc[0, "value"] = 99.0

    assert audit == LeakageAudit(DECISION_TIME, 2, 0, ())
    assert safe is not frame
    pd.testing.assert_frame_equal(frame, original)


def test_guard_reports_plural_rejected_rows_without_dropping() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                pd.Timestamp("2026-01-01 00:01", tz="UTC"),
                pd.Timestamp("2026-01-01 00:02", tz="UTC"),
            ]
        }
    )

    with pytest.raises(LeakageError, match="2 rows unavailable at decision time"):
        LeakageGuard().require_safe(frame, DECISION_TIME)

    assert len(frame) == 2


def test_audit_preserves_duplicate_index_strings_and_violation_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                DECISION_TIME,
                pd.Timestamp("2026-01-01 00:01", tz="UTC"),
                pd.Timestamp("2026-01-01 00:02", tz="UTC"),
            ]
        },
        index=["safe", 7, 7],
    )
    original = frame.copy(deep=True)

    audit = LeakageGuard().audit(frame, DECISION_TIME)

    assert audit.accepted_rows == 1
    assert audit.rejected_rows == 2
    assert audit.violations == (
        {"index": "7", "available_at": "2026-01-01T00:01:00+00:00"},
        {"index": "7", "available_at": "2026-01-01T00:02:00+00:00"},
    )
    pd.testing.assert_frame_equal(frame, original)


def test_available_at_accepts_aware_non_utc_timestamps_as_absolute_instants() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                pd.Timestamp("2026-01-01 08:00", tz="Asia/Shanghai"),
                pd.Timestamp("2026-01-01 08:01", tz="Asia/Shanghai"),
            ]
        }
    )

    audit = LeakageGuard().audit(frame, DECISION_TIME)

    assert audit.accepted_rows == 1
    assert audit.rejected_rows == 1
    assert audit.violations == ({"index": "1", "available_at": "2026-01-01T08:01:00+08:00"},)


def test_safe_subset_returns_deep_copied_legal_rows_and_matching_audit() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                pd.Timestamp("2025-12-31 23:59", tz="UTC"),
                DECISION_TIME,
                pd.Timestamp("2026-01-01 00:01", tz="UTC"),
            ],
            "value": [1.0, 2.0, 3.0],
        },
        index=[3, 4, 5],
    )
    original = frame.copy(deep=True)
    guard = LeakageGuard()

    subset, audit = guard.safe_subset(frame, DECISION_TIME)

    assert subset.index.tolist() == [3, 4]
    assert audit == guard.audit(frame, DECISION_TIME)
    assert len(subset) == audit.accepted_rows
    subset.loc[3, "value"] = 99.0
    pd.testing.assert_frame_equal(frame, original)


def test_empty_frame_is_safe() -> None:
    frame = pd.DataFrame(
        {
            "available_at": pd.Series([], dtype="datetime64[ns, UTC]"),
            "value": pd.Series([], dtype=float),
        }
    )
    guard = LeakageGuard()

    audit = guard.audit(frame, DECISION_TIME)
    required = guard.require_safe(frame, DECISION_TIME)
    subset, subset_audit = guard.safe_subset(frame, DECISION_TIME)

    assert audit == LeakageAudit(DECISION_TIME, 0, 0, ())
    assert subset_audit == audit
    assert required.empty and required is not frame
    assert subset.empty and subset is not frame


@pytest.mark.parametrize(
    "decision_time",
    [
        pd.Timestamp("2026-01-01 00:00"),
        pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai"),
        pd.NaT,
        "2026-01-01T00:00:00Z",
        [DECISION_TIME],
    ],
)
def test_guard_rejects_invalid_decision_time(decision_time: Any) -> None:
    frame = pd.DataFrame({"available_at": [DECISION_TIME]})

    with pytest.raises(LeakageError, match="decision_time.*UTC pandas Timestamp"):
        LeakageGuard().audit(frame, decision_time)


@pytest.mark.parametrize(
    "available_at",
    [
        pd.NaT,
        None,
        pd.Timestamp("2026-01-01 00:00"),
        "2026-01-01T00:00:00Z",
        [DECISION_TIME],
    ],
)
def test_guard_rejects_invalid_available_at_values(available_at: Any) -> None:
    frame = pd.DataFrame({"available_at": [available_at]})

    with pytest.raises(LeakageError, match="available_at.*timezone-aware"):
        LeakageGuard().audit(frame, DECISION_TIME)


def test_guard_requires_available_at_column() -> None:
    frame = pd.DataFrame({"value": [1.0]})

    with pytest.raises(LeakageError, match="available_at column is required"):
        LeakageGuard().audit(frame, DECISION_TIME)
