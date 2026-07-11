from typing import Any

import pandas as pd
import pytest

from climadc.errors import LeakageError
from climadc.validation.leakage import LeakageAudit, LeakageGuard


DECISION_TIME = pd.Timestamp("2026-01-01 00:00", tz="UTC")


class _MutableHashableCategory:
    def __init__(self, label: str) -> None:
        self.label = label
        self.payload = ["source"]

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MutableHashableCategory) and self.label == other.label


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


def test_require_safe_recursively_copies_nested_object_payloads() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [DECISION_TIME],
            "payload": [{"events": [{"name": "original"}]}],
        },
        index=["legal"],
    )

    safe = LeakageGuard().require_safe(frame, DECISION_TIME)
    safe.at["legal", "payload"]["events"][0]["name"] = "mutated"

    assert frame.at["legal", "payload"] == {"events": [{"name": "original"}]}
    assert safe.index is not frame.index
    assert safe["payload"].dtype == frame["payload"].dtype


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


def test_mixed_aware_timezones_compare_as_absolute_instants() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                DECISION_TIME,
                pd.Timestamp("2026-01-01 08:01", tz="Asia/Shanghai"),
                pd.Timestamp("2025-12-31 18:59", tz="America/New_York"),
            ]
        },
        index=["utc", "shanghai", "new_york"],
    )

    subset, audit = LeakageGuard().safe_subset(frame, DECISION_TIME)

    assert frame["available_at"].dtype == object
    assert audit.accepted_rows == 2
    assert audit.rejected_rows == 1
    assert subset.index.tolist() == ["utc", "new_york"]


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


def test_safe_subset_recursively_copies_nested_object_payloads() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                DECISION_TIME,
                pd.Timestamp("2026-01-01 00:01", tz="UTC"),
            ],
            "payload": [
                {"labels": ["original"], "metadata": {"count": 1}},
                {"labels": ["future"], "metadata": {"count": 2}},
            ],
        },
        index=["legal", "future"],
    )

    subset, _ = LeakageGuard().safe_subset(frame, DECISION_TIME)
    subset.at["legal", "payload"]["labels"].append("mutated")
    subset.at["legal", "payload"]["metadata"]["count"] = 99

    assert frame.at["legal", "payload"] == {
        "labels": ["original"],
        "metadata": {"count": 1},
    }
    assert subset.index is not frame.index
    assert subset["payload"].dtype == frame["payload"].dtype


def test_safe_subset_isolates_categorical_payloads_with_duplicate_axes() -> None:
    category = _MutableHashableCategory("shared")
    index = pd.Index(["legal", "legal"], name="sample")
    available_at = pd.Series([DECISION_TIME, DECISION_TIME], index=index, name="available_at")
    payload = pd.Series(
        pd.Categorical(
            [category, category],
            categories=[category],
            ordered=True,
        ),
        index=index,
        name="payload",
    )
    frame = pd.concat([available_at, payload, payload.copy(deep=True)], axis=1)

    subset, _ = LeakageGuard().safe_subset(frame, DECISION_TIME)

    assert subset.index.tolist() == ["legal", "legal"]
    assert subset.index.name == "sample"
    assert subset.columns.tolist() == ["available_at", "payload", "payload"]
    for position in (1, 2):
        assert subset.dtypes.iloc[position] == frame.dtypes.iloc[position]
        assert subset.iloc[:, position].cat.ordered
        assert subset.iloc[:, position].cat.codes.tolist() == [0, 0]
    subset.iloc[0, 1].payload.append("left")
    subset.iloc[0, 2].payload.append("right")

    assert frame.iloc[0, 1].payload == ["source"]
    assert frame.iloc[0, 2].payload == ["source"]
    assert subset.iloc[0, 1].payload == ["source", "left"]
    assert subset.iloc[0, 2].payload == ["source", "right"]


def test_require_safe_isolates_empty_categorical_categories() -> None:
    category = _MutableHashableCategory("unused")
    frame = pd.DataFrame(
        {
            "available_at": pd.Series([], dtype="datetime64[ns, UTC]"),
            "payload": pd.Series(
                pd.Categorical.from_codes(
                    [],
                    categories=[category],
                    ordered=True,
                )
            ),
        },
        index=pd.Index([], name="sample"),
    )

    safe = LeakageGuard().require_safe(frame, DECISION_TIME)

    assert safe.empty
    assert safe.index.name == "sample"
    assert safe["payload"].dtype == frame["payload"].dtype
    assert safe["payload"].cat.ordered
    assert safe["payload"].cat.codes.empty
    safe["payload"].cat.categories[0].payload.append("returned")

    assert frame["payload"].cat.categories[0].payload == ["source"]


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
