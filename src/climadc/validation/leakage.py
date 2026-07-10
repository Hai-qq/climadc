from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TypeGuard

import pandas as pd

from climadc.errors import LeakageError


def _deep_copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    copied: pd.DataFrame = frame.copy(deep=True)
    for position, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            values = [deepcopy(value) for value in frame.iloc[:, position].tolist()]
            copied.isetitem(position, pd.Series(values, index=copied.index, dtype=object))
    return copied


def _is_aware_timestamp(value: object) -> TypeGuard[pd.Timestamp]:
    return (
        isinstance(value, pd.Timestamp)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _is_utc_timestamp(value: object) -> TypeGuard[pd.Timestamp]:
    return _is_aware_timestamp(value) and str(value.tzinfo) == "UTC"


def _require_utc(decision_time: object) -> pd.Timestamp:
    if not _is_utc_timestamp(decision_time):
        raise LeakageError("decision_time must be a scalar timezone-aware UTC pandas Timestamp")
    return decision_time


def _require_available_at(frame: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    if "available_at" not in frame.columns:
        raise LeakageError("available_at column is required")

    timestamps: list[pd.Timestamp] = []
    for value in frame["available_at"].tolist():
        if not _is_aware_timestamp(value):
            raise LeakageError(
                "available_at values must be non-null scalar timezone-aware pandas Timestamps"
            )
        timestamps.append(value)
    return tuple(timestamps)


@dataclass(frozen=True)
class LeakageAudit:
    decision_time: pd.Timestamp
    accepted_rows: int
    rejected_rows: int
    violations: tuple[dict[str, object], ...]


class LeakageGuard:
    def audit(self, frame: pd.DataFrame, decision_time: pd.Timestamp) -> LeakageAudit:
        checked_decision_time = _require_utc(decision_time)
        timestamps = _require_available_at(frame)
        unsafe_positions = [
            position
            for position, available_at in enumerate(timestamps)
            if available_at.tz_convert("UTC") > checked_decision_time
        ]
        violations: tuple[dict[str, object], ...] = tuple(
            {
                "index": str(frame.index[position]),
                "available_at": timestamps[position].isoformat(),
            }
            for position in unsafe_positions
        )
        rejected_rows = len(unsafe_positions)
        return LeakageAudit(
            decision_time=checked_decision_time,
            accepted_rows=len(frame) - rejected_rows,
            rejected_rows=rejected_rows,
            violations=violations,
        )

    def require_safe(self, frame: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
        audit = self.audit(frame, decision_time)
        if audit.rejected_rows:
            noun = "row" if audit.rejected_rows == 1 else "rows"
            raise LeakageError(f"{audit.rejected_rows} {noun} unavailable at decision time")
        return _deep_copy_frame(frame)

    def safe_subset(
        self,
        frame: pd.DataFrame,
        decision_time: pd.Timestamp,
    ) -> tuple[pd.DataFrame, LeakageAudit]:
        audit = self.audit(frame, decision_time)
        timestamps = _require_available_at(frame)
        safe_positions = [
            position
            for position, available_at in enumerate(timestamps)
            if available_at.tz_convert("UTC") <= audit.decision_time
        ]
        return _deep_copy_frame(frame.iloc[safe_positions]), audit
