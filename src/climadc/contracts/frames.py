from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import pandas as pd

from climadc.errors import ContractError
from climadc.validation.units import validate_unit_consistency

CLIMATE_COLUMNS = (
    "site_id",
    "issue_time",
    "available_at",
    "valid_time",
    "variable",
    "value",
    "unit",
    "source",
    "quantile",
    "member",
)
TELEMETRY_COLUMNS = (
    "site_id",
    "device_id",
    "event_time",
    "available_at",
    "metric",
    "value",
    "unit",
    "quality",
)
WORKLOAD_COLUMNS = (
    "job_id",
    "site_id",
    "event_time",
    "available_at",
    "deadline",
    "resource_type",
    "demand",
    "unit",
    "flexible_fraction",
)
PREDICTION_COLUMNS = (
    "site_id",
    "issue_time",
    "valid_time",
    "target",
    "value",
    "unit",
    "model_id",
    "quantile",
)

CLIMATE_KEY = ("site_id", "issue_time", "valid_time", "variable", "quantile", "member")
TELEMETRY_KEY = ("site_id", "device_id", "event_time", "metric")
WORKLOAD_KEY = ("site_id", "job_id", "event_time", "resource_type")
PREDICTION_KEY = ("site_id", "model_id", "issue_time", "valid_time", "target", "quantile")

_NULLABLE_COLUMNS = frozenset({"device_id", "job_id", "deadline", "quantile", "member"})
_QUALITY_VALUES = frozenset({"observed", "imputed", "estimated"})

Invariant = Callable[[pd.DataFrame, str], None]


def _row_word(count: int) -> str:
    return "row" if count == 1 else "rows"


def _raise_row_error(contract_name: str, message: str, mask: pd.Series) -> None:
    count = int(mask.sum())
    if count:
        raise ContractError(f"{contract_name}: {message}: {count} offending {_row_word(count)}")


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_numeric(frame: pd.DataFrame, column: str, contract_name: str) -> None:
    invalid = pd.Series(
        [not _is_real_number(value) for value in frame[column]],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(contract_name, f"{column} must be numeric", invalid)


def _validate_quantiles(frame: pd.DataFrame, contract_name: str) -> None:
    invalid = pd.Series(False, index=frame.index, dtype=bool)
    present = frame["quantile"].notna()
    for index, value in frame.loc[present, "quantile"].items():
        invalid.loc[index] = not _is_real_number(value) or not 0 < float(value) < 1
    _raise_row_error(contract_name, "quantile must be strictly inside (0, 1)", invalid)


def _validate_unit_group(
    frame: pd.DataFrame,
    name_column: str,
    contract_name: str,
) -> None:
    try:
        validate_unit_consistency(frame, name_column, "unit")
    except ContractError as exc:
        raise ContractError(f"{contract_name}: {exc}") from exc


def _normalize_timestamp_column(
    frame: pd.DataFrame,
    column: str,
    contract_name: str,
) -> None:
    normalized: list[object] = []
    invalid = pd.Series(False, index=frame.index, dtype=bool)
    for index, value in frame[column].items():
        if pd.isna(value):
            normalized.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("naive timestamp")
            normalized.append(timestamp.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError):
            normalized.append(pd.NaT)
            invalid.loc[index] = True

    _raise_row_error(
        contract_name,
        f"{column} values must already be timezone-aware",
        invalid,
    )
    frame[column] = pd.Series(normalized, index=frame.index, dtype="datetime64[ns, UTC]")


def _normalize_contract(
    frame: pd.DataFrame,
    contract_name: str,
    required_columns: Sequence[str],
    timestamp_columns: Sequence[str],
    key: Sequence[str],
    invariant: Invariant,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"{contract_name}: expected a pandas DataFrame")

    actual_columns = list(frame.columns)
    required = list(required_columns)
    if len(actual_columns) != len(required) or set(actual_columns) != set(required):
        missing = sorted(set(required).difference(actual_columns))
        extra = sorted(set(actual_columns).difference(required))
        raise ContractError(
            f"{contract_name}: expected exact columns; missing={missing}, extra={extra}; "
            f"{len(frame)} offending {_row_word(len(frame))}"
        )

    normalized: pd.DataFrame = frame.loc[:, required].copy(deep=True)
    required_values = [column for column in required if column not in _NULLABLE_COLUMNS]
    for column in required_values:
        _raise_row_error(
            contract_name,
            f"{column} must not be null",
            normalized[column].isna(),
        )

    for column in timestamp_columns:
        _normalize_timestamp_column(normalized, column, contract_name)

    duplicate_rows = normalized.duplicated(subset=list(key), keep=False)
    _raise_row_error(contract_name, "duplicate key", duplicate_rows)

    invariant(normalized, contract_name)
    normalized.sort_values(
        by=list(key),
        kind="mergesort",
        na_position="last",
        inplace=True,
    )
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def _validate_climate_rows(frame: pd.DataFrame, contract_name: str) -> None:
    order_invalid = ~(
        (frame["issue_time"] <= frame["available_at"])
        & (frame["available_at"] <= frame["valid_time"])
    )
    _raise_row_error(
        contract_name,
        "expected issue_time <= available_at <= valid_time",
        order_invalid,
    )
    _validate_numeric(frame, "value", contract_name)
    _validate_quantiles(frame, contract_name)
    _validate_unit_group(frame, "variable", contract_name)


def _validate_telemetry_rows(frame: pd.DataFrame, contract_name: str) -> None:
    _raise_row_error(
        contract_name,
        "expected event_time <= available_at",
        ~(frame["event_time"] <= frame["available_at"]),
    )
    _raise_row_error(
        contract_name,
        f"quality must be one of {sorted(_QUALITY_VALUES)}",
        ~frame["quality"].isin(_QUALITY_VALUES),
    )
    _validate_numeric(frame, "value", contract_name)
    _validate_unit_group(frame, "metric", contract_name)


def _validate_workload_rows(frame: pd.DataFrame, contract_name: str) -> None:
    _raise_row_error(
        contract_name,
        "expected event_time <= available_at",
        ~(frame["event_time"] <= frame["available_at"]),
    )
    deadline_invalid = frame["deadline"].notna() & (frame["deadline"] < frame["event_time"])
    _raise_row_error(contract_name, "expected deadline >= event_time", deadline_invalid)
    _validate_numeric(frame, "demand", contract_name)
    invalid_fraction = pd.Series(
        [
            not _is_real_number(value) or not 0 <= float(value) <= 1
            for value in frame["flexible_fraction"]
        ],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(
        contract_name,
        "flexible_fraction must be numeric and inside [0, 1]",
        invalid_fraction,
    )
    _validate_unit_group(frame, "resource_type", contract_name)


def _validate_prediction_rows(frame: pd.DataFrame, contract_name: str) -> None:
    _raise_row_error(
        contract_name,
        "expected issue_time <= valid_time",
        ~(frame["issue_time"] <= frame["valid_time"]),
    )
    _validate_numeric(frame, "value", contract_name)
    _validate_quantiles(frame, contract_name)
    _validate_unit_group(frame, "target", contract_name)


@dataclass(frozen=True)
class ClimateForecastFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> ClimateForecastFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="ClimateForecastFrame",
            required_columns=CLIMATE_COLUMNS,
            timestamp_columns=("issue_time", "available_at", "valid_time"),
            key=CLIMATE_KEY,
            invariant=_validate_climate_rows,
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame


@dataclass(frozen=True)
class DCTelemetryFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> DCTelemetryFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="DCTelemetryFrame",
            required_columns=TELEMETRY_COLUMNS,
            timestamp_columns=("event_time", "available_at"),
            key=TELEMETRY_KEY,
            invariant=_validate_telemetry_rows,
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame


@dataclass(frozen=True)
class WorkloadFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> WorkloadFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="WorkloadFrame",
            required_columns=WORKLOAD_COLUMNS,
            timestamp_columns=("event_time", "available_at", "deadline"),
            key=WORKLOAD_KEY,
            invariant=_validate_workload_rows,
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame


@dataclass(frozen=True)
class PredictionFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> PredictionFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="PredictionFrame",
            required_columns=PREDICTION_COLUMNS,
            timestamp_columns=("issue_time", "valid_time"),
            key=PREDICTION_KEY,
            invariant=_validate_prediction_rows,
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame
