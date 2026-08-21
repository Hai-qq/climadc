from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd
from pint.errors import PintError

from climadc.errors import ContractError
from climadc.validation.units import (
    UNIT_REGISTRY,
    validate_expected_unit_dimension,
    validate_unit_consistency,
)

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
GRID_SIGNAL_COLUMNS = (
    "site_id",
    "region_id",
    "issue_time",
    "available_at",
    "valid_time",
    "signal",
    "value",
    "unit",
    "source",
    "quality",
    "quantile",
)
FLEXIBLE_WORKLOAD_COLUMNS = (
    "job_id",
    "site_id",
    "release_time",
    "available_at",
    "deadline",
    "energy",
    "energy_unit",
    "max_power",
    "power_unit",
    "preemptible",
    "priority",
)

CLIMATE_KEY = ("site_id", "issue_time", "valid_time", "variable", "quantile", "member")
TELEMETRY_KEY = ("site_id", "device_id", "event_time", "metric")
WORKLOAD_KEY = ("site_id", "job_id", "event_time", "resource_type")
PREDICTION_KEY = ("site_id", "model_id", "issue_time", "valid_time", "target", "quantile")
GRID_SIGNAL_KEY = (
    "site_id",
    "region_id",
    "source",
    "signal",
    "quality",
    "issue_time",
    "valid_time",
    "quantile",
)
FLEXIBLE_WORKLOAD_KEY = ("site_id", "job_id")

_QUALITY_VALUES = frozenset({"observed", "imputed", "estimated"})
_GRID_QUALITY_VALUES = frozenset({"forecast", "observed", "estimated"})
_GRID_SIGNAL_VALUES = frozenset({"carbon_intensity", "energy_price"})
_GRID_SIGNAL_UNIT_REFERENCES = {
    "carbon_intensity": ("gCO2e / kWh",),
    "energy_price": ("GBP / kWh", "USD / kWh", "EUR / kWh", "CNY / kWh"),
}

Invariant = Callable[[pd.DataFrame, str], None]


def _row_word(count: int) -> str:
    return "row" if count == 1 else "rows"


def _raise_row_error(contract_name: str, message: str, mask: pd.Series) -> None:
    count = int(mask.sum())
    if count:
        raise ContractError(f"{contract_name}: {message}: {count} offending {_row_word(count)}")


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_finite_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _validate_numeric(frame: pd.DataFrame, column: str, contract_name: str) -> None:
    non_numeric = pd.Series(
        [not _is_real_number(value) for value in frame[column]],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(contract_name, f"{column} must be numeric", non_numeric)
    non_finite = pd.Series(
        [not _is_finite_real_number(value) for value in frame[column]],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(contract_name, f"{column} must be finite", non_finite)


def _validate_quantiles(frame: pd.DataFrame, contract_name: str) -> None:
    invalid = pd.Series(
        [
            False if pd.isna(value) else not _is_real_number(value) or not 0 < float(value) < 1
            for value in frame["quantile"]
        ],
        index=frame.index,
        dtype=bool,
    )
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


def _validate_non_empty_strings(
    frame: pd.DataFrame,
    columns: Sequence[str],
    contract_name: str,
) -> None:
    for column in columns:
        invalid = frame[column].map(lambda value: not isinstance(value, str) or not value.strip())
        _raise_row_error(
            contract_name,
            f"{column} must be a non-empty string",
            invalid,
        )


def _normalize_timestamp_column(
    frame: pd.DataFrame,
    column: str,
    contract_name: str,
) -> None:
    normalized: list[object] = []
    invalid = pd.Series(False, index=frame.index, dtype=bool)
    for position, value in enumerate(frame[column].tolist()):
        if not pd.api.types.is_scalar(value):
            normalized.append(pd.NaT)
            invalid.iloc[position] = True
            continue
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
            invalid.iloc[position] = True

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
    nullable_columns: Sequence[str] = (),
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
    normalized.reset_index(drop=True, inplace=True)
    nullable = frozenset(nullable_columns)
    unknown_nullable = nullable.difference(required)
    if unknown_nullable:
        raise ContractError(
            f"{contract_name}: nullable columns are not in the contract: {sorted(unknown_nullable)}"
        )
    required_values = [column for column in required if column not in nullable]
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
            not _is_finite_real_number(value) or not 0 <= float(value) <= 1
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


def _validate_grid_signal_rows(frame: pd.DataFrame, contract_name: str) -> None:
    _validate_non_empty_strings(
        frame,
        ("site_id", "region_id", "signal", "source", "quality"),
        contract_name,
    )
    _raise_row_error(
        contract_name,
        f"quality must be one of {sorted(_GRID_QUALITY_VALUES)}",
        ~frame["quality"].isin(_GRID_QUALITY_VALUES),
    )
    _raise_row_error(
        contract_name,
        f"signal must be one of {sorted(_GRID_SIGNAL_VALUES)}",
        ~frame["signal"].isin(_GRID_SIGNAL_VALUES),
    )
    _validate_numeric(frame, "value", contract_name)
    _validate_quantiles(frame, contract_name)

    forecast = frame["quality"] == "forecast"
    realized = ~forecast
    _raise_row_error(
        contract_name,
        "forecast rows require issue_time",
        forecast & frame["issue_time"].isna(),
    )
    forecast_order_invalid = forecast & ~(
        (frame["issue_time"] <= frame["available_at"])
        & (frame["available_at"] <= frame["valid_time"])
    )
    _raise_row_error(
        contract_name,
        "forecast rows require issue_time <= available_at <= valid_time",
        forecast_order_invalid,
    )
    _raise_row_error(
        contract_name,
        "realized rows must not set issue_time",
        realized & frame["issue_time"].notna(),
    )
    _raise_row_error(
        contract_name,
        "realized rows require valid_time <= available_at",
        realized & ~(frame["valid_time"] <= frame["available_at"]),
    )
    _raise_row_error(
        contract_name,
        "realized rows must not set quantile",
        realized & frame["quantile"].notna(),
    )

    for _, unit_rows in frame.groupby(
        ["site_id", "signal"],
        dropna=False,
        sort=False,
        observed=True,
    ):
        try:
            validate_unit_consistency(unit_rows, "signal", "unit")
        except ContractError as exc:
            raise ContractError(f"{contract_name}: {exc}") from exc
    for signal, expected_units in _GRID_SIGNAL_UNIT_REFERENCES.items():
        rows = frame.loc[frame["signal"] == signal]
        if not rows.empty:
            validate_expected_unit_dimension(
                rows,
                "unit",
                expected_units,
                contract_name,
            )
    _raise_row_error(
        contract_name,
        "carbon_intensity must be nonnegative",
        (frame["signal"] == "carbon_intensity") & (frame["value"] < 0.0),
    )


def _validate_flexible_workload_rows(frame: pd.DataFrame, contract_name: str) -> None:
    _validate_non_empty_strings(frame, ("job_id", "site_id"), contract_name)
    _raise_row_error(
        contract_name,
        "expected release_time <= available_at <= deadline",
        ~(
            (frame["release_time"] <= frame["available_at"])
            & (frame["available_at"] <= frame["deadline"])
        ),
    )
    for column in ("energy", "max_power", "priority"):
        _validate_numeric(frame, column, contract_name)
    _raise_row_error(
        contract_name,
        "energy must be positive",
        frame["energy"] <= 0.0,
    )
    _raise_row_error(
        contract_name,
        "max_power must be positive",
        frame["max_power"] <= 0.0,
    )
    _raise_row_error(
        contract_name,
        "priority must be nonnegative",
        frame["priority"] < 0.0,
    )

    invalid_preemptible = pd.Series(
        [not isinstance(value, (bool, np.bool_)) for value in frame["preemptible"].tolist()],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(
        contract_name,
        "preemptible must be boolean",
        invalid_preemptible,
    )
    unsupported = pd.Series(
        [
            isinstance(value, (bool, np.bool_)) and not bool(value)
            for value in frame["preemptible"].tolist()
        ],
        index=frame.index,
        dtype=bool,
    )
    _raise_row_error(
        contract_name,
        "only preemptible jobs are supported in v0.2",
        unsupported,
    )

    validate_expected_unit_dimension(frame, "energy_unit", ("kWh",), contract_name)
    validate_expected_unit_dimension(frame, "power_unit", ("kW",), contract_name)

    infeasible = pd.Series(False, index=frame.index, dtype=bool)
    rows = zip(
        frame["energy"].tolist(),
        frame["energy_unit"].tolist(),
        frame["max_power"].tolist(),
        frame["power_unit"].tolist(),
        strict=True,
    )
    for position, (energy, energy_unit, power, power_unit) in enumerate(rows):
        try:
            runtime = (
                (float(energy) * UNIT_REGISTRY.parse_units(energy_unit))
                / (float(power) * UNIT_REGISTRY.parse_units(power_unit))
            ).to("hour")
            runtime_hours = float(runtime.magnitude)
        except (PintError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ContractError(f"{contract_name}: unable to compute minimum job runtime") from exc
        row = frame.iloc[position]
        window_hours = float((row["deadline"] - row["available_at"]) / pd.Timedelta(hours=1))
        infeasible.iloc[position] = (
            not isfinite(runtime_hours) or runtime_hours > window_hours + 1e-12
        )
    _raise_row_error(
        contract_name,
        "minimum runtime exceeds available window before deadline",
        infeasible,
    )


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
            nullable_columns=("quantile", "member"),
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
            nullable_columns=("device_id",),
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
            nullable_columns=("job_id", "deadline"),
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
            nullable_columns=("quantile",),
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame


@dataclass(frozen=True)
class GridSignalFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> GridSignalFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="GridSignalFrame",
            required_columns=GRID_SIGNAL_COLUMNS,
            timestamp_columns=("issue_time", "available_at", "valid_time"),
            key=GRID_SIGNAL_KEY,
            invariant=_validate_grid_signal_rows,
            nullable_columns=("issue_time", "quantile"),
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame


@dataclass(frozen=True)
class FlexibleWorkloadFrame:
    _frame: pd.DataFrame

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> FlexibleWorkloadFrame:
        normalized = _normalize_contract(
            frame=frame,
            contract_name="FlexibleWorkloadFrame",
            required_columns=FLEXIBLE_WORKLOAD_COLUMNS,
            timestamp_columns=("release_time", "available_at", "deadline"),
            key=FLEXIBLE_WORKLOAD_KEY,
            invariant=_validate_flexible_workload_rows,
        )
        return cls(normalized)

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy(deep=True) if copy else self._frame
