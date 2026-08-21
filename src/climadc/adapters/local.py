from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from climadc.contracts.frames import (
    CLIMATE_COLUMNS,
    FLEXIBLE_WORKLOAD_COLUMNS,
    GRID_SIGNAL_COLUMNS,
    TELEMETRY_COLUMNS,
    WORKLOAD_COLUMNS,
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
    WorkloadFrame,
)
from climadc.errors import ConfigurationError

LocalFormat = Literal["csv", "parquet"]


def _rename_columns(
    frame: pd.DataFrame,
    column_map: Mapping[str, str],
    required_columns: Sequence[str],
    context: str,
    *,
    allow_empty_noncanonical: bool = False,
) -> pd.DataFrame:
    pairs = list(column_map.items())
    if any(
        not isinstance(source, str)
        or not source
        or not isinstance(destination, str)
        or not destination
        for source, destination in pairs
    ):
        raise ConfigurationError(
            f"Invalid column map for {context}: names must be non-empty strings"
        )

    destinations = [destination for _, destination in pairs]
    if len(set(destinations)) != len(destinations):
        raise ConfigurationError(f"Invalid column map for {context}: duplicate destination")

    missing_sources = sorted(source for source, _ in pairs if source not in frame.columns)
    if missing_sources:
        raise ConfigurationError(
            f"Invalid column map for {context}: missing mapped source columns {missing_sources}"
        )

    if not pairs and not allow_empty_noncanonical and set(frame.columns) != set(required_columns):
        raise ConfigurationError(
            f"Invalid column map for {context}: an empty map requires canonical columns"
        )

    renamed_columns = [column_map.get(column, column) for column in frame.columns]
    if len(set(renamed_columns)) != len(renamed_columns):
        raise ConfigurationError(f"Invalid column map for {context}: rename collision")

    return cast(pd.DataFrame, frame.rename(columns=column_map))


def _timezone(timezone: str, context: str) -> ZoneInfo:
    if not isinstance(timezone, str) or not timezone:
        raise ConfigurationError(f"Invalid timezone for {context}: expected an IANA timezone")
    try:
        return ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ConfigurationError(f"Invalid timezone for {context}: {timezone!r}") from exc


def _normalize_timestamp_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    timezone: str,
    context: str,
) -> pd.DataFrame:
    zone = _timezone(timezone, context)
    normalized = frame.copy(deep=True)
    for column in columns:
        if column not in normalized.columns:
            continue
        values: list[object] = []
        for value in normalized[column].tolist():
            if not pd.api.types.is_scalar(value):
                raise ConfigurationError(
                    f"Invalid timestamp in {column!r} for {context}: expected a scalar"
                )
            if pd.isna(value):
                values.append(pd.NaT)
                continue
            try:
                timestamp = pd.Timestamp(value)
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    timestamp = timestamp.tz_localize(zone, ambiguous="raise", nonexistent="raise")
                values.append(timestamp.tz_convert("UTC"))
            except Exception as exc:
                raise ConfigurationError(
                    f"Invalid timestamp in {column!r} for {context}: {value!r}"
                ) from exc
        normalized[column] = pd.Series(values, index=normalized.index)
    return cast(pd.DataFrame, normalized)


def _normalize_nullable_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    for column in columns:
        if column in normalized.columns:
            normalized[column] = pd.Series(
                [pd.NA if pd.isna(value) else value for value in normalized[column]],
                index=normalized.index,
                dtype=object,
            )
    return cast(pd.DataFrame, normalized)


def _read(path: Path, format: LocalFormat) -> pd.DataFrame:
    try:
        if format == "csv":
            return pd.read_csv(path)
        if format == "parquet":
            return pd.read_parquet(path)
        raise ConfigurationError(f"Unsupported local format {format!r} for {path}")
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Unable to read local data {path}: {exc}") from exc


def read_climate(
    path: Path,
    format: LocalFormat,
    column_map: Mapping[str, str],
    timezone: str,
) -> ClimateForecastFrame:
    context = str(path)
    frame = _read(path, format)
    renamed = _rename_columns(frame, column_map, CLIMATE_COLUMNS, context)
    normalized = _normalize_timestamp_columns(
        renamed,
        ("issue_time", "available_at", "valid_time"),
        timezone,
        context,
    )
    normalized = _normalize_nullable_columns(normalized, ("quantile", "member"))
    return ClimateForecastFrame.from_pandas(normalized)


def read_telemetry(
    path: Path,
    format: LocalFormat,
    column_map: Mapping[str, str],
    timezone: str,
) -> DCTelemetryFrame:
    context = str(path)
    frame = _read(path, format)
    renamed = _rename_columns(frame, column_map, TELEMETRY_COLUMNS, context)
    normalized = _normalize_timestamp_columns(
        renamed,
        ("event_time", "available_at"),
        timezone,
        context,
    )
    normalized = _normalize_nullable_columns(normalized, ("device_id",))
    return DCTelemetryFrame.from_pandas(normalized)


def read_workload(
    path: Path,
    format: LocalFormat,
    column_map: Mapping[str, str],
    timezone: str,
) -> WorkloadFrame:
    context = str(path)
    frame = _read(path, format)
    renamed = _rename_columns(frame, column_map, WORKLOAD_COLUMNS, context)
    normalized = _normalize_timestamp_columns(
        renamed,
        ("event_time", "available_at", "deadline"),
        timezone,
        context,
    )
    normalized = _normalize_nullable_columns(normalized, ("job_id", "deadline"))
    return WorkloadFrame.from_pandas(normalized)


def read_grid_signals(
    path: Path,
    format: LocalFormat,
    column_map: Mapping[str, str],
    timezone: str,
) -> GridSignalFrame:
    context = str(path)
    frame = _read(path, format)
    renamed = _rename_columns(frame, column_map, GRID_SIGNAL_COLUMNS, context)
    normalized = _normalize_timestamp_columns(
        renamed,
        ("issue_time", "available_at", "valid_time"),
        timezone,
        context,
    )
    normalized = _normalize_nullable_columns(normalized, ("issue_time", "quantile"))
    return GridSignalFrame.from_pandas(normalized)


def read_flexible_workload(
    path: Path,
    format: LocalFormat,
    column_map: Mapping[str, str],
    timezone: str,
) -> FlexibleWorkloadFrame:
    context = str(path)
    frame = _read(path, format)
    renamed = _rename_columns(frame, column_map, FLEXIBLE_WORKLOAD_COLUMNS, context)
    normalized = _normalize_timestamp_columns(
        renamed,
        ("release_time", "available_at", "deadline"),
        timezone,
        context,
    )
    return FlexibleWorkloadFrame.from_pandas(normalized)
