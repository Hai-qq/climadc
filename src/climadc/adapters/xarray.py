from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from climadc.adapters.local import _normalize_timestamp_columns, _rename_columns
from climadc.contracts.frames import CLIMATE_COLUMNS, ClimateForecastFrame
from climadc.errors import ConfigurationError

_CLIMATE_BASE_COLUMNS = frozenset(CLIMATE_COLUMNS).difference({"source", "quantile", "member"})


def _contains_naive_timestamp(frame: pd.DataFrame) -> bool:
    for column in ("issue_time", "available_at", "valid_time"):
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            if pd.isna(value):
                continue
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                return True
    return False


def climate_from_xarray(
    dataset: object,
    mapping: Mapping[str, str],
    source: str,
) -> ClimateForecastFrame:
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        if exc.name == "xarray":
            raise ConfigurationError("Install climadc[xarray]") from exc
        raise

    if not isinstance(dataset, xr.Dataset):
        raise ConfigurationError("dataset must be an xarray.Dataset")

    frame = dataset.to_dataframe().reset_index()
    actual_columns = set(frame.columns)
    if not mapping and (
        not _CLIMATE_BASE_COLUMNS.issubset(actual_columns)
        or not actual_columns.issubset(set(CLIMATE_COLUMNS))
    ):
        raise ConfigurationError(
            "Invalid column_map for xarray dataset: an empty mapping requires canonical "
            "climate base columns"
        )
    renamed = _rename_columns(
        frame,
        mapping,
        CLIMATE_COLUMNS,
        "xarray dataset",
        allow_empty_noncanonical=True,
    )
    renamed["source"] = source
    for column in ("quantile", "member"):
        if column not in renamed.columns:
            renamed[column] = pd.NA

    timezone_value = dataset.attrs.get("timezone")
    if _contains_naive_timestamp(renamed) and timezone_value is None:
        raise ConfigurationError("Naive xarray timestamps require dataset.attrs['timezone']")
    timezone = timezone_value if timezone_value is not None else "UTC"
    if not isinstance(timezone, str):
        raise ConfigurationError("dataset.attrs['timezone'] must be an IANA timezone string")
    normalized = _normalize_timestamp_columns(
        renamed,
        ("issue_time", "available_at", "valid_time"),
        timezone,
        "xarray dataset",
    )
    return ClimateForecastFrame.from_pandas(normalized)
