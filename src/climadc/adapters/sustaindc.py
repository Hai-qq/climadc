from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pandas as pd

from climadc.contracts import DCTelemetryFrame, GridSignalFrame
from climadc.errors import ConfigurationError

_REQUIRED_COLUMNS = (
    "day",
    "hour",
    "dc_ITE_total_power_kW",
    "dc_HVAC_total_power_kW",
    "dc_total_power_kW",
    "outside_temp",
    "bat_avg_CI",
)
_TELEMETRY_COLUMNS = (
    ("dc_ITE_total_power_kW", "sustaindc-it", "it_power", "kW"),
    ("dc_HVAC_total_power_kW", "sustaindc-hvac", "cooling_power", "kW"),
    ("dc_total_power_kW", "sustaindc-facility", "total_power", "kW"),
    ("outside_temp", "sustaindc-weather", "air_temperature", "degC"),
)


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _exact_utc(value: object, field: str) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
        or str(value.tzinfo) != "UTC"
    ):
        raise ConfigurationError(f"{field} must be a scalar pandas Timestamp in exact UTC")
    return value


def _positive_interval(value: object) -> pd.Timedelta:
    if not isinstance(value, pd.Timedelta) or pd.isna(value) or value <= pd.Timedelta(0):
        raise ConfigurationError("interval must be a positive pandas Timedelta")
    return value


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool,
) -> pd.Series:
    values: list[float] = []
    for value in frame[column].tolist():
        if isinstance(value, bool):
            raise ConfigurationError(f"SustainDC column {column} must contain finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                f"SustainDC column {column} must contain finite numbers"
            ) from exc
        if not math.isfinite(number):
            raise ConfigurationError(f"SustainDC column {column} must contain finite numbers")
        if nonnegative and number < 0.0:
            raise ConfigurationError(f"SustainDC column {column} must be nonnegative")
        values.append(number)
    return cast(pd.Series, pd.Series(values, index=frame.index, dtype=float))


def _validate_upstream_clock(frame: pd.DataFrame, interval: pd.Timedelta) -> None:
    pairs = list(zip(frame["day"].tolist(), frame["hour"].tolist(), strict=True))
    if len(set(pairs)) != len(pairs) or pairs != sorted(pairs):
        raise ConfigurationError("SustainDC day/hour rows must be unique and strictly ordered")
    first_day, first_hour = pairs[0]
    interval_hours = float(interval / pd.Timedelta(hours=1))
    for position, (day, hour) in enumerate(pairs):
        elapsed = (float(day) - float(first_day)) * 24.0 + (float(hour) - float(first_hour))
        expected = position * interval_hours
        if not math.isclose(elapsed, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigurationError(
                "SustainDC day/hour cadence does not match the declared interval"
            )


@dataclass(frozen=True)
class SustainDCResult:
    telemetry: DCTelemetryFrame
    grid_signals: GridSignalFrame
    metadata: Mapping[str, str]


class SustainDCAdapter:
    """Convert SustainDC ``all_agents_episode_*.csv`` evaluation exports."""

    def read_evaluation(
        self,
        path: Path,
        *,
        site_id: str,
        region_id: str,
        start_time: pd.Timestamp,
        interval: pd.Timedelta = pd.Timedelta(minutes=15),
    ) -> SustainDCResult:
        if not isinstance(path, Path) or not path.is_file():
            raise ConfigurationError(f"SustainDC evaluation CSV does not exist: {path}")
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ConfigurationError(
                f"Unable to read SustainDC evaluation CSV {path}: {exc}"
            ) from exc
        return self.from_pandas(
            frame,
            site_id=site_id,
            region_id=region_id,
            start_time=start_time,
            interval=interval,
            source=str(path.resolve()),
        )

    def from_pandas(
        self,
        frame: pd.DataFrame,
        *,
        site_id: str,
        region_id: str,
        start_time: pd.Timestamp,
        interval: pd.Timedelta = pd.Timedelta(minutes=15),
        source: str = "SustainDC evaluation export",
    ) -> SustainDCResult:
        if not isinstance(frame, pd.DataFrame):
            raise ConfigurationError("SustainDC input must be a pandas DataFrame")
        if frame.empty:
            raise ConfigurationError("SustainDC evaluation export contains no rows")
        missing = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ConfigurationError(f"SustainDC evaluation export misses columns: {missing}")
        canonical_site = _non_empty(site_id, "site_id")
        canonical_region = _non_empty(region_id, "region_id")
        canonical_source = _non_empty(source, "source")
        start = _exact_utc(start_time, "start_time")
        duration = _positive_interval(interval)

        normalized = frame.copy(deep=True).reset_index(drop=True)
        normalized["day"] = _numeric_column(normalized, "day", nonnegative=False)
        normalized["hour"] = _numeric_column(normalized, "hour", nonnegative=False)
        for column in (
            "dc_ITE_total_power_kW",
            "dc_HVAC_total_power_kW",
            "dc_total_power_kW",
            "bat_avg_CI",
        ):
            normalized[column] = _numeric_column(normalized, column, nonnegative=True)
        normalized["outside_temp"] = _numeric_column(normalized, "outside_temp", nonnegative=False)
        _validate_upstream_clock(normalized, duration)

        slots = pd.date_range(start=start, periods=len(normalized), freq=duration)
        available = slots + duration
        telemetry_rows: list[dict[str, object]] = []
        for source_column, device_id, metric, unit in _TELEMETRY_COLUMNS:
            for position, value in enumerate(normalized[source_column].tolist()):
                telemetry_rows.append(
                    {
                        "site_id": canonical_site,
                        "device_id": device_id,
                        "event_time": slots[position],
                        "available_at": available[position],
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                        "quality": "estimated",
                    }
                )

        grid_rows = [
            {
                "site_id": canonical_site,
                "region_id": canonical_region,
                "issue_time": pd.NaT,
                "available_at": available[position],
                "valid_time": slots[position],
                "signal": "carbon_intensity",
                "value": value,
                "unit": "gCO2e/kWh",
                "source": canonical_source,
                "quality": "estimated",
                "quantile": pd.NA,
            }
            for position, value in enumerate(normalized["bat_avg_CI"].tolist())
        ]
        metadata = MappingProxyType(
            {
                "provider": "SustainDC evaluation export",
                "source": canonical_source,
                "start_time": start.isoformat(),
                "interval": str(duration),
                "availability_basis": "simulation interval end",
                "quality": "estimated simulator output",
                "workload_boundary": (
                    "normalized load-shifting fields are not converted into jobs because the export "
                    "does not carry job energy, release, and deadline contracts"
                ),
            }
        )
        return SustainDCResult(
            telemetry=DCTelemetryFrame.from_pandas(pd.DataFrame(telemetry_rows)),
            grid_signals=GridSignalFrame.from_pandas(pd.DataFrame(grid_rows)),
            metadata=metadata,
        )
