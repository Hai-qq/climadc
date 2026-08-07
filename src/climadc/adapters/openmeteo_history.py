from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from types import MappingProxyType
from typing import cast
from urllib.parse import urlencode

import pandas as pd

from climadc.contracts import ClimateForecastFrame, DCTelemetryFrame
from climadc.errors import ConfigurationError

_PREVIOUS_RUNS_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_VARIABLE = "temperature_2m_previous_day1"
_ACTUAL_VARIABLE = "temperature_2m"
_TIMEOUT_SECONDS = 30.0
_USER_AGENT = "climadc/0.2-reference"

Transport = Callable[[str], Mapping[str, object]]
Clock = Callable[[], pd.Timestamp]


def _transport(url: str) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Open-Meteo historical request failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Open-Meteo historical response must be a JSON object")
    return payload


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


def _coordinate(value: object, field: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ConfigurationError(f"{field} must be inside [{minimum}, {maximum}]")
    return float(value)


def _horizon(value: object) -> pd.Timedelta:
    if (
        not isinstance(value, pd.Timedelta)
        or pd.isna(value)
        or value <= pd.Timedelta(0)
        or value % pd.Timedelta(hours=1) != pd.Timedelta(0)
    ):
        raise ConfigurationError("horizon must be a positive whole-hour pandas Timedelta")
    return value


def _url(
    endpoint: str,
    latitude: float,
    longitude: float,
    start: pd.Timestamp,
    horizon: pd.Timedelta,
    variable: str,
) -> str:
    final_slot = start + horizon - pd.Timedelta(hours=1)
    query = urlencode(
        [
            ("latitude", str(latitude)),
            ("longitude", str(longitude)),
            ("start_date", start.strftime("%Y-%m-%d")),
            ("end_date", final_slot.strftime("%Y-%m-%d")),
            ("hourly", variable),
            ("timezone", "UTC"),
        ]
    )
    return f"{endpoint}?{query}"


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"Open-Meteo field {field} must be an array")
    return value


def _series(
    payload: Mapping[str, object], variable: str, expected_slots: pd.DatetimeIndex
) -> tuple[list[float], str]:
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, Mapping) or not isinstance(units, Mapping):
        raise ConfigurationError("Open-Meteo historical payload requires hourly and hourly_units")
    raw_times = _sequence(hourly.get("time"), "hourly.time")
    raw_values = _sequence(hourly.get(variable), f"hourly.{variable}")
    if len(raw_times) != len(raw_values):
        raise ConfigurationError("Open-Meteo historical hourly arrays have different lengths")
    parsed: dict[pd.Timestamp, float] = {}
    for raw_time, raw_value in zip(raw_times, raw_values, strict=True):
        if not isinstance(raw_time, str) or not raw_time:
            raise ConfigurationError("Open-Meteo historical time must be a non-empty ISO string")
        try:
            dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            timestamp = pd.Timestamp(dt)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"Invalid Open-Meteo historical time {raw_time!r}") from exc
        if timestamp in parsed:
            raise ConfigurationError("Open-Meteo historical times must be unique")
        if (
            not isinstance(raw_value, Real)
            or isinstance(raw_value, bool)
            or not math.isfinite(float(raw_value))
        ):
            raise ConfigurationError("Open-Meteo historical values must be finite numbers")
        parsed[timestamp] = float(raw_value)
    missing = [slot.isoformat() for slot in expected_slots if slot not in parsed]
    if missing:
        raise ConfigurationError(f"Open-Meteo historical payload misses slots: {missing}")
    unit = units.get(variable)
    if not isinstance(unit, str) or not unit.strip():
        raise ConfigurationError(f"Open-Meteo historical unit missing for {variable}")
    return [parsed[slot] for slot in expected_slots], unit


@dataclass(frozen=True)
class OpenMeteoHistoryResult:
    forecast: ClimateForecastFrame
    actual: DCTelemetryFrame
    metadata: Mapping[str, str]


class OpenMeteoHistoryAdapter:
    """Fetch fixed-lead historical weather plus post-hoc gridded weather estimates."""

    def __init__(self, transport: Transport | None = None, clock: Clock | None = None) -> None:
        self._transport = transport if transport is not None else _transport
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))

    def fetch(
        self,
        *,
        latitude: float,
        longitude: float,
        site_id: str,
        decision_time: pd.Timestamp,
        horizon: pd.Timedelta,
    ) -> OpenMeteoHistoryResult:
        lat = _coordinate(latitude, "latitude", -90.0, 90.0)
        lon = _coordinate(longitude, "longitude", -180.0, 180.0)
        decision = _exact_utc(decision_time, "decision_time")
        if decision != decision.floor("h"):
            raise ConfigurationError("decision_time must align to an exact UTC hour")
        duration = _horizon(horizon)
        if not isinstance(site_id, str) or not site_id.strip():
            raise ConfigurationError("site_id must be a non-empty string")
        forecast_url = _url(
            _PREVIOUS_RUNS_ENDPOINT, lat, lon, decision, duration, _FORECAST_VARIABLE
        )
        actual_url = _url(_ARCHIVE_ENDPOINT, lat, lon, decision, duration, _ACTUAL_VARIABLE)
        forecast_payload = self._transport(forecast_url)
        actual_payload = self._transport(actual_url)
        if not isinstance(forecast_payload, Mapping) or not isinstance(actual_payload, Mapping):
            raise ConfigurationError("Open-Meteo historical transport must return mappings")
        retrieved = _exact_utc(self._clock(), "retrieval timestamp")
        if retrieved < decision + duration:
            raise ConfigurationError("historical retrieval must not precede the replay horizon end")
        slots = pd.date_range(decision, periods=int(duration / pd.Timedelta(hours=1)), freq="1h")
        forecast_values, forecast_unit = _series(forecast_payload, _FORECAST_VARIABLE, slots)
        actual_values, actual_unit = _series(actual_payload, _ACTUAL_VARIABLE, slots)
        forecast_source = (
            f"provider=Open-Meteo; product=previous-runs; lead=24h; url={forecast_url}; "
            f"retrieved_at={retrieved.isoformat()}"
        )
        forecast_frame = pd.DataFrame(
            {
                "site_id": [site_id.strip()] * len(slots),
                "issue_time": slots - pd.Timedelta(hours=24),
                "available_at": [decision] * len(slots),
                "valid_time": slots,
                "variable": ["air_temperature"] * len(slots),
                "value": forecast_values,
                "unit": [forecast_unit] * len(slots),
                "source": [forecast_source] * len(slots),
                "quantile": [pd.NA] * len(slots),
                "member": [pd.NA] * len(slots),
            }
        )
        actual_frame = pd.DataFrame(
            {
                "site_id": [site_id.strip()] * len(slots),
                "device_id": ["open-meteo-grid"] * len(slots),
                "event_time": slots,
                "available_at": [retrieved] * len(slots),
                "metric": ["air_temperature"] * len(slots),
                "value": actual_values,
                "unit": [actual_unit] * len(slots),
                "quality": ["estimated"] * len(slots),
            }
        )
        metadata = MappingProxyType(
            {
                "provider": "Open-Meteo",
                "forecast_url": forecast_url,
                "actual_url": actual_url,
                "retrieved_at": retrieved.isoformat(),
                "forecast_timing_basis": "fixed 24h lead; availability is a scenario assumption",
                "actual_quality": "gridded model/reanalysis estimate, not station observation",
            }
        )
        return OpenMeteoHistoryResult(
            forecast=ClimateForecastFrame.from_pandas(cast(pd.DataFrame, forecast_frame)),
            actual=DCTelemetryFrame.from_pandas(cast(pd.DataFrame, actual_frame)),
            metadata=metadata,
        )
