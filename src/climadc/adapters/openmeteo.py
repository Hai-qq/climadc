from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlencode

import pandas as pd

from climadc.contracts.frames import ClimateForecastFrame
from climadc.errors import ConfigurationError

_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
_PROVIDER = "Open-Meteo"
_MODEL = "best_match"
_TIMEOUT_SECONDS = 30.0
_USER_AGENT = "climadc/0.1"

Transport = Callable[[str], Mapping[str, object]]
Clock = Callable[[], pd.Timestamp]


def _urllib_json_transport(url: str) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Open-Meteo request failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Open-Meteo response must be a JSON object")
    return payload


def _validate_coordinate(value: object, name: str, lower: float, upper: float) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not lower <= float(value) <= upper
    ):
        raise ConfigurationError(f"{name} must be a finite number inside [{lower}, {upper}]")
    return float(value)


def _exact_utc_timestamp(value: object, name: str) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
        or str(value.tzinfo) != "UTC"
    ):
        raise ConfigurationError(f"{name} must be a scalar pandas Timestamp in exact UTC")
    return value


def _validate_variables(variables: object) -> tuple[str, ...]:
    if isinstance(variables, str) or not isinstance(variables, Sequence):
        raise ConfigurationError("variables must be a non-empty sequence of unique strings")
    normalized = tuple(variables)
    if (
        not normalized
        or any(not isinstance(variable, str) or not variable.strip() for variable in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ConfigurationError("variables must be a non-empty sequence of unique strings")
    return normalized


def _validate_horizon(horizon: object) -> pd.Timedelta:
    if not isinstance(horizon, pd.Timedelta) or pd.isna(horizon) or horizon <= pd.Timedelta(0):
        raise ConfigurationError("horizon must be a positive pandas Timedelta")
    return horizon


def _build_url(
    latitude: float,
    longitude: float,
    variables: Sequence[str],
    horizon: pd.Timedelta,
) -> str:
    forecast_hours = math.ceil(horizon / pd.Timedelta(hours=1))
    query = urlencode(
        [
            ("latitude", str(latitude)),
            ("longitude", str(longitude)),
            ("hourly", ",".join(variables)),
            ("forecast_hours", str(forecast_hours)),
            ("timezone", "UTC"),
        ]
    )
    return f"{_ENDPOINT}?{query}"


def _payload_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"Open-Meteo payload field {name!r} must be an array")
    return value


def _payload_times(hourly: Mapping[str, object]) -> list[pd.Timestamp]:
    raw_times = _payload_sequence(hourly.get("time"), "hourly.time")
    times: list[pd.Timestamp] = []
    for value in raw_times:
        try:
            timestamp = pd.Timestamp(cast(Any, value))
            if pd.isna(timestamp):
                raise ValueError("NaT")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"Invalid Open-Meteo hourly time {value!r}") from exc
        times.append(timestamp)
    return times


def _rows_from_payload(
    payload: Mapping[str, object],
    latitude: float,
    longitude: float,
    issued_at: pd.Timestamp,
    retrieved_at: pd.Timestamp,
    variables: Sequence[str],
    horizon: pd.Timedelta,
    source: str,
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, Mapping) or not isinstance(units, Mapping):
        raise ConfigurationError("Open-Meteo payload requires hourly and hourly_units objects")

    times = _payload_times(hourly)
    end = issued_at + horizon
    eligible = [
        index
        for index, valid_time in enumerate(times)
        if issued_at < valid_time <= end and retrieved_at <= valid_time
    ]
    if not eligible:
        raise ConfigurationError("Open-Meteo payload contains no eligible forecast rows")

    site_id = f"open-meteo:{latitude:.6f},{longitude:.6f}"
    rows: list[dict[str, object]] = []
    for variable in variables:
        values = _payload_sequence(hourly.get(variable), f"hourly.{variable}")
        if len(values) != len(times):
            raise ConfigurationError("Open-Meteo hourly arrays must have equal lengths")
        unit = units.get(variable)
        if not isinstance(unit, str) or not unit.strip():
            raise ConfigurationError(f"Open-Meteo hourly_units missing {variable!r}")
        for index in eligible:
            value = values[index]
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ConfigurationError(
                    "Open-Meteo eligible forecast values must be finite numbers"
                )
            rows.append(
                {
                    "site_id": site_id,
                    "issue_time": issued_at,
                    "available_at": retrieved_at,
                    "valid_time": times[index],
                    "variable": variable,
                    "value": float(value),
                    "unit": unit,
                    "source": source,
                    "quantile": pd.NA,
                    "member": pd.NA,
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows))


class OpenMeteoAdapter:
    def __init__(
        self,
        transport: Transport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._transport = transport or _urllib_json_transport
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._metadata: Mapping[str, str] = MappingProxyType({})

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    def fetch(
        self,
        latitude: float,
        longitude: float,
        issued_at: pd.Timestamp,
        variables: Sequence[str],
        horizon: pd.Timedelta,
    ) -> ClimateForecastFrame:
        latitude_value = _validate_coordinate(latitude, "latitude", -90.0, 90.0)
        longitude_value = _validate_coordinate(longitude, "longitude", -180.0, 180.0)
        issued_at_value = _exact_utc_timestamp(issued_at, "issued_at")
        variable_values = _validate_variables(variables)
        horizon_value = _validate_horizon(horizon)

        url = _build_url(latitude_value, longitude_value, variable_values, horizon_value)
        payload = self._transport(url)
        if not isinstance(payload, Mapping):
            raise ConfigurationError("Open-Meteo transport must return a mapping")
        retrieved_at = _exact_utc_timestamp(self._clock(), "retrieval timestamp")
        if retrieved_at < issued_at_value:
            raise ConfigurationError("retrieval timestamp must not precede issued_at")

        retrieved_iso = retrieved_at.isoformat()
        source = f"provider={_PROVIDER}; model={_MODEL}; url={url}; retrieved_at={retrieved_iso}"
        frame = _rows_from_payload(
            payload,
            latitude_value,
            longitude_value,
            issued_at_value,
            retrieved_at,
            variable_values,
            horizon_value,
            source,
        )
        result = ClimateForecastFrame.from_pandas(frame)
        self._metadata = MappingProxyType(
            {
                "provider": _PROVIDER,
                "model": _MODEL,
                "url": url,
                "retrieved_at": retrieved_iso,
            }
        )
        return result
