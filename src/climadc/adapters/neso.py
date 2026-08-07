from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import cast

import pandas as pd

from climadc.contracts import GridSignalFrame
from climadc.errors import ConfigurationError

_ENDPOINT = "https://api.carbonintensity.org.uk/intensity"
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
        raise ConfigurationError(f"NESO Carbon Intensity request failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("NESO Carbon Intensity response must be a JSON object")
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


def _whole_hour_horizon(value: object) -> pd.Timedelta:
    if (
        not isinstance(value, pd.Timedelta)
        or pd.isna(value)
        or value <= pd.Timedelta(0)
        or value % pd.Timedelta(hours=1) != pd.Timedelta(0)
    ):
        raise ConfigurationError("horizon must be a positive whole-hour pandas Timedelta")
    return value


def _iso_path(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%MZ")


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"NESO field {field} must be an array")
    return value


def _finite(value: object, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ConfigurationError(f"NESO {field} must be a finite number")
    return float(value)


class NESOCarbonIntensityAdapter:
    """Normalize national half-hour NESO forecast/estimated intensity to hourly rows."""

    def __init__(self, transport: Transport | None = None, clock: Clock | None = None) -> None:
        self._transport = transport if transport is not None else _transport
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))
        self._metadata: Mapping[str, str] = MappingProxyType({})

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    def fetch(
        self,
        *,
        site_id: str,
        decision_time: pd.Timestamp,
        horizon: pd.Timedelta,
    ) -> GridSignalFrame:
        if not isinstance(site_id, str) or not site_id.strip():
            raise ConfigurationError("site_id must be a non-empty string")
        decision = _exact_utc(decision_time, "decision_time")
        if decision != decision.floor("h"):
            raise ConfigurationError("decision_time must align to an exact UTC hour")
        duration = _whole_hour_horizon(horizon)
        end = decision + duration
        url = f"{_ENDPOINT}/{_iso_path(decision)}/{_iso_path(end)}"
        payload = self._transport(url)
        if not isinstance(payload, Mapping):
            raise ConfigurationError("NESO transport must return a mapping")
        retrieved = _exact_utc(self._clock(), "retrieval timestamp")
        if retrieved < end:
            raise ConfigurationError("historical retrieval must not precede the replay horizon end")
        records = _sequence(payload.get("data"), "data")
        parsed: dict[pd.Timestamp, tuple[float, float]] = {}
        for position, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ConfigurationError(f"NESO data[{position}] must be an object")
            intensity = record.get("intensity")
            if not isinstance(intensity, Mapping):
                raise ConfigurationError(f"NESO data[{position}].intensity must be an object")
            raw_start = record.get("from")
            raw_finish = record.get("to")
            if not isinstance(raw_start, str) or not isinstance(raw_finish, str):
                raise ConfigurationError(f"NESO data[{position}] has invalid interval times")
            try:
                start = pd.Timestamp(raw_start)
                finish = pd.Timestamp(raw_finish)
                if start.tzinfo is None or finish.tzinfo is None:
                    raise ValueError("naive interval")
                start = start.tz_convert("UTC")
                finish = finish.tz_convert("UTC")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ConfigurationError(
                    f"NESO data[{position}] has invalid interval times"
                ) from exc
            if finish - start != pd.Timedelta(minutes=30) or start in parsed:
                raise ConfigurationError("NESO intervals must be unique half-hour windows")
            parsed[start] = (
                _finite(intensity.get("forecast"), "forecast"),
                _finite(intensity.get("actual"), "actual"),
            )
        half_hours = pd.date_range(
            decision, periods=int(duration / pd.Timedelta(minutes=30)), freq="30min"
        )
        missing = [slot.isoformat() for slot in half_hours if slot not in parsed]
        if missing:
            raise ConfigurationError(f"NESO payload misses required half-hours: {missing}")
        rows: list[dict[str, object]] = []
        source = (
            f"provider=NESO Carbon Intensity API; region=national-GB; aggregation=hourly-mean; "
            f"url={url}; retrieved_at={retrieved.isoformat()}"
        )
        for slot in pd.date_range(
            decision, periods=int(duration / pd.Timedelta(hours=1)), freq="1h"
        ):
            first = parsed[slot]
            second = parsed[slot + pd.Timedelta(minutes=30)]
            forecast = (first[0] + second[0]) / 2.0
            actual = (first[1] + second[1]) / 2.0
            rows.extend(
                [
                    {
                        "site_id": site_id.strip(),
                        "region_id": "GB",
                        "issue_time": decision,
                        "available_at": decision,
                        "valid_time": slot,
                        "signal": "carbon_intensity",
                        "value": forecast,
                        "unit": "gCO2e / kWh",
                        "source": source,
                        "quality": "forecast",
                        "quantile": pd.NA,
                    },
                    {
                        "site_id": site_id.strip(),
                        "region_id": "GB",
                        "issue_time": pd.NaT,
                        "available_at": retrieved,
                        "valid_time": slot,
                        "signal": "carbon_intensity",
                        "value": actual,
                        "unit": "gCO2e / kWh",
                        "source": source,
                        "quality": "estimated",
                        "quantile": pd.NA,
                    },
                ]
            )
        self._metadata = MappingProxyType(
            {
                "provider": "NESO Carbon Intensity API",
                "url": url,
                "retrieved_at": retrieved.isoformat(),
                "region_id": "GB",
                "aggregation": "arithmetic mean of two national half-hour intervals",
                "forecast_timing_basis": "issue and availability are scenario assumptions",
                "actual_quality": "provider estimated actual",
            }
        )
        return GridSignalFrame.from_pandas(cast(pd.DataFrame, pd.DataFrame(rows)))
