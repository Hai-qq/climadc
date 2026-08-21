from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlencode, urlsplit, urlunsplit

import pandas as pd

from climadc.contracts import GridSignalFrame
from climadc.errors import ConfigurationError

_TIMEOUT_SECONDS = 30.0
_USER_AGENT = "climadc/0.2-carbon-aware-sdk"

CarbonAwareTransport = Callable[[str], object]
Clock = Callable[[], pd.Timestamp]
ActualQuality = Literal["observed", "estimated"]


def _transport(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Carbon Aware SDK request failed: {exc}") from exc


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _base_url(value: object) -> str:
    text = _non_empty(value, "base_url")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("base_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("base_url must not contain a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


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


def _api_timestamp(value: object, field: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Carbon Aware SDK field {field} must be an ISO timestamp")
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("naive timestamp")
        return cast(pd.Timestamp, timestamp.tz_convert("UTC"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"Carbon Aware SDK field {field} must be a timezone-aware ISO timestamp"
        ) from exc


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"Carbon Aware SDK field {field} must be an array")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"Carbon Aware SDK field {field} must be nonnegative and finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"Carbon Aware SDK field {field} must be nonnegative and finite"
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise ConfigurationError(f"Carbon Aware SDK field {field} must be nonnegative and finite")
    return number


def _duration(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError("Carbon Aware SDK duration must be a positive integer in minutes")
    return value


def _actual_quality(value: object) -> ActualQuality:
    if value not in {"observed", "estimated"}:
        raise ConfigurationError("actual_quality must be 'observed' or 'estimated'")
    return cast(ActualQuality, value)


def _points(
    value: object,
    *,
    field: str,
    expected_location: str,
) -> tuple[list[tuple[pd.Timestamp, float, int]], str]:
    points = _sequence(value, field)
    if not points:
        raise ConfigurationError(f"Carbon Aware SDK field {field} contains no points")
    parsed: list[tuple[pd.Timestamp, float, int]] = []
    seen: set[pd.Timestamp] = set()
    for position, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ConfigurationError(f"Carbon Aware SDK {field}[{position}] must be an object")
        location = _non_empty(point.get("location"), f"{field}[{position}].location")
        if location != expected_location:
            raise ConfigurationError(
                f"Carbon Aware SDK point location {location!r} does not match {expected_location!r}"
            )
        timestamp = _api_timestamp(point.get("timestamp"), f"{field}[{position}].timestamp")
        if timestamp in seen:
            raise ConfigurationError("Carbon Aware SDK response contains duplicate timestamps")
        seen.add(timestamp)
        parsed.append(
            (
                timestamp,
                _finite_nonnegative(point.get("value"), f"{field}[{position}].value"),
                _duration(point.get("duration")),
            )
        )
    parsed.sort(key=lambda item: item[0])
    durations = {item[2] for item in parsed}
    duration_summary = str(next(iter(durations))) if len(durations) == 1 else "mixed"
    return parsed, duration_summary


@dataclass(frozen=True)
class CarbonAwareResult:
    grid_signals: GridSignalFrame
    metadata: Mapping[str, str]


class CarbonAwareSDKAdapter:
    """Convert Carbon Aware SDK Web API payloads without coupling to a provider plug-in."""

    def __init__(
        self,
        transport: CarbonAwareTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._transport = transport if transport is not None else _transport
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))

    def fetch_current_forecast(
        self,
        *,
        base_url: str,
        location: str,
        site_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        window_size_minutes: int | None = None,
    ) -> CarbonAwareResult:
        base = _base_url(base_url)
        region = _non_empty(location, "location")
        origin = _exact_utc(start, "start")
        boundary = _exact_utc(end, "end")
        if origin >= boundary:
            raise ConfigurationError("start must be earlier than end")
        query: list[tuple[str, str]] = [
            ("location", region),
            ("dataStartAt", origin.isoformat()),
            ("dataEndAt", boundary.isoformat()),
        ]
        if window_size_minutes is not None:
            query.append(("windowSize", str(_duration(window_size_minutes))))
        url = f"{base}/emissions/forecasts/current?{urlencode(query)}"
        payload = self._transport(url)
        retrieved = _exact_utc(self._clock(), "retrieval timestamp")
        return self.forecast_from_payload(
            payload,
            site_id=site_id,
            location=region,
            retrieved_at=retrieved,
            source_url=url,
        )

    def fetch_observed(
        self,
        *,
        base_url: str,
        location: str,
        site_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        actual_quality: ActualQuality = "estimated",
    ) -> CarbonAwareResult:
        base = _base_url(base_url)
        region = _non_empty(location, "location")
        origin = _exact_utc(start, "start")
        boundary = _exact_utc(end, "end")
        if origin >= boundary:
            raise ConfigurationError("start must be earlier than end")
        url = f"{base}/emissions/bylocation?{urlencode([('location', region), ('time', origin.isoformat()), ('toTime', boundary.isoformat())])}"
        payload = self._transport(url)
        retrieved = _exact_utc(self._clock(), "retrieval timestamp")
        return self.observed_from_payload(
            payload,
            site_id=site_id,
            location=region,
            retrieved_at=retrieved,
            actual_quality=actual_quality,
            source_url=url,
        )

    def forecast_from_payload(
        self,
        payload: object,
        *,
        site_id: str,
        location: str,
        retrieved_at: pd.Timestamp,
        source_url: str = "offline Carbon Aware SDK forecast payload",
    ) -> CarbonAwareResult:
        canonical_site = _non_empty(site_id, "site_id")
        region = _non_empty(location, "location")
        retrieved = _exact_utc(retrieved_at, "retrieved_at")
        responses = _sequence(payload, "response")
        matches = [
            response
            for response in responses
            if isinstance(response, Mapping) and response.get("location") == region
        ]
        if len(matches) != 1:
            raise ConfigurationError(
                f"Carbon Aware SDK forecast requires exactly one response for {region!r}"
            )
        response = matches[0]
        assert isinstance(response, Mapping)
        generated = _api_timestamp(response.get("generatedAt"), "generatedAt")
        requested = _api_timestamp(response.get("requestedAt"), "requestedAt")
        if not generated <= requested <= retrieved:
            raise ConfigurationError(
                "Carbon Aware SDK forecast requires generatedAt <= requestedAt <= retrieved_at"
            )
        points, duration_summary = _points(
            response.get("forecastData"), field="forecastData", expected_location=region
        )
        rows: list[dict[str, object]] = []
        for valid_time, value, _ in points:
            if retrieved > valid_time:
                raise ConfigurationError(
                    "Carbon Aware SDK forecast point precedes response retrieval; request a future range"
                )
            rows.append(
                {
                    "site_id": canonical_site,
                    "region_id": region,
                    "issue_time": generated,
                    "available_at": retrieved,
                    "valid_time": valid_time,
                    "signal": "carbon_intensity",
                    "value": value,
                    "unit": "gCO2e/kWh",
                    "source": _non_empty(source_url, "source_url"),
                    "quality": "forecast",
                    "quantile": pd.NA,
                }
            )
        metadata = MappingProxyType(
            {
                "provider": "Carbon Aware SDK-compatible Web API",
                "location": region,
                "generated_at": generated.isoformat(),
                "requested_at": requested.isoformat(),
                "retrieved_at": retrieved.isoformat(),
                "duration_minutes": duration_summary,
                "availability_basis": "HTTP response retrieval time",
            }
        )
        return CarbonAwareResult(
            grid_signals=GridSignalFrame.from_pandas(pd.DataFrame(rows)),
            metadata=metadata,
        )

    def observed_from_payload(
        self,
        payload: object,
        *,
        site_id: str,
        location: str,
        retrieved_at: pd.Timestamp,
        actual_quality: ActualQuality = "estimated",
        source_url: str = "offline Carbon Aware SDK emissions payload",
    ) -> CarbonAwareResult:
        canonical_site = _non_empty(site_id, "site_id")
        region = _non_empty(location, "location")
        retrieved = _exact_utc(retrieved_at, "retrieved_at")
        quality = _actual_quality(actual_quality)
        points, duration_summary = _points(payload, field="emissionsData", expected_location=region)
        rows: list[dict[str, object]] = []
        for valid_time, value, _ in points:
            if valid_time > retrieved:
                raise ConfigurationError(
                    "Carbon Aware SDK observed point must not be later than retrieval"
                )
            rows.append(
                {
                    "site_id": canonical_site,
                    "region_id": region,
                    "issue_time": pd.NaT,
                    "available_at": retrieved,
                    "valid_time": valid_time,
                    "signal": "carbon_intensity",
                    "value": value,
                    "unit": "gCO2e/kWh",
                    "source": _non_empty(source_url, "source_url"),
                    "quality": quality,
                    "quantile": pd.NA,
                }
            )
        metadata = MappingProxyType(
            {
                "provider": "Carbon Aware SDK-compatible Web API",
                "location": region,
                "retrieved_at": retrieved.isoformat(),
                "duration_minutes": duration_summary,
                "actual_quality": quality,
                "availability_basis": "HTTP response retrieval time",
            }
        )
        return CarbonAwareResult(
            grid_signals=GridSignalFrame.from_pandas(pd.DataFrame(rows)),
            metadata=metadata,
        )
