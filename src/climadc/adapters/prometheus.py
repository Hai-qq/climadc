from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlencode, urlsplit, urlunsplit

import pandas as pd

from climadc.contracts import DCTelemetryFrame
from climadc.errors import ConfigurationError

_TIMEOUT_SECONDS = 30.0
_USER_AGENT = "climadc/0.2-prometheus"

PrometheusTransport = Callable[[str], Mapping[str, object]]
Clock = Callable[[], pd.Timestamp]
TelemetryQuality = Literal["observed", "estimated"]
KeplerScope = Literal["node", "container", "pod", "process", "vm"]
KeplerComponent = Literal["cpu", "gpu"]

_KEPLER_POWER_SERIES: Mapping[tuple[str, str], tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        ("node", "cpu"): ("kepler_node_cpu_watts", ("node_name",)),
        ("node", "gpu"): ("kepler_node_gpu_watts", ("node_name",)),
        ("container", "cpu"): (
            "kepler_container_cpu_watts",
            ("node_name", "container_id"),
        ),
        ("container", "gpu"): (
            "kepler_container_gpu_watts",
            ("node_name", "container_id"),
        ),
        ("pod", "cpu"): ("kepler_pod_cpu_watts", ("node_name", "pod_id")),
        ("pod", "gpu"): ("kepler_pod_gpu_watts", ("node_name", "pod_id")),
        ("process", "cpu"): (
            "kepler_process_cpu_watts",
            ("node_name", "pid"),
        ),
        ("process", "gpu"): (
            "kepler_process_gpu_watts",
            ("node_name", "pid"),
        ),
        ("vm", "cpu"): ("kepler_vm_cpu_watts", ("node_name", "vm_id")),
    }
)


def _transport(url: str) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Prometheus range request failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Prometheus response must be a JSON object")
    return payload


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


def _positive_step(value: object) -> pd.Timedelta:
    if not isinstance(value, pd.Timedelta) or pd.isna(value) or value <= pd.Timedelta(0):
        raise ConfigurationError("step must be a positive pandas Timedelta")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"Prometheus field {field} must be an array")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"Prometheus {field} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"Prometheus {field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"Prometheus {field} must be a finite number")
    return number


def _quality(value: object) -> TelemetryQuality:
    if value not in {"observed", "estimated"}:
        raise ConfigurationError("quality must be 'observed' or 'estimated'")
    return cast(TelemetryQuality, value)


def _device_id(labels: Mapping[object, object], device_labels: tuple[str, ...]) -> str:
    parts: list[str] = []
    for label in device_labels:
        raw = labels.get(label)
        value = _non_empty(raw, f"Prometheus label {label!r}")
        parts.append(f"{label}={value}")
    return "|".join(parts)


@dataclass(frozen=True)
class PrometheusRangeResult:
    telemetry: DCTelemetryFrame
    metadata: Mapping[str, str]


class PrometheusRangeAdapter:
    """Read a Prometheus ``query_range`` matrix into canonical telemetry."""

    def __init__(
        self,
        transport: PrometheusTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._transport = transport if transport is not None else _transport
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))

    def fetch(
        self,
        *,
        base_url: str,
        query: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        step: pd.Timedelta,
        site_id: str,
        metric: str,
        unit: str,
        device_labels: Sequence[str],
        quality: TelemetryQuality = "estimated",
        nonnegative: bool = False,
    ) -> PrometheusRangeResult:
        base = _base_url(base_url)
        expression = _non_empty(query, "query")
        origin = _exact_utc(start, "start")
        boundary = _exact_utc(end, "end")
        if origin >= boundary:
            raise ConfigurationError("start must be earlier than end")
        resolution = _positive_step(step)
        canonical_site = _non_empty(site_id, "site_id")
        canonical_metric = _non_empty(metric, "metric")
        canonical_unit = _non_empty(unit, "unit")
        canonical_quality = _quality(quality)
        labels = tuple(_non_empty(label, "device label") for label in device_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ConfigurationError("device_labels must be a non-empty unique sequence")

        query_string = urlencode(
            [
                ("query", expression),
                ("start", origin.isoformat()),
                ("end", boundary.isoformat()),
                ("step", f"{resolution.total_seconds():g}s"),
            ]
        )
        url = f"{base}/api/v1/query_range?{query_string}"
        payload = self._transport(url)
        retrieved = _exact_utc(self._clock(), "retrieval timestamp")
        rows, series_count = self._rows(
            payload=payload,
            start=origin,
            end=boundary,
            retrieved=retrieved,
            site_id=canonical_site,
            metric=canonical_metric,
            unit=canonical_unit,
            device_labels=labels,
            quality=canonical_quality,
            nonnegative=nonnegative,
        )
        metadata = MappingProxyType(
            {
                "provider": "Prometheus HTTP API",
                "query_url": url,
                "query": expression,
                "retrieved_at": retrieved.isoformat(),
                "series_count": str(series_count),
                "availability_basis": "HTTP response retrieval time",
            }
        )
        return PrometheusRangeResult(
            telemetry=DCTelemetryFrame.from_pandas(pd.DataFrame(rows)),
            metadata=metadata,
        )

    @staticmethod
    def _rows(
        *,
        payload: Mapping[str, object],
        start: pd.Timestamp,
        end: pd.Timestamp,
        retrieved: pd.Timestamp,
        site_id: str,
        metric: str,
        unit: str,
        device_labels: tuple[str, ...],
        quality: TelemetryQuality,
        nonnegative: bool,
    ) -> tuple[list[dict[str, object]], int]:
        if payload.get("status") != "success":
            error = payload.get("error", "unknown error")
            raise ConfigurationError(f"Prometheus query failed: {error}")
        data = payload.get("data")
        if not isinstance(data, Mapping) or data.get("resultType") != "matrix":
            raise ConfigurationError("Prometheus range response must contain matrix data")
        result = _sequence(data.get("result"), "data.result")
        rows: list[dict[str, object]] = []
        keys: set[tuple[str, pd.Timestamp]] = set()
        for series in result:
            if not isinstance(series, Mapping):
                raise ConfigurationError("Prometheus result entries must be objects")
            raw_labels = series.get("metric")
            if not isinstance(raw_labels, Mapping):
                raise ConfigurationError("Prometheus series metric labels must be an object")
            device_id = _device_id(raw_labels, device_labels)
            values = _sequence(series.get("values"), "data.result.values")
            for sample in values:
                pair = _sequence(sample, "sample")
                if len(pair) != 2:
                    raise ConfigurationError("Prometheus samples must contain timestamp and value")
                raw_timestamp = pair[0]
                if not isinstance(raw_timestamp, Real) or isinstance(raw_timestamp, bool):
                    raise ConfigurationError("Prometheus sample timestamp must be numeric")
                try:
                    event_time = pd.Timestamp(float(raw_timestamp), unit="s", tz="UTC")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ConfigurationError("Prometheus sample timestamp is invalid") from exc
                if event_time < start or event_time > end:
                    raise ConfigurationError("Prometheus sample falls outside the requested range")
                value = _finite_number(pair[1], "sample value")
                if nonnegative and value < 0.0:
                    raise ConfigurationError("Prometheus power samples must be nonnegative")
                key = (device_id, event_time)
                if key in keys:
                    raise ConfigurationError(
                        "Prometheus response contains duplicate device samples"
                    )
                keys.add(key)
                rows.append(
                    {
                        "site_id": site_id,
                        "device_id": device_id,
                        "event_time": event_time,
                        "available_at": retrieved,
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                        "quality": quality,
                    }
                )
        if not rows:
            raise ConfigurationError("Prometheus range response contains no samples")
        return rows, len(result)


class KeplerPrometheusAdapter:
    """Query documented Kepler power gauges through the read-only Prometheus API."""

    def __init__(
        self,
        transport: PrometheusTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._prometheus = PrometheusRangeAdapter(transport=transport, clock=clock)

    def fetch_power(
        self,
        *,
        base_url: str,
        scope: KeplerScope,
        component: KeplerComponent,
        start: pd.Timestamp,
        end: pd.Timestamp,
        step: pd.Timedelta,
        site_id: str,
        quality: TelemetryQuality = "estimated",
    ) -> PrometheusRangeResult:
        specification = _KEPLER_POWER_SERIES.get((scope, component))
        if specification is None:
            raise ConfigurationError(
                f"Unsupported Kepler power series: scope={scope!r}, component={component!r}"
            )
        source_metric, labels = specification
        query = f"sum by ({', '.join(labels)}) ({source_metric})"
        result = self._prometheus.fetch(
            base_url=base_url,
            query=query,
            start=start,
            end=end,
            step=step,
            site_id=site_id,
            metric=f"{component}_power",
            unit="W",
            device_labels=labels,
            quality=quality,
            nonnegative=True,
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "adapter": "KeplerPrometheusAdapter",
                "kepler_scope": scope,
                "kepler_component": component,
                "kepler_metric": source_metric,
            }
        )
        return PrometheusRangeResult(
            telemetry=result.telemetry,
            metadata=MappingProxyType(metadata),
        )
