from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from climadc.adapters.prometheus import KeplerPrometheusAdapter, PrometheusRangeAdapter
from climadc.errors import ConfigurationError

START = pd.Timestamp("2026-08-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:10:00Z")
RETRIEVED = pd.Timestamp("2026-08-01T00:11:00Z")


def _matrix_payload() -> dict[str, object]:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"node_name": "node-a", "container_id": "ctr-a"},
                    "values": [[START.timestamp(), "120.5"], [END.timestamp(), "122.0"]],
                },
                {
                    "metric": {"node_name": "node-b", "container_id": "ctr-b"},
                    "values": [[START.timestamp(), "80.0"], [END.timestamp(), "81.5"]],
                },
            ],
        },
    }


def test_kepler_queries_documented_watt_gauge_and_preserves_retrieval_time() -> None:
    calls: list[str] = []

    def transport(url: str) -> Mapping[str, object]:
        calls.append(url)
        return _matrix_payload()

    result = KeplerPrometheusAdapter(transport=transport, clock=lambda: RETRIEVED).fetch_power(
        base_url="https://prometheus.example/base/",
        scope="container",
        component="cpu",
        start=START,
        end=END,
        step=pd.Timedelta(minutes=5),
        site_id="dc-1",
    )

    frame = result.telemetry.to_pandas()
    assert len(calls) == 1
    parsed = urlsplit(calls[0])
    assert parsed.path == "/base/api/v1/query_range"
    query = parse_qs(parsed.query)
    assert query["query"] == ["sum by (node_name, container_id) (kepler_container_cpu_watts)"]
    assert query["step"] == ["300s"]
    assert len(frame) == 4
    assert set(frame["device_id"]) == {
        "node_name=node-a|container_id=ctr-a",
        "node_name=node-b|container_id=ctr-b",
    }
    assert set(frame["metric"]) == {"cpu_power"}
    assert set(frame["unit"]) == {"W"}
    assert set(frame["quality"]) == {"estimated"}
    assert set(frame["available_at"]) == {RETRIEVED}
    assert result.metadata["kepler_metric"] == "kepler_container_cpu_watts"


def test_generic_prometheus_adapter_accepts_explicit_observed_gauge() -> None:
    result = PrometheusRangeAdapter(
        transport=lambda _: _matrix_payload(), clock=lambda: RETRIEVED
    ).fetch(
        base_url="http://prometheus:9090",
        query="facility_power_watts",
        start=START,
        end=END,
        step=pd.Timedelta(minutes=5),
        site_id="dc-1",
        metric="total_power",
        unit="W",
        device_labels=("node_name", "container_id"),
        quality="observed",
    )
    frame = result.telemetry.to_pandas()
    assert set(frame["metric"]) == {"total_power"}
    assert set(frame["quality"]) == {"observed"}
    assert result.metadata["series_count"] == "2"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "error", "error": "bad query"}, "bad query"),
        (
            {"status": "success", "data": {"resultType": "vector", "result": []}},
            "matrix",
        ),
        (
            {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"metric": {"node_name": "node-a"}, "values": []}],
                },
            },
            "container_id",
        ),
    ],
)
def test_prometheus_rejects_provider_contract_errors(
    payload: Mapping[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        PrometheusRangeAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED).fetch(
            base_url="http://prometheus:9090",
            query="metric",
            start=START,
            end=END,
            step=pd.Timedelta(minutes=5),
            site_id="dc-1",
            metric="total_power",
            unit="W",
            device_labels=("node_name", "container_id"),
        )


def test_kepler_rejects_negative_power_and_unsupported_series() -> None:
    payload = _matrix_payload()
    first = payload["data"]
    assert isinstance(first, dict)
    result = first["result"]
    assert isinstance(result, list)
    series = result[0]
    assert isinstance(series, dict)
    series["values"] = [[START.timestamp(), "-1"]]

    with pytest.raises(ConfigurationError, match="nonnegative"):
        KeplerPrometheusAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED).fetch_power(
            base_url="http://prometheus:9090",
            scope="container",
            component="cpu",
            start=START,
            end=END,
            step=pd.Timedelta(minutes=5),
            site_id="dc-1",
        )

    with pytest.raises(ConfigurationError, match="Unsupported"):
        KeplerPrometheusAdapter().fetch_power(
            base_url="http://prometheus:9090",
            scope="vm",
            component="gpu",
            start=START,
            end=END,
            step=pd.Timedelta(minutes=5),
            site_id="dc-1",
        )
