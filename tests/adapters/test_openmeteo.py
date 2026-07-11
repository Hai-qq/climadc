from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest

import climadc.adapters.openmeteo as openmeteo
from climadc.adapters.openmeteo import OpenMeteoAdapter, _urllib_json_transport
from climadc.errors import ConfigurationError


FIXTURE = Path(__file__).parents[1] / "fixtures" / "openmeteo_response.json"
PROVENANCE = FIXTURE.with_name("openmeteo_response.provenance.json")
ISSUED_AT = pd.Timestamp("2026-01-01 00:00", tz="UTC")
RETRIEVED_AT = pd.Timestamp("2026-01-01 00:30", tz="UTC")
VARIABLES = ("temperature_2m", "relative_humidity_2m")


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_openmeteo_fetch_uses_fixture_once_and_returns_filtered_long_frame() -> None:
    calls: list[str] = []

    def transport(url: str) -> Mapping[str, object]:
        calls.append(url)
        return _payload()

    adapter = OpenMeteoAdapter(transport=transport, clock=lambda: RETRIEVED_AT)
    result = adapter.fetch(
        52.52,
        13.405,
        ISSUED_AT,
        VARIABLES,
        pd.Timedelta(hours=2, minutes=1),
    ).to_pandas()

    expected_url = (
        "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.405"
        "&hourly=temperature_2m%2Crelative_humidity_2m&forecast_hours=3&timezone=UTC"
    )
    assert calls == [expected_url]
    assert len(result) == 4
    assert set(result["valid_time"]) == {
        pd.Timestamp("2026-01-01 01:00", tz="UTC"),
        pd.Timestamp("2026-01-01 02:00", tz="UTC"),
    }
    assert set(result["issue_time"]) == {ISSUED_AT}
    assert set(result["available_at"]) == {RETRIEVED_AT}
    assert result["quantile"].isna().all()
    assert result["member"].isna().all()
    assert set(result["site_id"]) == {"open-meteo:52.520000,13.405000"}
    assert all("Open-Meteo" in source for source in result["source"])
    assert all(expected_url in source for source in result["source"])
    assert all(RETRIEVED_AT.isoformat() in source for source in result["source"])


def test_openmeteo_fixture_has_consistent_synthetic_provenance() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    payload = _payload()

    assert provenance["ownership"] == "project-owned"
    assert provenance["status"] == "synthetic"
    assert provenance["request_url"] == (
        "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.405"
        "&hourly=temperature_2m%2Crelative_humidity_2m&forecast_hours=3&timezone=UTC"
    )
    assert provenance["fixture_date"] == "2026-07-11"
    assert provenance["time_range_utc"] == [
        payload["hourly"]["time"][0],
        payload["hourly"]["time"][-1],
    ]
    assert "offline" in provenance["purpose"].lower()
    assert "not a captured api response" in provenance["purpose"].lower()


@pytest.mark.parametrize("invalid_time", [0, True, ["2026-01-01T00:00"]])
def test_openmeteo_rejects_non_string_time_even_outside_window(invalid_time: object) -> None:
    payload = _payload()
    payload["hourly"]["time"][0] = invalid_time
    adapter = OpenMeteoAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED_AT)

    with pytest.raises(ConfigurationError, match="hourly time"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


@pytest.mark.parametrize("invalid_time", ["", "not-iso"])
def test_openmeteo_rejects_empty_or_invalid_iso_time(invalid_time: str) -> None:
    payload = _payload()
    payload["hourly"]["time"][0] = invalid_time
    adapter = OpenMeteoAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED_AT)

    with pytest.raises(ConfigurationError, match="hourly time"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


@pytest.mark.parametrize("order", ["duplicate", "decreasing"])
def test_openmeteo_requires_unique_strictly_increasing_times(order: str) -> None:
    payload = _payload()
    if order == "duplicate":
        payload["hourly"]["time"][1] = payload["hourly"]["time"][0]
    else:
        payload["hourly"]["time"][1], payload["hourly"]["time"][2] = (
            payload["hourly"]["time"][2],
            payload["hourly"]["time"][1],
        )
    adapter = OpenMeteoAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED_AT)

    with pytest.raises(ConfigurationError, match="strictly increasing"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


@pytest.mark.parametrize("invalid_value", ["1.0", True, float("nan"), float("inf")])
def test_openmeteo_validates_values_before_filtering(invalid_value: object) -> None:
    payload = _payload()
    payload["hourly"]["temperature_2m"][0] = invalid_value
    adapter = OpenMeteoAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED_AT)

    with pytest.raises(ConfigurationError, match="finite numbers"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


def test_openmeteo_uses_falsey_injected_transport_and_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FalseyTransport:
        calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, url: str) -> Mapping[str, object]:
            self.calls += 1
            return _payload()

    class FalseyClock:
        calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self) -> pd.Timestamp:
            self.calls += 1
            return RETRIEVED_AT

    def unexpected_default(_: str) -> Mapping[str, object]:
        raise AssertionError("default transport selected")

    monkeypatch.setattr(openmeteo, "_urllib_json_transport", unexpected_default)
    transport = FalseyTransport()
    clock = FalseyClock()
    adapter = OpenMeteoAdapter(transport=transport, clock=clock)

    result = adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2)).to_pandas()

    assert len(result) == 4
    assert transport.calls == 1
    assert clock.calls == 1


def test_openmeteo_metadata_is_immutable_and_complete() -> None:
    adapter = OpenMeteoAdapter(transport=lambda _: _payload(), clock=lambda: RETRIEVED_AT)
    adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))

    metadata = adapter.metadata
    assert isinstance(metadata, MappingProxyType)
    assert metadata == {
        "provider": "Open-Meteo",
        "model": "best_match",
        "url": (
            "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.405"
            "&hourly=temperature_2m%2Crelative_humidity_2m&forecast_hours=2&timezone=UTC"
        ),
        "retrieved_at": RETRIEVED_AT.isoformat(),
    }
    with pytest.raises(TypeError):
        metadata["provider"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("latitude", "longitude", "issued_at", "variables", "horizon"),
    [
        (True, 0.0, ISSUED_AT, VARIABLES, pd.Timedelta(hours=1)),
        (float("nan"), 0.0, ISSUED_AT, VARIABLES, pd.Timedelta(hours=1)),
        (91.0, 0.0, ISSUED_AT, VARIABLES, pd.Timedelta(hours=1)),
        (0.0, 181.0, ISSUED_AT, VARIABLES, pd.Timedelta(hours=1)),
        (0.0, 0.0, pd.Timestamp("2026-01-01"), VARIABLES, pd.Timedelta(hours=1)),
        (
            0.0,
            0.0,
            pd.Timestamp("2026-01-01", tz="Asia/Shanghai"),
            VARIABLES,
            pd.Timedelta(hours=1),
        ),
        (0.0, 0.0, ISSUED_AT, (), pd.Timedelta(hours=1)),
        (0.0, 0.0, ISSUED_AT, ("temperature_2m", "temperature_2m"), pd.Timedelta(hours=1)),
        (0.0, 0.0, ISSUED_AT, ("",), pd.Timedelta(hours=1)),
        (0.0, 0.0, ISSUED_AT, VARIABLES, pd.Timedelta(0)),
    ],
)
def test_openmeteo_rejects_invalid_request_before_transport(
    latitude: Any,
    longitude: Any,
    issued_at: Any,
    variables: Any,
    horizon: Any,
) -> None:
    calls = 0

    def transport(_: str) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        return _payload()

    adapter = OpenMeteoAdapter(transport=transport, clock=lambda: RETRIEVED_AT)
    with pytest.raises(ConfigurationError):
        adapter.fetch(latitude, longitude, issued_at, variables, horizon)
    assert calls == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("hourly"),
        lambda payload: payload["hourly"].pop("temperature_2m"),
        lambda payload: payload["hourly"]["temperature_2m"].pop(),
        lambda payload: payload["hourly_units"].pop("temperature_2m"),
        lambda payload: payload["hourly"]["temperature_2m"].__setitem__(1, float("nan")),
    ],
)
def test_openmeteo_rejects_malformed_payload(mutate: Any) -> None:
    payload = _payload()
    mutate(payload)
    adapter = OpenMeteoAdapter(transport=lambda _: payload, clock=lambda: RETRIEVED_AT)

    with pytest.raises(ConfigurationError):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


@pytest.mark.parametrize(
    "retrieved_at",
    [
        pd.Timestamp("2025-12-31 23:59", tz="UTC"),
        pd.Timestamp("2026-01-01 00:30"),
        pd.Timestamp("2026-01-01 00:30", tz="Asia/Shanghai"),
    ],
)
def test_openmeteo_rejects_invalid_retrieval_timestamp(retrieved_at: pd.Timestamp) -> None:
    adapter = OpenMeteoAdapter(transport=lambda _: _payload(), clock=lambda: retrieved_at)

    with pytest.raises(ConfigurationError, match="retrieval"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=2))


def test_openmeteo_rejects_payload_without_eligible_rows() -> None:
    adapter = OpenMeteoAdapter(
        transport=lambda _: _payload(),
        clock=lambda: pd.Timestamp("2026-01-01 04:00", tz="UTC"),
    )

    with pytest.raises(ConfigurationError, match="eligible"):
        adapter.fetch(52.52, 13.405, ISSUED_AT, VARIABLES, pd.Timedelta(hours=3))


def test_default_transport_uses_request_headers_timeout_and_utf8_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return '{"provider": "Open-Meteo"}'.encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = _urllib_json_transport("https://example.test/forecast")

    request = captured["request"]
    assert result == {"provider": "Open-Meteo"}
    assert captured["timeout"] == 30.0
    assert request.full_url == "https://example.test/forecast"  # type: ignore[attr-defined]
    assert request.get_header("User-agent") == "climadc/0.1"  # type: ignore[attr-defined]


def test_default_transport_rejects_non_mapping_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(ConfigurationError, match="JSON object"):
        _urllib_json_transport("https://example.test/forecast")


def test_default_transport_wraps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        raise TimeoutError("offline")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)

    with pytest.raises(ConfigurationError, match="Open-Meteo request failed"):
        _urllib_json_transport("https://example.test/forecast")


def test_default_transport_wraps_invalid_utf8_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(ConfigurationError, match="Open-Meteo request failed"):
        _urllib_json_transport("https://example.test/forecast")
