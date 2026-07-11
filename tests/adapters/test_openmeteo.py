from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest

from climadc.adapters.openmeteo import OpenMeteoAdapter, _urllib_json_transport
from climadc.errors import ConfigurationError


FIXTURE = Path(__file__).parents[1] / "fixtures" / "openmeteo_response.json"
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
