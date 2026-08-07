from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from climadc.adapters.carbon_aware import CarbonAwareSDKAdapter
from climadc.errors import ConfigurationError

REQUESTED = pd.Timestamp("2026-08-01T00:04:00Z")
RETRIEVED = pd.Timestamp("2026-08-01T00:05:00Z")
START = pd.Timestamp("2026-08-01T01:00:00Z")
END = pd.Timestamp("2026-08-01T02:00:00Z")


def _point(timestamp: str, value: float = 300.0) -> dict[str, object]:
    return {
        "location": "eastus",
        "timestamp": timestamp,
        "duration": 30,
        "value": value,
    }


def _forecast_payload() -> list[dict[str, object]]:
    return [
        {
            "generatedAt": "2026-08-01T00:00:00Z",
            "requestedAt": REQUESTED.isoformat(),
            "location": "eastus",
            "dataStartAt": START.isoformat(),
            "dataEndAt": END.isoformat(),
            "windowSize": 30,
            "optimalDataPoints": [_point("2026-08-01T01:30:00Z", 250.0)],
            "forecastData": [
                _point("2026-08-01T01:00:00Z", 310.0),
                _point("2026-08-01T01:30:00Z", 250.0),
            ],
        }
    ]


def test_carbon_aware_current_forecast_preserves_issue_and_retrieval_times() -> None:
    calls: list[str] = []

    def transport(url: str) -> object:
        calls.append(url)
        return _forecast_payload()

    result = CarbonAwareSDKAdapter(
        transport=transport, clock=lambda: RETRIEVED
    ).fetch_current_forecast(
        base_url="https://carbon.example/api/",
        location="eastus",
        site_id="dc-1",
        start=START,
        end=END,
        window_size_minutes=30,
    )

    frame = result.grid_signals.to_pandas()
    parsed = urlsplit(calls[0])
    assert parsed.path == "/api/emissions/forecasts/current"
    query = parse_qs(parsed.query)
    assert query["location"] == ["eastus"]
    assert query["windowSize"] == ["30"]
    assert len(frame) == 2
    assert set(frame["issue_time"]) == {pd.Timestamp("2026-08-01T00:00:00Z")}
    assert set(frame["available_at"]) == {RETRIEVED}
    assert set(frame["quality"]) == {"forecast"}
    assert set(frame["unit"]) == {"gCO2e/kWh"}
    assert result.metadata["requested_at"] == REQUESTED.isoformat()


def test_carbon_aware_observed_defaults_to_estimated_and_is_settlement_only() -> None:
    retrieved = pd.Timestamp("2026-08-02T00:00:00Z")
    payload = [
        _point("2026-08-01T01:00:00Z", 280.0),
        _point("2026-08-01T01:30:00Z", 275.0),
    ]
    calls: list[str] = []

    def transport(url: str) -> object:
        calls.append(url)
        return payload

    result = CarbonAwareSDKAdapter(transport=transport, clock=lambda: retrieved).fetch_observed(
        base_url="http://carbon-aware:8080",
        location="eastus",
        site_id="dc-1",
        start=START,
        end=END,
    )

    frame = result.grid_signals.to_pandas()
    assert urlsplit(calls[0]).path == "/emissions/bylocation"
    assert frame["issue_time"].isna().all()
    assert set(frame["quality"]) == {"estimated"}
    assert set(frame["available_at"]) == {retrieved}
    assert result.metadata["actual_quality"] == "estimated"


def test_carbon_aware_supports_offline_payload_conversion() -> None:
    forecast = CarbonAwareSDKAdapter().forecast_from_payload(
        _forecast_payload(),
        site_id="dc-1",
        location="eastus",
        retrieved_at=RETRIEVED,
    )
    assert len(forecast.grid_signals.to_pandas()) == 2
    assert forecast.metadata["duration_minutes"] == "30"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload[0].__setitem__("generatedAt", "2026-08-01T00:05:00Z"),
            "generatedAt",
        ),
        (
            lambda payload: payload[0].__setitem__(
                "forecastData", [_point("2026-08-01T00:01:00Z")]
            ),
            "precedes response retrieval",
        ),
        (
            lambda payload: payload[0].__setitem__(
                "forecastData", [_point("2026-08-01T01:00:00Z", -1.0)]
            ),
            "nonnegative",
        ),
    ],
)
def test_carbon_aware_rejects_noncausal_or_invalid_forecasts(mutator: object, message: str) -> None:
    payload = _forecast_payload()
    assert callable(mutator)
    mutator(payload)
    with pytest.raises(ConfigurationError, match=message):
        CarbonAwareSDKAdapter().forecast_from_payload(
            payload,
            site_id="dc-1",
            location="eastus",
            retrieved_at=RETRIEVED,
        )


def test_carbon_aware_rejects_future_observation_and_embedded_credentials() -> None:
    with pytest.raises(ConfigurationError, match="later than retrieval"):
        CarbonAwareSDKAdapter().observed_from_payload(
            [_point("2026-08-02T00:00:00Z")],
            site_id="dc-1",
            location="eastus",
            retrieved_at=RETRIEVED,
        )

    with pytest.raises(ConfigurationError, match="credentials"):
        CarbonAwareSDKAdapter(transport=lambda _: []).fetch_observed(
            base_url="https://user:secret@carbon.example",
            location="eastus",
            site_id="dc-1",
            start=START,
            end=END,
        )
