from __future__ import annotations

from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

import pandas as pd
import pytest

from climadc.adapters.neso import NESOCarbonIntensityAdapter
from climadc.adapters.openmeteo_history import OpenMeteoHistoryAdapter
from climadc.errors import ConfigurationError
from climadc.evidence.sources import RawHTTPResponse

DECISION = pd.Timestamp("2026-08-01T00:00:00Z")
RETRIEVED = pd.Timestamp("2026-08-06T12:46:02Z")
HORIZON = pd.Timedelta(hours=24)


def _weather_payload(variable: str, offset: float = 0.0) -> dict[str, object]:
    times = [
        timestamp.strftime("%Y-%m-%dT%H:%M")
        for timestamp in pd.date_range(DECISION, periods=24, freq="1h")
    ]
    return {
        "hourly": {
            "time": times,
            variable: [float(position) + offset for position in range(24)],
        },
        "hourly_units": {variable: "degC"},
    }


def _neso_payload() -> dict[str, object]:
    data: list[dict[str, object]] = []
    for position, start in enumerate(pd.date_range(DECISION, periods=48, freq="30min")):
        data.append(
            {
                "from": start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": (start + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {
                    "forecast": float(position),
                    "actual": float(position + 10),
                    "index": "low",
                },
            }
        )
    return {"data": data}


def test_openmeteo_history_separates_fixed_lead_forecast_and_estimated_actual() -> None:
    calls: list[str] = []

    def transport(url: str) -> Mapping[str, object]:
        calls.append(url)
        if "previous-runs-api" in url:
            return _weather_payload("temperature_2m_previous_day1")
        return _weather_payload("temperature_2m", offset=0.5)

    result = OpenMeteoHistoryAdapter(transport=transport, clock=lambda: RETRIEVED).fetch(
        latitude=51.5074,
        longitude=-0.1278,
        site_id="gb-london-reference",
        decision_time=DECISION,
        horizon=HORIZON,
    )

    forecast = result.forecast.to_pandas()
    actual = result.actual.to_pandas()
    assert len(calls) == 2
    assert "temperature_2m_previous_day1" in calls[0]
    assert urlsplit(calls[1]).hostname == "archive-api.open-meteo.com"
    assert len(forecast) == len(actual) == 24
    assert (forecast["issue_time"] == forecast["valid_time"] - pd.Timedelta(hours=24)).all()
    assert set(forecast["available_at"]) == {DECISION}
    assert set(actual["available_at"]) == {RETRIEVED}
    assert set(actual["quality"]) == {"estimated"}
    assert result.metadata["forecast_timing_basis"].endswith("scenario assumption")


def test_reference_adapters_expose_raw_capture_before_parsing() -> None:
    weather_captures: list[tuple[str, RawHTTPResponse]] = []
    weather = OpenMeteoHistoryAdapter(
        transport=lambda url: (
            _weather_payload("temperature_2m_previous_day1")
            if "previous-runs-api" in url
            else _weather_payload("temperature_2m", offset=0.5)
        ),
        clock=lambda: RETRIEVED,
    )
    weather.fetch(
        latitude=51.5,
        longitude=-0.1,
        site_id="site",
        decision_time=DECISION,
        horizon=HORIZON,
        raw_capture=lambda name, response: weather_captures.append((name, response)),
    )
    assert [name for name, _ in weather_captures] == [
        "openmeteo-forecast.json",
        "openmeteo-settlement.json",
    ]
    assert all(response.capture_kind == "injected_mapping" for _, response in weather_captures)

    neso_captures: list[tuple[str, RawHTTPResponse]] = []
    NESOCarbonIntensityAdapter(transport=lambda _: _neso_payload(), clock=lambda: RETRIEVED).fetch(
        site_id="site",
        decision_time=DECISION,
        horizon=HORIZON,
        raw_capture=lambda name, response: neso_captures.append((name, response)),
    )
    assert [name for name, _ in neso_captures] == ["neso-carbon.json"]
    assert neso_captures[0][1].capture_kind == "injected_mapping"


def test_openmeteo_history_rejects_missing_slot_and_early_retrieval() -> None:
    forecast = _weather_payload("temperature_2m_previous_day1")
    cast(list[object], cast(dict[str, object], forecast["hourly"])["time"]).pop()

    def missing_transport(url: str) -> Mapping[str, object]:
        return forecast if "previous-runs-api" in url else _weather_payload("temperature_2m")

    with pytest.raises(ConfigurationError, match="different lengths|misses slots"):
        OpenMeteoHistoryAdapter(transport=missing_transport, clock=lambda: RETRIEVED).fetch(
            latitude=51.5,
            longitude=-0.1,
            site_id="site",
            decision_time=DECISION,
            horizon=HORIZON,
        )

    with pytest.raises(ConfigurationError, match="horizon end"):
        OpenMeteoHistoryAdapter(
            transport=lambda url: (
                _weather_payload("temperature_2m_previous_day1")
                if "previous-runs-api" in url
                else _weather_payload("temperature_2m")
            ),
            clock=lambda: DECISION,
        ).fetch(
            latitude=51.5,
            longitude=-0.1,
            site_id="site",
            decision_time=DECISION,
            horizon=HORIZON,
        )


def test_neso_aggregates_complete_half_hours_and_labels_timing_assumption() -> None:
    adapter = NESOCarbonIntensityAdapter(
        transport=lambda _: _neso_payload(), clock=lambda: RETRIEVED
    )
    frame = adapter.fetch(
        site_id="gb-london-reference", decision_time=DECISION, horizon=HORIZON
    ).to_pandas()

    assert len(frame) == 48
    first = frame.loc[frame["valid_time"] == DECISION].set_index("quality")
    assert first.loc["forecast", "value"] == pytest.approx(0.5)
    assert first.loc["estimated", "value"] == pytest.approx(10.5)
    assert first.loc["forecast", "issue_time"] == DECISION
    assert pd.isna(first.loc["estimated", "issue_time"])
    assert set(frame["region_id"]) == {"GB"}
    assert adapter.metadata["forecast_timing_basis"].endswith("scenario assumptions")


def test_neso_rejects_missing_actual_and_incomplete_intervals() -> None:
    missing_actual = _neso_payload()
    cast(dict[str, object], cast(list[object], missing_actual["data"])[0])["intensity"] = {
        "forecast": 1.0,
        "actual": None,
    }
    with pytest.raises(ConfigurationError, match="actual"):
        NESOCarbonIntensityAdapter(
            transport=lambda _: missing_actual, clock=lambda: RETRIEVED
        ).fetch(site_id="site", decision_time=DECISION, horizon=HORIZON)

    incomplete = _neso_payload()
    cast(list[object], incomplete["data"]).pop()
    with pytest.raises(ConfigurationError, match="misses required"):
        NESOCarbonIntensityAdapter(transport=lambda _: incomplete, clock=lambda: RETRIEVED).fetch(
            site_id="site", decision_time=DECISION, horizon=HORIZON
        )


def test_reference_sources_require_hour_aligned_decision() -> None:
    decision = DECISION + pd.Timedelta(minutes=1)
    with pytest.raises(ConfigurationError, match="align"):
        OpenMeteoHistoryAdapter(transport=lambda _: {}, clock=lambda: RETRIEVED).fetch(
            latitude=51.5,
            longitude=-0.1,
            site_id="site",
            decision_time=decision,
            horizon=HORIZON,
        )

    with pytest.raises(ConfigurationError, match="align"):
        NESOCarbonIntensityAdapter(transport=lambda _: {}, clock=lambda: RETRIEVED).fetch(
            site_id="site", decision_time=decision, horizon=HORIZON
        )
