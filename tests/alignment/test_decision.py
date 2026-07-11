from datetime import timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from climadc.alignment.decision import DecisionViewBuilder
from climadc.contracts.frames import ClimateForecastFrame, DCTelemetryFrame


class _MutableHashableId:
    def __init__(self, label: str) -> None:
        self.label = label
        self.payload = ["source"]

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MutableHashableId) and self.label == other.label

    def __lt__(self, other: "_MutableHashableId") -> bool:
        return self.label < other.label


@pytest.fixture
def climate_frame() -> ClimateForecastFrame:
    return ClimateForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * 5,
                "issue_time": pd.to_datetime(
                    [
                        "2026-01-01 18:00Z",
                        "2026-01-01 18:00Z",
                        "2026-01-01 22:00Z",
                        "2026-01-02 01:00Z",
                        "2026-01-01 23:00Z",
                    ]
                ),
                "available_at": pd.to_datetime(
                    [
                        "2026-01-01 18:05Z",
                        "2026-01-01 18:05Z",
                        "2026-01-01 22:05Z",
                        "2026-01-02 01:05Z",
                        "2026-01-01 23:05Z",
                    ]
                ),
                "valid_time": pd.to_datetime(
                    [
                        "2026-01-02 00:00Z",
                        "2026-01-02 02:00Z",
                        "2026-01-02 02:00Z",
                        "2026-01-02 04:00Z",
                        "2026-01-02 05:00Z",
                    ]
                ),
                "variable": ["air_temperature"] * 5,
                "value": [19.0, 20.0, 21.0, 22.0, 23.0],
                "unit": ["degC"] * 5,
                "source": ["fixture"] * 5,
                "quantile": [pd.NA] * 5,
                "member": [pd.NA] * 5,
            }
        )
    )


@pytest.fixture
def telemetry_frame() -> DCTelemetryFrame:
    return DCTelemetryFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * 4,
                "device_id": ["meter-1", "meter-late", "meter-1", "meter-estimated"],
                "event_time": pd.to_datetime(
                    [
                        "2026-01-01 23:00Z",
                        "2026-01-01 23:30Z",
                        "2026-01-02 04:00Z",
                        "2026-01-02 04:00Z",
                    ]
                ),
                "available_at": pd.to_datetime(
                    [
                        "2026-01-01 23:05Z",
                        "2026-01-02 00:30Z",
                        "2026-01-02 04:05Z",
                        "2026-01-02 04:03Z",
                    ]
                ),
                "metric": ["total_power"] * 4,
                "value": [100.0, 105.0, 110.0, 111.0],
                "unit": ["kW"] * 4,
                "quality": ["observed", "observed", "observed", "estimated"],
            }
        )
    )


def test_decision_view_uses_latest_available_forecast(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
) -> None:
    origin = pd.Timestamp("2026-01-02 00:00", tz="UTC")

    view = DecisionViewBuilder().build(
        climate_frame,
        telemetry_frame,
        origin,
        pd.Timedelta("4h"),
    )

    assert view.forecast["available_at"].max() <= origin
    assert view.target_time == origin + pd.Timedelta("4h")


def test_decision_view_selects_latest_issue_and_filters_horizon(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
) -> None:
    origin = pd.Timestamp("2026-01-02 00:00", tz="UTC")

    view = DecisionViewBuilder().build(
        climate_frame,
        telemetry_frame,
        origin,
        pd.Timedelta("4h"),
    )

    assert view.forecast["valid_time"].tolist() == [pd.Timestamp("2026-01-02 02:00", tz="UTC")]
    assert view.forecast["issue_time"].tolist() == [pd.Timestamp("2026-01-01 22:00", tz="UTC")]
    assert view.forecast["value"].tolist() == [21.0]


def test_decision_view_excludes_late_history_and_separates_observed_labels(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
) -> None:
    origin = pd.Timestamp("2026-01-02 00:00", tz="UTC")

    view = DecisionViewBuilder().build(
        climate_frame,
        telemetry_frame,
        origin,
        pd.Timedelta("4h"),
    )

    assert view.telemetry_history["device_id"].tolist() == ["meter-1"]
    assert (view.telemetry_history["event_time"] <= origin).all()
    assert (view.telemetry_history["available_at"] <= origin).all()
    assert view.observed_targets["device_id"].tolist() == ["meter-1"]
    assert view.observed_targets["quality"].tolist() == ["observed"]
    assert (view.observed_targets["available_at"] > origin).all()
    assert view.observed_targets["event_time"].tolist() == [view.target_time]


@pytest.mark.parametrize(
    "origin",
    [
        pd.Timestamp("2026-01-02 00:00"),
        pd.Timestamp("2026-01-02 08:00", tz="Asia/Shanghai"),
        pd.Timestamp(
            "2026-01-02 00:00",
            tz=timezone(timedelta(hours=1), name="UTC"),
        ),
        pd.NaT,
        "2026-01-02T00:00:00Z",
        [pd.Timestamp("2026-01-02 00:00", tz="UTC")],
    ],
)
def test_decision_view_rejects_non_exact_utc_origin(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
    origin: Any,
) -> None:
    with pytest.raises(ValueError, match="origin.*UTC pandas Timestamp"):
        DecisionViewBuilder().build(
            climate_frame,
            telemetry_frame,
            origin,
            pd.Timedelta("4h"),
        )


@pytest.mark.parametrize(
    "horizon",
    [pd.Timedelta(0), pd.Timedelta("-1ns"), pd.NaT, "4h", 4, [pd.Timedelta("4h")]],
)
def test_decision_view_rejects_non_positive_or_non_timedelta_horizon(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
    horizon: Any,
) -> None:
    with pytest.raises(ValueError, match="horizon.*positive pandas Timedelta"):
        DecisionViewBuilder().build(
            climate_frame,
            telemetry_frame,
            pd.Timestamp("2026-01-02 00:00", tz="UTC"),
            horizon,
        )


def test_empty_legal_decision_frames_are_valid(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
) -> None:
    view = DecisionViewBuilder().build(
        climate_frame,
        telemetry_frame,
        pd.Timestamp("2025-12-31 00:00", tz="UTC"),
        pd.Timedelta("1h"),
    )

    assert view.forecast.empty
    assert view.telemetry_history.empty
    assert view.observed_targets.empty


def test_decision_view_returns_copies_without_mutating_contract_frames(
    climate_frame: ClimateForecastFrame,
    telemetry_frame: DCTelemetryFrame,
) -> None:
    climate_before = climate_frame.to_pandas(copy=False).copy(deep=True)
    telemetry_before = telemetry_frame.to_pandas(copy=False).copy(deep=True)
    view = DecisionViewBuilder().build(
        climate_frame,
        telemetry_frame,
        pd.Timestamp("2026-01-02 00:00", tz="UTC"),
        pd.Timedelta("4h"),
    )

    view.forecast.loc[:, "value"] = -1.0
    view.telemetry_history.loc[:, "value"] = -2.0
    view.observed_targets.loc[:, "value"] = -3.0

    pd.testing.assert_frame_equal(climate_frame.to_pandas(copy=False), climate_before)
    pd.testing.assert_frame_equal(telemetry_frame.to_pandas(copy=False), telemetry_before)


def test_decision_view_deeply_isolates_mutable_object_cells() -> None:
    forecast_source = _MutableHashableId("forecast-source")
    history_device = _MutableHashableId("history-device")
    target_device = _MutableHashableId("target-device")
    climate = ClimateForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"],
                "issue_time": pd.to_datetime(["2026-01-01 22:00Z"]),
                "available_at": pd.to_datetime(["2026-01-01 22:05Z"]),
                "valid_time": pd.to_datetime(["2026-01-02 02:00Z"]),
                "variable": ["air_temperature"],
                "value": [21.0],
                "unit": ["degC"],
                "source": [forecast_source],
                "quantile": [pd.NA],
                "member": [pd.NA],
            }
        )
    )
    telemetry = DCTelemetryFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1", "dc-1"],
                "device_id": [history_device, target_device],
                "event_time": pd.to_datetime(["2026-01-01 23:00Z", "2026-01-02 04:00Z"]),
                "available_at": pd.to_datetime(["2026-01-01 23:05Z", "2026-01-02 04:05Z"]),
                "metric": ["total_power", "total_power"],
                "value": [100.0, 110.0],
                "unit": ["kW", "kW"],
                "quality": ["observed", "observed"],
            }
        )
    )
    view = DecisionViewBuilder().build(
        climate,
        telemetry,
        pd.Timestamp("2026-01-02 00:00", tz="UTC"),
        pd.Timedelta("4h"),
    )

    returned_forecast_source = view.forecast.iloc[0]["source"]
    returned_history_device = view.telemetry_history.iloc[0]["device_id"]
    returned_target_device = view.observed_targets.iloc[0]["device_id"]
    returned_forecast_source.payload.append("returned")
    returned_history_device.payload.append("returned")
    returned_target_device.payload.append("returned")

    source_forecast_source = climate.to_pandas(copy=False).iloc[0]["source"]
    source_devices = {
        device.label: device for device in telemetry.to_pandas(copy=False)["device_id"].tolist()
    }
    assert source_forecast_source.payload == ["source"]
    assert source_devices["history-device"].payload == ["source"]
    assert source_devices["target-device"].payload == ["source"]
