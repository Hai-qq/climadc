from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

from climadc.contracts.frames import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
    PredictionFrame,
    WorkloadFrame,
)
from climadc.errors import ContractError


def _climate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "issue_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available_at": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "valid_time": [pd.Timestamp("2026-01-01 04:00", tz="Asia/Shanghai")],
            "variable": ["air_temperature"],
            "value": [30.0],
            "unit": ["degC"],
            "source": ["fixture"],
            "quantile": [pd.NA],
            "member": [pd.NA],
        }
    )


def _telemetry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "device_id": [pd.NA],
            "event_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available_at": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "metric": ["total_power"],
            "value": [100.0],
            "unit": ["kW"],
            "quality": ["observed"],
        }
    )


def _workload_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "job_id": [pd.NA],
            "event_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available_at": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "deadline": pd.Series([pd.NaT], dtype=object),
            "resource_type": ["GPU"],
            "demand": [4.0],
            "unit": ["hour"],
            "flexible_fraction": [0.5],
        }
    )


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "model_id": ["model-1"],
            "issue_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "valid_time": [pd.Timestamp("2026-01-01 04:00", tz="Asia/Shanghai")],
            "target": ["total_power"],
            "value": [100.0],
            "unit": ["kW"],
            "quantile": [pd.NA],
        }
    )


def _grid_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "region_id": ["GB-13"],
            "issue_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available_at": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "valid_time": [pd.Timestamp("2026-01-01 04:00", tz="Asia/Shanghai")],
            "signal": ["carbon_intensity"],
            "value": [180.0],
            "unit": ["gCO2e / kWh"],
            "source": ["fixture"],
            "quality": ["forecast"],
            "quantile": [pd.NA],
        }
    )


def _grid_realized_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "region_id": ["GB-13"],
            "issue_time": pd.Series([pd.NaT], dtype="datetime64[ns, Asia/Shanghai]"),
            "available_at": [pd.Timestamp("2026-01-01 04:05", tz="Asia/Shanghai")],
            "valid_time": [pd.Timestamp("2026-01-01 04:00", tz="Asia/Shanghai")],
            "signal": ["carbon_intensity"],
            "value": [190.0],
            "unit": ["gCO2e / kWh"],
            "source": ["fixture"],
            "quality": ["estimated"],
            "quantile": [pd.NA],
        }
    )


def _flexible_workload_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["batch-1"],
            "site_id": ["dc-1"],
            "release_time": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available_at": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "deadline": [pd.Timestamp("2026-01-01 04:00", tz="Asia/Shanghai")],
            "energy": [8.0],
            "energy_unit": ["kWh"],
            "max_power": [4.0],
            "power_unit": ["kW"],
            "preemptible": [True],
            "priority": [1.0],
        }
    )


FRAME_CASES: tuple[tuple[type[Any], Any], ...] = (
    (ClimateForecastFrame, _climate_frame),
    (DCTelemetryFrame, _telemetry_frame),
    (WorkloadFrame, _workload_frame),
    (PredictionFrame, _prediction_frame),
)


def test_climate_frame_rejects_naive_and_late_forecast() -> None:
    naive = pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "issue_time": [pd.Timestamp("2026-01-01 00:00")],
            "available_at": [pd.Timestamp("2026-01-01 00:05")],
            "valid_time": [pd.Timestamp("2026-01-01 04:00")],
            "variable": ["air_temperature"],
            "value": [30.0],
            "unit": ["degC"],
            "source": ["fixture"],
            "quantile": [pd.NA],
            "member": [pd.NA],
        }
    )
    with pytest.raises(ContractError, match="timezone-aware"):
        ClimateForecastFrame.from_pandas(naive)


def test_telemetry_requires_event_before_availability() -> None:
    frame = pd.DataFrame(
        {
            "site_id": ["dc-1"],
            "device_id": ["meter-1"],
            "event_time": [pd.Timestamp("2026-01-01 01:00", tz="UTC")],
            "available_at": [pd.Timestamp("2026-01-01 00:59", tz="UTC")],
            "metric": ["total_power"],
            "value": [100.0],
            "unit": ["kW"],
            "quality": ["observed"],
        }
    )
    with pytest.raises(ContractError, match="event_time <= available_at"):
        DCTelemetryFrame.from_pandas(frame)


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
@pytest.mark.parametrize("schema_change", ["missing", "extra"])
def test_frames_require_exact_columns(
    wrapper: type[Any], frame_factory: Any, schema_change: str
) -> None:
    frame = frame_factory()
    if schema_change == "missing":
        frame = frame.drop(columns=frame.columns[-1])
    else:
        frame["lead_time"] = pd.Timedelta(hours=4)

    with pytest.raises(ContractError, match="exact columns"):
        wrapper.from_pandas(frame)


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
def test_frames_normalize_aware_timestamps_to_utc(wrapper: type[Any], frame_factory: Any) -> None:
    normalized = wrapper.from_pandas(frame_factory()).to_pandas()
    timestamp_columns = [
        column
        for column in ("issue_time", "event_time", "available_at", "valid_time", "deadline")
        if column in normalized.columns
    ]

    for column in timestamp_columns:
        assert str(normalized[column].dtype).endswith("UTC]")


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
def test_frames_reject_duplicate_keys_with_null_key_values(
    wrapper: type[Any], frame_factory: Any
) -> None:
    frame = pd.concat([frame_factory(), frame_factory()], ignore_index=True)

    with pytest.raises(ContractError, match=r"duplicate key.*2 offending row"):
        wrapper.from_pandas(frame)


def test_climate_requires_ordered_timestamps() -> None:
    frame = _climate_frame()
    frame.loc[0, "available_at"] = pd.Timestamp("2026-01-01 05:00", tz="Asia/Shanghai")

    with pytest.raises(ContractError, match="issue_time <= available_at <= valid_time"):
        ClimateForecastFrame.from_pandas(frame)


def test_workload_requires_ordered_availability_and_deadline() -> None:
    before_availability = _workload_frame()
    before_availability.loc[0, "available_at"] = pd.Timestamp(
        "2025-12-31 23:59", tz="Asia/Shanghai"
    )
    with pytest.raises(ContractError, match="event_time <= available_at"):
        WorkloadFrame.from_pandas(before_availability)

    before_event = _workload_frame()
    before_event.loc[0, "deadline"] = pd.Timestamp("2025-12-31 23:59", tz="Asia/Shanghai")
    with pytest.raises(ContractError, match="deadline >= event_time"):
        WorkloadFrame.from_pandas(before_event)


def test_workload_rejects_naive_non_null_deadline() -> None:
    frame = _workload_frame()
    frame.loc[0, "deadline"] = pd.Timestamp("2026-01-01 01:00")

    with pytest.raises(ContractError, match="deadline.*timezone-aware"):
        WorkloadFrame.from_pandas(frame)


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
def test_frames_reject_non_numeric_values(wrapper: type[Any], frame_factory: Any) -> None:
    frame = frame_factory()
    value_column = "demand" if "demand" in frame.columns else "value"
    frame[value_column] = frame[value_column].astype(object)
    frame.loc[0, value_column] = "not-a-number"

    with pytest.raises(ContractError, match=f"{value_column} must be numeric"):
        wrapper.from_pandas(frame)


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf")])
def test_frames_reject_non_finite_numeric_values(
    wrapper: type[Any], frame_factory: Any, non_finite: float
) -> None:
    frame = frame_factory()
    value_column = "demand" if "demand" in frame.columns else "value"
    frame.loc[0, value_column] = non_finite

    with pytest.raises(
        ContractError,
        match=rf"{wrapper.__name__}: {value_column} must be finite.*1 offending row",
    ):
        wrapper.from_pandas(frame)


@pytest.mark.parametrize("flexible_fraction", [-0.01, 1.01, "half", pd.NA])
def test_workload_rejects_invalid_flexible_fraction(flexible_fraction: object) -> None:
    frame = _workload_frame()
    frame["flexible_fraction"] = frame["flexible_fraction"].astype(object)
    frame.loc[0, "flexible_fraction"] = flexible_fraction

    with pytest.raises(ContractError, match="flexible_fraction"):
        WorkloadFrame.from_pandas(frame)


@pytest.mark.parametrize("flexible_fraction", [float("nan"), float("inf"), float("-inf")])
def test_workload_rejects_non_finite_flexible_fraction(flexible_fraction: float) -> None:
    frame = _workload_frame()
    frame.loc[0, "flexible_fraction"] = flexible_fraction

    with pytest.raises(ContractError, match=r"WorkloadFrame: flexible_fraction.*1 offending row"):
        WorkloadFrame.from_pandas(frame)


@pytest.mark.parametrize("quality", ["raw", pd.NA])
def test_telemetry_rejects_invalid_quality(quality: object) -> None:
    frame = _telemetry_frame()
    frame.loc[0, "quality"] = quality

    with pytest.raises(ContractError, match="quality"):
        DCTelemetryFrame.from_pandas(frame)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1, "median"])
@pytest.mark.parametrize(
    ("wrapper", "frame_factory"),
    [
        (ClimateForecastFrame, _climate_frame),
        (PredictionFrame, _prediction_frame),
    ],
)
def test_probabilistic_frames_require_open_interval_quantiles(
    wrapper: type[Any], frame_factory: Any, quantile: object
) -> None:
    frame = frame_factory()
    frame.loc[0, "quantile"] = quantile

    with pytest.raises(ContractError, match=r"quantile.*\(0, 1\)"):
        wrapper.from_pandas(frame)


@pytest.mark.parametrize(("wrapper", "frame_factory"), FRAME_CASES)
def test_frames_validate_unit_groups(wrapper: type[Any], frame_factory: Any) -> None:
    frame = pd.concat([frame_factory(), frame_factory()], ignore_index=True)
    time_column = "valid_time" if "valid_time" in frame.columns else "event_time"
    frame.loc[1, time_column] += pd.Timedelta(hours=1)
    if time_column == "event_time":
        frame.loc[1, "available_at"] += pd.Timedelta(hours=1)
    frame.loc[1, "unit"] = "meter"

    with pytest.raises(ContractError, match="incompatible units"):
        wrapper.from_pandas(frame)


def test_frame_copy_defaults_protect_internal_state_and_false_returns_backing_frame() -> None:
    contract = DCTelemetryFrame.from_pandas(_telemetry_frame())

    copied = contract.to_pandas()
    copied.loc[0, "value"] = 999.0
    assert contract.to_pandas().loc[0, "value"] == 100.0

    backing = contract.to_pandas(copy=False)
    backing.loc[0, "value"] = 200.0
    assert contract.to_pandas(copy=False).loc[0, "value"] == 200.0


def test_frame_wrapper_is_frozen_and_sorts_by_contract_key() -> None:
    early = _prediction_frame()
    late = _prediction_frame()
    late.loc[0, "valid_time"] += pd.Timedelta(hours=1)
    frame = pd.concat([late, early], ignore_index=True)

    contract = PredictionFrame.from_pandas(frame)

    assert contract.to_pandas()["valid_time"].is_monotonic_increasing
    with pytest.raises(FrozenInstanceError):
        contract._frame = frame


def test_frame_rejects_non_scalar_timestamp_with_contract_error() -> None:
    frame = _climate_frame()
    frame["issue_time"] = pd.Series(
        [["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]],
        dtype=object,
    )

    with pytest.raises(
        ContractError,
        match=r"ClimateForecastFrame: issue_time.*timezone-aware.*1 offending row",
    ):
        ClimateForecastFrame.from_pandas(frame)


def test_timestamp_error_count_is_positional_with_duplicate_input_index() -> None:
    frame = pd.concat([_climate_frame(), _climate_frame()], ignore_index=True)
    frame.loc[1, "valid_time"] += pd.Timedelta(hours=1)
    frame["issue_time"] = frame["issue_time"].astype(object)
    frame.loc[1, "issue_time"] = pd.Timestamp("2026-01-01 00:00")
    frame.index = [7, 7]

    with pytest.raises(
        ContractError,
        match=r"ClimateForecastFrame: issue_time.*1 offending row$",
    ):
        ClimateForecastFrame.from_pandas(frame)


def test_quantile_error_count_is_positional_with_duplicate_input_index() -> None:
    frame = pd.concat([_prediction_frame(), _prediction_frame()], ignore_index=True)
    frame.loc[1, "valid_time"] += pd.Timedelta(hours=1)
    frame.loc[0, "quantile"] = 0.5
    frame.loc[1, "quantile"] = 1.0
    frame.index = [9, 9]

    with pytest.raises(
        ContractError,
        match=r"PredictionFrame: quantile.*1 offending row$",
    ):
        PredictionFrame.from_pandas(frame)


def test_prediction_requires_issue_time_not_after_valid_time() -> None:
    frame = _prediction_frame()
    frame.loc[0, "issue_time"] = pd.Timestamp("2026-01-01 05:00", tz="Asia/Shanghai")

    with pytest.raises(ContractError, match="issue_time <= valid_time"):
        PredictionFrame.from_pandas(frame)


def test_grid_signal_accepts_forecast_and_realized_rows_with_distinct_time_semantics() -> None:
    frame = pd.concat([_grid_realized_frame(), _grid_forecast_frame()], ignore_index=True)

    result = GridSignalFrame.from_pandas(frame).to_pandas()

    assert str(result["issue_time"].dtype).endswith("UTC]")
    assert str(result["available_at"].dtype).endswith("UTC]")
    assert str(result["valid_time"].dtype).endswith("UTC]")
    assert set(result["quality"]) == {"forecast", "estimated"}
    assert result.loc[result["quality"] == "estimated", "issue_time"].isna().all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.__setitem__("issue_time", pd.Series([pd.NaT], dtype=object)),
            "forecast rows require issue_time",
        ),
        (
            lambda frame: frame.__setitem__(
                "available_at", [pd.Timestamp("2025-12-31 23:59", tz="Asia/Shanghai")]
            ),
            "issue_time <= available_at <= valid_time",
        ),
        (lambda frame: frame.__setitem__("quality", ["raw"]), "quality"),
        (lambda frame: frame.__setitem__("signal", ["cost_proxy"]), "signal"),
        (lambda frame: frame.__setitem__("value", [-1.0]), "carbon_intensity.*nonnegative"),
    ],
)
def test_grid_forecast_rejects_invalid_semantics(mutate: Any, message: str) -> None:
    frame = _grid_forecast_frame()
    mutate(frame)

    with pytest.raises(ContractError, match=message):
        GridSignalFrame.from_pandas(frame)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.__setitem__(
                "issue_time", [pd.Timestamp("2026-01-01 03:00", tz="Asia/Shanghai")]
            ),
            "realized rows must not set issue_time",
        ),
        (
            lambda frame: frame.__setitem__(
                "available_at", [pd.Timestamp("2026-01-01 03:59", tz="Asia/Shanghai")]
            ),
            "valid_time <= available_at",
        ),
        (lambda frame: frame.__setitem__("quantile", [0.5]), "realized rows.*quantile"),
    ],
)
def test_grid_realized_signal_rejects_invalid_semantics(mutate: Any, message: str) -> None:
    frame = _grid_realized_frame()
    mutate(frame)

    with pytest.raises(ContractError, match=message):
        GridSignalFrame.from_pandas(frame)


@pytest.mark.parametrize(
    ("signal", "unit"),
    [
        ("carbon_intensity", "kW"),
        ("energy_price", "gCO2e / kWh"),
        ("energy_price", "JPY / kWh"),
    ],
)
def test_grid_signal_rejects_units_that_do_not_match_signal(signal: str, unit: str) -> None:
    frame = _grid_forecast_frame()
    frame.loc[0, "signal"] = signal
    frame.loc[0, "unit"] = unit

    with pytest.raises(ContractError, match="unit must be compatible"):
        GridSignalFrame.from_pandas(frame)


@pytest.mark.parametrize(
    ("signal", "unit"),
    [
        ("carbon_intensity", "kgCO2e / MWh"),
        ("energy_price", "GBP / MWh"),
        ("energy_price", "CNY / kWh"),
    ],
)
def test_grid_signal_accepts_supported_convertible_units(signal: str, unit: str) -> None:
    frame = _grid_forecast_frame()
    frame.loc[0, "signal"] = signal
    frame.loc[0, "unit"] = unit

    result = GridSignalFrame.from_pandas(frame).to_pandas()

    assert result.loc[0, "unit"] == unit


def test_grid_signal_rejects_duplicate_realized_keys() -> None:
    frame = pd.concat([_grid_realized_frame(), _grid_realized_frame()], ignore_index=True)

    with pytest.raises(ContractError, match="duplicate key"):
        GridSignalFrame.from_pandas(frame)


def test_flexible_workload_normalizes_times_and_accepts_convertible_units() -> None:
    frame = _flexible_workload_frame()
    frame.loc[0, "energy"] = 0.008
    frame.loc[0, "energy_unit"] = "MWh"
    frame.loc[0, "max_power"] = 0.004
    frame.loc[0, "power_unit"] = "MW"

    result = FlexibleWorkloadFrame.from_pandas(frame).to_pandas()

    assert result.loc[0, "release_time"] == pd.Timestamp("2025-12-31 16:00", tz="UTC")
    assert result.loc[0, "energy_unit"] == "MWh"


def test_flexible_workload_accepts_numpy_boolean_from_object_input() -> None:
    frame = _flexible_workload_frame()
    frame["preemptible"] = pd.Series([np.bool_(True)], dtype=object)

    result = FlexibleWorkloadFrame.from_pandas(frame).to_pandas()

    assert bool(result.loc[0, "preemptible"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("energy", 0.0, "energy must be positive"),
        ("energy", float("inf"), "energy must be finite"),
        ("max_power", 0.0, "max_power must be positive"),
        ("max_power", "fast", "max_power must be numeric"),
        ("priority", -1.0, "priority must be nonnegative"),
        ("preemptible", False, "only preemptible jobs"),
        ("job_id", "", "job_id must be a non-empty string"),
    ],
)
def test_flexible_workload_rejects_invalid_job_fields(
    column: str, value: object, message: str
) -> None:
    frame = _flexible_workload_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value

    with pytest.raises(ContractError, match=message):
        FlexibleWorkloadFrame.from_pandas(frame)


@pytest.mark.parametrize(
    ("column", "unit"),
    [("energy_unit", "kW"), ("power_unit", "kWh")],
)
def test_flexible_workload_rejects_wrong_physical_dimensions(column: str, unit: str) -> None:
    frame = _flexible_workload_frame()
    frame.loc[0, column] = unit

    with pytest.raises(ContractError, match="unit must be compatible"):
        FlexibleWorkloadFrame.from_pandas(frame)


def test_flexible_workload_rejects_impossible_energy_before_deadline() -> None:
    frame = _flexible_workload_frame()
    frame.loc[0, "energy"] = 100.0

    with pytest.raises(ContractError, match="minimum runtime exceeds available window"):
        FlexibleWorkloadFrame.from_pandas(frame)


@pytest.mark.parametrize(
    ("column", "timestamp", "message"),
    [
        ("available_at", "2025-12-31 23:59", "release_time <= available_at <= deadline"),
        ("deadline", "2026-01-01 00:04", "release_time <= available_at <= deadline"),
    ],
)
def test_flexible_workload_rejects_invalid_time_order(
    column: str, timestamp: str, message: str
) -> None:
    frame = _flexible_workload_frame()
    frame.loc[0, column] = pd.Timestamp(timestamp, tz="Asia/Shanghai")

    with pytest.raises(ContractError, match=message):
        FlexibleWorkloadFrame.from_pandas(frame)


def test_flexible_workload_rejects_duplicate_job_key() -> None:
    frame = pd.concat([_flexible_workload_frame(), _flexible_workload_frame()], ignore_index=True)

    with pytest.raises(ContractError, match="duplicate key"):
        FlexibleWorkloadFrame.from_pandas(frame)
