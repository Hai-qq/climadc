from dataclasses import FrozenInstanceError
from typing import Any

import pandas as pd
import pytest

from climadc.contracts.frames import (
    ClimateForecastFrame,
    DCTelemetryFrame,
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
