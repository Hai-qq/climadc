import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.decision import DecisionConstraints, ShadowScheduler


def prediction_frame(values: list[float], valid_times: pd.DatetimeIndex) -> PredictionFrame:
    count = len(values)
    return PredictionFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * count,
                "issue_time": [pd.Timestamp("2026-01-01 00:00Z")] * count,
                "valid_time": valid_times,
                "target": ["cost_proxy"] * count,
                "value": values,
                "unit": ["dimensionless"] * count,
                "model_id": ["model-1"] * count,
                "quantile": [pd.NA] * count,
            }
        )
    )


def workload_frame(
    events: list[object],
    demands: list[float],
    fractions: list[float],
    deadlines: list[object] | None = None,
) -> WorkloadFrame:
    count = len(events)
    event_times = pd.to_datetime(events, utc=True)
    deadline_values = (
        pd.Series([pd.NaT] * count, dtype="datetime64[ns, UTC]")
        if deadlines is None
        else pd.to_datetime(pd.Series(deadlines), utc=True)
    )
    return WorkloadFrame.from_pandas(
        pd.DataFrame(
            {
                "job_id": [f"job-{index}" for index in range(count)],
                "site_id": ["dc-1"] * count,
                "event_time": event_times,
                "available_at": event_times,
                "deadline": deadline_values,
                "resource_type": ["gpu_energy"] * count,
                "demand": demands,
                "unit": ["kWh"] * count,
                "flexible_fraction": fractions,
            }
        )
    )


@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    values=st.lists(
        st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=6,
    ),
    demands=st.lists(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=6,
    ),
    row_fraction=st.floats(min_value=0.0, max_value=1.0),
    run_fraction=st.floats(min_value=0.0, max_value=1.0),
    multiplier=st.floats(min_value=1.0, max_value=3.0),
)
def test_generated_feasible_schedules_conserve_energy_and_capacity(
    values: list[float],
    demands: list[float],
    row_fraction: float,
    run_fraction: float,
    multiplier: float,
) -> None:
    count = min(len(values), len(demands))
    slots = pd.date_range("2026-01-01 01:00Z", periods=count, freq="h")
    forecast = prediction_frame(values[:count], slots)
    workload = workload_frame(
        list(slots),
        demands[:count],
        [row_fraction] * count,
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(
            flexible_fraction=run_fraction,
            max_shift_multiplier=multiplier,
            peak_penalty=1.0,
            risk_penalty=0.5,
        ),
    )

    assert result.feasible
    assert result.schedule["flexible_after"].sum() == pytest.approx(
        sum(demands[:count]) * min(row_fraction, run_fraction), abs=1e-8
    )
    assert result.schedule["total_after"].sum() == pytest.approx(sum(demands[:count]), abs=1e-8)
    assert (result.schedule["total_after"] <= result.schedule["capacity"] + 1e-8).all()
    assert result.schedule["total_after"].max() <= result.metrics["scheduled_peak"] + 1e-8
    assert all(np.isfinite(value) for value in result.metrics.values())


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    first=st.floats(min_value=0.1, max_value=25.0, allow_nan=False, allow_infinity=False),
    second=st.floats(min_value=0.1, max_value=25.0, allow_nan=False, allow_infinity=False),
)
def test_disjoint_deadline_windows_conserve_each_rows_energy(first: float, second: float) -> None:
    slots = pd.date_range("2026-01-01 01:00Z", periods=2, freq="h")
    forecast = prediction_frame([10.0, 0.0], slots)
    workload = workload_frame(
        ["2026-01-01 00:30Z", "2026-01-01 02:00Z"],
        [first, second],
        [1.0, 1.0],
        deadlines=["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(
            flexible_fraction=1.0,
            max_shift_multiplier=2.0,
            peak_penalty=0.0,
            risk_penalty=0.0,
        ),
    )

    assert result.feasible
    assert result.schedule["flexible_after"].tolist() == pytest.approx([first, second])
