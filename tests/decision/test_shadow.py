from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.decision import DecisionConstraints, ShadowScheduler
from climadc.errors import ContractError


def prediction_frame(
    values: list[float],
    *,
    valid_times: pd.DatetimeIndex | None = None,
    site_ids: list[str] | None = None,
    issue_times: list[pd.Timestamp] | None = None,
    targets: list[str] | None = None,
    units: list[str] | None = None,
    model_ids: list[str] | None = None,
    quantiles: list[object] | None = None,
) -> PredictionFrame:
    count = len(values)
    slots = (
        valid_times
        if valid_times is not None
        else pd.date_range("2026-01-01 01:00Z", periods=count, freq="h")
    )
    return PredictionFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": site_ids or ["dc-1"] * count,
                "issue_time": issue_times or [pd.Timestamp("2026-01-01 00:00Z")] * count,
                "valid_time": slots,
                "target": targets or ["cost_proxy"] * count,
                "value": values,
                "unit": units or ["dimensionless"] * count,
                "model_id": model_ids or ["model-1"] * count,
                "quantile": quantiles or [pd.NA] * count,
            }
        )
    )


def workload_frame(
    events: list[str | pd.Timestamp],
    demands: list[float],
    fractions: list[float],
    *,
    deadlines: list[object] | None = None,
    site_ids: list[str] | None = None,
    resource_types: list[str] | None = None,
    units: list[str] | None = None,
) -> WorkloadFrame:
    count = len(events)
    event_times = pd.to_datetime(events, utc=True)
    if deadlines is None:
        deadline_series = pd.Series([pd.NaT] * count, dtype="datetime64[ns, UTC]")
    else:
        deadline_series = pd.to_datetime(pd.Series(deadlines), utc=True)
    return WorkloadFrame.from_pandas(
        pd.DataFrame(
            {
                "job_id": [f"job-{index}" for index in range(count)],
                "site_id": site_ids or ["dc-1"] * count,
                "event_time": event_times,
                "available_at": event_times,
                "deadline": deadline_series,
                "resource_type": resource_types or ["gpu_energy"] * count,
                "demand": demands,
                "unit": units or ["kWh"] * count,
                "flexible_fraction": fractions,
            }
        )
    )


def test_shadow_scheduler_conserves_flexible_energy() -> None:
    forecast = prediction_frame([4.0, 1.0, 3.0])
    workload = workload_frame(
        ["2026-01-01 01:00Z", "2026-01-01 02:00Z", "2026-01-01 03:00Z"],
        [10.0, 10.0, 10.0],
        [0.5, 0.5, 0.5],
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(
            flexible_fraction=0.4,
            max_shift_multiplier=2.0,
            peak_penalty=4.0,
            risk_penalty=0.25,
        ),
    )

    assert result.feasible
    assert result.schedule["flexible_before"].sum() == pytest.approx(
        result.schedule["flexible_after"].sum(), abs=1e-8
    )
    assert result.metrics["energy_conservation_error"] <= 1e-8


def test_schedule_has_deterministic_columns_metrics_and_slot_order() -> None:
    forecast = prediction_frame([3.0, 1.0])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [8.0, 8.0], [0.5, 0.5])

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert list(result.schedule.columns) == [
        "site_id",
        "valid_time",
        "forecast_value",
        "fixed",
        "flexible_before",
        "flexible_after",
        "total_before",
        "total_after",
        "capacity",
    ]
    assert result.schedule["valid_time"].is_monotonic_increasing
    assert set(result.metrics) == {
        "baseline_peak",
        "scheduled_peak",
        "peak_change",
        "baseline_cost_index",
        "scheduled_cost_index",
        "cost_index_change",
        "baseline_risk_exposure",
        "scheduled_risk_exposure",
        "risk_exposure_change",
        "energy_conservation_error",
        "solver_status",
    }
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_effective_flexibility_honors_row_fraction_and_run_cap() -> None:
    forecast = prediction_frame([2.0, 1.0])
    workload = workload_frame(
        ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
        [10.0, 20.0],
        [0.8, 0.1],
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(flexible_fraction=0.25),
    )

    assert result.schedule["flexible_before"].sum() == pytest.approx(4.5)
    assert result.schedule["fixed"].sum() == pytest.approx(25.5)


def test_between_slot_event_maps_baseline_to_first_future_slot() -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:30Z"], [10.0], [0.0])

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert result.feasible
    assert result.schedule["total_before"].tolist() == [0.0, 10.0]
    assert result.schedule["fixed"].tolist() == [0.0, 10.0]


def test_deadline_window_prevents_flexible_energy_from_moving_later() -> None:
    forecast = prediction_frame([10.0, 5.0, 0.0])
    workload = workload_frame(
        ["2026-01-01 00:30Z", "2026-01-01 02:00Z", "2026-01-01 03:00Z"],
        [10.0, 10.0, 10.0],
        [1.0, 0.0, 0.0],
        deadlines=["2026-01-01 02:00Z", pd.NaT, pd.NaT],
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(flexible_fraction=1.0, max_shift_multiplier=2.0),
    )

    assert result.feasible
    assert result.schedule.loc[2, "flexible_after"] == pytest.approx(0.0)
    assert result.schedule.loc[:1, "flexible_after"].sum() == pytest.approx(10.0)


def test_impossible_deadline_returns_infeasible_baseline_result() -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(
        ["2026-01-01 00:30Z"],
        [10.0],
        [0.5],
        deadlines=["2026-01-01 00:45Z"],
    )

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert not result.feasible
    assert "deadline" in result.violations[0]
    assert result.schedule["total_after"].tolist() == result.schedule["total_before"].tolist()
    assert result.metrics["solver_status"] < 0


def test_event_after_horizon_returns_infeasible_result() -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 03:00Z"], [10.0], [0.5])

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert not result.feasible
    assert any("horizon" in violation for violation in result.violations)


def test_solver_failure_returns_diagnostics_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])

    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            message="forced infeasible",
            x=None,
        ),
    )

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert not result.feasible
    assert result.metrics["solver_status"] == 2.0
    assert any("forced infeasible" in violation for violation in result.violations)


def test_solver_post_verification_failure_returns_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="bad numerical solution",
            x=np.zeros(5),
        ),
    )

    result = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    assert not result.feasible
    assert "solver verification failed" in result.violations[0]
    assert result.schedule["total_after"].tolist() == result.schedule["total_before"].tolist()


def test_constant_forecast_has_zero_risk_and_negative_values_are_allowed() -> None:
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])
    constant = ShadowScheduler().solve(
        prediction_frame([-2.0, -2.0]), workload, DecisionConstraints()
    )
    negative = ShadowScheduler().solve(
        prediction_frame([-1.0, -5.0]), workload, DecisionConstraints()
    )

    assert constant.feasible and negative.feasible
    assert constant.metrics["baseline_risk_exposure"] == 0.0
    assert constant.metrics["scheduled_risk_exposure"] == 0.0
    assert np.isfinite(negative.metrics["scheduled_cost_index"])


def test_constant_non_null_quantile_representation_is_accepted() -> None:
    forecast = prediction_frame([1.0, 2.0], quantiles=[0.9, 0.9])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])

    assert ShadowScheduler().solve(forecast, workload, DecisionConstraints()).feasible


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.assign(site_id=["dc-1", "dc-2"]),
        lambda frame: frame.assign(
            issue_time=pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 00:30Z"])
        ),
        lambda frame: frame.assign(model_id=["model-1", "model-2"]),
        lambda frame: frame.assign(target=["cost_proxy", "carbon_proxy"]),
        lambda frame: frame.assign(unit=["dimensionless", "percent"]),
        lambda frame: frame.assign(quantile=[pd.NA, 0.9]),
        lambda frame: frame.assign(
            valid_time=pd.to_datetime(["2026-01-01 01:00Z"] * 2),
            model_id=["model-1", "model-2"],
        ),
    ],
)
def test_rejects_ambiguous_or_mixed_forecast_series(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    base = prediction_frame([1.0, 2.0]).to_pandas()
    malformed = PredictionFrame.from_pandas(mutate(base))
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])

    with pytest.raises(ContractError, match="forecast"):
        ShadowScheduler().solve(malformed, workload, DecisionConstraints())


def test_rejects_non_future_forecast_slot() -> None:
    forecast = prediction_frame(
        [1.0, 2.0],
        valid_times=pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 01:00Z"]),
    )
    workload = workload_frame(["2026-01-01 00:30Z"], [10.0], [0.5])

    with pytest.raises(ContractError, match="strictly after"):
        ShadowScheduler().solve(forecast, workload, DecisionConstraints())


def test_rejects_empty_or_wrong_contract_inputs() -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])

    with pytest.raises(ContractError, match="forecast must be"):
        ShadowScheduler().solve(pd.DataFrame(), workload, DecisionConstraints())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="forecast must contain"):
        ShadowScheduler().solve(PredictionFrame(pd.DataFrame()), workload, DecisionConstraints())
    with pytest.raises(ContractError, match="workload must be"):
        ShadowScheduler().solve(forecast, pd.DataFrame(), DecisionConstraints())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="workload must contain"):
        ShadowScheduler().solve(forecast, WorkloadFrame(pd.DataFrame()), DecisionConstraints())
    with pytest.raises(ContractError, match="constraints"):
        ShadowScheduler().solve(forecast, workload, object())  # type: ignore[arg-type]


def test_rejects_non_utc_backing_data_and_duplicate_valid_times() -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    forecast.to_pandas(copy=False)["issue_time"] = pd.Series(
        [pd.Timestamp("2026-01-01 08:00", tz="Asia/Shanghai")] * 2
    )
    with pytest.raises(ContractError, match="exact UTC"):
        ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    duplicate = prediction_frame([1.0, 2.0])
    duplicate.to_pandas(copy=False)["valid_time"] = pd.to_datetime(
        ["2026-01-01 01:00Z", "2026-01-01 01:00Z"]
    )
    with pytest.raises(ContractError, match="unique valid_time"):
        ShadowScheduler().solve(duplicate, workload, DecisionConstraints())


def test_rejects_constant_workload_site_that_does_not_match_forecast() -> None:
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5], site_ids=["dc-other"])

    with pytest.raises(ContractError, match="match forecast"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


@pytest.mark.parametrize("input_name", ["forecast", "workload"])
def test_rejects_non_finite_values_even_if_contract_backing_was_mutated(
    input_name: str,
) -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    if input_name == "forecast":
        forecast.to_pandas(copy=False).loc[0, "value"] = float("inf")
    else:
        workload.to_pandas(copy=False).loc[0, "demand"] = float("nan")

    with pytest.raises(ContractError, match=rf"{input_name}.*finite"):
        ShadowScheduler().solve(forecast, workload, DecisionConstraints())


@pytest.mark.parametrize(
    "workload",
    [
        workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
            [10.0, 10.0],
            [0.5, 0.5],
            site_ids=["dc-1", "dc-2"],
        ),
        workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
            [10.0, 10.0],
            [0.5, 0.5],
            resource_types=["gpu_energy", "cpu_energy"],
        ),
        workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
            [10.0, 10.0],
            [0.5, 0.5],
            units=["kWh", "MWh"],
        ),
        workload_frame(["2026-01-01 01:00Z"], [-1.0], [0.5]),
    ],
)
def test_rejects_malformed_workload_series(workload: WorkloadFrame) -> None:
    with pytest.raises(ContractError, match="workload"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


def test_solve_does_not_mutate_inputs_or_share_schedule_between_runs() -> None:
    forecast = prediction_frame([2.0, 1.0])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])
    forecast_before = forecast.to_pandas()
    workload_before = workload.to_pandas()

    first = ShadowScheduler().solve(forecast, workload, DecisionConstraints())
    first.schedule.loc[0, "total_after"] = 999.0
    second = ShadowScheduler().solve(forecast, workload, DecisionConstraints())

    pd.testing.assert_frame_equal(forecast.to_pandas(), forecast_before)
    pd.testing.assert_frame_equal(workload.to_pandas(), workload_before)
    assert second.schedule.loc[0, "total_after"] != 999.0
