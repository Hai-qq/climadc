import warnings
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.decision import DecisionConstraints, ShadowScheduler
from climadc.errors import ContractError

SCHEDULE_COLUMNS = [
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


def assert_finite_infeasible(result: object, violation: str) -> None:
    assert not result.feasible  # type: ignore[attr-defined]
    assert any(violation in item for item in result.violations)  # type: ignore[attr-defined]
    assert list(result.schedule.columns) == SCHEDULE_COLUMNS  # type: ignore[attr-defined]
    numeric = result.schedule.select_dtypes(include=[np.number])  # type: ignore[attr-defined]
    assert np.isfinite(numeric.to_numpy()).all()
    assert all(np.isfinite(value) for value in result.metrics.values())  # type: ignore[attr-defined]


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

    assert list(result.schedule.columns) == SCHEDULE_COLUMNS
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


@pytest.mark.parametrize(
    ("solution", "violation"),
    [
        (np.array([5.0, 0.0, 0.0, 5.0, 10.0, 99.0]), "length"),
        (np.array([[5.0, 0.0, 0.0, 5.0, 10.0]]), "one-dimensional"),
        (np.array([5.0, np.nan, 0.0, 5.0, 10.0]), "finite"),
        (np.array([-1.0, 6.0, 0.0, 5.0, 11.0]), "nonnegative"),
        (np.array([5.0, 0.0, 0.0, 5.0, -1.0]), "nonnegative"),
        (np.array([5.0, 0.0, 1.0, 4.0, 11.0]), "ineligible"),
        (np.array([4.0, 0.0, 0.0, 5.0, 10.0]), "equality"),
        (np.array([5.0, 0.0, 0.0, 5.0, 9.0]), "peak"),
    ],
)
def test_solver_success_requires_complete_solution_postconditions(
    monkeypatch: pytest.MonkeyPatch,
    solution: np.ndarray,
    violation: str,
) -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(
        ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
        [10.0, 10.0],
        [0.5, 0.5],
        deadlines=[pd.NaT, "2026-01-01 02:00Z"],
    )
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="invalid success",
            x=solution,
        ),
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(flexible_fraction=0.5),
    )

    assert_finite_infeasible(result, violation)


def test_solver_success_rejects_capacity_upper_bound_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecast = prediction_frame([1.0, 0.0])
    workload = workload_frame(
        ["2026-01-01 01:00Z", "2026-01-01 01:00Z"],
        [10.0, 10.0],
        [1.0, 1.0],
    )
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="invalid capacity",
            x=np.array([0.0, 10.0, 0.0, 10.0, 20.0]),
        ),
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(flexible_fraction=1.0, max_shift_multiplier=1.0),
    )

    assert_finite_infeasible(result, "capacity")


def test_solver_tolerance_only_cleans_near_zero_bound_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecast = prediction_frame([1.0, 2.0])
    workload = workload_frame(
        ["2026-01-01 01:00Z", "2026-01-01 02:00Z"],
        [10.0, 10.0],
        [0.5, 0.5],
        deadlines=[pd.NaT, "2026-01-01 02:00Z"],
    )
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="tolerance noise",
            x=np.array([5.0 - 1e-10, 1e-10, 1e-10, 5.0 - 1e-10, 10.0]),
        ),
    )

    result = ShadowScheduler().solve(
        forecast,
        workload,
        DecisionConstraints(flexible_fraction=0.5),
    )

    assert result.feasible
    assert result.schedule["flexible_after"].tolist() == pytest.approx([5.0, 5.0], abs=1e-8)


def test_linprog_numeric_value_error_returns_finite_numeric_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_finite_lp(*args: object, **kwargs: object) -> object:
        raise ValueError("HiGHS numeric range")

    monkeypatch.setattr("climadc.decision.shadow.linprog", reject_finite_lp)
    result = ShadowScheduler().solve(
        prediction_frame([1.0, 2.0]),
        workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5]),
        DecisionConstraints(),
    )

    assert_finite_infeasible(result, "numeric")


def test_solver_success_rejects_non_numeric_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="non-numeric",
            x=[object(), object(), object()],
        ),
    )
    result = ShadowScheduler().solve(
        prediction_frame([1.0, 2.0]),
        workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5]),
        DecisionConstraints(),
    )

    assert_finite_infeasible(result, "numeric")


def test_solver_accumulated_tolerance_error_fails_aggregate_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="accumulated tolerance",
            x=np.array([4.999999994, 0.0, 0.0, 4.999999994, 10.0]),
        ),
    )
    result = ShadowScheduler().solve(
        prediction_frame([1.0, 2.0]),
        workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5]),
        DecisionConstraints(flexible_fraction=0.5),
    )

    assert_finite_infeasible(result, "aggregate energy")


def test_solver_failure_with_unrepresentable_metrics_adds_numeric_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            message="forced failure",
            x=None,
        ),
    )
    result = ShadowScheduler().solve(
        prediction_frame([1e308, 1e308]),
        workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.0, 0.0]),
        DecisionConstraints(),
    )

    assert_finite_infeasible(result, "numeric")
    assert any("forced failure" in violation for violation in result.violations)


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


def test_opposite_near_max_forecasts_use_overflow_safe_risk_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near_max = np.finfo(float).max * 0.99
    forecast = prediction_frame([-near_max, near_max])
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [1.0, 1.0], [0.5, 0.5])
    monkeypatch.setattr(
        "climadc.decision.shadow.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            message="stable normalization",
            x=np.array([0.5, 0.0, 0.0, 0.5, 1.0]),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ShadowScheduler().solve(
            forecast,
            workload,
            DecisionConstraints(flexible_fraction=0.5),
        )

    assert result.feasible
    assert all(np.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize("case", ["metric", "aggregate", "capacity", "objective"])
def test_extreme_finite_arithmetic_returns_numeric_infeasible_without_warnings(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    near_max = np.finfo(float).max * 0.75
    constraints = DecisionConstraints()
    if case == "metric":
        forecast = prediction_frame([1e308, 1e308])
        workload = workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.0, 0.0]
        )
        monkeypatch.setattr(
            "climadc.decision.shadow.linprog",
            lambda *args, **kwargs: SimpleNamespace(
                success=True,
                status=0,
                message="metric overflow",
                x=np.array([0.0, 0.0, 0.0, 0.0, 10.0]),
            ),
        )
    elif case == "aggregate":
        forecast = prediction_frame([1.0, 2.0])
        workload = workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 01:00Z"],
            [near_max, near_max],
            [0.0, 0.0],
        )
    elif case == "capacity":
        forecast = prediction_frame([1.0, 2.0])
        workload = workload_frame(["2026-01-01 01:00Z"], [near_max], [0.0])
        constraints = DecisionConstraints(max_shift_multiplier=2.0)
    else:
        forecast = prediction_frame([-1.0, near_max])
        workload = workload_frame(
            ["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [1.0, 1.0], [0.5, 0.5]
        )
        constraints = DecisionConstraints(risk_penalty=np.finfo(float).max)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ShadowScheduler().solve(forecast, workload, constraints)

    assert_finite_infeasible(result, "numeric")


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


def test_revalidates_duplicate_workload_key_from_backing_frame() -> None:
    workload = workload_frame(["2026-01-01 01:00Z", "2026-01-01 02:00Z"], [10.0, 10.0], [0.5, 0.5])
    backing = workload.to_pandas(copy=False)
    backing.loc[1, ["job_id", "event_time", "available_at"]] = backing.loc[
        0, ["job_id", "event_time", "available_at"]
    ].to_list()

    with pytest.raises(ContractError, match=r"WorkloadFrame: duplicate key"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


@pytest.mark.parametrize("fraction", [float("nan"), -0.1, 1.1])
def test_revalidates_mutated_workload_fraction(fraction: float) -> None:
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    workload.to_pandas(copy=False).loc[0, "flexible_fraction"] = fraction

    with pytest.raises(ContractError, match=r"WorkloadFrame: flexible_fraction"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


def test_revalidates_mutated_workload_event_availability_order() -> None:
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    workload.to_pandas(copy=False).loc[0, "available_at"] = pd.Timestamp("2026-01-01 00:59Z")

    with pytest.raises(ContractError, match=r"WorkloadFrame: expected event_time <= available_at"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


def test_revalidates_mutated_workload_deadline_order() -> None:
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    workload.to_pandas(copy=False).loc[0, "deadline"] = pd.Timestamp("2026-01-01 00:59Z")

    with pytest.raises(ContractError, match=r"WorkloadFrame: expected deadline >= event_time"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())


def test_revalidates_exact_columns_and_units_from_backing_frames() -> None:
    workload = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    workload.to_pandas(copy=False)["unexpected"] = "value"
    with pytest.raises(ContractError, match=r"WorkloadFrame: expected exact columns"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), workload, DecisionConstraints())

    invalid_unit = workload_frame(["2026-01-01 01:00Z"], [10.0], [0.5])
    invalid_unit.to_pandas(copy=False).loc[0, "unit"] = "not-a-unit"
    with pytest.raises(ContractError, match=r"WorkloadFrame: invalid unit"):
        ShadowScheduler().solve(prediction_frame([1.0, 2.0]), invalid_unit, DecisionConstraints())


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


def test_zero_flexible_row_remains_entirely_fixed_at_baseline_slot() -> None:
    result = ShadowScheduler().solve(
        prediction_frame([10.0, 0.0]),
        workload_frame(["2026-01-01 01:00Z"], [10.0], [0.0]),
        DecisionConstraints(flexible_fraction=1.0),
    )

    assert result.feasible
    assert result.schedule["fixed"].tolist() == [10.0, 0.0]
    assert result.schedule["flexible_before"].tolist() == [0.0, 0.0]
    assert result.schedule["flexible_after"].tolist() == [0.0, 0.0]


def test_zero_baseline_destination_has_zero_capacity_and_cannot_receive_load() -> None:
    result = ShadowScheduler().solve(
        prediction_frame([10.0, 0.0]),
        workload_frame(["2026-01-01 01:00Z"], [10.0], [1.0]),
        DecisionConstraints(
            flexible_fraction=1.0,
            max_shift_multiplier=2.0,
            peak_penalty=0.0,
            risk_penalty=0.0,
        ),
    )

    assert result.feasible
    assert result.schedule["capacity"].tolist() == [20.0, 0.0]
    assert result.schedule["flexible_after"].tolist() == pytest.approx([10.0, 0.0])
