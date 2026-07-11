from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog  # type: ignore[import-untyped]

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.decision.protocols import DecisionConstraints, DecisionResult
from climadc.errors import ContractError

_SCHEDULE_COLUMNS = [
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
_TOLERANCE = 1e-8


@dataclass(frozen=True)
class _PreparedInputs:
    site_id: object
    slots: pd.DatetimeIndex
    forecast_values: np.ndarray
    risk: np.ndarray
    fixed: np.ndarray
    flexible_before: np.ndarray
    capacity: np.ndarray
    flexible_energy: np.ndarray
    eligible: np.ndarray


def _require_one_value(frame: pd.DataFrame, column: str, label: str) -> object:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ContractError(f"{label}: {column} must contain exactly one constant value")
    return values.iloc[0]


def _require_exact_utc(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        values = frame[column].dropna()
        if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC":
            raise ContractError(f"{label}: {column} must use exact UTC timestamps")


def _normalized_risk(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    spread = float(np.max(values) - minimum)
    if spread == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / spread


def _validate_forecast(forecast: PredictionFrame) -> tuple[pd.DataFrame, object]:
    if not isinstance(forecast, PredictionFrame):
        raise ContractError("forecast must be a PredictionFrame")
    frame = forecast.to_pandas()
    if frame.empty:
        raise ContractError("forecast must contain at least one row")
    site_id = _require_one_value(frame, "site_id", "forecast")
    issue_time = _require_one_value(frame, "issue_time", "forecast")
    for column in ("target", "model_id", "unit"):
        _require_one_value(frame, column, "forecast")

    quantile = frame["quantile"]
    if quantile.isna().all():
        pass
    elif quantile.notna().all() and quantile.nunique(dropna=False) == 1:
        pass
    else:
        raise ContractError(
            "forecast: quantile must be all null or one constant probability representation"
        )

    if frame["valid_time"].duplicated().any():
        raise ContractError("forecast: exactly one value is required per unique valid_time")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ContractError("forecast: value must be finite")
    _require_exact_utc(frame, ("issue_time", "valid_time"), "forecast")
    if not (frame["valid_time"] > issue_time).all():
        raise ContractError("forecast: valid_time must be strictly after issue_time")
    frame.sort_values("valid_time", kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame, site_id


def _validate_workload(workload: WorkloadFrame, site_id: object) -> pd.DataFrame:
    if not isinstance(workload, WorkloadFrame):
        raise ContractError("workload must be a WorkloadFrame")
    frame = workload.to_pandas()
    if frame.empty:
        raise ContractError("workload must contain at least one row")
    workload_site = _require_one_value(frame, "site_id", "workload")
    if workload_site != site_id:
        raise ContractError("workload: site_id must match forecast site_id")
    _require_one_value(frame, "resource_type", "workload")
    _require_one_value(frame, "unit", "workload")
    _require_exact_utc(frame, ("event_time", "deadline"), "workload")
    demand = frame["demand"].to_numpy(dtype=float)
    if not np.isfinite(demand).all() or (demand < 0.0).any():
        raise ContractError("workload: demand must be finite and nonnegative")
    return frame


def _baseline_schedule(
    prepared: _PreparedInputs,
    flexible_after: np.ndarray | None = None,
) -> pd.DataFrame:
    after = prepared.flexible_before if flexible_after is None else flexible_after
    total_before = prepared.fixed + prepared.flexible_before
    total_after = prepared.fixed + after
    schedule: pd.DataFrame = pd.DataFrame(
        {
            "site_id": [prepared.site_id] * len(prepared.slots),
            "valid_time": prepared.slots,
            "forecast_value": prepared.forecast_values,
            "fixed": prepared.fixed,
            "flexible_before": prepared.flexible_before,
            "flexible_after": after,
            "total_before": total_before,
            "total_after": total_after,
            "capacity": prepared.capacity,
        },
        columns=_SCHEDULE_COLUMNS,
    )
    return schedule


def _metrics(
    schedule: pd.DataFrame,
    risk: np.ndarray,
    solver_status: float,
) -> dict[str, float]:
    baseline = schedule["total_before"].to_numpy(dtype=float)
    scheduled = schedule["total_after"].to_numpy(dtype=float)
    forecast_values = schedule["forecast_value"].to_numpy(dtype=float)
    baseline_peak = float(np.max(baseline))
    scheduled_peak = float(np.max(scheduled))
    baseline_cost = float(np.dot(baseline, forecast_values))
    scheduled_cost = float(np.dot(scheduled, forecast_values))
    baseline_risk = float(np.dot(baseline, risk))
    scheduled_risk = float(np.dot(scheduled, risk))
    energy_error = float(abs(np.sum(scheduled) - np.sum(baseline)))
    return {
        "baseline_peak": baseline_peak,
        "scheduled_peak": scheduled_peak,
        "peak_change": scheduled_peak - baseline_peak,
        "baseline_cost_index": baseline_cost,
        "scheduled_cost_index": scheduled_cost,
        "cost_index_change": scheduled_cost - baseline_cost,
        "baseline_risk_exposure": baseline_risk,
        "scheduled_risk_exposure": scheduled_risk,
        "risk_exposure_change": scheduled_risk - baseline_risk,
        "energy_conservation_error": energy_error,
        "solver_status": float(solver_status),
    }


def _infeasible(
    prepared: _PreparedInputs,
    violations: tuple[str, ...],
    solver_status: float,
) -> DecisionResult:
    schedule = _baseline_schedule(prepared)
    return DecisionResult(
        schedule=schedule,
        feasible=False,
        violations=violations,
        metrics=_metrics(schedule, prepared.risk, solver_status),
    )


def _prepare(
    forecast: PredictionFrame,
    workload: WorkloadFrame,
    constraints: DecisionConstraints,
) -> tuple[_PreparedInputs, tuple[str, ...]]:
    forecast_data, site_id = _validate_forecast(forecast)
    workload_data = _validate_workload(workload, site_id)
    slots = pd.DatetimeIndex(forecast_data["valid_time"])
    forecast_values = forecast_data["value"].to_numpy(dtype=float, copy=True)
    risk = _normalized_risk(forecast_values)
    row_count = len(workload_data)
    slot_count = len(slots)
    fixed = np.zeros(slot_count, dtype=float)
    flexible_before = np.zeros(slot_count, dtype=float)
    flexible_energy = np.zeros(row_count, dtype=float)
    eligible = np.zeros((row_count, slot_count), dtype=bool)
    violations: list[str] = []
    event_times = pd.DatetimeIndex(workload_data["event_time"])
    deadlines = workload_data["deadline"].tolist()
    demands = workload_data["demand"].to_numpy(dtype=float, copy=True)
    fractions = workload_data["flexible_fraction"].to_numpy(dtype=float, copy=True)

    for row_position in range(row_count):
        event_time = event_times[row_position]
        baseline_slot = int(slots.searchsorted(event_time, side="left"))
        effective_fraction = min(float(fractions[row_position]), constraints.flexible_fraction)
        row_flexible = float(demands[row_position]) * effective_fraction
        flexible_energy[row_position] = row_flexible
        if baseline_slot >= slot_count:
            violations.append(
                f"horizon: workload row {row_position} has no forecast slot at or after event_time"
            )
            continue

        row_fixed = float(demands[row_position]) - row_flexible
        fixed[baseline_slot] += row_fixed
        flexible_before[baseline_slot] += row_flexible
        row_eligible = slots >= event_time
        if not pd.isna(deadlines[row_position]):
            row_eligible &= slots <= pd.Timestamp(deadlines[row_position])
        eligible[row_position] = np.asarray(row_eligible, dtype=bool)
        if not row_eligible.any():
            violations.append(
                f"deadline: workload row {row_position} has no eligible forecast slot"
            )

    total_before = fixed + flexible_before
    capacity = constraints.max_shift_multiplier * total_before
    prepared = _PreparedInputs(
        site_id=site_id,
        slots=slots,
        forecast_values=forecast_values,
        risk=risk,
        fixed=fixed,
        flexible_before=flexible_before,
        capacity=capacity,
        flexible_energy=flexible_energy,
        eligible=eligible,
    )
    return prepared, tuple(violations)


class ShadowScheduler:
    """Offline, energy-conserving shadow scheduler; it performs no real control."""

    def solve(
        self,
        forecast: PredictionFrame,
        workload: WorkloadFrame,
        constraints: DecisionConstraints,
    ) -> DecisionResult:
        if not isinstance(constraints, DecisionConstraints):
            raise ContractError("constraints must be DecisionConstraints")
        prepared, input_violations = _prepare(forecast, workload, constraints)
        if input_violations:
            return _infeasible(prepared, input_violations, -1.0)

        row_count, slot_count = prepared.eligible.shape
        allocation_count = row_count * slot_count
        variable_count = allocation_count + 1
        peak_index = allocation_count

        cost = np.zeros(variable_count, dtype=float)
        slot_cost = prepared.forecast_values + constraints.risk_penalty * prepared.risk
        cost[:allocation_count] = np.tile(slot_cost, row_count)
        cost[peak_index] = constraints.peak_penalty

        equality = np.zeros((row_count, variable_count), dtype=float)
        for row_position in range(row_count):
            start = row_position * slot_count
            equality[row_position, start : start + slot_count] = 1.0

        inequality = np.zeros((slot_count * 2, variable_count), dtype=float)
        upper = np.zeros(slot_count * 2, dtype=float)
        for slot_position in range(slot_count):
            allocation_positions = np.arange(row_count) * slot_count + slot_position
            inequality[slot_position, allocation_positions] = 1.0
            upper[slot_position] = prepared.capacity[slot_position] - prepared.fixed[slot_position]
            peak_row = slot_count + slot_position
            inequality[peak_row, allocation_positions] = 1.0
            inequality[peak_row, peak_index] = -1.0
            upper[peak_row] = -prepared.fixed[slot_position]

        bounds: list[tuple[float, float | None]] = []
        for is_eligible in prepared.eligible.ravel():
            bounds.append((0.0, None if is_eligible else 0.0))
        bounds.append((0.0, None))

        solver = linprog(
            cost,
            A_ub=inequality,
            b_ub=upper,
            A_eq=equality,
            b_eq=prepared.flexible_energy,
            bounds=bounds,
            method="highs",
        )
        if not solver.success or solver.x is None:
            violation = f"solver status={solver.status}: {solver.message}"
            return _infeasible(prepared, (violation,), float(solver.status))

        allocations = np.asarray(solver.x[:allocation_count], dtype=float).reshape(
            row_count, slot_count
        )
        flexible_after = allocations.sum(axis=0)
        schedule = _baseline_schedule(prepared, flexible_after)
        metrics = _metrics(schedule, prepared.risk, float(solver.status))
        capacity_error = float(np.max(schedule["total_after"].to_numpy() - prepared.capacity))
        row_energy_error = float(np.max(np.abs(allocations.sum(axis=1) - prepared.flexible_energy)))
        if (
            metrics["energy_conservation_error"] > _TOLERANCE
            or row_energy_error > _TOLERANCE
            or capacity_error > _TOLERANCE
        ):
            violation = (
                "solver verification failed: "
                f"energy_error={metrics['energy_conservation_error']}, "
                f"row_energy_error={row_energy_error}, capacity_error={capacity_error}"
            )
            return _infeasible(prepared, (violation,), float(solver.status))
        return DecisionResult(
            schedule=schedule,
            feasible=True,
            violations=(),
            metrics=metrics,
        )
