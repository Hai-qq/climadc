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
# HiGHS must solve below the framework's reporting tolerance: its looser default
# can treat legal tiny equality RHS values as zero before postconditions run.
_HIGHS_FEASIBILITY_TOLERANCE = 1e-10
_NUMERIC_STATUS = -2.0


class _NumericError(Exception):
    pass


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


def _require_finite(label: str, *arrays: np.ndarray) -> None:
    if any(not np.isfinite(array).all() for array in arrays):
        raise _NumericError(f"{label} produced a non-finite value")


def _normalized_risk(values: np.ndarray) -> np.ndarray:
    scale = max(abs(float(np.min(values))), abs(float(np.max(values))))
    if scale == 0.0:
        return np.zeros_like(values, dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled = np.divide(values, scale)
            minimum = float(np.min(scaled))
            spread = float(np.subtract(np.max(scaled), minimum))
            if spread == 0.0:
                return np.zeros_like(values, dtype=float)
            risk: np.ndarray = np.asarray(
                np.divide(np.subtract(scaled, minimum), spread), dtype=float
            )
    except FloatingPointError as exc:
        raise _NumericError("risk normalization is not representable") from exc
    _require_finite("risk normalization", risk)
    return risk


def _validate_forecast(forecast: PredictionFrame) -> tuple[pd.DataFrame, object]:
    if not isinstance(forecast, PredictionFrame):
        raise ContractError("forecast must be a PredictionFrame")
    raw = forecast.to_pandas()
    if raw.empty:
        raise ContractError("forecast must contain at least one row")
    if "valid_time" in raw.columns and raw["valid_time"].duplicated().any():
        raise ContractError("forecast: exactly one value is required per unique valid_time")
    try:
        frame = PredictionFrame.from_pandas(raw).to_pandas()
    except ContractError as exc:
        raise ContractError(f"forecast: {exc}") from exc
    _require_exact_utc(raw, ("issue_time", "valid_time"), "forecast")
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
    if not (frame["valid_time"] > issue_time).all():
        raise ContractError("forecast: valid_time must be strictly after issue_time")
    frame.sort_values("valid_time", kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame, site_id


def _validate_workload(workload: WorkloadFrame, site_id: object) -> pd.DataFrame:
    if not isinstance(workload, WorkloadFrame):
        raise ContractError("workload must be a WorkloadFrame")
    raw = workload.to_pandas()
    if raw.empty:
        raise ContractError("workload must contain at least one row")
    if "demand" in raw.columns:
        try:
            demand = raw["demand"].to_numpy(dtype=float)
        except (TypeError, ValueError):
            demand = np.array([], dtype=float)
        if demand.size and not np.isfinite(demand).all():
            raise ContractError("workload: demand must be finite and nonnegative")
    try:
        frame = WorkloadFrame.from_pandas(raw).to_pandas()
    except ContractError as exc:
        raise ContractError(f"workload: {exc}") from exc
    _require_exact_utc(raw, ("event_time", "deadline"), "workload")
    workload_site = _require_one_value(frame, "site_id", "workload")
    if workload_site != site_id:
        raise ContractError("workload: site_id must match forecast site_id")
    _require_one_value(frame, "resource_type", "workload")
    _require_one_value(frame, "unit", "workload")
    if (frame["demand"] < 0.0).any():
        raise ContractError("workload: demand must be finite and nonnegative")
    return frame


def _baseline_schedule(
    prepared: _PreparedInputs,
    flexible_after: np.ndarray | None = None,
) -> pd.DataFrame:
    after = prepared.flexible_before if flexible_after is None else flexible_after
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            total_before = np.add(prepared.fixed, prepared.flexible_before)
            total_after = np.add(prepared.fixed, after)
    except FloatingPointError as exc:
        raise _NumericError("schedule totals are not representable") from exc
    _require_finite(
        "schedule", prepared.forecast_values, prepared.capacity, total_before, total_after
    )
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


def _empty_schedule() -> pd.DataFrame:
    schedule: pd.DataFrame = pd.DataFrame(columns=_SCHEDULE_COLUMNS)
    return schedule


def _metrics(
    schedule: pd.DataFrame,
    risk: np.ndarray,
    solver_status: float,
) -> dict[str, float]:
    baseline = schedule["total_before"].to_numpy(dtype=float)
    scheduled = schedule["total_after"].to_numpy(dtype=float)
    forecast_values = schedule["forecast_value"].to_numpy(dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            baseline_peak = float(np.max(baseline))
            scheduled_peak = float(np.max(scheduled))
            baseline_cost = float(np.dot(baseline, forecast_values))
            scheduled_cost = float(np.dot(scheduled, forecast_values))
            baseline_risk = float(np.dot(baseline, risk))
            scheduled_risk = float(np.dot(scheduled, risk))
            energy_error = float(abs(np.subtract(np.sum(scheduled), np.sum(baseline))))
            values = np.array(
                [
                    baseline_peak,
                    scheduled_peak,
                    np.subtract(scheduled_peak, baseline_peak),
                    baseline_cost,
                    scheduled_cost,
                    np.subtract(scheduled_cost, baseline_cost),
                    baseline_risk,
                    scheduled_risk,
                    np.subtract(scheduled_risk, baseline_risk),
                    energy_error,
                    solver_status,
                ],
                dtype=float,
            )
    except FloatingPointError as exc:
        raise _NumericError("decision metrics are not representable") from exc
    _require_finite("decision metrics", values)
    return {
        "baseline_peak": baseline_peak,
        "scheduled_peak": scheduled_peak,
        "peak_change": float(values[2]),
        "baseline_cost_index": baseline_cost,
        "scheduled_cost_index": scheduled_cost,
        "cost_index_change": float(values[5]),
        "baseline_risk_exposure": baseline_risk,
        "scheduled_risk_exposure": scheduled_risk,
        "risk_exposure_change": float(values[8]),
        "energy_conservation_error": energy_error,
        "solver_status": float(solver_status),
    }


def _numeric_infeasible(
    message: str,
    prepared: _PreparedInputs | None = None,
    violations: tuple[str, ...] = (),
) -> DecisionResult:
    schedule = _empty_schedule()
    if prepared is not None:
        try:
            schedule = _baseline_schedule(prepared)
        except _NumericError:
            schedule = _empty_schedule()
    baseline_peak = float(schedule["total_before"].max()) if not schedule.empty else 0.0
    metrics = {
        "baseline_peak": baseline_peak,
        "scheduled_peak": baseline_peak,
        "peak_change": 0.0,
        "baseline_cost_index": 0.0,
        "scheduled_cost_index": 0.0,
        "cost_index_change": 0.0,
        "baseline_risk_exposure": 0.0,
        "scheduled_risk_exposure": 0.0,
        "risk_exposure_change": 0.0,
        "energy_conservation_error": 0.0,
        "solver_status": _NUMERIC_STATUS,
    }
    return DecisionResult(
        schedule=schedule,
        feasible=False,
        violations=(*violations, f"numeric: {message}"),
        metrics=metrics,
    )


def _infeasible(
    prepared: _PreparedInputs,
    violations: tuple[str, ...],
    solver_status: float,
) -> DecisionResult:
    try:
        schedule = _baseline_schedule(prepared)
        metrics = _metrics(schedule, prepared.risk, solver_status)
    except _NumericError as exc:
        return _numeric_infeasible(str(exc), prepared, violations)
    return DecisionResult(
        schedule=schedule,
        feasible=False,
        violations=violations,
        metrics=metrics,
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

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for row_position in range(row_count):
                event_time = event_times[row_position]
                baseline_slot = int(slots.searchsorted(event_time, side="left"))
                effective_fraction = min(
                    float(fractions[row_position]), constraints.flexible_fraction
                )
                row_flexible = float(np.multiply(demands[row_position], effective_fraction))
                flexible_energy[row_position] = row_flexible
                if baseline_slot >= slot_count:
                    violations.append(
                        "horizon: workload row "
                        f"{row_position} has no forecast slot at or after event_time"
                    )
                    continue

                row_fixed = float(np.subtract(demands[row_position], row_flexible))
                fixed[baseline_slot] = np.add(fixed[baseline_slot], row_fixed)
                flexible_before[baseline_slot] = np.add(
                    flexible_before[baseline_slot], row_flexible
                )
                row_eligible = slots >= event_time
                if not pd.isna(deadlines[row_position]):
                    row_eligible &= slots <= pd.Timestamp(deadlines[row_position])
                eligible[row_position] = np.asarray(row_eligible, dtype=bool)
                if not row_eligible.any():
                    violations.append(
                        f"deadline: workload row {row_position} has no eligible forecast slot"
                    )

            total_before = np.add(fixed, flexible_before)
            capacity = np.multiply(constraints.max_shift_multiplier, total_before)
    except FloatingPointError as exc:
        raise _NumericError("baseline aggregation or capacity is not representable") from exc
    _require_finite(
        "baseline aggregation",
        fixed,
        flexible_before,
        flexible_energy,
        total_before,
        capacity,
    )
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
        try:
            prepared, input_violations = _prepare(forecast, workload, constraints)
        except _NumericError as exc:
            return _numeric_infeasible(str(exc))
        if input_violations:
            return _infeasible(prepared, input_violations, -1.0)

        row_count, slot_count = prepared.eligible.shape
        allocation_count = row_count * slot_count
        variable_count = allocation_count + 1
        peak_index = allocation_count

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                cost = np.zeros(variable_count, dtype=float)
                risk_cost = np.multiply(constraints.risk_penalty, prepared.risk)
                slot_cost = np.add(prepared.forecast_values, risk_cost)
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
                    upper[slot_position] = np.subtract(
                        prepared.capacity[slot_position], prepared.fixed[slot_position]
                    )
                    peak_row = slot_count + slot_position
                    inequality[peak_row, allocation_positions] = 1.0
                    inequality[peak_row, peak_index] = -1.0
                    upper[peak_row] = np.negative(prepared.fixed[slot_position])
        except FloatingPointError:
            return _numeric_infeasible(
                "LP objective or constraint arithmetic is not representable",
                prepared,
            )
        try:
            _require_finite(
                "LP objective and constraints",
                cost,
                equality,
                inequality,
                upper,
                prepared.flexible_energy,
            )
        except _NumericError as exc:
            return _numeric_infeasible(str(exc), prepared)

        bounds: list[tuple[float, float | None]] = []
        for is_eligible in prepared.eligible.ravel():
            bounds.append((0.0, None if is_eligible else 0.0))
        bounds.append((0.0, None))

        try:
            solver = linprog(
                cost,
                A_ub=inequality,
                b_ub=upper,
                A_eq=equality,
                b_eq=prepared.flexible_energy,
                bounds=bounds,
                method="highs",
                options={
                    "primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
                    "dual_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
                },
            )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            return _numeric_infeasible(f"linprog rejected finite LP arithmetic: {exc}", prepared)
        if not solver.success or solver.x is None:
            violation = f"solver status={solver.status}: {solver.message}"
            return _infeasible(prepared, (violation,), float(solver.status))

        status = float(solver.status)

        def invalid_solution(message: str) -> DecisionResult:
            return _infeasible(
                prepared,
                (f"solver verification failed: {message}",),
                status,
            )

        try:
            solution = np.asarray(solver.x, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return invalid_solution("solution must be numeric")
        if solution.ndim != 1:
            return invalid_solution("solution must be one-dimensional")
        if len(solution) != variable_count:
            return invalid_solution(
                f"solution length {len(solution)} does not match expected {variable_count}"
            )
        if not np.isfinite(solution).all():
            return invalid_solution("solution must contain only finite values")

        solution = solution.copy()
        # Reporting tolerance only cleans bound noise; legal positive eligible
        # allocations remain data, even when they are smaller than _TOLERANCE.
        negative_noise = (solution < 0.0) & (solution >= -_TOLERANCE)
        solution[negative_noise] = 0.0
        if (solution < 0.0).any():
            return invalid_solution("allocations and peak must be nonnegative")

        allocations = solution[:allocation_count].reshape(row_count, slot_count)
        peak = float(solution[peak_index])
        ineligible_noise = (~prepared.eligible) & (allocations > 0.0) & (allocations <= _TOLERANCE)
        allocations[ineligible_noise] = 0.0
        if (allocations[~prepared.eligible] > 0.0).any():
            return invalid_solution("ineligible allocation violates a zero bound")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                row_totals = np.sum(allocations, axis=1)
                row_energy_error = float(
                    np.max(np.abs(np.subtract(row_totals, prepared.flexible_energy)))
                )
                flexible_after = np.sum(allocations, axis=0)
                schedule = _baseline_schedule(prepared, flexible_after)
                total_after = schedule["total_after"].to_numpy(dtype=float)
                capacity_error = float(np.max(np.subtract(total_after, prepared.capacity)))
                peak_error = float(np.max(np.subtract(total_after, peak)))
        except (FloatingPointError, _NumericError) as exc:
            return _numeric_infeasible(
                f"solver postcondition arithmetic is not representable: {exc}",
                prepared,
            )
        try:
            _require_finite(
                "solver postconditions",
                row_totals,
                flexible_after,
                total_after,
                np.array([row_energy_error, capacity_error, peak_error]),
            )
        except _NumericError as exc:
            return _numeric_infeasible(str(exc), prepared)
        if row_energy_error > _TOLERANCE:
            return invalid_solution(f"row equality error {row_energy_error} exceeds {_TOLERANCE}")
        if capacity_error > _TOLERANCE:
            return invalid_solution(
                f"capacity upper-bound error {capacity_error} exceeds {_TOLERANCE}"
            )
        if peak_error > _TOLERANCE:
            return invalid_solution(f"peak inequality error {peak_error} exceeds {_TOLERANCE}")
        try:
            metrics = _metrics(schedule, prepared.risk, status)
        except _NumericError as exc:
            return _numeric_infeasible(str(exc), prepared)
        if metrics["energy_conservation_error"] > _TOLERANCE:
            return invalid_solution(
                "aggregate energy error "
                f"{metrics['energy_conservation_error']} exceeds {_TOLERANCE}"
            )
        return DecisionResult(
            schedule=schedule,
            feasible=True,
            violations=(),
            metrics=metrics,
        )
