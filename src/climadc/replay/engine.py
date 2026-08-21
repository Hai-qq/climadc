from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import linprog  # type: ignore[import-untyped]

from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ConfigurationError, ContractError
from climadc.replay.inputs import PreparedReplayInputs, prepare_replay_inputs
from climadc.replay.models import (
    FacilityEnergyModel,
    ReplayConfig,
    pareto_carbon_price,
    pareto_policy_name,
)

POLICY_NAMES = ("asap", "peak", "price", "carbon", "joint", "oracle")
RISK_AWARE_POLICY = "risk_aware"
ALL_POLICY_NAMES = (*POLICY_NAMES[:-1], RISK_AWARE_POLICY, POLICY_NAMES[-1])
_HIGHS_FEASIBILITY_TOLERANCE = 1e-10
_STATUS_COLUMNS = ("policy", "feasible", "solver_status", "message")
_ALLOCATION_COLUMNS = (
    "site_id",
    "policy",
    "job_id",
    "valid_time",
    "power_kw",
    "energy_kwh",
)
_PROFILE_COLUMNS = (
    "site_id",
    "policy",
    "valid_time",
    "forecast_temperature_c",
    "actual_temperature_c",
    "forecast_pue",
    "actual_pue",
    "fixed_it_power_kw",
    "flexible_it_power_kw",
    "total_it_power_kw",
    "forecast_facility_power_kw",
    "actual_facility_power_kw",
    "forecast_energy_price",
    "actual_energy_price",
    "forecast_carbon_kgco2e_per_kwh",
    "actual_carbon_kgco2e_per_kwh",
    "decision_basis",
    "decision_temperature_c",
    "decision_pue",
    "decision_energy_price",
    "decision_carbon_kgco2e_per_kwh",
)
_METRIC_COLUMNS = (
    "policy",
    "facility_energy_kwh",
    "it_energy_kwh",
    "cooling_energy_kwh",
    "estimated_location_based_emissions_kgco2e",
    "energy_charge",
    "demand_charge",
    "energy_cost",
    "peak_kw",
    "completed_jobs",
    "deadline_violations",
    "unserved_energy_kwh",
    "energy_balance_error_kwh",
    "shifted_energy_kwh",
    "energy_cost_change_vs_asap",
    "estimated_location_based_emissions_change_vs_asap_kgco2e",
    "peak_change_vs_asap_kw",
    "realized_objective",
    "objective_regret",
)


def replay_policy_names(config: ReplayConfig) -> tuple[str, ...]:
    if config.objective_mode == "pareto_analysis":
        points = tuple(
            pareto_policy_name(price) for price in config.pareto_carbon_prices_currency_per_tco2e
        )
        return (*POLICY_NAMES[:4], *points, "oracle")
    return ALL_POLICY_NAMES if config.risk_quantile is not None else POLICY_NAMES


@dataclass(frozen=True)
class _Solution:
    policy: str
    feasible: bool
    solver_status: int
    message: str
    allocation_kw: np.ndarray | None


@dataclass(frozen=True, init=False)
class ReplayResult:
    """Immutable replay output with defensive DataFrame accessors."""

    _status: pd.DataFrame
    _metrics: pd.DataFrame
    _allocations: pd.DataFrame
    _profiles: pd.DataFrame
    _violations: Mapping[str, tuple[str, ...]]
    currency: str
    accepted_jobs: int
    future_jobs: int

    def __init__(
        self,
        *,
        status: pd.DataFrame,
        metrics: pd.DataFrame,
        allocations: pd.DataFrame,
        profiles: pd.DataFrame,
        violations: Mapping[str, tuple[str, ...]],
        currency: str,
        accepted_jobs: int,
        future_jobs: int,
    ) -> None:
        for name, frame in (
            ("status", status),
            ("metrics", metrics),
            ("allocations", allocations),
            ("profiles", profiles),
        ):
            if not isinstance(frame, pd.DataFrame):
                raise ContractError(f"ReplayResult {name} must be a pandas DataFrame")
        if not isinstance(currency, str) or not currency:
            raise ContractError("ReplayResult currency must be a non-empty string")
        if not isinstance(accepted_jobs, int) or accepted_jobs < 0:
            raise ContractError("ReplayResult accepted_jobs must be a nonnegative integer")
        if not isinstance(future_jobs, int) or future_jobs < 0:
            raise ContractError("ReplayResult future_jobs must be a nonnegative integer")
        checked_violations: dict[str, tuple[str, ...]] = {}
        for policy, items in violations.items():
            if (
                not isinstance(policy, str)
                or not policy
                or any(not isinstance(item, str) for item in items)
            ):
                raise ContractError("ReplayResult violations are invalid")
            checked_violations[policy] = tuple(items)
        object.__setattr__(self, "_status", status.copy(deep=True))
        object.__setattr__(self, "_metrics", metrics.copy(deep=True))
        object.__setattr__(self, "_allocations", allocations.copy(deep=True))
        object.__setattr__(self, "_profiles", profiles.copy(deep=True))
        object.__setattr__(self, "_violations", MappingProxyType(checked_violations))
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "accepted_jobs", accepted_jobs)
        object.__setattr__(self, "future_jobs", future_jobs)

    @property
    def status(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._status.copy(deep=True))

    @property
    def metrics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._metrics.copy(deep=True))

    @property
    def allocations(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._allocations.copy(deep=True))

    @property
    def profiles(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._profiles.copy(deep=True))

    @property
    def violations(self) -> dict[str, tuple[str, ...]]:
        return dict(self._violations)

    def to_metrics(self) -> pd.DataFrame:
        return self.metrics


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.DataFrame(columns=list(columns)))


def _validate_pue(values: object, *, slots: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"facility model {label} PUE must be numeric") from exc
    if result.shape != (slots,) or not np.isfinite(result).all() or np.any(result < 1.0):
        raise ContractError(
            f"facility model {label} PUE must contain one finite value >= 1 per slot"
        )
    checked: np.ndarray = result.copy()
    return checked


def _global_violations(
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
) -> tuple[str, ...]:
    violations: list[str] = []
    horizon_end = prepared.decision_time + config.horizon
    flexible_capacity_kw = config.it_capacity_kw - config.fixed_it_power_kw
    for position, (_, row) in enumerate(prepared.jobs.iterrows()):
        job_id = str(row["job_id"])
        if cast(pd.Timestamp, row["deadline"]) > horizon_end:
            violations.append(f"job {job_id}: deadline extends beyond replay horizon")
            continue
        available_energy = (
            float(np.sum(prepared.eligible[position]))
            * min(float(row["max_power_kw"]), flexible_capacity_kw)
            * prepared.interval_hours
        )
        if available_energy + config.tolerance_kwh < float(row["energy_kwh"]):
            violations.append(f"job {job_id}: insufficient discretely eligible capacity")
    return tuple(violations)


def _constraints(
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
    pue: np.ndarray,
    *,
    policy: str,
    carbon_kgco2e_per_kwh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float | None]]]:
    job_count = len(prepared.jobs)
    slot_count = len(prepared.slots)
    variable_count = job_count * slot_count + 1
    peak_index = variable_count - 1

    a_eq = np.zeros((job_count, variable_count), dtype=float)
    b_eq = np.zeros(job_count, dtype=float)
    bounds: list[tuple[float, float | None]] = []
    for job_index, (_, row) in enumerate(prepared.jobs.iterrows()):
        for slot_index in range(slot_count):
            variable_index = job_index * slot_count + slot_index
            a_eq[job_index, variable_index] = prepared.interval_hours
            upper = float(row["max_power_kw"]) if prepared.eligible[job_index, slot_index] else 0.0
            bounds.append((0.0, upper))
        b_eq[job_index] = float(row["energy_kwh"])
    bounds.append((0.0, float(np.max(pue) * config.it_capacity_kw)))

    a_ub = np.zeros((2 * slot_count, variable_count), dtype=float)
    b_ub = np.zeros(2 * slot_count, dtype=float)
    flexible_capacity_kw = config.it_capacity_kw - config.fixed_it_power_kw
    for slot_index in range(slot_count):
        job_variables = [job_index * slot_count + slot_index for job_index in range(job_count)]
        a_ub[slot_index, job_variables] = 1.0
        b_ub[slot_index] = flexible_capacity_kw
        a_ub[slot_count + slot_index, job_variables] = pue[slot_index]
        a_ub[slot_count + slot_index, peak_index] = -1.0
        b_ub[slot_count + slot_index] = -pue[slot_index] * config.fixed_it_power_kw
    if config.objective_mode == "epsilon_constraint":
        extra_rows: list[np.ndarray] = []
        extra_bounds: list[float] = []
        if config.emissions_upper_bound_kgco2e is not None:
            constraint_row = np.zeros(variable_count, dtype=float)
            slot_coefficients = pue * carbon_kgco2e_per_kwh * prepared.interval_hours
            for job_index in range(job_count):
                start = job_index * slot_count
                constraint_row[start : start + slot_count] = slot_coefficients
            fixed_emissions = float(np.sum(slot_coefficients * config.fixed_it_power_kw))
            extra_rows.append(constraint_row)
            extra_bounds.append(config.emissions_upper_bound_kgco2e - fixed_emissions)
        if config.peak_upper_bound_kw is not None:
            constraint_row = np.zeros(variable_count, dtype=float)
            constraint_row[peak_index] = 1.0
            extra_rows.append(constraint_row)
            extra_bounds.append(config.peak_upper_bound_kw)
        if extra_rows:
            a_ub = np.vstack([a_ub, *extra_rows])
            b_ub = np.concatenate([b_ub, np.asarray(extra_bounds, dtype=float)])
    return a_eq, b_eq, a_ub, b_ub, bounds


def _joint_slot_cost(
    *,
    config: ReplayConfig,
    policy: str,
    pue: np.ndarray,
    price_per_kwh: np.ndarray,
    carbon_kgco2e_per_kwh: np.ndarray,
) -> tuple[np.ndarray, float]:
    pareto_price = pareto_carbon_price(policy, config)
    if config.objective_mode == "legacy_unscaled":
        return (
            pue
            * (config.cost_weight * price_per_kwh + config.carbon_weight * carbon_kgco2e_per_kwh),
            config.cost_weight * config.demand_charge_per_kw,
        )
    if config.objective_mode == "epsilon_constraint":
        return pue * price_per_kwh, config.demand_charge_per_kw
    carbon_price = config.carbon_price_currency_per_tco2e if pareto_price is None else pareto_price
    if config.objective_mode == "pareto_analysis" and pareto_price is None:
        carbon_price = config.pareto_carbon_prices_currency_per_tco2e[0]
    return (
        pue * (price_per_kwh + carbon_price * carbon_kgco2e_per_kwh / 1000.0),
        config.demand_charge_per_kw,
    )


def _asap_coefficients(prepared: PreparedReplayInputs) -> np.ndarray:
    job_count = len(prepared.jobs)
    slot_count = len(prepared.slots)
    coefficients = np.zeros(job_count * slot_count + 1, dtype=float)
    order_keys = [
        (
            -float(row["priority"]),
            cast(pd.Timestamp, row["deadline"]),
            cast(pd.Timestamp, row["release_time"]),
            str(row["job_id"]),
        )
        for _, row in prepared.jobs.iterrows()
    ]
    ordered_jobs = sorted(range(job_count), key=order_keys.__getitem__)
    delay_weights = np.zeros(job_count, dtype=float)
    for rank, job_index in enumerate(ordered_jobs):
        delay_weights[job_index] = float(job_count - rank)
    for job_index in range(job_count):
        for slot_index in range(slot_count):
            coefficients[job_index * slot_count + slot_index] = (
                (slot_index + 1) * delay_weights[job_index] * prepared.interval_hours
            )
    return coefficients


def _objective(
    policy: str,
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
    forecast_pue: np.ndarray,
    risk_pue: np.ndarray | None,
    actual_pue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    job_count = len(prepared.jobs)
    slot_count = len(prepared.slots)
    coefficients = np.zeros(job_count * slot_count + 1, dtype=float)
    if policy == "oracle":
        constraint_pue = actual_pue
    elif policy == RISK_AWARE_POLICY:
        if risk_pue is None:
            raise ConfigurationError("risk-aware policy requires configured quantile forecasts")
        constraint_pue = risk_pue
    else:
        constraint_pue = forecast_pue
    if policy == "asap":
        coefficients = _asap_coefficients(prepared)
    elif policy == "peak":
        coefficients[-1] = 1.0
    else:
        if policy == "price":
            slot_cost = forecast_pue * prepared.forecast_price_per_kwh
            peak_cost = config.demand_charge_per_kw
        elif policy == "carbon":
            slot_cost = forecast_pue * prepared.forecast_carbon_kgco2e_per_kwh
            peak_cost = 0.0
        elif policy == "joint" or pareto_carbon_price(policy, config) is not None:
            slot_cost, peak_cost = _joint_slot_cost(
                config=config,
                policy=policy,
                pue=forecast_pue,
                price_per_kwh=prepared.forecast_price_per_kwh,
                carbon_kgco2e_per_kwh=prepared.forecast_carbon_kgco2e_per_kwh,
            )
        elif policy == RISK_AWARE_POLICY:
            if (
                risk_pue is None
                or prepared.risk_price_per_kwh is None
                or prepared.risk_carbon_kgco2e_per_kwh is None
            ):
                raise ConfigurationError(
                    "risk-aware policy requires temperature, price, and carbon quantile forecasts"
                )
            slot_cost, peak_cost = _joint_slot_cost(
                config=config,
                policy=policy,
                pue=risk_pue,
                price_per_kwh=prepared.risk_price_per_kwh,
                carbon_kgco2e_per_kwh=prepared.risk_carbon_kgco2e_per_kwh,
            )
        elif policy == "oracle":
            slot_cost, peak_cost = _joint_slot_cost(
                config=config,
                policy=policy,
                pue=actual_pue,
                price_per_kwh=prepared.actual_price_per_kwh,
                carbon_kgco2e_per_kwh=prepared.actual_carbon_kgco2e_per_kwh,
            )
        else:
            raise ConfigurationError(f"unknown replay policy: {policy}")
        for job_index in range(job_count):
            start = job_index * slot_count
            coefficients[start : start + slot_count] = slot_cost * prepared.interval_hours
        coefficients[-1] = peak_cost
    return coefficients, constraint_pue


def _solve_policy(
    policy: str,
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
    forecast_pue: np.ndarray,
    risk_pue: np.ndarray | None,
    actual_pue: np.ndarray,
) -> _Solution:
    coefficients, constraint_pue = _objective(
        policy, prepared, config, forecast_pue, risk_pue, actual_pue
    )
    if policy == "oracle":
        constraint_carbon = prepared.actual_carbon_kgco2e_per_kwh
    elif policy == RISK_AWARE_POLICY:
        if prepared.risk_carbon_kgco2e_per_kwh is None:
            raise ConfigurationError("risk-aware policy requires carbon quantile forecasts")
        constraint_carbon = prepared.risk_carbon_kgco2e_per_kwh
    else:
        constraint_carbon = prepared.forecast_carbon_kgco2e_per_kwh
    a_eq, b_eq, a_ub, b_ub, bounds = _constraints(
        prepared,
        config,
        constraint_pue,
        policy=policy,
        carbon_kgco2e_per_kwh=constraint_carbon,
    )
    result = linprog(
        coefficients,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE},
    )
    if not result.success or result.x is None:
        return _Solution(policy, False, int(result.status), str(result.message), None)
    job_count = len(prepared.jobs)
    slot_count = len(prepared.slots)
    primary_allocation = np.asarray(result.x[: job_count * slot_count], dtype=float).reshape(
        job_count, slot_count
    )
    slot_rows = np.zeros((slot_count, job_count * slot_count + 1), dtype=float)
    for slot_index in range(slot_count):
        slot_rows[
            slot_index,
            [job_index * slot_count + slot_index for job_index in range(job_count)],
        ] = 1.0
    canonical_result = linprog(
        _asap_coefficients(prepared),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.vstack([a_eq, slot_rows]),
        b_eq=np.concatenate([b_eq, np.sum(primary_allocation, axis=0)]),
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE},
    )
    if not canonical_result.success or canonical_result.x is None:
        return _Solution(
            policy,
            False,
            int(canonical_result.status),
            f"deterministic allocation tie-break failed: {canonical_result.message}",
            None,
        )
    result = canonical_result
    allocation = np.asarray(result.x[: job_count * slot_count], dtype=float).reshape(
        job_count, slot_count
    )
    allocation[np.abs(allocation) < _HIGHS_FEASIBILITY_TOLERANCE] = 0.0
    job_energy = np.sum(allocation, axis=1) * prepared.interval_hours
    targets = prepared.jobs["energy_kwh"].to_numpy(dtype=float)
    capacity = np.sum(allocation, axis=0) + config.fixed_it_power_kw
    valid = (
        np.isfinite(allocation).all()
        and np.all(allocation >= -config.tolerance_kwh)
        and np.all(np.abs(job_energy - targets) <= config.tolerance_kwh)
        and np.all(capacity <= config.it_capacity_kw + config.tolerance_kwh)
        and np.all(allocation[~prepared.eligible] <= config.tolerance_kwh)
    )
    if not valid:
        return _Solution(
            policy,
            False,
            int(result.status),
            "solver result failed replay postconditions",
            None,
        )
    return _Solution(policy, True, int(result.status), str(result.message), allocation)


def _allocations_frame(
    solution: _Solution,
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
) -> pd.DataFrame:
    allocation = cast(np.ndarray, solution.allocation_kw)
    rows: list[dict[str, object]] = []
    for job_index, (_, job) in enumerate(prepared.jobs.iterrows()):
        for slot_index, slot in enumerate(prepared.slots):
            power = float(allocation[job_index, slot_index])
            rows.append(
                {
                    "site_id": config.site_id,
                    "policy": solution.policy,
                    "job_id": str(job["job_id"]),
                    "valid_time": slot,
                    "power_kw": power,
                    "energy_kwh": power * prepared.interval_hours,
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows, columns=list(_ALLOCATION_COLUMNS)))


def _profile_frame(
    solution: _Solution,
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
    forecast_pue: np.ndarray,
    risk_pue: np.ndarray | None,
    actual_pue: np.ndarray,
) -> pd.DataFrame:
    allocation = cast(np.ndarray, solution.allocation_kw)
    flexible = np.sum(allocation, axis=0)
    total_it = flexible + config.fixed_it_power_kw
    if solution.policy == RISK_AWARE_POLICY:
        if (
            config.risk_quantile is None
            or prepared.risk_temperature_c is None
            or prepared.risk_price_per_kwh is None
            or prepared.risk_carbon_kgco2e_per_kwh is None
            or risk_pue is None
        ):
            raise ContractError("risk-aware profile is missing configured quantile inputs")
        decision_basis = f"quantile:{config.risk_quantile:g}"
        decision_temperature = prepared.risk_temperature_c
        decision_pue = risk_pue
        decision_price = prepared.risk_price_per_kwh
        decision_carbon = prepared.risk_carbon_kgco2e_per_kwh
    elif solution.policy == "oracle":
        decision_basis = "realized"
        decision_temperature = prepared.actual_temperature_c
        decision_pue = actual_pue
        decision_price = prepared.actual_price_per_kwh
        decision_carbon = prepared.actual_carbon_kgco2e_per_kwh
    else:
        decision_basis = "point"
        decision_temperature = prepared.forecast_temperature_c
        decision_pue = forecast_pue
        decision_price = prepared.forecast_price_per_kwh
        decision_carbon = prepared.forecast_carbon_kgco2e_per_kwh
    return cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "site_id": [config.site_id] * len(prepared.slots),
                "policy": [solution.policy] * len(prepared.slots),
                "valid_time": prepared.slots,
                "forecast_temperature_c": prepared.forecast_temperature_c,
                "actual_temperature_c": prepared.actual_temperature_c,
                "forecast_pue": forecast_pue,
                "actual_pue": actual_pue,
                "fixed_it_power_kw": [config.fixed_it_power_kw] * len(prepared.slots),
                "flexible_it_power_kw": flexible,
                "total_it_power_kw": total_it,
                "forecast_facility_power_kw": forecast_pue * total_it,
                "actual_facility_power_kw": actual_pue * total_it,
                "forecast_energy_price": prepared.forecast_price_per_kwh,
                "actual_energy_price": prepared.actual_price_per_kwh,
                "forecast_carbon_kgco2e_per_kwh": (prepared.forecast_carbon_kgco2e_per_kwh),
                "actual_carbon_kgco2e_per_kwh": prepared.actual_carbon_kgco2e_per_kwh,
                "decision_basis": [decision_basis] * len(prepared.slots),
                "decision_temperature_c": decision_temperature,
                "decision_pue": decision_pue,
                "decision_energy_price": decision_price,
                "decision_carbon_kgco2e_per_kwh": decision_carbon,
            },
            columns=list(_PROFILE_COLUMNS),
        ),
    )


def _base_metrics(
    solution: _Solution,
    profile: pd.DataFrame,
    prepared: PreparedReplayInputs,
    config: ReplayConfig,
) -> dict[str, float | str]:
    allocation = cast(np.ndarray, solution.allocation_kw)
    interval_hours = prepared.interval_hours
    facility_power = profile["actual_facility_power_kw"].to_numpy(dtype=float)
    it_power = profile["total_it_power_kw"].to_numpy(dtype=float)
    facility_energy = float(np.sum(facility_power) * interval_hours)
    it_energy = float(np.sum(it_power) * interval_hours)
    cooling_energy = facility_energy - it_energy
    emissions = float(
        np.dot(facility_power * interval_hours, prepared.actual_carbon_kgco2e_per_kwh)
    )
    energy_charge = float(np.dot(facility_power * interval_hours, prepared.actual_price_per_kwh))
    peak = float(np.max(facility_power))
    demand_charge = peak * config.demand_charge_per_kw
    energy_cost = energy_charge + demand_charge
    job_energy = np.sum(allocation, axis=1) * interval_hours
    targets = prepared.jobs["energy_kwh"].to_numpy(dtype=float)
    deficits = np.maximum(targets - job_energy, 0.0)
    job_errors = np.abs(job_energy - targets)
    deadline_violations = float(np.sum(np.any((allocation > 1e-9) & ~prepared.eligible, axis=1)))
    facility_balance_error = abs(facility_energy - it_energy - cooling_energy)
    energy_balance_error = max(
        facility_balance_error,
        float(np.max(job_errors)) if len(job_errors) else 0.0,
    )
    return {
        "policy": solution.policy,
        "facility_energy_kwh": facility_energy,
        "it_energy_kwh": it_energy,
        "cooling_energy_kwh": cooling_energy,
        "estimated_location_based_emissions_kgco2e": emissions,
        "energy_charge": energy_charge,
        "demand_charge": demand_charge,
        "energy_cost": energy_cost,
        "peak_kw": peak,
        "completed_jobs": float(np.sum(job_errors <= config.tolerance_kwh)),
        "deadline_violations": deadline_violations,
        "unserved_energy_kwh": float(np.sum(deficits)),
        "energy_balance_error_kwh": energy_balance_error,
        "realized_objective": config.realized_objective_for_policy(
            solution.policy, energy_cost, emissions
        ),
    }


def _infeasible_result(
    *,
    violations: tuple[str, ...],
    prepared: PreparedReplayInputs,
    policies: tuple[str, ...],
) -> ReplayResult:
    status = pd.DataFrame(
        [
            {
                "policy": policy,
                "feasible": False,
                "solver_status": -1,
                "message": "; ".join(violations),
            }
            for policy in policies
        ],
        columns=list(_STATUS_COLUMNS),
    )
    return ReplayResult(
        status=status,
        metrics=_empty_frame(_METRIC_COLUMNS),
        allocations=_empty_frame(_ALLOCATION_COLUMNS),
        profiles=_empty_frame(_PROFILE_COLUMNS),
        violations={policy: violations for policy in policies},
        currency=prepared.currency,
        accepted_jobs=prepared.accepted_jobs,
        future_jobs=prepared.future_jobs,
    )


class ReplayEngine:
    """Solve and settle deterministic scheduling policies on one replay window."""

    def __init__(self, facility_model: FacilityEnergyModel) -> None:
        if not isinstance(facility_model, FacilityEnergyModel):
            raise ConfigurationError("facility_model must implement pue(temperature_vector)")
        self._facility_model = facility_model

    def run(
        self,
        *,
        decision_time: pd.Timestamp,
        climate_forecast: ClimateForecastFrame,
        actual_weather: DCTelemetryFrame,
        grid_signals: GridSignalFrame,
        workload: FlexibleWorkloadFrame,
        config: ReplayConfig,
    ) -> ReplayResult:
        prepared = prepare_replay_inputs(
            decision_time=decision_time,
            climate_forecast=climate_forecast,
            actual_weather=actual_weather,
            grid_signals=grid_signals,
            workload=workload,
            config=config,
        )
        forecast_pue = _validate_pue(
            self._facility_model.pue(prepared.forecast_temperature_c),
            slots=len(prepared.slots),
            label="forecast",
        )
        actual_pue = _validate_pue(
            self._facility_model.pue(prepared.actual_temperature_c),
            slots=len(prepared.slots),
            label="actual",
        )
        risk_pue = None
        if prepared.risk_temperature_c is not None:
            risk_pue = _validate_pue(
                self._facility_model.pue(prepared.risk_temperature_c),
                slots=len(prepared.slots),
                label="risk",
            )
        policies = replay_policy_names(config)
        violations = _global_violations(prepared, config)
        if violations:
            return _infeasible_result(violations=violations, prepared=prepared, policies=policies)

        solutions = [
            _solve_policy(policy, prepared, config, forecast_pue, risk_pue, actual_pue)
            for policy in policies
        ]
        failed = [solution for solution in solutions if not solution.feasible]
        if failed:
            combined = tuple(f"{solution.policy}: {solution.message}" for solution in failed)
            return _infeasible_result(violations=combined, prepared=prepared, policies=policies)

        allocation_frames = [
            _allocations_frame(solution, prepared, config) for solution in solutions
        ]
        profile_frames = [
            _profile_frame(solution, prepared, config, forecast_pue, risk_pue, actual_pue)
            for solution in solutions
        ]
        metrics = [
            _base_metrics(solution, profile, prepared, config)
            for solution, profile in zip(solutions, profile_frames, strict=True)
        ]
        allocation_by_policy = {
            solution.policy: cast(np.ndarray, solution.allocation_kw) for solution in solutions
        }
        asap_allocation = allocation_by_policy["asap"]
        metric_by_policy = {str(row["policy"]): row for row in metrics}
        asap_metrics = metric_by_policy["asap"]
        oracle_objectives = {
            policy: float(metric_by_policy["oracle"]["realized_objective"]) for policy in policies
        }
        if config.objective_mode == "pareto_analysis":
            for policy in policies:
                price = pareto_carbon_price(policy, config)
                if price is None:
                    continue
                point_config = replace(
                    config,
                    objective_mode="monetized",
                    carbon_price_currency_per_tco2e=price,
                    pareto_carbon_prices_currency_per_tco2e=(),
                )
                oracle_solution = _solve_policy(
                    "oracle",
                    prepared,
                    point_config,
                    forecast_pue,
                    risk_pue,
                    actual_pue,
                )
                if not oracle_solution.feasible:
                    raise ContractError(f"Pareto Oracle solve failed at carbon price {price}")
                oracle_profile = _profile_frame(
                    oracle_solution,
                    prepared,
                    point_config,
                    forecast_pue,
                    risk_pue,
                    actual_pue,
                )
                oracle_row = _base_metrics(oracle_solution, oracle_profile, prepared, point_config)
                oracle_objectives[policy] = float(oracle_row["realized_objective"])
        for row in metrics:
            policy = str(row["policy"])
            shifted = (
                0.5
                * float(np.sum(np.abs(allocation_by_policy[policy] - asap_allocation)))
                * prepared.interval_hours
            )
            oracle_objective = oracle_objectives[policy]
            regret = float(row["realized_objective"]) - oracle_objective
            objective_tolerance = 1e-9 * max(
                1.0, abs(oracle_objective), abs(float(row["realized_objective"]))
            )
            if regret < -objective_tolerance:
                raise ContractError("oracle objective is not minimal under replay constraints")
            if abs(regret) <= objective_tolerance:
                regret = 0.0
            row.update(
                {
                    "shifted_energy_kwh": shifted,
                    "energy_cost_change_vs_asap": float(row["energy_cost"])
                    - float(asap_metrics["energy_cost"]),
                    "estimated_location_based_emissions_change_vs_asap_kgco2e": float(
                        row["estimated_location_based_emissions_kgco2e"]
                    )
                    - float(asap_metrics["estimated_location_based_emissions_kgco2e"]),
                    "peak_change_vs_asap_kw": float(row["peak_kw"])
                    - float(asap_metrics["peak_kw"]),
                    "objective_regret": regret,
                }
            )

        status = pd.DataFrame(
            [
                {
                    "policy": solution.policy,
                    "feasible": solution.feasible,
                    "solver_status": solution.solver_status,
                    "message": solution.message,
                }
                for solution in solutions
            ],
            columns=list(_STATUS_COLUMNS),
        )
        metrics_frame = pd.DataFrame(metrics, columns=list(_METRIC_COLUMNS))
        allocations = pd.concat(allocation_frames, ignore_index=True)
        profiles = pd.concat(profile_frames, ignore_index=True)
        numeric = metrics_frame.drop(columns=["policy"]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ContractError("replay settlement produced non-finite metrics")
        return ReplayResult(
            status=status,
            metrics=metrics_frame,
            allocations=allocations,
            profiles=profiles,
            violations={policy: () for policy in policies},
            currency=prepared.currency,
            accepted_jobs=prepared.accepted_jobs,
            future_jobs=prepared.future_jobs,
        )
