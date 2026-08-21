from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
from pint.errors import PintError

from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.contracts.frames import FLEXIBLE_WORKLOAD_COLUMNS
from climadc.errors import ConfigurationError, ContractError
from climadc.replay.engine import (
    ALL_POLICY_NAMES,
    POLICY_NAMES,
    ReplayEngine,
)
from climadc.replay.models import FacilityEnergyModel, ReplayConfig
from climadc.validation.units import UNIT_REGISTRY

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
_ALLOCATION_COLUMNS = (
    "site_id",
    "policy",
    "job_id",
    "valid_time",
    "power_kw",
    "energy_kwh",
    "decision_time",
)
_REMAINING_COLUMNS = (
    "policy",
    "job_id",
    "remaining_energy_kwh",
    "completed",
    "overdue",
)


def _exact_utc(value: object, field: str) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
        or str(value.tzinfo) != "UTC"
    ):
        raise ConfigurationError(f"{field} must be a scalar pandas Timestamp in exact UTC")
    return value


def _positive_periods(value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ConfigurationError("periods must be a positive integer")
    return int(value)


def _rolling_step(value: object, config: ReplayConfig) -> pd.Timedelta:
    if not isinstance(value, pd.Timedelta) or pd.isna(value) or value <= pd.Timedelta(0):
        raise ConfigurationError("step must be a positive pandas Timedelta")
    if value % config.interval != pd.Timedelta(0):
        raise ConfigurationError("step must be an integer multiple of replay interval")
    if value > config.horizon:
        raise ConfigurationError("step must not exceed replay horizon")
    return value


def _quantity(value: object, unit: object, target: str, label: str) -> float:
    try:
        result = float(
            UNIT_REGISTRY.Quantity(float(cast(float, value)), str(unit)).to(target).magnitude
        )
    except (PintError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{label} cannot be converted to {target}") from exc
    if not np.isfinite(result):
        raise ConfigurationError(f"{label} conversion produced a non-finite value")
    return result


def _normalized_workload(
    workload: FlexibleWorkloadFrame,
    config: ReplayConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not isinstance(workload, FlexibleWorkloadFrame):
        raise ContractError("workload must be a FlexibleWorkloadFrame")
    frame = workload.to_pandas()
    frame = frame.loc[frame["site_id"] == config.site_id].copy()
    frame.sort_values(["deadline", "release_time", "job_id"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    initial: dict[str, float] = {}
    for index, row in frame.iterrows():
        job_id = str(row["job_id"])
        energy = _quantity(row["energy"], row["energy_unit"], "kWh", f"job {job_id} energy")
        power = _quantity(row["max_power"], row["power_unit"], "kW", f"job {job_id} max_power")
        frame.at[index, "energy"] = energy
        frame.at[index, "energy_unit"] = "kWh"
        frame.at[index, "max_power"] = power
        frame.at[index, "power_unit"] = "kW"
        initial[job_id] = energy
    return frame.loc[:, list(FLEXIBLE_WORKLOAD_COLUMNS)], initial


def _workload_for_origin(
    base: pd.DataFrame,
    remaining: Mapping[str, float],
    tolerance_kwh: float,
) -> FlexibleWorkloadFrame:
    active = base.loc[
        base["job_id"].map(lambda job_id: remaining[str(job_id)] > tolerance_kwh)
    ].copy()
    if not active.empty:
        active["energy"] = active["job_id"].map(lambda job_id: remaining[str(job_id)])
    return FlexibleWorkloadFrame.from_pandas(active.loc[:, list(FLEXIBLE_WORKLOAD_COLUMNS)])


@dataclass(frozen=True, init=False)
class RollingReplayResult:
    """Committed receding-horizon schedules with per-policy job state."""

    _status: pd.DataFrame
    _metrics: pd.DataFrame
    _allocations: pd.DataFrame
    _profiles: pd.DataFrame
    _decisions: pd.DataFrame
    _remaining_energy: pd.DataFrame
    _violations: Mapping[str, tuple[str, ...]]
    currency: str
    accepted_jobs: int
    future_jobs: int
    decision_count: int
    commit_interval: pd.Timedelta

    def __init__(
        self,
        *,
        status: pd.DataFrame,
        metrics: pd.DataFrame,
        allocations: pd.DataFrame,
        profiles: pd.DataFrame,
        decisions: pd.DataFrame,
        remaining_energy: pd.DataFrame,
        violations: Mapping[str, tuple[str, ...]],
        currency: str,
        accepted_jobs: int,
        future_jobs: int,
        decision_count: int,
        commit_interval: pd.Timedelta,
    ) -> None:
        for name, frame in (
            ("status", status),
            ("metrics", metrics),
            ("allocations", allocations),
            ("profiles", profiles),
            ("decisions", decisions),
            ("remaining_energy", remaining_energy),
        ):
            if not isinstance(frame, pd.DataFrame):
                raise ContractError(f"RollingReplayResult {name} must be a pandas DataFrame")
        if not isinstance(currency, str) or not currency:
            raise ContractError("RollingReplayResult currency must be a non-empty string")
        if accepted_jobs < 0 or future_jobs < 0 or decision_count <= 0:
            raise ContractError("RollingReplayResult counts are invalid")
        checked_violations: dict[str, tuple[str, ...]] = {}
        for policy, items in violations.items():
            if policy not in ALL_POLICY_NAMES or any(not isinstance(item, str) for item in items):
                raise ContractError("RollingReplayResult violations are invalid")
            checked_violations[policy] = tuple(items)
        object.__setattr__(self, "_status", status.copy(deep=True))
        object.__setattr__(self, "_metrics", metrics.copy(deep=True))
        object.__setattr__(self, "_allocations", allocations.copy(deep=True))
        object.__setattr__(self, "_profiles", profiles.copy(deep=True))
        object.__setattr__(self, "_decisions", decisions.copy(deep=True))
        object.__setattr__(self, "_remaining_energy", remaining_energy.copy(deep=True))
        object.__setattr__(self, "_violations", MappingProxyType(checked_violations))
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "accepted_jobs", accepted_jobs)
        object.__setattr__(self, "future_jobs", future_jobs)
        object.__setattr__(self, "decision_count", decision_count)
        object.__setattr__(self, "commit_interval", commit_interval)

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
    def decisions(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._decisions.copy(deep=True))

    @property
    def remaining_energy(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._remaining_energy.copy(deep=True))

    @property
    def violations(self) -> dict[str, tuple[str, ...]]:
        return dict(self._violations)

    def to_metrics(self) -> pd.DataFrame:
        return self.metrics


def _committed_allocations(
    *,
    allocation: pd.DataFrame,
    base_workload: pd.DataFrame,
    policy: str,
    decision_time: pd.Timestamp,
    slots: pd.DatetimeIndex,
    interval_hours: float,
) -> pd.DataFrame:
    selected = allocation.loc[
        (allocation["policy"] == policy) & allocation["valid_time"].isin(slots)
    ]
    lookup = {
        (str(row["job_id"]), cast(pd.Timestamp, row["valid_time"])): float(row["power_kw"])
        for _, row in selected.iterrows()
    }
    if len(lookup) != len(selected):
        raise ContractError("rolling allocation contains duplicate committed job slots")
    rows = []
    site_by_job = base_workload.set_index("job_id")["site_id"].astype(str).to_dict()
    for job_id in base_workload["job_id"].astype(str).tolist():
        for slot in slots:
            power = lookup.get((job_id, slot), 0.0)
            rows.append(
                {
                    "site_id": site_by_job[job_id],
                    "policy": policy,
                    "job_id": job_id,
                    "valid_time": slot,
                    "power_kw": power,
                    "energy_kwh": power * interval_hours,
                    "decision_time": decision_time,
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows, columns=list(_ALLOCATION_COLUMNS)))


def _remaining_frame(
    policies: tuple[str, ...],
    base: pd.DataFrame,
    remaining: Mapping[str, Mapping[str, float]],
    replay_end: pd.Timestamp,
    tolerance_kwh: float,
) -> pd.DataFrame:
    rows = []
    by_job = base.set_index("job_id")
    for policy in policies:
        for job_id, value in remaining[policy].items():
            deadline = cast(pd.Timestamp, by_job.loc[job_id, "deadline"])
            rows.append(
                {
                    "policy": policy,
                    "job_id": job_id,
                    "remaining_energy_kwh": value,
                    "completed": value <= tolerance_kwh,
                    "overdue": deadline <= replay_end and value > tolerance_kwh,
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows, columns=list(_REMAINING_COLUMNS)))


def _rolling_metrics(
    *,
    policies: tuple[str, ...],
    profiles: pd.DataFrame,
    allocations: pd.DataFrame,
    base: pd.DataFrame,
    initial: Mapping[str, float],
    remaining: Mapping[str, Mapping[str, float]],
    last_origin: pd.Timestamp,
    replay_end: pd.Timestamp,
    config: ReplayConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    interval_hours = float(config.interval / pd.Timedelta(hours=1))
    known_jobs = base.loc[base["available_at"] <= last_origin, "job_id"].astype(str).tolist()
    rows: list[dict[str, float | str]] = []
    violations: dict[str, tuple[str, ...]] = {}
    for policy in policies:
        profile = profiles.loc[profiles["policy"] == policy].sort_values("valid_time")
        schedule = allocations.loc[allocations["policy"] == policy].sort_values(
            ["job_id", "valid_time"]
        )
        facility = profile["actual_facility_power_kw"].to_numpy(dtype=float)
        total_it = profile["total_it_power_kw"].to_numpy(dtype=float)
        price = profile["actual_energy_price"].to_numpy(dtype=float)
        carbon = profile["actual_carbon_kgco2e_per_kwh"].to_numpy(dtype=float)
        facility_energy = float(np.sum(facility) * interval_hours)
        it_energy = float(np.sum(total_it) * interval_hours)
        cooling_energy = facility_energy - it_energy
        emissions = float(np.sum(facility * carbon) * interval_hours)
        energy_charge = float(np.sum(facility * price) * interval_hours)
        peak = float(np.max(facility))
        demand_charge = peak * config.demand_charge_per_kw
        energy_cost = energy_charge + demand_charge
        executed = schedule.groupby("job_id", observed=True)["energy_kwh"].sum().to_dict()
        balance_errors = [
            abs(initial[job_id] - float(executed.get(job_id, 0.0)) - remaining[policy][job_id])
            for job_id in initial
        ]
        overdue = []
        for _, job in base.iterrows():
            job_id = str(job["job_id"])
            if (
                cast(pd.Timestamp, job["deadline"]) <= replay_end
                and remaining[policy][job_id] > config.tolerance_kwh
            ):
                overdue.append(
                    f"job {job_id}: {remaining[policy][job_id]:.12g} kWh remains after deadline"
                )
        violations[policy] = tuple(overdue)
        unserved = sum(
            remaining[policy][str(job["job_id"])]
            for _, job in base.iterrows()
            if cast(pd.Timestamp, job["deadline"]) <= replay_end
        )
        rows.append(
            {
                "policy": policy,
                "facility_energy_kwh": facility_energy,
                "it_energy_kwh": it_energy,
                "cooling_energy_kwh": cooling_energy,
                "estimated_location_based_emissions_kgco2e": emissions,
                "energy_charge": energy_charge,
                "demand_charge": demand_charge,
                "energy_cost": energy_cost,
                "peak_kw": peak,
                "completed_jobs": float(
                    sum(remaining[policy][job_id] <= config.tolerance_kwh for job_id in known_jobs)
                ),
                "deadline_violations": float(len(overdue)),
                "unserved_energy_kwh": float(unserved),
                "energy_balance_error_kwh": max(balance_errors, default=0.0),
                "realized_objective": config.realized_objective_for_policy(
                    policy, energy_cost, emissions
                ),
            }
        )

    by_policy = {str(row["policy"]): row for row in rows}
    asap_schedule = allocations.loc[allocations["policy"] == "asap"].sort_values(
        ["job_id", "valid_time"]
    )
    asap = by_policy["asap"]
    oracle_objective = float(by_policy["oracle"]["realized_objective"])
    for row in rows:
        policy = str(row["policy"])
        schedule = allocations.loc[allocations["policy"] == policy].sort_values(
            ["job_id", "valid_time"]
        )
        if list(schedule[["job_id", "valid_time"]].itertuples(index=False, name=None)) != list(
            asap_schedule[["job_id", "valid_time"]].itertuples(index=False, name=None)
        ):
            raise ContractError("rolling schedule keys differ across policies")
        shifted = 0.5 * float(
            np.sum(
                np.abs(
                    schedule["power_kw"].to_numpy(dtype=float)
                    - asap_schedule["power_kw"].to_numpy(dtype=float)
                )
            )
            * interval_hours
        )
        regret = float(row["realized_objective"]) - oracle_objective
        tolerance = 1e-9 * max(1.0, abs(oracle_objective), abs(float(row["realized_objective"])))
        row.update(
            {
                "shifted_energy_kwh": shifted,
                "energy_cost_change_vs_asap": float(row["energy_cost"])
                - float(asap["energy_cost"]),
                "estimated_location_based_emissions_change_vs_asap_kgco2e": float(
                    row["estimated_location_based_emissions_kgco2e"]
                )
                - float(asap["estimated_location_based_emissions_kgco2e"]),
                "peak_change_vs_asap_kw": float(row["peak_kw"]) - float(asap["peak_kw"]),
                "objective_regret": 0.0 if abs(regret) <= tolerance else regret,
            }
        )
    metrics = pd.DataFrame(rows, columns=list(_METRIC_COLUMNS))
    numeric = metrics.drop(columns=["policy"]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ContractError("rolling replay produced non-finite metrics")
    return metrics, violations


class RollingReplayEngine:
    """Re-optimize each window, commit only ``step``, and carry job energy forward."""

    def __init__(self, facility_model: FacilityEnergyModel) -> None:
        self._engine = ReplayEngine(facility_model)

    def run(
        self,
        *,
        start_time: pd.Timestamp,
        periods: int,
        step: pd.Timedelta,
        climate_forecast: ClimateForecastFrame,
        actual_weather: DCTelemetryFrame,
        grid_signals: GridSignalFrame,
        workload: FlexibleWorkloadFrame,
        config: ReplayConfig,
    ) -> RollingReplayResult:
        if config.objective_mode == "pareto_analysis":
            raise ConfigurationError("pareto_analysis is currently limited to single-window replay")
        origin = _exact_utc(start_time, "start_time")
        count = _positive_periods(periods)
        commit_interval = _rolling_step(step, config)
        policies = ALL_POLICY_NAMES if config.risk_quantile is not None else POLICY_NAMES
        base, initial = _normalized_workload(workload, config)
        remaining: dict[str, dict[str, float]] = {policy: dict(initial) for policy in policies}
        origins = pd.date_range(origin, periods=count, freq=commit_interval)
        commit_slots = int(commit_interval // config.interval)
        interval_hours = float(config.interval / pd.Timedelta(hours=1))
        allocation_frames: list[pd.DataFrame] = []
        profile_frames: list[pd.DataFrame] = []
        decision_rows: list[dict[str, object]] = []
        currency: str | None = None

        for decision_time in origins:
            slots = pd.date_range(decision_time, periods=commit_slots, freq=config.interval)
            for policy in policies:
                state = _workload_for_origin(base, remaining[policy], config.tolerance_kwh)
                window = self._engine.run(
                    decision_time=decision_time,
                    climate_forecast=climate_forecast,
                    actual_weather=actual_weather,
                    grid_signals=grid_signals,
                    workload=state,
                    config=config,
                )
                if currency is None:
                    currency = window.currency
                elif currency != window.currency:
                    raise ContractError("rolling replay currency changed between decisions")
                status = window.status.loc[window.status["policy"] == policy]
                if len(status) != 1 or not bool(status.iloc[0]["feasible"]):
                    message = (
                        "missing policy status" if status.empty else str(status.iloc[0]["message"])
                    )
                    raise ConfigurationError(
                        f"rolling policy {policy} is infeasible at {decision_time.isoformat()}: "
                        f"{message}"
                    )
                committed = _committed_allocations(
                    allocation=window.allocations,
                    base_workload=base,
                    policy=policy,
                    decision_time=decision_time,
                    slots=slots,
                    interval_hours=interval_hours,
                )
                for job_id, executed in (
                    committed.groupby("job_id", observed=True)["energy_kwh"].sum().items()
                ):
                    key = str(job_id)
                    amount = float(executed)
                    if amount > remaining[policy][key] + config.tolerance_kwh:
                        raise ContractError(
                            "rolling replay executed more than remaining job energy"
                        )
                    remaining[policy][key] = max(0.0, remaining[policy][key] - amount)
                allocation_frames.append(committed)
                profile = window.profiles.loc[
                    (window.profiles["policy"] == policy)
                    & window.profiles["valid_time"].isin(slots)
                ].copy()
                if len(profile) != len(slots):
                    raise ContractError("rolling replay is missing committed profile slots")
                profile["decision_time"] = decision_time
                profile_frames.append(profile)
                decision_rows.append(
                    {
                        "decision_time": decision_time,
                        "policy": policy,
                        "feasible": True,
                        "solver_status": int(status.iloc[0]["solver_status"]),
                        "message": str(status.iloc[0]["message"]),
                        "accepted_jobs": window.accepted_jobs,
                        "future_jobs": window.future_jobs,
                        "committed_energy_kwh": float(committed["energy_kwh"].sum()),
                    }
                )

        allocations = pd.concat(allocation_frames, ignore_index=True)
        profiles = pd.concat(profile_frames, ignore_index=True)
        decisions = pd.DataFrame(decision_rows)
        replay_end = origins[-1] + commit_interval
        last_origin = origins[-1]
        metrics, violations = _rolling_metrics(
            policies=policies,
            profiles=profiles,
            allocations=allocations,
            base=base,
            initial=initial,
            remaining=remaining,
            last_origin=last_origin,
            replay_end=replay_end,
            config=config,
        )
        status = pd.DataFrame(
            [
                {
                    "policy": policy,
                    "feasible": not violations[policy],
                    "solver_status": 0 if not violations[policy] else -1,
                    "message": (
                        f"{count} rolling decisions solved"
                        if not violations[policy]
                        else "; ".join(violations[policy])
                    ),
                }
                for policy in policies
            ]
        )
        remaining_energy = _remaining_frame(
            policies, base, remaining, replay_end, config.tolerance_kwh
        )
        accepted_jobs = int((base["available_at"] <= last_origin).sum())
        future_jobs = int((base["available_at"] > last_origin).sum())
        return RollingReplayResult(
            status=status,
            metrics=metrics,
            allocations=allocations,
            profiles=profiles,
            decisions=decisions,
            remaining_energy=remaining_energy,
            violations=violations,
            currency=currency or "unknown",
            accepted_jobs=accepted_jobs,
            future_jobs=future_jobs,
            decision_count=count,
            commit_interval=commit_interval,
        )
