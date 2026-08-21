from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from climadc.errors import ConfigurationError, ContractError
from climadc.replay.config import ReplayStudyConfig
from climadc.replay.rolling import RollingReplayResult
from climadc.replay.study import ReplayStudyResult, ReplayStudyRunner

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DELTA_COLUMNS = (
    "energy_cost_change_vs_asap",
    "estimated_location_based_emissions_change_vs_asap_kgco2e",
    "peak_change_vs_asap_kw",
)
_DELTA_LABELS = {
    "energy_cost_change_vs_asap": "energy_cost",
    "estimated_location_based_emissions_change_vs_asap_kgco2e": "emissions",
    "peak_change_vs_asap_kw": "peak",
}
_SCENARIO_PREFIX_COLUMNS = (
    "scenario_id",
    "description",
    "study_id",
    "decision_time",
    "site_id",
    "mode",
    "currency",
    "policy",
    "feasible",
    "solver_status",
    "message",
)
_ROBUSTNESS_COLUMNS = (
    "policy",
    "scenario_count",
    "feasible_scenarios",
    "feasible_fraction",
    "cost_improvement_fraction_of_feasible",
    "mean_energy_cost_change_vs_asap",
    "worst_energy_cost_change_vs_asap",
    "worst_energy_cost_scenario",
    "emissions_improvement_fraction_of_feasible",
    "mean_estimated_location_based_emissions_change_vs_asap_kgco2e",
    "worst_estimated_location_based_emissions_change_vs_asap_kgco2e",
    "worst_emissions_scenario",
    "peak_improvement_fraction_of_feasible",
    "mean_peak_change_vs_asap_kw",
    "worst_peak_change_vs_asap_kw",
    "worst_peak_scenario",
    "pareto_efficient",
)
_PARETO_COLUMNS = (
    "mean_energy_cost_change_vs_asap",
    "mean_estimated_location_based_emissions_change_vs_asap_kgco2e",
    "mean_peak_change_vs_asap_kw",
)
_IMPROVEMENT_TOLERANCE = 1e-9


def _resolve(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


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


class ReplaySuiteScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: NonEmptyText
    description: NonEmptyText
    study: Path

    @field_validator("scenario_id")
    @classmethod
    def scenario_id_is_filesystem_safe(cls, value: str) -> str:
        if value in {".", ".."} or _SAFE_ID.fullmatch(value) is None:
            raise ValueError(
                "scenario_id must start with an alphanumeric character and contain only "
                "letters, numbers, dots, underscores, or hyphens"
            )
        return value

    def resolve_path(self, base_dir: Path) -> ReplaySuiteScenarioConfig:
        return self.model_copy(update={"study": _resolve(self.study, base_dir)})


class ReplaySuiteConfig(BaseModel):
    """Strict configuration for an equal-weight sensitivity or robustness suite."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    suite_id: NonEmptyText
    suite_type: Literal["sensitivity", "robustness"] = "sensitivity"
    robustness_dimensions: list[Literal["decision_date", "season", "location", "workload"]] = Field(
        default_factory=list
    )
    aggregation: Literal["equal_weight"] = "equal_weight"
    scenarios: list[ReplaySuiteScenarioConfig] = Field(min_length=2)
    assumptions: dict[str, object] = Field(default_factory=dict)
    limitations: list[NonEmptyText] = Field(default_factory=list)
    output_dir: Path = Path("replay-suite-runs")

    @model_validator(mode="after")
    def scenario_ids_are_unique(self) -> ReplaySuiteConfig:
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario_id values must be unique within a replay suite")
        if self.suite_type == "robustness" and not self.robustness_dimensions:
            raise ValueError(
                "robustness suites must declare at least one independent sample dimension"
            )
        if len(self.robustness_dimensions) != len(set(self.robustness_dimensions)):
            raise ValueError("robustness_dimensions must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ReplaySuiteConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = cls.model_validate(raw).resolve_paths(path.parent)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
            raise ConfigurationError(f"Invalid replay suite config {path}: {exc}") from exc
        return config

    def resolve_paths(self, base_dir: Path) -> ReplaySuiteConfig:
        scenarios = [scenario.resolve_path(base_dir) for scenario in self.scenarios]
        paths = [scenario.study for scenario in scenarios]
        if len(paths) != len(set(paths)):
            raise ValueError("replay suite study paths must resolve to distinct files")
        return self.model_copy(
            update={
                "scenarios": scenarios,
                "output_dir": _resolve(self.output_dir, base_dir),
            }
        )

    def with_output_dir(self, output_dir: Path) -> ReplaySuiteConfig:
        return self.model_copy(update={"output_dir": Path(output_dir).resolve()})


@dataclass(frozen=True)
class ReplaySuiteScenarioResult:
    scenario_id: str
    description: str
    study: ReplayStudyResult


@dataclass(frozen=True, init=False)
class ReplaySuiteResult:
    """Frozen suite output with defensive aggregate DataFrame accessors."""

    config: ReplaySuiteConfig
    scenarios: tuple[ReplaySuiteScenarioResult, ...]
    _scenario_metrics: pd.DataFrame
    _suite_metrics: pd.DataFrame
    pareto_frontier: tuple[str, ...]
    config_sha256: str
    started_at: pd.Timestamp
    currency: str
    mode: str
    policies: tuple[str, ...]

    def __init__(
        self,
        *,
        config: ReplaySuiteConfig,
        scenarios: tuple[ReplaySuiteScenarioResult, ...],
        scenario_metrics: pd.DataFrame,
        suite_metrics: pd.DataFrame | None = None,
        robustness_metrics: pd.DataFrame | None = None,
        pareto_frontier: tuple[str, ...],
        config_sha256: str,
        started_at: pd.Timestamp,
        currency: str,
        mode: str,
        policies: tuple[str, ...],
    ) -> None:
        if not scenarios:
            raise ContractError("ReplaySuiteResult requires at least one scenario")
        if suite_metrics is not None and robustness_metrics is not None:
            raise ContractError("provide suite_metrics, not both suite and robustness metrics")
        selected_metrics = suite_metrics if suite_metrics is not None else robustness_metrics
        if not isinstance(scenario_metrics, pd.DataFrame) or not isinstance(
            selected_metrics, pd.DataFrame
        ):
            raise ContractError("ReplaySuiteResult metrics must be pandas DataFrames")
        if suite_metrics is None:
            warnings.warn(
                "robustness_metrics constructor argument is deprecated; use suite_metrics",
                DeprecationWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "scenarios", tuple(scenarios))
        object.__setattr__(self, "_scenario_metrics", scenario_metrics.copy(deep=True))
        object.__setattr__(self, "_suite_metrics", selected_metrics.copy(deep=True))
        object.__setattr__(self, "pareto_frontier", tuple(pareto_frontier))
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "policies", tuple(policies))

    @property
    def scenario_metrics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._scenario_metrics.copy(deep=True))

    @property
    def suite_metrics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._suite_metrics.copy(deep=True))

    @property
    def robustness_metrics(self) -> pd.DataFrame:
        warnings.warn(
            "robustness_metrics is deprecated; use suite_metrics",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.suite_metrics


def _mode(result: ReplayStudyResult) -> str:
    return "rolling" if isinstance(result.replay, RollingReplayResult) else "single_window"


def _declared_signature(config: ReplayStudyConfig) -> tuple[object, ...]:
    rolling = config.rolling
    return (
        str(pd.Timedelta(config.replay.horizon)),
        str(pd.Timedelta(config.replay.interval)),
        rolling is not None,
        None if rolling is None else rolling.periods,
        None if rolling is None else str(rolling.step_timedelta),
        config.replay.risk_quantile is not None,
    )


def _scenario_metrics(
    scenarios: tuple[ReplaySuiteScenarioResult, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    first = scenarios[0].study
    metric_columns = tuple(str(column) for column in first.replay.metrics.columns)
    if not metric_columns or metric_columns[0] != "policy":
        raise ContractError("replay metrics must begin with a policy column")
    rows: list[dict[str, object]] = []
    expected_policies: tuple[str, ...] | None = None
    for scenario in scenarios:
        status = scenario.study.replay.status
        metrics = scenario.study.replay.metrics
        if tuple(str(column) for column in metrics.columns) != metric_columns:
            raise ConfigurationError("replay suite scenarios expose different metric schemas")
        policies = tuple(str(value) for value in status["policy"])
        if len(policies) != len(set(policies)):
            raise ContractError("replay status contains duplicate policies")
        if expected_policies is None:
            expected_policies = policies
        elif policies != expected_policies:
            raise ConfigurationError("replay suite scenarios expose different policy sets")
        metric_records = {
            str(record["policy"]): record
            for record in cast(list[dict[str, object]], metrics.to_dict(orient="records"))
        }
        if not set(metric_records).issubset(set(policies)):
            raise ContractError("replay metrics contain a policy absent from solver status")
        for status_record in cast(list[dict[str, object]], status.to_dict(orient="records")):
            policy = str(status_record["policy"])
            feasible = bool(status_record["feasible"])
            metric = metric_records.get(policy)
            if feasible and metric is None:
                raise ContractError("feasible replay policy is missing settlement metrics")
            row: dict[str, object] = {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "study_id": scenario.study.config.study_id,
                "decision_time": scenario.study.config.decision_time,
                "site_id": scenario.study.config.replay.site_id,
                "mode": _mode(scenario.study),
                "currency": scenario.study.replay.currency,
                "policy": policy,
                "feasible": feasible,
                "solver_status": int(cast(Any, status_record["solver_status"])),
                "message": str(status_record["message"]),
            }
            for column in metric_columns[1:]:
                row[column] = np.nan if metric is None else metric[column]
            rows.append(row)
    columns = [*_SCENARIO_PREFIX_COLUMNS, *metric_columns[1:]]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise ContractError("replay suite produced no scenario-policy rows")
    return cast(pd.DataFrame, frame), cast(tuple[str, ...], expected_policies)


def _delta_summary(
    rows: pd.DataFrame,
    column: str,
) -> tuple[float, float, str]:
    values = rows[column].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ContractError(f"feasible suite rows contain invalid {column} values")
    worst_position = int(np.argmax(values))
    return (
        float(np.mean(values)),
        float(values[worst_position]),
        str(rows.iloc[worst_position]["scenario_id"]),
    )


def _dominates(candidate: np.ndarray, target: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(candidate))), float(np.max(np.abs(target))))
    tolerance = _IMPROVEMENT_TOLERANCE * scale
    return bool(np.all(candidate <= target + tolerance) and np.any(candidate < target - tolerance))


def _suite_metrics(
    scenario_metrics: pd.DataFrame,
    policies: tuple[str, ...],
    scenario_count: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    records: list[dict[str, object]] = []
    for policy in policies:
        rows = scenario_metrics.loc[scenario_metrics["policy"] == policy]
        feasible = rows.loc[rows["feasible"]].reset_index(drop=True)
        feasible_count = len(feasible)
        record: dict[str, object] = {
            "policy": policy,
            "scenario_count": scenario_count,
            "feasible_scenarios": feasible_count,
            "feasible_fraction": feasible_count / scenario_count,
            "pareto_efficient": False,
        }
        for delta_column in _DELTA_COLUMNS:
            label = _DELTA_LABELS[delta_column]
            if feasible_count:
                mean, worst, worst_scenario = _delta_summary(feasible, delta_column)
                improvement_fraction = float(
                    np.mean(feasible[delta_column].to_numpy(dtype=float) < -_IMPROVEMENT_TOLERANCE)
                )
            else:
                mean = np.nan
                worst = np.nan
                worst_scenario = None
                improvement_fraction = np.nan
            if label == "energy_cost":
                record["cost_improvement_fraction_of_feasible"] = improvement_fraction
                record["mean_energy_cost_change_vs_asap"] = mean
                record["worst_energy_cost_change_vs_asap"] = worst
                record["worst_energy_cost_scenario"] = worst_scenario
            elif label == "emissions":
                record["emissions_improvement_fraction_of_feasible"] = improvement_fraction
                record["mean_estimated_location_based_emissions_change_vs_asap_kgco2e"] = mean
                record["worst_estimated_location_based_emissions_change_vs_asap_kgco2e"] = worst
                record["worst_emissions_scenario"] = worst_scenario
            else:
                record["peak_improvement_fraction_of_feasible"] = improvement_fraction
                record["mean_peak_change_vs_asap_kw"] = mean
                record["worst_peak_change_vs_asap_kw"] = worst
                record["worst_peak_scenario"] = worst_scenario
        records.append(record)

    frame = pd.DataFrame(records, columns=list(_ROBUSTNESS_COLUMNS))
    eligible = frame.loc[frame["feasible_scenarios"] == scenario_count]
    frontier: list[str] = []
    for index, row in eligible.iterrows():
        target = row[list(_PARETO_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ContractError("fully feasible policy has invalid suite-average deltas")
        dominated = False
        for other_index, other in eligible.iterrows():
            if other_index == index:
                continue
            candidate = other[list(_PARETO_COLUMNS)].to_numpy(dtype=float)
            if _dominates(candidate, target):
                dominated = True
                break
        if not dominated:
            policy = str(row["policy"])
            frontier.append(policy)
            frame.loc[frame["policy"] == policy, "pareto_efficient"] = True
    return cast(pd.DataFrame, frame), tuple(frontier)


def suite_assumptions_payload(result: ReplaySuiteResult) -> dict[str, object]:
    """Return the portable, path-independent assumptions frozen into a suite run."""

    first = result.scenarios[0].study.config
    rolling = first.rolling
    return {
        "schema_version": result.config.schema_version,
        "suite_id": result.config.suite_id,
        "suite_type": result.config.suite_type,
        "robustness_dimensions": result.config.robustness_dimensions,
        "aggregation": {
            "method": result.config.aggregation,
            "scenario_weight": "equal",
            "baseline": "ASAP within each scenario",
            "improvement_rate_denominator": "feasible scenarios for that policy",
            "improvement_rule": "signed delta < -1e-9 in the metric's published unit",
            "pareto_rule": "fully feasible policies; minimize three equal-weight mean deltas",
        },
        "comparability": {
            "mode": result.mode,
            "horizon": first.replay.horizon,
            "interval": first.replay.interval,
            "rolling_periods": None if rolling is None else rolling.periods,
            "rolling_step": None if rolling is None else rolling.step,
            "risk_policy_enabled": first.replay.risk_quantile is not None,
            "currency": result.currency,
        },
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "study_id": scenario.study.config.study_id,
                "decision_time": scenario.study.config.decision_time.isoformat(),
                "site_id": scenario.study.config.replay.site_id,
                "config_sha256": scenario.study.config_sha256,
                "input_hashes": scenario.study.input_hashes,
            }
            for scenario in result.scenarios
        ],
        "assumptions": result.config.assumptions,
        "limitations": result.config.limitations,
    }


class ReplaySuiteRunner:
    """Run comparable verified studies and summarize declared scenario variation."""

    def __init__(self, clock: Callable[[], pd.Timestamp] | None = None) -> None:
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))

    def run(self, config: ReplaySuiteConfig) -> ReplaySuiteResult:
        started_at = _exact_utc(self._clock(), "replay suite clock")
        loaded = [
            (scenario, ReplayStudyConfig.from_yaml(scenario.study)) for scenario in config.scenarios
        ]
        signatures = {_declared_signature(study) for _, study in loaded}
        if len(signatures) != 1:
            raise ConfigurationError(
                "replay suite scenarios must share horizon, interval, mode, rolling shape, "
                "and risk-policy availability"
            )
        runner = ReplayStudyRunner(clock=lambda: started_at)
        scenarios = tuple(
            ReplaySuiteScenarioResult(
                scenario_id=scenario.scenario_id,
                description=scenario.description,
                study=runner.run(study),
            )
            for scenario, study in loaded
        )
        if config.suite_type == "robustness":
            varied: set[str] = set()
            dates = {scenario.study.config.decision_time.date() for scenario in scenarios}
            if len(dates) > 1:
                varied.add("decision_date")
            seasons = {
                (scenario.study.config.decision_time.month % 12) // 3 for scenario in scenarios
            }
            if len(seasons) > 1:
                varied.add("season")
            sites = {scenario.study.config.replay.site_id for scenario in scenarios}
            if len(sites) > 1:
                varied.add("location")
            workload_hashes = {
                next(
                    value
                    for key, value in scenario.study.input_hashes.items()
                    if Path(key).name == scenario.study.config.inputs.workload.path.name
                )
                for scenario in scenarios
            }
            if len(workload_hashes) > 1:
                varied.add("workload")
            declared = set(config.robustness_dimensions)
            if not declared.intersection(varied):
                raise ConfigurationError(
                    "robustness suite does not vary any declared date, season, location, "
                    "or workload dimension"
                )
        modes = {_mode(scenario.study) for scenario in scenarios}
        currencies = {scenario.study.replay.currency for scenario in scenarios}
        if len(modes) != 1 or len(currencies) != 1:
            raise ConfigurationError("replay suite scenarios must share replay mode and currency")
        scenario_metrics, policies = _scenario_metrics(scenarios)
        suite_metrics, pareto = _suite_metrics(
            scenario_metrics,
            policies,
            len(scenarios),
        )
        provisional = ReplaySuiteResult(
            config=config,
            scenarios=scenarios,
            scenario_metrics=scenario_metrics,
            suite_metrics=suite_metrics,
            pareto_frontier=pareto,
            config_sha256="pending",
            started_at=started_at,
            currency=next(iter(currencies)),
            mode=next(iter(modes)),
            policies=policies,
        )
        config_sha256 = hashlib.sha256(
            _canonical_json(suite_assumptions_payload(provisional))
        ).hexdigest()
        return ReplaySuiteResult(
            config=config,
            scenarios=scenarios,
            scenario_metrics=scenario_metrics,
            suite_metrics=suite_metrics,
            pareto_frontier=pareto,
            config_sha256=config_sha256,
            started_at=started_at,
            currency=provisional.currency,
            mode=provisional.mode,
            policies=policies,
        )


__all__ = [
    "ReplaySuiteConfig",
    "ReplaySuiteResult",
    "ReplaySuiteRunner",
    "ReplaySuiteScenarioConfig",
    "ReplaySuiteScenarioResult",
    "suite_assumptions_payload",
]
