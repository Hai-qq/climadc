from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from html import unescape
from numbers import Real
from pathlib import Path, PurePosixPath
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
)

from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ClimaDCError, ConfigurationError
from climadc.evidence.checksums import (
    CHECKSUM_FILE,
    artifact_files,
    safe_relative_path,
    verify_checksums,
)
from climadc.evidence.manifest import EnvironmentRecord, RunManifest
from climadc.replay.manifest import SourceManifest
from climadc.reporting.artifacts import resolve_run_path
from climadc.validation.units import UNIT_REGISTRY

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_RUN_MANIFEST = "run-manifest.json"
_ENVIRONMENT = "environment.json"
_REPLAY_V1_FILES = frozenset(
    {
        "assumptions.yaml",
        "source-manifest.yaml",
        "lineage.json",
        "climate-forecast.parquet",
        "actual-weather.parquet",
        "grid-signals.parquet",
        "workload.parquet",
        "schedules.parquet",
        "profiles.parquet",
        "solver-status.json",
        "replay-metrics.json",
        "report.html",
    }
)
_REPLAY_V2_FILES = frozenset({*_REPLAY_V1_FILES, _RUN_MANIFEST, _ENVIRONMENT, CHECKSUM_FILE})
_BENCHMARK_V1_FILES = frozenset(
    {
        "run.yaml",
        "lineage.json",
        "splits.parquet",
        "predictions.parquet",
        "metrics.json",
        "leakage-report.json",
        "dataset-card.md",
        "report.html",
    }
)
_BENCHMARK_V2_FILES = frozenset({*_BENCHMARK_V1_FILES, _RUN_MANIFEST, _ENVIRONMENT, CHECKSUM_FILE})
_SUITE_V1_TOP_LEVEL = frozenset(
    {
        "suite.yaml",
        "lineage.json",
        "scenario-index.json",
        "scenario-metrics.parquet",
        "robustness-metrics.json",
        "pareto-frontier.json",
        "report.html",
        "scenarios",
    }
)
_SUITE_V2_TOP_LEVEL = frozenset(
    {
        "suite.yaml",
        "lineage.json",
        "scenario-index.json",
        "scenario-metrics.parquet",
        "suite-metrics.json",
        "pareto-frontier.json",
        "report.html",
        "scenarios",
        _RUN_MANIFEST,
        _ENVIRONMENT,
        CHECKSUM_FILE,
    }
)
_REPORT_DATA = re.compile(
    r'<template\s+id="climadc-report-data">(?P<payload>.*?)</template>', re.DOTALL
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUITE_PREFIX_COLUMNS = (
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
_SUITE_SUMMARIES = (
    (
        "energy_cost_change_vs_asap",
        "cost_improvement_fraction_of_feasible",
        "mean_energy_cost_change_vs_asap",
        "worst_energy_cost_change_vs_asap",
        "worst_energy_cost_scenario",
    ),
    (
        "estimated_location_based_emissions_change_vs_asap_kgco2e",
        "emissions_improvement_fraction_of_feasible",
        "mean_estimated_location_based_emissions_change_vs_asap_kgco2e",
        "worst_estimated_location_based_emissions_change_vs_asap_kgco2e",
        "worst_emissions_scenario",
    ),
    (
        "peak_change_vs_asap_kw",
        "peak_improvement_fraction_of_feasible",
        "mean_peak_change_vs_asap_kw",
        "worst_peak_change_vs_asap_kw",
        "worst_peak_scenario",
    ),
)
_SUITE_PARETO_FIELDS = (
    "mean_energy_cost_change_vs_asap",
    "mean_estimated_location_based_emissions_change_vs_asap_kgco2e",
    "mean_peak_change_vs_asap_kw",
)
_SUITE_IMPROVEMENT_TOLERANCE = 1e-9


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: NonEmptyText
    status: Literal["pass", "fail", "warning", "skipped"]
    message: NonEmptyText
    files: list[NonEmptyText] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Stable machine-readable result returned instead of relying on verifier process state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    valid: bool
    run_type: Literal["benchmark", "replay", "replay_suite", "unknown"]
    artifact_schema_version: Literal["1", "2", "unknown"]
    legacy: bool
    checks: list[VerificationCheck]
    limitations: list[NonEmptyText] = Field(default_factory=list)

    def to_json(self) -> str:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


class _Collector:
    def __init__(self) -> None:
        self.checks: list[VerificationCheck] = []

    def record(
        self,
        check_id: str,
        status: Literal["pass", "fail", "warning", "skipped"],
        message: str,
        *files: str,
    ) -> None:
        self.checks.append(
            VerificationCheck(
                check_id=check_id,
                status=status,
                message=message,
                files=list(files),
            )
        )

    def run(
        self,
        check_id: str,
        message: str,
        files: tuple[str, ...],
        action: Callable[[], object],
    ) -> object | None:
        try:
            result = action()
        # Artifact parsers (PyYAML, PyArrow, pandas, Pydantic) expose different
        # exception families. Malformed untrusted artifacts must become a failed
        # check instead of escaping the verifier process.
        except Exception as exc:
            self.record(check_id, "fail", str(exc), *files)
            return None
        self.record(check_id, "pass", message, *files)
        return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{path.name} must contain one JSON object with string keys")
    _require_finite(payload, path.name)
    return cast(dict[str, object], payload)


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{path.name} must contain one mapping with string keys")
    _require_finite(payload, path.name)
    return cast(dict[str, object], payload)


def _require_finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for nested in value.values():
            _require_finite(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_finite(nested, label)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _manifest(path: Path) -> RunManifest:
    try:
        return RunManifest.model_validate(_load_json(path))
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid run-manifest.json: {exc}") from exc


def _environment(path: Path) -> EnvironmentRecord:
    try:
        return EnvironmentRecord.model_validate(_load_json(path))
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid environment.json: {exc}") from exc


def _exact_top_level(root: Path, expected: frozenset[str], label: str) -> None:
    actual = {path.name for path in root.iterdir()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ConfigurationError(f"{label} artifact set mismatch: missing={missing}, extra={extra}")
    empty = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.stat().st_size == 0
    )
    if empty:
        raise ConfigurationError(f"{label} contains empty artifacts: {empty}")


def _validate_manifest_files(root: Path, manifest: RunManifest) -> None:
    actual = list(artifact_files(root))
    if manifest.artifacts != actual:
        missing = sorted(set(manifest.artifacts) - set(actual))
        extra = sorted(set(actual) - set(manifest.artifacts))
        raise ConfigurationError(
            f"run-manifest.json artifact list mismatch: missing={missing}, extra={extra}"
        )
    if _RUN_MANIFEST not in manifest.artifacts or CHECKSUM_FILE not in manifest.artifacts:
        raise ConfigurationError("run-manifest.json must declare itself and checksums.sha256")


def _validate_environment_consistency(
    environment: EnvironmentRecord, manifest: RunManifest
) -> None:
    version = environment.packages.get("climadc")
    if version != manifest.climadc_version:
        raise ConfigurationError(
            "environment.json packages.climadc differs from run-manifest.json climadc_version"
        )
    required = {"numpy", "pandas", "scipy", "pyarrow", "pydantic", "pint", "climadc"}
    missing = sorted(required - set(environment.packages))
    if missing:
        raise ConfigurationError(f"environment.json packages is missing {missing}")


def _numeric_frame(frame: pd.DataFrame, artifact: str) -> None:
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            values = series.dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ConfigurationError(f"{artifact} column {column!r} contains NaN or Infinity")


def _require_exact_utc(frame: pd.DataFrame, artifact: str, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in frame:
            raise ConfigurationError(f"{artifact} is missing timestamp column {column!r}")
        dtype = frame[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
            raise ConfigurationError(f"{artifact} column {column!r} must use exact UTC dtype")
        if bool(frame[column].isna().any()):
            raise ConfigurationError(f"{artifact} column {column!r} must not contain null times")


def _read_replay_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_parquet(root / name)
        for name in (
            "climate-forecast.parquet",
            "actual-weather.parquet",
            "grid-signals.parquet",
            "workload.parquet",
            "schedules.parquet",
            "profiles.parquet",
        )
    }
    for name, frame in frames.items():
        _numeric_frame(frame, name)
    ClimateForecastFrame.from_pandas(frames["climate-forecast.parquet"])
    DCTelemetryFrame.from_pandas(frames["actual-weather.parquet"])
    GridSignalFrame.from_pandas(frames["grid-signals.parquet"])
    FlexibleWorkloadFrame.from_pandas(frames["workload.parquet"])
    schedule_times: tuple[str, ...] = ("valid_time",)
    profile_times: tuple[str, ...] = ("valid_time",)
    if "decision_time" in frames["schedules.parquet"]:
        schedule_times = ("decision_time", "valid_time")
    if "decision_time" in frames["profiles.parquet"]:
        profile_times = ("decision_time", "valid_time")
    _require_exact_utc(frames["schedules.parquet"], "schedules.parquet", schedule_times)
    _require_exact_utc(frames["profiles.parquet"], "profiles.parquet", profile_times)
    return frames


def _records(payload: dict[str, object], field: str, artifact: str) -> list[dict[str, object]]:
    raw = payload.get(field)
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ConfigurationError(f"{artifact} field {field!r} must be an array of objects")
    return cast(list[dict[str, object]], raw)


def _metric_field(records: list[dict[str, object]]) -> tuple[str, str]:
    if records and "estimated_location_based_emissions_kgco2e" in records[0]:
        return (
            "estimated_location_based_emissions_kgco2e",
            "estimated_location_based_emissions_change_vs_asap_kgco2e",
        )
    return "emissions_kgco2e", "emissions_change_vs_asap_kgco2e"


def _finite_number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ConfigurationError(f"{label} must be {qualifier}")
    return number


def _objective_contract(value: object, artifact: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{artifact} objective must be an object")
    objective = cast(dict[str, object], value)
    mode = objective.get("mode")
    if mode == "legacy_unscaled":
        expected = {
            "version",
            "mode",
            "cost_weight",
            "carbon_weight",
            "demand_charge_per_kw",
            "dimensionally_unscaled",
            "deprecated",
        }
        if set(objective) != expected or objective.get("version") != "legacy-1":
            raise ConfigurationError(f"{artifact} legacy objective schema is invalid")
        for field in ("cost_weight", "carbon_weight", "demand_charge_per_kw"):
            _finite_number(objective[field], f"{artifact} objective.{field}", nonnegative=True)
        if (
            objective.get("dimensionally_unscaled") is not True
            or objective.get("deprecated") is not True
        ):
            raise ConfigurationError(f"{artifact} legacy objective flags are invalid")
        return objective
    if objective.get("version") != "1":
        raise ConfigurationError(f"{artifact} objective version must be '1'")
    if mode == "monetized":
        expected = {
            "version",
            "mode",
            "carbon_price_currency_per_tco2e",
            "demand_charge_per_kw",
        }
        if set(objective) != expected:
            raise ConfigurationError(f"{artifact} monetized objective schema is invalid")
        for field in ("carbon_price_currency_per_tco2e", "demand_charge_per_kw"):
            _finite_number(objective[field], f"{artifact} objective.{field}", nonnegative=True)
        return objective
    if mode == "epsilon_constraint":
        expected = {
            "version",
            "mode",
            "emissions_upper_bound_kgco2e",
            "peak_upper_bound_kw",
            "demand_charge_per_kw",
        }
        if set(objective) != expected:
            raise ConfigurationError(f"{artifact} epsilon objective schema is invalid")
        bounds = []
        for field in ("emissions_upper_bound_kgco2e", "peak_upper_bound_kw"):
            bound = objective[field]
            if bound is not None:
                numeric = _finite_number(bound, f"{artifact} objective.{field}")
                if numeric <= 0.0:
                    raise ConfigurationError(f"{artifact} objective.{field} must be positive")
                bounds.append(numeric)
        if not bounds:
            raise ConfigurationError(f"{artifact} epsilon objective requires a physical bound")
        _finite_number(
            objective["demand_charge_per_kw"],
            f"{artifact} objective.demand_charge_per_kw",
            nonnegative=True,
        )
        return objective
    if mode == "pareto_analysis":
        expected = {
            "version",
            "mode",
            "carbon_prices_currency_per_tco2e",
            "demand_charge_per_kw",
        }
        if set(objective) != expected:
            raise ConfigurationError(f"{artifact} Pareto objective schema is invalid")
        raw_prices = objective["carbon_prices_currency_per_tco2e"]
        if not isinstance(raw_prices, list):
            raise ConfigurationError(f"{artifact} Pareto prices must be an array")
        prices = [
            _finite_number(price, f"{artifact} Pareto price", nonnegative=True)
            for price in raw_prices
        ]
        if len(prices) < 2 or len(prices) != len(set(prices)) or prices != sorted(prices):
            raise ConfigurationError(f"{artifact} Pareto prices must be unique ascending points")
        _finite_number(
            objective["demand_charge_per_kw"],
            f"{artifact} objective.demand_charge_per_kw",
            nonnegative=True,
        )
        return objective
    raise ConfigurationError(f"{artifact} objective mode is unsupported")


def _objective_from_assumptions(assumptions: dict[str, object]) -> dict[str, object]:
    replay = assumptions.get("replay")
    if not isinstance(replay, dict):
        raise ConfigurationError("assumptions.yaml field 'replay' must be an object")
    objective = replay.get("objective")
    if objective is not None:
        return _objective_contract(objective, "assumptions.yaml")
    cost_weight = replay.get("cost_weight")
    carbon_weight = replay.get("carbon_weight")
    demand_charge = replay.get("demand_charge_per_kw")
    payload: dict[str, object] = {
        "version": "legacy-1",
        "mode": "legacy_unscaled",
        "cost_weight": 1.0 if cost_weight is None else cost_weight,
        "carbon_weight": 1.0 if carbon_weight is None else carbon_weight,
        "demand_charge_per_kw": 0.0 if demand_charge is None else demand_charge,
        "dimensionally_unscaled": True,
        "deprecated": True,
    }
    return _objective_contract(payload, "assumptions.yaml")


def _objective_consistency(
    assumptions: dict[str, object], metrics_payload: dict[str, object]
) -> None:
    if metrics_payload.get("objective") != _objective_from_assumptions(assumptions):
        raise ConfigurationError("replay-metrics.json objective differs from assumptions.yaml")


def _replay_metrics_payload(path: Path, *, legacy: bool) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("schema_version") != "1":
        raise ConfigurationError("replay-metrics.json schema_version must be '1'")
    mode = payload.get("mode")
    if mode not in {"single_window", "rolling"}:
        raise ConfigurationError("replay-metrics.json mode is invalid")
    decision_count = _strict_integer(
        payload.get("decision_count"), "replay-metrics.json decision_count"
    )
    if decision_count < 1:
        raise ConfigurationError("replay-metrics.json decision_count must be positive")
    commit_interval = payload.get("commit_interval")
    if (mode == "single_window" and commit_interval is not None) or (
        mode == "rolling" and (not isinstance(commit_interval, str) or not commit_interval)
    ):
        raise ConfigurationError("replay-metrics.json commit_interval disagrees with mode")
    for field in ("study_id", "currency"):
        if not isinstance(payload.get(field), str) or not cast(str, payload[field]).strip():
            raise ConfigurationError(f"replay-metrics.json {field} must be non-empty text")
    for field in ("accepted_jobs", "future_jobs"):
        if _strict_integer(payload.get(field), f"replay-metrics.json {field}") < 0:
            raise ConfigurationError(f"replay-metrics.json {field} must be non-negative")
    units = payload.get("units")
    if not isinstance(units, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in units.items()
    ):
        raise ConfigurationError("replay-metrics.json units must map names to unit strings")
    records = _records(payload, "policies", "replay-metrics.json")
    if not records:
        raise ConfigurationError("replay-metrics.json policies must not be empty")
    emissions_field, emissions_delta_field = _metric_field(records)
    expected_fields = {
        "policy",
        "facility_energy_kwh",
        "it_energy_kwh",
        "cooling_energy_kwh",
        emissions_field,
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
        emissions_delta_field,
        "peak_change_vs_asap_kw",
        "realized_objective",
        "objective_regret",
    }
    policies: list[str] = []
    for record in records:
        if set(record) != expected_fields:
            raise ConfigurationError("replay-metrics.json policy record schema is invalid")
        policy = record.get("policy")
        if not isinstance(policy, str) or not policy:
            raise ConfigurationError("replay-metrics.json policy must be non-empty text")
        policies.append(policy)
        for field in expected_fields - {"policy"}:
            _finite_number(record[field], f"replay-metrics.json {policy}.{field}")
    if len(policies) != len(set(policies)):
        raise ConfigurationError("replay-metrics.json contains duplicate policies")
    objective = payload.get("objective")
    if objective is None:
        if not legacy:
            raise ConfigurationError("replay-metrics.json v2 must contain an objective contract")
    else:
        _objective_contract(objective, "replay-metrics.json")
    pareto = payload.get("pareto_frontier", [] if legacy else None)
    if not isinstance(pareto, list) or any(not isinstance(item, dict) for item in pareto):
        raise ConfigurationError("replay-metrics.json pareto_frontier must be an array of objects")
    violations = payload.get("violations")
    if (
        not isinstance(violations, dict)
        or set(violations) != set(policies)
        or any(
            not isinstance(items, list) or any(not isinstance(item, str) for item in items)
            for items in violations.values()
        )
    ):
        raise ConfigurationError("replay-metrics.json violations must cover every policy")
    return payload


def _close(actual: object, expected: float, label: str) -> None:
    try:
        value = float(cast(Any, actual))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{label} must be a finite number") from exc
    tolerance = 1e-8 * max(1.0, abs(value), abs(expected))
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ConfigurationError(f"{label} mismatch: published={value}, reconstructed={expected}")


def _replay_keys(
    schedules: pd.DataFrame, profiles: pd.DataFrame, policies: set[str]
) -> tuple[list[str], list[str]]:
    schedule_keys = ["job_id", "valid_time"]
    profile_keys = ["valid_time"]
    if "decision_time" in schedules:
        schedule_keys.insert(0, "decision_time")
    if "decision_time" in profiles:
        profile_keys.insert(0, "decision_time")
    if schedules.duplicated(["policy", *schedule_keys]).any():
        raise ConfigurationError("schedules.parquet contains duplicate policy/job/time keys")
    if profiles.duplicated(["policy", *profile_keys]).any():
        raise ConfigurationError("profiles.parquet contains duplicate policy/time keys")
    for frame, keys, artifact in (
        (schedules, schedule_keys, "schedules.parquet"),
        (profiles, profile_keys, "profiles.parquet"),
    ):
        if set(frame["policy"].astype(str)) != policies:
            raise ConfigurationError(f"{artifact} policy set differs from replay-metrics.json")
        baseline = list(
            frame.loc[frame["policy"] == "asap", keys].itertuples(index=False, name=None)
        )
        for policy in sorted(policies):
            actual = list(
                frame.loc[frame["policy"] == policy, keys].itertuples(index=False, name=None)
            )
            if actual != baseline:
                raise ConfigurationError(f"{artifact} keys differ for policy {policy}")
    return schedule_keys, profile_keys


def _workload_limits(
    schedules: pd.DataFrame,
    workload: pd.DataFrame,
    solver_status: dict[str, object],
    interval_hours: float,
    tolerance: float,
) -> None:
    jobs = workload.set_index("job_id", drop=False)
    for _, row in schedules.iterrows():
        job_id = str(row["job_id"])
        if job_id not in jobs.index:
            raise ConfigurationError(f"schedules.parquet references unknown job {job_id!r}")
        job = jobs.loc[job_id]
        if isinstance(job, pd.DataFrame):
            raise ConfigurationError(f"workload.parquet contains duplicate job {job_id!r}")
        power = float(row["power_kw"])
        energy = float(row["energy_kwh"])
        if not math.isfinite(power) or power < -tolerance:
            raise ConfigurationError(f"schedules.parquet job {job_id} has invalid power_kw")
        expected_energy = power * interval_hours
        _close(energy, expected_energy, f"schedules.parquet job {job_id} energy_kwh")
        max_power = float(
            UNIT_REGISTRY.Quantity(float(job["max_power"]), str(job["power_unit"]))
            .to("kW")
            .magnitude
        )
        if power > max_power + tolerance:
            raise ConfigurationError(
                f"schedules.parquet job {job_id} exceeds max_power {max_power} kW"
            )
        slot = cast(pd.Timestamp, row["valid_time"])
        if power > tolerance and not (
            cast(pd.Timestamp, job["release_time"]) <= slot < cast(pd.Timestamp, job["deadline"])
        ):
            raise ConfigurationError(
                f"schedules.parquet job {job_id} executes outside release/deadline window"
            )
    executed = schedules.groupby(["policy", "job_id"], observed=True)["energy_kwh"].sum()
    expected_pairs = {(str(policy), str(job_id)) for policy, job_id in executed.index}
    mode = solver_status.get("mode")
    if mode not in {"single_window", "rolling"}:
        raise ConfigurationError("solver-status.json mode must be 'single_window' or 'rolling'")
    remaining_records = _records(solver_status, "remaining_energy", "solver-status.json")
    remaining: dict[tuple[str, str], float] = {}
    for record in remaining_records:
        remaining_policy = record.get("policy")
        remaining_job_id = record.get("job_id")
        value = record.get("remaining_energy_kwh")
        if not isinstance(remaining_policy, str) or not remaining_policy.strip():
            raise ConfigurationError("solver-status.json remaining_energy policy must be text")
        if not isinstance(remaining_job_id, str) or not remaining_job_id.strip():
            raise ConfigurationError("solver-status.json remaining_energy job_id must be text")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigurationError("solver-status.json remaining_energy_kwh must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < -tolerance:
            raise ConfigurationError(
                "solver-status.json remaining_energy_kwh must be finite and non-negative"
            )
        if not isinstance(record.get("completed"), bool) or not isinstance(
            record.get("overdue"), bool
        ):
            raise ConfigurationError(
                "solver-status.json remaining_energy completed/overdue must be boolean"
            )
        if bool(record["completed"]) != (numeric_value <= tolerance):
            raise ConfigurationError(
                "solver-status.json remaining_energy completed flag is inconsistent"
            )
        remaining_key = (remaining_policy, remaining_job_id)
        if remaining_key in remaining:
            raise ConfigurationError(
                "solver-status.json remaining_energy contains duplicate policy/job pairs"
            )
        remaining[remaining_key] = numeric_value
    if mode == "single_window" and remaining:
        raise ConfigurationError("solver-status.json single_window remaining_energy must be empty")
    if mode == "rolling" and set(remaining) != expected_pairs:
        raise ConfigurationError(
            "solver-status.json rolling remaining_energy pairs differ from schedules.parquet"
        )
    for executed_key, actual in executed.items():
        if not isinstance(executed_key, tuple) or len(executed_key) != 2:
            raise ConfigurationError("schedules.parquet energy grouping produced invalid keys")
        policy, job_id = executed_key
        job = jobs.loc[str(job_id)]
        if isinstance(job, pd.DataFrame):
            raise ConfigurationError(f"workload.parquet contains duplicate job {job_id!r}")
        expected = float(
            UNIT_REGISTRY.Quantity(float(job["energy"]), str(job["energy_unit"]))
            .to("kWh")
            .magnitude
        )
        unsettled = remaining.get((str(policy), str(job_id)), 0.0)
        _close(
            float(actual) + unsettled,
            expected,
            f"schedules.parquet {policy}/{job_id} job energy conservation",
        )


def _profile_and_metric_invariants(
    *,
    assumptions: dict[str, object],
    metric_records: list[dict[str, object]],
    schedules: pd.DataFrame,
    profiles: pd.DataFrame,
    workload: pd.DataFrame,
    solver_status: dict[str, object],
) -> None:
    replay = assumptions.get("replay")
    if not isinstance(replay, dict):
        raise ConfigurationError("assumptions.yaml field 'replay' must be an object")
    interval_hours = float(pd.Timedelta(cast(str, replay["interval"])) / pd.Timedelta(hours=1))
    tolerance = float(replay.get("tolerance_kwh", 1e-7))
    capacity = float(replay["it_capacity_kw"])
    legacy_demand_charge = replay.get("demand_charge_per_kw")
    demand_charge = 0.0 if legacy_demand_charge is None else float(legacy_demand_charge)
    objective = replay.get("objective")
    if isinstance(objective, dict):
        demand_charge = float(objective.get("demand_charge_per_kw", demand_charge))
    policies = {str(record.get("policy")) for record in metric_records}
    if "asap" not in policies:
        raise ConfigurationError("replay-metrics.json policies must contain the ASAP baseline")
    _replay_keys(schedules, profiles, policies)
    _workload_limits(schedules, workload, solver_status, interval_hours, tolerance)
    emissions_field, emissions_delta_field = _metric_field(metric_records)
    by_policy = {str(record["policy"]): record for record in metric_records}
    if len(by_policy) != len(metric_records):
        raise ConfigurationError("replay-metrics.json contains duplicate policy records")
    asap_schedule = schedules.loc[schedules["policy"] == "asap"].sort_values(
        [column for column in ("decision_time", "job_id", "valid_time") if column in schedules]
    )
    asap_metrics = by_policy["asap"]
    for policy, record in sorted(by_policy.items()):
        profile = profiles.loc[profiles["policy"] == policy].sort_values(
            [column for column in ("decision_time", "valid_time") if column in profiles]
        )
        schedule = schedules.loc[schedules["policy"] == policy].sort_values(
            [column for column in ("decision_time", "job_id", "valid_time") if column in schedules]
        )
        numeric_columns = (
            "actual_facility_power_kw",
            "total_it_power_kw",
            "flexible_it_power_kw",
            "fixed_it_power_kw",
            "actual_pue",
            "actual_energy_price",
            "actual_carbon_kgco2e_per_kwh",
        )
        for column in numeric_columns:
            if column not in profile or not pd.api.types.is_numeric_dtype(profile[column].dtype):
                raise ConfigurationError(f"profiles.parquet column {column!r} must be numeric")
        facility = profile["actual_facility_power_kw"].to_numpy(dtype=float)
        total_it = profile["total_it_power_kw"].to_numpy(dtype=float)
        flexible = profile["flexible_it_power_kw"].to_numpy(dtype=float)
        fixed = profile["fixed_it_power_kw"].to_numpy(dtype=float)
        pue = profile["actual_pue"].to_numpy(dtype=float)
        price = profile["actual_energy_price"].to_numpy(dtype=float)
        carbon = profile["actual_carbon_kgco2e_per_kwh"].to_numpy(dtype=float)
        time_keys = [
            column for column in ("decision_time", "valid_time") if column in profile.columns
        ]
        scheduled_power = (
            schedule.groupby(time_keys, sort=False, observed=True)["power_kw"]
            .sum()
            .reset_index()
            .sort_values(time_keys)
        )
        if list(scheduled_power[time_keys].itertuples(index=False, name=None)) != list(
            profile[time_keys].itertuples(index=False, name=None)
        ):
            raise ConfigurationError(
                f"schedules.parquet and profiles.parquet time keys differ for policy {policy}"
            )
        if not np.allclose(
            scheduled_power["power_kw"].to_numpy(dtype=float),
            flexible,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ConfigurationError(
                f"schedules.parquet and profiles.parquet flexible power differ for policy {policy}"
            )
        if np.any(total_it > capacity + tolerance):
            raise ConfigurationError(f"profiles.parquet policy {policy} exceeds IT capacity")
        if not np.allclose(total_it, flexible + fixed, rtol=0.0, atol=tolerance):
            raise ConfigurationError(f"profiles.parquet policy {policy} violates IT power balance")
        if not np.allclose(facility, pue * total_it, rtol=0.0, atol=tolerance):
            raise ConfigurationError(
                f"profiles.parquet policy {policy} violates facility PUE power relation"
            )
        facility_energy = float(np.sum(facility) * interval_hours)
        it_energy = float(np.sum(total_it) * interval_hours)
        cooling_energy = facility_energy - it_energy
        emissions = float(np.sum(facility * carbon) * interval_hours)
        energy_charge = float(np.sum(facility * price) * interval_hours)
        peak = float(np.max(facility))
        demand = peak * demand_charge
        reconstructed = {
            "facility_energy_kwh": facility_energy,
            "it_energy_kwh": it_energy,
            "cooling_energy_kwh": cooling_energy,
            emissions_field: emissions,
            "energy_charge": energy_charge,
            "demand_charge": demand,
            "energy_cost": energy_charge + demand,
            "peak_kw": peak,
        }
        for field, expected in reconstructed.items():
            if field not in record:
                raise ConfigurationError(f"replay-metrics.json policy {policy} misses {field}")
            _close(record[field], expected, f"replay-metrics.json {policy}.{field}")
        shifted = 0.5 * float(
            np.sum(
                np.abs(
                    schedule["power_kw"].to_numpy(dtype=float)
                    - asap_schedule["power_kw"].to_numpy(dtype=float)
                )
            )
            * interval_hours
        )
        _close(
            record["shifted_energy_kwh"],
            shifted,
            f"replay-metrics.json {policy}.shifted_energy_kwh",
        )
        _close(
            record["energy_cost_change_vs_asap"],
            float(cast(Any, record["energy_cost"])) - float(cast(Any, asap_metrics["energy_cost"])),
            f"replay-metrics.json {policy}.energy_cost_change_vs_asap",
        )
        _close(
            record[emissions_delta_field],
            float(cast(Any, record[emissions_field]))
            - float(cast(Any, asap_metrics[emissions_field])),
            f"replay-metrics.json {policy}.{emissions_delta_field}",
        )
        _close(
            record["peak_change_vs_asap_kw"],
            float(cast(Any, record["peak_kw"])) - float(cast(Any, asap_metrics["peak_kw"])),
            f"replay-metrics.json {policy}.peak_change_vs_asap_kw",
        )


def _source_manifest(root: Path, manifest: RunManifest | None) -> None:
    source = SourceManifest.from_yaml(root / "source-manifest.yaml")
    required = {
        root / "climate-forecast.parquet",
        root / "actual-weather.parquet",
        root / "grid-signals.parquet",
        root / "workload.parquet",
    }
    hashes = source.validate_files(root, required)
    if manifest is not None and hashes != manifest.input_hashes:
        raise ConfigurationError(
            "run-manifest.json input_hashes differs from source-manifest.yaml canonical inputs"
        )


def _report_payload(
    path: Path,
    expected: list[dict[str, object]],
    expected_objective: object,
) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    if "<script" in lowered or "<link" in lowered or " src=" in lowered:
        raise ConfigurationError("report.html contains an external or executable asset")
    match = _REPORT_DATA.search(text)
    if match is None:
        raise ConfigurationError("report.html is missing climadc-report-data")
    payload = json.loads(unescape(match.group("payload")), parse_constant=_reject_constant)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1"
        or payload.get("policies") != expected
        or payload.get("objective") != expected_objective
    ):
        raise ConfigurationError("report.html climadc-report-data differs from replay-metrics.json")


def _suite_report_payload(
    path: Path,
    metrics_payload: dict[str, object],
    pareto_payload: dict[str, object],
    suite_type: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    if "<script" in lowered or "<link" in lowered or " src=" in lowered:
        raise ConfigurationError("suite report.html contains an external or executable asset")
    match = _REPORT_DATA.search(text)
    if match is None:
        raise ConfigurationError("suite report.html is missing climadc-report-data")
    payload = json.loads(unescape(match.group("payload")), parse_constant=_reject_constant)
    expected = {
        "schema_version": "1",
        "suite_type": suite_type,
        "records": metrics_payload.get("records"),
        "pareto_policies": pareto_payload.get("policies"),
    }
    if payload != expected:
        raise ConfigurationError("suite report.html machine data differs from suite JSON")


def _solver_status(payload: dict[str, object], metric_records: list[dict[str, object]]) -> None:
    policies = _records(payload, "policies", "solver-status.json")
    names = [str(record.get("policy")) for record in policies]
    if len(names) != len(set(names)):
        raise ConfigurationError("solver-status.json contains duplicate policies")
    metric_names = {str(record.get("policy")) for record in metric_records}
    feasible_names: set[str] = set()
    for record in policies:
        if not isinstance(record.get("feasible"), bool):
            raise ConfigurationError("solver-status.json feasible must be boolean")
        status = record.get("solver_status")
        if not isinstance(status, int) or isinstance(status, bool):
            raise ConfigurationError("solver-status.json solver_status must be an integer")
        if bool(record["feasible"]):
            feasible_names.add(str(record["policy"]))
    if feasible_names != metric_names:
        raise ConfigurationError(
            "solver-status.json feasible policy set differs from replay-metrics.json"
        )


def _verify_replay_content(
    root: Path,
    collector: _Collector,
    *,
    manifest: RunManifest | None,
    artifact_version: Literal["1", "2"],
) -> None:
    assumptions_value = collector.run(
        "replay.config",
        "assumptions and config hash are internally consistent",
        ("assumptions.yaml", _RUN_MANIFEST) if manifest is not None else ("assumptions.yaml",),
        lambda: _load_yaml(root / "assumptions.yaml"),
    )
    assumptions = cast(dict[str, object] | None, assumptions_value)
    if assumptions is not None and manifest is not None:
        actual_config_hash = _canonical_hash(
            {key: value for key, value in assumptions.items() if key != "input_hashes"}
        )
        if actual_config_hash != manifest.config_sha256:
            collector.record(
                "replay.config_hash",
                "fail",
                "run-manifest.json config_sha256 does not match assumptions.yaml",
                "run-manifest.json",
                "assumptions.yaml",
            )
        else:
            collector.record(
                "replay.config_hash",
                "pass",
                "config_sha256 reconstructs from portable assumptions",
                "run-manifest.json",
                "assumptions.yaml",
            )
    collector.run(
        "replay.source_manifest",
        "source manifest binds every canonical input",
        ("source-manifest.yaml",),
        lambda: _source_manifest(root, manifest),
    )
    frames_value = collector.run(
        "replay.canonical_frames",
        "Parquet schemas, units, types, finite values, and UTC semantics are valid",
        (
            "climate-forecast.parquet",
            "actual-weather.parquet",
            "grid-signals.parquet",
            "workload.parquet",
            "schedules.parquet",
            "profiles.parquet",
        ),
        lambda: _read_replay_frames(root),
    )
    metrics_value = collector.run(
        "replay.metrics_json",
        "replay-metrics.json is finite, typed JSON with policy records",
        ("replay-metrics.json",),
        lambda: _replay_metrics_payload(
            root / "replay-metrics.json", legacy=artifact_version == "1"
        ),
    )
    status_value = collector.run(
        "replay.solver_json",
        "solver-status.json is finite and typed",
        ("solver-status.json",),
        lambda: _load_json(root / "solver-status.json"),
    )
    metrics_payload = cast(dict[str, object] | None, metrics_value)
    status_payload = cast(dict[str, object] | None, status_value)
    metric_records: list[dict[str, object]] | None = None
    if metrics_payload is not None:
        try:
            metric_records = _records(metrics_payload, "policies", "replay-metrics.json")
        except ConfigurationError as exc:
            collector.record("replay.metric_records", "fail", str(exc), "replay-metrics.json")
    if assumptions is not None and metrics_payload is not None and artifact_version == "2":
        collector.run(
            "replay.objective_contract",
            "objective contract agrees across assumptions, JSON, and HTML inputs",
            ("assumptions.yaml", "replay-metrics.json"),
            lambda: _objective_consistency(assumptions, metrics_payload),
        )
    if status_payload is not None and metric_records is not None:
        collector.run(
            "replay.solver_status",
            "solver status agrees with the settled policy set",
            ("solver-status.json", "replay-metrics.json"),
            lambda: _solver_status(status_payload, metric_records),
        )
    frames = cast(dict[str, pd.DataFrame] | None, frames_value)
    if (
        assumptions is not None
        and frames is not None
        and metric_records is not None
        and status_payload is not None
    ):
        collector.run(
            "replay.reconstruction",
            (
                "schedules/profiles conserve energy and reconstruct facility, cooling, cost, "
                "estimated location-based emissions, peak, and shifted energy"
            ),
            (
                "assumptions.yaml",
                "workload.parquet",
                "schedules.parquet",
                "profiles.parquet",
                "replay-metrics.json",
            ),
            lambda: _profile_and_metric_invariants(
                assumptions=assumptions,
                metric_records=metric_records,
                schedules=frames["schedules.parquet"],
                profiles=frames["profiles.parquet"],
                workload=frames["workload.parquet"],
                solver_status=status_payload,
            ),
        )
    if artifact_version == "2" and metric_records is not None and metrics_payload is not None:
        collector.run(
            "replay.html_consistency",
            "HTML machine payload matches replay-metrics.json",
            ("report.html", "replay-metrics.json"),
            lambda: _report_payload(
                root / "report.html",
                metric_records,
                metrics_payload.get("objective"),
            ),
        )
    elif artifact_version == "1":
        collector.record(
            "replay.html_consistency",
            "warning",
            "legacy v1 has no independent HTML numeric payload",
            "report.html",
        )


def _verify_benchmark_content(root: Path, collector: _Collector) -> None:
    def validate() -> None:
        run = _load_yaml(root / "run.yaml")
        lineage = _load_json(root / "lineage.json")
        metrics = _load_json(root / "metrics.json")
        leakage = _load_json(root / "leakage-report.json")
        for field in ("run_id", "study_id", "climadc_version", "started_at", "config_sha256"):
            if run.get(field) != lineage.get(field):
                raise ConfigurationError(f"run.yaml and lineage.json disagree on {field}")
        for payload, name in ((metrics, "metrics.json"), (leakage, "leakage-report.json")):
            _require_finite(payload, name)
        for name in ("splits.parquet", "predictions.parquet"):
            frame = pd.read_parquet(root / name)
            if frame.empty:
                raise ConfigurationError(f"{name} must not be empty")
            _numeric_frame(frame, name)

    collector.run(
        "benchmark.content",
        "benchmark lineage, JSON, and Parquet content are internally valid",
        tuple(sorted(_BENCHMARK_V1_FILES)),
        validate,
    )


def _resolve_directory(path: Path) -> Path:
    return resolve_run_path(Path(path))


def _report(
    collector: _Collector,
    *,
    run_type: Literal["benchmark", "replay", "replay_suite", "unknown"],
    artifact_version: Literal["1", "2", "unknown"],
    legacy: bool,
    limitations: list[str],
) -> VerificationReport:
    return VerificationReport(
        valid=not any(check.status == "fail" for check in collector.checks),
        run_type=run_type,
        artifact_schema_version=artifact_version,
        legacy=legacy,
        checks=collector.checks,
        limitations=limitations,
    )


def verify_run(directory: Path) -> VerificationReport:
    """Verify a benchmark/replay directory using only its published files."""

    collector = _Collector()
    try:
        root = _resolve_directory(directory)
    except ClimaDCError as exc:
        collector.record("directory", "fail", str(exc))
        return _report(
            collector,
            run_type="unknown",
            artifact_version="unknown",
            legacy=False,
            limitations=[],
        )
    collector.record("directory", "pass", "run directory resolves to a real directory")
    manifest: RunManifest | None = None
    if (root / _RUN_MANIFEST).is_file():
        manifest_value = collector.run(
            "manifest.schema",
            "run-manifest.json conforms to artifact schema v2",
            (_RUN_MANIFEST,),
            lambda: _manifest(root / _RUN_MANIFEST),
        )
        manifest = cast(RunManifest | None, manifest_value)
        if manifest is None:
            return _report(
                collector,
                run_type="unknown",
                artifact_version="unknown",
                legacy=False,
                limitations=[],
            )
        if manifest.run_type == "replay_suite":
            collector.record(
                "manifest.run_type",
                "fail",
                "verify-run does not accept replay_suite; use verify-suite",
                _RUN_MANIFEST,
            )
            return _report(
                collector,
                run_type="replay_suite",
                artifact_version="2",
                legacy=False,
                limitations=[],
            )
        collector.run(
            "checksums",
            "checksums cover every artifact except checksums.sha256 itself",
            (CHECKSUM_FILE,),
            lambda: verify_checksums(root),
        )
        collector.run(
            "manifest.artifacts",
            "manifest declares the exact recursive artifact set",
            (_RUN_MANIFEST,),
            lambda: _validate_manifest_files(root, manifest),
        )
        environment_value = collector.run(
            "environment.schema",
            "environment.json records platform, dependencies, solver, timezone, and seeds",
            (_ENVIRONMENT,),
            lambda: _environment(root / _ENVIRONMENT),
        )
        if environment_value is not None:
            collector.run(
                "environment.consistency",
                "environment and run manifest versions agree",
                (_ENVIRONMENT, _RUN_MANIFEST),
                lambda: _validate_environment_consistency(
                    cast(EnvironmentRecord, environment_value), manifest
                ),
            )
        expected = _REPLAY_V2_FILES if manifest.run_type == "replay" else _BENCHMARK_V2_FILES
        collector.run(
            "artifact_set",
            "top-level artifact contract is exact",
            tuple(sorted(expected)),
            lambda: _exact_top_level(root, expected, manifest.run_type),
        )
        if manifest.run_type == "replay":
            _verify_replay_content(root, collector, manifest=manifest, artifact_version="2")
        else:
            _verify_benchmark_content(root, collector)
        return _report(
            collector,
            run_type=manifest.run_type,
            artifact_version="2",
            legacy=False,
            limitations=[],
        )

    names = {path.name for path in root.iterdir()}
    if names == _REPLAY_V1_FILES:
        collector.record(
            "legacy.contract",
            "warning",
            "legacy replay artifact schema v1 detected; verification is read-only and partial",
        )
        _verify_replay_content(root, collector, manifest=None, artifact_version="1")
        return _report(
            collector,
            run_type="replay",
            artifact_version="1",
            legacy=True,
            limitations=[
                "Legacy v1 has no directory checksum contract.",
                "Legacy v1 has no environment, Git-state, or solver-version record.",
                "Legacy v1 config hashes and HTML cannot be reconstructed independently.",
            ],
        )
    if names == _BENCHMARK_V1_FILES:
        collector.record(
            "legacy.contract",
            "warning",
            "legacy benchmark artifact schema v1 detected; verification is read-only and partial",
        )
        _verify_benchmark_content(root, collector)
        return _report(
            collector,
            run_type="benchmark",
            artifact_version="1",
            legacy=True,
            limitations=[
                "Legacy v1 has no directory checksum or environment contract.",
                "Legacy benchmark report metrics are not independently reconstructible.",
            ],
        )
    collector.record(
        "artifact_set",
        "fail",
        f"unrecognized run artifact set: {sorted(names)}",
    )
    return _report(
        collector,
        run_type="unknown",
        artifact_version="unknown",
        legacy=False,
        limitations=[],
    )


def _safe_scenario_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ConfigurationError("scenario-index.json relative_run_path must be a string")
    relative = safe_relative_path(value)
    if not relative.startswith("scenarios/"):
        raise ConfigurationError("scenario run path must stay below scenarios/")
    path = root / PurePosixPath(relative)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigurationError("scenario run path escapes the suite directory") from exc
    return path


def _suite_scenarios(root: Path, index: dict[str, object]) -> None:
    records = _records(index, "scenarios", "scenario-index.json")
    if not records:
        raise ConfigurationError("scenario-index.json must contain scenario subruns")
    ids: list[str] = []
    for record in records:
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ConfigurationError("scenario-index.json scenario_id must be non-empty")
        ids.append(scenario_id)
        run_path = _safe_scenario_path(root, record.get("relative_run_path"))
        report = verify_run(run_path)
        if not report.valid:
            failures = [check.message for check in report.checks if check.status == "fail"]
            raise ConfigurationError(
                f"scenario {scenario_id} failed verify-run: {'; '.join(failures)}"
            )
        manifest = _manifest(run_path / _RUN_MANIFEST)
        if manifest.config_sha256 != record.get("config_sha256"):
            raise ConfigurationError(
                f"scenario {scenario_id} config_sha256 differs from its run manifest"
            )
        lineage = _load_json(run_path / "lineage.json")
        if lineage.get("input_hashes") != record.get("input_hashes"):
            raise ConfigurationError(
                f"scenario {scenario_id} input_hashes differ from its lineage record"
            )
        if manifest.study_id != record.get("study_id"):
            raise ConfigurationError(
                f"scenario {scenario_id} study_id differs from its run manifest"
            )
        assumptions = _load_yaml(run_path / "assumptions.yaml")
        replay = assumptions.get("replay")
        if not isinstance(replay, dict):
            raise ConfigurationError(f"scenario {scenario_id} replay assumptions are invalid")
        if assumptions.get("decision_time") != record.get("decision_time"):
            raise ConfigurationError(
                f"scenario {scenario_id} decision_time differs from assumptions.yaml"
            )
        if replay.get("site_id") != record.get("site_id"):
            raise ConfigurationError(
                f"scenario {scenario_id} site_id differs from assumptions.yaml"
            )
    if len(ids) != len(set(ids)):
        raise ConfigurationError("scenario-index.json contains duplicate scenario_id values")
    actual_ids = {path.name for path in (root / "scenarios").iterdir() if path.is_dir()}
    if actual_ids != set(ids):
        raise ConfigurationError(
            f"scenario directories differ from index: expected={sorted(ids)}, actual={sorted(actual_ids)}"
        )


def _strict_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{label} must be an integer")
    return value


def _suite_metric_rows(
    root: Path,
    frame: pd.DataFrame,
    scenario_records: list[dict[str, object]],
) -> tuple[str, ...]:
    if tuple(str(column) for column in frame.columns[: len(_SUITE_PREFIX_COLUMNS)]) != (
        _SUITE_PREFIX_COLUMNS
    ):
        raise ConfigurationError("scenario-metrics.parquet prefix schema is invalid")
    metric_columns = tuple(
        str(column) for column in frame.columns if str(column) not in _SUITE_PREFIX_COLUMNS
    )
    if not metric_columns:
        raise ConfigurationError("scenario-metrics.parquet has no replay metric columns")
    if (
        not isinstance(frame["decision_time"].dtype, pd.DatetimeTZDtype)
        or str(frame["decision_time"].dtype.tz) != "UTC"
    ):
        raise ConfigurationError("scenario-metrics.parquet decision_time must use exact UTC dtype")
    if not pd.api.types.is_bool_dtype(frame["feasible"].dtype):
        raise ConfigurationError("scenario-metrics.parquet feasible must be boolean")

    expected_keys: set[tuple[str, str]] = set()
    expected_policy_order: tuple[str, ...] | None = None
    for scenario in scenario_records:
        scenario_id = str(scenario["scenario_id"])
        run_path = _safe_scenario_path(root, scenario.get("relative_run_path"))
        metrics_payload = _load_json(run_path / "replay-metrics.json")
        status_payload = _load_json(run_path / "solver-status.json")
        metric_records = _records(metrics_payload, "policies", "replay-metrics.json")
        status_records = _records(status_payload, "policies", "solver-status.json")
        metric_by_policy: dict[str, dict[str, object]] = {}
        for metric_record in metric_records:
            policy = metric_record.get("policy")
            if not isinstance(policy, str) or not policy:
                raise ConfigurationError(f"scenario {scenario_id} metric policy must be text")
            if set(metric_record) != {"policy", *metric_columns}:
                raise ConfigurationError(
                    f"scenario {scenario_id} replay metric schema differs from suite Parquet"
                )
            if policy in metric_by_policy:
                raise ConfigurationError(f"scenario {scenario_id} has duplicate metric policies")
            metric_by_policy[policy] = metric_record

        policy_order = tuple(str(record.get("policy")) for record in status_records)
        if len(policy_order) != len(set(policy_order)) or any(
            policy == "None" for policy in policy_order
        ):
            raise ConfigurationError(f"scenario {scenario_id} has invalid solver policy records")
        if expected_policy_order is None:
            expected_policy_order = policy_order
        elif expected_policy_order != policy_order:
            raise ConfigurationError("scenario subruns expose different ordered policy sets")
        feasible_values: list[bool] = []
        for status in status_records:
            policy_value = status.get("policy")
            feasible = status.get("feasible")
            solver_status = status.get("solver_status")
            message = status.get("message")
            if not isinstance(policy_value, str) or not policy_value:
                raise ConfigurationError(f"scenario {scenario_id} solver policy must be text")
            if not isinstance(feasible, bool):
                raise ConfigurationError(f"scenario {scenario_id} feasible must be boolean")
            _strict_integer(solver_status, f"scenario {scenario_id} solver_status")
            if not isinstance(message, str):
                raise ConfigurationError(f"scenario {scenario_id} solver message must be text")
            policy_metric = metric_by_policy.get(policy_value)
            if feasible != (policy_metric is not None):
                raise ConfigurationError(
                    f"scenario {scenario_id} feasible status differs from replay metrics"
                )
            key = (scenario_id, policy_value)
            if key in expected_keys:
                raise ConfigurationError("scenario subruns contain duplicate scenario/policy keys")
            expected_keys.add(key)
            actual_rows = frame.loc[
                (frame["scenario_id"].astype(str) == scenario_id)
                & (frame["policy"].astype(str) == policy_value)
            ]
            if len(actual_rows) != 1:
                raise ConfigurationError(
                    f"scenario-metrics.parquet misses unique row {scenario_id}/{policy_value}"
                )
            actual = actual_rows.iloc[0]
            expected_prefix: dict[str, object] = {
                "description": scenario.get("description"),
                "study_id": scenario.get("study_id"),
                "decision_time": pd.Timestamp(cast(Any, scenario.get("decision_time"))),
                "site_id": scenario.get("site_id"),
                "mode": metrics_payload.get("mode"),
                "currency": metrics_payload.get("currency"),
                "feasible": feasible,
                "solver_status": solver_status,
                "message": message,
            }
            for field, expected in expected_prefix.items():
                actual_value = actual[field]
                if field == "decision_time":
                    if cast(pd.Timestamp, actual_value) != expected:
                        raise ConfigurationError(
                            f"scenario-metrics.parquet {scenario_id}/{policy_value}.{field} mismatch"
                        )
                elif actual_value != expected:
                    raise ConfigurationError(
                        f"scenario-metrics.parquet {scenario_id}/{policy_value}.{field} mismatch"
                    )
            for field in metric_columns:
                actual_value = actual[field]
                if policy_metric is None:
                    if not pd.isna(actual_value):
                        raise ConfigurationError(
                            f"scenario-metrics.parquet infeasible {scenario_id}/{policy_value}."
                            f"{field} must be null"
                        )
                    continue
                expected = policy_metric[field]
                if isinstance(expected, Real) and not isinstance(expected, bool):
                    _close(
                        actual_value,
                        float(expected),
                        f"scenario-metrics.parquet {scenario_id}/{policy_value}.{field}",
                    )
                elif actual_value != expected:
                    raise ConfigurationError(
                        f"scenario-metrics.parquet {scenario_id}/{policy_value}.{field} mismatch"
                    )
            feasible_values.append(feasible)
        declared_feasible = scenario.get("feasible")
        if not isinstance(declared_feasible, bool) or declared_feasible != all(feasible_values):
            raise ConfigurationError(
                f"scenario-index.json {scenario_id}.feasible differs from solver status"
            )
    actual_keys = set(
        zip(frame["scenario_id"].astype(str), frame["policy"].astype(str), strict=True)
    )
    if actual_keys != expected_keys:
        raise ConfigurationError("scenario-metrics.parquet keys differ from scenario subruns")
    return expected_policy_order or ()


def _suite_type_contract(
    suite: dict[str, object], scenario_records: list[dict[str, object]]
) -> Literal["sensitivity", "robustness"]:
    suite_type = suite.get("suite_type")
    if suite_type not in {"sensitivity", "robustness"}:
        raise ConfigurationError("suite.yaml suite_type must be sensitivity or robustness")
    raw_dimensions = suite.get("robustness_dimensions", [])
    allowed = {"decision_date", "season", "location", "workload"}
    if not isinstance(raw_dimensions, list) or any(
        not isinstance(value, str) or value not in allowed for value in raw_dimensions
    ):
        raise ConfigurationError("suite.yaml robustness_dimensions is invalid")
    dimensions = cast(list[str], raw_dimensions)
    if len(dimensions) != len(set(dimensions)):
        raise ConfigurationError("suite.yaml robustness_dimensions must be unique")
    if suite_type == "sensitivity":
        if dimensions:
            raise ConfigurationError("sensitivity suite must not declare robustness_dimensions")
        return "sensitivity"
    if not dimensions:
        raise ConfigurationError("robustness suite must declare an independent sample dimension")

    dates: set[object] = set()
    seasons: set[int] = set()
    locations: set[str] = set()
    workload_hashes: set[str] = set()
    for record in scenario_records:
        timestamp = pd.Timestamp(cast(Any, record.get("decision_time")))
        if timestamp.tzinfo is None or str(timestamp.tzinfo) != "UTC":
            raise ConfigurationError("scenario-index.json decision_time must use exact UTC")
        dates.add(timestamp.date())
        seasons.add((timestamp.month % 12) // 3)
        locations.add(str(record.get("site_id")))
        hashes = record.get("input_hashes")
        if not isinstance(hashes, dict):
            raise ConfigurationError("scenario-index.json input_hashes must be an object")
        candidates = {
            str(value) for key, value in hashes.items() if "workload" in Path(str(key)).name.lower()
        }
        if len(candidates) != 1:
            raise ConfigurationError(
                "scenario-index.json must identify one workload input hash per scenario"
            )
        workload_hashes.update(candidates)
    varied = {
        dimension
        for dimension, values in (
            ("decision_date", dates),
            ("season", seasons),
            ("location", locations),
            ("workload", workload_hashes),
        )
        if len(values) > 1
    }
    if not set(dimensions).intersection(varied):
        raise ConfigurationError(
            "robustness suite does not vary any declared independent sample dimension"
        )
    return "robustness"


def _suite_dominates(candidate: np.ndarray, target: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(candidate))), float(np.max(np.abs(target))))
    tolerance = _SUITE_IMPROVEMENT_TOLERANCE * scale
    return bool(np.all(candidate <= target + tolerance) and np.any(candidate < target - tolerance))


def _suite_aggregate(root: Path, index: dict[str, object], suite: dict[str, object]) -> None:
    frame = pd.read_parquet(root / "scenario-metrics.parquet")
    _numeric_frame(frame, "scenario-metrics.parquet")
    if frame.empty or frame.duplicated(["scenario_id", "policy"]).any():
        raise ConfigurationError(
            "scenario-metrics.parquet must contain unique non-empty scenario/policy rows"
        )
    scenario_records = _records(index, "scenarios", "scenario-index.json")
    expected_ids = {str(record["scenario_id"]) for record in scenario_records}
    if set(frame["scenario_id"].astype(str)) != expected_ids:
        raise ConfigurationError("scenario-metrics.parquet scenario set differs from index")
    policies = _suite_metric_rows(root, frame, scenario_records)
    scenario_count = len(expected_ids)
    suite_type = _suite_type_contract(suite, scenario_records)

    payload = _load_json(root / "suite-metrics.json")
    if (
        payload.get("schema_version") != "1"
        or payload.get("suite_id") != suite.get("suite_id")
        or payload.get("suite_type") != suite_type
        or payload.get("aggregation") != "equal_weight"
    ):
        raise ConfigurationError("suite-metrics.json metadata differs from suite.yaml")
    aggregate = _records(payload, "records", "suite-metrics.json")
    by_policy = {str(record.get("policy")): record for record in aggregate}
    if len(by_policy) != len(aggregate) or set(by_policy) != set(policies):
        raise ConfigurationError("suite-metrics.json policy set differs from scenario subruns")

    expected: dict[str, dict[str, object]] = {}
    for policy in policies:
        policy_rows = frame.loc[frame["policy"].astype(str) == policy]
        rows = policy_rows.loc[policy_rows["feasible"]].reset_index(drop=True)
        feasible_count = len(rows)
        record: dict[str, object] = {
            "policy": policy,
            "scenario_count": scenario_count,
            "feasible_scenarios": feasible_count,
            "feasible_fraction": feasible_count / scenario_count,
            "pareto_efficient": False,
        }
        for source, improvement_field, mean_field, worst_field, scenario_field in _SUITE_SUMMARIES:
            if source not in rows:
                raise ConfigurationError(f"scenario-metrics.parquet is missing {source}")
            if rows.empty:
                record[improvement_field] = None
                record[mean_field] = None
                record[worst_field] = None
                record[scenario_field] = None
            else:
                values = rows[source].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ConfigurationError(f"feasible suite rows contain invalid {source}")
                worst_position = int(np.argmax(values))
                record[improvement_field] = float(np.mean(values < -_SUITE_IMPROVEMENT_TOLERANCE))
                record[mean_field] = float(np.mean(values))
                record[worst_field] = float(values[worst_position])
                record[scenario_field] = str(rows.iloc[worst_position]["scenario_id"])
        expected[policy] = record

    eligible = [
        policy for policy in policies if expected[policy]["feasible_scenarios"] == scenario_count
    ]
    frontier: list[str] = []
    for policy in eligible:
        target = np.array([expected[policy][field] for field in _SUITE_PARETO_FIELDS], dtype=float)
        if not any(
            other != policy
            and _suite_dominates(
                np.array([expected[other][field] for field in _SUITE_PARETO_FIELDS], dtype=float),
                target,
            )
            for other in eligible
        ):
            expected[policy]["pareto_efficient"] = True
            frontier.append(policy)

    for policy, expected_record in expected.items():
        actual = by_policy[policy]
        if set(actual) != set(expected_record):
            raise ConfigurationError(f"suite-metrics.json {policy} record schema mismatch")
        for field, expected_value in expected_record.items():
            actual_value = actual[field]
            if expected_value is None:
                if actual_value is not None:
                    raise ConfigurationError(f"suite-metrics.json {policy}.{field} must be null")
            elif isinstance(expected_value, bool):
                if not isinstance(actual_value, bool) or actual_value != expected_value:
                    raise ConfigurationError(f"suite-metrics.json {policy}.{field} mismatch")
            elif isinstance(expected_value, Real):
                _close(
                    actual_value,
                    float(expected_value),
                    f"suite-metrics.json {policy}.{field}",
                )
            elif actual_value != expected_value:
                raise ConfigurationError(f"suite-metrics.json {policy}.{field} mismatch")

    pareto = _load_json(root / "pareto-frontier.json")
    if (
        pareto.get("schema_version") != "1"
        or pareto.get("suite_id") != suite.get("suite_id")
        or pareto.get("policies") != frontier
    ):
        raise ConfigurationError("pareto-frontier.json differs from reconstructed frontier")
    report = (root / "report.html").read_text(encoding="utf-8")
    if suite_type == "sensitivity" and "robustness" in report.lower():
        raise ConfigurationError("sensitivity report.html must not claim robustness")
    _suite_report_payload(root / "report.html", payload, pareto, suite_type)


def verify_suite(directory: Path) -> VerificationReport:
    """Recursively verify a suite and reconstruct its aggregate metrics without solving."""

    collector = _Collector()
    try:
        root = _resolve_directory(directory)
    except ClimaDCError as exc:
        collector.record("directory", "fail", str(exc))
        return _report(
            collector,
            run_type="unknown",
            artifact_version="unknown",
            legacy=False,
            limitations=[],
        )
    collector.record("directory", "pass", "suite directory resolves to a real directory")
    if not (root / _RUN_MANIFEST).is_file():
        names = {path.name for path in root.iterdir()}
        if names == _SUITE_V1_TOP_LEVEL:
            collector.record(
                "legacy.contract",
                "warning",
                "legacy replay-suite schema v1 detected; recursive checksum verification unavailable",
            )
            index_value = collector.run(
                "suite.index",
                "legacy scenario index is parseable",
                ("scenario-index.json",),
                lambda: _load_json(root / "scenario-index.json"),
            )
            if index_value is not None:
                records = _records(
                    cast(dict[str, object], index_value), "scenarios", "scenario-index.json"
                )
                for record in records:
                    run_path = _safe_scenario_path(root, record.get("relative_run_path"))
                    report = verify_run(run_path)
                    if not report.valid:
                        collector.record(
                            "suite.legacy_subruns",
                            "fail",
                            f"legacy scenario {record.get('scenario_id')} failed verify-run",
                        )
                        break
                else:
                    collector.record(
                        "suite.legacy_subruns",
                        "pass",
                        "all legacy scenario subruns passed partial v1 verification",
                    )
            return _report(
                collector,
                run_type="replay_suite",
                artifact_version="1",
                legacy=True,
                limitations=[
                    "Legacy suite v1 has no suite-wide checksums or environment record.",
                    "Legacy v1 aggregate HTML and robustness metrics are only partially checked.",
                ],
            )
        collector.record(
            "artifact_set", "fail", f"unrecognized suite artifact set: {sorted(names)}"
        )
        return _report(
            collector,
            run_type="unknown",
            artifact_version="unknown",
            legacy=False,
            limitations=[],
        )
    manifest_value = collector.run(
        "manifest.schema",
        "run-manifest.json conforms to suite artifact schema v2",
        (_RUN_MANIFEST,),
        lambda: _manifest(root / _RUN_MANIFEST),
    )
    manifest = cast(RunManifest | None, manifest_value)
    if manifest is None:
        return _report(
            collector,
            run_type="unknown",
            artifact_version="unknown",
            legacy=False,
            limitations=[],
        )
    if manifest.run_type != "replay_suite":
        collector.record(
            "manifest.run_type",
            "fail",
            f"verify-suite requires replay_suite, found {manifest.run_type}",
            _RUN_MANIFEST,
        )
    collector.run(
        "artifact_set",
        "suite top-level artifact contract is exact",
        tuple(sorted(_SUITE_V2_TOP_LEVEL)),
        lambda: _exact_top_level(root, _SUITE_V2_TOP_LEVEL, "replay_suite"),
    )
    collector.run(
        "checksums",
        "suite checksums cover every aggregate and scenario artifact",
        (CHECKSUM_FILE,),
        lambda: verify_checksums(root),
    )
    collector.run(
        "manifest.artifacts",
        "suite manifest declares the exact recursive artifact set",
        (_RUN_MANIFEST,),
        lambda: _validate_manifest_files(root, manifest),
    )
    environment_value = collector.run(
        "environment.schema",
        "suite environment record is valid",
        (_ENVIRONMENT,),
        lambda: _environment(root / _ENVIRONMENT),
    )
    if environment_value is not None:
        collector.run(
            "environment.consistency",
            "suite environment and run manifest agree",
            (_ENVIRONMENT, _RUN_MANIFEST),
            lambda: _validate_environment_consistency(
                cast(EnvironmentRecord, environment_value), manifest
            ),
        )
    suite_value = collector.run(
        "suite.config",
        "suite.yaml is finite and config-hash bound",
        ("suite.yaml", _RUN_MANIFEST),
        lambda: _load_yaml(root / "suite.yaml"),
    )
    suite = cast(dict[str, object] | None, suite_value)
    if suite is not None:
        actual_hash = _canonical_hash(suite)
        if actual_hash != manifest.config_sha256:
            collector.record(
                "suite.config_hash",
                "fail",
                "run-manifest.json config_sha256 does not match suite.yaml",
                _RUN_MANIFEST,
                "suite.yaml",
            )
        else:
            collector.record(
                "suite.config_hash",
                "pass",
                "suite config hash reconstructs from suite.yaml",
                _RUN_MANIFEST,
                "suite.yaml",
            )
    index_value = collector.run(
        "suite.index",
        "scenario-index.json is finite and parseable",
        ("scenario-index.json",),
        lambda: _load_json(root / "scenario-index.json"),
    )
    index = cast(dict[str, object] | None, index_value)
    if index is not None:
        collector.run(
            "suite.subruns",
            "all scenario subruns pass independent verify-run",
            ("scenario-index.json", "scenarios"),
            lambda: _suite_scenarios(root, index),
        )
    if index is not None and suite is not None:
        collector.run(
            "suite.aggregate",
            "aggregate metrics reconstruct from scenario subruns",
            (
                "scenario-index.json",
                "scenario-metrics.parquet",
                "suite-metrics.json",
                "pareto-frontier.json",
                "report.html",
            ),
            lambda: _suite_aggregate(root, index, suite),
        )
    return _report(
        collector,
        run_type="replay_suite",
        artifact_version="2",
        legacy=False,
        limitations=[],
    )


__all__ = ["VerificationCheck", "VerificationReport", "verify_run", "verify_suite"]
