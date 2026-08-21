from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import timezone
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from climadc import __version__
from climadc.evidence.manifest import SolverRecord
from climadc.evidence.writer import finalize_evidence
from climadc.errors import ConfigurationError
from climadc.replay.html import render_replay_report
from climadc.replay.manifest import SourceManifest, sha256_file
from climadc.replay.models import pareto_policy_name
from climadc.replay.rolling import RollingReplayResult
from climadc.replay.study import ReplayStudyResult, assumptions_payload
from climadc.reporting.artifacts import update_latest_pointer

REPLAY_ARTIFACTS = frozenset(
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
        "run-manifest.json",
        "environment.json",
        "checksums.sha256",
    }
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if hasattr(value, "item"):
        return cast(object, cast(Any, value).item())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_text(payload: object) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )


def replay_metrics_payload(result: ReplayStudyResult) -> dict[str, object]:
    if isinstance(result.replay, RollingReplayResult):
        mode = "rolling"
        decision_count = result.replay.decision_count
        commit_interval: str | None = str(result.replay.commit_interval)
    else:
        mode = "single_window"
        decision_count = 1
        commit_interval = None
    policies = result.replay.metrics.to_dict(orient="records")
    pareto_points: list[dict[str, object]] = []
    objective = result.config.replay.objective
    if objective is not None and objective.mode == "pareto_analysis":
        candidates = [row for row in policies if str(row["policy"]).startswith("pareto_cp_")]
        for row in candidates:
            policy = str(row["policy"])
            price = next(
                value
                for value in objective.carbon_prices_currency_per_tco2e
                if policy == pareto_policy_name(value)
            )
            cost = float(row["energy_cost"])
            emissions = float(row["estimated_location_based_emissions_kgco2e"])
            efficient = not any(
                float(other["energy_cost"]) <= cost
                and float(other["estimated_location_based_emissions_kgco2e"]) <= emissions
                and (
                    float(other["energy_cost"]) < cost
                    or float(other["estimated_location_based_emissions_kgco2e"]) < emissions
                )
                for other in candidates
            )
            pareto_points.append(
                {
                    "policy": policy,
                    "carbon_price_currency_per_tco2e": price,
                    "energy_cost": cost,
                    "estimated_location_based_emissions_kgco2e": emissions,
                    "peak_kw": float(row["peak_kw"]),
                    "pareto_efficient": efficient,
                }
            )
    return {
        "schema_version": "1",
        "study_id": result.config.study_id,
        "mode": mode,
        "decision_count": decision_count,
        "commit_interval": commit_interval,
        "currency": result.replay.currency,
        "units": {
            "facility_energy_kwh": "kWh",
            "it_energy_kwh": "kWh",
            "cooling_energy_kwh": "kWh",
            "estimated_location_based_emissions_kgco2e": "kgCO2e",
            "energy_cost": result.replay.currency,
            "peak_kw": "kW",
            "objective_regret": result.replay.currency,
        },
        "objective": result.config.replay.objective_payload(),
        "accepted_jobs": result.replay.accepted_jobs,
        "future_jobs": result.replay.future_jobs,
        "forecast": result.forecast_metrics,
        "policies": policies,
        "pareto_frontier": pareto_points,
        "violations": {policy: list(items) for policy, items in result.replay.violations.items()},
    }


def _solver_status_payload(result: ReplayStudyResult) -> dict[str, object]:
    if isinstance(result.replay, RollingReplayResult):
        mode = "rolling"
        decisions = result.replay.decisions.to_dict(orient="records")
        remaining_energy = result.replay.remaining_energy.to_dict(orient="records")
    else:
        mode = "single_window"
        decisions = []
        remaining_energy = []
    return {
        "schema_version": "1",
        "mode": mode,
        "feasible": bool(result.replay.status["feasible"].all()),
        "policies": result.replay.status.to_dict(orient="records"),
        "decisions": decisions,
        "remaining_energy": remaining_energy,
    }


def _normalize_object_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    for column in normalized.select_dtypes(include=["object"]).columns:
        non_null = [value for value in normalized[column] if not pd.isna(value)]
        if non_null and all(
            isinstance(value, Real) and not isinstance(value, bool) for value in non_null
        ):
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(float)
            continue
        normalized[column] = pd.Series(
            [None if pd.isna(value) else value for value in normalized[column]],
            index=normalized.index,
            dtype=object,
        )
    return cast(pd.DataFrame, normalized)


def _published_source_manifest(result: ReplayStudyResult, directory: Path) -> SourceManifest:
    input_to_output = {
        result.config.inputs.climate_forecast.path.resolve(): "climate-forecast.parquet",
        result.config.inputs.actual_weather.path.resolve(): "actual-weather.parquet",
        result.config.inputs.grid_signals.path.resolve(): "grid-signals.parquet",
        result.config.inputs.workload.path.resolve(): "workload.parquet",
    }
    records = []
    for record in result.manifest.records:
        input_path = (result.config.source_manifest.parent / record.artifact).resolve()
        output_name = input_to_output.get(input_path)
        if output_name is None:
            raise ConfigurationError(
                f"Verified source record does not map to a replay input: {record.artifact}"
            )
        output_path = directory / output_name
        records.append(
            record.model_copy(
                update={
                    "artifact": Path(output_name),
                    "sha256": sha256_file(output_path),
                    "bytes": output_path.stat().st_size,
                    "transformations": [
                        *record.transformations,
                        "Serialized the verified canonical input as Parquet for this run.",
                    ],
                }
            )
        )
    return SourceManifest(
        schema_version=result.manifest.schema_version,
        study_id=result.manifest.study_id,
        records=records,
    )


def _assert_close(actual: float, expected: float, label: str) -> None:
    tolerance = 1e-8 * max(1.0, abs(actual), abs(expected))
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} is not reconstructible from interval profiles")


def _validate_metric_reconstruction(
    metrics: list[dict[str, object]],
    profiles: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    interval_hours: float,
    demand_charge_per_kw: float,
) -> None:
    by_policy = {str(row["policy"]): row for row in metrics}
    if set(by_policy) != set(profiles["policy"].unique()):
        raise ValueError("metric and profile policy sets differ")
    asap_schedule = schedules.loc[schedules["policy"] == "asap"].sort_values(
        ["job_id", "valid_time"]
    )
    for policy, row in by_policy.items():
        profile = profiles.loc[profiles["policy"] == policy].sort_values("valid_time")
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
        demand_charge = peak * demand_charge_per_kw
        for label, reconstructed in (
            ("facility_energy_kwh", facility_energy),
            ("it_energy_kwh", it_energy),
            ("cooling_energy_kwh", cooling_energy),
            ("estimated_location_based_emissions_kgco2e", emissions),
            ("energy_charge", energy_charge),
            ("demand_charge", demand_charge),
            ("energy_cost", energy_charge + demand_charge),
            ("peak_kw", peak),
        ):
            _assert_close(float(cast(Any, row[label])), reconstructed, f"{policy}.{label}")

        policy_schedule = schedules.loc[schedules["policy"] == policy].sort_values(
            ["job_id", "valid_time"]
        )
        if list(
            policy_schedule[["job_id", "valid_time"]].itertuples(index=False, name=None)
        ) != list(asap_schedule[["job_id", "valid_time"]].itertuples(index=False, name=None)):
            raise ValueError("schedule keys differ across policies")
        shifted = 0.5 * float(
            np.sum(
                np.abs(
                    policy_schedule["power_kw"].to_numpy(dtype=float)
                    - asap_schedule["power_kw"].to_numpy(dtype=float)
                )
            )
            * interval_hours
        )
        _assert_close(
            float(cast(Any, row["shifted_energy_kwh"])),
            shifted,
            f"{policy}.shifted_energy_kwh",
        )


class ReplayArtifactWriter:
    def _write(self, directory: Path, result: ReplayStudyResult, run_id: str) -> None:
        assumptions = assumptions_payload(result.config)
        assumptions["input_hashes"] = result.input_hashes
        (directory / "assumptions.yaml").write_text(
            yaml.safe_dump(assumptions, sort_keys=True, allow_unicode=False),
            encoding="utf-8",
            newline="\n",
        )
        lineage = {
            "schema_version": "1",
            "run_id": run_id,
            "study_id": result.config.study_id,
            "climadc_version": __version__,
            "started_at": result.started_at.isoformat(),
            "config_sha256": result.config_sha256,
            "input_hashes": result.input_hashes,
        }
        (directory / "lineage.json").write_text(_json_text(lineage), encoding="utf-8", newline="\n")
        result.climate_forecast.to_pandas().to_parquet(
            directory / "climate-forecast.parquet", index=False
        )
        result.actual_weather.to_pandas().to_parquet(
            directory / "actual-weather.parquet", index=False
        )
        result.grid_signals.to_pandas().to_parquet(directory / "grid-signals.parquet", index=False)
        result.workload.to_pandas().to_parquet(directory / "workload.parquet", index=False)
        result.replay.allocations.to_parquet(directory / "schedules.parquet", index=False)
        result.replay.profiles.to_parquet(directory / "profiles.parquet", index=False)
        published_manifest = _published_source_manifest(result, directory)
        (directory / "source-manifest.yaml").write_text(
            yaml.safe_dump(
                published_manifest.model_dump(mode="json"), sort_keys=True, allow_unicode=False
            ),
            encoding="utf-8",
            newline="\n",
        )
        status = _solver_status_payload(result)
        (directory / "solver-status.json").write_text(
            _json_text(status), encoding="utf-8", newline="\n"
        )
        (directory / "replay-metrics.json").write_text(
            _json_text(replay_metrics_payload(result)), encoding="utf-8", newline="\n"
        )
        (directory / "report.html").write_text(
            render_replay_report(result, run_id), encoding="utf-8", newline="\n"
        )

    @staticmethod
    def _validate(directory: Path, result: ReplayStudyResult, run_id: str) -> None:
        names = {path.name for path in directory.iterdir()}
        if names != REPLAY_ARTIFACTS:
            raise ConfigurationError(
                "Invalid replay artifact set before publish: "
                f"missing={sorted(REPLAY_ARTIFACTS - names)}, "
                f"extra={sorted(names - REPLAY_ARTIFACTS)}"
            )
        empty = sorted(path.name for path in directory.iterdir() if path.stat().st_size == 0)
        if empty:
            raise ConfigurationError(f"Empty replay artifacts before publish: {empty}")
        try:
            assumptions = yaml.safe_load(
                (directory / "assumptions.yaml").read_text(encoding="utf-8")
            )
            expected_assumptions = assumptions_payload(result.config)
            expected_assumptions["input_hashes"] = result.input_hashes
            if assumptions != expected_assumptions:
                raise ValueError("assumptions.yaml differs from frozen replay result")
            published_manifest = SourceManifest.from_yaml(directory / "source-manifest.yaml")
            expected_manifest = _published_source_manifest(result, directory)
            if published_manifest.model_dump(mode="json") != expected_manifest.model_dump(
                mode="json"
            ):
                raise ValueError("source-manifest.yaml differs from published replay inputs")
            published_manifest.validate_files(
                directory,
                {
                    directory / "climate-forecast.parquet",
                    directory / "actual-weather.parquet",
                    directory / "grid-signals.parquet",
                    directory / "workload.parquet",
                },
            )
            lineage = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
            metrics = json.loads((directory / "replay-metrics.json").read_text(encoding="utf-8"))
            status = json.loads((directory / "solver-status.json").read_text(encoding="utf-8"))
            expected_lineage = {
                "schema_version": "1",
                "run_id": run_id,
                "study_id": result.config.study_id,
                "climadc_version": __version__,
                "started_at": result.started_at.isoformat(),
                "config_sha256": result.config_sha256,
                "input_hashes": result.input_hashes,
            }
            if lineage != expected_lineage:
                raise ValueError("lineage.json differs from frozen replay result")
            if metrics != json.loads(_json_text(replay_metrics_payload(result))):
                raise ValueError("replay-metrics.json differs from frozen replay result")
            expected_status = _solver_status_payload(result)
            if status != json.loads(_json_text(expected_status)):
                raise ValueError("solver-status.json differs from frozen replay result")
            frame_pairs = (
                ("climate-forecast.parquet", result.climate_forecast.to_pandas()),
                ("actual-weather.parquet", result.actual_weather.to_pandas()),
                ("grid-signals.parquet", result.grid_signals.to_pandas()),
                ("workload.parquet", result.workload.to_pandas()),
                ("schedules.parquet", result.replay.allocations),
                ("profiles.parquet", result.replay.profiles),
            )
            for name, expected in frame_pairs:
                actual = pd.read_parquet(directory / name)
                pd.testing.assert_frame_equal(
                    _normalize_object_nulls(actual).reset_index(drop=True),
                    _normalize_object_nulls(expected).reset_index(drop=True),
                    check_dtype=True,
                )
                numeric = actual.select_dtypes(include=[np.number]).to_numpy(dtype=float)
                if numeric.size and np.isinf(numeric).any():
                    raise ValueError(f"{name} contains an infinite numeric value")
            replay_config = result.config.replay.to_replay_config()
            _validate_metric_reconstruction(
                cast(list[dict[str, object]], metrics["policies"]),
                pd.read_parquet(directory / "profiles.parquet"),
                pd.read_parquet(directory / "schedules.parquet"),
                interval_hours=float(replay_config.interval / pd.Timedelta("1h")),
                demand_charge_per_kw=replay_config.demand_charge_per_kw,
            )
            report = (directory / "report.html").read_text(encoding="utf-8")
            if report != render_replay_report(result, run_id):
                raise ValueError("report.html differs from frozen replay result")
            lowered = report.lower()
            if "<script" in lowered or "<link" in lowered or " src=" in lowered:
                raise ValueError("report.html contains an external or executable asset")
            if not all(token in lowered for token in ("<html", "<body", "<table", "limitations")):
                raise ValueError("report.html is missing required document sections")
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Invalid replay artifact content: {exc}") from exc

    def write(self, result: ReplayStudyResult, output_dir: Path) -> Path:
        timestamp = result.started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}-{result.config_sha256[:8]}"
        runs_dir = Path(output_dir)
        final = runs_dir / run_id
        temporary: Path | None = None
        published = False
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise ConfigurationError(f"Replay run directory already exists: {final}")
            temporary = Path(tempfile.mkdtemp(prefix=".climadc-replay-", dir=runs_dir))
            self._write(temporary, result, run_id)
            published_manifest = SourceManifest.from_yaml(temporary / "source-manifest.yaml")
            canonical_hashes = {
                str(record.artifact): record.sha256 for record in published_manifest.records
            }
            finalize_evidence(
                temporary,
                run_type="replay",
                run_id=run_id,
                study_id=result.config.study_id,
                started_at=result.started_at,
                config_sha256=result.config_sha256,
                input_hashes=canonical_hashes,
                solver=SolverRecord(
                    name="SciPy HiGHS",
                    method=(
                        "scipy.optimize.linprog(method=highs), then fixed-aggregate "
                        "job-allocation tie-break"
                    ),
                    options={
                        "primal_feasibility_tolerance": 1e-10,
                        "allocation_tie_break": (
                            "ASAP priority/deadline/release/job_id at fixed aggregate slot power"
                        ),
                    },
                ),
            )
            self._validate(temporary, result, run_id)
            from climadc.evidence.verify import verify_run

            verification = verify_run(temporary)
            if not verification.valid:
                failures = [
                    check.message for check in verification.checks if check.status == "fail"
                ]
                raise ConfigurationError(
                    f"Independent replay verification failed: {'; '.join(failures)}"
                )
            temporary.rename(final)
            published = True
            update_latest_pointer(runs_dir, final)
            return _absolute(final)
        except ConfigurationError:
            if published and final.exists():
                shutil.rmtree(final)
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            raise
        except Exception as exc:
            if published and final.exists():
                shutil.rmtree(final)
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            raise ConfigurationError(f"Unable to write replay artifacts: {exc}") from exc
