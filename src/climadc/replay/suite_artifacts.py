from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from climadc import __version__
from climadc.evidence.manifest import RunManifest, SolverRecord
from climadc.evidence.writer import finalize_evidence
from climadc.errors import ConfigurationError
from climadc.replay.artifacts import REPLAY_ARTIFACTS, ReplayArtifactWriter
from climadc.replay.suite import ReplaySuiteResult, suite_assumptions_payload
from climadc.replay.suite_html import render_replay_suite_report
from climadc.reporting.artifacts import update_latest_pointer

REPLAY_SUITE_ARTIFACTS = frozenset(
    {
        "suite.yaml",
        "lineage.json",
        "scenario-index.json",
        "scenario-metrics.parquet",
        "suite-metrics.json",
        "pareto-frontier.json",
        "report.html",
        "scenarios",
        "run-manifest.json",
        "environment.json",
        "checksums.sha256",
    }
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_value(cast(Any, value).item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_text(payload: object) -> str:
    return (
        json.dumps(
            _json_value(payload),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    raw = cast(list[dict[str, object]], frame.to_dict(orient="records"))
    return cast(list[dict[str, object]], _json_value(raw))


def _lineage_payload(result: ReplaySuiteResult, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "suite_id": result.config.suite_id,
        "climadc_version": __version__,
        "started_at": result.started_at.isoformat(),
        "config_sha256": result.config_sha256,
        "scenario_config_sha256": {
            scenario.scenario_id: scenario.study.config_sha256 for scenario in result.scenarios
        },
    }


def _scenario_index_payload(
    result: ReplaySuiteResult,
    scenario_paths: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "suite_id": result.config.suite_id,
        "mode": result.mode,
        "currency": result.currency,
        "policies": list(result.policies),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "study_id": scenario.study.config.study_id,
                "decision_time": scenario.study.config.decision_time.isoformat(),
                "site_id": scenario.study.config.replay.site_id,
                "feasible": bool(scenario.study.replay.status["feasible"].all()),
                "config_sha256": scenario.study.config_sha256,
                "input_hashes": scenario.study.input_hashes,
                "relative_run_path": scenario_paths[scenario.scenario_id],
            }
            for scenario in result.scenarios
        ],
    }


def _suite_metrics_payload(result: ReplaySuiteResult) -> dict[str, object]:
    return {
        "schema_version": "1",
        "suite_id": result.config.suite_id,
        "suite_type": result.config.suite_type,
        "aggregation": "equal_weight",
        "baseline": "ASAP within each scenario",
        "improvement_rule": "signed delta < -1e-9 in the metric's published unit",
        "units": {
            "energy_cost_change_vs_asap": result.currency,
            "estimated_location_based_emissions_change_vs_asap_kgco2e": "kgCO2e",
            "peak_change_vs_asap_kw": "kW",
            "fractions": "ratio from 0 to 1",
        },
        "records": _frame_records(result.suite_metrics),
    }


def _pareto_payload(result: ReplaySuiteResult) -> dict[str, object]:
    return {
        "schema_version": "1",
        "suite_id": result.config.suite_id,
        "eligibility": "policy feasible in every declared scenario",
        "aggregation": "equal-weight arithmetic mean over scenarios",
        "direction": "minimize",
        "dominance_tolerance": "1e-9 times max(1, absolute compared values)",
        "objectives": [
            {
                "metric": "mean_energy_cost_change_vs_asap",
                "unit": result.currency,
            },
            {
                "metric": ("mean_estimated_location_based_emissions_change_vs_asap_kgco2e"),
                "unit": "kgCO2e",
            },
            {"metric": "mean_peak_change_vs_asap_kw", "unit": "kW"},
        ],
        "policies": list(result.pareto_frontier),
    }


class ReplaySuiteArtifactWriter:
    """Atomically publish suite summaries and every verified scenario sub-run."""

    def _write(
        self,
        directory: Path,
        result: ReplaySuiteResult,
        run_id: str,
    ) -> dict[str, str]:
        scenario_paths: dict[str, str] = {}
        scenario_root = directory / "scenarios"
        scenario_root.mkdir()
        writer = ReplayArtifactWriter()
        for scenario in result.scenarios:
            run_path = writer.write(
                scenario.study,
                scenario_root / scenario.scenario_id,
            )
            latest = run_path.parent / "latest"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            scenario_paths[scenario.scenario_id] = run_path.relative_to(directory).as_posix()

        (directory / "suite.yaml").write_text(
            yaml.safe_dump(
                suite_assumptions_payload(result),
                sort_keys=True,
                allow_unicode=False,
            ),
            encoding="utf-8",
            newline="\n",
        )
        (directory / "lineage.json").write_text(
            _json_text(_lineage_payload(result, run_id)),
            encoding="utf-8",
            newline="\n",
        )
        (directory / "scenario-index.json").write_text(
            _json_text(_scenario_index_payload(result, scenario_paths)),
            encoding="utf-8",
            newline="\n",
        )
        result.scenario_metrics.to_parquet(
            directory / "scenario-metrics.parquet",
            index=False,
        )
        (directory / "suite-metrics.json").write_text(
            _json_text(_suite_metrics_payload(result)),
            encoding="utf-8",
            newline="\n",
        )
        (directory / "pareto-frontier.json").write_text(
            _json_text(_pareto_payload(result)),
            encoding="utf-8",
            newline="\n",
        )
        (directory / "report.html").write_text(
            render_replay_suite_report(result, run_id, scenario_paths),
            encoding="utf-8",
            newline="\n",
        )
        return scenario_paths

    @staticmethod
    def _validate(
        directory: Path,
        result: ReplaySuiteResult,
        run_id: str,
        scenario_paths: dict[str, str],
    ) -> None:
        names = {path.name for path in directory.iterdir()}
        if names != REPLAY_SUITE_ARTIFACTS:
            raise ConfigurationError(
                "Invalid replay suite artifact set before publish: "
                f"missing={sorted(REPLAY_SUITE_ARTIFACTS - names)}, "
                f"extra={sorted(names - REPLAY_SUITE_ARTIFACTS)}"
            )
        empty = sorted(
            path.name for path in directory.iterdir() if path.is_file() and path.stat().st_size == 0
        )
        if empty:
            raise ConfigurationError(f"Empty replay suite artifacts before publish: {empty}")
        try:
            suite = yaml.safe_load((directory / "suite.yaml").read_text(encoding="utf-8"))
            if suite != suite_assumptions_payload(result):
                raise ValueError("suite.yaml differs from frozen replay suite result")
            expected_json = (
                ("lineage.json", _lineage_payload(result, run_id)),
                (
                    "scenario-index.json",
                    _scenario_index_payload(result, scenario_paths),
                ),
                ("suite-metrics.json", _suite_metrics_payload(result)),
                ("pareto-frontier.json", _pareto_payload(result)),
            )
            for name, expected in expected_json:
                actual = json.loads((directory / name).read_text(encoding="utf-8"))
                if actual != json.loads(_json_text(expected)):
                    raise ValueError(f"{name} differs from frozen replay suite result")
            actual_metrics = pd.read_parquet(directory / "scenario-metrics.parquet")
            pd.testing.assert_frame_equal(
                actual_metrics.reset_index(drop=True),
                result.scenario_metrics.reset_index(drop=True),
                check_dtype=True,
            )
            numeric = actual_metrics.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            if numeric.size and np.isinf(numeric).any():
                raise ValueError("scenario-metrics.parquet contains an infinite numeric value")

            scenario_root = directory / "scenarios"
            expected_ids = {scenario.scenario_id for scenario in result.scenarios}
            if {path.name for path in scenario_root.iterdir()} != expected_ids:
                raise ValueError("scenario artifact directories differ from suite configuration")
            for scenario in result.scenarios:
                parent = scenario_root / scenario.scenario_id
                run_path = directory / scenario_paths[scenario.scenario_id]
                if not run_path.is_dir():
                    raise ValueError(f"scenario run is missing: {scenario.scenario_id}")
                if {path.name for path in run_path.iterdir()} != REPLAY_ARTIFACTS:
                    raise ValueError(
                        f"scenario run has an invalid artifact set: {scenario.scenario_id}"
                    )
                if any(path.name == "latest" for path in parent.iterdir()):
                    raise ValueError(
                        f"scenario contains a non-portable latest pointer: {scenario.scenario_id}"
                    )
            report = (directory / "report.html").read_text(encoding="utf-8")
            if report != render_replay_suite_report(result, run_id, scenario_paths):
                raise ValueError("report.html differs from frozen replay suite result")
            lowered = report.lower()
            if "<script" in lowered or "<link" in lowered or " src=" in lowered:
                raise ValueError("report.html contains an external or executable asset")
            if not all(token in lowered for token in ("<html", "<body", "<table", "limitations")):
                raise ValueError("report.html is missing required document sections")
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Invalid replay suite artifact content: {exc}") from exc

    def write(self, result: ReplaySuiteResult, output_dir: Path) -> Path:
        timestamp = result.started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}-{result.config_sha256[:8]}"
        runs_dir = _absolute(Path(output_dir))
        final = runs_dir / run_id
        temporary: Path | None = None
        published = False
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise ConfigurationError(f"Replay suite run directory already exists: {final}")
            temporary = _absolute(
                Path(tempfile.mkdtemp(prefix=".climadc-replay-suite-", dir=runs_dir))
            )
            scenario_paths = self._write(temporary, result, run_id)
            combined_hashes: dict[str, str] = {}
            for scenario_id, relative in scenario_paths.items():
                subrun_manifest = RunManifest.model_validate(
                    json.loads(
                        (temporary / relative / "run-manifest.json").read_text(encoding="utf-8")
                    )
                )
                combined_hashes.update(
                    {
                        f"{scenario_id}/{name}": digest
                        for name, digest in subrun_manifest.input_hashes.items()
                    }
                )
            finalize_evidence(
                temporary,
                run_type="replay_suite",
                run_id=run_id,
                study_id=result.config.suite_id,
                started_at=result.started_at,
                config_sha256=result.config_sha256,
                input_hashes=combined_hashes,
                solver=SolverRecord(
                    name="SciPy HiGHS",
                    method="scenario replay aggregation; no aggregate solve",
                    options={"scenario_aggregation": "equal_weight"},
                ),
            )
            self._validate(temporary, result, run_id, scenario_paths)
            from climadc.evidence.verify import verify_suite

            verification = verify_suite(temporary)
            if not verification.valid:
                failures = [
                    check.message for check in verification.checks if check.status == "fail"
                ]
                raise ConfigurationError(
                    f"Independent replay suite verification failed: {'; '.join(failures)}"
                )
            temporary.rename(final)
            published = True
            update_latest_pointer(runs_dir, final)
            return final
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
            raise ConfigurationError(f"Unable to write replay suite artifacts: {exc}") from exc


__all__ = ["REPLAY_SUITE_ARTIFACTS", "ReplaySuiteArtifactWriter"]
