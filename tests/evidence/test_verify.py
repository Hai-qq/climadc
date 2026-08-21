from __future__ import annotations

import json
import hashlib
import re
import shutil
from html import escape, unescape
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from climadc.cli.app import app
from climadc.evidence.checksums import write_checksums
from climadc.evidence.verify import VerificationReport, verify_run, verify_suite
from climadc.reference import packaged_study_path, packaged_suite_path
from climadc.replay import (
    ReplayArtifactWriter,
    ReplayStudyConfig,
    ReplayStudyRunner,
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteRunner,
)
from climadc.replay.artifacts import REPLAY_ARTIFACTS

_REPORT_DATA = re.compile(
    r'<template\s+id="climadc-report-data">(?P<payload>.*?)</template>', re.DOTALL
)


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("verified-replay")
    config = ReplayStudyConfig.from_yaml(packaged_study_path())
    result = ReplayStudyRunner(clock=lambda: pd.Timestamp("2026-08-21T00:00:00Z")).run(config)
    return ReplayArtifactWriter().write(result, root / "runs")


@pytest.fixture(scope="module")
def suite_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("verified-suite")
    config = ReplaySuiteConfig.from_yaml(packaged_suite_path())
    result = ReplaySuiteRunner(clock=lambda: pd.Timestamp("2026-08-21T00:00:00Z")).run(config)
    return ReplaySuiteArtifactWriter().write(result, root / "runs")


def _copy(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _failure(report: VerificationReport, check_id: str) -> str:
    matches = [
        check.message
        for check in report.checks
        if check.check_id == check_id and check.status == "fail"
    ]
    assert matches, report.to_json()
    return matches[0]


def test_verify_run_accepts_a_fresh_v2_bundle(replay_run: Path) -> None:
    report = verify_run(replay_run)

    assert report.valid
    assert report.run_type == "replay"
    assert report.artifact_schema_version == "2"
    assert not report.legacy
    assert all(check.status == "pass" for check in report.checks)


@pytest.mark.parametrize("artifact", sorted(REPLAY_ARTIFACTS))
def test_every_replay_artifact_is_checksum_bound(
    artifact: str,
    replay_run: Path,
    tmp_path: Path,
) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / artifact
    if artifact == "checksums.sha256":
        text = path.read_text(encoding="utf-8")
        path.write_text("x" + text[1:], encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"\nTAMPERED")

    assert not verify_run(run).valid


def test_missing_required_and_extra_undeclared_artifacts_fail(
    replay_run: Path, tmp_path: Path
) -> None:
    missing = _copy(replay_run, tmp_path / "missing")
    (missing / "profiles.parquet").unlink()
    assert not verify_run(missing).valid

    extra = _copy(replay_run, tmp_path / "extra")
    (extra / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    assert not verify_run(extra).valid


def test_environment_inconsistency_fails_after_checksum_refresh(
    replay_run: Path, tmp_path: Path
) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "environment.json"
    payload = _json(path)
    packages = payload["packages"]
    assert isinstance(packages, dict)
    packages["climadc"] = "9.9.9"
    _write_json(path, payload)
    write_checksums(run)

    message = _failure(verify_run(run), "environment.consistency")
    assert "differs" in message


def test_config_hash_inconsistency_fails_after_checksum_refresh(
    replay_run: Path, tmp_path: Path
) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "assumptions.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["limitations"].append("tampered limitation")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    write_checksums(run)

    assert "does not match" in _failure(verify_run(run), "replay.config_hash")


def test_schedule_and_profile_key_drift_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "schedules.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "valid_time"] = frame.loc[0, "valid_time"] + pd.Timedelta(minutes=30)
    frame.to_parquet(path, index=False)
    write_checksums(run)

    assert "keys differ" in _failure(verify_run(run), "replay.reconstruction")


def test_energy_conservation_tamper_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "schedules.parquet"
    frame = pd.read_parquet(path)
    index = frame.index[frame["power_kw"] > 0][0]
    delta = min(0.01, float(frame.loc[index, "power_kw"]) / 2.0)
    frame.loc[index, "power_kw"] = float(frame.loc[index, "power_kw"]) - delta
    frame.loc[index, "energy_kwh"] = float(frame.loc[index, "energy_kwh"]) - delta
    frame.to_parquet(path, index=False)
    write_checksums(run)

    assert "energy conservation" in _failure(verify_run(run), "replay.reconstruction")


def test_deadline_window_tamper_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    schedule_path = run / "schedules.parquet"
    schedule = pd.read_parquet(schedule_path)
    workload = pd.read_parquet(run / "workload.parquet").set_index("job_id")
    candidate = None
    for index, row in schedule.loc[schedule["power_kw"] == 0].iterrows():
        job = workload.loc[str(row["job_id"])]
        if not (job["release_time"] <= row["valid_time"] < job["deadline"]):
            candidate = index
            break
    assert candidate is not None
    schedule.loc[candidate, "power_kw"] = 0.01
    schedule.loc[candidate, "energy_kwh"] = 0.01
    schedule.to_parquet(schedule_path, index=False)
    write_checksums(run)

    assert "outside release/deadline" in _failure(verify_run(run), "replay.reconstruction")


def test_max_power_tamper_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    schedule_path = run / "schedules.parquet"
    schedule = pd.read_parquet(schedule_path)
    workload = pd.read_parquet(run / "workload.parquet").set_index("job_id")
    index = schedule.index[schedule["power_kw"] > 0][0]
    job = workload.loc[str(schedule.loc[index, "job_id"])]
    schedule.loc[index, "power_kw"] = float(job["max_power"]) + 1.0
    schedule.loc[index, "energy_kwh"] = float(job["max_power"]) + 1.0
    schedule.to_parquet(schedule_path, index=False)
    write_checksums(run)

    assert "exceeds max_power" in _failure(verify_run(run), "replay.reconstruction")


def test_capacity_tamper_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    profile_path = run / "profiles.parquet"
    profiles = pd.read_parquet(profile_path)
    assumptions = yaml.safe_load((run / "assumptions.yaml").read_text(encoding="utf-8"))
    capacity = float(assumptions["replay"]["it_capacity_kw"])
    index = profiles.index[0]
    profiles.loc[index, "total_it_power_kw"] = capacity + 1.0
    profiles.loc[index, "flexible_it_power_kw"] = (
        capacity + 1.0 - float(profiles.loc[index, "fixed_it_power_kw"])
    )
    profiles.loc[index, "actual_facility_power_kw"] = float(profiles.loc[index, "actual_pue"]) * (
        capacity + 1.0
    )
    profiles.to_parquet(profile_path, index=False)
    write_checksums(run)

    message = _failure(verify_run(run), "replay.reconstruction")
    assert "flexible power differ" in message or "exceeds IT capacity" in message


def test_schedule_profile_power_mismatch_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    profile_path = run / "profiles.parquet"
    profiles = pd.read_parquet(profile_path)
    index = profiles.index[0]
    profiles.loc[index, "flexible_it_power_kw"] += 0.01
    profiles.loc[index, "total_it_power_kw"] += 0.01
    profiles.loc[index, "actual_facility_power_kw"] = float(
        profiles.loc[index, "actual_pue"]
    ) * float(profiles.loc[index, "total_it_power_kw"])
    profiles.to_parquet(profile_path, index=False)
    write_checksums(run)

    assert "flexible power differ" in _failure(verify_run(run), "replay.reconstruction")


@pytest.mark.parametrize("constant", [float("nan"), float("inf")])
def test_nonfinite_json_fails(
    constant: float,
    replay_run: Path,
    tmp_path: Path,
) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "replay-metrics.json"
    payload = _json(path)
    policies = payload["policies"]
    assert isinstance(policies, list) and isinstance(policies[0], dict)
    policies[0]["energy_cost"] = constant
    _write_json(path, payload, allow_nan=True)
    write_checksums(run)

    assert "non-finite" in _failure(verify_run(run), "replay.metrics_json")


def test_solver_status_type_drift_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "solver-status.json"
    payload = _json(path)
    policies = payload["policies"]
    assert isinstance(policies, list) and isinstance(policies[0], dict)
    policies[0]["solver_status"] = True
    _write_json(path, payload)
    write_checksums(run)

    assert "must be an integer" in _failure(verify_run(run), "replay.solver_status")


def test_metric_numeric_type_drift_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "replay-metrics.json"
    payload = _json(path)
    policies = payload["policies"]
    assert isinstance(policies, list) and isinstance(policies[0], dict)
    policies[0]["energy_cost"] = str(policies[0]["energy_cost"])
    _write_json(path, payload)
    write_checksums(run)

    assert "must be a finite number" in _failure(verify_run(run), "replay.metrics_json")


def test_objective_contract_drift_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "replay-metrics.json"
    payload = _json(path)
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["carbon_price_currency_per_tco2e"] = 999.0
    _write_json(path, payload)
    write_checksums(run)

    assert "differs" in _failure(verify_run(run), "replay.objective_contract")


def test_html_numeric_payload_drift_fails(replay_run: Path, tmp_path: Path) -> None:
    run = _copy(replay_run, tmp_path)
    path = run / "report.html"
    text = path.read_text(encoding="utf-8")
    match = _REPORT_DATA.search(text)
    assert match is not None
    payload = json.loads(unescape(match.group("payload")))
    payload["policies"][0]["energy_cost"] += 1.0
    replacement = escape(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        quote=False,
    )
    text = text[: match.start("payload")] + replacement + text[match.end("payload") :]
    path.write_text(text, encoding="utf-8")
    write_checksums(run)

    assert "differs" in _failure(verify_run(run), "replay.html_consistency")


def test_legacy_v1_replay_remains_readable_with_explicit_limitations(
    replay_run: Path, tmp_path: Path
) -> None:
    run = _copy(replay_run, tmp_path)
    for name in ("run-manifest.json", "environment.json", "checksums.sha256"):
        (run / name).unlink()

    report = verify_run(run)

    assert report.valid
    assert report.legacy
    assert report.artifact_schema_version == "1"
    assert report.limitations
    assert any(check.status == "warning" for check in report.checks)


def test_verify_run_cli_emits_json_and_nonzero_exit_for_invalid_bundle(
    replay_run: Path, tmp_path: Path
) -> None:
    valid = CliRunner().invoke(app, ["verify-run", str(replay_run), "--json"])
    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.stdout)["valid"] is True

    run = _copy(replay_run, tmp_path)
    (run / "profiles.parquet").unlink()
    invalid = CliRunner().invoke(app, ["verify-run", str(run), "--json"])
    assert invalid.exit_code == 1
    assert json.loads(invalid.stdout)["valid"] is False


def test_verify_suite_accepts_a_fresh_v2_bundle(suite_run: Path) -> None:
    report = verify_suite(suite_run)
    assert report.valid, report.to_json()
    assert report.run_type == "replay_suite"
    assert report.artifact_schema_version == "2"


def test_suite_recursively_rejects_semantically_tampered_subrun(
    suite_run: Path, tmp_path: Path
) -> None:
    suite = _copy(suite_run, tmp_path)
    index = _json(suite / "scenario-index.json")
    scenarios = index["scenarios"]
    assert isinstance(scenarios, list) and isinstance(scenarios[0], dict)
    subrun = suite / str(scenarios[0]["relative_run_path"])
    metrics_path = subrun / "replay-metrics.json"
    metrics = _json(metrics_path)
    policies = metrics["policies"]
    assert isinstance(policies, list) and isinstance(policies[0], dict)
    policies[0]["energy_cost"] += 1.0
    _write_json(metrics_path, metrics)
    write_checksums(subrun)
    write_checksums(suite)

    assert "failed verify-run" in _failure(verify_suite(suite), "suite.subruns")


def test_suite_reconstructs_aggregate_instead_of_trusting_json(
    suite_run: Path, tmp_path: Path
) -> None:
    suite = _copy(suite_run, tmp_path)
    path = suite / "suite-metrics.json"
    payload = _json(path)
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["mean_energy_cost_change_vs_asap"] += 1.0
    _write_json(path, payload)
    write_checksums(suite)

    assert "mismatch" in _failure(verify_suite(suite), "suite.aggregate")


def test_suite_cannot_relabel_same_day_sensitivity_as_robustness(
    suite_run: Path, tmp_path: Path
) -> None:
    run = _copy(suite_run, tmp_path)
    suite_path = run / "suite.yaml"
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    payload["suite_type"] = "robustness"
    payload["robustness_dimensions"] = ["decision_date"]
    suite_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    manifest_path = run / "run-manifest.json"
    manifest = _json(manifest_path)
    manifest["config_sha256"] = _canonical_hash(payload)
    _write_json(manifest_path, manifest)
    write_checksums(run)

    assert "does not vary" in _failure(verify_suite(run), "suite.aggregate")


def test_suite_html_machine_data_must_match_json(suite_run: Path, tmp_path: Path) -> None:
    run = _copy(suite_run, tmp_path)
    path = run / "report.html"
    text = path.read_text(encoding="utf-8")
    match = _REPORT_DATA.search(text)
    assert match is not None
    payload = json.loads(unescape(match.group("payload")))
    payload["records"][0]["mean_energy_cost_change_vs_asap"] += 1.0
    replacement = escape(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        quote=False,
    )
    text = text[: match.start("payload")] + replacement + text[match.end("payload") :]
    path.write_text(text, encoding="utf-8")
    write_checksums(run)

    assert "machine data differs" in _failure(verify_suite(run), "suite.aggregate")
