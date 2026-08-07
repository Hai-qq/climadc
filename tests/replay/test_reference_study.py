from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from climadc.adapters.neso import NESOCarbonIntensityAdapter
from climadc.adapters.openmeteo_history import OpenMeteoHistoryAdapter
from climadc.cli.app import app
from climadc.errors import ConfigurationError
from climadc.reference import packaged_study_path, refresh_carbon_shift
from climadc.replay import (
    ReplayArtifactWriter,
    ReplayStudyConfig,
    ReplayStudyRunner,
    SourceManifest,
)
from climadc.replay.artifacts import REPLAY_ARTIFACTS

DECISION = pd.Timestamp("2026-08-01T00:00:00Z")
RETRIEVED = pd.Timestamp("2026-08-06T12:46:02Z")


def test_hash_bound_reference_inputs_force_lf_checkouts() -> None:
    attributes = (Path(__file__).parents[2] / ".gitattributes").read_text(encoding="utf-8")
    rules = {line.strip() for line in attributes.splitlines() if not line.startswith("#")}

    assert "src/climadc/reference/fixtures/gb_london_24h/*.csv text eol=lf" in rules
    assert "src/climadc/reference/fixtures/gb_london_24h/*.yaml text eol=lf" in rules


def _result():
    config = ReplayStudyConfig.from_yaml(packaged_study_path())
    return ReplayStudyRunner(clock=lambda: RETRIEVED).run(config)


def _weather_payload(variable: str, offset: float) -> dict[str, object]:
    return {
        "hourly": {
            "time": [
                timestamp.strftime("%Y-%m-%dT%H:%M")
                for timestamp in pd.date_range(DECISION, periods=24, freq="1h")
            ],
            variable: [18.0 + offset + position / 10.0 for position in range(24)],
        },
        "hourly_units": {variable: "degC"},
    }


def _neso_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "from": start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": (start + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {
                    "forecast": 100.0 + position,
                    "actual": 110.0 + position,
                    "index": "moderate",
                },
            }
            for position, start in enumerate(pd.date_range(DECISION, periods=48, freq="30min"))
        ]
    }


def test_packaged_reference_manifest_binds_every_input_and_discloses_assumptions() -> None:
    config = ReplayStudyConfig.from_yaml(packaged_study_path())
    manifest = SourceManifest.from_yaml(config.source_manifest)
    required = {
        item.path.resolve()
        for item in (
            config.inputs.climate_forecast,
            config.inputs.actual_weather,
            config.inputs.grid_signals,
            config.inputs.workload,
        )
    }

    hashes = manifest.validate_files(config.source_manifest.parent, required)

    assert set(hashes) == {
        "climate-forecast.csv",
        "actual-weather.csv",
        "grid-signals.csv",
        "workload.csv",
    }
    assert {record.license for record in manifest.records} == {"CC BY 4.0", "Apache-2.0"}
    neso_forecast = next(
        record for record in manifest.records if record.source_id == "neso-carbon-forecast"
    )
    assert neso_forecast.timing.issue_time_basis == "scenario_assumption"
    assert "must not be treated as measured provenance" in neso_forecast.timing.note
    assert any("synthetic scenario" in item for item in config.limitations)


def test_reference_integrity_failure_stops_before_replay(tmp_path: Path) -> None:
    fixture = packaged_study_path().parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture, copied)
    with (copied / "climate-forecast.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(ConfigurationError, match="integrity check failed"):
        ReplayStudyRunner().run(ReplayStudyConfig.from_yaml(copied / "study.yaml"))


def test_packaged_reference_runs_all_policies_without_leakage_or_constraint_loss() -> None:
    result = _result()

    assert result.replay.status["feasible"].all()
    assert result.replay.accepted_jobs == 4
    assert result.replay.future_jobs == 0
    assert len(result.replay.metrics) == 6
    assert result.forecast_metrics["temperature_mae_c"] == pytest.approx(0.6833333333)
    assert result.forecast_metrics["carbon_intensity_mae_gco2e_per_kwh"] == pytest.approx(
        15.8333333333
    )
    assert result.forecast_metrics["upper_quantile_diagnostics"] is None
    assert result.forecast_metrics["upper_quantile_diagnostics_status"] == "not_configured"
    metrics = result.replay.metrics.set_index("policy")
    assert metrics.loc["oracle", "objective_regret"] == pytest.approx(0.0)
    assert (metrics["objective_regret"] >= -1e-8).all()
    assert (metrics["deadline_violations"] == 0.0).all()
    assert (metrics["energy_balance_error_kwh"] <= 1e-7).all()
    assert metrics.loc["price", "energy_cost_change_vs_asap"] < 0.0
    assert metrics.loc["price", "emissions_change_vs_asap_kgco2e"] > 0.0
    assert metrics.loc["carbon", "emissions_change_vs_asap_kgco2e"] < 0.0
    assert metrics.loc["carbon", "energy_cost_change_vs_asap"] > 0.0
    assert metrics.loc["peak", "peak_change_vs_asap_kw"] < 0.0


def test_replay_artifacts_are_complete_reconstructible_and_self_contained(tmp_path: Path) -> None:
    result = _result()
    run_path = ReplayArtifactWriter().write(result, tmp_path / "runs")

    assert {path.name for path in run_path.iterdir()} == REPLAY_ARTIFACTS
    metrics = json.loads((run_path / "replay-metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_path / "solver-status.json").read_text(encoding="utf-8"))
    manifest = SourceManifest.from_yaml(run_path / "source-manifest.yaml")
    schedules = pd.read_parquet(run_path / "schedules.parquet")
    profiles = pd.read_parquet(run_path / "profiles.parquet")
    report = (run_path / "report.html").read_text(encoding="utf-8")
    assert metrics["currency"] == "GBP"
    assert len(metrics["policies"]) == 6
    assert status["mode"] == "single_window"
    assert status["decisions"] == []
    assert status["remaining_energy"] == []
    assert {record.artifact.suffix for record in manifest.records} == {".parquet"}
    manifest.validate_files(
        run_path,
        {
            run_path / "climate-forecast.parquet",
            run_path / "actual-weather.parquet",
            run_path / "grid-signals.parquet",
            run_path / "workload.parquet",
        },
    )
    assert set(schedules["policy"]) == set(result.replay.metrics["policy"])
    assert set(profiles["policy"]) == set(result.replay.metrics["policy"])
    assert np.isfinite(profiles.select_dtypes(include=[np.number]).to_numpy()).all()
    assert "positive values are increases" in report
    assert "temperature-sensitive PUE" in report
    assert "No replay constraint violations were reported" in report
    assert "<script" not in report.lower()
    assert "<link" not in report.lower()
    assert (tmp_path / "runs" / "latest").resolve() == run_path


def test_cli_runs_packaged_and_generic_reference_replays(tmp_path: Path) -> None:
    runner = CliRunner()
    demo = runner.invoke(app, ["demo", "carbon-shift", "--output-dir", str(tmp_path / "demo-runs")])
    assert demo.exit_code == 0, demo.output
    demo_path = Path(demo.stdout.strip())
    assert demo_path.is_dir()

    generic = runner.invoke(
        app,
        [
            "replay",
            str(packaged_study_path()),
            "--output-dir",
            str(tmp_path / "generic-runs"),
        ],
    )
    assert generic.exit_code == 0, generic.output
    generic_path = Path(generic.stdout.strip())
    assert (
        json.loads((generic_path / "replay-metrics.json").read_text(encoding="utf-8"))["policies"]
        == json.loads((demo_path / "replay-metrics.json").read_text(encoding="utf-8"))["policies"]
    )
    report = runner.invoke(app, ["report", str(tmp_path / "generic-runs" / "latest")])
    assert report.exit_code == 0, report.output
    assert Path(report.stdout.strip()) == generic_path / "report.html"


def test_refresh_creates_new_verified_snapshot_and_refuses_overwrite(tmp_path: Path) -> None:
    def weather_transport(url: str) -> Mapping[str, object]:
        if "previous-runs-api" in url:
            return _weather_payload("temperature_2m_previous_day1", 0.0)
        return _weather_payload("temperature_2m", 0.5)

    weather = OpenMeteoHistoryAdapter(transport=weather_transport, clock=lambda: RETRIEVED)
    carbon = NESOCarbonIntensityAdapter(
        transport=lambda _: _neso_payload(), clock=lambda: RETRIEVED
    )
    destination = tmp_path / "refreshed"
    config_path = refresh_carbon_shift(
        destination,
        decision_time=DECISION,
        retrieved_at=RETRIEVED,
        weather_adapter=weather,
        carbon_adapter=carbon,
    )

    assert config_path == destination / "study.yaml"
    config = ReplayStudyConfig.from_yaml(config_path)
    result = ReplayStudyRunner(clock=lambda: RETRIEVED).run(config)
    assert result.replay.status["feasible"].all()
    assert len(SourceManifest.from_yaml(config.source_manifest).records) == 6
    with pytest.raises(ConfigurationError, match="already exists"):
        refresh_carbon_shift(
            destination,
            decision_time=DECISION,
            retrieved_at=RETRIEVED,
            weather_adapter=weather,
            carbon_adapter=carbon,
        )
