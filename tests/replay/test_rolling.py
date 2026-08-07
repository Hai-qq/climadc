from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner
import yaml

from climadc.cli.app import app
from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ConfigurationError
from climadc.replay import (
    ReplayArtifactWriter,
    ReplayConfig,
    ReplayStudyConfig,
    ReplayStudyRunner,
    TemperatureSensitivePUEModel,
)
from climadc.replay.manifest import sha256_file
from climadc.replay.rolling import RollingReplayEngine, RollingReplayResult

START = pd.Timestamp("2026-01-01T00:00:00Z")
SLOTS = pd.date_range(START, periods=8, freq="1h")


def _climate(*, risk: bool = False) -> ClimateForecastFrame:
    rows: list[dict[str, object]] = []
    for position, slot in enumerate(SLOTS):
        rows.append(
            {
                "site_id": "dc-1",
                "issue_time": START - pd.Timedelta(hours=2),
                "available_at": START - pd.Timedelta(hours=1),
                "valid_time": slot,
                "variable": "air_temperature",
                "value": 20.0,
                "unit": "degC",
                "source": "initial",
                "quantile": pd.NA,
                "member": pd.NA,
            }
        )
        if risk:
            rows.append(
                {
                    "site_id": "dc-1",
                    "issue_time": START - pd.Timedelta(hours=2),
                    "available_at": START - pd.Timedelta(hours=1),
                    "valid_time": slot,
                    "variable": "air_temperature",
                    "value": 26.0 if position % 2 == 0 else 24.0,
                    "unit": "degC",
                    "source": "initial-quantile",
                    "quantile": 0.9,
                    "member": pd.NA,
                }
            )
        if slot >= SLOTS[1]:
            rows.append(
                {
                    "site_id": "dc-1",
                    "issue_time": SLOTS[1],
                    "available_at": SLOTS[1],
                    "valid_time": slot,
                    "variable": "air_temperature",
                    "value": 30.0,
                    "unit": "degC",
                    "source": "refresh",
                    "quantile": pd.NA,
                    "member": pd.NA,
                }
            )
    return ClimateForecastFrame.from_pandas(pd.DataFrame(rows))


def _weather() -> DCTelemetryFrame:
    return DCTelemetryFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * len(SLOTS),
                "device_id": ["weather"] * len(SLOTS),
                "event_time": SLOTS,
                "available_at": SLOTS + pd.Timedelta(minutes=5),
                "metric": ["air_temperature"] * len(SLOTS),
                "value": [25.0] * len(SLOTS),
                "unit": ["degC"] * len(SLOTS),
                "quality": ["observed"] * len(SLOTS),
            }
        )
    )


def _grid(*, risk: bool = False) -> GridSignalFrame:
    rows: list[dict[str, object]] = []
    for signal, unit, initial, refreshed, actual, risk_high, risk_low in (
        ("energy_price", "GBP / kWh", 0.4, 0.2, 0.3, 0.35, 0.25),
        ("carbon_intensity", "gCO2e / kWh", 300.0, 200.0, 250.0, 300.0, 200.0),
    ):
        for position, slot in enumerate(SLOTS):
            rows.extend(
                [
                    {
                        "site_id": "dc-1",
                        "region_id": "grid-1",
                        "issue_time": START - pd.Timedelta(hours=2),
                        "available_at": START - pd.Timedelta(hours=1),
                        "valid_time": slot,
                        "signal": signal,
                        "value": initial,
                        "unit": unit,
                        "source": "initial",
                        "quality": "forecast",
                        "quantile": pd.NA,
                    },
                    {
                        "site_id": "dc-1",
                        "region_id": "grid-1",
                        "issue_time": pd.NaT,
                        "available_at": slot + pd.Timedelta(minutes=30),
                        "valid_time": slot,
                        "signal": signal,
                        "value": actual,
                        "unit": unit,
                        "source": "actual",
                        "quality": "observed",
                        "quantile": pd.NA,
                    },
                ]
            )
            if risk:
                rows.append(
                    {
                        "site_id": "dc-1",
                        "region_id": "grid-1",
                        "issue_time": START - pd.Timedelta(hours=2),
                        "available_at": START - pd.Timedelta(hours=1),
                        "valid_time": slot,
                        "signal": signal,
                        "value": risk_high if position % 2 == 0 else risk_low,
                        "unit": unit,
                        "source": "initial-quantile",
                        "quality": "forecast",
                        "quantile": 0.9,
                    }
                )
            if slot >= SLOTS[1]:
                rows.append(
                    {
                        "site_id": "dc-1",
                        "region_id": "grid-1",
                        "issue_time": SLOTS[1],
                        "available_at": SLOTS[1],
                        "valid_time": slot,
                        "signal": signal,
                        "value": refreshed,
                        "unit": unit,
                        "source": "refresh",
                        "quality": "forecast",
                        "quantile": pd.NA,
                    }
                )
    return GridSignalFrame.from_pandas(pd.DataFrame(rows))


def _workload() -> FlexibleWorkloadFrame:
    return FlexibleWorkloadFrame.from_pandas(
        pd.DataFrame(
            {
                "job_id": ["known", "arrives-later"],
                "site_id": ["dc-1", "dc-1"],
                "release_time": [START, SLOTS[1]],
                "available_at": [START, SLOTS[1]],
                "deadline": [SLOTS[4], SLOTS[5]],
                "energy": [2.0, 1.0],
                "energy_unit": ["kWh", "kWh"],
                "max_power": [1.0, 1.0],
                "power_unit": ["kW", "kW"],
                "preemptible": [True, True],
                "priority": [1.0, 1.0],
            }
        )
    )


def _config() -> ReplayConfig:
    return ReplayConfig(
        site_id="dc-1",
        horizon=pd.Timedelta(hours=4),
        interval=pd.Timedelta(hours=1),
        it_capacity_kw=2.0,
        fixed_it_power_kw=0.0,
        cost_weight=1.0,
        carbon_weight=0.0,
    )


def _run():
    return RollingReplayEngine(TemperatureSensitivePUEModel()).run(
        start_time=START,
        periods=5,
        step=pd.Timedelta(hours=1),
        climate_forecast=_climate(),
        actual_weather=_weather(),
        grid_signals=_grid(),
        workload=_workload(),
        config=_config(),
    )


def _write_study(tmp_path: Path, *, risk: bool = False) -> Path:
    inputs = {
        "climate-forecast.csv": _climate(risk=risk).to_pandas(),
        "actual-weather.csv": _weather().to_pandas(),
        "grid-signals.csv": _grid(risk=risk).to_pandas(),
        "workload.csv": _workload().to_pandas(),
    }
    for name, frame in inputs.items():
        frame.to_csv(tmp_path / name, index=False)
    records = []
    for name in inputs:
        path = tmp_path / name
        records.append(
            {
                "source_id": name.removesuffix(".csv"),
                "artifact": name,
                "selection": "Phase 4 deterministic rolling fixture",
                "role": "rolling replay input",
                "provider": "ClimaDC test fixture",
                "request_url": f"https://example.test/{name}",
                "retrieved_at": "2026-01-02T00:00:00Z",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "license": "CC0-1.0",
                "attribution": "ClimaDC project-generated test fixture",
                "provenance": "project_generated",
                "transformations": ["Generated in the test from deterministic values."],
                "timing": {
                    "issue_time_basis": "scenario_assumption",
                    "availability_basis": "scenario_assumption",
                    "note": "Timestamps are deterministic test assumptions.",
                },
                "limitations": ["Not operational data."],
            }
        )
    (tmp_path / "source-manifest.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": "1", "study_id": "rolling-test", "records": records},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": "1",
        "study_id": "rolling-test",
        "decision_time": START.isoformat(),
        "inputs": {
            "climate_forecast": {"path": "climate-forecast.csv"},
            "actual_weather": {"path": "actual-weather.csv"},
            "grid_signals": {"path": "grid-signals.csv"},
            "workload": {"path": "workload.csv"},
        },
        "source_manifest": "source-manifest.yaml",
        "replay": {
            "site_id": "dc-1",
            "horizon": "4h",
            "interval": "1h",
            "it_capacity_kw": 2.0,
            "fixed_it_power_kw": 0.0,
            "cost_weight": 1.0,
            "carbon_weight": 0.0,
        },
        "rolling": {"periods": 5, "step": "1h"},
        "limitations": ["Deterministic test fixture; not a production result."],
        "output_dir": "runs",
    }
    if risk:
        config["replay"]["risk_quantile"] = 0.9
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def test_rolling_replay_commits_each_step_and_carries_remaining_energy() -> None:
    result = _run()

    assert result.decision_count == 5
    assert result.status["feasible"].all()
    assert len(result.decisions) == 5 * 6
    assert set(result.allocations["valid_time"]) == set(SLOTS[:5])
    assert len(result.allocations) == 5 * 2 * 6
    assert (result.remaining_energy["remaining_energy_kwh"] <= 1e-8).all()
    assert not result.remaining_energy["overdue"].any()
    assert (result.metrics["completed_jobs"] == 2.0).all()
    assert (result.metrics["deadline_violations"] == 0.0).all()
    assert (result.metrics["energy_balance_error_kwh"] <= 1e-8).all()
    executed = result.allocations.groupby("policy", observed=True)["energy_kwh"].sum()
    assert executed.tolist() == pytest.approx([3.0] * len(executed))


def test_rolling_replay_uses_only_forecasts_available_at_each_origin() -> None:
    result = _run()
    asap = result.profiles.loc[result.profiles["policy"] == "asap"].set_index("valid_time")

    assert asap.loc[SLOTS[0], "forecast_temperature_c"] == 20.0
    assert asap.loc[SLOTS[1], "forecast_temperature_c"] == 30.0
    assert asap.loc[SLOTS[0], "decision_time"] == SLOTS[0]
    assert asap.loc[SLOTS[1], "decision_time"] == SLOTS[1]


def test_rolling_oracle_delta_can_be_negative_after_a_future_job_arrives() -> None:
    grid = _grid().to_pandas()
    actual_price = (grid["signal"] == "energy_price") & (grid["quality"] == "observed")
    price_by_slot = dict(zip(SLOTS, [1.0, 0.0, *([10.0] * 6)], strict=True))
    grid.loc[actual_price, "value"] = grid.loc[actual_price, "valid_time"].map(price_by_slot)
    jobs = FlexibleWorkloadFrame.from_pandas(
        pd.DataFrame(
            {
                "job_id": ["known", "future"],
                "site_id": ["dc-1", "dc-1"],
                "release_time": [SLOTS[0], SLOTS[1]],
                "available_at": [SLOTS[0], SLOTS[1]],
                "deadline": [SLOTS[2], SLOTS[3]],
                "energy": [1.0, 1.0],
                "energy_unit": ["kWh", "kWh"],
                "max_power": [1.0, 1.0],
                "power_unit": ["kW", "kW"],
                "preemptible": [True, True],
                "priority": [1.0, 1.0],
            }
        )
    )
    config = ReplayConfig(
        site_id="dc-1",
        horizon=pd.Timedelta(hours=2),
        interval=pd.Timedelta(hours=1),
        it_capacity_kw=1.0,
        fixed_it_power_kw=0.0,
        cost_weight=1.0,
        carbon_weight=0.0,
    )

    result = RollingReplayEngine(TemperatureSensitivePUEModel()).run(
        start_time=START,
        periods=3,
        step=pd.Timedelta(hours=1),
        climate_forecast=_climate(),
        actual_weather=_weather(),
        grid_signals=GridSignalFrame.from_pandas(grid),
        workload=jobs,
        config=config,
    )

    metrics = result.metrics.set_index("policy")
    assert metrics.loc["asap", "objective_regret"] < 0.0
    assert metrics.loc["oracle", "objective_regret"] == pytest.approx(0.0)


def test_rolling_result_is_frozen_and_defensively_copies_state() -> None:
    result = _run()
    mutated = result.remaining_energy
    mutated.loc[:, "remaining_energy_kwh"] = 999.0

    assert (result.remaining_energy["remaining_energy_kwh"] <= 1e-8).all()
    with pytest.raises(FrozenInstanceError):
        result.currency = "USD"  # type: ignore[misc]


def test_rolling_replay_handles_an_empty_workload() -> None:
    empty_workload = FlexibleWorkloadFrame.from_pandas(_workload().to_pandas().iloc[0:0])

    result = RollingReplayEngine(TemperatureSensitivePUEModel()).run(
        start_time=START,
        periods=2,
        step=pd.Timedelta(hours=1),
        climate_forecast=_climate(),
        actual_weather=_weather(),
        grid_signals=_grid(),
        workload=empty_workload,
        config=_config(),
    )

    assert result.status["feasible"].all()
    assert result.accepted_jobs == 0
    assert result.future_jobs == 0
    assert result.allocations.empty
    assert result.remaining_energy.empty
    assert list(result.allocations) == [
        "site_id",
        "policy",
        "job_id",
        "valid_time",
        "power_kw",
        "energy_kwh",
        "decision_time",
    ]
    assert (result.metrics["completed_jobs"] == 0.0).all()


@pytest.mark.parametrize(
    ("periods", "step", "message"),
    [
        (0, pd.Timedelta(hours=1), "periods"),
        (1, pd.Timedelta(minutes=30), "integer multiple"),
        (1, pd.Timedelta(hours=5), "exceed"),
    ],
)
def test_rolling_replay_rejects_invalid_window_controls(
    periods: int, step: pd.Timedelta, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        RollingReplayEngine(TemperatureSensitivePUEModel()).run(
            start_time=START,
            periods=periods,
            step=step,
            climate_forecast=_climate(),
            actual_weather=_weather(),
            grid_signals=_grid(),
            workload=_workload(),
            config=_config(),
        )


def test_rolling_replay_reports_jobs_outside_the_planning_horizon() -> None:
    workload = _workload().to_pandas()
    workload.loc[workload["job_id"] == "known", "deadline"] = SLOTS[5]

    with pytest.raises(ConfigurationError, match="deadline extends beyond replay horizon"):
        RollingReplayEngine(TemperatureSensitivePUEModel()).run(
            start_time=START,
            periods=1,
            step=pd.Timedelta(hours=1),
            climate_forecast=_climate(),
            actual_weather=_weather(),
            grid_signals=_grid(),
            workload=FlexibleWorkloadFrame.from_pandas(workload),
            config=_config(),
        )


@pytest.mark.parametrize(
    ("step", "message"),
    [("30min", "integer multiple"), ("5h", "exceed")],
)
def test_rolling_study_config_rejects_invalid_commit_steps(
    tmp_path: Path, step: str, message: str
) -> None:
    config_path = _write_study(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["rolling"]["step"] = step
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        ReplayStudyConfig.from_yaml(config_path)


def test_study_runner_rejects_risk_policy_without_complete_quantiles(tmp_path: Path) -> None:
    config_path = _write_study(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["replay"]["risk_quantile"] = 0.9
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    config = ReplayStudyConfig.from_yaml(config_path)

    with pytest.raises(ConfigurationError, match="requires quantile 0.9"):
        ReplayStudyRunner().run(config)


def test_rolling_study_config_publishes_reconstructible_existing_artifact_contract(
    tmp_path: Path,
) -> None:
    config = ReplayStudyConfig.from_yaml(_write_study(tmp_path))
    result = ReplayStudyRunner(clock=lambda: pd.Timestamp("2026-01-02T00:00:00Z")).run(config)

    assert isinstance(result.replay, RollingReplayResult)
    run_path = ReplayArtifactWriter().write(result, config.output_dir)
    metrics = json.loads((run_path / "replay-metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_path / "solver-status.json").read_text(encoding="utf-8"))
    schedules = pd.read_parquet(run_path / "schedules.parquet")
    report = (run_path / "report.html").read_text(encoding="utf-8")

    assert metrics["mode"] == "rolling"
    assert metrics["decision_count"] == 5
    assert status["mode"] == "rolling"
    assert len(status["decisions"]) == 30
    assert len(status["remaining_energy"]) == 12
    assert all(item["completed"] for item in status["remaining_energy"])
    assert all(item["remaining_energy_kwh"] <= 1e-8 for item in status["remaining_energy"])
    assert "decision_time" in schedules
    assert "5 rolling decisions" in report
    assert "signed cumulative difference can be negative" in report

    cli_output = tmp_path / "cli-runs"
    cli_result = CliRunner().invoke(
        app,
        ["replay", str(tmp_path / "study.yaml"), "--output-dir", str(cli_output)],
    )

    assert cli_result.exit_code == 0, cli_result.stdout
    cli_run_path = Path(cli_result.stdout.strip())
    cli_metrics = json.loads((cli_run_path / "replay-metrics.json").read_text(encoding="utf-8"))
    assert cli_metrics["mode"] == "rolling"
    assert cli_metrics["decision_count"] == 5


def test_rolling_risk_diagnostics_use_only_committed_slots_and_reach_artifacts(
    tmp_path: Path,
) -> None:
    config_path = _write_study(tmp_path, risk=True)
    config = ReplayStudyConfig.from_yaml(config_path)
    result = ReplayStudyRunner(clock=lambda: pd.Timestamp("2026-01-02T00:00:00Z")).run(config)

    assert result.forecast_metrics["upper_quantile_diagnostics_status"] == "computed"
    diagnostics = result.forecast_metrics["upper_quantile_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["sample_count"] == 5
    assert diagnostics["nominal_quantile"] == pytest.approx(0.9)
    signals = diagnostics["signals"]
    assert isinstance(signals, dict)
    for signal in signals.values():
        assert signal["sample_count"] == 5
        assert signal["covered_count"] == 3
        assert signal["exceedance_count"] == 2
        assert signal["empirical_coverage"] == pytest.approx(0.6)
        assert signal["coverage_gap"] == pytest.approx(-0.3)
        assert signal["wilson_95_lower"] < 0.6 < signal["wilson_95_upper"]

    assert signals["temperature"]["mean_positive_exceedance"] == pytest.approx(0.4)
    assert signals["temperature"]["mean_exceedance_when_exceeded"] == pytest.approx(1.0)
    assert signals["temperature"]["pinball_loss"] == pytest.approx(0.42)
    assert signals["energy_price"]["pinball_loss"] == pytest.approx(0.021)
    assert signals["carbon_intensity"]["pinball_loss"] == pytest.approx(0.021)

    run_path = ReplayArtifactWriter().write(result, config.output_dir)
    metrics = json.loads((run_path / "replay-metrics.json").read_text(encoding="utf-8"))
    report = (run_path / "report.html").read_text(encoding="utf-8")

    assert metrics["forecast"]["upper_quantile_diagnostics"]["sample_count"] == 5
    assert "Upper-quantile diagnostics" in report
    assert "Post-hoc marginal backtest over 5 committed slots at q=0.9" in report
    assert "60.0%" in report

    cli_result = CliRunner().invoke(
        app,
        ["replay", str(config_path), "--output-dir", str(tmp_path / "risk-cli-runs")],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_metrics = json.loads(
        (Path(cli_result.stdout.strip()) / "replay-metrics.json").read_text(encoding="utf-8")
    )
    assert cli_metrics["forecast"]["upper_quantile_diagnostics"]["sample_count"] == 5
